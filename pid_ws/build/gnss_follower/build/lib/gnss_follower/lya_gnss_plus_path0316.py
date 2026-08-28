import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped

import math

try:
    import pyproj
    _has_pyproj = True
except ImportError:
    _has_pyproj = False


def ecef_to_utm(x_ecef, y_ecef, z_ecef):
    """
    如果 /vectornav/pose 是 ECEF坐标，需要转换到UTM; 
    若已是UTM可直接返回 (x_ecef, y_ecef).
    """
    if not _has_pyproj:
        return x_ecef, y_ecef
    transformer = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:32654", always_xy=True)
    x_utm, y_utm, _ = transformer.transform(x_ecef, y_ecef, z_ecef)
    return x_utm, y_utm


def normalize_angle(rad):
    """归一化角度到 -pi ~ +pi"""
    return math.atan2(math.sin(rad), math.cos(rad))


class LYAController:
    """与第二段类似, 存储 Lyapunov 参数"""
    def __init__(self, v_t, lambda_v, lambda_a, k1, k2, omega_t=0.0):
        self.v_t = v_t
        self.lambda_v = lambda_v
        self.lambda_a = lambda_a
        self.k1 = k1
        self.k2 = k2
        self.omega_t = omega_t


class GnssLyaFollower(Node):
    def __init__(self):
        """
        在这里我们:
         - 订阅 /origin_gnss_path (在全局坐标下)
         - 订阅 /vectornav/pose (ECEF或UTM)
         - 第一次收到车辆位姿后, 设 (base_x_, base_y_, base_yaw_) 为局部原点+朝向
         - 把路径和车辆位姿都映射到局部坐标系来做 Lyapunov 控制
         - 新增了一个“findNearestIndex”函数(照抄你C++的写法)来更新 self.idx_
        """
        super().__init__('gnss_lya_follower_node')

        # =========== 可调参数 =========== 
        self.freq_ = 5.0       # 控制频率
        self.v_max_ = 5.0
        self.w_max_ = 10.0
        # Lyapunov控制参数
        self.lya = LYAController(
            v_t=2.0,
            lambda_v=0.1,
            lambda_a=1,
            k1=0.8,
            k2=25.0
        )

        # =========== 状态变量 =========== 
        # 用于设置局部原点
        self.base_set_ = False
        self.base_x_ = 0.0
        self.base_y_ = 0.0
        self.base_yaw_ = 0.0

        # 车辆当前(局部)位置 + yaw
        self.current_x_local_ = 0.0
        self.current_y_local_ = 0.0
        self.current_yaw_local_ = 0.0

        # 若先到path，再到pose, 需先存全局坐标
        self.path_points_global_ = []   
        self.path_points_local_ = []    
        self.idx_ = 0

        # 标记：是否已经把路径转换到局部坐标
        self.path_converted_ = False

        # ========== 下面几个是模仿你C++中用到的全局变量/成员变量 ==========
        self.pre_point_idx_ = 0   # 对应C++: pre_point_idx
        self.distance_ = 0.0      # 对应C++: distance_
        self.ld_ = 0.5            # 对应C++: ld_, 先随便给个初值; 你也可每次loop里更新
        self.enabled_ = False
        
        
        
        self.create_subscription(
            Bool,
            '/use_recognition_flag',
            self.flag_callback,
            5
        )
        # =========== 订阅 =========== 
        self.path_sub_ = self.create_subscription(
            Path,
            '/origin_gnss_path',
            self.pathCallback,
            10
        )
        self.vectornav_sub_ = self.create_subscription(
            PoseWithCovarianceStamped,
            '/aiformula_sensing/vectornav/pose',
            self.vectornavCallback,
            10
        )

        # =========== 发布 =========== 
        self.cmd_pub_ = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # =========== 定时器(控制循环) =========== 
        timer_period = 1.0 / self.freq_
        self.timer_ = self.create_timer(timer_period, self.loop)

        self.get_logger().info("GnssLyaFollower with local origin & local yaw. Initialized.")

    def flag_callback(self, msg: Bool):
        """
        当 /use_recognition_flag = True => 说明要使用'识别轨迹'，本节点应该不再控制
        当 /use_recognition_flag = False => 本节点才执行控制
        """
        self.enabled_ = (not msg.data)  # 若flag=True => 识别生效 => 此节点停用
        self.get_logger().info(f"GnssLyaFollower => enabled_={self.enabled_}")


    def pathCallback(self, msg: Path):
        """
        收到 /origin_gnss_path (全局坐标).
        仅保存全局路径; 若已设基准且还未转换，则进行转换。
        """
        self.path_points_global_ = msg.poses
        self.get_logger().info(f"Got path with {len(self.path_points_global_)} poses (global).")

        if self.base_set_ and not self.path_converted_:
            self.convertPathToLocal()
            self.path_converted_ = True


    def vectornavCallback(self, msg: PoseWithCovarianceStamped):
        """
        收到 /vectornav/pose, 转 UTM; 若还没设置基准，则设置基准，
        并在基准设置后若已收到路径且未转换，则转换路径为局部坐标。
        """
        self.get_logger().info("vectornavCallback triggered!")
    
        # 1) 转成UTM
        x_ecef = msg.pose.pose.position.x
        y_ecef = msg.pose.pose.position.y
        z_ecef = msg.pose.pose.position.z
        x_utm, y_utm = ecef_to_utm(x_ecef, y_ecef, z_ecef)

        # 2) 计算全局 yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
        yaw_global = math.atan2(siny_cosp, cosy_cosp)

        # 3) 若还没设置基准 => 设置
        if not self.base_set_:
            self.base_x_ = x_utm
            self.base_y_ = y_utm
            self.base_yaw_ = yaw_global
            self.base_set_ = True
            self.get_logger().info(
                f"Base origin set => x:{self.base_x_:.3f}, y:{self.base_y_:.3f}, yaw:{math.degrees(self.base_yaw_):.2f} deg"
            )
            if len(self.path_points_global_) > 0 and not self.path_converted_:
                self.convertPathToLocal()
                self.path_converted_ = True

        # 4) 计算局部 (x_local, y_local, yaw_local)
        x_local = x_utm - self.base_x_
        y_local = y_utm - self.base_y_
        yaw_local = yaw_global - self.base_yaw_
        yaw_local = normalize_angle(yaw_local)

        self.current_x_local_ = x_local
        self.current_y_local_ = y_local
        self.current_yaw_local_ = yaw_local

        # 日志
        self.get_logger().info(
            f"Vehicle local pose => x:{x_local:.3f}, y:{y_local:.3f}, yaw:{math.degrees(yaw_local):.2f} deg"
        )


    def convertPathToLocal(self):
        """
        将全局路径转换到以 (base_x_, base_y_, base_yaw_) 为基准的局部坐标+局部朝向(若需要)
        这里仅示例平移.
        """
        self.path_points_local_.clear()
        for poseG_stamped in self.path_points_global_:
            gx = poseG_stamped.pose.position.x
            gy = poseG_stamped.pose.position.y

            lx = gx - self.base_x_
            ly = gy - self.base_y_

            local_pose = PoseStamped()
            local_pose.pose.position.x = lx
            local_pose.pose.position.y = ly
            local_pose.pose.position.z = 0.0

            self.path_points_local_.append(local_pose.pose)

        self.get_logger().info(
            f"Converted path to local => {len(self.path_points_local_)} points."
        )


    # ============== 关键部分：照抄你的 C++ 逻辑 ==============
    def findNearestIndex(self, front_wheel_pos):
        """
        在原先基础上，增加点积判定：只有点在车辆前方 (dot >= 0)
        才会触发后续判断。
        """
        # 先计算车辆朝向的单位向量
        ux = math.cos(self.current_yaw_local_)
        uy = math.sin(self.current_yaw_local_)

        for self.idx_ in range(self.pre_point_idx_, len(self.path_points_local_)):
            dx = self.path_points_local_[self.idx_].position.x - front_wheel_pos.position.x
            dy = self.path_points_local_[self.idx_].position.y - front_wheel_pos.position.y
            self.distance_ = math.hypot(dx, dy)

            # 点积: 如果 dot<0, 说明在车辆后方 => 跳过
            dot = dx * ux + dy * uy
            if dot < 0.0:
                continue  # 不在前方，直接跳过本次循环

            # 其余判据保持原样，只是额外加上 dot>=0.0 这一层
            if (self.distance_ > self.ld_
                and self.idx_ > self.pre_point_idx_
                and self.path_points_local_[self.idx_].position.x > 30000):
                self.pre_point_idx_ = self.idx_ - 1
                break

        # 无返回值，只更新 self.idx_ / self.pre_point_idx_
        return



    def loop(self):
        """
        定时器回调：若基准没设或路径点不足则不处理，否则做 LYA 控制
        """
        self.get_logger().info("timer triggered!")
        if not self.enabled_:
            # Flag表示识别在用 => 我们不动
            return
        if not self.base_set_:
            self.get_logger().info("loop() return: base not set yet.")
            return
        if len(self.path_points_local_) < 3:
            self.get_logger().info(f"loop() return: only {len(self.path_points_local_)} local points, need >=3.")
            return

        # ---------------------------------------------------------------------
        # 1) 先构造一个 front_wheel_pos (C++里是 geometry_msgs::msg::Pose front_wheel_pos)
        #    如果你确实要用"前轮"位置，就在这加上轮距换算；这里就直接把车辆当前位置当成 front_wheel_pos
        # ---------------------------------------------------------------------
        front_wheel_pos = PoseStamped().pose
        front_wheel_pos.position.x = self.current_x_local_
        front_wheel_pos.position.y = self.current_y_local_



        self.findNearestIndex(front_wheel_pos)

        # 若接近末尾
        if self.idx_ >= len(self.path_points_local_) - 2:
            self.stopRobot()
            self.get_logger().info("Reached or near the local path end => stop.")
            return

        # ---------------------------------------------------------------------
        # 3) 取 a,b,c 来做Lyapunov控制
        # ---------------------------------------------------------------------
        a_idx = self.idx_
        b_idx = self.idx_ + 1
        c_idx = self.idx_ + 2
        poseA = self.path_points_local_[a_idx]
        poseB = self.path_points_local_[b_idx]
        poseC = self.path_points_local_[c_idx]

        # Lyapunov控制
        x_now = self.current_x_local_
        y_now = self.current_y_local_
        yaw_now = self.current_yaw_local_
        v_cmd, w_cmd = self.computeLyaCommand(poseA, poseB, poseC, x_now, y_now, yaw_now)

        # 限幅
        v_cmd = max(min(v_cmd, self.v_max_), -self.v_max_)
        w_cmd = max(min(w_cmd, self.w_max_), -self.w_max_)

        # 发布
        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub_.publish(cmd)

        self.get_logger().info(
            f"cmd => v={v_cmd:.3f}, w={w_cmd:.3f}, idx={self.idx_}"
        )


    def computeLyaCommand(self, poseA, poseB, poseC, x_now, y_now, yaw_now):
        """
        在局部坐标系下做 Lyapunov 控制 (参考第二段).
        保持你原先的写法
        """
        ax = poseA.position.x
        ay = poseA.position.y
        bx = poseB.position.x
        by = poseB.position.y
        cx = poseC.position.x
        cy = poseC.position.y

        dx = ax - x_now
        dy = ay - y_now
        r = math.hypot(dx, dy)

        theta_a = math.atan2(dy, dx)
        alpha = normalize_angle(theta_a - yaw_now)

        theta_ab = math.atan2(by - ay, bx - ax)
        theta_bc = math.atan2(cy - by, cx - bx)
        dtheta = normalize_angle(theta_bc - theta_ab)

        # A、B、C 是空间路径点；由局部曲率和目标速度计算参考角速度
        heading_spacing = 0.5 * (
            math.hypot(bx - ax, by - ay) + math.hypot(cx - bx, cy - by)
        )
        omega_t = (
            0.0 if heading_spacing <= 1.0e-6
            else self.lya.v_t * dtheta / heading_spacing
        )
        self.lya.omega_t = omega_t

        v_t = self.lya.v_t
        lambda_v = self.lya.lambda_v
        lambda_a = self.lya.lambda_a
        k1 = self.lya.k1
        k2 = self.lya.k2

        phi_t = theta_ab
        beta = normalize_angle(theta_a - phi_t)

        epsilon = 1e-6
        if r < epsilon:
            return 0.0, 0.0

        v = (v_t * math.cos(beta) + lambda_v * r) * math.cos(alpha)
        sin_alpha = math.sin(alpha)
        sin_beta  = math.sin(beta)

        w = (lambda_a * sin_alpha
             + (k1 / (sin_alpha if abs(sin_alpha) > epsilon else epsilon)) *
               ((sin_alpha/(k1*r)) + (sin_beta/(k2*r))) *
               ((math.sin(2*alpha)*math.cos(beta)/2) - sin_beta) * v_t
             - (self.lya.omega_t * sin_beta / k2) *
               (k1 / (sin_alpha if abs(sin_alpha) > epsilon else epsilon))
        )
        return v, w


    def stopRobot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub_.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GnssLyaFollower()
    rclpy.spin(node)
    rclpy.shutdown()
