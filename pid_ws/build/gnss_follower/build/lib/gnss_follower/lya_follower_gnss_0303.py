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
        """
        super().__init__('gnss_lya_follower_node')

        # =========== 可调参数 ===========
        self.freq_ = 2.0       # 控制频率
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
        self.path_points_local_ = []    # 转换后存到此
        self.idx_ = 0

        # =========== 订阅 =========== 
        self.path_sub_ = self.create_subscription(Path,
                                                  '/origin_gnss_path',
                                                  self.pathCallback,
                                                  10)
        self.vectornav_sub_ = self.create_subscription(PoseWithCovarianceStamped,
                                                       '/aiformula_sensing/vectornav/pose',
                                                       self.vectornavCallback,
                                                       10)

        # =========== 发布 ===========
        self.cmd_pub_ = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # =========== 定时器(控制循环) ===========
        timer_period = 1.0 / self.freq_
        self.timer_ = self.create_timer(timer_period, self.loop)

        self.get_logger().info("GnssLyaFollower with local origin & local yaw. Initialized.")


    def pathCallback(self, msg: Path):
        """
        收到 /origin_gnss_path (全局坐标).
        如果未设基准，则仅暂存；若基准已设，则转换成局部坐标.
        """
        self.path_points_global_ = [pose_st.pose for pose_st in msg.poses]
        self.idx_ = 0
        self.get_logger().info(f"Got path with {len(self.path_points_global_)} points (global).")

        # 若已设置基准, 立即转局部
        if self.base_set_:
            self.convertPathToLocal()
        else:
            self.get_logger().info("Path received but base not set => will convert after first GNSS pose is received.")


    def vectornavCallback(self, msg):
        """
        收到 /vectornav/pose, 转UMT. 如果没设基准, 第一次时同时设 base_x_, base_y_, base_yaw_.
        然后计算车辆在局部坐标中的 (x_local, y_local, yaw_local).
        """
        self.get_logger().info(f"[vectornavCallback] self id = {hex(id(self))}")

        self.get_logger().info("navcallback triggered") 

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

        # 3) 若还没设置基准，就设置
        if not self.base_set_:
            self.base_x_ = x_utm
            self.base_y_ = y_utm
            self.base_yaw_ = yaw_global
            self.base_set_ = True
            self.get_logger().info(
                f"Base origin set => x:{self.base_x_:.3f}, y:{self.base_y_:.3f}, yaw:{math.degrees(self.base_yaw_):.2f} deg"
            )

            # 如果已经有路径, 转到局部
            if len(self.path_points_global_) > 0:
                self.convertPathToLocal()

        # 4) 计算局部 (x_local, y_local, yaw_local)
        x_local = x_utm - self.base_x_
        y_local = y_utm - self.base_y_
        yaw_local = yaw_global - self.base_yaw_
        yaw_local = normalize_angle(yaw_local)

        self.current_x_local_ = x_local
        self.current_y_local_ = y_local
        self.current_yaw_local_ = yaw_local

        # 5) 日志
        self.get_logger().info(
            f"Vehicle local pose => x:{x_local:.3f}, y:{y_local:.3f}, yaw:{math.degrees(yaw_local):.2f} deg"
        )


    def convertPathToLocal(self):
        """
        将全局路径转换到以 (base_x_, base_y_, base_yaw_) 为基准的局部坐标+局部朝向(若需要)
        这里仅示例平移. 
        若要路径中也考虑旋转, 可以做对应处理(见备注).
        """
        self.path_points_local_.clear()
        for poseG in self.path_points_global_:
            gx = poseG.position.x
            gy = poseG.position.y

            # 先做平移
            lx = gx - self.base_x_
            ly = gy - self.base_y_

            # 若想连同 yaw 也一起转换(比如路径有 orientation),
            # 可再取 orientation->yaw, 减 base_yaw_, 并放回局部 orientation.
            # 这里就不多赘述, 仅示例2D点
            local_pose = PoseStamped()
            local_pose.pose.position.x = lx
            local_pose.pose.position.y = ly
            local_pose.pose.position.z = 0.0

            self.path_points_local_.append(local_pose.pose)

        self.get_logger().info(
            f"Converted path to local => {len(self.path_points_local_)} points."
        )


    def loop(self):
        self.get_logger().info(f"[loop] self id = {hex(id(self))}")

        self.get_logger().info("loop() timer triggered")  # 或者 debug
        if not self.base_set_:
            self.get_logger().info("base not set, skip loop")
            return
        if len(self.path_points_local_) < 3:
            self.get_logger().info("path_points_local_ too few, skip loop")
            return
    


        x_now = self.current_x_local_.conjugate
        y_now = self.current_y_local_
        yaw_now = self.current_yaw_local_

        # 若接近末尾
        if self.idx_ >= len(self.path_points_local_) - 2:
            self.stopRobot()
            self.get_logger().info("Reached or near the local path end => stop.")
            return

        # 如果离当前目标很近就 idx_++
        a_x = self.path_points_local_[self.idx_].position.x
        a_y = self.path_points_local_[self.idx_].position.y
        dist_a = math.sqrt((a_x - x_now)**2+(a_y - y_now)**2)
        if dist_a < 0.5:
            self.idx_ += 1
            if self.idx_ >= len(self.path_points_local_) - 2:
                self.stopRobot()
                return

        # 取 a,b,c
        a_idx = self.idx_
        b_idx = self.idx_ + 1
        c_idx = self.idx_ + 2
        poseA = self.path_points_local_[a_idx]
        poseB = self.path_points_local_[b_idx]
        poseC = self.path_points_local_[c_idx]

        # Lyapunov控制
        v_cmd, w_cmd = self.computeLyaCommand(poseA, poseB, poseC, x_now, y_now, yaw_now)

        # 限幅
        v_cmd = max(min(v_cmd, self.v_max_), -self.v_max_)
        w_cmd = max(min(w_cmd, self.w_max_), -self.w_max_)

        # 发布
        twist_msg = Twist()
        twist_msg.linear.x = v_cmd
        twist_msg.angular.z = w_cmd
        self.cmd_pub_.publish(twist_msg)

        self.get_logger().info(
            f"cmd => v={v_cmd:.3f}, w={w_cmd:.3f}, idx={self.idx_}"
        )


    def computeLyaCommand(self, poseA, poseB, poseC, x_now, y_now, yaw_now):
        """
        在局部坐标系下做 Lyapunov 控制 (参考第二段).
        poseA,B,C 也是局部坐标; x_now,y_now,yaw_now 都是相对 base_ yaw=0.
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

        # 由 b->c 的方向估算参考角速度
        omega_t = dtheta
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
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_pub_.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GnssLyaFollower()
    rclpy.spin(node)
    rclpy.shutdown()
