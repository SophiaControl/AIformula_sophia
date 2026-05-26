import tf2_geometry_msgs
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Pose2D, PoseStamped, TransformStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from example_interfaces.msg import Float32MultiArray
import math
import pandas as pd
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf_transformations

try:
    import pyproj
    _has_pyproj = True
except ImportError:
    _has_pyproj = False


def ecef_to_utm(x_ecef, y_ecef, z_ecef):
    """
    若 /vectornav/pose 是 ECEF坐标，则转 UTM; 
    若已是 UTM, 直接 return (x_ecef, y_ecef).
    """
    if not _has_pyproj:
        return x_ecef, y_ecef
    transformer = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:32654", always_xy=True)
    x_utm, y_utm, _ = transformer.transform(x_ecef, y_ecef, z_ecef)
    return x_utm, y_utm


def normalize_angle(rad):
    return math.atan2(math.sin(rad), math.cos(rad))


class LYAController:
    def __init__(self, v_t, lambda_v, lambda_a, k1, k2):
        self.v_t = v_t
        self.lambda_v = lambda_v
        self.lambda_a = lambda_a
        self.k1 = k1
        self.k2 = k2
        
        # 保留：给控制器一个默认 omega_t，后面由 '/filtered_omega_t' 回调来更新
        self.omega_t = 0.0


class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')
        
        self.enabled_ = False   # 默认不启用
        self.create_subscription(
            Bool,
            '/use_recognition_flag',  # 你定义的布尔话题
            self.flag_callback,
            5
        )

        # ============ 订阅 '/filtered_omega_t' ============ 
        self.create_subscription(
            Pose2D, 
            '/filtered_omega_t', 
            self.omega_t_callback, 
            5
        )

        # ============ 订阅 '/vectornav/pose'，作为我们的“GNSS”来源 ============
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/aiformula_sensing/vectornav/pose',
            self.gnss_callback,
            5
        )

        # ============ 订阅目标 (Pose2D)，和原来相同 ============ 
        self.create_subscription(
            Pose2D, 
            '/filtered_lane_pose', 
            self.trajectory_callback, 
            5
        )

        # ============ 发布速度指令 (与原来相同) ============ 
        self.velocity_publisher = self.create_publisher(
            Twist, 
            '/aiformula_control/game_pad/cmd_vel', 
            10
        )

        # ============ 初始化 tf buffer, listener, broadcaster (与原来相同) ============
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ============ 初始化 LYA 控制器 (与原来相同) ============
        self.lya = LYAController(v_t=3.0, lambda_v=0.15, lambda_a=3.5, k1=0.8, k2=25)

        # ============ 其他变量 ============
        self.target_point = []
        # self.current_pose = None   # 原先你存 odom回调，这里不用了
        self.previous_time = self.get_clock().now()

        # ===== 新增: 用于GNSS基准 =====
        self.base_set_ = False
        self.base_x_ = 0.0
        self.base_y_ = 0.0
        self.base_yaw_ = 0.0
        self.current_x_ = 0.0
        self.current_y_ = 0.0
        self.current_yaw_ = 0.0
        self.pose_valid_ = False

        self.get_logger().info("TrajectoryFollower node init done, waiting for GNSS data...")



    def flag_callback(self, msg: Bool):
        """
        当 /use_recognition_flag = True => 启用本节点的识别控制
        否则 => 不做事
        """
        self.enabled_ = msg.data
        self.get_logger().info(f"TrajectoryFollower => enabled_={self.enabled_}")


    # ============ A) 订阅 '/filtered_omega_t' 回调 (保留) ============
    def omega_t_callback(self, msg):
        """
        这条信息的 x,y 无意义,theta 才是真正需要的 omega_t
        
        """
        if not self.enabled_:
            return
        self.lya.omega_t = msg.theta
        self.get_logger().info(f"Received omega_t={msg.theta:.3f} from /filtered_omega_t")

    # ============ B) GNSS 回调 (代替原先 odom_callback) ============
    def gnss_callback(self, msg: PoseWithCovarianceStamped):
        """
        取自 /vectornav/pose, 可能是 ECEF => 转 UTM => (base_x_, base_y_, base_yaw_) => "gnss_map"
        并发布 gnss_map -> base_link
        """
        # 1) ECEF->UTM
        x_ecef = msg.pose.pose.position.x
        y_ecef = msg.pose.pose.position.y
        z_ecef = msg.pose.pose.position.z
        x_utm, y_utm = ecef_to_utm(x_ecef, y_ecef, z_ecef)

        # 2) 计算yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0*(qw*qz + qx*qy)
        cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
        yaw_global = math.atan2(siny_cosp, cosy_cosp)

        # 3) 若没设置 base => 此时设置
        if not self.base_set_:
            self.base_x_ = x_utm
            self.base_y_ = y_utm
            self.base_yaw_ = yaw_global
            self.base_set_ = True
            self.get_logger().info(
                f"GNSS base set => x:{self.base_x_:.3f}, y:{self.base_y_:.3f}, yaw(deg)={math.degrees(self.base_yaw_):.2f}"
            )

        # 4) 计算车辆在“gnss_map”下的坐标
        local_x = x_utm - self.base_x_
        local_y = y_utm - self.base_y_
        local_yaw = normalize_angle(yaw_global - self.base_yaw_)

        self.current_x_ = local_x
        self.current_y_ = local_y
        self.current_yaw_ = local_yaw
        self.pose_valid_ = True

        # 5) 发布 TF: gnss_map->base_link
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'gnss_map'   # 新的全局坐标系(相当于原 "odom")
        t.child_frame_id = 'base_link'

        t.transform.translation.x = local_x
        t.transform.translation.y = local_y
        t.transform.translation.z = 0.0

        q_out = tf_transformations.quaternion_from_euler(0, 0, local_yaw)
        t.transform.rotation.x = q_out[0]
        t.transform.rotation.y = q_out[1]
        t.transform.rotation.z = q_out[2]
        t.transform.rotation.w = q_out[3]

        self.tf_broadcaster.sendTransform(t)

    # ============ C) trajectory_callback (几乎不变, 只是把 "odom" 改成 "gnss_map") ============
    def trajectory_callback(self, msg):
        """
        1) 将Pose2D转为PoseStamped（在 base_link 下）
        2) 用tf_buffer.transform到 'gnss_map'
        3) 提取 global_x, global_y, global_theta
        4) 跟当前车辆位置 self.current_x_, self.current_y_, self.current_yaw_ 做Lyapunov
        5) 用 self.lya.omega_t
        6) 发布速度
        """
        if not self.enabled_:
            return
        
    
        # 先把 Pose2D -> PoseStamped (base_link)
        pose_in = PoseStamped()
        pose_in.header.stamp = self.get_clock().now().to_msg()
        pose_in.header.frame_id = "base_link"
        # 下面两行你原先就有，就保留：
        pose_in.header.stamp.sec = 0
        pose_in.header.stamp.nanosec = 0

        pose_in.pose.position.x = msg.x
        pose_in.pose.position.y = msg.y
        pose_in.pose.position.z = 0.0

        q = tf_transformations.quaternion_from_euler(0, 0, msg.theta)
        pose_in.pose.orientation.x = q[0]
        pose_in.pose.orientation.y = q[1]
        pose_in.pose.orientation.z = q[2]
        pose_in.pose.orientation.w = q[3]

        # 尝试从 base_link -> gnss_map
        try:
            pose_out = self.tf_buffer.transform(pose_in, 'gnss_map', timeout=Duration(seconds=0.2))
        except Exception as e:
            self.get_logger().warn(f"Failed to transform target point to gnss_map: {e}")
            return

        # 提取全局坐标
        global_x = pose_out.pose.position.x
        global_y = pose_out.pose.position.y
        _, _, global_theta = tf_transformations.euler_from_quaternion([
            pose_out.pose.orientation.x,
            pose_out.pose.orientation.y,
            pose_out.pose.orientation.z,
            pose_out.pose.orientation.w
        ])

        self.target_point = {"x": global_x, "y": global_y, "theta": global_theta}
        self.get_logger().info(
            f"New target (gnss_map): x={global_x:.2f}, y={global_y:.2f}, theta={global_theta:.2f}"
        )

        # 若还未收到GNSS数据 => 不执行后面控制
        if not self.pose_valid_:
            self.get_logger().warn("No valid GNSS pose yet, skip control.")
            return

        # 取目标 & 当前姿态
        target_x = self.target_point["x"]
        target_y = self.target_point["y"]
        phi_t    = self.target_point["theta"]

        current_x = self.current_x_
        current_y = self.current_y_
        current_yaw = self.current_yaw_

        self.get_logger().info(
            f"Current (gnss_map): x={current_x:.2f}, y={current_y:.2f}, yaw={current_yaw:.2f}"
        )

        # ---------- 计算控制误差 ----------
        dx = target_x - current_x
        dy = target_y - current_y
        r = math.sqrt(dx**2 + dy**2)

        theta = math.atan2(dy, dx)
        alpha = normalize_angle(theta - current_yaw)
        beta  = normalize_angle(theta - phi_t)

        # ---------- 除零保护 ----------
        epsilon = 1e-6
        if abs(r) < epsilon:
            v = self.lya.v_t
            omega = 0.0
        else:
            sin_alpha = math.sin(alpha)
            if abs(sin_alpha) < epsilon:
                sin_alpha = epsilon

            sin_beta = math.sin(beta)
            if abs(sin_beta) < epsilon:
                sin_beta = epsilon

            # ---------- 计算线速度 v ----------
            v = ( (self.lya.v_t)*math.cos(beta) + self.lya.lambda_v*dx ) * math.cos(alpha)

            # ---------- 计算角速度，需要 self.lya.omega_t ----------
            # 保留你原先那套公式:
            omega = ( self.lya.lambda_a * sin_alpha
                    + (self.lya.k1 / sin_alpha) * 
                      ((sin_alpha/(self.lya.k1*r)) + (sin_beta/(self.lya.k2*r))) * 
                      ((math.sin(2*alpha)*math.cos(beta)/2) - sin_beta) * self.lya.v_t
                    - (self.lya.omega_t * sin_beta / self.lya.k2) * (self.lya.k1 / sin_alpha)
                    + (self.lya.k1 / sin_alpha) * self.lya.lambda_v * (math.sin(2 * alpha) / 2) *
                      ((sin_alpha / self.lya.k1) + (sin_beta / self.lya.k2))
            )

        self.get_logger().info(f"Control => v={v:.3f}, omega={omega:.3f}")

        # ---------- 发布速度指令 ----------
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = omega
        self.velocity_publisher.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    follower = TrajectoryFollower()
    rclpy.spin(follower)
    rclpy.shutdown()

