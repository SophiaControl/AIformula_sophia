import tf2_geometry_msgs 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from example_interfaces.msg import Float32MultiArray
import math
import pandas as pd
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf_transformations

class LYAController:
    def __init__(self, v_t, lambda_v, lambda_a, k1, k2):
        self.v_t = v_t
        self.lambda_v = lambda_v
        self.lambda_a = lambda_a
        self.k1 = k1
        self.k2 = k2
        
        # ============ 新增：给这个控制器一个默认的omega_t ============
        self.omega_t = 0.0   # 初始先设为 0，后面在回调中再更新

class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')

        # ---- (1) 订阅里程计 ----
        self.create_subscription(Odometry, 
                                 '/aiformula_sensing/gyro_odometry_publisher/odom', 
                                 self.odom_callback, 
                                 5)
        
        # ---- (2) 发布速度指令 ----
        self.velocity_publisher = self.create_publisher(Twist, 
                                                        '/aiformula_control/game_pad/cmd_vel', 
                                                        10)

        # ---- (3) 初始化tf buffer和listener，用于坐标变换 ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ============ 新增：订阅 '/filtered_omega_t' 以获得 omega_t ============
        self.create_subscription(
            Pose2D, 
            '/filtered_omega_t', 
            self.omega_t_callback, 
            10
        )
        # ---- (4) 定义轨迹(订阅) ----
        self.create_subscription(Pose2D, 
                                 '/filtered_lane_pose', 
                                 self.trajectory_callback, 
                                 10)
        self.target_point = []
        self.current_pose = None



        # ---- (5) 初始化LYA控制器 ----
        self.lya = LYAController(
            v_t=1.25, 
            lambda_v=0.15, 
            lambda_a=1.5, 
            k1=1.0, 
            k2=20
        )
        # (此时 self.lya.omega_t = 0.0)

        # ---- 其他变量初始化 ----
        self.previous_time = self.get_clock().now()

    # ============ (A) 新增的回调，用于接收 filtered_omega_t =============
    def omega_t_callback(self, msg):
        """
        这条信息的 x,y 无意义,theta 才是真正需要的 omega_t
        """
        self.lya.omega_t = msg.theta
        self.get_logger().info(f"Received omega_t={msg.theta:.3f} from /filtered_omega_t")

    def odom_callback(self, msg: Odometry):
        # 获取当前位姿（在 odom 坐标系下）
        self.current_pose = msg.pose.pose  
        
        # 从里程计中提取平移和旋转
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_qx = self.current_pose.orientation.x
        current_qy = self.current_pose.orientation.y
        current_qz = self.current_pose.orientation.z
        current_qw = self.current_pose.orientation.w

        # 发布 odom -> base_link 的动态变换
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'         # 父坐标系
        t.child_frame_id = 'base_link'     # 子坐标系

        t.transform.translation.x = current_x
        t.transform.translation.y = current_y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = current_qx
        t.transform.rotation.y = current_qy
        t.transform.rotation.z = current_qz
        t.transform.rotation.w = current_qw

        self.tf_broadcaster.sendTransform(t)

    def trajectory_callback(self, msg):
        # 将Pose2D转为PoseStamped（假设输入坐标系为"base_link"）
        pose_in = PoseStamped()
        pose_in.header.stamp = self.get_clock().now().to_msg()
        pose_in.header.frame_id = "base_link"  
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
      
        try:
            # 根据实际情况选择目标坐标系, 这里假设 "odom" 为全局坐标系
            pose_out = self.tf_buffer.transform(pose_in, 'odom', timeout=Duration(seconds=0.2))
        except Exception as e:
            self.get_logger().warn(f"Failed to transform target point to global frame: {e}")
            return

        # 提取全局坐标
        global_x = pose_out.pose.position.x
        global_y = pose_out.pose.position.y
        # 将四元数转为yaw
        _, _, global_theta = tf_transformations.euler_from_quaternion([
            pose_out.pose.orientation.x,
            pose_out.pose.orientation.y,
            pose_out.pose.orientation.z,
            pose_out.pose.orientation.w
        ])

        # 更新目标点为全局坐标
        self.target_point = {"x": global_x, "y": global_y, "theta": global_theta}
        self.get_logger().info(
            f"New target received (global): x={global_x:.2f}, y={global_y:.2f}, theta={global_theta:.2f}"
        )

        # === 若还未收到里程计数据(或尚未初始化), 就先不执行后面控制 ===
        if self.current_pose is None:
            return

        # ---------- 提取目标 & 当前姿态 ----------
        target_x = self.target_point["x"]
        target_y = self.target_point["y"]
        phi_t    = self.target_point["theta"]

        current_x  = self.current_pose.position.x
        current_y  = self.current_pose.position.y
        current_qx = self.current_pose.orientation.x
        current_qy = self.current_pose.orientation.y
        current_qz = self.current_pose.orientation.z
        current_qw = self.current_pose.orientation.w

        # 使用 transform 库从四元数直接计算欧拉角
        current_roll, current_pitch, current_yaw = \
            tf_transformations.euler_from_quaternion([current_qx, 
                                                      current_qy, 
                                                      current_qz, 
                                                      current_qw])

        self.get_logger().info(f"Current pose: x={current_x:.2f}, y={current_y:.2f}, yaw={current_yaw:.2f}")

        # ---------- 计算控制误差 ----------
        dx = target_x - current_x
        dy = target_y - current_y
        r = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy, dx)
        alpha = theta - current_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # 归一化角度
        
        beta = theta - phi_t
        beta = math.atan2(math.sin(beta), math.cos(beta))     # 归一化角度

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
        v = ((self.lya.v_t) * math.cos(beta) + self.lya.lambda_v * dx) * math.cos(alpha)

        
            # ============ 计算角速度时，需要 self.lya.omega_t ============
            # 这里 self.lya.omega_t 就是我们在 omega_t_callback() 里更新的值
        omega = (self.lya.lambda_a * sin_alpha 
                + (self.lya.k1 / sin_alpha) * 
                    ((sin_alpha / (self.lya.k1 * r)) + (sin_beta / (self.lya.k2 * r))) * 
                    ((math.sin(2 * alpha) * math.cos(beta) / 2) - sin_beta) * self.lya.v_t 
                - (self.lya.omega_t * sin_beta / self.lya.k2) * (self.lya.k1 / sin_alpha) 
                + (self.lya.k1 / sin_alpha) * self.lya.lambda_v * (math.sin(2 * alpha) / 2) * 
                    ((sin_alpha / self.lya.k1) + (sin_beta / self.lya.k2))
                )
            
        # if abs(r) < 1e-5 :
        #     r = 1e-5

        # if abs(alpha) < 1e-5 :
        #     alpha = 1e-5

        # v = ((self.lya.v_t) * math.cos(beta) + self.lya.lambda_v * dx) * math.cos(alpha)
        
        # omega = alpha/(2 * alpha - beta) * (self.lya.lambda_a * alpha + (self.lya.v_t / r) * (math.sin(2 * alpha) * math.cos(beta)/2 - math.sin(beta)) + math.sin(2 * alpha) * self.lya.lambda_v / 2)
        # + (alpha - beta) * self.lya.omega_t / (2 *alpha - beta)
            
        self.get_logger().info(f"Control => v={v:.3f}, omega={omega:.3f}")

        max_linear_velocity = 2.0   
        min_linear_velocity = 0.5  
        max_angular_velocity = 1.0  
        min_angular_velocity = -1.0

        v = max(min_linear_velocity, min(v, max_linear_velocity))
        omega = max(min_angular_velocity, min(omega, max_angular_velocity))

        # if abs(omega) < 0.20 :
        #     omega = 0.0

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

if __name__ == '__main__':
    main()
