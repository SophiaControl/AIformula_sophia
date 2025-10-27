import tf2_geometry_msgs 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from example_interfaces.msg import Float32MultiArray
import math
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf_transformations

class LYAController:
    def __init__(self, omega_t, v_t, lambda_v,lambda_a,k1,k2):
        self.omega_t = omega_t
        self.v_t = v_t
        self.lambda_v = lambda_v
        self.lambda_a = lambda_a
        self.k1 = k1
        self.k2 = k2

class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')

        # 订阅里程计
        self.create_subscription(Odometry, '/aiformula_sensing/gyro_odometry_publisher/odom', self.odom_callback, 5)
        
        # 发布速度指令
        self.velocity_publisher = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # 初始化tf buffer和listener，用于坐标变换
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        # 定义轨迹
        self.create_subscription(Pose2D, '/filtered_lane_pose', self.trajectory_callback, 5)
        self.target_point = []
        self.current_pose = None
        # self.timer = self.create_timer(0.1, self.follow_trajectory)

        # 初始化LYA控制器
        self.lya = LYAController(omega_t=0.01, v_t=2.0, lambda_v=0.15, lambda_a=2.5, k1=0.8, k2=20)
      

        self.previous_time = self.get_clock().now()

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
        pose_in.header.frame_id = "base_link"  # 假设当前目标点是以车辆坐标系(base_link)为参考
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
            # 根据您的系统选择合适的目标坐标系，此处假设"odom"为全局坐标系
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
        self.get_logger().info(f"New target received (global): x={global_x}, y={global_y}, theta={global_theta}")

    # def follow_trajectory(self):
        if self.current_pose is None:
            return  # 等待里程计数据
 
        target_x = self.target_point["x"]
        target_y = self.target_point["y"]
        phi_t = self.target_point["theta"]


        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_qx, current_qy, current_qz, current_qw = (
        self.current_pose.orientation.x,
        self.current_pose.orientation.y,
        self.current_pose.orientation.z,
        self.current_pose.orientation.w
       )

        # 使用transformations库从四元数直接计算出欧拉角
        current_roll, current_pitch, current_yaw = tf_transformations.euler_from_quaternion([current_qx, current_qy, current_qz, current_qw])


        self.get_logger().info(f"Current pose received: x={current_x}, y={current_y}, yaw={current_yaw}")
        # 计算控制误差
        dx = target_x - current_x
        dy = target_y - current_y
   
        r = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        alpha = theta - current_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # 归一化角度
        
   
        beta = theta - phi_t
        beta = math.atan2(math.sin(beta), math.cos(beta))  # 归一化角度

        # 除零保护阈值设定
        epsilon = 1e-6

    # 如果r过小，说明已经接近目标点，可以不移动
        if abs(r) < epsilon:
          v = self.lya.v_t
          omega = 0.0
        else:
    # 确保sin(alpha)不为零，如果为零则给它一个非常小的值
          sin_alpha = math.sin(alpha)
        if abs(sin_alpha) < epsilon:
        # 当alpha非常接近0时，可以选择将其设为epsilon，避免除零
        # 或根据控制策略，如果alpha ~ 0，意味着朝向误差小，可以尝试简单策略。
           sin_alpha = epsilon

    # 同样检查sin(beta)，避免后续使用问题（如果有需要）
        sin_beta = math.sin(beta)
        if abs(sin_beta) < epsilon:
          sin_beta = epsilon


        v = ((self.lya.v_t) * math.cos(beta) + self.lya.lambda_v * r) * math.cos(alpha)

    # 在计算omega时替换math.sin(alpha)和math.sin(beta)为sin_alpha和sin_beta
    # 同时，r已检查过，不应为0
        omega = (self.lya.lambda_a * sin_alpha 
             + (self.lya.k1 / sin_alpha) * 
               ((sin_alpha / (self.lya.k1 * r)) + (sin_beta / (self.lya.k2 * r))) * 
               ((math.sin(2 * alpha) * math.cos(beta) / 2) - sin_beta) * self.lya.v_t 
             - (self.lya.omega_t * sin_beta / self.lya.k2) * (self.lya.k1 / sin_alpha) 
             + (self.lya.k1 / sin_alpha) * self.lya.lambda_v * (math.sin(2 * alpha) / 2) * 
               ((sin_alpha / self.lya.k1) + (sin_beta / self.lya.k2)))
             
        self.get_logger().info(f" v={v}, omega={omega}")

        # 发布速度指令
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
