import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from example_interfaces.msg import Float32MultiArray
import math


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
        self.create_subscription(Odometry, '/aiformula_sensing/gyro_odometry_publisher/odom', self.odom_callback, 10)
        
        # 发布速度指令
        self.velocity_publisher = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # 定义轨迹
        self.create_subscription(Pose2D, '/filtered_lane_pose', self.trajectory_callback, 10)
        self.target_point = []
        self.current_pose = None
        # self.timer = self.create_timer(0.1, self.follow_trajectory)

        # 初始化LYA控制器
        self.lya = LYAController(omega_t=0.01, v_t=0.1, lambda_v=0.75, lambda_a=0.15, k1=0.8, k2=50)
      

        self.previous_time = self.get_clock().now()

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose  # 获取当前位姿
      

    def trajectory_callback(self, msg):
        # 更新目标点
        self.target_point = {"x": msg.x, "y": msg.y, "theta" : msg.theta}
        self.get_logger().info(f"New target received: x={msg.x}, y={msg.y}")

    # def follow_trajectory(self):
        if self.current_pose is None:
            return  # 等待里程计数据
 
        target_x = self.target_point["x"]
        target_y = self.target_point["y"]
        phi_t = self.target_point["theta"]


        # 获取当前位姿
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)
        self.get_logger().info(f"Current pose received: x={current_x}, y={current_y}, yaw={current_yaw}")
        # 计算控制误差
        dx = target_x - current_x
        dy = target_y - current_y
   
        r = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        alpha = theta - current_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # 归一化角度/processed_points_with_angle
        
   
        beta = theta - phi_t
        beta = math.atan2(math.sin(beta), math.cos(beta))  # 归一化角度

        # 通过LYA控制器计算速度指令
        v = ((self.lya.v_t) * math.cos(beta) + self.lya.lambda_v * r) * math.cos(alpha)

        omega = (self.lya.lambda_a * math.sin(alpha) +(self.lya.k1 / math.sin(alpha)) * 
        ((math.sin(alpha) / (self.lya.k1 * r)) + (math.sin(beta) / (self.lya.k2 * r))) * ((math.sin(2 * alpha) * math.cos(beta) / 2) - math.sin(beta)) * self.lya.v_t 
        -(self.lya.omega_t * math.sin(beta) / self.lya.k2) * (self.lya.k1 / math.sin(alpha)) 
        + (self.lya.k1 / math.sin(alpha)) * self.lya.lambda_v * (math.sin(2 * alpha) / 2) * 
       ((math.sin(alpha) / self.lya.k1) + (math.sin(beta) / self.lya.k2)))

        # 发布速度指令
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = omega
        self.velocity_publisher.publish(twist_msg)

    def quaternion_to_yaw(self, orientation):
        qx, qy, qz, qw = (
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    follower = TrajectoryFollower()
    rclpy.spin(follower)
    rclpy.shutdown()

if __name__ == '__main__':
    main()