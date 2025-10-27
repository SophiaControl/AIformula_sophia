import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from example_interfaces.msg import Float32MultiArray
import math

class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')

        # 订阅里程计
        self.create_subscription(Odometry, '/aiformula_sensing/gyro_odometry_publisher/odom', self.odom_callback, 10)
        
        # 发布速度指令
        self.velocity_publisher = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # 定义轨迹
        self.create_subscription(Float32MultiArray, '/processed_points_with_angle', self.trajectory_callback, 10)
        self.current_target_index = 0

        self.current_pose = None
        self.timer = self.create_timer(0.1, self.follow_trajectory)

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose  # 获取当前位姿

    def trajectory_callback(self, msg):
        # 更新目标点
        self.target_point = {"x": msg.x, "y": msg.y}
        self.get_logger().info(f"New target received: x={msg.x}, y={msg.y}")

    def follow_trajectory(self):
        if self.current_pose is None:
            return  # 等待里程计数据

        # 获取当前目标点
        self.create_subscription[self.current_target_index]
        target_x = self.target_point["x"]
        target_y = self.target_point["y"]

        # 获取当前位姿
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # 控制器计算速度指令
        linear_velocity, angular_velocity = self.control_to_target(
            current_x, current_y, current_yaw, target_x, target_y
        )

        # 发布速度指令
        twist_msg = Twist()
        twist_msg.linear.x = linear_velocity
        twist_msg.angular.z = angular_velocity
        self.velocity_publisher.publish(twist_msg)

        # 检查是否接近目标点
        distance_error = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
        if distance_error < 0.1:  # 如果距离足够近，切换到下一个目标点
            self.current_target_index += 1
            if self.current_target_index >= len(self.create_publisher):  # 轨迹完成
                self.stop_robot()

    def control_to_target(self, current_x, current_y, current_yaw, target_x, target_y):
        dx = target_x - current_x
        dy = target_y - current_y
        distance_error = math.sqrt(dx ** 2 + dy ** 2)
        desired_yaw = math.atan2(dy, dx)
        yaw_error = desired_yaw - current_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))  # 归一化角度

        linear_velocity = min(0.5, distance_error)
        angular_velocity = 2.0 * yaw_error
        return linear_velocity, angular_velocity

    def quaternion_to_yaw(self, orientation):
        qx, qy, qz, qw = (
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def stop_robot(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.velocity_publisher.publish(twist_msg)
        self.get_logger().info("Trajectory completed.")
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    follower = TrajectoryFollower()
    rclpy.spin(follower)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
