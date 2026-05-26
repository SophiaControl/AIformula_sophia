import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
import math

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')

        # 订阅里程计
        self.create_subscription(Odometry, '/aiformula_sensing/gyro_odometry_publisher/odom', self.odom_callback, 10)
        
        # 发布速度指令
        self.velocity_publisher = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # 定义轨迹
        self.trajectory = [
            # 第一段直线
            {"x": 0.0, "y": 0.0},  
            {"x": 2.0, "y": 0.0},  
            {"x": 4.0, "y": 0.0},   
            {"x": 6.0, "y": 0.0},
            {"x": 8.0, "y": 0.0},
            {"x": 10.0, "y": 0.0},
            {"x": 12.0, "y": 0.0},
            # 第一段弧线
            {"x": 14.0, "y": 0.10},
            {"x": 15.96, "y": 0.44},
            {"x": 17.87, "y": 1.03},
            {"x": 19.74, "y": 1.78},
            {"x": 21.47, "y": 2.74},
            {"x": 23.11, "y": 3.90},
            {"x": 24.60, "y": 5.20},
            {"x": 25.94, "y": 6.73},
            {"x": 27.03, "y": 8.31},
            {"x": 28.00, "y": 10.0},
            # 第二段直线
            {"x": 28.00, "y": 10.00},
            {"x": 28.93, "y": 11.77},
            {"x": 29.86, "y": 13.54},
            {"x": 30.79, "y": 15.32},
            {"x": 31.71, "y": 17.09},
            {"x": 32.64, "y": 18.86},
            {"x": 33.57, "y": 20.63},
            {"x": 34.50, "y": 22.41},
            {"x": 35.43, "y": 24.18},
            {"x": 36.36, "y": 25.95},
            {"x": 37.28, "y": 27.74},
            {"x": 38.21, "y": 29.49},
            {"x": 39.14, "y": 31.26},
            {"x": 40.07, "y": 33.04},
            {"x": 41.00, "y": 34.79},
            {"x": 41.92, "y": 36.59},
            {"x": 42.85, "y": 38.35},
            {"x": 43.78, "y": 40.13},
            {"x": 44.71, "y": 41.90},
            {"x": 45.64, "y": 43.67},
            {"x": 46.57, "y": 45.49},
            {"x": 47.49, "y": 47.22},
            {"x": 48.42, "y": 48.99},
            {"x": 49.35, "y": 50.76},
            {"x": 50.00, "y": 52.00},
            # 第二段弧线
            {"x": 50.54, "y": 53.93},
            {"x": 50.87, "y": 55.90},
            {"x": 51.08, "y": 57.89},
            {"x": 51.06, "y": 59.89},
            {"x": 50.88, "y": 61.87},
            {"x": 50.55, "y": 63.88},
            {"x": 50.04, "y": 65.75},
            {"x": 49.35, "y": 67.66},
            {"x": 48.52, "y": 69.46},
            {"x": 47.55, "y": 71.21},
            {"x": 46.43, "y": 72.84},
            {"x": 45.08, "y": 74.47},
            {"x": 43.75, "y": 75.81},
            {"x": 42.20, "y": 77.11},
            {"x": 40.46, "y": 78.35},
            {"x": 38.85, "y": 79.27},
            {"x": 37.00, "y": 80.16},
            {"x": 35.15, "y": 80.86},
            {"x": 33.31, "y": 81.32},
            {"x": 31.23, "y": 81.76},
            {"x": 29.38, "y": 82.00},
            {"x": 28.00, "y": 82.00},
            # 第三段直线
            {"x": 26.00, "y": 82.00},
            {"x": 24.00, "y": 82.00},
            {"x": 22.00, "y": 82.00},
            {"x": 20.00, "y": 82.00},
            {"x": 18.00, "y": 82.00},
            {"x": 16.00, "y": 82.00},
            {"x": 14.00, "y": 82.00},
            {"x": 12.00, "y": 82.00},
            {"x": 10.00, "y": 82.00},
            {"x": 8.00, "y": 82.00},
            {"x": 6.00, "y": 82.00},
            {"x": 4.00, "y": 82.00},
            {"x": 2.00, "y": 82.00},
            {"x": 0.00, "y": 82.00},
            {"x": -2.00, "y": 82.00},
            {"x": -4.00, "y": 82.00},
            {"x": -6.00, "y": 82.00},
            {"x": -8.00, "y": 82.00},
            # 第三段弧线 
            {"x": -8.10, "y": 80.00},
            {"x": -8.40, "y": 78.03},
            {"x": -8.89, "y": 76.09},
            {"x": -9.58, "y": 74.21},
            {"x": -10.45, "y": 72.41},
            {"x": -11.49, "y": 70.72},
            {"x": -12.70, "y": 69.12},
            {"x": -14.07, "y": 67.65},
            {"x": -15.57, "y": 66.33},
            {"x": -17.19, "y": 65.17},
            {"x": -18.93, "y": 64.18},
            {"x": -20.75, "y": 63.36},
            {"x": -22.65, "y": 62.73},
            {"x": -24.03, "y": 62.40},
            {"x": -26.59, "y": 62.05},
            {"x": -28.00, "y": 62.00},
            # 第四段直线
            {"x": -28.00, "y": 62.00},
            {"x": -28.00, "y": 60.00},
            {"x": -28.00, "y": 58.00},
            {"x": -28.00, "y": 56.00},
            {"x": -28.00, "y": 54.00},
            {"x": -28.00, "y": 52.00},
            {"x": -28.00, "y": 50.00},
            {"x": -28.00, "y": 48.00},
            {"x": -28.00, "y": 46.00},
            {"x": -28.00, "y": 44.00},
            {"x": -28.00, "y": 42.00},
            {"x": -28.00, "y": 40.00},
            {"x": -28.00, "y": 38.00},
            {"x": -28.00, "y": 36.00},
            {"x": -28.00, "y": 34.00},
            {"x": -28.00, "y": 32.00},
            {"x": -28.00, "y": 30.00},
            {"x": -28.00, "y": 28.00},
            {"x": -28.00, "y": 26.00},
            {"x": -28.00, "y": 24.00},
            {"x": -28.00, "y": 22.00},
            {"x": -28.00, "y": 20.00},
            # 第四段弧线
            {"x": -27.90, "y": 18.00},
            {"x": -27.60, "y": 16.03},
            {"x": -27.11, "y": 14.09},
            {"x": -26.42, "y": 12.21},
            {"x": -25.55, "y": 10.41},
            {"x": -24.51, "y": 8.72},
            {"x": -23.30, "y": 7.12},
            {"x": -21.93, "y": 5.65},
            {"x": -20.43, "y": 4.33},
            {"x": -18.81, "y": 3.17},
            {"x": -17.07, "y": 2.18},
            {"x": -15.25, "y": 1.36},
            {"x": -13.34, "y": 0.73},
            {"x": -11.97, "y": 0.40},
            {"x": -9.41, "y": 0.05},
            {"x": -8.00, "y": 0.00},
            # 第五段直线
            {"x": -8.0, "y": 0.0},  
            {"x": -6.0, "y": 0.0},  
            {"x": -4.0, "y": 0.0},  
            {"x": -2.0, "y": 0.0},  
        ]
        self.current_target_index = 0

        self.current_pose = None
        self.timer = self.create_timer(0.1, self.follow_trajectory)

        # 初始化PID控制器
        self.linear_pid = PIDController(kp=0.5, ki=0.1, kd=0.05)
        self.angular_pid = PIDController(kp=2.0, ki=0.1, kd=0.1)

        self.previous_time = self.get_clock().now()

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose  # 获取当前位姿

    def follow_trajectory(self):
        if self.current_pose is None:
            return  # 等待里程计数据

        # 获取当前目标点
        target = self.trajectory[self.current_target_index]
        target_x = target["x"]
        target_y = target["y"]

        # 获取当前位姿
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # 计算控制误差
        dx = target_x - current_x
        dy = target_y - current_y
        distance_error = math.sqrt(dx ** 2 + dy ** 2)
        desired_yaw = math.atan2(dy, dx)
        yaw_error = desired_yaw - current_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))  # 归一化角度

        # 获取当前时间步长
        current_time = self.get_clock().now()
        dt = (current_time - self.previous_time).nanoseconds / 1e9
        self.previous_time = current_time

        # 通过PID控制器计算速度指令
        linear_velocity = self.linear_pid.compute(distance_error, dt)
        angular_velocity = self.angular_pid.compute(yaw_error, dt)

        # 限制速度
        linear_velocity = max(min(linear_velocity, 0.5), -0.5)
        angular_velocity = max(min(angular_velocity, 2.0), -2.0)

        # 发布速度指令
        twist_msg = Twist()
        twist_msg.linear.x = linear_velocity
        twist_msg.angular.z = angular_velocity
        self.velocity_publisher.publish(twist_msg)

        # 检查是否接近目标点
        if distance_error < 0.1:  # 如果距离足够近，切换到下一个目标点
            self.current_target_index += 1
            if self.current_target_index >= len(self.trajectory):  # 轨迹完成
                self.stop_robot()

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