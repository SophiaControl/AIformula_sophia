import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
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
        self.trajectory = [
            {"x": 0.0, "y": 0.0},  
            {"x": 2.0, "y": 0.0},  
            {"x": 4.0, "y": 0.0},   
            {"x": 6.0, "y": 0.0},
            {"x": 8.0, "y": 0.0},
            {"x": 10.0, "y": 0.0},
            {"x": 12.0, "y": 0.0},
            {"x": 14.0, "y": 0.0},
            {"x": 16.0, "y": 0.0},
            {"x": 18.0, "y": 0.0},
            {"x": 20.0, "y": 0.0},
            {"x": 22.0, "y": 0.0},
            {"x": 24.0, "y": 0.0},
            {"x": 26.0, "y": 0.0},
            {"x": 28.0, "y": 0.0},
            {"x": 30.0, "y": 0.0},
            {"x": 37.0, "y": -3.0},
            {"x": 41.0, "y": -6.0},
            {"x": 44.0, "y": -10.0},
            {"x": 47.0, "y": -20.0},
            {"x": 47.0, "y": -22.0},
            {"x": 47.0, "y": -24.0},
            {"x": 47.0, "y": -26.0},  
        ]
        self.current_target_index = 0
        self.current_pose = None
        self.timer = self.create_timer(0.1, self.follow_trajectory)

        # 初始化LYA控制器
        self.lya = LYAController(omega_t=0.01, v_t=1, lambda_v=0.75, lambda_a=0.15, k1=0.8, k2=50)
      

        self.previous_time = self.get_clock().now()

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose  # 获取当前位姿

    def follow_trajectory(self):
        if self.current_pose is None:
            return  # 等待里程计数据

        # 获取当前目标点
        if self.current_target_index >= len(self.trajectory) - 1:
            self.stop_robot()
            return
        
        target = self.trajectory[self.current_target_index]
        target_x = target["x"]
        target_y = target["y"]

        next_target = self.trajectory[self.current_target_index + 1]
        target_x_next = next_target["x"]
        target_y_next = next_target["y"]

        # 获取当前位姿
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # 计算控制误差
        dx = target_x - current_x
        dy = target_y - current_y
        dx_next = target_x_next - target_x
        dy_next = target_y_next - target_y

        r = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        alpha = theta - current_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # 归一化角度
        
        phi_t = math.atan2(dy_next, dx_next) #preset trajectory
        beta = theta - phi_t
        beta = math.atan2(math.sin(beta), math.cos(beta))  # 归一化角度

        # 获取当前时间步长
        current_time = self.get_clock().now()
        self.previous_time = current_time

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

        # 检查是否接近目标点
        if r < 0.5:  # 如果距离足够近，切换到下一个目标点
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