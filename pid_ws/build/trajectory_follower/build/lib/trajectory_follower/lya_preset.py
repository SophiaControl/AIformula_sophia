import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
import math
import numpy as np
from scipy.interpolate import splprep, splev


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
            {"x": 0.00, "y": 0.00},
            # {"x": 2.00, "y": 0.00},
            {"x": 4.00, "y": 0.00},
            # {"x": 6.00, "y": 0.00},
            {"x": 8.00, "y": 0.00},
            # {"x": 10.00, "y": 0.00},
            {"x": 12.00, "y": 0.00},
            # 第一段弧线
            {"x": 14.0, "y": 0.10},
            # {"x": 15.96, "y": 0.44},
            {"x": 17.87, "y": 1.03},
            # {"x": 19.74, "y": 1.78},
            {"x": 21.47, "y": 2.74},
            # {"x": 23.11, "y": 3.90},
            {"x": 24.60, "y": 5.20},
            # {"x": 25.94, "y": 6.73},
            {"x": 27.03, "y": 8.31},
            # {"x": 28.00, "y": 10.00},
            # 第二段直线
            {"x": 28.93, "y": 11.77},
            # {"x": 29.86, "y": 13.54},
            {"x": 30.79, "y": 15.32},
            # {"x": 31.71, "y": 17.09},
            {"x": 32.64, "y": 18.86},
            # {"x": 33.57, "y": 20.63},
            {"x": 34.50, "y": 22.41},
            # {"x": 35.43, "y": 24.18},
            {"x": 36.36, "y": 25.95},
            # {"x": 37.28, "y": 27.74},
            {"x": 38.21, "y": 29.49},
            # {"x": 39.14, "y": 31.26},
            {"x": 40.07, "y": 33.04},
            # {"x": 41.00, "y": 34.79},
            {"x": 41.92, "y": 36.59},
            # {"x": 42.85, "y": 38.35},
            {"x": 43.78, "y": 40.13},
            # {"x": 44.71, "y": 41.90},
            {"x": 45.64, "y": 43.67},
            # {"x": 46.57, "y": 45.49},
            {"x": 47.49, "y": 47.22},
            # {"x": 48.42, "y": 48.99},
            {"x": 49.35, "y": 50.76},
            # {"x": 50.00, "y": 52.00},
            # 第二段弧线
            {"x": 50.54, "y": 53.93},
            # {"x": 50.87, "y": 55.90},
            {"x": 51.08, "y": 57.89},
            # {"x": 51.06, "y": 59.89},
            {"x": 50.88, "y": 61.87},
            # {"x": 50.55, "y": 63.88},
            {"x": 50.04, "y": 65.75},
            # {"x": 49.35, "y": 67.66},
            {"x": 48.52, "y": 69.46},
            # {"x": 47.55, "y": 71.21},
            {"x": 46.43, "y": 72.84},
            # {"x": 45.08, "y": 74.47},
            {"x": 43.75, "y": 75.81},
            # {"x": 42.20, "y": 77.11},
            {"x": 40.46, "y": 78.35},
            # {"x": 38.85, "y": 79.27},
            {"x": 37.00, "y": 80.16},
            # {"x": 35.15, "y": 80.86},
            {"x": 33.31, "y": 81.32},
            # {"x": 31.23, "y": 81.76},
            {"x": 29.38, "y": 82.00},
            # {"x": 28.00, "y": 82.00},
            # 第三段直线
            {"x": 26.00, "y": 82.00},
            # {"x": 24.00, "y": 82.00},
            {"x": 22.00, "y": 82.00},
            # {"x": 20.00, "y": 82.00},
            {"x": 18.00, "y": 82.00},
            # {"x": 16.00, "y": 82.00},
            {"x": 14.00, "y": 82.00},
            # {"x": 12.00, "y": 82.00},
            {"x": 10.00, "y": 82.00},
            # {"x": 8.00, "y": 82.00},
            {"x": 6.00, "y": 82.00},
            # {"x": 4.00, "y": 82.00},
            {"x": 2.00, "y": 82.00},
            # {"x": 0.00, "y": 82.00},
            {"x": -2.00, "y": 82.00},
            # {"x": -4.00, "y": 82.00},
            {"x": -6.00, "y": 82.00},
            # {"x": -8.00, "y": 82.00},
            # 第三段弧线 
            {"x": -8.10, "y": 80.00},
            # {"x": -8.40, "y": 78.03},
            {"x": -8.89, "y": 76.09},
            # {"x": -9.58, "y": 74.21},
            {"x": -10.45, "y": 72.41},
            # {"x": -11.49, "y": 70.72},
            {"x": -12.70, "y": 69.12},
            # {"x": -14.07, "y": 67.65},
            {"x": -15.57, "y": 66.33},
            # {"x": -17.19, "y": 65.17},
            {"x": -18.93, "y": 64.18},
            # {"x": -20.75, "y": 63.36},
            {"x": -22.65, "y": 62.73},
            # {"x": -24.03, "y": 62.40},
            {"x": -26.59, "y": 62.05},
            # {"x": -28.00, "y": 62.00},
            # 第四段直线
            {"x": -28.00, "y": 60.00},
            # {"x": -28.00, "y": 58.00},
            {"x": -28.00, "y": 56.00},
            # {"x": -28.00, "y": 54.00},
            {"x": -28.00, "y": 52.00},
            # {"x": -28.00, "y": 50.00},
            {"x": -28.00, "y": 48.00},
            # {"x": -28.00, "y": 46.00},
            {"x": -28.00, "y": 44.00},
            # {"x": -28.00, "y": 42.00},
            {"x": -28.00, "y": 40.00},
            # {"x": -28.00, "y": 38.00},
            {"x": -28.00, "y": 36.00},
            # {"x": -28.00, "y": 34.00},
            {"x": -28.00, "y": 32.00},
            # {"x": -28.00, "y": 30.00},
            {"x": -28.00, "y": 28.00},
            # {"x": -28.00, "y": 26.00},
            {"x": -28.00, "y": 24.00},
            # {"x": -28.00, "y": 22.00},
            {"x": -28.00, "y": 20.00},
            # 第四段弧线
            {"x": -27.90, "y": 18.00},
            # {"x": -27.60, "y": 16.03},
            {"x": -27.11, "y": 14.09},
            # {"x": -26.42, "y": 12.21},
            {"x": -25.55, "y": 10.41},
            # {"x": -24.51, "y": 8.72},
            {"x": -23.30, "y": 7.12},
            # {"x": -21.93, "y": 5.65},
            {"x": -20.43, "y": 4.33},
            # {"x": -18.81, "y": 3.17},
            {"x": -17.07, "y": 2.18},
            # {"x": -15.25, "y": 1.36},
            {"x": -13.34, "y": 0.73},
            # {"x": -11.97, "y": 0.40},
            {"x": -9.41, "y": 0.05},
            # {"x": -8.00, "y": 0.00},
            # 第五段直线
            {"x": -6.00, "y": 0.00},
            # {"x": -4.00, "y": 0.00},
            {"x": -2.00, "y": 0.00},
        ]

        # # 使用 B 样条插值生成平滑轨迹
        # points_x = [pt["x"] for pt in raw_trajectory]
        # points_y = [pt["y"] for pt in raw_trajectory]
        # points = np.array([points_x, points_y])
        # # 参数 s 可调整平滑程度，k 表示 B 样条的阶数（这里采用 3 阶样条）
        # tck, u = splprep(points, s=2, k=3)
        # # 生成较多插值点，使得轨迹更平滑
        # u_new = np.linspace(0, 1, num=200)
        # x_new, y_new = splev(u_new, tck)
        # # 构造平滑轨迹，作为机器人运动的目标轨迹
        # self.trajectory = [{"x": x, "y": y} for x, y in zip(x_new, y_new)]

        self.current_target_index = 0
        self.current_pose = None
        self.timer = self.create_timer(0.1, self.follow_trajectory)

        # 初始化LYA控制器
        self.lya = LYAController(omega_t=0.01, v_t=1.5, lambda_v=0.075, lambda_a=0.015, k1=0.8, k2=50)
      
        # 初始化前一时间点为当前时间，避免初始化阶段的错误处理问题
        self.previous_time = self.get_clock().now()
        # 初始化时间差值
        self.dt = 0.1  # 默认值与timer周期一致

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

        # 确保不会访问超出轨迹范围的元素
        if self.current_target_index + 1 < len(self.trajectory):
            next_target = self.trajectory[self.current_target_index + 1]
            target_x_next = next_target["x"]
            target_y_next = next_target["y"]
        else:
            # 如果是最后一个点，使用当前点作为下一个点（保持方向）
            target_x_next = target_x
            target_y_next = target_y

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

        # 对 sin(alpha) 做防护：若 sin(alpha) 接近 0，则设置一个小阈值 epsilon
        epsilon = 1e-6
        safe_sin_alpha = math.sin(alpha)
        if abs(safe_sin_alpha) < epsilon:
            safe_sin_alpha = epsilon if safe_sin_alpha >= 0 else -epsilon

        # 改进的目标切换判据部分
        # 计算目标段向量 D = (D_x, D_y)
        D_x = target_x_next - target_x
        D_y = target_y_next - target_y
        D_norm = math.sqrt(D_x**2 + D_y**2) + epsilon  # 防止除零

        # 计算机器人当前位置相对于当前目标点的向量 v = (v_x, v_y)
        v_x = current_x - target_x
        v_y = current_y - target_y

        # 投影长度 proj = (v·D)/|D|
        proj = (v_x * D_x + v_y * D_y) / D_norm

        # 横向偏差：机器人到直线 (target -> next_target) 的距离
        lateral_error = abs(-D_y * v_x + D_x * v_y) / D_norm

        # 定义距离和横向误差阈值（根据实际情况调整）
        switch_distance_threshold = 0.5  # 距离目标点的阈值
        lateral_threshold = 0.2          # 横向偏差阈值

        # 判据：若距离小且横向偏差小，或投影距离超过目标段长度，则切换到下一个目标点
        if (r < switch_distance_threshold and lateral_error < lateral_threshold) or (proj >= D_norm):
            self.current_target_index += 1
            # 若轨迹已完成则停止机器人（统一使用相同的条件判断）
            if self.current_target_index >= len(self.trajectory) - 1:
                self.stop_robot()
                return

        # 获取当前时间并计算时间步长
        current_time = self.get_clock().now()
        dt = (current_time - self.previous_time).nanoseconds / 1e9  # 转换为秒
        if dt > 0:  # 确保时间步长有效
            self.dt = dt
        self.previous_time = current_time

        # 通过LYA控制器计算速度指令，使用safe_sin_alpha替代直接的math.sin(alpha)
        v = ((self.lya.v_t) * math.cos(beta) + self.lya.lambda_v * r) * math.cos(alpha)

        # 使用safe_sin_alpha来避免除零错误
        k1_over_sin_alpha = self.lya.k1 / safe_sin_alpha

        omega = (self.lya.lambda_a * safe_sin_alpha + k1_over_sin_alpha * 
        ((safe_sin_alpha / (self.lya.k1 * r)) + (math.sin(beta) / (self.lya.k2 * r))) * 
        ((math.sin(2 * alpha) * math.cos(beta) / 2) - math.sin(beta)) * self.lya.v_t 
        - (self.lya.omega_t * math.sin(beta) / self.lya.k2) * k1_over_sin_alpha 
        + k1_over_sin_alpha * self.lya.lambda_v * (math.sin(2 * alpha) / 2) * 
        ((safe_sin_alpha / self.lya.k1) + (math.sin(beta) / self.lya.k2)))
        
        # 对 v 和 omega 进行饱和限制，避免输出过大
        max_linear_velocity = 1.75   
        min_linear_velocity = 0.5  
        max_angular_velocity = 0.4  
        min_angular_velocity = 0.4

        v = max(min_linear_velocity, min(v, max_linear_velocity))
        omega = max(min_angular_velocity, min(omega, max_angular_velocity))

        # 发布速度指令
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = omega
        self.velocity_publisher.publish(twist_msg)

    def quaternion_to_yaw(self, orientation):
        qx, qy, qz, qw = (orientation.x, orientation.y, orientation.z, orientation.w)
        # 验证四元数是否有效
        norm = qx*qx + qy*qy + qz*qz + qw*qw
        if abs(norm - 1.0) > 0.1:  # 允许一定的误差
            # self.get_logger().warn(f"Invalid quaternion detected with norm: {norm}")
            # 归一化四元数
            if norm > 0:
                qx /= math.sqrt(norm)
                qy /= math.sqrt(norm)
                qz /= math.sqrt(norm)
                qw /= math.sqrt(norm)
            else:
                return 0.0  # 返回默认值
                
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def stop_robot(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.velocity_publisher.publish(twist_msg)
        self.get_logger().info("Stopping robot.")
        
def main(args=None):
    rclpy.init(args=args)
    try:
        follower = TrajectoryFollower()
        rclpy.spin(follower)
    except Exception as e:
        print(f"Error in trajectory follower: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
