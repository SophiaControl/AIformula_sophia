import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
import math
import numpy as np
from scipy.interpolate import splprep, splev


class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')

        # 订阅里程计
        self.create_subscription(Odometry, '/aiformula_sensing/gyro_odometry_publisher/odom', self.odom_callback, 10)
        
        # 发布速度指令
        self.velocity_publisher = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # 定义原始轨迹
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

        # # 使用 B 样条插值生成平滑轨迹，降低平滑参数使轨迹更接近原始点
        # points_x = [pt["x"] for pt in raw_trajectory]
        # points_y = [pt["y"] for pt in raw_trajectory]
        # points = np.array([points_x, points_y])
        # tck, u = splprep(points, s=0.5, k=3)  # 降低 s 参数，减少过度平滑
        # u_new = np.linspace(0, 1, num=200)
        # x_new, y_new = splev(u_new, tck)
        # self.trajectory = [{"x": x, "y": y} for x, y in zip(x_new, y_new)]

        self.current_target_index = 0
        self.current_pose = None
        self.timer = self.create_timer(0.1, self.follow_trajectory)
      
        self.previous_time = self.get_clock().now()
        self.last_valid_dt = 0.1  # 保存最后一个有效的时间增量

        # PID控制器参数及状态变量
        # 分离线速度和角速度的PID参数
        self.Kp_linear = 1.00
        self.Ki_linear = 0.01
        self.Kd_linear = 0.10

        self.Kp_angular = 1.50  # 增加比例增益以提高转向响应
        self.Ki_angular = 0.01
        self.Kd_angular = 0.20  # 增加微分增益以减少振荡

        # 积分限制
        self.max_linear_integral = 5.0
        self.max_angular_integral = 3.0

        self.linear_error_integral = 0.0
        self.previous_linear_error = 0.0

        self.angular_error_integral = 0.0
        self.previous_angular_error = 0.0

        # 添加异常处理相关参数
        self.max_error_jump = 5.0  # 最大允许的误差跳变
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        
        # 目标切换相关参数
        self.target_timeout = 10.0  # 10秒后强制切换目标
        self.current_target_time = 0.0

    def odom_callback(self, msg):
        try:
            self.current_pose = msg.pose.pose
        except Exception as e:
            self.get_logger().warn(f"Error in odometry callback: {e}")

    def follow_trajectory(self):
        if self.current_pose is None:
            self.get_logger().info("Waiting for odometry data...")
            return

        # 检查是否已经完成轨迹
        if self.current_target_index >= len(self.trajectory) - 1:
            self.stop_robot()
            return

        # 当前目标点
        target = self.trajectory[self.current_target_index]
        target_x = target["x"]
        target_y = target["y"]

        # 下一个目标点
        next_target = self.trajectory[self.current_target_index + 1]
        target_x_next = next_target["x"]
        target_y_next = next_target["y"]

        # 当前位姿
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        try:
            current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)
        except Exception as e:
            self.get_logger().warn(f"Error in quaternion conversion: {e}")
            return

        # 计算距离误差和期望航向
        dx = target_x - current_x
        dy = target_y - current_y
        r = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        alpha = theta - current_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # 归一化角度

        # 检测异常数据
        if hasattr(self, 'previous_r') and abs(r - self.previous_r) > self.max_error_jump:
            self.consecutive_failures += 1
            self.get_logger().warn(f"Detected possible odometry jump: {abs(r - self.previous_r)}")
            if self.consecutive_failures > self.max_consecutive_failures:
                self.get_logger().error("Too many consecutive failures, stopping robot")
                self.stop_robot()
                return
            self.previous_r = r
            return
        else:
            self.consecutive_failures = 0
            self.previous_r = r

        epsilon = 1e-6
        # 改进的目标切换判据
        D_x = target_x_next - target_x
        D_y = target_y_next - target_y
        D_norm = math.sqrt(D_x**2 + D_y**2) + epsilon

        v_x = current_x - target_x
        v_y = current_y - target_y
        proj = (v_x * D_x + v_y * D_y) / D_norm
        lateral_error = abs(-D_y * v_x + D_x * v_y) / D_norm

        # 改进目标切换逻辑
        switch_distance_threshold = 0.5
        lateral_threshold = 0.2

        # 更新目标超时计时器
        self.current_target_time += 0.1  # 假设此函数每0.1秒被调用一次

        # 切换条件：接近目标点，或者投影位置超过目标线段，或者超时
        if (r < switch_distance_threshold and lateral_error < lateral_threshold) or (proj >= D_norm) or (self.current_target_time > self.target_timeout):
            self.current_target_index += 1
            self.current_target_time = 0.0  # 重置目标超时计时器
            self.get_logger().info(f"Switching to target {self.current_target_index}")
            
            # 重置积分项以避免积分饱和
            self.linear_error_integral = 0.0
            self.angular_error_integral = 0.0
            
            if self.current_target_index >= len(self.trajectory) - 1:
                self.get_logger().info("Reached final target point")
                self.stop_robot()
                return

        # ------------------- PID控制部分 -------------------
        error_linear = r
        error_heading = alpha

        current_time = self.get_clock().now()
        dt = (current_time - self.previous_time).nanoseconds / 1e9
        
        # 处理dt为0的情况
        if dt < 0.001:  # 避免除以非常小的值
            dt = self.last_valid_dt
        else:
            self.last_valid_dt = dt  # 更新最后一个有效的dt
            
        self.previous_time = current_time

        # 计算线速度PID
        self.linear_error_integral += error_linear * dt
        # 积分限制防止饱和
        self.linear_error_integral = max(-self.max_linear_integral, min(self.linear_error_integral, self.max_linear_integral))
        
        derivative_linear = (error_linear - self.previous_linear_error) / dt
        v_command = self.Kp_linear * error_linear + self.Ki_linear * self.linear_error_integral + self.Kd_linear * derivative_linear
        self.previous_linear_error = error_linear

        # 计算角速度PID
        self.angular_error_integral += error_heading * dt
        # 积分限制防止饱和
        self.angular_error_integral = max(-self.max_angular_integral, min(self.angular_error_integral, self.max_angular_integral))
        
        derivative_angular = (error_heading - self.previous_angular_error) / dt
        omega_command = self.Kp_angular * error_heading + self.Ki_angular * self.angular_error_integral + self.Kd_angular * derivative_angular
        self.previous_angular_error = error_heading

        # 在接近目标点时降低速度的自适应速度控制
        max_linear_velocity = 2.0
        # 当距离目标点很近时，降低最小速度
        min_linear_velocity = 0.5
        max_angular_velocity = 1.57
        min_angular_velocity = -1.57

        v_command = max(min_linear_velocity, min(v_command, max_linear_velocity))
        omega_command = max(min_angular_velocity, min(omega_command, max_angular_velocity))
        # ------------------- PID控制部分结束 -------------------

        twist_msg = Twist()
        twist_msg.linear.x = v_command
        twist_msg.angular.z = omega_command
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