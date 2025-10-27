import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
import math
import numpy as np
# from scipy.interpolate import splprep, splev  # B样条相关代码已移除

class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')

        # 订阅里程计
        self.create_subscription(Odometry, '/aiformula_sensing/gyro_odometry_publisher/odom', self.odom_callback, 10)
        
        # 发布速度指令
        self.velocity_publisher = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # 使用原始轨迹跟踪
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

        # 计算每个轨迹点的曲率，用于后续速度规划
        self.calculate_trajectory_curvature()

        self.current_target_index = 0
        self.current_pose = None
        self.last_pose = None
        self.current_speed = 0.0
        self.trajectory_completed = False
        self.min_lookahead_distance = 0.5
        self.max_lookahead_distance = 2.5
        self.speed_factor = 0.5  # 控制前视距离随速度变化的系数

        # 创建定时器，频率根据实际需要调整（20Hz控制频率）
        self.control_frequency = 0.05
        self.timer = self.create_timer(self.control_frequency, self.follow_trajectory)

        # 设置车辆动力学约束
        self.max_linear_velocity = 2.0
        self.min_linear_velocity = 0.5
        self.max_angular_velocity = 1.57
        self.min_angular_velocity = -1.57
        self.max_linear_acceleration = 1.0  # 最大线加速度 m/s^2
        self.max_angular_acceleration = 1.0  # 最大角加速度 rad/s^2
        
        # 记录上一个控制命令以实现平滑过渡
        self.last_cmd_vel = Twist()
        
        # 设置轨迹完成判定阈值
        self.goal_threshold = 0.5  # 到终点的距离阈值
        self.min_progress_distance = 0.1  # 最小前进距离
        
        # 初始化累计前进距离
        self.accumulated_distance = 0.0
        
        self.get_logger().info('TrajectoryFollower initialized')

    def calculate_trajectory_curvature(self):
        """计算轨迹上每个点的曲率，用于速度规划"""
        for i in range(1, len(self.trajectory) - 1):
            p1 = self.trajectory[i-1]
            p2 = self.trajectory[i]
            p3 = self.trajectory[i+1]
            
            x1, y1 = p1["x"], p1["y"]
            x2, y2 = p2["x"], p2["y"]
            x3, y3 = p3["x"], p3["y"]
            
            area = ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / 2.0
            if abs(area) < 1e-6:
                self.trajectory[i]["curvature"] = 0.0
                continue
                
            a = math.hypot(x1 - x2, y1 - y2)
            b = math.hypot(x2 - x3, y2 - y3)
            c = math.hypot(x3 - x1, y3 - y1)
            
            try:
                self.trajectory[i]["curvature"] = 4.0 * abs(area) / (a * b * c)
            except ZeroDivisionError:
                self.trajectory[i]["curvature"] = 0.0
        
        self.trajectory[0]["curvature"] = self.trajectory[1]["curvature"]
        self.trajectory[-1]["curvature"] = self.trajectory[-2]["curvature"]

    def odom_callback(self, msg):
        self.last_pose = self.current_pose
        self.current_pose = msg.pose.pose
        
        if self.last_pose is not None:
            dx = self.current_pose.position.x - self.last_pose.position.x
            dy = self.current_pose.position.y - self.last_pose.position.y
            distance = math.hypot(dx, dy)
            self.current_speed = distance / self.control_frequency
            self.accumulated_distance += distance

    def find_closest_point(self, current_x, current_y):
        min_dist = float('inf')
        min_idx = 0
        
        for i in range(self.current_target_index, len(self.trajectory)):
            point = self.trajectory[i]
            dist = math.hypot(point["x"] - current_x, point["y"] - current_y)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx, min_dist

    def get_dynamic_lookahead_distance(self):
        return max(
            self.min_lookahead_distance,
            min(self.max_lookahead_distance, self.min_lookahead_distance + self.speed_factor * self.current_speed)
        )

    def follow_trajectory(self):
        if self.current_pose is None:
            self.get_logger().warning("No odometry data received yet.")
            return
            
        if self.trajectory_completed:
            self.stop_robot()
            return

        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        closest_idx, closest_dist = self.find_closest_point(current_x, current_y)
        
        if closest_idx > self.current_target_index:
            self.current_target_index = closest_idx
        
        if closest_idx >= len(self.trajectory) - 3:
            final_point = self.trajectory[-1]
            if math.hypot(final_point["x"] - current_x, final_point["y"] - current_y) < self.goal_threshold:
                self.trajectory_completed = True
                self.stop_robot()
                return
        
        lookahead_distance = self.get_dynamic_lookahead_distance()
        
        lookahead_point = None
        lookahead_index = self.current_target_index
        
        for i in range(self.current_target_index, len(self.trajectory)):
            point = self.trajectory[i]
            dx = point["x"] - current_x
            dy = point["y"] - current_y
            distance = math.hypot(dx, dy)
            if distance >= lookahead_distance:
                lookahead_point = point
                lookahead_index = i
                break
                
        if lookahead_point is None:
            lookahead_point = self.trajectory[-1]
            lookahead_index = len(self.trajectory) - 1
            if math.hypot(lookahead_point["x"] - current_x, lookahead_point["y"] - current_y) < lookahead_distance:
                lookahead_distance = math.hypot(lookahead_point["x"] - current_x, lookahead_point["y"] - current_y)

        angle_to_target = math.atan2(lookahead_point["y"] - current_y, lookahead_point["x"] - current_x)
        alpha = angle_to_target - current_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        if abs(lookahead_distance) < 0.1:
            curvature = 0.0
        else:
            curvature = (2 * math.sin(alpha)) / lookahead_distance

        path_curvature = 0.0
        count = 0
        for i in range(lookahead_index, min(lookahead_index + 5, len(self.trajectory))):
            if "curvature" in self.trajectory[i]:
                path_curvature += self.trajectory[i]["curvature"]
                count += 1
        
        if count > 0:
            path_curvature /= count
        
        curvature_factor = 1.0 / (1.0 + 5.0 * path_curvature)
        target_v = self.max_linear_velocity * curvature_factor
        
        last_v = self.last_cmd_vel.linear.x
        v_change = target_v - last_v
        max_v_change = self.max_linear_acceleration * self.control_frequency
        
        if v_change > max_v_change:
            v = last_v + max_v_change
        elif v_change < -max_v_change:
            v = last_v - max_v_change
        else:
            v = target_v
            
        v = max(self.min_linear_velocity, min(v, self.max_linear_velocity))
        
        target_omega = v * curvature
        
        last_omega = self.last_cmd_vel.angular.z
        omega_change = target_omega - last_omega
        max_omega_change = self.max_angular_acceleration * self.control_frequency
        
        if omega_change > max_omega_change:
            omega = last_omega + max_omega_change
        elif omega_change < -max_omega_change:
            omega = last_omega - max_omega_change
        else:
            omega = target_omega
            
        omega = max(self.min_angular_velocity, min(omega, self.max_angular_velocity))

        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = omega
        self.velocity_publisher.publish(twist_msg)
        
        self.last_cmd_vel = twist_msg

    def quaternion_to_yaw(self, orientation):
        qx, qy, qz, qw = (orientation.x, orientation.y, orientation.z, orientation.w)
        norm = qx*qx + qy*qy + qz*qz + qw*qw
        if abs(norm - 1.0) > 0.1:
            if norm > 0:
                qx /= math.sqrt(norm)
                qy /= math.sqrt(norm)
                qz /= math.sqrt(norm)
                qw /= math.sqrt(norm)
            else:
                return 0.0
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
