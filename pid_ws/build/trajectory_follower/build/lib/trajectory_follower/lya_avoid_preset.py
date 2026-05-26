import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import math
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.interpolate import splprep, splev

class LYAController:
    def __init__(self, omega_t, v_t, lambda_v, lambda_a, k1, k2):
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
        # 订阅图像数据，用于红色障碍检测
        self.create_subscription(Image, '/aiformula_sensing/zed_node/left_image/undistorted', self.image_callback, 10)
        self.bridge = CvBridge()

        # 发布速度指令
        self.velocity_publisher = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # 定义原始离散轨迹点（原始规划路径）
        self.raw_trajectory = [
            {"x": 0.00, "y": 0.00},  
            {"x": 2.00, "y": 0.00},  
            {"x": 4.00, "y": 0.00},   
            {"x": 6.00, "y": 0.00},
            {"x": 8.00, "y": 0.00},
            {"x": 10.00, "y": 0.00},
            {"x": 12.00, "y": 0.00},
            # 第一段弧线
            {"x": 14.0, "y": -0.10},
            {"x": 15.96, "y": -0.44},
            {"x": 17.87, "y": -1.03},
            {"x": 19.74, "y": -1.78},
            {"x": 21.47, "y": -2.74},
            {"x": 23.11, "y": -3.90},
            {"x": 24.60, "y": -5.20},
            {"x": 25.94, "y": -6.73},
            {"x": 27.03, "y": -8.31},
            {"x": 28.00, "y": -10.00},
            # 第二段直线
            {"x": 28.93, "y": -11.77},
            {"x": 29.86, "y": -13.54},
            {"x": 30.79, "y": -15.32},
            {"x": 31.71, "y": -17.09},
            {"x": 32.64, "y": -18.86},
            {"x": 33.57, "y": -20.63},
            {"x": 34.50, "y": -22.41},
            {"x": 35.43, "y": -24.18},
            {"x": 36.36, "y": -25.95},
            {"x": 37.28, "y": -27.74},
            {"x": 38.21, "y": -29.49},
            {"x": 39.14, "y": -31.26},
            {"x": 40.07, "y": -33.04},
            {"x": 41.00, "y": -34.79},
            {"x": 41.92, "y": -36.59},
            {"x": 42.85, "y": -38.35},
            {"x": 43.78, "y": -40.13},
            {"x": 44.71, "y": -41.90},
            {"x": 45.64, "y": -43.67},
            {"x": 46.57, "y": -45.49},
            {"x": 47.49, "y": -47.22},
            {"x": 48.42, "y": -48.99},
            {"x": 49.35, "y": -50.76},
            {"x": 50.00, "y": -52.00},
            # 第二段弧线
            {"x": 50.54, "y": -53.93},
            {"x": 50.87, "y": -55.90},
            {"x": 51.08, "y": -57.89},
            {"x": 51.06, "y": -59.89},
            {"x": 50.88, "y": -61.87},
            {"x": 50.55, "y": -63.88},
            {"x": 50.04, "y": -65.75},
            {"x": 49.35, "y": -67.66},
            {"x": 48.52, "y": -69.46},
            {"x": 47.55, "y": -71.21},
            {"x": 46.43, "y": -72.84},
            {"x": 45.08, "y": -74.47},
            {"x": 43.75, "y": -75.81},
            {"x": 42.20, "y": -77.11},
            {"x": 40.46, "y": -78.35},
            {"x": 38.85, "y": -79.27},
            {"x": 37.00, "y": -80.16},
            {"x": 35.15, "y": -80.86},
            {"x": 33.31, "y": -81.32},
            {"x": 31.23, "y": -81.76},
            {"x": 29.38, "y": -82.00},
            {"x": 28.00, "y": -82.00},
            # 第三段直线
            {"x": 26.00, "y": -82.00},
            {"x": 24.00, "y": -82.00},
            {"x": 22.00, "y": -82.00},
            {"x": 20.00, "y": -82.00},
            {"x": 18.00, "y": -82.00},
            {"x": 16.00, "y": -82.00},
            {"x": 14.00, "y": -82.00},
            {"x": 12.00, "y": -82.00},
            {"x": 10.00, "y": -82.00},
            {"x": 8.00, "y": -82.00},
            {"x": 6.00, "y": -82.00},
            {"x": 4.00, "y": -82.00},
            {"x": 2.00, "y": -82.00},
            {"x": 0.00, "y": -82.00},
            {"x": -2.00, "y": -82.00},
            {"x": -4.00, "y": -82.00},
            {"x": -6.00, "y": -82.00},
            {"x": -8.00, "y": -82.00},
            # 第三段弧线 
            {"x": -8.10, "y": -80.00},
            {"x": -8.40, "y": -78.03},
            {"x": -8.89, "y": -76.09},
            {"x": -9.58, "y": -74.21},
            {"x": -10.45, "y": -72.41},
            {"x": -11.49, "y": -70.72},
            {"x": -12.70, "y": -69.12},
            {"x": -14.07, "y": -67.65},
            {"x": -15.57, "y": -66.33},
            {"x": -17.19, "y": -65.17},
            {"x": -18.93, "y": -64.18},
            {"x": -20.75, "y": -63.36},
            {"x": -22.65, "y": -62.73},
            {"x": -24.03, "y": -62.40},
            {"x": -26.59, "y": -62.05},
            {"x": -28.00, "y": -62.00},
            # 第四段直线
            {"x": -28.00, "y": -60.00},
            {"x": -28.00, "y": -58.00},
            {"x": -28.00, "y": -56.00},
            {"x": -28.00, "y": -54.00},
            {"x": -28.00, "y": -52.00},
            {"x": -28.00, "y": -50.00},
            {"x": -28.00, "y": -48.00},
            {"x": -28.00, "y": -46.00},
            {"x": -28.00, "y": -44.00},
            {"x": -28.00, "y": -42.00},
            {"x": -28.00, "y": -40.00},
            {"x": -28.00, "y": -38.00},
            {"x": -28.00, "y": -36.00},
            {"x": -28.00, "y": -34.00},
            {"x": -28.00, "y": -32.00},
            {"x": -28.00, "y": -30.00},
            {"x": -28.00, "y": -28.00},
            {"x": -28.00, "y": -26.00},
            {"x": -28.00, "y": -24.00},
            {"x": -28.00, "y": -22.00},
            {"x": -28.00, "y": -20.00},
            # 第四段弧线
            {"x": -27.90, "y": -18.00},
            {"x": -27.60, "y": -16.03},
            {"x": -27.11, "y": -14.09},
            {"x": -26.42, "y": -12.21},
            {"x": -25.55, "y": -10.41},
            {"x": -24.51, "y": -8.72},
            {"x": -23.30, "y": -7.12},
            {"x": -21.93, "y": -5.65},
            {"x": -20.43, "y": -4.33},
            {"x": -18.81, "y": -3.17},
            {"x": -17.07, "y": -2.18},
            {"x": -15.25, "y": -1.36},
            {"x": -13.34, "y": -0.73},
            {"x": -11.97, "y": -0.40},
            {"x": -9.41, "y": -0.05},
            {"x": -8.00, "y": 0.00},
            # 第五段直线
            {"x": -6.00, "y": 0.00},  
            {"x": -4.00, "y": 0.00},  
            {"x": -2.00, "y": 0.00},  
        ]
        # 初始生成原始轨迹的平滑版本
        self.update_trajectory(self.raw_trajectory)
        # 保存原始轨迹（平滑后的）以便后续回归使用
        self.original_smooth_trajectory = self.trajectory.copy()
        
        self.current_target_index = 0
        self.current_pose = None
        self.timer = self.create_timer(0.1, self.follow_trajectory)

        # 初始化 LYA 控制器
        self.lya = LYAController(omega_t=0.01, v_t=1.5, lambda_v=0.75, lambda_a=2.0, k1=0.8, k2=25)
      
        self.previous_time = self.get_clock().now()
        # 避障状态管理：模式可为 "normal", "avoidance", "returning"
        self.mode = "normal"
        
        # 添加障碍物检测和模式切换的稳定性参数
        self.obstacle_detection_counter = 0
        self.obstacle_clear_counter = 0
        self.obstacle_detection_threshold = 3  # 连续检测到障碍物的次数阈值
        self.obstacle_clear_threshold = 5      # 连续未检测到障碍物的次数阈值
        
        # 动态调整参数
        self.switch_distance_threshold = 0.5
        self.lateral_threshold = 0.2
        self.max_linear_velocity = 3.0
        self.min_linear_velocity = 0.5
        self.max_angular_velocity = 1.57
        self.min_angular_velocity = -1.57
        self.current_velocity = 0.0           # 记录当前速度，用于平滑控制
        self.current_angular_velocity = 0.0   # 记录当前角速度
        self.velocity_smoothing_factor = 0.3  # 速度平滑因子

    def update_trajectory(self, trajectory_points, smoothing_factor=2.0, spline_degree=3):
        """
        利用 B 样条插值生成平滑轨迹，并更新 self.trajectory
        添加参数以动态调整平滑程度
        """
        if len(trajectory_points) < spline_degree + 1:
            # 如果点数不足以生成所需阶数的样条，则降低阶数
            spline_degree = min(spline_degree, len(trajectory_points) - 1)
            if spline_degree < 1:
                self.get_logger().warn("Too few points for spline interpolation, using linear interpolation.")
                self.trajectory = trajectory_points
                return
        
        points_x = [pt["x"] for pt in trajectory_points]
        points_y = [pt["y"] for pt in trajectory_points]
        points = np.array([points_x, points_y])
        
        try:
            # 参数 s 控制平滑程度，k 为样条阶数
            tck, u = splprep(points, s=smoothing_factor, k=spline_degree)
            # 生成更多插值点以获得更平滑的曲线
            u_new = np.linspace(0, 1, num=200)
            x_new, y_new = splev(u_new, tck)
            self.trajectory = [{"x": x, "y": y} for x, y in zip(x_new, y_new)]
            self.current_target_index = 0
            self.get_logger().info("Trajectory updated with smoothing factor: " + str(smoothing_factor))
        except Exception as e:
            self.get_logger().error("Failed to update trajectory: " + str(e))
            # 如果插值失败，直接使用原始点
            self.trajectory = trajectory_points
            self.current_target_index = 0

    def odom_callback(self, msg):
        """
        处理里程计数据回调
        """
        if self.current_pose is None:
            self.get_logger().info("First odometry received.")
        self.current_pose = msg.pose.pose
        
        # 计算当前速度
        if hasattr(self, 'previous_odom_msg'):
            dt = (self.get_clock().now() - self.previous_odom_time).nanoseconds / 1e9
            if dt > 0:
                # 使用里程计消息来计算实际速度
                dx = self.current_pose.position.x - self.previous_odom_msg.pose.pose.position.x
                dy = self.current_pose.position.y - self.previous_odom_msg.pose.pose.position.y
                self.current_velocity = math.sqrt(dx**2 + dy**2) / dt
                
                # 计算角速度变化
                current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)
                previous_yaw = self.quaternion_to_yaw(self.previous_odom_msg.pose.pose.orientation)
                yaw_diff = current_yaw - previous_yaw
                # 归一化角度差
                yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))
                self.current_angular_velocity = yaw_diff / dt
        
        self.previous_odom_msg = msg
        self.previous_odom_time = self.get_clock().now()

    def image_callback(self, msg):
        """
        图像回调函数，检测红色障碍，添加连续检测计数器以提高稳定性
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error("Failed to convert image: " + str(e))
            return

        # 转换为 HSV 色彩空间
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # 定义红色范围（覆盖 HSV 中红色的两端）
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # 添加形态学操作以去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 计算红色像素数量
        red_pixel_count = cv2.countNonZero(mask)
        red_threshold = 5000  # 阈值，根据实际场景调整

        if red_pixel_count > red_threshold:
            # 检测到障碍物，增加计数器
            self.obstacle_detection_counter += 1
            self.obstacle_clear_counter = 0
            
            # 只有当连续检测到障碍物达到阈值时才切换模式
            if self.obstacle_detection_counter >= self.obstacle_detection_threshold and self.mode in ["normal", "returning"]:
                self.get_logger().info(f"Red obstacle detected for {self.obstacle_detection_counter} consecutive frames! Switching to avoidance mode.")
                self.replan_avoidance_path()
                self.mode = "avoidance"
        else:
            # 未检测到障碍物，重置检测计数器，增加清除计数器
            self.obstacle_detection_counter = 0
            self.obstacle_clear_counter += 1
            
            # 只有当连续未检测到障碍物达到阈值时才切换模式
            if self.obstacle_clear_counter >= self.obstacle_clear_threshold and self.mode == "avoidance":
                self.get_logger().info(f"Obstacle cleared for {self.obstacle_clear_counter} consecutive frames. Planning return path.")
                self.replan_return_path()
                self.mode = "returning"

    def replan_avoidance_path(self):
        """
        改进的避障路径规划：根据当前位姿和障碍物位置计算避障偏移量
        """
        if self.current_pose is None:
            self.get_logger().warn("No odometry data available for planning avoidance path.")
            return
            
        offset = 2.0  # 基础避障偏移量
        
        # 根据当前位置与轨迹关系微调偏移方向
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        # 找到轨迹上最近点
        min_dist = float('inf')
        nearest_index = 0
        for i, pt in enumerate(self.trajectory):
            dist = math.hypot(current_x - pt["x"], current_y - pt["y"])
            if dist < min_dist:
                min_dist = dist
                nearest_index = i
        
        # 如果最近点不是轨迹的末端，则尝试根据轨迹方向调整避障偏移
        if nearest_index < len(self.trajectory) - 1:
            next_point = self.trajectory[nearest_index + 1]
            prev_point = self.trajectory[max(0, nearest_index - 1)]
            
            # 计算轨迹前进方向
            dx = next_point["x"] - prev_point["x"]
            dy = next_point["y"] - prev_point["y"]
            
            # 选择一个垂直于轨迹方向的偏移
            if abs(dx) > abs(dy):  # 轨迹主要沿 x 方向
                offset_direction = 1.0 if current_y > self.trajectory[nearest_index]["y"] else -1.0
                new_raw_trajectory = []
                for pt in self.raw_trajectory:
                    new_pt = {"x": pt["x"], "y": pt["y"] + offset * offset_direction}
                    new_raw_trajectory.append(new_pt)
            else:  # 轨迹主要沿 y 方向
                offset_direction = 1.0 if current_x > self.trajectory[nearest_index]["x"] else -1.0
                new_raw_trajectory = []
                for pt in self.raw_trajectory:
                    new_pt = {"x": pt["x"] + offset * offset_direction, "y": pt["y"]}
                    new_raw_trajectory.append(new_pt)
        else:
            # 默认沿 y 方向偏移
            new_raw_trajectory = []
            for pt in self.raw_trajectory:
                new_pt = {"x": pt["x"], "y": pt["y"] + offset}
                new_raw_trajectory.append(new_pt)
        
        # 更新轨迹并应用较小的平滑因子以保持轨迹形状
        self.update_trajectory(new_raw_trajectory, smoothing_factor=1.0)

    def replan_return_path(self):
        """
        改进的返回路径规划：考虑当前速度和朝向，确保平滑过渡
        """
        if self.current_pose is None:
            self.get_logger().warn("No odometry data available for planning return path.")
            return

        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # 找到原始轨迹中距离当前位姿最近的点，并考虑前进方向
        min_dist = float('inf')
        nearest_index = 0
        look_ahead_distance = max(3.0, self.current_velocity * 2.0)  # 基于当前速度的前视距离
        
        for i, pt in enumerate(self.original_smooth_trajectory):
            # 计算点到当前位置的距离
            dist = math.hypot(current_x - pt["x"], current_y - pt["y"])
            
            # 考虑机器人朝向，优先选择在前方的点
            dx = pt["x"] - current_x
            dy = pt["y"] - current_y
            heading_to_point = math.atan2(dy, dx)
            heading_diff = abs(heading_to_point - current_yaw)
            heading_diff = min(heading_diff, 2*math.pi - heading_diff)  # 归一化到 [0, π]
            
            # 结合距离和朝向因素
            weighted_dist = dist * (1.0 + heading_diff / math.pi)
            
            # 更新最近点
            if weighted_dist < min_dist:
                min_dist = weighted_dist
                nearest_index = i
        
        # 找到更适合的接入点：在最近点前方找到一个更好的接入点
        target_index = nearest_index
        for i in range(nearest_index, min(len(self.original_smooth_trajectory), nearest_index + 20)):
            pt = self.original_smooth_trajectory[i]
            dist = math.hypot(current_x - pt["x"], current_y - pt["y"])
            if dist > look_ahead_distance:
                target_index = i
                break
        
        # 生成平滑过渡段轨迹
        num_transition_points = max(20, int(min_dist * 5))  # 根据距离动态调整过渡点数量
        transition_trajectory = []
        
        # 添加当前位置作为起点
        transition_trajectory.append({"x": current_x, "y": current_y})
        
        # 计算前进方向的单位向量
        forward_x = math.cos(current_yaw)
        forward_y = math.sin(current_yaw)
        
        # 添加当前朝向前方的几个点，形成平滑的初始过渡
        for i in range(1, 5):
            t = i / 5.0
            dist_factor = self.current_velocity * t
            x = current_x + forward_x * dist_factor
            y = current_y + forward_y * dist_factor
            transition_trajectory.append({"x": x, "y": y})
        
        # 添加到目标点的平滑过渡
        target_pt = self.original_smooth_trajectory[target_index]
        for t in np.linspace(0.1, 1.0, num_transition_points - 5):
            # 使用贝塞尔曲线参数计算平滑过渡
            last_transition_pt = transition_trajectory[-1]
            p0 = np.array([last_transition_pt["x"], last_transition_pt["y"]])
            p1 = np.array([current_x + forward_x * min_dist * 0.5, current_y + forward_y * min_dist * 0.5])
            p2 = np.array([target_pt["x"], target_pt["y"]])
            
            # 二次贝塞尔曲线
            point = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
            transition_trajectory.append({"x": point[0], "y": point[1]})

        # 拼接过渡段和原始轨迹剩余部分
        new_trajectory_points = transition_trajectory + self.original_smooth_trajectory[target_index:]
        
        # 更新轨迹，使用较小的平滑因子确保轨迹形状保持
        self.update_trajectory(new_trajectory_points, smoothing_factor=1.5)
        
        # 不要立即切换到 normal 模式，等待稳定过渡
        self.get_logger().info("Return path planned. Entering returning mode.")

    def follow_trajectory(self):
        """
        改进的轨迹跟踪控制函数，增加错误处理和平滑控制
        """
        if self.current_pose is None:
            self.get_logger().warn("No odometry data available for trajectory following.")
            return  # 等待里程计数据

        if self.current_target_index >= len(self.trajectory) - 1:
            self.stop_robot()
            return
            
        # 检查轨迹是否有效
        if len(self.trajectory) < 2:
            self.get_logger().error("Invalid trajectory with too few points.")
            self.stop_robot()
            return
            
        # 获取当前目标点和下一个目标点
        target = self.trajectory[self.current_target_index]
        target_x = target["x"]
        target_y = target["y"]

        # 如果存在下一个目标点，则计算目标段向量
        if self.current_target_index + 1 < len(self.trajectory):
            next_target = self.trajectory[self.current_target_index + 1]
            target_x_next = next_target["x"]
            target_y_next = next_target["y"]
        else:
            target_x_next, target_y_next = target_x, target_y

        # 获取当前位姿
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # 计算控制误差
        dx = target_x - current_x
        dy = target_y - current_y
        r = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        alpha = theta - current_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # 归一化角度

        # 当前目标段方向
        phi_t = math.atan2(target_y_next - target_y, target_x_next - target_x)
        beta = theta - phi_t
        beta = math.atan2(math.sin(beta), math.cos(beta))  # 归一化角度

        # 对 sin(alpha) 做防护，避免数值不稳定
        epsilon = 1e-6
        # 不直接替换 sin(alpha)，而是在其接近零时调整控制律
        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)
        sin_beta = math.sin(beta)
        cos_beta = math.cos(beta)

        # 改进的目标切换判据：综合当前距离、横向误差和前进速度
        D_x = target_x_next - target_x
        D_y = target_y_next - target_y
        D_norm = math.sqrt(D_x**2 + D_y**2) + epsilon
        v_x = current_x - target_x
        v_y = current_y - target_y
        proj = (v_x * D_x + v_y * D_y) / D_norm
        lateral_error = abs(-D_y * v_x + D_x * v_y) / D_norm

        # 动态调整切换阈值，根据速度和轨迹曲率
        dynamic_switch_distance = self.switch_distance_threshold * (1.0 + 0.5 * self.current_velocity)
        dynamic_lateral_threshold = self.lateral_threshold * (1.0 + 0.5 * self.current_velocity)
            
        if (r < dynamic_switch_distance and lateral_error < dynamic_lateral_threshold) or (proj >= D_norm):
            self.current_target_index += 1
            self.get_logger().debug(f"Switching to next target point: {self.current_target_index}")
            if self.current_target_index >= len(self.trajectory):
                self.stop_robot()
                return

        # 改进的控制律计算
        try:
            # 基础线速度控制
            v = (self.lya.v_t * cos_beta + self.lya.lambda_v * dx) * cos_alpha
                
            # 角速度控制，处理 sin(alpha) 接近零的情况
            if abs(sin_alpha) < epsilon:
                # alpha 接近零时使用简化的控制律
                omega = self.lya.lambda_a * sin_alpha + self.lya.omega_t * sin_beta / self.lya.k2
            else:
                # 正常情况下的完整控制律
                omega = (self.lya.lambda_a * sin_alpha +
                        (self.lya.k1 / sin_alpha) *
                        ((sin_alpha / (self.lya.k1 * r)) + (sin_beta / (self.lya.k2 * r))) *
                        ((math.sin(2 * alpha) * cos_beta / 2) - sin_beta) * self.lya.v_t -
                        (self.lya.omega_t * sin_beta / self.lya.k2) * (self.lya.k1 / sin_alpha) +
                        (self.lya.k1 / sin_alpha) * self.lya.lambda_v * (math.sin(2 * alpha) / 2) *
                        ((sin_alpha / self.lya.k1) + (sin_beta / self.lya.k2)))
        except Exception as e:
            # 捕获任何数值计算异常，回退到简单控制方案
            self.get_logger().warn(f"Control law calculation error: {e}, falling back to simple control.")
            v = self.lya.v_t * cos_alpha
            omega = self.lya.lambda_a * sin_alpha
                    
        # 根据当前模式调整速度
        if self.mode == "avoidance":
            # 避障时降低速度
            v = v * 0.7
        elif self.mode == "returning":
            # 返回过程中平稳过渡
            v = v * 0.8
                
        # 限制输出范围
        v = max(self.min_linear_velocity, min(v, self.max_linear_velocity))
        omega = max(self.min_angular_velocity, min(omega, self.max_angular_velocity))
            
        # 平滑控制输出，避免剧烈变化
        smoothed_v = self.current_velocity + self.velocity_smoothing_factor * (v - self.current_velocity)
        smoothed_omega = self.current_angular_velocity + self.velocity_smoothing_factor * (omega - self.current_angular_velocity)
            
        # 更新当前控制输出记录
        self.current_velocity = smoothed_v
        self.current_angular_velocity = smoothed_omega

        # 发布速度指令
        twist_msg = Twist()
        twist_msg.linear.x = smoothed_v
        twist_msg.angular.z = smoothed_omega
        self.velocity_publisher.publish(twist_msg)
            
        # 添加模式转换检查
        if self.mode == "returning" and r < 0.5:
            # 如果回归路径已经接近原轨迹，切换到正常模式
            self.mode = "normal"
            self.get_logger().info("Successfully returned to original trajectory. Switching to normal mode.")

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
