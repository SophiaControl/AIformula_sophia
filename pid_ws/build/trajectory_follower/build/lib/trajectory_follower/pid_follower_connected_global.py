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

class PIDController:
    def __init__(self, k_vp, k_vi, k_vd, k_wp, k_wi, k_wd):
        # 线速度PID参数
        self.k_vp = k_vp
        self.k_vi = k_vi
        self.k_vd = k_vd
        
        # 角速度PID参数
        self.k_wp = k_wp
        self.k_wi = k_wi
        self.k_wd = k_wd
        
        # 线速度误差项初始化
        self.e_v = 0.0
        self.i_v = 0.0
        self.prev_e_v = 0.0
        
        # 角速度误差项初始化
        self.e_w = 0.0
        self.i_w = 0.0
        self.prev_e_w = 0.0
        
        # 时间相关
        self.prev_time = None

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

        # ---- (4) 定义轨迹(订阅) ----
        self.create_subscription(Pose2D, 
                                 '/filtered_lane_pose', 
                                 self.trajectory_callback, 
                                 10)
        self.target_point = []
        self.current_pose = None
        self.current_velocity = None

        # ---- (5) 初始化PID控制器 ----
        self.pid_controller = PIDController(
            k_vp=1.00, k_vi=0.00, k_vd=0.0,  # 线速度PID参数
            k_wp=3.00, k_wi=0.00, k_wd=0.4   # 角速度PID参数
        )

        # ---- 其他变量初始化 ----
        self.previous_time = self.get_clock().now()
        
        # ---- 期望速度相关参数 ----
        self.desired_speed = 1.5  # 期望线速度

    def odom_callback(self, msg: Odometry):
        # 获取当前位姿（在 odom 坐标系下）
        self.current_pose = msg.pose.pose  
        
        # 获取当前速度
        self.current_velocity = msg.twist.twist
        
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
        if self.current_pose is None or self.current_velocity is None:
            return

        # 执行PID控制
        self.execute_pid_control()

    def execute_pid_control(self):
        """执行PID控制算法"""
        # ---------- 提取目标 & 当前姿态 ----------
        target_x = self.target_point["x"]
        target_y = self.target_point["y"]
        target_theta = self.target_point["theta"]

        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        current_qx = self.current_pose.orientation.x
        current_qy = self.current_pose.orientation.y
        current_qz = self.current_pose.orientation.z
        current_qw = self.current_pose.orientation.w

        # 计算当前朝向角
        _, _, current_yaw = tf_transformations.euler_from_quaternion([
            current_qx, current_qy, current_qz, current_qw
        ])

        # 获取当前速度
        current_linear_vel = self.current_velocity.linear.x
        current_angular_vel = self.current_velocity.angular.z

        self.get_logger().info(f"Current pose: x={current_x:.2f}, y={current_y:.2f}, yaw={current_yaw:.2f}")
        self.get_logger().info(f"Current velocity: v={current_linear_vel:.2f}, w={current_angular_vel:.2f}")

        # ---------- 计算轨迹误差并得到期望速度 ----------
        # 由于轨迹点是相对于车身的固定纵向距离，我们直接使用这些点来计算控制量
        # 这里的target_point实际上是经过坐标变换后的全局坐标点
        
        # 计算横向误差（垂直于车身方向的误差）
        # 将目标点转换回车身坐标系来计算横向误差
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)
        
        # 全局坐标差值
        dx_global = target_x - current_x
        dy_global = target_y - current_y
        
        # 转换到车身坐标系
        dx_body = dx_global * cos_yaw + dy_global * sin_yaw  # 纵向误差
        dy_body = -dx_global * sin_yaw + dy_global * cos_yaw  # 横向误差（左正右负）
        
        # 计算期望线速度（保持恒定速度）
        desired_linear_vel = self.desired_speed
        
        # 计算期望角速度（基于横向误差）
        # 横向误差越大，需要转向越多
        # 使用简单的比例控制，可根据实际效果调整系数
        lateral_error_gain = 3.0  # 横向误差增益，可调整
        desired_angular_vel = lateral_error_gain * dy_body
        
        # 限制期望角速度
        max_angular_vel = 1.0
        desired_angular_vel = max(-max_angular_vel, min(desired_angular_vel, max_angular_vel))

        self.get_logger().info(f"Desired velocity: v={desired_linear_vel:.3f}, w={desired_angular_vel:.3f}")
        self.get_logger().info(f"Body frame errors: dx={dx_body:.3f}, dy={dy_body:.3f}")

        # ---------- PID控制器计算 ----------
        current_time = self.get_clock().now()
        
        if self.pid_controller.prev_time is None:
            self.pid_controller.prev_time = current_time
            dt = 0.01  # 初始时间步长
        else:
            dt = (current_time - self.pid_controller.prev_time).nanoseconds / 1e9
            self.pid_controller.prev_time = current_time
        
        # 避免dt为0或过小
        if dt <= 0:
            dt = 0.01

        # 线速度PID控制
        vel_error = desired_linear_vel - current_linear_vel
        self.pid_controller.i_v += vel_error * dt
        
        # 积分饱和限制
        max_integral = 1.0
        self.pid_controller.i_v = max(-max_integral, min(self.pid_controller.i_v, max_integral))
        
        vel_derivative = (vel_error - self.pid_controller.prev_e_v) / dt
        
        linear_control = (self.pid_controller.k_vp * vel_error + 
                         self.pid_controller.k_vi * self.pid_controller.i_v + 
                         self.pid_controller.k_vd * vel_derivative)
        
        self.pid_controller.prev_e_v = vel_error

        # 角速度PID控制
        angular_error = desired_angular_vel - current_angular_vel
        self.pid_controller.i_w += angular_error * dt
        
        # 积分饱和限制
        self.pid_controller.i_w = max(-max_integral, min(self.pid_controller.i_w, max_integral))
        
        angular_derivative = (angular_error - self.pid_controller.prev_e_w) / dt
        
        angular_control = (self.pid_controller.k_wp * angular_error + 
                          self.pid_controller.k_wi * self.pid_controller.i_w + 
                          self.pid_controller.k_wd * angular_derivative)
        
        self.pid_controller.prev_e_w = angular_error

        # ---------- 计算最终控制输出 ----------
        # 将PID输出加到当前速度上作为控制指令
        final_linear_vel = current_linear_vel + linear_control
        final_angular_vel = current_angular_vel + angular_control

        # ---------- 速度限制 ----------
        max_linear_velocity = 1.75   
        min_linear_velocity = 0.5
        max_angular_velocity = 0.5
        min_angular_velocity = -0.5

        final_linear_vel = max(min_linear_velocity, min(final_linear_vel, max_linear_velocity))
        final_angular_vel = max(min_angular_velocity, min(final_angular_vel, max_angular_velocity))

        self.get_logger().info(f"PID Control => v={final_linear_vel:.3f}, omega={final_angular_vel:.3f}")
        self.get_logger().info(f"Errors: vel_error={vel_error:.3f}, angular_error={angular_error:.3f}")

        # ---------- 发布速度指令 ----------
        twist_msg = Twist()
        twist_msg.linear.x = final_linear_vel
        twist_msg.angular.z = final_angular_vel
        self.velocity_publisher.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    follower = TrajectoryFollower()
    rclpy.spin(follower)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
