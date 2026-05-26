import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
import math
import numpy as np
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
        
        # 发布速度指令
        self.velocity_publisher = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # 轨迹参数
        self.omega_t = 0.01  # 轨迹角频率
        self.trajectory_duration = 2 * math.pi / self.omega_t  # 一个完整周期的时间
        self.t = 0.0  # 轨迹时间参数，从0开始
        
        self.current_pose = None
        self.timer = self.create_timer(0.1, self.follow_trajectory)

        # 初始化LYA控制器
        self.lya = LYAController(omega_t=0.01, v_t=1.5, lambda_v=0.075, lambda_a=0.015, k1=0.8, k2=50)
      
        # 初始化前一时间点为当前时间，避免初始化阶段的错误处理问题
        self.previous_time = self.get_clock().now()
        # 初始化时间差值
        self.dt = 0.1  # 默认值与timer周期一致
        
        # 添加轨迹完成标志
        self.trajectory_completed = False

    def get_trajectory_point(self, t):
        """根据时间t计算轨迹上的点"""
        x_t = 3 - 15 * math.cos(self.omega_t * t)
        y_t = 47 + 15 * math.sin(self.omega_t * t)
        return x_t, y_t

    def get_trajectory_derivative(self, t):
        """计算轨迹的导数（速度方向）"""
        dx_dt = 15 * self.omega_t * math.sin(self.omega_t * t)
        dy_dt = 15 * self.omega_t * math.cos(self.omega_t * t)
        return dx_dt, dy_dt

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose  # 获取当前位姿

    def follow_trajectory(self):
        if self.current_pose is None:
            return  # 等待里程计数据

        # 如果轨迹已完成，直接返回
        if self.trajectory_completed:
            return

        # 检查是否完成一个完整周期
        if self.t >= self.trajectory_duration:
            self.stop_robot()
            return

        # 获取当前目标点和下一个时刻的目标点
        dt_forward = 0.1  # 前瞻时间
        target_x, target_y = self.get_trajectory_point(self.t)
        target_x_next, target_y_next = self.get_trajectory_point(self.t + dt_forward)

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
        
        # 添加距离阈值检查，避免除零
        epsilon = 1e-6
        if r < epsilon:
            r = epsilon
        
        theta = math.atan2(dy, dx)
        alpha = theta - current_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # 归一化角度
        
        # 改进 phi_t 计算，使用轨迹导数
        dx_dt, dy_dt = self.get_trajectory_derivative(self.t)
        phi_t = math.atan2(dy_dt, dx_dt)  # 轨迹切线方向
        
        beta = theta - phi_t
        beta = math.atan2(math.sin(beta), math.cos(beta))  # 归一化角度

        # 对 sin(alpha) 做防护：若 sin(alpha) 接近 0，则设置一个小阈值 epsilon
        safe_sin_alpha = math.sin(alpha)
        if abs(safe_sin_alpha) < epsilon:
            safe_sin_alpha = epsilon if safe_sin_alpha >= 0 else -epsilon

        # 获取当前时间并计算时间步长
        current_time = self.get_clock().now()
        dt = (current_time - self.previous_time).nanoseconds / 1e9  # 转换为秒
        if dt > 0 and dt < 1.0:  # 确保时间步长有效且合理
            self.dt = dt
            self.t += self.dt  # 更新轨迹时间参数
        self.previous_time = current_time

        # 通过LYA控制器计算速度指令，使用safe_sin_alpha替代直接的math.sin(alpha)
        v = ((self.lya.v_t) * math.cos(beta) + self.lya.lambda_v * dx) * math.cos(alpha)

        # 使用safe_sin_alpha来避免除零错误
        k1_over_sin_alpha = self.lya.k1 / safe_sin_alpha

        # 分步计算omega以提高可读性和数值稳定性
        term1 = self.lya.lambda_a * safe_sin_alpha
        
        # 计算复合项
        sin_alpha_over_k1r = safe_sin_alpha / (self.lya.k1 * r)
        sin_beta_over_k2r = math.sin(beta) / (self.lya.k2 * r)
        combined_term = sin_alpha_over_k1r + sin_beta_over_k2r
        
        # 计算三角函数项
        trig_term = (math.sin(2 * alpha) * math.cos(beta) / 2) - math.sin(beta)
        
        term2 = k1_over_sin_alpha * combined_term * trig_term * self.lya.v_t
        
        term3 = -(self.lya.omega_t * math.sin(beta) / self.lya.k2) * k1_over_sin_alpha
        
        term4 = k1_over_sin_alpha * self.lya.lambda_v * (math.sin(2 * alpha) / 2) * combined_term
        
        omega = term1 + term2 + term3 + term4
        
        # 对 v 和 omega 进行饱和限制，避免输出过大
        max_linear_velocity = 1.25   
        min_linear_velocity = 0.6  # 提高最小速度，避免停滞
        max_angular_velocity = 0.4  
        min_angular_velocity = -0.4

        # 确保线速度不为负
        v = max(min_linear_velocity, min(abs(v), max_linear_velocity))
        omega = max(min_angular_velocity, min(omega, max_angular_velocity))
        
        # 添加数值检查，避免NaN或无穷大
        if not (math.isfinite(v) and math.isfinite(omega)):
            self.get_logger().warn(f"Invalid control values: v={v}, omega={omega}")
            v = min_linear_velocity
            omega = 0.0

        # 发布速度指令
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = omega
        self.velocity_publisher.publish(twist_msg)

        # 输出调试信息
        self.get_logger().info(f"Time: {self.t:.2f}s, Target: ({target_x:.2f}, {target_y:.2f}), Current: ({current_x:.2f}, {current_y:.2f}), Error: {r:.2f}")

    def quaternion_to_yaw(self, orientation):
        qx, qy, qz, qw = (orientation.x, orientation.y, orientation.z, orientation.w)
        
        # 验证四元数是否有效
        norm = qx*qx + qy*qy + qz*qz + qw*qw
        if abs(norm - 1.0) > 0.1:  # 允许一定的误差
            # 归一化四元数
            if norm > 1e-6:  # 避免除零
                norm_sqrt = math.sqrt(norm)
                qx /= norm_sqrt
                qy /= norm_sqrt
                qz /= norm_sqrt
                qw /= norm_sqrt
            else:
                self.get_logger().warn("Invalid quaternion with near-zero norm")
                return 0.0  # 返回默认值
                
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def stop_robot(self):
        if not self.trajectory_completed:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.velocity_publisher.publish(twist_msg)
            self.get_logger().info("Trajectory completed. Stopping robot.")
            self.trajectory_completed = True
        
def main(args=None):
    rclpy.init(args=args)
    try:
        follower = TrajectoryFollower()
        rclpy.spin(follower)
    except KeyboardInterrupt:
        print("Shutting down trajectory follower...")
    except Exception as e:
        print(f"Error in trajectory follower: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()