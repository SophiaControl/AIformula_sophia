import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import pandas as pd
import math
from tf_transformations import euler_from_quaternion

CSV_PATH = "/home/nvidia/pid_ws/src/trajectory_follower/trajectory_follower/course0611.csv"
LD = 1.5          # 前视距离 (m)
DT = 0.1          # 控制周期 (s)

# —— Lyapunov 增益 ——
LAM_V = 0.02
LAM_A = 0.25
K1, K2 = 0.7, 50.0
V_T = 2.0          # 期望线速度
EPS = 1e-4

# —— 速度极限 ——
MAX_V, MIN_V = 2.25, 0.60
MAX_W, MIN_W = 0.40, -0.40

class Follower(Node):
    def __init__(self):
        super().__init__('lya_follower_dynamic_3pt')

        # 读路径
        df = pd.read_csv(CSV_PATH)
        self.path = df[['x', 'y']].values
        self.idx = 0
        self.pre_idx = 0
        self.get_logger().info(f"Loaded {len(self.path)} way‑points from {CSV_PATH}")

        # 订阅里程计
        self.pose_ok = False
        self.x = self.y = self.yaw = 0.0
        self.create_subscription(Odometry, '/aiformula_sensing/gyro_odometry_publisher/odom', self.odom_cb, 30)

        # 发布
        self.cmd_pub = self.create_publisher(Twist, '/aiformula_control/game_pad/cmd_vel', 10)

        # 定时器
        self.create_timer(DT, self.control_loop)

    # ---------------- 里程计回调 -----------------
    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.pose_ok = True

    # ---------------- 控制主循环 -----------------
    def control_loop(self):
        if not self.pose_ok:
            return

        # 1. 取 A 为第一个 dist>LD 的点
        for i in range(self.pre_idx, len(self.path)):
            if math.hypot(self.path[i,0]-self.x, self.path[i,1]-self.y) > LD and i > self.pre_idx:
                self.idx = i
                self.pre_idx = i - 1
                break
        else:
            self.stop_robot(); return

        # 确保 B,C 存在
        if self.idx + 2 >= len(self.path):
            self.stop_robot(); return

        self.get_logger().info(f" idx={self.idx}")
        
        A = self.path[self.idx]
        B = self.path[self.idx+1]
        C = self.path[self.idx+2]

        # 2. 计算 θ_BA, θ_CB, omega_des
        theta_BA = math.atan2(B[1]-A[1], B[0]-A[0])   # 作为目标切向
        theta_CB = math.atan2(C[1]-B[1], C[0]-B[0])
        omega_des = math.atan2(math.sin(theta_CB - theta_BA), math.cos(theta_CB - theta_BA)) / DT

        # 3. 误差计算（严格按你原公式写法）
        dx, dy =  A[0] - self.x , A[1] - self.y
        self.get_logger().info(f" dx={dx}, dy={dy}")
        r = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)

        # alpha = theta - yaw (再归一化到 -π~π)
        alpha_raw = theta - self.yaw
        alpha = math.atan2(math.sin(alpha_raw), math.cos(alpha_raw))

        # beta  = theta - phi_t，其中 phi_t 取段 BA 的方向角
        beta_raw = theta - theta_BA
        beta = math.atan2(math.sin(beta_raw), math.cos(beta_raw))

        sin_alpha = math.sin(alpha) if abs(math.sin(alpha)) > EPS else EPS * math.copysign(1, alpha)
        sin_beta  = math.sin(beta)

        # 4. 速度 / 角速度指令（完全按原公式 + omega_des 前馈）
        v = (V_T * math.cos(beta) + LAM_V * dx) * math.cos(alpha)  # 仅用全局 dx
        omega = (
            LAM_A * math.sin(alpha)
            + (K1 / sin_alpha) * (
                (math.sin(alpha) / (K1 * r)) + (sin_beta / (K2 * r))
            ) * (
                (math.sin(2 * alpha) * math.cos(beta) / 2) - sin_beta
            ) * V_T
            - (omega_des * sin_beta / K2) * (K1 / sin_alpha)
            + (K1 / sin_alpha) * LAM_V * (math.sin(2 * alpha) / 2) * (
                (math.sin(alpha) / K1) + (sin_beta / K2)
            )
        )

        # 5. 饱和并发布
        v = max(MIN_V, min(v, MAX_V))
        omega = max(MIN_W, min(omega, MAX_W))
        self.get_logger().info(f" v={v}, omega={omega}")
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

    # ---------------- 停车 -----------------
    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        self.get_logger().info("轨迹完成 — 已停车")


def main():
    rclpy.init()
    node = Follower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
