import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose2D

TARGET_SPEED_MPS = 2.0
PATH_SPACING_EPSILON_M = 1.0e-6


class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')

        # === 1) 订阅和发布 Pose2D 数据 ===
        self.pose_subscription_a = self.create_subscription(
            Pose2D, '/processed_point_a',
            self.observation_callback_a,
            10
        )
        self.pose_subscription_b = self.create_subscription(
            Pose2D, '/processed_point_b',
            self.observation_callback_b,
            10
        )
        self.pose_subscription_c = self.create_subscription(
            Pose2D, '/processed_point_c',
            self.observation_callback_c,
            10
        )
        self.publisher_point = self.create_publisher(
            Pose2D, 'filtered_lane_pose', 10)
        
        self.publisher_omega_t = self.create_publisher(
            Pose2D, 'filtered_omega_t', 10)

        # === 2) 初始化卡尔曼滤波器参数 ===

        # 状态从 3 维扩展为 6 维: [A_x, A_y, B_x, B_y, C_x, C_y]
        self.x = np.zeros(6)        # 初始状态
        self.P = np.eye(6) * 500   # 初始协方差矩阵
        self.F = np.eye(6)          # 状态转移矩阵(简单假设静止, 仅用于 predict)
        self.Q = np.eye(6) * 0.01    # 过程噪声协方差(可根据实际情况调)
        
        # 对测量做局部更新，需要针对 A/B/C 定义不同的 H、R
        # A 点只观测 state[0], state[1]，即 A_x, A_y
        self.H_a = np.array([
            [1, 0, 0, 0, 0, 0],  # 对应 A_x
            [0, 1, 0, 0, 0, 0],  # 对应 A_y
        ])
        self.R_a = np.eye(2) * 0.1
        
        # B 点观测 state[2], state[3]
        self.H_b = np.array([
            [0, 0, 1, 0, 0, 0],  # B_x
            [0, 0, 0, 1, 0, 0],  # B_y
        ])
        self.R_b = np.eye(2) * 0.1
        
        # C 点观测 state[4], state[5]
        self.H_c = np.array([
            [0, 0, 0, 0, 1, 0],  # C_x
            [0, 0, 0, 0, 0, 1],  # C_y
        ])
        self.R_c = np.eye(2) * 0.1

        # 初始化存储点
        self.A = None
        self.B = None
        self.C = None

    def observation_callback_a(self, msg):
        """
        仅存储 A 点数据
        """
        self.A = np.array([msg.x, msg.y], dtype=float)

    def observation_callback_b(self, msg):
        """
        仅存储 B 点数据
        """
        self.B = np.array([msg.x, msg.y], dtype=float)

    def observation_callback_c(self, msg):
        """
        接收到 C 点数据后，进行滤波和计算
        """
        self.C = np.array([msg.x, msg.y], dtype=float)

        # 确保 A, B, C 点都已接收
        if self.A is not None and self.B is not None and self.C is not None:
            # 获取三个点的坐标
            Ax, Ay = self.A
            Bx, By = self.B
            Cx, Cy = self.C

            # === 2) 预测步骤 ===
            self.x = np.dot(self.F, self.x)
            self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

            # === 3) 更新步骤: 根据 label 选择对应的 H, R ===
            # 更新 A 点
            H_current = self.H_a
            R_current = self.R_a
            S = np.dot(H_current, np.dot(self.P, H_current.T)) + R_current
            K = np.dot(self.P, np.dot(H_current.T, np.linalg.inv(S)))
            y = np.array([Ax, Ay]) - np.dot(H_current, self.x)
            self.x = self.x + np.dot(K, y)
            I = np.eye(6)
            self.P = np.dot((I - np.dot(K, H_current)), self.P)

            # 更新 B 点
            H_current = self.H_b
            R_current = self.R_b
            S = np.dot(H_current, np.dot(self.P, H_current.T)) + R_current
            K = np.dot(self.P, np.dot(H_current.T, np.linalg.inv(S)))
            y = np.array([Bx, By]) - np.dot(H_current, self.x)
            self.x = self.x + np.dot(K, y)
            self.P = np.dot((I - np.dot(K, H_current)), self.P)

            # 更新 C 点
            H_current = self.H_c
            R_current = self.R_c
            S = np.dot(H_current, np.dot(self.P, H_current.T)) + R_current
            K = np.dot(self.P, np.dot(H_current.T, np.linalg.inv(S)))
            y = np.array([Cx, Cy]) - np.dot(H_current, self.x)
            self.x = self.x + np.dot(K, y)
            self.P = np.dot((I - np.dot(K, H_current)), self.P)

            # === 4) 限制输出范围 ===
            self.x[0] = np.clip(self.x[0], -3.5, 3.5)  # A_x
            self.x[1] = np.clip(self.x[1], -1.0, 1.0)  # A_y
            self.x[2] = np.clip(self.x[2], -3.5, 3.5)  # B_x
            self.x[3] = np.clip(self.x[3], -1.0, 1.0)  # B_y
            self.x[4] = np.clip(self.x[4], -3.5, 3.5)  # C_x
            self.x[5] = np.clip(self.x[5], -1.0, 1.0)  # C_y

            # === 5) 基于滤波后的 (A_x,A_y,B_x,B_y,C_x,C_y) 计算两段夹角 ===
            Ax, Ay, Bx, By, Cx, Cy = self.x
            theta_1 = np.arctan2(By - Ay, Bx - Ax)  # A->B
            theta_2 = np.arctan2(Cy - By, Cx - Bx)  # B->C

            # A、B、C 是空间点：先求局部曲率，再乘目标速度得到 omega_t
            heading_change = np.arctan2(
                np.sin(theta_2 - theta_1), np.cos(theta_2 - theta_1)
            )
            heading_spacing = 0.5 * (
                np.hypot(Bx - Ax, By - Ay) + np.hypot(Cx - Bx, Cy - By)
            )
            target_heading_rate = (
                0.0 if heading_spacing <= PATH_SPACING_EPSILON_M
                else TARGET_SPEED_MPS * heading_change / heading_spacing
            )

            # === 6) 将两个夹角和参考角速度打包发布 ===
            filtered_omega_t = Pose2D(
                x=theta_1,
                y=theta_2,
                theta=target_heading_rate
            )
            self.publisher_omega_t.publish(filtered_omega_t)

            filtered_pose = Pose2D(
                x=Ax,
                y=Ay,
                theta=theta_1
            )
            self.publisher_point.publish(filtered_pose)

            # === 7) 输出日志 ===
            self.get_logger().info(
                f"Filtered => A({Ax:.2f},{Ay:.2f}), B({Bx:.2f},{By:.2f}), C({Cx:.2f},{Cy:.2f}); "
                f"theta1={theta_1:.3f}, theta2={theta_2:.3f}, omega_t={filtered_omega_t.theta:.3f}; "
                f"Filtered => A({filtered_pose.x:.2f},{filtered_pose.y:.2f}), omega({filtered_pose.theta:.3f}) "
            )


def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
