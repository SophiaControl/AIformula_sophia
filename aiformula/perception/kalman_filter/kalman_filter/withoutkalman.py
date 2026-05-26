import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose2D


class DataProcessingNode(Node):
    def __init__(self):
        super().__init__('data_processing_node')

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

        # 初始化 A, B, C 为 None，确保每个点在接收时能够赋值
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
        接收到 C 点数据后，进行计算并发布
        """
        self.C = np.array([msg.x, msg.y], dtype=float)

        # 确保 A, B, C 点都已接收到
        if self.A is not None and self.B is not None and self.C is not None:
            # 获取三个点的坐标
            Ax, Ay = self.A
            Bx, By = self.B
            Cx, Cy = self.C

            # === 3) 计算两段夹角 ===
            theta_1 = np.arctan2(By - Ay, Bx - Ax)  # A->B
            theta_2 = np.arctan2(Cy - By, Cx - Bx)  # B->C

            # === 4) 将计算出的角度和omega_t封装为 Pose2D ===
            filtered_omega_t = Pose2D(
                x=theta_1,
                y=theta_2,
                theta=(theta_2 - theta_1) / 0.1
            )
            self.publisher_omega_t.publish(filtered_omega_t)

            filtered_pose = Pose2D(
                x=Ax,
                y=Ay,
                theta=theta_1
            )
            self.publisher_point.publish(filtered_pose)

            # === 5) 输出日志 ===
            self.get_logger().info(
                f"Filtered => A({Ax:.2f},{Ay:.2f}), B({Bx:.2f},{By:.2f}), C({Cx:.2f},{Cy:.2f}); "
                f"theta1={theta_1:.3f}, theta2={theta_2:.3f}, omega_t={filtered_omega_t.theta:.3f}; "
                f"Filtered => A({filtered_pose.x:.2f},{filtered_pose.y:.2f}), omega({filtered_pose.theta:.3f}) "
            )


def main(args=None):
    rclpy.init(args=args)
    node = DataProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
