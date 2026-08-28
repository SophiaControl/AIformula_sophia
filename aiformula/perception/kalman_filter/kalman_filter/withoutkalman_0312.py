#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Bool

TARGET_SPEED_MPS = 2.0
PATH_SPACING_EPSILON_M = 1.0e-6

class DataProcessingNode(Node):
    def __init__(self):
        super().__init__('data_processing_node')
        
        # 当前数据源标志，初始为 'processed'
        self.current_source = 'processed'
        # 初始订阅：接收来自 /processed_point_* 话题
        self.subscription_a = self.create_subscription(
            Pose2D, '/processed_point_a',
            self.observation_callback_a,
            10
        )
        self.subscription_b = self.create_subscription(
            Pose2D, '/processed_point_b',
            self.observation_callback_b,
            10
        )
        self.subscription_c = self.create_subscription(
            Pose2D, '/processed_point_c',
            self.observation_callback_c,
            10
        )
        # 订阅 flag 话题，用于切换数据源
        self.flag_subscription = self.create_subscription(
            Bool, '/point_flag',
            self.flag_callback,
            10
        )
        self.publisher_point = self.create_publisher(Pose2D, 'filtered_lane_pose', 10)
        self.publisher_omega_t = self.create_publisher(Pose2D, 'filtered_omega_t', 10)
        # 初始化 A, B, C 点
        self.A = None
        self.B = None
        self.C = None

    def observation_callback_a(self, msg: Pose2D):
        self.A = np.array([msg.x, msg.y], dtype=float)

    def observation_callback_b(self, msg: Pose2D):
        self.B = np.array([msg.x, msg.y], dtype=float)

    def observation_callback_c(self, msg: Pose2D):
        self.C = np.array([msg.x, msg.y], dtype=float)
        # 确保 A、B、C 均已接收后进行计算
        if self.A is not None and self.B is not None and self.C is not None:
            Ax, Ay = self.A
            Bx, By = self.B
            Cx, Cy = self.C
            theta_1 = np.arctan2(By - Ay, Bx - Ax)
            theta_2 = np.arctan2(Cy - By, Cx - Bx)
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
            self.get_logger().info(
                f"Filtered => A({Ax:.2f},{Ay:.2f}), B({Bx:.2f},{By:.2f}), C({Cx:.2f},{Cy:.2f}); "
                f"theta1={theta_1:.3f}, theta2={theta_2:.3f}, omega_t={filtered_omega_t.theta:.3f}"
            )

    def flag_callback(self, msg: Bool):
        """
        根据 flag 消息切换数据源：
          - flag True  : 切换到 /oa_point_* 话题
          - flag False : 切换回 /processed_point_* 话题
        """
        if msg.data:
            if self.current_source != 'oa':
                self.get_logger().info("Flag True received. Switching subscription to /oa_point_* topics.")
                self.destroy_subscription(self.subscription_a)
                self.destroy_subscription(self.subscription_b)
                self.destroy_subscription(self.subscription_c)
                self.subscription_a = self.create_subscription(
                    Pose2D, '/oa_point_a',
                    self.observation_callback_a,
                    10
                )
                self.subscription_b = self.create_subscription(
                    Pose2D, '/oa_point_b',
                    self.observation_callback_b,
                    10
                )
                self.subscription_c = self.create_subscription(
                    Pose2D, '/oa_point_c',
                    self.observation_callback_c,
                    10
                )
                self.current_source = 'oa'
                self.get_logger().info("Switched to /oa_point_* topics.")
        else:
            if self.current_source != 'processed':
                self.get_logger().info("Flag False received. Switching subscription to /processed_point_* topics.")
                self.destroy_subscription(self.subscription_a)
                self.destroy_subscription(self.subscription_b)
                self.destroy_subscription(self.subscription_c)
                self.subscription_a = self.create_subscription(
                    Pose2D, '/processed_point_a',
                    self.observation_callback_a,
                    10
                )
                self.subscription_b = self.create_subscription(
                    Pose2D, '/processed_point_b',
                    self.observation_callback_b,
                    10
                )
                self.subscription_c = self.create_subscription(
                    Pose2D, '/processed_point_c',
                    self.observation_callback_c,
                    10
                )
                self.current_source = 'processed'
                self.get_logger().info("Switched to /processed_point_* topics.")

def main(args=None):
    rclpy.init(args=args)
    node = DataProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
