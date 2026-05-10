#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import math

class DataJudgmentNode(Node):
    def __init__(self):
        super().__init__('data_judgment_node')
        
        self.sub_lane_center = self.create_subscription(
            PointCloud2,
            '/aiformula_perception/lane_line_publisher/lane_lines/center',
            self.lane_center_callback,
            10
        )
        self.sub_lane_left = self.create_subscription(
            PointCloud2,
            '/aiformula_perception/lane_line_publisher/lane_lines/left',
            self.lane_left_callback,
            10
        )
        self.sub_lane_right = self.create_subscription(
            PointCloud2,
            '/aiformula_perception/lane_line_publisher/lane_lines/right',
            self.lane_right_callback,
            10
        )
        
        self.sub_red_pixels = self.create_subscription(
            Int32,
            '/red_pixels_count',
            self.red_pixels_callback,
            10
        )
        
        # 发布重新命名后的车道线数据
        self.pub_lane_center = self.create_publisher(
            PointCloud2,
            'lane_line_center',  
            10
        )
        self.pub_lane_left = self.create_publisher(
            PointCloud2,
            'lane_line_left',
            10
        )
        self.pub_lane_right = self.create_publisher(
            PointCloud2,
            'lane_line_right',
            10
        )
        
        self.pub_path_point = self.create_publisher(
            Pose2D,
            'path_point',
            10
        )
        # # 三个发布者：分别发布 A、B、C 三个点的 Pose2D ====
        # self.pose_publisher_a = self.create_publisher(Pose2D, '/oa_point_a', 10)
        # self.pose_publisher_b = self.create_publisher(Pose2D, '/oa_point_b', 10)
        # self.pose_publisher_c = self.create_publisher(Pose2D, '/oa_point_c', 10)
        
        # 状态变量：是否正在执行预录路径点控制
        self.is_playing = False
        # 下次播放的路径组，初始设置为 'L'
        self.next_group = 'L'
        
        # 预录路径点数据，两组（L 和 R）
        self.new_path_L = [
            Pose2D(x=0.7, y=0.72),
            Pose2D(x=1.4, y=1.44),
            Pose2D(x=2.1, y=2.16),
            Pose2D(x=2.8, y=2.88),
            Pose2D(x=3.5, y=3.6),
            Pose2D(x=4.2, y=3.6),
            Pose2D(x=4.9, y=3.6)
        ]
        self.new_path_R = [
            Pose2D(x=0.7, y=-0.72),
            Pose2D(x=1.4, y=-1.44),
            Pose2D(x=2.1, y=-2.16),
            Pose2D(x=2.8, y=-2.88),
            Pose2D(x=3.5, y=-3.6),
            Pose2D(x=4.2, y=-3.6),
            Pose2D(x=4.9, y=-3.6)
        ]
        
        # 当前路径点列表和播放索引
        self.current_path = []
        self.playback_index = 0
        
        # 设置里程计判断的距离阈值（当距离目标点不足此值时认为已经到达）
        self.waypoint_threshold = 0.2  # 单位：米
        
        # 订阅 odometry 数据，用于判断机器人是否到达当前目标点
        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.get_logger().info('Data Judgment Node started.')

    # 车道线数据直接转发
    def lane_center_callback(self, msg: PointCloud2):
        if not self.is_playing:
            self.pub_lane_center.publish(msg)
            self.get_logger().info(f'Republished center lane data: {msg.data}')

    def lane_left_callback(self, msg: PointCloud2):
        if not self.is_playing:
            self.pub_lane_left.publish(msg)
            self.get_logger().info(f'Republished left lane data: {msg.data}')

    def lane_right_callback(self, msg: PointCloud2):
        if not self.is_playing:
            self.pub_lane_right.publish(msg)
            self.get_logger().info(f'Republished right lane data: {msg.data}')

    # 红色像素检测回调，当像素数达到阈值时启动路径点播放
    def red_pixels_callback(self, msg: Int32):
        if not self.is_playing and msg.data >= 800:
            self.get_logger().info(f'Red pixel count {msg.data} reached threshold. Initiating path playback.')
            self.start_playback()

    # 启动预录路径点播放流程
    def start_playback(self):
        self.is_playing = True
        if self.next_group == 'L':
            self.current_path = self.new_path_L
        else:
            self.current_path = self.new_path_R
        self.playback_index = 0
        self.get_logger().info(f'Starting playback for group {self.next_group}.')
        if self.current_path:
            current_wp = self.current_path[self.playback_index]
            self.pub_path_point.publish(current_wp)
            self.get_logger().info(f'Publishing waypoint {self.playback_index}: (x={current_wp.x}, y={current_wp.y})')

    # odom 回调用于判断机器人是否到达当前目标点
    def odom_callback(self, msg):
        if not self.is_playing:
            return
        
        # 获取当前机器人位姿（这里仅使用 x, y）
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        
        if self.playback_index >= len(self.current_path):
            return  
        
        target = self.current_path[self.playback_index]
        dx = target.x - current_x
        dy = target.y - current_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # 如果机器人已到达当前目标点（距离小于阈值）
        if distance < self.waypoint_threshold:
            self.get_logger().info(f"Reached waypoint {self.playback_index} (distance: {distance:.2f} m).")
            self.playback_index += 1
            if self.playback_index < len(self.current_path):
                # 发布下一个目标点
                next_point = self.current_path[self.playback_index]
                self.pub_path_point.publish(next_point)
                self.get_logger().info(f'Publishing next waypoint {self.playback_index}: (x={next_point.x}, y={next_point.y}).')
            else:
                # 所有路径点走完，退出路径点控制模式
                self.is_playing = False
                self.get_logger().info(f"Completed all waypoints for group {self.next_group}. Resuming lane line transmission.")
                # 交替LR更换
                self.next_group = 'R' if self.next_group == 'L' else 'L'
        else:
            self.pub_path_point.publish(target)

def main(args=None):
    rclpy.init(args=args)
    node = DataJudgmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
