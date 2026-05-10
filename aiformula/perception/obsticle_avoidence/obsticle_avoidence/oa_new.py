#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64

class CollisionResponseNode(Node):
    def __init__(self):
        super().__init__('collision_response_node')
        # 订阅原始车道线数据
        self.sub_center = self.create_subscription(
            PointCloud2, '/aiformula_perception/lane_line_publisher/lane_lines/center', self.center_callback, 10)
        self.sub_left = self.create_subscription(
            PointCloud2, '/aiformula_perception/lane_line_publisher/lane_lines/left', self.left_callback, 10)
        self.sub_right = self.create_subscription(
            PointCloud2, '/aiformula_perception/lane_line_publisher/lane_lines/right', self.right_callback, 10)
        # 订阅碰撞检测节点发布的碰撞角度
        self.sub_collision_angle = self.create_subscription(
            Float64, '/collision_angle', self.collision_angle_callback, 10)
        
        # 发布重新命名后的车道线数据
        self.pub_center = self.create_publisher(PointCloud2, 'lane_line_center', 10)
        self.pub_left = self.create_publisher(PointCloud2, 'lane_line_left', 10)
        self.pub_right = self.create_publisher(PointCloud2, 'lane_line_right', 10)
        # 发布碰撞响应角度
        self.pub_collision_response = self.create_publisher(Float64, 'collision_response', 10)
        
        self.is_playing = False
        self.collision_angle = 0.0
        self.playback_duration = 3.0  # 播放时间（秒）
        self.playback_timer = None
        self.playback_start_time = None

    def center_callback(self, msg: PointCloud2):
        if not self.is_playing:
            self.pub_center.publish(msg)

    def left_callback(self, msg: PointCloud2):
        if not self.is_playing:
            self.pub_left.publish(msg)

    def right_callback(self, msg: PointCloud2):
        if not self.is_playing:
            self.pub_right.publish(msg)

    def collision_angle_callback(self, msg: Float64):
        # 当收到碰撞角度且未处于播放模式时触发
        if not self.is_playing:
            self.get_logger().info(f"Collision detected, angle: {msg.data:.2f} rad")
            self.is_playing = True
            self.collision_angle = msg.data
            self.playback_start_time = self.get_clock().now()
            self.playback_timer = self.create_timer(0.2, self.playback_callback)

    def playback_callback(self):
        now = self.get_clock().now()
        elapsed = (now - self.playback_start_time).nanoseconds / 1e9
        if elapsed < self.playback_duration:
            # 在播放期间反复发布碰撞角度
            self.pub_collision_response.publish(Float64(data=self.collision_angle))
            self.get_logger().info(f"Publishing collision response angle: {self.collision_angle:.2f} rad")
        else:
            self.get_logger().info("Collision response playback completed, resuming normal operation.")
            self.is_playing = False
            if self.playback_timer is not None:
                self.playback_timer.cancel()
                self.playback_timer = None

def main(args=None):
    rclpy.init(args=args)
    node = CollisionResponseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
