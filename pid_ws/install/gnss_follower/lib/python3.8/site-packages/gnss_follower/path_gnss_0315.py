#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import math
import csv
import pyproj
import numpy as np
from scipy.interpolate import CubicSpline


class PathPublisherNode(Node):
    def __init__(self):
        super().__init__('path_publisher_node')

        # 声明参数（示例：频率、CSV路径）
        self.declare_parameter('path_publish_freq', 5)  # 默认 100 ms        	
        self.declare_parameter('file_path', '/home/nvidia/pid_ws/src/gnss_follower/gazebo_shihou_course.csv')

        self.freq = self.get_parameter('path_publish_freq').value
        self.file_path = self.get_parameter('file_path').value

        # 发布者
        self.publisher_ = self.create_publisher(Path, 'gnss_path', 10)
        self.origin_publisher_ = self.create_publisher(Path, 'origin_gnss_path', 10)

        # 存储原始 x,y 及相对坐标 x,y
        self.xs_ = []
        self.ys_ = []
        self.origin_xs_ = []
        self.origin_ys_ = []

        self.base_x_ = 0.0
        self.base_y_ = 0.0
        self.init_flag_ = True

        # 读取 CSV 并生成两条 Path
        self.load_csv()
        self.path_msg_ = self.create_path_msg(self.xs_, self.ys_)
        self.origin_path_msg_ = self.create_path_msg(self.origin_xs_, self.origin_ys_)

        # 启动定时器，周期发布
        timer_period = 1.0 / self.freq  # 频率: freq 次/s => 周期: 1/freq s
        self.timer_ = self.create_timer(timer_period, self.loop)

    def load_csv(self):
        """读取 CSV 文件，解析经纬度，转换为 UTM，并分别存入相对坐标和原始坐标数组"""
        self.get_logger().info(f"Trying to open: {self.file_path}")
        try:
            with open(self.file_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    lat = float(row[0])
                    lon = float(row[1])

                    # 转换到 UTM (EPSG:32654)
                    x_utm, y_utm = self.convert_gps_to_utm(lat, lon)

                    # 第一个点作为基准，后续减去基准得到相对坐标
                    if self.init_flag_:
                        self.set_init_pose(x_utm, y_utm)

                    self.xs_.append(x_utm - self.base_x_)
                    self.ys_.append(y_utm - self.base_y_)
                    self.origin_xs_.append(x_utm)
                    self.origin_ys_.append(y_utm)

            self.get_logger().info("Successfully loaded CSV file.")
        except Exception as e:
            self.get_logger().error(f"Failed to open CSV file: {str(e)}")

    def set_init_pose(self, x, y):
        """将首个UTM点作为 (0,0) 基准"""
        self.base_x_ = x
        self.base_y_ = y
        self.init_flag_ = False

    def create_path_msg(self, xs, ys):
        """
        使用样条插值把 (xs, ys) 插值得到更多点 (示例 100 个)，
        再打包成 Path 消息
        """
        spline_points = self.interpolate_spline(xs, ys, 100)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for (px, py) in spline_points:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = float(px)
            pose_stamped.pose.position.y = float(py)
            # 假设 z=0, orientation 不设置
            path_msg.poses.append(pose_stamped)

        return path_msg

    def interpolate_spline(self, xs, ys, num_points):
        """
        对离散点 (xs, ys) 做一次 Cubic Spline 插值
        返回插值后的 (x, y) 数组
        """
        if len(xs) < 2:
            # 数据量太少，就不插值了
            return list(zip(xs, ys))

        # 构建参数序列：0,1,2,...,N-1
        t = np.linspace(0, len(xs) - 1, len(xs))
        cs_x = CubicSpline(t, xs)
        cs_y = CubicSpline(t, ys)

        # 在 [0, N-1] 范围，均匀采样 num_points 个
        t_new = np.linspace(0, len(xs) - 1, num_points)

        result = []
        for tau in t_new:
            result.append((cs_x(tau), cs_y(tau)))

        return result

    def convert_gps_to_utm(self, lat, lon):
        """
        WGS84 -> UTM Zone 54N (EPSG:32654).
        如果需要别的分区，请修改EPSG编号。
        """
        # 检查是否越界
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            self.get_logger().error("Latitude or longitude out of valid range!")
            return (float('inf'), float('inf'))

        # 使用 pyproj.Transformer
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32654", always_xy=True)
        # 注意: Transformer.transform 输入是 (lon, lat)，顺序和EPSG坐标定义有关
        x_utm, y_utm = transformer.transform(lon, lat)
        self.get_logger().info(f"[convert_gps_to_utm] output => x_utm:{x_utm}, y_utm:{y_utm}")
        return (x_utm, y_utm)

    def loop(self):
        """定时发布两条 Path"""
        self.publisher_.publish(self.path_msg_)
        self.origin_publisher_.publish(self.origin_path_msg_)


def main(args=None):
    rclpy.init(args=args)
    node = PathPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
