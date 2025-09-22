#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist,Pose2D
import pandas as pd
import numpy as np
np.float = float
from datetime import datetime
import os

class VelocityRecorder(Node):
    def __init__(self):
        super().__init__('velovity_recorder')

        # 1) 订阅速度话题（假设是 geometry_msgs/Twist）
        #    你可修改成你实际的速度话题名字
        self.velocity_subscriber = self.create_subscription(
            Twist,
            '/aiformula_control/game_pad/cmd_vel',
            self.velocity_callback,
            10
        )
        self.velocity_subscriber  # 防止未使用警告


        self.trajectory_subscriber = self.create_subscription(
            Pose2D,
            '/filtered_lane_pose',
            self.trajectory_callback,
            10
        )
        self.trajectory_subscriber

        self.angular_subscriber = self.create_subscription(
            Pose2D,
            '/lya_angular',
            self.angular_callback,
            10
        )
        self.angular_subscriber
        
        # 2) 准备一个空的DataFrame，用于保存 (Time, V, Omega)
        self.data = pd.DataFrame(columns=["Time", "V", "Omega","X","Y","theta"])

        # 3) 文件夹路径和Excel文件名，可自定义
        self.folder_path = "./lane_analysis_data"
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.excel_file_path = os.path.join(self.folder_path, "lya_data.xlsx")

        self.get_logger().info("VelocityRecorder Node Initialized!")

    def velocity_callback(self, msg: Twist):
        """
        当订阅到Twist消息时，把linear.x和angular.z取出来，记录到self.data
        """
        v = msg.linear.x      # 线速度
        omega = msg.angular.z # 角速度

        # 获取当前时间（用系统本地时间即可）
        current_time_str = datetime.now().strftime("%H:%M:%S")

        # 添加一行 (Time, V, Omega)
        new_row = pd.DataFrame([[current_time_str, v, omega]],
                               columns=["Time", "V", "Omega"])
        self.data = pd.concat([self.data, new_row], ignore_index=True)

        # 打印到日志，方便查看
        self.get_logger().info(f"Received Twist => v={v:.3f}, omega={omega:.3f}")
        
    def trajectory_callback(self, msg: Pose2D):

        x = msg._x
        y = msg._y
        theta = msg._theta
        

        current_time_str = datetime.now().strftime("%H:%M:%S")

    	
        new_row = pd.DataFrame([[current_time_str, x, y, theta]],
                               columns=["Time", "X", "Y", "theta"])
        self.data = pd.concat([self.data, new_row], ignore_index=True)

    def angular_callback(self, msg: Pose2D):

        alpha = msg._x
        beta = msg._y
        r = msg._theta
        

        current_time_str = datetime.now().strftime("%H:%M:%S")

    	
        new_row = pd.DataFrame([[current_time_str, alpha, beta, r]],
                               columns=["Time", "alpha", "beta", "r"])
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        

    def save_data_to_excel(self):
        """
        把self.data写出到excel文件
        """
        try:
            self.data.to_excel(self.excel_file_path, index=False)
            self.get_logger().info(f"Velocity data saved to {self.excel_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save data to Excel: {e}")


def main(args=None):
    # 初始化rclpy
    rclpy.init(args=args)
    recorder = VelocityRecorder()
    try:
        # 运行节点，直到 Ctrl+C
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        # 节点退出时，保存数据到excel
        recorder.save_data_to_excel()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
