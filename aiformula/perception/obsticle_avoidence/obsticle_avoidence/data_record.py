#!/usr/bin/env python3
"""ROS 2 实验数据记录节点（改进版）

- 订阅 v/omega （/aiformula_control/game_pad/cmd_vel）
- 同时记录 IMU、路径点、计算耗时、系统资源
- 将全部数据写入带时间戳的 CSV 文件
- 自动启动 rosbag 录制，便于事后复现
"""
from __future__ import annotations

import atexit
import csv
import datetime as _dt
import signal
import subprocess
import time
from pathlib import Path

import psutil
import rclpy
from geometry_msgs.msg import Pose2D, Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

__all__ = ["main"]


class DataRecorder(Node):
    """订阅多个话题并记录到 CSV，同时录制 rosbag。"""

    def __init__(self) -> None:
        super().__init__("data_recorder")

        # —— 单调时钟基准 ——  （使用 ROS 时钟，方便与仿真 / 录包同步）
        self._clock = self.get_clock()

        # —— CSV 文件 ——
        timestamp_str = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"log_{timestamp_str}")
        out_dir.mkdir(exist_ok=True)

        self.accel_f = open(out_dir / "acceleration.csv", "w", newline="")
        self.path_f  = open(out_dir / "path.csv", "w", newline="")
        self.comp_f  = open(out_dir / "compute_time.csv", "w", newline="")
        self.sys_f   = open(out_dir / "system_usage.csv", "w", newline="")
        self.vel_f   = open(out_dir / "vel.csv", "w", newline="")

        # 提前创建 writer，避免每次回调重复创建对象
        self.accel_writer = csv.writer(self.accel_f)
        self.path_writer  = csv.writer(self.path_f)
        self.comp_writer  = csv.writer(self.comp_f)
        self.sys_writer   = csv.writer(self.sys_f)
        self.vel_writer   = csv.writer(self.vel_f)

        # 写表头
        self.accel_writer.writerow(["timestamp", "ax", "ay", "az"])
        self.path_writer.writerow(["timestamp", "x", "y"])
        self.comp_writer.writerow(["timestamp", "compute_time"])
        self.sys_writer.writerow(["timestamp", "cpu_percent", "mem_percent"])
        self.vel_writer.writerow(["timestamp", "v", "omega"])

        # —— 启动 rosbag 录制 ——
        bag_name = f"bag_{timestamp_str}"
        self.get_logger().info(f"Recording rosbag to '{bag_name}/' (all topics)")
        self._bag_proc = subprocess.Popen([
            "ros2", "bag", "record", "-a", "-o", bag_name
        ])
        atexit.register(self._shutdown)

        # QoS：cmd_vel 通常是高速、无损耗要求不高的数据，使用 BEST_EFFORT 即可
        cmd_vel_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
        )

        # —— 订阅话题 ——
        self.create_subscription(Imu, "/imu/data", self.cb_imu, 20)
        self.create_subscription(Pose2D, "/path_point", self.cb_path, 20)
        self.create_subscription(Float32, "/compute_time", self.cb_compute, 10)
        self.create_subscription(Twist, "/aiformula_control/game_pad/cmd_vel", self.cb_vel, cmd_vel_qos)

        # —— 定时记录系统资源 ——
        self.create_timer(1.0, self.cb_sys)

    # ---------- 工具 ----------
    def _now(self) -> float:
        """返回当前 ROS 时钟时间（秒）。"""
        stamp = self._clock.now()
        return stamp.nanoseconds * 1e-9

    def _flush_all(self) -> None:
        """确保数据及时落盘，防止意外掉电导致数据丢失。"""
        for f in (self.accel_f, self.path_f, self.comp_f, self.sys_f, self.vel_f):
            f.flush()

    # ---------- 回调 ----------
    def cb_imu(self, msg: Imu) -> None:
        t = self._now()
        self.accel_writer.writerow(
            [
                t,
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
        )

    def cb_path(self, msg: Pose2D) -> None:
        t = self._now()
        self.path_writer.writerow([t, msg.x, msg.y])

    def cb_compute(self, msg: Float32) -> None:
        t = self._now()
        self.comp_writer.writerow([t, msg.data])

    def cb_vel(self, msg: Twist) -> None:
        """记录控制器计算出的线、角速度（v/omega）。"""
        t = self._now()
        self.vel_writer.writerow([t, msg.linear.x, msg.angular.z])
        # 若需要实时查看，可取消下行注释
        # self.get_logger().debug(f"v={msg.linear.x:.3f}, ω={msg.angular.z:.3f}")

    def cb_sys(self) -> None:
        t = self._now()
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        self.sys_writer.writerow([t, cpu, mem])
        # 每秒刷盘一次，保证数据安全
        self._flush_all()

    # ---------- 结束清理 ----------
    def _shutdown(self) -> None:
        """终止 rosbag 并关闭文件。"""
        if self._bag_proc and self._bag_proc.poll() is None:
            self._bag_proc.send_signal(signal.SIGINT)
            try:
                self._bag_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._bag_proc.kill()
        for f in (
            self.accel_f,
            self.path_f,
            self.comp_f,
            self.sys_f,
            self.vel_f,
        ):
            try:
                f.close()
            except Exception:
                pass


# ---------- main ----------

def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

