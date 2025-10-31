#!/usr/bin/env python3
import rclpy, numpy as np
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from data_record.msg import PlannerMetrics
from data_record.path_geometry import path_length, curvature_stats

class MetricsCollector(Node):
    def __init__(self):
        super().__init__('metrics_collector')
        self.sub_path = self.create_subscription(Path, '/planned_path', self.path_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 50)
        self.sub_traj = self.create_subscription(Path, '/controller/trajectory', self.traj_cb, 10)
        self.pub = self.create_publisher(PlannerMetrics, '/planner/metrics', 10)

        self.last_path_info = None
        self.v_hist = []  # list of (time_sec, v)
        self.exec_latency_ms = 0.0
        self.publish_period = 0.5  # seconds
        self.last_pub_time = self.get_clock().now()

    def path_cb(self, msg: Path):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        length = path_length(pts)
        curv_avg, curv_max = curvature_stats(pts)
        self.last_path_info = (length, curv_avg, curv_max)
        self.try_publish()

    def odom_cb(self, msg: Odometry):
        v = (msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2) ** 0.5
        t = self.get_clock().now().nanoseconds * 1e-9
        self.v_hist.append((t, v))
        # keep 2 seconds
        self.v_hist = [(tt, vv) for tt, vv in self.v_hist if t - tt < 2.0]
        self.try_publish()

    def traj_cb(self, msg: Path):
        now_ns = self.get_clock().now().nanoseconds
        msg_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.exec_latency_ms = (now_ns - msg_ns) / 1e6

    def try_publish(self):
        now = self.get_clock().now()
        if (now - self.last_pub_time).nanoseconds < self.publish_period * 1e9:
            return
        if self.last_path_info is None or len(self.v_hist) < 2:
            return
        length, curv_avg, curv_max = self.last_path_info
        times, vs = zip(*self.v_hist)
        v_max = max(vs)
        accs = [(vs[i]-vs[i-1])/(times[i]-times[i-1]) for i in range(1, len(vs))]
        a_max = max(accs) if accs else 0.0

        m = PlannerMetrics()
        m.stamp.sec = now.seconds_nanoseconds()[0]
        m.stamp.nanosec = now.seconds_nanoseconds()[1]
        m.path_length = length
        m.curvature_avg = curv_avg
        m.curvature_max = curv_max
        m.v_max = float(v_max)
        m.a_max = float(a_max)
        m.plan_time_ms = 0.0  # to be filled by planner node if desired
        m.exec_latency_ms = float(self.exec_latency_ms)
        self.pub.publish(m)
        self.last_pub_time = now

def main(args=None):
    rclpy.init(args=args)
    node = MetricsCollector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
