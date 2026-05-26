#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b_spline_final_1a_resume_dyn.py · 2025-06-04
────────────────────────────────────────────
红色像素 > red_thresh_px  → 重新规划 B-样条并即时发布 3 个全局点
红色像素 < red_resume_px  → 以 3 点一组 + 动态原点方式把“剩余采样点”发完
全部发完后退出避障，恢复点云透传
"""

from __future__ import annotations
import cv2, numpy as np, scipy.interpolate as si, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge, CvBridgeError


class RedAlertDetector(Node):
    # ────────── 初始化 ──────────
    def __init__(self):
        super().__init__('red_alert_detector')

        # ----- 可调参数 -----
        self.m_per_px       = self.declare_parameter('meters_per_pixel', 0.01).value
        self.red_thresh_px  = self.declare_parameter('red_thresh_px', 900).value   # 进入避障
        self.red_resume_px  = self.declare_parameter('red_resume_px', 50).value    # 恢复行进
        self.min_offset_m   = self.declare_parameter('min_offset_amplitude_m', 0.50).value
        self.extra_safe_m   = self.declare_parameter('extra_safe_m', 0.50).value
        self.n_sample       = self.declare_parameter('n_sample', 60).value
        self.triplet_period = self.declare_parameter('triplet_period', 0.1).value  # 定时周期 (s)

        # ROI 梯形比例
        self.trap_top_ratio, self.trap_left_ratio  = 0.55, 0.40
        self.trap_right_ratio                      = 0.60
        self.trap_bottom_left_ratio, self.trap_bottom_right_ratio = 0.15, 0.85

        # HSV 红阈值
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])

        # ----- ROS 通信 -----
        qos = 10
        # 图像
        self.create_subscription(Image, '/aiformula_sensing/zed_node/left_image/undistorted', self.image_cb, qos)
        self.pub_dbg   = self.create_publisher(Image, '/processed_image', qos)
        self.pub_red   = self.create_publisher(Int32, '/red_pixels_count', qos)
        self.pub_flag  = self.create_publisher(Bool, '/lane_stop_flag', qos)

        # 三点输出
        self.pub_pt_a = self.create_publisher(Pose2D, '/processed_point_a', qos)
        self.pub_pt_b = self.create_publisher(Pose2D, '/processed_point_b', qos)
        self.pub_pt_c = self.create_publisher(Pose2D, '/processed_point_c', qos)

        # 点云透传（正常行驶时）
        self.pub_pc_center = self.create_publisher(PointCloud2, 'lane_line_center', qos)
        self.pub_pc_left   = self.create_publisher(PointCloud2, 'lane_line_left',   qos)
        self.pub_pc_right  = self.create_publisher(PointCloud2, 'lane_line_right',  qos)
        for n, cb in [('center', self.pc_center_cb),
                      ('left',   self.pc_left_cb),
                      ('right',  self.pc_right_cb)]:
            topic = f'/aiformula_perception/lane_line_publisher/lane_lines/{n}'
            self.create_subscription(PointCloud2, topic, cb, qos)

        # 状态
        self.bridge        = CvBridge()
        self.cur_h, self.cur_w, self.cx = 0, 0, 0
        self.avoiding      = False          # 正在执行避障路径
        self.sending_ok    = False          # 红像素已低，可发送剩余采样点
        self.segments: list[list[np.ndarray]] = []   # [[p0,p1,p2], ...]
        self.seg_idx       = 0
        self.prev_origin   = None           # 上批次第 2 个全局点
        self.lane_forward  = True           # 点云透传开关

        # 定时器：三点分批发送
        self.create_timer(self.triplet_period, self.timer_cb)

    # ────────── 点云透传 ──────────
    def pc_center_cb(self, msg): self.lane_forward and self.pub_pc_center.publish(msg)
    def pc_left_cb(self, msg):   self.lane_forward and self.pub_pc_left.publish(msg)
    def pc_right_cb(self, msg):  self.lane_forward and self.pub_pc_right.publish(msg)

    # ────────── 图像回调 ──────────
    def image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e)); return

        self.cur_h, self.cur_w = img.shape[:2]
        self.cx = self.cur_w // 2
        vis = img.copy()

        # ---------- 计算 ROI & 红色像素 ----------
        roi_cnt = self.get_trap_cnt()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        total_red = int(cv2.countNonZero(mask))

        roi_mask = np.zeros_like(mask); cv2.fillPoly(roi_mask, [roi_cnt], 255)
        col_mask = cv2.bitwise_and(mask, roi_mask)
        rect_red = int(cv2.countNonZero(col_mask))
        self.pub_red.publish(Int32(data=rect_red))

        # ---------- 状态机 ----------
        # ❶ 高于进入阈值 → 重新规划
        if rect_red > self.red_thresh_px:
            self.plan_new_path(col_mask)
            self.sending_ok = False   # 暂停分批发送

        # ❷ 红色降低至 resume 阈值 → 允许分批发送
        elif self.avoiding and rect_red < self.red_resume_px:
            self.sending_ok = True

        # ---------- 绘制调试 ----------
        self.draw_debug(vis, roi_cnt, rect_red, total_red)
        try:
            dbg = self.bridge.cv2_to_imgmsg(vis, 'bgr8'); dbg.header = msg.header
            self.pub_dbg.publish(dbg)
        except CvBridgeError:
            pass

    # ────────── 新路径规划 ──────────
    def plan_new_path(self, col_mask):
        ys, xs = np.where(col_mask > 0)
        left_cnt, right_cnt = np.sum(xs < self.cx), np.sum(xs >= self.cx)

        if left_cnt == 0 and right_cnt > 0:
            d, amp_m = +1, self.min_offset_m
        elif right_cnt == 0 and left_cnt > 0:
            d, amp_m = -1, self.min_offset_m
        else:
            if left_cnt < right_cnt:
                d = +1; dist_px = self.cx - xs[xs < self.cx].min()
            else:
                d = -1; dist_px = xs[xs >= self.cx].max() - self.cx
            amp_m = max(dist_px * self.m_per_px + self.extra_safe_m,
                        self.min_offset_m)

        # 起终点（像素 → 车体坐标）
        p0_px, p1_px = (self.cx, self.cur_h), (self.cx, int(self.cur_h * self.trap_top_ratio))
        p0_m, p1_m   = self.px2m(p0_px), self.px2m(p1_px)

        # 生成整条 B-样条路径
        path_m = self.build_nurbs(p0_m, p1_m, d, amp_m,
                                  n_ctrl=7, n_sample=self.n_sample, p=3)

        # ------ 选 3 个关键点并即时发布 ------
        desired_x = 0.3
        p_start = self.interp_x(path_m, desired_x)
        if p_start is None:                  # ← 这里不再用 or
            p_start = path_m[1]

        ys_path = [p[1] for p in path_m]
        idx_extreme = int(np.argmax(ys_path)) if d == 1 else int(np.argmin(ys_path))
        p_mid = path_m[idx_extreme]
        p_end = path_m[-1]

        for pub, p in zip((self.pub_pt_a, self.pub_pt_b, self.pub_pt_c),
                          (p_start, p_mid, p_end)):
            pub.publish(Pose2D(x=float(p[0]), y=float(p[1]), theta=0.0))

        # ------ 保存剩余采样点（完整路径） ------
        self.segments   = self.slice3_nonoverlap(path_m)
        self.seg_idx    = 0
        self.prev_origin = None
        self.avoiding   = True
        self.lane_forward = False
        self.pub_flag.publish(Bool(data=True))

    # ────────── 定时器：三点分批发送 ──────────
    def timer_cb(self):
        # 未在避障 / 还未允许发送 → 直接返回
        if not (self.avoiding and self.sending_ok):
            return

        # 全部发送完毕 → 退出避障
        if self.seg_idx >= len(self.segments):
            self.exit_avoid()
            return

        seg = self.segments[self.seg_idx]            # 3 × 全局点
        if self.prev_origin is None:
            transformed = seg                        # 首批：全局坐标
        else:
            transformed = [p - self.prev_origin for p in seg]  # 后续：相对坐标

        for pub, p in zip((self.pub_pt_a, self.pub_pt_b, self.pub_pt_c), transformed):
            pub.publish(Pose2D(x=float(p[0]), y=float(p[1]), theta=0.0))

        self.prev_origin = seg[1].copy()             # 更新原点
        self.seg_idx += 1

    # ────────── 退出避障 ──────────
    def exit_avoid(self):
        self.avoiding     = False
        self.sending_ok   = False
        self.segments.clear()
        self.prev_origin  = None
        self.lane_forward = True
        self.pub_flag.publish(Bool(data=False))

    # ────────── B-样条生成 ──────────
    @staticmethod
    def build_nurbs(p0, p1, d, amp, n_ctrl=7, n_sample=60, p=3):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        v = p1 - p0
        v /= (np.linalg.norm(v) + 1e-8)
        perp = np.array([-v[1], v[0]])
        ctrl = [p0.copy()]
        for i in range(1, n_ctrl - 1):
            t = i / (n_ctrl - 1)
            offset = amp * np.sin(np.pi * t)
            ctrl.append((1 - t) * p0 + t * p1 + d * offset * perp)
        ctrl.append(p1.copy()); ctrl = np.array(ctrl)

        # 非均匀 knots
        k = p + 1
        prefix, suffix = [0.0] * k, [1.0] * k
        n_internal = len(ctrl) + p + 1 - k * 2
        inner = [(i / (n_internal + 1)) ** 0.5 for i in range(1, n_internal + 1)]
        tck = (np.r_[prefix, inner, suffix],
               [ctrl[:, 0], ctrl[:, 1]], p)

        u = np.linspace(0.0, 1.0, n_sample)
        xs, ys = si.splev(u, tck)
        return [np.array([xs[i], ys[i]], float) for i in range(len(xs))]

    # ────────── 工具函数 ──────────
    def get_trap_cnt(self):
        h, w = self.cur_h, self.cur_w
        tl = (int(w * self.trap_left_ratio),  int(h * self.trap_top_ratio))
        tr = (int(w * self.trap_right_ratio), int(h * self.trap_top_ratio))
        bl = (int(w * self.trap_bottom_left_ratio),  h)
        br = (int(w * self.trap_bottom_right_ratio), h)
        return np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))

    def interp_x(self, path, x_desired):
        for i in range(1, len(path)):
            x0, y0 = path[i - 1]; x1, y1 = path[i]
            if (x0 <= x_desired <= x1) or (x1 <= x_desired <= x0):
                t = (x_desired - x0) / (x1 - x0 + 1e-12)
                return np.array([x_desired, y0 + t * (y1 - y0)])
        return None

    @staticmethod
    def slice3_nonoverlap(pts):
        return [pts[i:i + 3] for i in range(0, len(pts) - 2, 3)]

    def px2m(self, p): return np.array([(self.cur_h - p[1]) * self.m_per_px,
                                        (self.cx      - p[0]) * self.m_per_px], float)

    # ---------- 调试图像 ----------
    def draw_debug(self, vis, roi_cnt, rect_red, total_red):
        cv2.polylines(vis, [roi_cnt], True, (0, 255, 255), 2)
        cv2.line(vis, (self.cx, 0), (self.cx, self.cur_h), (0, 255, 0), 2)
        status = "Avoiding" if self.avoiding else "Normal"
        cv2.putText(vis, f"{status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if self.avoiding else (0, 255, 0), 2)
        cv2.putText(vis, f"Total:{total_red}  ROI:{rect_red}",
                    (10, self.cur_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)


# ────────── main ──────────
def main(args=None):
    rclpy.init(args=args)
    node = RedAlertDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
