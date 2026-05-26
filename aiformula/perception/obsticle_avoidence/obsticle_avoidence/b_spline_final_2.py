#!/usr/bin/env python3
# coding: utf-8

import rclpy, cv2, numpy as np, scipy.interpolate as si
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge


class RedAlertDetector(Node):
    # ────────────────── 初始化 ──────────────────
    def __init__(self):
        super().__init__('red_alert_detector')

        # ---- 核心参数 ----
        self.m_per_px        = self.declare_parameter('meters_per_pixel', 0.01).value
        self.min_offset_m    = self.declare_parameter('min_offset_amplitude_m', 0.50).value
        self.red_thresh_px   = 900                      # “需要重新规划”阈值
        self.single_side_amp = 0.30
        self.extra_safe_m    = 0.50
        self.triplet_period  = self.declare_parameter('triplet_period', 0.1).value  # ✦ 定时周期

        # ROI 比例
        self.trap_top_ratio,  self.trap_left_ratio   = 0.55, 0.40
        self.trap_right_ratio                        = 0.60
        self.trap_bottom_left_ratio, self.trap_bottom_right_ratio = 0.15, 0.85

        # HSV 红阈值
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])

        # ---- ROS 通信 ----
        self.create_subscription(Image, '/aiformula_sensing/zed_node/left_image/undistorted', self.img_cb, 10)
        for n, cb in [('center', self.pc_center_cb),
                      ('left',   self.pc_left_cb),
                      ('right',  self.pc_right_cb)]:
            self.create_subscription(PointCloud2,
                f'/aiformula_perception/lane_line_publisher/lane_lines/{n}', cb, 10)
        self.pub_pc_center = self.create_publisher(PointCloud2, 'lane_line_center', 10)
        self.pub_pc_left   = self.create_publisher(PointCloud2, 'lane_line_left',   10)
        self.pub_pc_right  = self.create_publisher(PointCloud2, 'lane_line_right',  10)

        # ✦ 三个点分别发到独立主题
        self.pub_pt_a = self.create_publisher(Pose2D, '/processed_point_a', 10)
        self.pub_pt_b = self.create_publisher(Pose2D, '/processed_point_b', 10)
        self.pub_pt_c = self.create_publisher(Pose2D, '/processed_point_c', 10)

        # 调试信息
        self.pub_dbg_img = self.create_publisher(Image, '/processed_image', 10)
        self.pub_red_cnt = self.create_publisher(Int32, '/red_pixels_count', 10)
        self.pub_stop_flag = self.create_publisher(Bool, '/lane_stop_flag', 10)

        # ---- 状态 ----
        self.bridge = CvBridge()
        self.h = self.w = self.cx = 0
        self.segments = []          # 路径被切成的 [ [p0,p1,p2], [p3,p4,p5], ... ]
        self.seg_idx  = 0           # 当前正在发送哪一组
        self.prev_origin = None     # 上一组第 2 点（全局坐标），用于动态原点
        self.avoiding = False
        self.lane_forward = True    # True → 正常转发点云

        # ✦ 定时器：负责三点分批发送
        self.create_timer(self.triplet_period, self.timer_cb)

    # ───── 坐标换算 ─────
    def px2robot(self, p):
        """像素 → 车体坐标 (m)"""
        return np.array([(self.h - p[1]) * self.m_per_px,
                         (self.cx - p[0]) * self.m_per_px], dtype=float)

    def robot2px(self, p):
        """车体坐标 → 像素"""
        return (int(self.cx - p[1] / self.m_per_px),
                int(self.h  - p[0] / self.m_per_px))

    # ───── 点云转发 ─────
    def pc_center_cb(self, m): (not self.avoiding) and self.pub_pc_center.publish(m)
    def pc_left_cb(self, m):   (not self.avoiding) and self.pub_pc_left.publish(m)
    def pc_right_cb(self, m):  (not self.avoiding) and self.pub_pc_right.publish(m)

    # ───── 图像处理 ─────
    def img_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(str(e)); return

        self.h, self.w = img.shape[:2]
        self.cx = self.w // 2
        vis = img.copy()

        # ROI 四边形顶/底点
        tl = (int(self.w * self.trap_left_ratio),  int(self.h * self.trap_top_ratio))
        tr = (int(self.w * self.trap_right_ratio), int(self.h * self.trap_top_ratio))
        bl = (int(self.w * self.trap_bottom_left_ratio),  self.h)
        br = (int(self.w * self.trap_bottom_right_ratio), self.h)
        roi_cnt = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))

        # ========== 如果正在避障，只画路径 ==========
        if self.avoiding:
            self.draw_common(vis, roi_cnt, True)
            pts_px = [self.robot2px(p) for seg in self.segments for p in seg]
            cv2.polylines(vis, [np.array(pts_px).reshape(-1, 1, 2)],
                          False, (255, 255, 0), 2)
            self.publish_dbg(vis, msg.header)
            return

        # ========== 常规检测 ==========
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        roi_mask = np.zeros_like(mask); cv2.fillPoly(roi_mask, [roi_cnt], 255)
        col_mask = cv2.bitwise_and(mask, roi_mask)

        rect_red = int(cv2.countNonZero(col_mask))
        total_red = int(cv2.countNonZero(mask))
        self.pub_red_cnt.publish(Int32(data=rect_red))

        # ── 检测到“需要避障” ──
        if rect_red > self.red_thresh_px:
            self.lane_forward = False
            self.pub_stop_flag.publish(Bool(data=True))  # 抬杆信号

            ys, xs = np.where(col_mask > 0)
            left_cnt, right_cnt = np.sum(xs < self.cx), np.sum(xs >= self.cx)

            if left_cnt == 0 and right_cnt > 0:
                offset_dir, amp_m = +1, self.single_side_amp
            elif right_cnt == 0 and left_cnt > 0:
                offset_dir, amp_m = -1, self.single_side_amp
            else:
                if left_cnt < right_cnt:
                    offset_dir = +1
                    dist_px = self.cx - xs[xs < self.cx].min()
                else:
                    offset_dir = -1
                    dist_px = xs[xs >= self.cx].max() - self.cx
                amp_m = max(dist_px * self.m_per_px + self.extra_safe_m,
                            self.min_offset_m)

            # 起终点 (像素 → 车体坐标)
            p0_px, p1_px = (self.cx, self.h), (self.cx, int(self.h * self.trap_top_ratio))
            p0_m, p1_m = self.px2robot(p0_px), self.px2robot(p1_px)

            # 生成整条 B-样条路径
            path_m = self.build_nurbs(p0_m, p1_m, offset_dir, amp_m,
                                      n_ctrl=7, n_sample=25, p=3)

            # ✦ 切片成「3 点一组」，非重叠
            self.segments = self.slice3_nonoverlap(path_m)
            self.seg_idx = 0
            self.prev_origin = None
            self.avoiding = True

            # 画路径 & ROI
            pts_px = [self.robot2px(p) for p in path_m]
            cv2.polylines(vis, [np.array(pts_px).reshape(-1, 1, 2)],
                          False, (255, 255, 0), 2)
            self.draw_common(vis, roi_cnt, True, rect_red, total_red)
            self.publish_dbg(vis, msg.header)
            return

        # ── 无需避障：保持转发 & 调试可视化 ──
        if not self.lane_forward:
            # 曾暂停 → 现在恢复
            self.lane_forward = True
            self.pub_stop_flag.publish(Bool(data=False))

        self.draw_common(vis, roi_cnt, False, rect_red, total_red)
        self.publish_dbg(vis, msg.header)

    # ───── 分批发送定时器 ─────
    def timer_cb(self):
        """定时器：把 self.segments[self.seg_idx] 发布到 /processed_point_a/b/c"""
        if not self.avoiding:
            return

        # 全部发完 → 退出避障模式
        if self.seg_idx >= len(self.segments):
            self.avoiding = False
            self.segments.clear()
            self.prev_origin = None
            return

        seg = self.segments[self.seg_idx]   # 本批 3 个「全局」点
        # —— 动态原点转换 ——  
        if self.prev_origin is None:
            # 第 1 批：直接全局坐标发送
            transformed = seg
        else:
            # 后续批次：全部减去 prev_origin
            transformed = [p - self.prev_origin for p in seg]

        # 依次发布到 a / b / c
        pubs = (self.pub_pt_a, self.pub_pt_b, self.pub_pt_c)
        for pub, p in zip(pubs, transformed):
            pub.publish(Pose2D(x=float(p[0]), y=float(p[1]), theta=0.0))

        # 更新“新的原点” → 当前批次第 2 个全局点
        self.prev_origin = seg[1].copy()
        self.seg_idx += 1

    # ───── B-样条生成 ─────
    @staticmethod
    def build_nurbs(p0, p1, d, amp, n_ctrl=7, n_sample=25, p=3):
        """生成非均匀、均权 B-样条路径，返回 [np.array([x,y]), ...]"""
        p0 = np.array(p0, float)
        p1 = np.array(p1, float)

        # 方向向量及其垂线
        v = p1 - p0
        v /= (np.linalg.norm(v) or 1.0)
        perp = np.array([-v[1], v[0]])

        # 1️⃣ 控制点：首尾固定，中间点沿垂直偏移
        ctrl = [p0.copy()]
        for i in range(1, n_ctrl - 1):
            t = i / (n_ctrl - 1)
            offset = amp * (1 - t**2)
            pt = (1 - t) * p0 + t * p1 + d * offset * perp
            ctrl.append(pt)
        ctrl.append(p1.copy())
        ctrl = np.array(ctrl)

        # 2️⃣ 非均匀节点向量
        prefix = [0.0] * (p + 1)
        suffix = [1.0] * (p + 1)
        total_knots = n_ctrl + p + 1
        n_internal = total_knots - len(prefix) - len(suffix)
        inner = [(i / (n_internal + 1))**0.5 for i in range(1, n_internal + 1)]
        knots = np.concatenate((prefix, inner, suffix))

        # 3️⃣ 均匀采样 t ∈ [0,1]
        t_vals = np.linspace(0.0, 1.0, n_sample)
        x_vals, y_vals = si.splev(t_vals, (knots, [ctrl[:, 0], ctrl[:, 1]], p))

        return [np.array([x_vals[i], y_vals[i]], dtype=float) for i in range(n_sample)]

    # ───── 工具函数 ─────
    @staticmethod
    def slice3_nonoverlap(pts):
        """按 3 点非重叠切片，如 [0,1,2] [3,4,5] ..."""
        groups = []
        for i in range(0, len(pts) - 2, 3):
            groups.append([pts[i], pts[i + 1], pts[i + 2]])
        return groups

    def draw_common(self, vis, roi, playing, rect_cnt=0, total_cnt=0):
        """调试图像：绘 ROI / 文字信息"""
        cv2.line(vis, (self.cx, 0), (self.cx, self.h), (0, 255, 0), 2)
        cv2.polylines(vis, [roi], True, (0, 255, 255), 2)
        txt = "Avoiding..." if playing else "No obstacle"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255) if playing else (0, 255, 0), 2)
        cv2.putText(vis, f"Total:{total_cnt}  ROI:{rect_cnt}",
                    (10, self.h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    def publish_dbg(self, vis, hdr):
        try:
            dbg = self.bridge.cv2_to_imgmsg(vis, 'bgr8')
            dbg.header = hdr
            self.pub_dbg_img.publish(dbg)
        except Exception as e:
            self.get_logger().error(str(e))


# ───── ROS 入口 ─────
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
