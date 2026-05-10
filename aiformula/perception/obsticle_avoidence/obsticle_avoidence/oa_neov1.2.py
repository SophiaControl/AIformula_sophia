#!/usr/bin/env python3
import rclpy
import cv2
import numpy as np
import scipy.interpolate as si

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge


class ConeMaskAvoider(Node):
    def __init__(self):
        super().__init__('cone_mask_avoider')

        # ===== 参数 =====
        self.m_per_px        = float(self.declare_parameter('meters_per_pixel', 0.01).value)
        self.min_offset_m    = float(self.declare_parameter('min_offset_amplitude_m', 1.2).value)
        self.single_side_amp = float(self.declare_parameter('single_side_amp', 0.30).value)
        self.extra_safe_m    = float(self.declare_parameter('extra_safe_m', 0.40).value)

        # 进入/退出阈值（像素数）
        self.cone_thresh_px  = int(self.declare_parameter('cone_thresh_px', 1000).value)  # enter
        self.cone_resume_px  = int(self.declare_parameter('cone_resume_px', 500).value)   # exit

        # 采样与三点发送
        self.n_sample        = int(self.declare_parameter('n_sample', 45).value)
        self.triplet_period  = float(self.declare_parameter('triplet_period', 0.1).value)

        # ROI 梯形比例（沿用你之前的写法）
        self.trap_top_ratio  = float(self.declare_parameter('trap_top_ratio', 0.5).value)
        self.trap_left_ratio, self.trap_right_ratio = 0.373, 0.625
        self.trap_bottom_left_ratio, self.trap_bottom_right_ratio = 0.15, 0.85

        # 可视化：mask 透明叠加（可关）
        self.overlay_mask    = bool(self.declare_parameter('overlay_mask', True).value)

        # ===== ROS IO =====
        self.bridge = CvBridge()

        # 底图：cone 节点画好 bbox 的图
        self.create_subscription(
            Image,
            '/cone_bbox_image',   # 如不一致，只改这里
            self.cone_image_cb,
            10
        )

        # cone mask（mono8 0/255）
        self.create_subscription(
            Image,
            '/cone_mask',
            self.mask_cb,
            10
        )

        # 车道线点云透传
        for n, cb in [('center', self.pc_center_cb), ('left', self.pc_left_cb), ('right', self.pc_right_cb)]:
            self.create_subscription(
                PointCloud2,
                f'/aiformula_perception/lane_line_publisher/lane_lines/{n}',
                cb,
                10
            )

        self.pub_pc_center = self.create_publisher(PointCloud2, 'lane_line_center', 10)
        self.pub_pc_left   = self.create_publisher(PointCloud2, 'lane_line_left', 10)
        self.pub_pc_right  = self.create_publisher(PointCloud2, 'lane_line_right', 10)

        self.pub_pt_a = self.create_publisher(Pose2D, '/processed_point_a', 10)
        self.pub_pt_b = self.create_publisher(Pose2D, '/processed_point_b', 10)
        self.pub_pt_c = self.create_publisher(Pose2D, '/processed_point_c', 10)

        self.pub_dbg_img   = self.create_publisher(Image, '/processed_image', 10)
        self.pub_cnt       = self.create_publisher(Int32, '/cone_pixels_count', 10)
        self.pub_stop_flag = self.create_publisher(Bool, '/lane_stop_flag', 10)

        self.triplet_pubs = [self.pub_pt_a, self.pub_pt_b, self.pub_pt_c]

        # ===== 状态缓存 =====
        self.latest_vis_img = None
        self.latest_vis_hdr = None
        self.latest_mask    = None

        self.h = self.w = self.cx = 0
        self.lane_forward = True

        self.current_path = []
        self.path_idx = 0
        self.reference_point = None
        self.timer_handle = None

        # 保存最近一次路径的像素坐标用于持续显示（避免“图像更新但路径不见”）
        self.last_path_px = None

        # 用定时器驱动绘制/判定（不依赖回调顺序）
        self.create_timer(0.03, self.tick)  # ~33Hz

    # ---------------- 回调 ----------------
    def cone_image_cb(self, msg: Image):
        try:
            self.latest_vis_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_vis_hdr = msg.header
        except Exception:
            self.latest_vis_img = None
            self.latest_vis_hdr = None

    def mask_cb(self, msg: Image):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        except Exception:
            self.latest_mask = None

    def pc_center_cb(self, m):
        if self.lane_forward:
            self.pub_pc_center.publish(m)

    def pc_left_cb(self, m):
        if self.lane_forward:
            self.pub_pc_left.publish(m)

    def pc_right_cb(self, m):
        if self.lane_forward:
            self.pub_pc_right.publish(m)

    # ---------------- 坐标 ----------------
    def px2robot(self, p):
        return np.array([(self.h - p[1]) * self.m_per_px,
                         (self.cx - p[0]) * self.m_per_px], float)

    def robot2px(self, p):
        return (int(self.cx - p[1] / self.m_per_px),
                int(self.h - p[0] / self.m_per_px))

    # ---------------- tick 主循环 ----------------
    def tick(self):
        if self.latest_vis_img is None or self.latest_mask is None:
            return

        vis = self.latest_vis_img.copy()
        mask = self.latest_mask

        self.h, self.w = vis.shape[:2]
        self.cx = self.w // 2

        if mask.shape[:2] != (self.h, self.w):
            mask = cv2.resize(mask, (self.w, self.h), cv2.INTER_NEAREST)

        # ROI 梯形
        tl = (int(self.w * self.trap_left_ratio), int(self.h * self.trap_top_ratio))
        tr = (int(self.w * self.trap_right_ratio), int(self.h * self.trap_top_ratio))
        bl = (int(self.w * self.trap_bottom_left_ratio), self.h)
        br = (int(self.w * self.trap_bottom_right_ratio), self.h)
        roi_cnt = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))

        roi_mask = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(roi_mask, [roi_cnt.reshape(-1, 2)], 255)

        col_mask = cv2.bitwise_and(mask, roi_mask)  # ROI 内 cone mask
        roi_total = int(cv2.countNonZero(col_mask))
        self.pub_cnt.publish(Int32(data=roi_total))

        # 统计左右像素数（你判定逻辑要用的）
        cm = (col_mask > 0)
        L_cnt = int(np.count_nonzero(cm[:, :self.cx]))
        R_cnt = int(np.count_nonzero(cm[:, self.cx:]))

        # 可选：叠加 mask（红色半透明）
        if self.overlay_mask:
            overlay = vis.copy()
            overlay[col_mask > 0] = (0, 0, 255)
            vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0.0)

        # 持续画最近路径（即便不在规划分支）
        if self.last_path_px is not None and len(self.last_path_px) >= 2:
            cv2.polylines(vis, [self.last_path_px], False, (255, 255, 0), 2)

        # ===== 分批发送期间：不重新规划，但持续画图与发布 =====
        if self.timer_handle is not None:
            self.draw_hud(vis, roi_cnt, roi_total, L_cnt, R_cnt)
            self.publish_dbg(vis)
            return

        # ===== 进入避障 =====
        if roi_total >= self.cone_thresh_px:
            if self.lane_forward:
                self.lane_forward = False
                self.pub_stop_flag.publish(Bool(data=True))

            self.compute_path(col_mask, L_cnt, R_cnt, vis)  # 内部会更新 last_path_px 并发布 A/B/C
            self.draw_hud(vis, roi_cnt, roi_total, L_cnt, R_cnt)
            self.publish_dbg(vis)
            return

        # ===== 退出避障：低于退出阈值 + 有路径才退出并开始分批发送剩余点 =====
        if (not self.lane_forward) and roi_total <= self.cone_resume_px and self.current_path:
            self.timer_handle = self.create_timer(self.triplet_period, self.broadcast_triplet)

        self.draw_hud(vis, roi_cnt, roi_total, L_cnt, R_cnt)
        self.publish_dbg(vis)

    # ---------------- 规划（保持你新逻辑：像素少的一侧绕；边界用 left max / right min） ----------------
    def compute_path(self, col_mask, L_cnt, R_cnt, vis):
        ys, xs = np.where(col_mask > 0)
        if xs.size == 0:
            return

        left_xs  = xs[xs < self.cx]
        right_xs = xs[xs >= self.cx]

        # 方向：向像素数少的一侧绕（更空）
        if L_cnt <= R_cnt:
            d = +1  # 左侧更空 => 左绕
            if left_xs.size == 0:
                amp_m = self.single_side_amp
            else:
                max_x_left = int(left_xs.max())              # 左侧靠近中心的边界：max
                dist_px = max(0, self.cx - max_x_left)
                amp_m = max(dist_px * self.m_per_px + self.extra_safe_m, self.min_offset_m)
        else:
            d = -1  # 右侧更空 => 右绕
            if right_xs.size == 0:
                amp_m = self.single_side_amp
            else:
                min_x_right = int(right_xs.min())            # 右侧靠近中心的边界：min
                dist_px = max(0, min_x_right - self.cx)
                amp_m = max(dist_px * self.m_per_px + self.extra_safe_m, self.min_offset_m)

        # 起点终点（机器人坐标）
        p0 = self.px2robot((self.cx, self.h))
        p1 = self.px2robot((self.cx, int(self.h * self.trap_top_ratio)))

        path = self.build_nurbs(p0, p1, d, amp_m, n_sample=self.n_sample)
        self.current_path = path

        # A/B/C 三点：A 取靠近起点的点；B 取 y 的极值；C 取终点
        pA = path[1]
        ys_path = [p[1] for p in path]
        mid_idx = int(np.argmax(ys_path)) if d == +1 else int(np.argmin(ys_path))
        pB = path[mid_idx]
        pC = path[-1]

        for pub, p in zip(self.triplet_pubs, [pA, pB, pC]):
            pub.publish(Pose2D(x=float(p[0]), y=float(p[1]), theta=0.0))

        self.reference_point = pB.copy()
        self.path_idx = 2

        # 更新 debug 路径（像素坐标），持续显示
        px_path = [self.robot2px(p) for p in path]
        self.last_path_px = np.array(px_path, dtype=np.int32).reshape(-1, 1, 2)

        # 在当前 vis 上也画一次
        cv2.polylines(vis, [self.last_path_px], False, (255, 255, 0), 2)

    # ---------------- 分批发送剩余点 ----------------
    def broadcast_triplet(self):
        # ---------- 1) 结束条件：点发完 ----------
        if self.path_idx >= len(self.current_path):
            if self.timer_handle is not None:
                self.timer_handle.cancel()
                self.timer_handle = None

            # 清状态
            self.current_path = []
            self.last_path_px = None
            self.path_idx = 0
            self.reference_point = None

            self.lane_forward = True
            self.pub_stop_flag.publish(Bool(data=False))
            return

        # ---------- 2) 取 3 个点（不够就补最后一个） ----------
        chunk = self.current_path[self.path_idx:self.path_idx + 3]
        while len(chunk) < 3:
            chunk.append(chunk[-1])

        # ---------- 3) 转为“当前局部点”并发布 ----------
        if self.reference_point is None:
            rel_chunk = chunk
        else:
            rel_chunk = [p - self.reference_point for p in chunk]

        for pub, p in zip(self.triplet_pubs, rel_chunk):
            pub.publish(Pose2D(x=float(p[0]), y=float(p[1]), theta=0.0))

        # ---------- 4) 滑动更新原点 + 推进索引 ----------
        self.reference_point = chunk[1].copy()
        self.path_idx += 3


    # ---------------- B样条（稳定、可复用） ----------------
    # @staticmethod
    # def build_nurbs(p0, p1, d, amp, n_sample=45, n_ctrl=7, p=3):
    #     p0 = np.asarray(p0, float)
    #     p1 = np.asarray(p1, float)

    #     v = p1 - p0
    #     v /= (np.linalg.norm(v) + 1e-12)
    #     perp = np.array([-v[1], v[0]], dtype=float)

    #     ctrl = [p0.copy()]
    #     for i in range(1, n_ctrl - 1):
    #         t = i / (n_ctrl - 1)
    #         offset = amp * ((1 - t)**0.7)
    #         ctrl.append((1 - t) * p0 + t * p1 + d * offset * perp)
    #     ctrl.append(p1.copy())
    #     ctrl = np.array(ctrl)

    #     # clamped knot
    #     prefix = [0.0] * (p + 1)
    #     suffix = [1.0] * (p + 1)
    #     total_knots = n_ctrl + p + 1
    #     n_internal = total_knots - len(prefix) - len(suffix)
    #     inner = [(i / (n_internal + 1)) ** 2 for i in range(1, n_internal + 1)]
    #     knots = np.concatenate((prefix, inner, suffix))

    #     t_vals = np.linspace(0.0, 1.0, int(n_sample))
    #     xs, ys = si.splev(t_vals, (knots, [ctrl[:, 0], ctrl[:, 1]], p))
    #     xs[-1], ys[-1] = p1[0], p1[1]
    #     return [np.array([xs[i], ys[i]], float) for i in range(len(t_vals))]

    #v1.20 末端非均匀取样
    @staticmethod
    def build_nurbs(p0, p1, d, amp, n_sample=45, n_ctrl=7, p=3, end_dense_gamma=3.0):
        """
        end_dense_gamma > 1: make samples denser near t=1 (tail).
        Typical: 2~4. Set 1.0 to recover uniform sampling.
        """
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)

        v = p1 - p0
        v /= (np.linalg.norm(v) + 1e-12)
        perp = np.array([-v[1], v[0]], dtype=float)

        ctrl = [p0.copy()]
        for i in range(1, n_ctrl - 1):
            t = i / (n_ctrl - 1)
            offset = amp * ((1 - t) ** 0.7)
            ctrl.append((1 - t) * p0 + t * p1 + d * offset * perp)
        ctrl.append(p1.copy())
        ctrl = np.array(ctrl)

        # clamped knot
        prefix = [0.0] * (p + 1)
        suffix = [1.0] * (p + 1)
        total_knots = n_ctrl + p + 1
        n_internal = total_knots - len(prefix) - len(suffix)
        inner = [(i / (n_internal + 1)) ** 2 for i in range(1, n_internal + 1)]
        knots = np.concatenate((prefix, inner, suffix))

        # --------- non-uniform sampling: denser near t=1 ----------
        N = int(n_sample)
        u = np.linspace(0.0, 1.0, N)
        g = float(end_dense_gamma)
        if g <= 1.0:
            t_vals = u
        else:
            t_vals = 1.0 - np.power(1.0 - u, g)   # dense near 1
        # ---------------------------------------------------------

        xs, ys = si.splev(t_vals, (knots, [ctrl[:, 0], ctrl[:, 1]], p))
        xs[-1], ys[-1] = p1[0], p1[1]
        return [np.array([xs[i], ys[i]], float) for i in range(len(t_vals))]

    # ---------------- 画 HUD：恢复你之前的文字/ROI/中心线 ----------------
    def draw_hud(self, vis, roi_cnt, roi_total, L_cnt, R_cnt):
        # ROI + center line
        cv2.polylines(vis, [roi_cnt], True, (0, 255, 255), 2)
        cv2.line(vis, (self.cx, 0), (self.cx, self.h), (0, 255, 0), 2)

        # 状态文字（你说“开始避障的文字”就是这个）
        playing = (not self.lane_forward)
        txt = "Avoiding..." if playing else "Forward"
        cv2.putText(
            vis, txt, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255) if playing else (0, 255, 0), 2
        )

        # 数值信息
        cv2.putText(
            vis,
            f"ROI_total={roi_total}  L={L_cnt}  R={R_cnt}",
            (10, self.h - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )
        cv2.putText(
            vis,
            f"th_enter={self.cone_thresh_px}  th_exit={self.cone_resume_px}",
            (10, self.h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

    def publish_dbg(self, vis):
        try:
            msg = self.bridge.cv2_to_imgmsg(vis, 'bgr8')
            if self.latest_vis_hdr is not None:
                msg.header = self.latest_vis_hdr
            self.pub_dbg_img.publish(msg)
        except Exception as e:
            self.get_logger().error(str(e))


def main(args=None):
    rclpy.init(args=args)
    node = ConeMaskAvoider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.timer_handle is not None:
            node.timer_handle.cancel()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
