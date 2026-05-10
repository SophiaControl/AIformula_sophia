#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import scipy.interpolate as si


class RedAlertDetector(Node):
    def __init__(self):
        super().__init__('red_alert_detector')

        self.image_subscription = self.create_subscription(
            Image,
            '/aiformula_sensing/zed_node/left_image/undistorted',
            self.image_callback,
            10
        )
        self.m_per_px = self.declare_parameter('meters_per_pixel', 0.01).value
        self.safety_margin_m = self.declare_parameter('safety_margin_m', 0.3).value
        self.min_offset_m = self.declare_parameter('min_offset_amplitude_m', 0.5).value

        # ROI 梯形比例
        self.trap_top_ratio = 0.55
        self.trap_left_ratio = 0.40
        self.trap_right_ratio = 0.60
        self.trap_bottom_left_ratio = 0.15
        self.trap_bottom_right_ratio = 0.85

        # HSV 红色
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])

        # ROS 通信

        self.processed_image_publisher = self.create_publisher(Image, '/processed_image', 10)
        self.red_pixels_publisher = self.create_publisher(Int32, '/red_pixels_count', 10)
        self.pub_path_point = self.create_publisher(Pose2D, 'path_point', 10)

        self.bridge = CvBridge()
        self.curve_groups: list[list[np.ndarray]] = []
        self.group_index: int = 0
        self.group_timer = self.create_timer(0.5, self.timer_cb)  # 0.5 s 循环输出



    # ---------- groups equality ----------
    @staticmethod
    def groups_equal(g1, g2, tol=1e-6):
        if len(g1) != len(g2):
            return False
        for a, b in zip(g1, g2):
            if len(a) != len(b):
                return False
            for pa, pb in zip(a, b):
                if not np.allclose(pa, pb, atol=tol):
                    return False
        return True

    # ---------- px↔m ----------
    def px2m(self, pt):
        return np.asarray(pt, float) * self.m_per_px

    def m2px(self, pt):
        return (np.asarray(pt, float) / self.m_per_px).astype(int)

    # ---------- B‑样条 ----------
    def generate_bspline(self, p0, p1, offset_dir, amplitude_m,
                         n_ctrl=7, n_sample=25):
        p0 = np.array(p0); p1 = np.array(p1)
        v = p1 - p0; v_norm = v / (np.linalg.norm(v) or 1.0)
        perp = np.array([-v_norm[1], v_norm[0]])
        ctrl = []
        for i in range(n_ctrl):
            t = i / (n_ctrl - 1)
            base = (1 - t) * p0 + t * p1
            offset = offset_dir * amplitude_m * np.sin(np.pi * t)
            ctrl.append(base + offset * perp)
        ctrl = np.asarray(ctrl)
        tck, _ = si.splprep([ctrl[:, 0], ctrl[:, 1]], s=0)
        u = np.linspace(0, 1, n_sample)
        x_new, y_new = si.splev(u, tck)
        return [np.array([x_new[i], y_new[i]]) for i in range(len(x_new))]

    def make_groups(self, pts):
        if len(pts) < 3:
            pts += [pts[-1]] * (3 - len(pts))
        return [[pts[i], pts[i + 1], pts[i + 2]] for i in range(len(pts) - 2)]

    # ---------- 定时发布 ----------
    def timer_cb(self):
        if not self.curve_groups:
            return
        for p in self.curve_groups[self.group_index]:
            self.pub_path_point.publish(Pose2D(x=float(p[0]), y=float(p[1]), theta=0.0))
        self.group_index = (self.group_index + 1) % len(self.curve_groups)

    # ---------- 图像回调 ----------
    def image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        h, w = img.shape[:2]
        vis = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        total_red = cv2.countNonZero(mask)
        self.red_pixels_publisher.publish(
            Int32(data=int(cv2.countNonZero(mask)))
        )

        # -------- ROI 梯形 --------
        tl = (int(w * self.trap_left_ratio), int(h * self.trap_top_ratio))
        tr = (int(w * self.trap_right_ratio), int(h * self.trap_top_ratio))
        bl = (int(w * self.trap_bottom_left_ratio), h)
        br = (int(w * self.trap_bottom_right_ratio), h)
        trap = np.array([[tl, tr, br, bl]], np.int32)

        mask_roi = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask_roi, trap, 255)
        col_mask = cv2.bitwise_and(mask, mask_roi)
        rect_red = cv2.countNonZero(col_mask)

        # ---------- 可视化 ----------
        cv2.putText(vis, f"Total:{total_red}  Rect:{rect_red}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        center_x = w // 2
        cv2.line(vis, (center_x, 0), (center_x, h), (0, 255, 0), 2)
        cv2.polylines(vis, trap, True, (0, 255, 255), 2)

        p0_px, p1_px = (center_x, h), (center_x, int(h * self.trap_top_ratio))
        p0_m, p1_m = self.px2m(p0_px), self.px2m(p1_px)

        # ---------- 路径生成 ----------
        if rect_red > 1000:
            cv2.putText(vis, "Avoiding...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            ys, xs = np.where(col_mask > 0)
            left_cnt, right_cnt = np.sum(xs < center_x), np.sum(xs >= center_x)
            offset_dir = -1 if right_cnt > left_cnt else 1
            extreme_x = xs.min() if offset_dir == -1 else xs.max()
            margin_px = self.safety_margin_m / self.m_per_px
            amplitude_px = abs(extreme_x - center_x) + margin_px
            amplitude_m = max(amplitude_px * self.m_per_px, self.min_offset_m)
            curve_m = self.generate_bspline(p0_m, p1_m, offset_dir, amplitude_m)
            # 曲线可视化
            curve_px = [tuple(self.m2px(p)) for p in curve_m]
            cv2.polylines(vis, [np.array(curve_px, np.int32).reshape(-1, 1, 2)], False, (255, 255, 0), 2)
        else:
            cv2.putText(vis, "No collision", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            curve_m = [p0_m, p1_m]

        # ---------- 更新曲线组 ----------
        new_groups = self.make_groups(curve_m)
        if not self.groups_equal(new_groups, self.curve_groups):
            self.curve_groups = new_groups
            self.group_index = 0

        # ---------- 发布调试图 ----------
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(vis, 'bgr8')
            debug_msg.header = msg.header
            self.processed_image_publisher.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(str(e))

    # end of image_callback


def main(args=None):
    """ROS 2 entry point"""
    rclpy.init(args=args)
    node = RedAlertDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
