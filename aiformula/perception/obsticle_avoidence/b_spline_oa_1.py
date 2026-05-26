#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import scipy.interpolate as si  # 用于 B 样条拟合

class RedAlertDetector(Node):
    def __init__(self):
        super().__init__('red_alert_detector')
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.processed_image_publisher = self.create_publisher(Image, '/processed_image', 10)
        self.red_pixels_publisher = self.create_publisher(Int32, '/red_pixels_count', 10)
        self.pub_path_point = self.create_publisher(Pose2D, 'path_point', 10)

        self.bridge = CvBridge()
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])

        # 矩形区域：上边在图像高度的40%，左右10%-90%
        self.trap_top_ratio = 0.4
        self.trap_left_ratio = 0.1
        self.trap_right_ratio = 0.9

        self.offset_amount = 50  # 基础偏移（像素或对应单位）
        self.extra_offset = 0.5  # 额外偏移，现实中 0.5 m

        self.curve_groups = []
        self.group_index = 0
        self.group_timer = None

    def generate_bspline_curve(self, mid_bottom, mid_top, offset_dir, num_control=7, num_sample=20):
        mid_bottom = np.array(mid_bottom, dtype=float)
        mid_top = np.array(mid_top, dtype=float)
        v = mid_top - mid_bottom
        norm = np.linalg.norm(v) or 1.0
        v_norm = v / norm
        perp = np.array([-v_norm[1], v_norm[0]])

        control_pts = []
        for i in range(num_control):
            t = i / (num_control - 1)
            base_pt = (1 - t) * mid_bottom + t * mid_top
            # 原 offset + 额外 0.5 单位
            base_offset = self.offset_amount * np.sin(np.pi * t)
            offset = offset_dir * (base_offset + self.extra_offset)
            control_pts.append(base_pt + offset * perp)

        control_pts = np.array(control_pts)
        x, y = control_pts[:,0], control_pts[:,1]
        tck, _ = si.splprep([x, y], s=0)
        u_new = np.linspace(0, 1, num_sample)
        x_new, y_new = si.splev(u_new, tck)
        return [np.array([x_new[i], y_new[i]]) for i in range(len(x_new))]

    def create_curve_groups(self, pts):
        return [[pts[i], pts[i+1], pts[i+2]] for i in range(len(pts)-2)]

    def group_timer_callback(self):
        if self.group_index < len(self.curve_groups):
            for pt in self.curve_groups[self.group_index]:
                msg = Pose2D(x=pt[0], y=pt[1], theta=0.0)
                self.pub_path_point.publish(msg)
            self.get_logger().info(f"Published group {self.group_index+1}/{len(self.curve_groups)}")
            self.group_index += 1
        else:
            if self.group_timer:
                self.group_timer.cancel()
                self.group_timer = None

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"{e}")
            return

        h, w = img.shape[:2]
        vis = img.copy()

        # 红色掩码
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        red_ov = np.zeros_like(img); red_ov[:] = (0,0,255)
        vis = cv2.addWeighted(vis,1.0, cv2.bitwise_and(red_ov, cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)),0.5,0)

        # 左右统计
        left, right = mask[:,:w//2], mask[:,w//2:]
        lcnt, rcnt, tcnt = cv2.countNonZero(left), cv2.countNonZero(right), cv2.countNonZero(mask)
        self.red_pixels_publisher.publish(Int32(data=tcnt))
        cv2.line(vis, (w//2,0),(w//2,h),(255,255,0),2)

        # 矩形区域
        tl = (int(w*self.trap_left_ratio), int(h*self.trap_top_ratio))
        tr = (int(w*self.trap_right_ratio), int(h*self.trap_top_ratio))
        br, bl = (w,h), (0,h)
        pts = np.array([[tl,tr,br,bl]],dtype=np.int32)
        cv2.polylines(vis, pts, True, (0,255,255),2)
        mask_rect = np.zeros((h,w),dtype=np.uint8)
        cv2.fillPoly(mask_rect, pts, 255)

        # 碰撞检测
        cm = cv2.bitwise_and(mask, mask_rect)
        cc = cv2.countNonZero(cm)
        cnts,_ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0,255,0),2)

        mb, mt = (w//2,h), ((tl[0]+tr[0])//2, tl[1])
        cv2.line(vis, mb, mt, (0,255,0),2)

        # 生成并绘制避碰路径
        if cc>200:
            od = -1 if rcnt>lcnt else 1
            curve = self.generate_bspline_curve(mb, mt, od)
            self.curve_groups = self.create_curve_groups(curve)
            self.group_index = 0
            if self.group_timer: self.group_timer.cancel()
            self.group_timer = self.create_timer(0.5, self.group_timer_callback)

            # 用青蓝色（cyan）绘制曲线和点
            pts_arr = np.array(curve, np.int32).reshape(-1,1,2)
            cv2.polylines(vis, [pts_arr], False, (255,255,0),2)
            for p in curve:
                cv2.circle(vis, (int(p[0]),int(p[1])),5,(255,255,0),-1)

            cv2.putText(vis,"Collision Trajectory Generated",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        else:
            cv2.putText(vis,"No collision",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.putText(vis, f"Total Red:{tcnt}  Rect Red:{cc}", (10,h-20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        try:
            out = self.bridge.cv2_to_imgmsg(vis, 'bgr8')
            out.header = msg.header
            self.processed_image_publisher.publish(out)
        except CvBridgeError as e:
            self.get_logger().error(f"{e}")

def main(args=None):
    rclpy.init(args=args)
    node = RedAlertDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
