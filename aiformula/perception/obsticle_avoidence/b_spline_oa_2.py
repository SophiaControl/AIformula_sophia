#!/usr/bin/env python3
"""
bspline_planner_nosw.py  ——  无“滑动窗”的非均匀 B-样条避障
订阅:  /camera/image_raw  /lane_points  /odom
发布:  /processed_image  /bspline_path  /path_point  /red_pixels_count
"""
import rclpy, math, cv2, numpy as np
from   rclpy.node            import Node
from   sensor_msgs.msg       import Image
from   geometry_msgs.msg     import Pose2D, PoseStamped
from   nav_msgs.msg          import Path, Odometry
from   std_msgs.msg          import Int32
from   cv_bridge             import CvBridge
from   scipy.interpolate     import BSpline
from   collections           import deque

class BSplinePlannerNoSW(Node):
    def __init__(self):
        super().__init__('bspline_planner_nosw')

        # ───────── 参数 ─────────
        self.declare_parameter('camera_topic',     '/camera/image_raw')
        self.declare_parameter('lane_point_topic', '/lane_points')
        self.declare_parameter('odom_topic',       '/odom')
        self.declare_parameter('lane_is_metric',   False)
        self.declare_parameter('pix2m',            0.005)
        self.declare_parameter('roi_h_ratio',      0.6)
        self.declare_parameter('roi_l_ratio',      0.2)
        self.declare_parameter('roi_r_ratio',      0.8)
        self.declare_parameter('base_offset_px',   50)
        self.declare_parameter('extra_margin_m',   0.5)
        self.declare_parameter('deg',              3)
        self.declare_parameter('red_thresh_px',    200)

        gp = self.get_parameter
        self.cam_t   = gp('camera_topic').value
        self.lane_t  = gp('lane_point_topic').value
        self.odom_t  = gp('odom_topic').value
        self.metric  = gp('lane_is_metric').value
        self.pix2m   = gp('pix2m').value
        self.roiH    = gp('roi_h_ratio').value
        self.lratio  = gp('roi_l_ratio').value
        self.rratio  = gp('roi_r_ratio').value
        self.base_px = gp('base_offset_px').value
        self.extra_px= gp('extra_margin_m').value / self.pix2m
        self.deg     = gp('deg').value
        self.red_th  = gp('red_thresh_px').value

        # ───────── ROS I/O ─────────
        self.create_subscription(Image,    self.cam_t,  self.img_cb,  10)
        self.create_subscription(Pose2D,   self.lane_t, self.lane_cb, 10)
        self.create_subscription(Odometry, self.odom_t, self.odom_cb, 10)

        self.proc_pub = self.create_publisher(Image, '/processed_image', 10)
        self.red_pub  = self.create_publisher(Int32, '/red_pixels_count',10)
        self.path_pub = self.create_publisher(Path,  '/bspline_path',    10)
        self.pt_pub   = self.create_publisher(Pose2D,'/path_point',      10)

        self.bridge = CvBridge()
        self.low_r  = np.array([0,120,70],np.uint8)
        self.up_r   = np.array([10,255,255],np.uint8)

        # 状态
        self.lane_buf = deque(maxlen=200)          # 保存足够多的车道点
        self.heading  = np.array([0,-1],float)     # 起点切线方向
        self.curve_grps=[]; self.grp_idx=0; self.timer=None

    # ───────── Lane 单点回调 ─────────
    def lane_cb(self, msg:Pose2D):
        if self.metric:
            self.lane_buf.append((msg.x/self.pix2m, msg.y/self.pix2m))
        else:
            self.lane_buf.append((msg.x, msg.y))

    # ───────── Odom 回调 ─────────
    def odom_cb(self, msg:Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y),
                         1-2*(q.y*q.y+q.z*q.z))
        self.heading = np.array([math.sin(yaw), -math.cos(yaw)])

    # ───────── B-样条生成 ─────────
    def make_bspline(self, ctrl:np.ndarray):
        k=self.deg; n=len(ctrl)
        knot=np.concatenate([np.zeros(k), np.linspace(0,1,n-k+1), np.ones(k)])
        return BSpline(knot, ctrl, k)

    def build_curve(self, p0:np.ndarray, lane_px, dir_sign:int):
        if len(lane_px) < self.deg+1:   # 至少 p+1 个控制点
            return None
        d = abs(self.base_px + self.extra_px)
        v0 = self.heading / (np.linalg.norm(self.heading)+1e-6)
        # 使用全部 lane 点
        lane = np.asarray(lane_px, float)

        # 尾段切线方向
        v_end = lane[-1]-lane[-2] if len(lane)>=2 else np.array([0,-1])
        v_end /= (np.linalg.norm(v_end)+1e-6)

        ctrl = [p0, p0+v0*d, *lane[:-1], lane[-1]-v_end*d, lane[-1]]

        amp = dir_sign*(self.base_px+self.extra_px)
        for i in range(1,len(ctrl)-1):
            t=i/(len(ctrl)-1)
            ctrl[i][0]+= amp*math.sin(math.pi*t)

        bs=self.make_bspline(np.array(ctrl))
        return bs(np.linspace(0,1,80))

    # ───────── 3-点组定时发送 ─────────
    def timer_cb(self):
        if self.grp_idx>=len(self.curve_grps):
            self.timer.cancel(); self.timer=None; return
        for p in self.curve_grps[self.grp_idx]:
            self.pt_pub.publish(Pose2D(x=p[0]*self.pix2m,
                                       y=p[1]*self.pix2m,
                                       theta=0.0))
        self.grp_idx+=1

    # ───────── 图像回调 ─────────
    def img_cb(self, msg:Image):
        try:
            img=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except: return
        H,W=img.shape[:2]

        # ROI
        y0=int(H*(1-self.roiH)); xL=int(W*self.lratio); xR=int(W*self.rratio)
        roi=img[y0:H,xL:xR].copy(); h,w=roi.shape[:2]

        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,self.low_r,self.up_r)

        tot=cv2.countNonZero(mask); self.red_pub.publish(Int32(data=tot))
        vis=cv2.addWeighted(roi,1.0,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),0.5,0)
        cv2.line(vis,(w//2,0),(w//2,h),(255,255,0),2)
        p0=np.array([w//2,h],float)

        lane_px=list(self.lane_buf)[::-1]          # 最新点在前
        if tot>self.red_th and len(lane_px)>=self.deg+1:
            xs=np.where(mask>0)[1]
            dir_sign=-1 if (xs>=w//2).sum()>(xs<w//2).sum() else 1
            curve=self.build_curve(p0,lane_px,dir_sign)

            if curve is not None:
                self.curve_grps=[curve[i:i+3] for i in range(len(curve)-2)]
                self.grp_idx=0
                if self.timer: self.timer.cancel()
                self.timer=self.create_timer(0.25,self.timer_cb)

                for i in range(len(curve)-1):
                    cv2.line(vis,tuple(curve[i].astype(int)),
                                 tuple(curve[i+1].astype(int)),
                                 (255,255,100),2)
                for p in curve[::6]:
                    cv2.circle(vis,tuple(p.astype(int)),4,(255,255,100),-1)

                path=Path(); path.header=msg.header
                for p in curve:
                    ps=PoseStamped(); ps.header=msg.header
                    ps.pose.position.x=p[0]*self.pix2m
                    ps.pose.position.y=p[1]*self.pix2m
                    path.poses.append(ps)
                self.path_pub.publish(path)

        cv2.rectangle(img,(xL,y0),(xR,H),(0,255,255),2)
        img[y0:H,xL:xR]=vis
        self.proc_pub.publish(self.bridge.cv2_to_imgmsg(img,'bgr8'))

# ───────── main ─────────
def main(args=None):
    rclpy.init(args=args)
    node=BSplinePlannerNoSW()
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()
