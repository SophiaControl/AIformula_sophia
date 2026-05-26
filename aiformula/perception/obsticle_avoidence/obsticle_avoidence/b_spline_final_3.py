#!/usr/bin/env python3

import rclpy, cv2, numpy as np, scipy.interpolate as si
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge


class RedAlertDetector(Node):
    def __init__(self):
        super().__init__('red_alert_detector')

        # ——— 参数 ———
        self.m_per_px        = self.declare_parameter('meters_per_pixel', 0.01).value
        self.min_offset_m    = self.declare_parameter('min_offset_amplitude_m', 0.50).value
        self.single_side_amp = self.declare_parameter('single_side_amp', 0.30).value
        self.extra_safe_m    = self.declare_parameter('extra_safe_m', 0.40).value
        self.red_thresh_px   = self.declare_parameter('red_thresh_px', 900).value
        self.red_resume_px   = self.declare_parameter('red_resume_px', 50).value
        self.n_sample        = self.declare_parameter('n_sample', 45).value
        self.triplet_period  = self.declare_parameter('triplet_period', 0.1).value

        # ROI
        self.trap_top_ratio  = self.declare_parameter('trap_top_ratio', 0.55).value
        self.trap_left_ratio, self.trap_right_ratio = 0.40, 0.60
        self.trap_bottom_left_ratio, self.trap_bottom_right_ratio = 0.15, 0.85

        # 红色 HSV 阈值
        self.lower_red = np.array([0, 120, 70]); self.upper_red = np.array([10, 255, 255])

        # 通信
        self.create_subscription(Image,'/aiformula_sensing/zed_node/left_image/undistorted',self.img_cb,10)
        for n,cb in [('center',self.pc_center_cb),('left',self.pc_left_cb),('right',self.pc_right_cb)]:
            self.create_subscription(PointCloud2,f'/aiformula_perception/lane_line_publisher/lane_lines/{n}',cb,10)
        self.pub_pc_center=self.create_publisher(PointCloud2,'lane_line_center',10)
        self.pub_pc_left  =self.create_publisher(PointCloud2,'lane_line_left',10)
        self.pub_pc_right =self.create_publisher(PointCloud2,'lane_line_right',10)
        self.pub_pt_a=self.create_publisher(Pose2D,'/processed_point_a',10)
        self.pub_pt_b=self.create_publisher(Pose2D,'/processed_point_b',10)
        self.pub_pt_c=self.create_publisher(Pose2D,'/processed_point_c',10)
        self.pub_dbg_img=self.create_publisher(Image,'/processed_image',10)
        self.pub_red_cnt=self.create_publisher(Int32,'/red_pixels_count',10)
        self.pub_stop_flag=self.create_publisher(Bool,'/lane_stop_flag',10)

        self.bridge=CvBridge()
        self.h=self.w=self.cx=0

        # 状态
        self.lane_forward=True
        self.current_path=[]          # 全局采样点
        self.path_idx=0               # 下一待发送索引
        self.timer_handle=None
        self.triplet_pubs=[self.pub_pt_a,self.pub_pt_b,self.pub_pt_c]
        self.reference_point=None     # 当前局部原点 (np.array 或 None)

    # ------------------------ 坐标转换 ------------------------
    def px2robot(self,p):
        return np.array([(self.h-p[1])*self.m_per_px,(self.cx-p[0])*self.m_per_px])
    def robot2px(self,p):
        return (int(self.cx-p[1]/self.m_per_px),int(self.h-p[0]/self.m_per_px))

    # ------------------------ 点云透传 ------------------------
    def pc_center_cb(self,m):
        if self.lane_forward:self.pub_pc_center.publish(m)
    def pc_left_cb(self,m):
        if self.lane_forward:self.pub_pc_left.publish(m)
    def pc_right_cb(self,m):
        if self.lane_forward:self.pub_pc_right.publish(m)

    # ---------------------------- 主回调 ----------------------------
    def img_cb(self,msg:Image):
        if self.timer_handle is not None:
            return  # 分批发送阶段不重新规划
        try:
            img=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except Exception as e:
            self.get_logger().error(str(e)); return

        self.h,self.w=img.shape[:2]; self.cx=self.w//2; vis=img.copy()

        # ROI
        tl=(int(self.w*self.trap_left_ratio),int(self.h*self.trap_top_ratio))
        tr=(int(self.w*self.trap_right_ratio),int(self.h*self.trap_top_ratio))
        bl=(int(self.w*self.trap_bottom_left_ratio),self.h)
        br=(int(self.w*self.trap_bottom_right_ratio),self.h)
        roi_cnt=np.array([tl,tr,br,bl],np.int32).reshape((-1,1,2))

        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,self.lower_red,self.upper_red)
        roi_mask=np.zeros_like(mask); cv2.fillPoly(roi_mask,[roi_cnt],255)
        col_mask=cv2.bitwise_and(mask,roi_mask)
        rect_red=int(cv2.countNonZero(col_mask))
        self.pub_red_cnt.publish(Int32(data=rect_red))

        # (1) 进入避障
        if rect_red>self.red_thresh_px:
            if self.lane_forward:
                self.lane_forward=False
                self.pub_stop_flag.publish(Bool(data=True))
            if self.timer_handle is not None:
                self.timer_handle.cancel(); self.timer_handle=None
            self.compute_and_store_path(col_mask,vis,mask)
            self.publish_dbg(vis,msg.header)
            return

        # (2) 障碍消失 → 开启分批发送剩余点
        if (not self.lane_forward) and rect_red<self.red_resume_px and self.current_path:
            if self.timer_handle is None:
                self.timer_handle=self.create_timer(self.triplet_period,self.broadcast_triplet)

        self.draw_common(vis,roi_cnt,not self.lane_forward,rect_red,int(cv2.countNonZero(mask)))
        self.publish_dbg(vis,msg.header)

    # -------------- 计算 B-样条并发送初始关键点 --------------
    def compute_and_store_path(self,col_mask,vis,mask):
        ys,xs=np.where(col_mask>0)
        left_cnt,right_cnt=np.sum(xs<self.cx),np.sum(xs>=self.cx)
        if left_cnt==0 and right_cnt>0:
            d,amp_m=+1,self.single_side_amp
        elif right_cnt==0 and left_cnt>0:
            d,amp_m=-1,self.single_side_amp
        else:
            if left_cnt<right_cnt:
                d=+1; dist_px=self.cx-xs[xs<self.cx].min()
            else:
                d=-1; dist_px=xs[xs>=self.cx].max()-self.cx
            amp_m=max(dist_px*self.m_per_px+self.extra_safe_m,self.min_offset_m)

        p0_m=self.px2robot((self.cx,self.h))
        p1_m=self.px2robot((self.cx,int(self.h*self.trap_top_ratio)))
        path=self.build_nurbs(p0_m,p1_m,d,amp_m,n_ctrl=7,n_sample=self.n_sample,p=3)
        self.current_path=path;   # 全局坐标

        # —— 关键点 ——
        desired_x=0.3; p_start=None; idx_start=1
        for i in range(1,len(path)):
            x0,y0=path[i-1]; x1,y1=path[i]
            if (x0<=desired_x<=x1) or (x1<=desired_x<=x0):
                t=(desired_x-x0)/(x1-x0+1e-12)
                p_start=np.array([desired_x, y0+t*(y1-y0)])
                idx_start=i; break
        if p_start is None:
            p_start=path[1]; idx_start=1

        ys_list=[p[1] for p in path]
        idx_mid=int(np.argmax(ys_list)) if d==1 else int(np.argmin(ys_list))
        p_mid=path[idx_mid]; p_end=path[-1]

        for pub,p in zip(self.triplet_pubs,[p_start,p_mid,p_end]):
            pub.publish(Pose2D(x=float(p[0]),y=float(p[1]),theta=0.0))

        # 设置局部原点 → p_mid
        self.reference_point=p_mid.copy()
        self.path_idx=idx_start+1  # 下一待发送点（p_start 已发）

        dbg_px=[self.robot2px(p) for p in path]
        cv2.polylines(vis,[np.array(dbg_px).reshape(-1,1,2)],False,(255,255,0),2)

    # ---------------------- 分批发送剩余点 ----------------------
    def broadcast_triplet(self):
        if self.path_idx>=len(self.current_path):
            if self.timer_handle is not None:
                self.timer_handle.cancel(); self.timer_handle=None
            self.current_path=[]; self.path_idx=0; self.reference_point=None
            self.lane_forward=True; self.pub_stop_flag.publish(Bool(data=False))
            return

        chunk=self.current_path[self.path_idx:self.path_idx+3]
        while len(chunk)<3: chunk.append(chunk[-1])

        # 相对坐标 = 全局点 - reference_point
        rel_chunk=[p-self.reference_point for p in chunk] if self.reference_point is not None else chunk
        for pub,rel in zip(self.triplet_pubs,rel_chunk):
            pub.publish(Pose2D(x=float(rel[0]),y=float(rel[1]),theta=0.0))

        # 更新原点为本批次的第 2 个全局点
        self.reference_point=chunk[1].copy()
        self.path_idx+=3

    # ------------------- 用户原版 build_nurbs -------------------
    @staticmethod
    def build_nurbs(p0,p1,d,amp,
                    n_ctrl=7,n_sample=25,p=3):
        p0=np.asarray(p0,float); p1=np.asarray(p1,float)
        v=p1-p0; v/=(np.linalg.norm(v) or 1.0)
        perp=np.array([-v[1],v[0]])
        # 控制点
        ctrl=[p0.copy()]
        for i in range(1,n_ctrl-1):
            t=i/(n_ctrl-1)
            offset=amp*(1-t**2)
            pt=(1-t)*p0 + t*p1 + d*offset*perp
            ctrl.append(pt)
        ctrl.append(p1.copy())
        ctrl=np.array(ctrl)
        # 节点向量
        prefix=[0.0]*(p+1); suffix=[1.0]*(p+1)
        total_knots=n_ctrl+p+1
        n_internal=total_knots-len(prefix)-len(suffix)
        inner=[(i/(n_internal+1))**2 for i in range(1,n_internal+1)]
        knots=np.concatenate((prefix,inner,suffix))
        # 采样
        t_vals=np.linspace(0.0,1.0,n_sample)
        x_vals,y_vals=si.splev(t_vals,(knots,
                          [ctrl[:,0],ctrl[:,1]],p))
        x_vals[-1],y_vals[-1]=p1[0],p1[1]
        return [np.array([x_vals[i],y_vals[i]],float)
                for i in range(n_sample)]

    # ---------------------------- 画辅助 ----------------------------
    def draw_common(self,vis,roi,playing,rect_cnt=0,total_cnt=0):
        if vis is None:  # 仅用于路径计算调试时可能传入 None
            return
        cv2.line(vis,(self.cx,0),(self.cx,self.h),(0,255,0),2)
        if roi is not None:
            cv2.polylines(vis,[roi],True,(0,255,255),2)
        txt="Avoiding..." if playing else "No obstacle"
        cv2.putText(vis,txt,(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,
                    (0,0,255) if playing else (0,255,0),2)
        cv2.putText(vis,f"Total:{total_cnt}  Rect:{rect_cnt}",
                    (10,self.h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,
                    (255,255,255),2)

    def publish_dbg(self,vis,hdr):
        try:
            dbg=self.bridge.cv2_to_imgmsg(vis,'bgr8')
            dbg.header=hdr
            self.pub_dbg_img.publish(dbg)
        except Exception as e:
            self.get_logger().error(str(e))


# ---------------------------- main ----------------------------

def main(args=None):
    rclpy.init(args=args)
    node=RedAlertDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.timer_handle is not None:
            node.timer_handle.cancel()
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
