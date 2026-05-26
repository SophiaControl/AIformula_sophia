import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Pose2D
import pandas as pd
import time

class LaneLineSubscriber(Node):
    def __init__(self, a, b, c):
        super().__init__('lane_line_subscriber')

        # ==== 1) 订阅 左、右、中心 三条车道线的话题 ====
        self.center_subscription = self.create_subscription(
            PointCloud2,
            '/aiformula_perception/lane_line_publisher/lane_lines/center',  
            self.center_callback,
            10
        )
        self.left_subscription = self.create_subscription(
            PointCloud2,
            '/aiformula_perception/lane_line_publisher/lane_lines/left',
            self.left_callback,
            10
        )
        self.right_subscription = self.create_subscription(
            PointCloud2,
            '/aiformula_perception/lane_line_publisher/lane_lines/right',
            self.right_callback,
            10
        )

        # ==== 2) 三个发布者：分别发布 A、B、C 三个点的 Pose2D ====
        self.pose_publisher_a = self.create_publisher(Pose2D, '/processed_point_a', 10)
        self.pose_publisher_b = self.create_publisher(Pose2D, '/processed_point_b', 10)
        self.pose_publisher_c = self.create_publisher(Pose2D, '/processed_point_c', 10)

        # ==== 3) 用于存储各条线的点云数据 ====
        self.left_points = []    
        self.right_points = []
        self.center_points = []

        # ==== 4) 其他参数 ====
        self.a = a  # 第一个点索引
        self.b = b  # 第二个点索引
        self.c = c  # 第三个点索引

        self.data_b = []  # 若不需要记录可以去掉
        self.last_received_time = 0.0

        # ==== 5) 车道宽度 (米) ====
        self.lane_width_m = 2.2

    # -------------------------------------------------------------
    #   回调函数： 左车道线
    # -------------------------------------------------------------
    def left_callback(self, msg):
        self.left_points = self.parse_pointcloud2(msg)
        self.get_logger().debug(f"Received LEFT lane with {len(self.left_points)} points")

    # -------------------------------------------------------------
    #   回调函数： 右车道线
    # -------------------------------------------------------------
    def right_callback(self, msg):
        self.right_points = self.parse_pointcloud2(msg)
        self.get_logger().debug(f"Received RIGHT lane with {len(self.right_points)} points")

    # -------------------------------------------------------------
    #   回调函数： 中心线
    # -------------------------------------------------------------
    def center_callback(self, msg):
        current_time = time.time()
        self.last_received_time = current_time

        self.center_points = self.parse_pointcloud2(msg)
        center_count = len(self.center_points)

        left_count = len(self.left_points)
        right_count = len(self.right_points)

        # 如果左右线都够，则直接用 center
        if left_count >= max(self.a, self.b, self.c)+1 and right_count >= max(self.a, self.b, self.c)+1:
            if center_count <= max(self.a, self.b, self.c):
                self.get_logger().warning("Center line not enough, but left & right are enough => using center anyway.")
            self.do_publish_points_by_center(current_time)

        # 如果仅左线足够
        elif left_count >= max(self.a, self.b, self.c)+1 and right_count < max(self.a, self.b, self.c)+1:
            self.get_logger().info("Right lane not enough, using Left + lane_width to simulate center.")
            self.do_publish_points_by_one_side(
                side_points=self.left_points,
                side='left',
                current_time=current_time
            )

        # 如果仅右线足够
        elif right_count >= max(self.a, self.b, self.c)+1 and left_count < max(self.a, self.b, self.c)+1:
            self.get_logger().info("Left lane not enough, using Right + lane_width to simulate center.")
            self.do_publish_points_by_one_side(
                side_points=self.right_points,
                side='right',
                current_time=current_time
            )

        else:
            # 两侧都不足 => 默认直行（这里只发布一个 Pose2D）
            self.get_logger().warning("Both lanes not enough => assume straight (x=2.5,y=0,theta=0)")
            pose_msg_a = Pose2D()
            pose_msg_a.x = 1.5
            pose_msg_a.y = 0.0
            pose_msg_a.theta = 0.0
            pose_msg_b = Pose2D()
            pose_msg_b.x = 2.0
            pose_msg_b.y = 0.0
            pose_msg_b.theta = 0.0
            pose_msg_c = Pose2D()
            pose_msg_c.x = 3.0
            pose_msg_c.y = 0.0
            pose_msg_c.theta = 0.0
            # 这里随意选一个发布者，也可以为直行单独再弄个发布者话题
            self.pose_publisher_a.publish(pose_msg_a)
            self.pose_publisher_b.publish(pose_msg_b)
            self.pose_publisher_c.publish(pose_msg_c)
            self.get_logger().info("Published default Pose2D (straight)")

    # -------------------------------------------------------------
    #   使用中心线 (已存到 self.center_points)，取第 a,b,c 点
    # -------------------------------------------------------------
    def do_publish_points_by_center(self, current_time):
        if len(self.center_points) <= max(self.a, self.b, self.c):
            self.get_logger().warning("Insufficient points in center line.")
            return

        point_a = self.center_points[self.a]
        point_b = self.center_points[self.b]
        point_c = self.center_points[self.c]

        # 发布三点 (A, B, C) 的 Pose2D
        self.publish_points_abc(point_a, point_b, point_c, current_time)

    # -------------------------------------------------------------
    #   仅一侧可用时，通过平移来模拟 center，然后发布
    # -------------------------------------------------------------
    def do_publish_points_by_one_side(self, side_points, side, current_time):
        if len(side_points) <= max(self.a, self.b, self.c):
            self.get_logger().warning(f"Side {side} not enough points => skip.")
            return

        pa = side_points[self.a]
        pb = side_points[self.b]
        pc = side_points[self.c]

        # 根据左右线决定往 Y 轴正/负方向平移
        if side == 'left':
            pa_center = (pa[0], pa[1] - self.lane_width_m/2, pa[2])
            pb_center = (pb[0], pb[1] - self.lane_width_m/2, pb[2])
            pc_center = (pc[0], pc[1] - self.lane_width_m/2, pc[2])
        else:  # side=='right'
            pa_center = (pa[0], pa[1] + self.lane_width_m/2, pa[2])
            pb_center = (pb[0], pb[1] + self.lane_width_m/2, pb[2])
            pc_center = (pc[0], pc[1] + self.lane_width_m/2, pc[2])

        # 发布三点 (A, B, C) 的 Pose2D
        self.publish_points_abc(pa_center, pb_center, pc_center, current_time)

    # -------------------------------------------------------------
    #   分别用三个发布者发布 (A, B, C) 三个 Pose2D，theta=0
    # -------------------------------------------------------------
    def publish_points_abc(self, point_a, point_b, point_c, current_time):
        # A 点
        pose_a = Pose2D()
        pose_a.x, pose_a.y, _ = point_a
        pose_a.theta = 0.0
        self.pose_publisher_a.publish(pose_a)
        self.get_logger().info(f"Publish Pose2D(A): x={pose_a.x:.2f}, y={pose_a.y:.2f}, theta=0")

        # B 点
        pose_b = Pose2D()
        pose_b.x, pose_b.y, _ = point_b
        pose_b.theta = 0.0
        self.pose_publisher_b.publish(pose_b)
        self.get_logger().info(f"Publish Pose2D(B): x={pose_b.x:.2f}, y={pose_b.y:.2f}, theta=0")

        # C 点
        pose_c = Pose2D()
        pose_c.x, pose_c.y, _ = point_c
        pose_c.theta = 0.0
        self.pose_publisher_c.publish(pose_c)
        self.get_logger().info(f"Publish Pose2D(C): x={pose_c.x:.2f}, y={pose_c.y:.2f}, theta=0")

        # 如果还需记录信息（如 B 点的 y 坐标），可以在此处理
        # timestamp = time.strftime("%H:%M:%S", time.localtime(current_time))
        # self.data_b.append((timestamp, pose_b.y))ith_angle

    # -------------------------------------------------------------
    #   解析点云
    # -------------------------------------------------------------
    def parse_pointcloud2(self, msg):
        points = []
        try:
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append(p)
        except Exception as e:
            self.get_logger().error(f"Error parsing PointCloud2: {e}")
        return points

    # -------------------------------------------------------------
    #   退出时保存数据 (若不需要可删除)
    # -------------------------------------------------------------
    def save_data_to_excel(self):
        folder_path = "./lane_analysis_data"
        b_file_path = f"{folder_path}/point_b_data1.xlsx"

        import os
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pd.DataFrame(self.data_b, columns=["Time", "Point B Y"]).to_excel(b_file_path, index=False)
        self.get_logger().info(f"Data saved to {folder_path}")

def main(args=None):
    rclpy.init(args=args)

    a = 4  
    b = 6  
    c = 8  

    lane_line_subscriber = LaneLineSubscriber(a, b, c)
    try:
        rclpy.spin(lane_line_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # lane_line_subscriber.save_data_to_excel()
        lane_line_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
