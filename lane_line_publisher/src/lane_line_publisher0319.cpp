#include "lane_line_publisher/lane_line_publisher.hpp"

namespace aiformula {

LaneLinePublisher::LaneLinePublisher() : Node("lane_line_publisher") {
    initMembers();
    initConnections();
    RCLCPP_INFO(get_logger(), "Launched %s", get_name());
}

void LaneLinePublisher::initMembers() {
    debug_ = getRosParameter<bool>(this, "debug");
    vehicle_frame_id_ = getRosParameter<std::string>(this, "vehicle_frame_id");
    xmin_ = getRosParameter<double>(this, "lane_line_publisher.roi.xmin");
    xmax_ = getRosParameter<double>(this, "lane_line_publisher.roi.xmax");
    ymin_ = getRosParameter<double>(this, "lane_line_publisher.roi.ymin");
    ymax_ = getRosParameter<double>(this, "lane_line_publisher.roi.ymax");
    spacing_ = getRosParameter<double>(this, "lane_line_publisher.spacing");
    const auto camera_frame_id = getRosParameter<std::string>(this, "camera_frame_id");
    const auto camera_name = getRosParameter<std::string>(this, "camera_name");
    const auto min_area = getRosParameter<int>(this, "lane_pixel_finder.min_area");
    const auto tolerance = getRosParameter<int>(this, "lane_pixel_finder.tolerance");

    getCameraParams(this, camera_name, camera_matrix_);
    camera_matrix_ = camera_matrix_.inv();
    vehicle_T_camera_ = getTf2Transform(this, vehicle_frame_id_, camera_frame_id);

    lane_pixel_finder_ = std::make_shared<const LanePixelFinder>(min_area, tolerance);
    cubic_line_fitter_ = std::make_shared<const CubicLineFitter>(xmin_, xmax_, spacing_);
}

void LaneLinePublisher::initConnections() {
    const auto queue_size = getRosParameter<int>(this, "lane_line_publisher.queue_size");
    mask_image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "mask_image", queue_size, std::bind(&LaneLinePublisher::imageCallback, this, std::placeholders::_1));
    lane_line_pubs_ = {create_publisher<sensor_msgs::msg::PointCloud2>("lane_lines/left", queue_size),
                       create_publisher<sensor_msgs::msg::PointCloud2>("lane_lines/right", queue_size),
                       create_publisher<sensor_msgs::msg::PointCloud2>("lane_lines/center", queue_size)};

    if (!debug_) return;

    annotated_mask_image_pub_ = create_publisher<sensor_msgs::msg::Image>("annotated_mask_image", queue_size);
    contour_point_pubs_ = {create_publisher<sensor_msgs::msg::PointCloud2>("contour_points/left", queue_size),
                           create_publisher<sensor_msgs::msg::PointCloud2>("contour_points/right", queue_size),
                           create_publisher<sensor_msgs::msg::PointCloud2>("contour_points/center", queue_size)};
}

void LaneLinePublisher::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) const {
    RCLCPP_DEBUG(get_logger(), "Recieved image [%ix%i]", msg->width, msg->height);

    auto cv_img = cv_bridge::toCvShare(msg, msg->encoding);
    if (cv_img->image.empty()) {
        RCLCPP_WARN(get_logger(), "Recieved empty image");
        return;
    }

    LaneLines lane_lines;
    findLaneLines(cv_img->image, msg->header.stamp, lane_lines);
    publishLaneLines(lane_lines, msg->header.stamp);
}

void LaneLinePublisher::findLaneLines(const cv::Mat& mask, const builtin_interfaces::msg::Time& timestamp,
                                      LaneLines& lane_lines) const {
    lane_pixel_finder_->findLanePixels(mask, lane_lines);
    if (debug_) publishAnnotatedMask(mask, timestamp, lane_lines);

    std::vector<std::vector<Eigen::Vector3d>> contour_points;
    std::vector<LaneLine*> lane_line_ptrs = {&lane_lines.left, &lane_lines.right, &lane_lines.center};
    for (auto* lane_line : lane_line_ptrs) {
        lane_line->toVehicleFrame(camera_matrix_, vehicle_T_camera_);
        if (debug_) contour_points.emplace_back(lane_line->points);
        lane_line->cropToRoi(xmin_, xmax_, ymin_, ymax_);
        lane_line->respacePoints(spacing_);
        lane_line->fitPoints(cubic_line_fitter_);
    }

    if (debug_) publishContourPoints(contour_points, timestamp);
}

void LaneLinePublisher::publishAnnotatedMask(const cv::Mat& mask, const builtin_interfaces::msg::Time& timestamp,
                                             const LaneLines& lane_lines) const {
    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = timestamp;
    cv_img.encoding = "bgr8";
    cv_img.image = lane_pixel_finder_->visualizeLanePixels(mask, lane_lines);

    sensor_msgs::msg::Image msg;
    cv_img.toImageMsg(msg);
    annotated_mask_image_pub_->publish(std::move(msg));
}

void LaneLinePublisher::publishContourPoints(const std::vector<std::vector<Eigen::Vector3d>>& contour_points,
                                             const builtin_interfaces::msg::Time& timestamp) const {
    for (int i = 0; i < NUM_LANE_LINES; ++i) {
        const auto& points = contour_points[i];
        pcl::PointCloud<pcl::PointXYZ> cloud;
        for (const auto& point : points) {
            auto& p = cloud.points.emplace_back();
            p.x = point.x();
            p.y = point.y();
        }

        sensor_msgs::msg::PointCloud2 pcl_msg;
        pcl::toROSMsg(cloud, pcl_msg);
        pcl_msg.header.stamp = timestamp;
        pcl_msg.header.frame_id = vehicle_frame_id_;
        contour_point_pubs_[i]->publish(pcl_msg);
    }
}

void LaneLinePublisher::publishLaneLines(const LaneLines& lane_lines,
                                         const builtin_interfaces::msg::Time& timestamp) const {
    const std::vector<const LaneLine*> lane_line_ptrs = {&lane_lines.left, &lane_lines.right, &lane_lines.center};

    for (int i = 0; i < NUM_LANE_LINES; ++i) {
        const auto& lane_line_points = lane_line_ptrs[i]->points;
        pcl::PointCloud<pcl::PointXYZ> cloud;
        for (const auto& point : lane_line_points) {
            auto& p = cloud.points.emplace_back();
            p.x = point.x();
            p.y = point.y();
        }

        sensor_msgs::msg::PointCloud2 pcl_msg;
        pcl::toROSMsg(cloud, pcl_msg);
        pcl_msg.header.stamp = timestamp;
        pcl_msg.header.frame_id = vehicle_frame_id_;
        lane_line_pubs_[i]->publish(pcl_msg);
    } 
}

}  // namespace aiformula

// #include "lane_line_publisher/lane_line_publisher.hpp"

// namespace aiformula {

// LaneLinePublisher::LaneLinePublisher() : Node("lane_line_publisher") {
//     initMembers();
//     initConnections();
//     RCLCPP_INFO(get_logger(), "Launched %s", get_name());
// }

// void LaneLinePublisher::initMembers() {
//     debug_ = getRosParameter<bool>(this, "debug");
//     vehicle_frame_id_ = getRosParameter<std::string>(this, "vehicle_frame_id");
//     xmin_ = getRosParameter<double>(this, "lane_line_publisher.roi.xmin");
//     xmax_ = getRosParameter<double>(this, "lane_line_publisher.roi.xmax");
//     ymin_ = getRosParameter<double>(this, "lane_line_publisher.roi.ymin");
//     ymax_ = getRosParameter<double>(this, "lane_line_publisher.roi.ymax");
//     spacing_ = getRosParameter<double>(this, "lane_line_publisher.spacing");
//     const auto camera_frame_id = getRosParameter<std::string>(this, "camera_frame_id");
//     const auto camera_name = getRosParameter<std::string>(this, "camera_name");
//     const auto min_area = getRosParameter<int>(this, "lane_pixel_finder.min_area");
//     const auto tolerance = getRosParameter<int>(this, "lane_pixel_finder.tolerance");

//     getCameraParams(this, camera_name, camera_matrix_);
//     camera_matrix_ = camera_matrix_.inv();
//     vehicle_T_camera_ = getTf2Transform(this, vehicle_frame_id_, camera_frame_id);

//     lane_pixel_finder_ = std::make_shared<const LanePixelFinder>(min_area, tolerance);
//     cubic_line_fitter_ = std::make_shared<const CubicLineFitter>(xmin_, xmax_, spacing_);
// }

// void LaneLinePublisher::initConnections() {
//     const auto queue_size = getRosParameter<int>(this, "lane_line_publisher.queue_size");
//     mask_image_sub_ = create_subscription<sensor_msgs::msg::Image>(
//         "mask_image", queue_size, std::bind(&LaneLinePublisher::imageCallback, this, std::placeholders::_1));
//     lane_line_pubs_ = {create_publisher<sensor_msgs::msg::PointCloud2>("lane_lines/left", queue_size),
//                        create_publisher<sensor_msgs::msg::PointCloud2>("lane_lines/right", queue_size),
//                        create_publisher<sensor_msgs::msg::PointCloud2>("lane_lines/center", queue_size)};

//     if (!debug_) return;

//     annotated_mask_image_pub_ = create_publisher<sensor_msgs::msg::Image>("annotated_mask_image", queue_size);
//     contour_point_pubs_ = {create_publisher<sensor_msgs::msg::PointCloud2>("contour_points/left", queue_size),
//                            create_publisher<sensor_msgs::msg::PointCloud2>("contour_points/right", queue_size),
//                            create_publisher<sensor_msgs::msg::PointCloud2>("contour_points/center", queue_size)};
// }

// void LaneLinePublisher::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) const {
//     RCLCPP_DEBUG(get_logger(), "Recieved image [%ix%i]", msg->width, msg->height);

//     auto cv_img = cv_bridge::toCvShare(msg, msg->encoding);
//     if (cv_img->image.empty()) {
//         RCLCPP_WARN(get_logger(), "Recieved empty image");
//         return;
//     }

//     LaneLines lane_lines;
//     findLaneLines(cv_img->image, msg->header.stamp, lane_lines);
//     publishLaneLines(lane_lines, msg->header.stamp);
// }


// void LaneLinePublisher::findLaneLines(const cv::Mat& mask, const builtin_interfaces::msg::Time& timestamp,
//                                       LaneLines& lane_lines) const {
//     // 先找出左右车道线的像素点
//     lane_pixel_finder_->findLanePixels(mask, lane_lines);
//     if (debug_) publishAnnotatedMask(mask, timestamp, lane_lines);

//     std::vector<std::vector<Eigen::Vector3d>> contour_points;
//     std::vector<LaneLine*> lane_line_ptrs = {&lane_lines.left, &lane_lines.right};

//     // 设置车道线点数的阈值，设定为 5 个点
//     const size_t min_points_threshold = 5;  
//     const double lane_width = 3.6;  // 假设车道宽度为3.6米，可以根据实际情况调整

//     // 计算左右车道线点数
//     size_t left_points_size = lane_lines.left.points.size();
//     size_t right_points_size = lane_lines.right.points.size();

//     // 预分配内存以避免频繁的动态分配
//     if (left_points_size < min_points_threshold) {
//         lane_lines.left.points.reserve(std::max(min_points_threshold, right_points_size)); // 预分配内存
//     }
//     if (right_points_size < min_points_threshold) {
//         lane_lines.right.points.reserve(std::max(min_points_threshold, left_points_size)); // 预分配内存
//     }

//     // 第一种情况：左车道线点数不足，通过右车道线补齐
//     if (left_points_size < min_points_threshold && right_points_size >= min_points_threshold) {
//         for (size_t i = left_points_size; i < right_points_size; ++i) {
//             Eigen::Vector3d right_point = lane_lines.right.points[i];
//             Eigen::Vector3d left_point = right_point + Eigen::Vector3d(-lane_width, 0, 0);  // 左车道线在右车道线左侧
//             lane_lines.left.points.emplace_back(left_point);  // 这里使用 push_back，因为我们预分配了足够的内存
//         }
//     }
//     // 第二种情况：右车道线点数不足，通过左车道线补齐
//     else if (right_points_size < min_points_threshold && left_points_size >= min_points_threshold) {
//         for (size_t i = right_points_size; i < left_points_size; ++i) {
//             Eigen::Vector3d left_point = lane_lines.left.points[i];
//             Eigen::Vector3d right_point = left_point + Eigen::Vector3d(lane_width, 0, 0);  // 右车道线在左车道线右侧
//             lane_lines.right.points.emplace_back(right_point);  // 这里也使用 push_back，因为我们预分配了足够的内存
//         }
//     }
//     // 第三种情况：两侧车道线点数都不足，返回空
//     else if (left_points_size < min_points_threshold && right_points_size < min_points_threshold) {
//         // 两侧车道线点数都不足，不进行补全操作，返回
//         return;  // 不影响后续运行，提前返回
//     }

//     // 先拟合左右车道线
//     for (auto* lane_line : lane_line_ptrs) {
//         lane_line->toVehicleFrame(camera_matrix_, vehicle_T_camera_);
//         if (debug_) contour_points.emplace_back(lane_line->points);
//         lane_line->cropToRoi(xmin_, xmax_, ymin_, ymax_);
//         lane_line->respacePoints(spacing_);
//         lane_line->fitPoints(cubic_line_fitter_);
//     }

//     // 使用左右车道线拟合后的点计算中心车道线的中点
//     for (size_t i = 0; i < lane_lines.left.points.size(); ++i) {
//         Eigen::Vector3d left_point = lane_lines.left.points[i];
//         Eigen::Vector3d right_point = lane_lines.right.points[i];
//         Eigen::Vector3d center_point = (left_point + right_point) / 2.0;  // 计算中点
//         lane_lines.center.points.emplace_back(center_point);  // 这里可以继续使用 emplace_back，因为中点是一个新的点
//     }

//     if (debug_) publishContourPoints(contour_points, timestamp);
// }




// // void LaneLinePublisher::findLaneLines(const cv::Mat& mask, const builtin_interfaces::msg::Time& timestamp,
// //                                       LaneLines& lane_lines) const {
// //     lane_pixel_finder_->findLanePixels(mask, lane_lines);
// //     if (debug_) publishAnnotatedMask(mask, timestamp, lane_lines);

// //     std::vector<std::vector<Eigen::Vector3d>> contour_points;
// //     std::vector<LaneLine*> lane_line_ptrs = {&lane_lines.left, &lane_lines.right, &lane_lines.center};
// //     for (auto* lane_line : lane_line_ptrs) {
// //         lane_line->toVehicleFrame(camera_matrix_, vehicle_T_camera_);
// //         if (debug_) contour_points.emplace_back(lane_line->points);
// //         lane_line->cropToRoi(xmin_, xmax_, ymin_, ymax_);
// //         lane_line->respacePoints(spacing_);
// //         lane_line->fitPoints(cubic_line_fitter_);
// //     }

// //     if (debug_) publishContourPoints(contour_points, timestamp);
// // }

// void LaneLinePublisher::publishAnnotatedMask(const cv::Mat& mask, const builtin_interfaces::msg::Time& timestamp,
//                                              const LaneLines& lane_lines) const {
//     cv_bridge::CvImage cv_img;
//     cv_img.header.stamp = timestamp;
//     cv_img.encoding = "bgr8";
//     cv_img.image = lane_pixel_finder_->visualizeLanePixels(mask, lane_lines);

//     sensor_msgs::msg::Image msg;
//     cv_img.toImageMsg(msg);
//     annotated_mask_image_pub_->publish(std::move(msg));
// }

// void LaneLinePublisher::publishContourPoints(const std::vector<std::vector<Eigen::Vector3d>>& contour_points,
//                                              const builtin_interfaces::msg::Time& timestamp) const {
//     for (int i = 0; i < NUM_LANE_LINES; ++i) {
//         const auto& points = contour_points[i];
//         pcl::PointCloud<pcl::PointXYZ> cloud;
//         for (const auto& point : points) {
//             auto& p = cloud.points.emplace_back();
//             p.x = point.x();
//             p.y = point.y();
//         }

//         sensor_msgs::msg::PointCloud2 pcl_msg;
//         pcl::toROSMsg(cloud, pcl_msg);
//         pcl_msg.header.stamp = timestamp;
//         pcl_msg.header.frame_id = vehicle_frame_id_;
//         contour_point_pubs_[i]->publish(pcl_msg);
//     }
// }

// void LaneLinePublisher::publishLaneLines(const LaneLines& lane_lines,
//                                          const builtin_interfaces::msg::Time& timestamp) const {
//     const std::vector<const LaneLine*> lane_line_ptrs = {&lane_lines.left, &lane_lines.right, &lane_lines.center};

//     for (int i = 0; i < NUM_LANE_LINES; ++i) {
//         const auto& lane_line_points = lane_line_ptrs[i]->points;
//         pcl::PointCloud<pcl::PointXYZ> cloud;
//         for (const auto& point : lane_line_points) {
//             auto& p = cloud.points.emplace_back();
//             p.x = point.x();
//             p.y = point.y();
//         }

//         sensor_msgs::msg::PointCloud2 pcl_msg;
//         pcl::toROSMsg(cloud, pcl_msg);
//         pcl_msg.header.stamp = timestamp;
//         pcl_msg.header.frame_id = vehicle_frame_id_;
//         lane_line_pubs_[i]->publish(pcl_msg);
//     }
// }

// }  // namespace aiformula
