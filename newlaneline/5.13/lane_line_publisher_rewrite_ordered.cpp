#include "lane_line_publisher/lane_line_publisher.hpp"

// =============================================================================
// 文件说明
// =============================================================================
// 1. 这个文件按信息流重新排序：
//      节点初始化 → 图像回调 → 主体检测流程 → 发布 → 具体 helper 实现。
// 2. 主体逻辑相关的辅助函数都放在本 cpp 内部，不再调用：
//      LanePixelFinder / CubicLineFitter / LaneLine::toVehicleFrame /
//      LaneLine::cropToRoi / LaneLine::fitPoints。
// 3. connected component 只做面积过滤，不额外做横线方向过滤。
// 4. 图像坐标一次拟合用于下一帧 ROI；车辆坐标二次拟合用于当前帧输出。
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

namespace aiformula {
namespace {

// =============================================================================
// 【模块 A：基础定义、状态变量、函数声明】
// 说明：
// 这里只放“全局要用到的定义”和 helper 函数声明。
// 真正的信息流从后面的 LaneLinePublisher 构造函数开始读。
// =============================================================================

constexpr int LEFT_INDEX = 0;
constexpr int RIGHT_INDEX = 1;
constexpr int CENTER_INDEX = 2;
constexpr double kPi = 3.14159265358979323846;
constexpr double kDegPerRad = 180.0 / kPi;

// 图像坐标系一次拟合线：u = a*v + b。
// u 是图像列坐标，v 是图像行坐标。
// 该线只用于下一帧 tracking ROI，不直接发布。
struct ImageLineFit {
    bool valid = false;
    double a = 0.0;
    double b = 0.0;
    int lost_count = 0;
};

// debug 图像显示用：保存当前帧该侧使用的 ROI 和提取到的像素点。
struct SideDebugInfo {
    bool use_tracking_roi = false;
    bool has_line = false;
    ImageLineFit line;
    std::vector<cv::Point> extracted_pixels;
};

// 为了不改 hpp，这里仍然把 tracking 状态放在 cpp 内部。
// 左右两侧分别保存上一帧一次拟合线。
std::array<ImageLineFit, 2> g_side_line_fits;
std::array<SideDebugInfo, 2> g_debug_info;

// 参数缓存。参数名保持旧 launch 兼容。
int g_min_area = 20;
int g_min_required_pixels = 75;
int g_max_lost_frames = 5;
double g_init_top_crop_ratio = 0.45;
double g_margin_near = 80.0;
double g_margin_far = 25.0;
double g_line_update_alpha = 0.75;
double g_min_line_angle_deg = 12.0;
double g_max_line_angle_deg = 168.0;

// ---- A.1 基础工具 ----
int clampInt(const int value, const int low, const int high);
double clampDouble(const double value, const double low, const double high);
int topRow(const int rows);
double marginAtRow(const int row, const int rows);

// ---- A.2 mask 预处理 ----
cv::Mat toBinaryMask(const cv::Mat& mask);
cv::Mat removeSmallComponents(const cv::Mat& binary_mask);

// ---- A.3 图像 ROI 像素提取 ----
std::vector<cv::Point> extractInitPixels(const cv::Mat& cleaned_mask, const int side_index);
std::vector<cv::Point> extractTrackingPixels(const cv::Mat& cleaned_mask, const ImageLineFit& previous_line);

// ---- A.4 图像坐标一次拟合、角度过滤、跳变检查 ----
bool fitImageLineUv(const std::vector<cv::Point>& pixels, ImageLineFit& fit, double& angle_deg);
bool isNearlyHorizontal(const double angle_deg);
bool hasExcessiveJump(const ImageLineFit& previous_line, const ImageLineFit& current_line,
                      const int rows, const int cols);

// ---- A.5 lost 处理和 stored line 更新 ----
void updateStoredLine(ImageLineFit& stored_line, const ImageLineFit& current_line);
void registerDetectionFailure(ImageLineFit& stored_line);

// ---- A.6 像素投影、vehicle ROI 裁剪、车辆坐标二次拟合 ----
bool pixelToVehicleGroundPoint(const cv::Point& pixel, const cv::Mat& camera_matrix_inv,
                               const tf2::Transform& vehicle_T_camera, Eigen::Vector3d& point_out);
std::vector<Eigen::Vector3d> pixelsToVehiclePoints(const std::vector<cv::Point>& pixels,
                                                   const cv::Mat& camera_matrix_inv,
                                                   const tf2::Transform& vehicle_T_camera);
std::vector<Eigen::Vector3d> cropVehiclePoints(const std::vector<Eigen::Vector3d>& points,
                                               const double xmin, const double xmax,
                                               const double ymin, const double ymax);
bool fitQuadraticVehicleLine(const std::vector<Eigen::Vector3d>& input_points,
                             const double xmin, const double xmax, const double spacing,
                             std::vector<Eigen::Vector3d>& output_points);

// ---- A.7 center 生成 ----
void makeCenterFromSides(const std::vector<Eigen::Vector3d>& left_points,
                         const std::vector<Eigen::Vector3d>& right_points,
                         std::vector<Eigen::Vector3d>& center_points);

// ---- A.8 debug image 绘制 ----
cv::Scalar sideColor(const int side_index);
void drawInitRoiBorder(cv::Mat& image, const int side_index);
void drawTrackingRoiBorder(cv::Mat& image, const ImageLineFit& line, const int side_index);
void drawSparsePixels(cv::Mat& image, const std::vector<cv::Point>& pixels, const int side_index);
cv::Mat makeAnnotatedMaskImage(const cv::Mat& cleaned_mask);

}  // namespace

// =============================================================================
// 【模块 B：节点初始化信息流】
// 信息流：构造函数 → 读取参数/相机/TF → 建立 topic 输入输出。
// =============================================================================

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

    // 继续沿用旧参数名，避免 launch 文件失效。
    g_min_area = std::max(1, min_area);
    g_min_required_pixels = 75;
    g_margin_far = std::max(20.0, 0.6 * static_cast<double>(std::max(1, tolerance)));
    g_margin_near = std::max(60.0, 2.0 * static_cast<double>(std::max(1, tolerance)));

    getCameraParams(this, camera_name, camera_matrix_);
    camera_matrix_ = camera_matrix_.inv();
    vehicle_T_camera_ = getTf2Transform(this, vehicle_frame_id_, camera_frame_id);
}

void LaneLinePublisher::initConnections() {
    const auto queue_size = getRosParameter<int>(this, "lane_line_publisher.queue_size");

    mask_image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "mask_image", queue_size,
        std::bind(&LaneLinePublisher::imageCallback, this, std::placeholders::_1));

    lane_line_pubs_ = {
        create_publisher<sensor_msgs::msg::PointCloud2>("lane_lines/left", queue_size),
        create_publisher<sensor_msgs::msg::PointCloud2>("lane_lines/right", queue_size),
        create_publisher<sensor_msgs::msg::PointCloud2>("lane_lines/center", queue_size)};

    if (!debug_) return;

    annotated_mask_image_pub_ = create_publisher<sensor_msgs::msg::Image>("annotated_mask_image", queue_size);
    contour_point_pubs_ = {
        create_publisher<sensor_msgs::msg::PointCloud2>("contour_points/left", queue_size),
        create_publisher<sensor_msgs::msg::PointCloud2>("contour_points/right", queue_size),
        create_publisher<sensor_msgs::msg::PointCloud2>("contour_points/center", queue_size)};
}

// =============================================================================
// 【模块 C：输入回调信息流】
// 信息流：收到 mask_image → 转 OpenCV Mat → findLaneLines → publishLaneLines。
// =============================================================================

void LaneLinePublisher::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) const {
    RCLCPP_DEBUG(get_logger(), "Received image [%ix%i]", msg->width, msg->height);

    auto cv_img = cv_bridge::toCvShare(msg, msg->encoding);
    if (cv_img->image.empty()) {
        RCLCPP_WARN(get_logger(), "Received empty image");
        return;
    }

    LaneLines lane_lines;
    findLaneLines(cv_img->image, msg->header.stamp, lane_lines);
    publishLaneLines(lane_lines, msg->header.stamp);
}

// =============================================================================
// 【模块 D：主体检测信息流】
// 信息流：
// mask → 二值化 → connected component 面积过滤 → 左右分别处理 → center → debug。
// =============================================================================

void LaneLinePublisher::findLaneLines(const cv::Mat& mask,
                                      const builtin_interfaces::msg::Time& timestamp,
                                      LaneLines& lane_lines) const {
    // -------------------------------------------------------------------------
    // 【D.1 mask 预处理】
    // -------------------------------------------------------------------------
    const cv::Mat binary_mask = toBinaryMask(mask);
    const cv::Mat cleaned_mask = removeSmallComponents(binary_mask);

    std::vector<std::vector<Eigen::Vector3d>> contour_points(NUM_LANE_LINES);
    std::array<bool, 2> side_output_valid = {false, false};

    lane_lines.left.pixels.clear();
    lane_lines.left.points.clear();
    lane_lines.right.pixels.clear();
    lane_lines.right.points.clear();
    lane_lines.center.pixels.clear();
    lane_lines.center.points.clear();

    // -------------------------------------------------------------------------
    // 【D.2 left/right 分别处理】
    // 每一侧的信息流：
    // stored line 判断 → ROI 像素提取 → 图像一次拟合 → 投影到 vehicle frame
    // → vehicle ROI crop → vehicle 二次拟合 → 成功后更新 stored line。
    // -------------------------------------------------------------------------
    for (int side = LEFT_INDEX; side <= RIGHT_INDEX; ++side) {
        LaneLine* lane_line = (side == LEFT_INDEX) ? &lane_lines.left : &lane_lines.right;
        auto& stored_line = g_side_line_fits[side];

        // 【D.2.1 选择 tracking ROI 或初始化半区】
        const bool use_tracking_roi = stored_line.valid && stored_line.lost_count <= g_max_lost_frames;
        std::vector<cv::Point> extracted_pixels = use_tracking_roi
                                                      ? extractTrackingPixels(cleaned_mask, stored_line)
                                                      : extractInitPixels(cleaned_mask, side);

        g_debug_info[side].use_tracking_roi = use_tracking_roi;
        g_debug_info[side].has_line = use_tracking_roi;
        g_debug_info[side].line = stored_line;
        g_debug_info[side].extracted_pixels = extracted_pixels;

        // 【D.2.2 图像坐标一次拟合：u = a*v + b】
        ImageLineFit current_line;
        double angle_deg = 0.0;
        const bool image_fit_ok = fitImageLineUv(extracted_pixels, current_line, angle_deg);
        const bool reject_horizontal = image_fit_ok && isNearlyHorizontal(angle_deg);
        const bool reject_jump = image_fit_ok &&
                                 hasExcessiveJump(stored_line, current_line, cleaned_mask.rows, cleaned_mask.cols);

        if (!image_fit_ok || reject_horizontal || reject_jump) {
            lane_line->pixels.clear();
            lane_line->points.clear();
            registerDetectionFailure(stored_line);
            RCLCPP_DEBUG(get_logger(),
                         "%s lane rejected before projection: pixels=%zu, fit_ok=%d, angle=%.2f, horizontal=%d, jump=%d, lost=%d",
                         side == LEFT_INDEX ? "left" : "right",
                         extracted_pixels.size(), image_fit_ok, angle_deg,
                         reject_horizontal, reject_jump, stored_line.lost_count);
            continue;
        }

        // 【D.2.3 像素投影到 vehicle frame】
        lane_line->pixels = extracted_pixels;
        std::vector<Eigen::Vector3d> vehicle_points =
            pixelsToVehiclePoints(extracted_pixels, camera_matrix_, vehicle_T_camera_);

        // 【D.2.4 vehicle-frame ROI crop】
        // debug contour_points/left/right 发布的是 crop 后、二次拟合前的原始点。
        vehicle_points = cropVehiclePoints(vehicle_points, xmin_, xmax_, ymin_, ymax_);
        contour_points[side] = vehicle_points;

        if (vehicle_points.size() < 3) {
            lane_line->points.clear();
            registerDetectionFailure(stored_line);
            RCLCPP_DEBUG(get_logger(), "%s lane rejected after crop: too few vehicle points, lost=%d",
                         side == LEFT_INDEX ? "left" : "right", stored_line.lost_count);
            continue;
        }

        // 【D.2.5 vehicle frame 二次拟合：y = c0 + c1*x + c2*x^2】
        std::vector<Eigen::Vector3d> fitted_points;
        const bool quadratic_ok = fitQuadraticVehicleLine(vehicle_points, xmin_, xmax_, spacing_, fitted_points);
        if (!quadratic_ok || fitted_points.empty()) {
            lane_line->points.clear();
            registerDetectionFailure(stored_line);
            RCLCPP_DEBUG(get_logger(), "%s lane rejected: quadratic fit failed, lost=%d",
                         side == LEFT_INDEX ? "left" : "right", stored_line.lost_count);
            continue;
        }

        // 【D.2.6 当前侧成功输出后，才更新 stored line】
        lane_line->points = fitted_points;
        side_output_valid[side] = true;
        updateStoredLine(stored_line, current_line);
    }

    // -------------------------------------------------------------------------
    // 【D.3 center 生成】
    // 只有 left/right 当前帧都有效时，center 才由二次拟合后的左右点取中点生成。
    // -------------------------------------------------------------------------
    if (side_output_valid[LEFT_INDEX] && side_output_valid[RIGHT_INDEX]) {
        makeCenterFromSides(lane_lines.left.points, lane_lines.right.points, lane_lines.center.points);
        contour_points[CENTER_INDEX] = lane_lines.center.points;
    }

    // -------------------------------------------------------------------------
    // 【D.4 debug 输出】
    // -------------------------------------------------------------------------
    if (debug_) {
        publishAnnotatedMask(cleaned_mask, timestamp, lane_lines);
        publishContourPoints(contour_points, timestamp);
    }
}

// =============================================================================
// 【模块 E：输出发布信息流】
// 说明：
// 1. lane_lines/* 是最终拟合输出。
// 2. contour_points/* 是 debug 点云。
// 3. annotated_mask_image 显示 ROI 边框和当前提取的像素点。
// =============================================================================

void LaneLinePublisher::publishLaneLines(const LaneLines& lane_lines,
                                         const builtin_interfaces::msg::Time& timestamp) const {
    const std::vector<const LaneLine*> lane_line_ptrs = {&lane_lines.left, &lane_lines.right, &lane_lines.center};

    for (int i = 0; i < NUM_LANE_LINES; ++i) {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        for (const auto& point : lane_line_ptrs[i]->points) {
            auto& p = cloud.points.emplace_back();
            p.x = point.x();
            p.y = point.y();
            p.z = point.z();
        }

        sensor_msgs::msg::PointCloud2 pcl_msg;
        pcl::toROSMsg(cloud, pcl_msg);
        pcl_msg.header.stamp = timestamp;
        pcl_msg.header.frame_id = vehicle_frame_id_;
        lane_line_pubs_[i]->publish(pcl_msg);
    }
}

void LaneLinePublisher::publishAnnotatedMask(const cv::Mat& mask,
                                             const builtin_interfaces::msg::Time& timestamp,
                                             const LaneLines& /*lane_lines*/) const {
    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = timestamp;
    cv_img.encoding = "bgr8";
    cv_img.image = makeAnnotatedMaskImage(mask);

    sensor_msgs::msg::Image msg;
    cv_img.toImageMsg(msg);
    annotated_mask_image_pub_->publish(std::move(msg));
}

void LaneLinePublisher::publishContourPoints(const std::vector<std::vector<Eigen::Vector3d>>& contour_points,
                                             const builtin_interfaces::msg::Time& timestamp) const {
    for (int i = 0; i < NUM_LANE_LINES; ++i) {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        for (const auto& point : contour_points[i]) {
            auto& p = cloud.points.emplace_back();
            p.x = point.x();
            p.y = point.y();
            p.z = point.z();
        }

        sensor_msgs::msg::PointCloud2 pcl_msg;
        pcl::toROSMsg(cloud, pcl_msg);
        pcl_msg.header.stamp = timestamp;
        pcl_msg.header.frame_id = vehicle_frame_id_;
        contour_point_pubs_[i]->publish(pcl_msg);
    }
}

namespace {

// =============================================================================
// 【模块 F：基础工具函数实现】
// =============================================================================

int clampInt(const int value, const int low, const int high) {
    return std::max(low, std::min(value, high));
}

double clampDouble(const double value, const double low, const double high) {
    return std::max(low, std::min(value, high));
}

int topRow(const int rows) {
    return clampInt(static_cast<int>(std::round(rows * g_init_top_crop_ratio)), 0, std::max(0, rows - 1));
}

double marginAtRow(const int row, const int rows) {
    const int v_top = topRow(rows);
    const int v_bottom = std::max(v_top + 1, rows - 1);
    const double ratio = static_cast<double>(row - v_top) / static_cast<double>(v_bottom - v_top);
    const double r = clampDouble(ratio, 0.0, 1.0);
    return g_margin_far + (g_margin_near - g_margin_far) * r;
}

// =============================================================================
// 【模块 G：mask 预处理实现】
// 信息流位置：D.1。
// =============================================================================

cv::Mat toBinaryMask(const cv::Mat& mask) {
    cv::Mat gray;
    if (mask.channels() == 3) {
        cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
    } else if (mask.channels() == 4) {
        cv::cvtColor(mask, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = mask;
    }

    cv::Mat gray_u8;
    if (gray.type() == CV_8UC1) {
        gray_u8 = gray;
    } else {
        gray.convertTo(gray_u8, CV_8U);
    }

    cv::Mat binary;
    cv::threshold(gray_u8, binary, 0, 255, cv::THRESH_BINARY);
    return binary;
}

cv::Mat removeSmallComponents(const cv::Mat& binary_mask) {
    cv::Mat labels, stats, centroids;
    const int num_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);

    cv::Mat cleaned = cv::Mat::zeros(binary_mask.size(), CV_8UC1);
    for (int label = 1; label < num_labels; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < g_min_area) continue;
        cleaned.setTo(255, labels == label);
    }
    return cleaned;
}

// =============================================================================
// 【模块 H：图像 ROI 像素提取实现】
// 信息流位置：D.2.1。
// =============================================================================

std::vector<cv::Point> extractInitPixels(const cv::Mat& cleaned_mask, const int side_index) {
    std::vector<cv::Point> pixels;
    const int rows = cleaned_mask.rows;
    const int cols = cleaned_mask.cols;
    const int v_top = topRow(rows);
    const int u_begin = (side_index == LEFT_INDEX) ? 0 : cols / 2;
    const int u_end = (side_index == LEFT_INDEX) ? cols / 2 : cols;

    for (int v = v_top; v < rows; ++v) {
        const auto* row_ptr = cleaned_mask.ptr<uchar>(v);
        for (int u = u_begin; u < u_end; ++u) {
            if (row_ptr[u] != 0) pixels.emplace_back(u, v);
        }
    }
    return pixels;
}

std::vector<cv::Point> extractTrackingPixels(const cv::Mat& cleaned_mask, const ImageLineFit& previous_line) {
    std::vector<cv::Point> pixels;
    const int rows = cleaned_mask.rows;
    const int cols = cleaned_mask.cols;
    const int v_top = topRow(rows);

    for (int v = v_top; v < rows; ++v) {
        const auto* row_ptr = cleaned_mask.ptr<uchar>(v);
        const double u_pred = previous_line.a * static_cast<double>(v) + previous_line.b;
        const double margin = marginAtRow(v, rows);
        const int u_min = clampInt(static_cast<int>(std::floor(u_pred - margin)), 0, cols - 1);
        const int u_max = clampInt(static_cast<int>(std::ceil(u_pred + margin)), 0, cols - 1);
        for (int u = u_min; u <= u_max; ++u) {
            if (row_ptr[u] != 0) pixels.emplace_back(u, v);
        }
    }
    return pixels;
}

// =============================================================================
// 【模块 I：图像一次拟合、横线判断、跳变检查实现】
// 信息流位置：D.2.2。
// =============================================================================

bool fitImageLineUv(const std::vector<cv::Point>& pixels, ImageLineFit& fit, double& angle_deg) {
    if (static_cast<int>(pixels.size()) < g_min_required_pixels) {
        angle_deg = 0.0;
        return false;
    }

    double sum_v = 0.0;
    double sum_u = 0.0;
    for (const auto& p : pixels) {
        sum_v += static_cast<double>(p.y);
        sum_u += static_cast<double>(p.x);
    }

    const double n = static_cast<double>(pixels.size());
    const double mean_v = sum_v / n;
    const double mean_u = sum_u / n;

    double var_v = 0.0;
    double cov_vu = 0.0;
    for (const auto& p : pixels) {
        const double dv = static_cast<double>(p.y) - mean_v;
        const double du = static_cast<double>(p.x) - mean_u;
        var_v += dv * dv;
        cov_vu += dv * du;
    }

    if (var_v < 1e-9) {
        angle_deg = 0.0;
        return false;
    }

    fit.valid = true;
    fit.a = cov_vu / var_v;
    fit.b = mean_u - fit.a * mean_v;

    angle_deg = std::atan2(1.0, fit.a) * kDegPerRad;
    if (angle_deg < 0.0) angle_deg += 180.0;
    return true;
}

bool isNearlyHorizontal(const double angle_deg) {
    return angle_deg < g_min_line_angle_deg || angle_deg > g_max_line_angle_deg;
}

bool hasExcessiveJump(const ImageLineFit& previous_line, const ImageLineFit& current_line,
                      const int rows, const int cols) {
    if (!previous_line.valid) return false;

    const int v_top = topRow(rows);
    const int v_bottom = rows - 1;
    const double prev_top = previous_line.a * static_cast<double>(v_top) + previous_line.b;
    const double curr_top = current_line.a * static_cast<double>(v_top) + current_line.b;
    const double prev_bottom = previous_line.a * static_cast<double>(v_bottom) + previous_line.b;
    const double curr_bottom = current_line.a * static_cast<double>(v_bottom) + current_line.b;

    const double top_jump = std::abs(curr_top - prev_top);
    const double bottom_jump = std::abs(curr_bottom - prev_bottom);
    const double top_threshold = std::max(3.0 * g_margin_far, 0.30 * static_cast<double>(cols));
    const double bottom_threshold = std::max(3.0 * g_margin_near, 0.30 * static_cast<double>(cols));
    return top_jump > top_threshold && bottom_jump > bottom_threshold;
}

// =============================================================================
// 【模块 J：tracking 状态更新实现】
// 信息流位置：D.2.6 或失败分支。
// =============================================================================

void updateStoredLine(ImageLineFit& stored_line, const ImageLineFit& current_line) {
    if (!stored_line.valid) {
        stored_line = current_line;
    } else {
        stored_line.a = g_line_update_alpha * current_line.a + (1.0 - g_line_update_alpha) * stored_line.a;
        stored_line.b = g_line_update_alpha * current_line.b + (1.0 - g_line_update_alpha) * stored_line.b;
        stored_line.valid = true;
    }
    stored_line.lost_count = 0;
}

void registerDetectionFailure(ImageLineFit& stored_line) {
    stored_line.lost_count += 1;
    if (stored_line.lost_count > g_max_lost_frames) {
        stored_line.valid = false;
    }
}

// =============================================================================
// 【模块 K：像素到 vehicle frame 转换实现】
// 信息流位置：D.2.3。
// =============================================================================

bool pixelToVehicleGroundPoint(const cv::Point& pixel, const cv::Mat& camera_matrix_inv,
                               const tf2::Transform& vehicle_T_camera, Eigen::Vector3d& point_out) {
    if (camera_matrix_inv.empty() || camera_matrix_inv.rows != 3 || camera_matrix_inv.cols != 3) return false;

    const double u = static_cast<double>(pixel.x);
    const double v = static_cast<double>(pixel.y);

    const double x_cam = camera_matrix_inv.at<double>(0, 0) * u +
                         camera_matrix_inv.at<double>(0, 1) * v +
                         camera_matrix_inv.at<double>(0, 2);
    const double y_cam = camera_matrix_inv.at<double>(1, 0) * u +
                         camera_matrix_inv.at<double>(1, 1) * v +
                         camera_matrix_inv.at<double>(1, 2);
    const double z_cam = camera_matrix_inv.at<double>(2, 0) * u +
                         camera_matrix_inv.at<double>(2, 1) * v +
                         camera_matrix_inv.at<double>(2, 2);

    tf2::Vector3 dir_cam(x_cam, y_cam, z_cam);
    if (dir_cam.length2() < 1e-12) return false;

    const tf2::Vector3 origin_vehicle = vehicle_T_camera.getOrigin();
    const tf2::Matrix3x3 rotation_vehicle_from_camera(vehicle_T_camera.getRotation());
    const tf2::Vector3 dir_vehicle = rotation_vehicle_from_camera * dir_cam;

    const double dz = dir_vehicle.z();
    if (std::abs(dz) < 1e-9) return false;

    const double t = -origin_vehicle.z() / dz;
    if (!std::isfinite(t) || t <= 0.0) return false;

    const tf2::Vector3 hit = origin_vehicle + t * dir_vehicle;
    if (!std::isfinite(hit.x()) || !std::isfinite(hit.y()) || !std::isfinite(hit.z())) return false;

    point_out = Eigen::Vector3d(hit.x(), hit.y(), hit.z());
    return true;
}

std::vector<Eigen::Vector3d> pixelsToVehiclePoints(const std::vector<cv::Point>& pixels,
                                                   const cv::Mat& camera_matrix_inv,
                                                   const tf2::Transform& vehicle_T_camera) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(pixels.size());
    for (const auto& pixel : pixels) {
        Eigen::Vector3d point;
        if (pixelToVehicleGroundPoint(pixel, camera_matrix_inv, vehicle_T_camera, point)) {
            points.emplace_back(point);
        }
    }
    return points;
}

// =============================================================================
// 【模块 L：vehicle ROI crop 和二次拟合实现】
// 信息流位置：D.2.4 和 D.2.5。
// =============================================================================

std::vector<Eigen::Vector3d> cropVehiclePoints(const std::vector<Eigen::Vector3d>& points,
                                               const double xmin, const double xmax,
                                               const double ymin, const double ymax) {
    std::vector<Eigen::Vector3d> cropped;
    cropped.reserve(points.size());
    for (const auto& p : points) {
        if (p.x() >= xmin && p.x() <= xmax && p.y() >= ymin && p.y() <= ymax) {
            cropped.emplace_back(p);
        }
    }
    return cropped;
}

bool fitQuadraticVehicleLine(const std::vector<Eigen::Vector3d>& input_points,
                             const double xmin, const double xmax, const double spacing,
                             std::vector<Eigen::Vector3d>& output_points) {
    output_points.clear();
    if (input_points.size() < 3 || xmax <= xmin || spacing <= 0.0) return false;

    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();

    for (const auto& p : input_points) {
        const double x = p.x();
        const double y = p.y();
        if (!std::isfinite(x) || !std::isfinite(y)) continue;

        const Eigen::Vector3d basis(1.0, x, x * x);
        normal += basis * basis.transpose();
        rhs += basis * y;
    }

    Eigen::FullPivLU<Eigen::Matrix3d> lu(normal);
    if (!lu.isInvertible()) return false;
    const Eigen::Vector3d coeff = lu.solve(rhs);

    for (double x = xmin; x <= xmax + 1e-9; x += spacing) {
        const double y = coeff[0] + coeff[1] * x + coeff[2] * x * x;
        if (!std::isfinite(y)) continue;
        output_points.emplace_back(x, y, 0.0);
    }

    return !output_points.empty();
}

// =============================================================================
// 【模块 M：center 生成实现】
// 信息流位置：D.3。
// =============================================================================

void makeCenterFromSides(const std::vector<Eigen::Vector3d>& left_points,
                         const std::vector<Eigen::Vector3d>& right_points,
                         std::vector<Eigen::Vector3d>& center_points) {
    center_points.clear();
    if (left_points.empty() || right_points.empty()) return;

    const std::size_t n = std::min(left_points.size(), right_points.size());
    center_points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        center_points.emplace_back(0.5 * (left_points[i] + right_points[i]));
    }
}

// =============================================================================
// 【模块 N：debug image 绘制实现】
// 信息流位置：E 中的 publishAnnotatedMask。
// =============================================================================

cv::Scalar sideColor(const int side_index) {
    return side_index == LEFT_INDEX ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
}

void drawInitRoiBorder(cv::Mat& image, const int side_index) {
    const int rows = image.rows;
    const int cols = image.cols;
    const int v_top = topRow(rows);
    const int u_begin = (side_index == LEFT_INDEX) ? 0 : cols / 2;
    const int u_end = (side_index == LEFT_INDEX) ? cols / 2 : cols - 1;
    cv::rectangle(image, cv::Point(u_begin, v_top), cv::Point(u_end, rows - 1), sideColor(side_index), 1);
}

void drawTrackingRoiBorder(cv::Mat& image, const ImageLineFit& line, const int side_index) {
    const int rows = image.rows;
    const int cols = image.cols;
    const int v_top = topRow(rows);

    std::vector<cv::Point> left_boundary;
    std::vector<cv::Point> right_boundary;
    left_boundary.reserve(rows - v_top);
    right_boundary.reserve(rows - v_top);

    for (int v = v_top; v < rows; ++v) {
        const double u_pred = line.a * static_cast<double>(v) + line.b;
        const double margin = marginAtRow(v, rows);
        const int u_min = clampInt(static_cast<int>(std::floor(u_pred - margin)), 0, cols - 1);
        const int u_max = clampInt(static_cast<int>(std::ceil(u_pred + margin)), 0, cols - 1);
        left_boundary.emplace_back(u_min, v);
        right_boundary.emplace_back(u_max, v);
    }

    if (left_boundary.size() >= 2) cv::polylines(image, left_boundary, false, sideColor(side_index), 1);
    if (right_boundary.size() >= 2) cv::polylines(image, right_boundary, false, sideColor(side_index), 1);
}

void drawSparsePixels(cv::Mat& image, const std::vector<cv::Point>& pixels, const int side_index) {
    const cv::Scalar color = sideColor(side_index);
    const std::size_t step = std::max<std::size_t>(1, pixels.size() / 2000 + 1);
    for (std::size_t i = 0; i < pixels.size(); i += step) {
        const auto& p = pixels[i];
        if (p.x >= 0 && p.x < image.cols && p.y >= 0 && p.y < image.rows) {
            image.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(static_cast<uchar>(color[0]),
                                                      static_cast<uchar>(color[1]),
                                                      static_cast<uchar>(color[2]));
        }
    }
}

cv::Mat makeAnnotatedMaskImage(const cv::Mat& cleaned_mask) {
    cv::Mat annotated;
    cv::cvtColor(cleaned_mask, annotated, cv::COLOR_GRAY2BGR);

    for (int side = LEFT_INDEX; side <= RIGHT_INDEX; ++side) {
        const auto& info = g_debug_info[side];
        if (info.use_tracking_roi && info.has_line) {
            drawTrackingRoiBorder(annotated, info.line, side);
        } else {
            drawInitRoiBorder(annotated, side);
        }
        drawSparsePixels(annotated, info.extracted_pixels, side);
    }

    return annotated;
}

}  // namespace
}  // namespace aiformula
