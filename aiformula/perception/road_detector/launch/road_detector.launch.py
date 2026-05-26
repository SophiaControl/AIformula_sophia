import os.path as osp
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
from common_python.launch_util import get_frame_ids_and_topic_names
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    PACKAGE_NAME = "road_detector"
    _, TOPIC_NAMES = get_frame_ids_and_topic_names()
    ROS_PARAM_CONFIG = (
        osp.join(get_package_share_directory(PACKAGE_NAME), "config", "normalization.yaml"),
    )

    launch_args = (
        DeclareLaunchArgument(
            "onnx_path",
            default_value=osp.join(
                get_package_share_directory(PACKAGE_NAME),
                "weights",
                "yolopv2.onnx",
            ),
            description="Path to the YOLOPv2 ONNX file.",
        ),
        DeclareLaunchArgument(
            "use_architecture",
            default_value="cuda",   # 改成 "cuda" 就走 GPU provider
            description="cpu or cuda",
        ),
        DeclareLaunchArgument(
            "ll_threshold",
            default_value="0.5",
            description="Lane-line threshold for ll output (single-channel).",
        ),
    )

    road_detector = Node(
        package=PACKAGE_NAME,
        executable=PACKAGE_NAME,
        name=PACKAGE_NAME,
        namespace="/aiformula_perception",
        output="screen",
        parameters=[
            [*ROS_PARAM_CONFIG],
            {
                "onnx_path": LaunchConfiguration("onnx_path"),
                "use_architecture": LaunchConfiguration("use_architecture"),
                "ll_threshold": LaunchConfiguration("ll_threshold"),
            },
        ],
        remappings=[
            ("sub_image", TOPIC_NAMES["sensing"]["zedx"]["left_image"]["undistorted"]),
            ("pub_mask_image_roi", TOPIC_NAMES["perception"]["mask_image"]),
            ("pub_annotated_mask_image", TOPIC_NAMES["visualization"]["annotated_mask_image"]),
        ],
    )
	##("pub_mask_image", TOPIC_NAMES["perception"]["mask_image"]),
    return LaunchDescription([
        *launch_args,
        road_detector,
    ])

