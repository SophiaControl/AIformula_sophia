from launch import LaunchDescription
from launch_ros.actions import Node
from common_python.launch_util import get_frame_ids_and_topic_names
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory
from launch.substitutions import LaunchConfiguration
import os
import os.path as osp

def generate_launch_description():

    road_detector = IncludeLaunchDescription(launch_description_source= PythonLaunchDescriptionSource(
        launch_file_path=os.path.join(
            get_package_share_directory("road_detector"),
            "launch/",
            "road_detector.launch.py"
        ),
    ))
    
    lane_line_publisher = IncludeLaunchDescription(launch_description_source= PythonLaunchDescriptionSource(
        launch_file_path=os.path.join(
            get_package_share_directory("lane_line_publisher"),
            "launch/",
            "lane_line_publisher.launch.py"
        ),
    ))
    
    lane_points = Node(
        package='lane_points',
        executable='lane_0215'
    )
    
    kalman_filter = Node(
        package='kalman_filter',
        executable='withoutkalman_0312'
    )






    return LaunchDescription([
        road_detector,
        lane_line_publisher,
        lane_points,
        kalman_filter
        
    ])
