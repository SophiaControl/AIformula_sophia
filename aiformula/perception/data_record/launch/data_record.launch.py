from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

bag_topics = [
    '/planned_path', '/odom',
    '/controller/trajectory', '/planner/metrics'
]

def generate_launch_description():
    return LaunchDescription([
        Node(package='data_record',
             executable='metrics_collector_node',
             name='metrics_collector'),

        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', 'metrics_bag'] + bag_topics,
            output='screen')
    ])
