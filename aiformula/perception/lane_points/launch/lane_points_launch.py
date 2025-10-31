from launch import LaunchDescription  
from launch_ros.actions import Node   


def generate_launch_description():     # 自动生成launch文件的函数
   
   return LaunchDescription([          # 返回launch文件的描述信息
      Node(                            # 配置一个节点的启动
         package='lane_points',        # 节点所在的功能包
         executable='lane_zuizhong',  # 节点的可执行文件名
         name='node1',                   # 对节点重新命名
      ),
      Node(                            # 配置一个节点的启动
         package='kalman_filter',        # 节点所在的功能包
         executable='kalman_filter_node4',  # 节点的可执行文件名
         name='node2',                   # 对节点重新命名
      ),
      
   ])
