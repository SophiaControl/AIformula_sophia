This repository contains code from the Control Engineering Laboratory at Sophia University for the Honda AI-Formula program.

Contributors

Prof. Wenjing Cao

Students: Zhewen Zheng, Mo Chen, Hongkang Yu, Wei Zhao

Contact

For any code-related questions, feel free to email Zhewen Zheng at z-zheng-9n2@eagle.sophia.ac.jp

Usage


cd workspace/ros2_ws/src/aiformula/launchers/shellscript/
./init_sensors.sh 


(without Obsticle Avoidance)


ros2 launch launchers all_nodes.launch.py 

ros2 launch auto_launch auto_yolop_launch.py 

ros2 run trajectory_follower lya_follower_connected_omegat_global  #for lane line visualization
(ros2 run trajectory_follower lya_follower_fixedpath_record )   #for bayesian optimization





(Obsticle Avoidance)

ros2 launch launchers all_nodes.launch.py 

ros2 launch road_detector road_detector.launch.py 

ros2 launch lane_line_publisher lane_line_publisher.launch.py 

ros2 run lane_points lane_0529oa 

ros2 run kalman_filter withoutkalman_0312 

ros2 run obsticle_avoidence b_spline

ros2 run trajectory_follower lya_oa




Version Updated

1.0(2024)

lane_0215.py

kalman0225.py / withoutkalman_0312.py

b_spline.py

lya_follower_connected_omegat_global.py / lya_follower_fixedpath_record.py

2.0(2025)

motor_controller.py

lane_0529oa.py

baye opt / safe baye opt swarm based

E2E controller
