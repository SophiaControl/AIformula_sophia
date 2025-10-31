cmake_minimum_required(VERSION 3.5)
project(data_record)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PlannerMetrics.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch resource
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
