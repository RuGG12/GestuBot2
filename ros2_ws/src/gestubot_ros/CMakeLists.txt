cmake_minimum_required(VERSION 3.8)
project(gestubot_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GestureStamped.msg"
  DEPENDENCIES std_msgs
)

# python nodes
ament_python_install_package(gestubot_ros)

install(PROGRAMS
  gestubot_ros/gesture_publisher.py
  gestubot_ros/gesture_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
