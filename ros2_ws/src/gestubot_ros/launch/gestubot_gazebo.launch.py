"""
Launch file: starts Gazebo with TurtleBot3, gesture_publisher, and gesture_bridge.

Usage:
    export TURTLEBOT3_MODEL=burger
    ros2 launch gestubot_ros gestubot_gazebo.launch.py

Make sure turtlebot3_gazebo is installed:
    sudo apt install ros-humble-turtlebot3-gazebo
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # default to burger if TURTLEBOT3_MODEL not set
    tb3_model = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    # turtlebot3 gazebo launch
    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_dir, 'launch', 'empty_world.launch.py')
        )
    )

    # gesture publisher (ML pipeline + camera)
    gesture_pub = Node(
        package='gestubot_ros',
        executable='gesture_publisher.py',
        name='gesture_publisher',
        output='screen',
        parameters=[{
            'camera_index': 0,
            'confidence_threshold': 0.7,
            'publish_rate': 30.0,
        }]
    )

    # gesture bridge (gesture -> /cmd_vel)
    gesture_bridge = Node(
        package='gestubot_ros',
        executable='gesture_bridge.py',
        name='gesture_bridge',
        output='screen',
        parameters=[{
            'linear_speed': 0.3,
            'angular_speed': 0.5,
            'timeout_sec': 0.5,
        }]
    )

    return LaunchDescription([
        gazebo_launch,
        gesture_pub,
        gesture_bridge,
    ])
