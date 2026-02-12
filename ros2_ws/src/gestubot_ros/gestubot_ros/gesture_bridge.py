#!/usr/bin/env python3
"""
Bridge node: subscribes to /gestubot/gesture and publishes
geometry_msgs/Twist to /cmd_vel for robot control.

This keeps the ML pipeline completely decoupled from the robot.
Swap the velocity profile here to target different platforms
without touching any ML code.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    from gestubot_ros.msg import GestureStamped
    USE_CUSTOM_MSG = True
except ImportError:
    from std_msgs.msg import Int32
    USE_CUSTOM_MSG = False


# gesture -> (linear.x, angular.z)
# tuned for TurtleBot3 Burger
VELOCITY_MAP = {
    0: (0.0,  0.0),    # fist -> stop
    1: (0.3,  0.0),    # palm -> forward
    2: (0.0,  0.5),    # point left -> rotate left
    3: (0.0, -0.5),    # point right -> rotate right
    4: (-0.2, 0.0),    # v-sign -> reverse
    5: (0.0,  0.0),    # background -> no motion
}


class GestureBridge(Node):
    """
    Converts gesture classifications into velocity commands.
    Publishes zero-velocity when no gestures arrive for >0.5s (safety timeout).
    """

    def __init__(self):
        super().__init__('gesture_bridge')

        # configurable speeds
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('timeout_sec', 0.5)

        lin = self.get_parameter('linear_speed').value
        ang = self.get_parameter('angular_speed').value
        self._timeout = self.get_parameter('timeout_sec').value

        # override default speeds with param values
        self._vel_map = {
            0: (0.0,  0.0),
            1: (lin,   0.0),
            2: (0.0,   ang),
            3: (0.0,  -ang),
            4: (-lin * 0.66, 0.0),
            5: (0.0,  0.0),
        }

        # publishers/subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        if USE_CUSTOM_MSG:
            self.create_subscription(GestureStamped, '/gestubot/gesture', self._gesture_cb, 10)
        else:
            self.create_subscription(Int32, '/gestubot/gesture', self._gesture_int_cb, 10)

        # safety: stop robot if no gestures arrive
        self._last_msg_time = self.get_clock().now()
        self.create_timer(0.1, self._safety_check)

        self._current_gesture = 5  # start with no-action
        self.get_logger().info('Gesture bridge ready — listening on /gestubot/gesture')

    def _gesture_cb(self, msg: 'GestureStamped'):
        self._current_gesture = msg.gesture_class
        self._last_msg_time = self.get_clock().now()
        self._publish_twist(msg.gesture_class)

        self.get_logger().debug(
            f'{msg.gesture_name} (conf={msg.confidence:.2f}, lat={msg.latency_ms:.1f}ms)'
        )

    def _gesture_int_cb(self, msg: 'Int32'):
        """Fallback for when custom msg isn't built."""
        self._current_gesture = msg.data
        self._last_msg_time = self.get_clock().now()
        self._publish_twist(msg.data)

    def _publish_twist(self, gesture_class: int):
        lin_x, ang_z = self._vel_map.get(gesture_class, (0.0, 0.0))
        twist = Twist()
        twist.linear.x = lin_x
        twist.angular.z = ang_z
        self.cmd_pub.publish(twist)

    def _safety_check(self):
        """Stop robot if we haven't heard from the gesture node in a while."""
        elapsed = (self.get_clock().now() - self._last_msg_time).nanoseconds / 1e9
        if elapsed > self._timeout and self._current_gesture != 5:
            self.get_logger().warn('No gesture messages — stopping robot', throttle_duration_sec=2.0)
            stop = Twist()  # all zeros
            self.cmd_pub.publish(stop)
            self._current_gesture = 5


def main(args=None):
    rclpy.init(args=args)
    node = GestureBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # send final stop command
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
