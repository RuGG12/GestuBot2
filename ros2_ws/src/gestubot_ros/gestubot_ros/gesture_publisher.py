#!/usr/bin/env python3
"""
ROS 2 node that runs the GestuBot ML pipeline and publishes
gesture classifications to /gestubot/gesture.

The heavy lifting (feature extraction, SVM classification, debouncing)
is all done by the existing inference engine in src/. This node
just wraps it and publishes the results as ROS messages.
"""

import sys
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

# add project src/ to path so we can import the ML pipeline
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

import cv2
import numpy as np
from utils import (
    extract_features, preprocess_frame, find_largest_contour,
    GESTURE_CLASSES
)
from confidence_filter import ConfidenceFilter

# import after path setup
try:
    from gestubot_ros.msg import GestureStamped
except ImportError:
    # fallback for testing outside colcon workspace
    GestureStamped = None


# default HSV thresholds (same as inference.py)
DEFAULT_HSV_LOWER = (0, 30, 60)
DEFAULT_HSV_UPPER = (20, 150, 255)


class GesturePublisher(Node):
    """
    Captures camera frames, runs them through the ML pipeline,
    and publishes GestureStamped messages at ~30 Hz.
    """

    def __init__(self):
        super().__init__('gesture_publisher')

        # ROS params
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('model_path', os.path.join(PROJECT_ROOT, 'models', 'gesture_svm.joblib'))

        cam_idx = self.get_parameter('camera_index').value
        model_path = self.get_parameter('model_path').value
        conf_thresh = self.get_parameter('confidence_threshold').value
        rate = self.get_parameter('publish_rate').value

        # load the trained SVM model
        import joblib
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found at {model_path}. Run trainer.py first.')
            raise FileNotFoundError(model_path)
        self.model = joblib.load(model_path)
        self.get_logger().info(f'Loaded SVM model from {model_path}')

        # confidence filter
        self.conf_filter = ConfidenceFilter(threshold=conf_thresh)

        # prediction buffer (simple mode filter, same as inference.py)
        from collections import deque
        from statistics import mode as stat_mode
        self._buffer = deque(maxlen=5)
        self._stat_mode = stat_mode

        # HSV thresholds
        self.hsv_lower = DEFAULT_HSV_LOWER
        self.hsv_upper = DEFAULT_HSV_UPPER

        # camera
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open camera {cam_idx}')
            raise RuntimeError('Camera init failed')
        self.get_logger().info(f'Camera {cam_idx} opened')

        # publisher
        if GestureStamped is not None:
            self.pub = self.create_publisher(GestureStamped, '/gestubot/gesture', 10)
        else:
            # fallback to std_msgs if custom msg not built yet
            from std_msgs.msg import Int32
            self.pub = self.create_publisher(Int32, '/gestubot/gesture', 10)
            self.get_logger().warn('Using Int32 fallback — build the package for GestureStamped')

        # timer drives the inference loop
        period = 1.0 / rate
        self.timer = self.create_timer(period, self._timer_callback)

        # latency tracking
        self._frame_count = 0
        self._latency_sum = 0.0

    def _debounce(self, prediction: int) -> int:
        """5-frame rolling mode filter."""
        self._buffer.append(prediction)
        if len(self._buffer) < 3:
            return prediction
        try:
            return self._stat_mode(self._buffer)
        except:
            return prediction

    def _timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Frame capture failed', throttle_duration_sec=5.0)
            return

        t_start = time.perf_counter()

        # 1. preprocess
        mask, _ = preprocess_frame(frame, self.hsv_lower, self.hsv_upper)

        # 2. contour
        contour = find_largest_contour(mask)

        # 3. classify
        if contour is None:
            raw_pred = 5
            confidence = 1.0  # certain there's no hand
        else:
            features = extract_features(contour)
            if features is None:
                raw_pred = 5
                confidence = 1.0
            else:
                features_2d = features.reshape(1, -1)
                raw_pred = int(self.model.predict(features_2d)[0])
                # get confidence from predict_proba
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(features_2d)[0]
                    confidence = float(proba.max())
                else:
                    confidence = 1.0

        # 4. confidence filter
        raw_pred, confidence = self.conf_filter.filter(raw_pred, confidence)

        # 5. debounce
        filtered_pred = self._debounce(raw_pred)

        latency_ms = (time.perf_counter() - t_start) * 1000
        self._frame_count += 1
        self._latency_sum += latency_ms

        # publish
        gesture_name = GESTURE_CLASSES.get(filtered_pred, 'unknown')

        if GestureStamped is not None:
            msg = GestureStamped()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            msg.gesture_class = filtered_pred
            msg.gesture_name = gesture_name
            msg.latency_ms = latency_ms
            msg.confidence = confidence
        else:
            from std_msgs.msg import Int32
            msg = Int32()
            msg.data = filtered_pred

        self.pub.publish(msg)

        # log every 100 frames
        if self._frame_count % 100 == 0:
            avg = self._latency_sum / self._frame_count
            self.get_logger().info(
                f'Frame {self._frame_count}: gesture={gesture_name} '
                f'conf={confidence:.2f} avg_latency={avg:.1f}ms'
            )

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GesturePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
