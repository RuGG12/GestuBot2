# GestuBot

Real-time hand gesture recognition system that maps 5 gesture classes to keyboard inputs (WASD) and controls a 3D robot arm in the browser. Built with OpenCV + scikit-learn.

The full pipeline — data collection, feature extraction, SVM training, live inference — runs at roughly **15–20 ms per frame**, well within the <100 ms real-time threshold.

## How It Works

Camera feed goes through an HSV skin mask and morphological cleanup to isolate the hand. The largest contour gets passed to a feature extractor that computes a 13-dimensional vector:

- **Geometric ratios** (3): aspect ratio, extent, solidity — captures hand shape independent of size
- **Hull defect count** (1): number of significant convexity defects, correlates with extended fingers
- **Hu Moments** (7): log-transformed, rotation/scale invariant shape descriptors
- **Normalized center of mass** (2): helps distinguish left-pointing vs right-pointing gestures

These features feed into an SVM (RBF kernel) trained with GridSearchCV. A 5-frame rolling buffer does mode filtering to prevent jittery outputs.

```
Camera → HSV Mask → Contour → Feature Extraction (13-dim) → SVM → Rolling Buffer → Keyboard / WebSocket
```

## Project Structure

```
GestuBot/
├── src/
│   ├── data_collector.py      # collect + label training samples with HSV tuning
│   ├── trainer.py             # SVM training, GridSearchCV, evaluation metrics
│   ├── inference.py           # real-time inference loop with latency profiler
│   ├── utils.py               # shared feature extraction + vision pipeline
│   ├── confidence_filter.py   # prediction gating + retrain buffer logging
│   └── ws_server.py           # WebSocket server for browser visualization
├── web/
│   └── robot_arm.html         # Three.js robot arm controlled via WebSocket
├── ros2_ws/src/gestubot_ros/  # ROS 2 package for Gazebo integration
│   ├── gestubot_ros/
│   │   ├── gesture_publisher.py  # ML pipeline → /gestubot/gesture topic
│   │   └── gesture_bridge.py     # gesture → /cmd_vel Twist
│   ├── msg/GestureStamped.msg
│   └── launch/gestubot_gazebo.launch.py
├── data/                      # training data + retrain buffer
├── models/                    # trained model + confusion matrix
└── requirements.txt
```

## Getting Started

### Install

```bash
git clone https://github.com/<your-username>/GestuBot.git
cd GestuBot
pip install -r requirements.txt
```

### Collect Training Data

```bash
cd src
python data_collector.py
```

Use the HSV trackbars to isolate your hand from the background. Press `0`-`5` to label samples for each gesture class. Try to get at least 50 samples per class.

| Key | Gesture | Mapped Key |
|-----|---------|------------|
| 0 | Fist | Release all (stop) |
| 1 | Open Palm | W (forward) |
| 2 | Point Left | A (left) |
| 3 | Point Right | D (right) |
| 4 | V-Sign | S (reverse) |
| 5 | Background | None |

Press `q` when done — dataset saves to `data/gestures.csv`.

### Train

```bash
python trainer.py
```

Runs GridSearchCV over `C` and `gamma` with 5-fold stratified cross-validation. Outputs the trained model to `models/gesture_svm.joblib` and a confusion matrix to `models/confusion_matrix.png`.

### Run Inference

```bash
python inference.py
```

Shows live camera feed with contour overlay, predicted gesture, per-frame latency, and a performance indicator. Press `q` to quit — prints a full latency summary before exiting.

### 3D Robot Arm (optional)

While inference is running, open `web/robot_arm.html` in a browser. It connects over WebSocket and moves the arm based on your gestures:

- Fist → reset position
- Palm → extend arm
- Point Left/Right → rotate
- V-Sign → toggle gripper

## Latency Benchmarking

There's a headless benchmark mode that skips the GUI and just measures pipeline latency:

```bash
python src/inference.py --benchmark --frames 300
```

Prints per-stage breakdown (preprocess, contour detection, classification, debouncing) with mean, median, p95, p99. This is how I validated the <20ms claim.

## Some Design Decisions

**Why SVM over a neural net?** For a 13-dim feature vector with 5 classes, an SVM is more than sufficient and the inference time is basically zero compared to the vision pipeline. No GPU dependency either.

**Why engineered features instead of raw pixels?** Hu Moments + geometric ratios give rotation/scale invariance out of the box. A CNN would need way more data and compute for marginal gains on this problem.

**StandardScaler in the pipeline is critical.** Hu Moments live in the 10⁻³ to 10⁻⁷ range while geometric ratios are 0-2. Without normalization the SVM basically ignores the moments entirely.

**Face shield.** The vision pipeline masks out the top 30% of the frame to prevent false detection when your face is in view (skin tone overlap). Simple but effective.

**Debouncing tradeoff.** The 5-frame buffer adds ~166ms of latency at 30 FPS, but it eliminates the jittering problem where predictions flip between classes frame-to-frame. Worth it for usable keyboard control.

## Confidence Filtering

The `--confidence-threshold` flag controls how certain the SVM needs to be before a gesture gets executed:

```bash
python src/inference.py --confidence-threshold 0.8
```

When confidence is below the threshold, the prediction gets overridden to "background" (no action). This prevents the robot from acting on ambiguous gestures.

Low-confidence frames get automatically logged to `data/retrain_buffer/` with their feature vectors, timestamps, and predicted classes. You can review these later and add them to the training set — basically a lightweight active learning loop.

The confidence stats print alongside the latency report when you quit:

```
  Confidence filter (threshold=0.70):
    42/1500 predictions rejected
    Rejection rate: 2.8%
```

## Gazebo Simulation (ROS 2)

The project includes a ROS 2 package that lets you control a TurtleBot3 in Gazebo with your gestures.

### Prerequisites

- Ubuntu 22.04 + ROS 2 Humble
- TurtleBot3 packages:
  ```bash
  sudo apt install ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-teleop
  ```

### Build and run

```bash
cd ros2_ws
colcon build --packages-select gestubot_ros
source install/setup.bash

export TURTLEBOT3_MODEL=burger
ros2 launch gestubot_ros gestubot_gazebo.launch.py
```

This starts Gazebo with a TurtleBot3, the gesture publisher (camera + ML pipeline), and the bridge node that converts gestures to velocity commands.

### How it works

The ML pipeline publishes to `/gestubot/gesture` and a separate bridge node converts those to `/cmd_vel` (Twist). This keeps the ML code completely decoupled from the robot platform — swapping robots just means changing the velocity mapping.

| Gesture | Robot Action | Twist |
|---------|-------------|-------|
| Fist | Stop | (0, 0) |
| Open Palm | Drive forward | (0.3, 0) |
| Point Left | Rotate left | (0, 0.5) |
| Point Right | Rotate right | (0, -0.5) |
| V-Sign | Reverse | (-0.2, 0) |

The bridge has a safety timeout — if no gesture messages arrive for 0.5s, it stops the robot.

## Requirements

- Python 3.8+
- Webcam
- Dependencies in [requirements.txt](requirements.txt)

## License

MIT
