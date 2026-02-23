# sigyn_oakd_detection

[![Apache-2.0 License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-green)](https://docs.ros.org/en/jazzy/)

OAK-D spatial YOLO can detector for the [Sigyn](https://github.com/wimblerobotics/Sigyn)
house-patrol robot.  Runs a custom-trained YOLOv5 single-output blob directly on the
OAK-D device, maps raw DepthAI spatial coordinates into the ROS optical frame, and
publishes annotated images and 3-D detection messages consumed by the behavior tree.

---

## Hardware requirement

This package is only useful on a host with an OAK-D camera physically attached.
In the Sigyn deployment it is enabled exclusively on **sigyn7900a**.

---

## Topics published

| Topic | Type | Description |
|---|---|---|
| `/oakd_top/rgb_preview` | `sensor_msgs/Image` | Raw 416×416 BGR preview |
| `/oakd_top/annotated_image` | `sensor_msgs/Image` | RGB image with bounding-box overlay |
| `/oakd_top/depth_image` | `sensor_msgs/Image` | Colour-mapped depth (throttled) |
| `/oakd_top/can_point_camera` | `geometry_msgs/PointStamped` | Axis-mapped 3-D point in camera frame |
| `/oakd_top/can_point_raw` | `geometry_msgs/PointStamped` | Raw DepthAI 3-D point (pre-mapping) |
| `/oakd_top/can_detections` | `sigyn_oakd_detection/OakdDetection` | Rich detection with bbox, spatial coords, diagnostics |
| `/oakd/object_detector_heartbeat` | `vision_msgs/Detection2DArray` | Heartbeat consumed by the BT `WaitForDetection` node |

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `blob_path` | string | *(see launch)* | Full path to the compiled DepthAI blob |
| `camera_frame` | string | `oak_rgb_camera_optical_frame` | TF frame of the RGB optical centre |
| `spatial_axis_map` | string | `x,y,z` | Axis remapping (see below) |
| `log_tf_debug` | bool | `true` | Log camera→base_link RPY per detection |
| `debug_logging` | bool | `false` | Verbose per-frame NN output logs |
| `debug_log_interval_sec` | float | `2.0` | Minimum seconds between debug log bursts |
| `depth_publish_every` | int | `5` | Publish depth vis every N frames |
| `confidence_threshold` | float | `0.65` | Minimum detection confidence |
| `iou_threshold` | float | `0.45` | NMS IoU threshold |
| `expected_target_base` | double[] | — | Expected can position in base_link for axis-map suggestion |
| `suggest_axis_map` | bool | `true` | Log best-fitting axis map on each detection |

---

## Spatial axis mapping

DepthAI spatial coordinates are not guaranteed to match ROS optical-frame conventions.
The `spatial_axis_map` parameter remaps the raw DepthAI `(x_raw, y_raw, z_raw)` vector
into the ROS camera optical frame before TF transforms are applied.

Format: three comma-separated tokens, each an optional `-` followed by one of `x`, `y`, `z`.

The token at position *i* in the output is built from axis *letter* of the raw vector,
optionally negated:

```
spatial_axis_map: "-z,x,y"
→  out[0] = -raw[2]   (negate raw z)
   out[1] =  raw[0]
   out[2] =  raw[1]
```

**To discover the correct value for a new camera mount**, run with:

```bash
ros2 launch sigyn_oakd_detection validate_detector.launch.py \
    expected_target_base:="[0.65, 0.0, 0.6]" \
    suggest_axis_map:=true
```

The node will log the best-fitting `spatial_axis_map` on every detection.

---

## Quick-start: standalone validation

```bash
# Build
cd ~/sigyn_oakd_detection_ws
colcon build --symlink-install
source install/setup.bash

# Run with RViz
ros2 launch sigyn_oakd_detection validate_detector.launch.py
```

---

## Integration with Sigyn

In **sigyn.launch.py** the detector is controlled by `do_oakd` (default `false`):

```bash
ros2 launch base sigyn.launch.py do_oakd:=true
```

The sub-launch `base/launch/sub_launch/oakd_detector.launch.py` includes this
package's `oakd_detector.launch.py` with the canonical Sigyn topic remappings.

---

## Package layout

```
sigyn_oakd_detection/
├── CMakeLists.txt
├── package.xml
├── config/
│   ├── can_detection.rviz          # RViz config for validation
│   └── nn/
│       └── can_yolov5.json         # YOLOv5 model meta (legacy depthai-ros pipeline)
├── launch/
│   ├── oakd_detector.launch.py     # Production launch (no RViz, no RSP)
│   └── validate_detector.launch.py # Standalone validation with RViz
├── models/
│   ├── can_detector.blob           # Active custom-trained YOLOv5 blob
│   ├── can_data.yaml               # Training dataset metadata
│   ├── oakd_yolov5_v5a.blob        # Reference blob (v5a variant)
│   └── oakd_yolov5_v5a_labels.txt  # Class labels
├── msg/
│   └── OakdDetection.msg           # Rich detection message
├── sigyn_oakd_detection/
│   ├── __init__.py
│   └── oakd_detector_node.py       # Main ROS2 node
└── test/
    └── test_oakd_detector.py       # pytest unit tests (no device required)
```

---

## Model notes

The active model is `models/can_detector.blob` — a custom YOLOv5 415×416 blob
trained on a CokeZero-can dataset (`FCC4.v1i`) and compiled for 6 MyriadX shaves.

`models/oakd_yolov5_v5a.blob` is an alternate (v5a) weight set kept for reference.

`config/nn/can_yolov5.json` is the metadata file required when using this model
through the legacy **depthai-ros** `yolo_spatial_detection` pipeline.  That
pipeline is not used by this package (which bypasses depthai-ros and uses
`dai.node.NeuralNetwork` directly), but the JSON is retained for reference.

---

## Message migration note

`OakdDetection.msg` is defined in this package for self-containment.  A future
release may consolidate it into
[sigyn_interfaces](https://github.com/wimblerobotics/sigyn_interfaces) so that
this package and the Pi Hailo detector (`pi_can_detector`) share a common type.
Consumers should be written to tolerate a package rename.

---

## Dependencies

### System

```bash
pip install depthai opencv-python numpy
# or via apt:
sudo apt install ros-jazzy-depthai-ros  # for udev rules / firmware
```

### ROS2

```
rclpy  std_msgs  sensor_msgs  geometry_msgs  vision_msgs
tf2_ros  tf2_geometry_msgs  cv_bridge  image_transport
```

---

## Running tests

```bash
cd ~/sigyn_oakd_detection_ws
colcon test --packages-select sigyn_oakd_detection
colcon test-result --verbose
```

Or directly with pytest (no colcon required):

```bash
cd ~/sigyn_oakd_detection_ws/src/sigyn_oakd_detection
source ~/sigyn_oakd_detection_ws/install/setup.bash
pytest test/ -v
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
