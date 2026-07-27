# OAK-D Lite AprilTag Detector

This node provides RGB imaging, point cloud generation, and AprilTag detection for a specific OAK-D Lite camera.

## Device Information

- **Camera MxID**: `1944301081303C1200`
- **Type**: OAK-D Lite

## Features

1. **RGB Image Stream**: High-resolution color images
2. **Point Cloud**: 3D point cloud from stereo depth
3. **AprilTag Detection**: Real-time detection with 3D pose estimation
4. **Aligned Depth**: Depth map aligned to RGB camera

## Installation

### Dependencies

Install the AprilTag detection library:

```bash
pip install dt-apriltags
```

Or in your virtual environment:

```bash
source ~/sigyn-venv/bin/activate
pip install dt-apriltags
```

### Build

```bash
cd ~/sigyn_ws
colcon build --packages-select sigyn_oakd_detection
source install/setup.bash
```

## Usage

### Basic Launch

```bash
ros2 launch sigyn_oakd_detection oakd_apriltag.launch.py
```

### Launch with RViz Visualization

```bash
ros2 launch sigyn_oakd_detection oakd_apriltag.launch.py use_rviz:=true
```

### Custom Parameters

```bash
ros2 launch sigyn_oakd_detection oakd_apriltag.launch.py \
    camera_mx_id:=1944301081303C1200 \
    camera_frame:=oakd_apriltag_optical_frame \
    rgb_resolution:=1080p \
    fps:=30 \
    apriltag_family:=tag36h11 \
    tag_size_m:=0.166
```

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/oakd_apriltag_node/rgb_image` | `sensor_msgs/Image` | RGB camera feed |
| `/oakd_apriltag_node/depth_image` | `sensor_msgs/Image` | Aligned depth map (16-bit) |
| `/oakd_apriltag_node/points` | `sensor_msgs/PointCloud2` | 3D point cloud |
| `/oakd_apriltag_node/camera_info` | `sensor_msgs/CameraInfo` | Camera calibration |
| `/oakd_apriltag_node/detections` | `vision_msgs/Detection3DArray` | AprilTag detections with 3D poses |
| `/oakd_apriltag_node/annotated_image` | `sensor_msgs/Image` | RGB with detection overlays |

## Parameters

### Camera Configuration

- **camera_mx_id** (string, default: `"1944301081303C1200"`): Device MxID to connect to
- **camera_frame** (string, default: `"oakd_apriltag_optical_frame"`): TF frame name
- **rgb_resolution** (string, default: `"1080p"`): Options: `1080p`, `4k`, `12mp`, `13mp`
- **depth_resolution** (string, default: `"400p"`): Options: `400p`, `480p`, `720p`, `800p`
- **fps** (int, default: `30`): Camera frame rate

### Point Cloud Configuration

- **point_cloud_publish_every** (int, default: `2`): Publish every N frames
- **point_cloud_stride** (int, default: `4`): Downsample factor
- **point_cloud_max_depth_m** (float, default: `5.0`): Maximum depth in meters

### AprilTag Detection

- **apriltag_family** (string, default: `"tag36h11"`): Tag family to detect
  - Options: `tag16h5`, `tag25h9`, `tag36h11`, `tagCircle21h7`, `tagStandard41h12`
- **apriltag_quad_decimate** (float, default: `2.0`): Speed vs. distance tradeoff
- **apriltag_quad_sigma** (float, default: `0.0`): Gaussian blur for noise reduction
- **apriltag_refine_edges** (bool, default: `true`): Improve accuracy (slower)
- **apriltag_decode_sharpening** (float, default: `0.25`): Decode sharpening
- **apriltag_max_hamming** (int, default: `1`): Error correction threshold
- **tag_size_m** (float, default: `0.166`): Physical tag size in meters (for pose estimation)

## AprilTag Tag Size

The default tag size (`0.166 m` or `6.5 inches`) is a common size for AprilTags. **You must set this to match your physical tags** for accurate pose estimation.

To measure your tags:
- Measure the outer black square edge-to-edge
- Convert to meters
- Update the `tag_size_m` parameter

Common sizes:
- Small tags: `0.088 m` (3.5")
- Medium tags: `0.127 m` (5")
- Large tags: `0.166 m` (6.5")
- Extra large: `0.216 m` (8.5")

## Example: View Point Cloud in RViz

1. Launch the node:
   ```bash
   ros2 launch sigyn_oakd_detection oakd_apriltag.launch.py
   ```

2. In another terminal, launch RViz:
   ```bash
   rviz2 -d ~/sigyn_ws/src/sigyn_oakd_detection/config/apriltag_detection.rviz
   ```

3. You should see:
   - RGB image
   - Annotated image with AprilTag overlays
   - Point cloud in 3D view

## Troubleshooting

### Camera Not Found

If you see "Camera with MxID ... not found", check available cameras:

```bash
python3 -c "import depthai as dai; [print(f'{d.getMxId()}') for d in dai.Device.getAllAvailableDevices()]"
```

### No AprilTags Detected

- Ensure tags are in the `tag36h11` family (or change the `apriltag_family` parameter)
- Check lighting conditions
- Verify tags are clearly visible and not too far away
- Try reducing `apriltag_quad_decimate` for better long-distance detection

### Low Frame Rate

- Reduce `rgb_resolution` to `1080p` or lower
- Increase `point_cloud_publish_every` to reduce bandwidth
- Increase `point_cloud_stride` to downsample more aggressively
- Set `apriltag_quad_decimate` to a higher value (e.g., `3.0` or `4.0`)

## Differences from `oakd_detector_node`

This node is **separate** from the existing `oakd_detector_node.py` which handles YOLO can detection:

| Feature | `oakd_detector_node` | `oakd_apriltag_node` |
|---------|---------------------|---------------------|
| Purpose | Can detection (YOLO) | AprilTag detection |
| Camera | Main OAK-D | OAK-D Lite (ID: 194...) |
| Detection | Neural network | AprilTag algorithm |
| Output | Can positions | Tag poses (6-DOF) |

Both nodes can run simultaneously on different cameras.

## References

- [AprilTag Paper](https://april.eecs.umich.edu/software/apriltag)
- [dt-apriltags Library](https://github.com/duckietown/lib-dt-apriltags)
- [DepthAI Documentation](https://docs.luxonis.com/)
