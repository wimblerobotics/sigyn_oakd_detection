#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Wimblerobotics
"""OAK-D Lite AprilTag detector with RGB, depth, and point cloud output.

This node connects to a specific OAK-D Lite camera and provides:
- RGB image stream
- Aligned depth point cloud
- AprilTag detections with 3D pose estimates

Topics published
----------------
/oakd_apriltag/rgb_image           sensor_msgs/Image
/oakd_apriltag/depth_image         sensor_msgs/Image
/oakd_apriltag/points              sensor_msgs/PointCloud2
/oakd_apriltag/camera_info         sensor_msgs/CameraInfo
/oakd_apriltag/detections          vision_msgs/Detection3DArray
/oakd_apriltag/annotated_image     sensor_msgs/Image (with AprilTag overlays)

Parameters
----------
camera_mx_id : str
    Device MxID to connect to (e.g., '1944301081303C1200').
camera_frame : str (default: 'oakd_apriltag_optical_frame')
    TF frame name for the RGB optical center.
rgb_resolution : str (default: '1080p')
    RGB camera resolution: '1080p', '4k', '12mp', '13mp'.
depth_resolution : str (default: '400p')
    Stereo depth resolution: '400p', '480p', '720p', '800p'.
fps : int (default: 30)
    Camera frame rate.
point_cloud_publish_every : int (default: 2)
    Publish point cloud every N frames to reduce bandwidth.
point_cloud_stride : int (default: 4)
    Downsample point cloud by this factor.
point_cloud_max_depth_m : float (default: 5.0)
    Maximum depth for point cloud generation.
apriltag_family : str (default: 'tag36h11')
    AprilTag family: 'tag16h5', 'tag25h9', 'tag36h11', 'tagCircle21h7', 'tagStandard41h12'.
apriltag_quad_decimate : float (default: 2.0)
    Detection speed improvement at cost of distance.
apriltag_quad_sigma : float (default: 0.0)
    Gaussian blur sigma for noise reduction.
apriltag_refine_edges : bool (default: True)
    Improve detection accuracy at cost of speed.
apriltag_decode_sharpening : float (default: 0.25)
    Sharpening for decode.
apriltag_max_hamming : int (default: 1)
    Maximum Hamming distance for error correction.
apriltag_detect_every : int (default: 10)
    Run AprilTag detection every N frames to reduce CPU and stderr flood.
tag_size_m : float (default: 0.166)
    Physical size of AprilTags in meters (for pose estimation).
"""

import math
import threading
from typing import Dict, List, Optional, Tuple

import cv2
from cv_bridge import CvBridge
import depthai as dai
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, TransformStamped, Vector3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
import tf2_ros

try:
    import pupil_apriltags as apriltag
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False


class OakdApriltagNode(Node):
    """ROS 2 node for OAK-D Lite with RGB, depth, point cloud, and AprilTag detection."""

    def __init__(self):
        super().__init__("oakd_apriltag_node")

        # Declare parameters
        self.declare_parameter("camera_mx_id", "1944301081303C1200")
        self.declare_parameter("camera_frame", "oakd_apriltag_optical_frame")
        self.declare_parameter("rgb_resolution", "1080p")
        self.declare_parameter("depth_resolution", "400p")
        self.declare_parameter("fps", 30)
        self.declare_parameter("point_cloud_publish_every", 2)
        self.declare_parameter("point_cloud_stride", 4)
        self.declare_parameter("point_cloud_max_depth_m", 5.0)
        self.declare_parameter("apriltag_family", "tag36h11")
        self.declare_parameter("apriltag_quad_decimate", 2.0)
        self.declare_parameter("apriltag_quad_sigma", 0.0)
        self.declare_parameter("apriltag_refine_edges", True)
        self.declare_parameter("apriltag_decode_sharpening", 0.25)
        self.declare_parameter("apriltag_max_hamming", 1)
        self.declare_parameter("apriltag_detect_every", 10)  # Detect every N frames
        self.declare_parameter("tag_size_m", 0.120) # 0.166)

        # Get parameters
        self.camera_mx_id = self.get_parameter("camera_mx_id").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.rgb_resolution = self.get_parameter("rgb_resolution").value
        self.depth_resolution = self.get_parameter("depth_resolution").value
        self.fps = self.get_parameter("fps").value
        self.point_cloud_publish_every = self.get_parameter("point_cloud_publish_every").value
        self.point_cloud_stride = self.get_parameter("point_cloud_stride").value
        self.point_cloud_max_depth_m = self.get_parameter("point_cloud_max_depth_m").value
        self.apriltag_family = self.get_parameter("apriltag_family").value
        self.apriltag_quad_decimate = self.get_parameter("apriltag_quad_decimate").value
        self.apriltag_quad_sigma = self.get_parameter("apriltag_quad_sigma").value
        self.apriltag_refine_edges = self.get_parameter("apriltag_refine_edges").value
        self.apriltag_decode_sharpening = self.get_parameter("apriltag_decode_sharpening").value
        self.apriltag_max_hamming = self.get_parameter("apriltag_max_hamming").value
        self.apriltag_detect_every = self.get_parameter("apriltag_detect_every").value
        self.tag_size_m = self.get_parameter("tag_size_m").value

        # Check AprilTag library availability
        if not APRILTAG_AVAILABLE:
            self.get_logger().error(
                "pupil_apriltags library not available. Install with: pip install pupil-apriltags"
            )
            raise RuntimeError("pupil_apriltags library required but not found")

        # Initialize AprilTag detector
        self.at_detector = apriltag.Detector(
            families=self.apriltag_family,
            nthreads=2,
            quad_decimate=self.apriltag_quad_decimate,
            quad_sigma=self.apriltag_quad_sigma,
            refine_edges=1 if self.apriltag_refine_edges else 0,
            decode_sharpening=self.apriltag_decode_sharpening,
            debug=0,
        )

        # Create publishers
        self.rgb_pub = self.create_publisher(Image, "~/rgb_image", qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, "~/depth_image", qos_profile_sensor_data)
        self.points_pub = self.create_publisher(PointCloud2, "~/points", qos_profile_sensor_data)
        self.camera_info_pub = self.create_publisher(CameraInfo, "~/camera_info", qos_profile_sensor_data)
        self.detections_pub = self.create_publisher(Detection3DArray, "~/detections", 10)
        self.annotated_pub = self.create_publisher(Image, "~/annotated_image", qos_profile_sensor_data)

        # CV Bridge
        self.bridge = CvBridge()

        # Frame counter for point cloud throttling
        self.frame_count = 0
        
        # Frame counter for AprilTag detection throttling
        self.detection_frame_count = 0
        self.last_detections = []

        # Camera intrinsics (will be filled when pipeline starts)
        self.camera_intrinsics: Optional[np.ndarray] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Start DepthAI pipeline in a separate thread
        self.pipeline_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.pipeline_thread.start()

        self.get_logger().info(
            f"OAK-D AprilTag node initialized for camera {self.camera_mx_id}"
        )

    def _create_pipeline(self) -> dai.Pipeline:
        """Create the DepthAI pipeline for RGB, depth, and stereo cameras."""
        pipeline = dai.Pipeline()

        # RGB camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(self._parse_rgb_resolution(self.rgb_resolution))
        cam_rgb.setFps(self.fps)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Stereo depth
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(self._parse_mono_resolution(self.depth_resolution))
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_left.setFps(self.fps)

        mono_right.setResolution(self._parse_mono_resolution(self.depth_resolution))
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_right.setFps(self.fps)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # XLink outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.isp.link(xout_rgb.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        return pipeline

    def _parse_rgb_resolution(self, res_str: str) -> dai.ColorCameraProperties.SensorResolution:
        """Parse RGB resolution string to DepthAI enum."""
        mapping = {
            "1080p": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
            "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
            "12mp": dai.ColorCameraProperties.SensorResolution.THE_12_MP,
            "13mp": dai.ColorCameraProperties.SensorResolution.THE_13_MP,
        }
        return mapping.get(res_str.lower(), dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    def _get_rgb_resolution_dimensions(self, res_str: str) -> Tuple[int, int]:
        """Get width and height for RGB resolution string."""
        mapping = {
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
            "12mp": (4056, 3040),
            "13mp": (4208, 3120),
        }
        return mapping.get(res_str.lower(), (1920, 1080))

    def _parse_mono_resolution(self, res_str: str) -> dai.MonoCameraProperties.SensorResolution:
        """Parse mono resolution string to DepthAI enum."""
        mapping = {
            "400p": dai.MonoCameraProperties.SensorResolution.THE_400_P,
            "480p": dai.MonoCameraProperties.SensorResolution.THE_480_P,
            "720p": dai.MonoCameraProperties.SensorResolution.THE_720_P,
            "800p": dai.MonoCameraProperties.SensorResolution.THE_800_P,
        }
        return mapping.get(res_str.lower(), dai.MonoCameraProperties.SensorResolution.THE_400_P)

    def _run_pipeline(self):
        """Run the DepthAI pipeline and process frames."""
        try:
            # Find device by MxID
            device_info = None
            for info in dai.Device.getAllAvailableDevices():
                if info.getMxId() == self.camera_mx_id:
                    device_info = info
                    break

            if device_info is None:
                self.get_logger().error(
                    f"Camera with MxID {self.camera_mx_id} not found. Available devices:"
                )
                for info in dai.Device.getAllAvailableDevices():
                    self.get_logger().info(f"  - {info.getMxId()}")
                return

            self.get_logger().info(f"Connecting to camera {device_info.getMxId()}")

            # Create pipeline and device
            pipeline = self._create_pipeline()
            with dai.Device(pipeline, device_info, usb2Mode=False) as device:
                self.get_logger().info("DepthAI pipeline started")

                # Get camera calibration for the actual resolution being used
                width, height = self._get_rgb_resolution_dimensions(self.rgb_resolution)
                calib = device.readCalibration()
                intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height)
                self.camera_matrix = np.array(intrinsics).reshape(3, 3)
                self.dist_coeffs = np.zeros(5)  # OAK-D provides rectified images
                
                self.get_logger().info(f"Camera matrix for {width}x{height}:\n{self.camera_matrix}")

                # Get output queues
                q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

                while rclpy.ok():
                    # Get RGB frame
                    in_rgb = q_rgb.get()
                    rgb_frame = in_rgb.getCvFrame()

                    # Get depth frame
                    in_depth = q_depth.get()
                    depth_frame = in_depth.getFrame()

                    # Process frames
                    timestamp = self.get_clock().now().to_msg()
                    self._process_frames(rgb_frame, depth_frame, timestamp)

        except Exception as e:
            self.get_logger().error(f"Pipeline error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _process_frames(self, rgb_frame: np.ndarray, depth_frame: np.ndarray, timestamp):
        """Process RGB and depth frames, detect AprilTags, and publish outputs."""
        header = Header()
        header.stamp = timestamp
        header.frame_id = self.camera_frame

        # Publish RGB image
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_frame, encoding="rgb8")
        rgb_msg.header = header
        self.rgb_pub.publish(rgb_msg)

        # Publish depth image
        depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, encoding="16UC1")
        depth_msg.header = header
        self.depth_pub.publish(depth_msg)

        # Publish camera info
        if self.camera_matrix is not None:
            cam_info = self._create_camera_info(header, rgb_frame.shape)
            self.camera_info_pub.publish(cam_info)

        # Detect AprilTags (throttled to reduce CPU and stderr flood)
        self.detection_frame_count += 1
        detections = self.last_detections  # Use last detections by default
        
        if self.detection_frame_count >= self.apriltag_detect_every:
            self.detection_frame_count = 0
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            detections = self.at_detector.detect(
                gray_frame,
                estimate_tag_pose=True,
                camera_params=[
                    self.camera_matrix[0, 0],  # fx
                    self.camera_matrix[1, 1],  # fy
                    self.camera_matrix[0, 2],  # cx
                    self.camera_matrix[1, 2],  # cy
                ],
                tag_size=self.tag_size_m,
            )
            self.last_detections = detections

        # Publish AprilTag detections
        if detections:
            self._publish_detections(detections, header, depth_frame)

        # Publish annotated image
        annotated_frame = self._draw_detections(rgb_frame.copy(), detections)
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="rgb8")
        annotated_msg.header = header
        self.annotated_pub.publish(annotated_msg)

        # Publish point cloud (throttled)
        self.frame_count += 1
        if self.frame_count % self.point_cloud_publish_every == 0:
            self._publish_point_cloud(depth_frame, header)

    def _create_camera_info(self, header: Header, image_shape: Tuple[int, int, int]) -> CameraInfo:
        """Create CameraInfo message from calibration data."""
        cam_info = CameraInfo()
        cam_info.header = header
        cam_info.height = image_shape[0]
        cam_info.width = image_shape[1]
        cam_info.distortion_model = "plumb_bob"
        
        if self.camera_matrix is not None:
            cam_info.k = self.camera_matrix.flatten().tolist()
            cam_info.p = [
                self.camera_matrix[0, 0], 0, self.camera_matrix[0, 2], 0,
                0, self.camera_matrix[1, 1], self.camera_matrix[1, 2], 0,
                0, 0, 1, 0,
            ]
        cam_info.d = [0.0] * 5
        cam_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        return cam_info

    def _publish_detections(self, detections: List, header: Header, depth_frame: np.ndarray):
        """Publish AprilTag detections as Detection3DArray."""
        detection_array = Detection3DArray()
        detection_array.header = header

        for detection in detections:
            det_3d = Detection3D()
            det_3d.header = header

            # Create hypothesis with tag ID
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(detection.tag_id)
            hyp.hypothesis.score = detection.decision_margin

            # Pose from AprilTag detection
            if detection.pose_R is not None and detection.pose_t is not None:
                try:
                    # Convert rotation matrix to quaternion
                    # Check for valid rotation matrix (positive determinant)
                    import numpy as np
                    det = np.linalg.det(detection.pose_R)
                    if det <= 0:
                        self.get_logger().warning(
                            f'Skipping tag {detection.tag_id}: invalid rotation matrix '
                            f'(determinant={det:.6f})'
                        )
                        continue
                    
                    rot = Rotation.from_matrix(detection.pose_R)
                    quat = rot.as_quat()  # [x, y, z, w]

                    pose = Pose()
                    pose.position.x = float(detection.pose_t[0][0])
                    pose.position.y = float(detection.pose_t[1][0])
                    pose.position.z = float(detection.pose_t[2][0])
                    pose.orientation.x = float(quat[0])
                    pose.orientation.y = float(quat[1])
                    pose.orientation.z = float(quat[2])
                    pose.orientation.w = float(quat[3])

                    hyp.pose.pose = pose
                except (ValueError, np.linalg.LinAlgError) as e:
                    self.get_logger().warning(
                        f'Skipping tag {detection.tag_id}: pose conversion failed - {e}'
                    )
                    continue

            det_3d.results.append(hyp)
            detection_array.detections.append(det_3d)

        self.detections_pub.publish(detection_array)
        
        self.get_logger().debug(
            f"Detected {len(detections)} AprilTag(s): "
            + ", ".join(f"ID {d.tag_id}" for d in detections)
        )

    def _draw_detections(self, image: np.ndarray, detections: List) -> np.ndarray:
        """Draw AprilTag detections on the image."""
        for detection in detections:
            # Draw bounding box
            corners = detection.corners.astype(int)
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            # Draw center
            center = tuple(detection.center.astype(int))
            cv2.circle(image, center, 5, (0, 0, 255), -1)

            # Draw tag ID
            cv2.putText(
                image,
                f"ID: {detection.tag_id}",
                (center[0] - 20, center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        return image

    def _publish_point_cloud(self, depth_frame: np.ndarray, header: Header):
        """Publish point cloud from depth frame."""
        if self.camera_matrix is None:
            return

        height, width = depth_frame.shape
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Create point cloud with stride
        stride = self.point_cloud_stride
        points = []

        for v in range(0, height, stride):
            for u in range(0, width, stride):
                z = depth_frame[v, u] / 1000.0  # Convert mm to meters
                if z > 0 and z < self.point_cloud_max_depth_m:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])

        if not points:
            return

        # Create PointCloud2 message
        points_array = np.array(points, dtype=np.float32)
        
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pc2_msg = PointCloud2()
        pc2_msg.header = header
        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.is_dense = True
        pc2_msg.is_bigendian = False
        pc2_msg.fields = fields
        pc2_msg.point_step = 12  # 3 floats * 4 bytes
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        pc2_msg.data = points_array.tobytes()

        self.points_pub.publish(pc2_msg)


def main(args=None):
    """Entry point for the OAK-D AprilTag node."""
    rclpy.init(args=args)
    node = OakdApriltagNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
