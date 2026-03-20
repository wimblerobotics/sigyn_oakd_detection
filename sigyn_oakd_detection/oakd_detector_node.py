#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Wimblerobotics
"""OAK-D YOLO can detector node for the Sigyn house-patrol robot.

Runs a DepthAI pipeline (RGB camera + stereo depth + generic neural network)
directly on an OAK-D device.  A custom-trained YOLOv5 single-output blob is
used to detect Coke cans.  Raw DepthAI spatial coordinates are axis-mapped to
the ROS camera optical frame before being published and transformed to
base_link.

Topics published
----------------
/oakd_top/rgb_preview          sensor_msgs/Image
/oakd_top/annotated_image      sensor_msgs/Image
/oakd_top/depth_image          sensor_msgs/Image  (throttled)
/oakd_top/can_point_camera     geometry_msgs/PointStamped  (mapped frame)
/oakd_top/can_point_raw        geometry_msgs/PointStamped  (pre-mapping)
/oakd_top/can_detections       sigyn_oakd_detection/OakdDetection
/oakd/object_detector_heartbeat  vision_msgs/Detection2DArray

Parameters
----------
blob_path : str
    Absolute path to the compiled DepthAI blob file.
camera_frame : str  (default: 'oak_rgb_camera_optical_frame')
    TF frame name of the RGB optical centre.
spatial_axis_map : str  (default: 'x,y,z')
    Remaps DepthAI's raw (x,y,z) to the camera optical frame axes.
    Example: '-z,x,y'  negates z and redirects axes.
log_tf_debug : bool  (default: True)
    Log roll/pitch/yaw of the camera→base_link transform each detection.
debug_logging : bool  (default: False)
    Enable verbose per-frame neural-network output logging.
debug_log_interval_sec : float  (default: 2.0)
    Minimum seconds between verbose debug log bursts.
depth_publish_every : int  (default: 5)
    Publish a depth visualisation frame only every N RGB frames.
confidence_threshold : float  (default: 0.65)
    Minimum detection confidence to keep a candidate.
iou_threshold : float  (default: 0.45)
    IoU threshold for non-maximum suppression.
expected_target_base : double[]  (optional)
    Expected [x,y,z] of the can in base_link.  When provided the node logs
    a suggested spatial_axis_map on every detection.
suggest_axis_map : bool  (default: True)
    Enable axis-map suggestion logging (requires expected_target_base).
"""

import math
import threading
import time
import traceback
from typing import List, Optional

import cv2
from cv_bridge import CvBridge
import depthai as dai
from geometry_msgs.msg import Point, PointStamped
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from sigyn_interfaces.msg import OakdDetection
from sigyn_oakd_detection.detection_utils import (
    apply_axis_map,
    axis_map_to_string,
    AxisMap,
    best_axis_map,
    non_maximum_suppression,
    parse_axis_map,
    quaternion_to_rpy,
)
from std_msgs.msg import Header
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection2DArray

# ── Constants ────────────────────────────────────────────────────────────────

# Neural-network input resolution (must match the compiled blob).
_NN_INPUT_WIDTH: int = 416
_NN_INPUT_HEIGHT: int = 416

# Approximate horizontal field-of-view of the OAK-D Lite RGB lens (degrees).
_OAKD_HFOV_DEG: float = 73.0

# Minimum bounding-box side length (pixels) below which a detection is
# considered a decode artifact.
_MIN_BOX_SIDE_PX: int = 6

# Maximum candidates forwarded to NMS to prevent O(n²) explosion on noisy output.
_MAX_NMS_CANDIDATES: int = 300

# Only the best detection per frame is published.
_MAX_FINAL_DETECTIONS: int = 1

# Throttle the "TF not yet available" warning to once per this many seconds.
_TF_WARN_INTERVAL_SEC: float = 5.0

# ── Main node ─────────────────────────────────────────────────────────────────


class OakdDetectorNode(Node):
    """ROS2 node that runs a DepthAI YOLO pipeline and publishes detections.

    The node spawns a background thread that owns the DepthAI device context.
    All ROS publishing is done from that thread; the main thread handles ROS
    spin and parameter declarations.
    """

    def __init__(self) -> None:
        """Initialise the node, declare parameters, create publishers."""
        super().__init__("oakd_detector")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("blob_path", "")
        self.declare_parameter("camera_frame", "oak_rgb_camera_optical_frame")
        self.declare_parameter("spatial_axis_map", "x,y,z")
        self.declare_parameter("log_tf_debug", True)
        self.declare_parameter("debug_logging", False)
        self.declare_parameter("debug_log_interval_sec", 2.0)
        self.declare_parameter("depth_publish_every", 5)
        self.declare_parameter("confidence_threshold", 0.65)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter(
            "expected_target_base", Parameter.Type.DOUBLE_ARRAY
        )
        self.declare_parameter("suggest_axis_map", True)

        self._blob_path: str = self.get_parameter("blob_path").value
        self._camera_frame: str = self.get_parameter("camera_frame").value
        self._axis_map: AxisMap = parse_axis_map(
            self.get_parameter("spatial_axis_map").value
        )
        self._log_tf_debug: bool = bool(self.get_parameter("log_tf_debug").value)
        self._debug_logging: bool = bool(self.get_parameter("debug_logging").value)
        self._debug_log_interval: float = float(
            self.get_parameter("debug_log_interval_sec").value
        )
        self._depth_publish_every: int = max(
            1, int(self.get_parameter("depth_publish_every").value)
        )
        self._confidence_threshold: float = float(
            self.get_parameter("confidence_threshold").value
        )
        self._iou_threshold: float = float(
            self.get_parameter("iou_threshold").value
        )
        self._expected_target_base: List[float] = list(
            self.get_parameter_or(
                "expected_target_base",
                Parameter(
                    "expected_target_base", Parameter.Type.DOUBLE_ARRAY, []
                ),
            ).value
        )
        self._suggest_axis_map: bool = bool(
            self.get_parameter("suggest_axis_map").value
        )

        if not self._blob_path:
            self.get_logger().error(
                "Parameter 'blob_path' is empty — detector will not start."
            )

        # ── Publishers ──────────────────────────────────────────────────────
        self._pub_rgb = self.create_publisher(Image, "/oakd_top/rgb_preview", 10)
        self._pub_depth = self.create_publisher(Image, "/oakd_top/depth_image", 10)
        self._pub_annotated = self.create_publisher(
            Image, "/oakd_top/annotated_image", qos_profile_sensor_data
        )
        self._pub_annotated_compressed = self.create_publisher(
            CompressedImage,
            "/oakd_top/annotated_image/compressed",
            qos_profile_sensor_data,
        )
        self._pub_point = self.create_publisher(
            PointStamped, "/oakd_top/can_point_camera", 10
        )
        self._pub_point_raw = self.create_publisher(
            PointStamped, "/oakd_top/can_point_raw", 10
        )
        self._pub_heartbeat = self.create_publisher(
            Detection2DArray, "/oakd/object_detector_heartbeat", 10
        )
        self._pub_detection = self.create_publisher(
            OakdDetection, "/oakd_top/can_detections", 10
        )

        # ── TF ──────────────────────────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── CV bridge ───────────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── Pipeline thread ─────────────────────────────────────────────────
        self._running: bool = True
        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline, daemon=True
        )
        self._pipeline_thread.start()

        self.get_logger().info(
            f"oakd_detector started: blob='{self._blob_path}', "
            f"camera_frame='{self._camera_frame}', "
            f"axis_map='{axis_map_to_string(self._axis_map)}'"
        )

    # ── DepthAI pipeline ──────────────────────────────────────────────────────

    def _build_pipeline(self) -> dai.Pipeline:
        """Construct the DepthAI pipeline graph.

        Creates RGB camera → ImageManip → NeuralNetwork, mono cameras →
        StereoDepth, and three XLinkOut nodes (rgb, nn, depth).

        Returns:
            A fully linked dai.Pipeline ready to open on a device.
        """
        pipeline = dai.Pipeline()

        # RGB camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(_NN_INPUT_WIDTH, _NN_INPUT_HEIGHT)
        cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(10)

        # Mono cameras for stereo depth
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Stereo depth aligned to RGB
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setSubpixel(False)

        # ImageManip: convert BGR preview to RGB for the NN
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        manip.setMaxOutputFrameSize(_NN_INPUT_WIDTH * _NN_INPUT_HEIGHT * 3)

        # Generic neural network (custom YOLOv5 single-output blob)
        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(self._blob_path)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)

        # XLinkOut nodes
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")

        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        # Links
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        cam_rgb.preview.link(manip.inputImage)
        manip.out.link(nn.input)
        cam_rgb.preview.link(xout_rgb.input)
        nn.out.link(xout_nn.input)
        stereo.depth.link(xout_depth.input)

        return pipeline

    def _run_pipeline(self) -> None:
        """Entry point for the background DepthAI pipeline thread.

        Loops until :attr:`_running` is False, reading RGB frames, depth
        frames, and NN outputs.  Delegates detection processing to
        :meth:`_process_nn_output`.
        """
        if not self._blob_path:
            self.get_logger().error(
                "Pipeline thread exiting: blob_path is not set."
            )
            return

        try:
            pipeline = self._build_pipeline()
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f"Pipeline build failed: {exc}")
            return

        self.get_logger().info("Opening DepthAI device…")
        try:
            with dai.Device(pipeline) as device:
                self.get_logger().info(
                    f"Connected to OAK-D id={device.getMxId()} "
                    f"usb={device.getUsbSpeed()}"
                )
                q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
                q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)
                q_depth = device.getOutputQueue(
                    "depth", maxSize=4, blocking=False
                )

                latest_bgr: Optional[np.ndarray] = None
                latest_depth: Optional[np.ndarray] = None
                depth_counter: int = 0
                last_debug_time: float = 0.0
                logged_layers: bool = False

                while self._running and rclpy.ok():
                    in_rgb = q_rgb.tryGet()
                    in_nn = q_nn.tryGet()
                    in_depth = q_depth.tryGet()

                    if in_rgb is not None:
                        latest_bgr = in_rgb.getCvFrame()
                        self._publish_rgb(latest_bgr)

                    if in_depth is not None:
                        latest_depth = in_depth.getFrame()
                        depth_counter += 1
                        if depth_counter % self._depth_publish_every == 0:
                            self._publish_depth(latest_depth)

                    if in_nn is not None and latest_bgr is not None:
                        debug_now = False
                        if self._debug_logging:
                            now = time.monotonic()
                            if now - last_debug_time >= self._debug_log_interval:
                                debug_now = True
                                last_debug_time = now

                        if not logged_layers:
                            names = in_nn.getAllLayerNames()
                            if names:
                                self.get_logger().info(
                                    f"NN output layers: {names}"
                                )
                                logged_layers = True

                        self._process_nn_output(
                            in_nn, latest_bgr, latest_depth, debug_now
                        )

                    time.sleep(0.001)

        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(
                f"Pipeline fatal error: {exc}\n{traceback.format_exc()}"
            )

    # ── Neural-network output parsing ─────────────────────────────────────────

    def _process_nn_output(
        self,
        in_nn,
        frame_bgr: np.ndarray,
        frame_depth: Optional[np.ndarray],
        debug_now: bool,
    ) -> None:
        """Parse one NeuralNetwork output packet and publish detections.

        Handles the NCD layout [1, 5, N] produced by Ultralytics YOLOv5
        single-output exports.

        Args:
            in_nn: DepthAI NNData packet.
            frame_bgr: Latest RGB frame (BGR, HWC).
            frame_depth: Latest depth frame, or None if not yet available.
            debug_now: Whether to emit verbose diagnostic logs this cycle.
        """
        try:
            layer_names = in_nn.getAllLayerNames()
            if not layer_names:
                return
            layer = "output0" if "output0" in layer_names else layer_names[0]

            raw = np.array(in_nn.getLayerFp16(layer), dtype=np.float32)
            if raw.size == 0:
                return

            # Normalise to shape (N, 5): [x_c, y_c, w, h, raw_conf]
            detections = self._reshape_output(raw, debug_now)
            if detections is None or len(detections) == 0:
                self._publish_heartbeat(frame_bgr)
                return

            if debug_now:
                confs = detections[:, 4]
                self.get_logger().debug(
                    f"Raw confidence range: [{confs.min():.3f}, {confs.max():.3f}]"
                )

            # Confidence filtering
            def sigmoid(x: np.ndarray) -> np.ndarray:
                return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

            raw_confs = detections[:, 4]
            probs = (
                sigmoid(raw_confs)
                if (raw_confs.min() < 0.0 or raw_confs.max() > 1.0)
                else raw_confs
            )
            mask = probs > self._confidence_threshold
            dets = detections[mask]
            confs = probs[mask]
            if len(dets) == 0:
                self._publish_heartbeat(frame_bgr)
                return

            # Pre-NMS cap
            if len(dets) > _MAX_NMS_CANDIDATES:
                top = np.argsort(confs)[-_MAX_NMS_CANDIDATES:]
                dets, confs = dets[top], confs[top]

            # Scale normalised coords if needed
            boxes_xywh = dets[:, :4].copy()
            if (
                np.percentile(np.abs(boxes_xywh[:, :2]), 95) <= 1.5
                and np.percentile(np.abs(boxes_xywh[:, 2:4]), 95) <= 1.5
            ):
                boxes_xywh[:, 0] *= _NN_INPUT_WIDTH
                boxes_xywh[:, 1] *= _NN_INPUT_HEIGHT
                boxes_xywh[:, 2] *= _NN_INPUT_WIDTH
                boxes_xywh[:, 3] *= _NN_INPUT_HEIGHT

            boxes_xywh[:, :4] = np.clip(
                boxes_xywh[:, :4],
                [0, 0, 1, 1],
                [
                    _NN_INPUT_WIDTH - 1,
                    _NN_INPUT_HEIGHT - 1,
                    _NN_INPUT_WIDTH,
                    _NN_INPUT_HEIGHT,
                ],
            )
            size_ok = (boxes_xywh[:, 2] >= _MIN_BOX_SIDE_PX) & (
                boxes_xywh[:, 3] >= _MIN_BOX_SIDE_PX
            )
            boxes_xywh, confs = boxes_xywh[size_ok], confs[size_ok]
            if len(boxes_xywh) == 0:
                self._publish_heartbeat(frame_bgr)
                return

            # Convert xywh→xyxy for NMS
            boxes_xyxy = np.column_stack(
                [
                    boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
                    boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
                    boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2,
                    boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2,
                ]
            )
            keep = non_maximum_suppression(
                boxes_xyxy, confs, self._iou_threshold
            )
            if not keep:
                self._publish_heartbeat(frame_bgr)
                return

            final_xyxy = boxes_xyxy[keep]
            final_confs = confs[keep]
            final_centers = boxes_xywh[keep, :2]

            # Limit to top-1
            if len(final_confs) > _MAX_FINAL_DETECTIONS:
                order = np.argsort(final_confs)[::-1][:_MAX_FINAL_DETECTIONS]
                final_xyxy = final_xyxy[order]
                final_confs = final_confs[order]
                final_centers = final_centers[order]

            annotated = frame_bgr.copy()
            for box, score, center in zip(final_xyxy, final_confs, final_centers):
                cx, cy = (
                    int(np.clip(center[0], 0, _NN_INPUT_WIDTH - 1)),
                    int(np.clip(center[1], 0, _NN_INPUT_HEIGHT - 1)),
                )
                z_mm = (
                    0.0 if frame_depth is None else float(frame_depth[cy, cx])
                )
                bbox = box.astype(int).tolist()

                if z_mm == 0.0:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2), (0, 255, 255), 2
                    )
                    cv2.putText(
                        annotated,
                        f"2D {score:.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )
                    continue

                focal = _NN_INPUT_WIDTH / (
                    2 * math.tan(math.radians(_OAKD_HFOV_DEG / 2))
                )
                x_mm = (cx - _NN_INPUT_WIDTH / 2) * z_mm / focal
                y_mm = (cy - _NN_INPUT_HEIGHT / 2) * z_mm / focal

                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{z_mm / 1000.0:.2f}m {score:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                self._publish_detection(x_mm, y_mm, z_mm, float(score), bbox)

            self._publish_annotated(annotated)
            self._publish_heartbeat_from_annotated()

        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(
                f"Error parsing NN output: {exc}\n{traceback.format_exc()}"
            )

    def _reshape_output(
        self, raw: np.ndarray, debug_now: bool
    ) -> Optional[np.ndarray]:
        """Normalise varied NN output shapes to (N, 5).

        Handles flat, 2-D, and 3-D tensors from different export variants.

        Args:
            raw: Flat float32 array from getLayerFp16.
            debug_now: Emit a shape log entry this cycle.

        Returns:
            Array of shape (N, 5) or None on failure.
        """
        if debug_now:
            self.get_logger().debug(f"Raw NN output size={raw.size}")

        num_cols = 5  # [x_c, y_c, w, h, conf]
        if raw.ndim == 1:
            if raw.size % num_cols != 0:
                self.get_logger().warning(
                    f"Flat NN output size {raw.size} not divisible by "
                    f"{num_cols}; skipping."
                )
                return None
            return raw.reshape(num_cols, raw.size // num_cols).T
        if raw.ndim == 3:
            return raw.transpose(0, 2, 1).reshape(-1, num_cols)
        if raw.ndim == 2:
            if raw.shape[1] == num_cols:
                return raw
            if raw.shape[0] == num_cols:
                return raw.T
        self.get_logger().warning(
            f"Unexpected NN output shape {raw.shape}; skipping."
        )
        return None

    # ── Detection publishing ───────────────────────────────────────────────────

    def _publish_detection(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        confidence: float,
        bbox: List[int],
    ) -> None:
        """Apply axis mapping, publish raw/mapped PointStamped and OakdDetection.

        Attempts a base_link transform; publishes camera-frame data when TF is
        not yet available.

        Args:
            x_mm: DepthAI x spatial coordinate in millimetres.
            y_mm: DepthAI y spatial coordinate in millimetres.
            z_mm: DepthAI z spatial coordinate (depth) in millimetres.
            confidence: Detection score in [0, 1].
            bbox: Bounding box [x1, y1, x2, y2] in preview-image pixels.
        """
        raw_m = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
        mapped_m = apply_axis_map(raw_m, self._axis_map)

        stamp = self.get_clock().now().to_msg()

        # Raw point (pre-mapping)
        pt_raw = PointStamped()
        pt_raw.header = Header(frame_id=self._camera_frame, stamp=stamp)
        pt_raw.point = Point(x=raw_m[0], y=raw_m[1], z=raw_m[2])
        self._pub_point_raw.publish(pt_raw)

        # Axis-mapped point
        pt_mapped = PointStamped()
        pt_mapped.header = Header(frame_id=self._camera_frame, stamp=stamp)
        pt_mapped.point = Point(x=mapped_m[0], y=mapped_m[1], z=mapped_m[2])
        self._pub_point.publish(pt_mapped)

        # TF to base_link
        pt_base: Optional[PointStamped] = None
        tf_available = False
        try:
            if self._tf_buffer.can_transform(
                "base_link", self._camera_frame, rclpy.time.Time()
            ):
                tf_available = True
                transform = self._tf_buffer.lookup_transform(
                    "base_link", self._camera_frame, rclpy.time.Time()
                )
                pt_base = tf2_geometry_msgs.do_transform_point(
                    pt_mapped, transform
                )

                if self._log_tf_debug:
                    roll, pitch, yaw = quaternion_to_rpy(
                        transform.transform.rotation
                    )
                    self.get_logger().debug(
                        f"Camera→base_link: "
                        f"t=({transform.transform.translation.x:.3f},"
                        f"{transform.transform.translation.y:.3f},"
                        f"{transform.transform.translation.z:.3f}) "
                        f"rpy=({math.degrees(roll):.1f}°,"
                        f"{math.degrees(pitch):.1f}°,"
                        f"{math.degrees(yaw):.1f}°)"
                    )

                # Optional axis-map suggestion
                if (
                    self._suggest_axis_map
                    and len(self._expected_target_base) == 3
                ):
                    self._log_axis_map_suggestion(raw_m, stamp, transform)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().debug(f"TF lookup error: {exc}")

        if not tf_available:
            now_s = time.monotonic()
            last_warn = getattr(self, "_last_tf_warn", 0.0)
            if now_s - last_warn > _TF_WARN_INTERVAL_SEC:
                self.get_logger().warning(
                    f"TF not yet available: '{self._camera_frame}' → 'base_link'"
                )
                self._last_tf_warn = now_s

        # Publish rich detection message
        if self._pub_detection is not None:
            msg = OakdDetection()
            msg.header = (
                pt_base.header
                if pt_base is not None
                else Header(frame_id=self._camera_frame, stamp=stamp)
            )
            msg.class_name = "can"
            msg.confidence = confidence
            msg.bbox_xmin = bbox[0]
            msg.bbox_ymin = bbox[1]
            msg.bbox_xmax = bbox[2]
            msg.bbox_ymax = bbox[3]
            msg.bbox_center_x = (bbox[0] + bbox[2]) // 2
            msg.bbox_center_y = (bbox[1] + bbox[3]) // 2
            msg.spatial_camera = Point(
                x=mapped_m[0], y=mapped_m[1], z=mapped_m[2]
            )
            if pt_base is not None:
                msg.spatial_base_link = pt_base.point
            else:
                msg.spatial_base_link = Point(
                    x=mapped_m[0], y=mapped_m[1], z=mapped_m[2]
                )
            msg.distance_from_camera = float(raw_m[2])
            msg.diagnostic_log = [] if tf_available else ["no_tf_base_link"]
            self._pub_detection.publish(msg)

    def _log_axis_map_suggestion(
        self, raw_m: List[float], stamp, transform
    ) -> None:
        """Log the best-fitting axis map to align raw DepthAI coords with base_link.

        Args:
            raw_m: Raw DepthAI spatial vector in metres.
            stamp: ROS timestamp for target point.
            transform: camera_frame → base_link TransformStamped.
        """
        try:
            tgt = PointStamped()
            tgt.header = Header(frame_id="base_link", stamp=stamp)
            tgt.point = Point(
                x=self._expected_target_base[0],
                y=self._expected_target_base[1],
                z=self._expected_target_base[2],
            )
            t_inv = self._tf_buffer.lookup_transform(
                self._camera_frame, "base_link", rclpy.time.Time()
            )
            tgt_cam = tf2_geometry_msgs.do_transform_point(tgt, t_inv)
            tgt_vec = [tgt_cam.point.x, tgt_cam.point.y, tgt_cam.point.z]
            suggested, err = best_axis_map(raw_m, tgt_vec)
            if suggested is not None:
                self.get_logger().info(
                    f"Suggested spatial_axis_map='{suggested}' "
                    f"(residual²={err:.4f})"
                )
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().debug(f"Axis-map suggestion failed: {exc}")

    # ── Image publishing helpers ───────────────────────────────────────────────

    def _publish_rgb(self, frame: np.ndarray) -> None:
        """Publish the raw RGB preview frame."""
        stamp = self.get_clock().now().to_msg()
        msg = self._bridge.cv2_to_imgmsg(frame, "bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self._camera_frame
        self._pub_rgb.publish(msg)

    def _publish_annotated(self, frame: np.ndarray) -> None:
        """Publish an annotated BGR frame."""
        stamp = self.get_clock().now().to_msg()
        msg = self._bridge.cv2_to_imgmsg(frame, "bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self._camera_frame
        self._pub_annotated.publish(msg)
        self._publish_annotated_compressed(frame, stamp)

    def _publish_annotated_compressed(self, frame: np.ndarray, stamp) -> None:
        """Publish a JPEG-compressed annotated frame for remote viewers."""
        success, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )
        if not success:
            self.get_logger().warning("Failed to JPEG-encode annotated frame")
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self._camera_frame
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self._pub_annotated_compressed.publish(msg)

    def _publish_depth(self, depth: np.ndarray) -> None:
        """Publish a colour-mapped depth visualisation (throttled)."""
        sub = depth[::4, ::4]
        max_val = sub.max()
        if max_val == 0:
            return
        vis = cv2.applyColorMap(
            (sub / max_val * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        msg = self._bridge.cv2_to_imgmsg(vis, "bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._camera_frame
        self._pub_depth.publish(msg)

    def _publish_heartbeat(self, frame: np.ndarray) -> None:
        """Publish passthrough annotated image and empty heartbeat."""
        self._publish_annotated(frame)
        self._publish_heartbeat_from_annotated()

    def _publish_heartbeat_from_annotated(self) -> None:
        """Publish Detection2DArray heartbeat (annotated image already published)."""
        hb = Detection2DArray()
        hb.header.stamp = self.get_clock().now().to_msg()
        hb.header.frame_id = self._camera_frame
        self._pub_heartbeat.publish(hb)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the pipeline thread to stop and wait for it to join."""
        self._running = False
        if self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=5.0)


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args=None) -> None:
    """Spin the OakdDetectorNode with a MultiThreadedExecutor."""
    rclpy.init(args=args)
    node = OakdDetectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
