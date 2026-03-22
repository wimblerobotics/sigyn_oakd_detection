# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Wimblerobotics
"""Pure-Python detection utilities for the OAK-D can detector.

This module is intentionally free of ROS2 and DepthAI imports so it can
be used in unit tests without requiring hardware or a running ROS graph.
"""

import itertools
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ── Type aliases ─────────────────────────────────────────────────────────────

# A parsed axis map: list of (source_axis_index, sign) tuples.
AxisMap = List[Tuple[int, int]]

# ── Constants ─────────────────────────────────────────────────────────────────

_AXIS_NAMES: Tuple[str, ...] = ("x", "y", "z")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ── Axis-mapping helpers ─────────────────────────────────────────────────────


def parse_axis_map(map_str: str) -> AxisMap:
    """Parse a spatial_axis_map string into an index/sign list.

    Args:
        map_str: Comma-separated tokens like ``'-z,x,y'``.  Each token is an
            optional ``'-'`` followed by one of ``'x'``, ``'y'``, ``'z'``.

    Returns:
        A list of three ``(source_index, sign)`` tuples, or the identity map
        ``[(0,1),(1,1),(2,1)]`` if the string is malformed.
    """
    tokens = [t.strip() for t in map_str.split(",") if t.strip()]
    if len(tokens) != 3:
        return [(0, 1), (1, 1), (2, 1)]
    result: AxisMap = []
    for token in tokens:
        sign = -1 if token.startswith("-") else 1
        axis = token[1:] if token.startswith("-") else token
        if axis not in _AXIS_INDEX:
            return [(0, 1), (1, 1), (2, 1)]
        result.append((_AXIS_INDEX[axis], sign))
    return result


def apply_axis_map(raw: Sequence[float], axis_map: AxisMap) -> List[float]:
    """Apply a pre-parsed axis map to a 3-vector.

    Args:
        raw: Input vector ``[v0, v1, v2]``.
        axis_map: Output of :func:`parse_axis_map`.

    Returns:
        Remapped vector as a list of three floats.
    """
    return [sign * raw[idx] for idx, sign in axis_map]


def axis_map_to_string(axis_map: AxisMap) -> str:
    """Convert a parsed axis map back to its canonical string form.

    Args:
        axis_map: Output of :func:`parse_axis_map`.

    Returns:
        A string like ``'-z,x,y'``.
    """
    parts = []
    for idx, sign in axis_map:
        parts.append(f"-{_AXIS_NAMES[idx]}" if sign < 0 else _AXIS_NAMES[idx])
    return ",".join(parts)


def best_axis_map(
    raw_vec: Sequence[float], target_vec: Sequence[float]
) -> Tuple[Optional[str], Optional[float]]:
    """Exhaustively search for the axis permutation/sign that best maps raw→target.

    Tries all 48 combinations (6 permutations × 8 sign patterns).

    Args:
        raw_vec: DepthAI raw spatial coordinates.
        target_vec: Desired output coordinates in the camera frame.

    Returns:
        ``(axis_map_string, residual_squared_error)``, or ``(None, None)`` if
        either input is empty.
    """
    if not raw_vec or not target_vec:
        return None, None

    best_err: Optional[float] = None
    best_perm: Optional[Tuple[int, ...]] = None
    best_signs: Optional[Tuple[int, ...]] = None

    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            mapped = [signs[i] * raw_vec[perm[i]] for i in range(3)]
            err = sum((mapped[i] - target_vec[i]) ** 2 for i in range(3))
            if best_err is None or err < best_err:
                best_err = err
                best_perm = perm
                best_signs = signs

    if best_perm is None or best_signs is None:
        return None, None

    am: AxisMap = [(best_perm[i], best_signs[i]) for i in range(3)]
    return axis_map_to_string(am), best_err


# ── NMS helper ───────────────────────────────────────────────────────────────


def non_maximum_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> List[int]:
    """Greedy non-maximum suppression.

    Args:
        boxes: Float array of shape ``(N, 4)`` in ``[x1, y1, x2, y2]`` format.
        scores: Float array of shape ``(N,)``.
        iou_threshold: Boxes with IoU above this threshold are suppressed.

    Returns:
        Indices of surviving detections in descending score order.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep


# ── Depth helpers ────────────────────────────────────────────────────────────


def estimate_object_depth(
    depth_frame: np.ndarray,
    bbox_xyxy: Sequence[float],
    inner_fraction: float = 0.5,
    foreground_percentile: float = 35.0,
    foreground_margin_mm: float = 75.0,
) -> Optional[Tuple[float, float, float]]:
    """Estimate a foreground depth sample inside a detection bounding box.

    The detector's box can include a lot of background. Using the center pixel or
    the median of the whole box often returns the wall or floor behind the object.
    This helper shrinks the ROI toward the centre, then prefers the nearest stable
    cluster of nonzero depth values inside that ROI.

    Args:
        depth_frame: Aligned depth frame in millimetres.
        bbox_xyxy: Bounding box ``[x1, y1, x2, y2]`` in depth-frame pixels.
        inner_fraction: Fraction of the box width/height to keep around the
            centre when forming the sampling ROI.
        foreground_percentile: Low percentile used to identify the foreground
            cluster inside the ROI.
        foreground_margin_mm: Additional tolerance above the foreground
            percentile when forming the cluster.

    Returns:
        ``(pixel_x, pixel_y, depth_mm)`` for the selected foreground sample, or
        ``None`` if no valid depth exists in the ROI.
    """
    if depth_frame.size == 0:
        return None
    if len(bbox_xyxy) != 4:
        return None

    frame_height, frame_width = depth_frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    half_width = max(0.5, 0.5 * (x2 - x1) * max(0.05, inner_fraction))
    half_height = max(0.5, 0.5 * (y2 - y1) * max(0.05, inner_fraction))

    roi_x0 = max(0, min(frame_width - 1, int(math.floor(center_x - half_width))))
    roi_x1 = max(0, min(frame_width - 1, int(math.ceil(center_x + half_width))))
    roi_y0 = max(0, min(frame_height - 1, int(math.floor(center_y - half_height))))
    roi_y1 = max(0, min(frame_height - 1, int(math.ceil(center_y + half_height))))
    if roi_x1 < roi_x0 or roi_y1 < roi_y0:
        return None

    roi = depth_frame[roi_y0:roi_y1 + 1, roi_x0:roi_x1 + 1]
    valid_mask = roi > 0
    if not np.any(valid_mask):
        return None

    valid_depths = roi[valid_mask].astype(np.float32)
    percentile_depth = float(np.percentile(valid_depths, foreground_percentile))
    foreground_limit = percentile_depth + float(max(0.0, foreground_margin_mm))
    foreground_mask = valid_mask & (roi <= foreground_limit)
    if not np.any(foreground_mask):
        foreground_mask = valid_mask

    foreground_depths = roi[foreground_mask].astype(np.float32)
    sample_depth_mm = float(np.median(foreground_depths))

    ys, xs = np.nonzero(foreground_mask)
    sample_x = float(roi_x0 + np.median(xs))
    sample_y = float(roi_y0 + np.median(ys))
    return sample_x, sample_y, sample_depth_mm


def deproject_depth_pixel(
    pixel_x: float,
    pixel_y: float,
    depth_mm: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> List[float]:
    """Project a depth pixel into the ROS optical camera frame.

    Args:
        pixel_x: Depth-frame x coordinate in pixels.
        pixel_y: Depth-frame y coordinate in pixels.
        depth_mm: Depth at the pixel in millimetres.
        fx: Camera focal length in pixels along x.
        fy: Camera focal length in pixels along y.
        cx: Camera principal point x in pixels.
        cy: Camera principal point y in pixels.

    Returns:
        Camera-frame ``[x_mm, y_mm, z_mm]`` in ROS optical convention.
    """
    if depth_mm <= 0.0:
        return [0.0, 0.0, 0.0]
    if fx == 0.0 or fy == 0.0:
        return [0.0, 0.0, depth_mm]
    x_mm = (float(pixel_x) - float(cx)) * float(depth_mm) / float(fx)
    y_mm = (float(pixel_y) - float(cy)) * float(depth_mm) / float(fy)
    return [x_mm, y_mm, float(depth_mm)]


def depth_frame_to_point_cloud_xyz(
    depth_frame: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 4,
    max_depth_mm: float = 0.0,
) -> np.ndarray:
    """Convert an aligned depth frame into an optical-frame XYZ point cloud.

    Args:
        depth_frame: Aligned depth image in millimetres.
        fx: Camera focal length in pixels along x.
        fy: Camera focal length in pixels along y.
        cx: Camera principal point x in pixels.
        cy: Camera principal point y in pixels.
        stride: Sample every Nth pixel in x and y.
        max_depth_mm: Optional upper depth cutoff. Non-positive disables it.

    Returns:
        Float32 array of shape ``(N, 3)`` containing ``[x_m, y_m, z_m]`` in the
        ROS optical frame.
    """
    if depth_frame.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if fx == 0.0 or fy == 0.0:
        return np.zeros((0, 3), dtype=np.float32)

    sample_stride = max(1, int(stride))
    sampled_depth = depth_frame[::sample_stride, ::sample_stride].astype(np.float32)
    if sampled_depth.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    valid_mask = sampled_depth > 0.0
    if max_depth_mm > 0.0:
        valid_mask &= sampled_depth <= float(max_depth_mm)
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32)

    ys = np.arange(0, depth_frame.shape[0], sample_stride, dtype=np.float32)
    xs = np.arange(0, depth_frame.shape[1], sample_stride, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    z_m = sampled_depth[valid_mask] / 1000.0
    x_m = ((grid_x[valid_mask] - float(cx)) * sampled_depth[valid_mask] / float(fx)) / 1000.0
    y_m = ((grid_y[valid_mask] - float(cy)) * sampled_depth[valid_mask] / float(fy)) / 1000.0
    return np.column_stack((x_m, y_m, z_m)).astype(np.float32, copy=False)


# ── TF helper ─────────────────────────────────────────────────────────────────


def quaternion_to_rpy(q) -> Tuple[float, float, float]:
    """Convert a geometry_msgs Quaternion to ``(roll, pitch, yaw)`` in radians.

    Accepts any object with ``.x``, ``.y``, ``.z``, ``.w`` attributes.

    Args:
        q: Quaternion-like object.

    Returns:
        Tuple of ``(roll, pitch, yaw)`` in radians.
    """
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = (
        math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    )

    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw
