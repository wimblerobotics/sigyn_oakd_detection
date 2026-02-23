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
