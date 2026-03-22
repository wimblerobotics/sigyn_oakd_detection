# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Wimblerobotics
"""Unit tests for sigyn_oakd_detection utility functions.

These tests cover the pure Python helpers — axis-map parsing, application,
NMS, and quaternion conversion — without requiring a live OAK-D device or
a running ROS2 node.  They are designed to run quickly under ``colcon test``
(and directly with ``pytest test/`` from the source tree).

All functions under test live in ``sigyn_oakd_detection.detection_utils``,
which has no heavy dependencies (no ROS, no DepthAI, no cv2).
"""

import math
import os
import sys

import numpy as np
import pytest

# ── Locate the package source and import utilities ───────────────────────────
# Import pure-Python utilities from detection_utils — no ROS / DepthAI / cv2
# deps, so this works in any plain pytest environment.

_src_root = os.environ.get(
    "OAKD_DETECTION_SRC",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from sigyn_oakd_detection.detection_utils import (  # noqa: E402
    apply_axis_map,
    axis_map_to_string,
    best_axis_map,
    depth_frame_to_point_cloud_xyz,
    deproject_depth_pixel,
    estimate_object_depth,
    non_maximum_suppression,
    parse_axis_map,
    quaternion_to_rpy,
)


# ── parse_axis_map ─────────────────────────────────────────────────────────


class TestParseAxisMap:
    """Tests for parse_axis_map()."""

    def test_identity(self):
        """'x,y,z' should produce the identity map."""
        am = parse_axis_map("x,y,z")
        assert am == [(0, 1), (1, 1), (2, 1)]

    def test_negation(self):
        """-z,x,y should negate the first component and redirect axes."""
        am = parse_axis_map("-z,x,y")
        assert am == [(2, -1), (0, 1), (1, 1)]

    def test_all_negated(self):
        """-x,-y,-z should negate all axes."""
        am = parse_axis_map("-x,-y,-z")
        assert am == [(0, -1), (1, -1), (2, -1)]

    def test_whitespace_tolerance(self):
        """Extra spaces around tokens should be ignored."""
        am = parse_axis_map(" x , -y , z ")
        assert am == [(0, 1), (1, -1), (2, 1)]

    def test_wrong_number_of_tokens_falls_back(self):
        """Two tokens should fall back to identity."""
        am = parse_axis_map("x,y")
        assert am == [(0, 1), (1, 1), (2, 1)]

    def test_invalid_axis_name_falls_back(self):
        """Unknown axis letter should fall back to identity."""
        am = parse_axis_map("x,y,q")
        assert am == [(0, 1), (1, 1), (2, 1)]


# ── apply_axis_map ─────────────────────────────────────────────────────────


class TestApplyAxisMap:
    """Tests for apply_axis_map()."""

    def test_identity_leaves_vector_unchanged(self):
        am = parse_axis_map("x,y,z")
        result = apply_axis_map([1.0, 2.0, 3.0], am)
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_negate_z_redirect(self):
        """-z,x,y:  output[0]=-raw[2], output[1]=raw[0], output[2]=raw[1]."""
        am = parse_axis_map("-z,x,y")
        result = apply_axis_map([1.0, 2.0, 3.0], am)
        assert result == pytest.approx([-3.0, 1.0, 2.0])

    def test_all_negated(self):
        am = parse_axis_map("-x,-y,-z")
        result = apply_axis_map([1.0, -2.0, 3.0], am)
        assert result == pytest.approx([-1.0, 2.0, -3.0])


# ── axis_map_to_string ─────────────────────────────────────────────────────


class TestAxisMapToString:
    """Tests for axis_map_to_string()."""

    def test_round_trip_identity(self):
        s = "x,y,z"
        assert axis_map_to_string(parse_axis_map(s)) == s

    def test_round_trip_negated(self):
        s = "-z,x,y"
        assert axis_map_to_string(parse_axis_map(s)) == s

    def test_round_trip_all_negated(self):
        s = "-x,-y,-z"
        assert axis_map_to_string(parse_axis_map(s)) == s


# ── best_axis_map ──────────────────────────────────────────────────────────


class TestBestAxisMap:
    """Tests for best_axis_map()."""

    def test_finds_identity(self):
        """When raw == target the best map should be identity."""
        suggested, err = best_axis_map([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert err == pytest.approx(0.0, abs=1e-9)
        am = parse_axis_map(suggested)
        assert apply_axis_map([1.0, 2.0, 3.0], am) == pytest.approx([1.0, 2.0, 3.0])

    def test_finds_known_map(self):
        """'-z,x,y' should be found when raw and target are consistent."""
        raw = [1.0, 2.0, 3.0]
        expected_map = "-z,x,y"
        target = apply_axis_map(raw, parse_axis_map(expected_map))
        suggested, err = best_axis_map(raw, target)
        assert err == pytest.approx(0.0, abs=1e-9)
        # The recovered map should reproduce the target.
        recovered = apply_axis_map(raw, parse_axis_map(suggested))
        assert recovered == pytest.approx(target, abs=1e-9)

    def test_returns_none_on_empty(self):
        """Empty input should return (None, None) gracefully."""
        suggested, err = best_axis_map([], [])
        assert suggested is None
        assert err is None


# ── non_maximum_suppression ────────────────────────────────────────────────


class TestNonMaximumSuppression:
    """Tests for non_maximum_suppression()."""

    def test_empty_input(self):
        result = non_maximum_suppression(np.zeros((0, 4)), np.zeros(0), 0.5)
        assert result == []

    def test_single_box_kept(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        scores = np.array([0.9])
        kept = non_maximum_suppression(boxes, scores, 0.5)
        assert kept == [0]

    def test_perfect_overlap_suppresses_lower(self):
        """Two identical boxes: only the higher-score one survives."""
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
        scores = np.array([0.7, 0.9])
        kept = non_maximum_suppression(boxes, scores, 0.5)
        # scores[1]=0.9 wins; scores[0] is suppressed.
        assert kept == [1]

    def test_non_overlapping_boxes_all_kept(self):
        boxes = np.array(
            [[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float
        )
        scores = np.array([0.8, 0.9])
        kept = non_maximum_suppression(boxes, scores, 0.5)
        assert set(kept) == {0, 1}

    def test_partial_overlap_below_threshold(self):
        """Boxes with IoU < threshold should both survive."""
        boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)
        scores = np.array([0.8, 0.7])
        # IoU = 5*10 / (100 + 100 - 50) = 50/150 ≈ 0.33 < 0.5
        kept = non_maximum_suppression(boxes, scores, 0.5)
        assert set(kept) == {0, 1}

    def test_partial_overlap_above_threshold(self):
        """Boxes with IoU > threshold: lower-score box is suppressed."""
        boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], dtype=float)
        scores = np.array([0.8, 0.7])
        # IoU = 9*10 / (100 + 100 - 90) = 90/110 ≈ 0.818 > 0.5
        kept = non_maximum_suppression(boxes, scores, 0.5)
        assert kept == [0]


# ── depth helpers ─────────────────────────────────────────────────────────


class TestEstimateObjectDepth:
    """Tests for estimate_object_depth()."""

    def test_prefers_foreground_cluster_over_background(self):
        depth = np.full((10, 10), 1200, dtype=np.uint16)
        depth[3:7, 4:6] = 820
        sample = estimate_object_depth(depth, [2, 2, 7, 7])
        assert sample is not None
        sample_x, sample_y, sample_depth = sample
        assert sample_depth == pytest.approx(820.0, abs=1.0)
        assert sample_x == pytest.approx(4.5, abs=1.0)
        assert sample_y == pytest.approx(4.5, abs=1.0)

    def test_returns_none_without_valid_depth(self):
        depth = np.zeros((6, 6), dtype=np.uint16)
        assert estimate_object_depth(depth, [1, 1, 4, 4]) is None


class TestDeprojectDepthPixel:
    """Tests for deproject_depth_pixel()."""

    def test_principal_point_projects_to_z_axis(self):
        point = deproject_depth_pixel(320.0, 180.0, 1000.0, 500.0, 500.0, 320.0, 180.0)
        assert point == pytest.approx([0.0, 0.0, 1000.0])

    def test_pixel_offset_projects_with_fx_and_fy(self):
        point = deproject_depth_pixel(420.0, 280.0, 1000.0, 500.0, 400.0, 320.0, 180.0)
        assert point == pytest.approx([200.0, 250.0, 1000.0])


class TestDepthFrameToPointCloudXyz:
    """Tests for depth_frame_to_point_cloud_xyz()."""

    def test_returns_empty_for_empty_depth(self):
        cloud = depth_frame_to_point_cloud_xyz(
            np.zeros((0, 0), dtype=np.uint16),
            500.0,
            500.0,
            1.0,
            1.0,
        )
        assert cloud.shape == (0, 3)

    def test_projects_valid_points_in_meters(self):
        depth = np.array(
            [
                [0, 1000],
                [2000, 3000],
            ],
            dtype=np.uint16,
        )
        cloud = depth_frame_to_point_cloud_xyz(
            depth,
            1000.0,
            1000.0,
            0.0,
            0.0,
            stride=1,
        )
        assert cloud == pytest.approx(
            np.array(
                [
                    [0.001, 0.0, 1.0],
                    [0.0, 0.002, 2.0],
                    [0.003, 0.003, 3.0],
                ],
                dtype=np.float32,
            )
        )

    def test_applies_stride_and_max_depth(self):
        depth = np.array(
            [
                [1000, 1000, 1000, 1000],
                [1000, 1000, 1000, 1000],
                [4000, 4000, 4000, 4000],
                [4000, 4000, 4000, 4000],
            ],
            dtype=np.uint16,
        )
        cloud = depth_frame_to_point_cloud_xyz(
            depth,
            1000.0,
            1000.0,
            0.0,
            0.0,
            stride=2,
            max_depth_mm=1500.0,
        )
        assert cloud.shape == (2, 3)
        assert cloud[:, 2] == pytest.approx([1.0, 1.0])


# ── quaternion_to_rpy ──────────────────────────────────────────────────────


class TestQuaternionToRpy:
    """Tests for quaternion_to_rpy()."""

    class _FakeQuat:
        def __init__(self, x, y, z, w):
            self.x, self.y, self.z, self.w = x, y, z, w

    def test_identity_quaternion_gives_zero_rpy(self):
        q = self._FakeQuat(0, 0, 0, 1)
        roll, pitch, yaw = quaternion_to_rpy(q)
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)
        assert yaw == pytest.approx(0.0, abs=1e-9)

    def test_90_deg_yaw(self):
        """Quaternion for 90° yaw: (0, 0, sin45°, cos45°)."""
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        q = self._FakeQuat(0, 0, s, c)
        _, _, yaw = quaternion_to_rpy(q)
        assert yaw == pytest.approx(math.pi / 2, abs=1e-6)

    def test_90_deg_pitch(self):
        """Quaternion for 90° pitch: (0, sin45°, 0, cos45°)."""
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        q = self._FakeQuat(0, s, 0, c)
        _, pitch, _ = quaternion_to_rpy(q)
        assert pitch == pytest.approx(math.pi / 2, abs=1e-6)
