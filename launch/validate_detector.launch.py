# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Wimblerobotics
r"""Standalone validation launch file for the OAK-D can detector.

Starts the detector node, a robot_state_publisher (for TF), and RViz with the
bundled can-detection visualisation config.  Useful for verifying detections
and axis-map alignment without running the full Sigyn stack.

Usage::

    # Basic validation
    ros2 launch sigyn_oakd_detection validate_detector.launch.py

    # Auto-suggest the correct spatial_axis_map
    ros2 launch sigyn_oakd_detection validate_detector.launch.py \\
        expected_target_base:="[0.65, 0.0, 0.6]" suggest_axis_map:=true

Launch arguments
----------------
blob_path : str
    Full path to the compiled DepthAI blob (default: bundled can_detector.blob).
spatial_axis_map : str
    Axis remapping string (default: 'x,y,z').
expected_target_base : str
    JSON-list expected can position in base_link for axis-map suggestion.
suggest_axis_map : bool
    Enable axis-map suggestion logging (default: true).
point_cloud_publish_every : int
    Publish the point cloud every N depth frames (default: 1).
point_cloud_stride : int
    Use every Nth depth pixel when generating the point cloud (default: 4).
point_cloud_max_depth_m : float
    Ignore points beyond this depth when generating the cloud (default: 5.0).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return a LaunchDescription for standalone detector validation."""
    pkg_share = get_package_share_directory("sigyn_oakd_detection")
    default_blob = os.path.join(pkg_share, "models", "can_detector.blob")
    rviz_config = os.path.join(pkg_share, "config", "can_detection.rviz")

    # ── Launch arguments ────────────────────────────────────────────────────
    blob_path_arg = DeclareLaunchArgument(
        "blob_path",
        default_value=default_blob,
        description="Full path to the DepthAI blob file",
    )
    spatial_axis_map_arg = DeclareLaunchArgument(
        "spatial_axis_map",
        default_value="x,y,z",
        description="DepthAI spatial axis remapping",
    )
    expected_target_base_arg = DeclareLaunchArgument(
        "expected_target_base",
        default_value="[0.65, 0.0, 0.6]",
        description="Expected can position in base_link [x,y,z]",
    )
    suggest_axis_map_arg = DeclareLaunchArgument(
        "suggest_axis_map",
        default_value="true",
        description="Log suggested spatial_axis_map on each detection",
    )
    point_cloud_publish_every_arg = DeclareLaunchArgument(
        "point_cloud_publish_every",
        default_value="1",
        description="Publish the point cloud every N depth frames",
    )
    point_cloud_stride_arg = DeclareLaunchArgument(
        "point_cloud_stride",
        default_value="4",
        description="Use every Nth depth pixel when generating the point cloud",
    )
    point_cloud_max_depth_m_arg = DeclareLaunchArgument(
        "point_cloud_max_depth_m",
        default_value="5.0",
        description="Ignore points deeper than this distance when generating the point cloud",
    )

    # ── Robot description (for TF / RViz) ──────────────────────────────────
    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("sigyn_description"),
                "launch",
                "description.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": "false",
            "publish_joints": "false",
            "do_rviz": "false",
        }.items(),
    )

    # ── Detector node ────────────────────────────────────────────────────────
    detector_node = Node(
        package="sigyn_oakd_detection",
        executable="oakd_detector_node.py",
        name="oakd_detector",
        output="screen",
        parameters=[
            {
                "blob_path": LaunchConfiguration("blob_path"),
                "camera_frame": "oak_rgb_camera_optical_frame",
                "spatial_axis_map": LaunchConfiguration("spatial_axis_map"),
                "log_tf_debug": True,
                "expected_target_base": LaunchConfiguration(
                    "expected_target_base"
                ),
                "suggest_axis_map": LaunchConfiguration("suggest_axis_map"),
                "point_cloud_publish_every": LaunchConfiguration("point_cloud_publish_every"),
                "point_cloud_stride": LaunchConfiguration("point_cloud_stride"),
                "point_cloud_max_depth_m": LaunchConfiguration("point_cloud_max_depth_m"),
            }
        ],
    )

    # ── RViz ────────────────────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription(
        [
            blob_path_arg,
            spatial_axis_map_arg,
            expected_target_base_arg,
            suggest_axis_map_arg,
            point_cloud_publish_every_arg,
            point_cloud_stride_arg,
            point_cloud_max_depth_m_arg,
            description_launch,
            detector_node,
            rviz_node,
        ]
    )
