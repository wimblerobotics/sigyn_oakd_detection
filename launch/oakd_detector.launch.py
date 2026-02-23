# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Wimblerobotics
"""Production launch file for the OAK-D can detector.

Intended to be included by the Sigyn bringup launch (sigyn.launch.py).
Does NOT start RViz or a robot_state_publisher — those are assumed to be
already running in the parent context.

Usage (standalone quick-test without the full Sigyn stack)::

    ros2 launch sigyn_oakd_detection oakd_detector.launch.py

Launch arguments
----------------
blob_path : str
    Full path to the compiled DepthAI blob.  Defaults to the blob shipped
    with this package.
camera_frame : str
    TF frame of the RGB optical centre (default: oak_rgb_camera_optical_frame).
spatial_axis_map : str
    Axis remapping string, e.g. '-z,x,y' (default: '-z,x,y').
log_tf_debug : bool
    Log camera→base_link transform on each detection (default: true).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return the LaunchDescription for the production OAK-D detector."""
    pkg_share = get_package_share_directory("sigyn_oakd_detection")
    default_blob = os.path.join(pkg_share, "models", "can_detector.blob")

    blob_path_arg = DeclareLaunchArgument(
        "blob_path",
        default_value=default_blob,
        description="Full path to the DepthAI compiled blob file",
    )
    camera_frame_arg = DeclareLaunchArgument(
        "camera_frame",
        default_value="oak_rgb_camera_optical_frame",
        description="TF frame of the RGB optical centre",
    )
    spatial_axis_map_arg = DeclareLaunchArgument(
        "spatial_axis_map",
        default_value="-z,x,y",
        description="DepthAI spatial axis remapping (e.g. '-z,x,y')",
    )
    log_tf_debug_arg = DeclareLaunchArgument(
        "log_tf_debug",
        default_value="true",
        description="Log camera frame RPY on each detection",
    )

    detector_node = Node(
        package="sigyn_oakd_detection",
        executable="oakd_detector_node.py",
        name="oakd_detector",
        output="screen",
        parameters=[
            {
                "blob_path": LaunchConfiguration("blob_path"),
                "camera_frame": LaunchConfiguration("camera_frame"),
                "spatial_axis_map": LaunchConfiguration("spatial_axis_map"),
                "log_tf_debug": LaunchConfiguration("log_tf_debug"),
            }
        ],
        remappings=[
            # Expose detections on the canonical Sigyn topic consumed by BT.
            ("/oakd_top/can_point_camera", "/oakd/can_detection"),
            ("/oakd_top/annotated_image", "/oakd/annotated_image"),
        ],
    )

    return LaunchDescription(
        [
            blob_path_arg,
            camera_frame_arg,
            spatial_axis_map_arg,
            log_tf_debug_arg,
            detector_node,
        ]
    )
