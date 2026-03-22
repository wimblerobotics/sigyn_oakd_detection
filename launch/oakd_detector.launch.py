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
    Axis remapping string, e.g. 'x,y,z' (default: 'x,y,z').
log_tf_debug : bool
    Log camera→base_link transform on each detection (default: true).
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
        default_value="x,y,z",
        description="Optional post-deprojection axis remapping (e.g. 'x,y,z'). Identity preserves the ROS optical frame.",
    )
    log_tf_debug_arg = DeclareLaunchArgument(
        "log_tf_debug",
        default_value="true",
        description="Log camera frame RPY on each detection",
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
                "point_cloud_publish_every": LaunchConfiguration("point_cloud_publish_every"),
                "point_cloud_stride": LaunchConfiguration("point_cloud_stride"),
                "point_cloud_max_depth_m": LaunchConfiguration("point_cloud_max_depth_m"),
            }
        ],
        remappings=[
            # Expose detections on the canonical Sigyn topics consumed by the BT.
            # Rich detection (class + confidence + 3D position) — primary BT subscription.
            ("/oakd_top/can_detections", "/oakd/can_detections"),
            # Annotated images for RViz.
            ("/oakd_top/annotated_image", "/oakd/annotated_image"),
            ("/oakd_top/annotated_image/compressed", "/oakd/annotated_image/compressed"),
        ],
    )

    return LaunchDescription(
        [
            blob_path_arg,
            camera_frame_arg,
            spatial_axis_map_arg,
            log_tf_debug_arg,
            point_cloud_publish_every_arg,
            point_cloud_stride_arg,
            point_cloud_max_depth_m_arg,
            detector_node,
        ]
    )
