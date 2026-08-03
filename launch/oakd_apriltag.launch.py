# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Wimblerobotics
"""Launch file for OAK-D Lite AprilTag detector.

This launch file starts the AprilTag detection node for a specific OAK-D Lite camera.
It produces RGB images, point clouds, and AprilTag detections with 3D poses.

Usage::

    ros2 launch sigyn_oakd_detection oakd_apriltag.launch.py

Launch arguments
----------------
camera_mx_id : str
    MxID of the OAK-D Lite camera to connect to (default: '1944301081303C1200').
camera_frame : str
    TF frame name for the camera optical center (default: 'oakd_apriltag_optical_frame').
rgb_resolution : str
    RGB camera resolution: '1080p', '4k', '12mp', '13mp' (default: '1080p').
depth_resolution : str
    Stereo depth resolution: '400p', '480p', '720p', '800p' (default: '400p').
fps : int
    Camera frame rate (default: 30).
point_cloud_publish_every : int
    Publish point cloud every N frames (default: 2).
point_cloud_stride : int
    Downsample point cloud by this factor (default: 4).
point_cloud_max_depth_m : float
    Maximum depth for point cloud (default: 5.0).
apriltag_family : str
    AprilTag family to detect (default: 'tag36h11').
    Options: 'tag16h5', 'tag25h9', 'tag36h11', 'tagCircle21h7', 'tagStandard41h12'
tag_size_m : float
    Physical size of AprilTags in meters (default: 0.166, ~6.5 inches).
use_rviz : bool
    Launch RViz2 with visualization config (default: false).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for OAK-D AprilTag detector."""
    pkg_share = get_package_share_directory("sigyn_oakd_detection")

    # Declare launch arguments
    camera_mx_id_arg = DeclareLaunchArgument(
        "camera_mx_id",
        default_value="1944301081303C1200",
        description="MxID of the OAK-D Lite camera",
    )
    
    camera_frame_arg = DeclareLaunchArgument(
        "camera_frame",
        default_value="oakd_apriltag_optical_frame",
        description="TF frame name for camera optical center",
    )
    
    rgb_resolution_arg = DeclareLaunchArgument(
        "rgb_resolution",
        default_value="1080p",
        description="RGB camera resolution (1080p, 4k, 12mp, 13mp)",
    )
    
    depth_resolution_arg = DeclareLaunchArgument(
        "depth_resolution",
        default_value="400p",
        description="Stereo depth resolution (400p, 480p, 720p, 800p)",
    )
    
    fps_arg = DeclareLaunchArgument(
        "fps",
        default_value="30",
        description="Camera frame rate",
    )
    
    point_cloud_publish_every_arg = DeclareLaunchArgument(
        "point_cloud_publish_every",
        default_value="2",
        description="Publish point cloud every N frames",
    )
    
    point_cloud_stride_arg = DeclareLaunchArgument(
        "point_cloud_stride",
        default_value="4",
        description="Downsample point cloud by this factor",
    )
    
    point_cloud_max_depth_m_arg = DeclareLaunchArgument(
        "point_cloud_max_depth_m",
        default_value="5.0",
        description="Maximum depth for point cloud generation",
    )
    
    apriltag_family_arg = DeclareLaunchArgument(
        "apriltag_family",
        default_value="tag36h11",
        description="AprilTag family to detect",
    )
    
    apriltag_detect_every_arg = DeclareLaunchArgument(
        "apriltag_detect_every",
        default_value="10",
        description="Run AprilTag detection every N frames",
    )
    
    apriltag_quad_decimate_arg = DeclareLaunchArgument(
        "apriltag_quad_decimate",
        default_value="3.0",
        description="Detection speed improvement (higher = faster, less accurate)",
    )
    
    tag_size_m_arg = DeclareLaunchArgument(
        "tag_size_m",
        default_value="0.166",
        description="Physical size of AprilTags in meters",
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="false",
        description="Launch RViz2 for visualization",
    )

    # AprilTag detector node
    apriltag_node = Node(
        package="sigyn_oakd_detection",
        executable="oakd_apriltag_node.py",
        name="oakd_apriltag_node",
        output="screen",
        parameters=[
            {
                "camera_mx_id": LaunchConfiguration("camera_mx_id"),
                "camera_frame": LaunchConfiguration("camera_frame"),
                "rgb_resolution": LaunchConfiguration("rgb_resolution"),
                "depth_resolution": LaunchConfiguration("depth_resolution"),
                "fps": LaunchConfiguration("fps"),
                "point_cloud_publish_every": LaunchConfiguration("point_cloud_publish_every"),
                "point_cloud_stride": LaunchConfiguration("point_cloud_stride"),
                "point_cloud_max_depth_m": LaunchConfiguration("point_cloud_max_depth_m"),
                "apriltag_family": LaunchConfiguration("apriltag_family"),
                "apriltag_detect_every": LaunchConfiguration("apriltag_detect_every"),
                "apriltag_quad_decimate": LaunchConfiguration("apriltag_quad_decimate"),
                "tag_size_m": LaunchConfiguration("tag_size_m"),
            }
        ],
    )

    # RViz2 (optional)
    rviz_config = os.path.join(pkg_share, "config", "apriltag_detection.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        output="screen",
    )

    return LaunchDescription([
        camera_mx_id_arg,
        camera_frame_arg,
        rgb_resolution_arg,
        depth_resolution_arg,
        fps_arg,
        point_cloud_publish_every_arg,
        point_cloud_stride_arg,
        point_cloud_max_depth_m_arg,
        apriltag_family_arg,
        apriltag_detect_every_arg,
        apriltag_quad_decimate_arg,
        tag_size_m_arg,
        use_rviz_arg,
        apriltag_node,
        rviz_node,
    ])
