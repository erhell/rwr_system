#!/usr/bin/env python3
"""
Camera and cube detection test: the OAK-D camera, cube_detection_node and cube_view.py, which shows the camera image
with the estimated cube in a window. No hand, AprilTag calibration, policy or RViz.

    ros2 launch experiments cube_view.launch.py \
        model_path:=/path/to/pose_estimation/finetuned_models/maskrcnn-finetuned_lightning_new_adamw.ckpt

The camera node waits up to 60 s for the camera after loading its firmware (DEPTHAI_BOOTUP_TIMEOUT): in a virtual
machine the camera reconnects as a new USB device at that moment, which the VM has to pass through as well.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value='',
                              description='Path to the cube detection checkpoint (.ckpt from the pose_estimation repo)'),
        DeclareLaunchArgument('camera_name', default_value='oakd_bottom_view', description='Camera name for TF frames'),
        DeclareLaunchArgument('camera_config', default_value='oakd_cube_task',
                              description='Camera configuration file name (without .yaml)'),
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context, *args, **kwargs):
    model_path = LaunchConfiguration('model_path').perform(context)
    camera_name = LaunchConfiguration('camera_name').perform(context)
    camera_config = LaunchConfiguration('camera_config').perform(context)
    if not model_path:
        raise RuntimeError("Set model_path:= to the cube detection checkpoint "
                           "(rl_inference/pose_estimation/finetuned_models/maskrcnn-finetuned_lightning_new_adamw.ckpt)")

    oakd_node = Node(
        package="ingress",
        executable="oakd_node.py",
        name="oakd_node",
        output="screen",
        parameters=[{
            "config_file": os.path.join(get_package_share_directory("experiments"), "cfgs", "cams",
                                        f"{camera_config}.yaml"),
            "use_depth": False,
        }],
        additional_env={"DEPTHAI_BOOTUP_TIMEOUT": "60000"},
    )
    cube_detection_node = Node(
        package='experiments',
        executable='cube_detection_node.py',
        name='cube_detection_node',
        output='screen',
        parameters=[{
            'camera_name': camera_name,
            'model_path': os.path.expanduser(model_path),
            'show_windows': False,
        }],
    )
    cube_view_node = Node(
        package='experiments',
        executable='cube_view.py',
        name='cube_view',
        output='screen',
        parameters=[{'camera_name': camera_name}],
    )
    return [oakd_node, cube_detection_node, cube_view_node]
