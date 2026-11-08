#!/usr/bin/env python3
"""
Complete RWR Task 1 Launch File with Cube Detection and Policy, for the ORCA v2 hand

This launch file combines everything needed for the complete system:
1. Robot hand visualization (URDF + state publisher), showing the pose measured by the motors
2. Hand controller node
3. OakD camera
4. Camera position from the AprilTag, measured anew at every start (tag→camera), and the static TF tag→hand_root
5. Cube detection node
6. Policy inference node (policy_inference_cube_spin_v2.py)
7. RViz2 visualization

The tag must be visible to the camera at startup: until it is found, there is no cube pose in the hand frame and the
policy waits. After that an r26 policy runs with or without cube detections; an r24/r25 policy waits for the first one.

Command line usage:
    ros2 launch experiments cube_spin_v2.launch.py \
        tag_id:=9 tag_size:=0.075 \
        model_path:=/path/to/pose_estimation/finetuned_models/maskrcnn-finetuned_lightning_new_adamw.ckpt \
        policy:=/path/to/exported/policy.pt \
        enable_hand_controller:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import numpy as np
import xacro


def generate_launch_description():
    """Generate launch description with all arguments"""
    
    # ===== Camera Calibration (AprilTag) and Policy Arguments =====
    # the camera position is measured from this tag at every start, there is no calibration file to pass
    tag_id_arg = DeclareLaunchArgument('tag_id', default_value='9', description='ID of the AprilTag next to the hand')
    tag_size_arg = DeclareLaunchArgument(
        'tag_size',
        default_value='0.075',
        description="Side of the tag's black square in metres (without the white border)"
    )
    tag_family_arg = DeclareLaunchArgument('tag_family', default_value='tag36h11', description='AprilTag family')
    fist_pose_arg = DeclareLaunchArgument(
        'fist_pose',
        default_value='',
        description="Change joints of the fist the hand makes while the tag is measured, e.g. 'thumb_abd=10 index_mcp=95'"
    )

    policy_arg = DeclareLaunchArgument(
        'policy',
        default_value='',
        description='Path to the policy exported by faive_lab (exported/policy.pt)'
    )

    dry_run_arg = DeclareLaunchArgument(
        'dry_run',
        default_value='false',
        description='true: the policy computes and logs actions but does not command the hand'
    )

    goal_ang_vel_arg = DeclareLaunchArgument(
        'goal_ang_vel',
        default_value='1.0',
        description='commanded spin speed about z in rad/s, the sign sets the direction (r25 and r26 are trained on 1.2-2.0)'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to the cube detection checkpoint (.ckpt from the pose_estimation repo)'
    )


    # ===== Tag to Hand Transform Arguments =====
    # Pose of the tag seen from the mount (mount_T_tag): x/y/z along the mount's axes, roll/pitch/yaw of the tag relative to them
    # NOTE: these defaults are for the ORCA v2 stand: hand on the post, palm up, fingers tilted 30 deg down towards the tag,
    # which lies flat on the base extrusion (tag 27 cm from the mount towards the fingers, 20 cm below it). Estimated from a
    # photo of the stand (about +-1.5 cm); measure them again if the stand changes.
    hand_to_tag_x_arg = DeclareLaunchArgument('hand_to_tag_x', default_value='0.035')
    hand_to_tag_y_arg = DeclareLaunchArgument('hand_to_tag_y', default_value='-0.04')
    hand_to_tag_z_arg = DeclareLaunchArgument('hand_to_tag_z', default_value='0.34')
    hand_to_tag_roll_arg = DeclareLaunchArgument('hand_to_tag_roll', default_value='60')
    hand_to_tag_pitch_arg = DeclareLaunchArgument('hand_to_tag_pitch', default_value='0')
    hand_to_tag_yaw_arg = DeclareLaunchArgument('hand_to_tag_yaw', default_value='0')
    
    # ===== Camera Arguments =====
    camera_name_arg = DeclareLaunchArgument(
        'camera_name',
        default_value='oakd_bottom_view',
        description='Camera name for TF frames'
    )
    
    camera_config_arg = DeclareLaunchArgument(
        'camera_config',
        default_value='oakd_cube_task',
        description='Camera configuration file name (without .yaml)'
    )
    
    # ===== Enable/Disable Components =====
    enable_hand_controller_arg = DeclareLaunchArgument(
        'enable_hand_controller',
        default_value='false',
        description='Enable hand controller node (connects to real robot)'
    )
    
    # Launch the nodes using OpaqueFunction
    nodes = OpaqueFunction(function=launch_setup)
    
    return LaunchDescription([
        tag_id_arg,
        tag_size_arg,
        tag_family_arg,
        fist_pose_arg,
        policy_arg,
        dry_run_arg,
        goal_ang_vel_arg,
        model_path_arg,
        hand_to_tag_x_arg,
        hand_to_tag_y_arg,
        hand_to_tag_z_arg,
        hand_to_tag_roll_arg,
        hand_to_tag_pitch_arg,
        hand_to_tag_yaw_arg,
        camera_name_arg,
        camera_config_arg,
        enable_hand_controller_arg,
        nodes,
    ])


def launch_setup(context, *args, **kwargs):
    """Setup function to create all nodes with parameters"""
    from scipy.spatial.transform import Rotation as R
    
    # Get launch arguments
    tag_id = int(LaunchConfiguration('tag_id').perform(context))
    tag_size = float(LaunchConfiguration('tag_size').perform(context))
    tag_family = LaunchConfiguration('tag_family').perform(context)
    fist_pose = LaunchConfiguration('fist_pose').perform(context).split()
    policy_file = LaunchConfiguration('policy').perform(context)
    dry_run = LaunchConfiguration('dry_run').perform(context).lower() == 'true'
    goal_ang_vel = float(LaunchConfiguration('goal_ang_vel').perform(context))
    model_path = LaunchConfiguration('model_path').perform(context)
    hand_to_tag_x = float(LaunchConfiguration('hand_to_tag_x').perform(context))
    hand_to_tag_y = float(LaunchConfiguration('hand_to_tag_y').perform(context))
    hand_to_tag_z = float(LaunchConfiguration('hand_to_tag_z').perform(context))
    hand_to_tag_roll = float(LaunchConfiguration('hand_to_tag_roll').perform(context))
    hand_to_tag_pitch = float(LaunchConfiguration('hand_to_tag_pitch').perform(context))
    hand_to_tag_yaw = float(LaunchConfiguration('hand_to_tag_yaw').perform(context))
    camera_name = LaunchConfiguration('camera_name').perform(context)
    camera_config = LaunchConfiguration('camera_config').perform(context)
    enable_hand_controller = LaunchConfiguration('enable_hand_controller').perform(context).lower() == 'true'
    
    # ===== 1. Robot Hand Visualization =====
    orca_xacro = os.path.join(
        get_package_share_directory('viz'),
        "models",
        "orcahand_v2",
        "orcahand_v2.urdf.xacro"
    )
    orca_robot_desc = xacro.process_file(orca_xacro, mappings={'hand_chirality': 'right'}).toxml()

    orca_scheme_filepath = os.path.join(
        get_package_share_directory("viz"),
        f"models/orcahand_v2/scheme_orcahand_v2.yaml"
    )
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='right/hand',
        name='hand_state_publisher',
        output='screen',
        parameters=[{'robot_description': orca_robot_desc}],
        remappings=[('/joint_states', '/right/hand/joint_states')]
    )
    
    # ===== 2. Hand Joint Visualizer =====
    # shows the joint positions measured by the motors
    visualize_joints = Node(
        package="viz",
        executable="visualize_joints.py",
        name="visualize_joints",
        namespace='right',
        output='log',
        parameters=[{
            "scheme_path": orca_scheme_filepath,
            "viz/hand_chirality": "right",
            "viz/joint_name_contains_chirality": False
        }],
        remappings=[('/right/joint_states', '/right/hand/joint_states'),
                    ('hand/policy_output', '/right/hand/joint_position_from_motor')]
    )
    
    # ===== 3. OakD Camera Node =====
    oakd_node = Node(
        package="ingress",
        executable="oakd_node.py", 
        name="oakd_node",
        output="log",
        parameters=[{
            "config_file": os.path.join(
                get_package_share_directory("experiments"), 
                "cfgs", "cams", f"{camera_config}.yaml"
            ),
            "use_depth": False
        }],
    )
    
    # ===== 4. Static TF Publishers =====
    # TF tree: camera_frame → tag_frame → hand_root
    # This allows: cube (detected in camera) → tag → hand
    
    # 4a. Tag → Hand (tag_T_hand transform)
    # hand_to_tag_x/y/z are measured from the mount (orcahand_v2.urdf.xacro), which sits at (-0.01, 0, -0.058) in hand_root
    hand_to_tag_t = np.array([hand_to_tag_x, hand_to_tag_y, hand_to_tag_z]) + np.array([-0.01, 0.0, -0.058])
    hand_to_tag_quat = R.from_euler('XYZ', [hand_to_tag_roll, hand_to_tag_pitch, hand_to_tag_yaw], degrees=True).as_quat()
    rot_inv = R.from_quat(hand_to_tag_quat).inv()
    tag_to_hand_t = -rot_inv.apply(hand_to_tag_t)
    tag_to_hand_quat = rot_inv.as_quat()


    static_tf_tag_to_hand = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_tag_to_hand',
        arguments=[
            str(tag_to_hand_t[0]), str(tag_to_hand_t[1]), str(tag_to_hand_t[2]),
            str(tag_to_hand_quat[0]), str(tag_to_hand_quat[1]), str(tag_to_hand_quat[2]), str(tag_to_hand_quat[3]),
            'tag_frame', 'hand_root'
        ]
    )
    
    # 4b. Tag → Camera: measured from the AprilTag at every start (like the hand calibration), never loaded from a file,
    # so a camera that was moved or bumped is never used with an old position. The node averages 30 detections
    # (about 3 s, collected again if they spread more than 1 cm), publishes tag_frame → <camera_name>_frame on
    # /tf_static and keeps running. Until then hand_root → cube does not exist and a policy using the cube waits.
    # The open hand hides the tag: once the hand controller runs, the node closes the hand (orca_core's power_grasp,
    # wrist unchanged), measures, opens it back to where it started and only then publishes; the policy waits for that
    # (wait_for_frame below). Without a hand controller it measures the tag if it is visible.
    # The last result is also written to ~/camera_extrinsics.npy, only for reference.
    camera_calibration_node = Node(
        package='experiments',
        executable='apriltag_calibration.py',
        name='apriltag_calibration',
        output='screen',
        arguments=[
            '--camera-topic', f'/{camera_name}/color',
            '--camera-name', camera_name,
            '--tag-family', tag_family,
            '--tag-id', str(tag_id),
            '--tag-size', str(tag_size),
            '--output', os.path.expanduser('~/camera_extrinsics.npy'),
            '--publish-tf',
        ] + (['--fist-pose'] + fist_pose if fist_pose else []),
    )

    # ===== 5. Cube Detection Node =====
    cube_detection_node = None
    if model_path:
        cube_detection_node = Node(
            package='experiments',
            executable='cube_detection_node.py',
            name='cube_detection_node',
            output='screen',
            parameters=[{
                'camera_name': camera_name,
                'model_path': os.path.expanduser(model_path),
                'show_windows': False
            }]
        )
    else:
        print("WARNING: model_path not set, skipping cube detection.")

    # ===== 6. Policy Inference Node (optional) =====
    policy_inference_node = None
    if policy_file:
        policy_inference_node = Node(
            package='experiments',
            executable='policy_inference_cube_spin_v2.py',
            name='policy_inference_node',
            output='screen',
            parameters=[{
                'policy_ckpt_path': os.path.expanduser(policy_file),
                'dry_run': dry_run,
                'goal_ang_vel': goal_ang_vel,
                # do not command the hand before the camera calibration has closed and opened it again
                'wait_for_frame': f'{camera_name}_frame',
                'hand_config_path': os.path.join(
                    get_package_share_directory('hand_control'),
                    'orca', 'configs', 'orcahand_v2', 'right', 'config.yaml'
                ),
            }],
        )

    # ===== 7. Hand Controller Node (optional) =====
    hand_controller_node = None
    if enable_hand_controller:
        hand_controller_node = Node(
            package="hand_control",
            executable="hand_control_node.py",
            name="hand_controller_node",
            output="screen",
            namespace="right",
            parameters=[{
                "hand_controller/hand_name": "orca_v2",
                "hand_controller/hand_chirality": "right",
                "hand_controller/hand_cfg_name": "orcahand_v2",
                "hand_controller/calibrate": False
            }],
        )
    
    # ===== 8. RViz2 =====
    rviz_config = os.path.join(
        get_package_share_directory('experiments'), 
        'cfgs', 'rviz', 'cubespin.rviz'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen'
    )
    
    # ===== Build Node List =====
    nodes_to_launch = [
        # Core visualization and robot
        robot_state_publisher,
        visualize_joints,
        oakd_node,
        
        # TF: camera position from the AprilTag (measured at every start), tag → hand
        camera_calibration_node,
        static_tf_tag_to_hand,

        # Detection and policy
        cube_detection_node,
        
        # Visualization
        rviz_node,
    ]
    
    # Add optional policy node
    if policy_inference_node is not None:
        nodes_to_launch.append(policy_inference_node)
    
    # Add optional hand controller
    if hand_controller_node is not None:
        nodes_to_launch.insert(3, hand_controller_node)  # After oakd_node

    return [node for node in nodes_to_launch if node is not None]
