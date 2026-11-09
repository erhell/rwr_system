#!/usr/bin/env python3
"""
Convert a CSV log of policy_inference_cube_spin_v2.py (csv_log_path) into an MCAP bag, e.g. for Foxglove.

Topics (joint names and order as in the policy, i.e. the USD order; positions in radians):
    /policy/joint_pos      sensor_msgs/JointState  measured position (from the normalised observation) and velocity
    /policy/joint_cmd      sensor_msgs/JointState  joint position command sent to the hand
    /policy/last_action    sensor_msgs/JointState  the policy's previous raw output ([-1, 1] = joint limits)
    /policy/goal_ang_vel   std_msgs/Float64        commanded spin speed about z (rad/s)
    /policy/cube_pose      geometry_msgs/PoseStamped  cube pose in the policy's frame, only while the cube was tracked

The CSV holds the time since the first policy step; the bag starts at the CSV file's last-modified time minus the run's
length (about when the run started), or at --start (Unix seconds, e.g. the first time stamp in the run's .log).

Usage (ROS 2 environment sourced):
    python policy_csv_to_mcap.py logs/policy_runs/r21ft_20261108_171924.csv
writes logs/policy_runs/r21ft_20261108_171924_csv/r21ft_20261108_171924_csv_0.mcap; open that file in Foxglove.
"""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import rosbag2_py
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import serialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

# same limits and observation layout as the node that wrote the CSV
from policy_inference_cube_spin_v2 import JOINT_LIMITS_USD_DEG

N_JOINTS = len(JOINT_LIMITS_USD_DEG)


def stamp(msg_header, t_ns: int):
    msg_header.stamp.sec, msg_header.stamp.nanosec = divmod(t_ns, 1_000_000_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="bag folder to create (default: <csv without .csv>_csv)")
    parser.add_argument("--start", type=float, help="Unix time of the first row (default: from the file's mtime)")
    parser.add_argument("--vel-scale", type=float, default=0.2, help="velocity_obs_scale the node ran with")
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader if row], dtype=np.float64)

    obs_cols = [i for i, name in enumerate(header) if name.startswith("obs_")]
    cmd_cols = [i for i, name in enumerate(header) if name.startswith("cmd_")]
    cube_cols = [header.index(n) for n in ("cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz")]
    joint_names = [header[i][len("cmd_"):] for i in cmd_cols]
    assert joint_names == list(JOINT_LIMITS_USD_DEG), f"CSV joints {joint_names} differ from the node's"

    # observation layout: joint_pos, joint_vel, [cube frames], goal_command (6), last_action, last_joint_cmd
    n_obs = len(obs_cols)
    obs = rows[:, obs_cols]
    limits = np.deg2rad(np.array(list(JOINT_LIMITS_USD_DEG.values())))
    low, high = limits[:, 0], limits[:, 1]
    joint_pos = (low + high) / 2 + obs[:, :N_JOINTS] * (high - low) / 2      # inverse of the node's normalize()
    joint_vel = obs[:, N_JOINTS:2 * N_JOINTS] / args.vel_scale
    goal_ang_vel = obs[:, n_obs - 2 * N_JOINTS - 1]
    last_action = obs[:, n_obs - 2 * N_JOINTS:n_obs - N_JOINTS]
    joint_cmd = rows[:, cmd_cols]
    cube = rows[:, cube_cols]

    t_rel = rows[:, header.index("time")]
    start = args.start if args.start is not None else os.path.getmtime(args.csv_path) - t_rel[-1]
    t_ns = ((start + t_rel) * 1e9).astype(np.int64)

    output = args.output or args.csv_path.with_name(args.csv_path.stem + "_csv")
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    topics = {
        "/policy/joint_pos": "sensor_msgs/msg/JointState",
        "/policy/joint_cmd": "sensor_msgs/msg/JointState",
        "/policy/last_action": "sensor_msgs/msg/JointState",
        "/policy/goal_ang_vel": "std_msgs/msg/Float64",
        "/policy/cube_pose": "geometry_msgs/msg/PoseStamped",
    }
    for i, (name, msg_type) in enumerate(topics.items()):
        writer.create_topic(rosbag2_py.TopicMetadata(i, name, msg_type, "cdr"))

    def joint_state(t, position, velocity=None):
        msg = JointState()
        stamp(msg.header, t)
        msg.name = joint_names
        msg.position = position.tolist()
        if velocity is not None:
            msg.velocity = velocity.tolist()
        return msg

    n_cube = 0
    for k, t in enumerate(t_ns.tolist()):
        writer.write("/policy/joint_pos", serialize_message(joint_state(t, joint_pos[k], joint_vel[k])), t)
        writer.write("/policy/joint_cmd", serialize_message(joint_state(t, joint_cmd[k])), t)
        writer.write("/policy/last_action", serialize_message(joint_state(t, last_action[k])), t)
        writer.write("/policy/goal_ang_vel", serialize_message(Float64(data=float(goal_ang_vel[k]))), t)
        if not any(math.isnan(v) for v in cube[k]):
            pose = PoseStamped()
            stamp(pose.header, t)
            pose.header.frame_id = "policy_frame"
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = cube[k, :3].tolist()
            pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z = (
                cube[k, 3:].tolist())
            writer.write("/policy/cube_pose", serialize_message(pose), t)
            n_cube += 1
    del writer  # closes the bag

    print(f"{len(rows)} policy steps ({t_rel[-1]:.1f} s, {n_cube} with the cube) -> {output}/")


if __name__ == "__main__":
    main()
