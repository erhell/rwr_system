#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Inference Node for the ORCA v2 hand - runs an rsl_rl policy trained in faive_lab
(task orca-v2-inhand-repose-v0) on the real hand, with or without the cube pose from the camera.

The policy is the TorchScript file that faive_lab's play.py exports (logs/rsl_rl/orca_v2_repose/<run>/exported/policy.pt);
it includes the observation normalisation, so it takes the raw observation vector.

Every 1/policy_hz seconds:
1. Read the hand's joint positions and velocities (radians, in the order of the hand's config.yaml joint_ids)
2. Build the policy observation exactly as in the simulation (USD joint order, 74 values, or 81 with the cube pose):
       joint_pos       joint positions normalised to [-1, 1] over the joint limits
       joint_vel       joint velocities * 0.2
       object_pos      cube position in the simulation's frame (only for a policy trained with the cube pose)
       object_quat     cube orientation (w, x, y, z) in the simulation's frame (idem)
       goal_command    [0, 0, -0.05, 0, 0, angular velocity about z]
       last_action     the policy's previous raw output
       last_joint_cmd  the previous joint position command (radians)
3. Run the policy, map its output onto the joint limits and smooth it with the same moving average as the simulation
   (target = alpha * new + (1 - alpha) * previous)
4. Publish the joint position command on /right/hand/policy_output, which hand_control_node executes (unless dry_run)

Subtask 2 (exteroception): every step the node also reads the cube pose from TF (hand_root -> cube, published by
cube_spin_v2.launch.py and cube_detection_node) and converts it to object_pos / object_quat as in the simulation. The
policy's input size decides whether and how they are part of the observation:
- 74 inputs: joint state only, the cube pose is only logged.
- 74 + 8 * N (faive_lab r26: N = 3, 98 inputs): the last N cube detections, oldest first, each as
  [object_pos, object_quat, age / cube_max_age_s], where age = now - the image's time stamp. A detection
  cube_max_age_s or older, or a slot without one yet, is [0] * 7 + [1]: "no camera". The policy was trained with
  detections missing, delayed, hidden for up to 2 s and with no camera at all, so it never waits for the cube: it
  starts on the joint state alone and uses the cube pose whenever the detector delivers one.
- 74 + 7 * N (r24: N = 1, r25: N = 3): the last N poses [object_pos, object_quat], one per policy step. These policies
  were trained with a pose every step, so they wait for the first detection (which then fills all N frames) and keep
  the last detected pose while the cube is not tracked.

The joint limits, observation order and scales below must match the training task (orca_v2_repose_env_cfg.py).

Usage (hand driver running, e.g. `ros2 launch hand_control orcahand_v2_controller_launch.yaml`):
    python policy_inference_cube_spin_v2.py --ros-args -p policy_ckpt_path:=/path/to/exported/policy.pt
"""

import csv
from camera_feedback import CubeDetectionHistory, deploy_meta_action_alpha, load_camera_contract
from pathlib import Path

import numpy as np
import rclpy
import rclpy.executors
import tf2_ros
import torch
import yaml
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float32MultiArray

from faive_system.src.common.ros_utils import numpy_to_float32_multiarray, float32_multiarray_to_numpy

DEFAULT_HAND_CONFIG = (
    Path(__file__).resolve().parents[2] / "src" / "hand_control" / "orca" / "configs" / "orcahand_v2" / "right" / "config.yaml"
)

# Joint order of the v2 USD model in Isaac Lab (= order of the policy's observations and actions), with the joint limits
# the task normalises over (soft limits of the v2 USD, degrees).
JOINT_LIMITS_USD_DEG = {
    "wrist": (-65.0, 35.0),
    "index_abd": (-25.0, 30.0),
    "middle_abd": (-27.0, 27.0),
    "pinky_abd": (-30.0, 30.0),
    "ring_abd": (-27.0, 27.0),
    "thumb_cmc": (-45.0, 33.0),
    "index_mcp": (-25.0, 100.0),
    "middle_mcp": (-25.0, 100.0),
    "pinky_mcp": (-25.0, 100.0),
    "ring_mcp": (-25.0, 100.0),
    "thumb_abd": (-18.0, 55.0),
    "index_pip": (-15.0, 107.0),
    "middle_pip": (-15.0, 107.0),
    "pinky_pip": (-15.0, 107.0),
    "ring_pip": (-15.0, 107.0),
    "thumb_mcp": (-25.0, 100.0),
    "thumb_dip": (-15.0, 107.0),
}
OBS_DIM = 17 + 17 + 6 + 17 + 17     # proprioception only
CUBE_POSE_DIM = 3 + 4               # object_pos and object_quat from the camera, per frame (r24, r25)
CUBE_DETECTION_DIM = 3 + 4 + 1      # object_pos, object_quat and the detection's age, per frame (r26)
NO_DETECTION = np.array([0.0] * CUBE_POSE_DIM + [1.0], dtype=np.float32)
MAX_CUBE_FRAMES = 5

# The simulation observes the cube in its world frame (mdp.root_pos_w), so the cube pose is converted from hand_root
# to that frame: robot init_state pos (0.0004, 0.1512, -0.0187), rot (w, x, y, z) (-0.2607, 0.9654, 0, 0) of the root
# link right_forearm, and hand_root -> forearm 90 deg about x (orcahand_v2.urdf.xacro): 120.2 deg about x.
# If your simulator observes the cube relative to the hand instead, note that the robot's root link is forearm, not
# hand_root: use forearm_T_hand_root (90 deg about -x) @ hand_T_cube.
OBS_T_HAND = np.array([
    [1.0,  0.0,     0.0,     0.0004],
    [0.0, -0.5034, -0.8641,  0.1512],
    [0.0,  0.8641, -0.5034, -0.0187],
    [0.0,  0.0,     0.0,     1.0],
])


class PolicyInferenceNodeV2(Node):
    def __init__(self):
        super().__init__("policy_inference_node_v2")

        # ========== ROS2 Parameters ==========
        self.declare_parameter("policy_ckpt_path", "")
        self.declare_parameter("hand_config_path", str(DEFAULT_HAND_CONFIG))
        self.declare_parameter("policy_hz", 20.0)          # the task runs the policy at 20 Hz (decimation 6, dt 1/120)
        # moving average of the task's action term; < 0 (default): the action_alpha in the deploy_meta.json next to the
        # policy (0.2 without one). r29-r32 trained with 0.3, r15-r28 with 0.2
        self.declare_parameter("action_alpha", -1.0)
        self.declare_parameter("velocity_obs_scale", 0.2)  # joint_vel observation scale in the task
        self.declare_parameter("goal_ang_vel", 1.0)        # rad/s about z; trained on |w| in [0.9, 1.5] (either sign = direction)
        self.declare_parameter("goal_flip_period_s", 0.0)  # > 0: flip the spin direction every this many seconds
        self.declare_parameter("dry_run", False)           # true: compute and log actions, but do not command the hand
        self.declare_parameter("log_period_s", 1.0)        # print the observation summary this often (0: never)
        self.declare_parameter("csv_log_path", "")         # non-empty: log observations and commands to this CSV
        self.declare_parameter("command_topic", "/right/hand/policy_output")  # what hand_control_node subscribes to

        # Frame IDs
        self.declare_parameter("hand_frame_id", "hand_root")
        self.declare_parameter("cube_frame_id", "cube")
        self.declare_parameter("cube_pose_timeout_s", 0.5)  # an older cube pose counts as not tracked
        # r26 policies: a detection this old reads as "no camera"; must match max_age_s of the training task
        self.declare_parameter("cube_max_age_s", 1.0)
        # non-empty: do not command the hand before hand_frame -> this frame exists in TF. cube_spin_v2.launch.py sets
        # the camera frame, which apriltag_calibration.py publishes only after it has closed the hand, measured the
        # tag and opened the hand again; until then the policy would fight that motion
        self.declare_parameter("wait_for_frame", "")

        self.policy_ckpt_path = self.get_parameter("policy_ckpt_path").value
        self.policy_hz = float(self.get_parameter("policy_hz").value)
        self.action_alpha = float(self.get_parameter("action_alpha").value)
        if self.action_alpha < 0:
            self.action_alpha = deploy_meta_action_alpha(self.policy_ckpt_path)
        self.velocity_obs_scale = float(self.get_parameter("velocity_obs_scale").value)
        self.goal_flip_period_s = float(self.get_parameter("goal_flip_period_s").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.log_period_s = float(self.get_parameter("log_period_s").value)
        self.hand_frame = self.get_parameter("hand_frame_id").value
        self.cube_frame = self.get_parameter("cube_frame_id").value
        self.cube_pose_timeout_s = float(self.get_parameter("cube_pose_timeout_s").value)
        self.cube_max_age_s = float(self.get_parameter("cube_max_age_s").value)
        self.wait_for_frame = self.get_parameter("wait_for_frame").value
        self.frame_ready = not self.wait_for_frame

        # ========== Joint orders and limits ==========
        # the hand driver publishes and accepts joints in the order of config.yaml's joint_ids
        hand_config = yaml.safe_load(Path(self.get_parameter("hand_config_path").value).read_text())
        self.joint_order = list(hand_config["joint_ids"])
        self.joint_order_usd = list(JOINT_LIMITS_USD_DEG)
        assert set(self.joint_order) == set(self.joint_order_usd), (
            f"hand config joints {self.joint_order} do not match the policy's joints {self.joint_order_usd}"
        )
        self.robot_to_usd_idx = [self.joint_order.index(name) for name in self.joint_order_usd]
        self.usd_to_robot_idx = [self.joint_order_usd.index(name) for name in self.joint_order]
        self.limits_usd = np.deg2rad(np.array(list(JOINT_LIMITS_USD_DEG.values()), dtype=np.float32))

        # ========== Policy ==========
        if not self.policy_ckpt_path:
            raise ValueError("Set policy_ckpt_path to the exported policy.pt (faive_lab logs/.../exported/policy.pt)")
        self.policy = torch.jit.load(self.policy_ckpt_path, map_location="cpu").eval()
        # the input size tells whether the policy was trained with the cube pose (camera feedback) or without, and
        # whether each camera frame carries its age (r26: works without detections) or not (r24, r25)
        self.obs_dim = None
        candidates = {OBS_DIM: (0, False)}
        for frames in range(1, MAX_CUBE_FRAMES + 1):
            candidates[OBS_DIM + CUBE_POSE_DIM * frames] = (frames, False)
            candidates[OBS_DIM + CUBE_DETECTION_DIM * frames] = (frames, True)
        for obs_dim in sorted(candidates):
            try:
                with torch.no_grad():
                    out = self.policy(torch.zeros(1, obs_dim))
            except RuntimeError:
                continue
            self.obs_dim = obs_dim
            break
        if self.obs_dim is None:
            raise ValueError(f"The policy accepts none of {sorted(candidates)} observations")
        assert out.shape[-1] == 17, f"policy outputs {out.shape[-1]} actions, expected 17"
        # A recurrent policy (r33: GRU student) keeps its memory in the TorchScript module, and the size probe above
        # already advanced it. Load it again so that the first control step starts from zero memory, as in the sim.
        self.policy = torch.jit.load(self.policy_ckpt_path, map_location="cpu").eval()
        self.num_cube_frames, self.cube_ages = candidates[self.obs_dim]
        self.use_cube_obs = self.num_cube_frames > 0
        self.obs_T_hand = OBS_T_HAND
        if self.cube_ages:
            contract = load_camera_contract(self.policy_ckpt_path, self.obs_dim, self.num_cube_frames,
                                            self.policy_hz, self.action_alpha, self.velocity_obs_scale)
            if contract is not None:
                self.cube_max_age_s = float(contract["max_age_s"])
                self.obs_T_hand = np.asarray(contract["obs_T_hand"])
        self.cube_history = CubeDetectionHistory(max(1, self.num_cube_frames), self.cube_max_age_s)
        if self.cube_ages:
            inputs = (f"with the last {self.num_cube_frames} cube detection(s) and their age (no detection after "
                      f"{self.cube_max_age_s} s = no camera): runs with and without the camera")
        elif self.use_cube_obs:
            inputs = f"with the last {self.num_cube_frames} cube pose(s) from the camera: needs the cube detected"
        else:
            inputs = "without the cube pose (only logged)"
        self.get_logger().info(
            f"Loaded policy {self.policy_ckpt_path} ({self.obs_dim} observations -> 17 actions), {inputs}"
        )

        # ========== ROS2 Publishers and Subscribers ==========
        self.hand_pub = self.create_publisher(Float32MultiArray, self.get_parameter("command_topic").value, 10)
        self.create_subscription(Float32MultiArray, "/right/hand/joint_position_from_motor", self.hand_pos_callback, 10)
        self.create_subscription(Float32MultiArray, "/right/hand/joint_velocity_from_motor", self.hand_vel_callback, 10)

        # ========== TF2 Listener ==========
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.last_object_pos = None
        self.last_object_quat = None
        self.last_cube_stamp = None  # image time stamp of the last detection (seconds)
        self.cube_frames = None  # r24/r25: the last num_cube_frames [pos, quat] given to the policy, oldest first
        # r26: the last num_cube_frames distinct detections as (image stamp, [pos, quat]), oldest first
        self.camera_sees_cube = False

        # ========== State (USD order) ==========
        self.current_joint_pos = None  # robot order, radians
        self.current_joint_vel = None
        self.last_action_usd = np.zeros(17, dtype=np.float32)  # raw policy output, as the task's last_action
        self.last_joint_cmd_usd = None                          # set from the measured pose on the first step, as at a reset
        self.goal_command = np.array([0.0, 0.0, -0.05, 0.0, 0.0, float(self.get_parameter("goal_ang_vel").value)],
                                     dtype=np.float32)
        self.start_time = None
        self.last_flip_time = None
        self.last_log_time = None

        self.csv_writer = None
        csv_log_path = self.get_parameter("csv_log_path").value
        if csv_log_path:
            self.csv_file = open(csv_log_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["time"] + [f"obs_{i}" for i in range(self.obs_dim)]
                                     + [f"cmd_{name}" for name in self.joint_order_usd]
                                     + ["cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz"])

        self.create_timer(1.0 / self.policy_hz, self.process_once)
        mode = "DRY RUN (not commanding the hand)" if self.dry_run else "COMMANDING THE HAND"
        self.get_logger().info(f"Policy inference v2 running at {self.policy_hz} Hz, alpha {self.action_alpha}: {mode}")

    def hand_pos_callback(self, msg: Float32MultiArray):
        self.current_joint_pos = float32_multiarray_to_numpy(msg)

    def hand_vel_callback(self, msg: Float32MultiArray):
        self.current_joint_vel = float32_multiarray_to_numpy(msg)

    def normalize(self, joint_pos_usd: np.ndarray) -> np.ndarray:
        """Isaac Lab's scale_transform (joint_pos_limit_normalized): limits -> [-1, 1], not clipped."""
        low, high = self.limits_usd[:, 0], self.limits_usd[:, 1]
        return (2.0 * (joint_pos_usd - (low + high) / 2) / (high - low)).astype(np.float32)

    def denormalize(self, action_usd: np.ndarray) -> np.ndarray:
        """Isaac Lab's unscale_transform (rescale_to_limits): [-1, 1] -> limits."""
        low, high = self.limits_usd[:, 0], self.limits_usd[:, 1]
        return (low + (action_usd + 1.0) / 2 * (high - low)).astype(np.float32)

    def lookup_transform_from_tf(self, parent_frame, child_frame):
        """Latest parent_T_child from TF as a 4x4 matrix with its time stamp (seconds), or (None, None)."""
        try:
            tf_msg = self.tf_buffer.lookup_transform(parent_frame, child_frame, Time())
        except tf2_ros.TransformException:
            return None, None
        t, q = tf_msg.transform.translation, tf_msg.transform.rotation
        values = np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w])
        if not np.isfinite(values).all() or np.linalg.norm(values[3:]) < 1e-8:
            return None, None
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()  # ROS order is (x, y, z, w)
        transform[:3, 3] = [t.x, t.y, t.z]
        return transform, Time.from_msg(tf_msg.header.stamp).nanoseconds * 1e-9

    def process_once(self):
        if self.current_joint_pos is None or self.current_joint_vel is None:
            self.get_logger().info("Waiting for hand state...", throttle_duration_sec=2.0)
            return
        joint_pos_usd = np.asarray(self.current_joint_pos, dtype=np.float32)[self.robot_to_usd_idx]
        joint_vel_usd = np.asarray(self.current_joint_vel, dtype=np.float32)[self.robot_to_usd_idx]
        if not (np.all(np.isfinite(joint_pos_usd)) and np.all(np.isfinite(joint_vel_usd))):
            self.get_logger().warn("Non-finite joint state, skipping this step", throttle_duration_sec=1.0)
            return
        if not self.frame_ready:
            # before the first step, so the policy starts (moving average, last action) from the pose after it
            if not self.tf_buffer.can_transform(self.hand_frame, self.wait_for_frame, Time()):
                self.get_logger().info(f"Waiting for the camera calibration ({self.hand_frame} -> "
                                       f"{self.wait_for_frame}): the hand closes, the tag is measured, the hand "
                                       "opens again...", throttle_duration_sec=2.0)
                return
            self.frame_ready = True
            self.get_logger().info("Camera calibrated, starting the policy")

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time is None:
            # as at a reset in the task: the moving average starts from the current pose, last action is zero
            self.start_time = self.last_flip_time = self.last_log_time = now
            self.last_joint_cmd_usd = joint_pos_usd.copy()
        if self.goal_flip_period_s > 0 and now - self.last_flip_time >= self.goal_flip_period_s:
            self.goal_command[5] *= -1.0
            self.last_flip_time = now
            self.get_logger().info(f"Spin direction flipped: goal angular velocity {self.goal_command[5]:+.2f} rad/s")

        # === cube pose (Subtask 2) ===
        hand_T_cube, stamp = self.lookup_transform_from_tf(self.hand_frame, self.cube_frame)
        # the camera position is a static transform, so the stamp is that of the detector's image: a new stamp is a
        # new detection (between detections TF keeps returning the last one)
        if self.cube_history.last_now is not None and now < self.cube_history.last_now:
            self.last_cube_stamp = self.last_object_quat = None
        detection_pose = None
        if (hand_T_cube is not None and np.isfinite(stamp) and 0 <= now - stamp < self.cube_max_age_s
                and (self.last_cube_stamp is None or stamp > self.last_cube_stamp)):
            self.last_cube_stamp = stamp
            obs_T_cube = self.obs_T_hand @ hand_T_cube
            object_pos = obs_T_cube[:3, 3].astype(np.float32)
            # Convert quaternion from ROS (x,y,z,w) to policy (w,x,y,z)
            x, y, z, w = Rotation.from_matrix(obs_T_cube[:3, :3]).as_quat()
            object_quat = np.array([w, x, y, z], dtype=np.float32)
            # q and -q are the same rotation; the simulation's quaternion changes continuously, so keep the previous sign
            if self.last_object_quat is not None and np.dot(object_quat, self.last_object_quat) < 0:
                object_quat = -object_quat
            self.last_object_pos, self.last_object_quat = object_pos, object_quat
            detection_pose = np.concatenate([object_pos, object_quat])
        tracked = self.last_cube_stamp is not None and 0 <= now - self.last_cube_stamp <= self.cube_pose_timeout_s
        object_pos, object_quat = (self.last_object_pos, self.last_object_quat) if tracked else (None, None)

        if self.cube_ages:
            # r26: the last detections with their age, "no camera" where there is none; never waits for the cube
            cube_obs = self.cube_history.observe(now, stamp, detection_pose)
            self.cube_frames = list(cube_obs)
            sees_cube = cube_obs[-1][-1] < 1.0
            if sees_cube != self.camera_sees_cube:
                if sees_cube:
                    self.get_logger().info("Cube detected: the policy uses the camera")
                else:
                    self.get_logger().warn(f"No cube detection for {self.cube_max_age_s} s: the policy runs on the "
                                           "joint state only until the cube is detected again")
                self.camera_sees_cube = sees_cube
        elif self.use_cube_obs:
            # r24/r25: the policy needs a cube pose every step: wait for the first detection, then keep the last
            # detected pose
            if not tracked:
                if self.cube_frames is None:  # nothing given to the policy yet: wait for a current detection
                    self.get_logger().info("Waiting for the first cube detection...", throttle_duration_sec=2.0)
                    return
                self.get_logger().warn("Cube not tracked, using its last detected pose", throttle_duration_sec=1.0)
                object_pos, object_quat = self.last_object_pos, self.last_object_quat
            # one frame per policy step, as in the simulation; the first pose fills all frames
            pose = np.concatenate([object_pos, object_quat])
            if self.cube_frames is None:
                self.cube_frames = [pose] * self.num_cube_frames
            else:
                self.cube_frames = self.cube_frames[1:] + [pose]

        # === observation, in the task's order ===
        obs = np.concatenate(
            [self.normalize(joint_pos_usd), joint_vel_usd * self.velocity_obs_scale]
            + (self.cube_frames if self.use_cube_obs else [])
            + [self.goal_command, self.last_action_usd, self.last_joint_cmd_usd]
        ).astype(np.float32)

        # === policy ===
        with torch.no_grad():
            action_usd = self.policy(torch.from_numpy(obs).unsqueeze(0))[0].numpy().astype(np.float32)
        if not np.isfinite(action_usd).all():
            self.get_logger().warn("Non-finite policy action, skipping this step", throttle_duration_sec=1.0)
            return
        # RslRlVecEnvWrapper clips before env.step: last_action must contain the clipped action too.
        action_usd = np.clip(action_usd, -1.0, 1.0)
        self.last_action_usd = action_usd

        # === action processing, as the task's EMAJointPositionToLimitsAction ===
        target_usd = self.denormalize(np.clip(action_usd, -1.0, 1.0))
        target_usd = self.action_alpha * target_usd + (1.0 - self.action_alpha) * self.last_joint_cmd_usd
        target_usd = np.clip(target_usd, self.limits_usd[:, 0], self.limits_usd[:, 1]).astype(np.float32)
        self.last_joint_cmd_usd = target_usd

        if not self.dry_run:
            self.hand_pub.publish(numpy_to_float32_multiarray(target_usd[self.usd_to_robot_idx]))
        if self.csv_writer is not None:
            cube = np.concatenate([object_pos, object_quat]).tolist() if tracked else [float("nan")] * 7
            self.csv_writer.writerow([now - self.start_time] + obs.tolist() + target_usd.tolist() + cube)
        if self.log_period_s > 0 and now - self.last_log_time >= self.log_period_s:
            self.last_log_time = now
            norm = self.normalize(joint_pos_usd)
            cube = (f"pos {np.round(object_pos, 3)} quat {np.round(object_quat, 2)} "
                    f"age {(now - self.last_cube_stamp) * 1000:.0f} ms" if tracked else "not tracked")
            self.get_logger().info(
                "joint_pos (normalised): " + " ".join(f"{n}={v:+.2f}" for n, v in zip(self.joint_order_usd, norm))
                + f" | |action| max {np.abs(action_usd).max():.2f} | cube {cube}"
            )

    def destroy_node(self):
        if self.csv_writer is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNodeV2()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        print("\nShutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():  # a signal may already have shut the context down
            rclpy.shutdown()


if __name__ == "__main__":
    main()
