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
       goal_command    [0, 0, -0.05, angular velocity about x, y, z]: only the goal_axis component is non-zero (z unless
                       goal_axis is set; x / y only for models trained on all axes, r100xyz)
       last_action     the policy's previous raw output
       last_joint_cmd  the previous joint position command (radians)
3. Run the policy, map its output onto the joint limits and smooth it with the same filter as the simulation: the
   moving average (target = alpha * new + (1 - alpha) * previous), or for models whose deploy_meta.json has
   action_filter "butter2" (r60, r61) the 2nd-order Butterworth low-pass with its coefficients
4. Publish the joint position command on /right/hand/policy_output, which hand_control_node executes (unless dry_run)

Servo profile: before the first step the node sets the servos' ACC register to what the model was trained with
(deploy_meta.json servo_profile, r60+: 254 on some joints) through hand_control_node's /right/hand/set_servo_acc, waits
until /right/hand/servo_acc reports it back and sets 150 again on exit. Models without a servo_profile get 150 on all
joints (orca_core's default, what they always ran with). A model with a servo_profile refuses to command the hand when
the controller cannot confirm it; the others run anyway, as before.

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
import json
import signal
import time
from camera_feedback import CubeDetectionHistory, deploy_meta_action_alpha, load_camera_contract
from pathlib import Path

import numpy as np
import rclpy
import rclpy.executors
import tf2_ros
import torch
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
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
DEFAULT_SERVO_ACC = 150.0  # orca_core's ACC register default: every model without a servo_profile ran with it
# Hardware caps of the ACC register (read back 2026-11-15 13:17 with experiments/orca/chirp.py: the wrist's HLS3930 holds
# 250 when 254 is written; the 16 finger motors, HLS3915, hold 254). A model's ACC above a cap is set and checked at the
# cap (r100*: 254 on the wrist -> 250, 1.6 % less ramp on a joint the node holds still); every other value must match.
SERVO_ACC_CAP = {"wrist": 250.0}
# Position-loop gain KP (register 50, 2026-11-15): 32 is the motors' own value (read on all 17) and what every model
# without servo_profile.kp in its deploy_meta.json ran with; 48 = the KP 48 recording / fit (r103, deployment variant
# r60e_max_acc_kp48). orca_core never rewrites KP, so the node checks it for EVERY model and sets 32 after a run that
# changed it. Same range as hand_controller_v2.py.
DEFAULT_SERVO_KP = 32
MIN_SERVO_KP, MAX_SERVO_KP = 16, 64
# Motor goal-current limit (register 44, 2026-11-15): 900 mA = orca_core's default and what every model without
# servo_profile.max_current_ma ran with; r104: 1000. The register counts 6.5 mA, rounded down (900 -> 897,
# 1000 -> 994.5). Checked for every model, set back to 900 after a run that changed it. Same range as
# hand_controller_v2.py.
DEFAULT_MAX_CURRENT_MA = 900.0
CURRENT_UNIT_MA = 6.5
MIN_MAX_CURRENT_MA, MAX_MAX_CURRENT_MA = 300.0, 1000.0


def held_current_ma(requested_ma):
    """What register 44 holds for a requested limit (orca_core rounds down to 6.5 mA units)."""
    return int(requested_ma / CURRENT_UNIT_MA) * CURRENT_UNIT_MA

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
        # axis of goal_ang_vel (2026-11-15): the simulation's world axes, z up (vertical; negative = clockwise seen from
        # above), y along the fingers, x across the palm. x / y only for models trained on all axes (deploy_meta.json
        # command axis_mode "principal": r100xyz); the node refuses them for the z-only models
        self.declare_parameter("goal_axis", "z")
        # false (default): the last action is fed back unclipped, as this node did before 2026-11-11; the policies spun
        # the cube faster on the real hand that way (values up to ~2.5 that the task never gives them). true: clipped to
        # [-1, 1] like the training (RslRlVecEnvWrapper), as from 2026-11-11 to 11-13. Joint targets are clipped either way
        self.declare_parameter("clip_last_action", False)
        self.declare_parameter("run_seconds", 0.0)         # > 0: stop after this long (from the first step); 0: until Ctrl+C
        # true: the wrist does not follow the policy but is held at fixed_wrist_deg (+ = fingertips towards the palm side,
        # i.e. up with the palm facing up; limits -65..35). Reached through the same moving average as the policy's
        # targets (~0.5 s). The observations stay truthful: measured wrist, the command actually sent, the policy's action
        self.declare_parameter("fix_wrist", False)
        self.declare_parameter("fixed_wrist_deg", 15.0)
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
        self.goal_axis = str(self.get_parameter("goal_axis").value).strip().lower()
        if self.goal_axis not in ("x", "y", "z"):
            raise ValueError(f"goal_axis must be x, y or z, got {self.goal_axis!r}")
        self.goal_index = 3 + "xyz".index(self.goal_axis)  # position in goal_command
        self.clip_last_action = bool(self.get_parameter("clip_last_action").value)
        self.run_seconds = float(self.get_parameter("run_seconds").value)
        self.fix_wrist = bool(self.get_parameter("fix_wrist").value)
        self.fixed_wrist_deg = float(self.get_parameter("fixed_wrist_deg").value)
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
        self.wrist_idx = self.joint_order_usd.index("wrist")
        lo, hi = JOINT_LIMITS_USD_DEG["wrist"]
        if self.fix_wrist and not lo <= self.fixed_wrist_deg <= hi:
            raise ValueError(f"fixed_wrist_deg {self.fixed_wrist_deg} is outside the wrist limits [{lo}, {hi}] deg")

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
        # recurrent policies (r33 students) keep a GRU memory inside policy.pt: clear what the probe calls left in it
        self.reset_policy_memory()
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

        # ========== Action filter and servo profile of the training task (deploy_meta.json, r60+) ==========
        meta_path = Path(self.policy_ckpt_path).with_name("deploy_meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        action_filter = meta.get("action_filter") or {"type": "ema"}
        self.butter = None  # (b, a) of the Butterworth filter, None: moving average
        if action_filter.get("type") == "butter2":
            b = [float(v) for v in action_filter["b"]]
            a = [float(v) for v in action_filter["a"]]
            if len(b) != 3 or len(a) != 3 or abs(a[0] - 1.0) > 1e-9:
                raise ValueError(f"deploy_meta.json action_filter: expected b[3] and a[3] with a[0] = 1, got {action_filter}")
            self.butter = (b, a)
            filter_text = f"Butterworth 2nd order {action_filter.get('cutoff_hz', '?')} Hz, b {np.round(b, 4)} a {np.round(a, 4)}"
        elif action_filter.get("type") == "ema":
            filter_text = f"moving average alpha {self.action_alpha}"
        else:
            raise ValueError(f"deploy_meta.json action_filter type {action_filter.get('type')!r} is not supported")
        servo_profile = meta.get("servo_profile")
        if servo_profile:
            acc = servo_profile.get("acc") or {}
            missing = [j for j in self.joint_order if j not in acc]
            if missing:
                raise ValueError(f"deploy_meta.json servo_profile has no ACC for {missing}")
            self.servo_acc = np.array([float(acc[j]) for j in self.joint_order])
            self.servo_profile_name = servo_profile.get("name", "from deploy_meta.json")
        else:
            self.servo_acc = np.full(len(self.joint_order), DEFAULT_SERVO_ACC)
            self.servo_profile_name = None
        if not np.all((self.servo_acc >= 0) & (self.servo_acc <= 254)):
            raise ValueError(f"servo ACC values must be 0..254, got {self.servo_acc.tolist()}")
        # what the motors can hold: the model's ACC, capped where the hardware caps it
        self.servo_acc_trained = self.servo_acc.copy()
        self.servo_acc = np.array([min(a, SERVO_ACC_CAP.get(j, 254.0)) for j, a in zip(self.joint_order, self.servo_acc)])
        self.servo_acc_capped = {j: (t, a) for j, t, a in zip(self.joint_order, self.servo_acc_trained, self.servo_acc)
                                 if t != a}
        self.servo_acc_set = False  # true once this node changed the hand's ACC (restored on exit)
        self.servo_kp = int((servo_profile or {}).get("kp", DEFAULT_SERVO_KP))
        if not MIN_SERVO_KP <= self.servo_kp <= MAX_SERVO_KP:
            raise ValueError(f"deploy_meta.json servo_profile.kp {self.servo_kp} outside {MIN_SERVO_KP}..{MAX_SERVO_KP}")
        self.servo_kp_set = False   # true once this node changed the hand's KP (set back to 32 on exit)
        self.max_current_ma = float((servo_profile or {}).get("max_current_ma", DEFAULT_MAX_CURRENT_MA))
        if not MIN_MAX_CURRENT_MA <= self.max_current_ma <= MAX_MAX_CURRENT_MA:
            raise ValueError(f"deploy_meta.json servo_profile.max_current_ma {self.max_current_ma} outside "
                             f"{MIN_MAX_CURRENT_MA:g}..{MAX_MAX_CURRENT_MA:g}")
        self.max_current_set = False  # true once this node changed the motors' current limit (900 again on exit)
        self.servo_ready = False    # set by setup_servo_profile(); no policy step before
        self.done = False           # set when run_seconds is over

        # ========== ROS2 Publishers and Subscribers ==========
        self.hand_pub = self.create_publisher(Float32MultiArray, self.get_parameter("command_topic").value, 10)
        self.create_subscription(Float32MultiArray, "/right/hand/joint_position_from_motor", self.hand_pos_callback, 10)
        self.create_subscription(Float32MultiArray, "/right/hand/joint_velocity_from_motor", self.hand_vel_callback, 10)
        self.servo_acc_state = None  # ACC per joint as hand_control_node last read it back (hand config order)
        self.servo_acc_pub = self.create_publisher(Float32MultiArray, "/right/hand/set_servo_acc", 10)
        self.create_subscription(
            Float32MultiArray, "/right/hand/servo_acc", self.servo_acc_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        self.max_current_state = None  # current limit per joint (mA) as hand_control_node last read it back
        self.max_current_pub = self.create_publisher(Float32MultiArray, "/right/hand/set_max_current_ma", 10)
        self.create_subscription(
            Float32MultiArray, "/right/hand/max_current_ma", self.max_current_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        self.servo_kp_state = None  # KP per joint as hand_control_node last read it back (hand config order)
        self.servo_kp_pub = self.create_publisher(Float32MultiArray, "/right/hand/set_servo_kp", 10)
        self.create_subscription(
            Float32MultiArray, "/right/hand/servo_kp", self.servo_kp_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))

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
        self.goal_command = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0], dtype=np.float32)
        self.goal_command[self.goal_index] = float(self.get_parameter("goal_ang_vel").value)
        axis_mode = (meta.get("command") or {}).get("axis_mode", "per_axis")
        if self.goal_axis != "z" and axis_mode != "principal":
            raise ValueError(f"goal_axis {self.goal_axis}: this model was trained only about z (deploy_meta.json command "
                             f"axis_mode {axis_mode!r}); x / y need a model trained on all axes (r100xyz)")
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
        self.get_logger().info(f"Policy inference v2 running at {self.policy_hz} Hz, {filter_text}: {mode}")
        trained = meta.get("command") or {}
        self.get_logger().info(
            f"goal {self.goal_command[self.goal_index]:+.2f} rad/s about {self.goal_axis}"
            + (" (clockwise seen from above)" if self.goal_axis == "z" and self.goal_command[5] < 0 else
               " (counter-clockwise seen from above)" if self.goal_axis == "z" else "")
            + f"; trained on: axis_mode {trained.get('axis_mode', 'per_axis (z)')}, "
            + (f"speeds {trained.get('principal_speed_range')}" if trained.get("axis_mode") == "principal" else
               f"max {trained.get('angular_vel_range', ['?'])[-1]} rad/s")
            + (f", {100 * trained['negative_prob']:.0f} % clockwise" if trained.get("negative_prob") is not None else ""))
        self.get_logger().info("last action fed back " + ("clipped to [-1, 1] (as in training)" if self.clip_last_action
                                                           else "unclipped (as before 2026-11-11)"))
        if self.fix_wrist:
            self.get_logger().info(f"wrist fixed at {self.fixed_wrist_deg:+.1f} deg (+ = fingertips towards the palm side), "
                                   "the policy moves only the fingers")


    def reset_policy_memory(self):
        """Clear the memory of a recurrent policy (r33 students: GRU, exported by Isaac Lab with its state in the
        hidden_state buffer, LSTMs also in cell_state), as the sim does at an episode start. No-op for the
        feed-forward policies (r21-r32)."""
        for name in ("hidden_state", "cell_state"):
            state = getattr(self.policy, name, None)
            if isinstance(state, torch.Tensor):
                state.zero_()

    # ========== servo profile (ACC register per joint) ==========
    def servo_acc_callback(self, msg: Float32MultiArray):
        self.servo_acc_state = np.array(msg.data, dtype=np.float64)

    def acc_text(self, acc):
        return " ".join(f"{j}={'?' if not np.isfinite(a) else int(a)}" for j, a in zip(self.joint_order, acc))

    def acc_matches(self, wanted):
        state = self.servo_acc_state
        return state is not None and state.shape == wanted.shape and np.allclose(state, np.round(wanted), atol=0.5)

    def spin_until(self, condition, timeout_s):
        end = time.monotonic() + timeout_s
        while not condition() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return condition()

    def send_servo_acc(self, wanted, timeout_s):
        """Publish the ACC values until hand_control_node reports them back (True) or timeout_s passes (False)."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if self.servo_acc_pub.get_subscription_count() > 0:
                self.servo_acc_pub.publish(Float32MultiArray(data=[float(v) for v in wanted]))
            if self.spin_until(lambda: self.acc_matches(wanted), min(1.0, max(0.0, end - time.monotonic()))):
                return True
        return False

    def setup_servo_profile(self):
        """Before the first step: set the ACC the model was trained with. Raises if a model with a servo_profile
        cannot get it; models without one (150 everywhere) run anyway, as before this existed."""
        wanted, name = self.servo_acc, self.servo_profile_name
        for j, (trained, held) in self.servo_acc_capped.items():
            self.get_logger().info(f"servo ACC {j}: the model's servo profile asks for {trained:g}, the motor holds at most {held:g} "
                                   f"(hardware cap): set and checked at {held:g}")
        label = f"servo profile '{name}'" if name else f"servo ACC {DEFAULT_SERVO_ACC:g} on all joints (no servo_profile)"
        if not self.spin_until(lambda: self.servo_acc_state is not None, 3.0):
            msg = ("hand_control_node does not report /right/hand/servo_acc (not running, or started before the "
                   "servo-ACC code of 2026-11-15: restart it)")
            if self.dry_run:
                self.get_logger().warn(f"{msg}; dry run, the model needs {label}: {self.acc_text(wanted)}")
            elif name:
                raise RuntimeError(f"The model needs {label}, but {msg}. Not commanding the hand.")
            else:
                self.get_logger().warn(f"{msg}; running with the controller's ACC (150 unless changed)")
        elif self.acc_matches(wanted):
            self.get_logger().info(f"{label}: the hand already has it")
        elif self.dry_run:
            self.get_logger().warn(f"dry run: the model needs {label} ({self.acc_text(wanted)}), the hand has "
                                   f"{self.acc_text(self.servo_acc_state)}; not changed")
        else:
            self.servo_acc_set = True
            if self.send_servo_acc(wanted, 5.0):
                self.get_logger().info(f"{label} set: {self.acc_text(wanted)}")
            elif name:
                raise RuntimeError(f"Could not set {label}: the hand reports {self.acc_text(self.servo_acc_state)}. "
                                   "Not commanding the hand.")
            else:
                self.get_logger().warn(f"Could not confirm {label}: the hand reports "
                                       f"{self.acc_text(self.servo_acc_state)}")
        self.setup_servo_kp()
        self.setup_max_current()
        self.servo_ready = True

    # ========== motor goal-current limit (register 44) ==========
    def max_current_callback(self, msg: Float32MultiArray):
        self.max_current_state = np.array(msg.data, dtype=np.float64)

    def current_matches(self, requested_ma):
        state, held = self.max_current_state, held_current_ma(requested_ma)
        return (state is not None and state.shape == (len(self.joint_order),)
                and bool(np.all(np.abs(state - held) < CURRENT_UNIT_MA / 2)))

    def ma_text(self, ma):
        return " ".join(f"{j}={'?' if not np.isfinite(m) else f'{m:g}'}" for j, m in zip(self.joint_order, ma))

    def send_max_current(self, requested_ma, timeout_s):
        """Publish the limit for all joints until hand_control_node reports it back (True) or timeout_s passes."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if self.max_current_pub.get_subscription_count() > 0:
                self.max_current_pub.publish(Float32MultiArray(data=[float(requested_ma)] * len(self.joint_order)))
            if self.spin_until(lambda: self.current_matches(requested_ma), min(1.0, max(0.0, end - time.monotonic()))):
                return True
        return False

    def setup_max_current(self):
        """Before the first step: the model's current limit on all motors (900 mA unless its deploy_meta says
        otherwise). Raises if it cannot be set; with a hand controller that does not report the limit only a model
        that needs another limit refuses."""
        wanted = self.max_current_ma
        label = (f"motor current limit {wanted:g} mA (register holds {held_current_ma(wanted):g})"
                 + (" (orca_core's default)" if wanted == DEFAULT_MAX_CURRENT_MA else " (this model's)"))
        if not self.spin_until(lambda: self.max_current_state is not None, 3.0):
            msg = ("hand_control_node does not report /right/hand/max_current_ma (started before the current-limit "
                   "code of 2026-11-15: restart it)")
            if self.dry_run:
                self.get_logger().warn(f"{msg}; dry run, the model needs {label}")
            elif wanted != DEFAULT_MAX_CURRENT_MA:
                raise RuntimeError(f"The model needs {label}, but {msg}. Not commanding the hand.")
            else:
                self.get_logger().warn(f"{msg}: current limit not checked. The controller sets 900 mA at its start")
            return
        if self.current_matches(wanted):
            self.get_logger().info(f"{label} on all joints: the hand already has it")
        elif self.dry_run:
            self.get_logger().warn(f"dry run: the model needs {label}, the hand has "
                                   f"{self.ma_text(self.max_current_state)}; not changed")
        else:
            self.max_current_set = True
            if self.send_max_current(wanted, 5.0):
                self.get_logger().info(f"{label} set on all joints")
            else:
                raise RuntimeError(f"Could not set {label}: the hand reports {self.ma_text(self.max_current_state)} "
                                   "(a motor's protection current may cap it). Not commanding the hand.")

    # ========== servo position-loop gain KP (register 50) ==========
    def servo_kp_callback(self, msg: Float32MultiArray):
        self.servo_kp_state = np.array(msg.data, dtype=np.float64)

    def kp_matches(self, kp):
        state = self.servo_kp_state
        return state is not None and state.shape == (len(self.joint_order),) and bool(np.all(np.abs(state - kp) < 0.5))

    def send_servo_kp(self, kp, timeout_s):
        """Publish KP for all joints until hand_control_node reports it back (True) or timeout_s passes (False)."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if self.servo_kp_pub.get_subscription_count() > 0:
                self.servo_kp_pub.publish(Float32MultiArray(data=[float(kp)] * len(self.joint_order)))
            if self.spin_until(lambda: self.kp_matches(kp), min(1.0, max(0.0, end - time.monotonic()))):
                return True
        return False

    def setup_servo_kp(self):
        """Before the first step: the model's KP on all motors (32 unless its deploy_meta says otherwise). Raises if it
        cannot be set; with a hand controller that does not report KP only a model that needs another KP refuses."""
        wanted = self.servo_kp
        label = f"servo KP {wanted}" + (" (the motors' own)" if wanted == DEFAULT_SERVO_KP else " (this model's)")
        if not self.spin_until(lambda: self.servo_kp_state is not None, 3.0):
            msg = ("hand_control_node does not report /right/hand/servo_kp (started before the KP code of 2026-11-15: "
                   "restart it)")
            if self.dry_run:
                self.get_logger().warn(f"{msg}; dry run, the model needs {label}")
            elif wanted != DEFAULT_SERVO_KP:
                raise RuntimeError(f"The model needs {label}, but {msg}. Not commanding the hand.")
            else:
                self.get_logger().warn(f"{msg}: KP not checked. The motors hold 32 unless a KP run left another value "
                                       "(power-cycle the hand if unsure)")
            return
        if self.kp_matches(wanted):
            self.get_logger().info(f"{label} on all joints: the hand already has it")
        elif self.dry_run:
            self.get_logger().warn(f"dry run: the model needs {label}, the hand has {self.acc_text(self.servo_kp_state)}; "
                                   "not changed")
        else:
            self.servo_kp_set = True
            if self.send_servo_kp(wanted, 5.0):
                self.get_logger().info(f"{label} set on all joints")
            else:
                raise RuntimeError(f"Could not set {label}: the hand reports {self.acc_text(self.servo_kp_state)}. "
                                   "Not commanding the hand.")

    def restore_servo_profile(self):
        """After the run: 900 mA, KP 32 and ACC 150 on all joints again where this node changed them."""
        if self.max_current_set:
            if self.send_max_current(DEFAULT_MAX_CURRENT_MA, 2.0):
                self.get_logger().info(f"motor current limit back to {DEFAULT_MAX_CURRENT_MA:g} mA on all joints")
            else:
                self.get_logger().error(f"Could not set the motor current limit back to {DEFAULT_MAX_CURRENT_MA:g} mA: "
                                        "restart hand_control_node (it sets 900 mA at its start)")
        if self.servo_kp_set:
            if self.send_servo_kp(DEFAULT_SERVO_KP, 2.0):
                self.get_logger().info(f"servo KP back to {DEFAULT_SERVO_KP} on all joints")
            else:
                self.get_logger().error(f"Could not set the servo KP back to {DEFAULT_SERVO_KP}: restart hand_control_node "
                                        "and check its 'servo KP at start' line, or power-cycle the hand")
        if not self.servo_acc_set:
            return
        default = np.full(len(self.joint_order), DEFAULT_SERVO_ACC)
        if self.send_servo_acc(default, 2.0):
            self.get_logger().info(f"servo ACC back to {DEFAULT_SERVO_ACC:g} on all joints")
        else:
            self.get_logger().error(f"Could not set the servo ACC back to {DEFAULT_SERVO_ACC:g}: restart "
                                    "hand_control_node before running a model without a servo_profile elsewhere")

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
        if not self.servo_ready or self.done:
            return
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
            # Butterworth states (last two inputs and outputs) at rest at the current pose, as at a reset in the task
            self.bx1 = self.bx2 = self.by1 = self.by2 = joint_pos_usd.copy()
            self.reset_policy_memory()
        if self.run_seconds > 0 and now - self.start_time >= self.run_seconds:
            # ends the loop in main(), which then restores the servo ACC and closes the CSV; the hand keeps the last
            # command
            self.get_logger().info(f"Run time of {self.run_seconds:g} s over, stopping")
            self.done = True
            return
        if self.goal_flip_period_s > 0 and now - self.last_flip_time >= self.goal_flip_period_s:
            self.goal_command[self.goal_index] *= -1.0
            self.last_flip_time = now
            self.get_logger().info(f"Spin direction flipped: goal angular velocity {self.goal_command[self.goal_index]:+.2f} "
                                   f"rad/s about {self.goal_axis}")

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
        # default: last_action unclipped, as before 2026-11-11. clip_last_action true: clipped like RslRlVecEnvWrapper
        # does before env.step in training. The joint target below is clipped either way
        if self.clip_last_action:
            action_usd = np.clip(action_usd, -1.0, 1.0)
        self.last_action_usd = action_usd

        # === action processing, as the task's EMAJointPositionToLimitsAction / DelayedEMAJointPositionToLimitsAction ===
        target_usd = self.denormalize(np.clip(action_usd, -1.0, 1.0))
        if self.fix_wrist:  # held still; the filter below eases it there from the start pose
            target_usd[self.wrist_idx] = np.deg2rad(self.fixed_wrist_deg)
        if self.butter is None:
            target_usd = self.action_alpha * target_usd + (1.0 - self.action_alpha) * self.last_joint_cmd_usd
            target_usd = np.clip(target_usd, self.limits_usd[:, 0], self.limits_usd[:, 1]).astype(np.float32)
        else:
            # y_k = b0 x_k + b1 x_k-1 + b2 x_k-2 - a1 y_k-1 - a2 y_k-2, y clamped to the limits (and kept clamped)
            (b0, b1, b2), (_, a1, a2) = self.butter
            x = target_usd
            y = b0 * x + b1 * self.bx1 + b2 * self.bx2 - a1 * self.by1 - a2 * self.by2
            y = np.clip(y, self.limits_usd[:, 0], self.limits_usd[:, 1]).astype(np.float32)
            self.bx2, self.bx1, self.by2, self.by1 = self.bx1, x, self.by1, y
            target_usd = y
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


def _interrupt(*_):
    raise KeyboardInterrupt


def main(args=None):
    # rclpy's own signal handler would shut ROS down at Ctrl+C before the servo ACC can be set back: handle SIGINT
    # (Python's KeyboardInterrupt) and SIGTERM (the spin panel's second try) here instead
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)  # also when started with SIGINT ignored (background job of a script)
    signal.signal(signal.SIGTERM, _interrupt)
    node, code = None, 0
    try:
        node = PolicyInferenceNodeV2()
        node.setup_servo_profile()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        print("\nShutting down...")
    except RuntimeError as e:  # the servo profile the model needs could not be set
        print(f"\nERROR: {e}", flush=True)
        code = 1
    finally:
        if node is not None:
            node.done = True  # no more policy steps while the restore below spins the node
            try:
                if rclpy.ok():
                    node.restore_servo_profile()
            except KeyboardInterrupt:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
