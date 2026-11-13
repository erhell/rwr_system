## ORCA v2 hand (RWR 2026)

### Run a faive_lab policy on the hand

`rl_inference/policy_inference_cube_spin_v2.py` runs a policy trained in faive_lab (task `orca-v2-inhand-repose-v0`) on the real hand. Use the TorchScript file that faive_lab's `play.py` exports: `logs/rsl_rl/orca_v2_repose/<run>/exported/policy.pt`.

1. Start the hand controller (see [src/hand_control/README.md](../src/hand_control/README.md)):
    ```bash
    ros2 launch hand_control orcahand_v2_controller_launch.yaml hand_chirality:=right launch_rviz:=true
    ```
2. In a second terminal, start the policy. It commands the hand right away:
    ```bash
    ros2 run experiments policy_inference_cube_spin_v2.py --ros-args -p policy_ckpt_path:=/path/to/exported/policy.pt
    ```
    To check a new policy without moving the hand first, add `-p dry_run:=true`: it then only computes and logs the actions.
3. Stop with Ctrl+C; the hand holds its last commanded pose. For an emergency stop: `ros2 service call /right/hand/estop std_srvs/srv/SetBool "{data: true}"`.

Useful parameters (add each as `-p name:=value`):

| Parameter | Default | Meaning |
|---|---|---|
| `dry_run` | `false` | `true`: compute and log actions, but do not command the hand |
| `goal_ang_vel` | `1.0` | commanded spin speed about z in rad/s; the sign sets the direction |
| `action_alpha` | `-1` | moving average of the joint commands; `-1` takes it from the `deploy_meta.json` next to the policy (r29-r32: 0.3, earlier: 0.2; 0.2 without one). Also a `cube_spin_v2.launch.py` argument |
| `goal_flip_period_s` | `0.0` | if > 0, flip the spin direction every this many seconds |
| `csv_log_path` | `""` | if set, log observations, joint commands and the cube pose to this CSV file |
| `policy_hz` | `20.0` | policy rate; must match the training task |
| `cube_max_age_s` | `1.0` | r26 policies: a cube detection this old counts as "no camera"; must match `max_age_s` of the training task |
| `clip_last_action` | `false` | `false`: the policy's previous action is fed back unclipped, as before 2026-11-11 (the policies spun faster on the hand that way); `true`: clipped to [-1, 1] like the training up to r33 |
| `fix_wrist`, `fixed_wrist_deg` | `false`, `15.0` | hold the wrist at this angle instead of following the policy (+ = fingertips towards the palm); r34 is trained that way |
| `run_seconds` | `0.0` | if > 0, stop by itself after this long (from the first step) |

The joint limits, observation order and scales in the script must match the faive_lab task (`orca_v2_repose_env_cfg.py`).

### Spin panel (web page for test series)

`rl_inference/spin_panel.py` starts the policy runs from a local web page and records what you measure on the hand.
With the hand controller running (step 1 above), in a second terminal:
```bash
python rl_inference/spin_panel.py        # then open http://localhost:8765
```
Pick the model, speed (1.0 / 1.3 / 1.6 / 2.0 or any), direction (CCW / CW / flip), run time, the clip and fixed-wrist
options, Start; afterwards type in the measured speed (or count turns), drops and a note. A grid shows which
combinations are done. Every run logs its CSV to `logs/policy_runs/` and the measurements go to
`logs/policy_runs/real_hand_results.csv`. The models and their run folders are listed in `MODELS` at the top of the
script. The same from the terminal: `source rl_inference/spin.zsh`, then `spin <model> <speed> [ccw|cw|flip] [real]`
or `spin_all` for a series.

### Look at a run in Foxglove

Foxglove opens MCAP bags (File → Open local file), not CSV files. Two ways to get one:

- **Record a bag during the run.** In another terminal, start this before the policy and stop it with Ctrl+C after it:
    ```bash
    ros2 bag record -s mcap -a -o logs/policy_runs/bag_$(date +%Y%m%d_%H%M%S)
    ```
    It records all topics: the commanded and measured joint states, TF, and the policy's log lines (`/rosout`, for Foxglove's Log panel).
- **Convert a CSV log** (`csv_log_path`):
    ```bash
    python rl_inference/policy_csv_to_mcap.py ../logs/policy_runs/<run>.csv
    ```
    It writes `<run>_csv/<run>_csv_0.mcap` with `/policy/joint_pos` (measured, rad), `/policy/joint_cmd` (commanded, rad), `/policy/last_action`, `/policy/goal_ang_vel` and `/policy/cube_pose` (only while the cube was tracked). The joints are in the policy's order (0 wrist, 1-4 abd index/middle/pinky/ring, 5 thumb_cmc, 6-9 mcp index/middle/pinky/ring, 10 thumb_abd, 11-14 pip index/middle/pinky/ring, 15 thumb_mcp, 16 thumb_dip). In a Plot panel, `/policy/joint_cmd.position[6]` and `/policy/joint_pos.position[6]` show how the index MCP follows its command.

### Measure the hand's lag

```bash
python rl_inference/joint_lag_real.py /path/to/<run>/exported/deploy_meta.json ../logs/policy_runs/<policy>_*.csv
```

prints per joint after how many policy steps a command change shows up in the measured joint velocity, and the policy loop timing. On the r21_sysid_16k runs (2026-11-09) it is 3 steps (150 ms) for most joints, while the sim (rwr_faive_lab's `scripts/rsl_rl/joint_lag.py`) gave 2 steps; rwr_faive_lab's r30 is trained with that extra latency.

### Detect the cube

The cube detector in `rl_inference/pose_estimation` finds the cube's corners in the OAK-D image and computes the cube pose from them (PnP).

0. Download the model files once: `rl_inference/pose_estimation/download_models.sh`.
1. Set your camera's ID (printed on the camera) in `cfgs/cams/oakd_cube_task.yaml`, and connect the camera to a USB 3 port with a USB 3 cable.
2. Start the camera, the cube detector and RViz:
    ```bash
    ros2 launch experiments cube_spin_v2.launch.py model_path:=/path/to/pose_estimation/finetuned_models/maskrcnn-finetuned_lightning_new_adamw.ckpt
    ```
3. Check the detection: in RViz, the cube should follow the real cube when you move and turn it. `ros2 run tf2_ros tf2_echo oakd_bottom_view_frame cube` prints the cube position relative to the camera; its distance should match a ruler measurement.

To test only the camera and the detection (no hand, AprilTag, policy or RViz), `cube_view.launch.py` starts the camera, the detector and `rl_inference/cube_view.py`, which shows the camera image in a window with the estimated cube drawn on it (edges and axes; green while the detection is fresh, orange when older than 0.2 s, gone after 1 s) and a status line (detection age, distance to the camera, detections per second, camera frames per second). In the window, `q` quits and `s` saves a PNG.
```bash
ros2 launch experiments cube_view.launch.py model_path:=/path/to/pose_estimation/finetuned_models/maskrcnn-finetuned_lightning_new_adamw.ckpt
```
RViz shows the camera image only with an **Image** display set to 'Reliability Policy: Best Effort', because the camera publishes it that way.

### Calibrate the camera

To get the cube pose relative to the hand, the camera's position is measured against an AprilTag (`tag36h11`). `cube_spin_v2.launch.py` does this at start by default and saves the result; with `use_saved_camera:=true` it reuses the saved one instead (step 5).

1. Mount the tag rigidly next to the hand where the camera sees it, and measure the size of its black square (without the white border). The defaults are the task description's tag, `tag_id:=25 tag_size:=0.08` (36h11, ID 25, 8 cm; until 2026-11-11 they were 9 and 0.075 by mistake); pass yours if they differ. A wrong size scales the measured camera distance and with it every cube position.
2. The open hand hides the tag, so at every launch the `apriltag_calibration` node first closes the hand into a fist (orca_core's `power_grasp` pose, wrist unchanged) as soon as the hand controller publishes the joint state (start it before or with the launch, e.g. `enable_hand_controller:=true`). It then averages 30 detections (about 3 s), collects again if they spread more than 1 cm, opens the hand back to where it started, and only then logs `Camera position published: tag_frame → oakd_bottom_view_frame` and publishes it on `/tf_static`. Keep the camera still meanwhile, and keep the cube out of the hand until it has opened.
3. Until then there is no cube pose relative to the hand, and the policy (`wait_for_frame`) does not command the hand, so it never fights the fist. Every 3 s the node logs why it waits (`AprilTag 25 ... not in view`, hand controller not running).
4. If the thumb or another finger still hides the tag in the fist, change those joints (degrees) at launch, e.g. `fist_pose:='thumb_abd=10 thumb_cmc=0'`; the node logs the fist it uses at startup. Without a hand controller the tag is measured if it happens to be visible.
5. The result is saved to `~/camera_extrinsics.npy`. To skip the measurement (and the fist) on later launches, add `use_saved_camera:=true`: the launch then publishes the saved position right away and logs when it was measured. Launch without it to measure again whenever the camera or the tag has moved.

To look at the detection, run the script on its own with the camera started, from the `experiments` folder: `python3 rl_inference/apriltag_calibration.py --tag-id 25 --tag-size 0.08 --output /tmp/cam_T_tag.npy --visualize`.

### Set the hand's position

The launch places the hand relative to the tag with the `hand_to_tag_*` arguments: the tag's pose seen from the mount, i.e. the centre of the forearm's flat bottom face (frame `mount` in `orcahand_v2.urdf.xacro`). The mount's axes: x from the index finger towards the little finger, y out of the palm, z up the forearm.

The defaults (`0.035 -0.04 0.34`, roll 60, pitch 0, yaw 0) are for the ORCA v2 stand: the hand on the post, palm up, fingers tilted 30° down towards the tag, which lies flat on the base extrusion about 27 cm from the mount towards the fingers and 20 cm below it. They were estimated from a photo of the stand (about ±1.5 cm), so do step 4 once; measure again only if the stand changes.

1. Measure `hand_to_tag_x/y/z` (in metres) from the mount to the tag's centre, along the mount's axes. The mount is hidden under the forearm: it is 4.65 cm in from the index and little-finger sides, and 4.8 cm in from the palm and back sides.
2. Find `hand_to_tag_roll/pitch/yaw` (in degrees): how the tag's axes (x red, y green, z blue into the tag, as drawn by the calibration) are turned relative to the mount's axes; roll about x, then pitch about the new y, then yaw about the new z. With the tag square to the hand, these are multiples of 90°. For example, a tag lying flat in front of the palm with its x along the mount's x: roll 180, pitch 0, yaw 0.
3. Launch with your values, e.g.:
    ```bash
    ros2 launch experiments cube_spin_v2.launch.py \
        hand_to_tag_x:=0 hand_to_tag_y:=0.15 hand_to_tag_z:=0 hand_to_tag_roll:=180 hand_to_tag_pitch:=0 hand_to_tag_yaw:=0
    ```
4. Check: `ros2 run tf2_ros tf2_echo mount tag_frame` prints your x/y/z, and in RViz the hand model stands where the real hand is.


### Troubleshooting

**Camera**
- `oakd camera <ID> not available` in the log: wrong ID in `oakd_cube_task.yaml`, no udev rule, or the camera was still resetting from the previous run (wait about 3 s and launch again).
- `StereoDepthProperties has no attribute 'MedianFilter'`: depthai 3 is installed; run `pixi run pip install "depthai<3"`.
- `ros2 topic hz /oakd_bottom_view/color` shows images, but RViz does not: use an **Image** display (not Camera) with 'Reliability Policy: Best Effort'.

**Hand controller**
- The hand is not found or the serial port is not accessible: check the USB cable and add yourself to the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and in again).
- Joint positions look wrong after rebuilding: `calibration.yaml` lives in `install/` (see [src/hand_control/README.md](../src/hand_control/README.md)); calibrate again.

**Policy**
- The script reads the policy's input size at startup and logs it:
    - 74 = joint state only (the cube pose is only logged).
    - 74 + 8 × N = trained with the last N cube detections and their age (faive_lab r26: 98). Such a policy uses the camera when the detector sees the cube and spins on the joint state alone when it does not: it starts without waiting for a detection, and runs without the camera or without `model_path:=`. The log says when it switches (`No cube detection for 1.0 s: the policy runs on the joint state only ...`, `Cube detected: the policy uses the camera`).
    - 74 + 7 × N = trained with the last N camera poses (`object_pos` + `object_quat` per frame; r24: 81, r25: 95). These need the camera: start the cube detector with `model_path:=`; the policy waits for the first detection and keeps the last detected pose while the cube is lost.
- r25 and r26 are trained on 1.2-2.0 rad/s, so run them with `goal_ang_vel` in that range (e.g. 1.5).
- `The policy accepts none of [...] observations`: the training task's observation differs; the observation in the script must match it.
- Before training a camera policy, run `rl_inference/measure_cube_noise.py` (see its header) to get the camera model's values for the simulation: noise, delay, how often the detector delivers a pose, outages.

**Cube detection**
- `checkpoint.pth.tar not found` when the detector starts: the model files are not downloaded yet; run `experiments/rl_inference/pose_estimation/download_models.sh`.
