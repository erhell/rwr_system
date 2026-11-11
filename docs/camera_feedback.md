# ORCA v2 camera feedback (r27)

r27a/r27 in `rwr_faive_lab` keep the sysID and the training recipe of
`2026-11-08_20-44-06_r21_sysid_16k`: 2,000 iterations with the r15 reset, then 500
with the aligned r21 reset, seed 42, 20 Hz, the same rewards/PPO/actions, and
0.9–1.5 rad/s commands. They add 24 camera observations after the 34 joint observations.

The existing OakD → MaskRCNN corners → PnP → filtered, image-stamped TF pipeline
supplies cube poses at deployment. Training uses a vectorized detector model rather
than rendering 32k images: 5 mm position noise, quaternion noise 0.05, 0–150 ms
latency, detections at 5–20 Hz, outages up to 2 s, and 20% of episodes with no camera.
These are the existing r26 assumptions, not a newly measured detector calibration;
`experiments/rl_inference/measure_cube_noise.py` can measure the real camera.

**Checked against the task description (2026-11-11).** The tracker here matches it (Mask R-CNN 8 corner keypoints,
PnP + RANSAC 5000 iterations / 3 px / >= 3 inliers, 45 mm cube, Kalman filter on the position, slerp alpha 0.25 on the
orientation, TF at <= 20 Hz). Two of its sim-to-real points were missing from the simulated camera of r26/r27 and are
now options of rwr_faive_lab's `camera_object_pose` (`mdp/camera_feedback.py`), off by default so r26/r27 are unchanged:
`tracker_filter=True` applies this node's Kalman filter and slerp to every simulated detection, reset after 5 frames
without a pose like the node (the slerp alone lags a cube turning at 2 rad/s by 3 frames, 0.3 rad), and
`symmetry_switch_prob` lets the reported orientation switch to another of the cube's 24 equivalent rotations until the
cube is lost ("the orientation can jump"). `tests/test_camera_feedback.py` checks both against the node's behaviour.
The noise and latency above are still assumptions: measure them with `measure_cube_noise.py` and turn both options on
for the next camera policy (r32, 2026-11-11, uses both, with the assumed noise). The calibration defaults of `cube_spin_v2.launch.py` were ID 9 / 7.5 cm (since 2026-11-05);
they are now the task description's tag, 36h11 ID 25 / 8 cm (a wrong size scales every measured distance).

Each policy receives three distinct detections, oldest first: `[x,y,z,qw,qx,qy,qz,age/1s]`.
Absent or expired detections are `[0,0,0,0,0,0,0,1]`. The policy keeps running on
proprioception when all detections expire, including before its first detection.
An exported `deploy_meta.json` records the observation contract and frame transform;
keep it beside `policy.pt`. Legacy exports remain supported.

Deployment with the established image estimator and a previously calibrated camera:

```bash
ros2 launch experiments cube_spin_v2.launch.py \
  policy:=/path/to/r27/exported/policy.pt \
  model_path:=/path/to/maskrcnn-finetuned_lightning_new_adamw.ckpt \
  use_saved_camera:=true dry_run:=true
```

Use `enable_hand_controller:=true dry_run:=false` for real commands once the dry run
is checked. To run without camera hardware or calibration:

```bash
ros2 launch experiments cube_spin_v2.launch.py \
  policy:=/path/to/r27/exported/policy.pt \
  enable_camera:=false dry_run:=true
```

With camera enabled, the startup calibration gate still prevents the policy from
competing with the calibration routine's hand motion. Losing cube detections after
calibration does not stop an age-aware policy. `enable_camera:=false` skips that
routine entirely. r24/r25 legacy policies still require an initial detection.

The GPU workspace is `/home/foldons/erhell/camera_feedback`. Its pipeline script
prints a plan unless passed `--start`. After approval it tests 32,768 environments
for three iterations, falls back to 16,384 only on memory/PhysX capacity errors, then
trains 2,000 + 500 iterations. The total observed GPU memory limit is 95 GiB.
It checks saved dynamics/PPO against the reference run before learning, preserves
TensorBoard/config/checkpoint logs, evaluates the reference and the new policy with
normal/full/no camera, exports TorchScript/ONNX/metadata, and records a video.

```bash
sudo bash /home/foldons/erhell/camera_feedback/rwr_faive_lab/scripts/rsl_rl/run_camera_sysid.sh --start
```

The pipeline writes its status/checkpoint paths to `logs/pipeline_*.json` and metrics
to `logs/evaluations_*.jsonl`. Judge robustness and benefit from these measurements;
the dropout training mechanism alone does not guarantee good no-camera performance.

CPU regression tests (Python with NumPy and PyTorch, both repos adjacent):

```bash
python -m unittest discover -s tests -p test_camera_feedback.py -v
```
