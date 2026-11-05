#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure the camera's cube pose as the policy receives it, and print the matching faive_lab camera model: the params of
``delayed_object_pose`` (faive_lab ``tasks/orca_inhand/mdp/observations.py``; r25 sets them in
``orca_v2_repose_reward_cfg.py``, ``CubeFramesPolicyObsCfg.object_pose``).

It records the cube pose that cube_detection_node publishes on /tf (camera frame → cube, after the detector's own
filter, as deployed) and samples it every 50 ms, one faive_lab policy step, as policy_inference_cube_spin_v2.py does.

    sim param          measured as
    pos_std            RMS per-axis spread of the position          (still recording only)
    quat_std           RMS per-axis spread of the rotation / 2      (still recording only; the sim adds this to each
                       quaternion component, which turns the cube by 2 * quat_std per axis)
    max_delay_steps    95th percentile age of the pose at a policy step + the detector's orientation filter lag
    hold_prob          share of policy steps that get no new pose, in gaps of 1-2 steps
    outage_prob        gaps of 3+ steps (cube hidden) started per policy step
    max_outage_steps   longest such gap

Two recordings, with the camera and the cube detector running
(``ros2 launch experiments cube_spin_v2.launch.py model_path:=...``), from the ``experiments`` folder:

    # 1. noise: the cube lying still in the open hand, nothing moving, 30 s
    python3 rl_inference/measure_cube_noise.py --still --duration 30

    # 2. missed detections and outages: turn the cube in the hand as the policy would, fingers covering it, 60 s
    python3 rl_inference/measure_cube_noise.py --duration 60

Each run saves the raw poses (``--save``); ``--load file.npz`` analyses a saved recording again.

The image stamp is set when the frame reaches the PC, so the camera's own latency (exposure + USB, typically 30-60 ms
for an OAK-D) is not in the measured age; add it with ``--camera-latency-ms`` if you know it.
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

STEP_S = 0.05  # faive_lab policy step: decimation 6 * sim dt 1/120 s
SHORT_GAP_MAX = 2  # gaps of up to this many policy steps count as missed detections (hold_prob), longer as outages


# ==================== Recording ====================

def record(args):
    """Record (receive time, stamp, position, quaternion xyzw) of every cube pose on /tf for args.duration seconds."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from rcl_interfaces.srv import GetParameters
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node = Node("measure_cube_noise")
    rows = []

    def on_tf(msg):
        now = node.get_clock().now().nanoseconds * 1e-9
        for tf in msg.transforms:
            if tf.child_frame_id != args.cube_frame:
                continue
            t, q = tf.transform.translation, tf.transform.rotation
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            rows.append([now, stamp, t.x, t.y, t.z, q.x, q.y, q.z, q.w])

    node.create_subscription(TFMessage, "/tf", on_tf, QoSProfile(depth=100))

    # the detector's filter settings decide the orientation lag
    filter_cfg = {"use_kalman_filter": None, "orientation_slerp_alpha": None}
    client = node.create_client(GetParameters, f"/{args.detector_node}/get_parameters")
    if client.wait_for_service(timeout_sec=3.0):
        future = client.call_async(GetParameters.Request(names=list(filter_cfg)))
        rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
        if future.result() is not None:
            values = future.result().values
            filter_cfg["use_kalman_filter"] = values[0].bool_value
            filter_cfg["orientation_slerp_alpha"] = values[1].double_value
    if filter_cfg["use_kalman_filter"] is None:
        print(f"⚠ Could not read the filter settings of /{args.detector_node}; assuming its defaults "
              "(use_kalman_filter=True, orientation_slerp_alpha=0.25)")
        filter_cfg = {"use_kalman_filter": True, "orientation_slerp_alpha": 0.25}

    print(f"Waiting for '{args.cube_frame}' on /tf (is the cube detector running and the cube in view?)...")
    wait_end = time.time() + 30.0
    while rclpy.ok() and not rows and time.time() < wait_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not rows:
        print("❌ No cube pose received in 30 s.")
        sys.exit(1)

    print(f"Recording for {args.duration:.0f} s" + (" - keep the cube and the hand still" if args.still else ""))
    end = time.time() + args.duration
    next_print = time.time() + 5.0
    try:
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.time() >= next_print:
                print(f"  {end - time.time():4.0f} s left, {len(rows)} poses")
                next_print += 5.0
    except KeyboardInterrupt:
        print("Stopped early, analysing what was recorded.")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    data = np.array(rows, dtype=np.float64)
    out = os.path.expanduser(args.save or time.strftime(
        f"~/cube_noise_{'still' if args.still else 'moving'}_%Y%m%d_%H%M%S.npz"))
    np.savez(out, data=data, still=args.still, use_kalman_filter=filter_cfg["use_kalman_filter"],
             orientation_slerp_alpha=filter_cfg["orientation_slerp_alpha"])
    print(f"Saved the raw poses to {out}")
    return data, filter_cfg


# ==================== Analysis ====================

def gap_runs(held):
    """Lengths of the runs of consecutive True in a boolean array."""
    runs, n = [], 0
    for h in held:
        if h:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return np.array(runs, dtype=int)


def analyse(data, still, filter_cfg, camera_latency_ms=0.0):
    recv, stamp, pos, quat = data[:, 0], data[:, 1], data[:, 2:5], data[:, 5:9]
    duration = recv[-1] - recv[0]
    rate = (len(recv) - 1) / duration
    result = {}

    print("\n" + "=" * 70)
    print(f"{len(recv)} cube poses over {duration:.1f} s: {rate:.1f} Hz")

    # --- the policy's view: the latest pose at every 50 ms step ---
    ticks = np.arange(recv[0] + STEP_S, recv[-1], STEP_S)
    latest = np.searchsorted(recv, ticks, side="right") - 1
    held = np.r_[False, latest[1:] == latest[:-1]]  # no new pose since the previous step
    runs = gap_runs(held)
    short, long_ = runs[runs <= SHORT_GAP_MAX], runs[runs > SHORT_GAP_MAX]
    steps_in_outage = int(long_.sum())
    result["hold_prob"] = short.sum() / len(ticks)
    result["outage_prob"] = len(long_) / max(1, len(ticks) - steps_in_outage)
    result["max_outage_steps"] = int(long_.max()) if len(long_) else None
    print(f"\nPolicy steps: {len(ticks)}, without a new pose: {held.sum()} ({held.mean():.1%})")
    print(f"  missed detections (gaps of 1-{SHORT_GAP_MAX} steps): {short.sum()} steps -> hold_prob "
          f"{result['hold_prob']:.3f}")
    if len(long_):
        print(f"  outages (gaps of {SHORT_GAP_MAX + 1}+ steps): {len(long_)}, lengths {sorted(long_.tolist())} steps "
              f"-> outage_prob {result['outage_prob']:.4f}, max_outage_steps {result['max_outage_steps']}")
    else:
        print("  no outages in this recording")

    # --- delay: age of the pose the policy gets, plus the detector's orientation smoothing ---
    latency = recv - stamp
    age = ticks - stamp[latest] + camera_latency_ms / 1000
    filter_lag = 0.0
    if filter_cfg["use_kalman_filter"]:
        # exponential smoothing of the orientation lags a steadily turning cube by (1 - alpha) / alpha poses;
        # the position filter (constant-velocity Kalman) has almost no lag
        alpha = filter_cfg["orientation_slerp_alpha"]
        filter_lag = (1 - alpha) / alpha / rate
    delay = np.percentile(age, 95) + filter_lag
    result["max_delay_steps"] = int(np.ceil(delay / STEP_S - 1e-6))
    print(f"\nDelay (image on the PC -> pose on /tf): median {np.median(latency) * 1000:.0f} ms, "
          f"95% {np.percentile(latency, 95) * 1000:.0f} ms")
    print(f"Age of the pose at a policy step: median {np.median(age) * 1000:.0f} ms, 95% "
          f"{np.percentile(age, 95) * 1000:.0f} ms" + (f" (incl. {camera_latency_ms:.0f} ms camera latency)"
                                                       if camera_latency_ms else " (camera latency not included)"))
    if filter_cfg["use_kalman_filter"]:
        print(f"Orientation smoothing (slerp alpha {alpha}): +{filter_lag * 1000:.0f} ms while the cube turns")
    print(f"  -> {delay * 1000:.0f} ms = {delay / STEP_S:.1f} steps -> max_delay_steps {result['max_delay_steps']}")

    # --- noise: spread of the pose while nothing moves (each pose once) ---
    if still:
        centre = pos.mean(axis=0)
        dev = pos - centre
        result["pos_std"] = float(np.sqrt((dev ** 2).sum(axis=1).mean() / 3))
        rot = Rotation.from_quat(quat)
        rotvec = (rot.mean().inv() * rot).as_rotvec()
        rot_axis_std = float(np.sqrt((rotvec ** 2).sum(axis=1).mean() / 3))
        result["quat_std"] = rot_axis_std / 2
        angle = np.degrees(np.linalg.norm(rotvec, axis=1))
        print(f"\nPosition spread: {result['pos_std'] * 1000:.2f} mm per axis (camera x/y/z: "
              f"{', '.join(f'{s * 1000:.2f}' for s in dev.std(axis=0))} mm; z is the depth) -> pos_std")
        print(f"Rotation spread: {np.degrees(rot_axis_std):.2f} deg per axis, mean angle error {angle.mean():.2f} deg, "
              f"95% {np.percentile(angle, 95):.2f} deg -> quat_std {result['quat_std']:.4f}")

        # a still recording should not drift; a drift means the cube or the hand moved and the spread is too large
        k = max(1, len(pos) // 5)
        drift = np.linalg.norm(pos[:k].mean(axis=0) - pos[-k:].mean(axis=0))
        turn = np.degrees((Rotation.from_quat(quat[:k]).mean().inv() * Rotation.from_quat(quat[-k:]).mean()).magnitude())
        if drift > 3 * max(result["pos_std"], 1e-4) or turn > 3 * np.degrees(rot_axis_std):
            print(f"⚠ The cube moved during the recording ({drift * 1000:.1f} mm, {turn:.1f} deg between the first and "
                  "last fifth): the noise is overestimated, record again with everything still.")
    else:
        print("\nNoise not measured: record with --still for pos_std and quat_std.")

    # --- paste-ready params ---
    fmt = {"pos_std": "{:.4f}", "quat_std": "{:.4f}", "max_delay_steps": "{}", "hold_prob": "{:.3f}",
           "outage_prob": "{:.4f}", "max_outage_steps": "{}"}
    print("\n" + "=" * 70)
    print("delayed_object_pose params (measured in this recording; keep the current value for the others):")
    for key, f in fmt.items():
        if result.get(key) is None:
            note = "not measured here" if key in ("pos_std", "quat_std") else "no outage seen, measure while moving"
            print(f'    # "{key}": ...,  {note}')
        else:
            print(f'    "{key}": {f.format(result[key])},')
    if still:
        print("  (a still recording sees few outages: take hold/outage values from a moving recording)")
    print("=" * 70)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--still", action="store_true", help="the cube lies still: also measure the noise")
    parser.add_argument("--duration", type=float, default=30.0, help="recording length in seconds (default 30)")
    parser.add_argument("--cube-frame", default="cube", help="child frame of the cube pose on /tf (default cube)")
    parser.add_argument("--detector-node", default="cube_detection_node",
                        help="name of the cube detector node, to read its filter settings")
    parser.add_argument("--camera-latency-ms", type=float, default=0.0,
                        help="camera latency before the image reaches the PC, added to the delay (default 0)")
    parser.add_argument("--save", default=None, help="where to save the raw poses (default ~/cube_noise_<time>.npz)")
    parser.add_argument("--load", default=None, help="analyse a saved .npz instead of recording")
    args = parser.parse_args()

    if args.load:
        saved = np.load(os.path.expanduser(args.load))
        data = saved["data"]
        still = bool(saved["still"]) or args.still
        filter_cfg = {"use_kalman_filter": bool(saved["use_kalman_filter"]),
                      "orientation_slerp_alpha": float(saved["orientation_slerp_alpha"])}
    else:
        data, filter_cfg = record(args)
        still = args.still
    if len(data) < 20:
        print(f"❌ Only {len(data)} poses, too few to analyse.")
        sys.exit(1)
    analyse(data, still, filter_cfg, args.camera_latency_ms)


if __name__ == "__main__":
    main()
