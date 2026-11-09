"""Collect ORCA v2 joint chirp data through orca_core, without ROS.

By default, only validate and print the experiment plan. Pass --execute to
connect to a previously calibrated hand and collect real measurements.

While one joint is swept, the others hold the hand's neutral pose, except that the other fingers are spread away from
the swept one (--spread-deg, default 20; 0 = plain neutral pose): fingers on its thumb side to +spread, fingers on its
pinky side to -spread, and its own abduction to 0 while its MCP or PIP is swept. For a thumb joint all four fingers
go to -spread (away from the thumb). The other joints move to this pose first, and only then the swept joint to the
start of the chirp, so it always has room to move. The pose of every sweep is saved in metadata.json
(hold_poses_deg), and faive_lab's chirp_characterization.py holds the simulated hand the same way.
"""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy import signal
import yaml


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "src/hand_control/orca/configs/orcahand_v2/right/config.yaml"
)
CSV_FIELDS = ["timestamp", "commanded_position", "measured_position"]
# fingers from the thumb side to the pinky side. On the ORCA v2 a positive *_abd turns a finger towards the thumb side
# (orcahand_v2.urdf.xacro; the ring_abd re-recording of 2026-11-08 moved the neighbours away the same way)
FINGERS = ("index", "middle", "ring", "pinky")


def hold_pose(joint, neutral, spread_deg):
    """Pose of all joints (degrees) while `joint` is swept: neutral, with the other fingers spread away from it."""
    pose = dict(neutral)
    if spread_deg == 0:
        return pose
    finger, _, part = joint.partition("_")
    if finger in FINGERS:
        k = FINGERS.index(finger)
        for i, other in enumerate(FINGERS):
            if other != finger:
                pose[f"{other}_abd"] = float(spread_deg if i < k else -spread_deg)
        if part != "abd":
            pose[f"{finger}_abd"] = 0.0  # straight, so its flexion is not pressed against a neighbour
    elif finger == "thumb":
        for other in FINGERS:
            pose[f"{other}_abd"] = float(-spread_deg)
    return pose


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="This hand's config.yaml; calibration.yaml must be beside it.")
    parser.add_argument("--joints", nargs="+", default=["index_mcp"],
                        help="Joint names to sweep sequentially (ORCA v2 uses thumb_cmc).")
    parser.add_argument("--ctrl-freq", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--f0", type=float, default=0.1)
    parser.add_argument("--f1", type=float, default=5.0)
    parser.add_argument("--amplitude-deg", type=float, default=10.0)
    parser.add_argument("--center-deg", type=float, default=None,
                        help="Sweep center; by default use each joint's ROM midpoint.")
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--spread-deg", type=float, default=20.0,
                        help="Abduction of the other fingers away from the swept one (0: neutral pose).")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/chirp_v2"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Connect and move real hardware.")
    mode.add_argument("--dry-run", action="store_true", help="Print the plan only (the default).")
    return parser


def make_plan(args, config):
    """Reject invalid or clipped experiments before constructing hardware."""
    for name in ("ctrl_freq", "duration", "f0", "f1", "amplitude_deg"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.settle_time) or args.settle_time < 0:
        raise ValueError("settle_time must be finite and non-negative")
    if not math.isfinite(args.spread_deg) or args.spread_deg < 0:
        raise ValueError("spread_deg must be finite and non-negative")
    if not args.f0 < args.f1 < args.ctrl_freq / 2:
        raise ValueError("Require 0 < f0 < f1 < ctrl_freq / 2 (Nyquist frequency)")
    if args.duration * args.ctrl_freq < 2:
        raise ValueError("The experiment must contain at least two samples")
    if len(set(args.joints)) != len(args.joints):
        raise ValueError("Do not repeat joints in --joints")

    joint_ids = config["joint_ids"]
    if not joint_ids or len(set(joint_ids)) != len(joint_ids):
        raise ValueError("config joint_ids must be non-empty and unique")
    unknown = set(args.joints) - set(joint_ids)
    if unknown:
        raise ValueError(f"Unknown joints: {sorted(unknown)}. Available: {joint_ids}")
    neutral = config["neutral_position"]
    roms = config["joint_roms"]
    for joint in joint_ids:
        lower, upper = roms[joint]
        position = neutral[joint]
        if not all(math.isfinite(x) for x in (lower, upper, position)):
            raise ValueError(f"Non-finite ROM or neutral position for {joint}")
        if lower >= upper or not lower <= position <= upper:
            raise ValueError(f"Invalid ROM or neutral position for {joint}")

    centers = {}
    for joint in args.joints:
        lower, upper = roms[joint]
        center = (lower + upper) / 2 if args.center_deg is None else args.center_deg
        if (not math.isfinite(center)
                or center - args.amplitude_deg < lower
                or center + args.amplitude_deg > upper):
            raise ValueError(f"{joint}: center +/- amplitude exceeds ROM [{lower}, {upper}]")
        centers[joint] = center
    hold_poses = {}
    for joint in args.joints:
        pose = hold_pose(joint, neutral, args.spread_deg)
        for other, value in pose.items():
            lower, upper = roms[other]
            if other != joint and not lower <= value <= upper:
                raise ValueError(f"{joint}: holding {other} at {value} deg exceeds its ROM [{lower}, {upper}]; "
                                 "use a smaller --spread-deg")
        hold_poses[joint] = pose
    return {
        "config_path": str(args.config.resolve()),
        "config": config,
        "joints": args.joints,
        "centers_deg": centers,
        "ctrl_freq_hz": args.ctrl_freq,
        "duration_s": args.duration,
        "f0_hz": args.f0,
        "f1_hz": args.f1,
        "amplitude_deg": args.amplitude_deg,
        "settle_time_s": args.settle_time,
        "sample_count_per_joint": len(sample_times(args)),
        "spread_deg": args.spread_deg,
        "hold_poses_deg": hold_poses,
    }


def sample_times(args):
    return np.arange(0, args.duration, 1.0 / args.ctrl_freq)


def collect_joint(hand, args, plan, joint, path, *, current=None, clock=time.monotonic, sleep=time.sleep):
    """Write each real sample immediately, preserving partial data on failure.

    current is the last command sent (default: the neutral pose); returns the last command of this sweep.
    """
    t = sample_times(args)
    trajectory = plan["centers_deg"][joint] + args.amplitude_deg * signal.chirp(
        t, f0=args.f0, t1=args.duration, f1=args.f1, method="logarithmic"
    )
    # Ramp for roughly one second rather than jumping, in two steps: first the other joints to the hold pose (the
    # neighbouring fingers spread away) with the swept joint where it is, then the swept joint to the first chirp sample.
    ramp = dict(num_steps=max(2, math.ceil(args.ctrl_freq)), step_size=1.0 / args.ctrl_freq)
    command = dict(plan["hold_poses_deg"][joint])
    command[joint] = (current or plan["config"]["neutral_position"])[joint]
    hand.set_joint_positions(command, **ramp)
    sleep(args.settle_time)
    command[joint] = float(trajectory[0])
    hand.set_joint_positions(command, **ramp)
    sleep(args.settle_time)

    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_FIELDS)
        stream.flush()
        start = clock()
        for nominal_time, position in zip(t, trajectory):
            deadline = start + float(nominal_time)
            sleep(max(0.0, deadline - clock()))
            # Catching up in a burst would invalidate the intended excitation.
            if clock() - deadline >= 1.0 / args.ctrl_freq:
                raise RuntimeError("Missed a full control period; stopped instead of sending a burst. "
                                   "Check bus latency and the partial CSV timestamps.")
            command[joint] = float(position)
            hand.set_joint_positions(command)
            measured = hand.get_joint_position().as_dict().get(joint)
            if measured is None or not math.isfinite(measured):
                raise RuntimeError(f"No finite position measurement for {joint}")
            writer.writerow([clock() - start, float(position), float(measured)])
            stream.flush()
        # Hold the last sample for a full period before changing pose or disabling torque.
        sleep(max(0.0, start + len(t) / args.ctrl_freq - clock()))
    return command


def run_hardware(args, plan, *, factory=None, clock=time.monotonic, sleep=time.sleep):
    if factory is None:
        # Preview/help work without orca_core, ROS or any serial access.
        from orca_core import load_hand
        factory = load_hand

    hand = factory(config_path=str(args.config.resolve()))
    metadata = dict(plan, status="starting", csv_files=[])
    run_dir = None
    try:
        # init_joints() can auto-calibrate. Refuse to enter it without a complete
        # existing calibration: calibration is a separate, deliberate operation.
        if not hand.calibrated or not hand.is_calibrated():
            raise RuntimeError("A complete calibration.yaml for this hand is required beside config.yaml. "
                               "Calibrate separately before running chirp.")
        ok, message = hand.connect(interactive=False)
        if not ok:
            raise RuntimeError(f"Could not connect: {message}")
        print(message)

        started_at = datetime.now(timezone.utc)
        stamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
        run_dir = args.output_dir / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata.update(status="running", orca_core_version=version("orca_core"),
                        run_id=stamp, started_at_utc=started_at.isoformat())
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        hand.init_joints(force_calibrate=False, move_to_neutral=True)
        current = None
        for joint in args.joints:
            filename = (f"chirp_{joint}_{stamp}_ctrl{args.ctrl_freq:g}Hz_"
                        f"{args.duration}s_{args.f0}-{args.f1}Hz_{args.amplitude_deg:g}deg.csv")
            path = run_dir / filename
            metadata["csv_files"].append(filename)
            print(f"Collecting {joint}: {path}")
            current = collect_joint(hand, args, plan, joint, path, current=current, clock=clock, sleep=sleep)
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Always attempt both operations, including if connect/init fails.
        active_exception = sys.exc_info()[0] is not None
        cleanup_errors = []
        operations = []
        if hand.is_connected():
            operations.append(("disable torque", hand.disable_torque))
        operations.append(("disconnect", hand.disconnect))
        for label, operation in operations:
            try:
                result = operation()
                if label == "disable torque" and result:
                    raise RuntimeError(f"Torque disable not acknowledged by motors {result}")
                if label == "disconnect" and result is not None and not result[0]:
                    raise RuntimeError(result[1])
            except Exception as exc:
                cleanup_errors.append(str(exc))
                print(f"Hardware cleanup ({label}) failed: {exc}", file=sys.stderr)
        if cleanup_errors:
            metadata.update(status="failed", cleanup_errors=cleanup_errors)
        if run_dir is not None:
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Experiment {metadata['status']}; data: {run_dir}")
        if cleanup_errors and not active_exception:
            raise RuntimeError("Hardware cleanup failed; check the hand connection")
    return run_dir


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        with args.config.open() as stream:
            config = yaml.safe_load(stream)
        plan = make_plan(args, config)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in plan.items() if key != "config"}, indent=2))
    if not args.execute:
        print("Preview only. Add --execute to connect and move the calibrated hand.")
        return 0
    try:
        run_hardware(args, plan)
    except KeyboardInterrupt:
        print("Stopped all sweeps; partial CSV data has been kept.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Chirp failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
