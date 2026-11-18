"""Calibrate only the wrist of the ORCA v2 hand by hand (orca_core's manual capture mode).

Why (2026-11-15): orca_core's normal calibration drives the wrist onto its two hardstops with the configured current.
With the palm up, flexing the wrist (fingertips towards the palm side) lifts the whole hand, the wrist stalls before
its stop, and the stall is recorded as the hardstop: three calibrations that day gave 142, 93 and 54 deg of wrist motor
travel, and commanded wrist angles did not match the real ones. Manual mode keeps the torque OFF; you move the wrist to
each hardstop by hand and press Enter, and orca_core records the limits there (same routine, same calibration.yaml).
Unlike the driven mode, orca_core's manual mode does not re-zero the Feetech position frame (INST_OFSCAL), so a frame
left by an earlier, stalled run can put the real hardstop outside the commandable raw range 0-4095 (15:47 run: palm
stop at raw 4515, commands above ~+6 deg would be clipped). This script therefore re-zeroes the frame at the first
hardstop exactly as the driven mode does (orca_core calibrate_offset with the routine's own direction sign: that pose
reads raw 3595 or 500) before it is recorded, and checks at the end that both limits lie inside 0-4095.

Usage (hand controller stopped: this needs the serial port; the fingers' calibration is not touched):
    python calibrate_wrist_manual.py ~/ros_ws/install/hand_control/share/hand_control/orca/configs/orcahand_v2/right/config.yaml
Pass the ROS install path: calibration.yaml lies next to that symlink and is what the hand controller reads.
--mock runs the same flow on orca_core's mock hand (nothing is written).
"""

import argparse
import sys
from pathlib import Path

DIRECTION_TEXT = {
    "flex": "FLEX = fingertips towards the PALM side, as far as the wrist goes",
    "extend": "EXTEND = fingertips towards the BACK of the hand, as far as the wrist goes",
}


def make_prompt(hand):
    """Prompt callback: the operator holds the wrist on its hardstop and presses Enter; at the first recorded hardstop
    the Feetech frame is re-zeroed there first (as orca_core's driven calibration does), then orca_core reads and
    records the position. Enter records, s skips, a aborts."""
    from orca_core.maintenance.calibration_routine import _direction_sign
    state = {"offset_done": False}

    def prompt(event: dict) -> str:
        joint, direction = event.get("joint"), event.get("direction")
        text = DIRECTION_TEXT.get(direction, direction)
        print(f"\n>>> Move the {joint} by hand to its {text}.")
        print("    Hold it there (gently against the stop) and press Enter to record   [s = skip, a = abort]")
        answer = {"": "record", "s": "skip", "a": "abort"}.get(input("    > ").strip().lower()[:1], "record")
        client = hand.motor_client
        if answer == "record" and not state["offset_done"] and getattr(client, "requires_offset_calibration", False):
            motor_id = hand.config.joint_to_motor_map[joint]
            upper = _direction_sign(hand, joint, direction) > 0
            if not client.calibrate_offset(motor_id, upper=upper):
                print("    ERROR: the position frame could not be re-zeroed here; aborting (nothing recorded)")
                return "abort"
            state["offset_done"] = True
            print(f"    position frame re-zeroed at this {direction} hardstop (motor {motor_id}, as the driven calibration does)")
        return answer
    return prompt


def check_raw_range(hand):
    """Both wrist limits must lie inside the raw range commands are clipped to (0-4095), else +angles get clipped."""
    from orca_core.hardware.feetech_client import DEFAULT_POS_SCALE, POSITION_DIRECTION
    motor_id = hand.config.joint_to_motor_map["wrist"]
    limits = hand.calibration.motor_limits_dict.get(motor_id)
    if not limits or None in limits:
        print("WARNING: no wrist limits recorded")
        return
    raw = [round(v * POSITION_DIRECTION / DEFAULT_POS_SCALE) for v in limits]
    ok = all(0 <= r <= 4095 for r in raw)
    print(f"wrist limits {limits[0]:.4f} / {limits[1]:.4f} rad = raw {raw[0]} / {raw[1]}: "
          + ("inside 0-4095, OK" if ok else "OUTSIDE 0-4095: wrist commands would be clipped, run this again"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_path", type=Path, help="the hand controller's config.yaml (ROS install path)")
    parser.add_argument("--mock", action="store_true", help="dry run on orca_core's mock hand")
    args = parser.parse_args()

    from orca_core import load_hand
    from orca_core.utils.cli import connect_hand, print_calibration_progress, shutdown_hand

    # absolute(), not resolve(): keep the symlink's folder, where calibration.yaml lies
    config = str(args.config_path.expanduser().absolute())
    if args.mock:
        from orca_core.hardware_hand import MockOrcaHand
        hand = MockOrcaHand(config_path=config)
    else:
        hand = load_hand(config_path=config)
    print(f"config: {config}\ncalibration file: {hand.config.calibration_path}")
    if not args.mock and (not hand.calibrated or not hand.is_calibrated()):
        sys.exit("The fingers have no complete calibration.yaml next to this config: run the full calibration first.")
    try:
        connect_hand(hand)
        print("Torque stays OFF for the whole run. The wrist goes limp: support the hand.")
        hand.calibrate(force_wrist=True, joints=["wrist"], manual=True, prompt_callback=make_prompt(hand),
                       progress_callback=print_calibration_progress)
        check_raw_range(hand)
    except KeyboardInterrupt:
        print("\nInterrupted: run it again before starting the hand controller.")
    finally:
        shutdown_hand(hand)


if __name__ == "__main__":
    main()
