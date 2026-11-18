"""ORCA v2 hand controller. Drives the hand through the orca_core package,
exposing the interface hand_control_node.py expects."""
import numpy as np
from orca_core import load_hand

# Position-loop gain KP (Feetech register 50, RAM, loaded from EEPROM at power-up; the motors' own value is 32, read
# 2026-11-15). orca_core never writes it, so a value set here stays until it is set again or the hand is power-cycled:
# the policy node sets the model's KP before every run and 32 after it. 48 = 1.5x (the sysID recording of 2026-11-15).
# Anything outside this range is refused (a safety net: a higher KP needs a deliberate change here).
DEFAULT_SERVO_KP = 32
MIN_SERVO_KP, MAX_SERVO_KP = 16, 64
# Goal-current limit (register 44, RAM, 6.5 mA units, orca_core rounds a requested value down): 900 mA = orca_core's
# default (config.yaml max_current: default), set at every controller start; 1000 mA = r104 (user 2026-11-15). The
# policy node sets its model's limit before a run and 900 after it. Requests outside this range are refused (a safety
# net: a higher limit needs a deliberate change here); each motor also caps it at its protection current (EEPROM 28).
DEFAULT_MAX_CURRENT_MA = 900.0
MIN_MAX_CURRENT_MA, MAX_MAX_CURRENT_MA = 300.0, 1000.0


class OrcaHandV2:
    def __init__(self, config_path: str):
        # An explicit config_path skips orca_core's hardware autodetection.
        # calibration.yaml is read from and saved next to config_path.
        self._hand = load_hand(config_path=config_path)
        config = self._hand.config
        self.joint_ids = list(config.joint_ids)
        self.num_motors = len(self.joint_ids)
        joint_index = {joint: i for i, joint in enumerate(self.joint_ids)}
        # get_motor_state() arrays are in motor_ids order; these map them to joint_ids order.
        self._joint_order = np.empty(self.num_motors, dtype=np.intp)
        self._joint_signs = np.ones(self.num_motors)
        for motor_idx, motor_id in enumerate(config.motor_ids):
            joint = config.motor_to_joint_dict[motor_id]
            self._joint_order[joint_index[joint]] = motor_idx
            if config.joint_inversion_dict.get(joint, False):
                self._joint_signs[joint_index[joint]] = -1.0

    def init_joints(self, calibrate: bool = False, initialize: bool = True, compliant_test_mode: bool = False):
        ok, message = self._hand.connect(interactive=False)
        if not ok:
            raise RuntimeError(f"Could not connect to the ORCA hand: {message}")
        # Calibrates automatically when no calibration exists; moves to the config's neutral pose.
        self._hand.init_joints(force_calibrate=calibrate, move_to_neutral=initialize)
        if compliant_test_mode:
            self.disable_torque()

    def set_joint_pos_array(self, joint_pos_deg):
        """Command joint positions in joint_ids order, in degrees (clamped to the ROMs by orca_core)."""
        self._hand.set_joint_positions(np.asarray(joint_pos_deg, dtype=np.float64))

    def read_joint_state(self):
        """One bus read -> (positions [deg], velocities [deg/s]) in joint_ids order."""
        state = self._hand.get_motor_state()
        joint_pos = self._hand._motor_to_joint_pos(state.position)
        pos = np.array([np.nan if joint_pos[j] is None else joint_pos[j] for j in self.joint_ids])
        ratios = self._hand.calibration.joint_to_motor_ratios_dict  # motor rad per joint deg
        motor_ids = self._hand.config.motor_ids
        motor_ratios = np.array([ratios[motor_ids[i]] or np.nan for i in self._joint_order])
        vel = self._joint_signs * np.asarray(state.velocity)[self._joint_order] / motor_ratios
        return pos, vel

    def _require_feetech(self):
        # ACC values are Feetech HLS register units (8.7 deg/s^2 each), what the sim's servo ramp models
        if self._hand.config.motor_type != "feetech":
            raise RuntimeError(f"servo ACC is only defined for Feetech motors, this hand has {self._hand.config.motor_type}")

    def get_servo_acc(self):
        """ACC register (0..254, 0 = no ramp) of each joint's motor, in joint_ids order; nan where the read failed."""
        from orca_core.hardware.feetech_registers import HLS
        self._require_feetech()
        profiles = self._hand.get_servo_profile()
        out = np.full(self.num_motors, np.nan)
        for j, joint in enumerate(self.joint_ids):
            p = profiles.get(self._hand.config.joint_to_motor_map[joint])
            if p is not None and p.acceleration_rad_s2 is not None:
                out[j] = round(p.acceleration_rad_s2 / HLS.ACC_SCALE_RAD_S2)
        return out

    def set_servo_acc(self, acc_raw):
        """Set the ACC register (0..254) of each joint's motor, in joint_ids order. RAM register: a restart of the
        controller sets orca_core's default (150) again."""
        from orca_core.hardware.feetech_registers import HLS
        from orca_core.hardware.motor_client import ServoProfile
        self._require_feetech()
        acc_raw = np.asarray(acc_raw, dtype=np.float64)
        if acc_raw.shape != (self.num_motors,) or not np.all((acc_raw >= 0) & (acc_raw <= HLS.ACC_MAX_RAW)):
            raise ValueError(f"expected {self.num_motors} ACC values in 0..{HLS.ACC_MAX_RAW}, got {acc_raw.tolist()}")
        self._hand.set_servo_profile({
            self._hand.config.joint_to_motor_map[joint]: ServoProfile(acceleration_rad_s2=round(a) * HLS.ACC_SCALE_RAD_S2)
            for joint, a in zip(self.joint_ids, acc_raw)})

    def get_servo_kp(self):
        """Position-loop gain KP (register 50) of each joint's motor, read back, in joint_ids order; nan where the read
        failed."""
        self._require_feetech()
        gains = self._hand.get_servo_gains()
        out = np.full(self.num_motors, np.nan)
        for j, joint in enumerate(self.joint_ids):
            g = gains.get(self._hand.config.joint_to_motor_map[joint])
            if g is not None and g.kp is not None:
                out[j] = g.kp
        return out

    def set_servo_kp(self, kp):
        """Set KP (register 50, integer) of each joint's motor, in joint_ids order. KD and KI stay as they are."""
        from orca_core.hardware.motor_client import ServoGains
        self._require_feetech()
        kp = np.asarray(kp, dtype=np.float64)
        if kp.shape != (self.num_motors,) or not np.all((kp >= MIN_SERVO_KP) & (kp <= MAX_SERVO_KP) & (kp == np.round(kp))):
            raise ValueError(f"expected {self.num_motors} integer KP values in {MIN_SERVO_KP}..{MAX_SERVO_KP}, got {kp.tolist()}")
        self._hand.set_servo_gains({
            self._hand.config.joint_to_motor_map[joint]: ServoGains(kp=int(v)) for joint, v in zip(self.joint_ids, kp)})

    def get_max_current_ma(self):
        """Goal-current limit (register 44) of each joint's motor, read back from the motor, in mA (6.5 mA per unit),
        in joint_ids order; nan where the read failed."""
        from orca_core.hardware.feetech_registers import HLS
        self._require_feetech()
        client = self._hand._motor_client
        out = np.full(self.num_motors, np.nan)
        with self._hand._motor_lock, client._bus_lock:
            for j, joint in enumerate(self.joint_ids):
                raw = client._read_word(self._hand.config.joint_to_motor_map[joint], HLS.GOAL_CURRENT, "goal current")
                if raw is not None:
                    out[j] = client.packet_handler.scs_tohost(raw, 15) * HLS.CURRENT_SCALE_MA
        return out

    def set_max_current_ma(self, current_ma):
        """Set the goal-current limit of each joint's motor (mA requested, joint_ids order) through orca_core, which
        rounds down to 6.5 mA units and clamps to each motor's protection current. A controller restart sets
        config.yaml's max_current (default 900 mA) again."""
        self._require_feetech()
        current_ma = np.asarray(current_ma, dtype=np.float64)
        if current_ma.shape != (self.num_motors,) or not np.all((current_ma >= MIN_MAX_CURRENT_MA) & (current_ma <= MAX_MAX_CURRENT_MA)):
            raise ValueError(f"expected {self.num_motors} current limits in {MIN_MAX_CURRENT_MA:g}..{MAX_MAX_CURRENT_MA:g} mA, "
                             f"got {current_ma.tolist()}")
        by_motor = {self._hand.config.joint_to_motor_map[j]: float(v) for j, v in zip(self.joint_ids, current_ma)}
        self._hand.set_max_current([by_motor[m] for m in self._hand.config.motor_ids])

    def enable_torque(self):
        self._hand.enable_torque()

    def disable_torque(self):
        self._hand.disable_torque()

    def disconnect(self):
        self._hand.disconnect()
