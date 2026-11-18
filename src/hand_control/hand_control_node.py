#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import numpy as np
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool
import os
from ament_index_python.packages import get_package_share_directory


class HandControllerNode(Node):
    def __init__(self, debug=False):
        super().__init__("hand_controller_node")

        # Determine which hand controller to use
        self.declare_parameter("hand_controller/hand_name", "orca_v2")
        self.declare_parameter("hand_controller/hand_chirality", "right")
        self.declare_parameter("hand_controller/calibrate", False)
        self.declare_parameter('hand_controller/compliant_test_mode', False)

        self.hand_name = self.get_parameter("hand_controller/hand_name").value
        self.chirality = self.get_parameter("hand_controller/hand_chirality").value
        self.calibrate = self.get_parameter("hand_controller/calibrate").value
        self.compliant_test_mode = self.get_parameter('hand_controller/compliant_test_mode').value
        self.get_logger().info(f"Hand Controller Node Started with hand: {self.hand_name}")
        self.get_logger().info(f"Hand Controller Node Started with chirality: {self.chirality}")

        # Franka gripper
        if self.hand_name == "franka_hand":
            from franka_gripper.hand_controller import HandController as FrankaGripperHandController
            self._hc = FrankaGripperHandController(hand_chirality = "right" if self.get_parameter("hand_controller/hand_chirality").value == "right" else "left")

        # Orca v2 hand, driven through the orca_core package
        elif self.hand_name == "orca_v2":
            self.declare_parameter('hand_controller/hand_cfg_name', 'orcahand_v2')
            hand_cfg_name = self.get_parameter('hand_controller/hand_cfg_name').value

            config_dir_path = os.path.join(
                get_package_share_directory("hand_control"),
                "orca",
                "configs",
                hand_cfg_name
            )
            # publisher for sensor values (in radians)
            self.joint_angle_pub = self.create_publisher(Float32MultiArray, f'/{self.chirality}/hand/joint_position_from_motor', 10)
            self.joint_velocity_pub = self.create_publisher(Float32MultiArray, f'/{self.chirality}/hand/joint_velocity_from_motor', 10)

            from faive_system.src.hand_control.orca.hand_controller_v2 import OrcaHandV2
            # orca_core requires the file name config.yaml; calibration.yaml is stored next to it
            self._hc = OrcaHandV2(os.path.join(config_dir_path, self.chirality, "config.yaml"))
            self._hc.init_joints(calibrate=self.calibrate, initialize=True, compliant_test_mode=self.compliant_test_mode)
            self._orca_timer = self.create_timer(
                0.01, # seconds
                self.orca_v2_control_cb,  # callback
            )
            # servo acceleration (ACC register 0..254 per joint, joint_ids order of config.yaml). The policy node sets
            # the profile its model was trained with (deploy_meta.json servo_profile, r60+) and sets 150 (orca_core's
            # default, which this node starts with) for all other models. /servo_acc holds the values read back from
            # the motors (latched)
            self.servo_acc_pub = self.create_publisher(
                Float32MultiArray, f"/{self.chirality}/hand/servo_acc",
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
            self.servo_acc_sub = self.create_subscription(
                Float32MultiArray, f"/{self.chirality}/hand/set_servo_acc", self.set_servo_acc_cb, 10)
            self.publish_servo_acc()
            # servo position-loop gain KP (register 50, 2026-11-15). orca_core never writes it, so a value set by a run
            # stays until set again or a power cycle: the policy node sets its model's KP (deploy_meta.json
            # servo_profile.kp, 32 = the motors' own when missing) before every run and 32 after it. /servo_kp holds the
            # values read back from the motors (latched); logged here at startup so a left-over value is visible
            self.servo_kp_pub = self.create_publisher(
                Float32MultiArray, f"/{self.chirality}/hand/servo_kp",
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
            self.servo_kp_sub = self.create_subscription(
                Float32MultiArray, f"/{self.chirality}/hand/set_servo_kp", self.set_servo_kp_cb, 10)
            kp = self.publish_servo_kp()
            self.get_logger().info("servo KP at start: " + self.kp_text(kp))
            # motor goal-current limit (register 44, mA, 2026-11-15): orca_core sets config.yaml's max_current (900 mA)
            # at start; the policy node sets its model's limit (deploy_meta.json servo_profile.max_current_ma, 900 when
            # missing) before every run and 900 after it. /max_current_ma: read back from the motors (latched)
            self.max_current_pub = self.create_publisher(
                Float32MultiArray, f"/{self.chirality}/hand/max_current_ma",
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
            self.max_current_sub = self.create_subscription(
                Float32MultiArray, f"/{self.chirality}/hand/set_max_current_ma", self.set_max_current_cb, 10)
            cur = self.publish_max_current()
            self.get_logger().info("motor current limit at start (mA): " + self.ma_text(cur))

        else:
            raise NotImplementedError("Hand name not recognized")

        self.joint_angle_sub = self.create_subscription(
            Float32MultiArray, f"/{self.chirality}/hand/policy_output", self.joint_angle_cb, 0
        )
        
        # Add emergency stop service, kills itself when value is True
        self.estop_service = self.create_service(
            SetBool, f"/{self.chirality}/hand/estop", self.faive_estop_cb
        )

        # Service to enable/disable torque on the underlying hand controller.
        self.torque_service = self.create_service(
            SetBool,
            f"/{self.chirality}/hand/set_torque_enabled",
            self.set_torque_enabled_cb,
        )

    def joint_angle_cb(self, msg):
        num_motors = self._hc.num_motors
        if len(msg.data) != num_motors:
            self.get_logger().error(f"Expected {num_motors} joint angles, got {len(msg.data)}")
            return
        joint_angles = np.array(msg.data)
        joint_angles = np.rad2deg(joint_angles)
        if self.hand_name == "orca_v2":
            self._hc.set_joint_pos_array(joint_angles)
        else:
            self._hc.command_joint_angles(joint_angles)


    def orca_v2_control_cb(self):
        """
        publish joint angles and velocities from the orca_core hand to ROS
        """
        q_deg, qd_deg = self._hc.read_joint_state()
        self.joint_angle_pub.publish(Float32MultiArray(data=np.deg2rad(q_deg).astype(np.float32)))
        self.joint_velocity_pub.publish(Float32MultiArray(data=np.deg2rad(qd_deg).astype(np.float32)))


    def publish_servo_acc(self):
        try:
            acc = self._hc.get_servo_acc()
        except Exception as e:
            self.get_logger().error(f"Reading the servo acceleration failed: {e}")
            acc = np.full(self._hc.num_motors, np.nan)
        self.servo_acc_pub.publish(Float32MultiArray(data=acc.astype(np.float32).tolist()))
        return acc

    def set_servo_acc_cb(self, msg):
        acc = np.array(msg.data, dtype=np.float64)
        try:
            self._hc.set_servo_acc(acc)
        except Exception as e:
            self.get_logger().error(f"Setting the servo acceleration failed: {e}")
        now = self.publish_servo_acc()
        self.get_logger().info("servo ACC: " + " ".join(
            f"{j}={'?' if np.isnan(a) else int(a)}" for j, a in zip(self._hc.joint_ids, now)))

    def kp_text(self, kp):
        return " ".join(f"{j}={'?' if np.isnan(k) else int(k)}" for j, k in zip(self._hc.joint_ids, kp))

    def publish_servo_kp(self):
        try:
            kp = self._hc.get_servo_kp()
        except Exception as e:
            self.get_logger().error(f"Reading the servo KP failed: {e}")
            kp = np.full(self._hc.num_motors, np.nan)
        self.servo_kp_pub.publish(Float32MultiArray(data=kp.astype(np.float32).tolist()))
        return kp

    def set_servo_kp_cb(self, msg):
        kp = np.array(msg.data, dtype=np.float64)
        try:
            self._hc.set_servo_kp(kp)
        except Exception as e:
            self.get_logger().error(f"Setting the servo KP failed: {e}")
        self.get_logger().info("servo KP: " + self.kp_text(self.publish_servo_kp()))

    def ma_text(self, ma):
        return " ".join(f"{j}={'?' if np.isnan(m) else f'{m:g}'}" for j, m in zip(self._hc.joint_ids, ma))

    def publish_max_current(self):
        try:
            cur = self._hc.get_max_current_ma()
        except Exception as e:
            self.get_logger().error(f"Reading the motor current limit failed: {e}")
            cur = np.full(self._hc.num_motors, np.nan)
        self.max_current_pub.publish(Float32MultiArray(data=cur.astype(np.float32).tolist()))
        return cur

    def set_max_current_cb(self, msg):
        cur = np.array(msg.data, dtype=np.float64)
        try:
            self._hc.set_max_current_ma(cur)
        except Exception as e:
            self.get_logger().error(f"Setting the motor current limit failed: {e}")
        self.get_logger().info("motor current limit (mA): " + self.ma_text(self.publish_max_current()))

    def set_torque_enabled_cb(self, request, response):
        """
        Enable or disable torque on the underlying hand controller via a SetBool service.
        """
        try:
            if request.data:
                self._hc.enable_torque()
                self.get_logger().info("Torque enabled via service")
                response.success = True
                response.message = "Torque enabled"
            else:
                self._hc.disable_torque()
                self.get_logger().info("Torque disabled via service")
                response.success = True
                response.message = "Torque disabled"
        except Exception as e:
            self.get_logger().error(f"Failed to set torque: {e}")
            response.success = False
            response.message = str(e)
        return response

    def faive_estop_cb(self, request, response):
        if request.data:
            self.get_logger().warn("Emergency stop received, shutting down")
            if self.hand_name == "orca_v2":
                # stop reading and commanding before the bus goes away. Without this, an e-stop crashes the node
                if not self._orca_timer.is_canceled():
                    self._orca_timer.cancel()
                    self.destroy_subscription(self.joint_angle_sub)
                    self.destroy_subscription(self.servo_acc_sub)
                    self.destroy_subscription(self.servo_kp_sub)
                    self.destroy_subscription(self.max_current_sub)
                    self._hc.disconnect()
            else:
                self._hc.disable_torque()
                self._hc.disconnect_from_dynamixels()

            response.success = True
            response.message = "Emergency stop executed"
            return response
        else:
            response.success = False
            response.message = "contents not True, Emergency stop not executed"
            return response

def main(args=None):
    rclpy.init(args=args)
    node = HandControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
