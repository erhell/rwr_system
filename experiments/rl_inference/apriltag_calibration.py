#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag Camera Calibration - Computes camera-to-tag transformation using AprilTag detection

This script:
1. Subscribes to camera images and intrinsics
2. Detects AprilTag markers in the camera view
3. Estimates camera pose relative to the tag (cam_T_tag)
4. Averages multiple samples for robustness
5. Saves cam_T_tag to file for use in cube_repose_inference_node.py

TRANSFORMATION OUTPUT:
=====================
    camera → tag (detected via AprilTag pose estimation)
    ───────────────────────────────────────
    cam_T_tag (saved to file)

The ROS node will handle world_T_tag transformation separately.

With --publish-tf (how cube_spin_v2.launch.py starts it, at every launch), the result is also published as the static
TF tag_frame → <camera_name>_frame and the node keeps running, so the camera position is measured anew at every start
and never comes from an old file. Until the tag is found there is no cube pose in the hand frame.

The open hand hides the tag, so in that mode the node first closes the hand (orca_core's power_grasp pose, wrist
unchanged; --fist / --no-fist, --fist-pose) through the hand controller's topics, measures, opens the hand back to where
it started, and only then publishes the transform: a policy waiting for the camera position starts with the hand open.
Without a running hand controller it measures the tag if it is visible.

If the samples spread more than 1 cm (tag partly hidden, camera or tag moving), they are discarded and collected again.

USAGE:
======
    python3 apriltag_calibration.py \
        --camera-topic /oakd_top_camera/color \
        --tag-family tag36h11 \
        --tag-size 0.08 \
        --tag-id 25 \
        --output camera_extrinsics.npy \
        --visualize
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path

import yaml
from typing import Optional, Tuple
from copy import deepcopy

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ingress.msg import CameraIntrinsics
from cv_bridge import CvBridge

# AprilTag detection
try:
    from dt_apriltags import Detector
    APRILTAG_AVAILABLE = True
except ImportError:
    print("⚠ Warning: dt_apriltags not installed. Install with: pip install dt-apriltags")
    APRILTAG_AVAILABLE = False

# Import rotation utilities
from rotation_utils import (
    rotation_matrix_to_quaternion,
    quaternion_to_rot_matrix,
    create_transform_matrix,
    euler_angles_to_rotation_matrix
)

# Import shared camera listener
from camera_utils import CameraListener

# The open hand hides the tag, so with --fist the node closes the fingers into orca_core's power_grasp pose (fractions
# of each joint's range of motion) while it measures, and opens them again afterwards. The wrist is left where it is.
try:
    from orca_core.demo_poses import load_demo_poses
    FIST_FRACTIONS = dict(load_demo_poses()["main"].pose_fractions["power_grasp"])
except Exception:  # orca_core without demo poses: the same values as its power_grasp
    FIST_FRACTIONS = {
        "thumb_cmc": 0.35, "thumb_abd": 0.55, "thumb_mcp": 0.2, "thumb_dip": 0.85,
        "index_mcp": 0.85, "middle_mcp": 0.85, "ring_mcp": 0.85, "pinky_mcp": 0.85,
        "index_pip": 0.9, "middle_pip": 0.9, "ring_pip": 0.9, "pinky_pip": 0.9,
    }
FIST_FRACTIONS.pop("wrist", None)
FIST_MOVE_S = 2.0  # time to close or open the hand
FIST_TOLERANCE_DEG = 15.0  # the hand counts as closed when every closing joint is this close to its target
FIST_SETTLE_TIMEOUT_S = 3.0  # measure anyway if the hand has not closed this long after the move


# ======================== AprilTag Calibration Node ========================

class AprilTagCalibration(Node):
    """
    ROS2 node for camera-to-tag calibration using AprilTag detection.
    
    Detects AprilTag in camera view and computes cam_T_tag transformation.
    """
    
    def __init__(self, args):
        super().__init__("apriltag_calibration")
        
        # Store arguments
        self.camera_topic = args.camera_topic
        self.camera_name = args.camera_name
        self.tag_family = args.tag_family
        self.tag_size = args.tag_size
        self.tag_id = args.tag_id
        self.output_file = args.output_file
        self.num_samples = args.num_samples
        self.visualize = args.visualize
        self.publish_tf = args.publish_tf
        self.parent_frame = args.parent_frame
        self.camera_frame = f"{self.camera_name}_frame"  # the frame cube_detection_node publishes the cube in
        self.max_translation_std = 0.01  # m; a calibration that spreads more is collected again

        # keeps the tag → camera transform on /tf_static for as long as this node runs
        self.tf_broadcaster = StaticTransformBroadcaster(self) if self.publish_tf else None

        # ---- hand: closed while measuring, as the open fingers hide the tag ----
        # none → closing → settling → closed → (measured) opening → open. Stays "none" while no hand controller runs:
        # then the tag is measured if it happens to be visible.
        self.use_fist = self.publish_tf if args.fist is None else args.fist
        self.hand_phase = "none"
        self.hand_pos = None  # rad, in the hand config's joint order
        self.hand_pos_time = None
        self.last_hand_cmd = None
        self.pending_cam_T_tag = None
        if self.use_fist:
            hand_cfg = yaml.safe_load(Path(args.hand_config).read_text())
            self.hand_joints = list(hand_cfg["joint_ids"])
            roms = hand_cfg["joint_roms"]
            self.fist_deg = {joint: roms[joint][0] + fraction * (roms[joint][1] - roms[joint][0])
                             for joint, fraction in FIST_FRACTIONS.items() if joint in roms}
            for item in args.fist_pose or []:
                joint, value = item.split("=")
                if joint not in self.hand_joints:
                    raise ValueError(f"--fist-pose: unknown joint {joint!r}, the hand has {self.hand_joints}")
                self.fist_deg[joint] = float(value)
            self.hand_cmd_pub = self.create_publisher(Float32MultiArray, args.hand_command_topic, 10)
            self.create_subscription(Float32MultiArray, args.hand_state_topic, self.hand_state_cb, 10)
            self.get_logger().info(
                "✊ The hand closes while measuring: " + ", ".join(f"{j} {d:.0f}°" for j, d in self.fist_deg.items())
            )

        self.get_logger().info("=" * 60)
        self.get_logger().info("📍 AprilTag Camera-to-Tag Calibration")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Tag Family: {self.tag_family}")
        self.get_logger().info(f"Tag Size: {self.tag_size:.4f} m")
        self.get_logger().info(f"Tag ID: {self.tag_id}")
        self.get_logger().info(f"Samples to collect: {self.num_samples}")
        self.get_logger().info(f"Output file: {self.output_file}")
        self.get_logger().info("=" * 60)
        
        # Check AprilTag library
        if not APRILTAG_AVAILABLE:
            self.get_logger().error("❌ dt_apriltags not available!")
            raise ImportError("Please install: pip install dt-apriltags")
        
        # Create camera listener
        self.listener = CameraListener(self, self.camera_topic, self.camera_name)
        
        # AprilTag detector (will be initialized when camera intrinsics arrive)
        self.detector = None
        
        # Storage for multiple samples
        self.cam_T_tag_samples = []
        self.calibration_complete = False
        
        # Create timer for processing
        self.timer = self.create_timer(0.1, self.process_once)  # 10 Hz
        
        self.get_logger().info("\n🎯 Waiting for camera stream and intrinsics...")
    
    
    def initialize_detector(self):
        """Initialize AprilTag detector once camera intrinsics are available."""
        if self.detector is not None:
            return
            
        if self.listener.mtx is None:
            return
        
        # Create detector
        self.detector = Detector(
            families=self.tag_family,
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        self.get_logger().info(f"✓ AprilTag detector initialized ({self.tag_family})")
    
    # ---------------- hand: close before measuring, open after ----------------

    def hand_state_cb(self, msg: Float32MultiArray):
        self.hand_pos = np.asarray(msg.data, dtype=np.float64)
        self.hand_pos_time = time.monotonic()

    def hand_is_live(self) -> bool:
        return (self.hand_pos is not None and time.monotonic() - self.hand_pos_time < 1.0
                and len(self.hand_pos) == len(self.hand_joints))

    def publish_hand(self, joint_pos_rad):
        self.last_hand_cmd = np.asarray(joint_pos_rad, dtype=np.float64)
        self.hand_cmd_pub.publish(Float32MultiArray(data=self.last_hand_cmd.astype(np.float32).tolist()))

    def start_hand_move(self, target, phase):
        self.move_from = self.last_hand_cmd if self.last_hand_cmd is not None else self.hand_pos.copy()
        self.move_to = np.asarray(target, dtype=np.float64)
        self.move_t0 = time.monotonic()
        self.hand_phase = phase

    def step_hand(self) -> bool:
        """Close the hand before measuring and open it again after. Returns True while the tag must not be sampled."""
        now = time.monotonic()

        if self.hand_phase == "none":
            if not self.hand_is_live():
                return False  # no hand controller (yet): measure if the tag happens to be visible
            if not np.all(np.isfinite(self.hand_pos)):
                self.get_logger().warn("The hand reports invalid joint positions, not closing it yet",
                                       throttle_duration_sec=3.0)
                return True
            # the hand is there: close it first, dropping samples taken before, as the fingers will move
            self.cam_T_tag_samples = []
            self.start_pose = self.hand_pos.copy()
            self.fist = self.start_pose.copy()
            for joint, deg in self.fist_deg.items():
                self.fist[self.hand_joints.index(joint)] = np.deg2rad(deg)
            self.start_hand_move(self.fist, "closing")
            self.get_logger().info("✊ Closing the hand so the camera sees the tag...")
            return True

        if self.hand_phase in ("closing", "opening"):
            t = min(1.0, (now - self.move_t0) / FIST_MOVE_S)
            self.publish_hand(self.move_from + t * (self.move_to - self.move_from))
            if t < 1.0:
                return True
            if self.hand_phase == "closing":
                self.hand_phase, self.settle_t0 = "settling", now
                return True
            # open again: only now may the camera position be used, so a policy starts after the hand has opened
            self.hand_phase = "open"
            self.get_logger().info("🖐 Hand open again")
            self.publish_camera_tf(self.pending_cam_T_tag)
            self.finish_publish()
            return True

        if self.hand_phase == "settling":
            closing = [self.hand_joints.index(joint) for joint in self.fist_deg]
            error = np.degrees(np.max(np.abs(self.hand_pos[closing] - self.fist[closing])))
            if error > FIST_TOLERANCE_DEG and now - self.settle_t0 < FIST_SETTLE_TIMEOUT_S:
                return True
            if error > FIST_TOLERANCE_DEG:
                self.get_logger().warn(f"The hand did not fully close (largest joint error {error:.0f}°), "
                                       "measuring anyway")
            self.hand_phase = "closed"
            self.get_logger().info("Hand closed: measuring the camera position")
            return False

        return False  # closed: measure

    def finish_publish(self):
        """--publish-tf: the transform is out; stay alive to keep it on /tf_static, but stop detecting."""
        self.timer.cancel()
        self.calibration_complete = True
        if self.visualize:
            cv2.destroyAllWindows()

    def process_once(self):
        """Main processing loop - detect tag and collect samples."""

        # Check if calibration is complete
        if self.calibration_complete:
            return

        # with --fist: no samples while the hand closes, settles or opens
        if self.use_fist and self.step_hand():
            return

        # Wait for camera to be ready
        if not self.listener.is_ready():
            return
        
        # Initialize detector if needed
        self.initialize_detector()
        if self.detector is None:
            self.get_logger().error("❌ Failed to initialize AprilTag detector")
            return
        
        # Get latest frame
        img_bgr, header = self.listener.get_frame()
        if img_bgr is None:
            self.get_logger().error("❌ Failed to get latest frame")
            return
        
        # Convert to grayscale for AprilTag detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        self.get_logger().debug("running detector...")
        # Detect tags
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[
                self.listener.fx,
                self.listener.fy,
                self.listener.cx,
                self.listener.cy
            ],
            tag_size=self.tag_size
        )
        self.get_logger().debug("detector finished")
        # Filter for our specific tag ID
        target_detections = [d for d in detections if d.tag_id == self.tag_id]
        self.get_logger().debug(f"target_detections: {len(target_detections)}")
        if len(target_detections) == 0:
            self.get_logger().warn(
                f"AprilTag {self.tag_id} ({self.tag_family}) not in view "
                f"(tags seen: {sorted(d.tag_id for d in detections)}); the camera position is not known yet. "
                + ("The hand is closed: something else hides the tag, or move it with --fist-pose"
                   if self.use_fist and self.hand_phase == "closed" else
                   "Start the hand controller: the hand then closes by itself so the camera sees the tag"
                   if self.use_fist else
                   "Uncover the tag for ~3 s; once the position is published, it may be covered again"),
                throttle_duration_sec=3.0,
            )
            if self.visualize:
                # Show frame with "no detection" message
                vis = img_bgr.copy()
                cv2.putText(vis, f"Searching for tag {self.tag_id}...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(vis, f"Samples: {len(self.cam_T_tag_samples)}/{self.num_samples}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("AprilTag Calibration", vis)
                cv2.waitKey(1)
            return
        
        # Get the detection
        detection = target_detections[0]
        
        # Extract camera_T_tag from detection
        R_cam_tag = detection.pose_R  # 3x3 rotation matrix
        t_cam_tag = detection.pose_t.flatten()  # 3x1 translation vector
        cam_T_tag = create_transform_matrix(R_cam_tag, t_cam_tag)
        
        # Store sample
        self.cam_T_tag_samples.append(cam_T_tag)
        
        self.get_logger().info(
            f"✓ Detected tag {self.tag_id} - Sample {len(self.cam_T_tag_samples)}/{self.num_samples} "
            f"(distance: {np.linalg.norm(t_cam_tag):.3f}m)"
        )
        
        # Visualize if requested
        if self.visualize:
            vis = img_bgr.copy()
            
            # Draw tag outline
            corners = detection.corners.astype(int)
            cv2.polylines(vis, [corners], True, (0, 255, 0), 3)
            
            # Draw tag ID
            center = tuple(corners.mean(axis=0).astype(int))
            cv2.putText(vis, f"ID: {detection.tag_id}", center,
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Draw axes (X=red, Y=green, Z=blue)
            self.draw_axis(vis, cam_T_tag, self.listener.mtx, self.listener.dist, self.tag_size)
            
            # Show progress
            cv2.putText(vis, f"Samples: {len(self.cam_T_tag_samples)}/{self.num_samples}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "Hold tag steady...", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("AprilTag Calibration", vis)
            cv2.waitKey(1)
        
        # Check if we have enough samples
        if len(self.cam_T_tag_samples) >= self.num_samples:
            self.compute_and_save_calibration()
    
    def draw_axis(self, img, cam_T_tag, K, dist, size):
        """Draw 3D coordinate axes on the image."""
        # Define axis points in tag frame
        axis_length = size * 0.5
        axis_points = np.array([
            [0, 0, 0],  # origin
            [axis_length, 0, 0],  # X axis (red)
            [0, axis_length, 0],  # Y axis (green)
            [0, 0, axis_length],  # Z axis (blue)
        ], dtype=np.float32)
        
        # Transform to camera frame
        R = cam_T_tag[:3, :3]
        t = cam_T_tag[:3, 3]
        
        # Project to image
        rvec, _ = cv2.Rodrigues(R)
        imgpts, _ = cv2.projectPoints(axis_points, rvec, t, K, dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        
        # Draw axes
        origin = tuple(imgpts[0])
        cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X - red
        cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y - green
        cv2.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z - blue
    
    def compute_and_save_calibration(self):
        """Average samples and compute final camera-to-tag transformation."""
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("📊 Computing camera-to-tag transformation from samples...")
        self.get_logger().info("=" * 60)
        
        # Average rotation and translation separately
        # For rotations, we'll average the rotation matrices (simple average is ok for small variations)
        rotations = [T[:3, :3] for T in self.cam_T_tag_samples]
        translations = [T[:3, 3] for T in self.cam_T_tag_samples]
        
        avg_rotation = np.mean(rotations, axis=0)
        avg_translation = np.mean(translations, axis=0)
        
        # Ensure rotation matrix is valid (orthogonalize via SVD)
        U, _, Vt = np.linalg.svd(avg_rotation)
        avg_rotation = U @ Vt
        
        # Create averaged cam_T_tag
        cam_T_tag_avg = create_transform_matrix(avg_rotation, avg_translation)
        
        # Compute standard deviations for quality check
        translation_std = np.std(translations, axis=0)
        
        self.get_logger().info(f"Average cam_T_tag translation: {avg_translation}")
        self.get_logger().info(f"Translation std: {translation_std}")
        self.get_logger().info(f"Max translation std: {np.max(translation_std):.6f} m")
        
        if np.max(translation_std) > self.max_translation_std:
            self.get_logger().warn(
                f"⚠ Samples spread {np.max(translation_std) * 100:.1f} cm (limit {self.max_translation_std * 100:.0f} cm): "
                "keep the camera and tag still and the tag fully visible. Collecting again..."
            )
            self.cam_T_tag_samples = []
            return

        self.get_logger().info("\n📐 Final Camera-to-Tag Transformation (cam_T_tag):")
        self.get_logger().info(f"\n{cam_T_tag_avg}")
        
        # Extract pose for readability
        t = cam_T_tag_avg[:3, 3]
        qx, qy, qz, qw = rotation_matrix_to_quaternion(cam_T_tag_avg[:3, :3])
        
        self.get_logger().info(f"\nTag position in camera frame:")
        self.get_logger().info(f"  x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}")
        self.get_logger().info(f"Tag orientation in camera frame (quaternion x,y,z,w):")
        self.get_logger().info(f"  {qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}")

        # Save to file
        try:
            np.save(self.output_file, cam_T_tag_avg)
            self.get_logger().info(f"\n✅ Camera-to-tag transformation saved to: {self.output_file}")
            self.get_logger().info("=" * 60)
        except Exception as e:
            self.get_logger().error(f"❌ Failed to save calibration: {e}")
            if not self.publish_tf:
                return

        if self.publish_tf:
            if self.use_fist and self.hand_phase == "closed":
                # open the hand first; the transform follows once it is open (step_hand), so a policy waiting for it
                # does not start while the hand still moves
                self.pending_cam_T_tag = cam_T_tag_avg
                self.start_hand_move(self.start_pose, "opening")
                self.get_logger().info("Measured: opening the hand again")
                return
            self.publish_camera_tf(cam_T_tag_avg)
            self.finish_publish()
            return

        # Show success message
        if self.visualize:
            img = np.zeros((400, 800, 3), dtype=np.uint8)
            cv2.putText(img, "CAMERA-TO-TAG CALIBRATION COMPLETE!", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(img, f"Saved to: {os.path.basename(self.output_file)}", (150, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Press any key to exit", (250, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("AprilTag Calibration", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        self.calibration_complete = True
        self.get_logger().info("🛑 Shutting down...")
        rclpy.shutdown()

    def publish_camera_tf(self, cam_T_tag):
        """Publish tag_T_cam as the static TF parent_frame → camera_frame (as the launch file did from the .npy)."""
        tag_T_cam = np.linalg.inv(cam_T_tag)
        qx, qy, qz, qw = rotation_matrix_to_quaternion(tag_T_cam[:3, :3])

        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.parent_frame
        msg.child_frame_id = self.camera_frame
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = (
            float(v) for v in tag_T_cam[:3, 3]
        )
        msg.transform.rotation.x, msg.transform.rotation.y = float(qx), float(qy)
        msg.transform.rotation.z, msg.transform.rotation.w = float(qz), float(qw)
        self.tf_broadcaster.sendTransform(msg)

        self.get_logger().info(
            f"✅ Camera position published: {self.parent_frame} → {self.camera_frame} "
            f"(camera {np.linalg.norm(tag_T_cam[:3, 3]):.3f} m from the tag)"
        )


# ==================== Main Entry Point ====================

def main():
    """Main function - parse arguments and run calibration."""
    
    parser = argparse.ArgumentParser(
        description="AprilTag Camera-to-Tag Calibration - Compute cam_T_tag transformation using AprilTag detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Camera parameters
    parser.add_argument(
        "--camera-topic",
        type=str,
        default="/oakd_bottom_view/color",
        help="Camera topic to subscribe to (default: /oakd_bottom_view/color)"
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="oakd_bottom_view",
        help="Camera name for logging (default: oakd_bottom_view)"
    )
    
    # AprilTag parameters
    parser.add_argument(
        "--tag-family",
        type=str,
        default="tag36h11",
        choices=["tag36h11", "tag25h9", "tag16h5", "tagCircle21h7", "tagCircle49h12", "tagStandard41h12", "tagStandard52h13"],
        help="AprilTag family (default: tag36h11)"
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.08,
        help="AprilTag size in meters (default: 0.08)"
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        default=25,
        help="AprilTag ID to detect (default: 25)"
    )
    
    
    # Output and collection
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=str,
        default="camera_extrinsics.npy",
        help="Output file for camera-to-tag transformation (default: camera_extrinsics.npy)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of samples to average (default: 30)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization window"
    )
    parser.add_argument(
        "--publish-tf",
        action="store_true",
        help="Also publish the result as static TF <parent-frame> → <camera-name>_frame and keep running "
             "(used by cube_spin_v2.launch.py at every start)"
    )
    parser.add_argument(
        "--parent-frame",
        type=str,
        default="tag_frame",
        help="Frame of the AprilTag in the TF tree (default: tag_frame)"
    )

    # Hand: closed while measuring (the open fingers hide the tag)
    try:
        from ament_index_python.packages import get_package_share_directory
        hand_share = Path(get_package_share_directory("hand_control"))
    except Exception:  # not built: the source tree next to experiments/
        hand_share = Path(__file__).resolve().parents[2] / "src" / "hand_control"
    parser.add_argument(
        "--fist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Close the hand (orca_core's power_grasp, wrist unchanged) while measuring and open it again after "
             "(default: on with --publish-tf). Acts once the hand controller publishes the joint state; "
             "without a hand controller the tag is measured if it is visible."
    )
    parser.add_argument(
        "--fist-pose",
        nargs="*",
        metavar="JOINT=DEG",
        help="Change joints of the closed pose, in degrees, e.g. --fist-pose thumb_abd=10 index_mcp=95"
    )
    parser.add_argument(
        "--hand-config",
        default=str(hand_share / "orca" / "configs" / "orcahand_v2" / "right" / "config.yaml"),
        help="Hand config.yaml with joint_ids (order of the hand topics) and joint_roms"
    )
    parser.add_argument("--hand-state-topic", default="/right/hand/joint_position_from_motor")
    parser.add_argument("--hand-command-topic", default="/right/hand/policy_output")

    # ros2 launch appends --ros-args ...; leave those to rclpy
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    
    print("🎯 AprilTag Camera-to-Tag Calibration")
    print("=" * 60)
    print("This script will:")
    print("1. Subscribe to camera images and intrinsics")
    print("2. Detect AprilTag markers in the camera view")
    print("3. Estimate camera pose relative to the tag")
    print("4. Average multiple samples for robustness")
    print("5. Save cam_T_tag transformation to file")
    print("=" * 60)
    
    # Initialize ROS
    rclpy.init()
    
    try:
        node = AprilTagCalibration(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n⚠ Keyboard interrupt - exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

