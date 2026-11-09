#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cube View - the camera image with the cube pose that cube_detection_node estimates, in an OpenCV window.

A test of the camera and the cube detection without the hand, the AprilTag calibration, the policy or RViz. Started by
cube_view.launch.py together with the camera and the detector:

    ros2 launch experiments cube_view.launch.py model_path:=/path/to/maskrcnn-finetuned_lightning_new_adamw.ckpt

It reads the image from <camera_topic> (best effort, like the camera publishes it) and the detected cube pose from TF
(<camera_name>_frame -> cube, as cube_detection_node publishes it), and draws:
    - the cube's 12 edges and its axes (x red, y green, z blue) where the detector thinks the cube is,
      green while the detection is fresh, orange when it is older than fresh_s, nothing after max_age_s,
    - a status line: detection age (now - image time stamp), distance to the camera, detections per second and
      camera frames per second.
max_age_s is the age after which an r26 policy treats the cube as not seen (policy_inference_cube_spin_v2.py,
cube_max_age_s). Press q in the window to quit, s to save the current view as a PNG in the current folder.
"""

import time
from collections import deque

import cv2
import numpy as np
import rclpy
import rclpy.executors
import tf2_ros
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation

from camera_utils import CameraListener

# the 12 edges of a cube, as index pairs into CORNER_SIGNS
CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.abs(CORNER_SIGNS[i] - CORNER_SIGNS[j]).sum() == 2]


def draw_overlay(img, K, dist, cam_T_cube, edge_length, age_s, fresh_s, max_age_s, status):
    """Draw the cube at cam_T_cube (4x4, camera optical frame) and the status text onto img (BGR, in place)."""
    if cam_T_cube is not None and age_s is not None and age_s < max_age_s:
        color = (0, 220, 0) if age_s < fresh_s else (0, 165, 255)
        rvec, _ = cv2.Rodrigues(cam_T_cube[:3, :3])
        tvec = cam_T_cube[:3, 3].reshape(3, 1)
        corners = CORNER_SIGNS * edge_length / 2
        pts, _ = cv2.projectPoints(corners, rvec, tvec, K, dist)
        pts = pts.reshape(-1, 2)
        if np.all(np.isfinite(pts)) and cam_T_cube[2, 3] > 0:  # in front of the camera
            for i, j in EDGES:
                cv2.line(img, tuple(np.round(pts[i]).astype(int)), tuple(np.round(pts[j]).astype(int)), color, 2,
                         cv2.LINE_AA)
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, edge_length * 0.8, 2)
    lines = status if isinstance(status, list) else [status]
    for k, text in enumerate(lines):
        y = 28 + 28 * k
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


class CubeView(Node):
    def __init__(self):
        super().__init__("cube_view")
        self.declare_parameter("camera_name", "oakd_bottom_view")
        self.declare_parameter("camera_topic", "")  # default /<camera_name>/color
        self.declare_parameter("cube_frame_id", "cube")
        self.declare_parameter("cube_edge_length", 0.045)  # m, as cube_detection_node
        self.declare_parameter("fresh_s", 0.2)
        self.declare_parameter("max_age_s", 1.0)  # as cube_max_age_s of the policy node
        self.declare_parameter("show_window", True)
        self.declare_parameter("snapshot_path", "")  # non-empty: also write the current view to this PNG every second

        self.camera_name = self.get_parameter("camera_name").value
        topic = self.get_parameter("camera_topic").value or f"/{self.camera_name}/color"
        self.camera_frame = f"{self.camera_name}_frame"
        self.cube_frame = self.get_parameter("cube_frame_id").value
        self.edge = float(self.get_parameter("cube_edge_length").value)
        self.fresh_s = float(self.get_parameter("fresh_s").value)
        self.max_age_s = float(self.get_parameter("max_age_s").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.snapshot_path = self.get_parameter("snapshot_path").value

        self.listener = CameraListener(self, topic, self.camera_name)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.detection_stamps = deque(maxlen=200)  # image stamps of distinct detections
        self.frame_times = deque(maxlen=200)  # receive times of distinct camera frames
        self.last_frame_stamp = None
        self.last_print = self.last_snapshot = 0.0
        self.create_timer(1.0 / 20.0, self.update)
        self.get_logger().info(f"Showing {topic} with the cube pose {self.camera_frame} -> {self.cube_frame}; "
                               "press q in the window to quit, s to save a PNG")

    def lookup_cube(self):
        """Latest camera_T_cube (4x4) and its image stamp in seconds, or (None, None)."""
        try:
            tf = self.tf_buffer.lookup_transform(self.camera_frame, self.cube_frame, Time())
        except tf2_ros.TransformException:
            return None, None
        t, q = tf.transform.translation, tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        return T, Time.from_msg(tf.header.stamp).nanoseconds * 1e-9

    def update(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        img, hdr = self.listener.get_frame()
        if img is None:
            self.get_logger().info(f"Waiting for camera images on {self.listener.camera_topic}...",
                                   throttle_duration_sec=3.0)
            return
        stamp = hdr.stamp.sec + hdr.stamp.nanosec * 1e-9
        if stamp != self.last_frame_stamp:
            self.last_frame_stamp = stamp
            self.frame_times.append(now)

        cam_T_cube, det_stamp = self.lookup_cube()
        if det_stamp is not None and (not self.detection_stamps or det_stamp > self.detection_stamps[-1]):
            self.detection_stamps.append(det_stamp)
        recent = [s for s in self.detection_stamps if now - s < 2.0]
        det_rate = len(recent) / 2.0
        cam_rate = sum(1 for t in self.frame_times if now - t < 2.0) / 2.0
        age = None if det_stamp is None else max(0.0, now - det_stamp)

        if self.listener.mtx is None:
            status = ["no camera intrinsics yet"]
        elif age is None:
            status = ["cube: never detected"]
        elif age >= self.max_age_s:
            status = [f"cube: not detected for {age:.1f} s (policy: no camera)"]
        else:
            dist = np.linalg.norm(cam_T_cube[:3, 3])
            status = [f"cube: detected, age {age * 1000:.0f} ms, {dist:.3f} m from the camera"]
        status.append(f"detections {det_rate:.1f}/s | camera {cam_rate:.1f} fps")

        if self.listener.mtx is not None:
            dist_coeffs = None if self.listener.dist is None else np.asarray(self.listener.dist, dtype=np.float64)
            draw_overlay(img, self.listener.mtx, dist_coeffs, cam_T_cube, self.edge, age, self.fresh_s,
                         self.max_age_s, status)
        else:
            draw_overlay(img, None, None, None, self.edge, None, self.fresh_s, self.max_age_s, status)

        if now - self.last_print >= 2.0:
            self.last_print = now
            self.get_logger().info(" | ".join(status))
        if self.snapshot_path and now - self.last_snapshot >= 1.0:
            self.last_snapshot = now
            cv2.imwrite(self.snapshot_path, img)
        if self.show_window:
            cv2.imshow("cube view", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                name = time.strftime("cube_view_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, img)
                self.get_logger().info(f"Saved {name}")
            elif key == ord("q"):
                raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = CubeView()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
