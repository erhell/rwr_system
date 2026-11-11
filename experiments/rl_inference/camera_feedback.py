"""Timestamped cube detections in the policy frame; independent of ROS and the detector."""
from collections import deque
import json
from pathlib import Path

import numpy as np

NO_DETECTION = np.array([0.0] * 7 + [1.0], dtype=np.float32)


class CubeDetectionHistory:
    def __init__(self, num_frames=3, max_age_s=1.0):
        if num_frames < 1 or not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('Camera history length and expiry must be positive')
        self.num_frames, self.max_age_s = num_frames, max_age_s
        self.detections = deque(maxlen=num_frames)
        self.last_now = None

    def observe(self, now, stamp=None, pose=None):
        if not np.isfinite(now):
            raise ValueError('Policy time must be finite')
        if self.last_now is not None and now < self.last_now:
            self.detections.clear()  # clock reset: discard detections from the old time domain
        self.last_now = now
        if pose is not None and stamp is not None:
            pose = np.asarray(pose, dtype=np.float32).copy()
            valid = (pose.shape == (7,) and np.isfinite(pose).all() and np.isfinite(stamp)
                     and 0 <= now - stamp < self.max_age_s)
            if valid and (not self.detections or stamp > self.detections[-1][0]):
                norm = np.linalg.norm(pose[3:])
                if norm > 1e-8:
                    pose[3:] /= norm
                    previous = self.detections[-1][1][3:] if self.detections else np.array([1., 0., 0., 0.])
                    if np.dot(pose[3:], previous) < 0:
                        pose[3:] *= -1
                    self.detections.append((stamp, pose))
        result = np.tile(NO_DETECTION, (self.num_frames, 1))
        offset = self.num_frames - len(self.detections)
        for i, (stamp, pose) in enumerate(self.detections, offset):
            age = now - stamp
            if 0 <= age < self.max_age_s:
                result[i, :7] = pose
                result[i, 7] = age / self.max_age_s
        return result


def deploy_meta_action_alpha(policy_path, default=0.2):
    """action_alpha of the training task from the deploy_meta.json next to the policy, or default without one.

    r15-r28 trained with 0.2, r29-r32 with 0.3; running a policy with the other value changes how fast its commands move.
    """
    path = Path(policy_path).with_name('deploy_meta.json')
    if not path.exists():
        return default
    value = json.loads(path.read_text()).get('action_alpha')
    return default if value is None else float(value)


def load_camera_contract(policy_path, obs_dim, num_frames, policy_hz, action_alpha, velocity_scale):
    """Validate new exports; older exports without the contract retain their legacy defaults."""
    path = Path(policy_path).with_name('deploy_meta.json')
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    camera = meta.get('camera_feedback')
    if camera is None:
        return None
    expected = [('joint_pos', 17), ('joint_vel', 17), ('object_pose', 8*num_frames),
                ('goal_command', 6), ('last_action', 17), ('last_joint_pos_cmd', 17)]
    layout = [(t['name'], t['dim']) for t in meta['obs_terms']]
    if layout != expected or sum(d for _, d in layout) != obs_dim:
        raise ValueError('Exported camera observation layout does not match the policy')
    if (camera.get('version') != 1 or camera.get('quaternion_order') != 'wxyz'
            or camera.get('num_frames') != num_frames or camera.get('frame_order') != 'oldest_first'):
        raise ValueError('Unsupported camera observation contract')
    if not np.isclose(meta['step_dt'], 1/policy_hz) or not np.isclose(meta['action_alpha'], action_alpha):
        raise ValueError('policy_hz/action_alpha differ from the training export')
    if not np.allclose(meta['obs_terms'][1]['scale'], velocity_scale):
        raise ValueError('velocity_obs_scale differs from training')
    transform = np.asarray(camera['obs_T_hand'], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError('Invalid camera-to-policy frame transform')
    CubeDetectionHistory(num_frames, camera['max_age_s'])
    return camera
