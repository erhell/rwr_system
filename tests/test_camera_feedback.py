"""Detector loss, timing and train/deploy parity; runs without ROS, Isaac Sim or a GPU."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

SYSTEM = Path(__file__).resolve().parents[1]
LAB = Path(os.environ.get('FAIVE_LAB_ROOT', SYSTEM.parent/'rwr_faive_lab'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


real = module('camera_real', SYSTEM/'experiments/rl_inference/camera_feedback.py')
sim = module('camera_sim', LAB/'source/faive_lab/faive_lab/tasks/orca_inhand/mdp/camera_feedback.py')


class CameraTests(unittest.TestCase):
    def test_missing_expired_duplicate_invalid_and_recovery(self):
        history = real.CubeDetectionHistory()
        absent = np.tile(real.NO_DETECTION, (3, 1))
        np.testing.assert_array_equal(history.observe(10), absent)
        pose = [0., 0., -0.05, -1., 0., 0., 0.]
        first = history.observe(10.2, 10., pose)
        self.assertAlmostEqual(first[-1, -1], .2)
        self.assertEqual(first[-1, 3], 1.)
        history.observe(10.3, 10., pose)
        self.assertEqual(len(history.detections), 1)
        history.observe(10.4, 10.5, pose)  # future image
        history.observe(10.4, 10.4, [float('nan')]*7)
        history.observe(10.4, 10.4, [0.]*7)
        self.assertEqual(len(history.detections), 1)
        np.testing.assert_array_equal(history.observe(11.), absent)
        self.assertLess(history.observe(11.2, 11.1, pose)[-1, -1], 1)
        np.testing.assert_array_equal(history.observe(1.), absent)  # clock rollback

    def test_sim_real_match_during_loss_recovery_and_reset(self):
        model = sim.CameraFeedback(1, 'cpu', pos_std=0, quat_std=0, max_delay_steps=0,
                                   detect_prob_range=(1., 1.), outage_prob=0, camera_off_prob=0)
        history = real.CubeDetectionHistory()
        for step in range(65):
            reset = step in (0, 55)
            if reset:
                history = real.CubeDetectionHistory()
            lost = 4 <= step < 48
            model.off[:] = lost
            pose = torch.tensor([[step*.001, 0., -.05, (-1.)**step, 0., 0., 0.]])
            expected = model.update(pose, torch.tensor([reset]), .05).numpy().reshape(3, 8)
            now = step*.05
            actual = history.observe(now, now if not lost else None, pose[0].numpy() if not lost else None)
            np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_delay_does_not_invent_pre_reset_detections(self):
        model = sim.CameraFeedback(32, 'cpu', max_delay_steps=3, pos_std=0, quat_std=0,
                                   detect_prob_range=(1, 1), outage_prob=0, camera_off_prob=0)
        pose = torch.zeros(32, 7); pose[:, 3] = 1
        result = model.update(pose, torch.ones(32, dtype=torch.bool), .05).reshape(32, 3, 8)
        delayed = model.delay > 0
        self.assertTrue(delayed.any())
        self.assertTrue((result[delayed, :, -1] == 1).all())
        self.assertTrue((result[delayed, :, :7] == 0).all())
        for step in range(1, 6):
            pose[:, 0] = step
            result = model.update(pose, torch.zeros(32, dtype=torch.bool), .05).reshape(32, 3, 8)
        torch.testing.assert_close(result[:, -1, 0], (5-model.delay).float())
        torch.testing.assert_close(result[:, -1, -1], model.delay.float()*.05)

    def test_no_camera_episode_and_partial_reset(self):
        model = sim.CameraFeedback(4, 'cpu', camera_off_prob=1, max_delay_steps=0)
        pose = torch.zeros(4, 7); pose[:, 3] = 1
        for i in range(30):
            result = model.update(pose, torch.tensor([i == 0, i == 0, i in (0, 5), i == 0]), .05).reshape(4, 3, 8)
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue((result[:, :, :7] == 0).all())
            self.assertTrue((result[:, :, 7] == 1).all())

    def test_exported_contract_is_checked(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'policy.pt'
            self.assertIsNone(real.load_camera_contract(path, 98, 3, 20, .2, .2))
            meta = {'step_dt': .05, 'action_alpha': .2,
                    'obs_terms': [{'name': k, 'dim': n, 'scale': .2 if k == 'joint_vel' else None}
                                  for k, n in [('joint_pos',17),('joint_vel',17),('object_pose',24),
                                               ('goal_command',6),('last_action',17),('last_joint_pos_cmd',17)]],
                    'camera_feedback': {'version': 1, 'quaternion_order': 'wxyz', 'num_frames': 3,
                                        'frame_order': 'oldest_first', 'obs_T_hand': np.eye(4).tolist(),
                                        'max_age_s': 1.}}
            path.with_name('deploy_meta.json').write_text(json.dumps(meta))
            self.assertEqual(real.load_camera_contract(path, 98, 3, 20, .2, .2)['max_age_s'], 1.)
            with self.assertRaises(ValueError):
                real.load_camera_contract(path, 98, 3, 30, .2, .2)
            meta['obs_terms'][2]['dim'] = 21
            path.with_name('deploy_meta.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                real.load_camera_contract(path, 98, 3, 20, .2, .2)

    def test_deployment_continues_without_detection(self):
        # Exercise the actual controller callback with fake ROS endpoints and a CPU policy.
        import ast
        from types import SimpleNamespace as NS
        source = ast.parse((SYSTEM/'experiments/rl_inference/policy_inference_cube_spin_v2.py').read_text())
        cls = next(x for x in source.body if isinstance(x, ast.ClassDef))
        fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'process_once')
        scope = {'np': np, 'torch': torch, 'numpy_to_float32_multiarray': lambda x: x}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<controller>', 'exec'), scope)
        logs = NS(info=lambda *a, **k: None, warn=lambda *a, **k: None)
        sent = []
        seen = []
        def policy(obs):
            seen.append(obs.clone())
            return torch.full((1, 17), 2.)
        node = NS(current_joint_pos=np.zeros(17), current_joint_vel=np.zeros(17),
                  robot_to_usd_idx=list(range(17)), usd_to_robot_idx=list(range(17)),
                  frame_ready=True, get_clock=lambda: NS(now=lambda: NS(nanoseconds=10_000_000_000)),
                  start_time=None, goal_flip_period_s=0, cube_history=real.CubeDetectionHistory(),
                  hand_frame='hand', cube_frame='cube', lookup_transform_from_tf=lambda *a: (None, None),
                  last_cube_stamp=None, last_object_pos=None, last_object_quat=None,
                  cube_pose_timeout_s=.5, cube_max_age_s=1., cube_ages=True, use_cube_obs=True,
                  camera_sees_cube=False, normalize=lambda x: x, denormalize=lambda x: x,
                  velocity_obs_scale=.2, goal_command=np.zeros(6), last_action_usd=np.zeros(17),
                  policy=policy, action_alpha=.2, limits_usd=np.tile([-1., 1.], (17, 1)),
                  dry_run=False, hand_pub=NS(publish=sent.append), csv_writer=None, log_period_s=0,
                  get_logger=lambda: logs)
        scope['process_once'](node)
        self.assertEqual(len(sent), 1)
        self.assertEqual(tuple(seen[0].shape), (1, 98))
        np.testing.assert_array_equal(seen[0][0, 34:58].reshape(3, 8).numpy(), np.tile(real.NO_DETECTION, (3, 1)))
        self.assertTrue(np.isfinite(sent[0]).all())
        np.testing.assert_array_equal(node.last_action_usd, np.ones(17))


def _yaw_quat(angle):
    return torch.stack((torch.cos(angle / 2), torch.zeros_like(angle), torch.zeros_like(angle), torch.sin(angle / 2)), -1)


def _spin(camera, steps, rate=2.0, missed=()):
    """Feed a cube turning at rate rad/s about z, every step detected except the steps in missed; last output."""
    for k in range(steps):
        angle = torch.full((camera.n,), rate * 0.05 * k)
        camera.off[:] = k in missed
        out = camera.update(torch.cat((torch.zeros(camera.n, 3), _yaw_quat(angle)), -1),
                            torch.zeros(camera.n, dtype=torch.bool), 0.05)
    return angle, out


def _newest_yaw(out):
    q = out[:, 16 + 3:16 + 7]  # 3 frames x [pos 3, quat 4, age 1], newest last
    return 2 * torch.atan2(q[:, 3], q[:, 0])


class TrackerModelTests(unittest.TestCase):
    """cube_detection_node's smoothing and the cube's symmetry in the simulated camera (2026-11-11, off by default)."""
    exact = dict(pos_std=0, quat_std=0, max_delay_steps=0, detect_prob_range=(1, 1), outage_prob=0, camera_off_prob=0)

    def test_options_off_change_nothing(self):
        outs = []
        for kwargs in ({}, dict(tracker_filter=False, symmetry_switch_prob=0.0)):
            torch.manual_seed(0)
            outs.append(_spin(sim.CameraFeedback(4, 'cpu', camera_off_prob=0.0, **kwargs), 40)[1])
        torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)

    def test_slerp_lags_three_frames(self):
        angle, out = _spin(sim.CameraFeedback(1, 'cpu', tracker_filter=True, **self.exact), 80)
        lag = float(torch.remainder(angle - _newest_yaw(out), 2 * np.pi))
        self.assertAlmostEqual(lag / (2.0 * 0.05), (1 - 0.25) / 0.25, places=2)  # (1 - alpha) / alpha frames

    def test_symmetry_switch_is_a_cube_rotation(self):
        rotations = sim.cube_rotations('cpu')
        self.assertEqual(rotations.shape[0], 24)
        angle, out = _spin(sim.CameraFeedback(64, 'cpu', symmetry_switch_prob=1.0, **self.exact), 3)
        true = _yaw_quat(angle)
        rel = sim.quat_mul(torch.cat((true[:, :1], -true[:, 1:]), -1), out[:, 16 + 3:16 + 7])
        self.assertTrue(bool(((rel[:, None] * rotations[None]).sum(-1).abs().max(dim=1).values > 0.9999).all()))
        self.assertTrue(bool((rel[:, 0].abs() < 0.9999).all()))  # never the identity

    def test_track_reset_after_five_missed_frames(self):
        for missed, reset in ((4, False), (5, True)):
            camera = sim.CameraFeedback(1, 'cpu', tracker_filter=True, **self.exact)
            angle, out = _spin(camera, 41 + missed, missed=set(range(40, 40 + missed)))
            lag = abs(float(torch.remainder(angle - _newest_yaw(out) + np.pi, 2 * np.pi) - np.pi))
            self.assertEqual(lag < 1e-4, reset, f'{missed} missed frames')


class DeployMetaTests(unittest.TestCase):
    def test_action_alpha_from_deploy_meta(self):
        runs = LAB/'logs/rsl_rl/orca_v2_repose'
        for run, alpha in (('2026-11-08_20-44-06_r21_sysid_16k', 0.2), ('2026-11-10_18-39-03_r30', 0.3)):
            policy = runs/run/'exported/policy.pt'
            if not policy.with_name('deploy_meta.json').exists():
                self.skipTest(f'{run} not pulled')
            self.assertAlmostEqual(real.deploy_meta_action_alpha(policy), alpha)
        self.assertEqual(real.deploy_meta_action_alpha('/nonexistent/policy.pt'), 0.2)


if __name__ == '__main__':
    unittest.main()
