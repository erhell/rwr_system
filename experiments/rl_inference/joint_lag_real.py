#!/usr/bin/env python3
"""Lag between the commanded and the measured joint positions in policy_inference_cube_spin_v2.py CSV logs.

Per joint, the lag is the shift (in policy steps) at which the measured joint velocity correlates best with the change
of the command (cmd_*, sent after the EMA filter). obs_0..16 are the measured joint positions normalized to the joint
limits of the policy's deploy_meta.json. The same analysis in the sim is rwr_faive_lab's scripts/rsl_rl/joint_lag.py;
compare the two to see whether the sim's latency matches the hand (rwr_faive_lab r30 is trained with the measured lag).

    python joint_lag_real.py <run>/exported/deploy_meta.json logs/policy_runs/r21_sysid_16k_*.csv
"""

import argparse
import csv
import json

import numpy as np

MAX_LAG = 12


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("meta", help="deploy_meta.json of the policy that was run (joint names and limits)")
    p.add_argument("csv", nargs="+", help="policy run CSV files (csv_log_path)")
    p.add_argument("--min_steps", type=int, default=150, help="skip shorter runs")
    a = p.parse_args()

    meta = json.load(open(a.meta))
    lim = np.array(meta["joint_limits_rad"])
    lo, hi = lim[:, 0], lim[:, 1]
    names = [n.replace("right_", "") for n in meta["joint_names"]]
    step_ms = 1000 * meta.get("step_dt", 0.05)

    per_joint = {n: [] for n in names}
    periods = []
    for f in a.csv:
        rows = list(csv.reader(open(f)))
        head = rows[0]
        data = np.array([[float(x) if x not in ("", "nan") else np.nan for x in r] for r in rows[1:]])
        if len(data) < a.min_steps:
            continue
        periods.append(np.diff(data[:, 0]))
        q = (data[:, 1:18] + 1) / 2 * (hi - lo) + lo
        cmd = data[:, [head.index("cmd_" + n) for n in names]]
        for j, n in enumerate(names):
            dq, dc = np.diff(q[:, j]), np.diff(cmd[:, j])
            if np.std(dc) < 1e-4:
                continue
            c = np.array([np.corrcoef(dc[: len(dc) - k], dq[k:])[0, 1] for k in range(MAX_LAG + 1)])
            k = int(np.nanargmax(c))
            per_joint[n].append((k, c[k], np.std(q[:, j]) / np.std(cmd[:, j])))

    if not periods:
        raise SystemExit("no run with enough steps")
    dt = np.concatenate(periods)
    print(f"{len(periods)} runs; policy loop period: median {1000 * np.median(dt):.1f} ms, "
          f"p95 {1000 * np.percentile(dt, 95):.1f} ms, max {1000 * dt.max():.0f} ms")
    print(f"{'joint':12s} {'lag':>14s} {'corr':>5s} {'amplitude':>9s}")
    lags = []
    for n, v in per_joint.items():
        if v:
            v = np.array(v)
            k = np.median(v[:, 0])
            lags.append(k)
            print(f"{n:12s} {k:4.1f} ({k * step_ms:4.0f} ms) {np.median(v[:, 1]):5.2f} {np.median(v[:, 2]):9.2f}")
    print(f"median over joints: {np.median(lags):.1f} steps ({np.median(lags) * step_ms:.0f} ms)")


if __name__ == "__main__":
    main()
