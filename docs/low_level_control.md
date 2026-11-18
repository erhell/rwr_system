# Low-level control of the ORCA v2 hand: what every model needs

Last updated 2026-11-15 ~14:00 UTC (r100 / r100a / r100c / r100xyz training; r102 and r103 queued; KP 48 recorded and
fitted; r101 and ACC 0 dropped). Policy list and test plan: `~/Documents/RWR/POLICIES_AND_TESTS.md`.

Every model is trained in a simulation that copies one particular low-level setup of the real hand: action filter,
servo acceleration (ACC), current limit, servo gains, sysID fit. If the hand runs a model with another model's setup,
the model meets joint dynamics it never saw. **The model's `exported/deploy_meta.json` is the source of truth.** The
policy node reads it, applies the settings, prints them at startup and refuses to command the hand when the servo ACC
cannot be set (from r103 on also the position gain KP, see section 4). This page lists what exists, what each model uses, and how to check it.

## 1. The chain from the policy to the joints

```
policy (20 Hz, 17 actions in [-1, 1])
 -> mapped onto the joint limits
 -> action filter (policy node; the same filter as the sim's action term)   EMA alpha  or  Butterworth 2nd order
 -> wrist held at a fixed angle (r34+, fix_wrist)
 -> /right/hand/policy_output -> hand_control_node -> orca_core set_joint_pos -> serial bus (1 Mbaud)
 -> inside each Feetech HLS servo (mode 0, "position under current limit"):
      trapezoidal profile: the servo's internal reference moves towards the goal with at most
        ACC (register 41) x 8.7 deg/s^2 at the motor, and at most GOAL_SPEED (register 46)
      position loop KP / KD / KI (registers 50-52) tracks that reference, motor current <= GOAL_CURRENT (register 44)
```

Motors (orca_core `feetech_client.py`): HLS3915 on the 16 finger joints (no-load 100 rpm), HLS3930 on the wrist
(45 rpm). hand_control_node reads all motors every 10 ms; the policy node runs at 20 Hz.

## 2. The servo registers: value, who sets it, is it in the simulation?

| Register (HLS) | Unit | Value on the hand | Range | Set by | In the simulation |
|---|---|---|---|---|---|
| **ACC** (41, RAM) | 8.7 deg/s^2 = 0.1518 rad/s^2 at the motor | **per model** (table in section 3); orca_core starts with 150 | 0-254; **254 = steepest ramp**; 0 = no ramp at all. **The wrist motor (HLS3930) holds at most 250** (read back 2026-11-15) | policy node, from `deploy_meta.json servo_profile.acc`, via `/right/hand/set_servo_acc`; set back to 150 when the node exits | r60+: the action term's `servo_acc` ramp (a_joint = ACC x 0.1518 / motor-rad-per-joint-rad). r21-r42: not modeled, trained for 150 (the old fits absorbed the ramp into damping and delay) |
| GOAL_SPEED (46, RAM) | 0.732 rpm | 1500 = 1098 rpm, i.e. no cap (the firmware limits to the motor's no-load speed) | 0-32767 (0 = stop!) | orca_core at start (`DEFAULT_SPEED`), never changed | not needed (no cap below the motor's speed) |
| GOAL_CURRENT (44, RAM) | 6.5 mA, rounded down | **900 mA -> 138 = 897 mA** on all motors; **r104: 1000 mA -> 153 = 994.5 mA** | 0-2047, capped by the protection current (EEPROM 28); rwr_system allows 300-1000 mA | orca_core at start (`config.yaml max_current: default`); since 2026-11-15 the policy node for every model (deploy_meta `servo_profile.max_current_ma`, missing = 900), via `/right/hand/set_max_current_ma`; 900 again after the run | implicitly: every sysID fit was recorded at 900 mA, so its effort limits belong to 900 mA. Written into `deploy_meta.json servo_profile.max_current_ma` (r100+ exports) for the record |
| **KP** (50, RAM, loaded from EEPROM 21 at power-up) | - | **32** on all 17 (read 2026-11-15); **r103 and r60e_max_acc_kp48: 48** | 0-254 (rwr_system allows 16-64) | the policy node for every model (deploy_meta `servo_profile.kp`, missing = 32), via hand_control_node `/right/hand/set_servo_kp`; 32 again after the run. **orca_core never rewrites KP: a value set stays until set again or a power cycle** | implicitly, in the fitted stiffness and damping: KP 32 for every fit up to 2026-11-14, KP 48 for `ORCA_V2_KP48_FIT_20261115` |
| KD / KI (51-52, RAM) | - | 32 / 0 on all 17 (read 2026-11-15) | 0-254 | nobody | implicitly, in the fits |
| Baud rate, mode, IDs | - | 1 Mbaud, mode 0 | - | calibration / orca_core | - |

Joint acceleration limit of the servo ramp (rad/s^2), `orca_v2.py servo_accel_limits`:

| Joint | ACC 150 | ACC 254 | Joint | ACC 150 | ACC 254 |
|---|---|---|---|---|---|
| wrist | 20.6 | 34.9 | middle_abd | 25.6 | 43.3 |
| thumb_cmc | 27.8 | 47.1 | middle_mcp | 22.0 | 37.2 |
| thumb_abd | 27.3 | 46.2 | middle_pip | 29.0 | 49.1 |
| thumb_mcp | 23.9 | 40.5 | ring_abd | 27.8 | 47.0 |
| thumb_dip | 26.7 | 45.2 | ring_mcp | 22.0 | 37.2 |
| index_abd | 27.5 | 46.5 | ring_pip | 27.5 | 46.6 |
| index_mcp | 22.9 | 38.8 | pinky_abd | 26.7 | 45.2 |
| index_pip | 27.5 | 46.6 | pinky_mcp | 26.5 | 44.9 |
| | | | pinky_pip | 29.7 | 50.4 |

Why the ACC matters: the chirps of 2026-11-13 (recorded at ACC 150) saturate exactly at these limits above ~2 Hz
(commanded up to 112 rad/s^2, measured 20-27). The servo ramp, not the motor, limits the real joints. With the ramp
modeled, the sysID fit error halves and the identified dead time drops from 40-67 ms to 0-27 ms (the earlier fits
called the ramp "delay"). On the hand (other session's note, 2026-11-15 morning), ACC 254 on thumb + ring (r60e) cut
the thumb_dip tracking error from ~40 to ~16 deg and sped the gait from 1.10 to 1.24 Hz.

## 3. Settings per model

Filter: what the policy node applies (EMA: `cmd = alpha * new + (1 - alpha) * cmd`). ACC columns: thumb (cmc, abd, mcp,
dip) / ring (abd, mcp, pip) / index, middle, pinky (abd, mcp, pip) / wrist (a wrist ACC of 254 is held as 250 on the
hand: hardware cap, the node sets and checks 250). "Ramp in sim": whether the training simulated the servo ramp with
these ACC values. Current limit: 900 mA for every model (orca_core default). KP 32 (the motors' own) for every model
except r103 (48); KD 32, KI 0 for all. Goal speed: 1500 for every model.

| Model | Filter | Servo profile (deploy_meta name) | ACC thumb / ring / idx-mid-pinky / wrist | Ramp in sim | sysID fit | Wrist | Inputs |
|---|---|---|---|---|---|---|---|
| r21_sysid_16k, r28 | EMA 0.2 | none -> default | 150 / 150 / 150 / 150 | no | 2026-11-08 (13 joints) | free | 74 |
| r26_8k | EMA 0.2 | none -> default | 150 everywhere | no | before the fits | free | 98 (camera) |
| r29, r30 | EMA 0.3 | none -> default | 150 everywhere | no | 2026-11-08 (16 joints) | free | 74 |
| r31 / r32 | EMA 0.3 | none -> default | 150 everywhere | no | 2026-11-11 refit | free | 74 / 98 |
| r34 / r34c / r35 | EMA 0.3 | none -> default | 150 everywhere | no | 2026-11-13 | 15 deg | 74 / 98 / 98 |
| r36, r36w | EMA 0.45 | none -> default | 150 everywhere | no | 2026-11-13 | 15 / 30 deg | 74 |
| r36wx | EMA 0.6 | none -> default | 150 everywhere | no | 2026-11-13 | 30 deg | 74 |
| r37, r37c, r42, r42s | EMA 0.3 | none -> default | 150 everywhere | no | 2026-11-13 | 15 deg | 74 |
| **r60** | Butterworth 3 Hz | fast_thumb_ring | 254 / 254 / 150 / 150 | yes | 2026-11-14 ramp | 15 deg | 74 |
| r60e | EMA 0.3 | fast_thumb_ring | 254 / 254 / 150 / 150 | yes | 2026-11-14 ramp | 15 deg | 74 |
| r61 | Butterworth 3 Hz | fast_thumb_ring | 254 / 254 / 150 / 150 | yes | 2026-11-14 ramp | 15 deg | 98 (camera) |
| r65 | EMA 0.3 | fast_all | 254 / 254 / 254 / 150 | yes | 2026-11-14 ramp | 15 deg | 74 |
| **r100** | **Butterworth 5 Hz** | **acc254_all** | **254 / 254 / 254 / 254** | yes | 2026-11-14 ramp | 15 deg | 74 |
| **r100a** | Butterworth 3 Hz | **acc254_all** | **254 / 254 / 254 / 254** | yes | 2026-11-14 ramp | 15 deg | 74 |
| **r100c** | Butterworth 3 Hz | **acc254_all** | **254 / 254 / 254 / 254** | yes | 2026-11-14 ramp | 15 deg | 98 (camera) |
| **r100xyz** | Butterworth 3 Hz | **acc254_all** | **254 / 254 / 254 / 254** | yes | 2026-11-14 ramp | 15 deg | 74 |
| r101 | - | - | - | - | - | - | **cancelled before training** (current limit 1000 mA; code parked) |
| **r60e_max_acc** (deployment variant, 2026-11-15) | EMA 0.3 | **acc254_all** (r60e trained with fast_thumb_ring) | **254 / 254 / 254 / 254** (trained: 254 / 254 / 150 / 150) | **no** for idx/mid/pinky/wrist at 254 | 2026-11-14 ramp | 15 deg | 74 |
| **r60e_max_acc_kp48** (deployment variant) | EMA 0.3 | **acc254_all + KP 48** (trained: fast_thumb_ring, KP 32) | **254 / 254 / 254 / 254** | **no** (ACC as above, KP 48 never trained) | 2026-11-14 ramp | 15 deg | 74 |
| **r102** (queued) | **Butterworth 5 Hz** | **acc254_all** | **254 / 254 / 254 / 254** | yes | 2026-11-14 ramp | 15 deg | 74 |
| **r103** (queued) | **Butterworth 5 Hz** | **acc254_all** + **KP 48** | **254 / 254 / 254 / 254** | yes | **2026-11-15 ramp ACC 254 KP 48** | 15 deg | 74 |
| **r104** (model files deleted 2026-11-16, user) | **EMA 0.3** (= r60e) | **acc254_all** + **1000 mA** | **254 / 254 / 254 / 254** | yes | 2026-11-14 ramp, **effort limits x1.109** | 15 deg | 74 |
| **r105a / r105** (queued) | Butterworth 3 Hz (= r100a) | **acc254_all** (= r100a), KP 32, 900 mA | **254 / 254 / 254 / 254** | yes, **+ motors' top speed** (sim only) | 2026-11-14 ramp | 15 deg | 74 |
| **r106** (queued) | Butterworth 3 Hz (= r100a) | **acc254_all** (= r100a), KP 32, 900 mA | **254 / 254 / 254 / 254** | yes, **+ motors' top speed** (sim only) | 2026-11-14 ramp | 15 deg | 74 |
| **r107** (queued) | **Butterworth 5 Hz** (= r100) | **acc254_all** (= r100), KP 32, 900 mA | **254 / 254 / 254 / 254** | yes | 2026-11-14 ramp | 15 deg | 74 |

All of r34+ feed the last action back unclipped (node default since 2026-11-13). Commands: every model is trained
about z (vertical) with CW and CCW equally, except r100 / r100a / r100c (80 % CW = negative z, seen from above) and
r100xyz (one of x / y / z, either sign); r102 and r103 are trained like r100 (80 % CW). deploy_meta.json `command` holds
it (`negative_prob`, `axis_mode`). r102 differs from r100 only in the rewards (penalty relief on all fingers), r103 only
in the actuator fit (recorded at KP 48) and therefore in KP on the hand. r62-r64 and r66 were never trained
here (r62/r63 dropped with the old queue, r64/r66 moved to DelftBlue). The "2026-11-14 ramp" fit is the chirp recording
of 2026-11-13 (ACC 150, 900 mA) fitted with the ramp: `rwr_faive_lab/source/faive_lab/faive_lab/assets/orca_v2.py`
`ORCA_V2_RAMP_FIT_20261114`. It is used at ACC 254 by r60+ (the ramp model scales with the ACC value).

Profiles (`orca_v2.py ORCA_V2_SERVO_PROFILES`): `default` = 150 on all; `fast_thumb_ring` = 254 on the 7 thumb and ring
motors; `fast_all` = 254 on the 16 finger motors, wrist 150; `acc254_all` = 254 on all 17.

## 4. How deployment applies and checks it

- **Policy node** (`experiments/rl_inference/policy_inference_cube_spin_v2.py`, also behind the spin panel and
  `spin.zsh`): reads `deploy_meta.json` next to `policy.pt`; prints the filter (with its coefficients) and the servo
  profile at startup; before the first step it publishes the model's ACC per joint on `/right/hand/set_servo_acc`,
  waits until hand_control_node reports the same values read back from the motors on `/right/hand/servo_acc`, and
  **refuses to command the hand** if a model with a servo profile cannot get it. On exit (also Ctrl+C) it sets 150 on
  all joints again. Models without a servo profile (r21-r42) get 150. A dry run only reports, it changes nothing.
- **hand_control_node** starts with orca_core's ACC 150 and publishes what the motors hold (latched). ACC is a RAM
  register: a restart or a power cycle of the hand sets 150 again; the node sets it per run anyway.
- **Check on the hand** while a model runs: `ros2 topic echo /right/hand/servo_acc --once` (17 values in the hand's
  joint order) and the node's startup lines (`servo profile 'acc254_all' set: wrist=254 ...`,
  `Policy inference v2 running at 20 Hz, Butterworth 2nd order 5.0 Hz, b [...]`).
- **ROS-free runner** `rwr_faive_lab/scripts/deploy/run_policy.py` has only the EMA and never sets the ACC: since
  2026-11-15 it refuses every model with a Butterworth filter or a servo profile other than 150 (r60+, r65, r100*).
- **Spin axis** (2026-11-15): node parameter `goal_axis` (`z` default; `x` / `y` only for models whose deploy_meta has
  `command.axis_mode: principal`, i.e. r100xyz; refused otherwise), panel "Spin axis". Axes are the simulation's world
  axes: z up, y along the fingers, x across the palm. The startup line `goal -1.60 rad/s about z (clockwise seen from
  above); trained on: ...` shows the command and what the model was trained on.
- **Wrist ACC cap** (2026-11-15): the node sets min(model's ACC, 250) on the wrist and checks that; for the r100
  family the startup log says `servo ACC wrist: model trained with 254, the motor holds at most 250 (hardware cap)`.
  Every other joint must hold exactly the model's value.
- **KP** (built 2026-11-15 ~17:00): hand_control_node reads KP back from the motors, publishes it latched on
  `/right/hand/servo_kp`, accepts `/right/hand/set_servo_kp` (hand_controller_v2 `set_servo_kp`: integers 16-64 only)
  and logs `servo KP at start: ...`. The policy node sets the model's KP (`deploy_meta.json servo_profile.kp`, 32 when
  missing = every model up to r102) before the first step, checks the read-back, refuses on mismatch, and sets 32
  after the run (also after an error). Because orca_core never rewrites KP, every model checks it: a run that left 48
  is corrected by the next model. With a hand controller started before this code, KP-32 models run with a warning and
  KP-48 models refuse. Tested against a fake hand (KP 48 set/restored, left-over 48 corrected, old controller, dry run)
  and on orca_core's mock motors; first real use: restart the hand controller and check its `servo KP at start` line.
  The ROS-free `run_policy.py` refuses KP-48 models.
- **Current limit** (built 2026-11-15 ~17:30): hand_control_node reads register 44 back, publishes it latched on
  `/right/hand/max_current_ma` (mA), accepts `/right/hand/set_max_current_ma` (300-1000 mA only) and logs `motor current
  limit at start`. The policy node sets the model's limit (`servo_profile.max_current_ma`, 900 when missing = every model
  except r104) and compares with what the register holds (900 -> 897, 1000 -> 994.5); refuses on mismatch (also when a
  motor's protection current caps it) and sets 900 after the run. Old controller: 900-mA models warn, r104 refuses.
  Tested against a fake hand (set + restore, left-over limit corrected, old controller, motors capped at 1000 mA).
- **Regression test 2026-11-15 ~17:35:** r60e, r60, r34, r21 through the node before and after the KP and current-limit
  code (same fake hand, same inputs, real mode): all 91 observation / command columns identical over 60 steps (max
  difference 0.0), the same ACC sent to the hand; the new node only reads KP 32 / 897 mA and finds them in place.
- Goal speed, KD/KI: never changed, so there is nothing to set; r100+ exports write them into
  `servo_profile` (`max_current_ma`, `goal_speed_raw`, `position_gains`, and `kp` from r102/r103 on) so every file is
  complete.

## 5. r100 / r100a / r100c / r100xyz (2026-11-15): maximum ACC on all joints

User request after the hand tests of 2026-11-15, where r60 was the best (2.25 rad/s at goal 1.6, CW, 80 s) and r61
second (1.8 at goal 2.0): retrain the best models with the maximum ACC on all joints, more low-level speed, training
mainly clockwise, and one more model that spins in all directions.

- **ACC 254 on all 17 motors** (`acc254_all`). 254 is the largest value of the register for a ramp. The only faster
  setting, ACC 0, switches the ramp off: the motor would then accelerate as fast as its position loop and 900 mA allow,
  which no chirp recording has seen (the fits have no motor inertia or speed limit because the ramp hid them), so the
  simulation could not reproduce it. For index, middle, pinky and wrist this is x1.69 the acceleration of r60.
- **r100 = r60 + Butterworth at 5 Hz** (r60: 3 Hz). Command delay of the filter at the ~1.2 Hz gait 73 -> 36 ms, full
  amplitude at both; jitter at 8 Hz passes 10.5 % instead of 2.7 %, so r60's jerk penalty stays. Coefficients
  b = [0.292893, 0.585786, 0.292893], a = [1, 0, 0.171573].

  | Filter | gain / delay at 1.0 Hz | 1.24 Hz | 2.0 Hz | 3.0 Hz |
  |---|---|---|---|---|
  | EMA 0.3 (r34, r60e, r65) | 0.75 / 91 ms | 0.68 / 83 ms | 0.50 / 60 ms | 0.37 / 41 ms |
  | Butterworth 3 Hz (r60, r61, r100a, r100c) | 1.00 / 72 ms | 0.99 / 73 ms | 0.93 / 79 ms | 0.71 / 83 ms |
  | Butterworth 5 Hz (r100) | 1.00 / 36 ms | 1.00 / 36 ms | 0.99 / 38 ms | 0.97 / 41 ms |

- **80 % clockwise** for r100 / r100a / r100c (task option `negative_prob` 0.8; r60 / r61: 50 / 50): the hand tests
  were best CW. Evaluations report CW only and CCW only separately.
- **r100a = r60 with only the ACC (and the CW share) changed** (safety net for r100: if 5 Hz jitters on the hand).
- **r100c = r61 (camera, Task 2) with only the ACC (and the CW share) changed.**
- **r100xyz = r60 with the ACC changed, on r62's all-axes command**: one of x / y / z per command (every 8 s), either
  sign 50 / 50, 1.0-2.5 rad/s; at most 3000 iterations. r60 on this task drops 80 % at first.
- Everything else is the base model's: rewards (thumb/ring penalty relief, jerk penalty), the 2026-11-14 ramp fit,
  delays + 0-50 ms, commands 1.5-3.0 rad/s about z (r100xyz: see above), wrist 10-20 deg in training (15 on the hand), unclipped last action,
  900 mA, motor gains. `rwr_faive_lab/scripts/rsl_rl/check_r100_params.py` compares every setting of the run with the
  base run's and fails on any other difference.
- Training: fine-tunes of r60 `model_6045` / r61 `model_6095`, 16384 envs each, all four in parallel, until
  converged (at most 2000 it, r100xyz 3000), in `~/erhell/r100` on the instance, after r65's training ended
  (`start_r100.sh`, `run_r100.sh`). Evaluations include a
  stress test with ACC 200 in the sim (the real ramp 21 % slower than modeled).
- Risk to watch on the hand: the ramp model was fitted at ACC 150 and is used at 254. r60/r60e already ran 254 on thumb
  + ring successfully. If the fingers overshoot or buzz, run r100a (3 Hz) or r60.

## 6. r102 and r103 (queued 2026-11-15)

Both fine-tune r100's final checkpoint and start in parallel as soon as the trainings of r100, r100a and r100c have
ended (user: "after r100"; r100xyz may still train). Workspaces `~/erhell/r102`, `~/erhell/r103` (own code copies),
`start_r10x.sh` -> `run_r10x.sh full` (env check, smoke, params check against r100 with the real base, training until
converged, evaluations, export with deploy_meta check, video). Early smoke tests with r100's model_6250: r102 OK
(r100 on the r102 task 5.9 % drops, 2.25 rad/s).

- **r102 = r100 + penalty relief on all five fingers** (joint target rate x0.1, joint velocity / action / action rate
  x0.25; r60 / r100 had it on thumb + ring only). Deploys exactly like r100.
- **r103 = r100 on the sysID fit recorded at KP 48.** On the hand: ACC 254 (wrist 250), **KP 48** (needs the KP code of
  section 4), 5 Hz, 80 % CW, 900 mA.

**Recordings of 2026-11-15** (rwr_system `experiments/orca/chirp.py`, new options `--servo-acc`, `--kp-scale`,
`--read-servo`; settings read back before / during / after into each `metadata.json`; ACC and KP restored at the end,
also after Ctrl+C; data in `rwr_system/logs/chirp_v2/`):

| Recording | What | Result |
|---|---|---|
| `servo_settings_20261115_131602.json` | read only | KP 32, KD 32, KI 0, 897 mA on all 17; thumb + ring held ACC 254 left from a policy run (every run sets its own) |
| `20261115_131618_127536`, `..._131654_927063` | ACC 0, index_mcp, 10 s, <= 2 Hz | **looked bad on the hand: ACC 0 dropped** (user) |
| `20261115_131743_021235` | ACC 254 + KP x1.5 | stopped before moving: the wrist holds ACC 250, not 254 (recorder then fixed) |
| `20261115_131846_207422` | ACC 254 + KP 48, index_mcp, 10 s | fine |
| **`20261115_131949_871684`** | **ACC 254 (wrist 250) + KP 48, all 17, 20 s, 0.1-5 Hz, +-10 deg** | complete; vs 2026-11-13 (ACC 150, KP 32) finger gain at 2.5-5 Hz 0.82-0.89 instead of 0.63-0.79, lag 100-150 instead of 150-200 ms, no ripple increase |

Fit (instance `~/erhell/sysid_kp48/fit_full_hand_kp48.sh`, 17 joints, 0 errors; results in
`rwr_faive_lab/scripts/motor_param_characterization/logs/optimizations/right_*_20261115_132759/`): stiffness 9-59
(x1.4-2.3 of the 2026-11-14 fit on most joints), errors 0.015-0.041 (wrist 0.102). On index_mcp, middle_mcp,
middle_abd and thumb_mcp the fit put damping at ~0 and 25-30 ms delay instead (the two trade off in these data).
In `orca_v2.py` as `ORCA_V2_KP48_FIT_20261115` (tag "2026-11-15 ramp ACC 254 KP 48"); only r103 uses it.

## 6b. r104 (queued 2026-11-15): current limit 1000 mA

User: "increase the maximal current of the low level controller ... from 900 mA" (first 1050, then "make it 1000 mA"),
based on r60e, the current best, together with the maximum ACC and mostly clockwise. **r104 = r60e + 1000 mA (register
153 = 994.5 mA, +10.9 %) + ACC 254 on all 17 + 80 % CW**; EMA 0.3, KP 32, the 2026-11-14 ramp fit, rewards, delays,
1.5-3.0 rad/s and wrist as r60e. Simulation: motor torque is proportional to current and the position loop is capped
by register 44, so the fit's effort limits are scaled by 994.5 / 897 = x1.1087 on all 17 joints; nothing else changes
(free-motion chirps cannot identify the limit: the ramp limits first). `check_r104_params.py` diffs the run against
r60e and allows only these four changes. Workspace `~/erhell/r104` (r60e's model_5445 copied from `~/erhell/r60`,
md5-checked); starts with r102 / r103 when the trainings of r100 / r100a / r100c end. Smoke (16:33 UTC, 1000 mA): all
checks OK (env check, base loaded, EMA 0.3, ACC 254 on 17, params: exactly the four changes vs r60e, effort x1.10870);
r60e on the r104 task 7.0 % drops, 2.05 rad/s. On the hand: needs the hand controller restarted once (current-limit code); the node then
sets 1000 mA for r104 and 900 mA (checked) for every other model.

## 6c. r105 / r106 / r107 (queued 2026-11-15): faster always pays; on the hand exactly like their base

Both fine-tune r100a (model_6794) and change **nothing on the hand side**: deploy_meta.json must equal r100a's
(Butterworth 3 Hz, acc254_all with the wrist held at 250 by the node, KP 32, 900 mA, goal speed 1500, ramp fit
2026-11-14, 80 % CW, wrist 15, delays); the export check in `run_r105.sh` compares every field. The only new
simulation setting is `servo_max_speed` (action term): the servo ramp's reference moves at most at the motor's no-load
speed (orca_v2.py `servo_speed_limits()`: HLS3915 100 rpm -> fingers 10-14 rad/s at the joint, HLS3930 45 rpm -> wrist
4.26 rad/s). It models the real motor, which never needed it before because the capped reward never asked for more
speed; nothing is set on the hand for it (GOAL_SPEED stays 1500 = no cap).
- r105a (stage 1): rotation reward uncapped (`max_ratio` 4), motion penalties (joint_target_rate, joint_vel_l2,
  action_l2, action_rate_l2, joint_target_jerk) 0, entropy 0.005 (r100a 0.0001). r105 (stage 2, from r105a): the same
  with those penalties at 10 % of r100a's and r100a's entropy.
- r106: rotation reward = spin speed / 2.25 rad/s (fixed `speed_ref`) instead of speed / commanded speed, uncapped
  (`max_ratio` 4); penalties and entropy as r100a. On the hand the panel's speed no longer matters, only the direction.
- Every earlier task: `max_ratio` 1, `speed_ref` None, `servo_max_speed` None = the unchanged code path (diffed
  2026-11-15; new options only, off by default).
- r107 (queued with them): r100 with `max_ratio` 4 only (no speed_ref, no servo_max_speed); deploys exactly like r100
  (Butterworth 5 Hz). The motors' top speed hardly matters at ACC 254: over its full joint range a finger's ramp peaks
  at 0.52-0.89x its motor's no-load speed (only the held wrist could exceed it), so r105 / r106 have it as a safety net.

## 7. Deployment variants (a trained policy run with other servo settings)

Folder `rwr_faive_lab/logs/rsl_rl/orca_v2_repose/deploy_<name>/` with a byte-identical copy of the trained
`policy.pt` and a `deploy_meta.json` that differs only in the servo profile; the trained profile is kept in
`servo_profile.deploy_override.trained_servo_profile`, plus `deploy_variant_of` and a README.txt. In the spin panel under
"Deployment variants". These run settings the policy was NOT trained for: hand tests only, a sim says nothing about them.

| Variant | Policy | Servo setting on the hand | Trained with |
|---|---|---|---|
| r60e_max_acc (`deploy_r60e_max_acc`) | r60e model_5445 (EMA 0.3, wrist 15) | ACC 254 on all 17 (wrist held as 250) | 254 thumb + ring, 150 index / middle / pinky / wrist |
| r60e_max_acc_kp48 (`deploy_r60e_max_acc_kp48`) | r60e model_5445 (EMA 0.3, wrist 15) | ACC 254 on all 17 (wrist 250) + **KP 48** on all | as above, KP 32 |

The panel reads extra models from `rwr_system/experiments/rl_inference/spin_panel_models.json` on every page load
(r60e_max_acc_kp48 is listed there), so new variants need no panel restart.

## 8. Decided against / parked (user, 2026-11-15)

- **ACC 0** (no ramp): tried on one joint, looked bad on the hand -> not used.
- **Current limit +100 mA (r101)**: 1000 mA requested -> register 153 = 994.5 mA (+10.9 %), sim effort limits x1.109.
  Prepared (task config, check, pipeline in `~/erhell/r101`), then stopped by the user before training. Free-motion
  chirps cannot identify it (the ramp limits first). If revisited: first log the motors' present current during a
  policy run to see whether 897 mA is ever reached.
- **Faster policy rate** (20 -> 30-50 Hz): needs a retraining from scratch and a redesign (substeps, per-step reward
  weights, filters, delays, deployment rate); not started.
