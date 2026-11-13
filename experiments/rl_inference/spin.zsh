# Real-hand runs of the joint-only cube-spin policies (no camera).
# Load once per terminal (after the ROS setup):  source ~/Documents/RWR/spin.zsh
#
# One run:
#   spin <model> <speed rad/s> [ccw|cw|flip|flip<seconds>] [real|dry] [clipped]
#
#   spin r31 1.5              dry run, counterclockwise (seen from above), 1.5 rad/s
#   spin r31 1.5 ccw real     real run, counterclockwise
#   spin r31 2.0 cw real      real run, clockwise
#   spin r31 1.5 flip real    real run, switches direction every 8 s (as in training)
#   spin r31 1.5 flip4 real   switches every 4 s
#   The policy's previous actions are fed back unclipped, as before 2026-11-11 (default of the node since 11-13).
#   "clipped" = clipped to [-1, 1] like the training (the node of 11-11 to 11-13). "old" is accepted = the default.
#   Runs until Ctrl+C; SPIN_SECONDS=30 spin ... stops by itself after 30 s.
#
# A test series (every model x speed x direction, real runs):
#   spin_all [seconds per run] [clipped] [models...]
#
#   spin_all                  all 7 models, speeds 1.0 1.3 1.6 2.0, ccw and cw, 30 s each (56 runs)
#   spin_all 20 r31 r28       only r31 and r28, 20 s each
#   spin_all 30 clipped r21   r21 with clipped previous actions (as in training)
#   SPIN_SPEEDS="1.5 2.0" spin_all 30 r31     other speeds
#   Before each run: Enter = start, s = skip, q = quit. Ctrl+C ends a run early (e.g. the cube fell).
#   After each run: number of drops and a note (Enter = 0 / none), r = repeat the run.
#   Summary of the series: rwr_system/logs/policy_runs/spin_all_<date>.csv
#
# Models: r31 r33B r30 r29 r28 r21 (r21_sysid_16k) r33A. Without "real" spin is a dry run (nothing is sent to the hand).
# Every run logs to rwr_system/logs/policy_runs/<model>_<direction>_<speed>[_clipped][_dry]_<date>.csv.

SPIN_RUNS=~/Documents/RWR/rwr_faive_lab/logs/rsl_rl/orca_v2_repose
SPIN_LOGS=~/Documents/RWR/rwr_system/logs/policy_runs
SPIN_MODELS=(r31 r33B r30 r29 r28 r21 r33A)

spin() {
  local model=$1 speed=${2#-} dir=${3:-ccw} mode=${4:-dry} variant=${5:-} run
  case $model in
    r31)               run=2026-11-11_15-44-24_r31 ;;
    r33B)              run=2026-11-13_02-59-19_r33B ;;
    r33A)              run=2026-11-12_15-58-48_r33A ;;
    r30)               run=2026-11-10_18-39-03_r30 ;;
    r29)               run=2026-11-11_00-15-49_r29 ;;
    r28)               run=2026-11-09_20-39-13_r28 ;;
    r21|r21_sysid_16k) run=2026-11-08_20-44-06_r21_sysid_16k ;;
    *) echo "usage: spin <r31|r33B|r30|r29|r28|r21|r33A> <speed> [ccw|cw|flip|flip<s>] [real] [clipped]"; return 1 ;;
  esac
  if [[ ! $speed =~ '^[0-9]+(\.[0-9]+)?$' ]]; then echo "speed must be a number, e.g. 1.5"; return 1; fi
  local goal=$speed flip=0
  case $dir in
    ccw)    ;;
    cw)     goal=-$speed ;;
    flip)   flip=8 ;;
    flip*)  flip=${dir#flip}
            if [[ ! $flip =~ '^[0-9]+(\.[0-9]+)?$' ]]; then echo "flip period must be a number, e.g. flip4"; return 1; fi ;;
    *) echo "direction must be ccw, cw, flip or flip<seconds>"; return 1 ;;
  esac
  local dry=true
  case $mode in
    real) dry=false ;;
    dry)  ;;
    *) echo "4th argument must be 'real' or 'dry' (nothing = dry run)"; return 1 ;;
  esac
  local clip=false
  case $variant in
    clipped) clip=true ;;
    old|"") ;;
    *) echo "5th argument must be 'clipped' (previous actions clipped as in training) or nothing"; return 1 ;;
  esac
  local secs=${SPIN_SECONDS:-0}
  if [[ ! $secs =~ '^[0-9]+(\.[0-9]+)?$' ]]; then echo "SPIN_SECONDS must be a number"; return 1; fi
  # ROS parameters declared as floats reject "2", so always pass decimals
  goal=$(printf '%.2f' $goal); flip=$(printf '%.1f' $flip); secs=$(printf '%.1f' $secs)
  local csv=$SPIN_LOGS/${model}_${dir}_${speed}$([[ $clip == true ]] && echo _clipped)$([[ $dry == true ]] && echo _dry)_$(date +%Y%m%d_%H%M%S).csv
  SPIN_LAST_CSV=$csv
  # quoted: zsh would expand ":=/path" inside an array assignment
  local args=(
    -p "policy_ckpt_path:=$SPIN_RUNS/$run/exported/policy.pt"
    -p "goal_ang_vel:=$goal"
    -p "goal_flip_period_s:=$flip"
    -p "clip_last_action:=$clip"
    -p "run_seconds:=$secs"
    -p "csv_log_path:=$csv"
    -p "dry_run:=$dry"
  )
  echo "spin: $model ($run), goal $goal rad/s, flip every $flip s (0 = never), dry_run=$dry, clip_last_action=$clip," \
       "run ${secs} s (0 = until Ctrl+C)"
  echo "      log $csv"
  ros2 run experiments policy_inference_cube_spin_v2.py --ros-args "${args[@]}"
}

spin_all() {
  local secs=30 variant=
  if [[ $1 =~ '^[0-9]+(\.[0-9]+)?$' ]]; then secs=$1; shift; fi
  if [[ $1 == clipped || $1 == old ]]; then [[ $1 == clipped ]] && variant=clipped; shift; fi
  local models=("$@")
  (( $#models )) || models=($SPIN_MODELS)
  local m
  for m in $models; do
    (( ${SPIN_MODELS[(Ie)$m]} )) || [[ $m == r21_sysid_16k ]] || { echo "unknown model $m (models: $SPIN_MODELS)"; return 1; }
  done
  local speeds=(${=SPIN_SPEEDS:-1.0 1.3 1.6 2.0}) dirs=(ccw cw)
  local n=$(( $#models * $#speeds * $#dirs )) i=0
  local summary=$SPIN_LOGS/spin_all_$(date +%Y%m%d_%H%M%S).csv
  echo "time,model,direction,speed,variant,run_seconds,drops,note,csv" > $summary
  echo "spin_all: $n real runs of $secs s (previous actions ${variant:-unclipped}): models $models, speeds $speeds, ccw + cw"
  echo "          summary $summary"
  local s d key drops note
  for m in $models; do
    for s in $speeds; do
      for d in $dirs; do
        (( i++ ))
        while true; do
          echo
          echo "===== [$i/$n] $m  $d  $s rad/s  ($secs s) ====="
          read -r "key?Place the cube. Enter = start, s = skip, q = quit: "
          if [[ $key == q ]]; then echo "stopped after $((i - 1)) of $n runs; summary $summary"; return 0; fi
          if [[ $key == s ]]; then echo "$(date +%T),$m,$d,$s,${variant:-unclipped},$secs,skipped,,"  >> $summary; break; fi
          SPIN_SECONDS=$secs spin $m $s $d real $variant
          read -r "drops?Drops (number, Enter = 0; r = repeat this run): "
          [[ $drops == r ]] && continue
          read -r "note?Note (Enter = none): "
          echo "$(date +%T),$m,$d,$s,${variant:-unclipped},$secs,${drops:-0},\"${note//\"/\'}\",$SPIN_LAST_CSV" >> $summary
          break
        done
      done
    done
  done
  echo "all $n runs done; summary $summary"
}
