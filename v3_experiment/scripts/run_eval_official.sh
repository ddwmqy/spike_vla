#!/usr/bin/env bash
# Clean-protocol evaluation of the OFFICIAL TurboVLA LIBERO checkpoint
# (DINOv3 ViT-B + BERT + ANN fusion), same protocol as run_eval.sh:
# clean LIBERO 8f1084e, FP32, seed 7, 50 trials/task, chunk 12, open-loop 12.
#
# Supports both release layouts (the HF README example is per-suite:
# checkpoints/libero/libero_object.pth):
#   single:    OFFICIAL_CKPT=<one checkpoint used for all four suites>
#   per-suite: OFFICIAL_CKPT_SPATIAL=... OFFICIAL_CKPT_OBJECT=... \
#              OFFICIAL_CKPT_GOAL=... OFFICIAL_CKPT_LIBERO_10=...
#
# WEIGHT_SOURCE=model (default; the official evaluate.py convention, raw
# model_state_dict) or ema (if the release ships an EMA copy). Recorded in
# summary.json; compare v2 against the same weight source.
#
# Caching: each suite's export and results are keyed by a fingerprint of the
# source checkpoint (path/size/mtime) + weight source + protocol constants.
# Stale exports/results from a different checkpoint are never reused.
#
# Usage:
#   OFFICIAL_CKPT=<path> [WEIGHT_SOURCE=model] EVAL_GPU_IDS=0,1,2,3 \
#   LIBERO_CHECKOUT=<clean 8f1084e checkout> ./run_eval_official.sh
set -Eeuo pipefail

EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/home/dwh/work/vla/libero_vla/resources}"
PYTHON_BIN="${PYTHON_BIN:-/home/dwh/venvs/turbovla-libero/bin/python}"
LIBERO_CHECKOUT="${LIBERO_CHECKOUT:-/home/dwh/work/vla/libero_vla/environments/LIBERO-8f1084e-clean}"

OFFICIAL_CKPT="${OFFICIAL_CKPT:-}"
declare -A SUITE_CKPTS=(
  [libero_spatial]="${OFFICIAL_CKPT_SPATIAL:-}"
  [libero_object]="${OFFICIAL_CKPT_OBJECT:-}"
  [libero_goal]="${OFFICIAL_CKPT_GOAL:-}"
  [libero_10]="${OFFICIAL_CKPT_LIBERO_10:-}"
)
WEIGHT_SOURCE="${WEIGHT_SOURCE:-model}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-${GPU_IDS:-}}"
OUT="${EVAL_OUTPUT_ROOT:-$EXPERIMENT_ROOT/output/eval/official_clean_8f1084e}"
MODEL_LABEL="${MODEL_LABEL:-Official TurboVLA LIBERO (DINOv3 ViT-B + BERT + ANN fusion)}"

DINOV3_PATH="$REPO_DIR/pretrained/dinov3-vitb16-pretrain-lvd1689m"
BERT_PATH="$REPO_DIR/pretrained/bert-base-uncased"
STATS_PATH="$REPO_DIR/experiments/libero/configs/libero_all4_stats.json"
EVAL_CKPT="$OUT/checkpoints/official_libero_export.pth"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

if [[ -n "$OFFICIAL_CKPT" ]]; then
  PER_SUITE=0
  echo "[WARN] single OFFICIAL_CKPT is used for all four suites. The official README"
  echo "[WARN] example is per-suite (checkpoints/libero/libero_object.pth); if the"
  echo "[WARN] release ships per-suite checkpoints, set OFFICIAL_CKPT_<SUITE> instead."
  [[ -f "$OFFICIAL_CKPT" ]] || die "Official checkpoint not found: $OFFICIAL_CKPT"
else
  PER_SUITE=1
  missing=()
  for suite in libero_spatial libero_object libero_goal libero_10; do
    [[ -n "${SUITE_CKPTS[$suite]}" ]] || missing+=("$suite")
  done
  (( ${#missing[@]} == 0 )) || \
    die "Set OFFICIAL_CKPT (single) or all four OFFICIAL_CKPT_<SUITE> vars; missing: ${missing[*]}"
  for suite in libero_spatial libero_object libero_goal libero_10; do
    [[ -f "${SUITE_CKPTS[$suite]}" ]] || die "Official checkpoint not found: ${SUITE_CKPTS[$suite]}"
  done
fi

[[ -n "$EVAL_GPU_IDS" ]] || die "Set EVAL_GPU_IDS to four free GPUs, for example 0,1,2,3"
[[ -x "$PYTHON_BIN" ]] || die "Python not executable: $PYTHON_BIN"
[[ -f "$STATS_PATH" ]] || die "Required file not found: $STATS_PATH"
for path in "$BERT_PATH" "$DINOV3_PATH" "$LIBERO_CHECKOUT"; do
  [[ -d "$path" ]] || die "Required directory not found: $path"
done
case "$WEIGHT_SOURCE" in
  model|ema) ;;
  *) die "WEIGHT_SOURCE must be 'model' or 'ema', got: $WEIGHT_SOURCE" ;;
esac

EVAL_GPU_IDS="${EVAL_GPU_IDS//[[:space:]]/}"
IFS=',' read -r -a GPU_ARRAY <<< "$EVAL_GPU_IDS"
(( ${#GPU_ARRAY[@]} == 4 )) || die "EVAL_GPU_IDS must contain exactly four GPUs"

suite_ckpt() {
  local suite="$1"
  if (( PER_SUITE )); then printf '%s' "${SUITE_CKPTS[$suite]}"; else printf '%s' "$OFFICIAL_CKPT"; fi
}
suite_export() {
  local suite="$1"
  if (( PER_SUITE )); then printf '%s' "$OUT/checkpoints/official_${suite}.pth"; else printf '%s' "$EVAL_CKPT"; fi
}
# Identity of everything that must match for a cached artifact to be reused:
# source checkpoint (path/size/mtime) + weight source + protocol constants +
# a hash over the evaluator/converter/policy/rollout/stats content. Known
# limitation: model-weight CONTENT is not hashed (size+mtime only).
PROTOCOL_SIG="seed=7|trials=50|chunk=12|openloop=12|precision=fp32|gl=egl|commit=${LIBERO_COMMIT}"
code_fp() {
  {
    printf '%s\n' "$PROTOCOL_SIG"
    sha256sum "$STATS_PATH" "$EXPERIMENT_ROOT/evaluate.py" \
      "$EXPERIMENT_ROOT/convert_official_ckpt.py" \
      "$EXPERIMENT_ROOT/code/turbovla/evaluation/policy.py" \
      "$EXPERIMENT_ROOT/third_party/vla_adapter/vla_adapter/rollout.py" 2>/dev/null | awk '{print $1}'
  } | sha256sum | cut -d' ' -f1
}
CODE_FP="$(code_fp)"
artifact_fp() {
  local src="$1"
  printf '%s|%s|%s|%s|%s' "$src" "$(stat -c '%s %Y' "$src")" "$WEIGHT_SOURCE" "$LIBERO_COMMIT" "$CODE_FP"
}

mkdir -p "$OUT/checkpoints" "$OUT/logs/dry_run" "$OUT/results" "$OUT/videos" "$OUT/libero_config"
export REPO_DIR
export PYTHONPATH="$EXPERIMENT_ROOT/code:$EXPERIMENT_ROOT/third_party/vla_adapter:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LIBERO_CONFIG_PATH="$OUT/libero_config"

"$PYTHON_BIN" - "$LIBERO_CHECKOUT" "$OUT/libero_config/config.yaml" >"$OUT/logs/environment_preflight.log" <<'PY'
from pathlib import Path
import subprocess
import sys

checkout = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(["git", "-C", str(checkout), "status", "--short"], text=True).strip()
if commit != "8f1084e3132a39270c3a13ebe37270a43ece2a01":
    raise SystemExit(f"unexpected LIBERO commit: {commit}")
if status:
    raise SystemExit(f"LIBERO checkout is dirty: {status}")
libero_root = checkout / "libero"
paths = {
    "assets": libero_root / "libero" / "assets",
    "bddl_files": libero_root / "libero" / "bddl_files",
    "benchmark_root": libero_root / "libero",
    "datasets": libero_root / "datasets",
    "init_states": libero_root / "libero" / "init_files",
}
for name, path in paths.items():
    if name != "datasets" and not path.exists():
        raise SystemExit(f"missing LIBERO path: {name}={path}")
config_path.write_text("".join(f"{name}: {path}\n" for name, path in paths.items()))
print(f"libero_commit={commit}")
PY

suites=(libero_spatial libero_object libero_goal libero_10)

echo "[$(date -Is)] CONVERT / fingerprint exports"
for suite in "${suites[@]}"; do
  src="$(suite_ckpt "$suite")"
  dst="$(suite_export "$suite")"
  fp="$(artifact_fp "$src")"
  fp_file="$dst.src_fp"
  if [[ -s "$dst" && -s "$fp_file" && "$(cat "$fp_file")" == "$fp" ]]; then
    echo "[$(date -Is)] SKIP convert suite=$suite (fingerprint matches)"
    continue
  fi
  echo "[$(date -Is)] CONVERT suite=$suite src=$src -> $dst (weight_source=$WEIGHT_SOURCE)"
  "$PYTHON_BIN" "$EXPERIMENT_ROOT/convert_official_ckpt.py" \
    --src "$src" --dst "$dst" --weight_source "$WEIGHT_SOURCE" \
    >"$OUT/logs/convert_${suite}.log" 2>&1 \
    || { echo "[ERROR] conversion failed, see $OUT/logs/convert_${suite}.log"; tail -20 "$OUT/logs/convert_${suite}.log"; exit 1; }
  printf '%s' "$fp" > "$fp_file"
done

common_args=(
  --dinov3_path "$DINOV3_PATH"
  --bert_path "$BERT_PATH"
  --stats_path "$STATS_PATH"
  --stats_key libero_all4_no_noops
  --libero_root "$LIBERO_CHECKOUT"
  --num_trials_per_task 50
  --chunk_size 12
  --num_open_loop_steps 12
  --seed 7
  --precision fp32
  --allow_hf_download false
  --save_video true
  --max_videos_per_task 2
  --video_out_path "$OUT/videos"
  --mujoco_gl egl
  --pyopengl_platform egl
)

first_export="$(suite_export libero_spatial)"
echo "[$(date -Is)] CHECK strict model load"
CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_BIN" "$EXPERIMENT_ROOT/evaluate.py" \
  --task_suite_name libero_spatial \
  --dry_run_model_load true \
  --ckpt_path "$first_export" \
  "${common_args[@]}" >"$OUT/logs/dry_run/model_load.log" 2>&1

run_suite() {
  local suite="$1" gpu="$2"
  local result="$OUT/results/$suite.json"
  local log="$OUT/logs/$suite.log"
  local export_fp
  export_fp="$(artifact_fp "$(suite_ckpt "$suite")")"
  local fp_file="$result.fp"
  if [[ -s "$result" && -s "$fp_file" && "$(cat "$fp_file")" == "$export_fp" ]]; then
    echo "[$(date -Is)] SKIP suite=$suite (result fingerprint matches)"
    return 0
  fi
  if [[ -s "$result" ]]; then
    echo "[$(date -Is)] STALE result suite=$suite (fingerprint changed); re-running"
  fi
  echo "[$(date -Is)] START gpu=$gpu suite=$suite"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$EXPERIMENT_ROOT/evaluate.py" \
    --task_suite_name "$suite" \
    --result_json_path "$result" \
    --ckpt_path "$(suite_export "$suite")" \
    "${common_args[@]}" >"$log" 2>&1
  printf '%s' "$export_fp" > "$fp_file"
  echo "[$(date -Is)] DONE gpu=$gpu suite=$suite"
}

pids=()
suite_names=()
for index in "${!suites[@]}"; do
  run_suite "${suites[$index]}" "${GPU_ARRAY[$index]}" &
  pids+=("$!")
  suite_names+=("${suites[$index]}")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"

# Pre-summarize integrity gate: every suite result must exist AND carry a
# fingerprint matching the current run; otherwise refuse to summarize.
for suite in "${suites[@]}"; do
  result="$OUT/results/$suite.json"
  fp_file="$result.fp"
  expected_fp="$(artifact_fp "$(suite_ckpt "$suite")")"
  [[ -s "$result" ]] || die "missing result: $result"
  [[ -s "$fp_file" ]] || die "missing fingerprint sidecar: $fp_file"
  [[ "$(cat "$fp_file")" == "$expected_fp" ]] || die "stale fingerprint for $suite (result does not belong to this run)"
done

"$PYTHON_BIN" - "$OUT/results" "$first_export" "$MODEL_LABEL" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
label = sys.argv[3]
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIALS, CHUNK, OPEN_LOOP, SEED = 50, 12, 12, 7

def check(item, suite):
    problems = []
    if item.get("task_suite_name") != suite:
        problems.append(f"task_suite_name={item.get('task_suite_name')!r} != {suite!r}")
    for key, expected in (
        ("seed", SEED), ("precision", "fp32"),
        ("num_trials_per_task", TRIALS), ("num_open_loop_steps", OPEN_LOOP),
    ):
        if item.get(key) != expected:
            problems.append(f"{key}={item.get(key)!r} != {expected!r}")
    tasks = item.get("tasks") or []
    if len(tasks) != 10:
        problems.append(f"{len(tasks)} tasks != 10")
    for task in tasks:
        if int(task.get("episodes", -1)) != TRIALS:
            problems.append(f"task {task.get('task_id')}: episodes={task.get('episodes')} != {TRIALS}")
    if int(item.get("total_episodes", -1)) != 10 * TRIALS:
        problems.append(f"total_episodes={item.get('total_episodes')} != {10 * TRIALS}")
    if sum(int(t.get("episodes", 0)) for t in tasks) != int(item.get("total_episodes", -1)):
        problems.append("tasks[] episodes do not sum to total_episodes")
    return problems

payloads = {}
errors = []
for suite in suites:
    item = json.loads((root / f"{suite}.json").read_text())
    problems = check(item, suite)
    if problems:
        errors.extend(f"{suite}: {p}" for p in problems)
    payloads[suite] = item
if errors:
    raise SystemExit("result integrity check failed:\n  " + "\n  ".join(errors))

successes = sum(int(item["total_successes"]) for item in payloads.values())
episodes = sum(int(item["total_episodes"]) for item in payloads.values())
if episodes != 2000:
    raise SystemExit(f"total episodes {episodes} != 2000")
summary = {
    "model": label,
    "libero_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "weight_source": checkpoint.get("selected_weight_source", "model_state_dict"),
    "precision": "fp32",
    "seed": SEED,
    "num_trials_per_task": TRIALS,
    "chunk_size": CHUNK,
    "num_open_loop_steps": OPEN_LOOP,
    "total_successes": successes,
    "total_episodes": episodes,
    "success_rate": successes / episodes,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[$(date -Is)] All suites complete"
