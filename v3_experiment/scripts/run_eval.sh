#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/home/dwh/work/vla/libero_vla/resources}"
PYTHON_BIN="${PYTHON_BIN:-/home/dwh/venvs/turbovla-libero/bin/python}"
LIBERO_CHECKOUT="${LIBERO_CHECKOUT:-}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-}"
TRAIN_STEP="${TRAIN_STEP:-80000}"
# Weight source to export/evaluate. Default ema (the snnve lineage ships EMA);
# the official anchor release has NO EMA (raw model_state_dict only), so the
# primary v2-vs-anchor comparison must use WEIGHT_SOURCE=raw on both sides.
WEIGHT_SOURCE="${WEIGHT_SOURCE:-ema}"
SOURCE_CKPT="${SOURCE_CKPT:-$PACKAGE_ROOT/output/train/checkpoints/turbovla_sdtv3_bert_snnfusion_learned_${TRAIN_STEP}.pth}"
OUT="${EVAL_OUTPUT_ROOT:-$PACKAGE_ROOT/output/eval/step_${TRAIN_STEP}_${WEIGHT_SOURCE}_clean_8f1084e}"

DINOV3_PATH="$REPO_DIR/pretrained/dinov3-vitb16-pretrain-lvd1689m"
BERT_PATH="${BERT_PATH:-$REPO_DIR/pretrained/bert-base-uncased}"
# Text weights dir: point BERT_PATH at $REPO_DIR/pretrained/SpikingLM when the
# checkpoint's text encoder is SpikingLM, or at .../pretrained/SmoothSpike/
# smoothspike-bert-base-fused for sootspike (the export step rewrites the stored
# text path to BERT_PATH in both cases).
# v3 addition: TEXT_SOURCE_PATH = patched SmoothSpike source tree, rewritten
# into the checkpoint for sootspike (portability). Leave empty for SpikingLM.
TEXT_SOURCE_PATH="${TEXT_SOURCE_PATH:-}"
SDTV3_CHECKPOINT="$REPO_DIR/pretrained/V3_19.0M_1x4.pth"
SDTV3_SOURCE="$REPO_DIR/third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py"
SPIKINGLM_SOURCE="$REPO_DIR/third_party/SpikingLM"
STATS_PATH="$REPO_DIR/experiments/libero/configs/libero_all4_stats.json"
EVAL_CKPT="$OUT/checkpoints/turbovla_sdtv3_bert_snnfusion_learned_${TRAIN_STEP}_${WEIGHT_SOURCE}.pth"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ -n "$LIBERO_CHECKOUT" ]] || die "Set LIBERO_CHECKOUT to a clean LIBERO commit 8f1084e checkout"
[[ -n "$EVAL_GPU_IDS" ]] || die "Set EVAL_GPU_IDS to four free GPUs, for example 0,1,2,3"
[[ -x "$PYTHON_BIN" ]] || die "Python not executable: $PYTHON_BIN"
for path in "$SOURCE_CKPT" "$SDTV3_CHECKPOINT" "$SDTV3_SOURCE" "$STATS_PATH"; do
  [[ -f "$path" ]] || die "Required file not found: $path"
done
for path in "$BERT_PATH" "$DINOV3_PATH" "$LIBERO_CHECKOUT"; do
  [[ -d "$path" ]] || die "Required directory not found: $path"
done

EVAL_GPU_IDS="${EVAL_GPU_IDS//[[:space:]]/}"
IFS=',' read -r -a GPU_ARRAY <<< "$EVAL_GPU_IDS"
# Four GPUs run the four suites in parallel; a single GPU runs them serially
# (slower but sufficient - e.g. one MIG slice on the dev pod).
(( ${#GPU_ARRAY[@]} == 4 || ${#GPU_ARRAY[@]} == 1 )) || die "EVAL_GPU_IDS must contain exactly one GPU (serial) or four GPUs (parallel)"
# Subset runs (e.g. smoke one suite first): default is the full protocol set.
EVAL_SUITES="${EVAL_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"

mkdir -p "$OUT/checkpoints" "$OUT/logs/dry_run" "$OUT/results" "$OUT/videos" "$OUT/libero_config"
export REPO_DIR
export PYTHONPATH="$PACKAGE_ROOT/code:$PACKAGE_ROOT/third_party/vla_adapter:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
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
print(f"libero_status={status!r}")
for name, path in paths.items():
    print(f"{name}={path}")
PY

# Identity of everything that must match for the cached export / results to be
# reused: source ckpt (path/size/mtime), text weights dir, protocol constants,
# and a hash over the evaluator/policy/rollout/stats content. Known limitation:
# model-weight CONTENT is not hashed (size+mtime only).
PROTOCOL_SIG="seed=7|trials=50|chunk=12|openloop=12|precision=fp32|gl=egl|commit=${LIBERO_COMMIT}"
code_fp() {
  {
    printf '%s\n' "$PROTOCOL_SIG"
    sha256sum "$STATS_PATH" "$PACKAGE_ROOT/evaluate.py" \
      "$PACKAGE_ROOT/code/turbovla/evaluation/policy.py" \
      "$PACKAGE_ROOT/third_party/vla_adapter/vla_adapter/rollout.py" 2>/dev/null | awk '{print $1}'
  } | sha256sum | cut -d' ' -f1
}
CODE_FP="$(code_fp)"
source_fp() {
  # TEXT_SOURCE_PATH only participates for sootspike runs, so v2 fingerprints
  # (TEXT_SOURCE_PATH empty) stay byte-identical to their pre-v3 values.
  local extra=""
  if [[ -n "$TEXT_SOURCE_PATH" ]]; then extra="|$TEXT_SOURCE_PATH"; fi
  printf '%s|%s|%s|%s|%s|%s%s' "$SOURCE_CKPT" "$(stat -c '%s %Y' "$SOURCE_CKPT")" "$BERT_PATH" "$LIBERO_COMMIT" "$CODE_FP" "$WEIGHT_SOURCE" "$extra"
}

export_fp="$(source_fp)"
export_fp_file="$EVAL_CKPT.src_fp"
if [[ ! -s "$EVAL_CKPT" || ! -s "$export_fp_file" || "$(cat "$export_fp_file")" != "$export_fp" ]]; then
  [[ -d "$SPIKINGLM_SOURCE" ]] || SPIKINGLM_SOURCE=""
  "$PYTHON_BIN" - "$SOURCE_CKPT" "$EVAL_CKPT" "$BERT_PATH" "$SDTV3_CHECKPOINT" "$SDTV3_SOURCE" "$SPIKINGLM_SOURCE" "$WEIGHT_SOURCE" "$TEXT_SOURCE_PATH" <<'PY'
import os
import sys
import torch

source, destination, bert_path, sdt_path, sdt_source, spikinglm_source, weight_source, text_source = sys.argv[1:9]
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
state_key = "ema_model_state_dict" if weight_source == "ema" else "model_state_dict"
state = checkpoint.get(state_key)
if not isinstance(state, dict) or not state:
    raise KeyError(f"{state_key} is missing or empty (WEIGHT_SOURCE={weight_source})")
model_config = checkpoint["model_config"]
model_config["text"]["model_name_or_path"] = bert_path
# The stored source path may point at the training machine; rewrite it to the
# local SpikingLM source tree so the export is portable.
if spikinglm_source and model_config.get("text", {}).get("encoder_type") == "spikinglm":
    model_config["text"]["model_source_path"] = spikinglm_source
if text_source and model_config.get("text", {}).get("encoder_type") == "sootspike":
    model_config["text"]["model_source_path"] = text_source
model_config["vision"]["model_name_or_path"] = sdt_path
model_config["vision"]["pretrained_checkpoint"] = sdt_path
model_config["vision"]["model_source_path"] = sdt_source
payload = {
    "model_state_dict": state,
    "selected_weight_source": weight_source,
    "model_config": model_config,
}
for key in ("global_step", "loss", "ema_decay", "distillation"):
    if key in checkpoint:
        payload[key] = checkpoint[key]
temporary = destination + ".tmp"
torch.save(payload, temporary)
os.replace(temporary, destination)
print(f"saved {destination}: {len(state)} tensors")
PY
  printf '%s' "$export_fp" > "$export_fp_file"
fi

common_args=(
  --ckpt_path "$EVAL_CKPT"
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

echo "[$(date -Is)] CHECK strict model load"
CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_BIN" "$PACKAGE_ROOT/evaluate.py" \
  --task_suite_name libero_spatial \
  --dry_run_model_load true \
  "${common_args[@]}" >"$OUT/logs/dry_run/model_load.log" 2>&1

IFS=',' read -r -a suites <<< "$EVAL_SUITES"
run_suite() {
  local suite="$1" gpu="$2"
  local result="$OUT/results/$suite.json"
  local log="$OUT/logs/$suite.log"
  local result_fp_file="$result.fp"
  local result_fp
  result_fp="$(source_fp)"
  if [[ -s "$result" && -s "$result_fp_file" && "$(cat "$result_fp_file")" == "$result_fp" ]]; then
    echo "[$(date -Is)] SKIP suite=$suite (result fingerprint matches)"
    return 0
  fi
  if [[ -s "$result" ]]; then
    echo "[$(date -Is)] STALE result suite=$suite (fingerprint changed); re-running"
  fi
  echo "[$(date -Is)] START gpu=$gpu suite=$suite"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$PACKAGE_ROOT/evaluate.py" \
    --task_suite_name "$suite" \
    --result_json_path "$result" \
    "${common_args[@]}" >"$log" 2>&1
  if (( $? != 0 )); then
    echo "[$(date -Is)] FAIL gpu=$gpu suite=$suite (see $log)"
    return 1
  fi
  printf '%s' "$result_fp" > "$result_fp_file"
  echo "[$(date -Is)] DONE gpu=$gpu suite=$suite"
}

status=0
if (( ${#GPU_ARRAY[@]} == 4 )); then
  pids=()
  for index in "${!suites[@]}"; do
    run_suite "${suites[$index]}" "${GPU_ARRAY[index % 4]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
else
  # Single GPU: run the suites one after another on that GPU.
  for suite in "${suites[@]}"; do
    run_suite "$suite" "${GPU_ARRAY[0]}" || status=1
  done
fi
(( status == 0 )) || exit "$status"

# Pre-summarize integrity gate: every suite result must exist AND carry a
# fingerprint matching the current run.
for suite in "${suites[@]}"; do
  result="$OUT/results/$suite.json"
  fp_file="$result.fp"
  [[ -s "$result" ]] || die "missing result: $result"
  [[ -s "$fp_file" ]] || die "missing fingerprint sidecar: $fp_file"
  [[ "$(cat "$fp_file")" == "$(source_fp)" ]] || die "stale fingerprint for $suite (result does not belong to this run)"
done

"$PYTHON_BIN" - "$OUT/results" "$EVAL_CKPT" "$EVAL_SUITES" <<'PY'
import json
import os
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
suites = tuple(sys.argv[3].split(","))
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
expected_episodes = 10 * TRIALS * len(suites)
if episodes != expected_episodes:
    raise SystemExit(f"total episodes {episodes} != {expected_episodes}")
summary = {
    "model": os.environ.get(
        "MODEL_LABEL",
        "SDT-V3 19M + SNN fusion (set MODEL_LABEL to describe the text encoder precisely)",
    ),
    "suites": list(suites),
    "checkpoint_step": int(checkpoint["global_step"]),
    "weight_source": checkpoint.get("selected_weight_source", "unknown"),
    "training_loss_at_checkpoint": float(checkpoint["loss"]),
    "libero_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
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
