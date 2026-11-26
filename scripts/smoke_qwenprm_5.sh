#!/bin/bash -l
set -euo pipefail

# ---------------------------------------------------------------------------
# Interactive 5-problem smoke test for llama-3b + Qwen PRM, using the REAL
# chunk-queue main.py (no code changes). Stands up a one-chunk queue, sets the
# SSDP_* env vars exactly like ssdp_debugscaling.pbs, runs, then evaluates
# accuracy (main.py itself only dumps results-*.json, not accuracy).
#
# Run inside an interactive allocation with >=2 GPUs (main.py pins gen->cuda:0,
# PRM->cuda:1):
#   qsub -I -q debug -l select=1:system=polaris:ngpus=4 -l walltime=01:00:00 \
#        -l filesystems=home:grand -A VeloC
#   cd /lus/grand/projects/VeloC/atkia/ssdp/SSDP/scripts
#   ./smoke_qwenprm_5.sh
# ---------------------------------------------------------------------------

REPO=/lus/grand/projects/VeloC/atkia/ssdp/SSDP
cd "$REPO"

MODEL="llama-3b"
REWARD_MODEL="qwen_prm-7b"
CFG="configs/inference/DPTS.json"
DATASET_NAME="math"
NUM_PROBLEMS="${1:-5}"
BUDGET="${2:-3000}"          # single budget for the smoke chunk

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$REPO/scripts/runlogs"
mkdir -p "$LOGDIR"
RUNLOG="$LOGDIR/smoke_qwenprm.${STAMP}.log"

# -------- env (mirrors ssdp_debugscaling.pbs / ssdp_dpts_4runs_8gpu.pbs) --------
module use /soft/modulefiles
module load conda
conda activate ssdp-paper

export HF_HOME=/lus/grand/projects/VeloC/atkia/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export HF_TOKEN="$(cat $HOME/.secrets/hf_token)"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BENCHMARK_ROOT="$REPO/benchmark"

# main.py hard-pins gen_device=cuda:0, aux_device=cuda:1, so expose >=2 GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# -------- build a one-chunk queue on Lustre --------
QDIR="$REPO/results/_queue/_smoke_${STAMP}"
RESULTS_DIR="$REPO/results/_smoke_${STAMP}"
mkdir -p "$QDIR/claim" "$QDIR/done" "$RESULTS_DIR"

# chunks.txt line format (from main.py): "cid budget start end"
CID="smoke_b${BUDGET}_000"
printf "%s %d %d %d\n" "$CID" "$BUDGET" 0 "$NUM_PROBLEMS" > "$QDIR/chunks.txt"
echo "[QUEUE] built 1 chunk: $(cat "$QDIR/chunks.txt")"

# -------- SSDP_* env exactly as the launcher sets them, but tuned for smoke --------
export SSDP_QDIR="$QDIR"
export SSDP_RESULTS="$RESULTS_DIR"
export SSDP_WORKER_RANK=0
# Give plenty of time and DON'T let MIN_LEFT_S refuse to start the chunk:
export SSDP_DEADLINE_S=3300
export SSDP_MIN_LEFT_S=60         # must be < DEADLINE_S so the single chunk starts

echo "===== SMOKE: $MODEL + $REWARD_MODEL, $NUM_PROBLEMS MATH problems, budget=$BUDGET =====" | tee "$RUNLOG"
echo "[QDIR]    $SSDP_QDIR"        | tee -a "$RUNLOG"
echo "[RESULTS] $SSDP_RESULTS"     | tee -a "$RUNLOG"
echo "[CUDA]    $CUDA_VISIBLE_DEVICES" | tee -a "$RUNLOG"
echo "[TIME]    $(date -Is)"       | tee -a "$RUNLOG"
echo "==============================================================================" | tee -a "$RUNLOG"

# Keep Accelerate single-process (strip any stray distributed launch vars).
unset PMI_RANK PMI_SIZE PMI_FD 2>/dev/null || true
unset PMIX_RANK PMIX_SERVER_URI2 PMIX_SERVER_URI3 2>/dev/null || true
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_RANK 2>/dev/null || true
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK 2>/dev/null || true

# -------- run the real main.py --------
python3 -u main.py \
  --config "$CFG" \
  --work-dir "$RESULTS_DIR" \
  --exp-name "smoke_${STAMP}" \
  --data "$DATASET_NAME" \
  --model "$MODEL" \
  --reward_model "$REWARD_MODEL" \
  --dtype bfloat16 \
  2>&1 | tee -a "$RUNLOG"

# -------- locate the chunk's results (RESULTS/MODEL_TAG/b{budget}/chunks/{cid}) --------
CHUNK_DIR="$RESULTS_DIR/$MODEL/b${BUDGET}/chunks/$CID"
echo | tee -a "$RUNLOG"
echo "===== RESULTS =====" | tee -a "$RUNLOG"
echo "[CHUNK_DIR] $CHUNK_DIR" | tee -a "$RUNLOG"

if [[ ! -d "$CHUNK_DIR" ]]; then
  echo "[WARN] chunk dir not found - the chunk may not have started (check MIN_LEFT_S) or main.py errored. See $RUNLOG." | tee -a "$RUNLOG"
  exit 1
fi

ls -l "$CHUNK_DIR" | tee -a "$RUNLOG"

# -------- evaluate accuracy (main.py doesn't; dataset.evaluate_results does) --------
python3 - "$CHUNK_DIR" "$DATASET_NAME" <<'PY' 2>&1 | tee -a "$RUNLOG"
import sys, os, glob, json
sys.path.insert(0, os.getcwd())
from inferenceKit.data_config import supported_dataset

chunk_dir, data_name = sys.argv[1], sys.argv[2]
dataset = supported_dataset[data_name]()

result_files = sorted(glob.glob(os.path.join(chunk_dir, "results-*.json")))
if not result_files:
    print("[WARN] no results-*.json in chunk dir - generation may have produced nothing.")
    sys.exit(1)

for rf in result_files:
    key = os.path.basename(rf)[len("results-"):-len(".json")]
    with open(rf) as f:
        results = json.load(f)
    # quick visibility: did generation actually produce non-empty responses?
    nonempty = sum(1 for r in results if r.get("response", "").strip().split("assistant")[-1].strip())
    try:
        ev = dataset.evaluate_results(results)
        print(f"[{key}] n={len(results)} nonempty_responses={nonempty} eval={json.dumps(ev)}")
    except Exception as ex:
        print(f"[{key}] n={len(results)} nonempty_responses={nonempty} eval_FAILED={ex}")
PY

echo | tee -a "$RUNLOG"
echo "[DONE] $(date -Is)" | tee -a "$RUNLOG"
echo "[LOG]  $RUNLOG"
echo "[CLEANUP] rm -rf $QDIR $RESULTS_DIR   # when finished inspecting"
