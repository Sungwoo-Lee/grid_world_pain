#!/usr/bin/env bash
# Per-node trajectory-collection worker, modelled on scripts/eval/dwell_sweep/sweep_worker.sh.
#
# One worklist line = ONE (run, checkpoint, BLOCK RANGE) cell, handled by one
# collect_trajectories.py process. The unit is a block RANGE, not a single block, because
# each process pays a fixed 15-17 s JAX-import + model-build + checkpoint-restore cost:
# one process per block would spend ~53 min per run on startup alone. Grouping consecutive
# blocks into one process amortises that to ~16 s per worker. This is required, not
# optional (plan §D11).
#
# Usage: collect_worker.sh <worklist_file> <node_id> <npar>
#   worklist line format:
#     RUN|CKPT|OUT_ROOT|SEED_BASE|EPISODES|BLOCK_LO:BLOCK_HI|DEVICE|BATCH_SIZE|SHARD_EPISODES|OBS_PRECISION|EXTRA
#   EXTRA is '-' or a literal flag string (e.g. '--allow-ambiguous-scene').
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # scripts/eval/traj_collect -> repo root (3 up)
cd "$R"
# Project-wide conda rule: explicit absolute interpreter path. Never `conda run`, never
# `conda activate`, never a bare `python`.
PY=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python

WL="$1"; NODE="${2:-x}"; NPAR="${3:-16}"
[ -f "$WL" ] || { echo "collect_worker.sh: worklist not found: $WL" >&2; exit 1; }

# Per-node persistent XLA compile cache: the compiled program depends only on shape, so
# every process after the first skips the ~7 s compile.
export JAX_COMPILATION_CACHE_DIR="/tmp/jaxcache_trajcollect_$NODE"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# Worklist lives at <scratch>/_worklists/worklist_<node>.txt; markers one level up.
MARK="$(dirname "$(dirname "$WL")")/_run_markers"; mkdir -p "$MARK"
rm -f "$MARK/done_$NODE"
echo "NODE=$NODE NPAR=$NPAR $(date)" > "$MARK/npar_$NODE"

export PY MARK NODE R
runcollect() {
  IFS='|' read -r run ckpt outroot seedbase episodes blocks device batch shard prec extra <<<"$1"

  # CPU/GPU knobs move in OPPOSITE directions and must be set together (plan §D7):
  # on CPU throughput comes from many single-threaded processes (L2); on GPU it comes
  # from one big vmap batch (L1), and a second JAX process on one GPU contends for
  # memory and is SLOWER, not faster.
  #
  # GPU LIMITATION, stated so nobody assumes otherwise: no CUDA_VISIBLE_DEVICES is set
  # and the spec carries no GPU-index field, so every GPU worker on a node lands on
  # GPU 0. Multi-GPU fan-out does NOT work today. Only reachable with `device: gpu`; the
  # default is cpu. Consult docs/environment/LAB_NODE_GPU_SPEC.md before using it.
  if [ "$device" = "gpu" ]; then
    export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset XLA_FLAGS
  else
    export JAX_PLATFORMS=cpu
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
  fi

  [ "$extra" = "-" ] && extra=""
  log="$MARK/log_${NODE}_$(echo "$run$blocks" | md5sum | cut -c1-10).txt"
  "$PY" scripts/eval/traj_collect/collect_trajectories.py \
      --run "$run" --checkpoint "$ckpt" --out-root "$outroot" \
      --seed-base "$seedbase" --episodes "$episodes" --blocks "$blocks" \
      --device "$device" --batch-size "$batch" --shard-episodes "$shard" \
      --obs-precision "$prec" $extra >"$log" 2>&1 \
    || echo "FAIL $1 (log: $log)" >> "$MARK/fail_$NODE"
}
export -f runcollect

t0=$(date +%s)
n=$(wc -l < "$WL")
xargs -P "$NPAR" -I@ bash -c 'runcollect "$1"' _ @ < "$WL"
echo "[$NODE] done $n cells in $(( $(date +%s)-t0 ))s $(date +%H:%M:%S)" >> "$MARK/prog_$NODE"
touch "$MARK/done_$NODE"
