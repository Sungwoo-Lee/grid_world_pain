#!/usr/bin/env bash
# Unified per-node dwell-sweep worker: EVAL ONLY (rollout -> .rec.gz recordings under a
# scratch dir). Aggregation into CSVs is done ONCE by run_sweep.py after all nodes finish
# (a single aggregation path shared by rPPO and Dreamer, via recursive glob).
#
# Promoted + unified from tmp/dist_metrics_worker.sh (rPPO) and
# tmp/dist_dreamer_worker_batched.sh (Dreamer). Both algorithms go through the SAME call
# to scripts/eval/eval_rollout.py --batched --device cpu --record (unified 2026-07-21);
# they differ only in --checkpoint form (rPPO: <run>/models/<step>, Dreamer:
# <run>/checkpoints/<episode>) and Dreamer's extra --agent_config (+ optional --episode).
#
# Usage: sweep_worker.sh <worklist_file> <node_id> <npar> <n_episodes>
#   worklist line format: CONFIG|AGENT_CONFIG_OR_-|CHECKPOINT|EPISODE_OR_-|OUTPUT_ROOT
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # scripts/eval/dwell_sweep -> repo root (3 up)
cd "$R"
PY=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python

# --- Tuning baked in from the CPU-bound eval-sweep diagnosis (see README.md) ---
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

WL="$1"; NODE="${2:-x}"; NPAR="${3:-18}"; NEP="${4:-30}"
[ -f "$WL" ] || { echo "sweep_worker.sh: worklist not found: $WL" >&2; exit 1; }

# Per-node persistent XLA compile cache: the compiled program depends only on shape,
# identical across a model's checkpoints, so eval #2..N skip the ~7s compile.
export JAX_COMPILATION_CACHE_DIR="/tmp/jaxcache_dwellsweep_$NODE"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# Worklist lives at <output_dir>/_scratch/_worklists/worklist_<node>.txt; markers go one
# level up, at <output_dir>/_scratch/_run_markers/.
MARK="$(dirname "$(dirname "$WL")")/_run_markers"; mkdir -p "$MARK"
echo "NODE=$NODE NPAR=$NPAR NEP=$NEP $(date)" > "$MARK/npar_$NODE"

export PY NEP MARK NODE
runeval() {
  IFS='|' read -r cfg agent ckpt ep out <<<"$1"
  mkdir -p "$out"
  extra=""
  [ "$agent" != "-" ] && extra="--agent_config $agent"
  [ "$ep" != "-" ] && extra="$extra --episode $ep"
  "$PY" scripts/eval/eval_rollout.py --config "$cfg" $extra --checkpoint "$ckpt" \
    --output-root "$out" --eval-n-episodes "$NEP" --record --record-n-episodes "$NEP" \
    --device cpu --quiet --batched --seed 0 >/dev/null 2>&1 \
    || echo "FAIL $1" >> "$MARK/fail_$NODE"
}
export -f runeval

t0=$(date +%s)
n=$(wc -l < "$WL")
xargs -P "$NPAR" -I@ bash -c 'runeval "$1"' _ @ < "$WL"
echo "[$NODE] done $n evals in $(( $(date +%s)-t0 ))s $(date +%H:%M:%S)" >> "$MARK/prog_$NODE"
touch "$MARK/done_$NODE"
