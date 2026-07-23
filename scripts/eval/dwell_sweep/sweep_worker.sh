#!/usr/bin/env bash
# Unified per-node dwell-sweep worker: EVAL ONLY (rollout -> .rec.gz recordings under a
# scratch dir). Aggregation into CSVs is done ONCE by run_sweep.py after all nodes finish
# (a single aggregation path shared by rPPO and Dreamer, via recursive glob).
#
# Promoted + unified from tmp/dist_metrics_worker.sh (rPPO) and
# tmp/dist_dreamer_worker_batched.sh (Dreamer). Both algorithms go through the SAME call
# to scripts/eval/eval_rollout.py --batched --device cpu --record --config-list (unified
# 2026-07-21, moved to --config-list 2026-07-23 -- see docs/environment/
# SCRIPTS_DEPENDENCY_MAP.md); they differ only in --checkpoint form (rPPO:
# <run>/models/<step>, Dreamer: <run>/checkpoints/<episode>) and Dreamer's extra
# --agent_config (+ optional --episode).
#
# CHECKPOINT-granularity worklist (one line = one checkpoint + ALL its pending
# conditions, evaluated in ONE eval_rollout.py process -- builds the model + restores
# the checkpoint ONCE instead of once per condition; see run_sweep.py's build_groups()
# docstring for why the grouping is per-checkpoint, not per-condition).
#
# Usage: sweep_worker.sh <worklist_file> <node_id> <npar> <n_episodes>
#   worklist line format: CHECKPOINT|AGENT_CONFIG_OR_-|EPISODE_OR_-|CFG1,OUT1;CFG2,OUT2;...
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
  IFS='|' read -r ckpt agent ep cfg_out_list <<<"$1"
  extra=""
  [ "$agent" != "-" ] && extra="--agent_config $agent"
  [ "$ep" != "-" ] && extra="$extra --episode $ep"

  # Expand the ';'-separated 'cfg,out' pairs into a --config-list file (one
  # '<cfg>\t<out>' line per pending condition for THIS checkpoint), creating each
  # condition's output dir up front (eval_rollout.py also mkdir -p's it, but this
  # keeps behavior identical to the pre-config-list worker).
  cl_file="$(mktemp)"
  IFS=';' read -ra pairs <<< "$cfg_out_list"
  for pair in "${pairs[@]}"; do
    pcfg="${pair%%,*}"
    pout="${pair#*,}"
    mkdir -p "$pout"
    printf '%s\t%s\n' "$pcfg" "$pout" >> "$cl_file"
  done

  "$PY" scripts/eval/eval_rollout.py --config-list "$cl_file" $extra --checkpoint "$ckpt" \
    --eval-n-episodes "$NEP" --record --record-n-episodes "$NEP" \
    --device cpu --quiet --batched --seed 0 >/dev/null 2>&1 \
    || echo "FAIL $1" >> "$MARK/fail_$NODE"
  rm -f "$cl_file"
}
export -f runeval

t0=$(date +%s)
n=$(wc -l < "$WL")
xargs -P "$NPAR" -I@ bash -c 'runeval "$1"' _ @ < "$WL"
echo "[$NODE] done $n evals in $(( $(date +%s)-t0 ))s $(date +%H:%M:%S)" >> "$MARK/prog_$NODE"
touch "$MARK/done_$NODE"
