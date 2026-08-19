"""trajectory_store.py — the single source of truth for the trajectory-store schema.

Plan: docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md
Doc:  docs/environment/TRAJECTORY_STORE_SCHEMA.md  (generated from the tables below)

What this store is
------------------
After training finishes we put the trained agent back into the *exact* world it was
trained in, run it for a very large number of episodes, and write down (a) everything
that happened step-by-step and (b) the per-episode random draws the environment made
in secret at reset (how many predators, how far each can see, how fast it moves, how
many bushes, ...).  Those draws have never been recorded anywhere before; they are the
independent-variable side of every analysis this store exists to serve.

The hard requirement that shapes this module: **one reader must work for every run,
forever.**  Every environment writes exactly the same named columns in exactly the
same order and the same Arrow types; only the *widths* of the per-entity array columns
change from world to world, and those widths live in the store manifest.

Row convention — ARRIVAL (one sentence, and there is only one)
-------------------------------------------------------------
Row ``t`` holds (a) the environment state **at time t** and (b) the action, reward and
transition outputs of the step that **arrived at** time ``t``.  Row ``t = 0`` is the
reset state, with ``action = -1``, ``reward = 0.0`` and every transition-output field
at its zero value.  An episode of length ``T`` therefore has ``T + 1`` rows.

The observation columns in row ``t`` are the observation of state ``t`` — the
observation the policy consumed when choosing the action recorded in row ``t + 1``.

Mandatory-key discipline
------------------------
Nothing in this module uses ``dict.get(key, default)`` for a manifest field.  Every
manifest read goes through ``_req``, which raises ``ValueError`` when the key is
absent (project Configuration Protocol — no fallback defaults).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import yaml

# ── Schema version ────────────────────────────────────────────────────────────
# MAINTENANCE CONTRACT: any change to the fixed key set, any dtype change, and any
# row-convention change MUST bump this number and update
# docs/environment/TRAJECTORY_STORE_SCHEMA.md in the same commit.  Readers hard-fail
# on an unknown SCHEMA_VERSION.
SCHEMA_VERSION = 1

# Parquet compression for every shard.  Lossless and exact — this is NOT the same
# thing as `obs_precision`, which is a lossy scientific choice (see the schema doc).
SHARD_COMPRESSION = "zstd"

VALID_OBS_PRECISION = ("float16", "float32")


# ── Column description ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Column:
    """One column of the fixed key list.

    name    — column name; identical for every environment, forever.
    width   — None for a scalar column, else a dimension expression evaluated
              against the manifest dims, e.g. "A", "B", "A*V", "D".
    dtype   — element dtype name.  The literal string "obs" means "whatever
              `obs_precision` says", and is used ONLY by the two observation
              columns.  Every other float column is float32 regardless.
    timing  — row-convention timing (step columns only).
    source  — where the value comes from, or (episode columns) what it means.
    """
    name: str
    width: Optional[str]
    dtype: str
    timing: str
    source: str


_ARROW_ELEM = {
    "int64": pa.int64,
    "int32": pa.int32,
    "int16": pa.int16,
    "int8": pa.int8,
    "bool": pa.bool_,
    "float32": pa.float32,
    "float16": pa.float16,
}

_NUMPY_ELEM = {
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "bool": np.bool_,
    "float32": np.float32,
    "float16": np.float16,
}

DIM_KEYS = ("A", "R", "B", "V", "VV", "D")


# ── The FIXED KEY LIST — per-step record (plan §D4.1) ─────────────────────────
# Order is part of the contract.  Nothing may be inserted, removed or reordered
# without a SCHEMA_VERSION bump.

STEP_COLUMNS: tuple[Column, ...] = (
    Column("episode_seed", None, "int64", "episode key", "the seed whose PRNGKey produced the episode"),
    Column("t", None, "int16", "index, 0…T", "—"),
    Column("action", None, "int8", "arriving (-1 at t=0)", "argmax(logits)"),
    Column("reward", None, "float32", "arriving (0.0 at t=0)", "jax_step return"),
    Column("agent_row", None, "int16", "state at t", "state.agent_pos[0]"),
    Column("agent_col", None, "int16", "state at t", "state.agent_pos[1]"),
    Column("satiation", None, "float32", "state at t", "state.satiation"),
    Column("nutrition", None, "float32", "state at t", "state.nutrition"),
    Column("injury_level", None, "float32", "state at t", "state.injury_level"),
    Column("rest_streak", None, "int16", "state at t", "state.rest_streak"),
    Column("last_collision_noc", None, "float32", "state at t", "state.last_collision_noc"),
    Column("terminated", None, "bool", "state at t", "state.terminated"),
    Column("damage", None, "float32", "arriving (0.0 at t=0)", "info['damage']"),
    Column("ate_food", None, "bool", "arriving (False at t=0)", "info['ate_food']"),
    Column("rested", None, "bool", "arriving (False at t=0)", "info['rested']"),
    Column("hit_predator", None, "bool", "arriving (False at t=0)", "info['hit_predator']"),
    Column("hit_neutral", None, "bool", "arriving (False at t=0)", "info['hit_neutral']"),
    Column("hit_hiding_predator", None, "bool", "arriving (False at t=0)", "info['hit_hiding_predator']"),
    Column("event_collided", None, "bool", "arriving (False at t=0)", "info['event_collided']"),
    Column("agent_in_bush", None, "bool", "state at t",
           "info['agent_in_bush'] for t>=1; recomputed at reset for t=0 (plan §D8)"),
    Column("termination_reason", None, "int8", "arriving (0 except final row)",
           "info['termination_reason']; 0=active, 1=max_steps, 2=starvation, 3=overeating, 4=injury"),
    Column("animal_row", "A", "int16", "state at t", "state.animal_pos[:,0]"),
    Column("animal_col", "A", "int16", "state at t", "state.animal_pos[:,1]"),
    Column("animal_state", "A", "int8", "state at t", "state.animal_state — 0=PATROL, 1=HUNT, 2=RETURN"),
    Column("animal_stamina", "A", "float32", "state at t", "state.animal_stamina"),
    Column("animal_move_timer", "A", "int16", "state at t", "state.animal_move_timer"),
    Column("animal_attack_timer", "A", "int16", "state at t", "state.animal_attack_timer"),
    Column("res_row", "R", "int16", "state at t", "state.res_pos[:,0]"),
    Column("res_col", "R", "int16", "state at t", "state.res_pos[:,1]"),
    Column("res_active", "R", "bool", "state at t", "state.res_active"),
    Column("res_cons_count", "R", "int16", "state at t", "state.res_cons_count"),
    Column("res_reg_timer", "R", "int16", "state at t", "state.res_reg_timer"),
    Column("obs_row", "B", "int16", "state at t",
           "state.obs_pos[:,0] — constant within an episode; deliberately kept per-step (plan §D4.1)"),
    Column("obs_col", "B", "int16", "state at t", "state.obs_pos[:,1]"),
    Column("obs_noised", "D", "obs", "state at t",
           "get_observation(state, params) — what the policy received"),
    Column("obs_true", "D", "obs", "state at t",
           "get_observation(state, params, apply_noise=False) — noise-free ground truth"),
)

# ── The FIXED KEY LIST — per-episode record (plan §D4.2) ──────────────────────
# All per-episode float columns are float32 regardless of obs_precision: they are the
# independent variables of every future analysis and cost ~1.8 KB per episode against
# ~27-44 KB of step data.

EPISODE_COLUMNS: tuple[Column, ...] = (
    Column("episode_seed", None, "int64", "", "join key to `steps`; seed = seed_base + episode_index"),
    Column("episode_index", None, "int64", "", "0 … n_episodes-1"),
    Column("block_id", None, "int32", "", "shard block this episode belongs to"),
    Column("length", None, "int32", "", "T (environment steps; the step record has T+1 rows)"),
    Column("termination_reason", None, "int8", "", "terminal code (never 0)"),
    Column("reward_sum", None, "float32", "",
           "sum of `reward` over the episode — data, not the evaluation metric; survival steps (`length`) is the metric"),
    Column("animal_active", "A", "bool", "", "realised draw — which animal slots exist this episode"),
    Column("animal_detect_sampled", "A", "int32", "", "realised draw — HUNT-trigger sight range"),
    Column("animal_max_stamina_sampled", "A", "float32", "", "realised draw"),
    Column("animal_recovery_sampled", "A", "float32", "", "realised draw"),
    Column("animal_hunt_thresh_sampled", "A", "float32", "", "realised draw"),
    Column("animal_lose_interest_sampled", "A", "float32", "", "realised draw"),
    Column("animal_move_int_sampled", "A", "int32", "", "realised draw — move interval (lower = faster)"),
    Column("animal_attack_delay_sampled", "A", "int32", "", "realised draw"),
    Column("animal_attack_range_sampled", "A", "int32", "", "realised draw — jump/pounce range; 0 = disabled"),
    Column("animal_property_sampled", "A*V", "float32", "", "realised draw, flattened row-major [A, V]"),
    Column("animal_visual_property_sampled", "A*VV", "float32", "", "realised draw, flattened row-major [A, VV]"),
    Column("res_allocated", "R", "bool", "", "realised draw — immutable per-episode resource allocation mask"),
    Column("res_property_sampled_init", "R*V", "float32", "",
           "realised draw AT RESET, flattened [R, V] — re-drawn on regeneration; see the schema doc's caveat"),
    Column("res_visual_property_sampled_init", "R*VV", "float32", "",
           "realised draw AT RESET, flattened [R, VV] — re-drawn on regeneration; see the schema doc's caveat"),
    Column("obs_active", "B", "bool", "", "realised draw — which obstacle slots exist this episode"),
    Column("obs_property_sampled", "B*V", "float32", "", "realised draw, flattened [B, V]"),
    Column("obs_visual_property_sampled", "B*VV", "float32", "", "realised draw, flattened [B, VV]"),
)

# The realised per-episode draws (episode columns 7..23) — the independent-variable
# side of every analysis.  Named here so validate_store_draws cannot silently miss one.
DRAW_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in EPISODE_COLUMNS
    if c.name not in ("episode_seed", "episode_index", "block_id", "length",
                      "termination_reason", "reward_sum")
)

# Realised-draw columns that have explicit [low, high] sampling bounds in the manifest.
# Maps column name -> (manifest low key, manifest high key).
BOUNDED_DRAW_COLUMNS: dict[str, tuple[str, str]] = {
    "animal_detect_sampled": ("animal_detect_low", "animal_detect_high"),
    "animal_max_stamina_sampled": ("animal_max_stamina_low", "animal_max_stamina_high"),
    "animal_recovery_sampled": ("animal_recovery_low", "animal_recovery_high"),
    "animal_hunt_thresh_sampled": ("animal_hunt_thresh_low", "animal_hunt_thresh_high"),
    "animal_lose_interest_sampled": ("animal_lose_interest_low", "animal_lose_interest_high"),
    "animal_move_int_sampled": ("animal_move_int_low", "animal_move_int_high"),
    "animal_attack_delay_sampled": ("animal_attack_delay_low", "animal_attack_delay_high"),
    "animal_attack_range_sampled": ("animal_attack_range_low", "animal_attack_range_high"),
}

# Realised-draw property columns, mapped to the manifest's per-element std array.
# std == 0 -> the column must carry exactly one distinct value across the whole store;
# std  > 0 -> it must carry more than one.
PROPERTY_DRAW_COLUMNS: dict[str, str] = {
    "animal_property_sampled": "animal_property_std",
    "animal_visual_property_sampled": "animal_visual_property_std",
    "res_property_sampled_init": "res_property_std",
    "res_visual_property_sampled_init": "res_visual_property_std",
    "obs_property_sampled": "obs_property_std",
    "obs_visual_property_sampled": "obs_visual_property_std",
}

# Activation-mask columns, mapped to (per-entry count_low key, count_high key).
ACTIVATION_COLUMNS: dict[str, tuple[str, str]] = {
    "animal_active": ("animal_count_low", "animal_count_high"),
    "res_allocated": ("res_count_low", "res_count_high"),
    "obs_active": ("obs_count_low", "obs_count_high"),
}


# ── Dimension arithmetic ──────────────────────────────────────────────────────

def _eval_width(expr: str, dims: dict) -> int:
    """Evaluate a width expression ('A', 'A*V', ...) against the manifest dims."""
    parts = expr.split("*")
    out = 1
    for p in parts:
        p = p.strip()
        if p not in dims:
            raise ValueError(
                f"Unknown dimension {p!r} in width expression {expr!r}; "
                f"known dimensions are {sorted(dims)}"
            )
        out *= int(dims[p])
    return out


def normalise_dims(dims) -> dict:
    """Accept a (A, R, B, V, VV, D) tuple or a dict; return a validated dict."""
    if isinstance(dims, dict):
        missing = [k for k in DIM_KEYS if k not in dims]
        if missing:
            raise ValueError(f"dims is missing required keys: {missing}")
        return {k: int(dims[k]) for k in DIM_KEYS}
    dims = tuple(dims)
    if len(dims) != len(DIM_KEYS):
        raise ValueError(f"dims tuple must be {DIM_KEYS}, got {dims!r}")
    return {k: int(v) for k, v in zip(DIM_KEYS, dims)}


# ── Arrow schema construction ─────────────────────────────────────────────────
#
# DEVIATION FROM THE PLAN, recorded here because it is load-bearing (see the
# Implementation Report of the plan doc).  The plan specifies Arrow's
# `fixed_size_list<T>[w]` for every per-entity array column.  Parquet CANNOT round-trip
# a fixed_size_list of width ZERO: pyarrow 24.0.0 writes such a column and reads it
# back as `[[None], [None], ...]` — silently wrong data, not merely a type change.
# Zero-width columns are not hypothetical: 18 of the 334 saved configs in the results
# tree have no animals at all (A = 0), and checkpoint C10 of the plan requires them to
# work.  So every array column is Arrow's variable-size `list<T>` instead, with the
# constant width enforced by the WRITER (offsets are built as a ramp of the manifest
# width) and re-checked by `validate_store_shapes`.
#
# This is strictly MORE invariant than the plan asked for, not less: with
# `fixed_size_list` the column TYPE differs between environments
# (`fixed_size_list<int16>[4]` vs `[22]`), whereas `list<int16>` is byte-identical for
# every environment.  Measured cost: 823 B vs 835 B for a 20,000-row x 22-wide int16
# column under zstd — the variable-size form is marginally SMALLER, because Parquet has
# no fixed-size-list physical type either and encodes both as a repeated group.

def _elem_type(dtype: str, obs_precision: Optional[str]):
    if dtype == "obs":
        if obs_precision not in VALID_OBS_PRECISION:
            raise ValueError(
                f"obs_precision must be one of {VALID_OBS_PRECISION}, got {obs_precision!r}. "
                "It is a MANDATORY, LOSSY choice — there is no default."
            )
        return _ARROW_ELEM[obs_precision]()
    return _ARROW_ELEM[dtype]()


def _numpy_elem(dtype: str, obs_precision: Optional[str]):
    if dtype == "obs":
        if obs_precision not in VALID_OBS_PRECISION:
            raise ValueError(
                f"obs_precision must be one of {VALID_OBS_PRECISION}, got {obs_precision!r}."
            )
        return _NUMPY_ELEM[obs_precision]
    return _NUMPY_ELEM[dtype]


# Parquet renames a list's child field to "element" on round-trip, so the writer names it
# "element" up front. Without this, `build_step_schema(...)` and `pq.read_schema(...)` of
# the file it produced would differ (`list<item: int16>` vs `list<element: int16>`) and
# every schema comparison would be noise.
LIST_CHILD_NAME = "element"


def _list_type(elem_type) -> pa.DataType:
    return pa.list_(pa.field(LIST_CHILD_NAME, elem_type, nullable=True))


def _build_schema(columns: Sequence[Column], dims, obs_precision: Optional[str]) -> pa.Schema:
    dims = normalise_dims(dims)
    fields = []
    for c in columns:
        et = _elem_type(c.dtype, obs_precision)
        fields.append(pa.field(c.name, et if c.width is None else _list_type(et)))
    return pa.schema(fields)


def build_step_schema(dims, obs_precision: str) -> pa.Schema:
    """Arrow schema of `steps_NNNNN.parquet`.  dims = (A, R, B, V, VV, D)."""
    return _build_schema(STEP_COLUMNS, dims, obs_precision)


def build_episode_schema(dims) -> pa.Schema:
    """Arrow schema of `episodes_NNNNN.parquet`.  Per-episode floats are ALWAYS
    float32 — `obs_precision` does not apply here (plan §D4.2)."""
    return _build_schema(EPISODE_COLUMNS, dims, obs_precision=None)


def column_widths(columns: Sequence[Column], dims) -> dict[str, int]:
    """{column name: list width} for every array column (scalars omitted)."""
    dims = normalise_dims(dims)
    return {c.name: _eval_width(c.width, dims) for c in columns if c.width is not None}


def render_arrow_type(c: Column, dims=None, obs_precision: Optional[str] = None) -> str:
    """Human-readable Arrow type for the schema doc's generated tables."""
    if c.dtype == "obs":
        elem = "float16 | float32" if obs_precision is None else obs_precision
    else:
        elem = c.dtype
    if c.width is None:
        return elem
    if dims is None:
        return f"list<{elem}>[{c.width}]"
    return f"list<{elem}>[{_eval_width(c.width, dims)}]"


# ── Table construction ────────────────────────────────────────────────────────

def _list_array(flat: np.ndarray, n_rows: int, width: int, elem_type) -> pa.Array:
    """Build a variable-size list array whose every row has exactly `width` values.

    The width is enforced structurally: offsets are a ramp of `width`, and the flat
    child buffer must be exactly n_rows * width long.
    """
    flat = np.asarray(flat).reshape(-1)
    if flat.size != n_rows * width:
        raise ValueError(
            f"list column payload has {flat.size} values, expected "
            f"{n_rows} rows x width {width} = {n_rows * width}"
        )
    offsets = pa.array(np.arange(n_rows + 1, dtype=np.int32) * width, type=pa.int32())
    child = pa.array(flat, type=elem_type)
    return pa.ListArray.from_arrays(offsets, child, type=_list_type(elem_type))


def build_table(columns: Sequence[Column], data: dict, dims, obs_precision: Optional[str]) -> pa.Table:
    """Assemble a pa.Table from a {column name: numpy array} dict.

    Every column in `columns` must be present — a missing column is a hard error, not
    a silently-null column.  Array columns accept either a flat (n*width,) array or a
    (n, width) / (n, w1, w2) array; they are reshaped and width-checked.
    """
    dims = normalise_dims(dims)
    extra = sorted(set(data) - {c.name for c in columns})
    if extra:
        raise ValueError(f"build_table got columns not in the fixed key list: {extra}")

    # Row count is derived from a SCALAR column, never from an array column: an array
    # column's leading axis is only the row count if the caller shaped it correctly, so
    # taking `n_rows` from one would let a mis-shaped payload define its own truth.  Both
    # fixed key lists start with the scalar `episode_seed`.
    first = columns[0]
    if first.width is not None:
        raise ValueError(
            f"build_table expects the first column ({first.name!r}) to be a scalar so the "
            "row count can be derived from it; the fixed key lists both start with "
            "`episode_seed`.")
    if first.name not in data:
        raise ValueError(f"build_table is missing required column {first.name!r}")
    n_rows = int(np.asarray(data[first.name]).reshape(-1).shape[0])

    arrays = []
    for c in columns:
        if c.name not in data:
            raise ValueError(f"build_table is missing required column {c.name!r}")
        arr = np.asarray(data[c.name])
        et = _elem_type(c.dtype, obs_precision)
        nt = _numpy_elem(c.dtype, obs_precision)
        if c.width is None:
            arr = arr.reshape(-1).astype(nt, copy=False)
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"column {c.name!r} has {arr.shape[0]} rows, expected {n_rows}")
            arrays.append(pa.array(arr, type=et))
        else:
            w = _eval_width(c.width, dims)
            arrays.append(_list_array(arr.astype(nt, copy=False), n_rows, w, et))
    schema = _build_schema(columns, dims, obs_precision)
    return pa.Table.from_arrays(arrays, schema=schema)


# ── Environment fingerprint ───────────────────────────────────────────────────

def env_fingerprint(resolved_cfg_dict: dict) -> str:
    """First 10 hex chars of SHA-256 over the canonicalised resolved config.

    Canonicalisation is `yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)`,
    so key order and flow style in the saved file cannot change the fingerprint.
    A different environment config -> a different store directory, which is what makes
    the 2026-07-04 overwrite incident structurally impossible (plan §D3/§A6).
    """
    canon = yaml.safe_dump(resolved_cfg_dict, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:10]


# ── Manifest ──────────────────────────────────────────────────────────────────

MANIFEST_NAME = "_manifest.json"

# Fields compared on resume.  A mismatch on ANY of these is a hard ValueError and
# nothing is written (plan §D3 layer 2, verification V7).
MANIFEST_GUARDED_FIELDS = (
    "schema_version",
    "env_fp",
    "seed_base",
    "n_episodes",
    "shard_episodes",
    "obs_precision",
    "dims",
    "max_steps",
    "checkpoint_path",
    # `device` is guarded because bit-level results are LOWERING-dependent, not merely
    # algorithm-dependent: `vmap(jax_reset)` and unbatched `jax_reset` already disagree by
    # one float32 ULP on `animal_property_sampled`, purely from how XLA fuses a scatter
    # (plan Implementation Report §4.3).  A store started on CPU and resumed on GPU could
    # therefore hold bit-inconsistent blocks for what the manifest claims is one
    # homogeneous episode population, with nothing in the data to reveal it.
    "device",
    # Whether the STRICT checkpoint-restore structural check was required for this store
    # ("strict") or was allowed to degrade ("weak_allowed", i.e. the operator passed
    # --allow-weak-restore-check).  Guarded so a store cannot be half one and half the
    # other, and recorded so a reader can see which regime produced it.
    "restore_check",
)


def _req(d: dict, key: str):
    """Mandatory dict access.  No fallback defaults anywhere in this module."""
    if not isinstance(d, dict) or key not in d or d[key] is None:
        raise ValueError(
            f"Strict manifest: required key {key!r} is missing. "
            f"Present keys: {sorted(d) if isinstance(d, dict) else type(d).__name__}"
        )
    return d[key]


def write_manifest(store_dir, manifest: dict) -> None:
    """Write `_manifest.json`.  Written FIRST, before any shard, and never rewritten
    (plan §D11 — losing it is the one unrecoverable loss, since it carries the
    resolved config, seed_base and obs_precision)."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(store_dir)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def read_manifest(store_dir) -> dict:
    path = Path(store_dir) / MANIFEST_NAME
    if not path.exists():
        raise ValueError(f"No {MANIFEST_NAME} in {store_dir} — not a trajectory store.")
    with open(path) as f:
        m = json.load(f)
    v = _req(m, "schema_version")
    if int(v) != SCHEMA_VERSION:
        raise ValueError(
            f"Unknown SCHEMA_VERSION {v} in {path} (this code speaks {SCHEMA_VERSION}). "
            "Readers hard-fail on an unknown schema version — see the Maintenance "
            "Contract in docs/environment/TRAJECTORY_STORE_SCHEMA.md."
        )
    return m


def assert_manifest_compatible(store_dir, expected: dict) -> None:
    """Hard-fail on ANY mismatch between an existing store and what we are about to
    write.  This is the check that makes the 2026-07-04 contamination incident
    unrepeatable (plan §A6/§D3, verification V7).

    Git SHA is deliberately NOT guarded — code can legitimately move between
    collection sessions — but a difference is warned about loudly.
    """
    have = read_manifest(store_dir)
    problems = []
    for k in MANIFEST_GUARDED_FIELDS:
        a, b = _req(have, k), _req(expected, k)
        # dims may arrive as list-vs-tuple; normalise before comparing.
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            a, b = list(a), list(b)
        if a != b:
            problems.append(f"  {k}: store has {a!r}, this run wants {b!r}")
    if problems:
        raise ValueError(
            "Manifest mismatch — refusing to write into an existing store that was "
            "built differently. Nothing has been written.\n"
            f"  store: {store_dir}\n" + "\n".join(problems) + "\n"
            "If this is genuinely a different collection, give it a different "
            "--out-root; the store path is content-addressed by env config "
            "fingerprint precisely so two configurations cannot collide."
        )
    if have.get("collection_git_sha") != expected.get("collection_git_sha"):
        warnings.warn(
            f"Store {store_dir} was created at git SHA "
            f"{have.get('collection_git_sha')!r} but this session is at "
            f"{expected.get('collection_git_sha')!r}. Provenance is recorded per "
            "shard; continuing.",
            stacklevel=2,
        )


# ── Atomic shard write ────────────────────────────────────────────────────────

_DIR_FSYNC_UNSUPPORTED = False


def _fsync_dir(d: Path) -> None:
    """fsync a directory fd so a rename survives a NODE crash, not just a process kill.

    N3 DECISION (recorded deliberately rather than scattered as an ad-hoc try/except):
    `results/` is a CIFS mount (//192.168.0.250/cocoanlab01) and directory fsync is not
    guaranteed to be supported there.  If the kernel refuses it we warn ONCE per process
    and continue, because:
      - the file fsync + os.replace already give durability against a process SIGKILL,
        which is the failure mode the resume design is actually built for;
      - directory fsync only adds durability of the RENAME across a node crash, and if
        the filesystem does not implement it there is no alternative mechanism to fall
        back to — failing the collection would trade a real capability for a guarantee
        that is unattainable on this mount;
      - a shard that loses its rename simply looks incomplete to `completed_blocks()`
        and is recomputed bit-identically on resume (plan §D10).
    Anything other than "unsupported" (e.g. EACCES) is re-raised.
    """
    global _DIR_FSYNC_UNSUPPORTED
    fd = None
    try:
        fd = os.open(str(d), os.O_RDONLY)
        os.fsync(fd)
    except OSError as e:
        import errno
        if e.errno in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EBADF, errno.EISDIR):
            if not _DIR_FSYNC_UNSUPPORTED:
                _DIR_FSYNC_UNSUPPORTED = True
                warnings.warn(
                    f"Directory fsync is unsupported on the filesystem holding {d} "
                    f"(errno {e.errno}: {e.strerror}). Shard renames remain durable "
                    "against a process kill but not against a node crash; a lost "
                    "rename is recovered by re-running the block. See "
                    "src/utils/trajectory_store._fsync_dir.",
                    stacklevel=2,
                )
        else:
            raise
    finally:
        if fd is not None:
            os.close(fd)


def write_shard_atomic(path, table: pa.Table) -> None:
    """Write `<name>.parquet.tmp`, fsync it, os.replace it into place, fsync the dir.

    A shard on disk is therefore always complete — there is no partial-file state for a
    reader to trip over, and a guard that raises before this call leaves NOTHING on disk.
    The fsyncs are not optional here: `results/` is a CIFS mount, so POSIX rename
    atomicity + durability must not be assumed (plan §D3, verification V6).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pq.write_table(table, f, compression=SHARD_COMPRESSION)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def shard_names(block_id: int) -> tuple[str, str]:
    return f"episodes_{block_id:05d}.parquet", f"steps_{block_id:05d}.parquet"


def completed_blocks(store_dir) -> set:
    """A block is complete iff BOTH its shards exist (atomic rename guarantees each
    file is whole).  Resume = list complete blocks, skip them, work the rest."""
    store_dir = Path(store_dir)
    if not store_dir.is_dir():
        return set()
    eps = {int(m.group(1)) for p in store_dir.glob("episodes_*.parquet")
           if (m := re.fullmatch(r"episodes_(\d{5})\.parquet", p.name))}
    sts = {int(m.group(1)) for p in store_dir.glob("steps_*.parquet")
           if (m := re.fullmatch(r"steps_(\d{5})\.parquet", p.name))}
    return eps & sts


def clear_partial_shards(store_dir, block_id: int) -> None:
    """Delete any `.tmp` leftovers for a block we are about to redo (plan §D10)."""
    store_dir = Path(store_dir)
    for name in shard_names(block_id):
        tmp = store_dir / (name + ".tmp")
        if tmp.exists():
            tmp.unlink()


# ── Guard: scene ambiguity (plan §D14a, verification V10) ─────────────────────

SCENE_AMBIGUITY_HELP = (
    "The saved training config carries BOTH a modern `environment.entities:` block "
    "and a legacy `environment.predators:` / `environment.neutral_animals:` block. "
    "The trainer's scene precedence CHANGED at commit 828b77e (2026-07-23, "
    "'fix(config): legacy-scene precedence'): before it, train.py resolved such a file "
    "by taking the merged base config's newer `entities:` scene and discarding the "
    "legacy list; today's loader takes the legacy branch instead. Nothing in the run "
    "directory records which branch actually built the world it trained in — train.py "
    "writes no training-time git SHA — so reloading this config may rebuild the scene "
    "the trainer DISCARDED, and every recorded episode would be of a different world "
    "with no internal signal that anything is wrong. See "
    "docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md §A11. "
    "This guard REFUSES; it deliberately does not pick a scene, because there is no "
    "correct scene to reconstruct — only a choice. If you have independently "
    "established which scene is correct, re-run with --allow-ambiguous-scene, which "
    "downgrades this to a warning and stamps scene_ambiguous: true into the manifest "
    "so every downstream reader inherits the caveat."
)


def assert_scene_unambiguous(cfg: dict, run_path, allow_override: bool = False) -> tuple:
    """Refuse a saved config whose scene format is ambiguous.

    Returns `(scene_format, scene_ambiguous)` for the manifest —
    `scene_format` is "entities" | "legacy" | "none", and `scene_ambiguous` is True only
    when this config WAS ambiguous and `allow_override` waved it through.  Both values
    come from the single predicate evaluation below; callers must not re-derive the
    ambiguity from the config themselves, because a third copy of the predicate is a
    third thing that can drift out of step with `config_loader.py`.

    N1 NOTE — the predicates below are deliberately IDENTICAL to
    `src/environment/config_loader.py:428-435`:
        has_entities = config.get('environment.entities') is not None   # present, even if []
        has_legacy   = bool(predators) or bool(neutral_animals)         # present AND non-empty
    An earlier draft of this guard used `bool(entities)` for the first predicate, which
    diverges from the loader on the present-but-EMPTY `entities: []` + legacy shape
    (zero of the 334 saved configs in the results tree have that shape today, so the
    divergence was harmless — but silent divergence from the code you are guarding is
    exactly the kind of thing that stops being harmless without announcing itself).
    Matching the loader means this guard refuses precisely the configs whose loader
    branch is decided by precedence.
    """
    env = cfg.get("environment") if isinstance(cfg, dict) else None
    if not isinstance(env, dict):
        raise ValueError(
            f"Saved config at {run_path} has no `environment:` block — this is not a "
            "resolved training config."
        )
    has_entities = env.get("entities") is not None
    has_legacy = bool(env.get("predators")) or bool(env.get("neutral_animals"))

    if has_entities and has_legacy:
        msg = (
            f"Ambiguous scene format in the saved config of run {run_path!r}.\n"
            + SCENE_AMBIGUITY_HELP
        )
        if not allow_override:
            raise ValueError(msg)
        warnings.warn("PROCEEDING WITH AN AMBIGUOUS SCENE (--allow-ambiguous-scene): " + msg,
                      stacklevel=2)
        # The loader will take the legacy branch; record what will actually be built.
        return "legacy", True

    if has_entities:
        return "entities", False
    if has_legacy:
        return "legacy", False
    return "none", False


# ── Guard: strict checkpoint restore (plan F3, checkpoint C0) ─────────────────

_VALUE_SUFFIX = re.compile(r"(\[|\.)'?value'?\]?$")
_QUOTED_INDEX = re.compile(r"\['(\d+)'\]")


def _normalise_keystr(s: str) -> str:
    """Make two flattened trees of the same model comparable.

    Two purely-representational differences have to be normalised away, otherwise the
    guard reports every key as a mismatch and becomes noise rather than signal:

      1. a trailing ``['value']``: ``nnx.state(model)`` stops at the ``nnx.Param`` (which
         proxies ``.shape``), while orbax's on-disk metadata tree descends one level
         further to ``…['kernel']['value']``;
      2. list indices: nnx flattens a sequence to ``['layers'][0]`` while the metadata
         tree carries string dict keys, ``['layers']['0']``.

    Neither carries information about the model's STRUCTURE. A collision after
    normalisation is itself raised as an error, so this cannot mask a real difference.
    """
    return _QUOTED_INDEX.sub(r"[\1]", _VALUE_SUFFIX.sub("", s))


def _tree_shapes(tree) -> dict:
    """{normalised keypath: shape} for any pytree of array-likes / metadata objects."""
    import jax.tree_util as jtu
    flat, _ = jtu.tree_flatten_with_path(tree, is_leaf=lambda x: hasattr(x, "shape"))
    out = {}
    for k, v in flat:
        key = _normalise_keystr(jtu.keystr(k))
        if key in out:
            raise ValueError(
                f"Key path {key!r} appears twice after key normalisation — cannot "
                "compare trees safely.")
        out[key] = tuple(getattr(v, "shape", ()) or ())
    return out


def assert_restored_tree_matches(restored_tree, model, checkpoint_tree=None) -> None:
    """Hard-fail if the checkpoint and the built model disagree on ANY key or ANY shape.

    Why this is a GUARD and not a verification: V2 (trajectory parity against the legacy
    per-episode loop) cannot catch a structurally-wrong model, because both of its paths
    rebuild from the same saved config and therefore agree with the same wrong agent.

    Two recorded bugs are covered by one check:
      - a missing or misspelled `agent.modulation` block silently builds an UNMODULATED
        baseline that restores "successfully" (orbax's partial_restore only fills the
        keys the target declares, so the checkpoint's extra modulator weights are
        dropped without a word);
      - model-size CLI flags possibly not persisted to the saved config.

    `restored_tree` is compared to `nnx.state(model)` in BOTH directions.  Because
    partial_restore filters the checkpoint down to the target's keys, that comparison
    alone can never see the checkpoint's EXTRA keys — so `checkpoint_tree` (the
    checkpoint's own metadata tree, read with `PyTreeCheckpointer().metadata()`) must
    also be passed for the unmodulated-agent case to be detectable at all.
    """
    import flax.nnx as nnx

    model_shapes = _tree_shapes(nnx.state(model))
    problems = []

    def _compare(other, label):
        other_shapes = _tree_shapes(other)
        for k, s in sorted(model_shapes.items()):
            if k not in other_shapes:
                problems.append(f"  present in the built model but MISSING from {label}: {k} {s}")
            elif other_shapes[k] != s:
                problems.append(
                    f"  shape mismatch at {k}: model {s} vs {label} {other_shapes[k]}")
        for k, s in sorted(other_shapes.items()):
            if k not in model_shapes:
                problems.append(f"  present in {label} but MISSING from the built model: {k} {s}")

    _compare(restored_tree, "the restored tree")
    if checkpoint_tree is not None:
        _compare(checkpoint_tree, "the checkpoint on disk")

    if problems:
        raise ValueError(
            "Checkpoint / model structure mismatch — refusing to roll out.\n"
            + "\n".join(problems)
            + "\n\nA checkpoint that restores 'successfully' into a structurally "
              "different model is the failure mode behind two recorded bugs: a missing "
              "`agent.modulation` block silently yields an UNMODULATED baseline, and "
              "model-size CLI flags may not be persisted into the saved config. Check "
              "that <run>/models/config.yaml's `agent:` block is the one this "
              "checkpoint was trained with."
        )


# ── Whole-store validation (plan §D16, §D10) ──────────────────────────────────

def _list_column_to_2d(tbl: pa.Table, name: str, width: int) -> np.ndarray:
    """Flatten a list column into a (n_rows, width) numpy array."""
    col = tbl.column(name).combine_chunks()
    if isinstance(col, pa.ChunkedArray):
        col = col.chunk(0) if col.num_chunks == 1 else pa.concat_arrays(col.chunks)
    flat = np.asarray(col.flatten())
    n = tbl.num_rows
    if width == 0:
        return np.zeros((n, 0), dtype=flat.dtype)
    if flat.size != n * width:
        raise ValueError(
            f"column {name!r} holds {flat.size} values for {n} rows — expected width "
            f"{width} on every row (the store's fixed-width invariant is violated)")
    return flat.reshape(n, width)


def validate_store_shapes(store_dir) -> None:
    """Assert every shard's schema equals the manifest's schema exactly, and that every
    array column carries exactly the manifest width on every row.

    With variable-size `list<T>` columns (see the deviation note above) the width is a
    writer invariant rather than a type guarantee, so it is checked here explicitly.
    """
    store_dir = Path(store_dir)
    m = read_manifest(store_dir)
    dims = normalise_dims(_req(m, "dims"))
    obs_precision = _req(m, "obs_precision")
    want_steps = build_step_schema(dims, obs_precision)
    want_eps = build_episode_schema(dims)
    sw = column_widths(STEP_COLUMNS, dims)
    ew = column_widths(EPISODE_COLUMNS, dims)

    for block in sorted(completed_blocks(store_dir)):
        ep_name, st_name = shard_names(block)
        for name, want, widths, cols in (
            (ep_name, want_eps, ew, EPISODE_COLUMNS),
            (st_name, want_steps, sw, STEP_COLUMNS),
        ):
            tbl = pq.read_table(store_dir / name)
            if tbl.schema.names != want.names:
                raise ValueError(
                    f"{store_dir/name}: column names/order differ from the manifest "
                    f"schema.\n  store: {tbl.schema.names}\n  want:  {want.names}")
            for f_have, f_want in zip(tbl.schema, want):
                if f_have.type != f_want.type:
                    raise ValueError(
                        f"{store_dir/name}: column {f_have.name!r} has type "
                        f"{f_have.type} but the manifest schema says {f_want.type}")
            for c in cols:
                if c.width is not None:
                    _list_column_to_2d(tbl, c.name, widths[c.name])


def validate_store_structure(store_dir, expected_blocks: Optional[Iterable[int]] = None) -> dict:
    """Structural validation (plan §D10): expected blocks present, no duplicate
    `episode_seed` anywhere, and `steps.groupby(episode_seed).size() == length + 1`
    for every episode.  Returns a small summary dict."""
    store_dir = Path(store_dir)
    m = read_manifest(store_dir)
    have = completed_blocks(store_dir)
    if expected_blocks is not None:
        missing = sorted(set(expected_blocks) - have)
        if missing:
            raise ValueError(f"{store_dir}: missing blocks {missing}")

    seen_seeds: set = set()
    n_eps = 0
    n_steps = 0
    for block in sorted(have):
        ep_name, st_name = shard_names(block)
        ep = pq.read_table(store_dir / ep_name, columns=["episode_seed", "length", "block_id"])
        st = pq.read_table(store_dir / st_name, columns=["episode_seed"])
        seeds = np.asarray(ep.column("episode_seed"))
        lengths = np.asarray(ep.column("length"))
        blocks = np.asarray(ep.column("block_id"))
        if np.unique(seeds).size != seeds.size:
            raise ValueError(f"{store_dir/ep_name}: duplicate episode_seed WITHIN the shard")
        dup = seen_seeds & set(seeds.tolist())
        if dup:
            raise ValueError(
                f"{store_dir/ep_name}: episode_seed values already present in an earlier "
                f"shard: {sorted(dup)[:10]}{'…' if len(dup) > 10 else ''}")
        seen_seeds |= set(seeds.tolist())
        if not np.all(blocks == block):
            raise ValueError(f"{store_dir/ep_name}: block_id column disagrees with the file name")

        st_seeds = np.asarray(st.column("episode_seed"))
        uniq, counts = np.unique(st_seeds, return_counts=True)
        order = np.argsort(seeds)
        if not np.array_equal(uniq, seeds[order]):
            raise ValueError(
                f"{store_dir/st_name}: step-shard episode_seed set differs from the "
                "episode shard")
        if not np.array_equal(counts, lengths[order] + 1):
            bad = np.nonzero(counts != lengths[order] + 1)[0][:5]
            raise ValueError(
                f"{store_dir/st_name}: row count != length + 1 for seeds "
                f"{uniq[bad].tolist()} (got {counts[bad].tolist()}, want "
                f"{(lengths[order][bad] + 1).tolist()})")
        n_eps += len(seeds)
        n_steps += len(st_seeds)

    return {"blocks": len(have), "episodes": n_eps, "step_rows": n_steps,
            "n_episodes_expected": _req(m, "n_episodes")}


def validate_store_draws(store_dir) -> None:
    """Whole-store columnar validation of the realised per-episode draws (plan §D16).

    V1 samples ~200 episodes; a degenerate sampler or a column wired to a constant can
    slip through a sample.  This is one cheap pass over the `episodes` shards only:

      1. BOUNDS — every bounded draw lies within its manifest [low, high].
      2. NON-DEGENERACY — where low < high, the column carries MORE THAN ONE distinct
         value across the store; where low == high, exactly one.
      3. ACTIVATION — activation-mask population counts lie within
         [sum(count_low), sum(count_high)], and vary where the bounds allow.

    A failure here is a hard stop: a store that passes structure but fails this has
    *plausible-looking* independent variables, the worst failure mode for the analyses
    this store exists to serve.
    """
    store_dir = Path(store_dir)
    m = read_manifest(store_dir)
    dims = normalise_dims(_req(m, "dims"))
    widths = column_widths(EPISODE_COLUMNS, dims)
    blocks = sorted(completed_blocks(store_dir))
    if not blocks:
        raise ValueError(f"{store_dir}: no complete blocks to validate")

    cols = list(DRAW_COLUMNS)
    tbl = pa.concat_tables([
        pq.read_table(store_dir / shard_names(b)[0], columns=cols) for b in blocks
    ])
    n = tbl.num_rows
    data = {c: _list_column_to_2d(tbl, c, widths[c]) for c in cols}
    problems = []

    # 1 + 2 — bounded draws
    for col, (lo_key, hi_key) in BOUNDED_DRAW_COLUMNS.items():
        w = widths[col]
        if w == 0:
            continue
        lo = np.asarray(_req(m, lo_key), dtype=np.float64).reshape(-1)
        hi = np.asarray(_req(m, hi_key), dtype=np.float64).reshape(-1)
        if lo.size != w or hi.size != w:
            problems.append(f"{col}: manifest bounds have length {lo.size}/{hi.size}, column width {w}")
            continue
        vals = data[col].astype(np.float64)
        below = vals < lo[None, :] - 1e-6
        above = vals > hi[None, :] + 1e-6
        if below.any() or above.any():
            i = np.argwhere(below | above)[0]
            problems.append(
                f"{col}: value {vals[i[0], i[1]]!r} at slot {i[1]} is outside its "
                f"sampling bounds [{lo[i[1]]}, {hi[i[1]]}]")
        for j in range(w):
            nun = np.unique(vals[:, j]).size
            if lo[j] < hi[j] and nun <= 1:
                problems.append(
                    f"{col}[slot {j}]: bounds are [{lo[j]}, {hi[j]}] (low < high) but the "
                    f"whole store carries a single distinct value {vals[0, j]!r} — the "
                    "sampler looks stuck or the column is wired to the wrong source")
            if lo[j] == hi[j] and nun != 1:
                problems.append(
                    f"{col}[slot {j}]: bounds are degenerate [{lo[j]}, {lo[j]}] but the "
                    f"store carries {nun} distinct values")

    # 2 — property draws, governed by their per-element std
    for col, std_key in PROPERTY_DRAW_COLUMNS.items():
        w = widths[col]
        if w == 0:
            continue
        std = np.asarray(_req(m, std_key), dtype=np.float64).reshape(-1)
        if std.size != w:
            problems.append(f"{col}: manifest {std_key} has {std.size} values, column width {w}")
            continue
        vals = data[col]
        for j in range(w):
            nun = np.unique(vals[:, j]).size
            if std[j] > 0 and nun <= 1 and n > 1:
                problems.append(
                    f"{col}[{j}]: {std_key}[{j}] = {std[j]} > 0 but the store carries a "
                    f"single distinct value {vals[0, j]!r}. CAVEAT before you treat this "
                    "as a wiring bug: the sampler CLIPS its draw (to [0,1] for chemical "
                    "properties, to [0, inf) for visual ones), so a property whose mean "
                    "sits exactly on a clip bound with a small std legitimately collapses "
                    "to a single value. Check the mean against the bound first")
            if std[j] == 0 and nun != 1:
                problems.append(
                    f"{col}[{j}]: {std_key}[{j}] = 0 but the store carries {nun} distinct values")

    # 3 — activation masks
    for col, (lo_key, hi_key) in ACTIVATION_COLUMNS.items():
        w = widths[col]
        if w == 0:
            continue
        lo = int(np.asarray(_req(m, lo_key)).sum())
        hi = int(np.asarray(_req(m, hi_key)).sum())
        pop = data[col].astype(np.int64).sum(axis=1)
        if pop.min() < lo or pop.max() > hi:
            problems.append(
                f"{col}: population count range [{pop.min()}, {pop.max()}] falls outside "
                f"the manifest's count bounds [{lo}, {hi}]")
        if lo < hi and np.unique(pop).size <= 1 and n > 1:
            problems.append(
                f"{col}: count bounds are [{lo}, {hi}] (low < high) but every episode in "
                f"the store has exactly {pop[0]} active slots")

    if problems:
        raise ValueError(
            f"Whole-store realised-draw validation FAILED for {store_dir} "
            f"({n} episodes across {len(blocks)} blocks):\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nThe realised draws are the entire point of this store. Do not analyse "
              "it; delete it and find the cause.")


# ── Reader ────────────────────────────────────────────────────────────────────

class TrajectoryStore:
    """The API every future analysis uses.  Must work for every run, forever."""

    def __init__(self, store_dir):
        self.path = Path(store_dir)
        self.manifest = read_manifest(self.path)          # validates SCHEMA_VERSION
        self.dims = normalise_dims(_req(self.manifest, "dims"))
        self.obs_precision = _req(self.manifest, "obs_precision")
        self._step_widths = column_widths(STEP_COLUMNS, self.dims)
        self._episode_widths = column_widths(EPISODE_COLUMNS, self.dims)

    # -- datasets ------------------------------------------------------------
    def _dataset(self, prefix: str, schema: pa.Schema) -> pads.Dataset:
        files = sorted(self.path.glob(f"{prefix}_*.parquet"))
        if not files:
            raise ValueError(f"{self.path}: no {prefix}_*.parquet shards")
        return pads.dataset([str(f) for f in files], format="parquet", schema=schema)

    def episodes(self, columns=None, filter=None) -> pa.Table:
        ds = self._dataset("episodes", build_episode_schema(self.dims))
        return ds.to_table(columns=columns, filter=filter)

    def steps(self, columns=None, filter=None) -> pa.Table:
        ds = self._dataset("steps", build_step_schema(self.dims, self.obs_precision))
        return ds.to_table(columns=columns, filter=filter)

    # -- helpers -------------------------------------------------------------
    def width(self, col: str) -> int:
        if col in self._step_widths:
            return self._step_widths[col]
        if col in self._episode_widths:
            return self._episode_widths[col]
        raise ValueError(f"{col!r} is not an array column of this schema")

    def reshape(self, col: str, arr) -> np.ndarray:
        """Flattened [A, V]-style draw -> (n, A, V).  Non-flattened array columns come
        back as (n, width)."""
        spec = {c.name: c for c in EPISODE_COLUMNS + STEP_COLUMNS}
        if col not in spec or spec[col].width is None:
            raise ValueError(f"{col!r} is not an array column of this schema")
        parts = [p.strip() for p in spec[col].width.split("*")]
        shape = tuple(self.dims[p] for p in parts)
        arr = np.asarray(arr)
        return arr.reshape((-1,) + shape)

    def to_2d(self, table: pa.Table, col: str) -> np.ndarray:
        """Read an array column of a table into a dense (n_rows, width) array."""
        return _list_column_to_2d(table, col, self.width(col))

    def entity_labels(self) -> dict:
        """index -> (name, class, behaviour) for animals, plus resource/obstacle names."""
        m = self.manifest
        return {
            "animals": [
                {"index": i, "tag": t, "class": c, "behaviour": b}
                for i, (t, c, b) in enumerate(zip(
                    _req(m, "animal_tags"), _req(m, "animal_classes"), _req(m, "animal_behaviours")))
            ],
            "resources": [{"index": i, "name": n} for i, n in enumerate(_req(m, "resource_names"))],
            "obstacles": [{"index": i, "name": n} for i, n in enumerate(_req(m, "obstacle_names"))],
        }

    def select_animals(self, class_name: str) -> np.ndarray:
        """Boolean mask over animal slots, the same way src/environment/state.py's
        `select_by_class` does for a live EnvParams."""
        return np.array([c == class_name for c in _req(self.manifest, "animal_classes")], dtype=bool)

    def __repr__(self):
        m = self.manifest
        return (f"<TrajectoryStore {self.path} run={m.get('run_dir_name')} "
                f"ckpt={m.get('ckpt_step')} dims={self.dims} "
                f"obs_precision={self.obs_precision}>")


def open_store(store_dir) -> TrajectoryStore:
    """Open a store, validating SCHEMA_VERSION."""
    return TrajectoryStore(store_dir)
