#!/usr/bin/env python3
"""collect_trajectories.py — single-run, single-process trajectory collector.

Plan: docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md
Doc:  docs/environment/TRAJECTORY_STORE_SCHEMA.md

Put a trained agent back into the EXACT world it was trained in, run it for a large
number of episodes, and record both what it did and the per-episode random draws the
environment made in secret at reset.  Output is a sharded Parquet store whose schema is
identical for every training run, forever (see `src/utils/trajectory_store.py`).

APPLICABILITY BOUNDARY — read before pointing this at an old run.
Runs trained after 2026-07-23 (commit 828b77e) are unambiguous: the trainer and today's
config loader resolve the scene identically.  Runs from before that date MAY NOT BE, and
any run whose saved config carries BOTH a modern `entities:` block and a legacy
`predators:`/`neutral_animals:` block cannot be faithfully reconstructed and is REFUSED
by default (`--allow-ambiguous-scene` overrides, loudly, and stamps the caveat into the
manifest).  Measured against the results tree as of 2026-08-19, exactly 12 of 334 saved
configs are affected, all dated 2026-05-29 to 2026-06-11.

Usage
-----
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
        scripts/eval/traj_collect/collect_trajectories.py \
        --run results/JAX_RecurrentPPO/<run_dir> \
        --checkpoint final --out-root results/trajectories \
        --episodes 1000000 --seed-base 1000000 --blocks 0:20 \
        --obs-precision float32 --device cpu

`--obs-precision` is MANDATORY and has no default, because it is a LOSSY scientific
choice.  The recommended value is `float32`, decided by measurement on 2026-08-20:
half precision saved 12.8 % store-wide on a real paired collection, below the 20 %
threshold pre-registered before the measurement.  `float16` remains supported and is a
sound accuracy choice (worst measured error 1.95e-03, ~100x below the environment's own
injected sensor noise).  Note that under `float32` the observation guard returns after
its finiteness check, so both magnitude clauses go dormant — there is nothing for them to
catch, but a float32 store therefore carries no recorded evidence about observation range.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # scripts/eval/traj_collect -> repo root (3 up)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import trajectory_store as ts
from src.utils.config import Config
from src.utils.provenance import git_commit
from src.utils.trajectory_store import (
    EPISODE_COLUMNS, SCHEMA_VERSION, STEP_COLUMNS,
    assert_manifest_compatible, assert_restored_tree_matches, assert_scene_unambiguous,
    build_table, clear_partial_shards, column_widths, completed_blocks, env_fingerprint,
    normalise_dims, read_manifest, shard_names, write_manifest, write_shard_atomic,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import traj_scan  # noqa: E402  (same-folder module)

DEFAULT_SHARD_EPISODES = 5000
DEFAULT_BATCH_SIZE = {"cpu": 1024, "gpu": 8192}


# ── Provenance ────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    """Full HEAD sha of THIS collection session, or "unknown".

    Thin wrapper over the shared helper in src/utils/provenance.py (one git-provenance
    implementation project-wide). Output format unchanged.
    """
    return git_commit(short=False, cwd=PROJECT_ROOT)


def read_training_provenance(run_dir: Path) -> dict:
    """Read `<run_dir>/models/provenance.json`, written by train.py at run startup.

    Three states, and they mean DIFFERENT things — a reader must be able to tell them
    apart, so they are not collapsed:

      * file present, git readable → the real sha / dirty flag / start time;
      * file present but a field is the string "unknown" → the run WAS stamped, but git
        could not be read on that node at that moment;
      * file ABSENT → `training_git_sha` etc. are `None`. Every run trained before this
        stamp existed (2026-08-20) is in this state. `None` means "pre-stamp run", which
        is emphatically not the same claim as "stamped, but unknown".

    Absence is NEVER an error: the whole existing `results/` corpus lacks the file and
    the collector must keep working on it.
    """
    path = run_dir / "models" / "provenance.json"
    try:
        with open(path) as f:
            p = json.load(f)
        if not isinstance(p, dict):
            return {"training_git_sha": None, "training_git_dirty": None,
                    "training_started_utc": None}
        return {
            "training_git_sha": p.get("git_sha", "unknown"),
            "training_git_dirty": p.get("git_dirty", "unknown"),
            "training_started_utc": p.get("started_utc", "unknown"),
        }
    except FileNotFoundError:
        return {"training_git_sha": None, "training_git_dirty": None,
                "training_started_utc": None}
    except Exception as e:            # unreadable / corrupt JSON — still not fatal
        warnings.warn(f"Could not read {path}: {e}. Recording training provenance as "
                      f"absent (null).", stacklevel=2)
        return {"training_git_sha": None, "training_git_dirty": None,
                "training_started_utc": None}


# ── Checkpoint selection (plan §D9) ───────────────────────────────────────────

def resolve_checkpoint(run_dir: Path, which: str) -> tuple[Path, int]:
    """Resolve `final` or an explicit step to a checkpoint directory.

    NUMERIC SORT IS MANDATORY.  The rPPO run used to size this pipeline has 591
    checkpoint directories with names like `100145`, `10100043`, `59100070`.  They are
    not zero-padded, so a LEXICOGRAPHIC maximum picks `9900021` — the wrong directory,
    silently, with no error anywhere downstream.
    """
    models = run_dir / "models"
    if not models.is_dir():
        raise ValueError(f"No models/ directory under {run_dir}")
    steps = sorted(int(p.name) for p in models.iterdir() if p.is_dir() and p.name.isdigit())
    if not steps:
        raise ValueError(f"No numerically-named checkpoint directories under {models}")
    if which == "final":
        step = max(steps)                      # int(), NOT lexicographic max()
    else:
        step = int(which)
        if step not in steps:
            raise ValueError(f"Checkpoint step {step} not found under {models} "
                             f"(have {len(steps)} steps, max {max(steps)})")
    return models / str(step), step


# ── Policy seam (plan File Changes) ───────────────────────────────────────────

def read_checkpoint_tree(ckpt_dir: Path):
    """The checkpoint's OWN key set + shapes, read from orbax metadata.

    This is the half of the restore guard that has teeth.  `partial_restore` filters the
    checkpoint down to the keys the target declares, so the restored tree can NEVER
    reveal the checkpoint's EXTRA keys — which is precisely the silent-unmodulated-agent
    case.  Returns None when the metadata is unreadable; the caller decides what that
    means (it is a hard failure unless `--allow-weak-restore-check` was passed).
    """
    import orbax.checkpoint as ocp
    md = ocp.PyTreeCheckpointer().metadata(str(ckpt_dir / "default"))
    tree = getattr(getattr(md, "item_metadata", None), "tree", None)
    if isinstance(tree, dict) and "model" in tree:
        return tree["model"]
    return None


def load_policy(agent_type: str, ckpt_dir: Path, cfg: dict, params,
                quiet: bool = False, allow_weak_restore_check: bool = False):
    """Build the model, restore the checkpoint, and hard-check the two agree.

    Returns `(model, initial_state_fn, policy_step)`.  Kept behind this seam — together
    with `policy_step(model, obs, h) -> (action, h)` — so Dreamer-SRL can be added later.
    rPPO is the ONLY supported and tested algorithm in this change.
    """
    if agent_type != "rppo":
        raise NotImplementedError(
            f"agent_type={agent_type!r} is not supported. This pipeline implements, "
            "tests and claims rPPO only; the load_policy / policy_step seams exist so "
            "Dreamer-SRL can be added later. See the 'Out of Scope' section of "
            "docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md."
        )

    import flax.nnx as nnx
    import jax
    import orbax.checkpoint as ocp
    from src.environment.sensor import get_observation_breakdown
    from src.models.recurrent_ppo_network import ActorCriticRNN

    agent_cfg = Config(cfg).get("agent")
    if agent_cfg is None:
        raise ValueError("Saved config has no `agent:` block — cannot rebuild the model.")
    acfg = Config({"agent": agent_cfg})

    obs_breakdown = get_observation_breakdown(params)
    input_dim = sum(obs_breakdown.values())
    action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)
    modulation_config = acfg.get("agent.modulation")
    if modulation_config is not None and modulation_config.get("type") is None:
        modulation_config = None

    model = ActorCriticRNN(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_size=acfg.get_mandatory("agent.hidden_size"),
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
        rnn_type=acfg.get_mandatory("agent.rnn_type"),
        activation=acfg.get_mandatory("agent.activation"),
        modulation_config=modulation_config,
        observation_breakdown=obs_breakdown,
        encoding_config=agent_cfg,
    )

    step = int(ckpt_dir.name)
    mngr = ocp.CheckpointManager(str(ckpt_dir.parent))
    current = nnx.state(model)
    dev = jax.local_devices()[0]

    def _cpu_restore_arg(_x):
        return ocp.ArrayRestoreArgs(restore_type=jax.Array,
                                    sharding=jax.sharding.SingleDeviceSharding(dev))

    restored = mngr.restore(
        step,
        args=ocp.args.PyTreeRestore(
            item={"model": current},
            restore_args={"model": jax.tree_util.tree_map(_cpu_restore_arg, current)},
            partial_restore=True,
        ),
    )["model"]

    # F3 — the strict structural check.  `partial_restore` filters the checkpoint down
    # to the target's keys, so the restored tree alone can NEVER reveal the checkpoint's
    # EXTRA keys; the checkpoint's own metadata tree is read separately for exactly that
    # reason (this is what catches the silent-unmodulated-agent bug).
    #
    # DEGRADATION IS A HARD FAILURE, not a printed warning.  Without the metadata tree
    # the comparison reduces to `nnx.state(model)` against a restore that was itself
    # filtered by `nnx.state(model)` — i.e. the model versus itself, structurally blind
    # to the one bug the guard exists for.  Under `xargs -P 16` with per-cell log
    # redirection, a printed warning is not a control: nobody reads 160 log files.
    try:
        ckpt_model_tree = read_checkpoint_tree(ckpt_dir)
        meta_error = None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        ckpt_model_tree, meta_error = None, f"{type(e).__name__}: {e}"

    if ckpt_model_tree is None:
        msg = (
            f"Could not read the checkpoint's own structure from {ckpt_dir/'default'} "
            f"({meta_error or 'metadata carried no model tree'}).\n"
            "Without it the restore check degrades to comparing the built model against "
            "a restore that orbax already filtered THROUGH that same model — structurally "
            "blind to the checkpoint's extra keys, which is exactly the "
            "silent-unmodulated-agent failure this guard exists to catch (a missing or "
            "misspelled `agent.modulation` block builds an UNMODULATED baseline that "
            "restores 'successfully').\n"
            "Refusing to roll out. If you have independently confirmed the architecture, "
            "re-run with --allow-weak-restore-check; the manifest will record that this "
            "store was collected without the strict check so every reader inherits the "
            "caveat."
        )
        if not allow_weak_restore_check:
            raise ValueError(msg)
        warnings.warn("WEAK RESTORE CHECK (--allow-weak-restore-check): " + msg, stacklevel=2)

    assert_restored_tree_matches(restored, model, checkpoint_tree=ckpt_model_tree)

    nnx.update(model, restored)
    if not quiet:
        print(f"[collect] rPPO model restored from step {step} "
              f"(input_dim={input_dim}, action_dim={action_dim}).", flush=True)

    def policy_step(m, obs, h):
        import jax.numpy as jnp
        logits, _value, h_new, _mod = m(obs, h)
        return jnp.argmax(logits, axis=-1), h_new

    return model, (lambda n: model.initial_state(batch_size=n)), policy_step


# ── Manifest construction ─────────────────────────────────────────────────────

def _np_list(x):
    return np.asarray(x).tolist()


# --- pre-v3.1 sensor compatibility -------------------------------------------------
# Commit 0e8a4ef (v3.1, 2026-08-21) added directional olfaction and anisotropic visual
# point-spread, and made five `sensory.*` keys MANDATORY.  Every run trained before that
# date lacks them, so `load_env_params` refuses their saved config and they cannot be
# replayed at all.
#
# Both features have an explicit trace-time OFF branch (`sensor.py:52` and `sensor.py:325`),
# so the values below reproduce the pre-v3.1 sensors exactly.  Verified end-to-end rather
# than assumed: replaying recorded actions for 25 real episodes (5,298 steps) through
# today's code and comparing against the observation vectors the OLD code wrote into the
# a01 store gives a maximum difference of 2.4e-07 across all 27 channels — float32
# rounding, not behaviour.  See `scripts/analysis/supplementary/parity.py`.
#
# This is deliberately an explicit opt-in flag and NOT a default.  The project's
# no-fallback-defaults rule exists to stop exactly this kind of silent substitution, and
# a run that genuinely used v3.1 sensors must never be quietly reinterpreted as pre-v3.1.
# Patching happens before `env_fingerprint`, so the store's `env_fp` — a guarded manifest
# field — reflects the config actually used.
PRE_V31_SENSOR_DEFAULTS = {
    "olfactory_grid_range": 0,        # 0 = the original single-point sample
    "visual_blur_enabled": False,
    # These three MUST be the shipped default.yaml values, not zeros. They were
    # zeros originally on the reasoning that blur sits behind a trace-time static
    # branch, so the values are behaviourally inert when it is off -- which is
    # still true. But commit 4a5fcd4 added load-time validation that rejects
    # anisotropy <= 0 and sigma_floor <= 0 REGARDLESS of whether blur is enabled,
    # precisely so a config cannot carry a divide-by-zero that only detonates when
    # someone later flips the flag. Zeros therefore no longer load at all.
    # Semantically these are also the right values: an old run had no such key, so
    # the honest reconstruction is whatever the base config would supply.
    "visual_blur_radial_scale": 0.5,
    "visual_blur_anisotropy": 3.0,
    "visual_blur_sigma_floor": 0.5,
}

# Commit 9771e98 (v3.2, 2026-08-26) added visual value modes and line-of-sight occlusion and made
# these mandatory.  Runs launched earlier the same day lack them.  Both features sit behind
# trace-time static branches in sensor.py -- `if params.visual_occlusion_enabled` and
# `if params.visual_value_mode == 'clamp'` -- so these values leave the original code path
# untouched rather than approximating it.  Cone and strength are deliberately absent: the loader
# only demands them when occlusion is on.
PRE_V32_SENSOR_DEFAULTS = {
    "visual_value_mode": "sum",          # 'sum' = the original per-cell accumulation
    "visual_occlusion_enabled": False,
}


def apply_sensor_compat(cfg: dict, run_dir: Path, defaults: dict, flag: str,
                        version: str) -> list[str]:
    """Inject a version's sensory keys at values that reproduce the earlier behaviour.

    Refuses if the config already carries any of them: a run that set them explicitly is a later
    run, and silently overwriting a real setting is the failure these flags exist to avoid.
    """
    sens = cfg.setdefault("sensory", {})
    present = [k for k in defaults if k in sens]
    if present:
        raise ValueError(
            f"{flag} was passed, but {run_dir.name} already sets {present} in sensory. That makes "
            f"it a {version}-or-later run; the flag would silently overwrite real settings. "
            "Drop the flag.")
    sens.update(defaults)
    return sorted(defaults)


def build_manifest(*, cfg, params, run_dir: Path, ckpt_dir: Path, ckpt_step: int,
                   dims: dict, seed_base: int, n_episodes: int, shard_episodes: int,
                   batch_size: int, device: str, obs_precision: str,
                   scene_format: str, scene_ambiguous: bool,
                   restore_check: str = "strict",
                   pre_v31_sensor_keys: list | None = None) -> dict:
    """Everything a future reader needs and cannot recover from the shards."""
    # int16 columns (`t`, `rest_streak`, `res_cons_count`, and the animal/resource timers)
    # are written with `astype`, which WRAPS silently rather than raising.  max_steps is
    # the quantity that bounds all of them.
    if int(params.max_steps) >= 32768:
        raise ValueError(
            f"environment.max_steps = {int(params.max_steps)} does not fit the schema's "
            "int16 step columns (`t`, `rest_streak`, `res_cons_count`, the move/attack/"
            "regeneration timers). numpy's astype wraps silently, so this would be "
            "recorded as plausible negative data. Raising the schema's step-column width "
            "is a SCHEMA_VERSION bump, not a config change.")
    # An episode is a pure function of `jax.random.PRNGKey(seed)`, and PRNGKey takes a
    # uint32.  With jax_enable_x64 off, `jnp.asarray(np.int64)` canonicalises to int32.
    # Measured: seeds in [2**31, 2**32) wrap to a negative int32 but round-trip to the
    # SAME uint32, so they are harmless; at 2**32 the uint32 itself wraps and seed
    # 2**32 + 5 produces byte-identical episodes to seed 5 while `episode_seed` records
    # the un-wrapped value — a recorded seed that does not reproduce its own episode.
    # Neither the PRNG parity guard (both sides consume the same wrapped seeds) nor the
    # duplicate-seed check (the recorded values differ) can see it.  Refuse well before
    # the cliff, at the point where no canonicalisation happens at all.
    if int(seed_base) < 0 or int(seed_base) + int(n_episodes) >= 2 ** 31:
        raise ValueError(
            f"seed_base + episodes = {int(seed_base) + int(n_episodes)} must lie in "
            f"[0, 2**31). Seeds are canonicalised to int32 before PRNGKey construction "
            "(jax_enable_x64 is off), and past 2**32 two different recorded "
            "`episode_seed` values silently produce the SAME episode. Nothing downstream "
            "can detect that.")
    return {
        "schema_version": SCHEMA_VERSION,
        "env_fp": env_fingerprint(cfg),
        "resolved_env_config": cfg,
        "run_path": str(run_dir),
        "run_dir_name": run_dir.name,
        "train_config_mtime": datetime.fromtimestamp(
            (run_dir / "models" / "config.yaml").stat().st_mtime, tz=timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt_dir),
        "ckpt_step": int(ckpt_step),
        "dims": {k: int(v) for k, v in dims.items()},
        "max_steps": int(params.max_steps),
        "action_dim": 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled),
        "seed_base": int(seed_base),
        "n_episodes": int(n_episodes),
        "shard_episodes": int(shard_episodes),
        "batch_size": int(batch_size),
        "device": device,
        "policy_mode": "deterministic_argmax",
        "obs_precision": obs_precision,
        "collection_git_sha": _git_sha(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        # Training-time provenance, read back from <run_dir>/models/provenance.json.
        # null  = the run predates the stamp (no file)  |  "unknown" = stamped, git
        # unreadable at train time. See read_training_provenance().
        **read_training_provenance(run_dir),
        "pre_v31_sensor_keys": pre_v31_sensor_keys,
        "scene_format": scene_format,
        "scene_ambiguous": bool(scene_ambiguous),
        "restore_check": restore_check,
        # -- human labels: array index i is meaningless without a name -----------
        "animal_tags": list(params.animal_tags),
        "animal_classes": list(params.animal_classes),
        "animal_behaviours": list(params.animal_behaviours),
        "resource_names": [f"res_type_{int(t)}" for t in np.asarray(params.res_type)],
        "obstacle_names": [params.obstacle_names[int(t)] if int(t) < len(params.obstacle_names)
                           else f"obs_type_{int(t)}" for t in np.asarray(params.obs_type)],
        "observation_breakdown": _obs_breakdown(params),
        # -- static per-entity params that are join partners for the draws ------
        "animal_damage": _np_list(params.animal_damage),
        "animal_is_damaging": _np_list(params.animal_is_damaging),
        "obs_hides_agent": _np_list(params.obs_hides_agent),
        "obs_blocking": _np_list(params.obs_blocking),
        "res_type": _np_list(params.res_type),
        "res_damage": _np_list(params.res_damage),
        "obs_damage": _np_list(params.obs_damage),
        # -- per-episode sampling bounds (the join partner for every realised draw)
        "animal_detect_low": _np_list(params.animal_detect_low),
        "animal_detect_high": _np_list(params.animal_detect_high),
        "animal_max_stamina_low": _np_list(params.animal_max_stamina_low),
        "animal_max_stamina_high": _np_list(params.animal_max_stamina_high),
        "animal_recovery_low": _np_list(params.animal_recovery_low),
        "animal_recovery_high": _np_list(params.animal_recovery_high),
        "animal_hunt_thresh_low": _np_list(params.animal_hunt_thresh_low),
        "animal_hunt_thresh_high": _np_list(params.animal_hunt_thresh_high),
        "animal_lose_interest_low": _np_list(params.animal_lose_interest_low),
        "animal_lose_interest_high": _np_list(params.animal_lose_interest_high),
        "animal_move_int_low": _np_list(params.animal_move_int_low),
        "animal_move_int_high": _np_list(params.animal_move_int_high),
        "animal_attack_delay_low": _np_list(params.animal_attack_delay_low),
        "animal_attack_delay_high": _np_list(params.animal_attack_delay_high),
        "animal_attack_range_low": _np_list(params.animal_attack_range_low),
        "animal_attack_range_high": _np_list(params.animal_attack_range_high),
        "animal_count_low": _np_list(params.animal_count_low),
        "animal_count_high": _np_list(params.animal_count_high),
        "res_count_low": _np_list(params.res_count_low),
        "res_count_high": _np_list(params.res_count_high),
        "obs_count_low": _np_list(params.obs_count_low),
        "obs_count_high": _np_list(params.obs_count_high),
        # -- per-element property stds (govern the non-degeneracy check) --------
        "animal_property_std": _np_list(np.asarray(params.animal_property_std).reshape(-1)),
        "animal_visual_property_std": _np_list(np.asarray(params.animal_visual_property_std).reshape(-1)),
        "res_property_std": _np_list(np.asarray(params.res_property_std).reshape(-1)),
        "res_visual_property_std": _np_list(np.asarray(params.res_visual_property_std).reshape(-1)),
        "obs_property_std": _np_list(np.asarray(params.obs_property_std).reshape(-1)),
        "obs_visual_property_std": _np_list(np.asarray(params.obs_visual_property_std).reshape(-1)),
    }


def _obs_breakdown(params):
    from src.environment.sensor import get_observation_breakdown
    return {k: int(v) for k, v in get_observation_breakdown(params).items()}


def env_dims(params) -> dict:
    from src.environment.sensor import get_observation_breakdown
    return normalise_dims({
        "A": int(np.asarray(params.animal_property).shape[0]),
        "R": int(np.asarray(params.res_property).shape[0]),
        "B": int(np.asarray(params.obs_property).shape[0]),
        "V": int(params.olfactory_vector_size),
        "VV": int(params.visual_vector_size),
        "D": int(sum(get_observation_breakdown(params).values())),
    })


# ── Chunk rollout ─────────────────────────────────────────────────────────────

def make_rollout_fn():
    """Build the `nnx.jit`-wrapped scan kernel ONCE per process.

    This must not be constructed inside `run_chunk`.  `nnx.jit` caches its compiled
    executable on the WRAPPER OBJECT, so a fresh wrapper per chunk retraces and
    recompiles every chunk — at 10^6 episodes and `batch_size=1024` that is ~977 full
    retrace+compiles of a `max_steps`-long scanned environment plus RNN, which would
    plausibly dominate the whole collection.  (The identical pattern at
    `eval_rollout.py:430` is harmless only because it runs once per process.)

    A run's LAST chunk is usually smaller than `batch_size`, which costs exactly one
    extra compile for that shape — correct and cheap.  Partial chunks are sliced, never
    padded, so no filler episode can reach the store.
    """
    import flax.nnx as nnx
    return nnx.jit(traj_scan._rollout_scan, static_argnames=("max_steps", "policy_step"))


def run_chunk(rollout_fn, model, policy_step, initial_state_fn, params, seeds, max_steps: int):
    """Roll out one chunk of episodes and return host-side row dicts.

    `rollout_fn` is the process-wide jitted kernel from `make_rollout_fn()` — pass the
    SAME object every chunk (see that function's docstring).

    Returns `(step_fields, ep_draws, T, reward_sum, term_reason)`.
    """
    import jax
    import jax.numpy as jnp
    from src.environment.core import jax_reset
    from src.environment.sensor import get_observation

    seeds_i64 = jnp.asarray(np.asarray(seeds, dtype=np.int64))
    keys = jax.vmap(jax.random.PRNGKey)(seeds_i64)
    states0 = jax.vmap(jax_reset, in_axes=(None, 0))(params, keys)

    traj_scan._prng_parity_guard(params, seeds_i64, states0)

    v_obs = jax.vmap(get_observation, in_axes=(0, None))
    v_obs_true = jax.vmap(lambda s, p: get_observation(s, p, apply_noise=False), in_axes=(0, None))
    obs0 = v_obs(states0, params)
    obs_true0 = v_obs_true(states0, params)
    h0 = initial_state_fn(len(seeds))

    final_draws, scan_out = rollout_fn(
        model, params, states0, h0, obs0, obs_true0, max_steps, policy_step)

    reset_rows = traj_scan._reset_row(states0, params, obs0, obs_true0)
    reset_draws = traj_scan._draw_block(states0)

    scan_out = jax.tree_util.tree_map(np.asarray, scan_out)
    reset_rows = jax.tree_util.tree_map(np.asarray, reset_rows)
    reset_draws = jax.tree_util.tree_map(np.asarray, reset_draws)
    final_draws = jax.tree_util.tree_map(np.asarray, final_draws)

    # C5 — the per-episode draws the environment treats as episode CONSTANTS must still
    # equal their reset values on the final state; a mismatch means the wrong state was
    # snapshotted.  The two resource `*_init` draws are excluded on purpose — `jax_step`
    # re-draws them on resource regeneration.  See `traj_scan.CONSTANT_DRAW_FIELDS` /
    # `MUTABLE_DRAW_FIELDS` for the mechanism and the line citations.
    unclassified = set(reset_draws) - traj_scan.CONSTANT_DRAW_FIELDS - traj_scan.MUTABLE_DRAW_FIELDS
    if unclassified:
        raise RuntimeError(
            f"Draw field(s) {sorted(unclassified)} are in the per-episode record but are "
            "classified neither constant nor mutable. Decide which, in "
            "traj_scan.CONSTANT_DRAW_FIELDS / MUTABLE_DRAW_FIELDS — an unclassified field "
            "would silently escape the C5 constancy check.")
    for k in sorted(traj_scan.CONSTANT_DRAW_FIELDS):
        if not np.array_equal(reset_draws[k], final_draws[k]):
            raise RuntimeError(
                f"Per-episode draw {k!r} CHANGED during the episode (reset value differs "
                "from the final-state value). Either the environment no longer treats it "
                "as a per-episode constant — in which case move it to "
                "traj_scan.MUTABLE_DRAW_FIELDS and say so in the schema doc — or the "
                "wrong state was snapshotted. Nothing has been written.")

    done = scan_out["done"]                       # (max_steps, B)
    T = np.argmax(done, axis=0) + 1               # max_steps truncation guarantees a hit
    if not bool(np.all(done[T - 1, np.arange(len(seeds))])):
        raise RuntimeError("An episode never reached done within max_steps — "
                           "impossible unless the truncation rule changed.")

    step_fields = traj_scan._flatten_to_rows(scan_out, reset_rows, T, seeds)

    valid = np.arange(done.shape[0])[:, None] < T[None, :]
    reward_sum = np.where(valid, scan_out["reward"], 0.0).sum(axis=0)
    term_reason = scan_out["termination_reason"][T - 1, np.arange(len(seeds))]

    # Every episode row must carry a REAL terminal code; 0 means "still active".  A zero
    # here is one of the two documented latent environment quirks (an injury-disabled
    # config where an instant predator kill leaves the reason below the death threshold),
    # and it would otherwise be recorded as plausible data.  Fail loudly instead.
    if not bool(np.all(term_reason != 0)):
        bad = np.nonzero(term_reason == 0)[0]
        raise RuntimeError(
            f"{bad.size} episode(s) ended with termination_reason == 0 ('still active'), "
            f"first at seed {int(np.asarray(seeds)[bad[0]])}. The schema promises this "
            "column is never 0 on an episode row. This is the documented latent quirk of "
            "termination-reason coding when a body system is switched off (see the schema "
            "doc's caveats); the store would record it as plausible data. Nothing has "
            "been written.")
    return step_fields, reset_draws, T, reward_sum, term_reason


# ── Table assembly ────────────────────────────────────────────────────────────

def steps_table(step_fields, dims, obs_precision):
    f = step_fields
    data = {
        "episode_seed": f["episode_seed"],
        "t": f["t"],
        "action": f["action"],
        "reward": f["reward"],
        "agent_row": f["agent_pos"][:, 0],
        "agent_col": f["agent_pos"][:, 1],
        "satiation": f["satiation"],
        "nutrition": f["nutrition"],
        "injury_level": f["injury_level"],
        "rest_streak": f["rest_streak"],
        "last_collision_noc": f["last_collision_noc"],
        "terminated": f["terminated"],
        "damage": f["damage"],
        "ate_food": f["ate_food"],
        "rested": f["rested"],
        "hit_predator": f["hit_predator"],
        "hit_neutral": f["hit_neutral"],
        "hit_hiding_predator": f["hit_hiding_predator"],
        "event_collided": f["event_collided"],
        "agent_in_bush": f["agent_in_bush"],
        "termination_reason": f["termination_reason"],
        "animal_row": f["animal_pos"][:, :, 0],
        "animal_col": f["animal_pos"][:, :, 1],
        "animal_state": f["animal_state"],
        "animal_stamina": f["animal_stamina"],
        "animal_move_timer": f["animal_move_timer"],
        "animal_attack_timer": f["animal_attack_timer"],
        "res_row": f["res_pos"][:, :, 0],
        "res_col": f["res_pos"][:, :, 1],
        "res_active": f["res_active"],
        "res_cons_count": f["res_cons_count"],
        "res_reg_timer": f["res_reg_timer"],
        "obs_row": f["obs_pos"][:, :, 0],
        "obs_col": f["obs_pos"][:, :, 1],
        "obs_noised": f["obs_noised"],
        "obs_true": f["obs_true"],
    }
    return build_table(STEP_COLUMNS, data, dims, obs_precision)


def episodes_table(draws, seeds, ep_indices, block_id, T, reward_sum, term_reason, dims):
    n = len(seeds)
    data = {
        "episode_seed": np.asarray(seeds, dtype=np.int64),
        "episode_index": np.asarray(ep_indices, dtype=np.int64),
        "block_id": np.full(n, block_id, dtype=np.int32),
        "length": np.asarray(T, dtype=np.int32),
        "termination_reason": np.asarray(term_reason, dtype=np.int8),
        "reward_sum": np.asarray(reward_sum, dtype=np.float32),
    }
    for c in EPISODE_COLUMNS:
        if c.name in data:
            continue
        data[c.name] = np.asarray(draws[c.name]).reshape(n, -1)
    return build_table(EPISODE_COLUMNS, data, dims, obs_precision=None)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="training run directory")
    p.add_argument("--checkpoint", default="final",
                   help="'final' (numeric max, NOT lexicographic) or an explicit step")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--episodes", required=True, type=int,
                   help="TOTAL episodes for this run (defines the episode population)")
    p.add_argument("--seed-base", required=True, type=int,
                   help="episode i uses seed_base + i; shared across runs makes "
                        "comparisons PAIRED (see the schema doc)")
    p.add_argument("--blocks", default=None,
                   help="'lo:hi' half-open block range for this worker; default = all")
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"episodes per chunk; default per device {DEFAULT_BATCH_SIZE}")
    p.add_argument("--shard-episodes", type=int, default=DEFAULT_SHARD_EPISODES)
    p.add_argument("--obs-precision", required=True, choices=list(ts.VALID_OBS_PRECISION),
                   help="MANDATORY and LOSSY: float16 discards information. No default.")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--algo", default="rppo", help="only 'rppo' is supported")
    p.add_argument("--allow-ambiguous-scene", action="store_true",
                   help="downgrade the scene-ambiguity guard to a warning and stamp "
                        "scene_ambiguous:true into the manifest. Use only when you have "
                        "INDEPENDENTLY established which scene the run trained on.")
    p.add_argument("--assume-pre-v32-sensors", action="store_true",
                   help="Supply the two sensory keys commit 9771e98 made mandatory "
                        "(visual_value_mode, visual_occlusion_enabled) at values that leave the "
                        "pre-v3.2 code path untouched. Needed for runs launched before "
                        "2026-08-26 16:00. Refuses if the run already sets them.")
    p.add_argument("--assume-pre-v31-sensors", action="store_true",
                   help="Supply the five sensory keys that commit 0e8a4ef made mandatory "
                        "(olfactory_grid_range, visual_blur_*) at the values that "
                        "reproduce the pre-v3.1 sensors exactly. Required to replay any run "
                        "trained before 2026-08-21. Refuses if the run already sets them.")
    p.add_argument("--allow-weak-restore-check", action="store_true",
                   help="permit collection when the checkpoint's own structure cannot be "
                        "read, degrading the restore guard to a self-comparison that is "
                        "blind to a structurally different model (the "
                        "silent-unmodulated-agent bug). Recorded in the manifest as "
                        "restore_check=weak_allowed so every reader inherits the caveat.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = args.run.resolve()

    ckpt_dir, ckpt_step = resolve_checkpoint(run_dir, args.checkpoint)
    cfg_path = run_dir / "models" / "config.yaml"
    if not cfg_path.exists():
        raise ValueError(f"No saved training config at {cfg_path}. The collector reads "
                         "the run's OWN resolved config, never configs/.")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Ordering matters: the scene guard runs BEFORE params are built, so an ambiguous
    # run cannot get far enough to create a store directory.
    scene_format, scene_ambiguous = assert_scene_unambiguous(
        cfg, run_dir, allow_override=args.allow_ambiguous_scene)

    pre_v31_keys = None
    if args.assume_pre_v31_sensors:
        pre_v31_keys = apply_sensor_compat(cfg, run_dir, PRE_V31_SENSOR_DEFAULTS,
                                           "--assume-pre-v31-sensors", "v3.1")
        if not args.quiet:
            print(f"[collect] --assume-pre-v31-sensors: supplied {pre_v31_keys} "
                  f"at pre-v3.1 values (env fingerprint reflects this)")
    if args.assume_pre_v32_sensors:
        k32 = apply_sensor_compat(cfg, run_dir, PRE_V32_SENSOR_DEFAULTS,
                                  "--assume-pre-v32-sensors", "v3.2")
        pre_v31_keys = sorted((pre_v31_keys or []) + k32)
        if not args.quiet:
            print(f"[collect] --assume-pre-v32-sensors: supplied {k32} "
                  f"at pre-v3.2 values (env fingerprint reflects this)")

    from src.environment.config_loader import load_env_params
    params = load_env_params(Config(cfg))
    dims = env_dims(params)
    max_steps = int(params.max_steps)

    shard_episodes = args.shard_episodes
    n_blocks = (args.episodes + shard_episodes - 1) // shard_episodes
    if shard_episodes < 1000 and n_blocks > 50:
        raise ValueError(
            f"--shard-episodes {shard_episodes} with {args.episodes} episodes would "
            f"create {n_blocks} blocks ({2 * n_blocks} files). File count — not bytes — "
            "is the scarce resource on this NAS (`du -sh results/` already times out "
            "after two minutes), so do not go below ~1000 episodes per shard at scale "
            "(plan §D11). Small shards are allowed for tests, where the block count "
            "stays small.")
    # Resolved (not defaulted-at-use) so the manifest records the value ACTUALLY used.
    batch_size = min(args.batch_size or DEFAULT_BATCH_SIZE[args.device], shard_episodes)

    store_dir = (args.out_root.resolve() / run_dir.name / str(ckpt_step)
                 / env_fingerprint(cfg))
    manifest = build_manifest(
        cfg=cfg, params=params, run_dir=run_dir, ckpt_dir=ckpt_dir, ckpt_step=ckpt_step,
        dims=dims, seed_base=args.seed_base, n_episodes=args.episodes,
        shard_episodes=shard_episodes, batch_size=batch_size, device=args.device, pre_v31_sensor_keys=pre_v31_keys,
        obs_precision=args.obs_precision, scene_format=scene_format,
        scene_ambiguous=scene_ambiguous,
        restore_check="weak_allowed" if args.allow_weak_restore_check else "strict",
    )

    if (store_dir / ts.MANIFEST_NAME).exists():
        assert_manifest_compatible(store_dir, manifest)
        manifest = read_manifest(store_dir)
    else:
        write_manifest(store_dir, manifest)

    if args.blocks:
        lo, hi = (int(x) for x in args.blocks.split(":"))
    else:
        lo, hi = 0, n_blocks
    hi = min(hi, n_blocks)

    model, initial_state_fn, policy_step = load_policy(
        args.algo, ckpt_dir, cfg, params, quiet=args.quiet,
        allow_weak_restore_check=args.allow_weak_restore_check)

    # ONE jitted kernel for the whole process — see make_rollout_fn's docstring. Built
    # here rather than inside run_chunk so the trace+compile happens once, not per chunk.
    rollout_fn = make_rollout_fn()

    done_blocks = completed_blocks(store_dir)
    todo = [b for b in range(lo, hi) if b not in done_blocks]
    if not args.quiet:
        print(f"[collect] store={store_dir}\n[collect] blocks {lo}:{hi} — "
              f"{len(todo)} to do, {len(set(range(lo, hi)) & done_blocks)} already complete",
              flush=True)

    t_start = time.time()
    n_done = 0
    for block in todo:
        clear_partial_shards(store_dir, block)
        b_lo = block * shard_episodes
        b_hi = min(b_lo + shard_episodes, args.episodes)
        ep_indices = np.arange(b_lo, b_hi, dtype=np.int64)
        seeds_all = ep_indices + int(manifest["seed_base"])

        step_parts, ep_parts = [], []
        t_block = time.time()
        for c0 in range(0, len(seeds_all), batch_size):
            seeds = seeds_all[c0:c0 + batch_size]
            sf, draws, T, rsum, treason = run_chunk(
                rollout_fn, model, policy_step, initial_state_fn, params, seeds, max_steps)

            # The float16 range guard — after the flatten, BEFORE the shard write.
            # A raise here leaves nothing on disk (writes are atomic-rename).
            traj_scan.assert_obs_representable(
                sf["obs_noised"], manifest["obs_precision"], params, "obs_noised")
            traj_scan.assert_obs_representable(
                sf["obs_true"], manifest["obs_precision"], params, "obs_true")

            step_parts.append(steps_table(sf, dims, manifest["obs_precision"]))
            ep_parts.append(episodes_table(
                draws, seeds, ep_indices[c0:c0 + len(seeds)], block, T, rsum, treason, dims))

        import pyarrow as pa
        ep_name, st_name = shard_names(block)
        write_shard_atomic(store_dir / ep_name, pa.concat_tables(ep_parts))
        write_shard_atomic(store_dir / st_name, pa.concat_tables(step_parts))
        n_done += len(seeds_all)
        if not args.quiet:
            dt = time.time() - t_block
            print(f"[collect] block {block:05d}: {len(seeds_all)} episodes in {dt:.1f}s "
                  f"({len(seeds_all)/dt:.1f} eps/s)", flush=True)

    if not args.quiet:
        dt = time.time() - t_start
        print(f"[collect] DONE: {n_done} episodes, {len(todo)} blocks, {dt:.1f}s"
              + (f" ({n_done/dt:.1f} eps/s)" if dt > 0 and n_done else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
