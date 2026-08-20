"""Verifications V1, V3, V4, V5, V7, V9, V10 and checkpoints C0/C2/C3/C4/C9/C10 for the
trajectory-collection pipeline.

Plan: docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md
Doc:  docs/environment/TRAJECTORY_STORE_SCHEMA.md

THE PRINCIPLE THIS FILE IS BUILT ON: **every guard is tested as FIRING, not as passing on
clean input.**  A test that feeds valid data to a guard and observes no exception proves
nothing — it passes identically whether the guard works or is commented out.  Four guards
get that treatment here (scene ambiguity, the float16 range guard, the manifest guard, the
strict checkpoint-restore guard); each has a companion test asserting it does NOT fire on
good input, so a guard that always raises is caught too.

V2 (trajectory parity against the legacy per-episode loop), V6 (SIGKILL + resume on the
NAS) and V8 (throughput on a lab node) are operator-run rather than pytest — they need a
real checkpoint rollout, a NAS-hosted hard kill, and a lab node respectively.

Rollout-based tests here build a RANDOMLY-INITIALISED policy rather than restoring a
checkpoint.  That is deliberate and stated: the schema, the row convention, the realised
draws and every guard under test are functions of the environment and the writer, not of
the weights, so requiring a checkpoint would make the suite depend on gitignored NAS data
for no added coverage.  The tests that genuinely concern a checkpoint (C0's restore guard,
C2's numeric selection) use the real one and skip if it is absent.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "eval" / "traj_collect"))

from src.utils.config import Config                                  # noqa: E402
from src.utils import trajectory_store as ts                         # noqa: E402

FIXTURES = _ROOT / "tests" / "fixtures" / "trajectory_collection"
DUAL_FORMAT_FIXTURE = FIXTURES / "dual_format_config.yaml"
SCHEMA_DOC = _ROOT / "docs" / "environment" / "TRAJECTORY_STORE_SCHEMA.md"

# A real, unambiguous (`entities:`-only) training run, used where a genuine artifact adds
# coverage.  Gitignored NAS data — every test that touches it skips when absent.
REAL_RUN = _ROOT / "results" / "JAX_RecurrentPPO" / "20260816-152742_rppo_restpremNH_a10_n112"
# The 12 real dual-format runs measured in plan §A11 (corpus-wide V10 case).
DUAL_FORMAT_RUNS = [
    "20260529-212737_recurrent_ppo_04-sameProp_R4_chasingRabbit_s42",
    "20260609-191226_recurrent_ppo_08-singlePredRabbit_disengage_s42",
    "20260611-150754_logcheck_ckpt100k_log200", "20260611-151553_logcheck_ckpt250k_log500",
    "20260611-150754_logcheck_ckpt250k_log500", "20260611-152445_logcheck_ckpt100k_log200_2M",
    "20260611-151157_logcheck_ckpt100k_log200", "20260611-152456_logcheck_ckpt200k_log500_2M",
    "20260611-151256_logcheck_ckpt250k_log500", "20260611-152707_logcheck_ckpt200k_log500_2M",
    "20260611-182431_logcheck_ckpt200k_log300_2M", "20260611-182449_logcheck_ckpt200k_log400_2M",
]

os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ── shared helpers ────────────────────────────────────────────────────────────

def _base_cfg() -> dict:
    """The real saved training config of REAL_RUN, with a short episode limit so the
    rollout-based tests stay fast.  `max_steps` does not touch anything under test."""
    if not (REAL_RUN / "models" / "config.yaml").exists():
        pytest.skip(f"{REAL_RUN} not present (gitignored NAS data)")
    cfg = yaml.safe_load((REAL_RUN / "models" / "config.yaml").read_text())
    cfg["environment"]["max_steps"] = 40
    return cfg


def _random_policy(params):
    """A randomly-initialised rPPO policy of the right shape — see the module docstring
    for why the rollout tests do not restore a checkpoint."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from src.environment.sensor import get_observation_breakdown
    from src.models.recurrent_ppo_network import ActorCriticRNN

    bd = get_observation_breakdown(params)
    model = ActorCriticRNN(
        input_dim=sum(bd.values()),
        action_dim=4 + int(params.rest_action_enabled) + int(params.eat_action_enabled),
        hidden_size=32, rngs=nnx.Rngs(jax.random.PRNGKey(0)),
        rnn_type="GRU", activation="relu", modulation_config=None,
        observation_breakdown=bd,
        encoding_config={"encoding_mode": "flat", "use_layer_norm": True},
    )

    def policy_step(m, obs, h):
        logits, _v, h_new, _mod = m(obs, h)
        return jnp.argmax(logits, axis=-1), h_new

    return model, (lambda n: model.initial_state(batch_size=n)), policy_step


def collect(store_root: Path, cfg: dict, *, episodes=8, obs_precision="float16",
            seed_base=1_000_000, shard=8, batch=8, obs_hook=None) -> Path:
    """Drive the collector's own flow (minus `load_policy`) into a store directory.

    Everything except the policy comes from `collect_trajectories` itself, so this
    exercises the real manifest, the real chunk rollout, the real guards and the real
    writer.  `obs_hook(step_fields)` lets a test corrupt the observation block after the
    scan and before the guard, which is how the range guard is driven deterministically.
    """
    import pyarrow as pa
    import collect_trajectories as ct
    import traj_scan
    from src.environment.config_loader import load_env_params

    scene_format, scene_ambiguous = ts.assert_scene_unambiguous(cfg, "test", allow_override=False)
    params = load_env_params(Config(copy.deepcopy(cfg)))
    dims = ct.env_dims(params)
    store_dir = store_root / ts.env_fingerprint(cfg)
    manifest = ct.build_manifest(
        cfg=cfg, params=params, run_dir=REAL_RUN, ckpt_dir=REAL_RUN / "models" / "0",
        ckpt_step=0, dims=dims, seed_base=seed_base, n_episodes=episodes,
        shard_episodes=shard, batch_size=batch, device="cpu",
        obs_precision=obs_precision, scene_format=scene_format,
        scene_ambiguous=scene_ambiguous)

    if (store_dir / ts.MANIFEST_NAME).exists():
        ts.assert_manifest_compatible(store_dir, manifest)
        manifest = ts.read_manifest(store_dir)
    else:
        ts.write_manifest(store_dir, manifest)

    model, init_fn, policy_step = _random_policy(params)
    rollout_fn = ct.make_rollout_fn()          # ONE per process, as production does
    done = ts.completed_blocks(store_dir)
    for block in range(0, (episodes + shard - 1) // shard):
        if block in done:
            continue
        ts.clear_partial_shards(store_dir, block)
        idx = np.arange(block * shard, min((block + 1) * shard, episodes), dtype=np.int64)
        seeds = idx + seed_base
        sf, draws, T, rsum, treason = ct.run_chunk(
            rollout_fn, model, policy_step, init_fn, params, seeds, int(params.max_steps))
        if obs_hook is not None:
            obs_hook(sf)
        traj_scan.assert_obs_representable(sf["obs_noised"], manifest["obs_precision"],
                                           params, "obs_noised")
        traj_scan.assert_obs_representable(sf["obs_true"], manifest["obs_precision"],
                                           params, "obs_true")
        ep_name, st_name = ts.shard_names(block)
        ts.write_shard_atomic(store_dir / ep_name, ct.episodes_table(
            draws, seeds, idx, block, T, rsum, treason, dims))
        ts.write_shard_atomic(store_dir / st_name, ct.steps_table(
            sf, dims, manifest["obs_precision"]))
    return store_dir


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """One collected store, reused by the read-side verifications."""
    return collect(tmp_path_factory.mktemp("store"), _base_cfg(), episodes=8)


def _snapshot(d: Path) -> dict:
    return {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(d.iterdir()) if p.is_file()}


# ══════════════════════════════════════════════════════════════════════════════
# V10 — the scene-ambiguity guard FIRES (regression test for the §A11 hazard)
# ══════════════════════════════════════════════════════════════════════════════

def test_v10_scene_guard_fires_on_real_dual_format_config(tmp_path):
    """Primary case: a byte-identical copy of a REAL dual-format saved config."""
    cfg = yaml.safe_load(DUAL_FORMAT_FIXTURE.read_text())
    assert cfg["environment"].get("entities"), "fixture must carry a modern entities block"
    assert cfg["environment"].get("predators"), "fixture must carry a legacy predators block"

    out = tmp_path / "store_root"
    with pytest.raises(ValueError) as ei:
        ts.assert_scene_unambiguous(cfg, DUAL_FORMAT_FIXTURE, allow_override=False)
    msg = str(ei.value)
    assert "entities" in msg and "predators" in msg and "neutral_animals" in msg
    assert "828b77e" in msg and "2026-07-23" in msg
    assert "§A11" in msg
    # The guard runs BEFORE params are built, so no store directory can exist.
    assert not out.exists(), "the guard must fire before anything is created on disk"


def test_v10_scene_guard_fires_through_the_collector_cli(tmp_path):
    """End-to-end: the collector itself refuses, and creates no store directory."""
    run = next((REAL_RUN.parent / r for r in DUAL_FORMAT_RUNS
                if (REAL_RUN.parent / r / "models" / "config.yaml").exists()), None)
    if run is None:
        pytest.skip("no dual-format run present (gitignored NAS data)")
    import collect_trajectories as ct
    out = tmp_path / "traj"
    with pytest.raises(ValueError, match="828b77e"):
        ct.main(["--run", str(run), "--out-root", str(out), "--episodes", "8",
                 "--seed-base", "0", "--shard-episodes", "8", "--obs-precision", "float16"])
    assert not out.exists(), "no store directory may be created for a refused run"


def test_v10_scene_guard_fires_on_the_whole_known_corpus():
    """All 12 real dual-format runs (plan §A11).  Skip-if-absent: `results/` is
    gitignored NAS data and is not present on a fresh clone."""
    present = [REAL_RUN.parent / r for r in DUAL_FORMAT_RUNS
               if (REAL_RUN.parent / r / "models" / "config.yaml").exists()]
    if not present:
        pytest.skip("none of the 12 dual-format runs present (gitignored NAS data)")
    for run in present:
        cfg = yaml.safe_load((run / "models" / "config.yaml").read_text())
        with pytest.raises(ValueError, match="828b77e"):
            ts.assert_scene_unambiguous(cfg, run, allow_override=False)


def test_v10_scene_guard_does_not_fire_on_unambiguous_configs():
    """The companion: a guard that always raises is not a guard.  Also pins the
    scene_format value that goes into the manifest."""
    dual = yaml.safe_load(DUAL_FORMAT_FIXTURE.read_text())

    entities_only = copy.deepcopy(dual)
    entities_only["environment"].pop("predators")
    entities_only["environment"].pop("neutral_animals")
    assert ts.assert_scene_unambiguous(entities_only, "x", False) == ("entities", False)

    legacy_only = copy.deepcopy(dual)
    legacy_only["environment"].pop("entities")
    assert ts.assert_scene_unambiguous(legacy_only, "x", False) == ("legacy", False)

    none_scene = copy.deepcopy(legacy_only)
    none_scene["environment"]["predators"] = []
    none_scene["environment"]["neutral_animals"] = []
    assert ts.assert_scene_unambiguous(none_scene, "x", False) == ("none", False)


def test_v10_scene_guard_override_downgrades_to_a_warning():
    cfg = yaml.safe_load(DUAL_FORMAT_FIXTURE.read_text())
    with pytest.warns(UserWarning, match="AMBIGUOUS SCENE"):
        fmt, ambiguous = ts.assert_scene_unambiguous(cfg, "x", allow_override=True)
    assert fmt == "legacy", "the loader takes the legacy branch; record what is built"
    assert ambiguous is True, (
        "the guard must REPORT the ambiguity flag rather than leaving each caller to "
        "re-derive the predicate — a third copy is a third thing that can drift")


def test_n1_guard_predicate_matches_the_loader_exactly():
    """N1: the guard's entities predicate must be `is not None`, matching
    `src/environment/config_loader.py:429`, NOT `bool(...)`.  They diverge on a
    present-but-EMPTY `entities:` alongside a legacy block — the shape that a
    `bool(...)` predicate would wave through while the loader still resolves it by
    precedence."""
    cfg = yaml.safe_load(DUAL_FORMAT_FIXTURE.read_text())
    cfg["environment"]["entities"] = []          # present but empty + legacy still there
    with pytest.raises(ValueError, match="828b77e"):
        ts.assert_scene_unambiguous(cfg, "x", allow_override=False)

    src = (_ROOT / "src" / "environment" / "config_loader.py").read_text()
    assert "config.get('environment.entities') is not None" in src, (
        "the loader's predicate changed; re-check assert_scene_unambiguous against it")


# ══════════════════════════════════════════════════════════════════════════════
# V5 — observation precision: the range guard FIRES, and fidelity holds
# ══════════════════════════════════════════════════════════════════════════════

def _params_for_guard():
    from src.environment.config_loader import load_env_params
    return load_env_params(Config(_base_cfg()))


def test_v5b_range_guard_fires_on_an_out_of_range_value(tmp_path):
    """Drive a synthetic out-of-range observation and assert the COLLECTION fails rather
    than silently clipping — and that no shard is left on disk."""
    def hook(sf):
        sf["obs_noised"][3, 22] = 1.2e5        # far above OBS_ABS_MAX and float16's cliff

    root = tmp_path / "root"
    with pytest.raises(ValueError) as ei:
        collect(root, _base_cfg(), episodes=8, obs_hook=hook)
    msg = str(ei.value)
    assert "index 22" in msg, msg
    assert "OBS_ABS_MAX" in msg or "1.2e+05" in msg or "120000" in msg, msg
    # The sensor NAME must be resolved, not just the index.
    from src.environment.sensor import get_observation_breakdown
    names = list(get_observation_breakdown(_params_for_guard()).keys())
    assert any(n in msg for n in names), f"no sensor name in message: {msg}"
    stores = list(root.glob("*/"))
    assert stores, "the manifest is written before the rollout, so the dir should exist"
    assert not list(stores[0].glob("*.parquet")), "a raise must leave NO shard on disk"
    assert not list(stores[0].glob("*.tmp")), "a raise must leave no partial file either"


def test_v5b_range_guard_fires_on_a_location_sensor_config(tmp_path):
    """A real channel carrying large magnitudes: the location sensor emits raw grid
    coordinates.  Scaled here to a grid large enough to exceed the ceiling, so the guard
    is exercised by genuine sensor output rather than only by an injected value."""
    from src.environment.config_loader import load_env_params
    from src.environment.sensor import get_observation_breakdown
    cfg = _base_cfg()
    cfg["sensory"]["location_sensor"] = True
    params = load_env_params(Config(copy.deepcopy(cfg)))
    assert "Location" in get_observation_breakdown(params), \
        "location sensor did not turn on — the guard test would be vacuous"

    import traj_scan
    D = sum(get_observation_breakdown(params).values())
    obs = np.zeros((4, D), dtype=np.float32)
    obs[:, D - 1] = 1.5e5                       # a location channel on a huge grid
    with pytest.raises(ValueError, match="Location"):
        traj_scan.assert_obs_representable(obs, "float16", params, "obs_noised")


def test_v5b_range_guard_does_not_fire_on_in_range_data():
    """The companion: the guard must pass real, in-range observations."""
    import traj_scan
    params = _params_for_guard()
    from src.environment.sensor import get_observation_breakdown
    D = sum(get_observation_breakdown(params).values())
    rng = np.random.default_rng(0)
    obs = rng.random((256, D), dtype=np.float32)
    traj_scan.assert_obs_representable(obs, "float16", params, "obs_noised")
    traj_scan.assert_obs_representable(obs, "float32", params, "obs_noised")


def test_v5b_range_guard_tolerates_tiny_subnormal_values():
    """Regression for the plan-correction recorded in `traj_scan.OBS_ABS_ERR_MAX`.

    A RELATIVE round-trip criterion (what the plan specified) fires on a legitimate
    near-zero reading — measured live on a real 5,000-episode collection: an
    interoceptive-nociception value of 1.34e-06 stored as 1.37e-06 is a 2.2 % relative
    error but a 2.9e-08 ABSOLUTE one, about 3e5 times below the sigma = 0.10 noise the
    environment injects into that channel on purpose.  The guard must not refuse it, or
    the plan's own default precision is uncollectable."""
    import traj_scan
    params = _params_for_guard()
    from src.environment.sensor import get_observation_breakdown
    D = sum(get_observation_breakdown(params).values())
    obs = np.zeros((4, D), dtype=np.float32)
    obs[1, 1] = 1.3415422e-06          # the exact value that tripped the relative form
    traj_scan.assert_obs_representable(obs, "float16", params, "obs_noised")


def test_v5b_range_guard_fires_before_the_outer_backstop():
    """The tight clause (absolute round-trip error) must have teeth well BEFORE the
    OBS_ABS_MAX = 1e4 backstop — otherwise it is decoration.  With a 1e-2 budget it fires
    once a channel's magnitude reaches ~32, which is ~4.6x the largest magnitude the
    reference config actually produces (olfaction, 6.95) — close enough to be a realistic
    drift on a larger grid or a higher `sensor_radius`, and 300x below the backstop."""
    import traj_scan
    params = _params_for_guard()
    from src.environment.sensor import get_observation_breakdown
    D = sum(get_observation_breakdown(params).values())
    obs = np.zeros((4, D), dtype=np.float32)
    obs[:, 3] = 40.02                  # |x| far below OBS_ABS_MAX, error above the budget
    assert np.abs(obs).max() < traj_scan.OBS_ABS_MAX / 100
    with pytest.raises(ValueError, match="ABSOLUTE error"):
        traj_scan.assert_obs_representable(obs, "float16", params, "obs_noised")


def test_v5b_range_guard_fires_on_nonfinite():
    import traj_scan
    params = _params_for_guard()
    from src.environment.sensor import get_observation_breakdown
    D = sum(get_observation_breakdown(params).values())
    obs = np.zeros((4, D), dtype=np.float32)
    obs[1, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        traj_scan.assert_obs_representable(obs, "float16", params, "obs_noised")
    with pytest.raises(ValueError, match="non-finite"):
        traj_scan.assert_obs_representable(obs, "float32", params, "obs_noised")


def test_v5a_precision_fidelity(tmp_path, store):
    """float16: within float16's worst case on [0,1].  float32: BIT-IDENTICAL, because
    compression is lossless — anything else would mean an unintended downcast."""
    import pyarrow.parquet as pq
    st16 = ts.open_store(store)
    tbl = st16.steps(columns=["obs_noised"])
    o16 = st16.to_2d(tbl, "obs_noised").astype(np.float32)
    assert st16.obs_precision == "float16"
    assert np.abs(o16[(o16 >= 0) & (o16 <= 1)]).size > 0
    # Quantisation error is bounded by float16's own worst case on the unit interval.
    err = np.abs(o16 - o16.astype(np.float16).astype(np.float32))
    assert err.max() == 0.0, "already float16 — the round-trip must be exact"

    st32 = ts.open_store(collect(tmp_path / "f32", _base_cfg(), episodes=8,
                                 obs_precision="float32"))
    t32 = st32.steps(columns=["obs_noised"])
    o32 = st32.to_2d(t32, "obs_noised")
    assert o32.dtype == np.float32
    raw = pq.read_table(sorted(Path(st32.path).glob("steps_*.parquet"))[0],
                        columns=["obs_noised"])
    assert raw.schema.field("obs_noised").type.value_type == __import__("pyarrow").float32()
    # Same episodes, same seeds -> float16 must be the float32 values rounded, nothing else.
    assert o16.shape == o32.shape
    assert np.array_equal(o16, o32.astype(np.float16).astype(np.float32))
    # float16 keeps ~10 explicit mantissa bits, so half-ulp rounding is bounded by
    # 2**-11 RELATIVE. On the unit interval that is the 2.44e-04 absolute figure quoted
    # in the schema doc; this config's observation vector is not entirely bounded by 1
    # (see the report), so the honest assertion is the relative one plus the unit-interval
    # absolute one restricted to values that are actually in [0, 1].
    assert np.abs(o32 - o16).max() <= 2.0 ** -11 * max(1.0, float(np.abs(o32).max()))
    unit = np.abs(o32) <= 1.0
    assert unit.any()
    assert np.abs(o32[unit] - o16[unit]).max() <= 2.44e-04


# ══════════════════════════════════════════════════════════════════════════════
# V7 — the manifest / overwrite-hazard guard FIRES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("case", ["mutated_env_config", "seed_base", "schema_version",
                                  "obs_precision"])
def test_v7_manifest_guard_fires_and_writes_nothing(tmp_path, case):
    """Direct regression test for the 2026-07-04 contamination incident.  Case
    'obs_precision' additionally guarantees no store can end up half float16 and half
    float32 — a silent mixed-precision corpus would be undetectable at read time."""
    cfg = _base_cfg()
    store_dir = collect(tmp_path / "root", cfg, episodes=8)
    before = _snapshot(store_dir)
    assert before, "precondition: the store must have files to protect"

    expected = ts.read_manifest(store_dir)
    if case == "mutated_env_config":
        mutated = copy.deepcopy(cfg)
        mutated["environment"]["width"] = int(mutated["environment"]["width"]) + 1
        expected = dict(expected, env_fp=ts.env_fingerprint(mutated))
    elif case == "seed_base":
        expected = dict(expected, seed_base=expected["seed_base"] + 1)
    elif case == "schema_version":
        expected = dict(expected, schema_version=ts.SCHEMA_VERSION + 1)
    elif case == "obs_precision":
        expected = dict(expected, obs_precision="float32")

    with pytest.raises(ValueError, match="Manifest mismatch|Unknown SCHEMA_VERSION"):
        ts.assert_manifest_compatible(store_dir, expected)
    assert _snapshot(store_dir) == before, "a refused resume must modify NO file"


def test_v7_manifest_guard_does_not_fire_on_an_identical_resume(tmp_path):
    store_dir = collect(tmp_path / "root", _base_cfg(), episodes=8)
    ts.assert_manifest_compatible(store_dir, ts.read_manifest(store_dir))


def test_reader_rejects_an_unknown_schema_version(tmp_path):
    store_dir = collect(tmp_path / "root", _base_cfg(), episodes=8)
    p = store_dir / ts.MANIFEST_NAME
    m = json.loads(p.read_text())
    m["schema_version"] = ts.SCHEMA_VERSION + 99
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Unknown SCHEMA_VERSION"):
        ts.open_store(store_dir)


# ══════════════════════════════════════════════════════════════════════════════
# C0 / F3 — the strict checkpoint-restore guard FIRES
# ══════════════════════════════════════════════════════════════════════════════

def _real_checkpoint_tree():
    import orbax.checkpoint as ocp
    models = REAL_RUN / "models"
    if not models.is_dir():
        pytest.skip("real run not present (gitignored NAS data)")
    step = max(int(p.name) for p in models.iterdir() if p.is_dir() and p.name.isdigit())
    md = ocp.PyTreeCheckpointer().metadata(str(models / str(step) / "default"))
    tree = getattr(getattr(md, "item_metadata", None), "tree", None)
    if not isinstance(tree, dict) or "model" not in tree:
        pytest.skip("checkpoint metadata unavailable in this orbax version")
    return tree["model"]


def _build_model(cfg: dict, params, **overrides):
    import flax.nnx as nnx
    import jax
    from src.environment.sensor import get_observation_breakdown
    from src.models.recurrent_ppo_network import ActorCriticRNN
    a = dict(cfg["agent"])
    a.update(overrides)
    bd = get_observation_breakdown(params)
    mod = a.get("modulation")
    if mod is not None and mod.get("type") is None:
        mod = None
    return ActorCriticRNN(
        input_dim=sum(bd.values()),
        action_dim=4 + int(params.rest_action_enabled) + int(params.eat_action_enabled),
        hidden_size=a["hidden_size"], rngs=nnx.Rngs(jax.random.PRNGKey(0)),
        rnn_type=a["rnn_type"], activation=a["activation"], modulation_config=mod,
        observation_breakdown=bd, encoding_config=a)


def test_c0_restore_guard_does_not_fire_on_the_correct_model():
    """Companion first: the guard must ACCEPT the model the checkpoint was trained with,
    otherwise 'it raised' proves nothing below."""
    import flax.nnx as nnx
    from src.environment.config_loader import load_env_params
    cfg = yaml.safe_load((REAL_RUN / "models" / "config.yaml").read_text())
    params = load_env_params(Config(copy.deepcopy(cfg)))
    model = _build_model(cfg, params)
    ts.assert_restored_tree_matches(nnx.state(model), model,
                                    checkpoint_tree=_real_checkpoint_tree())


def test_c0_restore_guard_fires_on_a_wrong_model_size():
    """Model-size CLI flags that failed to reach the saved config would build a
    differently-shaped model that still 'restores successfully'."""
    import flax.nnx as nnx
    from src.environment.config_loader import load_env_params
    cfg = yaml.safe_load((REAL_RUN / "models" / "config.yaml").read_text())
    params = load_env_params(Config(copy.deepcopy(cfg)))
    model = _build_model(cfg, params, hidden_size=64)
    with pytest.raises(ValueError, match="shape mismatch|MISSING"):
        ts.assert_restored_tree_matches(nnx.state(model), model,
                                        checkpoint_tree=_real_checkpoint_tree())


def test_c0_restore_guard_fires_on_a_structurally_different_encoder():
    """The silent-wrong-architecture bug — the same shape as the recorded
    'a missing agent-modulation block silently builds an UNMODULATED baseline' case:
    an `agent:` block that does not match the checkpoint produces a model with a
    DIFFERENT KEY SET, which orbax's `partial_restore` fills without a word of complaint
    because it only ever restores the keys the target declares.

    Here the checkpoint was trained with `encoding_mode: hierarchical`; rebuilding it flat
    changes which encoder submodules exist. The guard must see the missing/extra keys."""
    import flax.nnx as nnx
    from src.environment.config_loader import load_env_params
    cfg = yaml.safe_load((REAL_RUN / "models" / "config.yaml").read_text())
    assert cfg["agent"]["encoding_mode"] == "hierarchical", "precondition of this test"
    params = load_env_params(Config(copy.deepcopy(cfg)))
    model = _build_model(cfg, params, encoding_mode="flat")
    with pytest.raises(ValueError, match="MISSING from the checkpoint|MISSING from the built model"):
        ts.assert_restored_tree_matches(nnx.state(model), model,
                                        checkpoint_tree=_real_checkpoint_tree())


# ══════════════════════════════════════════════════════════════════════════════
# C2 — numeric checkpoint selection
# ══════════════════════════════════════════════════════════════════════════════

def test_c0_metadata_is_readable_so_the_strict_check_is_not_vacuous():
    """NOT a skip. If orbax stops exposing the checkpoint's structure, the restore guard
    silently degrades to a self-comparison AND every test above it degrades with it — the
    guard and its tests would go vacuous simultaneously, with no red signal anywhere.
    Failing here is the red signal."""
    import collect_trajectories as ct
    if not (REAL_RUN / "models").is_dir():
        pytest.skip("real run not present (gitignored NAS data)")
    ckpt, _ = ct.resolve_checkpoint(REAL_RUN, "final")
    tree = ct.read_checkpoint_tree(ckpt)
    assert tree is not None, (
        "orbax no longer exposes the checkpoint's own key set. The strict restore check "
        "(F3) cannot run, and the silent-unmodulated-agent bug becomes undetectable. Fix "
        "read_checkpoint_tree for this orbax version — do NOT relax this to a skip.")
    assert len(ts._tree_shapes(tree)) > 5


def test_c0_weak_restore_check_is_refused_by_default(tmp_path, monkeypatch):
    """The degradation path must HARD-FAIL, not print a warning.

    Under `xargs -P 16` with per-cell log redirection, a printed warning is not a control:
    nobody reads 160 log files. And the degraded comparison is structurally blind to the
    checkpoint's extra keys — exactly the silent-unmodulated-agent case the guard exists
    for."""
    import collect_trajectories as ct
    if not (REAL_RUN / "models").is_dir():
        pytest.skip("real run not present (gitignored NAS data)")
    monkeypatch.setattr(ct, "read_checkpoint_tree", lambda _d: None)
    out = tmp_path / "traj"
    with pytest.raises(ValueError, match="allow-weak-restore-check"):
        ct.main(["--run", str(REAL_RUN), "--out-root", str(out), "--episodes", "4",
                 "--seed-base", "0", "--shard-episodes", "4", "--batch-size", "4",
                 "--obs-precision", "float32", "--quiet"])
    assert not list(out.rglob("*.parquet")), "a refused restore must write no shard"


def test_c0_weak_restore_check_is_recorded_in_the_manifest(tmp_path, monkeypatch):
    """With the escape hatch, collection proceeds but the store says so — and
    `restore_check` is manifest-guarded, so a store cannot be half strict and half weak."""
    import collect_trajectories as ct
    if not (REAL_RUN / "models").is_dir():
        pytest.skip("real run not present (gitignored NAS data)")
    monkeypatch.setattr(ct, "read_checkpoint_tree", lambda _d: None)
    out = tmp_path / "traj"
    with pytest.warns(UserWarning, match="WEAK RESTORE CHECK"):
        ct.main(["--run", str(REAL_RUN), "--out-root", str(out), "--episodes", "4",
                 "--seed-base", "0", "--shard-episodes", "4", "--batch-size", "4",
                 "--obs-precision", "float32", "--quiet", "--allow-weak-restore-check"])
    store = next(out.rglob("_manifest.json")).parent
    m = ts.read_manifest(store)
    assert m["restore_check"] == "weak_allowed"
    assert "restore_check" in ts.MANIFEST_GUARDED_FIELDS
    with pytest.raises(ValueError, match="Manifest mismatch"):
        ts.assert_manifest_compatible(store, dict(m, restore_check="strict"))


def test_c2_checkpoint_selection_is_numeric_not_lexicographic():
    import collect_trajectories as ct
    if not (REAL_RUN / "models").is_dir():
        pytest.skip("real run not present (gitignored NAS data)")
    names = [p.name for p in (REAL_RUN / "models").iterdir()
             if p.is_dir() and p.name.isdigit()]
    ckpt, step = ct.resolve_checkpoint(REAL_RUN, "final")
    assert step == max(int(n) for n in names)
    # The test only has teeth where the two orderings actually disagree.
    assert max(names) != str(step), (
        "this run no longer distinguishes lexicographic from numeric ordering; "
        "the check is vacuous here")
    assert ckpt.name == str(step)


# ══════════════════════════════════════════════════════════════════════════════
# V1 — realised draws against an independent replay (the primary correctness check)
# ══════════════════════════════════════════════════════════════════════════════

# The ONE realised-draw column on which two runs of `jax_reset` are not bit-identical
# ACROSS COMPILATIONS, and the measured bound on that difference.  See
# `test_v1_animal_property_divergence_is_one_float32_ulp` for the diagnosis.
_XLA_DIVERGENT_DRAW_COLUMN = "animal_property_sampled"
_XLA_DIVERGENCE_ABS_MAX = 6e-08          # one float32 ULP at magnitude 1.0 (2**-24)


def test_v1_realised_draws_match_an_independent_unbatched_replay(store):
    """For each recorded episode, call `jax_reset` UNBATCHED, un-vmapped, outside any
    scan — a genuinely different execution path — and assert EXACT equality on the
    realised-draw columns.

    Tolerance policy: **exact equality, including for float columns**, for 16 of the 17
    columns.  Both paths execute the identical sampling arithmetic on the identical key,
    so the outputs are bit-identical or something is genuinely wrong.

    THE ONE DOCUMENTED EXCEPTION is `animal_property_sampled`, and it was diagnosed rather
    than waved through — see `test_v1_animal_property_divergence_is_one_float32_ulp` for
    the mechanism and the measured bound.  It is a compiler-fusion effect in the
    environment's own reset, not a property of this replay path.  Every other column,
    including the other five float property columns drawn by the same `_sample_property`
    helper, is exact.
    """
    import jax
    from src.environment.config_loader import load_env_params
    from src.environment.core import jax_reset
    import traj_scan

    st = ts.open_store(store)
    params = load_env_params(Config(yaml.safe_load(json.dumps(st.manifest["resolved_env_config"]))))
    ep = st.episodes()
    seeds = np.asarray(ep.column("episode_seed"))

    for i, seed in enumerate(seeds.tolist()):
        ref = traj_scan._draw_block(jax_reset(params, jax.random.PRNGKey(seed)))
        for col in ts.DRAW_COLUMNS:
            got = st.to_2d(ep, col)[i]
            want = np.asarray(ref[col]).reshape(-1)
            if col == _XLA_DIVERGENT_DRAW_COLUMN:
                assert np.abs(got.astype(np.float64) - want.astype(np.float64)).max() \
                    <= _XLA_DIVERGENCE_ABS_MAX, (
                    f"episode_seed={seed}, column {col}: divergence exceeds the "
                    f"documented one-ULP bound")
                continue
            assert np.array_equal(got, want), (
                f"episode_seed={seed}, column {col}: store {got!r} != independent "
                f"replay {want!r}")


def test_v1_animal_property_divergence_is_one_float32_ulp():
    """DIAGNOSIS + BOUND for the one column where two runs of the environment's reset are
    not bit-identical.

    WHAT: `jax_reset` on the same seed can yield a different `animal_property_sampled` for
    a minority of elements — measured 242 of 5,120 over 256 episodes on the reference
    training config — by at most **5.96e-08 absolute**, i.e. one float32 ULP at magnitude
    1.0.  Every integer and boolean draw is always exact, and no other realised-draw column
    diverges at all — including `res_property_sampled` and `obs_property_sampled`, drawn by
    the SAME `_sample_property` helper.

    WHY: **compiler-level arithmetic reordering.**  XLA may emit `mean + std * noise`
    either as a separate multiply and add, or fused into a single fused multiply-add.  FMA
    carries more intermediate precision, so the two forms differ in the last bit, and which
    one XLA picks depends on how the surrounding code lowers.  `animal_property_sampled` is
    the one property column assembled next to a `jnp.zeros_like(...).at[idx].set(...)`
    scatter (`src/environment/core.py:1128-1140`, the N2 per-class split), and that
    neighbouring scatter changes the fusion decision — which is exactly why its five
    siblings using the same helper are unaffected.

    NOT BATCHING.  The first explanation offered was that `vmap` was responsible; that was
    **refuted**.  `senior-developer` reproduced the same divergence — same column, same
    <=5.867e-08 magnitude, zero integer or boolean differences — using the project's
    shipped parity-fixture generator, which calls unbatched `jax_reset` only and contains
    no `vmap` anywhere.  Two unbatched runs disagreeing by exactly the effect blamed on
    batching eliminates batching as the cause.  This is a property of the ENVIRONMENT under
    different compilations, not of the batched collector — the pipeline merely happened to
    be the thing that noticed, because `eval_rollout.py` never recorded property draws.

    Full evidence chain (do not restate it here — link it):
    docs/llm_wiki/entries/env_entities/20260820_1606_reset_ulp_divergence_is_compiler_fusion.md

    This test PINS the bound: if the divergence ever grows beyond one ULP, or spreads to
    another column, that is a different phenomenon and must be re-diagnosed rather than
    absorbed into a wider tolerance.  It compares a batched against an unbatched reset only
    because that is a cheap way to get two DIFFERENT COMPILATIONS of the same code in one
    process — the batching is the instrument, not the cause.
    """
    import jax
    from src.environment.config_loader import load_env_params
    from src.environment.core import jax_reset
    import traj_scan

    params = load_env_params(Config(_base_cfg()))
    seeds = np.arange(2_000_000, 2_000_032, dtype=np.int64)
    keys = jax.vmap(jax.random.PRNGKey)(seeds)
    # Two different COMPILATIONS of the same reset code. The batched form is used here
    # purely because it is the cheapest way to obtain a second lowering in-process; the
    # divergence is not caused by batching (see the docstring).
    compiled_a = traj_scan._draw_block(jax.vmap(jax_reset, in_axes=(None, 0))(params, keys))

    worst = 0.0
    other_cols_exact = True
    for i, s in enumerate(seeds.tolist()):
        compiled_b = traj_scan._draw_block(jax_reset(params, jax.random.PRNGKey(s)))
        for col in ts.DRAW_COLUMNS:
            a = np.asarray(compiled_a[col])[i].reshape(-1)
            b = np.asarray(compiled_b[col]).reshape(-1)
            if a.size == 0:
                continue
            if col == _XLA_DIVERGENT_DRAW_COLUMN:
                worst = max(worst, float(np.abs(a.astype(np.float64)
                                                - b.astype(np.float64)).max()))
            elif not np.array_equal(a, b):
                other_cols_exact = False

    assert other_cols_exact, (
        "a realised-draw column other than animal_property_sampled now differs between "
        "two compilations of jax_reset — this is a NEW phenomenon, diagnose it rather "
        "than widening the tolerance")
    assert worst <= _XLA_DIVERGENCE_ABS_MAX, (
        f"animal_property_sampled divergence grew to {worst:.3e}, beyond the documented "
        f"one-ULP bound {_XLA_DIVERGENCE_ABS_MAX:.3e}")


def test_v1_would_catch_a_seed_to_row_misassociation(store):
    """The realistic failure mode V1 exists for is a seed-to-row shift.  Confirm the
    assertion is not vacuous: rotating the recorded rows by one must break it."""
    import jax
    from src.environment.config_loader import load_env_params
    from src.environment.core import jax_reset
    import traj_scan

    st = ts.open_store(store)
    params = load_env_params(Config(yaml.safe_load(json.dumps(st.manifest["resolved_env_config"]))))
    ep = st.episodes()
    seeds = np.asarray(ep.column("episode_seed"))
    col = "animal_detect_sampled"
    rolled = np.roll(st.to_2d(ep, col), 1, axis=0)
    mismatch = sum(
        not np.array_equal(rolled[i],
                           np.asarray(traj_scan._draw_block(
                               jax_reset(params, jax.random.PRNGKey(int(s))))[col]).reshape(-1))
        for i, s in enumerate(seeds.tolist()))
    assert mismatch > 0, "a one-row shift went undetected — V1 would be vacuous here"


# ══════════════════════════════════════════════════════════════════════════════
# V3 / C4 — row-convention self-consistency
# ══════════════════════════════════════════════════════════════════════════════

def test_v3_row_convention(store):
    st = ts.open_store(store)
    ep = st.episodes(columns=["episode_seed", "length", "termination_reason"])
    steps = st.steps(columns=["episode_seed", "t", "action", "reward", "terminated",
                              "termination_reason"]).to_pandas()
    lengths = dict(zip(np.asarray(ep.column("episode_seed")).tolist(),
                       np.asarray(ep.column("length")).tolist()))
    reasons = dict(zip(np.asarray(ep.column("episode_seed")).tolist(),
                       np.asarray(ep.column("termination_reason")).tolist()))

    for seed, g in steps.groupby("episode_seed"):
        g = g.sort_values("t")
        T = lengths[seed]
        assert len(g) == T + 1, f"seed {seed}: {len(g)} rows, expected length+1 = {T+1}"
        assert g.iloc[0]["t"] == 0
        assert g.iloc[0]["action"] == -1
        assert g.iloc[0]["reward"] == 0.0
        assert (g.iloc[:-1]["termination_reason"] == 0).all()
        assert g.iloc[-1]["termination_reason"] == reasons[seed] != 0
        assert bool(g.iloc[-1]["terminated"]) is True


# ══════════════════════════════════════════════════════════════════════════════
# V4 — agent_in_bush recomputed independently in NumPy
# ══════════════════════════════════════════════════════════════════════════════

def test_v4_agent_in_bush_recomputed_in_numpy(store):
    """A pure-NumPy reimplementation sharing NO code with the JAX environment.  Fails on
    the slot-0 hardcode of avoidance_stats_heatmap.py:77-79, a wrong obs_active mask, or
    a mis-transcribed reset-row helper."""
    st = ts.open_store(store)
    ep = st.episodes(columns=["episode_seed", "obs_active"])
    active = {int(s): row for s, row in
              zip(np.asarray(ep.column("episode_seed")), st.to_2d(ep, "obs_active"))}
    hides = np.asarray(st.manifest["obs_hides_agent"], dtype=bool)

    steps = st.steps(columns=["episode_seed", "t", "agent_row", "agent_col",
                              "obs_row", "obs_col", "agent_in_bush"])
    seeds = np.asarray(steps.column("episode_seed"))
    ar = np.asarray(steps.column("agent_row"))
    ac = np.asarray(steps.column("agent_col"))
    orow = st.to_2d(steps, "obs_row")
    ocol = st.to_2d(steps, "obs_col")
    got = np.asarray(steps.column("agent_in_bush"))

    act = np.stack([active[int(s)] for s in seeds])
    expected = np.any((orow == ar[:, None]) & (ocol == ac[:, None]) & hides[None, :] & act,
                      axis=1)
    assert np.array_equal(expected, got)
    assert expected.any(), "no row is in a bush — the comparison would be vacuous"


# ══════════════════════════════════════════════════════════════════════════════
# V9 — schema invariance across structurally different environments
# ══════════════════════════════════════════════════════════════════════════════

def _variant_configs():
    """Three structurally different environments: fewer animal slots, the full training
    world, and a location-sensor world with a different observation dimension."""
    base = _base_cfg()

    one_pred = copy.deepcopy(base)
    ents = one_pred["environment"]["entities"]
    keep = copy.deepcopy(ents[0])
    keep["count_low"], keep["count_high"] = 1, 1
    one_pred["environment"]["entities"] = [keep]

    location = copy.deepcopy(base)
    location["sensory"]["location_sensor"] = True

    return {"one_predator": one_pred, "full_training": base, "location_on": location}


def test_v9_schema_is_invariant_across_environments(tmp_path):
    import pyarrow.parquet as pq
    stores = {}
    for name, cfg in _variant_configs().items():
        stores[name] = ts.open_store(collect(tmp_path / name, cfg, episodes=4, shard=4,
                                             batch=4))

    dimsets = {n: tuple(s.dims.values()) for n, s in stores.items()}
    assert len(set(dimsets.values())) == 3, f"variants are not distinct: {dimsets}"

    schemas = {}
    for name, s in stores.items():
        f = sorted(Path(s.path).glob("steps_*.parquet"))[0]
        schemas[name] = pq.read_schema(f)

    ref = schemas["full_training"]
    for name, sch in schemas.items():
        assert sch.names == ref.names, f"{name}: column names/order differ"
        assert [str(f.type) for f in sch] == [str(f.type) for f in ref], (
            f"{name}: column TYPES differ — with variable-size list<T> columns the type "
            "must be byte-identical for every environment")

    # ONE reader function loads all three without branching.
    for name, s in stores.items():
        tbl = s.steps(columns=["episode_seed", "t", "animal_row", "obs_noised"])
        assert s.to_2d(tbl, "animal_row").shape[1] == s.dims["A"]
        assert s.to_2d(tbl, "obs_noised").shape[1] == s.dims["D"]


def test_v9_bare_pyarrow_read_matches_the_schema_doc(tmp_path):
    """The NON-CIRCULAR counterpart to C3.  Reads one shard with bare `pyarrow.parquet`,
    importing NOTHING from `trajectory_store`, and compares column names, order and types
    against the SCHEMA DOC's generated table.  That closes the loop between code, store
    and documentation without any of the three vouching for itself."""
    store_dir = collect(tmp_path / "bare", _base_cfg(), episodes=4, shard=4, batch=4)

    # -- everything below uses only pyarrow, json and the committed markdown ----
    import pyarrow.parquet as pq
    manifest = json.loads((store_dir / "_manifest.json").read_text())
    dims = manifest["dims"]
    precision = manifest["obs_precision"]
    schema = pq.read_schema(sorted(store_dir.glob("steps_*.parquet"))[0])

    doc = SCHEMA_DOC.read_text()
    block = re.search(r"BEGIN GENERATED: step_columns.*?-->(.*?)<!-- END GENERATED",
                      doc, re.DOTALL).group(1)
    rows = [ln for ln in block.strip().splitlines() if ln.startswith("|")][2:]
    doc_names, doc_types = [], []
    for ln in rows:
        # markdown escapes a literal pipe inside a cell as `\|` (the obs columns render
        # `list<float16 \| float32>[D]`), so split on UNESCAPED pipes only
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
        doc_names.append(cells[1].strip("`"))
        doc_types.append(cells[2].strip("`").replace("\\|", "|"))

    assert schema.names == doc_names, (
        "the parquet file's columns disagree with the schema doc's generated table")

    arrow_of = {"int64": "int64", "int32": "int32", "int16": "int16", "int8": "int8",
                "bool": "bool", "float32": "float", "float16": "halffloat"}
    for name, dtype, field in zip(doc_names, doc_types, schema):
        m = re.fullmatch(r"list<(.+)>\[(.+)\]", dtype)
        if m:
            elem, width_expr = m.group(1), m.group(2)
            if elem == "float16 | float32":
                elem = precision
            want_width = 1
            for part in width_expr.split("*"):
                want_width *= dims[part.strip()]
            # "element" is Parquet's own name for a list's child field; the doc renders
            # the logical type. Hardcoded here on purpose — this test imports nothing
            # from trajectory_store.
            assert str(field.type) == f"list<element: {arrow_of[elem]}>", (
                f"{name}: parquet type {field.type} != doc type {dtype}")
            # width is a writer invariant with list<T>; check it against the manifest
            tbl = pq.read_table(sorted(store_dir.glob("steps_*.parquet"))[0], columns=[name])
            flat = tbl.column(name).combine_chunks().flatten()
            assert len(flat) == tbl.num_rows * want_width, (
                f"{name}: rows do not all carry the manifest width {want_width}")
        else:
            assert str(field.type) == arrow_of[dtype], (
                f"{name}: parquet type {field.type} != doc type {dtype}")


# ══════════════════════════════════════════════════════════════════════════════
# C10 — zero-slot environment
# ══════════════════════════════════════════════════════════════════════════════

def test_c10_zero_slot_environment(tmp_path):
    """A world with NO animals (A = 0).  18 of the 334 saved configs in the results tree
    are like this.  The animal columns must be present as empty lists — the reader never
    branches — and the store must validate."""
    cfg = _base_cfg()
    cfg["environment"]["entities"] = []
    store_dir = collect(tmp_path / "zero", cfg, episodes=4, shard=4, batch=4)
    st = ts.open_store(store_dir)
    assert st.dims["A"] == 0

    steps = st.steps(columns=["episode_seed", "animal_row", "obs_noised"])
    assert "animal_row" in steps.schema.names, "the column must be PRESENT, not absent"
    assert st.to_2d(steps, "animal_row").shape == (steps.num_rows, 0)
    assert st.to_2d(steps, "obs_noised").shape[1] == st.dims["D"]

    ep = st.episodes()
    assert st.to_2d(ep, "animal_active").shape == (ep.num_rows, 0)
    ts.validate_store_structure(store_dir)
    ts.validate_store_shapes(store_dir)
    ts.validate_store_draws(store_dir)


# ══════════════════════════════════════════════════════════════════════════════
# C9 — resume is a no-op
# ══════════════════════════════════════════════════════════════════════════════

def test_c9_resume_is_a_noop(tmp_path):
    root = tmp_path / "root"
    cfg = _base_cfg()
    store_dir = collect(root, cfg, episodes=16, shard=8, batch=8)
    before = _snapshot(store_dir)
    assert len(ts.completed_blocks(store_dir)) == 2

    again = collect(root, cfg, episodes=16, shard=8, batch=8)
    assert again == store_dir
    assert _snapshot(store_dir) == before, "a resume of a complete store must write nothing"


def test_resume_after_a_lost_block_is_bit_identical(tmp_path):
    """Blocks are pure functions of (seed_base, shard_episodes), so a redone block must be
    byte-identical to what the dead process would have produced — that is what makes
    resume incapable of producing a mixed population."""
    root = tmp_path / "root"
    cfg = _base_cfg()
    store_dir = collect(root, cfg, episodes=16, shard=8, batch=8)
    ep1, st1 = ts.shard_names(1)
    want = ((store_dir / ep1).read_bytes(), (store_dir / st1).read_bytes())
    (store_dir / ep1).unlink()
    (store_dir / st1).unlink()
    collect(root, cfg, episodes=16, shard=8, batch=8)
    assert ((store_dir / ep1).read_bytes(), (store_dir / st1).read_bytes()) == want


# ══════════════════════════════════════════════════════════════════════════════
# Whole-store validation (plan §D16) — it must FIRE
# ══════════════════════════════════════════════════════════════════════════════

def test_validate_store_draws_passes_on_a_good_store(store):
    ts.validate_store_draws(store)
    ts.validate_store_shapes(store)
    ts.validate_store_structure(store)


def test_validate_store_draws_fires_on_a_degenerate_column(tmp_path):
    """A column silently wired to a constant is exactly the failure mode a 200-episode
    sample can miss.  Freeze one bounded draw column and assert the whole-store check
    catches it."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    store_dir = collect(tmp_path / "root", _base_cfg(), episodes=8)
    ts.validate_store_draws(store_dir)                      # precondition: passes clean

    name = ts.shard_names(0)[0]
    tbl = pq.read_table(store_dir / name)
    col = "animal_detect_sampled"
    dims = ts.normalise_dims(ts.read_manifest(store_dir)["dims"])
    w = ts.column_widths(ts.EPISODE_COLUMNS, dims)[col]
    frozen = np.full((tbl.num_rows, w), 3, dtype=np.int32)
    i = tbl.schema.get_field_index(col)
    new = tbl.set_column(i, tbl.schema.field(i),
                         ts._list_array(frozen, tbl.num_rows, w, pa.int32()))
    pq.write_table(new, store_dir / name, compression="zstd")

    with pytest.raises(ValueError, match="single distinct value|outside its sampling bounds"):
        ts.validate_store_draws(store_dir)


def test_validate_store_structure_fires_on_a_row_count_mismatch(tmp_path):
    import pyarrow.parquet as pq
    store_dir = collect(tmp_path / "root", _base_cfg(), episodes=8)
    ts.validate_store_structure(store_dir)                  # precondition: passes clean
    name = ts.shard_names(0)[1]
    tbl = pq.read_table(store_dir / name)
    pq.write_table(tbl.slice(0, tbl.num_rows - 1), store_dir / name, compression="zstd")
    with pytest.raises(ValueError, match="row count != length"):
        ts.validate_store_structure(store_dir)


# ══════════════════════════════════════════════════════════════════════════════
# C3 (circular, and labelled as such) + the doc↔code check
# ══════════════════════════════════════════════════════════════════════════════

def test_c3_schema_round_trip_through_the_reader(store):
    """NOTE THIS CHECK IS CIRCULAR: reader and writer share `build_step_schema`, so it
    catches wiring mistakes, not schema-definition mistakes.  The non-circular counterpart
    is `test_v9_bare_pyarrow_read_matches_the_schema_doc`.  Do not treat this as
    sufficient."""
    import pyarrow.parquet as pq
    st = ts.open_store(store)
    want = ts.build_step_schema(st.dims, st.obs_precision)
    have = pq.read_schema(sorted(Path(st.path).glob("steps_*.parquet"))[0])
    assert have.names == want.names
    assert [str(f.type) for f in have] == [str(f.type) for f in want]

    want_ep = ts.build_episode_schema(st.dims)
    have_ep = pq.read_schema(sorted(Path(st.path).glob("episodes_*.parquet"))[0])
    assert have_ep.names == want_ep.names
    assert [str(f.type) for f in have_ep] == [str(f.type) for f in want_ep]


def test_schema_doc_matches_code():
    """F4: the schema doc's tables are generated from STEP_COLUMNS / EPISODE_COLUMNS.  A
    code change that skips the regeneration must fail the suite rather than silently
    rotting the doc that everyone trusts."""
    r = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "eval" / "traj_collect" / "gen_schema_doc.py"),
         "--check"], capture_output=True, text=True, cwd=_ROOT)
    assert r.returncode == 0, (
        f"{SCHEMA_DOC} is stale — regenerate with "
        f"`python scripts/eval/traj_collect/gen_schema_doc.py`\n{r.stdout}{r.stderr}")


def test_schema_doc_records_the_current_version():
    doc = SCHEMA_DOC.read_text()
    assert f"`SCHEMA_VERSION = {ts.SCHEMA_VERSION}`" in doc


# ══════════════════════════════════════════════════════════════════════════════
# Store-level invariants
# ══════════════════════════════════════════════════════════════════════════════

def test_seeds_advance_per_episode_and_are_unique(store):
    """The existing eval harness pins ONE fixed seed per checkpoint so checkpoints compare
    on identical episodes.  This collector must advance seeds (seed_base + i) instead, and
    the duplicate check must be able to catch a regression to the fixed-seed behaviour."""
    st = ts.open_store(store)
    ep = st.episodes(columns=["episode_seed", "episode_index"])
    seeds = np.asarray(ep.column("episode_seed"))
    idx = np.asarray(ep.column("episode_index"))
    assert np.array_equal(np.sort(seeds), np.sort(idx + st.manifest["seed_base"]))
    assert len(set(seeds.tolist())) == len(seeds)
    assert len(set(seeds.tolist())) > 1, "a single repeated seed would make the store useless"


def test_env_fingerprint_is_canonical_and_discriminating():
    a = _base_cfg()
    b = copy.deepcopy(a)
    b["environment"] = dict(reversed(list(b["environment"].items())))   # reorder only
    assert ts.env_fingerprint(a) == ts.env_fingerprint(b), "key order must not matter"
    c = copy.deepcopy(a)
    c["environment"]["width"] = int(c["environment"]["width"]) + 1
    assert ts.env_fingerprint(a) != ts.env_fingerprint(c)


def test_c5_tolerates_resource_property_redraw(tmp_path):
    """REGRESSION: the C5 constancy guard must NOT assert on the two resource `*_init`
    draws.

    `jax_step` RE-DRAWS `res_property_sampled` / `res_visual_property_sampled` whenever a
    resource regenerates (`core.py:526-541`, assigned at `808-809`) — which is exactly why
    the schema names them `*_init`.  An earlier version compared every `_draw_block` field
    reset-vs-final, so any config with regenerating resources AND a nonzero resource
    `properties_std` would abort mid-collection, and would abort at the same block forever
    on resume.

    No config in the tree has a nonzero resource std today, so the shipped fixture is
    structurally unable to catch this. This test sets one, which is precisely the kind of
    per-episode-variance environment the store exists to serve.
    """
    import collect_trajectories as ct
    import traj_scan
    import jax
    import jax.numpy as jnp
    from src.environment.config_loader import load_env_params
    from src.environment.core import jax_reset
    from src.environment.sensor import get_observation

    # Deterministic setup, so the re-draw is guaranteed rather than hoped for: one food
    # slot pinned to a single cell, the agent started on that cell, a policy that does
    # nothing but eat, and a 1-step regeneration delay.
    cfg = _base_cfg()
    cfg["environment"]["max_steps"] = 60
    cfg["environment"]["random_start_pos"] = False
    cfg["environment"]["start_pos"] = [6, 6]                  # 1-indexed -> (5, 5)
    res = cfg["environment"]["resources"]
    assert res, "precondition: the reference config must declare resources"
    cfg["environment"]["resources"] = [res[0]]
    entry = cfg["environment"]["resources"][0]
    n = len(entry.get("properties", []) or [])
    assert n, "resource entry has no properties to jitter"
    entry["properties_std"] = [0.3] * n
    entry["regeneration_delay"] = 1
    entry["count_low"] = entry["count_high"] = 1
    entry["spawn_area"] = [[6, 6], [6, 6]]                    # -> randint in [5, 6) = 5

    params = load_env_params(Config(copy.deepcopy(cfg)))
    assert np.asarray(params.res_property_std).max() > 0
    assert bool(params.eat_action_enabled) and bool(params.rest_action_enabled)
    model, init_fn, _ = _random_policy(params)
    eat_idx = 5                                               # core.py:580, rest enabled

    def policy_step(m, obs, h):
        return jnp.full((obs.shape[0],), eat_idx, dtype=jnp.int32), h

    rollout_fn = ct.make_rollout_fn()

    seeds = np.arange(4_000_000, 4_000_008, dtype=np.int64)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    states0 = jax.vmap(jax_reset, in_axes=(None, 0))(params, keys)
    o0 = jax.vmap(get_observation, in_axes=(0, None))(states0, params)
    ot0 = jax.vmap(lambda s, p: get_observation(s, p, apply_noise=False),
                   in_axes=(0, None))(states0, params)
    final_draws, _ = rollout_fn(model, params, states0, init_fn(len(seeds)), o0, ot0,
                                int(params.max_steps), policy_step)
    reset_draws = traj_scan._draw_block(states0)

    # The precondition that makes this test meaningful: the re-draw ACTUALLY fired, so a
    # reset-vs-final comparison over all fields WOULD have raised.
    changed = [k for k in traj_scan.MUTABLE_DRAW_FIELDS
               if not np.array_equal(np.asarray(reset_draws[k]), np.asarray(final_draws[k]))]
    assert changed, (
        "no resource property was re-drawn, so this test would pass even with the buggy "
        "all-fields comparison — check the eat action index and the pinned spawn cell")
    assert "res_property_sampled_init" in changed

    # ... and the collector accepts it.
    ct.run_chunk(rollout_fn, model, policy_step, init_fn, params, seeds,
                 int(params.max_steps))


def test_c5_still_fires_on_a_genuinely_constant_field():
    """The companion: narrowing C5 must not have disarmed it. Every field the environment
    treats as an episode constant is still compared, and the classification is total."""
    import traj_scan
    from src.environment.config_loader import load_env_params
    from src.environment.core import jax_reset
    import jax

    params = load_env_params(Config(_base_cfg()))
    fields = set(traj_scan._draw_block(jax_reset(params, jax.random.PRNGKey(0))))
    assert fields == traj_scan.CONSTANT_DRAW_FIELDS | traj_scan.MUTABLE_DRAW_FIELDS, (
        "every per-episode draw must be classified constant or mutable; an unclassified "
        "field would silently escape the C5 check")
    assert traj_scan.MUTABLE_DRAW_FIELDS == {
        "res_property_sampled_init", "res_visual_property_sampled_init"}, (
        "only the two resource property draws are re-drawn mid-episode; widening this set "
        "silently weakens C5")
    assert not (traj_scan.CONSTANT_DRAW_FIELDS & traj_scan.MUTABLE_DRAW_FIELDS)


def test_rollout_fn_is_built_once_and_reused(tmp_path):
    """PERFORMANCE REGRESSION: `nnx.jit` caches its compiled executable on the WRAPPER
    OBJECT, so building a fresh wrapper inside `run_chunk` retraces every chunk — ~977
    full retrace+compiles at 10^6 episodes / batch 1024, which would plausibly dominate
    the collection.

    Asserted structurally (`run_chunk` takes the callable, and does not construct one) and
    behaviourally (a shared wrapper traces once across chunks of the same shape, a fresh
    one traces every time).
    """
    import inspect
    import collect_trajectories as ct

    src = inspect.getsource(ct.run_chunk)
    assert "nnx.jit" not in src, (
        "run_chunk must not construct a jitted wrapper — it retraces every chunk")
    assert "rollout_fn" in inspect.signature(ct.run_chunk).parameters

    # Behavioural half: a SHARED wrapper traces once across identical-shape calls; a
    # FRESH wrapper per call traces every time. This is the whole mechanism.
    import flax.nnx as nnx
    import jax.numpy as jnp

    traces = {"n": 0}

    def kernel(x):
        traces["n"] += 1          # runs at TRACE time, not at call time
        return x * 2

    x = jnp.arange(4.0)
    shared = nnx.jit(kernel)
    shared(x); shared(x); shared(x)
    assert traces["n"] == 1, f"shared wrapper traced {traces['n']} times, expected 1"

    traces["n"] = 0
    for _ in range(3):
        nnx.jit(kernel)(x)        # what the buggy version did, once per chunk
    assert traces["n"] == 3, (
        f"a fresh wrapper per call traced {traces['n']} times — if this is 1, the "
        "premise of the fix no longer holds and the hoisting can be reconsidered")

    assert ct.make_rollout_fn() is not ct.make_rollout_fn(), (
        "make_rollout_fn returns a fresh wrapper each call, which is exactly why main() "
        "must call it once and pass the result down")


def test_c1_rollout_scan_refuses_an_eager_call():
    """C1: `_rollout_scan` must REFUSE to run outside a trace.

    Why this matters more than it looks: after `nnx.update(model, restored_tree)`,
    calling the model eagerly reads a STALE view of the restored parameters and every
    trajectory silently diverges from step 0 — with no error, no warning and no
    downstream signal.  `lax.scan` alone does not fix it; only entering through
    `nnx.jit` performs nnx's graphdef/state split.  A refusal is the only way to make
    that failure loud."""
    import jax
    import jax.numpy as jnp
    import traj_scan
    from src.environment.config_loader import load_env_params
    from src.environment.core import jax_reset
    from src.environment.sensor import get_observation

    params = load_env_params(Config(_base_cfg()))
    model, init_fn, policy_step = _random_policy(params)
    seeds = jnp.asarray(np.arange(4, dtype=np.int64))
    keys = jax.vmap(jax.random.PRNGKey)(seeds)
    s0 = jax.vmap(jax_reset, in_axes=(None, 0))(params, keys)
    o0 = jax.vmap(get_observation, in_axes=(0, None))(s0, params)
    ot0 = jax.vmap(lambda s, p: get_observation(s, p, apply_noise=False),
                   in_axes=(0, None))(s0, params)

    with pytest.raises(RuntimeError, match="EAGERLY"):
        traj_scan._rollout_scan(model, params, s0, init_fn(4), o0, ot0, 3, policy_step)

    # ... and the supported entry point works, so the guard is not simply always-on.
    import flax.nnx as nnx
    draws, out = nnx.jit(traj_scan._rollout_scan,
                         static_argnames=("max_steps", "policy_step"))(
        model, params, s0, init_fn(4), o0, ot0, 3, policy_step)
    assert np.asarray(out["action"]).shape == (3, 4)


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    import pyarrow as pa
    p = tmp_path / "x.parquet"
    ts.write_shard_atomic(p, pa.table({"a": pa.array([1, 2, 3])}))
    assert p.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_device_is_manifest_guarded(tmp_path):
    """Bit-level results are COMPILATION-dependent — the environment's own reset differs
    by one float32 ULP on `animal_property_sampled` between compilations (compiler fusion;
    see `test_v1_animal_property_divergence_is_one_float32_ulp`) — so a store must not be
    startable on CPU and resumed on GPU."""
    store_dir = collect(tmp_path / "root", _base_cfg(), episodes=8)
    m = ts.read_manifest(store_dir)
    assert m["device"] == "cpu"
    with pytest.raises(ValueError, match="Manifest mismatch"):
        ts.assert_manifest_compatible(store_dir, dict(m, device="gpu"))


def test_seed_space_cliff_is_refused():
    """Seeds are canonicalised to int32 before PRNGKey construction (x64 is off).

    Measured: seeds in [2**31, 2**32) wrap to a negative int32 but round-trip to the SAME
    uint32, so they are harmless; at 2**32 the uint32 itself wraps and seed 2**32 + 5
    produces byte-identical episodes to seed 5 while `episode_seed` records the un-wrapped
    value. Neither the PRNG parity guard (both sides consume the same wrapped seed) nor
    the duplicate-seed check (the recorded values differ) can see that. Refuse well before
    the cliff, where no canonicalisation happens at all.
    """
    import jax
    import jax.numpy as jnp
    import collect_trajectories as ct
    from src.environment.config_loader import load_env_params

    # The mechanism, demonstrated rather than asserted from memory.
    collide = jax.vmap(jax.random.PRNGKey)(jnp.asarray(np.asarray([5, 2 ** 32 + 5],
                                                                  dtype=np.int64)))
    assert np.array_equal(np.asarray(collide[0]), np.asarray(collide[1])), (
        "the int32 seed cliff no longer collides — re-derive the bound before relaxing it")

    cfg = _base_cfg()
    params = load_env_params(Config(copy.deepcopy(cfg)))
    kw = dict(cfg=cfg, params=params, run_dir=REAL_RUN, ckpt_dir=REAL_RUN / "models" / "0",
              ckpt_step=0, dims=ct.env_dims(params), shard_episodes=5000, batch_size=1024,
              device="cpu", obs_precision="float32", scene_format="entities",
              scene_ambiguous=False)
    with pytest.raises(ValueError, match=r"2\*\*31"):
        ct.build_manifest(seed_base=2 ** 31 - 10, n_episodes=1000, **kw)
    with pytest.raises(ValueError, match=r"2\*\*31"):
        ct.build_manifest(seed_base=-1, n_episodes=10, **kw)
    ct.build_manifest(seed_base=1_000_000, n_episodes=1_000_000, **kw)   # comfortably fine


def test_max_steps_beyond_int16_is_refused():
    """`t`, `rest_streak`, `res_cons_count` and the timers are int16, and numpy's
    `astype` WRAPS silently rather than raising — negative step indices recorded as
    plausible data."""
    import collect_trajectories as ct
    from src.environment.config_loader import load_env_params
    cfg = _base_cfg()
    cfg["environment"]["max_steps"] = 40000
    params = load_env_params(Config(copy.deepcopy(cfg)))
    with pytest.raises(ValueError, match="int16"):
        ct.build_manifest(
            cfg=cfg, params=params, run_dir=REAL_RUN, ckpt_dir=REAL_RUN / "models" / "0",
            ckpt_step=0, dims=ct.env_dims(params), seed_base=0, n_episodes=10,
            shard_episodes=5000, batch_size=1024, device="cpu", obs_precision="float32",
            scene_format="entities", scene_ambiguous=False)


def test_manifest_carries_training_provenance(tmp_path):
    """The manifest must record WHICH CODE trained the checkpoint it rolled out.

    `train.py` writes `<run>/models/provenance.json` at startup (since 2026-08-20); the
    collector copies it into the manifest. Both states are asserted, because they are
    different claims and a reader must be able to tell them apart:

      * REAL_RUN was trained BEFORE the stamp existed → `null` (absent, not an error);
      * a run WITH the stamp → the real sha / dirty flag / start time.

    The helper-level three-state coverage (including `"unknown"`) lives in
    tests/test_provenance.py.
    """
    import shutil
    import collect_trajectories as ct
    from src.environment.config_loader import load_env_params
    from src.utils.provenance import write_provenance

    cfg = _base_cfg()
    params = load_env_params(Config(copy.deepcopy(cfg)))
    kw = dict(cfg=cfg, params=params, ckpt_dir=REAL_RUN / "models" / "0", ckpt_step=0,
              dims=ct.env_dims(params), seed_base=0, n_episodes=10, shard_episodes=5000,
              batch_size=1024, device="cpu", obs_precision="float32",
              scene_format="entities", scene_ambiguous=False)

    # (a) pre-stamp run — null sentinel, and NOT an error.
    assert not (REAL_RUN / "models" / "provenance.json").exists(), (
        "REAL_RUN acquired a provenance stamp — pick another pre-stamp run for this half")
    m_absent = ct.build_manifest(run_dir=REAL_RUN, **kw)
    assert m_absent["training_git_sha"] is None
    assert m_absent["training_git_dirty"] is None
    assert m_absent["training_started_utc"] is None

    # (b) stamped run — the real values.
    fake_run = tmp_path / "stamped_run"
    (fake_run / "models").mkdir(parents=True)
    shutil.copy(REAL_RUN / "models" / "config.yaml", fake_run / "models" / "config.yaml")
    write_provenance(fake_run / "models", argv=["train.py", "--config", "x.yaml"])
    truth = json.loads((fake_run / "models" / "provenance.json").read_text())

    m_present = ct.build_manifest(run_dir=fake_run, **kw)
    assert m_present["training_git_sha"] == truth["git_sha"] != "unknown"
    assert m_present["training_git_dirty"] == truth["git_dirty"]
    assert m_present["training_started_utc"] == truth["started_utc"]

    # Additive only: the collection-side provenance is untouched by this.
    assert m_present["collection_git_sha"] == m_absent["collection_git_sha"] != "unknown"


def test_list_column_width_enforcement_fires():
    """The fixed width of an array column is a WRITER invariant (the columns are
    variable-size `list<T>`; see the schema doc §3.3), so it needs a test that it is
    actually enforced rather than merely intended."""
    dims = {"A": 4, "R": 2, "B": 3, "V": 5, "VV": 8, "D": 27}
    n = 6
    data = {}
    for c in ts.EPISODE_COLUMNS:
        w = ts.column_widths(ts.EPISODE_COLUMNS, dims).get(c.name)
        data[c.name] = np.zeros(n if w is None else (n, w))
    ts.build_table(ts.EPISODE_COLUMNS, data, dims, obs_precision=None)      # baseline: OK

    data["animal_active"] = np.zeros((n, dims["A"] - 1))                    # wrong width
    with pytest.raises(ValueError, match="expected"):
        ts.build_table(ts.EPISODE_COLUMNS, data, dims, obs_precision=None)


def test_validate_store_shapes_fires_on_a_wrong_width(tmp_path):
    """And the reader-side re-check catches a shard whose rows do not all carry the
    manifest width."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    store_dir = collect(tmp_path / "root", _base_cfg(), episodes=8)
    ts.validate_store_shapes(store_dir)                       # precondition: passes clean

    name = ts.shard_names(0)[0]
    tbl = pq.read_table(store_dir / name)
    col = "animal_active"
    i = tbl.schema.get_field_index(col)
    ragged = pa.ListArray.from_arrays(
        pa.array(np.arange(tbl.num_rows + 1, dtype=np.int32) * 2),          # width 2, not A
        pa.array(np.zeros(tbl.num_rows * 2, dtype=bool)))
    pq.write_table(tbl.set_column(i, tbl.schema.field(i), ragged),
                   store_dir / name, compression="zstd")
    with pytest.raises(ValueError, match="expected width"):
        ts.validate_store_shapes(store_dir)


def test_build_table_refuses_a_missing_or_unknown_column():
    dims = {"A": 2, "R": 2, "B": 2, "V": 3, "VV": 4, "D": 5}
    with pytest.raises(ValueError, match="missing required column"):
        ts.build_table(ts.STEP_COLUMNS, {"episode_seed": np.zeros(1)}, dims, "float16")
    data = {c.name: np.zeros((1,) if c.width is None else (1, 1)) for c in ts.STEP_COLUMNS}
    data["bogus"] = np.zeros(1)
    with pytest.raises(ValueError, match="not in the fixed key list"):
        ts.build_table(ts.STEP_COLUMNS, data, dims, "float16")


def test_obs_precision_has_no_default():
    """Project Configuration Protocol: a LOSSY choice must never be silently defaulted."""
    dims = {"A": 1, "R": 1, "B": 1, "V": 1, "VV": 1, "D": 1}
    with pytest.raises(ValueError, match="obs_precision"):
        ts.build_step_schema(dims, None)
    with pytest.raises(ValueError, match="obs_precision"):
        ts.build_step_schema(dims, "bfloat16")


def _example_spec() -> dict:
    return yaml.safe_load(
        (_ROOT / "configs" / "trajectory_collection" / "example.yaml").read_text())


def test_spec_loader_requires_every_scientific_key(tmp_path):
    import run_collection as rc
    full = _example_spec()
    for key in rc.MANDATORY:
        spec = copy.deepcopy(full)
        spec.pop(key)
        p = tmp_path / f"{key}.yaml"
        p.write_text(yaml.safe_dump(spec))
        with pytest.raises(ValueError, match=re.escape(repr(key))):
            rc.load_spec(p)


def test_spec_loader_accepts_the_shipped_example(tmp_path):
    """The companion — a validator that rejects its own template is not a validator.
    Also pins that every run path in the shipped example actually exists."""
    import run_collection as rc
    if not REAL_RUN.exists():
        pytest.skip("results/ not present (gitignored NAS data)")
    p = _ROOT / "configs" / "trajectory_collection" / "example.yaml"
    spec = rc.load_spec(p)
    assert spec["obs_precision"] == "float32", (
        "the shipped template must recommend the measured decision (float32); see the "
        "schema doc §5")


def test_spec_loader_rejects_unknown_keys(tmp_path):
    """A silently-ignored key is how a typo becomes a scientific claim: `seed_bases:` on a
    run entry would inherit the batch-level base and manufacture exactly the ILLUSORY
    PAIRING the design warns about."""
    import run_collection as rc
    if not REAL_RUN.exists():
        pytest.skip("results/ not present (gitignored NAS data)")

    spec = _example_spec()
    spec["obs_precison"] = "float16"                      # top-level typo
    p = tmp_path / "top.yaml"
    p.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="unknown top-level spec key"):
        rc.load_spec(p)

    spec = _example_spec()
    spec["runs"][0]["seed_bases"] = 999                   # the dangerous per-run typo
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="seed_bases"):
        rc.load_spec(p)

    for key in rc.RUN_KEYS:                               # the allowed ones still pass
        spec = _example_spec()
        spec["runs"][0].setdefault(key, spec["runs"][0].get("path") if key == "path" else 1)
        p = tmp_path / f"ok_{key}.yaml"
        p.write_text(yaml.safe_dump(spec))
        rc.load_spec(p)


def test_spec_loader_preflights_run_paths(tmp_path):
    """A typo'd path must fail HERE, once, not per-cell after every node is launched."""
    import run_collection as rc
    spec = _example_spec()
    spec["runs"] = [{"label": "x", "path": "results/JAX_RecurrentPPO/does_not_exist"}]
    p = tmp_path / "badpath.yaml"
    p.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="models/config.yaml"):
        rc.load_spec(p)

    spec = _example_spec()
    spec["runs"] = [{"label": "x"}]                       # missing `path` entirely
    p = tmp_path / "nopath.yaml"
    p.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="missing the required `path`"):
        rc.load_spec(p)


@pytest.mark.parametrize("key,bad", [("episodes", 0), ("npar", 0), ("batch_size", -1),
                                     ("seed_base", -5), ("shard_episodes", 0)])
def test_spec_loader_rejects_nonpositive_numbers(tmp_path, key, bad):
    import run_collection as rc
    if not REAL_RUN.exists():
        pytest.skip("results/ not present (gitignored NAS data)")
    spec = _example_spec()
    spec[key] = bad
    p = tmp_path / f"{key}.yaml"
    p.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match=key):
        rc.load_spec(p)


def test_dry_run_does_not_delete_completion_markers(tmp_path):
    """A dry run against a spec name with a LIVE collection must not delete the markers
    the live driver is polling for — that would hang its wait loop forever."""
    import run_collection as rc
    if not REAL_RUN.exists():
        pytest.skip("results/ not present (gitignored NAS data)")
    spec = _example_spec()
    spec["out_root"] = str(tmp_path / "traj")
    spec["nodes"] = [101]
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec))

    markers = tmp_path / "traj" / "_scratch" / spec["name"] / "_run_markers"
    markers.mkdir(parents=True)
    live = markers / "done_101"
    live.write_text("a live collection finished this node")

    rc.main([str(p), "--dry-run"])
    assert live.exists(), "--dry-run must be inert with respect to a live collection"
    assert live.read_text() == "a live collection finished this node"
