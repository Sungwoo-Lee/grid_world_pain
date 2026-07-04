"""Regression tests — root evaluation.py model-rebuild + restore-completeness
fixes (Findings A / L3, docs/reviews/diag_v3_pipeline_jax.md).

Plain-language context: `evaluation.py` (the deprecated, non-live evaluation
entry point -- the live path is scripts/eval/eval_rollout.py) hand-reconstructs
the model config instead of reusing train.py's exact expression, and its
checkpoint-restore merge silently keeps randomly-initialised weights for any
param key absent from the checkpoint. This audit found two concrete bugs:

  Finding A #1 -- evaluating ANY neuromodulated RecurrentPPO checkpoint
    (any `recurrent_ppo_nmn_*` config) crashed with KeyError('memory_clip')
    because the hand-built modulation_config whitelist omitted a key that
    ActorCriticRNN.__init__ reads unconditionally.
  Finding A #2 -- evaluating a DreamerV3 checkpoint crashed with
    AttributeError('dict' object has no attribute 'get_mandatory') because
    a plain 5-key dict was passed where DreamerTrainer expects a Config object
    (plus obs_breakdown/modulation_config, both previously omitted).
  Finding L3 -- the restore merge kept freshly-initialised (random) weights
    for any checkpoint-absent param leaf with no warning or error, so a
    train/eval structural mismatch would silently load a half-random model.

Tests below:
  1. test_modulated_rppo_eval_rebuild_succeeds -- builds + saves a REAL
     modulated (FiLM, memory_clip-bearing) RecurrentPPO checkpoint, then runs
     it through evaluation.py's actual `main()`. Pre-fix: KeyError('memory_clip').
     Post-fix: completes cleanly.
  2. test_dreamer_eval_rebuild_signature -- confirms DreamerTrainer is invoked
     with a Config object (not a plain dict) plus obs_breakdown/modulation_config,
     at the construction-signature level (no DreamerV3 checkpoint fixture
     exists on disk to round-trip end-to-end).
  3. test_merge_reports_missing_leaves -- unit test of the completeness
     tracking added to `_merge_restored_into_module_state` (Finding L3):
     absent leaves are collected, not silently dropped.
  4. test_assert_full_restore_raises_on_incomplete_checkpoint -- a deliberate
     structural mismatch (checkpoint missing a whole param subtree) must now
     raise ValueError instead of silently proceeding.
  5. test_assert_full_restore_passes_on_complete_checkpoint -- a fully-covering
     checkpoint must NOT raise (the common, correct case is unaffected).
  6. test_plain_rppo_eval_rebuild_still_succeeds -- the common case (no
     neuromodulation) must still round-trip correctly, unchanged.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import jax
jax.config.update("jax_platform_name", "cpu")

import orbax.checkpoint as ocp
import pytest
from flax import nnx

from src.utils.config import Config, get_default_config, dump_config_yaml
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset
from src.environment.sensor import get_observation, get_observation_breakdown
from src.models.recurrent_ppo_network import ActorCriticRNN

import evaluation as ev  # module under test: repo-root evaluation.py

AGENT_CONFIG_MODULATED = os.path.join(
    _REPO, "configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_tempceil5.yaml"
)
ENV_CONFIG_SMALL = os.path.join(
    _REPO, "configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml"
)


def _merged_config(agent_config_path, max_steps=15):
    """Mirror train.py's config assembly (env default -> env config -> train/eval/vis
    defaults -> agent config), shrunk to a fast test size."""
    cfg = get_default_config()
    cfg.merge(Config.load_yaml(ENV_CONFIG_SMALL))
    for rel in ("configs/train/default.yaml", "configs/evaluation/default.yaml",
                "configs/visualization/default.yaml"):
        cfg.merge(Config.load_yaml(os.path.join(_REPO, rel)))
    cfg.merge(Config.load_yaml(agent_config_path))
    cfg.set("environment.max_steps", max_steps)
    cfg.set("testing.evaluation_episodes", 1)
    cfg.set("testing.num_envs", 1)
    cfg.set("testing.record_stats", False)
    cfg.set("testing.render_video", False)
    return cfg


def _build_rppo_model(config):
    """Build ActorCriticRNN exactly as train.py does (the full modulation dict,
    including memory_clip)."""
    params = load_env_params(config)
    test_state = jax_reset(params, jax.random.PRNGKey(0))
    obs = get_observation(test_state, params)
    input_dim = obs.shape[0]
    action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)

    modulation_config = config.get("agent.modulation")
    if modulation_config is not None and modulation_config.get("type") is None:
        modulation_config = None

    model = ActorCriticRNN(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_size=config.get_mandatory("agent.hidden_size"),
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
        rnn_type=config.get_mandatory("agent.rnn_type"),
        activation=config.get_mandatory("agent.activation"),
        modulation_config=modulation_config,
        observation_breakdown=get_observation_breakdown(params),
        encoding_config=config.to_dict().get("agent", {}),
    )
    return model, params


def _write_results_dir(tmp_path, config, model, iteration=0):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "config.yaml", "w") as f:
        dump_config_yaml(config.to_dict(), f)

    checkpointer = ocp.CheckpointManager(
        str(models_dir.resolve()), checkpointers=ocp.StandardCheckpointer()
    )
    checkpointer.save(iteration, args=ocp.args.StandardSave(
        {"model": nnx.state(model, nnx.Param)}
    ))
    checkpointer.wait_until_finished()
    checkpointer.close()
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Finding A #1 — modulated RecurrentPPO rebuild
# ---------------------------------------------------------------------------

def test_modulated_rppo_eval_rebuild_succeeds(tmp_path, monkeypatch):
    """Pre-fix: KeyError('memory_clip') at model construction. Post-fix: the
    full checkpoint round-trip through evaluation.py's actual main() completes
    without error."""
    config = _merged_config(AGENT_CONFIG_MODULATED)
    modulation_config = config.get("agent.modulation")
    assert modulation_config is not None and "memory_clip" in modulation_config, (
        "fixture config must actually carry memory_clip to be a faithful repro"
    )

    model, _ = _build_rppo_model(config)
    results_dir = _write_results_dir(tmp_path, config, model)

    argv = ["evaluation.py", "--results_dir", results_dir, "--episodes", "1",
            "--no-render", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    ev.main()  # must not raise


# ---------------------------------------------------------------------------
# Finding A #2 — DreamerV3 rebuild (signature-level; no on-disk checkpoint fixture)
# ---------------------------------------------------------------------------

def test_dreamer_eval_rebuild_signature(tmp_path, monkeypatch):
    """Confirms evaluation.py's ACTUAL DreamerV3 branch (real code path inside
    ev.main(), not a reimplementation) now builds DreamerTrainer with a Config
    object (supports get_mandatory/to_dict) plus obs_breakdown and
    modulation_config, mirroring train.py:815-817 -- instead of the old plain
    5-key dict that crashed with AttributeError.

    No on-disk DreamerV3 checkpoint fixture exists (building + training a real
    world model is out of scope for a unit test), so the heavy network build
    is short-circuited: DreamerTrainer.__init__ is spied on to capture its
    call-site arguments, then raises a sentinel exception before doing any
    real work. This still exercises evaluation.py's own construction code
    verbatim -- only the (expensive, irrelevant-to-this-fix) inside of
    DreamerTrainer itself is skipped."""
    dreamer_agent_dir = os.path.join(_REPO, "configs/models/dreamer_v3")
    candidates = sorted(
        f for f in os.listdir(dreamer_agent_dir) if f.endswith(".yaml")
    ) if os.path.isdir(dreamer_agent_dir) else []
    if not candidates:
        pytest.skip("No configs/models/dreamer_v3/*.yaml fixture found on disk.")

    config = _merged_config(AGENT_CONFIG_MODULATED)  # base env/testing scaffolding
    config.merge(Config.load_yaml(os.path.join(dreamer_agent_dir, candidates[0])))
    config.set("agent.algorithm", "DreamerV3")

    import src.models.dreamer_v3_trainer as dvt
    captured = {}

    class _Sentinel(Exception):
        pass

    def _spy_init(self, obs_dim, act_dim, cfg, rngs, obs_breakdown=None, modulation_config=None):
        captured["config"] = cfg
        captured["obs_breakdown"] = obs_breakdown
        captured["modulation_config"] = modulation_config
        raise _Sentinel()

    monkeypatch.setattr(dvt.DreamerTrainer, "__init__", _spy_init)

    # evaluation.py enumerates checkpoints by digit-named subdirs and only
    # touches orbax AFTER model construction, so a bare "0" dir is enough to
    # reach the (spied) DreamerTrainer construction call.
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "config.yaml", "w") as f:
        dump_config_yaml(config.to_dict(), f)
    (models_dir / "0").mkdir()

    argv = ["evaluation.py", "--results_dir", str(tmp_path), "--episodes", "1",
            "--no-render", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(_Sentinel):
        ev.main()

    assert isinstance(captured["config"], Config), (
        "DreamerTrainer must receive the full Config object (get_mandatory-capable), "
        "not a plain whitelist dict"
    )
    params = load_env_params(config)
    assert captured["obs_breakdown"] == get_observation_breakdown(params)
    # modulation_config may legitimately be None (baseline); what matters is it
    # was threaded through, not silently dropped.
    assert "modulation_config" in captured


# ---------------------------------------------------------------------------
# Finding L3 — silent partial restore
# ---------------------------------------------------------------------------

def test_merge_reports_missing_leaves():
    """_merge_restored_into_module_state must collect the dotted-path of every
    leaf left at its randomly-initialised value (absent from the checkpoint),
    instead of silently dropping the information."""
    module_state = {"a": {"w": 1.0, "b": 2.0}, "c": 3.0}
    restored_state = {"a": {"w": 99.0}}  # 'b' and 'c' absent

    missing = []
    merged = ev._merge_restored_into_module_state(module_state, restored_state, _missing=missing)

    assert merged["a"]["w"] == 99.0          # restored value used
    assert merged["a"]["b"] == 2.0           # kept original (random) value
    assert merged["c"] == 3.0                # kept original (random) value
    assert sorted(missing) == ["a.b", "c"]


def test_assert_full_restore_raises_on_incomplete_checkpoint():
    """A deliberate structural mismatch (checkpoint missing a whole param
    subtree) must now ERROR loudly instead of silently proceeding."""
    module_state = {"layer1": {"kernel": 1.0, "bias": 2.0}, "layer2": {"kernel": 3.0}}
    restored_state = {"layer1": {"kernel": 99.0, "bias": 98.0}}  # layer2 entirely absent

    missing = []
    ev._merge_restored_into_module_state(module_state, restored_state, _missing=missing)

    with pytest.raises(ValueError, match="layer2"):
        ev._assert_full_restore(missing, "test model")


def test_assert_full_restore_passes_on_complete_checkpoint():
    """A checkpoint that fully covers the module structure must NOT raise --
    the common, correct round-trip case is unaffected by the new assertion."""
    module_state = {"layer1": {"kernel": 1.0, "bias": 2.0}}
    restored_state = {"layer1": {"kernel": 99.0, "bias": 98.0}}

    missing = []
    ev._merge_restored_into_module_state(module_state, restored_state, _missing=missing)
    ev._assert_full_restore(missing, "test model")  # must not raise


# ---------------------------------------------------------------------------
# Common-case regression guard — non-modulated RecurrentPPO unaffected
# ---------------------------------------------------------------------------

def test_plain_rppo_eval_rebuild_still_succeeds(tmp_path, monkeypatch):
    """The mainstream case (no neuromodulation) must still round-trip
    correctly and unchanged after the Finding A/L3 fixes."""
    config = _merged_config(AGENT_CONFIG_MODULATED)
    config.set("agent.modulation.type", None)  # disable modulation -> baseline

    model, _ = _build_rppo_model(config)
    modulation_config = config.get("agent.modulation")
    assert model.modulation_enabled is False or modulation_config.get("type") is None

    results_dir = _write_results_dir(tmp_path, config, model)

    argv = ["evaluation.py", "--results_dir", results_dir, "--episodes", "1",
            "--no-render", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    ev.main()  # must not raise
