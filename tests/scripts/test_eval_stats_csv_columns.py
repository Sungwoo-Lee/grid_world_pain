"""Regression test — eval stats CSV: sensor headers must align with sensor values.

Bug under test (H8, docs/develop/active/issues/diag_fable5_20260704/
06_evaluation_path.md Finding 1 + fix_plan_h8h9_eval_output_correctness.md):

  1. The header builder in src/utils/evaluation_core.py omitted the
     "Interoceptive Nociception" sensor (enabled by default), so every sensor
     value from that slot onward sat under the PREVIOUS sensor's column name.
  2. `_write_episode_stats` counted obs columns via `startswith("obs_")`,
     which also matches the world-entity headers `obs_entity_{i}_r/_c` — on
     any env with obstacle entities (including the default env) the data row
     came out one field LONGER than the header, so pandas misassigned every
     column in the file (implicit-index shift).

Fix under test: `build_stat_headers()` + `_sensor_stat_columns()` derive the
header names from the SAME `get_observation_breakdown()` ordering that the
values are written in, raising on unmapped sensors; `_write_episode_stats`
excludes `obs_entity_` from the obs count and fails loudly on any
header-count/vector-dim mismatch.

Strategy: plant per-slot sentinel values (obs = arange, true = arange + 1000)
into one synthetic step on the DEFAULT env (interoceptive nociception on,
obstacle entities present — both defects exercised), write the CSV through the
real `_write_episode_stats`, and read every modality back BY COLUMN NAME.
"""
import csv
import os
import sys

import jax
import numpy as np
import pandas as pd
import pytest

jax.config.update("jax_platform_name", "cpu")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from src.utils.config import Config, get_default_config  # noqa: E402
from src.environment.config_loader import load_env_params  # noqa: E402
from src.environment.core import jax_reset  # noqa: E402
from src.environment.sensor import get_observation_breakdown  # noqa: E402
from src.utils import evaluation_core as ec  # noqa: E402


@pytest.fixture(scope="module")
def env():
    """Default env (interoceptive_nociception_enabled: true, obstacle entities
    present) merged with train/eval defaults, as the eval path assembles it."""
    cfg = get_default_config()
    for rel in ("configs/train/default.yaml", "configs/evaluation/default.yaml"):
        cfg.merge(Config.load_yaml(os.path.join(_REPO, rel)))
    params = load_env_params(cfg)
    breakdown = get_observation_breakdown(params)
    # Fixture guards (not the fix): both defect preconditions must hold.
    assert "Interoceptive Nociception" in breakdown
    assert params.obs_blocking.shape[0] > 0, "default env should have obstacle entities"
    return params, breakdown


@pytest.fixture(scope="module")
def written_csv(env, tmp_path_factory):
    """Write one synthetic sentinel step through the real production path
    (build_stat_headers + _write_episode_stats); return paths + layout info."""
    params, breakdown = env
    obs_dim = sum(breakdown.values())
    obs = np.arange(obs_dim, dtype=np.float32)
    true = obs + 1000.0

    stat_headers = ec.build_stat_headers(params, breakdown, record_true_obs=True)

    state = jax_reset(params, jax.random.PRNGKey(0))
    ep_state = {
        'agent_pos': state.agent_pos,
        'satiation': state.satiation,
        'nutrition': state.nutrition,
        'injury_level': state.injury_level,
        'rest_streak': state.rest_streak,
        'res_pos': state.res_pos,
        'res_active': state.res_active,
        'animal_pos': state.animal_pos,
        'obs_pos': state.obs_pos,
    }
    action_map = ["Up", "Right", "Down", "Left"]
    if params.rest_action_enabled:
        action_map.append("Rest")
    if params.eat_action_enabled:
        action_map.append("Eat")

    stats_dir = str(tmp_path_factory.mktemp("stats"))
    ec._write_episode_stats(
        stats_dir, 1,
        ep_jax_states=[ep_state], ep_jax_infos=[{}], ep_actions=[-1],
        ep_rewards=[0.0], ep_obs=[obs], stat_headers=stat_headers,
        action_map=action_map, params=params, ep_true_obs=[true],
    )
    return os.path.join(stats_dir, "000001ep_stats.csv"), breakdown, obs_dim


def _modality_offsets(breakdown):
    """(sensor_name, dim, offset-into-obs-vector) walking breakdown in order."""
    out, off = [], 0
    for name, dim in breakdown.items():
        out.append((name, dim, off))
        off += dim
    return out


def test_row_width_matches_header(written_csv):
    """Pre-fix (mode B, default env): each data row is one field longer than
    the header because obs_entity_* headers inflate the obs-column count."""
    csv_path, _, _ = written_csv
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        data_row = next(reader)
    assert len(data_row) == len(header), (
        f"stats-CSV row width {len(data_row)} != header width {len(header)}"
    )


def test_sensor_columns_read_back_by_name(written_csv, env):
    """Every planted sentinel must come back under its OWN sensor's column
    name — for both the obs_* and true_* blocks."""
    csv_path, breakdown, obs_dim = written_csv
    params, _ = env
    df = pd.read_csv(csv_path)

    for prefix, base in (("obs_", 0.0), ("true_", 1000.0)):
        for sensor_name, dim, offset in _modality_offsets(breakdown):
            names = ec._sensor_stat_columns(sensor_name, dim, params, prefix)
            assert len(names) == dim, (
                f"{prefix}{sensor_name}: {len(names)} column names for dim={dim}"
            )
            first, last = names[0], names[-1]
            assert first in df.columns, f"missing column {first!r}"
            assert last in df.columns, f"missing column {last!r}"
            assert df[first].iloc[0] == base + float(offset), (
                f"{first}: expected sentinel {base + offset}, got {df[first].iloc[0]}"
            )
            assert df[last].iloc[0] == base + float(offset + dim - 1), (
                f"{last}: expected sentinel {base + offset + dim - 1}, "
                f"got {df[last].iloc[0]}"
            )

    # Load-bearing hardcoded names (the exact mislabels of the pre-fix CSV):
    offsets = {name: off for name, _, off in _modality_offsets(breakdown)}
    # pre-fix: column absent entirely -> KeyError
    assert df["obs_intero_nociception"].iloc[0] == float(
        offsets["Interoceptive Nociception"])
    # pre-fix: obs_noc held the interoceptive-nociception slot value
    assert df["obs_noc"].iloc[0] == float(offsets["Extero Nociception"])
    # pre-fix: obs_olf_0 held the extero-nociception slot value
    assert df["obs_olf_0"].iloc[0] == float(offsets["Olfaction"])
    assert df["true_intero_nociception"].iloc[0] == 1000.0 + float(
        offsets["Interoceptive Nociception"])
    assert df["true_noc"].iloc[0] == 1000.0 + float(offsets["Extero Nociception"])
    assert df["true_olf_0"].iloc[0] == 1000.0 + float(offsets["Olfaction"])

    # The final element of the obs vector must be written and correctly named
    # (pre-fix mode A behavior dropped it).
    last_sensor, last_dim, last_off = _modality_offsets(breakdown)[-1]
    last_col = ec._sensor_stat_columns(last_sensor, last_dim, params, "obs_")[-1]
    assert df[last_col].iloc[0] == float(obs_dim - 1)


def test_unknown_sensor_raises(env):
    """A sensor added to get_observation_breakdown() without a column mapping
    must crash header build, never silently shift the CSV again."""
    params, _ = env
    with pytest.raises(ValueError, match="Bogus Sensor"):
        ec._sensor_stat_columns("Bogus Sensor", 1, params, "obs_")
