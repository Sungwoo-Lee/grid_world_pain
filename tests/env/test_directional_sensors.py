"""v3.1 directional sensors — parity, kernel geometry, masking, diamond behaviour.

Plan: docs/develop/active/sensors/DIRECTIONAL_SENSORS_PLAN.md

Consolidated into one module rather than the six the plan named; the sections
below map 1:1 onto that plan's test table.

Bit-exactness tests pin the CPU backend. Float results are lowering-dependent —
CPU and GPU legitimately differ, and GPU autotuning makes repeat GPU runs differ
from each other — so a bit-exact assertion is only meaningful on a fixed backend.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import sys
import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp
from src.utils.config import Config
from src.environment.config_loader import load_env_params, _read_visual_mask
from src.environment.sensor import (
    get_observation, get_observation_breakdown, get_visual_offsets,
    sense_visual, sense_olfaction_cells, _psf_weights, _visual_mask_gate,
)
from src.environment.wrapper import ParallelEnv

DEFAULT = os.path.join(_ROOT, 'configs/environment/default.yaml')


def params(**overrides):
    c = Config.load_yaml(DEFAULT)
    for k, v in overrides.items():
        c.set(k, v)
    return load_env_params(c)


def one_state(p, seed=7):
    env = ParallelEnv(p)
    st, obs = env.reset(jax.random.PRNGKey(seed), 1)
    return jax.tree.map(lambda x: x[0], st), np.asarray(obs[0])


# ---------------------------------------------------------------- parity ---

def test_shipped_defaults_are_off():
    p = params()
    assert p.olfactory_grid_range == 0
    assert p.visual_blur_enabled is False
    for m in (p.res_visual_mask, p.animal_visual_mask, p.obs_visual_mask):
        assert int(np.asarray(m).sum()) == 0, "no entity is masked by default"


def test_on_source_half_cell_floor_is_bit_identical_at_shipped_gamma():
    """Option B: 1/(0.5**gamma) replaces the literal 2.0. At gamma=1 they must be
    the same bits, in the traced form the sensor actually evaluates."""
    @jax.jit
    def traced(g):
        return 1.0 / jnp.power(jnp.float32(0.5), g)
    got = np.asarray(traced(jnp.float32(1.0)))
    assert got.tobytes() == np.float32(2.0).tobytes()


def test_observation_is_bit_identical_to_stored_pre_change_fixture():
    """CP1. Compares against an artefact captured BEFORE the change, not a fresh
    re-derivation from source."""
    fix = os.path.join(_ROOT, 'tests/env/fixtures/directional_sensors/obs_baseline.npz')
    if not os.path.exists(fix):
        pytest.skip('baseline fixture not captured')
    import scripts.verification.capture_sensor_baseline as cap
    ref = np.load(fix)['configs__environment__default.yaml']
    got = cap.rollout('configs/environment/default.yaml')
    assert got.shape == ref.shape
    assert got.tobytes() == ref.tobytes(), (
        f"max abs diff {np.abs(got - ref).max():.3e}")


# ------------------------------------------------------------- breakdown ---

@pytest.mark.parametrize('olf,vis', [(0, 0), (1, 0), (0, 2), (1, 2), (2, 1)])
def test_breakdown_sums_to_actual_observation_width(olf, vis):
    """CP2 — a mismatch here silently mis-assigns perceptual noise."""
    p = params(**{'sensory.olfactory_grid_range': olf, 'sensory.visual_sensor_range': vis})
    _, obs = one_state(p)
    bd = get_observation_breakdown(p)
    assert sum(bd.values()) == obs.shape[0]
    assert bd['Olfaction'] == (2 * olf ** 2 + 2 * olf + 1) * int(p.res_property.shape[-1])
    assert bd['Visual'] == (2 * vis ** 2 + 2 * vis + 1) * int(p.visual_vector_size)


# ---------------------------------------------------------- psf geometry ---

def _W(p, st):
    cells = st.agent_pos + get_visual_offsets(p.visual_sensor_range)
    pos = jnp.concatenate([st.res_pos, st.animal_pos, st.obs_pos], 0)
    return np.asarray(_psf_weights(cells, pos, st.agent_pos, p)), np.asarray(cells), np.asarray(pos)


def test_psf_is_finite_everywhere_including_on_the_agent_cell():
    """CP3. sigma_floor exists precisely so an entity at d=0 does not divide by zero."""
    p = params(**{'sensory.visual_sensor_range': 2, 'sensory.visual_blur_enabled': True})
    st, _ = one_state(p)
    W, _, _ = _W(p, st)
    assert np.isfinite(W).all() and not np.isnan(W).any()
    # force an entity onto the agent's own cell
    st2 = st.replace(res_pos=st.res_pos.at[0].set(st.agent_pos))
    W2, _, _ = _W(p, st2)
    assert np.isfinite(W2).all(), "sigma_floor must keep the on-agent case finite"


def test_anisotropy_one_is_isotropic():
    """CP3 — rho=1 must give equal weight to cells equidistant from the entity."""
    p = params(**{'sensory.visual_sensor_range': 2, 'sensory.visual_blur_enabled': True,
                  'sensory.visual_blur_anisotropy': 1.0})
    st, _ = one_state(p)
    W, cells, pos = _W(p, st)
    d = np.linalg.norm(cells - pos[0], axis=1)
    same = np.abs(d - d[0]) < 1e-6
    assert np.allclose(W[same, 0], W[same, 0][0], rtol=1e-5)


def test_total_weight_falls_off_with_distance():
    """CP4 — if this fails the normalisation is wrong and distance carries no signal."""
    p = params(**{'sensory.visual_sensor_range': 2, 'sensory.visual_blur_enabled': True})
    st, _ = one_state(p)
    agent = st.agent_pos
    cells = agent + get_visual_offsets(2)
    totals = []
    for dist in (1, 2, 3, 4, 5):
        e = jnp.array([[agent[0] - dist, agent[1]]], dtype=st.res_pos.dtype)
        totals.append(float(np.asarray(_psf_weights(cells, e, agent, p)).sum()))
    assert all(a > b for a, b in zip(totals, totals[1:])), f"not monotonic: {totals}"


# ---------------------------------------------------------------- masking ---

def test_far_mask_leaks_nothing_anywhere_including_the_centre_cell():
    """Regression for the Critical review finding: gating on the CELL's distance
    instead of the ENTITY's left a masked entity depositing its blur tail in the
    agent's own cell."""
    agent = jnp.array([5, 5])
    ents = jnp.array([[5, 5], [4, 5], [3, 5], [8, 8]])       # co-located, 1, 2, 6 away
    gate = np.asarray(_visual_mask_gate(agent, ents, jnp.array([1, 1, 1, 1])))[0]
    assert gate[0] == 1.0, "'far' stays visible when the agent stands on it"
    assert (gate[1:] == 0.0).all(), "'far' must contribute exactly zero at any d>=1"


def test_all_mask_hides_everywhere_and_none_hides_nothing():
    agent = jnp.array([5, 5])
    ents = jnp.array([[5, 5], [4, 5], [8, 8]])
    assert (np.asarray(_visual_mask_gate(agent, ents, jnp.array([2, 2, 2])))[0] == 0.0).all()
    assert (np.asarray(_visual_mask_gate(agent, ents, jnp.array([0, 0, 0])))[0] == 1.0).all()


def test_unknown_visual_mask_string_raises_at_load():
    with pytest.raises(ValueError, match='visual_mask'):
        _read_visual_mask({'visual_mask': 'invisible'}, 'TestEntity')
    assert _read_visual_mask({}, 'TestEntity') == 0            # absent -> none


# ------------------------------------------------------- olfactory diamond ---

def test_diamond_reading_is_highest_toward_the_source():
    p = params(**{'sensory.olfactory_grid_range': 1})
    st, _ = one_state(p)
    agent = jnp.array([5, 5])
    V = int(p.res_property.shape[-1])
    for (dr, dc), name in [((-2, 0), 'north'), ((2, 0), 'south'),
                           ((0, 2), 'east'), ((0, -2), 'west')]:
        st2 = st.replace(
            agent_pos=agent,
            res_pos=jnp.full_like(st.res_pos, 99).at[0].set(jnp.array([5 + dr, 5 + dc])),
            res_active=jnp.zeros_like(st.res_active).at[0].set(True),
            animal_active=jnp.zeros_like(st.animal_active),
            obs_active=jnp.zeros_like(st.obs_active))
        out = np.asarray(sense_olfaction_cells(st2, p)).reshape(-1, V)
        offs = np.asarray(get_visual_offsets(1))
        toward = int(np.argmin(np.abs(offs - np.array([np.sign(dr), np.sign(dc)])).sum(1)))
        away = int(np.argmin(np.abs(offs + np.array([np.sign(dr), np.sign(dc)])).sum(1)))
        assert out[toward, 0] > out[away, 0], f"{name}: gradient points the wrong way"


def test_out_of_bounds_diamond_cells_read_exactly_zero():
    """Without this the suite stays green while olfaction reads through walls."""
    p = params(**{'sensory.olfactory_grid_range': 1})
    st, _ = one_state(p)
    V = int(p.res_property.shape[-1])
    st2 = st.replace(agent_pos=jnp.array([0, 0]))              # top-left corner
    out = np.asarray(sense_olfaction_cells(st2, p)).reshape(-1, V)
    offs = np.asarray(get_visual_offsets(1))
    oob = [i for i, (dr, dc) in enumerate(offs) if dr < 0 or dc < 0]
    assert oob, "corner placement should put some cells out of bounds"
    assert (out[oob] == 0.0).all(), "out-of-bounds cells must be exactly zero"


def test_range_zero_matches_the_single_point_sensor_bitwise():
    p = params()
    st, _ = one_state(p)
    from src.environment.sensor import _sense_olfaction_at
    a = np.asarray(sense_olfaction_cells(st, p))
    b = np.asarray(_sense_olfaction_at(st.agent_pos, st, p))
    assert a.tobytes() == b.tobytes()
