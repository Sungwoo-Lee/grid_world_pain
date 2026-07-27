"""D-018 — Modality-hierarchical encoder/decoder tests (designs 2+3).

Plan: docs/develop/active/dreamer_srl_v2/hierarchical_modality_encoder_plan.md
(+ plan-review amendments C1/C2/C3 in
docs/reviews/plan_hierarchical_modality_encoder.md).

Covers:
  1. Flat default is unaffected: build_agent flat with the observation_breakdown
     kwarg supplied ≡ without it (identical param tree, values included).
  2. Reparameterisation: HeadsMLPDecoder with the flat MLPDecoder's weights
     transplanted (head kernel partitioned along the OUTPUT axis — columns in
     Flax (in, out) convention, plan-review C1) matches the flat decoder in
     outputs + obs loss (rtol 1e-5, plan-review C2) and per-block head
     gradients (rtol 1e-6).
  3. Shapes / finiteness / vmap-compat for designs 2+3.
  4. Fail-loud config validation (ValueError / KeyError matrix).
  5. Gradient reaches every encoder branch and every mirror-decoder branch
     (the memo's starvation warning makes this non-optional).
  6. RSSM consumes encoder.output_dim, not encoder dense_units — tested with
     hidden_size (192) ≠ dense_units (256) so the XS coincidence cannot mask
     wrong wiring (plan-review C3).
  7. Checkpoint save→restore roundtrip per design 2 and 3; restoring a FLAT
     checkpoint into a design-2/3 agent raises (cross-mode restore fails
     loudly).
  8. End-to-end train-step smoke for designs 2+3 (losses finite).
"""
import copy
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')

from src.algorithms.dreamer_srl.agent import (
    HeadsMLPDecoder,
    HierarchicalMLPEncoder,
    MirrorMLPDecoder,
    MLPDecoder,
    build_agent,
)
from src.algorithms.dreamer_srl.loss import SymlogDistribution

# ---------------------------------------------------------------------------
# Helpers — small sizing so the whole module runs fast on CPU
# ---------------------------------------------------------------------------

# Small breakdown (3 modalities, obs_dim 8) for most tests; the widths are
# deliberately unequal so slicing bugs cannot cancel out.
BD_SMALL = {"Satiation": 1, "Olfaction": 3, "Visual": 4}
OBS_DIM_SMALL = 8
ACTION_DIM = 4

HIER_PARAMS_SMALL = {
    "mirror_encoder_in_decoder": False,
    "default_mlp": [32],
    "multimodal_hub": [16, 16],
    "hidden_size": 48,
}


def _small_cfg(encoding_mode=None, hierarchical_params=None):
    cfg = {
        "algo": {
            "world_model": {
                "encoder": {"dense_units": 64, "mlp_layers": 1},
                "decoder": {"dense_units": 64, "mlp_layers": 1},
                "recurrent_model": {"recurrent_state_size": 64, "dense_units": 64},
                "transition_model": {"hidden_size": 64},
                "representation_model": {"hidden_size": 64},
                "reward_model": {"bins": 15, "dense_units": 64, "mlp_layers": 1},
                "continue_model": {"dense_units": 64, "mlp_layers": 1},
                "stochastic_size": 4,
                "discrete_size": 4,
            },
            "actor": {"dense_units": 64, "mlp_layers": 1},
            "critic": {"dense_units": 64, "mlp_layers": 1, "bins": 15},
            "unimix": 0.01,
        }
    }
    wm = cfg["algo"]["world_model"]
    if encoding_mode is not None:
        wm["encoding_mode"] = encoding_mode
    if hierarchical_params is not None:
        wm["hierarchical_params"] = copy.deepcopy(hierarchical_params)
    return cfg


def _hier_cfg(mirror: bool, **overrides):
    hp = dict(HIER_PARAMS_SMALL, mirror_encoder_in_decoder=mirror, **overrides)
    return _small_cfg(encoding_mode="hierarchical", hierarchical_params=hp)


def _build(cfg, breakdown=None, seed=0):
    return build_agent(
        obs_dim=OBS_DIM_SMALL,
        action_dim=ACTION_DIM,
        cfg=cfg,
        rngs=nnx.Rngs(jax.random.PRNGKey(seed)),
        observation_breakdown=breakdown,
    )


def _param_leaves(module):
    return jax.tree_util.tree_leaves_with_path(nnx.state(module, nnx.Param))


# ---------------------------------------------------------------------------
# 1. Flat default unaffected
# ---------------------------------------------------------------------------

def test_flat_with_and_without_breakdown_kwarg_identical():
    """Flat build_agent: supplying observation_breakdown changes nothing."""
    mods_no_kwarg = build_agent(
        obs_dim=OBS_DIM_SMALL, action_dim=ACTION_DIM, cfg=_small_cfg(),
        rngs=nnx.Rngs(jax.random.PRNGKey(7)),
    )
    mods_kwarg = build_agent(
        obs_dim=OBS_DIM_SMALL, action_dim=ACTION_DIM, cfg=_small_cfg(),
        rngs=nnx.Rngs(jax.random.PRNGKey(7)),
        observation_breakdown=BD_SMALL,
    )
    for m1, m2 in zip(mods_no_kwarg, mods_kwarg):
        l1, l2 = _param_leaves(m1), _param_leaves(m2)
        assert len(l1) == len(l2)
        for (p1, a1), (p2, a2) in zip(l1, l2):
            assert jax.tree_util.keystr(p1) == jax.tree_util.keystr(p2)
            np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))


def test_flat_explicit_mode_equals_default():
    """encoding_mode: flat (explicit) ≡ absent (defaulted)."""
    mods_default = build_agent(
        obs_dim=OBS_DIM_SMALL, action_dim=ACTION_DIM, cfg=_small_cfg(),
        rngs=nnx.Rngs(jax.random.PRNGKey(3)),
    )
    mods_explicit = build_agent(
        obs_dim=OBS_DIM_SMALL, action_dim=ACTION_DIM,
        cfg=_small_cfg(encoding_mode="flat"),
        rngs=nnx.Rngs(jax.random.PRNGKey(3)),
    )
    for m1, m2 in zip(mods_default, mods_explicit):
        for (_, a1), (_, a2) in zip(_param_leaves(m1), _param_leaves(m2)):
            np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))


# ---------------------------------------------------------------------------
# 2. Reparameterisation: heads decoder ≡ flat decoder after weight transplant
# ---------------------------------------------------------------------------

def _obs_loss(pred_symlog, target_real):
    """The obs-loss term exactly as loss.py builds it (SymlogDistribution)."""
    return -SymlogDistribution(pred_symlog, dims=1).log_prob(target_real).mean()


def test_heads_decoder_is_exact_reparameterisation():
    """Transplanted HeadsMLPDecoder ≡ flat MLPDecoder: outputs, loss, grads.

    The flat head kernel is (in=dense_units, out=obs_dim); per-key blocks are
    COLUMN slices W[:, lo:hi] + bias b[lo:hi] (output-axis partition in
    breakdown order — plan-review C1, memo §6.1). Tolerances: rtol 1e-5 for
    outputs/loss (fp32 re-association bound ~3e-6 — plan-review C2), rtol 1e-6
    for the per-block gradient comparison (identical dot products).
    """
    latent_dim, dense_units, mlp_layers = 40, 64, 2

    flat = MLPDecoder(
        latent_dim=latent_dim, obs_dim=OBS_DIM_SMALL,
        dense_units=dense_units, mlp_layers=mlp_layers,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    heads = HeadsMLPDecoder(
        latent_dim=latent_dim, observation_breakdown=BD_SMALL,
        dense_units=dense_units, mlp_layers=mlp_layers,
        rngs=nnx.Rngs(jax.random.PRNGKey(1)),
    )

    # Transplant trunk weights (flat body → heads trunk), incl. LayerNorms.
    for f_lin, h_lin in zip(flat.hidden_linears, heads.trunk_linears):
        h_lin.kernel = nnx.Param(f_lin.kernel[...])
    for f_norm, h_norm in zip(flat.hidden_norms, heads.trunk_norms):
        h_norm.scale = nnx.Param(f_norm.scale[...])
        h_norm.bias = nnx.Param(f_norm.bias[...])

    # Transplant head: COLUMN blocks of the (dense_units, obs_dim) kernel.
    W = flat.output_head.kernel[...]      # (in=dense_units, out=obs_dim)
    b = flat.output_head.bias[...]        # (obs_dim,)
    assert W.shape == (dense_units, OBS_DIM_SMALL)
    lo = 0
    for head, width in zip(heads.heads, BD_SMALL.values()):
        head.kernel = nnx.Param(W[:, lo:lo + width])
        head.bias = nnx.Param(b[lo:lo + width])
        lo += width

    latents = jax.random.normal(jax.random.PRNGKey(2), (64, latent_dim))
    target = jax.random.normal(jax.random.PRNGKey(3), (64, OBS_DIM_SMALL)) * 3.0

    out_flat = flat(latents)
    out_heads = heads(latents)
    np.testing.assert_allclose(np.asarray(out_heads), np.asarray(out_flat),
                               rtol=1e-5, atol=1e-6)

    loss_flat = _obs_loss(out_flat, target)
    loss_heads = _obs_loss(out_heads, target)
    np.testing.assert_allclose(float(loss_heads), float(loss_flat), rtol=1e-5)

    # Per-block head gradients: heads[k].kernel grad ≡ column block of the
    # flat head's kernel grad (rtol 1e-6 — plan-review C2 keeps this tight).
    def flat_loss_fn(dec):
        return _obs_loss(dec(latents), target)

    def heads_loss_fn(dec):
        return _obs_loss(dec(latents), target)

    g_flat = nnx.grad(flat_loss_fn)(flat)
    g_heads = nnx.grad(heads_loss_fn)(heads)

    gW = np.asarray(g_flat["output_head"]["kernel"][...])   # (dense_units, obs_dim)
    gb = np.asarray(g_flat["output_head"]["bias"][...])
    lo = 0
    for k, width in enumerate(BD_SMALL.values()):
        gk = np.asarray(g_heads["heads"][k]["kernel"][...])
        gbk = np.asarray(g_heads["heads"][k]["bias"][...])
        np.testing.assert_allclose(gk, gW[:, lo:lo + width], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(gbk, gb[lo:lo + width], rtol=1e-6, atol=1e-8)
        lo += width


# ---------------------------------------------------------------------------
# 3. Shapes / finiteness / vmap-compat (designs 2+3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mirror", [False, True])
def test_hier_shapes_finiteness_vmap(mirror):
    wm, actor, critic, tc = _build(_hier_cfg(mirror), breakdown=BD_SMALL)

    hidden = HIER_PARAMS_SMALL["hidden_size"]
    assert isinstance(wm.encoder, HierarchicalMLPEncoder)
    assert wm.encoder.output_dim == hidden
    expected_dec = MirrorMLPDecoder if mirror else HeadsMLPDecoder
    assert isinstance(wm.decoder, expected_dec)

    B = 6
    obs = jax.random.normal(jax.random.PRNGKey(0), (B, OBS_DIM_SMALL)) * 2.0
    emb = jax.vmap(wm.encoder)(obs)               # vmap exactly as observe() uses it
    assert emb.shape == (B, hidden)
    assert bool(jnp.isfinite(emb).all())

    latent_dim = 4 * 4 + 64  # stoch_flat + recurrent_state_size in _small_cfg
    latents = jax.random.normal(jax.random.PRNGKey(1), (B, latent_dim))
    rec = jax.vmap(wm.decoder)(latents)
    assert rec.shape == (B, OBS_DIM_SMALL)
    assert bool(jnp.isfinite(rec).all())

    # Full observe pass (encoder → RSSM → decoder wiring end-to-end)
    T, B2 = 3, 2
    out = wm.observe(
        jax.random.normal(jax.random.PRNGKey(2), (T, B2, OBS_DIM_SMALL)),
        jnp.zeros((T, B2, ACTION_DIM)),
        jnp.ones((T, B2, 1)),
        jax.random.PRNGKey(3),
    )
    assert out["reconstructed_obs"].shape == (T, B2, OBS_DIM_SMALL)
    assert bool(jnp.isfinite(out["reconstructed_obs"]).all())


# ---------------------------------------------------------------------------
# 4. Fail-loud config validation
# ---------------------------------------------------------------------------

def test_hierarchical_without_breakdown_raises():
    with pytest.raises(ValueError, match="observation_breakdown"):
        _build(_hier_cfg(False), breakdown=None)


def test_breakdown_sum_mismatch_raises():
    bad = {"Satiation": 1, "Olfaction": 3, "Visual": 5}  # sums to 9 != 8
    with pytest.raises(ValueError, match="sums to 9"):
        _build(_hier_cfg(False), breakdown=bad)


def test_flat_with_hierarchical_params_raises():
    cfg = _small_cfg(encoding_mode="flat", hierarchical_params=HIER_PARAMS_SMALL)
    with pytest.raises(ValueError, match="hierarchical_params"):
        _build(cfg, breakdown=BD_SMALL)


def test_defaulted_flat_with_hierarchical_params_raises():
    # encoding_mode absent (defaults to flat) + block present → still rejected
    cfg = _small_cfg(hierarchical_params=HIER_PARAMS_SMALL)
    with pytest.raises(ValueError, match="hierarchical_params"):
        _build(cfg, breakdown=BD_SMALL)


def test_flat_with_only_mirror_key_raises():
    # A hierarchical_params block containing ONLY mirror_encoder_in_decoder is
    # still the rejected flat+block combination (no silently-ignored key).
    cfg = _small_cfg(hierarchical_params={"mirror_encoder_in_decoder": True})
    with pytest.raises(ValueError, match="hierarchical_params"):
        _build(cfg, breakdown=BD_SMALL)


def test_non_boolean_mirror_raises():
    cfg = _hier_cfg(False)
    cfg["algo"]["world_model"]["hierarchical_params"]["mirror_encoder_in_decoder"] = "false"
    with pytest.raises(ValueError, match="mirror_encoder_in_decoder"):
        _build(cfg, breakdown=BD_SMALL)


def test_unknown_encoding_mode_raises():
    with pytest.raises(ValueError, match="encoding_mode"):
        _build(_small_cfg(encoding_mode="hierarchic"), breakdown=BD_SMALL)


def test_missing_hierarchical_params_raises_keyerror():
    cfg = _small_cfg(encoding_mode="hierarchical")
    with pytest.raises(KeyError):
        _build(cfg, breakdown=BD_SMALL)


@pytest.mark.parametrize("missing", [
    "mirror_encoder_in_decoder", "default_mlp", "multimodal_hub", "hidden_size",
])
def test_missing_hierarchical_subkey_raises_keyerror(missing):
    cfg = _hier_cfg(False)
    del cfg["algo"]["world_model"]["hierarchical_params"][missing]
    with pytest.raises(KeyError):
        _build(cfg, breakdown=BD_SMALL)


# ---------------------------------------------------------------------------
# 5. Gradient reaches every branch
# ---------------------------------------------------------------------------

def test_gradient_reaches_every_encoder_branch():
    enc = HierarchicalMLPEncoder(
        observation_breakdown=BD_SMALL, default_mlp=[32],
        multimodal_hub=[16], hidden_size=48,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    obs = jax.random.normal(jax.random.PRNGKey(1), (16, OBS_DIM_SMALL))

    def loss_fn(e):
        return (e(obs) ** 2).mean()

    g = nnx.grad(loss_fn)(enc)
    gW = np.asarray(g["branch_linears"][0]["weights"][...])  # [G, max_in, 32]
    for gi, name in enumerate(BD_SMALL):
        assert np.abs(gW[gi]).max() > 0.0, \
            f"encoder branch {name!r} receives no gradient"


def test_gradient_reaches_every_mirror_decoder_branch():
    latent_dim = 40
    dec = MirrorMLPDecoder(
        latent_dim=latent_dim, observation_breakdown=BD_SMALL,
        default_mlp=[32], multimodal_hub=[16], hidden_size=48,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    latents = jax.random.normal(jax.random.PRNGKey(1), (16, latent_dim))
    target = jax.random.normal(jax.random.PRNGKey(2), (16, OBS_DIM_SMALL))

    def loss_fn(d):
        return _obs_loss(d(latents), target)

    g = nnx.grad(loss_fn)(dec)
    gW = np.asarray(g["branch_linears"][0]["weights"][...])  # [G, hidden, 32]
    for gi, name in enumerate(BD_SMALL):
        assert np.abs(gW[gi]).max() > 0.0, \
            f"mirror-decoder branch {name!r} receives no gradient"
    for gi, name in enumerate(BD_SMALL):
        gk = np.asarray(g["output_heads"][gi]["kernel"][...])
        assert np.abs(gk).max() > 0.0, \
            f"mirror-decoder output head {name!r} receives no gradient"


# ---------------------------------------------------------------------------
# 6. RSSM consumes encoder.output_dim (hidden_size ≠ dense_units — C3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mirror", [False, True])
def test_rssm_consumes_encoder_output_dim_not_dense_units(mirror):
    """hidden_size 192 ≠ encoder dense_units 256: wiring must still work.

    At XS both are 256, so a build that (wrongly) passes enc_dense_units to
    the RSSM would pass every equal-size test — this asymmetric sizing is the
    only configuration that can catch the bug (plan-review C3).
    """
    cfg = _hier_cfg(mirror, hidden_size=192)
    cfg["algo"]["world_model"]["encoder"]["dense_units"] = 256
    wm, actor, critic, tc = _build(cfg, breakdown=BD_SMALL)

    assert wm.encoder.output_dim == 192
    # Structural probe: run observe end-to-end — a wrong RSSM input width
    # (e.g. dense_units=256 wrongly passed as encoder_output_dim) fails with
    # a shape error inside _representation.
    out = wm.observe(
        jax.random.normal(jax.random.PRNGKey(0), (2, 3, OBS_DIM_SMALL)),
        jnp.zeros((2, 3, ACTION_DIM)),
        jnp.ones((2, 3, 1)),
        jax.random.PRNGKey(1),
    )
    assert bool(jnp.isfinite(out["reconstructed_obs"]).all())


# ---------------------------------------------------------------------------
# 7. Checkpoint roundtrip + cross-mode restore fails loudly
# ---------------------------------------------------------------------------

def _save_ckpt(tmp_path, mods, episode=10):
    from src.algorithms.dreamer_srl.checkpoint import (
        make_checkpoint_manager, save_checkpoint)
    from src.algorithms.dreamer_srl.utils import moments_init
    wm, actor, critic, tc = mods
    manager = make_checkpoint_manager(str(tmp_path), max_to_keep=2)
    save_checkpoint(
        manager=manager, episode=episode, world_model=wm, actor=actor,
        critic=critic, target_critic=tc, moments=moments_init(),
        key=jax.random.PRNGKey(0), iter_num=1, policy_step=1,
        total_episodes_completed=episode, cumulative_grad_steps=1,
    )
    manager.wait_until_finished()
    return manager


@pytest.mark.parametrize("mirror", [False, True])
def test_checkpoint_roundtrip_hierarchical(mirror, tmp_path):
    from src.algorithms.dreamer_srl.checkpoint import restore_dreamer_training_state
    from src.algorithms.dreamer_srl.utils import moments_init

    mods = _build(_hier_cfg(mirror), breakdown=BD_SMALL, seed=0)
    _save_ckpt(tmp_path, mods)

    # Fresh same-design agent with different init — restore must reproduce mods
    mods2 = _build(_hier_cfg(mirror), breakdown=BD_SMALL, seed=99)
    restore_dreamer_training_state(
        tmp_path / "checkpoints", *mods2, moments_init(),
        jax.random.PRNGKey(0), quiet=True,
    )
    for m1, m2 in zip(mods, mods2):
        for (_, a1), (_, a2) in zip(_param_leaves(m1), _param_leaves(m2)):
            np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))


@pytest.mark.parametrize("mirror", [False, True])
def test_flat_checkpoint_into_hierarchical_raises(mirror, tmp_path):
    from src.algorithms.dreamer_srl.checkpoint import restore_dreamer_training_state
    from src.algorithms.dreamer_srl.utils import moments_init

    flat_mods = build_agent(
        obs_dim=OBS_DIM_SMALL, action_dim=ACTION_DIM, cfg=_small_cfg(),
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    _save_ckpt(tmp_path, flat_mods)

    hier_mods = _build(_hier_cfg(mirror), breakdown=BD_SMALL, seed=1)
    with pytest.raises(Exception):
        restore_dreamer_training_state(
            tmp_path / "checkpoints", *hier_mods, moments_init(),
            jax.random.PRNGKey(0), quiet=True,
        )


# ---------------------------------------------------------------------------
# 8. End-to-end train-step smoke (designs 2+3): losses finite, diags present
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mirror", [False, True])
def test_train_step_smoke_hierarchical(mirror):
    from src.algorithms.dreamer_srl.train import make_train_step
    from src.algorithms.dreamer_srl.utils import moments_init, make_optim_tx

    wm, actor, critic, tc = _build(_hier_cfg(mirror), breakdown=BD_SMALL)
    wm_opt = nnx.Optimizer(wm, make_optim_tx(1e-4, 1e-8, 1000.0), wrt=nnx.Param)
    actor_opt = nnx.Optimizer(actor, make_optim_tx(8e-5, 1e-5, 100.0), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, make_optim_tx(8e-5, 1e-5, 100.0), wrt=nnx.Param)

    train_step = make_train_step(
        horizon=3, gamma=0.99, lmbda=0.95, ent_coef=3e-4,
        kl_dynamic=0.5, kl_representation=0.1, kl_free_nats=1.0,
        obs_breakdown=tuple(BD_SMALL.items()),
    )
    T, B = 4, 3
    k = jax.random.PRNGKey(0)
    batch = {
        "obs": jax.random.normal(k, (T, B, OBS_DIM_SMALL)),
        "actions": jax.nn.one_hot(
            jax.random.randint(k, (T, B), 0, ACTION_DIM), ACTION_DIM),
        "rewards": jnp.zeros((T, B, 1)),
        "is_first": jnp.zeros((T, B, 1)),
        "terminated": jnp.zeros((T, B, 1)),
    }
    moments, losses = train_step(
        wm, actor, critic, tc, wm_opt, actor_opt, critic_opt,
        moments_init(), batch, jax.random.PRNGKey(1),
    )
    for name in ("world_model_loss", "policy_loss", "value_loss"):
        assert bool(jnp.isfinite(losses[name])), f"{name} not finite"
    # D-018 File Change 7: per-modality diagnostics present for every modality
    for mname in BD_SMALL:
        slug = mname.lower().replace(" ", "_")
        assert f"wm/recon_mse/{slug}" in losses
        assert f"wm/target_var/{slug}" in losses
        assert bool(jnp.isfinite(losses[f"wm/recon_mse/{slug}"]))


def test_train_step_without_breakdown_has_no_diag_keys():
    """Legacy path (obs_breakdown=None): losses key set unchanged."""
    from src.algorithms.dreamer_srl.train import make_train_step
    from src.algorithms.dreamer_srl.utils import moments_init, make_optim_tx

    wm, actor, critic, tc = build_agent(
        obs_dim=OBS_DIM_SMALL, action_dim=ACTION_DIM, cfg=_small_cfg(),
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    wm_opt = nnx.Optimizer(wm, make_optim_tx(1e-4, 1e-8, 1000.0), wrt=nnx.Param)
    actor_opt = nnx.Optimizer(actor, make_optim_tx(8e-5, 1e-5, 100.0), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, make_optim_tx(8e-5, 1e-5, 100.0), wrt=nnx.Param)
    train_step = make_train_step(
        horizon=3, gamma=0.99, lmbda=0.95, ent_coef=3e-4,
        kl_dynamic=0.5, kl_representation=0.1, kl_free_nats=1.0,
    )
    T, B = 4, 3
    k = jax.random.PRNGKey(0)
    batch = {
        "obs": jax.random.normal(k, (T, B, OBS_DIM_SMALL)),
        "actions": jax.nn.one_hot(
            jax.random.randint(k, (T, B), 0, ACTION_DIM), ACTION_DIM),
        "rewards": jnp.zeros((T, B, 1)),
        "is_first": jnp.zeros((T, B, 1)),
        "terminated": jnp.zeros((T, B, 1)),
    }
    _, losses = train_step(
        wm, actor, critic, tc, wm_opt, actor_opt, critic_opt,
        moments_init(), batch, jax.random.PRNGKey(1),
    )
    assert not any(kk.startswith("wm/recon_mse/") or kk.startswith("wm/target_var/")
                   for kk in losses)
