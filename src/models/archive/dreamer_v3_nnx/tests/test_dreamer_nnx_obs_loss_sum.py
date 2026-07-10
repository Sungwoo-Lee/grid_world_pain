"""Regression test for WP-NNX F3 (U3) — observation reconstruction loss
under-weighted by ~the observation dimension.

Plain-language context: the canonical DreamerV3 observation loss is a
symlog-MSE log-probability SUMMED over the feature (event) dimensions, then
averaged over batch and time only (sheeprl loss.py:61 via SymlogDistribution;
same in Hafner's code). Our implementation took the mean over batch, time,
AND feature dims — with this project's ~40-60-dim observation vectors that
under-weights reconstruction by ~40-60x relative to the reward, continue, and
KL terms, materially shifting what the latent state is forced to encode.

See docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/
fix_plan_nnx_parity.md (F3) and 05_dreamer_v3_nnx_conventions.md (U3).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

from src.models.dreamer_v3_trainer import dreamer_obs_recon_loss


def test_recon_loss_sums_over_feature_dim():
    """The recon loss must equal mean over (B, T) of the SUM over the feature
    dim of squared errors — i.e. exactly D x the (pre-fix) all-dims mean.
    Pre-fix: off by exactly xD."""
    B, T, D = 3, 5, 40
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    recon = jax.random.normal(key1, (B, T, D))
    obs = jax.random.normal(key2, (B, T, D))

    expected = jnp.mean(jnp.sum(jnp.square(recon - obs), axis=-1))
    old_mean_form = jnp.mean(jnp.square(recon - obs))

    got = dreamer_obs_recon_loss(recon, obs)

    # Guard: the two conventions genuinely differ by xD on this fixture.
    np.testing.assert_allclose(np.asarray(expected),
                               np.asarray(old_mean_form) * D, rtol=1e-6)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=1e-6,
        err_msg=(f"recon loss must SUM over the {D}-dim feature axis "
                 f"(recipe: sheeprl loss.py:61); the mean-form under-weights "
                 f"reconstruction by exactly x{D}"))


def test_recon_loss_scalar_and_batch_shape_agnostic():
    """Same convention on a flat (N, D) batch — sum over the last axis only."""
    N, D = 7, 13
    recon = jnp.ones((N, D)) * 2.0
    obs = jnp.zeros((N, D))
    # each element sq err = 4.0 -> sum over D = 4D, mean over N = 4D
    got = dreamer_obs_recon_loss(recon, obs)
    np.testing.assert_allclose(np.asarray(got), 4.0 * D, rtol=1e-6)
