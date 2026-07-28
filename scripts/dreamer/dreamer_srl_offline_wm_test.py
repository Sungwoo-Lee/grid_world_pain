#!/usr/bin/env python
"""Offline world-model imagination diagnostic for a frozen dreamer-srl v2 checkpoint.

Loads a saved checkpoint, rolls out the trained actor in eval mode for N_REAL_STEPS
(or replays from recorded eval episodes), samples M starting states, runs the WM in
imagination for H_max steps from each, compares predicted reward vs actual at horizons
{1, 5, 10, 25, 50} split by positive/negative true reward, and writes JSON + Markdown
reports under tmp/.

The headline question: does per-horizon reward MAE compound as horizon grows (the Z2
cascade finding: 0.18 → 3.05 at h=5→50), and is the compounding steeper on negative-
reward steps than positive ones?

Usage:
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
    scripts/dreamer_srl_offline_wm_test.py \\
    --checkpoint results/JAX_DreamerSRL/dreamer_srl_v2_10x10_ext_XS_envs_16_4M_s42 \\
    --env-config configs/environment/experiment/archive/hypervigilance/01-interoNocicept.yaml \\
    --agent-config configs/models/dreamer_srl/agent_xs.yaml \\
    --num-starts 200 \\
    --output tmp/20260518_dreamer_srl_wm_diag_yxij4lrc.json

All horizons and thresholds are pre-registered constants (do NOT change without
updating offline_wm_diagnostic_port_plan.md).

Failure-mode catalog:
  NaN/inf in any imagined output  → RuntimeError with start index.
  n_valid_starts < max(M//4, 50) → RuntimeError asking for more steps.
  Missing agent config key        → KeyError from build_agent.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple

# Disable JAX preallocation before any JAX import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Pre-registered constants (do NOT change without updating the plan doc)
# ---------------------------------------------------------------------------
HORIZONS = (1, 5, 10, 25, 50)
H_MAX = 50
N_REAL_STEPS = 2000
M_DEFAULT = 200
ENV_SEED = 42
MIN_SAMPLES_PER_SIDE = 30   # below this n, pos/neg MAE is flagged low_n and excluded from ratios

# ---------------------------------------------------------------------------
# Repo root + project imports (set up before any src.* imports)
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import orbax.checkpoint as ocp
from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params, load_env_config
from src.environment.wrapper import ParallelEnv
from src.environment.sensor import get_observation_breakdown
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.checkpoint import make_checkpoint_manager
from src.algorithms.dreamer_srl.loss import TwoHotEncoding
from src.algorithms.dreamer_srl.utils import symexp, symlog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Offline dreamer-srl v2 WM imagination diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run against the yxij4lrc winner (16-env XS, 4M steps):
  python scripts/dreamer_srl_offline_wm_test.py \\
    --checkpoint results/JAX_DreamerSRL/dreamer_srl_v2_10x10_ext_XS_envs_16_4M_s42 \\
    --env-config configs/environment/experiment/archive/hypervigilance/01-interoNocicept.yaml \\
    --agent-config configs/models/dreamer_srl/agent_xs.yaml

  # Quick CPU smoke run (used by the pytest smoke test):
  python scripts/dreamer_srl_offline_wm_test.py \\
    --checkpoint /tmp/ckpt_dir \\
    --env-config configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml \\
    --agent-config configs/models/dreamer_srl/01_food_only.yaml \\
    --num-real-steps 100 --num-starts 5 --horizon-max 5 --horizons 1,5 \\
    --source real_env --device cpu
""",
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to run directory (parent of checkpoints/) OR checkpoints/ directly.")
    p.add_argument("--checkpoint-step", type=int, default=None,
                   help="Specific checkpoint step to load (default: latest).")
    p.add_argument("--env-config", required=True,
                   help="Environment YAML config path.")
    p.add_argument("--agent-config", required=True,
                   help="Agent YAML config path.")
    p.add_argument("--source", choices=["real_env", "recordings"], default="real_env",
                   help="Observation source: 'real_env' (default, fresh rollout) or "
                        "'recordings' (replay from recordings/<step>/ on disk).")
    p.add_argument("--num-real-steps", type=int, default=N_REAL_STEPS,
                   help=f"Real-env rollout length (default {N_REAL_STEPS}).")
    p.add_argument("--num-starts", type=int, default=M_DEFAULT,
                   help=f"Number of imagination starting states M (default {M_DEFAULT}).")
    p.add_argument("--horizons", type=str, default=",".join(str(h) for h in HORIZONS),
                   help=f"Comma-separated horizons to report (default {','.join(str(h) for h in HORIZONS)}).")
    p.add_argument("--horizon-max", type=int, default=H_MAX,
                   help=f"Maximum imagination horizon (default {H_MAX}).")
    p.add_argument("--seed", type=int, default=ENV_SEED,
                   help=f"PRNG seed (default {ENV_SEED}).")
    p.add_argument("--output", type=str, default=None,
                   help="JSON output path (default: auto-generated under tmp/).")
    p.add_argument("--device", type=str, default="gpu",
                   help="JAX device: 'gpu', 'cpu', or 'gpu:N'.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_merged_config(env_config_path: str, agent_config_path: str):
    """Load and merge env + agent configs (mirrors dreamer_srl_main.py pattern)."""
    config = get_default_config()
    for sub in ("train", "evaluation", "logger", "visualization"):
        cand = os.path.join(_REPO, "configs", sub, "default.yaml")
        if os.path.exists(cand):
            config.merge(Config.load_yaml(cand))
    # load_env_config: if env YAML has `extends:`, its base is merged inside;
    # if standalone (archived), it loads as-is.
    config.merge(load_env_config(env_config_path))
    config.merge(Config.load_yaml(agent_config_path))
    return config


def prefer_saved_config(checkpoint_parent: str, env_config_path: str, agent_config_path: str):
    """Return merged config, preferring a saved config.yaml next to the checkpoint."""
    # Check both the checkpoint_parent and checkpoints/ subdir for a saved config
    for candidate in [
        os.path.join(checkpoint_parent, "config.yaml"),
        os.path.join(checkpoint_parent, "checkpoints", "config.yaml"),
    ]:
        if os.path.exists(candidate):
            print(f"[srl-diag] Using saved config.yaml from {candidate}")
            return Config.load_yaml(candidate)
    print("[srl-diag] No saved config.yaml found — merging from --env-config + --agent-config.")
    return load_merged_config(env_config_path, agent_config_path)


# ---------------------------------------------------------------------------
# Checkpoint normalization (defensive copy from original-Dreamer reference script)
# ---------------------------------------------------------------------------
def _normalize_checkpoint(d):
    """Normalize an orbax-restored checkpoint pytree for nnx.update.

    Two transformations (verbatim from scripts/dreamer_offline_wm_test.py):
    1. Convert string-digit keys to integers.
    2. Unwrap single-key {'value': array} leaf dicts.
    """
    if isinstance(d, dict):
        if set(d.keys()) == {'value'}:
            return _normalize_checkpoint(d['value'])
        return {
            (int(k) if isinstance(k, str) and k.isdigit() else k): _normalize_checkpoint(v)
            for k, v in d.items()
        }
    return d


# ---------------------------------------------------------------------------
# Reward decode (TwoHotEncoding → symexp → real space)
# ---------------------------------------------------------------------------
def decode_reward(reward_logits: jax.Array) -> np.ndarray:
    """Decode 255-bin reward logits to real-space reward.

    Uses TwoHotEncoding bins (linspace(-20, +20, 255) in SYMLOG space), then symexp.
    Matches training-time decode in loss.py:143:
        symexp(sum(probs * bins, axis=-1))

    Note: TwoHotEncoding(dims=0) sets self.dims=() which would NOT reduce over bins.
    We explicitly reduce over the last axis (bins=255) here, which is the correct
    decode for the reward head output.

    Args:
        reward_logits: [..., 255] unnormalized logits
    Returns:
        reward_real: [...] float32 array in real reward space
    """
    probs = jax.nn.softmax(reward_logits, axis=-1)   # [..., 255]
    # Use TwoHotEncoding only for its bins grid (bins in SYMLOG space)
    bins = jnp.linspace(-20.0, 20.0, reward_logits.shape[-1])  # (255,)
    # Weighted sum in symlog space, then symexp back to real space
    symlog_mean = jnp.sum(probs * bins, axis=-1)  # [...] — collapse bins
    return np.array(symexp(symlog_mean))


# ---------------------------------------------------------------------------
# Imagination rollout (mirrors WorldModel.imagine, agent.py:1727-1827)
# ---------------------------------------------------------------------------
def imagine_h_steps(
    world_model,
    actor,
    init_recurrent_state: np.ndarray,   # [recurrent_state_size]
    init_posterior_state: np.ndarray,   # [num_cat, num_cls]
    init_prev_action: np.ndarray,       # [action_dim]
    h_max: int,
    key: jax.Array,
) -> Dict[str, np.ndarray]:
    """Run deterministic imagination from a single starting state for h_max steps.

    Mirrors WorldModel.imagine (agent.py:1727-1827) but uses:
    - argmax actor (deterministic, no Gumbel noise)
    - argmax prior (_transition with sample_state=False) for deterministic rollout

    The action at step h feeds the RSSM at step h+1 (same shift as imagine()).

    Args:
        world_model: dreamer-srl WorldModel (frozen, already restored from checkpoint)
        actor:       dreamer-srl Actor (frozen, already restored from checkpoint)
        init_recurrent_state: [recurrent_state_size] numpy — posterior recurrent at t_start
        init_posterior_state: [num_cat, num_cls] numpy — posterior state at t_start
        init_prev_action:     [action_dim] numpy — previous action at t_start
        h_max:                imagination horizon
        key:                  PRNG key (actor and RSSM are deterministic, key not consumed)

    Returns:
        dict with:
            reward_pred:    (h_max,) float32 — predicted reward in real space
            cont_pred:      (h_max,) float32 — continuation probability (sigmoid)
            obs_pred_symlog:(h_max, obs_dim) float32 — decoded obs in symlog space
    """
    rssm = world_model.rssm
    stochastic_size = rssm.stochastic_size    # S*D (flat)
    recurrent_state_size = rssm.recurrent_state_size
    num_cat = rssm.num_categoricals
    num_cls = rssm.num_classes

    # Add batch dim B=1
    recurrent_state = jnp.array(init_recurrent_state)[None]  # [1, recurrent_state_size]
    posterior = jnp.array(init_posterior_state)[None]         # [1, S, D]
    prev_action = jnp.array(init_prev_action)[None]           # [1, action_dim]

    # Build initial latent = cat(posterior_flat, recurrent_state)
    # This mirrors WorldModel.imagine: imagined_prior_flat = init_latent[:, :stochastic_size]
    #                                  recurrent_state = init_latent[:, stochastic_size:]
    posterior_flat = posterior.reshape(1, -1)   # [1, S*D]
    latent = jnp.concatenate([posterior_flat, recurrent_state], axis=-1)  # [1, latent_dim]

    # Mirror WorldModel.imagine step 0: actor from init_latent
    # (Step 0 action feeds the RSSM at step 1 — action-shift §S2 pattern)
    key, k_act = jax.random.split(key)
    action_logits = actor.forward_logits(latent)            # [1, action_dim]
    action_idx = jnp.argmax(action_logits, axis=-1)         # [1]
    current_action = jax.nn.one_hot(action_idx, actor.action_dim)  # [1, action_dim]

    # imagined_prior at step 0 = posterior (already in [1, S, D] form)
    imagined_prior = posterior  # [1, S, D]

    reward_list = []
    cont_list = []
    obs_list = []

    # Mirror WorldModel.imagine loop: for i in range(1, horizon + 1)
    for h in range(1, h_max + 1):
        key, k_rssm, k_act = jax.random.split(key, 3)

        # --- RecurrentModel forward (same as agent.py:1795-1799 and dynamic() §S4)
        # No §S4 reset here — we are in imagination (no episode boundaries).
        # (The §S4 reset is only for the observe() path — imagination never crosses done.)
        prior_flat = imagined_prior.reshape(1, -1)   # [1, S*D]
        action = current_action                       # [1, action_dim]

        recurrent_input = jnp.concatenate([prior_flat, action], axis=-1)  # [1, S*D+A]
        recurrent_feat = rssm.recurrent_mlp_linear(recurrent_input)        # [1, recurrent_dense]
        recurrent_feat = rssm.recurrent_mlp_norm(recurrent_feat)           # [1, recurrent_dense]
        recurrent_feat = jax.nn.silu(recurrent_feat)                        # [1, recurrent_dense]
        recurrent_state = rssm.gru_cell(recurrent_feat, recurrent_state)   # [1, hx]

        # --- Prior transition (argmax/mode — deterministic)
        # sample_state=False → compute_stochastic_state returns mode (argmax one-hot)
        # This is the dreamer-srl analogue of _apply_rssm_mode in the original script.
        _, imagined_prior = rssm._transition(recurrent_state, sample_state=False, key=None)
        # imagined_prior: [1, S, D] (mode one-hot per categorical)

        # --- New latent = cat(prior_flat_new, recurrent_state)
        prior_flat_new = imagined_prior.reshape(1, -1)                           # [1, S*D]
        new_latent = jnp.concatenate([prior_flat_new, recurrent_state], axis=-1) # [1, latent_dim]

        # --- Decode reward, continue, obs
        reward_logits = world_model.reward_model(new_latent)    # [1, 255]
        reward_real = decode_reward(reward_logits)               # [1] or scalar
        reward_val = float(np.squeeze(reward_real))

        cont_logit = world_model.continue_model(new_latent)     # [1, 1]
        cont_val = float(jax.nn.sigmoid(cont_logit.ravel()[0]))

        obs_pred = world_model.decoder(new_latent)              # [1, obs_dim]

        reward_list.append(reward_val)
        cont_list.append(cont_val)
        obs_list.append(np.array(obs_pred[0]))                  # (obs_dim,)

        # --- Actor for next step (argmax, deterministic)
        action_logits = actor.forward_logits(new_latent)        # [1, action_dim]
        action_idx = jnp.argmax(action_logits, axis=-1)         # [1]
        current_action = jax.nn.one_hot(action_idx, actor.action_dim)  # [1, action_dim]

    return {
        'reward_pred': np.array(reward_list, dtype=np.float32),        # (H,)
        'cont_pred': np.array(cont_list, dtype=np.float32),            # (H,)
        'obs_pred_symlog': np.stack(obs_list, axis=0).astype(np.float32),  # (H, obs_dim)
    }


# ---------------------------------------------------------------------------
# Real-env rollout
# ---------------------------------------------------------------------------
def run_real_env_rollout(
    env,
    world_model,
    actor,
    n_steps: int,
    action_dim: int,
    obs_dim: int,
    seed: int,
    key: jax.Array,
):
    """Roll out the frozen actor in the real env for n_steps steps.

    Returns arrays for obs, reward, terminal per step, plus the per-step
    (recurrent_state, posterior_state, prev_action_onehot) stored AFTER the
    RSSM dynamic step (i.e., the state used to build the latent that seeds
    imagination).

    Mirrors the real-env rollout in scripts/dreamer_offline_wm_test.py.
    """
    rssm = world_model.rssm
    num_cat = rssm.num_categoricals
    num_cls = rssm.num_classes
    rec_size = rssm.recurrent_state_size

    obs_trace = np.zeros((n_steps, obs_dim), dtype=np.float32)
    rew_trace = np.zeros(n_steps, dtype=np.float32)
    term_trace = np.zeros(n_steps, dtype=np.bool_)

    # Per-step state AFTER rssm.dynamic (shape without batch dim)
    rec_trace = np.zeros((n_steps, rec_size), dtype=np.float32)
    post_trace = np.zeros((n_steps, num_cat, num_cls), dtype=np.float32)
    # prev_action at each step = the action chosen at that step (feeds next RSSM)
    action_trace = np.zeros((n_steps, action_dim), dtype=np.float32)

    # Initial RSSM state
    recurrent_state, posterior_state = rssm.get_initial_states(1)
    # recurrent_state: [1, rec_size], posterior_state: [1, S, D]
    prev_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

    key, env_key = jax.random.split(key)
    env_state, obs_batch = env.reset(env_key, 1)
    obs = np.array(obs_batch[0])  # (obs_dim,)
    is_first = np.ones((1, 1), dtype=np.float32)  # first step after reset

    episode_terminals = [False] * n_steps

    for t in range(n_steps):
        obs_trace[t] = obs

        # Encode + RSSM dynamic step
        obs_jax = jnp.asarray(obs[None], dtype=jnp.float32)      # [1, obs_dim]
        embedded = jax.vmap(world_model.encoder)(obs_jax)          # [1, dense_units]
        is_first_jax = jnp.asarray(is_first, dtype=jnp.float32)  # [1, 1]

        key, k_rssm = jax.random.split(key)
        recurrent_state, posterior_state, _, _, _ = rssm.dynamic(
            posterior_state,    # [1, S, D]
            recurrent_state,    # [1, rec_size]
            prev_action,        # [1, action_dim]
            embedded,           # [1, dense_units]
            is_first_jax,       # [1, 1]
            k_rssm,
        )
        # recurrent_state: [1, rec_size], posterior_state: [1, S, D]

        # Build latent and run actor (argmax, eval mode)
        posterior_flat = posterior_state.reshape(1, -1)   # [1, S*D]
        latent = jnp.concatenate([posterior_flat, recurrent_state], axis=-1)  # [1, latent_dim]
        action_logits = actor.forward_logits(latent)      # [1, action_dim]
        action_idx = jnp.argmax(action_logits, axis=-1)   # [1]
        action_onehot = jax.nn.one_hot(action_idx, action_dim)  # [1, action_dim]

        # Store state AFTER RSSM (what imagination seeds from)
        rec_trace[t] = np.array(recurrent_state[0])
        post_trace[t] = np.array(posterior_state[0])
        action_trace[t] = np.array(action_onehot[0])

        # Step env
        action_arr = jnp.array([int(action_idx[0])])
        env_state, next_obs_batch, reward_batch, done_batch, _ = env.step(env_state, action_arr)
        r = float(reward_batch[0])
        d = bool(done_batch[0])

        rew_trace[t] = r
        term_trace[t] = d
        episode_terminals[t] = d

        if d:
            # Auto-reset
            key, reset_key = jax.random.split(key)
            env_state, next_obs_batch = env.reset(reset_key, 1)
            recurrent_state, posterior_state = rssm.get_initial_states(1)
            prev_action = jnp.zeros((1, action_dim), dtype=jnp.float32)
            is_first = np.ones((1, 1), dtype=np.float32)
        else:
            prev_action = action_onehot
            is_first = np.zeros((1, 1), dtype=np.float32)

        obs = np.array(next_obs_batch[0])

    n_episodes = int(sum(episode_terminals))
    return obs_trace, rew_trace, term_trace, rec_trace, post_trace, action_trace, n_episodes, episode_terminals


# ---------------------------------------------------------------------------
# Recordings replay
# ---------------------------------------------------------------------------
def run_recordings_replay(
    recordings_dir: str,
    checkpoint_step: int,
    world_model,
    actor,
    action_dim: int,
    obs_dim: int,
    key: jax.Array,
):
    """Replay saved eval episode recordings through the frozen world model.

    Reads episode_*.rec.gz files from recordings_dir/<checkpoint_step>/.
    For each episode, runs encoder + rssm.dynamic step-by-step to recover
    the (recurrent_state, posterior_state) trajectory, just as real_env would.

    Returns same array tuple as run_real_env_rollout (except n_episodes is known
    from the number of files).
    """
    ep_dir = os.path.join(recordings_dir, str(checkpoint_step))
    if not os.path.isdir(ep_dir):
        raise FileNotFoundError(
            f"Recordings directory not found: {ep_dir}. "
            "Try --source real_env instead."
        )

    rssm = world_model.rssm
    num_cat = rssm.num_categoricals
    num_cls = rssm.num_classes
    rec_size = rssm.recurrent_state_size

    ep_files = sorted(
        f for f in os.listdir(ep_dir)
        if f.startswith("episode_") and f.endswith(".rec.gz")
    )
    if not ep_files:
        raise FileNotFoundError(f"No episode_*.rec.gz files in {ep_dir}.")

    all_obs, all_rew, all_term = [], [], []
    all_rec, all_post, all_act = [], [], []
    n_episodes = 0
    episode_terminals = []

    for ep_file in ep_files:
        path = os.path.join(ep_dir, ep_file)
        with gzip.open(path, 'rb') as fp:
            ep = pickle.load(fp)

        ep_obs = ep['obs'].astype(np.float32)    # (T, obs_dim)
        ep_rew = ep['rewards'].astype(np.float32) # (T,)
        ep_act = ep['actions']                    # (T,) int — may include -1 at t=0
        T = len(ep_obs)

        # Initialize RSSM state for this episode
        recurrent_state, posterior_state = rssm.get_initial_states(1)
        prev_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        ep_rec = np.zeros((T, rec_size), dtype=np.float32)
        ep_post = np.zeros((T, num_cat, num_cls), dtype=np.float32)
        ep_actoh = np.zeros((T, action_dim), dtype=np.float32)

        for t in range(T):
            obs_jax = jnp.asarray(ep_obs[t][None], dtype=jnp.float32)  # [1, obs_dim]
            embedded = jax.vmap(world_model.encoder)(obs_jax)            # [1, dense_units]
            is_first = jnp.array([[1.0 if t == 0 else 0.0]], dtype=jnp.float32)  # [1, 1]

            key, k_rssm = jax.random.split(key)
            recurrent_state, posterior_state, _, _, _ = rssm.dynamic(
                posterior_state, recurrent_state, prev_action,
                embedded, is_first, k_rssm,
            )

            # Action at this step (convert recorded action to one-hot)
            raw_act = int(ep_act[t])
            if raw_act < 0:
                # -1 at t=0 means "no previous action" — use zero prev_action (already set)
                action_onehot = jnp.zeros((1, action_dim), dtype=jnp.float32)
            else:
                action_onehot = jax.nn.one_hot(jnp.array([raw_act]), action_dim)  # [1, action_dim]

            ep_rec[t] = np.array(recurrent_state[0])
            ep_post[t] = np.array(posterior_state[0])
            ep_actoh[t] = np.array(action_onehot[0])

            prev_action = action_onehot

        # Terminal at last step of episode
        ep_term = np.zeros(T, dtype=np.bool_)
        ep_term[-1] = True

        all_obs.append(ep_obs)
        all_rew.append(ep_rew)
        all_term.append(ep_term)
        all_rec.append(ep_rec)
        all_post.append(ep_post)
        all_act.append(ep_actoh)
        episode_terminals.extend([False] * (T - 1) + [True])
        n_episodes += 1

    obs_trace = np.concatenate(all_obs, axis=0)
    rew_trace = np.concatenate(all_rew, axis=0)
    term_trace = np.concatenate(all_term, axis=0)
    rec_trace = np.concatenate(all_rec, axis=0)
    post_trace = np.concatenate(all_post, axis=0)
    act_trace = np.concatenate(all_act, axis=0)

    return obs_trace, rew_trace, term_trace, rec_trace, post_trace, act_trace, n_episodes, episode_terminals


# ---------------------------------------------------------------------------
# Episode-end index computation (shared by both sources)
# ---------------------------------------------------------------------------
def compute_episode_end(episode_terminals: List[bool], n_steps: int) -> np.ndarray:
    """For each step t, compute the index of the terminal step of its episode.

    Mirrors the reference script's episode_end logic.
    """
    episode_end = np.zeros(n_steps, dtype=np.int32)
    current_end = n_steps  # sentinel: steps past last terminal
    for t in range(n_steps - 1, -1, -1):
        if episode_terminals[t]:
            current_end = t
        episode_end[t] = current_end
    return episode_end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)

    # --- Device setup ---
    device_str = args.device.lower()
    if "cpu" in device_str:
        # Force CPU
        try:
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
        except Exception:
            pass
    elif ":" in device_str:
        backend, index = device_str.split(":", 1)
        if backend in ("gpu", "cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = index
            try:
                jax.config.update("jax_default_device", jax.devices("cuda")[0])
            except Exception:
                pass
    elif device_str in ("gpu", "cuda"):
        try:
            jax.config.update("jax_default_device", jax.devices("cuda")[0])
        except Exception:
            pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(_REPO, "tmp"), exist_ok=True)

    # Parse horizons
    horizons = tuple(int(h) for h in args.horizons.split(","))
    h_max = args.horizon_max
    seed = args.seed
    n_steps = args.num_real_steps
    m_starts = args.num_starts

    if args.output:
        json_path = args.output
    else:
        json_path = os.path.join(_REPO, "tmp", f"{ts}_dreamer_srl_wm_diag.json")
    md_path = os.path.splitext(json_path)[0] + ".md"

    # Git hash
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_REPO
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    print(f"[srl-diag] seed={seed}  n_steps={n_steps}  M={m_starts}  H_max={h_max}")
    print(f"[srl-diag] horizons={horizons}")
    print(f"[srl-diag] git={git_hash}")

    # ── 1. Config ──────────────────────────────────────────────────────────────
    print("[srl-diag] Loading config...")
    checkpoint_arg = os.path.abspath(args.checkpoint)
    # Resolve checkpoints/ dir (the CheckpointManager's home is the parent of checkpoints/)
    if os.path.isdir(os.path.join(checkpoint_arg, "checkpoints")):
        results_dir = checkpoint_arg
        ckpt_dir = os.path.join(checkpoint_arg, "checkpoints")
    elif os.path.basename(checkpoint_arg) == "checkpoints":
        ckpt_dir = checkpoint_arg
        results_dir = os.path.dirname(checkpoint_arg)
    else:
        # Assume the user passed the results_dir directly
        results_dir = checkpoint_arg
        ckpt_dir = os.path.join(checkpoint_arg, "checkpoints")

    config = prefer_saved_config(
        results_dir, os.path.abspath(args.env_config), os.path.abspath(args.agent_config)
    )
    env_config_path = os.path.abspath(args.env_config)
    agent_config_path = os.path.abspath(args.agent_config)

    # ── 2. Env ─────────────────────────────────────────────────────────────────
    print("[srl-diag] Building environment...")
    params = load_env_params(config)
    env = ParallelEnv(params)
    obs_breakdown = get_observation_breakdown(params)
    obs_dim = sum(obs_breakdown.values())
    action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)
    print(f"[srl-diag] obs_dim={obs_dim}  action_dim={action_dim}")
    print(f"[srl-diag] Observation breakdown: {dict(obs_breakdown)}")

    # ── 3. Build agent ─────────────────────────────────────────────────────────
    print("[srl-diag] Building agent...")
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    rngs = nnx.Rngs(init_key)

    cfg_dict = config.to_dict() if hasattr(config, 'to_dict') else config

    world_model, actor, critic, target_critic = build_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=cfg_dict,
        rngs=rngs,
        # D-018: required for hierarchical checkpoints (optional kwarg; flat ignores it).
        observation_breakdown=obs_breakdown,
    )

    # ── 4. Restore checkpoint ──────────────────────────────────────────────────
    print(f"[srl-diag] Restoring checkpoint from {ckpt_dir} ...")
    manager = make_checkpoint_manager(results_dir, max_to_keep=100)

    if args.checkpoint_step is not None:
        ckpt_step = args.checkpoint_step
    else:
        ckpt_step = manager.latest_step()
        if ckpt_step is None:
            raise RuntimeError(f"No checkpoint found under {ckpt_dir}")
    print(f"[srl-diag] Loading step {ckpt_step}")

    ckpt = manager.restore(ckpt_step)
    if ckpt is None:
        raise RuntimeError(f"manager.restore({ckpt_step}) returned None")

    # Defensive normalization (copies from reference script)
    ckpt = _normalize_checkpoint(ckpt)

    nnx.update(world_model, ckpt['world_model'])
    nnx.update(actor, ckpt['actor'])
    # critic/target_critic not needed for WM diagnostic
    print(f"[srl-diag] Checkpoint restored at step={ckpt_step}")

    # Sanity: confirm reward_model output linear is non-zero (checkpoint actually loaded)
    try:
        reward_wt = nnx.state(world_model.reward_model, nnx.Param)
        leaves = jax.tree_util.tree_leaves(reward_wt)
        max_abs = max(float(jnp.abs(jnp.array(l)).max()) for l in leaves)
        print(f"[srl-diag] reward_model max|param|={max_abs:.6f} (>0 confirms non-trivial load)")
        if max_abs == 0.0:
            print("[srl-diag] WARNING: all reward_model params are 0! Checkpoint may not have loaded correctly.")
    except Exception as e:
        print(f"[srl-diag] Could not inspect reward_model params: {e}")

    # ── 5. Observation source ──────────────────────────────────────────────────
    print(f"[srl-diag] Observation source: {args.source}")

    if args.source == 'real_env':
        print(f"[srl-diag] Running eval rollout for {n_steps} steps ...")
        (obs_trace, rew_trace, term_trace, rec_trace, post_trace,
         action_trace, n_episodes, episode_terminals) = run_real_env_rollout(
            env=env, world_model=world_model, actor=actor,
            n_steps=n_steps, action_dim=action_dim, obs_dim=obs_dim,
            seed=seed, key=key,
        )
        key, key = jax.random.split(key)  # advance key after rollout
    else:
        # recordings mode
        recordings_dir = os.path.join(results_dir, "recordings")
        print(f"[srl-diag] Replaying from recordings at {recordings_dir}/{ckpt_step}/ ...")
        (obs_trace, rew_trace, term_trace, rec_trace, post_trace,
         action_trace, n_episodes, episode_terminals) = run_recordings_replay(
            recordings_dir=recordings_dir, checkpoint_step=ckpt_step,
            world_model=world_model, actor=actor,
            action_dim=action_dim, obs_dim=obs_dim, key=key,
        )
        n_steps = len(obs_trace)  # override n_steps to actual length

    print(f"[srl-diag] Source done. Episodes: {n_episodes}  Steps: {n_steps}")

    if n_episodes == 0:
        raise RuntimeError(
            f"No episodes completed in {n_steps} steps. "
            "Try --num-real-steps 4000 or check the checkpoint."
        )

    # ── 6. Sample M starting states ────────────────────────────────────────────
    episode_end = compute_episode_end(episode_terminals, n_steps)

    # Valid starts: t such that at least h_max real steps remain within SAME episode
    valid_starts = [
        t for t in range(n_steps - h_max)
        if episode_end[t] > t + h_max
    ]
    n_valid = len(valid_starts)
    print(f"[srl-diag] Valid starting states: {n_valid}  (need {m_starts})")

    if n_valid < max(m_starts // 4, 50):
        raise RuntimeError(
            f"Only {n_valid} valid starting states (< max({m_starts}//4, 50)={max(m_starts//4, 50)}). "
            f"Re-run with --num-real-steps {n_steps * 2} or reduce --horizon-max."
        )

    rng_np = np.random.default_rng(seed)
    if n_valid >= m_starts:
        chosen = rng_np.choice(valid_starts, size=m_starts, replace=False)
    else:
        print(f"[srl-diag] Warning: only {n_valid} valid starts < M={m_starts}. Using all.")
        chosen = np.array(valid_starts)
        m_starts = len(chosen)

    # ── 7. Imagination rollouts ────────────────────────────────────────────────
    print(f"[srl-diag] Running {m_starts} imagination rollouts (H_max={h_max}) ...")

    all_rew_pred = np.zeros((m_starts, h_max), dtype=np.float64)
    all_rew_true = np.zeros((m_starts, h_max), dtype=np.float64)
    all_cont_pred = np.zeros((m_starts, h_max), dtype=np.float64)
    all_obs_se = np.zeros((m_starts, h_max, obs_dim), dtype=np.float64)  # symlog SE
    first_term_steps = np.full(m_starts, h_max + 1, dtype=np.float64)

    key, imag_base_key = jax.random.split(key)

    for m_idx, t_start in enumerate(chosen):
        imag_key, imag_base_key = jax.random.split(imag_base_key)

        pred = imagine_h_steps(
            world_model=world_model,
            actor=actor,
            init_recurrent_state=rec_trace[t_start],
            init_posterior_state=post_trace[t_start],
            init_prev_action=action_trace[t_start],
            h_max=h_max,
            key=imag_key,
        )

        rew_pred = pred['reward_pred']                  # (H,)
        cont_pred = pred['cont_pred']                   # (H,)
        obs_pred_sl = pred['obs_pred_symlog']            # (H, obs_dim)

        # Ground truth: steps [t_start+1 .. t_start+h_max]
        gt_slice = slice(t_start + 1, t_start + 1 + h_max)
        rew_true = rew_trace[gt_slice].astype(np.float64)
        obs_true_raw = obs_trace[gt_slice]              # (H, obs_dim)
        obs_true_sl = np.array(symlog(jnp.array(obs_true_raw)))  # symlog

        # NaN / inf guard
        for name, arr in [('reward_pred', rew_pred), ('cont_pred', cont_pred), ('obs_pred', obs_pred_sl)]:
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(
                    f"NaN/inf in {name} at start_idx={t_start} (m_idx={m_idx}). "
                    "Hard failure per plan."
                )

        all_rew_pred[m_idx] = rew_pred
        all_rew_true[m_idx] = rew_true
        all_cont_pred[m_idx] = cont_pred
        all_obs_se[m_idx] = (obs_pred_sl.astype(np.float64) - obs_true_sl.astype(np.float64)) ** 2

        # First imagined termination: first h where cont < 0.5 (1-indexed)
        term_mask = cont_pred < 0.5
        if term_mask.any():
            first_term_steps[m_idx] = float(np.argmax(term_mask) + 1)

    print("[srl-diag] Imagination rollouts complete.")

    # ── 8. Aggregate metrics ───────────────────────────────────────────────────
    metrics_by_horizon = {}
    neg_pos_ratio_by_horizon = {}

    for h in horizons:
        h_idx = h - 1  # 0-indexed
        if h_idx >= h_max:
            continue

        rew_pred_h = all_rew_pred[:, h_idx]   # (M,)
        rew_true_h = all_rew_true[:, h_idx]   # (M,)
        cont_h = all_cont_pred[:, h_idx]       # (M,)
        obs_se_h = all_obs_se[:, h_idx, :]    # (M, obs_dim)

        ae_h = np.abs(rew_pred_h - rew_true_h)  # (M,)

        mae_total = float(np.mean(ae_h))

        pos_mask = rew_true_h > 0.01
        neg_mask = rew_true_h < -0.01
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        low_n_pos = n_pos < MIN_SAMPLES_PER_SIDE
        low_n_neg = n_neg < MIN_SAMPLES_PER_SIDE

        mae_pos = float(np.mean(ae_h[pos_mask])) if n_pos > 0 else float('nan')
        mae_neg = float(np.mean(ae_h[neg_mask])) if n_neg > 0 else float('nan')

        # neg/pos ratio — excluded from headline if low_n
        if n_pos > 0 and n_neg > 0 and not low_n_pos and not low_n_neg:
            ratio = mae_neg / (mae_pos + 1e-8)
            neg_pos_ratio_by_horizon[str(h)] = float(ratio)
        else:
            neg_pos_ratio_by_horizon[str(h)] = None

        # Continuation accuracy (all ground-truth steps are in-episode → cont=True)
        cont_acc = float(np.mean(cont_h > 0.5))

        # Obs symlog-MSE aggregate and per-channel
        agg_mse = float(np.mean(obs_se_h))
        chan_mse = {}
        start = 0
        for ch_name, ch_dim in obs_breakdown.items():
            ch_se = obs_se_h[:, start:start + ch_dim]
            chan_mse[ch_name] = float(np.mean(ch_se))
            start += ch_dim

        metrics_by_horizon[str(h)] = {
            'reward_mae_total': mae_total,
            'reward_mae_pos': mae_pos,
            'reward_mae_neg': mae_neg,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'low_n_pos': low_n_pos,
            'low_n_neg': low_n_neg,
            'cont_accuracy': cont_acc,
            'obs_symlog_mse_aggregate': agg_mse,
            'obs_symlog_mse_per_channel': chan_mse,
        }

    # First-term distribution
    fts_mean = float(np.mean(first_term_steps))
    fts_std = float(np.std(first_term_steps))
    fts_le50 = float(np.mean(first_term_steps <= 50))
    fts_never = float(np.mean(first_term_steps > h_max))

    # ── 9. JSON output ─────────────────────────────────────────────────────────
    output_json = {
        'metadata': {
            'script': 'scripts/dreamer_srl_offline_wm_test.py',
            'git_hash': git_hash,
            'timestamp': ts,
            'checkpoint_dir': ckpt_dir,
            'checkpoint_step': int(ckpt_step),
            'env_config': env_config_path,
            'agent_config': agent_config_path,
            'source': args.source,
            'seed': seed,
            'M': m_starts,
            'H_max': h_max,
            'N_real_steps': n_steps,
            'horizons': list(horizons),
            'actor_mode': 'argmax',
            'rssm_mode': 'argmax_categorical',
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'obs_breakdown': dict(obs_breakdown),
            'n_episodes': n_episodes,
            'n_valid_starts': n_valid,
        },
        'metrics_by_horizon': metrics_by_horizon,
        'neg_pos_ratio_by_horizon': neg_pos_ratio_by_horizon,
        'first_term_step': {
            'mean': fts_mean,
            'std': fts_std,
            'frac_le50': fts_le50,
            'frac_never': fts_never,
        },
    }

    os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else ".", exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"[srl-diag] JSON written to {json_path}")

    # ── 10. Markdown report ────────────────────────────────────────────────────
    h_display = [h for h in horizons if str(h) in metrics_by_horizon]

    lines = []
    lines.append("# dreamer-srl v2 Offline WM Diagnostic")
    lines.append("")
    lines.append(f"**Checkpoint**: step={ckpt_step}  source={args.source}  M={m_starts}  seed={seed}")
    lines.append(f"**Run dir**: `{results_dir}`")
    lines.append("")

    # Plain-language verdict paragraph
    h1_data = metrics_by_horizon.get("1", {})
    h_max_data = metrics_by_horizon.get(str(h_max), metrics_by_horizon.get(str(h_display[-1]), {}))
    h1_mae = h1_data.get('reward_mae_total', float('nan'))
    hN_mae = h_max_data.get('reward_mae_total', float('nan'))

    if np.isfinite(h1_mae) and np.isfinite(hN_mae) and h1_mae > 0:
        cascade_ratio = hN_mae / h1_mae
        cascade_note = (
            f"Per-horizon reward MAE grows from {h1_mae:.3f} at h=1 to {hN_mae:.3f} at h={h_display[-1]} "
            f"({cascade_ratio:.1f}× cascade). "
            f"The original-Dreamer Z2 reference showed 0.18→3.05 across h=5→50 (17×). "
        )
        if cascade_ratio > 10:
            cascade_note += "This run shows a STRONG cascade matching Z2."
        elif cascade_ratio > 3:
            cascade_note += "This run shows a MODERATE cascade — flatter than Z2."
        else:
            cascade_note += "This run shows a WEAK/FLAT cascade — significantly flatter than Z2."
    else:
        cascade_note = "Cascade ratio could not be computed (NaN in data)."

    lines.append("## Headline verdict")
    lines.append("")
    lines.append(cascade_note)
    lines.append("")

    # Per-horizon table
    lines.append("## Per-horizon reward MAE (real space)")
    lines.append("")
    header = "| h | MAE total | MAE pos | MAE neg | neg/pos ratio | cont acc | n_pos / n_neg |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|")
    for h in h_display:
        d = metrics_by_horizon[str(h)]
        ratio = neg_pos_ratio_by_horizon.get(str(h))
        ratio_str = f"{ratio:.2f}" if ratio is not None else "low_n"
        lines.append(
            f"| {h} | {d['reward_mae_total']:.4f} | "
            f"{'%.4f'%d['reward_mae_pos'] if np.isfinite(d['reward_mae_pos']) else 'nan'} | "
            f"{'%.4f'%d['reward_mae_neg'] if np.isfinite(d['reward_mae_neg']) else 'nan'} | "
            f"{ratio_str} | "
            f"{d['cont_accuracy']:.4f} | "
            f"{d['n_pos']} / {d['n_neg']} |"
        )
    lines.append("")

    # Reference comparison
    lines.append("## Pre-registered comparison: Z2 cascade (original-Dreamer)")
    lines.append("")
    lines.append("Z2 reference: h=5 MAE ≈ 0.18 → h=50 MAE ≈ 3.05 (17× compound, "
                 "neg/pos ratio not separated in Z2).")
    lines.append("")
    lines.append(f"dreamer-srl v2 (this run): {cascade_note}")
    lines.append("")

    # Obs MSE per channel
    lines.append("## Observation symlog-MSE per channel")
    lines.append("")
    h_cols = " | ".join([f"h={h}" for h in h_display])
    lines.append(f"| Channel | {h_cols} |")
    lines.append("|---|" + "---|" * len(h_display))
    for ch in obs_breakdown.keys():
        ch_vals = []
        for h in h_display:
            v = metrics_by_horizon[str(h)]['obs_symlog_mse_per_channel'].get(ch, float('nan'))
            ch_vals.append(f"{v:.4f}")
        lines.append(f"| {ch} | {' | '.join(ch_vals)} |")
    lines.append("")

    # First-term distribution
    lines.append("## First-imagined-termination step distribution")
    lines.append("")
    lines.append(f"- Mean: {fts_mean:.1f}  (real ep_len ≈ 191 for yxij4lrc winner)")
    lines.append(f"- Std: {fts_std:.1f}")
    lines.append(f"- Frac never-terminates (>H_max={h_max}): {fts_never:.2%}")
    lines.append(f"- Frac terminates within 50 steps: {fts_le50:.2%}")
    lines.append("")

    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- Checkpoint dir: `{ckpt_dir}`  step={ckpt_step}")
    lines.append(f"- Env config: `{env_config_path}`")
    lines.append(f"- Agent config: `{agent_config_path}`")
    lines.append(f"- Source: {args.source}  M={m_starts}  H_max={h_max}  seed={seed}")
    lines.append(f"- git: `{git_hash}`")
    lines.append(f"- timestamp: {ts}")

    md_text = "\n".join(lines)
    with open(md_path, 'w') as f:
        f.write(md_text)
    print(f"[srl-diag] Markdown written to {md_path}")

    # Print summary to stdout
    print("\n" + "=" * 70)
    print(f"h=1 reward_mae_total = {metrics_by_horizon.get('1', {}).get('reward_mae_total', 'N/A'):.4f}")
    print(f"  pos={metrics_by_horizon.get('1', {}).get('reward_mae_pos', float('nan')):.4f}  "
          f"neg={metrics_by_horizon.get('1', {}).get('reward_mae_neg', float('nan')):.4f}  "
          f"ratio={neg_pos_ratio_by_horizon.get('1')}")
    for h in h_display:
        d = metrics_by_horizon[str(h)]
        print(f"h={h:2d}: MAE={d['reward_mae_total']:.4f}  "
              f"pos={d['reward_mae_pos']:.4f}  neg={d['reward_mae_neg']:.4f}  "
              f"ratio={neg_pos_ratio_by_horizon.get(str(h))}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
