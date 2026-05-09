#!/usr/bin/env python
"""Offline WM-imagination diagnostic for a frozen DreamerV3 checkpoint.

Loads a saved checkpoint, rolls out the trained actor in NoPred for N_REAL_STEPS,
samples M starting states, runs the WM in imagination for H_max steps from each,
compares predicted (obs/reward/cont) vs actual at horizons {1,2,5,10,15,25,50},
and writes JSON + Markdown reports under tmp/ with a pre-registered pass/fail verdict.

Usage:
  python scripts/dreamer_offline_wm_test.py \
    --checkpoint results/JAX_DreamerV3/20260509-050606_dreamer_conv_NoPred_rr06_s0_n113 \
    --env-config configs/experiment/basic/00-5X5_NoPred.yaml \
    --agent-config configs/models/dreamer_v3_rr06.yaml \
    --num-starts 200 \
    --output tmp/20260509_wm_imagination_test_A1.json

All thresholds and horizons are pre-registered constants below (do NOT edit without
noting a design-decision change in the plan doc).

Failure-mode catalog (verbatim from plan):

| Pattern | Interpretation | Next action |
|---|---|---|
| All thresholds pass | WM is healthy on NoPred. Survival=106 is actor/value bug. | Pivot to actor/value-learning diagnostic. |
| Obs MSE high, reward MAE fine | Decoder/encoder broken but reward head learned modal dynamics. | Investigate decoder capacity, recon-loss weight, or encoder bottleneck. |
| Obs MSE fine, reward MAE high | WM predicts obs but not food-eating outcomes. Reward head bottleneck. | Investigate loss_rew weight, two-hot bin spacing. |
| Obs MSE fine, cont accuracy low | WM predicts obs but predicts spurious terminations. | Investigate cont_loss_weight, class imbalance. |
| Fine at h=5, blows up h=15/25/50 | Short-horizon sharp, autoregressive drift past training horizon. | Normal for DreamerV3 outside training horizon. |
| First-imagined-term << real survival (106) | imagined_first_term_step_mean < 50 despite real survival ~106. | Strong evidence cont head miscalibration is task-general. |
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from collections import OrderedDict

# --- Disable JAX preallocation before any JAX import ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# ─────────────────────────────────────────────────────────────────────────────
# Pre-registered constants  (do NOT change without updating the plan doc)
# ─────────────────────────────────────────────────────────────────────────────
HORIZONS = [1, 2, 5, 10, 15, 25, 50]
H_MAX = 50          # longest horizon in the scan
N_REAL_STEPS = 2000 # eval rollout length (plan §Design step 3)
M_DEFAULT = 200     # default number of imagination starting states
ENV_SEED = 42       # default PRNG seed

# Pre-registered thresholds at h=5 (plan §D4)
THRESH_AGGREGATE_SYMLOG_MSE = 0.10
THRESH_REWARD_MAE           = 0.15
THRESH_CONT_ACCURACY        = 0.95
THRESH_PER_CHANNEL = {
    "Satiation":                0.05,
    "Interoceptive Nociception":0.05,
    "Proprioception":           0.05,
    "Olfaction":                0.15,
    "Collision":                0.10,
}

# ─────────────────────────────────────────────────────────────────────────────
# Local project imports (after env var set)
# ─────────────────────────────────────────────────────────────────────────────
# Add repo root to path if needed
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import orbax.checkpoint as ocp
from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.environment.wrapper import ParallelEnv
from src.environment.sensor import get_observation_breakdown
from src.models.dreamer_v3_trainer import DreamerTrainer
from src.models.dreamer_v3_util import symlog, symexp, from_twohot


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Offline DreamerV3 WM diagnostic")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the run directory (parent of models/)  OR  path to models/ directly. "
                        "The script accepts either form.")
    p.add_argument("--env-config", required=True,
                   help="Path to environment YAML (e.g. configs/experiment/basic/00-5X5_NoPred.yaml)")
    p.add_argument("--agent-config", required=True,
                   help="Path to agent YAML (e.g. configs/models/dreamer_v3_rr06.yaml)")
    p.add_argument("--env-seed", type=int, default=ENV_SEED,
                   help=f"PRNG seed for env reset + rollout (default {ENV_SEED})")
    p.add_argument("--num-real-steps", type=int, default=N_REAL_STEPS,
                   help=f"Eval rollout length (default {N_REAL_STEPS})")
    p.add_argument("--num-starts", type=int, default=M_DEFAULT,
                   help=f"Number of imagination starting states M (default {M_DEFAULT})")
    p.add_argument("--horizon-max", type=int, default=H_MAX,
                   help=f"Maximum imagination horizon (default {H_MAX})")
    p.add_argument("--output", type=str, default=None,
                   help="Path for JSON output (default: auto-generated under tmp/)")
    p.add_argument("--device", type=str, default="gpu",
                   help="JAX device: 'gpu', 'cpu', or 'gpu:N'")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading  (mirrors train.py §1)
# ─────────────────────────────────────────────────────────────────────────────
def load_merged_config(env_config_path: str, agent_config_path: str) -> Config:
    """Load and merge configs exactly as train.py does."""
    config = get_default_config()

    # Merge train / eval / logger / vis defaults (same order as train.py)
    _root = _REPO
    for sub in ("train", "evaluation", "logger", "visualization"):
        cand = os.path.join(_root, "configs", sub, "default.yaml")
        if os.path.exists(cand):
            config.merge(Config.load_yaml(cand))

    # Merge environment / ablation config
    env_cfg = Config.load_yaml(env_config_path)
    config.merge(env_cfg)

    # Merge agent config
    agent_cfg = Config.load_yaml(agent_config_path)
    config.merge(agent_cfg)

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint restore (mirrors train.py:981–1000)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_checkpoint(d):
    """Normalize an orbax-restored checkpoint pytree for nnx.update.

    Two transformations are applied together:
    1. Convert string-digit keys to integers:
       Checkpoints saved with an older NNX version serialize nnx.Sequential
       layer indices as strings ('0', '1', ...).  Current NNX uses integer keys.
    2. Unwrap single-key {'value': array} leaf dicts:
       nnx.Variable parameters are serialized by orbax as {'value': array}.
       nnx.update() expects the actual array at each variable leaf, not the wrapper.
    """
    if isinstance(d, dict):
        # If this dict is a leaf variable wrapper {'value': array_or_leaf}, unwrap it
        if set(d.keys()) == {'value'}:
            return _normalize_checkpoint(d['value'])
        # Otherwise recurse, converting string-digit keys to int
        return {
            (int(k) if isinstance(k, str) and k.isdigit() else k): _normalize_checkpoint(v)
            for k, v in d.items()
        }
    return d


def restore_checkpoint(trainer: DreamerTrainer, checkpoint_dir: str) -> int:
    """Restore trainer weights from an orbax checkpoint.  Returns the step loaded."""
    restore_mngr = ocp.CheckpointManager(os.path.abspath(checkpoint_dir))
    step = restore_mngr.latest_step()
    if step is None:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")

    restored = restore_mngr.restore(step, args=ocp.args.PyTreeRestore())
    # Normalize: int keys + unwrap {'value': array} parameter wrappers
    restored = _normalize_checkpoint(restored)
    nnx.update(trainer.agent.wm,        restored['wm'])
    nnx.update(trainer.agent.ac.actor,  restored['actor'])
    nnx.update(trainer.agent.ac.critic, restored['critic'])
    return step


# ─────────────────────────────────────────────────────────────────────────────
# Imagination helpers
# ─────────────────────────────────────────────────────────────────────────────
def _apply_rssm_mode(prior: dict) -> dict:
    """Replace stoch with the argmax (mode) of the prior categorical.

    Plan §D5: use argmax-of-prior-logits so the rollout is deterministic.
    prior['logits'] shape: (B, stoch_dim, discrete)
    prior['stoch']  shape: (B, stoch_dim * discrete)  (flat one-hot)
    """
    logits = prior['logits']                          # (B, S, D)
    argmax = jnp.argmax(logits, axis=-1)              # (B, S)
    D = logits.shape[-1]
    S = logits.shape[-2]
    mode_stoch = jax.nn.one_hot(argmax, D)            # (B, S, D)
    batch_shape = logits.shape[:-2]                   # (B,)
    mode_stoch_flat = mode_stoch.reshape(batch_shape + (S * D,))
    return {**prior, 'stoch': mode_stoch_flat}


def run_imagination(
    trainer: DreamerTrainer,
    start_state: dict,
    real_obs_trace: np.ndarray,
    real_rew_trace: np.ndarray,
    start_idx: int,
    h_max: int,
    rng_key,
) -> dict:
    """Run one deterministic imagination rollout from start_state.

    start_state: dict with keys from RSSM.initial(), values shape (obs_dim-like,)
                 i.e. NO batch dimension.  This function adds B=1 internally.

    Returns per-step dicts with keys:
      obs_pred_symlog: (h_max, obs_dim)
      rew_pred:        (h_max,)   raw space
      cont_pred:       (h_max,)
      obs_true_symlog: (h_max, obs_dim)
      rew_true:        (h_max,)
    """
    wm = trainer.agent.wm
    act_dim = trainer.agent.ac.actor.net.layers[-1].out_features

    obs_pred_list  = []
    rew_pred_list  = []
    cont_pred_list = []

    # Work entirely with B=1 batched state throughout the loop.
    # start_state values are (dim,) arrays — add batch dim.
    prior = {k: jnp.array(v)[None] for k, v in start_state.items()}
    # prior values: (1, dim)

    for h in range(h_max):
        rng_key, step_key = jax.random.split(rng_key)

        # --- Actor: argmax on current prior features (eval mode) ---
        feat = wm.get_feat(prior)                            # (1, feat_dim)
        actor_logits = trainer.agent.ac.actor(feat)          # (1, act_dim)
        action_idx = jnp.argmax(actor_logits, axis=-1)       # (1,)
        action_onehot = jax.nn.one_hot(action_idx, act_dim)  # (1, act_dim)

        # --- Imagination step: prior → next prior ---
        prior = wm.rssm.imagine_step(prior, action_onehot, step_key)
        # prior values: (1, dim)

        # --- Apply mode (plan §D5-Option-A) ---
        # Replace stoch with argmax of prior logits so rollout is deterministic
        prior = _apply_rssm_mode(prior)

        # --- Decode predictions ---
        feat = wm.get_feat(prior)                            # (1, feat_dim)

        obs_pred_symlog = wm.decoder(feat)[0]                      # (obs_dim,)
        rew_pred_raw = float(from_twohot(wm.reward_head(feat)).ravel()[0])  # scalar
        cont_logit = wm.continue_head(feat)                        # (1, 1) or (1,)
        cont_pred = float(jax.nn.sigmoid(cont_logit.ravel()[0]))   # scalar

        obs_pred_list.append(np.array(obs_pred_symlog))
        rew_pred_list.append(float(rew_pred_raw))
        cont_pred_list.append(cont_pred)

    obs_pred_symlog_arr = np.stack(obs_pred_list, axis=0)    # (H, obs_dim)
    rew_pred_arr        = np.array(rew_pred_list)            # (H,)
    cont_pred_arr       = np.array(cont_pred_list)           # (H,)

    # Ground-truth slice: steps [start_idx+1 .. start_idx+H_max]
    gt_slice = slice(start_idx + 1, start_idx + h_max + 1)
    obs_true_raw     = real_obs_trace[gt_slice]                   # (H, obs_dim)
    obs_true_symlog  = np.array(symlog(jnp.array(obs_true_raw))) # (H, obs_dim)
    rew_true         = real_rew_trace[gt_slice]                   # (H,)

    return {
        'obs_pred_symlog': obs_pred_symlog_arr,
        'rew_pred':        rew_pred_arr,
        'cont_pred':       cont_pred_arr,
        'obs_true_symlog': obs_true_symlog,
        'rew_true':        rew_true,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel aggregation
# ─────────────────────────────────────────────────────────────────────────────
def per_channel_mse(pred: np.ndarray, true: np.ndarray, breakdown: OrderedDict) -> dict:
    """Compute mean MSE per channel over the per-horizon axis.

    pred, true: (obs_dim,)  -- single h
    Returns: {channel_name: float}
    """
    out = {}
    start = 0
    for name, dim in breakdown.items():
        p = pred[start:start + dim]
        t = true[start:start + dim]
        out[name] = float(np.mean((p - t) ** 2))
        start += dim
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # --- Device setup ---
    device_str = args.device.lower()
    if ":" in device_str:
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
    os.makedirs("tmp", exist_ok=True)

    if args.output:
        json_path = args.output
        md_path   = os.path.splitext(args.output)[0] + ".md"
    else:
        json_path = f"tmp/{ts}_dreamer_offline_wm_NoPred_A1.json"
        md_path   = f"tmp/{ts}_dreamer_offline_wm_NoPred_A1.md"

    seed     = args.env_seed
    n_steps  = args.num_real_steps
    m_starts = args.num_starts
    h_max    = args.horizon_max

    # --- Git commit ---
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_REPO
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    print(f"[offline-wm] seed={seed}  n_steps={n_steps}  M={m_starts}  H_max={h_max}")
    print(f"[offline-wm] git={git_hash}")

    # ── 1. Config ──────────────────────────────────────────────────────────────
    print("[offline-wm] Loading config...")
    env_config_path   = os.path.abspath(args.env_config)
    agent_config_path = os.path.abspath(args.agent_config)

    # If the checkpoint dir contains a saved config.yaml, use it as the base
    # (it is the exact merged config the run used).  Then the user-provided
    # --env-config / --agent-config serve as provenance documentation only.
    ckpt_arg_abs = os.path.abspath(args.checkpoint)
    if os.path.isdir(os.path.join(ckpt_arg_abs, "models")):
        _ckpt_models_dir = os.path.join(ckpt_arg_abs, "models")
    else:
        _ckpt_models_dir = ckpt_arg_abs
    saved_config_yaml = os.path.join(_ckpt_models_dir, "config.yaml")
    if os.path.exists(saved_config_yaml):
        print(f"[offline-wm] Found saved config.yaml in checkpoint dir — using it directly.")
        config = Config.load_yaml(saved_config_yaml)
    else:
        print(f"[offline-wm] No saved config.yaml found — re-merging from --env-config + --agent-config.")
        config = load_merged_config(env_config_path, agent_config_path)

    # ── 2. Env ─────────────────────────────────────────────────────────────────
    print("[offline-wm] Building environment...")
    params = load_env_params(config)
    env    = ParallelEnv(params)

    obs_breakdown = get_observation_breakdown(params)
    obs_dim       = sum(obs_breakdown.values())
    act_dim       = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)

    print(f"[offline-wm] obs_dim={obs_dim}  act_dim={act_dim}")
    print(f"[offline-wm] Observation breakdown: {dict(obs_breakdown)}")
    assert sum(obs_breakdown.values()) == obs_dim, "Breakdown dims do not sum to obs_dim!"

    # ── 3. Trainer + checkpoint ────────────────────────────────────────────────
    # Resolve checkpoint dir
    ckpt_arg = os.path.abspath(args.checkpoint)
    if os.path.isdir(os.path.join(ckpt_arg, "models")):
        checkpoint_dir = os.path.join(ckpt_arg, "models")
    else:
        checkpoint_dir = ckpt_arg  # user passed the models/ dir directly

    print(f"[offline-wm] Checkpoint dir: {checkpoint_dir}")

    dreamer_mod_config = config.get('agent.modulation')
    if dreamer_mod_config is not None and dreamer_mod_config.get('type') is None:
        dreamer_mod_config = None

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    trainer = DreamerTrainer(
        obs_dim, act_dim, config,
        rngs=nnx.Rngs(init_key),
        obs_breakdown=obs_breakdown,
        modulation_config=dreamer_mod_config,
    )

    print("[offline-wm] Restoring checkpoint...")
    ckpt_step = restore_checkpoint(trainer, checkpoint_dir)
    print(f"[offline-wm] Restored checkpoint step={ckpt_step}")
    print(f"[offline-wm] modulation_enabled={trainer.agent.wm.modulation_enabled}")

    # ── 4. Eval rollout ────────────────────────────────────────────────────────
    print(f"[offline-wm] Running eval rollout for {n_steps} steps...")

    # Allocate trace arrays
    obs_trace      = np.zeros((n_steps, obs_dim),  dtype=np.float32)
    rew_trace      = np.zeros(n_steps,              dtype=np.float32)
    terminal_trace = np.zeros(n_steps,              dtype=np.float32)

    # RSSM state fields — discover from initial()
    init_state = trainer.agent.wm.rssm.initial(1)
    state_keys = list(init_state.keys())  # ['deter', 'stoch', 'logits', 'prev_action']
    state_shapes = {k: init_state[k].shape[1:] for k in state_keys}  # drop batch dim

    # Storage for posterior states (one per step)
    post_trace = {k: np.zeros((n_steps,) + state_shapes[k], dtype=np.float32)
                  for k in state_keys}

    # Reset env (single env)
    key, env_key = jax.random.split(key)
    env_state, obs = env.reset(env_key, 1)
    obs = np.array(obs[0])  # (obs_dim,)

    # Initial Dreamer state
    dreamer_state = trainer.agent.wm.rssm.initial(1)
    dreamer_state['prev_action'] = jnp.zeros((1, act_dim))

    episode_terminals = [False] * n_steps  # track where episodes end

    for t in range(n_steps):
        obs_trace[t] = obs

        key, action_key = jax.random.split(key)
        action_idx, next_dreamer_state = trainer.get_action(
            jnp.array(obs[None]),   # (1, obs_dim)
            prev_state=dreamer_state,
            eval_mode=True,
            rng=action_key,
        )
        action_scalar = int(action_idx[0])

        # Store posterior state (next_dreamer_state IS the posterior from this obs)
        for k in state_keys:
            if k in next_dreamer_state:
                post_trace[k][t] = np.array(next_dreamer_state[k][0])

        # Step env
        action_arr = jnp.array([action_scalar])
        env_state, next_obs, reward, done, _ = env.step(env_state, action_arr)
        next_obs = np.array(next_obs[0])
        r        = float(reward[0])
        d        = bool(done[0])

        rew_trace[t]      = r
        terminal_trace[t] = float(d)
        episode_terminals[t] = d

        if d:
            # Auto-reset
            key, reset_key = jax.random.split(key)
            env_state, next_obs_reset = env.reset(reset_key, 1)
            next_obs = np.array(next_obs_reset[0])
            dreamer_state = trainer.agent.wm.rssm.initial(1)
            dreamer_state['prev_action'] = jnp.zeros((1, act_dim))
        else:
            dreamer_state = next_dreamer_state

        obs = next_obs

    n_episodes = int(sum(episode_terminals))
    print(f"[offline-wm] Rollout done. Episodes completed: {n_episodes}")

    # Sanity check: non-zero episode count
    if n_episodes == 0:
        raise RuntimeError(
            f"No episodes completed in {n_steps} steps. "
            "Try --num-real-steps 4000 or check the checkpoint."
        )

    # ── 5. Sample M starting states ────────────────────────────────────────────
    # Build per-step episode membership: assign episode_id per step
    episode_id    = np.zeros(n_steps, dtype=np.int32)
    episode_end   = np.zeros(n_steps, dtype=np.int32)  # index of terminal for this episode
    ep = 0
    ep_end = -1
    for t in range(n_steps):
        episode_id[t] = ep
        if episode_terminals[t]:
            ep_end_t = t
            # Back-fill episode_end for steps in this episode
            # (done lazily below via forward pass)
            ep += 1

    # Forward pass to set episode_end[t] = index of terminal for episode containing t
    current_end = n_steps  # sentinel
    for t in range(n_steps - 1, -1, -1):
        if episode_terminals[t]:
            current_end = t
        episode_end[t] = current_end

    # Valid starting indices: t such that t + h_max < episode_end[t]
    # i.e., we have at least h_max real steps left within the SAME episode
    valid_starts = [t for t in range(n_steps - h_max)
                    if episode_end[t] > t + h_max]

    print(f"[offline-wm] Valid starting states: {len(valid_starts)}  (need {m_starts})")

    if len(valid_starts) < max(m_starts // 4, 50):
        raise RuntimeError(
            f"Only {len(valid_starts)} valid starting states (< 50). "
            "Re-run with --num-real-steps 4000 (plan §Inconclusive re-run criterion)."
        )

    rng_np = np.random.default_rng(seed)
    if len(valid_starts) >= m_starts:
        chosen = rng_np.choice(valid_starts, size=m_starts, replace=False)
    else:
        print(f"[offline-wm] Warning: only {len(valid_starts)} valid starts < M={m_starts}. Using all.")
        chosen = np.array(valid_starts)
        m_starts = len(chosen)

    # ── 6. Imagination rollouts ────────────────────────────────────────────────
    print(f"[offline-wm] Running {m_starts} imagination rollouts (H_max={h_max})...")

    # Accumulators: per-horizon metrics
    # Shape: (M, H_max, obs_dim) for obs, (M, H_max) for rew/cont
    all_obs_symlog_se = np.zeros((m_starts, h_max, obs_dim), dtype=np.float64)
    all_rew_ae        = np.zeros((m_starts, h_max),          dtype=np.float64)
    all_cont_pred     = np.zeros((m_starts, h_max),          dtype=np.float64)
    first_term_steps  = np.full(m_starts, h_max + 1,         dtype=np.float64)

    key, imag_base_key = jax.random.split(key)

    for m_idx, t_start in enumerate(chosen):
        # Build start state without batch dim — run_imagination adds B=1 internally
        start_state = {k: np.array(post_trace[k][t_start]) for k in state_keys}

        rng_key, imag_key = jax.random.split(imag_base_key)
        imag_base_key = rng_key  # advance for next iteration

        result = run_imagination(
            trainer=trainer,
            start_state=start_state,
            real_obs_trace=obs_trace,
            real_rew_trace=rew_trace,
            start_idx=t_start,
            h_max=h_max,
            rng_key=imag_key,
        )

        obs_pred  = result['obs_pred_symlog']  # (H, obs_dim)
        obs_true  = result['obs_true_symlog']  # (H, obs_dim)
        rew_pred  = result['rew_pred']          # (H,)
        rew_true  = result['rew_true']          # (H,)
        cont_pred = result['cont_pred']          # (H,)

        # Check NaN/inf
        for arr_name, arr in [('obs_pred', obs_pred), ('rew_pred', rew_pred), ('cont_pred', cont_pred)]:
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(
                    f"NaN/inf detected in {arr_name} at start_idx={t_start}. "
                    "Hard failure per plan §Inconclusive criteria."
                )

        all_obs_symlog_se[m_idx] = (obs_pred - obs_true) ** 2   # squared error
        all_rew_ae[m_idx]        = np.abs(rew_pred - rew_true)
        all_cont_pred[m_idx]     = cont_pred

        # First-imagined-termination step: first h where cont < 0.5 (1-indexed)
        term_mask = cont_pred < 0.5
        if term_mask.any():
            first_term_steps[m_idx] = float(np.argmax(term_mask) + 1)

    print("[offline-wm] Imagination rollouts complete.")

    # ── 7. Aggregate metrics ───────────────────────────────────────────────────
    # Per-channel symlog-MSE at each horizon h (mean over M and over dims within channel)
    # horizons list: [1, 2, 5, 10, 15, 25, 50] → indices [0, 1, 4, 9, 14, 24, 49]

    horizon_indices = {h: h - 1 for h in HORIZONS}   # h→0-indexed position in h_max dim

    def aggregate_at_h(h_idx: int):
        """Return (aggregate_mse, per_channel_mse, rew_mae, cont_acc) at horizon index h_idx."""
        obs_se_h = all_obs_symlog_se[:, h_idx, :]   # (M, obs_dim)
        rew_ae_h = all_rew_ae[:, h_idx]             # (M,)
        cont_h   = all_cont_pred[:, h_idx]          # (M,)

        # Aggregate MSE: mean over M and obs_dim
        aggregate_mse = float(np.mean(obs_se_h))

        # Per-channel MSE
        chan_mse = {}
        start = 0
        for name, dim in obs_breakdown.items():
            chan_se = obs_se_h[:, start:start + dim]  # (M, dim)
            chan_mse[name] = float(np.mean(chan_se))
            start += dim

        rew_mae  = float(np.mean(rew_ae_h))
        # Continuation accuracy: all ground-truth cont=1 (in-episode by construction)
        cont_acc = float(np.mean(cont_h > 0.5))

        return aggregate_mse, chan_mse, rew_mae, cont_acc

    metrics_by_horizon = {}
    for h in HORIZONS:
        h_idx = horizon_indices[h]
        if h_idx >= h_max:
            continue
        agg_mse, chan_mse, rew_mae, cont_acc = aggregate_at_h(h_idx)
        metrics_by_horizon[h] = {
            'aggregate_symlog_mse': agg_mse,
            'per_channel_symlog_mse': chan_mse,
            'reward_mae': rew_mae,
            'cont_accuracy': cont_acc,
            'cont_bce': float(-np.mean(np.log(np.clip(all_cont_pred[:, h_idx], 1e-8, 1.0)))),
        }

    # First-imagined-termination distribution
    fts_mean  = float(np.mean(first_term_steps))
    fts_std   = float(np.std(first_term_steps))
    fts_lt50  = float(np.mean(first_term_steps <= 50))
    fts_never = float(np.mean(first_term_steps > h_max))

    # ── 8. Verdict (h=5) ──────────────────────────────────────────────────────
    if 5 not in metrics_by_horizon:
        raise RuntimeError("h=5 not in computed horizons — check HORIZONS constant.")

    h5 = metrics_by_horizon[5]
    fails = []

    if h5['aggregate_symlog_mse'] >= THRESH_AGGREGATE_SYMLOG_MSE:
        fails.append(f"aggregate_symlog_mse={h5['aggregate_symlog_mse']:.4f} >= {THRESH_AGGREGATE_SYMLOG_MSE}")
    if h5['reward_mae'] >= THRESH_REWARD_MAE:
        fails.append(f"reward_mae={h5['reward_mae']:.4f} >= {THRESH_REWARD_MAE}")
    if h5['cont_accuracy'] <= THRESH_CONT_ACCURACY:
        fails.append(f"cont_accuracy={h5['cont_accuracy']:.4f} <= {THRESH_CONT_ACCURACY}")

    for ch_name, thresh in THRESH_PER_CHANNEL.items():
        ch_mse = h5['per_channel_symlog_mse'].get(ch_name)
        if ch_mse is not None and ch_mse >= thresh:
            fails.append(f"{ch_name} symlog_mse={ch_mse:.4f} >= {thresh}")

    wm_verdict = "WM BROKEN" if fails else "WM WORKING"
    if wm_verdict == "WM WORKING":
        verdict_sentence = (
            "The world model predicts NoPred correctly at h=5. "
            "Survival=106 is an actor/value-learning failure, not a WM failure."
        )
    else:
        verdict_sentence = (
            "World model FAILS pre-registered thresholds at h=5. "
            "Failures: " + "; ".join(fails) + ". "
            "WM is the primary blockage — downstream actor/value debugging is premature."
        )

    # Long-horizon stretch check
    if 5 in metrics_by_horizon and 50 in metrics_by_horizon:
        h5_agg  = metrics_by_horizon[5]['aggregate_symlog_mse']
        h50_agg = metrics_by_horizon[50]['aggregate_symlog_mse']
        if h5_agg > 0:
            ratio = h50_agg / h5_agg
        else:
            ratio = float('inf')
        if ratio <= 2.0:
            long_horizon_note = f"Long-horizon healthy: h50/h5 ratio={ratio:.2f} (<=2.0 — WM healthy at 3x training horizon)."
        elif ratio <= 5.0:
            long_horizon_note = f"Moderate autoregressive drift: h50/h5 ratio={ratio:.2f} (2<r<=5 — normal-ish)."
        else:
            long_horizon_note = f"Autoregressive drift dominates past training horizon: h50/h5 ratio={ratio:.2f} (>5)."
    else:
        long_horizon_note = "h=50 not computed — cannot assess long-horizon drift."

    # ── 9. JSON output ─────────────────────────────────────────────────────────
    output_json = {
        'metadata': {
            'script':          'scripts/dreamer_offline_wm_test.py',
            'git_hash':        git_hash,
            'timestamp':       ts,
            'checkpoint_dir':  checkpoint_dir,
            'checkpoint_step': ckpt_step,
            'env_config':      env_config_path,
            'agent_config':    agent_config_path,
            'seed':            seed,
            'M':               m_starts,
            'H_max':           h_max,
            'N_real_steps':    n_steps,
            'horizons':        HORIZONS,
            'actor_mode':      'argmax',
            'rssm_mode':       'argmax_categorical',
            'obs_dim':         obs_dim,
            'act_dim':         act_dim,
            'obs_breakdown':   dict(obs_breakdown),
            'n_episodes':      n_episodes,
            'n_valid_starts':  len(valid_starts),
        },
        'metrics_by_horizon': {str(h): v for h, v in metrics_by_horizon.items()},
        'first_term_step': {
            'mean':        fts_mean,
            'std':         fts_std,
            'frac_le50':   fts_lt50,
            'frac_never':  fts_never,
            'histogram':   np.histogram(first_term_steps, bins=10)[0].tolist(),
        },
        'verdict': {
            'h5_verdict':        wm_verdict,
            'failures':          fails,
            'verdict_sentence':  verdict_sentence,
            'long_horizon_note': long_horizon_note,
            'thresholds': {
                'aggregate_symlog_mse': THRESH_AGGREGATE_SYMLOG_MSE,
                'reward_mae':           THRESH_REWARD_MAE,
                'cont_accuracy':        THRESH_CONT_ACCURACY,
                'per_channel':          THRESH_PER_CHANNEL,
            },
        },
    }

    json_str = json.dumps(output_json, indent=2)
    with open(json_path, 'w') as f:
        f.write(json_str)
    print(f"[offline-wm] JSON written to {json_path}")

    # ── 10. Markdown report ────────────────────────────────────────────────────
    def fmt(val, decimals=4):
        return f"{val:.{decimals}f}"

    def pass_fail(val, thresh, direction='lt'):
        """direction: 'lt' means pass if val < thresh; 'gt' means pass if val > thresh."""
        if direction == 'lt':
            ok = val < thresh
        else:
            ok = val > thresh
        return "PASS" if ok else "FAIL"

    h_display = [h for h in HORIZONS if h in metrics_by_horizon]

    lines = []
    lines.append(f"# DreamerV3 Offline WM Diagnostic — NoPred Cell A1")
    lines.append(f"")
    lines.append(f"**Verdict: {wm_verdict}**")
    lines.append(f"")
    lines.append(f"> {verdict_sentence}")
    lines.append(f"")
    lines.append(f"{long_horizon_note}")
    lines.append(f"")
    lines.append(f"**Run info**: checkpoint step={ckpt_step}, M={m_starts}, seed={seed}, "
                 f"n_steps={n_steps}, n_episodes={n_episodes}, n_valid_starts={len(valid_starts)}")
    lines.append(f"")
    lines.append(f"**Obs breakdown**: {dict(obs_breakdown)}")
    lines.append(f"")

    # Per-channel obs MSE table
    lines.append(f"## Observation symlog-MSE per channel per horizon")
    lines.append(f"")
    h_cols = " | ".join([f"h={h}" for h in h_display])
    h5_thresh_col = "Thresh@h=5 | Pass?"
    lines.append(f"| Channel | {h_cols} | {h5_thresh_col} |")
    lines.append(f"|---|" + "---|" * (len(h_display)) + "---|---|")

    all_channels = list(obs_breakdown.keys())
    for ch in all_channels:
        ch_vals = [fmt(metrics_by_horizon[h]['per_channel_symlog_mse'].get(ch, float('nan')))
                   for h in h_display]
        thresh = THRESH_PER_CHANNEL.get(ch)
        if thresh is not None:
            pf = pass_fail(metrics_by_horizon[5]['per_channel_symlog_mse'].get(ch, 999), thresh, 'lt')
            thresh_str = f"< {thresh}"
        else:
            pf = "n/a"
            thresh_str = "n/a"
        row_vals = " | ".join(ch_vals)
        lines.append(f"| {ch} | {row_vals} | {thresh_str} | {pf} |")

    # Aggregate row
    agg_vals = [fmt(metrics_by_horizon[h]['aggregate_symlog_mse']) for h in h_display]
    agg_pf = pass_fail(metrics_by_horizon[5]['aggregate_symlog_mse'], THRESH_AGGREGATE_SYMLOG_MSE, 'lt')
    lines.append(f"| **Aggregate** | {' | '.join(agg_vals)} | < {THRESH_AGGREGATE_SYMLOG_MSE} | {agg_pf} |")
    lines.append(f"")

    # Reward MAE table
    lines.append(f"## Reward MAE (raw space) per horizon")
    lines.append(f"")
    lines.append(f"| Metric | {h_cols} | Thresh@h=5 | Pass? |")
    lines.append(f"|---|" + "---|" * len(h_display) + "---|---|")
    rew_vals = [fmt(metrics_by_horizon[h]['reward_mae']) for h in h_display]
    rew_pf = pass_fail(metrics_by_horizon[5]['reward_mae'], THRESH_REWARD_MAE, 'lt')
    lines.append(f"| Reward MAE | {' | '.join(rew_vals)} | < {THRESH_REWARD_MAE} | {rew_pf} |")
    lines.append(f"")

    # Continuation accuracy table
    lines.append(f"## Continuation accuracy per horizon")
    lines.append(f"")
    lines.append(f"| Metric | {h_cols} | Thresh@h=5 | Pass? |")
    lines.append(f"|---|" + "---|" * len(h_display) + "---|---|")
    cont_vals = [fmt(metrics_by_horizon[h]['cont_accuracy']) for h in h_display]
    cont_pf = pass_fail(metrics_by_horizon[5]['cont_accuracy'], THRESH_CONT_ACCURACY, 'gt')
    lines.append(f"| Cont accuracy | {' | '.join(cont_vals)} | > {THRESH_CONT_ACCURACY} | {cont_pf} |")
    lines.append(f"")

    # First-term-step distribution
    lines.append(f"## First-imagined-termination step distribution")
    lines.append(f"")
    lines.append(f"- Mean: {fts_mean:.1f}  (real survival mean ~106)")
    lines.append(f"- Std: {fts_std:.1f}")
    lines.append(f"- Frac never-terminates (>H_max={h_max}): {fts_never:.2%}")
    lines.append(f"- Frac terminates within 50 steps: {fts_lt50:.2%}")
    lines.append(f"")

    lines.append(f"## Metadata")
    lines.append(f"")
    lines.append(f"- Checkpoint: `{checkpoint_dir}` step={ckpt_step}")
    lines.append(f"- Env config: `{env_config_path}`")
    lines.append(f"- Agent config: `{agent_config_path}`")
    lines.append(f"- git: `{git_hash}`")
    lines.append(f"- modulation_enabled: {trainer.agent.wm.modulation_enabled}")

    md_text = "\n".join(lines)
    with open(md_path, 'w') as f:
        f.write(md_text)
    print(f"[offline-wm] Markdown written to {md_path}")

    # Print to stdout
    print("\n" + "=" * 70)
    print(md_text)
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
