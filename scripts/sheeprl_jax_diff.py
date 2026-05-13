"""sheeprl_jax_diff.py — per-function numerical diff tool for the dreamer-srl v3 rebuild.

Purpose
-------
Runs both the JAX (dreamer-srl) and PyTorch (vendored sheeprl@33b6366) implementations
of a named function on a shared `.npz` fixture and reports the max-absolute-difference.
Exits 0 on PASS, 1 on FAIL — so CI can gate on it.

Usage
-----
    # Single function
    python scripts/sheeprl_jax_diff.py \\
        --function twohot_encode \\
        --fixture tests/fixtures/dreamer_srl/twohot_encode_input.npz \\
        --threshold 1e-6

    # Whole checkpoint (runs all functions registered for CP<N>)
    python scripts/sheeprl_jax_diff.py --checkpoint CP5

Output (PASS)
    sheeprl: vendor/sheeprl/sheeprl/utils/distribution.py:L185-L260
    jax:     src/algorithms/dreamer_srl/loss.py:TwoHotEncoding
    fixture: shape=(16, 4, 255) logits + (16, 4, 1) target, seed=0xD3EAF
    max_abs_diff = 1.2e-7
    PASS  (< 1.0e-6 threshold)

Output (FAIL)
    max_abs_diff = 3.4e-3
    FAIL  (>= 1.0e-6 threshold)

Extending per checkpoint
------------------------
At each checkpoint (CP1–CP10) the developer adds a new entry to
FUNCTION_REGISTRY mapping the function name to a Callable that:
  1. Loads inputs from the fixture (passed as a numpy NpzFile).
  2. Runs the PyTorch (sheeprl) side and the JAX (dreamer-srl) side.
  3. Returns (jax_out: jnp.ndarray, torch_out: np.ndarray, metadata: str)
     where metadata is a multi-line string assembled by the runner that should
     include the sheeprl source location, the JAX function location, and the
     fixture shape/seed description — matching the intended output format::

       sheeprl: vendor/sheeprl/sheeprl/utils/distribution.py:L185-L260
       jax:     src/algorithms/dreamer_srl/loss.py:TwoHotEncoding
       fixture: shape=(16, 4, 255) logits + (16, 4, 1) target, seed=0xD3EAF

     compare() then prints ``fixture: <metadata>`` followed by max_abs_diff and
     PASS/FAIL.  The runner is responsible for assembling the header lines
     (sheeprl: ..., jax: ...) so compare() itself stays simple.

The compare() helper below converts torch → JAX, computes max-abs-diff,
and formats the PASS/FAIL message — comparison runners do NOT call
compare() themselves; the dispatcher calls it after the runner returns.

PRNG seed convention
--------------------
All fixtures use seed 0xD3EAF (decimal 868591).  The per-CP fixture
generation scripts live at scripts/fixtures/gen_<cp>_fixtures.py.
"""

import argparse
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — guarded so the skeleton does not crash if neither
# torch nor jax is importable (though in practice both should be available
# in the grid_world_pain conda env).
# ---------------------------------------------------------------------------
try:
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

try:
    import torch  # noqa: F401  (used by comparison runners)
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------

def load_fixture(path: str) -> np.lib.npyio.NpzFile:
    """Load a .npz fixture file.  Returns the NpzFile object (dict-like)."""
    if not os.path.exists(path):
        print(f"ERROR: fixture not found: {path}", file=sys.stderr)
        sys.exit(1)
    return np.load(path)


# ---------------------------------------------------------------------------
# compare() helper
# ---------------------------------------------------------------------------

def compare(
    jax_out,
    torch_out,
    threshold: float,
    metadata: str = "",
) -> tuple[float, bool]:
    """Compute max-abs-diff between JAX and PyTorch outputs and print PASS/FAIL.

    Parameters
    ----------
    jax_out   : jnp.ndarray or np.ndarray  (JAX side output)
    torch_out : np.ndarray or torch.Tensor  (sheeprl PyTorch side output)
    threshold : float                       (default 1e-6; overridden by --threshold)
    metadata  : str                         (shape / fixture description for the header)

    Returns
    -------
    (max_abs_diff: float, passed: bool)

    Side-effect: prints the diff + PASS/FAIL line to stdout.
    """
    # Convert torch tensor → numpy if needed
    if _TORCH_AVAILABLE:
        import torch as _torch
        if isinstance(torch_out, _torch.Tensor):
            torch_out = torch_out.detach().numpy()

    # Convert JAX array → numpy (np.asarray handles both JAX arrays and numpy arrays)
    jax_out_np = np.asarray(jax_out)

    torch_out_np = np.asarray(torch_out)

    # Explicit shape check — a numpy broadcast error here is opaque; name both sides.
    if jax_out_np.shape != torch_out_np.shape:
        raise ValueError(
            f"Shape mismatch: jax={jax_out_np.shape} torch={torch_out_np.shape}. "
            f"This is a structural deviation, not a numerical one — "
            f"log it in DEVIATION_LOG.md with the sheeprl source line."
        )

    max_abs_diff = float(np.max(np.abs(jax_out_np - torch_out_np)))
    passed = max_abs_diff < threshold

    if metadata:
        print(f"  fixture: {metadata}")
    print(f"  max_abs_diff = {max_abs_diff:.3e}")
    if passed:
        print(f"  PASS  (< {threshold:.1e} threshold)")
    else:
        print(f"  FAIL  (>= {threshold:.1e} threshold)")

    return max_abs_diff, passed


# ---------------------------------------------------------------------------
# FUNCTION_REGISTRY
# ---------------------------------------------------------------------------
# Maps function name → Callable[[np.lib.npyio.NpzFile], tuple[any, any, str]]
#
# Each entry returns (jax_out, torch_out, metadata_str).
# The dispatcher calls compare(jax_out, torch_out, threshold, metadata_str).
#
# CP1 (utils.py):       symlog, symexp, init_weights, uniform_init_weights,
#                        compute_lambda_values, moments_update, ratio, prepare_obs
# CP2 (agent.py):        layernorm_gru_cell
# CP2b (train.py):       action_shift
# CP3 (agent.py):        zero_init_reward_head, zero_init_critic_head
# CP3b (buffers.py + train.py): buffer_storage_state_after_deterministic_adds,
#                        buffer_sample_at_indices_matches_sheeprl,
#                        buffer_is_first_marker_placement_in_straddling_window,
#                        buffer_parallel_env_lane_non_interference,
#                        cadence_yaml_key_parity_with_sheeprl_xs,
#                        cadence_env_grad_step_trace_5000_iters
#                        (State-evolution bit-identity, not pure-function;
#                        D-004 memmap omission + D-005 unfilled-region exclusion pre-declared.)
# CP4 (agent.py):        rssm_transition, rssm_representation, get_initial_states
# CP4b (agent.py):       is_first_force_set, is_first_three_quantity_reset
# CP5 (loss.py):         twohot_bins_endpoints, twohot_encode, twohot_log_prob
# CP6 (train.py):        critic_loss_two_terms, critic_target_lambda, discount_weighting
# CP7 (train.py):        polyak_first_call, polyak_subsequent_call, polyak_before_train
# CP8 (end-to-end):      forward_parity (offline script, not registered here)
#
# Pre-CP0: registry is empty — only the CLI skeleton ships here.
# Populate incrementally as each CP lands.

# ---------------------------------------------------------------------------
# CP1 runners (utils.py) — added when CP1 landed
# ---------------------------------------------------------------------------

def _run_symlog(fixture) -> tuple:
    """symlog: sign(x)*log(|x|+1) — deterministic, expect max_abs_diff < 1e-6."""
    import jax.numpy as jnp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.utils import symlog
    x_np = fixture["x"]
    torch_out = fixture["torch_out"]
    jax_out = symlog(jnp.asarray(x_np))
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/utils/utils.py:L148-L149\n"
        f"  jax:     src/algorithms/dreamer_srl/utils.py:symlog\n"
        f"  fixture: x.shape={x_np.shape}, seed=0xD3EAF"
    )
    return jax_out, torch_out, metadata


def _run_symexp(fixture) -> tuple:
    """symexp: sign(x)*(exp(|x|)-1) — DEVIATION D-003: float32 ULP, threshold 2e-5.

    The diff tool reports FAIL if > 1e-6 (default threshold). Use --threshold 2e-5
    for symexp to match the D-003 approved relaxation. The diff tool call below
    uses the function-specific threshold when invoked via --checkpoint CP1.
    """
    import jax.numpy as jnp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.utils import symexp
    x_np = fixture["x"]
    torch_out = fixture["torch_out"]
    jax_out = symexp(jnp.asarray(x_np))
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/utils/utils.py:L152-L153\n"
        f"  jax:     src/algorithms/dreamer_srl/utils.py:symexp\n"
        f"  fixture: x.shape={x_np.shape}, seed=0xD3EAF\n"
        f"  NOTE: D-003 — float32 GPU exp ULP difference; threshold relaxed to 2e-5"
    )
    # Return a special marker so the dispatcher uses the D-003 threshold
    return jax_out, torch_out, metadata


def _run_init_weights(fixture) -> tuple:
    """init_weights: Hafner truncated-normal — DEVIATION D-002: distribution test only.

    F2 tighten (CP1): fixture out_features=16384 (N=1.24M elements); threshold
    tightened from 15% to 1%. Comparison is against std_target = std_theoretical *
    HAFNER_CONST (the actual expected std after ±2-sigma truncation), not against
    the inflated input-std (std_theoretical). The 12% gap in earlier tests was not
    a bug — it was comparing sampled-std vs inflated-input-std.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.utils import init_weights
    # Precise Hafner constant — must match src/algorithms/dreamer_srl/utils.py
    HAFNER_CONST = 0.87962566103423978
    in_features = int(fixture["in_features"])
    out_features = int(fixture["out_features"])
    std_theoretical = float(fixture["std_theoretical"])
    # std_target is the actual expected std of the sampled distribution (after truncation)
    std_target = std_theoretical * HAFNER_CONST
    key = jax.random.PRNGKey(int(fixture["jax_seed"]))
    jax_kernel = init_weights(in_features, out_features, key)
    jax_std = float(jnp.std(jax_kernel))
    rel_err = abs(jax_std - std_target) / std_target
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L143-L166\n"
        f"  jax:     src/algorithms/dreamer_srl/utils.py:init_weights\n"
        f"  fixture: in={in_features}, out={out_features}, std_theoretical={std_theoretical:.6f}\n"
        f"  NOTE: D-002 — stochastic, different RNG; testing std rel-err={rel_err:.4f} < 0.01 "
        f"(vs std_target={std_target:.6f} = std_theoretical*HAFNER_CONST)"
    )
    # F2: threshold tightened to 1% (was 15%). N=16384 → SE of std ≈ 0.06%, so
    # a 1% bound reliably catches the historical 0.8796 truncation bug class.
    if rel_err >= 0.01:
        # Force fail by returning a large diff
        return np.array([rel_err]), np.array([0.0]), metadata
    # Pass: return identical scalars so compare() sees 0 diff
    return np.array([0.0]), np.array([0.0]), metadata


def _run_uniform_init_weights(fixture) -> tuple:
    """uniform_init_weights: uniform kernel — DEVIATION D-002: distribution test only."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.utils import uniform_init_weights
    in_features = int(fixture["in_features"])
    out_features = int(fixture["out_features"])
    given_scale = float(fixture["given_scale"])
    limit = float(fixture["limit"])
    key = jax.random.PRNGKey(int(fixture["jax_seed"]))
    jax_kernel = uniform_init_weights(given_scale, in_features, out_features, key)
    max_abs = float(jnp.max(jnp.abs(jax_kernel)))
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L170-L186\n"
        f"  jax:     src/algorithms/dreamer_srl/utils.py:uniform_init_weights\n"
        f"  fixture: in={in_features}, out={out_features}, scale={given_scale}, limit={limit:.6f}\n"
        f"  NOTE: D-002 — stochastic; testing max_abs_val={max_abs:.6f} <= limit={limit:.6f}"
    )
    if max_abs > limit + 1e-5:
        return np.array([max_abs - limit]), np.array([0.0]), metadata
    return np.array([0.0]), np.array([0.0]), metadata


def _run_compute_lambda_values(fixture) -> tuple:
    """compute_lambda_values: λ-return recursion — deterministic, expect < 1e-6."""
    import jax.numpy as jnp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.utils import compute_lambda_values
    rewards = fixture["rewards"]
    values = fixture["values"]
    continues = fixture["continues"]
    lmbda = float(fixture["lmbda"])
    torch_out = fixture["torch_out"]
    jax_out = compute_lambda_values(
        jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(continues), lmbda=lmbda
    )
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L66-L77\n"
        f"  jax:     src/algorithms/dreamer_srl/utils.py:compute_lambda_values\n"
        f"  fixture: rewards/values/continues shape={rewards.shape}, lmbda={lmbda}, seed=0xD3EAF"
    )
    return jax_out, torch_out, metadata


def _run_moments_update(fixture) -> tuple:
    """moments_update: percentile EMA — deterministic, expect < 1e-6 (DEVIATION D-001: no all_gather)."""
    import jax.numpy as jnp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.utils import moments_init, moments_update
    x_np = fixture["x"]
    decay = float(fixture["decay"])
    max_ = float(fixture["max_"])
    percentile_low = float(fixture["percentile_low"])
    percentile_high = float(fixture["percentile_high"])
    torch_offset = fixture["torch_out_offset"]
    torch_invscale = fixture["torch_out_invscale"]
    state = moments_init(decay=decay, max_=max_,
                         percentile_low=percentile_low, percentile_high=percentile_high)
    _, jax_offset, jax_invscale = moments_update(
        state, jnp.asarray(x_np),
        decay=decay, max_=max_,
        percentile_low=percentile_low, percentile_high=percentile_high,
    )
    import numpy as np
    jax_out = np.array([float(jax_offset), float(jax_invscale)])
    torch_out_arr = np.array([float(torch_offset), float(torch_invscale)])
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L40-L63\n"
        f"  jax:     src/algorithms/dreamer_srl/utils.py:moments_update\n"
        f"  fixture: x.shape={x_np.shape}, seed=0xD3EAF\n"
        f"  NOTE: D-001 — fabric.all_gather omitted (single-process; all_gather is identity)"
    )
    return jax_out, torch_out_arr, metadata


def _run_ratio(fixture) -> tuple:
    """Ratio.__call__: replay-ratio scheduler — integer-exact, no float threshold needed."""
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.utils import Ratio
    ratio_val = float(fixture["ratio"])
    steps = fixture["steps"].tolist()
    torch_repeats = fixture["torch_out_repeats"].tolist()
    jax_ratio = Ratio(ratio_val)
    jax_repeats = [jax_ratio(s) for s in steps]
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/utils/utils.py:L259-L301\n"
        f"  jax:     src/algorithms/dreamer_srl/utils.py:Ratio\n"
        f"  fixture: ratio={ratio_val}, steps={steps}, seed=0xD3EAF"
    )
    return np.array(jax_repeats, dtype=np.float32), np.array(torch_repeats, dtype=np.float32), metadata


def _run_prepare_obs(fixture) -> tuple:
    """prepare_obs: obs-dict reshape — deterministic, expect < 1e-6."""
    import jax.numpy as jnp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.utils import prepare_obs
    obs_state = fixture["obs_state"]
    num_envs = int(fixture["num_envs"])
    torch_out = fixture["torch_out_state"]
    obs = {"state": obs_state}
    jax_obs = prepare_obs(obs, num_envs=num_envs)
    jax_out = jax_obs["state"]
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L80-L91\n"
        f"  jax:     src/algorithms/dreamer_srl/utils.py:prepare_obs\n"
        f"  fixture: obs_state.shape={obs_state.shape}, num_envs={num_envs}, seed=0xD3EAF"
    )
    return jax_out, torch_out, metadata


# ---------------------------------------------------------------------------
# CP3b runners (buffers.py + cadence wiring) — added when CP3b landed
# ---------------------------------------------------------------------------

def _run_buffer_storage_state_after_deterministic_adds(fixture) -> tuple:
    """State-evolution test: drive JAX buffer with identical add() sequence.

    Compares the filled region [:_pos] of both buffers' stored arrays.
    Unfilled region [_pos:] is excluded (np.empty — undefined memory).
    DEVIATION D-005: unfilled-region exclusion.

    Sheeprl source: vendor/sheeprl/sheeprl/data/buffers.py:L145-L221
    """
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer

    buffer_size = int(fixture["buffer_size"])
    n_envs = int(fixture["n_envs"])
    n_steps = int(fixture["n_steps"])
    expected_pos = int(fixture["sheeprl_pos"])
    step_keys = ["observations", "actions", "rewards", "terminated", "truncated", "is_first"]

    # Reconstruct add() sequence
    jax_rb = SequentialReplayBuffer(buffer_size=buffer_size, n_envs=n_envs)
    for i in range(n_steps):
        step = {}
        for k in step_keys:
            arr = fixture[f"step_{k}"]
            step[k] = arr[i]
        jax_rb.add(step)

    # Concatenate all keys' filled regions for comparison
    jax_parts = []
    sheeprl_parts = []
    for k in step_keys:
        jax_parts.append(jax_rb._buf[k][:expected_pos].ravel())
        sheeprl_parts.append(np.asarray(fixture[f"buf_{k}"])[:expected_pos].ravel())

    jax_out = np.concatenate(jax_parts)
    torch_out = np.concatenate(sheeprl_parts)
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/data/buffers.py:L145-L221\n"
        f"  jax:     src/algorithms/dreamer_srl/buffers.py:SequentialReplayBuffer.add\n"
        f"  fixture: buffer_size={buffer_size}, n_envs={n_envs}, n_steps={n_steps}, "
        f"seed=0xD3EAF\n"
        f"  NOTE: D-005 — comparing filled region [:_pos={expected_pos}] only; "
        f"unfilled [_pos:] excluded (np.empty uninitialized memory)"
    )
    return jax_out, torch_out, metadata


def _run_buffer_sample_at_indices_matches_sheeprl(fixture) -> tuple:
    """State-evolution test: _sample_at_indices with pre-computed indices.

    Bypasses PRNG — pre-computed start_idxes and env_idxes from sheeprl's RNG.
    Compares _get_samples output byte-for-byte.

    Sheeprl source: vendor/sheeprl/sheeprl/data/buffers.py:L467-L526
    """
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer

    buffer_size = int(fixture["buffer_size"])
    n_envs = int(fixture["n_envs"])
    n_steps = int(fixture["n_steps"])
    sequence_length = int(fixture["sequence_length"])
    batch_size = int(fixture["batch_size"])
    n_samples = int(fixture["n_samples"])
    precomputed_start_idxes = fixture["precomputed_start_idxes"].astype(np.intp)
    precomputed_env_idxes = fixture["precomputed_env_idxes"].astype(np.intp)
    step_keys = ["observations", "actions", "rewards", "terminated", "truncated", "is_first"]

    jax_rb = SequentialReplayBuffer(buffer_size=buffer_size, n_envs=n_envs)
    for i in range(n_steps):
        step = {}
        for k in step_keys:
            step[k] = fixture[f"step_{k}"][i]
        jax_rb.add(step)

    jax_samples = jax_rb._sample_at_indices(
        precomputed_start_idxes=precomputed_start_idxes,
        env_idxes=precomputed_env_idxes,
        sequence_length=sequence_length,
        batch_size=batch_size,
        n_samples=n_samples,
    )

    jax_parts = []
    sheeprl_parts = []
    for k in step_keys:
        jax_parts.append(jax_samples[k].ravel())
        sheeprl_parts.append(np.asarray(fixture[f"sheeprl_sample_{k}"]).ravel())

    jax_out = np.concatenate(jax_parts)
    torch_out = np.concatenate(sheeprl_parts)
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/data/buffers.py:L467-L526\n"
        f"  jax:     src/algorithms/dreamer_srl/buffers.py:SequentialReplayBuffer._sample_at_indices\n"
        f"  fixture: buffer_size={buffer_size}, n_envs={n_envs}, "
        f"seq_len={sequence_length}, batch_size={batch_size}, seed=0xD3EAF\n"
        f"  NOTE: D-002 class — PRNG bypassed; indices pre-computed from sheeprl's RNG"
    )
    return jax_out, torch_out, metadata


def _run_buffer_is_first_marker_placement_in_straddling_window(fixture) -> tuple:
    """State-evolution test: is_first=1 lands at correct offset in straddling window.

    Buffer has done at step DONE_AT, is_first=1 at DONE_AT+1.
    Window starting at START_IDX straddles the boundary.
    Compares JAX vs sheeprl _get_samples output for is_first and all keys.

    Sheeprl source: vendor/sheeprl/sheeprl/data/buffers.py:L395-L526
    """
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer

    buffer_size = int(fixture["buffer_size"])
    n_envs = int(fixture["n_envs"])
    n_steps = int(fixture["n_steps"])
    start_idx = int(fixture["start_idx"])
    sequence_length = int(fixture["sequence_length"])
    step_keys = ["observations", "actions", "rewards", "terminated", "truncated", "is_first"]

    jax_rb = SequentialReplayBuffer(buffer_size=buffer_size, n_envs=n_envs)
    for i in range(n_steps):
        step = {}
        for k in step_keys:
            step[k] = fixture[f"step_{k}"][i]
        jax_rb.add(step)

    start_idx_arr = np.array([start_idx], dtype=np.intp)
    env_idxes_arr = np.array([0], dtype=np.intp)
    jax_samples = jax_rb._sample_at_indices(
        precomputed_start_idxes=start_idx_arr,
        env_idxes=env_idxes_arr,
        sequence_length=sequence_length,
        batch_size=1,
        n_samples=1,
    )

    jax_parts = []
    sheeprl_parts = []
    for k in step_keys:
        jax_parts.append(jax_samples[k].ravel())
        sheeprl_parts.append(np.asarray(fixture[f"sheeprl_sample_{k}"]).ravel())

    jax_out = np.concatenate(jax_parts)
    torch_out = np.concatenate(sheeprl_parts)
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/data/buffers.py:L395-L526\n"
        f"  jax:     src/algorithms/dreamer_srl/buffers.py:SequentialReplayBuffer._sample_at_indices\n"
        f"  fixture: buffer_size={buffer_size}, start_idx={start_idx}, "
        f"seq_len={sequence_length}, seed=0xD3EAF\n"
        f"  NOTE: is_first=1 at offset {int(fixture['is_first_expected_offset'])} "
        f"(done_at={int(fixture['done_at'])}, is_first_at={int(fixture['is_first_at'])})"
    )
    return jax_out, torch_out, metadata


def _run_buffer_parallel_env_lane_non_interference(fixture) -> tuple:
    """State-evolution test: no cross-lane leakage across N parallel env lanes.

    Env column i has sentinel obs value (i+1). Windows sampled from each lane
    must contain only that lane's sentinel. Returns max_abs_diff=0 if no leakage.

    Sheeprl source: vendor/sheeprl/sheeprl/data/buffers.py:L480-L489
    """
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer

    buffer_size = int(fixture["buffer_size"])
    n_envs = int(fixture["n_envs"])
    n_steps = int(fixture["n_steps"])
    sequence_length = int(fixture["sequence_length"])
    step_keys = ["observations", "actions", "rewards", "terminated", "truncated", "is_first"]

    jax_rb = SequentialReplayBuffer(buffer_size=buffer_size, n_envs=n_envs)
    for i in range(n_steps):
        step = {}
        for k in step_keys:
            step[k] = fixture[f"step_{k}"][i]
        jax_rb.add(step)

    max_start = jax_rb._pos - sequence_length
    n_windows = min(int(fixture["n_windows"]), max_start + 1)
    start_idxes = np.arange(0, n_windows, dtype=np.intp)
    all_pass = True
    worst_diff = 0.0

    for env_idx in range(n_envs):
        sentinel = float(env_idx + 1)
        env_idxes = np.full(n_windows, env_idx, dtype=np.intp)
        samples = jax_rb._sample_at_indices(
            precomputed_start_idxes=start_idxes,
            env_idxes=env_idxes,
            sequence_length=sequence_length,
            batch_size=n_windows,
            n_samples=1,
        )
        obs = samples["observations"].ravel()
        diff = float(np.max(np.abs(obs - sentinel)))
        worst_diff = max(worst_diff, diff)
        if diff > 0:
            all_pass = False

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/data/buffers.py:L480-L489\n"
        f"  jax:     src/algorithms/dreamer_srl/buffers.py:SequentialReplayBuffer._sample_at_indices\n"
        f"  fixture: buffer_size={buffer_size}, n_envs={n_envs}, "
        f"n_steps={n_steps}, seq_len={sequence_length}, seed=0xD3EAF\n"
        f"  NOTE: cross-lane contamination → max_abs_diff > 0; clean isolation → 0.0"
    )
    # Return sentinel check as 0.0 (pass) or worst_diff (fail)
    return np.array([0.0]), np.array([worst_diff]), metadata


def _run_cadence_yaml_key_parity_with_sheeprl_xs(fixture) -> tuple:
    """Cadence test: agent_xs.yaml keys match sheeprl XS defaults.

    Loads configs/dreamer_srl/agent_xs.yaml and checks 9 cadence keys.
    Returns 0.0 diff if all keys match, else fails with mismatch values.

    Sheeprl source: vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml,
                    vendor/sheeprl/sheeprl/configs/exp/dreamer_v3.yaml
    """
    import sys, os
    import numpy as np
    import yaml
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "dreamer_srl", "agent_xs.yaml",
    )
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    algo = cfg.get("algo", {})
    env = cfg.get("env", {})
    critic = algo.get("critic", {})

    expected = {
        "learning_starts": int(fixture["expected_learning_starts"]),
        "replay_ratio": int(fixture["expected_replay_ratio"]),
        "per_rank_gradient_steps": int(fixture["expected_per_rank_gradient_steps"]),
        "per_rank_sequence_length": int(fixture["expected_per_rank_sequence_length"]),
        "per_rank_batch_size": int(fixture["expected_per_rank_batch_size"]),
        "per_rank_pretrain_steps": int(fixture["expected_per_rank_pretrain_steps"]),
        "per_rank_target_network_update_freq": int(fixture["expected_per_rank_target_network_update_freq"]),
        "total_steps": int(fixture["expected_total_steps"]),
        "num_envs": int(fixture["expected_num_envs"]),
    }

    actual = {
        "learning_starts": algo.get("learning_starts"),
        "replay_ratio": algo.get("replay_ratio"),
        "per_rank_gradient_steps": algo.get("per_rank_gradient_steps"),
        "per_rank_sequence_length": algo.get("per_rank_sequence_length"),
        "per_rank_batch_size": algo.get("per_rank_batch_size"),
        "per_rank_pretrain_steps": algo.get("per_rank_pretrain_steps"),
        "per_rank_target_network_update_freq": critic.get("per_rank_target_network_update_freq"),
        "total_steps": algo.get("total_steps"),
        "num_envs": env.get("num_envs"),
    }

    mismatches = {k: (actual[k], expected[k]) for k in expected if actual[k] != expected[k]}
    if mismatches:
        # Force fail by returning non-zero diff
        diff = float(len(mismatches))
        metadata = (
            f"sheeprl: vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml\n"
            f"  jax:     configs/dreamer_srl/agent_xs.yaml\n"
            f"  fixture: expected cadence keys from sheeprl XS at 33b6366\n"
            f"  MISMATCH: {mismatches}"
        )
        return np.array([diff]), np.array([0.0]), metadata

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml\n"
        f"  jax:     configs/dreamer_srl/agent_xs.yaml\n"
        f"  fixture: expected cadence keys from sheeprl XS at 33b6366\n"
        f"  All 9 cadence keys match: {list(expected.keys())}"
    )
    return np.array([0.0]), np.array([0.0]), metadata


def _run_cadence_env_grad_step_trace_5000_iters(fixture) -> tuple:
    """Cadence test: (env_step, grad_step, per_rank_gs) trace is bit-identical.

    Drives 5000 iterations of the training-cadence loop with XS config.
    Compares against sheeprl's pre-computed trace (generated by gen_cp3b_fixtures.py).
    A mismatch here is the 16x replay-ratio-class divergence — halt and investigate.

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L505-L515,
                    L550-L551, L661-L662
    """
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.utils import Ratio

    num_envs = int(fixture["num_envs"])
    world_size = int(fixture["world_size"])
    total_iters = int(fixture["total_iters"])
    replay_ratio = float(fixture["replay_ratio"])
    learning_starts_steps = int(fixture["learning_starts_steps"])
    per_rank_pretrain_steps = int(fixture["per_rank_pretrain_steps"])
    expected_trace = fixture["expected_trace"]  # [N, 3]

    policy_steps_per_iter = int(num_envs * world_size)
    learning_starts = learning_starts_steps // policy_steps_per_iter
    prefill_steps = learning_starts - int(learning_starts > 0)

    ratio = Ratio(replay_ratio, pretrain_steps=per_rank_pretrain_steps)

    policy_step = 0
    cumulative_grad_steps = 0
    jax_trace = []

    for iter_num in range(1, total_iters + 1):
        policy_step += policy_steps_per_iter
        per_rank_gradient_steps = 0
        if iter_num >= learning_starts:
            ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
            per_rank_gradient_steps = ratio(ratio_steps / world_size)
            cumulative_grad_steps += per_rank_gradient_steps
        jax_trace.append((policy_step, cumulative_grad_steps, per_rank_gradient_steps))

    jax_trace_arr = np.array(jax_trace, dtype=np.int64)
    expected_trace_arr = np.asarray(expected_trace, dtype=np.int64)

    # Return flattened traces for compare() to compute max-abs-diff
    jax_out = jax_trace_arr.ravel().astype(np.float64)
    torch_out = expected_trace_arr.ravel().astype(np.float64)
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L505-L515, L661-L662\n"
        f"  jax:     src/algorithms/dreamer_srl/utils.py:Ratio + cadence arithmetic\n"
        f"  fixture: {total_iters} iters, replay_ratio={replay_ratio}, "
        f"learning_starts={learning_starts_steps}, seed=0xD3EAF\n"
        f"  NOTE: max_abs_diff > 0 = 16x replay-ratio-class divergence — halt and investigate"
    )
    return jax_out, torch_out, metadata


FUNCTION_REGISTRY: dict[str, callable] = {
    # CP1 — utils.py
    "symlog":                _run_symlog,
    "symexp":                _run_symexp,
    "init_weights":          _run_init_weights,
    "uniform_init_weights":  _run_uniform_init_weights,
    "compute_lambda_values": _run_compute_lambda_values,
    "moments_update":        _run_moments_update,
    "ratio":                 _run_ratio,
    "prepare_obs":           _run_prepare_obs,
    # CP3b — buffers.py + cadence wiring
    "buffer_storage_state_after_deterministic_adds":        _run_buffer_storage_state_after_deterministic_adds,
    "buffer_sample_at_indices_matches_sheeprl":             _run_buffer_sample_at_indices_matches_sheeprl,
    "buffer_is_first_marker_placement_in_straddling_window": _run_buffer_is_first_marker_placement_in_straddling_window,
    "buffer_parallel_env_lane_non_interference":            _run_buffer_parallel_env_lane_non_interference,
    "cadence_yaml_key_parity_with_sheeprl_xs":              _run_cadence_yaml_key_parity_with_sheeprl_xs,
    "cadence_env_grad_step_trace_5000_iters":               _run_cadence_env_grad_step_trace_5000_iters,
}

# Per-function threshold overrides — applied when the function has a logged deviation
# that relaxes the default 1e-6. EACH override must have a corresponding DEVIATION_LOG entry.
# Format: function_name → threshold (float)
FUNCTION_THRESHOLDS: dict[str, float] = {
    # D-003: symexp float32 GPU exp ULP difference; max 1 ULP at |x|~5 → ~1.5e-5
    # Relaxed to 2e-5. Pending PI sign-off.
    "symexp": 2e-5,
    # D-002: init_weights/uniform_init_weights are stochastic; runner encodes
    # pass/fail as 0.0/large_diff, so threshold doesn't matter — kept at 1e-6.
}

# Maps checkpoint name → list of function names registered for that CP.
CHECKPOINT_REGISTRY: dict[str, list[str]] = {
    "CP1":  ["symlog", "symexp", "init_weights", "uniform_init_weights",
             "compute_lambda_values", "moments_update", "ratio", "prepare_obs"],
    "CP2":  ["layernorm_gru_cell"],
    "CP2b": ["action_shift"],
    "CP3":  ["zero_init_reward_head", "zero_init_critic_head"],
    "CP3b": ["buffer_storage_state_after_deterministic_adds",
             "buffer_sample_at_indices_matches_sheeprl",
             "buffer_is_first_marker_placement_in_straddling_window",
             "buffer_parallel_env_lane_non_interference",
             "cadence_yaml_key_parity_with_sheeprl_xs",
             "cadence_env_grad_step_trace_5000_iters"],
    "CP4":  ["rssm_transition", "rssm_representation", "get_initial_states"],
    "CP4b": ["is_first_force_set", "is_first_three_quantity_reset"],
    "CP5":  ["twohot_bins_endpoints", "twohot_encode", "twohot_log_prob"],
    "CP6":  ["critic_loss_two_terms", "critic_target_lambda", "discount_weighting"],
    "CP7":  ["polyak_first_call", "polyak_subsequent_call", "polyak_before_train"],
    "CP8":  [],  # end-to-end — handled by scripts/dreamer_srl_offline_check.py
    "CP9":  [],  # integration smoke — no Lever-A tests
    "CP9b": ["prefill_uniform_entropy", "no_gradient_before_learning_starts"],
    "CP10": [],  # wall-clock budget — no Lever-A tests
}


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------

def _effective_threshold(function_name: str, default_threshold: float) -> float:
    """Return the threshold for a function, using per-function override if present."""
    return FUNCTION_THRESHOLDS.get(function_name, default_threshold)


def run_single(function_name: str, fixture_path: str, threshold: float) -> bool:
    """Run one function's comparison.  Returns True on PASS."""
    if function_name not in FUNCTION_REGISTRY:
        print(f"ERROR: '{function_name}' not yet in FUNCTION_REGISTRY.", file=sys.stderr)
        print(f"  Registered functions: {sorted(FUNCTION_REGISTRY.keys()) or '(none — pre-CP0 skeleton)'}", file=sys.stderr)
        print(f"  Add an entry in scripts/sheeprl_jax_diff.py when the function is ported.", file=sys.stderr)
        sys.exit(1)

    eff_threshold = _effective_threshold(function_name, threshold)
    if eff_threshold != threshold:
        print(f"  NOTE: using per-function threshold {eff_threshold:.1e} for '{function_name}' "
              f"(override from FUNCTION_THRESHOLDS — see DEVIATION_LOG.md)")
    fixture = load_fixture(fixture_path)
    runner = FUNCTION_REGISTRY[function_name]
    jax_out, torch_out, metadata = runner(fixture)
    _, passed = compare(jax_out, torch_out, eff_threshold, metadata)
    return passed


def run_checkpoint(checkpoint: str, threshold: float) -> bool:
    """Run all functions registered for a checkpoint.  Returns True if all PASS.

    Exit semantics:
      - All registered functions PASS → True  (caller exits 0)
      - Any registered function FAILs → False (caller exits 1)
      - All expected functions are SKIP (none ported yet) → False (caller exits 1)
        This prevents a silent green light when the developer forgets to register a
        function in FUNCTION_REGISTRY.  The CP-level no-Lever-A checkpoints (CP8,
        CP9, CP10) are the only ones that legitimately return True with zero tests —
        they are explicitly empty lists by design.
    """
    if checkpoint not in CHECKPOINT_REGISTRY:
        print(f"ERROR: unknown checkpoint '{checkpoint}'.  "
              f"Valid: {sorted(CHECKPOINT_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    fns = CHECKPOINT_REGISTRY[checkpoint]
    if not fns:
        # Explicitly empty by design (integration / speed CP — no Lever-A tests).
        print(f"{checkpoint}: no Lever-A functions registered (integration / speed CP).")
        return True

    print(f"\n{'='*60}")
    print(f" Checkpoint {checkpoint} — {len(fns)} function(s)")
    print(f"{'='*60}\n")

    results = {}
    for fn in fns:
        print(f"--- {fn} ---")
        # Derive default fixture path from function name
        fixture_path = f"tests/fixtures/dreamer_srl/{fn}_input.npz"
        if fn not in FUNCTION_REGISTRY:
            print(f"  SKIP: '{fn}' not in FUNCTION_REGISTRY (not yet ported).")
            results[fn] = None
            continue
        fixture = load_fixture(fixture_path)
        runner = FUNCTION_REGISTRY[fn]
        jax_out, torch_out, metadata = runner(fixture)
        eff_threshold = _effective_threshold(fn, threshold)
        if eff_threshold != threshold:
            print(f"  NOTE: using per-function threshold {eff_threshold:.1e} (FUNCTION_THRESHOLDS override)")
        _, passed = compare(jax_out, torch_out, eff_threshold, metadata)
        results[fn] = passed
        print()

    print(f"\n{'='*60}")
    print(f" {checkpoint} summary")
    print(f"{'='*60}")
    any_ran = False
    all_pass = True
    for fn, result in results.items():
        if result is None:
            status = "SKIP (not ported)"
        elif result:
            status = "PASS"
            any_ran = True
        else:
            status = "FAIL"
            all_pass = False
            any_ran = True
        print(f"  {fn:<50s} {status}")

    if not any_ran:
        print(f"\n  WARN: 0 functions ran for {checkpoint} — all were SKIP (not yet ported).")
        print(f"  Register function(s) in FUNCTION_REGISTRY before marking {checkpoint} PASS.")
        all_pass = False

    print()
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-function numerical diff tool for dreamer-srl v3 rebuild. "
            "Runs both the JAX (dreamer-srl) and PyTorch (sheeprl@33b6366) "
            "sides on a shared .npz fixture and reports max-absolute-difference."
        )
    )
    parser.add_argument(
        "--function", "-f",
        metavar="NAME",
        help="Function name to compare (must be in FUNCTION_REGISTRY).",
    )
    parser.add_argument(
        "--fixture",
        metavar="PATH",
        help="Path to the .npz fixture file (required with --function).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        metavar="FLOAT",
        help="Max-abs-diff threshold for PASS/FAIL (default: 1e-6).",
    )
    parser.add_argument(
        "--checkpoint",
        metavar="CP<N>",
        help=(
            "Run all functions registered for checkpoint CP<N> "
            "(e.g. CP5).  Fixture paths are inferred from function names. "
            "Mutually exclusive with --function."
        ),
    )
    args = parser.parse_args()

    if args.checkpoint and args.function:
        parser.error("--checkpoint and --function are mutually exclusive.")

    if args.checkpoint:
        passed = run_checkpoint(args.checkpoint, args.threshold)
    elif args.function:
        if not args.fixture:
            parser.error("--fixture is required when using --function.")
        passed = run_single(args.function, args.fixture, args.threshold)
    else:
        parser.print_help()
        sys.exit(0)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
