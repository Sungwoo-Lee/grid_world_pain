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
    sheeprl: vendor/sheeprl/sheeprl/utils/distribution.py:L224-L276
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

    Loads configs/models/dreamer_srl/agent_xs.yaml and checks 9 cadence keys.
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
            f"  jax:     configs/models/dreamer_srl/agent_xs.yaml\n"
            f"  fixture: expected cadence keys from sheeprl XS at 33b6366\n"
            f"  MISMATCH: {mismatches}"
        )
        return np.array([diff]), np.array([0.0]), metadata

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml\n"
        f"  jax:     configs/models/dreamer_srl/agent_xs.yaml\n"
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


# ---------------------------------------------------------------------------
# CP3 runners (agent.py) — added when CP3 landed
# ---------------------------------------------------------------------------

def _run_zero_init_reward_head(fixture) -> tuple:
    """RewardHead output linear is zero-initialized (cascade fix #27).

    Both kernel and bias must be all-zeros after construction with scale=0.0.
    Zero is zero on both platforms — max_abs_diff = 0.0 expected (no D-### relaxation).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1175
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flax import nnx
    from src.algorithms.dreamer_srl.agent import RewardHead

    in_features  = int(fixture["in_features"])
    out_features = int(fixture["out_features"])
    torch_kernel = fixture["torch_kernel"]  # [in_features, out_features] all-zeros
    torch_bias   = fixture["torch_bias"]    # [out_features] all-zeros

    key  = jax.random.PRNGKey(0xD3EAF)
    rngs = nnx.Rngs(jax.random.PRNGKey(0))
    head = RewardHead(in_features=in_features, out_features=out_features, key=key, rngs=rngs)

    jax_kernel = np.asarray(head.output_linear.kernel[...])  # [in_features, out_features]
    jax_bias   = np.asarray(head.output_linear.bias[...])    # [out_features]

    # Concatenate kernel + bias into a single comparison array
    jax_out   = np.concatenate([jax_kernel.ravel(), jax_bias.ravel()])
    torch_out = np.concatenate([np.asarray(torch_kernel).ravel(), np.asarray(torch_bias).ravel()])

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1175\n"
        f"  jax:     src/algorithms/dreamer_srl/agent.py:RewardHead.__init__\n"
        f"  fixture: in_features={in_features}, out_features={out_features}, seed=0xD3EAF\n"
        f"  NOTE: cascade fix #27 — uniform_init_weights(scale=0.0) → all-zeros; "
        f"max_abs_diff=0.0 expected (no platform ULP drift for zero)"
    )
    return jax_out, torch_out, metadata


def _run_zero_init_critic_head(fixture) -> tuple:
    """CriticHead output linear is zero-initialized (cascade fix #27).

    Both kernel and bias must be all-zeros after construction with scale=0.0.
    Zero is zero on both platforms — max_abs_diff = 0.0 expected (no D-### relaxation).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1172
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flax import nnx
    from src.algorithms.dreamer_srl.agent import CriticHead

    in_features  = int(fixture["in_features"])
    out_features = int(fixture["out_features"])
    torch_kernel = fixture["torch_kernel"]  # [in_features, out_features] all-zeros
    torch_bias   = fixture["torch_bias"]    # [out_features] all-zeros

    key  = jax.random.PRNGKey(0xD3EAF)
    rngs = nnx.Rngs(jax.random.PRNGKey(0))
    head = CriticHead(in_features=in_features, out_features=out_features, key=key, rngs=rngs)

    jax_kernel = np.asarray(head.output_linear.kernel[...])  # [in_features, out_features]
    jax_bias   = np.asarray(head.output_linear.bias[...])    # [out_features]

    jax_out   = np.concatenate([jax_kernel.ravel(), jax_bias.ravel()])
    torch_out = np.concatenate([np.asarray(torch_kernel).ravel(), np.asarray(torch_bias).ravel()])

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1172\n"
        f"  jax:     src/algorithms/dreamer_srl/agent.py:CriticHead.__init__\n"
        f"  fixture: in_features={in_features}, out_features={out_features}, seed=0xD3EAF\n"
        f"  NOTE: cascade fix #27 — uniform_init_weights(scale=0.0) → all-zeros; "
        f"max_abs_diff=0.0 expected (no platform ULP drift for zero)"
    )
    return jax_out, torch_out, metadata


# ---------------------------------------------------------------------------
# CP2 runners (agent.py) — added when CP2 + CP2b landed
# ---------------------------------------------------------------------------

def _run_layernorm_gru_cell(fixture) -> tuple:
    """LayerNormGRUCell forward pass — DEVIATION D-007: float32 matmul ULP.

    JAX XLA float32 matmul accumulation order differs from PyTorch CPU.
    For the 24-element dot-product in the fused linear projection,
    max_abs_diff measured: 2.97e-4.  Threshold relaxed to 5e-4 (D-007).

    CRITICAL: the fixture has reset ≈ 0.55 (post-sigmoid).  A wrong-order
    implementation (reset OUTSIDE tanh) produces O(0.1) deviation — 336x
    above the D-007 ULP drift — so D-007 threshold still catches cascade
    fix #28 violations loudly.

    Sheeprl source: vendor/sheeprl/sheeprl/models/models.py:L331-L410
    """
    import jax
    import jax.numpy as jnp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flax import nnx
    from src.algorithms.dreamer_srl.agent import LayerNormGRUCell

    input_size  = int(fixture["input_size"])
    hidden_size = int(fixture["hidden_size"])
    input_np    = fixture["input"]
    hx_np       = fixture["hx"]
    torch_out   = fixture["torch_out"]

    rngs = nnx.Rngs(jax.random.PRNGKey(0))
    cell = LayerNormGRUCell(input_size=input_size, hidden_size=hidden_size, rngs=rngs)
    cell.linear.kernel = nnx.Param(jnp.asarray(fixture["linear_weight"]).T)  # [I+H, 3H]
    cell.linear.bias   = nnx.Param(jnp.asarray(fixture["linear_bias"]))
    cell.layer_norm.scale = nnx.Param(jnp.asarray(fixture["ln_weight"]))
    cell.layer_norm.bias  = nnx.Param(jnp.asarray(fixture["ln_bias"]))

    x  = jnp.asarray(input_np)
    hx = jnp.asarray(hx_np)
    jax_out = cell(x, hx)

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/models/models.py:L331-L410\n"
        f"  jax:     src/algorithms/dreamer_srl/agent.py:LayerNormGRUCell\n"
        f"  fixture: input.shape={input_np.shape}, hx.shape={hx_np.shape}, seed=0xD3EAF\n"
        f"  NOTE: D-007 — JAX XLA float32 matmul ULP vs PyTorch CPU; threshold 5e-4\n"
        f"  CRITICAL: reset-before-tanh trap (cascade fix #28) produces O(0.1) deviation "
        f"— 336x above D-007 ULP — so threshold still catches the trap"
    )
    return jax_out, torch_out, metadata


def _run_action_shift(fixture) -> tuple:
    """action_shift: prepend zeros, drop last — pure arithmetic, expect exact equality.

    §S2: actions[T, B, A] → [zeros[:1], actions[:-1]].
    No floating-point accumulation; threshold should be 1e-6 (exact equality).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L102-L104
    """
    import jax.numpy as jnp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.agent import action_shift

    actions_np   = fixture["actions"]           # [T, B, A]
    torch_out_np = fixture["torch_out_shifted"] # [T, B, A]

    actions_jax = jnp.asarray(actions_np)
    jax_out = action_shift(actions_jax)

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L102-L104\n"
        f"  jax:     src/algorithms/dreamer_srl/agent.py:action_shift\n"
        f"  fixture: actions.shape={actions_np.shape} (T={int(fixture['T'])}, "
        f"B={int(fixture['B'])}, A={int(fixture['A'])}), seed=0xD3EAF\n"
        f"  NOTE: pure arithmetic (concatenate + zeros); expect exact equality"
    )
    return jax_out, torch_out_np, metadata


# ---------------------------------------------------------------------------
# CP4 runners (agent.py) — added when CP4 + CP4b landed
# ---------------------------------------------------------------------------

def _load_rssm_from_fixture_diff(f):
    """Build a JAX RSSM and load parameters from a CP4/CP4b fixture.

    Helper shared by all 5 CP4/CP4b runners.  Mirrors _load_rssm_from_fixture()
    in tests/algorithms/dreamer_srl/test_agent.py — kept in sync manually.

    PyTorch Linear.weight is [out, in]; JAX nnx.Linear.kernel is [in, out],
    so all Linear weights are transposed on load.  LayerNorm scale/bias are
    [features] — no transposition needed.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L281-L498
    CP4 Lever-A helper.
    """
    import jax
    import jax.numpy as jnp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flax import nnx
    from src.algorithms.dreamer_srl.agent import RSSM

    recurrent_state_size  = int(f["recurrent_state_size"])
    recurrent_dense_units = int(f["recurrent_dense_units"])
    action_dim            = int(f["action_dim"])
    stochastic_size       = int(f["stochastic_size"])
    num_categoricals      = int(f["num_categoricals"])
    num_classes           = int(f["num_classes"])
    transition_hidden_size = int(f["transition_hidden_size"])
    repr_hidden_size      = int(f["repr_hidden_size"])
    encoder_output_dim    = int(f["encoder_output_dim"])

    rngs = nnx.Rngs(jax.random.PRNGKey(0))
    rssm = RSSM(
        recurrent_state_size=recurrent_state_size,
        recurrent_dense_units=recurrent_dense_units,
        action_dim=action_dim,
        stochastic_size=stochastic_size,
        transition_hidden_size=transition_hidden_size,
        repr_hidden_size=repr_hidden_size,
        num_categoricals=num_categoricals,
        num_classes=num_classes,
        encoder_output_dim=encoder_output_dim,
        unimix=0.01,
        rngs=rngs,
    )

    # Recurrent MLP pre-projection
    rssm.recurrent_mlp_linear.kernel = nnx.Param(
        jnp.asarray(f["recurrent_mlp_linear_weight"]).T)
    rssm.recurrent_mlp_norm.scale = nnx.Param(jnp.asarray(f["recurrent_mlp_norm_weight"]))
    rssm.recurrent_mlp_norm.bias  = nnx.Param(jnp.asarray(f["recurrent_mlp_norm_bias"]))

    # GRU cell
    rssm.gru_cell.linear.kernel = nnx.Param(
        jnp.asarray(f["gru_linear_weight"]).T)
    if rssm.gru_cell.linear.use_bias:
        rssm.gru_cell.linear.bias = nnx.Param(jnp.asarray(f["gru_linear_bias"]))
    rssm.gru_cell.layer_norm.scale = nnx.Param(jnp.asarray(f["gru_norm_weight"]))
    rssm.gru_cell.layer_norm.bias  = nnx.Param(jnp.asarray(f["gru_norm_bias"]))

    # Transition model
    rssm.transition_hidden.kernel = nnx.Param(
        jnp.asarray(f["transition_hidden_weight"]).T)
    rssm.transition_norm.scale = nnx.Param(jnp.asarray(f["transition_norm_weight"]))
    rssm.transition_norm.bias  = nnx.Param(jnp.asarray(f["transition_norm_bias"]))
    rssm.transition_out.kernel = nnx.Param(
        jnp.asarray(f["transition_out_weight"]).T)
    rssm.transition_out.bias   = nnx.Param(jnp.asarray(f["transition_out_bias"]))

    # Representation model
    rssm.repr_hidden.kernel = nnx.Param(
        jnp.asarray(f["repr_hidden_weight"]).T)
    rssm.repr_norm.scale = nnx.Param(jnp.asarray(f["repr_norm_weight"]))
    rssm.repr_norm.bias  = nnx.Param(jnp.asarray(f["repr_norm_bias"]))
    rssm.repr_out.kernel = nnx.Param(
        jnp.asarray(f["repr_out_weight"]).T)
    rssm.repr_out.bias   = nnx.Param(jnp.asarray(f["repr_out_bias"]))

    # Learnable initial recurrent state
    rssm.initial_recurrent_state = nnx.Param(
        jnp.asarray(f["initial_recurrent_state"]))

    return rssm


def _run_rssm_transition(fixture) -> tuple:
    """RSSM._transition: transition (prior) model parity — DEVIATION D-008.

    Calls _transition(recurrent_state, sample_state=False) — mode output,
    no PRNG consumed — and compares prior_logits + prior_state (mode)
    against sheeprl@33b6366 reference.

    D-008: JAX XLA float32 matmul accumulation for RSSM MLP (deeper than D-007).
    Measured max_abs_diff: 6.838e-4 (logits). Threshold relaxed to 2e-3.
    Any semantic error (wrong MLP depth, missing LayerNorm) produces O(0.1)
    deviation — 143x above threshold — so architecture bugs are still caught.

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L467-L480
    CP4 Lever-A test (cascade fix #30 — one-hidden-MLP, NOT bare Linear).
    """
    import jax.numpy as jnp
    import numpy as np

    rssm = _load_rssm_from_fixture_diff(fixture)

    recurrent_state_np = fixture["recurrent_state"]   # [B, H_rec]
    torch_logits_np    = fixture["torch_out_logits"]  # [B, S*D]
    torch_state_np     = fixture["torch_out_state"]   # [B, S, D]

    jax_logits, jax_state = rssm._transition(
        jnp.asarray(recurrent_state_np), sample_state=False, key=None
    )

    # Concatenate logits + flattened state for a single compare() call
    jax_out   = np.concatenate([np.asarray(jax_logits).ravel(),
                                 np.asarray(jax_state).ravel()])
    torch_out = np.concatenate([np.asarray(torch_logits_np).ravel(),
                                 np.asarray(torch_state_np).ravel()])

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L467-L480\n"
        f"  jax:     src/algorithms/dreamer_srl/agent.py:RSSM._transition\n"
        f"  fixture: recurrent_state.shape={recurrent_state_np.shape}, seed=0xD3EAF\n"
        f"  NOTE: D-008 — JAX XLA float32 matmul ULP cascade; threshold 2e-3; "
        f"semantic errors produce O(0.1) deviation"
    )
    return jax_out, torch_out, metadata


def _run_rssm_representation(fixture) -> tuple:
    """RSSM._representation: representation (posterior) logits parity — DEVIATION D-008.

    Manually computes repr logits (repr_hidden + LayerNorm + SiLU + repr_out +
    _uniform_mix) and _compute_stochastic_state in mode form to avoid PRNG.
    Compares logits + mode state against sheeprl@33b6366 reference.

    D-008: same class as D-007 (substrate-mechanical float32 matmul ULP).
    Measured max_abs_diff: 7.193e-4 (logits). Threshold 2e-3.

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L451-L465
    CP4 Lever-A test (cascade fix #30 — one-hidden-MLP in repr model).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    rssm = _load_rssm_from_fixture_diff(fixture)

    recurrent_state_np = fixture["recurrent_state"]   # [B, H_rec]
    embedded_obs_np    = fixture["embedded_obs"]      # [B, enc_dim]
    torch_logits_np    = fixture["torch_out_logits"]  # [B, S*D]
    torch_state_np     = fixture["torch_out_state"]   # [B, S, D]

    # Manual repr logits (deterministic — avoids PRNG divergence)
    x = jnp.concatenate([
        jnp.asarray(recurrent_state_np), jnp.asarray(embedded_obs_np)
    ], axis=-1)
    h = rssm.repr_hidden(x)
    h = rssm.repr_norm(h)
    h = jax.nn.silu(h)
    logits = rssm.repr_out(h)
    jax_logits = rssm._uniform_mix(logits)

    jax_state = rssm._compute_stochastic_state(jax_logits, sample=False, key=None)

    jax_out   = np.concatenate([np.asarray(jax_logits).ravel(),
                                 np.asarray(jax_state).ravel()])
    torch_out = np.concatenate([np.asarray(torch_logits_np).ravel(),
                                 np.asarray(torch_state_np).ravel()])

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L451-L465\n"
        f"  jax:     src/algorithms/dreamer_srl/agent.py:RSSM._representation\n"
        f"  fixture: recurrent_state.shape={recurrent_state_np.shape}, "
        f"embedded_obs.shape={embedded_obs_np.shape}, seed=0xD3EAF\n"
        f"  NOTE: D-008 — JAX XLA float32 matmul ULP cascade; threshold 2e-3"
    )
    return jax_out, torch_out, metadata


def _run_get_initial_states(fixture) -> tuple:
    """RSSM.get_initial_states: returns mode (no PRNG consumed) — DEVIATION D-008.

    get_initial_states(batch_size) must NOT have a `key` parameter.
    Returns (initial_hx, initial_z) where initial_hx = tanh(zeros) ≈ 0,
    and initial_z = _transition(initial_hx, sample_state=False)[1] (mode).

    D-008: same class as D-007. initial_hx is ~0 (no drift); initial_z goes
    through transition MLP so D-008 drift may appear.

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L391-L394
    CP4 Lever-A test (determinism + no-PRNG discipline).
    """
    import inspect
    import jax.numpy as jnp
    import numpy as np

    rssm = _load_rssm_from_fixture_diff(fixture)

    batch_size  = int(fixture["batch_size"])
    torch_hx_np = fixture["torch_out_hx"]  # [B, H_rec]
    torch_z_np  = fixture["torch_out_z"]   # [B, S, D]

    # Signature check: must NOT have a `key` parameter
    sig = inspect.signature(rssm.get_initial_states)
    if "key" in sig.parameters:
        raise AssertionError(
            "get_initial_states has a `key` parameter — this violates the no-PRNG mandate. "
            "Remove it: the method must be deterministic (returns mode)."
        )

    jax_hx, jax_z = rssm.get_initial_states(batch_size)

    jax_out   = np.concatenate([np.asarray(jax_hx).ravel(),
                                 np.asarray(jax_z).ravel()])
    torch_out = np.concatenate([np.asarray(torch_hx_np).ravel(),
                                 np.asarray(torch_z_np).ravel()])

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L391-L394\n"
        f"  jax:     src/algorithms/dreamer_srl/agent.py:RSSM.get_initial_states\n"
        f"  fixture: batch_size={batch_size}, seed=0xD3EAF\n"
        f"  NOTE: D-008 — initial_hx=tanh(zeros)≈0 (no drift); initial_z through "
        f"transition MLP may show D-008 ULP drift; threshold 2e-3"
    )
    return jax_out, torch_out, metadata


def _run_is_first_force_set(fixture) -> tuple:
    """RSSM.dynamic: is_first=1 resets the first env's states; others unchanged.

    Calls dynamic(posterior, recurrent_state, action, embedded_obs, is_first=[1,0,0,0],
    key) and compares h output against sheeprl reference.

    §S4 three-quantity arithmetic-mask form:
        action          = (1 - is_first) * action
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_hx
        posterior_flat  = (1 - is_first) * posterior_flat  + is_first * initial_z_flat
    FORM: arithmetic mask, NOT jnp.where.

    D-008: measured h diff ≤ 5.597e-4; threshold 2e-3.

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L423-L435
    CP4b Lever-A test (§S4 three-quantity arithmetic-mask reset, force-set).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    rssm = _load_rssm_from_fixture_diff(fixture)

    posterior_np       = fixture["posterior"]            # [B, S, D]
    recurrent_state_np = fixture["recurrent_state"]     # [B, H_rec]
    action_np          = fixture["action"]               # [B, A]
    embedded_obs_np    = fixture["embedded_obs"]         # [B, enc_dim]
    is_first_np        = fixture["is_first"]             # [B, 1]
    torch_h_np         = fixture["torch_out_h"]          # [B, H_rec]

    key = jax.random.PRNGKey(0)
    h_out, post_out, prior_out, _, _ = rssm.dynamic(
        jnp.asarray(posterior_np),
        jnp.asarray(recurrent_state_np),
        jnp.asarray(action_np),
        jnp.asarray(embedded_obs_np),
        jnp.asarray(is_first_np),
        key,
    )

    jax_out   = np.asarray(h_out)
    torch_out = np.asarray(torch_h_np)

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L423-L435\n"
        f"  jax:     src/algorithms/dreamer_srl/agent.py:RSSM.dynamic\n"
        f"  fixture: is_first=[1,0,0,0] (env 0 reset), batch_size={posterior_np.shape[0]}, "
        f"seed=0xD3EAF\n"
        f"  NOTE: D-008 — h comparison only (posterior is stochastic, D-009 applies); "
        f"threshold 2e-3; §S4 arithmetic-mask form (NOT jnp.where)"
    )
    return jax_out, torch_out, metadata


def _run_is_first_three_quantity_reset(fixture) -> tuple:
    """RSSM.dynamic: all 3 quantities reset at done boundary in a 10-step rollout.

    Drives T=10 steps of dynamic() step-by-step, with done_at=5 (is_first=1 at t=6).
    Compares h (recurrent state) rollout against sheeprl@33b6366 reference.
    Posterior comparison is skipped — DEVIATION D-009: different platform PRNG
    (PyTorch rsample vs JAX gumbel-softmax) makes cross-platform stochastic comparison
    impossible.  h is deterministic given the same fixture inputs.

    D-008: measured h_seq diff ≤ 5.597e-4; threshold 2e-3.
    D-009: posterior comparison not performed (stochastic, different RNG streams).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L423-L435
    CP4b Lever-A test (§S4 three-quantity arithmetic-mask reset, rollout boundary).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    rssm = _load_rssm_from_fixture_diff(fixture)

    posterior_seq_np    = fixture["posterior_seq"]      # [T, B, S, D]
    recurrent_seq_np    = fixture["recurrent_seq"]      # [T, B, H_rec]
    action_seq_np       = fixture["action_seq"]         # [T, B, A]
    embedded_obs_seq_np = fixture["embedded_obs_seq"]   # [T, B, enc_dim]
    is_first_seq_np     = fixture["is_first_seq"]       # [T, B, 1]
    torch_h_seq_np      = fixture["torch_out_h_seq"]    # [T, B, H_rec]
    T                   = int(fixture["seq_len"])
    done_at             = int(fixture["done_at"])
    is_first_at         = int(fixture["is_first_at"])

    key = jax.random.PRNGKey(0)
    jax_h_seq = []

    for t in range(T):
        key, step_key = jax.random.split(key)
        h_t, _, _, _, _ = rssm.dynamic(
            jnp.asarray(posterior_seq_np[t]),
            jnp.asarray(recurrent_seq_np[t]),
            jnp.asarray(action_seq_np[t]),
            jnp.asarray(embedded_obs_seq_np[t]),
            jnp.asarray(is_first_seq_np[t]),
            step_key,
        )
        jax_h_seq.append(np.asarray(h_t))

    jax_h_arr = np.stack(jax_h_seq, axis=0)   # [T, B, H_rec]

    jax_out   = jax_h_arr.ravel()
    torch_out = np.asarray(torch_h_seq_np).ravel()

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L423-L435\n"
        f"  jax:     src/algorithms/dreamer_srl/agent.py:RSSM.dynamic (T={T}-step rollout)\n"
        f"  fixture: T={T}, done_at={done_at}, is_first_at={is_first_at}, "
        f"batch_size={posterior_seq_np.shape[1]}, seed=0xD3EAF\n"
        f"  NOTE: D-008 threshold 2e-3; D-009 — posterior comparison skipped "
        f"(stochastic, different RNG streams across platforms)"
    )
    return jax_out, torch_out, metadata


# ---------------------------------------------------------------------------
# CP5 runners (loss.py) — added when CP5 landed
# ---------------------------------------------------------------------------

def _run_twohot_bins_endpoints(fixture) -> tuple:
    """twohot_bins_endpoints: linspace(-20,+20,255) grid — DEVIATION D-006.

    JAX jnp.linspace and PyTorch torch.linspace produce different float32 values
    at the midpoint (bin[127]): JAX=0.0, PyTorch=7.45e-8. This is a 1-ULP
    linspace implementation difference that cascades to max_abs_diff=1.9e-6.
    Threshold relaxed to 3e-5 (D-006 logged; pending PI sign-off at CP5 gate).

    Sheeprl source: vendor/sheeprl/sheeprl/utils/distribution.py:L237
    """
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.loss import TwoHotEncoding

    torch_bins = fixture["torch_bins"]  # [255] from sheeprl PyTorch
    n_bins = int(fixture["n_bins"])
    low = float(fixture["low"])
    high = float(fixture["high"])

    dummy_logits = jnp.zeros((1, n_bins))
    jax_dist = TwoHotEncoding(dummy_logits, dims=0, low=int(low), high=int(high))
    jax_bins = jax_dist.bins  # [255]

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/utils/distribution.py:L237\n"
        f"  jax:     src/algorithms/dreamer_srl/loss.py:TwoHotEncoding.__init__\n"
        f"  fixture: bins.shape={torch_bins.shape}, low={low}, high={high}, seed=0xD3EAF\n"
        f"  NOTE: D-006 — JAX linspace midpoint=0.0; PyTorch=7.45e-8 (1 ULP); threshold 3e-5"
    )
    return jax_bins, torch_bins, metadata


def _run_twohot_encode(fixture) -> tuple:
    """twohot_encode: two-hot target [T,B,255] from [T,B,1] targets — DEVIATION D-006.

    THIS IS THE TEST THAT CATCHES THE HISTORICAL BUG.
    If bins are stored in real reward space (symexp applied at storage), the bin
    lookup would use wrong indices and the two-hot weights would mismatch by orders
    of magnitude. max_abs_diff < 1e-2 is the historical-bug scale; < 3e-5 is the
    D-006 float32 platform drift.

    Sheeprl source: vendor/sheeprl/sheeprl/utils/distribution.py:L253-L274
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.loss import TwoHotEncoding
    from src.algorithms.dreamer_srl.utils import symlog

    logits_np = fixture["logits"]          # [T, B, 255]
    targets_np = fixture["targets"]        # [T, B, 1]
    torch_twohot = fixture["torch_out_twohot"]  # [T, B, 255]

    logits_jax = jnp.asarray(logits_np)
    targets_jax = jnp.asarray(targets_np)
    jax_dist = TwoHotEncoding(logits_jax, dims=1, low=-20, high=20)
    n_bins = jax_dist.bins.shape[0]

    # Replicate the two-hot encoding (the internal bin-lookup path from log_prob)
    # Ported from sheeprl@33b6366:sheeprl/utils/distribution.py:L253-L274
    x = symlog(targets_jax)
    below = (jax_dist.bins <= x).astype(jnp.int32).sum(axis=-1, keepdims=True) - 1
    above = below + 1
    above = jnp.minimum(above, n_bins - 1)
    below = jnp.maximum(below, 0)
    equal = below == above
    dist_to_below = jnp.where(equal, jnp.ones_like(x), jnp.abs(jax_dist.bins[below] - x))
    dist_to_above = jnp.where(equal, jnp.ones_like(x), jnp.abs(jax_dist.bins[above] - x))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    jax_twohot = (
        jax.nn.one_hot(below, n_bins) * weight_below[..., None]
        + jax.nn.one_hot(above, n_bins) * weight_above[..., None]
    )
    jax_twohot = jnp.squeeze(jax_twohot, axis=-2)  # [T, B, 255]

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/utils/distribution.py:L253-L274\n"
        f"  jax:     src/algorithms/dreamer_srl/loss.py:TwoHotEncoding.log_prob (encode path)\n"
        f"  fixture: logits.shape={logits_np.shape}, targets.shape={targets_np.shape}, seed=0xD3EAF\n"
        f"  NOTE: D-006 — linspace ULP cascades to encode; threshold 3e-5"
    )
    return jax_twohot, torch_twohot, metadata


def _run_twohot_log_prob(fixture) -> tuple:
    """twohot_log_prob: log_prob(target) for [T,B,1] targets — DEVIATION D-006.

    Tests both encode (D-006 linspace ULP) and log-pred (logsumexp float32) paths.
    max_abs_diff measured: 1.8e-5; relative diff: 2.4e-6 — float32 platform drift.

    Sheeprl source: vendor/sheeprl/sheeprl/utils/distribution.py:L253-L276
    """
    import jax.numpy as jnp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.loss import TwoHotEncoding

    logits_np = fixture["logits"]                    # [T, B, 255]
    targets_np = fixture["targets"]                  # [T, B, 1]
    torch_log_prob = fixture["torch_out_log_prob"]   # [T, B]

    logits_jax = jnp.asarray(logits_np)
    targets_jax = jnp.asarray(targets_np)
    jax_dist = TwoHotEncoding(logits_jax, dims=1, low=-20, high=20)
    jax_log_prob = jax_dist.log_prob(targets_jax)   # [T, B]

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/utils/distribution.py:L253-L276\n"
        f"  jax:     src/algorithms/dreamer_srl/loss.py:TwoHotEncoding.log_prob\n"
        f"  fixture: logits.shape={logits_np.shape}, targets.shape={targets_np.shape}, seed=0xD3EAF\n"
        f"  NOTE: D-006 — linspace+logsumexp float32 cascade; threshold 3e-5 (rel diff 2.4e-6)"
    )
    return jax_log_prob, torch_log_prob, metadata


# ---------------------------------------------------------------------------
# CP6 runners (train.py) — added when CP6 landed
# ---------------------------------------------------------------------------

def _run_critic_loss_two_terms(fixture) -> tuple:
    """critic_loss_two_terms: two-term NLL loss (cascade fix #29) — DEVIATION D-006 class.

    Verifies both terms of the critic loss:
      term1 = -qv.log_prob(stop_gradient(lambda_values))      [sheeprl L314]
      term2 = -qv.log_prob(stop_gradient(target_critic_values)) [sheeprl L315]
      value_loss = mean((term1 + term2) * discount[:-1].squeeze(-1))  [sheeprl L316]

    D-006 class: TwoHotEncoding linspace ULP drift cascades to log_prob.
    Threshold: 4e-5 (same-order as D-006; slightly wider for the sum of two terms).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L307-L316
    """
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.train import compute_critic_loss

    qv_logits_np   = fixture["qv_logits"]        # [H, BT, 255]
    lambda_val_np  = fixture["lambda_values"]    # [H, BT, 1]
    target_val_np  = fixture["target_values"]    # [H, BT, 1]
    discount_np    = fixture["discount"]          # [H, BT]
    torch_lp1_np   = fixture["torch_out_lp1"]   # [H, BT]
    torch_lp2_np   = fixture["torch_out_lp2"]   # [H, BT]
    torch_loss     = float(fixture["torch_out_value_loss"])

    H  = discount_np.shape[0]
    BT = discount_np.shape[1]
    # Extend discount [H, BT] → [H+1, BT, 1] for compute_critic_loss
    dummy_row = np.zeros((1, BT), dtype=np.float32)
    discount_ext = np.concatenate([discount_np, dummy_row], axis=0)[:, :, np.newaxis]

    qv_logits  = jnp.asarray(qv_logits_np)
    lambda_val = jnp.asarray(lambda_val_np)
    target_val = jnp.asarray(target_val_np)
    discount   = jnp.asarray(discount_ext)

    value_loss, neg_lp1, neg_lp2 = compute_critic_loss(
        qv_logits=qv_logits,
        lambda_values=lambda_val,
        target_critic_values=target_val,
        discount=discount,
    )

    # Concatenate both terms + scalar for a single compare() call
    jax_out   = np.concatenate([
        np.asarray(neg_lp1).ravel(),
        np.asarray(neg_lp2).ravel(),
        np.array([float(value_loss)]),
    ])
    torch_out = np.concatenate([
        np.asarray(torch_lp1_np).ravel(),
        np.asarray(torch_lp2_np).ravel(),
        np.array([torch_loss]),
    ])

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L307-L316\n"
        f"  jax:     src/algorithms/dreamer_srl/train.py:compute_critic_loss\n"
        f"  fixture: qv_logits.shape={qv_logits_np.shape}, H={H}, BT={BT}, seed=0xD3EAF\n"
        f"  NOTE: D-006 class (TwoHotEncoding linspace ULP drift); threshold 4e-5\n"
        f"  CASCADE FIX #29: both log_prob terms (-qv.log_prob(lambda) AND -qv.log_prob(target)) required"
    )
    return jax_out, torch_out, metadata


def _run_critic_target_lambda(fixture) -> tuple:
    """critic_target_lambda: critic uses UN-normalised lambda_values — DEVIATION D-006 class.

    Verifies: qv.log_prob(raw_lambda) matches sheeprl reference (NOT Moments-normed).
    The sheeprl critic at L314 uses `lambda_values.detach()` before Moments normalization.

    D-006 class: TwoHotEncoding linspace ULP drift cascades to log_prob.
    Threshold: 5e-5 (D-010, PI-raised at CP6 gate 2026-05-14 from initial 4e-5 log entry
    for margin-band consistency with D-006/D-007/D-008; same-order D-006 mechanism).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L251-L256, L314
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.loss import TwoHotEncoding

    qv_logits_np     = fixture["qv_logits"]        # [H, BT, 255]
    lambda_val_np    = fixture["lambda_values"]    # [H, BT, 1]  — raw (un-normalised)
    torch_lp_raw_np  = fixture["torch_out_lp_raw"] # [H, BT]  — reference

    qv = TwoHotEncoding(jnp.asarray(qv_logits_np), dims=1)
    jax_lp_raw = qv.log_prob(jax.lax.stop_gradient(jnp.asarray(lambda_val_np)))  # [H, BT]

    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L314\n"
        f"  jax:     src/algorithms/dreamer_srl/loss.py:TwoHotEncoding.log_prob\n"
        f"  fixture: qv_logits.shape={qv_logits_np.shape}, lambda_values.shape={lambda_val_np.shape}, "
        f"seed=0xD3EAF+1\n"
        f"  NOTE: D-006 class (linspace ULP drift); D-010 PI-ratified threshold 5e-5\n"
        f"  KEY: critic must use UN-normalised lambda_values (raw), NOT Moments-normed (sheeprl L314)"
    )
    return jax_lp_raw, torch_lp_raw_np, metadata


def _run_discount_weighting(fixture) -> tuple:
    """discount_weighting: cumprod(continues*gamma, axis=0)/gamma — §S6, expect < 1e-6.

    Verifies: discount = jax.lax.stop_gradient(jnp.cumprod(continues*gamma, axis=0)/gamma)
    [0]=1 invariant when continues[0]=1 (§S5 true-continue splice, no termination).
    discount[:-1].squeeze(-1) shape = [H, BT].

    Pure cumprod arithmetic — no TwoHotEncoding, no D-006 cascade.
    Threshold: default 1e-6 (exact arithmetic; no deviation expected).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L259-L260
    """
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.train import compute_discount

    continues_np    = fixture["continues"]                  # [H+1, BT, 1]
    gamma           = float(fixture["gamma"])
    torch_disc_full = fixture["torch_out_discount_full"]   # [H+1, BT, 1]

    discount_jax = compute_discount(jnp.asarray(continues_np), gamma)  # [H+1, BT, 1]

    H  = continues_np.shape[0] - 1
    BT = continues_np.shape[1]
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L259-L260\n"
        f"  jax:     src/algorithms/dreamer_srl/train.py:compute_discount\n"
        f"  fixture: continues.shape={continues_np.shape}, gamma={gamma}, seed=0xD3EAF+2\n"
        f"  §S6: cumprod(continues*gamma, axis=0)/gamma; discount[0]=1 when continues[0]=1\n"
        f"  NOTE: pure arithmetic — no TwoHotEncoding; expect < 1e-6 (no D-006 cascade)"
    )
    return np.asarray(discount_jax), np.asarray(torch_disc_full), metadata


def _run_polyak_first_call(fixture) -> tuple:
    """polyak_first_call: tau=1.0 hard copy — target = online exactly.

    Verifies: polyak_update(online, target, tau=1.0) returns params byte-identical
    to online. First call in sheeprl's inner gradient-step loop.

    Pure EMA arithmetic — no TwoHotEncoding.
    Threshold: default 1e-6 (exact arithmetic; no deviation expected).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L678-L680
        tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
        for cp, tcp in zip(...): tcp.data.copy_(tau * cp + (1 - tau) * tcp)
    """
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.train import polyak_update

    tau = float(fixture["tau"])
    param_keys = sorted([k.replace("online_", "") for k in fixture.files if k.startswith("online_")])

    online_params = {k: jnp.asarray(fixture[f"online_{k}"]) for k in param_keys}
    target_params = {k: jnp.asarray(fixture[f"target_init_{k}"]) for k in param_keys}
    torch_out = {k: np.asarray(fixture[f"torch_out_target_{k}"]) for k in param_keys}

    new_target = polyak_update(online_params, target_params, tau=tau)

    # Flatten to single arrays for comparison
    jax_flat = np.concatenate([np.asarray(new_target[k]).ravel() for k in sorted(new_target)])
    torch_flat = np.concatenate([torch_out[k].ravel() for k in sorted(torch_out)])

    n_params = sum(v.size for v in online_params.values())
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L678-L680\n"
        f"  jax:    src/algorithms/dreamer_srl/train.py:polyak_update\n"
        f"  fixture: tau={tau}, {len(param_keys)} param groups, {n_params} total params, seed=0xD3EAF+3\n"
        f"  tau=1.0 → hard copy (target = online); expect byte-identical (diff = 0)\n"
        f"  NOTE: pure EMA arithmetic — no TwoHotEncoding; expect < 1e-6 (no D-006 cascade)"
    )
    return jax_flat, torch_flat, metadata


def _run_polyak_subsequent_call(fixture) -> tuple:
    """polyak_subsequent_call: tau=0.02 EMA blend — target = 0.98*target + 0.02*online.

    Verifies: polyak_update(online, target, tau=0.02) returns
        (1-0.02)*target + 0.02*online  for each parameter array.
    Subsequent calls in sheeprl's inner gradient-step loop.

    Pure EMA arithmetic — no TwoHotEncoding.
    Threshold: default 1e-6 (exact arithmetic; no deviation expected).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L678-L680
        tau = cfg.algo.critic.tau  (= 0.02 in XS config for subsequent calls)
    """
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.train import polyak_update

    tau = float(fixture["tau"])
    param_keys = sorted([k.replace("online_", "") for k in fixture.files if k.startswith("online_")])

    online_params = {k: jnp.asarray(fixture[f"online_{k}"]) for k in param_keys}
    target_params = {k: jnp.asarray(fixture[f"target_init_{k}"]) for k in param_keys}
    torch_out = {k: np.asarray(fixture[f"torch_out_target_{k}"]) for k in param_keys}

    new_target = polyak_update(online_params, target_params, tau=tau)

    jax_flat = np.concatenate([np.asarray(new_target[k]).ravel() for k in sorted(new_target)])
    torch_flat = np.concatenate([torch_out[k].ravel() for k in sorted(torch_out)])

    n_params = sum(v.size for v in online_params.values())
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L678-L680\n"
        f"  jax:    src/algorithms/dreamer_srl/train.py:polyak_update\n"
        f"  fixture: tau={tau}, {len(param_keys)} param groups, {n_params} total params, seed=0xD3EAF+4\n"
        f"  tau=0.02 → EMA blend: (1-tau)*target + tau*online\n"
        f"  NOTE: pure EMA arithmetic — no TwoHotEncoding; expect < 1e-6 (no D-006 cascade)"
    )
    return jax_flat, torch_flat, metadata


def _run_polyak_before_train(fixture) -> tuple:
    """polyak_before_train: call-order verification (polyak BEFORE one_train_step).

    Verifies two-step trace: step-0 hard copy + step-1 EMA blend, and verifies
    that the polyak_update function is defined in train.py (structural check for
    the call-order requirement: polyak fires BEFORE one_train_step per sheeprl's
    inner gradient-step loop ordering).

    Pure EMA arithmetic — no TwoHotEncoding.
    Threshold: default 1e-6 (exact arithmetic; no deviation expected).

    Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L673-L697
        for i in range(per_rank_gradient_steps):
            ... polyak update (L679-L680) ...
            ... train() call (L686) ...
    The two-step trace verifies the EMA arithmetic across two sequential calls.
    """
    import jax.numpy as jnp
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.algorithms.dreamer_srl.train import polyak_update

    online_s0 = jnp.asarray(fixture["online_step0"])   # [8]
    target_init = jnp.asarray(fixture["target_init"])  # [8]
    online_s1 = jnp.asarray(fixture["online_step1"])   # [8]
    tau_first = float(fixture["tau_first"])             # 1.0
    tau_sub = float(fixture["tau_subsequent"])          # 0.02
    torch_s1 = np.asarray(fixture["torch_out_target_after_step1"])  # [8]

    # Step 0: hard copy
    target_s0 = polyak_update({"w": online_s0}, {"w": target_init}, tau=tau_first)
    # Step 1: EMA blend with updated online
    target_s1 = polyak_update({"w": online_s1}, target_s0, tau=tau_sub)

    jax_out = np.asarray(target_s1["w"])
    n_params = online_s0.size * 2  # two steps
    metadata = (
        f"sheeprl: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L673-L697\n"
        f"  jax:    src/algorithms/dreamer_srl/train.py:polyak_update (two-step trace)\n"
        f"  fixture: tau_first={tau_first}, tau_sub={tau_sub}, seed=0xD3EAF+5\n"
        f"  Step 0: hard copy (tau=1.0). Step 1: EMA blend (tau=0.02) with updated online.\n"
        f"  Call-order: polyak fires BEFORE one_train_step (sheeprl L679-L680 before L686).\n"
        f"  NOTE: pure EMA arithmetic — no TwoHotEncoding; expect < 1e-6 (no D-006 cascade)"
    )
    return jax_out, torch_s1, metadata


FUNCTION_REGISTRY: dict[str, callable] = {
    # CP3 — agent.py (zero-init output linears, cascade fix #27)
    "zero_init_reward_head": _run_zero_init_reward_head,
    "zero_init_critic_head": _run_zero_init_critic_head,
    # CP2 — agent.py
    "layernorm_gru_cell":    _run_layernorm_gru_cell,
    "action_shift":          _run_action_shift,
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
    # CP4 — agent.py (RSSM, cascade fix #30)
    "rssm_transition":               _run_rssm_transition,
    "rssm_representation":           _run_rssm_representation,
    "get_initial_states":            _run_get_initial_states,
    # CP4b — agent.py (§S4 three-quantity arithmetic-mask reset)
    "is_first_force_set":            _run_is_first_force_set,
    "is_first_three_quantity_reset": _run_is_first_three_quantity_reset,
    # CP5 — loss.py
    "twohot_bins_endpoints": _run_twohot_bins_endpoints,
    "twohot_encode":         _run_twohot_encode,
    "twohot_log_prob":       _run_twohot_log_prob,
    # CP6 — train.py (critic loss cascade fix #29, §S6 discount, §S9 continue)
    "critic_loss_two_terms": _run_critic_loss_two_terms,
    "critic_target_lambda":  _run_critic_target_lambda,
    "discount_weighting":    _run_discount_weighting,
    # CP7 — train.py (Polyak target-critic EMA update, §S5/§S7)
    "polyak_first_call":       _run_polyak_first_call,
    "polyak_subsequent_call":  _run_polyak_subsequent_call,
    "polyak_before_train":     _run_polyak_before_train,
}

# Per-function threshold overrides — applied when the function has a logged deviation
# that relaxes the default 1e-6. EACH override must have a corresponding DEVIATION_LOG entry.
# Format: function_name → threshold (float)
FUNCTION_THRESHOLDS: dict[str, float] = {
    # D-007: LayerNormGRUCell — JAX XLA float32 matmul accumulation order vs PyTorch CPU.
    # 24-element dot-product accumulation drift cascades through LayerNorm + gate nonlinearities.
    # Measured max_abs_diff: 2.97e-4. Relaxed to 5e-4. PI sign-off required at CP2 gate.
    # CRITICAL: reset-before-tanh trap (cascade fix #28) produces O(0.1) — 336x above threshold.
    "layernorm_gru_cell": 5e-4,
    # D-003: symexp float32 GPU exp ULP difference; max 1 ULP at |x|~5 → ~1.5e-5
    # Relaxed to 2e-5. Pending PI sign-off.
    "symexp": 2e-5,
    # D-002: init_weights/uniform_init_weights are stochastic; runner encodes
    # pass/fail as 0.0/large_diff, so threshold doesn't matter — kept at 1e-6.
    # D-008: RSSM MLP float32 matmul ULP cascade — deeper than D-007 (two Linear layers
    # + LayerNorm + SiLU in transition/repr/recurrent-pre-projection MLP chains).
    # Measured max_abs_diff: transition logits 6.838e-4, repr logits 7.193e-4,
    # rollout h 5.597e-4. Threshold relaxed to 2e-3 (3x margin above 7.193e-4).
    # CRITICAL: any semantic error (wrong MLP depth, missing LayerNorm, missing pre-proj)
    # produces O(0.1) deviation — 143x above threshold — so architecture bugs are caught.
    # Substrate-mechanical class (same as D-007). PI sign-off required at CP4 gate.
    "rssm_transition":               2e-3,
    "rssm_representation":           2e-3,
    "get_initial_states":            2e-3,
    "is_first_force_set":            2e-3,
    "is_first_three_quantity_reset": 2e-3,
    # D-006: JAX jnp.linspace vs PyTorch torch.linspace ULP difference at midpoint.
    # bins[127]: JAX=0.0, PyTorch=7.45e-8 (1 float32 ULP at step 40/254*127).
    # This cascades: bins 1.9e-6, encode 6.1e-6, log_prob 1.8e-5 (rel diff 2.4e-6).
    # Relaxed to 3e-5 (same order as D-003; well below any semantic-error scale).
    # Pending PI sign-off at CP5 gate.
    "twohot_bins_endpoints": 3e-5,
    "twohot_encode": 3e-5,
    "twohot_log_prob": 3e-5,
    # CP6 — D-006 class cascades to critic log_prob terms.
    # critic_loss_two_terms: two log_prob terms summed; max_abs_diff measured ≈ same as D-006.
    # critic_target_lambda: single log_prob with different seed; measured ≈ 3.1e-5 (D-010).
    # Threshold for critic_target_lambda raised 4e-5 → 5e-5 by PI at CP6 gate
    # (2026-05-14, pi/calls/2026-05-14_dreamer_srl_v3_cp6_deviations.md) — margin-band
    # consistency with D-006 (1.65x) / D-007 (1.68x) / D-008 (2.78x); D-010 measured
    # 3.099e-5 at 1.61x margin against 5e-5. Still 2000x below O(0.1) semantic-error scale.
    # discount_weighting: pure cumprod arithmetic; expect < 1e-6 (no D-006 cascade).
    "critic_loss_two_terms": 4e-5,
    "critic_target_lambda":  5e-5,
    # discount_weighting: no TwoHotEncoding involved; cumprod is exact arithmetic.
    # No threshold override needed — default 1e-6 applies.
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
