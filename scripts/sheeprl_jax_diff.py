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
# CP2 (agent.py):        layernorm_gru_cell, action_shift
# CP3 (agent.py):        zero_init_reward_head, zero_init_critic_head
# CP4 (agent.py):        rssm_transition, rssm_representation, get_initial_states,
#                         is_first_force_set, is_first_three_quantity_reset
# CP5 (loss.py):         twohot_bins_endpoints, twohot_encode, twohot_log_prob
# CP6 (train.py):        critic_loss_two_terms, critic_target_lambda, discount_weighting
# CP7 (train.py):        polyak_first_call, polyak_subsequent_call, polyak_before_train
# CP8 (end-to-end):      forward_parity (offline script, not registered here)
#
# Pre-CP0: registry is empty — only the CLI skeleton ships here.
# Populate incrementally as each CP lands.

FUNCTION_REGISTRY: dict[str, callable] = {
    # Example (uncomment when CP5 lands):
    # "twohot_encode": _run_twohot_encode,
}

# Maps checkpoint name → list of function names registered for that CP.
CHECKPOINT_REGISTRY: dict[str, list[str]] = {
    "CP1":  ["symlog", "symexp", "init_weights", "uniform_init_weights",
             "compute_lambda_values", "moments_update", "ratio", "prepare_obs"],
    "CP2":  ["layernorm_gru_cell", "action_shift"],
    "CP2b": ["action_shift"],
    "CP3":  ["zero_init_reward_head", "zero_init_critic_head"],
    "CP4":  ["rssm_transition", "rssm_representation", "get_initial_states",
             "is_first_force_set"],
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

def run_single(function_name: str, fixture_path: str, threshold: float) -> bool:
    """Run one function's comparison.  Returns True on PASS."""
    if function_name not in FUNCTION_REGISTRY:
        print(f"ERROR: '{function_name}' not yet in FUNCTION_REGISTRY.", file=sys.stderr)
        print(f"  Registered functions: {sorted(FUNCTION_REGISTRY.keys()) or '(none — pre-CP0 skeleton)'}", file=sys.stderr)
        print(f"  Add an entry in scripts/sheeprl_jax_diff.py when the function is ported.", file=sys.stderr)
        sys.exit(1)

    fixture = load_fixture(fixture_path)
    runner = FUNCTION_REGISTRY[function_name]
    jax_out, torch_out, metadata = runner(fixture)
    _, passed = compare(jax_out, torch_out, threshold, metadata)
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
        _, passed = compare(jax_out, torch_out, threshold, metadata)
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
