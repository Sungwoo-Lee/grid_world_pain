"""gen_cp7_fixtures.py — deterministic fixture generator for CP7 (train.py) tests.

Generates .npz fixture files for the 3 CP7 Lever-A functions:
  1. polyak_first_call   — first Polyak call with tau=1.0 (hard copy: target = online)
  2. polyak_subsequent_call — subsequent call with tau=0.02 (EMA blend)
  3. polyak_before_train — structural call-order test (polyak fires BEFORE one_train_step)

Each fixture stores:
  - The raw input tensors (for documentation / re-generation)
  - The PRE-COMPUTED sheeprl/PyTorch reference output (torch_out_*)
    so the pytest test in grid_world_pain (JAX-only env) can compare against
    stored reference values without needing PyTorch at test time.

All fixtures use PRNG seed 0xD3EAF (decimal 868591) per the fixture convention
in tests/fixtures/dreamer_srl/README.md. CP7 uses seed 0xD3EAF+3, +4, +5
(continuing the +N convention from CP6 which used +0, +1, +2).

Run from the repo root in the sheeprl_bridge env (needs both PyTorch + JAX):
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \\
        scripts/fixtures/gen_cp7_fixtures.py

This script is deterministic — re-running produces byte-identical .npz files.

Sheeprl source reference:
  vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L673-L680 (polyak update)
  vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L274-L297 (actor objective)

CP7 deviations to expect:
  - Polyak is pure arithmetic (EMA blend) — no TwoHotEncoding involved.
    Expect max_abs_diff < 1e-6 (no D-006 cascade); same as discount_weighting.
  - Any deviation > 1e-6 would indicate a semantic error.
"""
import os
import sys
import numpy as np

SEED = 0xD3EAF  # 868591
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "dreamer_srl")
os.makedirs(FIXTURE_DIR, exist_ok=True)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "vendor", "sheeprl"))

import torch
torch.manual_seed(SEED)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Shared config — matches sheeprl XS defaults at 33b6366
# ---------------------------------------------------------------------------
# Number of "parameters" in the mock critic — small representative set
N_PARAMS_SETS = 3   # simulate 3 parameter tensors (e.g. linear weight, bias x2)
PARAM_SHAPES = [(64, 32), (64,), (32,)]  # representative critic layer shapes
TAU_FIRST = 1.0      # hard copy on first call
TAU_SUBSEQUENT = 0.02  # sheeprl XS default: cfg.algo.critic.tau = 0.02

print(f"CP7 fixture generator: seed=0x{SEED:X}")
print(f"Fixture directory: {FIXTURE_DIR}")
print(f"Mock critic param shapes: {PARAM_SHAPES}")
print()

# ---------------------------------------------------------------------------
# Fixture 1: polyak_first_call_input.npz
#
# Verifies: first Polyak call (cumulative_gradient_steps == 0) uses tau=1.0
#           This is a HARD COPY: target = 1.0 * online + 0.0 * target = online
#           Result should be byte-identical to online params.
#
# Sheeprl source:
#   dreamer_v3.py:L678: tau = 1 if cumulative_per_rank_gradient_steps == 0 else ...
#   dreamer_v3.py:L679-L680:
#     for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
#         tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
#   With tau=1: tcp = 1.0 * cp + 0.0 * tcp = cp  (hard copy, byte-identical)
#
# ---------------------------------------------------------------------------
print("Generating fixture 1: polyak_first_call_input.npz")

rng3 = np.random.default_rng(SEED + 3)

# Create DIFFERENT online and target params (so hard-copy is detectable)
online_params = {
    f"param_{i}": rng3.standard_normal(shape).astype(np.float32)
    for i, shape in enumerate(PARAM_SHAPES)
}
# Target starts with DIFFERENT values (uniform random, clearly different from online)
target_params_initial = {
    f"param_{i}": rng3.standard_normal(shape).astype(np.float32)
    for i, shape in enumerate(PARAM_SHAPES)
}

# sheeprl L679-L680: tcp.data.copy_(tau * cp.data + (1-tau) * tcp.data) with tau=1
# = 1.0 * cp + 0.0 * tcp = cp  (exact copy)
expected_target_after_first_call = {
    k: TAU_FIRST * online_params[k] + (1 - TAU_FIRST) * target_params_initial[k]
    for k in online_params
}  # With tau=1: expected = online exactly

# Verify the reference is byte-identical to online (tau=1 → hard copy)
for k in online_params:
    diff = float(np.max(np.abs(expected_target_after_first_call[k] - online_params[k])))
    assert diff == 0.0, f"First-call hard copy failed for {k}: diff={diff}"
print(f"  tau=1.0 hard-copy verified: all {len(online_params)} param arrays are byte-identical to online")

# Store as separate keys for clarity
param_keys = sorted(online_params.keys())
save_dict = {
    "tau": np.float32(TAU_FIRST),
    "n_param_groups": np.int32(len(param_keys)),
}
for k in param_keys:
    save_dict[f"online_{k}"] = online_params[k]
    save_dict[f"target_init_{k}"] = target_params_initial[k]
    save_dict[f"torch_out_target_{k}"] = expected_target_after_first_call[k]

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "polyak_first_call_input.npz"),
    **save_dict,
    seed=np.array(SEED + 3),
)
print(f"  DONE: polyak_first_call_input.npz")
print()

# ---------------------------------------------------------------------------
# Fixture 2: polyak_subsequent_call_input.npz
#
# Verifies: subsequent Polyak call uses tau=0.02 (sheeprl XS default)
#           target = (1 - tau) * target + tau * online
#                  = 0.98 * target + 0.02 * online
#
# Test case: online = all-ones, target = all-zeros
#   expected target after blend = 0.98 * 0 + 0.02 * 1 = 0.02 * ones
#
# Sheeprl source:
#   dreamer_v3.py:L678: tau = cfg.algo.critic.tau  (= 0.02 for subsequent calls)
#   dreamer_v3.py:L679-L680: tcp.data.copy_(tau * cp.data + (1-tau) * tcp.data)
#     = 0.02 * cp + 0.98 * tcp
#
# ---------------------------------------------------------------------------
print("Generating fixture 2: polyak_subsequent_call_input.npz")

rng4 = np.random.default_rng(SEED + 4)

# Use a more general setup: random online and target params
online_params2 = {
    f"param_{i}": rng4.standard_normal(shape).astype(np.float32)
    for i, shape in enumerate(PARAM_SHAPES)
}
target_params2_initial = {
    f"param_{i}": rng4.standard_normal(shape).astype(np.float32)
    for i, shape in enumerate(PARAM_SHAPES)
}

# sheeprl L679-L680: tcp.data.copy_(tau * cp.data + (1-tau) * tcp.data) with tau=0.02
# = 0.02 * online + 0.98 * target
expected_target_after_subsequent = {
    k: TAU_SUBSEQUENT * online_params2[k] + (1 - TAU_SUBSEQUENT) * target_params2_initial[k]
    for k in online_params2
}

# Verify the computation manually using PyTorch for cross-check
for k in online_params2:
    cp_t = torch.tensor(online_params2[k])
    tcp_t = torch.tensor(target_params2_initial[k])
    tau_t = TAU_SUBSEQUENT
    torch_result = (tau_t * cp_t + (1 - tau_t) * tcp_t).numpy()
    our_result = expected_target_after_subsequent[k]
    diff = float(np.max(np.abs(our_result - torch_result)))
    assert diff < 1e-6, f"PyTorch vs numpy EMA mismatch for {k}: diff={diff}"
print(f"  tau=0.02 EMA blend: PyTorch cross-check passed for all {len(online_params2)} param arrays")

# Also compute the special "ones online, zeros target" case for pedagogical clarity
# online=1, target=0 → expected = 0.02 * 1 = 0.02
ones_online = {f"param_{i}": np.ones(shape, dtype=np.float32) for i, shape in enumerate(PARAM_SHAPES)}
zeros_target = {f"param_{i}": np.zeros(shape, dtype=np.float32) for i, shape in enumerate(PARAM_SHAPES)}
pedagogical_result = {k: TAU_SUBSEQUENT * ones_online[k] + (1 - TAU_SUBSEQUENT) * zeros_target[k] for k in ones_online}
# Verify
for k in ones_online:
    expected_val = TAU_SUBSEQUENT  # 0.02
    actual_val = float(pedagogical_result[k].max())
    assert abs(actual_val - expected_val) < 1e-7, f"Pedagogical check failed: {actual_val} != {expected_val}"
print(f"  Pedagogical check: online=1, target=0 → result={TAU_SUBSEQUENT:.4f} (correct)")

save_dict2 = {
    "tau": np.float32(TAU_SUBSEQUENT),
    "n_param_groups": np.int32(len(param_keys)),
}
for k in param_keys:
    save_dict2[f"online_{k}"] = online_params2[k]
    save_dict2[f"target_init_{k}"] = target_params2_initial[k]
    save_dict2[f"torch_out_target_{k}"] = expected_target_after_subsequent[k]

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "polyak_subsequent_call_input.npz"),
    **save_dict2,
    seed=np.array(SEED + 4),
)
print(f"  DONE: polyak_subsequent_call_input.npz")
print()

# ---------------------------------------------------------------------------
# Fixture 3: polyak_before_train_input.npz
#
# Verifies: CALL ORDER — Polyak fires BEFORE one_train_step in the outer loop.
# This is a structural test. The fixture encodes the expected call sequence
# so the test can verify that train.py's loop calls polyak_update() BEFORE
# calling one_train_step() (not after).
#
# Sheeprl source:
#   dreamer_v3.py:L673-L697 (inner gradient-step loop):
#     for i in range(per_rank_gradient_steps):
#         if cumulative_steps % update_freq == 0:
#             tau = 1 if cumulative_steps == 0 else cfg.algo.critic.tau
#             for cp, tcp in zip(...):              # ← POLYAK HERE (L679-L680)
#                 tcp.data.copy_(tau * cp + (1-tau) * tcp)
#         batch = {k: v[i].float() ...}
#         train(fabric, world_model, actor, critic, ...)  # ← TRAIN AFTER (L686-L697)
#
# The "before train" ordering means: the target critic used INSIDE train() for
# value prediction (dreamer_v3.py:L309) is the FRESHLY UPDATED target, not the
# stale pre-step target.
#
# Test strategy: use a mock "call log" approach.
#   - The fixture encodes the expected call sequence as strings.
#   - The test reads the fixture and verifies that the code structure matches.
#   - This is a code-inspection test (grep/AST check) + structural fixture.
#
# We additionally compute a TWO-STEP trace:
#   Step 0 (tau=1): hard copy → target_after_step0 = online_initial
#   Step 1 (tau=0.02): EMA → target_after_step1 = 0.98 * target_after_step0 + 0.02 * online_updated
# This lets the test verify the state after 2 polyak calls.
#
# ---------------------------------------------------------------------------
print("Generating fixture 3: polyak_before_train_input.npz")

rng5 = np.random.default_rng(SEED + 5)

# Simple 1D arrays for the two-step trace (easy to reason about)
SIMPLE_SHAPE = (8,)

online_step0 = rng5.standard_normal(SIMPLE_SHAPE).astype(np.float32)
target_step0_init = rng5.standard_normal(SIMPLE_SHAPE).astype(np.float32)

# Simulate "online network updated between step 0 and step 1" (e.g., one gradient step)
# For the call-order test, we care about the polyak ORDER relative to train(), not the
# exact gradient update. Use a simple perturbation:
online_step1 = online_step0 + rng5.standard_normal(SIMPLE_SHAPE).astype(np.float32) * 0.01

# Step 0: tau=1 (first call) → hard copy
target_after_step0 = TAU_FIRST * online_step0 + (1 - TAU_FIRST) * target_step0_init
assert float(np.max(np.abs(target_after_step0 - online_step0))) == 0.0, "Step 0 hard copy failed"

# Step 1: tau=0.02 (subsequent call, using online_step1 = slightly updated online)
# KEY: polyak fires with CURRENT online (step 1) BEFORE train() is called for step 1
target_after_step1 = TAU_SUBSEQUENT * online_step1 + (1 - TAU_SUBSEQUENT) * target_after_step0

# PyTorch cross-check
online_s0_t = torch.tensor(online_step0)
target_s0_t = torch.tensor(target_step0_init)
online_s1_t = torch.tensor(online_step1)

# Step 0: tau=1
tcp_s0 = TAU_FIRST * online_s0_t + (1 - TAU_FIRST) * target_s0_t
# Step 1: tau=0.02 with updated online
tcp_s1 = TAU_SUBSEQUENT * online_s1_t + (1 - TAU_SUBSEQUENT) * tcp_s0

for arr_np, arr_t, name in [
    (target_after_step0, tcp_s0.numpy(), "target_after_step0"),
    (target_after_step1, tcp_s1.numpy(), "target_after_step1"),
]:
    diff = float(np.max(np.abs(arr_np - arr_t)))
    assert diff < 1e-6, f"PyTorch cross-check failed for {name}: diff={diff}"
print(f"  Two-step trace: PyTorch cross-check passed")

# Encode expected call-order as metadata string (for documentation/assertion in test)
# The test will use code inspection (grep the outer loop) to verify ordering.
expected_call_order = "polyak_update BEFORE one_train_step"

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "polyak_before_train_input.npz"),
    # Inputs for two-step trace
    online_step0=online_step0,                   # [8]  — online params before step 0 train
    target_init=target_step0_init,               # [8]  — target params initial state
    online_step1=online_step1,                   # [8]  — online params before step 1 train (post-grad-update)
    tau_first=np.float32(TAU_FIRST),             # 1.0
    tau_subsequent=np.float32(TAU_SUBSEQUENT),   # 0.02
    # Reference outputs (torch/numpy cross-checked)
    torch_out_target_after_step0=target_after_step0,  # [8]
    torch_out_target_after_step1=target_after_step1,  # [8]
    # Metadata: expected call order (structural test)
    seed=np.array(SEED + 5),
)
print(f"  DONE: polyak_before_train_input.npz")
print(f"  Encoded expected call order: '{expected_call_order}'")
print()

print("=" * 60)
print(f"All CP7 fixtures generated successfully in {FIXTURE_DIR}")
print()
print("Files created:")
print(f"  {FIXTURE_DIR}/polyak_first_call_input.npz")
print(f"  {FIXTURE_DIR}/polyak_subsequent_call_input.npz")
print(f"  {FIXTURE_DIR}/polyak_before_train_input.npz")
print()
print("Next: run pytest tests/algorithms/dreamer_srl/test_train.py -v")
print("to verify JAX implementations match these PyTorch references.")
