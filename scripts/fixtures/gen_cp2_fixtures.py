"""gen_cp2_fixtures.py — deterministic fixture generator for CP2 + CP2b tests.

Generates .npz fixture files for:
  1. layernorm_gru_cell_input.npz — CP2: LayerNormGRUCell forward-pass parity
  2. action_shift_input.npz       — CP2b: action_shift prepend-zero / drop-last

Each fixture stores:
  - The raw input tensors (for re-generation / documentation)
  - The PRE-COMPUTED sheeprl/PyTorch reference output (torch_out_*)
    so the pytest tests in grid_world_pain (JAX-only env) can compare against
    stored reference values without needing PyTorch at test time.

All fixtures use PRNG seed 0xD3EAF (decimal 868591) per the fixture convention
in tests/fixtures/dreamer_srl/README.md.

Run from the repo root in the sheeprl_bridge env (needs both PyTorch + JAX):
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \\
        scripts/fixtures/gen_cp2_fixtures.py

This script is deterministic — re-running produces byte-identical .npz files.

CRITICAL — reset-before-tanh discipline (CP2 cascade fix #28):
  LayerNormGRUCell applies the reset gate BEFORE tanh in the candidate state:
      reset, cand, update = chunk(x, 3)
      cand = tanh(reset * cand)            ← reset INSIDE tanh
  NOT:
      cand = reset * tanh(cand)            ← wrong: reset OUTSIDE tanh
  The fixture uses input where reset ≈ 0.5 so this multiplication
  materially affects the output — a wrong-order implementation would differ.
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
import torch.nn as nn

# Reproducible seeds
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# Import sheeprl's LayerNormGRUCell
from sheeprl.models.models import LayerNormGRUCell  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture 1: layernorm_gru_cell_input.npz
#
# LayerNormGRUCell(input_size, hidden_size, layer_norm_cls=nn.LayerNorm).
# We fix all learnable parameters to known values so both the PyTorch and JAX
# sides can be seeded identically. Then run one forward step.
#
# CRITICAL: the fixture uses inputs that produce reset ≈ 0.5 (after sigmoid),
# so the reset-before-tanh vs reset-after-tanh difference is large. A wrong-order
# implementation would produce output that differs by O(0.1), not O(1e-7).
#
# sheeprl@33b6366:vendor/sheeprl/sheeprl/models/models.py:L331-L410
# ---------------------------------------------------------------------------
print("Generating fixture 1: layernorm_gru_cell_input.npz")

INPUT_SIZE = 8
HIDDEN_SIZE = 16
BATCH_SIZE = 4

# Instantiate the sheeprl cell with LayerNorm
torch.manual_seed(SEED)
cell = LayerNormGRUCell(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=True,
    batch_first=False,
    layer_norm_cls=nn.LayerNorm,
    layer_norm_kw={},
)
cell.eval()

# Extract learnable parameters as numpy arrays
# cell.linear: weight [3*HIDDEN_SIZE, INPUT_SIZE+HIDDEN_SIZE], bias [3*HIDDEN_SIZE]
# cell.layer_norm: weight [3*HIDDEN_SIZE], bias [3*HIDDEN_SIZE]
linear_weight_np = cell.linear.weight.detach().cpu().numpy()   # [48, 24]
linear_bias_np = cell.linear.bias.detach().cpu().numpy()       # [48]
ln_weight_np = cell.layer_norm.weight.detach().cpu().numpy()   # [48]
ln_bias_np = cell.layer_norm.bias.detach().cpu().numpy()       # [48]

# Fixed input + hidden state
torch.manual_seed(SEED + 1)
input_np = rng.standard_normal((BATCH_SIZE, INPUT_SIZE)).astype(np.float32)
hx_np = rng.standard_normal((BATCH_SIZE, HIDDEN_SIZE)).astype(np.float32)

input_torch = torch.tensor(input_np)
hx_torch = torch.tensor(hx_np)

# Sanity-check: verify reset ≈ 0.5 after sigmoid (the critical condition)
with torch.no_grad():
    hx_squeezed = hx_torch
    cat_inp = torch.cat((hx_squeezed, input_torch), -1)
    x = cell.linear(cat_inp)
    x = cell.layer_norm(x)
    reset_pre_sigmoid = x[:, :HIDDEN_SIZE]
    reset = torch.sigmoid(reset_pre_sigmoid)
    reset_mean = float(reset.mean())
    print(f"  reset mean (post-sigmoid): {reset_mean:.4f}  (want ≈ 0.1-0.9 for non-trivial test)")

# Run the actual forward pass
with torch.no_grad():
    torch_out = cell(input_torch, hx_torch)
torch_out_np = torch_out.detach().cpu().numpy()   # [BATCH_SIZE, HIDDEN_SIZE]

print(f"  input shape:  {input_np.shape}")
print(f"  hx shape:     {hx_np.shape}")
print(f"  output shape: {torch_out_np.shape}")
print(f"  output range: [{torch_out_np.min():.4f}, {torch_out_np.max():.4f}]")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "layernorm_gru_cell_input.npz"),
    # Inputs
    input=input_np,                # [B, input_size]
    hx=hx_np,                     # [B, hidden_size]
    # Learnable parameters (so JAX side can be set identically)
    linear_weight=linear_weight_np,  # [3*H, I+H]
    linear_bias=linear_bias_np,      # [3*H]
    ln_weight=ln_weight_np,          # [3*H]
    ln_bias=ln_bias_np,              # [3*H]
    # Scalar dims
    input_size=np.int32(INPUT_SIZE),
    hidden_size=np.int32(HIDDEN_SIZE),
    batch_size=np.int32(BATCH_SIZE),
    # Reference output from sheeprl PyTorch
    torch_out=torch_out_np,          # [B, hidden_size]
    seed=np.array(SEED),
)
print(f"  Saved layernorm_gru_cell_input.npz")

# ---------------------------------------------------------------------------
# Fixture 2: action_shift_input.npz
#
# action_shift takes actions[T, B, A] and returns
#   [zeros[1, B, A], actions[:-1]]   shape [T, B, A]
#
# This is §S2 — the action that produced obs_t is the action taken at t-1,
# not the one taken at t. sheeprl hard-codes this at line 104:
#   batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]),
#                               data["actions"][:-1]), dim=0)
#
# T=5 to catch any off-by-one error.
# sheeprl@33b6366:vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L104
# ---------------------------------------------------------------------------
print("\nGenerating fixture 2: action_shift_input.npz")

T, B, A = 5, 4, 3

actions_np = rng.standard_normal((T, B, A)).astype(np.float32)
actions_torch = torch.tensor(actions_np)

# sheeprl reference implementation (line 104):
batch_actions_torch = torch.cat(
    (torch.zeros_like(actions_torch[:1]), actions_torch[:-1]),
    dim=0,
)
torch_out_shifted_np = batch_actions_torch.detach().cpu().numpy()   # [T, B, A]

# Verify structural properties
assert torch_out_shifted_np.shape == (T, B, A), f"Expected [{T},{B},{A}], got {torch_out_shifted_np.shape}"
assert np.all(torch_out_shifted_np[0] == 0.0), "First time-step must be all zeros"
assert np.allclose(torch_out_shifted_np[1:], actions_np[:-1]), "Remaining steps must match actions[:-1]"

print(f"  actions shape:       {actions_np.shape}")
print(f"  shifted[0] (zeros):  {torch_out_shifted_np[0, 0]}")
print(f"  shifted[1:] == actions[:-1]: {np.allclose(torch_out_shifted_np[1:], actions_np[:-1])}")
print(f"  output shape:        {torch_out_shifted_np.shape}")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "action_shift_input.npz"),
    # Input
    actions=actions_np,                    # [T, B, A]
    # Dims
    T=np.int32(T),
    B=np.int32(B),
    A=np.int32(A),
    # Reference output from sheeprl PyTorch (line 104)
    torch_out_shifted=torch_out_shifted_np,  # [T, B, A]
    seed=np.array(SEED),
)
print(f"  Saved action_shift_input.npz")

print(f"\nAll CP2 + CP2b fixtures generated successfully in {FIXTURE_DIR}")
