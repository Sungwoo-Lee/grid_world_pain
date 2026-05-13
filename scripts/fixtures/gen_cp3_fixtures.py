"""gen_cp3_fixtures.py — deterministic fixture generator for CP3 tests.

Generates .npz fixture files for:
  1. zero_init_reward_head_input.npz — CP3: reward head output linear is zero-init
  2. zero_init_critic_head_input.npz — CP3: critic head output linear is zero-init

Both fixtures document that sheeprl applies uniform_init_weights(scale=0.0) to the
output linear layer of the reward model and critic model (cascade fix #27):

    world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
    critic.model[-1].apply(uniform_init_weights(0.0))

Source: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1175, L1172

With scale=0.0:
    limit = sqrt(3 * (0 / ((in + out) / 2))) = sqrt(0) = 0
    uniform(-0, 0) = 0  →  all-zeros kernel

So the PyTorch reference for both kernel and bias is identically zero.

The "fixtures" here are conceptually trivial (both sides must be all-zeros) but
the .npz stores the expected dimensions so the JAX test can construct the correct
module shape and verify zero-init byte-identically.

Shapes (matching sheeprl XS config):
  - Reward head output linear:  in_features = dense_units (default 512 in XS),
                                 out_features = num_bins (255 in DreamerV3)
  - Critic head output linear:  same shape (same dense_units and num_bins)

These are read from vendor/sheeprl sheeprl/configs to stay synchronized with
the sheeprl XS reference. If the XS config changes, re-run this script.

All fixtures use PRNG seed 0xD3EAF (decimal 868591) per fixture convention.

Run from the repo root in the sheeprl_bridge env (needs PyTorch):
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \\
        scripts/fixtures/gen_cp3_fixtures.py

This script is deterministic — re-running produces byte-identical .npz files.

Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1170-L1180
(build_agent final-init phase, hafner_initialization block).
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

# Import sheeprl's uniform_init_weights to run the PyTorch side
from sheeprl.algos.dreamer_v3.utils import uniform_init_weights  # noqa: E402

# ---------------------------------------------------------------------------
# Shapes from sheeprl XS config
# (vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml + exp/dreamer_v3.yaml)
#
# dense_units = 512  (default for all MLP heads in DreamerV3 XS)
# num_bins    = 255  (two-hot bins, DreamerV3 paper §B)
#
# The output linear of the reward model is:
#   world_model.reward_model.model[-1]  →  nn.Linear(dense_units, num_bins)
# The output linear of the critic is:
#   critic.model[-1]                    →  nn.Linear(dense_units, num_bins)
# ---------------------------------------------------------------------------
IN_FEATURES = 512   # dense_units
OUT_FEATURES = 255  # num_bins

# ---------------------------------------------------------------------------
# Fixture 1: zero_init_reward_head_input.npz
#
# sheeprl line 1175:
#   world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
#
# uniform_init_weights(scale=0.0) returns a weight-init function that sets
# weight to uniform(-limit, limit) where limit=sqrt(3*0)=0 → all-zeros.
# Bias is also set to all-zeros (the module's bias initialization path).
# ---------------------------------------------------------------------------
print("Generating fixture 1: zero_init_reward_head_input.npz")

# Create a fresh nn.Linear with random init, then apply zero-init
torch.manual_seed(SEED)
reward_linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)

# Store the pre-init (random) kernel for documentation purposes
pre_kernel_np = reward_linear.weight.detach().cpu().numpy().T  # [IN, OUT] col-major
pre_bias_np   = reward_linear.bias.detach().cpu().numpy()

# Apply sheeprl's zero-init (cascade fix #27)
# uniform_init_weights(0.0) returns an init function
# It calls nn.init.uniform_(weight, -limit, limit) where limit=sqrt(0)=0
reward_linear.apply(uniform_init_weights(0.0))
reward_linear.eval()

torch_kernel_np = reward_linear.weight.detach().cpu().numpy().T  # [IN, OUT]
torch_bias_np   = reward_linear.bias.detach().cpu().numpy()       # [OUT]

# Verify: both must be all-zeros
assert np.all(torch_kernel_np == 0.0), f"Reward kernel not zero after zero-init: max={np.max(np.abs(torch_kernel_np))}"
assert np.all(torch_bias_np == 0.0),   f"Reward bias not zero after zero-init: max={np.max(np.abs(torch_bias_np))}"
print(f"  reward linear shape: weight {reward_linear.weight.shape}, bias {reward_linear.bias.shape}")
print(f"  kernel max_abs after zero-init: {np.max(np.abs(torch_kernel_np)):.6f}  (expect 0.0)")
print(f"  bias   max_abs after zero-init: {np.max(np.abs(torch_bias_np)):.6f}    (expect 0.0)")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "zero_init_reward_head_input.npz"),
    # Dimensions
    in_features=np.int32(IN_FEATURES),
    out_features=np.int32(OUT_FEATURES),
    # Reference outputs from sheeprl PyTorch (both all-zeros)
    torch_kernel=torch_kernel_np,   # [IN, OUT] — col-major (JAX convention)
    torch_bias=torch_bias_np,       # [OUT]
    # Metadata
    seed=np.array(SEED),
    sheeprl_source=np.bytes_(b"vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1175"),
)
print(f"  Saved zero_init_reward_head_input.npz")

# ---------------------------------------------------------------------------
# Fixture 2: zero_init_critic_head_input.npz
#
# sheeprl line 1172:
#   critic.model[-1].apply(uniform_init_weights(0.0))
#
# Same shape as the reward head (same dense_units and num_bins in XS config).
# ---------------------------------------------------------------------------
print("\nGenerating fixture 2: zero_init_critic_head_input.npz")

torch.manual_seed(SEED)
critic_linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)

pre_kernel_c_np = critic_linear.weight.detach().cpu().numpy().T
pre_bias_c_np   = critic_linear.bias.detach().cpu().numpy()

critic_linear.apply(uniform_init_weights(0.0))
critic_linear.eval()

torch_kernel_c_np = critic_linear.weight.detach().cpu().numpy().T  # [IN, OUT]
torch_bias_c_np   = critic_linear.bias.detach().cpu().numpy()       # [OUT]

assert np.all(torch_kernel_c_np == 0.0), f"Critic kernel not zero: max={np.max(np.abs(torch_kernel_c_np))}"
assert np.all(torch_bias_c_np == 0.0),   f"Critic bias not zero: max={np.max(np.abs(torch_bias_c_np))}"
print(f"  critic linear shape: weight {critic_linear.weight.shape}, bias {critic_linear.bias.shape}")
print(f"  kernel max_abs after zero-init: {np.max(np.abs(torch_kernel_c_np)):.6f}  (expect 0.0)")
print(f"  bias   max_abs after zero-init: {np.max(np.abs(torch_bias_c_np)):.6f}    (expect 0.0)")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "zero_init_critic_head_input.npz"),
    # Dimensions
    in_features=np.int32(IN_FEATURES),
    out_features=np.int32(OUT_FEATURES),
    # Reference outputs from sheeprl PyTorch (both all-zeros)
    torch_kernel=torch_kernel_c_np,   # [IN, OUT]
    torch_bias=torch_bias_c_np,       # [OUT]
    # Metadata
    seed=np.array(SEED),
    sheeprl_source=np.bytes_(b"vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1172"),
)
print(f"  Saved zero_init_critic_head_input.npz")

print(f"\nAll CP3 fixtures generated successfully in {FIXTURE_DIR}")
