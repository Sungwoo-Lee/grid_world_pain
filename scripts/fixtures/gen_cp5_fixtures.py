"""gen_cp5_fixtures.py — deterministic fixture generator for CP5 (loss.py) tests.

Generates .npz fixture files for all 3 CP5 functions:
  1. twohot_bins_endpoints  — 255-bin linspace(-20, +20, 255) endpoint check
  2. twohot_encode          — two-hot distribution over bins for a batch of targets
  3. twohot_log_prob        — log-prob computation for a fixed batch

Each fixture stores:
  - The raw input tensors (for documentation / re-generation)
  - The PRE-COMPUTED sheeprl/PyTorch reference output (torch_out_*)
    so the pytest test in grid_world_pain (JAX-only env) can compare against
    stored reference values without needing PyTorch at test time.

All fixtures use PRNG seed 0xD3EAF (decimal 868591) per the fixture convention
in tests/fixtures/dreamer_srl/README.md.

Run from the repo root in the sheeprl_bridge env (needs both PyTorch + JAX):
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \\
        scripts/fixtures/gen_cp5_fixtures.py

This script is deterministic — re-running produces byte-identical .npz files.

CRITICAL — symlog-space discipline:
  The bin grid linspace(-20, 20, 255) is stored in SYMLOG SPACE — not real
  reward space. sheeprl applies transbwd (symexp) only at consumption (mean/mode),
  never to the stored bins. Targets are symlog-encoded BEFORE bin lookup (via
  transfwd = symlog inside log_prob). This fixture generator verifies the
  PyTorch-side reference uses the same discipline by calling TwoHotEncodingDistribution
  directly and storing its outputs.
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

from sheeprl.utils.distribution import TwoHotEncodingDistribution  # noqa: E402

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Shared config — matches sheeprl XS defaults at 33b6366
# ---------------------------------------------------------------------------
N_BINS = 255
LOW = -20
HIGH = 20
T, B = 4, 16   # batch shape: [T=4, B=16, 1] targets, [T=4, B=16, 255] logits
# Note: matches the plan's Lever A test description §CP5 "[T=4, B=16]"

# ---------------------------------------------------------------------------
# Fixture 1: twohot_bins_endpoints_input.npz
#
# The bin grid is a 255-element linspace(-20, +20, 255) in SYMLOG SPACE.
# sheeprl@33b6366:sheeprl/utils/distribution.py:L237
#   self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
#
# We store the expected endpoints so the JAX test can verify:
#   bins[0] == -20.0, bins[127] == 0.0, bins[254] == +20.0
# and the full grid shape/values for the max-abs-diff check.
# ---------------------------------------------------------------------------
print("Generating fixture 1: twohot_bins_endpoints_input.npz")

# Instantiate with dummy logits just to access .bins
dummy_logits = torch.zeros(1, N_BINS)
sheeprl_dist_bins = TwoHotEncodingDistribution(dummy_logits, dims=0, low=LOW, high=HIGH)
torch_bins = sheeprl_dist_bins.bins.detach().cpu().numpy()  # shape [255]

assert torch_bins[0] == -20.0, f"bins[0] expected -20.0, got {torch_bins[0]}"
# NOTE: bins[127] is the midpoint of linspace(-20, 20, 255) in float32. Due to
# float32 rounding, the midpoint is 7.45e-8, not exactly 0.0. The test asserts
# |bins[127]| < 1e-6 (within float32 linspace precision), not exact equality.
assert abs(torch_bins[127]) < 1e-6, f"bins[127] expected ~0.0, got {torch_bins[127]}"
assert torch_bins[254] == 20.0, f"bins[254] expected +20.0, got {torch_bins[254]}"

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "twohot_bins_endpoints_input.npz"),
    torch_bins=torch_bins,             # [255] full bin grid from sheeprl
    n_bins=np.int32(N_BINS),
    low=np.float32(LOW),
    high=np.float32(HIGH),
    expected_bins_0=np.float32(-20.0),
    # bins[127] is ~0.0 in float32 linspace (exact: 7.45e-8 from float32 rounding)
    expected_bins_127_abs_lt=np.float32(1e-6),  # |bins[127]| < this value
    expected_bins_254=np.float32(20.0),
    seed=np.array(SEED),
)
print(f"  twohot_bins_endpoints_input.npz  bins.shape={torch_bins.shape}  "
      f"bins[0]={torch_bins[0]}  bins[127]={torch_bins[127]} (abs<1e-6)  bins[254]={torch_bins[254]}")

# ---------------------------------------------------------------------------
# Fixture 2: twohot_encode_input.npz
#
# Encode a [T, B, 1] target batch to a [T, B, 255] two-hot distribution.
# The two-hot encoding (weight_below, weight_above over adjacent bins) is the
# output that the THIS IS THE TEST THAT CATCHES THE HISTORICAL BUG:
#   v1 stored bins in real reward space → bins[127] ≠ 0.0 → wrong bin lookup.
#   This test fails if bins are in real space because symlog(target) would
#   land on different bins.
#
# sheeprl@33b6366:sheeprl/utils/distribution.py:L253-L276 (log_prob internals)
#   transfwd = symlog applied to x before bin lookup.
#
# We compute two-hot targets by calling log_prob on both the original logits
# and on the known targets, then separately expose the two-hot encoding
# by isolating the target computation from the log_pred part.
# ---------------------------------------------------------------------------
print("Generating fixture 2: twohot_encode_input.npz")

# Random [T, B, 255] logits and [T, B, 1] reward targets
torch.manual_seed(SEED)
logits_np = rng.standard_normal((T, B, N_BINS)).astype(np.float32)
# Targets span a range that exercises both positive and negative symlog space
targets_np = rng.uniform(-5.0, 5.0, size=(T, B, 1)).astype(np.float32)

logits_torch = torch.tensor(logits_np)
targets_torch = torch.tensor(targets_np)

# dims=1: the event dimension (last non-bin dim) has size 1
sheeprl_dist = TwoHotEncodingDistribution(logits_torch, dims=1, low=LOW, high=HIGH)

# Extract the two-hot encoding by replicating the internal log_prob computation
# without the log_pred multiplication. This gives us the raw two-hot target.
# Porting sheeprl@33b6366:sheeprl/utils/distribution.py:L253-L274
import torch.nn.functional as F
from sheeprl.utils.utils import symlog as sheeprl_symlog

x_t = sheeprl_symlog(targets_torch)  # symlog-encode target first
bins = sheeprl_dist.bins
# below in [-1, len(bins) - 1]
below = (bins <= x_t).type(torch.int32).sum(dim=-1, keepdim=True) - 1
above = below + 1
# clamp
above = torch.minimum(above, torch.full_like(above, len(bins) - 1))
below = torch.maximum(below, torch.zeros_like(below))
equal = below == above
dist_to_below = torch.where(equal, torch.ones_like(x_t), torch.abs(bins[below] - x_t))
dist_to_above = torch.where(equal, torch.ones_like(x_t), torch.abs(bins[above] - x_t))
total = dist_to_below + dist_to_above
weight_below = dist_to_above / total
weight_above = dist_to_below / total
twohot_target = (
    F.one_hot(below, len(bins)) * weight_below[..., None]
    + F.one_hot(above, len(bins)) * weight_above[..., None]
).squeeze(-2)

twohot_target_np = twohot_target.detach().cpu().numpy()  # [T, B, 255]
assert twohot_target_np.shape == (T, B, N_BINS), f"Expected [{T},{B},{N_BINS}], got {twohot_target_np.shape}"

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "twohot_encode_input.npz"),
    logits=logits_np,            # [T, B, 255]  — not strictly needed for encode test
    targets=targets_np,          # [T, B, 1]   — the raw reward targets (pre-symlog)
    torch_out_twohot=twohot_target_np,  # [T, B, 255] two-hot from sheeprl
    n_bins=np.int32(N_BINS),
    low=np.float32(LOW),
    high=np.float32(HIGH),
    seed=np.array(SEED),
)
print(f"  twohot_encode_input.npz  logits.shape={logits_np.shape}  targets.shape={targets_np.shape}  "
      f"twohot.shape={twohot_target_np.shape}")

# ---------------------------------------------------------------------------
# Fixture 3: twohot_log_prob_input.npz
#
# Full log_prob computation: log_prob(target) for the [T, B, 1] target batch.
# This exercises both the encoding (fixture 2 prerequisite) AND the log-prob
# computation (cross-entropy of two-hot target against softmax of logits).
#
# sheeprl@33b6366:sheeprl/utils/distribution.py:L253-L276
#   log_prob returns (target * log_pred).sum(dim=self.dims)
# where log_pred = logits - logsumexp(logits) and target is the two-hot encoding.
# ---------------------------------------------------------------------------
print("Generating fixture 3: twohot_log_prob_input.npz")

# Reuse same logits + targets from fixture 2
torch_log_prob = sheeprl_dist.log_prob(targets_torch)  # [T, B] — dims=1 reduces event dim
torch_log_prob_np = torch_log_prob.detach().cpu().numpy()  # [T, B]
assert torch_log_prob_np.shape == (T, B), f"Expected [{T},{B}], got {torch_log_prob_np.shape}"

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "twohot_log_prob_input.npz"),
    logits=logits_np,            # [T, B, 255]
    targets=targets_np,          # [T, B, 1]
    torch_out_log_prob=torch_log_prob_np,   # [T, B]
    n_bins=np.int32(N_BINS),
    low=np.float32(LOW),
    high=np.float32(HIGH),
    seed=np.array(SEED),
)
print(f"  twohot_log_prob_input.npz  logits.shape={logits_np.shape}  targets.shape={targets_np.shape}  "
      f"log_prob.shape={torch_log_prob_np.shape}  "
      f"log_prob_range=[{torch_log_prob_np.min():.4f}, {torch_log_prob_np.max():.4f}]")

print(f"\nAll CP5 fixtures generated successfully in {FIXTURE_DIR}")
