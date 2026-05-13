"""gen_cp6_fixtures.py — deterministic fixture generator for CP6 (train.py) tests.

Generates .npz fixture files for the 3 CP6 functions:
  1. critic_loss_two_terms   — two-term critic NLL loss (lambda-target + EMA self-reg)
  2. critic_target_lambda    — verify critic regresses against UN-normalised lambda values
  3. discount_weighting      — cumprod(continues * gamma, axis=0) / gamma; [0]=1 invariant

Each fixture stores:
  - The raw input tensors (for documentation / re-generation)
  - The PRE-COMPUTED sheeprl/PyTorch reference output (torch_out_*)
    so the pytest test in grid_world_pain (JAX-only env) can compare against
    stored reference values without needing PyTorch at test time.

All fixtures use PRNG seed 0xD3EAF (decimal 868591) per the fixture convention
in tests/fixtures/dreamer_srl/README.md.

Run from the repo root in the sheeprl_bridge env (needs both PyTorch + JAX):
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \\
        scripts/fixtures/gen_cp6_fixtures.py

This script is deterministic — re-running produces byte-identical .npz files.

Sheeprl source reference:
  vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L240-L320 (train body)
  Specifically:
    L259-L260: discount cumprod
    L307-L316: critic loss two-term (qv.log_prob × 2) + discount weighting
    L246:      Independent(BernoulliSafeMode(...), 1).mode for continues

CP6 deviations to expect:
  - D-006 class: TwoHotEncoding float32 linspace ULP drift cascades to log_prob
    (bins midpoint differs by 1 ULP). Threshold relaxed to 3e-5 (same class as CP5).
  - D-010+ if additional platform ULP drift found during measurement.
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
import torch.nn.functional as F
torch.manual_seed(SEED)

from sheeprl.utils.distribution import TwoHotEncodingDistribution, BernoulliSafeMode  # noqa: E402
from torch.distributions import Independent  # noqa: E402

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Shared config — matches sheeprl XS defaults at 33b6366
# ---------------------------------------------------------------------------
N_BINS = 255       # two-hot bins
H = 15             # imagination horizon (sheeprl XS: horizon=15)
BT = 16            # batch_size * seq_len (4 * 4 for fixture — small but representative)
GAMMA = 0.99       # discount factor (sheeprl dreamer_v3_XL.yaml: gamma=0.99)
LOW = -20
HIGH = 20

print(f"CP6 fixture generator: H={H}, BT={BT}, gamma={GAMMA}, seed=0x{SEED:X}")
print(f"Fixture directory: {FIXTURE_DIR}")
print()

# ---------------------------------------------------------------------------
# Fixture 1: critic_loss_two_terms_input.npz
#
# Verifies: TWO-TERM critic NLL loss (cascade fix #29):
#   value_loss = -qv.log_prob(lambda_values.detach())
#                - qv.log_prob(predicted_target_values.detach())
#   value_loss = mean(value_loss * discount[:-1].squeeze(-1))
#
# The fixture stores:
#   - qv_logits:      [H, BT, 255]  — logits for the critic head
#   - lambda_values:  [H, BT, 1]    — un-normalised lambda-return targets
#   - target_values:  [H, BT, 1]    — EMA target-critic values (stop_gradient'd)
#   - discount:       [H, BT]       — pre-computed discount[:-1].squeeze(-1) (stop_gradient)
#   - torch_out_value_loss: scalar  — the full two-term discount-weighted loss
#   - torch_out_lp1:  [H, BT]      — -qv.log_prob(lambda_values)
#   - torch_out_lp2:  [H, BT]      — -qv.log_prob(target_values)
#
# Sheeprl source: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L307-L316
# ---------------------------------------------------------------------------
print("Generating fixture 1: critic_loss_two_terms_input.npz")

torch.manual_seed(SEED)
qv_logits_np = rng.standard_normal((H, BT, N_BINS)).astype(np.float32)
# lambda_values: real-space scalar targets (range covers typical value function range)
lambda_values_np = rng.uniform(-5.0, 5.0, size=(H, BT, 1)).astype(np.float32)
# target_values: EMA target-critic mean output (also real-space)
target_values_np = rng.uniform(-5.0, 5.0, size=(H, BT, 1)).astype(np.float32)
# discount: [H, BT] — cumprod-based (see fixture 3 for full computation)
# For this fixture, generate a simple discount array with the correct structure
# discount[0] = 1.0, discount[k] = gamma^k (when continues=1 everywhere)
discount_np = np.array(
    [[GAMMA ** k for _ in range(BT)] for k in range(H)],
    dtype=np.float32,
)  # [H, BT], discount_np[0] = 1.0 (k=0: gamma^0 = 1)

qv_logits_t = torch.tensor(qv_logits_np)
lambda_values_t = torch.tensor(lambda_values_np)
target_values_t = torch.tensor(target_values_np)
discount_t = torch.tensor(discount_np)

# sheeprl L307: qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
qv = TwoHotEncodingDistribution(qv_logits_t, dims=1)

# sheeprl L314: value_loss = -qv.log_prob(lambda_values.detach())
lp1_t = qv.log_prob(lambda_values_t.detach())  # [H, BT]
# sheeprl L315: value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
lp2_t = qv.log_prob(target_values_t.detach())  # [H, BT]

# Combined (both negative log_probs)
neg_lp1 = -lp1_t  # [H, BT]  (first NLL term)
neg_lp2 = -lp2_t  # [H, BT]  (second NLL term — EMA self-reg)

# sheeprl L316: value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))
# Note: in our fixture, discount is already [H, BT] (pre-sliced and squeezed)
value_loss_t = torch.mean((neg_lp1 + neg_lp2) * discount_t)  # scalar

# Store reference outputs
torch_out_lp1_np = neg_lp1.detach().cpu().numpy()  # [H, BT]
torch_out_lp2_np = neg_lp2.detach().cpu().numpy()  # [H, BT]
torch_out_value_loss = float(value_loss_t.item())

print(f"  qv_logits.shape={qv_logits_np.shape}  lambda_values.shape={lambda_values_np.shape}")
print(f"  target_values.shape={target_values_np.shape}  discount.shape={discount_np.shape}")
print(f"  torch_out_lp1.shape={torch_out_lp1_np.shape}  torch_out_lp2.shape={torch_out_lp2_np.shape}")
print(f"  torch_out_value_loss={torch_out_value_loss:.6f}")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "critic_loss_two_terms_input.npz"),
    # Inputs
    qv_logits=qv_logits_np,             # [H, BT, 255]
    lambda_values=lambda_values_np,     # [H, BT, 1]  — real-space lambda-return targets
    target_values=target_values_np,     # [H, BT, 1]  — EMA target-critic mean (real-space)
    discount=discount_np,               # [H, BT]     — pre-computed discount[:-1].squeeze(-1)
    # Reference outputs
    torch_out_lp1=torch_out_lp1_np,    # [H, BT]  — -qv.log_prob(lambda_values)
    torch_out_lp2=torch_out_lp2_np,    # [H, BT]  — -qv.log_prob(target_values)
    torch_out_value_loss=np.float32(torch_out_value_loss),  # scalar
    # Metadata
    H=np.int32(H),
    BT=np.int32(BT),
    N_BINS=np.int32(N_BINS),
    gamma=np.float32(GAMMA),
    seed=np.array(SEED),
)
print(f"  DONE: critic_loss_two_terms_input.npz")
print()

# ---------------------------------------------------------------------------
# Fixture 2: critic_target_lambda_input.npz
#
# Verifies: critic regresses against UN-normalised lambda_values (NOT Moments-normalised)
#
# Sheeprl L251-L256: lambda_values = compute_lambda_values(...)
# Sheeprl L276:      offset, invscale = moments(lambda_values, fabric)
# Sheeprl L277:      normed_lambda_values = (lambda_values - offset) / invscale  ← actor uses this
# Sheeprl L314:      value_loss = -qv.log_prob(lambda_values.detach())            ← critic uses raw!
#
# The test: run log_prob on both raw lambda_values and a fake normed version,
# confirm the reference was computed on the raw version (by matching torch_out exactly).
#
# This fixture directly shows the log_prob of the raw (un-normalised) target
# vs. what a normalised target would give, so the JAX test can assert:
#   jax_lp_raw matches torch_out (not jax_lp_normed)
#
# Sheeprl source: dreamer_v3.py:L251-L256, L276-L279, L314
# ---------------------------------------------------------------------------
print("Generating fixture 2: critic_target_lambda_input.npz")

torch.manual_seed(SEED + 1)
rng2 = np.random.default_rng(SEED + 1)

qv_logits_np2 = rng2.standard_normal((H, BT, N_BINS)).astype(np.float32)
lambda_values_np2 = rng2.uniform(-5.0, 5.0, size=(H, BT, 1)).astype(np.float32)

# Fake Moments normalization: offset = mean, invscale = 1/(std + eps)
lv_flat = lambda_values_np2.ravel()
offset = np.float32(lv_flat.mean())
invscale = np.float32(1.0 / max(1.0, float(np.std(lv_flat))))
normed_lambda_np2 = ((lambda_values_np2 - offset) * invscale).astype(np.float32)  # [H, BT, 1]

qv2 = TwoHotEncodingDistribution(torch.tensor(qv_logits_np2), dims=1)

# sheeprl uses RAW lambda_values (not normed)
lp_raw_t = qv2.log_prob(torch.tensor(lambda_values_np2).detach())    # [H, BT]
lp_normed_t = qv2.log_prob(torch.tensor(normed_lambda_np2).detach()) # [H, BT] — what WRONG impl would use

torch_out_lp_raw_np = lp_raw_t.detach().cpu().numpy()    # [H, BT]
torch_out_lp_normed_np = lp_normed_t.detach().cpu().numpy()  # [H, BT]

print(f"  qv_logits.shape={qv_logits_np2.shape}  lambda_values.shape={lambda_values_np2.shape}")
print(f"  normed_lambda.shape={normed_lambda_np2.shape}")
print(f"  log_prob_raw range=[{torch_out_lp_raw_np.min():.4f}, {torch_out_lp_raw_np.max():.4f}]")
print(f"  log_prob_normed range=[{torch_out_lp_normed_np.min():.4f}, {torch_out_lp_normed_np.max():.4f}]")
# These should differ — confirm the test will distinguish them
diff = float(np.max(np.abs(torch_out_lp_raw_np - torch_out_lp_normed_np)))
print(f"  max_abs_diff(raw vs normed) = {diff:.4e}  (should be >> 1e-4 to distinguish)")
assert diff > 1e-4, f"Raw vs normed lambda values are too similar ({diff:.4e}); fixture won't distinguish them"

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "critic_target_lambda_input.npz"),
    # Inputs
    qv_logits=qv_logits_np2,            # [H, BT, 255]
    lambda_values=lambda_values_np2,    # [H, BT, 1]  — raw lambda-return targets (un-normalised)
    normed_lambda=normed_lambda_np2,    # [H, BT, 1]  — Moments-normalised (what WRONG impl uses)
    offset=np.float32(offset),
    invscale=np.float32(invscale),
    # Reference output (raw lambda — this is what sheeprl uses)
    torch_out_lp_raw=torch_out_lp_raw_np,         # [H, BT]  — correct: log_prob of raw targets
    torch_out_lp_normed=torch_out_lp_normed_np,   # [H, BT]  — wrong:  log_prob of normed targets
    # Metadata
    H=np.int32(H),
    BT=np.int32(BT),
    N_BINS=np.int32(N_BINS),
    seed=np.array(SEED + 1),
)
print(f"  DONE: critic_target_lambda_input.npz")
print()

# ---------------------------------------------------------------------------
# Fixture 3: discount_weighting_input.npz
#
# Verifies: discount = cumprod(continues * gamma, axis=0) / gamma
#   - shape [H+1, BT, 1]
#   - discount[0] = continues[0] (= 1.0 when no termination at step 0)
#   - The [0]=1 invariant holds when continues[0]=1 (true_continue from buffer, §S5)
#   - discount[:-1].squeeze(-1) has shape [H, BT]
#
# Sheeprl source:
#   dreamer_v3.py:L259-L260 (inside torch.no_grad()):
#     discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma
#   dreamer_v3.py:L246-L248: continues construction:
#     continues = Independent(BernoulliSafeMode(logits=...), 1).mode  # [H+1, BT, 1]
#     true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)  # [1, BT, 1]
#     continues = torch.cat((true_continue, continues[1:]))  # §S5 true-continue splice
#
# The fixture tests:
#   1. discount[0] == 1.0 (when continues[0]=1, i.e., no termination at imagination start)
#   2. discount has the cumprod form
#   3. discount[:-1].squeeze(-1) shape = [H, BT]
#
# ---------------------------------------------------------------------------
print("Generating fixture 3: discount_weighting_input.npz")

torch.manual_seed(SEED + 2)
rng3 = np.random.default_rng(SEED + 2)

# Simulate continues[0] = 1 (true_continue = 1 - terminated, all not terminated)
# continues[1:] are imagined continues from the world model's continue head
# For the fixture: continues[0] = 1 (no termination); continues[1:] = random continue probs ≥ 0
continues_imagined = rng3.uniform(0.7, 1.0, size=(H, BT, 1)).astype(np.float32)  # [H, BT, 1]
true_continue = np.ones((1, BT, 1), dtype=np.float32)                              # [1, BT, 1]
continues_np = np.concatenate([true_continue, continues_imagined], axis=0)         # [H+1, BT, 1]

# sheeprl L259-L260 (stop_gradient = torch.no_grad equivalent):
# discount = torch.cumprod(continues * gamma, dim=0) / gamma
continues_t = torch.tensor(continues_np)
discount_full_t = torch.cumprod(continues_t * GAMMA, dim=0) / GAMMA  # [H+1, BT, 1]
discount_sliced_t = discount_full_t[:-1].squeeze(-1)                  # [H, BT]

discount_full_np = discount_full_t.detach().cpu().numpy()   # [H+1, BT, 1]
discount_sliced_np = discount_sliced_t.detach().cpu().numpy()  # [H, BT]

# Verify invariants
assert discount_full_np[0].mean() > 0.999, f"discount[0] not ≈ 1.0: {discount_full_np[0].mean()}"
assert discount_sliced_np.shape == (H, BT), f"discount[:-1].squeeze(-1) wrong shape: {discount_sliced_np.shape}"

print(f"  continues.shape={continues_np.shape}  (continues[0]=1.0 — no termination)")
print(f"  discount_full.shape={discount_full_np.shape}  discount_sliced.shape={discount_sliced_np.shape}")
print(f"  discount[0].mean()={discount_full_np[0].mean():.6f}  (should be 1.0)")
print(f"  discount[1].mean()={discount_full_np[1].mean():.6f}  (should be ~continues_imagined[0]*gamma)")
print(f"  discount[-1].mean()={discount_full_np[-1].mean():.6f}  (heavily discounted)")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "discount_weighting_input.npz"),
    # Inputs
    continues=continues_np,             # [H+1, BT, 1]  — §S5-splice continues
    gamma=np.float32(GAMMA),
    # Reference outputs
    torch_out_discount_full=discount_full_np,    # [H+1, BT, 1]
    torch_out_discount_sliced=discount_sliced_np, # [H, BT]
    # Metadata / invariant assertions
    expected_discount_0=np.float32(1.0),         # discount[0] should be 1.0 when continues[0]=1
    H=np.int32(H),
    BT=np.int32(BT),
    gamma_stored=np.float32(GAMMA),
    seed=np.array(SEED + 2),
)
print(f"  DONE: discount_weighting_input.npz")
print()

print("=" * 60)
print(f"All CP6 fixtures generated successfully in {FIXTURE_DIR}")
print("Next: run gen_cp6_fixtures.py from the repo root with the grid_world_pain env")
print("to verify JAX implementations match these PyTorch references.")
