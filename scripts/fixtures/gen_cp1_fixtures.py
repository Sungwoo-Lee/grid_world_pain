"""gen_cp1_fixtures.py — deterministic fixture generator for CP1 (utils.py) tests.

Generates .npz fixture files for all 7 CP1 functions:
  symlog, symexp, init_weights, uniform_init_weights,
  compute_lambda_values, moments_update, ratio, prepare_obs

Each fixture stores:
  - The raw input tensors (for documentation / re-generation)
  - The PRE-COMPUTED sheeprl/PyTorch reference output (torch_out_*)
    so the pytest test in grid_world_pain (JAX-only env) can compare against
    stored reference values without needing PyTorch at test time.

All fixtures use PRNG seed 0xD3EAF (decimal 868591) per the fixture convention
in tests/fixtures/dreamer_srl/README.md.

Run from the repo root in the sheeprl_bridge env (needs both PyTorch + JAX):
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \
        scripts/fixtures/gen_cp1_fixtures.py

This script is deterministic — re-running produces byte-identical .npz files.
"""
import os
import sys
import warnings
import numpy as np

SEED = 0xD3EAF  # 868591
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "dreamer_srl")
os.makedirs(FIXTURE_DIR, exist_ok=True)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "vendor", "sheeprl"))

import torch
torch.manual_seed(SEED)

from sheeprl.utils.utils import symlog as sheeprl_symlog
from sheeprl.utils.utils import symexp as sheeprl_symexp
from sheeprl.algos.dreamer_v3.utils import compute_lambda_values as sheeprl_clv
from sheeprl.utils.utils import Ratio as SheeprlRatio

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# symlog_input.npz
# ---------------------------------------------------------------------------
x_symlog = rng.uniform(-100.0, 100.0, size=(16, 4)).astype(np.float32)
torch_symlog_out = sheeprl_symlog(torch.tensor(x_symlog)).detach().numpy()
np.savez_compressed(
    os.path.join(FIXTURE_DIR, "symlog_input.npz"),
    x=x_symlog,
    torch_out=torch_symlog_out,
    seed=np.array(SEED),
)
print(f"  symlog_input.npz  x.shape={x_symlog.shape}  torch_out.shape={torch_symlog_out.shape}")

# ---------------------------------------------------------------------------
# symexp_input.npz
# ---------------------------------------------------------------------------
x_symexp = rng.uniform(-5.0, 5.0, size=(16, 4)).astype(np.float32)
torch_symexp_out = sheeprl_symexp(torch.tensor(x_symexp)).detach().numpy()
np.savez_compressed(
    os.path.join(FIXTURE_DIR, "symexp_input.npz"),
    x=x_symexp,
    torch_out=torch_symexp_out,
    seed=np.array(SEED),
)
print(f"  symexp_input.npz  x.shape={x_symexp.shape}  torch_out.shape={torch_symexp_out.shape}")

# ---------------------------------------------------------------------------
# init_weights_input.npz
# Stochastic: store (a) the theoretical std and (b) a sheeprl reference kernel
# generated with torch.manual_seed(SEED) so the test can compare distributions.
# The JAX port uses a JAX PRNGKey — we test distribution properties, not
# individual values (DEVIATION D-002).
# ---------------------------------------------------------------------------
in_features = 76
out_features = 16384
denoms = (in_features + out_features) / 2.0
scale_iw = 1.0 / denoms
std_theoretical = float(np.sqrt(scale_iw) / 0.87962566103423978)

import torch.nn as nn
torch.manual_seed(SEED)
lin_iw = nn.Linear(in_features, out_features)
nn.init.trunc_normal_(lin_iw.weight.data, mean=0.0, std=std_theoretical,
                      a=-2.0 * std_theoretical, b=2.0 * std_theoretical)
lin_iw.bias.data.fill_(0.0)
torch_iw_kernel = lin_iw.weight.data.detach().numpy()

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "init_weights_input.npz"),
    in_features=np.int32(in_features),
    out_features=np.int32(out_features),
    std_theoretical=np.float32(std_theoretical),
    torch_out_kernel=torch_iw_kernel,  # [out_features, in_features] — sheeprl weight shape
    jax_seed=np.int32(SEED & 0xFFFFFFFF),
    seed=np.array(SEED),
)
print(f"  init_weights_input.npz  in={in_features}, out={out_features}, std_theoretical={std_theoretical:.6f}")

# ---------------------------------------------------------------------------
# uniform_init_weights_input.npz
# Same: store the sheeprl reference kernel for distribution comparison.
# ---------------------------------------------------------------------------
given_scale = 1.0
denoms_u = (in_features + out_features) / 2.0
scale_u = given_scale / denoms_u
limit = float(np.sqrt(3 * scale_u))

torch.manual_seed(SEED)
lin_u = nn.Linear(in_features, out_features)
nn.init.uniform_(lin_u.weight.data, a=-limit, b=limit)
lin_u.bias.data.fill_(0.0)
torch_u_kernel = lin_u.weight.data.detach().numpy()

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "uniform_init_weights_input.npz"),
    in_features=np.int32(in_features),
    out_features=np.int32(out_features),
    given_scale=np.float32(given_scale),
    limit=np.float32(limit),
    torch_out_kernel=torch_u_kernel,
    jax_seed=np.int32(SEED & 0xFFFFFFFF),
    seed=np.array(SEED),
)
print(f"  uniform_init_weights_input.npz  in={in_features}, out={out_features}, scale={given_scale}, limit={limit:.6f}")

# ---------------------------------------------------------------------------
# compute_lambda_values_input.npz
# ---------------------------------------------------------------------------
T, B = 16, 4
rewards = rng.uniform(-1.0, 1.0, size=(T, B)).astype(np.float32)
values = rng.uniform(0.0, 2.0, size=(T, B)).astype(np.float32)
continues = rng.choice([0.0, 1.0], size=(T, B), p=[0.05, 0.95]).astype(np.float32)
lmbda = 0.95

torch_clv_out = sheeprl_clv(
    torch.tensor(rewards),
    torch.tensor(values),
    torch.tensor(continues),
    lmbda=lmbda,
).detach().numpy()

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "compute_lambda_values_input.npz"),
    rewards=rewards,
    values=values,
    continues=continues,
    lmbda=np.float32(lmbda),
    torch_out=torch_clv_out,
    seed=np.array(SEED),
)
print(f"  compute_lambda_values_input.npz  shape={rewards.shape}  torch_out.shape={torch_clv_out.shape}")

# ---------------------------------------------------------------------------
# moments_update_input.npz
# Manually replicate sheeprl Moments.forward without fabric.all_gather
# (single-process: all_gather is identity).
# ---------------------------------------------------------------------------
x_moments = rng.uniform(-5.0, 5.0, size=(T, B)).astype(np.float32)
decay = 0.99
max_ = 1e8
percentile_low = 0.05
percentile_high = 0.95

sheeprl_low_buf = torch.zeros((), dtype=torch.float32)
sheeprl_high_buf = torch.zeros((), dtype=torch.float32)
x_t = torch.tensor(x_moments).float()
low_new = torch.quantile(x_t, percentile_low)
high_new = torch.quantile(x_t, percentile_high)
sheeprl_low_buf = decay * sheeprl_low_buf + (1 - decay) * low_new
sheeprl_high_buf = decay * sheeprl_high_buf + (1 - decay) * high_new
sheeprl_max_t = torch.tensor(max_)
sheeprl_invscale_t = torch.max(1.0 / sheeprl_max_t, sheeprl_high_buf - sheeprl_low_buf)

torch_offset = sheeprl_low_buf.detach().numpy()
torch_invscale = sheeprl_invscale_t.detach().numpy()

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "moments_update_input.npz"),
    x=x_moments,
    low=np.float32(0.0),
    high=np.float32(0.0),
    decay=np.float32(decay),
    max_=np.float32(max_),
    percentile_low=np.float32(percentile_low),
    percentile_high=np.float32(percentile_high),
    torch_out_offset=torch_offset,
    torch_out_invscale=torch_invscale,
    seed=np.array(SEED),
)
print(f"  moments_update_input.npz  x.shape={x_moments.shape}  torch_offset={float(torch_offset):.6f}  torch_invscale={float(torch_invscale):.6f}")

# ---------------------------------------------------------------------------
# ratio_input.npz
# ---------------------------------------------------------------------------
ratio_value = 0.5
steps = [0, 2, 4, 6, 8, 10]

sheeprl_ratio = SheeprlRatio(ratio_value)
torch_ratio_repeats = [sheeprl_ratio(s) for s in steps]

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "ratio_input.npz"),
    ratio=np.float32(ratio_value),
    steps=np.array(steps, dtype=np.int32),
    torch_out_repeats=np.array(torch_ratio_repeats, dtype=np.int32),
    seed=np.array(SEED),
)
print(f"  ratio_input.npz  ratio={ratio_value}, steps={steps}, torch_repeats={torch_ratio_repeats}")

# ---------------------------------------------------------------------------
# prepare_obs_input.npz
# ---------------------------------------------------------------------------
obs_dim = 76
obs_state = rng.uniform(-1.0, 1.0, size=(obs_dim,)).astype(np.float32)
num_envs = 1

# sheeprl side (without Fabric device dispatch — pure reshape+float)
t_state = torch.from_numpy(obs_state.copy()).float()
torch_state_out = t_state.view(1, num_envs, -1).detach().numpy()

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "prepare_obs_input.npz"),
    obs_state=obs_state,
    num_envs=np.int32(num_envs),
    torch_out_state=torch_state_out,
    seed=np.array(SEED),
)
print(f"  prepare_obs_input.npz  obs_state.shape={obs_state.shape}  torch_out.shape={torch_state_out.shape}")

print("\nAll CP1 fixtures generated successfully in", FIXTURE_DIR)
