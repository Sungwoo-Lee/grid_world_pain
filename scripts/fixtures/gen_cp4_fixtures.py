"""gen_cp4_fixtures.py — deterministic fixture generator for CP4 + CP4b tests.

Generates .npz fixture files for:
  1. rssm_transition_input.npz          — CP4: RSSM._transition forward-pass parity
  2. rssm_representation_input.npz      — CP4: RSSM._representation forward-pass parity
  3. get_initial_states_input.npz       — CP4: RSSM.get_initial_states deterministic mode
  4. is_first_force_set_input.npz       — CP4b: is_first[0]=1 force-set at chunk start
  5. is_first_three_quantity_reset_input.npz — CP4b: 3-quantity reset at done boundary

Each fixture stores:
  - The raw input tensors + learnable parameters (identical for both sides)
  - The PRE-COMPUTED sheeprl/PyTorch reference output (torch_out_*)
    so the pytest tests in grid_world_pain (JAX-only env) can compare against
    stored reference values without needing PyTorch at test time.

All fixtures use PRNG seed 0xD3EAF (decimal 868591) per the fixture convention
in tests/fixtures/dreamer_srl/README.md.

Architecture used for all CP4/CP4b fixtures (small XS-like dims for speed):
  recurrent_state_size   = 64   (GRU hidden state)
  recurrent_dense_units  = 32   (RecurrentModel MLP output)
  action_dim             = 4
  stochastic_size        = 16   (S*D = 4*4)
  num_categoricals (S)   = 4
  num_classes (D)        = 4
  transition_hidden_size = 32
  repr_hidden_size       = 32
  encoder_output_dim     = 8
  unimix                 = 0.01
  batch_size (B)         = 4

Run from the repo root in the sheeprl_bridge env (needs both PyTorch + JAX):
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \\
        scripts/fixtures/gen_cp4_fixtures.py

This script is deterministic — re-running produces byte-identical .npz files.

Sheeprl source refs:
  - RecurrentModel: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L281-L341
  - RSSM: vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L344-L498
  - _transition: agent.py:L467-L480
  - _representation: agent.py:L451-L465
  - get_initial_states: agent.py:L391-L394
  - dynamic (§S4 reset): agent.py:L423-L435
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

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# Import sheeprl components
from sheeprl.models.models import MLP, LayerNormGRUCell as SheeprlGRUCell  # noqa: E402
from sheeprl.algos.dreamer_v3.agent import RecurrentModel, RSSM  # noqa: E402
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state  # noqa: E402
from sheeprl.algos.dreamer_v3.utils import init_weights  # noqa: E402

# ---------------------------------------------------------------------------
# Shared small-scale architecture config (all CP4/CP4b fixtures share this)
# ---------------------------------------------------------------------------
RECURRENT_STATE_SIZE = 64   # GRU hidden state
RECURRENT_DENSE_UNITS = 32  # RecurrentModel MLP output
ACTION_DIM = 4
STOCHASTIC_SIZE = 16        # S * D = 4 * 4
NUM_CATEGORICALS = 4        # S
NUM_CLASSES = 4             # D
TRANSITION_HIDDEN_SIZE = 32
REPR_HIDDEN_SIZE = 32
ENCODER_OUTPUT_DIM = 8
UNIMIX = 0.01
BATCH_SIZE = 4              # B
SEQ_LEN = 10               # for CP4b dynamic rollout test


def _build_sheeprl_rssm(seed: int) -> RSSM:
    """Build sheeprl RSSM with Hafner-init weights from a fixed seed."""
    torch.manual_seed(seed)

    # RecurrentModel: MLP(input → dense_units) + GRU
    # Sheeprl agent.py:L1010-L1016
    recurrent_model = RecurrentModel(
        input_size=int(STOCHASTIC_SIZE + ACTION_DIM),
        recurrent_state_size=RECURRENT_STATE_SIZE,
        dense_units=RECURRENT_DENSE_UNITS,
        layer_norm_cls=nn.LayerNorm,
        layer_norm_kw={"eps": 1e-3},
    )

    # Representation model MLP: cat(hx, obs_embed) → stochastic_size
    # Sheeprl agent.py:L1017-L1035
    repr_input_size = RECURRENT_STATE_SIZE + ENCODER_OUTPUT_DIM
    representation_model = MLP(
        input_dims=repr_input_size,
        output_dim=STOCHASTIC_SIZE,
        hidden_sizes=[REPR_HIDDEN_SIZE],
        activation=nn.SiLU,
        layer_args={"bias": False},   # bias=False (LayerNorm follows)
        flatten_dim=None,
        norm_layer=[nn.LayerNorm],
        norm_args=[{"eps": 1e-3, "normalized_shape": REPR_HIDDEN_SIZE}],
    )

    # Transition model MLP: hx → stochastic_size
    # Sheeprl agent.py:L1036-L1051
    transition_model = MLP(
        input_dims=RECURRENT_STATE_SIZE,
        output_dim=STOCHASTIC_SIZE,
        hidden_sizes=[TRANSITION_HIDDEN_SIZE],
        activation=nn.SiLU,
        layer_args={"bias": False},   # bias=False (LayerNorm follows)
        flatten_dim=None,
        norm_layer=[nn.LayerNorm],
        norm_args=[{"eps": 1e-3, "normalized_shape": TRANSITION_HIDDEN_SIZE}],
    )

    # Build RSSM with Hafner init_weights applied (sheeprl agent.py:L1057-L1065)
    rssm = RSSM(
        recurrent_model=recurrent_model.apply(init_weights),
        representation_model=representation_model.apply(init_weights),
        transition_model=transition_model.apply(init_weights),
        distribution_cfg={"type": "auto"},
        discrete=NUM_CLASSES,
        unimix=UNIMIX,
        learnable_initial_recurrent_state=True,
    )
    rssm.eval()
    return rssm


def _extract_rssm_params(rssm: RSSM) -> dict:
    """Extract all learnable parameters from sheeprl RSSM as numpy arrays."""
    params = {}

    # RecurrentModel MLP: one miniblock = Linear + LayerNorm + SiLU
    # MLP.model[0] = Linear (miniblock index 0)
    # MLP.model[1] = LayerNorm (miniblock index 1)
    # No SiLU parameter
    rec_mlp = rssm.recurrent_model.mlp
    params["recurrent_mlp_linear_weight"] = rec_mlp.model[0].weight.detach().cpu().numpy()  # [D, I]
    params["recurrent_mlp_linear_bias"]   = rec_mlp.model[0].bias.detach().cpu().numpy() if rec_mlp.model[0].bias is not None else np.zeros(RECURRENT_DENSE_UNITS, dtype=np.float32)
    params["recurrent_mlp_norm_weight"]   = rec_mlp.model[1].weight.detach().cpu().numpy()  # [D]
    params["recurrent_mlp_norm_bias"]     = rec_mlp.model[1].bias.detach().cpu().numpy()    # [D]

    # RecurrentModel GRU (LayerNormGRUCell)
    gru = rssm.recurrent_model.rnn
    params["gru_linear_weight"]   = gru.linear.weight.detach().cpu().numpy()     # [3H, I+H]
    params["gru_linear_bias"]     = gru.linear.bias.detach().cpu().numpy() if gru.linear.bias is not None else np.zeros(3 * RECURRENT_STATE_SIZE, dtype=np.float32)
    params["gru_norm_weight"]     = gru.layer_norm.weight.detach().cpu().numpy() # [3H]
    params["gru_norm_bias"]       = gru.layer_norm.bias.detach().cpu().numpy()   # [3H]

    # Transition model MLP: Linear(hx, hidden, bias=False) + LN + SiLU, Linear(hidden, S*D)
    # model[0] = Linear (hidden), model[1] = LayerNorm, model[2] = SiLU, model[3] = Linear (out)
    tm = rssm.transition_model
    params["transition_hidden_weight"] = tm.model[0].weight.detach().cpu().numpy()  # [H, hx]
    # bias=False for hidden linear
    params["transition_norm_weight"]   = tm.model[1].weight.detach().cpu().numpy()  # [H]
    params["transition_norm_bias"]     = tm.model[1].bias.detach().cpu().numpy()    # [H]
    params["transition_out_weight"]    = tm.model[3].weight.detach().cpu().numpy()  # [S*D, H]
    params["transition_out_bias"]      = tm.model[3].bias.detach().cpu().numpy()    # [S*D]

    # Representation model MLP: same structure, different input size
    rm = rssm.representation_model
    params["repr_hidden_weight"] = rm.model[0].weight.detach().cpu().numpy()  # [H, hx+enc]
    params["repr_norm_weight"]   = rm.model[1].weight.detach().cpu().numpy()  # [H]
    params["repr_norm_bias"]     = rm.model[1].bias.detach().cpu().numpy()    # [H]
    params["repr_out_weight"]    = rm.model[3].weight.detach().cpu().numpy()  # [S*D, H]
    params["repr_out_bias"]      = rm.model[3].bias.detach().cpu().numpy()    # [S*D]

    # Initial recurrent state parameter
    params["initial_recurrent_state"] = rssm.initial_recurrent_state.detach().cpu().numpy()  # [hx]

    return params


# ---------------------------------------------------------------------------
# Build the RSSM once (shared params across all fixtures)
# ---------------------------------------------------------------------------
print("Building sheeprl RSSM...")
torch.manual_seed(SEED)
sheeprl_rssm = _build_sheeprl_rssm(SEED)
rssm_params = _extract_rssm_params(sheeprl_rssm)
print(f"  RSSM built OK. initial_recurrent_state norm={float(np.linalg.norm(rssm_params['initial_recurrent_state'])):.4f}")

# Sanity check: verify model indices are correct
print(f"  transition_model layers: {[type(l).__name__ for l in sheeprl_rssm.transition_model.model]}")
print(f"  representation_model layers: {[type(l).__name__ for l in sheeprl_rssm.representation_model.model]}")
print(f"  recurrent_model.mlp layers: {[type(l).__name__ for l in sheeprl_rssm.recurrent_model.mlp.model]}")


# ---------------------------------------------------------------------------
# Fixture 1: rssm_transition_input.npz
#
# Test: RSSM._transition(recurrent_state) → (logits, state_mode)
# Uses sample_state=False (mode) to avoid PRNG divergence.
# sheeprl agent.py:L467-L480
# ---------------------------------------------------------------------------
print("\nGenerating fixture 1: rssm_transition_input.npz")

torch.manual_seed(SEED + 10)
recurrent_state_np = rng.standard_normal((BATCH_SIZE, RECURRENT_STATE_SIZE)).astype(np.float32)
recurrent_state_torch = torch.tensor(recurrent_state_np)

with torch.no_grad():
    # _transition internally uses _uniform_mix which requires >=3D input.
    # sheeprl's _uniform_mix: "The logits expected shape is 3 or 4" (agent.py:L442).
    # In training, sheeprl uses [T, B, S*D] (3D). We add a time dimension here
    # to match, then squeeze it back out.
    # _transition with sample_state=False → mode (deterministic)
    recurrent_state_3d = recurrent_state_torch.unsqueeze(0)  # [1, B, H_rec]
    prior_logits_torch, prior_state_torch = sheeprl_rssm._transition(
        recurrent_state_3d, sample_state=False
    )
    prior_logits_torch = prior_logits_torch.squeeze(0)  # [B, S*D]
    prior_state_torch = prior_state_torch.squeeze(0)    # [B, S, D]

prior_logits_np = prior_logits_torch.detach().cpu().numpy()   # [B, S*D]
prior_state_np = prior_state_torch.detach().cpu().numpy()     # [B, S, D]

print(f"  recurrent_state: {recurrent_state_np.shape}")
print(f"  prior_logits: {prior_logits_np.shape}, range=[{prior_logits_np.min():.3f},{prior_logits_np.max():.3f}]")
print(f"  prior_state: {prior_state_np.shape} (mode=argmax one-hot)")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "rssm_transition_input.npz"),
    # Inputs
    recurrent_state=recurrent_state_np,             # [B, H_rec]
    # Architecture dims
    recurrent_state_size=np.int32(RECURRENT_STATE_SIZE),
    recurrent_dense_units=np.int32(RECURRENT_DENSE_UNITS),
    action_dim=np.int32(ACTION_DIM),
    stochastic_size=np.int32(STOCHASTIC_SIZE),
    num_categoricals=np.int32(NUM_CATEGORICALS),
    num_classes=np.int32(NUM_CLASSES),
    transition_hidden_size=np.int32(TRANSITION_HIDDEN_SIZE),
    repr_hidden_size=np.int32(REPR_HIDDEN_SIZE),
    encoder_output_dim=np.int32(ENCODER_OUTPUT_DIM),
    batch_size=np.int32(BATCH_SIZE),
    # Learnable parameters
    **rssm_params,
    # Reference outputs (mode, sample_state=False)
    torch_out_logits=prior_logits_np,               # [B, S*D]
    torch_out_state=prior_state_np,                 # [B, S, D]
    seed=np.array(SEED),
)
print(f"  Saved rssm_transition_input.npz")


# ---------------------------------------------------------------------------
# Fixture 2: rssm_representation_input.npz
#
# Test: RSSM._representation(recurrent_state, embedded_obs) → (logits, state)
# Uses sampling=True BUT we need determinism → use a known gumbel noise injection.
# For simplicity, we test the logits only (before compute_stochastic_state).
# We test both logits AND the mode state (sample=False via argmax).
# sheeprl agent.py:L451-L465
# ---------------------------------------------------------------------------
print("\nGenerating fixture 2: rssm_representation_input.npz")

torch.manual_seed(SEED + 20)
recurrent_state2_np = rng.standard_normal((BATCH_SIZE, RECURRENT_STATE_SIZE)).astype(np.float32)
embedded_obs_np = rng.standard_normal((BATCH_SIZE, ENCODER_OUTPUT_DIM)).astype(np.float32)
recurrent_state2_torch = torch.tensor(recurrent_state2_np)
embedded_obs_torch = torch.tensor(embedded_obs_np)

with torch.no_grad():
    # Extract logits from representation model manually (before compute_stochastic_state)
    # _representation does: logits = repr_model(cat(hx, obs)); logits = _uniform_mix(logits)
    # _uniform_mix requires >=3D input; add time dim [1, B, ...] then squeeze back.
    repr_input_3d = torch.cat((
        recurrent_state2_torch.unsqueeze(0),  # [1, B, H_rec]
        embedded_obs_torch.unsqueeze(0),      # [1, B, enc]
    ), -1)  # [1, B, H_rec+enc]
    repr_logits_raw = sheeprl_rssm.representation_model(repr_input_3d)  # [1, B, S*D]
    repr_logits_mixed = sheeprl_rssm._uniform_mix(repr_logits_raw)       # [1, B, S*D]
    repr_logits_mixed = repr_logits_mixed.squeeze(0)  # [B, S*D]
    # Mode state (argmax) — deterministic comparison
    repr_state_mode = compute_stochastic_state(repr_logits_mixed, discrete=NUM_CLASSES, sample=False)

repr_logits_np = repr_logits_mixed.detach().cpu().numpy()   # [B, S*D]
repr_state_np = repr_state_mode.detach().cpu().numpy()      # [B, S, D]

print(f"  recurrent_state: {recurrent_state2_np.shape}")
print(f"  embedded_obs: {embedded_obs_np.shape}")
print(f"  repr_logits: {repr_logits_np.shape}, range=[{repr_logits_np.min():.3f},{repr_logits_np.max():.3f}]")
print(f"  repr_state (mode): {repr_state_np.shape}")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "rssm_representation_input.npz"),
    # Inputs
    recurrent_state=recurrent_state2_np,            # [B, H_rec]
    embedded_obs=embedded_obs_np,                   # [B, enc_dim]
    # Architecture dims
    recurrent_state_size=np.int32(RECURRENT_STATE_SIZE),
    recurrent_dense_units=np.int32(RECURRENT_DENSE_UNITS),
    action_dim=np.int32(ACTION_DIM),
    stochastic_size=np.int32(STOCHASTIC_SIZE),
    num_categoricals=np.int32(NUM_CATEGORICALS),
    num_classes=np.int32(NUM_CLASSES),
    transition_hidden_size=np.int32(TRANSITION_HIDDEN_SIZE),
    repr_hidden_size=np.int32(REPR_HIDDEN_SIZE),
    encoder_output_dim=np.int32(ENCODER_OUTPUT_DIM),
    batch_size=np.int32(BATCH_SIZE),
    # Learnable parameters
    **rssm_params,
    # Reference outputs (mode, sample_state=False for deterministic comparison)
    torch_out_logits=repr_logits_np,               # [B, S*D]
    torch_out_state=repr_state_np,                 # [B, S, D]
    seed=np.array(SEED),
)
print(f"  Saved rssm_representation_input.npz")


# ---------------------------------------------------------------------------
# Fixture 3: get_initial_states_input.npz
#
# Test: RSSM.get_initial_states(batch_shape) → (initial_hx, initial_z)
# Fully deterministic — no PRNG consumed (sample_state=False).
# sheeprl agent.py:L391-L394
# ---------------------------------------------------------------------------
print("\nGenerating fixture 3: get_initial_states_input.npz")

with torch.no_grad():
    # sheeprl's get_initial_states calls _transition → _uniform_mix which requires >=3D.
    # In sheeprl training this is called with recurrent_state.shape[:2] = (T, B).
    # We use (1, B) = (T=1, B) to match production semantics; then squeeze T=1 dim.
    init_hx_torch, init_z_torch = sheeprl_rssm.get_initial_states((1, BATCH_SIZE))
    init_hx_torch = init_hx_torch.squeeze(0)  # [1, B, H_rec] → [B, H_rec]
    init_z_torch = init_z_torch.squeeze(0)    # [1, B, S, D] → [B, S, D]

init_hx_np = init_hx_torch.detach().cpu().numpy()   # [B, H_rec]
init_z_np = init_z_torch.detach().cpu().numpy()     # [B, S, D]

print(f"  initial_hx: {init_hx_np.shape}, norm={float(np.linalg.norm(init_hx_np)):.4f}")
print(f"  initial_z: {init_z_np.shape} (mode one-hot)")
print(f"  initial_hx max_abs: {float(np.max(np.abs(init_hx_np))):.4f}")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "get_initial_states_input.npz"),
    # Architecture dims
    recurrent_state_size=np.int32(RECURRENT_STATE_SIZE),
    recurrent_dense_units=np.int32(RECURRENT_DENSE_UNITS),
    action_dim=np.int32(ACTION_DIM),
    stochastic_size=np.int32(STOCHASTIC_SIZE),
    num_categoricals=np.int32(NUM_CATEGORICALS),
    num_classes=np.int32(NUM_CLASSES),
    transition_hidden_size=np.int32(TRANSITION_HIDDEN_SIZE),
    repr_hidden_size=np.int32(REPR_HIDDEN_SIZE),
    encoder_output_dim=np.int32(ENCODER_OUTPUT_DIM),
    batch_size=np.int32(BATCH_SIZE),
    # Learnable parameters
    **rssm_params,
    # Reference outputs
    torch_out_hx=init_hx_np,   # [B, H_rec]
    torch_out_z=init_z_np,     # [B, S, D]
    seed=np.array(SEED),
)
print(f"  Saved get_initial_states_input.npz")


# ---------------------------------------------------------------------------
# Fixture 4: is_first_force_set_input.npz
#
# Test: is_first[0] is 1.0 at start of a chunk (S1 force-set).
# This is a structural test: the buffer must store is_first=1 for the first
# timestep of every chunk. The fixture verifies the convention by generating
# a sequence where is_first[0]=1 is set, and the test checks the JAX RSSM
# dynamic sees it correctly.
#
# We test: run dynamic with is_first[0]=1 → action/recurrent_state/posterior
# at t=0 must match the initial state (reset happened), not the input values.
# sheeprl agent.py:L423-L429 (dynamic §S4)
# ---------------------------------------------------------------------------
print("\nGenerating fixture 4: is_first_force_set_input.npz")

torch.manual_seed(SEED + 40)
# Create inputs for a single step with is_first=1 for the first item
posterior_np = rng.standard_normal((BATCH_SIZE, NUM_CATEGORICALS, NUM_CLASSES)).astype(np.float32)
recurrent_state3_np = rng.standard_normal((BATCH_SIZE, RECURRENT_STATE_SIZE)).astype(np.float32)
action3_np = rng.standard_normal((BATCH_SIZE, ACTION_DIM)).astype(np.float32)
embedded_obs3_np = rng.standard_normal((BATCH_SIZE, ENCODER_OUTPUT_DIM)).astype(np.float32)

# is_first: only env 0 has is_first=1, others have is_first=0
is_first_np = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
is_first_np[0, 0] = 1.0   # first env in batch is at episode start

# sheeprl's dynamic requires >=3D tensors (uses recurrent_state.shape[:2] internally).
# We add a T=1 leading dim to all inputs, call dynamic, then squeeze back.
posterior_torch = torch.tensor(posterior_np).unsqueeze(0)          # [1, B, S, D]
recurrent_state3_torch = torch.tensor(recurrent_state3_np).unsqueeze(0)  # [1, B, H_rec]
action3_torch = torch.tensor(action3_np).unsqueeze(0)              # [1, B, A]
embedded_obs3_torch = torch.tensor(embedded_obs3_np).unsqueeze(0)  # [1, B, enc]
is_first_torch = torch.tensor(is_first_np).unsqueeze(0)            # [1, B, 1]

with torch.no_grad():
    (h_out_torch, post_out_torch, prior_out_torch,
     post_logits_torch, prior_logits_torch) = sheeprl_rssm.dynamic(
        posterior_torch, recurrent_state3_torch, action3_torch,
        embedded_obs3_torch, is_first_torch
    )

h_out_np = h_out_torch.squeeze(0).detach().cpu().numpy()            # [B, H_rec]
post_out_np = post_out_torch.squeeze(0).detach().cpu().numpy()      # [B, S, D]
prior_out_np = prior_out_torch.squeeze(0).detach().cpu().numpy()    # [B, S, D]
post_logits_out_np = post_logits_torch.squeeze(0).detach().cpu().numpy()  # [B, S*D]
prior_logits_out_np = prior_logits_torch.squeeze(0).detach().cpu().numpy() # [B, S*D]

# Get initial states for verification (use (1, B) batch shape for sheeprl compat)
with torch.no_grad():
    init_hx_ref, init_z_ref = sheeprl_rssm.get_initial_states((1, BATCH_SIZE))
    init_hx_ref = init_hx_ref.squeeze(0)  # [B, H_rec]
    init_z_ref = init_z_ref.squeeze(0)    # [B, S, D]
init_hx_ref_np = init_hx_ref.detach().cpu().numpy()
init_z_ref_np = init_z_ref.detach().cpu().numpy()

print(f"  posterior: {posterior_np.shape}")
print(f"  is_first: {is_first_np.ravel()}")
print(f"  h_out[0] (should be reset): {h_out_np[0, :4]}")
print(f"  h_out[1] (should be normal): {h_out_np[1, :4]}")

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "is_first_force_set_input.npz"),
    # Inputs
    posterior=posterior_np,                         # [B, S, D]
    recurrent_state=recurrent_state3_np,            # [B, H_rec]
    action=action3_np,                              # [B, A]
    embedded_obs=embedded_obs3_np,                  # [B, enc_dim]
    is_first=is_first_np,                           # [B, 1]
    # Architecture dims
    recurrent_state_size=np.int32(RECURRENT_STATE_SIZE),
    recurrent_dense_units=np.int32(RECURRENT_DENSE_UNITS),
    action_dim=np.int32(ACTION_DIM),
    stochastic_size=np.int32(STOCHASTIC_SIZE),
    num_categoricals=np.int32(NUM_CATEGORICALS),
    num_classes=np.int32(NUM_CLASSES),
    transition_hidden_size=np.int32(TRANSITION_HIDDEN_SIZE),
    repr_hidden_size=np.int32(REPR_HIDDEN_SIZE),
    encoder_output_dim=np.int32(ENCODER_OUTPUT_DIM),
    batch_size=np.int32(BATCH_SIZE),
    # Learnable parameters
    **rssm_params,
    # Reference outputs
    torch_out_h=h_out_np,                          # [B, H_rec]
    torch_out_posterior=post_out_np,               # [B, S, D]
    torch_out_prior=prior_out_np,                  # [B, S, D]
    torch_out_post_logits=post_logits_out_np,      # [B, S*D]
    torch_out_prior_logits=prior_logits_out_np,    # [B, S*D]
    torch_initial_hx=init_hx_ref_np,              # [B, H_rec] for reset verification
    torch_initial_z=init_z_ref_np,                # [B, S, D] for reset verification
    seed=np.array(SEED),
)
print(f"  Saved is_first_force_set_input.npz")


# ---------------------------------------------------------------------------
# Fixture 5: is_first_three_quantity_reset_input.npz
#
# Test: done at index 5 of a 10-step rollout; assert all 3 quantities
# (action, recurrent_state, posterior) at index 6 are the reset state.
#
# We generate a sequence of T=10 steps, with done at t=5 (is_first at t=6).
# We run sheeprl's dynamic() step-by-step and store the full rollout outputs.
# The test checks step 6: action=0, recurrent=initial, posterior=initial.
#
# sheeprl agent.py:L423-L435 (dynamic — §S4 three-quantity reset)
# ---------------------------------------------------------------------------
print("\nGenerating fixture 5: is_first_three_quantity_reset_input.npz")

T = SEQ_LEN  # 10 steps

torch.manual_seed(SEED + 50)
# Generate T steps of inputs
posterior_seq_np = rng.standard_normal((T, BATCH_SIZE, NUM_CATEGORICALS, NUM_CLASSES)).astype(np.float32)
recurrent_seq_np = rng.standard_normal((T, BATCH_SIZE, RECURRENT_STATE_SIZE)).astype(np.float32)
action_seq_np = rng.standard_normal((T, BATCH_SIZE, ACTION_DIM)).astype(np.float32)
embedded_obs_seq_np = rng.standard_normal((T, BATCH_SIZE, ENCODER_OUTPUT_DIM)).astype(np.float32)

# is_first: done at index 5, so is_first=1 at index 6
is_first_seq_np = np.zeros((T, BATCH_SIZE, 1), dtype=np.float32)
DONE_AT = 5          # done at step 5
IS_FIRST_AT = DONE_AT + 1  # is_first=1 at step 6
is_first_seq_np[IS_FIRST_AT, :, 0] = 1.0   # all envs in batch reset at t=6

print(f"  is_first[{IS_FIRST_AT}] = {is_first_seq_np[IS_FIRST_AT, :, 0]}")

# Run sheeprl dynamic step-by-step; store each step's outputs
h_outputs_np = []
post_outputs_np = []
prior_outputs_np = []

with torch.no_grad():
    for t in range(T):
        # sheeprl dynamic requires >=3D; add T=1 dim, squeeze after
        post_t = torch.tensor(posterior_seq_np[t]).unsqueeze(0)         # [1, B, S, D]
        h_t = torch.tensor(recurrent_seq_np[t]).unsqueeze(0)            # [1, B, H_rec]
        a_t = torch.tensor(action_seq_np[t]).unsqueeze(0)               # [1, B, A]
        obs_t = torch.tensor(embedded_obs_seq_np[t]).unsqueeze(0)       # [1, B, enc]
        is_first_t = torch.tensor(is_first_seq_np[t]).unsqueeze(0)      # [1, B, 1]

        h_out_t, post_out_t, prior_out_t, _, _ = sheeprl_rssm.dynamic(
            post_t, h_t, a_t, obs_t, is_first_t
        )
        h_outputs_np.append(h_out_t.squeeze(0).detach().cpu().numpy())
        post_outputs_np.append(post_out_t.squeeze(0).detach().cpu().numpy())
        prior_outputs_np.append(prior_out_t.squeeze(0).detach().cpu().numpy())

h_outputs_np = np.stack(h_outputs_np, axis=0)     # [T, B, H_rec]
post_outputs_np = np.stack(post_outputs_np, axis=0) # [T, B, S, D]
prior_outputs_np = np.stack(prior_outputs_np, axis=0) # [T, B, S, D]

print(f"  h_outputs: {h_outputs_np.shape}")
print(f"  post_outputs: {post_outputs_np.shape}")
print(f"  h at t={IS_FIRST_AT} (reset step): norm={float(np.linalg.norm(h_outputs_np[IS_FIRST_AT])):.4f}")
print(f"  h at t={DONE_AT} (pre-reset):      norm={float(np.linalg.norm(h_outputs_np[DONE_AT])):.4f}")

# Get initial states for verification (use (1, B) batch shape for sheeprl compat)
with torch.no_grad():
    init_hx_seq, init_z_seq = sheeprl_rssm.get_initial_states((1, BATCH_SIZE))
    init_hx_seq = init_hx_seq.squeeze(0)  # [B, H_rec]
    init_z_seq = init_z_seq.squeeze(0)    # [B, S, D]
init_hx_seq_np = init_hx_seq.detach().cpu().numpy()
init_z_seq_np = init_z_seq.detach().cpu().numpy()

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "is_first_three_quantity_reset_input.npz"),
    # Input sequences [T, B, ...]
    posterior_seq=posterior_seq_np,                 # [T, B, S, D]
    recurrent_seq=recurrent_seq_np,                 # [T, B, H_rec]
    action_seq=action_seq_np,                       # [T, B, A]
    embedded_obs_seq=embedded_obs_seq_np,           # [T, B, enc_dim]
    is_first_seq=is_first_seq_np,                   # [T, B, 1]
    # Reset markers
    done_at=np.int32(DONE_AT),
    is_first_at=np.int32(IS_FIRST_AT),
    seq_len=np.int32(T),
    # Architecture dims
    recurrent_state_size=np.int32(RECURRENT_STATE_SIZE),
    recurrent_dense_units=np.int32(RECURRENT_DENSE_UNITS),
    action_dim=np.int32(ACTION_DIM),
    stochastic_size=np.int32(STOCHASTIC_SIZE),
    num_categoricals=np.int32(NUM_CATEGORICALS),
    num_classes=np.int32(NUM_CLASSES),
    transition_hidden_size=np.int32(TRANSITION_HIDDEN_SIZE),
    repr_hidden_size=np.int32(REPR_HIDDEN_SIZE),
    encoder_output_dim=np.int32(ENCODER_OUTPUT_DIM),
    batch_size=np.int32(BATCH_SIZE),
    # Learnable parameters
    **rssm_params,
    # Reference outputs (full T-step rollout)
    torch_out_h_seq=h_outputs_np,                   # [T, B, H_rec]
    torch_out_post_seq=post_outputs_np,             # [T, B, S, D]
    torch_out_prior_seq=prior_outputs_np,           # [T, B, S, D]
    # Initial states for reset verification
    torch_initial_hx=init_hx_seq_np,               # [B, H_rec]
    torch_initial_z=init_z_seq_np,                 # [B, S, D]
    seed=np.array(SEED),
)
print(f"  Saved is_first_three_quantity_reset_input.npz")

print(f"\nAll CP4 + CP4b fixtures generated successfully in {FIXTURE_DIR}")
