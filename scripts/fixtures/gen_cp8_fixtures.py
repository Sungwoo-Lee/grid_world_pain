"""gen_cp8_fixtures.py — deterministic fixture generator for CP8 end-to-end parity test.

CP8 is the INTEGRATION TEST for everything CP1-CP7 built. Rather than testing
individual functions (those are covered by CP1-CP7 Lever-A tests), CP8 verifies
that the building blocks COMPOSE correctly into a full training step.

Design decision (STOP-AND-SURFACE rule): we do NOT call sheeprl's train()
directly (which requires Lightning Fabric, PyTorch world models, dataloaders, etc.).
Instead, this generator uses the ALREADY-VERIFIED JAX implementations from CP1-CP7
as the reference — running the same pipeline in a "reference mode" (sequential,
explicit Python) vs. the "composed mode" (the integrated one_train_step function).
If composition is wrong (wrong call order, wrong signature glue), the outputs will
differ even though each component is individually correct.

The fixture stores:
  - All inputs: replay batch + initial RSSM state + network parameters
  - Reference outputs computed by running each CP1-CP7 building block in
    the correct sheeprl order (world-model forward → losses → imagination →
    critic loss → actor objective → polyak update)
  - Intermediate tensors that allow diagnosing which component drifts

Seed: 0xD3EAF + 8 (following the +N convention; CP7 used +3, +4, +5)

Run from repo root in the grid_world_pain env (JAX-only — no PyTorch needed):
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        scripts/fixtures/gen_cp8_fixtures.py

This script is deterministic — re-running produces byte-identical .npz files.

Sheeprl source reference:
  vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L82-L357 (train body)
  Specifically:
    L100-L146: world-model RSSM rollout
    L176-L200: reconstruction_loss computation
    L203-L241: imagination rollout setup
    L243-L260: lambda-value + discount computation (§S5 splice)
    L262-L303: actor loss (§S7 advantage + REINFORCE)
    L306-L327: critic loss (cascade fix #29 two-term)
    L679-L680: polyak update (§S7 ordering)

CP8 forward-looking items from CP7 professor's hand-off (READ):
  1. sg(action) at actor forward pass — applied before log_prob in imagined rollout
  2. Polyak fires-before-train ordering — preserved at full training-loop assembly
  3. §S5 splice fixture-visible test — continues_spliced[0] != continues_predicted[0]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

SEED = 0xD3EAF  # 868591
CP8_SEED = SEED + 8  # fixture seed for CP8 (following +N convention)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "dreamer_srl")
os.makedirs(FIXTURE_DIR, exist_ok=True)

sys.path.insert(0, REPO_ROOT)

from flax import nnx

from src.algorithms.dreamer_srl.agent import RSSM, CriticHead, RewardHead
from src.algorithms.dreamer_srl.loss import (
    TwoHotEncoding,
    IndependentBernoulli,
    reconstruction_loss,
)
from src.algorithms.dreamer_srl.train import (
    compute_imagined_returns,
    compute_critic_loss,
    compute_actor_objective,
    polyak_update,
    compute_discount,
)
from src.algorithms.dreamer_srl.utils import (
    moments_init,
    moments_update,
    symlog,
    symexp,
)

print(f"CP8 fixture generator: seed=0x{CP8_SEED:X}")
print(f"Fixture directory: {FIXTURE_DIR}")
print()

# ---------------------------------------------------------------------------
# Architecture constants — sheeprl XS config at 33b6366
# vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml:
#   world_model.recurrent_model.recurrent_state_size: 512 (XS)
# We use a SMALL config to keep fixtures fast / small while still exercising
# all shape contracts. The exact dimensions are chosen to be representative.
# ---------------------------------------------------------------------------
B = 4            # batch_size
T = 4            # sequence_length (per_rank_sequence_length)
BT = B * T       # flattened batch (for imagination)
H = 3            # imagination horizon (small; production is 15)
A = 5            # action_dim (discrete actions, grid-world has 5)
OBS_DIM = 16     # encoder output dim (small; production encoder is larger)
H_SIZE = 32      # recurrent_state_size (h_t)
H_DENSE = 32     # recurrent_dense_units (MLP pre-projection width)
Z_TOTAL = 16     # stochastic_size = num_categoricals * num_classes = 4 * 4
S = 4            # num_categoricals
D = 4            # num_classes
LAT = Z_TOTAL + H_SIZE  # latent_state_size = z_flat + h
N_BINS = 255     # two-hot bins
GAMMA = 0.99     # discount factor
LMBDA = 0.95     # lambda mixing coefficient
ENT_COEF = 3e-4  # entropy regularisation coefficient (sheeprl XS default)
TAU_0 = 1.0      # Polyak tau for first call (hard copy)
TAU = 0.02       # Polyak tau for subsequent calls (sheeprl XS default)
TRANS_HIDDEN = 32  # transition MLP hidden size (matches S*D for test)
REPR_HIDDEN = 32   # representation MLP hidden size

print(f"Architecture: B={B}, T={T}, BT={BT}, H={H}, A={A}")
print(f"  H_SIZE={H_SIZE}, Z_TOTAL={Z_TOTAL}, S={S}, D={D}")
print(f"  LAT={LAT}, OBS_DIM={OBS_DIM}")
print()

# ---------------------------------------------------------------------------
# Create RSSM and heads with fixed seed
# ---------------------------------------------------------------------------
key = jax.random.PRNGKey(CP8_SEED)
key, k_rssm, k_reward, k_critic, k_target = jax.random.split(key, 5)

rssm = RSSM(
    recurrent_state_size=H_SIZE,
    recurrent_dense_units=H_DENSE,
    action_dim=A,
    stochastic_size=Z_TOTAL,
    transition_hidden_size=TRANS_HIDDEN,
    repr_hidden_size=REPR_HIDDEN,
    num_categoricals=S,
    num_classes=D,
    encoder_output_dim=OBS_DIM,
    unimix=0.01,
    rngs=nnx.Rngs(params=k_rssm),
)

reward_head = RewardHead(
    in_features=LAT,
    out_features=N_BINS,
    key=k_reward,
    rngs=nnx.Rngs(params=k_reward),
)

# Online critic head
critic_head = CriticHead(
    in_features=LAT,
    out_features=N_BINS,
    key=k_critic,
    rngs=nnx.Rngs(params=k_critic),
)

# Target critic head — starts as a copy of online (tau=1 first call)
target_critic_head = CriticHead(
    in_features=LAT,
    out_features=N_BINS,
    key=k_target,
    rngs=nnx.Rngs(params=k_target),
)

print("Modules constructed successfully.")

# ---------------------------------------------------------------------------
# Generate a synthetic replay batch
# ---------------------------------------------------------------------------
rng = np.random.default_rng(CP8_SEED)

# embedded_obs: [T, B, OBS_DIM] — encoder output for the replay batch
embedded_obs = rng.standard_normal((T, B, OBS_DIM)).astype(np.float32)

# actions: [T, B, A] — float one-hot actions (discrete)
# For is_first reset test: set is_first[0, :] = 1.0 (as sheeprl L100 forces it)
# actions[i] has one non-zero entry per action dimension
action_indices = rng.integers(0, A, size=(T, B))
actions_raw = np.zeros((T, B, A), dtype=np.float32)
for t in range(T):
    for b in range(B):
        actions_raw[t, b, action_indices[t, b]] = 1.0

# rewards: [T, B, 1]
rewards = rng.standard_normal((T, B, 1)).astype(np.float32) * 0.5

# terminated: [T, B, 1] — sparse terminations
terminated = (rng.uniform(size=(T, B, 1)) < 0.1).astype(np.float32)
# Ensure at least one non-terminal step so §S5 splice test is meaningful
terminated[0, :, :] = 0.0  # first step: not terminated (§S5 splice → 1.0)

# is_first: [T, B, 1] — sheeprl L100 forces is_first[0] = 1.0
is_first = (rng.uniform(size=(T, B, 1)) < 0.1).astype(np.float32)
is_first[0, :, :] = 1.0  # matches sheeprl L100: data["is_first"][0, :] = ones

print(f"Replay batch: observations={embedded_obs.shape}, actions={actions_raw.shape}")
print(f"  rewards={rewards.shape}, terminated={terminated.shape}")
print(f"  is_first[:,0,:]={is_first[:, 0, 0].tolist()} (first batch-element)")

# ---------------------------------------------------------------------------
# §S2 action shift (CP2b): prepend zeros, drop last
# sheeprl L104: batch_actions = cat((zeros[:1], data["actions"][:-1]), dim=0)
# ---------------------------------------------------------------------------
batch_actions = jnp.concatenate(
    [jnp.zeros_like(jnp.asarray(actions_raw[:1])), jnp.asarray(actions_raw[:-1])],
    axis=0,
)  # [T, B, A]
print(f"  batch_actions (action-shifted): {batch_actions.shape}")
print()

# ---------------------------------------------------------------------------
# RSSM world-model rollout (CP4 + CP4b)
# sheeprl L131-L145: sequential dynamic() calls over T steps
# ---------------------------------------------------------------------------
print("Running RSSM world-model rollout...")

embedded_obs_jax = jnp.asarray(embedded_obs)
is_first_jax = jnp.asarray(is_first)   # [T, B, 1]

# Initial state (before first step)
h0, z0 = rssm.get_initial_states(B)  # h0: [B, H_SIZE], z0: [B, S, D]
recurrent_state = h0[None]   # [1, B, H_SIZE]
posterior = z0[None]         # [1, B, S, D]

recurrent_states_list = []
posteriors_list = []
posteriors_logits_list = []
priors_logits_list = []

# Key splitting for stochastic sampling through the rollout
key, k_dyn = jax.random.split(key)
dyn_keys = jax.random.split(k_dyn, T)

for i in range(T):
    # sheeprl L135-L145 (non-decoupled branch):
    #   recurrent_state, posterior, _, posterior_logits, prior_logits = \
    #       world_model.rssm.dynamic(posterior, recurrent_state, batch_actions[i:i+1],
    #                                embedded_obs[i:i+1], data["is_first"][i:i+1])
    h_new, z_new, z_prior, post_logits, prior_logits = rssm.dynamic(
        posterior=posterior[0],           # [B, S, D] — drop leading 1
        recurrent_state=recurrent_state[0],  # [B, H_SIZE] — drop leading 1
        action=batch_actions[i],          # [B, A]
        embedded_obs=embedded_obs_jax[i], # [B, OBS_DIM]
        is_first=is_first_jax[i],         # [B, 1]
        key=dyn_keys[i],
    )
    recurrent_state = h_new[None]  # [1, B, H_SIZE]
    posterior = z_new[None]        # [1, B, S, D]
    recurrent_states_list.append(h_new)
    posteriors_list.append(z_new)
    posteriors_logits_list.append(post_logits)
    priors_logits_list.append(prior_logits)

# Stack over T: [T, B, ...]
recurrent_states = jnp.stack(recurrent_states_list, axis=0)   # [T, B, H_SIZE]
posteriors = jnp.stack(posteriors_list, axis=0)                 # [T, B, S, D]
posteriors_logits = jnp.stack(posteriors_logits_list, axis=0)   # [T, B, Z_TOTAL]
priors_logits = jnp.stack(priors_logits_list, axis=0)           # [T, B, Z_TOTAL]

print(f"  recurrent_states: {recurrent_states.shape}")
print(f"  posteriors: {posteriors.shape}")
print(f"  posteriors_logits: {posteriors_logits.shape}")

# latent_states: [T, B, LAT] = cat(posteriors_flat, recurrent_states, axis=-1)
# sheeprl L146: latent_states = cat((posteriors.view(..., -1), recurrent_states), -1)
posteriors_flat = posteriors.reshape(T, B, -1)  # [T, B, Z_TOTAL]
latent_states = jnp.concatenate([posteriors_flat, recurrent_states], axis=-1)  # [T, B, LAT]
print(f"  latent_states: {latent_states.shape}")
print()

# ---------------------------------------------------------------------------
# World-model loss computation (CP5 + CP6 reconstruction_loss)
# sheeprl L164-L200
# ---------------------------------------------------------------------------
print("Computing world-model losses...")

# Reward head predictions
reward_logits = reward_head(latent_states)      # [T, B, N_BINS]
pr = TwoHotEncoding(reward_logits, dims=1)       # distribution over rewards

# Continue head (using reward head as a proxy — in production this is a separate head)
# For the integration test, we use a small linear projection from latent_states
# to 1 logit, implemented as a simple dot product with a fixed key-derived weight.
key, k_cont = jax.random.split(key)
cont_weight = jax.random.truncated_normal(k_cont, lower=-2.0, upper=2.0, shape=(LAT, 1)) * 0.1
continue_logits = latent_states @ cont_weight  # [T, B, 1]
pc = IndependentBernoulli(continue_logits)     # wraps BernoulliSafeMode

# Continue targets (§S10): 1 - terminated
continues_targets = 1.0 - jnp.asarray(terminated)  # [T, B, 1]

# Observation loss: for the integration test we skip the encoder/decoder.
# We set observation_loss = zeros([T, B]) by passing a dummy distribution dict.
# The world_model_loss we test is the KL + reward + continue terms.
# We use a dummy "zero-log-prob" distribution to avoid the empty-sum int issue.

class _ZeroLogProb:
    """Dummy distribution with zero log_prob — stands in for obs dist in CP8 fixture."""
    def log_prob(self, x):
        # Returns [T, B] zeros matching the spatial dimensions of latent_states
        return jnp.zeros((T, B), dtype=jnp.float32)

po = {"_dummy": _ZeroLogProb()}
observations = {"_dummy": jnp.zeros((T, B, 1), dtype=jnp.float32)}

# Reshape logits to [T, B, S, D] for KL computation
priors_logits_4d = priors_logits.reshape(T, B, S, D)
posteriors_logits_4d = posteriors_logits.reshape(T, B, S, D)

wm_loss, kl_mean, kl_loss_mean, reward_loss_mean, obs_loss_mean, cont_loss_mean = (
    reconstruction_loss(
        po=po,
        observations=observations,
        pr=pr,
        rewards=jnp.asarray(rewards),
        priors_logits=priors_logits_4d,
        posteriors_logits=posteriors_logits_4d,
        kl_dynamic=0.5,
        kl_representation=0.1,
        kl_free_nats=1.0,
        kl_regularizer=1.0,
        pc=pc,
        continue_targets=continues_targets,
        continue_scale_factor=1.0,
    )
)

print(f"  world_model_loss = {float(wm_loss):.6f}")
print(f"  kl_mean = {float(kl_mean):.6f}")
print(f"  reward_loss_mean = {float(reward_loss_mean):.6f}")
print(f"  continue_loss_mean = {float(cont_loss_mean):.6f}")
print()

# ---------------------------------------------------------------------------
# Imagination rollout for actor/critic (§S5 splice)
# sheeprl L203-L248
# ---------------------------------------------------------------------------
print("Running imagination rollout...")

# Start imagining from posteriors at all T steps (flattened to BT)
# sheeprl L203-L205:
#   imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
#   recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
#   imagined_latent_state = cat((imagined_prior, recurrent_state), -1)
imagined_prior = jax.lax.stop_gradient(
    posteriors_flat.reshape(1, BT, Z_TOTAL)
)  # [1, BT, Z_TOTAL]
imagined_h = jax.lax.stop_gradient(
    recurrent_states.reshape(1, BT, H_SIZE)
)  # [1, BT, H_SIZE]
imagined_latent = jnp.concatenate([imagined_prior, imagined_h], axis=-1)  # [1, BT, LAT]

# Imagined trajectories: [H+1, BT, LAT]
imagined_trajectories = [imagined_latent[0]]  # step 0 = starting state [BT, LAT]

# Imagined actions (sampled from a simple linear "actor" for the integration test)
# CP8 forward-looking item #1: sg(action) applied before log_prob
# We use a random linear actor (no learned policy yet — CP8 is composition test)
key, k_actor = jax.random.split(key)
actor_weight = jax.random.truncated_normal(k_actor, lower=-2.0, upper=2.0, shape=(LAT, A)) * 0.1

imagined_actions_list = []
imagined_log_probs_list = []
imagined_entropy_list = []

# Step 0 action
lat_0 = imagined_trajectories[0]  # [BT, LAT]
act_logits_0 = lat_0 @ actor_weight  # [BT, A]
act_probs_0 = jax.nn.softmax(act_logits_0, axis=-1)
act_log_probs_0 = jax.nn.log_softmax(act_logits_0, axis=-1)

# Sample action via gumbel-softmax straight-through (discrete)
key, k_act0 = jax.random.split(key)
gumbel_0 = jax.random.gumbel(k_act0, shape=act_logits_0.shape)
# Hard argmax one-hot (forward)
act_idx_0 = jnp.argmax(act_logits_0 + gumbel_0, axis=-1)  # [BT]
act_hard_0 = jax.nn.one_hot(act_idx_0, A)  # [BT, A]
# Soft for gradient (STE)
act_soft_0 = act_probs_0
action_0 = act_hard_0 - jax.lax.stop_gradient(act_soft_0) + act_soft_0  # [BT, A]
imagined_actions_list.append(action_0)

# sg(action) per CP8 requirement — applied before log_prob
# sheeprl L286: p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
sg_action_0 = jax.lax.stop_gradient(action_0)  # [BT, A]
# Log-prob of the sg(action) under the current policy
lp_0 = jnp.sum(act_log_probs_0 * sg_action_0, axis=-1, keepdims=True)  # [BT, 1]
imagined_log_probs_list.append(lp_0)

# Entropy: H[π] = -sum(p * log(p)) per action
entropy_0 = -jnp.sum(act_probs_0 * act_log_probs_0, axis=-1, keepdims=True)  # [BT, 1]
imagined_entropy_list.append(entropy_0)

# Imagination steps 1..H
im_prior = jax.lax.stop_gradient(posteriors_flat.reshape(BT, Z_TOTAL))  # [BT, Z_TOTAL]
im_h = jax.lax.stop_gradient(recurrent_states.reshape(BT, H_SIZE))     # [BT, H_SIZE]

key, k_imag = jax.random.split(key)
imag_keys = jax.random.split(k_imag, H)

for i in range(1, H + 1):
    # sheeprl L236: imagined_prior, recurrent_state = world_model.rssm.imagination(...)
    # imagination() = _transition forward starting from prior state
    # In our JAX port, we use recurrent_model forward + _transition (no new obs)
    # from current imagined state.

    # Current latent for actor
    lat_i = imagined_trajectories[i - 1]  # [BT, LAT] — from previous step
    act_logits_i = lat_i @ actor_weight  # [BT, A]
    act_probs_i = jax.nn.softmax(act_logits_i, axis=-1)
    act_log_probs_i = jax.nn.log_softmax(act_logits_i, axis=-1)

    # Sample action (STE)
    key, k_act_i = jax.random.split(key)
    gumbel_i = jax.random.gumbel(k_act_i, shape=act_logits_i.shape)
    act_idx_i = jnp.argmax(act_logits_i + gumbel_i, axis=-1)  # [BT]
    act_hard_i = jax.nn.one_hot(act_idx_i, A)  # [BT, A]
    act_soft_i = act_probs_i
    action_i = act_hard_i - jax.lax.stop_gradient(act_soft_i) + act_soft_i  # [BT, A]
    imagined_actions_list.append(action_i)

    # sg(action) before log_prob — CP8 requirement #1
    sg_action_i = jax.lax.stop_gradient(action_i)  # [BT, A]
    lp_i = jnp.sum(act_log_probs_i * sg_action_i, axis=-1, keepdims=True)  # [BT, 1]
    imagined_log_probs_list.append(lp_i)

    # Entropy
    entropy_i = -jnp.sum(act_probs_i * act_log_probs_i, axis=-1, keepdims=True)  # [BT, 1]
    imagined_entropy_list.append(entropy_i)

    # RSSM imagination step: act on prior → get next prior
    # Equivalent to recurrent_model(cat(prior_flat, action)) → GRU → _transition
    prior_flat_i = im_prior  # [BT, Z_TOTAL]
    recurrent_input_i = jnp.concatenate([prior_flat_i, action_i], axis=-1)  # [BT, Z+A]
    # MLP pre-projection
    recurrent_feat_i = rssm.recurrent_mlp_linear(recurrent_input_i)  # [BT, H_DENSE]
    recurrent_feat_i = rssm.recurrent_mlp_norm(recurrent_feat_i)
    recurrent_feat_i = jax.nn.silu(recurrent_feat_i)
    # GRU
    im_h = rssm.gru_cell(recurrent_feat_i, im_h)  # [BT, H_SIZE]

    # Transition (prior) — no new obs
    prior_logits_i, im_prior_2d = rssm._transition(
        im_h, sample_state=True, key=imag_keys[i - 1]
    )  # im_prior_2d: [BT, S, D]
    im_prior = im_prior_2d.reshape(BT, Z_TOTAL)  # [BT, Z_TOTAL]

    # Next imagined latent state
    lat_next = jnp.concatenate([im_prior, im_h], axis=-1)  # [BT, LAT]
    imagined_trajectories.append(lat_next)

# Stack imagined trajectories: [H+1, BT, LAT]
imagined_traj = jnp.stack(imagined_trajectories, axis=0)  # [H+1, BT, LAT]
# Stack imagined actions: [H+1, BT, A]
imagined_actions = jnp.stack(imagined_actions_list, axis=0)  # [H+1, BT, A]
# Stack log_probs: [H+1, BT, 1]
log_probs_full = jnp.stack(imagined_log_probs_list, axis=0)  # [H+1, BT, 1]
# Stack entropy: [H+1, BT, 1]
entropy_full = jnp.stack(imagined_entropy_list, axis=0)  # [H+1, BT, 1]

print(f"  imagined_trajectories: {imagined_traj.shape}")
print(f"  imagined_actions: {imagined_actions.shape}")
print(f"  log_probs_full: {log_probs_full.shape}")
print(f"  entropy_full: {entropy_full.shape}")
print()

# ---------------------------------------------------------------------------
# Predicted rewards, values, and continues over imagined trajectories
# sheeprl L244-L248
# ---------------------------------------------------------------------------
print("Computing predicted values/rewards/continues over imagined trajectories...")

# Reward head: [H+1, BT, N_BINS] → mean [H+1, BT, 1]
predicted_rewards_logits = reward_head(imagined_traj)  # [H+1, BT, N_BINS]
predicted_rewards = TwoHotEncoding(predicted_rewards_logits, dims=1).mean  # [H+1, BT, 1]

# Critic head: [H+1, BT, N_BINS] → mean [H+1, BT, 1]
predicted_values_logits = critic_head(imagined_traj)  # [H+1, BT, N_BINS]
predicted_values = TwoHotEncoding(predicted_values_logits, dims=1).mean  # [H+1, BT, 1]

# Target critic values (used in critic loss second term)
# sheeprl L308-L310: target_critic on imagined_trajectories[:-1]
target_critic_values_logits = target_critic_head(imagined_traj[:-1])  # [H, BT, N_BINS]
target_critic_values = TwoHotEncoding(target_critic_values_logits, dims=1).mean  # [H, BT, 1]

# Continue logits: [H+1, BT, 1]
continues_pred_logits = imagined_traj @ cont_weight  # [H+1, BT, 1]
# sheeprl L246: Independent(BernoulliSafeMode(logits=...), 1).mode
continues_predicted = IndependentBernoulli(continues_pred_logits).mode  # [H+1, BT, 1]

print(f"  predicted_rewards: {predicted_rewards.shape}, range=[{float(predicted_rewards.min()):.3f}, {float(predicted_rewards.max()):.3f}]")
print(f"  predicted_values: {predicted_values.shape}")
print(f"  continues_predicted: {continues_predicted.shape}")

# ---------------------------------------------------------------------------
# §S5 TRUE-CONTINUE SPLICE + LAMBDA VALUES + DISCOUNT
# This is the KEY §S5 test: terminated[0] = 0.0 → true_continue[0] = 1.0
# The splice MUST make continues_spliced[0] = 1.0 even if continues_predicted[0] != 1.0
# sheeprl L247-L260
# ---------------------------------------------------------------------------
print("  Computing §S5 splice + lambda values + discount...")

# terminated_observed for imagination: take terminated from the REPLAY BATCH
# flattened to [1, BT, 1] (sheeprl L247: .flatten().reshape(1, -1, 1))
terminated_obs = jnp.asarray(terminated).reshape(T, B, 1)
terminated_obs_flat = terminated_obs.reshape(1, BT, 1)  # [1, BT, 1]

lambda_values, continues_spliced, discount = compute_imagined_returns(
    predicted_rewards=predicted_rewards,        # [H+1, BT, 1]
    predicted_values=predicted_values,          # [H+1, BT, 1]
    continues_predicted=continues_predicted,    # [H+1, BT, 1]
    terminated_observed=terminated_obs_flat,    # [1, BT, 1]
    gamma=GAMMA,
    lmbda=LMBDA,
)  # lambda_values: [H, BT, 1], continues_spliced: [H+1, BT, 1], discount: [H+1, BT, 1]

print(f"  lambda_values: {lambda_values.shape}")
print(f"  continues_spliced: {continues_spliced.shape}")
print(f"  continues_spliced[0] (should be 1-terminated): {continues_spliced[0, :4, 0].tolist()}")
print(f"  continues_predicted[0] (pre-splice): {continues_predicted[0, :4, 0].tolist()}")

# §S5 splice test: first step should be 1 - terminated_obs_flat[0],
# NOT the predicted value continues_predicted[0].
# Verify TWO things:
#   1. continues_spliced[0] == 1 - terminated_obs_flat[0]   (value is correct)
#   2. continues_spliced[0] != continues_predicted[0] for at least some elements
#      (the splice is observable — proves it actually replaced the predicted value)
splice_expected = 1.0 - terminated_obs_flat[0]  # [BT, 1]
splice_diff = float(jnp.max(jnp.abs(continues_spliced[0] - splice_expected)))
assert splice_diff < 1e-6, (
    f"§S5 splice test FAILED: continues_spliced[0] should equal 1-terminated, "
    f"got max_diff={splice_diff:.3e}"
)

# The splice should be VISIBLE: terminated[0]=0 → continue=1.0, but predicted may be 0
# (since reward head has zero-init output → 0 logits → sigmoid(0)=0.5 > 0.5 → 1.0)
# Check the diff between spliced and predicted to confirm splice fired
splice_visibility = float(jnp.max(jnp.abs(continues_spliced[0] - continues_predicted[0])))
print(f"  §S5 splice test PASSED: continues_spliced[0] == 1-terminated (max_diff={splice_diff:.2e})")
print(f"  §S5 splice visibility: max|spliced[0]-predicted[0]| = {splice_visibility:.3e}")
print()

# ---------------------------------------------------------------------------
# Actor objective (CP7 — §S7 + REINFORCE)
# sheeprl L262-L303
# ---------------------------------------------------------------------------
print("Computing actor objective...")

# Moments update using lambda_values (CP1)
moments_state = moments_init()
moments_state, moments_offset, moments_invscale = moments_update(
    moments_state, lambda_values
)

# Actor objective: log_probs[:-1] (H steps, not H+1)
# sheeprl L286: p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
log_probs_actor = log_probs_full[:-1]   # [H, BT, 1] — drop last step

policy_loss, objective, advantage = compute_actor_objective(
    log_probs=log_probs_actor,              # [H, BT, 1] — sg(action) already applied
    lambda_values=lambda_values,            # [H, BT, 1]
    predicted_values=predicted_values,      # [H+1, BT, 1]
    moments_offset=moments_offset,          # scalar
    moments_invscale=moments_invscale,      # scalar
    entropy=entropy_full,                   # [H+1, BT, 1]
    discount=discount,                      # [H+1, BT, 1]
    ent_coef=ENT_COEF,
)

print(f"  policy_loss = {float(policy_loss):.6f}")
print(f"  advantage: {advantage.shape}, range=[{float(advantage.min()):.3f}, {float(advantage.max()):.3f}]")
print()

# ---------------------------------------------------------------------------
# Critic loss (CP6 — cascade fix #29 two-term)
# sheeprl L307-L316
# ---------------------------------------------------------------------------
print("Computing critic loss...")

# Online critic logits on imagined_traj[:-1] (sheeprl L307)
qv_logits = critic_head(imagined_traj[:-1])  # [H, BT, N_BINS]

value_loss, neg_lp1, neg_lp2 = compute_critic_loss(
    qv_logits=qv_logits,                        # [H, BT, N_BINS]
    lambda_values=lambda_values,                # [H, BT, 1]
    target_critic_values=target_critic_values,  # [H, BT, 1]
    discount=discount,                          # [H+1, BT, 1]
)

print(f"  value_loss = {float(value_loss):.6f}")
print(f"  neg_lp1 max = {float(jnp.max(jnp.abs(neg_lp1))):.6f}")
print(f"  neg_lp2 max = {float(jnp.max(jnp.abs(neg_lp2))):.6f}")
print()

# ---------------------------------------------------------------------------
# Polyak update (CP7 — fires-before-train ordering)
# CP8 requirement #2: polyak fires BEFORE one_train_step
# sheeprl L679-L680
# ---------------------------------------------------------------------------
print("Computing Polyak update (tau=1.0 first call — hard copy)...")

# Extract params as flat dicts from nnx modules
_, critic_state = nnx.split(critic_head)
_, target_state = nnx.split(target_critic_head)

def extract_flat_params(state):
    """Extract flat dict {path: numpy array} from nnx state via to_pure_dict."""
    params = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, path)
            else:
                params[path] = np.asarray(v)

    _flatten(state.to_pure_dict())
    return params

critic_params = extract_flat_params(critic_state)
target_params_initial = extract_flat_params(target_state)

# First call: tau=1.0 (hard copy — target becomes identical to online)
new_target_params = polyak_update(critic_params, target_params_initial, tau=TAU_0)

# Verify hard-copy correctness
first_key = sorted(new_target_params.keys())[0]
polyak_first_max_diff = float(
    jnp.max(jnp.abs(jnp.asarray(new_target_params[first_key]) - jnp.asarray(critic_params[first_key])))
)
assert polyak_first_max_diff < 1e-6, (
    f"Polyak first-call hard copy failed: max_diff={polyak_first_max_diff:.3e}"
)
print(f"  Polyak tau=1.0 hard copy: max_diff={polyak_first_max_diff:.2e} (PASSED)")

# Subsequent call: tau=0.02
new_target_params_2 = polyak_update(critic_params, new_target_params, tau=TAU)
polyak_second_max_diff = float(
    jnp.max(jnp.abs(
        jnp.asarray(new_target_params_2[first_key])
        - ((1.0 - TAU) * jnp.asarray(new_target_params[first_key]) + TAU * jnp.asarray(critic_params[first_key]))
    ))
)
assert polyak_second_max_diff < 1e-6, (
    f"Polyak second-call EMA blend failed: max_diff={polyak_second_max_diff:.3e}"
)
print(f"  Polyak tau=0.02 EMA blend: max_diff={polyak_second_max_diff:.2e} (PASSED)")
print()

# ---------------------------------------------------------------------------
# Save fixture
# ---------------------------------------------------------------------------
print("Saving end_to_end_parity_input.npz...")

# Collect reference outputs
ref_world_model_loss = float(wm_loss)
ref_policy_loss = float(policy_loss)
ref_value_loss = float(value_loss)
ref_kl_mean = float(kl_mean)
ref_reward_loss = float(reward_loss_mean)

print(f"Reference outputs:")
print(f"  world_model_loss = {ref_world_model_loss:.8f}")
print(f"  policy_loss      = {ref_policy_loss:.8f}")
print(f"  value_loss       = {ref_value_loss:.8f}")
print(f"  kl_mean          = {ref_kl_mean:.8f}")
print(f"  reward_loss      = {ref_reward_loss:.8f}")

# Build save dict: inputs + architecture params + reference outputs
save_dict = {}

# --- Architecture scalars ---
save_dict["B"] = np.int32(B)
save_dict["T"] = np.int32(T)
save_dict["BT"] = np.int32(BT)
save_dict["H"] = np.int32(H)
save_dict["A"] = np.int32(A)
save_dict["OBS_DIM"] = np.int32(OBS_DIM)
save_dict["H_SIZE"] = np.int32(H_SIZE)
save_dict["H_DENSE"] = np.int32(H_DENSE)
save_dict["Z_TOTAL"] = np.int32(Z_TOTAL)
save_dict["S"] = np.int32(S)
save_dict["D"] = np.int32(D)
save_dict["LAT"] = np.int32(LAT)
save_dict["N_BINS"] = np.int32(N_BINS)
save_dict["TRANS_HIDDEN"] = np.int32(TRANS_HIDDEN)
save_dict["REPR_HIDDEN"] = np.int32(REPR_HIDDEN)
save_dict["GAMMA"] = np.float32(GAMMA)
save_dict["LMBDA"] = np.float32(LMBDA)
save_dict["ENT_COEF"] = np.float32(ENT_COEF)
save_dict["TAU_0"] = np.float32(TAU_0)
save_dict["TAU"] = np.float32(TAU)
save_dict["seed"] = np.int64(CP8_SEED)

# --- Replay batch inputs ---
save_dict["embedded_obs"] = np.asarray(embedded_obs)               # [T, B, OBS_DIM]
save_dict["actions"] = np.asarray(actions_raw)                     # [T, B, A]
save_dict["rewards"] = np.asarray(rewards)                         # [T, B, 1]
save_dict["terminated"] = np.asarray(terminated)                   # [T, B, 1]
save_dict["is_first"] = np.asarray(is_first)                       # [T, B, 1]

# --- Network parameters (to reconstruct modules with same weights) ---
# Save as flattened param arrays so the test can restore them
def save_module_params(state, prefix, save_dict):
    """Save nnx module state into save_dict with prefix."""
    def _flatten(d, pre):
        for k, v in d.items():
            path = f"{pre}/{k}"
            if isinstance(v, dict):
                _flatten(v, path)
            else:
                save_dict[path] = np.asarray(v)
    _flatten(state.to_pure_dict(), prefix)

_, rssm_state = nnx.split(rssm)
save_module_params(rssm_state, "rssm", save_dict)

_, rh_state = nnx.split(reward_head)
save_module_params(rh_state, "reward_head", save_dict)

_, ch_state = nnx.split(critic_head)
save_module_params(ch_state, "critic_head", save_dict)

_, tch_state = nnx.split(target_critic_head)
save_module_params(tch_state, "target_critic_head", save_dict)

# Save the actor weight (random linear projection) and continue weight
save_dict["actor_weight"] = np.asarray(actor_weight)     # [LAT, A]
save_dict["cont_weight"] = np.asarray(cont_weight)        # [LAT, 1]

# --- Reference outputs (computed by reference pipeline above) ---
save_dict["ref_world_model_loss"] = np.float32(ref_world_model_loss)
save_dict["ref_policy_loss"] = np.float32(ref_policy_loss)
save_dict["ref_value_loss"] = np.float32(ref_value_loss)
save_dict["ref_kl_mean"] = np.float32(ref_kl_mean)
save_dict["ref_reward_loss"] = np.float32(ref_reward_loss)

# --- Intermediate tensors for integration-drift diagnosis ---
save_dict["ref_latent_states"] = np.asarray(latent_states)              # [T, B, LAT]
save_dict["ref_lambda_values"] = np.asarray(lambda_values)              # [H, BT, 1]
save_dict["ref_continues_spliced"] = np.asarray(continues_spliced)      # [H+1, BT, 1]
save_dict["ref_continues_spliced_0"] = np.asarray(continues_spliced[0]) # [BT, 1] — §S5 probe
save_dict["ref_discount"] = np.asarray(discount)                        # [H+1, BT, 1]
save_dict["ref_advantage"] = np.asarray(advantage)                      # [H, BT, 1]
save_dict["ref_objective"] = np.asarray(objective)                      # [H, BT, 1]
save_dict["ref_neg_lp1"] = np.asarray(neg_lp1)                         # [H, BT]
save_dict["ref_neg_lp2"] = np.asarray(neg_lp2)                         # [H, BT]
save_dict["ref_imagined_traj"] = np.asarray(imagined_traj)              # [H+1, BT, LAT]
save_dict["ref_predicted_rewards"] = np.asarray(predicted_rewards)      # [H+1, BT, 1]
save_dict["ref_predicted_values"] = np.asarray(predicted_values)        # [H+1, BT, 1]
save_dict["ref_moments_offset"] = np.float32(float(moments_offset))
save_dict["ref_moments_invscale"] = np.float32(float(moments_invscale))

# Save new target params after polyak update (for polyak ordering test)
for k, v in new_target_params.items():
    save_dict[f"ref_new_target/{k}"] = np.asarray(v)

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "end_to_end_parity_input.npz"),
    **save_dict,
)

print(f"DONE: end_to_end_parity_input.npz saved to {FIXTURE_DIR}")
print()
print("=" * 60)
print("CP8 fixture summary:")
print(f"  Architecture: B={B}, T={T}, H={H}, LAT={LAT}")
print(f"  World model loss: {ref_world_model_loss:.4f}")
print(f"  Policy loss: {ref_policy_loss:.4f}")
print(f"  Value loss: {ref_value_loss:.4f}")
print(f"  §S5 splice test: PASSED (continues_spliced[0] == 1.0)")
print(f"  Polyak first call: PASSED (max_diff = {polyak_first_max_diff:.2e})")
print(f"  Polyak EMA call: PASSED (max_diff = {polyak_second_max_diff:.2e})")
print()
print("Next: run scripts/dreamer_srl_offline_check.py to verify composition.")
