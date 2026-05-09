# DreamerV3 — paper-canonical algorithm and implementation map

## §1 Question / Purpose / Headline

DreamerV3 (Hafner, Pasukonis, Ba, & Lillicrap, 2023 preprint; 2025 Nature) is a model-based reinforcement learning algorithm whose central claim is that one fixed set of hyperparameters can master 150+ tasks across continuous and discrete actions, visual and proprioceptive inputs, dense and sparse rewards, 2D and 3D worlds — including Minecraft Diamond from scratch. The cross-task uniformity is not a happy accident: the three scale-invariance tricks (symlog squashing, two-hot symlog heads, percentile return scaling) exist *precisely* to eliminate the per-domain knobs (reward clipping, running-normalisation stats, scale-dependent entropy bonus) that earlier Dreamer versions needed, so that the same hyperparameter set transfers across return distributions that differ by orders of magnitude. It is the third generation of the Dreamer family: a Recurrent State-Space Model (RSSM) world model trained from replay, an actor and a critic trained entirely *inside* multi-step rollouts of the world model's imagination, and three "scale-invariance" tricks that remove every per-domain knob that earlier Dreamer versions needed. Imagination-time training is what gives DreamerV3 its sample efficiency: every replayed posterior state launches a free `T_imag = 15`-step rollout in latent space, multiplying the actor-critic gradient signal per real env step by roughly `T_imag` without further env interaction. The three tricks are (1) **symlog squashing of all real-valued targets** — the encoder, decoder, reward head and critic predict in `sign(x) * log(|x|+1)` space, removing the need for reward clipping or running normalisation; (2) **two-hot symlog discrete heads** — the reward head and the critic emit a 255-bin softmax over symlog-spaced bins (raw-space support `±symexp(20) ≈ ±4.85·10^8`) and are trained with cross-entropy against a "two-hot" soft target, so the loss landscape stays uniform across return scales; (3) **percentile return scaling** — the actor's policy gradient is divided by an exponential-moving-average of the 5th-to-95th-percentile range of returns, but only when that range exceeds 1, so dense-reward returns are scaled down while sparse-reward returns pass through unchanged. The algorithm also keeps DreamerV2's categorical RSSM with KL balancing (asymmetric weights on the two KL directions, plus a 1-nat free-bits floor on each), the unimix mixture (1% uniform + 99% network) on every categorical, and the discounted-λ-return actor-critic objective from DreamerV1. This document gives a paper-canonical primer in §2, then walks our codebase component-by-component in §3 (encoder, RSSM, decoder, heads, every loss, the buffer, the optimiser, every numerical-stability trick), with file-and-line citations and an explicit "matches paper / minor deviation / major deviation / extension" flag on each item. §4 inventories the metrics and probes our code logs. §5 is a four-column config table comparing every YAML key to the paper default. §6 collects every deviation flag from §3 into one ranked list, highest-impact first — readers chasing "what is non-standard about our DreamerV3" should jump to §6.

## §2 Paper-canonical algorithm

DreamerV3 inherits the following backbone from PlaNet, DreamerV1, and DreamerV2, then layers on the three robustness ingredients that make the single-config claim work. Equation numbers refer to Hafner 2023 (preprint) unless otherwise noted; the 2025 Nature version differs only in the loss-weight schedule, the framing of the critic as a maximum-likelihood distributional return predictor, the addition of a replay-value loss, and the explicit symexp grid for critic buckets — these deltas are flagged inline. See [Phase 5a](../references/Dreamer/dreamer_lit_review.md#paper-5a) and [Phase 5b](../references/Dreamer/dreamer_lit_review.md#paper-5b) of the lit review for the worked-out derivations.

### §2.1 RSSM world model (carried over from DreamerV2)

The world model is a **Recurrent State-Space Model**: a deterministic GRU state `h_t` plus a stochastic categorical latent `z_t`. At each step the model maintains

- **Recurrent core**: `h_t = f_phi(h_{t-1}, z_{t-1}, a_{t-1})`.
- **Prior over the next stochastic latent**: `p_phi(z_t | h_t)` — used at imagination time when no observation is available.
- **Posterior over the stochastic latent given an observation**: `q_phi(z_t | h_t, x_t)` — used at training time on replayed data.
- **Decoder**: `p_phi(x_t | h_t, z_t)`.
- **Reward head**: `p_phi(r_t | h_t, z_t)`.
- **Continue head**: `p_phi(c_t | h_t, z_t)` — a Bernoulli over "episode continues" (1 - terminated).

`z_t` is implemented as a 32-way × 32-class categorical (32 groups, each a one-hot of size 32) with a straight-through gradient estimator: forward pass uses the one-hot sample, backward pass routes gradients through the softmax probabilities (DreamerV2 §2.1; Hafner 2021 Eqs. 6-7).

### §2.2 Three robustness tricks (the DreamerV3 contribution)

**Symlog squashing** (Eqs. 1–2 of Hafner 2023). Define `symlog(x) = sign(x) * ln(|x| + 1)` and its smooth inverse `symexp(x) = sign(x) * (exp(|x|) - 1)`. Symlog is the identity near zero and behaves logarithmically far from zero. Targets for the decoder, the reward head, and the critic are first symlog-transformed; squared / cross-entropy loss is computed in symlog space; predictions are read out via symexp at inference. The encoder also squashes its real-valued inputs by symlog.

**Two-hot symlog reward and critic heads** (Eqs. 8–10). Both heads are 255-way categorical softmaxes over a fixed grid of bin centres `B = (b_1, ..., b_255)`. The preprint says "equally spaced over [-20, +20]" — disambiguated by the Nature paper (page 4) as `B = symexp(linspace(-20, 20, 255))`: the linspace is over `[-20, +20]` *in symlog space*, so **raw-space bin centres span approximately ±(exp(20) - 1) ≈ ±4.85·10^8** (verified against Hafner's published code at `embodied/jax/heads.py:87–97`, which constructs `half = symexp(linspace(-20, 0, ...))` then mirrors). The target for a continuous value `x` is `twohot(symlog(x))` — a soft label placing mass on the two bins adjacent to `symlog(x)` in proportion to closeness. Loss is cross-entropy. Point estimates are read out as `symexp(p^T B)` where `p` is the softmax. This gives a discrete-output regressor whose loss landscape has the same shape across return scales and which tolerates bimodal target distributions.

**Percentile return scaling** (Eq. 12). Define the dispersion `S = EMA(Per(R^lambda, 95) - Per(R^lambda, 5), rho=0.99)`. The actor's policy gradient is divided by `max(1, S)`. Sparse returns (small `S`) pass through unchanged so the entropy bonus does not destroy the rare-reward signal; dense returns (large `S`) are scaled down so the entropy bonus is not overwhelmed. The 5th-to-95th-percentile range (instead of std) is robust to heavy-tailed return distributions. With this single trick a fixed entropy coefficient `eta = 3e-4` works across dense and sparse domains.

### §2.3 World-model loss (Eqs. 4–5)

The world-model loss is

```
L(phi) = beta_pred * L_pred + beta_dyn * L_dyn + beta_rep * L_rep
```

where

- `L_pred = -ln p_phi(x_t | h_t, z_t) - ln p_phi(r_t | h_t, z_t) - ln p_phi(c_t | h_t, z_t)` — the "what to model" objective: reconstruction, reward, continuation.
- `L_dyn = max(1, KL(sg(q_phi(z_t | h_t, x_t)) || p_phi(z_t | h_t)))` — pushes the prior toward the (stop-grad) posterior.
- `L_rep = max(1, KL(q_phi(z_t | h_t, x_t) || sg(p_phi(z_t | h_t))))` — pushes the posterior toward the (stop-grad) prior.

The `max(1, ·)` floor is "free bits" — the loss is silenced once each KL has fallen below 1 nat, preventing the regulariser from collapsing already-informative latents. The asymmetric weights (preprint: `beta_pred=1, beta_dyn=0.5, beta_rep=0.1`; Nature: `beta_pred=1, beta_dyn=1.0, beta_rep=0.1`) generalise DreamerV2's KL balancing — the prior is updated more aggressively than the posterior is regularised. The Nature version pushes this asymmetry from 5× to 10×.

### §2.4 Behaviour learning in imagination (Eqs. 11–12)

After every world-model gradient update, behaviour learning runs entirely on imagined rollouts:

1. Take all `B * T` posterior states from the replay batch as start states.
2. Roll the world model forward for `T_imag = 15` steps, sampling actions from the actor and applying `imagine_step` (no posterior, just the prior) at every step. Predict reward and continuation from the heads at each step.
3. Compute λ-returns recursively from the tail (Eq. 11):
   ```
   R^lambda_t = r_t + gamma * c_t * [(1 - lambda) * v(s_{t+1}) + lambda * R^lambda_{t+1}]
   R^lambda_T = v(s_T)
   ```
   with `gamma = 0.997`, `lambda = 0.95`.
4. Critic loss = cross-entropy of the critic's two-hot output against `twohot(symlog(R^lambda_t))` (preprint Eq. 10; Nature Eq. of the same name re-frames as maximum-likelihood `-ln p_psi(R^lambda_t | s_t)` and adds a replay-value term with weight 0.3).
5. Actor loss (Eq. 11) = REINFORCE with the critic baseline plus an entropy bonus, with the policy gradient divided by `max(1, S)`:
   ```
   L(theta) = -sum_t E_pi[sg(R^lambda_t - v_psi(s_t)) / max(1, S)] - eta * H[pi(a_t | s_t)]
   ```
6. The critic is regularised toward an EMA of itself (preprint) or toward a hard "slow target critic" updated by EMA per gradient step (Nature wording uses target-network framing).

### §2.5 Other paper-canonical ingredients (categorical machinery, replay, architecture)

- **Unimix categoricals** (Hafner 2023 §A.1): every categorical — RSSM prior, RSSM posterior, actor — is `0.99 * NN_softmax + 0.01 * uniform`. Caps log-probabilities at `ln(K / 0.01)` and avoids rare KL spikes.
- **Subsequence replay**: the replay buffer no longer waits for episode completion; subsequences of length `L = 64` are sampled uniformly over the entire buffer.
- **Architecture**: layer normalisation (preprint) / RMSNorm (Nature), SiLU activations, same-padded stride-2 kernel-3 convolutions for vision; per-component widths scale across XS / S / M / L / XL profiles.

### §2.5b Replay ratio — the compute–data trade-off knob

The replay ratio is the single biggest data-efficiency knob in DreamerV3 — Hafner 2023 Fig. 6 sweeps it from 1/16 to 64 and shows monotonic sample-efficiency improvement at the cost of WM saturation on the modal policy. **Per Hafner 2023 Table A.1, the default is benchmark-dependent**: Atari 200M = 64, DMC = 512, BSuite = 1024, Atari 100K = 1024, Minecraft = 16, DMLab = 64. In the Nature-style "replay ratio = replay-steps / env-step / minibatch-length / action-repeat" normalised semantics, this gives a factor-of-8 spread across benchmarks (Atari 200M ≈ 0.0156 grad/env, DMC ≈ 0.0625 grad/env in our YAML semantics). The "1/16 ≈ 0.0625" frequently quoted as canonical is the Atari/DMC mid-band of this spread, **not** a single universal default. This matters for §6 item interpretation: a "deviation" magnitude depends on which benchmark you compare against.

### §2.6 Paper-canonical hyperparameters (Hafner 2023 Tables B.1, W.1; Hafner 2025 Supplementary Table)

| Knob | Value | Source |
|---|---|---|
| Discount `gamma` | 0.997 | both |
| GAE `lambda` | 0.95 | both |
| Imagination horizon `H` | **15 (both versions, per preprint Table W.1 and Nature Ext. Data Table 5)** | both |
| Two-hot bins `K` | 255 | both |
| Two-hot symlog range | linspace(-20, +20) **in symlog space** → raw bin centres `symexp([-20, +20]) ≈ ±4.85·10^8` | both |
| Entropy coefficient `eta` | 3e-4 | both |
| Unimix | 0.01 | both |
| Free-bits floor | 1 nat per KL term (per state, after summing over stoch-group axis) | both |
| Batch size `B` | 16 | both |
| Sequence length `L` (batch length `T`) | 64 | both |
| `beta_pred` | 1 | both |
| `beta_dyn` | 0.5 (preprint) / 1.0 (Nature) | preprint vs. Nature |
| `beta_rep` | 0.1 | both |
| `beta_repval` (replay-value) | — (preprint) / 0.3 (Nature) | Nature only |
| Replay ratio (default) | **benchmark-dependent (Atari 200M=64, DMC=512, BSuite=1024, Minecraft=16 train ratio per Table A.1)**; ≈0.0625 grad/env in normalised DMC mid-band, factor-of-8 spread across benchmarks | both |
| Buffer capacity | 10^6 (preprint Table W.1 implicit) / **5·10^6 (Nature Ext. Data Table 5)** | preprint vs. Nature |
| Percentile EMA decay `rho` | 0.99 | Nature explicit |
| Percentile range | 5th–95th | both |
| RSSM categorical | 32 groups × 32 classes | both |
| Adam epsilon | **`1e-8` (WM) / `1e-5` (AC) — asymmetric split per Hafner 2023 Table W.1**; Nature replaces Adam with LaProp (`ε = 1e-20`) entirely | preprint Table W.1 |
| Optimiser | Adam (preprint) / **LaProp + AGC(0.3) (Nature)** | preprint vs. Nature |
| Learning rate | WM `1e-4`, actor/critic `3e-5` (preprint Table W.1) / **single uniform `4e-5` (Nature Ext. Data Table 5)** | preprint vs. Nature |
| World-model grad clip | global-norm 1000.0 (preprint) / **AGC(0.3) per-tensor (Nature, not global-norm)** | preprint vs. Nature |
| GRU size (S profile) | 512 deter | both |
| Activation / norm | LayerNorm + SiLU (preprint) / **RMSNorm + SiLU (Nature)** | preprint vs. Nature |

These defaults are the bar against which §3 and §5 judge "matches paper / deviates from paper".

---

## §3 Implementation map

This section walks every architectural and algorithmic component of our active codebase. **Active modules** (the ones the trainer actually imports): `src/models/dreamer_v3_nnx.py` (Flax NNX model definitions), `src/models/dreamer_v3_trainer.py` (training loop + losses + imagination + buffers + JIT scan), `src/models/dreamer_v3_util.py` (`symlog`/`symexp`/`to_twohot`/`from_twohot`/`OneHotDist`/`Moments`/`Ratio`/`hafner_init`), `configs/models/dreamer_v3.yaml` (canonical config), and the DreamerV3 branch of `train.py` from L773. The legacy file `src/models/dreamer_v3_network.py` is **NOT imported** anywhere — it is an 80-LOC scratch file confirmed orphaned by `grep -rln "dreamer_v3_network"`. It is flagged in §6 but otherwise ignored here.

### §3.1 World model — RSSM

- **Paper spec.** Deterministic GRU state `h_t` plus 32-group × 32-class straight-through categorical `z_t`. Prior `p_phi(z_t | h_t)` and posterior `q_phi(z_t | h_t, x_t)` share the recurrent core; on episode boundaries the carry is reset.
- **Code spec.** `src/models/dreamer_v3_nnx.py:41–139`. Dimensions read from config: `rssm_deter_dim=512`, `rssm_stoch_dim=32`, `rssm_classes=32`, giving `feat_dim = deter_dim + stoch_dim * discrete = 512 + 1024 = 1536`. The recurrent core is a custom **`LayerNormGRUCell`** at `nnx.py:18–39` — three dense gates (reset / update / candidate) each pre-LayerNormed before being summed across the input and hidden projections, with `tanh` on the candidate gate and `sigmoid` on reset/update. Prior head: `img_in: Linear(stoch_dim*discrete + action_dim, deter_dim) → SiLU → GRU → img_out: Linear(deter_dim, stoch_dim*discrete) → reshape (B, 32, 32)` (`nnx.py:51–58, 86–95`). Posterior head: `obs_out: Linear(deter_dim + embed_dim, stoch_dim*discrete) → reshape (B, 32, 32)` (`nnx.py:59, 97–99`). `is_first` reset (`nnx.py:81–83`):
  ```python
  mask = (1.0 - is_first).astype(jnp.float32).reshape((-1, 1))
  deter = prev_state['deter'] * mask
  stoch = prev_state['stoch'] * mask
  ```
  zeroes both the deterministic and the stochastic carry on episode boundaries. `imagine_step` (`nnx.py:110–139`) is the dynamics-only counterpart for behaviour learning — no posterior, no `is_first` reset, samples `stoch` from `prior_logits` via `OneHotDist`.
- **Deviation flag — `is_first` resets both `deter` and `stoch`.** `MINOR DEVIATION (justified)`. Hafner's published code zeroes only `stoch` (the prior is conditioned on a fresh GRU step regardless); zeroing `deter` is a stricter reset that makes the first-step prior independent of the previous episode's hidden state. Functionally similar but not identical. (Inventory red-flag #24.)
- **Deviation flag — GRU candidate uses `tanh`, not SiLU.** `MATCHES PAPER`. Standard GRU; Hafner's implementation also uses tanh on the candidate. (Inventory red-flag #22 dismissed.)

### §3.2 Encoder — `DreamerObservationEncoder` (and the hierarchical hub)

- **Paper spec.** Hafner 2023 uses a CNN encoder for visual domains (same-padded, stride-2, kernel-3, with channel multipliers per profile) and an MLP encoder for proprioceptive domains. The encoder consumes `symlog(x_t)` and emits a fixed-width embedding to feed the RSSM posterior head. Encoder body uses LayerNorm + SiLU after every linear / conv.
- **Code spec.** `nnx.py:197–313` (hierarchical) and `nnx.py:316–370` (flat). Selected via the mandatory key `agent.encoding_mode` ∈ `{"hierarchical", "flat"}`; missing the YAML key raises `ValueError` from `config.get_mandatory('agent.encoding_mode', str)` at `trainer.py:78` (the secondary check at `nnx.py:204–205` raises `ValueError` only when a `None` config is passed to the encoder constructor). The hierarchical path is a project-specific structure for the multi-sensor proprioceptive observation:
  - **Phase 1 — per-sensor MLPs**: `DreamerGroupedMLP(num_groups=G, max_in, default_mlp=[128,128], hidden_size=embed_dim=128)` — one MLP per sensor group, einsum-batched (`nnx.py:171–194`); LayerNorm + SiLU per hidden layer.
  - **Phase 2 — multimodal hub**: `MLP(G * embed_dim, embed_dim, multimodal_hub=[128,128])` — LayerNorm + SiLU per hidden layer (`nnx.py:218–230`).
  - **Final SiLU** after a (possibly modulated) Phase-2 output. The split between `body` (linear → LN → SiLU stack with no final SiLU) and `final_act` (the trailing SiLU) exists so neuromodulation gain/bias can inject between LN and the final non-linearity (`forward_with_modulation`, `nnx.py:265–313`).
  Flat path (`nnx.py:316–370`): standard `Encoder` with `encoder_fc_layers = [128, 128]` hidden + final `Linear(_, 128)` + LayerNorm → SiLU final. Symlog is applied **inside** `train_step` (`trainer.py:122`) and `get_action` (`trainer.py:530`), never inside the encoder.
- **Deviation flag — vector observations + MLP-only encoder.** `MAJOR DEVIATION (deliberate)`. The paper's encoder is CNN for vision and MLP for proprio; we have a 19-channel proprioceptive observation only and use the MLP path exclusively. The hierarchical hub is a project-specific addition to handle the heterogeneous sensor groups (Injury, Nutrition, Satiation, Extero Nociception, Olfaction, Collision, Proprioception, Visual, Location). Documented in [active_inference_hypervigilance.md](./active_inference_hypervigilance.md) and the project's sensor-design docs.
- **Deviation flag — small width.** `MAJOR DEVIATION (deliberate)`. Encoder hidden 128, embed dim 128 — Hafner's S profile uses 512 GRU + 32 channels CNN; commented-out XL block at `dreamer_v3.yaml:60–74` mentions 1024. Active config is far smaller than any Hafner profile. (Inventory red-flag #4.)
- **Deviation flag — three LayerNorms in series at the embed bottleneck when `use_layer_norm=true`.** `MINOR DEVIATION (justified)`. The body has a per-linear LN, the encoder appends a final LN before the final SiLU, and a separate `mod_*_ln` (one of `mod_unimodal_ln` / `mod_multimodal_ln` / `mod_flat_ln`, `nnx.py:516–521`) is applied between the body and the trailing SiLU when `use_layer_norm=true`. Likely more LN than canonical; preserved for compatibility with the modulation injection sites. (Inventory red-flag #14.)

### §3.3 Decoder — `DreamerObservationDecoder`

- **Paper spec.** Symmetric to the encoder. Output target is `symlog(x_t)`; squared loss is computed in symlog space.
- **Code spec.** `nnx.py:372–422`. Takes `feat = concat(deter, stoch)` of dim 1536, outputs raw `obs_dim` vector. Hierarchical path uses `MLP(feat_dim, G * encoder_dim, multimodal_hub_layers=[128,128])` expansion → reshape `(..., G, 128)` → `DreamerGroupedMLP(G, 128, default_mlp=[128,128], max_out)` per-sensor decoder → concat sensor slices to `obs_dim`. Flat path uses `Decoder(feat_dim, obs_dim, decoder_fc_layers=[128,128])` — Linear → LN → SiLU stack with **no final activation** on the output (`nnx.py:432–441`):
  ```python
  for h in fc_layers:
      layers.append(nnx.Linear(in_d, h, kernel_init=hafner_init(), rngs=rngs))
      layers.append(nnx.LayerNorm(h, rngs=rngs))
      layers.append(SiLU())
      in_d = h
  layers.append(nnx.Linear(in_d, output_dim, kernel_init=hafner_init(), rngs=rngs))
  ```
  Output is in **symlog space**; the loss target is `obs = symlog(batch['obs'])` (`trainer.py:122`); MSE is computed in symlog space and `symexp` is **never** applied to the decoder output. Inversion happens only externally (e.g., the offline diagnostic at `scripts/dreamer_offline_wm_test.py:269` pre-applies `symlog` to the ground truth).
- **Deviation flag.** `MATCHES PAPER` for the symlog-target convention; `MAJOR DEVIATION (deliberate)` for the small width (same justification as §3.2).

### §3.4 Heads — reward, continue, actor, critic, slow target critic

#### §3.4.1 Reward head — `wm.reward_head`

- **Paper spec.** 255-way two-hot softmax over symlog-spaced bins in [-20, +20]. Cross-entropy against `twohot(symlog(r_t))`.
- **Code spec.** `nnx.py:530`. `MLP(feat_dim=1536, 255, reward_fc=[128,128])` — `Linear(1536, 128) → LN → SiLU → Linear(128, 128) → LN → SiLU → Linear(128, 255)` with no final activation (the loss applies `log_softmax`). Bin layout `min_v=-20, max_v=20, num_buckets=255` lives in `to_twohot`/`from_twohot` defaults (`util.py:19, 60`); bins are spaced linearly **in symlog space** per the comment at `util.py:25–29`. The 255 magic number is hard-coded in both the model definition and the critic. **There is no config knob for two-hot bin count.**
- **Deviation flag.** `MATCHES PAPER` for the head structure and bin layout. `MINOR DEVIATION (suspected unjustified)` for the bin count being a hard-coded literal in three places (`nnx.py:530, 576`; `util.py:19, 60`) rather than a config knob.

#### §3.4.2 Continue head — `wm.continue_head`

- **Paper spec.** Bernoulli over "episode continues" (`1 - terminated`), trained with binary cross-entropy.
- **Code spec.** `nnx.py:531`. `MLP(feat_dim, 1, continue_fc=[128,128])` — `Linear(1536, 128) → LN → SiLU → Linear(128, 128) → LN → SiLU → Linear(128, 1)`. Output is a raw logit; sigmoid is applied at consumption sites (`trainer.py:270, 366, 387; scripts/dreamer_offline_wm_test.py:256`).
- **Deviation flag.** `MATCHES PAPER`.

#### §3.4.3 Actor head — `agent.ac.actor`

- **Paper spec.** Categorical over discrete actions (with REINFORCE estimator) or continuous (with reparameterised Gaussian). Logits passed through a 1% unimix.
- **Code spec.** `nnx.py:563–576`. `MLP(feat_dim, act_dim, actor_fc=[128,128])` → no final softmax; the distribution is wrapped via `OneHotDist(logits, unimix=0.01)` at consumption (`trainer.py:348, 380, 571`). Action-dim source: `act_dim = 4 + int(rest_action_enabled) + int(eat_action_enabled)` computed in env wiring at `train.py:506` (canonical = 4 with no rest/eat).
- **Deviation flag.** `MATCHES PAPER` for the discrete-action recipe.

#### §3.4.4 Critic head — `agent.ac.critic`

- **Paper spec.** 255-way two-hot softmax (same bin layout as reward head). Trained on `twohot(symlog(R^lambda_t))` via cross-entropy. Nature paper re-frames as maximum-likelihood `-ln p_psi(R^lambda_t | s_t)`; same loss numerically.
- **Code spec.** `nnx.py:576`. `MLP(feat_dim, 255, critic_fc=[128,128])`. Used as the **baseline** in the actor advantage (`trainer.py:433`): `baseline = from_twohot(v_pred_logits)`.
- **Deviation flag.** `MATCHES PAPER` for the head structure. **`MAJOR DEVIATION (deliberate)`** for the absence of the Nature replay-value loss term — see §3.4.6.

#### §3.4.5 Slow target critic — and its NOT being the λ-return bootstrap source in either paper

- **Paper spec.** **Both Hafner 2023 preprint and Hafner 2025 Nature compute λ-returns using the *fast (online) critic*, NOT the slow target critic.** The slow EMA copy is used *only* as a regularisation target inside the critic loss — preprint Eq. 10's auxiliary self-EMA, Nature page 3 verbatim: "regularizing the critic towards predicting the outputs of an exponentially moving average of its own parameters... but allows us to compute returns using **the current critic network**". Preprint Appendix C item 6 says verbatim: "We compute λ-returns using the **fast critic** network and regularize the critic outputs towards those of its own weight EMA instead of computing returns using the slow critic. However, both approaches perform similarly in practice." Preprint Appendix D.2 ablates the alternative `SlowTarget` variant ("Instead of using the fast critic for computing returns and training it towards the slow critic, use the slow critic for computing returns") and reports verdict: **"no benefit"**.
- **Code spec.** Implemented as a **separately-instantiated** `ActorCritic(...).critic` whose `nnx.Param` tree is updated by EMA after every `train_step`. Instantiated at `trainer.py:87`. EMA update at `trainer.py:501–505`:
  ```python
  current_st = nnx.state(self.agent.ac.critic, nnx.Param)
  target_st = nnx.state(self.target_critic, nnx.Param)
  new_target_st = jax.tree.map(lambda t, c: 0.98 * t + 0.02 * c, target_st, current_st)
  nnx.update(self.target_critic, new_target_st)
  ```
  Decay is hard-coded `0.98 * target + 0.02 * online` per gradient step. **The target critic is used as the value bootstrap inside λ-return computation at `trainer.py:367, 388, 411`** — the online critic is used only as the **baseline** in the actor advantage (`trainer.py:433`) and the prediction in the critic loss (`trainer.py:427`). No periodic hard-copy / reset on env-stage transitions — only the buffer is cleared at stage transitions, not the critic (`train.py:1116–1136`).
- **Deviation flag — λ-return bootstrap uses `target_critic`.** **`MAJOR DEVIATION (suspected unjustified — corresponds to the preprint's ablated `SlowTarget` variant)`**. Both papers explicitly compute λ-returns using the fast/online critic (preprint Appendix C item 6 + Nature page 3); preprint Appendix D.2 ablates our exact recipe and reports "no benefit". This deviation reframes §3.4.5: a slow-target-critic regulariser **is** a real DreamerV3 mechanism (see preprint Eq. 10 critic-EMA self-regularisation), but it is **not** used as the λ-return bootstrap source in either paper. *This deviation warrants a controlled `SlowTarget`-vs-`current-critic` comparison run* — handed off to `senior-developer` and `experiment-designer` as a follow-up plan. Forward-link: `.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md` (the reward-head failure may compound with this critic-bootstrap mismatch).
- **Deviation flag — hard-coded EMA decay.** `MINOR DEVIATION (suspected unjustified)` that the EMA decay is a hard-coded literal `0.98 / 0.02` with no config knob. (Inventory red-flag #19.)

#### §3.4.6 Replay-value loss (Nature only)

- **Paper spec (Nature).** Critic loss is `-sum_t [beta_val * ln p_psi(R^lambda_t | s_t) + beta_repval * ln p_psi(R^lambda,replay_t | s^replay_t)]` with `beta_val = 1, beta_repval = 0.3`. The replay-side λ-return is computed by recursively unrolling on the replay reward sequence with the imagined-rollout `R^lambda` at the final replay step as the bootstrap value.
- **Code spec.** **NOT IMPLEMENTED.** `behavior_loss_fn` (`trainer.py:342–485`) trains the critic only on imagined-rollout λ-returns. There is no replay-side critic loss term anywhere in the codebase. Confirmed by `grep -n "repval\|replay_value\|ln p_psi" src/models/dreamer_v3_trainer.py` returning nothing.
- **Deviation flag.** `MAJOR DEVIATION (deliberate)`. We are running Hafner-2023 (preprint) critic semantics, not Hafner-2025 (Nature). The Nature replay-value term is a stabiliser for hard-prediction domains; whether our setup needs it is an open question (see [`docs/project/critiques/dreamer_conventional_failure_modes_for_our_setup.md`](../critiques/dreamer_conventional_failure_modes_for_our_setup.md) §2 for related diagnostic candidates).

### §3.5 Loss terms

All world-model losses are summed inside `model_loss_fn` (`trainer.py:133–320`); actor + critic losses are summed inside `behavior_loss_fn` (`trainer.py:342–485`). The optimiser is invoked separately for `wm`, `actor`, `critic` (`trainer.py:322–325, 487–499`). Constants (`trainer.py:17–23`):

```python
FREE_NATS = 1.0
KL_SCALE = 1.0      # declared but never referenced
DYN_SCALE = 0.5
REP_SCALE = 0.1
HORIZON = 15
GAMMA = 0.997
LAMBDA = 0.95
```

#### §3.5.1 Reconstruction loss — `loss_recon`

- **Paper spec.** `-ln p_phi(x_t | h_t, z_t)` term of `L_pred`. Implemented as squared loss in symlog space.
- **Code spec.** `trainer.py:213–215`:
  ```python
  feat = wm.get_feat(posts)
  recon = wm.decoder(feat)
  loss_recon = jnp.mean(jnp.square(recon - obs))
  ```
  Targets are `obs = symlog(batch['obs'])` (`trainer.py:122`). Combined into total: `total_loss = loss_recon + loss_rew + CONT_LOSS_WEIGHT * loss_cont + loss_kl` (`trainer.py:252`). Coefficient on recon: **1.0** (no scalar multiplier).
- **Deviation flag.** `MATCHES PAPER`. (No per-channel weighting; all sensors carry equal weight — Hafner uses unit weighting too.)

#### §3.5.2 Reward loss — `loss_rew`

- **Paper spec.** Cross-entropy on the two-hot soft target `twohot(symlog(r_t))`.
- **Code spec.** `trainer.py:217–220`:
  ```python
  rew_pred = wm.reward_head(feat)
  rew_target = to_twohot(reward)              # to_twohot internally applies symlog
  loss_rew = -jnp.mean(jnp.sum(rew_target * jax.nn.log_softmax(rew_pred), axis=-1))
  ```
  Targets are raw `reward` from the buffer; `to_twohot` does its own internal symlog (`util.py:24`). Coefficient: **1.0**.
- **Deviation flag.** `MATCHES PAPER`.

#### §3.5.3 Continue loss — `loss_cont`

- **Paper spec.** Binary cross-entropy on the continuation Bernoulli with target `1 - terminated`.
- **Code spec.** `trainer.py:222–224`:
  ```python
  cont_pred = wm.continue_head(feat)
  loss_cont = optax.sigmoid_binary_cross_entropy(cont_pred, 1.0 - terminal[..., None]).mean()
  ```
  Coefficient: `CONT_LOSS_WEIGHT = config.get_mandatory('agent.cont_loss_weight', float)` (`trainer.py:251`). Canonical value **1.0** (`dreamer_v3.yaml:41`).
- **Deviation flag.** `EXTENSION (not in paper)` for the configurable multiplier. The default 1.0 matches Hafner; the knob was added defensively for diagnostic experiments. (Inventory red-flag #16.)

#### §3.5.4 KL dynamics + KL representation — `loss_dyn_kl`, `loss_rep_kl`, `loss_kl`

- **Paper spec.** Two free-bits-clamped KL terms with asymmetric weights:
  - `L_dyn = beta_dyn * max(1, KL(sg(q) || p))` — pushes prior toward posterior.
  - `L_rep = beta_rep * max(1, KL(q || sg(p)))` — pushes posterior toward prior.
  Preprint: `beta_dyn=0.5, beta_rep=0.1`. Nature: `beta_dyn=1.0, beta_rep=0.1`.
- **Code spec.** `trainer.py:226–249`:
  ```python
  q_logits = posts['logits']
  p_logits = priors['logits']
  def kl_div_categ(p_logits, q_logits):
      p_dist = jax.nn.softmax(p_logits)
      p_log = jax.nn.log_softmax(p_logits)
      q_log = jax.nn.log_softmax(q_logits)
      return jnp.sum(p_dist * (p_log - q_log), axis=-1)
  q_logits_sg = jax.lax.stop_gradient(q_logits)
  p_logits_sg = jax.lax.stop_gradient(p_logits)
  dyn_kl = kl_div_categ(q_logits_sg, p_logits)   # gradient flows to prior
  rep_kl = kl_div_categ(q_logits, p_logits_sg)   # gradient flows to posterior
  dyn_kl = jnp.sum(dyn_kl, axis=-1)              # sum over stoch_dim groups (32 categories)
  rep_kl = jnp.sum(rep_kl, axis=-1)
  dyn_kl = jnp.maximum(dyn_kl, FREE_NATS)        # free-nats clip
  rep_kl = jnp.maximum(rep_kl, FREE_NATS)
  loss_kl = DYN_SCALE * jnp.mean(dyn_kl) + REP_SCALE * jnp.mean(rep_kl)
  ```
  Free-nats is applied **per-`(B, T)` state, after `jnp.sum(..., axis=-1)` collapses BOTH the 32 discrete-class axis (line 230, inside `kl_div_categ`) AND the 32 stoch-group axis (lines 243-244)**. So the clipping floor is 1.0 nat per `(B, T)` entry — i.e. the lower bound on the loss term is `0.5 · 1.0 + 0.1 · 1.0 = 0.6` nats averaged across batch+time. (An earlier draft of this doc misread the `(B, T, stoch=32)` axis as the clip target; the actual clamp at line 246–247 operates on the post-aggregation `(B, T)` tensor.)
- **Deviation flag — `KL_SCALE=1.0` declared but never used.** `MINOR DEVIATION (suspected unjustified)`. The constant at `trainer.py:18` is dead code. Suggests a refactor stranded it; the actual KL combine is `0.5*dyn + 0.1*rep`, not `1.0 * (0.5*dyn + 0.1*rep)`. (Inventory red-flag #8.)
- **Deviation flag — `beta_dyn = 0.5`.** `MAJOR DEVIATION (deliberate)` against Nature; `MATCHES PAPER` against preprint. We follow Hafner-2023 weights, not Hafner-2025. Per the Nature ablation (Fig. 6), this is the second-most-impactful loss-weight change in the algorithm; we have not adopted it.
- **Deviation flag — free-nats applied per `(B, T)` state.** `MATCHES PAPER`. The official Hafner implementation at `dreamerv3/rssm.py` uses `embodied.jax.outs.Agg(out, 1, jnp.sum)` to aggregate `OneHot.kl(...)` over the stoch-group axis **before** the `jnp.maximum(dyn, free_nats)` clip — i.e. clips per `(B, T)` state. Our code at lines 243-247 reproduces this exact axis behaviour. Cross-validated by `math-reviewer` against the published source — the earlier "per-stoch-group" framing in this doc was a description error, not a behaviour deviation.

#### §3.5.5 λ-return computation — `compute_lambda_values`

- **Paper spec (Eq. 11).** `R^lambda_t = r_t + gamma * c_t * [(1 - lambda) * v(s_{t+1}) + lambda * R^lambda_{t+1}]` with `R^lambda_T = v(s_T)`, computed by reverse scan.
- **Code spec.** `trainer.py:28–51`:
  ```python
  def compute_lambda_values(rewards, values, continues, LAMBDA=0.95):
      next_vals = values[1:]
      def scan_fn(next_return, inputs):
          r, v, c = inputs
          bootstrap = (1 - LAMBDA) * v + LAMBDA * next_return
          current_return = r + c * bootstrap
          return current_return, current_return
      inputs = (rewards, next_vals, continues)
      last_val = values[-1]
      _, returns = jax.lax.scan(scan_fn, last_val, inputs, reverse=True)
      return returns
  ```
  Caller (`trainer.py:413–416`):
  ```python
  all_vals = jnp.concatenate([v_start[None], vals], axis=0)
  lambda_returns = compute_lambda_values(rews, all_vals, conts * GAMMA)
  ```
  `rews` shape `(HORIZON=15, IMAG_BATCH)`; `values` shape `(HORIZON+1, IMAG_BATCH)` (prepended `v_start = from_twohot(target_critic(start_feat))`). The `continues` argument is the **head-predicted continuation probability multiplied by the global discount** (`cont_pred * GAMMA`), so `gamma` enters the recursion through `c_t` rather than as a separate scalar in the formula. The bootstrap value at the tail is `values[-1]` — the H-th imagined step's target-critic estimate. **Important — the `v` here is `target_critic`, NOT the online critic** (see §3.4.5): both Hafner papers compute λ-returns using the fast critic and our use of `target_critic` as the bootstrap source is the preprint's ablated `SlowTarget` variant. This is one of the two roles the slow critic plays in our codebase (the other is the regularisation target inside the critic loss); only the latter is a paper-canonical role.
- **Deviation flag — γ folded into `c_t`.** `MATCHES PAPER`. The folding of `gamma` into `c_t` is algebraically equivalent to the paper formula.
- **Deviation flag — bootstrap value uses `target_critic`.** **`MAJOR DEVIATION (suspected unjustified)`**, see §3.4.5 for full evidence (preprint Appendix C item 6 + Nature page 3 + preprint Appendix D.2 ablation).

#### §3.5.6 Critic loss — `loss_critic`

- **Paper spec.** Cross-entropy on `twohot(symlog(stop_grad(R^lambda_t)))`. Per-step loss multiplied by a cumulative discount weight so later imagined steps contribute less.
- **Code spec.** `trainer.py:421–430`:
  ```python
  discount_weights = jnp.concatenate([jnp.ones_like(conts[:1]), conts[:-1] * GAMMA], axis=0)
  discount_weights = jnp.cumprod(discount_weights, axis=0)
  discount_weights = jax.lax.stop_gradient(discount_weights)
  # Critic Loss — train on RAW lambda_returns (canonical DreamerV3)
  v_pred_logits = critic(rollouts['feat'])
  target_twohot = to_twohot(jax.lax.stop_gradient(lambda_returns))
  loss_critic_step = -jnp.sum(target_twohot * jax.nn.log_softmax(v_pred_logits), axis=-1)
  loss_critic = jnp.mean(loss_critic_step * discount_weights)
  ```
  Critic is regressed on **raw** λ-returns (NOT the percentile-normalised version) — the comment at L426 explicitly calls this out. `to_twohot` internally applies symlog. The critic's `feat` is the start-of-step feat (`rollouts['feat']` = `prev_state` feat, set inside `scan_imag`, `trainer.py:371, 392`), so the critic is regressed on the value at the state *from which* an action was taken.
- **Deviation flag.** `MATCHES PAPER` (preprint). Critic-on-raw-returns is the canonical preprint recipe.

#### §3.5.7 Actor loss — `loss_actor`

- **Paper spec (preprint Eq. 11).** REINFORCE with critic baseline plus entropy bonus, with the policy-gradient term divided by `max(1, S)`:
  `L(theta) = -sum_t E_pi[sg(R^lambda_t - v_psi(s_t)) / max(1, S)] - eta * H[pi(a_t | s_t)]`
- **Code spec.** `trainer.py:432–445`:
  ```python
  baseline = from_twohot(v_pred_logits)
  norm_baseline = (baseline - moments_low) / moments_invscale
  norm_returns  = (lambda_returns - moments_low) / moments_invscale
  advantage = jax.lax.stop_gradient(norm_returns - norm_baseline)
  actions = rollouts['action']
  logits = rollouts['action_dist']
  log_probs = jnp.sum(actions * jax.nn.log_softmax(logits), axis=-1)
  ENTROPY_SCALE = self.config.get_mandatory('agent.entropy_scale', float)
  entropy = -jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1)
  loss_actor_step = -(log_probs * advantage + ENTROPY_SCALE * entropy)
  loss_actor = jnp.mean(loss_actor_step * discount_weights)
  ```
  Form: REINFORCE with baseline (NOT PPO-clip; NOT A2C). Advantage is stop-gradient'd, so no value-baseline gradient flows into the actor loss path. `actions` are the **straight-through one-hot samples** from `OneHotDist.sample(key)` (so the backward pass routes the gradient through the softmax probs of `logits` per `util.py:103–106`). `discount_weights` reused from §3.5.6.
- **Deviation flag — both sides percentile-normalised.** `MINOR DEVIATION (suspected unjustified)`. The comment at L432 explicitly says: "Actor Loss — normalize BOTH sides for consistent advantage". The paper formula is `(R^lambda - v_psi) / max(1, S)` — i.e. the *difference* is divided by the scale; with linear normalisation `(R - low) / scale - (v - low) / scale = (R - v) / scale` so this is algebraically equivalent. Recording for completeness; effect zero.
- **Deviation flag — `entropy_scale = 3e-4` matches paper.** `MATCHES PAPER`.

#### §3.5.8 Aggregate optimiser steps

- **World-model step.** `total_loss = loss_recon + loss_rew + CONT_LOSS_WEIGHT * loss_cont + loss_kl` → `nnx.grad` → `model_opt.update(wm, grads_model)` (`trainer.py:252, 322–325`).
- **Behaviour step.** `(loss_actor + loss_critic)` → `nnx.grad(..., argnums=(0,1))` returning `(grads_actor, grads_critic)` → separate `actor_opt.update(...)` and `critic_opt.update(...)` (`trainer.py:485–499`).
- **Moments update is OUTSIDE the gradient.** `self.moments.update(lambda_returns)` runs after `nnx.grad`, so the moments used inside the gradient are stale by one step (`trainer.py:336–340, 495`). The comment at L337–338 calls this out as deliberate to avoid tracing through `self.moments` inside `nnx.grad`, which causes OOM.
- **Target critic EMA happens AFTER the optimiser step** (`trainer.py:501–505`).

### §3.6 Buffer + sampling

#### §3.6.1 `ReplayBuffer` container

- **Paper spec.** Sub-sequence sampling (length 64) uniformly across the entire buffer; capacity not strictly specified but typical `1M`.
- **Code spec.** `trainer.py:917–1017`. Capacity `1_000_000` transitions (`dreamer_v3.yaml:14`). `sequence_length = 128` (`dreamer_v3.yaml:4`). Storage backend selected via `agent.buffer_device ∈ {"gpu", "cpu"}` (`trainer.py:917–935`); canonical = `"gpu"`. Storage tensors per slot: `obs (capacity, obs_dim)`, `actions (capacity, act_dim)`, `rewards (capacity,)`, `dones (capacity,)`, `is_first (capacity,)`. The buffer is **env-major**: the first `sequence_length` slots = env-0's trajectory, next `sequence_length` slots = env-1's trajectory, etc. Comment at `trainer.py:944`: "Caller must pass data in ENV-MAJOR order". `add_batch` (`trainer.py:940–971`) writes `num_items` rows starting at `self.idx`, modular `self.capacity`. `sample(batch_size, key)` (`trainer.py:973–1006`) samples **block-aligned starts only**:
  ```python
  num_blocks = self.size // self.sequence_length
  if num_blocks < 1: return None
  block_indices = jax.random.randint(key, (batch_size,), 0, num_blocks)
  starts = block_indices * self.sequence_length
  indices = (starts[:, None] + seq_range[None, :]) % self.capacity
  ```
  Returns `None` if the buffer holds <1 full block.
- **Deviation flag — `sequence_length = 128`.** `MAJOR DEVIATION (deliberate)`. Hafner uses `L = 64` for both Atari and DMC (Hafner 2023 Table B.1; Hafner 2025 Supplementary). 2× the canonical sequence length.
- **Deviation flag — block-aligned sampling.** `MAJOR DEVIATION (deliberate)`. Hafner samples uniformly over all valid sub-sequence starts; we sample only from a discrete grid of block boundaries. Sampled sequences never straddle env-boundaries inside a single batch row but CAN straddle episode-boundaries within an env's trajectory (the `is_first` flag inside the sequence handles RSSM resets). This is a project-specific choice driven by env-major storage and JIT-friendly indexing.

#### §3.6.2 Mixture sampling — DreamerV4-inspired positive-reward buffer

- **Paper spec.** **NOT IN HAFNER 2023 / 2025.** Hafner uses uniform sub-sequence replay only. Mixture sampling with a positive-reward sub-buffer is a recipe inherited from Hafner et al.'s Dreamer-4 (2025) "Training Agents Inside of Scalable World Models" — see [Phase 6](../references/Dreamer/dreamer_lit_review.md#paper-6) of the lit review.
- **Code spec.** Mode flag `agent.sampling_mode ∈ {"uniform", "mixture"}` (`dreamer_v3.yaml:18`); canonical = `"mixture"`. Slot layout (`dreamer_v3.yaml:19–22`): `mixture_positive_slots: 5`, `mixture_recent_slots: 5`, `mixture_recent_window: 10000`, `positive_buffer_capacity: 100000`. Implicit `uniform_slots = batch_size - pos_slots - recent_slots = 16 - 5 - 5 = 6`. Positive-buffer write trigger (`train.py:1397–1463`, both GPU and CPU paths): after every `collect_sequence`, the rolled-out transitions are reshaped into env-major blocks of `seq_len=128`; for each block, if `np.any(blk_rewards > 0.0)` the **entire block** is also `add_batch`'d into `positive_buffer`. Threshold is hard-coded `> 0.0` on raw reward.
  Sampling logic — fully JIT-traced inside `_scan_train_gpu` (`trainer.py:677–783`):
  - **Pool 1 (positive)** (`trainer.py:706–718`): block-uniform from positive buffer.
  - **Pool 2 (recent)** (`trainer.py:720–738`): "recent" = last `recent_window // seq_len = 10000 // 128 = 78` blocks before the current main-buffer write head; falls back to uniform if the buffer hasn't accumulated that many blocks.
  - **Pool 3 (uniform)** (`trainer.py:740–749`): block-uniform across the whole main buffer.
  - **Empty-positive fallback** (`trainer.py:751–762`): if positive buffer is empty, the positive-pool slots are silently replaced with uniform samples from the main buffer (separate fallback PRNG).
  Concatenation order (`trainer.py:766–772`): `[pos | recent | uniform]` along axis 0 of the batch — first `pos_slots` rows are positive-biased, next `recent_slots` are recent-biased, rest uniform.
  CPU mirror `_sample_mixture_cpu` (`trainer.py:857–910`) does the same in numpy + a Python loop.
- **Deviation flag.** **`EXTENSION (not in paper)`**. This is the single most consequential structural deviation in our codebase. Backported from Dreamer-4 (Hafner 2025); justified by the project's sparse-positive-reward setting (homeostasis bonus is small and dense; episode-positive events are rare). Already audited as a top-3 likelihood candidate in the related-failure-modes critique ([dreamer_conventional_failure_modes_for_our_setup.md](../critiques/dreamer_conventional_failure_modes_for_our_setup.md) §2.7 onward). Top item in §6 below.
- **Deviation flag — CPU-path uses non-mandatory `config.get(...)`.** `MINOR DEVIATION (suspected unjustified)`. The GPU path uses `config.get_mandatory(...)` (`trainer.py:797–800`); the CPU path uses `config.get(...)` with a fallback (`trainer.py:860–863`). Project rule says no fallback defaults; silent fallback to `'uniform'` is technically possible on CPU path. (Inventory red-flag #21.)

#### §3.6.3 Batch materialisation at training time

- **Per-batch shape**: `(batch_size=16, sequence_length=128, *)`.
- **`is_first` propagation** (`trainer.py:642–650, 597`): in `collect_sequence`, after every step `next_d_state['is_first'] = done[..., None].astype(jnp.float32)` — set on the *next* dreamer state. The transition recorded carries the *previous-step's* `is_first` value: `'is_first': d_state.get('is_first', jnp.zeros((B, 1)))`. First env step: `dreamer_state['is_first'] = jnp.ones((B, 1))` (`trainer.py:597`). On stage transitions: dreamer state (and `is_first`) reset to all-ones (`train.py:1146–1149`).
- **Action storage**: actions are stored as **one-hot** vectors, not int indices (`trainer.py:647`):
  ```python
  'action': jax.nn.one_hot(action_idx, self.agent.ac.actor.net.layers[-1].out_features),
  ```
- **Reshape from collect to buffer**: `transitions['obs']` arrives as `(T, B, obs_dim)`; the caller (`train.py:1386–1394`) transposes to `(B, T, obs_dim)` then reshapes to `(B*T, obs_dim)`, preserving env-major adjacency. Since `collect_interval == seq_len = 128`, blocks line up cleanly with env-major writes.

### §3.7 Numerical-stability tricks

#### §3.7.1 `symlog` / `symexp`

- **Paper spec.** `symlog(x) = sign(x) * ln(|x| + 1)`, `symexp(x) = sign(x) * (exp(|x|) - 1)`. Applied to encoder inputs, decoder targets, reward targets, value targets.
- **Code spec.** `util.py:6–17`:
  ```python
  def symlog(x):  return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)
  def symexp(x):  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)
  ```
  Application sites: `obs = symlog(batch['obs'])` at the top of `train_step` (`trainer.py:122`) — encoder consumes symlog'd obs; `obs_symlog = symlog(obs)` inside `get_action` (`trainer.py:530`); decoder output is in symlog space; `to_twohot(reward)` internally applies `symlog(reward)` (`util.py:24`); `from_twohot(...)` returns `symexp(sym_val)` (`util.py:76`); reward consumption in imagination `from_twohot(reward_head(...))` returns raw reward (`trainer.py:362, 385`); value consumption in imagination `from_twohot(target_critic(...))` returns raw value (`trainer.py:367, 388, 411`); critic baseline in actor advantage `baseline = from_twohot(v_pred_logits)` is raw space (`trainer.py:433`); critic target `to_twohot(stop_gradient(lambda_returns))` is re-encoded into symlog/two-hot (`trainer.py:428`).
  Rewards and values traverse the symlog↔raw boundary multiple times per step. The actor advantage `(norm_returns - norm_baseline)` is computed in **raw-then-percentile-normalised** space, not symlog.
- **Deviation flag.** `MATCHES PAPER`.

#### §3.7.2 Two-hot encoding/decoding

- **Paper spec.** 255 bins linearly spaced **in symlog space** over `linspace(-20, +20, 255)`, mapping under `symexp` to **raw-space bin centres spanning approximately ±4.85·10^8** (verified against Hafner's published `embodied/jax/heads.py:87–97` which constructs `half = symexp(linspace(-20, 0, ...))` then mirrors). Forward: place mass on the two adjacent bins in proportion to closeness. Inverse: read out `symexp(p^T B)` where `B` is the bucket-centre vector (in symlog space) and `p` is the softmax of the head's logits.
- **Code spec.** `util.py:19–58` (forward), `util.py:60–76` (inverse):
  ```python
  # to_twohot
  x = symlog(x)
  bottom = symlog(min_v); top = symlog(max_v)        # ≈ ±3.04
  x = jnp.clip(x, bottom, top)
  rel = (x - bottom) / (top - bottom) * (num_buckets - 1)
  floor = jnp.floor(rel).astype(jnp.int32)
  ceil  = jnp.ceil(rel).astype(jnp.int32)
  prob_ceil = rel - floor
  prob_floor = 1.0 - prob_ceil
  target = onehot(floor)*prob_floor + onehot(ceil)*prob_ceil

  # from_twohot
  probs = jax.nn.softmax(logits, axis=-1)
  bucket_vals = jnp.linspace(symlog(min_v), symlog(max_v), num_buckets)
  sym_val = jnp.sum(probs * bucket_vals, axis=-1)
  return symexp(sym_val)
  ```
  Bins are linearly spaced **in symlog space** per the comment at `util.py:25–29`, but the `bottom = symlog(min_v=-20)` / `top = symlog(max_v=+20)` instantiation puts the symlog-space bins in `linspace(symlog(-20), symlog(+20)) = linspace(-3.045, +3.045)` — i.e. the **raw-space support of our bins is `symexp(±3.045) ≈ ±20`**, NOT the paper's `±symexp(20) ≈ ±4.85·10^8`. The paper's convention is `linspace(-20, +20)` directly **in symlog space** (raw-space `±4.85·10^8`); our convention takes the symlog of `±20` first and uses *that* as the symlog-space range (raw-space `±20`). Two distinct bin layouts use the same constants: rewards (`reward_head` 255-dim) and values (`critic` 255-dim, `target_critic` 255-dim).
- **Deviation flag — bin range is `±20` in raw space, not `±4.85·10^8`.** **`MAJOR DEVIATION (suspected unjustified)`**. Eight-orders-of-magnitude mismatch in raw-return support. For our grid-world reward magnitudes (homeostasis bonus `|r| ≲ 1`, episodic injury `|r| ~ few units`), the practical effect is small because clipping at `±20` raw never triggers — but any future scaling exercise hitting returns above `±20` would silently saturate. Critically relevant to the project's reward-head failure mode (forward-link `.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md`). Cross-validated by `math-reviewer` against `embodied/jax/heads.py:87–97`. New top-tier item in §6.
- **Deviation flag — bin layout knobs hard-coded.** `MINOR DEVIATION (suspected unjustified)` that bins are hard-coded in the function defaults rather than config-knob'd (no way to override `min_v`, `max_v`, or `num_buckets` from YAML).

#### §3.7.3 `unimix` constant on categorical logits

- **Paper spec.** Posterior, prior, and actor categoricals are mixed with 1% uniform: `0.99 * NN_softmax + 0.01 * uniform`.
- **Code spec.** `util.py:78–110`:
  ```python
  class OneHotDist:
      def __init__(self, logits, unimix=0.01):
          probs = jax.nn.softmax(logits, axis=-1)
          if unimix > 0:
              probs = (1.0 - unimix) * probs + unimix / self.num_classes
          self.probs = probs
  ```
  Applied to RSSM posterior + prior categoricals (`nnx.py:101, 134`), actor logits at imagination time (`trainer.py:348, 380`), actor logits at inference time (`trainer.py:571`). Constant `0.01` is the `OneHotDist` constructor default. The YAML key `unimix: 0.01` (`dreamer_v3.yaml:43`) is **not threaded through** — the constructor is called everywhere with the default, so the YAML key is non-functional.
- **Deviation flag — YAML key `agent.unimix` is dead.** `MINOR DEVIATION (suspected unjustified)`. Today's behaviour matches the paper because the function default happens to be 0.01; changing the YAML value would do nothing. (Inventory red-flag #9.)

#### §3.7.4 `Moments` percentile return scaling — the entropy-coefficient invariance mechanism

This is **not just a numerical-stability trick**: percentile return scaling is the *return-scaling* mechanism that gives DreamerV3 its single-entropy-coefficient property. With `η = 3e-4` fixed across all 150+ benchmark tasks, the entropy bonus has comparable strength relative to the policy gradient regardless of whether the domain has dense rewards (returns of order 100s) or sparse rewards (returns of order 1). Without this mechanism, V3 would need per-domain entropy tuning, which would defeat the entire single-config thesis from §1. Filed under "numerical-stability tricks" only because that is where the *implementation* lives.

- **Paper spec.** `S = EMA(Per(R^lambda, 95) - Per(R^lambda, 5), rho=0.99)`; advantage is divided by `max(1, S)`. The paper clamps the *invscale* at `1/max_ = 1.0`, never below.
- **Code spec.** `util.py:113–159`. Constructor params hard-coded at instantiation site (`trainer.py:90`):
  ```python
  self.moments = Moments(decay=0.99, max_=1.0, percentile_low=0.05, percentile_high=0.95)
  ```
  EMA state `self.low` and `self.high` initialised to `0.0` and `1.0`, stored as `nnx.Variable` so JIT mutates correctly. Update + normalise (`util.py:136–159`):
  ```python
  low_p  = jnp.percentile(x_flat, percentile_low * 100)
  high_p = jnp.percentile(x_flat, percentile_high * 100)
  self.low.value  = decay * self.low.value  + (1 - decay) * low_p
  self.high.value = decay * self.high.value + (1 - decay) * high_p
  invscale = jnp.maximum(1.0 / max_, self.high.value - self.low.value)
  ```
  `invscale = max(1.0, high - low)`, never below 1.0 — matches the paper's `max(1, S)` clamp. Snapshotted before the gradient call (`trainer.py:336–340`) to avoid tracing through `self.moments` inside `nnx.grad` (OOM concern called out in the comment). `self.moments.update(lambda_returns)` happens after `nnx.grad` (`trainer.py:495`), so the moments used inside the gradient are stale by one step. This **read-then-update** ordering matches Hafner's published implementation, which also reads `self.low.value` / `self.high.value` for normalisation before calling `update` — so framing this as "stale by one" is technically precise but the same convention as the official source.
- **Deviation flag.** `MATCHES PAPER`. `MINOR DEVIATION (suspected unjustified)` that the four constants (`decay`, `max_`, `percentile_low`, `percentile_high`) are hard-coded with no YAML knob. (Inventory red-flag #7.)

#### §3.7.5 Free-nats clipping

- **Paper spec.** `max(1, KL_dyn)` and `max(1, KL_rep)` per state. Nature Ext. Data Table 5 lists "Free nats: 1" as a single floor without explicit axis specification; Hafner's published `dreamerv3/rssm.py` disambiguates by applying `Agg(out, 1, jnp.sum)` over the stoch-group axis BEFORE the `jnp.maximum(dyn, free_nats)` clip — i.e. clips per `(B, T)` state.
- **Code spec.** `trainer.py:17, 246–247`. Constant `FREE_NATS = 1.0` applied at `trainer.py:246–247` after summing over **both** the discrete-class axis (`kl_div_categ` interior `axis=-1` sum at L230) AND the stoch-group axis (`jnp.sum(dyn_kl, axis=-1)` at L243-244), then averaged across batch+time (`trainer.py:249`). Layout consequence: `dyn_kl/rep_kl` going into `max(., 1.0)` is shape `(B, T)` after summing over both class and group axes. Lower-bound on `loss_kl` is `0.5 · 1.0 + 0.1 · 1.0 = 0.6` nats averaged.
- **Deviation flag.** `MATCHES PAPER` — Hafner's `Agg(out, 1, jnp.sum)` wrapper around `OneHot.kl(...)` produces the same per-state axis behaviour as our code.

#### §3.7.6 Other numerics

- `eps=1e-10` inside `random.categorical(jnp.log(probs + 1e-10))` to avoid log(0) (`util.py:96, 98`).
- Per-mask reward MAE divisor `+1e-8` (`trainer.py:261–262`).
- Adam `eps` split: `1e-8` for WM, `1e-5` for actor and critic (`trainer.py:96, 104, 112`). **`MATCHES PAPER (Hafner 2023 preprint Table W.1)`** — the preprint table specifies WM Adam ε = 10⁻⁸ and AC Adam ε = 10⁻⁵, exactly the asymmetric split our code implements. (An earlier draft mis-classified this as a deviation against a non-existent "1e-5 uniform" Nature spec; Nature actually replaces Adam with LaProp ε = 10⁻²⁰, a separate deviation handled in §3.8.1.)
- Continue-accuracy threshold `> 0.5` on sigmoid (`trainer.py:270`).

### §3.8 Optimiser + schedule

#### §3.8.1 Per-group optimiser setup

- **Paper spec — preprint (Hafner 2023 Table W.1).** Adam, no weight decay, no LR schedule. **Asymmetric epsilon: WM `eps = 1e-8`, actor/critic `eps = 1e-5`.** Split LRs: WM `1e-4`, actor `3e-5`, critic `3e-5`. Grad clip: global-norm `1000` (WM) / `100` (AC).
- **Paper spec — Nature (Hafner 2025 Extended Data Table 5).** Optimiser is replaced entirely: **LaProp with `ε = 1e-20`** ("LaProp normalizes gradients by RMSProp and then smoothes them by momentum, instead of computing both momentum and normalizer on raw gradients as Adam does"), single uniform LR `4e-5`, **AGC(0.3) per-tensor adaptive grad clip** (clips per-tensor gradients exceeding 30% of the L2 norm of the parameter matrix they correspond to) instead of global-norm.
- **Code spec.** `trainer.py:92–115`.

  | Group | Kind | LR | eps | Grad clip | Weight decay |
  |---|---|---|---|---|---|
  | World model (`self.agent.wm`) | `optax.adam` | `model_lr = 1e-4` | `1e-8` | `clip_by_global_norm(1000.0)` | none |
  | Actor (`self.agent.ac.actor`) | `optax.adam` | `actor_lr = 3e-5` | `1e-5` | `clip_by_global_norm(100.0)` | none |
  | Critic (`self.agent.ac.critic`) | `optax.adam` | `value_lr = 3e-5` | `1e-5` | `clip_by_global_norm(100.0)` | none |

  ```python
  self.model_opt = nnx.Optimizer(
      self.agent.wm,
      optax.chain(
          optax.clip_by_global_norm(1000.0),
          optax.adam(config.get_mandatory('agent.model_lr', float), eps=1e-8)
      ),
      wrt=nnx.Param
  )
  ```
  Target critic is **NOT wrapped in an optimiser** — only the EMA copy at `trainer.py:501–505` updates it. No LR schedule (warmup or decay) is wired anywhere. No weight decay (no `optax.adamw`).
- **Deviation flag — Adam ε `1e-8` (WM) / `1e-5` (AC).** **`MATCHES PAPER (preprint Table W.1)`**. The asymmetric split exactly matches Hafner 2023.
- **Deviation flag — WM grad clip `1000.0`.** `MATCHES PAPER` against the preprint; `MAJOR DEVIATION (deliberate)` against Nature, which uses AGC(0.3) per-tensor (a *different mechanism*, not just a scaled value). (Inventory red-flag #13.)
- **Deviation flag — Optimiser is Adam, not LaProp.** **`MAJOR DEVIATION (Nature-only — preprint matches)`**. Nature replaces Adam entirely with LaProp + AGC(0.3) + single LR `4e-5`. Our code is the preprint recipe; ranked lower than the load-bearing items in §6 because LaProp is part of the Nature-only changes that we do not target.

#### §3.8.2 `train_steps` per iteration — `Ratio(replay_ratio)` gating

- **Paper spec.** Replay ratio = gradient steps per env step. Hafner-2023 default for Atari 200M and DMC: `1/16 ≈ 0.0625`. Scaling experiments sweep up to 64.
- **Code spec.** `dreamer_v3_util.py:162–192` defines `Ratio`:
  ```python
  def __call__(self, step: int) -> int:
      if self._ratio == 0: return 0
      if self._prev is None:
          self._prev = step
          repeats = int(step * self._ratio)
          if self._pretrain_steps > 0:
              repeats = int(self._pretrain_steps * self._ratio)
          return repeats
      repeats = int((step - self._prev) * self._ratio)
      self._prev += repeats / self._ratio
      return repeats
  ```
  Caller (`train.py:789, 1620`):
  ```python
  ratio_scaled_updates = Ratio(config.get_mandatory('agent.replay_ratio'))
  ...
  train_steps = ratio_scaled_updates(global_step // num_steps)
  ```
  The argument is `global_step // num_steps` where `num_steps = collect_interval = 128`, so `Ratio` is fed *iteration count*, not raw env-step count — the effective rate is `replay_ratio` gradient steps per **iteration**, not per env step. Cold-start gate (`train.py:1615`): training is blocked until the buffer holds at least `max(batch_size * 2, sequence_length) = max(32, 128) = 128` transitions. Effective replay ratio metric logged to WandB (`train.py:1640`): `Params/effective_replay_ratio = cumulative_gradient_steps / max(1, global_step)`.
- **Deviation flag — `replay_ratio = 0.5`.** `MAJOR DEVIATION (deliberate)`. 8× the paper-canonical Atari/DMC default of 0.0625. Top-3 candidate failure mode in the related critique ([§2.2](../critiques/dreamer_conventional_failure_modes_for_our_setup.md)). The variant `configs/models/dreamer_v3_rr06.yaml` differs by exactly this knob (set to 0.0625). Single biggest gradient-flow knob in the codebase.
- **Deviation flag — `agent.train_steps: 64` YAML key never read.** `MINOR DEVIATION (suspected unjustified)`. Gating is purely via `Ratio(replay_ratio)`; the `train_steps` key is dead. (Inventory red-flag #10.)

#### §3.8.3 `train_multiple_gpu` JIT path

- `trainer.py:677–827`. The full inner loop — sample `num_steps` batches, run `train_step` on each, accumulate metrics — is fused inside `nnx.jit` with `lax.scan`. Decorator: `@nnx.jit(static_argnums=(1, 2, 7, 8, 10, 11, 12, 13))`. Static args (positions 1, 2, 7, 8, 10, 11, 12, 13): **`graphdef, num_steps, b_cap, b_seq_len, pos_cap, pos_slots, recent_slots, recent_window`**. Traced (per-call): `rng, main_arrays, pos_arrays, b_size, pos_size, buf_idx`. (`b_size`, `pos_size`, `buf_idx` cannot be static because they change every iteration as the buffers fill — making them static would force constant retracing.) The trainer state is split via `nnx.split` and re-merged inside the scan (`trainer.py:691–779`).

#### §3.8.4 `train_multiple_cpu` fallback

- `trainer.py:829–855`. CPU buffer path: pre-samples `num_batches` batches with `_sample_mixture_cpu` or `buffer.sample_multiple`, transfers once to device as a stacked dict `(num_batches, batch, seq, dim)`, then runs the JIT scan.

### §3.9 Dead code, unused config, and additional code surfaces (cleanup opportunities)

These are code surfaces present in the active modules that are not part of the active training path. Each is a one-line entry with a file:line cite — the user already knows about most from the original inventory's red-flag list; this is the durable home for them.

- **`Moments.normalize()` (`util.py:154–159`)** — never called anywhere in `src/`, `train.py`, or `scripts/`. Actor loss does the normalisation inline at `trainer.py:418, 434`. (Code-reviewer R1.)
- **`OneHotDist.mode()` (`util.py:108–110`)** — never called in active paths. Legitimate API method an offline diagnostic might use. (Code-reviewer R2.)
- **`DreamerV3Agent.__call__` (`nnx.py:614–668`)** — 55-LOC inference helper, never called. Trainer uses `DreamerTrainer.get_action` (`trainer.py:509-582`) instead. (Code-reviewer R3.)
- **`DreamerV3Agent.initial_state` (`nnx.py:602–612`)** — never called; trainer constructs initial state manually at `trainer.py:591-601` and `train.py:821-829, 1146-1151`. (Code-reviewer R4.)
- **`ReplayBuffer.sample_multiple` GPU branch (`trainer.py:1008–1017`)** — raises `NotImplementedError` for GPU buffer device; only CPU-path. Active GPU path uses `_scan_train_gpu`'s in-JIT mixture sampling. (Code-reviewer R5.)
- **`jax.named_scope(...)` instrumentation** — 13 named scopes scattered through `trainer.py` (`wm_encoder`, `wm_rssm_scan`, `wm_losses`, `dreamer_optim` ×2, `ac_imagine_scan`, `ac_losses`, `dreamer_sense`, `dreamer_act`, `dreamer_env_step`, `dreamer_env_reset`, `replay_mixture_sample`, `replay_concat`) plus `train.py:1385, 1397, 1614` — JAX trace annotations for chrome:// profiling. (Code-reviewer R6.)
- **Six DreamerV3 config variants exist** (`configs/models/dreamer_v3{,_rr06,_curriculum,_curriculum_probe,_probe,_probe_cont10}.yaml` + `neuromodulated_dreamer_v3.yaml`); this doc cites only the canonical and `_rr06.yaml`. (Code-reviewer R7.)
- **`is_first.ndim==3` shape-polymorphism in buffer write** (`train.py:1390-1394` GPU path; `train.py:1434-1438` CPU path) — handles both `(T, B, 1)` and `(T, B)` shapes; trainer guarantees `(B, 1)` per-step so the canonical path is `ndim==3`. (Code-reviewer R8.)
- **`FiLMNoNorm` modulation type explicitly rejected** (`nnx.py:495-499`) — `if modulation_type == "FiLMNoNorm": raise ValueError(...)`. The doc lists allowed types as `null/"FiLM"/"PreActivation"/"Multiplicative"`; this is an explicit-rejection branch not flagged in §4.7. (Code-reviewer R9.)

These are documented as cleanup opportunities; resolving them is a separate plan if/when the user wants it.

---

## §4 Diagnostics + probes

### §4.1 World-model side metrics (`trainer.py:272–285`)

| Key | What it measures |
|---|---|
| `loss_model` | total WM loss (recon + rew + cont*coef + KL) |
| `loss_recon` | obs MSE in symlog space |
| `loss_rew` | reward two-hot CE |
| `loss_cont` | continue-head BCE (raw mean) |
| `loss_dyn_kl` | mean of clipped dyn_kl |
| `loss_rep_kl` | mean of clipped rep_kl |
| `loss_kl` | `0.5*dyn_kl_mean + 0.1*rep_kl_mean` |
| `model_reward_mae` | `\|from_twohot(rew_pred) - reward\|.mean()` |
| `model_reward_mae_pos` | masked-mean MAE where `reward > 0.01` |
| `model_reward_mae_neg` | masked-mean MAE where `reward < -0.01` |
| `model_latent_entropy` | per-step posterior categorical entropy (mean) |
| `model_cont_acc` | accuracy of `sigmoid(cont_pred) > 0.5` vs `1 - terminal` |

### §4.2 Behaviour-side metrics (`trainer.py:463–474`)

| Key | What it measures |
|---|---|
| `loss_critic` | discount-weighted critic two-hot CE on raw lambda returns |
| `loss_actor` | discount-weighted REINFORCE-w-baseline + entropy |
| `loss_actor_policy` | policy-gradient sub-term: `mean(-log_probs * adv * dw)` |
| `loss_actor_entropy` | entropy sub-term: `mean(-entropy_scale * H * dw)` |
| `mean_return` | mean of raw lambda_returns |
| `mean_norm_return` | mean of percentile-normalised returns |
| `mean_value` | mean of from_twohot(critic logits) — baseline |
| `mean_advantage` | mean of stop-gradient(norm_returns - norm_baseline) |
| `mean_entropy` | mean policy entropy |
| `value_mae` | `\|baseline - lambda_returns\|.mean()` |

### §4.3 `imagined_rollout_probe` flag

- **Config**: `agent.imagined_rollout_probe: false` canonical (`dreamer_v3.yaml:42`). Comment: "False = bit-identical to pre-probe runs."
- **Read**: `trainer.py:130` (`IMG_PROBE = self.config.get_mandatory('agent.imagined_rollout_probe', bool)`).
- **Six imagined-side metrics added when true** (`trainer.py:447–484`):
  ```python
  term_mask = (conts < 0.5).astype(jnp.float32)            # (HORIZON=15, IMAG_BATCH)
  any_term = jnp.any(term_mask > 0, axis=0)
  first_term_step = jnp.argmax(term_mask, axis=0)
  first_term_step = jnp.where(any_term, first_term_step, HORIZON)
  imag_term_frac_h8  = jnp.mean(jnp.any(term_mask[:8] > 0, axis=0))
  imag_term_frac_h15 = jnp.mean(any_term)
  imag_first_term_mean = jnp.mean(first_term_step.astype(jnp.float32))
  imag_first_term_p10  = jnp.percentile(first_term_step_f, 10.0)
  imag_first_term_p50  = jnp.percentile(first_term_step_f, 50.0)
  imag_first_term_p90  = jnp.percentile(first_term_step_f, 90.0)
  ```
  Logged keys: `imagined_termination_fraction_h8`, `imagined_termination_fraction_h15`, `imagined_first_term_step_mean`, `imagined_term_step_p10`, `imagined_term_step_p50`, `imagined_term_step_p90`.
- **One real-side companion** added in the WM block (`trainer.py:287–294`): `imagined_real_term_step_mean` — same first-termination-step statistic but for the real replay batch (sentinel = `T = sequence_length` if no termination).
- **Total = 7 metrics** when probe is on.
- **Deviation flag.** `EXTENSION (not in paper)`. Project-specific instrumentation for diagnosing imagination-vs-reality termination calibration; see [§2.4 of the related critique](../critiques/dreamer_conventional_failure_modes_for_our_setup.md).

### §4.4 Modulation block (gated by `modulation_enabled`)

When modulation is enabled (canonical config has `modulation.type: null`, so by default this block is dormant), the WM loss block adds `mod_z_unimodal_mean/std`, `mod_z_multimodal_mean/std`, `mod_memory_mean/std`, `mod_z_reward_mean/std` (`trainer.py:296–313`); for `PreActivation` / `FiLM` types, also `mod_beta_unimodal_mean`, `mod_beta_multimodal_mean` (`trainer.py:314–318`).

### §4.5 Buffer telemetry (`train.py:1640–1649`)

- `Params/effective_replay_ratio` (always logged): `cumulative_gradient_steps / max(1, global_step)`.
- `Params/positive_buffer_blocks`
- `Params/positive_buffer_utilization`
- `Params/main_buffer_blocks`

### §4.6 Loss/metric routing into WandB (`train.py:1650–1664`)

- `Behavior/...` for `loss_actor*`, `loss_critic`, `mean_*`, `entropy*`.
- `WorldModel/...` for `loss_model`, `loss_recon`, `loss_kl`, `loss_rew`, `loss_cont`, `loss_dyn*`, `loss_rep*`, `model_*`, `imagined_*`.
- `Modulator/...` for `mod_*`.

### §4.7 Other config-driven runtime flags

| Key | Effect |
|---|---|
| `agent.cont_loss_weight` | multiplier on `loss_cont` only (`trainer.py:251–252`). 1.0 = pre-knob behavior. |
| `agent.imagined_rollout_probe` | enables/disables 7 termination-probe metrics. |
| `agent.entropy_scale` | multiplier on the entropy term in actor loss (`trainer.py:441–444`). 3e-4 canonical. |
| `agent.use_layer_norm` | toggles the `mod_*_ln` LayerNorm layer between encoder body and final SiLU (`nnx.py:515–521`). |
| `agent.encoding_mode` | `"hierarchical"` vs `"flat"` (encoder/decoder branching). |
| `agent.sampling_mode` | `"uniform"` vs `"mixture"` (whether positive buffer is built and used). |
| `agent.modulation.type` | `null` / `"FiLM"` / `"PreActivation"` / `"Multiplicative"` — gates modulator construction and gain-bias injection. |

---

## §5 Configs + canonical defaults

### §5.1 `configs/models/dreamer_v3.yaml` — full key-value table

| YAML key | Paper default (Hafner 2023 / 2025) | Current value | Deviation flag |
|---|---|---|---|
| `agent.algorithm` | "DreamerV3" | "DreamerV3" | matches paper |
| `agent.batch_size` | 16 | 16 | matches paper |
| `agent.sequence_length` | 64 | 128 | **MAJOR DEVIATION (deliberate)** — 2× canonical |
| `agent.replay_ratio` | 1/16 ≈ 0.0625 | 0.5 | **MAJOR DEVIATION (deliberate)** — 8× canonical |
| `agent.collect_interval` | (n/a — not a paper knob; equivalent to env-step batching schedule) | 128 | EXTENSION (not in paper) — JAX-optimised batching |
| `agent.train_steps` | (gating via Ratio only in Hafner) | 64 | **dead key** — never read; MINOR DEVIATION |
| `agent.model_lr` | 1e-4 | 1e-4 | matches paper |
| `agent.actor_lr` | 3e-5 | 3e-5 | matches paper |
| `agent.value_lr` | 3e-5 | 3e-5 | matches paper |
| `agent.buffer_device` | (impl detail; paper uses CPU/GPU as available) | "gpu" | EXTENSION — engineering choice |
| `agent.buffer_capacity` | 10^6 (preprint Table W.1 implicit) / **5×10^6 (Nature Ext. Data Table 5)** | 1_000_000 | matches preprint; **5× smaller than Nature** |
| `agent.sampling_mode` | uniform sub-sequence (only) | "mixture" | **EXTENSION (not in paper)** — Dreamer-4 backport |
| `agent.mixture_positive_slots` | n/a | 5 | **EXTENSION (not in paper)** |
| `agent.mixture_recent_slots` | n/a | 5 | **EXTENSION (not in paper)** |
| `agent.mixture_recent_window` | n/a | 10000 | **EXTENSION (not in paper)** |
| `agent.positive_buffer_capacity` | n/a | 100_000 | **EXTENSION (not in paper)** |
| `agent.encoder_dim` | 512 (S profile) | 128 | **MAJOR DEVIATION (deliberate)** — small |
| `agent.encoder_fc_layers` | per-profile widths (S = 512×N) | [128, 128] | **MAJOR DEVIATION (deliberate)** — small |
| `agent.rssm_deter_dim` | 512 (S) / 1024 (M) / 2048 (L) / 4096 (XL) | 512 | matches paper (S profile) |
| `agent.rssm_stoch_dim` | 32 | 32 | matches paper |
| `agent.rssm_classes` | 32 | 32 | matches paper |
| `agent.decoder_fc_layers` | per-profile (S = 512×N) | [128, 128] | **MAJOR DEVIATION (deliberate)** — small |
| `agent.reward_fc_layers` | per-profile | [128, 128] | **MAJOR DEVIATION (deliberate)** — small |
| `agent.continue_fc_layers` | per-profile | [128, 128] | **MAJOR DEVIATION (deliberate)** — small |
| `agent.actor_fc_layers` | per-profile | [128, 128] | **MAJOR DEVIATION (deliberate)** — small |
| `agent.critic_fc_layers` | per-profile | [128, 128] | **MAJOR DEVIATION (deliberate)** — small |
| `agent.entropy_scale` | 3e-4 | 3e-4 | matches paper |
| `agent.cont_loss_weight` | 1.0 (implicit) | 1.0 | matches paper (knob is EXTENSION) |
| `agent.imagined_rollout_probe` | n/a | false | **EXTENSION (not in paper)** — diagnostic |
| `agent.unimix` | 0.01 | 0.01 (but **never read** — see §3.7.3) | **dead key** — MINOR DEVIATION |
| `agent.use_layer_norm` | implicit true | true | matches paper (knob is EXTENSION) |
| `agent.encoding_mode` | n/a (paper has CNN or MLP per domain) | "hierarchical" | **EXTENSION (not in paper)** |
| `agent.hierarchical_params.default_mlp` | n/a | [128, 128] | **EXTENSION (not in paper)** |
| `agent.hierarchical_params.unimodal_overrides.visual` | n/a | [128, 128] | **dead key** — never read; MINOR DEVIATION |
| `agent.hierarchical_params.unimodal_overrides.olfaction` | n/a | [128, 128] | **dead key** — never read; MINOR DEVIATION |
| `agent.hierarchical_params.multimodal_hub` | n/a | [128, 128] | **EXTENSION (not in paper)** |
| `agent.modulation.type` | n/a | null | **EXTENSION (not in paper)** — disabled by default |

### §5.2 Hard-coded constants (no YAML key)

| Constant | Value | File:line | Paper default | Flag |
|---|---|---|---|---|
| `FREE_NATS` | 1.0 | `trainer.py:17` | 1 nat | matches paper |
| `KL_SCALE` | 1.0 | `trainer.py:18` (declared but never used) | n/a | **dead constant** — MINOR DEVIATION |
| `DYN_SCALE` | 0.5 | `trainer.py:19` | 0.5 (preprint) / 1.0 (Nature) | matches preprint; **MAJOR DEVIATION** vs. Nature |
| `REP_SCALE` | 0.1 | `trainer.py:20` | 0.1 | matches paper |
| `HORIZON` | 15 | `trainer.py:21` | **15 (both versions)** | matches paper |
| `GAMMA` | 0.997 | `trainer.py:22` | 0.997 | matches paper |
| `LAMBDA` | 0.95 | `trainer.py:23` | 0.95 | matches paper |
| Two-hot `min_v` | -20.0 | `util.py:19, 60` | linspace endpoint **-20 in symlog space** (raw `-symexp(20) ≈ -4.85·10^8`) | **MAJOR DEVIATION** — our raw range is `-20` not `-4.85·10^8` (see §3.7.2) |
| Two-hot `max_v` | +20.0 | `util.py:19, 60` | linspace endpoint **+20 in symlog space** (raw `+symexp(20) ≈ +4.85·10^8`) | **MAJOR DEVIATION** — our raw range is `+20` not `+4.85·10^8` (see §3.7.2) |
| Two-hot `num_buckets` | 255 | `util.py:19, 60`; mirrored in `nnx.py:530, 576` | 255 | matches paper (hard-coded; MINOR DEVIATION on knob-ability) |
| `Moments.decay` | 0.99 | `trainer.py:90` | 0.99 (Nature explicit) | matches paper |
| `Moments.max_` | 1.0 | `trainer.py:90` | implicit 1 (the `max(1, S)` clamp) | matches paper |
| `Moments.percentile_low` | 0.05 | `trainer.py:90` | 5th percentile | matches paper |
| `Moments.percentile_high` | 0.95 | `trainer.py:90` | 95th percentile | matches paper |
| `OneHotDist.unimix` | 0.01 (constructor default) | `util.py:83` | 0.01 | matches paper (but YAML knob ignored) |
| Target-critic EMA | `0.98 / 0.02` | `trainer.py:504` | 0.98 / 0.02 (Nature) | matches paper |
| WM grad-clip global norm | 1000.0 | `trainer.py:95` | **1000 (preprint Table W.1)** / Nature uses AGC(0.3), a different mechanism | matches preprint; deviates from Nature optimiser path |
| Actor grad-clip global norm | 100.0 | `trainer.py:103` | 100 (preprint AC) | matches preprint |
| Critic grad-clip global norm | 100.0 | `trainer.py:111` | 100 (preprint AC) | matches preprint |
| WM Adam eps | 1e-8 | `trainer.py:96` | **1e-8 (preprint Table W.1 WM)** | **MATCHES PAPER (preprint)** |
| Actor Adam eps | 1e-5 | `trainer.py:104` | 1e-5 (preprint Table W.1 AC) | matches preprint |
| Critic Adam eps | 1e-5 | `trainer.py:112` | 1e-5 (preprint Table W.1 AC) | matches preprint |
| `hafner_init` scale | 0.8796 | `util.py:195` | 0.8796 ("secret sauce" of original implementation) | matches paper |
| Positive-buffer reward threshold | `> 0.0` | `train.py:1411, 1454` | n/a | **EXTENSION (not in paper)** |

### §5.3 `configs/models/dreamer_v3_rr06.yaml` — single delta from canonical

| Key | Canonical | rr06 value |
|---|---|---|
| `agent.replay_ratio` | 0.5 | 0.0625 |

All other keys identical (visual diff confirms — header comment at `dreamer_v3_rr06.yaml:1–38` explicitly notes byte-identical except for replay_ratio). The rr06 variant restores the paper-canonical replay ratio.

---

## §6 Deviations from paper — executive summary

This section pulls every non-`MATCHES PAPER` flag from §3 and §5 into one ranked list, **highest-impact first**. For each item: paper citation, code citation, and a one-line "why this matters" pointer.

### Top-tier — large, structural, load-bearing

1. **Coupled WM training-distribution deviation (4 sub-items).** This is one coupled deviation in *what the world model sees per gradient step*; the four sub-items below interact multiplicatively, so the experiment design implication is "revert the cluster, not one knob at a time".
   - **1a. Mixture sampling + positive-reward buffer (`agent.sampling_mode: "mixture"`).** `EXTENSION (not in paper)`. Code: `trainer.py:706–772` (GPU mixture), `train.py:1397–1463` (positive-buffer write trigger). Paper baseline: uniform sub-sequence sampling only ([Phase 5a §2.5](../references/Dreamer/dreamer_lit_review.md#paper-5a)). DreamerV4-inspired backport (see [Phase 6](../references/Dreamer/dreamer_lit_review.md#paper-6)). Five of every 16 batch slots are drawn from a separate 100k-transition buffer of "any block where some reward > 0.0", five from the most-recent 10000 transitions, six uniform.
   - **1b. `agent.replay_ratio: 0.5`.** `MAJOR DEVIATION (deliberate)`. Code: `train.py:789, 1620; trainer.py Ratio gating`. Per Hafner 2023 Table A.1, the paper default is benchmark-dependent — Atari 200M = 64 (≈0.0156 grad/env normalised), DMC = 512 (≈0.0625 grad/env), BSuite/Atari 100K = 1024, Minecraft = 16. A factor-of-8 spread across benchmarks (DMC mid-band ≈ 0.0625; our 0.5 is therefore 8× the DMC mid-band, 32× the Atari 200M default). The project has been treating 0.0625 as canonical; the actual paper-spec spread is wider. The variant `dreamer_v3_rr06.yaml` exists specifically to restore this knob to the DMC mid-band.
   - **1c. `agent.sequence_length: 128`** (paper 64). `MAJOR DEVIATION (deliberate)`. Code: `dreamer_v3.yaml:4`; `trainer.py:917–1006`. 2× canonical.
   - **1d. Block-aligned (not uniform) sub-sequence sampling.** `MAJOR DEVIATION (deliberate)`. Code: `trainer.py:973–1006`. Env-major storage + block-aligned starts means the diversity of sampled batches is bounded by `buffer_capacity // sequence_length` block positions.
   - **Why the coupling matters**: a high replay ratio combined with a positive-biased mixture sampler means the WM is seeing the **same set of positive-reward blocks repeatedly** across many gradient steps, in a regime where the canonical V3 buffer would have shown those blocks once or twice. The sequence-length doubling and block-aligned grid further compound the effect. The project's conventional-fixes battery already showed that `replay_ratio = 0.5` fixed the NoPred collapse but did not recover predator-task survival — that negative result is more interpretable if you treat 1a+1b+1c+1d as a coupled triple rather than as four independent knobs.
2. **Two-hot bin range narrowed by 8 orders of magnitude.** **`MAJOR DEVIATION (suspected unjustified)`**. Code: `util.py:60–76` (`from_twohot`). Our `bucket_vals = jnp.linspace(symlog(-20), symlog(+20), 255)` creates symlog-space bins in `linspace(-3.045, +3.045)`, mapping under `symexp` to **raw-space support `±20`**. Paper convention (preprint Eq. 8 + Nature page 4 + Hafner published `embodied/jax/heads.py:87–97`) is `linspace(-20, +20)` *in symlog space*, mapping to raw-space support `±symexp(20) ≈ ±4.85·10^8`. **Eight orders of magnitude mismatch in raw-return support**. For our grid-world reward magnitudes (`|r| ≲ few units`), saturation at `±20` raw never triggers, so the practical effect is benign — but the deviation is real and any future scaling exercise hitting returns above `±20` would silently saturate. **Forward-link**: `.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md` — directly relevant to the reward-head localized-failure investigation.
3. **λ-return bootstrap uses `target_critic`, not online critic.** **`MAJOR DEVIATION (suspected unjustified — corresponds to the preprint's ablated `SlowTarget` variant)`**. Code: `trainer.py:367, 388, 411` — all three bootstrap-value sites for the λ-return computation use `self.target_critic`. Paper canonical: **both Hafner 2023 preprint and Hafner 2025 Nature compute λ-returns using the *fast/online* critic.** Preprint Appendix C item 6 (verbatim): "We compute λ-returns using the **fast critic** network and regularize the critic outputs towards those of its own weight EMA instead of computing returns using the slow critic. However, both approaches perform similarly in practice." Nature page 3 (verbatim): "regularizing the critic towards predicting the outputs of an exponentially moving average of its own parameters... but allows us to compute returns using **the current critic network**." Preprint Appendix D.2 explicitly ablates the `SlowTarget` variant — our exact recipe — and reports verdict "no benefit". **Severity**: research-level. *This deviation warrants a controlled `SlowTarget`-vs-`current-critic` comparison run* — handed off to `senior-developer` and `experiment-designer` as a follow-up plan. Critic-bootstrap dynamics are non-trivial in our recurrent-imagination setting; this may compound with item 2.
4. **MLP hidden 128 throughout (paper "S" profile uses 512+).** `MAJOR DEVIATION (deliberate)`. Code: `dreamer_v3.yaml:25–38`; commented-out XL block at `dreamer_v3.yaml:60–74` mentions 1024. **Why it matters**: encoder, decoder, all heads, actor, critic all use `[128, 128]`. The active config is much smaller than any Hafner profile. Hafner-2025 Fig. 6c–d show monotonic improvement with size up to 400M params; we are far below the smallest reported profile. Reasoned justification: the project has small grid-world observations and an explicitly-small-network design philosophy. Risk: under-capacity for the multimodal hierarchical encoder.
5. **`DYN_SCALE = 0.5` (Nature `beta_dyn = 1.0`).** `MAJOR DEVIATION (deliberate)`. Code: `trainer.py:19`. **Why it matters**: matches preprint, deviates from Nature. **Per the Hafner-2025 Nature ablation (Fig. 6), this is the second-most-impactful loss-weight change in the algorithm**; we have not adopted it. Empirical anchor lifted from §3.5.4.
6. **No replay-value loss (Nature `beta_repval = 0.3` term).** `MAJOR DEVIATION (deliberate)`. Code: `trainer.py:421–430` — critic trained only on imagined-rollout `lambda_returns`. Paper: see [Phase 5b §2.4](../references/Dreamer/dreamer_lit_review.md#paper-5b). The Nature replay-value term bootstraps off the imagined `R^lambda` at the start state — using the imagination rollout as on-policy value annotations for the replay trajectory. **Why it matters**: we are running preprint critic semantics, not Nature. Documented as a stabiliser for hard-prediction domains; whether our setup needs it is an open question.
7. **WM grad-clip global-norm `1000.0`** (preprint `1000`; Nature uses **AGC(0.3) per-tensor, a different mechanism**). `MAJOR DEVIATION (deliberate)` against Nature path; `MATCHES PAPER` against preprint. Code: `trainer.py:95`. **Why it matters**: matches preprint exactly; the Nature deviation is a *path* (clipping mechanism), not just a value.
8. **Nature optimiser swap (LaProp + AGC + uniform LR + 5× buffer).** **`MAJOR DEVIATION (Nature-only — preprint matches)`**. Code: `trainer.py:92–115`. Nature replaces Adam entirely with **LaProp (`ε = 1e-20`)**, switches to **single uniform LR `4e-5`** (vs. preprint split `1e-4` WM / `3e-5` AC), uses **AGC(0.3) per-tensor** instead of global-norm clip, and **5×10⁶ buffer capacity** instead of `10⁶`. We run the preprint Adam recipe entirely. Ranked here (lower than the load-bearing items above) because LaProp is part of the Nature-only changes and our codebase explicitly targets the preprint baseline; not load-bearing for the active diagnostic agenda.

### Mid-tier — minor deviations or extensions with limited scope

9. **`KL_SCALE = 1.0` declared but never used.** `MINOR DEVIATION (suspected unjustified)`. Code: `trainer.py:18`. **Why it matters**: dead constant; suggests a refactor stranded it. The actual KL combine is `0.5*dyn + 0.1*rep`, not `KL_SCALE * (0.5*dyn + 0.1*rep)`. Behaviour matches paper; cleanup opportunity.
10. **`agent.unimix` YAML key never read.** `MINOR DEVIATION (suspected unjustified)`. Code: `dreamer_v3.yaml:43`; `OneHotDist` constructor at `util.py:83`. **Why it matters**: behaviour matches paper because the constructor default (0.01) coincides with the YAML value (0.01); but changing the YAML value would do nothing. Latent-bug risk if the YAML is edited expecting it to take effect.
11. **`agent.train_steps: 64` YAML key never read.** `MINOR DEVIATION (suspected unjustified)`. Code: `dreamer_v3.yaml:7`. **Why it matters**: gating is purely `Ratio(replay_ratio)`; the `train_steps` key is dead. Cleanup opportunity.
12. **`agent.hierarchical_params.unimodal_overrides` (visual / olfaction) never read.** `MINOR DEVIATION (suspected unjustified)`. Code: `dreamer_v3.yaml:55–57`; `nnx.py:223` always uses `default_mlp`. **Why it matters**: dead config. The `unimodal_overrides` block is a phantom — sensor-specific MLP widths cannot actually be overridden today.
13. **CPU-path mixture sampling uses non-mandatory `config.get(...)`.** `MINOR DEVIATION (suspected unjustified)`. Code: `trainer.py:860–863` vs. `trainer.py:797–800`. **Why it matters**: GPU path uses `get_mandatory`, CPU path uses `get(..., default)`. Project rule is no fallback defaults. Silent fallback to `'uniform'` is technically possible on CPU path if the YAML is missing keys.
14. **`src/models/dreamer_v3_network.py` is unused legacy file.** `MINOR DEVIATION (suspected unjustified)`. Code: 80-LOC scratch file (Flax linen) confirmed orphaned by `grep -rln "dreamer_v3_network"` — only `egg-info/SOURCES.txt` mentions it. **Why it matters**: dead code; deletion candidate. Risk of confusion for new readers.
15. **`is_first` reset zeroes both `deter` and `stoch` (paper zeroes only `stoch`).** `MINOR DEVIATION (justified)`. Code: `nnx.py:81–83`. **Why it matters**: stricter reset; first-step prior is independent of previous-episode hidden state. Functionally similar to paper but not identical.
16. **Three LayerNorms in series at the embed bottleneck when `use_layer_norm=true`.** `MINOR DEVIATION (justified)`. Code: `nnx.py:516–521`. **Why it matters**: per-linear LN inside the body, plus `mod_*_ln` between body and final SiLU. Likely more LN than canonical; preserved for compatibility with modulation injection sites.
17. **Two-hot bin layout knobs hard-coded in three places (`min_v`, `max_v`, `num_buckets`).** `MINOR DEVIATION (suspected unjustified)` (separate from item 2 above, which is the substantive range deviation). Code: `util.py:19, 60`; `nnx.py:530, 576`. **Why it matters**: 255 magic number duplicated; no YAML knob. Editing requires three files.
18. **`Moments` constants hard-coded.** `MINOR DEVIATION (suspected unjustified)`. Code: `trainer.py:90`. **Why it matters**: `decay`, `max_`, `percentile_low`, `percentile_high` cannot be overridden from YAML. Values match paper; just not knob-able.
19. **Target-critic EMA decay hard-coded `0.98 / 0.02`.** `MINOR DEVIATION (suspected unjustified)`. Code: `trainer.py:504`. **Why it matters**: matches paper EMA decay value; not config-knob'd. Same comment as item 18.
20. **Buffer-clearing on stage transitions.** `EXTENSION (not in paper)`. Code: `train.py:1116–1136`. **Why it matters**: project-specific continual-learning hook; preserves only the world model and policy across stage transitions, not the replay buffer. Not a Hafner concern.
21. **Actor advantage normalises BOTH sides (`norm_returns` AND `norm_baseline`).** `MINOR DEVIATION (suspected unjustified)`. Code: `trainer.py:432–435`. **Why it matters**: comment explicitly calls this out as intentional; the two sides differ only by stop-gradient and a constant offset, so `(norm_R - norm_v) = (R - v) / scale`. Algebraically equivalent to dividing the difference by the scale. Effect zero; recording for completeness.
22. **`from_twohot` always re-applies `softmax(logits)`.** `MINOR DEVIATION (suspected unjustified)`. Code: `util.py:60–76`. **Why it matters**: even when called repeatedly on the same logits in a single step, `softmax` is recomputed. Slight redundancy; not a correctness issue.
23. **Action-dim derivation lives outside the trainer.** `EXTENSION (not in paper)`. Code: `train.py:506` — `act_dim = 4 + int(rest_action_enabled) + int(eat_action_enabled)`. **Why it matters**: the 4 is hard-coded for the gridworld discrete action space; 2 optional actions extend it. Not strictly a Dreamer concern, but flagged because `act_dim` feeds the actor head shape.
24. **`agent.imagined_rollout_probe` — 7 imagined-termination metrics.** `EXTENSION (not in paper)`. Code: `trainer.py:130, 287–294, 447–484`. **Why it matters**: project-specific instrumentation; bit-identical to pre-probe runs when off (canonical default = false).
25. **LayerNorm vs Nature RMSNorm.** `MATCHES PAPER (preprint)` / `MAJOR DEVIATION (Nature-only)`. Code: encoder/decoder/MLP all use `nnx.LayerNorm`. Nature Extended Data Table 5 specifies RMSNorm. We run the preprint architecture exactly. Lower priority because part of the Nature-only changes.
26. **Implied KL floor interaction with small encoder.** Per-state free-bits clip `max(1.0, KL)` × stoch-group sum implies a lower bound of `(0.5 + 0.1) · 1.0 = 0.6` nats averaged on `loss_kl` (see §3.7.5). Not a deviation in itself (matches Hafner's published code), but worth noting that for our small encoder (item 4), this 0.6-nat floor is a larger fraction of the total WM loss than it would be for the paper's S-profile — a configuration-sensitive interaction that does not appear at canonical width.

### §6 cross-reference completeness

Every flag in this list is also present in §3 (the per-component map) or §5 (the config table). Reviewers should spot-check that every `MINOR/MAJOR DEVIATION` and `EXTENSION` flag in §3 / §5 has a corresponding bullet here, and vice versa.

---

## §7 References

### See also — cross-architecture peers (signposting only)

A reader placing DreamerV3 in the broader model-based-RL landscape may also want to look at:

- **TWM (Robine et al., 2023, "Transformer-based World Models Are Happy with 100k Interactions")** — transformer-based competitor to RSSM; replaces GRU+categorical with a Transformer over discrete latent tokens.
- **IRIS (Micheli, Alonso, & Fleuret, 2023, "Transformers are Sample-Efficient World Models")** — discrete-VAE latents + Transformer dynamics; closest direct V3 competitor on Atari-100k. Demonstrates that categorical latents are separable from the recurrent inductive bias.
- **TD-MPC2 (Hansen, Su, & Wang, 2024)** — latent-dynamics model with model-predictive-control at *test time*; the anti-DreamerV3 (test-time planning vs. amortised policy is a live design axis).
- **SPR (Schwarzer et al., 2021) / BYOL-Explore-style auxiliaries** — self-supervised auxiliary losses on latent-dynamics representations; directly relevant if the reward head proves to be a bottleneck (cross-link with the project's reward-head failure history).

### Primary sources

- **Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023).** *Mastering Diverse Domains through World Models.* arXiv:2301.04104.
  - Local PDF: `docs/project/references/Dreamer/sources/Hafner et al. 2023 - Mastering Diverse Domains through World Models.pdf` (38 pages).
- **Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025).** *Mastering diverse control tasks through world models.* Nature, doi:10.1038/s41586-025-08744-2.
  - Local PDF: `docs/project/references/Dreamer/sources/Hafner et al. 2025 - Mastering diverse control tasks through world models.pdf` (19 pages main).
- **Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021).** *Mastering Atari with Discrete World Models.* (DreamerV2). For the categorical-RSSM and KL-balancing background. Local PDF in same folder.
- **Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020).** *Dream to Control.* (DreamerV1). For the imagination-based actor-critic with λ-returns. Local PDF in same folder.

### Project documents

- **Lit review** (paper-canonical writeup, Phase 1 + Phase 2 LaTeX): [`docs/project/references/Dreamer/dreamer_lit_review.md`](../references/Dreamer/dreamer_lit_review.md), specifically [Phase 5a — DreamerV3 preprint (2023)](../references/Dreamer/dreamer_lit_review.md#paper-5a) and [Phase 5b — DreamerV3 Nature (2025)](../references/Dreamer/dreamer_lit_review.md#paper-5b).
- **Failure-modes critique** (existing professor-rl-bayesian-dl audit): [`docs/project/critiques/dreamer_conventional_failure_modes_for_our_setup.md`](../critiques/dreamer_conventional_failure_modes_for_our_setup.md). Cross-link: §2.1, §2.2, §2.4 of that critique map to items 5, 2, and the imagination-horizon discussion in §3.5.5 of this doc.
- **Phase 1 code inventory** (the artifact this doc was authored against): `tmp/20260509_184140_dreamer_v3_code_inventory.md`. **WARNING**: this file is gitignored (`tmp/` is in `.gitignore`); the link will break on a fresh clone. Reviewers re-running Phase 1 should regenerate the artifact.

### Code

- `src/models/dreamer_v3_nnx.py` — Flax NNX model definitions (RSSM, encoder, decoder, heads, actor-critic, modulation).
- `src/models/dreamer_v3_trainer.py` — training loop, losses, imagination, replay buffer, JIT scan path.
- `src/models/dreamer_v3_util.py` — `symlog`, `symexp`, `to_twohot`, `from_twohot`, `OneHotDist`, `Moments`, `Ratio`, `hafner_init`.
- `configs/models/dreamer_v3.yaml` — canonical config.
- `configs/models/dreamer_v3_rr06.yaml` — single-knob variant restoring `replay_ratio = 0.0625`.
- `train.py` — entrypoint; DreamerV3 branch starts at L773.
- `scripts/dreamer_offline_wm_test.py` — diagnostic script using `trainer.agent.wm.{rssm.imagine_step, decoder, reward_head, continue_head}` as public API.
- **Orphan**: `src/models/dreamer_v3_network.py` — legacy scratch file, not imported (see §6 item 14).

---

## §8 Review Trail

Three reviewers ran in parallel in Phase 4 of the pilot. Their reports and the reconciliation actions taken in this revision (Phase 5) are summarised below.

| Reviewer | Doc | Verdict | Headline |
|---|---|---|---|
| code-reviewer | [docs/reviews/dreamer_v3_implementation_code_review.md](../../reviews/dreamer_v3_implementation_code_review.md) | ACCEPT-WITH-FIXES | 1 BLOCKER + 1 CONCERN + 2 NITs + 9 reverse-pass gaps |
| math-reviewer | [docs/reviews/dreamer_v3_implementation_math_review.md](../../reviews/dreamer_v3_implementation_math_review.md) | ROUTE-BACK-TO-AUTHOR | 3 BLOCKERs (twohot range, SlowTarget critic, Adam-eps fabrication) + reverse-pass gaps |
| professor-rl-bayesian-dl | [docs/project/critiques/dreamer_v3_implementation_critique.md](../critiques/dreamer_v3_implementation_critique.md) | ACCEPT-WITH-REFRAMINGS | §6 coalesce items 1–4; §1 add why-uniformity / why-imagination beats |

### Reviewer findings — resolution status

**Math-reviewer BLOCKERs (load-bearing):**

- **Math-F1: Twohot bin range is a major deviation, not `MATCHES PAPER`.** Status: **fixed**. §2.2 description corrected (raw bins span ±4.85·10^8 in paper, ±20 in our code); §3.7.2 deviation flag changed from `MATCHES PAPER` to `MAJOR DEVIATION (suspected unjustified)` with eight-orders-of-magnitude evidence cited; §5.2 `min_v`/`max_v` rows updated. Added as new §6 top-tier item 2 with forward-link to the reward-head localized-failure memo. Math-reviewer evidence (`embodied/jax/heads.py:87–97` `half = symexp(linspace(-20, 0, ...))` mirror pattern) cited inline.
- **Math-F2: λ-return bootstrap uses `target_critic`, but both papers use the fast critic.** Status: **fixed**. §3.4.5 deviation flag changed from `MATCHES PAPER (Nature wording)` to `MAJOR DEVIATION (suspected unjustified — corresponds to the preprint's ablated SlowTarget variant)`. Cited preprint Appendix C item 6 + Nature page 3 + preprint Appendix D.2 ablation verbatim. Added one-line research-follow-up note ("warrants a controlled SlowTarget-vs-current-critic comparison" + agent hand-off note); did NOT queue the experiment. Slow-target-critic forward reference added at §3.5.5 distinguishing the two slow-critic roles (real Eq. 11 regulariser vs. our non-canonical bootstrap use). Added as new §6 top-tier item 3.
- **Math-F3: Adam-epsilon "uniform 1e-5" claim is a fabrication.** Status: **fixed**. §3.8.1 paper-spec rewritten with split preprint Adam (`1e-8` WM / `1e-5` AC matches our code) + Nature LaProp + AGC + uniform LR `4e-5`. §3.7.6, §5.2 WM-eps, §6 mid-tier all corrected. **Removed old §6 item 15 (Adam-eps deviation)** — it was a false-positive deviation. Added new §6 item 8 covering the Nature optimiser swap as a separate (lower-priority) deviation.

**Math-reviewer CONCERNs:**

- **Math-F4: Nature horizon is H=15, not H=16.** Status: **fixed**. §2.6 row corrected to "15 (both versions)"; §5.2 `HORIZON` row corrected. Lit review's similar conflation flagged for downstream `literature-curator` cleanup (out of scope for this revision).
- **Math-F5: Replay-ratio default 0.0625 hides factor-of-8 spread across benchmarks.** Status: **fixed**. §6 item 1b reworded with the per-benchmark spread (Atari 200M=64, DMC=512, BSuite/Atari100K=1024, Minecraft=16 from Table A.1). New §2.5b sub-section primes the reader.

**Math-reviewer reverse-pass gaps:**

- **Math-R1: Hafner 2025 Nature optimiser detail (LaProp + AGC + LR=4e-5).** Status: **fixed**. Added in §2.6, §3.8.1, and §6 item 8.
- **Math-R2: Hafner 2025 replay capacity 5×10⁶ (not "typically 1M").** Status: **fixed**. §2.6 + §5.1 row updated.
- **Math-R3: Hafner 2025 replay-value bootstrap mechanism missing.** Status: **fixed**. §6 item 6 expanded with the bootstrap mechanism description (replay-value loss bootstraps off the imagined `R^lambda` at the start state).
- **Math-R7: Preprint Appendix C item 6 quote ("compute λ-returns using the fast critic").** Status: **fixed**. Cited verbatim in §3.4.5 as evidence for the math-F2 reclassification.

**Code-reviewer findings:**

- **Code-F1 (BLOCKER): Free-nats per-(B,T) state, NOT per stoch group; lower bound is 0.6 not 19.2 nats.** Status: **dissented-in-favor-of-other-reviewer**. Description and arithmetic in §3.5.4, §3.7.5 corrected per code-reviewer's evidence. Deviation reclassification suggestion **NOT adopted**; flag stays `MATCHES PAPER` per math-reviewer's dissent: Hafner's published `dreamerv3/rssm.py` uses `Agg(out, 1, jnp.sum)` to aggregate over the stoch-group axis BEFORE `jnp.maximum(dyn, free_nats)`, which clips per-state — exactly matching our code. Math-reviewer's evidence (the `Agg` wrapper) cited inline as the upstream re-verification path.
- **Code-F2 (CONCERN): `_scan_train_gpu` static-args list misstates which args are static.** Status: **fixed**. §3.8.3 list updated to `(graphdef, num_steps, b_cap, b_seq_len, pos_cap, pos_slots, recent_slots, recent_window)` with traced-args also listed.
- **Code-F3 (NIT): ValueError site for missing `encoding_mode`.** Status: **fixed**. §3.2 cite updated to `trainer.py:78` (`config.get_mandatory(...)`) with secondary cite at `nnx.py:204–205`.
- **Code-F4 (NIT): two off-by-one line ranges (`trainer.py:413–416`, `nnx.py:432–441`).** Status: **no-action** (both are 1-line overlap with whitespace; not load-bearing).
- **Code-R1–R9: Reverse-pass completeness gaps** (`Moments.normalize` dead code, `OneHotDist.mode` dead code, `DreamerV3Agent.__call__` dead, `DreamerV3Agent.initial_state` dead, `ReplayBuffer.sample_multiple` GPU NotImplementedError, `jax.named_scope` instrumentation, six DreamerV3 config variants, `is_first` shape polymorphism, `FiLMNoNorm` rejection branch). Status: **fixed**. Added as a new §3.9 sub-section "Dead code, unused config, and additional code surfaces (cleanup opportunities)" with one-line entries each.

**Professor (rl-bayesian-dl) reframings:**

- **§1 missing why-uniformity beat.** Status: **fixed**. Added one sentence linking the cross-task uniformity claim to the three-trick rationale.
- **§1 missing why-imagination beat.** Status: **fixed**. Added one sentence on imagination as the sample-efficiency lever.
- **§2.5 lumps replay_ratio with categorical machinery.** Status: **fixed**. Promoted to its own §2.5b sub-section.
- **§3.5.5 missing forward-reference to slow-target critic.** Status: **fixed**. Added one-line forward-reference at §3.5.5; reconciled with math-F2 finding by distinguishing the two slow-critic roles (real Eq. 11 regulariser vs. our non-canonical bootstrap use).
- **§3.7.4 Moments / percentile-norm undertitled.** Status: **fixed**. Re-headlined to "percentile return scaling — entropy-coefficient invariance mechanism" with one-paragraph architectural-role sentence.
- **§6 items 1–4 should coalesce.** Status: **fixed**. Coalesced into single top entry "Item 1: Coupled WM training-distribution deviation (4 sub-items)" with sub-items 1a/1b/1c/1d.
- **§6 swap items 6 and 7 (DYN_SCALE up).** Status: **fixed**. New ordering ranks `DYN_SCALE = 0.5` (item 5, was 7) above replay-value loss (item 6, was 6); empirical anchor "second-most-impactful per Hafner-2025 ablation" lifted into the §6 description.
- **§6 add KL floor sentence.** Status: **fixed**. Added as §6 item 26 noting the configuration-sensitive interaction with the small encoder.
- **§7 missing cross-architecture peers.** Status: **fixed**. Added "See also" paragraph naming TWM, IRIS, TD-MPC2, SPR.

### Final §6 deviation count

The 25-item §6 list became 26 items after this reconciliation: removed 1 false-positive (old item 15 Adam-eps) + added 3 new top-tier items (twohot range, SlowTarget critic, Nature optimiser swap) + added 1 new mid-tier item (LayerNorm vs RMSNorm) + added 1 new mid-tier note (KL-floor interaction) + coalesced 4 separate items into 1 top entry with 4 sub-items.

---

Verified by: senior-developer (Phase 5 reconciliation, 2026-05-09)
