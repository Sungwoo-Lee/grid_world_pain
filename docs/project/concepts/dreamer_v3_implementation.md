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

**Item count: 30** (26 original + 4 sheeprl-comparison additions, items 27–30 below; the 4 are appended in their natural-severity slot rather than re-numbered into items 4-7 to preserve reader memory of "item 2 = twohot range" / "item 3 = SlowTarget critic"). Severity ordering across the full 30-item list is described inline; the appended items 27–30 are flagged as **"appended; severity rank shown in body"**.

### Top-tier — large, structural, load-bearing

1. **Coupled WM training-distribution deviation (4 sub-items).** This is one coupled deviation in *what the world model sees per gradient step*; the four sub-items below interact multiplicatively, so the experiment design implication is "revert the cluster, not one knob at a time".
   - **1a. Mixture sampling + positive-reward buffer (`agent.sampling_mode: "mixture"`).** `EXTENSION (not in paper)`. Code: `trainer.py:706–772` (GPU mixture), `train.py:1397–1463` (positive-buffer write trigger). Paper baseline: uniform sub-sequence sampling only ([Phase 5a §2.5](../references/Dreamer/dreamer_lit_review.md#paper-5a)). DreamerV4-inspired backport (see [Phase 6](../references/Dreamer/dreamer_lit_review.md#paper-6)). Five of every 16 batch slots are drawn from a separate 100k-transition buffer of "any block where some reward > 0.0", five from the most-recent 10000 transitions, six uniform.
   - **1b. `agent.replay_ratio: 0.5`.** `MAJOR DEVIATION (deliberate)`. Code: `train.py:789, 1620; trainer.py Ratio gating`. Per Hafner 2023 Table A.1, the paper default is benchmark-dependent — Atari 200M = 64 (≈0.0156 grad/env normalised), DMC = 512 (≈0.0625 grad/env), BSuite/Atari 100K = 1024, Minecraft = 16. A factor-of-8 spread across benchmarks (DMC mid-band ≈ 0.0625; our 0.5 is therefore 8× the DMC mid-band, 32× the Atari 200M default). The project has been treating 0.0625 as canonical; the actual paper-spec spread is wider. The variant `dreamer_v3_rr06.yaml` exists specifically to restore this knob to the DMC mid-band.
   - **1c. `agent.sequence_length: 128`** (paper 64). `MAJOR DEVIATION (deliberate)`. Code: `dreamer_v3.yaml:4`; `trainer.py:917–1006`. 2× canonical.
   - **1d. Block-aligned (not uniform) sub-sequence sampling.** `MAJOR DEVIATION (deliberate)`. Code: `trainer.py:973–1006`. Env-major storage + block-aligned starts means the diversity of sampled batches is bounded by `buffer_capacity // sequence_length` block positions.
   - **Why the coupling matters**: a high replay ratio combined with a positive-biased mixture sampler means the WM is seeing the **same set of positive-reward blocks repeatedly** across many gradient steps, in a regime where the canonical V3 buffer would have shown those blocks once or twice. The sequence-length doubling and block-aligned grid further compound the effect. The project's conventional-fixes battery already showed that `replay_ratio = 0.5` fixed the NoPred collapse but did not recover predator-task survival — that negative result is more interpretable if you treat 1a+1b+1c+1d as a coupled triple rather than as four independent knobs.
2. **Two-hot bin range narrowed by 8 orders of magnitude.** **Status: `RESOLVED-PARTIAL` (was `MAJOR DEVIATION (suspected unjustified)`)**. Code: `util.py:60–76` (`from_twohot`); fix gated on new mandatory key `agent.paper_canonical_twohot_bins` (default `true`). Original-state description: our `bucket_vals = jnp.linspace(symlog(-20), symlog(+20), 255)` created symlog-space bins in `linspace(-3.045, +3.045)`, mapping under `symexp` to raw-space support `±20`. Paper convention is `linspace(-20, +20)` *in symlog space*, mapping to raw-space support `±symexp(20) ≈ ±4.85·10^8` — eight orders of magnitude mismatch. **Empirical resolution (Cell Z2, 2026-05-10 → 11)**: applying the paper-canonical bin construction delivered **a 36% additional reward-MAE reduction beyond Z1** (0.277 → 0.177 at h=5), with the load-bearing mechanistic prediction validated — training-time `model_reward_mae_neg` dropped **45% Z1→Z2** (0.649 → 0.359), exactly the negative-event-error narrowing the bin-coverage mechanism predicted; `mae_pos` regressed slightly (0.544 → 0.805) confirming the fix is specifically curative on negative events. The cumulative cascade A1 → Z1 → Z2 reads −54% on reward-MAE. H1 (< 0.15) narrowly missed by 0.027 — verdict is H2 partial improvement, not full repair. The deviation is no longer suspected unjustified; the paper-canonical recipe verifiably repaired what its mechanism predicted it would repair. The remaining residual is now flagged as candidate #1 (item 28 below, GRU reset gate) per the residual-error pattern. **Verification report**: [`docs/develop/active/diagnosis/dreamer_twohot_bin_range_fix.md`](../../develop/active/diagnosis/dreamer_twohot_bin_range_fix.md) §Verification Report. **Forward-link**: `.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md`.
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

### Sheeprl-comparison additions (items 27–30; appended, not renumbered)

The four items below come from §9 (sheeprl reference-impl comparison, 2026-05-10). Severity is shown inline; in a strict ranking they would slot into the top-tier (#27, #28) and mid-tier (#29, #30) regions of the list above. Appended rather than inlined so existing references to "item 2 = twohot range" / "item 3 = SlowTarget critic" remain stable. See §9.11 for the comparison evidence.

27. **[HIGH-tier; would-rank near item 4] Reward and critic output layers not zero-initialised.** **Status: `RESOLVED-PARTIAL` (was `MAJOR DEVIATION (suspected unjustified, ours)`)**. Fix gated on new mandatory key `agent.zero_init_reward_critic` (default `true`). **Empirical resolution (Cell Z1, 2026-05-10)**: applying sheeprl/Hafner-published-code-style zero-init on the final reward and critic output `Linear` layers delivered **a 28% reward-MAE reduction at h=5** (0.386 → 0.277). Training-time `model_reward_mae_pos` improved 49% but `model_reward_mae_neg` only 14% — that asymmetry is the residual-error pattern that directed item 2 (paper-canonical bins) as the next fix candidate (which then fired H2 with mae_neg dropping 45%; see item 2 above). H1 (< 0.15) was missed; verdict H2 partial improvement. The deviation is no longer suspected unjustified; the paper-canonical recipe verifiably repairs what its mechanism predicts (output-layer noise dominates the very first gradient steps of the two-hot CE head). **Verification report**: [`docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`](../../develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md) §Verification Report. **Memory insight**: [`.claude-memory/memories/dreamer_diagnosis/20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry.md`](../../../.claude-memory/memories/dreamer_diagnosis/20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry.md). Original-state description below preserved for traceability:

   *(original)* Code: `util.py:195` (`hafner_init` returns variance-scaled truncated normal, scale 0.8796); applied uniformly to every `nnx.Linear(..., kernel_init=hafner_init(), rngs=rngs)` in the model — including the reward and critic output projections at `nnx.py:530, 576` (`MLP(feat_dim, 255, ...)` whose final layer is constructed inside `MLP.__init__` at `nnx.py:458`). Paper convention (Hafner published `dreamerv3/agent.py` and `dreamerv3/jaxutils.py` — `outscale=0.0` argument routed into `Linear(... outscale=0.0)`) zero-initialises the **final** Linear of the reward and critic heads only (kernel = 0, bias = 0). Sheeprl mirrors this exactly: `agent.py:1170–1180` calls `critic.model[-1].apply(uniform_init_weights(0.0))` and `world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))` when `hafner_initialization: True`. Note: the override is **only on the final output Linear**, not on the MLP hidden layers (those keep `uniform_init_weights(1.0)` ≈ our `hafner_init`). **Why this matters**: with non-zero output init, the reward head emits a non-uniform softmax over the 255 two-hot bins from step 0, predicting random nonzero rewards that the head must then *unlearn* before learning the true signal. Paper recipe makes the head emit a flat (uniform) distribution at init — under the two-hot CE objective, the resulting gradient at step 0 is exactly `softmax(0) - target_twohot`, the cleanest possible learning signal. **Directly relevant to the reward-head MAE 0.39 finding** (forward-link `.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md`); this is the §9.11.4 candidate the user has approved for action — see [`docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`](../../develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md). Among the four sheeprl-comparison candidates, this is the one most narrowly targeted at the empirical reward-head failure (output-layer init dominates the *very first* gradient steps of a two-hot CE head).

28. **[HIGH-tier; would-rank near item 4] GRU reset gate computed but never applied to the candidate.** **`MAJOR DEVIATION (suspected unjustified — likely correctness bug, ours)`** — **NOW NEXT IN THE FIX CASCADE**. Code: `nnx.py:32–36` (`LayerNormGRUCell.__call__`): `reset, update, cand = jnp.split(gates, 3, axis=-1); reset = nnx.sigmoid(reset); update = nnx.sigmoid(update); cand = jnp.tanh(cand)` — `reset` is computed (sigmoid applied) but the variable is then **never multiplied into anything**; `cand` goes through plain `tanh` and on into the GRU update. Paper canonical (preprint Eq. 3 / Cho et al. 2014 GRU): `cand = tanh(reset * cand)` — the reset gate gates the candidate update relative to the previous hidden state. Hafner published `dreamerv3/nets.py` GRU does this; sheeprl `models/models.py:399–401` does `cand = torch.tanh(reset * cand)` after splitting the fused `(reset, cand, update)`. **Why this matters**: structural GRU change affecting *every* recurrent step (one such cell call per env step, scanned across the imagination horizon). The deeper concern is dead-computation suspicion: `reset` is computed but unused, which suggests a refactor stranded the reset gate. Severity is broad (impacts every WM forward pass and every imagined rollout). **Promoted to next-in-queue (2026-05-11)** after items 27 + 2 fired H2 partial-fixes — the residual MAE 0.18 still has H1 just out of reach, and the long-horizon error compounding observed in Cell Z2 (h=5 → h=50 reward MAE: 0.18 → 3.05) is the exact symptom GRU-cell dynamics quality affects. Cross-link: `.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md`.

29. **[MID-tier; would-rank in §6 mid-tier list] Critic self-EMA regularisation term against `target_critic` missing from `loss_critic`.** **`MAJOR DEVIATION (suspected unjustified, ours)`**. Code: `trainer.py:421–430` — only one CE term, `loss_critic_step = -jnp.sum(target_twohot * jax.nn.log_softmax(v_pred_logits), axis=-1)` (CE against λ-return two-hot target). Paper canonical (preprint Eq. 10 critic-EMA self-regularisation): `value_loss = -CE(λ-returns) - CE(target_critic_value)` — second term pushes the online critic's logits toward the slow-target critic's predicted value. Sheeprl `dreamer_v3.py:307–316`: `value_loss = -qv.log_prob(lambda_values.detach()) - qv.log_prob(predicted_target_values.detach())` — the second `-qv.log_prob(predicted_target_values.detach())` is exactly that term. **Distinct from §6 item 3** (which is about *bootstrap source* for λ-returns, an upstream choice): items 3 and 29 are independent loss-shaping mechanisms. **Why this matters**: items 3 + 29 together imply our slow critic plays only one role (bootstrap source) when paper-canonical it should play exactly the inverse role (regularisation target only, with online critic for bootstrap). Severity is value-learning-stability rather than reward-prediction; lower priority than items 27/28 for the active diagnostic agenda. The follow-up experiment-design hand-off already noted at item 3 should factor item 29 into the comparison condition.

30. **[MID-tier; would-rank in §6 mid-tier list] Prior + posterior heads have no hidden layer (paper has 1 hidden layer).** **`MINOR DEVIATION (suspected unjustified, ours)`**. Code: `nnx.py:51, 58–59`: `self.img_in = nnx.Linear(stoch_dim*discrete + action_dim, deter_dim, ...)`; `self.img_out = nnx.Linear(deter_dim, stoch_dim*discrete, ...)`; `self.obs_out = nnx.Linear(deter_dim + embed_dim, stoch_dim*discrete, ...)` — single Linear projections in each case. Paper canonical: representation and transition models are MLPs with one hidden layer of width = `representation.hidden_size` (sheeprl `agent.py:1018–1051`: `representation_model = MLP(input_dim, output=stoch*disc, hidden=[representation.hidden_size])`; Hafner published code agrees — one-hidden-layer MLP for both prior and posterior heads). **Why this matters**: head capacity reduced relative to paper; downstream effect on prior/posterior expressiveness, which sits upstream of KL terms and imagined dynamics. Architectural undersizing rather than a correctness bug. Severity: lower than item 28; queued behind items 27/28/29.

### §6 cross-reference completeness

Every flag in this list is also present in §3 (the per-component map) or §5 (the config table), **except items 27–30**: those four come from §9 (sheeprl-comparison reverse-pass) and have not yet been back-propagated into §3 / §5 (the §3 / §5 entries currently mark these surfaces as `MATCHES PAPER`; §6 items 27–30 supersede those local flags). A future revision should propagate the four into §3 (RSSM cell §3.1, prior/posterior heads §3.1, reward head §3.4.1, critic head §3.4.4, critic loss §3.5.6) and §5 (`hafner_init` row + new init-knob row). Reviewers should spot-check that every `MINOR/MAJOR DEVIATION` and `EXTENSION` flag in §3 / §5 has a corresponding bullet here, and vice versa.

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

## §9 Comparison with sheeprl reference implementation

### §9.1 Why compare against sheeprl, scope, framework caveat

`sheeprl` is the actively-maintained PyTorch + Lightning-Fabric implementation of Hafner 2023 by Eclectic Sheep (https://github.com/Eclectic-Sheep/sheeprl), widely used as a reference. It tracks the preprint exactly (the `dreamer_v3.py` docstring at L1–3 explicitly says "Adapted from the original implementation from https://github.com/danijar/dreamerv3"), with one `decoupled_rssm` extension we ignore here. This section gives a **second paper-grounded check** beyond §3's paper-vs-our-code audit: where sheeprl matches the paper but we do not, the deviation is doubly-confirmed; where we match the paper but sheeprl does not, the sheeprl-side anomaly is logged for the reader's awareness without changing our §6 list.

The walked artefact lives at `tmp/sheeprl/` checked out at git rev `33b6366` (2024-07-12, "Add tanh_normal dist to PPO (#312)"). Files inspected: `algos/dreamer_v3/{agent.py, dreamer_v3.py, loss.py, utils.py, evaluate.py}`, `configs/algo/dreamer_v3{,_XS,_S,_M,_L,_XL}.yaml`, `configs/exp/dreamer_v3.yaml`, plus dependencies in `models/models.py` (`LayerNormGRUCell`), `data/buffers.py` (`SequentialReplayBuffer`, `EnvIndependentReplayBuffer`), `utils/utils.py` (`symlog/symexp`, `Ratio`, `Moments` constants), `utils/distribution.py` (`SymlogDistribution`, `MSEDistribution`, `TwoHotEncodingDistribution`).

**Framework idiom is filtered from algorithmic substance.** PyTorch `nn.Module.forward()` vs Flax NNX `__call__`, `torch.optim.Adam` vs `optax.adam`, `torch.distributions.OneHotCategoricalStraightThrough` vs our hand-rolled `OneHotDist`, `lightning.fabric.clip_gradients` vs `optax.clip_by_global_norm`, `Independent(Bernoulli(...), 1)` vs our `optax.sigmoid_binary_cross_entropy` — all of these are framework idiom and are NOT listed below. Only differences with potential algorithmic effect are flagged.

### §9.2 Three-way comparison frame

Every difference falls into one of five categories:

- **Both match paper, both match each other** — no-op, not listed.
- **Sheeprl matches paper, ours diverges** — confirms a §6 deviation (cross-link to §6 item N).
- **Ours matches paper, sheeprl diverges** — sheeprl-side anomaly; flagged for reader awareness; does not change §6.
- **Both diverge from paper but in different ways** — each side flagged separately.
- **Both diverge from paper in the SAME way** — paper-vs-community drift; both implementations made the same call. Important if it suggests the deviation is universally adopted by community implementations.

The body of §9 organises findings by component (architecture, losses, buffer, numerics, optimiser, diagnostics, configs); §9.10 is the condensed all-in-one table; §9.11 lists proposed updates to §6.

### §9.3 Architectural differences (RSSM, encoder, decoder, heads)

#### §9.3.1 RSSM cell (recurrent core)

- **Sheeprl** (`algos/dreamer_v3/agent.py:281–342, models/models.py:331–410`):
  - `RecurrentModel` first runs a *pre-MLP*: `MLP(stoch_state ⊕ action, dense_units, hidden=[dense_units], LayerNorm, SiLU)` (`agent.py:309–317`). The cell's input is therefore `dense_units = 1024 (XL) / 512 (S)` after a SiLU+LN projection, NOT the raw `(stoch ⊕ action)`.
  - `LayerNormGRUCell` (`models/models.py:331–410`) uses **one fused linear** `Linear(input ⊕ hidden, 3*hidden, bias=False)` followed by **a single LayerNorm** on the fused output, then chunked into `(reset, cand, update)`. Updates: `reset = sigmoid(reset)`, `cand = tanh(reset * cand)` (reset gate **applied** to candidate), `update = sigmoid(update - 1)` (canonical Hafner `-1` bias on update gate), `hx = update * cand + (1 - update) * hx`.
  - The recurrent state has a **learnable initial state** (`nn.Parameter`, `agent.py:382–385`) passed through `tanh` at use, with `is_first` doing a hard `(1 - is_first) * recurrent_state + is_first * initial_recurrent_state` mix (`agent.py:427–428`) — i.e. the initial state is learned, not zero, and `is_first` *replaces* with the learned state (not zeroes).
- **Ours** (`src/models/dreamer_v3_nnx.py:18–39, 41–139`):
  - No pre-MLP. The cell input is `silu(img_in(stoch ⊕ action))` from a single `Linear(stoch_dim*discrete + action_dim, deter_dim)` directly into the cell (`nnx.py:51, 86–88`).
  - `LayerNormGRUCell` uses **two separate dense projections** (`dense_ih: Linear(hidden, 3*hidden, bias=False)` and `dense_hh: Linear(hidden, 3*hidden, bias=False)`), each followed by **its own LayerNorm** (`ln_ih`, `ln_hh`), then **summed**, then split into `(reset, update, cand)`. Updates: `reset = sigmoid(reset)`, `update = sigmoid(update)` (NO `-1` bias), `cand = tanh(cand)` — **the reset gate is computed but never multiplied into the candidate**, and `update = sigmoid(update)` initialises near 0.5 instead of the canonical near-0 (Hafner uses `sigmoid(x - 1)` ≈ 0.27 at init), `h_new = (1 - update) * h + update * cand`.
  - Initial state is hard zero (`initial(B)` returns `jnp.zeros`, `nnx.py:61–67`); `is_first` zeroes both `deter` AND `stoch` (already flagged in §3.1, §6 item 15).
- **Paper.** Hafner's published `dreamerv3/nets.py` GRU uses one fused linear, one LayerNorm, `cand = tanh(reset * cand)`, and `update = sigmoid(update - 1)` (the `-1` bias is the standard Hafner-DreamerV3 convention, present in the published code at `dreamerv3/nets.py:GRU.__call__`). Sheeprl's cell matches this exactly. Our cell deviates on three counts: split LN/dense layout, missing reset-gate application, missing `-1` update bias.
- **Classification.**
  - **Split-LN/two-dense layout** — `MINOR DEVIATION (justified, ours)`. Algebraically equivalent to `Linear(input ⊕ hidden, 3h)` only if the LayerNorm is applied AFTER the sum; pre-summing two LNs is *not* equivalent to LN of sum. May change the LN-normalised gate magnitudes slightly. Sheeprl matches paper.
  - **Missing reset-gate application to candidate** — `MAJOR DEVIATION (suspected unjustified, ours)`. This is a structural GRU change: paper and sheeprl both apply `cand = tanh(reset * cand)`, our code does `cand = tanh(cand)`. The reset gate is computed (`reset = sigmoid(reset)`) but never used. **NEW §6 candidate**. Suggests a refactor that stranded the reset gate; effect on training depends on whether the network learns to compensate via `update` and `cand` magnitudes.
  - **Missing `-1` bias on update gate** — `MINOR DEVIATION (suspected unjustified, ours)`. Paper convention `sigmoid(x - 1)` ≈ 0.27 at init biases the GRU toward keeping the previous hidden state; our `sigmoid(x)` ≈ 0.5 at init mixes 50/50. Effect: more aggressive mixing of new candidate vs. carry at initialisation. Likely corrected by training but the inductive bias is different.
  - **Missing pre-MLP before the cell** — `MINOR DEVIATION (justified, ours)`. We project `(stoch ⊕ action)` to `deter_dim` via a single Linear+SiLU; sheeprl does the same with one extra hidden layer (`MLP(..., hidden=[dense_units], ...)`). For our hidden=128 setup vs sheeprl's 1024, this is a width difference more than a structural one.
  - **Initial state is zero (ours) vs learnable (sheeprl, paper).** `MINOR DEVIATION (suspected unjustified, ours)`. Sheeprl flag `learnable_initial_recurrent_state: True` (`configs/algo/dreamer_v3.yaml:54`) is paper-canonical. Our `initial(B)` returns hard zero. Since `is_first` reset is hard at the first step anyway, the practical effect is small — but for `imagine_step` start states (which use the posterior `deter`, not the initial), this difference does not bite either way. Flag for completeness.

#### §9.3.2 Posterior + prior heads (representation + transition models)

- **Sheeprl** (`agent.py:1018–1051`):
  - **Representation model**: `MLP(deter_dim + embed_dim, stoch_dim*discrete, hidden=[representation.hidden_size=1024 (XL)/512 (S)], SiLU, LayerNorm)` — a one-hidden-layer MLP, NOT a single Linear.
  - **Transition model**: `MLP(deter_dim, stoch_dim*discrete, hidden=[transition.hidden_size=1024/512], SiLU, LayerNorm)` — also one hidden layer.
- **Ours** (`nnx.py:51–58`):
  - **Posterior head (`obs_out`)**: single `Linear(deter_dim + embed_dim, stoch_dim*discrete)` — no hidden layer.
  - **Prior head (`img_out`)**: single `Linear(deter_dim, stoch_dim*discrete)` after the GRU — no hidden layer.
- **Paper.** Hafner published code uses an MLP (one hidden layer) for both heads. Sheeprl matches. Our code is shallower.
- **Classification.** `MAJOR DEVIATION (suspected unjustified, ours)`. **NEW §6 candidate** — our prior/posterior heads are shallower than paper-canonical. With our small `deter_dim=512` and `embed_dim=128`, removing the hidden layer halves the head capacity; not paper-faithful. Confirmed by sheeprl as intended-paper behaviour.

#### §9.3.3 Encoder

- **Sheeprl** (`agent.py:100–151`): `MLPEncoder` is a flat `MLP(input_dim, [dense_units]*mlp_layers, SiLU, LayerNorm)`. For S profile: `dense_units=512`, `mlp_layers=2`; for XL: `1024 × 5`. Symlog applied inside `forward` (`agent.py:150`). One LayerNorm per linear; final activation is the last hidden's SiLU (no extra "final" SiLU). For vector observations sheeprl has only this flat path (CNN path is for vision). MultiEncoder concatenates CNN + MLP outputs.
- **Ours** (§3.2, `nnx.py:197–370`): Hierarchical hub (per-sensor MLP → multimodal hub MLP → final SiLU) with embed_dim=128, hidden 128. Three LayerNorms in series at the embed bottleneck when `use_layer_norm=true` (per-linear LN inside body, plus optional `mod_*_ln` LN, plus the body's last linear's own LN). Symlog applied inside `train_step` (not inside the encoder).
- **Paper.** MLP encoder for vector observations: `mlp_layers × dense_units`, LayerNorm + SiLU. Our hierarchical structure is project-specific.
- **Classification.**
  - Hierarchical hub — `EXTENSION (not in paper, ours)`. Already flagged in §3.2 / §6 item 1 as deliberate. Sheeprl confirms paper has only the flat MLP.
  - Width 128 — `MAJOR DEVIATION (deliberate, ours)`. §3.2 / §6 item 4. Confirmed against sheeprl's S=512, M=640, L=768, XL=1024 — our 128 is below sheeprl XS=256.
  - Three LNs in series — `MINOR DEVIATION (suspected unjustified, ours)`. §3.2. Sheeprl has one LN per hidden; we have body-LN + (optional) `mod_*_ln`. Confirmed by sheeprl as the more standard pattern.
  - Symlog applied inside `train_step` (not inside the encoder) — `FRAMEWORK-ONLY`. Sheeprl applies `symlog` inside `MLPEncoder.forward`; we apply it at the call site. Same numerical result; placement difference only.

#### §9.3.4 Decoder

- **Sheeprl** (`agent.py:229–278`): `MLPDecoder` is `MLP(latent_state_size, [dense_units]*mlp_layers, SiLU, LayerNorm)` followed by a per-key `Linear(dense_units, output_dim)` *head*. The decoder body is **shared across MLP keys**, with separate output heads for each. Output prediction is wrapped in `SymlogDistribution(..., dist="mse")` (`utils/distribution.py:152–193`) which computes `(symlog(target) - mode)**2` MSE — i.e. **decoder predicts symlog-space, target is symlog'd before MSE**. This matches our recipe.
- **Ours** (§3.3, `nnx.py:372–442`): Hierarchical decoder mirrors hierarchical encoder; per-sensor heads after a shared multimodal hub. Loss is `mean(square(recon - symlog(obs)))` in `trainer.py:213–215` — decoder predicts symlog space, target is symlog'd, no `symexp` on output.
- **Classification.** Same recipe (decoder-in-symlog-space MSE) — both match paper. Hierarchical structure of decoder is `EXTENSION (ours)`.

#### §9.3.5 Reward head

- **Sheeprl** (`agent.py:1099–1112`): `MLP(latent_state_size, output_dim=255 bins, hidden=[dense_units]*mlp_layers, SiLU, LayerNorm)`. The bin count is `cfg.algo.world_model.reward_model.bins = 255`. Wrapped in `TwoHotEncodingDistribution(logits, low=-20, high=20, transfwd=symlog, transbwd=symexp)`.
- **Ours** (§3.4.1, `nnx.py:530`): `MLP(feat_dim, 255, [128, 128])`. Bin count and range hard-coded in `to_twohot/from_twohot` defaults.
- **Classification.** Architecture (255-bin logit head with two-hot CE + symlog target) matches in both. **The two-hot bin range is the substantive difference** — see §9.6.

#### §9.3.6 Continue head

- **Sheeprl** (`agent.py:1114–1127`): `MLP(latent_state_size, output_dim=1, hidden=[dense_units]*mlp_layers, SiLU, LayerNorm)`. Wrapped in `Independent(BernoulliSafeMode(logits=...), 1)` (`dreamer_v3.py:167`); BCE via `-pc.log_prob(continue_targets)` where `continue_targets = 1 - terminated` (`dreamer_v3.py:168`).
- **Ours** (§3.4.2, `nnx.py:531`): Same head structure (`MLP(feat_dim, 1, [128, 128])`), BCE via `optax.sigmoid_binary_cross_entropy(cont_pred, 1.0 - terminal[..., None])`.
- **Classification.** Both match paper.

#### §9.3.7 Actor head

- **Sheeprl** (`agent.py:694–845`): `Actor` class with `MLP(latent_state_size, [dense_units]*mlp_layers=512×2 (S) / 1024×5 (XL), SiLU, LayerNorm)` body + per-action-component head `nn.Linear(dense_units, action_dim)`. Discrete distribution: `OneHotCategoricalStraightThrough(logits=_uniform_mix(logits))` (`agent.py:832`). Continuous distribution path supports `auto/normal/tanh_normal/scaled_normal` (we don't use). Unimix applied inside `_uniform_mix` (`agent.py:839–845`): `probs = (1 - unimix) * softmax(logits) + unimix / num_classes`, then `logits = probs_to_logits(probs)` — i.e. unimix-mixed logits are returned, and the distribution is constructed from those mixed logits. Unimix value `cfg.algo.unimix = 0.01` is threaded through (`agent.py:1149`).
- **Ours** (§3.4.3, `nnx.py:572–576`): `MLP(feat_dim, act_dim, [128, 128])` → `OneHotDist(logits, unimix=0.01)`. Unimix mixing inside `OneHotDist.__init__` (`util.py:83–89`). YAML key `agent.unimix` is dead — constructor default is the operative value (§3.7.3, §6 item 10).
- **Classification.**
  - Architecture matches paper at structural level; widths differ (§6 item 4).
  - Actor depth — `MAJOR DEVIATION (deliberate, ours)`. Sheeprl `mlp_layers=2` (S) / 5 (XL). Ours `[128, 128]` = 2 hidden layers. **Depth matches the S profile**, only width differs.
  - Unimix YAML key being dead — `MINOR DEVIATION (suspected unjustified, ours)`. §6 item 10. Sheeprl threads `cfg.algo.unimix` through; our YAML key `agent.unimix` is not threaded.
  - Action-clip parameter — `EXTENSION (sheeprl-only, but only used in continuous-action path)`. Sheeprl has `action_clip: 1.0` (`configs/algo/dreamer_v3.yaml:129`) for continuous actions; not relevant to our discrete-action setup.

#### §9.3.8 Critic head

- **Sheeprl** (`agent.py:1153–1166`): `MLP(latent_state_size, output_dim=cfg.algo.critic.bins=255, hidden=[dense_units]*mlp_layers, SiLU, LayerNorm)`. Wrapped in `TwoHotEncodingDistribution(logits, dims=1)`.
- **Ours** (§3.4.4, `nnx.py:576`): `MLP(feat_dim, 255, [128, 128])`. Same `to_twohot`/`from_twohot` machinery.
- **Classification.** Architecture matches; width is the §6 item 4 deviation. Actor and critic both depth=2 hidden layers on our side, vs sheeprl S=2 hidden — **depth matches sheeprl S profile exactly**, width is the smallness difference.

#### §9.3.9 Slow target critic

- **Sheeprl** (`dreamer_v3.py:674–680`): The target critic is updated every `cfg.algo.critic.per_rank_target_network_update_freq` gradient steps via `tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)` with `tau = 0.02` (config `algo.critic.tau`). Special case at gradient step 0: `tau = 1` (hard copy). This is a **periodic** EMA-style update (every `update_freq` grad steps), defaulting to **every 1 gradient step** with `tau = 0.02` — which makes it identical to per-step EMA in the canonical config. Sheeprl wraps the target_critic via `fabric_player.setup_module(target_critic)` (`agent.py:1220`); it lives outside the training Fabric.
- **Ours** (§3.4.5, `trainer.py:501–505`): Per-step EMA `0.98 * target + 0.02 * online` after every `train_step`. No update-frequency knob; effectively identical to sheeprl with `update_freq=1, tau=0.02`. **The EMA decay matches.**
- **Critical difference — what is bootstrapped at λ-return time.**
  - **Sheeprl** (`dreamer_v3.py:243–256`): `predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean` — uses **the ONLINE/fast critic** for `predicted_values` going into `compute_lambda_values`. The target critic `target_critic` is used **only** in the critic loss (`dreamer_v3.py:307–310`):
    ```python
    qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
    predicted_target_values = TwoHotEncodingDistribution(
        target_critic(imagined_trajectories.detach()[:-1]), dims=1
    ).mean
    ...
    value_loss = -qv.log_prob(lambda_values.detach())
    value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
    ```
    i.e. critic loss = CE on actual `lambda_values` PLUS a self-regularisation CE on `predicted_target_values` (the target critic's own value estimate, used as a soft target). This is paper Eq. 10 verbatim ("regularizing the critic towards predicting the outputs of an exponentially moving average of its own parameters").
  - **Ours** (`trainer.py:367, 388, 411`): `target_critic` is the bootstrap source for `compute_lambda_values`; the regularisation-via-target-critic CE term (sheeprl's `qv.log_prob(predicted_target_values.detach())`) does **not exist** in our code.
- **Classification.**
  - **λ-return bootstrap source uses target critic, not online critic** — `MAJOR DEVIATION (suspected unjustified, ours)`. **CONFIRMS §6 item 3** (`MAJOR DEVIATION (suspected unjustified — corresponds to the preprint's ablated SlowTarget variant)`). Sheeprl confirms paper-canonical recipe. The earlier §3.4.5 evidence (preprint Appendix C item 6 + Nature page 3) is now triply-confirmed: paper text + paper code + sheeprl reference impl all agree.
  - **Critic self-regularisation term against target critic is missing** — `MAJOR DEVIATION (suspected unjustified, ours)`. **NEW §6 candidate**. Sheeprl `value_loss = -qv.log_prob(lambda_values.detach()) - qv.log_prob(predicted_target_values.detach())` — the second term is paper-Eq.-10 self-EMA regularisation. Our code has only the first term. This is a *separate* deviation from §6 item 3: §6 item 3 is "wrong bootstrap source for λ-returns"; this new finding is "missing regularisation term in critic loss". The two deviations are entangled because both are about how the slow critic enters the loss, but they are distinct mechanisms.

### §9.4 Loss differences (recon, reward, continue, KL dynamics/repr, λ-returns, actor, critic)

#### §9.4.1 Reconstruction loss

- **Sheeprl** (`loss.py:61, dreamer_v3.py:152–161`): `observation_loss = -sum_k po[k].log_prob(observations[k])` where `po[k]` is `MSEDistribution` for CNN keys (raw MSE) or `SymlogDistribution(..., dist="mse")` for MLP keys (`(symlog(target) - mode)^2`). Aggregation: `sum` over the event dims (the trailing observation axis). The `.mean()` happens only at the final `reconstruction_loss = (kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss).mean()` (`loss.py:80`).
- **Ours** (§3.5.1, `trainer.py:213–215`): `loss_recon = jnp.mean(jnp.square(recon - obs))` where `obs = symlog(batch['obs'])`. **Mean over all axes**, not sum-over-event-dim then mean-over-batch.
- **Classification.** `MINOR DEVIATION (suspected unjustified, ours)`. **NEW §6 candidate (low-priority)**. Sheeprl does `sum_dim(squared_error) → mean_over_batch_time`; we do `mean_over_all`. Effect: our loss is sheeprl's divided by `obs_dim`. With `obs_dim ≈ 19` for our gridworld, the implicit weighting of the recon loss is `~19×` smaller than canonical. The relative weight of reconstruction vs other WM losses (which are already in mean form) shifts: in sheeprl `loss_recon` has implicit weight `obs_dim` (because of sum-over-channels then mean), in ours weight 1. Likely a contributor to why our reward head has been observed under-trained relative to recon (because both are scaled equally by us, and the reward head's output is effectively a 255-way classification while the recon is a 19-way regression — but in sheeprl semantics, recon would scale up by ~19 making it dominate).

#### §9.4.2 Reward loss

- **Sheeprl** (`loss.py:62, dreamer_v3.py:164`): `pr = TwoHotEncodingDistribution(world_model.reward_model(latent_states), dims=1)`; `reward_loss = -pr.log_prob(rewards)` where `log_prob` does `target * log_softmax(logits)` summed over the bin axis (`utils/distribution.py:253–276`). Final `.mean()` at the aggregate (`loss.py:80, 85`).
- **Ours** (§3.5.2, `trainer.py:217–220`): `loss_rew = -jnp.mean(jnp.sum(rew_target * jax.nn.log_softmax(rew_pred), axis=-1))` — sum over bins, mean over batch+time.
- **Classification.** Both match paper. Same numerical result.

#### §9.4.3 Continue loss

- **Sheeprl** (`loss.py:76–77, dreamer_v3.py:167–168`): `pc = Independent(BernoulliSafeMode(logits=continue_model(latent_states)), 1); continues_targets = 1 - terminated; continue_loss = continue_scale_factor * -pc.log_prob(continues_targets)`. `continue_scale_factor = 1.0` in canonical config (`configs/algo/dreamer_v3.yaml:51`). Final `.mean()` at the aggregate.
- **Ours** (§3.5.3, `trainer.py:222–224, 251`): `loss_cont = optax.sigmoid_binary_cross_entropy(cont_pred, 1.0 - terminal[..., None]).mean()`; multiplied by `CONT_LOSS_WEIGHT = config.get_mandatory('agent.cont_loss_weight', float)` (canonical 1.0).
- **Classification.** Both match paper. Both have a configurable `continue_scale_factor` / `cont_loss_weight` knob (default 1.0, both); knob is `EXTENSION` in both implementations. Confirms our §6 §3.5.3 "knob is EXTENSION but defaults match paper" framing.

#### §9.4.4 KL dynamics + KL representation

- **Sheeprl** (`loss.py:64–75, configs/algo/dreamer_v3.yaml:47–49`):
  ```python
  dyn_loss = kl = kl_divergence(
      Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
      Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
  )
  free_nats = torch.full_like(dyn_loss, kl_free_nats)  # 1.0
  dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)            # kl_dynamic = 0.5
  repr_loss = kl_divergence(
      Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
      Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
  )
  repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)   # kl_representation = 0.1
  kl_loss = dyn_loss + repr_loss
  ```
  Then multiplied by `kl_regularizer = 1.0` and added to the aggregate. Per-state aggregation: `Independent(..., 1)` makes `kl_divergence` sum over the rightmost (stoch_dim, classes) axes — i.e. clipping is per `(B, T)` state, after aggregation, identical to ours.
- **Ours** (§3.5.4, `trainer.py:17–23, 226–249`): `FREE_NATS = 1.0`, `DYN_SCALE = 0.5`, `REP_SCALE = 0.1`, `KL_SCALE = 1.0` (declared, never used).
- **Classification.**
  - **`free_nats = 1.0, dyn_scale = 0.5, rep_scale = 0.1`** — matches paper preprint, matches sheeprl. `MATCHES PAPER (preprint)` for both implementations. Confirms §3.5.4.
  - **`kl_regularizer = 1.0` is multiplied through in sheeprl, paper Eq. 4.** Our `KL_SCALE = 1.0` is dead. Both produce the same numerical output (since the value is 1.0). `MINOR DEVIATION (suspected unjustified, ours)`. §6 item 9 confirmed; sheeprl actually uses the constant.
  - **Per-state clipping axis aggregation.** Both sum over class+stoch_group axes BEFORE the `max(., free_nats)` clip. Identical. §3.5.4 / §3.7.5 are correct.
  - **`dyn_scale = 0.5` deviates from Nature.** §6 item 5. Sheeprl matches preprint (0.5), so this is a paper-vs-Nature deviation, not a sheeprl-vs-us deviation. Both implementations target the preprint.

#### §9.4.5 λ-return computation

- **Sheeprl** (`utils.py:66–77`):
  ```python
  def compute_lambda_values(rewards, values, continues, lmbda=0.95):
      vals = [values[-1:]]
      interm = rewards + continues * values * (1 - lmbda)
      for t in reversed(range(len(continues))):
          vals.append(interm[t] + continues[t] * lmbda * vals[-1])
      ret = torch.cat(list(reversed(vals))[:-1])
      return ret
  ```
  Caller (`dreamer_v3.py:251–256`):
  ```python
  lambda_values = compute_lambda_values(
      predicted_rewards[1:],
      predicted_values[1:],
      continues[1:] * cfg.algo.gamma,
      lmbda=cfg.algo.lmbda,
  )
  ```
  where `predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean` — **online critic, NOT target critic**. Note also: `predicted_rewards[1:]` and `predicted_values[1:]` slice off the first imagined step (which corresponds to the start state with the initial action); λ-returns are computed for steps 1..H, with the bootstrap at step H+1 = `predicted_values[-1]`. This indexing matches paper Eq. 11 with `R^lambda_T = v(s_T)`.
- **Ours** (§3.5.5, `trainer.py:28–51, 411–416`): Same recursive structure (one-pass reverse scan, `bootstrap = (1-λ)*v + λ*next_return`, `current = r + c*bootstrap`). **Bootstrap value `v` comes from `target_critic`** at every imagined step, NOT the online critic.
- **Classification.**
  - **Recursion math** — both match paper Eq. 11.
  - **γ folded into `c_t`** — both implementations do this (sheeprl `continues * gamma` at L254; ours `conts * GAMMA` at L416). `MATCHES PAPER` for both.
  - **Bootstrap source: ours uses target critic, sheeprl uses online critic.** `MAJOR DEVIATION (suspected unjustified, ours)`. **CONFIRMS §6 item 3.** Triply-confirmed: paper text + paper code + sheeprl reference all use the online critic.

#### §9.4.6 Critic loss

- **Sheeprl** (`dreamer_v3.py:307–316`):
  ```python
  qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
  predicted_target_values = TwoHotEncodingDistribution(
      target_critic(imagined_trajectories.detach()[:-1]), dims=1
  ).mean
  value_loss = -qv.log_prob(lambda_values.detach())                      # term A: CE on actual lambda_values
  value_loss = value_loss - qv.log_prob(predicted_target_values.detach()) # term B: self-EMA regularisation against target critic
  value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))         # discount-weighted mean
  ```
  Critic loss = **two-hot CE on actual `lambda_values`** + **two-hot CE on `predicted_target_values` (the target critic's value estimate as soft regularisation target)**, both summed, then discount-weighted mean. The second term is paper Eq. 10 self-EMA regularisation.
- **Ours** (§3.5.6, `trainer.py:421–430`): Only term A — `loss_critic = mean(-sum(target_twohot * log_softmax(v_pred_logits), -1) * discount_weights)` where `target_twohot = to_twohot(stop_gradient(lambda_returns))`. The self-EMA regularisation term against `target_critic` is **NOT PRESENT**.
- **Classification.**
  - Term A (CE on `lambda_returns`) — matches paper, matches sheeprl.
  - Term B (self-EMA regularisation against target critic) — `MAJOR DEVIATION (suspected unjustified, ours)`. **NEW §6 candidate**. Already noted in §9.3.9.
  - **Discount weighting** — both implementations use `cumprod(continues * gamma)` along the time axis as a stop-gradient weight. Both match paper.
- **Critic-on-RAW-lambda-returns** (no percentile-norm) — both implementations train critic on raw lambda_returns. `MATCHES PAPER` for both.

#### §9.4.7 Actor loss

- **Sheeprl** (`dreamer_v3.py:262–304`):
  ```python
  policies = actor(imagined_trajectories.detach())[1]
  baseline = predicted_values[:-1]                          # online critic, [0..H]
  offset, invscale = moments(lambda_values, fabric)         # update + read in one call
  normed_lambda_values = (lambda_values - offset) / invscale
  normed_baseline = (baseline - offset) / invscale
  advantage = normed_lambda_values - normed_baseline
  if is_continuous:
      objective = advantage
  else:  # discrete (our path)
      objective = (
          torch.stack([p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                       for p, imgnd_act in zip(policies, ...)], dim=-1).sum(dim=-1)
          * advantage.detach()
      )
  entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
  policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
  ```
  - For discrete: `objective = log_prob(action) * stop_gradient(advantage)` — REINFORCE with normalised advantage, same as ours.
  - **Both sides of the advantage are normalised** — `normed_lambda_values` AND `normed_baseline`. Same as ours.
  - Entropy uses `p.entropy()` from PyTorch's distribution; we compute `-sum(softmax(logits) * log_softmax(logits))` manually — same numerical result.
  - **`ent_coef` (entropy_scale) = 3e-4** in canonical config (`configs/algo/dreamer_v3.yaml:119`). Same as ours.
  - Discount weighting via `discount[:-1]` — `discount = cumprod(continues * gamma) / gamma` per `dreamer_v3.py:260` (so `discount[0] = 1` because of the divide-by-gamma). Ours: `cumprod` of `concat([1, conts[:-1] * GAMMA])` (`trainer.py:421–423`) — same first-element-1 convention, slightly different formulation but equivalent.
- **Ours** (§3.5.7, `trainer.py:432–445`): Same advantage formula, same entropy formula, same discount weighting.
- **Classification.** Both implementations match paper preprint Eq. 11. Both normalise both sides of the advantage (`MATCHES PAPER`, the §6 item 21 "deviation" is algebraically zero). Same `ent_coef = 3e-4`. Confirms §3.5.7.

### §9.5 Buffer + sampling differences

- **Sheeprl** (`data/buffers.py:363–526` `SequentialReplayBuffer`, `:529–...` `EnvIndependentReplayBuffer`):
  - `EnvIndependentReplayBuffer` instantiates **one independent buffer per env**, each of size `buffer_size // num_envs` (`dreamer_v3.py:478`). `add` dispatches per-env data into per-env buffers (`buffers.py:627–654`).
  - `sample(batch_size, sequence_length, n_samples)` (`buffers.py:395–465`) draws `batch_size * n_samples` start indices uniformly over `[0, _pos - sequence_length + 1)` (when not full) or `[0, _pos - sequence_length + 1) ∪ [_pos, buffer_size)` excluding the seam (when full). **Each batch row's start index is uniform over valid sub-sequence starts** — NOT block-aligned. Each row may come from a different env (`_get_samples` picks `env_idxes` per batch row, `buffers.py:483`).
  - **No `prioritize_ends`** in sheeprl's DreamerV3 buffer (the official Hafner code has a `prioritize_ends` knob; sheeprl's port omits it).
  - **No positive-reward sub-buffer**, no recent-window sub-buffer, no mixture sampling. `SequentialReplayBuffer` is the only buffer.
  - `sample_tensors` (`dreamer_v3.py:664`) calls `sample` with `n_samples=per_rank_gradient_steps` — i.e. all `n_samples` batches for one Lightning `train_step` are drawn in a single sample call.
  - Per-env buffer size = `cfg.buffer.size // num_envs = 1_000_000 / num_envs`. So total capacity is `1_000_000` like ours.
  - Sequence length: `cfg.algo.per_rank_sequence_length = 64` (`exp/dreamer_v3.yaml:14`), matches paper.
- **Ours** (§3.6, `trainer.py:917–1017`):
  - **Single env-major buffer** with `sequence_length = 128` block-aligned starts; mixture sampling with positive + recent + uniform sub-pools.
- **Classification.**
  - **Mixture sampling + positive-reward buffer** — `EXTENSION (not in paper, not in sheeprl, ours)`. **CONFIRMS §6 item 1a**. Sheeprl confirms the paper baseline is uniform sub-sequence sampling.
  - **Block-aligned (vs uniform) sub-sequence starts** — `MAJOR DEVIATION (deliberate, ours)`. **CONFIRMS §6 item 1d**. Sheeprl does uniform-over-valid-start indexing, exactly as paper.
  - **`sequence_length = 128` (vs paper 64, sheeprl 64)** — `MAJOR DEVIATION (deliberate, ours)`. **CONFIRMS §6 item 1c**.
  - **Per-env buffer (sheeprl) vs single env-major buffer (ours)** — `FRAMEWORK-ONLY-ish`. Sheeprl splits the buffer per env so each env's trajectory is sequentially stored independently; this enables uniform-over-valid-starts without straddling env-boundaries. Ours keeps env-major adjacency in a single buffer with block-aligned starts to enforce the no-straddle property. Different storage mechanisms, but both prevent cross-env contamination in a single sequence.
  - **`buffer_capacity = 10^6` matches paper preprint, deviates from Nature `5×10^6`** — sheeprl matches preprint (`buffer.size = 1000000` in `exp/dreamer_v3.yaml:25`). Same as ours. §6 item 8 noted Nature deviation; sheeprl confirms preprint baseline.
  - **No `replay_ratio` mechanism difference** — sheeprl uses the same `Ratio` class (`utils/utils.py:259–298`, "Directly taken from Hafner et al. (2023) implementation"), our `Ratio` (`util.py:162–192`) is functionally identical. Both ratchet `int(step * ratio)` minus a saved high-water mark. Default `replay_ratio` in sheeprl `exp/dreamer_v3.yaml:11`: **1** (one gradient step per env step), but the `algo/dreamer_v3*.yaml` set `replay_ratio: 1` as well. **Sheeprl's canonical default is `1`, NOT `0.0625`** — confirming that "1/16" is a per-benchmark-table default, not a universal canonical (matches §2.5b's Hafner Table A.1 spread). Our `0.5` is between sheeprl's `1` and the "DMC mid-band" Hafner reference of `0.0625`.

### §9.6 Numerical-stability tricks

#### §9.6.1 Symlog / symexp

- **Sheeprl** (`utils/utils.py:148–153`): `symlog(x) = sign(x) * log(1 + abs(x))`, `symexp(x) = sign(x) * (exp(abs(x)) - 1)`. Identical to ours (`util.py:6–17`). Both match paper.
- **Application sites in sheeprl**: encoder consumes `symlog(obs)` inside `MLPEncoder.forward` (`agent.py:150`); decoder predicts symlog space and `SymlogDistribution.log_prob` does `(symlog(target) - mode)^2` (`utils/distribution.py:177–185`); `TwoHotEncodingDistribution.log_prob` applies `transfwd = symlog` to incoming targets (`utils/distribution.py:253–254`); `TwoHotEncodingDistribution.mean` returns `symexp(probs · bins)` (`utils/distribution.py:246–247`).
- **Ours**: same set of application sites (§3.7.1). Both match paper.

#### §9.6.2 Two-hot encoding/decoding — **THE LOAD-BEARING DIFFERENCE**

- **Sheeprl** (`utils/distribution.py:224–276`):
  ```python
  class TwoHotEncodingDistribution:
      def __init__(self, logits, dims=0, low=-20, high=20, transfwd=symlog, transbwd=symexp):
          self.bins = torch.linspace(low, high, logits.shape[-1])  # → linspace(-20, +20, 255) IN SYMLOG SPACE
          self.transfwd = transfwd  # symlog
          self.transbwd = transbwd  # symexp
      @property
      def mean(self):
          return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))
          # → symexp(sum(softmax(logits) * linspace(-20, 20)))
      def log_prob(self, x):
          x = self.transfwd(x)  # x = symlog(value)
          # ... bucketize x in self.bins
          target = onehot(below)*weight_below + onehot(above)*weight_above  # in symlog space
          return (target * log_softmax(logits)).sum(dim=self.dims)
  ```
  **`bins = linspace(-20, +20, 255)` IS THE SYMLOG-SPACE GRID DIRECTLY.** Decoded raw-space support: `transbwd(linspace(-20, 20)) = symexp([-20, +20]) ≈ [-4.85·10^8, +4.85·10^8]`. **Matches paper exactly** (Hafner published `embodied/jax/heads.py:87–97` constructs `half = symexp(linspace(-20, 0, ...))` then mirrors).
- **Ours** (`util.py:19–76`):
  ```python
  def to_twohot(x, min_v=-20.0, max_v=20.0, num_buckets=255):
      x = symlog(x)
      bottom = symlog(min_v); top = symlog(max_v)            # bottom = symlog(-20) ≈ -3.044
      x = jnp.clip(x, bottom, top)
      rel = (x - bottom) / (top - bottom) * (num_buckets - 1)
      ...
  def from_twohot(logits, min_v=-20.0, max_v=20.0, num_buckets=255):
      probs = jax.nn.softmax(logits, axis=-1)
      bucket_vals = jnp.linspace(symlog(min_v), symlog(max_v), num_buckets)
      sym_val = jnp.sum(probs * bucket_vals, axis=-1)
      return symexp(sym_val)
  ```
  **`bucket_vals = linspace(symlog(-20), symlog(+20), 255) ≈ linspace(-3.044, +3.044, 255)`.** Decoded raw-space support: `symexp(±3.044) ≈ ±20`. **8 orders of magnitude smaller than paper.**
- **Classification.** **`MAJOR DEVIATION (suspected unjustified, ours)`. CONFIRMS §6 item 2.** Sheeprl confirms the paper convention (`linspace(-20, 20)` IS the symlog-space grid; the symlog-vs-raw confusion is one application of `symlog/symexp`, not two). Our `to_twohot` does `bottom = symlog(min_v)` which **applies `symlog` twice** — once to the input `x` (correct) and once to the range endpoints `±20` (wrong; the endpoints are already in symlog space per the paper convention). This is the underlying mechanism: we are interpreting `min_v=-20, max_v=20` as **raw-space** edges and symlog-ing them to get the symlog-space grid, while the paper interprets `low=-20, high=20` as **symlog-space** edges directly. The bug is one extra `symlog` call on the range constants.
- **Why this matters for our project.** With our reward magnitudes `|r| ≲ few units`, the practical effect is benign (saturation at `±20` raw never triggers). But the deviation is real, the failure mode is silent (numerical saturation rather than a Python error), and any future scaling exercise hitting returns above `±20` would silently truncate. Sheeprl's reference impl confirms that paper-canonical bin centres span 8 orders of magnitude — designed precisely to avoid the saturation our narrow grid would trigger at higher reward magnitudes.

#### §9.6.3 Unimix

- **Sheeprl** (`agent.py:437–449, 839–845`): RSSM `_uniform_mix` and Actor `_uniform_mix` both use `unimix = cfg.algo.unimix = 0.01` (threaded via `agent.py:1063, 1149`). Mechanism: `probs = softmax(logits); probs = (1-unimix)*probs + unimix/K; logits = probs_to_logits(probs)`. Matches paper.
- **Ours** (`util.py:78–110`): `OneHotDist(logits, unimix=0.01)`; `unimix=0.01` is the constructor default. The YAML key `agent.unimix` is **not threaded** (§6 item 10).
- **Classification.** Both produce the same numerical result (0.01). `MINOR DEVIATION (suspected unjustified, ours)` for the dead YAML key — sheeprl threads it, we don't. §6 item 10 confirmed.

#### §9.6.4 Free-nats

- **Sheeprl** (`loss.py:68–74`): `kl_free_nats = 1.0` config (`configs/algo/dreamer_v3.yaml:49`). Applied via `torch.maximum(dyn_loss, free_nats)` AFTER `kl_divergence(Independent(..., 1), Independent(..., 1))` aggregates over (stoch_dim, classes) axes. Per-state clip.
- **Ours**: Per-state clip after `jnp.sum(..., axis=-1)` over both axes (§3.7.5).
- **Classification.** Both match paper. Confirms §3.7.5 / §6 item 26 (KL-floor-interaction note).

#### §9.6.5 Percentile return scaling (`Moments`)

- **Sheeprl** (`utils.py:40–63`):
  ```python
  class Moments(nn.Module):
      def __init__(self, decay=0.99, max_=1e8, percentile_low=0.05, percentile_high=0.95):
          ...
      def forward(self, x, fabric):
          gathered_x = fabric.all_gather(x).float().detach()
          low = torch.quantile(gathered_x, self._percentile_low)
          high = torch.quantile(gathered_x, self._percentile_high)
          self.low = self._decay * self.low + (1 - self._decay) * low      # mutate
          self.high = self._decay * self.high + (1 - self._decay) * high
          invscale = torch.max(1 / self._max, self.high - self.low)
          return self.low.detach(), invscale.detach()                       # read post-update
  ```
  Note `__init__` default `max_ = 1e8` (so `1 / max_ = 1e-8`, i.e. essentially no lower bound on `invscale`); but the **algo config overrides to `max: 1.0`** (`configs/algo/dreamer_v3.yaml:134`), giving the paper's `max(1, S)` clamp.
  Order: **update first, return updated value**. Used in actor loss (`dreamer_v3.py:276–278`) where `offset, invscale = moments(lambda_values, fabric)` produces the values used for normalisation in the same call.
- **Ours** (`util.py:113–159, trainer.py:336–340, 495`):
  - Default `max_ = 1.0` (passed at instantiation `trainer.py:90`); `1 / max_ = 1.0` → `invscale = max(1.0, high - low)`. Matches paper.
  - **Read-then-update split**: snapshot `low, high, invscale` at `trainer.py:336–340` BEFORE `nnx.grad`; call `self.moments.update(lambda_returns)` AFTER the gradient at `trainer.py:495`. **Used moments are stale by one step** relative to the data they were updated on.
- **Classification.**
  - Default `max_` value differs (sheeprl init default `1e8` vs ours `1.0`). **Sheeprl algo config explicitly sets `max: 1.0` to override**, so behaviour matches in practice. Both end up with `max(1, S)` clamp. `MATCHES PAPER` for both at the config level, but **sheeprl's `Moments.__init__` default `1e8` would be a behaviour-changing bug if a caller forgot to override**. Our default `1.0` is safer.
  - **Update-order difference**: sheeprl updates inside `forward` and returns the updated value, used in the same step's gradient. Ours snapshots before grad and updates after. The OOM concern in our comment (`trainer.py:337–338`) is JIT-tracing-specific; sheeprl is eager so it doesn't apply. **`FRAMEWORK-ONLY` deviation in essence**, but the numerical effect is one-step-staleness in our path. The Hafner published code reads `self.low.value` / `self.high.value` THEN calls `update`, matching ours. Sheeprl's update-then-read is technically a sheeprl-side deviation from Hafner; flag for completeness.
- **Note (sheeprl-side anomaly).** `Moments.__init__` initialises `self.low = self.high = torch.zeros(())` (`utils.py:53–54`); in the first forward call `invscale = max(1, 0 - 0) = 1` regardless of EMA, so the first batch is always raw-scale-normalised. Same as our zero-init for `low.value` (and our `high.value = 1.0` init differs slightly: at first call `invscale = max(1, 1 - 0) = 1`, same result). Both safe.

### §9.7 Optimiser + schedule differences

- **Sheeprl** (`dreamer_v3.py:447–459`, `configs/algo/dreamer_v3.yaml:111–143, 156–160`):
  - Three optimisers, one per module group: `world_model`, `actor`, `critic`. Same as ours.
  - **All three use Adam** (`/optim@*.optimizer: adam` in `defaults` at `configs/algo/dreamer_v3.yaml:5–7`).
  - **Learning rates**: `world_model.optimizer.lr = 1e-4`, `actor.optimizer.lr = 8e-5`, `critic.optimizer.lr = 8e-5`.
  - **Adam epsilon**: `world_model.optimizer.eps = 1e-8`, `actor.optimizer.eps = 1e-5`, `critic.optimizer.eps = 1e-5`. **Asymmetric split, identical to ours.**
  - **Weight decay**: 0 (none).
  - **Gradient clipping**: `clip_gradients: 1000.0` (WM), `100.0` (actor), `100.0` (critic). Applied via `fabric.clip_gradients(..., max_norm=..., error_if_nonfinite=False)` (`dreamer_v3.py:194–199, 300–303, 320–325`) which is global-norm clipping. **Matches ours**.
- **Ours** (§3.8.1):
  - Adam, `1e-4` (WM) / `3e-5` (actor) / `3e-5` (critic), `eps = 1e-8 / 1e-5 / 1e-5`, `clip_by_global_norm(1000.0 / 100.0 / 100.0)`.
- **Classification.**
  - **Optimiser kind (Adam) and grad clip (global-norm 1000/100/100)** — match for both. Both follow preprint Table W.1, both deviate from Nature LaProp+AGC (§6 item 7, item 8).
  - **Adam epsilon (`1e-8` WM / `1e-5` AC)** — match for both. Both follow preprint Table W.1.
  - **Actor LR `3e-5` (ours) vs `8e-5` (sheeprl).** **`MAJOR DEVIATION (suspected unjustified, ours)` OR (suspected unjustified, sheeprl)** — depends on which preprint table you look at. The Hafner 2023 preprint Table W.1 specifies `actor lr = 3e-5, critic lr = 3e-5`. Sheeprl's `8e-5` is a sheeprl-side anomaly relative to the preprint, possibly a port from an intermediate version of Hafner's code. **CONFIRMS §3.8.1 / §5.2 actor/critic LR rows match preprint** (ours is correct preprint, sheeprl deviates from preprint upward). **Does NOT change §6** — flag sheeprl's deviation in §9 only.
  - **No LR schedule (warmup, decay) in either** — match.
- **Per-rank target update freq.** Sheeprl `cfg.algo.critic.per_rank_target_network_update_freq = 1, tau = 0.02` (canonical); functionally identical to our per-step `0.98 * target + 0.02 * online` EMA (§3.4.5).

### §9.8 Diagnostics + metrics differences

- **Sheeprl logged metrics** (`utils.py:20–36, dreamer_v3.py:331–352`):
  - `Loss/world_model_loss` — total WM loss (recon+kl+rew+cont aggregate).
  - `Loss/observation_loss`, `Loss/reward_loss`, `Loss/state_loss` (= `kl_loss`), `Loss/continue_loss`.
  - `Loss/policy_loss`, `Loss/value_loss`.
  - `State/kl` — pre-clip mean KL (just the dyn term, not weighted).
  - `State/post_entropy`, `State/prior_entropy` — categorical entropies.
  - `Grads/world_model`, `Grads/actor`, `Grads/critic` — global-norm gradient magnitudes.
  - `Rewards/rew_avg`, `Game/ep_len_avg` — env-side episode metrics.
  - `Params/replay_ratio` — `cumulative_per_rank_gradient_steps * world_size / policy_step` (running average).
  - `Time/sps_train`, `Time/sps_env_interaction`.
- **Our logged metrics** (§4.1, §4.2, §4.3, §4.5):
  - All of the above, plus:
    - `model_reward_mae`, `model_reward_mae_pos`, `model_reward_mae_neg` — pos/neg-masked reward MAE in raw space (project-specific).
    - `model_latent_entropy`, `model_cont_acc` — posterior entropy & continue-head accuracy.
    - `loss_actor_policy`, `loss_actor_entropy` — split actor sub-terms.
    - `mean_return`, `mean_norm_return`, `mean_value`, `mean_advantage`, `value_mae` — actor diagnostics.
    - `imagined_termination_fraction_h8`, `imagined_termination_fraction_h15`, `imagined_first_term_step_mean`, `imagined_term_step_p10/50/90`, `imagined_real_term_step_mean` — the **imagined-rollout probe** (`agent.imagined_rollout_probe = false` canonical; toggled on for diagnostics).
    - `mod_*` — neuromodulation diagnostics (gated by `modulation_enabled`).
    - `Params/positive_buffer_blocks`, `Params/positive_buffer_utilization`, `Params/main_buffer_blocks` — buffer telemetry.
- **Classification.**
  - **Imagined-rollout probe** — `EXTENSION (not in sheeprl)`. §4.3 / §6 item 24. Sheeprl has no equivalent imagined-termination diagnostic.
  - **Pos/neg-masked reward MAE** — `EXTENSION (not in sheeprl)`. Ours, project-specific. Reward-head failure-mode investigation surfaced this.
  - **Per-channel reconstruction MSE** — neither implementation logs this. Sheeprl reports a single `Loss/observation_loss`; ours reports a single `loss_recon`. Future-add candidate (no current proposal).
  - **Modulation diagnostics** — `EXTENSION (not in sheeprl)`. Ours, project-specific.
  - **Pre-clip mean KL** — sheeprl logs `State/kl` (pre-clip mean of dyn KL); we log `loss_dyn_kl, loss_rep_kl` (post-clip means). Sheeprl's pre-clip metric is an under-floor saturation diagnostic our metrics do not give. **Future-add candidate**: add a `model_dyn_kl_preclip` / `model_rep_kl_preclip` to detect whether the free-nats floor is consistently saturating.
  - **Gradient-norm logging** — sheeprl logs `Grads/world_model`, `Grads/actor`, `Grads/critic`. Our codebase logs neither pre- nor post-clip gradient norms in the WM/AC paths. **Future-add candidate** for diagnostic completeness — gradient-norm is a standard health metric. Do NOT queue an experiment off this; flag for the user only.

### §9.9 Config differences (size profiles + canonical defaults)

Sheeprl ships **6 size profiles** for DreamerV3, all defined as overrides of `dreamer_v3.yaml` (the XL canonical):

| Profile | `dense_units` | `mlp_layers` | `cnn_channels_multiplier` | `recurrent_state_size` | `transition.hidden_size` | `representation.hidden_size` |
|---|---|---|---|---|---|---|
| XS | 256 | 1 | 24 | 256 | 256 | 256 |
| S | 512 | 2 | 32 | 512 | 512 | 512 |
| M | 640 | 3 | 48 | 1024 | 640 | 640 |
| L | 768 | 4 | 64 | 2048 | 768 | 768 |
| **XL (canonical)** | **1024** | **5** | **96** | **4096** | **1024** | **1024** |

The exp config `configs/exp/dreamer_v3.yaml` overrides the algo to `dreamer_v3_S` for the canonical Atari run.

Our codebase has **one effective profile**: width `128`, depth `2 hidden layers` for all heads (`encoder_fc_layers=[128,128]`, `decoder_fc_layers=[128,128]`, `reward_fc_layers=[128,128]`, `continue_fc_layers=[128,128]`, `actor_fc_layers=[128,128]`, `critic_fc_layers=[128,128]`), `rssm_deter_dim=512`, `embed_dim=128`. Plus a commented-out XL block at `dreamer_v3.yaml:60–74` that mentions `1024` (dead).

**Comparison.**
- Our `rssm_deter_dim = 512` matches **sheeprl S** (`recurrent_state_size = 512`). Below sheeprl XS (`256`)? No: XS is 256, ours is 512 — between XS and S on the recurrent axis.
- Our `dense_units = 128` (effective MLP width) is **half of sheeprl XS** (256), the smallest profile. **Our codebase is below sheeprl's smallest profile on MLP width.**
- Our `mlp_layers = 2` matches **sheeprl S** (2). Below sheeprl XS (1)? Sheeprl XS has `mlp_layers=1` — ours has 2, so our depth is *one layer more* than XS. This is the only respect in which our active config exceeds any sheeprl profile.
- **Conclusion**: our active config is approximately "**sheeprl XS-with-128-width**" — narrower than XS on width (128 < 256), one layer deeper than XS, slightly above XS on RSSM deter (512 > 256). **No clean correspondence to any sheeprl profile**; smaller than XS overall in the parameter-count sense (XS has roughly `256² × 1 × 7 ≈ 460k` MLP params per head, ours has `128² × 2 × 7 ≈ 230k` — half the per-head count, but with ~2× the head-stack depth).

**Other config-level differences.**
- Sheeprl `unimix = 0.01` threaded through to RSSM and Actor; ours has the same value but the YAML key is dead (§9.6.3, §6 item 10).
- Sheeprl `learnable_initial_recurrent_state: True` (paper-canonical); ours hard zero (§9.3.1).
- Sheeprl `decoupled_rssm: False` (canonical); we have no such option (we are always coupled).
- Sheeprl `hafner_initialization: True` (`configs/algo/dreamer_v3.yaml:41`) applies the Hafner-prescribed init to specific output layers (`agent.py:1170–1180`): `actor.mlp_heads ← uniform(1.0)`, `critic[-1] ← uniform(0.0)` (zero-init of critic output), `transition_model[-1] ← uniform(1.0)`, `representation_model[-1] ← uniform(1.0)`, `reward_model[-1] ← uniform(0.0)` (zero-init of reward output), `continue_model[-1] ← uniform(1.0)`, `mlp_decoder.heads ← uniform(1.0)`. Ours has `hafner_init` (`util.py:195`, scale 0.8796) applied uniformly to all `Linear` kernels (§5.2 row "`hafner_init` scale 0.8796"). **`MAJOR DEVIATION (suspected unjustified, ours) — output-layer-specific zero-init missing**.** **NEW §6 candidate.** Sheeprl's `uniform(0.0)` zero-init of the reward and critic output layers means at init both heads emit a flat (uniform softmax) distribution — a known stabiliser for two-hot heads (Hafner published code does this; sheeprl mirrors it). Our uniform `hafner_init` across all layers does NOT zero-init the reward/critic output, so the initial output distribution is non-uniform. **This is directly relevant to the reward-head localized-failure investigation** (forward-link `.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md`) — under-trained reward head + non-uniform init = harder to escape from.

### §9.10 Differences summary table

Severity legend:
- `MAJOR` = potentially affects training dynamics or loss landscape (>= one-step impact).
- `MINOR` = code-cleanup, dead-key, or algebraically-equivalent restatement; no impact on dynamics.
- `FRAMEWORK-ONLY` = pure framework idiom; zero algorithmic effect.

| Component | Paper | Sheeprl | Ours | Cross-ref to §6 / §3 | Severity |
|---|---|---|---|---|---|
| **GRU layout** | 1 fused linear, 1 LN | 1 fused linear, 1 LN | 2 split linears, 2 LNs (summed) | (new) §9.3.1 | MINOR |
| **GRU `cand` reset gate** | `tanh(reset * cand)` | `tanh(reset * cand)` | `tanh(cand)` (reset gate computed but not applied) | (new) §9.3.1 — **NEW §6 candidate** | MAJOR |
| **GRU `update` `-1` bias** | `sigmoid(x - 1)` (≈ 0.27 at init) | `sigmoid(x - 1)` | `sigmoid(x)` (≈ 0.5 at init) | (new) §9.3.1 | MINOR |
| **GRU pre-MLP** | yes (1 hidden layer) | yes (1 hidden layer) | no (single Linear+SiLU) | (new) §9.3.1 | MINOR |
| **Initial recurrent state** | learnable | learnable (`learnable_initial_recurrent_state: True`) | hard zero | (new) §9.3.1 | MINOR |
| **`is_first` reset semantics** | replace with initial | replace with initial | zero `deter` and `stoch` | §6 item 15 | MINOR |
| **Posterior + prior heads** | 1 hidden MLP each | 1 hidden MLP each (`representation/transition.hidden_size`) | single Linear each (no hidden) | (new) §9.3.2 — **NEW §6 candidate** | MAJOR |
| **Encoder structure (vector obs)** | flat MLP | flat MLP | hierarchical (per-sensor + hub) | §6 item 1, item 4 | EXTENSION |
| **MLP widths** | per-profile (S=512, XL=1024) | per-profile (XS=256 to XL=1024) | 128 (below sheeprl XS=256) | §6 item 4 | MAJOR |
| **Head depth** | 2 hidden layers (S) | 2 hidden layers (S) | 2 hidden layers | matches sheeprl S | n/a |
| **Reward + critic output zero-init** | `uniform(0.0)` zero-init (Hafner published) | `uniform(0.0)` (when `hafner_initialization: True`) | uniform `hafner_init(0.8796)` (NOT zero-init) | (new) §9.9 — **NEW §6 candidate** | MAJOR |
| **Decoder symlog target convention** | symlog-space MSE | symlog-space MSE (`SymlogDistribution(dist="mse")`) | symlog-space MSE | §3.5.1 | n/a |
| **Recon loss aggregation** | sum over event dims, mean over batch | sum-then-mean | mean over all axes | (new) §9.4.1 — **NEW §6 candidate (low-priority)** | MINOR |
| **Reward loss** | two-hot CE | two-hot CE | two-hot CE | matches | n/a |
| **Continue loss** | BCE | BCE (`BernoulliSafeMode.log_prob`) | BCE (`optax.sigmoid_binary_cross_entropy`) | matches | FRAMEWORK-ONLY |
| **`continue_scale_factor` knob** | 1.0 implicit | 1.0 (`continue_scale_factor`) | 1.0 (`cont_loss_weight`) | §6 item — knob is EXTENSION in both | EXTENSION |
| **KL free nats** | per-state, 1 nat | per-state, 1 nat (after `Independent(...,1)` aggregation) | per-state, 1 nat (after `jnp.sum(.., axis=-1)`) | §3.7.5 | n/a |
| **`kl_dynamic, kl_representation`** | 0.5, 0.1 (preprint) | 0.5, 0.1 | 0.5, 0.1 | §3.5.4, §6 item 5 (preprint match, Nature deviation) | MAJOR (vs Nature) |
| **`kl_regularizer` (× outer KL)** | 1.0 | 1.0 (used) | 1.0 (declared as `KL_SCALE`, never applied) | §6 item 9 | MINOR |
| **λ-return recursion** | Eq. 11 | Eq. 11 | Eq. 11 | §3.5.5 | n/a |
| **λ-return bootstrap critic** | online | **online** (`predicted_values = critic(...)`) | **target** (`target_critic(...)`) | §6 item 3 — **CONFIRMED** | MAJOR |
| **Critic loss term A (CE on `lambda_returns`)** | yes | yes | yes | §3.5.6 | n/a |
| **Critic loss term B (self-EMA regularisation against target critic)** | yes (paper Eq. 10) | yes (`-qv.log_prob(predicted_target_values)`) | **NOT PRESENT** | (new) §9.3.9, §9.4.6 — **NEW §6 candidate** | MAJOR |
| **Both-side advantage normalisation** | `(R-v)/scale` | both sides normalised | both sides normalised | §6 item 21 (algebraically equivalent) | n/a |
| **Entropy coefficient** | 3e-4 | 3e-4 | 3e-4 | §3.5.7 | n/a |
| **Replay buffer architecture** | uniform sub-sequence | per-env independent buffers, uniform-over-valid-starts | env-major single buffer, block-aligned starts, **mixture sampling + positive-reward sub-buffer** | §6 item 1a/1d — **CONFIRMED** | EXTENSION |
| **`sequence_length`** | 64 | 64 (`per_rank_sequence_length: 64`) | 128 | §6 item 1c — **CONFIRMED** | MAJOR |
| **`prioritize_ends`** | yes (Hafner published) | NOT IMPLEMENTED | NOT IMPLEMENTED | (sheeprl-and-us deviation from paper) | MINOR (both impls drop it) |
| **`replay_ratio`** | benchmark-dependent (Atari200M=64..Minecraft=16, normalised 0.0156–0.0625) | 1 (canonical sheeprl); per-task overrides | 0.5 (canonical); 0.0625 in `_rr06.yaml` | §6 item 1b — **CONFIRMED**; sheeprl uses still-different default | MAJOR |
| **`Ratio` class** | `embodied/core/when.py` | direct port (`utils/utils.py:259–298`) | direct port (`util.py:162–192`) | §3.8.2 | FRAMEWORK-ONLY |
| **Symlog/symexp** | identical formula | identical | identical | §3.7.1 | n/a |
| **Two-hot bin range (raw-space support)** | `±symexp(20) ≈ ±4.85·10^8` | `±symexp(20) ≈ ±4.85·10^8` (`bins = linspace(-20, 20)` is symlog-space directly) | `±20` (extra `symlog(±20)` applied to range constants → symlog-space `±3.04`) | §6 item 2 — **CONFIRMED, +mechanism identified** | MAJOR |
| **Unimix value** | 0.01 | 0.01 (threaded from config) | 0.01 (constructor default; YAML key dead) | §6 item 10 | MINOR |
| **`Moments` `max_`** | 1.0 (paper `max(1, S)` clamp) | algo config sets `max: 1.0` (init default `1e8` would be unsafe if not overridden) | 1.0 hard-coded | §3.7.4 | n/a (matches via config) |
| **`Moments` update order** | read-then-update (Hafner published) | update-then-read | read-then-update (matches Hafner) | §9.6.5 | MINOR (sheeprl-side anomaly) |
| **Optimiser kind** | Adam (preprint) / LaProp (Nature) | Adam | Adam | §6 item 8 (Nature path missing in both) | MAJOR (vs Nature only) |
| **WM LR** | `1e-4` (preprint) | `1e-4` | `1e-4` | matches | n/a |
| **Actor / critic LR** | `3e-5` (preprint Table W.1) | `8e-5` | `3e-5` | (new) §9.7 — **sheeprl-side deviation from preprint** | MAJOR (sheeprl-side; ours matches paper) |
| **Adam epsilon split** | `1e-8` WM / `1e-5` AC (preprint Table W.1) | `1e-8 / 1e-5 / 1e-5` | `1e-8 / 1e-5 / 1e-5` | §3.7.6 | n/a |
| **Grad clip** | global-norm 1000/100/100 (preprint) | global-norm 1000/100/100 | global-norm 1000/100/100 | §6 item 7 | MAJOR (vs Nature only) |
| **Buffer capacity** | `10^6` (preprint) / `5×10^6` (Nature) | `10^6` | `10^6` | matches preprint | n/a |
| **Per-env vs single-buffer storage** | per-env (Hafner) | per-env (`EnvIndependentReplayBuffer`) | env-major single buffer | (new) §9.5 | FRAMEWORK-ONLY (mostly) |
| **Imagined-rollout probe** | n/a | n/a | 7 metrics | §6 item 24 | EXTENSION |
| **Pos/neg-masked reward MAE** | n/a | n/a | logged | §4.1 | EXTENSION |
| **Modulation block** | n/a | n/a | gated by `modulation.type` | §4.4 | EXTENSION |
| **Pre-clip KL diagnostic (`State/kl`)** | n/a | logged | NOT logged | (new) §9.8 — future-add candidate | MINOR |
| **Gradient-norm metrics** | n/a | logged | NOT logged | (new) §9.8 — future-add candidate | MINOR |
| **Buffer-clearing on stage transitions** | n/a | n/a | yes (`train.py:1116–1136`) | §6 item 20 | EXTENSION |

### §9.11 Findings that update §6

This sub-section originally listed four §6 candidates pending sign-off plus four reframings of existing §6 items. **Sign-off resolution (2026-05-10, user)**: all four candidates accepted into §6; only candidate #4 (output-layer zero-init) is being acted on now — see [`docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`](../../develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md). The other three are documented but deferred. The four candidates have been folded into §6 as items 27–30 (appended, not renumbered, to preserve existing references to "item 2 = twohot range" / "item 3 = SlowTarget critic"); see §6 items 27–30 directly. The reframings (§9.11.5–§9.11.8) and no-action notes (§9.11.9) below are unchanged.

#### §9.11.1–§9.11.4 Folded into §6 items 27–30 (2026-05-10)

| Original sub-section | New §6 item | Severity | Status |
|---|---|---|---|
| §9.11.1 GRU reset gate not applied | §6 item 28 | HIGH (would-rank near item 4) | Documented; deferred (queued behind item 27) |
| §9.11.2 Prior/posterior heads no hidden layer | §6 item 30 | MID | Documented; deferred |
| §9.11.3 Critic self-EMA regularisation missing | §6 item 29 | MID | Documented; deferred (factor into item 3 follow-up) |
| §9.11.4 Reward + critic output not zero-init | §6 item 27 | HIGH (would-rank near item 4) | **In flight** — see [`docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`](../../develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md). Targeted at the reward-head MAE 0.39 finding (Cell A1 NoPred). |

#### §9.11.5 Reframed: §6 item 2 (twohot bin range) — mechanism now identified

- **Old framing** (current §6 item 2): "Two-hot bin range narrowed by 8 orders of magnitude. Code: `util.py:60–76`. Our `bucket_vals = jnp.linspace(symlog(-20), symlog(+20), 255)` creates symlog-space bins in `linspace(-3.045, +3.045)`..."
- **Sheeprl evidence (mechanism)**: Sheeprl `bins = torch.linspace(low, high, ...)` with `low=-20, high=20` — **the literals `-20` and `+20` ARE the symlog-space grid endpoints**, with `transbwd = symexp` applied to map to raw space. Paper does the same. The "extra symlog" call on the range constants in our code (`bottom = symlog(min_v); top = symlog(max_v)` at `util.py:24`) is the underlying mechanism — we apply `symlog` twice (once to the input, once to the range bounds).
- **New framing**: §6 item 2 stays at top-tier; **add to the description**: *"Mechanism: our `to_twohot` applies `symlog` to both the input AND the range constants (`bottom = symlog(min_v)`, `top = symlog(max_v)`), interpreting `min_v=-20, max_v=20` as raw-space edges. Paper convention (and sheeprl) interpret `-20, +20` as symlog-space edges directly — the extra symlog call is the bug. Triply-confirmed by paper text + Hafner published code + sheeprl reference impl."*

#### §9.11.6 Reframed: §6 item 3 (λ-return bootstrap source) — sheeprl confirmation

- **Old framing**: "λ-return bootstrap uses `target_critic`, not online critic." Cited preprint Appendix C item 6 + Nature page 3 + preprint Appendix D.2 ablation.
- **Sheeprl evidence**: `dreamer_v3.py:243–256` — `predicted_values = TwoHotEncodingDistribution(critic(...)).mean` uses online critic. `target_critic` is used only in the critic loss self-regularisation term (`dreamer_v3.py:307–310`). **Triply-confirmed: paper text + paper code + sheeprl reference all use online critic.**
- **New framing**: §6 item 3 stays at top-tier; **add to the description**: *"Triply-confirmed: paper text (preprint App C item 6 + Nature page 3) + Hafner published code + sheeprl `dreamer_v3.py:243–256` all use the online critic for λ-return bootstrap and reserve the slow critic for self-regularisation only."* The follow-up experiment-design hand-off note remains.

#### §9.11.7 Reframed (low-priority): §6 item 1c (sequence_length=128) — confirmed

- **Old framing**: §6 item 1c notes 2× canonical sequence length.
- **Sheeprl evidence**: `configs/exp/dreamer_v3.yaml:14` — `per_rank_sequence_length: 64`. Confirmed.
- **New framing**: No change; sheeprl confirms the paper-canonical 64.

#### §9.11.8 Reframed: §6 item 1b (replay_ratio) — sheeprl uses yet-another default

- **Old framing**: §6 item 1b notes `replay_ratio: 0.5` is 8× the DMC mid-band of `0.0625` (and the §2.5b spread caveat).
- **Sheeprl evidence**: `configs/algo/dreamer_v3.yaml:16` and `configs/exp/dreamer_v3.yaml:11` both set `replay_ratio: 1` for the canonical Atari run. Sheeprl's chosen default is `1`, between Hafner's Atari-200M `64`-step normalised default (≈0.0156 grad/env) and the project's `0.5`. **Confirms that "1/16 = 0.0625" is not the universal default — sheeprl uses a different per-task default for Atari.**
- **New framing**: §6 item 1b stays at top-tier; **add to the description**: *"Sheeprl's canonical Atari default is `1` (one gradient step per env step, see `configs/exp/dreamer_v3.yaml:11`); the per-benchmark Hafner Table A.1 spread plus sheeprl's `1` confirms that 0.0625 was a DMC-specific anchor we have been treating as universal — the actual canonical answer is 'pick by benchmark'."*

#### §9.11.9 No-action items — sheeprl-side deviations not affecting §6

These are sheeprl-side anomalies relative to paper that DO NOT change our §6 list. Logged for reader awareness only.

- **Sheeprl actor / critic LR `8e-5`** vs paper preprint `3e-5`. We are correct; sheeprl deviates upward. (See §9.7.)
- **Sheeprl `Moments.__init__` default `max_ = 1e8`** would be unsafe if a caller forgot the override. Sheeprl always overrides via algo config. Not a behaviour bug today.
- **Sheeprl `Moments` update-then-read** vs Hafner's read-then-update. Numerical effect is one-step difference in moments; sheeprl-side anomaly.
- **`prioritize_ends` is missing from both sheeprl and ours** — Hafner published code has this knob (boost weight on episode endings inside sub-sequence sampling); both ports drop it. Both implementations equally deviate from paper here. Not a sheeprl-vs-us difference.

---


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

The 25-item §6 list became 26 items after the Phase 5 reconciliation: removed 1 false-positive (old item 15 Adam-eps) + added 3 new top-tier items (twohot range, SlowTarget critic, Nature optimiser swap) + added 1 new mid-tier item (LayerNorm vs RMSNorm) + added 1 new mid-tier note (KL-floor interaction) + coalesced 4 separate items into 1 top entry with 4 sub-items. After the 2026-05-10 sheeprl-comparison addition (§9.11), 4 more items were appended as items 27–30, bringing the total to **30 items**. The 4 are appended (not renumbered) so existing references to items by number remain stable; severity rank is shown inline on each.

### Sheeprl-comparison addition (§9, 2026-05-10)

| Reviewer | Doc | Verdict | Headline |
|---|---|---|---|
| senior-developer | (this document) | ADDED-§9 | Sheeprl-comparison addition; sheeprl checked out at `tmp/sheeprl/` (rev `33b6366`); new §9 covers components / losses / buffers / numerics / optimiser / configs. §9.11 listed 4 NEW §6 candidates (GRU reset gate not applied; prior/posterior heads shallow; missing critic self-EMA regularisation term; reward+critic output not zero-init) and 2 reframings (§6 item 2 mechanism identified; §6 item 3 triply-confirmed). |
| senior-developer | (this document) | FOLDED-§9.11→§6 (2026-05-10) | User signed off on all 4 candidates; folded into §6 as items 27–30 (appended, not renumbered). Only item 27 (reward+critic output zero-init) is in flight — see [`docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`](../../develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md). Items 28/29/30 documented but deferred. §6 count moved from 26 → 30. |

---

Verified by: senior-developer (Phase 5 reconciliation, 2026-05-09)
Updated by: senior-developer (Phase 6 sheeprl-comparison addition, 2026-05-10)
