---
title: "dreamer-srl Implementation Plan — Math Review"
topic: dreamer
status: superseded
created: 2026-05-12
last_updated: 2026-05-12
---

> **Superseded by**: [`docs/pi/calls/2026-05-12_dreamer_backend.md`](../../../pi/calls/2026-05-12_dreamer_backend.md) — PI pivot to sheeprl-direct (Option 1) shelves the dreamer-srl plan this review audits.

# dreamer-srl Implementation Plan — Math Review

## Verdict / Plain-language summary

The implementation plan at [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) proposes a from-scratch JAX/Flax (NNX) re-implementation of sheeprl's PyTorch DreamerV3 — every equation, every coefficient, every initialisation rule has to match the sheeprl source bit-identically for the parity gate (mean survival ~500 steps on food-only NoPred) to make sense. The plan is the design contract that the developer agent will translate into code.

This review walks every equation cited in the plan against the sheeprl source under `tmp/sheeprl/sheeprl/...` and against the walkthrough at [`docs/project/references/sheeprl_dreamer_v3/`](../../../project/references/sheeprl_dreamer_v3/). **The plan is largely faithful** — symlog/symexp, two-hot decoding, KL-balanced free-nats world-model loss, GRU reset-gate fix, REINFORCE actor with percentile-normed advantage, two-hot critic NLL with EMA self-regulariser, Polyak target update, and the Hafner truncated-normal init constant `0.87962566103423978` are all correctly described.

However, I found **one mathematical bug in the plan's verification spec** (the two-hot bin grid is described as living in *real* reward space when sheeprl actually stores it in *symlog* space — a developer following the plan's Checkpoint 5 literally would implement it the wrong way), and **five ambiguous-but-not-wrong gaps** where the plan under-describes a load-bearing detail (true-continue substitution at imagination step 0, action masking on `is_first`, discount weighting on both actor and critic losses, the `low`-offset cancellation in the advantage, and the per-element-then-mean order of the free-nats floor).

These are listed below with paper Eq. / sheeprl line citations and an unambiguous correction. None of them touch the algorithm's structure — they are details a careful implementer might catch from the source while ignoring the plan, but a literalist would get wrong. I recommend the plan be amended on the 🔴 item before implementation begins; the 🟡 items can be clarified at checkpoint time.

**Cascade items #2, #27, #29 — verdict:** all three replicated correctly in the plan, with the caveat on #2's verification check below.

**Paper Eqs. 5, 6, 9, 10, 11 — verdict:** all five replicated correctly in the plan, with the caveat that Eq. 11 (actor) and Eq. 10 (critic) under-describe their discount-weighting term.

---

## Equations under review (canonical sheeprl forms)

### Eq. 9 — Symlog / symexp (paper Eq. 9, sheeprl `utils/utils.py:148-153`)

$$\mathrm{symlog}(x) = \mathrm{sign}(x) \cdot \log(1 + |x|)$$
$$\mathrm{symexp}(x) = \mathrm{sign}(x) \cdot (\exp(|x|) - 1)$$

### Two-hot encoding (cascade #2, sheeprl `utils/distribution.py:224-279`)

Bin grid (in **symlog space**, this is the load-bearing point):
$$\mathbf{b} = \mathrm{linspace}(-20, +20, K=255)$$

Encoding of scalar target $y$: push through symlog, find bracketing bins $b_{\text{below}} \le \mathrm{symlog}(y) \le b_{\text{above}}$, assign weights inversely proportional to the symlog-space distance so the expectation matches:
$$w_{\text{below}} = \frac{|b_{\text{above}} - \mathrm{symlog}(y)|}{|b_{\text{above}} - b_{\text{below}}|}, \quad w_{\text{above}} = 1 - w_{\text{below}}$$

Decoding (extracting the scalar from logits): expectation in symlog space, then symexp:
$$\hat{y} = \mathrm{symexp}\!\left(\sum_{k} \mathrm{softmax}(\ell)_k \cdot b_k\right)$$

Cross-entropy log-prob:
$$\log p(y) = \sum_k \text{target}_k \cdot \log\mathrm{softmax}(\ell)_k$$

### Eq. 5 — World-model reconstruction loss (sheeprl `algos/dreamer_v3/loss.py:55-120`)

$$
\mathcal{L}_{\phi} = \mathbb{E}_{q_\phi}\!\left[\;
\underbrace{-\sum_k \log p_\phi(o^k\!\mid\!h,z)}_{\text{obs NLL}}
+ \underbrace{-\log p_\phi(r\!\mid\!h,z)}_{\text{reward NLL}}
+ \underbrace{-\log p_\phi(c\!\mid\!h,z)}_{\text{continue BCE}}
+ \beta_{\text{dyn}}\max\!\left(\mathrm{KL}[\mathrm{sg}(q)\,\|\,p],\,\nu\right)
+ \beta_{\text{rep}}\max\!\left(\mathrm{KL}[q\,\|\,\mathrm{sg}(p)],\,\nu\right)
\;\right]
$$

with $\beta_{\text{dyn}}=0.5$, $\beta_{\text{rep}}=0.1$, $\nu = 1.0$ (free nats), outer mean over $[T, B]$, and the whole KL term scaled by `kl_regularizer=1.0` (line 112). The free-nats floor is **per-element** (over the $[T, B]$ axes) applied before the mean (sheeprl lines 100-101).

### Eq. 6 — Lambda return (sheeprl `algos/dreamer_v3/utils.py:66-77`)

Backward TD($\lambda$) with the discount $\gamma$ pre-multiplied into the continue mask:

$$G_t^\lambda = r_t + \gamma c_t \left[(1-\lambda)\,v_t + \lambda\,G_{t+1}^\lambda\right]$$

with bootstrap $G_T^\lambda = v_T$ from `values[-1:]`. Critically the inner-bracket value is $v_t$ (current step) not $v_{t+1}$ — this is the DreamerV3 paper convention because the reward head $r_t$ is the predicted reward of the *current* latent state. This is mathematically equivalent to the Sutton-Barto textbook form under that indexing convention. Sheeprl's caller passes `predicted_rewards[1:]`, `predicted_values[1:]`, `continues[1:] * gamma`, `lmbda`.

### Eq. 10 — Critic loss (sheeprl `algos/dreamer_v3/dreamer_v3.py:312-318`)

$$\mathcal{L}_{\text{critic}} = -\mathbb{E}\!\left[\, \mathrm{disc}_{[:-1]} \cdot \big( \log p_{q_v}(\mathrm{sg}(G^\lambda)) + \log p_{q_v}(\mathrm{sg}(v^{\text{target}})) \big) \,\right]$$

where $\log p_{q_v}(\cdot)$ is the **two-hot NLL** of the critic (cascade #29: the second term is the EMA self-regulariser, also expressed as a two-hot NLL of the target-critic's predicted value, *not* an MSE). $\mathrm{disc}_t = (\prod_{s\le t} \gamma c_s) / \gamma$ (so $\mathrm{disc}_0=1$), and the gradient must NOT flow through `disc`.

### Eq. 11 — Actor loss (sheeprl `algos/dreamer_v3/dreamer_v3.py:272-304`)

For discrete actions (REINFORCE):

$$\mathcal{L}_{\pi} = -\mathbb{E}\!\left[\, \mathrm{disc}_{[:-1]} \cdot \big( \log \pi(\mathrm{sg}(a)) \cdot \mathrm{sg}(A) + \eta \cdot \mathcal{H}(\pi) \big) \,\right]$$

with $A = \tilde{G}^\lambda - \tilde{v}$ where the tilde denotes percentile-normalisation via `Moments`:

$$\tilde{x} = \frac{x - \text{low}}{\max(1/M, \text{high}-\text{low})}$$

The **`low` offset cancels** when computing $A = \tilde{G}^\lambda - \tilde{v}$, so the *effective* advantage is $A = (G^\lambda - v)/\max(1, \text{high}-\text{low})$ under sheeprl's default `max_=1.0`. Entropy coefficient $\eta = 3 \cdot 10^{-4}$ (sheeprl `dreamer_v3.yaml:119`).

### Polyak EMA target critic (sheeprl `algos/dreamer_v3/dreamer_v3.py:678-680`)

$$\theta_{\text{target}} \leftarrow \tau \, \theta_{\text{source}} + (1-\tau)\,\theta_{\text{target}}, \quad \tau = 0.02$$

with $\tau = 1$ on the very first call (hard copy). Frequency: every `per_rank_target_network_update_freq = 1` gradient step.

### GRU fused-gate (cascade #28, sheeprl `models/models.py:396-403`)

$$x = \mathrm{LN}(W [h_{t-1}\!\parallel\!u_t]) \in \mathbb{R}^{3H}$$
$$(\text{reset},\,\text{cand},\,\text{update}) = \mathrm{chunk}(x, 3)$$
$$\text{reset} = \sigma(\text{reset}), \quad \text{cand} = \tanh(\text{reset} \odot \text{cand}), \quad \text{update} = \sigma(\text{update} - 1)$$
$$h_t = \text{update} \odot \text{cand} + (1 - \text{update}) \odot h_{t-1}$$

The "$-1$" bias on the update gate is the Hafner "keep-old-state by default" trick. Cascade fix #28 is the application of the **reset gate to the candidate input** ($\text{reset} \odot \text{cand}$ inside the tanh), which the plan correctly captures.

### Categorical KL between RSSM posterior/prior (sheeprl `utils/distribution.py:405-407`)

Analytic, via PyTorch's `_kl_categorical_categorical` over each of the 32 independent categoricals, summed via `Independent(..., 1)`:

$$\mathrm{KL}[q \,\|\, p] = \sum_{i=1}^{32} \sum_{c=1}^{32} q_{i,c} \, (\log q_{i,c} - \log p_{i,c})$$

NOT Monte-Carlo estimated.

### Straight-through estimator (sheeprl `utils/distribution.py:398-401`)

PyTorch form (forward = hard one-hot; backward = gradient through softmax probs):
$$\text{ste} = \text{samples} + (\text{probs} - \text{sg}(\text{probs}))$$

JAX-equivalent form (the plan's translation):
$$\text{ste} = \text{probs} + \text{sg}(\text{onehot}(\text{samples}) - \text{probs})$$

Both produce the same forward value (a one-hot vector) and the same backward gradient (chain rule through `probs`). **These are gradient-identical.** Signing off on Risks §4.

---

## Findings table

| # | Severity | Plan location | Sheeprl source / paper | Issue | Correction |
|---|---|---|---|---|---|
| 1 | 🔴 Wrong (verification-spec bug) | Checkpoint 5 (line 478) | `utils/distribution.py:237` | The plan says the bin grid is `symexp(linspace(-20, 20, 255))` and that `bins[0]` should be approximately $-4.85\!\times\!10^8$. In sheeprl, `self.bins = torch.linspace(low, high, K)` — bins live in **symlog space**, so `bins[0] = -20`, `bins[127] = 0`, `bins[-1] = +20`. The `symexp` is applied lazily only in `mean`/`mode` to map the *expected* symlog-space scalar back to real space. A developer implementing `self.bins = symexp(linspace(-20, 20, 255))` will break `log_prob` because that function pushes the target through `symlog` and then searches `self.bins` — which must therefore be in symlog space. | Replace Checkpoint 5's bin-grid check with: **`bins` attribute holds `linspace(-20, 20, 255)` (so `bins[0]=-20`, `bins[127]=0`, `bins[-1]=+20`)**. The **real-space bin centers** are `symexp(bins)` (so `symexp(bins[0]) ≈ -4.85e8`, `symexp(bins[-1]) ≈ +4.85e8`), accessed only through `mean`/`mode` via `transbwd`. Both checks are useful but should target the right attribute. |
| 2 | 🟡 Ambiguous | Plan line 234 (actor / behaviour-learning sub-phase) | `dreamer_v3.py:246-248` | The plan does not mention that the **first-step `continues` value at imagination index 0 is replaced by the observed `(1 - terminated)` value** before being passed to `compute_lambda_values`. Without this substitution, the imagined trajectory's step-0 continue is the *predicted* continue head output, which deviates from sheeprl. | Add explicit step to the actor-phase description: "Before computing lambda values, splice the observed `(1 - data['terminated'])` at index 0 of the predicted-continues tensor, keeping `continues[1:]` from the continue head's mode: `continues = concat([true_continue, predicted_continues[1:]], axis=0)`. Cite `dreamer_v3.py:246-248`." |
| 3 | 🟡 Ambiguous | Plan line 190 (RSSM `dynamic`) | `agent.py:425` | The plan correctly mentions that `is_first` resets posterior and recurrent state, but does NOT call out that `action = (1 - is_first) * action` — i.e. the action also gets zeroed at first-step indices before being fed to the recurrent model. Without this, an in-flight action from the buffer's previous chunk would corrupt the new episode's first step. | Add to the `is_first` handling bullet: "On `is_first == 1`, the **action is also zeroed**: `action = (1 - is_first) * action` (`agent.py:425`). This is in addition to the posterior/recurrent reset. All three substitutions happen before the GRU call." |
| 4 | 🟡 Ambiguous | Plan line 234 (actor sub-phase) | `dreamer_v3.py:297` | The actor REINFORCE objective is described as `objective = log_prob * advantage.detach()` then "Add entropy bonus", but the plan does NOT mention the **discount weighting** $\mathrm{disc}_{[:-1]} \cdot (\text{objective} + \text{entropy}[:-1])$ from sheeprl line 297, nor the `[:-1]` slicing on entropy. Without the discount weighting, the actor sees a uniform-in-time objective that over-weights late imagination steps. | Update actor-phase description: "Policy loss is `-mean(disc[:-1].detach() * (objective + entropy_coef * entropy[:-1]))` per `dreamer_v3.py:297`. Both objective and entropy are sliced to `[:-1]` because the last imagined step has no log_prob target. The discount tensor `disc` is the same `cumprod(continues*gamma)/gamma` used for the critic, gradient-detached." |
| 5 | 🟡 Ambiguous | Plan line 234 (critic sub-phase, also re-stated in `train.py` symbol table) | `dreamer_v3.py:316` | The critic loss is described as two-hot NLL of lambda-target plus EMA-regulariser, but the plan does NOT mention the **discount weighting** `value_loss * discount[:-1].squeeze(-1)` from sheeprl line 316. Without it, the critic regresses uniformly across the imagined horizon rather than discounted-weighted. | Update critic-phase description: "Critic loss is `value_loss = mean( (-qv.log_prob(λ_target.sg) - qv.log_prob(target_critic_value.sg)) * disc[:-1].squeeze(-1) )` per `dreamer_v3.py:314-316`. The discount weighting is **identical to the actor's** (same `disc` tensor, same `[:-1]` slice, same `.detach()`)." |
| 6 | 🟡 Ambiguous | Plan line 234 (advantage normalisation description) | `dreamer_v3.py:276-279` | The plan says "Compute baseline-normed advantage via `Moments`" without writing out the actual formula. Sheeprl does `normed_λ = (λ - offset) / invscale; normed_baseline = (baseline - offset) / invscale; advantage = normed_λ - normed_baseline`. The `offset` (= `Moments.low`) **algebraically cancels** in the advantage subtraction, so the effective formula is $A = (G^\lambda - v) / \max(1, \text{high}-\text{low})$. This is helpful for the implementer to know — without it, they may worry about whether the offset is correctly applied. | Add a line: "Both lambda-values and baseline are normalised by `(x - offset) / invscale` with `offset = Moments.low`, `invscale = max(1, high - low)`. The `low` offset cancels in the subtraction, so the effective advantage is $(G^\lambda - v) / \max(1, \text{high}-\text{low})$ — but the implementation must still apply the per-term normalisation (the cancellation is algebraic, not coded)." |
| 7 | 🟡 Ambiguous | Plan line 219 (free-nats floor wording) | `loss.py:100-106` | The plan writes "`free_nats` floor: `0.5 * max(dyn_loss, 1.0) + 0.1 * max(repr_loss, 1.0)`. The total is mean over `[T, B]`." This is correct **if** `max(...)` is read as element-wise — but the wording is ambiguous. In sheeprl, `dyn_loss` from `kl_divergence` has shape `[T, B]`, and `torch.maximum(dyn_loss, free_nats_tensor_of_same_shape)` is per-element. Then the mean comes later. A literalist might implement `max(mean(dyn_loss), 1.0)` instead. | Add: "**The `max(·, 1.0)` floor is applied per-element of the `[T, B]` KL tensor, NOT to the post-mean scalar.** Sheeprl line 100: `free_nats = torch.full_like(dyn_loss, kl_free_nats)`; line 101: `dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)`. JAX equivalent: `jnp.maximum(dyn_loss, kl_free_nats)` over the `[T, B]` shape, before any reduction." |

---

## Cascade items ✓/✗

| Cascade # | Description | Paper / sheeprl source | Plan replicates correctly? |
|---|---|---|---|
| **#2** | Two-hot bins as `linspace(-20, +20, 255)` in symlog space (bin centers in real space are `symexp(linspace)`) | Hafner et al. 2023 paper §3 "Symlog two-hot loss"; sheeprl `utils/distribution.py:237` | ✓ in the YAML config (`reward_low: -20`, `reward_high: 20`, `reward_bins: 255`), ✓ in the `loss.py` symbol table description. ✗ in the **Checkpoint 5 verification spec** (see finding #1) — the spec misdescribes the `bins` attribute as living in real space rather than symlog space. Symbolic intent is right; numerical check is wrong. |
| **#27** | Zero-init reward + critic output Linears via `uniform_init_weights(0.0) → limit=0 → uniform(-0, 0) = 0` | Sheeprl `algos/dreamer_v3/agent.py:1170-1180` (the `if cfg.algo.hafner_initialization` block), `algos/dreamer_v3/utils.py:170-186` (`uniform_init_weights` factory) | ✓ — plan line 140 names the factory, plan line 197 enumerates each head's `apply(uniform_init_weights(0.0))` or `apply(uniform_init_weights(1.0))` call. Checkpoint 3 verifies `kernel.abs().max() == 0.0` post-init. |
| **#28** | GRU candidate uses reset gate: `cand = tanh(reset * cand_pre)` (NOT applied to recurrence sum) | Sheeprl `models/models.py:399-403` | ✓ — plan line 184 names the formula, Checkpoint 2 verifies post-construction. |
| **#29** | Critic two-hot NLL **plus** `-qv.log_prob(target_critic_value.detach())` EMA self-regulariser | Sheeprl `dreamer_v3.py:314-316` | ✓ on the structural form. ✗ on the **discount weighting** (finding #5): plan does not mention `value_loss * disc[:-1]`. |
| **#30** | RSSM `transition_model` and `representation_model` each carry one hidden layer (`hidden_sizes=[hidden_size]`), NOT bare Linear | Sheeprl `algos/dreamer_v3/agent.py:1021-1051` (build_agent constructs them as `MLP(..., hidden_sizes=[transition_hidden_size])`) | ✓ — plan line 190 names cascade fix #30 and Checkpoint 4 verifies the 2-layer MLP structure (one hidden Dense → LayerNorm → SiLU → terminal Dense). |

---

## Paper equations ✓/✗

| Paper Eq. | What it specifies | Sheeprl source | Plan replicates correctly? |
|---|---|---|---|
| **Eq. 5** (world-model loss) | $\mathcal{L}_\phi = $ obs NLL + reward NLL + continue BCE + $\beta_{\text{dyn}} \max(\mathrm{KL}[\mathrm{sg}(q)\,\|\,p], \nu) + \beta_{\text{rep}} \max(\mathrm{KL}[q\,\|\,\mathrm{sg}(p)], \nu)$ | `algos/dreamer_v3/loss.py:55-120` | ✓ — every coefficient ($\beta_{\text{dyn}}=0.5$, $\beta_{\text{rep}}=0.1$, $\nu=1.0$, `kl_regularizer=1.0`, `continue_scale_factor=1.0`) matches sheeprl exactly. Detach semantics (`sg` on alternating sides) named correctly (plan line 219). Free-nats floor wording slightly ambiguous (finding #7) but the YAML values are right. |
| **Eq. 6** (lambda return) | $G_t^\lambda = r_t + \gamma c_t [(1-\lambda) v_t + \lambda G_{t+1}^\lambda]$ with $G_T^\lambda = v_T$ | `algos/dreamer_v3/utils.py:66-77` | ✓ — plan line 141 explicitly says "JAX equivalent: `jax.lax.scan` with `reverse=True`...emit `interm[t] + continues[t] * lmbda * carry`". The recursion matches sheeprl. Note the per-paper convention `v_t` (not `v_{t+1}`) inside the bracket is correct under DreamerV3's reward-at-current-state convention; the audit checklist's textbook form $v_{t+1}$ is the Sutton-Barto convention, not what sheeprl uses. |
| **Eq. 9** (symlog/symexp) | $\mathrm{symlog}(x) = \mathrm{sign}(x) \log(1+|x|)$; $\mathrm{symexp}$ inverse | `utils/utils.py:148-153` | ✓ — plan line 138 explicit, uses `jnp.sign` + `jnp.log1p` for numerical stability. |
| **Eq. 10** (critic loss) | Two-hot NLL of $G^\lambda$ + EMA-target self-regulariser, discount-weighted | `dreamer_v3.py:312-318` | ✓ on structure (two terms, both NLL of two-hot, both `.detach()` on target). ✗ on **discount weighting** (finding #5). |
| **Eq. 11** (actor loss) | REINFORCE: $-\mathbb{E}[\text{disc} \cdot (\log\pi \cdot \mathrm{sg}(A) + \eta \mathcal{H})]$, $A$ percentile-normalised | `dreamer_v3.py:272-304` | ✓ on REINFORCE form, ✓ on percentile-EMA normalisation via `Moments`. ✗ on **discount weighting and entropy slicing `[:-1]`** (finding #4). |

---

## Derivation appendix: SoftSign of two key transformations

### A. Straight-through estimator equivalence (resolves plan Risks §4)

**Claim**: The JAX form `probs + sg(onehot - probs)` and PyTorch form `samples + (probs - sg(probs))` produce gradient-identical outputs.

**Proof**. Let $s = \mathrm{onehot}$ (hard one-hot sample, no gradient), $p = \mathrm{probs}$ (softmax output, gradient-bearing), $\mathrm{sg}$ = stop_gradient / detach.

Forward (JAX): $\text{ste} = p + \mathrm{sg}(s - p) = p + (s - p) = s$. Hard one-hot, as expected.

Backward (JAX): $\partial \text{ste} / \partial p = \mathbb{1} + \partial \mathrm{sg}(\cdot) / \partial p = \mathbb{1} + 0 = \mathbb{1}$. Gradient flows through $p$ as identity, so chain rule through any downstream loss sees the full $\partial \mathcal{L} / \partial p$.

Forward (PyTorch): $\text{ste} = s + (p - \mathrm{sg}(p)) = s + (p - p) = s$ (numerically; the gradient-shadow $p - \mathrm{sg}(p)$ is 0-valued in forward).

Backward (PyTorch): $\partial \text{ste} / \partial p = 0 + (\mathbb{1} - 0) = \mathbb{1}$. Same.

Both yield forward $= s$ and $\partial / \partial p = \mathbb{1}$, hence gradient-identical. The JAX form is the correct translation of sheeprl's PyTorch form for both (a) RSSM categorical latents and (b) discrete action sampling. **Signed off.**

### B. Advantage offset cancellation

**Claim**: With `offset = Moments.low` applied symmetrically to lambda-values and baseline, the advantage `(λ - offset)/inv - (v - offset)/inv = (λ - v)/inv`.

This is trivially algebraic — $(\lambda - o)/i - (v - o)/i = (\lambda - v)/i$ — but flagging it because the plan's "compute baseline-normed advantage via Moments" is one place the developer might wonder whether `offset` matters for the advantage. It does **not** matter for the advantage, but it does affect the **raw $\tilde{G}^\lambda$** that gets used internally as a critic regression target (after the actor loss, not before). The plan does not say the offset is used for the critic target — and looking at sheeprl `dreamer_v3.py:314` confirms `qv.log_prob(lambda_values.detach())` uses the **un-normalised** `lambda_values`, not `normed_lambda_values`. So `Moments` is used **only** for actor normalisation, and the offset cancels there. The implementer should know both facts.

---

## Numerical-constant audit (all values verified against sheeprl source)

| Constant | Plan location | Sheeprl source | Status |
|---|---|---|---|
| $\gamma = 0.996996996996997$ | YAML line 263 | `configs/algo/dreamer_v3.yaml:11` | ✓ exact |
| $\lambda = 0.95$ | YAML line 264 | `configs/algo/dreamer_v3.yaml:12` | ✓ exact |
| Horizon $H = 15$ | YAML line 265 | `configs/algo/dreamer_v3.yaml:13` | ✓ exact |
| `replay_ratio = 1` | YAML line 266 | `configs/algo/dreamer_v3.yaml:16` | ✓ exact |
| `learning_starts = 1024` | YAML line 267 | `configs/algo/dreamer_v3.yaml:17` | ✓ exact |
| `unimix = 0.01` | YAML line 271 | `configs/algo/dreamer_v3.yaml:40` | ✓ exact |
| `discrete_size = 32`, `stochastic_size = 32` | YAML lines 282-283 | `configs/algo/dreamer_v3.yaml:45-46` | ✓ exact (→ 32 categoricals of 32 classes) |
| $\beta_{\text{dyn}} = 0.5$ | YAML line 284 | `configs/algo/dreamer_v3.yaml:47` | ✓ exact |
| $\beta_{\text{rep}} = 0.1$ | YAML line 285 | `configs/algo/dreamer_v3.yaml:48` | ✓ exact |
| free_nats $\nu = 1.0$ | YAML line 286 | `configs/algo/dreamer_v3.yaml:49` | ✓ exact |
| `kl_regularizer = 1.0` | YAML line 287 | `configs/algo/dreamer_v3.yaml:50` | ✓ exact |
| `continue_scale_factor = 1.0` | YAML line 288 | `configs/algo/dreamer_v3.yaml:51` | ✓ exact |
| World-model clip = 1000.0 | YAML line 289 | `configs/algo/dreamer_v3.yaml:52` | ✓ exact |
| Recurrent state size = 256 (XS) | YAML line 292 | `configs/algo/dreamer_v3_XS.yaml:11` | ✓ exact |
| Transition hidden = 256 (XS, fix #30) | YAML line 293 | `configs/algo/dreamer_v3_XS.yaml:13` | ✓ exact |
| Representation hidden = 256 (XS, fix #30) | YAML line 294 | `configs/algo/dreamer_v3_XS.yaml:15` | ✓ exact |
| Reward bins = 255 | YAML line 295 | `configs/algo/dreamer_v3.yaml:100` | ✓ exact |
| WM lr = 1e-4, eps = 1e-8 | YAML lines 298-299 | `configs/algo/dreamer_v3.yaml:112-113` | ✓ exact |
| Actor ent_coef $\eta = 3 \cdot 10^{-4}$ | YAML line 303 | `configs/algo/dreamer_v3.yaml:119` | ✓ exact |
| Actor clip = 100.0 | YAML line 305 | `configs/algo/dreamer_v3.yaml:127` | ✓ exact |
| Actor lr = 8e-5, eps = 1e-5 | YAML lines 306-307 | `configs/algo/dreamer_v3.yaml:141-142` | ✓ exact |
| Moments decay = 0.99 | YAML line 308 | `configs/algo/dreamer_v3.yaml:133` | ✓ exact |
| Moments max = 1.0 | YAML line 309 | `configs/algo/dreamer_v3.yaml:134` | ✓ exact (overrides class default of 1e8) |
| Moments percentile low/high = 0.05/0.95 | YAML lines 310-311 | `configs/algo/dreamer_v3.yaml:136-137` | ✓ exact |
| Critic bins = 255, tau = 0.02, target_freq = 1, clip = 100.0 | YAML lines 315-318 | `configs/algo/dreamer_v3.yaml:151-154` | ✓ exact |
| Critic lr = 8e-5, eps = 1e-5 | YAML lines 319-320 | `configs/algo/dreamer_v3.yaml:158-159` | ✓ exact |
| Hafner init constant `0.87962566103423978` | Plan line 139 + Risks §5 (line 582) | `algos/dreamer_v3/utils.py:149` and `:159` | ✓ exact — plan explicitly says "Replicate verbatim; do not redo the derivation." |
| LayerNorm eps `1e-3` | YAML line 278 | `configs/algo/dreamer_v3.yaml:34-35` | ✓ exact |

**No arbitrary additions found.** Every coefficient in the plan's YAML traces to a sheeprl line. No "magic numbers" introduced.

---

## Conclusion

DEVIATIONS FOUND — 7 items: **1 🔴 wrong** (Checkpoint 5 misdescribes the two-hot bin grid as living in real space when sheeprl stores it in symlog space — implementer following the checkpoint literally would corrupt `log_prob`), **6 🟡 ambiguous** (true-continue substitution at imagination step 0, action masking on `is_first`, discount weighting on actor loss, discount weighting on critic loss, advantage offset cancellation explanation, free-nats floor element-vs-scalar wording).

Critical (`🔴`) finding #1 should be fixed in the plan **before** the developer begins Checkpoint 5. The other six are clarifications: implementer reading the sheeprl source directly will pick them up correctly, but a strict plan-only reader could miss them.

No coefficient, no equation form, no detach pattern is mathematically wrong in the plan's prose or YAML. The paper-canonical equations (Eqs. 5, 6, 9, 10, 11) and the five cascade items (#2, #27, #28, #29, #30) are correctly named and cited. The straight-through estimator JAX translation in plan Risks §4 is gradient-identical to sheeprl's PyTorch form — **signing off** on that risk.

Recommend: amend Checkpoint 5 (finding #1) before implementation. The 🟡 findings can be addressed by the developer at Checkpoints 6, 7, and 8 — each one will surface during forward-pass comparison.

Reviewed by: math-reviewer
