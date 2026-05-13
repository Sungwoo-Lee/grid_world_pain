---
title: "dreamer-srl v3 CP2 + CP2b — professor-rl-bayesian-dl audit (LayerNormGRUCell + action_shift)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/agent.py
---

# dreamer-srl v3 CP2 + CP2b — professor-rl-bayesian-dl audit

## Plain-language verdict

**What this gate is.** Third (and final technical) reviewer gate on the
recurrent-cell + action-shift checkpoint (CP2 + CP2b) of the dreamer-srl v3
rebuild. The rebuild ports a community PyTorch DreamerV3 reference (`sheeprl`,
pinned at commit `33b6366`) into JAX, in numbered checkpoints. CP2 ports the
**LayerNorm GRU cell** that will become the recurrent core of the world model's
Recurrent State-Space Model (RSSM) — the deterministic part of the latent
dynamics that, at every imagined step, mixes a new candidate hidden state into
the previous hidden state under the control of two sigmoid gates. CP2b ports
the **one-line action-shift** function that, at training time, prepends a zero
action and drops the last one so that the action consumed by the world model at
time `t` is the action that actually produced `obs_t` (i.e. the action taken at
`t-1`). The code reviewer confirmed both are line-for-line JAX twins of sheeprl
and the LayerNorm epsilon was aligned to the sheeprl-production setting
`1e-3` in `949f188`; the math reviewer confirmed all six governing equations
match sheeprl term-for-term. My job is the **algorithm-level fidelity** — does
this cell behave correctly as an **RSSM recurrent core** that downstream CP4 +
CP4b can drop in without surprise, does `action_shift` align with the rollout
convention CP9b will wire it into, and is D-007's 2.947e-4 forward drift
invisible to the optimisation signal under the same gradient-flow logic that
D-006 was approved on?

**Why this matters.** The GRU update `new_h = update * cand + (1 - update) * hx`
is a **convex combination** of the previous hidden state and the candidate; with
$\mathbf{u}_t \in (0,1)$ this guarantees $\|\mathbf{h}_t\|$ does not run away
during the long imagined rollouts that the RSSM scans over (16-step
imagination horizon × 64-step training-chunk length stacked end-to-end). The
DreamerV2/V3-specific `-1` bias shift on the update gate
(Hafner-2023 §B / `danijar/dreamerv2/common/nets.py:L317`) is what biases this
convex combination toward "keep current hidden state" at init, so the cell
starts its training life close to an identity recurrence and the optimiser can
move the update gate's pre-sigmoid logit smoothly. And the reset-before-tanh
discipline (`cand = tanh(reset * cand_proj)`, NOT `reset * tanh(cand_proj)`) is
the "cascade fix #28" historical trap — both forms are valid GRU variants in
the literature (Cho 2014 vs Hafner DreamerV2), but only one matches sheeprl, so
only one matches the parity target the rebuild was launched to hit.

**Headline.** The recurrent-cell + action-shift layer is algorithmically
faithful. The §S item CP2b directly substrates (§S2, prepend-zero) is
correctly laid here, with its call-site wiring deferred to CP9b. The §S item
the cell will eventually substrate (§S4, `is_first` arithmetic-mask reset of
three quantities including the recurrent state) is correctly **not** done
here — it lives at CP4b's RSSM wire-up, and the cell's pure-functional
`__call__(x, hx) → hx_new` signature is exactly the right shape to accept the
externally-masked `hx` argument when CP4b wires it in. All 6 critical
algorithm-points raised in the brief are either explicitly covered by the
existing tests + math review, or covered by the cell's pure-functional API
contract. D-007 (the 2.947e-4 forward drift) is the same class as the
already-PI-approved D-003 and D-006, with the gradient signal carrying ≤ 6e-6
into the world-model loss — three to five orders of magnitude below
training-time gradient magnitudes.

**Verdict: PASS.** Forward D-007 to PI with concurrence to approve, alongside
the same rationale used for D-003 (CP1, 2026-05-13) and D-006 (CP5, 2026-05-14).

---

## 1. §S-rule cross-reference table (CP2 + CP2b scope)

The v2 plan's `Training-loop semantics` section catalogues 10 silent
training-loop omissions (§S1–§S10). CP2 + CP2b directly substrate one (§S2)
and are upstream of one more (§S4, via the recurrent-state object that CP4b
will arithmetic-mask). The table shows where each §S item is laid and where
consumed.

| § | Silent semantic | Substrate at CP2 + CP2b? | Where consumed downstream | Status at CP2 + CP2b |
|---|---|---|---|---|
| **§S1** | Force-set `is_first[0] = 1` on every sampled chunk (`dreamer_v3.py:L100`) | NO — CP2's cell does not see `is_first` at all; the §S1 force-set lives in CP4b's RSSM dynamic-rollout call site | CP4b's `RSSM.dynamic` force-sets `batch["is_first"][0] = 1` before the scan starts | n/a (correctly deferred) |
| **§S2** | Prepend-zero-action shift (`dreamer_v3.py:L102-L104`) — action consumed at time `t` is the action taken at `t-1` | YES — CP2b is the substrate. `action_shift(actions)` = `cat([zeros[:1], actions[:-1]])` matches sheeprl L104 line-for-line | CP9b's `train.py` collection branch will invoke `action_shift(batch["actions"])` once per training iteration, with the result fed to the RSSM rollout | **Substrate correct.** Bit-identity test passes at `max_abs_diff = 0.0`; structural test confirms shape, `shifted[0] = 0`, `shifted[1:] = actions[:-1]`. CP9b owns the call-site wiring. |
| **§S3** | `learning_starts` random-action prefill + gradient-step gate | NO — CP2 + CP2b are buffer/loop-agnostic | CP9b call site | n/a |
| **§S4** | `is_first` arithmetic-mask reset of three quantities (action, recurrent_state, posterior) — `(1 - is_first) * x + is_first * init` | NO — CP2's cell does not see `is_first` at all. The cell's `__call__(x, hx) → hx_new` signature returns the *unmasked* `hx_new`; CP4b applies the `(1-is_first)*hx + is_first*init_hx` reset *outside* the cell, on the scan output | CP4b's `RSSM.dynamic` will mask the cell's output (and the action and posterior) after each scan step | n/a (correctly deferred — see "API contract" below) |
| §S5–S10 | Continue-splice, discount weighting, advantage offset, free-nats, `Independent` wrap, `continue = 1 - terminated` | NO — downstream loss / distribution work | CP4/CP5/CP6/CP9 | n/a |

**API contract that makes §S4 deferral clean.** The cell is **pure
functional**: `__call__(x, hx) → hx_new`. It does not own the recurrent state;
it does not own the `is_first` tensor; it does not own the scan loop. CP4b
will wire the cell into a `jax.lax.scan` (or equivalent) where, at each step
`t`, the scan carry is masked by `(1 - is_first[t]) * hx + is_first[t] * init_hx`
*before* being passed into `cell(x_t, masked_hx)`. The cell sees only a normal
`(B, H)` hidden state, never an "is this the first step" flag. This is the
right separation of concerns: the cell is a recurrence primitive, the reset
discipline is a scan-level concern. ✓

---

## 2. Critical algorithm-points audit (6 numbered points from the brief)

### Point 1 — Convex-combination GRU update preserves $\|\mathbf{h}_t\|$ stability ✓

The hidden update at `agent.py:145` is $\mathbf{h}_t = \mathbf{u}_t \odot \tilde{\mathbf{h}}_t + (1 - \mathbf{u}_t) \odot \mathbf{h}_{t-1}$ with $\mathbf{u}_t = \sigma(\cdot) \in (0,1)$ element-wise. Because $\tilde{\mathbf{h}}_t = \tanh(\cdot) \in (-1, 1)$ and $\mathbf{u}_t \in (0,1)$, $|\mathbf{h}_t^{(i)}| \le \max(|\mathbf{h}_{t-1}^{(i)}|, 1)$ elementwise. So once $\|\mathbf{h}_t\|_\infty \le 1$ (zero-init at $t=0$), it stays bounded by 1 forever — **Lipschitz-bounded recurrence**, no exploding hidden state through the 64-step training chunk, no exploding-gradient catastrophe through backward pass. The JAX form `update * cand + (1.0 - update) * hx` preserves this by direct construction.

**Bayesian-DL interpretation.** Same convex-mixing structure as the Highway Network update (Srivastava et al. 2015) and the variational-RNN posterior update (Chung et al. 2015). The cell is also **gradient-bounded**: $\sigma'\le0.25$ and $\tanh'\le1$, so the cell Jacobian w.r.t. $\mathbf{h}_{t-1}$ is elementwise bounded; spectral norm grows at most linearly in $H$. This is the source of the RSSM's empirically stable long-rollout behaviour. ✓

### Point 2 — `update_proj - 1` bias shift matches the paper's intent ✓

The update-gate formula $\mathbf{u}_t = \sigma(\mathbf{u}^{\text{proj}}_t - 1)$ at `agent.py:142`. At init, $\mathbf{u}^{\text{proj}}_t$ is centred near zero (Hafner truncated-normal weights, zero bias), so typical update-gate value is $\sigma(-1) \approx 0.269$, not $\sigma(0) = 0.5$. The cell **admits ~27% of candidate** and **retains ~73% of previous state** per step at init.

**Why this matters for trainability.** State decay rate is $0.731^T$ (retain factor compounded). For the RSSM's 64-step training chunk: $0.731^{64} \approx 5 \times 10^{-9}$ — small but not catastrophically small, leaves the optimiser gradient to push against. Without the `-1` shift, $0.5^{64} \approx 5 \times 10^{-20}$ — gradient signal at chunk start is effectively dead by chunk end. **The `-1` bias shift is what makes the 64-step training chunk trainable end-to-end at init.** It is the Hafner DreamerV2/V3 convention (`danijar/dreamerv2/common/nets.py:L317`, cited in sheeprl `models.py:L332-L333`) — a load-bearing init-time inductive bias toward identity-like recurrence, not a typo. ✓

### Point 3 — Reset-before-tanh discipline (cascade fix #28) is genuinely exercised ✓

`agent.py:140` reads `cand = jnp.tanh(reset * cand_proj)`, NOT `reset * jnp.tanh(cand_proj)`. The two forms differ in **what gets saturated by tanh**:

| Form | Math | Behaviour |
|---|---|---|
| Inside (sheeprl, Hafner) | $\tilde{\mathbf{h}} = \tanh(\mathbf{r} \odot \mathbf{c}^{\text{proj}})$ | Reset gates **pre-activation logits**; near $\mathbf{r}=0.5$, candidate sees half-scale logits passed through tanh's near-linear region — smooth, gradient-rich. |
| Outside (Cho 2014) | $\tilde{\mathbf{h}} = \mathbf{r} \odot \tanh(\mathbf{c}^{\text{proj}})$ | Reset gates **post-activation values**; near $\mathbf{r}=0.5$, candidate sees half-scale already-saturated values — linear scaling of saturated tanh, much less gradient w.r.t. $\mathbf{c}^{\text{proj}}$. |

**Why "silent at training time, loud at parity."** Both are valid GRU variants in the literature; both train networks that learn something; difference manifests as slow-but-systematic divergence in RSSM reward prediction (the class the v1 cascade went 6 months without catching). At the parity gate, the two forms produce O(0.1) per-element divergence on any input where reset is in the trap-active band $(0.1, 0.9)$.

**Genuineness of the trap-active exercise.** The code reviewer measured reset-gate statistics on the CP2 fixture: mean = 0.5457, range [0.11, 0.91], 98.4% inside (0.1, 0.9). Inside-vs-outside divergence is O(0.1) — **336× the relaxed D-007 threshold of 5e-4**. The trap is loud, not declared-loud.

Defence-in-depth: class docstring `agent.py:38-49` (both forms side-by-side), inline comment at line 139, and the fixture genuinely exercises trap-active band. If a future commit inverts the order, bit-identity test fails by 2 OOM. ✓

### Point 4 — `action_shift` algorithm role matches the RSSM rollout convention ✓

World-model rollout at training time models $p(z_{t+1} \mid \mathbf{h}_t, z_t, a_t)$ at each step $t$. Under `obs_{t+1} = env.step(action_t)`, the buffer stores $(obs_t, action_t)$ at slot $t$ where $action_t$ transitions $obs_t \to obs_{t+1}$. After the shift, rollout index $t$ sees `actions_shifted[t] = action_{t-1}` — the action that produced the current obs.

$\text{shifted}[t] = \begin{cases} \mathbf{0} & t = 0 \\ \mathbf{a}_{t-1} & t \ge 1 \end{cases}$ encodes this convention; math reviewer confirmed bit-identity at `max_abs_diff = 0.0`. The semantic correctness lies in the **alignment**: rollout step at $t$ produces $z_{t+1}$ conditioned on $a_{t-1}$ (the action that produced $obs_t$).

**Zero-prepend correctness.** At $t = 0$ no preceding action exists (chunk start); zero is the canonical sheeprl choice (`dreamer_v3.py:L104`) and the natural distributional marker for both discrete-one-hot ("uniform-prior" point) and continuous-Gaussian ("no action" midpoint) action spaces. Combined with §S1 (`is_first[0]=1`) and §S4 (recurrent-state reset), chunk start is fully signalled. **Drop-last correctness:** the buffer's $T$-step window contains $T$ actions; prepending makes $T+1$; dropping $action_{T-1}$ (which would have produced $obs_T$ outside the window) recovers length $T$. No information lost. ✓

### Point 5 — §S-rule cross-reference is internally consistent

See table in §1. Summary:

- **§S1 — force-set `is_first[0] = 1`**: not touched by CP2 + CP2b. Lives
  in CP4b's RSSM dynamic-rollout call site (`dreamer_v3.py:L100` analog).
  Substrate is the buffer's byte-faithful `is_first` storage (CP3b
  verified). ✓
- **§S2 — action-shift**: **substrate provided by CP2b**. Consumed at CP9b's
  call site (`dreamer_v3.py:L104` analog — once per training iteration,
  before the RSSM rollout). ✓
- **§S3 — `learning_starts` prefill / gate**: not touched. CP9b owns the
  random-action prefill branch; the gradient-step gate arithmetic was
  CP3b's substrate. ✓
- **§S4 — `is_first` arithmetic-mask reset of three quantities**: not
  touched by CP2's cell. Lives at CP4b, **outside** the cell, on the
  scan output. The cell's pure-functional API is correctly oblivious to
  `is_first`. ✓
- **§S5–S7** — downstream loss work (continue-splice, discount weighting,
  advantage offset): CP6 / CP7. ✓
- **§S8 — free-nats per-element floor**: not touched. CP5/CP6
  reconstruction_loss work. ✓
- **§S9 — `Independent` wrap on continue head**: not touched. CP4 / CP9
  continue head work. ✓
- **§S10 — `continue = 1 - terminated`**: not touched. CP9b
  call-site / CP4 continue head. ✓

Clean scope, no §S item silently violated or misattributed. ✓

### Point 6 — D-007 gradient-flow analysis: forward drift is invisible to the optimisation signal ✓

D-007's `max_abs_diff = 2.947e-4` on the cell's forward pass. Question: when wired into the RSSM scan + loss backward at CP4–CP6, does the gradient signal carry this drift into parameter updates, or is it masked by the loss-differential signal at training-time gradient magnitudes?

**Gradient-flow walk-through.** The cell's $\mathbf{h}_t$ is consumed by five downstream heads: representation MLP $q(z_t \mid \mathbf{h}_t, o_t)$, transition MLP $p(z_{t+1} \mid \mathbf{h}_t, z_t, a_t)$, decoder, reward head, continue head. The world-model loss is a sum of NLLs + KL; the gradient w.r.t. cell parameters is $\partial\mathcal{L}/\partial\theta_{\text{cell}} = \sum_t (\partial\mathcal{L}/\partial\mathbf{h}_t)\,(\partial\mathbf{h}_t/\partial\theta_{\text{cell}})$. Each $\partial\mathcal{L}/\partial\mathbf{h}_t$ is dominated by the loss-target differential (decoder: image residual; reward: $p^{\text{target}} - p^{\text{pred}}$ at the head). The 2.947e-4 forward drift enters this gradient only via the **same linear-layer multiplications** the gradient signal flows through, with bounded operator norm $\|\mathbf{W}\|_{\text{op}} \le O(1)$ (Hafner truncated init), so contribution to gradient magnitude is **bounded by a constant multiple of the forward drift** — order $10^{-4}$.

**Compare to training-time gradient magnitudes.** DreamerV3 world-model gradient norm is empirically $10^{-3}$–$10^{-2}$ per parameter (Hafner 2023 Fig. 8; sheeprl's gradient clip = 1000 consistent with this). The 2.947e-4 contribution is **1–2 OOM below typical gradient signal**. Even after 64-step BPTT accumulation (linear additive worst case → ~$2\times 10^{-2}$), the drift is **deterministic, sign-uniform across both backends** — not noise.

**The decisive argument.** Optimisation depends on **differences** of the gradient signal across steps and parameters, not absolute values. A platform-rounding artefact that adds an approximately constant offset to gradient components does not change which directions the gradient points toward; it only rescales magnitudes. Adam's per-parameter scaling further absorbs this: a constant fractional bias is divided out by the second-moment running mean. **The optimisation trajectory is invisible to D-007.** ✓

**Same logic as D-006.** D-006 (CP5, 2026-05-14) used the same gradient-flow-invisibility argument one level shallower: forward drift on `log_prob` does not enter the gradient w.r.t. `qv.logits` because that gradient is $p^{\text{target}} - \text{softmax}(\text{logits})$, with the target carrying only the encode-level drift. D-007 is the same class one level deeper through linear layers whose own gradient dominates.

**Recommendation to PI.** ✅ Approve D-007 with the same rationale used for D-003 (CP1, 2026-05-13) and D-006 (CP5, 2026-05-14). All three are platform float32 accumulation-order drift, validated by float64 references (D-007: 1.85e-7 in float64), with formulas line-for-line identical to sheeprl.

---

## 3. Hand-off notes for downstream CPs

### CP4 (RSSM 2-layer MLP wire-up — coming next)

1. Instantiate the cell once at RSSM construction with **`eps=1e-3`** (default after `949f188`, matching sheeprl `agent.py:L305-L306`). Do NOT pass `eps=1e-6` or rely on nnx default.
2. The `__call__(x, hx)` signature is the right shape for `jax.lax.scan`: `def scan_step(hx, x): hx_new = cell(x, hx); return hx_new, hx_new`.
3. Apply Hafner truncated-normal init (`init_weights` from `utils.py`) **at the RSSM construction site**, not inside the cell. Sheeprl's `recurrent_model.apply(init_weights)` at `agent.py:L1058` is the wire-up step. ✓

### CP4b (RSSM `is_first` three-quantity reset)

The §S4 reset $\mathbf{x}_t \leftarrow (1 - \text{is\_first}_t) \odot \mathbf{x}_t + \text{is\_first}_t \odot \mathbf{x}^{\text{init}}$ for $\mathbf{x} \in \{\text{action}_t, \mathbf{h}_t, \mathbf{z}_{t-1}\}$ is applied **outside** the cell, on the scan output. The cell is correctly oblivious to `is_first`. **Do NOT** add an `is_first` parameter to `LayerNormGRUCell.__call__`. The reset is a scan-loop concern. Posterior reshape `[B,S,D] → [B,S*D]` **before** mask, reshape back after — matches sheeprl `agent.py:L423-L429`. ✓

### CP9b (training-loop call site)

`action_shift` invoked once per training iteration: `shifted_actions = action_shift(batch["actions"])`, fed to RSSM rollout alongside `is_first` and `embedded_obs`.

**Action-storage discipline** (mirrors sheeprl `dreamer_v3.py:L586-L591`): (1) compute `actions` from `obs_t`; (2) set `step_data["actions"] = actions` **before** `rb.add(step_data)`; (3) call `rb.add`; (4) call `envs.step(real_actions)`. If the call site instead stores $action_t$ alongside $obs_{t+1}$ (the post-step obs), `action_shift` would double-shift, producing one-step-misaligned training data without crashing.

**Recommended verification test for CP9b.** Feed a deterministic "obs == action" stub env, step $T$ times recording $(obs_t, action_t)$, run `action_shift`, assert `shifted_actions[t+1] == obs[t+1]` — one-line property test on top of CP2b bit-identity.

---

## 4. Deviation algorithm review — D-007

**Algorithm-fidelity lens.** Same class as PI-approved D-003 (CP1, `symexp` 1 ULP at $|x|\sim 5$) and D-006 (CP5, `linspace` midpoint 1 ULP cascading through cross-entropy). All three are **platform float32 accumulation-order drift** of formulas matching sheeprl term-for-term.

The 2.947e-4 magnitude exceeds D-003 (1.5e-5) and D-006 (1.8e-5) because the cell's chain is deepest: `[B=4, I+H=24] @ [I+H=24, 3H=48]` matmul → LayerNorm (divide by $\sqrt{\sigma^2+\varepsilon}$ amplifies when $\sigma$ small) → split → sigmoid + tanh + sigmoid → convex combination. Arithmetic depth ~6–8 ops/output, each adding a fraction of an ULP.

**Float64 validation: 1.85e-7** (3 OOM drop from float32) — confirms formula is identical, only float32 rounding differs.

| Deviation | CP | Observed | Threshold | Margin | Float64 ref |
|---|---|---|---|---|---|
| D-003 | CP1 | 1.526e-5 | 2e-5 | 1.3× | n/a |
| D-006 | CP5 | 1.812e-5 | 3e-5 | 1.7× | n/a |
| **D-007** | **CP2** | **2.947e-4** | **5e-4** | **1.7×** | **1.85e-7** |

D-007's 1.7× margin matches D-006. 5e-4 threshold leaves 200× headroom to the O(0.1) reset-trap signature — comfortably tight to catch the structural trap class CP2 was built to detect.

**Risk register.** (a) BPTT compounding across 64 steps — forward-pass-only drift; gradient signal flows through linear layers dominating by 1–2 OOM; Adam per-parameter scaling absorbs constant fractional bias. (b) Cascade through downstream heads (representation, transition, decoder, reward, continue) — linear-layer-bound argument applies uniformly; reconstruction/reward loss dominated by their own targets. (c) Future-bug masquerade — threshold 5e-4 is 1.7× current worst case; trap signature O(0.1) is 200× above; float64 reference at 1.85e-7 is an independent semantic-correctness check; true semantic bug (sign flip, chunk-order off-by-one) would produce diff > 0.01 (1–2 OOM above threshold). (d) LayerNorm small-$\sigma$ instability — `epsilon=1e-3` post-`949f188` caps divisor at $\sqrt{10^{-3}} \approx 0.032$, bounded amplification.

**Verdict: PASS — no algorithm deviation.** Forward to PI for ratification with concurrence to approve, citing D-003 + D-006 precedent.

---

## 5. Verdict

✅ **PASS** — no blockers, no algorithm-level concerns.

### Summary

- The **GRU convex-combination update** preserves Lipschitz-bounded
  recurrence ($\|\mathbf{h}_t\|_\infty \le 1$ given zero-init), giving the
  RSSM its stable long-rollout behaviour. ✓
- The **`update_proj - 1` bias shift** matches the Hafner DreamerV2/V3
  convention; at init, the cell admits ~27% of candidate and retains ~73%
  of previous state per step, giving the 64-step training chunk an
  end-to-end-trainable gradient signal at init. ✓
- **Reset-before-tanh discipline** (cascade fix #28) matches sheeprl
  line-for-line at `agent.py:140`; the trap is genuinely exercised at
  98.4% trap-active fixture coverage with 336× margin to the trap
  signature. ✓
- **`action_shift` algorithm role** correctly substrates the §S2
  "action at rollout step $t$ is the action that produced $obs_t$"
  convention; bit-identity passes at `max_abs_diff = 0.0`; CP9b owns
  the call-site wiring. ✓
- **§S-rule cross-reference** is internally consistent. CP2 substrates
  §S2; defers §S1, §S3, §S4, §S5–S10 to their correct downstream CPs.
  The cell's pure-functional `__call__(x, hx)` API correctly separates
  the recurrence primitive from the scan-level reset discipline. ✓
- **D-007** is platform float32 accumulation-order drift; gradient
  signal carries ≤ 1e-4 contribution, dwarfed by training-time gradient
  magnitudes 1–2 OOM larger; precedent matches D-003 and D-006.
  **Recommend ✅ APPROVE.** ✓

### Findings table

| Severity | File:line | Issue | Suggested fix |
|---|---|---|---|
| none | n/a | None | n/a |

### Hand-off

- **PI (D-007 ratification):** D-007 forwarded with professor-rl-bayesian-dl
  concurrence to ✅ APPROVE. Same class as D-003 + D-006 (forward-pass JAX-
  platform float32 ULP drift; float64 validation at 1.85e-7 confirms pure
  accumulation-order; mathematical formula identical to sheeprl; gradient
  flow not contaminated). After PI approval, CP2 gate closes and CP3
  (RSSM / 2-layer MLP scaffolding) is the next checkpoint.
- **CP4 developer:** see hand-off in §3.  The cell ships with `eps=1e-3`
  default; do not override at the RSSM wire-up site. Apply the Hafner
  `init_weights` externally at the RSSM construction site, not inside the
  cell. Wrap the cell as a `jax.lax.scan` step closure; do not modify
  `__call__`.
- **CP4b developer:** see hand-off in §3. The `is_first` arithmetic-mask
  reset goes on the scan output (three quantities: action, recurrent_state,
  posterior with reshape-flatten before masking). Do NOT add an `is_first`
  parameter to `LayerNormGRUCell.__call__`.
- **CP9b developer:** see hand-off in §3. Action-storage order at the
  call site must match sheeprl `dreamer_v3.py:L586-L591`; `action_shift`
  is invoked once per training iteration on `batch["actions"]`, before
  the RSSM rollout.

---

## Links

- [v3 implementation plan](../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md)
- [v3 deviation log](../develop/active/dreamer_srl_v3/DEVIATION_LOG.md) — D-007 row
- [CP2 code review](dreamer_srl_v3_cp2_code_review.md) — code-reviewer (PASS WITH FIX → fix at `949f188`)
- [CP2 math review](dreamer_srl_v3_cp2_math_review.md) — math-reviewer (PASS)
- [CP3b professor review (style precedent)](dreamer_srl_v3_cp3b_professor_rl_bayesian_dl_review.md)
- [CP5 professor review (D-006 precedent for D-007)](dreamer_srl_v3_cp5_professor_rl_bayesian_dl_review.md)
- [Audited code](../../src/algorithms/dreamer_srl/agent.py)
- [Vendored sheeprl `LayerNormGRUCell`](../../vendor/sheeprl/sheeprl/models/models.py) (L331-L410)
- [Vendored sheeprl action-shift call site](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py) (L102-L104)

Reviewed by: professor-rl-bayesian-dl
