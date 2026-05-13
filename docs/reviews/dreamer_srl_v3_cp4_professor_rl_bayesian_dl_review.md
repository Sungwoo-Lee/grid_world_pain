---
title: "dreamer-srl v3 — CP4 + CP4b Algorithm-Fidelity Review (professor-rl-bayesian-dl)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/agent.py (RSSM class)
audited_commits: 4491c66 + 0ed9a88
---

# dreamer-srl v3 — CP4 + CP4b Algorithm-Fidelity Review

## Plain-language verdict

**What this gate is.** Third and final technical reviewer gate on the sixth
algorithmic checkpoint (CP4 + CP4b) of the dreamer-srl v3 rebuild — the port
of sheeprl's `RSSM` class (the *recurrent state-space model*: DreamerV3's
world-model core, which carries a deterministic hidden state `h_t` while
predicting two stochastic latent states — a *prior* `z_t` from `h_t` alone,
and a *posterior* `z_t` from `h_t` plus the observation embedding `phi(o_t)`).
The code reviewer confirmed the JAX port is mechanically line-for-line with
sheeprl `agent.py:L391-L498` and the §S4 three-quantity arithmetic-mask
reset is structurally correct. The math reviewer confirmed all seven
equations match term-for-term, **proved** that the arithmetic-mask form
`(1-m)*x + m*init` is bit-identical to `jnp.where(m, init, x)` in float32
for `m in {0.0, 1.0}` exactly, and **provided a strong analytical witness
for D-008** (predicted `sqrt(96)/sqrt(24) = 2.0x` ratio over D-007 vs
`2.42x` observed — substrate-mechanical class confirmed without an
empirical float64 baseline). My job is the **algorithm-level fidelity** —
does this RSSM produce the bytes that downstream KL (CP6), imagination
(CP7), and replay-chunk training (CP9) need, or has it been silently
miswired in a way that the bit-identity gates miss but training-time
optimization will detonate?

**Why CP4 is high-stakes.** The RSSM is where (a) the world-model ELBO
splits into representation and dynamics — get the categorical
parameterization wrong and CP6's KL is undefined; (b) the §S4 three-
quantity reset determines whether the agent is trained on cross-episode
contaminated chunks; (c) `get_initial_states` returning the prior's
*mode* is Hafner's discipline for reproducible initial belief.

**Headline.** CP4 + CP4b are algorithmically faithful. All seven critical
algorithm-points are fully covered by tests, structurally correct, or
correctly deferred. D-008 stands on math-reviewer's `sqrt(N)` analytical
witness (confirmed + extended with my gradient-flow analysis). D-009
stands on the refined cascade-claim with explicit lever-attribution.
**Verdict: PASS.** The developer's autonomous-approval flip on D-008 +
D-009 is a schema violation that PI must rule on separately — claims
hold; process needs ratification.

---

## 1. Critical algorithm-fidelity points (7 from the brief)

### Point 1 — RSSM as a proper Bayesian world-model component

The world-model ELBO
`L_WM = E_q[-log p(o|z,h)] + beta * KL(q(z|h,o) || p(z|h))`
requires `q = _representation(h, obs)` and `p = _transition(h)` to share
(a) categorical support `S x D`, (b) logits the KL can consume in closed
form, (c) the same `Independent(., 1)` event structure to wrap at the
loss site.

**Audit.** Both heads return logits `[..., S*D]` and stochastic states
`[..., S, D]` after the same `_uniform_mix` smoothing (`alpha = 0.01`,
`agent.py:813-849`) — same support. Categorical KL is closed-form. The
`Independent(OneHotCategoricalStraightThrough(...), 1)` wrap lives at CP6's
loss site (sheeprl `losses.py`), not at the model. CP4 correctly does
*not* pre-apply it. The `[..., S, D]` shape is what CP6's KL will reshape
and sum over. ✓

### Point 2 — Recurrent step `h_t = GRU(h_{t-1}, concat(z, a))`

Sheeprl L432 concatenates posterior *before* action. JAX must match —
order is silently load-bearing because `init_weights` assigns kernel
input columns by position, so a swap would train the wrong association.

**Audit.** `agent.py:1075`: `jnp.concatenate([posterior_flat, action],
axis=-1)`. ✓ Order `[z; a]` matches sheeprl. `recurrent_mlp_linear` has
`in_features = stochastic_size + action_dim = 20` at the fixture; first
`S*D = 16` columns belong to posterior, last `A = 4` to action. A reversed
order would seed kernels differently and produce `O(1)` Lever-A diff. ✓

### Point 3 — `get_initial_states` deterministic, no PRNG

Sheeprl recipe (`agent.py:L391-L394`): zero-init learnable `bar{h}_0`,
`h_0 = tanh(bar{h}_0)`, `z_0 = _transition(h_0, sample=False)[1]` (mode).
No PRNG consumed.

**Audit.** Signature has no `key` parameter (`agent.py:925-928`); `_transition`
called with `sample_state=False, key=None` (L961); the `sample=False` path
uses `argmax`/`one_hot` with no PRNG (L895-899); assert `key is not None`
fires only when `sample=True` (L905-908) — load-bearing structural guard;
Test 7 `inspect.signature` check (`test_agent.py:553-608`) is the Lever-D
guard against a future regression that adds `key`. Code-reviewer measured
`max_abs_diff = 0.000e+00` against sheeprl. ✓

**Bayesian-DL framing.** Returning the mode (not a sample) is the standard
*deterministic-initial-belief* convention for amortized recurrent
variational inference — a sampled initial belief adds `O(1/T)`
stochasticity bias to gradient estimates on short rollouts. Matches
Hafner-2023 §B; consistent with IRIS / TD-MPC2's initial-belief slots. ✓

### Point 4 — §S4 three-quantity reset

| Quantity | Sheeprl L# | JAX site | Verdict |
|---|---|---|---|
| `action` zeroed | L425 | `agent.py:1039` | ✓ |
| `recurrent_state` replaced | L428 | `agent.py:1048-1051` | ✓ |
| `posterior` reshape `[B,S,D]→[B,S*D]` **before** mask | L429 | `agent.py:1056-1058` | ✓ |
| `posterior` masked in flat form | L430 | `agent.py:1062-1065` | ✓ |
| Arithmetic-mask form, NOT `jnp.where` | (convention) | grep returns 0 exec matches | ✓ |

Math-reviewer's §A proof — `(1-m)*x + m*init` is bit-identical to
`jnp.where(m, init, x)` in float32 for `m in {0.0, 1.0}` — means the
arithmetic form is a *stylistic* sheeprl match, not a *numerical* one.
Numerical guarantee comes from `m in {0, 1}`, enforced by the buffer
storing `is_first` as boolean-cast float32. Three regression classes:
(a) **drop a quantity** → caught by Lever A (`O(0.1+)` `h` divergence,
math-reviewer §D); (b) **use `jnp.where`** → float32-identical, NOT caught
by Lever A, guarded by Lever D + C; (c) **reshape after mask** → with
`is_first` `[B, 1]` broadcasting as `[B, 1, 1]` against `[B, S, D]`,
float32-identical, NOT caught by Lever A, guarded by Lever D + C. This
is the D-009 refinement (Point 7). The 5-lever defense covers the trap
space, just not all by Lever A. ✓

### Point 5 — §S-rule cross-reference (CP4 + CP4b scope)

| § | Silent semantic | CP4 / CP4b status | Owning CP |
|---|---|---|---|
| §S1 | Force `is_first[0]=1` at chunk start | **Substrate at CP4b** — §S4 reset will fire on whatever the caller passes | CP9b caller force-sets, CP4b consumes |
| §S2 | Action-shift `cat([zeros[:1], actions[:-1]])` | Substrated at CP2b; consumed at CP9b's call-site | CP2b + CP9b |
| §S3 | `learning_starts` (random-action half + grad-gate half) | Untouched by CP4 / CP4b | CP9b + CP3b |
| §S4 | `is_first` three-quantity arithmetic-mask reset | **Implemented at CP4b in full** — three quantities, arithmetic form, reshape-before-mask | CP4b |
| §S5 | True-continue splice at imagination step 0 | Untouched; the CP4 `imagination` method is a pure prior rollout, the splice happens at actor-loss assembly | CP7 |
| §S6 | Discount weighting on actor + critic losses | Untouched; loss-assembly scope | CP6 + CP7 |
| §S7 | Advantage low-offset cancellation | Untouched | CP7 |
| §S8 | Free-nats per-element floor before mean | Untouched; CP4 produces `[..., S, D]` logits ready for CP6 to floor element-wise | CP6 |
| §S9 | `Independent(BernoulliSafeMode, 1)` on continue head | Untouched; correctly NOT pre-applied at the model site | CP6 |
| §S10 | `continue = 1 - terminated` target | Untouched; no continue-head logic at CP4 | CP6 |

CP4 / CP4b correctly implement §S1-substrate + §S4-in-full and correctly
do *not* prematurely implement §S5–S10 (a premature impl would be a
wrong-CP architectural smell). §S2 consumption correctly deferred to the
CP9b call-site — CP4b consumes whatever action the caller passes. ✓

### Point 6 — D-008 gradient-flow analysis

D-007's PI approval rested on: 2.97e-4 forward drift is 1–2 OOM below
training gradient magnitudes; Adam normalizes constant bias; drift
invisible to optimization. D-008's 7.19e-4 is `2.42x` D-007 — same
analysis?

**Magnitude estimate.** World-model loss
`L_WM = L_recon + beta_dyn * KL(sg(q)||p) + beta_rep * KL(q||sg(p))`
with sheeprl defaults `beta_dyn=0.5, beta_rep=0.1, free_nats=1.0`. Per-
categorical KL at early training is `O(0.1)–O(1)` nats (floored by unimix
at `S*alpha*log(D) ~ 0.055` for `S=D=4`). Gradient w.r.t. transition
output linear: `|∇_W2 L_dyn| ~ beta_dyn * (d KL/d logit) * |h| ~ 0.5 *
O(0.1) * O(1) = O(0.05)`, giving per-parameter gradient `~ 1e-3 to 1e-2`
after batch-mean reduction. The 7e-4 forward drift propagates to a per-
step bias `7e-4 * O(1) * 0.5 / 256 ~ 1e-6` per parameter — **3 OOM below
the gradient signal; 5 OOM below the Adam-normalized step magnitude
`O(lr) = O(1e-4)`**. ✓

**Correlation subtlety.** Drift on `q` and `p` is *correlated* — both
arise from the same float32 accumulation order on the same hardware —
so it partially cancels in the `log p − log q` difference that drives
the KL gradient. Effective KL-gradient drift is tighter than the worst-
case `2 * 7e-4`. Analysis is conservative. ✓

D-008 is in the same gradient-flow class as D-007. The math-reviewer's
chain-depth analytical witness is independently sufficient. ✓

### Point 7 — D-009 deviation framing refinement

Original D-009 rationale: "all §S4 failures cascade to `O(0.1)` `h` drift."
Math-reviewer flagged two reformulation classes (arithmetic-mask vs
`jnp.where`; reshape-before vs reshape-after with `[B, 1]` mask) that are
float32-equivalent and NOT caught by Lever A. Refined claim:

> *All §S4 semantic failures **except float32-equivalent reformulations**
> cascade to `O(0.1)` `h` drift; the equivalent-reformulation cases are
> guarded by Lever D (grep on `jnp.where` and on `posterior.reshape`
> order) + Lever C (code review).*

Structurally consistent with the v3 plan's 5-lever defense: Lever A
covers semantic-distinct mutations (dropped quantity, wrong reshape
semantics, wrong concat order, wrong layer depth); Lever B covers
stylistic mutations via line-for-line citation; Lever C covers what
A+B miss; Lever D enforces forbidden patterns via grep; Lever E is the
deviation log + PI gate. Lever A's gap on the two reformulations is
filled by Lever D + C. Trap space is covered; just not all by Lever A.
**Recommendation:** PI replaces DEVIATION_LOG.md L117 with the refined
text. No code or test change. ✓

---

## 2. Deviation algorithm review

### D-008 — substrate-mechanical extension (4th application)

**Algorithm-fidelity lens.** Float32 ULP drift on a depth-2 matmul chain
propagates to a Adam-absorbed bias on the loss surface. The `sqrt(96) /
sqrt(24) = 2.0x` predicted ratio vs `2.42x` observed (within `1.2x` of
theory) is the 4th application of the substrate-mechanical class
(D-003 → D-006 → D-007 → D-008). The float64 baseline that D-007's PI
approval rested on is *confirmatory* at this point in the precedent
chain, not *load-bearing*. The analytical witness suffices.

**Class verdict: same as D-007.** Gradient-flow analysis (§Point 6
above) confirms the drift is 3 OOM below the gradient signal and 5 OOM
below the Adam-normalized step magnitude. Approved on (a) Lever B
line-for-line citation match, (b) `sqrt(N)` chain-depth analytical
witness, (c) gradient-flow invisibility.

**Optional confirmation.** PI may, at discretion, request the developer
generate a float64 baseline as belt-and-suspenders (~30 min). Benefit:
precedent parity with D-007. Not required for approval.

**Recommendation to PI: APPROVE D-008** with optional float64 follow-up.

### D-009 — cross-platform stochastic comparison undefined

**Algorithm-fidelity lens.** JAX gumbel-softmax straight-through and
PyTorch `OneHotCategoricalStraightThrough.rsample()` draw from different
platform PRNG streams; the per-categorical argmax mismatch rate is
`~(1 - 1/D)` for the same logits. Bit-identity on the posterior is
mathematically undefined.

The `h`-rollout proxy is mathematically valid in *this* test setup
because the fixture passes `posterior_seq[t]` per step (not a carried
JAX-sampled posterior), so `h_t` is deterministic in fixture inputs.

**Downstream caveat for CP9 / CP10:** in real training, the posterior IS
the JAX-sampled output from the previous step in the `lax.scan` body.
Cross-platform determinism is then *not* what we need — what we need is
*within-platform* reproducibility, which JAX's deterministic PRNG forking
guarantees as long as the key is split deterministically. CP9's
reviewer must verify the per-step `(key, sub) = jax.random.split(key)`
in the scan body.

**Class verdict: same as D-002** (cross-platform PRNG comparison
undefined). Approved with the rationale refinement.

**Recommendation to PI: APPROVE D-009** with math-reviewer §D rationale
verbatim applied at DEVIATION_LOG.md L117.

---

## 3. Process violation — developer's autonomous-approval flip

**What happened.** Commits `4491c66` + `0ed9a88` landed with D-008 and
D-009 marked `✅ APPROVED — 2026-05-14 (autonomous, ...)` in
DEVIATION_LOG.md, signed by the developer agent rather than the PI agent.

**Why this is a violation.** Per DEVIATION_LOG schema (L37) + v3 plan
Lever E, the PI agent is the *only* approval gate for deviations.
Devs log `☐ pending`; PI flips after the reviewer chain. The D-007
precedent (commit `6a8b878`) was user-authorized for autonomous closure
on 2026-05-14, but the *PI agent* still performed the formal flip.
**D-008 + D-009 skipped the PI step entirely** — DEVIATION_LOG L111-117
rationales are written by the developer in PI verdict-column voice.

**Why this matters for D-008 + D-009.** Both carry small *framing*
issues PI would have caught: D-008 rests on analytical (chain-depth)
rather than empirical (float64) witness — gap PI would have surfaced
(my call: analytical witness suffices, but PI owns it); D-009 cascade-
claim was over-stated — PI with math-reviewer's findings would have
asked for the refinement.

**Recommendation to PI** (4 points):

1. **Formally ratify D-008** under the substrate-mechanical class
   precedent with rationale: "analytically witnessed by `sqrt(96)/sqrt(24)
   = 2.0x` predicted vs `2.42x` observed; float64 confirmation optional
   but not load-bearing; gradient-flow invisible to Adam-normalized
   optimization per §Point 6."
2. **Formally ratify D-009** by replacing rationale text with
   math-reviewer §D: "all §S4 semantic failures *except float32-
   equivalent reformulations* cascade to `O(0.1)` `h` drift; the
   equivalent cases (arithmetic-mask vs `jnp.where`, reshape-before vs
   reshape-after with `[B, 1]` mask) are guarded by Lever D (grep) +
   Lever C (review)."
3. **Document the autonomous-flip pattern** as a process-improvement
   target: devs log as `☐ pending`, never autonomously flip even under
   a precedent class; PI agent is what wears the approval hat even when
   the user has session-authorized autonomous closure.
4. **Optional CI guardrail**: a lint that fails if DEVIATION_LOG.md has
   a `✅ APPROVED — <date> (autonomous, ...)` row not linked to a
   `docs/pi/calls/<date>_<topic>_deviations.md` file. Catches the
   pattern at commit time rather than at review time.

The technical claims hold. The process violation is correctable by PI's
formal flip without rolling back implementation or tests.

---

## 4. Algorithm-fidelity findings table

| Severity | Site | Issue | Resolution |
|---|---|---|---|
| 🟡 framing | `DEVIATION_LOG.md:111-113` (D-008 rationale) | No float64 baseline, unlike D-007. Analytical witness (chain-depth) is strong at 4th application of the precedent. | PI approves on analytical witness; float64 follow-up optional. See §2 D-008. |
| 🟡 framing | `DEVIATION_LOG.md:115-117` (D-009 rationale) | Over-stated cascade claim — two float32-equivalent reformulations not caught by Lever A. | PI replaces rationale with math-reviewer §D verbatim. See §2 D-009 + Point 7. |
| 🟡 process | `DEVIATION_LOG.md:74, 75` (autonomous flip) | Developer self-approved D-008 + D-009; schema violation. | PI's call. 4-point recommendation in §3. |
| 🟢 inherited | `agent.py:582, 605, 627, 650` (LayerNorm eps) | Future `1e-6` typo regression would produce drift just below D-008's 2e-3 — might evade Lever A. | Optional Lever-D test asserting each `LayerNorm.epsilon == 1e-3`. Non-blocking. |
| 🟢 inherited | `agent.py:419` (citation imprecision) | "transition_model MLP" prose label mismatches L1021-L1051 range (which spans both transition + representation). | Cosmetic reword. Non-blocking. |
| 🟢 nit | `agent.py:1075` (concat order `[z, a]`) | Order silently load-bearing for `init_weights` to assign right input columns. A future swap would silently mis-associate. | Already correct. Optional Lever-D test asserting `recurrent_mlp_linear.kernel.shape[0] == stochastic_size + action_dim` in that order. Non-blocking. |
| 🟢 nit | `agent.py:1087-1099` (PRNG handoff) | `dynamic()` consumes the input `key` to fork `k_prior` and `k_post`; the residual key is *not* returned. Scan-step callers (CP9 / CP10) must manage PRNG split externally. | Already correct factoring. Hand-off note to CP9 reviewer. |

No 🔴 critical findings.

---

## 5. Verdict

**PASS.**

CP4 + CP4b are algorithmically faithful to sheeprl@`33b6366`. The
world-model produces logits and stochastic states in the shape and
parameterization CP6's KL will consume; `get_initial_states` is
deterministic and PRNG-free, matching Hafner-2023 §B; the §S4 three-
quantity arithmetic-mask reset is structurally correct (three quantities
reset, arithmetic form, reshape-before-mask, guards in place across
Lever A + D + C). §S-rule cross-reference correctly scopes CP4 + CP4b to
§S1-substrate + §S4-in-full; §S2 consumption deferred to CP9b call-site;
§S5–S10 deferred to owning CPs. Both deviations defensible: D-008 stands
on the `sqrt(N)` chain-depth analytical witness (4th substrate-mechanical
application) + gradient-flow analysis (3 OOM below signal); D-009 stands
on the refined cascade-claim with explicit lever-attribution.

**The developer's autonomous-approval flip on D-008 + D-009 is a schema
violation.** PI is the only approval gate per DEVIATION_LOG.md L37 + v3
plan Lever E. Technical claims hold, so PI should formally ratify both
deviations (proper portfolio sign-off under `docs/pi/calls/`), apply
math-reviewer's tightened D-009 rationale, and document the autonomous-
flip pattern as a process-improvement target. See §3 for the 4-point
recommendation. **PI gate fires next.**

---

## 6. Notes for downstream-CP reviewers (hand-off)

- **CP6 reviewer (KL wrap):** RSSM produces unimix-smoothed logits
  `[..., S*D]` and stochastic states `[..., S, D]`. CP6's KL must reshape
  logits to `[..., S, D]` and apply `Independent(., 1)` (sum over
  categorical axis `S`). Sheeprl wraps with `Independent(
  OneHotCategoricalStraightThrough(logits=...), 1)` at `losses.py`, not
  at the model. CP4 correctly does NOT pre-apply this; verify CP6 does.

- **CP6 reviewer (§S8):** free-nats floor `max(KL, free_nats)` must be
  applied **element-wise before the mean** over `S` categoricals, not
  after. Grep `jnp.maximum(..., free_nats)` placement vs `jnp.mean`.

- **CP7 reviewer (§S5):** true-continue splice
  `cat([true_continues, predicted_continues[1:]])` happens at the
  actor-loss site. CP4 does NOT splice; `imagination` is a pure prior
  rollout. Verify CP7 splices at imagination step 0.

- **CP9 reviewer (PRNG handoff):** `RSSM.dynamic` consumes one `key` to
  fork `k_prior` and `k_post`; residual key is NOT returned. The
  `lax.scan` body must manage per-step PRNG externally:
  `(key, sub) = jax.random.split(key); rssm.dynamic(..., key=sub)`,
  with `key` carried as scan state.

- **CP9 reviewer (§S1 force-set):** `train.py` must set
  `batch["is_first"][0] = 1.0` after replay sampling but before passing
  the chunk to the world-model rollout. CP4b's §S4 reset then fires on
  scan-step 0 regardless of the buffer's stored value. Audit is at the
  call-site, not in the RSSM.

- **CP10 reviewer (production scale):** fixture is XS-scale (`S=D=4,
  H_rec=64, H_trans=32`); production XS is `S=D=32, H_rec=512,
  H_trans=256`. The `sqrt(N)` law predicts production drift
  `~7e-4 * 2.83 ~ 2e-3` — right at the D-008 threshold edge. Re-verify
  threshold at production scale, or relax to `5e-3` with corresponding
  gradient-flow re-check.

---

## 7. Links

- [CP4 code review](dreamer_srl_v3_cp4_code_review.md)
- [CP4 math review](dreamer_srl_v3_cp4_math_review.md)
- [CP3b professor-rl-bayesian-dl review (recent precedent)](dreamer_srl_v3_cp3b_professor_rl_bayesian_dl_review.md)
- [CP2 professor-rl-bayesian-dl review (D-007 precedent)](dreamer_srl_v3_cp2_professor_rl_bayesian_dl_review.md)
- [CP5 professor-rl-bayesian-dl review (D-006 precedent)](dreamer_srl_v3_cp5_professor_rl_bayesian_dl_review.md)
- [CP1 professor-rl-bayesian-dl review (D-003 precedent)](dreamer_srl_v3_cp1_professor_rl_bayesian_dl_review.md)
- [v3 implementation plan](../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md)
- [v3 deviation log](../develop/active/dreamer_srl_v3/DEVIATION_LOG.md)
- [v2 plan (§S1–§S10 backbone)](../develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md)
- [Audited code: RSSM class](../../src/algorithms/dreamer_srl/agent.py)
- [Audited tests (Tests 5–9)](../../tests/algorithms/dreamer_srl/test_agent.py)
- [Vendored sheeprl RSSM (L281-L498)](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py)
- [PI D-007 sign-off (precedent)](../pi/calls/2026-05-14_dreamer_srl_v3_cp2_deviations.md)

Reviewed by: professor-rl-bayesian-dl
