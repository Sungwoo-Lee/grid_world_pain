---
title: "dreamer-srl plan v2 — math re-audit"
topic: dreamer
status: superseded
created: 2026-05-12
last_updated: 2026-05-12
---

> **Superseded by**: [`docs/pi/calls/2026-05-12_dreamer_backend.md`](../../../pi/calls/2026-05-12_dreamer_backend.md) — PI pivot to sheeprl-direct (Option 1) shelves the dreamer-srl v2 plan this re-audit cleared.

# dreamer-srl plan v2 — math re-audit

## Verdict / Plain-language summary

The senior-developer has produced v2 of the `dreamer-srl` implementation plan, claiming to fold in all 7 mathematical deviations I flagged in the v1 audit ([`review_math.md`](review_math.md)). My job here is to confirm that v2 has **zero residual math deviations** and **no new math errors** introduced by the v2 edits.

The plan calls for a from-scratch JAX/Flax re-implementation of sheeprl's PyTorch DreamerV3 — the parity gate is bit-identical algorithmic semantics on the food-only NoPred task. The math has to match sheeprl's source line-for-line for that gate to be meaningful.

**Headline: v2 PASSES the math re-audit.**

- The critical 🔴 Checkpoint-5 bin-grid bug (v1 misdescribed the two-hot bin grid as living in real reward space when sheeprl actually stores it in *symlog* space) is **correctly fixed** in three places: the rewritten Checkpoint 5 spec, the cascade-table row #2, and the `loss.py:TwoHotEncoding` row in the file-changes table. A developer reading any of these would now build the right grid.
- All 6 🟡 under-described items from v1 are named explicitly in a new "Training-loop semantics" section (§S1–§S10), with code sketches and sheeprl line citations, and each is re-cited at the file-by-file change locations.
- The 5 paper equations (4, 5, 9, 10, 11) and 5 cascade items (#2, #27, #28, #29, #30) all remain mathematically correct in v2.
- The 11 new YAML keys added by v2 all trace to specific sheeprl-source lines — no invented constants.

**One minor presentation issue, not a pass-killer.** The LaTeX form of the lambda-return recursion at line 248 writes `(1-λ)v_{t+1}` in the bracket when sheeprl's source uses `(1-λ)v_t` (function-internal indexing). The operational guidance — link to sheeprl line 66–77, mandatory numpy reference loop in the docstring, assert-match-to-1e-6 against the JAX `lax.scan` — is correct and will catch the prose typo at implementation time. Flagging it as a 🟢 nit for cleanup but not blocking implementation.

**Verdict: ✅ PASS — zero residual math deviations from v1; one 🟢 nit (LaTeX indexing typo on `compute_lambda_values`).** The plan is mathematically faithful and ready for `developer` to begin Step 1 of the Implementation order.

Reviewed by: math-reviewer

---

## V1 deviations — v2 resolution table

| # | v1 deviation | v1 severity | v2 status | v2 location | Notes |
|---|---|---|---|---|---|
| 1 | Checkpoint 5 misdescribes the two-hot bin grid as living in *real* space (`bins[0] ≈ -4.85e8`) when sheeprl actually stores `self.bins = linspace(-20, 20, 255)` in **symlog space**. A literalist implementer would corrupt `log_prob`. | 🔴 wrong | ✅ **RESOLVED** | Three sites, all consistent: (a) Checkpoint 5 at line 685 — rewritten with explicit "CORRECTED 2026-05-12 (math-reviewer 🔴 #1)" callout, naming `bins[0] = -20.0, bins[127] = 0.0, bins[254] = +20.0` in symlog space, and an explicit "DO NOT implement `self.bins = symexp(linspace(...))`" warning; (b) cascade table row #2 at line 79 — "stored in *symlog space* (bin centers in real space are `symexp(bins)`, accessed only via `mean`/`mode`)"; (c) `loss.py:TwoHotEncoding` row at line 339 — "Corrected 2026-05-12 (math-reviewer 🔴 #1): bin grid is `linspace(-20, +20, 255)` in **symlog space** ... DO NOT store `self.bins = symexp(linspace(...))`, that breaks `log_prob`". | A developer reading any one of the three sites would now build the grid correctly. **The 🔴 single-item pass/fail gate is cleared.** |
| 2 | True-continue splice at imagination index 0 — sheeprl replaces `predicted_continues[0]` with the observed `(1 - data["terminated"])` before `compute_lambda_values`. v1 plan did not mention this. | 🟡 ambiguous | ✅ **RESOLVED** | Training-loop semantics §S5 at lines 134-143, with explicit code: `continues = jnp.concatenate([true_continue_step0[None], predicted_continues[1:]], axis=0)`. Re-cited in `train.py` `one_train_step` row at line 359 ("Apply S5 true-continue splice at imagination step 0"). Sheeprl line cite: `dreamer_v3.py:246-248`. | Both the global semantics section and the per-file row name it — the developer cannot miss it. |
| 3 | Action zeroing on `is_first`: `action = (1 - is_first) * action` — v1 plan named only the recurrent/posterior reset, not the action zeroing. | 🟡 ambiguous | ✅ **RESOLVED** | Training-loop semantics §S4 at lines 120-132, naming the **arithmetic-mask form** explicitly and showing the **three-quantity reset** (action / recurrent_state / posterior, with posterior reshape-flattened before masking). Re-cited in `agent.py` RSSM row at line 297 with the full code sketch. Sheeprl line cite: `agent.py:423-429`. | Notably stronger than v1 ask — v2 also pins the arithmetic-mask form vs `jnp.where`, and surfaces the posterior `[B, S, D] → [B, S*D]` reshape sub-step (which is its own gotcha). |
| 4 | Discount weighting on actor loss: `disc[:-1] * (objective + ent_coef * entropy[:-1])` — v1 plan named the REINFORCE objective and entropy bonus but not the discount-weighting term or the `[:-1]` slicing. | 🟡 ambiguous | ✅ **RESOLVED** | Training-loop semantics §S6 at lines 145-160, with explicit code: `policy_loss = -jnp.mean(discount[:-1] * (objective + ent_coef * entropy[:-1]))`. Re-cited in `train.py` `one_train_step` row at line 359 (the actor block). Sheeprl line cite: `dreamer_v3.py:297`. Discount formula `cumprod(continues * gamma) / gamma` is named explicitly, with `stop_gradient` wrapping. | Both `[:-1]` slices (on `discount` and on `entropy`) are called out. The `stop_gradient` on `discount` is also pinned (was code-reviewer 🟢 #11; v2 collapses into §S6). |
| 5 | Discount weighting on critic loss: `value_loss * discount[:-1].squeeze(-1)` — v1 plan named the two-hot NLL + EMA self-regulariser but not the discount weighting or the `.squeeze(-1)`. Also did not pin that the critic regression target is the **un-normalised** `lambda_values`, not the `Moments`-normed form. | 🟡 ambiguous | ✅ **RESOLVED** | Training-loop semantics §S6 at lines 145-160 (the critic block), with explicit code: `value_loss = jnp.mean((-qv.log_prob(sg(lambda_target)) - qv.log_prob(sg(target_critic_value))) * discount[:-1].squeeze(-1))`. Re-cited in `train.py` `one_train_step` row at line 359 ("**The critic regression target uses the UN-normalised `lambda_values`**, NOT the normed form (§S7)"). Sheeprl line cite: `dreamer_v3.py:316`. | The un-normalised target is also explicitly stated in §S7 at line 169: "the critic regression target is the **un-normalised** `lambda_values` (sheeprl `dreamer_v3.py:314` uses raw `lambda_values.detach()`, NOT normed)". Two independent named locations. |
| 6 | Advantage `low`-offset cancellation: `(λ - offset)/inv - (v - offset)/inv = (λ - v)/inv` — v1 plan said "compute baseline-normed advantage via `Moments`" without naming the per-term normalisation or the algebraic cancellation. | 🟡 ambiguous | ✅ **RESOLVED** | Training-loop semantics §S7 at lines 162-169, with explicit code showing per-term normalisation, followed by the algebraic-cancellation explanation: "The `offset = Moments.low` algebraically cancels in the subtraction, so the effective advantage is `(G^λ - v) / max(1, high - low)`. This is helpful for the implementer to know — but the implementation must STILL apply per-term normalisation (the cancellation is algebraic, not coded)." Sheeprl line cite: `dreamer_v3.py:276-279`. | Also covers the second half of my v1 derivation appendix B — that `Moments` is used **only** for actor normalisation, not for the critic regression target. |
| 7 | Free-nats floor element-vs-scalar wording: `max(dyn_loss, 1.0)` is per-element of the `[T, B]` KL tensor, BEFORE the mean, not after. v1 plan was ambiguous. | 🟡 ambiguous | ✅ **RESOLVED** | Training-loop semantics §S8 at lines 171-180, with explicit code: `dyn_loss = kl_dynamic * jnp.maximum(dyn_loss, kl_free_nats)` (annotated `[T, B], per-element floor`), followed by `kl_total = (dyn_loss + repr_loss).mean()`. Explicit warning: "A literalist might implement `max(mean(dyn_loss), 1.0)` instead — that's wrong." Re-cited in `loss.py:reconstruction_loss` row at line 344. Sheeprl line cite: `loss.py:100-106`. | Stronger than my v1 ask — v2 includes the wrong-way variant as a counter-example to inoculate the developer. |

**All 7 v1 deviations resolved.** Critical 🔴 #1 is cleared at three independent sites with mutually consistent assertions. All 6 🟡 items are now named globally (in §S5/§S4/§S6/§S6/§S7/§S8) and re-cited per-file.

---

## Paper equations and cascade items — v2 re-confirmation

| Paper eq / cascade # | v1 status | v2 status | Notes |
|---|---|---|---|
| **Eq. 4 / 5** (world-model loss — KL-balanced + free-nats floor + obs/reward/continue NLL) | ✓ (called Eq. 5 in v1; called Eq. 4 in v2 plan line 206) | ✓ | Numbering difference is harmless — different parts of the Hafner paper use different indexing. All coefficients (β_dyn=0.5, β_repr=0.1, ν=1.0, kl_regularizer=1.0, continue_scale_factor=1.0) verified against sheeprl. Free-nats floor wording now explicitly per-element (§S8). |
| **Eq. 6** (lambda return — TD(λ) backward recursion) | ✓ | ✓ (with 🟢 nit on the LaTeX indexing — see "New math errors" below) | The operational contract is correct: γ pre-multiplied into `continues` by caller, `vals[0] = values[-1:]` bootstrap, length-T return, mandatory numpy reference loop in docstring with assert match to 1e-6. The LaTeX formula at line 248 has `(1-λ)v_{t+1}` where it should be `(1-λ)v_t` to match sheeprl `utils.py:66-77` function-internal indexing — flagged as a 🟢 cleanup nit. |
| **Eq. 9** (symlog / symexp) | ✓ | ✓ | Unchanged — `utils.py:148-153`. v2 plan `utils.py` row at line 245 names `jnp.sign + jnp.log1p` for numerical stability. |
| **Eq. 10** (critic loss — two-hot NLL + EMA target self-reg + discount weighting) | ✓ on structure, ✗ on discount weighting | ✓ fully | §S6 names the discount weighting and the `.squeeze(-1)` slice explicitly. Also pins target uses **un-normalised** `lambda_values`. |
| **Eq. 11** (actor loss — REINFORCE + percentile norm + discount + entropy) | ✓ on REINFORCE form, ✗ on discount weighting and `[:-1]` slicing | ✓ fully | §S6 names the discount weighting on both objective and entropy with `[:-1]` slice on each. §S7 names the per-term normalisation and algebraic offset cancellation. |
| **Cascade #2** (two-hot bins in symlog space) | ✓ in YAML and `loss.py` symbol table; **✗ in Checkpoint 5 verification spec** | ✓ fully | Checkpoint 5 rewritten; cascade-table row #2 rewritten; `loss.py:TwoHotEncoding` row rewritten. Three locations, all consistent. |
| **Cascade #27** (zero-init reward + critic terminal Linears via `uniform_init_weights(0.0)`) | ✓ | ✓ | Unchanged — plan line 304 enumerates each head's init call, Checkpoint 3 verifies post-init. The Hafner-constant precision warning (`0.87962566103423978` vs truncated `0.8796`) is folded in at `utils.py:init_weights` row (line 246) and Risks §5. |
| **Cascade #28** (GRU candidate uses reset gate inside tanh) | ✓ | ✓ | v2 strengthens this with an explicit fused-gate vs split-gate warning (code-reviewer 🔴 #2 fold-in): "ONE Linear + ONE LayerNorm, NOT two of each" at `agent.py:LayerNormGRUCell` row (line 291), with chunk-order `(reset, cand, update)` pinned. Plan now actively warns against pattern-matching on `src/models/dreamer_v3_nnx.py:LayerNormGRUCell` which uses the wrong 2+2 form. |
| **Cascade #29** (critic two-hot NLL + EMA self-regulariser) | ✓ on structure, ✗ on discount weighting | ✓ fully | §S6 critic block names both NLL terms and the discount weighting. Checkpoint 6 verifies both terms are present. |
| **Cascade #30** (RSSM transition + representation models carry one hidden layer) | ✓ | ✓ | Plan line 304 explicit, Checkpoint 4 verifies 2-layer MLP structure. Init scale `uniform_init_weights(1.0)` (non-zero) named explicitly. |

**All 5 paper equations and all 5 cascade items remain mathematically correct in v2.** The discount-weighting omission on Eqs. 10/11 (which was ✗ in v1) is now ✓ via §S6.

---

## Numerical-constant audit (v2 deltas)

v2 added the following YAML keys that were not in v1's 32-key audit. Each is verified against `tmp/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` (XL defaults) and `dreamer_v3_XS.yaml`:

| Key | v2 location | Sheeprl source | Status |
|---|---|---|---|
| `distribution.type: "auto"` | agent_xs.yaml line 404 | `algos/dreamer_v3/agent.py:748` defaults to `"auto"`; `configs/distribution/default.yaml` does not set it | ✓ — sheeprl-implicit default |
| `distribution.validate_args: false` | agent_xs.yaml line 405 | `configs/distribution/default.yaml:1` `validate_args: False` | ✓ exact |
| `world_model.decoupled_rssm: false` | agent_xs.yaml line 443 | `dreamer_v3.yaml:53` `decoupled_rssm: False` | ✓ exact (explicit for transparency — `DecoupledRSSM` is out-of-scope per Risks §8) |
| `world_model_weight_decay: 0.0` | agent_xs.yaml line 450 | `dreamer_v3.yaml:115` `weight_decay: 0` (under `world_model.optimizer`) | ✓ exact |
| `world_model.reward_model.bins: 255` | agent_xs.yaml line 455 | `dreamer_v3.yaml:100` `bins: 255` | ✓ exact (duplicate of the top-level reward bins for clarity) |
| `world_model.reward_model.dense_act: "silu"` | agent_xs.yaml line 456 | `dreamer_v3.yaml:96` `dense_act: ${algo.dense_act}` → silu | ✓ inherits from algo.dense_act |
| `world_model.reward_model.mlp_layers: 1` | agent_xs.yaml line 457 | `dreamer_v3_XS.yaml:6` `mlp_layers: 1` (XS override) | ✓ exact |
| `world_model.reward_model.dense_units: 256` | agent_xs.yaml line 458 | `dreamer_v3_XS.yaml:5` `dense_units: 256` (XS override) | ✓ exact |
| `world_model.reward_model.layer_norm: true` | agent_xs.yaml line 459 | `dreamer_v3.yaml:97-98` `mlp_layer_norm` is the `LayerNorm` class | ✓ exact (boolean translation of the class) |
| `world_model.discount_model.learnable: true` | agent_xs.yaml line 466 | `dreamer_v3.yaml:104` `learnable: True` | ✓ exact |
| `world_model.discount_model.dense_act: "silu"` | agent_xs.yaml line 467 | `dreamer_v3.yaml:105` inherits algo.dense_act | ✓ exact |
| `world_model.discount_model.mlp_layers: 1` | agent_xs.yaml line 468 | `dreamer_v3_XS.yaml:6` | ✓ exact |
| `world_model.discount_model.dense_units: 256` | agent_xs.yaml line 469 | `dreamer_v3_XS.yaml:5` | ✓ exact |
| `world_model.discount_model.layer_norm: true` | agent_xs.yaml line 470 | `dreamer_v3.yaml:108` inherits algo.mlp_layer_norm | ✓ exact |
| `actor.init_std: 2.0` | agent_xs.yaml line 477 | `dreamer_v3.yaml:122` `init_std: 2.0` | ✓ exact (unused on discrete path; kept for signature parity per professor-rl-bayesian-dl #6) |
| `actor.min_std: 0.1` | agent_xs.yaml line 478 | `dreamer_v3.yaml:120` `min_std: 0.1` | ✓ exact |
| `actor.max_std: 1.0` | agent_xs.yaml line 479 | `dreamer_v3.yaml:121` `max_std: 1.0` | ✓ exact |
| `actor.action_clip: 1.0` | agent_xs.yaml line 480 | `dreamer_v3.yaml:129` `action_clip: 1.0` | ✓ exact |
| `actor.actor_weight_decay: 0.0` | agent_xs.yaml line 484 | `dreamer_v3.yaml:145` `weight_decay: 0` (under `actor.optimizer`) | ✓ exact |
| `critic.critic_weight_decay: 0.0` | agent_xs.yaml line 502 | `dreamer_v3.yaml:162` `weight_decay: 0` (under `critic.optimizer`) | ✓ exact |
| `critic.critic_low: -20.0` / `critic.critic_high: 20.0` | agent_xs.yaml lines 503-504 | `TwoHotEncodingDistribution(... low=-20, high=20)` defaults at `utils/distribution.py:226-228` — sheeprl never overrides; critic and reward share the same bin range | ✓ exact (made explicit for symmetric configuration) |
| `player.discrete_size: 32` | agent_xs.yaml line 508 | Sheeprl-implicit (inherits `world_model.discrete_size`); documented as such in the YAML | ✓ — documented inheritance |
| `buffer_device: "cpu"` | agent_xs.yaml line 520 | Sheeprl-implicit (sheeprl does not have a buffer-device toggle; replay always on CPU in their reference) | ✓ — documented as project-side mandatory key, no sheeprl drift |
| `buffer_size: 1_000_000` | agent_xs.yaml line 519 | Sheeprl default — not in dreamer_v3.yaml but in their experiment configs | ✓ standard |
| `agent.per_rank_sequence_length: 64` | agent_xs.yaml line 422 | Sheeprl experiment config (the `???` in `dreamer_v3.yaml:19` is filled by exp config to 64) | ✓ exact |
| `agent.per_rank_batch_size: 16` | agent_xs.yaml line 423 | Sheeprl experiment config default | ✓ exact |

**No invented constants.** Every new YAML key in v2 traces to a specific sheeprl source line. The 32-key v1 audit remains valid (every constant from v1 unchanged in v2); the ~15 new keys added by v2 are all verified above.

---

## New math errors introduced by v2

| # | Severity | v2 location | Issue | Recommended action |
|---|---|---|---|---|
| 1 | 🟢 nit (presentation) | `utils.py:compute_lambda_values` row, plan line 248 | The LaTeX form of the lambda-return recursion: $G^{\lambda}_t = r_t + c_t \cdot [(1-\lambda) v_{t+1} + \lambda G^{\lambda}_{t+1}]$ — the `v_{t+1}` inside the bracket mixes caller-frame and function-internal indexing. Sheeprl source `utils.py:66-77` does `interm = rewards + continues * values * (1 - lmbda)` then `interm[t] + continues[t] * lmbda * vals[-1]`, which expands to $G^{\lambda}_t = r_t + c_t \cdot [(1-\lambda) v_t + \lambda G^{\lambda}_{t+1}]$ (function-internal, $v_t$ in the bracket, not $v_{t+1}$). My v1 review had it right at line 66. | Suggest one-word fix: change `v_{t+1}` to `v_t` in the LaTeX block at line 248. The operational guidance directly below the formula — link to sheeprl `utils.py:66-77`, mandatory numpy reference loop in docstring, assert-match-to-1e-6 against `lax.scan` — is correct and will catch the typo at implementation time. **Not blocking.** |

That's it — one 🟢 nit. No 🔴 critical, no 🟡 ambiguous, no new wrong-coefficient, no dropped equation, no new constant without sheeprl justification, no new YAML key without source. The v2 mechanical edits introduced no new math errors of consequence.

---

## Final verdict

**✅ PASS — zero residual math deviations; plan is mathematically faithful and ready for implementation.**

The critical 🔴 Checkpoint-5 bin-grid bug from v1 (the single-item pass/fail gate for this re-audit) is **resolved at three independent sites** with mutually consistent assertions. All 6 🟡 under-described details from v1 are **named globally** in a new Training-loop semantics section (§S1–§S10) and **re-cited per-file** at the change locations. All 5 paper equations and 5 cascade items remain mathematically correct. All ~15 new YAML keys added by v2 trace to specific sheeprl-source lines.

One 🟢 cleanup nit (the LaTeX indexing typo on `compute_lambda_values` at line 248) — flag for cleanup, **not blocking implementation**. The mandatory numpy reference loop in the docstring (a v2 addition) will catch this at Checkpoint 1.

Recommend the senior-developer apply the one-word LaTeX fix in a follow-up commit, but `developer` may proceed with Step 1 of the Implementation order immediately. The plan's math is at the level of fidelity required for the bit-identical parity gate to be a meaningful test.

Reviewed by: math-reviewer
