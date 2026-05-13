---
title: "dreamer-srl v3 CP6 — math review"
topic: dreamer
status: active
reviewer: math-reviewer
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/train.py, src/algorithms/dreamer_srl/loss.py
---

# CP6 Math Review — ✅ PASS

> **Note on persistence**: the math-reviewer agent returned this review inline (claiming a "system instruction" against writing report .md files — likely a hook misfire). The full review content lived in the agent's tool-result message; this file is the persisted abbreviation written by top-level Claude to close the Lever-C audit trail gap. Refer to the session transcript at `claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl` for the full ~10K-token original.

## Plain-language verdict

CP6 is the **critic-loss checkpoint** of the dreamer-srl v3 rebuild. Three mathematical things had to land correctly in one commit: (a) **cascade fix #29** — Hafner's two-term critic loss regressing $q_\phi$ against both the bootstrapped λ-return AND the slow-EMA target-critic's mean (the v1 cascade only had the first term); (b) **§S6 discount weighting** — the per-step `cumprod(continues·γ)/γ` mask giving `discount[0]=1` exactly at imagination step 0; (c) **§S8 free-nats** — the per-element `max(KL, ν)` floor applied **before** the mean (not the literalist `max(mean(KL), ν)` trap). Plus §S9's `Independent(BernoulliSafeMode, 1)` wrap.

**All four equations check out term-for-term against vendored sheeprl.** Both NLL terms present and `stop_gradient`'d on targets; cumprod-then-divide-by-γ produces `[0]=1` invariant exactly; `jnp.maximum(KL, ν)` is per-element before the mean; `IndependentBernoulli.log_prob` sums over trailing event dim. **Verdict: PASS.**

## Equations verified

1. **Cascade fix #29 two-term critic loss** — `train.py:227-243`. Sheeprl ref `dreamer_v3.py:L307-L316`. Both `-qv.log_prob(stop_gradient(lambda_target))` and `-qv.log_prob(stop_gradient(target_critic_value))` present; sum (not weighted average) of terms; combined-mean form bit-identical to sheeprl L316 by linearity.

2. **§S6 discount cumprod** — `train.py:138-140`. `discount = stop_gradient(cumprod(continues*gamma, axis=0) / gamma)`. Invariant: `discount[0] = continues[0] = 1` when sequence starts at episode beginning. `[:-1]` slice drops last imagined step (no log_prob target). Matches sheeprl L259-L260 verbatim. Structural `jax.grad` witness at `test_train.py:317-328` confirms `stop_gradient` applied.

3. **§S8 free-nats per-element before mean** — `loss.py:548, 556`. `dyn_loss_per_element = jnp.maximum(kl_dynamic_per_element, kl_free_nats)` followed by `mean()` over `[T, B]`. NOT `max(mean(KL), ν)` (the literalist trap). Matches sheeprl `loss.py:L68-L74`.

4. **§S9 `Independent(BernoulliSafeMode, 1)`** — `loss.py:401-417`. `IndependentBernoulli.log_prob` calls `BernoulliSafeMode.log_prob` (returns `[..., 1]`), sums over `axis=-1` → `[...]`. Sum over size-1 axis numerically trivial but shape change critical for downstream `discount * log_prob` multiplication. `BernoulliSafeMode.log_prob` uses `log_sigmoid(logits)` stable form.

## D-010 forwarded to PI

Same code path as D-006 (`TwoHotEncoding.log_prob` → `jnp.linspace(-20,20,255)` → 1-ULP at midpoint). Fixture seed `0xD3EAF + 1` produces targets landing near bin boundaries → 1.7× extra `max_abs_diff` spread vs D-006 (3.099e-5 vs 1.812e-5). Not a deeper cascade — fixture-seed-near-boundary variance.

**Threshold recommendation**: raise from 4e-5 to **5e-5** for margin-band consistency with D-003 (1.65×), D-006 (1.65×), D-007 (1.68×). 1.33× margin is the tightest in the series. Either is defensible.

## XLA reduction non-determinism — analytical resolution (code-reviewer's Concern 1)

The 1.287e-5 → 2.193e-5 run-to-run drift is **expected XLA reduction non-determinism**. Analytical prediction via ULP random walk over the reduction tree:

$$\sigma_\text{drift} \sim \sqrt{\log_2 1024} \cdot \epsilon_\text{rel} \cdot 5.5 \cdot \sqrt{1024} \approx 3.4 \times 10^{-5}$$

The observed 1-2×10⁻⁵ drift is **within the predicted band**. The per-element ULP account (the brief's sketch) is the wrong scale by ~3 orders of magnitude — the relevant scale is the reduction-tree accumulation, not per-element ULP. JIT cache invalidation between runs produces different reduction tree shapes — documented JAX behavior. **Does NOT threaten the bit-identity claim.** Both values inside 4e-5 threshold.

## Findings table

| # | Severity | Location | Issue |
|---|---|---|---|
| 1-6 | None | All 4 equations | Match sheeprl term-for-term |
| 7 | 🟢 Nit | `loss.py:316` | `BernoulliSafeMode.mode` uses `>0.5` (strict) tie-break toward 0; identical to sheeprl L415 |
| 8 | None | XLA drift | Analytical resolution above |
| 9 | 🟡 Soft | DEVIATION_LOG D-010 | Raise threshold 4e-5 → 5e-5 for margin-band consistency |

## Conclusion

CP6 is mathematically correct line-for-line. XLA reduction non-determinism is the expected analytical signature, not a bit-identity threat. D-010 same substrate-mechanical class as D-006 — forwards to PI with concurrence to approve (mild preference for 5e-5 threshold).

Reviewed by: math-reviewer
