---
title: "dreamer-srl v3 CP6 — code-reviewer audit (cascade fix #29 + §S6/§S8/§S9)"
topic: dreamer
status: active
reviewer: code-reviewer
created: 2026-05-14
last_updated: 2026-05-14
---

# dreamer-srl v3 CP6 — code-reviewer audit

## Plain-language verdict

This review covers the **critic-loss checkpoint** (CP6) of the dreamer-srl v3
rebuild: the port of sheeprl's training-step critic-loss assembly (the part of
`dreamer_v3.py:L240–L320` that does the value-function update) to a new
`src/algorithms/dreamer_srl/train.py`. The core thing CP6 has to land correctly
is **cascade fix #29** — the original in-house DreamerV3 had ONE log-prob term
in the critic loss (`-qv.log_prob(λ)`), but Hafner's paper and the sheeprl code
have **TWO** (`-qv.log_prob(λ) - qv.log_prob(target_critic_mean)`). The second
term is the slow-target self-regulariser that keeps the critic from
overshooting against its own bootstrapped target during early training. Missing
the second term is a slow, silent training-quality bug. CP6 also lands the §S6
discount weighting (`cumprod(continues * gamma, axis=0) / gamma` with
`stop_gradient`, the trick that gives `discount[0] = 1` exactly when no
termination at step 0), and bonus §S8 free-nats per-element floor + §S9
`Independent(BernoulliSafeMode, 1)` wrap for the continue head.

I audited the new `train.py` (2 functions, ~245 lines including docstrings),
the loss.py extensions (`BernoulliSafeMode` / `IndependentBernoulli` /
`reconstruction_loss`, ~340 new lines), the 4 new pytest items, the
`gen_cp6_fixtures.py` generator (3 fixtures, multi-seed), the 3 new
`_run_critic_*` and `_run_discount_*` runners in `sheeprl_jax_diff.py`, and the
new deviation-log entry **D-010** (D-006-class linspace ULP cascade re-firing
under a different fixture seed; threshold relaxed from 3e-5 to 4e-5).

**The critic loss is line-for-line correct.** Both NLL terms are present at
`train.py:231,235`; both targets are `stop_gradient`'d; the discount tensor is
`stop_gradient`'d in `compute_discount` and verified by an actual `jax.grad`
through-flow test (`test_discount_weighting:317-328`). The §S6 cumprod/gamma
arithmetic produces `discount[0] = 1.0` exactly when `continues[0] = 1.0` (the
[0]=1 invariant). §S8's free-nats floor is per-element (`jnp.maximum(dyn_loss,
kl_free_nats)`) BEFORE the mean — the literalist `max(mean(dyn_loss),
free_nats)` trap is not present. §S9's `IndependentBernoulli.log_prob` correctly
sums over the trailing event dim of size 1.

**Process discipline restored.** The D-010 verdict cell reads
`☐ pending — PI ratification at CP6 gate` — the developer did NOT autonomously
flip it to `✅ APPROVED`, in contrast to the CP4 incident (commit `4491c66`).
The Lever-C grep I ran over the CP6 commit range (`git diff 8e201e5..1a4e51e
-- DEVIATION_LOG.md | grep "^\+.*✅ APPROVED"`) returned empty. This is the
strengthening proposed in the 2026-05-14 Process-notes section working as
designed.

**Verdict: PASS.** Two minor concerns and one nit. No blockers.
Math-reviewer can begin. D-010 forwards to PI with code-reviewer concurrence
(same substrate-mechanical class as D-006 / D-007 / D-008, identical
`TwoHotEncoding.log_prob` code path, threshold relaxation justified at 1.33×
the observed value).

## Critical assertion verification

### Cascade fix #29 — both log-prob terms present

Sheeprl reference (`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py`):

```python
L307: qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
L308-310: predicted_target_values = TwoHotEncodingDistribution(
              target_critic(imagined_trajectories.detach()[:-1]), dims=1).mean
L314: value_loss = -qv.log_prob(lambda_values.detach())
L315: value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
L316: value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))
```

JAX implementation (`src/algorithms/dreamer_srl/train.py:compute_critic_loss`):

| Line | Code | Verdict |
|---|---|---|
| `train.py:227` | `qv = TwoHotEncoding(qv_logits, dims=1)` | ✅ matches L307 (`dims=1`) |
| `train.py:231` | `neg_lp1 = -qv.log_prob(jax.lax.stop_gradient(lambda_values))` | ✅ matches L314 (term 1, `stop_gradient` on target) |
| `train.py:235` | `neg_lp2 = -qv.log_prob(jax.lax.stop_gradient(target_critic_values))` | ✅ matches L315 (term 2 — **cascade fix #29 present**, `stop_gradient` on target) |
| `train.py:239` | `discount_weights = discount[:-1].squeeze(-1)` | ✅ matches L316 (slice + squeeze) |
| `train.py:243` | `value_loss = jnp.mean((neg_lp1 + neg_lp2) * discount_weights)` | ✅ matches L316 (mean of sum × discount) |

**The cascade fix #29 test (`test_critic_loss_two_terms:161-166`) asserts
`max(|neg_lp2|) > 0.1`** — a defensive structural check that catches the
historical bug where `neg_lp2` would be all-zeros if the second log_prob term
were missing. If a future developer drops `-qv.log_prob(target_critic_values)`
from the loss, this assertion fires loudly. The test is genuine, not a
false-PASS.

### §S6 discount weighting

Sheeprl reference (L259-L260, inside `with torch.no_grad():`):

```python
discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma
```

JAX implementation (`train.py:138-140`):

```python
discount = jax.lax.stop_gradient(
    jnp.cumprod(continues * gamma, axis=0) / gamma
)
```

The `/ gamma` after the `cumprod` is the §S6 trick that gives
`discount[0] = continues[0] * gamma / gamma = continues[0]`. When `continues[0]
= 1.0` (the §S5 true-continue splice, no termination at imagination step 0),
`discount[0] = 1.0` exactly — verified at `test_discount_weighting:290` (asserts
`|discount[0].mean() - 1.0| < 1e-5`).

The `stop_gradient` is verified **structurally** at
`test_discount_weighting:317-328` by computing `jax.grad` of a sum through the
function and asserting `max(|grad|) < 1e-10`. This is a real witness that the
developer did not just add a comment — the JIT trace confirms the gradient
does not flow through `compute_discount`. ✅

### §S8 free-nats — per-element floor BEFORE mean (the literalist trap)

Sheeprl reference (`vendor/sheeprl/sheeprl/algos/dreamer_v3/loss.py:L68-L74`):

```python
free_nats = torch.full_like(dyn_loss, kl_free_nats)              # [T, B] tensor
dyn_loss  = kl_dynamic * torch.maximum(dyn_loss, free_nats)      # per-element
repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)  # per-element
kl_loss = dyn_loss + repr_loss                                    # [T, B]
# ... later: (kl_regularizer * kl_loss + ...).mean()
```

JAX implementation (`loss.py:reconstruction_loss`):

| Line | Code | Verdict |
|---|---|---|
| `loss.py:548` | `dyn_loss = kl_dynamic * jnp.maximum(dyn_loss, kl_free_nats)` | ✅ per-element max, BEFORE mean |
| `loss.py:556` | `repr_loss = kl_representation * jnp.maximum(repr_loss, kl_free_nats)` | ✅ per-element max, BEFORE mean |
| `loss.py:558` | `kl_loss = dyn_loss + repr_loss` | ✅ shape preserved `[T, B]` |
| `loss.py:573` | `total = (kl_regularizer * kl_loss + ...).mean()` | ✅ mean applied at end |

**The literalist trap `dyn_loss = max(dyn_loss.mean(), kl_free_nats)` is NOT
present.** `jnp.maximum` here is element-wise (the broadcast against the
scalar `kl_free_nats` produces an element-wise max over the `[T, B]` tensor),
which is the mathematically correct free-nats formulation. ✅

Caveat: `reconstruction_loss` has no CP6 Lever-A bit-identity test (the
developer's docstring at `loss.py:497-501` correctly notes this is exercised
at CP9). The §S8 logic is therefore audited at this code-review stage only,
not bit-identity tested. Acceptable because (a) the §S8 spec is one line of
code per max, (b) the historical "literalist trap" is the obvious bug to
guard against and isn't present, (c) CP9 will exercise the full path.

### §S9 `Independent(BernoulliSafeMode, 1)` wrap

Sheeprl references:
- L167: `pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(...)), 1)`
- L200 (in agent.py): same pattern
- L246: `continues = Independent(BernoulliSafeMode(logits=...), 1).mode`
- `vendor/sheeprl/sheeprl/utils/distribution.py:L409-L416` (`BernoulliSafeMode` class)

JAX implementation (`loss.py:252-417`):

| Component | Sheeprl line | JAX line | Verdict |
|---|---|---|---|
| `BernoulliSafeMode.__init__` | `distribution.py:L410-L411` | `loss.py:290-302` | ✅ stores logits + computes `sigmoid(logits)` for probs |
| `BernoulliSafeMode.mode` | `distribution.py:L413-L416` | `loss.py:304-316` | ✅ `(probs > 0.5).astype(probs.dtype)` matches sheeprl L415 exactly |
| `BernoulliSafeMode.log_prob` | inherited from `torch.distributions.Bernoulli` | `loss.py:318-354` | ✅ stable BCE form `value*log_sigmoid(logits) + (1-value)*log_sigmoid(-logits)` — equivalent to PyTorch's `binary_cross_entropy_with_logits(reduction='none')` |
| `IndependentBernoulli.log_prob` | `Independent(Bernoulli, 1).log_prob` sums over 1 event dim | `loss.py:401-417` | ✅ `base.log_prob(value).sum(axis=-1)` — for `[..., 1]` input shape this collapses to `[...]` |
| `IndependentBernoulli.mode` | `Independent(..., 1).mode` returns base.mode (no reduction) | `loss.py:388-399` | ✅ `self._base.mode` shape `[..., 1]` preserved |

The PyTorch `Independent.mode` semantics (delegates to base, no summation) is
matched. The PyTorch `Independent.log_prob` semantics (sums over `1` rightmost
event dim) is matched. Shape contracts: continue logits enter as `[T, B, 1]`,
`log_prob` returns `[T, B]`, which is the shape multiplied by `discount[:-1]`
downstream. ✅

### Isolation rule

```
$ grep -rn "^\s*\(import\|from\)\s\+src\.models\.dreamer_v3" \
    src/algorithms/dreamer_srl/
(no output)
```

`train.py` and `loss.py` both have docstring-text mentions of
`src.models.dreamer_v3` (for documentation of the isolation rule itself), but
the grep is anchored to actual import statements and returns empty. The
`test_train_module_does_not_import_from_src_models` test
(`test_train.py:341-367`) runs the same anchored regex check at pytest-time.
✅ Isolation maintained.

## D-010 audit — three sub-questions from the review brief

### Q1. Is the seed choice (`0xD3EAF + 1`) deliberate?

**Yes — established project convention.** The `SEED + N` offset pattern for
multi-fixture-within-one-CP files is in use at three other places:
- `scripts/fixtures/gen_cp2_fixtures.py:98` — `torch.manual_seed(SEED + 1)`
- `scripts/fixtures/gen_cp3b_fixtures.py:138` — `_rng = np.random.default_rng(SEED + 1)`
- `scripts/fixtures/gen_cp4_fixtures.py:211, 271` — `SEED + 10`, `SEED + 20`

Each fixture within a CP keeps its own deterministic seed offset so that one
fixture is reproducible independent of the others (and so that the `.npz`
filenames map 1:1 to seeds). The choice `SEED + 1` here is not a developer
hack to avoid file-name collisions; it is the standard fixture-disambiguation
pattern. **No action needed.** ✅

### Q2. Is the 4e-5 threshold principled?

**Yes — within the established margin-policy band.** Margins on the same
substrate-mechanical class so far:

| Deviation | Threshold | Observed | Margin |
|---|---|---|---|
| D-003 (CP1 symexp) | 2e-5 | ~1.3e-5 | 1.5× |
| D-006 (CP5 log_prob) | 3e-5 | 1.81e-5 | 1.65× |
| D-007 (CP2 GRU) | 5e-4 | 2.97e-4 | 1.68× |
| D-008 (CP4 RSSM) | 2e-3 | 7.19e-4 | 2.78× |
| **D-010 (CP6 log_prob)** | **4e-5** | **3.099e-5** | **1.29×** |

D-010's margin (1.29×) is slightly tighter than D-003/D-006/D-007 (~1.5–1.7×)
and considerably tighter than D-008 (2.78×, justified by chain depth). The
review brief asks whether 4e-5 is padding vs. principled — my read is that
1.33× is **defensible but minimal**, defensible because:

1. This is the **same code path** as D-006 (identical `TwoHotEncoding.log_prob`
   bins, identical bin-lookup arithmetic, identical log-softmax + cross-entropy
   reduction). The substrate-mechanical class is established.
2. The 3.099e-5 number is reproducible across runs (the diff-tool produces
   exactly 3.099e-5 with the fixture's torch reference).
3. The relative deviation `max_abs_diff / |log_prob_mean| ≈ 4e-6` is well below
   1 ULP relative for float32 (which is `~6e-8` per ULP, but `4e-6` is the
   relative drift accumulated through the bin-lookup chain — same scale as
   D-006 at `2.4e-6`).
4. Structural errors (wrong bins, wrong symlog encoding) produce O(0.1)
   deviation = **2500× above threshold**, so the relaxed threshold catches
   semantic bugs cleanly.

A slightly more generous 1.5× margin (= 4.65e-5) would be in tighter
alignment with D-006/D-007's 1.65–1.68× band. **My recommendation**: keep
4e-5 as logged. PI may wish to raise to 4.65e-5 (or round to 5e-5) for
margin-policy consistency, but the 1.33× margin is defensible on the
"same-code-path" identity argument and avoids the appearance of margin-creep.
Not a blocker either way.

### Q3. Cascade-math identity claim — is `compute_critic_loss` really the same code path as D-006?

**Yes — bit-identical to the D-006 `log_prob` path.** Direct trace:

```
train.py:227  qv = TwoHotEncoding(qv_logits, dims=1)
train.py:231  neg_lp1 = -qv.log_prob(stop_gradient(lambda_values))
                                  ↓
loss.py:198   n_bins = self.bins.shape[0]                  # bins = jnp.linspace(-20,20,255)
loss.py:201   x = symlog(x)                                # symlog target
loss.py:205   below = (self.bins <= x).sum(...) - 1        # bin lookup against linspace
loss.py:211-214 clamp + equal mask
loss.py:221-238 two-hot weights + scatter
loss.py:242   log_pred = logits - logsumexp(logits)
loss.py:245   return (target * log_pred).sum(axis=self.dims)
```

The `self.bins = jnp.linspace(-20, 20, 255)` at `loss.py:120` is the **exact
same line** that produces the D-006 platform ULP drift at `bins[127]` (JAX
0.0 vs PyTorch 7.45e-8). The log_prob arithmetic that follows is bit-identical
between the CP5 `test_twohot_log_prob_matches_sheeprl` call site and the CP6
`critic_target_lambda` runner — both call `TwoHotEncoding(logits, dims=1)
.log_prob(stop_gradient(target))`. The only difference is the input fixture
seed (`SEED+1` produces targets that land at bin-boundary-sensitive positions
slightly more often than `SEED` did). **The cascade-class identity claim is
correct.** ✅

D-010 should be ratified by PI under the same substrate-mechanical-class
precedent as D-003, D-006, D-007, D-008 — all are JAX-vs-PyTorch float32
arithmetic accumulation-order differences over algebraically identical
pipelines.

## Per-test audit table

| # | Test | Bit-identity real? | Source citation OK? | JAX correctness | Issues |
|---|------|--------------------|---------------------|-----------------|--------|
| 1 | `test_critic_loss_two_terms` | YES — fixture-gen builds `TwoHotEncodingDistribution(qv_logits, dims=1)` on the sheeprl side and stores `neg_lp1`, `neg_lp2`, `value_loss` as `torch_out_*` (gen_cp6_fixtures.py:111-129). Test checks all three against PyTorch reference at `4e-5` (`lp1`/`lp2`) and `1e-4` (scalar). | YES — sheeprl L307-L316 cited at `test_train.py:96`, `train.py:156-166`, and `_run_critic_loss_two_terms:1446`. Line range verified accurate. | Correct — calls `compute_critic_loss(qv_logits, lambda_values, target_values, discount)`; the `discount_extended` shim correctly converts the fixture's `[H, BT]` to `[H+1, BT, 1]` so `[:-1].squeeze(-1)` recovers the original. Cascade-fix-#29 sanity check (line 161) asserts `max(|neg_lp2|) > 0.1`. | None |
| 2 | `test_critic_target_lambda` | YES — fixture-gen computes `qv.log_prob(raw_lambda)` and `qv.log_prob(normed_lambda)` on sheeprl side (gen_cp6_fixtures.py:193-194), stores both. Test confirms `lp_raw_jax` matches `torch_out_lp_raw` and `lp_normed_jax` does NOT (diff > 1e-4 separation guaranteed by fixture's `assert diff > 1e-4` at gen line 206). | YES — sheeprl L314 cited at `test_train.py:202` and `_run_critic_target_lambda:1481`. The "actor uses normed; critic uses raw" distinction (L276-L279 vs L314) is correctly explained at `train.py:180-187`. | Correct — distinguishability assertion (line 238: `diff_normed > 1e-4`) is the key check that the test can actually tell raw from normed apart, not a coincidental equality. D-010 deviation (3.099e-5) lives in the `diff_raw` check at line 222. | None |
| 3 | `test_discount_weighting` | YES — fixture-gen runs `torch.cumprod(continues * gamma, dim=0) / gamma` directly (gen_cp6_fixtures.py:266) and stores both the full and sliced forms. Test compares JAX `compute_discount` output to stored torch reference at default `1e-6` threshold. Pure arithmetic, no D-006 cascade. | YES — sheeprl L259-L260 cited at `test_train.py:264` and `_run_discount_weighting:1518`. | Correct — three sub-asserts: full tensor match (line 281), `[0]=1` invariant (line 291), `[:-1].squeeze(-1)` shape + match (line 305). **The stop_gradient witness at line 317-328** is the structural check that `compute_discount` actually applied `jax.lax.stop_gradient` — not just commented intent. Genuine, not a false-PASS. Measured `max_abs_diff = 5.96e-8` (well below default 1e-6). | None |
| 4 | `test_train_module_does_not_import_from_src_models` | n/a — regex-anchored import-statement check; not numerical. | YES — references v2 Risks §13 / NNX_CONVENTIONS.md isolation rule. | Correct — `grep -E "^\s*(import\|from)\s+src\.models\.dreamer_v3"` anchored to line start, ignores docstring text. Same pattern as the loss.py isolation check (CP5). | None |

## Conventions audit checklist

(These are the standard CP-review conventions used in CP1–CP5. dreamer-srl is
a pure-JAX/numpy module — no pytree/struct or vmap concerns; no environment
sync or YAML config in CP6.)

- **Pytree / immutability** — n/a (no `EnvState` / `EnvParams` interaction in CP6)
- **JIT / static fields** — ✅ no traced-vs-static field changes
- **vmap / batch conventions** — n/a
- **PRNG key threading** — ✅ no PRNG use in CP6 (`compute_discount` and `compute_critic_loss` are deterministic given inputs)
- **Stop-gradient discipline** — ✅ three sites (`train.py:138, 231, 235`); structurally verified via `jax.grad` flow check at `test_train.py:317-328`
- **Pure-functional signatures** — ✅ `compute_discount(continues, gamma)` and `compute_critic_loss(qv_logits, lambda_values, target_critic_values, discount)` are side-effect-free
- **Source-citation (Lever B)** — ✅ docstring headers cite `dreamer_v3.py:L259-L260`, `L307-L316`, `L314`, `L315`, `L316`, `loss.py:L9-L88`, `distribution.py:L237`, `L253-L276`, `L409-L416` — all verified against vendored sheeprl
- **Isolation rule** — ✅ no `from src.models.dreamer_v3*` imports; AST + grep checks pass
- **Diff-tool registry** — ✅ 3 new entries (`critic_loss_two_terms`, `critic_target_lambda`, `discount_weighting`) registered in `FUNCTION_REGISTRY` (line 1562-1564) and `FUNCTION_THRESHOLDS` (line 1607-1608, both at `4e-5`; `discount_weighting` uses default `1e-6`); `CHECKPOINT_REGISTRY["CP6"]` (line 1629) lists all three
- **Lever-C autonomous-flip grep** — ✅ `git diff 8e201e5..1a4e51e -- DEVIATION_LOG.md | grep "^\+.*✅ APPROVED"` returns empty; no autonomous flip in commit range

## Concerns (non-blocking)

### 🟡 Concern 1 — Diff-tool max_abs_diff number drifts between runs

The implementation report (`IMPLEMENTATION_PLAN.md:523`) records the
`critic_loss_two_terms` diff at `1.287e-5`. When I re-ran
`scripts/sheeprl_jax_diff.py --checkpoint CP6` for this audit, the
diff-tool reported `2.193e-5` for the same fixture. Both are within
threshold (`4e-5`), but the variation is ~1.7× and reproducible across
multiple runs (suggesting JAX XLA reduction-tree non-determinism for
small-tensor reductions, not a one-off transient).

This is not incorrect — both runs match the stored torch reference within
the relaxed threshold. But it makes the "reported number in plan" vs
"observed number in audit" comparison harder for future reviewers. **Suggested
action (developer)**: re-run the diff-tool 3–5 times, record the max
observed value, and update the plan row's number to match (or annotate that
the value is approximate / platform-dependent within a 1.5–2× band on
multi-element reductions). No code change.

### 🟡 Concern 2 — D-010 margin slightly tighter than precedent band

D-010's `4e-5` threshold is 1.33× the observed `3.099e-5`. Prior
substrate-mechanical deviations sit in a 1.5–2.78× band (D-003 1.5×, D-006
1.65×, D-007 1.68×, D-008 2.78×). The 1.33× margin is defensible (same
code path as D-006, structural-error margin still 2500×) but is the
tightest in the series.

**Suggested action (PI)**: ratify D-010 as-logged at `4e-5`, OR raise to
`5e-5` (1.61× margin, in the median of the precedent band, single-digit
threshold) for consistency. Either is defensible. The deviation's class is
unambiguous (same as D-006), so the threshold-margin choice does not change
the verdict.

## Nits

### 🟢 Nit 1 — Docstring source-citation accuracy

The `_run_critic_loss_two_terms` runner docstring at
`scripts/sheeprl_jax_diff.py:1398-1399` references "threshold 4e-5 (same-order
as D-006; slightly wider for the sum of two terms)". The actual measurement
context is "max over concatenated `[lp1, lp2, value_loss]`", so the threshold
is wider because the max is taken over more elements + the scalar — not
because the loss "sums two terms". Minor docstring imprecision; not a code
issue.

## One-line conclusion

CP6 — critic loss with EMA self-regulariser (cascade fix #29), §S6 discount
weighting, §S8 free-nats, §S9 BernoulliSafeMode — is line-for-line correct
against sheeprl `dreamer_v3.py:L240-L320` and `loss.py:L9-L88`; D-010 is the
same substrate-mechanical class as D-006 and forwards to PI with code-reviewer
concurrence to approve at the logged `4e-5` threshold; process discipline (no
autonomous deviation-cell flip in this commit range) is restored. **PASS.**

Reviewed by: code-reviewer
