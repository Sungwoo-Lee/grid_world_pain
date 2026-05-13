---
title: "dreamer-srl v3 CP9b — code review"
topic: dreamer
status: active
reviewer: code-reviewer
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/dreamer_srl_main.py (L387-L404, L490-L493), tests/algorithms/dreamer_srl/test_prefill.py, configs/dreamer_srl/01_food_only.yaml, configs/dreamer_srl/01_food_only_smoke.yaml
---

# CP9b Code Review — ⚠ PASS WITH NOTES

## Headline

CP9b replaces the CP9 placeholder random-action prefill (a Python `for`-loop calling NumPy's global RNG) with a clean JAX-RNG branch that uses the driver's seeded PRNG key. In plain language: for the first ~1024 environment steps of a fresh training run, the agent samples its actions uniformly at random (rather than using its random-init policy) — and CP9b makes that code path production-quality, seeded by the driver's `--seed`, and guarded by two automated tests. The implementation itself is small (8 lines replaced by 18), faithful to sheeprl's `iter_num <= learning_starts` action gate at L558-L571, and preserves the `actions_oh` dtype/shape contract that downstream code depends on. The two property tests are well-scoped and pass.

The notes (not blockers): (1) the production code's explanatory comment at L391-L393 contains an inaccurate claim about `ratio(0) == 0` at iter `learning_starts` — the inaccuracy is the same one the developer correctly flagged-and-fixed in Test 2's spec, but the false claim was copied into the production code and left there; (2) the dreamer-srl driver's train-gate at L490-L493 omits sheeprl's `prefill_steps * policy_steps_per_iter` subtraction from `ratio_steps`, which causes a debt-repayment burst (≈`learning_starts * replay_ratio` grad steps in a single iteration) the moment the gate opens — this is a pre-existing CP9 design choice, not a CP9b regression, but the CP9b production comment falsely claims the boundary fires zero grad steps. The "no gradient before `learning_starts`" *invariant* DOES hold (no grad step at iter < `learning_starts`); the comment's "zero grad steps AT learning_starts" sub-claim is the part that's wrong.

## Verdict

**⚠ PASS WITH NOTES** — ready to flip CP-PASS pending PI-N/A, provided the developer fixes the misleading code comment at L391-L393 (F1) in a follow-up commit before the parity-track launch. The two Lever-A tests are sound, the Lever-B citation is correct, and the production code itself is bit-faithful to sheeprl's action gate at L558-L571. No verdict-cell flip was attempted by the developer (CP9b row at IMPLEMENTATION_PLAN.md:527 still reads `NOT STARTED` — confirmed).

## Findings

### F1 — 🟡 Concern (production-code comment is wrong about train-gate boundary semantics)

`src/algorithms/dreamer_srl/dreamer_srl_main.py:391-393`:

```python
# The train-gate at L483 uses `>=`, so iteration `learning_starts` itself adds
# zero gradient steps (ratio(0) == 0), preserving the
# "no gradient before learning_starts" invariant.
```

This claim is **factually wrong for our driver as implemented**:

- The train-gate at L490-L493 reads `ratio_steps = policy_step; n_grad_steps = ratio(ratio_steps)`. There is NO subtraction of `prefill_steps * policy_steps_per_iter` (unlike sheeprl L661).
- At `iter_num == learning_starts == 1024`, `policy_step == 1024`, so `ratio.__call__(1024)` is invoked with `_prev=None` on its first call.
- Per `src/algorithms/dreamer_srl/utils.py:318-330` (and sheeprl `utils/utils.py:273-291`), the first-call branch returns `int(step * self._ratio) = int(1024 * 1.0) = 1024` grad steps — NOT 0.

The comment was lifted from the CP9b plan, where the same erroneous claim appears (CP9B_PLAN.md:111-112 and §Test 2 spec). The developer correctly noticed the inaccuracy when writing Test 2 and adjusted the test, recording the plan-reality discrepancy in the Implementation Report. But the same false explanation was carried over into the production-code comment block and never updated.

**Note on the actual invariant.** The "no gradient before `learning_starts`" invariant DOES hold — for `iter_num` in `[1, learning_starts - 1]`, the outer `if iter_num >= learning_starts:` guard is False, so `ratio()` is never called and no grad step fires. So the SAFETY property is preserved. The wrong claim is the SHARP edge: it asserts the gate fires exactly zero at iter `learning_starts`, when in fact it fires `int(learning_starts * replay_ratio)` (debt repayment).

**Sub-concern**: this debt-repayment behaviour is a substantive *behavioural* divergence from sheeprl. Sheeprl's `ratio_steps = policy_step - prefill_steps * policy_steps_per_iter` ensures that at iter `learning_starts` the first ratio call sees a small step value (≈1 with `num_envs=1, learning_starts=1024`) and returns ≈1 grad step. The dreamer-srl driver's first ratio call sees `policy_step = learning_starts` and returns ≈`learning_starts` grad steps in a single iter — a one-shot debt-repayment burst.

For the parity-track config (`learning_starts=1024`, `replay_ratio=1`, `num_envs=1`), this means **iteration 1024 fires ~1024 gradient steps in one iter**, then iteration 1025 fires ~1 grad step (matching sheeprl from there onward). The training trajectories will be visibly different from sheeprl at the boundary even though both reach steady-state replay-ratio = 1 within one extra iter. The second-clause guard `buffer._pos >= seq_len` does NOT save us because by iter 1024 `buffer._pos = 1024 >> seq_len = 64`.

**Fix options** (developer to choose under senior-dev direction):

1. **Minimal — fix the false comment**: replace L391-L393 with the actual semantics:

   ```
   # The train-gate at L490 uses `>=` and does NOT subtract prefill_steps from
   # policy_step (unlike sheeprl L661). At iter_num == learning_starts the
   # gate fires for the first time and the Ratio scheduler returns
   # int(learning_starts * replay_ratio) grad steps in a debt-repayment burst.
   # The "no gradient before learning_starts" invariant is preserved by the
   # outer if-guard, NOT by ratio(0) == 0.
   ```

2. **Stronger — port sheeprl's `prefill_steps` subtraction**: at L492 change `ratio_steps = policy_step` to `ratio_steps = policy_step - (learning_starts - 1) * num_envs` (with the guard `learning_starts > 0`), matching sheeprl L661. This eliminates the debt-repayment burst and brings the driver into bit-identity at the train-gate boundary. If you take this option, CP9b's Test 2 spec should be tightened: add a boundary assertion that at `iter_num == learning_starts`, `n_grad_steps == 1` (not `learning_starts`). The math-reviewer skip remains correct (still no new math); the code-reviewer charter says this is the better fix.

**Senior-developer call**: option 1 is a documentation-only fix that lands in minutes; option 2 is a 2-line behavioural fix that may need a CP9c or a D-014 entry for "dreamer-srl ratio_steps formula differs from sheeprl by `prefill_steps` term". My recommendation is option 2 plus a D-014 entry — the parity-track launch is the next checkpoint and a one-shot 1024-grad-step burst at iter 1024 is a non-trivial training-trajectory anomaly that's easier to fix now than to explain in the post-hoc parity-comparison report.

### F2 — 🟢 Nit (test 1's per-sample Python loop is slow; vectorisable)

`tests/algorithms/dreamer_srl/test_prefill.py:90-95`:

```python
for _ in range(n_samples):
    key, k_player = jax.random.split(key)
    actions_oh = _sample_prefill_action_oh(k_player, num_envs, action_dim)
    idx = int(np.argmax(actions_oh[0]))
    counts[idx] += 1
```

10,000 iterations × (split + randint + one_hot + argmax + scalar conversion) = 13.21 s, dominated by JAX dispatch overhead. The same property holds if you do:

```python
keys = jax.random.split(jax.random.PRNGKey(seed), n_samples + 1)
idx_batch = jax.random.randint(keys[0], (n_samples,), 0, action_dim)
counts = np.bincount(np.asarray(idx_batch), minlength=action_dim)
```

That would drop the test from ~13 s to <0.1 s. Not a blocker — test correctness is unaffected — but the Lever-A test suite is starting to add up, and this is the slow one. Suggest deferring to a follow-up commit (or to CP10 cleanup).

### F3 — 🟢 Nit (test 1 indirectly tests `jax.random.randint`, not the production gate)

`test_prefill_uniform_entropy_below_learning_starts` measures the empirical distribution of `jax.random.randint(key, (1,), 0, 4)` and verifies it's uniform. This is a property of `jax.random.randint` (which we trust as a JAX primitive); the test does NOT directly exercise the gate predicate `iter_num <= learning_starts`. The test's value is as a regression guard against three specific regression classes the docstring enumerates (accidental argmax bias, off-by-one on minval/maxval, RNG-key reuse, np.random.randint reintroduced) — all of which would break the helper, not the gate.

A stricter regression guard would set up a tiny stub of the driver loop (without an env or agent), run it for `iter_num` in `[1, 2*learning_starts]`, and assert that the per-iter action distribution is uniform when `iter_num <= learning_starts` and DOES NOT match uniform when `iter_num > learning_starts` (because the random-init policy is not perfectly uniform). The current test only catches the first half. Not a blocker — Test 2 covers the gate-predicate side via the gradient-counter — but worth noting.

### F4 — 🟢 Nit (test 1 has a tolerance-vs-N mismatch in the docstring)

`tests/algorithms/dreamer_srl/test_prefill.py:73-76`:

> "the standard error on each empirical probability is sqrt(p(1-p)/N) ≈ 0.0043, and the corresponding entropy error band at log(4)=1.3863 is ~0.005 with high probability. We set tol=0.01 (2 sigma)."

The arithmetic in the docstring is a bit loose. For multinomial counts at $N=10{,}000$, $k=4$, $p=1/4$, the SD of the empirical entropy via delta-method is closer to $\sigma_H \approx \sqrt{(k-1)/(2N^2) \cdot ...} \approx 0.003$, not 0.005. So tol=0.01 is roughly 3-sigma (one-sided), not 2-sigma. Not a flakiness concern — the test has more headroom than the docstring claims — but the docstring should match the actual margin. Suggest tightening the comment OR loosening tol to e.g. 0.005 for a stricter regression guard.

## Conventions audit

| Convention | Status | Notes |
|---|---|---|
| Pytree / immutability | ✅ | N/A (driver-level scalar control flow; no pytree mutation) |
| JIT recompilation | ✅ | The prefill conditional `if iter_num <= learning_starts:` is at the driver-loop level, NOT inside `@nnx.jit`. `iter_num` is a Python int; no leaked Python control flow into a traced function. |
| vmap / batch shape | ✅ | `jax.random.randint(k_player, (num_envs,), 0, action_dim)` returns one independent action per env (no broadcast collision). `jax.nn.one_hot` produces `[num_envs, action_dim]`. |
| PRNG key threading | ✅ | `key, k_player = jax.random.split(key)` is consumed identically in both branches (prefill and policy), preserving the post-action key trajectory regardless of branch. `jax.random.randint(k_player, ...)` cleanly seeds from the driver stream. No NumPy global RNG (the CP9 stub bug is gone). No key reuse. |
| Sensor / obs-breakdown sync | ✅ | N/A (no sensor or obs-layout changes) |
| Config protocol | ✅ | `agent_cfg.get_mandatory("algo.learning_starts", int)` at L197 — no `.get(..., default)` fallback. Smoke and parity configs both have explicit `learning_starts` keys with documented values. |
| Lever-B citation | ✅ | `# Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L558-L571` at L388. The cited line range exactly brackets the sheeprl `iter_num <= learning_starts` action gate (verified by reading the vendored file). Pin commit `33b6366` confirmed at `vendor/sheeprl/.pinned_commit`. |
| Deviation-log discipline | ✅ | CP9b row at IMPLEMENTATION_PLAN.md:527 remains `NOT STARTED`. The developer did not flip the verdict cell (post-CP4 / post-CP8 streak preserved). |
| Test placement | ✅ | `tests/algorithms/dreamer_srl/test_prefill.py` is consistent with the existing per-module organisation (`test_agent.py`, `test_buffers.py`, `test_loss.py`, `test_train.py`, `test_utils.py`, `test_end_to_end_parity.py`). |

## Per-checklist verification

### Lever-A correctness

**Test 1 (`test_prefill_uniform_entropy_below_learning_starts`)** — measures empirical action histogram from samples of `jax.random.randint` (production path mirrored line-for-line in the helper). Sample size `N=10_000` against `action_dim=4` gives ~3-sigma margin at `tol=0.01` against the asymptotic SD. Seed `0xD3EAF` matches the fixture-seed convention in `tests/fixtures/dreamer_srl/README.md`. Empirical entropy is measured on the SAMPLES, not on a policy-output distribution (policy is correctly bypassed during prefill). Test PASSES at 13.21 s and is deterministic across runs. The test catches the listed regression classes (np.random.randint reintroduced, key reuse, minval/maxval off-by-one). **Verdict: correct as a regression guard. F3/F4 are nits on scope/tolerance, not blockers.**

**Test 2 (`test_no_gradient_step_before_learning_starts`)** — simulates the driver's train-gate logic with a mock Ratio scheduler (not the full driver). For `learning_starts=10`, `replay_ratio=1`, the test asserts: (i) iters 1..9 have zero grad steps; (ii) at least one grad step fires at iter ≥ 10 (anti-vacuous). Sheeprl-side semantics: at iter 10 with `_prev=None`, `ratio(10)` returns `int(10*1) = 10` grad steps on the FIRST CALL (debt repayment) — so `grad_step_at_iter[9] == 10`, not `1` and not `0`. The test was correctly adjusted from the plan's spec (which incorrectly asserted zero at boundary) to preserve only the hard invariant. **Verdict: correct as a regression guard for the safety invariant. The adjustment matches the actual `Ratio.__call__` semantics. No new sheeprl deviation is introduced by the test itself — the gradient-counter test correctly probes the driver's actual behaviour.**

### JAX/Flax-NNX correctness (`src/algorithms/dreamer_srl/dreamer_srl_main.py`)

- **Random-action sample**: `jax.random.randint(k_player, (num_envs,), 0, action_dim)` — correct. No hidden NumPy / Python-`random` call. `jax.nn.one_hot(..., dtype=jnp.float32)` produces the expected `[num_envs, action_dim]` float32 array; `np.asarray(...)` converts to NumPy as downstream code expects.
- **PRNG threading**: `key, k_player = jax.random.split(key)` at L394 — clean split off the main stream. `k_player` is used in EXACTLY one branch (consumed by either `jax.random.randint` or `player.get_actions(...)`), preserving the post-action key trajectory regardless of branch. No collision with RSSM / policy keys (those are split downstream from the same `key` after action selection).
- **Vmap / batch shape**: shape `(num_envs,)` produces one independent random index per env (no broadcast). For `num_envs=1` this is `(1,)`; for `num_envs=N` (future parallel rollout) this scales correctly.
- **JIT recompilation**: the `if iter_num <= learning_starts:` is a Python-level branch at the driver-loop scope. `iter_num` is a Python int from `range(...)`, NOT a traced JAX value. The conditional does NOT live inside `@nnx.jit` (the train-step is JIT-compiled separately at `train_step = make_train_step(...)` at L287-L301, and the action-selection happens BEFORE `train_step` is called). No leaked Python control flow into tracing.
- **Train-gate boundary semantics**: `>=` is correct at L490 (matches sheeprl L660). `policy_step` is the right quantity (env-step count, incremented by `num_envs` per iter at L379) — matches sheeprl. **BUT** the dreamer-srl driver omits sheeprl's `prefill_steps * policy_steps_per_iter` subtraction (see F1). The "no gradient before `learning_starts`" invariant is preserved by the outer `if` guard, but the debt-repayment burst at iter `learning_starts` differs from sheeprl.

### Project-convention compliance

- **Mandatory-key discipline**: `agent_cfg.get_mandatory("algo.learning_starts", int)` at L197 ✅.
- **Config edits**: `01_food_only.yaml:15` has `learning_starts: 1024` with a CP9b-restoration comment ✅. `01_food_only_smoke.yaml:27` retains `learning_starts: 0` with an explicit smoke-only-deviation rationale at L19-L24 ✅.
- **Test placement**: `tests/algorithms/dreamer_srl/test_prefill.py` is a new file, consistent with the project's per-module convention (no per-feature-flag splits inside `test_train.py`).
- **No verdict-cell autoflip**: CP9b row at `IMPLEMENTATION_PLAN.md:527` remains `NOT STARTED` ✅. The developer's Implementation Report ends with "ready for code-reviewer + professor review chain" (not a flip claim).

### Plan-reality discrepancy disposition

The developer's flagged discrepancy is genuine, correctly diagnosed, and correctly handled in the TEST. The plan's claim that `ratio(0) == 0` at iter `learning_starts` is false because:

1. `ratio_steps = policy_step` (not `policy_step - prefill_steps * policy_steps_per_iter`) at L492.
2. `Ratio.__call__` on its first call (with `_prev=None`) returns `int(step * ratio)`, not 0 — see `utils.py:318-330` (mirrors sheeprl `utils/utils.py:273-291`).

The developer correctly adjusted Test 2 to assert only the hard invariant (zero grad steps before `learning_starts`) and noted the discrepancy as a documentation bug in the plan, not a code-vs-sheeprl divergence in the *test*. **However**, the developer did NOT propagate this fix to the production-code comment at L391-L393, which still asserts the same false `ratio(0) == 0` claim. That's F1.

**My recommendation to senior-developer on the plan-reality discrepancy**:

1. The test fix is correct as-is; the plan's verbal description was inaccurate, the test now matches reality.
2. CP9B_PLAN.md (the plan doc) should be amended post-hoc — strike the "ratio(0) == 0 at boundary" claim from §Analysis (lines 99-119) and from §Test 2 spec (lines 468-479), replacing with the actual debt-repayment semantics. This is a documentation-fix amendment, not a deviation log entry.
3. The bigger question: does the dreamer-srl driver need to be brought into sheeprl-parity at the train-gate boundary (option 2 in F1)? My view is yes — the 1024-grad-step burst at iter 1024 is a non-trivial training-trajectory anomaly that's easier to fix at CP9b/CP9c than to explain post-hoc. But this is a senior-developer call, possibly requiring a D-014 entry if the choice is to keep the current behaviour and document the divergence.

## Recommendation for senior-developer

**Ready to flip CP-PASS pending PI-N/A**, provided F1 is addressed before the parity-track launch. The technical work is sound; the implementation faithfully ports sheeprl's action gate at L558-L571 to JAX, the two Lever-A tests are well-scoped regression guards, and the developer correctly caught and disclosed the plan-reality discrepancy.

**Hard ask**: fix the misleading production-code comment at `dreamer_srl_main.py:391-393` (F1, option 1 — minimal). This costs minutes and removes a future-reader landmine.

**Soft ask** (recommend, do not block): port sheeprl's `prefill_steps` subtraction at L492 (F1, option 2). This brings the train-gate into behavioural parity with sheeprl at iter `learning_starts`, eliminating the one-shot debt-repayment burst. If the senior-developer chooses to keep the current behaviour, file a D-014 entry making the divergence explicit and bring it to the parity-launch PI consultation.

**Documentation amendment**: CP9B_PLAN.md §Analysis and §Test 2 spec contain the same false `ratio(0) == 0` claim that's in the production comment. Strike those, replace with the actual semantics. This is a post-hoc plan amendment, not a deviation.

F2/F3/F4 are nits; defer or close at senior-developer discretion.

## One-line conclusion

CP9b lands a clean, well-tested JAX-RNG random-action prefill that faithfully ports sheeprl's action gate; the only material issue is a false explanatory comment in the production code (and the same false claim in the plan) about train-gate boundary semantics — fix the comment, and consider porting sheeprl's `prefill_steps` subtraction to eliminate the debt-repayment burst at iter `learning_starts`.

Reviewed by: code-reviewer
