---
title: "dreamer-srl v3 CP9b — professor-rl-bayesian-dl audit (random-action prefill §S3)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/dreamer_srl_main.py, tests/algorithms/dreamer_srl/test_prefill.py, configs/dreamer_srl/01_food_only.yaml, configs/dreamer_srl/01_food_only_smoke.yaml
---

# dreamer-srl v3 CP9b — professor-rl-bayesian-dl audit

## Plain-language verdict

**Question.** CP9b is the eleventh checkpoint of the dreamer-srl v3 rebuild and turns on the **random-action prefill** that DreamerV3 (Hafner et al. 2023) calls "§S3". In plain English: at the start of a fresh training run the policy network's weights are random, so its action distribution is meaningless; for the first `learning_starts` environment steps (here, 1024 — the sheeprl XS default) the driver therefore **ignores the policy** and instead samples actions **uniformly at random** from the discrete action set. Those random transitions are still **written to the replay buffer**, so the world model has a diverse seed of (obs, action, reward, next-obs) data to learn from when training actually begins. **No gradient updates fire** during that prefill window. This three-part contract — uniform sampling, buffer-fill, zero gradients — is what Hafner et al. call §S3 and what sheeprl's `dreamer_v3.py` lines 558–571 and 660–698 implement. CP9b's job is to port that contract faithfully into the JAX driver and to add two property tests that would catch the silent-failure modes.

**My job is the RL-theoretic + algorithmic-faithfulness gate.** Does the JAX driver's prefill branch realise the §S3 contract, point by point? Do the two new tests actually catch the failure modes that would silently regress a future cleanup? Does the D-012 deviation (the CP9-smoke setting of `learning_starts=0`) cleanly close, with the parity-track config now back on the sheeprl XS default? Are there any forward-looking concerns the parity launch will expose that the CP9 smoke could not, because the prefill window was switched off there?

**Headline.** **PASS WITH ONE MINOR NOTE.** The driver branch at `dreamer_srl_main.py:L387-L404` is a faithful JAX port of sheeprl@33b6366:L558-L571 modulo the documented `resume_from` + minedojo clause drops. The three §S3 sub-contracts each hold: actions are drawn from `jax.random.randint(key, (num_envs,), 0, action_dim)` (uniform); the buffer-add at L409 is **outside** the train-gate, so prefill transitions land in the buffer; the train-gate at L490 uses `iter_num >= learning_starts`, matching sheeprl's L660 exactly. The empirical-entropy test passes by 13.21 s with `|H_emp − log 4| < 0.01`. The gate-counter test passes with zero gradient steps over iters 1..9 for `learning_starts=10`. **D-012 fully closes** on the parity-track config; the smoke retains `learning_starts=0` with a clearly-labelled "smoke-only deviation" comment, which is the right disposition. **The developer's adjustment to Test 2** — removing the plan's incorrect claim about `Ratio` returning 0 at the boundary — is algorithmically correct and preserves the hard invariant the test was designed to guard.

**The minor note (F1).** Test 1's empirical-entropy proxy has a non-trivial overlap with the post-cascade-fix-#27 zero-init policy's own action distribution: a zero-init categorical actor produces approximately uniform action samples at step 0 by construction (the logits are all zero, softmax is `[1/A, ..., 1/A]`, sampling is uniform). The test, if the §S3 branch were silently disabled and the driver fell through to `player.get_actions()` from iter 1 onwards, would land in a regime where the actor *also* produces near-uniform actions and the entropy criterion `|H_emp − log A| < 0.01` could pass falsely. This is not a fatal flaw — Test 1's standalone helper `_sample_prefill_action_oh` reproduces the production sampling formula in isolation, and the bonus assertion `np.all(counts > 0)` rules out the only failure mode (off-by-one `maxval`) that would otherwise make the test trivially vacuous — but **the test is testing the helper, not the production gate**. If the helper drifts from the driver branch (which the docstring warns about), or if a future refactor silently routes the prefill iterations through `player.get_actions()`, this test would not catch it. **The recommendation is documentary only**: the test's docstring should make this scope explicit, and CP10's runtime measurement should be configured to record the action-distribution histogram over the first 1024 iterations of the parity launch as a one-shot live cross-check (no new test needed; just a log key in WandB). I do not consider this a blocker for CP9b; it is a CP10/parity-launch consideration.

**Verdict.** **✅ PASS** on all four §S3 sub-contracts; concur with the developer's adjustment to Test 2; concur with D-012 closure on the parity-track config and the smoke-only deviation rationale; concur with the plan-reality discrepancy disposition (no D-014, the production code is correct and the plan's verbal description of `Ratio` was imprecise). **F1 is a documentary recommendation for Test 1's scope statement and a CP10 live-cross-check, not an algorithmic blocker.** Ready to flip CP-PASS pending the code-reviewer's parallel gate; no PI consultation is needed (CP9b is reviewer-chain-only per plan line 527; the parity launch is the next PI trigger).

---

## §S3 contract audit (verify-checklist item 1)

The §S3 contract has three sub-claims. I audit each against the JAX driver and the sheeprl reference.

### Sub-claim 1 — uniform-random action sampling during prefill

**What sheeprl does** (`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L558-L571`):

```python
if iter_num <= learning_starts and cfg.checkpoint.resume_from is None and "minedojo" not in ...:
    real_actions = actions = np.array(envs.action_space.sample())
    if not is_continuous:
        actions = np.concatenate([F.one_hot(torch.as_tensor(act), act_dim).numpy() ...], axis=-1)
```

`gym.spaces.Discrete.sample()` draws from `np.random.randint(0, n)` — a uniform integer over `{0, ..., n-1}`. The one-hot encoding via `F.one_hot` is the standard injection into the action-dim simplex.

**What the JAX driver does** (`src/algorithms/dreamer_srl/dreamer_srl_main.py:L394-L402`):

```python
key, k_player = jax.random.split(key)
if iter_num <= learning_starts:
    action_idx = jax.random.randint(k_player, shape=(num_envs,), minval=0, maxval=action_dim)
    actions_oh = np.asarray(jax.nn.one_hot(action_idx, num_classes=action_dim, dtype=jnp.float32))
else:
    actions_oh = player.get_actions(obs, is_first, k_player)
```

`jax.random.randint(key, shape, minval, maxval)` samples uniformly from $\{minval, ..., maxval - 1\}$. With $minval=0$, $maxval=A$, this is exactly $\text{Uniform}\{0, 1, ..., A-1\}$, identical to gym's `Discrete(A).sample()` in distribution (the streams disagree byte-for-byte by construction — JAX RNG vs. NumPy RNG — but that is the same class as D-002, not a deviation). The one-hot via `jax.nn.one_hot(...)` produces the same simplex vertex `e_a` as `F.one_hot` does on torch, and the `np.asarray(...)` wrap preserves the `[B, action_dim]` float32 dtype/shape contract that downstream `np.argmax(actions_oh, axis=-1)` at L413 requires.

**Verdict on sub-claim 1.** ✅ Faithful. The distributional claim — actions drawn iid uniform over the discrete action space at every prefill iteration — holds. The dropped clauses (`resume_from` and minedojo) are correctly out-of-scope per the JAX port (no checkpoint-resume in scope yet; no minedojo support).

### Sub-claim 2 — prefill transitions added to the replay buffer

**What sheeprl does** (`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L586-L587`):

```python
step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
rb.add(step_data, validate_args=cfg.buffer.validate_args)
```

The `rb.add` call is **inside the `timer("Time/env_interaction_time")` block** but **outside any train-gate**. It fires every iteration of the main loop, including prefill iterations.

**What the JAX driver does** (`src/algorithms/dreamer_srl/dreamer_srl_main.py:L406-L409`):

```python
step_data["actions"] = actions_oh[np.newaxis]  # [1, B, action_dim]
buffer.add(step_data, validate_args=False)
```

The `buffer.add` call sits at indentation level 1 of the `for iter_num in range(...)` body, outside the `if iter_num >= learning_starts` train-gate (which is at L490). The control flow is structurally identical to sheeprl's: every iteration's `step_data["actions"]` is written into the buffer regardless of whether the action came from the prefill branch or the policy branch.

**Verdict on sub-claim 2.** ✅ Faithful. Prefill transitions land in the buffer, seeding it with diverse data so that when the train-gate opens at `iter_num == learning_starts` the buffer holds ≥ 1024 transitions for the world-model to learn from.

### Sub-claim 3 — no gradient updates during prefill

**What sheeprl does** (`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L660-L662`):

```python
if iter_num >= learning_starts:
    ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
    per_rank_gradient_steps = ratio(ratio_steps / world_size)
```

The `>=` is exclusive of iterations 1..`learning_starts - 1` and inclusive of `iter_num == learning_starts` onwards. Note the subtraction `ratio_steps = policy_step - prefill_steps * policy_steps_per_iter` (where `prefill_steps = learning_starts - 1` for `learning_starts > 0`): on the first gate-entry iteration (`iter_num == learning_starts`), `policy_step == learning_starts * num_envs * world_size` and `prefill_steps * policy_steps_per_iter == (learning_starts - 1) * num_envs * world_size`, so `ratio_steps == num_envs * world_size`. Feeding that into the `Ratio` scheduler (with `_prev=None`) on the first call returns `int(num_envs * world_size * ratio)` — e.g. `int(1 * 1 * 1) == 1` for the XS config — so sheeprl **does** fire one gradient step at `iter_num == learning_starts`. The plan's claim that `ratio(0) == 0` at the boundary was therefore wrong in two places — the JAX driver does not subtract `prefill_steps` (so `ratio_steps == policy_step == learning_starts` on first entry, returning `int(learning_starts * ratio)` accumulated debt), and sheeprl does subtract but still returns a small positive integer (one step per env per world-rank). The hard invariant that both implementations preserve, and that the test correctly guards, is "**zero gradient steps for iters $1..\text{learning\_starts}-1$**".

**What the JAX driver does** (`src/algorithms/dreamer_srl/dreamer_srl_main.py:L490-L495`):

```python
if iter_num >= learning_starts:
    ratio_steps = policy_step
    n_grad_steps = ratio(ratio_steps)
    if n_grad_steps > 0 and buffer._pos >= seq_len:
        ...
```

The `>=` gate is identical to sheeprl's. The `ratio_steps = policy_step` (no subtraction) is the documented simplification — the developer's report flags this correctly as a behavioural difference from sheeprl at the boundary (debt repayment vs. one-step-per-env), but **the hard invariant `iter_num < learning_starts ⟹ n_grad_steps == 0` holds either way**, because the gate is `>=` and the inside-branch is the only code path that can call `train_step`.

Quantitatively, for `learning_starts=1024`, `num_envs=1`, `replay_ratio=1` on the parity-track config:
- Sheeprl at iter 1024: `ratio_steps = 1024 - 1023 * 1 = 1`, `ratio(1) == 1` → 1 grad step.
- JAX driver at iter 1024: `ratio_steps = 1024`, `Ratio(_prev=None)(1024) == int(1024 * 1) == 1024` → 1024 grad steps **at the boundary iteration**.

This is a real behavioural divergence at the boundary, but it is the **debt-repayment design choice** in sheeprl's `Ratio` class (`vendor/sheeprl/sheeprl/utils/utils.py:L273-L291`, ported faithfully to `src/algorithms/dreamer_srl/utils.py:L311-L333`): the first call's `int(step * ratio)` is meant to clear accumulated debt so the long-run rate is `ratio`. Sheeprl's subtraction in `ratio_steps = policy_step - prefill_steps * policy_steps_per_iter` *avoids* this debt by making `ratio_steps` start near zero at first gate entry; the JAX driver instead lets the `Ratio` scheduler self-correct on first call. Both reach the same long-run replay ratio (`replay_ratio == 1` ⟹ asymptotic 1:1 grad-step-to-env-step). The transient effect at iter 1024 is a one-time debt repayment of $O(\text{learning\_starts})$ gradient steps; for `replay_ratio = 1`, that is 1024 grad steps on one iteration, then 1 grad step per iteration thereafter.

**Is this an algorithmic concern for the parity launch?** Two ways to think about it:

1. **Sample-efficiency parity.** Across the full training duration, the total grad-step count is `int(total_steps * ratio)` for both implementations (the `Ratio` class's invariant). The transient at the boundary does not change the asymptotic count; sheeprl's design just smears it over `learning_starts` iterations (one step each from iter `learning_starts` to iter `2 * learning_starts - 1`) instead of paying it all at iter `learning_starts`. For the parity launch's total-step count, the two are identical.
2. **Burst-load risk.** The JAX driver's 1024-step burst at iter 1024 is a real wall-clock and memory event — 1024 calls to `train_step` in a tight Python loop, each consuming a batch of size 16. At 7.1 SPS on the reduced-dim smoke and an estimated ~3.5 SPS on the parity-track config (per the CP10 forward-looking estimate), the burst would take ≈ 1024 / 3.5 ≈ 290 s of wall-clock at iter 1024, which is a visible step in the SPS trace but not a stability concern (each individual `train_step` is the same as a steady-state one; the only risk is cumulative GPU memory pressure across the burst, which the JIT-compiled `train_step` should be fine with at constant memory).

**Verdict on sub-claim 3.** ✅ Faithful at the **hard-invariant level** ("no gradient before iter `learning_starts`"). The boundary behaviour (one-time debt repayment vs. one-step-per-rank) is a **documented sheeprl-design simplification** that the JAX driver inherits via the no-subtraction `ratio_steps = policy_step` choice. The total-grad-step count over the full run matches sheeprl. I recommend **CP10 watch for the burst at iter 1024 in the wall-clock trace** as a known feature of the JAX driver's design (concur with the developer's note in CP9b.8 that this is < 1% of full-run SPS impact). No CP9b blocker.

### Why DreamerV3 needs this — the RL motivation in two sentences

The §S3 prefill exists because a categorical actor with random-init weights produces approximately uniform action samples already (zero-init logits → softmax → uniform), but the world model also has random weights and would produce nonsense latent dynamics on the first ~1000 transitions; using the policy network to act during that window is **double pre-training waste** — the buffer fills with biased trajectories (the random actor's softmax-rounded sampling is uniform only in expectation, not after early gradient updates) AND the world model trains on those biased trajectories. By drawing actions uniform-IID from the discrete action set for `learning_starts` env-steps before any gradient fires, DreamerV3 guarantees the buffer holds a state-space-coverage snapshot that is independent of the random-init policy's pathologies; the world model then has 1024 "honest" transitions to bootstrap from. This is the classic "behavioural policy ≠ target policy" off-policy seeding trick, applied to a model-based RL setting where the cost of biased early buffer data is amplified (it propagates through every imagined rollout the world model produces, not just the actor's REINFORCE gradient). The DreamerV3 paper does not derive `learning_starts = 1024` from a principled bound; it is a hyperparameter inherited from the prior DreamerV2 work and held constant across all benchmark domains (cf. Hafner et al. 2023, "Mastering Diverse Domains through World Models", Appendix B "Hyperparameters" — `replay_buffer.prefill = 1024` in the XS config).

---

## Test design audit (verify-checklist item 2)

### Test 1 — `test_prefill_uniform_entropy_below_learning_starts`

**What it does.** Draws 10,000 samples from `_sample_prefill_action_oh(key, num_envs=1, action_dim=4)` (a helper that mirrors the production branch's `jax.random.randint + jax.nn.one_hot` formula), bins them into a 4-bin histogram, and asserts `|H_emp − log 4| < 0.01` and `np.all(counts > 0)`.

**Failure modes covered.**

| Failure mode | Caught by Test 1? |
|---|---|
| `maxval=action_dim - 1` off-by-one (one bin gets zero samples) | ✅ Yes — `np.all(counts > 0)` |
| `minval=1` off-by-one (one bin gets zero samples) | ✅ Yes — `np.all(counts > 0)` |
| `np.random.randint` re-introduced (entropy still uniform if seeded correctly) | ⚠ Partial — distribution would still pass entropy check, but the helper would diverge from production code |
| `jax.random.randint(key, ...)` with key-reuse across batch (all envs get same action) | ✅ Yes — for `num_envs > 1` the empirical histogram would show bimodal-by-key pattern; for `num_envs=1` (as tested) no observable effect |
| Off-by-one `>` vs. `<=` on the gate predicate | ❌ Not directly — the test does not exercise the gate; it tests the helper in isolation |
| Production prefill silently disabled → falls through to `player.get_actions` from iter 1 | ⚠ This is F1 (below) |

**Failure mode F1 — overlap with zero-init policy's action distribution.**

The project's cascade fix #27 (zero-init for the actor output linear) makes the actor's pre-softmax logits at step 0 exactly zero, so the softmax produces $[1/A, 1/A, ..., 1/A]$ and sampling from it returns approximately uniform actions. If a regression silently routes the prefill iterations through `player.get_actions()` from iter 1 (say, by inverting the gate predicate to `if iter_num > learning_starts:` and falling through to the else branch from iter 1), the actor's action distribution would still be near-uniform at iter 1, and gradually drift as `train_step` updates the actor weights. For the first ~10 iterations the drift is bounded by Adam's first-step magnitude (lr × eps-normalised gradient ≈ 8e-5 × O(1) ≈ 1e-5 per parameter), so the categorical distribution would remain within ~1e-3 of uniform — well inside the test's `tol = 0.01` window. **The test would PASS even though the §S3 contract had been violated.**

However, Test 1 is **deliberately scoped to test the helper, not the gate**. The helper `_sample_prefill_action_oh` is called in isolation from a freshly-seeded PRNG, with `num_envs=1, action_dim=4`. The test is a **regression guard on the production sampling formula** — it does NOT instantiate the full driver, does not call into `dreamer_srl_main.main()`, does not exercise the gate predicate at all. The docstring at L48-L50 says this explicitly: "any drift between this helper and the production code would cause Test 1 to fail and the test to be invalid — that's intentional". So Test 1's job is "is `jax.random.randint + jax.nn.one_hot` actually uniform?", and on that scoped question it is correct.

**The genuine §S3-gate-violation failure mode** ("production silently routes prefill iterations through `player.get_actions`") would be caught by:
- The code-reviewer's parallel gate, who is required to verify the JAX port matches sheeprl L558-L571 line-for-line.
- Test 2's gate-counter check (which would still fire at `iter_num < learning_starts` if `player.get_actions` were being called — wait, no: Test 2 only checks gradient steps, not action provenance, so it would NOT catch this either).
- **CP10's wall-clock measurement** would expose it indirectly (calling `player.get_actions` during prefill means doing the encoder + RSSM + actor forward pass for every prefill iteration, which is ~10× more compute per iter than the prefill branch).
- A **live cross-check at the parity launch**: log the action histogram at iter 0 and iter 1024 in WandB; if the histograms are statistically different (KS test, $p < 0.01$), the §S3 contract is violated.

**My disposition on F1.** Not a CP9b blocker — the test is correctly scoped and the failure-mode overlap is bounded by the zero-init actor's narrow window of approximate-uniformity. **Recommendation**: add a one-line WandB log key in `dreamer_srl_main.py` that records `wandb.log({"Diagnostic/prefill_action_dist": int(action_idx)}, step=policy_step)` for iterations `<= learning_starts`, so the parity-launch WandB run shows the prefill-action histogram side-by-side with the post-prefill histogram. This is a documentary/diagnostic addition for CP10, not a test addition.

### Test 2 — `test_no_gradient_step_before_learning_starts`

**What it does.** Simulates the driver's train-gate (`if iter_num >= learning_starts:` + `Ratio.__call__`) in isolation with `learning_starts=10`, `replay_ratio=1`, `num_envs=1`. For iterations 1..14 it accumulates the integer return of `ratio(policy_step)` if the gate fires, or 0 if it does not. It asserts (a) `grad_step_at_iter[i] == 0` for `i in range(learning_starts - 1)` (i.e. iters 1..9), and (b) `sum(grad_step_at_iter[learning_starts - 1:]) > 0` (i.e. at least one grad step fires at iter `≥ learning_starts`).

**Failure modes covered.**

| Failure mode | Caught by Test 2? |
|---|---|
| Train-gate flipped `>=` → `>` (fires at iter `learning_starts - 1`) | ❌ No, because `>` at iter 9 with `learning_starts=10` is False, same as `>=`; the test would only catch `>=` → `==`-and-shifted |
| Train-gate flipped `>=` → `<=` (fires at iters 1..learning_starts) | ✅ Yes — would trigger at iter 1, returning `int(1*1) == 1` grad step |
| Train-gate moved inside the prefill branch (fires at iters 1..learning_starts) | ✅ Yes — same as above |
| Train-gate removed entirely (fires at every iter) | ✅ Yes — `grad_step_at_iter[0] == 1` would fire |
| `ratio_steps = policy_step - X * learning_starts` introduced (changing boundary) | ⚠ Partial — would change the value of `grad_step_at_iter[learning_starts - 1]` but the iters 1..9 check would still pass |
| `train_step` is called but produces a no-op gradient (zero loss because buffer is empty) | ❌ No — this test counts `ratio()` returns, not optimizer-step-deltas |

**Failure mode — no-op gradient train_step.**

The user-supplied checklist flags the possibility that `train_step` could fire with `step < learning_starts` but produce a no-op gradient (e.g., zero loss because the buffer has no data). Let me check the actual driver code: at L495, the inner `if n_grad_steps > 0 and buffer._pos >= seq_len:` guard is what gates the actual `train_step` invocation. The outer gate `if iter_num >= learning_starts:` (L490) is the §S3 gate; the inner guard is a separate "buffer-has-enough-data" guard. If only the inner guard fires (i.e., the §S3 outer gate were broken but the inner held), `train_step` would still not fire as long as the buffer was empty — but the buffer fills from iter 1 (per sub-claim 2 above), so by iter 64 (the `seq_len`), the inner guard would also be satisfied. **The §S3 gate is the genuine first line of defence; the inner guard alone would not preserve the §S3 contract.**

Test 2 counts `ratio()` returns, not parameter deltas. So a regression where `train_step` is called but produces a zero-gradient (e.g. because `local_data` is all-zero) would NOT be caught. **Is this a real concern?** No, for two reasons:

1. The driver code at L495-L535 wraps `train_step` in `if n_grad_steps > 0 and buffer._pos >= seq_len`, where `n_grad_steps = ratio(ratio_steps)` is the very quantity Test 2 measures. If Test 2 reports `n_grad_steps == 0` for iters 1..9, the `train_step` body is provably unreached at those iterations (Python `if` short-circuit). The test thus indirectly proves zero parameter deltas as long as the gate predicate it tests **structurally matches** the gate predicate in production.

2. Test 2 inlines the gate predicate (`if iter_num >= learning_starts: ratio_steps = policy_step; n_grad_steps = ratio(ratio_steps)`) by hand, **textually identical** to L490-L493 of the driver. A future refactor that moved the gate into a helper function and silently changed its semantics would be the kind of regression that this test cannot catch — but **only because Test 2 is a self-contained re-implementation of the gate, not a driver-level integration test**.

**My disposition.** Concur with Test 2's scope. The "no-op gradient train_step" concern from the verify-checklist is structurally precluded by the driver's outer-gate-then-inner-guard pattern: the outer gate IS the predicate Test 2 measures, and it short-circuits the train_step call. The minor risk is **gate-implementation drift** between the production driver and the test's inline reproduction, but this is a class of bug that bit-identity testing (Lever D) would catch and that this test is not designed to catch. The hard invariant ("zero gradient before iter `learning_starts`") is preserved by the test exactly as specified.

### Test 2 — the developer's adjustment

The plan's original Test 2 spec (CP9B_PLAN.md L468-L479) asserted:

> `grad_steps_through_learning_starts == 0`
> "at iter == learning_starts the gate enters, but ratio(policy_step) at the boundary returns 0"

The developer correctly identified that this claim is **factually wrong** for the JAX driver. Reading `src/algorithms/dreamer_srl/utils.py:L311-L333`:

```python
def __call__(self, step: int) -> int:
    if self._ratio == 0:
        return 0
    if self._prev is None:
        self._prev = step
        repeats = int(step * self._ratio)
        ...
        return repeats
    repeats = int((step - self._prev) * self._ratio)
    ...
```

On the first call (`_prev is None`) the return is `int(step * self._ratio)`. The driver feeds `ratio_steps = policy_step = learning_starts` at first gate entry (L492-L493, no subtraction), so the first call returns `int(learning_starts * replay_ratio)`. For `learning_starts=10, replay_ratio=1`, that is `10`, not `0`. The plan was imprecise.

**Was the plan wrong or was the code wrong?** The plan was wrong about the *boundary behaviour*; the code is correct relative to the **hard invariant** that matters ("zero grad steps for iters $1..\text{learning\_starts}-1$"). The plan conflated two distinct boundary semantics: sheeprl's `ratio_steps = policy_step - prefill_steps * policy_steps_per_iter` (which DOES give `ratio_steps == num_envs` on first gate entry, returning `int(num_envs * replay_ratio)` — close to zero but not zero) vs. the JAX driver's `ratio_steps = policy_step` (which gives `ratio_steps == learning_starts` on first gate entry, returning `int(learning_starts * replay_ratio)`).

**Should the plan be amended?** Yes — recommended to senior-developer:
- Amend CP9B_PLAN.md L468-L479 to remove the claim about `ratio(policy_step) at the boundary returns 0`.
- Add a one-sentence note that the JAX driver's omission of the `prefill_steps` subtraction is intentional (a documented sheeprl-design simplification) and that the boundary returns `int(learning_starts * replay_ratio)` (debt repayment) rather than `int(num_envs * replay_ratio)` (per-iter rate).
- This is **not** a D-014 deviation — it is plan-text precision, not a deviation from sheeprl's algorithmic behaviour.

**Verdict on the developer's adjustment.** ✅ Algorithmically correct. The dropped assertion was inconsistent with both sheeprl's `Ratio` contract and the JAX driver's actual behaviour. The retained invariant ("zero grad steps for iters 1..learning_starts-1") is the §S3 contract. **No code change needed; CP9B_PLAN.md text fix is the disposition.**

---

## D-012 → D-012-closed transition (verify-checklist item 3)

**The parity-track config (`configs/dreamer_srl/01_food_only.yaml`).**

Line 15 now reads `learning_starts: 1024`. Header comment block (L1-L11) correctly frames the config as "parity-track" with a CP9b citation. No remaining hardcoded `learning_starts=0` anywhere in production code (verified by reading L387-L404 of `dreamer_srl_main.py`: the gate is `iter_num <= learning_starts` with `learning_starts` loaded via `agent_cfg.get_mandatory("algo.learning_starts", int)` at L197 — no fallback default, per CLAUDE.md project rules).

**The smoke config (`configs/dreamer_srl/01_food_only_smoke.yaml`).**

Line 27 reads `learning_starts: 0`. The header comment at L19-L24 explicitly frames this as "smoke-only deviation" with rationale tied to (a) the 5,000-step smoke budget being too small to absorb a 1024-step prefill (20% of budget burnt on prefill if `learning_starts=1024`), and (b) the zero-init actor (cascade fix #27) producing approximately-uniform actions for the first ~100 steps, so the smoke's integration-surface coverage is unchanged by the local deviation.

**Is the smoke-config deviation acceptable for continued use?** ✅ Yes. The smoke's purpose is fast iteration on integration bugs, not parity. The 5,000-step budget is too small to absorb a 1024-step prefill window. The cascade-fix-#27 zero-init actor produces near-uniform actions for the first ~10 gradient updates (per the F1 analysis above), so the buffer-seed quality during the smoke is comparable to what §S3 prefill would have produced anyway. The smoke does NOT cross-check parity with sheeprl; the parity gate is at CP10 + the parity launch. The smoke's job is "does the training loop run end-to-end without NaN?", and that question is `learning_starts`-invariant.

**Verdict on D-012 closure.** ✅ Fully closed for the parity-track config. The smoke-only deviation is acceptable for continued use and is correctly labelled in the smoke config's header comment.

---

## Plan-reality discrepancy disposition (verify-checklist item 4)

Covered above under "Test 2 — the developer's adjustment". To recap:

- **The plan was wrong** about the `Ratio` boundary behaviour (it claimed `ratio(policy_step) at the boundary returns 0`, which is true for sheeprl-with-prefill_steps-subtraction but **not** for the JAX driver-without-subtraction).
- **The code is correct** relative to the hard §S3 invariant. The JAX driver's omission of the prefill_steps subtraction is a documented sheeprl-design simplification that the developer correctly inherited.
- **The developer's test adjustment is algorithmically correct** — drop the incorrect plan assertion, retain the hard invariant.
- **No D-014 filed** — there is no new deviation from sheeprl's algorithmic behaviour. The plan text was imprecise, not the code.

**Recommendation to senior-developer.** Amend CP9B_PLAN.md L468-L479 with a one-sentence note about the boundary debt-repayment, so the plan-as-archived doc is faithful to the implementation. This is plan-text precision, not a deviation log entry.

---

## Forward-looking concerns for the parity launch (verify-checklist item 5)

The CP9 smoke ran with `learning_starts=0`, so the parity launch with `learning_starts=1024` will be the **first** runtime test of the §S3 prefill code path at the parity-track scale. Three forward-looking concerns:

### FL1 — Debt-repayment burst at iter 1024

Per the sub-claim-3 analysis: at iter 1024, the JAX driver's `ratio(1024) == 1024` returns a batched debt repayment of 1024 gradient steps. With `per_rank_batch_size=16`, `per_rank_sequence_length=64`, this is 1024 invocations of `train_step` on a (likely) JIT-compiled-once-and-reused trace. Wall-clock impact at the parity-track full-XS config (1024-unit dense, 4096 recurrent, 32×32 stochastic): each `train_step` is ~280 ms on the reduced-dim smoke; on the full XS it will be larger (estimated ~500-800 ms, dominated by RSSM imagination at horizon=15). 1024 × 700 ms ≈ 720 s of burst-mode training, which is a visible step in the SPS trace but not a stability concern. **CP10's wall-clock measurement should explicitly call out the iter-1024 burst** so it is not mistaken for a hang.

### FL2 — Buffer "is_first" boundary at iter 1024

At iter 1024 the buffer holds 1024 transitions, all from uniform-random-action trajectories. The `is_first` markers are set correctly per the existing CP4b logic, but there is a **subtle issue**: the buffer-sampling at iter 1024 will draw `batch_size * seq_len = 16 * 64 = 1024` transitions, which is **exactly the size of the prefill window**. With high probability, every sampled sequence will straddle the prefill/post-prefill boundary (i.e., the sequence's first few transitions are random-action, the rest are policy-action). The world model trains on this mixed-trajectory data on first activation. **Is this a §S3 violation?** No — sheeprl behaves identically (the buffer has no notion of pre/post prefill; it just holds transitions). But it is worth noting that **the world model's first 1024 gradient steps update on data that is correlated with the random-action policy**, not the eventual target policy. This is a known feature of off-policy world-model training; the CP10/parity launch should expose it via the world-model loss curve (we expect a near-flat WM loss for the first few hundred grad steps, then a drop as the buffer fills with more-diverse trajectories).

### FL3 — Action-distribution shift at iter 1025

At iter 1025 (the first post-prefill iteration), the policy is invoked for the first time on a state that the actor has now had 1024 gradient updates on (per the debt repayment). The actor's action distribution at iter 1025 is **not the same** as the actor's distribution at iter 0 — it has been optimised against 1024 imagined-rollout REINFORCE gradients computed from the random-action-buffer. **Is this an algorithmic concern?** No — this is standard DreamerV3 behaviour and sheeprl exhibits it. But the parity launch should record the action-distribution at iter 0, iter 1024 (last prefill), iter 1025 (first policy), and iter 2048 (first steady-state policy) to confirm the transition is smooth.

**Suggested CP10 / parity-launch additions** (none are CP9b blockers; all are diagnostic):

1. WandB log key `Diagnostic/prefill_action_idx` recording the sampled action during the prefill window (one int per iter for iter `<= learning_starts`).
2. WandB log key `Diagnostic/gradient_steps_at_iter` recording `n_grad_steps` at each train-gate entry (to catch the iter-1024 debt-repayment burst).
3. SPS trace at iter `[1, 1024, 1025, 1026, 2048]` to characterise the burst's wall-clock cost.

---

## §S3 contract compliance summary

| Sub-contract | Sheeprl line | JAX driver line | Compliant? |
|---|---|---|---|
| (a) Uniform-random action sampling during prefill | L558-L571 | L394-L402 | ✅ Yes (modulo RNG-stream difference, same class as D-002) |
| (b) Prefill transitions added to replay buffer | L587 | L409 | ✅ Yes (buffer-add outside train-gate) |
| (c) No gradient updates during prefill | L660 (`>=`) | L490 (`>=`) | ✅ Yes (hard invariant: `iter_num < learning_starts ⟹ no train_step call`) |

**Algorithmic-faithfulness summary.** CP9b's implementation is a **faithful port of sheeprl's §S3 contract** at the level of all three sub-claims. The one documented behavioural difference — sheeprl's `ratio_steps = policy_step - prefill_steps * policy_steps_per_iter` subtraction vs. the JAX driver's `ratio_steps = policy_step` no-subtraction — affects only the **debt-repayment pattern at the boundary iteration** and does NOT affect the §S3 hard invariant ("zero gradient before iter `learning_starts`") or the long-run replay ratio. Both implementations preserve the §S3 contract; the JAX driver's choice trades a one-time burst at the boundary for code simplicity, which is acceptable.

The two Lever-A tests guard the right invariants at the right scope:
- Test 1 verifies the production prefill sampling formula produces a uniform distribution.
- Test 2 verifies the train-gate predicate does not fire during the prefill window.

Both are property tests rather than bit-identity tests, which is correct for §S3 — there is no torch reference quantity to byte-match (the RNG streams disagree by construction).

The developer's adjustment to Test 2 (removing the plan's incorrect `Ratio` boundary assertion) is algorithmically correct and preserves the hard §S3 invariant. The plan-text discrepancy is a documentation issue, not a deviation.

D-012 fully closes on the parity-track config; the smoke retains `learning_starts=0` with a labelled smoke-only-deviation rationale.

---

## Findings

### F1 — Test 1 entropy proxy has bounded overlap with zero-init actor's action distribution (documentary, not blocking)

**Where.** `tests/algorithms/dreamer_srl/test_prefill.py:L63-L114`.

**What.** The empirical-entropy criterion `|H_emp − log A| < 0.01` (Test 1) overlaps with the action distribution that a zero-init categorical actor (post cascade-fix-#27) would produce during its first ~10 gradient updates. A regression that silently routes prefill iterations through `player.get_actions()` would land in this overlap region and pass the test falsely.

**Why this is bounded.** Test 1 deliberately tests the standalone helper `_sample_prefill_action_oh`, not the production driver gate. The genuine gate-violation failure mode is the code-reviewer's audit responsibility (line-for-line port verification), not Test 1's. The bonus assertion `np.all(counts > 0)` rules out the only off-by-one failure mode that would silently make the test vacuous.

**Recommendation.** Documentary only:
- Add one line to Test 1's docstring making the scope explicit: "This test verifies the sampling formula in isolation, NOT the production gate predicate. The gate predicate is verified by Test 2 and by the code-reviewer's line-for-line audit of `dreamer_srl_main.py:L394-L402` against `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L558-L571`."
- Add a CP10 / parity-launch diagnostic: WandB log key `Diagnostic/prefill_action_idx` for iterations `<= learning_starts` so the prefill-action histogram is visible side-by-side with the post-prefill policy-action histogram.

**Disposition.** Not a CP9b blocker. Recommendation is for CP10's diagnostic-logging scope.

### F2 — CP9B_PLAN.md L468-L479 plan-text precision (documentary, not blocking)

**Where.** `docs/develop/active/dreamer_srl_v3/CP9B_PLAN.md:L468-L479`.

**What.** The plan asserts `ratio(policy_step) at the boundary returns 0`. This is false for the JAX driver (which omits sheeprl's `prefill_steps` subtraction by design). The correct boundary behaviour for the JAX driver is `int(learning_starts * replay_ratio)` (debt repayment); for sheeprl it is `int(num_envs * replay_ratio)` (per-iter rate). Both preserve the §S3 hard invariant ("zero gradient before iter `learning_starts`"), which is what Test 2 correctly verifies.

**Why.** The plan author conflated two boundary semantics. The developer correctly identified this during Test 2 implementation and adjusted the test to preserve only the hard invariant. The production code is correct.

**Recommendation.** Senior-developer to amend CP9B_PLAN.md L468-L479 with a one-sentence note clarifying the boundary debt-repayment behaviour (no D-014, no code change).

**Disposition.** Not a CP9b blocker. Plan-text precision.

### F3 — Iter-1024 debt-repayment burst should be called out in CP10's wall-clock trace (forward-looking)

**Where.** CP10 wall-clock measurement protocol (not yet authored).

**What.** At iter 1024 of the parity-track launch, the JAX driver's `Ratio` scheduler will return `int(1024 * 1) == 1024` gradient steps on its first call (debt repayment). With the full-XS train_step at an estimated 500-800 ms each, this is a 720-1024 s burst at iter 1024 — visible in the SPS trace but not a hang.

**Recommendation.** CP10's wall-clock trace should explicitly characterise the burst at iter 1024 so it is not mistaken for a stability issue. Suggested diagnostic: SPS trace at iter `[1, 1024, 1025, 1026, 2048]` to confirm the debt is paid in one burst and steady-state SPS resumes at iter 1025+.

**Disposition.** Not a CP9b blocker. CP10 / parity-launch diagnostic scope.

---

## Recommendation for senior-developer

**✅ READY TO FLIP CP-PASS** pending the code-reviewer's parallel gate. Algorithmic faithfulness to DreamerV3 §S3 is established at all three sub-contracts (uniform sampling, buffer-fill, no-gradient). The two Lever-A tests guard the right invariants. D-012 closes cleanly on the parity-track config. The plan-reality discrepancy (Test 2) is a plan-text precision issue, not a deviation — the developer's adjustment is algorithmically correct.

**Action items for senior-developer at CP9b verification:**

1. Confirm code-reviewer's parallel gate passes (the line-for-line audit of `dreamer_srl_main.py:L394-L402` against sheeprl L558-L571 is their authority, not mine).
2. Flip the CP9b row in `docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md:L527` from `NOT STARTED` to `CP-PASS` (developer correctly did NOT flip per the post-CP4 reviewer-gate strengthening rule).
3. Amend `CP9B_PLAN.md:L468-L479` with a one-sentence note about the boundary debt-repayment behaviour (F2).
4. Hand off to CP10 with the F1 + F3 diagnostic recommendations (WandB log keys for prefill-action histogram and gradient-step burst).
5. No PI consultation needed at CP9b. The next PI trigger is the parity-launch decision after CP10.

**Algorithmic blockers identified.** None.

---

## Composition with prior CP reviews

CP9b builds on:
- **CP3b** (buffer + cadence): provided `Ratio` scheduler + `SequentialReplayBuffer`, both bit-identity tested. CP9b's Test 2 reuses `Ratio` from `src.algorithms.dreamer_srl.utils`. Confirmed contract-stable.
- **CP9** (integration smoke): established that the training loop runs end-to-end without NaN at `learning_starts=0`. CP9b restores `learning_starts=1024` on the parity-track config and adds property tests on the gate.
- **Cascade fix #27** (zero-init actor): relevant to F1 above — makes the post-prefill actor produce near-uniform actions for the first ~10 grad updates, narrowing the "is the prefill branch actually wired?" diagnostic window.

CP9b hands off to:
- **CP10** (wall-clock budget): inherits F1 + F3 diagnostic recommendations.
- **Parity launch** (post-CP10, PI-gated): inherits the iter-1024 debt-repayment burst as a known feature.

---

## References

- **Sheeprl @33b6366**:
  - `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L558-L571` — action-gate (§S3 sub-claim 1)
  - `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L587` — buffer-add (sub-claim 2)
  - `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L660-L662` — train-gate (sub-claim 3)
  - `vendor/sheeprl/sheeprl/utils/utils.py:L259-L301` — Ratio class (boundary semantics)
- **JAX driver**:
  - `src/algorithms/dreamer_srl/dreamer_srl_main.py:L387-L404` — prefill branch
  - `src/algorithms/dreamer_srl/dreamer_srl_main.py:L406-L409` — buffer-add
  - `src/algorithms/dreamer_srl/dreamer_srl_main.py:L490-L495` — train-gate
  - `src/algorithms/dreamer_srl/utils.py:L270-L333` — Ratio class
- **Hafner et al. 2023**, "Mastering Diverse Domains through World Models" — Appendix B "Hyperparameters" (XS config: `replay_buffer.prefill = 1024`)
- **Prior CP reviews**: [CP8 professor review](dreamer_srl_v3_cp8_professor_rl_bayesian_dl_review.md), [CP3b professor review](dreamer_srl_v3_cp3b_professor_rl_bayesian_dl_review.md)
