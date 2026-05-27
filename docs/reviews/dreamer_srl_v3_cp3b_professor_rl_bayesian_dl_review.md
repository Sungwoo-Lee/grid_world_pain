---
title: "dreamer-srl v3 — CP3b Algorithm-Fidelity Review (professor-rl-bayesian-dl)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/buffers.py
---

# dreamer-srl v3 — CP3b Algorithm-Fidelity Review

## Plain-language verdict

**What this gate is.** Third (and final technical) reviewer gate on the buffer +
cadence checkpoint (CP3b) of the dreamer-srl v3 rebuild. The rebuild ports a
community PyTorch DreamerV3 reference (`sheeprl`, pinned at commit `33b6366`)
into JAX, in numbered checkpoints. CP3b ports the *replay buffer* (how every
transition is stored — `obs`, `action`, `reward`, `done`, `is_first`,
`terminated`, `truncated`) plus the *training cadence* (how many gradient
updates fire per environment step, gated by `learning_starts` and modulated by
`replay_ratio`). The code reviewer confirmed the JAX file is mechanically a
line-for-line port; the math reviewer confirmed all 8 equations match
sheeprl@`33b6366`. My job is the **algorithm-level fidelity** — does the buffer
+ cadence produce *bytes that the downstream world-model rollout actually
needs*, at the *time index it expects*, with the *frequency it expects*?

**Why this matters.** The buffer is upstream of every algorithmic checkpoint
that follows. If `is_first` is stored at the wrong time index, the
three-quantity recurrent-state reset at CP4b (the §S4 fix that took the v2
reviewer chain three rounds to nail down) will silently never fire — the
RSSM's `is_first`-masked reset code will run, but the marker it looks for
will never be in the sampled window. If actions are stored at the wrong time
index, the §S2 prepend-zero shift at CP2b — which says "the action consumed
at time `t` is the action that *led to* `obs_t`, not the action taken *at*
`obs_t`" — will produce one-step-misaligned training data. And if the
cadence wiring is wrong, the 16× `replay_ratio` drift that motivated promoting
this layer to a real gate will recur.

**Headline.** The buffer + cadence layer is algorithmically faithful. The four
§S items most relevant to CP3b (§S1, §S2, §S3, §S4) have their substrate
correctly laid here, with their downstream consumption deferred to the right
CPs (CP4b, CP2b, CP9b). The 8 critical algorithm-points raised in the brief
are all either covered by the existing tests or covered by configuration
invariants that *cannot fail* under sheeprl XS semantics. Two surface
clarifications about test scope (Test 6's `collect_interval` invariance,
Test 3's straddling-window correctness) and one architectural note about
where §S3's *random-action prefill* lives — fold into hand-off; no blockers.

**Verdict: PASS.** PI gate fires next on D-004 + D-005.

---

## 1. §S-rule cross-reference table (CP3b scope)

The v2 plan's `Training-loop semantics` section catalogues 10 silent
training-loop omissions (§S1–§S10). CP3b is upstream of four of these. The
table shows which §S substrate is *laid* at CP3b and where it is *consumed*
downstream.

| § | Silent semantic | What CP3b must guarantee | Where consumed downstream | Status at CP3b |
|---|---|---|---|---|
| **§S1** | Force-set `is_first[0]=1` on every sampled chunk (`dreamer_v3.py:133`) | Buffer must STORE `is_first` byte-faithfully so that (a) `is_first[1:]` of the sampled chunk reflects the buffer's authentic stored values — these are what CP4b's reset reads — and (b) the `is_first=1` flag at the new-episode post-done position is present and findable | CP4b's `RSSM.dynamic` reads `batch["is_first"][0]=1` (force-set there) and `batch["is_first"][1:]` (authentic from buffer) for the three-quantity reset | **Substrate correct.** Tests 1+3+4 jointly cover: byte-identical `is_first` storage (Test 1), `is_first[t=500]=1` after a done at `t=499` (Test 3), per-env-lane independence (Test 4). |
| **§S2** | Prepend-zero-action shift (`dreamer_v3.py:137`) — action consumed at time `t` is the action taken at `t-1` | Buffer must STORE actions at the correct time index — specifically, `step_data["actions"] = actions` is set *before* `rb.add(step_data)` and *before* `envs.step(...)`, so the stored action at the *t-th* buffer slot is the action that *led to* `obs_{t+1}`, not the one taken *at* `obs_t` | CP2b's shift `cat([zeros[:1], actions[:-1]])` consumes this — assumes one-step lookahead alignment | **Substrate correct.** Action-storage timing is determined at the *training loop's call site*, not in `buffers.py`. The call site is in `dreamer_v3.py:586-589`: `step_data["actions"] = actions.reshape(...)` → `rb.add(step_data)` → `envs.step(real_actions)`. Buffer-side `add()` is action-timing-agnostic — it stores whatever bytes the caller hands it, in time order. CP3b correctly delegates this to CP9b (the train.py wiring CP). |
| **§S3** | `learning_starts` random-action prefill + gradient-step gate (`dreamer_v3.py:604-617, 706`) | (a) Collection branch uses uniform random actions for `iter_num <= learning_starts`. (b) Gradient updates gate on `iter_num >= learning_starts` | CP9b owns (a); the train-loop outer iteration owns (b) | **Substrate correct, with explicit scope split.** Test 6 verifies (b) — the gradient-update gate, via the `(env_step, grad_step, per_rank_gs)` trace for 5000 iterations. (a) is correctly deferred to CP9b. See critical point 3 below for the deferral rationale. |
| **§S4** | `is_first` arithmetic-mask reset of three quantities — action, recurrent_state, posterior (with posterior pre-flattened `[B, S, D]` → `[B, S*D]`) (`agent.py:423-429`) | Same substrate as §S1: buffer must STORE `is_first` byte-faithfully so the RSSM's three-quantity arithmetic-mask reset has the marker to mask on | CP4b's `RSSM.dynamic` applies `(1 - is_first) * x + is_first * init` to all three quantities | **Substrate correct (same as §S1).** No additional CP3b-side guarantees beyond what §S1 already requires. |

**Out-of-scope at CP3b** (handled at later CPs): §S5 (true-continue splice at
imagination step 0 — CP6 actor); §S6 (discount weighting on actor + critic
losses — CP6); §S7 (advantage `low`-offset cancellation — CP6); §S8 (free-nats
floor element-wise before mean — CP5 `loss.py`); §S9 (`Independent(BernoulliSafeMode, 1)`
wrap + decoder `dims=1` — CP4); §S10 (continue target = `1 - terminated` —
CP4 / CP6).

---

## 2. Critical algorithm-points audit (8 numbered points from the brief)

### Point 1 — §S1 prerequisite: `is_first` storage byte-faithful

**The question.** CP4b will force-set `batch["is_first"][0] = 1.0` at chunk
start regardless of the stored value, but `is_first[1:]` must be the buffer's
authentic stored sequence for the three-quantity reset to fire on the
correct downstream slots.

**Audit.** Test 1 + Test 3 + Test 4 jointly cover this:

- **Test 1** drives both buffers with an identical 100-step deterministic
  `add()` sequence and asserts `_buf["is_first"][:_pos]` is byte-identical
  to sheeprl's stored bytes. With `max_abs_diff = 0.000e+00` (math reviewer
  confirmed via Eq. 1 ring-buffer wrap), the JAX buffer's `is_first` column
  is byte-identical to sheeprl's after the same input.
- **Test 3** specifically tests the *interesting* `is_first` placement:
  done at `t=499` → `is_first=1` at `t=500`. A 10-step window starting at
  `t=495` straddles the boundary; the test asserts `is_first[5] == 1` and
  all others `== 0`. This is precisely the case CP4b's reset needs to
  catch. (The math reviewer's Eq. 5 derivation `offset = (d+1) - s = 500
  - 495 = 5` confirms the placement.)
- **Test 4** tests cross-lane independence — env 0's `is_first` slot
  cannot be contaminated by env 1's stream.

**Verdict.** Test scope catches the §S1 substrate question completely. ✓

### Point 2 — §S2 prerequisite: action storage at the correct time index

**The question.** The §S2 shift assumes that at buffer slot `t`, the
stored action is the action that *led to* `obs_{t+1}` (not the action
taken *at* `obs_t`). Buffer-side `add()` cannot enforce this — it only
stores whatever the caller passes.

**Audit.** I read the sheeprl call site at `dreamer_v3.py:540-591`:

```python
# Line 540-547: initial step_data setup with is_first=1
obs = envs.reset(seed=cfg.seed)[0]
step_data["is_first"] = np.ones_like(step_data["terminated"])

# Per-iter inner loop:
# Line 558-585: select action given obs (uniform-random pre-learning_starts
#               or policy-driven post)
# Line 586-587: STORE action + add to buffer BEFORE env.step
step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
rb.add(step_data, validate_args=cfg.buffer.validate_args)

# Line 589-591: env.step (which consumes the action just stored)
next_obs, rewards, terminated, truncated, infos = envs.step(real_actions...)

# Line 594: post-step, is_first reset to zero
step_data["is_first"] = np.zeros_like(step_data["terminated"])

# Line 628-629: update step_data["observations"] to next_obs
for k in obs_keys: step_data[k] = next_obs[k][np.newaxis]
```

So at buffer-slot `t`: `(obs_t, action_t)` is stored, where `action_t`
is the action that will *transition from* `obs_t` to `obs_{t+1}`. The
§S2 shift `cat([zeros[:1], actions[:-1]])` then aligns this at training
time: at world-model rollout step `t`, the action consumed is the one
that *led to* the obs at that index. This is the canonical "action-at-t
caused obs-at-t+1" convention.

The JAX-side `buffers.py:add()` is action-timing-agnostic: line 153 reads
`data_len = next(iter(data.values())).shape[0]` and stores whatever dict
the caller passes, in time-order. The action-storage timing is therefore
the *caller's* responsibility — the buffer's add() correctly does not
re-order or shift anything.

**CP3b verdict on §S2 substrate.** The buffer-side `add()` is correct
(time-agnostic, no shifting). The call-site discipline is deferred to
CP9b. ✓

**Note for CP9b reviewer.** When CP9b lands `train.py`'s collection branch,
the order must be: (1) compute `actions` from `obs_t`, (2) set
`step_data["actions"] = actions`, (3) `rb.add(step_data)`, (4)
`envs.step(actions)` — in this order. CP9b's test should assert that
`step_data["observations"]` is also set BEFORE `rb.add()`, not after
`envs.step`. The sheeprl `dreamer_v3.py:628-629` updates `step_data[k]`
to `next_obs[k]` AFTER `rb.add()`, so the next iteration's `add()` stores
`(obs_{t+1}, action_{t+1})`. This is the alignment §S2 expects.

### Point 3 — §S3 cadence-wiring scope split

**The question.** §S3 has two halves: (a) collection-time random-action
prefill, (b) gradient-update gate. Does CP3b cover both, or only (b)?

**Audit.** Looking at sheeprl `dreamer_v3.py:558-585` (the action-selection
branch), the random-action prefill is in the *outer training loop's
collection branch* — `if iter_num <= learning_starts: actions = uniform_random(...)`.
This is `train.py`-level wiring, not buffer-level. The gradient-update
gate at line 660 (`if iter_num >= learning_starts:`) is also `train.py`-level
wiring, but its arithmetic substrate (the `Ratio(replay_ratio)(ratio_steps /
world_size)` formula with `prefill_steps = learning_starts - int(learning_starts > 0)`)
is what CP3b's Test 6 verifies.

So the scope split is: CP3b covers the *arithmetic of the gate*; CP9b
covers the *call-site wiring of both branches*.

**Test 6 verifies the arithmetic.** It walks a 5000-iter trace, asserts
`per_rank_gradient_steps == 0` for `iter_num < 1024` and the correct
non-zero values from `iter_num = 1024` onward. The 16× drift class (the
empirical motivator for promoting CP3b) IS the regression Test 6 catches.

**Test 6 does NOT verify the random-action prefill** — that's the
collection branch, not the cadence branch. CP9b is the correct CP for it
(the v2 plan's Checkpoint 9b explicitly tests random-action entropy
≈ `log(action_dim)` on policy steps 0..1023 and policy-driven entropy
decrease thereafter — see archived plan line 701).

**Verdict.** The scope split is correct and matches the v2 plan's
allocation. ✓ Recommend the hand-off note explicitly mentions this so PI
doesn't expect CP3b to cover both halves of §S3.

### Point 4 — Replay-ratio × per-update batch-size semantics

**The question.** Does the JAX cadence formula respect the sheeprl
semantics that `replay_ratio` modulates the *frequency* of gradient
updates, not the *batch size* of each update? Each update consumes one
`[T=seq_len, B=batch_size]` window regardless of `replay_ratio`.

**Audit.** Looking at sheeprl `dreamer_v3.py:660-698`:

```python
if iter_num >= learning_starts:
    ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
    per_rank_gradient_steps = ratio(ratio_steps / world_size)
    if per_rank_gradient_steps > 0:
        local_data = rb.sample_tensors(
            cfg.algo.per_rank_batch_size,           # fixed batch_size
            sequence_length=cfg.algo.per_rank_sequence_length,  # fixed seq_len
            n_samples=per_rank_gradient_steps,      # variable n_samples
            ...
        )
        for i in range(per_rank_gradient_steps):
            batch = {k: v[i].float() ... }          # ONE batch per gradient step
            train(...)
```

The `Ratio.__call__(ratio_steps / world_size)` returns an integer
`per_rank_gradient_steps`. This integer is passed as `n_samples` to
`rb.sample_tensors`, which returns a tensor of shape `[per_rank_gradient_steps,
seq_len, batch_size, ...]`. The training loop then iterates
`for i in range(per_rank_gradient_steps): train(batch[i])`. So:

- Per-update batch shape: `[T=seq_len=64, B=batch_size=16]` — **fixed**.
- Number of updates per iter: `per_rank_gradient_steps` — **variable**,
  determined by `Ratio(replay_ratio)`.
- With `replay_ratio=1.0, num_envs=1` (sheeprl XS): one update per env
  step ⇒ 64×16 = 1024 transitions consumed per 1 env step.
- With `replay_ratio=0.0625, num_envs=16` (cascade): one update per
  16 env steps ⇒ 64×16/16 = 64 transitions consumed per 1 env step.

JAX-side: `Ratio.__call__` (per CP1 review) is byte-identical to sheeprl's
`Ratio.__call__`. Test 6 verifies it produces the correct
`per_rank_gradient_steps` integer at each iteration. The per-update
batch-size is determined by `agent_xs.yaml:per_rank_batch_size=16` and
`per_rank_sequence_length=64` — Test 5 verifies these match sheeprl XS
defaults.

**Verdict.** The JAX cadence formula respects the
`replay_ratio = frequency, not batch-size scale` semantic. ✓

### Point 5 — Sample-window straddling-done semantics

**The question.** Sheeprl explicitly allows sampled windows to cross
episode boundaries; the `is_first=1` marker at the post-done position
triggers CP4b's RSSM reset. If JAX silently forbids straddling, the
reset code at CP4b would be dead. Does Test 3 actually exercise a
straddling sample with a post-done step inside the window?

**Audit.** Test 3's fixture builds a buffer with `done` at `t=499` and
`is_first=1` at `t=500`. The sampled window is `start_idx=495, seq_len=10`,
so it spans absolute time indices `[495, 496, 497, 498, 499, 500, 501,
502, 503, 504]`. Done is at offset 4 (= 499 - 495). `is_first=1` is at
offset 5 (= 500 - 495). Post-done steps at offsets 5, 6, 7, 8, 9 are
*inside* the window.

The test asserts both (a) byte-identity with sheeprl's output, AND (b)
`is_first[5] == 1` and `is_first[other] == 0`. Both confirm the
straddling sample is correctly formed.

Additionally, looking at the JAX `sample()` implementation
(`buffers.py:180-262`), the `valid_idxes` arithmetic at lines 236-246
(matching sheeprl `buffers.py:444-456` per math reviewer's Eq. 3) only
excludes "the chunk that would overlap `self._pos`" — it does NOT
exclude done indices. So the sampling path itself permits straddling at
runtime, not just under the explicit-index `_sample_at_indices` path
that Test 3 uses.

**Verdict.** Test 3 exercises a genuine straddling window with a
post-done step at window offset 5 (of 10). The §S1+§S4 substrate that
CP4b will rely on is *both* stored correctly AND can flow through the
sampling path. ✓

**Bonus catch.** If a future CP (e.g. a refactor) ever adds a
`don't-straddle-dones` flag to JAX `sample()`, this test would still
pass (because `_sample_at_indices` bypasses the policy), but the
production path would silently break CP4b. Recommend that CP4b's review
re-checks this by exercising production `sample()` with the same
fixture, not `_sample_at_indices`.

### Point 6 — `Ratio` integration with `collect_interval`

**The question.** Sheeprl XS does not have a `collect_interval` key, but
our cascade `dreamer_v3_rr06.yaml` had `collect_interval=128`. With a
different `collect_interval`, does the JAX cadence formula still produce
sheeprl-faithful behavior?

**Audit.** I grep'd both `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`
and `configs/models/dreamer_srl/agent_xs.yaml`: **neither has a `collect_interval`
key**. Sheeprl's outer-loop iteration directly maps to env-step
collection — `policy_step += policy_steps_per_iter` per iter
(`dreamer_v3.py:551`), with `policy_steps_per_iter = num_envs * world_size`.
There is no separate "collect this many env steps before considering a
gradient block" knob in sheeprl's XS path.

The `collect_interval=128` knob in our cascade `dreamer_v3_rr06.yaml`
is a *legacy in-house extension* that does not exist in the sheeprl
reference. CP3b's design — replicate sheeprl XS faithfully — correctly
omits this knob. The `agent_xs.yaml` has no `collect_interval` key
(I confirmed by grep). Test 5's mandatory-key audit also does not check
for it, because it would be wrong to require something sheeprl XS
doesn't have.

**Test 6's scope coverage under sheeprl semantics.** With sheeprl XS's
single-env-step-per-iter cadence (`policy_steps_per_iter=1`), Test 6's
trace at `total_iters=5000` covers `policy_step ∈ [1, 5000]` —
significantly past `learning_starts=1024`, exercising both the prefill
phase AND the steady-state Ratio-driven cadence. The 16× drift class
that motivated CP3b *was* exactly the "wrong collect_interval +
Ratio integration" failure mode, and Test 6 directly catches its
analog in the sheeprl-XS-semantics regime.

**Where the gap remains.** If a future plan adds a `collect_interval`
extension on top of dreamer-srl (e.g. to match a non-sheeprl experiment
setup), Test 6 would not catch a `Ratio + collect_interval` mis-integration
because the fixture doesn't exercise that regime. This is *intentional*:
CP3b's binding contract is sheeprl-XS parity, not generality. The
hand-off note should flag that any future `collect_interval` extension
would need a *separate* CP3c-style test, not a CP3b retrofit.

**Verdict.** Test 6 is correctly scoped to sheeprl XS semantics; the
`collect_interval` concern is out of scope by design. ✓

### Point 7 — Buffer warmup vs `learning_starts`

**The question.** The buffer needs at least `seq_len=64` transitions
before the first sample can succeed. With `learning_starts=1024` and
`seq_len=64`, the first sample at iter 1024 sees a buffer with
≥ 1024 transitions; `1024 >> 64` so this is safe. But if
`learning_starts < seq_len`, the algorithm would sample from an
under-filled buffer.

**Audit.** Looking at sheeprl `buffers.py:430-432` (mirrored at JAX
`buffers.py:224-228`):

```python
if not self._full and self._pos - sequence_length + 1 < 1:
    raise ValueError(
        f"Cannot sample a sequence of length {sequence_length}. "
        f"Data added so far: {self._pos}"
    )
```

The `sample()` method *explicitly raises ValueError* if the buffer doesn't
have `≥ seq_len` transitions yet. The JAX port preserves this check
verbatim. So `learning_starts < seq_len` would not silently corrupt
training — it would raise loudly at the first `sample()` call.

Additionally, sheeprl XS pins `learning_starts=1024, seq_len=64` —
the `>>` relationship — so the issue is absent under XS semantics.
Test 5's mandatory-key audit locks in `learning_starts=1024, per_rank_sequence_length=64`,
making it impossible to silently configure `learning_starts < seq_len`.

**Verdict.** Both the runtime guard (ValueError) and the configuration
invariant (`learning_starts=1024 ≫ seq_len=64`) protect against the
under-filled-buffer case. ✓

### Point 8 — Parallel-env lane invariant (no cross-env trajectory windows)

**The question.** A sampled trajectory window of length `seq_len` must
come from contiguous transitions within ONE env's lane. The RSSM's
recurrent state is per-trajectory; cross-env sequences would break the
recurrent semantic.

**Audit.** Looking at JAX `_get_samples` lines 305-315 (matching sheeprl
`buffers.py:480-489` per math reviewer's Eq. 4):

```python
if self._n_envs == 1:
    env_idxes = np.zeros((np.prod(batch_shape),), dtype=np.intp)
else:
    env_idxes = self._rng.integers(0, self._n_envs, size=(batch_shape[0],), dtype=np.intp)
    env_idxes = np.reshape(env_idxes, (-1, 1))
    env_idxes = np.tile(env_idxes, (1, sequence_length))   # ← tile over seq_len
    env_idxes = np.ravel(env_idxes)

flattened_idxes = (flattened_batch_idxes * self._n_envs + env_idxes).flat
```

The key line is `env_idxes = np.tile(env_idxes, (1, sequence_length))`:
*one env index is sampled per sequence, then tiled across the full
sequence_length*. So for sequence `i`, every one of its `seq_len`
elements uses the same `env_idxes[i]`. The flat-index formula
`flat = t * n_envs + e` then selects every element from env-column `e`
across the time axis. There is no cross-env splicing.

**Test 4 verifies this empirically.** With 4 envs and per-env sentinels
`{1.0, 2.0, 3.0, 4.0}`, Test 4 calls `_sample_at_indices` with `env_idxes`
locked to one column, asserts `np.allclose(obs_flat, sentinel)` — every
element of every sampled window must equal the lane sentinel. With
`batch_size=10, seq_len=8, obs_dim=8` that's 640 floats per lane × 4 lanes
= 2,560 floats, every one of which must match its lane's sentinel. A
cross-lane leak at any time index in any lane would fail loudly.

**Verdict.** The lane invariant is laid in the buffer's `_get_samples`
formula (tile env_idx over seq_len → single-env-per-sequence), and
empirically verified by Test 4 across all 4 lanes × 10 windows × 8 time
steps × 8 obs dims. ✓

---

## 3. Deviation algorithm review

### D-004 — memmap omission

**Algorithm-fidelity lens.** Memmap-vs-RAM is a *storage backend choice*,
not a *what-bytes-the-algorithm-sees choice*. Sheeprl's `MemmapArray`
(`vendor/sheeprl/sheeprl/utils/memmap.py`) is a thin wrapper around
`np.memmap` that exposes the same `[]` / `[idxes]` getter/setter
semantics as `np.ndarray`. Both backends produce the same bytes when
read via `self._buf[k][idxes]` — the difference is purely whether
those bytes live in a page-cached file on disk or in anonymous RAM.

For the algorithm's behavior:

- Every read path (the entire `_get_samples` flow) materializes
  to an in-memory ndarray view via `np.take(np.reshape(v, ...),
  flattened_idxes, axis=0)`. The intermediate view is byte-identical
  regardless of backend.
- Every write path (`self._buf[k][idxes] = data_to_store[k]`) writes
  the same bytes to the same logical slots in both modes.
- No `is_first` placement, `action` storage timing, `_pos` arithmetic,
  `_full` flag transition, or sample-window arithmetic differs between
  modes.
- The omitted code is *never executed* in either training-loop or
  test — not just untested.

**Algorithm verdict: PASS — no algorithm deviation.** The omission is a
deployment-regime simplification (lab nodes have enough RAM for a
1M-transition buffer). Forwarded to PI for ratification.

### D-005 — Test 1's `[:_pos]` slice (filled-region-only comparison)

**Algorithm-fidelity lens.** Both implementations allocate buffers with
`np.empty(shape, dtype)` (sheeprl `buffers.py:214` / JAX `buffers.py:165-167`).
The slots in `[_pos:]` are *uninitialized host memory* whose values
are undefined and platform-specific (whatever bytes happen to be at
the allocated address). These bytes do not flow into the algorithm —
sheeprl's `sample()` (`buffers.py:430-432`) raises ValueError before
the unfilled region is touched (also see point 7 above), and the
production training loop will only sample past `learning_starts=1024`,
at which point `_pos ≥ 1024` and the relevant slots are all filled.

The semantic claim "the buffer's algorithm-relevant bytes are
byte-identical to sheeprl" is fully tested by `[:_pos]` because
`[_pos:]` is provably never reached by the algorithm under sheeprl XS
semantics:

- Pre-wraparound (the regime Test 1 exercises, with `_pos=100, _full=False`):
  `sample()` validates `_pos - seq_len + 1 ≥ 1` before any read, so
  any read past `[:_pos]` would raise.
- Post-wraparound (`_full=True`): the sample-window arithmetic
  excludes only the chunk overlapping `_pos`; once `_full=True`, every
  buffer slot is filled.

Additionally, Test 1 asserts `_pos` and `_full` independently
(test lines 93-94), so a hypothetical bug where JAX wrote garbage to
`[:_pos]` AND reported a matching `_pos` would have to be coordinated
across two independent assertions — implausible given the line-for-line
port.

**Algorithm verdict: PASS — no algorithm deviation.** The test scope
exactly matches the algorithm's bytes-of-interest. Forwarded to PI for
ratification.

---

## 4. Verdict

**PASS.**

The CP3b buffer + cadence layer is algorithmically faithful to
sheeprl@`33b6366`. The four §S items that CP3b is upstream of (§S1, §S2,
§S3, §S4) all have their substrate laid correctly here, with their
downstream consumption deferred to the right CPs (CP4b for §S1+§S4, CP2b
for §S2, CP9b for §S3). All 8 critical algorithm-points raised in the brief
are either explicitly covered by the test suite or guaranteed by the
sheeprl-XS configuration invariants that Test 5 locks in:

- §S1 / §S4 substrate covered by Tests 1 + 3 + 4 (byte-identical `is_first`
  storage, straddling-window placement, lane independence).
- §S2 substrate is buffer-time-agnostic by design; correctly deferred to
  the train.py call-site review at CP9b.
- §S3 gradient-update gate arithmetic covered by Test 6; random-action
  prefill branch correctly deferred to CP9b.
- Replay-ratio = frequency-not-batch-size semantic verified by Tests 5+6.
- Straddling-window semantic verified by Test 3 (with a post-done step
  at window offset 5 of 10).
- `collect_interval` is not a sheeprl concept and is correctly absent from
  `agent_xs.yaml`.
- Buffer-warmup vs `learning_starts` ordering protected by the
  `_pos - seq_len + 1 < 1 → ValueError` runtime guard AND by Test 5's
  config invariant.
- Cross-env lane invariant verified by the `tile(env_idxes, seq_len)` →
  single-env-per-sequence formula in `_get_samples`, with Test 4's 2,560-float
  per-lane assertion.

Both deviations (D-004 memmap omission, D-005 `[:_pos]`-filled-region slice)
are algorithm-invariant under the lens of this review — D-004 is a storage
backend that doesn't touch the bytes the algorithm sees, D-005 is a test-scope
match against `np.empty`'s undefined-tail-garbage that the algorithm provably
never reads.

**PI gate may fire next on D-004 + D-005.**

---

## 5. Notes for downstream-CP reviewers (hand-off)

- **CP4b reviewer:** when auditing §S4's three-quantity reset
  (`action`, `recurrent_state`, posterior with reshape-flatten), re-exercise
  the straddling-window case in Test 3 with the *production* `sample()` path,
  not `_sample_at_indices`. Test 3 currently exercises the bypass path only;
  if a refactor ever adds a "don't straddle dones" flag to `sample()`, the
  CP3b test would still pass but the CP4b code would silently never fire.

- **CP2b reviewer:** when auditing §S2's prepend-zero-action shift,
  verify the *training-loop call site* (CP9b's `train.py` collection branch)
  sets `step_data["actions"]` BEFORE `rb.add()` and BEFORE `envs.step()`,
  matching sheeprl `dreamer_v3.py:586-591`. The buffer-side `add()` is
  time-agnostic; the §S2 alignment depends entirely on the call site.

- **CP9b reviewer:** §S3 has two halves. Test 6 covers the gradient-update
  gate arithmetic; CP9b owns *both* call sites — (a) the collection-branch
  random-action prefill (`iter_num <= learning_starts` → uniform sample),
  and (b) the outer-loop gradient-step gate (`iter_num >= learning_starts`
  → call `one_train_step`). The v2 plan's Checkpoint 9b spec
  (archived plan line 701) covers (a) with an entropy assertion on the
  replay buffer's stored actions. Recommend re-using that pattern.

- **Future `collect_interval` extension (if/when):** CP3b deliberately does
  not test `collect_interval` integration because sheeprl XS does not have
  the concept. If a future plan re-introduces it, a separate CP3c-style
  cadence test is needed — do not retrofit CP3b's Test 6.

---

## Links

- [CP3B_SPEC.md](../develop/active/dreamer_srl_v3/CP3B_SPEC.md) — primary spec
- [CP3b code review](dreamer_srl_v3_cp3b_code_review.md)
- [CP3b math review](dreamer_srl_v3_cp3b_math_review.md)
- [v3 implementation plan](../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md)
- [v3 deviation log](../develop/active/dreamer_srl_v3/DEVIATION_LOG.md)
- [v2 plan (algorithmic backbone with §S1–§S10)](../develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md)
- [v2 professor-rl-bayesian-dl review (stylistic template)](../develop/archive/dreamer_srl/review_professor_rl_bayesian_dl_v2.md)
- [CP1 professor-rl-bayesian-dl review (recent precedent)](dreamer_srl_v3_cp1_professor_rl_bayesian_dl_review.md)
- [SPS comparison (empirical motivation for CP3b promotion)](../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md)
- [Audited code](../../src/algorithms/dreamer_srl/buffers.py)
- [Audited tests](../../tests/algorithms/dreamer_srl/test_buffers.py)
- [Vendored sheeprl `SequentialReplayBuffer`](../../vendor/sheeprl/sheeprl/data/buffers.py)
- [Vendored sheeprl train() — cadence wiring](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py)

Reviewed by: professor-rl-bayesian-dl
