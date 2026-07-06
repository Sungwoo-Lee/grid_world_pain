---
title: "Code review — Tier 2 batched eval-rollout (JAX/nnx correctness)"
topic: reviews
status: complete
created: 2026-07-06
last_updated: 2026-07-06
---

# Code review — Tier 2 batched eval-rollout (`--batched`)

> **Reviewed by**: code-reviewer
> **Date**: 2026-07-06
> **Scope**: `scripts/eval/eval_rollout.py` (`_rollout_scan_jit`, `_run_episodes_batched`, `--batched` branch) + `scripts/eval/parity_check_eval_rollout.py`. Dirty/uncommitted tree.
> **Plan**: [[EVAL_ROLLOUT_BATCHING_PERF]]

## Verdict (plain-language entry point)

**What was reviewed.** The eval pipeline that plays evaluation episodes against a frozen trained model just gained a fast "batched" path: instead of playing 30 episodes one-at-a-time in a Python loop, it now plays all 30 at once as a single vmapped batch driven by a `jax.lax.scan`. This is correctness-critical measurement tooling (it feeds the avoidance-behavior results tables), so the new path is gated by a bit-for-bit parity harness against the old one. My job was a JAX/Flax-nnx correctness pass on the four failure modes the parity gate is protecting.

**Bottom line: PASS — no defects found beyond what exact parity already proves.** All four concerns are correct *by construction*, not merely by empirical luck:

1. **nnx.jit read-after-restore** — the batched forward pass reads the *correct* restored checkpoint weights because the model is invoked exclusively through `nnx.jit` (the split/merge that materializes restored state), exactly mirroring how the legacy path's `get_action_and_value_nnx` is `@nnx.jit`-decorated. I found **no lurking eager model-weight read** anywhere in the batched path (`model.initial_state` returns pure zeros, `model.action_dim` is a static int).
2. **PRNG threading** — per-env reset key is the stacked, directly-vmapped `jax.random.PRNGKey(seed)`, never `ParallelEnv.reset`'s internal `split`. A permanent runtime guard enforces this on every run. Correct.
3. **vmap axis** — action selection uses `jnp.argmax(logits, axis=-1)` (the batched-safe axis), and the model has no cross-batch coupling (LayerNorm over features, **no BatchNorm/dropout**), so env *i* in a batch evolves identically to env *i* run alone. Correct.
4. **Done-masking** — because `done = terminated OR truncated` (core.py:716), every env is guaranteed a `done=True` within the `max_steps` scan window, so `argmax(done_seq)+1` always finds a real length. The die-at-step-0, mixed-death-step, and never-dies-until-truncation boundaries all resolve correctly. Correct.

The senior-developer's independent re-run (90/90 episodes bit-exact incl. raw `.npz` across b03/b04/b05) corroborates the by-construction argument. **Tier 2 is clear to sign off.** The residual risks below are all *out-of-scope regions the parity gate did not exercise* — none is a defect in the reviewed code; they are flagged so a future change into those regions re-runs the gate.

---

## Concern-by-concern audit

### 1. nnx.jit read-after-restore (highest priority) — ✅ correct, fix is by-construction

The developer's **fix** is sound and is the strongest possible guarantee: both the legacy path and the batched path invoke the model *only* through `nnx.jit`. Legacy goes through `get_action_and_value_nnx` (`@nnx.jit`, recurrent_ppo_network.py:387); batched goes through `nnx.jit(_rollout_scan_jit, static_argnames=("max_steps",))` (eval_rollout.py:~318). Passing the `model` as an `nnx.Module` argument to `nnx.jit` triggers nnx's graphdef/state split-merge, which materializes the restored weights inside the trace. This is exactly why the two paths produce identical numbers.

**No lurking eager-read sites.** I audited every model touch in `_run_episodes_batched`:
- `model.initial_state(batch_size=num_envs)` — returns pure `jnp.zeros(...)` for GRU/LSTM task-state and (if modulation is enabled) `modulator.initial_state` also returns pure zeros (neuromodulator.py:182-186). **No weight read**, so an eager call here is numerically safe.
- `model.action_dim` — a plain `int` attribute set in `__init__` (recurrent_ppo_network.py:208). Not an array. Safe.
- `v_reset` / `v_obs` / `v_obs_true` / `v_step` — vmapped *environment* functions (`jax_reset`, `get_observation`, `jax_step`); they never read model weights. Safe to be eager.
- The only model **forward** (`model(...)`) lives inside `scan_fn`, inside `nnx.jit`. Correct.

The model carries **no mutable state** (no BatchNorm running stats, no dropout — grep confirmed), so the `nnx.jit` split/merge is deterministic across calls and dropping the returned model-state is correct in eval.

**On the stated root-cause mechanism** (a caveat, not a defect): the docstring's claim — "eager attribute reads see a stale view of restored weights even though `nnx.state(model)` checksums identically" — is not a standard, documented nnx failure mode and is likely an *imprecise characterization* of whatever actually diverged in the first draft (the symptom — early deaths, injury pinned at cap — is equally consistent with an argmax-axis bug or an obs-plumbing bug in that discarded draft). This does **not** undermine the fix: correctness here rests on "both paths forward through `nnx.jit`" + exact parity, which is airtight regardless of the exact mechanism of the original bug. I note it only because a mis-stated mental model raises the chance of a future regression (see Residual risk R1).

### 2. PRNG threading — ✅ correct, permanently guarded

`keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])` → `states0 = jax.vmap(jax_reset, in_axes=(None, 0))(params, keys)`. This is the required keying: env *i* resets from `PRNGKey(seeds[i])`, identical to legacy `jax_reset(params, jax.random.PRNGKey(seeds[ep_idx]))`. `ParallelEnv.reset` (and its internal `jax.random.split`) is **not** used. `params` is broadcast (`in_axes=None`), never batched — convention-correct.

The in-function guard (eval_rollout.py:~300-315) recomputes `jax_reset(params, PRNGKey(s)).key` for every seed and raises `RuntimeError` on mismatch. Because `jax_reset` is a pure function of `(params, key)` and vmap is exact, matching `state.key[i]` is a sufficient proxy for the whole reset state matching. This is a genuine permanent regression guard, not a one-off.

### 3. vmap axis correctness — ✅ correct

- `action = jnp.argmax(logits, axis=-1)` inside `scan_fn` — logits are `(num_envs, action_dim)`, so this reduces over the action axis → `(num_envs,)`. The shared helper's bare `jnp.argmax(logits)` (network.py:393) is only correct unbatched; the batched path correctly does **not** reuse it, calling `model(obs, h)` directly instead.
- Post-scan asserts `scan_out["action"].shape == (max_steps, num_envs)` and `jnp.all(action < action_dim)` — the shape assert is a strong axis check; the range assert is weaker (see nit N1) but harmless.
- **No cross-batch coupling** in the model: only LayerNorm variants (normalize over the feature/last axis, per-sample), Linear (per-row), GRU/LSTM cell (per-row). No BatchNorm, no batch-axis reduction. Therefore batched == stacked-unbatched, which is the entire basis of parity.

### 4. Done-masking / no post-death pollution — ✅ correct across all boundaries

`T = np.argmax(done_seq, axis=0) + 1`, then per-env slice `[:Ti]` for step arrays and `[:Ti]` snapshot loop atop the initial snapshot (→ length `Ti+1`). Boundary analysis:

- **Never-dies (full survival):** `done = logical_or(done, truncated)` and `truncated = next_step >= params.max_steps` (core.py:700, 716) fire on the `max_steps`-th step. So `done_seq[max_steps-1] == True` always → `argmax` returns `max_steps-1` → `T = max_steps`. Matches legacy's `while step < max_steps` (exactly `max_steps` iterations). The all-`False` → spurious `argmax=0` failure mode **cannot occur** because truncation guarantees a `True`. ✅
- **Dies at step 0:** `done_seq[0]=True` → `argmax=0` → `T=1`; snapshots = initial + 1 = 2; last snapshot = terminal state. Matches legacy (one step, then loop exits). ✅
- **Mixed death steps in one batch:** `argmax(..., axis=0)` is per-env, so each env gets its own first-`True` index independently; post-death steps of early-dying envs are sliced away and never reach the measures. ✅
- **Terminal-state recording under (possible) auto-reset:** the last recorded snapshot is `next_state` of the death step for **both** paths (legacy `recorder.append(next_state, ...)` at the death iteration; batched `snap_*[Ti-1] = next_state`). If `jax_step` auto-resets on `done`, both record the identical reset state, so parity holds either way. And `argmax` picks the *first* `done`, so a second death after an in-scan auto-reset is correctly ignored. ✅
- **termination_reason:** batched reads `scan_out["termination_reason"][Ti-1, i]` = the reason at the death/truncation step = the last `info` legacy reads. ✅
- **nociception:** batched hardcodes `np.zeros(Ti)`. Verified against core.py's info schema — `info` carries **no** `nociception` / `exteroception_nociception` key, so legacy's `.get(..., 0.0)` fallback always returns `0.0`. Exact match (see Residual risk R4 for the fragility caveat). ✅
- **Recorder format:** batched replays sliced arrays through the real `EpisodeRecorder` + `_snapshot_state` (via a `SimpleNamespace` per (env,t)), so `.rec.gz` payload is byte-structurally identical by construction; `actions = [-1, a0, ...]`, `rewards = [0.0, r0, ...]`, `obs/true_obs` post-step — all match the legacy recorder's ordering. ✅

---

## Residual-risk regions (NOT exercised by parity on 3 checkpoints × 4 configs)

None of these is a defect in the reviewed code. They are regions the exact-parity gate did not cover; a change that moves into them should re-run the harness.

- **R1 — Silent stale-weight regression is only caught out-of-band.** The three in-function guards (action shape, `action < action_dim`, PRNG-key equality) would *all still pass* if someone deleted the `nnx.jit` wrapper and reintroduced an eager forward — actions stay in range, keys stay correct, numbers silently diverge into plausible-but-wrong policy behavior. The **only** protection against reintroduction is the external parity harness. Recommend keeping `parity_check_eval_rollout.py` as a CI/pre-sweep gate and treating the docstring warning as load-bearing.
- **R2 — Modulation-enabled models (FiLM / neuromodulator).** Parity ran on `rppo_basic03/04/05`, which are non-modulated. A modulated model carries extra recurrent state in `h = (task_h, mod_h)` and returns `mod_info`; the batched path handles the pytree structurally, but per-env-vs-batched equivalence of the modulator forward is unverified. Low risk (per-sample ops), but ungated.
- **R3 — LSTM `rnn_type` and partial recording (`record_n_episodes < n_eps`).** The harness always records every episode (Deviation #4) and the tested configs appear to be GRU. The `if record and i < record_n_episodes` logic reads correct, but the partial-record path and the LSTM tuple-carry path are ungated.
- **R4 — `nociception` fragility.** The hardcoded-zeros shortcut is correct *today* but has no runtime guard: if a future config or reward change adds a `nociception` key to `jax_step`'s `info`, the batched path silently returns `0.0` while legacy would return the real value. Cheap defense: assert the key is absent, or carry it in the scan like the other info fields.
- **R5 — `.rec.gz` obs / true_obs vectors are not stats-gated.** The parity harness compares the 11 measures (snapshots only) + raw `.npz` (which excludes obs/true_obs). The batched `.rec.gz` observation vectors are correct-by-construction (same pure `get_observation` under exact vmap) but are **not** byte-compared batched-vs-legacy — they feed only Tier-3 rendering. Worth a one-off byte-diff before Tier 3 relies on them.
- **R6 — Configs with both predators AND neutrals > 0 simultaneously.** Tested were predator-only and neutral-only (`avoid_pred*`, `avoid_rabbit*`) plus zero-animal. The mixed `dist_per_predator [T,P]` + `dist_per_neutral [T,N]` case uses the same slicing mechanism and is very low risk, but is not directly in the parity matrix.

---

## Conventions audit

| Convention | Status | Note |
|---|:--:|---|
| Pytree / immutability | ✅ | No in-place mutation; env state threaded functionally through the scan carry. |
| JIT recompilation | ✅ | `max_steps` correctly `static`; `params` traced/broadcast, not batched. |
| vmap axis | ✅ | axis 0 = env index; `EnvParams` broadcast (`in_axes=None`); argmax `axis=-1`. |
| PRNG threading | ✅ | Stacked `PRNGKey(seed)` + vmap; permanent guard; no `ParallelEnv.reset`. |
| Sensor / obs breakdown sync | ✅ (n/a) | No sensor added/renamed; `get_observation` reused unchanged. |
| Config protocol | ✅ | No new keys; `--batched` additive; `eval_policy_mode != deterministic` guarded with `NotImplementedError`. |

**Nit (N1):** the `assert jnp.all(action < action_dim)` axis check is weak (a wrong-axis argmax could still land in range). The `shape == (max_steps, num_envs)` assert already carries the real signal — fine to leave as-is.

---

**Conclusion**: Tier 2 batched rollout is JAX/nnx-correct by construction and matches the legacy reference bit-for-bit; no defects found. Clear to sign off. Residual risks R1–R6 are ungated *regions*, not bugs — re-run the parity harness before entering any of them (esp. modulated models, and before Tier 3 depends on `.rec.gz` obs vectors).

Reviewed by: code-reviewer
