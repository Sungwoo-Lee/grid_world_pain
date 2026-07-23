# Diagnosis findings — dreamer_srl eval.py + checkpoint.py (2026-07-23)

Unit: `src/algorithms/dreamer_srl/eval.py` (in-training + offline eval rollouts) and `src/algorithms/dreamer_srl/checkpoint.py` (Orbax save/restore helpers). Report-only pass; all code paths traced end-to-end; one read-only Python experiment (CPU, tiny nnx module + Orbax roundtrip) settled the eager-read-after-restore question empirically. (Written by the parent session on the reviewer's behalf — its Write tool was blocked.)

Plain-English headline: no P0s. The checkpoint helper's **restore** half is the weak spot — its documented usage is broken as written (every real caller carries a private workaround), and resume-critical state beyond the known Adam momentum (the gradient-step scheduler's counter, the replay buffer) is silently absent. On the eval side the two rollout paths are numerically sound, but recordings never store the noise-free observation, and a load-bearing root-cause docstring makes a claim that measurably does not reproduce.

## Finding 1 — [P2] checkpoint.py:113-123 — `load_checkpoint()`'s documented contract is broken

The docstring says: returns "the raw pytree dict as stored (call `nnx.update(module, ckpt['actor'])` … to restore)". Both halves fail:
1. **Same-device**: untargeted `manager.restore(episode)` returns leaves wrapped as `{'value': array}` with integer layer indices as *string* keys. `nnx.update(module, raw['actor'])` raises — verified empirically this pass: `nnx.update(raw dict) FAILED: KeyError Ellipsis`.
2. **Cross-device**: restoring a GPU-saved checkpoint in a CPU process (the standard offline-eval setup; `--device cpu` is the default) raises `ValueError: Topology mismatch detected` — documented empirically in `scripts/eval/dreamer_srl_probe_eval.py:42-63`.

Failure scenario: any new consumer following the docstring crashes; all three existing consumers ship private workarounds (`eval_rollout.py:1149-1151` full abstract target + `StandardRestore(item=…)`; `dreamer_srl_probe_eval.py` same; `scripts/dreamer/dreamer_srl_offline_wm_test.py:167-181,683` `_normalize_checkpoint` digit-key cast + `value` unwrap).
Fix direction: give `load_checkpoint` an optional target pytree (or build it from passed modules), restore via `StandardRestore(item=target)`, absorb `_normalize_checkpoint` for the no-target case, fix the docstring.

## Finding 2 — [P2] checkpoint.py:44-110 — Ratio scheduler state never checkpointed; a future resume fires a massive gradient-step burst

`save_checkpoint` persists params/moments/key/counters but not the `Ratio` replay-ratio scheduler's `_prev` marker, even though `Ratio.state_dict()`/`load_state_dict()` exist for exactly this (utils.py:398-415, ported from sheeprl, called by nothing). On resume (the write-only `stage` field at checkpoint.py:77-82 says one is planned), a fresh `Ratio` has `_prev=None` and its cold-start branch (utils.py:380-391) computes `repeats = int(step * ratio)` with the restored run's full policy-step count — resuming at 1M steps with ratio 0.5 owes ~500k gradient steps in one iteration. Same family: replay buffer and in-flight episode accumulators/env states unsaved. Distinct from the known Adam-momentum row and compounds it: a resume plan that only adds optimizer state still hits the Ratio burst.
Fix direction: `ckpt_data['ratio'] = ratio.state_dict()` (plain scalars, Orbax-safe) alongside the optimizer-state fix.

## Finding 3 — [P2] eval.py:122, 470 — "no noise API" is false; Dreamer eval recordings lack the noise-free observation

Both rollout paths record `true_obs=None` with the comment "no noise API". The API exists: `get_observation(state, params, apply_noise=False)` (sensor.py:290-291), and the rPPO batched eval uses it (`eval_rollout.py:249-251`, `v_obs_true`). For Dreamer runs with `perceptual_noise_enabled`, rendered eval videos and `.rec.gz` trajectory analysis have no clean-signal reference — the capability the FIXED "noise invisible in eval video" bug restored for rPPO is structurally unavailable for Dreamer, giving asymmetric tooling across the two stacks.
Fix direction: record `get_observation(…, apply_noise=False)` as `true_obs` (vmapped in the batched path), mirroring the rPPO branch.

## Finding 4 — [P2] eval.py:210-233 — the "eager read after nnx.update is stale" root-cause claim does not reproduce

The `_dreamer_rollout_scan_jit` docstring asserts eager forwards after `nnx.update(model, restored_tree)` read stale params and that this "applies identically" here — which would make the default offline eval path (non-`--batched` `eval_rollout.py:1196`, `dreamer_srl_probe_eval.py:235`, both fully eager on restored modules) a P1. Empirical test this pass: tiny nnx module saved via `StandardSave`, restored into a differently-initialized module via targeted `StandardRestore` + `nnx.update` — eager forward == `nnx.jit` forward == saved-model reference (all allclose True). Also consistent with production: training pairs per-iteration `nnx.update` write-backs (dreamer_srl_main.py:1927-1933) with the fully-eager `Player.get_actions`, and training learns. The claim is copied into two docstrings (eval.py and eval_rollout.py:247-266) and flagged for code-reviewer; the *real* cause of the historical rPPO batched-vs-legacy divergence is likely the era's untargeted raw-dict restore (Finding 1's tree), not eager evaluation, and remains undiagnosed.
Fix direction: re-run the rPPO parity harness isolating restore-style (targeted vs raw) from call-style (eager vs nnx.jit); rewrite both docstrings with the confirmed mechanism.

## Finding 5 — [P2] eval.py:364 vs :96 — the two paths derive `action_dim` from different sources; batched silently steps the env with clipped out-of-range actions on mismatch

Legacy computes `action_dim` from the env (`4 + rest + eat`, eval.py:96); batched takes `int(actor.action_dim)` (eval.py:364). On an agent/env action-space mismatch (offline eval of a checkpoint against a probe config whose rest/eat toggles differ): legacy gets a loud RSSM shape error (fail-fast, fine); batched is self-consistent at the actor's width, so `argmax` can emit e.g. index 5 into a 5-action env — `jax_step` clamps silently (`moves_map[jnp.clip(action, 0, 5)]`, core.py:562) and `eat_action_idx` aliasing (core.py:580-581) can turn it into an unintended Eat/Rest. The sanity assert at eval.py:430-433 checks `action < actor.action_dim` — the wrong bound — so it passes exactly when it matters. Result: plausible-looking, silently wrong survival-step results in `--batched` probe sweeps.
Fix direction: assert env-derived action count == `actor.action_dim` before the batched rollout.

## Bug-registry staleness note

KNOWN-OPEN "eval_rollout.py raises NotImplementedError for dreamer_srl checkpoints" is stale: `scripts/eval/eval_rollout.py:1083-1230` now has a complete dreamer branch (checkpoint-arg reconciliation, targeted restore, legacy + batched rollout, metadata.json). bug-curator should retire or re-scope the row.

## Fixed-bug regression check

- **Known OPEN momentum row (checkpoint.py:85-88) — VERIFIED, with depth**: `save_checkpoint` takes no optimizer objects at all; `ckpt_data` (checkpoint.py:84-96) holds only `nnx.state(m, nnx.Param)` for four modules plus scalars/moments/key. Fix is cheaper than the row implies: the driver already splits/merges all three `nnx.Optimizer` states as pure pytrees every iteration (dreamer_srl_main.py:1910-1933), so they are checkpoint-ready; the half-finished intent shows in `Ratio.state_dict()` existing unused (Finding 2). Restore-side wiring is the larger half (no resume path exists).
- **"continual eval rebuilt stage-0 env" — no dreamer regression**: in-training eval receives the rebound `env_params` (dreamer_srl_main.py:1579 → 1708/1749). Minor caller-owned cosmetic gap: `config=env_cfg` stays the launch config, so post-swap recordings' `run_meta.pkl` icons/source_path describe stage 0.
- **"noise invisible in eval video" — rPPO fix intact** (`v_obs_true` still recorded); Dreamer paths never adopted it (Finding 3) — nothing regressed, parity never achieved.
- **"resume silently restored nothing" / "continual resume wrong stage" — n/a** (old train.py stack; dreamer_srl has no resume; `stage` write-only as documented).
- **Known OPEN "eval curve mixes two estimators" — confirmed present, owned elsewhere** (dreamer_srl_main.py:1733-1765 logs both passes into the same `Eval/Mean*` keys; both passes share `seed=args.seed`, so the stats pass's first `eval_video_episodes` episodes duplicate the video pass's trajectories — the two estimators are not even independent).

## Reviewed but clean

- RSSM step parity: eval's per-step call/arg order matches `Player.get_actions` (dreamer_srl_main.py:304-352) and `RSSM.dynamic` (agent.py:975-990); `jax.vmap(encoder)` identical to training.
- Latent-state reset between episodes: legacy re-creates h0/z0, zero prev_action, is_first=1 per episode (eval.py:106-112); batched via the §S4 arithmetic-mask reset at t=0. No cross-episode leakage.
- Deterministic-action consistency: `forward_logits` returns post-unimix logits (agent.py:1487-1503); unimix is monotone, so argmax equals the training policy's mode; posterior sampling stays stochastic (matches DreamerV3 practice).
- Survival-step counting: both paths count the death step (legacy `step_count` includes the terminal step; batched `T = argmax(done)+1`), matching training's `ep_len = counters + 1` (WP-SRL P5). Truncation guaranteed inside the scan window (core.py:699-717), so the batched all-False-argmax edge (T=1) is unreachable.
- PRNG hygiene: eval forks a fresh `PRNGKey(seed)`, never touching the training key thread; no eval/training world overlap (training's first reset consumes `split(k_reset, num_envs)` sub-keys, wrapper.py:17-21; eval uses `k_reset` directly).
- Eval observation noise matches training: same `get_observation` default `apply_noise=True`, state-salted noise key (sensor.py:290-347).
- Metric aggregation: plain mean-of-episodes for both reward and length in both paths; no pooled-step mixing inside this unit.
- Save atomicity: Orbax finalizes via tmp-dir + atomic rename; `wait_until_finished()` called (checkpoint.py:109-110); a kill mid-save leaves an uncommitted step the manager ignores.
- Step collisions: in-driver saves gated by `last_ckpt_episode` boundary crossing (dreamer_srl_main.py:1671-1691); `results_dir` timestamped per run (dreamer_srl_main.py:895-901).
- Param-filter coverage: stack holds no non-Param variables (no BatchStat; LayerNorm params and the learned `initial_recurrent_state` are `nnx.Param`, agent.py:673); `moments` (percentile-EMA, flax.struct.dataclass) serializes via the `__dict__` fallback; key/iter_num/policy_step/episode/grad counters/stage all present.
- int32 counters (checkpoint.py:91-95): safe at realistic budgets (overflow at 2.1B policy steps; runs are ≤ tens of millions).
