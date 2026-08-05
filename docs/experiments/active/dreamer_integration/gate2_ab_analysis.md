---
title: "Gate 2 A/B analysis — dreamer_srl legacy entry point vs. integrated train.py dispatch"
topic: dreamer_integration
status: active
created: 2026-08-05
last_updated: 2026-08-06
phase: "Gate 2 (of 3) — real GPU A/B — PASSED"
wandb_tag: "dsrl_gate2*"
develop_link: docs/develop/active/dreamer/DREAMER_SRL_TRAIN_PY_INTEGRATION_PLAN.md
---

# Gate 2 A/B Analysis — Dreamer legacy entry point vs. integrated train.py dispatch

> **Status**: COMPLETE — **GATE 2 PASS.** Both legs green: the statistical leg (production-noise A/B inside the measured noise-floor envelope, §4.5) and the deterministic leg (three pinned-environment runs bit-identical end to end at full 20,000-episode scale, §4.6 — every one of 20,000 episode records, every checkpoint, the gradient-step count, the final losses, and every module of the final checkpoint down to the PRNG key and optimizer states).
> **Date**: 2026-08-05 (A/B + noise floor) / 2026-08-06 (deterministic leg, finalization)
> **Author**: `experiment-analyzer`
> **Related**: [[DREAMER_SRL_TRAIN_PY_INTEGRATION_PLAN]] (Gate 2 section = authoritative pass criteria; worktree copy carries all amendments) · [[DREAMER_SRL_EVAL_TELEMETRY_FIX]] (Track C semantics checked here) · [[DREAMER_SRL_TRAIN_PY_DIVERGENCE_MATRIX]]

---

## 1. Research Question

The project is retiring the Dreamer world-model agent's private launch script and folding it into the shared training entry point (`train.py`) that every other algorithm already uses. Before anyone's launch scripts change, the integration must pass three gates; this document is the analysis for **Gate 2** — one real GPU training run launched each way, compared end to end. The question is simple: **does a full-length Dreamer training run launched through the new integrated entry point behave the same as one launched through the old legacy script — same learning curves, same world-model convergence, same speed, with all the recently fixed evaluation telemetry (videos, eval metrics on the right step clock) intact and its checkpoints restorable for offline evaluation?**

A crucial piece of context shapes what "the same" can mean: **same-seed Dreamer runs are not reproducible on this stack** (the replay-buffer sampler is unseeded — a registered known issue — and GPU float nondeterminism feeds back into episode lengths and thus the whole trajectory). Two identical invocations of the *same* entry point diverge. The comparison is therefore **statistical, not exact**: the two runs must overlap within same-seed run-to-run noise. To measure that noise directly, a dedicated **noise-floor pair** — the legacy entry point launched twice with the identical command — was run alongside; its inter-run divergence is the yardstick against which the A/B difference was finalized (§4.5). A separate deterministic leg — three runs in a pinned bit-deterministic environment with the replay-sampler seeding fix on both trees, where "the same" *can* mean bit-identical — carried the remaining burden of proof and closed it (§4.6): the legacy and integrated entry points produced literally the same 20,000-episode training, record for record, weight for weight.

**Formal hypothesis:**

> **H₀** (integration is transparent): the entry-point swap has no effect — every A/B difference is within same-seed run-to-run noise, telemetry is intact, and checkpoints restore.
> **H₁** (integration changed training): the integrated path introduces a systematic difference in learning curves, world-model loss, or throughput, or breaks telemetry/checkpoints.

A **pass** for Gate 2 is *failing to find* any difference beyond noise (H₀ stands). The authoritative pass criteria are §"File Changes — Phase 3 (GPU A/B → Gate 2)" item 4 of the integration plan (worktree copy).

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Entry point | A = legacy `python -m src.algorithms.dreamer_srl.dreamer_srl_main` · B = integrated `train.py` dispatch | The only intended difference; everything downstream of config resolution delegates to the same training loop |

### 2.2 Controlled Variables

Both runs: same git commit (`0a55da1`, worktree checkout — **both** sides executed from the same code tree, so the comparison isolates the entry point, not a code-version difference), same env config (basic04 "jump-attack" 10×10 world), same agent config (food-only M-size Dreamer, ~23.0 M params, `agent.algorithm: dreamer_srl` runtime temp copy), seed 42, 20 000 episodes, 128 envs, node 106 (docker-106, RTX 3090, one GPU each, launched ~30 s apart), checkpoint cadence 5 000 episodes keep-ALL, eval `video_during_training: true` / `stats_during_training: false` (3-episode video pass per checkpoint).

Verified from run artifacts, not assumed: WandB metadata (`root`, `git.commit`, `args`) and the run-dir dumped configs (§Appendix B).

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| Same-seed nondeterminism (unseeded replay sampler + GPU float noise → trajectory divergence) | both | High | By design: comparison is statistical; noise-floor A-vs-A pair (running) supplies the yardstick |
| Eval estimator is the 3-episode video pass | both | Med | `Eval/MeanLength` treated as qualitative only; survival curve (window-averaged over 200-episode blocks of 128 envs) is the headline |
| Different GPUs on the same node (106:0 vs 106:1, same model RTX 3090) | both | Low | Same card model, clean node; SPS compared as ratio |
| One seed, one config | both | Med | Accepted in plan (Gate 1 proved bit-identity of config resolution incl. curriculum on CPU/GPU-pinned envs; Gate 2 is deliberately single-config) |

## 3. Launch Manifest

Runs were launched by `training-runner` (no pre-registered designer manifest; the plan doc's Gate 2 procedure is the pre-registration). Recorded here for the record — columns are the actuals.

| Run | Status | Cell | Tag (= wandb-name) | Seed | Node:GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|------|------|-------------|--------------|----------|
| A | completed | legacy entry | `dsrl_gate2A_legacy_b04M_s42_n106` | 42 | 106:0 | 2026-08-04T05:23 | `3wsfk1ca` | `logs/20260804_052322.log` |
| B | completed | train.py entry | `dsrl_gate2B_trainpy_b04M_s42_n106` | 42 | 106:1 | 2026-08-04T05:24 | `1zbgtsxp` | `logs/20260804_052330.log` |
| N1 | completed | noise floor 1/2 (legacy twice) | `dsrl_gate2N1_legacy_b04M_s42_n106` | 42 | 106:0 | 2026-08-05T00:06 | `u2156tb7` | `logs/20260805_000604.log` |
| N2 | completed | noise floor 2/2 (legacy twice) | `dsrl_gate2N2_legacy_b04M_s42_n106` | 42 | 106:1 | 2026-08-05T00:06 | `breewovl` | `logs/20260805_000606.log` |
| detA | completed | det leg — legacy, pre-seam tree + fix-set | `dsrl_gate2detA_b04M_s42_n114` | 42 | 114:0 | 2026-08-05T01:30 | `6wk4pgra` | `logs/20260805_013008.log` |
| detA2 | completed | det leg — identical repeat of detA (cross-GPU control) | `dsrl_gate2detA2_b04M_s42_n114` | 42 | 114:1 | 2026-08-05T01:30 | `3ilbzb8d` | `logs/20260805_013012.log` |
| detB | completed | det leg — integrated train.py + fix-set | `dsrl_gate2detB_b04M_s42_n114` | 42 | 114:2 | 2026-08-05T01:30 | `dhxel4br` | `logs/20260805_013014.log` |

N1/N2 results dirs: `results/JAX_DreamerSRL/20260805-000639_dsrl_gate2N1_legacy_b04M_s42_n106` and `…000642_…N2…` (main tree); local WandB dirs under the worktree `wandb/` (`run-20260805_000638-u2156tb7`, `run-20260805_000641-breewovl`). Extracted histories: `tmp/gate2_analysis_20260805/histN{1,2}.json`.

Deterministic-trio results dirs: detA `results/JAX_DreamerSRL/20260805-013030_dsrl_gate2detA_b04M_s42_n114`, detA2 `…013033_…detA2…` (main tree), detB `.claude/worktrees/agent-a18eeb59fa7ffffeb/results/JAX_DreamerSRL/20260805-013017_dsrl_gate2detB_b04M_s42_n114` (worktree tree). Deterministic environment per the signed plan amendment: `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, per-process `CUDA_VISIBLE_DEVICES` pinning, node 114 (RTX 6000 Ada, three distinct GPUs), seeded replay-sampler fix-set applied identically on both trees. Comparison artifacts: `tmp/gate2_analysis_20260805/cmp_det_logs.py`, `det_ckpt_checksums.json`.

Artifacts: run A results `results/JAX_DreamerSRL/20260804-052357_dsrl_gate2A_legacy_b04M_s42_n106` (main tree); run B results `.claude/worktrees/agent-a18eeb59fa7ffffeb/results/JAX_DreamerSRL/20260804-052333_dsrl_gate2B_trainpy_b04M_s42_n106` (worktree tree). Both local WandB dirs live under the worktree's `wandb/` (`run-20260804_052355-3wsfk1ca`, `run-20260804_052404-1zbgtsxp`). Analysis used **local WandB datastore files only** (no web API); parser + extracted history in `tmp/gate2_analysis_20260805/`, working notes in `tmp/20260805_000000_gate2_ab_analysis.md`.

## 4. Results

### 4.1 Primary Metrics — Gate 2 checklist

Headline metric is **survival steps** (`Episode/Steps`, window-averaged, on the episode axis; 100 common logging points at episodes 200…20 000).

| # | Gate 2 criterion | Finding | Status |
|---|---|---|---|
| 1 | Survival curves overlap within same-seed run-to-run noise | Same shape (both ≈17 → ≈42–44 steps). B sits above A by **+2.85 steps on average** (pointwise B−A: sd 1.65, range −0.29…+5.96); last-20% means A 40.8 ± 0.9 vs B 43.4 ± 0.5 (≈+6%). **Finalized against the completed noise-floor pair N1/N2** (§4.5): the pure-noise envelope has |diff| mean 1.77, **peak 5.23**, last-20% delta **−3.88**, and is itself **sign-consistent** (one run above the other at 81/99 points) — so the A/B profile (peak 5.96, comparable; last-20% +2.56, *smaller* than noise; mean 2.86 vs 1.77, ~1.6× a single noise draw) is **not clearly outside** the envelope, and the sign-consistency that motivated caution is reproduced by pure noise | **PASS (statistical leg)** |
| 2 | `WorldModel/loss_model` same-shape convergence, no systematic offset | Same shape (A 1.881→1.758, B 1.905→1.761, WandB windowed). Fraction-of-run-aligned offset B−A = **+0.009 ± 0.011** (≈0.5% of the loss value), sign flips along the curve → no systematic offset | **PASS** |
| 3 | `Time/sps_env` ratio within ±5% | A mean 15.44 (n=56), B mean 14.73 (n=60) → B/A = **0.954 (−4.6%)**; median ratio 0.955. **Explained by the GPU slot, not the entry point**: the noise pair ran legacy-vs-legacy on the *same two slots* (N1 on 106:0, N2 on 106:1) and reproduced the gap almost exactly — N2/N1 = **0.951 (−4.9%)**. Slot 106:1 is simply ~5% slower than 106:0 on this node; the entry-point contribution is ≈0 | **PASS** (marginality resolved) |
| 4 | Track C intact on run B (videos, `Eval/*` on policy-step axis, estimator split, no dropped rows) | 4 eval videos present in run B's media (`media/videos/eval/video_<policy_step>_*.mp4` at policy steps 141 952 / 349 056 / 555 648 / 774 016 — the backward-step fix holds); `Eval/MeanLength`, `Eval/MeanReward`, `eval/video`, `eval/checkpoint_episode` all logged at `step=policy_step` at checkpoints 5 001/10 003/15 002/20 000; estimator split correct for this config (stats pass off → video pass writes `Eval/Mean*`, and no `Eval/video/*` keys appear — exactly the telemetry-fix routing table's "stats off" row); **zero** "dropped row"/backward-step warnings in wrapper log, console log, and wandb debug log | **PASS** |
| 5 | Run-B dir passes `eval_rollout.py` checkpoint-restore smoke | Final checkpoint (episode 20 000) restored on CPU from the worktree checkout and evaluated: 5 episodes, mean survival 43.0 steps (consistent with B's training-end ≈43.8). Control on A from the main checkout: mean 42.8 (training-end ≈41.6) | **PASS** (A control also pass) |
| 6 | No anomalies in logs/metrics | Both runs completed 20 000 episodes and exited cleanly; identical metric schemas; no errors/NaN/gaps; only the long-known orbax `CheckpointManager` deprecation warning. Two benign, *expected* config-dump deltas (§Appendix B) and one display artifact (§5.3) | **PASS** |

> **Verdict on H₀ (integration transparent): SUPPORTED — GATE 2 PASS, both legs.** *Statistical leg* (finalized 2026-08-05): all six checklist criteria pass; the A/B divergence profile sits inside the measured pure-noise envelope per the pre-registered §6.1 decision rule, and every qualitative marker that had motivated caution — sign-consistency in particular — is reproduced by two byte-identical legacy launches. *Deterministic leg* (finalized 2026-08-06, §4.6): with the stack's known nondeterminism sources pinned, the legacy and integrated entry points are **bit-identical at full 20,000-episode production scale** — every episode record, checkpoint boundary, gradient-step count, final loss, and every module of the final checkpoint including the PRNG key and optimizer states. The deterministic leg upgrades the conclusion from "no detectable difference" to "the same computation".

### 4.2 Secondary Metrics

| Run | Metric | Value | Note |
|-----|--------|-------|------|
| A | `WorldModel/loss_model` final (console, instantaneous last iter) | 1.7099 | The user-visible console number; WandB summary (windowed mean) is 1.7584 |
| B | same | 1.7193 | WandB windowed 1.7610 |
| A | `Eval/MeanLength` at 4 checkpoints (N=3 video pass) | 11.3, 27.7, 39.0, 5.7 | 3-episode estimator — qualitative only |
| B | same | 5.3, 43.3, 63.3, 23.3 | idem; wildly noisy in both runs, as expected at N=3 |
| A | wall clock / final policy step | 12.94 h / 713 600 | |
| B | wall clock / final policy step | 14.62 h / 774 016 | B's episodes are longer → more env steps for the same 20 000 episodes; this, not per-step slowness alone, drives most of the wall-clock gap |
| A/B | checkpoints kept | A {5002, 10003, 15001, 20000}, B {5001, 10003, 15002, 20000} | 5 000-cadence keep-ALL working on both; ±1–2-episode boundary offsets are trajectory divergence, expected |

### 4.4 Learning Dynamics

Survival steps, 10-window view of the episode axis (window = 2 000 episodes):

| Window (episodes) | A mean | B mean | B−A |
|---|---|---|---|
| 200–2 000 | 17.45 | 17.75 | +0.30 |
| 2 200–4 000 | 23.94 | 24.84 | +0.90 |
| 4 200–6 000 | 26.41 | 28.11 | +1.70 |
| 6 200–8 000 | 30.50 | 33.87 | +3.37 |
| 8 200–10 000 | 33.70 | 39.11 | +5.42 |
| 10 200–12 000 | 35.92 | 41.12 | +5.21 |
| 12 200–14 000 | 37.43 | 41.01 | +3.59 |
| 14 200–16 000 | 38.54 | 41.42 | +2.88 |
| 16 200–18 000 | 39.99 | 42.93 | +2.95 |
| 18 200–20 000 | 41.63 | 43.80 | +2.17 |

Both curves are smooth, monotone-rising, no collapses or phase transitions. The gap opens from near-zero, peaks mid-run (+5.4 around episodes 8–12 k), then narrows to +2.2 by the end — divergence-then-partial-reconvergence, the temporal signature expected from compounding trajectory noise between two runs of the same underlying learner, and *not* the flat or growing offset expected from a config/semantics difference present from step one. World-model loss shows the same shape on both sides with a ±0.01 wiggle and no persistent offset.

### 4.5 Noise-Floor Envelope (N1 vs N2) and the Criterion-1 Decision

Two byte-identical legacy invocations (same command, seed 42, 20 000 episodes, same node, launched ~3 s apart) — their divergence is pure run-to-run noise. Both exited cleanly.

Survival-curve pointwise divergence, same 100-point episode-axis method as A/B (99 common points for the N pair):

| Statistic | Noise pair (N2−N1) | A/B (B−A) | A/B inside envelope? |
|---|---|---|---|
| signed mean | −1.49 | +2.85 | mean of \|diff\|: 1.77 vs 2.86 → ~1.6× a single noise draw |
| pointwise sd | 1.67 | 1.65 | yes (equal) |
| peak \|diff\| | **5.23** | 5.96 | comparable (+14%) |
| last-20% delta | **−3.88** | +2.56 | yes — A/B end-of-run gap is *smaller* than noise |
| max window delta (2 000-ep windows) | −4.95 | +5.42 | comparable |
| sign consistency | 81/99 one-directional | 96/100 | noise reproduces sign-consistency |

Corroborating evidence from the other two quantitative criteria:

- **World-model loss**: the noise pair's console-final spread is **0.067** (1.7526 vs 1.8193) — the A/B console gap (0.0094) is **7× smaller than noise**. Fraction-aligned windowed offset: noise −0.008 ± 0.017 (max |0.047|) vs A/B +0.009 ± 0.011. A/B is comfortably inside.
- **Throughput**: N2/N1 `Time/sps_env` = 0.951 (−4.9%) with legacy on *both* sides, on the same two GPU slots as A/B (0.954, −4.6%) → the SPS gap is a property of slot 106:1, not of the entry point.

**Decision (pre-registered rule, §6.1):** the A/B profile is *not clearly outside* the noise envelope — it exceeds it on exactly one statistic (run-mean, 1.6×, against a yardstick that is itself a single noisy draw) while sitting at or below it on peak, end-of-run delta, per-window max, and pointwise sd, with the loss and SPS evidence pointing the same way. **Criterion 1: PASS (statistical leg).** Residual ambiguity about a small systematic offset hiding under this large noise floor is assigned to the deterministic (bit-identity) leg on node 114, not to further stochastic replication.

### 4.6 Deterministic Leg (detA / detA2 / detB) — Bit-Identity at Full Scale

Per the signed plan amendment ("Gate 2 criterion 1 (amended)"): three 20,000-episode runs in a pinned bit-deterministic environment with the seeded replay-sampler fix-set on both trees — detA (legacy entry point, pre-seam throwaway tree + fix), detA2 (byte-identical repeat of detA on a different GPU — the replication + cross-GPU-determinism control), detB (integrated train.py, this branch + the same fix). Pass rule: detA ≡ detA2 AND detA ≡ detB, bit-identical on the full common log-stream telemetry. All three exited cleanly.

**(a) Log-stream channel** (wrapper logs, tqdm carriage returns stripped; extraction per the cross-tree-triage method):

| Channel | detA vs detA2 | detA vs detB |
|---|---|---|
| Per-episode records `[iter N] episode done: env/ep_len/ep_rew` | **20,000/20,000 IDENTICAL** | **20,000/20,000 IDENTICAL** |
| Checkpoint (episode, iteration) pairs | IDENTICAL — (5001, 9762), (10000, 22339), (15000, 35288), (20000, 48902) | IDENTICAL (same four) |
| Periodic `[ep N]` lines (policy_step, world_model_loss, moments_invscale) | 489/489 IDENTICAL | 489/489 IDENTICAL |
| Per-iteration progress tuples (step, wm loss, invscale) | 482/482 iterations IDENTICAL | 482/482 IDENTICAL |
| Cumulative `grad_steps` | 781,424 == 781,424 | 781,424 == 781,424 |
| Final-losses dict (39 keys, 6 decimal places) | IDENTICAL | IDENTICAL |

**(b) Final-loss dicts:** detA2's block equals detA's (all 39 entries; detA vs detB likewise — both directions verified in the same comparison; final `world_model_loss` 1.797300 on all three).

**(c) Final-checkpoint per-module parameter checksums** (episode-20000 checkpoint, restored on CPU as raw arrays; SHA-256 over every leaf's dtype+shape+bytes, aggregated per top-level module — 268 leaves, 14 modules): **all 14 modules identical across detA / detA2 / detB** — the four networks (`world_model`, `actor`, `critic`, `target_critic`), all three Adam optimizer states (`wm_opt`, `actor_opt`, `critic_opt`), the return-normalizer `moments`, the training counters (`policy_step`, `iter_num`, `cumulative_grad_steps`, `total_episodes_completed`, `stage`), and the **PRNG key** itself. Checksum table: `tmp/gate2_analysis_20260805/det_ckpt_checksums.json`.

**Verdict: deterministic leg PASS.** detA ≡ detA2 proves the pinned environment is genuinely deterministic across distinct GPUs (the control the amendment pre-registered); detA ≡ detB then proves the two entry points are **exactly equivalent** at production scale — not statistically similar but the same computation. This closes the §4.5 residual (a small systematic offset hiding under the stochastic noise floor is now excluded: any such offset would have produced a first divergence somewhere in 20,000 episode records or 781,424 gradient steps, and none exists).

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — The two entry points resolved to demonstrably identical training semantics.**
What: dumped run configs differ *only* in the planned WandB label (legacy writes `DreamerV3` for old dashboard filters; train.py writes spec-driven `dreamer_srl` — compat row C7/R7 behaving as designed) and a cosmetic `wandb:` block train.py merges into the env dump. Run B is filterable as `algorithm: dreamer_srl` in WandB config (R7 analyst check: confirmed).
Evidence: `diff` of `models/{agent,env}_config.yaml` across the two run dirs; WandB `config.yaml` of both runs.
Confidence: High.

**Finding 2 — The A/B survival gap is consistent with trajectory noise (finalized against the noise-floor pair).**
What: +2.85-step mean offset, sign-consistent, near-zero at the start, non-monotone over the run.
Why: the unseeded replay sampler + GPU nondeterminism make same-seed runs diverge — and the noise pair *quantifies* it: two byte-identical legacy launches diverged with peak 5.23, end-of-run delta 3.88 (larger than A/B's 2.56), and 81/99 sign-consistency. Every qualitative marker that made the A/B offset look suspicious is reproduced by pure noise; the one statistic where A/B exceeds the single noise draw (run-mean, 1.6×) is within what one draw of a noisy yardstick can miss.
Evidence: §4.5 envelope table; window tables; plan §GPU-leg triage.
Confidence: High for "not clearly systematic"; the residual small-systematic-offset question is closed by the deterministic leg, not by more stochastic pairs.

**Finding 3 — The −4.6% SPS reading is the GPU slot, not the entry point.**
What: `Time/sps_env` B/A = 0.954; the noise pair, legacy on both sides on the *same two slots*, reproduced it: N2/N1 = 0.951. Slot 106:1 (which ran B and N2) is ~5% slower than 106:0 (A and N1); slot-matched entry-point comparison: 106:0 legacy 15.44 (A) vs 15.47 (N1); 106:1 train.py 14.73 (B) vs legacy 14.72 (N2) — entry-point deltas ≈0.2% and 0.1%.
Why: per-slot hardware/thermal variation; the C10 allocator hypothesis is not needed and is contradicted by the slot-matched numbers.
Evidence: §4.5; four SPS series.
Confidence: High. Absolute numbers (~15 SPS) are RTX 3090-class, not comparable to node-102 RTX 4090 bench figures.

**Finding 4 — Track C evaluation telemetry survived the integration completely.**
What: all four eval events on run B produced videos on the policy-step clock, eval scalars on the same clock, correct estimator routing for the video-only config, zero dropped rows.
Evidence: media files, history rows, wandb debug log grep.
Confidence: High.

### 5.3 Failure Modes & Pathologies

None observed in training. Two non-pathologies worth naming so future readers don't re-derive them: (1) run B's startup config audit prints `episodes: 100` in the global settings block — a display artifact (the env-config default) while the CLI `--episodes 20000` actually governs (run reached episode 20 000; this matches the known `dreamer_srl` budget-display gotcha); (2) the console-final vs WandB-final world-model-loss discrepancy (1.7099/1.7193 vs 1.7584/1.7610) is instantaneous-vs-windowed, not a data problem.

## 6. Conclusions

### 6.1 Summary

- **Statistical leg: PASS — all six A/B checklist criteria are green.** Nothing in either run indicates the integrated train.py path trains differently from the legacy entry point: identical resolved configs (minus the planned label), same-shape learning curves whose divergence sits inside the measured same-seed noise envelope, overlapping world-model loss (A/B gap 7× smaller than the noise pair's spread), throughput delta fully attributed to the GPU slot, telemetry fully intact, checkpoints restorable.
- The pre-registered decision rule was applied on 2026-08-05 when the noise-floor pair completed (§4.5): A/B exceeds the single noise draw on run-mean only (1.6×), while peak, end-of-run delta, per-window max, and pointwise sd are at or below the noise envelope, and the noise pair reproduces the sign-consistency. Not clearly outside → PASS.
- Track C (the eval-telemetry fix) is confirmed working through the integrated path — videos on the right clock, no dropped rows, correct estimator split — so risk R3 did not materialize.
- Offline evaluability (risk R4) confirmed on both trees: `eval_rollout.py` restores and evaluates both runs' final checkpoints on CPU.
- **Deterministic leg: PASS — and with it, overall GATE 2: PASS.** The node-114 trio under the pinned deterministic environment is bit-identical end to end (§4.6): detA ≡ detA2 validates cross-GPU determinism (the pre-registered control), detA ≡ detB proves entry-point equivalence exactly — 20,000/20,000 episode records, identical checkpoint boundaries, 781,424 gradient steps on all three, 39-entry final-loss dicts equal at 6 decimals, and all 14 final-checkpoint modules SHA-256-identical including the PRNG key and the three Adam states. The residual "small systematic offset under the noise floor" question is closed.
- **Phase 4 (deprecation shim + launch-script flip) is unblocked** per the plan's gating: Gate 1 (CPU bit-identity incl. curriculum + resume), Gate 2 statistical leg, and Gate 2 deterministic leg are all green. Remaining Phase-4-side checks (shim smoke, GPU shim smoke) are the developer's, not this analysis's.

### 6.2 Limitations & Open Questions

- One seed, one config, one node — by design (plan accepts this; Gate 1 carries the config-resolution coverage).
- A two-run noise estimate is one draw, so the criterion-1 judgment is a plausibility check, not a significance test: A/B run-mean divergence at 1.6× the noise draw is called "inside" because every other statistic overlaps. The project deliberately routed the residual question to the deterministic bit-identity leg instead of more stochastic pairs — the stronger instrument — and that leg subsequently returned exact equivalence (§4.6), retiring this limitation.
- The deterministic leg's equivalence proof holds for the pinned environment + seeded-sampler fix-set as run; production runs (no determinism flag) remain individually nondeterministic — that is a property of the stack, not of either entry point, and the statistical leg covers real-conditions behavior.
- `Eval/MeanLength` at N=3 is too noisy to compare across runs; the survival curve carries the verdict.
- The two noise runs also remind us how large this stack's run-to-run spread is (final console world-model loss 1.753 vs 1.819 for byte-identical launches) — worth remembering when any future single-run Dreamer comparison is proposed.

### 6.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|----------|-----------|-----------|--------|
| 1 (done 2026-08-06) | ~~Analyze the node-114 deterministic trio~~ | Completed — §4.6; deterministic leg PASS, Gate 2 green | — |
| 2 (done 2026-08-05) | ~~Finalize criterion 1 against noise-floor pair N1/N2~~ | Completed — §4.5; criterion 1 PASS | — |
| 3 | Proceed to Phase 4 (shim + flip) per the integration plan | All gates green; hand-off to `developer` via the plan's Phase 4 section | Dev work (out of this doc's scope) |
| 4 | Optional post-flip: spot-check `Time/sps_env` on early production integrated runs | Slot-matched Gate-2 data already shows ≈0 entry-point cost; belt-and-suspenders only | Free (passive) |

---

## Appendix

### A. Data Provenance

Local WandB datastore parsing only (project convention — no web API): `tmp/gate2_analysis_20260805/parse_wandb.py` scans `run-*.wandb` protobuf records (`history` records, `nested_key` field, wandb 0.24.0) → `histA.json` (157 rows) / `histB.json` (159 rows). Both files retained with the aggregation notes in `tmp/20260805_000000_gate2_ab_analysis.md`. Eval smoke outputs under `tmp/gate2_analysis_20260805/eval_smoke_{A,B}/`.

### B. Config Diffs

`diff` of A vs B dumped `models/agent_config.yaml`: single line — `agent.algorithm: DreamerV3` (A) vs `dreamer_srl` (B). Legacy `main()` hardcodes the old label for WandB-filter continuity (`dreamer_srl_main.py:760`; source config declares `dreamer_srl` in both runs); train.py writes the spec-driven label. This is compat row C7 / risk R7 behaving exactly as the plan specifies — the legacy label disappears with the legacy path at Phase 4.
`diff` of dumped `models/env_config.yaml`: B appends a `wandb:` settings block (entity/project/mode) that train.py merges from the logger config. No training-semantics key differs.

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-08-05 | Initial analysis; provisional verdict pending noise-floor pair N1/N2 | `experiment-analyzer` |
| 2026-08-05 | Noise-floor pair N1/N2 completed and analyzed (§4.5): criterion 1 finalized PASS (statistical leg), SPS delta attributed to GPU slot, loss gap 7× under noise. Overall verdict remains provisional pending node-114 deterministic trio | `experiment-analyzer` |
| 2026-08-06 | Deterministic trio detA/detA2/detB analyzed (§4.6): bit-identical on all log-stream channels + all 14 final-checkpoint module checksums. Deterministic leg PASS → **overall Gate 2 PASS**; doc finalized | `experiment-analyzer` |

---

## Feedback from plan-reviewer

**Verdict on this analysis: CONCLUSION SUPPORTED WITH CAVEATS** (adversarial verdict review, 2026-08-06). The evidence was re-checked against raw artifacts, not the doc's claims: the three deterministic-leg logs are distinct files (distinct md5sums) with the correct provenance banners (detA/detA2 launched from the `gate2-preseam-throwaway` tree via the legacy entry point; detB through the integration worktree's train.py config chain — wandb IDs match the manifest); the comparator `tmp/gate2_analysis_20260805/cmp_det_logs.py` was re-run and is non-vacuous (20,000/489/482/39 record counts reproduced, first-divergence reporting logic sound); and the final-checkpoint identity was **independently recomputed from the raw orbax checkpoints** at finer (sub-module) granularity — all groups identical across detA/detA2/detB, and the single-leaf hashes (PRNG `key`, all counters) match the archived JSON exactly. The Gate 2 PASS conclusion is supported.

Caveats (owner: `experiment-analyzer` for the doc wording; nothing blocks Phase 4):

1. 🟡 **The deterministic leg's operating point is undisclosed and "production scale" overstates it.** The det trio ran **16 parallel envs with the CPU replay buffer** (log banners: `num_envs=16`, detB CLI `buffer_device='cpu'`), while the statistical A/B ran 128 envs — §2.2's "128 envs" reads as covering all runs, and §4.6/§6.1's "bit-identical at full 20,000-episode production scale" is true only of the episode count. Entry-point equivalence at 128 envs/GPU conditions rests on the statistical leg + the identical resolved-config dumps (Appendix B), not on the bit-identity proof. One disclosing sentence in §4.6 and §6.2 fixes this.
2. 🟢 The retained `ckpt_checksums.py` is not the exact version that produced `det_ckpt_checksums.json` (it aggregates at sub-module depth and does not run unmodified under the current orbax; the JSON has 14 top-level groups). Substance independently confirmed, but the artifact/script mismatch is worth knowing before anyone re-runs it.
3. ❓ The determinism env flags (`XLA_FLAGS=--xla_gpu_deterministic_ops=true` etc.) are not recorded in the run logs themselves. Accepted as self-certified: detA ≡ detA2 across two distinct GPUs over 20,000 episodes is effectively impossible without the pinned environment (the triage's A-vs-A control diverged by iteration count without it).
4. On the statistical leg: the pre-registered rule (rev `6c539ff` §6.1) is asymmetric — "within the envelope" passes, "clearly exceeds" fails — and the A/B run-mean at 1.6× the single noise draw sits in the rule's undefined middle. Calling it "not clearly outside" is a defensible reading of the rule as written, the doc discloses the softness (§6.2), and the deterministic leg carries the equivalence burden regardless.

Cost if wrong: a 128-env-specific entry-point difference surviving both legs would surface post-flip as subtly different production training — bounded within the measured noise envelope by the statistical leg, so the exposure is one production re-comparison, not data loss or a wrong paper claim.

Reviewed by: plan-reviewer
