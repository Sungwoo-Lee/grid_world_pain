---
title: "Gate 2 A/B analysis — dreamer_srl legacy entry point vs. integrated train.py dispatch"
topic: dreamer_integration
status: active
created: 2026-08-05
last_updated: 2026-08-05
phase: "Gate 2 (of 3) — real GPU A/B"
wandb_tag: "dsrl_gate2*"
develop_link: docs/develop/active/dreamer/DREAMER_SRL_TRAIN_PY_INTEGRATION_PLAN.md
---

# Gate 2 A/B Analysis — Dreamer legacy entry point vs. integrated train.py dispatch

> **Status**: ANALYZING — **PROVISIONAL VERDICT, not final.** The survival-curve criterion cannot be finalized until the A-vs-A noise-floor pair (two identical legacy launches, running now) completes and supplies the run-to-run-noise yardstick.
> **Date**: 2026-08-05
> **Author**: `experiment-analyzer`
> **Related**: [[DREAMER_SRL_TRAIN_PY_INTEGRATION_PLAN]] (Gate 2 section = authoritative pass criteria; worktree copy carries all amendments) · [[DREAMER_SRL_EVAL_TELEMETRY_FIX]] (Track C semantics checked here) · [[DREAMER_SRL_TRAIN_PY_DIVERGENCE_MATRIX]]

---

## 1. Research Question

The project is retiring the Dreamer world-model agent's private launch script and folding it into the shared training entry point (`train.py`) that every other algorithm already uses. Before anyone's launch scripts change, the integration must pass three gates; this document is the analysis for **Gate 2** — one real GPU training run launched each way, compared end to end. The question is simple: **does a full-length Dreamer training run launched through the new integrated entry point behave the same as one launched through the old legacy script — same learning curves, same world-model convergence, same speed, with all the recently fixed evaluation telemetry (videos, eval metrics on the right step clock) intact and its checkpoints restorable for offline evaluation?**

A crucial piece of context shapes what "the same" can mean: **same-seed Dreamer runs are not reproducible on this stack** (the replay-buffer sampler is unseeded — a registered known issue — and GPU float nondeterminism feeds back into episode lengths and thus the whole trajectory). Two identical invocations of the *same* entry point diverge. The comparison is therefore **statistical, not exact**: the two runs must overlap within same-seed run-to-run noise. To measure that noise directly, a dedicated **noise-floor pair** — the legacy entry point launched twice with the identical command — is running now; its inter-run divergence is the yardstick against which this A/B difference finalizes.

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
| N1 | running | noise floor 1/2 (legacy twice) | `dsrl_gate2N1_legacy_b04M_s42_n106` | 42 | 106:0 | 2026-08-05T00:08 | `u2156tb7` | — |
| N2 | running | noise floor 2/2 (legacy twice) | `dsrl_gate2N2_legacy_b04M_s42_n106` | 42 | 106:1 | 2026-08-05T00:08 | `breewovl` | — |

Artifacts: run A results `results/JAX_DreamerSRL/20260804-052357_dsrl_gate2A_legacy_b04M_s42_n106` (main tree); run B results `.claude/worktrees/agent-a18eeb59fa7ffffeb/results/JAX_DreamerSRL/20260804-052333_dsrl_gate2B_trainpy_b04M_s42_n106` (worktree tree). Both local WandB dirs live under the worktree's `wandb/` (`run-20260804_052355-3wsfk1ca`, `run-20260804_052404-1zbgtsxp`). Analysis used **local WandB datastore files only** (no web API); parser + extracted history in `tmp/gate2_analysis_20260805/`, working notes in `tmp/20260805_000000_gate2_ab_analysis.md`.

## 4. Results

### 4.1 Primary Metrics — Gate 2 checklist

Headline metric is **survival steps** (`Episode/Steps`, window-averaged, on the episode axis; 100 common logging points at episodes 200…20 000).

| # | Gate 2 criterion | Finding | Status |
|---|---|---|---|
| 1 | Survival curves overlap within same-seed run-to-run noise | Same shape (both ≈17 → ≈42–44 steps). B sits above A by **+2.85 steps on average** (pointwise B−A: sd 1.65, range −0.29…+5.96); last-20% means A 40.8 ± 0.9 vs B 43.4 ± 0.5 (≈+6%). Offset is **sign-consistent** (B ≥ A at 99/100 points) but starts near zero (first 2 000 episodes: +0.3) and compounds mid-run — the signature of trajectory divergence, not of a config difference. Whether +2.85 is inside the same-seed noise band is **exactly what the noise-floor pair measures** | **PROVISIONAL PASS** — finalizes against N1/N2 |
| 2 | `WorldModel/loss_model` same-shape convergence, no systematic offset | Same shape (A 1.881→1.758, B 1.905→1.761, WandB windowed). Fraction-of-run-aligned offset B−A = **+0.009 ± 0.011** (≈0.5% of the loss value), sign flips along the curve → no systematic offset | **PASS** |
| 3 | `Time/sps_env` ratio within ±5% | A mean 15.44 (n=56), B mean 14.73 (n=60) → B/A = **0.954 (−4.6%)**; median ratio 0.955. Within the ±5% policy, but marginal and consistent in sign (see Finding 3) | **PASS (marginal)** |
| 4 | Track C intact on run B (videos, `Eval/*` on policy-step axis, estimator split, no dropped rows) | 4 eval videos present in run B's media (`media/videos/eval/video_<policy_step>_*.mp4` at policy steps 141 952 / 349 056 / 555 648 / 774 016 — the backward-step fix holds); `Eval/MeanLength`, `Eval/MeanReward`, `eval/video`, `eval/checkpoint_episode` all logged at `step=policy_step` at checkpoints 5 001/10 003/15 002/20 000; estimator split correct for this config (stats pass off → video pass writes `Eval/Mean*`, and no `Eval/video/*` keys appear — exactly the telemetry-fix routing table's "stats off" row); **zero** "dropped row"/backward-step warnings in wrapper log, console log, and wandb debug log | **PASS** |
| 5 | Run-B dir passes `eval_rollout.py` checkpoint-restore smoke | Final checkpoint (episode 20 000) restored on CPU from the worktree checkout and evaluated: 5 episodes, mean survival 43.0 steps (consistent with B's training-end ≈43.8). Control on A from the main checkout: mean 42.8 (training-end ≈41.6) | **PASS** (A control also pass) |
| 6 | No anomalies in logs/metrics | Both runs completed 20 000 episodes and exited cleanly; identical metric schemas; no errors/NaN/gaps; only the long-known orbax `CheckpointManager` deprecation warning. Two benign, *expected* config-dump deltas (§Appendix B) and one display artifact (§5.3) | **PASS** |

> **Verdict on H₀ (integration transparent): PROVISIONALLY SUPPORTED.** Five of six criteria pass outright; the survival-curve criterion passes provisionally — shape and early-run agreement are exactly what a transparent integration predicts, but the sign-consistent +2.85-step offset can only be classified as noise vs. systematic once the A-vs-A noise-floor pair (N1/N2, running) provides the same-seed divergence yardstick. **Gate 2 must not be declared passed until that comparison is made.**

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

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — The two entry points resolved to demonstrably identical training semantics.**
What: dumped run configs differ *only* in the planned WandB label (legacy writes `DreamerV3` for old dashboard filters; train.py writes spec-driven `dreamer_srl` — compat row C7/R7 behaving as designed) and a cosmetic `wandb:` block train.py merges into the env dump. Run B is filterable as `algorithm: dreamer_srl` in WandB config (R7 analyst check: confirmed).
Evidence: `diff` of `models/{agent,env}_config.yaml` across the two run dirs; WandB `config.yaml` of both runs.
Confidence: High.

**Finding 2 — The A/B survival gap looks like trajectory noise, but only the noise-floor pair can prove it.**
What: +2.85-step mean offset, sign-consistent, but near-zero at the start and non-monotone over the run.
Why: the unseeded replay sampler + GPU nondeterminism make same-seed runs diverge (plan's GPU-leg triage demonstrated even A-vs-A control-flow divergence); an entry-point semantics bug would instead bite from episode 1.
Evidence: window table §4.4; plan §GPU-leg triage.
Confidence: Medium until N1/N2 complete — a sign-consistent 6–7% end-of-run offset is also what a small systematic difference would look like, so this is deliberately not called closed.

**Finding 3 — Throughput is within policy but B is consistently ~4.6% slower per env step.**
What: `Time/sps_env` B/A = 0.954 (means), 0.955 (medians) — inside ±5% but one-sided across the whole run.
Why (candidates): compat row C10 (train.py sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, a planned, watched delta — allocator overhead is the plan's own predicted mechanism); B's longer episodes shift the env-step/train-step mix; per-GPU silicon variation between 106:0 and 106:1.
Evidence: 56/60-point SPS series.
Confidence: Medium. Note the absolute numbers (~15 SPS) are RTX 3090-class and not comparable to the node-102 RTX 4090 bench figures. N1/N2 (both legacy, same two GPUs) will incidentally show how much of the gap is GPU-slot rather than entry-point.

**Finding 4 — Track C evaluation telemetry survived the integration completely.**
What: all four eval events on run B produced videos on the policy-step clock, eval scalars on the same clock, correct estimator routing for the video-only config, zero dropped rows.
Evidence: media files, history rows, wandb debug log grep.
Confidence: High.

### 5.3 Failure Modes & Pathologies

None observed in training. Two non-pathologies worth naming so future readers don't re-derive them: (1) run B's startup config audit prints `episodes: 100` in the global settings block — a display artifact (the env-config default) while the CLI `--episodes 20000` actually governs (run reached episode 20 000; this matches the known `dreamer_srl` budget-display gotcha); (2) the console-final vs WandB-final world-model-loss discrepancy (1.7099/1.7193 vs 1.7584/1.7610) is instantaneous-vs-windowed, not a data problem.

## 6. Conclusions

### 6.1 Summary

- **Provisional Gate 2 verdict: PASS-pending-noise-floor.** Nothing in either run indicates the integrated train.py path trains differently from the legacy entry point: identical resolved configs (minus the planned label), same-shape learning curves with near-identical early training, overlapping world-model loss, throughput within the ±5% policy, telemetry fully intact, checkpoints restorable.
- The single open item is quantitative, not qualitative: is a +2.85-step mean survival offset (sign-consistent, ≈6% at end of run) within the same-seed A-vs-A divergence of this stack? The noise-floor pair (two identical legacy launches, running since 2026-08-05 00:08) answers exactly this.
- Track C (the eval-telemetry fix) is confirmed working through the integrated path — videos on the right clock, no dropped rows, correct estimator split — so risk R3 did not materialize.
- Offline evaluability (risk R4) confirmed on both trees: `eval_rollout.py` restores and evaluates both runs' final checkpoints on CPU.
- Decision rule on completion of N1/N2: compute the same 100-point pointwise |N1−N2| survival divergence profile; if the A/B offset profile (mean +2.85, peak +5.96) lies within the noise pair's envelope (comparable mean/peak), criterion 1 flips to PASS and Gate 2 is green → Phase 4 (shim + flip) is unblocked. If A/B clearly exceeds the noise floor, Gate 2 fails and the offset must be investigated as systematic before any flip.

### 6.2 Limitations & Open Questions

- One seed, one config, one node — by design (plan accepts this; Gate 1 carries the config-resolution coverage).
- A two-run noise estimate is itself noisy: N1/N2 give one draw of the A-vs-A divergence, so the final judgment is a plausibility check, not a significance test. If N1−N2 divergence lands ambiguously close to the A/B offset, a second noise pair (or a B-vs-B pair) would be the cheap disambiguator.
- The −4.6% SPS delta is within policy but unexplained in detail; if it persists in future integrated runs on other cards it deserves a targeted look (C10 allocator setting is the first suspect).
- `Eval/MeanLength` at N=3 is too noisy to compare across runs; the survival curve carries the verdict.

### 6.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|----------|-----------|-----------|--------|
| 1 (blocking) | Finalize criterion 1 against noise-floor pair N1/N2 when it completes (~1 day) | The only item keeping the Gate 2 verdict provisional | Analysis only |
| 2 | If N1/N2 is ambiguous: one B-vs-B pair (train.py twice) | Symmetric noise estimate on the integrated path | 2 GPU-days |
| 3 | Post-flip: watch `Time/sps_env` on the first few production integrated runs | Confirm the −4.6% is GPU-slot/noise, not a persistent C10 cost | Free (passive) |

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
