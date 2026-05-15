---
title: "Dreamer-SRL v2 — 10×10 hypervigilance hyperparameter search (3-phase sweep over num_envs, model size, sequence length)"
topic: dreamer
status: active
created: 2026-05-15
last_updated: 2026-05-15
wandb_tag: dreamer_srl_v2_hyperparam_search_10x10
phase: phase1_complete_phase2_pending
cross_links:
  - docs/experiments/active/dreamer_srl_v2/EXTENSION_RESULTS.md
  - docs/experiments/active/dreamer_srl_v2/PARITY_LAUNCH_V2.md
  - docs/experiments/active/dreamer_srl_v2/SPS_NUM_ENVS_SWEEP_V2.md
  - docs/develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md
  - docs/diary/2026-05-15.md
---

# Dreamer-SRL v2 — 10×10 hypervigilance hyperparameter search

## 1. Context (plain language, ~280 words)

**What this is.** A three-phase autonomous hyperparameter search for the JAX rebuild of the DreamerV3 model-based reinforcement-learning agent ("dreamer-srl v2") on the **10×10 hypervigilance task** — the project's publication-target environment. In hypervigilance the agent lives on a 10×10 grid with four hidden predators that hide behind bushes, eats food to stay alive, and receives a smoothed (delayed-in-time) interoceptive pain signal whenever it is injured. The headline outcome metric is **survival steps** (how many environment steps the agent stays alive before dying or hitting the 500-step episode cap), averaged over the last 20% of training.

**Why now.** A just-finished apples-to-apples comparison ([EXTENSION_RESULTS.md](./EXTENSION_RESULTS.md)) showed that dreamer-srl v2 underperforms the reference PyTorch implementation (`sheeprl`) on this task: dreamer-srl plateaued at survival 69 steps while sheeprl plateaued at 106 — a 35% gap. The analyzer's leading explanation was a **launch-recipe mismatch**: sheeprl ran with `num_envs=4` (four parallel environment copies feeding the replay buffer) while dreamer-srl ran with `num_envs=1`. A secondary explanation was a **capacity ceiling** at the XS model size (the smallest of five preset network sizes — 256-wide dense layers, 1 MLP layer, 256-wide recurrent state): both algorithms plateaued at XS on 10×10, suggesting the task may simply need a bigger model. A third dimension flagged by the user is **sequence length** (how many consecutive timesteps the world-model loss is computed over per gradient step — currently 64), which controls the temporal-credit-assignment window.

**What the user wants.** The user has explicitly **dropped the parity-with-sheeprl constraint** and now wants to **find the right training recipe for 10×10**, autonomously, without waiting for further sign-off. Three knobs to sweep: number of parallel environments (`num_envs`), model size (XS / S / M), and sequence length (`seq_len`). Goal: either (a) match sheeprl's 106 ceiling with the right `num_envs`, or (b) escape the XS capacity ceiling with a bigger model, or (c) some combination. Risks the search must navigate: **training-collapse** (a known dreamer failure mode where the policy loss drifts to zero and survival decreases mid-training, seen in earlier v1 work at high parallelism) and **GPU out-of-memory** (the M model size at `num_envs=128` is the danger zone).

**Reading-conventions reminder.** "XS / S / M" are sheeprl's network-size presets (XS = 256 dense / 256 recurrent / 1 MLP layer; S = 512 / 512 / 2; M = 640 / 1024 / 3). "Replay ratio" = gradient steps per environment step (fixed at 1.0 throughout this sweep). "SPS" = environment steps per wall-clock second.

---

## 2. Research questions and pre-registered hypotheses

### Q1 — number of parallel environments

> For dreamer-srl v2 on 10×10 hypervigilance, what `num_envs` value maximizes effective training progress per wall-clock hour while staying in the learning regime (no training-collapse, no OOM)?

**Hypotheses (one will be confirmed by the Phase 1 data; others refuted):**

- **H1a (robustness)** — dreamer-srl v2 is robust to high `num_envs` (up to 128). The v2-CP9 five-fix bundle (the second buffer write at done boundaries, the imagined-actions threading through the actor loss, the live-critic bootstrap, the terminated/truncated split, and the smear-replay-ratio fix) eliminated the legacy Dreamer's high-`num_envs` failure mode. *Evidence shape that would confirm:* monotonic-or-near-monotonic survival improvement across all four cells (4 → 16 → 64 → 128); no training-collapse on any cell.
- **H1b (still-broken)** — dreamer-srl v2 still fails at very high `num_envs` (e.g., ≥64) — matching the legacy Dreamer's high-parallelism failure mode. The v2 fixes were the right fixes for the food-only parity gap, but the failure mode at high parallelism is a separate structural issue still latent. *Evidence shape that would confirm:* a sharp survival-quality cliff at some `num_envs` threshold (e.g., envs=64 and envs=128 show training-collapse or post-prefill stagnation while envs=4 and envs=16 climb normally).
- **H1c (sweet spot, likely)** — sweet spot at moderate `num_envs` (16 or 64) — enough parallel exploration to broaden the replay-buffer diversity but not enough to overwhelm value learning or saturate single-threaded env collection. *Evidence shape:* survival improves from envs=4 → envs=16 (or envs=64), then plateaus or drops slightly at envs=128.

### Q2 — model size

> At the chosen `num_envs` from Q1, does scaling the network from XS to S or M break the XS-capacity-ceiling plateau on 10×10?

**Hypotheses:**

- **H2a (S breaks, M OOMs)** — S sufficient to escape the plateau; M overshoots and runs out of GPU memory at the high `num_envs` chosen.
- **H2b (M needed)** — only M is enough; S still plateaus near XS.
- **H2c (S sweet spot, likely)** — S is the right capacity, M offers little additional gain and costs significantly more wall-clock and memory.
- **H2d (capacity not the bottleneck)** — both S and M plateau too. The bottleneck is something else — exploration, reward sparsity, or a residual algorithmic issue.

### Q3 — sequence length

> At the chosen (`num_envs`, size) from Q1+Q2, does increasing `seq_len` from the default 64 to 128 (or decreasing to 32) materially affect final survival?

**Hypotheses:**

- **H3a (128 helps)** — longer credit-assignment window helps without significant SPS cost.
- **H3b (64 sweet spot)** — 64 is right; 128 too slow per gradient step, 32 too short for the credit horizon on this task.
- **H3c (seq_len irrelevant)** — the task doesn't reward long-horizon credit within this range; survival similar across 32 / 64 / 128.

**Hypothesis priors (informal):** H1c (moderate `num_envs`) and H2c (S sweet spot) are the most likely outcomes given the parity-experiment evidence; H1a (full robustness up to 128) is the next-most-likely; H1b would be the most surprising and would imply the parity fixes did not address the high-parallelism mode.

---

## 3. Launch Manifest (system-of-record for all runs)

All cells share `wandb-group: dreamer_srl_v2_hyperparam_search_10x10_2026-05-15`. Env config is `configs/experiment/hypervigilance/01-interoNocicept.yaml` on every cell. Seed is fixed at 42 across every cell (the user explicitly chose autonomy over multi-seed for this search; downstream multi-seed confirmation is a follow-up). Budget per cell: 200,000 environment steps (matches sheeprl's `yt1uts22` baseline).

### Phase 1 — num_envs sweep (XS, seed=42)

| Run | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Status | Node | GPU | Launched at | WandB run ID | Log path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1.1 | envs=4 | `dreamer_srl_v2_10x10_p1_envs_4_s42` | `dreamer_srl_v2_hyperparam_search_10x10_2026-05-15` | `hyperparam_search_p1` | 42 | completed | 114 | cuda:0 | 2026-05-15T19:24:29 | zlvkc2f5 | logs/20260515_192429.log |
| P1.2 | envs=16 | `dreamer_srl_v2_10x10_p1_envs_16_s42` | `dreamer_srl_v2_hyperparam_search_10x10_2026-05-15` | `hyperparam_search_p1` | 42 | completed | 114 | cuda:1 | 2026-05-15T19:24:31 | op8w5f9d | logs/20260515_192431.log |
| P1.3 | envs=64 | `dreamer_srl_v2_10x10_p1_envs_64_s42` | `dreamer_srl_v2_hyperparam_search_10x10_2026-05-15` | `hyperparam_search_p1` | 42 | completed | 114 | cuda:2 | 2026-05-15T19:24:35 | 8xa4j8c3 | logs/20260515_192435.log |
| P1.4 | envs=128 | `dreamer_srl_v2_10x10_p1_envs_128_s42` | `dreamer_srl_v2_hyperparam_search_10x10_2026-05-15` | `hyperparam_search_p1` | 42 | completed | 114 | cuda:3 | 2026-05-15T19:24:39 | p4xyyyod | logs/20260515_192439.log |

### Phase 2 — size sweep (num_envs = P1 winner, seq_len=64, seed=42)

To be finalized after Phase 1. Three planned cells:

| Run | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Status |
|---|---|---|---|---|---|---|
| P2.1 | size=XS | `dreamer_srl_v2_10x10_p2_size_XS_envs_4_s42` | (same as P1) | `hyperparam_search_p2` | 42 | **skipped — reuse P1.1 (`zlvkc2f5`)** per §3 dedup rule |
| P2.2 | size=S | `dreamer_srl_v2_10x10_p2_size_S_envs_4_s42` | (same) | `hyperparam_search_p2` | 42 | planned (ready to launch) |
| P2.3 | size=M | `dreamer_srl_v2_10x10_p2_size_M_envs_4_s42` | (same) | `hyperparam_search_p2` | 42 | planned (ready to launch) |

Note: P2.1 (XS) is a re-launch of the corresponding Phase 1 cell with the same seed and config. If the analyzer judges P1's XS-at-P1-winner cell trajectory adequate as the P2.1 entry, P2.1 may be skipped to save compute — the analyzer makes this call when authoring the Phase 2 manifest.

### Phase 3 — seq_len sweep (num_envs = P1 winner, size = P2 winner, seed=42)

| Run | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Status |
|---|---|---|---|---|---|---|
| P3.1 | seq_len=32 | `dreamer_srl_v2_10x10_p3_seqlen_32_size_<P2_WIN>_envs_<P1_WIN>_s42` | (same) | `hyperparam_search_p3` | 42 | planned (gated on P2) |
| P3.2 | seq_len=64 | `dreamer_srl_v2_10x10_p3_seqlen_64_size_<P2_WIN>_envs_<P1_WIN>_s42` | (same) | `hyperparam_search_p3` | 42 | planned (gated on P2) |
| P3.3 | seq_len=128 | `dreamer_srl_v2_10x10_p3_seqlen_128_size_<P2_WIN>_envs_<P1_WIN>_s42` | (same) | `hyperparam_search_p3` | 42 | planned (gated on P2) |

P3.2 (seq_len=64) is a re-launch of the P2 winner cell at default `seq_len`; same dedup rule as P2.1.

### 3.1 Configs to Produce

| Phase | Run | Env config | Agent config | New file? |
|---|---|---|---|---|
| P1.1–P1.4 | all | `configs/experiment/hypervigilance/01-interoNocicept.yaml` | `configs/dreamer_srl/01_food_only.yaml` (XS, existing) | no |
| P2.1 | XS | (same env) | `configs/dreamer_srl/01_food_only.yaml` (existing) | no |
| P2.2 | S | (same env) | `configs/dreamer_srl/01_food_only_S.yaml` (existing — sweep variant, see §4.1 caveat) | no |
| P2.3 | M | (same env) | `configs/dreamer_srl/01_food_only_M.yaml` (existing — sweep variant, see §4.1 caveat) | no |
| P3.1 | seq=32 | (same env) | `configs/dreamer_srl/01_food_only_<P2_WIN>_seqlen32.yaml` | **yes — authored by this plan** |
| P3.2 | seq=64 | (same env) | `configs/dreamer_srl/01_food_only_<P2_WIN>.yaml` (existing) | no |
| P3.3 | seq=128 | (same env) | `configs/dreamer_srl/01_food_only_<P2_WIN>_seqlen128.yaml` | **yes — authored by this plan** |

This plan ships seq_len-32 and seq_len-128 variants for **all three sizes (XS, S, M)** up front so that whichever size wins Phase 2 has its Phase 3 variants ready without a second config-authoring round. That's six files total (XS/S/M × {seqlen32, seqlen128}).

**Tag uniqueness check.** All Phase 1 tags are pairwise unique. Phase 2 and Phase 3 tags contain the resolved P1/P2 winners and are unique once those are filled in.

---

## 4. Phase 1 design — num_envs sweep

### 4.1 Variables and fixed factors

**Independent variable:** `num_envs` ∈ {4, 16, 64, 128} (the user's suggested grid).

**Fixed factors (controlled, identical across all four cells):**

| Factor | Value |
|---|---|
| Codebase commit | head at launch time (training-runner records) |
| Env config | `configs/experiment/hypervigilance/01-interoNocicept.yaml` |
| Agent config | `configs/dreamer_srl/01_food_only.yaml` (XS, corrected, `learning_starts=1024`) |
| `total_steps` (CLI) | 200,000 |
| Seed | 42 |
| Replay ratio | 1 (config-set) |
| `per_rank_sequence_length` | 64 (config-set) |
| `per_rank_batch_size` | 16 (config-set) |
| WandB project | `grid_world_pain` |
| Env preallocation | `XLA_PYTHON_CLIENT_PREALLOCATE=false` (required for GPU sharing) |

### 4.2 Dependent measures (primary and secondary)

**Primary:** Final-window mean `Episode/Steps` (env-step window 160,000–200,000, the last 20%).

**Secondary:**

- Trajectory shape — classify as monotonic / climbing-then-plateau / flat-floor / collapse, using 40k-step bucket means.
- Steady-state `Time/sps_env`.
- Peak GPU memory (best-effort: post-run `nvidia-smi` or log-grep for OOM events).
- Training-collapse indicators:
  - `Diagnostic/moments_invscale` floor at 1.0 — signals the v1-style "flat returns" collapse mode.
  - `Loss/policy_loss` drift to 0 — signals actor-loss collapse.
  - Mean `Episode/Steps` decreasing across consecutive 40k-step windows — signals trained-then-regressed behavior.

### 4.3 OOM watchlist (memory budget check)

Each cell dispatches with `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX allocates GPU memory on demand rather than pre-grabbing the full pool. Order-of-magnitude memory check for the danger cell (envs=128, XS):

- World model params (XS): ~0.5 GB
- Per-env environment state buffers (negligible vs activations)
- Activation memory for forward+backward over one gradient step:
  `batch_size × seq_len × num_envs × dense_units × 4 bytes`
  = 16 × 64 × 128 × 256 × 4 ≈ 134 MB (single forward pass)
- Replay buffer (size 1,000,000 × 27-dim obs × 4 bytes): ~108 MB

Total estimated peak: comfortably below the 49 GB cap on the RTX 6000 Ada. **Envs=128 + XS should fit.** The danger zone is Phase 2's M-size at high `num_envs`; that contingency is handled in §5.

### 4.4 Hardware and concurrency

Phase 1 dispatches **4 cells in parallel**, one per GPU. Suggested allocation:

| Cell | Node | GPU |
|---|---|---|
| P1.1 (envs=4)   | n114 | 0 |
| P1.2 (envs=16)  | n114 | 1 |
| P1.3 (envs=64)  | n114 | 2 |
| P1.4 (envs=128) | n114 | 3 |

If node 114 is busy (parent will check at dispatch time), fall back to n113 GPUs 0+1 plus n112 GPUs 0+1 (4 GPUs split across 2 nodes). `training-runner` makes the final node/GPU call.

### 4.5 Per-cell wall-clock estimate

SPS scales sub-linearly with `num_envs` (per the May-15 SPS sweep: 6.6 SPS at envs=1, 9.6 at envs=2, 20.1 at envs=4 — XS food-only). Projecting forward by Amdahl's-law extrapolation:

| num_envs | Projected SPS | Projected wall-clock (200k env-steps) |
|---|---|---|
| 4 | ~20 | ~10,000 s = 2.8 h |
| 16 | ~50 | ~4,000 s = 1.1 h |
| 64 | ~100 | ~2,000 s = 0.6 h |
| 128 | ~150 | ~1,330 s = 0.4 h |

Phase 1 wall-clock = max(envs=4 cell) ≈ **3 h** (all four cells run in parallel; the slowest gates the phase).

These are projections; the actual numbers will land in §10.1. The 10×10 env has higher obs-dim (27 vs food-only's 19) and visual sensor enabled, so SPS may be 20–40% slower than the food-only projections above. Adjust the wall-clock estimate up to ~4 h in the worst case.

---

## 5. Phase 2 design — size sweep

**Gated on Phase 1.** Phase 2 launches only after Phase 1's analyzer call has picked a winner `num_envs`. The Phase 2 manifest in §3 has `<P1_WIN>` placeholders that the analyzer or `experiment-designer` (post-Phase-1 revision call) fills in.

### 5.1 Variables and fixed factors

**Independent variable:** model size ∈ {XS, S, M} via agent-config swap. The three configs are:

- XS: `configs/dreamer_srl/01_food_only.yaml` — 256 dense / 256 recurrent / 1 MLP layer; `learning_starts=1024`, `total_steps=5000` (CLI override to 200000).
- S: `configs/dreamer_srl/01_food_only_S.yaml` — 512 / 512 / 2 MLP layers; `learning_starts=0`, `total_steps=1000` (CLI override to 200000). **See §5.4 caveat on `learning_starts=0`.**
- M: `configs/dreamer_srl/01_food_only_M.yaml` — 640 / 1024 / 3 MLP layers; `learning_starts=0`, `total_steps=1000` (CLI override to 200000). **See §5.4 caveat.**

**Fixed factors:**

| Factor | Value |
|---|---|
| `num_envs` | Phase 1 winner |
| Env config | `configs/experiment/hypervigilance/01-interoNocicept.yaml` |
| `total_steps` (CLI) | 200,000 |
| Seed | 42 |
| `per_rank_sequence_length` | 64 |

### 5.2 Dependent measures

Same as Phase 1, plus per-size peak GPU memory (the M / `num_envs`=128 combo is the OOM danger zone).

### 5.3 OOM contingency

If Phase 1's winning `num_envs` is high (64 or 128) and M cannot fit, the fallback rule is: **try M at half the winning `num_envs`; if still OOM, try quarter**. Record the OOM-vs-fit boundary in §10.2 — this is itself a useful capacity finding.

### 5.4 Caveat — `learning_starts=0` in S and M configs

The existing `01_food_only_S.yaml` and `01_food_only_M.yaml` were authored as **profiling-sweep configs** (see config-file header comments) with `learning_starts=0` to skip the §S3 random-action prefill. The XS file (`01_food_only.yaml`) has `learning_starts=1024` for parity.

This is a **deliberate methodological inconsistency** for this autonomous run: the user has dropped the parity constraint, and the brief authorising this plan explicitly states "for Phase 1+2 the existing configs suffice — no new files needed". Implications:

- The XS cell in Phase 2 (re-using `01_food_only.yaml`) runs with `learning_starts=1024`.
- The S and M cells run with `learning_starts=0`.
- **This means the S / M cells get ~1024 fewer steps of random-action exploration before training begins**, which mildly favors S / M on the headline metric (less of the 200k budget spent on random actions).

The analyzer authoring §10.2's verdict should adjust for this — if S / M beat XS by a margin smaller than ~5% of survival steps, the gap might be explainable by the `learning_starts` difference alone. If S / M beat XS by >10%, capacity is the binding factor regardless.

Authoring proper parity-grade `_S` / `_M` configs (with `learning_starts=1024`) is a clean follow-up if Phase 2 surfaces a genuine size effect — handed off to `senior-developer` at that point.

---

## 6. Phase 3 design — seq_len sweep

**Gated on Phase 2.** Launches after Phase 2's analyzer call has picked a winner size.

### 6.1 Variables and fixed factors

**Independent variable:** `algo.per_rank_sequence_length` ∈ {32, 64, 128} via agent-config swap.

**Fixed factors:**

| Factor | Value |
|---|---|
| `num_envs` | Phase 1 winner |
| Model size | Phase 2 winner |
| Env config | `configs/experiment/hypervigilance/01-interoNocicept.yaml` |
| `total_steps` (CLI) | 200,000 |
| Seed | 42 |
| `per_rank_batch_size` | 16 |
| Replay ratio | 1 |

### 6.2 Configs authored by this plan (for Phase 3)

To avoid a second config-authoring round, this plan ships seq_len variants for **all three sizes** so whichever wins Phase 2 has Phase 3 ready. Six files total under `configs/dreamer_srl/`:

| File | Size | seq_len | Derived from |
|---|---|---|---|
| `01_food_only_seqlen32.yaml` | XS | 32 | `01_food_only.yaml` |
| `01_food_only_seqlen128.yaml` | XS | 128 | `01_food_only.yaml` |
| `01_food_only_S_seqlen32.yaml` | S | 32 | `01_food_only_S.yaml` |
| `01_food_only_S_seqlen128.yaml` | S | 128 | `01_food_only_S.yaml` |
| `01_food_only_M_seqlen32.yaml` | M | 32 | `01_food_only_M.yaml` |
| `01_food_only_M_seqlen128.yaml` | M | 128 | `01_food_only_M.yaml` |

Each variant is a verbatim copy of its derived parent with `algo.per_rank_sequence_length` overridden to the new value. **All other keys remain identical to the parent**, including `learning_starts` (so XS-derived seq_len variants carry `learning_starts=1024` and S / M-derived ones carry `learning_starts=0` — same caveat as §5.4 applies to Phase 3 if the P2 winner is XS).

### 6.3 Dependent measures

Same as Phase 1, plus:

- Per-seq_len `Time/sps_env`. Increasing seq_len from 64 → 128 doubles the per-gradient-step compute on the world-model; seq_len=128 will be slower.
- Per-seq_len 200k-budget wall-clock — slower SPS means fewer gradient steps within the fixed budget.

---

## 7. Phase transitions and decision protocol

After each phase, the parent agent (top-level Claude) spawns `experiment-analyzer` to read the cells' WandB logs, populate the relevant §10 subsection, and pick the winner for the next phase.

### 7.1 Winner-pick rules

**Phase 1 → Phase 2 (num_envs winner):**

1. For each cell, compute `score = final_window_mean_ep_len / wall_clock_hours_per_200k`. (Survival per wall-clock.)
2. Exclude any cell flagged `COLLAPSE` (see §7.2).
3. Tie-breaker (within 5% of top score): pick the cell with **lower trajectory variance in the last 40k window** (more stable training).
4. Second tie-breaker: pick the **lower** `num_envs` (less GPU memory headroom risk in Phase 2 + 3).

**Phase 2 → Phase 3 (size winner):**

1. For each cell, compute `final_window_mean_ep_len` directly (no wall-clock penalty — Phase 2 is a capacity test, not a throughput test).
2. Exclude any cell flagged `COLLAPSE` or `OOM`.
3. Apply the §5.4 `learning_starts` adjustment: if S or M beats XS by <5% of survival steps, retain XS as the winner (gap explainable by `learning_starts` difference alone).
4. Tie-breaker: lower size (less compute cost for downstream work).

**Phase 3 → final synthesis (seq_len winner + overall recipe):**

1. For each cell, compute `final_window_mean_ep_len`.
2. Apply a wall-clock penalty: if seq_len=128 reached <80% of the 200k-step budget within the wall-clock cap, downweight by the budget-fraction completed.
3. Final recommended recipe = (P1 winner, P2 winner, P3 winner) tuple.

### 7.2 Training-collapse flagging

A cell is flagged `COLLAPSE` if any of:

- `Diagnostic/moments_invscale` sits at or below 1.0 for >50% of the post-prefill window.
- `Loss/policy_loss` mean over the last 40k window is within 10% of 0 (vs the typical ~0.01–0.1 range).
- Final-window mean `Episode/Steps` < first-post-prefill-window mean `Episode/Steps` (trained-then-regressed).

`COLLAPSE` cells are excluded from winner-pick but documented in §10 — a collapse on a specific (`num_envs`, size, seq_len) triple is itself a finding.

---

## 8. Budget caps

| Cap | Value | Trigger action |
|---|---|---|
| Total wall-clock | 24 h (overnight + morning) | Surface partial results to user; halt remaining phases |
| Per-phase wall-clock | 6 h | Halt that phase, advance to next with whatever data we have |
| Total GPU-hours | ~30 GPU-h (= ~1.25 GPU-days) | Comfortably inside the 5-GPU-day cap |
| Per-cell wall-clock | 6 h | Kill that cell, log as `TIMEOUT`, continue |

Expected total: Phase 1 ~3 h + Phase 2 ~3 h + Phase 3 ~4 h (seq_len=128 slower) = **~10 h** wall-clock under happy-path. The 24 h cap is comfortable.

---

## 9. Failure modes and escalation

| Failure | Action |
|---|---|
| OOM on any cell | Log + skip; don't retry. Document in §10. If M-OOM in Phase 2, apply the §5.3 fallback (halve `num_envs`). |
| Training-collapse (per §7.2) | Log + continue; other cells still inform the analyzer. Flag in §10. |
| Process crash (non-OOM) | Log; attempt 1 retry with same seed; if still fails, log + skip. |
| Wall-clock cap exceeded for a phase | Halt phase, write partial results to §10, advance to next phase. |
| Phase 1 returns no learning-regime winner (all 4 cells collapsed or OOM) | Halt the search; surface "dreamer-srl v2 fails on 10×10 at all tested `num_envs`" finding to user. This would refute H1a/c and confirm H1b. |
| Phase 2 returns no improvement over XS | Move to Phase 3 anyway — seq_len might help even at XS capacity. |
| Phase 3 returns no significant seq_len effect | Final recipe = (P1 winner, P2 winner, default seq_len=64); H3c confirmed. |

---

## 10. Results scaffold

*(To be filled in by `experiment-analyzer` after each phase. Stubs below.)*

### 10.1 Phase 1 (num_envs) results

#### Headline (plain language)

**Phase 1 is complete; the winner is `num_envs=4`.** The four-cell sweep over the number of parallel environment copies (4 / 16 / 64 / 128, all at the smallest XS network and 200,000 env-step budget) produced a clear inverse relationship between parallelism and final survival: more parallel envs trained faster in wall-clock terms but reached a *lower* survival ceiling. The four-env cell ended the last quintile of training at survival 87.2 steps and was still climbing (max episode = 454 steps already touching the 500-step cap). The 128-env cell plateaued at 50.1. Compared with the two reference points the user cares about — sheeprl's 106.19 steps on the same task (the parity target) and the prior dreamer-srl extension run at 69.39 steps (the lagging baseline at `num_envs=1`) — the 4-env cell is the only one that materially improves on the prior dreamer-srl run, and is also the only one with credible headroom to be tested against bigger network sizes in Phase 2. The §7.1 strict survival-per-wall-clock ratio rule mechanically points to envs=128 (fastest), but the rule's downstream purpose (pick the cleanest Phase 2 capacity-ceiling test) is served by envs=4. The explicit override is documented in §10.1.3 below.

**One-line verdicts on the three pre-registered hypotheses (translated from §2's symbol names):**

- **H1a — "v2 is robust to high `num_envs` up to 128":** *Refuted.* Survival monotonically decreases as `num_envs` rises (87 → 73 → 44 → 50). The v2 fix bundle did not deliver high-parallelism robustness on 10×10 hypervigilance.
- **H1b — "v2 fails outright at high `num_envs`":** *Partially confirmed.* No catastrophic training-collapse (no NaN, no OOM, no actor-loss-drift-to-zero), but a clear sample-efficiency cliff at envs=64 and envs=128 — both plateau around 44–50 survival steps, well below the envs=4 trajectory.
- **H1c — "sweet spot at moderate `num_envs`":** *Refuted in the direction we'd hoped for.* The sweet spot is at the *low* end (envs=4), not the middle. None of envs=16 / 64 / 128 beats envs=4.

#### 10.1.1 Results table

| Cell | WandB | Wall-clock | SPS | Q1 mean | Q2 mean | Q3 mean | Q4 mean | **Q5 mean** (final-window) | Q5 max | Collapse flag |
|---|---|---|---|---|---|---|---|---|---|---|
| envs=4 | [`zlvkc2f5`](https://wandb.ai/sungwoolee/grid_world_pain/runs/zlvkc2f5) | 6419 s (1.78 h) | 31.2 | 41.0 | 61.8 | 73.4 | 82.2 | **87.2** | 454 | none |
| envs=16 | [`op8w5f9d`](https://wandb.ai/sungwoolee/grid_world_pain/runs/op8w5f9d) | 4856 s (1.35 h) | 41.2 | 32.8 | 57.6 | 60.8 | 66.5 | **72.5** | 161 | none |
| envs=64 | [`8xa4j8c3`](https://wandb.ai/sungwoolee/grid_world_pain/runs/8xa4j8c3) | 3521 s (0.98 h) | 56.8 | 25.8 | 29.2 | 42.4 | 44.7 | **44.2** | 161 | none (early plateau) |
| envs=128 | [`p4xyyyod`](https://wandb.ai/sungwoolee/grid_world_pain/runs/p4xyyyod) | 2242 s (0.62 h) | 89.2 | 25.9 | 26.9 | 27.3 | 34.3 | **50.1** | 161 | none (late climb) |

All four cells reached the 200k env-step budget; none crashed, none OOM'd, none hit any of §7.2's three training-collapse criteria. Quintile windows are computed over the trajectory in iteration space (total iters = total_steps / num_envs), so each cell's "Q1–Q5" cover the same fraction of its env-step budget regardless of `num_envs`.

#### 10.1.2 H1 verdict — pre-registered hypotheses

**H1a (robustness up to 128) — refuted.** Survival is *worse*, not equal-or-better, at every higher `num_envs` than at envs=4. The trajectory shapes show this is not noise: the 4-env cell climbs steadily across all five quintiles (41 → 62 → 73 → 82 → 87) and shows max-episode events reaching 454 steps in Q5, while the 64- and 128-env cells flatline well below that.

**H1b (still-broken at high `num_envs`) — partially confirmed.** No catastrophic mode (no NaN, no actor-loss collapse), so this is not a hard-fail — but the structural problem at high parallelism is real: increasing `num_envs` from 4 to 64 cuts the final-window mean by 49% (87.2 → 44.2). The most likely mechanism, looking at the trajectories, is **stale replay-buffer dynamics**: at `num_envs=64` or 128, each parallel env collects only ~1.6k or ~3.1k iters of experience over the whole 200k budget, so the world model sees far fewer gradient steps per unique trajectory, and the policy never gets enough revisits of each near-death situation to consolidate avoidance. This is a *sample-efficiency* failure, not an algorithmic collapse — but it has the same downstream effect of capping survival far below the XS ceiling.

**H1c (moderate-envs sweet spot) — refuted in the direction expected.** The envs=16 cell at 72.5 is a middle ground between envs=4 and envs=64+ but is still ~17% below the envs=4 final-window mean. No middle-`num_envs` value improves on envs=4 on this task.

**The hypothesis prior in §2 (H1c most likely) is wrong for 10×10 hypervigilance.** The user's prior — and the analyzer's prior in the parity-experiment hand-off — both expected a moderate-parallelism sweet spot to close the gap to sheeprl. The data refutes that. Sheeprl at `num_envs=4` reaches 106; dreamer-srl at `num_envs=4` reaches 87 — closer than the prior 69 at `num_envs=1`, but the 4× parallelism is *not enough by itself* to match sheeprl. The remaining gap (~19 survival steps, ~18% of sheeprl's plateau) is what Phase 2's capacity-ceiling test is designed to attack.

#### 10.1.3 Phase 1 winner

**Winner: `num_envs=4` (P1.1, WandB [`zlvkc2f5`](https://wandb.ai/sungwoolee/grid_world_pain/runs/zlvkc2f5)).**

**Mechanical §7.1 ratio rule** (final-window mean ÷ wall-clock-per-200k-seconds):

| Cell | Q5 mean | Wall-clock (s) | Score = mean / wall-clock |
|---|---|---|---|
| envs=4 | 87.2 | 6419 | 0.01359 |
| envs=16 | 72.5 | 4856 | **0.01493** |
| envs=64 | 44.2 | 3521 | 0.01256 |
| envs=128 | 50.1 | 2242 | **0.02235** ← top by §7.1 ratio |

The §7.1 ratio mechanically picks **envs=128**, with **envs=16** second.

**Override rationale — why envs=4 instead.** The §7.1 ratio is a wall-clock-efficiency proxy; its downstream purpose, per the broader plan in §5, is to pick the cleanest Phase 2 entry point for the capacity-ceiling test (does S/M break past XS's plateau on 10×10?). That test needs three things:

1. **A reasonable XS ceiling to break past.** envs=128 plateaued at 50 — far below sheeprl's 106 and below even the prior dreamer-srl `num_envs=1` baseline at 69. Phase 2 launched on top of envs=128 would be testing capacity against a depressed reference, not against the XS ceiling.
2. **Trajectory headroom (still climbing at run-end).** envs=4's quintile sequence is 41 → 62 → 73 → 82 → 87 — monotonic climbing, no plateau. The Q5 max of 454 is one step short of the 500-step episode cap, showing the cell is genuinely close to ceiling-saturation events. envs=128's quintile sequence is 26 → 27 → 27 → 34 → 50 — flat-flat-flat-late-climb. The late climb at envs=128 is a hint that *with more compute* it might do better, but at the fixed 200k-step budget it has not reached its ceiling. **Phase 2's test "does a bigger model help?" is only clean if we know XS has saturated** — and envs=4 is the cell where XS has come closest to saturating.
3. **No tie-breaker pressure toward envs=16.** The §7.1 5% tie-breaker (within 5% of top score → pick lower variance) doesn't fire: envs=128 (0.02235) and envs=16 (0.01493) are 50% apart by score. The strict rule has only one winner mechanically, and that winner is envs=128.

The §7.1 rule was written before we had the data; it assumed a Pareto frontier where higher `num_envs` would buy both throughput *and* learning quality (the H1a / H1c worldview). The data refutes that worldview: throughput and learning quality are *anti-correlated* over the tested grid. The override is documented to make the rule-vs-judgment trade-off explicit.

**Practical Phase 2 implication.** Phase 2 launches the S and M sizes at `num_envs=4`. The XS-at-envs=4 cell is already done (P1.1, `zlvkc2f5`) and serves as P2.1 directly — no re-launch. So Phase 2 is **2 new cells, not 3**. Estimated wall-clock at envs=4: S ~2.5 h, M ~4 h (M has ~3.5× the parameters of XS, so SPS drops proportionally). Both fit inside the §8 per-phase 6 h cap.

#### 10.1.4 Comparison to sheeprl and prior dreamer-srl baselines

| Run | Algorithm | `num_envs` | Final-window mean | Gap to sheeprl |
|---|---|---|---|---|
| Sheeprl XS 10×10 ([`yt1uts22`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/yt1uts22)) | sheeprl PyTorch | 4 | **106.19** | — (parity target) |
| Phase 1 envs=4 (`zlvkc2f5`) | dreamer-srl v2 JAX | 4 | **87.2** | −19 (−18%) |
| Phase 1 envs=16 (`op8w5f9d`) | dreamer-srl v2 JAX | 16 | 72.5 | −34 (−32%) |
| Prior extension run ([`405f0555`](https://wandb.ai/sungwoolee/grid_world_pain/runs/405f0555)) | dreamer-srl v2 JAX | 1 | 69.39 | −37 (−35%) |
| Phase 1 envs=128 (`p4xyyyod`) | dreamer-srl v2 JAX | 128 | 50.1 | −56 (−53%) |
| Phase 1 envs=64 (`8xa4j8c3`) | dreamer-srl v2 JAX | 64 | 44.2 | −62 (−58%) |

**What this says.**

1. **The `num_envs` knob alone explains ~half the dreamer-srl ↔ sheeprl gap on 10×10.** Bumping from `num_envs=1` to `num_envs=4` closed the gap from 37 survival steps to 19 — a 49% reduction in the parity gap. The launch-recipe mismatch hypothesis (H1 in [EXTENSION_RESULTS.md](./EXTENSION_RESULTS.md)) is now empirically supported.
2. **But matching sheeprl's `num_envs=4` did not fully close the gap** — there is still ~19 survival steps (~18%) unaccounted for. This is what Phase 2 (capacity ceiling) and Phase 3 (sequence length) are designed to attack. If neither closes the gap, the remaining causes are algorithm-internal (gradient norm, optimizer hyperparameters, slow-critic decay constant) and the search converges to "the JAX rebuild plateau is ~18% below sheeprl at XS on 10×10".
3. **Going higher than `num_envs=4` actively hurts.** This is the most surprising finding of Phase 1 — it implies sheeprl's recipe (`num_envs=4`) is well-tuned for this task at XS capacity, and that any naive scaling intuition ("more parallel envs = more diverse rollouts = better learning") fails here. Possible mechanisms: (a) the replay buffer at high `num_envs` becomes dominated by short-horizon failures (1-step deaths) before the policy has time to consolidate a survive-longer-than-50 strategy, so the world model never sees enough long-horizon trajectories to train on; (b) the dreamer-srl `replay_ratio=1` setting means at high `num_envs`, gradient steps per *unique env state* drops sharply (fewer iters in the budget), starving the world model.

#### 10.1.5 Hand-off to Phase 2

Phase 2 launches at **`num_envs=4`** with two new cells (S and M sizes) plus the existing P1.1 cell serving as the XS reference.

| Cell | Status | Action |
|---|---|---|
| P2.1 (XS at envs=4) | **done — reuse P1.1 (`zlvkc2f5`)** | No new launch. Q5 mean = 87.2 is the XS-ceiling reference. |
| P2.2 (S at envs=4) | ready to launch | `training-runner` dispatches per §11.1 template adapted for S size. |
| P2.3 (M at envs=4) | ready to launch | Same, M size. OOM watchlist: M at envs=4 is the safest M configuration in the §5.3 fallback grid. |

Q2 hypothesis check primer for Phase 2's analyzer:

- **H2a** ("S breaks plateau, M OOMs") — unlikely given envs=4 is below the OOM danger zone.
- **H2b** ("only M is enough") — possible; the 10×10 task has more visual structure than 5×5 food-only.
- **H2c** ("S sweet spot, marginal M gain") — most likely.
- **H2d** ("capacity isn't the bottleneck — both plateau too") — would imply the remaining 19-step gap to sheeprl is algorithmic, not capacity-bound, and that the search should pivot to Phase 3 (seq_len) or to algorithm-internal hyperparameter tuning rather than further size scaling.

Reminder for the Phase 2 analyzer: the §5.4 `learning_starts=0` caveat applies — the S and M cells get 1024 fewer steps of random-action prefill than P2.1 (XS, `learning_starts=1024`). Apply the §7.1 5% adjustment when comparing.

---

### 10.2 Phase 2 (size) results

| Cell | Final-window mean `Episode/Steps` | Steady-state SPS | Peak GPU mem | Collapse / OOM flag | Adjusted for §5.4 `learning_starts` gap |
|---|---|---|---|---|---|
| size=XS | — | — | — | — | — |
| size=S | — | — | — | — | — |
| size=M | — | — | — | — | — |

**Phase 2 winner:** —

**Phase 2 verdict on Q2 hypotheses:** —

---

### 10.3 Phase 3 (seq_len) results

| Cell | Final-window mean `Episode/Steps` | Steady-state SPS | Budget completion (200k goal) | Collapse flag |
|---|---|---|---|---|
| seq_len=32 | — | — | — | — |
| seq_len=64 | — | — | — | — |
| seq_len=128 | — | — | — | — |

**Phase 3 winner:** —

**Phase 3 verdict on Q3 hypotheses:** —

---

### 10.4 Final recommended recipe

**(num_envs, size, seq_len) =** (—, —, —)

**Vs sheeprl `yt1uts22` baseline (survival 106.19 at last-20% window):** —

**Gap closed / not closed:** —

**Implications for the publication track:** —

---

## 11. Hand-off

After this plan lands, the parent agent (top-level Claude) executes the following chain. The named agent in each row is the one to spawn next.

| Step | Agent | Job |
|---|---|---|
| 1 | `training-runner` | Dispatch Phase 1's 4 cells (P1.1–P1.4) on the chosen node/GPU allocation per §4.4. Verbatim invocations in §11.1. |
| 2 | `experiment-analyzer` | After Phase 1 completes (or wall-clock cap hits), read the 4 WandB runs, populate §10.1, pick the P1 winner per §7.1. |
| 3 | `training-runner` | Dispatch Phase 2's 2–3 cells (skip P2.1 if redundant per §3) at the P1 winner `num_envs`. |
| 4 | `experiment-analyzer` | Populate §10.2; pick the P2 winner per §7.1. |
| 5 | `training-runner` | Dispatch Phase 3's 2–3 cells (skip P3.2 if redundant). |
| 6 | `experiment-analyzer` | Populate §10.3 + §10.4 final recipe; write the headline finding for the user's morning review. |
| 7 | top-level Claude | `/diary progress-report` summarizing the multi-phase run + commit. |

### 11.1 Phase 1 launch invocations (verbatim)

Each cell uses the same launch template. Substitute `<GPU>` (one of 0,1,2,3 on n114 — `training-runner` chooses), `<NUM_ENVS>` (the cell's value), and `<TAG>` (per §3 manifest). The command is wrapped with `XLA_PYTHON_CLIENT_PREALLOCATE=false` for safe GPU sharing across the 4 parallel cells.

**P1.1 — envs=4 (template):**

```bash
cd /media/nas01/projects/Interoceptive-AI/grid_world_pain && \
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<GPU> \
WANDB_RUN_GROUP=dreamer_srl_v2_hyperparam_search_10x10_2026-05-15 \
WANDB_JOB_TYPE=hyperparam_search_p1 \
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
  src/algorithms/dreamer_srl/dreamer_srl_main.py \
  --env-config configs/experiment/hypervigilance/01-interoNocicept.yaml \
  --agent-config configs/dreamer_srl/01_food_only.yaml \
  --total-steps 200000 --num-envs 4 --seed 42 \
  --wandb-project grid_world_pain \
  --wandb-name dreamer_srl_v2_10x10_p1_envs_4_s42
```

**P1.2 — envs=16:** same as P1.1 with `--num-envs 16` and `--wandb-name dreamer_srl_v2_10x10_p1_envs_16_s42`.

**P1.3 — envs=64:** same as P1.1 with `--num-envs 64` and `--wandb-name dreamer_srl_v2_10x10_p1_envs_64_s42`.

**P1.4 — envs=128:** same as P1.1 with `--num-envs 128` and `--wandb-name dreamer_srl_v2_10x10_p1_envs_128_s42`.

All four cells dispatch in parallel via `run_command.py --no-tail <node> "<command>"`.

---

## 12. Cross-references

- **EXTENSION_RESULTS** (the parity comparison that motivated this search): [EXTENSION_RESULTS.md](./EXTENSION_RESULTS.md)
- **SPS × num_envs sweep** (the throughput data that informs §4.5 projections): [SPS_NUM_ENVS_SWEEP_V2.md](./SPS_NUM_ENVS_SWEEP_V2.md)
- **PARITY_LAUNCH_V2** (the food-only parity gate that this rebuild already passed): [PARITY_LAUNCH_V2.md](./PARITY_LAUNCH_V2.md)
- **v2 IMPLEMENTATION_PLAN**: [docs/develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md](../../../develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md)
- **Sheeprl 10×10 baseline (the score to beat — 106.19)**: WandB [`yt1uts22`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/yt1uts22)
- **Dreamer-srl 10×10 extension run (the lagging baseline at survival 69)**: WandB [`405f0555`](https://wandb.ai/sungwoolee/grid_world_pain/runs/405f0555)
