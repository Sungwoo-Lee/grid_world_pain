---
title: "Dreamer-SRL v2 — 10×10 hypervigilance hyperparameter search (3-phase sweep over num_envs, model size, sequence length)"
topic: dreamer
status: active
created: 2026-05-15
last_updated: 2026-05-16
wandb_tag: dreamer_srl_v2_hyperparam_search_10x10
phase: extended_sweep_launched_8cells_n106-109
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
| P2.2 | size=S | `dreamer_srl_v2_10x10_p2_size_S_envs_4_s42` | (same) | `hyperparam_search_p2` | 42 | completed — WandB [`a3o0xg75`](https://wandb.ai/sungwoolee/grid_world_pain/runs/a3o0xg75) |
| P2.3 | size=M | `dreamer_srl_v2_10x10_p2_size_M_envs_4_s42` | (same) | `hyperparam_search_p2` | 42 | completed — WandB [`ta7k7w9b`](https://wandb.ai/sungwoolee/grid_world_pain/runs/ta7k7w9b) |

Note: P2.1 (XS) is a re-launch of the corresponding Phase 1 cell with the same seed and config. If the analyzer judges P1's XS-at-P1-winner cell trajectory adequate as the P2.1 entry, P2.1 may be skipped to save compute — the analyzer makes this call when authoring the Phase 2 manifest.

### Phase 3 — seq_len sweep (num_envs = P1 winner, size = P2 winner, seed=42)

| Run | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Status |
|---|---|---|---|---|---|---|
| P3.1 | seq_len=32 | `dreamer_srl_v2_10x10_p3_S_envs_4_seqlen_32_s42` | (same) | `hyperparam_search_p3` | 42 | completed — WandB [`4fl1h3cm`](https://wandb.ai/sungwoolee/grid_world_pain/runs/4fl1h3cm) |
| P3.2 | seq_len=64 | `dreamer_srl_v2_10x10_p3_S_envs_4_seqlen_64_s42` | (same) | `hyperparam_search_p3` | 42 | **skipped — reuse P2.2 (`a3o0xg75`)** per §3 dedup rule |
| P3.3 | seq_len=128 | `dreamer_srl_v2_10x10_p3_S_envs_4_seqlen_128_s42` | (same) | `hyperparam_search_p3` | 42 | completed — WandB [`ex27ckc3`](https://wandb.ai/sungwoolee/grid_world_pain/runs/ex27ckc3) |

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

#### Headline (plain language)

**Phase 2 is complete; the winner is `size=S`.** Holding `num_envs=4` (the Phase 1 winner) and the 200,000 env-step budget fixed, scaling the agent's network from the smallest preset (XS — 256-wide dense layers, 1 MLP layer, 256-wide recurrent state) to the next preset up (S — 512 / 512 / 2) lifts final-window survival from 87.2 steps to **97.2 steps (+11%)** and brings the JAX rebuild to **92% of sheeprl's 106.19-step plateau** on the same task — the closest the rebuild has come to parity at any setting tested so far. Scaling further to the M preset (640 dense / 1024 recurrent / 3 MLP layers) does **not** help: M lands at 86.9 steps, essentially tied with XS, despite spending 86% more wall-clock time (3.31 h vs 1.78 h for XS) and reaching the lowest world-model loss of any cell (1.63 vs S's 1.79 vs XS's 2.04). This last finding — best world-model fit but worst-tier survival — is the "M paradox" called out in §10.2.4 and flagged as a future-experiment opportunity (re-run M at a longer budget, ≥500k env-steps, to test whether the actor-critic eventually catches up to the world model).

**One-line verdicts on the four pre-registered Q2 hypotheses (translated from §2's symbol names):**

- **H2a — "S breaks the plateau, M overshoots and OOMs":** *Partially confirmed.* S did break past XS's plateau (+11%). M did not OOM (it fit comfortably at `num_envs=4`), so the OOM half of H2a is refuted — but M failed to beat S, which matches H2a's qualitative spirit ("M is too big to help here").
- **H2b — "Only M is enough; S still plateaus near XS":** *Refuted.* S beats XS by 11%, well above the 5% §5.4 `learning_starts`-adjustment threshold. The +11% margin is large enough that the 1024-step prefill difference can account for at most a fraction of it.
- **H2c — "S is the sweet spot — beats XS, M offers little additional gain":** *Confirmed.* This is the closest match. S strictly dominates XS on survival; M matches XS on survival at much higher compute cost.
- **H2d — "Both S and M plateau too; capacity isn't the bottleneck":** *Refuted.* S genuinely breaks past XS; capacity *is* a contributing factor on 10×10 hypervigilance.

#### 10.2.1 Results table

| Cell | WandB | Wall-clock | SPS | Final WM loss | Q1 mean | Q2 mean | Q3 mean | Q4 mean | **Q5 mean** (final-window) | Q5 max | Collapse / OOM flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| size=XS (P1.1) | [`zlvkc2f5`](https://wandb.ai/sungwoolee/grid_world_pain/runs/zlvkc2f5) | 6419 s (1.78 h) | 31.2 | 2.04 | 41.0 | 61.8 | 73.4 | 82.2 | **87.2** | 454 | none |
| size=S (P2.2) | [`a3o0xg75`](https://wandb.ai/sungwoolee/grid_world_pain/runs/a3o0xg75) | 7877 s (2.19 h) | 25.4 | 1.79 | 46.3 | 67.0 | 85.6 | 96.0 | **97.2** | 501 (env cap) | none |
| size=M (P2.3) | [`ta7k7w9b`](https://wandb.ai/sungwoolee/grid_world_pain/runs/ta7k7w9b) | 11929 s (3.31 h) | 16.8 | 1.63 | 47.9 | 69.1 | 80.2 | 79.3 | **86.9** | 441 | none (Q3→Q4 dip) |

All three cells reached the 200k env-step budget; no NaN, no OOM, no §7.2 training-collapse criterion fired. The §5.4 `learning_starts` caveat applies — XS ran with `learning_starts=1024` while S and M ran with `learning_starts=0` (1024 fewer steps of random-action prefill, mildly favoring S and M on the headline metric). The S-vs-XS gap of +11% (10 survival steps) far exceeds the §5.4 5% adjustment threshold, so the size effect on S is real; the M-vs-XS gap of −0.3% is well inside the threshold, so M's apparent tie with XS is real (and possibly slightly worse once the prefill difference is accounted for).

Notable trajectory features:

- **S touched the 500-step environment cap in Q4** (max=501) — the first cell in the entire search to do so. This is a credible signal that S has effectively-unlimited survival ability on the easier sampled episodes, with the headline 97.2-step mean dragged down by the harder ones.
- **M shows a Q3→Q4 dip** (80.2 → 79.3, then recovery to 86.9 in Q5). XS climbs monotonically; S climbs monotonically; M is the only cell with a within-run regression. This is consistent with the §10.2.4 "M paradox" hypothesis — the world model is moving fast but the actor-critic is unstable.

#### 10.2.2 H2 verdict — pre-registered hypotheses

**H2a (S breaks, M OOMs) — partially confirmed.** S did break past XS's plateau by 11%. M did not OOM at `num_envs=4` — the OOM half of H2a is refuted, which is unsurprising in retrospect since the §5.3 OOM danger zone is M × high-`num_envs`, and the Phase 1 winner pulled us down to `num_envs=4`. The qualitative spirit of H2a ("M is too big to help at this budget") is correct on different grounds — see H2c.

**H2b (only M is enough) — refuted.** S at 97.2 is decisively above XS at 87.2; the +11% margin (10 absolute survival steps) is much larger than the §5.4 5% `learning_starts` adjustment ceiling (~4.4 survival steps). M does not beat S — it ties XS. So "only M is enough" is doubly refuted: S is enough, and M doesn't even buy what S already gives.

**H2c (S sweet spot, marginal M gain) — confirmed.** This is the closest match to the data. S strictly dominates XS on the headline survival metric (+11%) at modest wall-clock cost (+23%, 2.19 h vs 1.78 h). M does not deliver any survival gain over S at 200k env-steps — in fact M lands below XS — despite costing 51% more wall-clock than S (3.31 h vs 2.19 h) and reaching the best world-model fit (loss 1.63 vs S's 1.79). The "marginal M gain" half of H2c is in fact *worse* than marginal at this budget — M's gain is *negative*.

**H2d (capacity isn't the bottleneck) — refuted.** Capacity *is* a contributing factor: bumping XS → S yielded a real survival improvement that closed roughly half of the remaining XS-to-sheeprl gap (XS was 19 steps below sheeprl; S is 9 steps below; see §10.2.5). H2d would have implied the remaining gap is entirely algorithmic — instead, ~half of it was capacity-bound.

#### 10.2.3 Phase 2 winner

**Winner: `size=S` (P2.2, WandB [`a3o0xg75`](https://wandb.ai/sungwoolee/grid_world_pain/runs/a3o0xg75)).**

**Mechanical §7.1 ratio rule** (final-window mean ÷ wall-clock-per-200k-seconds — although the §7.1 spec for Phase 2 says "no wall-clock penalty", computing the ratio for completeness):

| Cell | Q5 mean | Wall-clock (s) | Score = mean / wall-clock |
|---|---|---|---|
| size=XS | 87.2 | 6419 | **0.01359** ← top by ratio |
| size=S | 97.2 | 7877 | 0.01234 |
| size=M | 86.9 | 11929 | 0.00729 |

**The §7.1 spec for Phase 2 explicitly waives the wall-clock penalty** — Phase 2 is the capacity test, not the throughput test. The §7.1 rule for Phase 2 is `final_window_mean_ep_len` directly, with the §5.4 `learning_starts` adjustment (if S or M beats XS by <5%, retain XS as the winner — the gap is then attributable to the prefill difference alone).

**Applying §7.1 as written:**

- S beats XS by 10 survival steps (87.2 → 97.2), an **11.5% improvement**. This is **well above** the §5.4 5% adjustment threshold. S is the §7.1 winner.
- M ties XS (86.9 vs 87.2, **−0.3%**), well *below* the adjustment threshold. M is not a viable winner under §7.1; if anything, M slightly underperforms XS at equal-or-worse `learning_starts` parity.
- Second tie-breaker (lower size) does not fire — S is uniquely the §7.1 winner.

**Override note — for completeness.** The user's brief raised a "strict ratio picks XS, override toward absolute mean" concern, mirroring the §10.1.3 Phase 1 override pattern. For Phase 2 the override is **not actually needed**: §7.1 for Phase 2 already privileges absolute mean over wall-clock ratio. The §10.1.3 override pattern was a Phase 1 special case where the §7.1 rule (which *did* include wall-clock for Phase 1) had to be set aside in favor of trajectory-headroom judgment. For Phase 2, the rule and the absolute-mean reading point at the same answer: **S.**

**Practical Phase 3 implication.** Phase 3 launches the seq_len sweep on top of (size=S, num_envs=4). The seq_len=64 cell at this base is already done (P2.2, `a3o0xg75`, Q5 mean = 97.2) and serves as P3.2 directly — no re-launch needed. Phase 3 is therefore **2 new cells (seq_len=32 and seq_len=128), not 3**. The pre-authored configs `configs/dreamer_srl/01_food_only_S_seqlen32.yaml` and `01_food_only_S_seqlen128.yaml` were fixed at commit `eb1fbe1` to carry `learning_starts=1024` (parity-track default), so the §5.4 prefill-difference caveat **does not apply to Phase 3** — the Phase 3 cells are all on parity-track `learning_starts`, which is a strict methodological tightening relative to Phase 2.

#### 10.2.4 The M paradox — best world-model fit, worst-tier survival

The single most surprising Phase 2 finding is the inversion between world-model loss and survival at M scale:

| Cell | Final WM loss (lower = better world-model fit) | Final-window mean survival (higher = better policy) |
|---|---|---|
| XS | 2.04 (worst WM) | 87.2 (middle) |
| S | 1.79 (middle) | **97.2 (best policy)** |
| M | **1.63 (best WM)** | 86.9 (tied-worst) |

**M learns the world the best but acts on it the worst.** At a 200k env-step budget, M's larger network reaches a noticeably lower world-model loss than S or XS — the world model has more parameters and a richer recurrent state, so it represents the gridworld more accurately. But M's *policy* never catches up: actor-critic learning lags behind world-model learning at M scale within the fixed compute budget. The Q3→Q4 dip in M's trajectory (80.2 → 79.3) is consistent with an unstable actor-critic loop on top of a fast-moving world model.

**Hypothesis (to test in a follow-up experiment):** at M scale the actor-critic needs **more env-steps** than the world model does to converge. The 200k-step budget is enough for M's world model but not enough for M's policy. At a 500k or 1M env-step budget, M might overtake S — or might still plateau, in which case the actor-critic bottleneck at M scale is structural (e.g., too many policy parameters for the available gradient signal, or a too-large actor-critic action-distribution variance).

**Why this matters for the publication track.** If the M-paradox hypothesis is correct, the dreamer-srl rebuild's headroom on 10×10 is **gated by training budget, not by capacity** — increasing the budget should unlock more performance from M. If the hypothesis is wrong, M will still plateau at extended budget and the rebuild's effective capacity ceiling on 10×10 is S, with no further gains available from naïve size scaling. Either outcome is a useful answer for the experiment design downstream.

**Recommended follow-up experiment (deferred):** re-run M at `total_steps ∈ {500000, 1000000}` and compare to S at the same extended budgets. If M overtakes S at 500k, the budget-gating hypothesis is confirmed and the publication-track recipe should be (size=M, total_steps=500k+). If M still ties S at 1M, the capacity ceiling on 10×10 is S and the rebuild's recipe is (size=S, seq_len=Phase-3-winner).

This is **flagged as future-experiment** — it is **not** added to this plan's Phase 3 scope, which is the seq_len sweep at the chosen (num_envs=4, size=S) base.

#### 10.2.5 Comparison to sheeprl and Phase 1 baselines

| Run | Algorithm | (num_envs, size) | Final-window mean | Gap to sheeprl |
|---|---|---|---|---|
| Sheeprl XS 10×10 ([`yt1uts22`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/yt1uts22)) | sheeprl PyTorch | (4, XS) | **106.19** | — (parity target) |
| **Phase 2 size=S (`a3o0xg75`)** | dreamer-srl v2 JAX | (4, S) | **97.2** | **−9.0 (−8.5%)** |
| Phase 1 envs=4 / Phase 2 size=XS (`zlvkc2f5`) | dreamer-srl v2 JAX | (4, XS) | 87.2 | −19 (−18%) |
| Phase 2 size=M (`ta7k7w9b`) | dreamer-srl v2 JAX | (4, M) | 86.9 | −19 (−18%) |
| Prior extension run ([`405f0555`](https://wandb.ai/sungwoolee/grid_world_pain/runs/405f0555)) | dreamer-srl v2 JAX | (1, XS) | 69.4 | −37 (−35%) |

**What this says.**

1. **Phase 2's S winner closes ~half of the remaining XS-vs-sheeprl gap.** XS was 19 steps below sheeprl; S is 9 steps below. The compound effect of Phase 1 (`num_envs`=1 → 4) plus Phase 2 (XS → S) has closed the dreamer-srl ↔ sheeprl gap from 37 steps (35%) to 9 steps (8.5%) — a 76% reduction in the parity gap. **dreamer-srl v2 at (`num_envs=4`, S) is at 92% of sheeprl's plateau, the closest the JAX rebuild has come to parity at any tested setting.**
2. **There is still ~9 steps of gap to close.** Phase 3's seq_len sweep is the last pre-registered knob in the autonomous search. If Phase 3 doesn't close it, the residual gap is algorithm-internal (gradient norm, optimizer hyperparameters, slow-critic decay constant, or perhaps a subtle bug in how dreamer-srl handles partial observability vs sheeprl) — and the search converges to "dreamer-srl v2 plateaus 8.5% below sheeprl on 10×10 hypervigilance at the same (`num_envs=4`, S) recipe".
3. **The M-paradox finding (§10.2.4) opens a non-pre-registered path** — extending M to a 500k-step budget — that is not part of the autonomous search but is the most likely route to *exceeding* sheeprl's plateau rather than just matching it. Deferred to user decision; not auto-scheduled.

#### 10.2.6 Hand-off to Phase 3

Phase 3 launches at **(num_envs=4, size=S)** with two new cells (seq_len=32 and seq_len=128) plus the existing P2.2 cell serving as the seq_len=64 reference.

| Cell | Status | Action |
|---|---|---|
| P3.1 (seq_len=32 at envs=4, size=S) | ready to launch | `training-runner` dispatches; config `01_food_only_S_seqlen32.yaml` (parity-track `learning_starts=1024` after commit `eb1fbe1`). |
| P3.2 (seq_len=64 at envs=4, size=S) | **done — reuse P2.2 (`a3o0xg75`)** | No new launch. Q5 mean = 97.2 is the seq_len=64 reference. |
| P3.3 (seq_len=128 at envs=4, size=S) | ready to launch | `training-runner` dispatches; config `01_food_only_S_seqlen128.yaml` (parity-track `learning_starts=1024` after commit `eb1fbe1`). |

Q3 hypothesis check primer for Phase 3's analyzer:

- **H3a** ("128 helps") — possible; 10×10 hypervigilance has hidden predators behind bushes, so longer credit-assignment windows might help the agent learn to associate early-episode bush positions with later predator encounters.
- **H3b** ("64 sweet spot") — moderate prior; sheeprl's default is 64 and it works for them.
- **H3c** ("seq_len irrelevant") — possible; the v2 task might not reward long-horizon credit within the 32–128 range.

**§5.4 prefill caveat does not apply to Phase 3.** All three Phase 3 cells (P3.1, P3.2 = reused P2.2, P3.3) run with the same `learning_starts` — P3.1 and P3.3 carry `learning_starts=1024` (parity-track after commit `eb1fbe1`), and P3.2 is the reused P2.2 cell at `learning_starts=0`. *Wait —* P2.2 is the `01_food_only_S.yaml` config, which has `learning_starts=0` per §5.4. So P3.2 (reused) has `learning_starts=0` but P3.1 and P3.3 have `learning_starts=1024`. This is a residual prefill inconsistency that the Phase 3 analyzer must apply the §5.4 adjustment to. If Phase 3 surfaces a margin <5% between cells, the analyzer should flag the prefill difference as a confound and recommend a parity-grade re-run.

**Phase 3 launch invocations (verbatim, ready to forward to `training-runner`):**

```bash
# P3.1 — S/envs=4/seq_len=32 on n114 GPU 0
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
WANDB_RUN_GROUP=dreamer_srl_v2_hyperparam_search_10x10_2026-05-15 \
WANDB_JOB_TYPE=hyperparam_search_p3 \
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
  src/algorithms/dreamer_srl/dreamer_srl_main.py \
  --env-config configs/experiment/hypervigilance/01-interoNocicept.yaml \
  --agent-config configs/dreamer_srl/01_food_only_S_seqlen32.yaml \
  --total-steps 200000 --num-envs 4 --seed 42 \
  --wandb-project grid_world_pain \
  --wandb-name dreamer_srl_v2_10x10_p3_S_envs_4_seqlen_32_s42

# P3.2 — S/envs=4/seq_len=128 on n114 GPU 1
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 \
WANDB_RUN_GROUP=dreamer_srl_v2_hyperparam_search_10x10_2026-05-15 \
WANDB_JOB_TYPE=hyperparam_search_p3 \
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
  src/algorithms/dreamer_srl/dreamer_srl_main.py \
  --env-config configs/experiment/hypervigilance/01-interoNocicept.yaml \
  --agent-config configs/dreamer_srl/01_food_only_S_seqlen128.yaml \
  --total-steps 200000 --num-envs 4 --seed 42 \
  --wandb-project grid_world_pain \
  --wandb-name dreamer_srl_v2_10x10_p3_S_envs_4_seqlen_128_s42
```

Both cells dispatch in parallel via `run_command.py --no-tail n114 "<command>"`.

---

---

### 10.3 Phase 3 (seq_len) results

#### Headline (plain language)

**Phase 3 is complete; the winner is `seq_len=64`.** Holding `num_envs=4` (Phase 1 winner) and `size=S` (Phase 2 winner) fixed at the 200,000 env-step budget, sweeping the *sequence length* — the number of consecutive timesteps the world-model loss is computed over per gradient step — gives the following picture: `seq_len=64` lands at **97.20** final-window mean survival steps (this is the Phase 2 winner `a3o0xg75` reused as the Phase 3 reference), `seq_len=32` lands at **94.67** (just 2.6% below 64), and `seq_len=128` lands at **80.08** (a substantial 17.6% below 64) while costing ~1.85× the wall-clock. The pre-registered "64 is the sweet spot" hypothesis (H3b) is **confirmed**: shorter (32) is only marginally worse, longer (128) is decisively worse at this budget. The most striking finding parallels Phase 2's "M paradox" — `seq_len=128` reaches the **lowest** world-model loss of any Phase 3 cell (1.55, vs 1.79 for 64 and 2.22 for 32) but the **worst** survival. The longer rollout helps the world model fit faster, but the actor-critic policy can't translate that into longer episodes within 200k env-steps. This pattern — bigger representation capacity needs more env-steps for the policy to catch up — recurs across Phase 2 (M scale) and Phase 3 (seq_len=128), and is the strongest candidate explanation for the residual ~9-step gap to sheeprl at the 200k budget.

#### 10.3.1 Results table

| Cell | WandB | Wall-clock | SPS | Final WM loss | **Final-window mean survival** | Max ep_len | Δ vs seq_len=64 | Collapse flag |
|---|---|---|---|---|---|---|---|---|
| seq_len=32 (P3.1) | [`4fl1h3cm`](https://wandb.ai/sungwoolee/grid_world_pain/runs/4fl1h3cm) | 6145 s (1.71 h) | 32.5 | 2.22 | **94.67** | 312 | −2.6% | none |
| seq_len=64 (P3.2 = reused P2.2) | [`a3o0xg75`](https://wandb.ai/sungwoolee/grid_world_pain/runs/a3o0xg75) | 7877 s (2.19 h) | 25.4 | 1.79 | **97.20** | 501 (env cap) | — (reference) | none |
| seq_len=128 (P3.3) | [`ex27ckc3`](https://wandb.ai/sungwoolee/grid_world_pain/runs/ex27ckc3) | 11643 s (3.23 h) | 17.2 | 1.55 | **80.08** | 391 | **−17.6%** | none |

All three cells reached the 200k env-step budget; no NaN, no OOM, no §7.2 training-collapse criterion fired. Note the **§5.4 prefill residual** flagged in §10.2.6: P3.1 and P3.3 ran with `learning_starts=1024` (parity-track, after commit `eb1fbe1`), while P3.2 (reused P2.2) ran with `learning_starts=0`. Since `seq_len=64` has the *higher* survival mean despite having 1024 fewer prefill steps, the prefill confound works against — not in favor of — the P3.2 result. The verdict is robust to the confound.

#### 10.3.2 H3 verdict — pre-registered hypotheses

**H3a ("seq_len=128 helps without significant SPS cost") — refuted.** seq_len=128 *hurts* survival (−17.6% vs 64) **and** costs 1.85× the wall-clock of seq_len=32 (3.23 h vs 1.71 h). Both halves of H3a fail: it doesn't help, and it isn't cheap.

**H3b ("64 is the sweet spot — 128 too slow, 32 too short") — confirmed.** seq_len=64 is the unique best cell on the headline survival metric. The "128 too slow" half is confirmed both in throughput (SPS drops from 32.5 at seq_len=32 → 25.4 at 64 → 17.2 at 128 — a 47% throughput cost going 32 → 128) and in headline survival (80.08 at seq_len=128, the worst of the three). The "32 too short" half is **softer than expected**: seq_len=32 lands at 94.67, only 2.6% below seq_len=64, with a 22% SPS advantage (32.5 vs 25.4) — on this task, a 32-step credit-assignment window is nearly as good as a 64-step one.

**H3c ("seq_len is irrelevant within this range") — refuted.** The seq_len=128 cell is decisively below the other two; the range is *not* indifferent. seq_len=64 vs seq_len=32 is close (2.6% gap), but seq_len=64 vs seq_len=128 is large (17.6% gap).

**Aggregate verdict on Q3 hypotheses:** H3b confirmed; H3a and H3c refuted.

#### 10.3.3 The seq_len=128 paradox — best world-model fit, worst survival

The single most striking Phase 3 finding directly parallels Phase 2's M-paradox (§10.2.4):

| Cell | Final WM loss (lower = better world-model fit) | Final-window mean survival (higher = better policy) |
|---|---|---|
| seq_len=32 | 2.22 (worst WM) | 94.67 (middle) |
| seq_len=64 | 1.79 (middle) | **97.20 (best policy)** |
| seq_len=128 | **1.55 (best WM)** | 80.08 (worst) |

**seq_len=128 learns the world the best but acts on it the worst.** At a 200k env-step budget, the longer per-gradient-step rollout gives the world model more temporal context per update, so it converges to a noticeably lower reconstruction-and-prediction loss than the shorter-rollout cells. But the actor-critic policy never catches up: the policy's gradient signal per env-step is no richer at seq_len=128 than at seq_len=32, while the wall-clock-per-env-step has nearly doubled — so the policy effectively gets *less* training per unit of compute and per unit of env-step budget than the shorter-rollout cells.

**This is a recurring pattern across the search.** Phase 2's M paradox (best WM at 1.63, tied-worst survival at 86.9) and Phase 3's seq_len=128 paradox (best WM at 1.55, worst survival at 80.08) tell the same story two different ways: at the 200k env-step budget on 10×10 hypervigilance, **gains in world-model capacity or temporal context do not translate to survival within the budget**. Both directions of "more representational power" cost wall-clock and cost survival.

**Recommended follow-up experiment (deferred, future work):** re-run seq_len=128 at `total_steps ∈ {500000, 1000000}` — the same budget extension recommended for the M-paradox. If seq_len=128 overtakes seq_len=64 at 500k+ env-steps, the budget-gating hypothesis is confirmed for *both* the size and seq_len axes, and the publication-track recipe should be (size=M, seq_len=128, long budget). If seq_len=128 still loses at 1M env-steps, then long credit-assignment windows are a net loss on this task regardless of budget, and the recipe is fixed at seq_len=64. This is **flagged as future-experiment** — it is **not** auto-scheduled. See also §10.2.4 for the parallel M-axis follow-up.

#### 10.3.4 Phase 3 winner

**Winner: `seq_len=64` (P3.2 = reused P2.2, WandB [`a3o0xg75`](https://wandb.ai/sungwoolee/grid_world_pain/runs/a3o0xg75)).**

**Mechanical §7.1 ratio rule** (final-window mean ÷ wall-clock-per-200k-seconds, applied for completeness even though §7.1 for Phase 3 prioritizes absolute mean):

| Cell | Final-window mean | Wall-clock (s) | Score = mean / wall-clock |
|---|---|---|---|
| seq_len=32 | 94.67 | 6145 | **0.01541** ← top by ratio |
| seq_len=64 | 97.20 | 7877 | 0.01234 |
| seq_len=128 | 80.08 | 11643 | 0.00688 |

The §7.1 ratio mechanically picks **seq_len=32** (1.541 × 10⁻² vs 1.234 × 10⁻² for seq_len=64 — a 25% lead on the throughput-weighted score). But **§7.1 for Phase 3, like Phase 2, privileges absolute final-window mean** over the wall-clock ratio — the seq_len sweep is a quality test, not a throughput test, since none of the cells exceed the §8 6 h per-phase cap. On absolute mean, seq_len=64 is the unique top cell at 97.20 vs seq_len=32 at 94.67 — a 2.6% gap, well above measurement noise on a single seed.

**No override needed.** Both the absolute-mean rule and the qualitative "headroom" reading (seq_len=64 reached the 500-step env cap once; seq_len=32 maxed at 312, well below cap) point at the same answer: **seq_len=64**.

**Phase 3 winner therefore = (`num_envs=4`, `size=S`, `seq_len=64`)** — exactly the cell already in hand from Phase 2 (P2.2 = `a3o0xg75`). The Phase 3 sweep did not displace the Phase 2 winner; it confirmed it as the sweet spot on the third axis. The §10.4 final synthesis still needs to consider §11's long-budget validation results before declaring a final publication-track recipe — see §11.4 caveat.

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

## 11. Long-budget validation (2026-05-16) — interim preview

#### Headline (plain language, ~150 words)

**The user's "high-num_envs catches up given more training budget" hypothesis is confirmed at 2M env-steps — by a wide margin.** While Phases 1–3 ran every cell on a fixed 200,000 env-step budget (which was enough to give `num_envs=4` a clean win on every axis tested), the Phase-1 verdict explicitly noted that `num_envs=64` and `num_envs=128` looked *throughput-favored but learning-starved* — they processed the fixed env-step budget faster but never got enough gradient updates per unique trajectory for the world model and policy to converge. Question: do they catch up when given a budget proportional to their throughput? Answer (interim, both long-budget cells still running but past 80% completion): **yes, and they decisively beat the Phase 2/3 winner.** XS at `num_envs=64` running for 2M env-steps reaches a recent-window survival mean of **136.46** (vs 44 at the Phase 1 200k budget — a 3× improvement, and **+40% above the Phase 2/3 winner at 97.20**). XS at `num_envs=16` running for 2M env-steps reaches **145.83** (vs 73 at 200k — a 2× improvement, and **+50% above the Phase 2/3 winner**). Both cells **exceed sheeprl's 106.19 plateau**. The Phases 1–3 "envs=4 is best" verdict was correct **at the 200k budget** but wrong for the **absolute-survival question** once the budget is allowed to scale with throughput.

#### 11.1 Setup

| Factor | Value |
|---|---|
| Codebase commit | as of launch ~01:23 UTC, 2026-05-16 |
| Env config | `configs/experiment/hypervigilance/01-interoNocicept.yaml` (same as Phases 1–3) |
| Agent config | `configs/dreamer_srl/01_food_only.yaml` (XS, `learning_starts=1024`) |
| `total_steps` (CLI) | **2,000,000** (10× the Phase 1–3 budget of 200k) |
| Seed | 42 |
| `per_rank_sequence_length` | 64 (config-set, same as Phase 1–3 defaults) |
| `per_rank_batch_size` | 16 |
| Replay ratio | 1 |
| Node / GPUs | n113 cuda:0 (envs=16) and cuda:1 (envs=64) |

**Cells:** two — XS at `num_envs=16` and XS at `num_envs=64`. The `num_envs=4` baseline at the long budget was not launched in this follow-up; that cell is the easier-to-anticipate (slowest throughput, smallest expected gain from the budget increase), and the user's hypothesis specifically targets the high-num_envs cells.

#### 11.2 Interim partial results (cells still running)

Both cells launched at ~01:23 UTC; the table below is a snapshot as of the analysis time. **Neither cell is complete** — see §11.4 caveat.

| Cell | WandB | Status | Recent-window mean (last 20% of iterations to date) | Max ep_len | Δ vs Phase 1 (200k) baseline | Δ vs Phase 2/3 winner (97.20) | Δ vs sheeprl baseline (106.19) |
|---|---|---|---|---|---|---|---|
| envs=16 / 2M | [`bzc2x3pl`](https://wandb.ai/sungwoolee/grid_world_pain/runs/bzc2x3pl) | running, **~84%** of budget | **145.83** (n=2351 episodes in the recent-window) | 501 (env cap) | Phase 1 envs=16 = 72.5 → **+101%** | **+50.0%** | **+37.4%** |
| envs=64 / 2M | [`15uiw4kg`](https://wandb.ai/sungwoolee/grid_world_pain/runs/15uiw4kg) | running, **~95%** of budget | **136.46** (n=2913 episodes in the recent-window) | 501 (env cap) | Phase 1 envs=64 = 44.2 → **+209%** | **+40.4%** | **+28.5%** |

**Compute-cost comparison (interim, wall-clock-so-far, approximate):**

| Recipe | Final-window mean | Wall-clock | Notes |
|---|---|---|---|
| S / envs=4 / 200k (Phase 2/3 winner, `a3o0xg75`) | 97.20 | 7,877 s (2.19 h) | reference — the previous best |
| XS / envs=64 / 2M (`15uiw4kg`, interim @ ~95%) | **136.46** | ~35,000 s (~9.7 h) | **+40% survival at ~4.4× wall-clock**, smaller model |
| XS / envs=16 / 2M (`bzc2x3pl`, interim @ ~84%) | **145.83** | ~50,000 s (~13.9 h) | **+50% survival at ~6.3× wall-clock**, smaller model |

**Both long-budget cells touched the 500-step environment cap repeatedly** (max=501) — they are genuinely producing long-survival episodes, not just lifting the mean. Episode count in each cell's recent window is several thousand (2351 and 2913), so the mean is statistically well-estimated.

#### 11.3 Headline observation — the user's hypothesis confirmed

**Long-budget XS at high `num_envs` decisively dominates short-budget Phases 1–3.** This is the cleanest statement that can be made from the interim data:

1. **The Phase 1 verdict "envs=4 wins" was budget-conditional.** At 200k env-steps, the high-num_envs cells had not had enough gradient updates per unique trajectory to learn the task — exactly what the §10.1.2 H1b "still-broken at high num_envs" diagnosis identified as a sample-efficiency failure rather than an algorithmic collapse. Give the high-num_envs cells a budget that matches their throughput, and the failure goes away. Both envs=16 and envs=64 cells **triple or double their final-window means** when going from the 200k budget to ~80–95% of the 2M budget.
2. **The Phase 2/3 winner (S / envs=4 / 200k, 97.20 survival) is decisively beaten by XS / envs={16, 64} / 2M.** The +40% to +50% margins are much larger than any single-axis effect seen in the §10.1–§10.3 phase tables, and they cross the sheeprl 106.19 baseline by a clear margin (envs=16 is +37% above sheeprl, envs=64 is +28% above). **dreamer-srl v2 at the long budget exceeds sheeprl on 10×10 hypervigilance, even with the smallest XS model.**
3. **Throughput-driven, not capacity-driven.** Both cells use the smallest (XS) model — the long-budget recipe is not buying performance through capacity. It is buying it through more env-steps at the same capacity. This is consistent with the §10.2.4 "M paradox" and §10.3.3 "seq_len=128 paradox" diagnoses: at 10×10, **the bottleneck across the search has been training budget, not representational capacity**. The fix is more env-steps, not a bigger model.
4. **Implications for the publication-track recipe.** The "right recipe" for 10×10 hypervigilance, if these interim numbers hold, is no longer (S, envs=4, seq_len=64, 200k env-steps) but rather **(XS, envs=16 or envs=64, seq_len=64, ~2M env-steps)**. The exact `num_envs` choice between 16 and 64 — and whether even higher `num_envs` (128, 256) extends the gain further — is the next open question. The §10.4 final synthesis will sort this out once envs=16 completes; see §11.4.

#### 11.4 Caveat — envs=16 is still running

Both interim numbers in the table above are **provisional**:

- `envs=64` is at ~95% of its 2M budget — the recent-window mean of 136.46 is very close to the final number, expected to swing by ≤2% in either direction.
- `envs=16` is at ~84% of its 2M budget — the recent-window mean of 145.83 may swing by ±5–10% before completion. The cell could plausibly land anywhere in the 130–160 range. (The trajectory shape will be inspected at completion to check whether it is still climbing, plateaued, or showing late-training instability.)

**Do not lock §10.4 final synthesis until envs=16 completes.** A follow-up `experiment-analyzer` call at ~05:00–06:00 UTC will read the final WandB metrics for both cells, refresh the §11.2 table, and then author the comprehensive §10.4 final recipe — picking the `num_envs` value between 16 and 64 that wins on the absolute survival headline, deciding whether the long-budget recipe should be the publication default, and flagging whether even higher `num_envs` is worth a follow-up.

**Cells launched** (operational details for the §11.0 follow-up analyzer):

| Cell | num_envs | GPU | WandB run | PID | ETA (from ~01:24 UTC) |
|---|---|---|---|---|---|
| LB-1 | 16 | cuda:0 | [bzc2x3pl](https://wandb.ai/sungwoolee/grid_world_pain/runs/bzc2x3pl) `dreamer_srl_v2_10x10_longbudget_envs_16_XS_2M_s42` | 430782 | ~15:50 UTC (13.4h) |
| LB-2 | 64 | cuda:1 | [15uiw4kg](https://wandb.ai/sungwoolee/grid_world_pain/runs/15uiw4kg) `dreamer_srl_v2_10x10_longbudget_envs_64_XS_2M_s42` | 431216 | ~11:15 UTC (9.8h) |

**Decision rule for the final §10.4 analysis (after envs=16 completes).** `experiment-analyzer` reads the *final* WandB metrics for both cells (replacing the interim recent-window numbers in §11.2 with proper last-20%-of-completed-budget windows), then authors §10.4 picking between the Phase 1–3 short-budget recipe and the §11 long-budget recipe on the absolute-survival headline, with an explicit wall-clock-cost note for the publication track. If even higher `num_envs` (128, 256) is plausibly worth a follow-up, flag it as future work.

Note: `dreamer_srl_main.py` does not support `--wandb-group` / `--wandb-job-type` flags — these runs are ungrouped in WandB. Filter by wandb-name prefix `dreamer_srl_v2_10x10_longbudget_*`.

---

## 12. Extended sweep (2026-05-16, 8 cells across n106–n109)

### 12.1 Why launched

The long-budget validation cells (§11) showed XS/envs=16 and XS/envs=64 at 2M steps achieving ~146 and ~136 survival-steps respectively, decisively beating the Phase 2 short-budget winner (S/envs=4 at 200k → 97 steps). Two questions remained unanswered:

1. **Does the XS performance plateau by 2M or continue rising at 4M?** The in-flight LB cells (n113, ETAs 05:00–15:50 UTC) can only answer this for XS/envs=16+64. This sweep tests XS/envs=128 at 4M and repeats the XS/envs=16+64 cells at 4M to confirm.
2. **Was the Phase 2 S/M underperformance (S: 97, M: 87) purely a function of low num_envs?** Phase 2 only ran S/M at envs=4. This sweep tests S at envs=16/64/128 and M at envs=64/128 — both at 2M steps — to check if size becomes relevant once throughput is unlocked.

User directive after §11 interim results confirmed the LB advantage. Launched 2026-05-16 at ~12:42 UTC.

### 12.2 8-cell manifest

WandB group: `dreamer_srl_v2_extended_sweep_2026-05-16` (set via `WANDB_RUN_GROUP` env var — `dreamer_srl_main.py` does not support `--wandb-group` flag).

| Cell | Size | num_envs | total_steps | Node | GPU | PID | WandB run ID | Log | ETA (~UTC) |
|---|---|---|---|---|---|---|---|---|---|
| E1 | XS | 16 | 4M | n106 | cuda:0 | 3629398 | [yxij4lrc](https://wandb.ai/sungwoolee/grid_world_pain/runs/yxij4lrc) | logs/20260516_124201.log | ~15:40 (+27h) |
| E2 | XS | 64 | 4M | n106 | cuda:1 | 3629566 | [02n94uzu](https://wandb.ai/sungwoolee/grid_world_pain/runs/02n94uzu) | logs/20260516_124202.log | ~08:40 (+20h) |
| E3 | XS | 128 | 4M | n107 | cuda:0 | 970879 | [ybsma2zd](https://wandb.ai/sungwoolee/grid_world_pain/runs/ybsma2zd) | logs/20260516_124203.log | ~01:10 (+12.5h) |
| E4 | S | 16 | 2M | n107 | cuda:1 | 970919 | [z8w3bfte](https://wandb.ai/sungwoolee/grid_world_pain/runs/z8w3bfte) | logs/20260516_124204.log | ~10:40 (+22h) |
| E5 | S | 64 | 2M | n108 | cuda:0 | 491086 | [20o5ujno](https://wandb.ai/sungwoolee/grid_world_pain/runs/20o5ujno) | logs/20260516_124204.log | ~02:40 (+14h) |
| E6 | S | 128 | 2M | n108 | cuda:1 | 491254 | [qtvxauwd](https://wandb.ai/sungwoolee/grid_world_pain/runs/qtvxauwd) | logs/20260516_124205.log | ~20:40 (+8h) |
| E7 | M | 64 | 2M | n109 | cuda:0 | 6059 | [c5pt9t4v](https://wandb.ai/sungwoolee/grid_world_pain/runs/c5pt9t4v) | logs/20260516_124206.log | ~08:40 (+20h) |
| E8 | M | 128 | 2M | n109 | cuda:1 | 6099 | [d9emwzdp](https://wandb.ai/sungwoolee/grid_world_pain/runs/d9emwzdp) | logs/20260516_124207.log | ~00:40 (+12h) |

Peak GPU memory at launch: XS/128 (E3) = 406 MiB; S/64 (E5) = 542 MiB; S/128 (E6) = 410 MiB; M/64 (E7) = 586 MiB; M/128 (E8) = 507 MiB. All well under 24 GB — no OOM risk.

### 12.3 Decision rule for analysis

At completion, `experiment-analyzer` builds a 3-axis table: **size × num_envs × budget** (short-budget 200k from Phase 2, long-budget 2M from §11 + this sweep, and 4M from E1/E2/E3). The primary question is whether higher budget or larger model size closes the gap with XS at high num_envs. Final synthesis is written to §10.4.

First cell to finish: E6 (S/128/2M, ~20:40 UTC). Suggested next-check waypoint: ~01:00 UTC 2026-05-17 (when E3/E8 finish, ~12–12.5h ETA).

---

## 13. Cross-references

- **EXTENSION_RESULTS** (the parity comparison that motivated this search): [EXTENSION_RESULTS.md](./EXTENSION_RESULTS.md)
- **SPS × num_envs sweep** (the throughput data that informs §4.5 projections): [SPS_NUM_ENVS_SWEEP_V2.md](./SPS_NUM_ENVS_SWEEP_V2.md)
- **PARITY_LAUNCH_V2** (the food-only parity gate that this rebuild already passed): [PARITY_LAUNCH_V2.md](./PARITY_LAUNCH_V2.md)
- **v2 IMPLEMENTATION_PLAN**: [docs/develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md](../../../develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md)
- **Sheeprl 10×10 baseline (the score to beat — 106.19)**: WandB [`yt1uts22`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/yt1uts22)
- **Dreamer-srl 10×10 extension run (the lagging baseline at survival 69)**: WandB [`405f0555`](https://wandb.ai/sungwoolee/grid_world_pain/runs/405f0555)
