---
title: "Dreamer-SRL v3 — parity launch (3 seeds × corrected XS × food-only NoPred)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
launched_at: 2026-05-14T17:03:10
wandb_tag: dreamer_srl_parity
phase: results-in
cross_links:
  - docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md
  - docs/experiments/active/dreamer_srl_v3/SPS_SIZE_NUM_ENVS_SWEEP.md
  - docs/pi/calls/2026-05-14_d013_parity_launch_disposition.md
  - docs/develop/active/dreamer_srl_v3/CP10B_SPEC.md
  - docs/experiments/active/sheeprl_bridge/JAX_SHEEPRL_MATCHED_SPS_DESIGN.md
---

# Dreamer-SRL v3 — parity launch (3 seeds × corrected XS × food-only NoPred)

## 1. Context (plain language, ~200 words)

**The question.** Does our JAX rebuild of DreamerV3 — a model-based reinforcement-learning agent that learns by predicting the future inside an internal world model and rolling out imagined trajectories there — match the reference PyTorch implementation (`sheeprl`) on the simplest grid-world task we have, measured by **survival steps** (the project-standard "how long does the agent stay alive before starving", capped at 500 steps per episode)?

**Why this gate exists.** Over the last two days we closed an arc of 13 internal-consistency checkpoints (CP1 through CP10b) that each verified one piece of the rebuild was algorithmically faithful to upstream sheeprl. Internal consistency is necessary but not sufficient — a port can be checkpoint-clean and still fail to actually *learn the task*. This launch is the empirical gate: three independent random-initializations of the rebuild, trained for the same number of environment steps as sheeprl's baseline (200,000 steps), tested against the survival number sheeprl achieved on the same task and same recipe. The sheeprl baseline is the run [`kfsvh1qk`](https://wandb.ai/sungwoolee/grid_world_pain/runs/kfsvh1qk) (2026-05-12, food-only no-predator 5×5 grid, 200,000 environment steps, `dreamer_v3_XS` preset), which reached `Game/ep_len_avg = 500` — i.e. the agent learned to survive until the 500-step environment cap. Earlier sheeprl runs on the same task ([`jzgkcep4`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/jzgkcep4) and [`i4ulpn95`](https://wandb.ai/sungwoolee/grid_world_pain/runs/i4ulpn95)) reached the same cap. Our pass criterion is mean survival across 3 JAX seeds within ±10% of that 500-step cap — i.e. ≥ 450 steps.

**Speed envelope.** The wall-clock per seed is projected at **3.47 hours** (the steady-state SPS of 16.0 measured at CP10b applied to 200,000 steps), so 3 seeds in parallel on 3 free GPUs of node 114 finish in ~3.5 hours wall-clock; sequential would be ~10.5 hours. We launch all 3 in parallel.

**What "XS" means here.** XS is the smallest of sheeprl's five DreamerV3 size presets — 256-wide dense layers, 256-wide recurrent state, 1 MLP layer per block. The full table of presets and what they scale lives in the [PI call](../../../pi/calls/2026-05-14_d013_parity_launch_disposition.md) §"Plain-language sheeprl size presets". The parity target is XS because that is what sheeprl ran on the baseline; running anything larger would change the parity-claim shape.

---

## 2. Hypotheses (pre-registered)

We pre-register three hypotheses with explicit pass thresholds so the post-launch analyzer has a deterministic verdict rule. The **headline statistic is `Game/ep_len_avg` averaged over the last 20% of training (steps 160,000–200,000) per seed, then mean-of-seeds across 3 seeds**. The "mean across the last 20% of the run" smooths over the per-episode noise (the env-cap is 500 steps, so a single episode that ends at 487 vs. 500 should not move the verdict).

Let `M_JAX` = mean across the last-20%-window of `Game/ep_len_avg` across all 3 JAX seeds. Let `M_sheeprl ≈ 500` be the saturated sheeprl baseline.

| Predicate | Plain-language meaning | Pass rule | Failure mode |
|---|---|---|---|
| **H₁a — Parity (primary, two-sided)** | The JAX rebuild's mean survival sits within ±10% of sheeprl's 500-step cap, i.e. within [450, 550]. Because the env caps at 500, the operative bound is `M_JAX ≥ 450 AND per-seed M_seed ≥ 400`. The per-seed floor catches the case where one seed at 220 and two seeds at 495 average to 403 — a one-seed collapse. | `M_JAX ≥ 450` AND `min(per_seed_M) ≥ 400` | PASS-WITH-NOTE if 425 ≤ M_JAX < 450 OR per-seed floor at 350; FAIL if M_JAX < 425 OR any per-seed M < 350 |
| **H₁b — Outperform (one-sided, soft)** | The JAX rebuild's mean survival sits at the env-cap (≥ 495), indistinguishable from sheeprl up to a 5-step rounding margin. This is what "I am sheeprl" looks like. | `M_JAX ≥ 495` AND `min(per_seed_M) ≥ 480` | Soft outcome — strengthens the parity claim from "matches" to "matches at saturation". Not separately publishable but worth recording. |
| **H₀ — Refutation (collapse or divergence)** | The JAX rebuild fails to learn the task at all — either training is unstable (NaN, value-loss explosion, world-model loss diverges past CP10b's 1.40 by > 50%) or final survival is at or near the random-policy floor (`M_JAX ≤ 110`, the env's max episode length of `MaxSteps=500` capped by starvation typically resolves around 100 for uniform-random policy per the CP9/CP10 observations). | `M_JAX ≤ 110` (random-policy floor) OR training-stability FAIL per §6 | The JAX rebuild does not match sheeprl. Next experiments: rerun with `learning_starts=0` and longer training, or escalate to senior-developer for a bug hunt. |

**Floor criterion (sanity gate, applies to every seed):**
Any individual seed with `M_seed < 110` is flagged as a per-seed failure. The random-policy ep_len floor in food-only NoPred is approximately ep_len ≈ 100 (the env's max_steps cap is 500 but starvation typically resolves in 100 steps for an actor that doesn't learn to eat the food). Anything in the 100–200 band is sub-learning. The `M_seed ≥ 400` per-seed pass bar means each seed must reach at least 80% of the env cap individually before the mean is meaningful — this is the **anti-mean-washing rule** the project uses to prevent one good seed plus two collapsed seeds from being read as parity.

**Variance budget.** Sheeprl's three known runs on this exact task (`jzgkcep4`, `kfsvh1qk`, and the 50k-step `i4ulpn95`) all reached the env-cap (the 50k one landed at 399.4, but that was a 4× shorter run — at 200k all three saturate). So sheeprl's seed-variance at 200k is effectively zero on this task. The JAX seed variance is unknown; the per-seed floor and the 3-seed mean rule together give us a deterministic call even at moderate variance.

---

## 3. Experimental design

### 3.1 Independent variable — the only thing that varies

| Variable | Values | Rationale |
|---|---|---|
| `seed` (RNG seed for the JAX PRNGKey + numpy random + env sampling) | {0, 1, 2} | 3 seeds is the project minimum for a meaningful mean+spread. The CLI driver's `--seed` argument seeds `np.random.seed(args.seed)` AND `jax.random.PRNGKey(args.seed)` (driver lines 234-235), so a single int sets everything. Three seeds is enough to detect a one-seed collapse (per the §2 anti-mean-washing rule); five would be needed if sheeprl's variance were noticeably nonzero, which it isn't on this task. |

### 3.2 Fixed (controlled) variables — held constant across all 3 seeds

| Variable | Value | Source |
|---|---|---|
| Agent config (network sizes, optimizer, KL balance, RSSM dims) | `configs/dreamer_srl/01_food_only.yaml` (the corrected real-XS, commit `4fe3d8b`) | sheeprl XS preset — 256/1/256/256/24 — verified at CP10b |
| Env config | `configs/experiment/dreamer_curriculum/01_food_only.yaml` | food-only NoPred 5×5 grid, no predator, no rabbit; `max_steps: 500`, `bush.count: 3`; same env CP9, CP10b, and the SPS sweep used |
| `total_steps` | 200,000 | sheeprl baseline step budget (matched to `kfsvh1qk`, `jzgkcep4`) — passed via CLI `--total-steps 200000` to override the agent YAML's `total_steps: 5000` smoke default |
| `num_envs` | 1 | XS / num_envs=1 verdict from [SPS sweep §6.4](SPS_SIZE_NUM_ENVS_SWEEP.md#64-verdict-for-the-parity-launch). Multi-env gave only +13% throughput; the parity claim is strongest 1:1 with sheeprl's XS recipe |
| `learning_starts` | 1024 (from YAML; not CLI-overridden) | sheeprl XS default; §S3 random-action prefill active |
| `replay_ratio` | 1 (from YAML) | sheeprl XS default; 1 gradient step per env step |
| `per_rank_sequence_length` | 64 (from YAML) | sheeprl exp default |
| `per_rank_batch_size` | 16 (from YAML) | sheeprl exp default |
| `horizon` | 15 (from YAML) | sheeprl XS default — imagination rollout length |
| Hardware | node 114, RTX 6000 Ada (49 GB VRAM) | Same hardware class as sheeprl `kfsvh1qk` baseline. GPU 0 / 1 / 2 (or whichever 3 are free at launch time — confirmed by training-runner pre-flight) |
| Env preallocation | `XLA_PYTHON_CLIENT_PREALLOCATE=false` | required so 3 parallel runs on the same node share VRAM dynamically; same setting as the SPS sweep |

### 3.3 Dependent variables (what we measure)

| Variable | Source | Notes |
|---|---|---|
| **`Game/ep_len_avg` per-episode** (primary) | `wandb.log({"Game/ep_len_avg": ep_len, ...})` at `dreamer_srl_main.py:451` | Logged at every episode-done boundary. The post-launch analyzer extracts the trajectory, computes the per-seed mean over the last-20% window (steps 160,000–200,000), then means across 3 seeds. |
| **`Rewards/rew_avg`** (secondary) | same site | Reward is a homeostatic-drive-delta in this env, so it correlates with but does not equal ep_len. Useful as a sanity check (positive reward should track survival). |
| **`Loss/world_model_loss`** (training-health, secondary) | wandb scalar | Should follow the CP10b shape (decreasing then plateau near ~1.4). A diverging or NaN trajectory triggers the §6 stability-fail handling. |
| **`Loss/value_loss`, `Loss/policy_loss`** (training-health, secondary) | wandb scalar | Same — used to discriminate "training itself broke" from "training is fine but the policy can't learn the task". |
| **`Diagnostic/moments_invscale`** (training-health, secondary) | wandb scalar | Per the CP-line discipline — must stay ≥ 1.0 (safe-floor) and ideally finish below ~20 (a value here >> 20 indicates the actor moments are not adapting). |
| **`Time/sps_env`** (operational) | wandb scalar | Steady-state should track the CP10b 16.0 SPS. A value < 12 SPS suggests a parallel-job VRAM-contention slowdown — re-investigate. |
| **Wall-clock per seed** | training-runner records launch + completion times in the manifest | Pass rule: ≤ 25 hours (the 2× sheeprl-baseline budget gate). Projected 3.47 h based on CP10b. |

### 3.4 Confounds & limitations

| Confound | Severity | Mitigation |
|---|---|---|
| **GPU contention on node 114** — 3 parallel runs on the same node could compete for VRAM/PCIe/CPU. | Low. CP10b at XS / num_envs=1 used 1 MiB of VRAM post-completion (i.e. the running model fits well under 5 GB). 3 × 5 GB = 15 GB on 4× 49 GB = 196 GB total node VRAM is comfortable. CPU contention from env-step Python loop is the only realistic risk. | XLA_PYTHON_CLIENT_PREALLOCATE=false (each run grows its pool dynamically); training-runner pre-flight confirms 3 GPUs are free; if SPS drops below 12 on any seed, kill and serialize. |
| **Single env config (no replicate env seed)** — the env config's `random_start_pos` is true but governed by the same JAX RNG seed, so the per-seed differences come from BOTH the network init AND the env stochasticity. | Low. This is by design — sheeprl's seed does the same thing. Both stochasticity sources are bundled into the "seed" variable. | Documented as a single-seed-axis design choice; if a future replication needs them split, that's a follow-up. |
| **Last-20%-window means** — if a seed only converges at step 195,000, the last-20% window catches only the converged tail. If it converges at step 180,000 it catches both pre- and post-convergence. | Low for sheeprl-shape trajectories (saturation by step 25k per `jzgkcep4` and `kfsvh1qk`). | The CP10b precedent suggests JAX dreamer-srl should saturate even faster; we explicitly check that `Game/ep_len_avg` at step 100,000 is already ≥ 400 as a within-run health gate (§6). If not, the last-20% mean is conservative anyway. |
| **No noise modality enabled** — `perceptual_noise.enabled: false` in the env config. This means the parity launch tests the "noise-off" branch, which is exactly the sheeprl baseline branch. The neuromodulation experiments will enable noise; that's downstream. | None (this is the design). | n/a |
| **Per-seed run failures (NaN, OOM, crash)** | Medium — these are pre-decided as run-class failures and trigger a re-launch of just that seed, not a verdict change (§6). | Documented in §6 failure-mode catalog. |

### 3.5 Statistical-power note

Three seeds is the minimum that allows a meaningful seed-spread estimate (a 95% CI from 3 points has very wide tails, but the absolute-threshold rule `M_JAX ≥ 450 AND per-seed M ≥ 400` does not depend on a CI — it depends on the floor + the mean independently, which 3 seeds resolves deterministically). Sheeprl's three known 200k runs on this task all reached the cap (variance ≈ 0), so the parity test is structurally well-posed: we are testing whether the JAX side matches a known floor, not whether two noisy distributions differ.

If the result lands in the PASS-WITH-NOTE band (425–450) or any per-seed lands in the 350–400 caveat band, the post-launch decision is "add 2 more seeds (3, 4) and re-evaluate" — surfaced explicitly in the failure-mode catalog (§6).

---

## 4. Launch Manifest (system-of-record)

System-of-record for every parity-launch run. **The designer** writes the planned columns. **The training-runner** fills the actual columns (Node, GPU, Launched at, WandB run ID, Log path) at launch time, in place. **The experiment-analyzer** reads this table to find run folders and writes the §5 results post-completion.

All 3 runs share:
- **wandb-project**: `grid_world_pain_dreamer_srl_parity` (clean separation from `_smoke` and `_sweep`)
- **wandb-group**: `dreamer_srl_parity_2026-05-14` — injected via `WANDB_RUN_GROUP` env var (the driver does not currently expose `--wandb-group` as a CLI flag; the runner sets it via env var before invoking python)
- **wandb-job-type**: `parity` — injected via `WANDB_JOB_TYPE` env var (same reason)
- **Node**: 114
- **Env config**: `configs/experiment/dreamer_curriculum/01_food_only.yaml`
- **Agent config**: `configs/dreamer_srl/01_food_only.yaml`
- **`total_steps`**: 200,000 (CLI `--total-steps 200000`)
- **`num_envs`**: 1
- **Step budget per seed**: 200,000 env steps; projected 3.47 h/seed wall-clock

| Run | Status | Tag (= wandb-name) | Seed | GPU | Launched at | WandB run ID | Log path |
|---|---|---|---|---|---|---|---|
| 1 | completed | `dreamer_srl_parity_s0` | 0 | cuda:1 | 2026-05-14T17:03:10 | h666pcrv | logs/20260514_170310.log |
| 2 | completed | `dreamer_srl_parity_s1` | 1 | cuda:2 | 2026-05-14T17:03:13 | ny3npz68 | logs/20260514_170313.log |
| 3 | completed | `dreamer_srl_parity_s2` | 2 | cuda:3 | 2026-05-14T17:03:18 | hzsa984v | logs/20260514_170318.log |

**Tag-naming rule** — every Tag value is unique and identical to its wandb-name (so `dreamer_srl_main.py` does not synthesize a name and the analyzer can grep by tag). Format: `dreamer_srl_parity_s<seed>`. The wandb-group above ties the 3 rows together for WandB-side filtering.

### 4.1 Configs to produce

**Approach chosen — single config + CLI `--seed` override (NOT three-seed-variant configs).** The driver at `src/algorithms/dreamer_srl/dreamer_srl_main.py:174` accepts `--seed` as a first-class CLI argument that seeds both numpy (`np.random.seed`) and JAX (`jax.random.PRNGKey`) globally. Producing 3 seed-variant YAML files that differ in only one integer would duplicate the corrected-XS contents 3 times and make any future XS hyperparameter change require 3 edits. The CLI-arg approach is simpler, more maintainable, and matches the existing pattern used by both the sweep doc (`configs/dreamer_srl/01_food_only.yaml` + `--seed 0`) and the matched-config SPS measurement (`configs/models/dreamer_v3_sheeprl_matched.yaml` + `--seed 0`).

| Run | Env config | Agent config | Seed (CLI override) | Status |
|---|---|---|---|---|
| 1, 2, 3 (all) | `configs/experiment/dreamer_curriculum/01_food_only.yaml` (existing, read-only) | `configs/dreamer_srl/01_food_only.yaml` (existing, corrected to real-XS at commit `4fe3d8b`) | `--seed 0` / `1` / `2` | both files exist; no new configs needed |

**No new YAMLs are produced** by this design. The only new artifact is this design doc.

### 4.2 Per-seed command template

Authoritative reference; the runner adapts as needed (e.g. log redirect, GPU pinning). The runner is responsible for setting `WANDB_RUN_GROUP` and `WANDB_JOB_TYPE` env vars before each invocation so the group/job-type get attached to the WandB run.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=<GPU> \
WANDB_RUN_GROUP=dreamer_srl_parity_2026-05-14 \
WANDB_JOB_TYPE=parity \
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
    src/algorithms/dreamer_srl/dreamer_srl_main.py \
    --env-config configs/experiment/dreamer_curriculum/01_food_only.yaml \
    --agent-config configs/dreamer_srl/01_food_only.yaml \
    --total-steps 200000 \
    --num-envs 1 \
    --seed <SEED> \
    --wandb-project grid_world_pain_dreamer_srl_parity \
    --wandb-name dreamer_srl_parity_s<SEED>
```

Substitute `<GPU>` ∈ {0, 1, 2, 3} (whichever 3 are free at launch time) and `<SEED>` ∈ {0, 1, 2} per row.

### 4.3 Failure-handling policy

| Failure mode | Action |
|---|---|
| **CUDA OOM at JIT compile** (extremely unlikely — CP10b used 1 MiB post-completion) | Kill that seed, restart on the same GPU with no other co-tenant. If it OOMs alone, escalate — the corrected XS is not fitting and the §6 stop-rule fires. |
| **NaN in any of the 7 `Loss/*` keys during training** | Kill that seed, mark as run-class failure, log NaN-onset step. Re-launch the same seed once (could be a JAX-PRNGKey-init artifact); if it NaNs again, treat as a per-seed structural failure (still gives 2 valid seeds, log the third as crashed in the manifest). |
| **Wall-clock > 6 hours per seed** (vs. 3.47 h projected — anything beyond ~1.7× projection signals contention) | Investigate co-tenant. Kill and re-launch on a different GPU if contention is confirmed; otherwise let it run to completion. |
| **`Time/sps_env` drops below 12 SPS in the steady-state window** | Likely VRAM/CPU contention. Pause the lowest-priority co-tenant; if no co-tenant, escalate. |
| **One seed crashes for non-NaN, non-OOM reason** (e.g. JAX device error) | Mark that seed's row as `failed`, retain the run state, retry once. If it crashes again, surface to the user with the stack trace before deciding to drop the seed. |
| **All 3 seeds complete cleanly** | Hand off to `experiment-analyzer` per §7 to fill in §5 results. |

The runner does NOT preempt the other 2 seeds when one fails — they are independent runs. The verdict in §2 explicitly handles the "2 of 3 valid seeds" case via the per-seed floor rule.

---

## 5. Analysis plan (pre-specified) — to be applied by `experiment-analyzer` after all 3 seeds complete

The analyzer fills these results into a new §6 "Results" section and writes the §7 verdict.

### 5.1 Primary statistic

For each seed:
- Pull `Game/ep_len_avg` time-series from WandB.
- Identify the step-160,000–200,000 window (or equivalent — the last 20% of logged points).
- Compute `M_seed = mean(Game/ep_len_avg | step ∈ [160000, 200000])`.

Then:
- `M_JAX = mean(M_seed_0, M_seed_1, M_seed_2)`
- `min_per_seed = min(M_seed_0, M_seed_1, M_seed_2)`
- Per-seed 95% CI via bootstrap on the per-episode values within the window (informational; the pass rule uses point estimates, not CIs).

### 5.2 Secondary

- World-model loss trajectory per seed — confirm decreasing-then-plateau shape, no late divergence.
- Value-loss and policy-loss — confirm no late blowup.
- `moments_invscale` — confirm stays ≥ 1.0 throughout.
- `Time/sps_env` per seed — should track the CP10b 16.0 SPS within ±20%.
- Wall-clock per seed vs the 3.47-h projection — within ±50% is fine; > 6 h triggers the §4.3 investigation.

### 5.3 Verdict rule (deterministic, applies the §2 thresholds)

1. **PASS-OUTPERFORM** (H₁b): `M_JAX ≥ 495` AND `min_per_seed ≥ 480`.
2. **PASS** (H₁a, clean): `M_JAX ≥ 450` AND `min_per_seed ≥ 400`.
3. **PASS-WITH-NOTE**: `425 ≤ M_JAX < 450` AND `min_per_seed ≥ 350`. Recommend 2 more seeds before declaring the gate closed.
4. **FAIL** (H₀ confirmed): `M_JAX < 425` OR `min_per_seed < 350`. Triggers the §6 next-step decision.

### 5.4 Time-locked check — within-run health gate at step 100,000

Sheeprl's reference runs saturated `Game/ep_len_avg = 500` by step 25,000 — i.e. half-way to 50k. As an early-warning, check at step 100,000 (50% of the parity budget) whether the JAX rebuild has at least crossed `Game/ep_len_avg = 400`. If it hasn't, the run is on a slow-or-stuck trajectory and the analyzer flags this in the report — does not change the verdict (still applies the §5.3 rule on the last-20% window), but informs the next experiment.

### 5.5 Output

Analyzer writes a §6 Results section with a per-seed table:

| Seed | M_seed (last 20%) | World-model loss (final) | SPS (steady-state) | Wall-clock (h) | Notes |
|---|---|---|---|---|---|
| 0 | TBD | TBD | TBD | TBD | TBD |
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| **mean** | **TBD** | — | — | — | — |

…followed by the §7 verdict (PASS / PASS-OUTPERFORM / PASS-WITH-NOTE / FAIL) plus a paragraph of plain-language interpretation per the CLAUDE.md framing rule.

---

## 6. Failure-mode catalog (pre-decided)

| Outcome | Verdict it triggers | Next experiment |
|---|---|---|
| All 3 seeds reach ≥ 495 mean ep_len in last 20% | **PASS-OUTPERFORM** (H₁b confirmed) | Parity gate closes definitively. The JAX rebuild is shippable as a sheeprl-equivalent backbone. Proceed to neuromodulation hook integration (the downstream phase of the v3 plan). |
| All 3 seeds reach ≥ 400 individually AND mean ≥ 450 | **PASS** (H₁a confirmed) | Parity gate closes. Same next step as PASS-OUTPERFORM. Note any per-seed gaps in the verdict text. |
| Mean ∈ [425, 450) with all seeds ≥ 350 | **PASS-WITH-NOTE** | Add seeds 3, 4 (re-launch with `--seed 3` and `--seed 4`). Re-evaluate the 5-seed mean against the same thresholds. If still in PASS-WITH-NOTE band, escalate to PI for budget-vs-precision trade-off. |
| Mean < 425 OR any seed < 350 | **FAIL** (H₀ partially confirmed) | Diagnose: (a) is world-model loss converging? (b) is value loss stable? (c) is `moments_invscale` healthy? If yes to all three, the policy is learning *something* but slowly — extend `total_steps` to 400,000 and re-run 3 seeds. If world-model or value loss is broken, escalate to senior-developer for a code-bug hunt — CP10b passed, so a regression between commit `4fe3d8b` and launch time would be the suspect. |
| Mean ≤ 110 (random-policy floor) | **FAIL — complete collapse** | The rebuild is not learning the task at all. Cross-check: was the corrected XS config actually loaded? Was the env actually food-only? Was `learning_starts=1024` active (i.e. did §S3 prefill happen)? Compare per-seed `Loss/world_model_loss` trajectory to CP10b's. If the trajectory diverges from CP10b's, a regression has landed since CP10b commit `c388064`. |
| 1 of 3 seeds NaNs out, other 2 reach pass thresholds | **PASS-WITH-NOTE — investigate NaN seed** | Apply the §5.3 PASS rule to the 2 surviving seeds (with a documented caveat about the smaller sample). Re-run the NaN seed once on a freshly-restarted process; if it NaNs again, log it as a structural seed-failure for the post-launch report. |
| All 3 seeds wall-clock > 25 hours | **BUDGET-FAIL (operational, not scientific)** | Should not happen — CP10b projected 3.47 h. If observed, the §4.3 investigation table fires. Cross-check that XLA_PYTHON_CLIENT_PREALLOCATE was set, that no other GPU job was running on the node, and that the SPS table from CP10b matches the launched runs. |

---

## 7. Sanity check — env config and ep_len interpretation

The env config `configs/experiment/dreamer_curriculum/01_food_only.yaml` is the same one CP9, CP9b, CP10b, and the SPS sweep used. The relevant settings for ep_len interpretation:

| Setting | Value | Effect on ep_len |
|---|---|---|
| `environment.max_steps` | 500 | Episodes hit a hard cap at 500 steps. `Game/ep_len_avg = 500` means the agent never died — saturated at the cap. |
| `environment.height × width` | 5 × 5 | Small grid; the agent can in principle navigate any cell-pair in ≤ 5 steps. |
| `body.metabolic_cost` | 1.0 per step | Each step costs 1.0 nutrition. |
| `body.start_nutrition` | 100 | Starts at full nutrition; pure-no-eat survival is ~100 steps (the random-policy floor — agent walks around, never eats, starves at step ~100). |
| `body.food_nutrition_gain` | 18 per food | Each food eaten gains 18 nutrition. |
| `body.max_consumption` (food) | 12 | Each food can be eaten 12 times before depleting. |
| `body.death_penalty` | 100 | Starvation triggers a −100 reward and episode end. |
| `body.overeating_death` | false | No overeating death; the agent can max-out nutrition safely. |
| `food.count` | 1 | One food entity, spawning anywhere in 1,1..5,5. |
| `food.regeneration_delay` | 0 | Food regenerates immediately. |
| `bush.count` | 3 | 3 hiding bushes (relevant for predator tasks; in food-only NoPred they are passive terrain). |
| `predators[0].count` | 0 | No mobile predator. |
| `hiding_predator.count` | 0 | No static lethal entity. |
| `rabbits.count` | 0 | No neutral animal. |
| `perceptual_noise.enabled` | false | No sensor noise — clean observation channel. |

**Ep_len-to-interpretation mapping** (used by the §2 thresholds and the §6 failure catalog):

| `Game/ep_len_avg` value | Interpretation |
|---|---|
| ~100 | Random-policy floor — agent walks around, never eats, dies of starvation when nutrition reaches 0 at metabolic_cost × ~100 steps. |
| 100–200 | Partial learning — agent occasionally finds food but inconsistently. |
| 200–400 | Substantial learning — agent eats reliably but suffers occasional starvation. |
| 400–495 | Near-saturation — agent rarely starves; pass band. |
| 495–500 | Saturation — agent essentially never starves; hits the env cap. |

**Theoretical max**: 500 steps (the env's `max_steps` cap). The agent cannot exceed this even with optimal play.

**Sheeprl baseline at this env** (commit-checked):
- `kfsvh1qk` (200k steps, 2026-05-12, food-only NoPred replica): `ep_len_avg = 500` (saturated cap) — the primary parity anchor.
- `jzgkcep4` (200k steps, 2026-05-11, original sheeprl smoke on this env): `ep_len_avg = 500` (saturated cap) — confirms `kfsvh1qk` is not an outlier.
- `i4ulpn95` (50k steps, 2026-05-12, with `apply_noise: True`): `ep_len_avg = 399.4` — used at the PI call as a wall-clock anchor but NOT the parity target (different `apply_noise` setting AND a 4× shorter run; both reach saturation by 100k and onwards in the longer-budget runs).

The parity target is **the 200k-step, noise-off, food-only sheeprl behavior — `Game/ep_len_avg = 500` (saturated env cap).** Our pass criterion (mean ≥ 450, per-seed ≥ 400) is "within 10% of the cap with no per-seed collapse" — i.e. a 90%-ceiling parity claim.

---

## 8. Pre-flight

This launch reuses two configs both already pre-flighted:
- **`configs/dreamer_srl/01_food_only.yaml`** — corrected XS at commit `4fe3d8b`, verified at CP10b (WandB `s31wc1a1`) where it ran cleanly for 20,000 steps without OOM, NaN, or stability issue.
- **`configs/experiment/dreamer_curriculum/01_food_only.yaml`** — env config unchanged since CP9, CP10b, and the SPS sweep used it. `env-config-auditor` has previously cleared it.

**No new YAML schema, no new keys, no new code paths.** Only the CLI invocation differs from CP10b (200,000 steps vs 20,000; `--seed 0/1/2` vs `--seed 0` only).

**env-config-auditor explicit request**: not strictly required since both files are unchanged from prior auditor-cleared launches. The user can choose to re-run the auditor for belt-and-suspenders; otherwise the pre-flight is implicit-by-precedent.

---

## 9. Hand-off

After the user authorizes:

1. **`training-runner`** launches all 3 seeds in parallel on node 114 (GPUs to be confirmed by runner pre-flight — recommend GPUs 1, 2, 3 to leave GPU 0 for ad-hoc work; the SPS sweep ran on GPUs 1/2/3 with this exact split). Per-seed launch commands in §4.2.
2. **Runner fills** the §4 manifest with Launched-at / WandB run ID / GPU / Log-path columns at launch.
3. **Runner monitors** for §4.3 failure modes during the first 30 minutes (JIT-compile success, no NaN in first log lines, SPS in the 12–20 range).
4. **Runner reports** completion via the `/diary` `training-done` subcommand (3 separate entries, one per seed).
5. **After all 3 seeds complete** (~3.5 h wall-clock if all parallel, ~10.5 h sequential): the user invokes `experiment-analyzer` to apply §5 and write §6 + §7.

**The `experiment-analyzer` writes the verdict; the runner just launches and reports launch state.** This is the standard split — runners do not draw scientific conclusions, analyzers do.

---

## 10. Results

### 10.1 Headline (plain language, ~200 words)

**The parity gate failed at the random-policy floor.** All 3 JAX dreamer-srl seeds trained cleanly for the full 200,000 environment-step budget — no NaN, no OOM, world-model loss converged to the same ~1.37 value CP10b reached at 20k steps — but the agent never learned to eat food. Across the full last-20% window (env-steps 160,000–200,000) every seed's mean `Game/ep_len_avg` (the survival statistic — how many steps the agent stays alive before starving, capped at 500) sits at **~101–105 steps**, which is the random-policy baseline (the agent walks around, never eats, dies of starvation at metabolic_cost × 100 nutrition). The 3-seed mean is **103.8 steps**. The pass bar was 450; we are 346 steps short. The sheeprl reference implementation on the identical config (same env YAML, same XS preset, same 200k budget) reached the env-cap `Game/ep_len_avg = 500` by env-step 25,000 — i.e. learned the task in ~12% of the budget. The JAX rebuild has not crossed `ep_len = 400` even once across 597,000 cumulative env-steps of training. The trajectory is essentially flat at the floor with brief excursions to ep_len ≈ 200–322 that decay back. The world model is healthy; the actor–critic loop is the failure point.

### 10.2 Per-seed table (§5.5 format)

| Seed | WandB ID | M_seed (steps 160k–200k) | World-model loss (final) | Value loss (early → late) | Policy loss (early → late) | `moments_invscale=1.0` (floor) frac in [100k, 200k] | SPS (steady) | Wall-clock | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | [h666pcrv](https://wandb.ai/sungwoolee/grid_world_pain_dreamer_srl_parity/runs/h666pcrv) | **101.3** | 1.362 | 1.854 → 1.267 | −0.286 → −0.019 | 35.3% | 17.4 | 3.30 h | FAIL — random-policy floor |
| 1 | [ny3npz68](https://wandb.ai/sungwoolee/grid_world_pain_dreamer_srl_parity/runs/ny3npz68) | **104.7** | 1.414 | 1.920 → 1.254 | −0.278 → −0.001 | 21.6% | 17.6 | 3.27 h | FAIL — random-policy floor |
| 2 | [hzsa984v](https://wandb.ai/sungwoolee/grid_world_pain_dreamer_srl_parity/runs/hzsa984v) | **105.4** | 1.367 | 1.910 → 1.185 | −0.224 → +0.015 | 58.8% | 17.5 | 3.29 h | FAIL — random-policy floor |
| **mean** | — | **103.8** | 1.381 | — | — | 38.6% | 17.5 | 3.29 h | — |

`M_JAX = 103.8`, `min_per_seed = 101.3`, per-seed range = 4.1 steps (essentially zero spread — all three seeds collapse to the same floor).

### 10.3 Trajectory evolution (env-step buckets)

Three seeds, `Game/ep_len_avg` mean per env-step bucket — never substantively departs from ~101:

| Env-step bucket | Seed 0 | Seed 1 | Seed 2 | Sheeprl `kfsvh1qk` (env-step proxy via `trainer/global_step`) |
|---|---|---|---|---|
| [0, 5k) | 102.7 | 103.1 | 105.3 | 101.8 (start) |
| [5k, 25k) | 101.2 | 101.3 | 101.4 | climbing: 100.7 → 151.1 → **500** by step 25k |
| [25k, 50k) | 103.3 | 101.1 | 101.2 | 500 (saturated) |
| [50k, 100k) | 102.9 | 102.9 | 103.4 | 500 |
| [100k, 150k) | 104.0 | 105.2 | 101.3 | 500 |
| [150k, 200k) | 103.1 | 103.9 | 104.7 | 500 |

**First env-step at which `ep_len_avg ≥ 400` (the §5.4 within-run health gate is ≥ 400 at env-step 100k):**

| Run | First step ≥ 200 | First step ≥ 300 | First step ≥ 400 | Step-100k health gate (target ≥ 400) |
|---|---|---|---|---|
| Seed 0 (h666pcrv) | 46,547 | 145,847 | **never** | **101.0 — FAIL** |
| Seed 1 (ny3npz68) | 64,178 | 106,596 | **never** | **101.0 — FAIL** |
| Seed 2 (hzsa984v) | 61,544 | **never** | **never** | **101.0 — FAIL** |
| Sheeprl kfsvh1qk | ≤ 20,000 | ≤ 20,000 | **≤ 25,000** | (n/a — already saturated) |

### 10.4 Sheeprl-baseline comparison (the empirical centerpiece)

Sheeprl's two known 200k-step runs on this identical recipe ([kfsvh1qk](https://wandb.ai/sungwoolee/grid_world_pain/runs/kfsvh1qk) and [jzgkcep4](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/jzgkcep4)):

| Statistic | Sheeprl `kfsvh1qk` | Sheeprl `jzgkcep4` | Dreamer-srl 3-seed mean |
|---|---|---|---|
| Last-20% mean `Game/ep_len_avg` | **500.0** (saturated, n=9, range [500, 500]) | **500.0** (saturated, n=9, range [500, 500]) | **103.8** |
| First env-step at ep_len = 500 | env-step 25,000 (5% of budget) | env-step ≤ 20,000 (≤ 10% of budget) | never |
| Ratio of budget to reach saturation | ~12% | ~10% | n/a |

Sheeprl's pre-25k trajectory shape (`ep_len_avg` at the trailing env-step landmarks): 5k = 101.8, 10k = 100.7, 15k = 106.6, 20k = 151.1, **25k = 500.0**. Sheeprl spends ~20k env-steps at the same random-policy floor the JAX rebuild is stuck on, then breaks away over a ~5k-step window. The JAX rebuild does not show that break-away phase even 8× later. **This is the key signal: it is not a "needs more steps" pattern — it is a "the actor never finds the food-eating policy" pattern.**

### 10.5 Training-health diagnostics

The runs are **operationally healthy** — every secondary metric the design pre-registered for stability passed:

- **World-model loss** converged on the CP10b trajectory: ~1.47 → 1.42 → 1.39 → 1.38 → 1.37–1.38 plateau across env-step buckets. Final values 1.362 / 1.414 / 1.367 vs CP10b's 1.404. The world model is learning the dynamics.
- **Value loss** decreased (early ~1.9 → late ~1.2) on all 3 seeds — consistent with a critic that is learning to predict the (mostly-negative-301) returns of a starving agent.
- **Policy loss** drifted toward zero on all 3 seeds (−0.28 → −0.02 to +0.01) — the actor's gradient signal collapsed to near-zero.
- **`moments_invscale = 1.0` floor fraction** in the [100k, 200k] window: 35.3% / 21.6% / **58.8%** of logged points. Seed 2 spent the majority of late training with no return-spread to normalize against — the actor's imagined-rollout advantage is structurally flat for that seed.
- **No NaN, no OOM, SPS 17.4–17.6** (above the CP10b 16.0 projection; the parity-budget run was actually faster per step than the smoke).

**Wall-clock**: all 3 seeds finished in 3.27–3.30 h, well inside the projected 3.47 h and the §4.3 6-h kill-rule.

### 10.6 CP10b comparison — same pattern, 10× the budget

[CP10b](../../../develop/active/dreamer_srl_v3/CP10B_SPEC.md) (WandB `s31wc1a1`, 20k env-steps, seed 0, same configs as this launch) also ended at `Game/ep_len_avg = 101` and `Rewards/rew_avg = -301`. CP10b's verdict was operational-PASS (no NaN, no divergence, world-model loss 1.40 ≈ CP10b reference). **CP10b never claimed the agent learned the task** — it claimed the training loop is numerically stable. The parity launch reproduces CP10b's flat-floor trajectory at 10× the env-step budget. That is, the issue is not a training-budget question; the same regime persists.

---

## 11. Conclusions

### 11.1 Verdict (per §5.3)

**❌ FAIL (H₀ confirmed — complete collapse to the random-policy floor).**

Applying the §5.3 deterministic rule to the §10.2 numbers:
- `M_JAX = 103.8 < 425` → criterion-cell **FAIL**.
- `min_per_seed = 101.3 < 350` → criterion-cell **FAIL**.
- Step-100k within-run health gate (§5.4): all 3 seeds at 101 — **FAIL on every seed**.
- §6 failure-mode catalog row "Mean ≤ 110 (random-policy floor)" fires: **FAIL — complete collapse**.

This is not a borderline call. The PASS bar was 450; we are 346 steps short, with zero per-seed spread (range 4.1 steps), and a sheeprl reference that learned the same task in 12% of the budget. The H₀ predicate is unambiguously confirmed.

### 11.2 Failure-mode diagnosis (the world model is fine; the actor/critic loop is broken)

The 3-row diagnostic picture, in plain English:
1. **World model** learns the dynamics — its loss tracks CP10b verbatim. The imagination simulator is not broken.
2. **Critic** learns a value function, but the value function it's learning is "the agent is starving everywhere, expected return is approximately −300" — consistent with what the actor actually does. So the critic is healthy *given* the policy it's evaluating; it does not pull the policy toward food-eating because there is no positive-return signal in the rollouts.
3. **Actor** does not generate meaningfully different imagined rollouts. The `moments_invscale = 1.0` floor fraction (35–59% of late training) is the canonical "flat returns across imagined trajectories" signature. With no advantage variance, REINFORCE gradient ≈ 0 (`Loss/policy_loss` drifts to ~0 in all 3 seeds), and the actor never learns to prefer food-eating actions over wandering.

The mechanism is a **classic imagined-rollout return collapse**: the world model can imagine a 15-step horizon, but every imagined trajectory under the current policy ends with a starving agent, so every imagined return is approximately equal, so the advantage signal vanishes, so the policy doesn't update toward the (rarely-sampled) food-eating action. Sheeprl with the same config breaks out of this in ~5k env-steps — something about sheeprl's exploration, target-network update, or λ-return computation pushes the policy past the random-policy floor where it can begin to see the +18-reward food signal often enough to bootstrap. **The JAX rebuild does not have that breakout dynamic.**

### 11.3 Ranked hypotheses for the breakout-gap (most likely first)

1. **Actor exploration / entropy regularization mis-port (HIGH confidence).** Sheeprl's actor includes an entropy bonus on the categorical distribution; in the JAX rebuild this may be present but mis-weighted, or the temperature on the categorical may be wrong, so the policy concentrates too quickly on the random-walk modes before discovering food. CP7's actor-REINFORCE block is the suspect. **Audit target**: `src/algorithms/dreamer_srl/` actor loss — is `policy_entropy` weighted at the same coefficient (typically 3e-4 for DreamerV3 XS) and is it summed correctly into `Loss/policy_loss`?

2. **`learning_starts = 1024` may be too short OR the random-action prefill may be wrong (MEDIUM confidence).** The first 1024 env-steps are supposed to be uniform-random actions to seed the replay buffer. If the prefill is using the policy network from the start (i.e. an untrained categorical that's biased toward one action), the buffer is filled with degenerate trajectories that the world model learns, the value function fits, and the actor inherits — and we're stuck. **Audit target**: §S3 random-action prefill in `dreamer_srl_main.py` around the `learning_starts` branch — confirm `act = env.action_space.sample()` (or equivalent uniform-categorical) is in force for `step < learning_starts`, and that the actor is not being called for action selection during that window.

3. **λ-return / discount / continue-flag scaling bug in imagination (MEDIUM-LOW confidence).** If the imagined-rollout returns are computed with the wrong discount, a wrong continue-flag (e.g. `continues` always 1 instead of `(1 - terminal)`), or a sign error on the reward inside imagination, the advantage signal would be structurally suppressed. CP6's critic loss + CP7's λ-return are the suspect lines. **Audit target**: the imagination-rollout return computation in the dreamer-srl module — sanity-check that `imagined_return[t] = imagined_reward[t] + gamma * (1 - terminal[t]) * lambda_return[t+1]`-equivalent, that gamma matches sheeprl's 0.997, and that the reward is signed correctly.

### 11.4 Named next experiments (project-routed)

- **Audit-first, re-launch-second** (RECOMMENDED). Hand off to `senior-developer` to write an audit plan covering the three hypotheses above against the corresponding lines in sheeprl's `algos/dreamer_v3/agent.py` and `dreamer_v3.py`. The audit should produce either a single named regression (likely actor-entropy weighting per H1) or a verified "no regression — likely a hyperparameter/exploration recipe difference" finding. The latter would mean we need to look at sheeprl's `train_step` for any non-XS-preset behavior we missed.
- **Do not extend `total_steps` to 400k.** The §6 catalog row for "Mean ≤ 110" pre-decided this. The trajectory is flat, not slow — extending the budget on a flat-trajectory regime burns 6 more hours of compute to reproduce the same number. Sheeprl gets to 500 in 25k steps; budget is not the binding constraint.
- **Reproducibility check**: re-run sheeprl's `kfsvh1qk` config on the same node 114 with seed = 0 (already done as `kfsvh1qk`), seed = 1 (already done as `jzgkcep4`). Sheeprl variance is already known to be ~0 here; we don't need to spend more compute on baseline replication.

### 11.5 Implications for the project

- **CP10b's "PASS" verdict is technically correct** (its acceptance criterion was world-model-loss-convergence + no-NaN, not task-learning) — but it should be flagged as **insufficient as a parity-blocker**. A useful CP-line going forward needs at least one CP that gates on `Game/ep_len_avg ≥ 200` at some env-step before declaring the rebuild ready for parity. Surface this to `senior-developer` as a recommendation; **do not flip CP10b's verdict** — this analyzer does not have that authority per project hard rules.
- **The parity-track is blocked** until the actor-loop audit completes. The downstream phase of the v3 plan (neuromodulation hook integration) cannot be honestly bootstrapped on a backbone that has not demonstrated task-learning. **Hand off to `pi`** for a portfolio-level call: is the right next move (a) the actor-loop audit + re-launch, or (b) freeze the dreamer-srl rebuild and continue the publication track on the sheeprl bridge (which already learns this task at the env-cap)?
- **No re-opening of an earlier CP-PASS verdict by this analyzer.** CP6 / CP7 are flagged as audit-targets for `senior-developer`; the analyzer does not have CP-flip authority.

### 11.6 Cross-references

- Design + criteria (this doc, §1–§9): hypothesis was pre-registered; the verdict follows the §5.3 rule deterministically.
- CP10b (the 20k-step smoke): [`docs/develop/active/dreamer_srl_v3/CP10B_SPEC.md`](../../../develop/active/dreamer_srl_v3/CP10B_SPEC.md). CP10b WandB run `s31wc1a1` also ended at ep_len = 101.
- Sheeprl baseline `kfsvh1qk`: 200k env-steps, `Game/ep_len_avg = 500` (saturated cap, first hit at env-step 25k).
- PI call that authorized this launch: [`docs/pi/calls/2026-05-14_d013_parity_launch_disposition.md`](../../../pi/calls/2026-05-14_d013_parity_launch_disposition.md).

### 11.7 Metrics Requested

None new. Every metric needed for this verdict is already logged. The diagnostic in §11.2 was constructed entirely from existing WandB scalars (`Game/ep_len_avg`, `Loss/world_model_loss`, `Loss/value_loss`, `Loss/policy_loss`, `Diagnostic/moments_invscale`, `Time/sps_env`).

### 11.8 Related Issues

- **`senior-developer` audit** on actor-loop / exploration / imagined-return computation per §11.3 hypotheses. Three ranked targets, all auditable from existing source.
- **`pi` consultation** for portfolio-level scope: continue debugging the JAX rebuild, or shift the publication track to the sheeprl bridge that already works on this task.
