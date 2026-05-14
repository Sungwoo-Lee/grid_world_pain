---
title: "Dreamer-SRL v3 — parity launch (3 seeds × corrected XS × food-only NoPred)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
wandb_tag: dreamer_srl_parity
phase: pre-launch-design
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
| 1 | planned | `dreamer_srl_parity_s0` | 0 | cuda:? (runner picks) | — | — | — |
| 2 | planned | `dreamer_srl_parity_s1` | 1 | cuda:? (runner picks) | — | — | — |
| 3 | planned | `dreamer_srl_parity_s2` | 2 | cuda:? (runner picks) | — | — | — |

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

_To be filled by `experiment-analyzer` after all 3 seeds complete. Format per §5.5._

## 11. Conclusions

_To be filled by `experiment-analyzer` after applying the §5.3 verdict rule to §10 results. Format: a 4–6-sentence plain-language verdict that translates the predicate to English, references the threshold met, names the surviving / failing seeds, and routes the next experiment per §6._
