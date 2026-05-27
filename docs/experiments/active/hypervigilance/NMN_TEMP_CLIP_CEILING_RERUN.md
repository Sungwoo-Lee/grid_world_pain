---
title: "NMN temp_clip ceiling re-run — is the FiLM-MC failure a hyperparameter artefact or structural?"
topic: hypervigilance
status: active
created: 2026-05-08
last_updated: 2026-05-09
phase: 1
wandb_tag: "rppo_nmn_tempceil10_<profile>_film_g1_s<seed>"
develop_link: docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md
supersedes: null
superseded_by: null
---

# NMN temp_clip ceiling re-run — is the FiLM-MC failure a hyperparameter artefact or structural?

> **Status**: COMPLETE — all 5 cells reached 10.0 M ep; Results / Analysis / Conclusions filled 2026-05-09.
> **Date**: 2026-05-08
> **Author**: experiment-designer
> **Related**:
> - [NMN_NOISE_HETEROGENEITY_SWEEP](NMN_NOISE_HETEROGENEITY_SWEEP.md) — the parent sweep this follows up. §10.4 surfaced profile-dependent temp_max saturation; §11.3 + §12.3 recommended the temp-ceiling test BEFORE committing to the §6.3 P3 architectural redesign.
> - [NMN_PERFORMANCE_DIAGNOSIS_v8 §4.3.3](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) — original "MC FiLM saturates temp at 3.0" finding; the heterogeneity sweep generalised it to "saturation kicks in at R ≈ 5".

> **One-line scope.** Five FiLM g1 cells (P3 × 3 seeds + P4 × 1 seed + P5 × 1 seed), all with `temp_clip = [0.5, 10.0]` (3.3× the prior 3.0 ceiling). Tests whether FiLM under-performance on the saturated profiles is **a hyperparameter artefact (ceiling was binding)** or **structural (temperature head does not engage productively past ~3.0 even when allowed)**. Unmod baselines reused from the parent sweep; no new env configs; one new agent config.

---

## 1. Research Question

When the temperature-output ceiling of the FiLM modulator is raised from 3.0 → 10.0 on the noise-heterogeneity profiles where it previously saturated (P3, P4, P5), does the FiLM g1 LN MC architecture's steady-state survival improve enough to either match or exceed the existing Unmod LN MC baseline?

> **H₀ (null):** Raising `temp_clip` ceiling from 3.0 to 10.0 produces no change in steady-state survival on saturated profiles (P3/P4/P5). Δ_SS_new − Δ_SS_old ≤ |1 step| at every profile. The temperature head does not productively use the freed range; the FiLM-vs-Unmod gap is structural, not a clip artefact.

> **H₁a (ceiling binding, FiLM rescued):** Raising the ceiling produces a survival gain on at least one saturated profile such that the *new* Δ_SS = SS_FiLM_new − SS_Unmod_old > +5 steps on that profile. The original sweep's null was a hyperparameter artefact, not a structural FiLM failure.

> **H₁b (ceiling binding, FiLM matched but not rescued):** Raising the ceiling reduces the FiLM-vs-Unmod deficit by at least 5 steps on at least one saturated profile (improvement Δ_SS_new − Δ_SS_old > +5) but does not flip the sign (FiLM still ≤ Unmod). The temperature head was binding *and the modulator does meaningfully use the freed range*, but freeing it is not sufficient for FiLM to overtake the unmodulated baseline. Architectural redesign is still needed.

> **H₁c (head non-engagement):** Raising the ceiling produces no measurable temp_max increase relative to the prior sweep (`temp_max_new ≈ temp_max_old ≈ 3.0` on P3/P4/P5, despite the new ceiling being 10.0). The temperature head's *target* output sits at ~3.0; the prior `temp_clip: [0.5, 3.0]` was incidentally binding but not actually limiting the architecture's expressed range. Confirms the prior sweep's structural verdict at lower cost than a re-architecture.

### 1.1 What confirms vs. refutes — pre-specified

| Outcome | Predicate | Verdict |
|---|---|---|
| **H₁a confirmed** | At ≥1 of {P3 (≥2/3 seeds), P4, P5}: SS_FiLM_new − SS_Unmod_old > +5 steps | FiLM is rescued by raising the ceiling. The §6.3 P3 redesign is **not** needed; instead, propagate the new `temp_clip` to all NMN configs and re-write the v8 diagnosis. **Hand back to experiment-designer to scope a 3-seed cross-profile replication confirming the rescue.** |
| **H₁b confirmed** | At ≥1 saturated profile: improvement Δ_SS_new − Δ_SS_old > +5 AND new SS_FiLM still ≤ SS_Unmod | The temperature head was binding *and uses the freed range*, but freeing it is insufficient. Hand to **senior-developer for §6.3 P3 (architectural redesign)** with the additional constraint "any redesign must not regress the partial improvement seen here". |
| **H₁c confirmed (head non-engagement)** | On all of {P3-meanseed, P4, P5}: temp_max_new < 3.5 (i.e., the new ceiling is barely used; the head's natural target is ~3.0) AND no SS gain (\|Δ_SS_new − Δ_SS_old\| ≤ 1 step) | Original `temp_clip: [0.5, 3.0]` was not binding. The FiLM-vs-Unmod failure is structural, downstream of the temperature head. **Hand to senior-developer for §6.3 P3** with the temperature head explicitly de-prioritised in the redesign (don't fix what isn't broken upstream). |
| **H₀ confirmed (no-op)** | All saturated cells produce \|Δ_SS_new − Δ_SS_old\| ≤ 1 step AND temp_max_new ∈ [3.0, 10.0] uniformly using the freed range without survival improvement | Temperature head DOES use the freed range, but the modulator downstream cannot productively cash out the dynamic range into survival. Same hand-off as H₁c (architectural redesign), with a stronger statement: the ceiling was binding *and* irrelevant. |
| **P3 3-seed disambiguation** | At P3 with `temp_clip: [0.5, 10.0]`, ≥ 2/3 seeds produce SS_FiLM_new > SS_Unmod_old (Δ_SS > 0 across seeds) | The prior single-seed P3 (Δ_SS = +0.08) is not a coincidence; FiLM matches Unmod on P3 specifically. Combined with H₁a/H₁b, this localises whatever survives the ceiling test to a profile-specific effect. |
| **Refutation by training instability** | Any cell NaNs, value/grad explodes, or Term_Starvation > 90% | Single cell can be marked inconclusive. If ≥ 2/5 cells fail this way, the new ceiling is destabilising; halt and report. NOTE: the architecture's default `temp_clip` is `(0.1, 10.0)` per `src/models/neuromodulator.py:71`, so 10.0 is within the original numerical envelope; we do not expect new instability. |

### 1.2 Why a single ceiling (`[0.5, 10.0]`), not a sweep over {5.0, 10.0}

The user's brief offered both `[0.5, 5.0]` (modest) and `[0.5, 10.0]` (aggressive) and asked the designer to decide. **Picked `[0.5, 10.0]` only.** Justification:

1. **Disambiguation.** The prior sweep's temp_max landed at 2.96 / 2.99 / 3.00 on P3f / P4f / P5f — saturation pinned exactly at the ceiling. If we raise to 5.0 and observe temp_max = 4.95, we cannot tell apart "ceiling still binding" from "the head's natural target is somewhere in [3.0, 5.0) but not specifically 3.0". Either reading would still need a 10.0 follow-up. 10.0 collapses two diagnostic rounds into one.
2. **Cost-symmetric.** A single 10.0 ceiling on 5 cells (3+1+1 seeds) is 5 cell-runs. A 5.0+10.0 sweep over the same profile coverage would be 10 cell-runs. The structural question (does the head engage in [3.0, ~10.0)?) is answered with one ceiling at the upper bound; if it does engage, the precise functional shape can be characterised in a follow-up.
3. **Numerical safety.** `temp_clip: [0.5, 10.0]` is the architecture's *original* numerical envelope per `src/models/neuromodulator.py:71`. We are not introducing a new instability surface; the prior `[0.5, 3.0]` was a tightening, not a default.

The acceptable risk: if the head WOULD engage productively in [3.0, 5.0] but DOES NOT continue to climb past ~5.0, the 10.0 run will see temp_max somewhere in [3.0, 10.0) — fine, that is informative. The risk we cannot recover from is one we already covered: if temp_max lands at 10.0 (saturates again), we know unambiguously the head wants more dynamic range than 10× provides, and the structural redesign is mandatory. With 5.0, that question stays open.

---

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|---|---|---|
| `agent.modulation.temp_clip` | `[0.5, 10.0]` | Single ceiling; see §1.2. |
| Noise profile | {P3, P4, P5} | Saturated profiles from parent sweep (§10.4). P1/P2 not re-run — their FiLM cells did not saturate (temp_max 2.30 / 1.50 < 3.0), so the new ceiling is non-binding there. See §2.4 for the assumption this rests on. |
| Seed (P3 only) | {0, 1, 2} | 3-seed replication at P3 — disambiguates the lone Δ_SS ≈ 0 cell from coincidence (parent §12.3 Item 1) AND tests the new ceiling on the same profile. Conflates two factors; see §2.5 for why this conflation is acceptable here. |

### 2.2 Controlled Variables

Pinned across all 5 cells. Identical to parent sweep §3.2 EXCEPT the agent's `temp_clip`:

```yaml
# Environment — REUSE parent sweep's env configs verbatim
configs/experiment/nmn_noise_heterogeneity/p3_moderate.yaml   # for P3 cells
configs/experiment/nmn_noise_heterogeneity/p4_high.yaml       # for P4 cell
configs/experiment/nmn_noise_heterogeneity/p5_extreme.yaml    # for P5 cell

# Agent — IDENTICAL to recurrent_ppo_nmn_het_film_g1.yaml except temp_clip
agent:
  algorithm: RecurrentPPO
  return_mode: MC                     # parent sweep choice; v8 §4.1.5
  use_layer_norm: true
  rnn_type: GRU
  activation: relu
  encoding_mode: hierarchical
  hidden_size: 128
  sequence_length: 128
  K_epochs: 4
  lr_actor: 0.0005, lr_critic: 0.0001
  entropy_coef: 0.01, eps_clip: 0.1
  num_envs: 128

agent.modulation:
  type: FiLM
  mod_hidden_size: 16
  grouping_size: 1
  percept_bias_init: 3.0
  memory_bias_init: 0.0
  temp_clip: [0.5, 10.0]              # <-- the only diff vs parent sweep agent config
  memory_clip: [-2.0, 2.0]
```

**Episode budget:** 10M episodes per cell, matching parent sweep. The parent sweep's P3f/P4f/P5f reached 7.4–8.8M; if compute pressure is similar, the same partial-budget caveat applies, but the comparison to parent-sweep cells is at matched effective budget.

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|---|---|---|---|
| **Cross-experiment seed drift.** Parent sweep ran P3f at seed 0; this experiment runs P3 at seeds {0, 1, 2}. The seed-0 cell here is **not** byte-identical to parent's P3f because the agent config differs (temp_clip). So the seed-0 P3 cell is not literally a re-run; it is a different config at the same seed. | P3 cells | Med | Pre-registered: comparison is "this experiment's three P3 cells (seeds 0, 1, 2)" vs "parent's P3f (seed 0)" as a 3-vs-1 comparison; the parent's P3f stays anchored as a single-seed observation under the OLD ceiling. |
| **Conflation of seed-disambiguation with ceiling-test on P3.** The 3-seed P3 cells run at the *new* ceiling, conflating "is +0.08 reproducible across seeds at the OLD ceiling?" with "does +0.08 grow at the NEW ceiling?". | P3 cells | Med-High | See §2.5 — explicit decision to fold these in rather than run separate 3-seed-old-ceiling replication; conflation cost is acknowledged and the decision rule (§1.1) accommodates it. |
| **Single seed at P4 and P5.** Parent sweep had this same limitation at saturated profiles. Cross-seed variance under the new ceiling is unmeasured. | P4, P5 | Med | Pre-registered: H₁a / H₁b confirmation requires Δ_SS gain > 5 steps — the same defensive threshold as the parent sweep. P4/P5 single-seed cells can refute the ceiling-rescue hypothesis (a clean null at single seed), but cannot confirm it alone. If P4 or P5 produces a > 5-step gain, follow-up requires 3 seeds. |
| **P1/P2 not re-run as un-saturated controls.** We assume P1/P2 FiLM cells (temp_max 2.30 / 1.50 in parent) would land at ≈ same temp_max under the raised ceiling (the new ceiling is not binding there). | None directly | Low | Justification: in the parent sweep, P1f / P2f temp_max was well below 3.0 (the *original* ceiling), so the new 10.0 ceiling is doubly non-binding. Survival should not change. The risk is a small one (P1/P2 might shift slightly because gradient flow through the temperature softplus is sensitive to ceiling proximity), but second-order; not worth 2 cells of compute. If the user wants a P1/P2 sanity check, that is a 2-cell add-on. |
| **P4f's parent-sweep budget shortfall (7.4M ep)**. Parent's P4f Δ_SS = −8.9 was preliminary at 7.4M episodes. If the new-ceiling P4 cell also stops short, comparison is matched-budget but both are sub-target. | P4 cell | Low-Med | Pre-registered: report SS_SS at the smaller of {parent P4f's 7.4M, this run's actual} for the matched-budget comparison; report SS_SS at this run's actual for the absolute reading. Both go in the §4 results table. |
| **Cross-experiment hardware / driver drift.** Parent ran on nodes 101-105 between 2026-05-07 and 2026-05-08. This re-run may be on different nodes / GPU types. | All | Low | Pre-registered: this experiment's tags are sufficient for analysis grouping; absolute SS_SS values may shift by a few steps due to nondeterminism, but ΔSS within-experiment is unaffected. |

### 2.4 Why P1/P2 are out of scope (un-saturated controls assumption)

The mechanism this experiment tests is "the 3.0 ceiling was binding on P3+ where temp_max → 3.00, releasing the ceiling rescues survival on those profiles." On P1/P2 the parent sweep showed temp_max = 2.30 / 1.50, both well below the 3.0 ceiling. The ceiling was therefore **not binding** at P1/P2. Raising it from 3.0 to 10.0 should produce no behavioural change on those profiles. We assume this and do not re-run P1/P2 in this experiment. **If P1/P2 cells under the new ceiling did somehow shift**, that would be a second-order effect (e.g., gradient flow through softplus near a clipped boundary affects optimisation when *near* the clip but *not* clipped) and would be a secondary finding to surface in a follow-up — but it does not gate the primary structural verdict.

### 2.5 Folded-in 3-seed P3 replication (the conflation decision)

The parent sweep's §12.3 listed two follow-ups:
- (1) 3-seed P3 replication to disambiguate Δ_SS ≈ 0.08 from coincidence
- (2) `temp_clip` ceiling test on saturated profiles

Two ways to combine them:
- **Option A (separate):** 3-seed P3 at the OLD ceiling (3.0), disambiguates +0.08, then a separate ceiling test on P3/P4/P5 at the NEW ceiling. **Cost:** 6 cells (3 + 3) total.
- **Option B (fold in):** 3-seed P3 at the NEW ceiling (10.0), simultaneously seed-disambiguates AND ceiling-tests P3. **Cost:** 5 cells (3 + 1 + 1).

**Decision: Option B.** Justification:
1. The single most-actionable follow-up the parent sweep produced is the temp-ceiling test (per §11.3 mechanism + §12.3 recommendation). Option A delays it by a factor-of-two compute round; Option B addresses it now.
2. The conflation cost — "if 3-seed P3 wins, was it the seeds or the ceiling?" — is bounded and tractable. The 4 references to disambiguate against:
   - Parent's P3f (1 seed, OLD ceiling, Δ_SS = +0.08): the original anchor.
   - This experiment's P3 (3 seeds, NEW ceiling): the test.
   - This experiment's P4 + P5 (1 seed each, NEW ceiling): saturated cousins of P3, isolate the "ceiling change" signal independent of P3.
   - Parent's P4f + P5f (1 seed each, OLD ceiling): cross-experiment matched-profile comparison at the saturated cousins.
   If all three 3-seed P3 cells beat parent's P3f AND P4/P5 also improve under the new ceiling, the parsimonious read is "ceiling did it" (consistent across profiles + seeds). If 3-seed P3 wins but P4/P5 don't, the read is "P3 is profile-specific, not ceiling-driven, and the +0.08 was real." Either way the experiment is informative and the conflation is recoverable.
3. Option A's "old ceiling 3-seed P3" cell would teach us *only* whether +0.08 is real, not whether it would become +5 with more headroom. The marginal value of running it before the ceiling test is low; running it *after* a positive ceiling test is also low (the ceiling test would already have answered the +5-question).

The acceptable risk is exactly the one named: if 3-seed P3 produces SS_FiLM > SS_Unmod by a small but reliable amount (Δ_SS ~ 1–2 steps across seeds), we cannot tell whether seeds or ceiling drove it. That risk is acceptable because: (a) the §1.1 H₁a/H₁b thresholds are set at 5 steps, well above this band; (b) within-experiment cross-profile evidence (does P4/P5 also improve?) breaks the tie; (c) if the result genuinely is in the 1–2 step inconclusive band, a 3-seed-old-ceiling P3 follow-up is exactly the experiment to run next, and it is small.

---

## 3. Launch Manifest

System-of-record for all 5 runs. **Designer columns locked.** Runner fills `Status / Launched at / WandB run ID / Log path` at launch time. Node + GPU pre-assigned by user.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | running | p3_film_g1_tempceil10_s0 | `rppo_nmn_tempceil10_p3_film_g1_s0` | nmn_temp_clip_ceiling | prod | 0 | 106 | cuda:0 | 2026-05-08T20:34:40 | f96lhxpe | logs/20260508_203440.log |
| 2 | running | p3_film_g1_tempceil10_s1 | `rppo_nmn_tempceil10_p3_film_g1_s1` | nmn_temp_clip_ceiling | prod | 1 | 106 | cuda:1 | 2026-05-08T20:36:34 | hfhop3qu | logs/20260508_203634_rppo_nmn_tempceil10_p3_film_g1_s1.log |
| 3 | running | p3_film_g1_tempceil10_s2 | `rppo_nmn_tempceil10_p3_film_g1_s2` | nmn_temp_clip_ceiling | prod | 2 | 107 | cuda:0 | 2026-05-08T20:39:10 | cwnvye4m | logs/20260508_203906.log |
| 4 | running | p4_film_g1_tempceil10_s0 | `rppo_nmn_tempceil10_p4_film_g1_s0` | nmn_temp_clip_ceiling | prod | 0 | 107 | cuda:1 | 2026-05-08T20:41:54 | dq8yhmlp | logs/20260508_204154.log |
| 5 | running | p5_film_g1_tempceil10_s0 | `rppo_nmn_tempceil10_p5_film_g1_s0` | nmn_temp_clip_ceiling | prod | 0 | 108 | cuda:0 | 2026-05-08T20:45:00 | hgvionh6 | logs/20260508_204500.log |

**Tag pattern:** `rppo_nmn_tempceil10_<profile>_film_g1_s<seed>`.
- `tempceil10` is the experiment-distinguishing infix (NEW ceiling = 10.0; the legacy 3.0 ceiling cells from the parent sweep used the bare `het` infix and are NOT re-run).
- `<profile>` ∈ {p3, p4, p5}.
- `s<seed>` ∈ {s0, s1, s2}.

All Tags are unique within the manifest AND unique versus the parent sweep's tags (no overlap with `rppo_nmn_het_*`). Tag = wandb-name per project memory; runner copies verbatim. wandb-group = `nmn_temp_clip_ceiling` — a NEW dir under WandB to keep the ceiling-test cells visually separate from the parent sweep when browsing.

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Cell | Config (env — REUSED from parent sweep) | Config (agent — NEW) |
|---|---|---|---|
| 1 | p3_film_g1_tempceil10_s0 | `configs/experiment/nmn_noise_heterogeneity/p3_moderate.yaml` | `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` |
| 2 | p3_film_g1_tempceil10_s1 | `configs/experiment/nmn_noise_heterogeneity/p3_moderate.yaml` | `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` |
| 3 | p3_film_g1_tempceil10_s2 | `configs/experiment/nmn_noise_heterogeneity/p3_moderate.yaml` | `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` |
| 4 | p4_film_g1_tempceil10_s0 | `configs/experiment/nmn_noise_heterogeneity/p4_high.yaml`     | `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` |
| 5 | p5_film_g1_tempceil10_s0 | `configs/experiment/nmn_noise_heterogeneity/p5_extreme.yaml`  | `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` |

**File changes (total):**
- **0** changes to `src/`, `scripts/`, or `train_command*.sh`.
- **0** new env configs (P3/P4/P5 env configs reused verbatim from the parent sweep).
- **1** new agent config: `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml`.

**Diff vs `recurrent_ppo_nmn_het_film_g1.yaml`** (the parent sweep's FiLM agent config) — lines that change:
```yaml
# OLD (recurrent_ppo_nmn_het_film_g1.yaml):
    temp_clip: [0.5, 3.0]

# NEW (recurrent_ppo_nmn_het_film_g1_tempceil10.yaml):
    temp_clip: [0.5, 10.0]
```
Plus an updated header comment block documenting the diff. Every other line is byte-identical.

**Schema additions: zero.** `temp_clip` is already loaded at `src/models/recurrent_ppo_network.py:264` via `tuple(modulation_config['temp_clip'])` — direct dict access, mandatory key (raises `KeyError` if absent), no fallback default. Numerical envelope `[0.5, 10.0]` is within the architecture's original default `(0.1, 10.0)` per `src/models/neuromodulator.py:71`. No schema additions, no config-loader edits, no `senior-developer` route required.

### 3.2 Reference launch command (template the runner adapts per row)

For Run 1 (p3_film_g1_tempceil10_s0):
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/nmn_noise_heterogeneity/p3_moderate.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml \
  --num-envs 128 \
  --episodes 10000000 \
  --checkpoint-frequency 100000 \
  --seed 0 \
  --device <as-assigned-by-user> \
  --log-interval 10 \
  --wandb-group nmn_temp_clip_ceiling \
  --wandb-job-type prod \
  --wandb-name "rppo_nmn_tempceil10_p3_film_g1_s0" \
  --tag "rppo_nmn_tempceil10_p3_film_g1_s0"
```

For Runs 2–5: substitute `--config`, `--agent_config` (constant — same agent file), `--seed`, `--wandb-name`, `--tag` per §3.1 + manifest.

---

## 4. Analysis Plan (Pre-Specified)

### 4.1 Primary statistic

For each cell, **steady-state survival steps (SS_SS)** = mean episodic survival over the **last 20% of episodes** (windows 8–10 of 10), matching parent sweep §5.1 verbatim for direct comparability.

For each profile P_k ∈ {P3, P4, P5}, two effects:
- **Within-experiment Δ_SS_new(P_k) = SS_FiLM_new(P_k) − SS_Unmod_old(P_k)** — the FiLM-vs-Unmod gap under the new ceiling, using the parent sweep's existing Unmod cells as baselines.
- **Cross-experiment improvement = Δ_SS_new(P_k) − Δ_SS_old(P_k)** — how much the gap closed under the new ceiling.

For P3, additionally compute:
- SS_FiLM_new mean ± std across 3 seeds.
- Per-seed Δ_SS_new (3 values).

### 4.2 Existing baselines reused (no re-run)

The parent sweep's Unmod cells under the same env configs are valid baselines (Unmod has no temperature head; the agent config's `temp_clip` change has no effect on Unmod). **DO NOT re-run Unmod cells.** Reference WandB run IDs from the parent sweep's manifest:

| Profile | Unmod baseline (parent sweep) | WandB ID | SS_SS_old |
|---|---|---:|---:|
| P3 | `rppo_nmn_het_p3_unmod_c5` | `6eqjr62y` | 221.69 |
| P4 | `rppo_nmn_het_p4_unmod_c7` | `cicjphwf` | 226.04 |
| P5 | `rppo_nmn_het_p5_unmod_c9` | `mb0huxq1` | 226.42 |

For P1/P2 (not in scope; un-saturated controls assumed): `jcxin0g8` and `q2o3vydr` are the parent's Unmod IDs if needed for any auxiliary check.

### 4.3 Effect-size thresholds (pre-registered)

| Predicate | Threshold | Decision |
|---|---|---|
| **H₁a (FiLM rescued)** | At ≥1 of {P3 (≥2/3 seeds), P4, P5}: SS_FiLM_new − SS_Unmod_old > +5 steps | confirm H₁a; ceiling was the bottleneck. |
| **H₁b (FiLM partially rescued)** | At ≥1 saturated profile: improvement Δ_SS_new − Δ_SS_old > +5 steps AND new SS_FiLM still ≤ SS_Unmod_old | confirm H₁b; partial mechanism. Architectural redesign still needed but with fixed ceiling. |
| **H₁c (head non-engagement)** | All 5 cells: temp_max_new < 3.5 (head doesn't use freed range) | confirm H₁c; structural failure downstream of temperature. |
| **H₀ (no-op or no-rescue)** | All cells: \|Δ_SS_new − Δ_SS_old\| ≤ 1 step | confirm H₀; the ceiling was not the bottleneck regardless of whether the head used the freed range. |
| **Inconclusive band** | 1 < max improvement ≤ 5 steps AND temp_max ∈ [3.0, 10.0) | inconclusive; recommend 3-seed replication on P4 + P5 at the new ceiling. |
| **P3 seed-disambiguation** | At least 2/3 P3 cells produce SS_FiLM > SS_Unmod_old (any positive Δ_SS) | parent's P3f +0.08 is real (not coincidence). |

### 4.4 Temporal evolution check (mandatory per project conventions)

Per project_plan, end-of-training snapshots alone are insufficient. For each cell, plot SS by 10 equal-episode windows and check:

1. **Convergence by window 8.** If FiLM cell still climbing > 5 steps from window 8 → 10, mark *inconclusive* — the new-ceiling effect could continue to grow beyond the SS window. The parent sweep's P3f / P4f / P5f had window 8 → 10 climbs of +5.4 / +2.2 / +4.1; the new-ceiling cells should be compared at matched windows.
2. **Mid-training divergence.** Plot SS_FiLM_new vs SS_FiLM_old at every window. The parent sweep showed FiLM trails Unmod from window 1 onward. If the new-ceiling FiLM cell trails at window 1 but catches up by window 5+, this is *prima facie* evidence the ceiling was binding *during late training* (the temperature head doesn't actually want to climb past 3.0 until well into training). Note as a finding even if SS_SS is unchanged.
3. **Temperature dynamics over training.** For each FiLM cell, plot temp_max and temp_mean by window. Parent sweep showed temp_max → 3.00 by SS on saturated profiles; we want to know *when* temp_max first hits 3.0 in the parent (early or late) and whether the new-ceiling cell continues climbing past 3.0 at the same training step.

### 4.5 Diagnostic fingerprints per cell (matched to parent sweep §5.4 + §10.4)

| Metric | Source | What it diagnoses |
|---|---|---|
| `gamma_multi_mean`, `gamma_multi_std` | NMN logger (existing) | Whether raised ceiling changes gate diversity. Parent sweep had P3f/P4f/P5f gamma_multi_std at 1.39 / 0.92 / 1.03; if the new ceiling lets gamma_multi_std climb monotonically with R, the modulator is more functional. |
| `temp_mean`, `temp_min`, `temp_max` | NMN logger (existing) | THE central diagnostic for this experiment. Parent: 2.20 / 2.37 / 2.35 (mean) and 2.96 / 2.99 / 3.00 (max). H₁c is refuted iff temp_max climbs > 3.5 on ≥1 cell. |
| `loss/value`, `loss/grad_norm`, `loss/entropy` | trainer (existing) | Standard loss diagnostics. Parent had P3f/P4f/P5f grad_norm ≈ 0.18–0.20 (no instability). New-ceiling cells should be similar; if grad_norm spikes above ~1.0, the freed range is destabilising training. |
| Term_Injury / Term_Starvation / Term_MaxSteps | trainer (existing) | Decomposition of survival outcome. Parent: Term_Starvation 0.60–0.65 on saturated FiLM cells. |
| FoodEaten, TotalDamage | trainer (existing) | Behavioural decomposition. |

All metrics already logged by the existing pipeline. **No new metrics requested.**

### 4.6 Cross-cell + cross-experiment comparisons (specific tests)

1. **Δ_SS_new(R) trend** — primary plot. x-axis = log R; y-axis = Δ_SS_new (FiLM_new − Unmod_old). 3 points (P3 mean ± std, P4, P5). Annotate the 5-step H₁a threshold AND the parent sweep's Δ_SS_old(R) on the same axes for direct visual comparison.
2. **temp_max(R) trend** — old ceiling vs new ceiling, both plotted. Reveals whether the head climbs into [3, 10) when allowed.
3. **Per-seed P3 Δ_SS scatter** — 3 points; tests whether the parent's +0.08 was within the seed-noise band.
4. **gamma_multi_std(R) vs old sweep** — does the freed temperature range allow more diverse gates?

### 4.7 Reporting format

Fill §5–§7 of this doc post-launch, following [docs/TEMPLATES/training_analysis.md](../../../TEMPLATES/training_analysis.md). Hand to `experiment-analyzer` for the actual fill-in.

---

## 5. Failure-Mode Catalog (pre-registered)

| Failure mode | Pre-registered interpretation |
|---|---|
| **All 5 cells: temp_max ≈ 3.0 (no climb past prior ceiling)** despite new ceiling = 10.0 | H₁c confirmed (head non-engagement). The original `temp_clip: [0.5, 3.0]` was incidentally binding but the head's natural target is ~3.0. The FiLM-vs-Unmod failure is structural and downstream of the temperature head. **Hand to senior-developer for §6.3 P3 (architectural redesign)** with a strong prior that the temperature head is NOT the bottleneck. |
| **All 5 cells: temp_max climbs into [3.0, 10.0) AND survival improvement > 5 steps on ≥1 cell** | H₁a (or H₁b) confirmed. The ceiling was the bottleneck. **Most actionable single finding the project has produced on FiLM.** Hand back to `experiment-designer` to scope a 3-seed cross-profile replication that pins the rescue. Re-write v8's MC FiLM verdict; propagate `temp_clip: [0.5, 10.0]` to all NMN configs. |
| **Cells: temp_max climbs into [3.0, 10.0) but no SS gain (\|Δ_SS_new − Δ_SS_old\| ≤ 1)** | H₀ confirmed in the rich form: head DOES use the freed range, modulator downstream cannot cash it out into survival. Hand to senior-developer for §6.3 P3 redesign with the constraint "any redesign must explain why dynamic-range expansion at the temperature head produces no survival benefit". |
| **3-seed P3 cells produce a wide spread (e.g., σ_SS > 5 across seeds)** | Single-seed status was masking high cross-seed variance on P3 specifically. Even if mean Δ_SS_new is in the H₁a band, variance erodes confidence. Recommend extending to 5 seeds before architectural commitment. |
| **NaN / gradient explosion in any cell** | Mark `inconclusive` for that cell. The architecture's default `temp_clip` is `(0.1, 10.0)`, so the new ceiling does not exceed the original numerical envelope; instability under 10.0 would suggest a value-loss explosion specific to the noisy regime, not a ceiling artefact. If ≥ 2/5 cells fail, route to senior-developer. |
| **Term_Starvation > 90% on any FiLM cell** | The freed temperature range may be enabling pathological exploration (over-modulation suppressing the foraging signal). Note as a finding even if survival shape supports H₁c. Cap and re-run with `temp_clip: [0.5, 6.0]` if this fires (cheaper than committing to the redesign on a single pathological cell). |
| **Both old-ceiling and new-ceiling P3 produce ≈ same SS_FiLM** | The ceiling was not the bottleneck on P3 specifically (consistent with parent's +0.08 being a profile-specific anomaly unrelated to the temperature head). If P4/P5 also show no improvement, H₀ confirmed; if P4/P5 do improve, the ceiling was binding on P4/P5 but the P3 result is decoupled from the ceiling — interesting, route back to designer for a denser sweep. |
| **All 5 cells fail to reach 10M episodes within wallclock** | Operational. Re-allocate or accept reduced steps; declare any cell <8M episodes inconclusive for the convergence question. Match comparison to parent's 7.4–8.8M effective budget where applicable. |

---

## 6. Metrics Requested

None for this experiment. All §4.5 metrics are logged by the existing pipeline. **No code changes needed.** Verified against parent sweep's §10.4 + §10.5 (which used the same metric set successfully).

---

## 7. Hand-offs

### 7.1 env-config-auditor (next agent)

Audit the §3.1 configs:
- 1 NEW agent config: `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml`. Verify it is byte-identical to `recurrent_ppo_nmn_het_film_g1.yaml` EXCEPT `temp_clip: [0.5, 10.0]` (and the comment header). Use `diff` directly.
- 0 NEW env configs (re-use of `p3_moderate.yaml` / `p4_high.yaml` / `p5_extreme.yaml` from parent sweep). No need to re-audit these — they passed audit for the parent sweep on 2026-05-07 and have not changed.
- 5 manifest tags are unique strings AND do not collide with any parent sweep tag (`rppo_nmn_het_*`).
- `agent.modulation.type: "FiLM"`, `grouping_size: 1`, `mod_hidden_size: 16`, `percept_bias_init: 3.0`, `temp_clip: [0.5, 10.0]`, `memory_clip: [-2.0, 2.0]` in the new agent config.
- `agent.return_mode: "MC"` and `agent.use_layer_norm: true` in the new agent config.
- Verify `temp_clip: [0.5, 10.0]` is within the architecture default `(0.1, 10.0)` per `src/models/neuromodulator.py:71` (numerical envelope check).

### 7.2 user (gating step)

After auditor passes, user authorises launch and provides node + GPU assignments per row of §3.

### 7.3 training-runner

Launches each manifest row using §3.2's command template (per-row substitutions per §3.1). One `run_command.py` invocation per row. Per project memory: post-launch `pgrep -af '<TAG>'` on each node to verify exactly one PID per tag — duplicates are halt-conditions. Uses /tmp CIFS-bypass on each node per current convention. Pre-flight `python -c 'import jax'` per node before the first launch on that node.

### 7.4 experiment-analyzer (post-training)

After all 5 cells reach the SS window (≥8M episodes or budget-matched to parent), pulls:
- Steady-state survival per cell (P3 mean ± std across 3 seeds, P4/P5 single-seed).
- Δ_SS_new(P_k) for each profile (using parent's Unmod cells as baselines).
- Cross-experiment improvement Δ_SS_new − Δ_SS_old per profile.
- temp_max distribution across {old-ceiling parent, new-ceiling this experiment}: 3 + 1 + 1 + (parent's 1 + 1 + 1) = 8 numbers, plotted.
- gamma_multi_std comparison (this exp vs parent).
- Term_* breakdown per cell.

Then fills §5–§7 of this doc against §1.1 + §4.3 pre-registered predicates.

### 7.5 experiment-designer (results phase, looped back)

Designer (this agent) returns to fill §6 Conclusions + §6.3 Next Experiments based on the analyzer's output, applying the §1.1 + §4.3 confirmation/refutation rules without ambiguity (decisions were locked before launch).

If H₁a or H₁b confirms, the next experiment is a 3-seed cross-profile replication of the rescue (P4 + P5 at the new ceiling, 3 seeds each) to lock the finding before propagating `temp_clip: [0.5, 10.0]` to all NMN configs project-wide.

If H₀ / H₁c confirm, hand to `senior-developer` for §6.3 P3 architectural redesign with the temperature head explicitly removed from the redesign-target list (don't re-invent what isn't broken upstream).

---

## 8. Cross-references

- [NMN_NOISE_HETEROGENEITY_SWEEP](NMN_NOISE_HETEROGENEITY_SWEEP.md) §10.4, §11.3, §12.3 — the temp-saturation finding that motivated this experiment, and the parent sweep §12.3 next-experiments table.
- [NMN_PERFORMANCE_DIAGNOSIS_v8 §4.3.3](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) — original "MC FiLM saturates temp at 3.0" finding.
- [src/models/neuromodulator.py:71](../../../../src/models/neuromodulator.py) — `temp_clip` default `(0.1, 10.0)`; numerical envelope reference.
- [src/models/recurrent_ppo_network.py:264](../../../../src/models/recurrent_ppo_network.py) — `temp_clip` config-load site (mandatory dict access; no fallback).
- [docs/TEMPLATES/training_analysis.md](../../../TEMPLATES/training_analysis.md) — template this doc follows.

---

## 9. Results

> **Pre-registered statistic** (§4.1): SS_SS = mean Episode/Steps over windows 8–10 of 10 equal-episode bins, matching parent sweep convention.
> **Working file**: [tmp/20260509_124500_nmn_tempceil10_analysis.md](../../../../tmp/20260509_124500_nmn_tempceil10_analysis.md) (raw extraction at `tmp/20260509_nmn_tempceil10_extraction.json`).

### 9.1 Per-cell SS_SS and convergence

All 5 cells reached 10.0 M episodes (no preliminary caveats).

| Cell | Tag | SS_SS (w8-10) | SS_SS (w9-10) | w10 − w8 | Status |
|---|---|---:|---:|---:|---|
| P3s0 | `rppo_nmn_tempceil10_p3_film_g1_s0` | 222.63 | 223.98 | +5.32 | complete (10M); marginally still climbing |
| P3s1 | `rppo_nmn_tempceil10_p3_film_g1_s1` | 212.59 | 213.73 | +4.69 | complete (10M); plateaued |
| P3s2 | `rppo_nmn_tempceil10_p3_film_g1_s2` | 221.03 | 222.22 | +4.25 | complete (10M); plateaued |
| P4s0 | `rppo_nmn_tempceil10_p4_film_g1_s0` | 223.49 | 224.50 | +3.46 | complete (10M); plateaued |
| P5s0 | `rppo_nmn_tempceil10_p5_film_g1_s0` | 229.71 | 230.78 | +3.83 | complete (10M); plateaued |

P3s0 marginally exceeds the +5 threshold (matches parent P3f's +5.4 climb at the same profile — same regime, not a new pathology). All other cells plateaued.

### 9.2 Δ_SS_new and cross-experiment improvement (primary statistic)

Baselines from parent sweep, not re-run: P3u=221.69, P4u=226.04, P5u=226.42. Old-ceiling FiLM cells: P3f=221.77 (Δ_old=+0.08), P4f=217.12 (Δ_old=−8.92), P5f=221.71 (Δ_old=−4.71).

| Cell | SS_FiLM_new | SS_Unmod_old | Δ_SS_new | Δ_SS_old | improvement |
|---|---:|---:|---:|---:|---:|
| P3s0 | 222.63 | 221.69 | +0.94 | +0.08 | +0.86 |
| P3s1 | 212.59 | 221.69 | −9.10 | +0.08 | −9.18 |
| P3s2 | 221.03 | 221.69 | −0.66 | +0.08 | −0.75 |
| **P3 mean (3 seeds)** | 218.75 | 221.69 | **−2.94 ± 4.40** | +0.08 | **−3.02** |
| P4s0 | 223.49 | 226.04 | **−2.55** | −8.92 | **+6.37** |
| P5s0 | 229.71 | 226.42 | **+3.28** | −4.71 | **+7.99** |

### 9.3 Pre-registered predicate verdicts (design §1.1)

| Hypothesis | Predicate | Trigger numbers | Verdict |
|---|---|---|---|
| **H₁a (FiLM rescued)** | ≥1 of {P3 ≥2/3 seeds, P4, P5}: Δ_SS_new > +5 | P3 0/3 seeds ≥ +5; P4 = −2.55; P5 = +3.28 | **REFUTED** |
| **H₁b (gap narrowed, FiLM still ≤ Unmod)** | ≥1 profile: improvement > +5 AND Δ_SS_new ≤ 0 | P4: improvement +6.37, Δ_SS_new = −2.55 | **CONFIRMED on P4** |
| **H₁c (head non-engagement)** | ALL of {P3-mean, P4, P5}: temp_max < 3.5 AND \|improvement\| ≤ 1 | P4 temp_max=4.65, P5=4.51; P4 improvement +6.37, P5 +7.99 | **REFUTED** (head DOES use freed range) |
| **H₀ (no-op with full range usage)** | All cells temp_max ∈ [3.0, 10.0] AND \|improvement\| ≤ 1 | 2/5 cells temp_max < 3.0; P4/P5 improvement > +5 | **REFUTED** |
| **P3 3-seed disambiguation** | ≥ 2/3 P3 seeds: Δ_SS_new > 0 | 1/3 positive (+0.94, −9.10, −0.66) | **REFUTED** |

### 9.4 Modulator gate diagnostics (last 20%)

Reference (parent sweep, OLD ceiling 3.0): P3f temp_max=2.96, P4f=3.00, P5f=3.00 (saturated).

| Cell | temp_min | temp_mean | temp_max | gamma_multi_mean | gamma_multi_std | gamma_uni_mean | gamma_uni_std |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3s0 | 1.140 | 2.035 | **3.556** | 1.324 | 1.255 | 1.281 | 1.687 |
| P3s1 | 0.619 | 1.445 | 2.954 | 1.076 | 1.302 | 0.950 | 1.335 |
| P3s2 | 0.738 | 1.386 | 2.546 | 1.272 | 1.155 | 1.105 | 1.451 |
| P4s0 | 1.805 | 3.378 | **4.650** | 1.405 | 1.129 | 1.257 | 1.375 |
| P5s0 | 1.746 | 3.487 | **4.511** | 1.133 | 1.140 | 1.197 | 1.474 |

**temp_max OLD vs NEW ceiling:**

| Profile | OLD temp_max | NEW temp_max | climb |
|---|---:|---:|---:|
| P3 (3-seed mean) | 2.959 | 3.018 | +0.059 |
| P4 | 3.000 | 4.650 | **+1.650** |
| P5 | 3.000 | 4.511 | **+1.511** |

P4/P5 unambiguously engage the freed headroom; P3 stays near 3.0 on average (one seed pushes to 3.56, two stay below 3.0). The ceiling **was binding on P4 and P5** under the old regime; the head's natural target is meaningfully above 3.0 there.

### 9.5 Termination decomposition + loss diagnostics

| Cell | Term_Starv | Term_Injury | Term_MaxSteps | FoodEaten | TotalDamage | loss/value | grad_norm | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3s0 | 0.648 | 0.293 | 0.059 | 29.74 | 239.09 | 0.277 | 0.188 | −0.443 |
| P3s1 | 0.630 | 0.322 | 0.048 | 27.65 | 236.27 | 0.283 | 0.204 | −0.437 |
| P3s2 | 0.644 | 0.298 | 0.058 | 29.25 | 238.21 | 0.276 | 0.204 | −0.445 |
| P4s0 | 0.674 | 0.267 | 0.058 | 29.25 | 238.98 | 0.267 | 0.188 | −0.462 |
| P5s0 | 0.656 | 0.278 | 0.066 | 30.67 | 241.44 | 0.267 | 0.193 | −0.472 |

- Max grad_norm across cells: 0.204 (vs §5 instability threshold > 1.0). **No instability.**
- No cell has Term_Starvation > 0.90. Foraging signal intact.
- All Term decompositions are within parent-sweep envelope (Term_Starv 0.60–0.68; matches parent's saturated FiLM cells at 0.60–0.65).

### 9.6 Window-by-window survival evolution (mandatory temporal check)

| Cell | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3s0 | 154.5 | 188.4 | 200.8 | 207.7 | 213.9 | 216.2 | 218.6 | 219.9 | 222.8 | 225.2 |
| P3s1 | 133.8 | 178.2 | 186.6 | 195.4 | 199.5 | 204.6 | 209.3 | 210.3 | 212.5 | 215.0 |
| P3s2 | 151.1 | 188.4 | 200.2 | 206.2 | 211.8 | 215.9 | 217.6 | 218.6 | 221.6 | 222.9 |
| P4s0 | 159.2 | 188.2 | 197.7 | 206.6 | 212.0 | 215.9 | 218.2 | 221.5 | 224.1 | 224.9 |
| P5s0 | 161.8 | 194.0 | 205.0 | 212.0 | 216.9 | 220.2 | 223.7 | 227.5 | 230.2 | 231.4 |

P3s1 starts ~21 steps below the other two P3 seeds at w1 and remains the persistently-lower trajectory throughout — a per-seed effect from the start, not late divergence. P4s0 and P5s0 are smooth monotone climbers that plateau cleanly. None of P3s2 / P4s0 / P5s0 show late-training instability or regression.

---

## 10. Analysis

### 10.1 Key findings

**Finding 1 — The 3.0 ceiling was binding on P4 and P5 (the most-saturated profiles), but raising it to 10.0 produces only partial recovery.**
*What*: P4 temp_max climbed 3.00 → 4.65; P5 temp_max climbed 3.00 → 4.51. Both old and new ceilings sit below the head's natural target (otherwise temp_max would saturate at 10.0 too). The new headroom drives improvement of +6.37 (P4) and +7.99 (P5) in Δ_SS, but neither flips the FiLM-vs-Unmod sign.
*Why*: The FiLM modulator's temperature head, given more dynamic range, finds an operating point ~50% past the prior ceiling. That additional range cashes into ~+6 to +8 steps of survival. The remaining ~3-step deficit (P4) / ~3-step surplus (P5) suggests the temperature head is not the only bottleneck — gamma/beta gating accounts for some residual FiLM-vs-Unmod gap.
*Evidence*: §9.2 (Δ tables), §9.4 (temp_max climbs).
*Confidence*: High for the "head engages new headroom" claim (1.5+ unit climbs in temp_max are far above noise). Medium for "improvement is bounded" (single seed at P4 / P5).

**Finding 2 — H₁b confirms specifically at P4 (improvement +6.37, FiLM still 2.55 steps below Unmod). P5 produces an even bigger improvement (+7.99) and FiLM now LEADS Unmod by +3.28 steps — but at single seed, this is on the edge of seed noise.**
*What*: P5 is the only profile where FiLM_new > Unmod_old. Δ_SS_new = +3.28 is below the H₁a +5 threshold, so it does not confirm rescue, but it is the clearest sign of partial structural rescue at the highest-noise profile.
*Why*: Higher noise = more value for adaptive temperature → more value to the rescue. This is consistent with the parent-sweep mechanism (saturation worsened with R) and with ceiling-as-bottleneck.
*Evidence*: §9.2 row P5s0; cross-profile comparison: improvement P3 −3.02 → P4 +6.37 → P5 +7.99 is monotonically increasing in noise R.
*Confidence*: Medium — the +3.28 lead at P5 is below seed-noise envelope (P3 had σ_seeds = 4.4 across 3 seeds), so P5's lead would need 3-seed replication to confirm.

**Finding 3 — P3 is profile-specifically decoupled from the ceiling: 3-seed P3 produces 1/3 positive Δ (refuting the disambiguation) and the temp_max barely climbs (3.02 vs old 2.96).**
*What*: Of the 3 P3 seeds, only P3s0 has Δ_SS_new > 0 (+0.94); P3s1 is a strong negative (−9.10) and P3s2 is essentially zero (−0.66). The 3-seed mean is **−2.94 ± 4.40**, so the P3 deviation is dominated by seed dispersion, not by the ceiling change.
*Why*: P3 (R=5.0, log R=1.61) is on the boundary between un-saturated (P1/P2) and saturated (P4/P5). The parent's +0.08 was a single-seed coincidence inside a high-variance distribution, not a real signal. The new ceiling is non-binding here (temp_max stays at 3.02).
*Evidence*: §9.2 P3 seed scatter; §9.4 P3 mean temp_max climb of only +0.06 vs P4's +1.65 and P5's +1.51; §9.6 row P3s1 starts ~21 steps below other seeds at w1 (a seed pathology, not a late training collapse).
*Confidence*: High — the per-seed scatter and the absence of temp_max climb both point the same direction.

**Finding 4 — The temperature head's "natural target" lives in [3.0, ~5.0), not at 10.0. Both P4 and P5 stabilise around 4.5–4.7; neither saturates at the new ceiling.**
*What*: New temp_max for P4 = 4.65 and P5 = 4.51. These are *not* pinned at 10.0, so the new ceiling is not binding even on the most-saturated profiles. The head's expressed target is in the [3.0, 5.0) window.
*Why*: Had we picked `[0.5, 5.0]` instead of `[0.5, 10.0]`, it would still have been binding on P4/P5 (4.65 > 5.0 not quite, but 4.51 close to ceiling), and we wouldn't have known whether the head wanted more range. Now we know: it doesn't.
*Evidence*: §9.4 temp_max table.
*Confidence*: High — single seed but 2 profiles, and they converge to similar values.

**Finding 5 — No instability and no behavioural pathology. The freed range is safely usable.**
*What*: grad_norm stays at 0.19–0.20 (parent sweep was 0.18–0.20); Term_Starvation at 0.63–0.67 (parent: 0.60–0.65); FoodEaten 27.7–30.7 (parent: 28–31). The new ceiling does not introduce any new failure mode.
*Why*: 10.0 is within the architecture's original numerical envelope (`(0.1, 10.0)` per `src/models/neuromodulator.py:71`); the prior `[0.5, 3.0]` was a tightening, not a default.
*Evidence*: §9.5; failure-mode catalog §5 — no entry triggered.
*Confidence*: High.

### 10.2 Cross-run comparisons

**Old ceiling (3.0) vs new ceiling (10.0), matched profile, matched baseline:**

| Profile | Δ_SS_old (3.0) | Δ_SS_new (10.0) | improvement | new SS_FiLM | new SS_Unmod | new sign |
|---|---:|---:|---:|---:|---:|:-:|
| P3 (3-seed) | +0.08 (1 seed) | −2.94 ± 4.40 | −3.02 | 218.75 | 221.69 | FiLM ≤ Unmod |
| P4 | −8.92 | −2.55 | **+6.37** | 223.49 | 226.04 | FiLM ≤ Unmod |
| P5 | −4.71 | +3.28 | **+7.99** | 229.71 | 226.42 | FiLM > Unmod |

The improvement-vs-R trend is monotone-increasing: the higher the noise (and thus the more saturated under the old ceiling), the larger the rescue. Consistent with "ceiling was binding". P3 is anomalous — improvement is negative, but absolute Δ is within seed noise (σ_seeds = 4.4), so the P3 result is operationally null.

### 10.3 Failure modes & pathologies

§5 of the design pre-registered 8 failure modes. Outcomes:

| Pre-registered failure mode | Observed? |
|---|:-:|
| All 5 cells temp_max ≈ 3.0 (head non-engagement) | No (P4/P5 climbed +1.5) |
| temp_max climbs AND survival improvement > 5 on ≥1 cell | **Yes (P4 +6.37, P5 +7.99) — H₁b path** |
| temp_max climbs but no SS gain (rich H₀) | No (gain on P4/P5) |
| 3-seed P3 wide spread (σ > 5) | Borderline (σ = 4.40) — recommend extended seeds |
| NaN / gradient explosion | No (max grad_norm 0.204 ≪ 1.0) |
| Term_Starvation > 90% | No (max 0.674) |
| Old vs new P3 ≈ same SS | Yes (Δ_old +0.08 vs Δ_new mean −2.94 — P3-specific decoupling) |
| <8M ep cell | No (all 10M) |

**One non-pre-registered note:** P3 cross-seed σ = 4.40 is high relative to the 5-step rescue threshold. If P4/P5 also have σ in this range at single seed, the +3.28 P5 lead and the +6.37 P4 improvement are both within one σ — important context for any rescue claim downstream.

---

## 11. Conclusions

### 11.1 Summary

- **Pre-registered hypothesis: REFUTED.** H₁a (FiLM rescued by raising the ceiling) does not hold on any profile at the +5-step threshold. The original `temp_clip: [0.5, 3.0]` was not the bottleneck rescuing FiLM into the Unmod regime; it was a contributor that explains ~+6 to +8 steps of the parent sweep's FiLM deficit, no more.
- **H₁b CONFIRMED on P4** (improvement +6.37, Δ_SS_new still −2.55). The temperature head was binding *and uses the freed range*, but freeing it is insufficient. Per design §1.1, this routes to senior-developer for §6.3 P3 architectural redesign with the constraint "any redesign must not regress this partial improvement".
- **P5 is the marginal case**: improvement +7.99 and Δ_SS_new = +3.28 (FiLM now LEADS Unmod), but +3.28 is below the +5 H₁a threshold and is at single seed within the seed-noise envelope. *Suggestive* of architectural rescue at the highest-noise profile; *not* confirmation.
- **The temperature head's natural target is [3.0, 5.0).** Both P4 (4.65) and P5 (4.51) settle well below the new 10.0 ceiling. The redesign should NOT plan around expanding the ceiling further; the right structural intervention is downstream of the temperature head (gamma/beta gating, modulator output capacity, or a different conditioning path entirely).
- **P3 was always profile-specific.** The parent-sweep +0.08 at single seed was inside a 3-seed σ of 4.4 — a coincidence, not a signal. P3 disambiguation REFUTED. Any narrative tying "P3 is special" to architecture should be retracted.

### 11.2 Limitations & open questions

- **P4/P5 at single seed.** The +6.37 P4 improvement and +3.28 P5 lead need 3-seed replication before architectural commitment. The P3 σ=4.4 result is a concrete warning that single-seed numbers in this regime can be off by ±5.
- **The 3-seed P3 σ is large for the 10-step regime.** Even if mean Δ is at zero, the spread (4.4) erodes confidence that a 1–3-step rescue at any profile would be reproducible. Recommend running ≥5 seeds when the next architectural variant is benchmarked, not 3.
- **P3s0 is marginally still climbing** (w8→w10 = +5.32). Matches parent P3f's +5.4 — same regime. Not a bug, but a reminder that 10M ep is the floor for these cells, not a cushion.
- **The temperature head saturates around [3, 5).** Why? The design did not pre-register a mechanism for *why* the head's optimal output sits at ~4.5 on P4/P5. A targeted ablation (different `temp_clip` shape, e.g., asymmetric, or no clip at all) could confirm this is a head-output property, not a clip-edge artefact.

### 11.3 Recommended next experiments

| Priority | Experiment | Rationale | Effort |
|---|---|---|---|
| **High** | §6.3 P3-style architectural redesign of FiLM-MC modulator, with the temperature head explicitly de-prioritised. Redesign should target the gamma/beta gating path or the encoder→modulator interface, NOT the temperature head. | H₁b confirms ceiling was a partial bottleneck; structural intervention now needed. Per design §5: hand-off to senior-developer. | 1–2 weeks platform planning + 1 week training |
| **High** | Before locking the redesign target, run **3-seed replication of P4 and P5 at `temp_clip: [0.5, 10.0]`** (6 cells total). | The +6.37 P4 / +3.28 P5 / +7.99 P5-improvement numbers are at single seed; P3's σ=4.4 warning means we should not commit to a redesign baseline on numbers that may be ±5. | 6 cells × ~6 hr = 36 cell-hr; can run in parallel batches of 3 on lab nodes. |
| Med | **Update v8 diagnosis (`docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md` §4.3.3)** to reflect: (a) the ceiling was *contributory* (partial — +6 to +8 step recovery on P4/P5), not the binder; (b) the head's natural target is in [3.0, 5.0); (c) the P3 +0.08 was not real. | The v8 diagnosis was the original "saturates at 3.0" finding; this re-run upgrades it from "ceiling artefact" to "partial bottleneck with structural component". | <1 day docs work |
| Low | Asymmetric / no-clip ablation on the temperature head — does the head saturate around 4.5 *because* the clip exists at 10.0, or is 4.5 its true natural target? | Would refine the redesign brief but does not gate it. | 2–3 cells |
| Low | Propagate `temp_clip: [0.5, 10.0]` to *non*-NMN-noise NMN configs (the v8 single-seed configs, etc.) to check whether the +6 to +8 step recovery generalises to non-NMN P1-style settings. | Wider validation of the partial-rescue finding. Low priority because the rescue does not flip FiLM-vs-Unmod sign in the heterogeneity regime. | 4–6 cells |

### 11.4 Hand-off (per design §5 + §1.1 verdicts)

**Verdict path:** H₁b CONFIRMED at P4 → design's pre-registered route is `senior-developer` for §6.3 P3 architectural redesign with "any redesign must not regress this partial improvement".

However, given Finding 2 (P5 +3.28 lead is suggestive but at single seed) and the P3 σ=4.4 cross-seed warning, the **most cost-effective next step before committing to the redesign** is the high-priority replication in the table above (3 seeds × P4/P5). The replication is 6 cells (~36 cell-hr) and resolves whether the rescue is real or seed noise. The redesign can proceed in parallel — the redesign brief is independent of whether the partial rescue is +6 or +0.

---

## Appendix

### A. Parent-sweep saturation table (reproduced for reference)

From [NMN_NOISE_HETEROGENEITY_SWEEP §10.4](NMN_NOISE_HETEROGENEITY_SWEEP.md#104-film-gate-diagnostics-last-20-means-design-54):

| Cell | R | gamma_multi_std | temp_mean | temp_max | Saturated? |
|---|---:|---:|---:|---:|---|
| P1f | 2.00 | 1.293 | 0.812 | 2.299 | No |
| P2f | 2.75 | 1.159 | 0.874 | 1.505 | No |
| P3f | 5.00 | **1.394** | **2.196** | 2.959 | Yes (just barely) |
| P4f | 9.67 | 0.922 | **2.369** | **3.000** | Yes (pinned) |
| P5f | 18.0 | 1.031 | **2.350** | **3.000** | Yes (pinned) |

### B. Config Diffs

The single new file `recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` differs from the parent sweep's `recurrent_ppo_nmn_het_film_g1.yaml` by exactly one substantive line: `temp_clip: [0.5, 10.0]` (was `[0.5, 3.0]`). Comment headers updated to document the diff. All other lines byte-identical.

### C. Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-08 | Initial design + 1 new agent config + 5-row launch manifest | experiment-designer |
| 2026-05-09 | §9 / §10 / §11 filled from completed 5-cell extraction; H₁b CONFIRMED on P4; H₁a/H₁c/H₀/P3-disambiguation REFUTED; status → COMPLETE | experiment-analyzer |
