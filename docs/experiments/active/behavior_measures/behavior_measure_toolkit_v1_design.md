---
title: "Fine-grained behavior-measure toolkit v1 — pre-registered protocol (M1, M2, M5, M7)"
topic: behavior_measures
status: planned
created: 2026-05-11
last_updated: 2026-05-11
phase: 1
wandb_tag: "behavior-measures-v1"
supersedes: []
---

# Fine-grained behavior-measure toolkit v1 — pre-registered protocol

> **Status**: PLANNED — implementation not yet started; awaits `senior-developer` plan.
> **Author**: experiment-designer
> **Related**:
> - Idea memo (definitions + biological grounding): [`docs/project/ideas/20260510_behavior_measure_toolkit.md`](../../../project/ideas/20260510_behavior_measure_toolkit.md)
> - Immediate motivating verdict: insight `20260510_2237_sameprop_round25_no_class_avoidance` ([file](../../../../docs/llm_wiki/entries/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md))
> - Architectural precedent (per-tag distance): [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md)
> - Round 2.5 design (the toolkit's first paper-grade demonstration target): [`docs/experiments/active/hypervigilance/sameprop_round25_design.md`](../hypervigilance/sameprop_round25_design.md)

---

## 0. Plain-English entry point

For most of this project, behaviour has been read off mean-statistics — average distance to predator vs. rabbit, hits per episode, survival steps. The just-resolved sameProp study (Round 2.5) showed why that is not enough. When the patrolling predator and the neutral rabbits carried identical olfactory smells, the agent ended up keeping the predator ~3.86 cells farther away than rabbits on aggregate. That looked like clean class recognition. The per-tag distance metric shipped this week (one number per labelled rabbit and labelled predator) revealed the truth: the same-corner predator and the same-corner rabbit ended up at *identical* distance from the agent (Δ_TL = +0.004 cells). The agent never visited the predator's quadrant at all — it survived by **camping the safe corner**, not by recognising the predator class. Mean distances dissolved trajectory-level dynamics; per-tag distances recovered the spatial picture but are still episode-mean statistics. They cannot tell us whether the agent **interrupts feeding** when a predator approaches, **dives into a bush**, or whether the same threat triggers the same response on its 3rd encounter as on its 1st.

This document is the pre-registered protocol for the **first round of fine-grained behaviour measures** — four measures lifted from the postdoc's 8-candidate menu that the user has authorised to ship. The four measures are: (M1) **interrupted-feeding rate** — when the agent is eating and a class-c entity enters its olfactory radius, what fraction of the time does it stop eating in the next 5 steps; (M2) **bush-dive rate** — what fraction of "threat enters radius" events trigger an active dive into cover; (M5) **eat-under-threat ratio** — risk-discounted foraging, the per-step probability of eating with a predator nearby divided by the same probability when safe; (M7) **defensive-motif repertoire** — offline clustering of K-step trajectory windows around threat-onset events into named behavioural motifs (`freeze`, `flight`, `bush_dive`, `freeze_then_flight`, `ignore`, `approach`).

The first paper-grade demonstration of this toolkit is to **recover the trajectory-level class-conditional dynamics that mean-distance metrics dissolved away** — applied first to the saved Round-2.5 Cell A1 (corner-camping) and Cell C (food-decoupled, bilateral rabbit avoidance) checkpoints. The toolkit lives long-term in the codebase: every future hypervigilance / NMN / cross-architecture study reads these measures.

---

## 1. Research Question

State each measure as a falsifiable predicate on Round-2.5 Cell A1 and Cell C agents (n=1 seed each, saved checkpoints) and on any future agent the toolkit is applied to.

### 1.1 Formal predicates

Let `R = 3.0` cells (cue radius) and `K = 5` steps (observation window) be the locked toolkit parameters. Let `c ∈ {predator, rabbit}` index the entity class. Let `t` index per-step time within an episode of length `T_ep`.

- **M1 — `interrupted_feeding_rate_<c>`**. Per-episode scalar, fraction of steps `t` where the agent is eating *and* a class-`c` entity is within radius `R`, that are followed within `K` steps by the agent stopping eating.

  Formally, let `cand_t = info['ate_food'][t] ∧ (min(dist_per_<c>[t]) < R)`. Let `interrupt_t = ¬info['ate_food'][t+1:t+K+1].any()` (no eating event in the next K steps; equivalently, `(info['ate_food'][t+1:t+K+1] == False).all()`). Then `interrupted_feeding_rate_c = (Σ_t cand_t · interrupt_t) / max(Σ_t cand_t, 1)`.

  - **H₀(M1)**: agent's `interrupted_feeding_rate_predator` = `interrupted_feeding_rate_rabbit` (the agent does not class-condition feeding interruption).
  - **H₁(M1)**: `interrupted_feeding_rate_predator > interrupted_feeding_rate_rabbit + 0.20` (the agent is at least 20 percentage points more likely to stop eating when a predator approaches than when a rabbit does).

- **M2 — `bush_dive_rate_<c>`**. Per-episode scalar, fraction of "threat enters radius" events that trigger an active bush dive.

  Formally, let `enter_t = (min(dist_per_<c>[t-1]) ≥ R) ∧ (min(dist_per_<c>[t]) < R) ∧ ¬info['agent_in_bush'][t]` (threat just entered radius and agent is not already in a bush). Let `dive_t = info['agent_in_bush'][t+1:t+K+1].any()` (agent enters a bush within K steps). Then `bush_dive_rate_c = (Σ_t enter_t · dive_t) / max(Σ_t enter_t, 1)`.

  - **H₀(M2)**: agent's `bush_dive_rate_predator` = `bush_dive_rate_rabbit` (no class-conditional active defence).
  - **H₁(M2)**: `bush_dive_rate_predator > bush_dive_rate_rabbit + 0.15` AND `bush_dive_rate_predator > 0.10` (the agent dives into a bush in at least 10% of predator-approach events, and at least 15 percentage points more often than for rabbit-approach events).

- **M5 — `eat_under_threat_ratio_<c>`**. Per-episode scalar, ratio of per-step eat probability under threat vs. safe.

  Formally, let `under_threat_t = (min(dist_per_<c>[t]) < R)` and `safe_t = ¬under_threat_t`. Let `P_eat|threat = (Σ_t under_threat_t · info['ate_food'][t]) / max(Σ_t under_threat_t, 1)`, `P_eat|safe = (Σ_t safe_t · info['ate_food'][t]) / max(Σ_t safe_t, 1)`. Then `eat_under_threat_ratio_c = P_eat|threat / max(P_eat|safe, ε)` with `ε = 1e-6`.

  - **H₀(M5)**: ratio ≈ 1.0 for both classes (no risk discounting).
  - **H₁(M5)**: `eat_under_threat_ratio_predator < 0.7` AND (`eat_under_threat_ratio_rabbit > 0.85` OR `eat_under_threat_ratio_predator < eat_under_threat_ratio_rabbit − 0.20`) (agent is risk-averse specifically to predators).

- **M7 — `defensive_motif_repertoire`**. Offline-only. For each agent / condition, the **distribution over named motifs** of the population of K-step `(state-summary, action)` windows around threat-onset events, computed across all eval-rollout episodes. Hypothesis is qualitative — see §4.2 for the cluster-redistribution check.

### 1.2 Sanity / refutation criteria

- **R1**: All four measures must produce finite values in their canonical range (`[0, 1]` for M1/M2; `[0, ∞)` for M5; valid simplex for M7). NaN / inf indicates a wiring bug.
- **R2**: Aggregate per-tag fan-outs must satisfy: `interrupted_feeding_rate_predator = mean over tags ∈ predator of interrupted_feeding_rate_predator_<tag>` (same-class consistency).
- **R3**: M5's `P_eat|safe` numerator must be > 0 in ≥ 90% of episodes; if not, the agent is failing to eat in safe periods and M5 is uninterpretable for that agent.
- **R4 (M7 only)**: motif clusters must have silhouette score > 0.20 OR the chosen `K_clusters` must justify in §4.2 why a lower silhouette is acceptable (e.g., very imbalanced motif sizes). Below 0.20 with no justification → M7 reads as "uninterpretable" and the figure is dropped.

---

## 2. Measure Specifications

For each of M1, M2, M5, M7: operational definition, WandB key list, online vs offline placement, and a pre-registered baseline prediction for Round-2.5 Cell A1 and Cell C.

### 2.1 M1 — `interrupted_feeding_rate_<class>` (online)

**Operational definition.** Per-episode, count "eating-while-threat-near" candidates and the subset that interrupt within K steps. See §1.1 M1 for the formal predicate.

**Conditioning**: per-episode aggregation; per-class fan-out over `c ∈ {predator, rabbit}`; per-tag fan-out over the existing `EnvParams.predator_tags` / `EnvParams.neutral_tags` tuples (e.g., `_TL`, `_BR`, `_idx0`).

**Edge cases**: if `Σ_t cand_t = 0` for a given class (the agent never ate while a class-c entity was within radius), the episode-scalar is emitted as `NaN` and the WandB step is **skipped for that key** (not logged as 0). The episode counter for "non-NaN episodes" is logged separately as `Episode/InterruptedFeedingDenominator_<class>` so a run with all-NaN can be diagnosed at the analysis stage. (Same NaN-skip pattern as the existing per-tag distance keys when an entity list is empty.)

**WandB key list** (every key emitted; per-class fan-out always emitted, per-tag fan-out emitted iff that tag exists in the live config):

```
Episode/InterruptedFeedingRate_predator
Episode/InterruptedFeedingRate_rabbit
Episode/InterruptedFeedingRate_predator_<tag>      (one per predator_tag)
Episode/InterruptedFeedingRate_rabbit_<tag>        (one per neutral_tag)
Episode/InterruptedFeedingDenominator_predator     (Σ_t cand_t, integer)
Episode/InterruptedFeedingDenominator_rabbit       (Σ_t cand_t, integer)
```

**Placement**: online. Three integer accumulators per class per episode (`candidates`, `interrupted`, `denom`). No JAX changes — uses the existing `info['ate_food']` (bool) and `info['dist_per_<class>']` (vector) per-step. Lives in the same five `train.py` accumulator sites the per-tag work touches. Online-suitable because the K-step look-ahead can be implemented with a **per-env circular buffer of length K** holding the most recent `info['ate_food']` values; at step `t+K` the buffer's value at offset 0 reveals whether the candidate at step `t` was followed by a stop-eating in the next K steps. See §7 for the implementation surface; the senior-developer authors the actual plan.

**Baseline predictions** (Round-2.5 saved checkpoints; n=1 each; eval-rollout protocol §3):

| Agent | Predicted M1_predator | Predicted M1_rabbit | Predicted gap | Reading |
|---|---:|---:|---:|---|
| Round-2.5 Cell A1 (`nm8gn7y2`, corner-camping) | NaN OR ≤ 5% | NaN OR ≤ 5% | undefined | Agent rarely eats anywhere except BR; predator is in TL; threat-near-while-eating events are vanishingly rare. The denominator `Episode/InterruptedFeedingDenominator_predator` should be near zero on most episodes, exposing the camping signature directly. |
| Round-2.5 Cell C (`bdnfc0lu`, bilateral rabbit avoidance) | 10–25% | 10–25% (within 5pp of predator) | < 0.05 | The Round-2.5 verdict is "no class-conditional avoidance under sameProp." If the verdict is correct, M1 should match — predator and rabbit interruption rates within 5 percentage points. |
| Round-1 baseline (`rg5nl1ov` or similar, before per-tag) | 30–50% | 25–45% | < 0.10 | Round-1 had no quadrant constraint; agent eats freely; class-conditional interruption is the live question, but the prior on "no genuine class avoidance under sameProp" predicts a small (< 10pp) gap. |

If any cell shows a gap > 20pp at the predicted-NaN cells (A1) or > 10pp (C, R1), that is **either a real class-conditional defensive signal or a measure bug** — the next step is the §5 failure-mode catalog before claiming a signal.

### 2.2 M2 — `bush_dive_rate_<class>` (online; requires new env hook)

**Operational definition.** Per-episode, count "threat enters radius" events and the subset that trigger an agent-enters-bush event within K steps. See §1.1 M2.

**New env hook required**: `info['agent_in_bush']` (bool, per step). The bush concealment computation already exists in `core.py` line 149-152 (`agent_hidden = jnp.any(jnp.logical_and(jnp.all(obs_pos == agent_pos, axis=-1), params.obs_hides_agent))`) — but it lives inside the predator state-machine block, not in the `info` dict. Surfacing it costs one extra computation in `step()` (or reuse of the existing `agent_hidden` value if scope permits — the senior-developer's call). Names this as a `senior-developer` ticket in §7.

**Conditioning**: per-episode aggregation; per-class fan-out (M2 conditions on the class `c` that triggered the threat-enter event, NOT on the bush instance — bushes are class-agnostic refuge); per-tag fan-out over the threat tag.

**Edge cases**: if `Σ_t enter_t = 0` (no threat ever entered radius this episode), emit NaN + log `Episode/BushDiveDenominator_<class> = 0`. If the agent **starts** the episode within a bush, mask early `enter_t` predicates accordingly (the `¬info['agent_in_bush'][t]` clause already handles this). If `params.obs_hides_agent` is all-False (no bushes in the config), emit NaN for all M2 keys and a config-level warning at startup.

**WandB key list**:

```
Episode/BushDiveRate_predator
Episode/BushDiveRate_rabbit
Episode/BushDiveRate_predator_<tag>
Episode/BushDiveRate_rabbit_<tag>
Episode/BushDiveDenominator_predator    (Σ_t enter_t, integer)
Episode/BushDiveDenominator_rabbit      (Σ_t enter_t, integer)
```

**Placement**: online. Three integer accumulators per class per episode. Per-env state machine holds `prev_dist_under_R_<class>` (1 bit per class per env) + a length-K circular buffer of `info['agent_in_bush']` to look ahead. Lives in the same five sites.

**Baseline predictions**:

| Agent | Predicted M2_predator | Predicted M2_rabbit | Reading |
|---|---:|---:|---|
| Round-2.5 Cell A1 (corner-camping) | NaN (denominator ≈ 0) | NaN OR ≤ 5% | The agent never lets either entity enter its radius in BR (TL predator far; BR rabbit static). Denominator near zero — diagnoses camping directly. |
| Round-2.5 Cell C (bilateral rabbit avoidance) | 5–15% | 5–15% | Per the Round-2.5 verdict (no class-conditional defence). Both rates non-trivially > 0 because the agent does occasionally cross paths with both classes en route to food in TR/BL. |
| Round-1 baseline | 5–20% | 0–10% | Mild prior on "agent prefers cover when predator approaches more than rabbit", but Round-1's apparent rabbit-vs-predator gap was the food confound, so M2 may also be near-equal. |

**Bushes in the existing configs**: spot-check before launch — see §6 schema for the `behavior_measures.enabled` gate which the env-config-auditor must verify each config's `obstacles` list contains at least one entry with `hides_agent: true` if M2 is enabled.

### 2.3 M5 — `eat_under_threat_ratio_<class>` (online)

**Operational definition.** Per-episode ratio of per-step eat probability under threat vs. safe. See §1.1 M5.

**Conditioning**: per-episode; per-class fan-out (predator-threat vs. rabbit-threat); per-tag fan-out for predator + rabbit threats. M5 does NOT need a K-window — it is an episode-level ratio of step counts, which makes it the cheapest of the four to compute.

**Edge cases** (R3 sanity criterion): if `Σ_t safe_t = 0` (the entire episode was under threat; rare but possible), emit NaN + log `Episode/EatUnderThreatSafeSteps_<class> = 0`. If `P_eat|safe < 0.005` (the agent ate < 0.5% of safe steps — a degenerate non-eater), still emit but flag the run; downstream M5 readings on such runs are uninterpretable. The numerator/denominator step-counts are logged separately to make this diagnosable.

**WandB key list**:

```
Episode/EatUnderThreatRatio_predator
Episode/EatUnderThreatRatio_rabbit
Episode/EatUnderThreatRatio_predator_<tag>
Episode/EatUnderThreatRatio_rabbit_<tag>
Episode/EatUnderThreatRate_predator             (P_eat|threat, fraction in [0,1])
Episode/EatUnderThreatRate_rabbit
Episode/EatSafeRate_predator                    (P_eat|safe; predator-safe = predator not within R)
Episode/EatSafeRate_rabbit
Episode/EatUnderThreatSafeSteps_predator        (Σ_t safe_t, integer; for R3 sanity)
Episode/EatUnderThreatSafeSteps_rabbit
```

**Placement**: online. Four integer counters per class per episode (`under_threat_steps`, `safe_steps`, `eats_under_threat`, `eats_safe`). Five-site accumulator pattern.

**Baseline predictions**:

| Agent | Predicted M5_predator | Predicted M5_rabbit | Reading |
|---|---:|---:|---|
| Round-2.5 Cell A1 (corner-camping) | undefined / NaN | undefined / NaN | Almost no "under_threat" steps because the agent never approaches the TL predator. R3 likely fires for the predator branch on every episode. Diagnoses camping cleanly. |
| Round-2.5 Cell C (bilateral rabbit avoidance) | 0.6–0.9 | 0.6–0.9 (within 0.10) | Per the verdict; no class-conditional risk discounting. |
| Round-1 baseline | 0.6–0.9 | 0.7–0.95 | Prior assumes mild risk-discounting on predator + minimal on rabbit. Gap predicted < 0.20 (per the same sameProp verdict). |

### 2.4 M7 — `defensive_motif_repertoire` (offline only)

**Operational definition.** Cluster K-step trajectory windows around threat-onset events into named motifs; report the distribution.

**Window definition**: For each threat-onset event in an eval-rollout episode (let `t*` be the first step where `min(dist_per_<class>[t*]) < R` AND `min(dist_per_<class>[t* − 1]) ≥ R`), extract the window `[t* − 2, t* + K_motif]` where `K_motif = 7` steps (slightly longer than the online `K = 5` to leave room for delayed responses; rationale: pilot-feel based on the postdoc's recommendation of "K-step or slightly longer windows"; if §4.2 finds clusters dominated by `K = 5` truncation effects, revise to `K_motif = K = 5` in v2). The window is **threat-window-only** (not full-episode) per user directive.

**Window contents (per step)**:
- `agent_pos[t]` — `[2]` int
- `action[t]` — int (0–5; the env's 6-action space)
- `dist_per_predator[t]`, `dist_per_neutral[t]` — `[num_pred]`, `[num_neutral]` float
- `info['ate_food'][t]` — bool
- `info['agent_in_bush'][t]` — bool (depends on M2's env hook)
- `info['hit_predator'][t]`, `info['hit_neutral'][t]` — bool
- `info['drive_hunger'][t]`, `info['drive_injury'][t]` — float
- The class `c` that triggered the onset (predator or rabbit)
- The tag of the triggering instance (e.g., `predator_TL`)
- Episode index, agent identifier (run tag), checkpoint step

**Featurisation** (handcrafted; ship 10 features in v1, room to grow):

| # | Feature | Definition over the K_motif+2 step window |
|---|---|---|
| 1 | `net_displacement` | `||agent_pos[t* + K_motif] − agent_pos[t* − 2]||₂` |
| 2 | `path_length` | `Σ_t ||agent_pos[t+1] − agent_pos[t]||₂` over `[t* − 2, t* + K_motif − 1]` |
| 3 | `threat_distance_change_rate` | `(dist_to_threat[t* + K_motif] − dist_to_threat[t* − 2]) / (K_motif + 2)`; threat = the triggering instance (positive = retreat, negative = approach) |
| 4 | `min_threat_distance` | `min over t of dist_to_threat[t]` across the window |
| 5 | `bush_occupancy_fraction` | `Σ_t info['agent_in_bush'][t] / (K_motif + 3)` |
| 6 | `eat_events_per_window` | `Σ_t info['ate_food'][t]` (integer, capped at K_motif + 3) |
| 7 | `action_entropy` | `H(p)` where `p` is the empirical action histogram over the window (6 bins; natural log) |
| 8 | `mode_action_fraction` | `(count of mode action) / (K_motif + 3)` (mode-action concentration; high = stereotyped, low = mixed) |
| 9 | `stay_in_place_fraction` | `(count of `stay`-equivalent actions, i.e., position unchanged) / (K_motif + 2)` (proxy for freezing) |
| 10 | `drive_injury_change` | `info['drive_injury'][t* + K_motif] − info['drive_injury'][t* − 2]` (positive = took damage in the window) |

The final feature set may be 8–12 — the senior-developer / analysis script may drop or merge if any feature is degenerate at implementation time. Locked at the value of 10 for v1 unless the senior-developer surfaces a reason in their plan.

**Clustering**:
- **Algorithm**: k-means with k = 6, fixed (matches the postdoc's named motif vocabulary `freeze`, `flight`, `bush_dive`, `freeze_then_flight`, `ignore`, `approach`). v1 ships fixed k; v2 (after the first run) may switch to silhouette-score-based k selection over `k ∈ {4, 5, 6, 7, 8}` if the fixed-k clusters look noisy.
- **Standardisation**: per-feature z-score using the pooled-across-conditions mean and std (so cluster centroids are comparable across agents).
- **Labelling**: post-hoc human inspection of the 5 nearest-centroid exemplar windows per cluster, after the first eval-rollout completes. Each cluster gets one English name (the v1 vocabulary above is the default; any cluster that doesn't fit gets a custom name documented in the analysis doc).
- **Reproducibility**: clustering seed locked in `behavior_measures.motif_kmeans_seed` (mandatory YAML key, §6).

**Deliverable**: stacked-bar chart of motif distribution per agent / per condition (Round-2.5 Cell A1 vs. Cell C vs. Round-1 baseline; in future studies, NMN vs. baseline).

**Placement**: offline only. Reads the trajectory-dump parquet files produced by `scripts/eval_rollout.py` (§3); clustering performed by `scripts/motif_cluster.py` (§7); both scripts authored by `senior-developer` after this design lands.

**WandB key list** (offline → published as a fixed-name JSON sidecar in the eval-rollout output dir; NOT pushed to WandB during training):

```
results/eval/<run_tag>/<checkpoint_step>/motifs/
  feature_vectors.parquet      # one row per threat-onset window
  cluster_assignments.parquet  # one row per window, labels assigned post-hoc
  motif_distribution.json      # {cluster_label: fraction, ...} per agent
  exemplars.json               # nearest-centroid window indices per cluster
  cluster_centroids.npy        # K_clusters × num_features
  silhouette.json              # {silhouette_mean: float, per_cluster: [...]}
```

**Baseline predictions**:

| Agent | Predicted dominant motifs | Reading |
|---|---|---|
| Round-2.5 Cell A1 | `ignore` (≥ 80%) — almost no threat-onset events at all (denominator near zero); whatever onsets occur look like `ignore` because the agent's response is "stay in BR corner". | Confirms camping is the dominant policy; NOT a class-conditional defensive repertoire. |
| Round-2.5 Cell C | `flight` (40–60%) and `ignore` (30–50%); `bush_dive` likely < 5%. | Bilateral rabbit avoidance reads as flight on both classes; the absence of `bush_dive` confirms the agent did not learn an active defence routine — it learned a kinematic avoidance. |
| Round-1 baseline | `ignore` (40–60%), `flight` (20–35%), `bush_dive` (5–15%), `approach` (5–15%) | Less spatially constrained → richer repertoire; but no strong class-asymmetric prediction (per the same verdict). |

---

## 3. Eval-Rollout Protocol

The offline path (M7 + sanity-replay of M1/M2/M5 from saved checkpoints) needs a deterministic eval rollout. Specs below are the v1 defaults; all knobs are mandatory YAML keys (§6).

### 3.1 Rollout parameters

| Parameter | v1 default | Rationale |
|---|---|---|
| `eval_n_episodes` | 200 | Postdoc-recommended; enough for the 6-cluster k-means to find stable centroids (rule-of-thumb: ≥ 30 windows per cluster; 200 episodes × ~3 onsets/episode ≈ 600 windows). |
| `eval_seeds` | `[1000, 1001, ..., 1199]` (200 contiguous seeds, locked in YAML) | Fixed held-out seed list, disjoint from training seeds (training uses 0–999). Reproducibility across reruns. |
| `eval_policy_mode` | `deterministic` | argmax over policy logits (RPPO) or actor-mean (Dreamer); reproducible across reruns. Stochastic mode available as a v2 sub-study. |
| `eval_max_steps` | match the env's `max_steps` (typically 500) | Same as training. |
| `eval_obs_noise` | match training-time noise schedule at the eval checkpoint (NOT zeroed) | Eval is a "held-out test set" of the training distribution, not a noise ablation. Override with a sub-study if a noise-off eval is wanted. |

### 3.2 Output directory schema

```
results/eval/<run_tag>/<checkpoint_step>/
  metadata.json                   # config snapshot, training step, wall-clock time, JAX version, commit hash
  episodes/
    <ep_idx>.npz                  # compressed numpy: per-step (s, a, info) for the entire episode (NOT just threat windows — see §3.3)
  windows/
    threat_onsets.parquet         # one row per threat-onset event across all episodes
    feature_vectors.parquet       # M7 features, one row per onset event
  online_replay.json              # M1/M2/M5 keys re-computed offline as a sanity cross-check on the online accumulator
```

### 3.3 Trajectory-dump scope (per user directive)

**Threat-window-only dumps** (NOT full-episode) for the M7 path. The user's directive: M7 motif clustering reads K-step windows around threat-onset events; full-episode dumps are unnecessary disk cost.

**However** — the implementation surface in `scripts/eval_rollout.py` should write full-episode `.npz` files under `episodes/`, with `windows/threat_onsets.parquet` indexing into them (episode_idx, window_start_t, window_end_t). Reasons:

1. **Sanity replay**: re-computing M1/M2/M5 offline from full-episode dumps cross-checks the online accumulators caught no NaN-mask, off-by-one, or stage-wipe bug. This is exactly the check that caught the per-tag implementation bug at site 1 (RPPO main) — see [`per_quadrant_and_per_rabbit_logging.md`](../../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md) Verification Report.
2. **Future motif-window expansion**: if v2 wants `K_motif = 10` instead of `7`, it can re-run the offline featurisation without re-running the rollout.
3. **Disk cost is bounded**: 500 steps × ~150 floats per step × 200 episodes × ~50 bytes per float (JSON-equivalent) ≈ 750 MB per checkpoint. Compressed npz ≈ 100–200 MB. Within the existing `results/` budget.

**Trade-off accepted**: ~150 MB per checkpoint is the disk price for sanity-replay + featurisation flexibility. The senior-developer is free to revisit if disk pressure becomes binding.

### 3.4 Computational budget per checkpoint

- **Wall-clock**: 200 episodes × 500 steps × 1 env (deterministic, no parallelism necessary) on one RTX 3090 ≈ 5–10 minutes per checkpoint (rough — see senior-developer plan for actual benchmark).
- **Disk**: ≈ 150 MB per checkpoint compressed.
- **Memory**: trivial — eval is single-env, no replay buffer.

### 3.5 Checkpoint-selection rule

For the v1 demonstration on Round-2.5 Cell A1 + Cell C:
- The **final** saved checkpoint of each run (10 M episodes), to match the analysis window of the original Round-2.5 verdict.
- **Optionally** earlier checkpoints (e.g., the mid-training transient at windows 4–7 in Cell A1) if the senior-developer's plan supports it cheaply. Not load-bearing for v1.

For future studies, the experiment-designer specifies the checkpoint-selection rule per study (e.g., "every 1 M episodes" for learning-curve analyses).

---

## 4. Pre-Registered Analysis Plan

What each measure produces in the paper, and the headline numbers.

### 4.1 Per-measure paper figures

| Measure | Paper figure (v1 demonstration on Round-2.5 + Round-1) | Headline number |
|---|---|---|
| **M1** | Bar chart: `InterruptedFeedingRate_predator` vs. `_rabbit` per agent (Cell A1, Cell C, Round-1). 95% CI = bootstrap over episodes (single-seed for v1; multi-seed in future studies). | The predator-vs-rabbit gap per agent, in percentage points. Hypothesis: < 5pp gap on Cell A1 + Cell C (no class-conditional response under sameProp); the value on Round-1 is the immediate cross-check. |
| **M2** | Bar chart: `BushDiveRate_predator` vs. `_rabbit` per agent. Same CI. | Per-agent `BushDiveRate_predator`. Hypothesis: < 5% on Cell A1 (no opportunity), 5–15% on Cell C (no class-conditional defence). |
| **M5** | Scatter: x = `EatSafeRate`, y = `EatUnderThreatRate`, points = (agent × class); identity line + slope-< -1 line annotated. | Per-agent `EatUnderThreatRatio_predator − EatUnderThreatRatio_rabbit`. Hypothesis: ≈ 0 across all three agents under sameProp. |
| **M7** | Stacked bar chart: motif distribution per agent. Cell A1 / Cell C / Round-1 side-by-side. | Per-agent motif fractions; Cell A1 should be > 80% `ignore`; Cell C should be `flight`-dominant. |

### 4.2 M7 cross-cell motif redistribution check

The headline test for M7 working as intended: **does the motif distribution differ between Cell A1 and Cell C in the way the per-tag distance metric predicts?**

- **Cell A1 prior** (camping): motif distribution dominated by `ignore` (> 80%), with the residual 20% being whatever rare events occur on the few episodes where the agent left BR. If M7 instead shows a balanced distribution across multiple motifs, the **measure is detecting noise** — recheck featurisation, cluster count, and onset-detection logic.

- **Cell C prior** (food-decoupled, bilateral rabbit avoidance): motif distribution should show non-trivial `flight` (40–60%) with `bush_dive` < 5%. Active defence (`bush_dive` ≥ 15%) on either class would be a surprising finding requiring a follow-up — but the §1.1 H₀(M2) is the specific predicate.

- **Cell A1 vs Cell C divergence**: at minimum, Cell A1's `ignore` fraction should exceed Cell C's by ≥ 30 percentage points. If not, M7 is **not discriminating between agents whose other behavioural metrics differ massively**, and v2 needs revised features.

- **Cell A1 vs Round-1 divergence**: similarly, Round-1 should show > 30pp lower `ignore` than Cell A1. If not, M7 is not separating spatially-constrained from spatially-free agents.

Cluster-count selection: fixed `k = 6` for v1 per §2.4. In the analysis doc, report the silhouette score (R4 sanity); if < 0.20 with no justification, drop M7 from the v1 paper figure and re-design the featurisation in v2.

### 4.3 Cross-cell correlation analyses (pre-registered)

To check that the four measures are not redundant:

- **Pearson correlation across episodes** (within-agent): `M1_predator` vs. `M2_predator` vs. `(1 - M5_predator)` (the latter is "risk discounting"; flipped so all three "should be high under threat" point in the same direction). Predicted ρ_pairwise ≈ 0.3–0.7 (correlated but not redundant). ρ > 0.95 on any pair = the two measures are sliding off the same indicator and one can be dropped in v2.

- **M7 motif fraction vs. M1/M2/M5**: per agent, does the agent's `bush_dive` motif fraction correlate with its `BushDiveRate_predator` online metric? (Yes, by construction — but the magnitude tells us how much M7's offline window-detection deviates from the online step-detection. Mismatch > 30% is a wiring-bug signal.)

### 4.4 Per-tag fan-out usage

Per-tag versions of M1, M2, M5 are emitted for every config that has tagged entities. The first analysis step on every agent is a **per-tag sanity check** (analogous to the Round-2.5 per-tag distance check that exposed corner-camping):

- For Cell A1: `InterruptedFeedingRate_predator_TL` vs. `InterruptedFeedingRate_rabbit_TL` (same-corner pair under matched smells). A non-trivial gap here would be the smoking-gun signal for class recognition that the per-tag distance metric did not find.

- For Cell C: `EatUnderThreatRatio_rabbit_TL` vs. `_rabbit_BR` (does rabbit-class avoidance generalise across rabbit corners or is it driven by one)?

---

## 5. Failure-Mode Catalog

Pre-decide ambiguous outcomes. Each entry: diagnostic predicate that fires + next-step recommendation.

### 5.1 M1 — agent never eats while a threat is near

**Symptom**: `Episode/InterruptedFeedingDenominator_<class>` is 0 (or near 0) for ≥ 90% of episodes.

**Reading**: NOT a refutation of the hypothesis. The agent's policy avoids the eating-with-threat-near state altogether — which is itself a class-conditional spatial pattern (e.g., Cell A1 BR-camping). M1 reads as "uninterpretable for this agent" and the analysis doc reports the denominator zero-rate as the headline finding instead.

**Next step**: cross-check with M5 (`EatUnderThreatSafeSteps`) — if the agent has near-zero "under_threat" steps, both M1 and M5 are uninterpretable and the verdict is "the agent's policy keeps it spatially disjoint from threats — class-conditional defence is not testable on this run." Combined with M7's expected `ignore` dominance, this becomes the headline reading for Cell A1.

### 5.2 M2 — agent never enters a bush

**Symptom**: `Episode/BushDiveRate_<class>` is 0 across all episodes AND `Σ_t info['agent_in_bush'][t]` (a separate diagnostic) is 0 across all episodes for the entire eval set.

**Reading**: agent does not use bush cover at all. This refutes H₁(M2) but does not refute the toolkit — it is a meaningful behavioural readout ("this policy ignores cover").

**Next step**: M2's verdict is "no active defence". Report and move on. Cross-check `obs_hides_agent` mask in the env config — if the config has 0 bushes, M2 is uninterpretable structurally and a config-side warning should have fired (see §6).

### 5.3 M5 — episode-windows where no threat ever appears

**Symptom**: `Episode/EatUnderThreatSafeSteps_<class>` is `T_ep` (entire episode is "safe"; denominator on the threat-side is 0).

**Reading**: ratio is undefined (NaN). Same uninterpretability finding as M1. Same cross-check pattern: combine with M1 + M7 to characterise the agent's relationship to threats spatially.

**Next step**: emit NaN, log the denominator, do not impute. The downstream paper figure displays NaN runs as "denominator below threshold (n=X episodes)" annotation rather than dropping them.

### 5.4 M7 — clusters all collapse to one motif

**Symptom**: After clustering, ≥ 90% of windows fall into one cluster. Silhouette score < 0.10.

**Reading**: motif clustering failed to find structure. Either (a) the agent genuinely has a degenerate behavioural repertoire (Cell A1's `ignore` cluster could legitimately swallow > 90% — this is a real finding, not a failure), or (b) the featurisation is too narrow.

**Next step**: cross-check with the **per-feature variance** within each agent's onset windows. If feature-vector variance is > 1.5× the inter-cluster variance, the featurisation is at fault → revise in v2. If variance is matched and 90% of windows really are similar, that IS the agent's repertoire and the figure shows it.

### 5.5 General — measure values change across reruns of the same checkpoint

**Symptom**: Re-running `scripts/eval_rollout.py` on the same checkpoint produces M1/M5 values differing by > 1% across reruns.

**Reading**: non-determinism leak. Either `eval_policy_mode` is not in fact deterministic, OR JAX PRNG / shuffle keys are not seeded, OR per-tag-tuple reordering is non-stable.

**Next step**: senior-developer's bug. Block downstream analysis until reproducibility is verified by re-running a fixed seed twice and confirming bitwise-identical output JSONs.

### 5.6 General — sample-size inadequate

**Symptom**: 95% CIs (bootstrap over the 200 eval episodes) on the predator-vs-rabbit gap straddle zero by > the H₁ effect-size threshold.

**Reading**: cannot reject H₀ with this sample size. Possible fixes: (a) increase `eval_n_episodes` (cheap; one config knob); (b) acknowledge the verdict is provisional and route the experiment to a multi-seed re-run.

**Next step**: report inconclusive with the CI width as the headline number. Do not claim H₁ refuted unless the CI is fully outside the H₁ region.

---

## 6. New Config Keys (No-Fallback-Defaults Rule)

All new YAML keys live under a new top-level `behavior_measures:` block. Every key uses `config.get_mandatory(...)` semantics — missing key raises `ValueError` at config load. The new keys are NOT yet read by `src/utils/config.py` or any loader; this is **a schema-affecting change** that must route through `senior-developer` → `developer` BEFORE any config that uses these keys is launched. See §7 for the implementation surface.

```yaml
behavior_measures:
  enabled: true                                # bool. If false, no online M1/M2/M5 keys emitted; offline still works.
  cue_radius: 3.0                              # float. R in cells. Mandatory.
  obs_window: 5                                # int. K in steps. Mandatory.

  # Online M1, M2, M5 — same R, K used for all three (locked here for v1).
  # Per-class fan-out always emitted; per-tag fan-out emitted iff tags exist.

  # Offline M7 — eval-rollout protocol
  eval_n_episodes: 200                         # int. Mandatory.
  eval_seeds: [1000, 1001, ..., 1199]          # list[int], length must equal eval_n_episodes. Mandatory.
                                               # (Author 200 entries explicitly; no range-shorthand.)
  eval_policy_mode: deterministic              # enum: deterministic | stochastic. Mandatory.
  eval_max_steps: 500                          # int. Should match env max_steps. Mandatory.
  eval_obs_noise: training                     # enum: training | zero | custom. Mandatory.
                                               # 'training' = match training-time noise; 'zero' = noise-off ablation.

  # Offline M7 — motif clustering
  motif_window_K: 7                            # int. K_motif (slightly longer than online K).  Mandatory.
  motif_features:                              # list[str]. The 10 v1 features (locked names).
    - net_displacement
    - path_length
    - threat_distance_change_rate
    - min_threat_distance
    - bush_occupancy_fraction
    - eat_events_per_window
    - action_entropy
    - mode_action_fraction
    - stay_in_place_fraction
    - drive_injury_change
  motif_kmeans_k: 6                            # int. Fixed k for v1. Mandatory.
  motif_kmeans_seed: 42                        # int. Reproducibility lock. Mandatory.
  motif_standardise: zscore_pooled             # enum: zscore_pooled | zscore_per_agent | none. Mandatory.

  # Output paths
  eval_output_root: results/eval               # str. Mandatory.
```

**Required schema-loader work** (routes through senior-developer + developer per CLAUDE.md):

| Key | Type | Validator |
|---|---|---|
| `behavior_measures.enabled` | bool | required |
| `behavior_measures.cue_radius` | float | > 0 |
| `behavior_measures.obs_window` | int | ≥ 1 |
| `behavior_measures.eval_n_episodes` | int | ≥ 1 |
| `behavior_measures.eval_seeds` | list[int] | `len == eval_n_episodes`; all unique |
| `behavior_measures.eval_policy_mode` | str | in `{deterministic, stochastic}` |
| `behavior_measures.eval_max_steps` | int | ≥ 1 |
| `behavior_measures.eval_obs_noise` | str | in `{training, zero, custom}` |
| `behavior_measures.motif_window_K` | int | ≥ 1 |
| `behavior_measures.motif_features` | list[str] | non-empty; subset of feature names supported by `scripts/motif_cluster.py` |
| `behavior_measures.motif_kmeans_k` | int | ≥ 2 |
| `behavior_measures.motif_kmeans_seed` | int | required |
| `behavior_measures.motif_standardise` | str | in `{zscore_pooled, zscore_per_agent, none}` |
| `behavior_measures.eval_output_root` | str | required |

**Bush-presence sanity check** (env-config-auditor's domain): when `behavior_measures.enabled: true`, the auditor must check that `environment.obstacles` contains at least one entry with `hides_agent: true`. If not, M2 is uninterpretable and the auditor should fail the config or warn loudly.

**Per-class fan-out is always on** when `enabled: true` — it is not gated by a separate key. The per-tag fan-out follows the same convention as the existing per-tag distance keys (emitted iff tags exist).

---

## 7. Implementation Surface (high-level — `senior-developer` writes the actual plan)

This section is descriptive. The senior-developer authors the implementation plan; the experiment-designer does not specify line numbers or actual code.

### 7.1 New env hook (single line in `core.py`)

`info['agent_in_bush']` must be surfaced from inside `jax_step`. The boolean computation already exists at `src/environment/core.py:149-152` for the predator state machine; either lift the existing `agent_hidden` value into the outer scope or recompute it in the `info`-dict region of the function. One line of code in the surfacing pattern of `info['event_collided']`. **Senior-developer's call** on whether to refactor or recompute.

### 7.2 New `train.py` accumulators (online; M1, M2, M5)

Each measure adds 3–5 integer accumulators per env per class, mirroring the per-tag-distance pattern at the same five sites (RPPO main, PPO branch B, Dreamer branches A/B/C). The K-step look-ahead for M1 and M2 needs a **per-env circular buffer of length K** holding the most recent `info['ate_food']` (M1) and `info['agent_in_bush']` (M2) values; the candidate event at step `t` is resolved at step `t + K` when the buffer's offset-0 element is read.

The senior-developer must:
- audit which of the 5 accumulator sites need the wiring (per the per-tag plan, all 5 do, but the K-step look-ahead may interact with the leftover-after-done logic in DreamerV3 batch site (Site 2) — see deviation #2 in the per-tag verification report);
- decide whether to vectorise the K-buffer in JAX or carry a numpy buffer per env (numpy is simpler; matches the existing accumulator pattern);
- handle episode boundaries (when an episode ends mid-window, the candidate event with no resolved look-ahead is dropped from the denominator);
- handle `enabled: false` cleanly (zero overhead path).

### 7.3 New script: `scripts/eval_rollout.py`

Loads a frozen agent checkpoint + its config, instantiates a single-env (or small-batch for parallelism) eval environment with `behavior_measures.eval_seeds`, rolls out `eval_n_episodes` deterministic episodes, dumps per-step `(state, action, info)` to `results/eval/<run_tag>/<checkpoint_step>/episodes/`. Also emits `windows/threat_onsets.parquet` indexing into the per-episode dumps. Re-computes M1/M2/M5 offline as a sanity cross-check (`online_replay.json`).

Senior-developer's call on:
- checkpoint-loading mechanics (how the agent's frozen weights + RNG state load from `results/<run_tag>/checkpoints/`);
- per-episode dump format (npz vs. parquet — npz recommended for `(state, action, info)` per-step ndarrays; parquet for the per-event window index);
- noise-schedule replay (eval at `eval_obs_noise: training` must match the noise schedule that was active at the loaded checkpoint's training step).

### 7.4 New script: `scripts/motif_cluster.py`

Reads `windows/threat_onsets.parquet` + per-episode `episodes/*.npz`, computes the 10 features in §2.4, standardises per `motif_standardise`, runs k-means with `motif_kmeans_k` clusters and `motif_kmeans_seed`, writes `motif_distribution.json` + `cluster_centroids.npy` + `silhouette.json` + `exemplars.json` per agent.

Senior-developer's call on:
- numpy / sklearn vs. JAX clustering (sklearn is simpler; clustering is offline + small);
- exemplar-selection rule (5 nearest-centroid windows per cluster is the v1 default).

### 7.5 Schema loader changes (routes through senior-developer + developer)

The new `behavior_measures:` block requires `src/utils/config.py` (or wherever mandatory keys are read) to register the new keys with `config.get_mandatory(...)`. **Until this loader work lands, no config below this design's `## 6` schema is launchable.** The experiment-designer's configs are written *after* the schema work completes — see §10.

### 7.6 Test plan (senior-developer plans; developer implements)

Sketched here so the senior-developer's plan can absorb it:

- **Unit T1**: M1 accumulator on a fixture trajectory with a known number of candidate / interrupted events → exact match to hand-computed value.
- **Unit T2**: M2 accumulator on a fixture trajectory with a known threat-onset → bush-dive sequence → exact match.
- **Unit T3**: M5 accumulator on a 100-step fixture trajectory with hand-counted threat / safe / eat steps → exact ratio match.
- **Unit T4**: M7 featuriser on a fixture window → all 10 features within 1e-6 of hand-computed values.
- **Unit T5**: Schema validator rejects missing `cue_radius` with `ValueError` mentioning the missing key.
- **Integration T6**: 100-iteration RPPO smoke run on `02-sameProp_R2_passivePredator.yaml` with `behavior_measures.enabled: true` — confirm all keys appear in WandB with finite values + reasonable orders of magnitude.
- **Integration T7**: `scripts/eval_rollout.py` on the saved Round-2.5 Cell C checkpoint produces a `results/eval/.../bdnfc0lu/.../` tree with the documented schema; `online_replay.json` matches the WandB-logged `Episode/EatUnderThreatRatio_predator` to within 1% (cross-check).
- **Integration T8**: `scripts/motif_cluster.py` on the same eval-rollout output produces a `motif_distribution.json` with all 6 motifs present (or fewer + a documented reason).

---

## 8. Cross-Links to Dependencies and Downstream

### 8.1 Upstream (read-only — defines what this design implements)

| Doc | Role | Key contribution |
|---|---|---|
| [`docs/project/ideas/20260510_behavior_measure_toolkit.md`](../../../project/ideas/20260510_behavior_measure_toolkit.md) | Source of truth for measure definitions + biological grounding | The 8-candidate menu; this design selects M1/M2/M5/M7. |
| [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md) | Architectural precedent | The 5-site `train.py` accumulator pattern this design extends; the per-tag fan-out idiom; the verification-report bug-catch (RPPO Site 1) that motivates §3.3's offline sanity-replay. |
| Insight `20260510_2237_sameprop_round25_no_class_avoidance` | The motivation | The mean-distance-dissolves-dynamics finding that this toolkit is the answer to. |

### 8.2 Downstream (this toolkit feeds these)

| Doc / future doc | Adoption pattern |
|---|---|
| [`docs/experiments/active/hypervigilance/sameprop_round25_design.md`](../hypervigilance/sameprop_round25_design.md) | First user — the v1 demonstration runs the four measures on saved Cell A1 (`nm8gn7y2`) and Cell C (`bdnfc0lu`) checkpoints; results land in §9–§11 of that doc as a follow-on analysis. |
| Future Round 2.6 design (queued, Cell C seed 44) | Should adopt M1/M2/M5/M7 in its analysis plan. |
| Future Round 3 design (food in all 4 quadrants) | Should adopt as default measures alongside per-tag distances. |
| NMN-vs-baseline comparison study | Headline figure draws from M2 (`bush_dive_rate`) and M7 (motif redistribution) per the postdoc's §5 figure-mapping table. |

### 8.3 Pending agent hand-offs

| Agent | Next deliverable |
|---|---|
| **`senior-developer`** | Authors the implementation plan: env hook (§7.1), `train.py` accumulators (§7.2), `scripts/eval_rollout.py` (§7.3), `scripts/motif_cluster.py` (§7.4), schema loader (§7.5), test plan (§7.6). |
| **`developer`** | Implements the plan after senior-developer review. |
| **`experiment-designer` (this agent, second pass)** | After implementation lands: writes the actual `configs/experiment/behavior_measures/*.yaml` configs that exercise the new schema; sends them through `env-config-auditor`. |
| **`env-config-auditor`** | Audits each config for: `obstacles` contains a `hides_agent: true` row when `behavior_measures.enabled: true`; `eval_seeds` length matches `eval_n_episodes`; no tag-collision with existing entity tags. |
| **`experiment-analyzer`** | After eval-rollouts complete: writes the v1 demonstration paper figures + verdicts on H₀(M1), H₀(M2), H₀(M5), and the M7 cluster-redistribution check, against Cell A1 + Cell C + Round-1 baselines. |

---

## 9. Designer's Predicted Outcomes

For each of the 4 measures, applied to the saved Round-2.5 checkpoints, here is the prior. All values are point estimates; CI widths come from the analysis stage.

### 9.1 Round-2.5 Cell A1 (`nm8gn7y2`, seed 43, corner-camping)

| Measure | Predicted value | Reading if confirmed | Reading if refuted |
|---|---|---|---|
| `M1_predator` | NaN OR ≤ 5% | Camping signature: agent never eats while a predator is in radius. Cross-tabulates with `M5` denominator-near-zero for the "under_threat" branch. | Surprise — would mean the agent does occasionally eat near the predator; revisit camping verdict. |
| `M1_rabbit` | NaN OR ≤ 5% | Same camping reading — agent rarely eats while a rabbit is in radius (the BR rabbit is static and the agent has reached an equilibrium with it). | Surprise — agent encounters the BR rabbit non-trivially. |
| `M2_predator` | NaN (`Σ_t enter_t ≈ 0`) | Predator never enters BR radius; no opportunity to dive. | Predator-onset events are non-zero — would refute the "agent never visits TL" reading. |
| `M2_rabbit` | NaN OR ≤ 5% | BR rabbit largely static; few "enters radius" events from a kinematic-equilibrium standpoint. | If the agent has many BR-rabbit-enter events, the equilibrium is dynamic. |
| `M5_predator` | NaN (denominator zero) | R3 fires; M5 uninterpretable for predator. Headline finding: "agent's policy keeps it disjoint from the predator class spatially, so risk-discounting is structurally untestable." | Surprise — would mean predator does enter radius; revisit camping. |
| `M5_rabbit` | 0.5–1.0 | Some risk-discounting on the static BR rabbit, but the value is uninformative because the agent and rabbit are in equilibrium. | — |
| **M7 motif distribution** | `ignore` ≥ 80%, residual is mostly `flight`-shaped | Confirms the camping policy at the trajectory level. | Multi-modal distribution would mean the agent's behaviour is more varied than the per-tag distance metric showed. |

**Headline reading if priors hold**: "Cell A1's behavioural repertoire is overwhelmingly `ignore` (i.e., spatial-disengagement); class-conditional defensive measures are structurally untestable on this run because the policy never produces the candidate events the measures count." This is itself a **valid scientific finding** — the toolkit's purpose is to make this readable, not to report a positive H₁.

### 9.2 Round-2.5 Cell C (`bdnfc0lu`, seed 42, food-decoupled, bilateral rabbit avoidance)

| Measure | Predicted value | Reading if confirmed |
|---|---|---|
| `M1_predator` | 10–25% | Per the verdict: no class-conditional response. |
| `M1_rabbit` | 10–25% (within 5pp of predator) | Same. The toolkit fails to find a class-conditional gap → matches the per-tag distance verdict. |
| `M2_predator` | 5–15% | Some bush-cover use, not class-conditional. |
| `M2_rabbit` | 5–15% (within 5pp of predator) | Same. |
| `M5_predator` | 0.6–0.9 | Mild risk discounting around the predator. |
| `M5_rabbit` | 0.6–0.9 | Mild discounting around rabbits too — bilateral avoidance reads here. |
| `M5_predator − M5_rabbit` | within ±0.10 | No class-conditional risk discounting. |
| **M7 motif distribution** | `flight` 40–60%, `ignore` 30–50%, `bush_dive` < 5%, `approach` < 5% | The bilateral rabbit-avoidance verdict reads as "flight on both classes" at the motif level. |

**Headline reading if priors hold**: "Cell C's behavioural repertoire is `flight`-dominated, bilaterally — confirming the per-tag distance metric's verdict that the agent does not class-condition. The motif distribution is comparable across `predator-onset` and `rabbit-onset` partitions of the same agent's data." That last clause requires partitioning the M7 windows by triggering class — which the threat_onsets.parquet schema in §3.2 already supports.

### 9.3 Round-1 baseline (e.g., `rg5nl1ov`, seed 42, R1 sameProp config)

| Measure | Predicted value | Reading if confirmed |
|---|---|---|
| `M1_predator − M1_rabbit` | ±10pp | Round-1's apparent +0.6-cell rabbit-vs-predator distance gap was attributed to the food confound; M1 should show a similarly small gap. |
| `M2_predator` | 5–20% | Less spatially constrained than R2.5; some bush-cover use. |
| `M2_rabbit` | 0–10% | If R1 has any class-conditional active defence, it should show here — but the prior is small. |
| `M5_predator` | 0.6–0.9 | — |
| `M5_predator − M5_rabbit` | ±0.20 | — |
| **M7 motif distribution** | `ignore` 40–60%, `flight` 20–35%, `bush_dive` 5–15%, `approach` 5–15% | Richer repertoire than R2.5 (no quadrant constraint); but no strong class-asymmetric prediction. |

### 9.4 What would surprise the designer

- **A class-conditional `M2` gap on Round-1** (predator BushDiveRate > rabbit BushDiveRate by ≥ 15pp). Would re-open the Round-1 verdict; would partially revive the original sameProp survey's class-recognition reading.
- **A `bush_dive`-dominant motif on any of the three agents**. Would mean active class-conditional defence is the dominant policy, which the per-tag distance verdict refuted. Would require a follow-up investigation for measurement bug vs. genuine signal.
- **`M5_predator` < 0.4 on Cell C**. Would mean strong predator-specific risk discounting that the aggregated per-tag distance missed. Possible because M5 is per-step and dist is averaged. Would be the most interesting positive surprise.

---

## 10. Launchable Status

This design is the **protocol** — no training launch is gated by it directly. Three downstream gates must clear before the v1 demonstration runs.

- ⏸ **Gate 1 — Schema loader** (`senior-developer` + `developer`): the `behavior_measures:` block must be readable by `config.get_mandatory(...)` in `src/utils/config.py` (or wherever the loader lives). Until then, configs that use the new keys cannot launch.
- ⏸ **Gate 2 — Online accumulators** (`senior-developer` + `developer`): M1/M2/M5 accumulators wired into the 5 `train.py` sites + `info['agent_in_bush']` env hook surfaced. Until then, online keys are not emitted during training.
- ⏸ **Gate 3 — Eval-rollout + motif scripts** (`senior-developer` + `developer`): `scripts/eval_rollout.py` + `scripts/motif_cluster.py` shipped. Until then, M7 and offline sanity-replay don't run.
- ⏸ **Gate 4 — User authorisation** after the senior-developer implementation plan lands (per CLAUDE.md "experiment-designer routes schema-affecting changes through the human").
- ⏸ **Gate 5 — `experiment-designer` (this agent, second pass)** writes the actual `configs/experiment/behavior_measures/*.yaml` configs that exercise the new schema.
- ⏸ **Gate 6 — `env-config-auditor`** signs off on each config before launch.

The first user of the toolkit (the v1 demonstration on saved Round-2.5 checkpoints) does **not** require new training launches — only `scripts/eval_rollout.py` runs. So Gates 1, 3, 5, 6 are load-bearing for the v1 demonstration; Gate 2 is load-bearing for *future* trainings that emit the online keys live.

### 10.1 Launch Manifest (deferred to v1 demonstration design)

This protocol does not launch any training itself. The first training that adopts these measures will be a future Round 2.6 / Round 3 / NMN design that fills in its own Launch Manifest using the new measure keys. No table here.

The v1 demonstration **eval-rollout** has its own results-dir schema (§3.2); its outputs are not WandB runs, so they don't enter the diary's `training-start`/`training-done` rows. They show up as `experiment-analyzer` analyses against existing WandB run IDs (`nm8gn7y2`, `bdnfc0lu`).

---

## Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-11 | Initial design — protocol for M1/M2/M5/M7 with R = 3.0, K = 5, eval N = 200, fixed k = 6 motif clustering | experiment-designer |
