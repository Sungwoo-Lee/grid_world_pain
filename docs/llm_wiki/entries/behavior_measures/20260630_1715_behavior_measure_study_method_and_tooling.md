---
id: 20260630_1715_behavior_measure_study_method_and_tooling
date: 2026-06-30
time: 17:15
folder: behavior_measures
tags: [design, decision, meta]
summary: "Interoceptive behavior-measure study: discover MEASURES of foraging/avoidance vs nutrition/injury by trajectory-first→metric-second on a frozen in-distribution agent; configs split core/explore; reusable scripts/behavior_measures/ heatmap tooling; deterministic probe → small initial-state jitter for statistics."
related: ["20260616_1514_experimental_env_as_behavior_platform", "20260622_1745_frozen_probe_eval_match_sensory_renderer", "20260624_0517_indist_random_init_reverses_hypervig", "20260630_1716_foraging_hunger_timing_fixed_opening", "20260630_1717_avoidance_reflex_needs_motion_and_olfaction", "20260630_1718_cover_use_late_emerging_run_vs_hide"]
session_origin: claude_code
session_label: "behavior-measure study build (foraging + avoidance probes)"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# Behavior-measure study: method + config tiering + reusable heatmap tooling

## Key conclusion
A long-running study to **discover the right measures** of an agent's foraging and avoidance behavior as a function of its internal state (nutrition, injury). The measure is the deliverable, not an input. Method is fixed: **trajectory-first → metric-second** — build a controlled probe, read the rollout step-by-step (the `trajectory-story` skill), let the behavioral signal reveal itself, then crystallize it into an episode-level metric grounded in observed behavior. Probes load a **frozen, in-distribution** checkpoint (no retraining).

## Evidence, measurements, facts
- Subject model: `20260627-015427_rppo_basic05_randinit_n112`, ckpt `models/8900007` (~8.9M). Trained on random nutrition[0,100] / injury[0,100], decay_power 2.0, 10×10 → **every probe start-state is in-distribution** (removes the off-distribution artifact of [[20260624_0517_indist_random_init_reverses_hypervig]]).
- Anchor doc (re-readable spine): `docs/experiments/active/behavior_measures/interoceptive_behavior_measure_study.md` — holds method, subject, phase plan, findings log, and the canonical "Measures & definitions" table.
- Config tiering (user decision): `configs/environment/experiment/behavior_probes/` split into **`core/`** (canonical, maintained: forage_nutrition, forage_direction, avoidance) and **`explore/`** (iterative/superseded: conflict, hypervigilance, nutrition_sweep, nutrition_sweep_d2). A probe graduates explore→core only once it produces a trusted measure; promotion = `git mv` + fix `extends:` paths + doc refs (stale config paths silently fall back to defaults).
- Reusable tooling: new experiment-specific subfolder **`scripts/behavior_measures/`** — `avoidance_stats_heatmap.py` (auto-discovers config subdirs, computes per-episode measures, aggregates mean±std, writes CSV + heatmap) and `_heatmap_style.py` (journal-style renderer: card cells, per-criterion colour — sequential for magnitude, diverging-at-0 for signed metrics — column-group headers, PNG 300dpi + PDF vector).
- Statistics method: the eval policy is **deterministic** (argmax), so seeds alone don't vary behavior. To get distributions, inject **small initial-state jitter** (injury/nutrition bands) via stat-variant configs and run many seeds; record `.rec.gz` but skip video render.
- Eval recordings carry per-step `agent_pos`, `animal_pos`, `obs_pos`(=bush cell), `nutrition`, `injury_level`; measures are computed from these.

## Decisions and actions
- Phase plan: 1a foraging vs nutrition (done), 1b naturalistic check (pending), 2 injury vs avoidance (avoidance matrix done), later nutrition×injury map.
- Adopted percentage-based, referent-explicit metric names (no "fraction"/"proximity"/"pursuit" jargon) — see the foraging/avoidance insights.

## Open questions and follow-ups
- Phase 1b (naturalistic / random food placement) not yet run.
- Foraging-measure heatmap script not yet written (only avoidance).

## References
- Builds on the behavior-platform reframe [[20260616_1514_experimental_env_as_behavior_platform]] and the frozen-probe eval discipline [[20260622_1745_frozen_probe_eval_match_sensory_renderer]].
- Findings produced under this study: [[20260630_1716_foraging_hunger_timing_fixed_opening]], [[20260630_1717_avoidance_reflex_needs_motion_and_olfaction]], [[20260630_1718_cover_use_late_emerging_run_vs_hide]].
- **Why a new folder**: closest existing is `hypervigilance` (threat-specific behavior probes), but this study measures GENERAL foraging + avoidance behavior (hypervigilance is one now-superseded sub-thread), has its own anchor doc, config tier, and tooling, and will keep generating foraging/avoidance/recovery findings. Definition lock: `Behavior-measure platform & probes`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260630_1715.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260630_1716_foraging_hunger_timing_fixed_opening]] (behavior_measures, 2026-06-30) — Phase-1a foraging probe: hunger changes the TIMING of foraging, not the path — f
- [[20260630_1717_avoidance_reflex_needs_motion_and_olfaction]] (behavior_measures, 2026-06-30) — Avoidance probe (predator/rabbit + bush): the flee-to-cover reflex requires BOTH
- [[20260630_1718_cover_use_late_emerging_run_vs_hide]] (behavior_measures, 2026-06-30) — Cross-model: an EARLY model (randpred, 1.5M) RUNS/kites around the grid perimete
- [[20260704_2012_noise_matched_frozen_probe]] (behavior_measures, 2026-07-04) — A noise-trained agent must be probed with a config carrying its EXACT training p
- [[20260704_2013_probe_rerun_stale_checkpoint_contamination]] (behavior_measures, 2026-07-04) — Re-running an avoidance-probe sweep on an ADVANCED checkpoint writes the new rec
- [[20260704_2014_deterministic_probe_significance_inflates]] (behavior_measures, 2026-07-04) — A near-deterministic behavior probe (tiny within-model variance) makes p-values 
- [[20260710_1634_behavior_probe_eval_speed_parallel_batched_fdsafe]] (behavior_measures, 2026-07-10) — The avoidance behavior-probe eval was sped ~8.8x (measured A/B/C). The dominant 
<!-- END BACKLINKS -->
