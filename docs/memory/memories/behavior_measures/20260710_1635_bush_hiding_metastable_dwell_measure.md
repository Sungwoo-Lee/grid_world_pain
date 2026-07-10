---
id: 20260710_1635_bush_hiding_metastable_dwell_measure
date: 2026-07-10
time: 16:35
folder: behavior_measures
tags: [learned_lesson, refutation, decision, meta, design]
summary: "Bush-hiding on the hard-predator task is a metastable/intermittent mode (policy flickers run<->hide across checkpoints), not a stable phase; coarse 1M sampling ALIASES it into a false spike. Use dwell% (time-in-bush), not bush_use% (binary entry, overstates). The config-system era does NOT explain the behavior: predator difficulty + training phase do. Extended 128-env training modestly raises dwell but stays bistable."
related: ["20260630_1717_avoidance_reflex_needs_motion_and_olfaction", "20260630_1718_cover_use_late_emerging_run_vs_hide", "20260704_2014_deterministic_probe_significance_inflates"]
session_origin: claude_code
session_label: "128-env relaunch + config-owns-values + bush-hiding checkpoint probes"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# Bush-hiding is metastable, not a phase; measure it by dwell; config-era doesn't explain it

## Key conclusion
Bush-hiding on the hard-predator random-init task is a **metastable / intermittent** behavior: the policy flickers between perimeter-running and bush-camping from checkpoint to checkpoint (both survive), never committing. It is NOT a stable "phase," and a single-checkpoint read misleads. Coarse 1M-spaced checkpoint sampling ALIASES the intermittency into a false single-checkpoint "spike"; only fine (~0.1M) sampling reveals the true pattern. The honest metric is **dwell% (fraction of steps ON the bush)** — `bush_use%` (binary "ever entered") over-reports hiding because it counts one-step pass-throughs. Crucially, the config-system update did NOT change behavior: at matched predator difficulty AND training phase, old (16-env-era) and new (128-env) agents behave identically. Behavior is governed by **predator difficulty** (easy → stable hiding; hard → transient) and **training phase**, not the config era.

## Evidence, measurements, facts
- b03 full-history scan (16-env, 100 ckpts @0.1M): strong-camp checkpoints scattered across training (0.4M, 4.4–5.5M, 7.4–7.6M, dense 8.2–8.9M, 9.2–9.9M); 18/99 ckpts hide, mean bush_use 21%. The earlier "isolated 8.5M spike" was a 1M-grid aliasing artifact — the fine scan first showed an 8.2–8.9M window, then multiple windows elsewhere.
- **dwell vs entry**: mean bush_use 21% but mean dwell only ~8%; three "entered" checkpoints (4.4/5.1/5.5M) had dwell ~2% (pass-through, not hiding). By dwell: ~85% of ckpts run (<20%), ~11% strong-camp (≥40%).
- **Config-era refutation** (matched comparison from saved configs): `v1_hiding` (old, easy predator detect 5/stamina 30, at 10M) hides 100%; `b03_randinit` (new, HARD predator detect [1,7]/stamina [30,150], at 10M) kites 0%; `b05_alcomb` (old, the SAME hard config as b03, probed at 4.6M) hid — because caught early. So easy predator → stable hiding at depth; hard predator → transient hiding.
- **Extended 128-env training** (b03, 259 ckpts, 0→25.8M): dwell PRE-10M mean 13.1% → POST-10M 19.5%; strong-camp share 20% → 27%; peak dwell 73% → 82%. Upward tendency, still bistable.
- **Measure caveats**: `avoidance_stats_heatmap.episode_measures` takes the bush as `obs_pos[0]` (first obstacle — fine for one bush, would undercount multi-bush) and requires exact-cell occupancy (adjacent-to-bush does not count).
- **Stale-video red herring**: regenerating the 10M heatmap WITHOUT re-rendering videos left an 8.5M bush-hiding video next to a 10M 0%-bush table → looked like a "measure bug." Ground-truth trajectory at 10M showed perimeter-running (0/30 episodes entered the bush) — the measure was correct; it was a checkpoint mismatch. Lesson: keep videos and tables on the same checkpoint.

## Decisions and actions
- Switched primary reporting from `bush_use%` to `dwell%`.
- Generated `dwell_history.png` checkpoint-evolution curves for the current 128-env b02/b03/b04/b05 runs (predator condition, 30 eps/ckpt).
- Reframed the research question from "did the config break hiding" to "what makes hiding the CONVERGED strategy vs a transient one" (a predator-difficulty sweep).

## Open questions and follow-ups
- Does the extended training (03/04/05 → 100M) eventually stabilize camping, or stay bistable? Re-run the dwell scan later.
- Consider a "longest consecutive on-bush run" (camping-bout length) measure — sharper than total dwell for "how long did it stay."

## References
- `results/eval/avoidance/b03_128env_dwell_history/` (+ b02/b04/b05 equivalents).
- Refines [[20260630_1718_cover_use_late_emerging_run_vs_hide]] (late-emerging cover-seeking → is actually intermittent/metastable, not monotonic emergence).
- [[20260704_2014_deterministic_probe_significance_inflates]] (deterministic-probe stats), [[20260630_1717_avoidance_reflex_needs_motion_and_olfaction]] (avoidance mechanism).
- Raw conversation: synced via `./sync-agent-data.sh claude push` (session 4efbe660-28c2-4643-b231-d3c6d2635b5a).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
