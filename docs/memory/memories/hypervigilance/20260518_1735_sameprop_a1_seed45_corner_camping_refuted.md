---
id: 20260518_1735_sameprop_a1_seed45_corner_camping_refuted
date: 2026-05-18
time: "17:35"
folder: hypervigilance
tags: [hypervigilance, refutation, learned_lesson, decision]
summary: "R2.6 Cell A1 seed 45 finished 10M ep on n102 with Episode/Steps=98/500 — H₁(A1-stable, corner-camping is a generic basin) clearly refuted. Agent learned to avoid the TL corner where both predator and TL-rabbit live, but then starved (Term_Starvation=0.84, FoodEaten=1.77/ep). This is a THIRD A1 policy regime, distinct from seed 43's corner-camping (R2.5, 486/500 steps) and from the seed-45 partial mid-training reading on n106 (~491/500 'stay-and-eat'). A1's spatial-level class-blindness survives (per-tag Δ_TL = +0.04 cells), but corner-camping itself is seed-specific."
related: ["20260510_2237_sameprop_round25_no_class_avoidance", "20260512_1428_sameprop_class_discriminating_defence_event_level", "20260518_1736_sameprop_c_seed44_directional_replication", "20260518_1737_wandb_post_crash_frozen_state_misread"]
session_origin: claude_code
session_label: "hypervigilance Round 2.6 re-launch + check"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/d79a0d50-3ac8-4fa2-9e7e-6a6437d6b059.jsonl
raw_completeness: full
---

# R2.6 Cell A1 seed 45 — corner-camping is seed-specific; the agent collapsed to a third regime (avoid-TL-and-starve)

## Key conclusion

The Round 2.6 seed-lock for Cell A1 (passivePredator) finished its full 10 M-episode budget on n102 cuda:0 and clearly refuted the pre-registered H₁(A1-stable) predicate, which required `Episode/Steps ≥ 470/500` for the corner-camping policy to count as a generic basin. The observed value is `Episode/Steps = 97.85/500`, 4–5× below the bar. Term_Starvation dominated (0.84) with very low predator contact (PredatorHits = 0.034) — the agent learned to **avoid the entire TL corner** (where both the patrolling predator and the TL-rabbit live) and forage near the BR rabbit, but couldn't sustain enough food intake (FoodEaten = 1.77/ep, MeanDistFood = 2.92) to survive. The per-tag distance check shows the agent is **still spatially class-blind** within the TL corner (MeanDistPredator_TL = 7.61 vs MeanDistRabbit_TL = 7.56; Δ_TL = +0.04 cells, well below the ±0.30 threshold), so the R2.5 spatial-class-blindness conclusion survives. What does NOT survive is the *specific* corner-camping policy: seed 43 (R2.5) found it, seed 45 (R2.6) didn't. The Cell A1 verdict is now: under matched smells, the agent reliably ends up class-blind in space, but **which class-blind policy it lands in is seed-specific** — corner-camp at seed 43, avoid-TL-and-starve at seed 45.

## Evidence, measurements, facts

- **Run identity**: WandB `m5h4m8dl`, name `hypervigilance-round26-A1-seed45_n102_gpu0_relaunch`. Launched 2026-05-16T13:21:41 KST; finished 47.75 h later, `state=finished`, `_step = 10,000,063`.
- **Pre-registered H₁(A1-stable)** (from `sameprop_round26_design.md` §1): `Episode/Steps ≥ 470/500` AND per-tag `|Δ_TL| ≤ 0.30 cells` AND `M2_BushDiveRate_*` near zero for both classes (M2_predator < 0.10 AND M2_rabbit < 0.10).
- **Observed**:
  - `Episode/Steps = 97.85/500` → **refutes** the Steps criterion at ~4.8× margin.
  - per-tag `MeanDistPredator_TL = 7.6052` vs `MeanDistRabbit_TL = 7.5606`; `Δ_TL = +0.04 cells` (~7× under the ±0.30 threshold) → spatial class-blindness criterion ✓ at convergence-ish (with the caveat that "convergence" here is into a starvation policy, not a survival policy).
  - `BushDiveRate_predator = 0.27` and `BushDiveRate_rabbit = 0.06` — both above the < 0.10 corner-camp signature. Denominator for predator is tiny (0.23 events/ep) because the agent almost never has the predator within R=3 (it stays away from TL); for rabbit denominator = 6.01 (sees BR rabbit often). The "rates" on tiny denominators are noisy and not interpretable as M2 in the camping-policy sense — they reflect a different policy that didn't satisfy the criterion's intent.
- **Termination breakdown**: `Term_Starvation = 0.84`, `Term_Injury = 0.16`, `Term_MaxSteps = 0.00`, `Term_Overeating = 0.00`. The agent never survives to the step cap; deaths are dominated by starvation, with some injury from HidingPredator (DangerHits = 0.50, DamageDanger = 15.1) and the BR rabbit (RabbitHits = 2.40 — the agent walks into the BR rabbit while foraging).
- **Reward** = −203.2 (vs ~−241.4 max possible negative; lots of damage but truncated by starvation rather than getting hit by the patrol predator).
- **Spatial picture**: MeanDistFood = 2.92 (close-ish to food), MeanDistRabbit_BR = 3.88 (forages near BR), MeanDistPredator_TL = 7.61, MeanDistRabbit_TL = 7.56, MeanDistHidingPredator = 2.74 (close to rock-perched HidingPredators — takes danger hits). MeanDistPredator (aggregated) = 7.61 = MeanDistPredator_TL because the patrol predator is restricted to TL only.
- **Comparison to seed 43 (R2.5, `nm8gn7y2`)**:
  - Steps: 97.85 (s45) vs 486.0 (s43) — a 5× drop in survival.
  - Per-tag Δ_TL: +0.04 (s45) vs +0.004 (s43) — both essentially zero, **spatial class-blindness reproduces**.
  - Policy regime: s45 = avoid-TL-and-starve; s43 = corner-camp-BR-and-survive.
- **Comparison to seed 45 partial on n106 (`k08v38af`, crashed @8.7% of budget)**: the prior re-summary at 2026-05-13 14:20 described seed 45 as a "stay-and-eat-and-take-hits" policy with ep_len ~491. That reading was from training-time mid-trajectory and presumably reflected the agent's early exploration; by 10 M ep (final convergence), the policy had collapsed to avoid-TL-and-starve. So **three different policy regimes have now been observed under the Cell A1 config**:
  1. Seed 43 final (R2.5): corner-camp BR, 486/500.
  2. Seed 45 partial @8.7% (R2.6 on n106): stay-and-eat, ~491/500 (transient — did not survive to convergence; the run crashed before we could see whether it would have stayed in this regime).
  3. Seed 45 final (R2.6 on n102): avoid-TL-and-starve, 98/500.

## Decisions and actions

- **A1 verdict updated to**: *class-blindness in space is robust under sameProp; the specific high-survival policy is seed-specific and fragile.* This is what the user warned about pre-launch ("single-seed nulls are weaker than single-seed positives") — empirically confirmed.
- **Don't supersede [[20260510_2237_sameprop_round25_no_class_avoidance]]**: that insight was about R2.5 seed 43 specifically; it's correct at its level and the per-tag Δ_TL=+0.004 finding is reproduced (this run's Δ_TL=+0.04 is also ~zero). What's new is the additional negative datapoint that the seed-43 policy doesn't generalise; that goes here as a new insight rather than overwriting the older one.
- **Cross-cell comparison stays clean**: the R2.5 §12 toolkit appendix said *Cell A1 is class-blind at every layer (spatial AND event-level)*; that statement also holds for seed 45 (per-tag Δ_TL ~ 0; bush-dive rates near zero in the camping-relevant sense). The "every layer" framing survives the seed change.
- **No new R3 design required** to handle this; Round 3 (food in all four quadrants) was already queued as the closure of the spatial-camping loophole, and the seed-45 result *strengthens* the case for it: the agent's avoidance of TL combined with starvation suggests the spatial layout under A1 is too narrow to support stable foraging-under-threat policies.

## Open questions and follow-ups

- **Was the seed-45 stay-and-eat policy a real attractor that the agent eventually left, or just a transient on the way to avoid-TL-and-starve?** The n106 crash at 6.6 h / 0.87 M episodes makes this unanswerable from this study; the n102 finish proves only the **final** policy at 10 M ep. A future-Claude could check this by reading the WandB time-series for run `m5h4m8dl` and looking at when `Episode/Steps` collapsed from ~500 to ~100; if it collapsed early, the stay-and-eat read was a brief transient; if late, the agent did learn it and then unlearned it.
- **Why didn't the agent persist with stay-and-eat?** A reasonable conjecture: stay-and-eat near BR with TL-rabbit and BR-rabbit hits accumulated; the agent eventually preferred starvation over rabbit-contact damage. But the RabbitHits = 2.40/ep on the final policy is actually moderate, so the trade-off is non-obvious. Would benefit from a behaviour-trajectory analysis (per-episode policy fingerprinting across training time).
- **Cell A1's interpretability problem**: with three different policies at two seeds across two nodes, A1 is now a poor cell to anchor *any* claim about class-conditional behaviour. The R2.6 seed-lock specifically failed to be a seed-lock; instead it surfaced policy fragility. Whether to keep A1 in the headline cells of the sameProp narrative or demote it is a paper-shaping decision the PI should weigh in on.

## References

- Design doc: [`docs/experiments/active/hypervigilance/sameprop_round26_design.md`](../../../experiments/active/hypervigilance/sameprop_round26_design.md) §1 (pre-registered predicates), §3 (Launch Manifest — Run 4 row).
- Spatial-level prior verdict (R2.5 seed 43, complementary not superseded): [[20260510_2237_sameprop_round25_no_class_avoidance]].
- Event-level verdict insight that this study extends: [[20260512_1428_sameprop_class_discriminating_defence_event_level]].
- Companion insight from this session — Cell C R2.6 directional replication (still running): [[20260518_1736_sameprop_c_seed44_directional_replication]].
- Companion insight — methodological lesson about WandB post-crash reads: [[20260518_1737_wandb_post_crash_frozen_state_misread]].
- WandB run: `m5h4m8dl` ([wandb.ai/sungwoolee/grid_world_pain/runs/m5h4m8dl](https://wandb.ai/sungwoolee/grid_world_pain/runs/m5h4m8dl)).
- Prior re-summary that mis-framed the partial as "in flight" and described the stay-and-eat regime: [`docs/experiments/summaries/20260513_1420_sameprop_rabbit_avoidance_study.md`](../../../experiments/summaries/20260513_1420_sameprop_rabbit_avoidance_study.md).
- Corrected reader-friendly re-summary (with R2.6 crash correction): [`docs/experiments/summaries/20260514_2332_sameprop_rabbit_avoidance_study.md`](../../../experiments/summaries/20260514_2332_sameprop_rabbit_avoidance_study.md).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume d79a0d50-3ac8-4fa2-9e7e-6a6437d6b059` (re-enter the session) or `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/d79a0d50-3ac8-4fa2-9e7e-6a6437d6b059.jsonl /tmp/20260518_1735.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260518_1736_sameprop_c_seed44_directional_replication]] (hypervigilance, 2026-05-18) — R2.6 Cell C seed 44 at 62.5% of 10M-episode budget on n101 (still running) repro
- [[20260518_1737_wandb_post_crash_frozen_state_misread]] (cluster_ops, 2026-05-18) — When a WandB run crashes, its `run.summary` keys stay frozen at the last logged 
<!-- END BACKLINKS -->
