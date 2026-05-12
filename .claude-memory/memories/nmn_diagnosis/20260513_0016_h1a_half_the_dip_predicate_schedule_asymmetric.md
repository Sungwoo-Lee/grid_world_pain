---
id: 20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric
date: 2026-05-13
time: "00:16"
folder: nmn_diagnosis
tags: [nmn, hypervigilance, learned_lesson, design]
summary: "The R2 continual probe's H₁a 'half-the-dip stability gap' predicate is schedule-asymmetric and methodologically malformed: the design assumed stage 2 is a stress dip, but in the active→passive→active→passive→active schedule, stage 2 is the *easier* passive stage, so survival rises rather than dips. The modulator's actual contribution lives in recovery speed at passive→active boundaries, not in dip-depth attenuation. H₁a is INCONCLUSIVE, not refuted — the predicate needs re-formalisation."
related: ["20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin"]
session_origin: claude_code
session_label: "NMN R2 continual + 6-specialist analyzer verdict — first positive FiLM finding"
importance: medium
status: active
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# H₁a half-the-dip predicate is schedule-asymmetric — modulator's contribution lives in recovery speed, not dip-depth

## Key conclusion

When analyzing the R2 continual sister pair, the experiment-analyzer found that the design doc's H₁a hypothesis (the "half-the-dip stability gap" — that the modulator should attenuate the survival dip at stage transitions by at least 50% compared to the unmodulated baseline) cannot be cleanly evaluated as written. The predicate implicitly assumed every stage transition is a dip, but the 5-stage `active → passive → active → passive → active` schedule has a built-in asymmetry: passive stages are easier than active ones, so the transitions go `harder → easier → harder → easier → harder` — only 3 of the 4 transitions involve a survival drop. On the active-to-passive transitions, survival *rises* rather than dips, so a "half-the-dip" predicate is undefined there. Where dips do occur (passive → active), both arms drop equivalently in magnitude; the modulator's actual contribution lives in **recovery speed after the dip**, not in the dip's depth. H₁a is INCONCLUSIVE rather than refuted; the predicate needs re-formalisation as a recovery-speed metric.

## Evidence, measurements, facts

- The design doc's H₁a pre-registered predicate: "modulator attenuates the survival dip at stage transitions by ≥ 50% compared to baseline" (`docs/experiments/active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md` §4.1).
- The 5-stage schedule (from `configs/continual/nmn_double_return.yaml`): `01_active → 02_passive → 03_active → 04_passive → 05_active`. So 4 transitions: active→passive, passive→active, active→passive, passive→active.
- Empirically (from the analyzer's per-stage breakdown): survival *rises* at active→passive transitions (the agent gets an easier world), so "dip depth" is undefined. Survival *drops* at passive→active transitions for both arms, with comparable initial drop magnitudes — but modulator recovers faster.
- The clean positive signals from this run (H₁b catastrophic forgetting, H₁c reusable subnetwork — see [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]]) are about **steady-state survival within a returned-to stage**, not about transition dip depth. The modulator wins by recovering faster, not by dipping less.
- The analyzer flagged this as a methodological flaw, not an empirical refutation — the predicate was malformed for this specific schedule shape.

## Decisions and actions

- The H₁a verdict in the design doc is recorded as **INCONCLUSIVE**, with an explicit note that the predicate needs re-formalisation. This is captured in the §5.5 Results and §6 Conclusions of `NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md`.
- Future continual-schedule experiments must check the schedule's asymmetry at predicate-authoring time, not at analysis time. Add to the experiment-designer's checklist: "Does the predicate's sign match the direction of survival change at every transition in the schedule?"
- New predicate candidate (not yet pre-registered): **recovery half-life** — number of episodes after the passive→active transition until survival recovers to 90% of the pre-passive steady state. Modulator hypothesized to have shorter recovery half-life than baseline.

## Open questions and follow-ups

- What is the formal definition of "recovery speed" — episodes-to-90%-of-steady-state, or area-under-the-recovery-curve, or peak-survival-on-the-return stage?
- Should the next continual probe use a single-direction schedule (`passive → active` only) to give H₁a a clean predicate, or a symmetric schedule that preserves the catastrophic-forgetting test?
- Does the asymmetry generalise to schedules with more stages, or only to the specific 5-stage cycle used here?

## References

- Design doc: [`docs/experiments/active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md`](../../../docs/experiments/active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md) §4.1 (original predicate), §5.5 + §6 (analyzer flag)
- Sibling insight: [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]]
- Config: [`configs/continual/nmn_double_return.yaml`](../../../configs/continual/nmn_double_return.yaml)
- Commit: `46dc0b1` (analysis docs)
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric.md`.
