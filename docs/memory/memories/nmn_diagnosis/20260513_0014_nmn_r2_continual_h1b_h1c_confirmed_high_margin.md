---
id: 20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin
date: 2026-05-13
time: "00:14"
folder: nmn_diagnosis
tags: [nmn, film, hypervigilance, decision, learned_lesson]
summary: "R2 continual sister-pair probe (5.1M ep, 5-stage active↔passive schedule) delivers the first clearly-positive architecture-vitality finding for the FiLM modulator: H₁b (catastrophic-forgetting resistance) and H₁c (Tsuda-style reusable subnetwork) both CONFIRMED at ~25× the seed-noise floor. Modulator beats unmodulated baseline by +107 steps on the 1st return-to-active stage and +132 steps on the 2nd; modulator's 2nd return ≥ 1st (+5.75), baseline shows monotone decay (−19.79)."
related: ["20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse", "20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4", "20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric", "20260513_0017_mod_h_logging_gap_blocks_cka_precheck"]
session_origin: claude_code
session_label: "NMN R2 continual + 6-specialist analyzer verdict — first positive FiLM finding"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# R2 continual sister pair: NMN catastrophic-forgetting + reusable-subnetwork hypotheses confirmed at ~25× seed-noise floor

## Key conclusion

The five-stage continual probe (`active → passive → active → passive → active`, 5.1M episodes on n106 RTX 3090) delivers the **first clearly-positive architecture-vitality finding** for the project's FiLM modulator. Across the two return-to-active stages, the modulator-augmented agent outperforms the unmodulated baseline by 107–132 survival steps — roughly 25× the project's measured seed-noise floor (±4.4 steps), far above any single-seed plausibility concern. The catastrophic-forgetting-resistance hypothesis (H₁b in the design doc) and the Tsuda hypertube reusable-subnetwork hypothesis (H₁c) are both CONFIRMED with a clean sign-flip on H₁c. The half-the-dip stability-gap hypothesis (H₁a) is INCONCLUSIVE because the formal predicate was schedule-asymmetric — see sibling insight [[20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric]].

## Evidence, measurements, facts

- **Runs**: modulator [`8eorbxhq`](https://wandb.ai/sungwoolee/grid_world_pain/runs/8eorbxhq) (config `recurrent_ppo_nmn_film_g1_tempceil5.yaml`, NOT `het_unmod` as the launch table claimed — verified via local wandb config); baseline [`lrzvg8k6`](https://wandb.ai/sungwoolee/grid_world_pain/runs/lrzvg8k6) (config `recurrent_ppo_nmn_het_unmod.yaml`).
- **Episodes**: 5,097,472 (mod) vs 5,093,982 (unmod); runtime 13.3 h vs 9.5 h.
- **Per-stage survival on the two return-to-active stages** (the load-bearing measurement targets):
  - 1st return: mod ~237 steps, unmod ~131 steps → **+107**
  - 2nd return: mod ~243 steps, unmod ~111 steps → **+132**
- **H₁b (catastrophic-forgetting resistance) — CONFIRMED**: modulator forgets 65% less on first return, 72% less on second.
- **H₁c (Tsuda reusable subnetwork) — CONFIRMED with sign-flip**: modulator's 2nd return ≥ 1st return (+5.75); baseline shows monotone decay (−19.79).
- **Mechanism check**: modulator `temperature_mean` swings ≥3σ at 3 of 4 stage transitions, sign-consistent (sharper for active stages, softer for passive). Modulator is mechanistically engaged at stage boundaries.
- **Seed-noise floor**: ±4.4 steps (measured in prior `nmn_diagnosis` work). The 107–132 step gap is ~25× this floor.
- **Final-stage (stage 5) mean survival** (steady-state, not return-to-active): mod 246, unmod 88 (~2.8×). This is the headline summary number the analyzer extracted on first pass; the return-to-active numbers are the load-bearing ones.

## Decisions and actions

- Verdict written into `docs/experiments/active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md` (§5.5 Results, §6 Conclusions). Status flipped to `ANALYZED`.
- The "modulator partially engages but never beats baseline" framing from `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse` and `20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4` is now **outdated for the continual-schedule regime** — those refutations stand for steady-state heterogeneity sweeps but do NOT generalize to schedule-changing environments where the modulator's value lives at stage boundaries.
- Open follow-up: 3-seed replication of this R2 pair to seed-lock the finding (single-seed positive is strong but standard practice is to lock).

## Open questions and follow-ups

- Does the H₁b / H₁c effect hold at 3 seeds? (Recommended replication.)
- Is the modulator's recovery-speed contribution larger at the 2nd return than the 1st, or does it plateau? (The +132 vs +107 numbers suggest growth, but n=1 each.)
- Does the same effect appear if the schedule is permuted (e.g., `passive → active → passive → active → passive`)?
- Can the analyzer-flagged "modulator's contribution lives in recovery speed, not dip-depth attenuation" be reformulated into a sharper pre-registered predicate?

## References

- Design doc: [`docs/experiments/active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md`](../../../docs/experiments/active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md)
- Prior verdicts this reframes: [[20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse]], [[20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4]]
- Sibling insights from same session: [[20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric]] (H₁a predicate flaw), [[20260513_0017_mod_h_logging_gap_blocks_cka_precheck]] (Mahalanobis test unevaluable)
- Continual config: [`configs/continual/nmn_double_return.yaml`](../../../configs/continual/nmn_double_return.yaml)
- WandB: [`8eorbxhq`](https://wandb.ai/sungwoolee/grid_world_pain/runs/8eorbxhq), [`lrzvg8k6`](https://wandb.ai/sungwoolee/grid_world_pain/runs/lrzvg8k6)
- Commits: `0684eda` (diary), `46dc0b1` (analysis docs)
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260513_0015_active_swapped_geq_matched_reframes_meta]] (hypervigilance, 2026-05-13) — Specialist ceiling table from the 6-world unmodulated probe (single-seed, 10M ep
- [[20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric]] (nmn_diagnosis, 2026-05-13) — The R2 continual probe's H₁a 'half-the-dip stability gap' predicate is schedule-
- [[20260513_0017_mod_h_logging_gap_blocks_cka_precheck]] (nmn_diagnosis, 2026-05-13) — The R2 continual probe's modulator-engagement Mahalanobis check is unevaluable b
<!-- END BACKLINKS -->
