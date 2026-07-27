---
id: 20260513_0015_active_swapped_geq_matched_reframes_meta
date: 2026-05-13
time: "00:15"
folder: hypervigilance
tags: [hypervigilance, learned_lesson, refutation, design]
summary: "Specialist ceiling table from the 6-world unmodulated probe (single-seed, 10M ep each) shows active_swapped (402 steps) > active_matched (338 steps). Swap is NOT harder than matched in isolation, which refutes the original 'swap is load-bearing' framing for the upcoming meta head-to-head and reframes it as a CKA-factorisation test instead."
related: ["20260510_2237_sameprop_round25_no_class_avoidance", "20260512_1428_sameprop_class_discriminating_defence_event_level", "20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin"]
session_origin: claude_code
session_label: "NMN R2 continual + 6-specialist analyzer verdict — first positive FiLM finding"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# Specialist ceiling table refutes "swap is load-bearing"; reframes meta head-to-head as a CKA-factorisation test

## Key conclusion

The six unmodulated specialist runs (10M episodes each, one specialist per `{active, passive} × {matched, distinct, swapped}` cell) produced per-world survival ceilings that **invert the design-time intuition that the swapped-olfactory condition is the hardest**. For the unmodulated baseline, `active_swapped` reaches 402 steps while `active_matched` reaches only 338 — i.e. swapping the olfactory mapping (predator smells like rabbit) is *easier* in isolation than the matched condition. This refutes the original "swap is load-bearing" framing for the upcoming meta head-to-head and reframes the question: the meta-head-to-head is not "can the modulator survive when smells are swapped?" but "can the modulator factor a single network across two contexts whose individual ceilings are 400 and 491?" — a CKA-factorisation test.

## Evidence, measurements, facts

- **Per-world ceilings (single-seed, 10M episodes, unmodulated baseline)**:
  - `passive_matched` 491 steps
  - `passive_swapped` 488 (median)
  - `passive_distinct` 457
  - `active_swapped` **402**
  - `active_distinct` 382
  - `active_matched` **338**
- **The flip**: active_swapped (402) > active_matched (338) by **+64 steps**.
- All 6 specialists ran with `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_unmod.yaml`; world configs in `configs/experiment/nmn_meta_2x3_mixture/` and `configs/experiment/hypervigilance/{01-interoNocicept_sameProp,02-sameProp_R2_passivePredator}.yaml`.
- WandB IDs: `p9g5kjx3` (active_matched, crashed at 99.7%), `iktjhpmm` (active_distinct), `2p5zgdk4` (active_swapped), `958mba24` (passive_matched, user-terminated at 96.7%), `z5dfkzw5` (passive_distinct), `44rumz7m` (passive_swapped).
- **Caveat — single-seed**: the 64-step gap is single-seed; replicate at 3 seeds before treating as load-bearing for the meta design.
- **Caveat — partial runs**: `active_matched` (p9g5kjx3) crashed at 99.7% with a Python `onerror(os.rmdir,…)` exception. `passive_matched` (958mba24) was terminated by user SIGINT at 96.7% (clean WandB flush captured). Both terminal ceilings are likely representative but verify per-stage averages aren't transient.

## Decisions and actions

- Specialist-arm results section written into `docs/experiments/active/hypervigilance/NMN_META_2x3_MIXTURE_PROBE.md` (§5.5).
- The meta head-to-head design (still blocked on `--mixture-mode` dev work) needs its hypothesis section rewritten: the original H₁c-meta predicate ("swap is the hardest condition; modulator wins by handling it") no longer matches the ceilings. New framing should be a CKA-factorisation test ("can the modulator support two simultaneous policy modes when each has a different ceiling?").
- This finding does NOT supersede `20260512_1428_sameprop_class_discriminating_defence_event_level` — that insight is about class discrimination *within* a single world; this one is about cross-world ceiling structure.

## Open questions and follow-ups

- Why is `active_swapped` easier than `active_matched` for the unmod baseline? Plausible: in the swapped condition the predator's olfactory signature is unique (matches no other entity), which paradoxically may serve as a clearer warning signal than the matched condition where olfactory information is ambiguous.
- All 3 passive cells show a **synchronised mid-training collapse at 5–7M episodes** that recovers — possible systemic exploration/exploitation rebalance crisis worth investigating. Could be a generic finding for the unmodulated baseline at this training scale.
- `passive_matched` shows 95.5% MaxSteps termination + asymmetric corner distances — corner-camping pattern reappearing at 10M ep scale; check whether the toolkit-v1 event-level measures (bush-dive rate, eat-under-threat) detect class discrimination here as they did at R2.5 (see [[20260512_1428_sameprop_class_discriminating_defence_event_level]]).
- 3-seed replication of the 6 specialists before locking the ceiling table.

## References

- Design doc: [`docs/experiments/active/hypervigilance/NMN_META_2x3_MIXTURE_PROBE.md`](../../../docs/experiments/active/hypervigilance/NMN_META_2x3_MIXTURE_PROBE.md)
- Related: [[20260512_1428_sameprop_class_discriminating_defence_event_level]] (event-level class discrimination within sameProp matched), [[20260510_2237_sameprop_round25_no_class_avoidance]] (R2.5 no class avoidance at spatial level)
- Sibling insights: [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]] (R2 continual win for modulator)
- WandB runs: [`p9g5kjx3`](https://wandb.ai/sungwoolee/grid_world_pain/runs/p9g5kjx3), [`iktjhpmm`](https://wandb.ai/sungwoolee/grid_world_pain/runs/iktjhpmm), [`2p5zgdk4`](https://wandb.ai/sungwoolee/grid_world_pain/runs/2p5zgdk4), [`958mba24`](https://wandb.ai/sungwoolee/grid_world_pain/runs/958mba24), [`z5dfkzw5`](https://wandb.ai/sungwoolee/grid_world_pain/runs/z5dfkzw5), [`44rumz7m`](https://wandb.ai/sungwoolee/grid_world_pain/runs/44rumz7m)
- Commit: `46dc0b1` (analysis docs)
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260513_0015_active_swapped_geq_matched_reframes_meta.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
