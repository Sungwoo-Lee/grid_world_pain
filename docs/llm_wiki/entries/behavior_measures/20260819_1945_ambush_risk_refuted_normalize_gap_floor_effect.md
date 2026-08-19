---
id: 20260819_1945_ambush_risk_refuted_normalize_gap_floor_effect
date: 2026-08-19
time: "19:45"
folder: behavior_measures
tags: [refutation, hypervigilance, learned_lesson, decision]
summary: "REFUTED: ambush risk does NOT explain why injury suppresses cover use. Re-training all 10 rest-premium arms with the 2-12 hiding predators removed left the injury-suppression gap positive in 10/10 arms. The ABSOLUTE gap shrank (+5.9 -> +3.0 pp) but only because overall bush use collapsed (uninjured 14.2% -> 7.7%); the RELATIVE gap barely moved (41% -> 34%). Method rule: when an intervention moves the baseline, a shrinking absolute difference is a floor effect, not an effect — normalize before concluding."
related: ["20260805_0118_regime_matched_probe_and_run_sweep_probe_dir", "20260810_1753_bushrefuge_injury_suppresses_bush_use_heal_by_rest", "20260818_1620_rest_premium_sweep_refuted", "20260819_1946_probe_snapshot_staleness_verdict_on_partial_training"]
session_origin: claude_code
session_label: "no-hiding-predator verdict + cluster teardown"
importance: high
status: settled
valid_until: null
confidence: medium
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# Ambush risk refuted as the cause of injury-suppressed cover use — and the floor-effect trap that nearly hid it

## Key conclusion
The hypothesis (user + collaborator): an injured agent won't cross open ground to the refuge bush because 2-12 invisible `hiding_predator` ambushers could be anywhere, so the trip is too risky. Test: re-train all 10 rest-premium arms with the hiding predators removed from the training environment, then re-probe.

**Refuted.** Removing the risk did not restore injured cover use, so the risk was not what was driving it. The **heal-by-rest confound** ([[20260810_1753_bushrefuge_injury_suppresses_bush_use_heal_by_rest]]) remains the standing explanation: injury heals ONLY via the Rest action and healing is rewarded, so an injured agent's best move is to freeze and heal wherever it stands — it is not avoiding the trip, it has no reason to make it.

**The methodological half is the more portable lesson.** The absolute injury-suppression gap DID shrink, +5.9 -> +3.0 pp, and reporting that alone would have read as partial support. But the intervention also halved the baseline the gap is measured against. Normalized, the suppression is essentially unchanged. **When an intervention moves the baseline, always normalize the difference before interpreting its change.**

## Evidence, measurements, facts
- Probe: 10 arms x 12 conditions, 3,940 cells / 47,280 evals, 0 failures, nodes 101/102/103/105/113. Parents and no-hide arms step-matched at ~35.7-40.3M episodes.
- Wander condition (`avoid_rabbitwander`), tail-25% of checkpoints, paired per-checkpoint:

  | | uninjured | injured | absolute gap | gap as % of uninjured |
  |---|---|---|---|---|
  | with ambush (parent) | 14.2% | 8.3% | +5.9 pp | **41%** |
  | no ambush (new) | 7.7% | 4.7% | +3.0 pp | **34%** |

- Direction unanimous: gap positive in **10/10** arms in BOTH conditions (sign test p ~ 0.001 under a no-direction null). Per-arm block-bootstrap 95% CIs (L=20, B=4000) exclude zero on the positive side in 7/10 no-hide arms; the other 3 straddle zero; **none** go negative.
- Removing ambushers made the bush far less valuable to everyone, which is the floor effect: mean uninjured dwell 14.2% -> 7.7%. Arm a01 is the clearest case — parent 27.9% -> 17.7% (37% suppression), no-hide 4.9% -> 2.7% (45% suppression): relatively WORSE, not better.
- Survival was 101 steps (the episode cap) in every wander cell on both sides, confirming all 20 agents trained fine and the difference is not a competence artifact.
- Figures: `results/eval/avoidance/restprem_nohide/SUMMARY_nohide_vs_parent.png` (4-panel) and `SUMMARY_nohide_relative_gap.png` (absolute-collapse / absolute-gap / relative-gap, the panel that settles it).

## Decisions and actions
- Ambush-risk hypothesis closed as refuted. Two interventions have now failed against the same signature — re-pricing rest continuity ([[20260818_1620_rest_premium_sweep_refuted]]) and removing ambush risk — both leaving injury-suppression intact.
- User terminated all 10 no-hide runs at ~58-60M episodes once the pattern was clear.
- Direction implication surfaced to the user: the evidence increasingly points at the **reward structure itself** (heal-by-rest), which is a training-environment change, not a parameter sweep. No new intervention launched.
- Standing caveat carried over from the premium sweep: **n=1 seed per arm**. Arm a05 is a +10.8 pp outlier. The unanimous SIGN across 20 independently-trained agents is the robust part; individual arm magnitudes are not.

## Open questions and follow-ups
- The verdict rests on checkpoints up to ~40M only — see [[20260819_1946_probe_snapshot_staleness_verdict_on_partial_training]]. Parents have 100M-episode checkpoints on disk that were never probed.
- A frozen-injury probe was considered and REJECTED as off-distribution (the agent never experienced frozen injury during training, so it cannot be expected to act sensibly under it). Decoupling injury from rest must happen in the TRAINING environment.

## References
- Configs: `configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a{01..10}.yaml` (commit `cfc0293`); probe spec `configs/eval_sweeps/restprem_nohide_rppo.yaml` (`d0da3ce`).
- Data: `results/eval/avoidance/restprem_nohide/` vs `results/eval/avoidance/restprem/`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a` or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260819_1946_probe_snapshot_staleness_verdict_on_partial_training]] (behavior_measures, 2026-08-19) — A watermark-incremental probe only covers checkpoints that existed WHEN IT LAST 
<!-- END BACKLINKS -->
