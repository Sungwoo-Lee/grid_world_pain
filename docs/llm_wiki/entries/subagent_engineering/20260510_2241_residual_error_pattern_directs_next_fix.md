---
id: 20260510_2241_residual_error_pattern_directs_next_fix
date: 2026-05-10
time: "22:41"
folder: subagent_engineering
tags: [subagent, learned_lesson, decision, meta]
summary: "Methodology pattern: when a partial fix lands (H2-band outcome), the residual-error pattern in the diagnostic output identifies the next-best candidate, NOT the original list-order or the plan's prescribed-next-step. Concretely: Z1's residual error was disproportionate on negative rewards → bin-range deviation (§6 item 2, mechanistic match) was selected over the plan §4 H2 prescription (candidate #1, GRU reset gate)."
related: ["20260509_1537_professor_analysis_resets_exotic_investigation", "20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry", "20260510_2240_reference_impl_compare_only_act_intersections"]
session_origin: claude_code
session_label: "dreamer_sheeprl_compare_and_zero_init_2026-05-10"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4ae5401e-443a-406e-93e1-5431c67b5f59.jsonl
raw_completeness: full
---

# Residual-error pattern dictates the next fix, not list order or plan prescription

## Key conclusion
When a partial fix lands in an iterative debugging cascade — a step closer but not all the way home — the right way to pick the next candidate is to **read the residual-error pattern in the diagnostic output and find the candidate whose mechanistic prediction matches that residual**. NOT the next item in the original candidate list, NOT the next item the plan's pre-registered failure-mode catalog prescribed. The concrete example this session: Cell Z1's offline diagnostic showed reward MAE 0.39 → 0.28 (H2 band); the residual breakdown showed pos-MAE improved 49% but neg-MAE only 14%. That asymmetry directly fingered the two-hot bin-range deviation (§6 item 2: bins span ±20, death penalty −100 unrepresentable) as the mechanistic match — overriding the plan §4 H2 prescription which named candidate #1 (GRU reset gate). The bin-range fix was selected because the residual *had* a specific shape, and one specific candidate's mechanistic prediction *had* that exact shape.

## Evidence, measurements, facts
- **The cascade context**: 4 candidate deviations from sheeprl comparison were flagged in §6. The user's rule (sibling insight `20260510_2240_reference_impl_compare_only_act_intersections`) said only act on those intersecting the live failure. Top-level Claude's initial recommendation: act on #4 (zero-init) first because it's the candidate most directly targeted at the reward-head failure. The user agreed; Z1 launched.
- **Z1 was a partial fix, not a full fix.** MAE @ h=5: 0.386 → 0.277 (28% reduction). Inside H2 band [0.15, 0.30); outside H1 < 0.15. The plan §4 H2 prescription said: "queue candidate #1 (GRU reset gate)." This was a generic rule "if H2, do the next item on the list."
- **Top-level Claude looked at the residual breakdown instead.** Training-time pos-MAE improved 49%, neg-MAE 14%. The asymmetry pointed at: a deviation that affects negative rewards but not positive ones. Two-hot bin range fits exactly: bins span raw `±20`, so any reward with `|r| > 20` saturates at the boundary bin. NoPred has no death-penalty event but does have small dense satiation rewards (some negative); the pattern is consistent. Sheeprl + Hafner-published code use bins spanning raw `±4.85×10⁸`. The mechanistic prediction matches the residual shape.
- **The plan-prescribed candidate (#1 GRU reset gate) was NOT the mechanistic match.** GRU reset gate not being applied would degrade EVERY observation channel uniformly — but the offline diagnostic showed most channels predicted fine; only reward (and the prev-action one-hot, an even more constrained channel) failed. So #1 doesn't predict the reward-specific residual.
- **The selected candidate (§6 item 2, bin range) was acted on**: senior-developer plan `docs/develop/active/diagnosis/dreamer_twohot_bin_range_fix.md`, code change in commits `f5df600` + `8089ee2`, Cell Z2 launched (`dreamer_twohotrng_NoPred_rr06_s0_n113`, WandB `q66macky`). Verdict pending.
- **The general principle**: in a multi-candidate cascade where each fix may be partial, the diagnostic output between fixes carries information that's NOT in the candidate list itself. Specifically: which channels / metrics / horizons the residual error concentrates on. That information gets thrown away if you traverse candidates by plan-order. Conversely, mechanistic-match selection means each fix's failure-mode signature directly determines the next test.
- **This is a complement to the "professor analysis resets exotic investigation" rule** (sibling `20260509_1537_professor_analysis_resets_exotic_investigation`). That rule was about *initial routing* when investigation has gone exotic. This rule is about *iterative routing* once a candidate set exists and partial fixes are landing.

## Decisions and actions
- **Codified the rule** for future cascades:
  1. After a fix lands and a partial-improvement is observed (H2-band or "closer but not done"), open the diagnostic output and characterize the residual: which channel, which sign, which horizon, which event type.
  2. For each candidate on the deferred list, predict its mechanistic effect on the residual. Any candidate whose predicted effect overlaps with the observed residual's shape is a mechanistic match.
  3. If exactly one mechanistic match exists, act on it. If multiple, surface to the user with mechanism-by-mechanism reasoning. If none, the residual is novel and the candidate list is exhausted — escalate.
  4. The plan's H1/H2/H0 prescriptions are useful as defaults BUT should be overridden when the residual signature points elsewhere.
- **Surfaced explicitly to the user during this session** as a 4-option AskUserQuestion: "twohot bin range (mechanistic match)" vs "candidate #1 GRU reset gate (plan prescription)" vs "both in sequence" vs "pause to read." The user picked the mechanistic-match option, validating the rule.

## Open questions and follow-ups
- Will Z2 close the gap below H1 (MAE < 0.15)? If yes, the rule's first application to this codebase produced a clean two-step cascade (zero-init → bin range → done). If H2 again, there will be a NEW residual pattern, and the rule applies recursively to candidates #1, #3 (critic-EMA), #2 (prior/posterior heads).
- Does the rule generalize beyond Dreamer? Likely yes for any iterative multi-candidate debugging — the pattern is "diagnostic that quantifies per-channel / per-event error" + "candidates with mechanistic predictions." Tests of the rule on the NMN investigation lineage might be informative.
- The rule has a corner case: what if the residual error pattern matches NO candidate in the list? That's a signal that the candidate list is incomplete OR the failure has multiple compounding causes. The rule should escalate (re-spawn professor analysis) rather than picking the closest-matching candidate by default.

## References
- Sibling insight (the scientific finding that produced the example): `20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry`.
- Sibling insight (the related routing rule from the same session): `20260510_2240_reference_impl_compare_only_act_intersections`.
- Predecessor methodology rule: `20260509_1537_professor_analysis_resets_exotic_investigation` (initial routing when investigation has gone exotic; this insight covers iterative routing once a candidate set exists).
- Concept doc with the residual-pattern verdict tables: `docs/project/concepts/dreamer_v3_implementation.md` §6 (deviation list, including item 2 = twohot bin range).
- Plan + Verification Report for Z1 (the partial fix that triggered the residual analysis): `docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`.
- Plan for Z2 (the residual-directed next candidate): `docs/develop/active/diagnosis/dreamer_twohot_bin_range_fix.md`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4ae5401e-443a-406e-93e1-5431c67b5f59` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
