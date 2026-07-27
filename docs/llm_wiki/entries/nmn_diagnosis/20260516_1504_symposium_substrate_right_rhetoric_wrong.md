---
id: 20260516_1504_symposium_substrate_right_rhetoric_wrong
date: 2026-05-16
time: "15:04"
folder: nmn_diagnosis
tags: [nmn, film, design, decision, learned_lesson, meta]
summary: "Four professors (neuromod / rl-bayesian-dl / bayesian-brain / pain-modeling) independently converged on the same recommendation in their own languages: v4 of the direction memo is architecturally fine but rhetorically over-committed (per-site one-to-one channel attribution); each lens's reformulation is the same claim."
related: ["20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin", "20260516_1505_target_one_acknowledge_many_defer_full_coverage", "20260516_1506_film_hypernet_bhn_bnn_2x2_lineage", "20260516_1508_v5_last_rhetorical_round_before_experiments", "20260516_1512_multi_agent_symposium_pattern"]
session_origin: claude_code
session_label: "pi-probe-prioritization / symposium round + v5"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# Symposium round produces "substrate right, rhetoric wrong" four-way convergence

## Key conclusion

The 2026-05-16 impactful-vs-reasonable symposium ran four professor agents (neuromod / rl-bayesian-dl / bayesian-brain / pain-modeling) in parallel on v1→v4 + the concept memo + the lineage investigation. All four independently converged on the same headline diagnosis: **v4's architecture (FiLM at three sites + Row EE-6 + T/P split + experimental program) is correct; v4's per-site one-to-one channel attribution ("Site A = α channel; Site C = β channel") is the *same* one-to-one error v4 §3.0 just called out in Doya, one level down**. Each professor proposed the same recommendation in their own domain language — the convergence itself is the symposium's load-bearing finding, not any individual professor's framing.

## Evidence, measurements, facts

- **Neuromod prof**: "target-as-basket-of-effects" — each injection site is a *target* on which a single LC-NA-analog signal acts, producing a basket of effects via target-specific (receptor-density-analog) read-out matrices. Site A produces learning-rate-like + precision-like + divisive-normalisation-like + expected-uncertainty-like effects *simultaneously*.
- **RL-bayesian-dl prof**: "observation-point reframing" — sites A/B/C as observation points on a single modulator signal; dissociation arises from per-site read-out matrices, not channel decomposition. Galanti & Wolf 2020 Theorem 4 is the formal identifiability anchor for one-to-many readouts from a shared conditioner. v4 is "not architecturally broken — only rhetorically over-committed".
- **Bayesian-brain prof**: "one signal, many precisions" — under a precision-coding framework, a single modulator naturally produces multiple downstream effects depending on which prediction-error stream it weights. The project's substrate may be the first interoceptive-RL implementation of the "one signal, many precisions" prediction the precision-coding tradition has carried for two decades.
- **Pain-modeling prof**: "target-flexible interoceptive modulation" — R2 anchor reread as target-flexible modulation under interoceptive non-stationarity (distinguishes from Lee 2024 / Wang 2024's exteroceptive non-stationarity). Pain-construct gate restated in behaviourally-defined language (drops channel attribution).
- **No hard contradictions** across the four sidecars (per `postdoc_synthesis.md` §3.4); the closest was a positioning disagreement (biological-lead vs architectural-lead → surfaces as Path A vs Path B trade-off).
- **Re-framing cost = zero code, zero experiments, zero equations** — every professor agreed (v4's predictions §5 program is preserved verbatim in v5; only the rhetorical scaffolding changes).
- Five files in [`docs/project/symposium/20260516_impact_vs_reasonable/`](../../../docs/project/symposium/20260516_impact_vs_reasonable/) carry the contributions + synthesis.

## Decisions and actions

- v5 of the direction memo absorbed the convergent finding: target-and-basket-of-effects framing replaces per-site one-to-one; §3.5 "interpretive lenses" added (Doya-channel / precision-coding / hypernet-BHN / target-flexibility as four parallel lenses on the same substrate, none load-bearing).
- v5 committed as `1f65cae` on v1.4; v4 flipped to status: superseded.
- The "substrate right, rhetoric wrong" framing also informs the [[20260516_1508_v5_last_rhetorical_round_before_experiments]] pace-flag stance.

## Open questions and follow-ups

- The framing pick among the four candidate paths (A / B / C / E) is still pending user decision — see [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]] for the constraint that scoped the paths.
- The substrate's *empirical* one-to-many claim (that one signal genuinely produces dissociable behavioural effects across the three targets) is still untested experimentally — v5 §5 program is the test.

## References

- Symposium materials: [`docs/project/symposium/20260516_impact_vs_reasonable/`](../../../docs/project/symposium/20260516_impact_vs_reasonable/) (4 prof contributions + `postdoc_synthesis.md`).
- v5 of direction memo: [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md`](../../../docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md).
- Related insights: [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]] (user constraint), [[20260516_1506_film_hypernet_bhn_bnn_2x2_lineage]] (math lineage), [[20260516_1508_v5_last_rhetorical_round_before_experiments]] (PI pace flag), [[20260516_1512_multi_agent_symposium_pattern]] (workflow pattern).
- Empirical anchor: [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]] (R2 finding the symposium reviewed).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]] (nmn_diagnosis, 2026-05-16) — User's refined scoping constraint for single-paper neuromodulator coverage: targ
- [[20260516_1506_film_hypernet_bhn_bnn_2x2_lineage]] (nmn_diagnosis, 2026-05-16) — Formal 2×2 lineage of FiLM / Hypernet / Bayesian Hypernet / Bayesian Neural Net 
- [[20260516_1508_v5_last_rhetorical_round_before_experiments]] (nmn_diagnosis, 2026-05-16) — PI pace flag: project cycled through 4 versions of direction memo + concept memo
- [[20260516_1509_na_lc_natural_target_for_r2_anchor]] (nmn_diagnosis, 2026-05-16) — Given the refined target-one constraint, NA/LC (noradrenergic / locus coeruleus)
- [[20260516_1512_multi_agent_symposium_pattern]] (subagent_engineering, 2026-05-16) — Multi-agent symposium pattern (evolution of the v2 research chain): 4 professors
- [[20260528_0217_episode_direction_4x3_framework_two_papers]] (nmn_diagnosis, 2026-05-28) — project_plan.md rewritten from 760-line Nature MI staged plan to 138-line stable
<!-- END BACKLINKS -->
