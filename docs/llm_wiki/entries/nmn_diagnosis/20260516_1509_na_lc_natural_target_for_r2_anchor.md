---
id: 20260516_1509_na_lc_natural_target_for_r2_anchor
date: 2026-05-16
time: "15:09"
folder: nmn_diagnosis
tags: [nmn, design, decision, learned_lesson]
summary: "Given the refined target-one constraint, NA/LC (noradrenergic / locus coeruleus) is the natural targeted system for the NMN paper because the R2 anchor's modulator phasic burst at regime change IS the canonical Aston-Jones & Cohen 2005 LC-NA signature."
related: ["20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin", "20260516_1504_symposium_substrate_right_rhetoric_wrong", "20260516_1505_target_one_acknowledge_many_defer_full_coverage"]
session_origin: claude_code
session_label: "pi-probe-prioritization / symposium round + v5"
importance: medium
status: settled
valid_until: null
confidence: medium
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# NA / LC is the natural targeted system for the NMN paper

## Key conclusion

Once the project commits to single-system targeting (per the [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]] rule), **NA / LC (noradrenergic / locus coeruleus) is the natural pick** because the R2 anchor's modulator phasic burst at the active↔passive regime change *is* the canonical Aston-Jones & Cohen 2005 LC-NA signature (sharp at boundary, quiet during steady state). v5 of the direction memo names NA/LC explicitly in §1 + §3 + frontmatter (`targeted_neuromodulator: NA`).

## Evidence, measurements, facts

- R2 anchor empirical signature (per [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]]): modulator hidden vector shows sharp phasic activity at each of the 4 active↔passive regime transitions, quiet during steady state within each stage. Sign-consistent at 3 of 4 transitions (≥3σ from steady-state baseline).
- **Aston-Jones & Cohen 2005 LC-NA canonical signature**: phasic burst at unexpected-uncertainty events / regime change, tonic baseline at steady state. The R2 modulator pattern matches this verbatim.
- Alternative readings considered + rejected:
  - DA-RPE (Schultz): would expect modulator activity to correlate with TD error magnitude continuously, not just at boundary. The phasic-only pattern is less compatible.
  - ACh / basal forebrain (Yu & Dayan 2005): would expect modulator activity to track expected uncertainty (precision), not regime change as a discrete event.
  - 5-HT / dorsal raphe (Daw et al.): long-horizon / aversive control — wrong timescale (5-HT is tonic-dominated; the R2 modulator is sharply phasic).
- Project's biological-substrate professor (`professor-neuromodulation`) confirmed NA/LC as the most likely channel attribution in multiple rounds (v1 §8 Q3; v3-round symposium contribution).

## Decisions and actions

- v5 of the direction memo names NA/LC explicitly:
  - Frontmatter `targeted_neuromodulator: NA (noradrenergic / locus coeruleus) — Aston-Jones & Cohen 2005 phasic-burst-at-regime-change signature`
  - §1 plain-English entry-point: "The project targets noradrenergic / locus-coeruleus modulation as its biological anchor"
  - §3 unification claim: explicit citation of the LC-NA phasic-burst signature
- The non-target Doya channels (ACh, DA, 5-HT) are deferred to future work in v5 §5.9.
- NA/LC framing is consistent with all four v5 §11 path options (A / B / C / E) — none of them require a different targeted system.

## Open questions and follow-ups

- Whether the basket-of-effects framing (per [[20260516_1504_symposium_substrate_right_rhetoric_wrong]]) means the project's modulator should be interpreted as an LC-NA-analog producing multiple Doya-like effects, or as a different biological signal entirely, is an empirically-decidable question (v5 §5 program is the test).
- The Bayesian-brain lens reads the same modulator as "policy precision" (one of A=sensory / B=state-transition / C=policy precisions); compatible with the NA-LC reading but not identical (precision is a more general framework than NA-LC).

## References

- v5 of direction memo (NA-LC named throughout): [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md`](../../../docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md).
- R2 empirical anchor: [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]].
- Refined targeting rule that motivates the pick: [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]].
- Symposium contribution from neuromod prof: [`docs/project/symposium/20260516_impact_vs_reasonable/professor_neuromodulation_contribution.md`](../../../docs/project/symposium/20260516_impact_vs_reasonable/professor_neuromodulation_contribution.md).
- Key papers: Aston-Jones & Cohen 2005 (LC-NA phasic-burst signature), Wainstein 2025 (pupillometry + gain neuromodulation in RNN).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]] (nmn_diagnosis, 2026-05-16) — User's refined scoping constraint for single-paper neuromodulator coverage: targ
<!-- END BACKLINKS -->
