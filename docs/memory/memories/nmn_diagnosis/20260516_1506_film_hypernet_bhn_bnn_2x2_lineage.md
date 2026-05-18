---
id: 20260516_1506_film_hypernet_bhn_bnn_2x2_lineage
date: 2026-05-16
time: "15:06"
folder: nmn_diagnosis
tags: [nmn, film, design, decision, learned_lesson]
summary: "Formal 2×2 lineage of FiLM / Hypernet / Bayesian Hypernet / Bayesian Neural Net with five rigorous claims (FiLM ⊂ Hypernet; Hypernet ⊆ point-mass BHN as limit; BNN ⊥ BHN; FiLM-Ensemble ≈ finite-mixture BHN approximation; Galanti & Wolf 2020 Thm 4 modularity carries through)."
related: ["20260516_1504_symposium_substrate_right_rhetoric_wrong", "20260516_1507_gamma_bellman_not_forward_pass_filmable", "20260516_1511_math_reviewer_catches_silent_direction_errors"]
session_origin: claude_code
session_label: "pi-probe-prioritization / lineage investigation + symposium + v5"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# FiLM / Hypernet / BHN / BNN — formal 2×2 lineage with five rigorous claims

## Key conclusion

The `professor-rl-bayesian-dl` lineage-investigation memo produced a clean 2×2 taxonomy over {deterministic, stochastic} × {unconditional, context-conditional} weights — yielding standard NN / Hypernet / BNN / Bayesian Hypernet — and proved five formal claims that fix the project's substrate position. **The project sits at the top-right cell (deterministic context-conditional = Hypernet, with FiLM strictly inside). Row EE-6 (FiLM-Ensemble + heteroscedastic-precision compound) upgrades the substrate toward the bottom-right (BHN) via finite-mixture approximation — reframed as "first instantiation of an approximate Bayesian Hypernetwork with heteroscedastic likelihood for an RL policy".**

## Evidence, measurements, facts

- **2×2 taxonomy**:
  | | Unconditional weights | Context-conditional weights |
  |---|---|---|
  | Deterministic | Standard NN | **Hypernet** (FiLM strictly inside) ← project sits here |
  | Stochastic posterior | **BNN** | **Bayesian Hypernet** |
- **Claim 1 — FiLM ⊂ Hypernet (RIGOROUS)**: FiLM is a hypernet where the generated "weights" are restricted to diagonal-scaling matrices + bias. Established via Ha 2016's scaling-vector trick + Abdollahzadeh 2021's reinterpretation lemma (per-channel γ on conv activations = degenerate per-channel-uniform kernel modulation; Tsuda 2021's scalar weight-gain is the further-degenerate scalar instance; FiLM-on-activations is the more expressive intermediate). Nested chain: **Tsuda ⊂ vanilla FiLM ⊂ KML**.
- **Claim 2 — Hypernet ⊆ point-mass BHN AS A LIMIT (not as sub-architecture)**: deterministic hypernet is the limit of BHN where the posterior degenerates to a point mass. The "as a limit" qualifier is load-bearing — the prof explicitly distinguished limit from sub-architecture.
- **Claim 3 — BNN ⊥ BHN**: orthogonal uncertainty levels. BNN puts uncertainty on the *base* network's weights (no context); BHN puts uncertainty on the *generator* of context-conditional weights. Combinable but no corpus paper does it.
- **Claim 4 — FiLM-Ensemble ≈ finite-mixture BHN approximation (NOT clean BHN)**: Turkoglu 2022's FiLM-Ensemble is joint-trained members without an ELBO; approximates the posterior but is not a principled variational BHN.
- **Claim 5 — Modularity (Galanti & Wolf 2020 Thm 4) carries through the family**: the formal identifiability of dissociable readouts from a shared conditioner. THIS is the math result the symposium leveraged for "one-signal-many-readouts" (see [[20260516_1504_symposium_substrate_right_rhetoric_wrong]]).
- **Row EE-6 reframing**: the concept memo's "novel substrate-level combination" is more precisely "first instantiation of an approximate Bayesian Hypernetwork with a heteroscedastic likelihood for an RL policy" — novelty preserved, lineage clean.
- All claims math-reviewer-audited as "holds" or "holds with one-sentence clarification" (no fundamental re-derivation needed).

## Decisions and actions

- v4 of the direction memo absorbed the lineage as §3.0 in the concept memo. v5 promotes the lineage to one of four interpretive lenses in direction-memo §3.5 (alongside Doya-channel, precision-coding, target-flexibility).
- Row EE-6 reframing landed in v4 and is preserved in v5.
- Investigation memo file: [`docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md`](../../../docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md).
- Math-reviewer audit landed in concept memo §10; full audit committed as `0485b19`.

## Open questions and follow-ups

- Whether the project actually implements the BHN limit (via Row EE-6's finite-mixture approximation) experimentally is gated by Path B vs Path A (see §11 of v5).
- BNN ⊥ BHN combination as future work — no paper has done it, no urgency for the first paper.

## References

- Lineage investigation memo: [`docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md`](../../../docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md).
- Concept memo §3.0 lineage callout: [`docs/project/concepts/film_neuromod_integration.md`](../../../docs/project/concepts/film_neuromod_integration.md).
- v5 §3.5 interpretive lenses: [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md`](../../../docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md).
- Related insights: [[20260516_1507_gamma_bellman_not_forward_pass_filmable]] (Q2 of the investigation), [[20260516_1504_symposium_substrate_right_rhetoric_wrong]] (uses Claim 5 for the one-signal-many-readouts framing), [[20260516_1511_math_reviewer_catches_silent_direction_errors]] (audit value).
- Key papers: Perez 2018 (FiLM), Ha 2016 (HyperNetworks), Krueger 2017 (Bayesian Hypernets), Galanti & Wolf 2020 (modularity), Turkoglu 2022 (FiLM-Ensemble), Abdollahzadeh 2021 (reinterpretation lemma), Kendall & Gal 2017 (heteroscedastic precision).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260516_1504_symposium_substrate_right_rhetoric_wrong]] (nmn_diagnosis, 2026-05-16) — Four professors (neuromod / rl-bayesian-dl / bayesian-brain / pain-modeling) ind
- [[20260516_1507_gamma_bellman_not_forward_pass_filmable]] (nmn_diagnosis, 2026-05-16) — γ_Bellman cannot be carried by a forward-pass FiLM substrate: the GRU update gat
- [[20260516_1511_math_reviewer_catches_silent_direction_errors]] (subagent_engineering, 2026-05-16) — Math-reviewer audit caught two silent direction-memo errors (Hessian-direction r
<!-- END BACKLINKS -->
