---
id: 20260516_1507_gamma_bellman_not_forward_pass_filmable
date: 2026-05-16
time: "15:07"
folder: nmn_diagnosis
tags: [nmn, film, refutation, learned_lesson, decision]
summary: "γ_Bellman cannot be carried by a forward-pass FiLM substrate: the GRU update gate is per-unit recurrent-state retention, not a value-function horizon. Clean γ_Bellman lives at the loss (TD-target γ(c)), not in the forward pass. v4/v5 follow Lee 2024 §6 and name γ_Bellman as out-of-substrate."
related: ["20260513_0017_mod_h_logging_gap_blocks_cka_precheck", "20260516_1505_target_one_acknowledge_many_defer_full_coverage", "20260516_1506_film_hypernet_bhn_bnn_2x2_lineage"]
session_origin: claude_code
session_label: "pi-probe-prioritization / math audit + lineage investigation"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# γ_Bellman cannot live in the forward-pass FiLM substrate

## Key conclusion

Two independent reviews — math-reviewer flag (d) on the concept memo, then the `professor-rl-bayesian-dl` lineage+critic-modulation investigation memo — concluded that **the GRU update gate is per-unit recurrent-state retention, not a value-function horizon**. A clean Bellman-discount γ modulation requires modulating the TD target γ(c) directly, which lives at the *loss* / *learning update*, not in the forward pass. **A forward-pass FiLM substrate cannot carry γ_Bellman without architectural extension (hybrid system with a learned-discount-network feeding the TD target).** v4 and v5 of the direction memo follow Lee 2024 §6's precedent and name γ_Bellman as out-of-substrate.

## Evidence, measurements, facts

- **Math-reviewer flag (d)**: "the conceptual claim that 'additive β at GRU update-gate ⇔ effective Bellman γ change' is dimensionally incoherent — the GRU update gate produces per-unit hidden-state retention, not a global value-function horizon."
- **Same conflation warned by `professor-rl-bayesian-dl` in v1 §8 Q3 originally** — the math-reviewer's audit corroborated the prior warning, making it a TWO-independent-review finding.
- **Lineage investigation memo §3 candidate modulation points**:
  | Candidate | What it actually is |
  |---|---|
  | (a) Critic-head FiLM | Value-scale modulation, NOT horizon control |
  | (b) TD-target γ(c) | Clean γ_Bellman but lives at LOSS, leaves FiLM substrate |
  | (c) GAE λ | Effective n-step horizon — related but distinct |
  | (d) Learned discount network feeding TD target | Same as (b); still loss-level |
- **No paper in either corpus implements γ_Bellman modulation**. Lee 2024 §6 explicitly drops the 5-HT branch ("no convergent normative theory"). Xing 2022, Wang 2024, Rodriguez-Garcia 2026 all leave γ fixed.
- v4 site B renamed from "memory/discount" to "plasticity gating", measured as recurrent-state autocorrelation τ_h, not as effective γ.
- Concept memo §5.5 4th NO-MAP row in v4: Doya 2002 §3.2 5-HT/γ_Bellman, joining Rodriguez-Garcia (gradient-level), Wainstein (internal-RNN-gain), Osman (Hopfield-attractor) as out-of-substrate.

## Decisions and actions

- v4 absorbed the math-reviewer flag (d): site B renamed plasticity gating.
- v4 added the lineage investigation's recommendation: 5-HT/γ as out-of-substrate (4th NO-MAP).
- v5 carries this forward in §5.9 — γ_Bellman as one of three out-of-substrate channels (with Rodriguez-Garcia gradient-level and Wainstein internal-RNN-gain).
- Future work option B: a hybrid architecture (FiLM + learned-discount-network) IS the route if a future paper wants the γ channel; not in scope for v5.

## Open questions and follow-ups

- Whether the project ever returns to γ_Bellman is a v5+ scoping decision; deferred per [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]].

## References

- Math-reviewer audit (flag d): [`docs/project/concepts/film_neuromod_integration.md`](../../../docs/project/concepts/film_neuromod_integration.md) §10.
- Lineage+critic-modulation investigation: [`docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md`](../../../docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md) §3.
- v5 §5.9 substrate-NOT: [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md`](../../../docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md).
- Related insights: [[20260516_1506_film_hypernet_bhn_bnn_2x2_lineage]] (the lineage frame that ruled this out), [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]] (defer-rule that absorbs this), [[20260513_0017_mod_h_logging_gap_blocks_cka_precheck]] (related substrate flag).
- Key paper: Lee 2024 §6 (drops the 5-HT branch explicitly — precedent for the project's decision).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260516_1506_film_hypernet_bhn_bnn_2x2_lineage]] (nmn_diagnosis, 2026-05-16) — Formal 2×2 lineage of FiLM / Hypernet / Bayesian Hypernet / Bayesian Neural Net 
- [[20260516_1511_math_reviewer_catches_silent_direction_errors]] (subagent_engineering, 2026-05-16) — Math-reviewer audit caught two silent direction-memo errors (Hessian-direction r
<!-- END BACKLINKS -->
