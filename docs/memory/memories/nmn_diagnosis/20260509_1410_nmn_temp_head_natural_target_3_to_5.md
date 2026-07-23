---
id: 20260509_1410_nmn_temp_head_natural_target_3_to_5
date: 2026-05-09
time: "14:10"
folder: nmn_diagnosis
tags: [nmn, hypervigilance, film, learned_lesson]
summary: "MC FiLM temperature head's natural output target sits in [3.0, 5.0), profile-dependently — head wants ~3.0 on P3 (R=5), ~4.7 on P4 (R=9.67), ~4.5 on P5 (R=18). Old [0.5, 3.0] ceiling was binding on P4/P5 (head pinned at 3.0) but NOT on P3 (head naturally settled there). For future NMN configs: temp_clip [0.5, 5.0] minimum to avoid binding at R ≥ 9."
related: ["20260508_2004_profile_dependent_temp_saturation_mc_film", "20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4"]
session_origin: claude_code
session_label: "nmn_tempceil10_rerun_analysis_2026-05-09"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/6bbe7739-79ae-4486-b230-d8b7b8263893.jsonl
raw_completeness: full
---

# MC FiLM temperature head's natural target — `[3.0, 5.0)`, profile-dependent, plateaus past R ≈ 10

## Key conclusion
With `temp_clip` raised from `[0.5, 3.0]` → `[0.5, 10.0]`, the MC FiLM temperature head's natural output target is now visible: it sits in `[3.0, 5.0)`, profile-dependently. The head wants ~3.0 on P3 (R=5), ~4.7 on P4 (R=9.67), and ~4.5 on P5 (R=18) — increasing with R but plateauing past R ≈ 10. This refines yesterday's "saturation kicks in at R ≥ 5" finding: the old `[0.5, 3.0]` ceiling was **binding on P4/P5** (head pinned at 3.0 because it actually wanted ~4.5–4.7) but **NOT on P3** (head naturally settled at ~3.0; the parent sweep's saturation at P3f was incidental, not constrained). The new `[0.5, 10.0]` ceiling is non-binding everywhere — well clear of the head's natural maximum. For future NMN config design, `temp_clip [0.5, 5.0]` is the minimum that avoids binding at R ≥ 9; the canonical FiLM agent config should adopt this default.

## Evidence, measurements, facts
- temp_max per cell at the **new** `[0.5, 10.0]` ceiling (from analyzer extraction `tmp/20260509_nmn_tempceil10_extraction.json`):
  | Cell | R | temp_max_new |
  |---|---:|---:|
  | P3s0 | 5.00 | 3.56 |
  | P3s1 | 5.00 | 2.95 |
  | P3s2 | 5.00 | 2.55 |
  | P3 mean | 5.00 | 3.02 |
  | P4s0 | 9.67 | 4.65 |
  | P5s0 | 18.00 | 4.51 |
- temp_max per cell at the **old** `[0.5, 3.0]` ceiling (parent sweep, from `tmp/20260508_nmn_het_extraction.json`): P1f = 1.50, P2f = 2.30, P3f = 2.96, P4f = 2.99, P5f = 3.00.
- Increment from old → new ceiling per profile:
  - P3: 2.96 → 3.02 (Δ ≈ +0.06; head was NOT actually binding at 3.0 — saturation was incidental).
  - P4: 2.99 → 4.65 (Δ ≈ +1.66; head WAS binding, climbed +55%).
  - P5: 3.00 → 4.51 (Δ ≈ +1.51; head WAS binding, climbed +50%).
- Natural-target curve (R, temp_max_natural): (5, ~3.0), (9.67, ~4.7), (18, ~4.5). Approximately monotonic in R but **plateaus past R ≈ 10** — the head's expressed range bounds at ~4.7, not at the new ceiling 10.0.
- Numerical safety: `src/models/neuromodulator.py:71` default is `(0.1, 10.0)`, so `[0.5, 10.0]` is within the architecture's original numerical envelope. No NaN, no gradient explosion. Max grad_norm at temp_max ≈ 4.7: 0.20 (parent at temp_max ≈ 3.0: 0.16). Headroom up to 10.0 is unused; raising further would not improve the diagnostic.
- Implication for `[0.5, 5.0]` as a candidate ceiling: it would have freed P4/P5 (heads at ~4.65 / ~4.51 are below 5.0 by ~0.35–0.49 — small but unambiguous headroom) without raising instability, but the diagnostic would have looked ambiguous because temp_max would have approached but not cleanly cleared the new ceiling (one cannot disambiguate "head wants 5.0 but is binding" from "head wants 4.65 freely"). `[0.5, 10.0]` was the right design call; `[0.5, 5.0]` is the right operating default.

## Decisions and actions
- For future NMN config design, `temp_clip` should be **`[0.5, 5.0]` minimum** to avoid binding on R ≥ 9 profiles. The parent sweep's `[0.5, 3.0]` is now known to be a misconfiguration that conflated FiLM's structural performance with a hyperparameter artefact at R ≥ 9.
- The canonical FiLM agent config (`configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1.yaml` and any v8-anchored derivative) should adopt `[0.5, 5.0]` as the new default — this is a `senior-developer` scope call (it changes a project default), not analyzer scope. Flag it in the senior-developer hand-off triggered by the sister insight.
- Future R-sweeps that explore profiles with R > 18 should preview the head's expressed temp_max BEFORE locking the ceiling — the `[3.0, 5.0)` plateau may not extend; if R > 50 turns out to push the head past 5.0, the new default would re-bind.

## Open questions and follow-ups
- Why does the head's natural target plateau ~4.7 instead of climbing monotonically with R? Hypothesis: the temperature head saturates the modulator's downstream gating capacity past temp ≈ 5; the increase in heterogeneity past R ≈ 10 is not turned into more dynamic range because the downstream gate cannot use it. Out of scope for this analysis; flag for v9 diagnosis.
- Does P3's flat temp_max (~3.02 mean) explain its high seed-σ (±4.40 in survival)? Plausible: P3 sits exactly at the inflection where the head's natural target equals the old binding ceiling, so initial-condition variance dictates which side of the saturation regime each seed lands in. Worth pre-registering for any P3 single-seed cell going forward — single-seed P3 is **structurally noisy**.
- Should the `[0.5, 5.0]` default also apply to GAE FiLM cells, where the parent sweep showed temp head was differently engaged? Out of scope here; the `[0.5, 5.0]` recommendation is MC-specific until verified on GAE.

## References
- Sister insight this session (verdict): `20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4`.
- Parent mechanism insight (this refines, does NOT supersede): `20260508_2004_profile_dependent_temp_saturation_mc_film`.
- Design doc (this session): `docs/experiments/active/hypervigilance/NMN_TEMP_CLIP_CEILING_RERUN.md`.
- Architecture default location: `src/models/neuromodulator.py:71` (`temp_clip: Tuple[float, float] = (0.1, 10.0)`).
- Canonical FiLM config that should adopt the new default: `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1.yaml` (currently `[0.5, 3.0]`).
- Analyzer working files: `tmp/20260509_nmn_tempceil10_extraction.json`, `tmp/20260509_124500_nmn_tempceil10_analysis.md`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 6bbe7739-79ae-4486-b230-d8b7b8263893` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260513_0017_mod_h_logging_gap_blocks_cka_precheck]] (nmn_diagnosis, 2026-05-13) — The R2 continual probe's modulator-engagement Mahalanobis check is unevaluable b
- [[20260723_1911_rppo_nmn_config_boundary_traps]] (nmn_diagnosis, 2026-07-23) — The rPPO-NMN modulated forward is train-equals-eval correct (probe-verified), bu
<!-- END BACKLINKS -->
