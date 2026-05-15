---
id: 20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4
date: 2026-05-09
time: "14:09"
folder: nmn_diagnosis
tags: [nmn, hypervigilance, film, refutation]
summary: "5-cell temp_clip[0.5,10.0] re-run confirms H1b on P4 (improvement +6.37 over old-ceiling P4f, but FiLM still trails Unmod by −2.55); H1a/H0/H1c refuted; P3 3-seed disambiguation refuted (σ=±4.40, parent's +0.08 was noise). Reframes parent verdict: 'FiLM-worse everywhere' was PARTIALLY a hyperparameter artefact, not purely structural — but freeing the ceiling is not sufficient for FiLM to overtake Unmod."
related: ["20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse", "20260508_2004_profile_dependent_temp_saturation_mc_film", "20260509_1410_nmn_temp_head_natural_target_3_to_5"]
session_origin: claude_code
session_label: "nmn_tempceil10_rerun_analysis_2026-05-09"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/6bbe7739-79ae-4486-b230-d8b7b8263893.jsonl
raw_completeness: full
---

# NMN temp_clip[0.5, 10.0] re-run verdict — H₁b confirmed on P4; ceiling was partial binder, not full structural cause

## Key conclusion
The 5-cell temp_clip-ceiling re-run (P3 × 3 seeds, P4 × 1 seed, P5 × 1 seed; all 10.0M ep, clean) confirms **H₁b on P4** (improvement +6.37 steps over the old-ceiling P4f, FiLM still trails Unmod by −2.55) and produces a **borderline P5 result** (improvement +7.99, Δ_SS_new = +3.28, BELOW the H₁a +5 threshold). H₁a, H₀, and H₁c are refuted by their pre-registered predicates. The P3 3-seed disambiguation is also **refuted** — only 1/3 seeds positive, σ across seeds = ±4.40, so the parent sweep's lone P3f Δ_SS = +0.08 was single-seed coincidence. The verdict reframes yesterday's "FiLM-worse everywhere" pattern: at the original `[0.5, 3.0]` ceiling, FiLM appeared structurally degenerate at R ≥ 5; at `[0.5, 10.0]`, FiLM closes most of the gap on P4 and turns P5 slightly positive. The original null was **partially a hyperparameter artefact** at R ≥ 9, but freeing the ceiling is **not sufficient** for FiLM to overtake Unmod on any profile.

## Evidence, measurements, facts
- Per-cell results (windows 8–10 of 10):
  | Cell | SS_FiLM_new | Δ_SS_new | Δ_SS_old | Δ_new − Δ_old | temp_max_new |
  |---|---:|---:|---:|---:|---:|
  | P3s0 | 222.63 | +0.94 | +0.08 | +0.86 | 3.56 |
  | P3s1 | 212.59 | −9.10 | +0.08 | −9.18 | 2.95 |
  | P3s2 | 221.03 | −0.66 | +0.08 | −0.75 | 2.55 |
  | **P3 mean (n=3)** | 218.75 ± 4.40 | **−2.94** | +0.08 | −3.02 | 3.02 |
  | P4s0 | 223.49 | −2.55 | −8.92 | **+6.37** | 4.65 |
  | P5s0 | 229.71 | +3.28 | −4.71 | **+7.99** | 4.51 |
- Reused parent-sweep Unmod baselines (no re-run): P3u `6eqjr62y` = 221.69, P4u `cicjphwf` = 226.04, P5u `mb0huxq1` = 226.42.
- Reused old-ceiling FiLM baselines: P3f `tp0vz8gb` = 221.77, P4f `ay78aocv` = 217.12, P5f `rq7voqt8` = 221.71.
- Predicate verdicts (from §1.1 of `NMN_TEMP_CLIP_CEILING_RERUN.md`, locked pre-launch):
  - **H₀ REFUTED** — head DOES engage the new headroom (P4 temp_max 4.65, P5 4.51) AND P4/P5 produce > +5 improvement.
  - **H₁a REFUTED** — no profile clears Δ_SS_new > +5 (P5's +3.28 is closest; P3 mean −2.94, P4 −2.55).
  - **H₁b CONFIRMED on P4** (improvement +6.37, FiLM still ≤ Unmod). P5 nearly qualifies (+7.99 improvement) but Δ_SS_new is positive there, not just narrowed.
  - **H₁c REFUTED** — temp_max climbs +1.65 on P4, +1.51 on P5 above old ceiling.
  - **P3 3-seed disambiguation REFUTED** — only 1/3 seeds positive; parent sweep's +0.08 was noise.
- All 5 cells reached 10.0M ep cleanly. Max grad_norm 0.20 (≪ 1.0); max Term_Starvation 0.674 (≪ 0.90); no NaN. No §5 failure modes fired.
- WandB run IDs: P3s0=`f96lhxpe`, P3s1=`hfhop3qu`, P3s2=`cwnvye4m`, P4s0=`dq8yhmlp`, P5s0=`hgvionh6`. Group `nmn_temp_clip_ceiling`.

## Decisions and actions
- §9–§11 of `NMN_TEMP_CLIP_CEILING_RERUN.md` filled by analyzer; status → COMPLETE; `last_updated: 2026-05-09`.
- 5 `training-done` rows logged to diary at 12:45 KST (in 2026-05-08.md, where the corresponding `training-start` rows live).
- **Hand-off recommendation**: `senior-developer` for v8 §6.3 P3 architectural redesign, with the explicit constraint **"any redesign must not regress P4's +6.37 and P5's +7.99 partial gains; temperature head should NOT be the redesign target — its natural output is `[3.0, 5.0)` and the 10.0 ceiling is non-binding."** (Companion insight `20260509_1410_nmn_temp_head_natural_target_3_to_5` carries the head-target detail.)
- **Parallel high-priority follow-up**: `experiment-designer` to author a 3-seed P4 + P5 replication at `temp_clip [0.5, 10.0]` (6 cells, ~36 cell-hr). P3's σ = ±4.40 across 3 seeds is the concrete warning that single-seed Δ_SS values can be ±5 off; P4 and P5's single-seed numbers (currently anchoring the redesign) need replication before the redesign anchors on them.
- Updates yesterday's parent-sweep verdict insight (`20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse`): the "FiLM-worse everywhere" pattern was partially ceiling-driven, not purely structural. That insight stands for its pre-rerun premise but its forward-looking recommendations (§6.3 P3 redesign) need the constraints above.

## Open questions and follow-ups
- 3-seed P4 + P5 replication at the new ceiling — gating step before any redesign anchoring on the +6.37 / +7.99 numbers.
- After replication: senior-developer for §6.3 P3 redesign with the (verified) P4/P5 gains as performance constraints.
- Update `NMN_PERFORMANCE_DIAGNOSIS_v8` doc — supersede the "MC FiLM is structurally degenerate at R ≥ 5" finding with the more nuanced "old `[0.5, 3.0]` ceiling was profile-dependently binding on P4/P5; with `[0.5, 10.0]` FiLM closes most of the gap on P4/P5 but does not overtake. P3 remains anomalous at high seed-σ".
- P3 specifically remains a problem — even at the new ceiling, P3 mean is −2.94 with high seed variance (±4.40). Why does P3 (R=5, where saturation just kicks in) misbehave more than P4/P5 with a higher ceiling? Hypothesis (testable): P3 is precisely where the temp head's natural target ~3.0 sits at the inflection between the old binding ceiling and the new free range; initial-condition variance dominates which side of the saturation regime each seed lands in. Worth pre-registering for the 3-seed replication round.

## References
- Design doc (now contains §9–§11 Results / Analysis / Conclusions, status COMPLETE): `docs/experiments/active/hypervigilance/NMN_TEMP_CLIP_CEILING_RERUN.md`.
- Parent-sweep verdict insight (this insight refines, does NOT supersede — that one stands for its pre-rerun premise): `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse`.
- Parent mechanism insight (this refines): `20260508_2004_profile_dependent_temp_saturation_mc_film`.
- Companion insight this session (head natural target detail): `20260509_1410_nmn_temp_head_natural_target_3_to_5`.
- v8 anchor: `docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md` §4.3.3, §6.3 P3.
- Analyzer working files: `tmp/20260509_124500_nmn_tempceil10_analysis.md`, `tmp/20260509_nmn_tempceil10_extraction.json`.
- Diary: 5 `training-done` rows in `docs/diary/2026-05-08.md` at 12:45 KST.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 6bbe7739-79ae-4486-b230-d8b7b8263893` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view). Same session as the launch-side and analysis chain begun on 2026-05-08 19:30 KST.
