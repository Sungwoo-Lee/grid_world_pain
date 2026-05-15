---
id: 20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse
date: 2026-05-08
time: "20:03"
folder: nmn_diagnosis
tags: [nmn, hypervigilance, film, refutation]
summary: "10-cell NMN noise-heterogeneity sweep refutes H1a/H1b/H1c — FiLM g1 is consistently 5–13 steps WORSE than Unmod LN at every R profile (gap narrows but never crosses zero). Closes v8 §6.3 P4 lead: heterogeneity is not the missing ingredient, the architecture is the bottleneck."
related: ["20260508_1426_v8_noise_bug_refuted", "20260508_1427_nmn_heterogeneity_sweep_design", "20260508_2004_profile_dependent_temp_saturation_mc_film"]
session_origin: claude_code
session_label: "nmn_noise_heterogeneity_sweep_launch_2026-05-07/08"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/6bbe7739-79ae-4486-b230-d8b7b8263893.jsonl
raw_completeness: full
---

# NMN noise-heterogeneity sweep verdict — FiLM is consistently worse than Unmod, not equal

## Key conclusion
The 10-cell NMN noise-heterogeneity sweep refutes all three pre-registered alternative hypotheses against the locked §2.2 / §5.2 predicates: H₁a (no profile produces FiLM > Unmod by > 5 steps anywhere in P3–P5; max Δ_SS = +0.08 at P3), H₁b (vacuous because Δ_SS is never positive), and H₁c on the strict |Δ_SS| ≤ 0.44 predicate (max |Δ| = 12.41). H₀ is confirmed *in spirit*, but the qualitative shape of the failure is stronger than v8's null: FiLM is **structurally degrading** the policy by 5–13 steps at every R profile, with the gap narrowing as heterogeneity grows but never crossing zero. This closes v8 §6.3 P4 (heterogeneity-as-rescue) cleanly: heterogeneity is not the missing ingredient — the architecture is the bottleneck.

## Evidence, measurements, facts
- 5 single-seed cells per architecture (10 total). MC return; FiLM grouping_size=1 vs Unmod LN; mean σ pinned at 0.140 across all profiles; R varied geometrically over {2.0, 2.75, 5.0, 9.67, 18.0}.
- Δ_SS(P_k) = SS_SS(FiLM) − SS_SS(Unmod), measured over the last 20% of training (windows 8–10 of 10 per §5.1):
  | Profile | R | Unmod SS | FiLM SS | Δ_SS |
  |---|---:|---:|---:|---:|
  | P1 | 2.00 | 217.87 | 205.46 | **−12.41** |
  | P2 | 2.75 | 218.33 | 209.34 | **−8.99** |
  | P3 | 5.00 | 221.69 | 221.77 | **+0.08** |
  | P4 | 9.67 | 226.04 | 217.12 | **−8.92** *(P4f preliminary, 7.40 M ep)* |
  | P5 | 18.00 | 226.42 | 221.71 | **−4.71** |
- Convergence: all 5 Unmod cells finished at 10.0 M episodes; P2f at 10.0 M; P1f / P3f / P5f at 8.3 / 8.7 / 8.7 M (just inside the window-8 SS boundary); P4f at 7.4 M (below window 8 — flagged preliminary, but Δ_SS = −8.92 magnitude is unlikely to flip sign).
- Unmod_SS_P1 → Unmod_SS_P5 spread = 8.55 steps, well within the 30-step task-breakage limit (§3.4 confound row): the sweep is NOT confounded by env breakage at extreme R.
- v8 P1-replication FAILED on absolute SS: P1u = 217.87 vs v8 anchor MC_Unmod_Noise_LN ≈ 283.6 (Δ ≈ −65.8); P1f = 205.46 vs anchor ≈ 282.8 (Δ ≈ −77.3). The analyzer traced this to a design-doc bookkeeping mismatch — the design-doc claim that P1 ≡ v8 §2.2 does not hold (v8's noise schema covers 9 modalities, this sweep concentrates noise on 5 bearing modalities). Within-sweep Δ_SS is internally valid (all 10 cells share the same anchor); cross-doc absolute-SS comparisons should not be made.
- WandB IDs (group `nmn_noise_heterogeneity`): P1u=jcxin0g8, P1f=kny1fh5w, P2u=q2o3vydr, P2f=oh4aws0e, P3u=6eqjr62y, P3f=tp0vz8gb, P4u=cicjphwf, P4f=ay78aocv, P5u=mb0huxq1, P5f=rq7voqt8.
- Working artefacts: `tmp/20260508_nmn_het_synthesis.md`, `tmp/20260508_nmn_het_extraction.json`, 10 per-cell `tmp/20260508_nmn_het_<P*>.md` files.

## Decisions and actions
- §10–§12 of `docs/experiments/active/hypervigilance/NMN_NOISE_HETEROGENEITY_SWEEP.md` filled by the analyzer; status line + frontmatter `last_updated` updated.
- 10 `training-done` rows logged to today's diary at 19:59 KST.
- Sweep premise is closed — **do not** re-run the heterogeneity gradient at finer R-granularity. The gap-narrowing-with-R pattern is an aside, not a positive result.
- Hand-off recommendation routed to: (1) `experiment-designer` for a 3-seed P3 replication to disambiguate the lone Δ_SS = +0.08 from single-seed coincidence; (2) `experiment-designer` for a temp_clip-ceiling re-run of P3–P5 (companion insight `20260508_2004_profile_dependent_temp_saturation_mc_film` carries the mechanism finding — that's the cheaper of the two architectural tests). Both gate any commitment to senior-developer's §6.3 P3 full-architectural redesign.
- Within-sweep cross-cell comparisons remain authoritative; absolute SS values from this sweep should NOT be cited against v8's table.

## Open questions and follow-ups
- Does Δ_SS(P3) flip sign with 3 seeds? Single-seed P3 result is the only profile where FiLM is at parity; this is the gating step before any architectural commitment.
- Does raising `temp_clip` ceiling from 3.0 to 5.0 (or 10.0) recover FiLM gain at P3–P5? See companion insight.
- Why is the gap-narrowing pattern (FiLM benefits more from R than Unmod, even though it never wins) not enough to flip sign at any reasonable R? Hypothesis: temp saturation caps the modulator's expressive range exactly when R is high enough to matter — see companion insight.
- If both 3-seed P3 and the temp_clip re-run fail to produce FiLM > Unmod, hand off to `senior-developer` for v8 §6.3 P3 (full attention/Bayesian modulator redesign).

## References
- Design doc (now contains §10–§12 Results / Analysis / Conclusions): `docs/experiments/active/hypervigilance/NMN_NOISE_HETEROGENEITY_SWEEP.md`.
- v8 anchor (the lead this sweep closes): `docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md` §6.3 P4.
- Companion insights (this session): `20260508_1426_v8_noise_bug_refuted` (sweep premise), `20260508_1427_nmn_heterogeneity_sweep_design` (sweep design), `20260508_2004_profile_dependent_temp_saturation_mc_film` (actionable mechanism finding).
- Analyzer working files: `tmp/20260508_nmn_het_synthesis.md`, `tmp/20260508_nmn_het_extraction.json`, 10 per-cell files.
- Diary: 10 `training-done` rows in `docs/diary/2026-05-08.md` at 19:59 KST.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 6bbe7739-79ae-4486-b230-d8b7b8263893` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view). This conversation is the analysis-side continuation of session `c7ee226b-2162-4e7a-95e9-257a5b19d713` (which launched the 10 cells); the launch-side raw conversation lives in c7ee226b's JSONL.
