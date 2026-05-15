---
id: 20260508_2004_profile_dependent_temp_saturation_mc_film
date: 2026-05-08
time: "20:04"
folder: nmn_diagnosis
tags: [nmn, hypervigilance, film, learned_lesson]
summary: "MC FiLM temp_max saturates at the temp_clip 3.0 ceiling for P3f/P4f/P5f (R ≥ 5) but stays at 1.50–2.30 for P1f/P2f. Saturation is profile-dependent at R ≈ 5; raising temp_clip ceiling and re-running P3–P5 is the minimum-viable architectural test before any FiLM redesign."
related: ["20260508_1427_nmn_heterogeneity_sweep_design", "20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse"]
session_origin: claude_code
session_label: "nmn_noise_heterogeneity_sweep_launch_2026-05-07/08"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/6bbe7739-79ae-4486-b230-d8b7b8263893.jsonl
raw_completeness: full
---

# Profile-dependent MC FiLM temperature saturation — temp_clip ceiling becomes the bottleneck at R ≥ 5

## Key conclusion
The "MC temp head saturates at 3.0 ceiling" failure mode flagged in v8 §4.3.3 is **profile-dependent** in this heterogeneity sweep: FiLM cells P1f / P2f stay at temp_max ≈ 1.50 / 2.30 (well below the ceiling), but P3f / P4f / P5f all hit temp_max 2.96–3.00 (saturated). The transition kicks in at R ≈ 5. The temperature head is structurally pinned by `temp_clip: [0.5, 3.0]` exactly when heterogeneity should be giving it the most signal to differentiate. This is the **actionable mechanism finding** from the sweep: the minimum-viable architectural test is to raise `temp_clip` ceiling (e.g. 5.0 or 10.0) and re-run P3–P5 single-seed before committing to any structural FiLM redesign — if FiLM gains there, the v8 + this-sweep null reduces to a hyperparameter choice, not an architectural failure.

## Evidence, measurements, facts
- Per-cell `temp_max` over the last 20% of training, from analyzer extraction (`tmp/20260508_nmn_het_extraction.json`):
  | Cell | R | temp_max | Saturated? |
  |---|---:|---:|---|
  | P1f | 2.00 | 1.50 | No |
  | P2f | 2.75 | 2.30 | No |
  | P3f | 5.00 | 2.96 | Yes (~99 % of ceiling) |
  | P4f | 9.67 | 2.99 | Yes |
  | P5f | 18.00 | 3.00 | Yes (at ceiling) |
- v8 §4.3.3 already flagged this as a structural concern and recommended raising the ceiling. This sweep confirms the recommendation has empirical traction at R ≥ 5; v8's data point was at R ≈ 2 (P1-equivalent) where saturation is NOT present, which is presumably why the v8 recommendation lacked direct in-doc evidence.
- Saturation is a soft-cap effect, not a numerical-stability issue: no NaN, no gradient explosion, no Term_Starvation > 90 % anywhere in the FiLM cells.
- Bound location: `configs/models/recurrent_ppo_nmn_het_film_g1.yaml` has `temp_clip: [0.5, 3.0]`; the agent code reads it as a hard sigmoid clip on the temperature head's output.
- The saturation correlates with the gap-narrowing pattern in the sister insight: as R rises, FiLM temp saturates AND the FiLM-vs-Unmod gap narrows from −12.41 to −4.71. Plausible reading: heterogeneity is giving the temperature head signal, the head wants to push temperature higher, but the ceiling pins it — leaving residual gating capability on the table.

## Decisions and actions
- Before any structural FiLM redesign, run the minimum-viable hyperparameter test:
  - `experiment-designer` to author a 3-cell P3–P5 single-seed re-run with `temp_clip: [0.5, 10.0]` (or [0.5, 5.0]). Approximately 30 h on 2 GPUs, low risk.
  - Pre-registered branch:
    - If FiLM gains > 5 steps on any of P3–P5 with the raised ceiling → v8 + this-sweep null was a hyperparameter artefact; document and move on.
    - If FiLM still trails Unmod → ceiling is not the bottleneck; route to `senior-developer` for full §6.3 P3 architectural redesign per v8.
- This experiment runs in parallel with the 3-seed P3 replication from the sister insight; both are gating steps before the senior-developer redesign.

## Open questions and follow-ups
- Does temp_max scale continuously past the ceiling when raised, or hit a new equilibrium below the new ceiling? If the former, FiLM is using the freed range; if the latter, the head was previously *artificially* binding and 3.0 was just enough.
- What is the relationship between temp_max and gamma_multi_std at the new ceiling? Sweep §5.4 metric `gamma_multi_std` should also rise if the temperature head is genuinely engaging differential per-channel weighting.
- Should `mod_hidden_size: 16` be raised in tandem? The hypothesis-only path is "ceiling is the binder", but `mod_hidden_size` could be a parallel binder; out of scope for the minimum-viable test but worth a follow-up if the ceiling test alone underwhelms.
- Fold this finding into the next NMN_PERFORMANCE_DIAGNOSIS doc (v9?) before any architectural commitment — v8 §4.3.3 needs an update with the R-dependence empirical confirmation.

## References
- Sister insight (sweep verdict): `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse`.
- Design doc §6 ("MC temp saturates at 3.0 ceiling on every FiLM cell" — partial trigger): `docs/experiments/active/hypervigilance/NMN_NOISE_HETEROGENEITY_SWEEP.md`.
- v8 anchor (originally flagged this failure mode): `docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md` §4.3.3, §6.3.
- Config to edit for the test: `configs/models/recurrent_ppo_nmn_het_film_g1.yaml` (`temp_clip` field).
- Analyzer working files: `tmp/20260508_nmn_het_extraction.json`, `tmp/20260508_nmn_het_synthesis.md`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 6bbe7739-79ae-4486-b230-d8b7b8263893` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view). Analysis-side continuation of launch session `c7ee226b-2162-4e7a-95e9-257a5b19d713`.
