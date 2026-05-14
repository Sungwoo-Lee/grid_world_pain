# Summaries — multi-experiment study reports

> One file per study (typically a multi-experiment thread completed within days or weeks): `docs/experiments/summaries/YYYYMMDD_HHMM_<slug>.md`. Each file is a high-level reader-facing report that links out to the authoritative design docs, analyses, and memory insights.

## Purpose

Where `docs/experiments/active/` carries the **detailed design + results** for each individual experiment, and `.claude-memory/memories/` carries the **per-finding multi-section insights**, this folder carries **study-level summaries**:

- A single document that names every experiment in a study (e.g., "the NMN comparison study"), the question each one tested, what was found, and what changed about our understanding.
- Written for a reader who has **not seen the detailed plans** — explanations are in plain language, no `H₁a / H₁b / H₁c` jargon without translation, no bare WandB IDs, no bare config paths.
- Always **timestamped in the filename** so a reader can tell at a glance when the summary was generated and which window of work it covers.

The pattern matches `docs/diary/README.md` in spirit: this folder is an **index for the reader**, not a duplicate of the source documents.

## Filename convention

```
YYYYMMDD_HHMM_<slug>.md
```

- `YYYYMMDD_HHMM` = generation timestamp (Asia/Seoul), to the minute. Matches the `.claude-memory/` insight filename convention.
- `<slug>` = English snake_case, ≤ ~6 words, identifies the study (e.g., `nmn_comparison_study`, `dreamer_v3_diagnosis`, `hypervigilance_round3`).

The timestamp anchors the summary to a specific moment in the project's history. Re-running a summary on the same study after new experiments land is encouraged — write a new file with a fresh timestamp; do **not** edit an old summary in place. The index below shows which summaries exist; older summaries remain readable as snapshots of what we knew at the time.

## What goes inside each summary

Recommended sections (mirrored across summaries for scanability):

1. **Study question** — one paragraph, plain English, what the study is trying to answer.
2. **Experiments completed** — a table with one row per experiment: name + question + manipulation + finding + what it changed. High-level interpretation, not numbers-heavy.
3. **Where this leaves us** — bullets summarising the cumulative state of understanding.
4. **What's next** — open follow-ups + their owner agent (e.g. `experiment-designer`, `senior-developer`).
5. **Links** — every authoritative document the summary points at: design docs in `../active/<topic>/`, analyses, memory insights in `../../../.claude-memory/memories/<topic>/`, working files in `../../../tmp/`, diary days in `../../diary/`. **Repo-relative paths** so links stay valid on any clone.

A summary that is missing the Links section is broken — the whole point is to be a reader-facing entry point that fans out to the full record.

## What's NOT here

- The full design + results for each individual experiment → `docs/experiments/active/<topic>/<doc>.md`.
- Per-finding insights with rationale and rejected alternatives → `.claude-memory/memories/<topic>/<id>.md`.
- Implementation plans → `docs/develop/active/<topic>/<plan>.md`.
- Daily event log (start/end of session, training-start/done, insight rows) → `docs/diary/YYYY-MM-DD.md`.

## Index of summaries (newest first)

| Date | Time | File | Study | Scope |
|---|---|---|---|---|
| 2026-05-14 | 23:32 | [20260514_2332_sameprop_rabbit_avoidance_study](20260514_2332_sameprop_rabbit_avoidance_study.md) | SameProp rabbit-avoidance (reader-friendly re-summary + R2.6 crash correction) | Reader-friendly rewrite of the 2026-05-13 re-summary, leading with a 6-bullet take-home-messages section. Same two-level verdict (spatial: class-blind; event: +37 pp bush-dive gap, opposite-direction eat-suppression). **Corrects Round 2.6 status**: both runs (Cell C s44, Cell A1 s45) crashed at ~6.6 h on n106 (1.18 M / 0.87 M episodes vs 10 M target, `state=failed` in WandB, within 3 min of each other → likely node-level event). Seed-lock not yet earned; re-launch on a healthy node is the immediate next step (2026-05-07 → 2026-05-14) |
| 2026-05-13 | 14:20 | [20260513_1420_sameprop_rabbit_avoidance_study](20260513_1420_sameprop_rabbit_avoidance_study.md) | SameProp rabbit-avoidance (re-summary — two-level verdict) | Adds the §12 toolkit-appendix event-level verdict to the prior spatial verdict: under matched smells the agent is **not** class-discriminating at the mean-distance level but **is** at the event level (+37 pp bush-dive gap, 0.75× / 1.19× eat-suppression opposite direction near predator vs rabbit, per-tag fan-out within noise). Includes Round 2.6 in-flight snapshot at 29% elapsed: Cell C-s44 replicating the +28.5 pp gap (still climbing); Cell A1-s45 converging to a "stay-and-eat" policy rather than corner-camping, vindicating the pre-launch call that A1 needed a second seed too (2026-05-07 → 2026-05-13). **Caveat (added post-hoc):** Round 2.6 had already crashed by the time of this query; the "in-flight" claim is incorrect — see [20260514_2332](20260514_2332_sameprop_rabbit_avoidance_study.md). |
| 2026-05-13 | 03:21 | [20260513_0321_nmn_comparison_study](20260513_0321_nmn_comparison_study.md) | NMN comparison (re-summary — vitality probe verdict) | Adds the schedule-changing regime verdict: under a 5-stage active↔passive continual schedule, the modulator clearly beats the baseline (+107 / +132 steps on the two return-to-active stages, ~25× seed-noise floor) — first clearly-positive FiLM finding. Catastrophic-forgetting + reusable-subnetwork hypotheses confirmed; "half-the-dip" predicate inconclusive (schedule-asymmetric). Specialist ceiling table (6 worlds, 10M ep each) refutes "swap is load-bearing" framing for upcoming meta head-to-head; reframes as factorisation test. Three logging gaps surfaced (raw modulator hidden vector / predator-Term / occupancy histograms) (2026-05-07 → 2026-05-13) |
| 2026-05-11 | 15:59 | [20260511_1559_dreamer_v3_fix_cascade](20260511_1559_dreamer_v3_fix_cascade.md) | DreamerV3 fix cascade (Z2 verdict) | Adds Z2 verdict (H2 partial, cumulative cascade −54%) + diagnostic-script silent-bug patch + §6 empirical-resolution update. Candidate #1 (GRU reset gate) now queued as next, selected by long-horizon-compounding residual pattern (2026-05-07 → 2026-05-11) |
| 2026-05-10 | 22:55 | [20260510_2255_dreamer_v3_fix_cascade](20260510_2255_dreamer_v3_fix_cascade.md) | DreamerV3 fix cascade | Continuation of `dreamer_v3_diagnosis` — sheeprl reference-impl comparison + zero-init reward+critic fix (Z1, finished, H2 partial) + paper-canonical twohot bins fix (Z2, in flight). 1 doc-side event + 1 finished cell + 1 in-flight cell, 2026-05-07 → 2026-05-10 |
| 2026-05-10 | 22:53 | [20260510_2253_sameprop_rabbit_avoidance_study](20260510_2253_sameprop_rabbit_avoidance_study.md) | SameProp rabbit-avoidance (re-summary) | Adds Round 2.5 verdict (10M ep, n=1 each cell): both hypotheses refuted, no genuine class-conditional avoidance under matched smells; provisional pending Round 2.6 seed 44 (2026-05-07 → 2026-05-10) |
| 2026-05-09 | 15:55 | [20260509_1555_dreamer_v3_diagnosis](20260509_1555_dreamer_v3_diagnosis.md) | DreamerV3 diagnosis | Diagnostic battery + probe battery + conventional-fixes battery + offline WM test (4 + 4 + 2 cells + 1 inference-time test, 2026-05-07 → 2026-05-09) |
| 2026-05-09 | 15:52 | [20260509_1552_sameprop_rabbit_avoidance_study](20260509_1552_sameprop_rabbit_avoidance_study.md) | SameProp rabbit-avoidance | Survey + R1 relog (n=2 seeds) + R2 partial (Cells C+A1, SIGINT'd at 0.4M ep) + per-tag metrics ship (2026-05-07 → 2026-05-09) |
| 2026-05-09 | 14:21 | [20260509_1421_nmn_comparison_study](20260509_1421_nmn_comparison_study.md) | NMN comparison | Heterogeneity sweep + temp-clip rerun (10 + 5 cells, 2026-05-07 → 2026-05-09) |

## Conventions

- **English only**; no emojis unless the user explicitly asks.
- **Repo-relative links** so the document is portable across clones.
- **Plain-language framing** — assume the reader has not seen the design docs. Translate symbolic predicate names (H₀, H₁a, etc.) to plain English on first mention.
- **Timestamped filename, not just timestamped frontmatter** — a directory listing should already tell the reader when each summary was generated.
- **Append-only**: re-summarising a study after new experiments land = write a fresh file with a new timestamp; old summaries stay as historical snapshots.
