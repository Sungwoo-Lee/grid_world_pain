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
| 2026-05-09 | 15:55 | [20260509_1555_dreamer_v3_diagnosis](20260509_1555_dreamer_v3_diagnosis.md) | DreamerV3 diagnosis | Diagnostic battery + probe battery + conventional-fixes battery + offline WM test (4 + 4 + 2 cells + 1 inference-time test, 2026-05-07 → 2026-05-09) |
| 2026-05-09 | 15:52 | [20260509_1552_sameprop_rabbit_avoidance_study](20260509_1552_sameprop_rabbit_avoidance_study.md) | SameProp rabbit-avoidance | Survey + R1 relog (n=2 seeds) + R2 partial (Cells C+A1, SIGINT'd at 0.4M ep) + per-tag metrics ship (2026-05-07 → 2026-05-09) |
| 2026-05-09 | 14:21 | [20260509_1421_nmn_comparison_study](20260509_1421_nmn_comparison_study.md) | NMN comparison | Heterogeneity sweep + temp-clip rerun (10 + 5 cells, 2026-05-07 → 2026-05-09) |

## Conventions

- **English only**; no emojis unless the user explicitly asks.
- **Repo-relative links** so the document is portable across clones.
- **Plain-language framing** — assume the reader has not seen the design docs. Translate symbolic predicate names (H₀, H₁a, etc.) to plain English on first mention.
- **Timestamped filename, not just timestamped frontmatter** — a directory listing should already tell the reader when each summary was generated.
- **Append-only**: re-summarising a study after new experiments land = write a fresh file with a new timestamp; old summaries stay as historical snapshots.
