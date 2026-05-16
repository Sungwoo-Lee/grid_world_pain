---
id: 20260509_1619_summarize_study_skill_design_and_ship
date: 2026-05-09
time: "16:19"
folder: memory_system_design
tags: [memory, design, skill, decision]
summary: "Added a third documentation layer to the project: docs/experiments/summaries/ for study-level summaries (where docs/experiments/active/ has per-experiment design+results, and docs/memory/ has per-finding insights). First executed manually (commit d075fdc, NMN comparison study summary), then promoted to a /summarize-study skill (commit 2b52c08) that automates the flow on demand. Same append-only versioning + timestamped-filename + auto-commit pattern as /memorize."
related: ["20260508_0429_memorize_skill_design_and_ship", "20260508_0447_recall_skill_design_and_ship", "20260509_1620_documentation_framing_policy"]
session_origin: claude_code
session_label: "nmn_meta_continual_pivot_2026-05-09"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/6bbe7739-79ae-4486-b230-d8b7b8263893.jsonl
raw_completeness: full
---

# Documentation framing — added `docs/experiments/summaries/` + shipped `/summarize-study`

## Key conclusion
The project gained a third documentation layer this session: `docs/experiments/summaries/` for **study-level reader-facing summaries**. Where `docs/experiments/active/<topic>/` carries per-experiment detailed design + results and `docs/memory/memories/<topic>/` carries per-finding multi-section insights with rationale, the new layer is for a single document that names every experiment in a coherent multi-experiment thread (e.g., "the NMN comparison study"), explains what each one tested, what was found, and what changed about understanding — written for a reader who has not seen the prior plans. The flow was first executed manually for the NMN comparison study (commit `d075fdc`, file `20260509_1421_nmn_comparison_study.md`), validated by the user, then promoted to a Claude Code skill at `.claude/skills/summarize-study/SKILL.md` (commit `2b52c08`) that automates the flow on demand. The skill's design decisions — append-only versioning (re-summaries write fresh dated files, do NOT overwrite older ones), timestamped filenames matching the `docs/memory/` convention `YYYYMMDD_HHMM_<slug>.md`, hybrid auto-discovery with user confirmation, plain-language enforcement in §§1-4 with all symbolic detail in §5 Links, and auto-commit + diary-note bundling like `/memorize` Step 9 — are all directly inherited from the manually-executed worked example.

## Evidence, measurements, facts
- Folder created: `docs/experiments/summaries/` with `README.md` (folder operating contract) + first summary `20260509_1421_nmn_comparison_study.md` (covers the heterogeneity sweep + temp-clip rerun this week).
- Folder commit: `d075fdc docs(experiments): 📚 add summaries/ folder + first NMN-study summary` (3 files, +162 lines).
- Skill commit: `2b52c08 feat(skills): ✨ add /summarize-study skill` (1 file, 279 lines under .claude/skills/summarize-study/SKILL.md).
- Skill's `description` frontmatter is the harness's trigger surface; explicit-only invocation per user choice ("/summarize-study", "summarize this study", "write me a study report"). Distinct-from clauses in the description name 4 neighbouring layers: `/memorize` (per-finding rationale), `/diary` (short-timeline events), `experiment-designer` (per-experiment design docs), `experiment-analyzer` (per-experiment results).
- Skill's flow: Step 1 read README + index; Step 2 hybrid scope discovery (auto-detect ANALYZED/COMPLETE design docs + matching memory insights, show user, confirm); Step 3 timestamped filename + slug; Step 4 write summary with locked 6-section structure (§1 Study question / §2 Experiments table / §3 Where this leaves us / §4 What's next / §5 Links / §6 Reading order); Step 5 prepend row to README index; Step 6 diary `note` row; Step 7 auto-commit by name (3 files: summary, README, diary).
- Worked-example evidence: `docs/experiments/summaries/20260509_1421_nmn_comparison_study.md` is the canonical example the skill points at; opening it shows what good output looks like.
- Test status: not formally evaled (no `evals/evals.json`). The manually-executed worked example IS the canary — the skill's job is to reproduce that flow on demand.

## Decisions and actions
- **Folder name**: chose `summaries` (recommended) over `reports` / `digests` / `studies`. Plain English; matches the existing `active` / `meta` siblings under `docs/experiments/`.
- **Filename pattern**: `YYYYMMDD_HHMM_<slug>.md` (recommended) over `YYYY-MM-DD_<slug>.md` / range-encoded / slug-first. Matches `docs/memory/` insight convention; sortable; minute-granular.
- **Append-only re-summaries**: locked in. Older summaries are historical snapshots of "what we knew on date X"; never overwritten. Every re-summary writes a fresh dated file.
- **Auto-commit + diary-note bundling**: yes, mirroring `/memorize` Step 9. The skill bundles the summary file + the README index update + the diary note row into one focused commit with `docs(experiments)` scope.
- **Scope discovery**: hybrid — user names the study; skill auto-discovers experiments from `docs/experiments/active/<topic>/` + `docs/memory/memories/<topic>/`; always confirms the auto-detected list before writing.
- **Trigger model**: explicit-only (recommended). Matches `/init`, `/security-review` style — does not auto-fire after experiment-analyzer completion. The user invokes when they want a summary.

## Open questions and follow-ups
- The skill has no formal eval suite (`evals/evals.json`). The next time `/summarize-study` is invoked, compare its output against the manually-executed worked example (commit `d075fdc`) and check whether the auto-discovered experiment list matches what a human would pick. If divergence is consistent, write an eval suite then.
- Tier 3 of the documentation-framing sweep (`memorize`, `diary`, `summarize-study` skill files) was deferred per user choice; if revisited, this skill's `## Plain-language enforcement` rule should be cross-linked back to `CLAUDE.md "Documentation framing"` (the policy the skill helped motivate — see companion insight `20260509_1620_documentation_framing_policy`).
- Re-summarisation policy is set to "append fresh", not "supersede + archive" like `docs/develop/`. Consider whether a future `summaries/` doc should mark predecessors with `supersedes:` frontmatter for retrieval (probably yes once the index gets long).

## References
- Manually-executed first summary: `docs/experiments/summaries/20260509_1421_nmn_comparison_study.md` (commit `d075fdc`).
- Folder operating contract: `docs/experiments/summaries/README.md`.
- Skill: `.claude/skills/summarize-study/SKILL.md` (commit `2b52c08`).
- Companion skill insights: `20260508_0429_memorize_skill_design_and_ship`, `20260508_0447_recall_skill_design_and_ship`, `20260509_0311_diary_auto_session_backfill` — same family, same design pattern.
- Companion policy insight (this session): `20260509_1620_documentation_framing_policy` — the project-wide rule the skill's strict plain-language enforcement was promoted to.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 6bbe7739-79ae-4486-b230-d8b7b8263893` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260516_1431_v2_three_role_architecture]] (memory_system_design, 2026-05-16) — Memory System v2 separates code/memory knowledge into three surfaces — curated s
<!-- END BACKLINKS -->
