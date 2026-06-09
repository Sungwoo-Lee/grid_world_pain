---
id: 20260509_1620_documentation_framing_policy
date: 2026-05-09
time: "16:20"
folder: memory_system_design
tags: [design, decision, meta]
summary: "Established a project-wide Documentation framing policy in CLAUDE.md: every plan/design/analysis/summary/review/direction doc must lead with a plain-language entry-point section a reader without prior context can follow. No bare WandB IDs, config paths, or predicate names in the entry-point. 16 files updated in commit 62f96b9 (CLAUDE.md + 2 templates + 13 doc-producing agent profiles); first retro-application to v2 synthesis (commit 5c5adf3) added a 350-word TL;DR. The /summarize-study skill is the worked example."
related: ["20260508_0315_claude_memory_system_genesis", "20260509_1619_summarize_study_skill_design_and_ship"]
session_origin: claude_code
session_label: "nmn_meta_continual_pivot_2026-05-09"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/6bbe7739-79ae-4486-b230-d8b7b8263893.jsonl
raw_completeness: full
---

# Documentation framing policy — plain-language entry point on every plan doc

## Key conclusion
After the user observed that a research-synthesis memo was effectively unreadable without the prior memos in front of them ("I don't follow your whole plan as there are some jargons"), the project adopted a formal **Documentation framing policy** as a new section in `CLAUDE.md`. The rule: every plan / design / analysis / summary / review / direction document must lead with a plain-language entry-point section (Question / Purpose / Context / Headline / Verdict / equivalent) that a reader without prior context can follow. The first ~200 words are the load-bearing surface; symbolic / numerical / file-path / equation-heavy detail is allowed in later sections. Concretely: translate every cited result on first mention ("the modulator did not beat the baseline (H₁a refuted)", not just "H₁a refuted"); no bare WandB run IDs, no bare config paths, no bare predicate names in the entry-point. The check: open the document, read the first 200 words; could a fresh reader who has not seen the prior memos understand what this doc is about, why it exists, and what it's claiming? The policy was promoted from one skill (`/summarize-study`) to project-wide by adding a CLAUDE.md section + updating 2 doc templates + adding a one-line reference to 13 doc-producing agent profiles, all in commit `62f96b9`. The first retro-application landed at `5c5adf3` — added a 350-word TL;DR to the v2 synthesis memo, which previously opened with `Δ-1, Δ-2, Candidate A1, P3, Δ_CKA, T/P split` in its first 30 words.

## Evidence, measurements, facts
- Policy commit: `62f96b9 docs(policy): 📚 documentation-framing policy — plain-language entry point on every plan/design/analysis doc` (16 files changed, +115/-5 lines).
- First retro-application commit: `5c5adf3 docs(project): 📚 add plain-language TL;DR to nmn_meta_continual_synthesis_v2` (1 file changed, +134 insertions — the `_v2` synthesis was uncommitted at the time, so this also captured the body the postdoc had written earlier).
- Worked example the policy points at: the `/summarize-study` skill (`.claude/skills/summarize-study/SKILL.md`, commit `2b52c08`) — its `## Plain-language enforcement` section is the strictest form of the rule, with a concrete check-and-replace table.
- 16 files in the policy commit, by tier:
  - Canonical home: `CLAUDE.md` — new `## Documentation framing` section between `## Project-Wide Rules` and `## Session memory`.
  - Templates: `docs/TEMPLATES/issue_plan.md` (Context section reframed) + `docs/TEMPLATES/training_analysis.md` (Research Question reframed; H₀/H₁ example now translates the symbol on first mention).
  - 13 doc-producing agent profiles each gain a top-level `## Documentation framing` subsection cross-linking back to `CLAUDE.md`: `senior-developer`, `experiment-designer`, `experiment-analyzer`, `research-postdoc`, 4 professors (`bayesian-brain`, `pain-modeling`, `rl-bayesian-dl`, `neuromodulation`), 3 reviewers (`code-reviewer`, `math-reviewer`, `env-config-auditor`), 2 literature agents (`literature-reviewer`, `literature-curator`).
- Tiers explicitly NOT applied: Tier 3 (skills `memorize` / `diary` / `summarize-study` SKILL.md cross-links — the policy already references `summarize-study` in the other direction) and Tier 4 (retro-update existing docs that violate the rule — high churn for unclear value).
- Retro-application worked-example metric: v2 synthesis's first 30 body-words went from 100% jargon (`Two specific deltas on the meta side; continual unchanged. Δ-1 — The meta task scales from 2 to 6 contexts. v1's Candidate A1 was a single 2-context mix...`) to 0% jargon — new TL;DR opens with `The project trains an AI agent in a small grid world to survive — find food, avoid predators, manage hunger.`

## Decisions and actions
- **Canonical home**: `CLAUDE.md` new section (recommended). Single home for the rule; agent profiles + templates cross-reference.
- **Sweep scope**: Tier 2 (recommended) — templates + 13 doc-producing agent profiles. NOT Tier 3 (skills) and NOT Tier 4 (retro-sweep of existing docs). The policy applies going forward; existing docs are not retroactively rewritten unless a reader is actively confused by one.
- **Word target**: ~200 words for the entry-point section. Soft target; the v2 synthesis's TL;DR landed at 349 words after trimming and the user accepted it (the doc has 7 distinct ideas to introduce; ~50 words per idea is reasonable).
- **The policy is forward-only**: existing docs that violate the rule are not retro-fixed, except when a reader is actively confused by one. The v2 synthesis was retro-fixed because the user explicitly flagged it ("I don't follow your whole plan as there are some jargons").
- **Concrete fix pattern**: when retro-fixing an existing doc, insert a `## TL;DR — What this memo is about` section BEFORE the existing first heading — keeps existing section numbering intact, no churn to internal cross-references.

## Open questions and follow-ups
- Tier 3 sweep (skills cross-link) was deferred. If the policy becomes friction-bearing in skill outputs (e.g., `/memorize` insights themselves violate the rule in their `## Key conclusion` sections), revisit.
- Tier 4 retro-sweep of existing docs is deferred. The next time a reader is actively confused by an existing doc, retro-apply per the v2 synthesis pattern (insert TL;DR before existing first heading).
- The 5 other research-chain docs from the NMN meta/continual pivot session (`triage/20260509_1517_...`, `directions/nmn_meta_context_conditioning.md` and `_v2`, `directions/nmn_continual_lifelong_probe.md`, `ideas/nmn_meta_continual_synthesis.md`) are still uncommitted and also fail the policy. User pending decision: retro-apply TL;DRs to all of them, or commit as-is and retro-fix only when confused.
- Validation: the next time the `summarize-study` skill is invoked, check whether the policy is being followed in the auto-generated summary. The skill's own `## Plain-language enforcement` table is the canary.

## References
- Policy text: `CLAUDE.md` "Documentation framing" section (after "Project-Wide Rules", before "Session memory").
- Worked example: `.claude/skills/summarize-study/SKILL.md` `## Plain-language enforcement` (the strictest form).
- Templates: `docs/TEMPLATES/issue_plan.md`, `docs/TEMPLATES/training_analysis.md`.
- First retro-application: `docs/project/ideas/nmn_meta_continual_synthesis_v2.md` `## TL;DR` section (commit `5c5adf3`).
- Companion insight (this session): `20260509_1619_summarize_study_skill_design_and_ship` — the skill that motivated promoting the rule project-wide.
- Genesis insight: `20260508_0315_claude_memory_system_genesis` — the original `docs/memory/` design, which this policy generalises to all doc-producing surfaces.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 6bbe7739-79ae-4486-b230-d8b7b8263893` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260516_1431_v2_three_role_architecture]] (memory_system_design, 2026-05-16) — Memory System v2 separates code/memory knowledge into three surfaces — curated s
- [[20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases]] (memory_system_design, 2026-05-28) — Project-wide doc-linking convention adopted: [[filename]] wikilinks for new cros
- [[20260609_1725_env_docs_tutorial_primer_pattern]] (memory_system_design, 2026-06-09) — Pattern for turning reference docs into a tutorial set: re-sync to code-as-truth
<!-- END BACKLINKS -->
