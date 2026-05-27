---
id: 20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases
date: 2026-05-28
time: "02:16"
folder: memory_system_design
tags: [memory, design, decision, meta]
summary: "Project-wide doc-linking convention adopted: [[filename]] wikilinks for new cross-doc references (Foam-resolved, survives moves); aliases: [<id>] in frontmatter only on ~10–20 god-nodes (survives renames); old [text](path.md) links kept, no bulk migration."
related: ["20260509_1620_documentation_framing_policy", "20260528_0215_foam_excludes_required_not_search_exclude"]
session_origin: claude_code
session_label: "EPISODE project_plan rewrite + Foam wikilink convention + VSCode startup fix"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/e386e7cb-3b2e-4604-abc2-921d1b6c973a.jsonl
raw_completeness: full
---

# Doc-linking convention — Foam wikilinks, filename-as-id default, sparse aliases on god-nodes

## Key conclusion

Cross-doc references across the EPISODE project (everywhere except `docs/memory/`, which already had this) now default to `[[filename]]` wikilink form, resolved at runtime by Foam in VSCode (and Obsidian in the vault). The filename stem IS the id — no separate `id:` frontmatter field needed; every doc is already wikilink-resolvable on installation. Aliases (`aliases: [<stable-id>]` in YAML frontmatter) are an opt-in upgrade applied only to ~10–20 load-bearing "god-node" docs whose filenames might evolve. Old `[text](path.md)` links are kept; no bulk migration. This is the same `[[id]]` pattern already used in the memory tree, generalised to other doc trees with human-readable filenames.

## Evidence, measurements, facts

- Two resolution layers established:
  - **Filename-as-id** (default, covers every doc automatically): `[[NEUROMODULATION_ALGORITHM]]` resolves to `NEUROMODULATION_ALGORITHM.md` wherever it lives in the workspace. Survives `git mv` to `archive/` automatically (Foam re-indexes on the filesystem event). Does NOT survive a rename of the filename.
  - **Frontmatter alias** (opt-in, sparse): `aliases: [project_direction]` on `docs/project/project_plan.md` lets `[[project_direction]]` resolve via Foam's alias-aware match. Survives both moves AND renames as long as the alias entry stays.
- 6 load-bearing docs got aliases in this session (commit `bb522d4`): `project_plan.md → project_direction`, `NEUROMODULATION_ALGORITHM.md → neuromodulation_algorithm`, `ENVIRONMENT_SUMMARY.md → environment_summary`, `FRONTMATTER_CONTRACT.md → frontmatter_contract`, `docs/memory/CLAUDE.md → memory_operating_manual`, `AGENT_PLAYBOOK.md → agent_playbook`. For three of these (ENVIRONMENT_SUMMARY, memory/CLAUDE.md, AGENT_PLAYBOOK), a minimal YAML frontmatter block was added since none existed before.
- Convention doc written to `docs/develop/active/meta/doc_linking_convention.md` (commit `e2453e8`, 94 lines). One-line bullet added to project root `CLAUDE.md` under "Project-Wide Rules" pointing at the convention doc.
- The convention deliberately AVOIDS adding aliases to every doc: aliases are a "published API" — free to add, expensive to remove (every `[[old_alias]]` in the repo would break). Namespace collision risk also grows with size: with 300+ docs, duplicate aliases turn Foam from a resolver into a disambiguation-prompter.
- Tooling that already exists is sufficient — no script work needed:
  - Foam (VSCode extension) builds its index from filenames + frontmatter `aliases:` on cold start; re-indexes on file change. No pre-render step.
  - `scripts/regen_memory_links.py` and `scripts/regen_memory_graph.py` already handle the memory tree's wikilinks; the broader doc trees rely entirely on Foam's runtime resolution.
- The convention mirrors the project's existing pattern of "policy promoted from one layer to all doc-producing surfaces" — same shape as the documentation-framing rule's promotion (see [[20260509_1620_documentation_framing_policy]]).

## Decisions and actions

- New cross-doc links: prefer `[[filename]]` over `[text](path.md)`. Old path-based links remain valid; convert only when touching the file for other reasons. No bulk-migration commit allowed.
- Aliases added only to docs that are BOTH referenced from many places AND likely to be renamed. The 6 added this session are the starter set; future additions are case-by-case as god-nodes emerge.
- VSCode extension stack to install: Foam (`foam.foam-vscode`) is the primary resolver; Markdown All in One, Markdown Preview Enhanced, markdownlint as companions. Foam's setting `foam.files.ignore` MUST be configured on this NAS-backed workspace to avoid 5-minute cold starts — see [[20260528_0215_foam_excludes_required_not_search_exclude]].

## Open questions and follow-ups

- Whether to write a dedicated `scripts/regen_doc_links.py` (analog to `regen_memory_links.py`) for the broader doc tree. Decision: not now — Foam's runtime resolution is sufficient, and adding a pre-commit hook would create unnecessary friction. Revisit only if GitHub web-rendering of wikilinks becomes a load-bearing requirement (currently it isn't — Foam users see resolved links; GitHub web users see literal `[[X]]` text, which is acceptable for an internal-team workflow).
- A `docs/` reorganisation pass to clean outdated documents is queued (the "dreaming" reorganisation plan). The convention is the prerequisite; the broken-link audit + script-driven repair come next.

## References

- Convention doc: [`doc_linking_convention.md`](../../../develop/active/meta/doc_linking_convention.md) — has the full archive workflow and god-node candidate list.
- CLAUDE.md bullet: project root, "Project-Wide Rules" section, "Doc linking — Foam wikilinks" line.
- Commits: `e2453e8` (convention doc + CLAUDE.md bullet), `bb522d4` (6 god-nodes aliased), `4416c13` (Foam excludes that make this convention practical at cold-start time).
- Prior policy-promotion pattern: [[20260509_1620_documentation_framing_policy]]
- Companion Foam-performance insight: [[20260528_0215_foam_excludes_required_not_search_exclude]]
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume e386e7cb-3b2e-4604-abc2-921d1b6c973a` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260528_0215_foam_excludes_required_not_search_exclude]] (cluster_ops, 2026-05-28) — Foam doesn't honor VSCode's search.exclude; needs its own foam.files.ignore. Add
- [[20260528_0217_episode_direction_4x3_framework_two_papers]] (nmn_diagnosis, 2026-05-28) — project_plan.md rewritten from 760-line Nature MI staged plan to 138-line stable
<!-- END BACKLINKS -->
