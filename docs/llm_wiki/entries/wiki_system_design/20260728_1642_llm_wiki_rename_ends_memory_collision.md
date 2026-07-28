---
id: 20260728_1642_llm_wiki_rename_ends_memory_collision
date: 2026-07-28
time: "16:42"
folder: wiki_system_design
tags: [wiki, design, decision, meta]
summary: "The project's in-repo session-insight layer and Claude Code's built-in auto-memory were both called 'memory', forcing a disambiguation every time either came up. Renamed the in-repo layer to the LLM Wiki: docs/memory/ -> docs/llm_wiki/, memories/ -> entries/, /memorize -> /wiki-write, /recall -> /wiki-read, lint_memory + regen_memory_* -> lint_wiki + regen_wiki_*, tag `memory` -> `wiki`. 276 files / 1462 replacements, commit 46f0310. Entry IDs were deliberately NOT renamed."
related: ["20260508_0315_claude_memory_system_genesis", "20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases", "20260728_1643_bulk_rename_component_path_and_domain_term_traps", "20260728_1644_wiki_pull_gate_widened_before_any_task", "20260728_1645_wiki_search_subagent_rejected_on_economics"]
session_origin: claude_code
session_label: "LLM Wiki rename + consult gate"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/dec28250-837d-46e8-b4c9-5ce845ae7698.jsonl
raw_completeness: full
---

# The in-repo memory layer is now the LLM Wiki — one name per layer

## Key conclusion

This project keeps two places to remember things, and until now both were called "memory": Claude Code's own built-in store (one-line rules, machine-local, injected into context every turn) and this repo's version-controlled layer of multi-section session insights. Every conversation about either one had to open with a disambiguation. The in-repo layer is now the **LLM Wiki** at `docs/llm_wiki/`; the built-in store keeps the name "auto-memory" and was not touched. The split rule itself is unchanged — short typed rule to auto-memory, decision-with-rationale to the wiki — only the names moved apart.

## Evidence, measurements, facts

- Scale: 1462 string replacements across 276 files, plus 209 git-tracked renames. Commit `46f0310`.
- Directory: `docs/memory/` -> `docs/llm_wiki/`; `memories/<topic>/` -> `entries/<topic>/`; `memories/memory_system_design/` -> `entries/wiki_system_design/`; `TEMPLATES/insight.md` -> `TEMPLATES/entry.md`.
- Skills: `/memorize` -> `/wiki-write`, `/recall` -> `/wiki-read` (both autocomplete from `/wiki`; the harness re-registered them from the directory rename alone).
- Scripts: `lint_memory.py` -> `lint_wiki.py`, `regen_memory_links.py` -> `regen_wiki_links.py`, `regen_memory_graph.py` -> `regen_wiki_graph.py`.
- Design docs: `claude_memory_system{,_v2}_design.md` -> `llm_wiki_system{,_v2}_design.md`. Tag `memory` -> `wiki`.
- Verification: `regen_wiki_links.py --check` clean, `regen_wiki_graph.py` regenerates, `regen_dev_index.py` reindexes 184 docs, `lint_wiki.py` passes all 11 checks. Its 1 broken wikilink + 29 orphan warnings were confirmed pre-existing at HEAD via `git show`.

## Decisions and actions

- **Entry IDs are immutable and were NOT renamed.** Files like `20260508_0315_claude_memory_system_genesis.md` keep their names even though they contain the old word. The `id` IS the filename stem (see [[20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases]]) and is the target of 326 `[[wikilink]]` edges plus `related:` / `supersedes:` chains. Renaming them would mean rewriting the whole graph to change identifiers the operating manual defines as unique and stable. The historical record keeps its old vocabulary; everything live speaks "wiki".
- **Natural-language trigger phrases were kept.** `/wiki-write` still fires on "remember this", "memorize this", "save to memory" — those are how the user actually talks, and stripping them would trade triggering accuracy for cosmetic purity.
- **The unit noun in machinery is now "entry"** (directory `entries/`, `TEMPLATES/entry.md`); historical bodies that say "insight" were left alone, since "insight" was never the confusing word.
- Two literal quotations were deliberately left un-renamed: a real historical commit subject (`fix(memory): regen_memory_links strip BACKLINKS blocks...`) and the diary note describing the rename. Rewriting a quoted commit message would falsify the record.
- The coexistence rule from [[20260508_0315_claude_memory_system_genesis]] is refined, not superseded — the two-layer split by insight density still holds; only the naming changed.

## Open questions and follow-ups

- Topic-index rows average ~600 bytes each when a scannable index wants ~120; tightening them is queued and would make the whole drill-down ~5x cheaper. See [[20260728_1645_wiki_search_subagent_rejected_on_economics]].
- `docs/llm_wiki/CLAUDE.md` has a pre-existing broken relative link in its header (`../docs/develop/...` resolves to `docs/docs/develop/`); left as-is because it predates this work.

## References

- Commit `46f0310` (the rename), `a2fb405` (diary note).
- Operating manual: `docs/llm_wiki/CLAUDE.md`; design rationale: `docs/develop/active/meta/llm_wiki_system_design.md`.
- Mechanical traps hit during this rename: [[20260728_1643_bulk_rename_component_path_and_domain_term_traps]].
- Consult-gate change made in the same session: [[20260728_1644_wiki_pull_gate_widened_before_any_task]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume dec28250-837d-46e8-b4c9-5ce845ae7698` (re-enter the session) or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260728_1643_bulk_rename_component_path_and_domain_term_traps]] (wiki_system_design, 2026-07-28) — Two traps in a repo-wide mechanical rename, both of which a naive sed would have
<!-- END BACKLINKS -->
