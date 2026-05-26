---
title: Doc-Linking Convention — Foam wikilinks for archive safety
topic: meta
status: active
created: 2026-05-27
last_updated: 2026-05-27
aliases: [doc_linking_convention]
---

# Doc-Linking Convention

## Purpose

This document explains how cross-doc links work in the EPISODE project so that archive moves and file renames do not break navigation. The short version: write new cross-doc links as `[[filename]]` (wikilinks); old `[text](path.md)` links stay valid and do not need bulk rewrites. The viewer tooling — Foam in VSCode, Obsidian in the vault — resolves wikilinks to the target's current location regardless of where it lives, so a `git mv` to `archive/` does not invalidate the link.

The convention exists because the project will archive outdated docs over time, and the prior path-based linking pattern would create many broken links per archive move. Wikilink-based references survive moves by viewer-side resolution rather than by stored paths.

## The convention

### Two layers of stability

Foam (VSCode) and Obsidian (vault) resolve `[[X]]` to a file in two ways. Knowing which layer is in play tells you what stays stable:

| Layer | What `[[X]]` matches | Survives... |
|---|---|---|
| **Filename-as-id** (default) | A file whose filename stem equals `X`. `[[project_plan]]` resolves to `project_plan.md` wherever it lives in the workspace. | File moves (e.g., into `archive/`) — yes. File renames — no. |
| **Frontmatter alias** (opt-in) | A file whose YAML frontmatter contains `aliases: [..., X, ...]`. Matched in addition to the filename. | File moves AND file renames — yes. |

For most docs, filename-as-id is enough. Aliases are a small upgrade for the handful of docs whose filename might evolve.

### What to write

- **New cross-doc links** — prefer `[[filename]]` over `[text](path.md)`. Example: write `[[NEUROMODULATION_ALGORITHM]]` instead of `[NEUROMODULATION_ALGORITHM](../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)`. Both Ctrl-click the same way in Foam, but the wikilink form survives archive moves.
- **Old `[text](path.md)` links** — leave them. They still work. Convert only when you are editing the file for other reasons. Bulk conversion adds noise to git history without earning anything.
- **Docs whose filename might change later** — add `aliases: [<stable-id>]` in YAML frontmatter. Use a short, kebab-case or snake_case identifier (`project_direction`, `environment_summary`). Then any `[[<stable-id>]]` reference resolves stably even if the filename changes.

### What NOT to do

- **Do not bulk-migrate** every existing doc to wikilink form. The convention is incremental — old links are not bugs.
- **Do not add aliases to every doc.** Only the small set of high-traffic docs warrants the extra frontmatter (see below).
- **Do not use opaque IDs** (UUIDs, hashes) for aliases. Use readable identifiers so they are greppable and memorable.
- **Do not break filename uniqueness.** If two files in the workspace share a filename stem (e.g., two `phase_1.md` in different folders), Foam cannot unambiguously resolve `[[phase_1]]` and will prompt you to pick. Keep filenames unique enough across the doc tree to make wikilinks unambiguous.

## Archive workflow

When archiving an outdated doc:

1. **Open the doc in VSCode.**
2. **Check Foam's backlinks panel** — every file linking to this one is listed. This is the blast-radius view.
3. **Decide per inbound link:**
   - Wikilink already (`[[filename]]`)? — nothing to do; it resolves to the new location after the `git mv`.
   - Old `[text](path.md)` link? — three choices:
     - Rewrite to `[[filename]]` (best — survives any future move).
     - Update the path to point at `archive/<filename>` (works but brittle to future moves).
     - Leave broken (acceptable if the referencing doc is itself outdated and slated for archive).
4. **`git mv <doc> docs/<tree>/archive/`.**
5. **(Optional, for develop-tree docs)** Set `status: superseded` and add `supersedes:` / `superseded_by:` in the frontmatter — see [FRONTMATTER_CONTRACT.md](FRONTMATTER_CONTRACT.md). This is the existing lifecycle pattern; the linking convention does not replace it.

After the move, Foam re-indexes on the next save and `[[filename]]` references resolve to the new path.

## Load-bearing docs that warrant aliases

A small set of docs are referenced from many places across the project. Locking these down with an alias makes future renames safe. For each, add an `aliases:` entry to its YAML frontmatter:

| Doc | Suggested alias | Why |
|---|---|---|
| `docs/project/project_plan.md` | `project_direction` | Referenced from ~20 agent profiles; may eventually be renamed (e.g. drop "plan") |
| `docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md` | `neuromodulation_algorithm` | Frequently linked across project memos and professor outputs |
| `docs/environment/ENVIRONMENT_SUMMARY.md` | `environment_summary` | Foundational reference; may get reorganised as the env evolves |
| `docs/develop/active/meta/FRONTMATTER_CONTRACT.md` | `frontmatter_contract` | Governance doc; referenced in workflow instructions |
| `docs/memory/CLAUDE.md` | `memory_operating_manual` | Memory layer governance |
| `docs/AGENT_PLAYBOOK.md` | `agent_playbook` | Cross-cutting workflow doc; referenced by the agent-manager |

This list is not exhaustive. Add aliases to any doc that you expect to be linked from many places AND whose name might change in the future. Most docs do not need aliases.

## Tooling

- **Foam (VSCode extension)** — runtime wikilink resolver. Provides Ctrl-click navigation, backlinks panel, graph view, orphans panel. No build step required.
- **Obsidian (vault viewer)** — alternate viewer for the same vault (the project has `docs/.obsidian/`). Same wikilink + alias semantics. Optional; not required for the convention to work.
- **No pre-render or link-rewrite script.** The convention relies entirely on viewer-side resolution. There is no `regen_doc_links.py` to run; Foam re-indexes on file save. This is intentional — fewer moving parts than a pre-commit rewrite hook.

## Relationship to other doc conventions

- **`docs/memory/`** — already uses `[[YYYYMMDD_HHMM_slug]]` wikilinks where the id IS the filename stem. This convention extends the same idea to other doc trees where filenames are human-readable. The memory tree's `regen_memory_links.py` / `regen_memory_graph.py` scripts are memory-specific and do not run on other trees.
- **[FRONTMATTER_CONTRACT.md](FRONTMATTER_CONTRACT.md) (develop tree)** — governs `title, topic, status, created, last_updated` for `docs/develop/` files. The `aliases:` field added by this convention is an *optional* addition to that contract, not a replacement of any existing field. A develop-tree doc may add `aliases:` without changing any other frontmatter behavior.
- **`docs/TEMPLATES/`** — templates (`issue_plan.md`, `training_analysis.md`) do not currently show wikilink examples. They can be updated opportunistically as authors touch them; not blocking for this convention.
- **CLAUDE.md "Project-Wide Rules"** — carries a single one-line bullet pointing at this document. That bullet is the entry point future-Claude reads every session. Mechanism-level detail lives here, not in CLAUDE.md.

## Resolving link conflicts in Foam

If `[[X]]` is ambiguous (two files have stem `X`), Foam shows a quick-pick. Two ways to fix:

- **Rename one of the files** to make the stem unique. Cheapest fix; preferred when the duplicate is accidental.
- **Use an alias on the intended target** and reference via that alias. Use this when both filenames are legitimately the same word in different contexts (rare).
