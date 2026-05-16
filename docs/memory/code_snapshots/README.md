# Code-graph snapshots

Point-in-time records of the structural state of `src/` at moments worth remembering. Live code graph (always current) lives at gitignored `src/graphify-out/`; snapshots here are immutable, dated commits worth keeping. Captured via `python scripts/snapshot_code_graph.py <label>`.

## Snapshots

| Captured | Label | Source commit | Stats | Link |
|---|---|---|---|---|
| 2026-05-16 13:22 | v2_memory_build_complete | 0dab341 | 735 nodes / 1009 edges / 59 communities | [snapshot](20260516_1322_v2_memory_build_complete.md) |

## Why snapshot?

Typical triggers for capturing a snapshot:

- **Major refactors** — before or after a significant architectural change so you can diff "before" vs "after" structurally.
- **Ship milestones** — at a version tag or release cut (`v1.4_ship`, `v2.0_ship`) to mark the code state that went out.
- **Pivot moments** — "this is what we threw away when we pivoted to JAX Dreamer" — the abandoned architecture preserved for reference.
- **Architecturally significant moments** — any point where future-you would benefit from asking "what did the codebase look like then?"

Not every commit needs a snapshot. The live graph at `src/graphify-out/GRAPH_REPORT.md` (regenerable on demand via `python scripts/regen_code_graph.py`) is the right surface for "what does the code look like *now*." Snapshots are for "what did it look like *then*."

## Usage

```bash
# Capture a snapshot (regenerates the live graph, then snapshots it)
python scripts/snapshot_code_graph.py v1_4_ship

# Skip re-running graphify (use the already-generated report)
python scripts/snapshot_code_graph.py v1_4_ship --no-regen

# Scan the full repo instead of just src/
python scripts/snapshot_code_graph.py pre_jax_dreamer_pivot --scope all

# Preview without committing
python scripts/snapshot_code_graph.py test_snapshot --no-commit
```

## Snapshot file format

Each snapshot is a Markdown file with a YAML frontmatter header followed by the verbatim `GRAPH_REPORT.md` content from graphify:

```
---
snapshot_label: <label>
captured: YYYY-MM-DD HH:MM
source_commit: <short-sha>
scope: src|all
graphify_version: <version>
session_id: <session-id or unknown>
---

# Code-graph snapshot — `<label>`
...
```

Snapshots are **immutable** once written — never edited, only superseded by newer snapshots. The `/memorize` skill optionally surfaces a snapshot candidate when the session involves 5 or more changed files in `src/`.
