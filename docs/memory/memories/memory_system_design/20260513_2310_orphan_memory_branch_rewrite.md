---
id: 20260513_2310_orphan_memory_branch_rewrite
date: 2026-05-13
time: "23:10"
folder: memory_system_design
tags: [memory, design, learned_lesson, decision, meta]
summary: "When a worktree branch carrying a single .claude-memory insight commit falls behind its base by N commits and another session edits the same docs/memory/ index files (ROOT_INDEX counts, _topic_index rows, _global_tags history) in the meantime, the merge produces synthetic conflicts on the index files only — the insight body itself is usually unique on the orphan branch. Rather than resolve the synthetic index-count conflicts by hand, the cleaner pattern is to rewrite the insight on the current base: copy the insight body via `git show <orphan-branch>:<path>`, regenerate the indexes from the CURRENT counts on the base, drop the orphan branch. Costs ~5 minutes; produces linear history; preserves the insight content; the orphan commit hash survives in the reflog as 30-day safety net."
related: ["20260508_0429_memorize_skill_design_and_ship", "20260508_0430_worktree_isolation_path_safety", "20260509_0311_diary_auto_session_backfill"]
session_origin: claude_code
session_label: "dreamer-srl v3 CP1 closure — JAXVectorEnv insight orphan branch resolution"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# Orphan-memory-branch rewrite vs merge for synthetic docs/memory/ index conflicts

## Key conclusion

Background sessions write `docs/memory/` captures through ephemeral worktree branches. If the worktree falls behind its base while parallel sessions capture other insights on the base, the merge back produces conflicts only on the **index** files (`ROOT_INDEX.md` total-insight count, `<topic>/_topic_index.md` count + row table, `_global_tags.md` change history) — the **insight body file itself** is normally unique on the orphan branch (a brand-new ID, no collision with parallel captures). The conflict markers are entirely in synthetic numbers (47 vs 52 total insights, 7 vs 8 dreamer_diagnosis insights, etc.) which can be regenerated mechanically from the current base state. The cheaper resolution is: copy the insight body verbatim onto the current base via `git show <orphan-branch>:<path>`, update indexes with the CURRENT counts (not the conflict-marker counts), commit fresh, drop the orphan branch. ~5 minutes; linear history; insight content preserved; orphan commit survives in reflog for 30 days as a safety net.

## Evidence, measurements, facts

- **Concrete instance**: the `memorize-jaxvec-v1v2-v2` worktree branch carried one commit (`36921aa`) capturing the JAXVectorEnv v1+v2 spike closure into `docs/memory/memories/dreamer_diagnosis/20260513_1417_jax_vmap_no_speedup_tiny_env.md`. While that worktree was in use, the parent v1.3 branch advanced by ~20 commits, 5 of which were other `docs/memory/` captures (NMN R2 continual + 6-specialist analyzer verdict session, etc.). When the merge was attempted via `git merge-tree`, the conflict surface was:
  - `docs/memory/ROOT_INDEX.md`: total-insight count 48 (orphan: 47 + my 1) vs 52 (base: 47 + others' 5). Per-folder rows also conflicted on count + date.
  - `docs/memory/memories/dreamer_diagnosis/_topic_index.md`: count line + change-history block.
  - `docs/memory/memories/_global_tags.md`: change-history block.
  - The insight body file itself (`20260513_1417_jax_vmap_no_speedup_tiny_env.md`) was **unique to the orphan branch — no conflict**.
- **Rewrite procedure used** (and validated):
  1. `git show memorize-jaxvec-v1v2-v2:docs/memory/memories/dreamer_diagnosis/20260513_1417_jax_vmap_no_speedup_tiny_env.md > docs/memory/memories/dreamer_diagnosis/20260513_1417_jax_vmap_no_speedup_tiny_env.md` on the current base (v1.3 at `289ea54`). Verified content match via `git diff v1.3 memorize-jaxvec-v1v2-v2 -- docs/memory/memories/dreamer_diagnosis/20260513_1417_jax_vmap_no_speedup_tiny_env.md` returning empty.
  2. Updated `ROOT_INDEX.md` with the current counts (52 → 53 total; `dreamer_diagnosis` 7 → 8; `Last update` 2026-05-13). Wrote a change-history entry at the TOP.
  3. Updated `dreamer_diagnosis/_topic_index.md`: count 7 → 8, inserted new row at the top of the "Insights (newest first)" table, change-history block at TOP.
  4. Updated `_global_tags.md`: appended change-history line at TOP (no new tags promoted — all reused).
  5. Appended diary `insight` row via `scripts/diary_append.py`.
  6. Single commit (`3178249`). Branch + worktree later cleanly removed via `git worktree remove --force` + `git branch -D`.
- **Outcome**: ~5 minutes of work, clean linear history, no synthetic conflict markers polluting the index files, orphan commit `36921aa` survives in reflog for 30 days.

## Decisions and actions

- **Decision (default pattern for memory-layer orphan branches)**: when the orphan branch's only value-add is a unique insight body file (or a small set of them) and the conflicts are entirely on the synthetic index files, rewrite on the current base rather than merge. The criterion is: *does the merge's conflict surface include any file that BOTH branches semantically edited (vs. mechanically incremented)?* If yes, fall back to a real merge resolution. If no (just index counts on both sides), rewrite.
- **Decision (preserved insight, lost commit hash)**: the orphan-branch commit hash is sacrificed (no `cherry-pick`, no merge commit referencing it), but the reflog keeps it for 30 days as a recovery net. Anyone referencing the orphan hash externally before the merge happens would need to chase it through reflog; nobody currently does, so the loss is theoretical.
- **Action (this session)**: applied the rewrite on the JAXVectorEnv-closure insight; orphan branch + worktree dropped cleanly post-merge.

## Open questions and follow-ups

- **Should `/memorize` skill detect and warn about behind-base worktree state at capture time?** If the skill reads ROOT_INDEX.md (per Step 1) and the worktree's `git rev-parse v1.3` differs from the worktree branch's merge-base parent, it could warn "your worktree is N commits behind v1.3; capturing now will likely produce synthetic index-count conflicts at merge time — consider rebasing first or accepting that the merge will use the rewrite pattern". Low priority; the rewrite pattern works fine.
- **Convention to write/update across all worktrees simultaneously?** Currently each worktree writes its own `docs/memory/` state independently and merges later. An alternative is to have the skill write into the shared checkout's `docs/memory/` regardless of which worktree invoked it — but that defeats worktree isolation and risks parallel-session races. The current per-worktree-write + later-merge model is correct; the rewrite pattern is the cheap conflict resolution.

## References

- v1.3 commit that landed the rewritten insight: `3178249` (docs(memory): 📚 capture JAXVectorEnv v1+v2 spike closure insight on v1.3)
- Insight file (single source of truth on v1.3 / develop / v1.4 now): [docs/memory/memories/dreamer_diagnosis/20260513_1417_jax_vmap_no_speedup_tiny_env.md](../../../docs/memory/memories/dreamer_diagnosis/20260513_1417_jax_vmap_no_speedup_tiny_env.md)
- Orphan-branch commit (now in reflog only): `36921aa` (was `memorize-jaxvec-v1v2-v2`)
- Related insight (memorize skill design rationale): [[20260508_0429_memorize_skill_design_and_ship]]
- Related insight (diary auto-backfill — similar parallel-session-coordination pattern): [[20260509_0311_diary_auto_session_backfill]]
- Related insight (worktree-isolation path-safety origins): [[20260508_0430_worktree_isolation_path_safety]]
- `docs/memory/CLAUDE.md` operating manual sections 5 (insight template), 6 (fragmentation safeguards), 7 (raw archive — not relevant here, but the layer's overall doctrine)
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260516_1431_v2_three_role_architecture]] (memory_system_design, 2026-05-16) — Memory System v2 separates code/memory knowledge into three surfaces — curated s
- [[20260516_1510_worktree_misses_post_branch_main_assets]] (subagent_engineering, 2026-05-16) — Background-session worktrees branched from origin/main miss lit-review assets th
<!-- END BACKLINKS -->
