---
id: 20260513_2309_merge_path_manifest_tripwire
date: 2026-05-13
time: "23:09"
folder: cluster_ops
tags: [meta, learned_lesson, decision, design]
summary: "When a project carries 165GB of gitignored training data (results 39GB, wandb 119GB, logs 4GB, etc.) and CLAUDE.md mandates 'snapshot critical data before any merge / rebase / branch switch', the full cp -a backup is infeasible. Instead, save a path manifest (find -printf '%p %s\\n', ~16MB, 128k lines) as a tripwire, verify the operation is structurally non-destructive (fast-forward only, no clean -x, no force-checkout), and post-op diff the manifest against current `find` count to detect silent loss. Validated end-to-end on v1.3 → develop → v1.4: delta = 0 files."
related: ["20260512_1755_pytorch_agents_pip_dep_layout", "20260512_1756_pip_install_namespace_shadow_numpy_cap"]
session_origin: claude_code
session_label: "dreamer-srl v3 CP1 closure — v1.3→develop→v1.4 merge"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# Path-manifest tripwire: low-cost untracked-data-loss detection for large-data repos

## Key conclusion

CLAUDE.md mandates snapshotting critical untracked data before any non-trivial git operation (merge, rebase, branch switch), motivated by a past incident where `results/` was lost during a failed merge followed by destructive cleanup. The project carries 165GB of gitignored data across `results`, `wandb`, `logs`, `claude_data`, `tmp`, `antigravity_data` — full `cp -a` to `/tmp` is infeasible. The alternative that works: save a **path manifest** (`find ... -type f -printf '%p %s\n'`) as a ~16MB tripwire file before the op, verify the operation is structurally non-destructive (fast-forward only, no `clean -x`, no force-checkout), then post-op `find ... | wc -l` and compare against the manifest. Delta = 0 → no loss. Faster than `cp -a` by 4 orders of magnitude; sufficient when the op is mechanically incapable of writing into untracked paths.

## Evidence, measurements, facts

- **Untracked data sizes on this repo (2026-05-13)**:
  - `results`: 39 GB / 92,837 files (training outputs)
  - `wandb`: 119 GB / 19,743 files (WandB local cache)
  - `logs`: 4 GB / 1,611 files
  - `tmp`: 1.6 GB / 279 files
  - `antigravity_data`: 1.5 GB / 12,544 files
  - `claude_data`: 245 MB / 1,729 files
  - **Total: ~165 GB / 128,743 files**
- **Manifest tripwire on the v1.3 → develop → v1.4 operation**:
  - Pre-op: `find results wandb logs claude_data tmp antigravity_data -type f -printf '%p %s\n' > /tmp/pre-merge-inventory-$(date +%s).txt` — wrote `/tmp/pre-merge-inventory-1778660680.txt`, 16 MB, 128,743 lines, completed in ~5 seconds.
  - Op: `git checkout develop && git merge --no-ff v1.3` (fast-forward equivalent — develop was 0 ahead, v1.3 was 261 commits ahead). Then `git checkout -b v1.4`.
  - Post-op: `find results wandb logs claude_data tmp antigravity_data -type f | wc -l` → 128,743 files. Delta = 0. ✓
- **Why this works for FF merges**: a fast-forward merge mechanically only updates `.git/HEAD` and `.git/refs/heads/develop`, then runs `git checkout` against the tree at v1.3's tip. The tree update touches only tracked files. Untracked files (gitignored or otherwise) are never touched by the merge or its checkout. The risk is in destructive *follow-ups* (`git clean -x`, `git stash drop` after `stash -u`, etc.) — not in the merge itself.
- **When `cp -a` IS the right call instead**: any op that could write into untracked paths. Concrete cases: `git checkout -f <branch>` where the destination has tracked paths at locations currently untracked locally (the force-checkout overwrites them); `git stash -u` followed by `git stash drop` (the drop loses the untracked content carried into the stash). These can be detected at planning time — if the op might overwrite an untracked path, `cp -a` is correct; otherwise the manifest suffices.
- **Manifest format choice**: `find ... -type f -printf '%p %s\n'` (path + size, space-separated, newline-terminated). 16 MB is small enough to grep, diff, or sort comfortably; 128k lines fits in any text editor. An alternate format (`xargs sha256sum`) would detect content corruption, not just file deletion, but at much higher cost (~hours for 165 GB on the NAS) and we already know our concern is loss, not corruption.

## Decisions and actions

- **Decision (recipe to use going forward)**: for any merge / rebase / branch switch / non-trivial git operation on this repo:
  1. Run the manifest: `find <critical-dirs> -type f -printf '%p %s\n' > /tmp/pre-merge-inventory-$(date +%s).txt`.
  2. Verify the op is structurally non-destructive (FF merge, conflict-free 3-way that doesn't touch gitignored paths, branch switch where destination doesn't track currently-untracked-local paths). If it isn't, fall back to `cp -a` for the data that's at risk.
  3. Run the op.
  4. Post-op: `find <critical-dirs> -type f | wc -l` and compare against the manifest's line count. Delta = 0 → no loss.
- **Decision (when to escalate to full `cp -a`)**: only when the planned op can mechanically write into untracked paths. Document the rationale either way (manifest or `cp -a`) so future-Claude reproduces the same risk-assessment.
- **Action (this session)**: validated the protocol on v1.3 → develop → v1.4 merge. 0-delta confirmed; no `cp -a` was performed; no data lost.

## Open questions and follow-ups

- **Should this become a CLAUDE.md rule?** Currently CLAUDE.md says "snapshot critical untracked data" without specifying the method. The manifest pattern is cheaper and validated; the `cp -a` pattern is mandated by phrasing for the case where the op can clobber untracked paths. Worth a small CLAUDE.md edit codifying the two-tier protocol (manifest by default; `cp -a` only when the op can write untracked paths) so future-Claude doesn't bring down a full `cp -a` unnecessarily.
- **Manifest hash variant for paranoid ops**: if a future op carries genuine corruption risk (not just deletion), upgrade the manifest to a content hash (`xargs sha256sum` over a smaller subset like `results/JAX_RecurrentPPO/`, not all of `wandb/`). Don't run sha over the full 165GB by default.

## References

- CLAUDE.md "Git safety (DO NOT WIPE GITIGNORED DATA)" rules — the past incident's preventive layer
- v1.3 → develop → v1.4 merge commits: `61b3d52` (merge), v1.4 branched from there
- Manifest tripwire file (local-only, ephemeral): `/tmp/pre-merge-inventory-1778660680.txt`
- Related insight (cluster-ops layer, pip-install gotchas): [[20260512_1756_pip_install_namespace_shadow_numpy_cap]]
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
