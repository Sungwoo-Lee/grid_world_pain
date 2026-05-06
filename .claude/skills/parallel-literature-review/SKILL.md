---
name: parallel-literature-review
description: Parallel academic literature review for large corpora (typically 10+ papers) by sharding the corpus across multiple literature-reviewer agent instances and merging the results. Use this skill when the user explicitly asks for parallel processing of many papers — phrases like "review these 30 papers in parallel", "split this corpus across reviewers", "speed up the lit review by parallelizing", "I have a deadline, can you do these papers in parallel". For small or medium corpora (under ~10 papers), use a single literature-reviewer instance instead — the parallel overhead is not worth it. Make sure to confirm corpus size and use the parallel skill only when the user has signaled they want parallelism.
---

# Parallel Literature Review

This skill shards a large corpus across multiple `literature-reviewer` agent instances, runs them concurrently, then merges their outputs into a single master review document.

It is **not** the default literature review path — for small to medium corpora, use a single `literature-reviewer` instance. Parallelism has real overhead (sharding, coordination, merging), and the speed-up only pays off when the corpus is large.

## When to Use

- The corpus is large enough that sequential processing would be slow (rough threshold: ~10+ papers, but use judgment).
- Papers are independent — no paper's review depends on another's.
- The user has explicitly requested parallelism, OR the corpus size makes it the obviously sensible choice.

## When NOT to Use

- The corpus is small (under ~10 papers). Single `literature-reviewer` is faster end-to-end.
- The papers form a tightly coupled narrative (e.g., a sequence of papers from one lab building on each other) where reviewer N's understanding of paper N benefits from having read paper N-1. Sequential review preserves this context.
- The user wants thematic synthesis across papers — the merge step in this skill produces a paper-by-paper master doc; thematic regrouping is a separate task best handled afterward by `senior-developer` reading the merged output.

## The Five Phases

```
[1] Plan the shard       — Claude decides corpus, shard size, output paths
[2] User confirms        — quick checkpoint on the shard plan
[3] Spawn N reviewers    — parallel literature-reviewer instances
[4] Each reviewer writes  to its own shard file
[5] Merge into master    — single combined doc with auto-generated TOC
```

### Phase 1 — Shard Planning

Decide:

- **Corpus enumeration.** If the input is a directory of PDFs, glob `*.pdf` and list every file. If the input is a NotebookLM link, query for the paper list.
- **Shard count (K).** Default heuristic: `K = ceil(N / 5)` papers per shard, capped at 6 shards total. (Each shard processes its papers sequentially; cap on K prevents excessive coordination overhead.)
- **Shard assignment.** Distribute papers across K shards as evenly as possible. Order does not matter unless the user has specified one.
- **Output paths.** Each reviewer writes to `tmp/lit_review_shard_<i>.md`. Final merged doc goes to `docs/literature/<topic>.md` (or a path the user specifies).

Save the shard plan to `tmp/YYYYMMDD_HHMMSS_lit_shard_plan.md`.

### Phase 2 — User Confirmation

Show the user:
- Total paper count.
- Number of shards and shard size (e.g., "5 shards of ~6 papers each").
- Output paths.
- Estimated wall-clock benefit (rough — "should be ~Kx faster than sequential").

Wait for go-ahead. If the user says "actually just do it sequentially," fall back to a single `literature-reviewer`.

### Phase 3 — Spawn N Reviewers

Spawn K instances of the `literature-reviewer` agent **in a single message** (parallel tool calls). Each reviewer receives:

- Its assigned paper subset (explicit list).
- Its shard output path (`tmp/lit_review_shard_<i>.md`).
- A note that it is shard `i` of `K` and should write its TOC and section anchors with shard-prefixed IDs to avoid collisions when merged.
- The standard literature-review workflow (4-step backbone, Phase 1/Phase 2, LaTeX formatting) — already encoded in the agent profile, no need to repeat it.

### Phase 4 — Reviewers Run Independently

Each reviewer processes its shard sequentially (per-paper, per the literature-reviewer profile's per-paper loop) and writes to its assigned shard file. Reviewers do not communicate with each other.

Wait for all K to complete.

### Phase 5 — Merge

Combine the K shard files into the final master doc:

1. Read each `tmp/lit_review_shard_<i>.md`.
2. Concatenate them in shard order (1, 2, ..., K).
3. Generate a unified Table of Contents at the top spanning all papers.
4. Verify there are no duplicate paper entries (a paper accidentally assigned to two shards — should not happen if Phase 1 was correct, but check).
5. Verify all LaTeX renders correctly (no broken `$$` blocks at shard boundaries).
6. Write the merged result to `docs/literature/<topic>.md`.

Do NOT delete the shard files immediately — keep them in `tmp/` for one-pass reference, in case the merge needs to be redone.

## Hand-off Artifact Summary

| Phase | Produces | Read by |
|---|---|---|
| 1 | `tmp/YYYYMMDD_HHMMSS_lit_shard_plan.md` | User (Phase 2) |
| 3–4 | `tmp/lit_review_shard_<i>.md` (one per shard) | Phase 5 merge |
| 5 | `docs/literature/<topic>.md` (final master) | User; possibly senior-developer for downstream synthesis |

## Why Per-Paper Stays Sequential Inside Each Shard

Parallelism happens at the **corpus level** (across shards), not within a shard. Inside a single `literature-reviewer` instance, papers are still processed one-by-one because the 4-step backbone (section list → per-section content → master append → deep dive) needs focused attention per paper. Trying to parallelize within a paper produces fragmented, inconsistent reviews.

The sweet spot: K independent reviewers, each working its shard sequentially with full attention.

## Choosing K

| Corpus size | Recommended K |
|---|---|
| < 10 papers | 1 (use single `literature-reviewer` directly, skip this skill) |
| 10–20 papers | 2–3 |
| 20–40 papers | 4–5 |
| 40+ papers | 5–6 (cap) |

Larger K reduces wall-clock time but increases coordination overhead and merge complexity. Beyond K=6 the marginal benefit is small.

## Common Mistakes to Avoid

- **Overusing the skill.** Most literature reviews are small enough that sequential is faster end-to-end. Default to single-reviewer unless the user opts in.
- **Sharding by topic instead of by file count.** Topical sharding sounds appealing but couples reviewers to corpus knowledge they do not have yet. Even, mechanical sharding is simpler and produces equivalent quality.
- **Forgetting to dedupe at merge.** If the same paper ends up in two shards (e.g., two filenames pointing to the same paper), the master doc will have it twice. The merge step must check.
- **Skipping the user confirmation.** Parallel processing has real cost — a 50-paper corpus split into 6 shards is 6 concurrent agent runs. Always pause for user approval before spawning.
- **Asking each reviewer to also write to the master doc.** That creates merge conflicts. Each reviewer writes to its own shard file; merge happens once at the end.
