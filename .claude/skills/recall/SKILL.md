---
name: recall
description: "Surface insights from this project's in-repo session-memory layer at docs/memory/. Use whenever the user asks 'what did we decide', 'what did we work on', 'remind me about', 'last week', 'yesterday', 'memory of', 'show me my memories', '/recall', or any question about prior decisions, debugging conclusions, or design rationale that might live in the memory layer. Two modes: natural-language recall (default — plain English, folder definitions hidden as definitions, time-aware) and technical query (raw indexes, folder names, counts) when the user says 'show ROOT_INDEX', 'list folders', 'audit', 'tag dictionary', 'fragmentation', 'L1/L2/L3'. Default behavior with no arguments: list the 10 most recent insights, reverse-chronological, then offer drill-down. Accepts optional range hints in the user phrasing ('today', 'yesterday', 'this week', 'earlier', 'all') for time-grouped output. Do NOT use this skill for capture — that's /memorize. Do NOT load raw archives (L4) without explicit user confirmation."
---

# Recall — surface insights from `docs/memory/`

This skill reads the in-repo memory layer at `docs/memory/` and produces a human-friendly listing of stored insights, with optional drill-down to a specific insight's body. It is the read counterpart to `/memorize`.

**Authoritative contract**: `docs/memory/CLAUDE.md` §8 (natural-language recall) and §9 (lazy-load levels). Treat it as ground truth.

## When to use

Trigger on:

- `/recall`, "show me my memories", "show recent memories".
- Natural-language recall phrases (per operating manual §8): "what did we decide", "what did we work on", "remind me about", "last week", "yesterday", "memory of", "what we did".
- Technical-query phrases: "show ROOT_INDEX", "list folders", "audit", "fragmentation", "tag dictionary", "L1 / L2 / L3".

Optional time-range hints in the user phrase: `today`, `yesterday`, `this week`, `earlier`, `all`. If absent → default to "10 most recent".

Do **not** use for:

- Capture ("remember this", "save this", "wrap up") — that's `/memorize`.
- Reading the raw conversation archive at `_archive/raw_conversations/` — only on explicit user confirmation, with the operating-manual §10 large-file warning protocol.

## Mode selection

Pick one mode for the whole response:

| User phrasing contains | Mode |
|---|---|
| "ROOT_INDEX", "list folders", "audit", "fragmentation", "lazy load", "L0/L1/L2/L3/L4", "tag dictionary", "raw indexes", "frontmatter" | **Technical** |
| anything else (default) | **Natural-language** |

Natural-language mode hides: raw folder names, lazy-load codes, frontmatter field names, raw tag strings, token costs, fragmentation jargon. Technical mode shows them verbatim.

## Step-by-step flow

### Step 1 — Read context

1. Read `docs/memory/ROOT_INDEX.md` — gives the list of topic folders + their 1-line definitions + counts.
2. For each active topic folder listed, read `docs/memory/memories/<topic>/_topic_index.md` — gives all insights in that topic with date, time, id, summary.

If `docs/memory/` is missing or `ROOT_INDEX.md` shows 0 active folders, halt with:

> No memories yet. Use `/memorize` to capture this conversation, or seed the `docs/memory/` layer first.

### Step 2 — Determine range

Parse the user's phrasing for a time hint:

| Phrase contains | Range |
|---|---|
| "today" | Insights with `date == today`. |
| "yesterday" | `date == today - 1`. |
| "this week" or "week" | `date >= today - 6`. |
| "earlier" | `date < today - 6`. |
| "all" | All insights. |
| (none — default) | 10 most recent insights, ignoring time grouping. |

If a time-range phrase is present, switch to **time-grouped output format** (see §3 below). If no range phrase, use **flat-list format**.

For the "10 most recent" default, do NOT bucket into Today/Yesterday/This week — just produce one flat reverse-chronological list of 10. The user opted into the simpler default at design time.

### Step 3 — Output format

Both modes are reverse-chronological at every level (newest first within each group).

#### Natural-language mode

**Flat list (default, no time hint)**:

```
## 10 most recent

- YYYY-MM-DD HH:MM — <summary> — *<folder definition>*
- YYYY-MM-DD HH:MM — <summary> — *<folder definition>*
- ...

Want details on any of these? Reply with the ID, the date+time, or a phrase from the summary.
```

Use the folder's **definition** from `ROOT_INDEX.md` (e.g. *"Claude memory system's own design decisions"*), NOT the raw folder name (`memory_system_design`). Italics signal "this is the topic, in plain English".

**Time-grouped (when user says "today", "yesterday", "this week", "earlier", "all")**:

```
## Today
- HH:MM — <summary> — *<folder definition>*
- HH:MM — <summary> — *<folder definition>*

## Yesterday
No new entries yesterday.

## This week
- YYYY-MM-DD HH:MM — <summary> — *<folder definition>*
- ...

## Earlier
- YYYY-MM-DD HH:MM — <summary> — *<folder definition>*
- ...

Want details on any of these? Reply with the ID or a phrase.
```

Rules per operating manual §8:
- `Today` and `Yesterday` headers always appear; if the bucket is empty, write `No new entries today.` / `No new entries yesterday.`
- `This week` and `Earlier` headers only appear if non-empty.
- Reverse-chronological at every level.

#### Technical mode

Show the raw indexes:

```
ROOT_INDEX.md (docs/memory/ROOT_INDEX.md)

Active folders: N
Total insights: M
Last audit: YYYY-MM-DD or (none)

| Folder | Definition | Insights | Last update | Top tags |
|---|---|---|---|---|
| <folder_name_raw> | <definition> | <n> | <date> | [<tags>] |
...

Per-topic indexes:
- memories/<folder>/_topic_index.md: N insights, last YYYY-MM-DD
...
```

If the user specifically asked for the tag dictionary, also dump `_global_tags.md`'s active tags table verbatim. If they asked for "audit", check the policy at the bottom of `ROOT_INDEX.md` and report whether thresholds are crossed (≥10 folders or 30 days since last audit).

### Step 4 — Drill-down (on user follow-up only)

If the user replies after the listing with:
- An insight ID (e.g. `20260508_0429_memorize_skill_design_and_ship`)
- A unique date+time prefix (e.g. `2026-05-08 04:29`)
- A phrase that matches one summary unambiguously
- A bullet number ("show me #2")

→ Read that specific insight file and render it. Format:

```
# <insight title>

**Topic**: *<folder definition>* (raw: `<folder_name>` — only show raw if technical mode)
**Date**: YYYY-MM-DD HH:MM
**Importance**: <high|medium|low>
**Status**: <settled|active|superseded>
**Tags**: <tag1>, <tag2>, ...

## Key conclusion
<verbatim>

## Decisions and actions
<verbatim>

## Open questions and follow-ups
<verbatim if non-empty, else skip>

(Evidence and References sections available — say "more" to expand.)
```

After rendering, check the insight's `valid_until` field. If non-null and in the past, append: `⚠️  This insight's valid_until passed on YYYY-MM-DD — re-verify before acting.`

By default skip `## Evidence, measurements, facts` and `## References` to keep the response tight. Surface them only on a "more" or "show evidence" follow-up.

If the insight's `raw_source` is non-`none` and the user explicitly asks for the raw conversation archive, apply the operating-manual §10 large-file warning protocol before reading. Do NOT load `_archive/raw_conversations/*.md` proactively.

If the match is ambiguous (multiple insights match the user's reply), list the candidates and ask which.

## Hard rules

- **Repo-relative paths only.** Read only inside `docs/memory/`. Never read `~/.claude/projects/.../memory/MEMORY.md` from this skill (different layer; recall there is the harness's job).
- **Lazy-load discipline**: read `ROOT_INDEX.md` + topic indexes for the listing. Read individual insight files only on drill-down. Read raw archives never, unless the user explicitly asks AND confirms after the size-warning prompt.
- **Natural-language mode hides**: raw folder names, L0/L1/L2/L3/L4, frontmatter field names, raw tag strings, token costs, "audit" / "fragmentation" jargon. Operating manual §8 forbids exposing them in this mode.
- **`valid_until` expiry warning (exception to "do not expose frontmatter")**: when surfacing any insight in natural-language output (flat list, time-grouped, or drill-down), check the insight's `valid_until` field. If it is non-null and its date is in the past (strictly before today's date), append this one-line warning immediately after the insight's summary line: `⚠️  This insight's valid_until passed on YYYY-MM-DD — re-verify before acting.` This warning is user-facing safety signal and IS permitted in natural-language mode. If `valid_until` is `null` or absent, do not emit any warning.
- All output in English; no emojis in warnings unless the user explicitly asks (the ⚠️ glyph is a safety signal, not decoration — it is always included).
- **Do not capture** anything from this skill. If the listing reveals something the user wants to save, route them to `/memorize`.
- If `docs/memory/CLAUDE.md` and this SKILL.md disagree, the operating manual wins — flag and ask.

## References

- `docs/memory/CLAUDE.md` §8 (natural-language recall format), §9 (lazy-load levels), §10 (large-file warning protocol).
- `docs/memory/ROOT_INDEX.md` — topic registry + folder definitions.
- `docs/memory/memories/<folder>/_topic_index.md` — per-topic insight rows.
- Companion skill: `.claude/skills/memorize/SKILL.md` (capture; this skill is recall).
