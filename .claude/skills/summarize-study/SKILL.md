---
name: summarize-study
description: "Generate a study-level summary report under docs/experiments/summaries/ that fans out to design docs, memory insights, working files, diary days, and commits. Use whenever the user says '/summarize-study', 'summarize this study', 'write me a study report', 'make a summary report of <topic>', 'generate a study summary for <topic>', 'summarise the <topic> work', or 'give me a reader-facing summary of the <topic> experiments'. Reader-facing — the output is written for someone who has not seen the detailed plans, so symbolic predicate names (H₀, H₁a, etc.) are translated on first mention and bare WandB IDs / config paths are hidden behind links. The skill writes a 6-section summary file with a timestamped filename, prepends a row to the folder's README index, appends a `note` row to today's diary, and auto-commits all three files. Re-summaries write a fresh dated file (timestamped to the minute) — older summaries stay as historical snapshots; do not edit them in place. Distinct from /memorize (per-finding rationale), /diary (short-timeline events), experiment-designer (per-experiment design docs), and experiment-analyzer (per-experiment results)."
---

# Summarize-Study — generate a multi-experiment study report under `docs/experiments/summaries/`

This skill produces a single reader-facing summary document that names every experiment in a study (typically 2+ experiments completed within days or weeks), explains what each one tested, what was found, and what changed about our understanding — then fans out via repo-relative links to the authoritative design docs, memory insights, working files, diary days, and commits.

**Authoritative contract**: `docs/experiments/summaries/README.md`. Treat it as ground truth for the folder's purpose, filename rules, and recommended section structure. This `SKILL.md` describes the *invocation flow*; the README carries the *contract*. If they disagree, the README wins — flag it and ask the user.

## When to use

Trigger on:

- `/summarize-study` (slash command).
- "summarize this study", "write me a study report", "make a summary report of `<topic>`", "generate a study summary for `<topic>`".
- "summarise the `<topic>` work", "give me a reader-facing summary of the `<topic>` experiments".
- Any phrasing that asks for a multi-experiment overview written for a reader who hasn't seen the detailed plans.

Do **not** use for:

- Per-experiment design docs → `experiment-designer` (writes to `docs/experiments/active/<topic>/`).
- Per-experiment analysis fill-in (Results / Analysis / Conclusions sections) → `experiment-analyzer`.
- Per-finding insights with rationale, rejected alternatives, debugging arcs → `/memorize`.
- Implementation plans → `senior-developer`.
- Daily event log (start/end of session, training-start/done, insight rows) → `/diary`.
- Single-experiment reports — at minimum 2 experiments should be in scope. If only one is available, suggest the user wait or use the experiment doc itself.

## Quick reference (read first; details on demand)

| Thing | Value | Notes |
|---|---|---|
| Authoritative contract | `docs/experiments/summaries/README.md` | Folder purpose, filename rules, section structure. |
| Output folder | `docs/experiments/summaries/` | Sibling to `active/` and `meta/` under `docs/experiments/`. |
| Filename pattern | `YYYYMMDD_HHMM_<slug>.md` | `date +%Y%m%d_%H%M`; slug ≤ ~6 words English snake_case. Matches `.claude-memory/` insight convention. |
| Frontmatter (5 fields) | `title`, `study`, `generated`, `window`, `status: snapshot` | `window` is the date range the summary covers, NOT the generation date. |
| Body sections (6, in order) | §1 Study question / §2 Experiments / §3 Where this leaves us / §4 What's next / §5 Links / §6 Reading order | Section names verbatim — do not re-style. |
| Diary row subcommand | `note` | Free-form; the structured subcommands (`session-start`, `implemented`, `insight`, etc.) don't fit "summary-doc-created". |
| Commit scope | 3 files: new summary, README index, today's diary | Stage by name only — never `git add -A`. |
| Conda Python | `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` | For the diary helper script. |

## Why a separate folder + skill

The project already has three documentation layers — design docs in `docs/experiments/active/<topic>/` (full pre-registered design + results per experiment), memory insights in `.claude-memory/memories/<topic>/` (per-finding 5-section files with rationale), and the daily diary in `docs/diary/` (one-line event rows). What's missing is a **study-level reader-facing layer** that names a coherent multi-experiment thread, explains it in plain language, and points at everything else. That's what `summaries/` provides; that's what this skill writes.

The reader-facing constraint is the load-bearing one: a teammate (or future-you) opening a summary should be able to understand the study without first reading the design docs. So predicate names like H₀ / H₁a get translated on first mention; bare WandB IDs and config-file paths stay out of the prose; the prose framings are "did the modulator beat the baseline?" not "did the H₁a confirmation criterion fire?". Symbolic notation may appear in §5 Links or inside the linked memory insights themselves — but not in the summary's prose.

## Step-by-step flow

### Step 1 — Read context

Read two files:

1. `docs/experiments/summaries/README.md` — the folder's operating contract.
2. The README's existing summary index — to see prior summaries' study-labels and timestamps. Avoid duplicate-slug collisions; if a related study has been summarised before, note that the new summary refines or replaces an older one (in `§5 Links` or in a one-line note at the top, not by editing the older file).

If the folder doesn't exist yet, halt and ask the user: "`docs/experiments/summaries/` is not seeded. Create it (with a stub README) before the first invocation, or run `/summarize-study --bootstrap` (if implemented)." The folder + README came in via commit `d075fdc`; on a fresh clone where `git pull` brought it down, it should already exist.

### Step 2 — Determine scope

The user invokes the skill in one of four ways:

| Input shape | What you do |
|---|---|
| Study label / topic name (e.g., "NMN comparison study") | Auto-discover from `docs/experiments/active/<topic>/` + `.claude-memory/memories/<topic>/`. |
| Topic folder under `docs/experiments/active/<topic>/` | Auto-discover from that folder + the matching memory subfolder. |
| Date window (e.g., "2026-05-07 to 2026-05-09") | Scan all topic folders for design docs whose latest analysis row falls in the window. |
| Explicit list of experiment doc paths | Use those exactly; no auto-discovery. |

For the auto-discovery cases, the heuristic is:

- A design doc counts as **in scope** if its frontmatter `status` is `ANALYZED` / `COMPLETE` / `analyzed` / `complete` / `done`, OR if its top-of-file Status line says one of those.
- A memory insight counts as **in scope** if its frontmatter `folder` matches the topic AND its `date` falls in the same window as the in-scope design docs (± a day for analyzer chain lag).
- A diary day counts as in scope if its date falls in the window of any in-scope experiment.

**Always show the auto-detected list back to the user** with one-line summaries and ask:

> Including these N experiments + M memory insights + K diary days — proceed?
>
> Experiments:
> 1. `<doc-stem>` — `<one-liner from frontmatter title>`
> 2. ...
>
> Insights:
> 1. `<insight-id>` — `<one-line summary>`
> 2. ...
>
> (Y = all / numbers like "1,3" / N = let me name the docs explicitly / edit window)

If non-interactive, default to Y on all auto-detected items.

If no experiments match, halt: "No COMPLETE / ANALYZED experiments found for `<topic>` in the requested window. Either name an experiment doc explicitly, or wait until at least one experiment in the topic is analyzed." Don't write a summary against unfinished work — readers will misinterpret a snapshot taken mid-experiment.

### Step 3 — Compute filename and slug

Timestamp via `date +%Y%m%d_%H%M`. Slug rules:

- English snake_case, ≤ ~6 words (e.g., `nmn_comparison_study`, `dreamer_v3_diagnosis`, `hypervigilance_round3`).
- Descriptive of the *study*, not of any single experiment in it. If the study spans two distinct topic folders, name the conceptual thread (e.g., `precision_modulation_thread`).
- Confirm slug back to the user before writing — slug + filename appear in the README index forever.

If a summary already exists at the same `HHMM` minute (extremely unlikely), bump by one minute and re-confirm.

### Step 4 — Write the summary file

Path: `docs/experiments/summaries/YYYYMMDD_HHMM_<slug>.md`.

**Frontmatter** (5 fields):

```yaml
---
title: "<plain-English title>"
study: <slug>
generated: YYYY-MM-DDTHH:MM
window: "YYYY-MM-DD → YYYY-MM-DD"
status: snapshot
---
```

`window` is the date range the summary *covers* — when the experiments ran — not the generation date. `status: snapshot` is the only valid value (this is an append-only layer; old summaries don't get edited).

**Body — six sections in order, names verbatim:**

#### §1 Study question

One paragraph, plain English. State what the study is trying to answer and why it matters. Translate any predicate framing into reader-friendly language: "does the modulated agent beat the baseline?" not "does H₁a hold?".

#### §2 Experiments completed

A table with one row per experiment in scope. Anchor experiments (e.g., a prior `v8` diagnosis) may appear as a context row explicitly tagged `(prior anchor — not run this window)`.

| # | Experiment | Question (plain English) | What was varied | High-level finding | What it changed about our understanding |

The first column is `0` for an anchor row, `1`/`2`/... for in-window experiments. Columns are reader-facing — the "Question" column says "Does the modulated agent beat the baseline when noise is heterogeneous?" not "Does H₁a hold for any profile P3–P5?"; the "High-level finding" says "Modulated never wins; consistently 5-13 steps worse" not "H₁a refuted on all profiles".

#### §3 Where this leaves the study

Bullets summarising the cumulative state of understanding. This is the section a hurried reader skips to. Hit:

- The study's headline finding.
- Any cross-experiment pattern (e.g., "the bottleneck is downstream of the temperature head").
- Any methodological surprise that affects future work (e.g., "single-seed Δ_SS in this regime carries ±4-5 steps of noise; future runs need ≥3 seeds").
- Any practical fix that fell out of the work (e.g., "raise canonical FiLM `temp_clip` ceiling from 3.0 to 5.0").

#### §4 What's next (still pending decision)

Numbered list. Each item is a future experiment, redesign, or doc-update, with:

- A plain-English description of what it does.
- The named owner agent in parentheses (e.g., `(experiment-designer to author)`, `(senior-developer to plan)`).
- Any blocking dependency (e.g., "blocked on item 1").

This is the section that turns a passive summary into actionable next steps.

#### §5 Links

Repo-relative paths only. Five subsections (write `(none for this study)` rather than omit a heading — this guarantees the document is portable and self-describing):

- **Design docs** — every in-scope `docs/experiments/active/<topic>/<doc>.md`.
- **Anchor diagnosis** — the prior diagnosis the study extends (e.g., `docs/develop/active/diagnosis/<...>_v8.md`).
- **Memory insights** — every in-scope `.claude-memory/memories/<topic>/<id>.md`, with a one-line gloss explaining what each carries (verdict, mechanism, design-rationale, etc.).
- **Working files** — every relevant `tmp/<file>.md` or `tmp/<file>.json` from the analyzer chain.
- **Diary days** — every `docs/diary/YYYY-MM-DD.md` covering the window, with a one-line gloss of what happened that day.
- **Implementation commits** — relevant commit hashes (the design-doc commit, the analyzer-fill commit, the memory-capture commit) — these let a reader run `git show <hash>` to see the actual diff.

#### §6 Reading order if you have 10 minutes

A short ranked list — 3 to 5 items — telling a hurried reader which docs to read in what order. The summary itself is item 1 (5 min). Then the most-up-to-date verdict, then the most-actionable mechanism finding. Optionally, "If you have 30 minutes, also read…" with a second tier.

### Plain-language enforcement (critical)

Before saving, do a self-review pass over §§1–4 (NOT §5, which is allowed to be technical):

| If the prose contains | Replace with |
|---|---|
| `H₀`, `H₁a`, `H₁b`, `H₁c` (or any locked predicate name) | the plain-English equivalent on first mention; subsequent mentions may use the symbol if context is clear |
| A bare WandB run ID like `f96lhxpe` | a link to the memory insight or design doc that cites it |
| A bare config path like `configs/models/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` | a description of what the config changed (e.g., "raised the temperature ceiling from 3.0 to 10.0") + a link to the design doc |
| Project-internal jargon without a one-clause translation | the translation, then optionally the jargon in parens |

The goal is that a reader who has never opened the design docs can still follow the headline story. Symbolic / numeric / path-shaped detail belongs in §5 or in the linked sources.

### Step 5 — Update the README index

Prepend ONE row (newest first) to the Index table in `docs/experiments/summaries/README.md`:

```
| YYYY-MM-DD | HH:MM | [filename-stem](filename.md) | <study label> | <one-line scope> |
```

Do **not** edit any other section of the README — it is the operating contract, not a per-summary log. If the README's structure has drifted from what this skill expects (e.g., the Index table moved or its columns changed), surface the drift to the user and halt; do not rewrite the README.

### Step 6 — Append a diary `note` row

Why a `note` and not a structured row: `.claude/skills/diary/SKILL.md`'s structured subcommands (`session-start`, `implemented`, `verified`, `insight`, `training-start`, `training-done`) don't have a "summary-doc-created" type. The diary's `note` subcommand is the catch-all for events that don't fit a structured row.

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/diary_append.py note \
  --text "Summary report: [<filename-stem>](../experiments/summaries/<filename>.md) — <one-line scope>. Covers <N> experiments + <M> memory insights from <window>."
```

The diary file is at `docs/diary/<today>.md`; the helper handles flock + path-formatting. Use the conda Python interpreter, never `python3`.

### Step 7 — Auto-commit

Mandatory; mirrors `/memorize` Step 9. After all writes succeed, bundle every file this skill touched into a single commit. Stage by name (NEVER `git add -A` or `git add .`):

```bash
git add \
  docs/experiments/summaries/<new-summary>.md \
  docs/experiments/summaries/README.md \
  docs/diary/<today>.md

git commit -m "$(cat <<'EOF'
docs(experiments): 📚 add <slug> study summary

<one-paragraph framing — what study, what's covered, what changed>

- summaries/<filename>: <one-line scope>
- summaries/README.md: index updated
- diary/<today>.md: note row pointing at the new summary
EOF
)"
```

Commit-message rules:

- Conventional commit type: `docs`. Scope: `experiments`. Gitmoji: `📚`.
- Subject ≤ ~70 chars.
- Body: one-paragraph framing + bullet list of the 3 staged files.
- Co-Authored-By trailer at the end (project standard — check `git log --oneline -- docs/experiments/` for tone reference; the manual run produced commit `d075fdc` as a worked example).
- No `--no-verify`, no secrets, no push.

Hard rules for the commit:

- **Stage by name only.** Other parallel-session changes in the working tree must NOT slip in.
- **Skip the commit if any prior step errored.** A failed write or diary call means the capture is incomplete; surface the error and let the user resolve. Don't commit a partial state.
- **One commit per `/summarize-study` invocation** regardless of how many experiments were summarised.

### Step 8 — Confirm and report

Report in plain English:

> Wrote `docs/experiments/summaries/<filename>.md` covering <N> experiments + <M> memory insights from `<window>`. README index updated; diary note row appended. Committed as `<short-hash>`.
>
> Read it: `docs/experiments/summaries/<filename>.md`

Don't offer to push. Skill ends at the commit.

## Hard rules

- **Repo-relative paths only** in every link. Never absolute paths from `/home/...` or system roots — links must be valid on a fresh clone.
- **Plain-language framing in §§1–4.** Predicate names like H₀ / H₁a get translated on first mention; bare WandB IDs and config-file paths stay out of the prose. Symbolic / path-shaped detail belongs in §5 Links or in the linked source documents.
- **Append-only re-summarisation.** Never edit older summary files. A re-summary writes a fresh dated file with a new timestamp; the older summary remains as a historical snapshot of what we knew on that date.
- **Conda Python**: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` for the diary helper. Never `python3`.
- **Section names verbatim.** §1 Study question, §2 Experiments completed, §3 Where this leaves the study, §4 What's next, §5 Links, §6 Reading order. Do not re-style. Future tooling may key on these.
- **Stage by name only** in Step 7. Other parallel-session changes must NOT slip in.
- **One commit per invocation.** Even if N experiments are summarised, the result is one summary file → one commit.
- **At least 2 experiments per summary.** Single-experiment reports belong in `docs/experiments/active/<topic>/` itself; the `summaries/` layer is for multi-experiment threads.

## Worked example

The user executed this flow by hand for the NMN comparison study before this skill existed. Use these as canonical examples of input shape and output shape:

- **Generated summary**: `docs/experiments/summaries/20260509_1421_nmn_comparison_study.md`.
- **Folder README**: `docs/experiments/summaries/README.md`.
- **Diary note row**: in `docs/diary/2026-05-09.md`, the `## Notes` bullet added at 14:24 KST.
- **Resulting commit**: `d075fdc` (`git show d075fdc --stat`).

Reading any of those files in full will show what good output looks like — section structure, plain-language framing, link layout, README index row format, diary note format, commit message tone. The skill's job is to reproduce this flow on demand for any future study.

## References

- `docs/experiments/summaries/README.md` — folder operating contract (sections: Purpose, Filename convention, What goes inside, What's NOT here, Index, Conventions).
- `.claude/skills/memorize/SKILL.md` — companion skill; same auto-commit pattern at Step 9.
- `.claude/skills/diary/SKILL.md` — companion skill; this skill calls the `note` subcommand.
- `scripts/diary_append.py` — diary helper script (see `--help` for arg details).
- Project root `CLAUDE.md` — project-wide rules (no fallback defaults, conda env, auto-commit authorization, git safety).
