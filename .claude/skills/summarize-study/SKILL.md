---
name: summarize-study
description: "Generate a study-level summary report under docs/experiments/summaries/ that fans out to design docs, memory insights, working files, diary days, and commits. Use whenever the user says '/summarize-study', 'summarize this study', 'write me a study report', 'make a summary report of <topic>', 'generate a study summary for <topic>', 'summarise the <topic> work', or 'give me a reader-facing summary of the <topic> experiments'. **Stand-alone principle**: the summary must be readable cold — a reader who never opens any of the links should be able to understand the verdict, the methods, and the implications. Output is a 7-section summary (headline blockquote → 6-bullet take-home → §0 Vocabulary → §1 Question → §2 Experiments + pre-registered thresholds inlined → §3 Verdict table + narrative → §4 What's next → §5 Links → §6 Reading order) plus an optional Appendix A glossary whenever the study uses non-standard or project-specific behaviour metrics. Symbolic predicate names (H₀, H₁a, Δ_X) are translated to plain English; project shorthand (cell names, config slugs, run IDs) is defined in §0 Vocabulary on first use. The skill writes the summary file with a timestamped filename, prepends a row to the folder's README index, appends a `note` row to today's diary, and auto-commits the three files. Re-summaries write a fresh dated file (timestamped to the minute) — older summaries stay as historical snapshots; do not edit them in place. Distinct from /memorize (per-finding rationale), /diary (short-timeline events), experiment-designer (per-experiment design docs), and experiment-analyzer (per-experiment results)."
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
| Filename pattern | `YYYYMMDD_HHMM_<slug>.md` | `date +%Y%m%d_%H%M`; slug ≤ ~6 words English snake_case. Matches `docs/memory/` insight convention. |
| Frontmatter (5 fields) | `title`, `study`, `generated`, `window`, `status: snapshot` | `window` is the date range the summary covers, NOT the generation date. |
| Document shape (in order) | Headline-paragraph blockquote → Take-home bullets → §0 Vocabulary → §1 Study question → §2 Experiments + pre-registered thresholds → §3 Verdict table + narrative → §4 What's next → §5 Links → §6 Reading order → (optional) Appendix A | Section names verbatim — do not re-style. §0 Vocabulary is required whenever the study uses any project-specific shorthand; Appendix A is required whenever the study uses non-standard behaviour metrics. |
| Stand-alone principle | A reader who never opens any link should be able to understand the verdict, methods, and implications. | The summary inlines pre-registered thresholds + verdict numbers; it does NOT just point at the design doc. |
| Diary row subcommand | `note` | Free-form; the structured subcommands (`session-start`, `implemented`, `insight`, etc.) don't fit "summary-doc-created". |
| Commit scope | 3 files: new summary, README index, today's diary | Stage by name only — never `git add -A`. |
| Conda Python | `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` | For the diary helper script. |

## Why a separate folder + skill

The project already has three documentation layers — design docs in `docs/experiments/active/<topic>/` (full pre-registered design + results per experiment), memory insights in `docs/memory/memories/<topic>/` (per-finding 5-section files with rationale), and the daily diary in `docs/diary/` (one-line event rows). What's missing is a **study-level reader-facing layer** that names a coherent multi-experiment thread, explains it in plain language, and points at everything else. That's what `summaries/` provides; that's what this skill writes.

The reader-facing constraint is the load-bearing one. There are two specific commitments:

1. **Stand-alone principle.** A reader who never opens any of the links should be able to understand the *verdict*, the *methods*, and the *implications*. This means the summary inlines the things it needs to be self-contained: pre-registered confirmation thresholds, the closing-analysis verdict numbers, the cross-seed agreement table, and (when the study uses non-standard metrics) a behaviour-metric glossary appendix with formulas and code. Links exist for readers who want to *go deeper*, not for readers who need them to *understand at all*.
2. **Plain-language framing.** Predicate names like `H₀` / `H₁a` get translated on first mention; project shorthand like cell names (`Cell C`), config slugs (`decoupleFood`), and run IDs (`ja5fu5k3`) get defined once in §0 Vocabulary and replaced with descriptive labels in the body ("the food-decoupling experiment"). Prose framings are *"did the modulator beat the baseline?"* not *"did the H₁a confirmation criterion fire?"*. Symbolic notation and bare config paths appear in §5 Links, in §0 Vocabulary entries, or inside Appendix A code blocks — but not in body prose.

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
| Study label / topic name (e.g., "NMN comparison study") | Auto-discover from `docs/experiments/active/<topic>/` + `docs/memory/memories/<topic>/`. |
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

**Body — in this exact order. Section names verbatim.**

The structure layers from coldest-read (the reader has not seen the rest of the project) to warmest (deep technical detail). Every section above §5 must be readable without opening any link.

#### Headline-paragraph blockquote (very top, before take-home bullets)

A single self-contained paragraph (~150–300 words) inside a Markdown blockquote, immediately under the H1 title. It tells a cold reader, in one read, **what the study asked, what it found, and what the verdict is**. No symbolic notation. No bare config paths. No "see below" — it must stand alone.

Followed by a second short blockquote stating *"this document is the closing/interim re-summary"*, what it updates relative to the prior re-summary (if any), and the append-only rule reminder.

#### Take-home messages — the whole study in 6 bullets

Skim layer. **6 numbered bullets**, each ≤ ~3 sentences, in plain English. Cover:

1. **The setup** — one sentence on the experimental manipulation.
2. **The headline finding(s)** — what was learned. If the study produced a two-layer / multi-layer reading (different metrics gave different verdicts), call it out here.
3. **The replication / seed-stability status** — at one seed? at multiple seeds? cross-seed agreement?
4. **Caveats / nuance** — what the headline *doesn't* say. Sister experiments that complicate the story.
5. **The methodological lesson** — what this study taught us about how to measure things in this project.
6. **Where the work goes next** — name the candidate next moves; flag that no decision has been made.

Bullets are reader-facing — use descriptive labels for experiments (*"the food-decoupling experiment"*) rather than project shorthand (*"Cell C"*).

#### §0 Vocabulary — terms used in this document

**Required whenever the study uses any project-specific shorthand** (cell names, config slugs, round numbers, run IDs, custom metric short-forms). Skip *only* if the entire body of the summary uses no project-specific jargon — rare.

**Scope of §0**: project-specific shorthand the body uses repeatedly. **NOT** here: standard RL terms (episode, policy, checkpoint, seed, gradient) or environment basics (the grid, the entities, the agent's actions, the sensors). Standard RL is assumed knowledge for the project's readers; environment basics belong in §1 Study question's setup paragraph, where they're described in context.

Structure as 3 sub-tables (use whichever subset is non-empty for the study):

- **0.1 What this study manipulated** — every experimental knob with both its plain-English label and its project shorthand (e.g., *"Matched smells ('sameProp') — the predator and rabbits carry the same olfactory property vector."*). Each round / cell / config variant gets one row. Round numbers and seed numbers also live here.
- **0.2 How we measured the agent** — analysis-protocol terms the body refers to repeatedly: eval-rollout, deterministic-policy mode, danger radius, lookahead window, threat-onset event, pre-registered confirmation criteria, layer-of-measurement distinctions, etc. Define each one once and reuse the plain-English label thereafter.
- **0.3 The behaviour metrics (short form)** — one row per non-standard metric, with "plain-English question" as the meaning column. Points readers to Appendix A for the full definitions. Required when Appendix A is present.

Tables, not bullet lists — readers scan tables faster. Each cell is one or two sentences. If a term you're tempted to add is "what is an episode / a policy / a checkpoint", drop it (assumed knowledge). If it's "what is the grid / who is the predator / what does the smell sensor do", move it to §1 (described once in setup context). §0 stays compact — it's a cheat-sheet, not a textbook.

#### §1 Study question

The world: name the environment in 2–3 sentences for someone who has never seen it. List the entities, their actions, and the agent's sensors. Then state the manipulation, the puzzle, and why the puzzle matters for the broader project (one paragraph each). Translate any predicate framing into reader-friendly language.

#### §2 Experiments completed this study

A row-per-experiment table. Anchor experiments (e.g., a prior `v8` diagnosis) appear as a context row explicitly tagged `(prior anchor — not run this window)`.

| # | Experiment | Plain-English question | What was changed | High-level finding |

The first column is `0` for an anchor row, `1`/`2`/… for in-window experiments. Columns are reader-facing — describe experiments by what they *do*, not their shorthand name. Use descriptive labels (*"the food-decoupling experiment"*) and only mention the project shorthand the first time the experiment appears, in parens.

**Reader-pointer blockquote at the top** of §2 if the table uses any behaviour-metric short names: a 2-3 sentence callout linking to §0 Vocabulary and Appendix A.

**Immediately below the experiment table, when applicable: a Pre-registered confirmation criteria table.** Required whenever the study had a pre-registered analysis plan with numeric thresholds. Inline the thresholds so the verdict in §3 can be read without opening the design doc:

| Number | What it asks | Threshold for confirmation | Threshold for refutation |

#### §3 Where this leaves the study

Two sub-sections — verdict table first, narrative bullets second.

##### §3.1 The closing-analysis verdict in one table

Required when there is a closing analysis with cross-something agreement (cross-seed, cross-round, cross-cell, multi-run). Inline the verdict numbers and their cross-condition deltas:

| Number | Threshold | Observed (this round) | Observed (previous round) | Cross-something delta |

End with a one-paragraph "Verdict: …" sentence stating whether the criteria were met.

##### §3.2 What that means for the study

Narrative bullets summarising the cumulative state of understanding. This is the section a hurried reader skips to from the verdict table. Hit:

- The study's headline finding (now seed-locked / now refuted / now provisional).
- Any cross-experiment pattern.
- Any caveat — sister experiments that didn't replicate, sanity checks that landed in a soft-miss band, residual disagreements.
- The methodological surprise that affects future work.
- Any practical fix that fell out of the work.
- The wider arc — how this study positions the next research move.

#### §4 What's next (still pending decision)

Numbered list. Each item is a future experiment, redesign, or doc-update, with a plain-English description and (where relevant) a blocking dependency. **First item** is usually a portfolio-level decision call — list the (a)/(b)/(c) candidate next moves as a sub-bulleted enumeration. Owner-agent attributions optional; the prose should name the owner when relevant.

This is the section that turns a passive summary into actionable next steps.

#### §5 Links

Repo-relative paths only. Subsections (write `(none for this study)` rather than omit a heading — this guarantees the document is portable and self-describing):

- **Design docs** — every in-scope `docs/experiments/active/<topic>/<doc>.md`.
- **Supporting plans** — relevant `docs/develop/active/<topic>/<plan>.md`.
- **Memory insights** — every in-scope `docs/memory/memories/<topic>/<id>.md`, with a one-line gloss explaining what each carries (verdict, mechanism, design-rationale, etc.).
- **Working files** — every relevant `tmp/<file>.md` or `tmp/<file>.json` from the analyzer chain.
- **Eval-rollout outputs / results paths** — when applicable, point at `results/eval/.../<ckpt>/` or equivalent.
- **Diary days** — every `docs/diary/YYYY-MM-DD.md` covering the window, with a one-line gloss of what happened that day.
- **Prior summaries (this study)** — when re-summarising, list older dated summaries with their scope blurbs.
- **Implementation commits (load-bearing)** — relevant commit hashes (the design-doc commit, the analyzer-fill commit, the memory-capture commit, the closing-analysis commit) — these let a reader run `git show <hash>` to see the actual diff.

#### §6 Reading order if you have 10 minutes

**Opens with the stand-alone principle**: *"This document is intended to stand alone — you should not need to open any of the links to understand the verdict, the methods, or the implications. The reading order below points at the next-most-useful layer of detail for readers who do want to go deeper."*

Then a short ranked list. **The summary itself is item 1** (~5–6 min). **Appendix A is item 2** when the study has non-standard metrics. Then the most-up-to-date verdict, then the most-actionable mechanism finding. Optionally, "If you have 30 minutes, also read…" with a second tier.

#### Appendix A — `<metric-family-name>` glossary (optional but strongly recommended)

**Required whenever the study uses non-standard or project-specific behaviour metrics** (i.e., metrics that aren't off-the-shelf RL / behavioural-neuroscience measures the reader can look up elsewhere). The appendix's job is to let a reader who has never seen the metric definitions verify, on their own, what every headline number in §2 / §3 means.

Structure (per metric):

1. **Plain-English question.** What does the metric ask?
2. **Plain-English walk-through.** Step-by-step description of the computation.
3. **Formula.** Compact mathematical form, in a Markdown code block, ASCII-friendly.
4. **Edge case → NaN.** What makes the metric undefined; when readers should ignore it.
5. **Code extract.** Simplified Python (~15–30 lines) drawn from the actual implementation in `scripts/<file>.py:LINE`, sufficient to follow the state machine, not the full bookkeeping. Cite the source file + line number.
6. **Where it lands in this study.** The actual numbers, with cross-condition agreement when applicable.

Open the appendix with **A.0 Shared setup** — protocol parameters table (any constants the metrics share: radii, lookahead windows, eval-rollout episode counts, cluster counts, seeds) and a "key terms used below" mini-glossary.

Close with **A.<last> Sanity criteria** when the metric family has pre-registered "is the measure even meaningful?" gates (R1 / R2 / R3 / R4 style). One row per criterion: trigger condition + what it forces.

Anchor links: use the GitHub-flavoured-Markdown anchor of the appendix header to support `[Appendix A](#appendix-a--…)` references from §0 Vocabulary, §2, and §6.

### Plain-language enforcement (critical)

Before saving, do a self-review pass over the headline blockquote + take-home bullets + §§0–4 (NOT §5 or Appendix A, which are allowed to be technical):

| If the prose contains | Replace with |
|---|---|
| `H₀`, `H₁a`, `H₁b`, `Δ_X` (or any locked predicate name / symbolic delta) | The plain-English equivalent: "the agent did discriminate" / "the corner-camping policy did not replicate" / "the cross-class gap on bush-dive rate". Symbol may appear once in parens after a definition in §0 Vocabulary; should not appear in §§1–4 body prose. |
| A project shorthand like `Cell C` / `decoupleFood` / `sameProp` | Either a descriptive label ("the food-decoupling experiment", "matched smells") or the shorthand-with-translation on first mention; define once in §0.2 and use the descriptive label thereafter. |
| A bare WandB run ID like `f96lhxpe` | A link to the memory insight or design doc that cites it. Run IDs are allowed in §5 Links and inside Appendix A worked-example numerics. |
| A bare config path like `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` | A description of what the config changed ("raised the temperature ceiling from 3.0 to 10.0") + a §5 link to the design doc that pins the path. |
| Bare metric short-names (`M2`, `M5`, `Δ_M2`) without a plain-English label nearby | Either the plain-English label (`in-cover rate`, `eat-suppression ratio`) or the short name paired with the label on first use (`in-cover rate (M2)`). |
| Project-internal jargon without a one-clause translation OR a §0 Vocabulary entry | Add the translation inline, or add a §0 row and reference it. |

**The pass check**: after writing, re-read the headline blockquote + take-home bullets + §§0–3 cold. If any sentence requires opening a link to understand, rewrite that sentence (inline the missing context) or add a §0 Vocabulary entry.

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

- **Stand-alone principle.** A reader who never opens any link must be able to understand the verdict, the methods, and the implications. The summary inlines the pre-registered thresholds, the closing-analysis verdict numbers, and the cross-something agreement table; it does not just point at the design doc.
- **Repo-relative paths only** in every link. Never absolute paths from `/home/...` or system roots — links must be valid on a fresh clone.
- **Plain-language framing in headline blockquote + take-home + §§0–4.** Predicate names like H₀ / H₁a, symbolic deltas like Δ_X, project shorthand like cell names, bare WandB IDs, and bare config paths are all banned from body prose. Define once in §0 Vocabulary and use plain-English equivalents thereafter. Symbolic / path-shaped detail belongs in §5 Links, §0 Vocabulary entries, or Appendix A code blocks.
- **§0 Vocabulary is required** whenever the body uses any project-specific shorthand. Skip *only* if the body is genuinely jargon-free — rare.
- **Appendix A is required** whenever the study uses non-standard or project-specific behaviour metrics (anything a reader cannot look up in a standard RL / behavioural-neuroscience textbook). Cross-link from §0.5 and from §2's reader-pointer blockquote.
- **Inline data, not just pointers.** Pre-registered confirmation thresholds and verdict numbers live *in* the summary, not behind a link. The §3.1 verdict table is the load-bearing artifact for "did the study pass?".
- **Append-only re-summarisation.** Never edit older summary files. A re-summary writes a fresh dated file with a new timestamp; the older summary remains as a historical snapshot of what we knew on that date. (Minor in-place corrections — fixing a typo, removing a leftover duplicate paragraph — are allowed; verdict-moving updates require a new dated file.)
- **Conda Python**: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` for the diary helper. Never `python3`.
- **Section names verbatim.** Headline blockquote → Take-home bullets → §0 Vocabulary → §1 Study question → §2 Experiments completed this study → §3 Where this leaves the study (§3.1 verdict table, §3.2 narrative) → §4 What's next (still pending decision) → §5 Links → §6 Reading order if you have 10 minutes → Appendix A. Do not re-style. Future tooling may key on these.
- **Stage by name only** in Step 7. Other parallel-session changes must NOT slip in.
- **One commit per invocation.** Even if N experiments are summarised, the result is one summary file → one commit.
- **At least 2 experiments per summary.** Single-experiment reports belong in `docs/experiments/active/<topic>/` itself; the `summaries/` layer is for multi-experiment threads.

## Worked examples

### Current gold standard (full new shape)

The 2026-05-21 sameProp closing re-summary is the canonical example of the full document shape — headline blockquote, take-home bullets, §0 Vocabulary, pre-registered-thresholds inline, §3.1 verdict table with cross-seed agreement, and Appendix A (M1/M2/M5/M7 behaviour-metric glossary with formulas + code + worked examples + sanity-criteria gates).

- **Generated summary**: [`docs/experiments/summaries/20260521_1546_sameprop_rabbit_avoidance_study.md`](../../../docs/experiments/summaries/20260521_1546_sameprop_rabbit_avoidance_study.md). Read this first when authoring a new summary; the shape it lands on is what this skill should produce.
- **Resulting commits**: `f62ccc2` (initial reader-facing draft), `bb0b51f` (Appendix A added), `0b1a3b7` (jargon reduction + §0 Vocabulary).

Reading the file in full shows what good output looks like — every section, every inlined table, every formula, every code snippet, every cross-reference between §0 / §2 / §6 / Appendix A. The skill's job is to reproduce this shape on demand for any future study.

### Prior worked example (pre-Appendix-A, pre-§0)

For historical reference — the user executed the original flow by hand for the NMN comparison study before this skill existed. Earlier-shape example only; the current gold standard above supersedes it for document shape:

- **Generated summary**: `docs/experiments/summaries/20260509_1421_nmn_comparison_study.md`.
- **Folder README**: `docs/experiments/summaries/README.md`.
- **Diary note row**: in `docs/diary/2026-05-09.md`, the `## Notes` bullet added at 14:24 KST.
- **Resulting commit**: `d075fdc` (`git show d075fdc --stat`).

Reproduce the commit-message tone, README-index row format, and diary note format from this earlier example; reproduce the document shape from the current gold standard.

## References

- `docs/experiments/summaries/README.md` — folder operating contract (sections: Purpose, Filename convention, What goes inside, What's NOT here, Index, Conventions).
- `.claude/skills/memorize/SKILL.md` — companion skill; same auto-commit pattern at Step 9.
- `.claude/skills/diary/SKILL.md` — companion skill; this skill calls the `note` subcommand.
- `scripts/diary_append.py` — diary helper script (see `--help` for arg details).
- Project root `CLAUDE.md` — project-wide rules (no fallback defaults, conda env, auto-commit authorization, git safety).
