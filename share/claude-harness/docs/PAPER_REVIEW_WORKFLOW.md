# The paper-review pipelines, end to end

## Purpose

"Reviewing a paper" means two completely different jobs, and this bundle has a separate pipeline for
each. Mixing them up is the most common way to get disappointing output, so the first thing this
document does is separate them.

**Track A — reading other people's papers.** You have 40 PDFs and you need to actually know what is in
them, in a form you can cite six months from now. The output is a durable, structured review document
per topic that lives in your repo and grows over time. This is the one I use daily, and it is my own
work: two agents (`literature-reviewer`, `literature-curator`) plus a PDF-fetching skill.

**Track B — reviewing a manuscript as a referee would.** You have a draft — yours or someone else's —
and you want a hard, structured critique before submission. The output is five independent reviewer
reports plus an editorial decision letter and a revision roadmap. This comes from a third-party skill
pack I vendored (`academic-research-skills`), which also covers the write-and-revise side.

They compose: Track A builds the literature base you write from, Track B critiques what you wrote.

---

# Track A — building a literature corpus

## The moving parts

| Piece | Type | Job |
|---|---|---|
| `academic-pdf-fetch` | skill | Get one paper's version-of-record PDF into the library |
| `literature-reviewer` | agent | Extract one paper into a structured review. Never synthesizes across papers |
| `literature-curator` | agent | Synthesize across papers that are already extracted. Never opens a raw PDF |
| `notebooklm` | skill | Alternative source path when the corpus lives in a Google NotebookLM notebook |

The split between the two agents is deliberate, and it is the design decision that makes the whole
thing work. **Extraction is a per-paper job that rewards narrow focus; synthesis is a corpus-level job
that needs the whole picture in view.** One agent doing both does neither well — it starts
generalizing after paper three and stops actually reading. So the reviewer is forbidden from
reorganizing, and the curator is forbidden from opening a PDF.

## File layout

Everything for one topic lives in one folder. Review documents sit at the topic root, raw sources sit
one level down:

```
docs/project/references/<topic>/
├── <topic>_lit_review.md      ← the master review; this is the deliverable
├── <topic>_synthesis.md       ← optional cross-paper synthesis companion
└── sources/
    ├── Kendall & Gal 2017 - What uncertainties do we need.pdf
    └── Perez et al. 2018 - FiLM Visual reasoning.pdf
```

`<topic>` is just the folder name (`uncertainty`, `FiLM`, `computational_models_of_pain`). Both agents
derive every path from it, so adding a topic means creating a folder and nothing else.

One master document per topic, appended to forever, rather than one file per paper. It means the
review of paper 40 sits next to the review of paper 1, which is exactly when you notice they
contradict each other.

## Step 0 — get the PDF

The `academic-pdf-fetch` skill handles one paper at a time and **escalates lazily, stopping at the
first verified file**:

| Tier | Method | Used when |
|---|---|---|
| 0 | Resolve the identifier — Unpaywall, Semantic Scholar, PMC, publisher landing page | Always. Classifies candidates as published version vs preprint |
| 1 | Plain `curl` with a browser user-agent | The common case |
| 2 | Same `curl`, plus a publisher `Referer` | Paywalled but no bot check |
| 3 | A real headed Chrome on a virtual display | **Only** for Cloudflare bot checks — and only after asking you first |

Two ideas here are worth keeping regardless of your setup. First, the skill separates the **two locks**
explicitly: a paywall asks *"where are you connecting from"* (opened by an institutional IP), a
Cloudflare challenge asks *"are you a real browser"* (opened only by a genuine headed browser —
headless Chrome and cookie-replay both fail). Knowing which lock you are facing tells you which tier to
jump to. Second, verification is by **magic bytes** (`head -c 4 | grep %PDF`), because the failure mode
is silently saving an HTML login page as `paper.pdf` and only discovering it 20 papers later.

The skill also prefers the **version of record over a preprint**, deduplicates against what is already
in `sources/`, and labels a preprint as one in the filename when that is all it can get.

**What you must change:** the skill assumes the machine it runs on already has a campus IP (true for
our lab container, probably false for your laptop — you would need a VPN or institutional proxy), and
you must put your own email in the Unpaywall call. It refuses to use Sci-Hub; keep it that way.

## Step 1 — extract, one paper at a time (`literature-reviewer`)

For each paper the agent runs a **4-step completeness guard first**, before any synthesis:

1. **Extract the section list** — the paper's full section and subsection structure.
2. **Extract core content per section** — key claims, methods, equations, results.
3. **Append a section-by-section summary to the master document, in the paper's original order.**
   (The existing reviews call this "the backbone".)
4. **Deep-dive the most relevant sections** — the agent picks these itself, based on project context.

These four run without stopping to check in. Only then does it write the actual review, in two layers:

- **Phase 1 — foundational overview.** Undergraduate level. The core problem, the main results, why it
  matters, in plain terms.
- **Phase 2 — graduate-level deep dive.** Full technical breakdown, **all** critical equations in
  LaTeX, and step-by-step derivations. The profile is explicit: *never simply state a formula* — show
  how the result is reached.

Phase 1 and 2 are **not** bound to the paper's section order; they regroup by theme. The backbone stays
underneath as `### Appendix: Section-by-Section Backbone`, so the rewrite is always traceable back to
the source. That appendix is what makes the review trustworthy a year later — you can check whether a
claim came from the paper or from the model's enthusiasm.

Three rules keep it honest:

- **Never write Phase 1/2 from a skim.** The synthesis must be derived from a completed backbone.
- **Papers are processed strictly sequentially inside one reviewer instance.** Parallelism happens at
  the corpus level (split 40 papers across several reviewer instances, each with its own batch), never
  inside one.
- **Intermediate extractions are written to `tmp/` after each step**, never accumulated only in
  context. A 40-paper review that dies at paper 31 should not lose 31 papers of work.

The agent can write **only** under `docs/project/` — never `src/`, `configs/`, or `scripts/`. If a
review surfaces a needed code change, it writes a note and names the agent that should own it.

## Step 2 — synthesize across papers (`literature-curator`)

Once ten or more papers are in, the curator earns its keep. It produces four things:

- **Thematic regrouping** — reorders the master document's table of contents by theme rather than by
  the accident of processing order, and opens each themed cluster with a 2–4 sentence framing
  paragraph saying what the cluster is about and why it matters.
- **Cross-paper comparison tables** — e.g. one row per paper, columns for method, formulation, loss
  term, reported gain, and *"tested in this project yet?"*. That last column is the one that turns a
  literature review into a research agenda.
- **Synthesis sections** stating three things separately: the **consensus**, the **disagreement axes**
  (where papers contradict each other, *and why* — different domains? scales? metrics?), and the
  **gaps** (what the corpus does not answer, and which experiment would).
- **Conflict reconciliation** — when two papers disagree on a concrete choice, state the conflict,
  diagnose its cause, and recommend which side your project should default to, with reasons.

Its hard constraints: it never deletes a per-paper review or a backbone appendix, it never opens a raw
PDF, and it proposes a new structure before any bulk rewrite.

## Step 3 — optional critique layer

Two other agents plug into this pipeline:

- A **professor** agent for domain critique — "the corpus says X, does that actually hold for our
  setup?" They append signed feedback rather than editing the review in place.
- **`math-reviewer`** when a paper's equations have been implemented in code. It checks the code
  against the paper: dimensional consistency, derivation steps, faithfulness. This closes the loop that
  a literature review normally leaves open — you read the paper, you implemented *something*, and
  nobody ever checked they match.

## A worked run

```
You: download 10.1038/s41593-021-00980-9 into the pain topic
     → academic-pdf-fetch resolves the DOI, finds no OA copy, fetches through
       institutional access, verifies magic bytes, saves as
       docs/project/references/computational_models_of_pain/sources/
       Author et al. 2021 - Title.pdf

You: review the papers in docs/project/references/computational_models_of_pain/
     → literature-reviewer globs sources/*.pdf, then per paper:
       backbone (4 steps) → Phase 1 → Phase 2 → append to the master review →
       update the TOC → next paper. Checkpoints to tmp/ throughout.

You: the pain corpus has 14 papers now — regroup it by theme and tell me
     where the papers disagree
     → literature-curator proposes a themed TOC, you approve, it rewrites the
       structure in place, adds a comparison table and a
       consensus/disagreement/gaps synthesis, leaves every backbone intact.
```

## Adapting Track A to your project

The two agent profiles are prompts, so adapting them is editing prose. Point your Claude at
`literature-reviewer.md` and `literature-curator.md` in this bundle, at your own repo, and ask it to
write your versions. The specific things that must change:

1. **The path convention.** Both profiles derive every path from `docs/project/references/<topic>/`
   with sources in `<topic>/sources/`. Any layout works, but both agents must agree on it, and the
   profiles must state it explicitly — this is the single most common thing to get wrong, because a
   reviewer that writes to a different folder than the curator reads from fails silently.
2. **The project-specific anchors in `literature-curator`.** It currently ties every synthesis back to
   my project's gates, hypotheses, and phase plan. Replace with yours, or drop the section — but
   replacing is better: an anchor is what turns "here is what the field says" into "here is what the
   field says *about the thing we are stuck on*".
3. **The "three legacy filenames" paragraph** in both profiles — that is my repo's history.
4. **The two dangling references** — `parallel-literature-review` (skill) and `literature-deepdive`
   (agent). Neither was ever built.
5. **In `academic-pdf-fetch`:** your email for the Unpaywall call, and a realistic assessment of
   whether the machine Claude runs on has institutional network access. The profile also hard-codes
   facts about our container (shell, browser, virtual-display tool, which PDF utilities exist) — it
   tells Claude to re-verify these at runtime rather than trust them, which is the right instinct, but
   the Tier 3 recipe is written for our machine.

The 4-step backbone, the Phase 1/Phase 2 split, the appendix-retention rule, and the
extraction/synthesis separation are domain-independent. Keep those as they are — they are the part
that makes the output trustworthy a year later.

---

# Track B — peer review of a manuscript

This is `.claude/skills/academic-research-skills/`, a vendored third-party pack (v3.3) with four
skills and 24 modes. **It is not my work.** Licensed CC BY-NC 4.0 — noncommercial academic use only,
and it asks to be cited as:

> Wu, C.-I. (2026). *Academic Research Skills for Claude Code* (Version 3.3) [Computer software].
> https://github.com/Imbad0202/academic-research-skills

## The review team

`academic-paper-reviewer` simulates a full journal review by running **seven roles** — it first
identifies the paper's field, then configures five reviewers with field-appropriate expertise, then
synthesizes:

| Role | What it attacks |
|---|---|
| Field analyst | Identifies field and methodology type, then configures the rest of the panel |
| Editor-in-Chief | Overall significance, fit, and the decision itself |
| Methodology reviewer | Design, statistics, threats to validity |
| Domain reviewer | Correctness and novelty within the field |
| Perspective reviewer | The cross-disciplinary read — what an outsider would object to |
| **Devil's advocate** | Attacks the *core argument*: logical fallacies, the strongest counter-argument you did not address |
| Editorial synthesizer | Merges five reports into one decision letter + revision roadmap |

The four review perspectives are explicitly **non-overlapping**, which is what stops five reviewers
from writing the same review five times. The devil's advocate is the one that earns its cost — it is
the only role instructed to argue that the paper is wrong rather than that it needs improvement.

## Modes

`academic-paper-reviewer` (6):

| Mode | Output | Use when |
|---|---|---|
| `full` | 5 reports + editorial decision + revision roadmap | Before first submission |
| `re-review` | Verification checklist + residual issues | After revising, to check you actually addressed the comments |
| `quick` | EIC assessment + key issues | 15-minute quality read |
| `methodology-focus` | In-depth methods review | Stats and design only |
| `guided` | Socratic issue-by-issue dialogue | You want to work through the problems yourself |
| `calibration` | Calibration report (FNR/FPR/AUC) + confidence disclosure | Measuring how accurate the simulated reviewer actually is |

That last mode is unusual and worth noting: the pack ships a way to **measure whether its own reviewer
is any good**, rather than asking you to take its verdicts on faith.

The sibling skills: `deep-research` (7 modes — including PRISMA systematic review, fact-check, and a
Socratic mode for when you cannot yet state your research question), `academic-paper` (10 modes —
drafting, outlining, revision, citation-check, and a venue-specific AI-disclosure generator), and
`academic-pipeline`, a 10-stage orchestrator chaining research → write → integrity check → review →
revise → re-review → final integrity check → finalize.

The pipeline's design principle is the one I would flag to you: **mandatory human checkpoints**. Each
stage stops and waits for your explicit confirmation, integrity gates cannot be skipped, and there is a
hard cap of two revision loops — after which remaining issues become *"Acknowledged Limitations"*
rather than being quietly resolved by the model. The authors are explicit that "full mode" means full
*pipeline*, not full *autonomy*.

## A worked run

```
You: review this paper — <attach draft.pdf>
     → identifies field and method type
     → configures 5 reviewers with matching expertise
     → 5 independent reports, four non-overlapping angles
     → editorial decision letter + prioritized revision roadmap

...you revise...

You: check whether my revisions address the comments
     → re-review mode: point-by-point verification + residual issues
```

Budget note from the pack's own quickstart: a full end-to-end pipeline run is a few dollars of API
usage and 2–4 hours of collaborative work. A `full` review alone is much less.

---

# Which track, and setup

| You want to... | Use |
|---|---|
| Know what is in a stack of PDFs, durably | Track A — `literature-reviewer` |
| Find where a body of literature agrees, disagrees, and has gaps | Track A — `literature-curator` |
| Get one specific paper's PDF | `academic-pdf-fetch` |
| Explore a topic you cannot yet state as a question | Track B — `deep-research` socratic mode |
| A systematic review with PRISMA | Track B — `deep-research` systematic-review mode |
| Hard critique of a draft before submission | Track B — `academic-paper-reviewer` full mode |
| Check that your revisions landed | Track B — `academic-paper-reviewer` re-review mode |
| Everything end to end | Track B — `academic-pipeline` |

## Getting it running

Three moving parts, and it is worth knowing what each one needs:

- **The two agents** are self-contained Markdown. Once a file exists at `.claude/agents/<name>.md` in
  your project and Claude Code has been restarted, they are live — `/agents` will list them. Adapt them
  per the section above before first use, not after.
- **The corpus folder** has to exist before the first run, with whatever layout your profiles declare.
  The reviewer globs for PDFs at a path it derives from the topic folder name; if the folder is not
  there, it has nothing to glob.
- **`notebooklm` is the only piece with a real dependency** — the `notebooklm-py` Python package plus a
  one-time Google OAuth login. Its `SKILL.md` carries the current install and auth commands; read them
  from there rather than from me, since that package moves. Skip it entirely if your papers are PDFs.

The path of least resistance is to hand the whole thing to your own Claude:

> Read `docs/PAPER_REVIEW_WORKFLOW.md` and the two literature agent profiles in this bundle. Set up
> the equivalent in my project: adapted agent profiles using my layout, the corpus folder, and the
> PDF-fetch skill with my email. Tell me what you changed and what you dropped before writing.

## Rough edges

- `literature-reviewer.md` references a `parallel-literature-review` skill and a `literature-deepdive`
  agent. **Neither exists.** Delete the mentions, or batch a large corpus by hand — spawn several
  reviewer instances yourself, each given an explicit slice of the PDFs and its own `tmp/` batch file.
- Both profiles say "the five professors" and then list six. Cosmetic.
- The vendored pack is a **snapshot** with its `.git` removed — clone it fresh from
  `github.com/Imbad0202/academic-research-skills` if you want updates.
- Track A's output is deliberately long. A 14-paper master review runs to hundreds of KB. That is the
  point — it is a reference document, not a summary — but do not paste one into a chat window.
