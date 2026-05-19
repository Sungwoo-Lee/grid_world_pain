---
title: "Paper Review Policy — NotebookLM-grounded, 4-layer lineage"
topic: meta
status: active
created: 2026-05-19
last_updated: 2026-05-19
phase: null
---

# Paper Review Policy — NotebookLM-grounded, 4-layer lineage

## Purpose

This policy replaces the previous "Claude reads PDFs directly" workflow for literature reviews with a **NotebookLM-grounded pipeline** in which every claim a Claude agent makes about a paper is traceable back to a NotebookLM citation. The motivation is integrity: NotebookLM's Gemini answers are source-grounded and citation-backed, while a direct PDF read by Claude can drift, paraphrase, or invent. By making the upload of a PDF into a NotebookLM notebook the *only* path from a source to a per-paper review, every downstream document — integrated reviews, ideas, critiques, direction memos — inherits that grounding by construction.

The new pipeline has **four layers** in strict order:

1. **Source** — the canonical PDF, uploaded into a NotebookLM notebook the user owns. NotebookLM is the source-of-truth surface for every claim made about that paper.
2. **Per-paper review** — produced *only* via NotebookLM queries (no direct PDF read by Claude). Strictly **project-neutral**: a faithful record of what the paper says, with zero references to this project's specific constructs (FiLM, NMN, hypervigilance, precision modulation, etc.). Lives at `docs/project/references/<topic>/<topic>_lit_review.md` per the existing convention.
3. **Integrated review** — a cross-paper synthesis derived *from* the per-paper reviews, not from raw PDFs. May use project-relevant framing for organisation, but **categories are derived from the corpus itself**, not imposed from project priors. Lives at `docs/project/references/<topic>/<topic>_synthesis.md`.
4. **Derived memos** — ideas, critiques, direction proposals. Built on top of layer 3 (and optionally layer 2 for citations), **never on raw PDFs**. Live wherever the producing agent's profile already says (e.g., `docs/project/concepts/`, `docs/project/directions/`, `docs/project/critiques/`).

A reader can verify whether any claim in layer 4 ultimately rests on a NotebookLM-grounded extraction simply by following the layer-3 → layer-2 → layer-1 chain. If any link breaks, that claim does not belong in the project.

---

## Layer 1 — Source (NotebookLM)

### Coverage check (gate at the front of every review request)

Before a `literature-reviewer` agent (or any equivalent surface) begins per-paper extraction for `<topic>`, it MUST:

1. List the source PDFs on disk: `ls docs/project/references/<topic>/sources/*.pdf`.
2. List the PDFs currently in the user's NotebookLM notebooks: `~/.local/notebooklm-py/bin/notebooklm source list --notebook-id <id>` (or `notebooklm metadata --notebook-id <id>` for the structured form).
3. Compute the difference (on-disk PDFs **not** in any notebook).

If the difference is empty → proceed to per-paper review.

If the difference is non-empty → **halt and ask the user**:

> The following PDFs under `docs/project/references/<topic>/sources/` are not yet in any NotebookLM notebook:
>
> 1. `<filename>.pdf`
> 2. `<filename>.pdf`
> 3. ...
>
> Which notebook should they go into? (Type an existing notebook name from `notebooklm list`, or a new notebook name to create.)

Once the user picks a destination notebook, the agent runs:

```bash
# Create the notebook if the user named a new one:
~/.local/notebooklm-py/bin/notebooklm create "<notebook-name>"

# Add each missing PDF (one at a time so failures are isolated):
~/.local/notebooklm-py/bin/notebooklm source add <path-to-pdf> --notebook-id <id>
```

After all sources are added, the agent re-runs the coverage check. If still non-empty (e.g., upload failed for one file), surface the failures to the user and let them resolve (manual upload via NotebookLM web UI, file too large, OCR issue, etc.). Do NOT silently skip uploads that fail.

### Notebook-per-topic convention

The default convention is **one notebook per `<topic>` folder** (e.g., notebook `FiLM` holds the PDFs under `docs/project/references/FiLM/sources/`). This makes coverage checks cheap and keeps the per-paper review queries scoped tightly to one corpus, which materially improves answer relevance — Gemini's source-grounded retrieval narrows to the active notebook only.

The user is free to override:
- **One big project notebook** is viable on the Pro tier (the ~300-source/notebook ceiling comfortably covers all 141 PDFs). The trade-off is looser query scope — a NotebookLM query for "what does the FiLM corpus say about γ scaling?" will be retrieved against the full project corpus, not just the FiLM papers. Acceptable when the user is explicitly asking cross-topic questions; suboptimal for per-paper extraction.
- **Finer-grained sub-notebooks** (e.g., `FiLM_core` vs. `FiLM_extensions`) when a topic is large enough to warrant the split.

The agent should ask which destination on first encounter and remember the user's choice for that topic in the doc-level frontmatter.

### Hard rule — no direct PDF reads

Under this policy, `literature-reviewer`, `literature-curator`, the five professor agents, and `research-postdoc` **must not** open or pdf-extract PDFs in `docs/project/references/<topic>/sources/`. Every claim about a paper must be traceable to a NotebookLM query in the per-paper review.

**The one allowed exception**: when NotebookLM is unreachable (network down, quota exhausted, notebook deleted), the agent halts the review and reports the blocker to the user. The agent does NOT fall back to direct PDF reading. The user decides whether to wait, switch accounts, or — explicitly and case-by-case — authorise a degraded PDF-read pass for that specific paper.

The `pdf` skill is retained for non-review uses (extracting a figure for a memo, OCR'ing a scanned handout, etc.) but is removed from the `literature-reviewer` agent's input-source table.

---

## Layer 2 — Per-paper review (project-neutral)

### Workflow

For each PDF in the notebook, the `literature-reviewer` agent runs the existing 4-step backbone (TOC → per-section extraction → backbone append → relevance-weighted deep dive) and the Phase 1 / Phase 2 synthesis — **but every extraction is a NotebookLM query**, not a PDF read.

Concretely:

```bash
# Step 1 — TOC
~/.local/notebooklm-py/bin/notebooklm ask "What is the section structure of <paper-title>? List the section and subsection headings in order." --notebook-id <id>

# Step 2 — one query per section
~/.local/notebooklm-py/bin/notebooklm ask "In <paper-title>, what does section '<section-title>' say? Quote the key claims, equations, and results with citations." --notebook-id <id>
```

The "EXTREMELY IMPORTANT: Is that ALL you need to know?" follow-up that NotebookLM appends to every answer is the agent's signal to ask its own follow-ups *within* the per-paper loop, before moving on to the next section.

### Hard project-neutrality rule

Per-paper reviews **contain zero project-specific references**. Concretely, a layer-2 doc must NOT mention by name:

- This project's architectures: FiLM-modulator, NMN, hypervigilance head, precision head, Recurrent PPO, Dreamer-NMN, FiLM-Ensemble, etc.
- This project's experimental constructs: H₁a / H₁b / H₁c predicates, Δ_SS, sameProp, paper-bins, …
- This project's run IDs, config paths, agent names, or commit hashes.
- This project's existing memos by path (e.g., `docs/project/concepts/...`).
- Any phrase shaped like "relevance to this project," "implication for our work," or equivalent.

If the reviewer notices a connection to the project while reading, that observation is parked for layer 3 — it does not go in the per-paper review. The verification check is mechanical: a reviewer (or a regex pass) should be able to read the per-paper section and not be able to tell which research project the reader belongs to.

### What the per-paper review DOES contain

- A plain-English entry-point section (per the project-wide doc-framing rule), naming the paper and its core question, written so a fresh reader can pick up cold.
- The 4-step backbone (TOC + per-section extraction), preserving the paper's own section order.
- Phase 1 (foundational, undergrad-level): regrouped neutral summary.
- Phase 2 (graduate-level deep dive): equations + derivations in LaTeX.
- An appendix preserving the raw section-by-section backbone for traceability.

### NotebookLM citations are inlined

NotebookLM returns citations of the form `[1]`, `[2]`, etc., resolving to specific source-document snippets. The agent inlines those citations verbatim in the per-paper review (do not strip them). A reader can then click through in the NotebookLM web UI to see the exact source snippet — that is the layer-2 ↔ layer-1 traceability surface.

---

## Layer 3 — Integrated review (categorisation from sources)

### When this layer is produced

After all per-paper reviews under a `<topic>` are complete, the `literature-curator` agent produces a single integrated review at `docs/project/references/<topic>/<topic>_synthesis.md`. The integrated review's job is to make the corpus **token-efficient at retrieval time**: a future Claude (or human reader) can read the integrated review first and only descend into individual per-paper reviews when it needs detail.

### Categorisation is corpus-driven, not project-driven

The previous default was for the curator to organise the corpus around themes salient to the project's current research questions (precision, neuromodulation, conditional modulation, …). Under this policy that ordering is **reversed**: the curator first asks NotebookLM what natural categories the corpus itself suggests, then proposes those to the user before writing the synthesis.

Concretely:

```bash
# Ask NotebookLM for the corpus's own taxonomy:
~/.local/notebooklm-py/bin/notebooklm ask \
  "Across all sources in this notebook, what are the 4–8 natural thematic categories that group the papers? For each category, list the papers that fall into it. Avoid imposing categories from outside the corpus." \
  --notebook-id <id>
```

The agent surfaces the proposed taxonomy to the user via `AskUserQuestion` (or its closest equivalent in the running context):

> NotebookLM suggested the following categories for `<topic>`:
> 1. <category A> — papers X, Y, Z
> 2. <category B> — papers P, Q
> ...
>
> Use this taxonomy / edit it / propose your own?

The user picks or edits the taxonomy. The curator then writes the integrated review using that taxonomy. **Project-relevant phrasing is allowed in layer 3** — but only as a lens *on top of* the corpus-derived categories, never as a substitute for them.

### What the integrated review contains

- A plain-English entry-point section: what's in this corpus, what taxonomy organises it, what the corpus collectively says.
- One section per category, each a cross-paper synthesis citing the relevant per-paper reviews (and through them, NotebookLM citations to the source PDFs).
- A short "links to project" section at the END (optional, marked as such) where project-relevant connections are noted. This is the *only* place in layers 2 and 3 where project-specific phrasing is allowed.
- A traceability table mapping each cited claim back to the per-paper review and (transitively) the NotebookLM source.

### Update protocol

When a new paper is added to the topic's NotebookLM notebook, the per-paper review is added (layer 2), and the integrated review is **regenerated**, not patched. Regeneration is cheap relative to layer 2 because no new NotebookLM queries are needed — the curator works from the existing per-paper reviews and the existing or updated taxonomy.

---

## Layer 4 — Derived memos

Idea memos, critiques, direction proposals, and similar work product live wherever the producing agent's profile already specifies (e.g., `docs/project/concepts/`, `docs/project/directions/`, `docs/project/critiques/`, `docs/project/ideas/`).

Under this policy, the only change is provenance: **layer-4 docs cite layer 3 (integrated review) by default, and layer 2 (per-paper review) only when a specific paper-level claim is being made.** They do NOT cite the raw PDF, and they do NOT bypass the per-paper review to query NotebookLM directly for a generative question — that would re-introduce the project-flavoured drift the per-paper neutrality rule was designed to prevent.

The five professor agents and `research-postdoc` may consult the NotebookLM CLI for verification (e.g., "did paper X actually say this?") — those are *verification* queries, not extraction queries, and their job is to confirm or refute a claim already in the layer-2/3 chain. They do not produce per-paper reviews themselves.

---

## Migration of existing reviews

There are 16 topic folders under `docs/project/references/` and roughly 141 PDFs. The project's NotebookLM account is on the **Pro tier** (Gemini Advanced / Google One AI Premium), which provides a substantially higher daily query budget than the free tier (typically ~500 queries/day vs. 50, and ~300 sources/notebook vs. 50 — exact ceilings are set by Google and may change). A typical paper consumes 6–10 queries (1 TOC + 5–9 section queries + 0 for synthesis), so 141 papers ≈ 850–1400 queries. At the Pro rate that fits inside ~3–5 working days end-to-end — though clock-time per paper is dominated by browser-automation latency, not the daily ceiling, so the practical pace is more like one topic per working session.

The migration is **pilot-first**, not big-bang, regardless of the rate envelope:

### Stage 0 — Pilot topic

The user picks one topic to migrate first. The candidate that gives the most learning per dollar is one that (a) is small enough to finish in 1–2 NotebookLM days, (b) has an existing PDF-direct review to compare against, and (c) is already familiar to the user so they can judge whether the new layer-2 output is acceptable. Reasonable candidates: `Hypernetwork` (small corpus), `Fiber_bundle` (small, recent), or `Bayesian_Neural_net` (small-to-mid).

The agent does NOT pick the pilot — the user does, after seeing the candidate list.

### Stage 1 — Run the pilot under this policy

The `literature-reviewer` agent executes layers 1–3 for the chosen topic under the new policy. Output: layer-2 `<topic>_lit_review.md` and layer-3 `<topic>_synthesis.md`, both fully NotebookLM-grounded.

### Stage 2 — User reviews the pilot

The user reads the pilot output (layer 2 and layer 3) and decides:
- Does the hard-neutrality rule produce reviews the user can actually use?
- Is the corpus-driven taxonomy in layer 3 a real improvement, or does the project-driven taxonomy serve the user better?
- Is the NotebookLM coverage-check and upload workflow tolerable, or are there friction points to fix?

If revisions to this policy are needed, they happen here — before any other topic is migrated. The pilot is a sacrificial first attempt by design.

### Stage 3 — Sequential rollout

After the pilot's lessons are folded back into the policy, remaining topics are migrated in a user-chosen order, paced by the NotebookLM rate limit. Each topic's pre-migration review is moved into `docs/project/references/<topic>/_archive/` (preserving `git log --follow`) with a one-line frontmatter note `policy: pre-2026-05-19`. The new review lands under the canonical filename.

The user may freeze migration at any point — partially-migrated state is acceptable, since each topic is independent.

---

## Files this policy will touch (follow-up turn)

This policy doc is the *first* step. None of the affected agent profiles or skill descriptions have been edited yet. In a follow-up turn (after the user approves the policy), the following files will be updated:

- `.claude/agents/literature-reviewer.md` — replace the `pdf` skill default with the NotebookLM-grounded workflow described in layer 2; remove direct-PDF-read paths; add the coverage-check gate.
- `.claude/agents/literature-curator.md` — add the corpus-driven taxonomy mechanism in layer 3; tighten the project-neutrality boundary so the curator owns the layer-2 → layer-3 lens shift.
- `.claude/agents/professor-*.md` (5 profiles) and `.claude/agents/research-postdoc.md` — add the "verification-only, not extraction" boundary; route derivation work through layer 3 first.
- `CLAUDE.md` — add a one-line pointer to this policy in the Researchers section so future-Claude reads it before starting any review-shaped task.
- `docs/develop/INDEX.md` — auto-regenerated by `scripts/regen_dev_index.py` after this doc is committed.

---

## Verification check (per the project doc-framing rule)

Open this doc and read the first ~200 words. A reader who has never seen the prior memos / agent profiles / commit history should be able to answer:

- **What is this doc about?** A policy that routes every paper-review extraction through NotebookLM instead of direct PDF reads.
- **Why does it exist?** Because direct PDF reads by Claude can drift; NotebookLM provides source-grounded, citation-backed extraction by construction.
- **What is it claiming?** That a 4-layer pipeline (source → neutral per-paper → corpus-driven integrated → derived memos) produces work product that is verifiable end-to-end, while a direct-PDF pipeline does not.

If those three questions can be answered from the Purpose section without scrolling, the entry-point rule is satisfied.

---

## Open questions for the user (before any agent profile edits)

These are deliberately surfaced here so they get resolved before the follow-up turn touches agent profiles:

1. **Pilot topic** — which folder under `docs/project/references/` should be the pilot? Candidates suggested above are `Hypernetwork`, `Fiber_bundle`, `Bayesian_Neural_net`. User picks.
2. **Notebook-per-topic confirmation** — is one notebook per `<topic>` folder the right default, or should the agent suggest a different scheme (e.g., one project-wide notebook with tag-based filtering)?
3. **Layer-3 user-approval gate** — every time the curator produces a new integrated review, should the corpus-driven taxonomy be surfaced to the user for approval (current draft), or should the curator decide and the user review only the final synthesis?
4. **Existing reviews — archive vs. delete** — the migration plan says "move to `_archive/`". Confirm — or would you prefer the old reviews are deleted outright once the new layer-2 doc lands?
5. **Failure mode for NotebookLM rate-limit hit mid-review** — on the Pro tier this is an edge case rather than a routine event (~500 queries/day vs. ~850–1400 total for the full migration). Current policy says "halt and surface to user." Alternative: pause the review and resume the next day automatically. Worth keeping the answer on file in case the project ever drops back to the free tier, or a single notebook unexpectedly gets a higher per-notebook ceiling.

These questions are not blocking the policy itself — answers can be filled in inline once you've reviewed the rest of the doc.

