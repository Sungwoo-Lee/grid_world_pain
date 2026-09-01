## Agent Team

Delegate to the matching agent — read its profile in `.claude/agents/` for full responsibilities and tool scope.

| Agent | Model | Role | Scope |
|---|---|---|---|
| [agent-manager](.claude/agents/agent-manager.md) | opus | Plan multi-agent flows; route + sequence + parallelize. Returns a routing plan to the parent (top-level Claude), which spawns the sub-agents | returns plan; writes nothing; does NOT spawn |
| [pi](.claude/agents/pi.md) | opus | Principal Investigator — portfolio-level focus-vs-explore calls at major decision points. Surfaces 2–4 candidate paths via `AskUserQuestion`; user decides; PI logs the call. Manual + proactive triggers (pre-launch, post-analysis, roadmap-level plan, new-direction proposal) | `docs/pi/` (primary); cross-process feedback append-allowed under any `docs/` subtree |
| [senior-developer](.claude/agents/senior-developer.md) | opus | Platform-development planning + post-impl verification | `docs/develop/` |
| [developer](.claude/agents/developer.md) | opus | Implement approved plans, test, report | full code |
| [code-reviewer](.claude/agents/code-reviewer.md) | fable | JAX/Flax/vmap/PRNG correctness review | `docs/reviews/` |
| [math-reviewer](.claude/agents/math-reviewer.md) | fable | Verify equations match cited papers | `docs/reviews/` |
| [plan-reviewer](.claude/agents/plan-reviewer.md) | fable | Adversarial review of (a) any drafted plan before code is written or training launched — unverifiable steps, circular verification, unstated assumptions, project-rule violations, data-loss hazards — and (b) any finished analysis verdict before it is believed: does the evidence shown actually support the conclusion | `docs/reviews/` (Critical findings only); appends signed feedback to the plan doc |
| [env-config-reviewer](.claude/agents/env-config-reviewer.md) | fable | YAML/env soundness, obs↔noise sync, pre-flight before training | `docs/reviews/` |
| [artifact-format-reviewer](.claude/agents/artifact-format-reviewer.md) | fable | **Renders** a generated artifact page in headless Chrome and reports what is visually broken. Format only — layout, overflow, legibility, colour consistency. Runs before every artifact publish | reports only; writes nothing |
| [bug-curator](.claude/agents/bug-curator.md) | opus | Owns + serves the Known Bugs registry — returns only the rows matching a query so callers skip the full doc; records/updates bugs. Does NOT fix code | `docs/develop/active/issues/KNOWN_BUGS.md` |
| [experiment-designer](.claude/agents/experiment-designer.md) | opus | Experiment design + config generation | `configs/`, `docs/experiments/active/<topic>/` |
| [experiment-analyzer](.claude/agents/experiment-analyzer.md) | opus | Post-hoc training-result analysis (WandB, run comparisons) | `docs/experiments/active/<topic>/` |
| [training-runner](.claude/agents/training-runner.md) | opus | Pre-flight check + launch training on lab nodes (101–114) via `run_command.py`; configs are read-only | `train_command-new.sh` |

For known-bug context ("is this a known issue in X?"), **consult `bug-curator`** — it returns only the matching rows — rather than reading the full `docs/develop/active/issues/KNOWN_BUGS.md` into context.

**Reviewer sequencing (owned by `agent-manager`):** the team has five reviewers covering six objects — `plan-reviewer` (a drafted plan, and separately a finished analysis verdict), `math-reviewer` (equations against the cited paper), `code-reviewer` (a diff against JAX/Flax conventions), `env-config-reviewer` (YAML against the schema + critical-settings registry), `artifact-format-reviewer` (a generated page against its own rendering). Their coverage **deliberately overlaps**, and that is a feature: each reads a different object against a different ground truth, so a rule enforced at plan time can still be broken in the code, and two reviewers reaching the same finding is corroboration rather than waste. Do not trim a reviewer's checklist because another reviewer "owns" that check. The analysis gate is the one that is easy to forget: after `experiment-analyzer` produces a verdict, `plan-reviewer` runs before `pi`, because a wrong plan costs a rerun while a wrong verdict becomes a claim in a paper — and `pi` asks whether to continue, not whether the inference holds. Which reviewers a given job needs, and in what order, is `agent-manager`'s call — it sequences upstream-first (plan → config → code, since a finding is cheapest to fix earliest), parallelizes same-stage reviewers, and omits any reviewer whose object does not exist yet. Where two reviewers disagree on a verdict, both go to the user; nobody arbitrates silently.

### Researchers

Domain-expert agents that generate mathematical concepts, literature reviews, and publication-direction memos for the project. Their **primary write home is `docs/project/`** (their own standalone memos), but they may also **append cross-process feedback** to any doc under `docs/` — including `docs/develop/`, `docs/experiments/`, `docs/reviews/`, `docs/pi/` — when invited to comment on an in-flight plan, design, analysis, or strategic call. They **never** edit `src/`, `configs/`, or `scripts/`. Cross-process feedback must always **append** (never silently rewrite) and be signed with a clear "Feedback from `<agent-name>`" header so the original author's voice stays distinct. Recommendations that imply code or experiment changes are handed off (named) to `senior-developer`, `experiment-designer`, etc.

| Agent | Model | Role | Primary write home (under `docs/project/`) |
|---|---|---|---|
| [research-postdoc](.claude/agents/research-postdoc.md) | opus | First responder for open-ended research questions; triages to professors or writes first-pass synthesis | `ideas/`, `triage/` |
| [professor-bayesian-brain](.claude/agents/professor-bayesian-brain.md) | fable | Perceptual decision making, predictive coding, active inference, Bayesian decision theory | `concepts/`, `directions/`, `critiques/` |
| [professor-pain-modeling](.claude/agents/professor-pain-modeling.md) | fable | Computational pain science; construct-validity guardian for "pain-like" claims | `concepts/`, `directions/`, `critiques/` |
| [professor-rl](.claude/agents/professor-rl.md) | fable | Reinforcement learning — PPO, distributional / risk-sensitive RL, world models, POMDP, exploration, auxiliary objectives | `concepts/`, `directions/`, `critiques/` |
| [professor-bayesian-nn](.claude/agents/professor-bayesian-nn.md) | fable | Bayesian neural networks only — VI, MC dropout, ensembles, heteroscedastic / evidential / Laplace, calibration, uncertainty disentanglement | `concepts/`, `directions/`, `critiques/` |
| [professor-dl-theory](.claude/agents/professor-dl-theory.md) | fable | Mathematical theory of deep learning + conditional architectures — fiber bundles / gauge theory, geometric DL, FiLM / hypernet / MoE, NTK / mean-field, information geometry | `concepts/`, `directions/`, `critiques/` |
| [professor-neuromodulation](.claude/agents/professor-neuromodulation.md) | fable | Computational models of ascending modulatory systems (ACh / NE / DA / 5-HT / opioid); biological-plausibility guardian | `concepts/`, `directions/`, `critiques/` |
| [literature-reviewer](.claude/agents/literature-reviewer.md) | opus | Per-paper review (section-by-section summary + Phase 1/2 LaTeX) of PDFs in `docs/project/references/<topic>/sources/` | `references/<topic>/<topic>_lit_review.md` (review at topic root, source PDFs in `sources/`) |
| [literature-curator](.claude/agents/literature-curator.md) | opus | Cross-paper synthesis, TOC, thematic regrouping of existing master reviews | `references/<topic>/<topic>_lit_review.md`, `references/<topic>/<topic>_synthesis.md` |

**Researcher routing:** open-ended research questions ("what direction?", "is there a connection between …?", "give me ideas") default to `research-postdoc`, which triages and either writes a first-pass synthesis or hands off to one or more professors / literature agents. Direct invocation is fine when the question is unambiguously in one agent's domain (e.g., "review these PDFs" → `literature-reviewer`, "regroup the lit review by theme" → `literature-curator`).

**Default routing:** for any task involving 2+ agents in sequence — feature add, bug fix, training experiment, literature review of 10+ papers, post-impl verification — spawn `agent-manager` to get a routing plan, then **spawn the named sub-agents yourself** (top-level Claude executes the plan; the manager has no `Agent` tool). Single-agent tasks (e.g., "audit this config", "review this PR", "ask a professor", "/pi") bypass the manager and route directly.

**PI consultation (proactive):** at the major decision points named in [.claude/agents/pi.md](.claude/agents/pi.md) — pre-launch of a multi-run experiment, post-analysis of a multi-run comparison, a roadmap-level plan, a new research-direction proposal — the `agent-manager` flags PI consultation as a canonical step in its routing plan; the parent then spawns `pi`, which surfaces the focus-vs-explore trade-off to the user via `AskUserQuestion`. The user makes the final call. The PI does not run for bug fixes, single-config tweaks, or one-off launches.

**Artifact format review (proactive):** before publishing or republishing **any** artifact — an
analysis report, a results visualization — spawn [`artifact-format-reviewer`](.claude/agents/artifact-format-reviewer.md).
It **renders** the page in headless Chrome at several viewport widths via
`scripts/claude/check_artifact_layout.py`, looks at the screenshots, and checks it against the
known-defect register in [artifact_format_bugs.md](docs/develop/active/meta/artifact_format_bugs.md).
This gate exists because a page shipped with three lists rendering one word per line after **two**
rounds of careful review that read the HTML and CSS without ever rendering them; the user saw it
instantly. Static analysis of a stylesheet cannot find a defect that lives in the box tree. Chrome is
installed in this container — render the page.

**Three requirements every results artifact must meet** (guide §11, enforced at build time, so a
violation is a build failure rather than a review finding): every figure caption states **both axes
in words**, with units, repeating across figures rather than assuming the reader remembers; every
figure declares **how much data it used** — used / available / percentage per subset, with a reason,
emitted by the figure script and never typed by hand; and every **"How it is computed"** block is
written for a colleague who was not in the room, roughly 150–250 words, with any statistical term
glossed where it appears.

**Plan review (proactive):** whenever a plan gets drafted — an implementation / bug-fix / refactor plan from `senior-developer`, or an experiment design + configs from `experiment-designer` — spawn [`plan-reviewer`](.claude/agents/plan-reviewer.md) **before the user approves it**. It runs an adversarial advance failure check: steps with no failure-detectable check, verification that exercises the same code path suspected of being broken, unstated critical assumptions, silent violations of project rules (fallback defaults, reward-based evaluation, unpaired maintenance-contract doc updates), data-loss hazards, and collisions with already-known bugs. Findings come back inline with a one-line verdict (`SOUND` / `SOUND WITH CONCERNS` / `NOT READY`); it writes `docs/reviews/plan_<topic>.md` only when it finds a 🔴 Critical finding. Distinct from `code-reviewer` (reviews code after it exists) and from `senior-developer`'s Verification Protocol (its plan-adherence check) (checks adherence after implementation) — this gate runs on the plan itself, while changing it is still free.

The canonical flows, parallelism heuristics, cross-cutting constraints, and anti-patterns are documented in [docs/AGENT_PLAYBOOK.md](docs/AGENT_PLAYBOOK.md). The manager reads this whenever it produces a routing plan; other agents may reference it for context.

---

## Working Principles

- **Think before coding.** State assumptions; if uncertain, ask. If multiple interpretations exist, surface them — don't pick silently. If a simpler approach exists, say so.
- **Ask actively, not always.** Use the `AskUserQuestion` tool whenever a decision could reasonably go more than one way and picking silently risks rework — ambiguous scope, unstated constraints, multiple plausible interpretations, or trade-offs the user should own (perf vs. simplicity, breaking change vs. shim, which file to touch). Not every task needs a question, but err on the side of asking rather than guessing. Batch related questions into one prompt; don't drip them.
- **Simplicity first.** Minimum code that solves the stated problem. No speculative features, abstractions for single-use code, configurability that wasn't requested, or error handling for impossible scenarios. If 200 lines could be 50, rewrite.
- **Surgical changes.** Every changed line should trace to the request. Don't "improve" adjacent code, refactor what isn't broken, or restyle to your preference. Remove orphans your changes created; don't delete pre-existing dead code unless asked — mention it instead.
- **Goal-driven execution.** Reframe tasks as verifiable goals before starting ("fix the bug" → "write a test that reproduces it, then make it pass"). For multi-step work, state a short plan with a verification check per step so you can loop without re-asking.
- **Plain-language default, details on demand.** When answering — plans, experiment-result analyses, explanations, anything — treat the user as new to the topic. Explain in plain English; translate jargon and shorthand (`H₁a`, `Δ_SS`, FiLM, WandB run IDs, bare config paths) on first use. Don't dump details inline by default — point at the document path that owns the answer (e.g. "details in [`docs/experiments/active/foo/foo.md`](docs/experiments/active/foo/foo.md)") and let the user open it. Go deep only when the user explicitly asks for the details.
- **Verify actual state, not a re-derivation.** To confirm something really works as intended, inspect the ground truth the *system produced* (a run's saved config, real logs/outputs) — not a fresh reload of the source that *should* produce it. Never confirm a hypothesis with a tool that shares the suspected-broken code path; it echoes your wrong assumption back as 'proof' (e.g. checking a training config with the tooling loader that resolves `extends:`, when the trainer's loader doesn't). A user's repeated, specific, contradicting observation is evidence your model is wrong — distrust the model the first time it contradicts them, not the third.
- **Math in answers — renderer constraints.** The chat renderer supports **only `$$...$$` display blocks on their own lines**. Inline `$...$`, `\(...\)` and `\[...\]` do **not** render — they show as raw LaTeX. Separately, markdown emphasis eats underscores in prose (`Z_τ ... f_θ` renders italic with the subscripts silently lost). So: real equations (fractions, sums, integrals) → a `$$` display block; inline expressions containing subscripts → backticks; single symbols → Unicode (γ β τ ψ ⊙ ∈ ℝ). Same rule for docs meant to be read in the renderer.

---

## Project-Wide Rules

- **Conda env.** All Python is run inside the `grid_world_pain` conda env. Invoke the interpreter directly at `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python <script>`. Do not use `conda run` / `conda activate`, and never invoke the system `python3`. (Per-experiment exception: the `sheeprl_bridge` env exists for the upstream-sheeprl drop-in test under [`docs/develop/active/diagnosis/sheeprl_training_howto.md`](docs/develop/active/diagnosis/sheeprl_training_howto.md). Same explicit-interpreter rule — never `conda run`.)
- **Survival-step evaluation.** Performance is measured in survival steps, never cumulative reward.
- **No fallback defaults.** Critical configs use `config.get_mandatory('key')`; missing key → `ValueError`. New keys listed in the plan's File Changes section.
- **Config system.** [docs/environment/CONFIG_GUIDE.md](docs/environment/CONFIG_GUIDE.md) is the canonical config-system reference (`extends:` layering, authoring, schema changes) and carries the Maintenance Contract that keeps it + `02_config_schema.md` in step with the code.
- **Critical config settings.** [docs/environment/CONFIG_CRITICAL_SETTINGS.md](docs/environment/CONFIG_CRITICAL_SETTINGS.md) — registry of high-impact settings (canonical values + meaning) + dated change log. Changing a registry setting requires a same-commit change-log entry; config/training agents consult and maintain it.
- **Scripts dependency map.** [docs/environment/SCRIPTS_DEPENDENCY_MAP.md](docs/environment/SCRIPTS_DEPENDENCY_MAP.md) records who calls every file in `scripts/` (training entry points, `src/` subprocess paths, the test suite, Claude skills/agents, and the settings allowlist) and the `sys.path` repo-root depth hazard. It carries a Maintenance Contract: any change that **adds, moves, renames, or deletes a file under `scripts/`** — or changes a caller of one — must update the map in the **same change**.
- **Lab GPU spec.** [docs/environment/LAB_NODE_GPU_SPEC.md](docs/environment/LAB_NODE_GPU_SPEC.md) — the cluster is **heterogeneous**: RTX 2080 Ti (11 GB, nodes 101/103/104/105), RTX 3090 (24 GB, 106–112), RTX 4090 (24 GB, 102/113), RTX 6000 Ada (49 GB, **node 114 — 4 GPUs `0–3`**); all other nodes have GPUs `0,1` only. Consult before assigning a `node:GPU` for a launch: only use indices that exist on that node, and **match card to job** (routine small rPPO/Dreamer → low/mid tier; reserve 4090/Ada for heavy jobs). For LIVE free/busy state + running processes run `scripts/lab/gpu_status.py` (or the `gpu-status` skill) — the source of truth for what is free right now. **Pack-node-first**: when launching multiple runs, fill all free GPUs on one node before moving to the next (people claim whole nodes) — full policy in the GPU spec doc.
- **No version numbers without permission.** Never invent, assign, or bump a version for anything — the environment, a config, a schema, a document, a flag, or an interface. Version-like strings in commit subjects (`(v3.0)`, `(v3.1)`, `(v3.2)`) are that author's prose, **not** a project scheme: do not treat them as established, do not propagate them into names, filenames, flags or docs, and do not extend the sequence. If a change looks like it needs a version, **ask first**. The same applies to any other convention that merely *appears* established — check it is real before adopting it, and say where you found it.
- **Templates.** Plans go in `docs/` using [issue_plan](docs/TEMPLATES/issue_plan.md) (what to build/fix) or [training_analysis](docs/TEMPLATES/training_analysis.md) (what happened — hypothesis-driven). Cross-link related docs both directions.
- **Develop docs.** Files under `docs/develop/` carry YAML frontmatter (`title, topic, status, created, last_updated`) per [FRONTMATTER_CONTRACT.md](docs/develop/active/meta/FRONTMATTER_CONTRACT.md). `docs/develop/INDEX.md` is auto-generated by `scripts/claude/regen_dev_index.py` — do not hand-edit. When superseding a doc, set `status: superseded`, link with `supersedes:` / `superseded_by:`, and use `git mv` to move it under `archive/` so history is preserved.
- **Doc linking — Foam wikilinks.** Cross-doc references default to `[[filename]]` wikilink form (resolved by Foam in VSCode and Obsidian in the vault — survives archive moves). Existing `[text](path.md)` links remain valid; convert only when touching the file. Load-bearing docs may declare `aliases: [<stable-id>]` in YAML frontmatter for rename-stability. Convention details + archive workflow in [doc_linking_convention.md](docs/develop/active/meta/doc_linking_convention.md).
- **Working files.** Intermediate results to `tmp/YYYYMMDD_HHMMSS_<topic>.md`, written after each step.
- **Parallelism.** OK for independent tasks; sequential for hand-off chains; shell loops beat parallel agents for mechanical batch work.
- **Auto-commit.** Standing authorization to `git add` and `git commit` without asking, at logical sub-task boundaries — one coherent change per commit, separate commits for unrelated work (do not batch). Skip if the change is incomplete (failing tests, half-done implementation). Stage specific files by name (never `git add -A` / `.`) **and commit with an explicit pathspec** — `git commit -F msg -- <file> <file>`. A bare `git commit` commits whatever is in the index, which in this repo includes files a parallel session staged but has not yet committed; the pathspec makes sweeping them structurally impossible. Checking `git diff` on your own files does not protect you here — foreign work is already *staged*, so it is invisible to an unstaged diff (verify with `git diff --cached --name-only` if you must commit without a pathspec). **Parallel sessions edit shared files** — `CLAUDE.md`, agent profiles, the playbook — with no locking, so before staging one, `git diff` it and confirm every hunk is yours. If a hunk is not, leave that file unstaged and say so rather than committing another session's half-finished work under your message. **A locked index means retry, not intervene.** If git reports `Unable to create '.git/index.lock'`, another session is mid-operation — often just a slow `git status`, which takes the lock to refresh the index and is slow on this NAS. Retry the command a few times, seconds apart, **with stderr visible**. Never delete `index.lock` (a live process may hold it; removing it risks index corruption), and never gate the retry on `[ ! -f .git/index.lock ]` — checking then acting is a race that loses on a slow index. Match the existing message style — check `git log` for the project's conventional-commit + gitmoji format. Never commit secrets, never use `--no-verify`, never push.
- **Git safety (DO NOT WIPE GITIGNORED DATA).** This repo lives on a NAS that **does not support symlinks**, so the standard "move data outside the repo and symlink in" protection is unavailable. Gitignored data (`results/`, `claude_data/`, `wandb/`, `logs/`, `tmp/`, `legacy/`, `antigravity_data/`, etc.) is therefore destroyed unrecoverably by any aggressive `git clean`. Hard rules:
  - **Never** use `git clean -x` / `-X` / `-fdx` / `-fdX` — the `-x`/`-X` flag deletes gitignored files. Allowed: `git clean -fd` (untracked-only; gitignored survives). Always preview with `git clean -fdn` (dry-run) first, and surface the listed paths to the user before executing the real clean.
  - **Never** force-checkout / force-switch between branches without first checking whether the destination branch tracks paths that are currently untracked locally — `git checkout -f <branch>` will overwrite an untracked file at a path tracked on the destination.
  - **Avoid** `git stash -u` followed by `git stash drop` — `-u` carries untracked into the stash, and `drop` then loses them. Use `git stash list` + `git stash show -u` to confirm before any drop.
  - **`git reset --hard` is safe** for gitignored data (only resets tracked files); the danger is in `git clean -x` that often follows it.
  - **Before any merge / rebase / branch switch / non-trivial git operation**, snapshot critical untracked data: `cp -a results /tmp/results-bk-$(date +%s)` (or `claude_data`, `wandb`, etc., as relevant). The cost is one second; the alternative is re-training.
  - **Past incident**: the `results/` directory (training outputs across `JAX_RecurrentPPO` / `JAX_DreamerV3` / `JAX_Sandbox` / `JAX_PPO`) was lost during a failed merge when destructive cleanup followed. Recovery required re-training. This rule exists because of that incident.

---

## Documentation framing

Every plan, design, analysis, summary, review, and direction document the project produces must lead with a **plain-language interpretation** that a reader without prior context can follow. Docs in this project get re-read by future-Claude, parallel sessions, the user months later, and (eventually) external collaborators or reviewers — so a doc whose entry point requires the reader to already know `H₁a` / `Δ_SS` / `f96lhxpe` / `01-interoNocicept_sameProp.yaml` is a doc that gates re-readers behind a context tax.

### The rule

The **first body section** of any plan / design / analysis / summary / review / direction doc is a plain-English entry point — title in the family of *Question*, *Purpose*, *Context*, *Headline finding*, *Verdict*, *Study question*, or equivalent. Its 200-word job is to tell a fresh reader **what the doc is about, why it exists, and what it's claiming**.

In that entry-point section:

- **Translate every cited result on first mention.** "The modulator did not beat the baseline (H₁a refuted)", not just "H₁a refuted". The English first; the symbol after, in parens.
- **No bare WandB run IDs** like `f96lhxpe` — link through the design doc or wiki entry that names them.
- **No bare config paths** like `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1.yaml` — describe what the config does ("the FiLM agent config with the modulator's temperature ceiling raised from 3.0 to 10.0") and link the path elsewhere.
- **No bare predicate / shorthand names** (`H₁a`, `H₀`, `Δ_SS`, `Cand. A1`, `Phase 0`, `T/P split`, etc.) without a one-clause translation.
- **Concrete examples beat abstract claims.** If the doc proposes "a 2-context mixture", show what the two contexts ARE — not just "context A vs. context B".

Symbolic / numerical / file-path / equation-heavy detail belongs in **later sections** of the same doc — Methods, Manifest, Links, Derivations, Tables, Appendix. The rule is about the *entry point*, not the whole document. Math-heavy memos still earn their math; they just have to introduce their question without it.

### The check

Open the document, read the first ~200 words. Could a fresh reader who has not seen the prior memos / design docs / commit history understand what this doc is about, why it exists, and what it's claiming? If yes, it passes. If no, rewrite the entry point.

### Where this rule is encoded

- **Templates**: [docs/TEMPLATES/issue_plan.md](docs/TEMPLATES/issue_plan.md) (Context section) and [docs/TEMPLATES/training_analysis.md](docs/TEMPLATES/training_analysis.md) (Research Question section) enforce the structure at the point new docs are authored.
- **Agent profiles**: every doc-producing agent (`senior-developer`, `experiment-designer`, `experiment-analyzer`, `research-postdoc`, the four professors, the three reviewers, `literature-reviewer`, `literature-curator`) carries a one-line reference to this rule in its profile.
- **Worked example**: the `summarize-study` skill ([.claude/skills/summarize-study/SKILL.md](.claude/skills/summarize-study/SKILL.md)) encodes this rule in its strictest form, with a concrete check-and-replace table — read it for a worked example of what good output looks like.

The rule applies to every new doc going forward. **Existing docs are not rewritten after the fact** unless a reader is actively confused by one.

---

## LLM Wiki

This project carries two recall layers; future-Claude must know which one to write to. They are deliberately named apart: **Claude's built-in auto-memory** is the harness's own store, while the **LLM Wiki** is this repo's.

- **Built-in auto-memory** at `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` — short typed rules another agent must obey on every invocation, with a sibling `feedback_*.md` per rule. User-and-machine-local; not under git.
- **In-repo LLM Wiki** at `docs/llm_wiki/` — multi-section session insights with rationale, decisions, follow-ups, and an optional raw-conversation archive. Version-controlled with the repo.

**Consult before acting (the pull gate).** Before any non-trivial task — bug fix, feature, refactor, experiment launch, config change — read the **Active folders table** at the top of `docs/llm_wiki/ROOT_INDEX.md` (~2 KB; stop before the change-log tail, which is another 50 KB) and check whether a folder covers the area you are about to touch. Drill into that folder's `_topic_index.md` only on a match, and open an individual entry only when its one-line summary looks relevant. On no match, proceed. The goal is to learn *whether* the wiki knows something, not to load what it knows — cost model and drill-down levels in `docs/llm_wiki/CLAUDE.md` §3.

When wiki work itself is requested (capture, recall, audit), read `docs/llm_wiki/CLAUDE.md` (operating manual) first. The division-of-labor decision rule and capture triggers live there; the design rationale and worked routing examples live in [docs/develop/active/meta/llm_wiki_system_design.md](docs/develop/active/meta/llm_wiki_system_design.md).

---

## Diary protocol

The project keeps a daily event log at [docs/diary/YYYY-MM-DD.md](docs/diary/) — a single-glance status board across all parallel Claude sessions. **Update it whenever a notable event fires, even without an explicit user request.** Missing entries create invisible gaps when a parallel session looks at "what's happening today". Top-level Claude is responsible for `session-start` / `session-end` and `progress-report` on multi-step wrap-up; sub-agents fire their own subcommand on completion; `/wiki-write` chains `insight` rows automatically.

The full contract — subcommand-to-event mapping, Session-column convention (full UUID vs. 8-char prefix, sub-agent `parent/role` form), link auto-formatting rules, and the per-call auto-commit pattern — lives in the `/diary` skill at [.claude/skills/diary/SKILL.md](.claude/skills/diary/SKILL.md). The skill calls `scripts/claude/diary_append.py`, which holds a `flock` so parallel sessions queue rather than clobber.
