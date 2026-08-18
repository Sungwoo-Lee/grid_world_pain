# What is in this harness, and what you have to change

## Purpose

This document is the catalog for the bundle. It exists because a bare folder of 21 agent files and
11 skill folders tells you nothing about which ones are generally useful, which ones only work on my
lab's machines, and which ones you should rewrite before trusting.

Every entry below is marked one of three ways:

| Mark | Meaning |
|---|---|
| ✅ **Portable** | Copy it, use it. Nothing in it assumes my project. |
| 🔧 **Adapt** | The *structure* is what is worth having. The domain content (my RL environment, my file layout, my libraries) needs swapping for yours. Usually 10–30 minutes of editing. |
| 🔒 **Project-only** | Depends on my lab cluster, my training code, or my data formats. Included as a worked example of shape, not as working software. |

If you want the short version: the **researcher agents, the reviewers, the planning agents, and the
memory/diary skills are the transferable part**. The training, config, and GPU machinery is not.

---

## 1. How the pieces fit together

Claude reads three things:

1. **`CLAUDE.md`** — loaded into context on *every* turn. This is where the routing rules live: "for
   any task involving 2+ agents, ask `agent-manager` for a plan first", "never measure performance by
   reward", "never `git add -A`". Keep it short — everything in it is a tax you pay per message.
2. **`.claude/agents/<name>.md`** — a role. Only loaded when that agent is spawned. Each one starts
   with YAML frontmatter that Claude uses to decide *when* to call it:

   ```yaml
   ---
   name: literature-reviewer
   description: Dedicated academic literature reviewer... Trigger phrases: "review these papers", ...
   tools: Read, Grep, Glob, Write, Edit, Bash, WebFetch, Skill, ToolSearch
   model: opus
   ---
   ```

   The `description` is the *only* part the main Claude sees when deciding whether to delegate, so it
   is written as a routing advertisement, not a summary. The `tools` line is a real permission
   boundary — my `literature-reviewer` has no way to touch source code because `Bash` is granted but
   the profile forbids writing outside `docs/project/`, and my `training-runner` genuinely cannot edit
   config files. That containment is most of the value.

   **Note on the `model:` field** — I use `opus` for agents that plan or synthesize and `fable` for
   the reviewers (fast, cheap, and adversarial review does not need the biggest model). If a model
   name is not available on your plan, change it to `sonnet` or delete the line to inherit the
   session's model.

3. **`.claude/skills/<name>/SKILL.md`** — a procedure. Same idea: a `description` line decides when it
   fires, the body is the checklist. Skills can carry helper files (scripts, references, templates)
   alongside the Markdown, and Claude only reads those when the procedure says to.

The rule of thumb I settled on: **if it needs its own context window and its own refusals, make it an
agent; if it is a procedure the current Claude should follow, make it a skill.**

---

## 2. The agents (21)

### 2.1 Orchestration and strategy

| Agent | Portability | What it does |
|---|---|---|
| `agent-manager` | 🔧 | Given a request, returns a **routing plan** — which agents, in what order, what runs in parallel, what the hand-offs are. It deliberately has **no ability to spawn agents itself**; it hands the plan to the main Claude, which executes it. That split keeps one context (the manager's) cheap and readable. Adapt: the roster of agent names is mine. |
| `pi` | ✅ | A Principal Investigator role that holds one specific tension: *"we want a publishable paper, so exploring every thread wastes time — but tunnelling on one thread risks missing the better paper."* At decision points it surfaces 2–4 candidate directions through the `AskUserQuestion` tool, **the human picks**, and the PI logs the decision with its rationale. This is the single most useful agent I have that is not about code. |

### 2.2 Engineering

| Agent | Portability | What it does |
|---|---|---|
| `senior-developer` | 🔧 | Investigates the codebase and writes the plan; later verifies that what got built matches what was planned. Writes only to `docs/develop/`. Deliberately cannot analyze experiment results — that is a different agent, so its context stays about code. |
| `developer` | ✅ | Implements an already-approved plan, runs the tests, reports back. Full code access. Does no planning. |
| `bug-curator` | ✅ | Owns a single `KNOWN_BUGS.md` registry and, crucially, **serves queries against it** — you ask "any known bugs in the config loader?" and it returns the two matching rows instead of loading a 900-line document into your context. A cheap and very effective pattern for any long-lived reference doc. |

### 2.3 Reviewers — four of them, overlapping on purpose

| Agent | Portability | Reviews what, against what |
|---|---|---|
| `plan-reviewer` | ✅ | A **plan, before anyone writes code**. Hunts for: steps with no failure-detectable check, verification that runs through the same code path suspected of being broken, unstated assumptions, data-loss hazards. Also reviews a **finished analysis verdict** before its conclusion is believed. Returns `SOUND` / `SOUND WITH CONCERNS` / `NOT READY`. |
| `code-reviewer` | 🔧 | A code diff against my stack's specific footguns (JAX/Flax: pytree mutation, JIT recompilation triggers, `vmap` axis errors, PRNG key reuse). Swap the footgun list for your language's. |
| `math-reviewer` | 🔧 | Equations in the code against the equations in the cited paper. Checks dimensional consistency and derivation steps. Genuinely useful if your project implements published methods. |
| `env-config-reviewer` | 🔒 | YAML configs against my schema, plus a registry of high-impact settings. Shape is reusable; contents are not. |

The overlap between these is intentional and worth explaining, because it looks like waste. Each one
reads a **different object** against a **different ground truth**: a rule enforced at plan time can
still be violated in the code, and two reviewers independently reaching the same finding is
corroboration, not duplication. The gate people forget is the last one — after an analysis produces a
verdict, `plan-reviewer` checks the verdict *before* anyone acts on it, because a wrong plan costs a
rerun while a wrong verdict becomes a claim in a paper.

### 2.4 Experiments

| Agent | Portability | What it does |
|---|---|---|
| `experiment-designer` | 🔧 | Turns a research question into a design — independent/dependent variables, controls, seed counts, and **pre-registered confirmation and refutation criteria written before the run starts** — then generates the config files. The pre-registration is the part worth stealing. |
| `training-runner` | 🔒 | Pre-flight checks, then launches on our cluster. Hard-wired to our nodes and launcher. |
| `experiment-analyzer` | 🔧 | Post-hoc analysis of runs: fills in the Results/Analysis/Conclusions of an existing design doc, compares runs, diagnoses a single run. Reads WandB. |

### 2.5 Researchers — the domain-expert layer

These write memos and are **hard-locked out of `src/`, `configs/`, and `scripts/`**. Their home is
`docs/project/`. They may *append* signed feedback to another agent's document ("Feedback from
professor-rl — 2026-08-18") but never silently rewrite it.

| Agent | Portability | Domain |
|---|---|---|
| `research-postdoc` | ✅ | First responder to open-ended questions ("is there a connection between A and B?", "give me three ideas for the next paper"). Triages, takes a first pass, escalates to a professor when real depth is needed. |
| `professor-bayesian-brain` | 🔧 | Predictive coding, active inference, free-energy, evidence accumulation |
| `professor-pain-modeling` | 🔧 | Computational pain science; guards against overclaiming "pain-like" behaviour |
| `professor-rl` | 🔧 | PPO, distributional and risk-sensitive RL, world models, POMDPs, exploration |
| `professor-bayesian-nn` | 🔧 | Bayesian neural nets — VI, MC dropout, ensembles, heteroscedastic/evidential, calibration |
| `professor-dl-theory` | 🔧 | DL theory + conditional architectures — fiber bundles, FiLM/hypernets, NTK |
| `professor-neuromodulation` | 🔧 | ACh / NE / DA / 5-HT / opioid systems; biological-plausibility guard |
| `literature-reviewer` | ✅ | Per-paper structured reviews of PDFs. **See the paper-review doc.** |
| `literature-curator` | ✅ | Cross-paper synthesis and thematic regrouping of existing reviews. **See the paper-review doc.** |

**Why six narrow professors instead of one "expert" agent.** Two reasons, both learned the hard way.
First, a narrow scope produces a real refusal: my Bayesian-NN professor will say "that is an RL
question, ask `professor-rl`" instead of confabulating an answer at the edge of its competence.
Second, disagreement becomes visible — when two professors give conflicting framings, that conflict
surfaces to me rather than being averaged into mush inside one agent's head. Swap the six domains for
yours; keep the narrowness.

---

## 3. The skills (11)

### 3.1 Research and writing

| Skill | Portability | What it does |
|---|---|---|
| `academic-pdf-fetch` | 🔧 | Gets the version-of-record PDF of one paper into the reference library. Escalates lazily and stops at the first verified file: Unpaywall/OA lookup → plain `curl` → institutional access → (only with your confirmation) a real headed Chrome to clear Cloudflare. **Adapt:** it assumes the machine's own outbound IP is already a campus IP, which is true for our lab container and probably false for your laptop. Also replace `YOUR_EMAIL@example.com` in the Unpaywall call. Legitimate access only — no Sci-Hub. |
| `notebooklm` | ✅ | Full programmatic access to Google NotebookLM — create notebooks, add sources, generate podcasts, download artifacts. Needs the `notebooklm-py` package and a one-time `notebooklm login`. |
| `academic-research-skills/` | ✅ (third-party) | A vendored pack of four skills — `deep-research`, `academic-paper`, `academic-paper-reviewer`, `academic-pipeline` — covering the research→write→review→revise pipeline. **Not my work**; CC BY-NC 4.0, noncommercial academic use, cite the authors. Covered in the paper-review doc. |
| `summarize-study` | 🔧 | Writes a stand-alone, reader-facing summary of a multi-experiment study: headline verdict, six-bullet take-home, vocabulary section, the pre-registered thresholds, verdict table, what-next, links. Built so a reader who opens none of the links still understands the result. |

### 3.2 Memory — the part I would recommend first

| Skill | Portability | What it does |
|---|---|---|
| `wiki-write` | ✅ | Captures the current conversation as structured insight files under `docs/llm_wiki/` — decision, rationale, rejected alternatives, follow-ups — and updates the topic and global indexes. Fires on "remember this", "wrap up", "we're done". |
| `wiki-read` | ✅ | Recall from that wiki. Two modes: plain-English ("what did we decide about X last week?") and technical (raw indexes, counts, fragmentation audit). |
| `diary` | ✅ | Appends one short row per notable event to `docs/diary/YYYY-MM-DD.md` — a single-glance status board across **parallel Claude sessions**. The backing script takes a file lock, so two sessions writing at once queue rather than clobber. |

The reason these matter: Claude Code's built-in memory holds short rules, but it does not hold *why*
you rejected an approach three weeks ago. The wiki does, in the repo, under version control, and a
cheap index means a new session can find out *whether* the wiki knows something without loading what
it knows. If you take one thing from this bundle, consider taking this.

Requires `scripts/claude/*.py` from this bundle (stdlib-only Python, no project imports).

### 3.3 Infrastructure and workflow

| Skill | Portability | What it does |
|---|---|---|
| `wandb-analysis` | ✅ | Query and compare WandB runs, benchmark training speed, pull metric histories. Needs `scripts/wandb/*.py` (bundled) and your WandB API key. |
| `wake` | ✅ | A self-paced poller: "check X every 20 minutes and tell me when it finishes." Session-bound — it dies when the session ends. For anything that must outlive the session use the built-in `/schedule`. |
| `gpu-status` | 🔒 | Live GPU free/busy across our 14 lab nodes, over SSH. The *pattern* (never pick a machine from a stale spec doc; query live state first) transfers; the node list does not. |
| `trajectory-story` | 🔒 | Step-by-step qualitative read of what an RL agent actually did in a recorded episode, for when the aggregate metrics and the video disagree. Bound to our recording format; the script is **not** included because it imports our source tree. |
| `/ask` (command) | ✅ | Forces a decision into a structured multiple-choice question instead of letting Claude guess. Lives in `.claude/commands/ask.md`. |

---

## 4. What I would copy first, in order

1. **`.claude/agents/` — the researchers and the reviewers.** `research-postdoc`, the professor
   pattern, `literature-reviewer`, `literature-curator`, `plan-reviewer`, `pi`. Rewrite the professors'
   domains for your field and you are done.
2. **The wiki + diary skills** and `scripts/claude/`. Highest long-term payoff, lowest adaptation cost.
3. **`examples/CLAUDE.md.example`.** Read it, then write your own from scratch rather than editing
   mine — most of its length is my project's plumbing. What is genuinely reusable is in the sections
   "Working Principles", "Documentation framing", and the git-safety rules.
4. **`academic-research-skills/`** if you want the peer-review pipeline (or clone it fresh from GitHub).
5. Everything else only if you happen to run RL training on a GPU cluster.

---

## 5. Conventions in `CLAUDE.md` that are doing real work

Copy these ideas even if you copy nothing else. Each one exists because something went wrong.

- **No fallback defaults.** Critical settings are read through a `get_mandatory('key')` accessor that
  raises on a missing key. A silent default is a run that trains for six hours on the wrong setting.
- **Verify actual state, never a re-derivation.** To confirm something works, inspect what the system
  actually produced — the saved config, the real log — not a fresh reload of the code that *should*
  have produced it. Never confirm a hypothesis with a tool that shares the suspected-broken code path;
  it echoes your wrong assumption back as proof.
- **Every doc leads with plain language.** The first ~200 words of any plan, analysis, or review must
  be readable by someone with no prior context: no bare run IDs, no bare config paths, no undefined
  shorthand. Symbols and paths move to later sections. Docs get re-read by future-you and by other
  Claude sessions, and an entry point that assumes context is a doc nobody re-reads.
- **Auto-commit with an explicit pathspec.** Claude may commit without asking at logical boundaries,
  but must stage files *by name* and commit with `git commit -F msg -- <file>`. A bare `git commit`
  sweeps in whatever a **parallel session** has staged. This bundle assumes you may run several Claude
  sessions on one repo at once, which is where most of the odd-looking rules come from.
- **Never `git clean -x`.** Our repo lives on a NAS with no symlink support, so training outputs sit in
  gitignored directories inside the working tree. One aggressive clean cost us weeks of re-training.
- **Measure the thing you care about, not the proxy.** Ours reads "performance is measured in survival
  steps, never cumulative reward". Yours will be different, but stating it once in `CLAUDE.md` stops
  every future agent from quietly optimizing the proxy.

---

## 6. Rough edges you will hit

Honest list, so you are not debugging my mistakes:

- **Two dangling references.** `literature-reviewer.md` and `agent-manager.md` mention a
  `parallel-literature-review` skill and a `literature-deepdive` agent. **Neither exists** — they were
  planned and never built. Delete those mentions or build them; Claude will otherwise occasionally try
  to invoke something that is not there.
- **"Five professors" vs six files.** Several profiles say "the five professors" and then list six.
  Cosmetic, but fix it when you rename them for your domain.
- **Relative links point into my repo.** Agent profiles link to things like
  `docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md`. Those paths will not resolve in
  your project. They are inert — Claude just cannot open the file — but they add noise.
- **The model tiers may not match your plan.** Reviewers are set to `fable`, planners to `opus`.
  Change the `model:` line or remove it to inherit the session model.
- **Helper-script paths are relative to the repo root.** Skills call `scripts/claude/diary_append.py`
  and expect to run from the project root with a specific Python interpreter (mine names a conda env
  explicitly, because letting Claude guess the interpreter went badly). Update those invocations.
- **`trajectory-story` has no script.** Its helper imports our source tree, so I left it out. The
  SKILL.md is there to show the shape.
