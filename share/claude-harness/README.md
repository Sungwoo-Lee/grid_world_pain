# Claude Code harness — a worked example you can adapt

This folder is my day-to-day Claude Code setup on an RL / computational-neuroscience research project:
21 agents, 11 skills, and the project rules that tell Claude when to use which.

**It is not an installer, and you should not copy it wholesale.** Roughly half of it encodes my
project — my lab's GPU cluster, my config schema, my JAX conventions, my file layout. Copied verbatim
into your repo, those parts are dead weight that Claude will still read and occasionally try to act on.

Treat it instead as a **worked example**: read how the mechanism works, decide which pieces you want,
and have your own Claude build them for your project. There is a suggested way to do exactly that at
the bottom of this file.

---

## How Claude Code actually finds this stuff

There is no registry and no install step. Claude reads plain Markdown out of conventional locations,
every time it starts:

| What | Where | Loaded when |
|---|---|---|
| **Project rules** | `CLAUDE.md` at the repo root | Every single turn, always |
| **Agents** | `.claude/agents/<name>.md` (project) or `~/.claude/agents/` (all your projects) | Only when that agent is delegated a task |
| **Skills** | `.claude/skills/<name>/SKILL.md` (project) or `~/.claude/skills/` | When its trigger matches, or on `/<skill-name>` |
| **Slash commands** | `.claude/commands/<name>.md` | When you type `/<name>` |

Project-level definitions win over user-level ones with the same name. `.claude/` is a hidden folder —
`ls -a` to see it. Restart Claude Code after adding files; `/agents` lists what it can currently see.

### What an agent file looks like

```yaml
---
name: literature-reviewer
description: Dedicated academic literature reviewer... Produces a master review document at
  docs/project/references/<topic>/<topic>_lit_review.md... Trigger phrases: "review these papers",
  "literature review of <folder>", "extract findings from this PDF".
tools: Read, Grep, Glob, Write, Edit, Bash, WebFetch, Skill, ToolSearch
model: opus
---

You are the Literature Reviewer on this project...
(the body is the agent's system prompt: its job, its scope, and what it refuses to do)
```

Four things are doing work here, and they are the four things worth understanding before you build
your own:

- **`description` is a routing advertisement, not a summary.** It is the *only* part the main Claude
  sees when deciding whether to delegate. Write it for the dispatcher: what the agent does, when to
  reach for it, explicit trigger phrases, and what it is *not* (mine say things like "distinct from
  `code-reviewer`, which reviews written code rather than plans"). A vague description means an agent
  that never gets called, or one that gets called for the wrong things.
- **`tools` is a real permission boundary.** An agent without `Write` cannot write. This is how you
  make a research agent that is structurally incapable of touching your source tree.
- **`model` picks the tier per role.** I use a big model for agents that plan or synthesize, and a
  fast one for reviewers — adversarial review does not need the largest model. Omit the line to
  inherit the session's model.
- **The body is a system prompt, and its most valuable half is the refusals.** Every one of my agent
  files has a "What You Do NOT Do" section. That is what stops an agent from drifting into another
  agent's job and quietly duplicating work in a context you cannot see.

### What a skill file looks like

Same idea, smaller: `.claude/skills/<name>/SKILL.md` with a `name` and a `description` in frontmatter,
and a procedure in the body. The description decides when it fires. A skill folder can carry helper
files — scripts, reference documents, templates — and Claude reads those only when the procedure tells
it to, so a skill can be large without costing context until it is used.

### Agent or skill?

**An agent is *someone*; a skill is *something to do*.** An agent gets its own context window and its
own refusals — that is how a 40-paper literature review stops eating the context you need for coding.
A skill is a checklist the current Claude follows itself: nearly free, and what you want for
consistency on a recurring job.

---

## What is in the bundle

```
.claude/
  agents/          21 agent profiles
  skills/          10 of my own skills + one vendored third-party pack
  commands/        1 slash command (/ask)
scripts/
  claude/          machinery behind the diary + wiki skills (stdlib-only Python, portable)
  wandb/           WandB query/compare helpers (portable)
  lab/             GPU-status query across our cluster (example only — our node list)
examples/
  CLAUDE.md.example        my live project rules — the file that wires everything together
  docs/AGENT_PLAYBOOK.md   who-follows-whom orchestration patterns
  docs/TEMPLATES/          the two doc templates my agents write into
docs/
  AGENTS_AND_SKILLS.md     ← START HERE. Every agent and skill, and how much of it transfers
  PAPER_REVIEW_WORKFLOW.md ← the paper-review pipelines, end to end
```

## Read in this order

1. **`docs/AGENTS_AND_SKILLS.md`** — the catalog. Each entry is marked portable / needs-adapting /
   project-only, so one pass tells you what is worth having.
2. **`docs/PAPER_REVIEW_WORKFLOW.md`** — the literature and peer-review pipelines in detail.
3. **`examples/CLAUDE.md.example`** — the glue: the always-loaded file that tells Claude *when* to
   reach for which agent. Read it for shape. Write your own rather than editing mine — the sections
   that generalize are "Working Principles", "Documentation framing", and the git-safety rules;
   everything else is my plumbing.

---

## The way I would actually set this up

Put this folder somewhere your Claude can read it, open Claude Code **in your own project**, and give
it something like:

> There is a Claude Code harness at `<path to this bundle>` from a colleague. Read its
> `docs/AGENTS_AND_SKILLS.md` first, then look at my project to see what it does and what stack it
> uses. I want the paper-review setup — `literature-reviewer`, `literature-curator`, and
> `academic-pdf-fetch`. Write adapted versions into my `.claude/`, using my file layout instead of
> theirs, and drop the parts that only make sense in their project. Tell me what you changed and what
> you left out before you write anything.

Then iterate: use it for a week, and when something is consistently wrong, tell Claude to fix the
agent profile rather than correcting it in conversation each time. That is the whole loop — these
files are just prompts you edit.

Two reasons this beats copying the files. Your Claude can read *your* repo, so it will get the paths,
the language, and the domain right, which is most of the adaptation work. And it will not silently
carry over my project's assumptions — several of my agent profiles link to documents that exist only
in my repo, which are inert but noisy, and my reviewers are written against JAX/Flax footguns that are
irrelevant if you are not using JAX.

If you would rather start from the raw files and edit by hand, that works too — the catalog's rough-edges
section lists everything you would need to strip.

---

## Two caveats

- **Some of this cannot work for you.** Anything touching our lab cluster, our training configs, or
  our evaluation-recording format is project-only. It is here as an example of *shape*. The catalog
  flags every such case.
- **`.claude/skills/academic-research-skills/` is not mine.** Third-party pack
  (github.com/Imbad0202/academic-research-skills, CC BY-NC 4.0 — noncommercial academic use only),
  vendored here with its `.git` removed. If you use it, cite it as its authors ask (see
  `POSITIONING.md` inside that folder), and prefer cloning it fresh so you get updates.
