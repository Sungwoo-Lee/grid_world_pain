# Claude Code harness — shareable bundle

This folder is a copy of the Claude Code setup I use day-to-day on an RL / computational-neuroscience
research project. It is packaged so you can drop it into your own project and have the same agents and
skills available immediately.

Nothing here is a plugin or an install — Claude Code reads plain Markdown files out of a `.claude/`
folder in your project. Copy the folder, restart Claude, done.

## What "agents" and "skills" actually are

Three layers, and it helps to keep them straight:

| Layer | Lives in | What it is | When it loads |
|---|---|---|---|
| **Project rules** | `CLAUDE.md` at the repo root | House rules Claude must follow — conventions, hard prohibitions, where things go | Every single turn |
| **Agents** | `.claude/agents/*.md` | A specialist sub-Claude with its own system prompt, its own tool permissions, and its own model. It runs in a **separate context window** and reports a summary back | When the main Claude delegates a task to it |
| **Skills** | `.claude/skills/<name>/SKILL.md` | A written procedure for one recurring job. Claude reads the description line, decides it is relevant, and follows the steps | On matching trigger phrases, or on `/<skill-name>` |

The practical difference between an agent and a skill: an **agent** is *someone* (a role with a
personality, a scope, and things it refuses to do), a **skill** is *something to do* (a checklist that
the current Claude follows itself). Agents cost a fresh context window and give you isolation — they
are how you stop a 40-paper literature review from eating the context you need for coding. Skills cost
almost nothing and give you consistency.

## Install into your project

```bash
cd /path/to/your/project

# 1. Agents and skills (the .claude folder is HIDDEN — use ls -a to see it)
cp -r /path/to/this/bundle/.claude .

# 2. Optional: the helper scripts some skills call
cp -r /path/to/this/bundle/scripts .

# 3. Optional: start your own CLAUDE.md from mine
cp /path/to/this/bundle/examples/CLAUDE.md.example ./CLAUDE.md

# 4. Restart Claude Code, then check they were picked up
claude
> /agents          # lists agents Claude can see
> /gpu-status      # any skill name works as a slash command
```

If you already have a `.claude/` folder, copy the subfolders instead of the whole thing
(`cp -r bundle/.claude/agents/* .claude/agents/`) so you do not clobber your own settings.

## What is in here

```
.claude/
  agents/          21 agent profiles (one Markdown file each)
  skills/          10 of my own skills + one vendored third-party pack
  commands/        1 slash command (/ask)
scripts/
  claude/          machinery behind the diary + wiki skills (portable)
  wandb/           WandB query/compare helpers (portable)
  lab/             GPU-status query across our cluster (needs adapting)
examples/
  CLAUDE.md.example        my live project rules — read this to see how it all wires together
  docs/AGENT_PLAYBOOK.md   who-follows-whom orchestration patterns
  docs/TEMPLATES/          the two doc templates my agents write into
docs/
  AGENTS_AND_SKILLS.md     ← START HERE. Every agent and skill, what it does, and what you must change
  PAPER_REVIEW_WORKFLOW.md ← the paper-review pipelines, end to end
```

## Read in this order

1. **`docs/AGENTS_AND_SKILLS.md`** — the catalog. Each entry is marked portable / needs-adapting /
   project-only, so you can tell in one pass what is worth keeping.
2. **`docs/PAPER_REVIEW_WORKFLOW.md`** — the literature and peer-review pipelines in detail, since
   that is the part you said you cared about most.
3. **`examples/CLAUDE.md.example`** — the glue. This is the file that tells Claude *when* to reach for
   which agent. Roughly half of it is generic advice you can lift verbatim; the other half is my
   project's plumbing (lab GPU nodes, config schema, JAX conventions). The catalog says which is which.

## Two honest caveats

- **Some of this will not work out of the box for you.** Anything that touches our lab cluster, our
  training configs, or our evaluation-recording format is project-only. It is included as a worked
  example of *shape*, not as working software. The catalog flags every such case.
- **`.claude/skills/academic-research-skills/` is not mine.** It is a third-party pack
  (github.com/Imbad0202/academic-research-skills, CC BY-NC 4.0 — noncommercial academic use only).
  I have vendored it here with its `.git` removed. If you use it, cite it as its authors ask
  (see `POSITIONING.md` inside that folder) and consider cloning it yourself so you get updates.
