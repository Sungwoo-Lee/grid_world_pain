---
title: "Continual-learning probe program — tracker + plan + progress log"
status: active-program
audience: user, experiment-designer, training-runner, experiment-analyzer, pi
last_updated: 2026-05-15
related_idea_memo: docs/project/ideas/20260513_continual_learning_probe_followups.md
---

# Continual-learning probe program — tracker

## What this document is

A living tracker for the four-probe continual-learning program that follows up on the project's first clearly-positive FiLM-modulator finding. It contains the high-level intuitive explanation, the four probes in compact form, the step-by-step plan, and a progress log that gets updated as cells launch and verdicts come in.

The detailed probe descriptions, professor reviews, and candidate-mechanism analysis live in the **idea memo**: [`docs/project/ideas/20260513_continual_learning_probe_followups.md`](20260513_continual_learning_probe_followups.md). This tracker is the entry point for someone who wants the program at-a-glance; the idea memo is what you read if you want the reasoning behind each probe.

---

## What we just found (plain English)

A reinforcement-learning agent that survives in a small grid world has two versions: a **plain** version and a **modulated** version (the modulated version has an extra small side-network that scales and shifts the main brain's activity). Until last week, the two versions performed indistinguishably across a year of experiments.

Then last week we ran the agent on a five-stage schedule where the predator's behaviour toggles: active hunter → passive wanderer → active hunter → passive wanderer → active hunter. On the two *returns* to the active-predator stage — the same world the agent had trained on before — the modulated agent survived roughly twice as long as the plain agent, a gap about 25× larger than the project's seed-to-seed noise floor.

In plain English: **when the world changed back to a setting the agent had seen before, the modulated agent remembered how to cope; the plain agent had largely forgotten.**

The modulator's behaviour during that win matched the timing pattern of the brain's **noradrenaline / locus-coeruleus system** — sharp bursts at moments of world-change, quiet during steady state. We have one positive seed on one schedule. The next batch's job is to lock in *why* this happens and on what footing it can become a paper.

## What the program needs to answer

Two questions, in order:

1. **Validate** — is the win real and general? (Replication, perturbation tests.)
2. **Isolate** — *what exactly* gives the modulator its advantage? Of seven candidate explanations, which survive?

User's priority for this round: **paper-shape findings first; verification deferred until after the headline finding lands.** Single seed per cell. No grid search.

---

## The four probes

| # | Probe | What it tests (plain English) | Role | Status |
|---|---|---|---|---|
| **1** | **Replication** | Same schedule with multiple seeds + a stage-length-perturbed cell. Is the win real and not single-seed luck? | Validation gate | **Deferred** (verification) |
| **2** | **Peri-boundary parameter freeze** | Stop the modulator from learning, but only in narrow windows around each stage transition. Does the modulator earn its keep by *what it stored earlier* or by *what it does at the moment the world changes*? | Mechanism — when does the modulator fire? | **Planning** |
| **3** | **Expressivity-matched controls** | Give the plain agent the same multiplicative-conditioning capacity through a gated / batch-norm baseline. Can the plain agent close the gap if given one extra signal? | Defensive — rule out "modulator is just a context tag" | **Planning** |
| **4** | **Long dormancy** | Stretch one passive stage 2–3× and ask whether the modulator's gain on the next return survives, decays, or grows. Separates biological-consolidation from gradient-routing from plasticity-rescue. | Mechanism — does the gain survive a long quiet stretch? | **Planning** |

---

## Compute slice + constraints (this round)

- **GPUs**: nodes `n106` → `n110`, only `cuda:0` and `cuda:1` on each. **5 nodes × 2 GPUs = 10 GPUs total.**
- **Single seed per cell.** No multi-seed replication this round (verification is deferred).
- **No grid search.** Highest-information cells only.
- **Paper-shape over verification.** Probe 1 (replication) is explicitly off the menu for this round.

---

## Step-by-step plan

| Step | Action | Owner | Cells / GPUs | Wall-clock | Status |
|---|---|---|---|---|---|
| 1 | **Senior-developer dev prerequisites** — land 4 logging hooks (`mod_h` hidden vector, per-group gradient norm, one-step temperature derivative, Hessian-sharpness eval) + peri-boundary stop-gradient mechanism (Probe 2) + gated and CBN baseline configs (Probe 3) | `senior-developer` → `developer` → `code-reviewer` | n/a | ~1–2 weeks | **Pending** |
| 2 | **Probe 4 design + launch** — minimal dev needed (just a stretched-passive YAML); can launch first | `experiment-designer` → `training-runner` | 2 cells, 2 GPUs | ~18 h training (1 overnight) | **Pending** |
| 3 | **Probe 4 analyzer pass** | `experiment-analyzer` | 0 | ~hours | Pending |
| 4 | **Probe 2 design + launch** (after Step 1 lands) | `experiment-designer` → `training-runner` | 5 cells, 5 GPUs | ~12 h training (1 overnight) | Pending |
| 5 | **Probe 2 analyzer pass** | `experiment-analyzer` | 0 | ~hours | Pending |
| 6 | **Probe 3 design + launch** (after Step 1; can run parallel with Step 4 on remaining 5 GPUs) | `experiment-designer` → `training-runner` | 5 cells, 5 GPUs | ~12 h training (1 overnight) | Pending |
| 7 | **Probe 3 analyzer pass** | `experiment-analyzer` | 0 | ~hours | Pending |
| 8 | **Verdict synthesis** — PI + postdoc look at all three verdicts together, pick the headline paper-shape finding | `pi` + `research-postdoc` | 0 | ~hours | Pending |
| 9 | **Verification pass** — multi-seed replication of WHICHEVER probe carried the paper. Probe 1 may fold in here. | `experiment-designer` → `training-runner` | tbd by Step 8 | ~1–3 overnights | **Out of scope for this round** |

### Calendar shape

```
Week 1   ████ Senior-dev critical path (logging + mechanisms)
          ▓ Probe 4 launches (overnight)
          ░ Probe 4 verdict
Week 2   ████ Dev finishes
          ▓▓ Probe 2 + Probe 3 launch (parallel, 1 overnight)
          ░░ Verdicts
          ◆ Verdict synthesis
Week 3+  (Verification round — separate scope decision)
```

`█` dev · `▓` GPU training · `░` analyzer pass · `◆` PI decision

**First GPU launches:** ~day 1–3 (Probe 4 needs only a YAML change).
**First paper-shape verdict:** ~day 4 (Probe 4 alone) or ~day 12 (Probes 2 + 3 added after dev).

---

## Progress log

Each row is an event that moves the program forward. Newest at top. Edit in place — this is the living record.

| Date | Event | Detail / link |
|---|---|---|
| 2026-05-15 | Program tracker doc created | This file. |
| 2026-05-14 | PI prioritisation in progress | Three candidate paths surfaced (Path A: Probe 2+4 batched; Path B: Probe 4 alone; Path C: Probe 3 alone). User decision pending. PI call to be logged under `docs/pi/calls/`. |
| 2026-05-14 | User constraints locked | 10-GPU slice on n106–n110 cuda:0+1; single seed; no grid search; paper-shape first; verification deferred. |
| 2026-05-13 | Idea memo + 2 professor reviews + postdoc synthesis | [Idea memo](20260513_continual_learning_probe_followups.md). NE/LC framing adopted; "regime switch" not "task switch"; Probe 2 redesigned to peri-boundary parameter-freeze 2×2; Probe 3 redesigned to gated + CBN baselines; Probe 4 (long dormancy) added. |
| 2026-05-13 | Anchor finding memorised | [Memory insight](../../../docs/memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md). |
| 2026-05-12 | R2 continual sister pair finishes | First clearly-positive FiLM finding: +107 / +132 steps on the two return-to-active stages (~25× seed-noise floor). [Design doc §5.5](../../experiments/active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md). |

---

## Decision still pending

The user has not yet picked the first batch's probe scope. Three live paths:

- **Path A** — Probe 2 + Probe 4 batched (mechanism + biology in one overnight; 7 cells; ~2 weeks dev critical path).
- **Path B** — Probe 4 alone (biology-only; 2 cells; minimal dev; could launch this week).
- **Path C** — Probe 3 alone (defensive null; 5 cells; modest dev; no paper headline of its own).

The PI agent recommended a path-B-first, path-A-next sequencing under the original (multi-seed) constraint. Under the no-multi-seed constraint the user just imposed, Path A becomes the most paper-shape-per-overnight option and is the natural new default. **The next action is the user picking the scope here, then this tracker's "Step-by-step plan" and "Progress log" tables both get updated.**

---

## Hand-offs (once scope is picked)

- `experiment-designer` — turn the chosen probe(s) into pre-registered designs + matching YAML configs.
- `senior-developer` — plan the dev prerequisites (logging hooks + stop-gradient mechanism + baseline configs); hand off to `developer`.
- `training-runner` — launch on n106–n110 cuda:0+1.
- `experiment-analyzer` — verdict after each probe finishes; fill in the design doc's Results / Conclusions sections.
- `pi` — verdict synthesis after all three verdicts in hand; pick the paper-shape headline.

---

## Links

- **Idea memo** (the full reasoning): [`20260513_continual_learning_probe_followups.md`](20260513_continual_learning_probe_followups.md).
- **Anchor finding** (the R2 win this program follows up on): [`NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md`](../../experiments/active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md).
- **Re-summary of the study** (reader-facing): [`20260513_0321_nmn_comparison_study.md`](../../experiments/summaries/20260513_0321_nmn_comparison_study.md).
- **Headline memory insight**: [`20260513_0014`](../../../docs/memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md).
- **Logging-gap blocker insight**: [`20260513_0017`](../../../docs/memory/memories/nmn_diagnosis/20260513_0017_mod_h_logging_gap_blocks_cka_precheck.md).
