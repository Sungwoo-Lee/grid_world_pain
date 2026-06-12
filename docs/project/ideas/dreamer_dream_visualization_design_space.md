# Visualizing how the Dreamer agent "dreams" — visualization-design + tooling memo

## Purpose (plain-language entry point)

We have a trained world-model agent (`dreamer_srl`) that, at any moment, can **imagine the future**: from where it is now, it can roll its internal model forward several steps without looking at the real world, using its own policy to choose imagined actions. The standard way researchers show this off is a "dream film strip" — a row of imagined camera frames next to a row of the real frames that actually happened, so a reader can see where the dream stays faithful and where it drifts.

Our agent has **no camera**. Its senses are a 27-number vector split across seven very different channels — a hunger scalar, two pain scalars, a five-number smell gradient, a five-cell "is there a wall here" diamond, a six-way "which way did I just move" indicator, and an eight-category "what kind of thing is in my cell" label. You cannot diff two of these as images. So the design question this memo owns is: **how do we render imagined-vs-actual across these mismatched channels so a human can read, at a glance, what the agent's dream got right and where it diverged from reality?**

The headline finding from grounding in the codebase: **most of this is a composition job, not a build-from-scratch job.** The agent already has a decoder that turns its internal dream-state back into a 27-number observation (so we *can* show "what the dream thinks it sees", not just abstract latent vectors), and the renderer already has a two-layer "perceived vs. true" overlay mode that we can feed dream-vs-real into directly. The expensive part is not the visualization — it is that the Dreamer agent has no offline eval/rollout entry point yet, so the recording harness needs new Dreamer-specific glue regardless of which visualization we pick.

**Project anchor.** This is qualitative tooling, not a hypothesis test — it does not change how G1/G2 are tested, which is the honest signal that this is *future-direction / mechanism-inspection* infrastructure rather than a current G1/G2 blocker. Its primary value is at the **representational level of analysis** ([project_plan.md §3.2](../project_plan.md)): a dream-fidelity readout is direct evidence about *what the agent's network represents about damage and threat*, which is exactly the level Paper 1 wants at least one or two categories to carry. It most naturally serves **Recovery** and **Hypervigilance** (does the dream over-predict threat after injury?). It belongs to **Phase 4 (hypervigilance readout)** in spirit, but is usable as soon as a Dreamer checkpoint exists.

> **Scope note.** A parallel professor-rl memo owns the RL/world-model *semantics* (what open-loop divergence means, whether latent KL or reward error is the right fidelity metric, teacher-forcing theory). This memo owns the *practical visualization design + tooling integration*. Where the two touch — e.g. "is decoded-obs divergence meaningful or should we trust only reward/continuation heads" — I flag it and defer the semantics call to professor-rl.

---

## 1. Grounding: what already exists (reuse > rebuild)

Three findings from the code that constrain every design below.

**(F1) The decoder exists — the "no decoder" branch does NOT apply to `dreamer_srl`.** `src/algorithms/dreamer_srl/agent.py` defines `MLPDecoder` (latent → reconstructed 27-dim obs) and `WorldModel.decoder`. So we are squarely in the **decoder-available** regime: we can render imagined observations in the *same* obs space the agent actually senses. The feasibility question professor-rl is checking resolves favorably for the richest visualization family. (I still mark below which designs survive if, for some checkpoint, the decoder is untrustworthy — they degrade to the reward/continuation/latent-only designs.)

**(F2) The renderer ALREADY does imagined-vs-actual overlay.** `src/environment/renderer.py::render_jax_state` consumes a `sensory_data` list in which every modality tile carries both an observed vector and a `true_vector` / `true_intensity`. It draws the "true" layer ghosted (low alpha) and the "observed" layer solid, per-modality, across all seven channel types (intensity bars, olfaction spectrum, collision diamond, visual categorical grid, proprioception radial). This overlay was built for "perception under noise vs. reality" but is structurally **exactly** the imagined-vs-actual comparison we need.

**(F3) `build_sensory_viz(obs, state, params, true_obs=None)` is the integration seam.** `src/environment/sensor.py::build_sensory_viz` is the function that splits a flat 27-vector into the per-modality tiles the renderer wants, attaching the correct labels (`['GRS','SND','PLN','FOD','DNG','PRD','NEU','RCK']` for visual/olfaction, etc.). It already accepts a separate `true_obs`. **The dream-viz hijack is one line of intent:** call it with `obs = decoded_dream_obs` and `true_obs = real_observation`. The ghosted layer becomes "what actually happened", the solid layer becomes "what the dream predicted".

**(F4) The cost center is the Dreamer eval/rollout harness, not the rendering.** `scripts/eval_rollout.py` raises `NotImplementedError` for `agent_type == "dreamer"` — the whole offline rollout + `.rec.gz` recording path is rPPO-only today. So *any* dream-viz needs new Dreamer glue: load a `dreamer_srl` checkpoint, run a real episode while at chosen steps calling `WorldModel.imagine(init_latent, actor, horizon, key)` and decoding each imagined latent. That glue is the real build cost and is shared across every design below; the visualization layer rides on top of it cheaply.

---

## 2. The design space, structured along three axes

### Axis A — WHAT to compare (the fidelity signal)

| Signal | Source in code | What it tells you | Needs decoder? |
|---|---|---|---|
| **Per-modality decoded obs** | `decoder(imagined_latent)` → 27-vec, split by `build_sensory_viz` | Channel-resolved fidelity: does the dream get smell right but hallucinate threat? | Yes |
| **Reward prediction** | `WorldModel` reward head on imagined latents | Does the dream foresee the value/danger of the path? | No (reward head only) |
| **Continuation prediction** | `WorldModel` continue head | Does the dream foresee death/termination? | No |
| **Latent divergence** | imagined prior latent vs. posterior latent from the *real* next obs | Pure model-drift, modality-agnostic | No (latent-only) |

The first row is the rich, legible one (and is available — F1). The bottom three survive even if a decoder is distrusted, and professor-rl may argue latent-KL or reward-error is the *semantically* correct fidelity metric. Design C below is the decoder-free fallback.

### Axis B — HOW to encode each modality

This follows the modality taxonomy the codebase already commits to (`build_sensory_viz` tile types). No new encodings needed — reuse the renderer's vocabulary:

| Modality (dims) | Native encoding (already in renderer) | Imagined-vs-actual reading |
|---|---|---|
| Satiation, Intero-Noc, Extero-Noc (1 each) | dual capsule bar (solid=dream, ghost=real) | over/under-prediction of pain & hunger |
| Reward, continuation (1 each, NEW tiles) | scalar line-overlay over horizon | dream's foresight of value/death |
| Olfaction (5) | spectrum bars | chemical-gradient drift |
| Collision (5) | Manhattan diamond heatmap | does dream hallucinate walls? |
| Visual (8-channel, 1 cell here) | categorical bar grid w/ class labels | object-class hallucination (the hypervigilance-relevant one: does the dream see PRD/DNG that isn't there?) |
| Proprioception (6) | radial / categorical strip | imagined action vs. taken action |

### Axis C — LAYOUT (how panels compose over the imagination horizon)

1. **Per-modality time-ribbon** — one horizontal strip per modality, time on x, H imagined steps; dream solid / real ghosted in each cell. The literal "film strip", decomposed by sense.
2. **Synthesized egocentric scene** — collapse the decoded 8-channel visual + 5-cell collision into a single reconstructed "what the dream thinks the local grid looks like", rendered through the existing minimap/local-view machinery. Most intuitive, least faithful (throws away scalar channels).
3. **Latent-divergence plot** — a single curve of model-drift vs. horizon (and/or per-step reward error). Decoder-free, aggregate-friendly, least "dreamy".
4. **Scalar prediction dashboard** — line overlays for the scalar channels (satiation, both nociceptions, reward, continuation): dream trajectory vs. realized trajectory over H. The pain-and-survival-focused view.

---

## 3. Candidate designs

### Design A — "Sensorium ribbon" (per-modality time-strip, decoder-driven)

The full multimodal film strip. For a chosen step `t`, imagine `H` steps; decode each imagined latent to a 27-vec; render a column per imagined step, stacked into per-modality ribbons, dream solid over real ghosted.

```
            t+1   t+2   t+3   t+4   t+5   t+6        (imagination horizon →)
SATIATION   ▓░    ▓░    ▓▒    ▓▒    ▓▒    ▒▓     dual-bar: solid dream / ghost real
INTERO-NOC  ░     ░     ▒░    ▓▒    ▓▒    ▓░     ← dream predicts pain spike at t+3
EXTERO-NOC  ░     ░     ░     ▓░    ░     ░
OLFACTION   ▁▃▂   ▁▄▂   ▂▅▃   ▃▆▄   ▄▆▅   ▅▆▆    5-bar spectrum, dream vs ghost
COLLISION   ◇     ◇     ◇▪    ◇▪    ◇▪▪   ◇▪▪    Manhattan diamond: hallucinated wall?
VISUAL      |FOD| |FOD| |DNG| |PRD| |PRD| |PRD|  8-class bars — dream "sees" PRD by t+4
PROPRIO     →     →     ↓     ↓     ←     ⊝       imagined action sequence
REWARD      ▔▔▁▁▁▁  (line overlay: dream solid, realized ghost)
CONTINUE    ▔▔▔▔▂▂  (dream foresees termination at t+5)
─────────────────────────────────────────────
REAL (ghost row underlay shows what actually happened at each t+k)
```

- **Modalities → where:** all seven sensor channels each get a ribbon; reward + continuation get scalar line-overlay rows at the bottom.
- **Reuse:** `build_sensory_viz` (F3) per imagined step; renderer tile drawers (F2). New: reward/continue line rows (small matplotlib add), the ribbon stacking loop, and the Dreamer rollout harness (F4).
- **Tradeoffs:** **+** maximally faithful and channel-resolved — directly shows the hypervigilance signature ("dream hallucinates PRD/DNG after injury"). **+** reuses the most existing code. **−** dense; a reader needs a legend. **−** highest per-frame compute (decode × H × render). Build cost: medium (harness) + low (viz).
- **Survives without decoder?** No — collapses to the reward/continue rows only (→ Design C).

### Design B — "Dream-scene reconstruction" (synthesized egocentric view)

Render the decoded visual (8-channel) + collision (5-cell) of each imagined step back into a single reconstructed local grid, played as a short clip beside the real local-view clip. "Here's the room the dream thinks it's in."

```
   DREAM (imagined)            REALITY (actual)
  ┌───────────┐              ┌───────────┐
  │   . D .   │              │   . . .   │   D = dream hallucinates predator
  │ . [A]F.   │   vs.        │ . [A]F.   │   one cell NE; reality has none
  │   . . R   │              │   . R .   │   rabbit drifts one cell
  └───────────┘              └───────────┘
        t+3 of 6                t+3 of 6
   (scalar channels shown as side-gauges: NOC▓ SAT▒ REW▔)
```

- **Modalities → where:** visual + collision → reconstructed grid; scalars (satiation, both nociceptions, reward) → side gauges so they aren't lost.
- **Reuse:** the local-view / minimap drawing in `render_jax_state` is grid-centric — but it currently draws from *true env state*, not from a *decoded obs*. Reconstructing a scene from the 8-channel-per-cell decoded vector is **new** inverse-rendering glue (decoded class probabilities → icon choices). Moderate new code.
- **Tradeoffs:** **+** by far the most intuitive for a non-technical reader / a figure in the paper. **+** makes "spatial hallucination" legible instantly. **−** least faithful: discards the scalar interoceptive channels into side-gauges, and the single-cell visual sensor (the agent only sees its *own* cell's class, range-0) means the "reconstructed scene" is thin unless visual range > 0. **−** decoded class one-hots are soft; need a threshold/argmax policy (a small decision). Build cost: medium-high.
- **Survives without decoder?** No — this design *is* the decoder output.

### Design C — "Divergence dashboard" (decoder-free, scalar + latent)

The robust fallback / aggregate view. Plot, vs. imagination horizon: (i) per-step latent divergence (imagined prior vs. real-obs posterior), (ii) reward-prediction error, (iii) continuation-prediction error, plus dream-vs-real line overlays for the scalar sensors (satiation, both nociceptions) which need no decoder beyond their own 1-D slots.

```
divergence │      ╭─ latent KL (model drift)
   ↑       │   ╭──╯
           │╭──╯           reward err ····╮
           ┼──────────────────────────────····──→  horizon t+1..t+H
  NOC: dream ▔▔╲▁▁  realized ▔▔▔╲▁   (overlay)
  SAT: dream ▔▔▔▔▂  realized ▔▔▔▔▂
```

- **Reuse:** existing `visualization.learning_curves` plotting style; reward/continue/latent are direct `WorldModel` outputs. No renderer changes.
- **Tradeoffs:** **+** survives a distrusted decoder; **+** the only design that aggregates cleanly across many onsets (mean divergence curve ± band) — good for a quantitative figure. **+** lowest build cost on the viz side. **−** least "dreamy"/evocative; doesn't show *what* the dream hallucinated, only *that* it diverged. Build cost: low (viz) + medium (harness).
- **Survives without decoder?** Yes — this is the no-decoder design.

### Design D (recommended first cut) — "Ribbon + scalar dashboard, single-episode microscope"

Design A's per-modality ribbon for the **discrete/spatial** channels (olfaction, collision, visual, proprioception) **fused with** Design C's scalar line-overlays for the **interoceptive + reward + continuation** channels, on a single chosen step of a single recorded episode — a microscope, not an aggregate. This is the smallest thing that shows both *that* and *what* the dream got wrong, reuses the most code, and degrades gracefully (drop the ribbon, keep the dashboard, if the decoder is distrusted).

---

## 4. Recommended first cut + the decisions the user must make

**Recommendation: build Design D.** Rationale: it sits on the existing `build_sensory_viz` + renderer overlay seam (cheapest path to a rich result), it carries both the channel-resolved hallucination story (paper-figure value, hypervigilance-relevant) and the robust scalar/divergence story (survives decoder doubt), and it scopes the new work to the one thing that must be built anyway — the Dreamer rollout+imagine harness (F4). Start single-episode (microscope); add aggregate-across-onsets (Design C's mean-divergence band) as a fast follow once the harness exists.

These are the open decisions — they map cleanly onto AskUserQuestion items for when you synthesize with professor-rl:

1. **Open-loop vs. teacher-forced imagination.** Pure open-loop (policy supplies all imagined actions, no real obs ever re-fed) is the canonical "dream" and the honest divergence test. Teacher-forced (re-feed the *real* action sequence the agent actually took, so only world-model dynamics are tested, not policy drift) isolates model error from policy error. *Defer the semantics rationale to professor-rl; the user picks based on what story the figure should tell.*

2. **Horizon H.** How many steps forward to imagine (e.g. 6 / 10 / 15). Longer = more drift visible but less faithful; tied to the agent's training imagination horizon (check the agent config's `horizon`).

3. **Per-modality ribbons vs. synthesized scene (Design A/D vs. B).** Microscope-for-researchers (ribbon, faithful, dense) vs. figure-for-readers (reconstructed scene, intuitive, lossy). Could build both later; pick the first.

4. **Single-episode microscope vs. aggregate divergence.** One vivid recorded episode at chosen onset steps, vs. mean divergence/hallucination-rate across all threat-onsets (reuses `eval_rollout.py`'s existing `threat_onsets` index). Microscope first is my recommendation; aggregate is the quantitative follow-up.

5. **Which steps to dream from.** Every step (expensive), fixed cadence (every k), or **threat-onset-triggered** (dream from the rising edge of `dist < cue_radius` — reuses the existing `_detect_threat_onsets`). Onset-triggered is the hypervigilance-relevant choice and reuses existing infra.

6. **Decoder-trust gate (coordinate with professor-rl).** If professor-rl's feasibility check finds the decoder reconstruction is poor on this checkpoint, the user should pre-decide the fallback is Design C (divergence dashboard) rather than abandoning the effort.

---

## 5. Hand-off

- **Visualization design + decisions:** this memo. Synthesize with the parallel **professor-rl** memo (world-model semantics: open-loop vs teacher-forced rationale, the right fidelity metric, latent-vs-reward divergence) and put decisions 1–6 to the user via `AskUserQuestion`.
- **When a design is chosen → `experiment-designer` / `senior-developer`** own the build. The load-bearing engineering fact to hand them: **the Dreamer branch of `scripts/eval_rollout.py` is unimplemented (`NotImplementedError`)** — a `dreamer_srl` checkpoint loader + `WorldModel.imagine()`+decode rollout is the prerequisite for *any* of these designs. The visualization itself is a thin layer over `build_sensory_viz` (`src/environment/sensor.py`) and `render_jax_state` (`src/environment/renderer.py`), which already support the dream-solid / real-ghost overlay.
- **No `src/` / `configs/` / `scripts/` edits made here** (postdoc scope).

### Code anchors (for the downstream builder, not the entry-point reader)
- Decoder + imagine: `src/algorithms/dreamer_srl/agent.py` — `MLPDecoder` (~L1203), `WorldModel.imagine()` (L1727), `WorldModel.observe()` reconstructed_obs (L1712).
- Render overlay: `src/environment/renderer.py::render_jax_state` (`sensory_data` two-layer draw).
- Integration seam: `src/environment/sensor.py::build_sensory_viz(obs, state, params, true_obs)`.
- Rollout harness gap: `scripts/eval_rollout.py` — `NotImplementedError` for `agent_type == "dreamer"` (~L651); recorder pattern at `_run_episode_with_recording` to mirror.
- Onset triggering: `scripts/eval_rollout.py::_detect_threat_onsets`.
