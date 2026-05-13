# Triage — NMN meta / continual pivot

**Author**: research-postdoc
**Date**: 2026-05-09 15:17 KST
**Mode**: Triage (escalates to professors after frame disambiguation)
**Anchors**: project_plan §4 (four causes of v8 null), Phases 1–4; NEUROMODULATION_ALGORITHM.md Blocks I §4 / III §10 / IV §13–15 / V §22; nmn_comparison summary 2026-05-09 14:21.

---

## 1. Restated question

This week's NMN comparison thread (heterogeneity sweep + temp-clip rerun) closed both
sensory-modulation explanations of the v8 null without flipping the sign: even with a
heterogeneous noise landscape *and* with the temperature head fully unblocked, the
FiLM-modulated agent does not beat the unmodulated baseline at survival. The user is
not abandoning the sensory-modulation question — that remains the headline story
(project_plan §1, four-property fingerprint, G2′). But before pushing further on the
hardest question we have, the user wants a **floor demonstration** that the NMN
architecture is non-broken in regimes where the literature has consistently reported
gains:

> Show that **this** modulator (the §3 architecture, or its current pre-Phase-3 ancestor)
> delivers a measurable behavioural benefit *somewhere*, in our env, in the form
> the literature already says it should — meta-RL fast-adaptation across related tasks,
> or continual-RL resistance to catastrophic interference across task switches.

If the modulator can't show a gain in **either** of those regimes either, the project
needs to seriously revisit whether the v8/Experiment-1/Experiment-2 null is a
sensory-modulation finding or a more fundamental one about the architecture / training
pipeline. If it *can*, we have a clean baseline of "the architecture works" against
which the harder sensory-modulation question can be re-attacked with the §3 upgrades
(C1 EMA, T/P split, opioid head, precision auxiliary).

This is **adjacent** to the project_plan, not a replacement: the meta/continual results
become Phase 0.5 — an **architecture-vitality check** that gates the Phase 3 factorial.

## 2. Frame disambiguation

"Meta-learning" and "continual-learning" are not interchangeable in the NMN literature.
There are at least three distinct meta-RL frames and two distinct continual-RL frames
in our reference set, and they require different env manipulations.

### 2.1 Meta-RL frames

| Frame | What "meta" means here | Canonical paper in our corpus | Test signature |
|---|---|---|---|
| **(a) Few-shot adaptation across a task family** (MAML-style) | Train on a distribution $p(\mathcal{T})$; at test time, given $K$ episodes / steps in a held-out task $\mathcal{T}^* \sim p(\mathcal{T})$, the agent is faster to adapt than a randomly-initialised baseline. | Wang 2024 (Neuromodulated Meta-Learning) | Adaptation curve on $\mathcal{T}^*$; held-out-task survival vs. step-budget. |
| **(b) Context-conditioned meta-RL** (RL² / PEARL-style; what NPN actually does) | The agent uses *recurrent state* (or an inferred latent context) to identify which task-instance it is in, and the modulator's job is to **gate on context** so the same backbone policy can encode mutually-incompatible behaviours. | **Ben-Iwhiwhu 2022** (NPN on Meta-World ML1/ML45 + CT-Graph). NEUROMODULATION_ALGORITHM.md §B.6: NPN ~2× SPN success on ML45. | Within-episode survival on a held-out *instance* of a known task family; or "modulator off" ablation regresses to baseline-like performance. |
| **(c) Doya meta-parameters as the locus of "meta"** | The neuromodulator's role *is* meta-learning — adjusting α, β, γ, or precision online from the agent's own experience-statistics. The NMN doesn't need to *learn faster across tasks*; it needs to *adjust its own hyperparameters* better than a fixed-α/β baseline. | Doya 2002 (§I-§4); Lee 2024 (§G.1, "Doya-DaYu"); Rodriguez-Garcia 2026 (§I.3, entropy-driven gain). | Survival under a controllable shift in **task statistics** (e.g., reward-noise change, predator-density jump) where the optimal α/β changes. |

### 2.2 Continual-RL frames

| Frame | What "continual" means here | Canonical paper in our corpus | Test signature |
|---|---|---|---|
| **(d) Catastrophic-forgetting-resistance across staged tasks** | Train sequentially on $\mathcal{T}_1, \mathcal{T}_2, \ldots$. At end of training, evaluate retained survival on **all earlier tasks**. NMN gain: forgetting curve is shallower than the unmodulated agent's. | Lee 2024 (§G.5, non-stationary 5-arm bandit, hidden context boundary). Ben-Iwhiwhu 2022b "Modulating Masks" (lifelong PPO/IMPALA). | $\Delta\text{survival}_{\mathcal{T}_1}$ between end of stage 1 and end of stage K; smaller is better. |
| **(e) Stability-gap reduction in joint training** (Rodriguez-Garcia 2026) | Sequential exposure causes a transient "stability gap" — a dip in loss/perf on the *current* task immediately after a switch, before recovery. NMN's role: shrink the dip via entropy-driven gain bursts. | Rodriguez-Garcia 2026 (§I.5: gain $g$ flattens the Hessian during the switch; eigenvalues scale $\lambda \to \lambda/g^2$). | Depth and duration of the post-switch survival dip on $\mathcal{T}_2$ (and $\mathcal{T}_3$, …). |

### 2.3 Which frame did I pick — and the open question for the user

The user said *"start from the meta- and continual-learning parts first"* and *"from
our environment settings"*, with a compute budget of ≤ ~10 cells. That eliminates (a)
(needs a task family with sampled instances and a held-out test split — much heavier
infrastructure than 10 cells) and (c) (needs Doya-DaYu rebuild, weeks of engineering;
the §3 architecture is also a Doya-meta agent in spirit but isn't trained that way yet).

**My pick: (b) for meta + (d) for continual.** Justification:

- **(b)** is what the NPN paper actually demonstrates and is the most direct
  "load-bearing for the architecture" claim in our corpus. We have a recurrent
  modulator with branched heads (NMN_ARCHITECTURE_REVIEW §2); the "modulator gates
  context-dependent behaviour" claim is what it is *for*.
- **(d)** maps cleanly onto our **already-implemented** continual-schedule
  infrastructure: `train.py` lines 135–195 + `configs/continual/example_schedule.yaml`
  already accept a list of stage configs and episode boundaries, switch envs at the
  boundaries, and keep the agent's parameters across the boundary.

Frame (e) (stability-gap) is a **near-zero-cost addendum** to (d) — same runs, just an
extra metric on the post-switch transient. I include it as a free-rider in candidate D.

**Ambiguity to flag for the user:**

1. **Is the user open to a "meta" frame that does *not* involve a held-out test
   split** (i.e., we measure within-training context-conditioning gains, not
   out-of-distribution generalisation)? My picks above answer "yes". If the user
   wants strict held-out-task evaluation, the budget jumps and frame (a) needs a
   parallel triage.
2. **Phase 0 architecture upgrades** — the §3 T/P split + opioid head + precision
   head are not yet implemented (project_plan §5 Phase 0 punch list). Do we run the
   meta/continual probe on the **current** FiLM-only modulator (cheap, fast, but
   leaves Phase 3 architecture untested) or **block** on the Phase 0 engineering
   first (clean test of the headline architecture, but adds 2-3 weeks before any
   data)? My recommendation in §6 is to run the probe on the current architecture
   first — if even a richer-precondition test fails on the simpler modulator, it
   tells us about the modulator family generically; if it succeeds, we have a
   pre-Phase-3 floor we can build on.

## 3. Literature anchors

Per-paper one-line claims relevant to each frame; pulled from our internal review
([NEUROMODULATION_ALGORITHM.md](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)).
"BIB" indicates the paper is unprocessed in `references/neuromodulatory_algorithms/sources/`
(no master review yet — possible literature-reviewer hand-off).

| # | Paper | Internal review section | One-line claim | Frame |
|---|---|---|---|---|
| 1 | **Doya 2002** — Meta-learning and neuromodulation | Block I §4; Paper Review E (§E.1–E.4) | Maps four neuromodulators to four RL hyperparameters (α, β, γ, δ) and proposes that meta-learning *is* online adjustment of these from experience statistics. | (c) — backbone for any meta-as-hyperparam-adjustment frame |
| 2 | **Vecoven 2020** — NMN for adaptive behaviours | Block III §10; Paper Review A | Separate modulator network outputs gain $z$; sReLU with $z$-modulated slope; demonstrated cross-task adaptation in MuJoCo. The architectural template our project inherits. | (b), (d) |
| 3 | **Ben-Iwhiwhu 2022a** — Context Meta-RL via Neuromodulation (NPN) | Paper Review B (§B.1–B.7) | Intra-layer feedforward modulator gates pre-activations via $\tanh \cdot W_g \cdot x$; on **Meta-World ML45**, NPN achieves ~2× SPN success rate; CKA shows NPNs build *task-specific subnetworks*. | (b) — the canonical meta-RL-via-NMN result |
| 4 | **Ben-Iwhiwhu 2022b** — Modulating Masks (TMLR) | Block III §B.7 (note: full text not in NotebookLM) | Frozen backbone; per-task learnable masks composed linearly across tasks; PPO/IMPALA lifelong RL. | (d) |
| 5 | **Lee 2024** — Lifelong RL via Neuromodulation (Doya-DaYu) | Block IV §15; Paper Review G | Ensemble-derived ε/α uncertainties drive ACh-α and NA-β online; on a non-stationary 5-arm bandit with hidden context boundaries, the agent auto-resets exploration on context switches without explicit task IDs. | (c), (d) |
| 6 | **Rodriguez-Garcia 2026** — Noradrenergic Gain-Modulated SGD | Block V §22; Paper Review I | Entropy-driven leaky-integrator gain; in **joint training** across sequential tasks, the gain spike at task boundary flattens the Hessian (eigenvalues scale $1/g^2$), shrinking the post-switch performance dip. | (d), (e) |
| 7 | **Tsuda 2021** — Hypertube shifting (BIB processed in §IV §13) | Block IV §13; Paper Review K | A *single* fixed weight matrix can store mutually-exclusive behaviours in distinct "hypertubes" indexed by a global modulator scalar $f$. | (b) — explains *why* (b) is mechanistically possible in NMNs |
| 8 | **Wainstein 2025** — Phasic gain bursts | Block IV §14 | Phasic NA bursts liquify attractor stability ($\lambda_{\text{barrier}} \propto 1/g$); empirical perceptual-switch evidence. | (b), (e) — phasic mechanism for context detection |
| 9 | **Wang 2024** — Neuromodulated Meta-Learning (BIB) | (Not in current internal review; PDF only) | Neuromodulator-driven inner-loop; meta-train on a task family. | (a) — would be primary anchor if user picks frame (a) |
| 10 | **Durstewitz 2025** — What neuroscience can tell AI about continuously changing environments | Block I §5 | BTSP / one-shot eligibility traces for non-stationary environments; multiscale plasticity. | (d), (e) — biological grounding for "fast slot" + "slow slot" decomposition |

The cross-cutting structural claim from these papers — and the load-bearing one for
*both* the meta and continual frame: **a recurrent modulator with branched, gain-style
outputs can build context-specific subnetworks within a fixed backbone, allowing
storage of behaviours that would otherwise interfere.** This is exactly the function
the project's NMN was designed for; we have not yet shown it does this.

## 4. Candidate experiment designs

Four candidates, ordered by "cheapness of clean YES/NO". A and B are config-only
(launchable this week); C requires light env-side work; D is config-only on the
already-existing continual schedule. All are **survival-step** comparisons, with
≥ 3 seeds per cell (Experiment 2 surfaced ±4-5 step seed noise — single-seed is unsafe).

### Candidate A — Two-context FiLM probe with hidden context (Frame b)

**Regime**: Meta-RL (frame b — context-conditioned).

**Manipulation**: Train a single agent on a 50/50 mixture of two env configs that
share the obs space but require *opposite* policies. Two natural mixes from the
existing config family:

- **A1 — Predator-mode mix:** `01-interoNocicept_sameProp.yaml` (active hunting
  predator) ∪ `02-sameProp_R2_passivePredator.yaml` (predator stays in patrol —
  kinematically rabbit-like). Both already have matched olfactory properties, so
  the only available context cue is the predator's *movement signature* in
  recurrent state.
- **A2 — Resource-layout mix:** swap food spawn quadrant on episode reset
  (TL+BR vs. TR+BL). Same agent, alternating layouts.

The episode-id which mix is used is **never given to the agent** — it must infer
from observations. The unmodulated baseline shares the obs space and recurrent
state but has no modulator to gate on.

**DV**: (i) **Joint mixture survival** — modulated vs. unmodulated, on the
50/50 mix. (ii) **Modulator-state-clamp ablation** — at eval, freeze the
modulator hidden state to its average value across context A → measure survival
on context-B episodes. The drop is the *causal* contribution of context-conditioning.

**Compute**: 2 mixes × 2 architectures (mod / unmod) × 3 seeds = 12 cells. ~13h/cell on
the lab nodes → roughly one overnight on 4–6 nodes. Within budget.

**Expected positive signature**: modulated agent matches or beats the per-context-trained
specialist on the harder of the two, and the clamp-ablation drops survival by
≥ 1 SD on the mismatched context.

**Expected null signature**: modulated ≈ unmodulated on joint survival, *and*
clamp-ablation has no effect. This null would be the strongest possible: the
modulator does not even build context-dependent state when the env literally
requires it.

**Cleanness**: Highest. A1 in particular has the property that the *only* available
discriminator is the recurrent signature → if the modulator does anything, it
should show up here.

### Candidate B — Doya-style task-statistics shift within episode (Frame c, lite)

**Regime**: Meta-RL (frame c — meta-parameter adjustment).

**Manipulation**: A single env where, mid-episode, the **predator damage range** or
**reward setpoint** quietly shifts. The optimal policy needs different temperature β
before vs. after the shift. The shift is hidden from the agent.

**DV**: Time-locked survival around the shift — does the *modulated* agent's policy
re-organise faster than baseline? Specifically: temperature trajectory pre/post shift,
and survival in the 50-step window after the shift.

**Compute**: 2 shift types × 2 architectures × 3 seeds = 12 cells.

**Expected positive signature**: modulator's temperature head shifts within ~5–20
steps post-shift; baseline takes longer or never re-organises within the episode.

**Expected null signature**: modulator's temperature head is flat or changes at the
same rate as baseline's overall policy entropy.

**Cleanness**: Medium. A *positive* result here is a strong meta-c claim, but a
null is hard to interpret — the GRU in the unmodulated baseline is itself
recurrent, so it can also adapt within-episode without the modulator. Useful only
as a follow-up to a positive Candidate A.

**Note on env-side work**: requires either a new env config that supports a
hidden mid-episode parameter change, or a small env code change to re-resolve
predator/reward params at a fixed step. The existing continual schedule is
*episode*-level, not *within-episode*. This is **not config-only** — flag it as a
~few-day env-engineer touch by `developer`.

### Candidate C — Held-out env axis (Frame a, scoped down)

**Regime**: Meta-RL (frame a — generalisation).

**Manipulation**: Train on hypervigilance preset variants where one
hyperparameter (e.g., `predator.detection_range` ∈ {3, 4, 5, 7}) is sampled per
episode from a training set; evaluate on a held-out value (e.g., `detection_range = 6`).

**DV**: Survival on the held-out value, modulated vs. unmodulated. Both should drop
relative to in-distribution; the *gap* is what matters.

**Compute**: ~6–8 env presets × 2 architectures × 3 seeds = ~48 cells. **Over
budget** at this stage; flag as a Phase 1.5 candidate, not a "this week" candidate.

**Expected positive signature**: modulated holds onto more survival on the held-out
value than baseline; CKA representation similarity higher between train and held-out.

**Expected null signature**: equal survival drop. This says the architecture isn't
extracting a generalisable context representation.

**Cleanness**: Highest theoretical force, lowest tractability now. Recommend deferring.

### Candidate D — Continual schedule with hidden boundary (Frames d + e)

**Regime**: Continual-RL — catastrophic-forgetting + stability-gap.

**Manipulation**: Use the **existing** `train.py` continual mode
(`--configs-dir` + `--continual-schedule`) to run a 3-stage schedule with hidden
boundaries:

| Stage | Env (existing) | Episode boundary |
|---|---|---|
| 1 | `01-interoNocicept_noise.yaml` (canonical, active predator) | 1500 ep |
| 2 | `02-sameProp_R2_passivePredator.yaml` (passive predator → predator-as-rabbit) | 3000 ep |
| 3 | back to stage 1 (catastrophic-forgetting probe) | 4000 ep |

**DVs**:
- **(d)** Catastrophic forgetting: at end of stage 3, what fraction of stage-1
  survival is recovered, modulated vs. unmodulated? Lee 2024 §G.5 predicts
  modulator agents recover faster.
- **(e)** Stability gap (Rodriguez-Garcia 2026 §I.5): depth and duration of the
  survival dip in the first 200 episodes of stage 2 and stage 3.
- (free-rider) **Modulator-state trace at the boundary** — does the
  modulator's hidden state actually *change* at the boundary, or does it sit on
  the same attractor? This is direct evidence for "modulator does context-detection".

**Compute**: 2 architectures × 3 seeds = 6 cells, with the 3-stage schedule
totalling ~4000 ep ≈ 1.3× single-stage cost ≈ 17h/cell. Within budget on
4–6 nodes overnight.

**Expected positive signature**: modulated agent's stage-3 stage-1-survival recovery
is ≥ 1 SD above unmodulated (catastrophic-forgetting resistance); stability-gap
depth ≤ 0.5× unmodulated.

**Expected null signature**: same forgetting + same gap. This null + Candidate A
null together would be the project-level equivalent of "the modulator does not
build context-dependent state in any way the literature predicts" — much stronger
than the v8 null and worth re-examining the architecture from scratch.

**Cleanness**: Very high. Reuses already-shipped continual machinery; the only new
artifact is the schedule YAML and a simple analyzer that reads stage-segmented
WandB metrics.

### 4.5 Ranking and recommended sequence

| Cand | Frame | Compute | Cleanness | Config-only | This-week-able |
|---|---|---|---|---|---|
| **D** | d + e (continual) | 6 cells × ~17h | very high | yes | yes |
| **A** | b (meta-context) | 12 cells × ~13h | very high | yes | yes (1 night on 6 nodes) |
| B | c (meta-param) | 12 cells × ~13h | medium | **no** (env code) | not until env touch |
| C | a (held-out) | ~48 cells | very high | yes (config combinatorics) | over budget |

**Recommended sequence** (assuming user accepts §2.3 frame picks):
1. **Candidate D first** (continual + stability-gap) — config-only, reuses
   existing infrastructure, gives **two** signals (forgetting + stability-gap)
   in one set of runs.
2. **Candidate A in parallel** (meta context-conditioning) — orthogonal evidence
   on the same architecture; if both are positive, the architecture is vindicated;
   if both are null, the architecture is in serious trouble and Phase 3 should
   pause before more compute is spent.
3. Candidate B as a follow-up to A only if A is positive.
4. Candidate C deferred until budget grows.

## 5. Env-capability triage

| Need | Status | Notes |
|---|---|---|
| Sequential env-config switch with shared agent across boundary | **Already implemented**. `train.py` lines 135–195 + `configs/continual/example_schedule.yaml`. | Used previously for Dreamer curriculum studies. The schedule is **episode-indexed**, not step-indexed. |
| Mixture-of-envs (50/50 mix per episode reset) | **Not directly supported** — but achievable via `--configs-dir` with `episode_boundaries: [1, 2, 3, ...]` alternating tiny stages, **OR** via a new env-loader wrapper. Cleaner: a small new flag `--mixture-mode` (a few hours of `developer` work). | Candidate A blocker if we want a *clean* config-only path. Workaround: alternate-episode boundaries in continual mode. |
| Within-episode hidden parameter change | **Not supported**. Requires env-side change to re-resolve predator/reward params at a step boundary. ~2–3 day touch. | Candidate B blocker. |
| Predator behaviour variants (active/passive, fast/slow, etc.) | **Already config-driven** — `hunt_stamina_threshold`, `detection_range`, `attack_delay`, `move_interval`. We already have an active/passive pair. | Supports A1, A2, D directly. |
| Resource layout swap (food quadrant) | Already config-driven via per-resource `count` and `spawn_area`. | Supports A2 directly. |
| Modulator-state freeze probe at eval | **Not implemented** as a first-class eval mode — needs a small `developer` patch to expose modulator hidden-state clamping in `evaluate.py`. ~1 day. Or post-hoc replay from logged `mod_h` traces. | Needed to make the *causal* arm of Candidate A's DV (ii) work. The non-causal joint-survival arm works without it. |
| Per-stage / per-segment WandB metrics for analyzer | Continual mode already logs `stage` boundaries (train.py line 1092). Stage-segmented analysis is straightforward. | Supports D. |

**Net assessment**: Candidate **D is fully config-only and runnable now**. Candidate
**A is config-only with a workaround** (alternate-episode boundaries in continual
mode), but the clean version + the modulator-state-freeze ablation each want a
small developer touch. **B and C have real engineering blockers**.

## 6. Recommendation

### 6.1 Primary recommendation: parallel professor escalation

This is a **two-professor question** — RL/Bayesian-DL and Neuromodulation each own a
distinct piece. The work is naturally **parallel** (the two professors can write their
memos independently; postdoc synthesises after).

- **professor-rl-bayesian-dl** owns the meta-RL / context-conditioning side. Their
  load-bearing texts in our corpus: NPN (Ben-Iwhiwhu 2022a/b), HyperZero, FiLM
  (Perez 2018), Tsuda hypertube. They should derive **what mechanistic property
  the architecture must have** to deliver a (b)-frame win, and **what minimal
  measurement on the existing modulator** would falsify the claim that the
  architecture has that property.
- **professor-neuromodulation** owns the Doya-DaYu / continual / stability-gap
  side. Their load-bearing texts: Doya 2002, Lee 2024, Rodriguez-Garcia 2026,
  Wainstein 2025, Durstewitz 2025. They should derive **why a recurrent
  modulator with branched gain heads should attenuate the stability gap and
  catastrophic forgetting**, and **what biological-plausibility constraints**
  should shape Candidate D's schedule choice.

### 6.2 Optional: literature-reviewer in parallel

The 18 PDFs in `docs/project/references/neuromodulatory_algorithms/sources/` have no
master review (only the in-develop NEUROMODULATION_ALGORITHM.md, which is
*develop*-tree). Spawning `literature-reviewer` against the meta/continual subset
(Doya, Ben-Iwhiwhu, Wang 2024, Lee, Rodriguez-Garcia, Vecoven, Durstewitz — 7 PDFs)
in parallel with the professors would give the project a properly-cited
`docs/project/references/neuromodulatory_algorithms/neuromodulatory_algorithms_lit_review.md`
to anchor the eventual experiment-design doc.

This is a "while we're here" win, not a blocker. **User's call.**

### 6.3 What I am *not* doing

- I am not picking the experiment that gets run. The professors should sign off
  first; the user then locks the choice and routes to `experiment-designer`.
- I am not deriving any free-energy / heteroscedastic-NLL math here — that's
  professor-bayesian-brain territory, and the meta/continual question doesn't need
  it on the critical path. (It will become relevant when we re-attack
  sensory-modulation post-pivot.)
- I am not writing config YAMLs or training plans. After professor sign-off and
  user lock, that's `experiment-designer`'s job.

### 6.4 Ready-to-spawn professor prompts

Both prompts are scoped to ≤ 2 pages, ask for a `directions/` memo (the natural
output type for "should we do this experiment"), and explicitly contrast against an
alternative.

---

#### 6.4.1 Prompt for `professor-rl-bayesian-dl`

> **Topic**: NMN architecture vitality check via context-conditioned meta-RL —
> directions memo.
>
> **Background**: This week's NMN comparison study (heterogeneity sweep + temp-clip
> rerun, summary at `docs/experiments/summaries/20260509_1421_nmn_comparison_study.md`)
> closed the v8 sensory-modulation null without flipping it: the modulated agent
> does not beat the unmodulated baseline at survival. The user is pivoting to first
> demonstrate the architecture works in a regime where the literature reports
> consistent gains, before re-attacking sensory modulation. Triage memo at
> `docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md` recommends
> a context-conditioned meta-RL probe (Ben-Iwhiwhu 2022a-style "frame b") on the
> existing FiLM modulator.
>
> **Your task**: Write a 1–2 page directions memo at
> `docs/project/directions/nmn_meta_context_conditioning.md` that:
>
> 1. **Derives** (sketch-level — no full proof needed) what mechanistic property
>    the project's NMN must have to produce an NPN-style ~2× advantage on a
>    two-context mixture (Candidate A in the triage memo): is it sufficient that
>    the modulator's GRU build context-distinguishable hidden states, or does the
>    branched-head structure (NMN_ARCHITECTURE_REVIEW.md §2.4) impose additional
>    requirements (e.g., spatial grouping needs to align with task-conflicting
>    feature subspaces)?
>
> 2. **Names the falsifying measurement** that, if absent on the trained modulator,
>    rules out the (b)-frame claim. (Candidate: CKA distance between modulator
>    hidden states across the two contexts must exceed CKA distance within
>    contexts by some threshold; or the modulator-state-clamp ablation drop must
>    exceed some threshold.)
>
> 3. **Explicitly contrasts** with an alternative interpretation: maybe the
>    *unmodulated* GRU's recurrent state is already enough for context-conditioning
>    (the recurrent baseline is not feature-poor — it has a 128-dim GRU). Under
>    what conditions would the modulator's contribution be *additive over* the GRU
>    rather than *redundant with* it? Tie this to the architectural-bottleneck
>    argument in NEUROMODULATION_ALGORITHM.md §1.
>
> 4. **Recommends** whether Candidate A in the triage memo (active vs. passive
>    predator mix) or a different 2-context mixture is the cleanest test of (3).
>
> Length: 1–2 pages. Math is sketch-level; full derivations are out of scope.
> Cite paper-review section numbers from
> `docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md` (e.g.,
> §B.6 for NPN ML45 results, §K for Tsuda hypertube). Use existing project notation
> (G1/G2, H1–H5, Injection A/B/C). Hand off downstream named: if your verdict is
> "run Candidate A", name `experiment-designer` for the config; if your verdict
> is "blocker first, run a measurement-only probe on existing v8 runs", name
> `experiment-analyzer` for the measurement.

---

#### 6.4.2 Prompt for `professor-neuromodulation`

> **Topic**: NMN architecture vitality check via continual-RL forgetting +
> stability-gap probe — directions memo.
>
> **Background**: Same as 6.4.1 above. Triage memo at
> `docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md` recommends a
> continual-RL probe (Lee 2024 Doya-DaYu-style + Rodriguez-Garcia 2026 stability-gap)
> on the existing FiLM modulator using the project's already-shipped continual
> schedule infrastructure (Candidate D in triage).
>
> **Your task**: Write a 1–2 page directions memo at
> `docs/project/directions/nmn_continual_lifelong_probe.md` that:
>
> 1. **Derives** (sketch-level) why a recurrent modulator with branched gain
>    heads (the project's current architecture per NMN_ARCHITECTURE_REVIEW.md)
>    *should* attenuate the stability-gap dip seen in joint training
>    (Rodriguez-Garcia 2026 §I.5: gain $g$ flattens the Hessian, eigenvalues
>    scale $\lambda \to \lambda/g^2$). Tie the project's `temperature` head and
>    `z_unimodal` heads to the gain-modulation role.
>
> 2. **Derives** (sketch-level) why an entropy-driven or uncertainty-driven
>    modulator should auto-detect the hidden context boundary in Candidate D's
>    schedule, mapping the Lee 2024 ε/α-uncertainty mechanism (§G.2-G.3) onto our
>    existing modulator inputs (full obs vector incl. interoception). What is
>    the analog of Lee's "ensemble-derived ε/α" in a single-modulator architecture?
>    If our architecture *cannot* compute that signal, that is a useful
>    biological-plausibility critique.
>
> 3. **Names the falsifying measurement** for catastrophic-forgetting resistance:
>    e.g., $\Delta\text{survival}_{\mathcal{T}_1}\!\!\big|_{\text{end stage 3}} -
>    \Delta\text{survival}_{\mathcal{T}_1}\!\!\big|_{\text{end stage 1}}$ contrast
>    between modulated and unmodulated, with a magnitude derived from §G.5.
>
> 4. **Recommends** whether Candidate D's three-stage active→passive→active
>    schedule is the cleanest test, or whether a different staging
>    (e.g., gradual-vs-abrupt boundary; multiple stage-1 returns) is more
>    diagnostic. Tie schedule choice to the slow-α-tone vs. fast-NA-burst
>    biological mapping (Doya 2002 + Sara 2009 (BIB)). Pre-condition: the
>    schedule must run on the existing `--configs-dir`/`--continual-schedule`
>    machinery without engineering work.
>
> 5. **Notes which Phase 0 architecture upgrades** (T/P split per project_plan
>    §3.2; opioid head per §3.3) you would expect to *improve* the result, and
>    whether it is informative to run Candidate D *first* on the current
>    architecture (postdoc's recommendation) or *block* on Phase 0.
>
> Length: 1–2 pages. Math is sketch-level. Cite paper-review section numbers
> from NEUROMODULATION_ALGORITHM.md (§E for Doya, §G for Lee, §I for
> Rodriguez-Garcia, §K for Tsuda). Hand off downstream named: if "run", name
> `experiment-designer`; if "block on Phase 0 first", name `senior-developer` for
> the engineering plan.

---

## 7. Hand-off

- **Memo location**: `docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md`
  (this file).
- **Top-level Claude action**: spawn `professor-rl-bayesian-dl` and
  `professor-neuromodulation` **in parallel** with the prompts in §6.4.1 and
  §6.4.2. Optionally spawn `literature-reviewer` in parallel against the
  meta/continual subset of `references/neuromodulatory_algorithms/sources/`
  (Doya, Ben-Iwhiwhu 2022a, Wang 2024, Lee 2024, Rodriguez-Garcia 2026, Vecoven
  2020, Durstewitz 2025).
- **Postdoc synthesis follow-up**: once the two professor memos exist, the
  postdoc returns to write
  `docs/project/ideas/nmn_meta_continual_synthesis.md` consolidating their
  positions, identifying agreement/disagreement, and recommending the single
  experiment to lock with the user. That synthesis is the final input to
  `experiment-designer`.
