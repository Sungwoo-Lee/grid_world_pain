---
title: "Continual-learning probe follow-ups — what we just found, what we still don't know, and the four probes that would tell us"
status: idea
audience: user, experiment-designer, professor-neuromodulation, professor-rl-bayesian-dl
window: 2026-05-13 → next overnight slot
last_updated: 2026-05-13
---

# Continual-learning probe follow-ups — what we just found, what we still don't know, and the four probes that would tell us

## What we just found

This project trains a small reinforcement-learning agent to survive in a grid world that contains a predator. For the past year, we have been comparing two versions of that agent: a plain version, and a version with an extra "modulator" — a tiny side-network whose only job is to scale and shift the main brain's activity in response to what's happening in the world. Until this week, the two versions performed indistinguishably. This week, on a five-stage schedule in which the predator's behaviour toggles (active hunter → passive wanderer → active hunter → passive wanderer → active hunter), the modulated agent survived roughly twice as long as the plain agent on each of the two *returns* to the active-predator world — a gap about 25× larger than the project's seed-to-seed noise floor. In plain English: **when the world changed back to a setting the agent had seen before, the modulated agent remembered how to cope; the plain agent had largely forgotten.**

The modulator's behaviour during that win — sharp, sign-flipping bursts of "gain" exactly at the moments the world changes, then quiet during each stage — is the closest match in any of our experiments to the way the brain's **noradrenaline system (locus coeruleus)** is known to fire: a phasic "reset" signal at moments of regime change, not a tonic dial during steady state. That is the framing the project should adopt for any external write-up: the modulator is doing something *NE-like*, not *ACh-like*, and the schedule is best described as a **regime / context switch under stable value structure** — the rules of survival never change, only the predator's policy does — not as a cognitive-science "task switch", where reward, action, or value structure would also change.

## What we still don't know

We have one win on one seed on one schedule shape. Before we lean on it, we need to know **whether it replicates**, and **why the modulator wins**. There are now six live explanations, none of which the current evidence can separate:

1. **Reusable subnetwork.** The modulator carves out world-specific "channels" during the first active stage; when the active world returns, those channels light up wholesale rather than being rebuilt.
2. **Regime-change detector.** The modulator detects the moment the world flips and re-tunes a small set of channels at that moment — the win is about *triggering* re-tuning, not about *storing* world-specific state.
3. **Context-conditional gain.** The modulator is doing what it always did — re-weighting sensory inputs — but only an agent with a dedicated modulation pathway has anywhere to *put* context-conditional gain. The plain agent has no such slot.
4. **Gradient routing.** The modulator's parameters update only when its context is active, so they suffer less catastrophic-forgetting damage than the fully-shared baseline weights. The advantage is in *what doesn't get overwritten*, not in any active mechanism.
5. **Smoother loss landscape.** The modulator may simply nudge optimisation into a wider, flatter basin where prior tasks are easier to retain — same observable win, different cause, lives in *what gets optimised* rather than *what gets stored*. (Mirzadeh et al. 2020.)
6. **Loss-of-plasticity rescue.** Long continual training is known to collapse a network's effective rank — modulation may keep the GRU's representational capacity alive so it can *re-learn* fast on re-entry. (Dohare et al. 2024, *Nature*.) Again, same observable win, different cause.
7. **Architectural-confound null.** The win is not about modulation at all. The modulator adds parameters, changes per-layer gradient magnitudes, and may simply have shifted the network into a more stable training regime — any other extra-parameter / extra-pathway tweak would produce the same gain. **This is the explanation every probe must rule out.**

We also cannot currently run a representation-level test (whether the agent's internal state on the two active-predator stages looks "the same") because the modulator's raw hidden vector was never logged. Fixing that logging gap is a prerequisite for every probe below.

## The four probes

### Probe 1 — Does it replicate?

**Plain English.** Run the same comparison again, but with three seeds per arm, and once more on a schedule whose stage *lengths* are randomly stretched/shrunk by up to 30%. If the gap survives both, the win is real.

**Manipulation.** Three seeds × {modulated, plain} on the original five-stage active↔passive schedule, plus one additional pair on a stage-length-perturbed version of the same schedule.

**Pre-registered call.**
- *Win:* modulated beats plain by ≥ 50 survival steps on the second return-to-active stage on every seed *and* on the perturbed schedule.
- *Loss:* any seed reverses sign on the second return-to-active stage, OR the perturbed-schedule gap shrinks to within ±20 steps. Downgrade to "schedule-specific".
- *Inconclusive:* mean win preserved but one seed sits within ±20 steps of plain — add a fourth seed.

**What it kills or supports.** Doesn't isolate any single mechanism — all six positive explanations predict replication — but it rules out the most damaging confound (single-seed luck) and tests the architectural-null indirectly: if the gap survives a stretched schedule, the modulator is responding to *something the world is doing*, not to a memorised episode counter.

### Probe 2 — Does the modulator earn its keep at the *moments* the world changes?

**Plain English.** Stop the modulator from learning, but only in a thin window around each stage-boundary (the few seconds either side of the world flipping). If the modulator's win comes from doing something *active* at boundaries, this kills the win. If the win comes from *what the modulator has stored* from earlier, the win survives.

**Manipulation.** Train the modulated agent normally, but apply a **stop-gradient on the modulator's parameters** (not on its output) inside a narrow peri-boundary window (≈ ±5% of each stage's episode budget around every stage transition). Run this as a 2×2 with **output-freeze** (a separate cell where the modulator's *output* is clamped to its pre-boundary mean across the same window) so the two manipulations are not conflated. Plus an unfrozen modulator (this week's cell) and a plain baseline as reference.

This redesign matters: freezing only the modulator's *output* lets the rest of the network re-fit around the fixed value and produces a half-win by construction regardless of mechanism. The clean intervention is freezing *parameters*, i.e. blocking learning into the modulator. And the right *temporal* shape is transient peri-boundary, not chronic stage-2-onwards — that's the biological analog (acute locus-coeruleus blockade, not chronic ablation) and the analytically clean one.

**Pre-registered call.**
- *Reusable-subnetwork (1) wins:* peri-boundary parameter-freeze barely dents the gap on return-to-active. The channels were already built; boundary-time updates are not load-bearing.
- *Regime-change-detector (2) wins:* peri-boundary parameter-freeze collapses the win to within ±20 steps of plain. The modulator must update *at* the boundary to deliver the gain.
- *Architectural-null (7) wins:* both freeze conditions collapse to plain — the modulator only helps while it is being trained, end of story.
- *Inconclusive:* freeze cells land mid-range with no clean ordering.

**What it kills or supports.** Discriminates "what gets stored" from "what fires at the moment of regime change", and replaces a confounded freeze design with one whose null result would be interpretable.

### Probe 3 — Could you get the same win for free by just *telling* the plain agent which world it's in?

**Plain English.** If we hand the plain agent an extra signal that says "you're in active-predator mode now / you're in passive mode now", does it close the gap on its own? If yes, the whole modulator architecture is an expensive way to deliver one bit of context. If no, the modulator is doing something a side-channel input cannot replace.

**Manipulation.** Two control cells against the modulated agent. The first is a **gated baseline**: the plain agent gets a sigmoid gate on each channel of its main representation, with the gate conditioned on a stage-id input — same multiplicative conditioning shape as the modulator, but no separate side-network. The second is a **conditional batch-norm (CBN) baseline**: stage-id drives a per-channel scale/shift on the batch-norm output. Both are well-known multiplicative-conditioning controls that match the modulator's expressive class.

This redesign matters: a one-hot stage-id channel concatenated to the agent's normal observation, which was the original control, is strictly less expressive than the modulator (additive vs multiplicative) — a null result would be capacity-confounded and therefore uninterpretable. Gated and CBN baselines fix the expressivity match.

**Pre-registered call.**
- *Context-conditional gain (3) confirmed:* both gated and CBN baselines stay ≥ 40 steps below the modulator on return-to-active. The modulator's separate pathway does work that an in-stream gate cannot.
- *"It's just a context tag":* either control recovers to within ±20 steps of the modulator. The architecture is wallpapering over a missing input. **This is the most embarrassing possible negative for the modulator narrative; the probe is worth running specifically to rule it out.**
- *Inconclusive:* controls split — one closes the gap, one doesn't — which itself is informative about *what kind* of conditioning the modulator is doing.

**What it kills or supports.** Asks the symmetric question to Probe 2: what does the *plain* agent need to match the modulator? An expressivity-matched answer is the only way to rule out the "it's just a context tag" reading.

### Probe 4 — Does the modulator's win survive long dormancy?

**Plain English.** Make one of the passive stages much longer than the others — 2–3× the length — and ask whether the modulator's win on the *next* return to the active world grows, holds, or shrinks. The biological NE-consolidation story predicts the win survives long dormancy. A pure gradient-routing story predicts the win decays with dormancy length. A loss-of-plasticity-rescue story predicts decay for a different reason.

**Manipulation.** Same five-stage schedule, but with the second passive stage stretched 2–3×. Compare modulated vs. plain on the third active stage.

**Pre-registered call.**
- *Storage / NE-consolidation (1, partly 3) wins:* gap on return-to-active matches or exceeds the original. The modulator's gain survives the long pause.
- *Gradient-routing (4) wins:* gap shrinks roughly proportionally to dormancy length. Modulator parameters get overwritten by the long passive stage.
- *Plasticity-rescue (6) wins:* gap shrinks too, but the *plain* agent also degrades (its effective rank collapses further) — distinguishable from gradient-routing because the absolute level, not just the gap, drops.

**What it kills or supports.** Separates storage-like mechanisms from optimisation-like mechanisms in a single cheap manipulation that none of the other probes can do, because none of the others vary dormancy duration.

## What we still need to control

Three confounds that need to be measured or designed out before any of these probes can carry weight on their own:

- **Parameter-count differential.** The modulated agent has more parameters than the plain agent. Every probe should sit next to an iso-parameter control — a plain-agent variant with the same parameter count delivered via an unconditioned residual head — or the architectural-null is not actually being tested.
- **Per-group effective learning rate.** FiLM placement changes gradient magnitudes per layer, which can produce an "effective" LR schedule difference under a nominal LR match. Logging gradient-norm per parameter group across stages is a one-line addition.
- **Temperature-head saturation at boundaries.** The modulator's temperature output is clipped to a finite range. Saturation precisely at stage boundaries can *mimic* a regime-change-detector signal without any detection happening. Probe 2's pre-registered calls assume non-saturating boundary dynamics; that needs to be checked before interpretation.

In addition, two cheap *measurements* should be logged on every probe run, because each puts a missing mechanism on the table for the cost of one extra eval pass:

- **Loss-landscape sharpness** (largest Hessian eigenvalue via power iteration) at the end of each stage — tests the smoother-loss-landscape explanation (5).
- **Effective rank of the GRU hidden state** across stages — tests the loss-of-plasticity-rescue explanation (6).

## Recommended sequencing

If only one probe fits this round: **Probe 1**. Cheapest, addresses the most pressing question ("is the result real"), and a failure here invalidates everything downstream.

If two fit: **Probe 1 → Probe 2**. Probe 2 is the cleanest mechanism-discriminator after the redesign and the most novel scientifically — separating "what gets stored" from "what fires at the regime change" is not a distinction the continual-RL literature normally draws.

If three: add **Probe 3** with the gated / CBN controls, because a "the modulator is just a context tag" result would *meaningfully change* what the project can claim.

If four: add **Probe 4** last. It is the cheapest dormancy-vs-storage disambiguation and the one that most clearly separates biological-style consolidation from optimisation-style smoothness.

**Prerequisite for all four:** the modulator's raw hidden vector must be logged, plus per-group gradient norms and a stage-end Hessian-sharpness eval hook. Route via senior-developer before any of these launch.

## Hand-offs

- **`experiment-designer`** turns the chosen probe(s) into pre-registered designs. Specifically: build Probe 2 as a 2×2 (parameter-freeze × output-freeze, both peri-boundary), build Probe 3 with gated and CBN controls (not concat-tag), and include an iso-parameter residual-head cell in every probe.
- **`senior-developer`** lands the modulator-hidden-vector logging, the per-group gradient-norm logging, the one-step temperature-derivative logging, and the Hessian-sharpness eval hook before any of these probes launch.
- **`professor-bayesian-brain`** (optional) — if the project later wants a precision-weighting reading of the modulator, that is the right consultation. The current empirical signature favours an NE-analog framing, not an ACh-analog one.

## Links

- Anchor finding (the win): [`NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md`](../../experiments/active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md) §5.5–§6
- Re-summary across the study window: [`20260513_0321_nmn_comparison_study.md`](../../experiments/summaries/20260513_0321_nmn_comparison_study.md)
- Headline memory insight: [`20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md`](../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md)
- Predicate-malformation lesson (shapes criterion-writing): [`20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric.md`](../../../.claude-memory/memories/nmn_diagnosis/20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric.md)
- Logging-gap blocker (prerequisite for all four probes): [`20260513_0017_mod_h_logging_gap_blocks_cka_precheck.md`](../../../.claude-memory/memories/nmn_diagnosis/20260513_0017_mod_h_logging_gap_blocks_cka_precheck.md)
- Prior synthesis that motivated the continual probe: [`nmn_meta_continual_synthesis_v2.md`](nmn_meta_continual_synthesis_v2.md)

---

*The two professor-feedback sections below are preserved as historical record. The substantive points from both — the NE-analog framing, the "regime switch" vs. "task switch" correction, the parameter-freeze redesign of Probe 2, the gated/CBN redesign of Probe 3, the dormancy probe (now Probe 4), the loss-landscape and loss-of-plasticity candidate mechanisms, and the parameter-count / effective-LR / temperature-saturation confound flags — have been folded into the revised body above. The original sections remain for traceability.*

---

## Feedback from `professor-rl-bayesian-dl`

*Signed: `professor-rl-bayesian-dl` — 2026-05-13. Architectural / inference-scheme rigour pass on the three probes. Appended, not a rewrite.*

**Headline.** The trichotomy (reusable-subnetwork / task-switch detector / gradient routing) is sharper than most continual-RL memos manage, and Probe 2's freeze design is the right discriminator in spirit. But two of the three probes have outcome spaces that **do not disambiguate** the candidates as cleanly as the prose suggests, and one of the most-likely mechanisms (functional plasticity / loss-landscape effect) is **missing from the table entirely**.

**1. Probe well-formedness.**

- **Probe 2 (freeze) is partially confounded.** Freezing the modulator's *output* to the stage-1 mean is **not the same** as freezing its parameters. With a frozen output, the downstream FiLM transform becomes a fixed affine scaling — and SGD on the rest of the network will *adapt to that fixed scaling*, partially recovering the gain via the residual pathway. Result: cell (a) ("full freeze") will look intermediate by construction, regardless of which mechanism is true. Concretely, $\gamma, \beta$ frozen at stage-1 means $h_{\ell+1} = \gamma_{\text{frozen}} \odot h_\ell + \beta_{\text{frozen}}$, and the trunk re-fits around it — this is exactly the "static FiLM" regime [NMN_PERFORMANCE_DIAGNOSIS_v8](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) already showed underperforms by less than people expect. Cleaner manipulation: freeze the modulator's **parameters** (stop-gradient + no Polyak update) while still letting it see the live observation; this isolates *gradient flow into the modulator* from *modulator-output dynamics*. Even cleaner: do both (parameter-frozen, output-frozen) as a 2×2.
- **Probe 3's "context tag" baseline is the wrong control.** A one-hot stage-id channel concatenated to the observation does **not** match the modulator's expressive capacity — FiLM applies a *multiplicative* transformation, while concat-then-MLP applies an additive one through the first layer (the well-known result that concat $\subsetneq$ FiLM in expressivity; Perez et al. 2018; Dumoulin 2018). A null result here would be confounded by capacity, not interpretable as "the modulator is just a context tag". Stronger control: **CBN baseline** (concat-conditional batch-norm) or **gated baseline** (sigmoid gate on each channel, stage-id-conditioned) — both deliver multiplicative conditioning without the separate modulator pathway.

**2. Missing mechanisms.**

- **Loss-landscape / implicit-regularisation account.** Adding the modulator changes the Hessian and gradient noise around the converged solution; a wider basin survives distribution shift better (Mirzadeh et al. 2020, "Linear Mode Connectivity in Continual Learning"). This predicts the **same R2 win as the reusable-subnetwork story** but for a completely different reason — it lives in *what gets optimised*, not *what gets stored*. None of the three probes distinguish this from (1). Discriminator: measure **loss-landscape sharpness** ($\lambda_{\max}$ of the Hessian, or SAM-style perturbed-loss) at the stage-2 endpoint; the implicit-reg account predicts a measurably flatter minimum for the modulator agent.
- **Functional plasticity / loss-of-plasticity rescue.** Dohare et al. 2024 (Nature) showed unmodulated nets *lose plasticity* under continual training; modulation can act as a regulariser that keeps effective rank high. This explains the modulator's **better re-learning** without invoking storage at all. Discriminator: measure effective rank of the GRU hidden state across stages — drops sharply for baseline, stays flat for modulator → plasticity-rescue is the story.

**3. Confounds the postdoc didn't flag.**

- **Parameter-count differential.** Until probe 3 includes an iso-parameter control (modulator-sized residual head, no FiLM coupling), the architectural-confound null (5) is not actually tested.
- **Effective learning rate differential.** FiLM placement changes per-layer gradient magnitudes; the modulator agent may simply have a *different effective LR schedule* under the same nominal LR. Log $\|\nabla\|_2$ per parameter group across stages.
- **Modulator-head saturation at boundaries.** The temperature head clipped at $[0.5, 3.0]$ (or $5.0$) saturates exactly at the boundary transitions where the win lives — saturation can mimic "task-switch detector" behaviour without any actual detection.

**4. Recommendation.** Run Probe 1 as proposed. Replace Probe 2(a) with parameter-freeze (not output-freeze). Replace Probe 3's "one-hot tag" cell with CBN or a gated-baseline cell. Add a fourth cell across all probes: **iso-parameter residual-head control** (kills confound 5 properly). Add Hessian-sharpness logging at stage-2 endpoint as a one-line addition to the eval pass — this is the cheapest way to put the loss-landscape mechanism on the table.

**Next steps:**
- `experiment-designer` — incorporate parameter-freeze, CBN-baseline, iso-parameter control before pre-registration.
- `senior-developer` — the `mod_h` logging request already planned should also include per-group $\|\nabla\|_2$ and a Hessian-eigenvalue eval hook (the latter via Lanczos / power iteration; ~$\mathcal{O}(10)$ extra eval passes, not training-loop cost).
- `professor-neuromodulation` — orthogonal pass on the biological-plausibility framing of reusable-subnetwork vs. plasticity-rescue.

---

## Feedback from `professor-neuromodulation`

*Reviewer: `professor-neuromodulation` — 2026-05-13. Biological-plausibility verdict on the three proposed probes: **partial — Probe 2 maps to a real biological intervention but with the wrong freeze schedule; one literature-motivated probe is missing; the memo's "task-switch" framing should be tightened.***

### 1. Which biological modulator best matches this finding?

The closest single analog is **noradrenaline (NE) from locus coeruleus**, not acetylcholine. The R2 signature — a sharp swing at stage boundaries (≥3σ at 3 of 4), sign-flipping with stage identity, supporting policy preservation across re-entries — is the textbook **phasic-NE / "network reset" pattern** of Aston-Jones & Cohen 2005 and Bouret & Sara 2005, refined by Eldar et al. 2013 (NE-like gain modulation reshapes representations at regime change). NE rises at unexpected uncertainty (a new world appears), gates a brief representational re-binding, then quiets — exactly the temperature dynamics the analyzer reports.

The signatures this finding does **not** match: ACh-as-precision (Yu & Dayan 2005) predicts tonic gain *during* a stage rather than phasic swings *at* boundaries; DA would predict the swings to scale with reward-prediction error rather than with stage identity; 5-HT-as-discount-factor predicts no phasic boundary response at all. Doya 2002's clean mapping puts this signal squarely in NE's column (`exploration / noise gain`), with the temperature head as the cleanest behavioural readout.

This matters for framing: an **NE-analog claim is both more defensible and more novel** than an ACh-analog claim for this project. The memo should adopt the NE framing explicitly before publication. The `professor-rl-bayesian-dl` concern about phasic-vs-tonic timescale lines up here: an NE-analog operates on the seconds-to-minutes timescale, which at training-time is the env-step timescale — consistent with the GRU update frequency.

### 2. Does Probe 2's freeze schedule map to a clean biological intervention?

Probe 2 is implicitly testing **acute LC inactivation / β-adrenergic blockade**, not ACh or DA blockade — those have different continual-learning signatures (DA-blockade impairs *new* policy learning, not re-entry; ACh-blockade impairs *within-stage* sensory weighting). The right biological reference is muscimol / clonidine LC suppression during a regime change (e.g., Tervo et al. 2014; Sales et al. 2019).

**The proposed schedule is the wrong pattern for that analog.** Acute LC blockade is a **transient peri-boundary intervention**, not a chronic freeze from stage 2 onwards. Freezing for the entire stages 2–5 (cell a) conflates two effects — blocked boundary detection AND blocked within-stage modulation — and the `professor-rl-bayesian-dl` feedback above shows it is *also* confounded with downstream trunk re-adaptation. The clean biological analog is **freeze the modulator in a narrow window around each transition** (≈±5% of a stage's episode budget at every stage boundary, modulator free elsewhere). Cell (b) (passive-only freeze) is closer in spirit but freezes during a stage rather than *at* the transitions that are the load-bearing event. Recommend adding a **(d) peri-boundary freeze** cell — this is the cell that most cleanly tests the NE / phasic-reset hypothesis.

### 3. The probe the memo is missing

A **delayed-recall / dormancy probe**: insert a long dormancy stage between active re-entries (e.g., `active → passive → active → passive(long) → active`, with the second passive stage 2–3× the others), and ask whether the modulator's win on the final return-to-active *grows*, *holds*, or *shrinks*. NE-dependent consolidation literature (Sara 2009; Roozendaal & McGaugh 2011) predicts the gain should **survive long dormancy**; a pure gradient-routing account predicts decay proportional to dormancy length; a plasticity-rescue account (cf. `professor-rl-bayesian-dl` §2) predicts decay for a different reason. None of the current probes vary dormancy duration, so this trichotomy is not separable from Probes 1–3.

### 4. Construct-validity flag on "task switch"

The schedule is **not a task switch** in the cognitive-neuroscience sense (Monsell 2003) — value-relevance, the action space, and the reward function are unchanged across stages; only the predator's policy changes. This is a **context / regime switch under stable value structure**, closer to Wilson et al. 2014 (latent-state inference) than to Monsell's task-switching paradigm. The biological prediction (phasic-NE at boundaries, no DA-analog change) follows directly from this distinction. Recommend rewording "task switch" → "regime switch" or "context switch" throughout to avoid an over-claim that reviewers familiar with the task-switching literature will catch.

### Hand-offs from this section

- **`experiment-designer`**: add a peri-boundary-freeze cell (d) to Probe 2; consider a dormancy-duration arm as a fourth probe.
- **`senior-developer`**: when the `mod_h` logging lands, also log a **one-step temperature derivative** — the NE-analog signature is the *transient* at boundaries, which is invisible to the current mean-based summary.
- **`professor-rl-bayesian-dl`**: the NE-analog framing pushes against a tonic-gain reading of FiLM and reinforces the parameter-freeze (not output-freeze) recommendation above.
- **`professor-bayesian-brain`**: if the project wants precision-weighting as a load-bearing claim, ACh would be a better substrate than NE — but the empirical signature here favours NE.
