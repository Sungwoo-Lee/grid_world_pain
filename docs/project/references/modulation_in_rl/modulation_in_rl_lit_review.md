---
title: "FiLM-Style Modulation in Reinforcement Learning — Master Reference Review (10 papers, 2024–2026)"
topic: modulation_in_rl
status: in-progress
created: 2026-08-05
last_updated: 2026-08-05
papers_reviewed: 10
sources_held: 17
related:
  - ../FiLM/film_rl_recent_variants_survey.md
  - ../FiLM/film_in_rl_survey.md
  - ../FiLM/film_modulation_granularity_synthesis.md
  - ../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md
scope: |
  The folder's REVIEWED corpus is **FiLM-style scale-and-shift modulation in
  reinforcement learning and robot policy learning** — mechanisms in which a
  conditioning signal generates a multiplicative gain and an additive offset
  applied to a network's activations or to a normalisation layer's output
  (FiLM proper; conditional / adaptive normalisation, adaLN, AdaLN-Zero; and
  constrained or factorised variants of the same operator). Ten of the
  seventeen PDFs held in ./sources/ implement that operator and receive a full
  review each: section-by-section backbone, Phase 1 / Phase 2 synthesis, and a
  fixed extraction block so the folder is queryable later.

  The folder is named "modulation_in_rl" rather than "FiLM" because the live
  variant lines have outgrown the FiLM name — the same gain-and-offset operator
  now travels under "adaptive layer normalisation" in robot policy learning.
  The name is NOT an invitation to treat hypernetwork weight generation,
  mixture-of-experts, or biophysical neuromodulation as first-class subjects.
  Those are deliberately OUT of scope.

  Six further PDFs are retained in ./sources/ as CONTEXT ONLY and are not
  reviewed: HyperMARL (full weight generation), DyMoDreamer (discrete stochastic
  latents, not affine), MENTOR (mixture-of-experts), SPARC (concatenation),
  pi-0 (mixture-of-experts + attention), Neuro-Vesicles (position paper, no
  experiments). Where one of them bears on a question ABOUT FiLM it appears in
  the one-paragraph "Adjacent evidence" appendix, and nowhere else.
---

# FiLM-Style Modulation in Reinforcement Learning — Master Reference Review

## 1. Plain-English entry point

Some neural networks contain a small side-network whose job is not to answer the
question but to **change how the rest of the network computes** — turn some units'
responsiveness up, others down, make the network more or less decisive. The
standard recipe is called **FiLM**, "feature-wise linear modulation": take a hidden
layer's activity, multiply it by a learned **gain** and add a learned **offset**,
where both numbers are produced by a side-network reading a steering signal such as
a task label, a language instruction, or a clock.

This folder collects and reviews the papers from 2024–2026 that use *exactly that
operator* inside a reinforcement-learning or robot-policy-learning system. It exists
because a web survey of recent machine-learning conferences found that the operator
is alive and well but has **changed its name and moved into robotics**: in
diffusion- and flow-based robot policies the same gain-and-offset is folded into a
normalisation layer and called **adaptive layer normalisation** (adaLN), and it is
almost always initialised to do nothing at all at the start of training. Ten papers
implement the operator and are reviewed in full here.

Collectively they say four things. **First**, the operator is now standard equipment
in robot action heads, and the design conversation has moved on from "should we
modulate?" to "how cheaply, and under what constraints?" — one paper shares a single
gain-and-offset generator across all eighteen layers of a transformer, another
constrains the gains so they cannot break a network's built-in geometric symmetry,
another reuses the gains as scores for deciding which inputs to discard entirely.
**Second**, modulation genuinely pays off in *single-task* settings, which earlier
reviews had doubted: what predicts a gain is not task variety but whether the
steering signal carries structure the network can exploit, on a problem hard enough
for that structure to matter. **Third**, two ways of giving a network the same
information are **not** equivalent — routing a signal through the multiplicative
path beats concatenating it onto the input, by as much as 62 percentage points of
success rate on hard robot tasks. **Fourth**, on the two design choices this project
most wanted precedent for — sharing one gain across a *block* of units, and letting
the modulator read the very same observation it modulates — the reviewed corpus
returns nothing at all for the first, and a split, partly-negative verdict on the
second.

Section 2 defines the shared vocabulary; section 3 is the corpus manifest and the
scope boundary; section 4 is the evidence ledger against five open questions;
section 5 records which claims from the motivating survey full-text reading
**confirmed** and which it **corrected**; sections 6 onward are one per paper. The
appendix at the end carries a short "adjacent evidence" note on the six PDFs held in
the folder but deliberately not reviewed.

---

## 2. Shared vocabulary and taxonomy

The axes are reused verbatim from
[`../FiLM/film_modulation_granularity_synthesis.md` §2](../FiLM/film_modulation_granularity_synthesis.md)
and its sister
[`../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md` §2](../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md),
so results compose across folders.

| Axis | Question | Cells / values |
|---|---|---|
| **Axis 1 — granularity** | how many distinct $(\gamma,\beta)$ values exist per modulated tensor | **(a)** per-unit · **(b)** per-channel · **(c)** grouped · **(d)** layer-scalar · **(e)** network-global |
| **Axis 2 — placement** | which layers are modulated, and how many injection sites exist | one/many, early/late |
| **Axis 3 — parameterisation** | one shared generator + per-site heads, vs. per-site generators, vs. one signal broadcast unchanged | — |
| **Axis 4 — grouping / tying** | explicit grouping of units, low-rank tying, population partitioning | — |
| **Axis 5 — self-conditioning** | is the modulator's input the same information source as the stream it modulates? | yes / no / partial |

**Two caveats carried over.** (i) In a fully-connected layer there are no spatial
positions, so "per-channel" (b) and "per-unit" (a) **coincide**; the cells only
separate on convolutional feature maps or token sequences. (ii) "Layer-wise" in
these papers nearly always means *Axis 3* ("each layer has its own parameter set"),
not *Axis 1d* ("one scalar per layer"); every table below states which sense
applies.

**The one vocabulary item this corpus adds.** *adaLN* (adaptive layer normalisation)
is FiLM applied to the output of a `LayerNorm` whose own affine parameters have been
removed:

$$
\mathrm{adaLN}(h, c) \;=\; \gamma(c)\odot\mathrm{LN}(h) + \beta(c).
$$

*AdaLN-Zero* additionally zero-initialises the final linear layer of the generator —
and usually adds a third, **gate** output $\alpha(c)$, also zero-initialised —
so that at step 0 the modulated block is exactly the identity. This is the single
most common mechanism in the reviewed corpus and is the direct descendant of
conditional batch normalisation. **It is the same operator as FiLM**; the only
differences are that it acts post-normalisation and that the field has settled on
zero-initialising it.

### 2.1 This project, in one paragraph (for the relevance notes)

This project runs a small **recurrent PPO** agent — a GRU with 128 hidden units — in
a **single-task survival gridworld**. Its neuromodulator is one shared GRU of 16
hidden units that reads the **raw observation** and feeds six output heads: a
gain/offset pair for a unimodal encoder stage, a gain/offset pair for a multimodal
hub, a bias into the task GRU's update gate, and a scalar action temperature. Each
head emits $\lceil 128/G \rceil$ values expanded to 128 by repeating across
contiguous blocks (grouping size $G$), plus a per-neuron learned baseline.
Empirically: per-neuron modulation ($G=1$) led early then **crashed** around 34 M
episodes with the temperature railed at its clip ceiling and gain variance
ballooning; single-environment comparisons against an unmodulated baseline came back
neutral-to-negative; the one clear win was a continual-learning probe (forgetting
resistance plus a reusable subnetwork). Every per-paper section ends with an honest
relevance note — **"this paper does not bear on our design" is a valid note** and is
used where true.

---

## 3. Corpus manifest and scope boundary

### 3.1 Reviewed in full — 10 papers implementing scale-and-shift modulation

Priority order is the user's. Venue column reflects **what the PDF itself states** —
see each paper's extraction block for exact wording.

| # | Paper | Venue as printed in PDF | What its $(\gamma,\beta)$ does | Section |
|---|---|---|---|---|
| 1 | Yuan 2024 — *Unpacking the Individual Components of Diffusion Policy* | unstated (arXiv 2412.00084v1) | FiLM on the observation, **ablated against concatenation** | [§6](#6-yuan-2024--unpacking-the-individual-components-of-diffusion-policy) |
| 2 | Reuss et al. 2025 — *FLOWER* | CoRL 2025 | **Global-AdaLN-Zero** — one $(\gamma,\beta)$ weight set shared across all 18 layers | [§7](#7-reuss-et-al-2025--flower-democratizing-generalist-robot-policies-with-efficient-vision-language-action-flow-policies) |
| 3 | Yoon et al. 2026 — *PAPL* | unstated (arXiv 2602.09370v2) | FiLM on a **within-episode phase**; **actor AND critic**; single task | [§8](#8-yoon-et-al-2026--papl-phase-aware-policy-learning-for-skateboard-riding-of-quadruped-robots-via-feature-wise-linear-modulation) |
| 4 | Zhu et al. 2025 — *EquAct* | pending | **iFiLM** — equivariance-constrained $(\gamma,\beta)$ | [§9](#9-zhu-et-al-2025--equact-an-se3-equivariant-multi-task-transformer-for-open-loop-robotic-manipulation) |
| 5 | Li et al. 2025 — *CogVLA* | pending | FiLM output reused as **routing scores** | [§10](#10-li-et-al-2025--cogvla-cognition-aligned-vision-language-action-model-via-instruction-driven-routing--sparsification) |
| 6 | NVIDIA 2025 — *GR00T N1* | pending | adaLN as the standard action-head conditioner | [§11](#11-nvidia-et-al-2025--gr00t-n1-an-open-foundation-model-for-generalist-humanoid-robots) |
| 7 | Marquis & Farhood 2026 — *Hypernetwork-conditioned RL under actuator failures* | pending | **FiLM vs LoRA head-to-head under PPO** | [§12](#12-marquis--farhood-2026--hypernetwork-conditioned-reinforcement-learning-for-robust-control-of-fixed-wing-aircraft-under-actuator-failures) |
| 8 | Kang et al. 2026 — *SplitAdapter* | pending | **factorised two-source FiLM** | [§13](#13-kang-et-al-2026--splitadapter-load-aware-humanoid-loco-manipulation-via-factorized-adaptation) |
| 9 | Guo et al. 2026 — *GEAR* | pending | FiLM + geometric symmetry | [§14](#14-guo-et-al-2026--gear-multi-task-reinforcement-learning-of-drone-aerobatics-by-exploiting-geometric-symmetries) |
| 10 | Guo et al. 2026 — *MoE-ACT* | pending | FiLM on action tokens (**FiLM part only**) | [§15](#15-guo-et-al-2026--moe-act-scaling-multi-task-bimanual-manipulation-with-sparse-language-conditioned-mixture-of-experts-transformers) |

### 3.2 Held in `sources/` but NOT reviewed — 6 papers

These do not implement scale-and-shift modulation and are **context only**. They are
not first-class subjects of this folder and their own contributions are not
reviewed. Where one bears on a question *about FiLM*, it gets one paragraph in
[§17 Adjacent evidence](#17-appendix--adjacent-evidence-from-the-six-unreviewed-pdfs).

| Paper | Mechanism | Why excluded |
|---|---|---|
| Tessera et al. 2024 — *HyperMARL* | full weight generation from an agent embedding | hypernetwork, not affine modulation. **Retained in §17** — its gradient-interference result is the principal counter-evidence on self-conditioning |
| Zhang et al. 2025 — *DyMoDreamer* | 32×32 stochastic categorical latents **concatenated** into an RSSM state | not affine, and not modulation — concatenation |
| Huang et al. 2024 — *MENTOR* | mixture-of-experts backbone | conditional computation, not conditional gain |
| Grooten et al. 2025 — *SPARC* | plain concatenation of context latents | negative control. **Retained in §17** as displacement evidence |
| Black et al. 2024 — *π0* | decoder-only MoE transformer + cross-attention | **Retained in §17** as displacement evidence |
| Li et al. 2025 — *Neuro-Vesicles* | position paper, biophysical neuromodulation | no experiments |

### 3.3 Known gap

**One paper from the survey's high-priority list is deliberately absent**: *Volume
Transmission Implements Context Factorization to Target Online Credit Assignment and
Enable Compositional Generalization* (the **e-nmRNN** paper), NeurIPS 2025 Poster,
OpenReview `S9Y89poypx`. Acquisition is pending a separate decision. It was the
survey's provisional candidate for a grouped-modulation precedent, though on the
narrowed scope it would in any case have been context-only (its modulation is
biophysically derived, not affine). Nothing in this review should be read as having
checked it.

Two further survey entries were not downloaded: *Hyper-GoalNet* (NeurIPS 2025) and
*HyPoGen* (ICLR 2025) — both full-weight hypernetwork papers, hence out of scope
under the narrowed definition anyway.

---

## 4. Evidence ledger — five open questions

Every row cites the paper's own section or table. Verdicts are over the **reviewed
corpus** (the 10 papers of §3.1); where an unreviewed PDF bears on a question it is
marked *(adjacent — [§17](#17-appendix--adjacent-evidence-from-the-six-unreviewed-pdfs))*.

### Q1 — Grouped modulation: does any paper share one $(\gamma,\beta)$ across a *block of units*?

**Verdict: YES — exactly once, and as a by-product of a symmetry constraint rather
than a design choice. A second paper then tests that constraint in RL and finds it
harmful when the symmetry is only approximate.**

| Paper | Finding | Evidence |
|---|---|---|
| **EquAct** ([§9](#9-zhu-et-al-2025--equact-an-se3-equivariant-multi-task-transformer-for-open-loop-robotic-manipulation)) | **Genuine cell (c).** One scalar $\alpha_l$ scales an entire $(2l+1)$-dimensional irreducible-representation block; with $L_{\max}=3$ the block sizes are 1, 3, 5, 7. Additive offset survives only on $l=0$. | §4.3 Eqs. 7–9; Prop. 4.4; Proof A.4 |
| — *why* | **Forced by Schur's lemma**, not chosen. A per-component gain commutes with the Wigner D-matrix only if all its entries are equal. The paper never uses the words *grouped*, *granularity*, *coarse* or *parameter efficiency*; group boundaries are fixed by representation theory and cannot be swept. | §9.3 |
| **GEAR** ([§14](#14-guo-et-al-2026--gear-multi-task-reinforcement-learning-of-drone-aerobatics-by-exploiting-geometric-symmetries)) | **Independent test, and the constraint loses.** Substituting EquAct's iFiLM for ordinary FiLM in an RL policy: mean success **95.46 % vs 98.85 %**, with Roll falling 99.71 → 87.70 and Hover 99.90 → 94.92. *"enforces stronger invariance but sacrifices robustness in roll."* | Table I, EMLP block |
| **FLOWER** ([§7](#7-reuss-et-al-2025--flower-democratizing-generalist-robot-policies-with-efficient-vision-language-action-flow-policies)) | **NOT grouped — verified.** Global-AdaLN-Zero shares one *generator weight set* across the **18 layers** (Axis 3); within any layer modulation stays **per-channel**. Ties *sites*, not *units*. | §3.2, quoted verbatim in §7.2 |
| All others | per-unit or per-channel; no grouping | — |

**Synthesis.** Grouping pays **if and only if the tied units are genuinely
interchangeable under a structure the task respects.** EquAct's blocks are
interchangeable by construction; GEAR's Roll manoeuvre has real asymmetries the
constraint forbids expressing, and it loses 12 points there. **This supersedes the
"unattested anywhere" conclusion of both prior syntheses** — but it does not supply
evidence that grouping *helps*, and our own contiguous-index grouping has no
interchangeability justification at all.

### Q2 — Self-conditioning: does any modulator read the same raw observation as the pathway it modulates?

**Verdict: NO reviewed paper reads the same raw, high-dimensional observation as the
stream it modulates. Every successful one uses a low-dimensional conditioner, and
several separate it from the observation deliberately. The nearest positive
precedent is PAPL; the sharpest negative is adjacent.**

| Paper | Axis-5 status | Evidence |
|---|---|---|
| **PAPL** ([§8](#8-yoon-et-al-2026--papl-phase-aware-policy-learning-for-skateboard-riding-of-quadruped-robots-via-feature-wise-linear-modulation)) | **Partial YES — the closest positive precedent.** $\Phi_t = (\cos\phi_t,\sin\phi_t)$ is the FiLM conditioner **and** a component of the observation tuple $o$ fed directly to both actor and critic. So `No-FiLM` is a pure **routing** comparison (concatenation-only vs concatenation-plus-modulation), and modulation wins. But the conditioner is **2-dimensional and exogenous** — a clock, not an observation. | §IV-B; Table II |
| **Yuan 2024** ([§6](#6-yuan-2024--unpacking-the-individual-components-of-diffusion-policy)) | **NO — but the corpus's strongest routing evidence.** The conditioner *is* the full observation, but the modulated stream (action-noise) has **no other route to it**. FiLM vs observation-as-direct-input, 8 tasks: **PegInsertionSide 80 % vs 44 %**, **Relocate 64 % vs 2 %**, TurnFaucet 59 vs 27, PushChair 60 vs 36, Door 95 vs 79; nil or slightly negative on the 2 easy tasks. **Both survey-quoted figures verified exactly against Table 5.** | Table 5 |
| **Marquis** ([§12](#12-marquis--farhood-2026--hypernetwork-conditioned-reinforcement-learning-for-robust-control-of-fixed-wing-aircraft-under-actuator-failures)) | **NO — separation is explicit and deliberate.** Policy reads 34-D state; the 6-D fault vector goes **only** to the hypernetwork. The MLP baseline concatenates all 40-D. Third routing comparison; conditioned wins **only out of distribution** (rudder flutter worst-case 26.73 m vs the MLP's 159.91 m; in-distribution the routes are near-equivalent). | §V, §V-D, Table I |
| **HyperMARL** *(adjacent — [§17](#17-appendix--adjacent-evidence-from-the-six-unreviewed-pdfs))* | **The direct negative.** `w/o GD` conditions the generator on $[o_t, e_i]$ while the generated policy also reads $o_t$ — structurally our configuration — and **degrades on both environments**. Plus a derivation: the variance decomposition requires the generator Jacobian to be mini-batch-deterministic (A3), which self-conditioning destroys. | §6.1, Fig. 8; §F.2 |
| All others | conditioner and modulated stream are **disjoint information sources** (language→vision, timestep→action noise, task ID→action queries) | — |

**Synthesis.** The field's 2025–26 position is not "never condition on the
observation" but **"keep the conditioning path low-dimensional and its gradient path
separate."** Our modulator violates both: a 16-unit GRU reading the **full raw
observation** and modulating an encoder that reads the same input.

### Q3 — Instability: is gain blow-up attested anywhere?

**Verdict: gain blow-up remains UNATTESTED across all ten reviewed papers. The
instabilities that *are* reported cluster on mixture-of-experts isolation and on
capacity hyperparameters — not on affine gains. One paper supplies a quantitative
diagnostic and a named remedy.**

| Report | Paper | What actually failed |
|---|---|---|
| **NaN losses** | FLOWER App. A.1 | **A mixture-of-experts design** — shared MLP combined with action-specific specialist MLPs — *"more memory intensive and had issues with NaN losses… slower convergence."* The **simpler shared Global-AdaLN was the stable fallback.** **This corrects the survey's framing** (§5). |
| undiagnosed collapse | FLOWER App. A.3 | 4.44 → **0.8** average sequence length in an early variant, later attributed to *"instabilities in an earlier model variant"*; **cause never identified.** |
| **expert collapse** | PAPL §V-B | MoE actor backbone: *"the pushing-phase expert collapsed without producing meaningful actions."* Partly confounded by compute (fewer parallel envs). |
| **mode collapse** | MoE-ACT Eq. 11 | MoE needs three auxiliary losses (entropy, balance, orthogonality) *"to prevent mode collapse"*; the co-located FiLM needs none. |
| **capacity blow-up** | **Marquis §V-B** | **LoRA rank 48: "significant instability", MaxPE > 150 m**, breaking an otherwise monotone 8→16→32→64 trend. *"minor deviations can lead to optimization instability rather than predictable performance gains."* |
| **critic-adaptation instability** | **Marquis §V-A** | LoRA + conditioned critic **nearly doubles error**; diagnosed as *"high-variance gradients or numerical instability during the iterative PPO update."* |
| **modulation alone harms** | GEAR Table I | `FiLM Only` **degrades** Flip 58.11 → **34.08 %**; *"modulation without value disentanglement is insufficient for highly nonlinear dynamics."* |

**The corpus's prophylactics, and one that is unique.** Zero-init (FLOWER, explicitly
*"for stable training"*); identity-centred $(1+\gamma)$ (CogVLA, MoE-ACT, Marquis);
identity-init stated (SplitAdapter); RMSNorm + QK-norm + per-condition norm statistics
(FLOWER §A.4). **Marquis alone additionally *damps*:** $p_{\mathrm{scale}} = 1 + 0.1\,p_h$
confines gains to a narrow band around identity **throughout training**, not merely at
step 0.

**The diagnostic we did not have.** Marquis §V-C estimates each modulator's **Lipschitz
bound** as the product of its weight matrices' spectral norms, and finds it tracks
performance across six configurations (LoRA $L$: 28.38 → 21.89 → 11.16 → **3.73** as
rank rises, monotonically improving; FiLM+HC $L=13.67$ vs a weaker variant's 18.76).
Recommended remedy: **spectral normalisation**. Our $G=1$ crash was characterised by
**ballooning gain variance**, which is precisely what this quantity measures — and it
is computable on checkpoints we already hold.

### Q4 — Single-task gains: does modulation help without a task distribution?

**Verdict: YES — four of the ten reviewed papers show it, and the moderator is not
task diversity but *exploitable conditioner structure plus difficulty or distribution
shift*.**

| Paper | Setting | What drives the gain |
|---|---|---|
| **PAPL** ([§8](#8-yoon-et-al-2026--papl-phase-aware-policy-learning-for-skateboard-riding-of-quadruped-robots-via-feature-wise-linear-modulation)) | one skill (skateboarding), **within-episode gait phase**, no task ID | multi-modality *inside* the task: t-SNE shows PUSHING and CARVING actions/observations in separated clusters (§V-A) |
| **Yuan 2024** ([§6](#6-yuan-2024--unpacking-the-individual-components-of-diffusion-policy)) | 8 tasks trained **independently**, no task embedding anywhere | **task difficulty** — +16 to +62 points on 6 hard tasks, ≈0 on 2 easy ones |
| **Marquis** ([§12](#12-marquis--farhood-2026--hypernetwork-conditioned-reinforcement-learning-for-robust-control-of-fixed-wing-aircraft-under-actuator-failures)) | one path-following task, **within-episode actuator fault** | **distribution shift** — routes near-equivalent in-distribution, MLP diverges catastrophically on unseen flutter faults |
| **SplitAdapter** ([§13](#13-kang-et-al-2026--splitadapter-load-aware-humanoid-loco-manipulation-via-factorized-adaptation)) | one task (box manipulation), **within-episode load context** | **load beyond the training range** — all methods saturate at 2/4 kg; the entire separation is at 6 kg |
| DyMoDreamer *(adjacent)* | 26 Atari games trained separately | information the encoder cannot represent (1-pixel moving objects) |
| FLOWER, EquAct, CogVLA, GR00T, GEAR, MoE-ACT | multi-task only | — |

**Synthesis.** Our single-task survival gridworld is **not** a reason to expect a
modulator to fail. But every single-task success here conditions on a variable
carrying **within-episode structure** (phase, fault, load), and three of the four show
the benefit concentrated **outside the training distribution**. Our null results came
from in-distribution single-environment comparisons; our one clear win was a
continual-learning probe. **Marquis exhibits exactly that pattern inside one
controlled experiment**, which makes it a citable explanation rather than a post-hoc
story — and implies our *evaluation protocol* may need changing before our architecture does.

### Q5 — Actor vs critic: what does each paper modulate, and why?

**Verdict: three reviewed papers condition the critic; all three report it helping or
being essential; the one controlled isolation shows FiLM-on-critic cutting error by
roughly half under PPO — but the sign is mechanism-dependent.**

| Paper | Actor | Critic | Stated reason | Isolated? |
|---|---|---|---|---|
| **Marquis** | FiLM | **FiLM (`+HC`)** | — | **YES.** FiLM+HC vs FiLM on rudder failures: static MPE **5.33 → 2.56**, MaxPE 16.49 → 9.64; flutter MPE 15.72 → 7.69, MaxPE 43.67 → 18.70 — **40–50 % error reduction.** *"modulating the critic's activations is more effective for advantage estimation than treating $\lambda$ as a standard input feature."* **But LoRA+HC nearly doubles error** — the sign flips with the mechanism. |
| **PAPL** | FiLM | **FiLM** | **The reward itself is phase-conditioned** ($R_M = R_{\mathrm{dep}} + R_{\mathrm{ind}}$, where $R_{\mathrm{dep}}$ measures *different quantities* per mode), and critic-value histograms separate by phase (§V-A) — verified **before** the design was committed | No — `No-FiLM` removes both |
| **GEAR** | FiLM | FiLM **+ separate value head per task** | *"task-specific reward signals differ in both structure and sensitivity, making a single shared value head prone to conflating them"* | Partly — `Multi-Head Only` beats `FiLM Only` on 3 of 4 tasks; **`FiLM Only` actively degrades Flip**, i.e. conditioning the actor while leaving the critic to conflate reward structures is *harmful* |
| EquAct | — | — | no actor/critic split (implicit action-value, imitation) | — |
| FLOWER, CogVLA, GR00T, MoE-ACT, SplitAdapter | actor / action head only | not modulated | imitation learning (no critic) or frozen-base adaptation | — |

**Synthesis, and the criterion it yields.** Three independent papers reach the same
diagnosis — *a value function forced to represent structurally different reward or
dynamics regimes with shared parameters* — and treat it three ways (FiLM the critic;
FiLM the critic; separate value heads). **The old corpus concern (Ben-Iwhiwhu et al.
2022's report that modulating the value estimator was unstable) is not reproduced by
any of them**, though none engages with it directly.

**The transferable criterion is PAPL's, and we currently fail it:** modulate the critic
when the value function **genuinely varies with the conditioner**. Our reward is not a
function of the modulator's input. The corresponding diagnostic is cheap and we are not
running it — **plot the distribution of critic values conditioned on the candidate
phase variable; if it does not separate, the critic has nothing to gain.**

---

## 5. Survey claims confirmed vs. corrected

Full-text reading against the abstract-level claims in
[`../FiLM/film_rl_recent_variants_survey.md`](../FiLM/film_rl_recent_variants_survey.md).
**Confirmed** = verified verbatim from the PDF. **Corrected** = the mechanism or figure
differs materially. **Refined** = true but imprecise in a way that changes how it may be cited.

### 5.1 Corrected — five substantive mechanism errors

| # | Survey claim | What the full text shows |
|---|---|---|
| **C1** | **CogVLA**: *"the FiLM output is not only a gain — it drives token aggregation and token pruning… FiLM becomes a discrete-computation controller"* | **Over-read.** FiLM conditions the representation; **separate learned routers make the discrete decisions.** Pruning scores come from $R^j_l = \mathrm{MLP}(Z^j_l)$ (Eq. 14) — a different network reading the *token*, not the FiLM parameters. Cross-encoder fusion uses its own sigmoid MLP (Eq. 11). The module names say "FiLM **based**", and "based" is doing real work. **Modulation-as-routing is not a new primitive here.** ([§10.2](#102-verification-of-the-surveys-claims--one-substantive-correction)) |
| **C2** | **EquAct**: granularity *"(b), restricted to symmetry-invariant (type-0) features"* | **Wrong restriction.** Modulation reaches **every** type $l$ (Eq. 8), not just type-0. What is restricted is the *form*: a **shared scalar gain with no offset** on each $l>0$ block, full affine only on $l=0$. And that shared-scalar form makes it **cell (c) grouped** — the corpus's only instance. ([§9.3](#93-open-question-1--grouped-modulation-found-with-the-caveat-that-decides-how-we-cite-it)) |
| **C3** | **FLOWER**: *"a richer shared-plus-specialist modulation MLP produced NaN losses"* — cited as the only report of a **modulation** design abandoned for instability | **Correct as fact, wrong in kind.** App. A.1's own heading is **"Mixture-of-Experts Approaches for the Flow Transformer"**; the failing design is action-type-specific specialist MLPs plus a shared MLP — **conditional computation, not a conditional gain**. The affine path did not produce the NaN. Correctly stated, the finding is *more* favourable to coarse modulation: **the simpler, coarser modulation was the stable one.** ([§7.2](#72-verification-of-the-surveys-claims)) |
| **C4** | **FLOWER** as *"the strongest published evidence that modulation capacity is over-provisioned by default"* | **Not supportable.** The comparison is not coarse-vs-fine: FLOWER **adds per-layer LoRA adapters** to compensate, and **there is no `− LoRA` row in Table 3, no LoRA line in the parameter table, and no ablation isolating the coarsening.** Per-layer capacity was *relocated*, not removed. Two readings survive and the paper cannot distinguish them. ([§7.3](#73-the-confound-the-survey-could-not-see-lora-is-never-ablated)) |
| **C5** | **Grouped modulation**: *"AlKilany & Goodman 2025 remains the only exact precedent in existence"* / *"unattested in both corpora"* | **Superseded.** EquAct's iFiLM shares one scalar across a $(2l+1)$-unit block. **Both prior syntheses should be annotated.** With the essential qualifier that it is a **by-product of Schur's lemma**, not a designed regulariser — and that GEAR's independent RL test finds the constraint *harmful* where the symmetry is approximate. ([§9.3](#93-open-question-1--grouped-modulation-found-with-the-caveat-that-decides-how-we-cite-it), [§14.3](#143-the-finding-the-survey-could-not-have--a-head-to-head-against-equacts-ifilm)) |

### 5.2 Refined — true, but imprecise in a citation-relevant way

| # | Survey claim | Refinement |
|---|---|---|
| R1 | **FLOWER**: *"−20 % parameters with no performance loss"* | It is 20 % of the **action head**, not the model (§1 and the conclusion say *"head parameters"*). And **the counterfactual parameter count is never published** — Table 6 lists `Global-AdaLN 28.3M` but no standard-AdaLN figure, so the 20 % cannot be reconstructed. Scope of the parity claim: **one benchmark, 3 seeds, ≤100k steps, no pretraining.** |
| R2 | **GR00T N1 / $\pi_0$**: *"attention displaces affine modulation at frontier scale"* | **Both models use both, for different signals.** GR00T routes the scalar timestep through adaLN, vision+language through cross-attention, and proprioception through concatenation. The accurate statement is **"mechanism follows signal structure"**, not "attention won". ([§11.2](#112-verification-of-the-surveys-claim--confirmed-and-the-confirmation-is-thin)) |
| R3 | **CogVLA** conditioners: *"(i) the language instruction; (ii) the action intent"* | (ii) is the **instruction tokens $t_l$ at layer $l$** — a contextualised version of the same signal. "Action intent" is the paper's cognitive-science framing (the SMA analogy), not a distinct input. |
| R4 | **GR00T N1** as the canonical adaLN citation | Confirmed — **and the confirmation is one clause.** `adaptive layer normalization` appears **once** in 36 pages; `FiLM`, `AdaLN`, `modulation`, $\gamma$, $\beta$ appear **nowhere**; no equation, no granularity, no site count, no initialisation, no ablation. Cite it for **the fact of adoption only**; cite Peebles & Xie for the mechanism. |
| R5 | **EquAct** *"ICLR 2026 Poster"*; **PAPL** *"ICRA 2026"* | Neither is stated in the held PDF. EquAct's PDF says **"Preprint. Under review."**; PAPL's has no venue line. The survey's sources (an OpenReview record and the arXiv comments field) are legitimate but **external** — attach that provenance when citing. |
| R6 | **GEAR** *"98.85 % success"* | Not printed as such. It is the **mean of the four per-task rates** (99.90 / 95.80 / 99.71 / 100.00) in Table I — a correct aggregate, reconstructed rather than quoted. |

### 5.3 Confirmed

| Survey claim | Verification |
|---|---|
| **Yuan 2024**: FiLM 80 % vs 44 % (PegInsertionSide), 64 % vs 2 % (Relocate) | **Exact.** All 8 rows of Table 5 reproduced in [§6.2](#62-the-table-5-numbers-verified-verbatim); every cell matches. |
| **Yuan 2024**: *"the modulated stream is the action-noise pathway, which does not read the observation by any other route"* — i.e. not full self-conditioning | **Confirmed**; the caveat survives full-text reading. |
| **FLOWER**: Global-AdaLN-Zero *"ties sites not units"* | **Confirmed verbatim** — *"shares a single set of modulation weights across all layers"* (§3.2). Upholds the survey's own caution. |
| **FLOWER**: CoRL 2025 | **Confirmed** from the PDF's page-1 footer. |
| **CogVLA**: NeurIPS 2025; 97.4 % LIBERO; 2.5× cheaper training, 2.8× lower latency | **Confirmed** — footer; Table 1 / Table 7 ($97.4\pm0.4$ over 3 seeds); §3.3 reports 2.49× and 2.79×. |
| **PAPL**: within-episode cyclic phase, not a task ID; **modulates both actor and critic** | **Confirmed** — §IV-A, Table II. And the paper supplies a principled reason the survey did not have ([§8.4](#84-the-finding-the-survey-missed-2--why-the-critic-is-modulated)). |
| **EquAct**: iFiLM constrains $(\gamma,\beta)$ so modulation commutes with SE(3) transformations; SOTA on 18 RLBench + 4 physical tasks | **Confirmed** — Prop. 4.4, Proof A.4; Tables 1–2. |
| **HyperMARL** *(adjacent)*: coupling the conditioner into the observation stream causes cross-agent gradient interference; policy-gradient variance empirically reduced | **Confirmed** — mechanism in §3.2, metric in App. E.4 (inter-agent gradient cosine similarity), result in Fig. 3b. Variance claim is **directional only** — Fig. 4c is a bar chart and **no numerical value appears in the text**. |
| **Marquis** (U1): only direct FiLM-vs-LoRA head-to-head inside on-policy RL; generalises to time-varying failure modes unseen in training | **Every clause confirmed.** Plus three findings the survey could not see: the actor/critic isolation, the rank-48 instability, and the Lipschitz analysis. |
| **SplitAdapter** (U2): hierarchical FiLM from two factorised context encoders; benchmarked against world-model FiLM baselines | **Confirmed** — §3.5; `AnyAdapter-style WM-FiLM` baseline in Table 1. |
| **GEAR** (U3): equivariant actor + FiLM task modulation + multi-head critic | **Architecture confirmed exactly** (§IV). |
| **MoE-ACT** (U4): FiLM modulates action tokens during decoding while a sparse MoE handles task decomposition | **Confirmed** — Eqs. 5–6 modulate the decoder action queries at every layer; MoE sits in the encoder. |
| **SPARC** (U-tier, adjacent): concatenates rather than modulates | **Confirmed** as the corpus's negative control. |

### 5.4 Net effect on the two prior syntheses

| Doc | What changes |
|---|---|
| [`film_modulation_granularity_synthesis.md`](../FiLM/film_modulation_granularity_synthesis.md) | **§3.3 and §7.3 superseded on cell (c)** — grouped modulation is now attested once (EquAct), by structural necessity. Its Axis-3 verdict ("shared generator + per-site heads is the norm") gains its first counter-instance in FLOWER, **with the LoRA confound attached**. |
| [`film_rl_recent_variants_survey.md`](../FiLM/film_rl_recent_variants_survey.md) | **§5.3 superseded** (grouping now attested). **§5.5's NaN attribution corrected** (C3). **§5.6 strengthened** — single-task gains now rest on four reviewed papers, two of which (Yuan, Marquis) the survey had not classified that way. **§5.4's net verdict stands and sharpens**: keep the conditioning path low-dimensional and its gradient path separate. |
| [`film_in_rl_survey.md`](../FiLM/film_in_rl_survey.md) | **§5's "gains need task diversity" verdict is now clearly refuted** (Q4). **§7's "no failure mode matches ours" survives** — gain blow-up remains unattested across ten more papers (Q3), but Marquis now supplies a *measurement* (Lipschitz bound) and a *remedy* (spectral normalisation) the corpus previously lacked. |
| [`../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md`](../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md) | Its §8.2 conclusion about our modulator's input is further supported: **no reviewed paper conditions on a full raw observation**, and the corpus-wide conditioner dimensionality is 1–32 ([§16.3](#163-conditioner-dimensionality--the-corpuss-most-uniform-regularity)). |

---

## 6. Yuan 2024 — *Unpacking the Individual Components of Diffusion Policy*

**PDF:** `docs/project/references/modulation_in_rl/sources/Yuan 2024 - Unpacking the individual components of diffusion policy (preprint).pdf`
**arXiv:** [2412.00084v1](https://arxiv.org/abs/2412.00084), submitted 27 Nov 2024. Single author: Xiu Yuan (UC San Diego).

### 6.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **Unstated in PDF.** The PDF carries only the arXiv stamp `arXiv:2412.00084v1 [cs.LG] 27 Nov 2024`; no venue line, no acceptance note, no "submitted to" statement. Cite as a 2024 preprint. |
| **Conditioning signal** | The **observation sequence** $[O_t, O_{t-1}, \dots, O_{t-T_o}]$ — low-dimensional proprioceptive/object state, not images (§4.1: "Observation Modalities: low-dimensional state observation"). This is the *entire* observation the policy has; nothing is withheld from it. |
| **Modulated component** | The **denoising network** of Diffusion Policy — a 1-D U-Net acting on the action-noise sequence. In our vocabulary: the **action head**. Not the encoder, not a critic (there is no critic; this is imitation learning). |
| **Granularity cell** | **(b) per-channel**, inherited from Chi et al.'s Conv1D U-Net. **Caveat: the PDF never states this.** Its only evidence is Fig. 6, which draws `Linear → (a, b)` then `a*x + b` applied to the action-noise tensor. The per-channel reading comes from the Diffusion Policy codebase, not from this paper. Recorded as **(b), inferred, not stated**. |
| **Placement** | **Not enumerated in the PDF.** Fig. 6 depicts **two** injection sites (two `Linear` heads, two `a*x+b` boxes) drawn schematically; the underlying Diffusion Policy applies FiLM in every residual block of the U-Net. Recorded as **many sites, inherited, count not stated**. |
| **Parameterization** | **One shared conditioning signal + per-site linear heads** — Fig. 6 shows the single observation vector feeding two separate `Linear` layers, each producing its own $(a,b)$. This is the textbook Axis-3 arrangement. Inferred from the figure; not stated in prose. |
| **Initialization** | **Not stated anywhere in the PDF.** No zero-init, no identity-init, no mention that initialisation matters. |
| **Ablation present?** | **Yes — this is the paper's entire contribution**, and it is the corpus's cleanest modulation ablation. Table 5 pits *FiLM conditioning* against *observation as direct network input*, holding everything else fixed, on 8 tasks. Verbatim numbers below (§6.2). Author's takeaway box: *"FiLM Conditioning significantly enhances the performance of the Diffusion Policy on hard tasks, but is not needed for easy tasks."* |
| **Reported instability** | **None for modulation.** The only stability remark in the paper is about *architecture*: the transformer denoiser of Peebles & Xie "demonstrated stronger performance, albeit with less stable training" (§3.4). No gain blow-up, no NaN, no entropy or plasticity discussion. |
| **Single-task or multi-task** | **Single-task, eight times over.** Each of the 8 tasks is trained and evaluated separately; there is no task distribution and no task embedding anywhere. **The moderator is task *difficulty*, not task diversity** — the paper's own framing. |

### 6.2 The Table 5 numbers, verified verbatim

Reproduced exactly as printed (PDF p. 9). The "Task Difficulty" column is the
paper's own label.

| Task | Task Difficulty | FiLM Conditioning | Direct Inputs | Δ |
|---|---|---|---|---|
| ManiSkill: StackCube | Easy | 99 % | 97 % | +2 |
| ManiSkill: PegInsertionSide | Difficult | **80 %** | 44 % | **+36** |
| ManiSkill: TurnFaucet | Difficult | **59 %** | 27 % | **+32** |
| ManiSkill: PushChair | Difficult | **60 %** | 36 % | **+24** |
| Adroit: Door | Difficult | **95 %** | 79 % | **+16** |
| Adroit: Pen | Easy | 71 % | 75 % | −4 |
| Adroit: Hammer | Difficult | 17 % | 18 % | −1 |
| Adroit: Relocate | Difficult | **64 %** | 2 % | **+62** |

**Verdict on the survey's claim.** The motivating survey quoted "80 % vs 44 %;
64 % vs 2 % on hard tasks". **Both figures are exact and correctly attributed**;
the full 8-row table is reproduced above and every cell matches. **Confirmed, no
correction needed.**

### 6.3 What the ablation actually manipulates — and why it is *not* self-conditioning

This is the load-bearing detail, and it needs stating precisely because it is easy
to over-claim.

Diffusion Policy's denoiser $\epsilon_\theta$ predicts the noise added to an action
chunk. The two arms of Table 5 differ **only in how the observation reaches it**:

- **FiLM arm** — the denoiser's input is the noisy action chunk alone,
  $x = A^k$. The observation enters *only* through the modulation path:
  $\mathrm{FiLM}(x; O) = a(O)\odot x + b(O)$.
- **Direct-input arm** — the observation is concatenated to the denoiser's input,
  $x = [A^k, O]$, and no modulation is applied.

Both arms give the denoiser exactly the same information. **The manipulation is
purely one of routing: multiplicative-and-additive path versus concatenated-input
path.** That is what makes it valuable — it isolates the *mechanism*, not the
information.

But the modulated stream, the action-noise pathway, **has no other route to the
observation in the FiLM arm**. So this is *"observation modulates a stream that
does not otherwise read it"*, i.e. **Axis 5 = no**, not the full self-conditioning
loop this project runs (where the modulator reads the observation and modulates the
encoder that reads the same observation). The survey stated this caveat correctly
and it survives full-text reading.

### 6.4 Quality caveats the survey did not flag

Full-text reading surfaces three problems that a citation of this paper must carry.

1. **No error bars, no seeds, no evaluation-episode count.** Every number in every
   table is a single bare percentage. The paper never states how many seeds were
   run, how many evaluation episodes each percentage is over, or any measure of
   spread. A 2-point difference (StackCube 99 vs 97) and a 62-point difference
   (Relocate 64 vs 2) are presented with identical typographic authority. The
   *large* differences are almost certainly real; the small ones are uninterpretable.
2. **The bibliography is broken in a way that signals low review pressure.**
   Reference [1], cited repeatedly in-text as the ACT paper, is *"Norman Abramson.
   The aloha system: Another alternative for computer communications, 1970"* — a
   1970 networking paper. Reference [?] appears unresolved in §3.4.
3. **Single-author preprint, no code release statement, no hyperparameter appendix.**
   The FiLM-vs-direct-input comparison does not report whether the two arms were
   parameter-matched; concatenating $O$ to the input changes the first layer's
   input width, so the arms differ slightly in parameter count in an unstated
   direction.

None of this overturns the headline — a 62-point gap on Relocate and a 36-point gap
on PegInsertionSide are not seed noise. But the paper should be cited as
**"suggestive single-preprint evidence"**, not as a settled result.

### 6.5 Phase 1 — Foundational overview

**Introduction.** Diffusion Policy is a way of making a robot choose what to do:
instead of outputting one action directly, it starts from random noise shaped like
a short sequence of actions and repeatedly "cleans" it, a bit at a time, until a
sensible action sequence emerges. It works very well, and because it works well,
people have been copying it and quietly changing pieces of it. This paper asks the
obvious unasked question: **which pieces actually matter?**

It names five pieces — feeding a stack of past observations rather than one; acting
out several predicted actions rather than one; predicting more actions than you
execute and throwing the rest away; using a U-shaped convolutional network rather
than a plain stack of layers as the "cleaner"; and finally **using the observation
to *steer* the cleaner rather than feeding it in as input**. It then removes each
piece one at a time across eight robot tasks.

**Key findings.** The fifth piece is the one this folder cares about, and the result
is stark. On genuinely hard tasks, steering the denoiser with the observation
(FiLM) rather than feeding the observation in as input is worth 16 to **62**
percentage points of success rate. On easy tasks it is worth nothing — sometimes
slightly negative. The other four pieces show the same pattern of
"matters-when-it-matters": past observations only help when the robot must output
absolute positions; executing action sequences helps everywhere except when the
task needs fast reflexes (hammering); discarding surplus predicted actions helps
only on long tasks; the U-shaped network only beats a plain one on hard tasks.

**Initial takeaway.** Two ways of giving a network the same information are not
equivalent. Multiplying by it is not the same as concatenating it — and the gap
between them grows with how hard the problem is. For anyone building a conditioned
policy, this is the cheapest possible argument that the conditioning *mechanism*
deserves a design decision of its own.

### 6.6 Phase 2 — Graduate-level deep dive

#### 6.6.1 The conditional denoising formulation

Diffusion Policy models the conditional action distribution $p(A_t \mid O_t)$ where
$A_t = [a_t, a_{t+1}, \dots, a_{t+T_p-1}]$ is a *chunk* of $T_p$ future actions and
$O_t = [o_{t-T_o+1}, \dots, o_t]$ is a window of $T_o$ past observations. Sampling
runs the Denoising Diffusion Probabilistic Model reverse chain, which the paper
writes in its Fig. 2 as an $\times K$ loop over

$$
A^{k-1}_t \;=\; \alpha\Bigl( A^{k}_t \;-\; \gamma_k\,\epsilon_\theta\bigl(O_t,\, A^{k}_t,\, k\bigr) \Bigr) \;+\; \mathcal{N}\bigl(0,\, \sigma^2 I\bigr),
$$

for $k = K, K-1, \dots, 1$, with $\epsilon_\theta$ the learned noise predictor,
$\alpha, \gamma_k, \sigma$ the noise-schedule constants, and $A^K_t$ drawn from
$\mathcal{N}(0, I)$. Training minimises the standard denoising objective

$$
\mathcal{L} \;=\; \mathbb{E}_{(O_t, A_t)\sim\mathcal{D}} \;\mathbb{E}_{k\sim\mathcal{U}\{1..K\}} \;\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)} \Bigl\| \epsilon - \epsilon_\theta\bigl(O_t,\, A_t + \epsilon_k,\, k\bigr) \Bigr\|^2 .
$$

The paper's Fig. 2 also annotates the score interpretation $\nabla E(a)$: the
denoiser is, up to scaling, the gradient of an implicit energy over action chunks,
so $\epsilon_\theta$ is a *learned vector field on action space*. **This is why the
conditioning-route question is not cosmetic** — see §6.6.3.

#### 6.6.2 The two conditioning routes, written out

Let $x_\ell \in \mathbb{R}^{C_\ell \times T_p}$ be the activation of U-Net layer
$\ell$ (channels $\times$ chunk length), and let $e = \phi(O_t) \in \mathbb{R}^{d}$
be the flattened observation embedding (concatenated with the diffusion-step
embedding in the reference implementation).

**FiLM route** (the paper's Fig. 6, generalised over $L$ sites):

$$
\begin{aligned}
\bigl[a^{(\ell)};\, b^{(\ell)}\bigr] &= W^{(\ell)} e + c^{(\ell)}, \qquad W^{(\ell)} \in \mathbb{R}^{2C_\ell \times d},\\[2pt]
\mathrm{FiLM}^{(\ell)}\bigl(x_\ell \mid O_t\bigr) &= a^{(\ell)}(e) \odot x_\ell + b^{(\ell)}(e),
\end{aligned}
$$

with $\odot$ broadcasting each per-channel scalar $a^{(\ell)}_c$ across the whole
temporal extent of channel $c$ — the canonical Axis-1 cell **(b)**. The denoiser's
*input tensor* is $A^k_t$ alone.

**Direct-input route** (the ablation arm):

$$
\epsilon_\theta\bigl(O_t, A^k_t, k\bigr) \;=\; f_\theta\Bigl( \bigl[\,A^k_t \,;\, \mathrm{broadcast}(e)\,\bigr] \Bigr),
$$

i.e. $e$ is tiled along the temporal axis and concatenated to the channel axis of
the input, and every $\mathrm{FiLM}^{(\ell)}$ is replaced by the identity.

#### 6.6.3 Why the routes are not equivalent — a derivation of the difference

The natural objection is that a concatenated input can *simulate* FiLM: the first
layer sees both $x$ and $e$, so surely a sufficiently wide network can learn
$a(e)\odot x$. It can, but only at a cost that this derivation makes explicit.

Consider a single linear layer with concatenated input:

$$
y \;=\; W_x x + W_e e + c .
$$

This is **affine in $x$ with a fixed coefficient matrix $W_x$** — the observation
can only shift the output, never rescale it. The map $x \mapsto y$ has the *same
Jacobian* $\partial y/\partial x = W_x$ for every observation. Contrast FiLM:

$$
y \;=\; W_x\bigl(a(e)\odot x + b(e)\bigr) + c \quad\Longrightarrow\quad \frac{\partial y}{\partial x} \;=\; W_x\,\mathrm{diag}\bigl(a(e)\bigr).
$$

**The Jacobian is now observation-dependent.** Concatenation gives the network a
family of *parallel* affine maps (one intercept per observation, one shared slope);
FiLM gives it a family of *rescaled* affine maps (one slope per observation).

To recover a multiplicative interaction from concatenation you need at least a
second layer plus a nonlinearity to synthesise the product $a(e)x$. Writing the
degree-2 Taylor expansion of a two-layer ReLU/GELU network around the origin, the
cross term $x_i e_j$ appears with coefficient
$\sum_h W^{(2)}_{\cdot h}\,\sigma''(0)\,W^{(1)}_{h i} W^{(1)}_{h j}$ —
i.e. **the product must be *manufactured* out of second-order curvature of shared
hidden units**, whereas FiLM supplies it as a first-class primitive. In a denoiser
whose output is a vector field $\nabla E(a)$ over action space, this matters
concretely: the observation must *reshape the field*, not merely translate it,
because the modes of the action distribution move and change shape with the scene.
A shift-only conditioning path can translate the field's zeros but cannot move
where they are relative to one another.

**Prediction implied by the derivation, and matched by Table 5.** The gap between
routes should scale with **how strongly the target vector field's *shape* depends
on the observation** — small when the demonstration distribution is nearly
unimodal and observation-invariant (StackCube 99 vs 97; Pen 71 vs 75), large when
the observation selects among qualitatively different action modes (Relocate 64 vs
2, where object and target positions are both randomised and the required
trajectory shape changes completely). This is precisely the easy/hard split the
paper reports, and it is a mechanistic explanation the paper itself does not give.

#### 6.6.4 The other four components, compactly

For completeness, and because they interact with the FiLM result:

| Component | Manipulation | Result pattern | Moderator identified |
|---|---|---|---|
| Observation sequence input | $T_o = $ window vs. $T_o = 1$ | −10 pp on Absolute Control (Door 95→83, Relocate 64→47); ≈0 on Delta Control | **Control mode**, not difficulty |
| Action sequence execution | execute $T_a$ actions vs. 1 | +10–20 pp on 7/8 tasks; **reversed** on Hammer (17→27) | **Reactivity requirement** |
| Receding horizon control | $T_a < T_p$ vs. $T_a = T_p$ | ≈−15 pp on the 7 long-horizon tasks (Relocate 64→11); +2 on the one short-horizon task (Pen 71→73) | **Task horizon** |
| Denoising architecture | U-Net vs. MLP | catastrophic on hard tasks (PegInsertion 80→21, Relocate 64→7); nil on easy (StackCube 99→99) | **Difficulty** |
| **FiLM conditioning** | modulate vs. concatenate | +16 to +62 pp on 6 hard tasks; ≈0 or slightly negative on 2 easy | **Difficulty** |

Note the **collinearity hazard**: FiLM and the U-Net architecture share the same
moderator (difficulty) and the same worst-case task (Relocate: 64→7 without U-Net,
64→2 without FiLM). The paper ablates them independently but never jointly, so it
cannot separate "FiLM matters" from "FiLM matters *because* it is the only
conditioning route into a convolutional denoiser whose input channel axis is
action-shaped". A joint 2×2 was not run.

### 6.7 Relevance to this project

**Bears on our design — this is the corpus's single most decision-relevant paper,
but it argues for a *different* architecture than the one we built.**

- **It supports "modulate rather than concatenate", in a single-task setting.**
  This is direct evidence for open question 4 (single-task gains) and it does not
  depend on a task distribution at all: 8 independent single-task trainings, the
  moderator is difficulty. Our survival gridworld is single-task, so the older
  corpus verdict ("gains need task diversity") is not a reason to expect our
  modulator to fail.
- **It does *not* support our self-conditioning loop.** In the FiLM arm the
  denoiser has **no other path to the observation**; ours does. The gain here comes
  from making an observation-blind stream observation-aware. Our encoder is already
  observation-aware, so the same argument predicts a *smaller* effect for us — the
  multiplicative path is adding a Jacobian-reshaping capability on top of an
  already-informed pathway, not supplying missing information. This is a concrete,
  testable reason our single-environment comparisons could come back neutral.
- **It offers a cheap and directly runnable diagnostic.** The paper's manipulation
  is exactly the ablation our null results lack: hold the information constant,
  vary only the route. Our analogue would be "modulator GRU output as FiLM" versus
  "modulator GRU output concatenated onto the encoder input", both reading the same
  raw observation. If concatenation matches FiLM in our gridworld, our task is in
  the paper's *easy* regime and the modulator has nothing to do; if FiLM wins, the
  mechanism is live and the crash is a separate optimisation problem. **This would
  need a `senior-developer` issue plan; it is not a config-only change.**
- **The difficulty moderator is actionable.** Our survival gridworld may simply be
  too easy in the paper's sense — a near-unimodal action distribution whose shape
  varies little with the observation. Reporting task difficulty alongside our null
  results would strengthen them considerably.
- **Cite with the §6.4 caveats attached.** No seeds, no error bars, broken
  bibliography, single-author preprint. The 36-point and 62-point gaps carry the
  argument; nothing under ~10 points from this paper should be cited at all.

### 6.8 Appendix: Section-by-Section Backbone

Sections in the paper's original order.

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | Names the five components of Diffusion Policy: observation sequence input, action sequence execution, receding horizon, U-Net-or-Transformer architecture, FiLM conditioning. Experiments on ManiSkill and Adroit. |
| 1 | **Introduction** | Motivation: practitioners modify Diffusion Policy without examining components — cites Ren et al. (single frame, no receding horizon, MLP denoiser) and Ze et al. (4 actions instead of 16). Defines the five components. States the five findings verbatim and converts each into a recommendation. Fig. 1 shows the aggregate 8-task bar chart per component. |
| 2 | **Related Works** | One paragraph. Diffusion models as policies split into (i) RL uses — offline, offline-to-online, online — and (ii) imitation-learning uses. Diffusion Policy is category (ii). |
| 3 | **Diffusion Policy** | Umbrella section; Fig. 2 gives the reverse-diffusion loop with the score annotation $\nabla E(a)$. |
| 3.1 | Observation Sequence Input | Formal definition of the observation window $[O_t, O_{t-1}, \dots, O_{t-T_o}]$; positions Diffusion Policy in the history-dependent family with BeT and ACT. |
| 3.2 | Action Sequence Execution | Executes $[a_t, \dots, a_{t+T_a}]$ per inference, unlike single-action imitation methods. |
| 3.3 | Receding Horizon Control | Predicts $T_p$ actions, executes the first $T_a < T_p$. Framed as a trade-off between re-planning frequency and temporal action consistency. |
| 3.4 | Denoising Network Architecture | History: U-Net standard since DDPM; transformer denoiser stronger but "less stable training"; MLP was the RL default until Diffusion Policy popularised U-Net/transformer in robotics. |
| 3.5 | **FiLM Conditioning** | The shortest section in the paper — two sentences. Most imitation methods use the observation as network *input*; Diffusion Policy instead "treat[s] the observation sequence as FiLM conditioning on the denoising network", borrowed from vision. **No equation, no granularity statement, no site count, no initialisation.** |
| 4 | **Experiments** | Five research questions, one per component. Fig. 6 is the FiLM schematic (observation → two Linear layers → $(a,b)$ pairs → $a*x+b$ on action noise). |
| 4.1 | Experimental Setup | Benchmarks ManiSkill + Adroit; task types spanning stationary arm, mobile, dual-arm, dexterous, articulated, high-precision; demo sources TAMP / MPC / RL / human; **low-dimensional state observations only**. |
| 4.1.1 | Task Description | ManiSkill: StackCube, PegInsertionSide (3 mm clearance), TurnFaucet, PushChair — 1000 demos each. Adroit: Door, Hammer, Pen, Relocate — 24-DoF hand, 25 human teleoperation demos each. |
| 4.2 | Observation Sequence Input | Table 1. Splits tasks by Delta vs. Absolute control mode. −10 pp for Absolute, ≈0 for Delta. Takeaway box. |
| 4.3 | Action Sequence Execution | Table 2. +10–20 pp on most tasks; Hammer reverses (17 with, 27 without) due to loss of real-time feedback. Takeaway box. |
| 4.4 | Receding Horizon Control | Table 3, with a task-horizon/truncation column. 7 long-horizon tasks lose ≈15 pp without it (Relocate 64→11); the one short-horizon task (Pen, horizon 30) gains 2. Takeaway box. |
| 4.5 | Denoising Network Architecture | Table 4. U-Net vs. MLP, tasks labelled Easy/Hard. Hard tasks collapse under MLP; easy tasks unaffected. Takeaway box. |
| 4.6 | **FiLM Conditioning** | Table 5 — the corpus's key ablation, reproduced in §6.2 above. FiLM vs. observation-as-direct-input, 8 tasks, Easy/Difficult labels. Takeaway box: significant on hard, unnecessary on easy. |
| 5 | **Conclusions** | Restates the decomposition and the per-component recommendations. No limitations section, no future work, no discussion of variance. |

---
## 7. Reuss et al. 2025 — *FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies*

**PDF:** `docs/project/references/modulation_in_rl/sources/Reuss et al. 2025 - FLOWER - democratizing generalist robot policies.pdf`
**arXiv:** [2509.04996v1](https://arxiv.org/abs/2509.04996), 5 Sep 2025. Reuss, Zhou, Rühle, Yağmurlu, Otto, Lioutikov (KIT Intuitive Robots Lab / Microsoft Research). Project page + weights: `intuitive-robots.github.io/flower_vla/`.

### 7.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **CoRL 2025**, printed on the PDF's page-1 footer: *"9th Conference on Robot Learning (CoRL 2025), Seoul, Korea."* Verified from the PDF. |
| **Conditioning signal** | A concatenation of three things, none of which is the observation image: (i) the **flow timestep** $t$ (sinusoidal + MLP), (ii) an **action-space category** embedding (e.g. delta-end-effector vs. joint-angle), (iii) **embodiment / frequency metadata** and proprioception where available. Vision and language reach the transformer by a **different route entirely** — cross-attention on intermediate VLM tokens, not modulation. **This separation is architecturally explicit** and is the paper's own design (Fig. 2). |
| **Modulated component** | Every block of the **18-layer Flow Transformer action head**. Not the VLM, not an encoder, not a critic (imitation learning; no critic exists). |
| **Granularity cell** | **(b) per-channel** over the 1024-dim latent, unchanged from standard adaLN. **The paper's novelty is emphatically *not* on Axis 1** — it never coarsens the per-channel signal. |
| **Placement** | **All 18 blocks**, with multiple modulation points *inside* each block. Read from Fig. 3, the block applies: `Scale & Shift` before self-attention, a `Scale` (gate) after it, `Scale & Shift` before cross-attention, and `Scale & Shift` before the SwiGLU MLP — i.e. the standard DiT six-output adaLN pattern extended to a cross-attention sublayer. |
| **Parameterization** | **The contribution. One single set of modulation weights shared across all 18 layers** — "Action-Space Global-AdaLN-Zero" — replacing standard AdaLN-Zero's *distinct scale-and-shift parameters per layer*. Differentiation comes from the *input* (a distinct modulation signal per action category), not from per-layer weights. **Lost per-layer expressiveness is compensated by injecting lightweight LoRA adapters into each layer** (Fig. 3). This is an **Axis-3** move, not an Axis-1 or Axis-4 move. |
| **Initialization** | **Zero-init, explicitly, and explicitly for stability.** §3.2: the modulation signals are *"initialized with zeros for stable training"*. The paper does not ablate this — it treats zero-init as settled practice inherited from AdaLN-Zero. |
| **Ablation present?** | **Yes, one row, and it is the decisive one — but it is confounded (see §7.3).** Table 3 (CALVIN ABC, 3 seeds, ≤100k steps, average completed-sequence length out of 5): **FLOWER (Global-AdaLN + LoRA) = 4.44 ± 0.04** vs **"+ standard AdaLN" = 4.43 ± 0.03**. Indistinguishable, at 20 % fewer head parameters. **There is no `− LoRA` row anywhere in the paper.** |
| **Reported instability** | **Yes — two reports, and the corpus's only NaN.** (i) App. A.1: a richer **mixture-of-experts** flow-transformer design (a shared MLP combined with action-specific specialist MLPs, on top of action-type-specific LayerNorms) *"was more memory intensive and had issues with **NaN losses**. It also showed slower convergence."* Abandoned in favour of the simpler shared Global-AdaLN. (ii) App. A.3: an early model variant collapsed from 4.44 to **0.8** average sequence length; on later investigation this was attributed to *"instabilities in an earlier model variant"* — **cause never diagnosed**. Against these, App. A states of the final architecture: *"We did not notice major training instabilities."* No gain blow-up, no entropy or plasticity discussion. |
| **Single-task or multi-task** | **Massively multi-task and multi-embodiment**: 190 tasks, 10 benchmarks, 4 embodiments, ~250k pretraining trajectories. The conditioning variable (action-space category) exists *only because* the data is heterogeneous. **Gains here depend entirely on cross-embodiment diversity** and the paper offers no single-task evidence at all. |

### 7.2 Verification of the survey's claims

**Claim 1 — "Global-AdaLN-Zero ties *sites*, not *units*." CONFIRMED, verbatim, and this is the distinction open question 1 turns on.**

§3.2, quoted exactly:

> *"Standard AdaLN-Zero uses distinct scale-and-shift parameters per layer, adding up to 30 % extra parameters in diffusion transformers. In contrast, Action-Space Global-AdaLN-Zero **shares a single set of modulation weights across all layers**, while generating unique modulation signals for each action category (e.g., delta-EEF vs. joint angle), initialized with zeros for stable training."*

Parsed against the taxonomy: the object being shared is *the generator's weight set*, and the thing it is shared *across* is *transformer layers*. Within any one layer the modulation remains one $(\gamma,\beta)$ **per channel of the 1024-dim latent** — cell **(b)**, untouched. **Nothing is shared across a block of units.** FLOWER is therefore an **Axis-3** result and contributes **nothing** to the grouped-modulation question. The survey's caution was correct and is upheld.

**Claim 2 — "−20 % parameters with no performance loss (their Table 3)." CONFIRMED, with two precision corrections.**

- *Correction (a): it is 20 % of the **head**, not of the model.* The abstract's "cuts parameters by 20 %" is ambiguous, but §1 says *"reduces head parameters by 20 %"* and the conclusion says *"reduces transformer head parameters by 20 %"*. FLOWER's total is 947 M; the head is the 18-layer Flow Transformer. Citing "−20 % parameters" without "of the action head" overstates it.
- *Correction (b): the counterfactual parameter count is not published.* Table 6 lists `Global-AdaLN 28.3M`, `Flow Transformer 339M`, `Action Heads 31.8K`, total 947 M — but **gives no parameter count for the standard-AdaLN variant**, so the 20 % figure cannot be independently reconstructed from the paper. Cite it as the authors' claim.
- The performance parity itself is solid on its own terms: $4.44 \pm 0.04$ vs $4.43 \pm 0.03$ is well inside noise. But note the scope — **one benchmark (CALVIN ABC), 3 seeds, ≤100k steps, no pretraining.** The pretrained headline 4.53 (Appendix B) was never run in the ablation arm.

**Claim 3 — "FLOWER reports NaN losses from a richer modulation design." CONFIRMED as fact, CORRECTED as to what kind of design.**

The text is App. A.1's third bullet, and its own heading is **"Mixture-of-Experts Approaches for the Flow Transformer"**. The failing configuration is *action-type-specific specialist MLPs combined with a shared MLP* — a **shared-expert mixture-of-experts**, i.e. conditional *computation*, not a conditional *gain*. The affine modulation path is not what produced the NaN.

**This matters for open question 3 and the survey's framing should be tightened.** The survey called it *"the only 2024–2026 report I found of a modulation design being abandoned for numerical instability"* — true under this folder's broad definition of modulation, but a reader will hear "adaLN blew up", which is not what happened. **The correct statement is: the *simpler, coarser* modulation was the stable one, and the *expert-mixture* elaboration was the unstable one.** That is, if anything, *more* favourable to coarse modulation than the survey implied — and it remains the case that **no paper in this corpus reports gain blow-up.**

### 7.3 The confound the survey could not see: LoRA is never ablated

This is the single most important correction in this section, and it weakens how FLOWER can be cited.

The survey used FLOWER as *"the strongest published evidence to date that modulation capacity is over-provisioned by default"*. Full text shows the experiment cannot support that reading, because the comparison is not `coarse` vs `fine`. It is:

| Arm | Per-layer modulation weights | Per-layer LoRA adapters | Result |
|---|---|---|---|
| **FLOWER** | ✗ (one shared set) | ✓ | 4.44 ± 0.04 |
| **+ standard AdaLN** | ✓ (per-layer) | **unstated — Fig. 3 draws LoRA only on the Global-AdaLN block** | 4.43 ± 0.03 |

The paper states the design intent plainly: *"Given this lower per-layer expressiveness, we additionally inject lightweight LoRA adapters into each layer. This provides fine-grained, layer-specific modulation with only a few extra parameters per block."* **Per-layer capacity was not removed — it was relocated from the adaLN generator into LoRA adapters.** Table 3 contains no `− LoRA` row; Table 6 contains no LoRA parameter line; nothing anywhere isolates the coarsening.

**Two readings survive the evidence, and the paper cannot distinguish them:**

1. *(the survey's reading)* Per-layer modulation weights are over-provisioned; removing them costs nothing.
2. *(equally consistent)* Per-layer modulation capacity is necessary, and LoRA is simply a cheaper parameterisation of it. The gain is a **parameterisation efficiency** result, not a **capacity redundancy** result.

**Recommended citation form:** *"FLOWER shows that per-layer adaLN weight sets can be replaced by one shared set plus per-layer low-rank adapters at ~20 % fewer head parameters and no measured loss (CoRL 2025, Table 3) — but the low-rank compensation is not ablated, so this is evidence about parameterisation, not about how much modulation capacity a network needs."*

### 7.4 Phase 1 — Foundational overview

**Introduction.** A "vision-language-action model" is a robot policy that takes a camera image and a typed instruction ("put the red block in the drawer") and outputs motor commands. The best ones work, but they are enormous — 3 to 8 billion parameters — which means only well-funded labs can train or even run them. FLOWER's goal is stated in its title: make one that is competitive at **under one billion parameters**, trainable in **200 GPU-hours** instead of 21,500.

It gets there by two structural cuts. The first is about *where* the language model hands off to the action generator. Everyone else either fuses at the very start (input level) or at the very end (final layer output); FLOWER **throws away the last 30–50 % of the language model's layers** and taps the middle, on the argument that late layers over-specialise in predicting the next *word*, which a robot does not need. The second is about the steering machinery. Diffusion transformers normally give **every layer its own private set of steering knobs**; FLOWER gives all eighteen layers **one shared set**, and distinguishes the different robot types by feeding a different signal into that shared set rather than by having different knobs.

**Key findings.** Middle-layer fusion is not a small effect: on one benchmark it scores 93.4 % against 33.4 % for input-level fusion and 61.8 % for output-level fusion. The shared steering knobs cost nothing measurable (4.44 vs 4.43) while cutting the action head by about a fifth. The resulting 947 M-parameter model matches or beats models 3–8× its size across 190 tasks, runs at 311 Hz on a consumer graphics card (against 6 Hz for the 7-billion-parameter competitor), and fits in 1.85 GB of video memory.

**Initial takeaway.** Two things a builder of conditioned networks should take away. First, **where you tap a pretrained model matters more than how big it is**. Second, **giving every layer its own steering parameters may be a habit rather than a requirement** — but FLOWER did add a cheap per-layer patch when it removed them, so the honest lesson is "the per-layer capacity can be bought much more cheaply", not "you never needed it".

### 7.5 Phase 2 — Graduate-level deep dive

#### 7.5.1 Rectified flow as the action generator

FLOWER models the conditional action distribution $\pi_\theta(\bar a_{n,k} \mid \bar s_n, g, e)$ over an action chunk $\bar a_{n,k}\in\mathbb{R}^{d_a}$, given state $\bar s_n$, language goal $g$, and embodiment metadata $e$. Rectified flow posits a **straight-line** interpolant between data and noise:

$$
z_t = (1-t)\,\bar a_{n,k} + t\, z_1, \qquad z_1 \sim \mathcal{N}(0, I), \tag{1}
$$

with normalised flow time $t \in [0,1]$ drawn as $t \sim \sigma(\mathcal{N}(0,I))$ — a **logit-normal** schedule concentrating samples near $t=1/2$, where the velocity field is hardest to learn. The training objective regresses the (constant, by construction) velocity of that straight line:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, z_1}\Bigl[\bigl\lVert z_1 - \bar a_{n,k} - v_\theta\bigl(z_t, t, \bar s_n, g, e\bigr)\bigr\rVert^2\Bigr]. \tag{2}
$$

**Why the straight-line target is what makes the head cheap.** Differentiating (1), $\mathrm{d}z_t/\mathrm{d}t = z_1 - \bar a_{n,k}$, which is *independent of $t$*. The regression target is therefore a single displacement vector per sample rather than a $t$-dependent score, so the learned ODE $\dot z = v_\theta(z,t,\cdot)$ can be integrated with very few Euler steps — FLOWER uses $N=4$ for single-arm and $N=8$ for 50 Hz dual-arm. This is the direct source of the 311 Hz throughput, and it is why the conditioning mechanism must be cheap per step: it is evaluated $N$ times per action chunk.

#### 7.5.2 Standard AdaLN-Zero, then FLOWER's variant, written out

Let $h^{(\ell)} \in \mathbb{R}^{T\times d}$ be block $\ell$'s token activations, $d=1024$, $L=18$. Let $c$ be the conditioning embedding formed from flow timestep, action-space category, frequency and proprioception.

**Standard AdaLN-Zero** (the DiT baseline, one generator *per block*):

$$
\bigl[\gamma^{(\ell)}_1, \beta^{(\ell)}_1, \alpha^{(\ell)}_1, \gamma^{(\ell)}_2, \beta^{(\ell)}_2, \alpha^{(\ell)}_2\bigr] = \mathrm{MLP}^{(\ell)}(c), \qquad \mathrm{MLP}^{(\ell)} : \mathbb{R}^{d}\to\mathbb{R}^{6d},
$$

applied as

$$
\begin{aligned}
u &= h^{(\ell)} + \alpha^{(\ell)}_1 \odot \mathrm{Attn}\Bigl(\gamma^{(\ell)}_1 \odot \mathrm{Norm}\bigl(h^{(\ell)}\bigr) + \beta^{(\ell)}_1\Bigr),\\
h^{(\ell+1)} &= u + \alpha^{(\ell)}_2 \odot \mathrm{MLP}_{\mathrm{SwiGLU}}\Bigl(\gamma^{(\ell)}_2 \odot \mathrm{Norm}(u) + \beta^{(\ell)}_2\Bigr),
\end{aligned}
$$

with the final linear layer of each $\mathrm{MLP}^{(\ell)}$ zero-initialised, so $\alpha^{(\ell)}_\cdot = 0$ at step 0 and every block is exactly the identity. Parameter cost: $L \cdot 6d^2 \approx 18 \times 6 \times 1024^2 \approx 113\text{M}$ — which is why the paper says standard AdaLN-Zero adds *"up to 30 % extra parameters in diffusion transformers"*.

**FLOWER's Action-Space Global-AdaLN-Zero**: one generator, indexed by action category $k$ rather than by layer,

$$
\bigl[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2\bigr] = \mathrm{MLP}_{\text{global}}\bigl(c_k\bigr), \qquad c_k = \mathrm{emb}(k) + \mathrm{emb}_t(t) + \dots
$$

with **the same output reused at all $L$ blocks**, plus per-layer LoRA:

$$
W^{(\ell)}_{\text{eff}} = W^{(\ell)} + B^{(\ell)} A^{(\ell)}, \qquad B^{(\ell)} \in \mathbb{R}^{d\times r},\; A^{(\ell)} \in \mathbb{R}^{r\times d},\; r \ll d.
$$

**The algebraic point.** Standard AdaLN gives the network $L$ *independent* points in modulation space, one per depth. Global-AdaLN gives it **one** point, reused $L$ times — the modulation trajectory through depth is now *constant*. Depth-specific behaviour must therefore be recovered from the $L$ rank-$r$ weight perturbations $B^{(\ell)}A^{(\ell)}$, which multiply the *weights* rather than the *activations*. FLOWER trades $L$ per-layer **activation** modulations for $L$ per-layer **weight** modulations of rank $r$, and the parameter saving is $L\cdot 6d^2 - (6d^2 + L\cdot 2rd)$, positive whenever $r \ll 3d(L-1)/L$. **Both arms have per-layer capacity; only its algebraic form changed.** This is the derivation behind §7.3's confound.

#### 7.5.3 Intermediate fusion — the paper's largest effect, and why it is not modulation

The conditioning that carries vision and language is **cross-attention**, not modulation: VLM hidden states from an intermediate layer are mapped through a linear layer followed by RMSNorm (*"for increased stability"*) and cross-attended by each flow block. The ablation (Table 1, 3 seeds):

| Fusion strategy | Florence-VLM: CALVIN-ABC | Florence-VLM: LIBERO-Long | SmolVLM: C-ABC | SmolVLM: L-Long |
|---|---|---|---|---|
| Early (concatenate at LLM input) | 57.1 ± 5.3 | 33.4 ± 6.0 | 25.8 ± 3.9 | 44.5 ± 2.7 |
| **Intermediate (ours)** | **89.5 ± 1.0** | **93.4 ± 2.0** | **72.1 ± 5.0** | **70.7 ± 2.3** |
| Late (final VLM output) | 71.2 ± 2.2 | 61.8 ± 2.5 | 66.3 ± 2.0 | 69.2 ± 1.9 |

Layer-pruning sweep (Table 2, SmolVLM): Full 66.3, drop 20 % → 68.6, drop 30 % → **72.1**, drop 50 % → 66.4. A clear interior optimum.

**Worth naming explicitly**: FLOWER's two conditioning routes are *deliberately different mechanisms for different signal types* — cross-attention for the rich, token-structured signal (vision + language), affine modulation for the low-dimensional, categorical signal (timestep, action space, embodiment). This is the same split $\pi_0$ makes (see [§17](#17-appendix--adjacent-evidence-from-the-six-unreviewed-pdfs)) and matches the survey's "affine modulation is displaced by attention once the conditioner is a rich token sequence" trend statement.

#### 7.5.4 The full design-decision ablation (Table 3, CALVIN ABC, avg. sequence length / 5)

| Variant | Avg. Len. | Δ vs FLOWER |
|---|---|---|
| **FLOWER** | **4.44 ± 0.04** | — |
| + standard AdaLN | 4.43 ± 0.03 | −0.01 (**null**) |
| − Custom LR scheduler | 4.40 ± 0.05 | −0.04 |
| + small Florence | 4.26 ± 0.04 | −0.18 |
| − VLM | 3.42 ± 0.07 | −1.02 |
| − Flow Head (L1 head, ACT-style) | 3.33 ± 0.04 | −1.11 |
| − No VLM train (frozen VLM) | 2.65 ± 0.36 | −1.79 |
| + Smaller Head (384-dim, 6 layers) | 2.60 ± 0.09 | −1.84 |
| + Discrete Token | 1.12 ± 0.12 | −3.32 |

**The modulation ablation is the smallest effect in the table by an order of magnitude.** Everything that matters here is about capacity allocation and the generative objective; the modulation *parameterisation* is free either way. Pretrained FLOWER reaches **4.53 ± 0.04** (Appendix B), the paper's SOTA claim.

#### 7.5.5 The stability apparatus, catalogued

FLOWER stacks an unusual number of numerical-stability devices, which is worth recording as a picture of 2025 best practice for a modulated transformer:

| Device | Where | Stated purpose |
|---|---|---|
| **Zero-initialised modulation** | Global-AdaLN output | *"initialized with zeros for stable training"* |
| **RMSNorm on the VLM→flow projection** | fusion interface | *"for increased stability"* |
| **RMSNorm instead of LayerNorm** | throughout | *"yielding smoother gradients and reduced training instability"*; $\mathrm{RMSNorm}(x) = x / \sqrt{\tfrac1d\sum_j x_j^2 + \epsilon}$ (Eq. 4) |
| **QK-normalisation** | self- and cross-attention | *"to mitigate large softmax outputs"* |
| **Per-action-type dual-RMSNorm parameters** | per action space | *"capturing action-specific activation statistics"* under heterogeneous data frequencies |
| **SwiGLU sandwich MLP** | FFN | Eq. 3, $y = \mathrm{Norm}(W_1 h)\odot\mathrm{SiLU}(\mathrm{Norm}(V_1 h))$ |
| **BF16 + dual-optimizer LR schedule** | training | separate warm-up/constant/cosine phases for VLM ($10^{-5}$) and flow transformer ($10^{-4}$) |

App. A.4 summarises: *"(i) extended RoPE embeddings, (ii) action-specific encoders/decoders, (iii) a shared AdaLN controller yielding action-specific normalization signals, and (iv) RMSNorm with SwiGLU MLPs enable our Flow Transformer to train stably across heterogeneous data and multiple action spaces."*

### 7.6 Relevance to this project

**Bears on our design in three specific ways, and is irrelevant in two others. Be careful which.**

- **It is *not* a grouped-modulation precedent, and must not be cited as one.** Nothing here shares a $(\gamma,\beta)$ across a block of units. Our $G$ sweep gains no support from FLOWER. Even the weaker claim ("modulation capacity is over-provisioned") is undercut by the unablated LoRA compensation (§7.3). **This is a correction to how our own survey framed it, and it should propagate to any grant text or paper draft that used FLOWER as premise for the grouping screen.**
- **Zero-initialisation as universal prophylactic — one more independent instance, and a concrete code check.** FLOWER zero-inits the modulation generator's output *for stability*, with no ablation because it is treated as settled. Our modulator adds onto a zero-initialised per-neuron baseline, which is the additive half of the same idea. **Whether our generator's output weights are zero-initialised at step 0 is a code question worth answering**, and if they are not, this paper plus GR00T ([§11](#11-nvidia-et-al-2025--gr00t-n1-an-open-foundation-model-for-generalist-humanoid-robots)) and HyperMARL (see [§17](#17-appendix--adjacent-evidence-from-the-six-unreviewed-pdfs)) are three independent citations for making them so. Hand to `senior-developer`.
- **The stability catalogue is a menu we have not shopped from.** Our $G=1$ crash featured a railing temperature and ballooning gain variance. FLOWER's answer to "large activations in a conditioned transformer" is QK-normalisation plus RMSNorm plus zero-init plus per-condition normalisation statistics. We use none of these on the modulator path. **None is a proven fix for our failure mode** — nobody has reported our failure mode at all (open question 3) — but the list is where a fix would be looked for first.
- **The two-route split is a design pattern we could adopt and currently do not.** FLOWER routes the *rich, high-dimensional* signal (vision+language) through cross-attention and the *low-dimensional, categorical* signal (timestep, action space) through modulation. Our modulator does the opposite: it pushes the **raw, full-dimensional observation** through the modulation path. Every paper in this corpus that modulates does so with a *low-dimensional* conditioner. That is a structural difference between us and the field worth stating plainly in our write-ups — and it is orthogonal to (and compounds with) the self-conditioning divergence flagged in [§17](#17-appendix--adjacent-evidence-from-the-six-unreviewed-pdfs).
- **The NaN report does not transfer.** It is a mixture-of-experts instability in a 950 M-parameter transformer trained in BF16 on 250k trajectories. Our failure is a 128-unit GRU policy in a gridworld. There is no mechanistic bridge; do not cite FLOWER as precedent for our crash.
- **Single-task: no evidence either way.** FLOWER is maximally multi-task. Open question 4 gains nothing here.

### 7.7 Appendix: Section-by-Section Backbone

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | Two named contributions: **intermediate-modality fusion** (prune up to 50 % of LLM layers) and **action-specific Global-AdaLN conditioning** (−20 % parameters). 950 M-parameter VLA, 200 H100-hours pretraining, 190 tasks / 10 benchmarks, new SOTA 4.53 on CALVIN ABC. |
| 1 | **Introduction** | The capacity trade-off: a deep VLM backbone starves the diffusion head; shrinking the VLM strips semantics. Fig. 1: VRAM 1.85 GB vs π0's 6.69 and OpenVLA's 14.57; pretraining 192 h vs 21,500 (OpenVLA) and 35,000 (RDT-1B). Four contributions listed. |
| 2 | **Related Work** | Data-scaling lineage (Pinto & Gupta 50k grasps; OXE 1.4 M trajectories). Diffusion policies without VLMs (Octo, RDT-1B — one month on 48 A100s). VLM-based VLAs (OpenVLA 7.7B; π0 and GR00T-N1 both >2B, closed data). Fusion-strategy taxonomy: early / late / **intermediate (ours)**. Notes GR00T-N1 also explored intermediate fusion *"without any ablation and insights on the impact"*. |
| 3 | **Method** | Objective $\mathcal{L}_{IL} = \mathbb{E}[\log\pi_\theta(a\mid s,g,e)]$ over heterogeneous action spaces, observation spaces and task specifications. |
| 3.1 | Intermediate Modality Fusion | Motivated by LLM-explainability work showing the penultimate quarter of layers carries broad semantics while final layers specialise for next-token prediction. Encoder–decoder VLMs (Florence-2): drop the whole decoder (−50 % layers). Decoder-only VLMs (SmolVLM2-Video): drop the final 30 %. VLM hidden states → linear → RMSNorm → cross-attention into every flow block. |
| 3.2 | **Cross-Action Space Flow Transformer** | **The Global-AdaLN section.** Fig. 3 contrasts a standard DiT block with the Global-AdaLN + per-layer-LoRA block. Standard AdaLN-Zero = distinct scale-and-shift per layer, up to +30 % parameters; Global-AdaLN-Zero = one shared modulation weight set across all layers, unique signal per action category, zero-initialised; LoRA adapters restore per-layer expressiveness; per-action-type encoder/decoder handles differing action dimensions. |
| 3.3 | Rectified Flow for Action Generation | Eq. 1 (straight-line interpolant, logit-normal $t$), Eq. 2 (velocity regression loss). $N=4$ denoising steps single-arm, $N=8$ dual-arm. |
| 3.4 | FLOWER | Half of Florence-2-L as backbone; **18-layer Flow Transformer, 1024 latent**; 947 M total, 1.85 GB VRAM. Pretraining "OXE-soup" of eight datasets (~250k trajectories), 75 % from Droid / Google Robot / BridgeV2 for scene diversity; chunk length 20; 360k steps in 48 h ≈ 200 GPU-hours; 74 % delta-EEF / 26 % joint-state. |
| 4 | **Evaluation** | Three research questions: design-element impact, efficiency vs SOTA, cross-embodiment robustness. |
| 4.1 | Evaluation of Critical Design Decisions | Table 1 (fusion strategy: intermediate wins on both VLMs, both benchmarks). Table 2 (pruning fraction: 0.3 optimal). **Table 3 (the modulation ablation and seven others).** Backbone comparison: Florence-2's detection/grounding pretraining transfers better than SmolVLM's general-reasoning pretraining. Head capacity is crucial (384-dim/6-layer head → 2.60). |
| 4.2 | Simulation Experiments | CALVIN (A–D, 34 tasks, 24k demos, 1000 instruction chains of 5), LIBERO (Long/Spatial/Object/Goal + LIBERO-90), Aloha (bimanual 50 Hz), SIMPLER (Bridge + Google Robot real2sim). Fig. 5 vs DP, OpenVLA, π0, best-baseline. |
| 4.3 | Real-World Evaluation and Generalization | Franka Panda kitchen, 20 tasks, 417 demos from 45 min of kinesthetic teaching, 6 Hz joint space. FLOWER 61 % vs OpenVLA 31 %, CrossFormer 21 %, Octo 10 %. Generalization: novel object 33.3 vs 10.0, flashlight 50.0 vs 25.0, background distractors 69.5 vs 41.7, new task composition 51.1 vs 16.7 (avg 51.0 vs 23.4). Table 4 inference: 311 Hz / 0.052 s / 1848 MB vs π0 288 Hz / 6692 MB vs OpenVLA 6.09 Hz / 14574 MB. |
| 5 | **Conclusion** | Restates the two mechanisms and the headline efficiency numbers. |
| 6 | **Limitations** | Iterative sampling is slower than a single forward pass; validated on only three manipulation action spaces (no locomotion / navigation); SIMPLER Google Robot zero-shot still weak, hypothesised to need larger models; ~1 B may still be too big for some deployments; **8 of 10 benchmarks are simulation**. |
| A | Pretraining Details | Table 7 dataset mix (bridge 28.6 %, fractal 24.7 %, droid 23.5 %, …). 4× H100, 48 h, BF16, batch 256 × 4 accumulation = 1024, 300–400k steps. *"We did not notice major training instabilities."* Table 8 full hyperparameters (dual AdamW, flow LR 1e-4, VLM LR 1e-5, no EMA). |
| A.1 | **Pretraining Ablation Experiences** | Three things that did not work: variable-length action chunks (slow convergence); multiple images with masking (too slow to converge in 2 days); **mixture-of-experts flow transformer — "more memory intensive and had issues with NaN losses… slower convergence"**, resolved by falling back to action-specific Global-AdaLN with everything else shared. |
| A.2 | Language Prompt | *"Agent Type: [robot type], Action Space: [action space], Task: [task description]"* — the embodiment metadata is literally text in the VLM prompt. |
| A.3 | Custom Learning Rate Scheduler | The 4.44-vs-0.8 collapse, later attributed to *"instabilities in an earlier model variant rather than the absence of the scheduler"*; with the final architecture a constant $2\times10^{-5}$ costs almost nothing. |
| A.4 | Details for Cross-Action Space Flow Transformer | Per-action-type dual-RMSNorm; Eq. 3 SwiGLU; Eq. 4 RMSNorm; QK-normalisation; the four-device stability summary. |
| B | Detailed Experiments | Per-benchmark tables. **CALVIN ABC: FLOWER non-pretrained 4.44 ± 0.04, pretrained 4.53 ± 0.04** (99.4 / 95.8 / 90.7 / 84.9 / 77.8 % across the five sequential tasks). LIBERO, Aloha, SIMPLER details; OpenVLA baseline trained 150k steps with LoRA updates for memory reasons. |

---
## 8. Yoon et al. 2026 — *PAPL: Phase-Aware Policy Learning for Skateboard Riding of Quadruped Robots via Feature-wise Linear Modulation*

**PDF:** `docs/project/references/modulation_in_rl/sources/Yoon et al. 2026 - PAPL - phase-aware policy learning via FiLM.pdf`
**arXiv:** [2602.09370v2](https://arxiv.org/abs/2602.09370), v2 dated 21 Apr 2026. Minsung Yoon\*, Jeil Jeong\*, Sung-Eui Yoon.

> **This is the closest architectural match to our agent in the entire corpus**: a small dense FiLM-modulated MLP, trained with PPO, in a single task, with modulation applied pre-activation and **no normalisation layer anywhere in the modulation path**. Read §8.6 carefully.

### 8.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **Unstated in PDF.** The PDF carries only `arXiv:2602.09370v2 [cs.RO] 21 Apr 2026` — no venue line, no footer, no acceptance note. The motivating survey recorded "ICRA 2026" from the **arXiv comments field**, which is a legitimate but *external* source. Per this folder's rule, recorded here as **unstated in PDF**; cite the venue only with the arXiv-comments provenance attached. |
| **Conditioning signal** | The **phase embedding** $\Phi_t = (\cos\phi_t, \sin\phi_t)^\top \in \mathbb{R}^2$, where $\phi_t = 2\pi t / T_\phi \bmod 2\pi$ is an **open-loop clock**, not an inferred state. $T_\phi \sim \mathcal{U}(4.0, 12.0)\,\mathrm{s}$ is resampled per episode. **Two dimensions total.** Crucially, $\Phi_t$ is *also* a component of the observation tuple $o$ fed directly to both networks — see §8.3. |
| **Modulated component** | **Both the actor and the critic** (Table II: $f^{\text{critic}}_\theta$ = $\mathcal{F}(1024, 512, 256, 1)$, $f^{\text{actor}}_\theta$ = $\mathcal{F}(512, 256, 128, 12)$, where $\mathcal{F}(\cdot)$ denotes a FiLM-modulated MLP). **The encoders are not modulated** — the scan encoder $e^{\text{scan}}_\theta$, intrinsic encoder $e^{\text{enc}}_\theta$ and all four DAgger estimators are plain MLPs or CNN-GRUs, marked `[·]` in Table II. |
| **Granularity cell** | **(a) per-unit** (≡ (b), dense layers). $\gamma_\ell(\Phi_t)\odot z$ with $\odot$ element-wise over a dense pre-activation vector, so one $(\gamma,\beta)$ per hidden unit — 512 + 256 + 128 for the actor, 1024 + 512 + 256 for the critic. **A 2-dimensional conditioner expands to ~1900 modulation parameters per forward pass.** |
| **Placement** | **Every layer of both MLPs**, applied **pre-activation and with no normalisation layer** — $h_\ell = \sigma(\gamma_\ell(\Phi_t)\odot z + \beta_\ell(\Phi_t))$ where $z = W_\ell h_{\ell-1} + b_\ell$. This is *raw* Perez-style FiLM on dense layers, not adaLN. **Ambiguity flagged:** the prose says "each layer", and Table II's caption says the listed dimensions are "intermediate and final layer dimensions", so whether the 12-dim actor output and 1-dim critic output are themselves modulated is **not resolvable from the PDF**. |
| **Parameterization** | **Not stated.** The paper writes $\gamma_\ell(\Phi_t)$ and $\beta_\ell(\Phi_t)$ without specifying whether one shared generator feeds per-layer heads or each layer has its own generator. Table II gives no parameter count for the modulation path. A genuine gap. |
| **Initialization** | **Not stated anywhere.** No zero-init, no identity-init, no claim that it matters. Given the absence of any normalisation in the modulation path, this is a notable omission. |
| **Ablation present?** | **Yes — the mechanism itself is ablated (`No-FiLM`), but reported only graphically.** Fig. 7 compares PAPL against **No-FiLM** (standard MLP backbones for **both** actor and critic), No-Extero, No-Priv, and a **Mixture-of-Experts MLP actor**. Verdict quoted: *"the No-FiLM variant … exhibits a narrower region with higher errors. This empirically demonstrates that the inductive bias introduced by the FiLM architecture is crucial to address the multi-modality inherent in skateboarding motions."* **No numerical values are printed anywhere** — the result exists only as tracking-error heatmaps and command-area curves. Quotable as direction, not as magnitude. |
| **Reported instability** | **Yes, and it is a genuine find — but it is an *expert* collapse, not a gain blow-up.** *"We also evaluated the Mixture-of-Experts (MoE) MLP as the actor backbone in place of the FiLM MLP, but it failed to achieve effective skateboarding behaviors as **the pushing-phase expert collapsed without producing meaningful actions**."* Diagnosed as phase-specific isolation, limited information sharing between experts, and parameter count forcing fewer parallel environments. Separately, the `No-Priv` variant *"converges to a standing-still behavior to avoid penalties"* — a degenerate-policy collapse attributed to privileged learning, not modulation. **No gain blow-up, no NaN, no entropy or temperature discussion.** |
| **Single-task or multi-task** | **Single task — one skill (skateboard riding), no task distribution, no task ID.** The conditioner is a **within-episode** variable. Variation exists only in the velocity/yaw command $c = (c_v, c_\omega)$ and the cycle period $T_\phi$. **This confirms the survey's claim and is the corpus's cleanest single-task FiLM-in-RL result.** |

### 8.2 Verification of the survey's claims

| Survey claim | Verdict | Evidence |
|---|---|---|
| Venue ICRA 2026 | **Not verifiable from PDF** | No venue line in the PDF; survey's source was the arXiv comments field. Not a contradiction — a provenance downgrade. |
| Conditioning on a cyclic phase index, a *within-episode* variable rather than a task ID | **Confirmed** | §IV-A: $\phi_t = 2\pi t/T_\phi \bmod 2\pi$, embedded as $(\cos\phi_t,\sin\phi_t)$ |
| **Modulates both the actor and the critic** | **Confirmed, and the paper gives a principled reason (§8.4)** | Table II: both networks are $\mathcal{F}(\cdot)$ |
| Granularity (a)/(b) per-unit in dense layers | **Confirmed** | §IV-B modulation equation, element-wise on dense pre-activations |
| "the configuration Ben-Iwhiwhu et al. 2022 reported as unstable" | **No contradiction found, but no engagement either** | PAPL does not cite Ben-Iwhiwhu, does not discuss critic-modulation stability, and reports no instability from it. It is a silent existence proof, not a rebuttal. |
| Per-component ablations validated in simulation; real-world transfer demonstrated | **Confirmed** | Fig. 7 ablations; §V-D zero-shot Unitree Go1 transfer under perturbation, low light, uneven sidewalks |

### 8.3 The finding the survey missed (1) — the conditioner is *also* a direct network input

Table II lists the actor's input as $[o, z_{\text{per}}, z_{\text{int}}, x_{\text{ext}}]$ (91 dimensions) and the critic's as $[o, x_{\text{scan}}, x_{\text{full}}]$ (1702 dimensions). §IV-B defines $o$ explicitly:

> *"the observation $o$ includes proprioceptive information $o_{\text{prop}}$, the command $c$, **the phase embedding $\Phi_t$**, and the previous action $a_{t-1}$"*

**So $\Phi_t$ enters each network twice: once as two raw input features, and once as the FiLM conditioner for every layer.** The modulator's information is a strict subset of the modulated network's direct input.

**Why this matters for open question 2.** This is *not* our architecture — our modulator reads the full raw observation, ours is recurrent, and ours is 16 units wide. But it *is* a published instance of the structural property that our surveys flagged as unprecedented: **a modulator whose input the modulated stream already receives**, in an on-policy RL policy, working, and beating the unmodulated baseline. It is the closest thing to a positive precedent this corpus contains, and it is much closer than Yuan 2024 (where the denoiser has no other route to the observation).

**And it sharpens the design question.** Because $\Phi_t$ reaches the network by both routes simultaneously, the `No-FiLM` ablation is *not* "with vs. without phase information" — the No-FiLM network still gets $\Phi_t$ as input. **It is a pure routing comparison: concatenation-only versus concatenation-plus-modulation.** That makes it the second such comparison in the corpus (after Yuan 2024's Table 5), run in a completely different setting, and it goes the same way: **the multiplicative route adds something concatenation alone does not deliver.** The survey did not identify this, and it is the paper's most transferable result.

### 8.4 The finding the survey missed (2) — *why* the critic is modulated

Open question 5 asks for every paper's actor/critic choice **and any stated reason**. PAPL is the only paper in the corpus that supplies one, and it is a good one.

The reward is **mode-dependent by construction**: $R_M = R_{\text{dep}} + R_{\text{ind}}$, where $R_{\text{dep}} = \sum_{i=1}^{7} r_i$ has **structurally different formulations** in CARVING versus PUSHING (Table III). For instance the command-tracking error term is

$$
\varepsilon_1 = \begin{cases} \lvert c_\omega - \omega^S_{D,z}\rvert & M(\phi_t) = \text{CARVING}\\[2pt] 0.6\lvert c_v - v^S_{D,x}\rvert + 0.2\lvert v^S_{D,y}\rvert + 0.2\lvert\omega^S_{D,z}\rvert & M(\phi_t) = \text{PUSHING} \end{cases}
$$

— it does not merely take different *values* by phase, it measures a **different quantity**. Same for body position, foot position, riding posture, foot slip, contact pattern and foot clearance ($r_2$–$r_7$). Only $r_8$–$r_{11}$ (alignment, smoothness, stabilisation, safety) are phase-independent.

**Hence the value function is genuinely a different function per phase**, and §V-A verifies this empirically before the design is defended: Fig. 6 shows that the **critic-value histogram exhibits distinct mode-dependent distributions**, alongside t-SNE projections in which PUSHING and CARVING actions and observations occupy separated regions with TRANSITION in between.

**The extractable rule.** *Modulate the critic when the reward function itself is conditioned on the modulating variable.* PAPL's critic modulation is not a bold generalisation of actor modulation; it is a direct consequence of a phase-conditioned reward. **This is a necessary condition our project does not currently meet** — our reward is not a function of the modulator's input — and that is a specific, checkable reason our critic-side modulation might not pay off the way PAPL's does.

### 8.5 Phase 1 — Foundational overview

**Introduction.** Legged robots walk well but waste energy doing it. A skateboard is a cheap fix: no motor, low rolling resistance, and steering that works by leaning. The catch is that riding one is not a single skill — it is a **cycle of qualitatively different skills**. In the *pushing* phase one foot is on the ground shoving; in the *carving* phase all four feet are on the board and the robot steers by tilting; in between there are transitions where feet move between the ground and the deck. Each phase has different contact patterns, different physics, and different goals.

The obvious approaches both fail. One network trained on everything blurs the phases together. Separate networks per phase cannot share what they all need to know about being a quadruped, and hand off badly at transitions. PAPL takes a third route: **one network, whose internal computation is re-tuned on the fly by a clock**. A cyclic phase signal (just two numbers, the sine and cosine of a clock angle) is used to generate a gain and an offset for every hidden unit in both the action network **and the value network**, so a single set of weights behaves like a different network in each phase while sharing everything general underneath.

**Key findings.** The steering works: a real Unitree Go1 rides a real skateboard, transferring from simulation with no real-world fine-tuning, and keeps working under shoves, in the dark, and on uneven pavement. Skateboarding uses less motor power than either walking or wheel-legged locomotion over a 30 m course. The ablations show the phase modulation is not decorative — removing it and using ordinary networks narrows the range of speed and turn-rate commands the robot can follow accurately. And the alternative way of getting phase-specific behaviour — a *mixture of experts*, with a separate sub-network per phase — **failed outright**: the pushing expert collapsed and produced nothing.

**Initial takeaway.** When a single task has internal *phases*, you do not need a task distribution to benefit from conditioning. A clock is enough. And the phase signal should reach the **value estimator** too, whenever the thing being rewarded changes between phases.

### 8.6 Phase 2 — Graduate-level deep dive

#### 8.6.1 The phase-conditioned POMDP

The task is cast as a **phase-conditioned POMDP** $(\mathcal{S}, \mathcal{O}, \mathcal{A}, \mathcal{T}, R_M, \rho_0, \gamma)$ where the reward $R_M$ depends on a discrete mode determined by a continuous clock:

$$
\phi_t = \frac{2\pi t}{T_\phi} \bmod 2\pi, \qquad M(\phi_t) = \begin{cases} \text{CARVING} & \phi_t \in [0.2\pi,\, 0.8\pi]\\ \text{PUSHING} & \phi_t \in [1.2\pi,\, 1.8\pi]\\ \text{TRANSITION} & \text{otherwise.} \end{cases}
$$

The objective marginalises over both the command distribution and the cycle period:

$$
J(\theta) = \mathbb{E}_{c\sim P(c),\; T_\phi\sim P(T_\phi)}\Biggl[\mathbb{E}_{s_0\sim\rho_0,\;(s,a)\sim\rho_{\pi_\theta}}\Bigl[\sum_{t=0}^{\infty}\gamma^t R_{M(\phi_t)}(s_t, a_t \mid c)\Bigr]\Biggr]. \tag{2}
$$

**Three design consequences worth naming.**

1. **The clock is exogenous and open-loop.** $\phi_t$ advances with wall time regardless of what the robot achieves. The policy cannot stall the phase to buy time; it must *meet* the clock. This converts a hard mode-discovery problem into a tracking problem — an enormous simplification, and the reason a 2-dimensional conditioner suffices.
2. **Randomising $T_\phi \sim \mathcal{U}(4,12)\,\mathrm{s}$ per episode forces the modulation to encode phase *shape*, not phase *timing*.** If $T_\phi$ were fixed, $\phi_t$ would be a deterministic function of $t$ and the network could learn a fixed open-loop trajectory. Randomising the period breaks that shortcut. **This is the single most transferable trick in the paper** and its analogue for us would be randomising whatever period or threshold defines an interoceptive phase.
3. **TRANSITION has no reward terms at all.** §IV-B: *"While the TRANSITION mode has no dedicated reward terms, maximizing expected returns induces mounting and dismounting behaviors."* The transition behaviour is *emergent*, driven purely by the need to be well-positioned when the next scored phase begins. This is an elegant demonstration that a phase-conditioned reward need not specify every phase.

#### 8.6.2 The modulation, written out

Let $h_{\ell-1}$ be the layer-$(\ell-1)$ activation. The FiLM-modulated MLP layer $\ell$ computes:

$$
z = W_\ell h_{\ell-1} + b_\ell, \qquad h_\ell = \sigma\Bigl(\gamma_\ell(\Phi_t)\odot z + \beta_\ell(\Phi_t)\Bigr),
$$

with $\Phi_t = (\cos\phi_t, \sin\phi_t)^\top$, $\odot$ element-wise, and $\sigma$ a nonlinearity.

**Four properties, in decreasing order of importance to us.**

**(i) Modulation is pre-activation and there is no normalisation layer.** Compare adaLN (FLOWER, GR00T), where $\gamma,\beta$ multiply the *output of a LayerNorm* whose scale is already controlled. Here $z$ is a raw affine pre-activation with unconstrained scale, and $\gamma_\ell$ multiplies it directly. **The gain therefore has no variance regulator standing behind it.** That is exactly our architecture, and it is the *only* paper in this corpus of which that is true. It also means PAPL is the one paper where gain blow-up was structurally *possible*; none is reported.

**(ii) A 2-dimensional conditioner drives ~1900 modulation parameters.** The actor's modulation path maps $\mathbb{R}^2 \to \mathbb{R}^{2(512+256+128)} = \mathbb{R}^{1792}$; the critic's maps $\mathbb{R}^2 \to \mathbb{R}^{2(1024+512+256)} = \mathbb{R}^{3584}$. Because $(\cos\phi,\sin\phi)$ traces a **one-dimensional closed curve** (the unit circle), the reachable set of modulation vectors is a 1-D loop embedded in that high-dimensional space — the network's per-layer behaviour is constrained to a **smooth cyclic family**, which is precisely the inductive bias claimed: *"facilitating smooth transitions and efficient motor skill reuse across the skateboard-riding cycle."*

**(iii) The $(\cos,\sin)$ encoding is load-bearing, not cosmetic.** A raw $\phi_t \in [0,2\pi)$ would be **discontinuous at the wrap point**: $\phi = 2\pi - \varepsilon$ and $\phi = 0$ are adjacent in time but maximally distant as inputs, so any Lipschitz generator would produce a modulation jump exactly at the cycle boundary. Embedding on the circle makes the generator's input continuous in time, so $\gamma_\ell$ and $\beta_\ell$ vary smoothly through the wrap. **Any project encoding a cyclic or phase-like variable must do this**; it is the standard fix and PAPL applies it correctly.

**(iv) Modulation as a Jacobian-rescaling family.** As derived for Yuan 2024 (§6.6.3), $\partial h_\ell / \partial h_{\ell-1} = \mathrm{diag}(\sigma')\,\mathrm{diag}(\gamma_\ell(\Phi_t))\,W_\ell$. The phase does not merely translate the layer's output; it **reweights which input directions propagate**. Under a mode-dependent reward whose *arguments change* between modes (§8.4), this is exactly the right capability: different phases need different proprioceptive channels amplified. §V-A's t-SNE separation of PUSHING and CARVING observations is the empirical shadow of this.

#### 8.6.3 Asymmetric privileged learning + distillation

PAPL's modulation sits inside a two-stage scheme that is worth recording because it determines what the ablations mean.

**Stage 1 (PPO, asymmetric actor–critic).** The critic sees the full privileged state $x_{\text{full}}$ (1702-dim input); the actor sees only $[o, z_{\text{per}}, z_{\text{int}}, x_{\text{ext}}]$ (91-dim), of which $z_{\text{per}}$ (perceptual latent from a scan encoder), $z_{\text{int}}$ (intrinsic dynamics latent) and $x_{\text{ext}}$ (deck pose relative to body, foot states) are simulation-only.

**Stage 2 (DAgger distillation).** Three estimators replace the simulation-only quantities at deployment, trained by

$$
\mathcal{L}_{\text{DAgger}}(\varphi) = \mathbb{E}_{\mathcal{D}}\Bigl[\lVert z_{\text{per}} - \hat z_{\text{per}}(\varphi)\rVert_2^2 + \lVert x_{\text{ext}} - \hat x_{\text{ext}}(\varphi)\rVert_2^2 + \lVert z_{\text{int}} - \hat z_{\text{int}}(\varphi)\rVert_2^2\Bigr]. \tag{3}
$$

The perceptual estimator $e^{\text{per}}_\varphi$ maps five belly-camera frames ($128\times256\times5$) through a 2-D CNN-GRU to $\hat z_{\text{per}}$; RGB is HSV-filtered by deck colour into semantic masks and perturbed with salt-and-pepper noise or zeroed entirely for robustness. The extrinsic and intrinsic estimators are 1-D CNN-GRUs over 10-step observation histories.

**Note what this implies about the ablations.** The **estimators and encoders are not FiLM-modulated**. Only the two decision networks are. So `No-FiLM` isolates modulation of the *policy and value* computation specifically, with the perception stack held fixed — a cleaner comparison than it first appears.

#### 8.6.4 The MoE failure, and what it adds to the corpus

Fig. 5 sets up exactly three MLP variants: (a) standard, (b) FiLM-modulated with phase conditioning, (c) **Mixture-of-Experts with phase-based expert weight blending**. Variant (c) was run as the actor backbone and failed:

> *"it failed to achieve effective skateboarding behaviors as the pushing-phase expert collapsed without producing meaningful actions. This failure was likely due to phase-specific isolation, limited information sharing among experts, and the large parameter count, which under limited GPU memory restricted training to fewer environments, reducing data collection and exploration efficiency and leading to trivial behaviors such as remaining inactive."*

**This is the second independent report in this corpus of a mixture-of-experts elaboration failing where the simpler affine modulation succeeded** — FLOWER ([§7](#7-reuss-et-al-2025--flower-democratizing-generalist-robot-policies-with-efficient-vision-language-action-flow-policies)) hit NaN losses with a shared-plus-specialist expert MLP and fell back to shared Global-AdaLN. Two papers, different fields (robot flow policies vs. quadruped PPO), different failure signatures (NaN vs. expert collapse), same conclusion.

**The diagnosis is also mechanistically interesting and directly relevant to grouping.** PAPL attributes the collapse to *"phase-specific isolation, limited information sharing among experts."* A mixture of experts is the **discrete limit** of conditional computation — each condition gets its own parameters, nothing is shared. FiLM is the opposite extreme — **all** parameters shared, with only a gain and offset per condition. PAPL's result is that on this task the fully-shared end wins and the fully-isolated end collapses. **That is an argument for parameter sharing under conditioning, and it is the same axis (though not the same cell) our grouping sweep explores.**

**Honest limit:** the diagnosis is partly confounded by compute. Fewer parallel environments due to MoE memory cost is an *optimisation-budget* explanation, not an *architectural* one, and the paper says so. Cite the observation; do not over-cite the mechanism.

#### 8.6.5 What the paper does *not* report

For completeness, because these are the gaps a reviewer would ask about:

- **No numbers for the FiLM ablation.** Fig. 7 is heatmaps and curves only. There is no table of tracking errors, no seed count, no confidence interval anywhere in the paper.
- **No parameter count for the modulation path**, hence no cost/benefit statement.
- **No actor-only vs. critic-only ablation.** `No-FiLM` removes modulation from **both** networks simultaneously. So the paper establishes that "modulating both beats modulating neither" — it does **not** establish that modulating the critic helps, which is precisely the question §8.4's reasoning raises. **The corpus still has no direct evidence on critic modulation in isolation.**
- **No initialisation or stability analysis** of the modulation path.
- **No learning curves** — only final-policy evaluation.

### 8.7 Relevance to this project

**The single most relevant paper in the corpus for architecture, and the source of the most concrete actionable proposal.**

- **Architectural match is close, and unique.** Dense MLP, per-unit FiLM, pre-activation, **no normalisation layer in the modulation path**, on-policy PPO, single task, small networks. Every other modulation paper here is either adaLN-inside-a-transformer (FLOWER, GR00T, CogVLA), or is not scale-and-shift modulation at all and therefore out of scope here (HyperMARL, DyMoDreamer — [§17](#17-appendix--adjacent-evidence-from-the-six-unreviewed-pdfs)). **When we need one citation for "this class of architecture works", it is this one.**
- **It is a working precedent for a *single-task, within-episode* conditioner.** Combined with Yuan 2024 ([§6](#6-yuan-2024--unpacking-the-individual-components-of-diffusion-policy)) — and, outside the reviewed scope, DyMoDreamer — open question 4 is now answered affirmatively three times over. Our single-task setting is not the problem.
- **The conditioner specification is where we differ, and the fix is specifiable.** PAPL's conditioner is **2-dimensional, exogenous, smooth, and cyclic**. Ours is the **full raw observation, endogenous, high-dimensional, and non-cyclic**. Every paper in this corpus that succeeds with affine modulation uses a low-dimensional conditioner. The project's own hypothesised phase variable — *uninjured vs. injured / hypervigilant* — is exactly the right shape. **A concrete proposal: replace the modulator's raw-observation input with a low-dimensional interoceptive-phase encoding** (a scalar or a 2-D embedding), keeping the injection machinery unchanged. This requires a `senior-developer` issue plan, and it should be paired with PAPL's period-randomisation trick so the network cannot learn an open-loop schedule.
- **The critic question now has a testable criterion, and we currently fail it.** PAPL modulates the critic because its reward is phase-conditioned, making the value function genuinely phase-dependent — and it verified this empirically (phase-separated critic-value histograms) *before* committing to the design. **Our reward is not a function of the modulator's input.** So the PAPL argument does not license our critic modulation. The corresponding diagnostic is cheap and we are not running it: **plot the distribution of critic values conditioned on the candidate phase variable.** If it does not separate, the critic has nothing to gain from being told the phase.
- **The MoE-vs-FiLM result supports our sharing direction.** Two papers now report that maximally-isolated conditional computation is *less* stable than maximally-shared affine modulation. Our grouping sweep pushes further along the sharing axis than either. That is not a precedent for grouping (open question 1 remains unanswered), but it is a consistent directional signal — the field's failures are on the *isolation* side, not the *sharing* side.
- **The absence of any normalisation in PAPL's modulation path is the corpus's most interesting silence for us.** PAPL is structurally capable of gain blow-up and does not report it — but it also uses a smooth 2-D cyclic conditioner whose reachable modulation set is a closed 1-D loop. **Our conditioner's reachable set is unconstrained.** That contrast is the sharpest available hypothesis for why our gain variance ballooned and theirs (presumably) did not, and it is testable: constrain the modulator's output to a low-dimensional manifold and see whether the crash disappears.

### 8.8 Appendix: Section-by-Section Backbone

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | Skateboarding is hard for legged robots due to perception-driven interactions and multi-modal objectives across phases. PAPL integrates phase-conditioned FiLM layers into **actor and critic** networks for a unified policy that captures phase-dependent behaviours while sharing robot-specific knowledge. Simulation validation, ablations, efficiency comparison, real-world transfer. |
| I | **Introduction** | Legged locomotion is robust but energy-inefficient; skateboards are non-motorised, low-rolling-resistance, and steer by weight shift (no manipulation needed). Prior work: model-based (motion library + linearised tracking, brittle to slippage) and RL (assumed foot–board attachment, restricting to gliding rather than steering). PAPL relaxes the attachment assumption and adds a **belly-mounted RGB camera** (Fig. 1) — claimed as one of the first uses of onboard exteroception for quadruped skateboarding. Two-stage asymmetric privileged learning + distillation. |
| II | Variable Notation | Frames $\mathcal{B}$ (body), $\mathcal{S}$ (skateboard), $\mathcal{D}$ (deck), $\mathcal{W}$ (world); 12 actuated joints; foot index sets $\mathcal{F}_{\text{all}}, \mathcal{F}_{\text{left}}, \mathcal{F}_{\text{right}}$; deck roll $\psi_D$, axle yaws $\delta_F, \delta_R$. |
| III | **Skateboard Dynamics Modeling** | — |
| III-A | Steering Dynamics | Eq. 1: roll-to-yaw coupling $\delta_i = \arctan(\gamma^1_i\sin(\gamma^2_i\psi_D))$; PD controllers emulate the passive truck; deck roll dynamics from bushing stiffness/damping and rider-induced torques. |
| III-B | Propulsion Dynamics | Physics engines model wheel–ground contact poorly, so effective wheel torques are computed explicitly: $\tau^{\text{net}}_{i,j} = \tau^{\text{ext}}_i - \tau^{\text{fric}}_{i,j}$, with $\tau^{\text{ext}}_i = \tfrac{r}{2}f^{\text{drive}}_i$ from projecting deck force onto the axle direction $\hat d^S_i$, and a two-regime friction torque (static below $\epsilon$, viscous + rolling resistance above). |
| IV | **Skateboard-Riding Policy Learning** | — |
| IV-A | Formulation | Phase clock $\phi_t$, mode map $M(\phi_t)$, phase-conditioned POMDP, Eq. 2 objective. Grid-adaptive command scheduling expands $P(c)$ with policy proficiency; $T_\phi \sim \mathcal{U}(4,12)$ s. Two-stage privileged learning: asymmetric PPO then DAgger distillation of three estimators. |
| IV-B | **Phase-Aware Policy Composition** | **The FiLM section.** Fig. 5 compares standard MLP / FiLM-MLP / MoE-MLP. Modulation equation with $\Phi_t = (\cos\phi_t,\sin\phi_t)$. Actor outputs joint displacements $\Delta q$ from nominal posture $q_0$, converted to torques by joint PD. Input tuple defined (**including $\Phi_t$ inside $o$**). Table I domain randomisation ranges (payload, CoM shift, friction, restitution, PD gains, deck mass, truck/bushing PD). Table II network architectures. Eq. 3 DAgger loss. Table III reward composition: mode-dependent $r_1$–$r_7$ with **different formulations per mode**; mode-independent $r_8$–$r_{11}$ (alignment, smoothness, stabilisation, safety). TRANSITION has no dedicated rewards; mounting/dismounting emerge. |
| IV-C | Implementation Details | Isaac Gym, **4096 parallel environments**, Unitree Go1, board $0.69\times0.27\times0.13$ m with 0.43 m wheelbase. Steering params $\gamma^1_F=1.0,\gamma^2_F=1.12,\gamma^1_R=0.7,\gamma^2_R=0.9$ (10° deck roll → 11.0° front / 6.2° rear axle yaw). Random impulses every 3 s, command resample every 5 s. Diagonal-Gaussian stochastic policy with learnable $\theta_{\text{std}}\in\mathbb{R}^{12}$. **~10 h on an RTX 4090 + i9-9900K, plus 5 h distillation.** |
| V | **Experimental Results** | — |
| V-A | **Multi-Modality of Skateboarding Motions** | Fig. 6: t-SNE of actions and proprioceptive observations over 80 s shows PUSHING and CARVING in separated clusters with TRANSITION between; **critic-value histograms show distinct mode-dependent distributions.** The empirical justification for the whole design, including critic modulation. |
| V-B | Evaluation of Skateboarding Performance | Fig. 7: tracking-error heatmaps + command-area curves for PAPL, **No-FiLM**, No-Extero, No-Priv, and MoE. Error metric $\lvert c_v - v^S_{D,x}\rvert\mathbb{1}_{\text{PUSHING}} + \lvert c_\omega - \omega^S_{D,z}\rvert\mathbb{1}_{\text{CARVING}}$, swept at 0.05 resolution over $c_v\in[0,1.5]$, $c_\omega\in[-0.5,0.5]$, ten environments per command, 40 s horizon, perturbations every 4 s. Overturns or >0.5 m deviation counted as constraint violations. **No-FiLM: narrower region, higher errors. MoE: pushing expert collapsed. No-Priv: converged to standing still.** Fig. 8: extrinsic estimator accuracy. |
| V-C | Power Consumption Analysis | Fig. 9: normalised motor power over a 30 m traversal vs legged and wheel-legged baselines at 1.5 m/s. Skateboarding is cheaper, with a long-tailed distribution (cheap steering, expensive pushing/mounting). Phase clock was strategically halted mid-carve when speed exceeded 0.7 m/s. |
| V-D | Real-World Experiments | Fig. 10: zero-shot sim-to-real on a Unitree Go1 with a physical skateboard; pushing / transition / carving reproduced; trials under external perturbation, low light, and uneven sidewalks without collapse. |
| VI | **Conclusion** | Recaps the four ingredients (phase-conditioned RL formulation, FiLM layer modulation, asymmetric privileged learning, exteroceptive sensing). Future work: a high-level navigation policy that generates velocity commands **and modulates the phase clock** in response to the environment, plus front-facing perception. |

---

## 9. Zhu et al. 2025 — *EquAct: An SE(3)-Equivariant Multi-Task Transformer for Open-Loop Robotic Manipulation*

**PDF:** `docs/project/references/modulation_in_rl/sources/Zhu et al. 2025 - EquAct - SE(3) equivariant FiLM.pdf`
**arXiv:** [2505.21351v1](https://arxiv.org/abs/2505.21351), 27 May 2025. Xupeng Zhu, Yu Qi\*, Yizhe Zhu\*, Robin Walters†, Robert Platt† (Northeastern University).

> **This is the most consequential paper in the corpus for open question 1.** Its `iFiLM` layer **shares a single gain across a contiguous block of feature components** — the first instance of grouped modulation found in any refereed FiLM-family paper across this reference library. §9.3 states the evidence and, equally importantly, the limits on how it can be cited.

### 9.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **Unstated in PDF — the PDF says "Preprint. Under review."** (page-1 footnote), with the stamp `arXiv:2505.21351v1 [cs.RO] 27 May 2025`. The motivating survey recorded "ICLR 2026 Poster" from an **OpenReview proceedings record** ([forum d1wuA8oIH0](https://openreview.net/forum?id=d1wuA8oIH0)) under a slightly different title. That is credible external evidence, but **this held PDF is the earlier preprint version and does not state it**. Cite the venue only with that provenance attached. |
| **Conditioning signal** | The **natural-language instruction embedding** $k$ — a CLIP tokenizer feeding a Transformer encoder — **treated as a type-0 (rotation-invariant scalar) feature**. Its invariance is not incidental: it is the load-bearing assumption (§9.5.1). |
| **Modulated component** | The **SE(3)-Equivariant Point Transformer U-Net (EPTU)** — the observation encoder / policy trunk that turns the point cloud into latent spherical features $h$. **Not** the downstream field networks that score actions. |
| **Granularity cell** | **(c) grouped — genuinely, and this is the corpus's only instance.** One scalar gain $\alpha_l$ multiplies an entire type-$l$ spherical-Fourier block $c_l = [c_l^{-l}, \dots, c_l^{l}] \in \mathbb{R}^{2l+1}$. With the paper's $L_{\max}=3$, the block sizes are **1, 3, 5, 7** components. On the invariant $l=0$ channels the modulation degenerates to standard per-unit FiLM (block size 1) and is the *only* place an additive offset survives. See §9.3. |
| **Placement** | Inside EPTU. Fig. 3(b) presents `iFiLM` as one of three EPTU module types (alongside spherical-Fourier maxpool `smaxpool` and upsample `sup`), interleaved with EquiformerV2 graph-attention blocks across the U-Net's down- and up-sampling stages with skip connections. **The exact number of injection sites is not enumerated in the PDF.** |
| **Parameterization** | Eq. 7: $\alpha_l, \beta, \gamma = \mathrm{MLP}(k)$ — a multi-layer perceptron projects the language embedding into the modulation parameters. **Whether one MLP is shared across sites or each site has its own is not stated.** No parameter count for the modulation path is given anywhere. |
| **Initialization** | **Not stated anywhere.** No zero-init, no identity-init, no claim that initialisation matters. |
| **Ablation present?** | **Yes — the modulation mechanism itself is ablated against vanilla FiLM.** Table 3, `iFiLM → FiLM`, 10-demo setting, 4 RLBench tasks: **average 52.8 → 50.3**. Per-task: place\_wine 45→**68**, place\_cups **62**→24, drag 90→90, insert\_peg 14→**19**. Read §9.4 before quoting: the average difference is driven by a single task and the arithmetic cuts both ways. |
| **Reported instability** | **None from modulation.** The single stability-adjacent remark concerns the equivariant backbone, not the gains: *"we hypothesize that data augmentation reduces numerical error in the equivariant neural networks."* No NaN, no gain blow-up, no entropy or plasticity discussion. Limitations name only keyframe-formulation scope, absence of pretrained vision models, and inference speed. |
| **Single-task or multi-task** | **Multi-task** — one unified model over 18 RLBench tasks with **249 language variations**, plus 4 physical tasks with 11 variations. The conditioner exists *because* of task diversity. But note the headline margins grow with **geometric** difficulty, not task count: EquAct beats baselines by 2.6 % with 100 demos, 6.2 % with 10 demos, and **15.4 % in the 10-demo SE(3) setting**. The moderator is pose randomisation, not instruction variety. |

### 9.2 Verification of the survey's claims

| Survey claim | Verdict | Evidence |
|---|---|---|
| Venue ICLR 2026 Poster | **Not verifiable from this PDF** | PDF says "Preprint. Under review."; survey's source was an OpenReview record under an earlier title |
| Conditioning signal = language instruction embedding | **Confirmed** | §4, step 2: CLIP tokenizer + Transformer encoder, treated as type-0 |
| Modulates the equivariant point-cloud U-Net trunk | **Confirmed** | §4.3, Fig. 3(b) |
| Granularity "(b), restricted to symmetry-invariant (type-0) features" | **CORRECTED — this is the survey's one substantive mechanism error** | The restriction is *not* "modulation applies only to type-0 features". Eq. 8 modulates **every** type $l>0$ as well; what is restricted is the *form*: a **shared scalar gain with no offset** on each $l>0$ block, and full affine only on $l=0$. So modulation reaches the whole feature space, in two different algebraic forms. See §9.3. |
| iFiLM makes $(\gamma,\beta)$ SE(3)-invariant so modulation commutes with 3-D rotation/translation | **Confirmed** | Proposition 4.4 and Proof A.4 |
| "vanilla FiLM destroys geometric equivariance when injected into an equivariant backbone" | **Confirmed as the paper's motivation**, though the paper states it as a guarantee gap rather than proving destruction: *"Unlike standard FiLM layers, which do not guarantee equivariance or invariance, the iFiLM layer is provably SE(3) invariant"* | §4.3 |
| SOTA on 18 RLBench under SE(3)/SE(2) perturbation + 4 physical tasks | **Confirmed** | Table 1 (53.3 / 60.1 / 89.4 average across the three settings); Table 2 (65.0 % vs 3DDA's 12.5 % on physical) |

### 9.3 Open question 1 — grouped modulation, found, with the caveat that decides how we cite it

**The claim, stated precisely.** In `iFiLM`, **one scalar multiplies an entire contiguous block of $2l+1$ feature components**, for every spherical-harmonic degree $l > 0$. That is Axis-1 cell **(c)**, grouped, with group size $2l+1$.

**The evidence, quoted.** §4.3 (PDF p. 6), Eqs. 7–9 verbatim:

$$
c' = \mathrm{iFiLM}(c, k), \qquad \alpha_l, \beta, \gamma = \mathrm{MLP}(k) \tag{7}
$$

$$
c'_l = \alpha_l\, c_l, \qquad \text{for } l > 0 \tag{8}
$$

$$
c'_0 = \beta\, c_0 + \gamma, \qquad \text{for } l = 0 \tag{9}
$$

and the accompanying prose: *"iFiLM first uses a multi-layer perceptron to project the condition $k$ into type-0 modulation scales $\alpha, \beta$ and bias $\gamma$. Then, iFiLM **scales the type-$l$ input feature $c_l$ by $\alpha_l$** for all $l > 0$, and applies an affine transformation to the type-0 features using $\beta$ and $\gamma$."*

**What the block is.** §2 (Background, "Spherical harmonics") defines the object being scaled: a type-$l$ spherical Fourier coefficient is the $(2l+1)$-dimensional vector

$$
c_l = \bigl[c^{-l}_l,\, c^{-l+1}_l,\, \dots,\, c^{l}_l\bigr],
$$

indexed by order $m$ with $-l \le m \le l$. Equation 8 multiplies **all $2l+1$ of these components by the same $\alpha_l$**. With the paper's $L_{\max} = 3$ (established by the `l = 3 → 2` ablation), the modulation therefore partitions each feature into blocks of size **1** ($l=0$), **3** ($l=1$, a 3-D vector), **5** ($l=2$), and **7** ($l=3$) — each block receiving exactly one gain.

**Why the grouping is forced — Schur's lemma.** This is the part that determines how the result may be used. Proposition 4.4 asserts iFiLM is SO(3)-invariant in $k$ and SO(3)-equivariant in $c$, i.e. $D(r)\cdot c' = \mathrm{iFiLM}(D(r)\cdot c,\, k)$. Proof A.4 (PDF p. 16) is two lines. For $l = 0$ the Wigner D-matrix is the identity, so the full affine passes through (Eq. 25). For $l>0$ (Eq. 26):

$$
\alpha_l\bigl(D(r)\cdot c_l\bigr) = D(r)\cdot(\alpha_l c_l) = D(r)\cdot c'_l,
$$

*"applying Schur's lemma"*. The underlying algebra: $D_l$ is an **irreducible** representation of $SO(3)$ on $\mathbb{R}^{2l+1}$, and Schur's lemma says any linear map $M$ commuting with an irreducible representation is a scalar multiple of the identity, $M = \alpha I$. A per-component gain $\mathrm{diag}(\gamma^{-l},\dots,\gamma^{l})$ commutes with every $D_l(r)$ **only if all its entries are equal**. Likewise an additive offset on an $l>0$ block would break equivariance, since $D(r)\cdot(c_l + \gamma_l) \ne D(r)\cdot c_l + \gamma_l$ in general — which is exactly why Eq. 8 has **no bias term** and Eq. 9 does.

So: **equivariance ⟹ the modulation must be constant within each irreducible block ⟹ grouped, with group size $2l+1$, and gain-only above $l=0$.**

**Distinguishing this from FLOWER, precisely.** The two are on different axes and must not be conflated:

| | What is shared | Across what | Axis | What stays fine-grained |
|---|---|---|---|---|
| **FLOWER** Global-AdaLN-Zero ([§7](#7-reuss-et-al-2025--flower-democratizing-generalist-robot-policies-with-efficient-vision-language-action-flow-policies)) | the generator's **weight set** | the 18 transformer **layers** | **Axis 3** (parameterisation across sites) | within any layer, modulation is still **per-channel** — every channel gets its own $(\gamma,\beta)$ |
| **EquAct** iFiLM | a single **scalar gain value** | the $2l+1$ **components of one irrep block** | **Axis 1c** (granularity across units) | each block, at each site, still gets its **own** scalar |

FLOWER ties *sites*; EquAct ties *units*. Our project's `grouping_size` sweep ties units. **EquAct is therefore the first genuine on-axis precedent, and FLOWER remains off-axis** — which upholds, rather than overturns, the caution recorded in [`film_rl_recent_variants_survey.md` §5.3](../FiLM/film_rl_recent_variants_survey.md).

**The caveat that decides the citation — this is a by-product, not a designed regulariser.** State this plainly whenever the paper is cited:

- The grouping is **imposed by a symmetry constraint**, not chosen. The paper's stated purpose for iFiLM is *"to enforce the geometric invariance of natural language conditioning in the policy"* — full stop. Grouping is the algebraic consequence.
- The paper **never frames it as coarsening**. The words *grouped*, *granularity*, *coarse*, *block*, *shared across channels*, and *parameter efficiency* appear nowhere in connection with iFiLM. No parameter saving is claimed or measured.
- The group boundaries are **not a free hyperparameter**. They are fixed at $2l+1$ by representation theory. There is no analogue of our $G$ knob, and no sweep over group size exists or could exist without breaking the guarantee the layer was built to provide.
- Consequently the ablation in §9.4 does **not** test "grouped versus per-unit modulation" as a design question. It tests "equivariance-preserving conditioning versus equivariance-breaking conditioning", in which grouping is one of two simultaneous changes (the other being the removal of the offset on $l>0$ blocks).

**Recommended citation form:** *"Grouped modulation — one gain shared across a contiguous block of units — appears in the refereed conditional-architecture literature exactly once, in EquAct's iFiLM layer (Zhu et al. 2025, §4.3), where a single scalar $\alpha_l$ scales an entire $(2l+1)$-dimensional irreducible-representation block. There it is a **consequence of Schur's lemma under an SE(3)-equivariance constraint** rather than a deliberate coarsening, and its group boundaries are fixed by representation theory rather than chosen. It is therefore an existence proof that grouped modulation can be trained to state-of-the-art performance, not evidence about grouping as a design axis."*

That is a real strengthening of our position — *"unattested anywhere"* becomes *"attested once, for structural reasons"* — and it is honest about what it does not establish.

### 9.4 The ablation, read honestly

Table 3 (10-demo setting, 4 RLBench tasks, **no seed count and no error bars stated anywhere**):

| Variant | avg. SR | place wine | place cups | drag | insert peg |
|---|---|---|---|---|---|
| **Ours (full EquAct)** | **52.8** | 45 | **62** | 90 | 14 |
| `aug. → no aug.` | 50.5 | 36 | 71 | 85 | 10 |
| **`iFiLM → FiLM`** | **50.3** | **68** | 24 | 90 | **19** |
| `l = 3 → 2` | 45.5 | 64 | 28 | 80 | 10 |
| `equ. → no equ.` | 12.3 | 14 | 0 | 35 | 0 |

**What the paper concludes:** *"while `iFiLM → FiLM` performs similarly to the full model in most cases, it shows a notable performance degradation on precision-demanding tasks such as 'place\_cups', demonstrating the generalization advantage of the proposed iFiLM layers."*

**What the arithmetic actually shows.** The 2.5-point average advantage is not a broad win — it is **one task carrying the result against two losses**:

- place\_cups: iFiLM **+38** (62 vs 24) — the entire margin, and more
- place\_wine: vanilla FiLM **+23** (68 vs 45)
- insert\_peg: vanilla FiLM **+5** (19 vs 14)
- drag: exact tie (90 vs 90)

With four tasks, no seeds, and no confidence intervals, a $+38 / -23 / -5 / 0$ split is not a clean demonstration. **Cite the mechanism and the existence proof; do not cite the 52.8-vs-50.3 gap as evidence that grouped modulation outperforms per-unit modulation.** It does not carry that weight.

**The comparison that *is* decisive in this table is a different one.** `equ. → no equ.` — replacing **a single** equivariant graph-attention layer in the field networks with a Roformer layer — collapses the model from 52.8 to **12.3**, with two tasks going to zero. The paper's reading is right: *"Even though only a single equivariant layer is replaced, `equ. → no equ.` results in the largest performance drop, underscoring the critical role of maintaining geometric structure."* **Equivariance is load-bearing; the specific form of the conditioning layer is second-order.** That ordering matters for us, because it means iFiltM's grouped structure is being carried along by a much larger effect, not isolated by this experiment.

### 9.5 Phase 1 — Foundational overview

**Introduction.** A robot arm is told, in plain English, to "put the ring on the orange spoke", and has to work out where to move its gripper. Modern systems do this with a transformer reading a 3-D point cloud plus the sentence. The trouble is that these networks have no built-in notion that **space behaves consistently**: if you pick up the whole scene and rotate it 40 degrees, the correct gripper pose rotates by exactly the same 40 degrees — but a standard transformer does not know that, and has to learn it, badly and expensively, from examples.

EquAct builds that knowledge into the architecture. It is **equivariant** to 3-D rotations and translations: rotate the scene, and the predicted action rotates with it, guaranteed by construction rather than by training. This turns out to be worth a great deal — in the hardest setting tested, where objects are placed at arbitrary 3-D orientations and only 10 demonstrations per task are available, it beats the previous best method by more than 15 percentage points, and on real hardware it succeeds 65 % of the time where the baseline manages 12.5 %.

**Where the steering comes in.** The language instruction has to reach this network somehow, and the natural tool is FiLM — generate a gain and an offset from the sentence and apply them to the network's features. But that breaks the guarantee. The instruction "put the ring on the spoke" means the same thing no matter which way the table is facing, so it must not itself impose a direction on the features. Ordinary FiLM does exactly that.

The fix, **iFiLM**, is disarmingly small. The network's features come in bundles that represent geometric quantities: single numbers, 3-D vectors, and higher-order shapes. iFiLM multiplies each *whole bundle* by one number, and adds an offset only to the bundles that are single numbers. That is all. Scaling a 3-D vector by a number keeps it pointing where it pointed; adding something to it does not. So a rule about *what the language may do to the features* falls straight out of *what geometry allows*.

**Key findings.** State of the art across 18 simulated tasks under both 2-D and 3-D scene randomisation, and across 4 physical tasks, at a training and inference cost comparable to the baselines. The ablations show that geometric structure is what carries the result: breaking equivariance at a *single* layer collapses average success from 52.8 % to 12.3 %.

**Initial takeaway.** A constraint on the conditioning signal can be *derived* rather than tuned. And a side effect worth noticing for its own sake: once you insist that the steering respect geometry, you are forced to give **one gain to a whole block of features** instead of one per feature — the coarser scheme is not a compromise, it is the only one the mathematics permits.

### 9.6 Phase 2 — Graduate-level deep dive

#### 9.6.1 The two symmetry assumptions

EquAct predicts a keyframe action $a$ from an observation $o = \{s, e\}$ (point cloud $s$, gripper state $e = \{e_T, e_{\text{open}}\}$) and a language instruction $n$. Its structural claim (Eq. 2) is

$$
\pi(g\cdot o,\, n) = g\cdot a, \qquad g \in SE(3) = SO(3) \ltimes T(3).
$$

Read carefully, this couples **two distinct** statements:

1. **Equivariance in the observation** — transform the scene, and the action transforms identically. Standard in the equivariant-policy literature.
2. **Invariance in the language** — for a *fixed* instruction, the action's transformation depends **solely** on the observation's transformation. The instruction contributes semantics, never geometry.

The paper flags (2) as its own identification: *"our work is the first to explicitly identify the SE(3) invariance of natural language instructions in the context of robotic policies."* Proposition 4.1 establishes both by induction over layers — every layer must individually satisfy the property, which is precisely why the conditioning layer needs redesigning rather than reusing.

**Why (2) forces the iFiLM form and nothing weaker.** Suppose modulation were an unrestricted per-component affine $c^m_l \mapsto \gamma^m_l(k)\, c^m_l + \beta^m_l(k)$. Under a rotation $r$, the input transforms as $c^{n\prime}_l = \sum_m D^l_{mn}(r)\, c^m_l$. For the layer to commute with this, the diagonal matrix $\Gamma_l = \mathrm{diag}(\gamma^{-l}_l, \dots, \gamma^{l}_l)$ would have to satisfy $\Gamma_l D_l(r) = D_l(r) \Gamma_l$ for **all** $r \in SO(3)$. Since $D_l$ is irreducible, Schur's lemma forces $\Gamma_l = \alpha_l I$. And the offset must satisfy $\beta_l = D_l(r)\beta_l$ for all $r$, whose only solution on a non-trivial irrep is $\beta_l = 0$. **Both restrictions in Eqs. 8–9 are thus not merely sufficient but necessary**: the gain must be block-constant, and the offset must vanish except on $l = 0$, where $D_0 \equiv 1$ makes both constraints vacuous.

This is a stronger statement than the paper makes — it proves only sufficiency (Prop. 4.4) — but it follows immediately from the same lemma and is worth recording, because it means **iFiLM is the unique maximal equivariance-preserving FiLM**, not one design among several.

#### 9.6.2 What the modulation can and cannot express

The consequence for representational power is concrete. Write the modulated feature at one site as a block-diagonal operator:

$$
c' = \underbrace{\begin{pmatrix} \beta & & & \\ & \alpha_1 I_3 & & \\ & & \alpha_2 I_5 & \\ & & & \alpha_3 I_7 \end{pmatrix}}_{\text{block-constant, } L_{\max}=3} c \;+\; \begin{pmatrix} \gamma \\ 0 \\ 0 \\ 0\end{pmatrix}.
$$

So for a single channel with $L_{\max}=3$ the language controls **4 gains and 1 offset over a 16-dimensional feature** ($1+3+5+7$), rather than 16 gains and 16 offsets. The reachable set of modulations is a 5-parameter family instead of a 32-parameter one — a **6.4× reduction in modulation degrees of freedom per channel**, achieved with no loss of state-of-the-art performance.

**What survives.** The language can still re-weight *how much* each geometric order contributes: suppress the scalar channels and amplify the $l=1$ vector channels, or vice versa. Since $l$ indexes angular resolution — $l=0$ is direction-blind magnitude, $l=1$ is direction, higher $l$ is finer angular detail — the instruction is effectively choosing **which angular frequency bands the policy attends to**. That is a meaningful and interpretable control channel.

**What is lost.** The language cannot rotate features, cannot single out a particular spatial direction, and cannot translate them. Which is exactly the point: those are the operations that would let a sentence smuggle a coordinate frame into a network that is supposed not to have one.

#### 9.6.3 The surrounding architecture, in brief

iFiLM sits inside the **Equivariant Point Transformer U-Net (EPTU)**, whose other two novel modules are also equivariance-preserving reformulations of standard operations, and whose proofs follow the same pattern:

**Spherical Fourier maxpooling** (Eq. 3) — select, per degree $l$, the neighbour whose coefficient block has the largest norm:

$$
c'_{l,x} = \mathrm{smaxpool}\{c_{l,p} \mid p \in \mathrm{knn}(x)\} = c_{l,p^*}, \qquad p^* = \arg\max_{p\in\mathrm{knn}(x)} \lVert c_{l,p}\rVert^2_2 .
$$

Equivariant (Prop. 4.2) because Wigner D-matrices are orthogonal, so $\lVert D(r) c_l\rVert_2 = \lVert c_l\rVert_2$ — the **argmax is rotation-invariant**, hence the selection commutes with rotation. Note the structural parallel with iFiLM: the operation is forced to act on the block **as a whole** (a norm over the block, not per component), for the same representation-theoretic reason.

**Spherical Fourier upsampling** (Eq. 5) — inverse-distance-weighted interpolation applied coefficient-wise:

$$
c'_{l,x} = \mathrm{sup}\{c_{l,p}, x \mid p\in \mathrm{knn}(x)\} = \mathrm{softmax}_{p\in\mathrm{knn}}\!\left(\frac{1}{\lVert x-p\rVert}\right) c_{l,p}.
$$

Equivariant (Prop. 4.3) because the weights depend only on distances, which are $SE(3)$-invariant, and the weighted sum is linear — so $D(r)$ passes through.

**Equivariant field networks** evaluate action values at arbitrary query poses rather than only at cloud points. Translation values $q_t$ take only the **type-0** part of the aggregated feature, since translational action value must be rotation-*invariant*. Rotation values use a spherical CNN at the predicted translation, $q_r(a_r, a_t, h) = (\phi \star \psi)[a_r] = \mathcal{F}^{-1}(\hat\phi\cdot\hat\psi)[a_r]$ — evaluating the whole $SO(3)$ action distribution in **one shot** via a Fourier-domain product, which is where the single-forward-pass efficiency comes from (0.7 s inference vs 3DDA's 3.7 s).

**Training** is classification, not regression:

$$
\mathcal{L} = \mathbb{E}_{(o,n,\bar a)\sim\mathcal{D},\, a\sim\mathcal{A}}\Bigl[H\bigl(Q_a(o,n,a),\, \bar a\bigr)\Bigr] = \mathbb{E}\Bigl[H\bigl(\textstyle\sum_{i=1}^{3} Q_t(a^i_t, o, n), \bar a_t\bigr) + H\bigl(Q_r, \bar a_r\bigr) + H\bigl(Q_{\text{open}}, \bar a_{\text{open}}\bigr)\Bigr]
$$

with $H$ cross-entropy — *"the goal is the policy to correctly choose the expert action from among all available actions"* — plus explicit rotation augmentation of $[\pm5°, \pm5°, \pm45°]$ about $[x,y,z]$.

**Note for open question 5.** There is **no actor/critic split** here. This is imitation learning with a single implicit action-value function $Q_a(o,n,a)$ that *is* the policy. Modulation is applied in the **shared trunk upstream of all three value heads** ($Q_t$, $Q_r$, $Q_{\text{open}}$). EquAct therefore contributes an architectural data point but no actor-versus-critic evidence.

#### 9.6.4 Results, as printed

Table 1, multi-task success rate (%) on 18 RLBench, three settings — `100` = 100 demos with SE(2) object poses, `10` = 10 demos SE(2), `10*` = 10 demos with **SE(3)** object poses:

| Method | 10\* (SE(3)) | 10 | 100 | Train (h) | Infer (s) | Mem (GB) |
|---|---|---|---|---|---|---|
| **EquAct** | **53.3** | **60.1** | **89.4** | 240 | 0.7 | 21 |
| SAM2ACT | 37.0 | 52.2 | 86.8 | 225 | 0.1 | 21 |
| 3DDA (3D Diffuser Actor) | 37.9 | 50.3 | 81.3 | 253 | 3.7 | 20 |

Margins: **+2.6 %** (100 demos), **+6.2 %** (10 demos), **+15.4 %** (10 demos, SE(3)). *"the more difficult the setting is, the more EquAct outperforms the baselines."* Largest per-task wins are on precision tasks — place\_cups (21/62/76 vs SAM2ACT's 0/4/47) and stack\_cups (59/18/68 vs 0/12/78). The paper is candid about its failure mode: *"EquAct underperforms baselines in the tasks in which the object's pose is fixed, e.g. 'sweep\_to\_dustpan'"* — where SE(3) equivariance buys nothing and costs capacity.

Table 2, 4 physical tasks on a UR5 with Robotiq 2F-85 gripper and three RealSense D455 cameras, 11 variations / 135 demonstrations, 10 episodes each: **EquAct 65.0 %** average (disassemble pipe 90, pluck flower 70, pick fruit 50, install toilet roll 50) vs **3DDA 12.5 %** (0, 20, 30, 0).

### 9.7 Relevance to this project

**Bears on our design in exactly one deep way, and in several shallow ways not at all. The deep one is worth a great deal.**

- **It converts "unattested" into "attested once, for structural reasons" on our headline design axis.** Both prior syntheses concluded that no paper in this library shares one $(\gamma,\beta)$ across a block of units. **That conclusion is now superseded** — [`film_modulation_granularity_synthesis.md` §3.3 and §7.3](../FiLM/film_modulation_granularity_synthesis.md) and [`film_rl_recent_variants_survey.md` §5.3](../FiLM/film_rl_recent_variants_survey.md) should both be annotated. Our grouping screen is no longer describable as "unprecedented in the conditional-architecture literature".
- **But it does not license the claim we would most like to make.** EquAct's grouping is **forced by Schur's lemma**, not chosen as a regulariser; its group boundaries are fixed at $2l+1$ by representation theory rather than swept; and it is never framed as coarsening or parameter saving. **We cannot cite it as evidence that grouping helps, or that modulation capacity is over-provisioned.** What we *can* cite it for is the existence claim: a state-of-the-art policy in which one gain governs a block of units, trained without incident. Paired with the honest statement that the two systems arrive at the same structure for entirely different reasons, that is a genuinely useful citation and an interesting one — but it must be presented that way, or a referee who reads EquAct will (rightly) say we over-claimed.
- **It supplies a *principled* answer to "where should group boundaries go?", which our design currently answers arbitrarily.** Our blocks are **contiguous runs of $\lceil 128/G\rceil$ units in an arbitrary ordering** — the partition carries no meaning, so grouping units 0–7 together is as good or bad as any other assignment. EquAct's blocks are **irreducible representations**: units inside a block are genuinely interchangeable under the symmetry, so sharing a gain across them costs nothing that the task could have used. **That is the sharpest critique of our grouping design available in the corpus, and it comes from the one paper that shares our structure.** It suggests a concrete alternative: if a meaningful partition of the 128 hidden units exists — by modality (interoceptive vs. exteroceptive), by learned functional cluster, or by any equivalence the task actually respects — grouping along *that* partition has a rationale, whereas grouping along index order does not. **This is a research direction, not a config change; it would need a `senior-developer` issue plan.**
- **A cheap diagnostic falls out of this.** If our modulator's per-neuron gains, at convergence, are near-constant within some partition of the 128 units, then a group structure exists in the data and $G$ should be set to respect it. If they are not, contiguous-index grouping is destroying information and the $G$ sweep is measuring damage rather than regularisation. **Clustering the learned per-neuron gain vectors across states is a post-hoc analysis of runs we already have.**
- **The gain-only-above-$l{=}0$ detail is a second, smaller idea.** iFiLM drops the additive offset wherever it would break structure, keeping it only on invariant channels. We currently apply both $\gamma$ and $\beta$ everywhere, uniformly. There is no symmetry in our gridworld to force a similar split — so this does **not** transfer as a constraint — but it is a reminder that gain and offset need not be applied at the same granularity, and our design has never questioned that they should.
- **Nothing here bears on our crash, our conditioner, or our single-task setting.** No instability of any kind is reported. The conditioner is a language embedding — low-dimensional, external, and carrying information the trunk provably lacks, i.e. the opposite of ours on every count. The setting is multi-task imitation learning with no reward, no critic, and no exploration. Open questions 2, 3, 4 and 5 gain nothing from this paper.

### 9.8 Appendix: Section-by-Section Backbone

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | Transformers learn language-conditioned multi-task 3-D manipulation well but lack geometric guarantees, giving unpredictable behaviour under SE(3) scene transformations. EquAct = SE(3)-equivariant multi-task transformer with two components: (1) an efficient equivariant point-cloud U-Net with spherical Fourier features, (2) **SE(3)-invariant FiLM (iFiLM) layers for language conditioning**. Benchmarked on 18 RLBench tasks under SE(3) and SE(2) perturbation and 4 physical tasks. |
| 1 | **Introduction** | Keyframe policies and language instructions both carry 3-D geometric structure that standard transformers ignore, limiting generalisation to novel scene configurations and inflating data requirements. EquAct enforces **equivariance in the policy** and **invariance in the language**. First method achieving continuous SE(3) equivariance (rotation *and* translation) for multi-task keyframe policies, versus prior work enforcing translation only. Introduces 18 RLBench tasks with SE(3) initialisation. Three contributions: architecture, proofs, empirical validation. Limitations stated up front: keyframe-only, no pretrained vision models. |
| 2 | **Background** | Keyframe imitation learning ($\pi(o)=a$, observation $o=\{s,e\}$, motion planner fills in trajectories). Equivariance $f(g\cdot x) = g\cdot f(x)$; $SE(3) = SO(3)\ltimes T(3)$; three routes to equivariance (data augmentation, canonicalisation, equivariant architectures). **Spherical harmonics**: $\hat f_s = \{c^m_l\}$, degree $l$, order $-l\le m\le l$; rotation acts via Wigner D-matrices $c'^n_l = \sum_m D^l_{mn}(g) c^m_l$; type-0 = scalar (identity D-matrix), type-1 = 3-D vector. **Spherical CNNs**: convolution as a Fourier-domain product. |
| 3 | **Related works** | Keyframe/multi-task policies split into multi-view projection methods (RVT, RVT2, SAM2ACT) and direct 3-D methods (Act3D, 3D Diffuser Actor); none combine translation *and* rotation equivariance. Rotation prediction via discretised Euler angles (gimbal lock, discontinuity) or SO(3) diffusion (expensive). Equivariant policy learning: mostly SE(2), mostly single-policy or pick-and-place-only, often discrete approximations. Language + equivariance is nascent; EquAct claims first explicit identification of language's SE(3) invariance. |
| 4 | **Method** | Policy as implicit value $Q_a(o,n,a)$; three inference steps (EPTU encoding → iFiLM language fusion → equivariant field network scoring). Cross-entropy training loss over sampled query actions; rotation augmentation $[\pm5°,\pm5°,\pm45°]$. |
| 4.1 | Equivariance assumptions | Eq. 2 and Fig. 2. **Proposition 4.1** (EquAct is SE(3)-equivariant in observation and SE(3)-invariant in language), proved by induction over layers in App. A.1. |
| 4.2 | Equivariant Point Transformer U-net (EPTU) | Fig. 3. U-Net over EquiformerV2 graph-attention blocks with skip connections. **Spherical Fourier maxpool** (Eq. 3, Prop. 4.2, proved via orthogonality of Wigner D-matrices). **Spherical Fourier upsampling** (Eq. 5, Prop. 4.3, proved via Schur's lemma and linearity). Simpler than prior equivariant point U-Nets — no graph caching required. |
| 4.3 | **Invariant FiLM Layers (iFiLM)** | **The modulation section.** Eqs. 7–9. Contrast with standard FiLM, which *"do[es] not guarantee equivariance or invariance"*. **Proposition 4.4**: SO(3)-invariant in the condition $k$, SO(3)-equivariant in the feature $c$. |
| 4.4 | Equivariant field network | Values evaluated over the whole $SE(3)$ pose space rather than at cloud points. $q_t$ and $q_{\text{open}}$ take type-0 features only (translational value must be rotation-invariant); coarse-to-fine translation refinement. $q_r$ aggregates spherical features at the predicted translation and convolves with a learned filter — full $SO(3)$ distribution in one shot. |
| 5.1 | Simulation experiments | 18 RLBench tasks, Franka Panda, four RGB-D cameras, 2–60 variations per task, **249 variations total**. Binary reward, 25 episodes per task, 25 steps max. Baselines SAM2ACT and 3DDA, all on a single RTX 4090 (24 GB). Table 1 results across the `100` / `10` / `10*` settings. |
| 5.2 | Physical experiments | UR5 + Robotiq 2F-85 + 3× RealSense D455; SpaceMouse teleoperation. 4 tasks, 11 variations, 135 demonstrations; objects SE(3)-randomised except pick\_fruit. Table 2: 65.0 % vs 3DDA's 12.5 %; 3DDA *"often skipping keyframe actions"*. |
| 5.3 | **Ablation study** | Table 3, four ablations on the 10-demo setting: `aug.→no aug.`, **`iFiLM→FiLM`**, `l=3→2`, `equ.→no equ.` Largest drop from breaking equivariance at a single layer (52.8→12.3). High-degree Fourier coefficients matter (`l=3→2` costs 7.3). Data augmentation hypothesised to reduce numerical error in the equivariant network. |
| 6 | **Conclusion and limitations** | Margins of 2.6 / 6.2 / 15.4 % across settings. Limitations: keyframe formulation cannot solve fine-grained closed-loop tasks; no pretrained vision models, so semantic generalisation could improve; slower training and inference than the best baseline. |
| A | **Proofs** | A.1 Prop. 4.1 by induction. A.2 Prop. 4.2 via orthogonality of Wigner D-matrices. A.3 Prop. 4.3 via Schur's lemma and linearity of the SO(3) action. **A.4 Prop. 4.4** — Eq. 25 ($l=0$, identity D-matrix) and **Eq. 26 ($l>0$, Schur's lemma): $\alpha_l(D(r)\cdot c_l) = D(r)\cdot(\alpha_l c_l)$.** |
| B | 18 RLBench tasks with SE(3) initialisation | Table 4: per-task variation type, perturbed object, SO(3) perturbation range, language template. |
| C | Details of 4 physical tasks | Setup (Fig. 6) and per-task specifications: disassemble pipe (3 variations, 3–10 keyframes), pluck flowers (3, 4), pick fruit (3, 3), install toilet roll (2, 5). |
| D | Hyperparameters | Table 5. EquAct sim: 450/3000 query translations train/test, lr $10^{-4}$, no scheduler, batch size 2, $8\times10^5$ training iterations. |

---

## 10. Li et al. 2025 — *CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification*

**PDF:** `docs/project/references/modulation_in_rl/sources/Li et al. 2025 - CogVLA - modulation as routing.pdf`
**arXiv:** [2508.21046v3](https://arxiv.org/abs/2508.21046), v3 dated 27 May 2026. Wei Li, Renshan Zhang, Rui Shao\*, Jie He, Liqiang Nie (Harbin Institute of Technology, Shenzhen). Project page: `jiutian-vl.github.io/CogVLA-page`.

### 10.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **NeurIPS 2025**, printed on the PDF's page-1 footer: *"39th Conference on Neural Information Processing Systems (NeurIPS 2025)."* Verified from the PDF. |
| **Conditioning signal** | **Two different conditioners at two sites.** (i) *Encoder-FiLM*: the **language instruction embedding** $t_r$, obtained from the language model's embedding layer. (ii) *LLM-FiLM*: the **instruction tokens $t_l$ at transformer layer $l$** — i.e. a progressively contextualised version of the same instruction, re-read at each of the LLM's 32 layers. Both are language; neither is the visual observation. |
| **Modulated component** | (i) The **two vision encoders** (SigLIP and DINOv2) — modulation is applied inside their blocks, on the encoder's own self-attention output. (ii) The **visual tokens flowing through the language-model trunk**. Not an actor, not a critic (imitation-learning VLA; no value function exists). |
| **Granularity cell** | **(b) per-channel.** $\gamma_i, \beta_i$ are described as *"scale and shift **vectors**"* combined by $\odot$ (element-wise), broadcast across the token axis — one $(\gamma,\beta)$ per feature channel of the token embedding, shared across all tokens. |
| **Placement** | **Many sites, across two subsystems.** Encoder-FiLM is applied *"through iterative visual encoder blocks"* in each of the two encoder branches. LLM-FiLM is applied at LLM transformer layers indexed $l = 1..L$ with **$L = 32$** (App. A.1). So the modulation spans essentially the whole depth of both the perception stack and the reasoning stack. |
| **Parameterization** | **Per-site generators, and deliberately cheap ones.** App. A.1: Encoder-FiLM's $\gamma_i,\beta_i$ are *"derived from a **linear transformation** of the text embedding"*; LLM-FiLM's $\gamma_{\mathrm{LLM}}(\cdot),\beta_{\mathrm{LLM}}(\cdot)$ are *"both implemented using two-layer MLPs, with a hidden layer dimension of 2048, resulting in a parameter count almost identical to that of a direct linear layer."* |
| **Initialization** | **Not stated — but the parameterisation is identity-centred by construction.** Both Eq. 10 and Eq. 13 use the **residual form $(1 + \gamma(\cdot))\odot x + \beta(\cdot)$**, so at $\gamma=\beta=0$ the modulation is exactly the identity. This is the same device as De Vries et al.'s conditional batch norm and achieves the effect of AdaLN-Zero without saying so. **No explicit initialisation scheme is reported, and no claim is made that it matters.** |
| **Ablation present?** | **Partially — and not in the way the survey implies.** Table 4 ablates whole *stages* (intra-encoder aggregation, cross-encoder aggregation, pruning, task-guided pruning, coupled attention) at a fixed 8× sparsification: full model **98.6** on LIBERO-Spatial, ablated variants spanning **91.2–96.2**. Dropping one component costs ≈2.4–3.4 points; dropping two costs ≈6.6–7.4. Critically, **the caption states "Pruning and TG-Pruning denote LFP-Routing w/o and w/ instruction guidance", so instruction conditioning *is* ablated — but there is no row anywhere that removes the FiLM modulation while keeping the stage.** The mechanism is never isolated. (The row-to-column checkmark mapping in Table 4 is not reliably recoverable from text extraction, so no per-row attribution is quoted here.) |
| **Reported instability** | **None.** No NaN, no gain blow-up, no entropy or plasticity discussion. The nearest positive statement is App. C.1: across 3 random seeds, standard deviations are **0.2–0.6 %**, which the authors read as *"stable learning behavior"*. Limitations (App. E.2) name only the reliance on **predefined sparsity ratios**. |
| **Single-task or multi-task** | **Multi-task, language-conditioned.** LIBERO's four suites (Spatial, Object, Goal, Long), 500 trials per suite, plus real-world ALOHA tasks including dual-arm T-shirt folding. The conditioner exists because instructions vary. **No single-task evidence.** |

### 10.2 Verification of the survey's claims — one substantive correction

| Survey claim | Verdict | Evidence |
|---|---|---|
| NeurIPS 2025 Poster | **Confirmed** | PDF page-1 footer |
| Conditioners: (i) the language instruction, (ii) the action intent | **Confirmed as to (i); imprecise as to (ii)** | LFP-Routing's conditioner is the **instruction tokens $t_l$ at layer $l$**, not a separate action-intent signal. "Action intent" is the paper's *cognitive-science framing* of what those tokens carry (the SMA analogy), not a distinct input. |
| Modulates (i) the vision encoder, (ii) the LM trunk | **Confirmed** | §2.3.1, §2.3.2 |
| Granularity (b) per-channel | **Confirmed** | "scale and shift vectors", element-wise |
| "2 sites, one per stage of a 3-stage pipeline" | **Confirmed at the stage level, understated at the layer level** | Two *kinds* of FiLM, but each is applied throughout its subsystem — LLM-FiLM alone spans 32 layers |
| **"the FiLM output is not only a gain — it drives token aggregation and token pruning… FiLM becomes a discrete-computation controller"** | **CORRECTED — this over-reads the mechanism** | See below |
| 97.4 % LIBERO / 70.0 % real-world; 2.5× cheaper training, 2.8× lower latency | **Confirmed** | Table 1 average 97.4 %; Table 7 gives $97.4\pm0.4$ over 3 seeds; §3.3 reports *"2.79× faster inference time, 22.54× higher throughput, 3.12× lower FLOPs, and 2.49× reduction in training cost"* |

**The correction, stated precisely.** The survey's headline reading was that CogVLA turns FiLM itself into a router — that the $(\gamma,\beta)$ *are* the aggregation and pruning scores. Full text shows a **two-stage pipeline in which FiLM conditions the representation and a separate learned router then makes the discrete decision**:

- **Pruning (Stage 2).** Eq. 13 applies FiLM to the visual tokens, $f_{FP}(Z_l,t_l) = \mathrm{Prune}\bigl((1+\gamma_{\mathrm{LLM}}(t_l))\odot Z_l\bigr) + \beta_{\mathrm{LLM}}(t_l)$. But the retention decision comes from Eq. 14, a **different network**: $R^j_l = \mathrm{MLP}(Z^j_l)$ — routing weights computed **from the token itself**, not from the FiLM parameters — thresholded at the $\beta$-th percentile $P^l_\beta$ (Eq. 15). FiLM shapes *what the router sees*; it does not *emit* the score.
- **Aggregation (Stage 1).** Cross-encoder fusion is likewise its own module: $\alpha = \mathrm{Sigmoid}\bigl(W_2(\sigma(W_1 t_r + b_1)) + b_2\bigr)$ (Eq. 11), a separate two-layer MLP on the text embedding, used as $v_{\mathrm{agg}} = \alpha\, v^{\mathrm{SigLIP}}_{\mathrm{agg}} + (1-\alpha)\, v^{\mathrm{DINOv2}}_{\mathrm{agg}}$ (Eq. 12).

So the accurate statement is: **CogVLA composes affine modulation with discrete routing in a single instruction-driven pipeline, but the two are separate learned modules.** The module names ("Encoder-FiLM **based** Aggregation Routing", "LLM-FiLM **based** Pruning Routing") describe a dependency, not an identity — and the word "based" is doing real work.

**Why this matters for us.** Under the survey's reading, CogVLA would have been evidence that a modulation signal can be *repurposed* as a discrete controller — which would have been a genuinely new role for $(\gamma,\beta)$ and worth chasing. Under the correct reading it is a more ordinary, if well-executed, instance: **conditional gains used to make a downstream selection problem easier.** That is still interesting, but it is not a new primitive.

**One internal inconsistency worth flagging** for anyone reading the paper closely: Eqs. 6–7 describe cross-encoder fusion as a softmax over $N$ encoders, $\alpha = \mathrm{Softmax}(\mathrm{MLP}_{\text{route}}(t_r))$, while Eqs. 11–12 describe it as a sigmoid producing a single interpolation weight between exactly two branches. These are the general and the $N{=}2$ instantiated forms of the same idea, but the paper never reconciles the notation.

### 10.3 Phase 1 — Foundational overview

**Introduction.** A vision-language-action model reads a camera image and a typed instruction ("place the red cup at the corner of the table") and outputs robot motor commands. The good ones are built on top of large pretrained vision-language models, and they are ruinously expensive: fine-tuning a 7-billion-parameter one on a *single* benchmark task consumes over 600 GPU-hours.

Most of that cost is spent looking at things that do not matter. The image becomes hundreds of "visual tokens", and the network processes all of them at every layer — the tablecloth, the far wall, the objects the instruction never mentions. The obvious fix is to throw some away, and several methods do. CogVLA's observation is that existing methods throw them away **without consulting the instruction**, so they discard exactly the fine details the task needed while retaining bulk background.

Its proposal is to let the instruction decide, in three stages modelled loosely on how humans coordinate looking and acting. First, the instruction is used to **steer the vision encoders themselves**, so that a small set of summary tokens absorbs the instruction-relevant content and the rest of the image can be dropped — down to 25 % of the original. Second, inside the language model, the instruction again steers the visual tokens, after which a small scoring network **deletes about half** of them layer by layer. Third, a custom attention pattern keeps the surviving compressed representation coherent enough to still produce a sensible sequence of actions.

**Key findings.** On the LIBERO benchmark CogVLA reaches 97.4 % success — the best in its comparison table — while using 8× fewer visual inputs, running 2.79× faster, and costing 2.49× less to train than OpenVLA. Real-world dual-arm experiments, including folding a T-shirt, reproduce the advantage. Across three random seeds the results vary by half a percentage point or less.

**Initial takeaway.** Conditioning is not only a way to make a network behave differently — it can be a way to make it **do less work**. If the steering signal tells you what matters, it also tells you what you are allowed to discard. The steering and the discarding remain two separate mechanisms here, but they are trained to cooperate.

### 10.4 Phase 2 — Graduate-level deep dive

#### 10.4.1 Stage 1 — Encoder-FiLM and intra-encoder aggregation

CogVLA runs $N$ vision encoders (in practice $N=2$: SigLIP and DINOv2). Each branch carries a set of learnable **aggregation tokens** $v^{(i)}_{\mathrm{agg}}$ — 64 per encoder (App. A.1) — alongside the image tokens $I^{(i)}$. The modulated block is (Eq. 10):

$$
\begin{aligned}
f_{FA}\bigl(I^{(i)}, v^{(i)}_{\mathrm{agg}}, t_r\bigr) &= \bigl(1 + \gamma_i(t_r)\bigr)\odot \mathrm{Self\text{-}Att}\bigl(I^{(i)}, v^{(i)}_{\mathrm{agg}}\bigr) + \beta_i(t_r),\\
v^{(i)}_{\mathrm{agg}} &\leftarrow \mathrm{Aggregate}\bigl(\mathrm{FFN}(f_{FA}(\cdot))\bigr) + v^{(i)}_{\mathrm{agg}} .
\end{aligned}
$$

**Three things are worth extracting from this equation.**

**(i) The $(1+\gamma)$ residual form is an initialisation strategy hiding in the algebra.** Writing the scale as $1+\gamma$ rather than $\gamma$ means the *learned* quantity is a **deviation from identity**. If $\gamma$ and $\beta$ start near zero — which a linear layer with small init does by default — the block begins as an unmodulated self-attention block and the conditioning grows in. This is functionally what AdaLN-Zero achieves by explicitly zeroing the generator's last layer, and what De Vries et al. 2017 called the residual-offset form $\gamma_c(z) = \gamma_c + \Delta\gamma_c(z)$. **CogVLA gets the property for free from the parameterisation and never discusses it.** For a corpus-level count of "papers that ensure the modulation starts at identity", this one belongs in the yes column, by construction rather than by declaration.

**(ii) The modulation target is the attention output, not a normalisation output.** Unlike adaLN (FLOWER, GR00T), $\gamma$ here multiplies the raw self-attention result. Like PAPL ([§8](#8-yoon-et-al-2026--papl-phase-aware-policy-learning-for-skateboard-riding-of-quadruped-robots-via-feature-wise-linear-modulation)), there is no normalisation layer regulating the scale that $\gamma$ then rescales — but unlike PAPL, the $(1+\gamma)$ form anchors it at 1.

**(iii) The modulation is what makes the compression *lossy in the right direction*.** After the encoder blocks, *"only the final $v^{(i)}_{\mathrm{agg}}$ is retained while the image tokens $I^{(i)}$ are discarded."* The aggregation tokens are a bottleneck, and the FiLM is what determines which image content flows into that bottleneck. **The modulation's job is to bias an information bottleneck, not to change a decision boundary** — a genuinely different use from every other paper in this corpus.

Cross-encoder fusion then applies a second, separate instruction-conditioned gate (Eqs. 11–12), letting the instruction decide how much to trust SigLIP's semantics versus DINOv2's geometry — *"the fusion ratio is dynamically predicted for different tasks to reflect instruction-dependent visual preferences."*

#### 10.4.2 Stage 2 — LLM-FiLM and task-guided pruning

Inside the language model, at layer $l$, with visual tokens $Z_l$ and instruction tokens $t_l$ (Eq. 13):

$$
\begin{aligned}
f_{FP}(Z_l, t_l) &= \mathrm{Prune}\Bigl(\bigl(1 + \gamma_{\mathrm{LLM}}(t_l)\bigr)\odot Z_l\Bigr) + \beta_{\mathrm{LLM}}(t_l),\\
Z_{l+1} &= \mathrm{FFN}\bigl(\mathrm{Self\text{-}Att}(f_{FP}(\cdot))\bigr) + Z_l .
\end{aligned}
$$

The pruning operator is defined by a **Task-Guided Pruning Router** (Eqs. 14–15):

$$
R^j_l = \mathrm{MLP}\bigl(Z^j_l\bigr), \qquad
Z^j_{l+1} = \begin{cases} R^j_l \times f_{SF}\bigl([Z^j_l, t_l]\bigr) + Z^j_l, & \text{if } R^j_l > P^l_\beta,\\[2pt] Z^j_l, & \text{otherwise,}\end{cases}
$$

where $P^l_\beta$ is the $\beta$-th percentile of the routing weights at layer $l$ and $f_{SF}$ is that layer's self-attention-plus-FFN. **Note the form of the skip: tokens below threshold are not deleted from the sequence — they are passed through unchanged, bypassing the layer's computation.** That is Mixture-of-Depths-style *computation skipping*, which is why it saves FLOPs without breaking the sequence structure that later layers and the attention mask depend on.

The retention ratio follows a **shifted cosine schedule** over depth (App. A.1, Eq. 21):

$$
\beta_l = \tfrac{1}{2}\cos\frac{\pi l}{L} + \eta, \qquad l = 1,\dots,L,\quad L = 32,\ \eta = 0.5,
$$

clamped to $[0.05, 0.85]$, giving ≈50 % average pruning. **Reading the schedule:** at $l=0$, $\beta \approx 1.0$ (clamped to 0.85) — keep almost everything early; at $l = L/2$, $\beta = 0.5$; at $l = L$, $\beta \approx 0$ (clamped to 0.05) — keep almost nothing late. So the network is **progressively narrowed with depth**, on the reasoning that early layers still need broad visual context to resolve references while late layers only need the few tokens the action depends on. This is a hand-designed schedule, and it is precisely the *"reliance on predefined sparsity ratios"* the paper names as its main limitation.

#### 10.4.3 Stage 3 — V-L-A Coupled Attention

Given the layer-$l$ sequence $\tilde X = [Z_l, t_l, A_l] \in \mathbb{R}^{(M+T+K)\times D}$ with action chunk $A_l = [a^l_0,\dots,a^l_{K-1}]$, CAtten applies **causal** attention within the vision-language segment,

$$
\mathrm{Attn}_{VL}\bigl([Z_l,t_l]\bigr) = \mathrm{Softmax}\!\left(\frac{[Z_l,t_l][Z_l,t_l]^\top}{\sqrt{d}} + M^{VL}_{\mathrm{causal}}\right)[Z_l,t_l],
$$

with $M^{VL}_{\mathrm{causal}}$ lower-triangular, and **bidirectional** attention within the action segment so that all $K$ actions in the chunk can be decoded in **parallel** rather than autoregressively (Eq. 9: $A_t = f_{\mathrm{parallel}}([Z_0, t_0, \mathbf{0}_0,\dots,\mathbf{0}_{K-1}])$, with placeholder action tokens). Chunk size $K = 8$ (App. A.2). The parallel decoding, not the sparsification, is what produces the 22.54× throughput figure.

#### 10.4.4 Results and ablations, as printed

**LIBERO (Table 1, 500 trials per suite; Table 7, 3 seeds):**

| Method | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA | $84.7\pm0.9$ | $88.4\pm0.8$ | $79.2\pm1.0$ | $53.7\pm1.3$ | $76.5\pm0.6$ |
| $\pi_0$ fine-tuned | 96.8 | **98.8** | 95.8 | 85.2 | 94.2 |
| STAR (ICML'25) | $95.5\pm0.6$ | $98.3\pm0.2$ | $95.0\pm0.7$ | $88.5\pm0.3$ | $94.3\pm0.1$ |
| **CogVLA** | $\mathbf{98.5\pm0.5}$ | $\mathbf{98.8\pm0.4}$ | $96.5\pm0.6$ | $\mathbf{95.2\pm1.1}$ | $\mathbf{97.4\pm0.4}$ |

The margin is concentrated in **LIBERO-Long** (95.2 vs the next-best 88.5), the long-horizon suite — consistent with the claim that instruction-guided compression preserves referential coherence that naive compression destroys. The paper is candid that it ranks only second on LIBERO-Goal, *"primarily due to a deliberate trade-off between performance and efficiency."*

**Component ablation (Table 4, LIBERO-Spatial, fixed 8× sparsification):** full model **98.6**; variants **91.2–96.2**. Removing one component costs ≈2.4–3.4 points, removing two costs ≈6.6–7.4.

**Sparsification allocation (Table 5, fixed 8× total):**

| Stage 1 ratio | Stage 2 ratio | Total | Spatial SR |
|---|---|---|---|
| 1× | 8× | 8× | 91.2 (−7.4) |
| 8× | 1× | 8× | 92.0 (−6.6) |
| 2× | 4× | 8× | 94.6 (−4.0) |
| **4×** | **2×** | 8× | **98.6** |

Both extremes lose ~7 points; the interior split wins by a wide margin. *"both stages contribute to performance gains, with larger Stage 1 ratios yielding greater improvements, as Stage 2 further filters instruction-relevant tokens based on Stage 1 outputs."* **This is a genuine and non-obvious finding: where in the pipeline you spend a fixed compression budget matters more than the budget itself.**

**Versus generic compression (Table 6):** CogVLA's Stages 1+2 reach 98.6 against FastV's 88.2 (−10.4) and SliME's 77.6 (−21.0), at the same ratio — the clearest support for "instruction-driven beats instruction-agnostic sparsification".

### 10.5 Relevance to this project

**Bears on our design only weakly and indirectly. The honest summary is that this is a scale-and-modality regime we do not occupy.**

- **It does not speak to any of our five open questions.** Granularity is standard per-channel (nothing for question 1); the conditioner is language and the modulated stream is vision, so the two information sources are disjoint (nothing for question 2); no instability of any kind is reported (nothing for question 3); it is purely multi-task (nothing for question 4); there is no critic (nothing for question 5). **This is the clearest "does not bear on our design" in the reviewed corpus so far**, and saying so is more useful than manufacturing a connection.
- **One transferable idea, and it is a real one: modulation to bias a bottleneck.** Every other paper here modulates to change *what a network computes*. CogVLA modulates to change *what a network keeps*. Our agent has a structural analogue we have never exploited — the **multimodal hub**, where unimodal streams are fused. Modulating a fusion point is a decision about *which modality gets through*, which is a bottleneck-biasing role, not a decision-boundary role. We already inject $\gamma/\beta$ there; we have never analysed the learned gains as a **modality-attention readout**. Doing so is a post-hoc analysis of runs we already have: correlate the hub's learned gains against which modality is task-relevant at that timestep. If the modulator has learned anything interpretable, that is where it will show.
- **The $(1+\gamma)$ residual parameterisation is a small, concrete, checkable design detail.** CogVLA, De Vries et al. 2017, FLOWER (zero-init) and GR00T all ensure the modulation starts at or near identity, by three different mechanisms. Our design achieves the additive half via a zero-initialised per-neuron baseline. **Whether our multiplicative path is anchored at 1 — i.e. whether we compute $\gamma \odot x$ or $(1+\gamma)\odot x$ — is a code question with a real consequence for early training dynamics**, and this is now the fourth independent citation for anchoring it. Worth a `senior-developer` issue plan alongside the zero-init check raised in [§7](#7-reuss-et-al-2025--flower-democratizing-generalist-robot-policies-with-efficient-vision-language-action-flow-policies).
- **The mechanism-isolation gap is a lesson about our own ablations, not about theirs.** CogVLA ablates *stages*, never the FiLM inside a stage, so it cannot say what its modulation contributed. Our null-result series has the same shape: we compare "modulator vs. no modulator" rather than "modulator vs. same information delivered another way". Yuan 2024 ([§6](#6-yuan-2024--unpacking-the-individual-components-of-diffusion-policy)) shows what the sharper experiment looks like.
- **Scale caveat, stated plainly.** This is a ~7 B-parameter model, LoRA-tuned for 60 k steps at batch 64, on multi-task language-conditioned imitation data. Our agent is a 128-unit GRU trained with PPO in a gridworld. Nothing about the *dynamics* here transfers; only the *design patterns* do.

### 10.6 Appendix: Section-by-Section Backbone

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | VLA models built on pretrained VLMs need extensive post-training. Existing sparsification (Mixture-of-Depths, layer skipping, early exit) neglects semantic coupling across modalities and focuses narrowly on intra-LLM computation. CogVLA = 3-stage progressive architecture: **EFA-Routing** (Encoder-FiLM based Aggregation), **LFP-Routing** (LLM-FiLM based Pruning), **CAtten** (V-L-A Coupled Attention). 97.4 % LIBERO / 70.0 % real-world; 2.5× cheaper training, 2.8× lower latency vs OpenVLA. |
| 1 | **Introduction** | Cost framing: fine-tuning a 7 B VLA with action chunking on one LIBERO task exceeds 600 A100-hours. Three named failure modes of modular sparsification: visual compression discards task-relevant detail; token skipping breaks reference resolution; action generation lacks causal reasoning over state transitions. Cognitive-science motivation: **VAS** (visual attention system) → Encoder-FiLM, **SMA** (supplementary motor area) → LLM-FiLM, **PMC** (premotor cortex) → CAtten. Stage 1 compresses visual tokens to 25 %; Stage 2 skips attention over ~50 % of remaining tokens. |
| 2.1 | Preliminary: Parallel Decoding in Action Chunk | Autoregressive vs parallel decoding; action chunk $A = [a_0,\dots,a_{K-1}] \in \mathbb{R}^{K\times D}$, $D=7$ for 3-DoF translation plus rotation and gripper. |
| 2.2 | **CogVLA: Framework** | Eq. 5 (Encoder-FiLM per branch), Eqs. 6–7 (softmax cross-branch routing, general $N$), Eq. 8 ($Z_{l+1} = \mathrm{CAtten}(\mathrm{LFP\text{-}Routing}(Z_l,t_l))$), Eq. 9 (parallel decode with placeholder tokens). |
| 2.3.1 | **Encoder-FiLM based Aggregation Routing** | Step 1 intra-encoder aggregation, **Eq. 10** — the $(1+\gamma)\odot\mathrm{Self\text{-}Att} + \beta$ form; image tokens discarded after aggregation, leaving 25 %. Step 2 cross-encoder aggregation, **Eqs. 11–12** — sigmoid gate on the text embedding fusing SigLIP and DINOv2. |
| 2.3.2 | **LLM-FiLM based Pruning Routing** | **Eq. 13** (FiLM then Prune), **Eq. 14** (router MLP on the token), **Eq. 15** (percentile-thresholded computation skip). Retention ratio $\beta$ governs sparsity. |
| 2.3.3 | V-L-A Coupled Attention | Eq. 16 sequence definition; **Eq. 17** causal vision-language attention with lower-triangular mask; bidirectional attention within the action segment for parallel decoding. |
| 3.1 | Experiments Setting | LIBERO benchmark, four suites; real-world ALOHA setup. |
| 3.2 | Performance improvement | Table 1 (SR and rank across 13 methods); 500 trials per suite. Real-world: Object Placement, Drawer Manipulation, dual-arm T-shirt Folding. Candid about ranking second on LIBERO-Goal as an efficiency trade-off. |
| 3.3 | Efficiency Optimization | Table 3: **2.79× faster inference, 22.54× higher throughput, 3.12× lower FLOPs, 2.49× lower training cost** vs OpenVLA; also beats OpenVLA-OFT and PD-VLA. |
| 3.4 | Qualitative Analysis | Fig. 4: CogVLA avoids failures such as drawer collisions; efficiency advantage grows with task length. |
| 3.5 | **Ablation Studies** | Table 4 (component ablation at fixed 8× sparsification), Table 5 (sparsification ratio allocation — 4×/2× optimal), Table 6 (vs FastV and SliME). |
| 4 | Related Work | VLA lineage (CLIPort, PerAct, RT series, Octo, OpenVLA, $\pi$ series). Efficiency work split into LLM-centric (Mixture-of-Depths, dynamic depth, sparse MoE, TinyVLA) and vision-centric (patch selection, cropping, compression modules); argues naive adaptation causes cross-modal semantic inconsistency. |
| 5 | Conclusion | Instruction-driven multimodal sparsification as the route to scalable embodied AI. |
| A.1 | Model Details | **64 aggregation tokens per encoder → 25 % of original visual tokens. Encoder-FiLM $\gamma_i,\beta_i$ from a linear transformation of the text embedding. LLM-FiLM $\gamma_{\mathrm{LLM}},\beta_{\mathrm{LLM}}$ as two-layer MLPs, hidden dim 2048.** Eq. 21 shifted cosine retention schedule, $L=32$, $\eta=0.5$, clamped to $[0.05,0.85]$, ≈50 % pruning. |
| A.2 | Training Details | OpenVLA backbone, chunk size $K=8$, LoRA rank 32 / $\alpha$ 64, 60 K steps, batch 64, initial LR 5e-4, checkpointed every 10 K. |
| B | Experimental Details | Real-world task specifications including dual-arm T-shirt folding and multi-attribute grounding tasks. |
| C.1 | **Multi-Seed Evaluation** | Table 7: 3 independent seeds per suite; CogVLA std dev **0.2–0.6 %** — *"stable learning behavior"*. |
| C.2–C.3 | Extended results and ablations | Additional real-world tasks and ablations. |
| D | Qualitative Analysis | Instruction-to-observation attention visualisations. |
| E.1 | Motivation | The VAS / SMA / PMC mapping spelled out: Encoder-FiLM as the visual attention system, LLM-FiLM as the *"intention planner"*. |
| E.2 | **Limitation and Future Work** | *"the current instruction-to-vision routing relies on predefined sparsity ratios"* — the cosine schedule is hand-designed rather than learned. |

---

## 11. NVIDIA et al. 2025 — *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*

**PDF:** `docs/project/references/modulation_in_rl/sources/NVIDIA et al. 2025 - GR00T N1 (preprint).pdf`
**arXiv:** [2503.14734](https://arxiv.org/abs/2503.14734). PDF header dated 2025-3-28, authored as "NVIDIA¹" (contributor list in App. A). Model, training data and simulation benchmarks publicly released.

> **Read §11.2 first.** This paper is in the corpus as the canonical citation for *"adaptive layer normalisation is the standard action-head conditioner"*. Full-text reading confirms that claim — and shows the paper devotes **exactly one clause** to it. That is itself the finding.

### 11.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **Unstated in PDF.** No venue line, no conference footer, no "submitted to" note; the header carries only the date `2025-3-28` and the corporate author `NVIDIA`. This is an industrial technical report. **Cite as a preprint.** |
| **Conditioning signal** | The **denoising (flow-matching) timestep** $\tau \in [0,1]$. Nothing else reaches the model by modulation: vision and language arrive by **cross-attention** on Eagle-2 VLM tokens, and proprioceptive state arrives by **concatenation** into the self-attention stream. |
| **Modulated component** | The **Diffusion Transformer (DiT) action module** — "System 1" — which alternates self-attention blocks (over noised action tokens $A^\tau_t$ plus state embeddings $q_t$) with cross-attention blocks (over VLM tokens $\varphi_t$). Not the VLM, not an encoder, not a critic (imitation learning; no value function). |
| **Granularity cell** | **(b) per-channel, inferred from the DiT lineage — not stated in this PDF.** The paper says only *"a variant of DiT (Peebles and Xie, 2023), which is a transformer with denoising step conditioning via adaptive layer normalization"*. Standard adaLN is per-channel; **this paper never says so**, gives no equation for it, and never names $\gamma$ or $\beta$. Recorded as **(b), inherited, not stated**. |
| **Placement** | **Every DiT block** (Fig. 3 shows "DiT Blocks × N" with alternating self- and cross-attention). **The block count $N$ is not reported anywhere in the paper**, nor is the DiT hidden dimension. Total model = 2.2 B parameters, of which 1.34 B is the VLM, so the DiT plus encoders account for ~0.86 B. |
| **Parameterization** | **Not stated.** No description of the adaLN generator, no parameter count, no statement of whether it is shared across blocks. |
| **Initialization** | **Not stated.** AdaLN-Zero is not mentioned. Whether the modulation is zero-initialised is unrecoverable from this document. |
| **Ablation present?** | **No — the modulation mechanism is neither ablated nor discussed.** The paper's ablations are all about *data*: neural-trajectory augmentation across three data regimes (Fig. 9), latent versus inverse-dynamics-model action labels, and 10 %-versus-full data. One architectural finding is reported, and it concerns *where the VLM is tapped*, not modulation: *"using middle-layer instead of final-layer LLM embeddings resulted in both faster inference speed and higher downstream policy success rate"* — layer 12 for GR00T-N1-2B. |
| **Reported instability** | **None.** No NaN, no gain blow-up, no entropy or plasticity discussion. Limitations (§4.6) name only short-horizon tabletop scope, the need for a stronger VLM backbone, and synthetic-data quality. |
| **Single-task or multi-task** | **Massively multi-task and multi-embodiment** — *"a massively multi-task, language-conditioned policy that supports a wide range of robot embodiments"*, trained on human egocentric video, simulation, neural-generated trajectories and real robot demonstrations, using up to 1024 GPUs and ~50,000 H100-hours. **No single-task evidence.** |

### 11.2 Verification of the survey's claim — confirmed, and the confirmation is thin

The survey listed GR00T N1 in its component-use table with the note: *"adaptive layer normalisation (DiT-style) for denoising-step conditioning; cross-attention for vision-language conditioning… Verified from the rendered full text: 'a variant of DiT … a transformer with denoising step conditioning via adaptive layer normalization'."*

**That is exactly right, and it is the whole of the evidence.** The string `adaptive layer normalization` appears **once** in a 36-page document (§2.1, "Diffusion Transformer Module (System 1)"). The words `FiLM`, `modulation`, `modulate`, `AdaLN`, `adaLN`, `scale and shift`, `$\gamma$` and `$\beta$` appear **nowhere**. There is no equation for the modulation, no granularity statement, no site count, no initialisation scheme, and no ablation.

**The finding is the silence, and it is worth stating explicitly in our write-ups.** At a frontier robotics lab, in a 2.2 B-parameter flagship model trained for 50,000 GPU-hours, feature-wise modulation is **so completely standardised that it is mentioned in passing as a property of an architecture family and never discussed again**. That is a *stronger* form of "this is the field convention" than an enthusiastic advocacy paper would be — nobody argues for the thing everyone already does.

**But it has a direct consequence for how we may cite it.** GR00T N1 supports the claim *"adaLN is the default conditioning mechanism for sub-billion-parameter action heads in 2025"*. It supports **no** claim about granularity, placement depth, parameterisation, initialisation, or stability, because it makes none. Any citation of the form "GR00T N1 uses per-channel modulation at every block with zero-initialisation" would be an inference from the DiT lineage dressed up as a citation. **Cite Peebles & Xie 2023 for the mechanism and GR00T N1 only for the fact of adoption.**

**One useful corrective to the survey's framing.** The survey grouped GR00T N1 and $\pi_0$ as jointly showing that "attention displaces affine modulation at frontier scale". GR00T N1 shows something more specific and more interesting: **the two mechanisms are used side by side for different signal types in the same model.** The denoising timestep — a scalar, low-dimensional, and needed identically by every token — goes through adaLN. Vision and language — a rich token sequence — go through cross-attention. Proprioceptive state goes through concatenation. **Three conditioning signals, three different mechanisms, chosen by the structure of the signal.** That matches FLOWER's split ([§7](#7-reuss-et-al-2025--flower-democratizing-generalist-robot-policies-with-efficient-vision-language-action-flow-policies)) exactly, and it is a more accurate summary of 2025 practice than "attention won".

### 11.3 Phase 1 — Foundational overview

**Introduction.** A humanoid robot needs both a capable body and something to drive it. GR00T N1 is NVIDIA's attempt at the latter: a single "foundation model" that takes camera images and a typed instruction ("pick up the apple and place it into the bottom shelf") and produces motor commands, across many different robot bodies — single arms, bimanual setups, dexterous hands.

Its organising idea is a **two-system split** borrowed from dual-process psychology. "System 2" is a pretrained vision-language model that reads the scene and the instruction slowly and semantically. "System 1" is a much faster network that turns that understanding into smooth motor commands in real time. Both are trained together end-to-end.

The second idea is about **data**, and it is where most of the paper's effort goes. Real robot demonstrations are scarce and expensive, so the authors build a "data pyramid": web-scale human videos at the base, synthetically generated robot trajectories in the middle, and a small peak of real teleoperation data. To use human video, which has no robot actions attached, they train a separate model to infer a "latent action" from consecutive video frames and train on that instead. To multiply the real data, they use a video generation model to invent plausible new trajectories — expanding 88 hours of teleoperation into 827.

**Key findings.** Trained on 1024 GPUs for roughly 50,000 GPU-hours, the resulting 2.2-billion-parameter model beats from-scratch baselines across three simulation benchmarks (45.0 % average against 33.4 % for Diffusion Policy) and on a real Fourier GR-1 humanoid (76.8 % against 46.4 %). Most strikingly, it is **data-efficient**: fine-tuned on just 10 % of the real demonstrations, it comes within 3.8 points of a Diffusion Policy trained on all of them.

**Where the steering fits.** The fast action module has to be told *how far along* it is in its iterative denoising process — a single number between 0 and 1. That number is delivered by generating a gain and an offset for the network's normalisation layers. The paper mentions this once, in a subordinate clause, and never returns to it. **That is what "standard practice" looks like from the inside.**

### 11.4 Phase 2 — Graduate-level deep dive

#### 11.4.1 The flow-matching objective and where the timestep enters

GR00T N1 predicts action chunks $A_t = [a_t, a_{t+1}, \dots, a_{t+H-1}]$ with horizon $H = 16$. Given a ground-truth chunk $A_t$, a flow-matching timestep $\tau \in [0,1]$ and noise $\epsilon \sim \mathcal{N}(0,I)$, the noised chunk is the linear interpolant

$$
A^\tau_t = \tau A_t + (1-\tau)\epsilon,
$$

and the network $V_\theta$ regresses the denoising vector field:

$$
\mathcal{L}_{\mathrm{fm}}(\theta) = \mathbb{E}_\tau\Bigl[\bigl\lVert V_\theta\bigl(\varphi_t, A^\tau_t, q_t\bigr) - (\epsilon - A_t)\bigr\rVert^2\Bigr]. \tag{1}
$$

The timestep is sampled from a **Beta distribution biased toward low $\tau$** (i.e. toward the noisy end): $p(\tau) = \mathrm{Beta}\bigl(\tfrac{s-\tau}{s}; 1.5, 1\bigr)$ with $s = 0.999$, following Black et al. Inference is $K$-step forward Euler integration,

$$
A^{\tau + 1/K}_t = A^\tau_t + \tfrac{1}{K} V_\theta\bigl(\varphi_t, A^\tau_t, q_t\bigr),
$$

with **$K = 4$** sufficient across all embodiments — giving 63.9 ms per 16-action chunk on an L40 in bf16.

**Why $\tau$ is the archetypal adaLN conditioner, and why this is the right design choice.** Three properties make the timestep uniquely suited to feature-wise modulation, and naming them clarifies what modulation is *for*:

1. **It is scalar.** There is no token structure to attend over, so cross-attention would be wasteful — a single query attending to a single key.
2. **It is global to the forward pass.** Every token in the chunk is at the same noise level, so a per-token mechanism would be redundant. A signal that must reach *all* units identically is exactly what a broadcast gain delivers.
3. **It must change the computation, not add information.** The denoiser's *function* differs qualitatively between $\tau \approx 0$ (predict coarse structure from near-pure noise) and $\tau \approx 1$ (refine fine detail). As derived in [§6](#6-yuan-2024--unpacking-the-individual-components-of-diffusion-policy), a concatenated input can only translate a layer's output, while modulation rescales its Jacobian — it can genuinely re-weight which directions propagate. **The timestep needs to reshape the vector field, not shift it.**

**The contrast within the same model is the instructive part.** GR00T N1 routes three conditioning signals three different ways, and each choice matches the signal's structure:

| Signal | Structure | Mechanism | Why |
|---|---|---|---|
| Flow timestep $\tau$ | scalar, global, changes the *function* | **adaLN** | broadcast gain/offset; no token structure to attend over |
| Vision + language $\varphi_t$ | long token sequence, content-addressable | **cross-attention** | needs selective, per-token retrieval |
| Proprioceptive state $q_t$ | vector, same "rank" as the action tokens | **concatenation into self-attention** | participates as a peer token, not a controller |

#### 11.4.2 The architecture around it

**System 2 (vision-language).** Eagle-2, itself a SmolLM2 language model plus a SigLIP-2 image encoder. Images at $224\times224$ followed by pixel shuffle → **64 image tokens per frame**. The reported architectural finding: **middle-layer LLM features beat final-layer features** on both speed and downstream success, and layer 12 is used for GR00T-N1-2B. This is the same result FLOWER reports as its "intermediate fusion" contribution — FLOWER notes that GR00T explored it *"without any ablation and insights on the impact"*, which full-text reading confirms: GR00T states the finding in one sentence with no supporting numbers.

**System 1 (DiT).** Alternating cross-attention and self-attention blocks in the style of Flamingo and VIMA. Self-attention operates over noised action tokens plus state embeddings; cross-attention conditions on VLM tokens. **Embodiment-specific MLPs** encode state and action into a shared embedding dimension — the mechanism by which one model spans robots with different degrees of freedom — and an embodiment-specific Action Decoder MLP maps the final $H$ tokens back to motor commands. As in $\pi_0$, the **Action Encoder MLP also encodes the diffusion timestep together with the noised action vector**, meaning $\tau$ reaches the network by *two* routes: through the action encoder and through adaLN. The paper does not comment on this redundancy.

#### 11.4.3 Results, as printed

**Simulation (Table 2, 100 demonstrations per task; 24 RoboCasa tasks, 9 DexMG, 24 GR-1):**

| Method | RoboCasa | DexMG | GR-1 | Average |
|---|---|---|---|---|
| BC Transformer | 26.3 % | 53.9 % | 16.1 % | 26.4 % |
| Diffusion Policy | 25.6 % | 56.1 % | 32.7 % | 33.4 % |
| **GR00T-N1-2B** | **32.1 %** | **66.5 %** | **50.0 %** | **45.0 %** |

**Real-world (Table 3, Fourier GR-1 humanoid):**

| Method | Pick-and-Place | Articulated | Industrial | Coordination | Average |
|---|---|---|---|---|---|
| Diffusion Policy (10 % data) | 3.0 % | 14.3 % | 6.7 % | 27.5 % | 10.2 % |
| Diffusion Policy (full data) | 36.0 % | 38.6 % | 61.0 % | 62.5 % | 46.4 % |
| GR00T-N1-2B (10 % data) | 35.0 % | 62.0 % | 31.0 % | 50.0 % | 42.6 % |
| **GR00T-N1-2B (full data)** | **82.0 %** | **70.9 %** | **70.0 %** | **82.5 %** | **76.8 %** |

**Pre-training zero-shot on the real robot:** 76.6 % (11.5/15) on a task requiring a bimanual hand-off, 73.3 % (11/15) on placing a novel object into an unseen container.

**None of these numbers speak to modulation.** They are evidence about pre-training scale and data strategy. Recording them here is for context only.

### 11.5 Relevance to this project

**Bears on our design almost not at all as evidence — but it is the citation we would use for one specific sentence, and it sharpens one design question.**

- **It is the canonical citation for "affine modulation is standard practice in 2025 action heads", and nothing more.** When our write-ups need to establish that feature-wise modulation is a live, mainstream mechanism rather than a 2018 curiosity, this is the reference — a flagship industrial model that uses it without feeling any need to justify it. **Do not cite it for granularity, placement, initialisation or stability**, on all of which it is silent.
- **The three-mechanisms-by-signal-structure pattern is a genuine design principle, and it is the second independent instance.** GR00T N1 and FLOWER independently arrive at: *scalar/global signals → modulation; token-structured signals → attention; peer-rank vectors → concatenation.* **Our modulator pushes the full raw observation — a high-dimensional, token-structured-ish, peer-rank signal — through the modulation path.** By this principle that is the wrong mechanism for that signal. This is now the clearest cross-paper consensus in the reviewed corpus, and it points the same way as the PAPL comparison in [§8](#8-yoon-et-al-2026--papl-phase-aware-policy-learning-for-skateboard-riding-of-quadruped-robots-via-feature-wise-linear-modulation): every paper here that modulates successfully does so with a **low-dimensional** conditioner.
- **The timestep case clarifies what modulation is *good at*, which our design has never articulated.** $\tau$ works as a modulation conditioner because it must change the network's *function* uniformly across all units. Read against that, our own six heads split cleanly: the **scalar action temperature** is a perfect fit for this description (one number, global effect, changes the policy's function), while the **per-neuron $\gamma/\beta$ driven by the full observation** is not. That is a reason to consider whether the two head families deserve the same conditioner at all — and it is a design question worth stating in our documentation even if we do not act on it.
- **The redundant timestep routing is a small, checkable curiosity.** GR00T feeds $\tau$ both through the action encoder MLP *and* through adaLN, without comment. Our architecture has a comparable redundancy — the modulator reads the observation that the encoder also reads — but ours is the object of a live concern (see [§17](#17-appendix--adjacent-evidence-from-the-six-unreviewed-pdfs)) while theirs is unremarked. The difference is that GR00T's duplicated signal is one scalar and ours is the entire observation vector.
- **Nothing here bears on our crash, our single-task setting, or our critic.** No instability is reported, there is no single-task evidence, and there is no value function. Open questions 1, 2, 3, 4 and 5 all gain nothing.

### 11.6 Appendix: Section-by-Section Backbone

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | GR00T N1 = open VLA foundation model for humanoids with a **dual-system architecture**: vision-language module (System 2) interprets, diffusion transformer (System 1) generates fluid motor actions in real time; both jointly trained end-to-end. Trained on a heterogeneous mixture of real robot trajectories, human videos and synthetic data. Deployed on the Fourier GR-1 humanoid. |
| 1 | **Introduction** | Motivation for robot foundation models. Fig. 1 introduces the **data pyramid** (web/human video base → synthetic middle → real teleoperation peak). GR00T-N1-2B is 2.2 B parameters, 1.34 B of which is the VLM; 63.9 ms to sample a 16-action chunk on an L40 in bf16. Three highlighted features: compositional System 1 / System 2 design, mixed-source pre-training strategy, massively multi-task language-conditioned policy with data-efficient post-training. |
| 2.1 | **Model Architecture** | *State and Action Encoders*: per-embodiment MLPs project varying-dimensional states/actions to a shared embedding; the action encoder also encodes the diffusion timestep. Action chunks with $H=16$. *Vision-Language Module (System 2)*: Eagle-2 (SmolLM2 + SigLIP-2), $224\times224$ images → 64 tokens/frame, **middle-layer (12th) LLM features beat final-layer**. *Diffusion Transformer Module (System 1)*: **"a variant of DiT, which is a transformer with denoising step conditioning via adaptive layer normalization"** — the corpus's entire textual evidence for this paper's modulation. Alternating cross- and self-attention à la Flamingo/VIMA. Eq. 1 flow-matching loss; Beta-distributed $\tau$; $K=4$ Euler steps. |
| 2.2 | Training Data Generation | *Latent Actions*: a VQ-VAE inverse-dynamics model over $(x_t, x_{t+H})$ frame pairs produces pseudo-action labels for human video, treated as a distinct "LAPA" embodiment; Fig. 4 shows the same latent action retrieved across robot *and* human embodiments. *Neural Trajectories*: video-generation models expand 88 h of in-house teleoperation to **827 h**. Simulation trajectories expand the middle of the pyramid. |
| 2.3 | Training Details | Pre-training on all sources with flow-matching. Post-training per embodiment with the VL backbone's **language component frozen**. Post-training with neural trajectories co-trained 1:1 with real data; actions labelled by latent or IDM models. Infrastructure: NVIDIA OSMO, H100 + Quantum-2 InfiniBand, Ray, **up to 1024 GPUs, ~50,000 H100-hours** for GR00T-N1-2B. Compute-constrained finetuning notes for a single A6000. |
| 3 | Pre-Training Datasets | 3.1 real-world (in-house GR-1 humanoid data, OpenX-Embodiment, AgiBot-Alpha); 3.2 synthetic (simulation + neural trajectories); 3.3 human video datasets. |
| 4.1–4.3 | Evaluation setup | Three simulation benchmarks (RoboCasa 24 tasks, DexMG 9, GR-1 24) at 30/100/300 demonstrations per task; real-world GR-1 tasks; batch size 128 for data-limited post-training. |
| 4.4 | **Quantitative Results** | Pre-training zero-shot: 76.6 % bimanual hand-off, 73.3 % novel object. Table 2 simulation; Table 3 real-world. Fig. 9 neural-trajectory ablations across data regimes, comparing latent versus IDM action labels. |
| 4.5 | Qualitative Results | Rollout examples. |
| 4.6 | **Limitations** | Short-horizon tabletop manipulation only; long-horizon loco-manipulation needs hardware, architecture and corpus advances; a stronger VLM backbone would help spatial reasoning; synthetic data still struggles with diversity and physical plausibility. **No architectural or stability limitations named.** |
| 5 | Related Work | Foundation models in robotics; VLA lineage. |
| 6 | Conclusions | Open release of GR00T-N1-2B, training data and benchmarks. |
| A | Contributors and Acknowledgments | The corporate author list. |
| B | Detailed Experiment Results | Tables 4–5, per-task GR00T N1 vs Diffusion Policy; Fig. 10 bar plot. |
| C | Additional Qualitative Results | Figs. 11–12 pre- and post-training rollouts. |
| D | **Hyperparameters** | Table 6, pre- and post-training. Mostly shared; smaller post-training batch sizes to avoid overfitting in data-limited settings. **No DiT block count, hidden dimension, or modulation parameters reported.** |
| E | System Design | E.1 dataset format built on LeRobot with four extensions (modality configuration file, fine-grained modality specification, multiple annotation support, rotation type specification); normalisation conventions (6-D rotation states, axis-angle rotation actions, min-max scaling). |
| F | Additional Training Details | Inverse-dynamics-model training details. |

---

## 12. Marquis & Farhood 2026 — *Hypernetwork-Conditioned Reinforcement Learning for Robust Control of Fixed-Wing Aircraft under Actuator Failures*

**PDF:** `docs/project/references/modulation_in_rl/sources/Marquis et al. 2026 - Hypernetwork-conditioned RL under actuator failures (preprint).pdf`
**arXiv:** [2604.03392v1](https://arxiv.org/abs/2604.03392), 3 Apr 2026. Dennis J. Marquis, Mazen Farhood (Kevin T. Crofton Dept. of Aerospace & Ocean Engineering, Virginia Tech).

> **The most decision-relevant paper in the corpus for questions 3 and 5, and the only one whose architecture is at our scale.** A 64×64 tanh policy with a 32×32 hypernetwork, trained with **PPO**, with FiLM applied to hidden layers and **an explicit actor-versus-critic ablation**. It also reports a modulation instability and a quantitative sensitivity analysis nobody else in the corpus attempts.

### 12.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **Unstated in PDF.** IEEE two-column conference format with no venue line, no footer, no acceptance note; only `arXiv:2604.03392v1 [eess.SY] 3 Apr 2026`. Cite as a 2026 preprint. |
| **Conditioning signal** | The **actuator-failure parameterisation** $\lambda_k \in \mathbb{R}^6$ — for each of three actuators (right aileron, left aileron, rudder), a binary "is it stuck" flag $\lambda^{\mathrm{fail}}_{i,k}$ and a stuck-deflection level $\lambda^{\mathrm{val}}_{i,k}\in[-1,1]$ as a fraction of the saturation limit. **Six dimensions, and deliberately excluded from the policy's own observation** — see §12.3. |
| **Modulated component** | **The hidden layers of the actor**, and — in the `+HC` ("hyper-conditioned critic") variants — **the hidden layers of the critic as well**, sharing the same hypernetwork conditioning. Both actor and critic are 2 hidden layers × 64 neurons with `tanh`. |
| **Granularity cell** | **(a) per-unit** ≡ (b). $p^{(\ell)}_{\mathrm{scale}}$ and $p^{(\ell)}_{\mathrm{shift}}$ are *"element-wise scaling and shifting vectors"* applied with $\odot$ to a dense layer's pre-activation — one $(\gamma,\beta)$ per hidden unit, 64 per layer. |
| **Placement** | **The hidden layers**, pre-activation, with **no normalisation layer** — $h^{(\ell+1)}_k = \sigma\bigl(p^{(\ell)}_{\mathrm{scale}} \odot (W^{(\ell)}h^{(\ell)}_k + b^{(\ell)}) + p^{(\ell)}_{\mathrm{shift}}\bigr)$, $\sigma = \tanh$. With $L$ = 2 hidden layers, that is **2 injection sites per network**, 4 with the conditioned critic. |
| **Parameterization** | **One shared hypernetwork per network**, feedforward with 2 hidden layers × 32 neurons and `tanh`, mapping $\lambda_k \mapsto$ all modulation parameters. Table IV: FiLM 23,405 parameters, FiLM+HC 31,510, versus MLP baseline 13,897 — and versus **~434k** for a full weight-generating hypernetwork ($6 \to 32 \to 32 \to 13{,}125$), *"an order of magnitude reduction"*. |
| **Initialization** | **Explicit, principled, and — uniquely in this corpus — *damped*.** §II-A: $p^{(\ell)}_{\mathrm{scale}} = 1 + 0.1\,p^{(\ell)}_{h,\mathrm{scale}}$ and $p^{(\ell)}_{\mathrm{shift}} = 0.1\,p^{(\ell)}_{h,\mathrm{shift}}$, where $p_h$ are the raw hypernetwork outputs; the LoRA output is likewise scaled by $1/n_r$. *"This parameterization serves a similar purpose to the principled initialization strategies in [13]"* (Chang et al.'s hypernetwork initialisation). **The gain is identity-centred AND attenuated by a factor of 10.** See §12.4. |
| **Ablation present?** | **Yes — three, and two of them are unique in this corpus.** (i) **FiLM vs LoRA vs plain MLP** under identical PPO training. (ii) **`+HC`: hyper-conditioning the critic, ablated separately for FiLM and for LoRA** (Table II) — the corpus's only direct actor-versus-critic modulation experiment. (iii) **LoRA rank sweep** $n_r \in \{8,16,32,48,64\}$ (Table III). Plus a Lipschitz-constant analysis (§V-C). |
| **Reported instability** | **Yes — two distinct reports, both in modulated on-policy RL.** (i) **LoRA at rank 48 "resulted in significant instability, with MaxPE exceeding 150 m in the flutter case"**, breaking an otherwise monotone rank-versus-performance trend; the authors conclude *"hypernetwork-conditioned RL remains sensitive to subtle architectural and initialization choices… minor deviations can lead to optimization instability rather than predictable performance gains."* (ii) **LoRA + conditioned critic nearly doubles error**, attributed to *"high-variance gradients or numerical instability during the iterative PPO update."* **No gain blow-up is reported** — but this is the closest thing to our failure mode anywhere in the corpus. |
| **Single-task or multi-task** | **Single task — one path-following objective — with a *within-episode* conditioning variable.** There is no task distribution and no task ID; $\lambda_k$ describes the aircraft's own failure state, which can change *during* an episode (an actuator becomes stuck at a random time $N_{\mathrm{fail}}$). **Gains do not depend on task diversity; they depend on operating-regime diversity.** |

### 12.2 Verification of the survey's claim

The survey listed this as unverified-venue entry **U1**: *"The only direct FiLM-vs-LoRA head-to-head inside on-policy RL (PPO) I found in 2024–2026. Conditioner = actuator-fault parameterisation; both are framed as parameter-efficient hypernetwork conditioning; reports generalisation to time-varying failure modes unseen in training."*

**Every clause confirmed.** FiLM and LoRA are trained under identical PPO settings against a shared MLP baseline; the conditioner is $\lambda_k$; and the headline generalisation result is to "flutter" failures — oscillatory, non-stationary faults never seen in training. Venue remains unconfirmed, correctly.

**What the survey could not know, and what makes this the corpus's most useful paper for us**, is contained in §V-A, §V-B and §V-C: an actor-versus-critic ablation, an instability report, and a Lipschitz analysis. Those are covered in §12.3–§12.5.

### 12.3 Open question 5 — the actor-versus-critic answer, and it depends on the mechanism

This is the only experiment in the reviewed corpus that isolates critic conditioning. Table II, rudder failures (the hardest mode), average Mean Path Error and Maximum Path Error in metres — **lower is better**:

| Policy | Static MPE | Static MaxPE | Flutter MPE | Flutter MaxPE |
|---|---|---|---|---|
| FiLM | 5.33 | 16.49 | 15.72 | 43.67 |
| **FiLM + HC** | **2.56** | **9.64** | **7.69** | **18.70** |
| LoRA (16) | 4.34 | 14.86 | 11.75 | 31.06 |
| LoRA (16) + HC | 10.40 | 26.68 | 18.53 | 48.66 |

**The two mechanisms respond in opposite directions.**

- **FiLM: conditioning the critic helps, a lot.** Errors fall by **~52 %** (static MPE 5.33 → 2.56), **~42 %** (static MaxPE), **~51 %** (flutter MPE) and **~57 %** (flutter MaxPE). The authors' reading: *"for FiLM, modulating the critic's activations is more effective for advantage estimation than treating $\lambda$ as a standard input feature."*
- **LoRA: conditioning the critic hurts, badly.** Errors roughly **double** across all four metrics. Diagnosis: *"simultaneously adapting both actor and critic weight matrices via low-rank updates introduces larger optimization complexity. This could lead to high-variance gradients or numerical instability during the iterative PPO update, where a standard input-feature representation of $\lambda$ proves more robust for the value function."*

**Three things follow, and they matter for us.**

1. **"Should the critic be modulated?" is not a question with a mechanism-independent answer.** Activation modulation and weight adaptation behave oppositely on the same task, same conditioner, same optimiser. Any project-level claim about critic modulation must name the mechanism.
2. **The reason FiLM-on-critic helps is specific and checkable.** PPO's advantage estimate $\hat A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots$ depends on $V(s)$, and under a failure the *value* of a state genuinely changes — a given attitude is worth less with a stuck rudder. If the critic must represent $V(s,\lambda)$ with $\lambda$ merely concatenated, it faces exactly the gradient-interference problem the paper's introduction describes: *"data collected under different dynamics lead to updates that can push shared parameters in opposing directions."* Modulation gives it per-regime gains instead. **This is the same argument PAPL makes from a different direction** ([§8](#8-yoon-et-al-2026--papl-phase-aware-policy-learning-for-skateboard-riding-of-quadruped-robots-via-feature-wise-linear-modulation) modulates the critic because the *reward* is phase-conditioned); here the *dynamics* are regime-conditioned, which makes the value function regime-conditioned by the same logic.
3. **It costs nothing at deployment.** *"hyper-conditioning the value function increases training parameters but incurs no additional runtime cost"* — the critic is discarded after training.

**Combined verdict across the corpus on question 5.** Two papers modulate the critic (this one and PAPL); both report it working; this one gives the only controlled comparison and finds a **~50 % error reduction** attributable to critic modulation alone, under PPO. The old corpus concern — Ben-Iwhiwhu et al. 2022's report that modulating the value estimator was unstable — is **not reproduced by either**, though neither paper engages with it directly.

### 12.4 Open question 3 — the closest thing to our failure mode, plus a quantitative sensitivity tool

**(i) A capacity-driven instability, non-monotonic in the capacity knob.** Table III, rudder failures:

| Policy | Static MPE | Static MaxPE | Flutter MPE | Flutter MaxPE |
|---|---|---|---|---|
| LoRA (8) | 3.39 | 13.14 | 20.75 | 60.70 |
| LoRA (16) | 4.34 | 14.86 | 11.75 | 31.06 |
| LoRA (32) | 4.24 | 14.03 | 7.09 | 21.70 |
| **LoRA (64)** | 3.52 | **9.74** | 7.47 | **20.09** |
| **LoRA (48)** | — | — | — | **">150 m" — "significant instability"** |

Rank 48 is **not in the printed table** — it is reported only in prose, as an outlier that breaks the otherwise monotone $8 \to 16 \to 32 \to 64$ improvement. The authors do not diagnose it beyond *"remains sensitive to subtle architectural and initialization choices"* and note that power-of-two ranks are conventional for hardware reasons. **This is a real, if under-characterised, instability in a modulated on-policy RL policy, triggered by a capacity hyperparameter.**

**Relevance to our $G$ sweep, stated carefully.** Our grouping size $G$ is also a capacity knob on the modulation path, and our $G=1$ (maximum capacity) run is the one that crashed. Marquis's rank sweep is the only published precedent for *"a modulation-capacity hyperparameter with non-monotone behaviour, where one setting destabilises training"*. It is **not** the same failure — theirs is a control-performance blow-up on an out-of-distribution evaluation, ours is a mid-training collapse with a railed temperature — and their conditioner and mechanism differ from ours. **Do not claim it as a precedent for our crash.** Do cite it as evidence that modulation-capacity settings can behave non-monotonically and that the field treats this as a known tuning hazard rather than a solved problem.

**(ii) The Lipschitz analysis — the corpus's only quantitative handle on modulator sensitivity.** §V-C estimates each hypernetwork's Lipschitz bound as the product of the spectral norms of its weight matrices (valid because `tanh` is 1-Lipschitz):

| Policy | Lipschitz bound $L$ | Performance |
|---|---|---|
| LoRA (8) | 28.38 | worst |
| LoRA (16) | 21.89 | ↓ |
| LoRA (32) | 11.16 | ↓ |
| LoRA (64) | **3.73** | **best** |
| FiLM + HC (main net 2×64) | 13.67 | good |
| FiLM + HC (main net 2×32) | 18.76 | *"poor control performance"* |

*"Lower values of $L$ correlate with improved tracking performance… these results motivate the use of spectral normalization to explicitly constrain sensitivity when training future controllers."* The conclusion repeats it as future work: *"we will incorporate spectral normalization to constrain hypernetwork sensitivity."*

**Why this is the single most actionable finding in the corpus for our crash.** Our $G=1$ failure was characterised by **gain variance ballooning** — i.e. the modulator's output becoming increasingly volatile with respect to its input. That is precisely what a Lipschitz constant measures. Marquis provides (a) a cheap estimator for it, (b) evidence that it correlates with performance across six configurations, and (c) a named remedy. **We can compute this on checkpoints we already have**: take the product of spectral norms of the modulator GRU's weight matrices over training, and check whether it grows before the collapse. If it does, we have a quantitative early-warning signal and a published, principled intervention (spectral normalisation on the modulator) — for a failure mode the field otherwise has no name for.

**(iii) The damped initialisation is a second, simpler prophylactic we do not use.** $p_{\mathrm{scale}} = 1 + 0.1\,p_h$ does two things at once: it **anchors the gain at 1** (like CogVLA's $(1+\gamma)$ and AdaLN-Zero) *and* **attenuates the modulation's dynamic range by 10×**. The second part is not present in any other paper here — FLOWER, CogVLA and GR00T anchor at identity but do not damp. **This is a one-line change with a direct bearing on gain-variance growth**, and it is the cheapest thing on the list of candidate interventions.

### 12.5 Open question 2 — a third routing comparison, with the conditioner deliberately separated

The baseline and the treatment differ **only in how $\lambda$ reaches the network** (§V-D):

- **MLP baseline**: a 40-dimensional observation $o_k$ that **includes $\lambda_k$** — i.e. the failure vector is concatenated onto the observation.
- **FiLM / LoRA**: inputs *"separate[d] into state (34D) and failure (6D) vectors"* — the policy reads 34 dimensions, and the 6-dimensional $\lambda$ goes only to the hypernetwork.

**Same information, two routes.** This is the third such comparison in the reviewed corpus, after Yuan 2024 ([§6](#6-yuan-2024--unpacking-the-individual-components-of-diffusion-policy)) and PAPL ([§8](#8-yoon-et-al-2026--papl-phase-aware-policy-learning-for-skateboard-riding-of-quadruped-robots-via-feature-wise-linear-modulation)), and it goes the same way — but with an important refinement: **on in-distribution failures the routes are nearly equivalent, and the gap opens only out of distribution.** Table I, worst-case errors:

| Policy | Static rudder WC | **Flutter rudder WC** | Flutter rudder MaxPE SD |
|---|---|---|---|
| MLP | 36.83 m | **159.91 m** | ≈30.44 m |
| FiLM + HC | 21.34 m | **26.73 m** | — |
| LoRA (64) | 21.55 m | **29.91 m** | ≈4.84 m |

The paper's own summary: *"For static failures, all architectures exhibit a nearly uniform and comparable error profile, indicating that the baseline MLP is generally capable of handling these types of failures. However, under flutter failures, the error profiles diverge significantly… the MLP exhibits catastrophic divergence."* The MLP's flutter variability is **over five times higher** than LoRA's.

**The extractable rule, and it echoes Yuan 2024's difficulty moderator in a new form:** *routing a conditioning signal through the modulation path rather than the input path buys little when the test conditions match training, and a great deal when they do not.* Yuan found the moderator was task **difficulty**; Marquis finds it is **distribution shift**. Both are versions of "modulation pays when the problem is hard enough for the structure to matter".

**Direct relevance.** Our single-environment comparisons came back neutral-to-negative, and our one clear win was a **continual-learning probe** — a distribution-shift setting. **Marquis is the corpus's best explanation for that pattern**: an architecture whose benefit is concentrated out-of-distribution will look worthless on an in-distribution comparison. That is a substantive, citable reframing of our null results rather than an excuse for them, because the same paper shows both halves of the pattern in one experiment.

### 12.6 Phase 1 — Foundational overview

**Introduction.** A small fixed-wing aircraft flying a planned route has three control surfaces that can fail: the two ailerons (which roll the aircraft) and the rudder (which yaws it). A failure here means a surface gets **stuck** at some deflection — not merely dead, but actively pushing the aircraft the wrong way. Each failure changes the aircraft's physics qualitatively: a jammed rudder does not just weaken yaw control, it alters how yaw, roll and sideways motion couple together.

A single neural-network controller trained on all of these situations has a problem the paper names in its introduction: **gradient interference**. Flight data from a stuck-rudder episode pushes the network's weights one way; data from a stuck-aileron episode pushes them another; and because both share the same weights, the network settles for a compromise that is mediocre everywhere. The alternative — one controller per failure mode with switching logic — needs the failure space chopped into discrete bins, and the number of bins explodes.

The proposal is to keep one controller but let a **small side-network re-tune it** based on a six-number description of the current failure. Two ways of doing that are compared: **FiLM**, which multiplies and shifts the controller's internal activations, and **LoRA**, which makes small low-rank edits to the controller's weight matrices. Both are trained together with the controller from scratch using PPO — unlike the language-model setting these techniques come from, where a pretrained model is adapted afterwards.

**Key findings.** On the failures seen in training, all three controllers — including the plain baseline — do fine. The difference appears on failures **never seen in training**: a "flutter", where the stuck surface oscillates unpredictably rather than sitting still. Here the baseline diverges catastrophically, with worst-case path errors of 160 metres and five times the variability; the conditioned controllers stay under 30 metres.

Three further results are worth more than the headline. **Also conditioning the value estimator** — the part of PPO that judges how good a situation is — halves FiLM's error, but nearly doubles LoRA's. **The amount of adaptation capacity matters non-monotonically**: LoRA improves steadily from rank 8 to 64, except at rank 48, where training destabilised entirely. And the authors find a **numerical predictor** of controller quality: a measure of how sensitive the network is to small input changes, which tracks performance across every configuration tested and suggests a concrete fix (spectral normalisation).

**Initial takeaway.** Where a conditioning signal enters a network matters mainly when the world stops matching the training set. And two superficially similar ways of conditioning a network — scaling its activations versus editing its weights — can behave in opposite directions when applied to the same component.

### 12.7 Phase 2 — Graduate-level deep dive

#### 12.7.1 The two adaptation mechanisms, side by side

The unadapted main network is a plain MLP with `tanh`:

$$
h^{(\ell+1)}_k = \sigma\bigl(W^{(\ell)} h^{(\ell)}_k + b^{(\ell)}\bigr), \quad \ell = 0,\dots,L-1, \qquad h^{(0)}_k = o_k,\ a_k = h^{(L)}_k .
$$

**FiLM** modulates the activations:

$$
h^{(\ell+1)}_k = \sigma\Bigl(p^{(\ell)}_{\mathrm{scale}} \odot \bigl(W^{(\ell)}h^{(\ell)}_k + b^{(\ell)}\bigr) + p^{(\ell)}_{\mathrm{shift}}\Bigr),
$$

**LoRA** modulates the weights:

$$
W^{(\ell)} \;\longrightarrow\; W^{(\ell)} + U^{(\ell)}\,\mathrm{diag}\bigl(r^{(\ell)}\bigr)\,V^{(\ell)\top}, \qquad U^{(\ell)}\in\mathbb{R}^{n_{\ell+1}\times n_r},\ V^{(\ell)}\in\mathbb{R}^{n_\ell\times n_r},
$$

where **only $r^{(\ell)} \in \mathbb{R}^{n_r}$ is generated by the hypernetwork** — $U$ and $V$ are ordinary learned parameters. This is a neat and under-appreciated design: the hypernetwork emits just $n_r$ numbers, and the conditioning acts by **re-weighting a learned basis of rank-1 weight perturbations**.

**The algebraic relationship, which the paper does not spell out but which explains its results.** Both mechanisms modify the layer's Jacobian, but with different structure:

$$
\frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}}\Bigg|_{\mathrm{FiLM}} = \mathrm{diag}(\sigma')\,\mathrm{diag}\bigl(p_{\mathrm{scale}}\bigr)\,W^{(\ell)}, \qquad
\frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}}\Bigg|_{\mathrm{LoRA}} = \mathrm{diag}(\sigma')\,\bigl(W^{(\ell)} + U\,\mathrm{diag}(r)\,V^\top\bigr).
$$

FiLM's conditioning is a **diagonal rescaling of the output space** — it can amplify or suppress each unit, but cannot change *which input directions map to which output directions*. LoRA's conditioning is a **rank-$n_r$ additive perturbation of the map itself** — it can rotate and mix directions. LoRA is strictly more expressive at equal parameter count in the conditioner, which is why it eventually wins at rank 64 (MaxPE 9.74 vs FiLM+HC's 9.64 — essentially tied) but is also more fragile, which is the paper's rank-48 and `+HC` findings.

**Parameter counts (Table IV)** confirm both are cheap:

| Policy | Parameters | Train time/iter | FLOPs |
|---|---|---|---|
| MLP | 13,897 | 24.0 s | 14k |
| FiLM | 23,405 | 28.7 s | 32k |
| **FiLM + HC** | 31,510 | 23.8 s | 32k |
| LoRA (16) | 19,629 | 28.8 s | 26k |
| LoRA (16) + HC | 23,958 | 33.7 s | 26k |
| LoRA (64) | 33,645 | 30.8 s | 57k |

*"Training time per iteration remains consistent across architectures, indicating that computational cost is dominated by environment simulation and rollout collection rather than network size."* Inference is $10^4$–$10^5$ FLOPs against a Raspberry Pi's $\sim\!10^9$ FLOP/s at a 25 Hz control rate — several orders of margin.

#### 12.7.2 The damped identity-centred parameterisation

Stated in §II-A and easy to miss:

$$
p^{(\ell)}_{\mathrm{scale}} = 1 + 0.1\,p^{(\ell)}_{h,\mathrm{scale}}, \qquad p^{(\ell)}_{\mathrm{shift}} = 0.1\,p^{(\ell)}_{h,\mathrm{shift}}, \qquad r^{(\ell)} \;\text{scaled by}\; 1/n_r,
$$

with $p_h$ the raw hypernetwork outputs. **Two separate design decisions are compressed into this line.**

1. **Anchoring at identity.** With $p_h \approx 0$ at initialisation, $p_{\mathrm{scale}} \approx 1$ and $p_{\mathrm{shift}} \approx 0$, so the adapted network starts as the unadapted one. This is the same property as AdaLN-Zero (FLOWER, GR00T) and CogVLA's $(1+\gamma)$ form.
2. **Damping the dynamic range.** The factor $0.1$ means that for the gain to reach 2, the hypernetwork's raw output must reach 10. **The modulation is confined to a narrow band around identity unless the optimiser works hard to escape it.** This is a *soft constraint on gain magnitude that persists throughout training*, not just at initialisation — and it is the only such device in the reviewed corpus.

The LoRA analogue $1/n_r$ is the standard scaling that keeps the perturbation's magnitude independent of rank, so that increasing $n_r$ adds expressiveness without inflating the update norm.

**This is the corpus's most direct answer to "what would we do differently".** Our modulator has no equivalent damping factor. Adding one is a single hyperparameter, is trivially reversible, and directly targets the gain-variance growth observed in the $G=1$ crash.

#### 12.7.3 Training and evaluation protocol

**PPO** via Stable-Baselines3 + PyTorch, on Virginia Tech's ARC cluster, with rollouts from multiple parallel environments. Training episodes mix (i) nominal, (ii) a single actuator stuck for the whole episode, (iii) a single actuator becoming stuck at a random time $N_{\mathrm{fail}}$ and remaining so.

**Deliberate train/test asymmetry — three levels, and this is what makes the paper's evidence unusually clean:**

| Level | Stuck-deflection values | Purpose |
|---|---|---|
| Training | $\Lambda_{\mathrm{train}} = \{0, \pm0.25, \pm0.5\}$ | 5 discrete levels per actuator |
| Static evaluation | $\Lambda_{\mathrm{eval}} = \{0, \pm0.125, \pm0.25, \pm0.375, \pm0.5\}$ | adds **interpolation** points |
| Flutter evaluation | $\lambda^{\mathrm{val}}_{i,k}$ varies piecewise-constant in $[\bar\lambda^{\mathrm{val}}_i - 0.2,\ \bar\lambda^{\mathrm{val}}_i + 0.2]$, held 0.2–1.0 s, over a 1–10 s episode | **out-of-distribution non-stationarity** |

Training deliberately excludes saturation-level failures because they render path-following *"dynamically infeasible"*. Evaluation is **1,000 episodes per set**, scored by Mean Path Error and Maximum Path Error.

The simulation is a high-fidelity six-degree-of-freedom CZ-150 model with Gaussian sensor noise on every channel, steady wind of 3–5 m/s at random heading, Dryden-model gusts at a 30-knot reference, up-to-one-step control delay, and stochastically perturbed aerodynamic coefficients, integrated by RK4 at $\Delta t = 0.04$ s. **This is a genuinely stochastic, partially observed control problem, not a toy.**

#### 12.7.4 The qualitative failure mode

§V-E contrasts worst-case rudder-flutter episodes under identical wind and noise. The MLP *"exhibits excessive bank and pitch angles, leading to a significant altitude gain of nearly 40 m and substantial path deviation"*, and critically *"lack[s] the aggressive aileron compensations observed in the FiLM + HC policy, which instead utilizes roll-to-yaw coupling to maintain stability."*

**The conditioned policy learned a qualitatively different control strategy** — using ailerons to generate yaw via roll-yaw coupling, compensating for a rudder it knows is unreliable. That is the behavioural signature the paper's whole architecture argument predicts: not "the same policy, executed better", but "a different policy, selected by the conditioner". The paper also notes, fairly, that the MLP *"demonstrates the ability to effectively recover to the reference path once the rudder returns to nominal operation"* — its failure is transient, not terminal.

### 12.8 Relevance to this project

**The most transferable paper in the corpus. Same algorithm (PPO), same scale (tens of thousands of parameters), same modulation form (per-unit FiLM on dense pre-activations with no normalisation layer), single task, within-episode conditioner.**

- **It gives us the actor/critic answer we lacked, and it is positive for FiLM.** Modulating the critic as well as the actor cut errors by roughly half under PPO. Our architecture modulates the encoder and the policy GRU but **not** the critic. This is a concrete, evidenced candidate change — and one whose cost is training-time only. **PAPL supplies the "when" criterion** (modulate the critic when the value function genuinely varies with the conditioner) and **Marquis supplies the "how much" evidence**. Both point the same way; the intervening question for us is whether our value function actually varies with the modulator's input, which is a cheap diagnostic (see [§8](#8-yoon-et-al-2026--papl-phase-aware-policy-learning-for-skateboard-riding-of-quadruped-robots-via-feature-wise-linear-modulation)).
- **It gives us a quantitative instrument for the crash, computable on existing checkpoints.** The Lipschitz bound — the product of spectral norms of the modulator's weight matrices — correlates with performance across all six of their configurations. **Computing it over training on our $G=1$ run is a post-hoc analysis requiring no retraining**, and if it grows monotonically before the collapse we have both a named diagnostic and a published remedy (spectral normalisation) for a failure mode the field otherwise has no vocabulary for. This is the highest-value follow-up the corpus has produced.
- **It gives us the cheapest candidate intervention: damp the modulation.** $p_{\mathrm{scale}} = 1 + 0.1\,p_h$ constrains gains to a narrow band around identity *throughout* training, not just at step 0. Nothing else in the corpus does this. Against a failure characterised by ballooning gain variance, it is the obvious first thing to try.
- **It reframes our null results without excusing them.** In-distribution, the plain MLP baseline matched the conditioned policies; the gap appeared only under distribution shift. Our single-environment comparisons were in-distribution and came back neutral; our one clear win was a continual-learning probe, which is a distribution-shift test. **Marquis exhibits exactly that pattern within a single controlled experiment**, which makes it a citable explanation rather than a post-hoc story. It also implies our evaluation protocol, not our architecture, may be what needs changing first.
- **It is a fourth independent instance of a low-dimensional conditioner.** $\lambda_k$ is 6-dimensional and **explicitly withheld from the policy's observation** (34D state to the policy, 6D failure to the hypernetwork). Every successful modulation paper in this corpus separates the conditioner from the modulated stream and keeps it small. **Our design does neither.**
- **The capacity non-monotonicity is a caution for the $G$ sweep, not a precedent for the crash.** Rank 48 destabilising while 32 and 64 are fine is a warning that modulation-capacity sweeps can have isolated bad settings, and that a single anomalous point should not be over-interpreted in either direction. Applied to our sweep: a bad $G$ value is not automatically evidence of a trend.
- **Honest limits.** Different domain (fixed-wing flight dynamics), feedforward rather than recurrent policy, no temperature head, a conditioner that is a clean external fault descriptor rather than a raw observation, and a preprint with no seed counts or confidence intervals reported for any table. The evaluation is 1,000 episodes per set, which is substantial, but **the number of training seeds is never stated** — so the rank-48 instability could in principle be a single unlucky seed, and the paper does not rule that out.

### 12.9 Appendix: Section-by-Section Backbone

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | RL path-following controller for a fixed-wing sUAS robust to actuator failures, conditioned on a fault parameterisation via hypernetwork adaptation. Parameter-efficient FiLM and LoRA formulations trained with PPO. Hypernetwork-conditioned policies improve robustness over standard MLPs and **generalise to time-varying failure modes not seen in training**. Validated in high-fidelity 6-DoF simulation. |
| I | **Introduction** | Names **gradient interference** as the core problem: *"data collected under different dynamics lead to updates that can push shared parameters in opposing directions, leading to overly conservative solutions, overfitting to dominant regimes, or unstable training."* Rejects switched-MLP designs because the failure space must be discretised. Positions hypernetworks as mapping scheduling variables to main-network parameters, *"enabling the representation of a family of specialized controllers rather than a single static policy."* Notes FiLM and LoRA come from LLM adaptation and that *"their use as a mechanism for control policy conditioning in RL remains relatively unexplored."* Four contributions, including *"the benefit of value-function conditioning for FiLM."* |
| II-A | **Hypernetwork-Based Policy Adaptation** | The three equations: unadapted MLP, FiLM modulation, LoRA weight update. **The damped identity-centred parameterisation** $p_{\mathrm{scale}} = 1+0.1p_h$, $p_{\mathrm{shift}} = 0.1p_h$, LoRA scaled by $1/n_r$. Emphasises joint end-to-end training with PPO, *"in contrast to large language model settings where adaptation is typically applied to a pre-trained network."* |
| II-B | sUAS Dynamics | Six-DoF nonlinear CZ-150 model: inertial position, body-frame linear/angular velocities, Euler attitude, flight-path and course angles. Wind-relative velocity. First-order actuator dynamics for elevator, aileron, rudder, throttle. Aerodynamic force/moment coefficients. **Individual left/right aileron deflections $\delta_{Al}, \delta_{Ar}$** introduced so asymmetric faults can be modelled, with effective deflection $\delta^{\mathrm{eff}}_A = \tfrac12(\delta_{Al}+\delta_{Ar})$ and an asymmetric pitching term with coefficient $-0.02$. Virtual-vehicle formulation for path following. |
| III-A | Simulation Environment | Custom Python sim with Gaussian sensor noise on every channel (specific SDs listed), steady wind 3–5 m/s at random heading, Dryden gusts at 30-knot reference, up-to-one-step control delay, stochastically perturbed aerodynamic coefficients. RK4 at $\Delta t = 0.04$ s. |
| III-B | **Failure Parameterization** | Faults as stuck deflections. $\lambda_k = [\lambda^{\mathrm{fail}}_{Ar}, \lambda^{\mathrm{val}}_{Ar}, \lambda^{\mathrm{fail}}_{Al}, \lambda^{\mathrm{val}}_{Al}, \lambda^{\mathrm{fail}}_{R}, \lambda^{\mathrm{val}}_{R}]^\top \in \mathbb{R}^6$, with $\delta^{\mathrm{stuck}}_{i,k} = \lambda^{\mathrm{val}}_{i,k}\delta^{\mathrm{sat}}_i$ and $\lambda^{\mathrm{val}}_{i,k}\in[-1,1]$. |
| III-C | Action and Observation Spaces | Observation split into 34D state and 6D failure vector for conditioned policies; 40D combined for the MLP. |
| III-D | Reward Design | Path-following reward with a 25 m early-termination threshold. |
| III-E | Reference Path Generation | — |
| IV-A | **Controller Architectures** | Main network and value network: **2 hidden layers × 64 neurons, `tanh`**. Hypernetwork: **2 hidden layers × 32 neurons, `tanh`**. LoRA ranks $n_r \in \{8,16,32,48,64\}$. **`+HC` notation defined**: the value network shares the same hypernetwork-based conditioning as the policy. |
| IV-B | Training Configuration | PPO via Stable-Baselines3 + PyTorch on the ARC cluster. Episode mixture: nominal / fully stuck / stuck at random onset. $\Lambda_{\mathrm{train}} = \{0,\pm0.25,\pm0.5\}$, restricted to keep the task dynamically feasible. |
| IV-C | Evaluation Protocol | **1,000 episodes per set.** Mean Path Error and Maximum Path Error. Static evaluation on the interpolated $\Lambda_{\mathrm{eval}}$; **flutter** evaluation with piecewise-constant oscillatory faults of 1–10 s duration in a $\pm0.2$ band, values held 0.2–1.0 s. |
| V | **Results** | Table I: static vs flutter, per actuator, MPE / MaxPE / worst-case. All architectures comparable on static; **MLP diverges catastrophically on flutter (rudder WC 159.91 m, SD ≈30.44 m vs LoRA's ≈4.84 m)**. Fig. 1 error profiles across deflection magnitudes. Attributes MLP failure to overfitting the static training dynamics plus a training-exposure bias (aileron failures seen twice as often as rudder). |
| V-A | **Effectiveness of Hyper-Conditioned Value Functions** | Table II. **FiLM+HC improves 40–50 %; LoRA+HC nearly doubles error.** Mechanism-dependent, with the high-variance-gradient / numerical-instability diagnosis for LoRA. |
| V-B | **LoRA Rank Sensitivity** | Table III, ranks 8→64 monotone improvement on flutter; **rank 48 an outlier with MaxPE > 150 m, "significant instability"**. Rank selection *"acts as a critical tuning parameter."* |
| V-C | **Lipschitz Characterization** | Spectral-norm product estimate ( `tanh` is 1-Lipschitz). LoRA $L$ falls 28.38 → 3.73 as rank rises; FiLM+HC $L=13.67$ vs a smaller main network's $18.76$ with worse performance. **Motivates spectral normalisation.** |
| V-D | Computational Considerations | Table IV parameter counts, training time, FLOPs. <35k parameters vs ~434k for full weight generation. **Critic conditioning costs training parameters but no runtime.** |
| V-E | Simulation Example | Worst-case rudder-flutter comparison: MLP gains ~40 m altitude with excessive bank/pitch; **FiLM+HC uses roll-to-yaw coupling** to compensate for the unreliable rudder. MLP recovers once the fault clears. |
| VI | **Conclusion** | Conditioning on fault parameters improves robustness to static and non-stationary failures; performance is sensitive to LoRA rank and to critic conditioning. Future work: **spectral normalisation to constrain hypernetwork sensitivity**, plus flight tests. |

---

## 13. Kang et al. 2026 — *SplitAdapter: Load-Aware Humanoid Loco-Manipulation via Factorized Adaptation*

**PDF:** `docs/project/references/modulation_in_rl/sources/Kang et al. 2026 - SplitAdapter - two-source factorised FiLM (preprint).pdf`
**arXiv:** [2606.03297v1](https://arxiv.org/abs/2606.03297), 2 Jun 2026.

### 13.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **Unstated in PDF.** CoRL-style formatting with a `Keywords:` line but no venue; only `arXiv:2606.03297v1 [cs.RO] 2 Jun 2026`. Cite as a 2026 preprint. |
| **Conditioning signal** | **Two factorised context latents, deliberately separated.** (i) **Object/load context** $(z_{\mathrm{obj},t}, m_{\mathrm{est},t})$ — a 16-D latent plus an *online-estimated object mass and loaded-state probability*. (ii) **Dynamics context** $z_{\mathrm{dyn},t}$ — a 16-D latent trained by robot-transition prediction. Both are inferred from a **50-step history of 123-D observations** through a shared Conv1d temporal encoder, then split into two branches. |
| **Modulated component** | A **frozen** pretrained PhysHSI-style AMP box-manipulation policy (actor MLP `[512,256,256]`). The base policy's weights never change; only the adapters train, using *"the same reinforcement-learning objective and reward as the base policy."* |
| **Granularity cell** | **(a)/(b) per-feature.** Eq. 14: $h' = \gamma(z)\odot h + \beta(z)$, described as *"feature-wise scale and bias terms"*. FiLM hidden dimension 128, generator hidden size 128. |
| **Placement** | **The contribution, and the corpus's only depth-assigned conditioner split.** Object/load FiLM modulates **earlier layers** *"to shape coarse task-level behavior, such as payload-conditioned lifting posture and whole-body coordination"*; dynamics-aware FiLM modulates **later layers closer to the action head** *"to compensate for dynamics and environment mismatch."* Described as *"hierarchical"* and *"layer-dependent"*, with **soft-split routing** (Table 5). |
| **Parameterization** | Two separate generators, one per context branch, each fed by its own encoder head (object branch `Linear(320,64)→Linear(64,64)`; dynamics branch `Linear(320,64)→Linear(64,16)`) off a shared Conv1d temporal encoder. |
| **Initialization** | **Explicit identity-init, stated in one clean sentence:** *"The modulation layers are initialized to preserve the original frozen-policy behavior at the beginning of training."* No ablation of this choice. |
| **Ablation present?** | **Yes — the modulation structure is ablated.** Table 1 (MuJoCo sim-to-sim, 90 trials per method = 3 masses × 3 heights × 10 trials), Full-task / Lift-up success: **SplitAdapter 86/90, 90/90**; `w/o hierarchical FiLM` **81/90, 90/90**; `w/o split latent` 81/90, 88/90; `w/o GRL` 84/90, 90/90; `AnyAdapter-style WM-FiLM` (unified latent) 75/90, 86/90; frozen base policy 71/90, 86/90. **Caveat: the paper never states whether `w/o hierarchical FiLM` removes FiLM entirely or merely removes the depth assignment.** |
| **Reported instability** | **None.** No NaN, no gain blow-up, no entropy or plasticity discussion. Limitations (§6) name only rigid-object scope and the limited scale of real-world testing (the Unitree G1's ~3 kg per-arm payload spec makes repeated 6 kg trials hard on the hardware). |
| **Single-task or multi-task** | **Single task** — box lift, transport and placement — with **within-episode** context variation. Conditioners are estimated online and change during the episode (Fig. 4 shows mass and loaded-probability estimates converging after lift-off). Object masses 2/4/6 kg and heights 0/30/60 cm are *conditions*, not tasks. |

### 13.2 Verification of the survey's claim

The survey listed this as unverified entry **U2**: *"Hierarchical FiLM driven by two factorised context encoders (object/load vs. robot dynamics) instead of one; explicitly benchmarks against 'world-model FiLM baselines'."* **All three clauses confirmed** — the two encoders, the hierarchical application, and the `AnyAdapter-style WM-FiLM` baseline (75/90 versus SplitAdapter's 86/90).

### 13.3 The finding that matters — conditioners assigned to depths by abstraction level

Every other paper in this corpus either uses one conditioner everywhere (FLOWER, GR00T, PAPL, Marquis, EquAct) or uses different conditioners in different *subsystems* (CogVLA: language in the encoder, instruction in the LM). **SplitAdapter is the only one that splits *one network's depth* between two conditioners**, and assigns them by what each level of representation is for:

$$
h'_\ell = \begin{cases} \gamma_{\mathrm{obj}}\bigl(z_{\mathrm{obj},t},\, m_{\mathrm{est},t}\bigr)\odot h_\ell + \beta_{\mathrm{obj}}(\cdot), & \ell \text{ early — coarse task-level behaviour}\\[4pt] \gamma_{\mathrm{dyn}}\bigl(z_{\mathrm{dyn},t}\bigr)\odot h_\ell + \beta_{\mathrm{dyn}}(\cdot), & \ell \text{ late — dynamics and environment mismatch.}\end{cases}
$$

**The reasoning, and why it is more than a heuristic.** Early layers of a control policy encode *what to do* (posture, coordination strategy); late layers encode *how to actuate it* (compensating for the specific robot's dynamics). Payload mass changes the former — a 6 kg box requires a different lifting posture. Actuator mismatch changes the latter — the same intended posture requires different torques. **Assigning each conditioner to the depth whose representation it actually governs is a principled use of the Axis-2 placement dimension**, and it is the first instance in this reference library.

**The evidence is real but modest.** Removing the hierarchy costs 5/90 Full-task successes (86 → 81) while **Lift-up success is unchanged at 90/90**. The paper reads this correctly and without overclaiming: *"hierarchical FiLM is not critical for initiating the lift itself, but is beneficial for subsequent whole-body coordination during transport and placement under changing load conditions."* The dissociation between the two metrics is the most informative part — the modulation helps precisely where sustained, load-dependent whole-body coordination is required, not where a one-shot behaviour is required.

**Two caveats on citation.** (i) The ablation's meaning is ambiguous, as noted above — "without hierarchical FiLM" may or may not retain non-hierarchical FiLM, and the difference matters for what the 5-point gap measures. (ii) There are no seeds or confidence intervals; 86/90 versus 81/90 over 10 trials per cell is a small margin. The **real-world** result is stronger in kind if not in volume: 26/27 Full-task and 27/27 Lift-up on a Unitree G1 under zero-shot deployment.

### 13.4 Phase 1 — Foundational overview

**Introduction.** A humanoid robot picking up a box faces two separate problems that feel like one. The **box** changes: a 6 kg load needs a different posture and different whole-body coordination than a 2 kg one. The **robot** changes: real actuators do not behave like simulated ones, so a policy trained in simulation misbehaves on hardware. Existing adaptation methods watch the recent history of sensor readings and compress everything they infer into a single summary vector — which blends "the box is heavy" together with "my knee actuator is weaker than the simulator thought". The paper's claim is that blending them is what breaks down under heavy loads.

SplitAdapter's answer is to **freeze the existing controller entirely** and bolt on two separate estimators: one that works out the object's mass and whether it is currently being carried, and one that works out how the robot's own dynamics differ from expectation. Each estimator produces its own summary, and each is trained to predict a different thing — the object's motion versus the robot's motion. An adversarial trick (a gradient reversal layer) actively discourages each branch from encoding the other's information.

The two summaries then steer the frozen controller at **different depths**: the box-related one adjusts the early layers, where the overall movement strategy lives, and the robot-related one adjusts the late layers, just before the motor commands come out.

**Key findings.** Across 90 simulated trials spanning three box weights and three pickup heights, the full method completes the whole sequence 86 times against the frozen baseline's 71 and a single-summary adapter's 75. The gap widens with weight — at 6 kg, where the baseline succeeds 5 times out of 10 at floor height, SplitAdapter succeeds 10 times. On a real Unitree G1 robot with no fine-tuning it completes 26 of 27 trials.

**Initial takeaway.** When a controller must adapt to two genuinely different kinds of change, keeping their descriptions separate — and delivering each one to the part of the network it actually concerns — beats merging them into one signal.

### 13.5 Phase 2 — Graduate-level deep dive

#### 13.5.1 The factorisation and how it is enforced

The architecture has three enforcement mechanisms, escalating in strength:

**(i) Architectural split.** A shared temporal encoder — `Conv1d(123, 64, k=6, s=2) → ELU → Conv1d(64, 32, k=4, s=2) → ELU`, over a 50-step history of 123-D observations, producing a 320-D shared feature — feeds two separate MLP branches producing $z_{\mathrm{obj}} \in \mathbb{R}^{16}$ and $z_{\mathrm{dyn}} \in \mathbb{R}^{16}$.

**(ii) Split world-model objectives.** Each branch is trained to predict a *different* transition target: an object world model (`MLP[64,64]`, object delta target) for the object branch, and a robot world model (`MLP[128,128]`, 79-D robot transition target) for the dynamics branch. **This is the primary specialisation pressure** — each latent must be sufficient for its own prediction problem.

**(iii) Gradient-reversal cross-adversarial regularisation.** Eq. 13 makes the object latent *adversarially useless* for predicting robot dynamics:

$$
\hat s^{\mathrm{adv}}_{t+1} = A_{\mathrm{robot}\leftarrow\mathrm{obj}}\bigl(s_t,\, a_t,\, \mathrm{GRL}(z_{\mathrm{obj},t}),\, m_{\mathrm{est},t}\bigr),
$$

where $\mathrm{GRL}$ is the identity forward and negates the gradient backward. *"This discourages $z_{\mathrm{obj},t}$ from encoding robot dynamics and environment mismatch."* GRL coefficient 0.1, weight $10^{-3}$, warmup 10,000 steps.

**How the paper verifies the factorisation worked** — and this is the methodologically strongest part of the paper:

- **Fig. 5(a)**: the factorised design yields *lower world-model transition losses for both targets*, i.e. splitting helps each branch do its own job better, not merely differently.
- **Fig. 5(b)**: a **predictability gap** metric, defined as *"the difference in probe regression $R^2$ between the matched branch-target pair and the cross-branch pair for each transition target."* Larger gaps mean each branch explains its assigned transition more strongly than the other does. GRL widens the gap.
- **Fig. 3(a)**: t-SNE of $[z_{\mathrm{obj}}; z_{\mathrm{dyn}}]$ shows *"clearer mass-dependent organization in the latent space"* than the unified-latent baseline.

**This probe-based verification is a technique worth borrowing.** It answers "did the conditioner learn what we intended?" with a measurement rather than an assertion — which is exactly what our own modulator analysis lacks.

#### 13.5.2 Results

Table 1, MuJoCo sim-to-sim, Full-task success counts out of 10 per cell:

| Method | 2 kg (0/30/60 cm) | 4 kg | 6 kg | **Full-task** | **Lift-up** |
|---|---|---|---|---|---|
| Base (PhysHSI, frozen) | 10 / 8 / 9 | 9 / 10 / 10 | 5 / 5 / 5 | 71/90 | 86/90 |
| AnyAdapter-style WM-FiLM (unified latent) | 10 / 10 / 9 | 9 / 9 / 10 | 4 / 6 / 8 | 75/90 | 86/90 |
| w/o split latent | 10 / 9 / 10 | 9 / 10 / 9 | 6 / 9 / 9 | 81/90 | 88/90 |
| **w/o hierarchical FiLM** | 10 / 10 / 10 | 9 / 10 / 9 | 7 / 8 / 8 | **81/90** | 90/90 |
| w/o GRL | 10 / 8 / 10 | 10 / 10 / 10 | 8 / 9 / 9 | 84/90 | 90/90 |
| **SplitAdapter** | 10 / 10 / 10 | 9 / 10 / 10 | **10 / 9 / 9** | **86/90** | **90/90** |

**The structure of the result is more informative than the totals.** At 2 kg and 4 kg every method is near-saturated; the entire separation happens at 6 kg — *beyond the training range* — where the base policy drops to 5/10 and SplitAdapter holds 10/9/9. The paper is explicit that this is a **transport-phase** failure, not a lifting failure: *"although Lift-up success often remains high, the base policy and the AnyAdapter-style WM-FiLM baseline frequently lose postural stability during transport and fail before completing placement."*

This is the same shape as Marquis ([§12](#12-marquis--farhood-2026--hypernetwork-conditioned-reinforcement-learning-for-robust-control-of-fixed-wing-aircraft-under-actuator-failures)) and Yuan ([§6](#6-yuan-2024--unpacking-the-individual-components-of-diffusion-policy)): **in-distribution the conditioning buys nothing; the benefit is concentrated where the condition exceeds what was trained on.**

### 13.6 Relevance to this project

- **The depth-assignment idea is directly applicable and we currently ignore it.** Our modulator drives a unimodal encoder stage, a multimodal hub, a GRU update-gate bias and a temperature — four sites at different depths, **all fed by the same undifferentiated conditioner** (the raw observation through one 16-unit GRU). SplitAdapter's principle says each site should receive the aspect of the context that governs *that level of representation*. A concrete analogue: an interoceptive/injury signal driving the early encoder (which perceptual channels to weight), and a separate arousal/urgency signal driving the temperature (how decisively to act). **This would require a `senior-developer` issue plan** — it changes the modulator's head structure, not just a config value.
- **The probe-based verification of factorisation is a method we should copy.** The predictability-gap measure ($R^2$ of a probe regressor from each latent to each target) is cheap and directly answers whether our modulator's hidden state encodes anything distinguishable from what the encoder already has. **This is post-hoc analysis on existing checkpoints.**
- **Identity-initialisation of the modulation layers, stated plainly.** Fifth independent instance in the corpus (with FLOWER, CogVLA, GR00T's lineage, and Marquis). At this point the practice is unanimous among the reviewed papers, and our failure to confirm it in our own code is the corpus's most consistent implied criticism.
- **The frozen-base-policy setting limits transfer.** SplitAdapter adapts a *fixed* policy; the modulation cannot co-adapt with the features it modulates. Our modulator trains jointly from scratch, which is a harder optimisation problem and the one in which our crash occurred. **Do not cite SplitAdapter's stability as evidence for ours.**
- **Small-sample caveat.** No seeds, no confidence intervals, 10 trials per cell. The 6 kg column carries the paper.

### 13.7 Appendix: Section-by-Section Backbone

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | Humanoid loco-manipulation needs stable whole-body control under varying masses and heights; existing history-based adapters compress load and dynamics into one latent, weakening heavy-load robustness. SplitAdapter freezes a pretrained box-manipulation policy and adds object/load and dynamics-aware context encoders trained with **split world-model objectives, GRL-based cross-adversarial regularisation, and hierarchical FiLM**. Improves Full-task success over base and world-model-FiLM baselines across 2/4/6 kg and 0/30/60 cm, with the largest gains under heavy load. |
| 1 | Introduction | Sim-to-real transfer fails under payload variation *and* dynamics mismatch, which interact during contact. Prior adaptation (extrinsics estimation, privileged-state reconstruction, world-model prediction) mixes the two. Three contributions: the frozen-policy adaptation framework, the factorised structure, and the hierarchical modulation. |
| 3.1–3.4 | Factorized context encoders and split world models | Shared temporal encoder → object/load branch (estimates mass $m_{\mathrm{est}}$ and loaded state) and dynamics branch (robot-transition prediction). **Eq. 13** the GRL cross-adversarial objective. |
| 3.5 | **Hierarchical FiLM-Based Feature Modulation** | **Eq. 14** $h' = \gamma(z)\odot h + \beta(z)$. Object/load FiLM → earlier layers; dynamics FiLM → later layers near the action head. *"The modulation layers are initialized to preserve the original frozen-policy behavior at the beginning of training."* Adapter training uses the base policy's own RL objective and reward. |
| 4 | Experimental Results | Three questions: does factorised adaptation help under payload/height variation; does factorising beat a unified latent; does it improve sim-to-real under heavy load. |
| 4.1 | Simulation Evaluation | Isaac Gym training, MuJoCo sim-to-sim verification. Six methods compared. Table 1. Fig. 3 t-SNE and trajectory comparisons; Fig. 4 online mass/loaded-state estimation converging after lift; Fig. 5 world-model losses and GRL predictability gaps. |
| 4.2 | Real-World Evaluation | Unitree G1, zero-shot, 9 conditions × 3 trials. **SplitAdapter 26/27 Full-task, 27/27 Lift-up.** |
| 5 | Conclusion | Recaps the four components. |
| 6 | **Limitations** | Rigid boxes only, no deformables or broader contact-rich manipulation; limited real-world scale because the **G1's arm payload is ~3 kg per arm**, making repeated heavy-load testing hard on hardware. |
| Tables 4–5 | Hyperparameters and architecture | PPO-style: $\gamma = 0.99$, GAE $\lambda = 0.95$, desired KL 0.003, actor/critic `MLP[512,256,256]`. Adapter: 50-step history of 123-D observations, **FiLM hidden dimension 128**, latents 16/16, GRL coefficient 0.1 / weight $10^{-3}$ / warmup 10,000, **FiLM generator hidden size 128 with soft-split routing**. |

---

## 14. Guo et al. 2026 — *GEAR: Multi-Task Reinforcement Learning of Drone Aerobatics by Exploiting Geometric Symmetries*

**PDF:** `docs/project/references/modulation_in_rl/sources/Guo et al. 2026 - GEAR - drone aerobatics (preprint).pdf`
**arXiv:** [2602.10997v1](https://arxiv.org/abs/2602.10997), 11 Feb 2026.

> **This paper contains the corpus's only independent test of EquAct's constrained iFiLM, in a reinforcement-learning setting — and iFiLM loses.** That directly qualifies the grouped-modulation finding in [§9](#9-zhu-et-al-2025--equact-an-se3-equivariant-multi-task-transformer-for-open-loop-robotic-manipulation). See §14.3.

### 14.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **Unstated in PDF.** IEEE conference format, no venue line; only `arXiv:2602.10997v1 [cs.RO] 11 Feb 2026`. Cite as a 2026 preprint. |
| **Conditioning signal** | The **task command** $c$ — which aerobatic primitive to execute (Hover, Flip, Roll, Rotate) plus its motion parameters. Scalar command inputs are explicitly assigned to the trivial (rotation-invariant) representation. |
| **Modulated component** | **Both the actor and the critic.** Fig. 1 caption: *"The policy consists of an actor and a multi-head critic, both combining FiLM layers with an Equivariant MLP (EMLP)."* The actor is SO(2)-equivariant; the critic has a shared equivariant backbone plus **separate value heads per task**. |
| **Granularity cell** | **(a)/(b) per-feature.** Eq. 17: $\mathrm{FiLM}(x,c) = x\,\gamma(c) + \beta(c)$, with $x$ the state features. |
| **Placement** | **Before the group-representation grouping**, and this is load-bearing: *"FiLM modulation is applied before the features are organized into group representations"*, and *"FiLM layers applied before grouping allow controlled symmetry breaking when task asymmetries are present."* Fig. 1(a) shows two FiLM layers interleaved with two Equivariant MLP blocks. |
| **Parameterization** | A "FiLM generator" per site producing $(\gamma, \beta)$ from the command features (Fig. 1(b)). No parameter count given. |
| **Initialization** | **Not stated.** No zero-init or identity-init mentioned. |
| **Ablation present?** | **Yes — a near-complete 2×2 factorial plus a mechanism substitution, over two backbones.** Table I crosses {Backbone Only, FiLM Only, Multi-Head Only, Multi-Head+FiLM} against {MLP, EMLP} backbones, adds **Multi-Head+IFiLM** (EquAct's module) and a re-implemented SOTA baseline, over **4096 test episodes** across four tasks. This is the most thorough modulation ablation design in the corpus. |
| **Reported instability** | **No numerical instability — but a clear negative result for modulation used alone.** On the MLP backbone, `FiLM Only` **degrades** Flip success from 58.11 % (Backbone Only) to **34.08 %**, and the authors state *"modulation without value disentanglement is insufficient for highly nonlinear dynamics."* Also `EMLP Backbone Only` collapses to 0.00 % on Rotate, confirming *"strict equivariance alone reduces policy expressiveness."* |
| **Single-task or multi-task** | **Multi-task** — four aerobatic primitives learned by one policy, composable at deployment into complex manoeuvres. Conditioner is the task command. Gains **do** depend on task diversity here. |

### 14.2 Verification of the survey's claim

Survey entry **U3**: *"Equivariant actor + FiLM-based task modulation + multi-head critic; 98.85 % success across aerobatic manoeuvres. Second instance of the symmetry + FiLM combination (cf. EquAct)."*

**Architecture confirmed exactly.** The **98.85 %** figure does not appear in Table I as printed; the full GEAR configuration (EMLP, Multi-Head+FiLM) reports per-task success rates of **99.90 / 95.80 / 99.71 / 100.00 %**, whose mean is 98.85 %. So the number is a correct aggregate, reconstructed rather than quoted. Worth noting when citing.

### 14.3 The finding the survey could not have — a head-to-head against EquAct's iFiLM

Table I's EMLP block contains a direct substitution: replace GEAR's ordinary FiLM with **EquAct's SO(2)-equivariant iFiLM**, holding everything else fixed.

| EMLP configuration | Hover SR | Flip SR | **Roll SR** | Rotate SR | mean |
|---|---|---|---|---|---|
| Multi-Head + **IFiLM** (EquAct's module) | 94.92 | **99.22** | **87.70** | 100.00 | 95.46 |
| Multi-Head + **FiLM** (unconstrained) | **99.90** | 95.80 | **99.71** | 100.00 | **98.85** |

The authors' verdict: *"Replacing FiLM with Multi-Head+IFiLM, introduced in EquAct, enforces stronger invariance but sacrifices robustness in roll."*

**Why this matters for the grouped-modulation question.** [§9](#9-zhu-et-al-2025--equact-an-se3-equivariant-multi-task-transformer-for-open-loop-robotic-manipulation) established that iFiLM is the corpus's only genuinely *grouped* modulation — one scalar gain per irreducible-representation block — and that the grouping is **forced by Schur's lemma** rather than chosen. GEAR now supplies the missing control: **what happens when you impose that same constraint on a task whose symmetry is only approximate?** Answer: you lose 12 points on the manoeuvre (Roll) that most violates the symmetry assumption, and 5 points on Hover.

GEAR's entire design rationale is the mirror image of EquAct's. Where EquAct constrains modulation *to preserve* equivariance, GEAR uses modulation *to escape* it:

> *"by introducing additional bias flexibility, FiLM allows the network to deviate from overly rigid equivariance when necessary, while still exploiting SO(2) symmetry where it remains advantageous."*

**The synthesis, and it is the most useful thing this pair of papers gives us.** Constrained (hence grouped) modulation wins when the symmetry it encodes is *exact* — EquAct's SE(3) scene transformations genuinely leave the task invariant. Unconstrained per-unit modulation wins when the symmetry is *approximate* — a drone's Roll manoeuvre has real yaw-asymmetries that the constraint forbids expressing. **Coarsening the modulation is beneficial exactly when the units being tied together are genuinely interchangeable, and harmful when they are not.**

This is a sharper statement of the caution already recorded in §9.7: our grouping ties units by **contiguous index**, an ordering with no meaning, so there is no sense in which the tied units are interchangeable. GEAR is the empirical demonstration that when you tie units that should not be tied, you pay — and you pay most on the hardest sub-task.

### 14.4 Phase 1 — Foundational overview

**Introduction.** Small drones can in principle fly aerobatics — flips, rolls, orbits — but training one controller to do several of these is hard, because each manoeuvre wants different things from the network and their reward signals interfere.

GEAR combines three ideas. First, it builds the **rotational symmetry of flight** into the network: if you rotate the whole scenario about the vertical axis, the correct control response rotates with it, and a network that knows this needs far less data to learn. Second, because that symmetry assumption is not perfectly true for every manoeuvre, it adds **FiLM layers that let the task command bend the rules** where a manoeuvre genuinely is asymmetric. Third, since the four manoeuvres are rewarded for completely different things, it gives the value estimator **a separate output head per task** instead of one head trying to score all four.

**Key findings.** The three ingredients are individually insufficient and jointly excellent. The symmetric network *alone* is worse than a plain one — it fails one manoeuvre entirely. FiLM *alone* actually makes one manoeuvre worse. Separate value heads alone help but leave the policy inflexible. All three together reach 98.85 % average success across the four manoeuvres, over 4096 test episodes, and transfer to real hardware.

The most interesting comparison is with a stricter version of the same idea. Substituting a symmetry-*preserving* FiLM (borrowed from EquAct) for the ordinary one makes the network more principled and **worse** — because the whole point of the FiLM here was to let the network break a symmetry that is not quite true.

**Initial takeaway.** An inductive bias is only free when it is correct. Feature-wise modulation is being used here not to add information but to provide a **controlled escape hatch** from a constraint that is right most of the time and wrong some of the time.

### 14.5 Phase 2 — Graduate-level deep dive

#### 14.5.1 Group structure and where modulation sits relative to it

The state is partitioned by transformation behaviour under $SO(2)$ (yaw rotation):

- **Equivariant pairs** — $(p_{\mathrm{rel},x}, p_{\mathrm{rel},y})$, $(v_{\mathrm{rel},x}, v_{\mathrm{rel},y})$, $(\omega_{\mathrm{rel},x}, \omega_{\mathrm{rel},y})$ and attitude-matrix columns — transform under the 2-D irreducible representation $\rho_{\mathrm{vec}}(\theta) = R(\theta)$.
- **Invariants** — $p_{\mathrm{rel},z}$, $v_{\mathrm{rel},z}$, $\omega_{\mathrm{rel},z}$, total thrust $f_\Sigma$, and the scalar command — follow the trivial representation $\rho_{\mathrm{sca}}(\theta) = 1$.

The overall group action is $\mathrm{diag}(\rho^\oplus_{\mathrm{vec}}, \rho^\oplus_{\mathrm{sca}})$, and the body-frame action space is invariant.

**The critical placement decision.** GEAR applies FiLM **before** features are organised into group representations. Contrast with EquAct, which applies iFiLM **to** already-organised spherical-Fourier blocks and constrains it so the organisation survives. The consequence is exactly the intended one:

- **EquAct**: modulation acts *within* the representation structure, so Schur's lemma applies, forcing one scalar per irrep block. Equivariance is preserved by construction.
- **GEAR**: modulation acts *before* the structure is imposed, so it is unconstrained and per-unit — and it can therefore introduce components that do not respect the group action. This is what the paper calls *"controlled symmetry breaking."*

**So the same operator, moved across one architectural boundary, changes from cell (c) grouped-and-exact to cell (a) per-unit-and-approximate.** That is a clean illustration that granularity is not always a free design choice — it can be determined by *where* in a structured network the modulation is inserted.

#### 14.5.2 The multi-head critic, and question 5

GEAR's critic rationale is the third independent statement of the same principle in this corpus:

> *"the critic must faithfully capture reward structures that differ across the four aerobatic tasks. For example, Hover requires maintaining absolute position and orientation stability, Flip emphasizes accurate angular velocity tracking… These task-specific reward signals differ in both structure and sensitivity, making a single shared value head prone to conflating them."*

Compare PAPL ([§8](#8-yoon-et-al-2026--papl-phase-aware-policy-learning-for-skateboard-riding-of-quadruped-robots-via-feature-wise-linear-modulation)): modulate the critic because the reward is phase-conditioned. Compare Marquis ([§12](#12-marquis--farhood-2026--hypernetwork-conditioned-reinforcement-learning-for-robust-control-of-fixed-wing-aircraft-under-actuator-failures)): condition the critic because the dynamics are regime-conditioned. **Same diagnosis — a value function that must represent structurally different reward regimes with shared parameters — and three different treatments:**

| Paper | Treatment of the critic | Result |
|---|---|---|
| PAPL | FiLM the critic with the same phase signal as the actor | works (not isolated) |
| Marquis | FiLM the critic with the same fault signal as the actor | **−50 % error, isolated** |
| **GEAR** | **separate value head per task**, shared equivariant trunk | **essential** — `Multi-Head Only` beats `FiLM Only` on 3 of 4 tasks |

GEAR's evidence for the critic treatment is strong: on the MLP backbone, `Multi-Head Only` reaches Roll 66.44 % where `FiLM Only` reaches **1.56 %**. *"Multi-Head only stabilizes optimization by separating reward signals, leading to more balanced performance across tasks."*

#### 14.5.3 The full ablation, read carefully

Table I, mean success rate (%) over 4096 test episodes:

| Backbone | Configuration | Hover | Flip | Roll | Rotate |
|---|---|---|---|---|---|
| **MLP** | Backbone Only | 98.34 | 58.11 | 1.37 | 98.14 |
| MLP | **FiLM Only** | 98.54 | **34.08** ↓ | 1.56 | 99.90 |
| MLP | Multi-Head Only | 98.73 | 89.36 | 66.44 | 100.00 |
| MLP | Multi-Head+FiLM | 98.63 | 89.26 | 84.42 | 99.90 |
| MLP | Baseline (SOTA re-impl.) | 99.71 | 99.02 | 56.84 | 81.05 |
| **EMLP** | Backbone Only | 81.07 | 71.48 | 22.71 | **0.00** |
| EMLP | FiLM Only | 98.34 | 94.63 | 48.83 | 100.00 |
| EMLP | Multi-Head Only | 52.63 | 48.54 | 45.76 | 58.98 |
| EMLP | Multi-Head+IFiLM | 94.92 | **99.22** | 87.70 | 100.00 |
| **EMLP** | **Multi-Head+FiLM (GEAR)** | **99.90** | 95.80 | **99.71** | **100.00** |

**Four readings worth extracting.**

1. **FiLM alone can hurt.** MLP + `FiLM Only` drops Flip from 58.11 to 34.08. This is one of very few *negative* modulation results anywhere in the corpus, and the diagnosis is specific: *"modulation without value disentanglement is insufficient for highly nonlinear dynamics"* — i.e. conditioning the actor while leaving the critic to conflate four reward structures makes the advantage estimates worse, not better. **This is corroborating evidence for the actor-critic coupling that Marquis measures.**
2. **The equivariant backbone alone is much worse than a plain MLP** (EMLP Backbone Only: Rotate 0.00 %). Symmetry is only an asset once something can relax it.
3. **The interaction is superadditive on the constrained backbone.** *"These gains are especially pronounced in EMLP-based models: because stronger equivariance constraints reduce flexibility, FiLM and Multi-Head become essential to relax overly rigid symmetry."* Roll goes 22.71 (EMLP alone) → 48.83 (+FiLM) → **99.71** (+FiLM+Multi-Head).
4. **The `Multi-Head Only` EMLP row is anomalous** (52.63 / 48.54 / 45.76 / 58.98 — worse than MLP `Multi-Head Only` across the board), and the paper does not comment on it. With no seeds or confidence intervals reported anywhere, this row is a reminder that individual cells in this table should not be over-read.

**Honest overall caveat.** 4096 test episodes is a large evaluation, but **the number of training seeds is never stated** and no confidence intervals appear. The FiLM-versus-iFiLM comparison rests on a single training run per configuration.

### 14.6 Relevance to this project

- **It qualifies the §9 grouping finding, and this is its main value to us.** iFiLM's grouped structure loses to unconstrained per-unit FiLM when the symmetry justifying the grouping is only approximate. Taken together, EquAct and GEAR say: **grouping pays if and only if the tied units are genuinely interchangeable under a structure the task respects.** Our contiguous-index grouping has no such justification. This is the sharpest available framing of the risk in our $G$ sweep, and it should appear wherever we present that screen.
- **Third independent instance of "the critic needs help when reward structure varies", with a different treatment.** GEAR uses separate heads rather than modulation. That is a **fourth option for our project** that we have not considered: if our value function does turn out to vary with the modulator's input, separate value heads are an alternative to modulating the critic, and GEAR's evidence for heads is strong.
- **A rare negative result for modulation alone, with a mechanism.** `FiLM Only` degrading Flip is the clearest corpus evidence that **modulating the actor while leaving the critic unconditioned can be actively harmful** in multi-regime RL. Our architecture modulates the policy path and leaves the critic untouched. Combined with Marquis's +50 % from critic conditioning, this is now two papers pointing at the same asymmetry in our design.
- **Placement determines granularity — a structural insight we can use.** Inserting modulation before versus after a structuring operation changes what granularity is even expressible. Our multimodal hub is a structuring operation. Whether our $\gamma/\beta$ enter before or after modality fusion is therefore not a neutral implementation detail, and this is worth checking in code.
- **Limited transfer otherwise.** Multi-task, feedforward, no temperature head, no recurrence, drone dynamics. Nothing here bears on our crash (question 3) or our single-task setting (question 4).

### 14.7 Appendix: Section-by-Section Backbone

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | Flight control for micro aerial vehicles; multi-task aerobatics via an SO(2)-equivariant actor network, **FiLM-based task modulation, and a multi-head critic**. |
| I–II | Introduction / Related Work | Agility unlocks rapid-response capabilities. Prior multi-task RL for MAVs uses multiple critics and shared encoders. Tension between symmetry, reward shaping and task diversity motivates a more flexible framework. |
| III | Problem formulation | Four aerobatic primitives — Hover, Flip, Roll, Rotate — composable at deployment. Establishes the tension: *"strict equivariance can impose excessive constraints: different maneuvers interact with symmetry in distinct ways, and heterogeneous reward shaping emphasizes task-specific features."* |
| IV-A | Group Representation | Equivariant pairs under $\rho_{\mathrm{vec}}(\theta) = R(\theta)$; invariants under $\rho_{\mathrm{sca}} = 1$; body-frame action space invariant. **"FiLM layers applied before grouping allow controlled symmetry breaking when task asymmetries are present."** |
| IV-B | SO(2)-Equivariant Actor Network | EMLP backbone encoding yaw symmetry; notes it must stay flexible enough for task asymmetries. |
| IV-C | **Feature-wise Linear Modulation** | **Eq. 17** $\mathrm{FiLM}(x,c) = x\gamma(c) + \beta(c)$. Two stated purposes: conditioning one shared backbone on task commands, and providing **bias flexibility to deviate from overly rigid equivariance**. |
| IV-D | **Multi-Head Critic** | Per-task reward structures differ in *"structure and sensitivity"*; a single head conflates them. Shared equivariant backbone with separate value heads. Critic discarded at deployment. |
| V-A | Experimental setup | 4096 test episodes across four tasks; metrics are success rate, SCD, and per-task primary error. |
| V-B | **Ablation Study** | **Table I** — the 2×2×2 design plus IFiLM substitution plus SOTA baseline. Baseline achieves 84.16 % overall but is limited by poor Roll and needs 10.1 h training (over an hour more than GEAR). Notes the baseline's encoder mixes observations before both actor and critic, preventing a fair EMLP substitution. |
| V-C | Real World Experiment | Learned primitives transfer to hardware; complex manoeuvres composed by manually selecting task types and motion parameters. |
| VI | Conclusion | Combining geometric priors with adaptive conditioning and modular value estimation yields one end-to-end policy for multi-task aerobatics. |

---

## 15. Guo et al. 2026 — *MoE-ACT: Scaling Multi-Task Bimanual Manipulation with Sparse Language-Conditioned Mixture-of-Experts Transformers*

**PDF:** `docs/project/references/modulation_in_rl/sources/Guo et al. 2026 - MoE-ACT (preprint).pdf`
**arXiv:** [2603.15265v1](https://arxiv.org/abs/2603.15265), 16 Mar 2026.

> **Scope note.** MoE-ACT combines a mixture-of-experts encoder with a FiLM-conditioned decoder. Per this folder's scope, **only the FiLM half is reviewed**; the MoE encoder, its routing, and its auxiliary losses are described only where needed to interpret the FiLM ablation.

### 15.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue + year** | **Unstated in PDF.** IEEE conference format, no venue line; only `arXiv:2603.15265v1 [cs.RO] 16 Mar 2026`. Cite as a 2026 preprint. |
| **Conditioning signal** | A **task embedding** $z_{\mathrm{task}}$. Notably it is **not** a raw language embedding: *"The language instruction is first processed by a classifier to identify the task and assign a unique task embedding."* So the conditioner is a **discrete task identity** recovered from language — a lookup, not a continuous encoding. |
| **Modulated component** | The **action queries $Q$ in the Transformer decoder** — i.e. the action head. Not the encoder (that is where the MoE lives), not a critic (imitation learning, ACT/CVAE framework). |
| **Granularity cell** | **(b) per-channel** over $d_{\mathrm{model}}$, broadcast across the $k$ action queries. $Q \in \mathbb{R}^{k\times d_{\mathrm{model}}}$, and $\gamma, \beta$ from a single MLP. |
| **Placement** | **Every decoder layer** — *"we apply task-conditioned FiLM at each decoder layer"*, applied to the query **before** the two cross-attention blocks (Eqs. 7–8). |
| **Parameterization** | One generator: $[\gamma, \beta] = \mathrm{MLP}_{\mathrm{FiLM}}(z_{\mathrm{task}})$ (Eq. 5). Whether it is shared across decoder layers is **not stated**. |
| **Initialization** | **Not stated explicitly — but identity-centred by construction.** Eq. 6 uses the residual form $Q' = (1+\gamma)\odot Q + \beta$, so at $\gamma = \beta = 0$ the modulation is the identity. Same device as CogVLA and Marquis; **sixth independent instance in the corpus.** |
| **Ablation present?** | **Yes, and the modulation is isolated.** Table II, RoboTwin 2.0, six bimanual tasks: **Full 55 % average vs `w/o FiLM` 46 %** (−9). Per-task in §15.3. |
| **Reported instability** | **None reported for FiLM.** The paper does discuss **mode collapse** — but in the MoE module, which it prevents with three auxiliary losses ($-\alpha\mathcal{L}_{\mathrm{entropy}} + \beta\mathcal{L}_{\mathrm{balance}} + \gamma\mathcal{L}_{\mathrm{ortho}}$). **Third corpus instance of a mixture-of-experts requiring explicit anti-collapse machinery** (after FLOWER's NaN and PAPL's expert collapse) — while the affine modulation alongside it needs none. |
| **Single-task or multi-task** | **Multi-task** — six bimanual RoboTwin 2.0 tasks plus two real-world tasks, one unified policy. The conditioner is a task ID, so the mechanism has no meaning without task diversity. |

### 15.2 Verification of the survey's claim

Survey entry **U4**: *"FiLM modulates action tokens during decoding while a sparse MoE handles task decomposition — the two competing conditional mechanisms used together rather than as alternatives."* **Confirmed exactly**, and the division of labour is explicit in the paper's own framing: the MoE sits in the encoder for *"task-specific feature decoupling"* while FiLM sits in the decoder to ensure *"action generation is consistent with task instructions."*

### 15.3 The ablation, and the honest reading

Table II, RoboTwin 2.0 success rate (%):

| Method | Bottle | H/O | Pot | Fries | Skillet | Cab. | **Avg.** |
|---|---|---|---|---|---|---|---|
| **Full Version** | **52** | **70** | **76** | **90** | 12 | 28 | **55** |
| `w/o FiLM` | 24 | 64 | 60 | 82 | **16** | **32** | 46 |
| `w/o Multi-Scale Cross-Attention` | 40 | 68 | 64 | 74 | 10 | 28 | 47 |

Paper's claim: *"removing FiLM reduces average success rate… FiLM-based conditioning helps the model adapt"* to differing task objectives.

**The honest reading is that the average conceals a split.** FiLM helps substantially on four tasks — Bottle **+28**, Pot **+16**, Fries **+8**, Handover **+6** — and **hurts on the two hardest**: Skillet 12 vs 16, Cabinet 28 vs 32. The two tasks where the full system performs worst are the two where removing the modulation improves it.

**This is the inverse of Yuan 2024's pattern**, and worth flagging as a genuine tension in the corpus. Yuan ([§6](#6-yuan-2024--unpacking-the-individual-components-of-diffusion-policy)) found FiLM's benefit concentrated on *hard* tasks and absent on easy ones. MoE-ACT finds the benefit on tasks it already does well and a small penalty on tasks it does badly. With no seeds or confidence intervals — and 12 % versus 16 % being roughly one trial's difference at typical RoboTwin evaluation counts — **the correct conclusion is that the two negative cells are probably noise, and that this ablation supports "FiLM helps on average" but cannot support any moderator claim.** It should not be cited against Yuan.

### 15.4 Phase 1 — Foundational overview

**Introduction.** A two-armed robot that can perform six different jobs — handing over a bottle, lifting a pot, picking up fries, manipulating a skillet, opening a cabinet — from one set of network weights runs into a familiar problem: the jobs interfere. Training on all of them at once produces a policy that is mediocre at each.

MoE-ACT attacks this from two directions at once. In the part of the network that *understands the scene*, it installs a **mixture of experts**: many small sub-networks, of which a router activates only a few per input, so different tasks can recruit different specialists. In the part that *produces the motion*, it uses **FiLM**: the task identity generates a gain and an offset applied to the internal "action queries" at every decoder layer, so the motion generator is re-tuned per task.

The design intent is a division of labour — the experts decouple *features*, the modulation keeps the *output* aligned with the instruction.

**Key findings.** The combination reaches 55 % average success across six simulated bimanual tasks against the strongest baseline's 49 %, and 60 % against 47 % on two real-world tasks. Removing the modulation costs 9 percentage points of average success; removing the multi-scale visual pathway costs 8.

**Initial takeaway.** Mixture-of-experts and feature-wise modulation are usually presented as competing answers to the same question. Here they are used as complements at different stages, on the reasoning that they solve different problems — sharing capacity versus aligning output.

### 15.5 Phase 2 — Graduate-level deep dive

#### 15.5.1 The modulation, and why it targets the query

The conditioning path is short:

$$
[\gamma, \beta] = \mathrm{MLP}_{\mathrm{FiLM}}(z_{\mathrm{task}}), \qquad
Q' = \mathrm{FiLM}(Q, z_{\mathrm{task}}) = (1+\gamma)\odot Q + \beta, \tag{5,6}
$$

with $Q \in \mathbb{R}^{k\times d_{\mathrm{model}}}$ the $k$ action queries. The modulated query then drives two chained cross-attention blocks (Eqs. 7–8):

$$
\mathrm{Out}_{\mathrm{inter}} = \mathrm{CA}(Q = Q',\, K = F_{\mathrm{low}},\, V = F_{\mathrm{low}}), \qquad
\mathrm{Out}_{\mathrm{final}} = \mathrm{CA}(Q = \mathrm{Out}_{\mathrm{inter}},\, K = F_{\mathrm{enc}},\, V = F_{\mathrm{enc}}),
$$

where $F_{\mathrm{low}}$ are intermediate ResNet-18 features and $F_{\mathrm{enc}}$ the final MoE-encoder output.

**Modulating the query is a meaningfully different target from modulating a hidden activation**, and the paper does not spell out why it matters. In cross-attention, the query determines **what the decoder retrieves** from the visual features. Scaling and shifting $Q$ therefore rotates and rescales the retrieval directions — it changes *which visual evidence each action slot attends to*, per task. This is closer in spirit to CogVLA's use of modulation to bias an information bottleneck ([§10](#10-li-et-al-2025--cogvla-cognition-aligned-vision-language-action-model-via-instruction-driven-routing--sparsification)) than to the classical use of changing a layer's transfer function. **Same operator, third distinct functional role in this corpus**: change the computation (PAPL, Marquis), bias a bottleneck (CogVLA), steer attention (MoE-ACT).

**The conditioner is a discrete lookup, which limits the claim.** Because language is first classified into a task ID, the modulation can only ever produce one of six $(\gamma,\beta)$ pairs per layer. That is closer to a learned per-task parameter table (Dumoulin et al.'s conditional instance normalisation) than to a general conditioning function, and it means the architecture cannot generalise to an unseen instruction.

#### 15.5.2 Surrounding architecture and training

Built on **ACT** (Action Chunking Transformer) with its CVAE structure. Total loss:

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{kl}}\mathcal{L}_{\mathrm{kl}} + \lambda_{\mathrm{MoE}}\mathcal{L}_{\mathrm{MoE}}, \qquad
\mathcal{L}_{\mathrm{rec}} = \sum_t \lVert \hat a_t - a_t\rVert_1, \quad \mathcal{L}_{\mathrm{kl}} = D_{\mathrm{KL}}\bigl(q(z\mid O_t, a_{t:t+k})\,\|\,p(z)\bigr).
$$

The MoE auxiliary loss $\mathcal{L}_{\mathrm{MoE}} = -\alpha\mathcal{L}_{\mathrm{entropy}} + \beta\mathcal{L}_{\mathrm{balance}} + \gamma\mathcal{L}_{\mathrm{ortho}}$ exists *"to prevent mode collapse and ensure expert diversity"*.

**Worth recording for the corpus-level pattern.** The MoE half needs three auxiliary losses to remain stable. The FiLM half needs none, and gets its stability for free from the $(1+\gamma)$ parameterisation. Across FLOWER (NaN losses from an expert MLP), PAPL (pushing-phase expert collapsed), and MoE-ACT (three anti-collapse losses), **the corpus's consistent finding is that mixture-of-experts conditioning is the fragile mechanism and affine modulation is the robust one.**

#### 15.5.3 Results

Table I, RoboTwin 2.0 bimanual multi-task (%):

| Method | Bottle | H/O | Pot | Fries | Skillet | Cab. | Avg. |
|---|---|---|---|---|---|---|---|
| ACT | 42 | 4 | 59 | 10 | 4 | 15 | 22 |
| DP | 23 | 29 | 10 | 5 | 3 | 43 | 19 |
| RDT | 55 | 60 | 63 | 66 | 5 | 43 | 49 |
| BAKU | 26 | 68 | 0 | 46 | 2 | 28 | 31 |
| **MoE-ACT** | 52 | **70** | **76** | **90** | **12** | 28 | **55** |

Table III, real-world bimanual: MoE-ACT 68 / 52 (avg **60**) vs BAKU 57 / 37 (47) vs ACT 44 / 26 (35).

Note the Skillet column: every method scores in single digits or low teens. **That task is unsolved by all six methods**, which is the context in which its 12-versus-16 ablation cell should be read.

### 15.6 Relevance to this project

**Weak. This is a "does not bear on our design" paper with two small exceptions.**

- **It does not touch any of our five open questions.** Standard per-channel granularity (nothing for question 1); the conditioner is a discrete task ID and the modulated stream is an action query, so the two are unrelated information sources (nothing for question 2); no instability in the modulation path (nothing for question 3); purely multi-task with a task-ID conditioner, so it is the *opposite* of our setting (nothing for question 4); no critic exists (nothing for question 5).
- **Exception 1 — the anti-collapse pattern.** This is the third corpus instance of mixture-of-experts requiring explicit stabilisation while co-located affine modulation requires none. That pattern is now robust enough to state in our documentation: **within this corpus, the failures of conditional architectures cluster on the expert-isolation side, not the parameter-sharing side.** Our grouping sweep pushes further toward sharing than any of these, which is at least directionally reassuring — though it remains a directional signal, not evidence.
- **Exception 2 — the $(1+\gamma)$ identity anchoring.** Sixth independent instance. Combined with FLOWER, CogVLA, GR00T's lineage, Marquis and SplitAdapter, **every reviewed paper that says anything about initialisation ensures the modulation begins at identity.** Whether our multiplicative path is anchored at 1 is a code question that this corpus now answers unanimously in one direction.
- **What is not transferable.** Imitation learning, no reward, no exploration, a discrete lookup conditioner, transformer decoder cross-attention as the modulated object, and no seeds or error bars anywhere.

### 15.7 Appendix: Section-by-Section Backbone

| § | Title | Core content |
|---|---|---|
| — | **Abstract** | Robots handling multiple tasks under a unified policy; MoE for task-specific feature decoupling in the encoder and **FiLM to dynamically modulate action tokens** in the decoder. |
| I | Introduction | Fig. 1 overview: MoE in the Transformer encoder, FiLM in the decoder. *"The MoE module enables task-specific feature decoupling, while FiLM ensures that action generation is consistent with task instructions."* |
| II | Related Work | ACT, Diffusion Policy, RDT, BAKU; sparse MoE lineage from Shazeer et al.; VLA models. |
| III-A/B | Encoder and MoE module | Sparse language-conditioned mixture-of-experts in the Transformer encoder. |
| III-C | **Decoder** | **1) Task-conditioned FiLM** — language classified to a task ID, task embedding $z_{\mathrm{task}}$, **Eqs. 5–6**, applied *"at each decoder layer"*. **2) Multi-Scale Cross-Attention** — **Eqs. 7–8**, two cross-attention blocks per layer attending to intermediate ResNet-18 features and final encoder output respectively, because *"standard decoders attend only to the final encoder output, which can discard low-level visual details."* |
| III-D | **Training Objectives** | **Eq. 9** composite loss; **Eq. 10** CVAE reconstruction ($L_1$) and KL terms; **Eq. 11** MoE auxiliary loss with entropy, balance and orthogonality terms *"to prevent mode collapse and ensure expert diversity."* |
| IV-A | Comparison with baselines | Table I, RoboTwin 2.0, six bimanual tasks vs ACT, DP, RDT, BAKU. |
| IV-B | **Ablation Study** | Table II — `w/o FiLM` (removes task conditioning of the action queries) and `w/o Multi-Scale Cross-Attention`. **1) Effect of FiLM Conditioning**: removing it reduces average success rate by 9 points. |
| IV-C | Real-world experiments | Table III, two dual-arm tasks (handover bottle, putting cubes into a box). |

---

## 16. Cross-paper summary tables

### 16.1 The fixed extraction block, all ten papers at a glance

| # | Paper | Venue **as printed in PDF** | Conditioning signal | Dim. | Modulated component | Granularity |
|---|---|---|---|---|---|---|
| 6 | Yuan 2024 | unstated (arXiv 2412.00084v1) | observation sequence | full obs | denoising U-Net (action head) | (b) inferred |
| 7 | FLOWER | **CoRL 2025** | flow timestep + action-type + embodiment | low | 18-layer flow transformer | (b) |
| 8 | PAPL | unstated (arXiv 2602.09370v2) | cyclic phase $(\cos\phi,\sin\phi)$ | **2** | **actor AND critic** | (a) |
| 9 | EquAct | unstated ("Preprint. Under review.") | language embedding (type-0) | low | equivariant point-cloud U-Net | **(c) grouped** |
| 10 | CogVLA | **NeurIPS 2025** | (i) instruction; (ii) instruction tokens per layer | low | (i) vision encoders; (ii) LM trunk | (b) |
| 11 | GR00T N1 | unstated (industrial report) | denoising timestep | **1** | DiT action head | (b) not stated |
| 12 | Marquis | unstated (arXiv 2604.03392v1) | actuator-fault vector $\lambda$ | **6** | **actor AND critic** | (a) |
| 13 | SplitAdapter | unstated (arXiv 2606.03297v1) | object/load latent **+** dynamics latent | 16+16 | frozen policy, **split by depth** | (a)/(b) |
| 14 | GEAR | unstated (arXiv 2602.10997v1) | task command | low | actor + multi-head critic | (a)/(b) |
| 15 | MoE-ACT | unstated (arXiv 2603.15265v1) | task ID (classified from language) | discrete | decoder action queries | (b) |

| # | Paper | Placement | Parameterization | Initialization | Modulation ablated? | Instability | Task setting |
|---|---|---|---|---|---|---|---|
| 6 | Yuan 2024 | many, inherited, count not stated | shared signal + per-site heads | **not stated** | **Yes — vs concatenation, 8 tasks** | none | **single-task ×8** |
| 7 | FLOWER | all 18 blocks, multi-point per block | **one weight set shared across all layers** + per-layer LoRA | **zero-init, for stability** | Yes (4.44 vs 4.43) — **LoRA confound** | **NaN, but from an MoE design** | multi-task |
| 8 | PAPL | every layer, **pre-activation, no norm** | not stated | **not stated** | **Yes — `No-FiLM`, graphical only** | MoE expert collapse | **single-task** |
| 9 | EquAct | inside EPTU, count not stated | MLP per site | not stated | **Yes — vs vanilla FiLM** | none | multi-task |
| 10 | CogVLA | encoder blocks + **32 LM layers** | per-site generators (linear / 2-layer MLP) | **identity-centred $(1+\gamma)$** | stages only, **not the FiLM itself** | none (σ 0.2–0.6 %) | multi-task |
| 11 | GR00T N1 | every DiT block, $N$ not stated | **not stated** | **not stated** | **No** | none | multi-task |
| 12 | Marquis | 2 hidden layers, **pre-activation, no norm** | one shared hypernetwork per network | **identity-centred AND damped ×0.1** | **Yes — vs LoRA vs MLP, + critic** | **rank-48 blow-up; LoRA+HC** | **single-task** |
| 13 | SplitAdapter | **early layers ← load; late layers ← dynamics** | two generators, one per context | **identity-init, stated** | Yes (86/90 vs 81/90) | none | **single-task** |
| 14 | GEAR | **before group-representation grouping** | per-site generators | not stated | **Yes — 2×2 factorial + iFiLM sub** | `FiLM Only` **degrades** Flip | multi-task |
| 15 | MoE-ACT | every decoder layer, on the query | one MLP | **identity-centred $(1+\gamma)$** | **Yes (55 vs 46)** | MoE mode collapse (3 aux losses) | multi-task |

### 16.2 What the modulation is *for* — three distinct functional roles

A pattern the corpus makes visible that no single paper states: the same $(\gamma,\beta)$ operator is used for three different jobs.

| Role | What the gain does | Papers |
|---|---|---|
| **Change the computation** | rescale the layer's Jacobian so the network computes a different function per condition | Yuan, PAPL, Marquis, GEAR, FLOWER, GR00T, EquAct, SplitAdapter |
| **Bias an information bottleneck** | decide what survives a compression step | CogVLA (aggregation tokens) |
| **Steer retrieval** | rescale attention queries, changing *which* evidence is read | MoE-ACT |

### 16.3 Conditioner dimensionality — the corpus's most uniform regularity

| Conditioner | Dim. | Paper |
|---|---|---|
| denoising timestep | 1 | GR00T N1 |
| cyclic phase $(\cos\phi, \sin\phi)$ | 2 | PAPL |
| actuator-fault vector | 6 | Marquis |
| task ID (one of 6) | discrete | MoE-ACT |
| task command | low | GEAR, FLOWER |
| language embedding | low (compressed) | EquAct, CogVLA |
| two context latents | 16 + 16 | SplitAdapter |
| **full observation sequence** | **full** | **Yuan 2024 — and only into a stream that has no other route to it** |
| **raw observation, into a stream that already reads it** | **full** | **— no paper in this corpus** |

---

## 17. Appendix — adjacent evidence from the six unreviewed PDFs

These six PDFs are held in `sources/` but are **not** scale-and-shift modulation and are therefore **not reviewed**. Each gets one paragraph, covering only what it says about a question *about FiLM*. Their own contributions are deliberately not assessed.

**Tessera et al. 2024 — HyperMARL** (NeurIPS 2025, per the PDF's page-1 footer). Full weight generation from an agent embedding, so out of scope as a mechanism — but it carries **the corpus's principal counter-evidence on self-conditioning**, and it is the reason the paper was downloaded. Its `HyperMARL w/o GD` ablation conditions the generator on $[o_t, e_i]$ instead of $e_i$ alone, so the generator reads the **raw current observation** while the network it parameterises reads that same observation — structurally our Axis-5 configuration. The result is negative on both tested environments: *"Gradient decoupling is consistently critical across both environments"* (§6.1, Fig. 8). The paper also supplies a derivation-level argument: its variance decomposition $\mathrm{Var}(\hat g) = \sum_{i,j} J_i \mathrm{Cov}(\hat Z_i, \hat Z_j) J_j^\top$ requires the generator Jacobian $J_i$ to be **deterministic with respect to the mini-batch** (assumption A3), which fails the moment the observation enters the generator. Two honest confounds: the manipulation regenerates an entire weight tensor per timestep (far heavier than emitting $(\gamma,\beta)$), and HyperMARL's headline failure mode — a *shared weight tensor* absorbing conflicting updates from a *discrete* conditioner — is not our architecture. **Cite the `w/o GD` ablation and the determinism argument; do not cite the headline.**

**Zhang et al. 2025 — DyMoDreamer** (NeurIPS 2025, per the PDF's page-1 footer). Excluded because its "dynamic modulation" is **not modulation**: the 32×32 stochastic categorical latent $d_t$ enters every consumer by **concatenation** — *"The model state is formed by concatenating $h_t$, $z_t$ and $d_t$"* (§2.2) — and a keyword scan of the full 36-page PDF finds no multiplicative interaction anywhere. It is architecturally on the *concatenation* side of the very axis Yuan 2024 manipulates. Its one bearing on a FiLM question is on **single-task gains** ([§4](#4-evidence-ledger--five-open-questions), question 4): it sets a new Atari-100k record (156.6 % mean human-normalised) with no task distribution at all, and its ablations isolate *why* — not added capacity (a DreamerV3 with a proportionally larger latent is worse) and not differencing per se (latent-space differencing underperforms), but specifically that pixel-space differencing surfaces small moving objects the encoder provably discards. **That is a corroborating instance of "a second stream earns its keep by carrying information the main stream cannot represent", reached without any affine gain.**

**Huang et al. 2024 — MENTOR** (ICML 2025 per the survey's OpenReview record). Mixture-of-experts backbone replacing an RL agent's MLP — conditional *computation*, not conditional *gain*, hence out of scope. Relevant only as a fourth data point in the corpus's mixture-of-experts pattern (see question 3) and as further evidence that conditional capacity can help in single-task visual RL.

**Grooten et al. 2025 — SPARC** (AAAI 2026 Oral per the survey's arXiv-comments record). Retained as **negative evidence**: a 2026 top-tier paper whose entire subject is injecting context into an RL policy, which **concatenates** context and history latents before the final layers rather than modulating. It is the corpus's cleanest demonstration that concatenation remains the default in mainstream contextual RL, and therefore the baseline against which Yuan's, PAPL's and Marquis's routing comparisons should be read.

**Black et al. 2024 — $\pi_0$.** Decoder-only mixture-of-experts transformer in which an action expert cross-attends to observation tokens; AdaLN-Zero appears **only** in the non-VLM ablation baseline $\pi_0$-small. Retained as **displacement evidence** — the clearest instance of affine modulation being superseded by attention at frontier scale. Read together with GR00T N1 ([§11](#11-nvidia-et-al-2025--gr00t-n1-an-open-foundation-model-for-generalist-humanoid-robots)) and FLOWER ([§7](#7-reuss-et-al-2025--flower-democratizing-generalist-robot-policies-with-efficient-vision-language-action-flow-policies)), the accurate trend statement is **not** "attention won" but "mechanism follows signal structure": scalar/global signals go through modulation, token-structured signals through attention, peer-rank vectors through concatenation.

**Li et al. 2025 — Neuro-Vesicles.** A position paper arguing that FiLM/hypernetwork modulation is the degenerate dense, short-lived limit of a stochastic vesicle-population process. Self-described as early-stage theoretical design with **no experiments**. Rhetorically useful if the project ever needs to frame its neuromodulator against a biophysical alternative; it contributes no evidence to any of the five questions and is not otherwise assessed.

### 17.1 The e-nmRNN gap

**One paper from the original high-priority list is deliberately absent**: *Volume Transmission Implements Context Factorization to Target Online Credit Assignment and Enable Compositional Generalization* (**e-nmRNN**), NeurIPS 2025 Poster, OpenReview `S9Y89poypx`. Acquisition is pending a separate decision. Under the narrowed scope it would in any case have been context-only — its modulation is derived from a biophysical release model, not an affine generator. It was, however, the earlier surveys' provisional candidate for a grouped-modulation precedent, so **the answer to question 1 below is stated over the reviewed corpus and does not include it.**

Two further survey entries were never downloaded and are likewise uncovered: *Hyper-GoalNet* (NeurIPS 2025) and *HyPoGen* (ICLR 2025), both full-weight hypernetwork papers, out of scope under the narrowed definition.
