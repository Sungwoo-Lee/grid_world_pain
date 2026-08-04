---
title: "Recent FiLM / Feature-Wise-Modulation Variants in Reinforcement Learning — Web Survey (2024–2026)"
topic: FiLM
status: curated
created: 2026-08-05
last_updated: 2026-08-05
related:
  - film_in_rl_survey.md
  - film_modulation_granularity_synthesis.md
  - ../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md
  - film_lit_review.md
  - film_synthesis.md
scope: |
  Forward-looking web survey (NOT a corpus synthesis) of feature-wise modulation
  in reinforcement learning published 2024–2026, prioritising NeurIPS, ICLR, ICML
  and AAAI, with CoRL / ICRA / RSS flagged as a clearly-separated secondary tier.
  Companion to film_in_rl_survey.md, which surveys the papers we already hold.
  Reuses the granularity / placement / parameterisation / grouping taxonomy of
  film_modulation_granularity_synthesis.md and the self-conditioning axis of
  film_in_rl_survey.md. Every entry carries a verifiable identifier; entries whose
  publication venue could not be confirmed from a source are segregated into their
  own tables and never mixed with verified ones.
---

# Recent FiLM / Feature-Wise-Modulation Variants in Reinforcement Learning

## 1. Plain-English entry point

Many neural networks are built so that one signal can **steer** another network's
computation. The standard recipe, called **FiLM** ("feature-wise linear
modulation"), takes a hidden layer's activity, multiplies it by a learned gain and
adds a learned offset, with both numbers produced by a small side-network reading
a steering signal — a language instruction, a task identifier, a risk level. Our
own agent uses this recipe, so it matters whether the field still does.

This document reports a **web search**, not a re-reading of papers we own. It asks:
across the 2024, 2025 and 2026 editions of the top machine-learning conferences — NeurIPS, the
International Conference on Learning Representations (ICLR), the International
Conference on Machine Learning (ICML) and the AAAI conference, 2024 through 2026 —
what has happened to this steering trick inside reinforcement-learning agents?

**The headline: FiLM is alive, but it has moved house and changed its name.** In
mainstream reinforcement-learning papers (the ones about learning from reward on
Atari, control suites or gridworlds) feature-wise modulation has all but vanished
from abstracts; plain concatenation of the extra signal onto the network input
remains the default, and the fashionable way to add conditional capacity is now a
**mixture of experts** (many small sub-networks, of which a router picks a few per
input). But in **robot policy learning** the same affine gain-and-offset operator
is everywhere — absorbed into a normalisation layer of diffusion and flow
transformers under the name **adaptive layer normalisation** (adaLN). Four genuine
new *variants* of the mechanism appeared at these venues — three of them in robot
policy learning, one inside a video-game world model. Meanwhile the two design
choices our project most wants precedent for — sharing one gain across a *block* of
neurons, and letting the modulator read the very observation it modulates — remain,
respectively, without any new precedent at all and without any exact one.

---

## 2. Method, coverage, and how this composes with existing docs

### 2.1 What was searched, and how

| Source | How queried | What it gives |
|---|---|---|
| **OpenReview API** (`api2.openreview.net/notes/search`) with a `venueid` filter | ~40 keyword queries across `ICLR.cc/{2025,2026}`, `NeurIPS.cc/{2024,2025}`, `ICML.cc/{2025,2026}`, `robot-learning.org/CoRL/{2024,2025}`, `rl-conference.cc/RLC/2025`; ~4 300 unique notes collected and filtered locally | title + abstract + keywords + **the venue label from the proceedings record** — this is the primary venue-verification channel |
| **arXiv API** (`export.arxiv.org/api/query`) | ~20 phrase queries restricted to `submittedDate:[2024-01-01 TO 2026-12-31]`; ~1 950 unique entries collected and filtered locally | arXiv ID, submission date, and the **comments field**, which is where authors record "Accepted at X" |
| **arXiv rendered full text** (`arxiv.org/html/<id>`) | fetched for six papers and grep-ed for `FiLM`, `adaLN`, `AdaLN-Zero`, `modulat`, `concat` | mechanism-level verification, not just abstract-level claims |

**Two sources refused to serve us and are therefore not part of the evidence
base:** the Semantic Scholar search API returned HTTP 429 (rate-limited) on every
attempt, and dblp.org timed out. OpenReview's *bulk* listing endpoint is behind a
bot challenge, so venue-wide enumeration was impossible; only its search endpoint
worked. AAAI does not host its proceedings on OpenReview, so AAAI coverage comes
solely from arXiv comments-field matches ("Accepted at AAAI 2026").

### 2.2 The coverage limit that matters most

**All searching was abstract-level.** FiLM is very often an unmentioned
implementation detail — Diffusion Policy's own paper does not put "FiLM" in its
abstract — so a paper that quietly uses feature-wise modulation inside its
architecture is invisible to this method. Consequently:

- The **variant** tables (§3) are close to complete for 2024–2026 top venues,
  because a paper that *modifies* the modulation mechanism almost always says so in
  the abstract.
- The **component-use** table (§4) is a *sample*, not a census. Read it as "here is
  what standard practice looks like", never as "here is how many papers use FiLM".

### 2.3 Taxonomy — reused, not redefined

Cells and axes come verbatim from
[`film_modulation_granularity_synthesis.md` §2](film_modulation_granularity_synthesis.md)
and its sister
[`../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md` §2](../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md):

| Axis | Question | Cells / values |
|---|---|---|
| **Axis 1 — granularity** | how many distinct $(\gamma,\beta)$ values per modulated tensor | **(a)** per-unit · **(b)** per-channel (collapses to (a) in dense layers) · **(c)** grouped · **(d)** layer-scalar · **(e)** global |
| **Axis 2 — placement** | which layers, how many injection sites | early/late, one/many |
| **Axis 3 — parameterisation** | shared generator + per-site heads, vs. per-site generators | — |
| **Axis 4 — grouping / tying** | explicit grouping, low-rank tying, population partitioning | — |
| **Axis 5 — self-conditioning** (added by [`film_in_rl_survey.md` §2](film_in_rl_survey.md)) | is the modulator's input the same information source as the modulated stream? | yes / no |

---

## 3. Category 1 — Papers proposing a FiLM **variant**

A *variant* modifies the modulation mechanism itself: new granularity, new
generator, new placement, new coupling, new initialisation, or a new constraint on
$(\gamma,\beta)$.

### 3.1 Verified — top-tier venues (NeurIPS / ICLR / ICML / AAAI)

| # | Paper | Venue (verified) | ID | Conditioning signal (generator input) | What is modulated | Granularity cell | Placement | What it CHANGES vs. vanilla FiLM | Problem solved | Headline result | Held? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | Zhu et al., *EquAct: An SE(3)-Equivariant Multi-Task Transformer for 3D Robotic Manipulation* | **ICLR 2026 Poster** ([forum](https://openreview.net/forum?id=d1wuA8oIH0)) | arXiv [2505.21351](https://arxiv.org/abs/2505.21351) (earlier title "…for Open-Loop Robotic Manipulation") | language instruction embedding | equivariant point-cloud U-Net trunk of a manipulation policy | **(b)**, restricted to symmetry-invariant (type-0) features | inside the U-Net, language-conditioning sites | **iFiLM** — an SE(3)-*invariant* FiLM: $(\gamma,\beta)$ are constrained so modulation commutes with 3-D rotation/translation of the scene, preserving the network's equivariance guarantee | vanilla FiLM destroys geometric equivariance when injected into an equivariant backbone | state of the art on 18 RLBench tasks under SE(3) and SE(2) scene perturbations, plus 4 physical tasks | ✘ |
| V2 | Li et al., *CogVLA: Cognition-Aligned Vision-Language-Action Models via Instruction-Driven Routing & Sparsification* | **NeurIPS 2025 Poster** ([forum](https://openreview.net/forum?id=Fg9HufTI0K)) | arXiv [2508.21046](https://arxiv.org/abs/2508.21046) (comments: "Accepted to NeurIPS 2025") | (i) the language instruction; (ii) the action intent | (i) the **vision encoder**; (ii) the **language-model trunk** | **(b)** per-channel signals used as **routing scores** | **2 sites**, one per stage of a 3-stage pipeline | **Modulation-as-routing**: the FiLM output is not only a gain — it drives *token aggregation* (EFA-Routing, compressing dual-stream visual tokens) and *token pruning* (LFP-Routing, deleting instruction-irrelevant tokens). FiLM becomes a discrete-computation controller | VLA post-training cost; sparsify without breaking cross-modal coherence | 97.4 % success on LIBERO / 70.0 % real-world; **2.5× cheaper training, 2.8× lower latency** vs. OpenVLA | ✘ |
| V3 | Zhang et al., *DyMoDreamer: World Modeling with Dynamic Modulation* | **NeurIPS 2025 Poster** ([forum](https://openreview.net/forum?id=SYKwGnik3w)) | arXiv [2509.24804](https://arxiv.org/abs/2509.24804) | inter-frame **differential observations** (a motion mask computed from consecutive frames) | the **recurrent state-space world model** (RSSM) | *not affine* — stochastic **categorical** modulation latents | inside the RSSM latent | Modulation of a *world model*, which the Dreamer lineage has never had (see [`film_in_rl_survey.md` §6.3](film_in_rl_survey.md)). The modulation variable is **discrete and stochastic**, not a continuous $(\gamma,\beta)$ | world models process frames holistically and cannot separate moving objects from static background | **156.6 %** mean human-normalised score on Atari 100k (new SOTA); **832** on DeepMind Visual Control; **+9.5 %** on Crafter at 1 M steps | ✘ |
| V4 | *Volume Transmission Implements Context Factorization to Target Online Credit Assignment and Enable Compositional Generalization* (e-nmRNN) | **NeurIPS 2025 Poster** ([forum](https://openreview.net/forum?id=S9Y89poypx)) | no arXiv located | **endogenous** neuromodulator concentrations, produced by the network's own activity via a rate reduction of neuromodulator release | recurrent connectivity (multiplicative gating) **and** the learning rule (plasticity gating) | modulation by a small set of neuromodulator species over a whole population → effectively **(c)/(e)** in our cells | network-wide | Derives the modulator from a **biophysical release model** rather than positing a side-network; the same signal gates *both* activity and plasticity; multiple modulator timescales emerge | unify "neuromodulators change dynamics" with "neuromodulators gate learning" | compositional generalisation on seq-to-seq tasks beating baselines with greater hyper-parameter robustness; in RL tasks the modulator dynamics learn to encode **reward-prediction error** | ✘ |
| V5 | Tessera et al., *HyperMARL: Adaptive Hypernetworks for Multi-Agent RL* | **NeurIPS 2025 Poster** ([forum](https://openreview.net/forum?id=56CgYnf9Dr)) | arXiv [2412.04233](https://arxiv.org/abs/2412.04233) (comments: "To appear at NeurIPS 2025") | agent identity | all policy parameters (generated) | full-weight (finer than (a)) | policy-wide | Argues the conditioner must be **decoupled from the observation input stream**: coupling agent IDs into observations causes cross-agent gradient interference; a hypernetwork separates observation- and agent-conditioned gradients | parameter sharing suppresses behavioural diversity in multi-agent RL | competitive with six baselines across 22 scenarios / up to 30 agents while retaining non-shared-parameter diversity; empirically reduced policy-gradient variance | ✘ |
| V6 | *Hyper-GoalNet: Goal-Conditioned Manipulation Policy Learning with HyperNetworks* | **NeurIPS 2025 Poster** ([forum](https://openreview.net/forum?id=aWWRPyGMie)) | no arXiv located | goal specification | generated policy parameters | full-weight | policy-wide | States the separation principle explicitly: "goal interpretation determines network parameters while state processing applies these parameters to the current observation" — i.e. the conditioner and the modulated stream must be different information sources | goal-conditioned manipulation under environmental randomisation | improvements over state of the art, largest under high-variability conditions; validated on real hardware | ✘ |
| V7 | *HyPoGen: Optimization-Biased Hypernetworks for Generalizable Policy Generation* | **ICLR 2025 Poster** ([forum](https://openreview.net/forum?id=CJWMXqAnAy)) | no arXiv located | task specification only (no demonstration data at test time) | generated policy parameters | full-weight | policy-wide | Structures the generator to *mimic an optimiser* (a fixed number of latent-space optimisation steps) rather than a plain feed-forward map | zero-demonstration transfer to unseen tasks | higher success rates than state of the art on locomotion and manipulation benchmarks for unseen target tasks | ✘ |
| V8 | *DIVERSE: Disagreement-Inducing Vector Evolution for Rashomon Set Exploration* | **ICLR 2026 Poster** ([forum](https://openreview.net/forum?id=kQjSUHC84V)) | no arXiv located | a searched latent vector (evolution strategy), **not** a task signal | a frozen pretrained network augmented with FiLM layers | **(b)** | added post-hoc to a pretrained model | Uses the FiLM latent as a **search space**, optimised by CMA-ES without gradients, to enumerate functionally different models of equal accuracy | building diverse model sets cheaply | comparable diversity to retraining at lower compute (MNIST, PneumoniaMNIST, CIFAR-10) | ✘ |

**Not RL** — V8 is included only because it demonstrates a novel *role* for the
modulation latent (a low-dimensional behaviour-space handle), which is directly
relevant to reading our modulator's hidden state as an interpretable variable.

### 3.2 Verified — secondary tier (robotics venues, clearly separated)

| # | Paper | Venue (verified) | ID | Conditioning signal | What is modulated | Granularity | Placement | What it CHANGES | Result | Held? |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | Reuss et al., *FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies* | **CoRL 2025** ([forum](https://openreview.net/forum?id=JeppaebLRD); arXiv comments: "Published at CoRL 2025") | arXiv [2509.04996](https://arxiv.org/abs/2509.04996) | flow-matching timestep + **action-type** embedding + embodiment metadata | every block of an 18-layer flow transformer action head | **(b)** signal; the novelty is on **Axis 3** | all 18 blocks | **Action-Space Global-AdaLN-Zero**: standard adaptive layer norm keeps *distinct scale-and-shift parameters per layer* (up to +30 % parameters); FLOWER **shares one set of modulation weights across all layers** and generates a distinct modulation signal per action category, zero-initialised. Lost per-layer expressiveness is restored by tiny per-layer LoRA adapters | **−20 % parameters with no performance loss** on CALVIN ABC (their Table 3); FLOWER overall reaches 4.53 on CALVIN ABC across 190 tasks / 10 benchmarks at 950 M parameters | ✘ |
| S2 | Yoon, Jeong et al., *Phase-Aware Policy Learning for Skateboard Riding of Quadruped Robots via Feature-wise Linear Modulation* (PAPL) | **ICRA 2026** (arXiv comments: "ICRA 2026") | arXiv [2602.09370](https://arxiv.org/abs/2602.09370) | **cyclic phase index** of the skateboarding gait — a *within-episode* variable, not a task ID | **both the actor and the critic** networks | **(a)/(b)** per-unit in dense layers | actor and critic | Conditions on an intra-task phase rather than a task distribution, and modulates the **critic as well as the actor** — the configuration Ben-Iwhiwhu et al. 2022 reported as unstable | validated command-tracking accuracy in simulation with per-component ablations; real-world transfer demonstrated | ✘ |

### 3.3 Venue **unconfirmed** — preprints only (do not cite as published)

These are topically important but I could not confirm acceptance anywhere. They are
listed separately and must not be mixed with §3.1/§3.2.

| # | Paper | ID | Why it matters | Status |
|---|---|---|---|---|
| U1 | *Hypernetwork-Conditioned Reinforcement Learning for Robust Control of Fixed-Wing Aircraft under Actuator Failures* | arXiv [2604.03392](https://arxiv.org/abs/2604.03392) | The only **direct FiLM-vs-LoRA head-to-head inside on-policy RL** (PPO) I found in 2024–2026. Conditioner = actuator-fault parameterisation; both are framed as parameter-efficient hypernetwork conditioning; reports generalisation to time-varying failure modes unseen in training | arXiv only, no venue comment |
| U2 | *SplitAdapter: Load-Aware Humanoid Loco-Manipulation via Factorized Adaptation* | arXiv [2606.03297](https://arxiv.org/abs/2606.03297) | **Hierarchical FiLM** driven by *two factorised context encoders* (object/load vs. robot dynamics) instead of one; explicitly benchmarks against "world-model FiLM baselines" | arXiv only |
| U3 | *Multi-Task Reinforcement Learning of Drone Aerobatics by Exploiting Geometric Symmetries* (GEAR) | arXiv [2602.10997](https://arxiv.org/abs/2602.10997) | Equivariant actor + **FiLM-based task modulation** + multi-head critic; 98.85 % success across aerobatic manoeuvres. Second instance of the symmetry + FiLM combination (cf. V1) | arXiv only |
| U4 | *MoE-ACT: Scaling Multi-Task Bimanual Manipulation with Sparse Language-Conditioned Mixture-of-Experts Transformers* | arXiv [2603.15265](https://arxiv.org/abs/2603.15265) | FiLM **modulates action tokens** during decoding while a sparse MoE handles task decomposition — the two competing conditional mechanisms used *together* rather than as alternatives | arXiv only |
| U5 | *Events as Triggers for Behavioral Diversity in Multi-Agent Reinforcement Learning* | arXiv [2605.12388](https://arxiv.org/abs/2605.12388) | An **event-triggered hypernetwork generating LoRA modules** over a shared team policy — conditioning on a *within-episode event* rather than an agent or task identity | arXiv only |
| U6 | *Dynamics-Aligned Shared Hypernetworks for Contextual RL under Discontinuous Shifts* (DMA*-SH) | arXiv [2602.06550](https://arxiv.org/abs/2602.06550) | One hypernetwork trained **only by dynamics prediction** generates adapter weights *shared across* the dynamics model, policy and Q-function; includes expressivity-separation theory for hypernetwork modulation and policy-gradient variance bounds | Submitted to ICLR 2026 (not accepted); appeared as an **EWRL 2025 workshop poster** ([forum](https://openreview.net/forum?id=6gdvQqkFKT)) |
| U7 | *Neuro-Vesicles: Neuromodulation Should Be a Dynamical System, Not a Tensor Decoration* | arXiv [2512.06966](https://arxiv.org/abs/2512.06966) | Explicitly argues that FiLM/hypernetwork-style modulation is the *degenerate dense, short-lived limit* of a stochastic vesicle-population process. Rhetorically relevant to our framing, but self-described as "early-stage theoretical design ... posted as a record of original contribution" with no experiments | arXiv only; treat as an opinion piece |

---

## 4. Category 2 — Papers **using** off-the-shelf modulation as a component

This is a sample of standard practice, not a census (see §2.2).

| Paper | Venue | ID | Modulation used | Where | Note |
|---|---|---|---|---|---|
| NVIDIA et al., *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots* | **venue unconfirmed** (NVIDIA technical report) | arXiv [2503.14734](https://arxiv.org/abs/2503.14734) | **adaptive layer normalisation** (DiT-style) for denoising-step conditioning; cross-attention for vision-language conditioning | action head | *Verified from the rendered full text*: "a variant of DiT … a transformer with denoising step conditioning via adaptive layer normalization" |
| Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control* | **venue unconfirmed** | arXiv [2410.24164](https://arxiv.org/abs/2410.24164) | **AdaLN-Zero only in the non-VLM baseline** $\pi_0$-small; the main model uses a decoder-only **mixture-of-experts** transformer in which the action expert attends to observation tokens | action expert | *Verified from full text*. This is the single clearest instance of affine modulation being **displaced by attention/MoE** at frontier scale |
| Reuss et al., *FLOWER* | CoRL 2025 | arXiv [2509.04996](https://arxiv.org/abs/2509.04996) | Global-AdaLN-Zero (see S1) + per-layer LoRA + cross-attention from the vision-language backbone | flow transformer | Uses **three** conditioning mechanisms simultaneously — affine, low-rank, attention |
| Yuan, *Unpacking the Individual Components of Diffusion Policy* | **venue unconfirmed** | arXiv [2412.00084](https://arxiv.org/abs/2412.00084) | FiLM conditioning of the denoising U-Net **on the observation sequence** — ablated against feeding the observation as a direct network input | denoiser | The most decision-relevant ablation found; numbers in §5.4 |
| *SPARC: Out-of-Distribution Generalization … Racing 100 Unseen Vehicles with a Single Policy* | **AAAI 2026 Oral** (arXiv comments) | arXiv [2511.09737](https://arxiv.org/abs/2511.09737) | **none** — context and history latents are **concatenated** before the final layers | contextual RL policy | *Verified from full text.* Negative evidence: at a 2026 AAAI Oral on contextual RL, concatenation is still the default |
| Huang et al., *MENTOR: Mixture-of-Experts Network with Task-Oriented Perturbation for Visual Reinforcement Learning* | **ICML 2025 Poster** ([forum](https://openreview.net/forum?id=t46uezeQH8)) | arXiv [2410.14972](https://arxiv.org/abs/2410.14972) | **MoE replaces the MLP backbone** of an RL agent | agent trunk | Conditional computation without any task distribution |
| *Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL* | **ICLR 2025 Spotlight** ([forum](https://openreview.net/forum?id=8oCrlOaYcc)) | — | SoftMoE in online RL | value network | Finds the benefit comes from tokenisation rather than expert specialisation — the sharpest mechanism analysis of MoE-in-RL |
| *Vocal Sandbox: Continual Learning and Adaptation for Situated Human-Robot Collaboration* | **CoRL 2024** ([forum](https://openreview.net/forum?id=ypaYtV1CoG)) | — | FiLM-family conditioning in a lightweight adaptable policy | policy | Component use in the BC-Z lineage |

---

## 5. Answers to the six questions

### 5.1 Q1 — Is FiLM still current in RL, or has it been displaced?

**Verdict: it survives, but it has migrated out of "reinforcement learning" proper
and into "robot policy learning", and inside that field it is usually not called
FiLM any more.** Three separate findings, each with different evidence strength.

**(i) In core RL, feature-wise modulation is effectively absent.** Across ~4 300
OpenReview records pulled from ICLR 2025/2026, NeurIPS 2024/2025 and ICML
2025/2026 with modulation-and-RL queries, the intersection of "FiLM-family term"
and "reinforcement learning" in title/abstract/keywords returned **single-digit
hits**, and none of them was a policy-gradient or value-learning paper conditioning
a policy with $(\gamma,\beta)$. The strongest confirming negative is
**SPARC (AAAI 2026 Oral)**: a 2026 paper whose entire subject is injecting context
into an RL policy, which concatenates. Where extra conditional *capacity* is added
in core RL, the 2024–2026 instrument is **mixture-of-experts** — ICLR 2025
Spotlight (SoftMoE) and ICML 2025 (MENTOR) — not affine modulation.

**(ii) In robot policy learning, the operator is standard, under the name adaLN.**
Adaptive layer normalisation is FiLM applied to a normalisation layer's output: a
conditioning embedding produces a per-channel scale and shift, and `AdaLN-Zero`
initialises the generator at zero so the block starts at the identity. Verified in
the rendered full text of **GR00T N1** (denoising-step conditioning via adaptive
layer normalisation) and **FLOWER** (CoRL 2025). This is exactly the mechanism
[`film_modulation_granularity_synthesis.md` §3.1](film_modulation_granularity_synthesis.md)
catalogues as conditional batch/instance normalisation, one architecture
generation later.

**(iii) At the frontier scale, attention and token conditioning displace it.**
$\pi_0$'s main model has no adaLN — conditioning is done by an action expert
attending to observation tokens inside a decoder-only mixture-of-experts
transformer; adaLN appears only in its non-VLM ablation baseline. This matches the
Dreamer finding already recorded in
[`film_in_rl_survey.md` §6.3](film_in_rl_survey.md): Dreamer 4 conditions by token
insertion, not modulation. **The honest trend statement is: affine modulation is
the default for sub-billion-parameter action heads and is displaced by attention
once the conditioner is a rich token sequence.**

**Consequence for us.** Our agent is a small recurrent policy with a
low-dimensional conditioner. That is squarely in the regime where affine modulation
is still the field's instrument, so the architecture choice is defensible. What is
*not* defensible by appeal to current practice is the specific conditioner (§5.4).

### 5.2 Q2 — What are the live variant lines in 2024–2026?

Six, named:

1. **Norm-layer modulation and its efficiency variants.** adaLN / AdaLN-Zero as the
   default conditioning primitive in diffusion and flow action heads, plus
   parameter-reduction variants — FLOWER's **Global-AdaLN-Zero** shares one
   modulation weight set across all layers and adds per-layer LoRA (CoRL 2025).
2. **Symmetry-constrained modulation.** EquAct's **iFiLM** makes the modulation
   SE(3)-invariant so it does not break an equivariant backbone (ICLR 2026); GEAR
   pairs equivariance with FiLM task modulation (unverified venue). This line did
   not exist before 2025 and is the clearest genuinely-new *constraint* on
   $(\gamma,\beta)$.
3. **Modulation-as-routing.** CogVLA (NeurIPS 2025) uses FiLM outputs to aggregate
   and prune tokens rather than merely scale them, turning a continuous operator
   into a discrete-computation controller.
4. **Hypernetwork / LoRA conditioning as the "more expressive rung".** HyperMARL,
   Hyper-GoalNet (NeurIPS 2025), HyPoGen (ICLR 2025); event-triggered LoRA
   generation and dynamics-trained shared hypernetworks in the unverified tier.
   This continues the line our corpus already holds (Beck 2023, Schöpf 2022).
5. **Mixture-of-experts as the competing conditional mechanism in core RL.**
   SoftMoE (ICLR 2025 Spotlight), MENTOR (ICML 2025).
6. **Biophysically-derived endogenous neuromodulation.** e-nmRNN (NeurIPS 2025) —
   the modulator is derived from a release model rather than posited as a
   side-network, and gates activity and plasticity with the same signal.

### 5.3 Q3 — Has anyone moved toward GROUPED modulation? *(the decision-relevant one)*

**No. Searched hard; found nothing new.** Dedicated queries for grouped / coarse /
population-shared / block-shared gains across ICLR 2025–2026, NeurIPS 2024–2025,
ICML 2025–2026 returned zero papers that share one $(\gamma,\beta)$ across a
contiguous block of units. **AlKilany & Goodman 2025 (which we hold) remains the
only exact precedent in existence as far as this search can determine**, exactly as
[`film_modulation_granularity_synthesis.md` §7.3](film_modulation_granularity_synthesis.md)
concluded over the corpus.

Two 2025 results are *adjacent* and both point the same direction — coarser
modulation costs nothing:

- **FLOWER's Global-AdaLN-Zero (CoRL 2025)** coarsens along **Axis 3** (one
  generator shared across all 18 layers instead of per-layer parameter sets) and
  reports **−20 % parameters with no performance loss**. This is the strongest
  published evidence to date that modulation capacity is over-provisioned by
  default. It is *not* our axis — it ties sites, not units — and note the
  compensation FLOWER pays: per-layer LoRA adapters restore some site specificity.
- **e-nmRNN (NeurIPS 2025)** implements grouping *by construction*: volume
  transmission means a handful of neuromodulator species reach an entire
  population, which is our cell **(c)/(e)** with a biological derivation. I have
  only the abstract, so the exact broadcast structure is unverified.

**What this means for the grouping screen.** Our sweep axis is still essentially
unprecedented in the conditional-architecture literature, and should be presented
as a transfer from spiking neuromodulation, citing AlKilany & Goodman — not as a
standard FiLM variant. FLOWER is now a second, independent citation for the weaker
claim "modulation capacity can be cut substantially at no cost", and it is a
top-tier-adjacent, quantitative one.

### 5.4 Q4 — Has anyone conditioned a modulator on the raw current observation while modulating the pathway that reads that same observation?

**No exact instance found — but this search produced the two most decision-relevant
data points in the whole survey, and they point in opposite directions.**

**Evidence *for* observation-as-conditioner (positive, quantitative).** Diffusion
Policy's design routes the **observation sequence through FiLM** into the action
denoiser instead of feeding it as a direct network input. The 2024 component study
(arXiv 2412.00084, venue unconfirmed) ablates exactly this, on 8 tasks:

| Task | Difficulty | FiLM conditioning | Observation as direct input |
|---|---|---|---|
| ManiSkill StackCube | easy | 99 % | 97 % |
| ManiSkill PegInsertionSide | hard | **80 %** | 44 % |
| ManiSkill TurnFaucet | hard | **59 %** | 27 % |
| ManiSkill PushChair | hard | **60 %** | 36 % |
| Adroit Door | hard | **95 %** | 79 % |
| Adroit Pen | easy | 71 % | 75 % |
| Adroit Hammer | hard | 17 % | 18 % |
| Adroit Relocate | hard | **64 %** | 2 % |

Author's stated takeaway: *"FiLM conditioning significantly enhances the
performance of the Diffusion Policy on hard tasks, but is not needed for easy
tasks."* This partially undercuts the design principle our corpus extracted from
Vecoven et al. 2020 ("the modulator must carry information the main network
lacks"), because here the conditioner carries **exactly the information the network
would otherwise have received as input**, and routing it through the multiplicative
path helps a lot. The residual difference from our design still matters: the
modulated stream is the *action-noise* pathway, which does not read the observation
by any other route. So this is "observation modulates a different stream", not full
self-conditioning.

**Evidence *against* entangling conditioner and observation (mechanistic).**
**HyperMARL (NeurIPS 2025)** finds that the common practice of *coupling the
conditioning variable (agent ID) into the observation vector* is what causes
cross-agent gradient interference, and that routing it instead through a separate
generator decouples the two gradient paths and reduces policy-gradient variance.
**Hyper-GoalNet (NeurIPS 2025)** states the separation as a design principle:
goal interpretation determines parameters, state processing applies them.

**Net verdict.** The literature's 2025 position is not "never condition on the
observation" — it is "**keep the conditioning gradient path separate from the
observation gradient path**". Our architecture violates that: one 16-unit recurrent
generator reads the observation and modulates the encoder that reads the same
observation, so both gradient paths run through the same input. HyperMARL supplies
the first published mechanism (gradient interference / variance inflation) for why
that could hurt, and it is measurable. The Diffusion Policy ablation supplies the
first published reason to think it might nevertheless help on *hard* tasks.

### 5.5 Q5 — Recent work on modulator instability (gain blow-up, temperature saturation, entropy collapse, plasticity loss)?

**No paper found that reports gain blow-up in a modulated network.** That gap,
first flagged in
[`film_modulation_granularity_synthesis.md` §8](film_modulation_granularity_synthesis.md)
and confirmed corpus-wide in [`film_in_rl_survey.md` §7](film_in_rl_survey.md),
survives a 2024–2026 web search. What the search *did* find:

1. **Zero-initialisation is now the field's universal prophylactic.** Every
   verified modulation variant here initialises the modulation generator at zero
   (`AdaLN-Zero`; FLOWER states "initialized with zeros for stable training"). This
   is the descendant of Beck et al. 2023's Bias-HyperInit, promoted from a paper's
   fix to an unremarked default. **Our modulator adds onto a zero-initialised
   per-neuron baseline, which is the right half of this practice; whether the
   generator's output weights are zero-initialised is a code question worth
   checking.**
2. **A modulation-architecture instability, not a gain instability.** FLOWER's
   appendix reports that a richer shared-plus-specialist modulation MLP produced
   **NaN losses** and slower convergence, and that the simpler shared Global-AdaLN
   worked best. That is the only 2024–2026 report I found of a modulation design
   being abandoned for numerical instability.
3. **The plasticity-loss line continues and still never mentions modulation.**
   ICLR 2026 alone has *The Rank and Gradient Lost in Non-stationarity: Sample
   Weight Decay for Mitigating Plasticity Loss in RL*
   ([forum](https://openreview.net/forum?id=5DpzzTPnJZ)) and *Barriers for Learning
   in an Evolving World: Mathematical Understanding of Loss of Plasticity*
   ([forum](https://openreview.net/forum?id=g6kof5fSba)); ICML 2025 has *Mitigating
   Plasticity Loss in Continual RL by Reducing Churn*
   ([forum](https://openreview.net/forum?id=EkoFXfSauv)). None connects to
   feature-wise modulation. The "multiplicative gains accelerate dormancy"
   hypothesis in [`film_in_rl_survey.md` §7](film_in_rl_survey.md) remains
   untested by anyone.
4. **Entropy collapse is now a large literature — but in language-model RL.**
   E.g. RiskPO (ICLR 2026) attributes limited reasoning gains to entropy collapse
   under group-relative policy optimisation. The symptom family matches our
   temperature head railing at its ceiling (a one-way drift into a state training
   cannot leave), but every instance is in a token-generation setting, so transfer
   is by analogy only.

**Verdict: our g=1 collapse — modulator leads early, then crashes with the
temperature at its clip and gain variance growing — is still unreported anywhere.
If it is real and characterised, it is publishable as a negative result, because
the field has no name for it.**

### 5.6 Q6 — Is there recent evidence of modulation helping in *single-task* RL?

**Yes — and it changes our corpus verdict.** The corpus conclusion was "gains come
from task diversity" ([`film_in_rl_survey.md` §5](film_in_rl_survey.md)). Four
2024–2026 results say the moderator is not diversity as such:

| Evidence | Setting | What it shows |
|---|---|---|
| **DyMoDreamer** (NeurIPS 2025) | Atari 100k, DeepMind Visual Control, Crafter — **one task per run, no task distribution** | Modulating a world model with a motion-derived signal sets a new state of the art (156.6 % human-normalised on Atari 100k). The conditioning variable carries *within-episode* structure (what is moving), not task identity |
| **MENTOR** (ICML 2025) | single-task visual RL | Conditional computation (mixture-of-experts backbone) improves sample efficiency with no task distribution at all |
| **Diffusion Policy component study** (venue unconfirmed) | 8 single tasks | FiLM helps on **hard** tasks (up to 64 % vs. 2 %) and does nothing on easy ones. **Difficulty, not diversity, is the moderator** |
| **PAPL** (ICRA 2026) | one skill (skateboarding), multiple **gait phases** | FiLM conditioned on an intra-episode phase variable, injected into **both actor and critic** |

**Reframed rule: modulation pays when the conditioning variable carries structure
the network can exploit and the task is hard enough for that structure to matter.
Task identity is only one such variable; motion, phase and difficulty also work.**

This is directly actionable. Our agent is single-task but has an obvious phase
variable — *uninjured vs. injured / hypervigilant*. PAPL is the first precedent for
phase-conditioned FiLM inside a single-task RL policy, and it modulates the critic
too. Re-specifying our conditioner from "the raw observation" to "an explicit
interoceptive-phase signal" moves us from *zero* precedent (§5.4) to *one verified*
precedent, without changing the injection machinery.

---

## 6. What we already hold vs. what is new

**We hold none of the papers in §3 or §4.** The FiLM folder's newest RL-relevant
holdings are Nikulin et al. 2023 (offline RL) and Moon et al. 2023 (Crafter); the
newest modulation holdings anywhere in the library are AlKilany & Goodman 2025 and
Rodriguez-Garcia et al. 2026 (noradrenergic gain modulation, in
`neuromodulatory_algorithms/sources/`). Dreamer 4 (Hafner et al. 2025, *Training
agents inside of scalable world models*) is held and is the reference point for the
token-conditioning alternative.

**What is genuinely new since our corpus was assembled:**

| New since the corpus | Where it lands |
|---|---|
| adaLN / AdaLN-Zero as the de-facto conditioning primitive of action heads | §5.1(ii) — updates the "field convention" claim in `film_modulation_granularity_synthesis.md` §4 |
| Symmetry-constrained modulation (iFiLM) | a constraint axis our taxonomy does not have |
| Modulation-as-routing (CogVLA) | a new *purpose* for the modulation output |
| Sharing modulation weights across all layers at no cost (FLOWER) | Axis 3 — first quantitative evidence that per-site heads are over-provisioned |
| Modulation inside a Dreamer-family world model (DyMoDreamer) | fills the precedent gap named in `film_in_rl_survey.md` §6.3 |
| Conditioner/observation gradient entanglement as a named failure mode (HyperMARL) | Axis 5 — the first mechanism story for our redundancy hypothesis |
| Difficulty, not task diversity, as the moderator of FiLM's benefit | revises `film_in_rl_survey.md` §5's verdict |

---

## 7. Prioritised download list

**Do not download anything from this list — the parent session owns downloading.**
Ordered by decision-relevance to the project, not by prestige.

### 7.1 High priority

| # | Paper | ID to fetch | Why first |
|---|---|---|---|
| 1 | *Unpacking the Individual Components of Diffusion Policy* | arXiv **2412.00084** | The only quantitative FiLM-vs-direct-input ablation where the conditioner **is** the observation. Directly bears on our Q4 divergence and on H1 |
| 2 | *FLOWER: Democratizing Generalist Robot Policies…* | arXiv **2509.04996** (CoRL 2025) | Global-AdaLN-Zero: the "coarser modulation costs nothing" result, with an ablation table. Second citation for the grouping screen's premise |
| 3 | *HyperMARL: Adaptive Hypernetworks for Multi-Agent RL* | arXiv **2412.04233** (NeurIPS 2025) | Names and measures the harm of coupling the conditioner into the observation stream — the mechanism our null-result series lacks |
| 4 | *Volume Transmission Implements Context Factorization…* (e-nmRNN) | OpenReview **S9Y89poypx** (NeurIPS 2025) | Endogenous, population-shared neuromodulation at a top venue; the closest thing to our architecture's biological story, and possibly a grouped-modulation precedent |
| 5 | *DyMoDreamer: World Modeling with Dynamic Modulation* | arXiv **2509.24804** (NeurIPS 2025) | Modulation inside a Dreamer-family world model, single-task gains. Relevant to both the Dreamer thread and Q6 |
| 6 | *Phase-Aware Policy Learning … via Feature-wise Linear Modulation* (PAPL) | arXiv **2602.09370** (ICRA 2026) | Phase-conditioned FiLM in a single-task RL policy, modulating **both actor and critic**. The template for re-specifying our conditioner |
| 7 | *EquAct* (iFiLM) | arXiv **2505.21351** (ICLR 2026) | The newest genuine constraint on $(\gamma,\beta)$; also the only 2026 ICLR paper with FiLM in its contribution list |

### 7.2 Medium priority

| # | Paper | ID | Why |
|---|---|---|---|
| 8 | *CogVLA* | arXiv **2508.21046** (NeurIPS 2025) | Modulation-as-routing; two injection sites with different conditioners |
| 9 | *GR00T N1* | arXiv **2503.14734** | Canonical citation for "adaLN is the standard action-head conditioner" |
| 10 | *$\pi_0$* | arXiv **2410.24164** | Canonical citation for "attention/MoE displaces affine modulation at frontier scale" |
| 11 | *MENTOR* | arXiv **2410.14972** (ICML 2025) | Conditional computation helping in single-task visual RL |
| 12 | *SPARC* | arXiv **2511.09737** (AAAI 2026 Oral) | The concatenation baseline that a 2026 top-tier contextual-RL paper still uses |
| 13 | *Hypernetwork-Conditioned RL … Actuator Failures* | arXiv **2604.03392** | FiLM-vs-LoRA head-to-head under PPO (venue unconfirmed — cite as preprint) |

### 7.3 Low priority

| # | Paper | ID | Why |
|---|---|---|---|
| 14 | *SplitAdapter* | arXiv **2606.03297** | Two-source factorised FiLM (preprint) |
| 15 | *GEAR* (drone aerobatics) | arXiv **2602.10997** | Second symmetry+FiLM instance (preprint) |
| 16 | *MoE-ACT* | arXiv **2603.15265** | FiLM and MoE used together (preprint) |
| 17 | *Events as Triggers…* | arXiv **2605.12388** | Event-triggered LoRA hypernetwork (preprint) |
| 18 | *Neuro-Vesicles* | arXiv **2512.06966** | Rhetorically useful critique of FiLM-as-neuromodulation; no experiments |

---

## 8. Verification status — what is confirmed and what is not

### 8.1 Confirmation method per entry class

| Class | Basis | Entries |
|---|---|---|
| **Venue verified from the proceedings record** | OpenReview note carries a `venue` field written by the venue itself (e.g. "NeurIPS 2025 poster", "ICLR 2026 Poster", "ICML 2025 poster", "CoRL 2025 Poster") | V1–V8, S1, MENTOR, SoftMoE, Vocal Sandbox |
| **Venue verified from the arXiv comments field** | Author-declared acceptance string on the arXiv record | CogVLA ("Accepted to NeurIPS 2025"), HyperMARL ("To appear at … NeurIPS 2025"), FLOWER ("Published at CoRL 2025"), PAPL ("ICRA 2026"), SPARC ("Accepted as an oral at AAAI 2026") |
| **Mechanism verified from rendered full text** | `arxiv.org/html/<id>` fetched and grep-ed | $\pi_0$ (AdaLN-Zero only in the non-VLM baseline), GR00T N1 (adaptive layer norm for denoising-step conditioning), FLOWER (Global-AdaLN-Zero definition + ablation), SPARC (concatenation), Diffusion Policy component study (the 8-task table quoted in §5.4) |
| **Venue unconfirmed** | arXiv record with no acceptance string, and no proceedings record found | U1–U7, GR00T N1, $\pi_0$, Diffusion Policy component study |

### 8.2 What could NOT be confirmed

1. **AAAI coverage is weak.** AAAI does not publish to OpenReview, Semantic Scholar
   was rate-limited throughout, and dblp timed out. AAAI 2025 and AAAI 2026 were
   reached **only** through arXiv comments-field matches. A FiLM-variant paper at
   AAAI 2025/2026 whose authors did not annotate their arXiv entry would be
   invisible to this survey. Treat AAAI as **not surveyed**.
2. **ICML 2026 and ICLR 2026 are searchable but recent.** Both had accepted-paper
   records in OpenReview at the time of search (ICML 2026 shows `regular` /
   `spotlight` labels), so the window genuinely covers them — but camera-ready text
   was not available for most, so only abstract-level claims are made about them.
3. **$\pi_0$ and GR00T N1 venues.** Both are widely-deployed industrial models
   whose publication venue I could not confirm from a source. Their *mechanisms*
   are verified from full text; their *venues* are not. Cite as preprints.
4. **e-nmRNN mechanism detail.** NeurIPS 2025 venue is verified; no arXiv version
   was located and the OpenReview HTML page is behind a bot challenge, so
   everything said about its modulation structure comes from the abstract. The
   claim that it constitutes a grouped-modulation precedent is **provisional**.
5. **DyMoDreamer mechanism detail.** Venue verified; `arxiv.org/html/2509.24804`
   returns 404 for all versions, so the description of its modulation as
   "stochastic categorical, not affine" rests on the abstract alone.
6. **No census is possible for component use.** See §2.2. Statements of the form
   "N papers use FiLM" appear nowhere in this document by design.
7. **Explicit negative findings** (searched, nothing found — these are results, not
   gaps in effort): no 2024–2026 top-venue work on **grouped/block-shared
   modulation** (§5.3); no instance of a modulator reading the same raw observation
   as the pathway it modulates (§5.4); no report of **modulation gain blow-up**
   anywhere (§5.5); no work connecting feature-wise modulation to **plasticity
   loss** (§5.5).

---

## 9. Cross-references and recommended follow-ups

**Composes with:**

- [`film_in_rl_survey.md`](film_in_rl_survey.md) — the corpus-based companion. This
  document **revises** two of its verdicts: its §5 claim that modulation gains
  require task diversity becomes "require exploitable conditioner structure plus
  task difficulty" (§5.6 here); its §7 claim that no failure mode matches ours is
  strengthened, since a fresh 2024–2026 search still finds no gain blow-up.
- [`film_modulation_granularity_synthesis.md`](film_modulation_granularity_synthesis.md)
  — taxonomy source. Its §7.3 judgment that grouped modulation has exactly one
  precedent **survives** this search unchanged; its Axis 3 verdict ("shared
  generator + per-site heads is the norm") gains a first counter-instance in
  FLOWER's Global-AdaLN-Zero.
- [`../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md`](../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md)
  — its §8.2 conclusion about our modulator's input is now supported by an
  independent, top-venue, *mechanistic* argument (HyperMARL's gradient-interference
  result) rather than only by Vecoven's assertion.

**Project docs:**
[NEUROMODULATION_ALGORITHM.md](../../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) (H1–H5, injection points) ·
[NMN_FILM_GROUPING_SCREEN.md](../../../experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md) ·
[NMN_PERFORMANCE_DIAGNOSIS_v8.md](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)

### Recommended follow-ups

- **`experiment-designer` — a phase-conditioned modulator arm.** §5.6 shows the
  live single-task precedent is *phase* conditioning (PAPL, ICRA 2026), not raw
  observation. Add an arm whose modulator reads an explicit injury/health phase
  variable instead of the full observation vector. This costs one config and
  converts our sharpest divergence from the literature into a supported design.
- **`experiment-analyzer` — measure the gradient-interference signature.**
  HyperMARL's diagnosis is a measurable quantity (policy-gradient variance, and the
  correlation between the modulator's and the encoder's gradients on the shared
  observation input). Computable from existing runs; would test the redundancy
  hypothesis directly rather than by ablation.
- **`senior-developer` — confirm the generator's output-weight initialisation.**
  §5.5 establishes zero-initialisation of the modulation generator as the field's
  universal default. We add onto a zero-initialised per-neuron baseline, which is
  half the practice; whether the generator heads themselves start at zero is a
  one-line check with a real stability implication.
- **`literature-reviewer` — pull items 1–7 of §7.1** once the parent session has
  downloaded them. Items 1 and 3 are the two that bear directly on the null result;
  items 2 and 4 bear on the grouping screen.
- **`pi` — a framing decision.** §5.1 shows the field's centre of gravity for
  affine modulation has moved into robot policy learning, while core RL has moved
  to mixture-of-experts. If we intend to publish this as an RL contribution, the
  positioning question (is this an RL paper, a neuro-AI paper, or a robot-learning
  paper?) is now a portfolio-level call with evidence attached.

---

## Full-text verification of this survey — literature-reviewer, 2026-08-05

Seventeen PDFs from §7's download list were acquired; the ten implementing
scale-and-shift modulation were read end-to-end and reviewed in
[`../modulation_in_rl/modulation_in_rl_lit_review.md`](../modulation_in_rl/modulation_in_rl_lit_review.md).
That review's §5 records every claim confirmed, corrected or refined. Summary of the
**corrections** that affect this document:

- **§5.3 superseded.** Grouped modulation is now attested — EquAct's `iFiLM` shares one
  scalar across a $(2l+1)$-unit irrep block — though as a **by-product of Schur's
  lemma**, and GEAR's independent RL test finds the constraint *harmful* where the
  symmetry is approximate.
- **§5.3 / §5.5 on FLOWER.** Two corrections. (i) The "−20 % parameters at no cost"
  result is confounded: **per-layer LoRA adapters compensate for the coarsening and are
  never ablated**, so it is parameterisation evidence, not capacity-redundancy
  evidence. (ii) The 20 % is of the **action head**, and the counterfactual parameter
  count is not published.
- **§5.5's NaN attribution corrected.** FLOWER's NaN losses came from a
  **mixture-of-experts** design (App. A.1's own heading), not from an affine-modulation
  design. Correctly stated the finding is *more* favourable to coarse modulation: the
  simpler, coarser modulation was the stable fallback. **Gain blow-up remains
  unattested across all ten papers** — but Marquis & Farhood 2026 now supplies a
  *measurement* (Lipschitz bound as the product of spectral norms, correlating with
  performance across six configurations) and a *remedy* (spectral normalisation).
- **§3.1's EquAct row.** Granularity is not *"(b) restricted to type-0 features"*.
  Modulation reaches **every** type $l$; what is restricted is the form — shared scalar
  gain with no offset above $l=0$, full affine only at $l=0$.
- **§3.1's CogVLA row.** "Modulation-as-routing" over-reads the mechanism. FiLM
  conditions the representation; **separate learned routers** compute the aggregation
  and pruning scores (Eqs. 11, 14). It is not a new role for $(\gamma,\beta)$.
- **§5.6 strengthened.** Single-task gains now rest on **four** reviewed papers — PAPL,
  Yuan 2024, Marquis, SplitAdapter — two of which this survey had not classified that
  way. The moderators are difficulty (Yuan), distribution shift (Marquis) and load
  beyond training range (SplitAdapter).
- **§5.4's net verdict stands and sharpens.** No reviewed paper conditions on a full
  raw observation; corpus-wide conditioner dimensionality is 1–32.
- **Venue provenance.** EquAct's held PDF says *"Preprint. Under review."* and PAPL's
  has no venue line, so the ICLR 2026 / ICRA 2026 attributions are external
  (OpenReview record, arXiv comments) rather than PDF-verifiable. GEAR's "98.85 %" is
  a correct aggregate of four per-task rates, reconstructed rather than quoted.
