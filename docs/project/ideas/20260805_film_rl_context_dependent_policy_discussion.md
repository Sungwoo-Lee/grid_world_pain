---
title: "FiLM-style modulation in RL — discussion log toward a context-dependent policy"
topic: modulation_in_rl
status: active
created: 2026-08-05
last_updated: 2026-08-31
related:
  - ../references/modulation_in_rl/modulation_in_rl_lit_review.md
  - ../references/FiLM/film_rl_recent_variants_survey.md
  - ../references/FiLM/film_in_rl_survey.md
  - ../references/FiLM/film_modulation_granularity_synthesis.md
  - ../references/neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md
---

# FiLM-style modulation in RL — discussion log toward a context-dependent policy

## 1. What this document is

This is a record of a working discussion, held 2026-08-05, about a single
question: **how does the published literature actually build networks whose
behaviour changes with a context signal, and what does that imply for the agent
we are building?**

The concrete goal behind the question is a *context-dependent policy*. We want a
survival agent whose behaviour changes with its own internal bodily state — for
example, an agent that **hides in bushes when its injury level is high** and
**ignores the same predator when its injury level is low**. This is a narrower
goal than the project's longer-term ambition (using the same mechanism to tune
learning hyperparameters), and it was deliberately scoped down for now.

The mechanism under discussion is **FiLM** — *Feature-wise Linear Modulation*.
The idea is simple: a small side-network reads some context signal and outputs
two vectors, a multiplier and an offset, which rescale and shift the hidden
activations of the main network. Nothing else about the main network changes.
Because a neuromodulator in the brain is, computationally, something that scales
and shifts neural activity, FiLM is the natural formal analogue — which is why
this project uses it.

The discussion covered four things: what the literature's conventions actually
are, where our current implementation sits relative to them, which published
algorithm is closest to the context-dependent-policy goal, and what could be
borrowed from a distributional-RL method called IQN. **Section 8 is the most
actionable part**; sections 2–4 are the evidence base.

Throughout, a distinction is maintained between what is **verified from full
text**, what rests on an **abstract or an external record**, and what is
**our inference**. Several claims made earlier in the discussion were corrected
once the papers were read in full; those corrections are marked.

---

## 2. The mechanism, written out

FiLM (Perez et al., AAAI 2018). Given a hidden activation vector
h ∈ ℝ^d and a conditioning input c:

$$
\mathrm{FiLM}(h \mid c) \;=\; \gamma(c) \odot h \;+\; \beta(c),
$$

where γ(c), β(c) ∈ ℝ^d come from a *generator*
sub-network and ⊙ is the element-wise (Hadamard) product. In words: the
context independently rescales and shifts every feature.

Three design axes recur throughout this document, and the whole literature can
be filed against them:

| Axis | Question | Cells |
|---|---|---|
| **1 — Granularity** | how many distinct (γ,β) values exist per modulated tensor | (a) per-unit · (b) per-channel · (c) grouped · (d) one scalar per layer · (e) one signal for the whole network |
| **2 — Placement** | which layers are modulated, how many sites | — |
| **3 — Parameterisation** | do sites share a generator, share parameters, or have their own | — |

Note that in a **dense** layer, cells (a) and (b) coincide: "per channel" and
"per unit" are the same thing. The distinction only matters for convolutions.

---

## 3. What the literature's conventions actually are

Two literatures were surveyed, and **they converge on opposite answers.**

| Literature | Corpus | Convention |
|---|---|---|
| FiLM / conditional architectures | 23 FiLM + 8 hypernetwork papers | **Cell (a)/(b) — per-unit.** 14 of 15 modulating papers |
| Computational neuromodulation | 41 papers | **Cell (e) — one global scalar.** 15 of ~24 papers |

This is a field boundary, not a disagreement about what is correct. FiLM comes
from vision, where "per channel" means one knob per feature map. The
neuroscience-inspired work inherits the biological picture of a neuromodulator
as a chemical broadcast over a region, so one scalar is the default.

**On "layer-wise", which is an ambiguous term.** Two readings must be kept apart:

- *One scalar per layer* (cell d) — **unattested in both corpora.** Nobody does this.
- *Each layer has its own parameter set* — **this is the standard**, implemented
  as one shared generator with a separate output head per site. Chauhan's survey
  names the pattern `generate-multiple`.

**On grouping.** Sharing one (γ,β) across a *block of units* is
essentially absent: one instance in ~100 papers. Where the literature does share
parameters, it ties the **generator** (rank-1 factorisation, chunked generation,
tiled kernels) while the emitted signal stays per-unit. The motivation is always
parameter efficiency; **regularisation is never offered as a rationale for
grouping.**

The biological analogue of grouping — modulating distinct sub-populations
differently — is rich (six schemes across the neuromodulation corpus) but is
always **functional**: cell type, excitatory vs. inhibitory, projection target.
Never contiguous-by-index.

---

## 4. Where our implementation sits

Our neuromodulator (`src/models/neuromodulator.py`, class `NeuromodulatorRNN`)
is one shared GRU with 16 hidden units that reads the **raw observation** and
feeds six linear heads:

| Head | Target | Injection site |
|---|---|---|
| γ,β (unimodal) | encoder stage 1 output | after the per-sensor MLPs |
| γ,β (multimodal) | encoder stage 2 output | after the fusion hub |
| memory | task-GRU update gate | inside the recurrent cell |
| temperature (scalar) | policy logits | logits / T |

Each non-scalar head emits ⌈ 128/G ⌉ raw values, expanded to the
128-unit target by repeating each value across a contiguous block:

$$
\mathrm{sig} \;=\; \underbrace{\texttt{repeat}(\mathrm{raw},\, G)[:128]}_{\text{grouped dynamic part}} \;+\; \underbrace{b}_{\substack{\text{per-neuron learned}\\ \text{baseline},\ b \in \mathbb{R}^{128}}}
$$

with G = `grouping_size` swept 1 → 128.

### 4.1 Scorecard against the conventions

| Axis | Ours | Verdict |
|---|---|---|
| Per-site parameters (shared generator + per-site heads) | yes | ✅ **the most conventional part of our design** |
| Identity-ish initialisation | per-neuron baseline + pass-through init | ✅ conventional |
| Multi-target from one trunk | 6 heads | ✅ mostly conventional |
| Critic directly modulated | no | ✅ matches the cautious precedent |
| **Granularity: grouped, contiguous blocks** | cells (c) | ⚠️ one precedent in ~100 papers |
| **Generator reads the raw observation** | yes | ⚠️ unattested; now *negative* evidence exists |
| **Temperature on the same trunk as perceptual gain** | yes | ⚠️ unprecedented |

The parts that look exotic are conventional. The genuinely unusual choices are
the last three — and **grouping is not the most consequential of them.**

### 4.2 An interpretation hazard in the grouping sweep

Because a full 128-dimensional per-neuron baseline b is added at every G,
each neuron retains full *static* freedom regardless of G. Therefore
`grouping_size` controls **dynamic resolution only, not total modulation
capacity.** A null result from the grouping sweep cannot distinguish
"grouping is inert" from "the baseline absorbed the modulation".

Notably, this same decomposition appears verbatim in AlKilany & Goodman (2025),
with the rationale stated: it keeps the network "fully heterogeneous". So the
design is not novel — it reproduces the cited paper — but the confound stands.

---

## 5. What the evidence says about our three unusual choices

### 5.1 Grouping — two independent negatives

**A refereed precedent now exists, but it does not support us.** EquAct's
`iFiLM` layer ties an entire block of features to one scalar. Given a
spherical-Fourier feature block of degree l,

$$
c_l = \bigl[c_l^{-l}, \dots, c_l^{l}\bigr] \in \mathbb{R}^{2l+1},
\qquad
c_l' = \alpha_l(k)\, c_l ,
$$

one gain α_l multiplies all 2l+1 components (block sizes 1,3,5,7),
with **no additive offset** for l>0; full affine survives only on the
invariant l=0 channels.

But the grouping is **forced by Schur's lemma**, not designed: a per-component
gain commutes with the rotation matrix only if all its entries are equal. Group
boundaries are fixed by representation theory and **cannot be swept**. The paper
never uses the words "grouped", "granularity", or "coarse".

**And GEAR supplies the control the literature was missing:** substituting
`iFiLM` for ordinary FiLM in an RL policy **loses** — 95.46 % vs 98.85 % mean;
Roll 87.70 vs 99.71.

> **Extracted rule.** Grouping pays *iff* the tied units are genuinely
> interchangeable. EquAct's are — they are components of one irreducible
> representation. Our contiguous-index blocks have no such justification.

Add AlKilany's own G-sweep, which came out **flat**, and there are three
independent reasons to doubt the grouping axis.

### 5.2 Self-conditioning — the evidence turned negative

Earlier in this discussion it was said that the Diffusion Policy component study
was a precedent for conditioning a modulator on the observation it modulates.
**Full-text reading corrected this.** Its two arms are:

- **FiLM arm** — denoiser input is the noisy action chunk alone, x = A^k; the
  observation enters *only* through FiLM(x  O).
- **Direct-input arm** — x = [A^k, O], no modulation.

Both arms give the denoiser identical information, so it is a **pure routing
comparison**. But the modulated stream has *no other route* to the observation,
so this is not the self-conditioning loop we run.

Meanwhile **HyperMARL (NeurIPS 2025)** supplies active counter-evidence:
coupling the conditioner into the observation stream causes **cross-agent
gradient interference**, and its `w/o GD` ablation — *which is our
configuration* — degrades on both environments. Self-conditioning also violates
assumption (A3) of HyperMARL's own policy-gradient-variance proof.

**The nearest positive precedent is PAPL** (§6), where the conditioner is both
the FiLM input and a direct network input — but it is **2-dimensional**, not a
full observation vector. That difference is the whole point: a low-dimensional
conditioner cannot be a redundant copy of the encoder.

### 5.3 Instability — still nameless

**Gain blow-up is unattested across all 10 reviewed papers.** A survey claim
that FLOWER reported NaN losses from modulation was **corrected**: FLOWER's own
appendix heading is *"Mixture-of-Experts Approaches"* — the failing design was
expert MLPs, not affine gain. Instabilities in this corpus cluster on
mixture-of-experts isolation and capacity knobs, not on γ/β.

Two tools from Marquis & Farhood are computable on runs we already have:

1. A **Lipschitz bound** — the product of layer spectral norms — which tracked
   performance across six configurations (28.38 → 3.73), with **spectral
   normalisation** recommended as the remedy.
2. A **gain-damping parameterisation**, p_scale = 1 + 0.1 p_h.

---

## 6. The closest published algorithm to the goal: PAPL

*Phase-Aware Policy Learning for Skateboard Riding of Quadruped Robots via
Feature-wise Linear Modulation* — Yoon, Jeong et al., arXiv 2602.09370.
Venue: the PDF states none; "ICRA 2026" comes from the arXiv comments field.

**The problem it solves.** A quadruped riding a skateboard is *one task made of
several qualitatively different skills in a cycle*: a **pushing** phase (one
foot shoving on the ground), a **carving** phase (all four feet on the deck,
steering by leaning), and transitions. One network trained on everything blurs
the phases; separate networks per phase cannot share general quadruped knowledge
and hand off badly. Their mixture-of-experts attempt **failed outright** — the
pushing expert collapsed.

**The solution.** One network re-tuned on the fly by a clock. The phase
embedding is just two numbers,

$$
\Phi_t = (\cos\phi_t,\ \sin\phi_t)^\top \in \mathbb{R}^2,
\qquad
\phi_t = \frac{2\pi t}{T_\phi} \bmod 2\pi ,
$$

and every layer of **both the actor and the critic** is modulated pre-activation
with no normalisation anywhere in the modulation path:

$$
h_\ell = \sigma\bigl(\gamma_\ell(\Phi_t) \odot z_\ell + \beta_\ell(\Phi_t)\bigr),
\qquad z_\ell = W_\ell h_{\ell-1} + b_\ell .
$$

Two conditioner dimensions expand to roughly 1900 modulation parameters per
forward pass. Encoders are **not** modulated.

**Why it maps onto our goal:**

| Our design | PAPL |
|---|---|
| conditioner = injury level (low-dim, internal, varies within episode) | conditioner = gait phase (2-dim, internal, varies within episode) |
| one environment, behaviour varies by internal state | one skill, behaviour varies by phase |
| hiding is worth more when injured | reward is phase-conditioned |
| the same signal also reaches the policy | Φ_t is *also* a direct network input |

### 6.1 Three things PAPL gives us

1. **It licenses a conditioner drawn from the observation.** Φ_t enters each
   network twice — as raw input features and as the FiLM conditioner. So the
   `No-FiLM` ablation is *not* "with vs. without phase information"; it is a
   **pure routing comparison**, and the multiplicative route wins.
2. **A principled criterion for modulating the critic.** PAPL's reward terms do
   not merely take different values by phase — they measure *different
   quantities*. For instance the tracking-error term is
   $$
   \varepsilon_1 = \begin{cases}
     \lvert c_\omega - \omega^S_{D,z}\rvert & M(\phi_t)=\text{CARVING}\\[2pt]
     0.6\lvert c_v - v^S_{D,x}\rvert + 0.2\lvert v^S_{D,y}\rvert + 0.2\lvert\omega^S_{D,z}\rvert & M(\phi_t)=\text{PUSHING}
   \end{cases}
   $$
   Hence the value function genuinely differs per phase.
3. **A pre-check to run before building anything.** They verified that the
   **critic-value histogram separates by phase** *before* defending the design.

### 6.2 The criterion, and how our project stands against it

> **PAPL's rule is: modulate the critic when the reward itself is conditioned on
> the modulating variable.**

**CORRECTION (2026-08-31, verified in `src/environment/core.py`).** An earlier
version of this section claimed our project does not meet this condition. That
was wrong, and the error propagated into the site-refactor plan before being
caught. **Our reward IS a function of injury.** The live configs
(`basic/04`, `sensory_directional/A_baseline`) set `use_homeostatic_reward: True`,
and the reward is the reduction in homeostatic drive:

$$
r_t = D(s_{t-1}) - D(s_t),
\qquad
D = \bigl\lVert (\mathrm{satiation},\ \mathrm{injury}) - (\mathrm{setpoint},\ 0) \bigr\rVert_2
$$

so healing lowers D and is rewarded directly. The condition is met more strongly
than a merely additive injury term would give, because the marginal value of
healing depends on the whole body state:

$$
\frac{\partial D}{\partial\, \mathrm{injury}} = \frac{\mathrm{injury}}{D}
$$

Healing is worth more when injury is high, **and** its worth depends on how hungry
the agent is, since satiation enters through D. Hunger and injury are coupled in
the value function rather than separable — the "managing conflicting needs" case.
**Critic modulation is therefore licensed on PAPL's own criterion.**

A caution for anyone reading the reward code: two different "drive" quantities
live in the same function. `calculate_drive` (the unnormalised L2 norm above) is
what feeds the reward; `drive_hunger` + `drive_injury` (normalised, sum of
squares) are computed for logging only, and the inline comment describing them
sits directly above the reward computation.

The **remaining** disanalogy is different: PAPL's conditioner is an
**open-loop clock**: perfectly predictable, so
the modulator has a trivially learnable signal. Injury level is sensed and
changes contingently. That is a real disanalogy.

---

## 7. Venue reality check

The anchoring question was asked directly: is there a top-venue paper doing
this? The answer is no.

| Paper | Venue | Fits the goal? |
|---|---|---|
| PAPL | ICRA 2026 (from arXiv comments) | ✅ closest structurally — but robotics tier |
| Ben-Iwhiwhu 2022 | *Neural Networks* (Elsevier journal) | closest neuromodulation-RL; a journal |
| Vecoven 2020 | *PLOS ONE* | journal |
| **IQN (Dabney et al. 2018)** | **ICML 2018** | multiplicative conditioning of a value net on a varying scalar — conditioner is exogenous |
| EquAct / CogVLA | ICLR 2026 / NeurIPS 2025 | top venue, but multi-task, task/language conditioned |
| **Perez et al. 2018 (FiLM)** | **AAAI 2018** | the mechanism itself; not RL |

**PAPL's own citations for FiLM are [30] Perez 2018 (AAAI), [31] Bauersfeld
2023 (ICRA), [32] Chi 2023 (IJRR).** So even PAPL has no top-tier RL ancestor
for the idea — its only high-venue anchor is the original FiLM paper.

> **Conclusion.** There is no top-tier-conference paper doing FiLM-style
> modulation to make a *single-task* RL policy context-dependent on a *sensed
> internal state*. The space is open — but there is also no top-venue result to
> stand on. A realistic citation stack is **Perez 2018 (AAAI, mechanism) + IQN
> (ICML, value-side precedent) + PAPL (ICRA, closest application) +
> Ben-Iwhiwhu (journal, neuromodulation framing)**.

---

## 8. IQN, and the idea most worth stealing

*Implicit Quantile Networks for Distributional Reinforcement Learning* —
Dabney, Ostrovski, Silver & Munos, **ICML 2018**.

### 8.1 The main idea

Standard RL learns the **expected** return. Distributional RL learns the whole
**distribution** of returns. IQN represents it *implicitly*: rather than a fixed
set of quantiles, it learns a function mapping any quantile level
τ ∈ [0,1] to its return value.

$$
Z_\tau(s,a) := F^{-1}_{Z(s,a)}(\tau),
\qquad
Z_\tau(s,a) \;\approx\; f\bigl(\psi(s) \odot \phi(\tau)\bigr)_a ,
$$

with ψ : 𝒮 → ℝ^d the state embedding (a convolutional
trunk), φ : [0,1] → ℝ^d the τ-embedding, and
f : ℝ^d → ℝ^|𝒜| a small head.

Because τ is sampled randomly during training, the network learns to be
**all quantiles at once**. At acting time, the choice of τ *is* a risk
preference: low τ acts on the pessimistic tail (risk-averse), high τ
on the optimistic tail.

### 8.2 This is FiLM

Set h = ψ(s), c = τ, γ(τ) = φ(τ), β(τ) = 0:

$$
\mathrm{FiLM}\bigl(\psi(s) \mid \tau\bigr) = \phi(\tau) \odot \psi(s),
$$

which is IQN's equation exactly. **IQN is FiLM with no shift term and a
cosine-basis generator**, published at ICML 2018 and validated at Atari-57
scale. It predates the term in the RL literature.

Note φ maps into the **same dimension d as the state embedding** — a
single scalar expands to a full-width per-unit gain vector, the same shape as
our G=1 setting.

### 8.3 The cosine embedding, and why the Fourier basis matters

$$
\phi_j(\tau) \;=\; \mathrm{ReLU}\!\left( \sum_{i=0}^{n-1} \cos(\pi i \tau)\, w_{ij} + b_j \right),
\qquad n = 64 .
$$

**The problem it solves.** Feed a scalar straight into a linear layer and each
gain is γ_j(τ) = w_jτ + b_j — a straight line. Add a ReLU and you
get one kink; still monotone. Behaviours like *"flat until a threshold, then
rise"* or *"high at both extremes, low in the middle"* are unrepresentable.

**The fix.** Expand τ first through fixed non-linear functions
{cos(πiτ)} for i = 0…63. Over τ ∈ [0,1], term i completes
i/2 periods — i=0 is constant, i=1 is monotone 1 → -1, i=2 is one
full period, i=63 oscillates ~31 times. A *single linear layer* over these
features then produces

$$
\gamma_j(\tau) = \sum_{i=0}^{63} w_{ij}\cos(\pi i \tau) + b_j ,
$$

a **truncated Fourier cosine series**. Any reasonably smooth function on [0,1]
is approximable this way.

**The key property:** this is non-linear and non-monotone *in τ*, yet
linear *in the learnable weights* w. Rich conditioning shapes, easy
optimisation.

**Why this matters more than it sounds.** Neural networks have a documented bias
toward low-frequency functions and are specifically poor at learning
high-frequency detail from **low-dimensional** inputs (the finding behind
Fourier features in NeRF and positional encodings in transformers). An injury
scalar is exactly that regime.

**Concretely, for our agent.** Suppose the correct behaviour is: low injury →
forage boldly; medium injury → hide; very high injury → forage anyway, because
starvation now kills faster than the predator. That is a **U-shape** in injury,
which a linear map cannot represent *at all*. Cosine features build it from two
or three low-frequency terms.

A further detail: the trailing **ReLU forces gains to be non-negative**, so
modulation can suppress or amplify a feature but never invert its sign. We do
not impose this — our γ may go negative — and that is one plausible
contributor to the G=1 instability.

### 8.4 IQN ablated the combination rule — our third routing vote

| Form | Result |
|---|---|
| ψ(s) ⊙ φ(τ) — Hadamard | **robust, slightly preferred — chosen** |
| [ψ(s)  φ(τ)] — concatenation | worse |
| ψ(s) ⊙ (1 + φ(τ)) — residual | worse |

Multiplicative beat concatenation for a **third** time (after Yuan's Table 5 and
PAPL's `No-FiLM`), in a different algorithm and setting.

The second row also qualifies earlier advice: ψ ⊙ (1+φ) is exactly
CogVLA's identity-centred form, and IQN found plain Hadamard slightly better. So
"always centre modulation at identity" is **contested, not universal**.

### 8.5 The training-scheme alternative — the most interesting idea in this log

IQN samples τ at **training** time and applies the risk distortion only at
**acting** time. The value head learns every quantile; the modulator merely
*selects* among them afterwards.

Our design does the opposite — the modulator changes the policy during both
training and acting. Adopting IQN's split would mean: **train the full return
distribution once, then let injury level pick which quantile to act on.**

This is attractive for three separate reasons:

1. It **decouples the modulator from training dynamics**, which sidesteps the
   gradient-interference problem HyperMARL identified.
2. It **removes the gain-blow-up failure mode**, because nothing multiplicative
   is trained through.
3. It gives "high injury → act risk-aversely" a **literal implementation**
   rather than a metaphor — the exact behaviour we want (hide when injured) is
   the definitional consequence of sampling a low quantile.

**The gap:** IQN is value-based DQN with discrete actions; we run PPO, on-policy
actor–critic. Porting requires a distributional critic for PPO or a
distributional-PPO variant. That is real work, not a config change.

---

## 9. Open questions this discussion did not settle

1. **Does our value function already separate by injury level?** PAPL's
   pre-check, applied to our existing baseline runs. If value estimates at high
   vs. low injury do not separate, the conditioner has no signal to exploit and
   FiLM will not create one. **This is cheap and should come first.**
2. **Should the conditioner be the full observation or a low-dimensional
   interoceptive slice?** Every precedent that works uses a low-dimensional
   conditioner. This is the single most consequential open design choice.
3. ~~**Is a reward that is *not* conditioned on injury a blocker for critic
   modulation?**~~ **RESOLVED 2026-08-31 — the premise was false.** The reward IS
   a function of injury (see the correction in §6.2), so PAPL's criterion is met
   and critic modulation is licensed. The residual questions are narrower: PAPL's
   conditioner is an open-loop clock versus our contingent sensed injury, and
   Marquis shows the sign of critic conditioning is mechanism-dependent
   (positive for FiLM, negative for LoRA).
4. **Fourier expansion of the conditioner** — untested here, cheap, and
   addresses a documented failure mode.
5. **Non-negative gains** — should γ be constrained ≥ 0?
6. **Is the grouping sweep worth continuing** given three independent negatives
   and the baseline-absorption confound?
7. **e-nmRNN** (NeurIPS 2025, OpenReview `S9Y89poypx`) remains unread — the last
   candidate for a *designed* grouping precedent.

---

## 10. Provenance notes

- Venue claims for **EquAct** and **PAPL** come from external records
  (OpenReview; arXiv comments), *not* from the held PDFs, which state
  "Preprint. Under review." and nothing respectively. **CogVLA's NeurIPS 2025
  status is printed on its own page-1 footer.**
- The **Diffusion Policy component study** has no seeds, no error bars and no
  evaluation-episode counts; its bibliography is broken (reference [1], cited as
  the ACT paper, is a 1970 networking paper). Cite as *suggestive single-preprint
  evidence*.
- **FLOWER's** -20% parameter result is confounded: per-layer LoRA adapters
  compensate for the removed per-layer modulation and are **never ablated**.
- **AAAI was effectively not surveyed** in the recent-variants search (not on
  OpenReview; Semantic Scholar rate-limited; dblp timed out).
