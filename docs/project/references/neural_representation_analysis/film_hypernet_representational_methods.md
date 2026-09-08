---
title: "How FiLM and hypernetwork papers look inside the network — a methods survey of the existing corpus"
topic: neural_representation_analysis
status: curated
created: 2026-09-08
last_updated: 2026-09-08
author: literature-curator
scope: |
  A cross-paper methods survey over the reference library already held in
  docs/project/references/. It asks one question of 31 empirical papers whose
  own mechanism is FiLM-style feature-wise modulation or a weight-generating
  hypernetwork: does the paper measure anything INSIDE the trained network, or
  does it report task performance and ablations only? Where a paper does
  measure something inside, the technique is named, described by its inputs /
  outputs / required contrast / licensed conclusion, and rated for whether it
  would run on this project's saved checkpoints. No new papers were extracted;
  everything is synthesised from per-paper reviews that literature-reviewer
  already produced.
related:
  - ../modulation_in_rl/modulation_in_rl_lit_review.md
  - ../FiLM/film_in_rl_survey.md
  - ../FiLM/film_modulation_granularity_synthesis.md
  - ../Hypernetwork/hypernetwork_lit_review.md
  - ../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md
  - ../behavior_analysis/behavior_analysis_lit_review.md
  - ../../critiques/nmn_input_site_grid_optimisation_dynamics.md
---

# How FiLM and hypernetwork papers look inside the network

## 1. Question, and the short answer

**The question.** We built a small side-network — a "neuromodulator" — that reads the
agent's senses every step and emits, for up to four places inside the main policy
network, a per-unit multiplier (a **gain**, written γ) and a per-unit shift (an
**offset**, written β). Thirty-two training runs later, a separate behavioural
analysis found **no difference between agents that have the side-network and agents
that do not**, in the state-dependent behaviours it looked for. So the question has
moved: if the side-network changes nothing you can see from the outside, does it
change anything *inside*? And before we invent a way to answer that, what have other
people already done?

This document answers the second half. It reads the project's existing library of
paper reviews and asks, of every paper whose own mechanism is the gain-and-offset
operator or a network that generates another network's weights: **did that paper
measure anything inside its own trained network?** Where the answer is yes, the
technique is written down in enough detail to be copied, and rated for whether it
would run on what we have on disk.

**The short answer, in three parts.**

*First, the negative result, and it is the most useful finding here.* Of **31
empirical papers** in the library whose mechanism is gain-and-offset modulation or a
weight-generating side-network, **16 report task performance and ablations only** and
never look inside. Ten do look inside in some serious way, and five do something
partial. Narrowing to the sharpest version of the question — how many inspect the
**hidden unit activity of the network that was modulated** — the answer is **four out
of thirty-one**, and among the ten reinforcement-learning papers reviewed in full in
the `modulation_in_rl` folder, it is **zero**. That corpus's papers inspect
observations, actions, value estimates and modulator weights; not one opens the
policy's hidden layers. Publishing a careful internal characterisation of a modulated
policy would therefore be a small but genuinely unoccupied slot.

*Second, the methods we want do exist in this library — just not in the machine-learning
wing of it.* The computational-neuroscience papers held under
`neuromodulatory_algorithms/` invert the ratio completely: seven of them run exactly
the kind of analysis we need (freeze the modulator and replay; correlate the
modulator's moment-to-moment output against an external variable; measure how similar
two internal states are; lesion one modulation channel at a time; project the hidden
state into a low-dimensional space and colour it by internal state). Two are close
enough in architecture to copy almost directly, and both were reviewed for this
project already. So the correct framing for our own work is not "nobody has done
this" — it is "the biology-facing half of the field does this routinely and the
robot-policy half does not, and we sit in between."

*Third, and this is the practically decisive point.* The in-house optimisation review
that prompted this survey argued that the *time-averaged* gain and offset are
book-keeping quantities the modulated layer can absorb into its own weights, so the
average tells us nothing, and that the numbers we currently log cannot tell a
genuinely context-varying modulator apart from one that has collapsed into a fixed
re-tuning. It proposed two remedies: split the variability of the gain into a
"different per unit" part and a "changes with context" part, and re-run the saved
policy with the gain and offset pinned at their averages to see whether behaviour
changes. **Both have direct published precedent, and the second one has the strongest
precedent in the entire corpus** — the original FiLM paper ran exactly that
experiment, and reports that replacing the gain with its training-set average costs
65.4 percentage points of accuracy while doing the same to the offset costs 1.0. That
is a template, a headline number to compare against, and a reason to believe the
experiment discriminates.

**What this document is not.** It does not extract from any PDF that has not already
been reviewed; where the library's coverage is thin, §8 says so and names the paper to
fetch rather than pretending. It does not design an experiment or write code — §7
ranks candidates and hands off.

---

## 2. How to read this, and what counted as "looking inside"

### 2.1 The scoped corpus

A paper is **in scope** if the mechanism it proposes or deploys is one of:

- **feature-wise affine modulation** — a conditioning signal produces a per-unit or
  per-channel multiplier γ and shift β applied to activations (FiLM proper;
  conditional batch / instance / layer normalisation; adaptive layer normalisation,
  "adaLN"; and constrained or factorised variants of the same operator); or
- a **hypernetwork** — one network generates another network's weights.

Papers held in the same folders purely as background (attention, batch-norm theory,
mixture-of-experts, module networks, the uncertainty survey) are **out of scope**;
they are listed in the manifest with a reason so the count is auditable.

Two further tiers appear separately and are labelled as such:

- **Tier 2 — biological gain modulation** (`neuromodulatory_algorithms/`): mechanisms
  that scale a neuron's responsiveness or a weight matrix rather than emitting a
  per-unit γ and β. The project's own [FiLM ↔ neuromodulation integration
  synthesis](../FiLM/film_neuromod_integration_synthesis.md) already maps most of
  these onto FiLM variants row by row, which is why their methods transfer even
  though their mechanism differs.
- **Tier 3 — recurrent-network analysis with no modulator at all**: papers whose
  conditioning is a plain concatenated input, included only because the *analysis*
  technique is the thing we want.

### 2.2 What counted

"Looking inside" means the paper reports a measurement of a quantity that lives
*inside* the trained network, as opposed to (a) task performance, (b) an ablation
scored purely by task performance, or (c) a training curve. Three categories were
counted separately, because they answer different questions and have very different
costs for us:

| Category | What is measured | Example |
|---|---|---|
| **P — modulation parameters** | the distribution, geometry or time-course of γ and β themselves | γ histograms; a 2-D map of γ/β coloured by question type |
| **H — hidden representations** | the activity of the units that were modulated, or of the network downstream | a classifier trained to read a variable off the hidden state |
| **M — modulator internals / optimisation geometry** | the modulator's own weights, gradients, or the curvature it induces | the product of the modulator's weight-matrix spectral norms |

An **intervention** (freeze, clamp, lesion, shuffle) is counted under whichever
category its *readout* belongs to, and is flagged separately in §5 because
interventions license causal claims that measurements do not.

---

## 3. The manifest and the count

### 3.1 In-scope papers, and whether they look inside

Sources are the per-paper reviews under `FiLM/reviews/`, the master reviews
[`modulation_in_rl_lit_review.md`](../modulation_in_rl/modulation_in_rl_lit_review.md)
and [`hypernetwork_lit_review.md`](../Hypernetwork/hypernetwork_lit_review.md), and
[`TD_lit_review_C_hypernet_rl.md`](../TD/TD_lit_review_C_hypernet_rl.md).

| # | Paper | Mechanism | Looks inside? | What it measures |
|---|---|---|---|---|
| 1 | Perez et al. 2018 — FiLM | per-channel γ/β | **Yes (P, H)** | γ/β histograms; 2-D embedding of γ/β by question type; activation maps; mean-substitution intervention |
| 2 | Dumoulin et al. 2017 — Conditional Instance Norm | per-style γ/β table | **Yes (P)** | linear interpolation in γ/β space, scored by style loss |
| 3 | Huang & Belongie 2017 — AdaIN | γ/β = style-image statistics | Partial | reframes instance-norm as style normalisation via an input-manipulation experiment; no internal measurement of the modulated net |
| 4 | Birnbaum et al. 2019 — Temporal FiLM | per-block γ/β from an LSTM | **Yes (P)** | visualises the learned γ/β; they cluster by speaker gender |
| 5 | Ha et al. 2016 — HyperNetworks | generated weights | **Yes (P, H)** | per-timestep weight-change trace; histograms of hidden-cell activations vs LayerNorm |
| 6 | Krueger et al. 2017 — Bayesian Hypernets | noise-conditioned weights | No | task metrics + uncertainty scores |
| 7 | Galanti & Wolf 2020 — modularity of hypernets | — | n/a (theory) | no experiments of this kind |
| 8 | Turkoglu et al. 2022 — FiLM-Ensemble | per-member γ/β | Partial | member-disagreement and pairwise-divergence metrics — output space, not internal |
| 9 | Takeda et al. 2021 — multi-task feature modulation | per-task γ/β | No | output interpolation as the conditioning vector is swept |
| 10 | Abdollahzadeh et al. 2021 — Kernel Modulation | per-weight modulation | Partial | cross-task transference histogram — a gradient-level, not representational, measure |
| 11 | Jang et al. 2022 — BC-Z | FiLM at every ResNet block | No | success rates; conditioner-type comparison |
| 12 | Moon et al. 2023 — contrastive achievements | FiLM fuses action into state | **Yes (H)** | linear probe on the frozen encoder predicting the next sub-goal, before and after the new objective |
| 13 | Nikulin et al. 2023 — anti-exploration RND | FiLM in the prior network | **Yes (H, M)** | gradient-field visualisation; a diagnostic that strips the critic to test whether the actor can minimise the bonus |
| 14 | Wisnu et al. 2025 — STSM-FiLM | scalar-conditioned FiLM | No | speech-quality metrics + FiLM ablation |
| 15 | Yan & Guo 2025 — CaFiLM | batch-statistic-conditioned FiLM | No | accuracy ablations only |
| 16 | Gorishniy et al. 2025 — TabM | shared weights + per-member affine | Partial | implicit-submodel decomposition — output space |
| 17 | Jiang et al. 2021 — dynamic predictive coding | hypernet over a basis of transition matrices | **Yes (H, P)** | reverse-correlation receptive fields of first-level units; timescale analysis of the modulatory latent; generative sampling from it |
| 18 | Borycki et al. 2022 — Bayesian MAML hypernet | weight generation | No | few-shot accuracy |
| 19 | Schöpf et al. 2022 — Hypernetwork-PPO | weight generation | No | continual-RL return curves; reports training instability in prose |
| 20 | Beck et al. 2023 — hypernets in meta-RL | weight generation + FiLM baseline | No | initialisation study and parameter-count controls, scored by return |
| 21 | Rezaei-Shoshtari et al. 2023 — HyperZero | weight generation | No | zero-shot transfer scores |
| 22 | Chauhan et al. 2023 — hypernetwork survey | — | n/a (survey) | explicitly names interpretability of generated weights as an **open problem** |
| 23 | Sarafian et al. 2021 — RL building blocks with hypernets | hypernet + a FiLM-shaped inner layer | Partial (M) | cosine similarity between the critic's action-gradient and an empirically estimated reference gradient |
| 24 | Yuan 2024 — diffusion-policy components | FiLM vs concatenation | No | success rates across 8 tasks |
| 25 | Reuss et al. 2025 — FLOWER | one adaLN weight set shared across 18 layers | No | benchmark scores; a design-decision ablation |
| 26 | Yoon et al. 2026 — PAPL | per-unit γ/β on actor **and** critic | **Yes (H, adjacent)** | 2-D embedding of actions and proprioceptive observations coloured by gait phase; value-estimate histograms split by phase |
| 27 | Zhu et al. 2025 — EquAct | symmetry-constrained γ/β | No | manipulation success rates |
| 28 | Li et al. 2025 — CogVLA | γ reused as a token-keeping score | No | success rates; stage ablations |
| 29 | NVIDIA 2025 — GR00T N1 | adaLN action head | No | benchmark scores |
| 30 | Marquis & Farhood 2026 — hypernet-conditioned flight control | FiLM vs low-rank weight edits, under PPO | **Yes (M)** | Lipschitz sensitivity bound = product of the modulator's weight-matrix spectral norms, across six configurations |
| 31 | Kang et al. 2026 — SplitAdapter | two-source factorised FiLM | **Yes (H)** | probe-regression "predictability gap" between each context latent and each prediction target; 2-D embedding of the joint latent by payload mass |
| 32 | Guo et al. 2026 — GEAR | FiLM + symmetry + multi-head critic | No | task-success ablation grid |
| 33 | Guo et al. 2026 — MoE-ACT | FiLM on decoder queries | No | success rates |

**Excluded from the count as background-only** (held in `FiLM/sources/` but not
FiLM-mechanism papers): Andreas et al. 2016 and Hu et al. 2017 (module networks),
Shazeer et al. 2017 (mixture-of-experts), Vaswani et al. 2017 (attention), Santurkar
et al. 2018 (batch-norm theory), Kendall & Gal 2017 and Gawlikowski et al. 2023
(uncertainty). Rows 7 and 22 are in scope but non-empirical.

### 3.2 The count

| | Count | Share of the 31 empirical in-scope papers |
|---|---|---|
| Look inside in a substantive way | **10** | 32 % |
| Partial — an internal-ish measure, but in output or gradient space | **5** | 16 % |
| **Task metrics and ablations only** | **16** | **52 %** |
| Inspect the **hidden units of the modulated network** | **4** (Perez, Ha, Jiang, Moon) | 13 % |
| Of the ten FiLM-in-RL papers reviewed in full: inspect the modulated policy's hidden units | **0** | 0 % |

Two further framings sharpen this.

**By recency.** Every paper that looks inside the *modulated network's* hidden layers
is from 2016–2021 (Ha, Perez, Jiang) plus one 2023 outlier (Moon). The 2024–2026
robot-policy wave — ten papers, all with the same operator — inspects nothing
internal at all beyond Marquis's modulator-weight norm and SplitAdapter's
conditioner-latent probe. This is a field-level drift toward benchmark-only reporting,
and the `modulation_in_rl` review reaches the same conclusion by a different route:
its evidence audit found that **exactly four genuine mechanism ablations of
feature-wise conditioning exist in the whole reinforcement-learning corpus**, and that
"the evidence base for 'FiLM helps in RL' is much thinner than the frequency of FiLM
in RL papers suggests" ([`film_in_rl_survey.md` §8](../FiLM/film_in_rl_survey.md)).

**By wing of the field.** Restrict to Tier 2 — the biological gain-modulation papers —
and the ratio inverts. Seven of them run an internal analysis: Vecoven et al. 2020,
Ben-Iwhiwhu et al. 2022, Costacurta et al. 2024, Tsuda et al. 2021, AlKilany & Goodman
2025, plus Wainstein et al. 2025 and Driscoll et al. 2022 in Tier 3. **The methods
this project needs are already in its own library; they are filed under
neuroscience, not under FiLM.**

### 3.3 One gap the corpus itself already flagged

The FiLM-in-RL survey lists, among the things the corpus does not settle: *"Whether
multiplicative modulation accelerates plasticity loss. The RL plasticity literature
and the RL modulation literature do not cite each other anywhere in this library.
**Nobody has measured dormant-neuron fraction in a FiLM-modulated agent.**"*
([`film_in_rl_survey.md` §10 item 5](../FiLM/film_in_rl_survey.md)). That is a
publishable measurement, it needs no new training, and §4.2 gives the technique.

---

## 4. The technique catalogue

Each entry gives: **what it takes in**, **what it puts out**, **what contrast it
needs**, **what conclusion it licensed in the source paper**, and **whether it runs
here**. "Runs here" is judged against our actual assets: a recurrent PPO policy with a
128-unit gated recurrent trunk on a 10×10 grid, a 27-number observation, **one random
seed per configuration**, 32 runs, 50 saved checkpoints per run, trajectory stores of
300,000 episodes per checkpoint, and per-timestep γ and β that the model already
computes but the evaluation loop throws away (`scripts/eval/eval_rollout.py:317` binds
the modulation output to a discarded variable).

### 4.1 Family P — analyses of the gain and offset themselves

#### P1. Distribution of γ and β, with attention to sign and to exact zeros

- **Source.** Perez et al. 2018 §4.2, as extracted in
  [`FiLM/reviews/perez_2018_film.md`](../FiLM/reviews/perez_2018_film.md).
- **In.** All γ and β values produced over a validation set.
- **Out.** Two histograms per site. Perez reports γ spanning −15 to 19 with **a sharp
  peak at zero** (whole channels switched off) and **36 % of gains negative**; β
  spanning −9 to 16 with **76 % negative**.
- **Contrast needed.** None — but the interpretation depends on what follows the
  modulation. A negative gain in front of a rectifier flips which half of the
  activation distribution survives, which single-signed gates cannot express.
- **Licensed.** "The operator is being used to gate, not merely to rescale." Perez
  reinforces this with a constraint sweep: forcing γ into (0,1), (−1,1) or (0,∞) all
  hurt, so unrestricted sign and magnitude are load-bearing.
- **Runs here?** **Yes, immediately, from checkpoints.** Output is a distribution over
  128 units × timesteps per site, so it is interpretable at one seed. Note that our
  gain is not sign-constrained either, so the negative-γ fraction is directly
  comparable to Perez's 36 %.

#### P2. Low-dimensional embedding of the (γ, β) vector, coloured by a known state label

- **Sources.** Perez et al. 2018 §4.2 (t-SNE of the modulation parameters at the first
  and last modulated block, coloured by question type — clusters by *low-level*
  function early and by *high-level* function late, which the authors read as a
  self-organised hierarchy); Birnbaum et al. 2019 §5.4 (the same idea on audio; the
  learned modulation clusters by **speaker gender**, a variable never supplied as a
  label); Kang et al. 2026 Fig. 3(a) (2-D embedding of the joint context latent shows
  "clearer mass-dependent organisation" than a unified-latent baseline).
- **In.** The per-example or per-timestep concatenated (γ, β) vector.
- **Out.** A scatter plot; qualitative cluster structure.
- **Contrast needed.** A **label the modulator was never given**. This is what makes
  the result non-trivial: Birnbaum's modulator was never told the speaker's gender.
- **Licensed.** "The modulation parameter space has organised itself around a
  semantically meaningful variable" — i.e. the modulation is not noise.
- **Runs here?** **Yes, cheaply.** Our natural colouring labels are injury level,
  satiation, predator presence and distance. **But add a quantitative companion**, and
  this is a real methodological criticism of all three source papers: a 2-D embedding
  is a qualitative artefact, and at one seed a qualitative artefact is easy to
  over-read. The scalar companion is P3 or H2 — how well can a simple regressor
  recover the label from the (γ, β) vector.

#### P3. Regressing the modulation output on a known state variable

- **Sources.** No FiLM paper does exactly this. The nearest is Kang et al. 2026's
  **predictability gap** (H2 below) and AlKilany & Goodman 2025's cross-correlation
  (P4). The in-house optimisation review proposes it directly as the
  explained-variance test of whether the modulator reads the body.
- **In.** Per-timestep γ (unit-averaged, and per unit) plus the state variables of
  interest.
- **Out.** A coefficient of determination per unit and per site — a **distribution over
  128 units**.
- **Contrast needed.** A shuffled-in-time control (regress γ against a time-permuted
  copy of the state variable), which gives the null distribution.
- **Licensed.** "The modulation tracks *this specific* variable, above the level
  explicable by autocorrelation alone."
- **Runs here?** **Yes, and it is one of the two highest-value items.** Flag it as a
  partly-original contribution: the corpus supplies the ingredient in the conditioner
  latent (Kang) but nobody regresses the *emitted gain* on the *conditioning variable*
  in a FiLM policy.

#### P4. Cross-correlation of the modulation output against an external signal, at lags

- **Source.** AlKilany & Goodman 2025 §2.3, Figs. 6–7, reviewed in
  [`neuromodulatory_algorithms/reviews/alkilany_goodman_2025_snn_dynamic_sensory.md`](../neuromodulatory_algorithms/reviews/alkilany_goodman_2025_snn_dynamic_sensory.md).
  Their modulator emits neuron parameters over time; they show firing rate is
  **suppressed during noise peaks and boosted during noise dips**, that this
  generalises to noise frequencies never trained on, and that against natural
  background noise the correlation is **r = −0.60 with a peak at a lag of −47 ms**.
  They also plot the trajectories of each modulated parameter and report which ones
  show the clearest time-locking (threshold and reset; less so the time constants).
- **In.** The modulator's per-timestep output, and an external time series.
- **Out.** A correlation-versus-lag curve, per modulated parameter — a **distribution
  over lags**, not a single number.
- **Contrast needed.** An unmodulated control for the downstream effect; a
  phase-scrambled or circularly-shifted covariate for the null.
- **Licensed.** "The modulator implements an identifiable, anticipatory strategy" —
  the negative lag is what makes it anticipatory rather than reactive.
- **Runs here?** **Yes, and this is the closest published template to the question we
  actually care about.** Our analogue: cross-correlate site-wise mean γ against
  interoceptive nociception onset, satiation, and predator distance, at lags of ±50
  steps. A negative-lag peak would be evidence of anticipatory gain change; a
  zero-lag-only peak would say the modulator is a passive relay of its own input.
  AlKilany's paper is additionally the only one in the library that sweeps a
  *grouping* parameter — one modulator output driving G neurons identically — which
  the project's [granularity synthesis §7.2](../FiLM/film_modulation_granularity_synthesis.md)
  already identifies as our exact design.

#### P5. Linear structure of the modulation space — interpolation and vector algebra

- **Sources.** Dumoulin et al. 2017 §3.5 (blending two styles' γ/β linearly produces
  smooth intermediate outputs, and the loss against each endpoint varies
  *monotonically* with the mixing weight); Perez et al. 2018 §4.5 (compute γ/β for an
  unseen attribute combination as `A + B − C` of related ones; gains 3.2 points
  overall and 9.2 points on the applicable questions, with **no training**);
  Huang & Belongie 2017 (the same blending, but with statistics-derived γ/β).
- **In.** Two or more sets of (γ, β) plus the frozen main network.
- **Out.** Whether outputs interpolate smoothly and whether synthesised parameters do
  something sensible.
- **Contrast needed.** A held-out combination never seen in training.
- **Licensed.** "The modulation space is quasi-disentangled and linearly composable."
- **Runs here?** **Partially, and worth flagging as awkward.** Our conditioner is not a
  discrete set of labelled conditions; it is a continuous, endogenous observation
  stream. The honest port is: take the mean γ conditioned on injured-and-hungry,
  injured-and-fed, uninjured-and-hungry, synthesise the fourth by vector algebra,
  clamp the modulator to it, and check the resulting behaviour against the real
  uninjured-and-fed condition. Needs no new training. **Medium value, medium effort** —
  it is a genuine test of compositional structure, but only interesting if P3 has
  already shown the modulator tracks those variables at all.

#### P6. Effective rank of the gain matrix

- **Sources.** The measure is Kumar et al. 2021's **stable rank** and Lyle et al.
  2022's **feature rank**, both held under `continual_learning/reviews/`; the
  application to γ specifically is a project proposal recorded in
  [`neuromodulatory_algorithms_predictions_synthesis.md` §6.1](../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md),
  where it is offered as the analogue of Vecoven's per-dimension recruitment analysis.
- **In.** The matrix of γ values, timesteps × units, per site.
- **Out.** The number of singular directions carrying most of the spectral mass. The
  **whole singular-value spectrum** is the richer object and is distribution-valued.
- **Contrast needed.** The same quantity at initialisation, and at the unmodulated
  control's matched layer where applicable.
- **Licensed.** "The modulator is effectively a k-knob controller." Rank 1 would mean
  a single global gain dressed up as 128; high rank would mean genuinely per-unit
  control.
- **Runs here?** **Yes, cheaply.** Report the spectrum, not just the scalar summary —
  see §6.

### 4.2 Family H — analyses of the modulated network's hidden representations

#### H1. Per-unit activity fraction and the dormancy score

- **Sources.** Sokar et al. 2023 (the dormancy score: a unit's mean absolute
  activation divided by the layer's mean, with a unit called τ-dormant at or below a
  threshold; τ = 0.025 in the paper's analysis, loosened to 0.1 for benchmarking; they
  show the dormant count **rises steadily through training**, that dormant units
  **rarely reactivate** — the overlap between the current and historical dormant sets
  climbs — and that pruning the always-dormant units **does not hurt performance**,
  confirming they are inert); Lyle et al. 2024 (adds a second pathology, units that
  have gone effectively linear rather than dead); Abbas et al. 2023. All under
  `continual_learning/`.
- **In.** Pre-activation and activation values for every unit over a rollout.
- **Out.** A per-unit score — **a distribution over 128 units per site**.
- **Contrast needed.** **The unmodulated control agent's own layer at the same
  checkpoint.** A converged rectified policy carries dead units anyway; the question
  is whether modulation *adds* them.
- **Licensed.** "Modulation is (or is not) silencing units, and the silencing is (or is
  not) permanent."
- **Runs here?** **Yes — highest value per unit of effort in the whole list, and the
  literature says nobody has done it in a modulated agent.** Two refinements from the
  in-house review matter: report the *distribution* of each unit's on-fraction rather
  than the layer-mean on-fraction, because a layer-mean of 0.3 is equally consistent
  with "every unit is on 30 % of the time" (which is the intended gating behaviour) and
  "30 % of units are always on, 70 % never" (which is the pathology); and report
  Sokar's score alongside it, because a unit can be positive yet negligible.

#### H2. Probing — train a simple readout from the hidden state to a known variable

- **Sources.** Kang et al. 2026 (the **predictability gap**: fit a probe regressor
  from each of two context latents to each of two prediction targets, and report the
  difference in explained variance between the matched pair and the crossed pair —
  larger gap means each latent has specialised; the gradient-reversal regulariser
  widens it. The `modulation_in_rl` review calls this "the methodologically strongest
  part of the paper" and recommends copying it); Moon et al. 2023 §3.2 (a **linear
  probe** on the frozen encoder predicting the next sub-goal in a 22-way task: 44.9 %
  top-1 for plain PPO with poor confidence, rising to 73.6 % after their contrastive
  objective — used to argue the objective changes *representation quality*, not just
  the score); Simmons-Edler et al. 2025 (ridge-regression decoding of past and future
  displacement from a recurrent hidden state, one decoder per time offset, trained on
  the first 75 % of each episode and tested on the last 25 % specifically to defeat
  leakage from temporal autocorrelation — fully worked, with the closed-form solution
  derived, in [`behavior_analysis_lit_review.md` §5.3.5](../behavior_analysis/behavior_analysis_lit_review.md)).
- **In.** Hidden states plus ground-truth labels.
- **Out.** Accuracy or explained variance — a single scalar unless structured. **Make
  it structured**: per time offset (Simmons-Edler), per matched/crossed pair (Kang),
  or per unit.
- **Contrast needed.** This is where probing is easy to get wrong. Three controls, in
  increasing order of strictness: a chance baseline; a temporally shuffled control; and
  a matched-difficulty *control task* (see §8 — the canonical reference for this is not
  in our library).
- **Licensed.** "The information is present and linearly available." **Not** "the
  network uses it." That second claim needs an intervention (family C).
- **Runs here?** **Yes, and it is the second of the two highest-value items** — but only
  once the evaluation loop stops discarding the hidden state. The behaviour-analysis
  review has already costed this: extending the evaluation recording with the hidden
  state, the value estimate and the policy entropy unlocks five analyses at once, and
  storage is not a barrier (~256 KB per episode uncompressed at 128 units × 500 steps).
  That is a code change and belongs to `senior-developer`.

  **The version specific to our question**: probe the *modulator's* 16-unit hidden
  state for interoceptive nociception and satiation, and probe the *policy trunk's*
  128-unit state for the same variables, and report the gap. If the trunk already
  carries the body state as well as the modulator does, the modulator is adding no new
  information and the null behavioural result has an explanation. This is Kang's
  predictability gap, applied to our architecture.

#### H3. Representation-similarity measures — CKA

- **Source.** Ben-Iwhiwhu et al. 2022 §5.2, reviewed in
  [`neuromodulatory_algorithms/reviews/beniwhiwhu_2022_context_meta_rl.md`](../neuromodulatory_algorithms/reviews/beniwhiwhu_2022_context_meta_rl.md).
  They compute **centred kernel alignment** — a similarity score between two sets of
  hidden-layer activations — across tasks, for a modulated and an unmodulated network.
  On an easy task both networks already produce task-distinct representations; on the
  hard task **only the modulated network does**, the unmodulated one's representations
  staying too uniform. They pair it with heatmaps of the *modulator's own output*
  across tasks, showing strongly non-uniform off-diagonal structure, to argue the
  modulator is the source of that diversity. Crucially they also run
  **parameter-matched controls** — a wider and a deeper unmodulated network — which do
  not close the gap, so the effect is architectural rather than a capacity artefact.
- **In.** Two activation matrices (examples × units).
- **Out.** One similarity value per pair — but across C conditions you get a **C×C
  matrix**, whose off-diagonal entries are a distribution.
- **Contrast needed.** The unmodulated control, and at least two conditions to compare
  across. For us the conditions are internal states (injured / uninjured, hungry /
  sated, predator near / far) rather than tasks.
- **Licensed.** "Modulation makes the network's internal state more (or less)
  distinguishable across contexts." This is close to a direct operationalisation of the
  question this document exists to answer.
- **Runs here?** **Yes, once hidden states are recorded.** It is the single most
  on-target technique in the corpus for the question "does the modulator do anything
  representationally", and one of the two source papers is already in this project's
  reference folder for exactly this reason. **Watch one caveat**: Ben-Iwhiwhu compares
  representations *after few-shot adaptation across tasks*; we have one task and no
  adaptation step, so our comparison is across *within-episode internal states*, which
  is a weaker contrast. Say so when citing.

#### H4. Per-unit tuning — regress each unit's activity on the state

- **Sources.** Jiang et al. 2021 (reverse-correlation on the first-level units of a
  hypernetwork-driven predictive-coding model, recovering both orientation-selective
  and direction-selective receptive fields resembling primary visual cortex);
  Simmons-Edler et al. 2025 Appendix D (one generalised linear model per unit, mapping
  a coarse-grained position code to that unit's normalised activity, then clustering
  the coefficient vectors — about 100 of 512 units are well fit by position alone, and
  both the count and the coefficient magnitude **increase with distance from the
  origin**, which they read as a distance-accumulation circuit); Driscoll et al. 2022
  (the **variance matrix**: variance of each unit's activity across conditions within
  each task period, normalised, then hierarchically clustered — the block structure is
  robust to architecture and is absent in untrained networks).
- **In.** Per-unit activity plus state variables.
- **Out.** A fit quality and a coefficient vector **per unit** — richly
  distribution-valued.
- **Contrast needed.** The untrained network (Driscoll's control), and the unmodulated
  agent.
- **Licensed.** "Specific units encode specific variables, and the population is
  organised into functional groups."
- **Runs here?** **Yes, and the state space is small enough to make it easy** — 100
  cells, plus injury, satiation and predator distance. The behaviour-analysis review
  flags that our 10×10 arena is too small for the position-binning scheme Simmons-Edler
  uses (they coarse-grain 9,216 cells into 196 bins; we have 100 cells and 128 units,
  so a per-cell design matrix is already near-saturated), so use the physiological
  variables as regressors rather than position. **The modulation-specific version**: fit
  the tuning twice, once with the modulator live and once with it frozen at its mean,
  and ask whether tuning *sharpens*. That is the machine-learning analogue of the
  classic gain-versus-selectivity distinction that Ferguson & Cardin 2020 lays out —
  a multiplicative gain change should rescale a tuning curve without moving its peak,
  whereas an additive offset in front of a rectifier sharpens it (the "iceberg
  effect"). Our γ and β do both at once, and this analysis separates them.

#### H5. Low-dimensional projection of the trajectory, coloured by modulator level

- **Sources.** Tsuda et al. 2021 (principal-component projection of the recurrent
  state under different neuromodulator settings; trajectories occupy **non-overlapping
  "hypertubes"**, three components capture 80–92 % of the variance, intermediate
  settings trace a smooth curved arc between tubes, and the curvature of that arc —
  their "angle of departure" — correlates at R = 0.60 with the network's sensitivity);
  Costacurta et al. 2024 §5 (the modulated network reuses a shared ring-shaped attractor
  for angle memory across tasks, with the task distinction living in a *later*
  principal component); Yoon et al. 2026 §V-A (a 2-D embedding of actions and
  proprioceptive observations over 80 seconds, showing the two gait phases in separated
  clusters with the transition in between).
- **In.** Hidden-state trajectories.
- **Out.** A projection; qualitative geometry, plus scalar summaries such as
  between-versus-within cluster separation.
- **Contrast needed.** Two or more distinct modulator levels or internal states.
- **Licensed.** "The modulator moves the network's activity into distinct regions of
  state space", which is the strongest available operationalisation of "the modulator
  reconfigures the computation."
- **Runs here?** **Yes.** The predictions synthesis already names this as translating
  cleanly. Add a quantitative separation score so it is not purely visual.

#### H6. Activation-distribution comparison against a normalisation baseline

- **Source.** Ha et al. 2016 §4.5, per
  [`Learning_Rate_lit_review_D_gated_conditional_arch.md`](../Learning_Rate/Learning_Rate_lit_review_D_gated_conditional_arch.md):
  histograms of hidden-cell activations show a plain recurrent cell saturating heavily,
  layer normalisation reducing saturation, and the hypernetwork *increasing* saturation
  relative to layer normalisation while reaching equal or better loss. The reading is
  that dynamic rescaling does something **qualitatively different** from statistical
  normalisation even at matched performance.
- **Runs here?** **Yes, trivially, and it is directly relevant.** Our encoder applies
  layer normalisation *before* the modulation, so we are in exactly the regime Ha
  compares: the modulated activation distribution versus the normalised one. Cheap
  companion to H1.

### 4.3 Family C — interventions

Interventions are separated because they license a different claim. A probe shows
information is *present*; an intervention shows it is *used*.

#### C1. Replace the modulation with its average — the freeze-at-mean replay

- **Source, and it is the strongest precedent in this document.** Perez et al. 2018
  §4.3: at test time, **replacing γ with its training-set mean drops accuracy by 65.4
  percentage points; doing the same to β drops it by 1.0.** The asymmetry is the
  result — the gain carries essentially all the conditioning weight and the offset
  almost none.
- **Secondary source.** Vecoven et al. 2020's freeze/unfreeze protocol, reviewed in
  [`vecoven_2020_neuromod_dnn.md`](../neuromodulatory_algorithms/reviews/vecoven_2020_neuromod_dnn.md):
  they freeze the modulatory signal at its initial value (the agent shows its
  exploration strategy), unfreeze it (the agent solves the task), re-freeze at a
  task-adapted value (performance is largely preserved), then **switch which target is
  rewarding while the signal stays frozen** — and the agent goes to the wrong target,
  showing it cannot adapt without updating the modulator. Unfreezing fixes it. This is
  a four-condition design, not a single ablation, and it is the cleanest causal
  argument for a modulator in the library.
- **In.** A saved checkpoint plus a rollout.
- **Out.** Task performance under substitution. **Scalar per run** — see the fix below.
- **Contrast needed.** The unmodified policy on the same episodes.
- **Licensed.** "The moment-to-moment variation in the modulation is (or is not) doing
  work." As the in-house review notes, this is definitive whatever the variance split
  says, because a contextual variation the policy does not use is not modulation.
- **Runs here?** **Yes, no new training, and it should be run first.** Two design notes.
  (i) **Run the gain and the offset separately**, as Perez did — the asymmetry was his
  most informative result and our two offsets were still drifting at the end of
  training, so the same asymmetry is a live possibility. (ii) **Pair the episodes.**
  Replay the same 300,000 stored episode initial conditions with and without the
  substitution and take the per-episode paired difference in survival steps. That
  converts a single number per run into a paired distribution over episodes, which is
  what makes it interpretable at one seed (§6).
- **A caution the in-house review makes precise.** Because the time-averaged gain and
  offset can be absorbed into the modulated layer's own weights, freezing at the mean is
  mathematically the same as deleting the modulator and folding the constants in. So a
  null result here says "the modulator's *variation* is unused"; it does **not** say
  "the modulator changed nothing during training". Those are different claims and the
  write-up must not blur them.

#### C2. Clamp or lesion one channel at a time

- **Sources.** Costacurta et al. 2024 §4 (ablate each modulatory gain dimension in turn
  and find each controls a specific computational sub-step — interval-setting, ramping,
  terminating); Driscoll et al. 2022 Fig. 5 (zero the output of one unit cluster
  identified by the variance matrix; **only tasks depending on that cluster's dynamics
  degrade**, and the cause is geometric — the lesion destroys the relevant attractor,
  not merely some capacity); Wainstein et al. 2025 (lesion the inhibitory cluster paired
  with the *dominant* population versus the *competing* one, and only the first slows
  switching, which identifies the circuit mechanism); Zou et al. 2020 §3.3 (ablate each
  of two neuromodulators; one carries the choice signal, the other the switch detector,
  and both are necessary).
- **Runs here?** **Yes, no new training.** We have four modulation sites; clamp each to
  identity in turn and measure paired per-episode survival differences. This is the
  project's own per-site clamp proposal and the predictions synthesis already names
  Costacurta's design as its direct analogue. **High value, low effort.**

#### C3. Route the same information a different way

- **Sources.** Yuan 2024 Table 5 (modulation versus concatenation of the same signal —
  up to 62 percentage points of success rate); Yoon et al. 2026 (the phase signal reaches
  the network **twice**, as raw input and as the modulation conditioner, so the no-FiLM
  ablation is a pure routing comparison rather than an information ablation); Marquis &
  Farhood 2026 §V-D (the baseline concatenates the fault vector onto the observation
  while the treatment sends it only to the modulator — **in-distribution the two routes
  are near-equivalent and the gap opens only out of distribution**).
- **Licensed.** "Where a signal enters matters, and it matters mainly when the test
  conditions differ from training."
- **Runs here?** **Needs new training** — this is an architecture comparison, not a
  replay. Flag it anyway, because the `modulation_in_rl` review's sharpest criticism of
  our own experimental design is that we compare "modulator versus no modulator" rather
  than "modulator versus the same information delivered another way", and because
  Marquis is the corpus's best explanation for why our in-distribution comparisons came
  back neutral while our one clear win was a continual-learning probe.

#### C4. Freeze the trunk and retrain only the modulator

- **Sources.** Costacurta et al. 2024 §5 (freeze the output network, retrain only the
  modulatory subnetwork on a held-out task; the shared attractor is preserved and
  reused); Driscoll et al. 2022 Fig. 6 (train only the task-rule input weights for a
  held-out task and learn in a fraction of the time, provided the needed dynamical
  motifs were pre-trained); Dumoulin et al. 2017 §3.4 (freeze the backbone and train
  only one new style's γ/β row — roughly eight times faster than training a new network).
- **Runs here?** **Needs new training, but cheaply** — only the modulator heads are
  trainable. **Medium value**: it answers "is the modulator a sufficient interface for
  reconfiguring this policy?", which is a different and more positive question than the
  one that motivated this survey.

### 4.4 Family M — the modulator's own parameters and the geometry it induces

#### M1. Lipschitz sensitivity bound from spectral norms

- **Source.** Marquis & Farhood 2026 §V-C, per
  [`modulation_in_rl_lit_review.md` §12.4](../modulation_in_rl/modulation_in_rl_lit_review.md).
  Estimate the modulator's Lipschitz bound as the **product of the spectral norms of
  its weight matrices** (valid because their nonlinearity is 1-Lipschitz), and find it
  tracks control performance across six configurations — from 28.38 at the worst to 3.73
  at the best. The recommended remedy is spectral normalisation, and the paper's
  conclusion repeats it as future work.
- **Runs here?** **Yes, from checkpoint parameters alone — no rollout needed.** Compute
  it across all 50 checkpoints per run. The `modulation_in_rl` review calls this "the
  single most actionable finding in the corpus" for our earlier crash, because
  ballooning gain variance is precisely what a Lipschitz constant measures. **Cheapest
  item in this document. A scalar per checkpoint, but a 50-point trajectory per run**,
  which is what makes it usable at one seed.

#### M2. Weight-change trace over time

- **Source.** Ha et al. 2016 §4.4: plot the norm of the change in the generated weights
  between consecutive timesteps during generation. The weights are **nearly constant
  through the body of a word and jump sharply at word boundaries, brackets and stroke
  boundaries** — the network has learned an implicit segmenter.
- **Runs here?** **Yes, directly, and it is a good figure.** Plot the step-to-step change
  in γ over an episode and mark the events — a bite of food, a wound, a predator
  appearing. If the modulation is event-locked, this shows it in one panel; the
  quantitative version is P4.

#### M3. Curvature and the fast/slow decomposition induced by a gain

- **Source.** Rodriguez-Garcia et al. 2026, Appendix B, reviewed in
  [`rodriguezgarcia_2026_ne_stability_gap.md`](../neuromodulatory_algorithms/reviews/rodriguezgarcia_2026_ne_stability_gap.md).
  They prove that multiplying weights by a gain before the forward pass **divides the
  local curvature by the square of that gain**, and separately that a transient gain
  boost is equivalent to a fast/slow weight decomposition. Their ablation is a model of
  how to do this honestly: removing the gain dynamics appears to help, but only because
  the network then barely learns the new task at all, so they explicitly rule out
  "stability via stagnation".
- **Runs here?** **In principle yes** (curvature can be probed on a checkpoint without
  forming the full matrix), but **medium effort, medium value**. Its main use is
  theoretical: it is the published statement closest to the in-house review's claim that
  the time-averaged gain is absorbable, and it should be cited when that claim is
  written up.

#### M4. Gradient-quality and gradient-field measures

- **Sources.** Sarafian et al. 2021 (cosine similarity between the critic's action
  gradient and an empirically estimated reference gradient, over the training
  trajectory); Nikulin et al. 2023 (gradient-field visualisation of the bonus landscape
  under four different conditioning operators, plus a diagnostic that strips the critic
  and asks whether the actor can drive the bonus down at all — concatenation fails,
  FiLM succeeds).
- **Runs here?** Sarafian's method **does not port**: it needs a continuous action space
  and a state-action value function, and estimates the reference gradient with 15 extra
  rollouts per state. We have discrete actions and a state-value critic. Nikulin's
  "can the actor minimise it" diagnostic is specific to an exploration-bonus setting.
  **Mark both out of reach**, but note the family: an analysis of *where the gradient
  lives*. The in-house review's finding that the modulator's share of the squared
  gradient norm rises to 40–75 % late in training is in this family, and no paper in the
  corpus reports the equivalent — which makes it a small original observation.

### 4.5 Methods in the corpus that we should mark out of reach

| Method | Source | Why it does not run here |
|---|---|---|
| Ensemble-diversity metrics (member disagreement, pairwise divergence) | Turkoglu et al. 2022 | needs many members; **one seed per configuration**. Checkpoints are not independent members |
| Cross-task transference histogram | Abdollahzadeh et al. 2021 | needs a task distribution and hundreds of source-target pairs; we have one task |
| Action-gradient cosine similarity | Sarafian et al. 2021 | continuous actions + state-action value function + 15 perturbation rollouts per state |
| Reverse-correlation receptive fields | Jiang et al. 2021 | defined for a continuous natural-stimulus stream; our 27-number observation has no analogous stimulus ensemble. The per-unit regression (H4) is the right substitute |
| Pupillometry, functional imaging, human ratings, neural recordings | Wainstein et al. 2025; Birnbaum et al. 2019 (listener ratings) | no analogue in a simulated agent |
| Real-hardware transfer as a validity check | Yoon et al. 2026; Kang et al. 2026 | no robot |
| Delay-discounting / impulsivity assays | Doya 2002 branch | the gridworld does not natively dissociate immediate from delayed reward |
| Formal disentanglement metrics | — | not present in the corpus at all, and our conditioner is not factorised into known generative factors. Perez's vector algebra (P5) is the corpus's only disentanglement-flavoured test |

---

## 5. Cross-paper comparison table

Every row is a technique that at least one in-scope or Tier-2 paper actually ran. "Cat."
is the category from §2.2. "n=1 safe" marks methods whose output is a distribution over
units, timesteps, episodes or lags rather than a single scalar per run (§6).

| Technique | Cat. | Papers that ran it | Input | Output | Required contrast | Conclusion it licensed | Runs on our checkpoints? | n=1 safe |
|---|---|---|---|---|---|---|---|---|
| γ/β histograms, sign + zero structure | P | Perez 2018 | γ, β over a rollout | two histograms per site | none (interpretation needs the downstream nonlinearity) | "the operator gates, it does not merely rescale" | **Yes, now** | ✅ per unit |
| 2-D embedding of (γ,β) by state label | P | Perez 2018; Birnbaum 2019; Kang 2026 | (γ,β) vectors | scatter + clusters | a label never given to the modulator | "modulation space is semantically organised" | **Yes, now** | ⚠ qualitative — pair with a score |
| Regress γ on a state variable | P | *(no FiLM paper; nearest Kang 2026)* | γ + state | R² per unit | time-shuffled control | "modulation tracks *this* variable" | **Yes, now** | ✅ per unit |
| Cross-correlate γ against a covariate at lags | P | AlKilany & Goodman 2025 | γ + covariate | correlation-vs-lag curve | phase-scrambled covariate | "anticipatory, identifiable strategy" | **Yes, now** | ✅ per lag/unit |
| Interpolation / vector algebra in γ/β space | P | Dumoulin 2017; Perez 2018; Huang & Belongie 2017 | ≥2 γ/β sets, frozen net | smoothness + held-out synthesis | an unseen combination | "quasi-disentangled, linearly composable" | Yes, awkward fit | ⚠ |
| Effective rank / spectrum of the γ matrix | P | *(measure from Kumar 2021 / Lyle 2022)* | γ matrix | singular-value spectrum | init + control | "the modulator is a k-knob controller" | **Yes, now** | ✅ spectrum |
| Per-unit activity fraction + dormancy score | H | Sokar 2023; Lyle 2024; Abbas 2023 | pre-activations | per-unit score | **unmodulated control, same checkpoint** | "modulation does/doesn't silence units" | **Yes, now** | ✅ per unit |
| Probe / decode a variable from hidden state | H | Kang 2026; Moon 2023; Simmons-Edler 2025 | hidden states + labels | R² or accuracy | chance + shuffle + control task | "the information is linearly available" | Needs hidden states recorded | ✅ if per offset/pair |
| CKA representation similarity | H | Ben-Iwhiwhu 2022 | two activation matrices | similarity matrix | unmodulated control × ≥2 contexts | "modulation makes states more context-distinct" | Needs hidden states recorded | ✅ off-diagonals |
| Per-unit tuning / encoding model | H | Jiang 2021; Simmons-Edler 2025; Driscoll 2022 | unit activity + state | fit + coefficients per unit | untrained net; unmodulated control | "specific units encode specific variables" | Needs hidden states recorded | ✅ per unit |
| Low-D trajectory projection by modulator level | H | Tsuda 2021; Costacurta 2024; Yoon 2026 | hidden trajectories | projection + separation | ≥2 modulator levels / states | "the modulator moves activity to distinct regions" | Needs hidden states recorded | ⚠ pair with a score |
| Activation-distribution vs normalisation baseline | H | Ha 2016 | activations | histograms | LayerNorm-only baseline | "dynamic rescaling ≠ statistical normalisation" | **Yes, now** | ✅ |
| **Freeze the modulation at its mean** | C | **Perez 2018 (−65.4 vs −1.0 pts)**; Vecoven 2020 | checkpoint + rollout | performance delta | unmodified policy, same episodes | "the variation is / is not used" | **Yes, now** | ✅ if episode-paired |
| Clamp / lesion one channel at a time | C | Costacurta 2024; Driscoll 2022; Wainstein 2025; Zou 2020 | checkpoint + rollout | per-site deltas | all sites live | "this site carries this function" | **Yes, now** | ✅ if episode-paired |
| Route the same signal differently | C | Yuan 2024; Yoon 2026; Marquis 2026 | two architectures | performance gap | matched information, two routes | "where the signal enters matters, mainly out of distribution" | **Needs new training** | ⚠ |
| Freeze trunk, retrain modulator only | C | Costacurta 2024; Driscoll 2022; Dumoulin 2017 | frozen trunk | learning speed | full retraining | "the modulator is a sufficient interface" | Needs small new training | ⚠ |
| Lipschitz bound from spectral norms | M | Marquis 2026 | modulator weights | scalar per checkpoint | across configurations / over training | "sensitivity tracks performance; spectral norm is the remedy" | **Yes, now (params only)** | ✅ as a 50-point trajectory |
| Weight-change trace over time | M | Ha 2016 | generated params over time | trace with event markers | none | "the modulator is an implicit segmenter" | **Yes, now** | ✅ per timestep |
| Curvature / fast-slow decomposition | M | Rodriguez-Garcia 2026 | checkpoint | curvature ratio | gain on vs off | "a gain flattens the loss surface" | Yes, medium effort | ⚠ |
| Parameter-matched capacity control | — | Ben-Iwhiwhu 2022; AlKilany 2025; Beck 2023 | two architectures | performance | wider / deeper unmodulated net | "the gain is architectural, not parameter count" | **Needs new training** | ⚠ |

---

## 6. The one-seed problem, and which methods survive it

With **one seed per configuration**, any method whose output is a single number per run
has no error bar. This is not a detail — it decides which of the methods above can
carry a claim.

**Methods that produce a distribution, and are therefore interpretable at one seed:**

1. **Anything computed per unit** — the on-fraction and dormancy score across 128 units;
   per-unit explained variance when regressing γ or activity on a state variable;
   per-unit γ averages; per-unit tuning coefficients; the singular-value spectrum of the
   γ matrix. You get 128 samples per site per checkpoint and can state a median and an
   interquartile range within the run.
2. **Anything computed per timestep or per lag** — the correlation-versus-lag curve; the
   step-to-step weight-change trace; the γ trajectory itself. The *shape* of a curve is
   informative even when its height is not comparable across runs.
3. **Anything computed per episode and paired** — the freeze-at-mean and clamp
   interventions, run on the *same* stored episode initial conditions with and without
   the manipulation. The paired per-episode difference over 300,000 episodes has a
   genuine sampling distribution.
4. **Anything pairwise** — the off-diagonal entries of a CKA similarity matrix over
   contexts; the matched-versus-crossed gap in a probing design. Kang's predictability
   gap is a *difference of two probe scores*, which is exactly the right shape: the
   common nuisance variation cancels.
5. **Anything traced across the 50 checkpoints of a run** — the Lipschitz bound, the
   effective rank, the dormant fraction. A monotone 50-point trajectory within one run
   is evidence in a way a single endpoint is not.

**Methods that give one scalar per run, and therefore cannot carry a claim as stated:**
a single probe accuracy; a single effective-rank number at the final checkpoint; a
single Lipschitz value; mean survival steps under an unpaired ablation; any
between-arm comparison of a pooled mean. Each of these is rescued by pairing,
per-unit decomposition, or a checkpoint trajectory — and should be reported that way or
not at all.

**The honest caveat, which must appear in any write-up.** A bootstrap over units,
timesteps or episodes gives a confidence interval on the *estimate within this run*. It
says nothing about **seed-to-seed variability** — whether a different initialisation
would have produced a different modulator. So a per-unit distribution licenses "in this
run, the dormant fraction under modulation is X and under the control is Y"; it does
**not** license "modulation causes dormancy". Claims of the second kind need either
seeds or a within-run intervention (family C), which is a further reason to weight the
freeze and clamp experiments above the pure measurements. The in-house optimisation
review makes the same point about its own arm-level attributions, and the discipline
should be uniform.

---

## 7. Ranked by value per unit of effort

**Tier 1 — run these first. No new training, no code beyond a replay script, and each
produces a distribution.**

| Rank | Technique | Why first |
|---|---|---|
| 1 | **Freeze the gain at its mean, then the offset, separately** (C1) | The definitive test of whether the *variation* is used, with the corpus's strongest precedent and a headline number to compare against (Perez: −65.4 vs −1.0 points). Episode-paired, so interpretable at one seed. If the answer is "no change", every other measurement becomes a question about a component the policy does not use — which is itself the answer to the question this document was commissioned to inform |
| 2 | **Per-unit on-fraction + dormancy score, against the unmodulated control** (H1) | Cheap, distribution-valued, and the FiLM-in-RL survey states outright that **nobody has measured this in a modulated agent**. Addresses the one concern the in-house review classed as a genuine worry |
| 3 | **Split the variability of γ and β into a per-unit part and a per-timestep part** (P3/P6 machinery) | The in-house review's central proposal. Distinguishes a context-varying modulator from a collapsed constant re-parameterisation, which pooled statistics cannot. Report per unit and per site |
| 4 | **Per-site clamp to identity** (C2) | Four one-line interventions, episode-paired, and it answers "which site does anything" — a question the current design cannot address. Costacurta's per-dimension ablation is the template |
| 5 | **Lipschitz bound across all 50 checkpoints** (M1) | Cheapest item here — parameters only, no rollout — and gives a 50-point trajectory per run. Published remedy attached if it grows |
| 6 | **γ/β histograms with sign and zero structure** (P1) | Ten minutes' work, directly comparable to Perez's published numbers, and a required figure in any write-up |

**Tier 2 — one enabling code change (record the hidden state, the value estimate and
the policy entropy during evaluation), then high value.** The behaviour-analysis review
has already scoped this change and costed the storage; it belongs to
`senior-developer`.

| Rank | Technique | Why |
|---|---|---|
| 7 | **CKA between hidden representations across internal states, modulated vs control** (H3) | The most on-target technique in the corpus for "does the modulator change the representation", with a matched precedent (Ben-Iwhiwhu) and a parameter-matched control design to copy |
| 8 | **Probe the modulator's hidden state and the policy trunk's hidden state for the same body variables, and report the gap** (H2, Kang's predictability gap) | Directly tests whether the modulator carries information the trunk lacks. If not, the null behavioural result is explained |
| 9 | **Cross-correlate γ against nociception and satiation at lags of ±50 steps** (P4) | AlKilany's design, ported. Distinguishes anticipatory gain change from passive relay. Distribution over lags |
| 10 | **Per-unit tuning, fit with the modulator live and again with it frozen** (H4) | Separates the multiplicative from the additive arm in the way the gain-modulation literature defines them. More work than the above; correspondingly more interesting |
| 11 | **Low-dimensional projection of the trunk trajectory coloured by internal state** (H5) | Good figure, cheap once hidden states exist, but pair it with a separation score |

**Tier 3 — needs new training. Worth planning, not worth blocking on.**

| Rank | Technique | Why later |
|---|---|---|
| 12 | **Route the body signal through concatenation instead of modulation, matched otherwise** (C3) | The corpus's sharpest criticism of our experimental design, and the comparison three papers run. But it is a new arm |
| 13 | **Parameter-matched unmodulated control (wider / deeper)** | Rules out "the modulator is just extra parameters". Ben-Iwhiwhu and AlKilany both run it; we have not |
| 14 | **Freeze the trunk, retrain only the modulator on a shifted environment** (C4) | Tests the modulator as a re-configuration interface. A different, more positive question |

**Deliberately not recommended.** Ensemble-diversity metrics, cross-task transference,
action-gradient cosine similarity, reverse-correlation receptive fields, and formal
disentanglement metrics — see §4.5 for why each is out of reach or a poor fit.

---

## 8. What the corpus does not cover, and what to fetch

Several of the techniques recommended above are used in the library **only at second
hand** — a reviewed paper applies the method and cites its origin, but the primary
methods paper is not held. That is fine for running the analysis and **not** fine for
writing it up, because the methodological caveats live in the primaries. For
`literature-reviewer`, in priority order:

1. **Kornblith, Norouzi, Lee & Hinton 2019, *Similarity of Neural Network
   Representations Revisited* (ICML).** The origin of the CKA measure that
   Ben-Iwhiwhu 2022 uses and that ranks 7th above. We hold the application, not the
   method. **Highest priority.**
2. **Hewitt & Liang 2019, *Designing and Interpreting Probes with Control Tasks*
   (EMNLP).** The control-task discipline that turns a probe accuracy into a
   *selectivity* — the difference between the real task and a matched random-label
   task. At one seed this matters more than usual, because it converts a bare number
   into a difference. Directly governs recommendations 8 and 10. **Highest priority.**
3. **Belinkov 2022, *Probing Classifiers: Promises, Shortcomings, and Advances*
   (Computational Linguistics).** The standard statement that a probe's success shows
   information is *present*, not *used*. This is precisely the gap our freeze-at-mean
   experiment fills, so it is the citation that positions the pairing of H2 with C1 as a
   deliberate design rather than two unrelated analyses. **High priority.**
4. **Mante, Sussillo, Shenoy & Newsome 2013 (Nature), and Sussillo & Barak 2013,
   *Opening the Black Box* (Neural Computation).** Cited across at least four reviews in
   this library (Tsuda, Driscoll, Wainstein, Jiang) as the origin of both
   targeted dimensionality reduction and the fixed-point toolkit, and held by none of
   them. Targeted dimensionality reduction — regress population activity on task
   variables and project onto the resulting axes — is arguably a better fit for our
   question than a generic low-dimensional projection, because it *builds the axes from
   the variables we care about*. **High priority.**
5. **Alain & Bengio 2017, *Understanding Intermediate Layers Using Linear Classifier
   Probes*.** The origin of layer-wise linear probing. Moon 2023 uses it without
   attribution in our review. **Medium priority.**
6. **Kriegeskorte, Mur & Bandettini 2008, *Representational Similarity Analysis*.** The
   task brief asked specifically about RSA. **The library contains no RSA paper at all**
   — the closest is Ben-Iwhiwhu's CKA. If RSA proper is wanted (comparing
   dissimilarity *matrices* rather than raw representations, which is the natural way to
   compare a 16-unit modulator state against a 128-unit trunk state without aligning
   dimensions), this is the primary. **Medium priority, but note the gap is real.**
7. **Stroud, Porter, Hennequin & Vogels 2018, *Motor Primitives in Space and Time via
   Targeted Gain Modulation* (Nature Neuroscience).** Cited by both Tsuda 2021 and
   Costacurta 2024 as the closest neuroscience precedent for analysing targeted gain
   modulation; not held. **Medium priority.**
8. **A participation-ratio / neural-dimensionality primary** (e.g. Litwin-Kumar,
   Harris, Axel, Sompolinsky & Abbott 2017, or Gao et al. 2017). The brief asked about
   participation ratio; the library's dimensionality measures are all
   reinforcement-learning-flavoured rank measures (Kumar's stable rank, Lyle's feature
   rank), which are related but not the same object and were introduced for a different
   purpose. **Low-to-medium priority** — the rank measures we hold will do the job, but
   the framing differs.

Two further coverage notes:

- **`Fiber_bundle/` holds three PDFs and no review document.** Nothing in this survey
  draws on it. If the geometric framing is going to appear in a paper, that folder
  needs a review pass before it can be cited.
- **The six papers named in the in-house optimisation review's own missing-reference
  list** (on scale symmetry, conserved quantities under gradient flow, effective
  learning rate, dying rectifiers, and gradient clipping) are already routed to
  `literature-reviewer` and are **not** duplicated here. They support the *gauge*
  argument; this document's list supports the *measurement* argument.

---

## 9. Relationship to the rest of the library

- [`../../critiques/nmn_input_site_grid_optimisation_dynamics.md`](../../critiques/nmn_input_site_grid_optimisation_dynamics.md)
  — the memo that prompted this survey. It supplies the *argument* (why the average
  gain and offset are not identifiable, and why the pooled statistics cannot separate
  contextual modulation from constant re-tuning). This document supplies the *method
  precedents* for the two remedies it proposes, and finds strong precedent for both.
- [`../modulation_in_rl/modulation_in_rl_lit_review.md`](../modulation_in_rl/modulation_in_rl_lit_review.md)
  — the source for all ten 2024–2026 reinforcement-learning modulation papers. Its §4
  evidence ledger answers *design* questions; this document answers *measurement*
  questions over the same corpus.
- [`../FiLM/film_in_rl_survey.md`](../FiLM/film_in_rl_survey.md) §8 and §10 — the
  evidence-quality audit and the "what the corpus does not settle" list, one item of
  which (nobody has measured dormancy in a modulated agent) is recommendation 2 above.
- [`../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md`](../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md)
  §4 and §6 — `professor-neuromodulation`'s table of measures already tried by the
  neuroscience corpus, with a translatability column. **That table and this document
  overlap deliberately and disagree nowhere**; it is organised by *prediction*, this one
  by *technique and feasibility*. Read that one to know what the biology predicts; read
  this one to know how to measure it.
- [`../behavior_analysis/behavior_analysis_lit_review.md`](../behavior_analysis/behavior_analysis_lit_review.md)
  §5.3.5 and §5.4.1 — the fully worked treatment of hidden-state decoding and per-unit
  encoding models for a recurrent policy, including the derivation, the leakage-proof
  train/test protocol, the portability table, and the storage arithmetic for the one
  enabling recording change. **Where that document covers a method, defer to it rather
  than to the summaries in §4.2 here.**
- `neural_representation_analysis_lit_review.md` *(in this folder, being written
  concurrently by `literature-reviewer`)* — full per-paper reviews of the three PDFs in
  this folder's `sources/`: Ben-Iwhiwhu et al. 2022, Vecoven et al. 2020, and
  Simmons-Edler et al. 2025. Those three are the three most method-relevant papers in
  the whole library, and §4 above summarises them **from their existing reviews under
  `neuromodulatory_algorithms/reviews/` and `behavior_analysis/`**. When the full
  reviews land, defer to them for detail and treat this document as the index over
  techniques. *(Link to be made bidirectional once that file exists.)*

---

## 10. Hand-offs

- **`senior-developer`** — one enabling change gates every Tier-2 item: extend the
  evaluation recording to keep the hidden state, the value estimate and the modulation
  output, all of which are already computed and discarded at
  `scripts/eval/eval_rollout.py:317`. The behaviour-analysis review has scoped and
  costed it. The Tier-1 items need only a checkpoint-replay script and no change to
  `src/`.
- **`experiment-designer`** — recommendations 1 and 4 (freeze-at-mean, per-site clamp)
  are interventions on saved checkpoints, and both need an episode-paired design over
  the stored initial conditions to be interpretable at one seed. Recommendation 12
  (concatenation-versus-modulation routing) is a new arm and needs a design.
- **`literature-reviewer`** — the eight primaries in §8, with items 1–4 marked high
  priority. Suggest they live in this folder's `sources/` rather than a new topic, since
  they are methods papers for this survey.
- **`plan-reviewer`** — when the analysis plan is drafted, the specific thing to check
  is §6: that no headline claim rests on a single scalar per run, and that the
  distinction in §4.3 between "the modulator's *variation* is unused" and "the modulator
  changed nothing" is preserved in the wording.
