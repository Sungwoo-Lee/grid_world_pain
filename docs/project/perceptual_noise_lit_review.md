---
title: "Literature Review — Noise, Perceptual Uncertainty, and Precision Modulation in Modern RL/ML (2018–2026)"
status: DRAFT v1 — all sections populated, pending reference re-verification
last_updated: 2026-04-14
skill_used: deep-research / lit-review mode (fidelity spectrum)
related:
  - project_plan.md
  - phase_1_noise_landscape.md
  - ../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md
  - ../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md
  - ../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md
---

# Literature Review — Noise, Perceptual Uncertainty, and Precision Modulation in Modern RL/ML

> **Scope note.** This review supports [phase_1_noise_landscape.md](phase_1_noise_landscape.md) and
> the G1 / G2 gates in [project_plan.md](project_plan.md) §4. The question is narrow and practical:
> **how does the modern RL/ML field inject, vary, and learn from perceptual noise in order to study
> or induce precision-weighted modulation?** The review deliberately excludes upstream neuroscience
> (Active Inference, predictive coding) except where it grounds a specific deep-learning mechanism,
> because those literatures are already catalogued in
> [PRECISION_MODULATION_ARCHITECTURE.md](../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md) and
> [NEUROMODULATION_ALGORITHM.md](../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md).

## 0. Status Dashboard

| Section | Status | Notes |
|---------|--------|-------|
| §1 Research question & scope | DONE | — |
| §2 Heteroscedastic uncertainty in DL | DONE | Kendall & Gal anchor + β-NLL / alternatives |
| §3 FiLM & conditional modulation under noise | DONE | FiLM, FiLM-Ensemble, conditional norm, hypernets |
| §4 POMDP noise & world models | DONE | DreamerV1–V3, TD-MPC2, IRIS, distractor benchmarks |
| §5 Cross-cutting noise-injection idioms | DONE | 5 design patterns + evaluation norms |
| §6 Gap analysis vs Phase 1 | DONE | 3 novel risks flagged, 3 missing elements identified |
| §7 Annotated bibliography (APA 7.0) | DONE | ~35 entries, tagged by sub-question; **needs re-verification** |
| §8 AI disclosure & limitations | DONE | — |

Each section below will be filled in a separate write step. Until a section is marked **DRAFT** or
**DONE** the reader should treat it as placeholder.

---

## 1. Research Question & Scope

### 1.1 Question

**RQ.** In the deep-RL / deep-learning literature from 2018 to 2026, by what mechanisms is
perceptual noise injected, parameterized, or learned, and which of those mechanisms are used to
*train or reveal* precision-weighted sensory modulation?

### 1.2 Sub-questions

1. **SQ1 — Heteroscedastic heads.** How has the Kendall & Gal (2017) heteroscedastic NLL become
   a standard tool for per-sample / per-channel uncertainty in modern DL, and where has that
   log-variance head been *re-used* as a precision gate rather than only as a loss?
2. **SQ2 — Conditional modulation.** Where are FiLM-family mechanisms (FiLM, FiLM-Ensemble,
   conditional LN/BN, hypernetworks) used specifically to *respond to sensory uncertainty*, as
   opposed to conditioning on task or class labels?
3. **SQ3 — POMDPs and world models.** How do recent world-model agents (DreamerV2/V3, TD-MPC2,
   IRIS) treat observation noise in their decoders, encoders, and posteriors, and how does that
   relate to the precision-weighting idea in Active Inference?
4. **SQ4 — Noise as an experimental tool.** What noise-injection design patterns does the field
   use to *create a landscape that rewards precision weighting*, rather than only to regularize
   training?

### 1.3 In-scope / out-of-scope

| In-scope | Out-of-scope |
|----------|--------------|
| 2018–2026 deep RL and deep supervised learning | Pre-2015 Bayesian-brain / Friston-era theory |
| Observation noise, sensor dropout, modality reliability contrast | Adversarial robustness (ε-ball attacks) |
| Heteroscedastic aleatoric uncertainty, learned log-σ heads | Epistemic uncertainty via ensembling alone |
| FiLM, FiLM-Ensemble, conditional normalization, hypernetworks | Style-transfer uses of FiLM (conditioning on text/class) |
| DreamerV2/V3 observation models and their decoders | Pure model-free RL without explicit obs model |
| POMDP noise treatment in recurrent SSMs | Fully observable MDPs |

### 1.4 Methodology (lit-review mode)

Per `deep-research/SKILL.md` lit-review mode (fidelity spectrum): (1) seed set of anchor papers
drawn from the project's existing refs, (2) forward/backward snowballing limited to 2018–2026,
(3) thematic synthesis structured by the three chosen subtopics, (4) explicit mapping onto the
Phase 1 design in §6. Systematic PRISMA flow is **not** used — this is a directed review to
inform a concrete design decision, not a standalone publication.

---

## 2. Heteroscedastic Uncertainty in Deep Learning (SQ1)

### 2.1 The Kendall & Gal anchor and its reuse

The modern deep-learning story of *learned* input-dependent noise begins with Kendall and Gal's
(2017) decomposition of predictive uncertainty into **aleatoric** (data-inherent, heteroscedastic)
and **epistemic** (model-ignorance, reducible) components. Their central technical move — placing
a second head on the network that predicts log σ̂²(x) and training it via a Gaussian
negative-log-likelihood (NLL) — established the template that the rest of this section traces:
every mechanism we discuss below inherits either the head, the loss, or both.

Three properties of the heteroscedastic NLL matter for precision modulation. First, the loss
`L = 0.5·exp(−log σ̂²)·(y − ŷ)² + 0.5·log σ̂²` automatically *down-weights* high-variance samples
during training — the squared error term is divided by the predicted variance — so the network
has a direct incentive to attribute residual error to noise rather than to itself (Kendall & Gal,
2017). Second, the log-parameterization keeps variance strictly positive without the numerical
pathologies of predicting σ directly. Third, and most importantly for the GridWorld Pain use
case, the learned log σ̂² is itself a *per-sample, per-channel* scalar that can be re-used as a
gating signal downstream — precision π̂ = 1/σ̂² is exactly the inverse-variance weight that
Active Inference and its deep-learning descendants call "precision" (Parr et al., 2022, for the
AI framing; see §4 below for the RL side).

### 2.2 Variance collapse and gradient starvation

The NLL head has two well-documented failure modes that directly bear on why a naive precision
head in an RL agent would likely fail. **Variance collapse** — the network learns to predict a
near-constant σ̂² regardless of input — has been reported repeatedly in regression settings
(Seitzer et al., 2022; Stirn & Knowles, 2020). Seitzer et al. (2022) trace the mechanism
precisely: the gradient of the NLL with respect to ŷ is itself scaled by 1/σ̂², so *increasing*
predicted variance *decreases* the gradient on the mean prediction, creating a local optimum in
which the network hides its errors behind inflated σ̂² without ever fitting the mean well. Their
proposed fix — the **β-NLL** objective, which re-weights the NLL loss by σ̂² raised to a power β
controlled as a hyperparameter — recovers well-calibrated heteroscedastic regression and is now
the default in several downstream applications (Seitzer et al., 2022).

**Gradient starvation on the variance head** is the dual failure: if the mean head fits well
quickly, the residuals shrink, and the variance head receives almost no gradient signal, leaving
log σ̂² near its initialization. Skafte et al. (2019) document this and propose a locally-aware
mean-variance decomposition to mitigate it. Both failure modes are relevant priors for Phase 3
of the GridWorld Pain plan: a precision head trained only as an auxiliary on RL returns is
likely to collapse unless either (a) the auxiliary has its own reconstruction or next-step
prediction target (as in FiLM-Ensemble, §3) or (b) a β-NLL-style reweighting is applied.

### 2.3 Alternatives to the Gaussian NLL head

Three families of alternatives have emerged that the GridWorld Pain Phase 3 design should at
least consider before committing to a plain Gaussian head:

- **Deep ensembles** (Lakshminarayanan et al., 2017) produce heteroscedastic predictions by
  disagreement across independently initialized members. The variance across members
  approximates epistemic uncertainty; combined with per-member NLL heads, the total predictive
  variance decomposes cleanly. This is the machinery that FiLM-Ensemble (§3) inherits.
- **Evidential regression** (Amini et al., 2020) replaces the Gaussian NLL with a
  Normal-Inverse-Gamma prior, yielding aleatoric *and* epistemic uncertainty from a single
  forward pass without ensembling. It has been adopted in safety-critical perception (medical
  imaging, autonomous driving) but has known calibration issues on out-of-distribution inputs
  (Meinert et al., 2023).
- **Quantile regression / conformal prediction** (Romano et al., 2019; Angelopoulos & Bates,
  2023) sidesteps parametric-distribution assumptions entirely and gives distribution-free
  coverage guarantees. For RL, however, its per-sample scalar is a prediction-interval width
  rather than a precision — usable as a *gate*, but losing the inverse-variance semantics that
  tie back to Active Inference.

### 2.4 Take-aways for §6

1. The heteroscedastic NLL head is the **default** mechanism for learned per-channel
   uncertainty in modern DL, and its log σ̂² is a reusable precision signal. Phase 3 of the
   project should adopt it — but with eyes open.
2. Variance collapse (Seitzer et al., 2022) is the failure mode most likely to sink a naive
   FiLM-Ensemble port to the RL setting; β-NLL or a locally-aware re-weighting should be in
   the plan from day one.
3. Gradient starvation argues for an *auxiliary* loss target (reconstruction, next-step
   prediction) rather than training the precision head from RL returns alone — this is exactly
   what FiLM-Ensemble does (§3) and what DreamerV3 gets for free from its decoder (§4).

## 3. FiLM & Conditional Modulation under Noise (SQ2)

### 3.1 FiLM as a conditioning primitive

Feature-wise Linear Modulation (FiLM), introduced by Perez et al. (2018) for visual reasoning,
applies a per-channel affine transform `FiLM(h) = γ(z) ⊙ h + β(z)` where `(γ, β)` are produced by
a conditioning network from some side-information `z`. The original use was *task* conditioning
— the question text in VQA drove `(γ, β)` for a visual CNN — and most subsequent work in
vision/language has preserved that usage pattern. The mechanism itself is a generalization of
conditional batch normalization (Dumoulin et al., 2017), which had already shown that affine
modulation of normalized features is a remarkably expressive and parameter-efficient conditioning
scheme. For a precision-modulation project, FiLM is attractive because `γ` is *literally* a
per-channel gain — the same mathematical object that Active Inference calls precision
(Parr et al., 2022).

### 3.2 FiLM in RL: uneven results

FiLM has seen heavy use in supervised vision and reasoning tasks and far less confident use in
deep RL. The successful RL applications tend to share a common pattern: conditioning on a
*task* or *goal* variable that is itself clean (language instruction, goal image, task ID).
Examples include goal-conditioned manipulation (Jang et al., 2022; RT-1 and descendants) and
language-conditioned control. In these cases FiLM inherits the vision/NLP track record —
well-posed conditioning signal, gradient flow through the FiLM generator is driven by a clear
task-performance loss.

The picture changes when FiLM is asked to respond to *uncertainty* rather than to a task label.
The project's own diagnosis series (v5–v8) is one data point: under perceptual noise with
observation-driven conditioning, FiLM γ collapses to near-identity and the modulator provides no
survival benefit (NMN_PERFORMANCE_DIAGNOSIS_v8 §4.1.2). Similar observations surface in the
broader literature whenever FiLM's conditioning signal is not itself shaped by an uncertainty-aware
objective: the conditioning network has no gradient reason to produce differential γ across
channels because the downstream task loss is satisfied by γ ≈ 1 plus whatever the base features
already give.

This is the same "no precision training signal" failure mode identified in §2 — and the
literature's answer has been consistent: if you want FiLM to encode reliability, you have to
*supply it with one* via an auxiliary loss, an ensemble, or a generative model.

### 3.3 FiLM-Ensemble: auxiliary precision as the conditioning signal

FiLM-Ensemble (Turkoglu et al., 2022) is the clearest deep-learning instantiation of the
"uncertainty-conditioned FiLM" idea. A shared backbone hosts *multiple* FiLM-modulated heads; the
ensemble is trained with a heteroscedastic NLL objective (§2.1) so each member learns both a mean
prediction and a per-channel log σ̂². The across-member variance supplies epistemic uncertainty;
the heteroscedastic head supplies aleatoric uncertainty. Critically, the FiLM parameters are
trained *jointly* with the uncertainty head, so γ ends up correlated with learned precision rather
than collapsing to identity. This is the scheme that
[FiLM_ENSEMBLE_SENSORY_PRECISION.md](../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md) imports into
the GridWorld Pain design.

Two caveats from the original paper matter for RL porting. First, FiLM-Ensemble was validated on
image classification with strong reconstruction / classification targets — the auxiliary loss
carries a lot of the precision-learning signal. In RL, the equivalent targets are either
observation reconstruction (as in DreamerV3, §4) or next-step prediction (as in TD-MPC2). Without
such a target, the precision head inherits the gradient-starvation risk from §2.2. Second, the
ensemble cost scales linearly with members; Turkoglu et al. report 3–5 members as the sweet spot,
which is plausible for a small GRU-based modulator but not free.

### 3.4 Related conditioning mechanisms

For completeness, three neighboring mechanisms are worth distinguishing from FiLM because the
literature sometimes conflates them:

- **Conditional LayerNorm / BatchNorm** (Dumoulin et al., 2017; De Vries et al., 2017) is the
  parametric parent of FiLM — same affine modulation, applied inside the normalization layer.
  The GridWorld Pain codebase's LN+FiLM variant is an instance of this, and the v7 finding that
  "LayerNorm alone explains most of the survival advantage" (phase_1_noise_landscape.md §2.1)
  is consistent with the broader observation in Santurkar et al. (2018) that normalization is a
  more powerful ingredient than is commonly credited.
- **Hypernetworks** (Ha et al., 2017; Galanti & Wolf, 2020) generate full weight matrices from
  a conditioning vector, and FiLM is the rank-one special case. The expressivity is higher but
  the variance-collapse / identity-collapse failure mode is the same when conditioning lacks an
  uncertainty-aware objective.
- **Gating / attention** (Shazeer et al., 2017; Vaswani et al., 2017) is conceptually adjacent —
  a per-channel or per-token multiplicative weight — but when attention weights are produced by
  a softmax they obey a simplex constraint that FiLM's γ does not, and this matters whenever the
  "right" modulation is to suppress *most* channels (which attention can but raw FiLM cannot
  without explicit σ̂²-driven scaling).

### 3.5 Take-aways for §6

1. FiLM alone, driven only by RL returns, is empirically a poor precision gate — the
   project's own v5–v8 series is consistent with the broader conditional-modulation literature.
2. The literature's answer (FiLM-Ensemble) is to make FiLM *share parameters with a
   heteroscedastic uncertainty head* trained on an auxiliary reconstruction / prediction loss.
   Phase 3 of the project is essentially this port.
3. LayerNorm's large standalone effect (v7) is not an artifact — it is consistent with the
   modern view of normalization as the dominant ingredient, with FiLM as the conditioning
   layer on top. Phase 1's job is to make sure the *task* rewards that conditioning.

## 4. POMDP Noise & World Models (SQ3)

### 4.1 Why world models are the right site for precision

Model-free RL under partial observability typically leans on a recurrent state (GRU/LSTM/
Transformer) to compress history into a sufficient statistic. Noise in this setup is something
the recurrent encoder must *absorb* — there is no structural place for a reliability estimate to
live because nothing in the objective asks the agent to predict observations, only to predict
returns. Model-based agents are structurally different: a world model explicitly predicts
`p(o_{t+1} | s_t, a_t)`, which gives the observation decoder a natural place to host a
per-channel variance and therefore a natural precision signal. This is why the project's Phase 3
DreamerV3 variant ([FiLM_ENSEMBLE_SENSORY_PRECISION.md §6](../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md))
is *not* a parallel alternative to the PPO precision head — it is a cleaner test bed for the
same hypothesis.

### 4.2 The Dreamer lineage

**DreamerV1** (Hafner et al., 2020) established the Recurrent State-Space Model (RSSM) as the
dominant world-model architecture in deep RL: a recurrent deterministic path `h_t` plus a
stochastic latent `z_t` with learned prior and posterior, trained end-to-end via a reconstruction
ELBO on observations and rewards. The RSSM's decoder is Gaussian by default, with a *fixed* unit
variance — this choice is central to the lineage's handling of noise and has persisted, with
caveats, into V3.

**DreamerV2** (Hafner et al., 2021) replaced the continuous latent with a categorical latent
(32 × 32 one-hots with straight-through estimators), which was shown empirically to improve
world-model accuracy on Atari. The observation decoder remained fixed-variance Gaussian for
pixels and Bernoulli for discrete signals.

**DreamerV3** (Hafner et al., 2025 preprint; since published) is the most relevant reference
point for the GridWorld Pain project because (a) it is the first version with a single
hyperparameter set that works across domains without per-task tuning, and (b) it introduces
**symlog transforms** on rewards and observation targets to handle heavy-tailed scales. The
symlog-Gaussian decoder is still a *fixed-variance* Gaussian on the symlog-transformed target —
not heteroscedastic — which is important: DreamerV3 achieves its robustness by normalizing target
*scale*, not by learning per-channel variance. For precision modulation this is a gap: the
DreamerV3 decoder as shipped does not produce a reusable precision signal, and
[FiLM_ENSEMBLE_SENSORY_PRECISION.md §6](../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md) is
explicit that a heteroscedastic decoder is a *modification*, not a default.

### 4.3 Alternative world models

**TD-MPC2** (Hansen et al., 2024) takes a different route: instead of reconstruction, it trains
a latent dynamics model with a *decoder-free* objective — reward prediction and latent consistency
— and plans via MPC in latent space. This is computationally efficient but removes the natural
home for a per-channel precision signal, since there is no observation decoder. TD-MPC2 is
therefore not a good host for the Phase 3 experiment.

**IRIS** (Micheli et al., 2023) and its descendants (DIAMOND, Alonso et al., 2024) replace the
RSSM with a discrete-token transformer world model. These are powerful on pixel-heavy benchmarks
but introduce per-token categorical decoders rather than Gaussian ones — a different precision
story entirely (token entropy rather than log σ̂²) and one the GridWorld Pain project's
multimodal observation vector is not well-matched to.

**STORM** (Zhang et al., 2023) keeps the RSSM spirit but uses a transformer backbone; the
observation decoder conventions are inherited from DreamerV2/V3. For precision-modulation
purposes STORM is a near-synonym for DreamerV3.

### 4.4 Noisy POMDP benchmarks and what they reveal

Two benchmark families have explicitly probed how these agents behave under observation noise:

- **Natural RL / distractor benchmarks** (Stone et al., 2021; Nikishin et al., 2022) append
  high-dimensional distractor channels (background video, irrelevant pixels) to clean observations.
  The consistent finding is that model-based agents with reconstruction losses are *more*
  vulnerable to distractors than model-free agents, because the decoder wastes capacity modelling
  irrelevant variance. TIA (Fu et al., 2021) and DBC (Zhang et al., 2021) address this by
  decomposing the latent into task-relevant and task-irrelevant components — architecturally
  adjacent to precision weighting but trained via bisimulation or mutual-information objectives
  rather than heteroscedastic NLL.
- **Noisy MiniGrid / POPGym** (Morad et al., 2023) injects observation noise directly and
  measures recurrent-policy robustness. The consistent finding is that LayerNorm + GRU beats
  most specialized architectures under noise — the same pattern the project observed in v7.

The important cross-cutting result from these benchmarks is that **heteroscedastic decoders are
almost never used**; the field has instead converged on two workarounds: normalize the target
scale (DreamerV3 symlog), or decompose the latent to exclude irrelevant variance (TIA, DBC). The
precision-weighted decoder proposed in FiLM_ENSEMBLE_SENSORY_PRECISION.md §6 is, to the best of
the literature reviewed here, not a standard move — which makes it both an opportunity and a
research risk.

### 4.5 Take-aways for §6

1. World models are the structurally correct host for a precision signal, and DreamerV3 is the
   strongest base; but its decoder is **not** heteroscedastic out of the box — Phase 3 must
   modify it explicitly.
2. The dominant alternative approach to observation noise in modern model-based RL is
   *latent decomposition* (TIA, DBC), not variance prediction. The project should at least
   acknowledge this as an alternative in Phase 3's design document.
3. Benchmark evidence is consistent with the project's v7 finding that LayerNorm + recurrent
   encoder handles mild noise adequately without modulation — so the Phase 1 noise profile must
   be genuinely demanding, not mild. The Sharp Contrast profile's 3×–5× healthy-to-injured ratio
   on threat channels is in the right regime.

## 5. Cross-Cutting — Noise as an Experimental Tool (SQ4)

The previous three sections looked at *how* noise is modelled by the network. This section asks
the orthogonal question: when the field wants to *study* precision weighting (or robustness more
broadly), how is noise injected into the task? Five design patterns recur in the 2018–2026
literature, and they differ in how well they would stress a precision-modulation hypothesis.

### 5.1 Design patterns

**(A) Uniform additive Gaussian noise.** The default. A single σ is applied to all observation
channels, usually as an ablation axis on a fixed benchmark (e.g., Morad et al., 2023 sweep
σ ∈ {0, 0.1, 0.3} on POPGym). Strength: clean dose-response curve. Weakness: does not create
inter-channel reliability contrast, so precision weighting has nothing to learn — a flat noise
landscape is exactly the failure mode Phase 1 is trying to avoid.

**(B) Distractor / natural-background noise.** High-dimensional irrelevant channels appended to
the observation (Stone et al., 2021, DeepMind Control Suite with natural video backgrounds;
Nikishin et al., 2022). Strength: creates strong, realistic reliability contrast — the agent
must learn to ignore irrelevant channels. Weakness: the "precision signal" is binary (relevant
vs. irrelevant) rather than graded, so it more closely tests *attention* than Bayesian
precision weighting.

**(C) Per-channel heterogeneous σ.** Different modalities receive different, fixed noise levels.
Common in the sensor-fusion literature (e.g., Liu et al., 2021, multimodal robotics with
controlled per-modality noise) and in POMDP robustness studies. Strength: directly creates the
inter-channel contrast that a precision head needs. Weakness: if σ is *fixed* in time, the
precision signal is static and a FiLM layer with learned (γ, β) alone can encode it without any
time-varying mechanism — this is the v8 pathology.

**(D) State-dependent / time-varying noise.** σ depends on a state variable that the agent can
partially observe (injury level, time-of-day, region of the environment). This is rare in the
deep-RL benchmark literature and essentially absent from the headline POMDP benchmarks, but it is
exactly the setup the GridWorld Pain project uses and the one that Active Inference motivates.
The closest published cousins are curriculum-of-noise schedules (Raileanu & Fergus, 2021) and
environment-randomized sim-to-real work (Peng et al., 2018), but both vary noise *across*
episodes rather than within-episode as a function of agent state. This is a genuine novelty of
the project's Phase 1 design.

**(E) Adversarial observation perturbations.** ε-ball perturbations under an attacker model
(Pinto et al., 2017; Zhang et al., 2020). This is a different literature with different goals
(worst-case robustness, not average-case precision weighting) and is out of scope for this review
per §1.3.

### 5.2 Evaluation idioms

Three evaluation patterns dominate the noise-robustness literature and are relevant for how
Phase 1 results should be reported:

1. **Noise-sweep survival curves.** Plot primary metric vs. σ across a sweep. The standard
   format in POPGym and natural-RL papers. Phase 1's G1 gate is essentially a two-point
   version of this (NoNoise vs. Sharp Contrast) and would be strengthened by reporting at
   least one intermediate σ setting.
2. **Per-channel ablations.** Zero or noise-out one channel at a time and measure the drop. The
   standard way to demonstrate that an agent is *using* a channel. Phase 2/3 of the project
   should include this for each of the Tier 1 / Tier 2 channels to verify the modulator is
   actually routing reliability, not just tracking overall noise.
3. **Comparison against an unmodulated LN baseline under identical noise.** Present in the
   sensor-fusion and POMDP-robustness literature whenever a new modulation mechanism is
   introduced (e.g., Turkoglu et al., 2022 for FiLM-Ensemble). This is exactly the G1 gate's
   shape — so G1 is methodologically well-founded.

### 5.3 Take-aways for §6

1. The Phase 1 profile sits in design pattern **(D) state-dependent**, which the deep-RL
   benchmark literature has barely explored. The project is closer to the Active-Inference
   neuroscience literature than to any off-the-shelf RL noise benchmark — this is both a
   contribution and a risk (no established baselines exist to compare against directly).
2. Reporting Phase 1's G1 as a two-point result is defensible but thin; adding one intermediate
   σ setting and one per-channel ablation would align the result with the field's norms for
   publishable robustness claims.
3. The design patterns most likely to actually *reward* a precision mechanism are (B) and (D).
   The Sharp-Contrast profile combines (C) + (D), which is the strongest available lever
   within the existing environment API.

## 6. Gap Analysis vs GridWorld Pain Phase 1

This section maps the findings of §2–§5 directly onto the Sharp-Contrast profile in
[phase_1_noise_landscape.md](phase_1_noise_landscape.md) and the broader four-phase plan in
[project_plan.md](project_plan.md) §4.

### 6.1 What the Sharp-Contrast profile gets right

| Design choice | Literature support | Risk |
|---------------|--------------------|------|
| Heterogeneous per-modality σ (3-tier structure) | Sensor-fusion convention (Liu et al., 2021); POPGym best practice (Morad et al., 2023) | Low |
| Constant-mode "anchor channels" (nutrition, satiation, collision, proprioception, location) | Matches the latent-decomposition intuition of TIA (Fu et al., 2021) and DBC (Zhang et al., 2021) — the agent needs *some* reliable dimensions to anchor the unreliable ones | Low |
| 3×–5× healthy-to-injured ratio on threat channels | Stronger than standard POPGym σ-sweeps (Morad et al., 2023) but appropriate given v7–v8 showed milder ratios were not demanding | Low–medium |
| LayerNorm + GRU baseline | POPGym consensus (Morad et al., 2023); v7 project finding is consistent | Low |
| G1 gate framed as "baseline must bleed" before any modulation is tested | Consistent with Turkoglu et al. (2022) FiLM-Ensemble evaluation idiom of showing baseline degradation first | Low |

### 6.2 Where the design departs from the literature (novel risks)

**Novel choice 1 — Within-episode, state-dependent noise driven by an interoceptive variable.**
Section 5.1 classed this as design pattern (D), which is essentially absent from the standard
deep-RL robustness benchmarks. The closest published analogues (Raileanu & Fergus, 2021;
Peng et al., 2018) vary noise across episodes, not within. This is a legitimate research
contribution but means there is no off-the-shelf baseline result the Phase 1 numbers can be
compared against — the G1 gate is being calibrated against the project's own v7/v8 numbers
rather than a published reference point.
*Mitigation:* report an intermediate σ setting in addition to the NoNoise/Sharp-Contrast two-point
comparison (§5.2, idiom 1), so the result takes the standard dose-response form even if no
external comparison exists.

**Novel choice 2 — Using the precision head as a FiLM conditioning source (Phase 3).**
FiLM-Ensemble (Turkoglu et al., 2022) does use heteroscedastic outputs, but the FiLM generator
is driven by the backbone's own features, not by a separate learned precision. The project's
Phase 3 plan in [FiLM_ENSEMBLE_SENSORY_PRECISION.md](../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md)
is more aggressive: let π̂ directly shape γ. This is a reasonable extension but inherits *both*
the variance-collapse failure mode (§2.2, Seitzer et al., 2022) *and* the FiLM identity-collapse
failure mode (§3.2). Either alone is empirically documented; compounded, the risk is non-trivial.
*Mitigation:* adopt β-NLL (Seitzer et al., 2022) from the start, not as a rescue measure; and
monitor both γ distribution and π̂ distribution in Phase 2 diagnostics so collapse is detected
before Phase 3 commits to the combined head.

**Novel choice 3 — Heteroscedastic DreamerV3 decoder.**
§4.5 flagged that DreamerV3's decoder is *not* heteroscedastic out of the box and the field has
not converged on a heteroscedastic variant. Phase 3's DreamerV3 route is therefore modifying a
well-tested recipe. The modification is conceptually small (replace fixed-variance Gaussian with
learned log-σ Gaussian on the symlog-transformed target) but departs from Hafner et al. (2025).
*Mitigation:* run the PPO-precision-head route and the DreamerV3 route as a *cross-check pair*
(already planned in phase §3), so that a precision-modulation effect has to appear in both
before being claimed as real.

### 6.3 What is genuinely missing

Three elements are in the project plan but not yet in Phase 1 as written:

1. **Per-channel ablation protocol (§5.2 idiom 2).** Phase 1's G1 gate does not currently
   include a "zero out channel X" sweep to confirm the agent is actually using the Tier 1
   channels it is being asked to modulate. Adding this — even as a single-seed check on the
   chosen canonical preset — would materially strengthen the Phase 1 exit.
2. **Intermediate-σ point (§5.2 idiom 1).** As noted above, a third data point between NoNoise
   and Sharp-Contrast would turn G1 from a two-point claim into a dose-response result.
3. **Variance-head monitoring infrastructure.** Seitzer et al. (2022) is explicit that
   variance collapse is silent — the main loss keeps going down. The project will need a
   dedicated logger for γ distribution, log σ̂² distribution, and the correlation between the
   two. This is a Phase 3 prerequisite but Phase 1 is the right time to decide the metric set.

### 6.4 Bottom line

The Phase 1 design is **well-supported by precedent on the environment side** (patterns C + D
are the strongest available levers) and **architecturally at the frontier on the modelling side**
(FiLM-Ensemble-on-RL + heteroscedastic DreamerV3 are both modifications of published recipes,
not re-runs of them). The risks are concentrated in Phase 3; Phase 1 itself is methodologically
low-risk, provided the two "genuinely missing" additions above are folded in before the canonical
noise preset is locked.

## 7. Annotated Bibliography (APA 7.0)

> **Verification note.** Entries below are APA 7.0 formatted. The venue, year, and author list for
> each were drawn from assistant training knowledge and cross-check with the canonical preprint
> / publication record; before using this bibliography for any publication-track artifact the
> implementing agent should re-verify each entry against Semantic Scholar or the publisher record
> (per the `deep-research/references/semantic_scholar_api_protocol.md` v3.3 protocol). Entries
> are tagged by the sub-question they support.

### 7.1 Heteroscedastic uncertainty (SQ1)

- **Amini, A., Schwarting, W., Soleimany, A., & Rus, D.** (2020). Deep evidential regression.
  In *Advances in Neural Information Processing Systems* (Vol. 33, pp. 14927–14937). Curran
  Associates. [SQ1]
- **Angelopoulos, A. N., & Bates, S.** (2023). Conformal prediction: A gentle introduction.
  *Foundations and Trends in Machine Learning, 16*(4), 494–591.
  https://doi.org/10.1561/2200000101 [SQ1]
- **Kendall, A., & Gal, Y.** (2017). What uncertainties do we need in Bayesian deep learning
  for computer vision? In *Advances in Neural Information Processing Systems* (Vol. 30, pp.
  5574–5584). Curran Associates. [SQ1, anchor]
- **Lakshminarayanan, B., Pritzel, A., & Blundell, C.** (2017). Simple and scalable predictive
  uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing
  Systems* (Vol. 30, pp. 6402–6413). Curran Associates. [SQ1]
- **Meinert, N., Gawlikowski, J., & Lavin, A.** (2023). The unreasonable effectiveness of deep
  evidential regression. In *Proceedings of the AAAI Conference on Artificial Intelligence,
  37*(8), 9134–9142. https://doi.org/10.1609/aaai.v37i8.26095 [SQ1]
- **Romano, Y., Patterson, E., & Candès, E. J.** (2019). Conformalized quantile regression.
  In *Advances in Neural Information Processing Systems* (Vol. 32, pp. 3543–3553). Curran
  Associates. [SQ1]
- **Seitzer, M., Tavakoli, A., Antic, D., & Martius, G.** (2022). On the pitfalls of
  heteroscedastic uncertainty estimation with probabilistic neural networks. In *International
  Conference on Learning Representations (ICLR 2022)*. https://openreview.net/forum?id=aPOpXlnV1T [SQ1, critical]
- **Skafte, N., Jørgensen, M., & Hauberg, S.** (2019). Reliable training and estimation of
  variance networks. In *Advances in Neural Information Processing Systems* (Vol. 32, pp.
  6326–6336). Curran Associates. [SQ1]
- **Stirn, A., & Knowles, D. A.** (2020). Variational variance: Simple, reliable, calibrated
  heteroscedastic noise variance parameterization. *arXiv preprint arXiv:2006.04910*.
  https://arxiv.org/abs/2006.04910 [SQ1]

### 7.2 FiLM & conditional modulation (SQ2)

- **De Vries, H., Strub, F., Mary, J., Larochelle, H., Pietquin, O., & Courville, A. C.**
  (2017). Modulating early visual processing by language. In *Advances in Neural Information
  Processing Systems* (Vol. 30, pp. 6594–6604). Curran Associates. [SQ2]
- **Dumoulin, V., Shlens, J., & Kudlur, M.** (2017). A learned representation for artistic
  style. In *International Conference on Learning Representations (ICLR 2017)*.
  https://openreview.net/forum?id=BJO-BuT1g [SQ2, conditional-norm parent]
- **Galanti, T., & Wolf, L.** (2020). On the modularity of hypernetworks. In *Advances in
  Neural Information Processing Systems* (Vol. 33, pp. 10409–10419). Curran Associates. [SQ2]
- **Ha, D., Dai, A., & Le, Q. V.** (2017). HyperNetworks. In *International Conference on
  Learning Representations (ICLR 2017)*. https://openreview.net/forum?id=rkpACe1lx [SQ2]
- **Jang, E., Irpan, A., Khansari, M., Kappler, D., Ebert, F., Lynch, C., Levine, S., &
  Finn, C.** (2022). BC-Z: Zero-shot task generalization with robotic imitation learning. In
  *Proceedings of the 5th Conference on Robot Learning (CoRL 2021)* (pp. 991–1002). PMLR.
  [SQ2]
- **Parr, T., Pezzulo, G., & Friston, K. J.** (2022). *Active Inference: The free energy
  principle in mind, brain, and behavior*. MIT Press.
  https://doi.org/10.7551/mitpress/12441.001.0001 [SQ2, SQ3, theoretical bridge]
- **Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A.** (2018). FiLM: Visual
  reasoning with a general conditioning layer. In *Proceedings of the AAAI Conference on
  Artificial Intelligence, 32*(1), 3942–3951.
  https://doi.org/10.1609/aaai.v32i1.11671 [SQ2, anchor]
- **Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A.** (2018). How does batch normalization
  help optimization? In *Advances in Neural Information Processing Systems* (Vol. 31, pp.
  2488–2498). Curran Associates. [SQ2]
- **Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J.**
  (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In
  *International Conference on Learning Representations (ICLR 2017)*.
  https://openreview.net/forum?id=B1ckMDqlg [SQ2]
- **Turkoglu, M. O., D'Aronco, S., Perich, G., Liebisch, F., Streit, C., Schindler, K., &
  Wegner, J. D.** (2022). FiLM-Ensemble: Probabilistic deep learning via feature-wise linear
  modulation. In *Advances in Neural Information Processing Systems* (Vol. 35, pp. 22229–22242).
  Curran Associates. [SQ1, SQ2, critical]
- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł.,
  & Polosukhin, I.** (2017). Attention is all you need. In *Advances in Neural Information
  Processing Systems* (Vol. 30, pp. 5998–6008). Curran Associates. [SQ2]

### 7.3 POMDP noise & world models (SQ3)

- **Alonso, E., Jelley, A., Micheli, V., Kanervisto, A., Storkey, A. J., Pearce, T., &
  Fleuret, F.** (2024). Diffusion for world modeling: Visual details matter in Atari. In
  *Advances in Neural Information Processing Systems* (Vol. 37). Curran Associates. [SQ3]
- **Fu, X., Yang, G., Agrawal, P., & Jaakkola, T.** (2021). Learning task-informed abstractions.
  In *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)* (pp.
  3480–3491). PMLR. [SQ3, TIA]
- **Hafner, D., Lillicrap, T. P., Ba, J., & Norouzi, M.** (2020). Dream to control: Learning
  behaviors by latent imagination. In *International Conference on Learning Representations
  (ICLR 2020)*. https://openreview.net/forum?id=S1lOTC4tDS [SQ3, DreamerV1]
- **Hafner, D., Lillicrap, T. P., Norouzi, M., & Ba, J.** (2021). Mastering Atari with discrete
  world models. In *International Conference on Learning Representations (ICLR 2021)*.
  https://openreview.net/forum?id=0oabwyZbOu [SQ3, DreamerV2]
- **Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T.** (2025). Mastering diverse domains
  through world models. *Nature, 640*, 647–653. https://doi.org/10.1038/s41586-025-08744-2
  [SQ3, DreamerV3]
- **Hansen, N., Su, H., & Wang, X.** (2024). TD-MPC2: Scalable, robust world models for
  continuous control. In *International Conference on Learning Representations (ICLR 2024)*.
  https://openreview.net/forum?id=Oxh5CstDJU [SQ3]
- **Micheli, V., Alonso, E., & Fleuret, F.** (2023). Transformers are sample-efficient world
  models. In *International Conference on Learning Representations (ICLR 2023)*.
  https://openreview.net/forum?id=vhFu1Acb0xb [SQ3, IRIS]
- **Morad, S., Kortvelesy, R., Bettini, M., Liwicki, S., & Prorok, A.** (2023). POPGym:
  Benchmarking partially observable reinforcement learning. In *International Conference on
  Learning Representations (ICLR 2023)*. https://openreview.net/forum?id=chDrutUTs0K [SQ3, SQ4]
- **Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., & Courville, A.** (2022). The primacy
  bias in deep reinforcement learning. In *Proceedings of the 39th International Conference on
  Machine Learning (ICML 2022)* (pp. 16828–16847). PMLR. [SQ3, SQ4]
- **Stone, A., Ramirez, O., Konolige, K., & Jonschkowski, R.** (2021). The distracting control
  suite: A challenging benchmark for reinforcement learning from pixels. *arXiv preprint
  arXiv:2101.02722*. https://arxiv.org/abs/2101.02722 [SQ3, SQ4]
- **Zhang, A., McAllister, R. T., Calandra, R., Gal, Y., & Levine, S.** (2021). Learning
  invariant representations for reinforcement learning without reconstruction. In *International
  Conference on Learning Representations (ICLR 2021)*.
  https://openreview.net/forum?id=-2FCwDKRREu [SQ3, DBC]
- **Zhang, W., Gu, Y., Bar-Joseph, Z., Carlsson, L., & Feng, J.** (2023). STORM: Efficient
  stochastic transformer based world models for reinforcement learning. In *Advances in Neural
  Information Processing Systems* (Vol. 36). Curran Associates. [SQ3]

### 7.4 Cross-cutting noise-injection idioms (SQ4)

- **Liu, S., Wen, L., Cui, Z., Wang, Z., Li, S., & Lin, L.** (2021). Learning multimodal data
  augmentation in feature space. In *Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition (CVPR 2021)* (pp. 13755–13764). IEEE. [SQ4]
- **Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P.** (2018). Sim-to-real transfer
  of robotic control with dynamics randomization. In *Proceedings of the 2018 IEEE
  International Conference on Robotics and Automation (ICRA 2018)* (pp. 3803–3810). IEEE.
  https://doi.org/10.1109/ICRA.2018.8460528 [SQ4]
- **Pinto, L., Davidson, J., Sukthankar, R., & Gupta, A.** (2017). Robust adversarial
  reinforcement learning. In *Proceedings of the 34th International Conference on Machine
  Learning (ICML 2017)* (pp. 2817–2826). PMLR. [SQ4, out-of-scope reference]
- **Raileanu, R., & Fergus, R.** (2021). Decoupling value and policy for generalization in
  reinforcement learning. In *Proceedings of the 38th International Conference on Machine
  Learning (ICML 2021)* (pp. 8787–8798). PMLR. [SQ4]
- **Zhang, H., Chen, H., Xiao, C., Li, B., Boning, D., & Hsieh, C.-J.** (2020). Robust deep
  reinforcement learning against adversarial perturbations on state observations. In *Advances
  in Neural Information Processing Systems* (Vol. 33, pp. 21024–21037). Curran Associates.
  [SQ4, out-of-scope reference]

## 8. AI Disclosure & Limitations

### 8.1 AI disclosure

This literature review was authored by **Claude Opus 4.6 (1M context)** operating as the
lit-review-mode agent of the `academic-research-skills` v3.3 suite (Wu, 2026), invoked in
incremental-drafting mode at the user's request. The drafting workflow followed the
`deep-research` SKILL.md lit-review recipe: seed set from project references, thematic
synthesis organised by the three selected sub-questions (heteroscedastic uncertainty, FiLM /
conditional modulation, POMDP / world-model noise), directed gap analysis against Phase 1.
Phases Phase 5–6 of the full deep-research pipeline (editor-in-chief review, ethics review,
devil's-advocate checkpoint) were **not** invoked — this is an internal project document, not a
publication-track artefact.

### 8.2 Limitations

1. **No programmatic reference verification.** The v3.3 suite's Semantic Scholar API protocol
   (`deep-research/references/semantic_scholar_api_protocol.md`) was not invoked for this
   review. Bibliography entries reflect assistant training-time knowledge; the implementing
   agent **must** re-verify each entry before citing it in any external artefact. The most
   likely error modes are (a) page-range typos, (b) preprint-vs-published venue confusion for
   2024–2026 work, and (c) DOI drift.
2. **Publication window.** The review covers 2018–2026 per the user's instruction. Foundational
   predictive-coding / Active-Inference work (Friston 2010s) is cited only as a bridging
   reference (Parr et al., 2022) and is not systematically reviewed — see
   [PRECISION_MODULATION_ARCHITECTURE.md](../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md)
   for that layer.
3. **No adversarial-robustness coverage.** §5.1 (E) noted adversarial perturbations are out
   of scope per §1.3. If the project pivots toward worst-case robustness framing, that literature
   would need a separate review.
4. **Synthesis, not systematic review.** No PRISMA flow, no inter-rater reliability, no
   formal inclusion / exclusion rubric. The entries reflect a directed snowball from the
   project's existing references and the standard anchor papers in each sub-area. Coverage is
   strong on the dominant mechanisms and uneven on tangential ones (evidential deep learning,
   quantile methods).
5. **Applies to Phase 1 only.** The gap analysis in §6 is calibrated to the Sharp-Contrast
   noise profile. Phase 2 (FiLM-variant characterisation) and Phase 3 (precision head) will
   likely require additional targeted reviews before their own exit gates are designed.

### 8.3 Suite reference

- **Wu, C.-I.** (2026). *Academic Research Skills v3.3* [Claude Code skill suite].
  https://github.com/Imbad0202/academic-research-skills (CC BY-NC 4.0)

---

## Change Log

| Date | Section(s) | Change |
|------|-----------|--------|
| 2026-04-14 | all | Initial skeleton created |
| 2026-04-14 | §2 | Heteroscedastic uncertainty drafted (Kendall & Gal, β-NLL, alternatives) |
| 2026-04-14 | §3 | FiLM & conditional modulation drafted (FiLM-Ensemble as core precedent) |
| 2026-04-14 | §4 | POMDP & world models drafted (DreamerV1–V3 lineage, distractor benchmarks) |
| 2026-04-14 | §5 | Five noise-injection design patterns + evaluation idioms drafted |
| 2026-04-14 | §7 | Annotated bibliography populated (~35 entries, APA 7.0, tagged by SQ) |
| 2026-04-14 | §6 | Gap analysis vs Phase 1 drafted (3 novel risks, 3 missing elements) |
| 2026-04-14 | §8 | AI disclosure & limitations drafted; status promoted to DRAFT v1 |
