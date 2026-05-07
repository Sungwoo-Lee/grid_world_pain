---
title: GridWorld Pain — Project Plan (Nature MI rewrite)
last_updated: 2026-05-07
---

# GridWorld Pain — Project Plan (Nature MI rewrite)

> Main project context document. Records the goal, target behaviour,
> central research issue, and staged plan. Develop-tree docs under
> [docs/develop/](../develop/) carry technical depth.

## How this rewrite supersedes the prior plan

This document supersedes
[archive/project_plan_2026-04-12.md](archive/project_plan_2026-04-12.md).
The rewrite is authorised by a four-professor consultation chain plus
a cross-professor synthesis (the authorising chain):

- Pain-modeling: [concepts/pain_vs_nociception_construct.md](concepts/pain_vs_nociception_construct.md) — operational distinction nociception/pain, the four-property fingerprint, the C0/C1/C2 controls; verdict on prior plan "partial → overclaimed".
- Bayesian-brain: [concepts/active_inference_hypervigilance.md](concepts/active_inference_hypervigilance.md) — G1 restated as channel-rank non-degeneracy; FP-1 as the technical content of the v8 null; four-property AI fingerprint.
- RL/Bayesian-DL: [directions/architecture_for_pain_computation.md](directions/architecture_for_pain_computation.md) — smallest architecture that escapes FP-1; multiplicative π̂ blend dropped on construct-validity grounds; ~108-run factorial sizing.
- Neuromodulation: [critiques/biological_plausibility_of_three_site_modulation.md](critiques/biological_plausibility_of_three_site_modulation.md) — strong-H5 not defensible on a single GRU; T/P split + opioid-analog descending head as the minimum upgrade.
- Synthesis: [ideas/nature_mi_paper_framing_synthesis.md](ideas/nature_mi_paper_framing_synthesis.md).

The rewrite adopts the user-locked **Hybrid Option A + B-on-H5**:
sober register, §1.5 union architecture, {1, 3, 5} rungs, *plus* the
Option B upgrade on H5 (T/P split + opioid descending head). The
multiplicative π̂ blend is dropped per RL-BDL §1.4. The framing
scaffold (six minimum claims, seven gaps G-A through G-G) is in
[ideas/nature_mi_paper_framing.md](ideas/nature_mi_paper_framing.md).

## 1. Project Goal

**GridWorld Pain** is a reinforcement-learning research platform for
**studying interoceptive neuromodulation as a substrate for pain-like
behaviour**. The aim is not to attach a "negative reward" to damage
events but to construct an environment and agent rich enough that a
*coordinated perceptual–mnemonic–policy reweighting under bodily
threat* emerges from the optimisation, and is measurably distinct
from reflex avoidance under nociception alone.

The project asks: given an embodied RL agent with homeostatic drives
(hunger, satiation, injury) and multimodal exteroception (nociception,
olfaction, vision, collision, proprioception, location), can an
*interoceptive ascending modulatory analog* produce channel-selective
precision-weighting, threat-context retention, and risk-sensitive
policy together — coordinately, on a shared slow latent — and
dissociate that joint reweighting from richer-but-still-peripheral
nociception baselines?

The grid world is designed with sufficient mechanical depth (predators
with heterogeneous threat profile, regenerating resources, injury
smoothing, metabolic cost, homeostatic reward, per-modality
state-dependent noise) to make precision-weighting a non-trivial
computation rather than a learned constant.

### 1.1 Nociception vs. pain — the operational distinction

Per pain-modeling memo
[§1](concepts/pain_vs_nociception_construct.md#1-the-operational-distinction-reviewer-acceptable-form):

- **Nociception** is a likelihood channel $o_{\text{noc}, t}$ —
  peripheral, fast, driver-of-reflex. CIP-style loss abolishes the
  reflex but leaves perceptual/mnemonic threat-processing intact
  (Brand & Yancey 1997; Cox et al. 2006).
- **Pain** is the posterior over a hidden bodily-threat variable
  $s_t$:
  $$p(s_t \mid o_{1:t}, \text{prior}_t) \propto p(o_t \mid s_t)\,
  p(s_t \mid s_{t-1})\, p(\text{prior}_t),$$
  with three decisive properties co-occurring: slow dynamics
  (Seymour, Crook & Chen 2023); joint downstream effect on
  perception, memory, policy via a common pathway (Vlaeyen & Linton
  2000; Eccleston & Crombez 1999); dissociability from input
  (chronic pain / placebo analgesia — Apkarian et al. 2009; Wiech
  2016; Büchel et al. 2014).

The §3 architecture operationalises the first two properties; the
third — dissociability — is what the C0/C1/C2 controls (§2bis) and
the Phase-3 opioid-analog descending head must demonstrate.

### 1.2 Robotics framing (editor-level hook)

Embodied agents need *more* than nociception. A thresholded force-kill
(§6) keeps a manipulator off a table; it does not reproduce the
post-injury reweighting seen in animals — persistent attentional bias
toward threat, longer integration of past harm, conflict between
protective withdrawal and competing drives. We argue this is a
*second* computation, distinct from nociception, in which an
interoceptive ascending modulatory analog reweights perception,
memory, and policy *together* via a shared slow latent. The robotics
contribution is a concrete minimum architecture for this. See §6.

## 2. Target Behavioural Signature

The four-property hypervigilance fingerprint is the empirical target.
Loose forms ("more cautious after injury") are consistent with
reward-shaping (FP-3), slow-input mimicry (FP-2), and modulator
collapse (FP-1) — all rule-out targets per pain-modeling
[§2.3](concepts/pain_vs_nociception_construct.md#23-false-positive-hypervigilance--what-to-rule-out).

### 2.1 The four properties (post-injury, time-locked, within-agent)

Per Crombez et al. (2005); pain-modeling §2:

- **(a) Attentional bias** — encoder gain $\gamma$ rises *selectively*
  on threat-coupled channels $\mathcal{T} = \{\text{noc}, \text{olf},
  \text{vis}\}$, not uniformly.
- **(b) Difficulty disengaging** — auto-correlation length of
  $\gamma_{\mathcal{T}}$ post-injury exceeds pre-injury baseline.
- **(c) Memory bias** — task-GRU update-gate bias $z_{\text{memory}}$
  shifts toward retention; correlated within-trial with (a).
- **(d) Avoidance / freezing** — PPO temperature drop / Dreamer
  reward-scale drop, *context-modulated*, outlasting the input.

The **joint** signature — all four co-modulated, time-locked,
within-agent, channel-selective, outlasting input — is the
fingerprint. Any one alone is a much weaker construct.

### 2.2 Construct register

Per the synthesis
[§1.1](ideas/nature_mi_paper_framing_synthesis.md#11-sober-register-for-the-headline)
the paper uses three sober registers in different surfaces:

| Surface | Register |
|---|---|
| Title / abstract / cover letter | "Interoceptive neuromodulation as a substrate for pain-like behaviour" |
| Body, mechanism sections | "An interoceptive ascending modulatory analog producing coordinated perceptual–mnemonic–policy reweighting under bodily threat" |
| Discussion | "Pain computation" foreshadowed only conditional on (i) C0/C1/C2 dissociation, (ii) cross-rung transfer, (iii) the H5 chronic-regime test passing under the §3 T/P + opioid architecture |

The phrase **"full pain syndrome" is dropped** (per pain-modeling §4.3
and neuromodulation §7.1). The affective/sensory dissociation (Price
2000; Rainville et al. 1997) is declared out of scope (pain-modeling
§5.3) and listed as a limitation.

## 2bis Construct-validity controls (C0, C1, C2)

Three standing controls run alongside the modulated agent at every
phase from Phase 2 onward (pain-modeling
[§3](concepts/pain_vs_nociception_construct.md#3-the-nociception-only-control--adjudication);
RL-BDL [§4](directions/architecture_for_pain_computation.md#4-c1-architectural-specification--making-the-construct-guardians-richer-nociception-baseline-real)):

| Control | What it is | What it rules out |
|---|---|---|
| **C0** — pure nociception | Damage signal in obs, no modulator, standard recurrent PPO/Dreamer | Survival-benefit floor; failure → fall back from Claim 2(a) to 2(b). |
| **C1** — slow-input mimic | Trainable scalar EMA $m_t = (1-\alpha) m_{t-1} + \alpha\,o_{\text{noc},t}$, $\alpha = \sigma(\theta_\alpha)$ learnable; broadcast through per-injection learned linear read-outs $\gamma_i = w_{\gamma,i}^A m_t + b_{\gamma,i}^A$ at A/B/C; same auxiliary loss; same seed budget; modulator hidden-size budget matched in extra FiLM-head capacity. | FP-2. **Cannot reach channel-selective $\gamma$** because $m_t$ is one number — per-channel post-injury *rate-of-rise* is a forced scalar broadcast (Bayesian-brain §3.2(c)). |
| **C2** — interoception-masked | Full modulator architecture, modulator's *input* has `injury, satiation, nutrition` zeroed (policy's encoder still sees them) | Architecture-only-no-interoceptive-read; C2's $r$ contrast is the §4.2 G2′ item 3 metric. |

The **load-bearing dissociation metric** (the spine of Claim 2(b))
is the **cross-channel rate-of-rise difference**

$$\Delta_t \gamma_{\mathcal{T}}^{(i)} - \Delta_t \gamma_{\mathcal{T}}^{(j)}$$

between two threat channels with different injury-scaled noise
profiles ($\alpha = 2.0$ on olfaction, $\alpha = 3.0$ on visual, in
the canonical preset). The modulator can produce a non-zero value;
C1 cannot, by construction. This single metric operationalises
"pain ≠ smoothed nociception".

Phase 4 is gated on **C0/C1/C2 dissociation + G1′ + G2′**, not on
G1+G2 alone (pain-modeling §4.2).

## 3. Architecture

Shared interoceptive ascending modulatory analog whose outputs are
injected at three sites of the main agent network. Locks the
synthesis §1.5 union spec, drops the multiplicative π̂ blend, adds
per-injection-site learnable-τ filters, and adopts the Option B
upgrade for H5 (two-modulator T/P split plus opioid-analog descending
head).

### 3.1 Injection sites (A, B, C)

| Site | What it does | Substrate analog (loose) | Hypothesis |
|---|---|---|---|
| **A — Perception** | FiLMNoNorm at the encoder. $\gamma$ unconstrained, init bias 1.0; $\beta$ unconstrained, init bias 0.0. **No LayerNorm in the modulated branch.** | ACh-like (Yu & Dayan 2005); precision-on-likelihood. | H1 |
| **B — Memory** | Gate-bias on task-GRU update gate. | 5-HT-like patience (Cohen et al. 2015); weak NE-tonic. | H2 |
| **C — Decision-making** | PPO temperature $T_\pi$ / Dreamer $z_{\text{reward}}$. | 5-HT aversive-inhibition + DA-tonic seam (Daw, Kakade & Dayan 2002; Niv et al. 2007). | H3 |

Each injection site has an explicit **low-pass filter on the modulator
state with a learnable time constant per site** (neuromodulation §4.2
option 1). The post-synaptic integration time of the modulator's
target population differs across A/B/C; making this explicit converts
the Bayesian-brain
[§4.2](concepts/active_inference_hypervigilance.md#42-the-fifth-property-ai-adds-timescale-currently-underspecified-in-g2)
timescale-ordering prediction $\tau_A < \tau_B < \tau_C$ into a
measurable property of the trained model.

### 3.2 Two-modulator T/P split (Option B upgrade for H5)

The single GRU is replaced with two recurrent cores (neuromodulation
[§2.3, §3.2](critiques/biological_plausibility_of_three_site_modulation.md#23-minimum-upgrade-to-defend-strong-h4)):

- **Modulator P (phasic, fast).** $\tau_P \sim 1$–$5$ steps.
  Drives Injection A and the phasic component of C. Substrate analog:
  basal-forebrain ACh + LC NE phasic.
- **Modulator T (tonic, slow).** $\tau_T \sim 50$–$200$ steps.
  Drives Injection B and the persistent component of C. Substrate
  analog: dorsal-raphe 5-HT + descending opioid tone.
- **Asymmetric coupling.** Learned $T \leftarrow P$ pathway (phasic
  events drive tonic — Sara 2009) plus weak $P \leftarrow T$ feedback
  (adaptive-gain). The *learned coordination* is what makes H4-strong
  a finding rather than an assumption.
- **Hysteresis.** $T$'s decay is gated by a learned variable; under
  sufficient cumulative $P$ excursion, $T$ enters a high-tonic
  near-absorbing state that does not relax even after $P$ returns to
  baseline. Substrate analog of the chronic-pain transition (Apkarian
  et al. 2009; Bannister & Dickenson 2017).
- **Per-modulator teaching signals.** Each modulator has a documented
  gradient source: $P$ trained on the heteroscedastic precision NLL
  via the precision head; $T$ trained on a sustained-survival
  auxiliary (or the policy's pragmatic-value gradient)
  (neuromodulation §6). Without per-modulator teaching, the
  multi-modulator analog is rhetorical even if the architecture is.

Combined parameter count is comparable to the prior single-GRU at
hidden size 16. Engineering cost: ~2–3 weeks before the Phase-3
factorial (neuromodulation §3.2); +30–60 runs in the headline budget.

### 3.3 Opioid-analog descending modulation head

A learned multiplicative gate on the **nociception channel itself**,
conditioned on the tonic modulator state $h_T$:

$$o_{\text{noc}, t}^{\text{eff}} = g(h_T) \cdot o_{\text{noc}, t},
\qquad g(\cdot) \in (0, 1].$$

This is the architectural pre-condition for the placebo-analgesia
analog (Eippert et al. 2009; Wager & Atlas 2015; Wiech 2016).
Failure of descending tone under chronic high-$T$ is the substrate
analog of chronic-pain controller failure (Bannister & Dickenson
2017).

### 3.4 Precision head (parallel auxiliary, no multiplicative blend)

Two-layer MLP on $h_P$ emits per modality a log-precision $\hat{s}_i
= \log \hat{\pi}_i$ and a reconstructed $\hat{o}_{t+1}$. Trained by
per-modality Kendall–Gal NLL on observation reconstruction (RL-BDL
[§1.5](directions/architecture_for_pain_computation.md#15-the-smallest-architectural-commitment-formally);
Bayesian-brain [§3.2(b)](concepts/active_inference_hypervigilance.md#b-heteroscedastic-precision-loss--phase-3)):

$$L_{\text{prec}} = \sum_{i=1}^{9} \tfrac{1}{2}\,\exp(\hat{s}_i)
\cdot \overline{\varepsilon_i^2} - \tfrac{1}{2}\,\hat{s}_i,
\qquad L_{\text{total}} = L_{\text{PPO}} + \lambda_{\text{prec}}
\cdot L_{\text{prec}}.$$

**$\hat{\pi}$ does not multiplicatively gate $\gamma$** (no blend
$\pi_{\text{gate}} \cdot x_{\text{mod}} + (1-\pi_{\text{gate}})
\cdot x_{\text{bypass}}$). The gradient of $L_{\text{prec}}$ flows
into $h_P$ and thence to all FiLM heads, but $\hat{\pi}$ itself is
*measurement only* in the headline factorial. The reasoning is
RL-BDL [§1.4](directions/architecture_for_pain_computation.md#14-why-i-recommend-removing-the-precision-gating-blend-from-phase-3-for-now):
the blend creates a non-identifiability between $\gamma_i$ and
$\hat{\pi}_i$, and *destroys* the C1 dissociation argument because
$\pi_{\text{gate}}^{(i)}$ is itself per-channel — C1 could fake
channel-selective *effective* gain through the gate. AI theory does
not require the blend (Bayesian-brain §1.2). The blend may be
re-introduced as a follow-up cell after the simpler architecture
clears G2′.

### 3.5 Modulator inputs

Both $P$ and $T$ read the full obs vector including interoception
(`injury, satiation, nutrition`). C2 zeroes these in the modulator
input only. Modulators are causal.

### 3.6 Architecture summary

| Component | Specification |
|---|---|
| Injection A | FiLMNoNorm; identity-init; no LayerNorm in modulated branch (RL-BDL §1.3, §1.5). |
| Injection B | Gate-bias on task-GRU update gate, driven by $h_T$. |
| Injection C | Existing $T_\pi$ / $z_{\text{reward}}$ head, driven by $h_T$ + phasic $h_P$. |
| Per-injection $\tau$ filter | Learnable-$\tau$ low-pass at each of A/B/C (neuromodulation §4.2 option 1). |
| Modulator P | GRU, fast; reads full obs + interoception. |
| Modulator T | GRU, slow; learned $T \leftarrow P$ coupling; hysteresis on decay. |
| Opioid descending head | Multiplicative gate on $o_{\text{noc}}$ conditioned on $h_T$. |
| Precision head | MLP on $h_P$; per-modality Kendall–Gal NLL; **no multiplicative gate on $\gamma$**. |
| Auxiliary loss | $\lambda_{\text{prec}} \in \{0.1, 0.5, 1.0\}$ swept on the canonical rung. |

## 4. Main Issue and Success Gates

The central blocker remains empirical: the diagnosis series
[NMN_PERFORMANCE_DIAGNOSIS_v1–v8](../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)
shows the modulated agent does not reliably outperform the LayerNorm
baseline. Bayesian-brain
[§3.1](concepts/active_inference_hypervigilance.md#31-the-degeneracy-in-active-inference-language)
re-reads the v8 null as the *correct free-energy solution to a
homogeneous-noise environment* — under $\Sigma_i = \sigma^2 I$, the
optimal precision is uniform and the modulator's optimal output is
the constant collapse.

Four interacting causes:

1. **No precision training signal.** RL loss rewards behaviour, not
   calibrated gating. Phase 3's heteroscedastic NLL is the gradient
   $\partial F_t / \partial s_i$ the AI fixed point requires
   (Bayesian-brain §3.2(b)).
2. **Static FiLM cannot express per-timestep reliability.** Coupled
   with the heteroscedastic loss (giving $h_P$ time-varying precision
   information) and Phase-1 state-dependent noise, the modulator's
   $\gamma$ becomes per-step per-channel reliability-tracking via the
   shared modulator state.
3. **Noise landscape does not reward precision.** Homogeneous noise
   makes the AI fixed point degenerate. Phase 1's heterogeneous,
   state-dependent noise is the AI-theoretic precondition for
   non-degeneracy (Bayesian-brain §3.2(a)).
4. **Temperature saturation and critic instability.** v8's PPO
   temperature head saturates; under GAE the modulator destabilises
   the critic. The headline factorial is therefore **MC-only**
   (RL-BDL §2.4); GAE reported as supplementary on one or two cells.
   Saves ~50% of run budget.

### 4.1 G1′ — channel-rank non-degeneracy on the LayerNorm baseline

G1 in the prior plan said "noise creates headroom" via baseline
survival drop. Bayesian-brain
[§3.2(a)](concepts/active_inference_hypervigilance.md#a-heterogeneous-noise-landscape--phase-1)
sharpens this to a measurable property of the *environment*: the AI
fixed point under the canonical preset has a non-trivial channel
ranking iff per-channel residual variance is non-degenerate.

**Operational definition** (RL-BDL [§5.2](directions/architecture_for_pain_computation.md#52-the-cheaper-diagnostic)):
G1′ holds iff, on the LayerNorm baseline with a *frozen-weights*
reconstruction head (random-init or held-out-trained, not in the loss
path), the per-modality MSE max/min ratio under high-injury
(`injury > 0.5`) exceeds 2.0:

$$\frac{\max_i \mathrm{MSE}_i(\text{injury} > 0.5)}{\min_i
\mathrm{MSE}_i(\text{injury} > 0.5)} > 2.$$

Single measurement, ~1 wallclock-hour per noise preset, reusing the
LayerNorm baseline's already-trained encoder. **G1 alone is necessary
but not sufficient; G1′ must also pass before Phase 2.** The
threshold is replaceable with the theory-derived $\Delta\gamma
\propto \log(\Sigma_{\mathcal{N}}/\Sigma_{\mathcal{T}})$ from the
canonical preset's $\Sigma_i(s_t)$.

### 4.2 G2′ — four-property AI fingerprint (pre-registered)

Five pre-registrable quantities, in order of decreasing theoretical
force, per Bayesian-brain
[§4.3](concepts/active_inference_hypervigilance.md#43-what-this-means-for-the-papers-pre-registration):

1. **Sign** of all three injection-site responses post-injury:
   $\gamma_{\mathcal{T}} \uparrow$, $z_{\text{memory}} \downarrow$,
   $T_\pi \downarrow$. *Load-bearing; falsification target.*
2. **Channel selectivity in $\gamma$**: $\gamma_{\mathcal{T}} -
   \gamma_{\mathcal{N}} > 0$ post-injury. Magnitude derived from
   $\Delta\gamma \propto \log(\Sigma_{\mathcal{N}}/\Sigma_{\mathcal{T}})$,
   not a round 0.2 number.
3. **Cross-domain zero-lag correlation contrast** between the
   modulated agent and C2: modulated $r$ exceeds C2 $r$ by
   $\geq 0.2$. (Pain-modeling's absolute $r \geq 0.3$ is replaced
   because C1 fakes high absolute $r$.)
4. **Timescale ordering** $\tau_A < \tau_B < \tau_C$ with each ratio
   $\geq 2$. *Architecturally a predicate*, made measurable by §3.1's
   per-injection-site learnable-$\tau$ filters; with the §3.2 T/P
   split it is also a between-modulator predicate.
5. **C1 dissociation** — cross-channel rate-of-rise difference of
   §2bis. *The construct-spine metric.*

Lag bounded: peak cross-correlation at lag $\leq \min(5,
2\tau_{\text{mod}})$. Items 1, 2, 4, 5 are qualitative theoretical
predictions with falsification meaning (no multiple-comparisons
inflation); item 3 is the only quantitative cross-correlation
contrast and uses the C2 contrast rather than absolute level.
Pre-registration of cross-correlation lag and null distribution is
an `experiment-designer` deliverable before Phase 4 reads
(Bayesian-brain §4).

### 4.3 Phase exit criteria

| Phase | Exit criterion |
|---|---|
| 1 | G1 (baseline survival drop) **and** G1′ (channel-rank non-degeneracy on canonical preset). |
| 2 | Characterised null on canonical preset; one variant identified as the Phase-3 starting point. |
| 3 | C0/C1/C2 dissociate from the modulated agent on the §2bis metric; G2′ holds across seeds. |
| 4 | All Phase-3 criteria + cross-rung modulator-state-freeze transfer probe (gap G-G) holds on rungs 1 and 5. |

## 4bis Phase 0 — Environment ladder ({1, 3, 5})

Per pain-modeling [§5.1](concepts/pain_vs_nociception_construct.md#5-implications-for-the-plan-construct-validity-side)
and RL-BDL [§3.2](directions/architecture_for_pain_computation.md#32-minimum-rung-set):
construct-mandatory rung set is {1, 3, 5}. Rungs 2 and 4 are
methodology-flavour, listed as planned post-submission extensions.

| Rung | Description | Necessary for |
|---|---|---|
| **1** | Empty grid + injury. No predators, no food. | C0/reflex floor for Claim 1; rules out drive-conflict confound (FP-3 lower bound). Pain-modeling §5.1 condition (i). |
| **3** | + predators with heterogeneous threat profile + canonical heterogeneous noise preset (state-dependent injury-scaled $\Sigma_i$ on $\mathcal{T}$). | Channel-selective fingerprint test; the 11-cell factorial + C0/C1/C2 (the canonical-rung 63-run cell of RL-BDL §2.3). |
| **5** | + slow injury dynamics (long recovery window $\gg$ episode length). | H5 chronic-pain test under the §3 T/P + opioid architecture; timescale ordering. |

The §3 architecture is the same across all three rungs; the only
legitimate per-rung free hyperparameter is $\lambda_{\text{prec}}$
(RL-BDL §3.3). Re-tuning the head architecture would compromise the
platform claim.

## 5. Phases

Each phase clears a concrete gate before the next. Phase 4 is the
**primary scientific output** with run-level metrics specified up
front (no longer an "appendix" deferred behind G1+G2).

### Phase 0 — Engineering preconditions

Hand-off to `developer` (per RL-BDL §8 and neuromodulation §7.2):

1. C1 EMA modulator (`modulation.type = "EMAControl"`).
2. Per-injection-site learnable-$\tau$ low-pass filter on modulator
   state.
3. Remove the multiplicative π̂ blend; precision head emits
   $(\hat{o}_{t+1}, \hat{s})$ as parallel auxiliary outputs only.
4. T/P split modulator (two GRU cores; learned $T \leftarrow P$
   coupling; hysteresis on $T$ decay).
5. Opioid-analog descending head: multiplicative gate on
   $o_{\text{noc}}$ conditioned on $h_T$.
6. Frozen-head reconstruction logger on the LayerNorm baseline (G1′
   measurement); per-modality MSE conditioned on `injury > 0.5` vs.
   `injury < 0.1`.
7. Rung-1 and rung-5 environment configs.

### Phase 1 — Reshape the noise landscape until baseline bleeds *and* G1′ passes

- **Heterogeneous per-modality noise** with sharp reliability contrast
  (high-$\sigma$ olfaction/visual; low-$\sigma$ proprioception/
  collision).
- **State-dependent injury-scaled noise** on $\mathcal{T}$ so
  reliability is time-varying and correlated with interoceptive
  state (the signal a shared modulator can latch onto).
- **Magnitudes tuned against G1 + G1′.** Sweep $\sigma_{\text{base}}$
  and `injury_scale` on the LayerNorm baseline (no modulator) across
  $\geq 3$ seeds; pick a profile where (a) baseline survival drops by
  a meaningful margin vs. no-noise *and* (b) the frozen-head MSE
  ratio exceeds 2.0 at high injury. **If no preset clears G1′,
  retune before Phase 2.**

Exit: **G1 + G1′ on the LayerNorm baseline at the canonical preset.**

### Phase 2 — Reproduce the null, probe FiLM variants

- Re-run the unmodulated baseline and FiLM variants
  ([FILM_MODULATION_PLAN.md](../develop/active/filim/FILM_MODULATION_PLAN.md);
  [FiLM_PAPERS_REVIEW.md](../develop/active/filim/FiLM_PAPERS_REVIEW.md)):
  Multiplicative, PreActivation, FiLM (LayerNorm-targeted),
  FiLMNoNorm — under identical canonical noise, seeds, horizon.
- Record per-variant $\gamma / \beta$ distributions, gate health,
  temperature trajectories, *and* Phase-4 metrics (per-step γ,
  $z_{\text{memory}}$, $T_\pi$, with injury-event annotations).
- Honest expected outcome: the null replicates. Diagnostic value:
  *why* each variant fails (collapse to identity, temperature
  saturation, critic destabilisation).

Exit: characterised null on canonical noise; FiLMNoNorm identified as
the Phase-3 starting point (RL-BDL §1.3).

### Phase 3 — Add the precision training signal (T/P split + opioid head)

The §3 architecture is now active in full. This is the
construct-spine phase — Phase 3's exit tests Claims 1, 2, 3.

- **$\lambda_{\text{prec}}$ sweep** on the canonical rung (full
  modulator only): $\lambda \in \{0.1, 0.5, 1.0\}$, 3 × 5 = 15 runs
  (RL-BDL §2.2).
- **Locked-$\lambda$ factorial on canonical rung 3**: the $2^3 = 8$
  injection-site lesion subsets + shared-core lesion + C0 / C1 / C2.
  $11 \times 5 + 1 \times 8 = 63$ runs. The shared-core lesion
  replaces $h_P$ and $h_T$ with three independent per-injection GRUs
  of comparable capacity — H4-strong negative test.
- **Opioid-head sub-factorial on rung 5**: T-only / P-only / T+P /
  T+P+opioid-head ablation cells, ~30–60 runs (neuromodulation §3.2;
  the H5 upgrade).

Exit:
- **Construct dissociation.** C0/C1/C2 dissociate from the modulated
  agent on the §2bis metric (cross-channel rate-of-rise $> 0$ for
  modulated, $\approx 0$ for C1; cross-domain $r$ contrast modulated
  vs. C2 $\geq 0.2$).
- **AI fingerprint.** G2′ items 1–4 hold across seeds.

### Phase 4 — Hypervigilance as the primary scientific readout

The figure-producing phase. Once Phase 3 dissociates, the project
pivots from "does it work" to "what is it showing us."

**Run-level metric specs (recorded from Phase 1 onward, primary
observables not diagnostic):**

- Per-step $\gamma_i$ for all 9 modalities, 1-step resolution,
  injury-event annotated.
- Per-step $z_{\text{memory}}$, $T_\pi$ / $z_{\text{reward}}$.
- Per-step $h_P$, $h_T$ trajectories; descending-head gate $g(h_T)$
  activation level.
- Per-modality residual variance from the precision head (or the
  frozen-head logger in Phase 1/2).
- Episode budget: as needed for the 2-event-per-episode minimum on
  the within-trial $r$ estimator at the headline cell (8 seeds).

**Phase 4 analyses:**

1. Time-locked traces of $\gamma$, $z_{\text{memory}}$, $T_\pi$,
   opioid-head gate around injury events; bootstrap CIs across seeds.
2. Cross-correlation across A/B/C — H4 weak (shared-latent
   coordination); H4 strong (T/P learned coordination) under the
   upgraded architecture.
3. Modulator-state timescale measurements $\tau_A, \tau_B, \tau_C$;
   pre-registered ordering test.
4. **H5 chronic-regime test.** Under sustained injury exposure, does
   $h_T$ enter the high-tonic absorbing state? Does the opioid
   descending analog *fail* to suppress nociception in the chronic
   regime, producing behaviorally-detectable persistent caution
   under low $o_{\text{noc}}$? Three-regime acute/recovery/chronic
   classifier per neuromodulation §3.2.
5. **Cross-rung modulator-state-freeze transfer probe (gap G-G).**
   Train the modulator on rung 3, freeze it, deploy on rungs 1 and
   5. Does the channel-selective $\gamma$ structure transfer? Does
   the H5 chronic regime probe still fire on rung 5? This is what
   makes the modulator-state representation a *generative model*
   rather than a per-rung tuning artefact — the strongest evidence
   licensing the discussion-section "pain computation"
   foreshadowing.
6. **Survival benefit Δ.** Modulated vs. C0 on rungs 1 and 3; full
   modulated vs. T-only / P-only / T+P / T+P+opioid-head on rung 5.

### Phase 5 — Post-submission planned extensions (out of scope)

Rungs 2 / 4. AI vs. heteroscedastic-BDL regional contrast
(Bayesian-brain §2.3). Affective vs. sensory dissociation (declared
limitation per pain-modeling §5.3). Full Doya four-modulator
decomposition.

## 5bis Run budget

Per RL-BDL [§2.3](directions/architecture_for_pain_computation.md#23-the-total)
plus the neuromodulation §3.2 H5 upgrade:

| Component | Cells × seeds | Runs |
|---|---|---|
| $\lambda_{\text{prec}}$ sweep on canonical rung (full modulator) | 3 × 5 | 15 |
| Factorial + shared-core + C0/C1/C2 on canonical rung at locked $\lambda$ | 11 × 5 + 1 × 8 | 63 |
| Cross-rung transfer (headline + C0 + C1) on rungs 1 and 5 | 3 × 2 × 5 | 30 |
| **Option A subtotal** | | **108** |
| H5 upgrade: T/P sub-factorial on rung 5 + opioid-head ablations + chronic-regime detection | ~6–12 × 5 | **30–60** |
| **Total headline budget (Hybrid A + B-on-H5)** | | **138–168** |

Engineering: ~2–3 weeks for the T/P + opioid head before the Phase-3
factorial launches. **MC-only** (no GAE in the headline factorial).

## 6. Robotics motivation (closes G-D)

Industrial manipulators today implement *thresholded force-kill
nociception*: torque/force sensors, hard-stop on threshold breach,
reflexive retreat. Effective for the immediate damage event;
invariant under context. This is the alarm-bell of pain-modeling
§1.

**Whole-policy reweighting after damage-history** is qualitatively
different. Once a manipulator has experienced damage in a context,
the *next* contact in any nominally-similar context should be
approached with elevated channel-selective sensory gain (force,
proximity), longer integration of past contact outcomes (memory),
and risk-sensitive policy (slower trajectories, more conservative
grasps). This generalises the protective behaviour of the alarm bell
across drive contexts — productivity vs. safety, fast vs. precise —
without per-context engineering.

The closed hypothesis: **whole-policy reweighting after
damage-history is the cheap way to generalise the protective
behaviour of nociception across drive contexts.** The §3 architecture
is the minimum substrate — a shared interoceptive ascending
modulatory analog driving channel-selective sensory gain,
threat-context retention, and risk-sensitive policy from a common
slow latent.

This is also the cover-letter framing. It commits to an *engineering*
claim about generalisable protective behaviour, not to a clinical
"pain computation" claim.

## 7. Platform release (closes G-F)

The simulation-methodology contribution requires a release story.

- **Code release.** All training, environment, modulator, FiLM,
  precision-head, T/P-split, opioid-head code under the project's
  existing license at [src/](../../src/). Reproducibility tag at
  submission.
- **Single-GPU baseline reproducibility.** A documented config and
  command that reproduces the headline cell on a single consumer GPU
  in a documented wallclock time. The §3 architecture is rung-stable
  (RL-BDL §3.3): single config family, not per-rung surgery.
- **How-to-add-a-rung extension path.** Documented
  `configs/continual/` template for adding rungs 2, 4, or new rungs
  (custom predator dynamics, custom noise profile). The 9-modality
  obs-vector structure is preserved; the precision head's
  per-modality output dim is rung-stable unless a new modality is
  added.
- **Pre-registration document.** The five G2′ quantities and the C1
  cross-channel rate-of-rise dissociation metric, with theory-derived
  thresholds, released alongside the code.
- **Modulator-state-freeze transfer probe** released as a standalone
  notebook so reviewers can verify cross-rung generalisation without
  re-training.

This is documentation work, not new science; it is what makes the
methodology contribution credible. Hand-off to `senior-developer`
once the headline factorial lands.

## 8. Pain-computation lineage (closes G-E)

The paper engages an explicit lineage:

- **Predictive-coding accounts of chronic pain.** Büchel, Geuter,
  Sprenger, Eippert (2014, *Neuron*); Wiech (2016, *Science*);
  Tabor & Burr (2019, *Curr. Opin. Behav. Sci.*).
- **Interoceptive inference.** Seth & Friston (2016, *Phil. Trans.
  B*); Barrett & Simmons (2015, *Nat. Rev. Neurosci.*).
- **Bayesian placebo decomposition.** Geuter, Koban & Wager (2017,
  *Annu. Rev. Neurosci.*).
- **Fear-avoidance.** Vlaeyen & Linton (2000, *Pain*); Vlaeyen,
  Crombez & Linton (2016, *Pain*).
- **Hypervigilance / attentional bias.** Eccleston & Crombez (1999,
  *Psychol. Bull.*); Crombez, Van Damme & Eccleston (2005, *Pain*).
- **Chronic-pain controller failure.** Apkarian, Baliki, Geha (2009,
  *Prog. Neurobiol.*); Bannister & Dickenson (2017, *J. Physiol.*).
- **Three-tier nociception/post-injury/chronic taxonomy.** Seymour,
  Crook & Chen (2023, *Nat. Rev. Neurosci.*).

**Which subset the paper operationalises.** The §3 architecture
operationalises **(a)** the predictive-coding / interoceptive-
inference account of post-injury hypervigilance (Bayesian-brain §1)
through the precision-head + heteroscedastic NLL on $h_P$;
**(b)** the descending-opioid component of the Bayesian placebo
decomposition through the §3.3 opioid-analog descending head; and
**(c)** the chronic-pain controller-failure analog through the T/P
split's hysteresis. **Out of scope:** affective vs. sensory
dissociation (Price 2000; Rainville et al. 1997), declared as a
limitation. Pre-existing references live under
[docs/project/references/](references/); the four professor memos
add the complete bibliography in their §8 sections.

## 9. Key references

**Develop tree (technical depth):**

- [ENVIRONMENT_SUMMARY.md](../environment/ENVIRONMENT_SUMMARY.md) — environment mechanics and sensor layout.
- [NEUROMODULATION_ALGORITHM.md](../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) — three-injection-site neuromodulation, hypotheses H1–H5.
- [FILM_MODULATION_PLAN.md](../develop/active/filim/FILM_MODULATION_PLAN.md), [FiLM_PAPERS_REVIEW.md](../develop/active/filim/FiLM_PAPERS_REVIEW.md) — FiLM variants and injection strategies.
- [FiLM_ENSEMBLE_SENSORY_PRECISION.md](../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md) — heteroscedastic precision loss.
- [PRECISION_MODULATION_ARCHITECTURE.md](../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md), [PRECISION_MODULATION.md](../develop/active/precision/PRECISION_MODULATION.md) — precision-gated modulation architecture (PPO + DreamerV3).
- [NMN_PERFORMANCE_DIAGNOSIS_v8.md](../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) — null-result diagnosis series.

**Project tree (the authorising chain for this rewrite):**

- [concepts/pain_vs_nociception_construct.md](concepts/pain_vs_nociception_construct.md) — pain-modeling memo.
- [concepts/active_inference_hypervigilance.md](concepts/active_inference_hypervigilance.md) — Bayesian-brain memo.
- [directions/architecture_for_pain_computation.md](directions/architecture_for_pain_computation.md) — RL/Bayesian-DL memo.
- [critiques/biological_plausibility_of_three_site_modulation.md](critiques/biological_plausibility_of_three_site_modulation.md) — neuromodulation memo.
- [ideas/nature_mi_paper_framing_synthesis.md](ideas/nature_mi_paper_framing_synthesis.md) — cross-professor synthesis.
- [ideas/nature_mi_paper_framing.md](ideas/nature_mi_paper_framing.md) — original framing memo (six claims, seven gaps).
- [archive/project_plan_2026-04-12.md](archive/project_plan_2026-04-12.md) — archived prior plan.

## 10. How this document is used

Read before any work on env tuning, modulator architecture, or
training analysis. It does not commit to exact hyperparameters,
WandB run names, or file-level diffs; each phase spawns its own
issue-plan or training-analysis doc in
[docs/develop/](../develop/) (per [docs/TEMPLATES/](../TEMPLATES/))
where the tactical details live.

Each phase maps to a subset of the **six minimum claims** of
[ideas/nature_mi_paper_framing.md §2](ideas/nature_mi_paper_framing.md#2-minimum-claims-the-paper-has-to-defend):

| Phase | Claims defended |
|---|---|
| 0 (engineering) | Precondition for all. |
| 1 (G1′) | Precondition for Claim 2(b); the AI-fixed-point environment check. |
| 2 (replicate null) | Diagnostic; not load-bearing. |
| 3 (precision head + T/P + opioid) | **C1 (nociception ≠ pain), C2 (pain computation buys something), C3 (whole-network)** via canonical-rung factorial + C0/C1/C2 + shared-core lesion. |
| 4 (hypervigilance readout, transfer, chronic regime) | **C4 (time-locked, persistent, replicates), C5 (methodology)**, plus discussion-section foreshadowing of Register 3 conditional on transfer + chronic regime. |
| Platform release (§7) | **C6 (platform).** |
| Robotics framing (§6) | Editor-level framing for all claims; mostly C1, C2. |

When a phase's exit gate is cleared, cross-link the resulting
analysis back to this plan so that the project's state is always
derivable from this single entry point.
