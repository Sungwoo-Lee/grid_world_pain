---
title: Active Inference and Hypervigilance — Concept Memo
topic: active-inference / predictive-coding / precision
status: draft
author: professor-bayesian-brain
created: 2026-05-07
last_updated: 2026-05-07
closes_gaps: G-E (Bayesian-brain side); supplies theoretical spine for Claim 2(b)
---

# Active Inference and Hypervigilance — Concept Memo

> One-line summary: **Hypervigilance is not "increased gain" — it is *channel-selective* increased gain on threat-coupled channels under an injury-elevated prior over the latent threat state. Active inference predicts a four-property fingerprint (sign, lag, selectivity, timescale) that is sharper than G2 currently states; the v8 null is the *degenerate* fixed point of free-energy minimisation in a homogeneous-noise environment, and only the heterogeneous-noise landscape (Phase 1), the heteroscedastic precision loss (Phase 3), and the C1 slow-input control jointly force the *non-degenerate* solution that earns the "pain computation" headline.**

> Construct register verdict: **The Bayesian-brain account does not require — and at this point in the project does not yet license — the headline term "pain computation". It requires the sober register ("interoceptive neuromodulation as a substrate for pain-like behaviour") until the C1 dissociation is in hand. I concur with the pain-modeling memo §4.4.**

> Anchors: [project_plan.md §§2–4, G1, G2, Phase 1, Phase 3, Phase 4](../project_plan.md);
> [NEUROMODULATION_ALGORITHM.md §1.4 H1–H5](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md);
> [PRECISION_MODULATION_ARCHITECTURE.md §§1, 2.4, 3](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md);
> [FiLM_ENSEMBLE_SENSORY_PRECISION.md §§1.1, 3.1](../../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md);
> [NMN_PERFORMANCE_DIAGNOSIS_v8.md](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md);
> [pain_vs_nociception_construct.md §§2.2, 2.3, 3](pain_vs_nociception_construct.md).

---

## 1. The Generative Model and Free-Energy Decomposition

I write down the generative model the project is *implicitly* committing to when it claims active-inference framing, then read off what the precision matrix has to do.

### 1.1 Latents, observations, prior

Let $s_t \in \mathbb{R}^{d_s}$ be the hidden body/world state at time $t$ (injury, danger of context, predator proximity). Let $o_t \in \mathbb{R}^{d_o}$ be the multimodal observation (nine modalities indexed by $i \in \{\text{noc}, \text{olf}, \text{vis}, \text{col}, \text{prop}, \text{loc}, \text{inj}, \text{sat}, \text{nut}\}$). Partition the modalities into a **threat-coupled subset** $\mathcal{T} = \{\text{noc}, \text{olf}, \text{vis}\}$ and a **threat-irrelevant subset** $\mathcal{N} = \{\text{prop}, \text{col}, \text{loc}\}$. Interoceptive channels $\mathcal{I} = \{\text{inj}, \text{sat}, \text{nut}\}$ drive the prior (Seth & Friston 2016).

Generative model:
$$
p(o_{1:T}, s_{1:T}, a_{1:T}) \;=\; p(s_0)\,\prod_{t=1}^{T} p(s_t \mid s_{t-1}, a_{t-1})\, p(o_t \mid s_t).
$$

Linearise the per-step likelihood around the agent's expectation $\mu_t \approx \mathbb{E}_q[s_t]$. With Gaussian channel-wise noise,
$$
p(o_t^{(i)} \mid s_t) \;=\; \mathcal{N}\!\bigl(g_i(s_t),\, \Sigma_{i}(s_t)\bigr), \qquad
\Pi_s(s_t) \;\equiv\; \mathrm{blkdiag}_i\!\bigl(\Sigma_{i}(s_t)^{-1}\bigr).
$$

The state-dependence of $\Sigma_i$ is the project's key environmental commitment: under the canonical noise preset of [project_plan.md §Phase 1](../project_plan.md#phase-1--reshape-the-noise-landscape-until-the-baseline-bleeds), olfaction and vision become *less* reliable when injury is high (Phase 1 calls this "injury-scaled noise"), so $\Sigma_i$ is a function of $s_t$, not a constant. **This is the architectural commitment that makes precision a *computation* rather than a fixed parameter** — without it, precision degenerates to a learned constant and the active-inference framing has no purchase.

### 1.2 Variational free energy with explicit precision

Adopt a Gaussian variational posterior $q(s_t) = \mathcal{N}(\mu_t, \Sigma_t^q)$. The per-step variational free energy is
$$
F_t \;=\; \underbrace{\tfrac{1}{2}\,\varepsilon_t^\top\, \Pi_s\, \varepsilon_t}_{\text{precision-weighted prediction error}} \;-\; \underbrace{\tfrac{1}{2}\log\det \Pi_s}_{\text{precision regulariser}} \;+\; \underbrace{\mathrm{KL}\!\bigl[\,q(s_t) \,\|\, p(s_t \mid s_{t-1}, a_{t-1})\bigr]}_{\text{prior-divergence term}} \;+\; \text{const},
$$
where $\varepsilon_t = o_t - g(\mu_t)$ is the multimodal sensory prediction error. This is the same expression as the heteroscedastic loss of Kendall & Gal (2017) (cf. [FiLM_ENSEMBLE_SENSORY_PRECISION.md §3.1](../../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md), eq. on log-variance head) but read in the active-inference direction: **it is the quantity the brain (and the agent) is supposed to *minimise jointly over $\mu_t$ and $\Pi_s$***.

The gradient of $F_t$ with respect to the per-channel log-precision $s_i = \log \pi_i$ (treating channels as independent for clarity; the project's per-modality grouping in [PRECISION_MODULATION_ARCHITECTURE.md §2.3](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md) preserves this separability) is
$$
\frac{\partial F_t}{\partial s_i} \;=\; \tfrac{1}{2}\,\bigl(\,\pi_i\,\varepsilon_{i,t}^2 \;-\; 1\,\bigr),
\qquad
\pi_i^{*} = \frac{1}{\mathbb{E}[\varepsilon_{i,t}^2]}.
$$

This is Friston's precision-update equation $\dot{\Pi}_s \propto \beta - \varepsilon_s^2$ in gradient form ([PRECISION_MODULATION_ARCHITECTURE.md §2.4](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md), Friston 2023). The three architectural facts that matter follow:

1. **The fixed point of $s_i$ is the inverse empirical variance of channel $i$ — not a global gain knob.** If channels differ in their state-conditional noise, the optimal precision profile is *non-uniform across channels*, and is *time-varying* in lockstep with the state-dependent noise.
2. **Increased precision on threat channels only minimises $F_t$ when those channels' prediction error has dropped relative to others.** Hypervigilance, in this framing, is not "trust threat channels more because we are scared"; it is "the agent's posterior over $s_t$ has shifted into a regime where threat channels are *more informative* about the latent, and therefore optimally weighted higher".
3. **The injury-elevated prior is what couples the perceptual-precision shift to the policy shift.** This is what the next subsection makes explicit.

### 1.3 Expected free energy and policy

Active inference promotes policies $\pi$ to random variables and selects them by minimising expected free energy:
$$
G(\pi) \;=\; \underbrace{\mathbb{E}_{q(o,s\mid\pi)}\bigl[\,\mathrm{KL}[q(s\mid o,\pi)\,\|\,q(s\mid\pi)]\,\bigr]}_{\text{epistemic value (info gain)}} \;-\; \underbrace{\mathbb{E}_{q(o\mid\pi)}\bigl[\log p(o\mid C)\bigr]}_{\text{pragmatic value (preference)}},
$$
where $p(o\mid C)$ is the prior preference (homeostatic targets + injury-avoidance). Two facts matter for the project:

- **Pragmatic value couples to interoception.** The prior $p(o\mid C)$ over preferred observations places mass on low-injury, satiated, fed states. When injury is high, expected pragmatic loss is dominated by injury cost, biasing $G(\pi)$ towards risk-averse policies. *This is exactly* the role Injection C plays in [NEUROMODULATION_ALGORITHM.md H3](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md): PPO temperature drop / Dreamer reward-scale drop *should* be readable as a shift in $\arg\min_\pi G(\pi)$ under an injury-elevated $p(o\mid C)$, not as an independent "freezing knob".
- **Epistemic value couples to precision.** The KL inside $G(\pi)$ depends on $\Pi_s$: under high precision on threat channels, threat-related observations are expected to *resolve more uncertainty* about $s_t$, which in turn makes information-seeking policies (orient toward threat, scan) appear epistemically valuable. **This is the active-inference derivation of attentional capture by threat under injury** — and it predicts a *specific sign*: epistemic-value gradients should rise on threat channels post-injury, not on threat-irrelevant channels.

### 1.4 Mapping to Injection sites A / B / C

| AI quantity | Project component | Hypothesis (cross-ref) | Sign / direction post-injury |
|---|---|---|---|
| $\Pi_s$ on $\mathcal{T}$ (threat-channel precision) | **Injection A.** Encoder FiLM $\gamma$ on threat-coupled features; precision head $\hat{\pi}_i$ ([PRECISION_MODULATION_ARCHITECTURE.md §2](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md)) | H1 | $\gamma_{\mathcal{T}} \uparrow$, $\gamma_{\mathcal{N}}$ flat or $\downarrow$ |
| Prior dominance $p(s_t \mid s_{t-1})$ over likelihood | **Injection B.** Memory-gate bias $z_{\text{memory}}$ on task-GRU update gate ([NEUROMODULATION_ALGORITHM.md H2](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)) | H2 | $z_{\text{memory}} \downarrow$ (more retention) — only if posterior-over-threat persists |
| $\arg\min_\pi G(\pi)$ under injury-elevated $p(o\mid C)$ | **Injection C.** PPO temperature / Dreamer reward scale ([NEUROMODULATION_ALGORITHM.md H3](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)) | H3 | $T_\pi \downarrow$ in $\mathcal{T}$-rich regions; reward-scale $\downarrow$ in dangerous regions |
| Shared modulator GRU $h_{\text{mod}}$ | "Affective tone" / shared precursor signal | H4 (coordination) | All three channels move *together*, with cross-correlation peaking at lag 0 ± 1 step |

The mapping is exact for A and well-defined for B/C *only under the active-inference reading*. Under a non-AI Bayesian-DL reading (§2 below), Injections B and C are not principled — they would be heuristic add-ons.

---

## 2. Active Inference vs. Heteroscedastic Bayesian DL — What Distinguishes Them in GridWorld

The paper has to choose how strongly to commit to active inference. Here is the basis for that choice.

### 2.1 The two accounts on the same data

**Account 1 (Active Inference).** A single normative principle (free-energy minimisation) generates *all three* injection-site behaviours from one shared latent. Precision shift, prior dominance, and policy shift are coupled because they all derive from $F$ and $G$ over a common generative model.

**Account 2 (Heteroscedastic Bayesian DL — Kendall & Gal 2017; FiLM-Ensemble).** Aleatoric precision is a learned auxiliary head; epistemic precision is ensemble disagreement ([FiLM_ENSEMBLE_SENSORY_PRECISION.md §§3.1, 3.4](../../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md)). The *perception* head is principled. Memory-gate bias and policy-temperature shift, in this account, are *separate* heuristic modulators that happen to share a recurrent state — they are not theoretical consequences of the precision estimator, they are engineering choices.

Both accounts predict H1 (post-injury threat-channel gain). They diverge on H2 + H3 + H4.

### 2.2 The discriminating predictions

| Prediction | Active Inference | Heteroscedastic Bayesian DL |
|---|---|---|
| **H4 cross-correlation** between $\gamma_{\mathcal{T}}$, $z_{\text{memory}}$, $T_\pi$ | **Required:** $r > 0$ at small lag, derivable from shared $F/G$ | Permitted but *not derivable*; observed $r$ is an empirical regularity of the shared GRU, not theory |
| **Sign of $z_{\text{memory}}$ shift** | Theory predicts retention $\uparrow$ *iff* the posterior over threat persists past sensory return-to-baseline (i.e. iff $p(s_t\mid s_{t-1})$ is broad) | No principled prediction; could go either way depending on training |
| **Direction of policy entropy** | Theory predicts $T_\pi \downarrow$ tied to *pragmatic value of injury avoidance*, not to *uncertainty per se* — so $T_\pi$ should drop *most* in regions where threat is *predictable* (high precision + injury-elevated prior) | Predicts $T_\pi$ drops with *epistemic uncertainty* (ensemble disagreement / variance head spike) — i.e. opposite sign in well-modelled threat regions |
| **Effect of removing the heteroscedastic loss** | Modulator can still in principle minimise $F$ via the RL objective alone (slowly), because $F$ is *recoverable* from the survival signal under the right environment; loss is acceleration, not necessity | Without the heteroscedastic loss the precision head is uninformative; this is the dominant effect (cf. v7/v8 null root cause: "no precision training signal", [project_plan.md §4 cause 1](../project_plan.md#4-main-issue-neuromodulation-does-not-outperform-the-baseline)) |
| **Transfer across environment rungs** (cf. postdoc memo gap G-G) | Predicts transfer: same generative model, same precision/prior structure | Does not predict transfer: each rung is a different aleatoric profile, learned fresh |

### 2.3 The empirical test that resolves it

The cleanest single experiment is **a regional contrast**: regions of the grid in which threat is *predictable* (high $\Pi_s$ regime, e.g. predator with stereotyped trajectory) versus regions in which threat is *uncertain* (low $\Pi_s$, novel predator). Active inference predicts $T_\pi \downarrow$ *more* in the predictable-threat region (high pragmatic gradient under accurate posterior). Heteroscedastic BDL predicts $T_\pi \downarrow$ *more* in the uncertain-threat region (epistemic uncertainty drives caution). These are *opposite-sign* predictions. The current canonical task does not separate the two regimes; the postdoc memo's rung 3 (heterogeneous predator threat) does.

**Recommendation for the paper.** Commit to active inference *as the framing for Claim 2(b)* (the clinical-signature claim) and to heteroscedastic Bayesian DL *as the implementation route* (the precision head). They are not competitors at the architecture level — Phase 3's heteroscedastic loss is the practical method by which the AI fixed point is reached. They are competitors only at the *interpretation* level, and only on the discriminating predictions in §2.2. The paper should run the regional contrast (or a close analog) to *resolve* the interpretation, and should be honest in §1.1 that the choice between them is empirical, not theoretical.

---

## 3. The Degenerate Solution (FP-1) — and What Forces Selectivity

The pain-modeling memo flags FP-1 to me as the most important specific hand-off: the modulator can satisfy free-energy minimisation by *flattening* $\Pi_s$ uniformly rather than by *selectively* elevating threat-channel precision. I take this as the central technical problem of the memo.

### 3.1 The degeneracy, in active-inference language

Consider the homogeneous-noise environment used in v7 ([NMN_PERFORMANCE_DIAGNOSIS_v8.md §4](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)) — every channel has the same $\Sigma_i = \sigma^2 I$. The fixed point of channelwise log-precision is then
$$
s_i^{*} = -\log\!\bigl(\mathbb{E}[\varepsilon_{i,t}^2]\bigr)
\;\;\xrightarrow{\;\text{homogeneous noise}\;}\;\;
s_i^{*} = -\log \sigma^2 \quad \forall\,i.
$$
Every channel sits at the *same* precision; the modulator's optimal output is the constant $\gamma_i = \text{const}$, $\beta_i = 0$ — which is **exactly** the v8 observation: "gates collapse to near-identity ... the modulator is effectively bypassed" ([project_plan.md §4 cause 1](../project_plan.md#4-main-issue-neuromodulation-does-not-outperform-the-baseline); [NMN_PERFORMANCE_DIAGNOSIS_v8.md §4 finding 1](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)).

**The v8 null is not a training failure. It is the correct active-inference solution to a degenerate environment.** The modulator did its job. The environment did not give it a problem to solve.

This is the technical content of FP-1: in homogeneous noise, the manifold of free-energy-equivalent precision profiles is one-dimensional (a uniform shift along $s$), and the *selectivity* component — the $\Delta\gamma_{\mathcal{T}\setminus\mathcal{N}}$ that the pain-modeling memo demands — is unidentifiable.

### 3.2 What forces selectivity? Three constraints, each necessary, jointly sufficient

For the modulator to land on a *non-degenerate*, *channel-selective*, *threat-coupled* precision profile, the optimisation problem has to be reshaped on three axes. Each maps onto a current project phase.

#### (a) Heterogeneous noise landscape — Phase 1

$\Sigma_i(s_t)$ must differ across channels *and* be state-dependent. Specifically:
- **Inter-channel reliability contrast.** $\Sigma_{\mathcal{T}}/\Sigma_{\mathcal{N}}$ must be bounded away from 1 across the canonical noise preset, so the *ranking* $s_i^*$ across channels is identifiable. This is exactly Phase 1's "heterogeneous per-modality noise" requirement ([project_plan.md §Phase 1](../project_plan.md#phase-1--reshape-the-noise-landscape-until-the-baseline-bleeds)) — and it is now *re-derived* from active-inference theory, not chosen for engineering reasons.
- **State-dependent (injury-coupled) reliability.** $\Sigma_i$ must be a function of $s_t$, with $\partial \Sigma_{\mathcal{T}}/\partial \text{injury} \neq 0$. Under that condition, the optimal $s_i^*$ becomes *time-varying* in lockstep with injury, and the modulator's job is *dynamic precision estimation*, not static precision recall. This is what the project calls "injury-scaled noise on threat-relevant channels"; AI theory gives a stronger version: **without state-dependent reliability, no architectural change in Phase 3 will produce a non-degenerate modulator**, because there is nothing to track.

**Theoretical re-statement of G1.** G1 ("noise creates headroom") in [project_plan.md §4](../project_plan.md) is currently stated as a baseline-survival drop. The active-inference re-statement is sharper: *G1 holds when, and only when, the active-inference fixed point* $s^*$ *under the canonical preset has a non-trivial channel ranking*, i.e. $\max_i s_i^* - \min_i s_i^* > \tau$ for some $\tau$ identifiable from the empirical channel-wise residual variances of a trained baseline. This is a *measurable property of the environment*, not the agent. The team should verify it on the LayerNorm baseline before Phase 2 — the analysis is just per-channel residual variance from an unmodulated agent's reconstruction error, which Phase 3's precision-head architecture will produce anyway.

#### (b) Heteroscedastic precision loss — Phase 3

Even with heterogeneous noise, the RL objective alone provides only a weak gradient on $s$: the survival signal is many timesteps removed and credit-assignment to "channel $i$ should be down-weighted right now" is buried under the policy gradient. The Kendall–Gal heteroscedastic loss ([FiLM_ENSEMBLE_SENSORY_PRECISION.md §3.1](../../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md), [PRECISION_MODULATION_ARCHITECTURE.md §2.4](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md)) provides exactly the missing gradient: $\partial F_t / \partial s_i$ in §1.2 *is* the heteroscedastic-loss gradient. **Phase 3's auxiliary loss is not "an engineering choice"; it is the gradient computation that the AI fixed point requires.** Without it, the agent might in principle find $s^*$ via the RL signal alone (cf. §2.2 row 4), but in practice it does not — that is the v8 null.

#### (c) C1 control as the construct-validity test

Now the central point. The pain-modeling memo's C1 control replaces the modulator with a trainable EMA of nociception, wired into Injections A/B/C. **In active-inference language, the C1 architecture cannot in principle reach the non-degenerate fixed point of §1.2**, and this is the strongest theoretical statement the paper can make.

The argument:

1. C1's modulator output is a scalar $m_t = \mathrm{EMA}_\tau(o_{\text{noc},t})$ — one number.
2. The selective fixed point requires a *vector* $s^* \in \mathbb{R}^{|\mathcal{T}\cup\mathcal{N}|}$ with non-trivial channel ranking. A scalar broadcast through Injection A's per-modality FiLM cannot produce channel-selective gain (every channel's $\gamma_i$ is a function of the *same* scalar; the ranking is fixed at initialisation).
3. Therefore C1 cannot satisfy the *selectivity* component of free-energy minimisation: the manifold of solutions C1 can reach does not include any non-uniform $\Pi_s$ that depends on channel-level prediction-error statistics.
4. C1 *can* mimic the *temporal envelope* of post-injury hypervigilance (a slowly-rising-and-falling caution signal). It cannot mimic the *cross-channel selectivity* — the $\Delta\gamma \geq 0.2$ between $\mathcal{T}$ and $\mathcal{N}$ in the pain-modeling memo §2.3 FP-1 rule-out.

**This is the theoretical spine of Claim 2 that the postdoc memo asks for.** The statement is: "Pain computation, in active-inference terms, is the operation that maps a state-dependent reliability landscape into a channel-selective precision profile. A slow filter of nociception is not pain because it cannot perform that map; only an architecture that conditions per-channel precision on a learned posterior over the bodily-threat state can." The dissociation modulated-vs-C1 on the *channel-selectivity* metric (and only that metric) is what earns the headline.

### 3.3 What does *not* force selectivity (warnings)

- **Sharing a GRU is not enough.** H4 coordination (shared $h_{\text{mod}}$ driving A/B/C) is necessary for joint signature but says nothing about *selectivity within Injection A*. The shared core can output a uniform $\gamma$ and still be coordinated across A/B/C. Selectivity is a within-A constraint.
- **Multi-seed stability is not enough.** A flat $\gamma$ profile is itself a stable optimum; consistency across seeds confirms convergence to that flat optimum, not non-degeneracy.
- **Phase 1 alone is not enough.** Heterogeneous noise is necessary but not sufficient; without the Phase-3 heteroscedastic gradient, the modulator may still collapse to the homogeneous solution because it never learns to track $\Sigma_i(s_t)$.

The conjunction Phase-1 ∧ Phase-3 ∧ (modulated agent dissociates from C1 on channel-selectivity) is the smallest set under which "active-inference hypervigilance" is a defensible claim. Drop any one and FP-1 is not ruled out.

---

## 4. What Active Inference Predicts for G2 — Pre-Registrable Fingerprint

The postdoc memo §5.3 flags G2's looseness; the pain-modeling memo §2.2 sharpens it to operational thresholds. Active-inference theory does *most* of the work the team needs, *but does not derive every threshold quantitatively* — some of the pain-modeling memo's numbers are calibration choices, not theory. I separate them here so the team knows which to treat as load-bearing and which as tunable.

### 4.1 The four AI-derivable fingerprint properties

| Property | What AI theory derives | Pain-modeling threshold (§2.2) | AI verdict on the threshold |
|---|---|---|---|
| **Sign** | $\gamma_{\mathcal{T}} \uparrow$, $z_{\text{memory}} \downarrow$ (retention), $T_\pi \downarrow$ post-injury | sign-only | **Derived.** Any opposing sign is a falsification of the AI account. |
| **Lag** | All three signals respond to injury through the same $h_{\text{mod}}$, which is updated *one step* after the injury observation; cross-correlation peak at lag $\leq 1$ step + recurrent integration time | lag $\leq 5$ steps | **Mostly derived.** AI predicts peak at lag 0–1 with width set by $\tau_{\text{mod}}$ (modulator-GRU effective timescale). The "5 steps" upper bound is consistent with $\tau_{\text{mod}} \sim 5$, which is an empirical property of the trained modulator, not of theory. Pre-register lag $\leq \min(5, 2\tau_{\text{mod}})$ — strictly tighter than current G2. |
| **Channel selectivity** | $\gamma_{\mathcal{T}} - \gamma_{\mathcal{N}} > 0$ post-injury, with magnitude proportional to the *injury-scaled reliability contrast* $\Delta\Sigma_{\mathcal{T}\setminus\mathcal{N}}$ in the canonical preset | $\Delta\gamma \geq 0.2$ | **Direction derived; magnitude calibrated.** The 0.2 number is a calibration on the noise preset, not theory; theory says $\Delta\gamma \propto \log(\Sigma_\mathcal{N}/\Sigma_\mathcal{T})$ at injury. The team can compute a *theory-predicted* $\Delta\gamma$ from the canonical preset's $\Sigma_i(s_t)$ and pre-register *that*, not 0.2. |
| **Within-trial cross-domain correlation** | $r > 0$ at zero-lag between $\gamma_{\mathcal{T}}$, $z_{\text{memory}}$, $T_\pi$, derived from shared $h_{\text{mod}}$ driving all three | $r \geq 0.3$ at lag $\leq 5$ | **Direction derived; magnitude calibrated.** AI predicts $r$ approaches 1 in the limit of one-dimensional $h_{\text{mod}}$ and 0 in the limit of orthogonal heads. The 0.3 number is a sensible lower bound but is not theory-derived; the theory-derived statement is "$r$ greater than the same statistic computed in the C2 (interoception-masked) control by a margin of $\geq 0.2$". |

### 4.2 The fifth property AI adds (timescale, currently underspecified in G2)

AI theory predicts a *timescale ordering*:
$$
\tau_{\text{precision update}} \;\lesssim\; \tau_{\text{modulator GRU}} \;\lesssim\; \tau_{\text{policy persistence}}.
$$
That is: the precision shift (Injection A) updates first and fastest; memory persistence (Injection B) integrates over the modulator timescale; policy persistence (Injection C) outlasts both, because pragmatic value under an injury-elevated $p(o\mid C)$ persists past the precision/memory shifts. **The chronic-pain analog (H5) is theoretically defensible only if this ordering holds.** The pain-modeling memo's §2.2(d) "policy-shift duration $\geq$ 3× injury-recovery half-life" is consistent with this, but the more theoretically meaningful *pre-registrable* version is the *ordering*: $\tau_A < \tau_B < \tau_C$, with each ratio $\geq 2$. Phase 4's modulator-timescale sweep can test all three at once.

### 4.3 What this means for the paper's pre-registration

**Recommendation for `experiment-designer`** (downstream): pre-register the following five quantities, in order of decreasing theoretical force:

1. **Sign of all three injection-site responses** post-injury — load-bearing; falsification target.
2. **Channel-selectivity ranking** $\gamma_{\mathcal{T}} > \gamma_{\mathcal{N}}$, with the *direction* fixed by theory and the *magnitude* threshold computed from the canonical preset's $\Sigma_i(s_t)$ (not the round 0.2 number).
3. **Cross-domain zero-lag correlation** *contrast* between modulated agent and C2 (interoception-masked control); AI theory predicts a margin, not an absolute level.
4. **Timescale ordering** $\tau_A < \tau_B < \tau_C$ with each ratio $\geq 2$; this is the chronic-pain claim's theoretical version.
5. **Channel-selectivity dissociation** between modulated agent and C1 (slow-input mimic) — the §3.2(c) test, not an AI fingerprint per se but the theoretical-spine test for Claim 2(b).

The looseness flag in postdoc §5.3 is, under this re-statement, mostly resolved: items 1, 2, 4, and 5 are *qualitative theoretical predictions* with falsification meaning, not quantitative cross-correlations open to multiple-comparisons inflation.

---

## 5. Headline-Term Recommendation

The postdoc memo §5.1 and pain-modeling memo §4.4 ask the Bayesian-brain professor to weigh in on whether AI theory licenses "pain computation" as the headline term.

**It does not, at this point in the project.** The AI account distinguishes:

- *Interoceptive precision-weighted perception* — the architectural claim. Defensible from Phase 3 results alone, even if C1 dissociation fails.
- *Active-inference hypervigilance* — the joint-signature claim. Defensible if the four-property fingerprint of §4.1 holds and C1 dissociation passes on channel-selectivity (§3.2c).
- *Pain computation* — the construct claim. Defensible if and only if AI hypervigilance holds *and* the modulator-state representation transfers across environment rungs (gap G-G), i.e. the *generative model itself* generalises rather than the per-rung tuning.

Until C1 dissociation is in hand, the paper is at level 1, possibly level 2. Headline term: **"interoceptive neuromodulation as a substrate for pain-like behaviour"** (concurring with pain-modeling §4.4). The cover letter and abstract should hold this register; the discussion can foreshadow the stronger claim conditional on transfer.

I am specifically *not* recommending the paper invoke "active inference" in the title either. The framing is in the body, where it earns its keep. The title should describe what the agent does, not which theoretical school the authors prefer.

---

## 6. Construct + AI Verdict Together

Combining the pain-modeling memo's verdict and this one:

- **Architecture:** defensible operationalisation of post-injury hypervigilance under the predictive-coding / active-inference account. The three-injection-site, shared-GRU design is what AI theory recommends, and the precision-head proposal in Phase 3 is the right gradient computation.
- **Environment (Phase 1):** *necessary* for non-degeneracy; the heterogeneous-noise landscape is the AI-theoretic precondition for the modulator to have anything to learn. G1's empirical re-statement (§3.2a) is sharper than the current plan's.
- **Training (Phase 3):** *necessary* gradient signal; the heteroscedastic loss is the only practical route to the AI fixed point given the homogeneous-noise degeneracy of the RL loss.
- **Controls (C0, C1, C2 — pain-modeling memo §3):** C1 is the AI-theoretically meaningful dissociation. Without it, "pain computation" is rhetoric. With it (and channel-selectivity-positive), "active-inference hypervigilance" is the right language and "pain-like behavioural substrate" is the right register.
- **Phase 4:** the pre-registrable fingerprint of §4.1 is sharper than current G2 and resolves postdoc §5.3.
- **Headline term:** sober register until C1 dissociates on channel-selectivity. Concurring with pain-modeling §4.4.

---

## 7. Next Steps (Hand-offs)

1. **`professor-rl-bayesian-dl`** (parallel with neuromodulation, both spawned after this memo) — should adjudicate whether the *minimum architectural commitment* (precision head + Phase-3 heteroscedastic loss + C1-comparison wiring) can be implemented without re-tuning per environment rung, which is what the cross-rung generalisation claim of §5 (level 3) requires. Specifically: does the precision head's per-modality grouping in [PRECISION_MODULATION_ARCHITECTURE.md §2.3](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md) generalise across rungs without surgery? If not, level-3 ("pain computation") is unreachable in the time budget.
2. **`professor-neuromodulation`** (parallel with the above) — should rule on whether the §4.2 timescale ordering $\tau_A < \tau_B < \tau_C$ is biologically plausible under a single-GRU modulator, or whether it requires the multi-timescale tonic/phasic split flagged in the postdoc memo §4.2(d).
3. **`experiment-designer`** (downstream) — pre-register the five quantities of §4.3. Compute the theory-predicted $\Delta\gamma$ from the canonical preset's $\Sigma_i(s_t)$, replacing the 0.2 placeholder. Add the regional-contrast experiment (§2.3) to the analysis plan if the paper wants to commit to active inference vs. heteroscedastic BDL.
4. **`senior-developer`** (downstream) — minimal architecture work for C1 (scalar EMA of nociception broadcast through Injections A/B/C) and for logging per-channel residual variance from the LayerNorm baseline (needed for the empirical G1 re-statement of §3.2a). The latter requires no new components, only that the precision-head's reconstruction error be computed and logged for the unmodulated condition as a measurement, not a training signal.
5. **Postdoc** (cross-professor synthesis after RL-BDL and neuromodulation memos land) — the synthesis should resolve the level-1 / level-2 / level-3 framing of §5 against the architectural and biological constraints those two memos surface, and recommend a paper register accordingly.

---

## 8. References (papers anchoring §§1–4)

Papers cited above. Where not already in `docs/project/references/`, they should be added so the AI-side lineage is documented (gap G-E, AI side).

- Bastos, A.M., Usrey, W.M., Adams, R.A., Mangun, G.R., Fries, P., Friston, K.J. (2012). "Canonical microcircuits for predictive coding." *Neuron.* — the canonical-microcircuit reading of precision-weighted message passing; relevant to the Injection A mapping.
- Friston, K. (2005). "A theory of cortical responses." *Phil. Trans. B.* — foundational free-energy formulation.
- Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nat. Rev. Neurosci.* — the synthesis.
- Friston, K. (2023). "Computational psychiatry: from synapses to sentience." *Phil. Trans. B.* — the precision-dynamics equation $\dot{\Pi}_s \propto \beta - \varepsilon_s^2$ used in §1.2.
- Friston, K., Da Costa, L., Hafner, D., Hesp, C., Parr, T. (2021). "Sophisticated Inference." *Neural Computation.* — formal expected-free-energy treatment used in §1.3.
- Kanai, R., Komura, Y., Shipp, S., Friston, K. (2015). "Cerebral hierarchies: predictive processing, precision and the pulvinar." *Phil. Trans. B.* — precision = attention.
- Kendall, A., Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS 2017. — heteroscedastic-loss formulation; the §2.1 alternative account.
- Pouget, A., Beck, J.M., Ma, W.J., Latham, P.E. (2013). "Probabilistic brains: knowns and unknowns." *Nat. Neurosci.* — probabilistic population codes; relevant to the FiLM-Ensemble alternative.
- Rao, R.P.N., Ballard, D.H. (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nat. Neurosci.* — origin of the predictive-coding architecture.
- Seth, A.K., Friston, K.J. (2016). "Active interoceptive inference and the emotional brain." *Phil. Trans. B.* — interoceptive-prior coupling; used in §1.1.
