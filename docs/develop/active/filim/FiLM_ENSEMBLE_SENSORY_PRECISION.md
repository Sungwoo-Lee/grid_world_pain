---
title: FiLM-Ensemble for Sensory Precision Modulation
topic: filim
status: active
created: 2026-03-23
last_updated: 2026-04-12
---

# FiLM-Ensemble for Sensory Precision Modulation

> **Date**: 2026-03-20
> **Author**: Claude (analysis), Sungwoo (direction)
> **Context**: Integrating Active Inference precision weighting with FiLM-Ensemble implicit ensembles for the interoceptive agent
> **Sources**: Turkoglu et al. (2022, NeurIPS), Gorishniy et al. (2025, TabM), Perez et al. (2018), Krueger et al. (2017, Bayesian Hypernetworks), Kendall & Gal (2017), Gawlikowski et al. (2023, Survey), NotebookLM FiLM notebook, PRECISION_MODULATION.md, NMN_PERFORMANCE_DIAGNOSIS_v7/v8
> **Related**: [PRECISION_MODULATION.md](PRECISION_MODULATION.md), [NMN_PERFORMANCE_DIAGNOSIS_v8.md](NMN_PERFORMANCE_DIAGNOSIS_v8.md), [FiLM_PAPERS_REVIEW.md](FiLM_PAPERS_REVIEW.md)

---

## 1. Motivation: Precision Modulation as the Core Research Goal

### 1.1 The Active Inference Framework

Our project's theoretical foundation is **perceptual precision** from Active Inference (see [PRECISION_MODULATION.md](PRECISION_MODULATION.md)). In this framework, precision is the inverse variance (π = 1/σ²) of the likelihood distribution P(o|s). An agent that can modulate its precision — dynamically adjusting how much it trusts each sensory channel — gains a fundamental computational advantage:

- **High precision (↑π, ↓σ²)**: Sensory data is "trusted." Posterior beliefs track observations closely. Good for stable environments, risky under sensory artifacts.
- **Low precision (↓π, ↑σ²)**: Sensory data is "blurred." The agent relies on priors and temporal integration. Biologically observed during injury, high arousal, or rapid movement.

The key insight is that precision modulation is not just noise filtering — it is a **computational strategy** for weighting the contribution of different information sources to the agent's belief state. This is what biological neuromodulation (e.g., dopaminergic, noradrenergic, cholinergic systems) achieves.

### 1.2 Two Kinds of Uncertainty (Kendall & Gal, 2017)

Following the formal framework established by Kendall & Gal (2017), any agent operating in an uncertain world faces two fundamentally different sources of uncertainty:

| Type | Definition | Source | Reducible? | Example in Our Agent |
|------|-----------|--------|------------|---------------------|
| **Aleatoric** | Irreducible noise in the data-generating process | Environment / sensors | No (inherent in the world) | Perceptual noise: olfaction σ=0.15 degrades under injury to σ_eff=0.45 |
| **Epistemic** | Uncertainty from insufficient knowledge/data | Model limitations | Yes (with more data/capacity) | Agent doesn't know what's behind an unexplored wall; policy uncertain in novel states |

Our environment already implements aleatoric uncertainty through the perceptual noise system — state-dependent, injury-scaled Gaussian noise on each sensory modality. What we lack is a principled mechanism for the agent to **estimate and act on** its uncertainty about the world.

A precision-modulating agent needs to handle both:
- **Aleatoric precision**: "My olfaction sensor is unreliable right now because I'm injured" — requires knowing per-channel noise levels
- **Epistemic precision**: "I've never been in this state before, so my value estimates are unreliable" — requires knowing model confidence

### 1.3 What v7/v8 Taught Us

The v7 and v8 experiments tested whether a single-network FiLM modulator could learn sensory precision modulation:

| Experiment | Condition | Finding |
|-----------|-----------|---------|
| v7 | NoNoise | FiLM modulation ≈ unmodulated LN baseline. Modulator learns near-identity gates. LayerNorm alone drives performance. |
| v8 | Noise enabled | FiLM modulation provides zero benefit under noise. Best FiLM (MC g16: 283.44) ties with unmodulated (MC: 283.64). GAE FiLM configs actively hurt (-18 to -52 steps). |

The root cause: a single FiLM layer applies **deterministic γ,β** to features. It has no mechanism to represent *how confident it is* in its own transform. It can learn "always scale olfaction by 0.8" but not "olfaction is unreliable right now — I should be uncertain about features derived from it."

**Crucially, the failure was not in the goal (precision modulation) but in the mechanism (single deterministic FiLM).** The goal remains scientifically valid and central to the interoceptive AI framework. We need a better mechanism.

### 1.4 FiLM-Ensemble: Precision Through Diversity

FiLM-Ensemble (Turkoglu et al., 2022) offers a mechanism that naturally produces uncertainty estimates through **ensemble disagreement**. Instead of one modulator trying to learn the "correct" precision, M modulators each learn a different interpretation of the same sensory input. Where they agree, the agent can be confident; where they disagree, the agent should be cautious.

This maps directly onto the precision framework:

| Active Inference Concept | FiLM-Ensemble Analog |
|--------------------------|---------------------|
| Precision (π = 1/σ²) | Inverse ensemble variance: π ∝ 1/Var(predictions) |
| High precision → trust observations | Low disagreement → act decisively |
| Low precision → rely on priors | High disagreement → be conservative / explore more |
| Precision-weighted prediction error | Disagreement-weighted value/policy update |
| Neuromodulatory gain control | Per-member γ,β creating diverse "neural populations" |

The biological analogy is compelling: rather than a single neuromodulatory signal adjusting all neurons uniformly, FiLM-Ensemble creates **diverse neural sub-populations** (ensemble members) that respond differently to the same stimulus. Population-level agreement/disagreement naturally encodes precision — this is how biological neural populations are thought to represent uncertainty (Pouget et al., 2013, probabilistic population codes).

---

## 2. FiLM-Ensemble: Mechanism Summary

### 2.1 Core Formulation (Turkoglu et al., 2022)

FiLM-Ensemble creates an implicit ensemble where all network weights are shared except for per-member FiLM parameters:

```
FiLM(F_n | γ_n^m, β_n^m) = γ_n^m ∘ F_n + β_n^m     # member m, layer n
```

- **Shared**: All convolutional/linear weights, all normalization statistics
- **Per-member**: Only γ^m and β^m vectors at each normalization layer (~0.1% additional params per member)
- **Aggregation**: Simple average: ŷ = (1/M) Σ y_m

### 2.2 Diversity Through Initialization (ρ Parameter)

The critical innovation is **ρ-controlled Xavier initialization**:

```
γ, β ~ Uniform(±√(3/D_n) · ρ)
```

- `D_n` = feature dimension at layer n
- `ρ` = gain factor controlling initial diversity
- `ρ → 0`: ensemble collapses to single model
- `ρ = 2`: optimal for vision tasks
- `ρ ∈ {4, 8, 16, 32}`: needed for sequential/genomic data

**No explicit diversity loss** — diversity is maintained solely through initialization. Standard cross-entropy loss trains all members jointly. During training, all shared parameters θ plus per-member (γ^m, β^m) are optimized jointly.

### 2.3 Parallelized Forward Pass

All M members run in a single forward pass by replicating inputs along the batch dimension:

```
Input batch [B, D] → replicate → [B×M, D]
Each member's γ^m, β^m applied to its slice
Output [B×M, C] → reshape → [B, M, C] → average → [B, C]
```

This achieves near-zero computational overhead on GPU hardware by exploiting tensor parallelism.

### 2.4 Key Results vs Alternatives

| Method | CIFAR-100 Acc | ECE ↓ | OOD AUROC | Params (M=16) |
|--------|---------------|-------|-----------|---------------|
| Deep Ensemble | **81.6%** | 0.041 | 78.06% | 16× base |
| FiLM-Ensemble | 79.4% | **0.038** | **79.85%** | 1.013× base |
| BatchEnsemble | 77.7% | 0.052 | 75.04% | ~1.01× base |
| MC-Dropout | 75.5% | 0.064 | — | 1× base |

Key achievements:
- **Better calibration** than deep ensembles (ECE 0.038 vs 0.041)
- **Better OOD detection** (AUROC 79.85% vs 78.06%)
- **Higher diversity** (9.2% disagreement vs 6.8% for naive ensembles)
- At **<2% parameter cost** vs 1500% for deep ensembles (M=16)

### 2.5 Critical Limitation: Normalization Layer Dependency

FiLM-Ensemble is implemented by **replacing BatchNorm affine parameters** with per-member γ,β. It inherently requires normalization layers in the base architecture. For architectures without normalization (e.g., simple MLPs for tabular data), Gorishniy et al. (2025) proposed TabM — a BatchEnsemble-based alternative using rank-1 weight perturbations that avoids this dependency.

---

## 3. Related Uncertainty Estimation Literature

Three additional papers provide critical theoretical grounding and practical guidance for our approach.

### 3.1 Kendall & Gal (2017) — Formal Aleatoric + Epistemic Framework

**Paper**: Kendall, A. & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS 2017.

This paper provides the formal framework for combining both uncertainty types in a single model — the exact theoretical structure we need.

**Formal Definitions**:
- **Aleatoric uncertainty**: Noise inherent in the observations (sensor noise, environmental stochasticity). Cannot be reduced with more data. Further divided into *homoscedastic* (constant across inputs) and *heteroscedastic* (varies with input).
- **Epistemic uncertainty**: Model ignorance — uncertainty about which model generated the data. Can be reduced with more training data.

**Key Mechanism — Predicting Variance as a Network Output**:

The network head is split to predict both a mean ŷ and a log-variance ŝ = log σ̂²:

```
[ŷ, ŝ] = f_Ŵ(x)     # single forward pass outputs both prediction and uncertainty
```

The loss function jointly captures both:

```
L(θ) = (1/D) Σ_i [ (1/2) exp(-s_i) ||y_i - ŷ_i||² + (1/2) s_i ]
```

- First term: residual weighted by predicted uncertainty (high σ̂² → low penalty for errors)
- Second term: regularization preventing infinite uncertainty

**Epistemic uncertainty** is captured via MC-Dropout — multiple stochastic forward passes produce a distribution of predictions. The variance of predicted means = epistemic uncertainty; the mean of predicted variances = aleatoric uncertainty.

**Total predictive uncertainty** = Var(predicted means) + Mean(predicted variances) = epistemic + aleatoric.

**Key finding for our work**: In big-data regimes, epistemic uncertainty is largely explained away, making **aleatoric uncertainty the dominant component**. This is relevant because our agent trains on billions of timesteps — epistemic uncertainty may diminish over training, but aleatoric uncertainty (sensor noise) persists forever.

**Direct relevance to our architecture**: We could adapt this by having each FiLM-Ensemble member predict not just action logits + value, but also a **per-member aleatoric variance estimate**. This gives us:
- Aleatoric uncertainty: predicted variance (per observation channel, if we design it that way)
- Epistemic uncertainty: disagreement across ensemble members
- Both available without MC-Dropout overhead — the ensemble provides the stochastic forward passes

### 3.2 Krueger et al. (2017) — Bayesian Hypernetworks

**Paper**: Krueger, D., Huang, C.-W., Islam, R., Turner, R., Lacoste, A., & Courville, A. (2017). *Bayesian Hypernetworks.* arXiv:1710.04759.

This paper uses a hypernetwork to generate **weight distributions** for a target network — a more principled Bayesian version of what FiLM-Ensemble does.

**Core Idea**: A hypernetwork (a Differentiable Directed Generator Network / DDGN) learns to transform simple random noise z ~ N(0, I) into a complex, potentially multimodal approximate posterior distribution over the primary network's parameters. Because the hypernetwork is invertible (using normalizing flows like RealNVP or IAF), it can compute the exact probability density of generated weights, enabling proper variational inference.

**FiLM Connection — Scale-Only Weight Generation**:

Generating full weight matrices would require quadratic scaling. Instead, the BHN uses **weight normalization** and only generates scalar scaling factors (g) for each unit — while directional weights (v) are fixed maximum-likelihood estimates:

```
W = g · v / ||v||    # hypernetwork generates g, v is learned directly
```

This is explicitly FiLM-inspired — the hypernetwork outputs scale parameters (like γ), not full weights. The computational cost scales linearly rather than quadratically.

**Advantages over standard BNNs**: Standard Bayesian NNs (Bayes by Backprop) restrict the posterior to independent factorial Gaussians — unimodal, no correlations between weights. BHNs can represent flexible, multimodal, correlated posteriors.

**Results vs MC-Dropout and Deep Ensembles**:
- On par with MC-Dropout for accuracy
- Substantially **outperforms MC-Dropout** on active learning, anomaly detection, and adversarial example detection — BHNs are much less confident on OOD data
- Avoids the M× cost of deep ensembles by generating an implicit ensemble on-the-fly from noise samples

**Relevance to our work**: The BHN approach suggests a more principled path than fixed per-member FiLM parameters. Instead of M fixed (γ^m, β^m) sets, a small hypernetwork could generate γ, β from noise samples z, creating a *continuous* ensemble rather than a discrete one. This is architecturally close to our existing modulator (which generates γ,β from hidden state) — but conditioned on random noise rather than state, producing diverse weight samples for uncertainty estimation.

**Possible hybrid**: Condition the hypernetwork on **both** random noise z and hidden state h_t:
```
(γ, β) = HyperNet(z, h_t)    # z provides stochasticity for uncertainty, h_t provides state-dependence
```

### 3.3 Gawlikowski et al. (2023) — Survey of Uncertainty in Deep Neural Networks

**Paper**: Gawlikowski, J. et al. (2023). *A Survey of Uncertainty in Deep Neural Networks.* Artificial Intelligence Review.

This comprehensive survey provides the taxonomic context for positioning our approach.

**Taxonomy of Uncertainty Methods** (2 dimensions):

| | Single Network | Multiple Networks |
|-|---------------|-------------------|
| **Deterministic** | Prior Networks, Evidential NNs, gradient metrics | Deep Ensembles, test-time data augmentation |
| **Stochastic** | MC-Dropout, variational inference, MCMC | Bootstrapped ensemble + dropout hybrid |

**Key Findings Relevant to Our Design**:

1. **Ensemble methods remain the gold standard** for uncertainty estimation, but explicit deep ensembles are prohibitively expensive. Implicit ensembles (BatchEnsemble, and by extension FiLM-Ensemble) trade some independence for parameter efficiency — "shared parts mean members are not truly independent."

2. **For RL specifically**, the survey recommends:
   - Bootstrapped model ensembles (training on different data subsets)
   - Dropout sampling for Bayesian inference
   - Hybrid approaches: "mixture of deep Bayesian networks performing dropout sampling on an ensemble of bootstrapped models"

3. **Combining aleatoric + epistemic** is essential: "effectively removing or accounting for model uncertainty allows the remaining predicted data uncertainty to be much more accurate, which directly leads to a better-calibrated model." This directly supports the Kendall & Gal approach of predicting variance + using ensemble disagreement.

4. **The survey does not specifically address small networks or recurrent architectures (GRU/LSTM)** for uncertainty estimation — this is a gap our work could contribute to.

### 3.4 Synthesis: Where FiLM-Ensemble Sits in the Landscape

| Method | Aleatoric | Epistemic | Cost | Our Fit |
|--------|-----------|-----------|------|---------|
| MC-Dropout | No (unless variance head added) | Yes (weight sampling) | Low (T forward passes) | Compatible with our GRU; but underperforms on OOD |
| Deep Ensemble | No (unless variance head added) | Yes (member disagreement) | High (M× full networks) | Too expensive for our RL setting |
| FiLM-Ensemble | No (unless variance head added) | Yes (member disagreement) | **Low** (~0.1% params/member) | **Best fit** — parameter-efficient, parallelizable |
| Bayesian Hypernetwork | Implicitly (weight distribution) | Yes (posterior sampling) | Medium (hypernetwork overhead) | Future extension — more principled but complex |
| Kendall & Gal (variance head) | **Yes** (predicted σ²) | Requires MC-Dropout or ensemble | Minimal (one extra output) | **Combine with FiLM-Ensemble** |
| **Our proposed approach** | **Yes** (variance head, Kendall & Gal style) | **Yes** (FiLM-Ensemble disagreement) | **Low** | Aleatoric via predicted variance + epistemic via ensemble |

**The recommended combination**: FiLM-Ensemble for epistemic uncertainty (member disagreement) + Kendall & Gal-style variance prediction for aleatoric uncertainty. This gives both uncertainty types at minimal computational cost — no MC-Dropout overhead, no M× full network cost.

---

## 4. Adaptation to Our Architecture

### 4.1 Architecture Comparison

| Aspect | FiLM-Ensemble (Original) | Our NMN Agent |
|--------|--------------------------|---------------|
| Architecture | CNN (VGG/ResNet/EfficientNet) | MLP + GRU (recurrent policy) |
| Normalization | BatchNorm | **LayerNorm** |
| Task | Supervised classification | RL (PPO) |
| Loss | Cross-entropy | PPO clipped surrogate + value loss |
| Batch dimension | Standard mini-batch | Vectorized environments (128 envs) |
| Output | Class probabilities | Policy distribution + value estimate |

Three adaptation gaps: **(A)** LayerNorm vs BatchNorm, **(B)** RL/PPO vs supervised, **(C)** precision-as-disagreement integration.

### 4.2 Gap A: LayerNorm Adaptation

**Option 1 — Conditional LayerNorm (Recommended)**:
Replace LayerNorm's learned γ, β with per-member versions:

```python
# Standard LayerNorm: y = γ * (x - μ) / σ + β     (γ, β learned, shared)
# FiLM-Ensemble LN:  y = γ^m * (x - μ) / σ + β^m  (γ^m, β^m per-member)
```

Mathematically identical to the original approach — the normalization statistics (μ, σ) are computed per-sample (LayerNorm) rather than per-batch (BatchNorm), but the affine modulation works the same way. The TabM paper explicitly notes this as a viable adaptation: "for backbones with normalization layers, add per-member affine adapters after normalization."

**Option 2 — Post-Norm FiLM Injection**:
Keep LayerNorm unchanged, add FiLM parameters after it:

```python
x = layer_norm(x)              # Standard LN with shared γ, β
x = γ^m * x + β^m             # Additional per-member FiLM
```

Decouples normalization from ensemble diversity. Perez et al. (2018) showed FiLM does **not** need to be coupled with normalization — it works equally well applied anywhere.

**Option 3 — TabM-style Rank-1 Perturbation (BN-free)**:
Apply per-member rank-1 adapters directly to linear layers:

```python
# For linear layer l(x) = Wx + b, member i:
l_i(x_i) = s_i ⊙ (W(r_i ⊙ x_i)) + b_i
# Parallel: l_BE(X) = ((X ⊙ R) W) ⊙ S + B
```

More expressive (perturbs the weight matrix itself) but adds more parameters than FiLM.

**Recommendation**: Start with **Option 1** (Conditional LayerNorm) — closest to the original, and our architecture already uses LayerNorm at every layer. Escalate to Option 3 if diversity proves insufficient.

### 4.3 Gap B: RL Training Adaptation

**What stays the same**:
- Joint optimization of shared weights + per-member FiLM params
- ρ-initialized diversity
- Parallel forward pass via batch replication

**What changes**:

| Aspect | Supervised | RL (PPO) Adaptation |
|--------|-----------|---------------------|
| Loss | Cross-entropy | PPO clipped surrogate + value loss |
| Aggregation | Average logits | Average **action logits** across members → single policy distribution |
| Value estimate | N/A | Average value estimates; use variance as epistemic uncertainty |
| Exploration | N/A | Ensemble disagreement modulates entropy bonus or conservatism |
| Rollout | N/A | All members see same trajectory; diversity from different γ,β transforms |

**Where to ensemble**:

```
Observations → Shared Encoder (MLP + LN) → Shared GRU → [BRANCH POINT]
                                                              ↓
                                              M ensemble heads (each with own γ^m, β^m)
                                                              ↓
                                              M action logits + M value estimates
                                                              ↓
                                    Average → single policy + single value + disagreement σ²
```

Three options with increasing scope:
1. **Head-only ensemble**: Shared encoder + GRU, per-member γ,β only in actor/critic heads — moderate diversity, fewer parameters
2. **Encoder-only ensemble**: Per-member γ,β in encoder LN layers, shared heads — diverse representations feeding shared decisions
3. **Full-network ensemble**: Per-member γ,β at every LN layer — maximum diversity, most parameters

**Recommendation**: Start with **head-only ensemble** (option 1) to minimize disruption to the recurrent core and establish the baseline behavior.

### 4.4 Gap C: Precision-as-Disagreement — The Core Integration

This is where the Active Inference precision framework meets the FiLM-Ensemble mechanism. The ensemble naturally provides two uncertainty signals:

#### Epistemic Precision from Value Disagreement

```python
# M forward passes produce M value estimates
values = [v_1, v_2, ..., v_M]

# Epistemic precision = inverse variance of value predictions
σ²_epistemic = Var(values)
π_epistemic = 1 / (σ²_epistemic + ε)

# High π_epistemic → members agree → confident state → act decisively
# Low π_epistemic → members disagree → uncertain state → explore / be cautious
```

#### Aleatoric Precision from the Environment

The environment already provides state-dependent noise (see [PRECISION_MODULATION.md](PRECISION_MODULATION.md)):

```
σ_eff = σ_base · (1 + α · Î)    where Î = injury / max_injury
```

At high injury, olfaction noise quadruples (σ_eff = 0.45), visual noise triples (σ_eff = 0.20). This is the aleatoric uncertainty the agent must deal with.

#### Combining Both: Precision-Weighted Decision Making

The agent can use both precision signals:

```python
# Policy: average action logits, but weight by member confidence
action_logits = mean([logits_1, ..., logits_M])

# Value: precision-weighted average
value = mean(values)
value_uncertainty = std(values)  # epistemic uncertainty

# Exploration: modulate entropy coefficient by precision
#   High epistemic uncertainty → increase exploration (lower precision → explore)
#   Low epistemic uncertainty → exploit (higher precision → commit)
effective_entropy_coef = entropy_coef * (1 + λ · value_uncertainty)
```

This creates a natural feedback loop:
1. **Noisy observations** (high aleatoric uncertainty) → ensemble members disagree more on value → high epistemic uncertainty
2. **High epistemic uncertainty** → agent explores more / acts conservatively
3. **In familiar, low-noise states** → members converge → low uncertainty → agent acts decisively

This is precisely the precision-modulated behavior that Active Inference predicts — without hand-coding precision weights.

### 4.5 Relation to Our Existing Modulator Architecture

Our current NMN generates state-dependent γ, β from the GRU hidden state:

```
GRU hidden state → Modulator MLP → (γ, β) → applied to encoder features
```

FiLM-Ensemble's γ, β are **not state-dependent** — they are fixed learned parameters per member:

| | Current NMN Modulator | FiLM-Ensemble |
|-|----------------------|---------------|
| γ, β source | Generated by modulator MLP from state | Fixed learned parameters |
| State-dependent | Yes (different γ,β per timestep) | No (same γ,β always) |
| Purpose | Adaptive filtering (failed) | Member differentiation → precision via disagreement |
| Precision signal | None (implicit, not measurable) | Explicit (ensemble variance) |

**Hybrid possibility — State-Dependent Ensemble**:

```
γ_effective^m = γ_modulator(h_t) + Δγ^m    # state-dependent base + member perturbation
β_effective^m = β_modulator(h_t) + Δβ^m
```

This preserves state-dependent modulation (the modulator adjusts the *base* precision based on internal state) while adding ensemble diversity (per-member perturbations create the *population code* for uncertainty). However, given that the modulator showed zero benefit in v7/v8, the pure ensemble approach is the cleaner first experiment.

---

## 5. Practical Considerations

### 5.1 Parameter Budget

With our architecture (hidden_size=128, LN at each layer):

| Component | Params per Member | M=4 Total | M=8 Total |
|-----------|-------------------|-----------|-----------|
| FiLM per LN layer | 256 (128 γ + 128 β) | 1024 | 2048 |
| 3 LN layers × FiLM | 768 | 3072 | 6144 |
| As % of base model (~50K) | 1.5% | 6.1% | 12.3% |

Compare to current modulator: ~2K–5K params (mod_hidden_size=16, MLP). FiLM-Ensemble with M=4 adds comparable parameters but distributes them across ensemble members rather than a single modulator pathway.

### 5.2 Computational Cost

- **Forward pass**: M× more FiLM operations (negligible — element-wise multiply + add)
- **Batch replication**: Input [128_envs, D] → [128×M, D] — increases memory by M×
- **GRU hidden state**: If ensembling the full network, need M hidden states per env — M× GRU memory
- **Practical M**: Start with M=4 (4× batch expansion). With 128 envs, this becomes 512 effective batch size — well within GPU memory for our small network.

### 5.3 New Diagnostic Metrics

The FiLM-Ensemble approach introduces new metrics that directly measure precision modulation:

| Metric | Definition | What It Tells Us |
|--------|-----------|-----------------|
| `ensemble/value_disagreement` | std(v_1, ..., v_M) | Epistemic uncertainty — does the agent know what's happening? |
| `ensemble/policy_disagreement` | mean KL divergence between member policies | How differently do members interpret the same state? |
| `ensemble/disagreement_vs_injury` | correlation(value_disagreement, injury_level) | Does epistemic uncertainty increase when aleatoric uncertainty increases? **This is the key test.** |
| `ensemble/disagreement_vs_noise` | correlation(value_disagreement, σ_eff) | Direct test: does the ensemble "know" when observations are unreliable? |
| `ensemble/member_γ_diversity` | std of γ values across members per layer | Are members maintaining distinct interpretations? |

The **critical success metric** is `disagreement_vs_injury`: if ensemble disagreement correlates positively with injury level (which drives noise), then the ensemble is implicitly learning precision modulation — the agent "knows" its observations become less reliable when injured, without being told.

### 5.4 Recommended Experimental Design

**Phase 1 — Proof of Concept (Low effort)**:
- M=4, head-only ensemble (per-member γ,β only in actor/critic MLPs after GRU)
- ρ ∈ {1, 2, 4} sweep
- Conditional LayerNorm (Option 1)
- MC return mode only (GAE too sensitive to added complexity per v7/v8)
- NoNoise first → establish that ensemble doesn't hurt baseline
- Then Noise → measure disagreement correlation with injury
- **Key deliverable**: Does value_disagreement correlate with injury_level?

**Phase 2 — Precision-Modulated Exploration (Medium effort)**:
- Use disagreement to modulate entropy_coef (uncertainty-driven exploration)
- Extend γ,β to encoder LN layers
- Test M ∈ {2, 4, 8}
- Compare survival: FiLM-Ensemble vs unmodulated LN baseline under noise
- **Key deliverable**: Does precision-modulated exploration improve survival under noise?

**Phase 3 — Hybrid State-Dependent Ensemble (High effort)**:
- Combine modulator-generated base γ,β with per-member perturbations
- The modulator provides state-dependent precision (responds to injury/internal state)
- The ensemble provides epistemic precision (measures model confidence)
- Compare: pure ensemble vs hybrid vs current NMN
- **Key deliverable**: Does state-dependent base modulation add value on top of ensemble diversity?

### 5.5 Success Criteria

| Metric | Current NMN (v8 best) | Success Threshold | Interpretation |
|--------|----------------------|-------------------|----------------|
| MC Survival (NoNoise) | ~357 (FiLM g1) | ≥ 355 | Ensemble doesn't hurt — first do no harm |
| MC Survival (Noise) | ~284 (Unmod LN) | > 290 | Ensemble-derived precision improves noise robustness |
| Value disagreement–injury corr | N/A | r > 0.3 | Ensemble implicitly tracks aleatoric uncertainty |
| Member diversity (training end) | N/A | γ_std > 0.1 | Members haven't collapsed to identical solutions |

**The most important success criterion** is not survival improvement but **demonstrating that ensemble disagreement functions as a precision signal** — that the agent's epistemic uncertainty naturally covaries with the environment's aleatoric uncertainty. This would validate FiLM-Ensemble as a mechanistic implementation of Active Inference precision weighting.

---

## 6. Risks and Concerns

### 6.1 Known Risks

1. **Diversity collapse during RL training**: FiLM-Ensemble relies solely on initialization for diversity — no explicit regularization. PPO's on-policy updates may push all members toward the same solution faster than supervised training. *Mitigation*: Monitor member γ diversity; if it collapses, add a diversity regularization term (e.g., pairwise KL divergence penalty between member outputs).

2. **GRU hidden state coupling**: If the GRU is shared, all members see identical temporal context. Diversity can only emerge from how they interpret the same hidden state. *Mitigation*: Start with head-only ensemble; escalate to per-member GRU hidden state projections if needed.

3. **Implicit ensembles underperform in RL**: NotebookLM sources indicate that "popular efficient ensemble approaches fail to deliver performance comparable to naive deep ensembles in offline RL" (Nikulin et al., 2023). However, this finding is for *offline* RL with Q-functions; our on-policy PPO setting with small networks may behave differently.

4. **Aleatoric–epistemic conflation**: Ensemble disagreement captures epistemic uncertainty, but we want it to also reflect aleatoric uncertainty (noise). This only works if noisy inputs genuinely cause model disagreement — which depends on whether the ensemble members have learned sufficiently different feature extractors. If members converge, noisy and clean inputs produce identical disagreement.

### 6.2 Why It Might Work Despite These Concerns

- Our network is **much smaller** (~50K params) than typical RL benchmarks (256×4 MLPs in D4RL). Small networks may benefit more from ensemble diversity — each member can specialize on a genuinely different feature subspace.
- PPO is **on-policy** — every update uses fresh data, reducing the distribution shift that plagues offline RL ensembles.
- **128 parallel environments** provide a rich batch for ensemble training — effective batch 128×M.
- The primary value is **scientific** — demonstrating precision-as-disagreement as a computational principle — even if survival improvement is modest.

---

## 7. Beyond FiLM-Ensemble: Alternative Paths to Precision Modulation

### 7.1 The Deeper Diagnosis: A Missing Training Signal

The v7/v8 finding — that single FiLM learns near-identity gates — is typically attributed to "single deterministic γ,β cannot represent uncertainty" (Section 1.3). But this explanation is incomplete. The deeper issue is:

**The agent has no training signal that rewards precision estimation.**

The PPO loss cares only whether the agent took the right action. It does not penalize the agent for being miscalibrated — for failing to know *which sensory channels are unreliable*. If the agent can learn a reasonable policy despite noise (v8 shows ~283 steps with or without FiLM), there is no gradient pressure to learn precision modulation at all. The identity gate is not a failure of representation — it is the *optimal solution* when no objective rewards non-identity gates.

FiLM-Ensemble addresses this indirectly: diverse members create natural disagreement signals. But it doesn't create an explicit *objective* for precision learning. And in our specific RL setting — 128 parallel environments, same on-policy data for all members, PPO's clipped loss — the gradient pressure toward member convergence is strong. Diversity collapse (Section 6.1) is not just a risk; it is the expected equilibrium.

This motivates exploring approaches that directly create a training signal for precision, rather than relying on ensemble diversity as a proxy.

### 7.2 Path 1: Heteroscedastic FiLM (Stochastic Modulation Parameters)

The most natural extension of the current modulator. Instead of predicting deterministic γ,β, the modulator predicts a **distribution** over γ,β:

```
h_t → Modulator MLP → (μ_γ, log σ²_γ, μ_β, log σ²_β)

γ_t ~ N(μ_γ, σ²_γ)
β_t ~ N(μ_β, σ²_β)
```

During each forward pass, γ and β are *sampled* from the predicted distribution. The reparameterization trick (Kingma & Welling, 2014) enables gradient flow: γ = μ_γ + σ_γ · ε, where ε ~ N(0,1).

**Why σ²_γ IS a precision signal**: When the modulator is confident about how to transform features (familiar, low-noise state), σ²_γ is small → samples cluster → near-deterministic behavior. When the input state is ambiguous or noisy, σ²_γ is large → samples spread → stochastic behavior.

**Why this addresses the identity gate problem**: Even if μ_γ → 1 (identity transform), σ²_γ can still be informative. The modulator can learn "I don't know how to transform this" (large σ²_γ) independently of "the best transform is identity" (μ_γ ≈ 1). These are orthogonal degrees of freedom that single deterministic FiLM conflates.

**Comparison to FiLM-Ensemble**:

| Aspect | FiLM-Ensemble | Heteroscedastic FiLM |
|--------|--------------|---------------------|
| Diversity source | M fixed (γ^m, β^m) sets | Continuous sampling from predicted distribution |
| State-dependent | No (same γ^m always) | Yes (μ_γ, σ²_γ change with h_t) |
| Uncertainty signal | Discrete (M member disagreement) | Continuous (predicted variance σ²_γ) |
| Batch overhead | M× replication | None (single sample per forward pass; T samples for MC estimate) |
| Diversity collapse risk | High (shared loss, same data) | N/A (variance is explicitly parameterized) |
| Training signal for uncertainty | Indirect (from initialization diversity) | **Still indirect** — PPO alone won't train σ²_γ meaningfully |

**Critical limitation**: Like single FiLM, heteroscedastic FiLM still lacks an explicit objective for the variance parameters. The PPO loss provides no gradient signal to σ²_γ unless the stochasticity in γ directly affects policy performance. Without an auxiliary loss, σ²_γ may collapse to zero (deterministic, recovering vanilla FiLM) or remain at initialization. This motivates Path 2.

### 7.3 Path 2: Precision-Weighted Prediction Error — The Active Inference Path (Recommended)

This is the most theoretically principled approach. In Active Inference, precision is not estimated from ensemble disagreement or variance parameters — it is a **model parameter** inferred jointly with the hidden state, driven by a **prediction error objective**. The agent maintains an internal model of its sensory dynamics and uses prediction errors to estimate precision.

#### Core Mechanism

Add an auxiliary prediction head that forecasts the next observation and learns per-channel precision:

```
PredictionHead(h_t, a_t) → ô_{t+1}, s_i = log π̂_i     # predict next obs + per-channel log-precision
```

Train with the Kendall & Gal (2017) heteroscedastic loss:

```
L_precision = Σ_i [ (1/2) exp(s_i) · (o_{t+1,i} - ô_{t+1,i})² - (1/2) s_i ]

where s_i = log π̂_i (learned log-precision per observation channel i)
```

- **First term**: Precision-weighted prediction error. If π̂_i (precision) is high, errors are penalized heavily — the model claims this channel is reliable, so errors matter. If π̂_i is low, errors are attenuated — "I predicted this channel would be noisy."
- **Second term**: Regularizer preventing the model from setting all precisions to zero (trivially ignoring everything). This creates pressure toward *accurate* precision estimation, not just low precision.

#### Why This Creates the Missing Training Signal

When olfaction is noisy (high injury, σ_eff = 0.45):
1. Prediction errors for olfaction become **large and irreducible** — no amount of modeling can predict the noise realization
2. The loss function penalizes high π̂_olfaction under large errors (first term) but also penalizes low π̂_olfaction directly (second term)
3. The equilibrium: π̂_olfaction settles at a value reflecting the *actual* noise level — the model learns the precision

When location sensing is clean (σ = 0.01):
1. Prediction errors for location are small and predictable
2. High π̂_location is rewarded (low errors × high precision = small loss)
3. The model learns to trust location

**This is literally precision estimation from prediction errors — the core computation of Active Inference, now with an explicit gradient signal.**

#### Connection to FiLM: Precision-Gated Feature Modulation

The learned precision π̂_i connects to FiLM through feature gating:

```python
# Modulator generates FiLM parameters (as current architecture)
γ_i, β_i = Modulator(h_t)

# Precision head generates per-channel precision from prediction errors
π̂_i = PrecisionHead(h_t, a_t)  # trained by L_precision
π_normalized = sigmoid(log π̂_i)  # squash to [0, 1] for gating

# Precision-gated FiLM: high precision → full modulation; low precision → rely on prior
feature_out = π_normalized · (γ_i · feature_i + β_i) + (1 - π_normalized) · feature_i
```

Interpretation:
- **High precision (π̂_i large)**: The agent trusts this channel → FiLM transform has full effect → features are actively modulated
- **Low precision (π̂_i small)**: The agent distrusts this channel → FiLM transform is suppressed → features pass through unchanged (relying on temporal priors from GRU hidden state)

This is the Active Inference analog: precision gates the influence of sensory prediction errors on belief updates. Here, precision gates the influence of sensory-derived features on downstream policy computation.

#### Architecture

```
Observation o_t ─────────────────────────────→ Encoder (MLP + LN) → features
                                                          ↓
                                              GRU(features, h_{t-1}) → h_t
                                                          ↓
                                               ┌─────────┼─────────────┐
                                               ↓         ↓             ↓
                                          Modulator   Actor/Critic  PredictionHead
                                          (γ_i, β_i)  (logits, V)  (ô_{t+1}, log π̂_i)
                                               ↓                       ↓
                                     Precision-gated FiLM         L_precision
                                     applied to encoder            (auxiliary loss)
                                     features for NEXT step
```

**Important architectural note**: The precision-gated FiLM modulates *encoder features*, not actor/critic outputs. This means precision affects how raw sensory information is represented before being integrated temporally (GRU) or used for decisions (actor/critic). This matches the biological picture: precision weighting occurs at the *sensory processing* stage, not the *decision* stage.

#### Prediction Target Design

The prediction head forecasts the *next full observation vector* o_{t+1}. Per-channel precision is defined at the **modality level**, not the individual observation element level:

| Modality | Obs Dims | Precision Params | Rationale |
|----------|----------|-----------------|-----------|
| Injury | 1 | 1 π̂ | Single scalar |
| Nutrition | 1 | 1 π̂ | Single scalar |
| Satiation | 1 | 1 π̂ | Single scalar |
| Extero Nociception | 1 | 1 π̂ | Single scalar |
| Olfaction | 5 | 1 π̂ (shared) | All 5 gradient dims share same noise regime |
| Collision | 1 | 1 π̂ | Binary signal |
| Proprioception | 5 | 1 π̂ (shared) | Action encoding shares noise regime |
| Visual | variable | 1 π̂ (shared) | All visual features share injury-dependent noise |
| Location | 2 | 1 π̂ (shared) | x,y share noise regime |

Total: **9 precision parameters** — one per modality, matching the environment's per-modality noise configuration. This is a deliberate alignment: the agent's precision estimation operates at the same granularity as the environment's noise generation.

#### Total Loss

```
L_total = L_PPO + λ_pred · L_precision

where:
  L_PPO = standard PPO clipped surrogate + value loss + entropy bonus
  L_precision = heteroscedastic prediction loss (Kendall & Gal)
  λ_pred = auxiliary loss coefficient (hyperparameter, start with 0.1–1.0)
```

The λ_pred coefficient balances RL performance against precision estimation quality. Too high → agent over-invests in prediction at the cost of policy quality. Too low → precision signal is too weak to drive FiLM gating. This requires tuning but is a single scalar — far simpler than tuning ensemble hyperparameters (M, ρ, diversity regularization).

### 7.4 Path 3: FiLM-Ensemble (As Documented in Sections 2–6)

FiLM-Ensemble remains a valid approach for **epistemic** precision — uncertainty arising from the model's limitations rather than from sensor noise. The mechanism (M implicit members, disagreement as precision) is well-characterized in Sections 2–6.

However, two clarifications are warranted given the deeper diagnosis:

1. **Ensemble disagreement captures epistemic uncertainty, not aleatoric uncertainty directly.** For ensemble disagreement to correlate with injury-driven noise (the key success metric in Section 5.5), noisy inputs must cause *different members to disagree more*. This only holds if members have learned sufficiently different feature extractors — which is precisely what diversity collapse threatens.

2. **In the big-data regime of RL training (billions of timesteps), epistemic uncertainty is largely explained away** (Kendall & Gal, 2017, Section 3.1). The dominant uncertainty at convergence is aleatoric — exactly the type that ensemble disagreement does not directly capture. Path 2's prediction-error-based precision is better suited for the uncertainty type that persists at scale.

FiLM-Ensemble's best role is as a **complement** to Path 2, not a replacement. The ensemble provides epistemic precision (useful early in training, in novel states); the prediction head provides aleatoric precision (useful throughout training, in noisy states).

### 7.5 Path 4: Stochastic FiLM + Prediction Error (The Full Hybrid)

The strongest theoretical approach combines Paths 1, 2, and optionally 3:

```
h_t → Modulator MLP → (μ_γ, log σ²_γ, μ_β, log σ²_β)    # stochastic FiLM (Path 1)
h_t, a_t → PredictionHead → ô_{t+1}, log π̂_i              # precision from prediction (Path 2)

γ_t = μ_γ + σ_γ · ε,  ε ~ N(0,1)                          # reparameterized sampling
β_t = μ_β + σ_β · ε'

# Precision gates the stochastic FiLM
feature_out = π_normalized · (γ_t · feature + β_t) + (1 - π_normalized) · feature

L_total = L_PPO + λ_pred · L_precision + λ_KL · KL(q(γ,β|h_t) || p(γ,β))
```

The KL term (optional) regularizes the stochastic FiLM toward a prior — preventing the variance from collapsing to zero while also preventing it from exploding. This is standard variational inference.

**What each component provides**:
- **Prediction head (π̂_i)**: Aleatoric precision — "this channel is noisy right now"
- **Stochastic FiLM (σ²_γ)**: Modulatory confidence — "I'm not sure how to transform this feature"
- **FiLM-Ensemble (optional M members)**: Epistemic precision — "different interpretations of this state disagree"

This is the most complete implementation of precision modulation, but also the most complex. It should be a later-phase experiment after validating the simpler components individually.

### 7.6 Comparison of Paths

| Dimension | Path 1: Heteroscedastic FiLM | Path 2: Prediction Precision | Path 3: FiLM-Ensemble | Path 4: Full Hybrid |
|-----------|------------------------------|------------------------------|----------------------|---------------------|
| **Uncertainty type** | Modulatory confidence | **Aleatoric** (per-channel) | Epistemic (model-level) | Both + modulatory |
| **Training signal** | Indirect (PPO only) | **Explicit** (prediction loss) | Indirect (diversity init) | Explicit + regularized |
| **State-dependent** | Yes | Yes | No (fixed γ^m) | Yes |
| **Batch overhead** | None | None | M× replication | Depends on ensemble inclusion |
| **New parameters** | 2× modulator output | Prediction MLP + 9 precision params | M × (γ,β per LN layer) | All of the above |
| **Collapse risk** | σ²_γ → 0 (without aux loss) | Low (prediction loss drives π̂) | High (shared PPO loss) | Low (multiple objectives) |
| **Active Inference alignment** | Partial (stochastic gain) | **Direct** (precision-weighted PE) | Indirect (population code) | Most complete |
| **Implementation complexity** | Low | **Medium** | Medium | High |
| **Recommended phase** | Skip (subsumed by Path 4) | **Phase 1** | Phase 2 (complement) | Phase 3 |

### 7.7 Recommendation and Experimental Design

**Start with Path 2 alone** (precision-weighted prediction error + FiLM gating). Rationale:

1. **It creates the training signal that v7/v8 were missing.** The prediction error gives the modulator a *reason* to learn non-identity gates. Under noise, prediction errors for noisy channels become large and irreducible → the model learns to downweight those channels via low precision.

2. **It is the most direct implementation of Active Inference precision.** Precision is computed from prediction errors — exactly as the theory prescribes — not approximated through ensemble disagreement. This makes the scientific narrative cleaner and more publishable.

3. **It is architecturally minimal.** One additional MLP head for observation prediction + 9 precision parameters. No batch replication, no M× memory, no diversity collapse monitoring.

4. **The success criterion is clean and directly measurable.** If learned π̂_olfaction drops when injury increases (because olfaction prediction errors become irreducible), precision modulation is demonstrated — an unambiguous, interpretable signal. Compare this to ensemble disagreement correlation, which conflates aleatoric and epistemic effects.

5. **It composes well.** FiLM-Ensemble (Path 3) can be layered on top later for epistemic precision. Stochastic FiLM (Path 1) can be added for modulatory confidence. Each addition is incremental and independently testable.

#### Recommended Experimental Plan

**Phase 1 — Precision from Prediction Error (Path 2)**:
- Add prediction head: MLP (hidden_size → hidden_size → obs_dim + 9 precision params)
- Precision-gated FiLM applied to encoder features
- λ_pred ∈ {0.1, 0.5, 1.0} sweep
- MC return mode only (GAE too sensitive per v7/v8)
- NoNoise first → establish baseline (prediction loss converges, precision uniform)
- Then Noise → measure per-channel precision vs injury correlation
- **Key deliverable**: Does π̂_olfaction decrease (and π̂_location remain stable) as injury increases?

**Phase 2 — Add FiLM-Ensemble for Epistemic Precision (Path 3)**:
- Keep prediction head from Phase 1
- Add M=4 head-only ensemble with Conditional LN
- Now have: aleatoric precision (π̂_i from prediction) + epistemic precision (ensemble disagreement)
- Compare: Path 2 alone vs Path 2+3
- **Key deliverable**: Does ensemble disagreement provide additional survival benefit beyond prediction-derived precision?

**Phase 3 — Full Hybrid (Path 4)**:
- Make FiLM parameters stochastic (heteroscedastic modulator)
- Add KL regularization
- Compare: Path 2 vs Path 2+3 vs Path 4
- **Key deliverable**: Does stochastic modulation + prediction precision outperform either alone?

#### Ablation Design for Phase 1

| Condition | Prediction Head | Precision Gating | FiLM Modulator | Expected Outcome |
|-----------|----------------|-----------------|----------------|------------------|
| A: Baseline (LN, no mod) | No | No | No | ~284 survival under noise (v8 replication) |
| B: FiLM only (v8 control) | No | No | Yes (deterministic) | ~284 survival (v8 replication: identity gates) |
| C: Prediction only | Yes | No | No | L_precision converges; π̂ correlates with noise; no survival benefit (no gating mechanism) |
| D: **Prediction + Gated FiLM** | Yes | Yes | Yes | π̂ correlates with noise AND survival > 284 (precision gates noisy channels) |

The critical comparison is **C vs D**: both learn precision, but only D *uses* it. If D > C in survival, precision gating has functional value. If D ≈ C, precision is learned but not useful — suggesting the gating mechanism needs refinement.

The secondary comparison is **B vs D**: both have FiLM, but only D has the prediction loss driving non-identity gates. If D > B, the prediction loss is the key ingredient that v7/v8 were missing.

#### Success Criteria (Updated)

| Metric | Current Best (v8) | Path 2 Success Threshold | Interpretation |
|--------|-------------------|--------------------------|----------------|
| MC Survival (NoNoise) | ~357 | ≥ 355 | Prediction loss doesn't hurt baseline |
| MC Survival (Noise) | ~284 | **> 290** | Precision gating improves noise robustness |
| π̂_olfaction at Î=0 vs Î=1 | N/A | Significant decrease (p < 0.05) | Agent learns olfaction unreliable under injury |
| π̂_location at Î=0 vs Î=1 | N/A | No significant change | Agent correctly identifies location as injury-invariant |
| L_precision convergence | N/A | Decreasing over training | Prediction model is learning |
| Prediction error vs π̂ correlation | N/A | r > 0.5 per channel | Precision tracks actual prediction difficulty |

**The most important criterion** is the per-channel precision profile under injury: if the learned π̂ vector mirrors the environment's noise configuration (high precision for low-noise channels, low precision for injury-scaled channels), precision modulation is demonstrated at the mechanistic level — regardless of survival improvement.

---

## 8. Path 5: DreamerV3 as Native Precision Substrate

### 8.1 The Key Realization

Path 2 (Section 7.3) proposes adding an **auxiliary** prediction head to PPO — a separate MLP that predicts o_{t+1} and learns per-channel precision from heteroscedastic loss. But we already have an architecture whose *entire purpose* is predicting observations from latent states: **DreamerV3's world model**.

DreamerV3's RSSM + decoder already trains via reconstruction loss:

```
L_world = L_reconstruction + L_reward + L_continue + L_KL

where L_reconstruction = -log p(o_t | z_t, h_t)    # observation prediction from latent state
```

This reconstruction loss IS the prediction error that Path 2 bolts on as an auxiliary objective. In DreamerV3, it is the **core** training signal — not competing with PPO for gradient bandwidth, but the primary world model objective. The implication is fundamental: **DreamerV3 natively provides the prediction-error substrate for precision learning. Path 2's auxiliary head is a lightweight approximation of what a world model does as its main job.**

### 8.2 Our DreamerV3 Implementation

The local codebase contains a high-fidelity JAX/NNX DreamerV3 implementation (see [DREAMER_IMPLEMENTATION_AUDIT.md](DREAMER_IMPLEMENTATION_AUDIT.md), [DREAMER_REVIEW.md](DREAMER_REVIEW.md)). Key components relevant to precision modulation:

| Component | Implementation | Relevance to Precision |
|-----------|---------------|----------------------|
| **RSSM** | GRU + categorical latent (32×32), ST-gradient | Deterministic h_t carries temporal context; stochastic z_t captures sensory surprise |
| **Encoder** | MLP processing 33-dim multimodal observation vector | Entry point for per-channel precision gating |
| **Decoder** | MLP reconstructing observations from latent state | Currently deterministic — **make heteroscedastic** for precision |
| **Symlog two-hot** | 255-bin discretization in symlog space | Scale-invariant reconstruction; precision must operate in same space |
| **KL balancing** | 0.5/0.1 prior/posterior with 1-nat free bits | Prior–posterior divergence already captures a form of latent surprise |
| **Imagination** | 15-step latent rollout via `jax.lax.scan` | Actor-critic trains in latent space — precision must propagate through latent representation |
| **Vectorized envs** | 128 parallel environments, GPU-resident replay buffer | Same batch structure as PPO experiments |

**Audit status** (from [DREAMER_IMPLEMENTATION_AUDIT.md](DREAMER_IMPLEMENTATION_AUDIT.md)): Rated **S-Tier** with exact parity on RSSM bottleneck, symlog, two-hot, KL balancing, advantage normalization, and action space. Two minor gaps: ELU/SiLU inconsistency in RSSM transitions, and standard (not Hafner-constant) weight initialization.

### 8.3 Heteroscedastic World Model: Precision-Aware Reconstruction

The core modification: make the observation decoder **heteroscedastic** — predict both the reconstruction AND per-channel precision:

```
Current decoder:     ô_t = Decoder(z_t, h_t)                    # deterministic reconstruction
Proposed decoder:    ô_t, s_i = Decoder(z_t, h_t)               # reconstruction + log-precision per modality
                     where s_i = log π̂_i for modality i
```

The reconstruction loss becomes the Kendall & Gal heteroscedastic loss:

```
L_reconstruction = Σ_i [ (1/2) exp(s_i) · ||o_{t,i} - ô_{t,i}||² - (1/2) s_i ]
```

**What changes vs. standard DreamerV3**:
- The decoder outputs 9 additional parameters (one log-precision per modality)
- The reconstruction loss is precision-weighted rather than uniform
- All other world model components (RSSM, KL, reward head, continue head) remain unchanged

**What changes vs. Path 2 (PPO + auxiliary head)**:
- No auxiliary head needed — precision emerges from the world model's core training objective
- No λ_pred hyperparameter — the reconstruction loss is already part of L_world with unit weight
- Precision is learned in the **generative model** (world model), not the **discriminative model** (policy) — this is the theoretically correct placement in the Active Inference framework
- The latent state (z_t, h_t) can encode precision information for downstream actor-critic use

### 8.4 Precision Flow Through the Architecture

In DreamerV3, the actor-critic trains entirely in **latent imagination** — it never sees raw observations. This raises the question: how does learned precision reach the policy?

```
                          WORLD MODEL (trained on real data)
                          ┌─────────────────────────────────────────────┐
Observation o_t ─→ Encoder ─→ Posterior z_t ─→ RSSM h_t ─→ Decoder ─→ ô_t, π̂_i
                          │                         │            (heteroscedastic)
                          │                         │
                          │    IMAGINATION (latent rollout, no observations)
                          │    ┌────────────────────────────────────────┐
                          │    │  Prior ẑ_{t+k} ─→ RSSM h_{t+k} ─→ Actor → a_{t+k}
                          │    │                          │          Critic → V_{t+k}
                          │    │                          ↓
                          │    │                   Reward head → r̂_{t+k}
                          │    │                   Continue head → ĉ_{t+k}
                          │    │                   **Precision head → π̂_{t+k}**
                          │    └────────────────────────────────────────┘
                          └─────────────────────────────────────────────┘
```

Three mechanisms for precision to reach the actor-critic:

**Mechanism A — Latent Encoding (Implicit)**:
The RSSM hidden state h_t and stochastic state z_t are trained to support accurate heteroscedastic reconstruction. To predict *both* ô_t and π̂_i, the latent state must encode information about observation reliability. This information is then available to the actor-critic, which operates on the same latent features `feat = concat(h_t, z_t)`.

This is implicit — precision is "baked into" the latent representation without the actor explicitly receiving a precision signal. The actor may learn to use this information or may ignore it.

**Mechanism B — Precision as Imagined Signal (Explicit)**:
During imagination, query the precision head alongside the reward and continue heads:

```python
def scan_imag(prev_state, key):
    feat = wm.get_feat(prev_state)
    action = actor(feat).sample(key)
    prior = wm.rssm.imagine_step(prev_state, action, key)

    reward = from_twohot(wm.reward_head(get_feat(prior)))
    cont = sigmoid(wm.continue_head(get_feat(prior)))
    precision = wm.precision_head(get_feat(prior))    # NEW: imagined precision

    return prior, {'reward': reward, 'continue': cont, 'precision': precision}
```

The imagined precision can then modulate the actor-critic loss or exploration:

```python
# Precision-weighted value targets
effective_reward = reward * mean(precision)  # downweight rewards from imprecise states

# Precision-modulated exploration
entropy_coef_effective = entropy_coef * (1 + λ * (1 - mean(precision)))
# Low precision → more exploration; high precision → exploit
```

**Mechanism C — Precision-Gated FiLM in the Encoder (Direct)**:
Apply FiLM with precision gating at the encoder level, exactly as in Path 2 (Section 7.3):

```python
# During world model training (real observations):
features = encoder(o_t)
γ_i, β_i = modulator(h_t)
π̂_i = precision_from_decoder(z_t, h_t)  # learned by heteroscedastic reconstruction
π_gate = sigmoid(log π̂_i)

features_gated = π_gate * (γ_i * features + β_i) + (1 - π_gate) * features
```

During imagination, the encoder is not used (no real observations). But the precision signal propagates through the latent state (Mechanism A) and can be queried explicitly (Mechanism B).

**Recommendation**: Start with **Mechanism A** (implicit, zero additional complexity) and measure whether the latent state captures precision information by probing. If insufficient, add **Mechanism B** (explicit precision in imagination). Reserve **Mechanism C** for Phase 2 if direct feature gating is needed.

### 8.5 Why DreamerV3 Resolves the Core Concerns

| Concern (from Sections 6–7) | PPO + Path 2 | DreamerV3 + Heteroscedastic Decoder |
|------------------------------|-------------|-------------------------------------|
| **Missing training signal** (7.1) | Addressed via auxiliary loss — but competes with PPO for gradients | **Resolved natively** — reconstruction IS the main world model objective |
| **λ_pred tuning** (7.7) | Critical hyperparameter; too high hurts policy | **Eliminated** — reconstruction loss uses unit weight (DreamerV3 design principle) |
| **Prediction target design** | Must design prediction head architecture | **Already exists** — decoder head is proven; add 9 precision outputs |
| **Prediction ≠ policy objective** | Auxiliary loss may not help policy | World model trains representations that actor-critic uses — precision in latent state directly affects policy quality |
| **Temporal coherence** | Single-step prediction; no recurrent context for precision | RSSM's h_t provides full temporal context — precision can reflect injury *trajectory*, not just current state |
| **Scale invariance** | Must handle observation scales manually | Symlog two-hot handles arbitrary scales; precision operates in normalized space |

### 8.6 Connection to Active Inference: The Generative Model

This approach achieves the closest alignment with Active Inference of any path considered:

| Active Inference Concept | DreamerV3 + Heteroscedastic Decoder |
|--------------------------|-------------------------------------|
| **Generative model** P(o,s) | RSSM world model: P(o_t, z_t, h_t) |
| **Variational inference** q(s\|o) | Encoder + posterior: q(z_t \| h_t, o_t) |
| **Precision** π = 1/σ² | Learned log-precision s_i per modality in decoder |
| **Free energy** F = E_q[-log P(o,s)] + KL[q\|\|p] | L_world = L_reconstruction + L_KL (+ L_reward, L_continue) |
| **Precision-weighted prediction error** π·(o - ô)² | Heteroscedastic loss: exp(s_i)·\|\|o_i - ô_i\|\|² |
| **Active inference agent** minimizes F by acting | Actor trained in imagination to maximize (precision-weighted) reward |
| **Precision on beliefs** (epistemic) | KL divergence between prior and posterior reflects latent surprise; ensemble disagreement (optional) |
| **Precision on observations** (aleatoric) | **Directly learned** by heteroscedastic decoder — per-channel, state-dependent |

In Active Inference, the **generative model** is where precision lives — it parameterizes the agent's beliefs about the reliability of its own sensory predictions. DreamerV3's world model IS the generative model. Making the decoder heteroscedastic is not a hack or auxiliary objective — it is the theoretically correct placement of precision estimation within the agent's architecture.

By contrast, Path 2 (PPO + auxiliary prediction) places precision in a **discriminative** framework — a bolt-on to a policy optimizer that has no generative model. It works, but it is an approximation of what a world model provides natively.

### 8.7 DreamerV3 World Model Ensemble: Epistemic Precision via Plan2Explore

DreamerV3 also has a natural path to epistemic precision through **world model ensembles**, as documented in Plan2Explore (Section 8.2 of [DREAMER_REVIEW.md](DREAMER_REVIEW.md)):

```
r_intrinsic = Variance({p_θ^k(z_t | h_t, a_{t-1})})_{k=1}^K
```

An ensemble of K RSSM transition models, each predicting different prior distributions, captures epistemic uncertainty in the **dynamics** — the agent's uncertainty about how the world will evolve. This is more meaningful than FiLM-Ensemble's disagreement on value estimates, because it measures uncertainty about the *world model itself*, not just the policy's interpretation.

**Combined precision architecture**:
- **Aleatoric precision**: Heteroscedastic decoder (per-channel, learned from reconstruction error)
- **Epistemic precision**: World model ensemble disagreement (per-state, from transition model variance)
- Both available during imagination, enabling precision-modulated policy learning

This is exactly the Kendall & Gal decomposition (aleatoric + epistemic) but implemented within the generative model framework rather than as discriminative add-ons.

### 8.8 Practical Considerations for DreamerV3 Integration

#### Implementation Effort

| Change | Effort | Files Affected |
|--------|--------|----------------|
| Heteroscedastic decoder (add 9 precision outputs) | **Low** | `dreamer_v3_nnx.py` (decoder head), `dreamer_v3_trainer.py` (loss) |
| Kendall & Gal reconstruction loss | **Low** | `dreamer_v3_trainer.py` (replace uniform reconstruction loss) |
| Precision logging (π̂ per modality per step) | **Low** | `dreamer_v3_trainer.py` (add to metrics) |
| Precision as imagined signal (Mechanism B) | **Medium** | `dreamer_v3_trainer.py` (imagination loop) |
| Precision-gated FiLM in encoder (Mechanism C) | **Medium** | `dreamer_v3_nnx.py` (encoder), new modulator module |
| World model ensemble for epistemic precision | **High** | `dreamer_v3_nnx.py` (K RSSM copies), `dreamer_v3_trainer.py` (training loop) |

#### Symlog Compatibility

The current decoder reconstructs observations in **symlog two-hot** space (255 bins). The heteroscedastic extension has two options:

**Option A — Precision on continuous reconstruction**: Add a parallel continuous decoder head that predicts (ô, log π̂) alongside the existing two-hot head. The two-hot head drives the primary reconstruction; the continuous head drives precision learning. This avoids modifying the proven two-hot pipeline.

**Option B — Precision-weighted two-hot**: Modify the two-hot reconstruction loss to be precision-weighted:
```
L_recon = Σ_i exp(s_i) · [-log p_twohot(o_i | z, h)] - (1/2) s_i
```
This is cleaner (single head) but changes the proven loss function.

**Recommendation**: Start with **Option A** — add a small parallel continuous decoder for precision learning, keep the two-hot reconstruction unchanged. This is the safer path that doesn't risk regressing the world model's proven reconstruction quality.

#### Comparison with PPO-Based Experiments

| Dimension | PPO + Path 2 | DreamerV3 + Heteroscedastic Decoder |
|-----------|-------------|-------------------------------------|
| **Existing codebase** | PPO agent (proven, v7/v8 experiments) | DreamerV3 agent (implemented, S-tier audit, less experimental history) |
| **Maturity** | Many training runs, well-understood baselines | Fewer experiments in our grid world |
| **Precision training** | Auxiliary loss (competes with PPO) | Core objective (unit weight, no competition) |
| **Theoretical alignment** | Discriminative (policy + aux prediction) | **Generative** (Active Inference aligned) |
| **Complexity** | Lower (add one MLP head to existing agent) | Higher (work within world model framework) |
| **Risk** | Low (PPO baseline well-established) | Medium (DreamerV3 baseline in grid world less established) |
| **Scientific value** | Good (demonstrates precision modulation in model-free RL) | **Higher** (demonstrates precision within generative model — closer to Active Inference) |

### 8.9 Revised Recommendation: Two Parallel Tracks

The DreamerV3 consideration changes the recommendation. Rather than a single linear sequence (Path 2 → Path 3 → Path 4), we now have **two parallel tracks**:

**Track A — Model-Free (PPO + Path 2)**: Lower risk, faster iteration, well-established baselines. Demonstrates that precision modulation works as a computational principle, even in a discriminative framework. Use this track for rapid proof-of-concept.

**Track B — Model-Based (DreamerV3 + Heteroscedastic Decoder)**: Higher scientific value, closer Active Inference alignment, precision learning as a first-class objective. Use this track for the theoretically principled demonstration.

Both tracks test the same core hypothesis — **can an agent learn per-channel sensory precision from prediction errors?** — but in different architectural contexts. If Track A succeeds (PPO agent learns precision), Track B should succeed more cleanly (DreamerV3 provides a better substrate). If Track A fails, Track B tells us whether the failure was due to the auxiliary-loss formulation or a deeper issue with precision learning itself.

#### Revised Experimental Plan

**Phase 1a — PPO + Prediction Precision (Track A, low risk)**:
- As described in Section 7.7 — add prediction head + precision-gated FiLM to PPO agent
- Quick validation that prediction-error-based precision works at all
- Baseline comparison against v7/v8 results

**Phase 1b — DreamerV3 Baseline (Track B, establish baseline)**:
- Run standard DreamerV3 (no precision modifications) on grid world with and without noise
- Establish survival baselines comparable to PPO v8 experiments
- Verify world model reconstruction quality across modalities

**Phase 2 — DreamerV3 + Heteroscedastic Decoder (Track B, core experiment)**:
- Add precision outputs to decoder + Kendall & Gal reconstruction loss (Option A — parallel continuous head)
- Mechanism A first (implicit precision in latent state)
- Measure: does the decoder learn per-channel precision that correlates with environmental noise?
- Compare survival under noise: standard DreamerV3 vs heteroscedastic DreamerV3

**Phase 3 — Cross-Track Comparison**:
- Compare precision profiles: PPO-learned π̂ vs DreamerV3-learned π̂
- Compare survival improvement: which framework benefits more from precision?
- If DreamerV3 shows stronger precision learning → validates the generative-model-based approach
- Add world model ensemble (epistemic) and/or precision-gated FiLM (Mechanism C) as needed

---

## 9. Theoretical Connection: Neural Population Codes and Precision (FiLM-Ensemble)

The FiLM-Ensemble approach has a deeper theoretical justification within computational neuroscience:

**Probabilistic Population Codes (PPCs)**: Pouget, Beck, and colleagues have shown that biological neural populations can optimally represent probability distributions, with the **width of the population response** encoding precision. A narrow, peaked population response = high precision; a broad, flat response = low precision.

FiLM-Ensemble's M members function as a **computational population code**:
- Each member m applies its own γ^m, β^m, creating a distinct "neural sub-population" response
- The spread of member predictions encodes precision (narrow spread = high precision)
- The mean of member predictions encodes the best estimate

This is more biologically plausible than a single modulator network, which has no analog to population-level uncertainty coding. A single network can output a point estimate but cannot represent *how confident it is* in that estimate without additional architecture (e.g., distributional RL). The ensemble approach gets confidence estimation "for free" from the population structure.

**Connection to neuromodulation**: Biological neuromodulators (acetylcholine, norepinephrine) are thought to adjust the **gain** of neural populations, effectively changing how tightly the population codes cluster. FiLM's γ (scaling) parameter directly implements this gain control — different γ^m values create populations with different gains, and the resulting agreement/disagreement encodes precision.

---

## 10. Summary

| Question | Answer |
|----------|--------|
| What is the core goal? | Implement **sensory precision modulation** — the Active Inference principle that agents should dynamically weight sensory channels by reliability. |
| Why did single-FiLM fail? | Two reasons: (1) Single deterministic γ,β cannot represent uncertainty. (2) **Deeper**: PPO provides no training signal that rewards precision estimation — identity gates are optimal when no objective penalizes miscalibration. |
| What does FiLM-Ensemble add? | M implicit ensemble members create a **population code** where agreement = high precision, disagreement = low precision. Best for **epistemic** uncertainty. |
| What does Prediction Precision add? | Auxiliary prediction loss creates an **explicit training signal** for per-channel precision — the missing ingredient from v7/v8. Best for **aleatoric** uncertainty. |
| Two uncertainty types | **Aleatoric** (sensor noise, from environment) + **Epistemic** (model uncertainty, from ensemble). Kendall & Gal (2017) formalized their combination; we adapt it for RL. |
| Aleatoric modeling (recommended) | **Prediction-error precision**: auxiliary head predicts o_{t+1} with heteroscedastic loss; learned π̂_i per channel gates FiLM modulation. Directly implements Active Inference precision-weighted prediction error. |
| Epistemic modeling | FiLM-Ensemble member disagreement: Var(predictions) = epistemic uncertainty. Secondary to aleatoric precision in big-data RL regime. |
| Bayesian Hypernetworks link | Krueger et al. (2017) show hypernetworks can generate FiLM-like scale parameters from noise — a more principled continuous ensemble. Future extension path. |
| Survey positioning | Gawlikowski et al. (2023): our approach combines prediction-error precision (aleatoric) + optional implicit ensembles (epistemic) — addresses a gap for small recurrent RL networks. |
| Can it work with LayerNorm? | Yes — Conditional LayerNorm replaces LN affine params with per-member versions (ensemble). Precision gating is architecture-agnostic (prediction). |
| Can it work with PPO? | Yes — prediction loss is an auxiliary objective added to L_PPO. Precision gating modulates encoder features before policy/value computation. |
| Key success metric | **Per-channel precision profile**: does learned π̂_olfaction decrease (and π̂_location remain stable) as injury increases? This directly tests whether the agent has learned modality-specific precision modulation. |
| Biological plausibility | Prediction-error precision ≈ Active Inference free energy minimization. Precision gating ≈ cholinergic/noradrenergic gain control. FiLM-Ensemble members ≈ neural sub-populations (population codes). |
| Key risk (Ensemble) | Diversity collapse during RL training; implicit ensembles have underperformed in offline RL (though on-policy PPO may differ). |
| Key risk (Prediction) | λ_pred tuning: too high → agent over-invests in prediction at policy cost; too low → precision signal too weak. Prediction target design must align with modality structure. |
| DreamerV3 connection | DreamerV3's world model **natively provides** the observation prediction that Path 2 adds as an auxiliary loss. Making the decoder heteroscedastic gives precision learning as a first-class objective — not an auxiliary add-on competing with PPO. |
| DreamerV3 advantage | Precision lives in the **generative model** (world model) — the theoretically correct placement in Active Inference. Latent states encode precision for downstream actor-critic. No λ_pred tuning needed (unit weight). |
| DreamerV3 + epistemic | Plan2Explore-style world model ensemble provides epistemic precision via transition model disagreement — more meaningful than FiLM-Ensemble's value disagreement. |
| **Recommended approach** | **Two parallel tracks**: Track A (PPO + Path 2) for rapid proof-of-concept; Track B (DreamerV3 + heteroscedastic decoder) for theoretically principled demonstration. Both test the same core hypothesis. |
| Recommended Phase 1a | **Track A**: PPO + prediction head + precision-gated FiLM. MC return, λ_pred sweep {0.1, 0.5, 1.0}. 2×2 ablation: ±prediction_head × ±FiLM_gating. |
| Recommended Phase 1b | **Track B**: DreamerV3 baseline on grid world ± noise. Establish survival baselines. |
| Recommended Phase 2 | **Track B**: DreamerV3 + heteroscedastic decoder (Option A — parallel continuous head). Measure per-channel π̂ vs injury correlation. |
| Recommended Phase 3 | Cross-track comparison: PPO-learned precision vs DreamerV3-learned precision. Add world model ensemble and/or FiLM gating as needed. |
