> **Per-paper review — continual-learning corpus, paper 17 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§17); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 17. Lyle et al. 2024 — Disentangling the Causes of Plasticity Loss in Neural Networks

**PDF:** `docs/project/references/continual_learning/sources/Lyle et al. 2024 - Disentangling the Causes of Plasticity Loss.pdf`
**Venue:** Preprint (arXiv:2402.18762), Mar 2024. **Authors:** Clare Lyle, Zeyu Zheng, Khimya Khetarpal, Will Dabney, Hado van Hasselt, Razvan Pascanu, James Martens (Google DeepMind).

## Phase 1: Foundational Overview (Lyle 2024)

**The plain-language question.** Lyle 2023 showed that *no single number* explains plasticity
loss. So a natural worry follows: if you fix one cause, the network can still lose plasticity
through a *different* cause you didn't address. This follow-up paper asks: **how many
independent mechanisms are there, do they overlap, and can we combine one fix per mechanism to
get a robustly plastic learner?**

**The "Swiss cheese" model.** The paper's organizing metaphor: each mitigation is a slice of
Swiss cheese with holes (failure modes it misses). Any single slice lets some plasticity loss
through its holes. Stack slices whose holes *don't line up* — each targeting an independent
mechanism — and almost nothing gets through.

**Key findings.**
- **Plasticity loss decomposes into (at least) three independent mechanisms:**
  1. **Preactivation distribution shift** — the inputs to nonlinearities drift, producing not
     only the known "dead units" but also a *newly identified* pathology: **unit linearization
     ("zombie units")**, where a unit collapses into an effectively linear map and stops
     contributing nonlinearity.
  2. **Regression-target magnitude** — simply regressing on targets with a *large mean* (even
     stationary!) destroys plasticity. This alone explains a large fraction of previously
     reported RL plasticity loss, because value targets grow like $(1-\gamma)^{-1}$.
  3. **Parameter-norm growth** — weights grow → loss-landscape sharpness rises → optimization
     slows; also saturates normalization/softmax components.
- **All three roads end at the same place: a degenerate empirical Neural Tangent Kernel
  (eNTK).** However plasticity is lost, the network's eNTK collapses toward a
  *diagonal-plus-rank-1* structure — a shared "fingerprint" that is *predictive* of training
  difficulty and serves as a unified diagnostic.
- **Divide-and-conquer works.** Find the best fix for each mechanism in isolation, then combine:
  **layer normalization** (fixes preactivation shift) **+ L2 weight decay** (fixes norm growth)
  **+ scale-invariant / categorical output** (fixes target magnitude) gives *additive* benefit
  and drives plasticity loss to near-zero across synthetic non-stationary benchmarks, Atari
  (C51/Rainbow), DeepMind Control (SAC humanoid), and natural distribution shift (iWildcam).

**Initial takeaway.** Plasticity loss is not one disease but several, each with its own cure;
the practical recipe that emerges — **layer norm + weight decay + scale-invariant output** — is
cheap, combinable, and near-complete, and the eNTK gives a single diagnostic that all failure
modes share.

*Primer connection:* this is the Phase-3 paper that turns the "mechanisms" pillar into a
*prescription*. The primer's headline recommendation — LN + weight decay as the default
plasticity guardrail — comes directly from here; the target-magnitude mechanism also links to
the distributional-RL / two-hot thread the primer tracks.

## Phase 2: Graduate-Level Deep Dive (Lyle 2024)

### 2.1 What kinds of non-stationarity induce plasticity loss?

Two independent problem-side factors:

**(a) Regression-target scale (dose-response).** Construct a *stationary* pretraining task:
targets are random labels offset by a constant, $f(x) = c + \epsilon(x)$ on MNIST inputs. Then
fine-tune on fresh targets (offset $c$ or zero-mean). Result: **larger pretraining offset $c$
$\Rightarrow$ worse fine-tuning**, monotonically — a clean dose-response curve. This is
decisive because the pretraining task is stationary; it is the *target mean*, not the
*existence* of non-stationarity, that drove the earlier DQN contextual-bandit plasticity loss
(where the optimal value is a one-hot plus a bias converging to $(1-\gamma)^{-1}$).

**(b) Smoothness of the distribution shift.** Sequentially re-randomize a fraction
$\epsilon\in[0.01,1]$ of CIFAR-10 labels each iteration. Larger $\epsilon$ (more abrupt shift)
$\Rightarrow$ more severe, more precipitous plasticity loss; $\epsilon=0.01$ barely hurts,
$\epsilon=0.1$ already accelerates the decline. Sudden shifts induce large gradient-magnitude
spikes (cf. the Adam instability of Lyle 2023).

### 2.2 Mechanism 1 — preactivation distribution shift, dead units, and unit *linearization*

Let $z = \phi(a)$ where $a$ is the preactivation and $\phi$ the nonlinearity. Two failure modes
of a shifting preactivation distribution:

- **Dead unit.** $a<0$ for *all* inputs (ReLU) or $|a|$ huge (tanh) ⇒ $\phi'(a)=0$ ⇒ no
  gradient flows to the incoming weights ⇒ they freeze permanently (unless the *input*
  distribution later shifts). Solution candidate: non-saturating activations (Leaky ReLU).
- **Unit linearization ("zombie unit") — the new contribution.** A ReLU with *only positive*
  preactivations acts as the identity $\phi(a)=a$; a smooth unit with very *low-variance*
  inputs behaves near-linearly. Unlike dead units, zombies propagate gradients perfectly (even
  "perfect" signal propagation) — but they contribute **no nonlinearity**, so the network's
  *effective expressive power* silently collapses (Montúfar et al. 2014, Raghu et al. 2017:
  count of linear regions). This is invisible to dead-unit counters, which is why it had been
  missed.

**Two-phase post-task-change dynamics.** Immediately after a task switch the network passes
through (i) an **erasing phase**: predictive entropy spikes, incorrect logits are pushed toward
zero, gradients on first-layer incoming weights push *all* preactivations the same direction
(uniform sign of $\langle \nabla \ell(\theta,X), x\rangle$), coupled with large step sizes ⇒ a
burst of units entering the linearized/saturated regime; then (ii) a **disentanglement phase**:
as the loss shifts to *raising correct* logits, gradient directions diversify and nonlinearity
partially recovers — *provided* enough units did not permanently saturate.

### 2.3 Mechanism 2 — regression-target magnitude via the bias-encoding singular value

Why does a large target mean poison the features? SVD of the penultimate-layer feature matrix
$\Phi\in\mathbb{R}^{n\times d}$ (n sampled inputs) reveals the mechanism. Training on
$f(x)=100+\epsilon(x)$:

- Without normalization, the **maximal singular value explodes** from $\mathcal{O}(10^3)$
  (mean-zero targets) to $\mathcal{O}(10^8)$ (mean-100 targets).
- The network *does not* use its output-layer **bias weight** to represent the large mean
  (bias norm is non-monotone, stays small). Instead it **encodes the constant offset into a
  single feature direction** with which all embeddings have roughly constant dot product — a
  learned "pseudo-bias" living in feature space.
- Consequence: one singular value dwarfs the rest; the feature Gram matrix becomes
  **ill-conditioned** ($\sigma_1/\sigma_i \to \infty$), and lower-order singular values collapse
  relative to $\sigma_1$. Since the last-layer least-squares / gradient dynamics have condition
  number set by $\sigma_1^2/\sigma_i^2$, learning any *new* direction (which lives in the
  crushed subspace) becomes extremely slow. Layer norm bounds $\sigma_1$ (it caps feature norm)
  but does *not* fully fix the relative decay of lower singular values — hence LN alone is
  insufficient, motivating the combination with scale-invariant output encoding.

### 2.4 Mechanism 3 — parameter-norm growth → sharpness → saturation

Two learning difficulties from growing $\lVert\theta\rVert$:
- **Sharpness coupling.** Empirically parameter norm tracks the top Hessian eigenvalue (Fig. 3);
  last-layer norm ↔ sharpness is formalized in edge-of-stability analyses (Damian et al. 2022).
  Recall from Lyle 2023 §2.2: growing $\lambda_1$ shrinks the stable learning rate and worsens
  conditioning ⇒ slower adaptation.
- **Uneven-layer growth & component saturation.** If layers grow at different rates, the
  per-layer "effective learning rate" (which infinite-width analyses, e.g. μP/Yang, want equal
  across layers scaled by fan-in/out) becomes imbalanced. Large magnitudes also saturate
  softmax-attention heads and normalization layers (Wortsman et al. 2023; Merrill et al. 2021),
  shrinking the output change per fixed step. **Non-monotone caveat:** a network whose units are
  *all* saturated stops propagating gradients, so its norm stops growing — this is why parameter
  norm can be *causal* yet *not monotonically* related to plasticity loss.

### 2.5 The unifying signature: empirical NTK collapse

The empirical NTK is the matrix of *output* (not loss) gradient dot products:
$K_\theta(x,x') = \langle \nabla_\theta f(\theta,x), \nabla_\theta f(\theta,x')\rangle$.
Decompose $K_\theta = D_\theta + G_\theta$ (diagonal + off-diagonal). Intuition: $G_\theta=0$
⇒ no generalization between inputs; $G_\theta = c\mathbf{1}$, $D_\theta=0$ ⇒ rank-1, the net is
"multiply by zero, add a learned constant."

**Why the eNTK matters under non-stationarity (heuristic derivation).** To first order, one
gradient-descent step on $\ell(\theta)=\lVert f(\theta,X)-y\rVert^2$ changes the loss by

$$
\ell(\theta_t) - \ell(\theta_{t+1}) \;\approx\; \big(f(\theta_t,X)-y\big)^\top K_{\theta_t}(X,X)\,\big(f(\theta_t,X)-y\big) + \text{h.o.t.}
$$

*Derivation sketch.* With learning rate $\eta$, the update is
$\theta_{t+1}=\theta_t - \eta\,\nabla_\theta \ell(\theta_t)$, and
$\nabla_\theta \ell = 2\,J^\top r$ where $J = \nabla_\theta f(\theta_t,X)$ is the Jacobian and
$r = f(\theta_t,X)-y$ the residual. First-order Taylor of $\ell$ along the update:

$$
\ell(\theta_{t+1}) \approx \ell(\theta_t) + \nabla_\theta\ell^\top(\theta_{t+1}-\theta_t)
= \ell(\theta_t) - \eta\,\lVert \nabla_\theta \ell\rVert^2
= \ell(\theta_t) - 4\eta\, r^\top (J J^\top)\, r,
$$

and $JJ^\top = K_{\theta_t}(X,X)$ is precisely the eNTK. So the per-step loss reduction is the
residual passed through the eNTK quadratic form. **The problem under non-stationarity:** the
targets $y_t$ evolve, so an eNTK $K_{\theta_t}$ that was well-aligned with the residual
$y-f(\theta_t,X)$ at one moment need not remain aligned as $y_t$ moves. A **collapsed
(low-rank) eNTK** has few directions in which it can reduce *any* residual ⇒ it cannot chase a
moving target. Hence *maintaining a non-collapsed eNTK is critical*.

**Empirical fingerprint (Fig. 3).** A random init has a rich eNTK. All three
plasticity-losing conditions collapse toward *diagonal-plus-low-rank*: (i) large-target-mean
regression ⇒ eNTK ≈ diagonal + rank-1 within a few hundred steps; (ii) ReLU random-label
memorization ⇒ block-diagonal (dead-unit blocks); (iii) Leaky-ReLU version ⇒ same pathology,
milder. Strikingly *different learning dynamics, same eNTK endpoint*.

### 2.6 Mitigation: one fix per mechanism, then combine

| Mechanism | Best intervention | Notes |
|---|---|---|
| Unbounded norm growth | **Layer/batch normalization** (hard) + **L2** | Hard norm *constraint* on features works; regularizing feature norm softly is weaker; rescaling weight norm to init hurt optimization speed. |
| Preactivation shift | **Layer/batch norm** of preactivations; ReDo (reset dead) | Normalization aids both plasticity *and* single-task convergence; ReDo helps but can slow single-task convergence and misses init-time signal-prop issues (excluded on ResNets). |
| Loss-landscape conditioning | gradient-norm penalty, InFeR, Shampoo | None consistently beats **LN + L2**. |
| Target scale | **Two-hot categorical (distributional) output** + label smoothing | Two-hot lets the net represent large outputs *without* ill-conditioned features; label smoothing (mixture with uniform) fixes the saturated-softmax loss spike for $\gamma=0$; scale-invariant output is *necessary* in value-based RL. |

**Headline recipe: layer normalization + L2 weight decay.** LN fixes the preactivation
distribution; L2 caps weight-norm growth. The two are independent and additive. Validated on:
20-task synthetic CIFAR-10 non-stationarity sweeps (continual / composite / growing modes;
MLP/CNN/ResNet-18, L2 $=10^{-5}$) where accuracy *improves* over tasks; C51 + Rainbow on Atari
(LN gives modest but consistent gains; BN and L2 *interfere* with RL and are excluded); SAC on
DeepMind Control (LN gives near-uniform gains, striking on humanoid); iWildcam natural
distribution shift (LN + L2 widen the final-accuracy gap over 20 location shifts).

**Distributional-RL insight (mechanistic).** Categorical/two-hot losses help plasticity for
*two* reasons: (1) smoother gradients (Imani & White 2018), *and* (2) they let the network
encode large output values without developing an ill-conditioned, bias-dominated feature
representation — a genuinely new explanation of a known empirical benefit.

**Limitation flagged.** L2 controls norm at the cost of convergence speed; better norm-control
strategies that don't slow single-task training are open.

## Appendix: Section-by-Section Backbone (Lyle 2024)

- **Abstract.** The stationarity assumption underlies NN design; violating it (deep RL) makes
  learning brittle. Loss of plasticity = updating predictions gets harder over training. Claim:
  plasticity loss decomposes into *multiple independent mechanisms*; single interventions are
  insufficient but *combined* interventions are highly robust. **LN + weight decay** maintains
  plasticity across synthetic non-stationarities and ALE.
- **§1 Introduction.** Single-task training is easy; real relationships are dynamic; RL changes
  its own data distribution. Resets are expensive. Plasticity loss observed independently in RL
  and SL. Lyle et al. 2023 gave negative results (no single quantity). Existing methods either
  target one pathology (risk leakage via others) or regularize abstract properties (risk
  interfering with objective). Goal: a model combining both → target several independent
  mechanisms. Three questions: what non-stationarity induces it; what structural changes occur;
  what do plasticity-lost networks share. Answers: mechanisms unified by *preactivation
  distribution shift*; degeneracies in *empirical NTK*; new mechanism *unit linearization*;
  *target magnitude* explains much RL plasticity loss. → "Swiss cheese" mitigation model with
  additive benefit.
- **§2 Background & related work.** §2.1 Training/signal propagation/preactivation
  distributions: layer norm forces preactivation sample mean 0 / variance 1; signal-propagation
  literature; good init keeps preactivations $\approx\mathcal{N}(\mu,\sigma^2)$; no universal
  $(\mu,\sigma^2)$ but $(0,1)$ usually fine; LN enforces 1st/2nd-order statistics exactly. §2.2
  Loss of plasticity: two usages of "plasticity" (generalization vs trainability) — adopt
  *trainability*; plasticity = quality of a point as an optimization start; can include
  optimizer state; not necessarily permanent but transient loss still harmful. Prior mechanisms;
  none causal in isolation (Lyle 2023).
- **§3 A deeper look into plasticity.** §3.1 What non-stationarity induces it: *regression
  target scale* (stationary large-mean pretraining reproduces plasticity loss; dose-response;
  Appendix E.1 bias-encoding); *smoothness of distribution shift* (larger reset fraction ⇒
  worse; Fig. 1). §3.2 Mechanisms: *linearization & preactivation shift* (dead units, degraded
  signal propagation, zombie/linearized units; two-phase erasing/disentanglement dynamics,
  Fig. 2); *parameter-norm growth* (instability, sharpness ↔ Hessian, uneven layer growth,
  component saturation; non-monotone relationship, Fig. 3). §3.3 Characterizing
  plasticity-lost networks: empirical NTK $= D_\theta+G_\theta$; first-order loss-decrease
  formula; moving-target argument; all conditions collapse eNTK toward diagonal+low-rank
  (Fig. 3) → eNTK as diagnostic.
- **§4 Mitigation strategies.** §4.1 Per-mechanism: unbounded norm growth (hard LN/BN + L2 beats
  soft feature-norm regularization; weight-norm rescaling hurts, Fig. 4 left); preactivation
  statistics (LN/BN + ReDo, Fig. 4 center); loss-landscape conditioning (grad-norm penalty,
  InFeR, Shampoo — none beats LN+L2, Fig. 4 right); target scale (two-hot categorical + label
  smoothing; scale-invariant output necessary in RL, Fig. 5). §4.2 Evaluation: SL (20-task
  CIFAR-10 modes; accuracy improves over tasks); RL (C51/Rainbow + LN on ALE; SAC + LN on DMC
  humanoid; BN/L2 interfere with RL); natural shift (iWildcam, LN+L2 widen the gap).
- **§5 Conclusions.** No single property explains all plasticity loss, but a handful of
  independent mechanisms cover most cases; some new (large target offsets, zombification).
  Divide-and-conquer reduces combinatorial search. L2 controls norm but slows convergence —
  open problem.
- **Appendix E (theory highlights).** E.1: SVD of penultimate features; max singular value
  $\mathcal{O}(10^3)\to\mathcal{O}(10^8)$ with offset; network encodes bias in a
  single feature direction rather than the bias weight; ill-conditioning follows (Figs. 13–15).
  E.2: uneven per-layer norm growth. E.4/E.11: RL per-game/DMC results.

---
