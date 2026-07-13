> **Per-paper review — continual-learning corpus, paper 16 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§16); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 16. Lyle et al. 2023 — Understanding Plasticity in Neural Networks

**PDF:** `docs/project/references/continual_learning/sources/Lyle et al. 2023 - Understanding Plasticity in Neural Networks.pdf`
**Venue:** ICML 2023 (PMLR 202). **Authors:** Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney (Google DeepMind).

## Phase 1: Foundational Overview (Lyle 2023)

**The plain-language question.** A neural network that has already trained on one stream of
data often becomes *worse at learning new things* than a fresh, randomly-initialized network
of the same size. This is "loss of plasticity" — the network's ability to quickly update its
predictions in response to new information decays over training. It is a *different* problem
from catastrophic forgetting (which is about losing *old* knowledge); here the network loses
the ability to acquire *new* knowledge. The problem is especially acute in reinforcement
learning (RL), where the prediction targets constantly change (the value-function "bootstrap"
target moves as the policy improves), so non-stationarity is baked in rather than incidental.

**What the paper does.** It is a *systematic empirical autopsy* of plasticity loss. The
authors ask three questions in sequence: (1) what does plasticity loss look like in simple,
interpretable settings? (2) which of the popular "culprit" quantities — weight norm, feature
rank, number of dead units, weight-matrix rank — actually *cause* it? (3) which interventions
fix it?

**Key findings.**
- **A falsification framework kills the popular single-number explanations.** For every
  candidate culprit (weight norm, feature rank, dead-unit count, weight rank), you can build a
  learning problem where it *positively* correlates with plasticity loss and another where it
  *negatively* correlates. A quantity that flips sign depending on the dataset cannot be the
  causal driver. So none of these simple statistics is a reliable explanation on its own.
- **The real signature is the curvature of the loss landscape.** Plasticity loss tracks
  changes in the *sharpness* (largest Hessian eigenvalue) and the *gradient interference*
  structure of the loss surface that new tasks induce on the trained parameters. Importantly,
  this happens *even when no units are saturated* — so "dead ReLUs" is not the whole story.
- **Gradient descent itself is part of the problem.** Compared to a random walk of the same
  step size, gradient-based optimization drives parameters into regions where the landscape is
  *sharper* and gradients *interfere* more — i.e., the inductive bias of SGD actively worsens
  future trainability, beyond what mere movement away from initialization would do.
- **Adaptive optimizers can catastrophically self-destruct under abrupt task changes.** When
  the loss suddenly jumps (e.g., re-randomized labels, or a target-network update), Adam's
  moment estimates become stale and the update explodes, killing most ReLU units. A larger
  `ε` and faster second-moment decay fixes it — which is why deep-RL practitioners already use
  a large Adam `ε` by folk wisdom.
- **The best fix is smoothing the loss landscape, not perturbing parameters.** Across MLP /
  CNN / ResNet / ViT, *layer normalization* (and categorical "two-hot" output encoding) beat
  resetting, weight decay, spectral norm, and shrink-and-perturb. Adding layer norm to a
  vanilla Double-DQN improves performance across the 57-game Atari benchmark with *no*
  hyperparameter retuning.

**Initial takeaway.** Loss of plasticity is not reducible to any one tidy scalar. The most
predictive lens is the *geometry of the loss surface* (curvature + gradient interference),
and the most reliable practical lever is an architectural choice — normalization — that keeps
that geometry well-behaved. This is the paper that reframed the field from "which statistic to
regularize" toward "keep the optimization landscape trainable."

*Primer connection:* this is a Phase-3 "mechanisms" pillar. It is the paper that establishes
**layer normalization** as the field's default first-line defense, a thread the primer traces
through to Lyle 2024's LN+weight-decay recipe.

## Phase 2: Graduate-Level Deep Dive (Lyle 2023)

### 2.1 Formal setting: TD learning as a non-stationarity generator

The analysis is grounded in temporal-difference (TD) learning, chosen because it manufactures
non-stationarity even with a *fixed* data distribution. Given sampled transitions
$\tau_t = (s_{t-1}, a_t, r_t, s_t)$ and a network $f:\Theta\times\mathcal{S}\times\mathcal{A}\to\mathbb{R}$,
the TD loss is

$$
\ell(\theta, \tau_t) = \Big( f(\theta, s_{t-1}, a_t) - \boxed{\big(r_t + \gamma\, f(\theta', s_t, a')\big)} \Big)^2 ,
$$

where $\boxed{\;\cdot\;}$ denotes a stop-gradient and $\theta'$ are the (typically stale)
target-network parameters. The crucial point: the regression target
$r_t + \gamma f(\theta', s_t, a')$ **depends on parameters and changes as learning proceeds**,
so the objective is non-stationary *independent of the exploration policy*. This isolates
"target drift" as the non-stationarity source.

### 2.2 Two loss-landscape probes: the Hessian spectrum and the gradient covariance

Two objects carry the entire mechanistic argument.

**(a) The Hessian and its spectrum.** For loss $\ell(\theta)$,

$$
H_\ell(\theta) = \nabla^2_\theta \ell(\theta) \in \mathbb{R}^{d\times d}, \qquad
\Lambda(H_\ell(\theta)) = (\lambda_1 \ge \cdots \ge \lambda_d).
$$

The top eigenvalue $\lambda_1$ measures **sharpness** (Dinh et al. 2017); the condition number
$\kappa = \lambda_1/\lambda_d$ governs gradient-descent convergence. A first-order intuition:
for a quadratic model $\ell(\theta)\approx \tfrac12(\theta-\theta^\*)^\top H (\theta-\theta^\*)$,
gradient descent with step $\alpha$ contracts error along eigen-direction $i$ by a factor
$(1-\alpha\lambda_i)$. Stability requires $\alpha < 2/\lambda_1$, so a growing $\lambda_1$
forces a smaller admissible learning rate; and convergence *speed* along the slowest direction
scales like $(1 - \alpha\lambda_d) = (1 - \alpha\lambda_1/\kappa)$, so a large $\kappa$ means
crawling progress. A sharpening landscape thus *directly* slows the fitting of new targets —
which is exactly what "loss of plasticity" is operationally.

**(b) The normalized gradient covariance (interference) matrix.** For sampled points
$x_1,\dots,x_k$,

$$
C_k[i,j] = \frac{\big\langle \nabla_\theta \ell(\theta, x_i),\ \nabla_\theta \ell(\theta, x_j)\big\rangle}
{\lVert \nabla_\theta \ell(\theta, x_i)\rVert\, \lVert \nabla_\theta \ell(\theta, x_j)\rVert}.
$$

This is the cosine similarity between per-example gradients. Negative off-diagonal entries
signal **interference**: reducing loss on $x_i$ *increases* it on $x_j$, so the network cannot
simultaneously satisfy both. A low-rank / block-structured $C_k$ means gradients are
near-collinear — helpful (generalization) when the dot product is positive, harmful
(interference) when negative.

### 2.3 Defining plasticity operationally

Plasticity is defined (following Lyle et al. 2021) via an optimization operator
$\mathcal{O}:(\theta,\ell)\mapsto\theta^\*$ that runs a *fixed budget* of updates (2000 steps
in experiments). Over a distribution $\mathcal{L}$ of probe losses,

$$
\mathcal{P}(\theta_t) = b - \mathbb{E}_{\ell\sim\mathcal{L}}\big[\ell(\theta^\*_t)\big],
\qquad \theta^\*_t = \mathcal{O}(\theta_t, \ell),
$$

with $b$ a baseline (e.g., target variance). **Plasticity loss** over a trajectory is
$\mathcal{P}(\theta_t) - \mathcal{P}(\theta_0)$; it is *independent of the baseline* $b$, so it
measures relative degradation of a checkpoint as an optimization *starting point*. The probe
targets are

$$
g(x) = a + \sin\!\big(10^5\, f(x;\omega_0)\big),
$$

with $\omega_0$ a fresh initialization and offset $a$ set to the network's *current mean
prediction* (so random-init baselines are not unfairly favored). The high-frequency $\sin$
makes the targets effectively a uniform random direction in output space — a task-agnostic
probe of "can this network still be pushed in an arbitrary direction?"

### 2.4 Case study 1 — Adam instability under abrupt non-stationarity

Adam's update is

$$
u_t = \alpha\,\frac{\hat m_t}{\sqrt{\hat v_t} + \bar\epsilon} + \epsilon,
$$

with $\hat m_t$ the first-moment (mean-gradient) and $\hat v_t$ the second-moment
(mean-squared-gradient) EMAs. Because gradient magnitude scales roughly with the loss, a sudden
loss jump (re-randomized labels; a target-network refresh) makes both estimates stale. Under
default hyperparameters $\hat m_t$ (decay $\beta_1=0.9$) updates faster than $\hat v_t$ (decay
$\beta_2=0.999$), so immediately after a task change the numerator has grown while the
denominator lags:

$$
u_t \;\sim\; \frac{\text{(large fresh gradient)}}{\sqrt{\text{(small stale second moment)}}}
\;\Rightarrow\; \text{huge step} \;\Rightarrow\; \text{ReLU death / divergence.}
$$

The fix is to raise $\bar\epsilon$/$\epsilon$ (damping the denominator when it is small) and
lower $\beta_2$ (making $\hat v_t$ track faster). The paper notes DQN's canonically large
Adam/RMSProp $\epsilon$ is exactly this stabilization, converged on empirically by the RL
community.

### 2.5 Case study 2 — SGD's inductive bias sharpens the landscape

Controlled experiment: two trajectories from the *same* initialization, applying updates of
*equal norm* — one following the true gradient, the other a Gaussian random walk (Brownian
motion). To probe how the *local geometry for arbitrary new targets* evolves, they measure the
Hessian of a stop-gradient perturbation objective

$$
\ell(\theta) = \big[\, f_\theta(X) - \boxed{f_\theta(X)} + \epsilon \,\big]^2, \qquad \epsilon\sim\mathcal{N}(0,1),
$$

i.e., "how hard is it to nudge the current outputs by random noise?" — deliberately *not* the
primary objective (whose Hessian trivially differs between trajectories). Result: both
trajectories increase the Hessian spectral norm, but **the gradient-descent trajectory grows
the outlier eigenvalues far faster and develops negative gradient interference** absent in the
random walk. Conclusion: SGD's inductive bias, not mere displacement from initialization,
pushes parameters into less-trainable regions.

### 2.6 The falsification framework (the paper's central methodological contribution)

The premise (after Bühlmann 2020, invariant/causal prediction): *a genuinely causal predictor
of plasticity loss must keep a consistent correlation sign across interventions*. They train
128 DQN agents across tasks × observation spaces (CIFAR-10 vs MNIST) × optimizers × seeds and
log, per checkpoint: weight norm, weight (matrix) rank, dead-unit count, feature rank, plus
plasticity. For **each** candidate, there exists one environment with positive and one with
negative correlation to plasticity loss — e.g., weight norm correlates *positively* with
plasticity loss under CIFAR-10 observations but *slightly negatively* under MNIST; feature rank
and sparsity reverse sign depending on the reward function. Sign-reversal under intervention
$\Rightarrow$ the quantity is **falsified as a universal causal explanation**. (Note the paper
is careful: these are still useful *diagnostics*, just not causal levers.)

### 2.7 Learning-curve diagnosis: slower slopes, not higher plateaus

Probing checkpoints at training iterations 0, 10, 20, 50, 100: later checkpoints do **not**
plateau early at a high loss (which would indicate bad local minima). Instead their learning
curves have *shallower slopes* and *higher variance / non-monotonicity*. In full-batch terms,
non-monotonic loss under fixed step size is the signature of an over-sharp landscape
(edge-of-stability, Cohen et al. 2021); in mini-batch terms they additionally observe rising
inter-minibatch interference. So plasticity loss = *difficulty navigating the landscape*, not
*entrapment in a minimum*.

### 2.8 Interventions and the scaling result

- **Scaling is insufficient.** Widening a CNN to the single-GPU memory limit reduces but does
  not eliminate plasticity loss on the toy classification MDPs, especially when the task is
  misaligned with the architecture's inductive bias (MLP on CIFAR-10) or the net is
  under-expressive.
- **Intervention ranking (Fig. 6).** Landscape-smoothing methods win: **layer normalization**
  and **two-hot categorical output** give the largest reductions, often exceeding
  last-layer resetting. Parameter-perturbing / regularizing methods (shrink-and-perturb,
  weight decay $10^{-5}$, spectral norm) help less. Caveat: two-hot destabilized the policy in
  some cases and needed different optimizer hyperparameters — not a drop-in.
- **Atari validation.** Adding layer norm after each hidden layer of Double-DQN (RMSProp,
  ε-greedy, frame stacking, 200M frames) robustly improves human-normalized score across the
  57 games with no retuning; the biggest wins occur where the *default* agent had degenerate
  gradient-covariance / ill-conditioned Hessian, and LN restores better-behaved gradient
  covariance — closing the mechanistic loop.

**Trade-off flagged for future work.** Two-hot / categorical encodings smooth the landscape but
change output scale/semantics; there is a genuine tension between *preserving a trainable
gradient structure* and *accurately representing an evolving value function*.

## Appendix: Section-by-Section Backbone (Lyle 2023)

- **Abstract.** Plasticity = ability to quickly change predictions on new info; essential for
  RL. Networks lose it even on simple problems; mechanism poorly understood. Systematic
  empirical analysis → loss of plasticity deeply connected to *loss-landscape curvature
  changes*, but *often occurs without saturated units*. Identify parameterization/optimization
  choices that preserve plasticity; validate layer norm on Atari (ALE).
- **§1 Introduction.** Non-stationary training → reduced ability to solve new tasks; worst when
  input→target relationship changes over time (network must "overwrite" priors) — rare in SL,
  baked into RL. Existing fixes (layer/unit resets, feature regularization) probably act via
  *different* mechanisms, so hard to improve. Contributions: two interpretable case studies;
  a falsification framework (after Dziugaite et al. 2020) showing no single property uniquely
  explains plasticity loss; evidence that *loss-landscape curvature* is the crucial factor;
  broad intervention study → landscape-smoothing architecture choices (categorical output,
  normalization) win; LN on DQN improves ALE.
- **§2 Background.** Distinguishes catastrophic forgetting (old-task performance) from
  plasticity loss (new-task performance falls below a fresh random net). §2.1 Preliminaries: TD
  learning (Eq. 1) as non-stationarity source; Hessian $H_\ell$ (Eq. 2) and its spectrum;
  normalized gradient covariance $C_k$ (Eq. 3) and interpretation (interference vs
  generalization; low rank ⇒ collinearity). §2.2 Defining plasticity: classical complexity
  (VC/Rademacher) is capacity-agnostic to trainability → unsuitable; adopt Lyle et al. 2021
  operator-based definition; optimization operator $\mathcal{O}$; probe-loss distribution
  (Eq. 4); plasticity $\mathcal{P}$ (Eq. 5); trajectory loss $\mathcal{P}(\theta_t)-\mathcal{P}(\theta_0)$, baseline-independent.
- **§3 Methodology & motivating questions.** §3.1 Measuring plasticity: uniform prior over
  future targets via $g(x)=a+\sin(10^5 f(x;\omega_0))$, offset $a$ = current mean prediction,
  2000-step budget, 10 sampled functions. §3.2 Environments: block-MDP analogue of image
  classification over 10 states/actions with CIFAR-10 or MNIST observations — three variants:
  *true-label* (reward $\delta_{a=s}$), *random-label* (labels randomized; inductive-bias
  misaligned), *sparse-reward* (reward $\delta_{a=s=9}$, policy affects visitation). §3.3
  Outline: §4 what happens; §5 what properties cause it; §6 how to mitigate.
- **§4 Two simple studies.** §4.1 Optimizer instability: MLP memorizing re-randomized MNIST
  labels → default Adam diverges, kills ReLUs (Fig. 1); mechanism via Adam moment staleness
  (Eq. 6); fix = larger $\epsilon$, faster $\beta_2$. §4.2 Loss-landscape evolution: GD vs
  equal-norm Brownian motion from same init; stop-gradient perturbation Hessian; GD grows
  spectral outliers faster and induces negative gradient interference (Fig. 2).
- **§5 Explaining plasticity loss.** §5.1 Setting: DQN on each MDP×observation combo; probe
  every 5000 steps. §5.2 Falsification: 128 agents; each of weight norm / weight rank / dead
  units / feature rank reverses correlation sign across environments (Fig. 3) → none is a
  universal cause. §5.3 Learning-curve evolution: later checkpoints have shallower, more
  non-monotone probe-task learning curves (Fig. 4) → slow navigation, not bad minima.
- **§6 Solutions.** §6.1 Scaling reduces but doesn't eliminate plasticity loss (Fig. 5). §6.2
  Interventions across MLP/CNN/ResNet/ViT (Fig. 6): landscape-smoothing (LN, two-hot) beats
  perturbation/regularization; two-hot caveats. §6.3 Larger benchmark: LN on Double-DQN
  improves ALE across 57 games, changes gradient-covariance structure (Fig. 7).
- **§7 Related work.** Trainability/initialization (Glorot, He, mean-field, deep kernel
  shaping); ResNets bias toward identity → better gradients; loss-landscape smoothness &
  generalization; edge-of-stability, catapult, linear mode connectivity (all stationary SL);
  continual learning; resetting/distillation literature.
- **§8 Conclusions.** Contrast with large-model pretraining (good objectives accelerate
  adaptation) vs plasticity loss (bad objectives hurt adaptation). Stabilizing the loss
  landscape is the crucial lever, with ancillary benefits (easier optimization, better
  generalization).
- **Appendix A.** Case-study details (MNIST memorization MLP width-1024, Adam lr 1e-3 default
  vs tuned $\beta_2{=}0.9,\bar\epsilon{=}10^{-3}$; Brownian Q-learning, SGD lr 1e-3, batch 512;
  Lanczos Hessian eigenvalue density; probe protocol; MLP/CNN/ResNet-18/ViT architectures;
  Double-DQN ALE protocol, LN after each hidden layer, replay 100k, ε=0.1).

---
