# Dreamer-Family Literature Review

## Scope

This review covers the world-model RL lineage from Ha & Schmidhuber's *World
Models* (2018) through PlaNet, DreamerV1, DreamerV2, the DreamerV3 preprint
and Nature paper, and Dreamer 4 (2025). All sources are local PDFs in
`docs/project/references/Dreamer/`; an earlier turn attempted NotebookLM and
hit auth blockers, so this run reads the PDFs directly. The master file is
written in two reviewer turns: this run covers papers 1–3 (chronological); the
second run will append papers 4–6.

## Reading order

- **Paper 1 — World Models (Ha & Schmidhuber, 2018).** Establishes the V/M/C
  decomposition (perception VAE, MDN-RNN forward model, tiny linear/CMA-ES
  controller) and the "train inside the dream" recipe — the conceptual
  ancestor of every later paper here.
- **Paper 2 — PlaNet (Hafner et al., 2019, ICML).** Introduces the Recurrent
  State-Space Model (RSSM) — the deterministic-plus-stochastic latent that
  becomes the universal Dreamer backbone — and uses it for CEM planning rather
  than learned policies.
- **Paper 3 — DreamerV1 / Dream to Control (Hafner et al., 2020, ICLR).**
  Replaces CEM planning with an actor-critic trained *inside* RSSM
  imagination, using λ-returns and reparameterised gradients flowing back
  through learned dynamics.
- **Paper 4 — DreamerV2 / Mastering Atari with Discrete World Models (Hafner
  et al., 2021, ICLR).** Swaps the Gaussian latent for a categorical (one-hot)
  latent and adds KL balancing — first model-based agent to beat humans on
  Atari-200M. *Pending second reviewer turn.*
- **Paper 5a — DreamerV3 preprint, *Mastering Diverse Domains Through World
  Models* (Hafner et al., 2023).** Single set of hyperparameters across 150+
  tasks, symlog/two-hot reward and value heads, percentile-based return
  normalisation. *Pending second reviewer turn.*
- **Paper 5b — DreamerV3 Nature, *Mastering Diverse Control Tasks Through
  World Models* (Hafner et al., 2025).** Published version of DreamerV3.
  *Pending second reviewer turn.*
- **Paper 6 — Dreamer 4 / Training Agents Inside of Scalable World Models
  (Hafner, Yan & Lillicrap, 2025).** Scales the world-model-as-environment
  paradigm to large-scale offline training. *Pending second reviewer turn.*

## Table of Contents

1. [Paper 1 — World Models (2018)](#paper-1) — written
2. [Paper 2 — PlaNet (2019)](#paper-2) — written
3. [Paper 3 — DreamerV1 (2020)](#paper-3) — written
4. [Paper 4 — DreamerV2 (2021)](#paper-4) — written
5. a. [Paper 5a — DreamerV3 preprint (2023)](#paper-5a) — written
   b. [Paper 5b — DreamerV3 Nature (2025)](#paper-5b) — written
6. [Paper 6 — Dreamer 4 (2025)](#paper-6) — written
7. [Cross-paper synthesis](#cross-paper-synthesis)

---

## <a id="paper-1"></a>Paper 1 — World Models (Ha & Schmidhuber 2018)

**Citation:** Ha, D., & Schmidhuber, J. (2018). *World Models.* arXiv:1803.10122.

**PDF:** `docs/project/references/Dreamer/sources/Ha and Schmidhuber 2018 - World Models.pdf` (21 pages)

### Backbone

- **Problem.** Model-free deep RL is bottlenecked by credit assignment when
  the policy network is large. The authors propose splitting the agent into a
  large *world model* trained unsupervised on raw pixel rollouts plus a
  *small* controller trained with evolution strategies, so that the heavy
  lifting (pixel compression, temporal prediction) is offloaded to gradient
  descent on V and M while CMA-ES need only optimize a few hundred parameters
  in C.
- **Method — V / M / C decomposition** (Section 2; Figs. 4–8).
  - **V (Vision).** A Convolutional Variational Autoencoder compresses each
    $64\times64\times3$ frame into a low-dimensional latent
    $z \in \mathbb{R}^{N_z}$ (32 for CarRacing, 64 for VizDoom). Trained on a
    one-shot dataset of 10,000 random-policy rollouts.
  - **M (Memory).** An LSTM with a Mixture-Density-Network output head
    (MDN-RNN) models $P(z_{t+1}\mid a_t, z_t, h_t)$ as a mixture of diagonal
    Gaussians. The hidden state $h_t$ is the temporal summary the controller
    sees alongside $z_t$. In the VizDoom variant, M additionally predicts the
    binary "done" event $d_{t+1}$.
  - **C (Controller).** A single linear layer:
    $a_t = W_c [z_t\; h_t] + b_c$ (CarRacing C has only 867 parameters),
    optimised by CMA-ES against true environment return.
- **Key equations and mechanisms** (with one-line explanation each).
  - Linear controller $a_t = W_c [z_t\; h_t] + b_c$ — minimal policy; all
    expressiveness lives in V and M.
  - MDN-RNN density $P(z_{t+1}\mid a_t, z_t, h_t)$ as a $K$-component mixture
    of diagonal Gaussians — captures discrete environment events
    (e.g. monster decides to fire) that a single Gaussian cannot.
  - Temperature trick: sampling $z_{t+1} \sim P(\cdot)^{1/\tau}$ broadens the
    mixture for $\tau > 1$, narrowing it for $\tau < 1$. Higher $\tau$ injects
    stochasticity that prevents C from exploiting M's idiosyncrasies during
    "dream" training.
- **Results** (Sections 3–4).
  - **CarRacing-v0.** Full V+M+C scores 906 ± 21 over 100 random tracks,
    crossing the 900 threshold and reportedly the first method to do so.
    V-only ablations: 632 ± 251 (linear) and 788 ± 141 (linear with hidden
    layer).
  - **VizDoom Take Cover.** Trained *entirely inside the M-generated dream*
    (no actual game-engine rollouts during policy search). Transferred to the
    real environment, the policy survives 1092 ± 556 steps versus the 750
    threshold and the leaderboard's 820 ± 58.
  - **Temperature sweep on VizDoom (Table 2).** $\tau = 0.10$ gets a perfect
    virtual score (2086) but transfers terribly (193 in actual). $\tau \approx
    1.15$ is the sweet spot (1092 actual). Too high ($\tau = 1.30$) makes the
    dream so noisy the agent cannot learn.
- **Limitations** (Section 7).
  - Unsupervised V is not task-aware — it spent capacity on irrelevant detail
    (Doom wall textures) and missed task-relevant features (CarRacing road
    tiles).
  - LSTM-based M has bounded weight capacity and suffers catastrophic
    forgetting under the iterative training loop.
  - The "cheat the world model" failure mode (Section 4.5): C finds
    adversarial trajectories that look optimal under M but break in reality.
    Temperature is a partial mitigation; the deeper fix (a Bayesian /
    uncertainty-aware M, or letting C *use* M's subroutines without trusting
    them) is left open.
  - Step-by-step rollout in M is not human-like hierarchical planning.

### Phase 1 — Foundational synthesis

Imagine learning to drive by first watching a thousand random dashcam clips
until you have a good intuitive picture of "what frames look like" and "what
typically happens next given the steering wheel." Once that intuitive model is
in place, you can practise driving entirely inside your imagination and only
walk out to the real road once your imagined performance is decent. *World
Models* is exactly this idea, formalised as three interacting neural networks.

The first network, **V (Vision)**, is a Variational Autoencoder. Its job is
to take a raw $64 \times 64$ pixel frame and compress it into a small vector
$z$ — typically 32 numbers. The VAE is trained simply to reconstruct the
frame from $z$, with a Gaussian prior on the latent code that keeps it
well-behaved. Crucially V is trained on data collected by a *random* policy:
you do not need a smart agent to gather a useful dataset.

The second network, **M (Memory)**, is a recurrent network — an LSTM —
whose job is to predict the *next* latent vector $z_{t+1}$ given the current
$z_t$, the current action $a_t$, and its own running hidden state $h_t$. The
prediction is not deterministic: it outputs the parameters of a *mixture of
Gaussians* (an MDN-RNN). This matters because real environments contain
discrete random events — a monster choosing to shoot, a car suddenly
appearing — that one smooth Gaussian cannot capture but a mixture can.

The third network, **C (Controller)**, is a tiny linear map from
$[z_t, h_t]$ to actions. It has fewer than a thousand parameters. It is so
small that you do not even need backpropagation: an evolutionary algorithm
(CMA-ES) explores the few hundred numbers and finds weights that maximise
total reward.

The decomposition matters because each piece is doing what it is best at.
Backpropagation handles the high-dimensional but easy problem of compressing
images and predicting next frames. Evolution handles the low-dimensional but
hard problem of credit assignment, where rewards come hundreds of steps after
the action that earned them.

The most striking result is **dreaming**. Once V and M are trained, you can
discard the actual game engine entirely. Roll M forward, sample $z_{t+1}$
from its mixture-density head, feed that back in along with C's action, and
you have a fully self-contained simulator inside the network. Train C inside
that simulator, then transfer the policy to the real game. On VizDoom Take
Cover this works: a controller that never saw the real game during
policy-search outperforms specialised baselines on the real environment.

The catch is that M is imperfect. C is a relentless adversary against any
modelling error in M, and quickly finds "exploits" — policies that achieve
high reward in the dream by abusing artefacts that do not exist in reality.
The fix is a temperature knob $\tau$: dialling up the entropy of M's mixture
output makes the dream noisier and harder, so policies that survive the
nightmare must be robust enough to also survive the real world. This idea —
that *uncertainty in the world model is a feature, not a bug* — is the seed
that PlaNet, Dreamer, and especially DreamerV3 will keep cultivating.

For world-model RL, the paper's lasting contribution is the **separation of
concerns**: a heavy unsupervised perceptual+temporal model trained on
random data, plus a tiny task-specific controller. Every later paper in this
review keeps the spirit of that decomposition, even as the controller goes
from a linear CMA-ES policy to a deep actor-critic with reparameterised
gradients.

### Phase 2 — Graduate-level deep dive

#### V — Convolutional VAE training (Sections 2.1, A.1)

V is a standard ConvVAE (Kingma & Welling 2013) with $N_z = 32$ for CarRacing
and $N_z = 64$ for VizDoom. The encoder produces $(\mu, \sigma)$, the latent
is reparameterised as $z = \mu + \sigma \odot \varepsilon$ with
$\varepsilon \sim \mathcal{N}(0, I)$, and the decoder maps $z$ back to a
$64{\times}64{\times}3$ image. The training objective is the negative ELBO,
which the paper writes as L2 reconstruction plus KL to the standard normal:

$$
\mathcal{L}_{\mathrm{VAE}}
= \mathbb{E}_{q_\phi(z\mid x)}\!\left[\|x - \hat x(z)\|_2^2\right]
+ \beta \, D_{\mathrm{KL}}\!\left[q_\phi(z\mid x)\,\|\,\mathcal{N}(0, I)\right].
$$

For diagonal Gaussian $q_\phi(z\mid x) = \mathcal{N}(\mu(x), \mathrm{diag}(\sigma^2(x)))$, the
KL term has the closed form

$$
D_{\mathrm{KL}}\!\left[q_\phi\,\|\,\mathcal{N}(0, I)\right]
= \tfrac{1}{2} \sum_{i=1}^{N_z} \left( \mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2 \right).
$$

The Gaussian prior is doing real work: it bounds the per-frame information
content of $z$, which the paper notes makes the *world model more robust to
unrealistic $z$ vectors generated by the M model*. In a regular autoencoder
the latent space could be arbitrarily peaky, so an adversarial controller
inside the dream could nudge $z$ into regions the decoder never saw. The
Gaussian prior smooths this out.

#### M — MDN-RNN training (Sections 2.2, A.2)

M is an LSTM whose hidden state $h_t$ is fed into a Mixture Density Network
head producing the parameters of a $K$-component diagonal-Gaussian mixture
over the next latent $z_{t+1}$ (CarRacing and VizDoom both use $K = 5$). At
each step the head emits

$$
\big(\pi_t^{(k)},\, \mu_t^{(k)},\, \sigma_t^{(k)}\big)_{k=1}^{K},
\qquad
\pi_t^{(k)} = \frac{\exp(\tilde\pi_t^{(k)})}{\sum_{j} \exp(\tilde\pi_t^{(j)})},
\quad
\sigma_t^{(k)} = \exp(\tilde\sigma_t^{(k)}),
$$

so the predictive density is

$$
p(z_{t+1}\mid a_t, z_t, h_t)
= \sum_{k=1}^{K} \pi_t^{(k)}\,\mathcal{N}\!\left(z_{t+1};\,\mu_t^{(k)},\,\mathrm{diag}\big((\sigma_t^{(k)})^2\big)\right).
$$

For VizDoom the head additionally outputs a Bernoulli "done" probability
$p(d_{t+1} = 1 \mid a_t, z_t, h_t)$ so M can terminate dream rollouts. Training
maximises the log-likelihood over recorded trajectories with teacher forcing,
i.e. minimises

$$
\mathcal{L}_{\mathrm{MDN\text{-}RNN}}
= -\sum_t \log p(z_{t+1}\mid a_t, z_t, h_t)
\;-\; \alpha \sum_t \big[\, d_{t+1}^* \log p(d_{t+1}=1) + (1 - d_{t+1}^*) \log p(d_{t+1}=0)\big],
$$

where $\alpha$ trades off the (rare) done event. As an over-fitting guard,
each training step *resamples* $z$ from the stored $(\mu, \sigma)$ of the VAE
encoder rather than reusing the same draw — a small but principled stochastic
data-augmentation pass.

#### Temperature and the exploit problem (Sections 4.5, 5)

When sampling from the MDN-RNN, the paper raises the mixture to a power
$1/\tau$ and renormalises:

$$
p_\tau(z_{t+1}\mid \cdot) \propto \sum_k \pi_t^{(k)\,1/\tau}\;
\mathcal{N}\!\left(z_{t+1};\,\mu_t^{(k)},\,\tau\cdot\mathrm{diag}\big((\sigma_t^{(k)})^2\big)\right).
$$

(Operationally this is implemented componentwise — softmax temperature on the
mixture weights and a multiplicative bump on the variance — but the net
effect is the standard "inverse temperature on log-density" scaling.)
Choosing $\tau > 1$ widens the predictive distribution and forces the
controller to generalise across a noisier dream; choosing $\tau \to 0$
collapses to the most-likely mode of the mixture, which the paper notes
yields *deterministic* rollouts where the dream Doom monsters never fire and
C trivially survives forever — but transfers to a real-world score of 193
(below the random baseline). Table 2's sweep is the empirical evidence that
$\tau$ is a knob trading **realism vs. exploitability**: too small and C
cheats, too large and C cannot learn at all.

#### C — CMA-ES on the linear controller (Sections 2.3, 2.4, A.4)

C is parameterised by $\theta = (W_c, b_c)$. The fitness function is the
expected total reward of a rollout:

$$
F(\theta) = \mathbb{E}_{\xi}\!\left[ \sum_{t=0}^{T-1} r_t \,\Big|\, a_t = W_c [z_t\; h_t] + b_c \right],
$$

estimated by running 16 random rollouts per individual. CMA-ES maintains a
multivariate-Gaussian search distribution
$\theta \sim \mathcal{N}(m_g, \Sigma_g)$ at generation $g$. At each
generation it (i) samples a population of size $\lambda = 64$, (ii) evaluates
each member's fitness, (iii) updates the mean toward the top-$\mu$ elite, and
(iv) adapts $\Sigma_g$ along the elite's covariance. CMA-ES is preferred here
because: it requires only a scalar fitness (so noisy or sparse rewards work);
it is trivially parallel across CPU cores; and the search space is small
enough (~10³ parameters) that covariance estimation is tractable. CarRacing
needed ~1800 generations to break score 900.

#### Dreaming as a simulator-replacement (Sections 4.1, 4.3)

After V and M are trained, the dream rollout is a closed feedback loop:
$z_t \to a_t = C(z_t, h_t) \to (z_{t+1}, h_{t+1}, d_{t+1}) = M(z_t, a_t, h_t)$.
There is no decoder call inside the loop — C operates entirely in latent
space, so the dream is essentially free relative to running the real game
engine. The agent's "observation" is exactly $z_t$ (plus $h_t$); pixels are
only rendered for visualisation. The paper highlights this as a major
practical advantage: training in latent space means we no longer pay the
graphics cost of the game engine, and the dream environment can be arbitrarily
parallelised.

#### Iterative procedure for harder tasks (Section 5)

For environments where a random policy never visits the relevant states, the
authors sketch an iterative loop:

1. Initialise M, C randomly.
2. Roll C in the actual environment; log $(x_t, a_t, r_t, d_t)$.
3. Train M on $P(x_{t+1}, r_{t+1}, a_{t+1}, d_{t+1}\mid x_t, a_t, h_t)$ and C
   to optimise reward inside M.
4. Repeat from step 2.

They also propose using $\mathcal{L}_{\mathrm{MDN\text{-}RNN}}$ itself,
sign-flipped, as an intrinsic-curiosity bonus: states where M's predictive
log-likelihood is low are exactly the under-modelled regions. This is the
seed of the prediction-error-as-curiosity ideas that recur throughout the
Dreamer line.

#### Where the paper draws the line

The authors are explicit (Section 7) that their step-by-step rollout in M is
*not* human-like hierarchical planning, and that with PILCO-style Bayesian
dynamics the exploit problem could be more principled. They also note that
the VAE may waste capacity on task-irrelevant details — a known failure mode
that PlaNet's reconstruction-based RSSM training will inherit, and that
DreamerV2's discrete latent and DreamerV3's symlog/two-hot reward heads will
each chip away at.

### Relevance to this project (interoceptive-pain RL)

- **V/M/C as a template for interoception.** The clean split of perception,
  forward model, and policy maps directly onto the project's RPPO baselines
  vs. world-model variants; one can read "interoceptive precision" as a head
  of M that predicts noisy proprioceptive signals.
- **Temperature $\tau$ as a precision knob.** The paper's $\tau$ on the
  MDN-RNN is functionally a Bayesian-precision control over the *prior* the
  agent uses to decide. For pain-modeling work it is a candidate analogue of
  modulating perceptual precision under threat — worth a side-by-side with
  professor-bayesian-brain's predictive-coding framing.
- **Train-in-dream as a continual-learning probe.** The repo's recent
  RPPO continual-learning demo (commit `de102b8`, NoPred → PredInt3) is a
  task-curriculum on the *real* environment. World Models suggests an
  alternative: train the predictor first, then run continual learning of the
  policy entirely inside the predictor's dream, with $\tau$ as a difficulty
  knob.
- **Adversarial-policy failure mode.** The "cheat the world model" failure
  is exactly the sort of thing senior-developer should plan for if any
  imagined-rollout component is added — recommend a written
  `issue_plan` with explicit OOD-detection or KL-to-data regularisation
  before any imagination-based training is wired in.
- **Iterative curiosity bonus.** The sign-flipped MDN-RNN loss as intrinsic
  reward is a low-cost candidate addition to the existing exploration
  baselines; flagging for the experiment-designer.

---

## <a id="paper-2"></a>Paper 2 — PlaNet (Hafner et al. 2019)

**Citation:** Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D.,
Lee, H., & Davidson, J. (2019). *Learning Latent Dynamics for Planning from
Pixels.* ICML 2019 (PMLR 97).

**PDF:** `docs/project/references/Dreamer/sources/Hafner et al. 2019 - Learning Latent Dynamics for Planning from Pixels.pdf` (11 pages)

### Backbone

- **Problem.** Model-based RL has long under-performed model-free RL on
  pixel-based continuous control because learned dynamics models accumulate
  error over multi-step rollouts, fail to capture stochasticity, and grow
  overconfident off-distribution. Earlier latent-dynamics work (E2C, RCE)
  could only solve toy tasks (cartpole balance, 2-link arm) under full
  observability and dense reward. PlaNet aims to plan online from raw pixels
  on harder DeepMind Control Suite tasks (sparse reward, partial
  observability, contact dynamics) using a *learned* latent forward model.
- **Method — three pieces.**
  1. **Recurrent State-Space Model (RSSM).** A hybrid latent dynamics model
     with a *deterministic* recurrent path $h_t$ (a GRU rolling forward
     $h_{t-1}, s_{t-1}, a_{t-1} \mapsto h_t$) and a *stochastic* latent
     $s_t$ sampled from $p(s_t \mid h_t)$ at each step. Observations and
     rewards decode from $(h_t, s_t)$. The deterministic path lets the model
     remember information across many steps; the stochastic path lets it
     hedge over multiple plausible futures (Section 3, Eq. 4 / Fig. 2).
  2. **Variational ELBO with the filtering posterior.** The encoder
     $q(s_t \mid h_t, o_t)$ depends on *past* observations only (filtering,
     not smoothing) because planning will need to step the model forward
     without seeing the future (Eq. 3).
  3. **Latent overshooting.** A novel multi-step regulariser that pushes
     priors $p(s_t \mid s_{t-d})$ for all distances $d \le D$ toward the
     informed posteriors *in latent space*, without paying the cost of
     decoding multi-step pixel reconstructions (Eq. 6, Eq. 7, Fig. 3).
- **Planning.** Pure model-based control via Cross-Entropy Method (CEM) over
  action-sequence distributions: at every environment step, fit a Gaussian
  over $a_{t:t+H}$ by sampling $J$ candidates, evaluating predicted return
  under the RSSM, and re-fitting to the elite top-$K$. After $I$ iterations,
  execute $\mu_t$ and replan (MPC). No policy or value network is learned.
- **Key equations and mechanisms.**
  - RSSM transitions $h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$, $s_t \sim p(s_t \mid h_t)$ — the deterministic-plus-stochastic split that becomes the
    universal Dreamer backbone.
  - Standard ELBO: reconstruction $\mathbb{E}_q[\ln p(o_t \mid s_t)]$ minus
    one-step KL $\mathrm{KL}[q(s_t \mid o_{\le t}) \| p(s_t \mid s_{t-1})]$.
  - Multi-step prior $p(s_t \mid s_{t-d}) = \mathbb{E}_{p(s_{t-1}\mid s_{t-d})}[p(s_t \mid s_{t-1})]$
    and the resulting multi-step bound (Eq. 6).
  - Latent-overshooting objective averages the multi-step bounds for $d \in \{1, \dots, D\}$, weighted by $\beta_d$ (Eq. 7).
- **Results** (Section 5, Fig. 4, Table 1).
  - On six DeepMind-Control tasks (Cartpole-Swingup, Reacher-Easy,
    Cheetah-Run, Finger-Spin, Cup-Catch, Walker-Walk) trained from
    $64{\times}64{\times}3$ pixels, PlaNet matches or beats D4PG (a strong
    model-free baseline run for 100k episodes) using only ~1k episodes —
    a 40–500× data-efficiency gain. On Cheetah-Run, PlaNet exceeds D4PG by
    26%.
  - Ablations (Fig. 4) show: a *purely deterministic* GRU world model
    learns nothing on most tasks; a *purely stochastic* SSM is similarly
    bad. Both paths are required, with the stochastic component the
    more critical one ("the agent does not learn without it").
  - Ablations (Fig. 5) on agent design: random data collection cripples
    learning on cartpole/finger/walker (sparse-reward, exploration-hard);
    "random shooting" (1000 candidates, no CEM iteration) loses to
    iterative CEM on every task.
  - A single shared agent trained on all six tasks at once also solves
    them, only slower than per-task agents — first hint that the same
    RSSM scales across domains.
- **Limitations** (Section 7, Discussion).
  - Fixed action repeat is a manual hyperparameter (a temporal-abstraction
    placeholder).
  - No value function — return beyond the planning horizon $H$ is ignored.
  - Reconstruction-based ELBO ties the latent to pixel detail; visually
    diverse domains may need representation learning *without*
    reconstruction.
  - CEM is robust but throws compute at planning every step; gradient-based
    planning is left as future work.

### Phase 1 — Foundational synthesis

PlaNet asks: can a robot teach itself to do continuous control from raw
pixels by *imagining* the consequences of candidate plans? The answer is
yes, with two design choices that each look simple but each fix a problem
that had blocked the field.

The first design choice is the **architecture of the latent forward model**.
Picture two extremes. At one extreme, a recurrent network (GRU/LSTM) with
purely deterministic transitions: $h_t = f(h_{t-1}, a_{t-1})$. This
remembers the past well — that is what RNNs are for — but it cannot
represent uncertainty, and a planner is free to exploit any single confident
prediction error. At the other extreme, a stochastic state-space model:
$s_t \sim p(s_t \mid s_{t-1}, a_{t-1})$. This can hedge over multiple
futures, but the noise on the recurrence makes it bad at carrying
information — the cart's position can leak away within a few steps.

PlaNet's *Recurrent State-Space Model* (RSSM) takes the union: keep a
deterministic recurrent state $h_t$ for memory and add a stochastic latent
$s_t$ sampled fresh each step from $p(s_t \mid h_t)$ for hedging. The
ablations make this concrete — drop either piece and the agent fails. The
RSSM is the architectural innovation that the entire Dreamer line will
inherit unchanged for several years.

The second design choice is the **training objective**. PlaNet trains the
RSSM as a sequential VAE: encode each observation into a posterior over
$s_t$ given the past, decode pixels from $s_t$, predict reward from $s_t$,
and add a KL between the posterior and the one-step prior. So far this is a
standard variational lower bound. The trouble is that a planner needs *good
multi-step* predictions, and the standard ELBO only penalises *one-step*
KLs. With a finite-capacity model, getting one-step prediction right does
not imply that you get five-step prediction right; the small one-step errors
compound.

The fix is *latent overshooting*: a multi-step KL term in latent space.
Roll the prior forward $d$ steps to get $p(s_t \mid s_{t-d})$; train it to
match the encoder's posterior $q(s_t \mid o_{\le t})$. Average over a range
$d \in \{1, \dots, D\}$ with weights $\beta_d$. The trick is that this
is done *without* reconstructing pixels at every multi-step horizon — the
KL lives in the (small) latent space, so it is essentially free. The
authors then note an honest caveat: their final RSSM agent ends up *not*
needing latent overshooting (Appendix D), although other architectures
(deterministic-only, stochastic-only) benefit from it. So latent
overshooting is best read as a contribution to dynamics-model training
generally, not as the secret sauce of the headline RSSM.

The headline number is the **data efficiency**. PlaNet reaches D4PG's 100k-
episode performance in roughly 1k episodes — a 40× to 500× gain depending
on task. The agent does this with no policy network, no value network, just
an RSSM and a CEM planner that re-optimises an action distribution every
environment step. For a model-based-RL paper in 2019, beating a tuned
model-free baseline at all on pixel-based DeepMind-Control was the
accomplishment; doing it 100× faster was startling.

What PlaNet leaves on the table — and what DreamerV1 will grab — is the
absence of any *amortised* policy. CEM is doing the heavy lifting at
inference time, and it has to start from scratch every step. If you want
to actually deploy a fast policy, or to plan beyond the model's accurate
horizon, you need a value function and an actor — exactly the next paper.

### Phase 2 — Graduate-level deep dive

#### Variational ELBO for the latent state-space model (Section 3, Eq. 3)

Treat the trajectory model as a generative process over $s_{1:T}$ with
$p(s_t \mid s_{t-1}, a_{t-1})$, $p(o_t \mid s_t)$, $p(r_t \mid s_t)$. Apply
Jensen's inequality to the data log-likelihood:

$$
\ln p(o_{1:T} \mid a_{1:T})
= \ln \int \prod_{t=1}^{T} p(s_t \mid s_{t-1}, a_{t-1})\, p(o_t \mid s_t)\, ds_{1:T}
\;\ge\;
\sum_{t=1}^{T}\Big( \mathbb{E}_{q(s_t\mid o_{\le t})}[\ln p(o_t \mid s_t)]
\;-\; \mathbb{E}_{q(s_{t-1}\mid o_{\le t-1})}\!\big[\mathrm{KL}\!\left[q(s_t\mid o_{\le t})\,\|\,p(s_t\mid s_{t-1})\right]\big]\Big).
$$

The first term is reconstruction (per-step); the second is the *one-step*
KL between filtering posterior and prior. Reward terms follow by analogy.
Two important details: (i) a Gaussian observation likelihood with unit
variance reduces $\ln p(o_t \mid s_t)$ to MSE up to a constant — that is
how the reconstruction is implemented in practice; (ii) outer expectations
are estimated with a *single* reparameterised sample, so the whole
objective trains by gradient ascent (Kingma & Welling 2013).

#### RSSM transition (Section 3, Eq. 4)

The RSSM splits the latent into deterministic $h_t$ and stochastic $s_t$:

$$
\begin{aligned}
\text{Deterministic state:} & \quad h_t = f(h_{t-1}, s_{t-1}, a_{t-1}), \\
\text{Stochastic state (prior):} & \quad s_t \sim p(s_t \mid h_t), \\
\text{Observation:} & \quad o_t \sim p(o_t \mid h_t, s_t), \\
\text{Reward:} & \quad r_t \sim p(r_t \mid h_t, s_t).
\end{aligned}
$$

Here $f$ is a GRU. The encoder is changed accordingly to
$q(s_{1:T} \mid o_{1:T}, a_{1:T}) = \prod_t q(s_t \mid h_t, o_t)$, a
diagonal Gaussian whose mean and variance are produced by a CNN feature
extractor on $o_t$ concatenated with $h_t$. **Crucially, all observation
information must pass through the sampling step of the encoder** — the
deconvolution decoder reads from $(h_t, s_t)$, but $s_t$ is the only
quantity that observations reach via $q$, so the model cannot "cheat" by
piping pixels into reconstructions through a deterministic shortcut.

#### Multi-step prediction and the d-step bound (Section 4, Eq. 5–6)

Define the $d$-step prior by marginalising over intermediate states:

$$
p(s_t \mid s_{t-d}) \;\triangleq\;
\int \prod_{\tau=t-d+1}^{t} p(s_\tau \mid s_{\tau-1})\; ds_{t-d+1:t-1}
\;=\; \mathbb{E}_{p(s_{t-1}\mid s_{t-d})}\big[p(s_t \mid s_{t-1})\big].
$$

For $d = 1$ this recovers the one-step transition. The corresponding
$d$-step variational bound replaces the one-step KL by a $d$-step KL:

$$
\ln p_d(o_{1:T})
\;\ge\;
\sum_{t=1}^{T}\Big(\mathbb{E}_{q(s_t\mid o_{\le t})}\!\left[\ln p(o_t \mid s_t)\right]
- \mathbb{E}_{p(s_{t-1}\mid s_{t-d})\,q(s_{t-d}\mid o_{\le t-d})}\!\Big[\mathrm{KL}\!\left[q(s_t\mid o_{\le t})\,\|\,p(s_t \mid s_{t-1})\right]\Big]\Big).
$$

The data-processing inequality argument is straightforward: in a Markov
chain, $I(s_t; s_{t-d}) \le I(s_t; s_{t-1})$ for $d \ge 1$, so
$\mathbb{E}[\ln p_d(o_{1:T})] \le \mathbb{E}[\ln p(o_{1:T})]$ — the
$d$-step bound is *also* a (looser) bound on the original log-likelihood.
This justifies summing or averaging over multiple $d$.

#### Latent overshooting (Section 4, Eq. 7)

Average the bounds over distances $d = 1, \dots, D$ with weights $\beta_d$:

$$
\frac{1}{D}\sum_{d=1}^{D} \ln p_d(o_{1:T})
\;\ge\;
\sum_{t=1}^{T}\Bigg(\mathbb{E}_{q(s_t\mid o_{\le t})}\!\left[\ln p(o_t\mid s_t)\right]
\;-\; \frac{1}{D}\sum_{d=1}^{D} \beta_d\,
\mathbb{E}_{p(s_{t-1}\mid s_{t-d})\,q(s_{t-d}\mid o_{\le t-d})}\!\Big[\mathrm{KL}\!\left[q(s_t\mid o_{\le t})\,\|\,p(s_t\mid s_{t-1})\right]\Big]\Bigg).
$$

In practice gradients to the posterior are stopped for $d > 1$ — the
multi-step priors are pushed toward the *informed* one-step posteriors,
not the other way around, so the encoder stays anchored to actual
observations. The reconstruction term is unchanged: only the KL is
multi-step. This makes latent overshooting a (relatively) cheap
regulariser — the cost is $D$ extra prior rollouts and KL evaluations in
latent space, no extra pixel decodes.

#### CEM planning in latent space (Section 2, Algorithm 2)

Maintain a Gaussian over future actions $a_{t:t+H} \sim \mathcal{N}(\mu_{t:t+H}, \mathrm{diag}(\sigma^2_{t:t+H}))$, initialised at zero mean and unit
variance. For $I$ iterations:
1. Sample $J$ candidate sequences $\{a^{(j)}_{t:t+H}\}_{j=1}^{J}$.
2. Roll the RSSM forward: starting from a posterior sample $s_t \sim q(s_t \mid o_{\le t})$, repeatedly sample $s_{\tau+1} \sim p(s_{\tau+1} \mid h_{\tau+1})$ and compute $\hat r_\tau = \mathbb{E}[p(r_\tau \mid h_\tau, s_\tau)]$. Score by $R^{(j)} = \sum_{\tau=t}^{t+H-1} \hat r_\tau$.
3. Pick the elite top-$K$ by $R$, refit $\mu_{t:t+H}, \sigma_{t:t+H}$ to
   their empirical mean and variance.

After $I$ iterations execute $\mu_t$, observe $o_{t+1}$, and replan from
fresh $\mathcal{N}(0, I)$ — the warm-start is *deliberately discarded* to
avoid local optima. Hyperparameters: $H = 12$, $I = 10$, $J = 1000$, $K = 100$. Because reward decodes directly from $(h_t, s_t)$, the planner never
materialises pixels — large CEM populations are cheap.

#### Why both stochastic and deterministic paths are essential (Section 5, Fig. 4)

The ablation is the main empirical defence of the RSSM. With a purely
deterministic GRU dynamics, planning fails on every task — the planner
exploits modelling errors that the model has no way to express as
uncertainty. With a purely stochastic SSM, planning fails because the model
forgets the cart's position once it leaves the camera frame. Only the
RSSM's split — memory in $h_t$, hedging in $s_t$ — works. The argument is
half empirical and half theoretical: a stochastic transition can in
principle collapse $\sigma \to 0$ to imitate a deterministic recurrence, but
in practice the optimiser does not find that solution.

### Relevance to this project (interoceptive-pain RL)

- **RSSM as the canonical recurrent latent for partial observability.**
  The repo's RPPO baselines use a recurrent policy on raw observations.
  An RSSM-style separation of $h_t$ (long-horizon memory) and $s_t$
  (per-step uncertainty) is the natural extension if the project ever needs
  an explicit forward model — e.g. a "predicted-interoceptive-state" head
  whose KL to actual interoceptive observations is itself an alarm signal.
- **Latent overshooting as a multi-step prediction-error signal.**
  The $d$-step KL framework is a clean way to score how stable an
  interoceptive prediction is over the next several steps — relevant if
  the project wants to extend the "prediction" channel of RPPO from one-step
  to multi-step, especially under noisy observations.
- **CEM planning as a baseline against learned policies.** When the
  project's analyses (e.g. the dreamer-vs-RPPO speed profiling in
  `tmp/20260506_185353_dreamer_v3_vs_rppo_profile`) need a "planning
  ceiling," CEM with the true simulator (PlaNet's Table 1) shows the
  pattern: report (i) CEM with learned model, (ii) CEM with true simulator,
  (iii) learned policy. The middle line bounds how much of the gap is
  representation vs. planning.
- **No value function, no actor.** The fact that PlaNet ignores returns
  beyond the planning horizon is the explicit gap that DreamerV1 closes —
  the project's choice of whether to add an actor-critic on top of any
  imagined rollout is the same design decision in miniature.
- **Reconstruction-based representation learning.** PlaNet's pixel ELBO ties
  the latent to image detail. For interoceptive signals (lower-dim, higher
  signal-to-noise) this is less of a problem than for pixels, but it is
  still worth checking whether reconstruction-free objectives (contrastive,
  CPC) match RSSM under interoceptive observations — a candidate direction
  to flag for `professor-rl-bayesian-dl`.

---

## <a id="paper-3"></a>Paper 3 — DreamerV1 (Hafner et al. 2020)

**Citation:** Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). *Dream
to Control: Learning Behaviors by Latent Imagination.* ICLR 2020.

**PDF:** `docs/project/references/Dreamer/sources/Hafner et al. 2020 - Dream to control - Learning behaviors by latent imagination.pdf` (20 pages)

### Backbone

- **Problem.** PlaNet's online CEM planning has two structural weaknesses:
  (i) it ignores all reward beyond the planning horizon $H$, so policies are
  shortsighted on tasks like Acrobot or Hopper that need long credit
  assignment; (ii) it discards the world model's *differentiability* by
  using gradient-free CEM. Earlier model-based work that *did* use analytic
  gradients (DDPG, SAC) only used immediate Q-value gradients, not gradients
  through learned dynamics. Dreamer aims to combine PlaNet's RSSM data
  efficiency with proper long-horizon credit assignment via reparameterised
  gradients flowing through learned dynamics.
- **Method — three components, run interleaved (Algorithm 1).**
  1. **Dynamics learning.** Reuse PlaNet's RSSM, trained on real episodes
     stored in a replay buffer. The world model is fixed during behaviour
     learning. Three representation-learning variants are studied: pixel
     reconstruction (the default, inheriting PlaNet's ELBO), contrastive
     (NCE) prediction, and reward-only prediction.
  2. **Behaviour learning by latent imagination.** From every model state
     $s_t$ on a sampled real chunk, branch an *imagined* trajectory
     $\{(s_\tau, a_\tau, r_\tau)\}_{\tau=t}^{t+H}$ purely in latent space:
     actions from $a_\tau \sim q_\phi(a_\tau \mid s_\tau)$, next state
     $s_{\tau+1} \sim q_\theta(s_{\tau+1}\mid s_\tau, a_\tau)$, reward
     $\hat r_\tau = \mathbb{E}[q_\theta(r_\tau \mid s_\tau)]$. Train an
     **action model** $q_\phi(a_\tau\mid s_\tau)$ and a **value model**
     $v_\psi(s_\tau)$ on these imagined trajectories with $\lambda$-returns
     and reparameterisation gradients (Section 3, Eqs. 2–8).
  3. **Environment interaction.** Use the action model with exploration
     noise to add real episodes to the dataset.
- **Key innovations.**
  - $\lambda$-return $V_\lambda$ as a bias-variance-balanced multi-step
    target — bootstraps via $v_\psi$ at horizon $H$ so that the policy can
    care about returns *beyond* $H$.
  - Reparameterised actions $a_\tau = \tanh(\mu_\phi(s_\tau) + \sigma_\phi(s_\tau)\,\varepsilon)$, $\varepsilon \sim \mathcal{N}(0, I)$, so
    $\nabla_\phi V_\lambda$ flows analytically back through the dynamics,
    rewards, and value model — the *core* algorithmic contribution.
  - Discount-factor head: in tasks with early termination, the world model
    additionally predicts a per-state discount, so policy and value updates
    are downweighted by the cumulative product of predicted survival
    probabilities (Section 3, "tasks with early termination").
- **Results** (Section 6, Figs. 4, 6, 7, 8; Appendix C).
  - **20 DM-Control tasks from pixels.** Dreamer reaches an average score of
    823 over 20 tasks in 5×10⁶ steps, beating PlaNet (332) and matching/
    exceeding D4PG (786 at 10⁸ steps) — and 8× wall-clock faster than
    PlaNet (~3 hours vs ~11 hours per 10⁶ steps), 8× faster than D4PG.
  - **Long-horizon ablation (Fig. 4, 7).** Dreamer is robust to imagination
    horizon $H \in [10, 40]$ on Cartpole-Swingup, Cheetah-Run, Quadruped-
    Walk, Walker-Walk, while a *no-value* variant (only $V_R$ from Eq. 4)
    and PlaNet's online CEM both collapse at short $H$. The value head is
    what unlocks Acrobot Swingup, Cartpole Swingup Sparse, Hopper, and
    Pendulum Swingup — tasks PlaNet cannot solve.
  - **Representation learning (Fig. 8).** Pixel reconstruction wins on
    most tasks, contrastive NCE solves about half, reward-only fails. The
    paper frames this as a clean separability claim: representation
    learning is *orthogonal* to the actor-critic-on-imagination machinery,
    so future representation advances should drop in directly.
  - **Discrete control (Appendix C).** Same recipe with
    straight-through gradients on a categorical action head, $H = 10$,
    $\beta = 0.1$ KL scale, tanh-bounded rewards. Applied to a subset of
    Atari and DM-Lab — not yet SOTA, but a feasibility demonstration that
    DreamerV2 will exploit.
- **Limitations** (Section 7, Conclusion).
  - Reconstruction-based RSSM still ties latents to pixel detail; performs
    worse on visually complex domains.
  - Single-Gaussian latents constrain hedging (the paper inherits PlaNet's
    factored-Gaussian $s_t$ — the discrete-latent insight comes in
    DreamerV2).
  - The action model is a tanh-Gaussian for continuous tasks, which has
    known issues at the action boundaries; the paper uses a scaling-by-5
    trick on the mean to allow saturation.
  - Reparameterisation through long imagined trajectories can be unstable
    in principle (the chaotic-gradients problem of Parmas et al. 2019);
    the paper observes empirically robust training but does not claim a
    theoretical fix.

### Phase 1 — Foundational synthesis

DreamerV1 starts from PlaNet's setup — a pixel-input recurrent
state-space model (RSSM) trained on a replay buffer — and replaces
PlaNet's slow online CEM planner with a *learned* policy and *learned*
value function trained inside the dream. The motivation is intuitive: if
you have built a differentiable simulator inside your network, why throw
that away and use derivative-free search? Push the gradient *through* the
simulator into the policy.

The actor-critic loop runs entirely in the RSSM's latent space. Sample a
batch of real chunks from the replay buffer; for each step in each chunk,
imagine an $H$-step rollout: at each imagined step, the action model
samples an action from a Gaussian, the transition model samples the next
latent state, and the reward and value models read off rewards and value
estimates. The clever bit is the *return target*. Sutton-style $V_R$
(just sum the rewards within the horizon) leaves the agent shortsighted:
acrobot needs many seconds of swinging to get above the bar, and that is
beyond any reasonable $H$. So Dreamer uses an $n$-step value estimator
$V_N^k$ that bootstraps via the value model after $k$ steps, then averages
those for $k = 1, \dots, H$ with exponentially decaying weights — the
standard $\lambda$-return. The point is that the *value function*
absorbs reward beyond the horizon, while the policy gets cleaner gradients
from the multi-step targets.

The actor-critic objectives are then standard. The value head regresses
the $\lambda$-return at every imagined state (with a stop-gradient so the
target stays fixed). The action head maximises the $\lambda$-return —
*via reparameterised gradients*. This last phrase is the point of the
paper. Because actions are reparameterised, $a_\tau = \tanh(\mu + \sigma\,\varepsilon)$, and the dynamics, reward, and value are all neural
networks, you can write down $\nabla_\phi V_\lambda$ analytically and
backpropagate through every step of the rollout. No REINFORCE-style
high-variance estimator. No baseline correction (the value head still
helps via the bootstrap, but it is not playing the variance-reduction
role that A3C/PPO use it for). Just clean reparameterisation gradients
through the dynamics.

Why does this matter? Empirically: DreamerV1 *catches up to PlaNet's data
efficiency and exceeds D4PG's asymptotic performance*, while running 8×
faster wall-clock than PlaNet (no online CEM at every step) and 8× faster
than D4PG (no 10⁸-step training run). Conceptually: it is the first
strong demonstration that gradient-through-learned-dynamics can solve
visual continuous control at scale. Earlier attempts (SVG, ME-TRPO,
DistGBP) had been confined to small problems, partly because of the
"chaotic gradients" worry — that gradients through long stochastic
rollouts blow up. Dreamer's $H = 15$ horizon, RSSM stochasticity, and
$\lambda$-return mixing seem to keep this in check on DM-Control.

The other big move is the *separability* claim about representation
learning. The actor-critic-on-imagination algorithm is independent of how
the world model learns its latents. The paper studies three options:
pixel reconstruction (PlaNet's ELBO), contrastive NCE prediction (CPC-
flavoured), and reward-only prediction. Reconstruction wins; contrastive
solves about half the tasks; reward-only collapses on most. The framing
is forward-looking: as representation learning improves, drop the new
recipe into Dreamer and you should improve. (DreamerV2 cashes this in by
swapping the latent for a discrete one.)

DreamerV1 is also where the modern Dreamer training loop — interleaved
dynamics-learning steps, imagination-based behaviour-learning steps, and
short bursts of environment interaction — gets fixed in its current form.
Algorithm 1 of this paper is, with minor tweaks, still the algorithm of
DreamerV2 and DreamerV3.

### Phase 2 — Graduate-level deep dive

#### Notation and the imagination MDP (Section 3)

Real environment time steps are $t$; *imagined* time steps within an
$H$-step rollout are $\tau \in \{t, t+1, \dots, t+H\}$. The world model is
trained as in PlaNet:

$$
\begin{aligned}
\text{Representation:} & \quad p_\theta(s_t \mid s_{t-1}, a_{t-1}, o_t),\\
\text{Transition:} & \quad q_\theta(s_t \mid s_{t-1}, a_{t-1}),\\
\text{Reward:} & \quad q_\theta(r_t \mid s_t).
\end{aligned}
$$

Notation note: Dreamer uses $p$ for posteriors that *do* see observations
and $q$ for the prior that does not — opposite of the typical VAE
convention. The latent $s_t$ here is the full RSSM state, i.e. a
deterministic component plus a stochastic 30-dim diagonal Gaussian
(Appendix A).

Imagination starts at a real model state $s_t \sim p_\theta(s_t \mid s_{t-1}, a_{t-1}, o_t)$ obtained by running the encoder forward on a real
chunk, then samples forward purely from the prior:

$$
a_\tau \sim q_\phi(a_\tau \mid s_\tau), \qquad
s_{\tau+1} \sim q_\theta(s_{\tau+1}\mid s_\tau, a_\tau), \qquad
r_\tau \sim q_\theta(r_\tau\mid s_\tau).
$$

#### Action model and reparameterisation (Section 3, Eq. 3)

The action model is a tanh-squashed diagonal Gaussian:

$$
a_\tau = \tanh\!\big(\mu_\phi(s_\tau) + \sigma_\phi(s_\tau)\,\varepsilon\big),
\qquad \varepsilon \sim \mathcal{N}(0, I),
$$

where $\mu_\phi$ has a $\times 5$ scaling so the saturated tanh region is
reachable (Appendix A). For *discrete* actions (Atari, DM-Lab) the head
emits categorical logits and uses straight-through gradients
(Bengio et al. 2013) for the sampling step.

#### Three return estimators (Section 3, Eqs. 4–6)

Define imagined trajectory returns from state $s_\tau$ with horizon $H$.
The naive horizon-truncated sum,

$$
V_R(s_\tau) \;\triangleq\; \mathbb{E}_{q_\theta, q_\phi}\!\left[\sum_{n=\tau}^{t+H} r_n\right],
$$

ignores reward beyond $t + H$. The *$k$-step bootstrapped* estimate
adds a value-model bootstrap at time $h = \min(\tau + k, t + H)$:

$$
V_N^k(s_\tau)
\;\triangleq\; \mathbb{E}_{q_\theta, q_\phi}\!\left[\sum_{n=\tau}^{h-1} \gamma^{n-\tau}\, r_n + \gamma^{h-\tau}\, v_\psi(s_h)\right],
\qquad h = \min(\tau + k, t + H).
$$

The *$\lambda$-return* mixes these with exponentially-decaying weights
$\lambda^{k-1}$:

$$
V_\lambda(s_\tau)
\;\triangleq\; (1 - \lambda)\sum_{n=1}^{H-1} \lambda^{n-1}\, V_N^n(s_\tau) \;+\; \lambda^{H-1}\, V_N^H(s_\tau).
$$

This is exactly the GAE-style $\lambda$-return adapted to imagined
rollouts. The Dreamer hyperparameter setting is $\gamma = 0.99$,
$\lambda = 0.95$, $H = 15$ (Appendix A).

#### Actor and critic objectives (Section 3, Eqs. 7–8)

The actor maximises the imagination-averaged $\lambda$-return:

$$
\max_\phi \; \mathbb{E}_{q_\theta, q_\phi}\!\left[\sum_{\tau=t}^{t+H} V_\lambda(s_\tau)\right].
$$

The critic regresses $V_\lambda$ at every imagined state, with the target
treated as constant (stop-gradient):

$$
\min_\psi \; \mathbb{E}_{q_\theta, q_\phi}\!\left[\sum_{\tau=t}^{t+H} \tfrac{1}{2}\,\big(v_\psi(s_\tau) - \mathrm{sg}\,V_\lambda(s_\tau)\big)^2\right].
$$

The actor's gradient is computed *analytically*: each $V_\lambda$ depends
on rewards and bootstrap values, which depend on imagined states, which
depend on imagined actions, which depend on $\phi$ via reparameterisation.
The full chain is

$$
\nabla_\phi \mathbb{E}\!\left[\sum_\tau V_\lambda(s_\tau)\right]
= \mathbb{E}\!\left[ \sum_\tau \frac{\partial V_\lambda(s_\tau)}{\partial s_\tau}\, \frac{\partial s_\tau}{\partial a_{\tau-1}}\, \frac{\partial a_{\tau-1}}{\partial \phi} \right]
+ \cdots,
$$

with the $\cdots$ collecting analogous chains for every $r_n$ in
$V_\lambda$'s expansion. In implementation this is just one
backpropagation through the unrolled RSSM. The world-model parameters
$\theta$ are *frozen* during behaviour learning; only $\phi, \psi$ move.

#### Discount-factor head for early termination (Section 3, brief paragraph)

In tasks where episodes can end early (Atari, DM-Lab, walker-fall), the
world model adds a binary discount head whose target is $\gamma$ when the
real step did not terminate and $0$ when it did. During imagination,
weight every term of Eqs. 7 and 8 by the cumulative product
$\prod_{n=t}^{\tau-1} \hat\gamma_n$ of predicted discounts. This is the
elegant generalisation of "early termination" to model-based imagination:
early-termination probability becomes a smooth modulator of the
$\lambda$-return rather than a hard truncation.

#### Information-bottleneck derivation of the world-model loss (Appendix B, Eqs. 13–16)

The paper rederives PlaNet's reconstruction ELBO and the contrastive
alternative from a single information-bottleneck objective:

$$
\max_\theta \; I(s_{1:T}; (o_{1:T}, r_{1:T}) \mid a_{1:T}) \;-\; \beta\, I(s_{1:T}; i_{1:T} \mid a_{1:T}),
$$

where $i_t$ is a dataset index (so $p(o_t \mid i_t)$ is a delta on the
observed image). Lower-bounding the predictive term by KL non-negativity
yields the reconstruction loss

$$
\mathcal{J}_{\mathrm{REC}}
= \mathbb{E}\!\left[\sum_t \big(\ln q_\theta(o_t\mid s_t) + \ln q_\theta(r_t \mid s_t) - \beta\,\mathrm{KL}\!\left[p_\theta(s_t\mid s_{t-1}, a_{t-1}, o_t)\,\|\, q_\theta(s_t\mid s_{t-1}, a_{t-1})\right]\big)\right].
$$

For the *contrastive* objective, replace observation reconstruction by
predicting the state from the image, $q_\theta(s_t \mid o_t)$, and apply
the InfoNCE bound (Poole et al. 2019) over the mini-batch of observations:

$$
\mathcal{J}_{\mathrm{NCE}}
= \mathbb{E}\!\left[\sum_t \Big(\ln q_\theta(s_t\mid o_t) - \ln \sum_{o'\in \mathcal{B}} q_\theta(s_t\mid o') + \ln q_\theta(r_t\mid s_t) - \beta\,\mathrm{KL}[\cdot \| \cdot]\Big)\right].
$$

Both objectives keep the same KL-on-the-prior regulariser (the second
term of the IB objective). KL is clipped below 3 free nats and
$\beta = 1$ (continuous), $\beta = 0.1$ (discrete). The unifying
derivation matters because it makes "swap reconstruction for
contrastive" a one-line edit rather than a different algorithm.

#### Why analytic gradients beat REINFORCE here (Section 3, "Comparison to actor critic methods")

A2C/PPO use REINFORCE-style score-function estimators with baselines:

$$
\nabla_\phi J = \mathbb{E}\!\left[\sum_\tau \nabla_\phi \log \pi_\phi(a_\tau\mid s_\tau)\,\big(\hat A_\tau\big)\right].
$$

The variance scales linearly with $H$ and the entropy of the action
distribution. DDPG/SAC use deterministic-policy gradients but only
through the *immediate* Q-value:

$$
\nabla_\phi J_{\mathrm{DDPG}} = \mathbb{E}\!\left[\nabla_a Q(s, a)\big|_{a = \pi_\phi(s)}\, \nabla_\phi \pi_\phi(s)\right].
$$

Dreamer's update is the natural generalisation: backpropagate through *H
steps* of dynamics rather than zero. SVG (Heess et al. 2015) had done
something similar for one step; MVE/STEVE (Feinberg et al. 2018; Buckman
et al. 2018) for multi-step *Q-targets* but still using model-free
updates. Dreamer is the first agent to combine all three: a learned
latent forward model + analytic gradient through it + a value bootstrap
for return-beyond-horizon.

#### Hyperparameters and training loop (Appendix A; Algorithm 1)

- Latent: 30-dim diagonal Gaussian $s$, plus the RSSM deterministic state
  $h$ (size from Hafner et al. 2018).
- Networks: ELU activations, three hidden layers of 300 each for non-conv
  modules; CNN encoder/decoder borrowed from World Models (Ha &
  Schmidhuber 2018).
- Optimisation: Adam, lr $6{\times}10^{-4}$ (world model),
  $8{\times}10^{-5}$ (action and value), gradient norm clipped at 100.
- Sequence batches: $B = 50$ chunks of $L = 50$.
- Imagination: $H = 15$, $\gamma = 0.99$, $\lambda = 0.95$.
- Exploration: $\mathcal{N}(0, 0.3)$ noise added on top of the action mode
  for continuous tasks; $\varepsilon$-greedy linearly $0.4 \to 0.1$ for
  discrete. Action repeat fixed at $R = 2$ across all tasks.
- Per training iteration: 100 update steps interleaved with 1 episode of
  data collection; 5 random seed episodes to bootstrap the buffer.

### Relevance to this project (interoceptive-pain RL)

- **Direct comparator to RPPO continual-learning demo (commit `de102b8`).**
  Dreamer's actor-critic-on-imagination is structurally analogous to the
  prediction-augmented RPPO in the project: both train a recurrent latent
  with a prediction head. Dreamer's $\lambda$-return-on-imagination is
  the natural upgrade if the project ever wants to use the predictor
  for planning rather than only for representation. A focused
  side-by-side experiment (Dreamer vs. RPPO with prediction) is the
  obvious next experiment-designer task — flag for `senior-developer` to
  scope.
- **Reparameterisation through dynamics is sensitive to noise.** The
  paper's "chaotic gradients" caveat (Parmas et al. 2019 ref) is a
  warning to anyone wiring imagined gradients into a policy under noisy
  observations. Pain-modeling work that explicitly inflates observation
  noise should benchmark Dreamer-style gradients against
  $\lambda$-return-only updates without backprop-through-time.
- **Discount-factor head as a survival probability.** The discount head is
  a clean place to plug in interoceptive "death-probability" predictions —
  the value bootstrap is automatically downweighted by predicted survival.
  For survival-step evaluation (the project's primary metric) this is the
  more principled framing than a hard-coded discount.
- **Representation-learning orthogonality.** The Section-4 / Fig-8
  comparison of reconstruction vs. NCE vs. reward-only is a useful
  template for the project: report the same actor-critic on (a)
  reconstruction of pixels + interoception, (b) contrastive prediction,
  (c) reward-only. The project has more interoceptive structure than
  generic DM-Control, so the relative ordering may flip.
- **Action-space saturation trick.** The "$\times 5$ scaling on the
  Gaussian mean before tanh" detail (Appendix A) is a known
  implementation-level fix for tanh-squashed continuous policies; worth
  noting if the project ever wires Dreamer-style policies into the
  current continuous-action environments.

---

## <a id="paper-4"></a>Paper 4 — DreamerV2 (Hafner et al. 2021)

**Citation:** Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). *Mastering Atari with Discrete World Models.* International Conference on Learning Representations (ICLR), arXiv:2010.02193.

**PDF:** `docs/project/references/Dreamer/sources/Hafner et al. 2021 - Mastering Atari with Discrete World Models.pdf` (26 pages)

### Backbone

- **Problem.** DreamerV1 mastered DeepMind Control Suite from pixels but no
  prior model-based agent had matched human-gamer performance on the full
  55-game Atari benchmark at 200M frames. Dyna-style world-model agents
  (PlaNet, DreamerV1, SimPLe) used Gaussian latents that struggle to fit the
  multi-modal, sparse, and abruptly-changing dynamics of arcade games (room
  transitions, item disappearance, enemy state flips); MuZero achieved strong
  Atari results only with MCTS, vast compute, and closed-source code. The goal
  is a single-GPU model-based agent that learns a world model accurate enough
  to derive Atari behaviours purely inside imagination.
- **Method — DreamerV2 = DreamerV1 + discrete latents + KL balancing**
  (Section 2; Figs. 2–3; Appendix C lists the modifications).
  - **RSSM with categorical latents.** Same factorisation as PlaNet —
    deterministic recurrent state $h_t$ from a GRU plus stochastic latent
    $z_t$ — but $z_t$ is now a vector of **32 categorical variables, each a
    one-hot over 32 classes** (1024 binary bits, 32 active). The five
    components are written explicitly as

    $$
    h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}),\;
    z_t \sim q_\phi(z_t\mid h_t, x_t),\;
    \hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t),\;
    \hat{x}_t \sim p_\phi(\hat{x}_t\mid h_t, z_t),\;
    \hat{r}_t,\hat{\gamma}_t.
    $$
  - **Straight-through gradients.** One-hot samples have no gradient, so the
    paper uses $\text{sample} = \text{sample} + \text{probs} -
    \text{stop\_grad}(\text{probs})$ (Algorithm 1) to attach the
    straight-through gradient of the softmax probabilities.
  - **KL balancing.** The ELBO KL term $\text{KL}(q \,\Vert\, p)$ is split
    into two stop-gradient halves so the prior is updated with weight
    $\alpha = 0.8$ and the posterior with weight $1-\alpha = 0.2$
    (Algorithm 2). This pushes $p$ toward $q$ faster than it pushes $q$
    toward $p$, avoiding the failure mode of regularising the posterior
    against an under-trained prior.
  - **Actor-critic in imagination.** Horizon-15 rollouts under $p_\phi$ feed
    a $\lambda$-target ($\lambda=0.95$) critic and an actor whose objective
    blends Reinforce (with value baseline) and dynamics backprop via mixing
    coefficient $\rho$, plus an entropy bonus $\eta\,H[a_t\mid \hat{z}_t]$.
    Atari uses pure Reinforce ($\rho=1$, $\eta=10^{-3}$); continuous control
    uses pure dynamics backprop ($\rho=0$, $\eta=10^{-5}$).
  - **Discount predictor.** A Bernoulli head $\hat\gamma_t$ replaces the
    fixed $\gamma$ at terminal states so the imagined critic learns to
    down-weight rewards past episode ends.
- **Key equations.**
  - World-model loss (Eq. 2):
    $\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\!\left[\sum_t -\ln p_\phi(x_t\mid h_t,z_t) -\ln p_\phi(r_t\mid h_t,z_t) -\ln p_\phi(\gamma_t\mid h_t,z_t) + \beta\,\text{KL}\!\big(q_\phi(z_t\mid h_t,x_t)\,\Vert\,p_\phi(z_t\mid h_t)\big)\right]$
    with $\beta=0.1$ for Atari and $\beta=1$ for continuous control.
  - $\lambda$-return (Eq. 4):
    $V^\lambda_t = \hat r_t + \hat\gamma_t\!\big[(1-\lambda)v_\xi(\hat z_{t+1}) + \lambda V^\lambda_{t+1}\big]$ for $t<H$,
    $V^\lambda_H = v_\xi(\hat z_H)$.
  - Actor loss (Eq. 6):
    $\mathcal{L}(\psi) = \mathbb{E}\!\left[\sum_t -\rho\,\ln p_\psi(\hat a_t\mid \hat z_t)\,\text{sg}(V^\lambda_t - v_\xi(\hat z_t)) - (1-\rho)\,V^\lambda_t - \eta\,H[a_t\mid \hat z_t]\right]$.
  - KL balancing (Algorithm 2):
    $\text{kl} = \alpha\,\text{KL}(\text{sg}(q)\,\Vert\,p) + (1-\alpha)\,\text{KL}(q\,\Vert\,\text{sg}(p))$.
- **Results** (Section 3; Tables 1–2; Figs. 4–5).
  - First model-based agent to surpass human-gamer median on Atari-200M
    sticky-action with a single GPU and a single environment instance.
    Gamer-median 2.15, gamer-mean 11.33, clipped-record-mean 0.28 — beats
    Rainbow (1.47/9.12/0.17) and IQN (1.29/8.85/0.21) at the same wall-clock
    (~10 days).
  - 468B imagined model states per 200M real steps (10⁴× ratio).
  - Ablations (Table 2): removing image gradients collapses performance
    (0.04 gamer median); removing categorical latents drops to 1.08; removing
    KL balancing drops to 0.84; removing Reinforce drops to 0.69. Stopping
    *reward* gradients into the encoder *helps slightly* on average — image
    reconstruction alone is enough to learn useful representations.
  - Continuous control bonus (Appendix A): solves Humanoid-Walk from 64×64
    pixels — first published agent to do so.
  - Hard-exploration bonus (Appendix B): matches Rainbow+ICM on Montezuma's
    Revenge with $\gamma=0.99$, no explicit exploration bonus.
- **Limitations.**
  - Reconstruction loss can fail when the task-relevant object occupies
    very few pixels (Video Pinball failure mode discussed in Section 3.1).
  - Per-task hyper-parameters (β, η, γ schedule) still differ between
    Atari and DM-Control — the "single set of hyperparameters" goal is left
    to DreamerV3.
  - Discrete latents add a discrete-sampling bias that the straight-through
    estimator only partially corrects; the paper offers four hypotheses
    (Section 3.2) but no decisive theory.

### Phase 1 — Foundational synthesis

DreamerV2 is the first model-based RL agent to match human Atari performance
on a single GPU, and it does so by changing two ingredients of DreamerV1
rather than rebuilding the algorithm. The world-model backbone is still the
RSSM from PlaNet — a deterministic GRU state $h_t$ paired with a stochastic
latent $z_t$ — and the behaviour-learning loop is still actor-critic on
$\lambda$-returns inside imagined trajectories. The two changes are
(i) replacing the Gaussian latent with a vector of categorical (one-hot)
variables, and (ii) splitting the KL term of the ELBO so the prior is pulled
toward the posterior faster than the posterior is pulled toward the prior.
Both choices respond to specific failure modes of Gaussian RSSMs on Atari:
arcade games have abrupt, discrete events (rooms switch, sprites disappear,
lives reset) that a single Gaussian cannot represent, and a poorly-trained
Gaussian prior can drag an informative posterior toward a useless mean.

The categorical latent is implemented as 32 separate one-hot vectors of size
32, giving a sparse 1024-bit code with exactly 32 active bits per timestep.
Sampling a one-hot is non-differentiable, so DreamerV2 uses the
**straight-through estimator** — the forward sample is the discrete one-hot,
but the backward gradient flows through the underlying softmax probabilities.
This three-line trick makes discrete latents work with ordinary
auto-differentiation and is one of the most-cited operational details of the
paper.

The second change, **KL balancing**, recognises that the standard ELBO KL
term $\text{KL}(q \Vert p)$ has two simultaneous effects: it teaches the
prior $p$ to mimic the posterior $q$ (good — that is what we want for
imagination), and it teaches the posterior to back off toward the prior
(bad early in training when $p$ is uninformative). Hafner et al. attach a
stop-gradient to half of the KL term and use $\alpha=0.8$ for the
prior-update half and $1-\alpha=0.2$ for the posterior-regularisation half.
The prior thus learns four times faster than the posterior is regularised,
yielding a tighter aggregate-posterior fit and smoother imagined rollouts.

The downstream actor-critic is also modified. DreamerV1 used pure dynamics
backpropagation through the reparameterised Gaussian latents; with discrete
actions and discrete latents this gradient is biased, so DreamerV2 mixes
**Reinforce** (unbiased, high variance) with straight-through dynamics
backprop, controlled by a mixing coefficient $\rho$. Atari uses pure
Reinforce ($\rho=1$); continuous control uses pure dynamics backprop
($\rho=0$). An entropy bonus on the actor distribution replaces the external
exploration noise of DreamerV1.

The headline result is a gamer-median of 2.15 across the 55 sticky-action
Atari games at 200M frames on a single V100 — better than Rainbow and IQN at
the same compute. The ablations cleanly attribute most of the gain to
discrete latents and KL balancing, with image-reconstruction gradients being
the *single most important* learning signal: removing them collapses scores
by two orders of magnitude. Reward gradients into the encoder, by contrast,
help only marginally — sometimes hurt — suggesting the world model can learn
generally-useful state from images alone.

### Phase 2 — Graduate-level deep dive

**Categorical RSSM and the straight-through estimator.** The DreamerV2 RSSM
keeps the deterministic GRU recurrence $h_t = f_\phi(h_{t-1}, z_{t-1},
a_{t-1})$ but redefines the stochastic latent as

$$
z_t = (z_t^{(1)}, \ldots, z_t^{(32)}), \quad z_t^{(i)} \in \{e_1, \ldots, e_{32}\}
\subset \mathbb{R}^{32},
$$

a 32-tuple of one-hot vectors. The posterior $q_\phi(z_t \mid h_t, x_t)$
emits per-component categorical logits $\ell_t^{(i)} \in \mathbb{R}^{32}$;
the prior $p_\phi(\hat z_t \mid h_t)$ emits its own logits. A draw from
either distribution yields a sparse 1024-bit vector with exactly 32 active
bits — a structure that, the authors hypothesise (Section 3.2), is closer
to the multi-modal, piecewise-constant nature of arcade dynamics than any
unimodal Gaussian. They also argue that a *categorical prior can perfectly
match the aggregate categorical posterior* (a mixture of categoricals is a
categorical), whereas a Gaussian prior cannot match a Gaussian-mixture
posterior — a structural advantage for the reparameterised KL.

Sampling $z_t$ is non-differentiable, but the **straight-through (ST)
gradient** (Bengio et al., 2013) gives a usable — though **biased** —
gradient through the sampling step, via the identity in Algorithm 1:

$$
\tilde z_t \;=\; \text{onehot}(\text{draw}(\ell_t)) \;+\; \text{softmax}(\ell_t)
\;-\; \text{sg}(\text{softmax}(\ell_t)).
$$

In the forward pass $\tilde z_t = \text{onehot}(\text{draw}(\ell_t))$ exactly
(the last two terms cancel); in the backward pass the only non-zero term is
$\text{softmax}(\ell_t)$, so $\partial \tilde z_t / \partial \ell_t =
\partial\,\text{softmax}(\ell_t) / \partial \ell_t$. The estimator is biased
(it ignores the contribution of the sampling step to the loss) but has dramatically
lower variance than score-function gradients on per-step losses such as the
image, reward, and discount likelihoods.

> **CORRECTION (2026-07-16).** This passage previously rendered Algorithm 1's
> sampling step as `onehot(argmax(ℓ_t))` and described the ST gradient as
> *unbiased*. Both were wrong, and verified against the PDF
> (`Hafner et al. 2021`, Algorithm 1, p.4):
>
> ```
> sample = one_hot(draw(logits))          # sample has no gradient
> probs  = softmax(logits)                # want gradient of this
> sample = sample + probs - stop_grad(probs)
> ```
>
> The paper says **`draw`** — a *sample* from `Categorical(softmax(ℓ))` — not
> `argmax`, which is the deterministic *mode*. The difference is load-bearing:
> `argmax` always returns the top class, whereas `draw` returns the 9.9%-class
> 9.9% of the time, and that stochasticity is what makes imagined rollouts
> diverse. The paper also states plainly: *"This results in a **biased** gradient
> estimate with low variance."* (§ Discrete latents.)
>
> Note the paper leaves **`draw` unspecified**. sheeprl fills it with PyTorch's
> multinomial sampler (`OneHotCategoricalStraightThrough`); our JAX port fills it
> with Gumbel-max (`agent.py:915`). Both are exact samplers; see D-014-adjacent
> reasoning in `docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md` D-009.
> **No Dreamer paper mentions Gumbel** (verified: 0 hits across the 2021, 2023,
> and 2025 PDFs) — the papers cite Bengio et al. 2013 for straight-through, not
> Jang/Maddison for Gumbel-Softmax. Gumbel-Softmax is a *different* technique
> (a continuous relaxation, `softmax((ℓ+g)/τ)`), which Dreamer does **not** use.
> See [[dreamer_qa_log]] Q19.

**KL balancing as two stop-gradient halves.** The ELBO regulariser
$\text{KL}(q \Vert p) = \mathbb{E}_q[\log q - \log p]$ has gradient
contributions from both $q$ and $p$. Decomposing,

$$
\nabla_{q,p}\text{KL}(q\Vert p) \;=\; \underbrace{\nabla_p\,\mathbb{E}_q[-\log p]}_{\text{train prior toward }q}
\;+\;\underbrace{\nabla_q\,\mathbb{E}_q[\log q - \log p]}_{\text{regularise }q\text{ toward }p}.
$$

Hafner et al. give each half a separate weight via stop-gradient (Algorithm 2):

$$
\mathcal{L}_{\text{KL}} \;=\; \alpha\,\text{KL}\!\big(\text{sg}(q)\,\Vert\,p\big)
\;+\;(1-\alpha)\,\text{KL}\!\big(q\,\Vert\,\text{sg}(p)\big),\qquad \alpha = 0.8.
$$

The first term has gradient only with respect to $p$ — the prior moves
toward a *frozen* posterior. The second term has gradient only with respect
to $q$ — the posterior is regularised against a *frozen* prior. With
$\alpha > 0.5$ the prior outpaces the posterior, so the posterior never has
to be sacrificed to a poorly-trained prior. Ablations (Table 2) show this
swing alone moves clipped-record-mean from 0.16 to 0.25.

**Mixed actor gradient.** Let $V^\lambda_t$ denote the $\lambda$-return
(Eq. 4) and $A^\lambda_t = V^\lambda_t - v_\xi(\hat z_t)$ the advantage. The
actor loss (Eq. 6) is

$$
\mathcal{L}(\psi) \;=\; -\mathbb{E}\!\left[
\sum_{t=1}^{H-1}
\rho\,\ln p_\psi(\hat a_t\mid \hat z_t)\,\text{sg}(A^\lambda_t)
\;+\;(1-\rho)\,V^\lambda_t
\;+\;\eta\,H[a_t\mid \hat z_t]
\right].
$$

The first term is the score-function (Reinforce) estimator, unbiased but
high-variance; it does not require gradient flow through the RSSM. The
second term is the dynamics-backprop estimator: $V^\lambda_t$ depends on
$\hat z_{t+1:H}$, which depend on $\hat a_{t:H-1}$ via the prior — a
straight-through path through both the discrete actor sample and the
discrete latent gives a biased but low-variance gradient. The third term is
an entropy bonus that replaces external action noise. The paper finds
$\rho=1$ (pure Reinforce) optimal on Atari; $\rho=0$ (pure dynamics
backprop) optimal on continuous control.

**Critic with target network.** The critic regresses to the $\lambda$-return
under a stop-gradient (Eq. 5):

$$
\mathcal{L}(\xi) \;=\; \tfrac12\,\mathbb{E}\!\left[\sum_{t=1}^{H-1}\big(v_\xi(\hat z_t) - \text{sg}(V^\lambda_t)\big)^2\right],
$$

with $V^\lambda_t$ computed by a target network refreshed every 100 gradient
steps, mirroring DQN-style target stabilisation.

**Hyperparameters (Table D.1).** B=50 sequences of length L=50, dataset 2M
transitions FIFO, RSSM 600 GRU units, KL scale β=0.1 (Atari) / 1 (control),
KL balancing α=0.8, world-model LR 2·10⁻⁴, actor LR 4·10⁻⁵, critic LR 10⁻⁴,
imagination horizon H=15, discount γ=0.995, λ=0.95, entropy η=10⁻³ (Atari) /
10⁻⁵ (control), gradient clip 100, 4 policy steps per gradient step. Total
22M trainable parameters.

**Why image-reconstruction gradients dominate (Section 3.2 ablation).** The
"No Image Gradients" row of Table 2 collapses to clipped-record-mean 0.01 —
two orders of magnitude worse than the next-worst ablation. This says
DreamerV2's policy success is *parasitic on a representation learned from
unsupervised pixel reconstruction*, not from reward prediction. Stopping
reward gradients into the encoder actually helps on 15 tasks. The authors
read this as: representations not specialised to past rewards generalise
better to novel reward configurations — a finding that motivates the
DreamerV3 decision to keep reward and value heads on top of a frozen-style
world-model representation rather than letting them shape it.

### Relevance to this project (interoceptive-pain RL)

- **Discrete latents for interoceptive states.** Categorical latents are an
  attractive substrate for explicitly *symbolic* interoceptive states
  (homeostatic regime, hunger/thirst/pain class, "danger present yes/no")
  that the project may want to expose in its latent code. The 32×32
  one-hot factorisation gives a tractable inductive bias that mirrors
  multi-channel categorical interoception (e.g., one channel per modality).
- **KL balancing to protect a learned interoceptive prior.** If the project
  ever trains an explicit world model with an interoceptive temporal prior,
  KL balancing is the right default — it prevents an early, miscalibrated
  prior from collapsing the encoder's interoceptive representation, which
  is exactly the failure mode "precision-as-prior" agents typically suffer
  in early training.
- **Reward-gradient ablation matters for pain.** DreamerV2's finding that
  *reward-gradient-into-encoder* hurts on many tasks is directly relevant
  to a pain-RL setup: pain is a sparse, high-magnitude, possibly mis-scaled
  signal, and forcing the encoder to specialise to it could harm
  generalisation across episodes. A safer recipe is to keep the encoder
  reconstruction-driven and let pain enter only through reward and value
  heads — the project should re-test this ablation on its interoceptive
  data.
- **Reinforce + entropy on discrete actions.** The grid-world has a small
  discrete action space; DreamerV2's Atari recipe (pure Reinforce, entropy
  bonus, no external noise) is closer to the project's existing PPO/A2C
  setups than DreamerV1's pure dynamics backprop, and is likely the right
  starting point for a Dreamer-style baseline.
- **Image-loss as a foundation, not pain-loss.** Even though the project
  uses small grid-world frames, the same logic applies: rich
  pixel-reconstruction (or contrastive) targets stabilise the latent more
  than a sparse pain reward. The project's existing "PredInt" auxiliary
  losses play this role and should be considered the substitute for
  DreamerV2's image gradients.
- **Hyper-parameter starting points.** The Atari Table-D.1 settings are a
  reasonable initialisation for the first Dreamer-style run on the
  grid-world: B=50, L=50, H=15, λ=0.95, β=0.1, KL-balance α=0.8. The
  project should expect to retune β and η first.

## <a id="paper-5a"></a>Paper 5a — DreamerV3 preprint (Hafner et al. 2023)

**Citation:** Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering Diverse Domains through World Models.* arXiv:2301.04104.

**PDF:** `docs/project/references/Dreamer/sources/Hafner et al. 2023 - Mastering Diverse Domains through World Models.pdf` (38 pages)

### Backbone

- **Problem.** Every prior Dreamer paper required per-benchmark
  hyper-parameter tuning — KL scale, entropy bonus, reward clipping, action
  noise, image-loss weight — and that tuning loop is the practical bottleneck
  for applying world-model RL to new domains. The goal: a *single* set of
  hyperparameters that works across continuous and discrete actions, low-
  and high-dimensional inputs, dense and sparse rewards, 2D and 3D worlds.
  As a stress test, the paper targets the Minecraft-Diamond challenge —
  procedurally generated 3D, 12-step sparse-reward chain, no human data.
- **Method — three robustness ingredients on top of the V2 RSSM.**
  - **Symlog predictions** (Eqs. 1–2). All real-valued targets that the
    network must predict (decoder pixel/proprioceptive output, reward head,
    critic) are first transformed by
    $\text{symlog}(x) = \text{sign}(x)\,\ln(|x|+1)$, the squared loss is
    computed in symlog space, and predictions are read out via the inverse
    $\text{symexp}(x) = \text{sign}(x)(\exp|x|-1)$. The encoder also squashes
    its real-valued inputs by symlog. This removes any need to clip or
    normalise rewards across domains.
  - **Twohot reward and value heads** (Eqs. 8–10). Critic and reward
    predictor are reformulated as discrete classification over $K=255$
    fixed buckets $b_i \in [-20,+20]$ in symlog space. The target — also a
    symlog-transformed return — is encoded as a *twohot* soft label putting
    mass on the two adjacent buckets in proportion to closeness; the
    network minimises categorical cross-entropy. A continuous prediction is
    recovered as $\hat v(s) = \text{symexp}(\sum_i p_i b_i)$. This is a
    minimal form of distributional RL.
  - **Robust return scaling** (Eq. 12). Returns $R^\lambda$ are divided by
    $S = \max(1, \text{Per}(R^\lambda, 95) - \text{Per}(R^\lambda, 5))$ —
    i.e. by the EMA of the **5th–95th percentile range**, but only if that
    range exceeds 1. Large ranges are scaled down; small (sparse-reward)
    ranges are *not* amplified. With this scaling a single entropy coefficient
    $\eta = 3\cdot 10^{-4}$ works across dense and sparse domains.
  - **Free bits + KL balancing combined** (Eq. 5). The world-model loss
    splits into three weighted parts:
    $\mathcal{L} = \beta_{\text{pred}} \mathcal{L}_{\text{pred}}
    + \beta_{\text{dyn}} \mathcal{L}_{\text{dyn}}
    + \beta_{\text{rep}} \mathcal{L}_{\text{rep}}$
    with $\beta_{\text{pred}}=1$, $\beta_{\text{dyn}}=0.5$,
    $\beta_{\text{rep}}=0.1$. Each KL term is separately clipped at 1 nat
    of "free bits". This generalises V2's KL balancing (the asymmetric
    $\alpha$ split is now a per-direction loss with its own weight) and
    re-introduces the "free nats" idea from PlaNet/DreamerV1.
  - **Unimix categoricals.** Posterior and prior categorical distributions
    are 1% uniform + 99% network output to keep KL terms bounded and avoid
    occasional KL spikes seen in pure-categorical training.
  - **Architecture refresh.** Layer normalisation, SiLU activations,
    same-padded conv stride-2 kernel-3, larger networks; per-component
    sizes scale from XS (8M) to XL (200M) parameters.
  - **Critic EMA regulariser** replaces the V2 slow-target network — the
    fast critic is regularised toward its own EMA on the same minibatch.
  - **Subsequence replay.** Replay buffer no longer waits for episode
    completion; subsequences of length $L=64$ are sampled uniformly from
    the entire buffer.
- **Key equations.**
  - $\text{symlog}(x) = \text{sign}(x)\ln(|x|+1)$,
    $\text{symexp}(x) = \text{sign}(x)(\exp|x|-1)$ (Eq. 2).
  - World-model losses (Eqs. 4–5):
    $\mathcal{L}_{\text{pred}}(\phi) = -\ln p_\phi(x_t\mid h_t,z_t) - \ln p_\phi(r_t\mid h_t,z_t) - \ln p_\phi(c_t\mid h_t,z_t)$,
    $\mathcal{L}_{\text{dyn}}(\phi) = \max\big(1,\;\text{KL}(\text{sg}\,q_\phi \,\Vert\, p_\phi)\big)$,
    $\mathcal{L}_{\text{rep}}(\phi) = \max\big(1,\;\text{KL}(q_\phi \,\Vert\, \text{sg}\,p_\phi)\big)$.
  - Twohot critic (Eqs. 8–10):
    $v(s) = \text{symexp}(p^\top B)$;
    $\text{twohot}(x)$ places mass $|b_{k+1}-x|/|b_{k+1}-b_k|$ on bucket $k$
    and the complement on $k+1$;
    $\mathcal{L}_{\text{critic}} = -\sum_t y_t^\top \ln p_\psi(\cdot\mid s_t)$
    with $y_t = \text{sg}(\text{twohot}(\text{symlog}(R^\lambda_t)))$.
  - Actor loss (Eq. 11):
    $\mathcal{L}(\theta) = -\sum_t \mathbb{E}_{\pi,p}\big[\text{sg}(R^\lambda_t)/\max(1,S)\big] - \eta\,H[\pi_\theta(a_t\mid s_t)]$
    with $S = \text{Per}(R^\lambda,95) - \text{Per}(R^\lambda,5)$ (Eq. 12).
- **Results** (pages 8–11; Fig. 1, Fig. 6, appendices L–V).
  - **Single hyperparameter set across 7 benchmarks, 150+ tasks.** New
    state-of-the-art on Proprio Control (18 tasks), Visual Control (20),
    BSuite (23), Crafter (1), DMLab (8). Strong on Atari 100k (26)
    (outperforms IRIS, SPR, SimPLe; below EfficientZero with its tree
    search). New state-of-the-art on Atari 200M (302% gamer median vs.
    DreamerV2's 219%, Rainbow's ~150%).
  - **First-from-scratch Minecraft Diamond.** Across 40 seeds at 100M
    steps, DreamerV3 collects diamonds without any human data or curricula —
    previous methods (VPT, MineRL) required human demonstrations.
  - **Scaling laws (Fig. 6).** Five model sizes XS→XL (8M→200M) trained on
    Breakout, MsPacman, Crafter, DMLab. Larger models give *both* higher
    final score and higher data-efficiency monotonically — counter to the
    common finding that bigger RL networks need more data.
  - **Training-ratio scaling (Fig. 6a).** Increasing replay-to-environment
    ratio from 1 to 64 monotonically improves data-efficiency.
- **Limitations.**
  - The single-set claim is "fixed across 150 tasks given the V100 budget"
    — extreme regimes (tens-of-billions of frames, massive distributed
    training) are not in scope.
  - Twohot symlog buckets are still bounded ($\pm 20$ in symlog space, i.e.
    roughly $\pm \exp(20)$ in raw scale); domains exceeding this require
    re-bucketing.
  - The Minecraft result requires the standard "fast block break" speed-up
    used by prior MineRL work — not a pure-vanilla environment.
  - No theoretical analysis of why the percentile scaling preserves the
    optimal policy under varying $S$.

### Phase 1 — Foundational synthesis

DreamerV3 takes the V2 algorithm and rebuilds the *prediction layer* so the
same network and the same hyper-parameters can master continuous control,
Atari, BSuite, Crafter, DMLab and Minecraft without retuning. The world-model
backbone is unchanged in spirit — RSSM with categorical latents,
straight-through gradients, KL regulariser — but every place where DreamerV2
needed a per-domain knob has been replaced by something scale-invariant.

The first ingredient is **symlog**. Reconstruction targets, reward targets,
and value targets are first transformed by $\text{symlog}(x) =
\text{sign}(x)\,\ln(|x|+1)$ — a logarithm that respects sign and is
near-identity around zero. The squared-error loss is then computed in
symlog space and the prediction is unsquashed at inference. This removes
the need to clip rewards (DQN-style) or to normalise them with running
statistics (PPO-style); large rewards no longer drive the network to
saturating outputs and small rewards are not amplified. The encoder
likewise squashes its inputs.

The second ingredient is the **twohot value/reward head**. Instead of
regressing a scalar with squared error, the critic is a 255-way softmax over
fixed buckets $b_i \in [-20, +20]$ in symlog space; the target is the
*twohot* soft label that linearly interpolates between the two adjacent
buckets. Cross-entropy training is more stable than regression on heavy-
tailed return distributions, and the prediction is recovered as the
symexp of the expected bucket value. This is a minimal flavor of
distributional RL — no quantile regression, no full categorical Bellman —
and the paper finds it especially helpful for sparse-reward domains.

The third ingredient is **percentile return scaling**. Returns are divided
by an EMA of their 5th-to-95th percentile range $S$, but only if $S > 1$.
Sparse-reward returns (small $S$) pass through unchanged, so the entropy
bonus still drives exploration; dense-reward returns (large $S$) are scaled
down so they do not overwhelm the entropy term. With this single trick a
fixed entropy coefficient $\eta = 3\cdot 10^{-4}$ replaces the per-domain
$\eta \in \{10^{-5}, 10^{-3}\}$ split that DreamerV2 needed.

The KL term is also revised. DreamerV2 used a single KL with the asymmetric
$\alpha = 0.8$ split. DreamerV3 keeps that split as two separate losses
($\mathcal{L}_{\text{dyn}}$ and $\mathcal{L}_{\text{rep}}$), gives them
independent loss scales (0.5 and 0.1), and applies a **free-bits floor of
1 nat** to each — the loss is silenced when the KL is already small. This
combination — KL balancing × free bits — is what allows the same encoder to
work for the pixel-rich but mostly-static Atari frame and for the cluttered
3D Minecraft block stream without a per-domain regulariser.

Architecture-wise the network is upgraded to layer-norm + SiLU and
same-padded convolutions, and the categorical encoder/dynamics/actor
distributions are replaced by 1%-uniform + 99%-network mixtures to bound
log probabilities. The replay buffer is changed to subsequence sampling
without waiting for episode completion, shortening the feedback loop.

The headline empirical result is a single set of hyperparameters that beats
specialist baselines on 4 of 7 benchmarks (Proprio Control, Visual Control,
BSuite, Crafter), is competitive on the others, and is the *first* method
ever to collect a Minecraft diamond from scratch. Scaling experiments show
favourable model-size scaling laws — bigger DreamerV3 networks give both
higher final score and higher sample-efficiency, monotonically from 8M to
200M parameters.

### Phase 2 — Graduate-level deep dive

**Symlog as a scale-invariant predictor head.** Define
$\text{symlog}(x) = \text{sign}(x)\ln(|x|+1)$ and its inverse
$\text{symexp}(x) = \text{sign}(x)(\exp|x|-1)$. Both are smooth, bijective,
and approximate the identity near zero ($\text{symlog}(x) \approx x$ for
small $|x|$, with $\text{symlog}'(0) = 1$). For large $|x|$,
$\text{symlog}(x) \approx \text{sign}(x)\ln|x|$, compressing magnitude
logarithmically. The network is trained to predict $\text{symlog}(y)$
under squared loss

$$
\mathcal{L}(\theta) \;=\; \tfrac12\,\big(f(x;\theta) - \text{symlog}(y)\big)^2,
$$

and the readout is $\hat y = \text{symexp}(f(x;\theta))$. The gradient with
respect to $f$ is $f - \text{symlog}(y)$, bounded for any $y$, so reward
spikes no longer cause exploding gradients. This eliminates two prior hacks:
DQN-style reward clipping ($r \leftarrow \tanh r$) which discards
information, and PopArt-style running normalisation which introduces
non-stationarity into the critic target. Symlog is stationary by
construction.

**Twohot symlog critic and the implicit categorical Bellman update.** The
critic outputs a softmax distribution $p_\psi(\cdot\mid s_t) \in \Delta^{K-1}$
over $K=255$ buckets $B = (b_1,\ldots,b_{255})$, equally spaced over
$[-20, +20]$ in symlog space. The point estimate is

$$
v_\psi(s_t) \;=\; \text{symexp}\!\big(\,\mathbb{E}_{p_\psi(\cdot\mid s_t)}[b_i]\,\big)
\;=\; \text{symexp}\!\big(\,p_\psi(\cdot\mid s_t)^\top B\,\big).
$$

For a continuous target $x \in \mathbb{R}$ in symlog space, the twohot
encoding (Eq. 9) places mass on the two buckets adjacent to $x$:

$$
\text{twohot}(x)_i \;=\;
\begin{cases}
(b_{k+1}-x)/(b_{k+1}-b_k) & i = k \\
(x-b_k)/(b_{k+1}-b_k)     & i = k+1 \\
0 & \text{else}
\end{cases},\quad k = \max\{j: b_j < x\}.
$$

The critic loss is categorical cross-entropy

$$
\mathcal{L}_{\text{critic}}(\psi) \;=\; -\sum_{t=1}^{T} y_t^\top \ln p_\psi(\cdot\mid s_t),
\qquad
y_t = \text{sg}\big(\text{twohot}(\text{symlog}(R^\lambda_t))\big).
$$

Because twohot is linear in $x$, the *expected* prediction
$\mathbb{E}_{p_\psi}[b_i]$ converges to $\mathbb{E}[\text{symlog}(R)]$ — i.e.
the critic learns a scalar regression, but via a softmax that smooths the
loss landscape and accommodates bimodal return distributions naturally.
This yields visible gains in sparse-reward domains where returns are
near-zero with rare large spikes.

**Free-bits + KL balancing as decoupled regularisers.** DreamerV3 splits
the V2 KL into two separate losses,

$$
\mathcal{L}_{\text{dyn}}(\phi) = \max\!\big(1,\,\text{KL}(\text{sg}\,q_\phi(z_t\mid h_t,x_t) \,\Vert\, p_\phi(z_t\mid h_t))\big),
$$

$$
\mathcal{L}_{\text{rep}}(\phi) = \max\!\big(1,\,\text{KL}(q_\phi(z_t\mid h_t,x_t) \,\Vert\, \text{sg}\,p_\phi(z_t\mid h_t))\big),
$$

with weights $\beta_{\text{dyn}}=0.5$, $\beta_{\text{rep}}=0.1$. The
asymmetry $\beta_{\text{dyn}} > \beta_{\text{rep}}$ replaces V2's
$\alpha=0.8$: the prior is pushed toward the posterior with five times the
weight that the posterior is regularised toward the prior. The
$\max(1, \cdot)$ "free bits" floor — borrowed from PlaNet/Dreamer V1 — sets
each KL to zero once it falls below 1 nat, preventing the regulariser from
collapsing already-informative latents. This is what allows the *same*
β values to work for low-bandwidth Atari frames and high-bandwidth Minecraft
voxels.

**Unimix categoricals.** Both encoder ($q$) and dynamics predictor ($p$)
are now $0.99\,\text{NN}(\cdot) + 0.01\,\mathcal{U}$. With probability mass
floor $0.01/K$ on every class, the log-probabilities $\ln q$ and $\ln p$
are bounded by $\ln(K/0.01) \approx 10.1$ nats. This eliminates the rare
KL spikes seen in V2 training where a posterior temporarily collapsed to a
single class. The actor distribution also uses this trick.

**Percentile return scaling and the entropy invariance argument.** Define
the dispersion estimator

$$
S \;\leftarrow\; \text{EMA}\big(\,\text{Per}(R^\lambda, 95) - \text{Per}(R^\lambda, 5)\,\big),
$$

and replace the actor objective with

$$
\mathcal{L}(\theta) \;=\; -\sum_t \mathbb{E}_{\pi_\theta, p_\phi}\!\Big[\frac{\text{sg}(R^\lambda_t)}{\max(1, S)}\Big]
\;-\;\eta\,H[\pi_\theta(a_t\mid s_t)],
\qquad \eta = 3\cdot 10^{-4}.
$$

The clamp $\max(1, S)$ is the crucial asymmetry: in dense-reward domains
$S \gg 1$ shrinks the policy-gradient term so the fixed entropy bonus is
not overwhelmed; in sparse-reward domains $S < 1$ leaves returns
*unscaled*, so the entropy bonus does not destroy the rare-reward signal by
amplifying noise. The 5th–95th percentile range (rather than std-dev) is
robust to the heavy-tailed return distributions that arise in stochastic
domains, where a few lucky episodes can dominate variance.

**Hyperparameters (Table B.1, W.1).** $\beta_{\text{pred}}=1$,
$\beta_{\text{dyn}}=0.5$, $\beta_{\text{rep}}=0.1$, free-bits = 1 nat,
discount $\gamma=0.997$, $\lambda=0.95$, imagination horizon $T=16$,
twohot buckets $K=255$ in $[-20,+20]$, entropy $\eta=3\cdot 10^{-4}$,
unimix $0.01$, batch shape $B=16, L=64$, replay subsequence sampling,
training ratio 1 (default). Five model sizes XS/S/M/L/XL with GRU units
$\{256, 512, 1024, 2048, 4096\}$ and CNN multipliers $\{24, 32, 48, 64,
96\}$. The L size (≈70M params) is used for headline benchmarks.

### Relevance to this project (interoceptive-pain RL)

- **Symlog as the right transform for pain-magnitude returns.** Pain
  signals can have very heterogeneous magnitudes across episodes (mild
  discomfort vs. acute pain spikes). Symlog applied to the reward and
  value heads — and to interoceptive observation channels themselves — is
  a clean alternative to ad-hoc reward clipping and is directly compatible
  with the project's existing scalar-reward pipeline.
- **Twohot value head for sparse-pain-reward credit assignment.** If the
  project's environments produce mostly-zero rewards with rare large pain
  spikes, the twohot symlog critic is the right baseline — its categorical
  loss tolerates bimodal targets that crash a regression critic.
- **Percentile scaling as a precision/uncertainty proxy.** The
  percentile-based dispersion $S$ is essentially an empirical estimate of
  return-variability *precision*. The same idea — modulate policy
  gradient by an EMA of return percentile range — is a candidate
  noise/precision injection point for interoceptive-pain RL: when
  homeostatic dispersion is high, scale returns down; when it is low,
  scale through.
- **Free bits + KL balancing protects an interoceptive prior.** The
  combined regulariser is the right default for any project world model
  with both high-bandwidth (e.g., grid-world pixels) and low-bandwidth
  (interoception, scalar pain) channels — without it the model tends to
  ignore the small-bandwidth modality.
- **Single-hyperparameter framework lowers the experimental cost.** The
  project frequently runs a fixed pipeline across many environment
  variants (5×5, 7×7, NoPred / PredInt3, etc.); a Dreamer-V3-style
  hyper-parameter-stable baseline reduces the per-variant tuning burden
  that has historically dominated the project's experiment time.
- **Scaling laws encourage large interoception world models.** The Fig. 6
  result that bigger Dreamer networks improve both final score and
  data-efficiency monotonically argues against the project's instinct to
  use small networks "to fit the small grid". A 70M-parameter Dreamer is
  cheap on a single GPU and may be the right starting point.

## <a id="paper-5b"></a>Paper 5b — DreamerV3 Nature (Hafner et al. 2025)

**Citation:** Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). *Mastering diverse control tasks through world models.* Nature, doi:10.1038/s41586-025-08744-2.

**PDF:** `docs/project/references/Dreamer/sources/Hafner et al. 2025 - Mastering diverse control tasks through world models.pdf` (19 pages, main; supplementary in same volume)

This paper is the peer-reviewed Nature publication of DreamerV3. The
algorithmic core is identical to the 2023 preprint (Paper 5a) — symlog
predictions, twohot critic, percentile return normalisation, KL balancing
with free bits, RSSM with categorical latents and straight-through
gradients. **The deep-dive below focuses on the deltas: refinements to the
loss function, the move from equally-spaced to exponentially-spaced critic
buckets, the addition of distributional return prediction with replay-value
loss, an extra benchmark domain (ProcGen), an updated hardware substrate
(A100 instead of V100), and a wider scaling experiment up to 400M
parameters.** Equations identical to 5a are *referenced, not re-derived*.

### Backbone

- **Problem (unchanged from 5a).** A single configuration of an RL
  algorithm that masters 150+ tasks across continuous/discrete actions,
  visual/proprioceptive inputs, dense/sparse rewards, 2D/3D worlds, and
  procedural generation. Headline stress test remains Minecraft Diamond
  from scratch.
- **Method — what changed vs. preprint (5a).**
  - **Loss weight $\beta_{\text{dyn}}$ increased from 0.5 → 1.0.** The
    Nature paper writes
    $\mathcal{L}(\phi) = \mathbb{E}_q\!\sum_t [\beta_{\text{pred}}\mathcal{L}_{\text{pred}} + \beta_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \beta_{\text{rep}}\mathcal{L}_{\text{rep}}]$
    with $\beta_{\text{pred}}=1$, $\beta_{\text{dyn}}=1$,
    $\beta_{\text{rep}}=0.1$. The asymmetry between the prior-update
    direction ($\beta_{\text{dyn}}$) and the posterior-regularisation
    direction ($\beta_{\text{rep}}$) is now 10×, vs. 5× in the preprint —
    pushing the prior toward the posterior even more aggressively.
  - **Distributional critic, not just twohot scalar.** The preprint's
    twohot critic was a *scalar value head* implemented via a categorical
    softmax over 255 fixed buckets; the Nature paper makes the critic
    explicitly a **distributional return predictor** trained by maximum
    likelihood:
    $\mathcal{L}_{\text{critic}}(\psi) = -\sum_t \ln p_\psi(R^\lambda_t \mid s_t)$
    with the distributional output read out as $v_\psi(s_t) =
    \mathbb{E}[p_\psi(\cdot\mid s_t)]$.
  - **Exponentially-spaced critic buckets.** The bucket centres are now
    $B = \text{symexp}([-20, \ldots, +20])$ — i.e. equally spaced *in
    symlog space* but exponentially spaced *in raw return space*. The
    preprint's wording "$B = [-20, \ldots, +20]$ equally spaced" is now
    explicitly placed inside the symexp wrapper.
  - **Replay-value loss.** A new on-policy regulariser is added — the
    critic loss is computed *both* on imagined rollouts (with weight
    $\beta_{\text{val}}=1$) and on replay-buffer trajectories (with weight
    $\beta_{\text{repval}}=0.3$). The replay loss uses the
    imagined-rollout $R^\lambda$ at the start state as on-policy value
    annotation, then computes $\lambda$-returns from the replayed rewards.
    This was not in the preprint and is a stabiliser for hard-prediction
    domains (e.g., DMLab, Minecraft).
  - **EMA decay for percentile scaling stated explicitly as 0.99.** The
    preprint described the EMA without specifying the decay; the Nature
    version writes
    $S \leftarrow \text{EMA}(\text{Per}(R^\lambda, 95) - \text{Per}(R^\lambda, 5),\,0.99)$.
  - **Symexp twohot framing unified.** Both reward predictor and critic
    use the same "symexp twohot" loss
    $\mathcal{L}(\theta) = -\text{twohot}(y)^\top \ln \text{softmax}(f(x;\theta))$
    (see Eq. block on p. 4). The preprint described twohot only in the
    critic context.
  - **Architecture and scaling.** Six model sizes 12M → 400M parameters
    (vs. 5×, 8M → 200M in the preprint). Hardware substrate is now A100
    (preprint used V100). Hyperparameters explicitly tuned to be domain-
    invariant on a held-out validation set per the supplementary
    information.
  - **New benchmark — ProcGen.** 16 procedurally generated games with
    visual distractors, 50M frame budget. Outperforms PPG and Rainbow
    with fixed hyperparameters. The preprint did not include ProcGen.
  - **Atari coverage broadened to 57 games** (preprint: 55). Atari 200M
    median improved from V2's 219% to >300% (preprint reported 302%) —
    Nature wording emphasises "outperforms MuZero with a fraction of the
    compute".
  - **DMLab 30 tasks** (preprint: 8 tasks). 100M-frame Dreamer matches
    1B-frame IMPALA / R2D2+ — a 1000% data-efficiency gain (preprint had
    quoted 13000% on the smaller 8-task subset).
  - **PPO baseline at fixed hyperparameters added** as a "general
    algorithm" comparator across all eight domains; previously implicit.
- **Key equations (only those that changed).**
  - World-model loss with new weights:
    $\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\!\sum_t \big[1\cdot \mathcal{L}_{\text{pred}} + 1\cdot \mathcal{L}_{\text{dyn}} + 0.1\cdot \mathcal{L}_{\text{rep}}\big]$.
  - Distributional critic ML loss (replaces preprint Eq. 10):
    $\mathcal{L}_{\text{critic}}(\psi) = -\sum_{t=1}^{T}\Big[\beta_{\text{val}}\ln p_\psi(R^\lambda_t\mid s_t) + \beta_{\text{repval}}\ln p_\psi(R^\lambda_t\mid s_t^{\text{replay}})\Big]$
    with $R^\lambda_t = r_t + \gamma c_t\big[(1-\lambda) v_\psi(s_{t+1}) + \lambda R^\lambda_{t+1}\big]$, $R^\lambda_T = v_\psi(s_T)$.
  - Symexp bucket grid:
    $B = \text{symexp}\big([-20, -20+\Delta, \ldots, +20]\big)$, $|B| = 255$.
  - Percentile EMA explicit:
    $S \leftarrow \text{EMA}\big(\text{Per}(R^\lambda,95) - \text{Per}(R^\lambda,5),\,0.99\big)$.
  - Other equations (symlog, twohot, KL with free bits, $\lambda$-return,
    actor loss with $-\eta H[\pi]$ entropy bonus, percentile clamp
    $\max(1,S)$) are **identical to the preprint** — see Paper 5a.
- **Results delta (Fig. 4 of Nature; ablations Fig. 6).**
  - **8 benchmark domains**, all under one configuration: BSuite, Atari
    100k, Atari 200M (57 games), ProcGen, DMLab (30 tasks), Proprio
    Control (20), Visual Control (20), Minecraft Diamond. Dreamer matches
    or exceeds tuned expert in *every* domain; outperforms PPO at fixed
    hyperparameters in *all* domains.
  - **Minecraft Diamond.** All seeds collect a diamond within 100M steps
    on a single A100 over ~9 days. Compared favourably to VPT (720 GPUs
    for 9 days, requires human play data) and Voyager (uses language-model
    + bot-scripting commands). Dreamer is the only method that uses neither
    human data nor handcrafted high-level actions.
  - **Ablations (Fig. 6).** All robustness techniques contribute. Most
    important: KL balancing + free bits (world model) and return
    normalisation + symexp twohot regression (behaviour). Stopping
    *task-specific* (reward + value) gradients into the encoder hurts
    less than stopping *task-agnostic* (reconstruction) gradients —
    confirming and tightening the V2 finding that the world model rests
    predominantly on unsupervised reconstruction.
  - **Scaling.** 6 model sizes 12M–400M parameters; replay ratios
    1, 2, 4, 8, 16, 32, 64. Both axes scale monotonically and predictably
    on Crafter and a DMLab task.
- **Limitations / new caveats.**
  - The peer-review cycle did not change the algorithm but did tighten
    several loss-weight values; the *exact* preprint hyperparameters do
    not reproduce the Nature-paper Atari 200M score — practitioners must
    use the Nature-supplementary table.
  - Distributional critic + replay-value loss adds memory cost
    (factor ≈1.3 in critic-update memory, per Supplementary Information).
  - ProcGen results require the 50M-frame easy-mode setup; the hard-mode
    benchmark is not reported.
  - Despite Nature framing as a Minecraft solution, the
    block-break-speedup is still applied (as in the preprint); pure-vanilla
    Minecraft is still open.

### Phase 1 — Foundational synthesis

The Nature paper publishes the algorithm that the 2023 preprint introduced,
and the conceptual story is unchanged: a single configuration of a
RSSM-based world-model agent that beats specialised baselines across
150+ tasks via three robustness mechanisms — symlog prediction, twohot
critic, and percentile return scaling — supported by KL balancing and
free bits. What the peer-review process produced is a set of refinements
that move the agent from "robust enough to publish" to "robust enough that
all 8 benchmarks share one config and one PPO baseline runs underneath".

The first refinement is in the **world-model loss weights**. The preprint
gave $\beta_{\text{dyn}}=0.5$ to the prior-update KL and
$\beta_{\text{rep}}=0.1$ to the posterior-regularisation KL — a 5×
asymmetry in favour of training the prior. Nature pushes the prior even
harder: $\beta_{\text{dyn}}=1.0$ vs. $\beta_{\text{rep}}=0.1$, a 10×
asymmetry. The intuition, sharpened in the supplementary, is that
under-trained priors are the bigger failure mode across diverse domains —
once the encoder has any reasonable representation, the prior must catch
up quickly to make imagined rollouts useful.

The second refinement is to the **critic head**. The preprint described
the critic as a softmax over 255 equally-spaced buckets in $[-20, +20]$
*symlog units*, with a twohot soft target. Nature reframes this as a
proper *distributional return predictor*, trained by **maximum-likelihood**
on the twohot encoding of the bootstrapped $\lambda$-return, with bucket
centres explicitly $\text{symexp}([-20, \ldots, +20])$ — equally spaced in
symlog space, exponentially spaced in raw return space. This "symexp
twohot loss" is unified across the reward predictor and the critic.
Operationally the change is small; conceptually it brings DreamerV3
closer to C51 / IQN / MuZero distributional critics.

The third addition is a **replay-value loss**. The critic is now trained
not only on imagined rollouts (weight 1) but also on replay-buffer
trajectories (weight 0.3), where the imagined-rollout $R^\lambda$ at the
start state seeds the on-policy value annotation. This stabilises learning
in domains where the imagined rollouts diverge from real returns — DMLab
spatial navigation and the long-horizon Minecraft chain — and does so
without changing the actor objective.

Two operational details are made explicit. The percentile EMA decay is now
written as 0.99. The categorical critic buckets are now exponentially
spaced (the preprint wording was ambiguous; many re-implementations
read it as linear).

Empirically the Nature version adds **ProcGen** as an eighth domain (the
preprint had seven), broadens **Atari** to all 57 games (preprint: 55),
broadens **DMLab** to 30 tasks (preprint: 8), and reports a wider
scaling sweep — six model sizes from 12M to 400M parameters and seven
replay ratios. A high-quality fixed-hyperparameter PPO is run as a
"general-algorithm baseline" across every domain, against which Dreamer
wins everywhere. The Minecraft Diamond result is robust enough across
seeds that the paper claims *all* trained agents collect a diamond within
100M steps on a single A100 over ~9 days — a sharper claim than the
preprint's "first-from-scratch".

### Phase 2 — Graduate-level deep dive

**Distributional critic with maximum-likelihood update.** The preprint
critic loss (Eq. 10 there) was the cross-entropy on the twohot soft label
$y_t = \text{twohot}(\text{symlog}(R^\lambda_t))$ — equivalent to a
maximum-likelihood update under a categorical likelihood, but framed as a
scalar value head. The Nature version makes the distributional reading
explicit and adds the replay-value loss term. With $p_\psi(\cdot\mid s_t)$
the critic's softmax distribution over $|B|=255$ exponentially-spaced
buckets,

$$
\mathcal{L}_{\text{critic}}(\psi) \;=\; -\sum_{t=1}^{T}\Big[\beta_{\text{val}}\,\ln p_\psi(R^\lambda_t\mid s_t)
\;+\; \beta_{\text{repval}}\,\ln p_\psi(R^{\lambda,\text{replay}}_t\mid s_t^{\text{replay}})\Big],
$$

with $\beta_{\text{val}}=1$, $\beta_{\text{repval}}=0.3$. The replay
$\lambda$-return $R^{\lambda,\text{replay}}$ is computed by recursively
unrolling on the replay reward sequence, with the bootstrap value at the
last replay-step replaced by the *imagined* $R^\lambda$ for that state —
this "splices" off-policy replay returns into on-policy imagined returns.
The point estimate read out from the distribution remains
$v_\psi(s) = \text{symexp}\big(p_\psi(\cdot\mid s)^\top B\big)$ with
$B = \text{symexp}([-20,\ldots,+20])$.

**Why exponential bucket spacing.** A symexp grid means that the bucket
centres in *raw return space* are $\{\pm(\exp k - 1) : k \in 0\ldots 20\}$
— densely packed near zero (where most rewards live) and exponentially
sparser at large magnitudes. In symlog space, however, the buckets are
linear, so the cross-entropy loss has uniform gradient across the support.
This decouples the *gradient magnitude* from the *raw return scale*
exactly the same way symlog regression does, but in a discrete-output
form: the loss has the same shape whether the agent is learning Atari
returns of order 100 or Minecraft returns of order 12. The preprint's
"equally spaced" wording referred to this exact construction; Nature
disambiguates it.

**KL balancing — the 10× asymmetry.** Nature writes (page 3)

$$
\mathcal{L}(\phi) \;=\; \mathbb{E}_{q_\phi}\!\sum_{t=1}^{T}\Big[1\cdot \mathcal{L}_{\text{pred}}
\;+\; 1\cdot \mathcal{L}_{\text{dyn}}
\;+\; 0.1\cdot \mathcal{L}_{\text{rep}}\Big],
$$

with the same $\mathcal{L}_{\text{dyn}}, \mathcal{L}_{\text{rep}}$
free-bits-clamped KLs as Paper 5a. Compared to the preprint
($\beta_{\text{dyn}}=0.5$), the prior-update direction now has *twice* the
loss weight while the posterior-regularisation direction is unchanged at
0.1 — the effective KL-balance ratio is $\beta_{\text{dyn}}/\beta_{\text{rep}}
= 10$ vs. 5 in the preprint. In V2 language this corresponds to
$\alpha = 10/(10+1) \approx 0.91$ — even more aggressive than V2's 0.8.
The implication is that across the broader 8-domain suite, the dominant
remaining failure mode was a lagging prior, not over-regularised
posteriors.

**Percentile EMA dispersion explicit.** The preprint formula was
$S = \text{Per}(R^\lambda, 95) - \text{Per}(R^\lambda, 5)$ with EMA
"smoothed". Nature writes

$$
S \;\leftarrow\; \text{EMA}\big(\text{Per}(R^\lambda, 95) - \text{Per}(R^\lambda, 5),\;0.99\big),
$$

i.e. an exponential moving average with decay $\rho = 0.99$ across batches.
The actor loss (Eq. of "Actor learning" on p. 4) is otherwise the
identical preprint Eq. 11, with $\eta = 3\cdot 10^{-4}$ and
$\max(1, S)$ clamp. The Nature paper explicitly compares against three
alternative scalings (advantage normalisation; std-deviation
normalisation; constrained-entropy optimisation) and shows that none can
hold a single hyperparameter set across all 8 domains.

**Encoder/decoder scaling.** Six model sizes M$_{12}$ to M$_{400}$. The
supplementary specifies CNN multipliers and GRU widths that scale roughly
$\sqrt{2}$× per step. The Crafter and DMLab scaling curves
(Nature Fig. 6c–d) show monotonic improvement in *both* final score and
data-efficiency — confirming the preprint result on a wider sweep and at
larger top-end (200M → 400M).

**Hyperparameters that did NOT change vs. preprint.** $\gamma=0.997$,
$\lambda=0.95$, imagination horizon $T=16$, twohot bucket count
$K=255$ in symlog range $[-20,+20]$, entropy $\eta=3\cdot 10^{-4}$, unimix
$0.01$, free-bits floor 1 nat, batch shape $B=16, L=64$, replay
subsequence sampling, RSSM categorical 32×32. The same actor gradient
estimator (Reinforce for discrete, reparameterised continuous) is used.

**Why the Nature paper still calls it "Dreamer", not "DreamerV3"
explicitly.** The text consistently writes "the third generation of the
Dreamer algorithm" rather than "DreamerV3" — the Nature framing positions
the algorithm as a single mature recipe rather than a versioned snapshot.
This is consistent with "Paper 6 — Dreamer 4" then breaking the
versioning explicitly to mark a paradigm shift in how the agent is
trained inside the world model.

### Relevance to this project (interoceptive-pain RL)

- **Use the Nature hyperparameters, not the preprint.** If the project
  builds a Dreamer-V3 baseline, take the Nature supplementary table as
  source of truth — $\beta_{\text{dyn}}=1$, distributional critic with
  exponentially-spaced symexp buckets, replay-value loss with weight 0.3,
  percentile EMA decay 0.99. The preprint values reproduce a slightly
  weaker agent.
- **Distributional critic for pain-RL.** Pain rewards are heavy-tailed
  and bimodal (long stretches of zero, occasional large negative spikes).
  The Nature distributional critic with symexp twohot loss is the natural
  baseline; it directly models the multi-modality the project's reward
  distribution exhibits.
- **Replay-value loss for sparse-pain credit assignment.** When pain
  events are sparse (e.g., predator-encounter episodes), imagined
  rollouts may rarely produce them, so the imagination-only critic
  diverges from real returns. Adding the Nature replay-value term
  (weight 0.3) is a low-risk stability fix the project should adopt.
- **PPO baseline at fixed hyperparameters.** The Nature paper's
  fixed-hyperparameter PPO comparator is exactly the apparatus the
  project already has (RPPO, PredInt). The "Dreamer beats PPO under one
  config" result motivates trying Dreamer-V3 as the project's
  flagship-baseline replacement.
- **400M-parameter scaling reach.** The project tends to train small
  models on small grids; the Fig. 6 monotonic scaling argues that even
  on a 5×5 grid, a larger Dreamer can sample-efficiency-dominate a
  small one — the GPU budget is not a reason to stay tiny.
- **ProcGen as a domain-randomisation analogue.** The Nature paper's
  ProcGen result (16 procedurally-generated games at fixed
  hyperparameters) is the closest domain to the project's
  cell-randomised grid environments and gives a credible expectation
  that DreamerV3 will transfer across grid layouts without retuning.

## <a id="paper-6"></a>Paper 6 — Dreamer 4 (Hafner, Yan & Lillicrap 2025)

**Citation:** Hafner, D.\*, Yan, W.\*, & Lillicrap, T. (2025). *Training Agents Inside of Scalable World Models.* arXiv:2509.24527. (\* equal contribution; Google DeepMind.)

**PDF:** `docs/project/references/Dreamer/sources/Hafner et al. 2025 - Training agents inside of scalable world models.pdf` (32 pages)

### Backbone

- **Problem.** DreamerV3 mastered 150+ tasks but its RSSM-based world model
  cannot fit complex real-video distributions and cannot accurately predict
  long-horizon object interactions in 3D worlds with hundreds of items.
  Controllable video models (Genie, Lucid, Oasis, MineWorld) handle the
  visual distribution but fail at precise game mechanics and require many
  GPUs for real-time inference, making them unusable for imagination-based
  RL. Dreamer 4 targets a single integrated agent that (a) learns a video
  world model accurate enough that a human can play in it, (b) runs at
  real time (>20 FPS) on a single GPU, (c) trains a controlled policy
  *purely offline* from a fixed video+action dataset, and (d) is the first
  to obtain Minecraft diamonds without any online environment interaction.
- **Method — three-phase training pipeline (Algorithm 1; Fig. 2).**
  - **Phase 1 — World-model pretraining.** A *causal tokenizer* and an
    *interactive dynamics transformer* are trained on 2.5K hours of VPT
    contractor video.
    - Tokenizer: block-causal transformer encoder/decoder with a
      tanh-projected continuous latent bottleneck; trained as a masked
      autoencoder with mean-squared-error + 0.2·LPIPS loss (Eq. 5),
      patch dropout $p \sim U(0, 0.9)$.
    - Dynamics: block-causal 2D (space×time) transformer that consumes
      the interleaved sequence of tokenizer latents, actions, and
      shortcut-noise control tokens. Trained with a **shortcut forcing**
      objective (Eq. 7) — a fusion of diffusion-forcing and shortcut
      models reformulated in *x-space* (clean-target prediction) so that
      long autoregressive rollouts do not accumulate error.
    - Loss-RMS normalisation across the multi-task losses keeps weights
      auto-tuned across modalities.
  - **Phase 2 — Agent finetuning.** Task tokens are *inserted into the
    same dynamics transformer* with one-way attention (agent tokens see
    everything; nothing sees agent tokens — prevents causal confusion).
    Multi-token-prediction (MTP) heads of length L=8 read out actions
    and rewards (Eq. 9). The reward head reuses DreamerV3's symexp
    twohot output. Behaviour-cloning loss runs on relevant sequences;
    dynamics loss continues on the uniform-sequence half of each batch
    to preserve world-model fidelity.
  - **Phase 3 — Imagination training.** Policy and value heads are
    finetuned by RL *inside the frozen world model*. Imagined rollouts
    seed from real-video contexts; the policy and value heads update
    while the transformer stays frozen. The critic uses the V3
    distributional symexp twohot return loss (Eq. 10). The actor is
    trained with **PMPO** (Eq. 11), a sign-of-advantage variant that
    *ignores advantage magnitude* and uses a behavioural-prior KL
    regulariser to keep the policy near demonstrated behaviour.
- **Key equations.**
  - Flow-matching base loss (Eq. 1):
    $\mathcal{L}(\theta) = \|f_\theta(x_\tau,\tau) - (x_1 - x_0)\|^2$
    with $x_\tau = (1-\tau)x_0 + \tau x_1$.
  - Shortcut forcing in x-space (Eq. 7):
    $\mathcal{L}(\theta) =
    \begin{cases}
    \|\hat z_1 - z_1\|_2^2 & d = d_{\min}\\
    (1-\tau)^2\,\big\|(\hat z_1 - \tilde z)/(1-\tau) - \text{sg}(b_1+b_2)/2\big\|_2^2 & d > d_{\min}
    \end{cases}$
    with bootstrap velocities computed in v-space then rescaled to x-space.
  - Ramp loss weight (Eq. 8): $w(\tau) = 0.9\,\tau + 0.1$ — concentrates
    capacity on high-signal $\tau$.
  - Behaviour-cloning + reward MTP (Eq. 9):
    $\mathcal{L}(\theta) = -\sum_{n=0}^L \ln p_\theta(a_{t+n}\mid h_t) - \sum_{n=0}^L \ln p_\theta(r_{t+n}\mid h_t)$
    with L=8.
  - Distributional value head (Eq. 10):
    $\mathcal{L}(\theta) = -\sum_t \ln p_\theta(R^\lambda_t\mid s_t)$,
    $R^\lambda_t = r_t + \gamma c_t[(1-\lambda)v_t + \lambda R^\lambda_{t+1}]$,
    $R^\lambda_T = v_T$, $\gamma = 0.997$.
  - PMPO actor (Eq. 11):
    $\mathcal{L}(\theta) = \frac{1-\alpha}{|\mathcal{D}_-|}\sum_{i\in \mathcal{D}_-}\ln\pi_\theta(a_i\mid s_i) - \frac{\alpha}{|\mathcal{D}_+|}\sum_{i\in \mathcal{D}_+}\ln\pi_\theta(a_i\mid s_i) + \frac{\beta}{N}\sum_i \text{KL}[\pi_\theta(a_i\mid s_i)\,\Vert\,\pi_{\text{prior}}]$
    with $\alpha=0.5$, $\beta=0.3$, $\mathcal{D}_+ = \{s_i: A_i \geq 0\}$,
    $\mathcal{D}_- = \{s_i: A_i < 0\}$, $A_t = R^\lambda_t - v_t$.
- **Architecture details.**
  - 2D block-causal transformer with **separate space-only and time-only
    attention layers** (3D attention is replaced by alternating 2D
    blocks); temporal attention applied only every 4 layers; GQA in
    dynamics; QK-Norm; SwiGLU; RoPE; pre-RMSNorm; attention-logit soft
    capping.
  - K=4 sampling steps per imagined frame ($d=1/4$); context perturbed
    to $\tau_{\text{ctx}}=0.1$ to make the model robust to its own
    imperfections.
  - Total: 400M parameter tokenizer + 1.6B parameter dynamics model =
    **2B parameters**. Trained on 256–1024 TPU-v5p with batch size 1
    per device + FSDP.
  - 256 spatial tokens × 192 frames context for Minecraft (9.6 s of
    video); 512 spatial × 96 frames for the robotics dataset.
- **Results.**
  - **Offline Diamond Challenge.** Dreamer 4 obtains Minecraft diamonds
    in 0.7% of 60-minute episodes purely from the 2.5K-hour VPT
    contractor dataset, *with no environment interaction*. VPT-finetuned
    (270K hours of synthetic-labelled YouTube + finetuning), BC, and
    VLA-Gemma-3 baselines all reach 0% diamonds. Dreamer 4 also reaches
    iron pickaxe at 29% (vs. VLA's 11%). Across ten milestone items
    (Fig. 3) Dreamer 4 dominates every offline baseline.
  - **World-model accuracy (Table 1, Fig. 5).** A human-in-the-loop
    test on 16 Minecraft tasks: Dreamer 4 succeeds on 14/16 (Oasis
    large 5/16, Lucid 0/16, MineWorld not real-time). Inference: 21 FPS
    on a single H100 with 9.6 s context — 6× longer context than prior
    Minecraft world models at higher framerate.
  - **Action-data efficiency (Fig. 7).** Action conditioning generalises
    from a small action-paired subset to the full 2.5K-hour video
    dataset; even leaving out a domain (Nether) for action labelling
    yields good action conditioning there via cross-domain transfer.
  - **Robotics transfer (Fig. 6).** Same world-model recipe, trained on
    a real-world robotics video dataset, supports counterfactual human
    interaction (pick-up, flip, press, throw) — overcoming the causal
    confusion of pure video models.
  - **Ablations (Fig. 4 + Section 4.4).** WM+BC (no imagination) lags
    full Dreamer 4; world-model representations outperform Gemma-3
    representations for BC, indicating video prediction implicitly
    learns control-relevant structure.
- **Limitations.**
  - 2B params + 1024 TPU pretraining is well outside single-GPU
    research budgets — though inference is single-GPU.
  - Diamond rate is still 0.7% — far below VPT-online (~20%); the gap
    is the cost of pure-offline training.
  - The behavioural-prior KL in PMPO depends on task-conditional
    behaviour cloning; it does not learn purely from reward.
  - Inventory items are sometimes incoherent over long rollouts; the
    9.6 s context still bounds long-horizon consistency.
  - The block-break-acceleration used in DreamerV3 Minecraft is *not*
    used in Dreamer 4 (the foveated mouse + 23-key binary action space
    is the raw VPT space) — but task prompting via 20 hand-listed
    Minecraft tasks is a softer form of curriculum.

### Phase 1 — Foundational synthesis

Dreamer 4 is the architectural break in the line. The first three
generations (PlaNet → DreamerV1 → DreamerV2 → DreamerV3) all use the
recurrent state-space model — a GRU latent paired with a categorical
stochastic state — and train an actor-critic *online* on imagined RSSM
rollouts. Dreamer 4 replaces the RSSM with a **2-billion-parameter
diffusion-forcing transformer** trained on raw video and trains the
agent **purely offline**, inside that frozen world model, on a fixed
2.5K-hour Minecraft dataset. The result is the first agent to obtain
Minecraft diamonds with no environment interaction.

The world model has two parts. A *causal tokenizer* compresses each
video frame into continuous latent tokens via a masked-autoencoder
objective; the latents pass through a tanh-projected low-dimensional
bottleneck so the dynamics transformer has a clean substrate to predict.
A separate *dynamics transformer* — a 2D block-causal transformer
operating on the interleaved sequence of latents, actions, and shortcut
noise/step tokens — denoises future latents via **shortcut forcing**, a
combination of diffusion forcing (per-timestep noise levels in a
sequence) and shortcut models (variable-step-size denoising). The key
choice is to predict *clean* latents (x-prediction) rather than
velocities (v-prediction): clean-target prediction does not produce
high-frequency outputs that would accumulate error over long
autoregressive rollouts. The model runs at K=4 sample steps per frame,
hitting 21 FPS on a single H100 — fast enough that a human can play
inside it interactively.

The agent is built by a *three-phase pipeline*. Phase 1 pretrains
tokenizer and dynamics on video. Phase 2 inserts task tokens into the
same dynamics transformer (with a one-way attention pattern that keeps
the world model unaffected by tasks) and adds behaviour-cloning + reward
heads that read out via multi-token prediction. Phase 3 trains the
policy and value heads by reinforcement learning *purely on imagined
rollouts* — the world model is frozen, but the policy explores in
imagination and is trained to maximise imagined return.

The actor objective is **PMPO** (Eq. 11). Where DreamerV3 uses
percentile-scaled return as a policy-gradient signal, Dreamer 4
discards advantage *magnitudes* entirely: every imagined state with
positive advantage is pushed up (maximum-likelihood on its own action),
every state with negative advantage is pushed down (minimum-likelihood
on its own action), and a KL-prior term keeps the policy close to its
behavioural-cloning initialisation. This sign-only update is more robust
when imagined returns can be unreliable far out from data, and it
removes the need for any return-normalisation hyperparameter.

The headline result is the first 0.7% diamond success on the offline
Minecraft challenge — beating the VPT offline finetune (0%) despite using
100× less data (2.5K vs. 270K hours), and beating a Gemma-3-initialised
VLA baseline that reaches only 11% iron pickaxe vs. Dreamer 4's 29%.
Even without the RL phase, the WM+BC baseline (just behaviour-cloning
inside the pretrained world-model representation) beats Gemma-3 BC,
showing that video-prediction representations outperform large
vision-language pretraining for control. A robotics dataset experiment
(pick-up, flip, throw counterfactuals) shows the same recipe transfers
to real-world video.

### Phase 2 — Graduate-level deep dive

**Shortcut forcing as the world-model objective.** Diffusion forcing
trains a sequence model with a per-timestep signal level $\tau_t$, so
each frame serves both as a denoising target and as conditioning
context. Shortcut models train a single network conditioned on both
$\tau$ and the requested step size $d$ so that inference can use a few
large steps without discretisation error. Dreamer 4 fuses both. Each
sequence is corrupted as $\tilde z_t = (1-\tau_t)z^0_t + \tau_t z^1_t$
with per-step $\tau_t$ and per-step $d_t$. The dynamics network outputs
a clean-latent prediction $\hat z_1 = f_\theta(\tilde z, \tau, d, a)$.
For the finest step size $d_{\min}$ the loss is squared error to the
clean latent (flow-matching); for larger step sizes the loss is the
"bootstrap" — the network output is compared to the average of two
half-step network outputs computed in v-space and rescaled into x-space:

$$
b_1 = (f_\theta(\tilde z,\tau,d/2,a) - \tilde z)/(1-\tau),\quad
z' = \tilde z + b_1\,d/2,
$$

$$
b_2 = (f_\theta(z',\tau+d/2,d/2,a) - z')/(1-(\tau+d/2)),
$$

$$
\mathcal{L}_{\text{boot}} = (1-\tau)^2 \big\|(\hat z_1 - \tilde z)/(1-\tau) - \text{sg}(b_1 + b_2)/2\big\|_2^2.
$$

The $(1-\tau)^2$ factor restores comparability with the x-space MSE so
the loss has uniform gradient magnitude across $\tau$. The ramp weight
$w(\tau) = 0.9\tau + 0.1$ (Eq. 8) further concentrates capacity on
high-signal $\tau$ where the bootstrap targets are the most informative.
At inference K=4 forward passes per frame are used; past frames are
re-corrupted to $\tau_{\text{ctx}}=0.1$ so the model is robust to its own
imperfections.

**Why x-prediction over v-prediction for video imagination.** With
v-prediction the network outputs a velocity $v = z^1 - z^0$ which is a
high-frequency signal (bounded magnitude, fast oscillation). When such a
network is iterated autoregressively for thousands of frames, small
high-frequency errors compound and the rollout drifts. x-prediction
outputs the clean latent itself — a low-frequency signal — and the v
needed for shortcut bootstrapping is computed analytically as
$\hat v_\tau = (\hat x_1 - x_\tau)/(1-\tau)$. The MSE relation
$\|\hat x_1 - x_1\|_2^2 = (1-\tau)^2 \|\hat v_\tau - v_\tau\|_2^2$ shows
the two formulations are equivalent at $\tau$ but the gradient
magnitudes differ — the x-prediction gradient does not blow up near
$\tau \to 0$ as v-prediction does.

**Agent finetuning by token insertion.** Phase 2 inserts a new "agent
token" modality into the dynamics transformer. The attention mask is
asymmetric:

- agent tokens attend to themselves and to every other modality
  (tokenizer latents, actions, shortcut noise);
- no other modality attends to the agent tokens.

This one-way visibility prevents *causal confusion*: the world-model's
prediction of next-frame latents must remain conditioned only on prior
frames and actions, not on the current task. Two MTP heads — policy and
reward — read out from the agent tokens via small MLPs with one output
per MTP horizon $n \in \{0, \ldots, L=8\}$, giving the loss

$$
\mathcal{L}(\theta) = -\sum_{n=0}^{L}\ln p_\theta(a_{t+n}\mid h_t) - \sum_{n=0}^{L}\ln p_\theta(r_{t+n}\mid h_t).
$$

The reward head reuses DreamerV3's symexp twohot output. The policy
head is categorical (mouse: 121-class foveated grid; keyboard: 23
binary distributions). To preserve world-model fidelity during this
phase, a 50/50 batch mix of *relevant* sequences (BC loss applied) and
*uniform* sequences (dynamics loss applied) is used.

**Imagination RL with frozen transformer.** Phase 3 unrolls the world
model from real-video contexts, with the dynamics transformer **frozen**
and only the policy and value heads training. One rollout per context is
generated (vs. multiple in DreamerV3) — to maximise data diversity at
fixed memory. The critic loss reuses DreamerV3 Nature Eq. 10:
distributional ML on $R^\lambda$ via symexp twohot. The actor loss is
PMPO:

$$
\mathcal{L}(\theta) = \frac{1-\alpha}{|\mathcal{D}_-|}\sum_{i \in \mathcal{D}_-}\ln \pi_\theta(a_i\mid s_i)
\;-\; \frac{\alpha}{|\mathcal{D}_+|}\sum_{i \in \mathcal{D}_+}\ln\pi_\theta(a_i\mid s_i)
\;+\; \frac{\beta}{N}\sum_i \text{KL}[\pi_\theta(\cdot\mid s_i)\,\Vert\,\pi_{\text{prior}}(\cdot\mid s_i)]
$$

with $\alpha=0.5$, $\beta=0.3$, $A_t = R^\lambda_t - v_t$,
$\mathcal{D}_\pm$ the positive/negative-advantage sets across the
(batch × time) imagined rollout. The first two terms balance positive
and negative advantage *equally* (regardless of dataset size or
magnitude); the third term is a reverse-KL toward the
behaviour-cloning prior $\pi_{\text{prior}} = \pi_{\theta^{\text{BC}}}$
that prevents the imagination-trained policy from drifting into
hallucinated regions where the world model is unreliable. Dreamer 4
notes the choice of *reverse* KL (vs. forward in original PMPO) better
constrains the policy to the support of the data.

**Why PMPO instead of percentile-scaled REINFORCE.** Sign-only
advantage updates ignore the magnitude of imagined return, which is
critical when the imagined return distribution can be miscalibrated by
the world model — DreamerV3's percentile scaling assumes the relative
ordering of returns is correct but their distribution is consistent
across episodes; PMPO assumes only that the *sign* is correct. This
relaxation is what allows reliable RL purely inside an imperfect 2B-param
video world model, where return *magnitudes* are unreliable but signs
correlate with the underlying reward labels.

**Efficient transformer architecture.** The 2B-param network is split
into a 400M tokenizer + 1.6B dynamics. The dynamics transformer is 2D
(space×time) with three speed tricks:

1. **Factored attention** — separate space-only and time-only attention
   layers replace full 3D attention.
2. **Sparse temporal attention** — temporal attention is only every 4th
   layer (the model is 4× more "spatial layers" than "temporal layers").
3. **GQA** — multiple query heads share the same key-value heads,
   shrinking the KV cache.

Combined with QK-Norm, attention-logit soft capping, RMSNorm pre-norm,
RoPE, and SwiGLU, this lets a 1.6B-param dynamics model run at 21 FPS
with 9.6 s context on a single H100.

**Loss-RMS normalisation.** Because Phase 1 trains many losses
(MSE+LPIPS for tokenizer, shortcut forcing for dynamics, action-MTP and
reward-MTP later), each is normalised by an EMA of its running RMS, so
no manual loss-weight tuning is needed when modalities are added.

### Relevance to this project (interoceptive-pain RL)

- **Offline imagination training is the right paradigm for pain.** The
  project's pain experiments are exactly the regime where partial-policy
  rollouts are unsafe / undesirable — letting an under-trained agent
  loose on a real environment is wasteful, and in physiological pain
  RL it is meaningless. Dreamer 4's recipe (pretrain world model on
  recorded data; train policy purely in imagination) is the
  paradigmatic offline-pain pipeline.
- **PMPO sign-of-advantage update for unreliable imagined pain.** When
  imagined trajectories under a learned interoceptive world model
  have miscalibrated *magnitudes* but reliable *signs*, PMPO is the
  natural objective. The project's existing reward-shaping work has
  produced reward distributions that are sign-stable but
  magnitude-unstable — exactly PMPO's regime.
- **Behavioural-prior KL as safety floor.** The PMPO behavioural-prior
  KL ($\beta = 0.3$) is a "stay near demonstrated behaviour"
  regulariser — directly applicable to a project that wants the agent
  to remain in physiologically-plausible behaviour space rather than
  exploit world-model hallucinations to "avoid pain" in an unreal way.
- **Token-insertion for interoceptive task conditioning.** Dreamer 4's
  one-way attention trick (task tokens see all, no modality sees task
  tokens) is the right architecture for inserting task-conditioning
  *or* interoceptive setpoints into a pretrained world model without
  contaminating the dynamics — directly transferable to the project's
  multi-task-pain experiments.
- **Scale is plausible.** A 2B-parameter world model is far beyond the
  project's current budget, but Dreamer 4 demonstrates its value:
  representations from a video world model outperform Gemma-3-scale
  vision-language pretraining for behaviour cloning. For the
  project, this argues that even modest-scale interoceptive video
  pretraining (e.g., on grid-world episodes) may produce control-useful
  representations beyond what supervised pretraining gives.
- **Dynamics-loss-on-uniform / BC-on-relevant 50/50 mix.** This is a
  generally useful trick for any project that wants to keep a world
  model "honest" while finetuning task heads — the project's
  interoceptive experiments often suffer from world-model drift when
  task-specific data dominates the batch.
- **Robotics generalisation evidence.** The Fig. 6 robotics result
  (counterfactual pick-up/flip/throw) suggests the recipe is
  modality-agnostic — a credible path for the project to scale beyond
  grid worlds when interoceptive data becomes available.

---

## <a id="cross-paper-synthesis"></a>Cross-paper synthesis

### Contribution map

| Paper | World-model representation | Transition dynamics | Behaviour-learning method | Key innovation | Headline benchmark |
|---|---|---|---|---|---|
| 1. World Models (Ha & Schmidhuber 2018) | VAE on raw pixels: $z_t \in \mathbb{R}^{32-64}$ + LSTM hidden $h_t$ | MDN-RNN: $K$-component Gaussian-mixture next-latent given $(z_t, a_t, h_t)$ | Tiny linear controller $a_t = W_c[z_t; h_t] + b_c$ optimised by CMA-ES | "Train inside the dream" — V/M/C decomposition; first agent fully trained inside its own learned model | CarRacing-v0 906±21; VizDoom Take-Cover 1092±556 (dream-trained) |
| 2. PlaNet (Hafner et al. 2019) | RSSM: deterministic GRU $h_t$ + Gaussian stochastic $z_t$, jointly latent | Learned latent transition $p(z_{t+1} \mid h_t, z_t, a_t)$, decoder reconstructs pixels | CEM planner over $H$-step latent rollouts (no learned policy) | First RSSM; deterministic+stochastic split; multi-step / overshoot KL | DM-Control from pixels — competitive with state-based SAC |
| 3. DreamerV1 (Hafner et al. 2020) | RSSM (Gaussian latents), reconstruction-driven | Same RSSM | Actor-critic on imagined latent rollouts, $\lambda$-returns, **dynamics backprop** through reparameterised Gaussian | Replace planning with learned policy + value; backprop value gradients through learned dynamics | DM-Control 20-task suite: 1184/1000 mean score |
| 4. DreamerV2 (Hafner et al. 2021) | RSSM with **categorical** latents (32×32 one-hot, ST gradients) | Same RSSM, categorical | Actor-critic, mixed Reinforce + dynamics-backprop, **KL balancing** ($\alpha=0.8$), entropy bonus | Discrete latents + KL balancing; first model-based agent to beat humans on Atari-200M | Atari-55 sticky-action: 2.15 gamer median, single GPU |
| 5a. DreamerV3 preprint (2023) | Same categorical RSSM, **unimix** + free bits + KL balancing | Same RSSM | Actor-critic + **symlog/twohot** value+reward, **percentile return scaling**, fixed entropy | One config across 150+ tasks; symlog stationary head; first scratch Minecraft Diamond | 7 benchmarks (Proprio/Visual Control, Atari 100k, Atari 200M, BSuite, Crafter, DMLab, Minecraft) |
| 5b. DreamerV3 Nature (2025) | Same categorical RSSM (no architectural change) | Same RSSM, $\beta_{\text{dyn}}$ raised to 1.0 | Distributional ML critic (symexp twohot, exponentially-spaced buckets) + **replay-value loss** ($\beta=0.3$); same percentile actor | Refined V3 hyperparameters; ProcGen added; PPO at fixed hyperparameters as universal baseline; $1000\times$ DMLab data-efficiency over IMPALA | 8 benchmarks, 150+ tasks, A100 single-GPU; all Minecraft seeds reach diamonds |
| 6. Dreamer 4 (2025) | **2B-param transformer**: causal MAE tokenizer + 2D block-causal dynamics; continuous tanh-projected latents | **Shortcut forcing** (diffusion-forcing × shortcut models) in x-space, K=4 sample steps/frame, ramp loss weight | **Three-phase pipeline**: world-model pretrain → BC+reward MTP finetune → **PMPO** imagination-RL on frozen transformer; behaviour-prior KL | Replace RSSM with diffusion-forcing transformer; first agent to obtain Minecraft diamonds **purely offline** | Offline Minecraft Diamond Challenge (no env interaction): 0.7% diamonds vs. 0% for VPT-finetune / BC / VLA |

### Narrative arc — V/M/C → RSSM → discrete latents → robust scaling → scaled-up world models

The line begins with **Ha & Schmidhuber's V/M/C decomposition**, which made
the conceptually-clean argument that the *world model* should carry the
parameter mass and the *controller* should be tiny — perception (V) and
prediction (M) are gradient-trainable, decision (C) only needs a few
hundred CMA-ES-tunable parameters. The big idea — *train the policy
entirely inside the world model's dream* — is established and demonstrated
on VizDoom Take-Cover.

**PlaNet** rebuilds V and M as a single end-to-end **Recurrent State-Space
Model**: a deterministic GRU pathway $h_t$ paired with a stochastic
Gaussian latent $z_t$, jointly trained to reconstruct pixels via a
multi-step ELBO with KL overshoot. This RSSM becomes the universal
backbone of every Dreamer paper through V3. PlaNet still uses a CEM
*planner* rather than a learned policy.

**DreamerV1** replaces CEM with an **actor-critic trained on imagined
RSSM rollouts** using $\lambda$-returns and dynamics-backprop value
gradients flowing through the reparameterised Gaussian latent. This is
the canonical Dreamer architecture: world model + actor + critic, all
gradient-trainable, all working inside imagination.

**DreamerV2** changes one thing about the RSSM — Gaussian → categorical
(32×32 one-hot with straight-through gradients) — and one thing about the
loss — KL balancing with $\alpha=0.8$ — and as a result becomes the first
model-based agent to clear human Atari-gamer median on a single GPU.

**DreamerV3** (preprint and Nature) is the engineering achievement of
the line. The world model, latent structure, and actor-critic are the
same V2 RSSM. What changes is the *prediction layer*: symlog-transformed
targets remove the need for reward clipping; a twohot/distributional
critic over symexp-spaced buckets removes the need for return
normalisation; percentile return scaling lets a single entropy
coefficient work across dense and sparse rewards; KL balancing combined
with free-bits removes the need for per-domain KL tuning. The result is
*one configuration* that masters 150+ tasks and collects Minecraft
diamonds from scratch.

**Dreamer 4** is the architectural break. The RSSM — which has been
stable since 2019 — is replaced with a **2-billion-parameter
diffusion-forcing transformer** trained offline on raw video. The agent
is built by token-insertion into the same transformer and trained
*entirely offline* via a sign-of-advantage objective (PMPO) plus a
behavioural-prior KL. The headline result — first diamond from
*pure offline learning* — is qualitatively different: it shows the line
can absorb the architectural advances of large video models (diffusion,
shortcut models, sparse-attention transformers) and still produce a
control-useful agent.

The conceptual constants across all seven papers: (i) the world model
carries the parameter mass; (ii) behaviour is learned by some form of
imagination / dream training; (iii) the actor-critic objective is built
around bootstrapped multi-step returns; (iv) the world model is trained
predominantly by reconstruction or video-prediction signals, not by
reward gradients into the encoder.

### Tying the line to interoceptive-pain RL

- **Discrete categorical latents (V2) + symlog/twohot reward heads (V3)
  are the right baseline for an interoceptive world model.** The 32×32
  categorical structure mirrors multi-channel categorical
  interoception (homeostatic regime classes); symlog reward heads
  handle the heavy-tailed, sign-stable, magnitude-unstable pain
  reward distribution that the project's environments produce. Both
  techniques are easy to drop into the project's existing pipeline.
- **Percentile return scaling (V3) and PMPO sign-of-advantage (Dreamer 4)
  are direct candidates for "precision/uncertainty-modulated"
  policy-gradient updates.** The percentile range $S$ of imagined
  returns is, formally, an EMA precision estimator on returns; PMPO's
  sign-only update implements a maximally-conservative version of the
  same idea. Either can be wired into the project's RPPO/PredInt code
  as a precision-on-pain modulator.
- **KL balancing + free bits (V2/V3) is the right regulariser for any
  project world model that has both high-bandwidth (pixels) and
  low-bandwidth (interoception, scalar pain) channels.** Without it,
  the small-bandwidth modality is collapsed by the KL term and the
  encoder ignores interoception. Adopt $\beta_{\text{dyn}}=1.0$ /
  $\beta_{\text{rep}}=0.1$ + 1-nat free-bits floor as the project
  default.
- **Image-reconstruction (V2 ablation) and unsupervised
  video-prediction (Dreamer 4 ablation) carry the representation
  weight, not reward gradients.** The project should keep its
  encoder driven by reconstruction / PredInt-style auxiliary losses
  and let pain enter only through reward and value heads — both V2
  and Dreamer 4 ablations show this protects representation
  generalisation.
- **Offline imagination training (Dreamer 4) is the long-term target
  paradigm for pain-RL.** Training partial policies on real
  physiological data is impractical; pretrain a world model on
  recorded data and train the policy inside it. The
  PMPO + behaviour-prior KL recipe is the right starting objective
  because it does not trust imagined-return *magnitudes* — only
  signs — which matches the project's reality of unreliable
  reward calibration.
- **Single-hyperparameter framework across project variants
  (V3 Nature).** The project runs many environment variants
  (5×5/7×7, NoPred/PredInt-N, predator-on/off). DreamerV3's
  single-config result argues for a Dreamer-V3 baseline as the
  *flagship* baseline that does not need per-variant tuning, freeing
  the project's tuning budget for the actual scientific contrast
  (precision/uncertainty heads, etc.).
- **Token-insertion task conditioning (Dreamer 4) is the cleanest way
  to expose interoceptive set-points to a pretrained dynamics
  model.** The asymmetric attention pattern — task tokens see all,
  no modality sees task tokens — gives the project a way to add
  homeostatic-setpoint conditioning to a pretrained world model
  without contaminating its dynamics, directly relevant to multi-task
  pain-RL experiments.
