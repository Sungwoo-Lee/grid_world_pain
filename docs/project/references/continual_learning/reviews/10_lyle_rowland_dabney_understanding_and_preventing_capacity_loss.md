> **Per-paper review — continual-learning corpus, paper 10 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§10); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 10. Lyle, Rowland & Dabney (2022) — Understanding and Preventing Capacity Loss in RL

**PDF:** `docs/project/references/continual_learning/sources/Lyle et al. 2022 - Understanding and Preventing Capacity Loss in RL.pdf`
**Venue:** ICLR 2022. **Authors:** Clare Lyle (Oxford), Mark Rowland & Will Dabney (DeepMind).

## Phase 1 — Foundational Overview

**The problem in one sentence.** As a value-based RL agent trains, it must keep re-fitting a *sequence* of prediction targets (its value estimates and policy keep changing). Lyle et al. show that training on this sequence of non-stationary targets erodes the network's ability to *quickly fit new targets* — they name this **capacity loss** — and in the extreme (**representation collapse**) the features shrink toward a low-dimensional or even zero subspace, at which point the agent *cannot make learning progress at all*. This is especially lethal in **sparse-reward** games like Montezuma's Revenge.

**Two ways they measure the degradation.**
1. **Target-fitting capacity** — take an agent's checkpoint, freeze it as a starting point, and measure how well it can fit a *fresh random target function* within a limited optimization budget. This *declines* over the course of training.
2. **Feature rank** — how many independent directions the penultimate-layer features span. In sparse-reward games this *collapses*; a high-enough feature rank turns out to be a *necessary* (though not sufficient) condition for learning progress.

**Key findings.**
- Capacity loss occurs on toy iterative-MNIST tasks (strongest for *smaller* networks — under-parameterization relative to task difficulty), and on Atari for DQN, QR-DQN, Rainbow.
- Feature rank correlates with performance on hard Atari games; representation collapse (rank→0) coincides with total failure to learn; an "unlucky" Pong seed only starts learning *after* it climbs out of representation collapse.
- **Sparse reward is the danger zone** — dense-reward and auxiliary-task signals keep feature rank up; sparse reward lets it collapse.
- **Fix — InFeR (Initial Feature Regularization).** Add a few auxiliary linear output heads and regularize them to keep matching their *values at initialization*. This preserves the network's early capacity, raises feature rank across training, and delivers large gains on sparse-reward Atari — most strikingly letting a *naively-exploring* DDQN agent make progress on Montezuma's Revenge *without any smart exploration algorithm*.

**Initial takeaway.** Part of what looks like an *exploration* failure in sparse-reward RL is actually a *representation-learning* failure — the agent degrades its own capacity to represent value functions. Preserving initial capacity is therefore as important as exploring well. InFeR is the primer's "regularize-toward-init" corrective (primer §4).

## Phase 2 — Graduate-Level Deep Dive

**Background.** Value-based RL with the Q-learning bootstrap target $\mathcal{T}Q(x,a) = \mathbb{E}[R(x_0,a_0) + \gamma\max_{a'}Q(x_1,a')\mid x_0=x, a_0=a]$. Deep RL minimizes, on sampled transition $\tau=(x_t,a_t,r_t,x_{t+1})$ with target params $\bar\theta$:

$$\ell_{TD}(Q_\theta, \tau) = \big(R_{t+1} + \gamma\max_{a'}Q_{\bar\theta}(X_{t+1},a') - Q_\theta(X_t,A_t)\big)^2 . \tag{2}$$

Features $\phi_\theta(x)$ = penultimate-layer outputs.

**Definition 1 (Target-fitting capacity).** For an input distribution $P_X$ and a distribution $P_F$ over target functions $f:\mathcal{X}\to\mathbb{R}$, with network-init pair $N=(g_\theta,\theta_0)$ and supervised optimizer $O$,

$$C(N,O,\mathcal{D}) = \mathbb{E}_{f\sim P_F}\Big[\mathbb{E}_{x\sim P_X}\big[(g_{\theta'}(x) - f(x))^2\big]\Big], \qquad \theta' = O(\theta_0, P_X, f). \tag{3}$$

It is the residual MSE after fitting a *new* target $f$ from the *current* parameters within a fixed budget — a direct operationalization of "can this network still learn something new fast?" Lower $C$ = more capacity. Target functions are chosen *independent of current parameters* (random-net outputs) to avoid the degenerate zero-function shortcut in sparse-reward settings.

**Two hypotheses.** *H1:* networks trained to iteratively fit dissimilar targets lose capacity to fit new ones. *H2:* the non-stationary prediction problems of deep RL also cause capacity loss.

*H1 test (iterative MNIST).* Generate target $f_{\theta}(x)$ from a randomly-initialized net; train the network to fit it for a fixed budget; reinitialize the target net; repeat 30 times, always warm-starting from the previous iteration. Result (Fig. 1): fitting error on later targets *increases*, worst for *smaller* MLPs. Over-parameterized nets ($\sim10^6$ params for $10^3$ points) show *positive* forward transfer; under-parameterized nets show monotonically rising error. This frames the central question — are deep RL benchmarks in the over- or under-parameterized regime?

*H2 test (Atari checkpoints).* Load agent checkpoints at time $t$, sample replay-buffer states, regress onto a fresh random-net target for 50k steps, measure MSE. DQN/QR-DQN/Rainbow checkpoints get *modestly worse* at fitting random targets as training progresses (Fig. 2).

**Definition 2 (Feature rank).** For feature map $\phi:\mathcal{X}\to\mathbb{R}^d$ and $n$ states $X_n$ sampled from $P$, with $\phi(X_n)\in\mathbb{R}^{n\times d}$ the feature matrix and $\mathrm{SVD}$ its multiset of singular values,

$$\rho(\phi,P,\epsilon) = \lim_{n\to\infty}\mathbb{E}_{X_n\sim P}\Big[\big|\{\sigma\in\mathrm{SVD}(\tfrac{1}{\sqrt n}\phi(X_n)) : \sigma>\epsilon\}\big|\Big], \tag{4}$$

with consistent finite-sample estimator

$$\hat\rho_n(\phi,X,\epsilon) = \big|\{\sigma\in\mathrm{SVD}(\tfrac{1}{\sqrt n}\phi(X)) : \sigma>\epsilon\}\big|. \tag{5}$$

At $\epsilon=0$ (finite $\mathcal{X}$) this equals the dimension of the feature-span subspace; $\epsilon>0$ discards small components. It measures how easily states can be *distinguished by updating only the final layer* — a cheap proxy for fast adaptivity.

**Contrast with Kumar et al.'s srank (important).** Two deliberate differences: (i) Lyle's estimator does *not* normalize by $\sigma_{\max}$ — this lets it capture **representation collapse** where *all* singular values (and the features themselves) go to zero, which a ratio-based srank would miss; (ii) Lyle studies the *unlimited-interaction* online regime rather than the data-limited regime. So feature rank and srank measure related-but-distinct pathologies (magnitude-collapse vs. relative-spectrum-collapse).

**Empirical rank↔performance link.** DDQN, QR-DQN, and RC-DQN (double DQN + auxiliary random-cumulant prediction) on Atari; $\hat\rho_n$ with $n=5000$. Denser signals (environment reward or auxiliary tasks) → higher feature rank (Fig. 3). In Montezuma's Revenge, the higher rank from RC-DQN's auxiliary task → higher performance; but that same auxiliary loss *hurts* dense-reward games (interference). Rank collapse is consistent only in *sparse-reward* games (QR-DQN most dramatic). Figure 4a: on hard Atari games, points cluster into low-rank / low-score vs. high-rank / higher-score — high feature rank is a *necessary but not sufficient* condition for progress (other factors: credit assignment, update-rule stability, optimizer, exploration).

**InFeR — the fix.** Add $k$ auxiliary linear heads $g_i$ on top of features $\phi_\theta$. Snapshot init params $\theta_0$; regress each head's *current* output toward its *initialization* output, scaled by $\beta$:

$$\mathcal{L}_{\mathrm{InFeR}}(\theta,\theta_0; B,\beta) = \mathbb{E}_{x\sim B}\Big[\sum_{i=1}^{k}\big(g_i(x;\theta) - \beta\, g_i(x;\theta_0)\big)^2\Big], \tag{6}$$

added to the TD loss with weight $\alpha$; $B$ = replay-buffer sampling. Interpretation: *amplify and preserve* subspaces of the features that were present at initialization (the $\beta>1$ scaling amplifies). Default: $k=10$, $\beta=100$, $\alpha=0.1$. It parallels function-space anti-forgetting regularizers (Benjamin et al.) but here the goal is *preserving plasticity/capacity*, not preserving old-task performance.

*Results.* On 57-game Atari, Rainbow$+$InFeR gives a net improvement, concentrated on hard sub-human games; it also reduces MNIST iterative-regression error (Fig. 5b). Striking case: DDQN (pure $\epsilon$-greedy, zero reward throughout without help) $+$InFeR makes progress on Montezuma's Revenge, exceeding Rainbow's noisy-net exploration in the last 40M frames. Trade-off: it slows a few dense games (Asteroids, Jamesbond).

**Which mechanism? (two hypotheses).** *H1 — random-subspace preservation:* InFeR just hands the final layer a preserved random feature subspace. *Tested* by concatenating a *frozen* random net's outputs to the learned features and training a linear head on top — this performs like vanilla Rainbow, *not* like InFeR (Fig. 6 left). So H1 is *rejected*: the effect on *earlier layers* is what matters. *H2 — whole-network dynamics:* InFeR slows the drift of features (at every layer) away from init in function space, preventing collapse/over-fitting. *Tested* by doubling penultimate-layer width (DoubleRainbow): the extra degrees of freedom reduce/eliminate/reverse the performance cost InFeR incurred on games where it hurt (Fig. 6 right). Conclusion: InFeR works by **regularizing the entire network's learning dynamics**, not by supplying a lucky random subspace.

**Project relevance.** "Capacity loss" is the primer's Lyle-2022 entry (§2 Phase 2), and the target-fitting-capacity operationalization is a clean way to *measure* the project's suspected plasticity deficit (fit a fresh random target from a checkpoint). Feature rank (Def. 2) is the diagnostic that Lyle's own later papers (2023/2024, other shards) decompose further. InFeR is the "regularize-toward-init" corrective in primer §4, and its finding that *sparse-reward failure is partly a representation problem* is directly relevant to the project's sparse survival-signal setting.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Deep RL is brittle in sparse-reward tasks (seed-to-seed variance) vs. robust supervised learning; blames non-stationary prediction targets. Thesis: agents lose capacity to quickly fit new targets; extreme = representation collapse = no progress. Proposes InFeR. Striking claim: Montezuma-type games improvable *without* smart exploration given the right representation objective ⇒ poor sparse-reward performance is partly a representation-learning, not exploration, failure.
- **§2 Background.** MDP, $Q^\pi$, Q-learning bootstrap target, deep-RL TD loss Eq. 2 (replay buffer, target net $\bar\theta$), features = penultimate layer $\phi_\theta$.
- **§3 Capacity Loss.** §3.1 Target-fitting capacity (Def. 1, Eq. 3); H1 (iterative MNIST, Fig. 1, worse for small nets, over- vs under-parameterized); H2 (Atari checkpoints fit random targets worse over time, Fig. 2). §3.2 Representation collapse & performance: Feature rank (Def. 2, Eqs. 4–5), contrast with Kumar srank (no $\sigma_{\max}$ normalization; online regime); denser signal → higher rank (Fig. 3); rank↔score clustering, necessary-not-sufficient (Fig. 4a); recovery-from-collapse Pong seed (Fig. 4b).
- **§4 InFeR.** §4.1 Method: $k$ auxiliary heads regressed to init outputs, loss Eq. 6, $\beta$ amplification; results on 57 Atari (Fig. 5), Montezuma DDQN/Rainbow, MNIST. §4.2 Mechanism: H1 (random-subspace) rejected via frozen-random-features control (Fig. 6 left); H2 (whole-network dynamics) supported via DoubleRainbow width-doubling (Fig. 6 right).
- **§5 Related Work.** Auxiliary tasks; value-function geometry/stability; implicit under-parameterization (Kumar 2021) and spectral normalization (Gogianu 2021); sub-task interference & catastrophic forgetting (EWC, GEM, distillation).

---
