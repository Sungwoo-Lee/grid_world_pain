> **Per-paper review — continual-learning corpus, paper 8 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§8); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 8. Kumar et al. (2021) — Implicit Under-Parameterization Inhibits Data-Efficient Deep RL

**PDF:** `docs/project/references/continual_learning/sources/Kumar et al. 2021 - Implicit Under-Parameterization.pdf`
**Venue:** ICLR 2021. **Authors:** Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, Sergey Levine (UC Berkeley / Google Research / MILA).

## Phase 1 — Foundational Overview

**The problem in one sentence.** In value-based deep RL (Q-learning, actor-critic), the network is trained to regress onto *targets it generated itself* one step earlier — this is called **bootstrapping**. Kumar et al. show that repeating this self-regression, with gradient descent, gradually *collapses the rank* of the value network's internal features: a 512-dimensional feature layer ends up using only 20–100 truly independent directions. The network behaves as if it had far fewer parameters than it does — hence "implicit under-parameterization" — and performance drops with it.

**What "rank collapse" means intuitively.** The value network maps each state (or state-action pair) to a feature vector, then a final linear layer reads out the value. If the network maps *different* states to *nearly-parallel* feature vectors ("aliasing"), the final layer can no longer tell those states apart, so it cannot represent value functions that need to distinguish them. Effective rank counts how many genuinely-distinct feature directions survive; when it collapses, expressivity collapses.

**Key findings.**
- Rank collapse is demonstrated on Atari (DQN), continuous-control Gym (SAC), and a tabular-comparable gridworld (neural fitted-Q), in *both* offline RL (fixed dataset) and data-efficient online RL.
- **Lower rank ⇒ worse performance.** Across domains, the rank curve and the return curve fall together; when rank collapses, the network can no longer fit its own TD targets (TD error rises) nor the true optimal values $Q^*$.
- **More data reuse makes it worse.** Increasing the number of gradient updates per environment step ($n$) — the lever you pull for sample efficiency — accelerates rank collapse and degrades performance. Offline RL (infinite reuse of a fixed dataset) is the worst case.
- **Bootstrapping is the cause, isolated by controls.** Rank collapse persists even when you re-initialize the network from scratch each fitting iteration (rules out bad init / non-stationarity) and in pure policy-evaluation (rules out the max operator). It *disappears* when you regress to Monte-Carlo returns instead of bootstrapped targets. So it is the *self-referential target*, not the data or the control problem, that drives the collapse.
- **Fix (partial).** A singular-value penalty $\mathcal{L}_p(\Phi) = \sigma_{\max}^2(\Phi) - \sigma_{\min}^2(\Phi)$ that balances the feature spectrum mitigates collapse and improves DQN on 16/16 and CQL on 11/16 offline Atari games. It treats the symptom, not the root cause.

**Initial takeaway.** This is the first paper to tie a concrete, measurable representational pathology (feature-rank collapse) to the interaction of *bootstrapping* + *the implicit regularization of gradient descent*. It is the RL-specific sibling of the warm-start / capacity-loss story: the learner degrades its own substrate through self-referential training.

## Phase 2 — Graduate-Level Deep Dive

**Preliminaries.** MDP $(\mathcal{S},\mathcal{A},R,P,\gamma)$. $Q^\pi$ is the fixed point of the Bellman operator $\mathcal{T}^\pi Q(s,a) = R(s,a) + \gamma\,\mathbb{E}_{s'\sim P,\,a'\sim\pi}[Q(s',a')]$; $Q^*$ the fixed point of $\mathcal{T}Q(s,a) = R(s,a) + \gamma\,\mathbb{E}_{s'}[\max_{a'}Q(s',a')]$. Practical deep Q-learning minimizes the mean-squared TD error

$$\mathcal{L}(\theta) = \sum_{s,a}\big(R(s,a) + \gamma\,\bar{Q}_\theta(s',a') - Q_\theta(s,a)\big)^2 ,$$

where $\bar{Q}_\theta$ is a delayed target network. Write the penultimate-layer features as $\Phi \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|\times d}$ so $Q(s,a) = w^\top \Phi(s,a)$. The abstraction studied is **fitted Q-iteration (FQI)**: at fitting iteration $k$, form targets $y_k = R + \gamma P^\pi Q_{k-1}$ and take $T$ gradient steps to minimize $(Q_\theta - y_k)^2$.

**The measurement — effective rank.** For threshold $\delta$ (they use $0.01$),

$$\mathrm{srank}_\delta(\Phi) = \min\Big\{ k : \frac{\sum_{i=1}^{k}\sigma_i(\Phi)}{\sum_{i=1}^{d}\sigma_i(\Phi)} \ge 1-\delta \Big\},$$

with singular values $\sigma_1 \ge \dots \ge \sigma_d \ge 0$. It counts the number of leading singular directions that carry $(1-\delta)$ of the spectral mass — the number of "effective" independent feature components. High $\mathrm{srank}\approx d$ means states map to near-orthogonal features; low $\mathrm{srank}$ means aliasing onto a small subspace.

**Definition 1 (Implicit under-parameterization).** A reduction in $\mathrm{srank}_\delta(\Phi)$ that occurs implicitly as a by-product of learning the deep Q-network. Note: rank reduction *also* occurs in supervised learning where it is *beneficial* (a generalization-friendly implicit bias); the claim is that bootstrapping drives it *further*, into a *harmful* collapse.

### Theoretical analysis I — kernel-regression (NTK) view

Model each bootstrapping round as squared-TD regression with a universal-kernel regularizer (coefficient $c\ge0$) capturing the inductive bias of gradient descent under early stopping (following Mobahi et al.'s self-distillation analysis):

$$Q_{k+1} \leftarrow \arg\min_{Q\in\mathcal{Q}} \sum_{s_i,a_i\in\mathcal{D}} \big(Q(s_i,a_i) - y_k(s_i,a_i)\big)^2 \;+\; c\sum_{(s,a)}\sum_{(s',a')} u\big((s,a),(s',a')\big)\,Q(s,a)Q(s',a'). \tag{1}$$

The closed-form solution is $Q_{k+1}(s,a) = g_{(s,a)}^\top (cI+G)^{-1} y_k$, with Gram matrix $G$ of the induced positive-definite kernel and $g_{(s,a)}$ the corresponding row. Substituting the FQI target $y_k = R + \gamma P^\pi Q_{k-1}$ and defining $A = G(cI+G)^{-1}$ gives the recurrence (with $Q_0 = 0$):

$$Q_{k+1} = A\,[R + \gamma P^\pi Q_k] = A\sum_{i=1}^{k}\gamma^{k-i}(P^\pi A)^{k-i} R \;=:\; A\,M_k\,R. \tag{2}$$

Here $M_k$ linearly maps rewards to Q-values, so the *expressivity of $M_k$ bounds what value functions the learner can represent*.

**Theorem 4.1 (spectrum of $M_k$ sparsifies).** Let $S = \gamma P^\pi A$ be a normal matrix. Then there is a strictly increasing sequence of iterations $(k_l)_{l\ge1}$, $k_1=0$, such that for any two singular values $\sigma_i(S) < \sigma_j(S)$ and any $l' \ge l$,

$$\frac{\sigma_i(M_{k_{l'}})}{\sigma_j(M_{k_{l'}})} < \frac{\sigma_i(M_{k_l})}{\sigma_j(M_{k_l})} \le \frac{\sigma_i(S)}{\sigma_j(S)}.$$

Hence $\mathrm{srank}_\delta(M_{k_{l'}}) \le \mathrm{srank}_\delta(M_{k_l})$; if $S$ is PSD the decrease is monotone in *every* iteration.

*Proof sketch / intuition.* For a normal $S$, singular values equal $|\text{eigenvalues}|$, and $M_k = \sum_{i=1}^k \gamma^{k-i}(P^\pi A)^{k-i}R$ is a matrix polynomial in $S$. Raising $S$ to increasing powers along the sequence exponentially amplifies the *ratio* between any two distinct singular values (the larger one dominates), so the relative weight of the smaller singular directions shrinks toward zero. As the ratios $\sigma_i/\sigma_j$ collapse, the number of directions carrying $(1-\delta)$ of the mass — the effective rank — decreases. $\square$ The takeaway: bootstrapping's repeated composition drives a *generally decreasing* (not necessarily every-iteration) rank trend, unlike self-distillation which is monotone.

### Theoretical analysis II — deep-linear-network view (pinpoints *when*)

Represent $Q(s,a) = W_N W_\phi [s;a]$ with $N\ge3$ layers, $W_\phi = W_{N-1}\cdots W_1$ mapping input to penultimate features $\Phi$. Under a continuous-time gradient-flow model with a "balancedness" assumption on all but the last layer, the singular values of the feature matrix $W_\phi(k,t)$ (fitting iteration $k$, inner step $t$) evolve as

$$\dot\sigma_r(k,t) = -N\cdot\big(\sigma_r^2(k,t)\big)^{1-\frac{1}{N-1}} \cdot \Big\langle W_N(k,t)^\top \frac{d\mathcal{L}_{N,k+1}(W_{k,t})}{dW},\; u_r(k,t)\,v_r(k,t)^\top \Big\rangle, \tag{4}$$

with $u_r, v_r$ the left/right singular vectors. **Proposition 4.1** reads off Eq. (4): the multiplicative factor $\sigma_r^{2(1-1/(N-1))}$ means *larger singular values grow (or decay) exponentially faster than smaller ones*, so the gap between top and bottom singular values widens with $t$ — driving $\mathrm{srank}_\delta(W_\phi)$ down *within a single fitting iteration*. This is confirmed empirically (Seaquest: $\sigma_{\max}$ orders of magnitude above $\sigma_{100}$).

**Compounding across iterations.** The within-iteration rank drop is captured by an equivalent penalized objective

$$\min_{W_\phi, W_N\in\mathcal{M}} \|W_N W_\phi[s;a] - y_k(s,a)\|^2 + \lambda_k\,\mathrm{srank}_\delta(W_\phi), \tag{5}$$

i.e. the fitted solution trades a little TD error for a lower effective rank ($\lambda_k>0$). In the *self-regression* special case ($R=0$, $P^\pi=I$), "copy over" $W_\phi(k-1)$ is feasible with zero TD error and no rank change — but Eq. (5) prefers to *lower* $\mathrm{srank}$ at the cost of small TD error, so rank strictly drops each round and *compounds*.

**Proposition 4.2 (bound after $k$ rounds).** Assuming closure of the function class under the Bellman backup and that dynamics/reward transformations raise rank by at most $c_k$,

$$\mathrm{srank}_\delta(W_\phi(k)) \;\le\; \mathrm{srank}_\delta(W_\phi(0)) + \sum_{j=1}^{k} c_j - \sum_{j=1}^{k}\frac{\|Q_j - y_j\|}{\lambda_j}.$$

Rank *decreases* through the $\|Q_j - y_j\|/\lambda_j$ terms (gradient descent's low-rank preference) but can *increase* through the $c_j$ terms (reward $R$ and dynamics $P^\pi$ inject new structure).

**Theorem 4.2 (collapse near the fixed point).** When targets are close to the previous estimate, $y_k = Q_{k-1}+\varepsilon$ with $|\varepsilon|\ll|Q_{k-1}|$, there is a constant $\epsilon_0$ such that for $\|\varepsilon\| < \epsilon_0$, $c_k = 0$, and thus

$$\mathrm{srank}_\delta(W_\phi(k)) \le \mathrm{srank}_\delta(W_\phi(k-1)) - \|Q_k - y_k\|/\lambda_k .$$

*Interpretation.* As the value function approaches the Bellman fixed point ($y_k\approx Q_{k-1}$), bootstrapping degenerates into self-regression, the rank-increasing $c_k$ vanishes, and rank collapses monotonically — which paradoxically *increases the distance to the fixed point* (Fig. 5), potentially *preventing convergence to $Q^*$ from a good initialization*. This is the sharpest theoretical result: the pathology is worst exactly where you'd hope it would be benign.

### Mitigation

Since $\mathrm{srank}_\delta$ is non-differentiable, use the surrogate penalty

$$\mathcal{L}_p(\Phi) = \sigma_{\max}^2(\Phi) - \sigma_{\min}^2(\Phi), \tag{6}$$

added to the TD loss with weight $\alpha=0.001$, computed via SVD on a minibatch feature matrix. Minimizing the top singular value while lifting the bottom one balances the spectrum (effective rank is maximized when singular values are equal-magnitude). Result: on 5%-replay offline Atari, DQN$+\mathcal{L}_p$ improves 16/16 games (median $+74.5\%$), CQL$+\mathcal{L}_p$ improves 11/16 (median $+14.1\%$). Online (Rainbow): median $+20.6\%$; but DQN online got *worse* ($-11.5\%$) — the penalty is a symptom-level fix that "does not address the root cause."

**Distinction from prior rank work.** Yang et al. (2019) studied the rank of the $Q^*$-*matrix* ($|\mathcal{S}|\times|\mathcal{A}|$, upper-bounded by $\#$actions) and argued low rank is good. Kumar et al. study a *different object* — the learned *features* $\Phi$ — and show *feature*-rank collapse *hurts*. Network re-initialization (proposed by Igl et al./Fedus et al. for non-stationarity) does *not* prevent the collapse (Fig. 4c), further isolating bootstrapping as cause.

**Project relevance.** "Rank collapse" is one of the named mechanisms in the primer's Phase-3 decomposition (Lyle 2024 lists rank as one of several independent causes). Kumar is the origin of the feature-rank diagnostic that Lyle 2022 (§4 below) adapts (Lyle drops the max-singular-value normalization to also capture *representation collapse toward zero*). The $n$-updates-per-step finding directly underpins the project's replay-ratio study (`replay_ratio_speed_vs_performance`).

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Names "implicit under-parameterization": bootstrapping + GD makes an expressive value net behave under-parameterized via feature over-aliasing; aggravated by higher data re-use (offline RL = worst). Contributions: identify, demonstrate (Atari/Gym, offline+online), analyze (kernel + deep-linear), mitigate (SVD penalty).
- **§2 Preliminaries.** MDP, Bellman operators $\mathcal{T}^\pi/\mathcal{T}$, mean-squared TD error, target network, feature matrix $\Phi$ ($Q=w^\top\Phi$). Abstracts practical methods into generic FQI (Alg. 1): targets $y_k = R+\gamma P^\pi Q_{k-1}$, $T$ inner gradient steps.
- **§3 Implicit Under-Parameterization in Deep Q-Learning.** Defines $\mathrm{srank}_\delta(\Phi)$. Offline RL (Fig. 2): rank drops after initial learning, final rank tiny (20–100 of 512); worse with more gradient steps; persists with 4× data and under CQL (rules out coverage / distribution mismatch). Online RL (Fig. 3): higher $n$ (updates/env-step) ⇒ faster collapse + worse return. §3.1 Mechanism: as rank falls, TD error and $Q^*$-fitting error rise (Fig. 4a,b). Controls: rank collapse survives per-iteration re-initialization (4c) and FQE policy-evaluation; *vanishes* with Monte-Carlo targets (4d) ⇒ bootstrapping is the cause.
- **§4 Theoretical Analysis.** §4.1 Kernel regression: regularized objective Eq. 1, recurrence Eq. 2 ($Q_{k+1}=AM_kR$), Theorem 4.1 (singular-value ratios shrink ⇒ $\mathrm{srank}(M_k)$ decreases). §4.2 Deep linear nets: singular-value ODE Eq. 4, Prop. 4.1 (larger $\sigma$ evolve faster ⇒ within-iteration rank drop), abstract penalized objective Eq. 5, Prop. 4.2 (bound across iterations), Theorem 4.2 (near-fixed-point $\Rightarrow$ monotone collapse, increases distance to fixed point; Fig. 5).
- **§5 Mitigation.** Penalty $\mathcal{L}_p = \sigma_{\max}^2-\sigma_{\min}^2$ (Eq. 6), $\alpha=0.001$; prevents collapse (Fig. 6) and improves offline DQN 16/16, CQL 11/16 (Fig. 7); online results mixed; explicitly a symptom-level fix.
- **§6 Related Work.** Tabular/linear Q-learning error-propagation & divergence; convergence guarantees under restrictive assumptions contradicted near the optimum by Thm 4.2; distinction from Yang et al.'s $Q^*$-matrix rank; re-init (Igl/Fedus) doesn't help.
- **§7 Discussion.** Root-cause is GD's implicit regularization on bootstrapped objectives; future work: auxiliary losses that preserve rank passively.

---
