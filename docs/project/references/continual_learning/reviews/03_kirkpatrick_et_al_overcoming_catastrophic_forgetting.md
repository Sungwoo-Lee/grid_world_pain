> **Per-paper review — continual-learning corpus, paper 3 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§3); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 3. Kirkpatrick et al. (2017) — Overcoming Catastrophic Forgetting (EWC)

**PDF:** `docs/project/references/continual_learning/sources/Kirkpatrick et al. 2017 - Overcoming Catastrophic Forgetting (EWC).pdf`
**Venue:** *PNAS* 114(13), 3521–3526 (DeepMind). **Type:** algorithm / empirical (supervised + RL).
**Primer link:** the canonical **regularization** family member in Phase 1 — the diagonal-Laplace / Fisher weight-anchoring method the whole family is named after.

## Phase 1: Foundational Overview

**Introduction (plain language).** Progressive nets (§2) avoid forgetting by never touching old weights and growing new ones — but that costs ever-more parameters. Elastic Weight Consolidation (EWC) keeps a *single, fixed-size* network and instead asks: which weights *mattered* for the old task? It then makes those specific weights *stiff* — reluctant to move — while leaving the rest free to learn the new task. The mental image is a spring ("elastic") anchoring each important weight to its old value, with a stiffness proportional to how important that weight was. Unimportant weights feel no spring and adapt freely. This is directly inspired by neuroscience: when a mouse learns a skill, certain dendritic spines enlarge and *persist* through later learning; erase them and the skill is lost. EWC is the artificial analogue of that *task-specific synaptic consolidation*.

**Key finding.** EWC lets one fixed-capacity network learn a long sequence of tasks with only modest error growth, where plain SGD forgets catastrophically and plain L2 regularization fails the opposite way (it protects *all* weights equally, so it can't learn the new task). Demonstrated on (a) **permuted MNIST** (each task = MNIST with a fixed random pixel permutation) with many sequential tasks, and (b) **ten sequential Atari 2600 games** with a DQN agent, where EWC agents learn to play multiple games while plain-SGD agents never exceed one.

**Initial takeaway.** EWC is the reference "regularization" method: cheap (linear in parameters and data), grounded in a Bayesian/Laplace argument, and biologically motivated. Its known weakness — which the paper is honest about — is that it *underestimates* parameter uncertainty (the diagonal-Fisher point estimate is over-confident), so it doesn't reach the score of ten independently trained networks.

## Phase 2: Graduate-Level Deep Dive

**The Bayesian setup.** Training is framed as finding the most probable parameters given data $D$. By Bayes' rule (in log form):

$$
\log p(\theta \mid D) = \log p(D \mid \theta) + \log p(\theta) - \log p(D). \tag{1}
$$

The log-likelihood $\log p(D\mid\theta)$ is just the negative loss, $-\mathcal{L}(\theta)$. Now split the data into two independent parts: $D_A$ (task A) and $D_B$ (task B). Because they are independent, $p(D\mid\theta) = p(D_A\mid\theta)\,p(D_B\mid\theta)$, and Bayes' rule can be re-arranged so the posterior over *everything* is expressed via the posterior after task A:

$$
\log p(\theta \mid D) = \log p(D_B \mid \theta) + \log p(\theta \mid D_A) - \log p(D_B). \tag{2}
$$

**Derivation of Eq. (2) (step by step).** Start from the full posterior and apply Bayes to the *joint* data, then factor the likelihood:

$$
\log p(\theta \mid D_A, D_B) = \log p(D_A, D_B \mid \theta) + \log p(\theta) - \log p(D_A, D_B).
$$

Using independence $p(D_A, D_B\mid\theta) = p(D_A\mid\theta)p(D_B\mid\theta)$ and $p(D_A,D_B)=p(D_A)p(D_B)$:

$$
= \log p(D_B\mid\theta) + \big[\log p(D_A\mid\theta) + \log p(\theta) - \log p(D_A)\big] - \log p(D_B).
$$

The bracketed term is exactly $\log p(\theta\mid D_A)$ (Bayes' rule applied to task A alone), giving Eq. (2). $\blacksquare$

The crucial reading of Eq. (2): the right-hand side depends on the new data *only* through $\log p(D_B\mid\theta)$; **everything the network needs to know about task A is compressed into the posterior $p(\theta\mid D_A)$.** So if we had that posterior, we could learn B while respecting A. The posterior is intractable, so EWC approximates it.

**The Laplace approximation.** Following MacKay (1992), approximate $p(\theta\mid D_A)$ as a Gaussian centered at the task-A solution $\theta_A^*$ with a **diagonal precision** given by the diagonal of the Fisher information matrix $F$. The Fisher has three properties that make it the right choice: (a) near a minimum it equals the second derivative of the loss (the Hessian), so it captures loss curvature = "how much does moving this weight hurt"; (b) it can be computed from first-order gradients alone (cheap even for huge models); (c) it is positive semi-definite (so the quadratic penalty is a valid bowl). Concretely the diagonal Fisher for parameter $i$ is

$$
F_i = \mathbb{E}_{x\sim D_A}\!\left[ \left(\frac{\partial \log p(x\mid\theta)}{\partial \theta_i}\right)^{\!2} \right]\Bigg|_{\theta = \theta_A^*}.
$$

**The EWC loss.** A Gaussian posterior with mean $\theta_A^*$ and diagonal precision $F$ contributes $-\log p(\theta\mid D_A) \approx \tfrac{1}{2}\sum_i F_i (\theta_i - \theta_{A,i}^*)^2 + \text{const}$. Substituting into Eq. (2) (and writing the task-B negative log-likelihood as its loss $\mathcal{L}_B$) gives the EWC objective:

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2}\, F_i\, (\theta_i - \theta_{A,i}^*)^2, \tag{3}
$$

where $\mathcal{L}_B(\theta)$ is the loss on task B alone, $\lambda$ sets how much the old task matters relative to the new one, and $i$ indexes parameters. **This is the "elastic" spring**: each weight $\theta_i$ is pulled toward its old value $\theta_{A,i}^*$ with stiffness $\lambda F_i$ — large for weights important to A (high Fisher), zero for irrelevant weights (Fisher ≈ 0). Contrast with plain L2, which is Eq. (3) with $F_i \equiv 1$ for all $i$ (uniform stiffness) — the paper's Fig. 2A shows this fails because it protects unimportant weights too, starving task B of capacity.

**Extending to $\geq 3$ tasks.** For a third task C, EWC anchors to *both* A and B. Because the sum of two quadratics is itself a quadratic, one can either keep two separate penalty terms or fold them into a single quadratic with an accumulated Fisher. (This "one online quadratic" choice is exactly where later "online EWC" variants differ; the paper notes both options.)

**Why over-parameterization makes this work.** EWC relies on the empirical fact (Nielsen 1989, Sussmann 1992) that many weight configurations yield the same performance. Over-parameterization makes it *likely* that a good task-B solution $\theta_B^*$ exists in the low-error neighbourhood of $\theta_A^*$ — so the constraint "stay near $\theta_A^*$ on the important axes" does not preclude solving B. This is the schematic of Fig. 1: gradient descent on B alone (blue) leaves A's basin; uniform constraint (green) can't reach B; EWC (red) threads into the *intersection* of A's and B's low-error regions by only constraining the A-important directions.

**Supervised results (permuted MNIST).** Plain SGD: task A collapses the instant training switches to B (Fig. 2A blue). L2: A protected but B never learned (green). EWC: both retained (red). EWC scales to many permutations with modest error growth where dropout-SGD does not (Fig. 2B). A representation-overlap probe (Fig. 2C) measures similarity between tasks' Fisher matrices by network depth: near-identical tasks share weights throughout; dissimilar tasks share less in early layers but still reuse output-side layers (input domain differs, label domain shared) — evidence EWC *shares* representation rather than partitioning the net per task.

**RL results (sequential Atari).** A DQN-based agent plays 10 games in randomized, interleaved segments. Extra machinery beyond vanilla DQN: (a) a **task-recognition module** — the task context is the latent of a Hidden Markov Model over observations, with the ability to *spawn new generative models* when recent data is poorly explained (a "Forget-Me-Not"–style, Bayesian non-parametric process); (b) per-task short-term replay buffers; (c) per-layer task-specific biases and multiplicative gains; (d) the EWC penalty, applied per game once it has seen ≥ 20M frames. Fisher recomputed at each task switch, scaled by $\lambda = 400$. Result (Fig. 3B): EWC agents learn multiple games (total human-normalized score rises), plain SGD stays below 1 (learns one game, forgets it when it returns). Providing the *true* task label instead of the inferred one gives only a modest gain — the HMM task inference is good enough.

**The honest limitation (Fig. 3C).** They test the Fisher's quality by perturbing a single-game (Breakout) agent's weights with covariance either uniform (black), inverse-Fisher $(F+\lambda I)^{-1}$ (blue, mimicking EWC's allowed moves), or uniform *within the Fisher's nullspace* (orange, directions EWC deems irrelevant). Inverse-Fisher perturbations hurt less than uniform — confirming the diagonal Fisher identifies important weights. But perturbing in the *nullspace* hurts *as much* as inverse-Fisher, which under the approximation should have *zero* effect. Conclusion: EWC is **over-confident that certain parameters are unimportant** — it under-estimates parameter uncertainty. This is the Laplace-point-estimate weakness; the paper suggests full Bayesian NNs (Blundell et al. 2015) as a remedy.

**The per-layer transformation and Fisher-overlap metric (appendix).** Each Atari layer applies task-specific gain $g^c_i$ and bias $b^c_i$: $y_i = \big(\sum_j W_{ij} x_j + b^c_i\big) g^c_i$. Fisher overlap between two tasks is the Fréchet distance between unit-trace-normalized Fishers $\hat{F}_1, \hat{F}_2$:

$$
d^2(\hat{F}_1, \hat{F}_2) = \tfrac{1}{2}\,\mathrm{tr}\!\left(\hat{F}_1 + \hat{F}_2 - 2(\hat{F}_1 \hat{F}_2)^{1/2}\right) = \tfrac{1}{2}\big\| \hat{F}_1^{1/2} - \hat{F}_2^{1/2} \big\|_F^2,
$$

bounded in $[0,1]$; overlap $= 1 - d^2$, where $0$ = disjoint weight sets and $1$ means $F_1 = \alpha F_2$.

*Relevance note.* EWC is the natural "regularization" comparator for any weight-anchoring idea in this project, and its Bayesian/Laplace framing ties directly into the project's Bayesian-brain and Bayesian-NN threads (the "each synapse stores a weight *and* its uncertainty" reading in the discussion). SI (§4) is its online sibling and shares the exact same penalty *form* — see the next entry.

## Appendix: Section-by-Section Backbone

- **Abstract.** Selectively slow learning on weights important for old tasks; scalable; demonstrated on permuted-MNIST classification and sequential Atari.
- **§1 Introduction.** Continual learning = learn consecutive tasks without forgetting. Interleaved multitask training avoids forgetting but is impractical (memory ∝ #tasks). Biological motivation: dendritic-spine enlargement persists across learning (Yang, Hayashi-Takagi, Cichon & Gan); erasing spines erases skill ⇒ task-specific synaptic consolidation.
- **§2 Elastic Weight Consolidation.** Quadratic-penalty spring intuition; stiffness varies per weight. Bayesian derivation Eqs. (1)–(2); all task-A info in $p(\theta\mid D_A)$; Laplace/Fisher-diagonal approximation (MacKay); Fisher's three properties; EWC loss Eq. (3); extension to 3+ tasks via summed quadratics.
- **§2.1 Supervised (permuted MNIST).** SGD forgets, L2 over-protects, EWC succeeds (Fig. 2A). Scales past dropout-SGD (Fig. 2B). Fisher-overlap by depth shows representation sharing (Fig. 2C).
- **§2.2 RL (Atari).** DQN + EWC on 10 interleaved games. Fixed capacity vs. progressive-net capacity growth. Task-recognition via HMM with spawnable generative models (FMN process). Per-task replay buffers; per-layer task-specific gains/biases. EWC after 20M frames, $\lambda=400$. EWC learns multiple games; SGD stays <1 (Fig. 3B). True-label control only modest gain. Fig. 3C perturbation test reveals under-estimated uncertainty (nullspace perturbations hurt).
- **§3 Discussion.** EWC grounded in Bayesian learning (prior = previous posterior). vs. French & Chater (slow), ELLA (matrix inversion). Linear-time cost via factorized-Gaussian + diagonal-Fisher point estimate; point estimate is the key weakness. Parallels to synaptic-uncertainty theories (Aitchison & Latham): synapse stores weight + variance + mean.
- **Appendix.** 4.1 MNIST settings (FC ReLU nets, dropout 0.2/0.5, early stopping). 4.2 Atari details (image preprocessing, 3 conv + FC, Double-Q, RMSProp, $\lambda=400$, EWC start 20M frames; HMM task model $p(c,t{+}1)=\sum_{c'}p(c',t)\Gamma(c,c')$ with switch rate $\alpha=1/t$; Dirichlet-multinomial generative models, spawn on hold-out selection). 4.3 Fisher overlap = $1-$ Fréchet distance between normalized Fishers.

---
