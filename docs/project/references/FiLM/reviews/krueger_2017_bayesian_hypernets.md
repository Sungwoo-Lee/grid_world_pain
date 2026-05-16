---
title: "Bayesian Hypernetworks"
authors: ["David Krueger", "Chin-Wei Huang", "Riashat Islam", "Ryan Turner", "Alexandre Lacoste", "Aaron Courville"]
year: 2017
venue: "arXiv (later workshop / journal)"
slug: krueger_2017_bayesian_hypernets
source_pdf: "sources/Krueger et al. 2017 - Bayesian Hypernetworks.pdf"
topic: FiLM
---

# Krueger et al. 2017 — Bayesian Hypernetworks

## Plain-English entry point

A standard neural network has one fixed set of weights. A **Bayesian neural network** instead has a *distribution* over weights — we want to know not just "the best weights" but "the range of weights consistent with the data", so the model can express uncertainty when the data is ambiguous. The hard part is representing that distribution: most existing methods (mean-field variational inference, MC-dropout) assume each weight is independent of the others, which throws away rich correlation structure and leads to badly underestimated uncertainty.

This paper proposes the **Bayesian hypernetwork (BHN)**: a second neural network $h$ that takes Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ as input and outputs a sample of the primary network's weights, $\theta = h(\epsilon)$. The distribution over $\theta$ that you induce this way can be arbitrarily complex (multimodal, correlated across weights) — it's just whatever the hypernetwork pushes the noise into. To train this with variational inference you need to evaluate $\log q(\theta)$, which is hard for general transformations; the authors get around this by making $h$ an **invertible** transformation (a "normalizing flow", specifically RealNVP or Inverse Autoregressive Flow). Invertibility lets you compute $q(\theta)$ exactly via the change-of-variables formula.

A second engineering trick scales BHNs to large primary networks: instead of generating *all* weights with the hypernetwork, generate only the **weight-norm scaling factors** $g$ in a weight-normalization parametrization (so weights become $\theta_j = g \cdot v / \|v\|$ where $v$ is a deterministic ML estimate). This restricts the flexibility but cuts computation from quadratic to linear in primary-network width. Experiments on MNIST/CIFAR-10 classification, active learning, anomaly detection, and adversarial-example detection show BHNs match or beat Bayes-by-Backprop, MC-dropout, and deterministic baselines.

## Section-ordered backbone

**1. Introduction.** Calibrated parameter uncertainty matters for active learning, anomaly detection, adversarial robustness, and active human-in-the-loop systems. Most variational Bayesian DNNs (Bayes-by-Backprop, MC-dropout) use simple unimodal posteriors and underestimate uncertainty. The paper proposes Bayesian hypernets: an extremely flexible $q(\theta)$ parametrized by a neural network that transforms isotropic Gaussian noise into a posterior sample. The key engineering insight is to make this transformation invertible, enabling exact log-density evaluation needed for the ELBO.

**2. Related work.**
- **2.1 Bayesian DNNs**: MCMC (Welling-Teh SGLD, Neal HMC), variational inference (Bayes-by-Backprop with factorial Gaussians, MC-dropout, multiplicative normalizing flows by Louizos & Welling 2017). Bayes-by-Backprop is a degenerate BHN where $h$ is just element-wise scale-and-shift. Louizos & Welling proposed BHNs but rejected them due to scaling concerns, which this paper resolves.
- **2.2 Hypernetworks**: refers back to Ha 2016, De Brabandere 2016, Bertinetto 2016. Notes that **FiLM, CBN, CIN are special cases of hypernetworks** that output only per-channel $\gamma, \beta$ instead of full weights.
- **2.3 Invertible generative models**: differentiable directed generator networks (DDGNs) parametrize complex distributions by transforming noise. Normalizing flows (Dinh et al. RealNVP, Kingma et al. IAF) make the Jacobian determinant computationally cheap.

**3. Methods.**

**3.1 Variational inference.** Standard ELBO setup: maximize $\mathbb{E}_q[\log p(\mathcal{D}|\theta) + \log p(\theta) - \log q(\theta)]$ on minibatches.

**3.2 Bayesian hypernets.** Define $\theta = h(\epsilon)$ with $\epsilon \sim \mathcal{N}(0, I_D)$, $h: \mathbb{R}^D \to \mathbb{R}^D$. Sampling is trivial (forward pass). The entropy term $\mathbb{E}_q[-\log q(\theta)]$ requires evaluating $q(\theta)$, which generally requires integrating out $\epsilon$. Solution: make $h$ invertible and use the change-of-variables formula. The authors try both **RealNVP** (coupling-layer flows) and **Inverse Autoregressive Flow** (IAF with MADE) and find IAF performs better.

**3.3 Efficient parametrization.** Naive BHN where $h$ outputs all $\theta$ scales quadratically with primary-net width. Use **weight normalization** reparametrization $\theta_j = g \, v / \|v\|_2$ with $g \in \mathbb{R}$, and let only $g$ be drawn from $q$ while $v$ is learned by MLE. This is linear in width and still allows multi-modal, correlated $q(g)$. Also use weight norm inside the hypernet, small initialization, and clipping outputs into $(0.001, 0.999)$ for numerical stability.

**4. Experiments.**

**4.1 Qualitative.** Toy 1D regression (Blundell 2015 setup): uncertainty grows away from the data; the weight-norm-only BHN matches the full BHN. On the overparametrized identity $\hat y = a \cdot b \cdot x$ with the symmetric ground truth $a = b = \pm 1$, BHNs learn both modes — Hamiltonian Monte Carlo (NUTS) reference confirms this is the correct posterior.

**4.2 Classification.** On full MNIST/CIFAR-10, BHN performance scales monotonically with the number of coupling layers in the flow (more flow = more flexibility = better posterior). 8-layer BHN on MNIST: 98.63% vs MLE 98.73%, dropout 98.73%. On CIFAR-10 (small CNN): 8-coupling-layer BHN 74.90% vs MLE 72.75%, dropout 74.08%. On MNIST-5000 (small data), BHN clearly beats dropout. Across 10 trials, BHN-IAF wins consistently.

**4.3 Active learning.** Replicates Gal et al. 2017's setup on MNIST. With *warm-starting* (don't reinitialize after each acquisition), BHNs outperform MC-dropout for both random and BALD acquisition functions. Cold-starting helps other methods but hurts BHN.

**4.4 Anomaly detection.** Replicates Hendrycks & Gimpel 2016. BHN competitive with MC-dropout on Uniform, OmniGlot, CIFAR-bw, notMNIST out-of-distribution detection; MC-dropout slightly stronger overall on this suite, but BHN distinguishes itself in the next section.

**4.5 Adversarial examples.** BHN provides better defense against adversarial attacks than dropout — explicit support for the paper's safety motivation.

**5. Conclusion.** Bayesian hypernets combine the flexibility of normalizing flows with the meta-network structure of hypernetworks to give a tractable variational posterior over neural-network weights with multimodality and parameter correlations.

## Phase 1 — undergraduate-level synthesis

**Key idea.** Make a neural network's weights *random*. Use a *second* neural network $h$ (the hypernetwork) to turn random noise into a sample of those weights. The set of weights you get from many noise samples is the *posterior distribution* — but represented implicitly, by the hypernetwork itself.

**Why bother making weights random?** A standard ("MAP") neural network commits to one answer per weight. When data is sparse or ambiguous, this hides uncertainty. A Bayesian network keeps the spread, so:
- it knows when to be unsure (anomaly detection),
- it gives a built-in ensemble (better generalization),
- it resists adversarial inputs (an attacker can't fool one weight setting if there's a distribution).

**Setup.**
1. Sample $\epsilon$ from a standard Gaussian.
2. Push it through the hypernet $h$ to get weights $\theta = h(\epsilon)$.
3. Run the primary network with those weights.
4. To train, you need to know how likely each $\theta$ is under your distribution. Make $h$ *invertible* (using "normalizing flows" — clever neural-network designs whose Jacobian determinant is easy to compute). The change-of-variables formula then gives you $q(\theta) = q(\epsilon) / |\det(\partial h/\partial \epsilon)|$.

**Scaling trick.** Generating *all* primary-network weights from $h$ blows up with network size. Instead, use weight normalization: $\theta_j = g \cdot v / \|v\|$ and only sample the scalar $g$ from $h$; learn $v$ deterministically. This restricts your posterior shape but makes BHNs scale.

**Headline result.** On MNIST/CIFAR-10 classification with limited data, on active learning, anomaly detection, and adversarial-example detection, BHNs match or outperform MC-dropout and Bayes-by-Backprop. The clearest win is that *more flow layers = better posterior* — there's a tunable knob for how rich the posterior can be.

**Initial takeaway.** Bayesian hypernets are the meeting point of three ideas: (1) hypernetworks (Ha 2016) — one network produces another's weights; (2) variational inference — replace the intractable true posterior with a learned approximation; (3) normalizing flows (RealNVP, IAF) — build flexible-but-tractable density transformations. The product is the first practical way to get a multimodal, correlated weight posterior in a deep net.

## Phase 2 — graduate-level deep dive

### Variational inference for Bayesian DNNs (review)

For weights $\theta \in \mathbb{R}^D$, dataset $\mathcal{D}$, approximate posterior $q(\theta)$:

$$
\log p(\mathcal{D}) = \mathrm{KL}\big(q(\theta) \,\|\, p(\theta\mid\mathcal{D})\big) + \mathbb{E}_q\!\Big[ \log p(\mathcal{D}\mid\theta) + \log p(\theta) - \log q(\theta) \Big].
$$

Since $\mathrm{KL} \geq 0$,

$$
\log p(\mathcal{D}) \;\geq\; \mathcal{L}(q) := \mathbb{E}_q\!\Big[ \log p(\mathcal{D}\mid\theta) + \log p(\theta) - \log q(\theta) \Big].
$$

This is the **ELBO**. Maximizing $\mathcal{L}(q)$ over a parametric family is equivalent to minimizing $\mathrm{KL}(q \| p(\cdot\mid\mathcal{D}))$.

Three terms:
- **Reconstruction**: $\mathbb{E}_q[\log p(\mathcal{D}\mid\theta)]$, the data fit, estimable by MC.
- **Prior**: $\mathbb{E}_q[\log p(\theta)]$, the regularizer, cheap if $p$ is a tractable density (e.g., isotropic Gaussian).
- **Entropy**: $-\mathbb{E}_q[\log q(\theta)] = H[q]$. This is the term that needs invertibility.

### Bayesian hypernetwork formulation

Let $\epsilon \sim q_\epsilon(\epsilon) = \mathcal{N}(0, I_D)$ and let $h: \mathbb{R}^D \to \mathbb{R}^D$ be a parametric **invertible** neural network. Then $\theta = h(\epsilon)$ has density given by the change-of-variables formula:

$$
\boxed{\,q(\theta) = q_\epsilon(\epsilon) \,\Big|\det \frac{\partial h(\epsilon)}{\partial \epsilon}\Big|^{-1}, \qquad \theta = h(\epsilon).\,}
$$

Equivalently in log form:

$$
\log q(\theta) = \log q_\epsilon(\epsilon) - \log \Big|\det \frac{\partial h(\epsilon)}{\partial \epsilon}\Big|.
$$

For Gaussian $q_\epsilon$: $\log q_\epsilon(\epsilon) = -\tfrac{D}{2}\log(2\pi) - \tfrac{1}{2}\|\epsilon\|^2$.

The Bayesian-hypernet ELBO is:

$$
\mathcal{L}(\theta_h) = \mathbb{E}_{\epsilon \sim q_\epsilon}\!\Big[ \log p(\mathcal{D} \mid h(\epsilon)) + \log p(h(\epsilon)) - \log q_\epsilon(\epsilon) + \log\big|\det J_h(\epsilon)\big| \Big],
$$

where $\theta_h$ are the parameters of the hypernetwork and $J_h(\epsilon) = \partial h / \partial \epsilon$. This is differentiable in $\theta_h$ via reparametrization (Kingma & Welling 2013) — the noise $\epsilon$ has no learnable parameters, so $\nabla_{\theta_h} \mathcal{L}$ flows through $h$. Both $\log p(\mathcal{D}\mid\theta)$ (a standard NN forward pass with the sampled weights) and the Jacobian term are tractable when $h$ is a normalizing flow.

### Normalizing-flow choice

The authors instantiate $h$ as either:

**RealNVP coupling layers (Dinh et al. 2016).** Split $\epsilon = (\epsilon_a, \epsilon_b)$ and define one layer as

$$
\epsilon_a \mapsto \epsilon_a, \qquad \epsilon_b \mapsto \epsilon_b \odot \exp(s(\epsilon_a)) + t(\epsilon_a),
$$

with $s, t$ small MLPs (1-layer ReLU with 200 hidden units in this paper). The Jacobian is triangular, so $\log|\det J| = \sum_i s_i(\epsilon_a)$ — linear cost.

**Inverse Autoregressive Flow (IAF; Kingma et al. 2016).** With MADE-style autoregressive masks, each $h_i$ depends only on $\epsilon_{1:i-1}$:

$$
\theta_i = \mu_i(\epsilon_{1:i-1}) + \sigma_i(\epsilon_{1:i-1}) \cdot \epsilon_i,
$$

and $\log|\det J| = \sum_i \log \sigma_i$. IAF is fast for sampling (one parallel pass) but slow for density evaluation at arbitrary points — but in BHN training the density is only evaluated at the sampled $\theta = h(\epsilon)$, so IAF's asymmetry is not a problem. The authors find IAF works better than RealNVP in practice.

### Weight-norm scaling trick

Naively letting $h: \mathbb{R}^D \to \mathbb{R}^D$ output all $D$ weights of the primary net makes both $h$ and the Jacobian determinant $O(D^2)$ or worse. The trick is the **weight-normalization** parametrization (Salimans & Kingma 2016): for each unit $j$ with input weights $\theta_j$,

$$
\theta_j = g_j \, u_j, \qquad u_j := \frac{v_j}{\|v_j\|_2}, \qquad g_j \in \mathbb{R}.
$$

Now only the scalar $g_j$ per unit is treated as random. The hypernet $h: \mathbb{R}^N \to \mathbb{R}^N$ produces the vector $g = (g_1, \dots, g_N)$, where $N$ is the *number of units* (not the number of weights). The direction $v_j$ is learned by MLE as if it were a deterministic parameter. The induced posterior

$$
q(\theta) = q(g) \cdot \prod_j \delta\!\big(\theta_j - g_j v_j / \|v_j\|\big)
$$

is restricted (rank-1 per unit) but still admits multimodality and cross-unit correlation through $q(g)$. Cost scales linearly with $N$.

### Multimodality demonstration

The toy task $\hat y = a \cdot b \cdot x$ trained on $y = x + \epsilon$ has a symmetric posterior — both $a = b = 1$ and $a = b = -1$ fit the data equally well. Mean-field variational families collapse to one mode. BHN with a flow $h$ recovers both modes; HMC (NUTS) gives the same answer, confirming the BHN posterior is genuinely correct, not just flexible.

### Empirical anchor

| Task | Baseline | BHN best |
|---|---|---|
| MNIST-50k (acc) | 98.73% (dropout) | 98.63% (8 coupling) |
| CIFAR-10 (acc) | 74.08% (dropout) | 74.90% (8 coupling) |
| MNIST-5k 800u (acc) | 95.58% (dropout) | 96.16% (12 coupling) |
| Active learning | MC-dropout | warm-start BHN beats MC-dropout |
| Adversarial detection | dropout | BHN clearly better |

## Connections

- **`ha_2016_hypernetworks.md`** (this batch): direct architectural ancestor. Krueger 2017 lifts Ha's deterministic hypernetwork to a Bayesian one by making $h$ stochastic-and-invertible. The static-CNN hypernet in Ha 2016 corresponds to Krueger's primary network whose weights are sampled.
- **`galanti_wolf_2020_hypernet_modularity.md`** (this batch): the modularity argument carries over to BHNs — even when the posterior is over weights, the *primary* network is still small. Capacity goes into the flow $h$.
- **`shazeer_2017_sparse_moe.md`** (this batch): MoE shares the "many specialists, one router" theme but is *deterministic discrete*; BHN is *stochastic continuous*. Both make the effective model larger than the trained parameter count, but for different reasons.
- **`perez_2018_film.md`** (other batch — FiLM): the paper explicitly notes FiLM is a special case of a hypernetwork (Section 2.2). A "Bayesian FiLM" would be the natural specialization — only the per-channel $\gamma, \beta$ would be sampled from a flow. Useful for context-dependent uncertainty in the project's NMN-as-modulator direction.
- **Connection to project's uncertainty corpus** (e.g., Gawlikowski et al. 2023, Kendall & Gal 2017 in `references/uncertainty/`): BHN is one concrete instantiation of "epistemic uncertainty via parameter posterior". Where MC-dropout is cheap-but-restricted and ensembles are expensive-but-flexible, BHNs sit in between.
