---
title: "On the Modularity of Hypernetworks"
authors: ["Tomer Galanti", "Lior Wolf"]
year: 2020
venue: "NeurIPS 2020"
slug: galanti_wolf_2020_hypernet_modularity
source_pdf: "sources/Galanti and Wolf 2020 - On the modularity of hypernetworks.pdf"
topic: FiLM
---

# Galanti & Wolf 2020 — On the Modularity of Hypernetworks

## Plain-English entry point

This is a *theory* paper that asks: **why are hypernetworks more parameter-efficient than embedding-based networks for the same conditional-modelling task?** Both architectures are trying to learn a function $y(x, I)$ where $x$ is a regular input and $I$ is a "task identifier" or conditioning signal. There are two natural ways to handle the conditioning:

- **Embedding method**: encode $I$ as a vector $e(I)$, concatenate it to $x$, and feed the result through a fixed primary network $q$. The primary network is the same for every $I$.
- **Hypernetwork method**: have a "hypernetwork" $f$ output the *weights* $\theta_I = f(I)$ of a smaller primary network $g$, and apply $g(x; \theta_I)$. The primary network is *different* for every $I$.

The authors define **modularity** as the property that the primary network $g$ can be as small as the minimal complexity needed to fit a single $y_I = y(\cdot, I)$. They prove that hypernetworks have this property (Theorem 4) but embedding methods do not (Theorems 2–3): in the embedding case, the primary network $q$ must be exponentially larger in the conditioning dimension to reach the same approximation error. They also extend a classical lower bound on neural-network approximation (Theorem 1) by removing the prior literature's "robust selection" assumption.

The headline: when you fix the approximation target $y \in W^{r,m}$ (an $r$-smooth function on $m = m_1 + m_2$ dimensions), a hypernetwork's primary $g$ needs $O(\epsilon^{-m_1/r})$ parameters, while the embedding method's primary $q$ needs $\Omega(\epsilon^{-(m_1+m_2)})$ parameters. The hypernetwork's advantage scales exponentially with the conditioning dimension $m_2$. Experiments on synthetic regression and self-supervised image-rotation prediction (MNIST, CIFAR-10) confirm the theory.

## Section-ordered backbone

**1. Introduction.** Conditioning shows up everywhere (WaveNets, StyleGAN, conditional generation). Two architectural choices: embedding-concatenation vs. hypernetwork. Hypernetworks hold state-of-the-art on continual learning, few-shot learning, NAS, hyperparameter prediction; but no theory explains *why*. The paper provides a complexity-theoretic justification. Four claimed contributions: (i) Thm. 1 lower-bounds NN approximation complexity without the robust-selection condition of Devore et al. (1989); (ii) Thms. 2–4 compare the primary-network complexities of $q$ and $g$; (iii) Thm. 5 bounds the *total* hypernetwork complexity under a structured-selector assumption; (iv) synthetic + real-world experiments.

**2. Problem setup.** Target $y: X \times I \to \mathbb{R}$ with $X = [-1,1]^{m_1}$, $I = [-1,1]^{m_2}$, $m = m_1 + m_2$. Embedding method: $h(x, I; \theta_e, \theta_q) = q(x, e(I; \theta_e); \theta_q)$. Hypernetwork: $h(x, I) = g(x; f(I; \theta_f))$. Target space: Sobolev space $W^{r,n}$ (functions with bounded derivatives up to order $r$). Defines "spectral complexity" $C(f) := L^{k-1} \prod_i \|W_i\|_1$ (Lipschitz-bound proxy). Two assumptions: unique best approximator (Assumption 1, validated empirically) and monotone improvement on adding a neuron (Assumption 2, proven for shallow networks under $L^2$).

**3. Degrees of approximation.** Classical results (Mhaskar 1996, etc.) give upper bound $O(\epsilon^{-n/r})$ for approximating $W^{r,n}$ with neural nets; matching lower bound $\tilde d_N(W^{r,m}) = \Omega(N^{-m/r})$ via the nonlinear $N$-width framework of DeVore et al. (1989), but only under a **robust selection** condition (the parameter-selection map $S: y \mapsto \theta$ must be continuous). The robust condition fails for typical NNs because identical functions can have different parameterizations.

**Theorem 1 (extension).** For piecewise-$C^1$ activations $\sigma$ with $\sigma' \in BV(\mathbb{R})$, the lower bound $N_f = \Omega(\epsilon^{-m/r})$ holds *without* requiring robust selection. Proof idea: construct a "wide" subclass $Y' \subset Y$ and a continuous selector restricted to it; then $d(f; Y) \geq d(f; Y') \geq \tfrac{1}{\alpha}\tilde d(f; Y') = \Omega(N^{-m/r})$.

**4. Expressivity of hypernetworks.**

**4.1 Comparing complexities of $q$ vs. $g$.**

**Theorem 2.** For embedding method $E_{e,q}$ with constant-output-dimension embedding $e$, target class $Y = W^{1,m}$: if $d(E_{e,q}, Y) \leq \epsilon$, then the *primary* $q$ has complexity $N_q = \Omega(\epsilon^{-(m_1+m_2)})$.

**Theorem 3.** Extension to variable embedding output dimension: $N_q = \Omega(\epsilon^{-\min(m, 2m_1)})$.

**Theorem 4 (Modularity).** For any $y \in W^{r,m}$ whose slices $y_I$ are not exact NNs, there exist classes $g$ (with $\sigma$ activations) and $f$ (with ReLU) such that $h(x, I) = g(x; f(I; \theta_f))$ achieves error $\leq \epsilon$ with $N_g = O(\epsilon^{-m_1/r})$. The primary $g$ only depends on $m_1$ (the "task-internal" dimension), not on $m_2$ (the conditioning dimension).

Proof of Thm. 4: (i) take $g$ of size $O(\epsilon^{-m_1/r})$ achieving $d(g; \{y_I\}_{I \in I}) \leq \epsilon$; (ii) prove a continuous selector $S: \{y_I\} \to \Theta_g$ exists; (iii) approximate $S$ uniformly by a sufficiently large ReLU network $f(I; \theta_f)$.

**4.2 Total parameter complexity (Thm. 5).** Under the practical assumption that the selector $S(I) = W \cdot h(I)$ factorizes as a linear projection of a low-rank feature map (class $\mathcal{P}^{k_1, k_2}_{r, w, c}$), the *total* hypernetwork parameter count $N_f$ reduces from $O(\epsilon^{-m/r})$ to $O(\epsilon^{-m_2/r} + \epsilon^{-m_1/r})$ — an additive decomposition over the two input dimensions.

**5. Experiments.**
- **Validating Assumption 1.** Two independent shallow ReLU MLPs trained to match a CNN target on MNIST / CIFAR-10 / $[-1,1]^{28\times28}$. Distance $\|f_1 - f_2\|$ converges below $\|f_i - y\|$, confirming uniqueness of best approximator.
- **Synthetic.** Target $y(x, I) = \langle x, h(I) \rangle$ with $h$ a 3-layer sigmoid MLP. Hypernetwork beats embedding method when $f$, $e$ have $\geq 3$ layers; hypernetwork performance improves with more layers in $f$, while embedding does not improve with more layers in $e$ or with larger embedding dimension. Matches Thms. 2–4.
- **Real-world.** Self-supervised image-rotation prediction on MNIST/CIFAR-10. Hypernetwork outperforms embedding by a wide margin and scales with depth; embedding plateaus.

**6. Conclusions.** Hypernetworks are modular: their primary network is as small as the per-task minimal complexity. Embedding methods are not. Theory says modularity is *achievable*, experiments suggest SGD typically *finds* modular solutions.

## Phase 1 — undergraduate-level synthesis

**Key idea.** When you have a task with an "input" $x$ and a "task identifier" $I$, two ways to combine them produce networks of very different sizes.

- *Embedding*: turn $I$ into a vector and feed it side-by-side with $x$ into one fixed network $q$.
- *Hypernetwork*: have one network $f$ *generate the weights* of a smaller network $g$, and run $g$ on $x$.

**Headline claim.** To reach the same accuracy, the *primary* network in the embedding case has to be much bigger — by a factor that grows exponentially with how many dimensions the task identifier $I$ has. The hypernetwork's primary $g$ can stay small (it only "knows" about $x$); all the task-routing work is offloaded to $f$.

**The proof in a sentence.** Pick a smooth target function space. There's a classical lower bound on how few weights a network needs to approximate any function in that space to error $\epsilon$. The authors show: (a) for the embedding method, that bound applies to the *primary* network alone, scaled by the full input dimension $m_1 + m_2$ — see Theorems 2, 3; (b) for the hypernetwork, the bound applies to the primary only scaled by $m_1$ alone — see Theorem 4. Conditioning is "absorbed" by the hypernetwork.

**Empirical validation.** On a synthetic regression task and on self-supervised image rotation prediction (MNIST, CIFAR-10), increasing the depth of $f$ in the hypernetwork keeps improving accuracy; increasing the depth of $e$ in the embedding method plateaus quickly. Same architecture footprint, different scaling behaviour.

**Initial takeaway.** The "modularity" theorem is a *capacity-allocation* argument: the right place to spend parameters in a conditional network is on the weight-generator $f$, not on a giant shared primary $q$. This justifies the entire hypernetwork family (FiLM, MoE routing, conditional batch norm) as not just empirically good but provably more efficient than concatenation-style conditioning under mild assumptions.

## Phase 2 — graduate-level deep dive

### Problem setup

For a target function $y: X \times I \to \mathbb{R}$ with $X = [-1,1]^{m_1}$, $I = [-1,1]^{m_2}$, define:

**Neural embedding class.** Given two function families $\mathcal{q} = \{q(x, z; \theta_q) : \theta_q \in \Theta_q\}$ and $\mathcal{e} = \{e(I; \theta_e) : \theta_e \in \Theta_e\}$,

$$
\mathcal{E}_{e,q} := \{ q(x, e(I; \theta_e); \theta_q) \,|\, \theta_q \in \Theta_q,\, \theta_e \in \Theta_e \}.
$$

**Hypernetwork class.** Given families $\mathcal{f} = \{f(I; \theta_f) \in \Theta_g\}$ and $\mathcal{g} = \{g(x; \theta_g) : \theta_g \in \Theta_g\}$,

$$
\mathcal{H}_{f,g} := \{ g(x; f(I; \theta_f)) \,|\, \theta_f \in \Theta_f \}.
$$

**Approximation error.** For a target class $Y$ and candidate class $\mathcal{P}$,

$$
d(\mathcal{P}; Y) := \sup_{y \in Y} \inf_{p \in \mathcal{P}} \|y - p\|_\infty.
$$

**Target space.** Sobolev ball $W^{r,n} := \{h \in C^r([-1,1]^n) : \|h\|_\infty + \sum_{1 \leq |k|_1 \leq r} \|D^k h\|_\infty \leq 1\}$.

### Theorem 1 (lower bound without robust selection)

Let $\sigma$ be a piecewise-$C^1(\mathbb{R})$ activation with $\sigma' \in BV(\mathbb{R})$ (bounded variation). Let $\mathcal{f}$ be a class of neural networks with $\sigma$ activations. Let $Y = W^{r,m}$. Assume every non-constant $y \in Y$ is not exactly in $\mathcal{f}$. Then

$$
d(\mathcal{f}; Y) \leq \epsilon \quad \Longrightarrow \quad N_f = \Omega\!\big(\epsilon^{-m/r}\big).
$$

**Proof idea.** The classical result of DeVore et al. (1989) gives the nonlinear $N$-width

$$
\tilde d_N(Y) := \inf_{\mathcal{f}: N_f = N} \inf_{S: Y \to \mathbb{R}^N \text{ cont.}} \sup_{y \in Y} \|f(\cdot; S(y)) - y\|_\infty = \Omega(N^{-m/r}),
$$

but only over *robust* selectors $S$ (continuous). Galanti & Wolf carve out a wide subclass $Y' \subset Y$ such that (a) $\tilde d_N(Y') = \Omega(N^{-m/r})$ and (b) there exists a continuous selector $S: Y' \to \Theta_f$ satisfying $\|f(\cdot; S(y)) - y\|_\infty \leq \alpha \cdot \inf_\theta \|f(\cdot; \theta) - y\|_\infty$ for some $\alpha > 0$. Then

$$
d(\mathcal{f}; Y) \geq d(\mathcal{f}; Y') \geq \tfrac{1}{\alpha} \tilde d(\mathcal{f}; Y') = \Omega(N_f^{-m/r}).
$$

### Theorem 2 (embedding lower bound, fixed embedding dim)

Let $\sigma$ be universal, piecewise-$C^1$ with $\sigma' \in BV$ and $\sigma(0) = 0$. Let $\mathcal{e}$ be a class of $C^1$ NNs with zero biases, output dim $k = O(1)$ and spectral complexity $C(e) \leq \ell_1$. Let $\mathcal{q}$ have $\sigma$ activations and $C(q) \leq \ell_2$. Let $Y = W^{1,m}$. Assume non-constant $y \in Y$ is not exactly representable as a $\sigma$-NN. If $d(\mathcal{E}_{e,q}; Y) \leq \epsilon$,

$$
\boxed{\,N_q = \Omega\!\big(\epsilon^{-(m_1 + m_2)}\big).\,}
$$

### Theorem 3 (embedding lower bound, variable embedding dim)

Same setting as Thm. 2 but $k$ may depend on $\epsilon$, with the first layer of $q$ bounded $\|W_1\|_1 \leq c$:

$$
\boxed{\,N_q = \Omega\!\big(\epsilon^{-\min(m,\, 2m_1)}\big).\,}
$$

### Theorem 4 (Modularity of Hypernetworks) — the main result

Let $\sigma$ be as in Thm. 2. Let $y \in W^{r,m}$ such that every slice $y_I$ is not exactly a $\sigma$-NN. Then there exists a class $\mathcal{g}$ of $\sigma$-NNs and a ReLU network $f(I; \theta_f)$ such that

$$
h(x, I) = g(x; f(I; \theta_f))
$$

approximates $y$ with $\|h - y\|_\infty \leq \epsilon$ and

$$
\boxed{\,N_g = O\!\big(\epsilon^{-m_1 / r}\big).\,}
$$

**Crucially**, $N_g$ depends only on $m_1$ (the $x$-dimension), not on $m_2$.

**Proof sketch (three-step).**

1. **Universal approximation of the slice family.** Since $\sigma$ is universal and $\{y_I\}_{I \in I}$ is a family of $W^{r,m_1}$ functions, choose $\mathcal{g}$ of size $O(\epsilon^{-m_1/r})$ so that

   $$
   d(\mathcal{g}; \{y_I\}) := \sup_I \inf_{\theta_g} \|g(\cdot; \theta_g) - y_I\|_\infty \leq \epsilon.
   $$

2. **Continuous selector existence.** The authors construct $S: \{y_I\} \to \Theta_g$ (continuous, near-optimal) satisfying $\sup_I \|g(\cdot; S(y_I)) - y_I\|_\infty \leq 2 d(\mathcal{g}; \{y_I\}) \leq 2\epsilon$. Composing with the continuous map $I \mapsto y_I$ gives $\hat S(I) := S(y_I)$, also continuous.

3. **Replace selector by a hypernetwork.** Since $g$ is uniformly continuous in both $x$ and $\theta_g$, ensuring $\inf_{\theta_f} \|f(\cdot; \theta_f) - \hat S(\cdot)\|_\infty$ is small suffices to make $\sup_I \|g(\cdot; f(I; \theta_f)) - g(\cdot; \hat S(I))\|_\infty \leq \epsilon$. By the ReLU universal-approximation result, a large enough ReLU $f$ achieves this. Triangle inequality:

   $$
   \sup_I \|g(\cdot; f(I; \theta_f)) - y_I\|_\infty \leq \sup_I \|g(\cdot; f) - g(\cdot; \hat S)\|_\infty + \sup_I \|g(\cdot; \hat S(I)) - y_I\|_\infty \leq 3\epsilon. \quad \blacksquare
   $$

### Theorem 5 (total parameter complexity)

Under the *structured selector* assumption — there exists a continuous selector $S \in \mathcal{P}^{m_2, N_g}_{r, w, c}$, i.e., $S(I) = W \cdot \tilde S(I)$ with $\tilde S: I \to \mathbb{R}^w$ each-coordinate $r$-smooth and $W$ a bounded-norm $N_g \times w$ projection — the *total* hypernetwork complexity is

$$
\boxed{\,N_f = O\!\Big( w^{1 + m_2/r} \cdot \epsilon^{-m_2/r} + w \cdot N_g \Big) = O\!\big( \epsilon^{-m_2/r} + \epsilon^{-m_1/r} \big).\,}
$$

Compare against Thm. 1's lower bound $\Omega(\epsilon^{-(m_1+m_2)/r})$ for monolithic NNs and the implicit $\Omega(\epsilon^{-(m_1+m_2)/r})$ for embedding methods. The hypernetwork's total parameter count decomposes *additively* over the two dimensions $m_1, m_2$, where the embedding/monolithic network decomposes *multiplicatively*. This is the precise sense in which hypernetworks are exponentially more parameter-efficient.

### Why the asymmetry?

The authors note that *the embedding method is a special case of a hypernetwork*: if $f(I)$ outputs only the first-layer biases of $g$, you recover concatenation. Hypernetworks strictly contain embedding methods. The reverse is not true: embedding cannot emulate a general hypernetwork because $q$'s architecture is fixed to take the concatenation $[x, e(I)]$ as input.

## Connections

- **`ha_2016_hypernetworks.md`** (this batch): provides the architectural setup whose generalization properties Galanti & Wolf prove. Ha 2016 is the constructive paper; this is the complexity-theoretic justification.
- **`krueger_2017_bayesian_hypernets.md`** (this batch): orthogonal extension (Bayesian uncertainty over $\theta_f$) but the modularity argument carries through — the hypernet still concentrates capacity in $f$.
- **`shazeer_2017_sparse_moe.md`** (this batch): MoE is a *discrete* hypernetwork that produces a $\theta_I$ by selecting from a fixed bank. The modularity argument is closely related — MoE makes the primary "expert" small and pushes capacity into the routing system; Galanti & Wolf's theory loosely justifies why this works.
- **`perez_2018_film.md`** (other batch — FiLM): FiLM is a hypernetwork restricted to producing per-channel affine $(\gamma, \beta)$ only. Whether the modularity bound of Thm. 4 is *tight* for FiLM is an open question — FiLM's restricted output structure may make $N_g$ scale less favourably than the general $O(\epsilon^{-m_1/r})$.
- **`andreas_2016_neural_module_networks.md` / `hu_2017_e2e_module_networks.md`** (this batch): Module networks are a different way to instantiate modularity — discrete composition of pre-trained modules vs. continuous weight generation. The theory here does not directly cover module networks, but the conceptual takeaway (capacity belongs in the router/composer, not in shared primaries) transfers.
