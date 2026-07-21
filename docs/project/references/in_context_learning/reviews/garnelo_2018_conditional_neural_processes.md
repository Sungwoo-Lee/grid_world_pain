> **Per-paper review — in-context-learning corpus, paper 18 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§18); content is identical. Manifest: [[in_context_learning_sources]].

# 18. Garnelo et al. 2018 — Conditional Neural Processes

**PDF:** `docs/project/references/in_context_learning/sources/Garnelo et al. 2018 - Conditional Neural Processes.pdf` · ICML 2018 · arXiv:1807.01613

## Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain terms.** Two families of models sit at opposite extremes. Deep neural networks are flexible and scalable but data-hungry — you train one from scratch for every new function, needing lots of points. Gaussian Processes (GPs) are the opposite: they exploit a prior to infer a function's shape from just a handful of points and give calibrated uncertainty — but they are computationally expensive (cost grows cubically, $O((n+m)^3)$) and it is hard to hand-design a good prior kernel. **Conditional Neural Processes (CNPs)** aim to get the best of both: GP-like data efficiency and uncertainty, but as a neural network trained by gradient descent, scaling linearly.

**The core idea.** A CNP looks at a **context set** of observed input-output pairs $O = \{(x_i,y_i)\}$, squeezes it into a single fixed-length summary vector $r$, and then predicts the output distribution at any query ("target") input $x$ by conditioning on $r$. Concretely: each context pair is passed through a shared neural net to get an embedding, the embeddings are **averaged** (a permutation-invariant aggregation, so order does not matter), and the average is fed together with the target input into a decoder that outputs a mean and variance. That's it — one summary, then cheap per-target predictions.

**How it's trained.** You repeatedly sample a random function (e.g. a random GP curve), give the model a *random subset* of its points as context, and ask it to predict the outputs at all points (observed and unobserved), maximizing the log-likelihood. By seeing many functions with varying amounts of context, the CNP learns a *data-driven prior*: with few context points it predicts the average over all plausible functions (with wide uncertainty); as context grows, predictions sharpen. Crucially the "prior knowledge" now comes from *training data*, not a hand-picked kernel.

**Key findings / initial takeaway.** On 1-D regression, image completion (MNIST, CelebA), and one-shot Omniglot classification, CNPs make sensible few-shot predictions with meaningful uncertainty, at $O(n+m)$ test-time cost instead of $O(nm)$ or $O((n+m)^3)$. With few context points they beat kNN and GP baselines; they are robust to context ordering. The trade-off, stated honestly by the authors: CNPs give up the mathematical guarantees of true stochastic processes (their conditionals need not be consistent with any single prior process) in exchange for flexibility and scalability. The takeaway: **a permutation-invariant encode-aggregate-decode network can amortize few-shot function prediction, learning its prior directly from data.**

## Phase 2: Graduate-Level Deep Dive

**Stochastic-process setup.** Let $O = \{(x_i, y_i)\}_{i=0}^{n-1} \subset X\times Y$ be observations and $T = \{x_i\}_{i=n}^{n+m-1} \subset X$ be target inputs. Assume outputs are a realization of a stochastic process: let $P$ be a distribution over functions $f: X\to Y$; for $f\sim P$, set $y_i = f(x_i)$. $P$ defines a joint over $\{f(x_i)\}$ and hence a conditional $P(f(T)\mid O, T)$; the task is to predict $f(x)$ for $x\in T$ given $O$. For a GP, this conditional is Gaussian with closed form but costs $O((n+m)^3)$.

**The CNP as a directly-parametrized conditional process.** A CNP is a conditional stochastic process $Q_\theta$ that defines distributions over $f(x)$ for $x\in T$, **inheriting permutation invariance** in both $O$ and $T$: for permutations $O', T'$,

$$
Q_\theta(f(T)\mid O, T) = Q_\theta(f(T')\mid O, T') = Q_\theta(f(T)\mid O', T).
$$

Invariance in $T$ is enforced by a **factored** likelihood:

$$
Q_\theta(f(T)\mid O, T) = \prod_{x\in T} Q_\theta\big(f(x)\mid O, x\big).
$$

This factorization is the "deterministic-path" signature of the family: each target is predicted independently given the context summary — which is exactly what makes the model cheap and permutation-invariant, but also what prevents it from producing *coherent joint samples* (there is no modeled covariance between targets). This limitation motivates the latent-variable extension below and, later, the whole ANP/TNP lineage.

**The encode–aggregate–decode architecture (Eqs. 1–3).** The defining characteristic is conditioning on $O$ through a **fixed-dimensional** embedding:

$$
r_i = h_\theta(x_i, y_i) \quad \forall (x_i,y_i)\in O \tag{1}
$$
$$
r = r_1 \oplus r_2 \oplus \cdots \oplus r_n \tag{2}
$$
$$
\phi_i = g_\theta(x_i, r) \quad \forall x_i\in T \tag{3}
$$

where $h_\theta: X\times Y \to \mathbb{R}^d$ is the encoder MLP, $\oplus$ is a **commutative** aggregation operator $\mathbb{R}^d\times\mathbb{R}^d\to\mathbb{R}^d$ (in practice the **mean**, $\tfrac1n\sum_i r_i$), $g_\theta: X\times\mathbb{R}^d\to\mathbb{R}^e$ is the decoder, and $\phi_i$ parametrizes the output distribution $Q_\theta(f(x_i)\mid O, x_i) = Q(f(x_i)\mid\phi_i)$. For **regression**, $\phi_i = (\mu_i, \sigma_i^2)$ parametrizes a Gaussian $\mathcal N(\mu_i, \sigma_i^2)$; for **classification**, $\phi_i$ are the logits of a categorical over $c$ classes. The mean aggregator gives two structural properties: (i) permutation invariance in $O$, and (ii) **streaming/online** updates — since $r_1\oplus\cdots\oplus r_n$ is computable in $O(1)$ from $r_1\oplus\cdots\oplus r_{n-1}$, new observations are absorbed incrementally. Overall test-time complexity is $O(n+m)$.

**The training objective — the amortized conditional-predictive loss.** Let $f\sim P$, $O=\{(x_i,y_i)\}_{i=0}^{n-1}$, and $N\sim\text{uniform}[0,\dots,n-1]$. Condition on the first $N$ points $O_N = \{(x_i,y_i)\}_{i=0}^{N}\subset O$ and minimize the negative conditional log-probability over *all* $n$ outputs (observed and unobserved):

$$
\mathcal{L}(\theta) = -\,\mathbb{E}_{f\sim P}\Bigg[\, \mathbb{E}_N\Big[\, \log Q_\theta\big(\{y_i\}_{i=0}^{n-1}\mid O_N, \{x_i\}_{i=0}^{n-1}\big)\Big]\Bigg]. \tag{4}
$$

The gradient is estimated by Monte Carlo, sampling $f$ and $N$. **Interpretation / derivation of what this trains.** Randomizing $N$ forces the model to predict well across *every* context size — small $N$ (much uncertainty) to large $N$ (little uncertainty). Scoring the *observed* points too (targets include $O_N$) gives the model a signal of the residual uncertainty inherent in $P$. Note the parallel to the PFN loss of §1: both are held-out conditional log-likelihoods under a data/function-generating distribution; the CNP samples $f\sim P$ (a GP or dataset generator), the PFN samples $\phi\sim p(\phi)$ then $D\sim p(D\mid\phi)$ — structurally identical amortization, and minimizing either drives $Q_\theta$ toward the true conditionals of the generating process (the CNP paper is careful to note that unlike the PFN framing, the learned conditionals are **not guaranteed to be mutually consistent** with any single prior process — that consistency is exactly the mathematical guarantee traded away).

**The latent-variable extension (§4.2.3) — introducing the latent path.** The factored model cannot produce coherent samples over multiple targets (no covariance). To fix this, add a latent $z$: compute the context representation $r$, which parametrizes a Gaussian over $z$; sample $z$ **once**, then decode *all* targets conditioned on that single draw. Different draws of $z$ give different globally-coherent functions. Training uses a VAE-style variational lower bound of the log-likelihood, with a **conditional prior** $p(z\mid O)$ (sees only observed pixels) and a **posterior** $p(z\mid O, T)$ (sees observed and target pixels). This is the seed of the deterministic-path vs. latent-path distinction that Kim et al. (§3) formalize: the deterministic path (Eqs. 1–3) gives per-target point predictions; the latent path gives a distribution *over functions* with target-correlated samples, and the latent CNP is explicitly described as "an approximated amortized version of Bayesian deep learning."

**Empirical results.** (i) *1-D regression* on GP-generated curves (fixed and switching kernels): CNP recovers mean and uncertainty from few context points, sharpening as context grows; the switching-kernel task is easy for CNP (just change the training set) but hard for a stationary-kernel GP. (ii) *Image completion* (MNIST, CelebA) cast as regression $f:[0,1]^2\to[0,1]^{(3)}$: with 1 non-informative context point the prediction is the dataset average; more context → image-specific detail; variance localizes to edges. CNP can condition on arbitrary pixel patterns (even unseen at training) and query at unseen resolutions/subpixels. Table 1: with few context points CNP beats kNN and GP on CelebA MSE and is order-robust. (iii) *Omniglot one-shot classification*: convolutional encoder, aggregate within each class; CNP reaches 95.3% (5-way 1-shot) at $O(n+m)$ vs $O(nm)$ for MANN/Matching Networks — comparable accuracy, cheaper.

## Appendix: Section-by-Section Backbone

- **Abstract.** DNNs excel at function approximation but train from scratch per function; GPs exploit priors for fast test-time inference but are expensive and hard to design priors for. CNPs combine both: inspired by stochastic-process flexibility, structured as NNs, trained by gradient descent. Accurate few-shot predictions, scale to large datasets. Demonstrated on regression, classification, image completion.
- **§1 Introduction.** Data-efficiency via two-phase learning (learn domain statistics, then a specific task from few points). Supervised learning as function approximation over observations/targets. Two approaches: (a) random-init parametric $g$ per task (most deep learning, limited prior sharing) vs. (b) probabilistic stochastic-process view (GPs; Bayesian inference over function space, but intractable as data grows). CNPs directly parametrize conditional distributions over functions given observations, permutation-invariant, $O(n+m)$ test-time; trained to maximize conditional likelihood of random target subsets. Emphasis: CNPs do *not* implement Bayesian inference directly and conditionals may be inconsistent.
- **§2 Model.** §2.1 Stochastic processes ($P$ over functions, conditional $P(f(T)|O,T)$; GP as example with $O((n+m)^3)$ cost). §2.2 CNPs ($Q_\theta$ permutation-invariant conditional process; factored likelihood $\prod_{x\in T}Q_\theta(f(x)|O,x)$; encode–aggregate–decode Eqs. 1–3; mean aggregator; Gaussian for regression, categorical for classification; streaming $O(1)$ update; $O(n+m)$). §2.3 Training CNPs (random $N$, loss Eq. 4 scoring both observed and unobserved, MC gradient; shifts prior-knowledge burden from analytic prior to empirical data; three summary properties — conditional-distribution model, permutation invariance, $O(n+m)$ scalability; note on latent-variable extension for coherent samples).
- **§3 Related research.** §3.1 Gaussian Processes (sparse GPs, deep GPs, deep kernel learning). §3.2 Meta-learning (GQN as a special case; PixelCNN/memory-augmented few-shot density estimation; Matching Networks as a CNP with concatenation aggregator + explicit distance kernel, $O(nm)$; neural statistician / variational homoencoder; latent CNP as amortized Bayesian DL).
- **§4 Experimental Results.** §4.1 Function regression (1-D GP data, fixed & switching kernels; 3-layer MLP encoder d=128, mean aggregate, 5-layer decoder; GP is an upper bound; CNP recovers mean+uncertainty, handles switching kernels trivially). §4.2 Image completion (regression over pixels). §4.2.1 MNIST (average-of-digits with one context point; edge-localized variance; simple active exploration by max-variance pixel selection). §4.2.2 CelebA (flexibility in context patterns and target resolutions; Table 1 CNP beats kNN/GP with few context points, order-robust). §4.2.3 Latent variable model (add $z$, sample once for coherent targets; VAE lower bound; conditional prior $p(z|O)$, posterior $p(z|O,T)$; sample diversity shrinks as observations grow). §4.3 Classification (Omniglot one-shot; conv encoder, class-wise aggregation; Table 2: 95.3% 5-way 1-shot at $O(n+m)$).
- **§5 Discussion.** CNP flexible at test time, extracts prior from training data; demonstrated across regression/classification/image completion; relations to GPs, deep learning, meta-/few-shot learning; a step toward learning reusable high-level abstractions (a trained CNP encapsulates statistics of a *family* of functions). Implementations are proofs-of-concept, extensible.

---
