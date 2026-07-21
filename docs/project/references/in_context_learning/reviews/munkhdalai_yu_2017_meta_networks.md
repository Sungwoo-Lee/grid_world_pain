> **Per-paper review — in-context-learning corpus, paper 30 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§30); content is identical. Manifest: [[in_context_learning_sources]].

# 30. Munkhdalai & Yu 2017 — Meta Networks

**PDF:** `docs/project/references/in_context_learning/sources/Munkhdalai and Yu 2017 - Meta Networks.pdf`
· ICML 2017 · arXiv 1703.00837

## Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain terms.** A standard neural network needs lots of labelled data and, worse,
*forgets* old skills when you train it on something new. Humans learn a new concept from a single
example and keep their old skills. Meta Networks (**MetaNet**) is an early (2017) "learning to learn"
architecture that gives a network this rapid-adaptation ability by splitting it into two cooperating
learners with weights that change on *different time-scales*.

- A **base learner** does the actual task (e.g. classify an image). It has ordinary **slow weights**
  (trained by gradient descent) plus **fast weights** that are conjured up fresh for each input.
- A **meta learner** is the weight factory. It watches how the base learner is doing — specifically it
  reads the **loss gradients** the base learner produces on a few example images ("meta information") —
  and from those gradients it *generates* the fast weights that will make the base learner good at the
  current task.

The trick that ties them together is **layer augmentation**: each layer runs its input through *both*
the slow weights and the freshly-generated fast weights, applies a nonlinearity to each, and adds the
two activation vectors. Fast and slow weights are "feature detectors in two numeric domains" unified by
the ReLU.

**Key findings.** On the standard one-shot image benchmarks of the day (Omniglot, Mini-ImageNet) MetaNet
beat prior methods by up to 6% (49.2% on 5-way 1-shot Mini-ImageNet vs 43.6% for Matching Nets). It also
showed three properties the project should note: (i) a meta learner trained to parameterize one base
network could parameterize a *different, fixed-weight* CNN at test time; (ii) it could be trained on
5-way tasks and generalize to 20-way (and even 100-way) tasks by hot-swapping a new softmax layer that
the meta learner parameterizes on the fly; (iii) it exhibited limited **continual learning** — training
on MNIST after Omniglot could even *improve* Omniglot accuracy (reverse transfer) up to a point.

**Initial takeaway.** MetaNet is a conceptual ancestor of the whole "weight-generation" arc: it makes
the case that *loss gradients* are a useful conditioning signal, and that generating fast weights (rather
than fine-tuning slow ones) is a viable route to one-shot adaptation. Its limitation — juggling three
weight time-scales via crude additive layer augmentation — is exactly what the later outer-product FWPs
and Transformer hypernetworks streamline.

## Phase 2: Graduate-Level Deep Dive

**Weight-generation mechanism.** MetaNet uses *loss gradients as meta-information* and two learned
weight-generators. Let $b$ be the base learner with slow weights $W$, $u$ the dynamic representation
network with slow weights $Q$, and let $m$ (weights $Z$) and $d$ (weights $G$) be the fast-weight
generators. The full trainable parameter set is $\theta=\{W,Q,Z,G\}$.

*Example-level fast weights.* For each support example $x'_i$ the base learner incurs a task loss and its
gradient w.r.t. the slow weights is the meta-information:
$$
\mathcal{L}_i=\mathrm{loss}_{\text{task}}\big(b(W,x'_i),y'_i\big),\qquad
\nabla_i=\nabla_W \mathcal{L}_i,\qquad
W^{*}_i = m(Z,\nabla_i).
$$
So the generator $m$ is a neural network that maps a *gradient tensor* to a *weight tensor* $W^{*}_i$.
Crucially $m$ is applied **coordinate-wise across the gradient** (parameters $Z$ shared across
coordinates, gradients preprocessed with the Andrychowicz et al. $p=7$ rule) — the same trick used by
learned optimizers — which is what makes generating a large weight tensor tractable.

*Task-level fast weights.* A representation loss $\mathrm{loss}_{\text{emb}}$ (cross-entropy for 1-shot,
contrastive when >1 example/class) produces gradients that the generator $d$ (an LSTM over the $T$
sampled examples) summarizes into a single per-task weight $Q^{*}$:
$$
\mathcal{L}_i=\mathrm{loss}_{\text{emb}}\big(u(Q,x'_i),y'_i\big),\quad
\nabla_i=\nabla_Q \mathcal{L}_i,\quad
Q^{*}=d\big(G,\{\nabla_i\}_{i=1}^{T}\big).
$$

*Storing and reading fast weights (a soft-attention memory).* The example-level fast weights
$\{W^{*}_i\}_{i=1}^{N}$ are stored in a memory $M$, indexed by task-dependent embeddings
$r'_i=u(Q,Q^{*},x'_i)$ collected in $R$. At query time, for input $x_i$ the model embeds it
$r_i=u(Q,Q^{*},x_i)$, computes cosine-similarity attention against the index, and reads a **soft mixture
of stored fast weights**:
$$
a_i=\mathrm{attention}(R,r_i),\qquad
W^{*}_i=\mathrm{softmax}(a_i)^{\top} M .
$$
This is a differentiable, content-addressable weight memory — an idea that reappears in the FWP lineage
(where the "memory" is a single accumulating fast-weight matrix rather than a slot table).

**Layer augmentation (the fast⊕slow integration rule).** For an augmented layer with input $h$, slow
weight $W$, fast weight $W^{*}$ and nonlinearity $\phi$ (ReLU),
$$
\mathrm{AugLayer}(h)=\phi(Wh)+\phi(W^{*}h),
$$
with the final softmax layer aggregating *before* normalization. The paper found that a base learner
built from fast weights **alone** fails to converge (collapses to a constant classifier); the slow-weight
"anchor" is necessary. This is a load-bearing empirical fact for the project: **pure hypernetwork
generation of a whole layer can be unstable; a residual/additive combination with a stably-trained base
path helps** — directly analogous to why FiLM modulates (scales/shifts) a base feature rather than
replacing it.

**Training loop (Algorithm 1, condensed).** (1) Sample $T$ support examples, compute representation-loss
gradients, generate $Q^{*}=d(G,\{\nabla\})$. (2) For each of $N$ support examples, compute task-loss
gradient $\nabla_i$, generate $W^{*}_i=m(Z,\nabla_i)$, store in memory $M$, store index $r'_i$ in $R$.
(3) For each of $L$ training examples, embed, attend over $R$, read $W^{*}_i=\mathrm{softmax}(a_i)^\top M$,
accumulate $\mathcal{L}_{\text{train}}$. (4) Update $\theta=\{W,Q,Z,G\}$ by $\nabla_\theta
\mathcal{L}_{\text{train}}$. Note the outer loss backpropagates *through* the fast-weight generation, so
$m,d$ learn to produce useful weights end-to-end.

**Why this matters for the project.** MetaNet is the "explicit hypernetwork + gradient-as-context" end of
the spectrum. Two of its findings transfer directly to any modulation-based agent: (i) *gradients are an
informative conditioning signal* (relevant if a modulator were ever conditioned on a TD-error or
prediction-error signal), and (ii) *fast weights must be added to, not substituted for, a stable slow
path*. Its RL relevance is speculative in the paper ("MetaNet can readily be applied to parameterize
policies in RL") but never demonstrated — the later FWP papers (Irie 2021/2022) actually do the RL
experiments.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Deep nets need large data and forget; humans do one-shot + continual learning via
  meta-learning (Harlow 1949). MetaNet = base learner (task space) + meta learner (task-agnostic meta
  space) + external memory. Three weight time-scales: slow (gradient descent), task-level fast (per task),
  example-level fast (per input). Meta-information = **loss gradients**. Two losses: representation
  (embedding) loss and main task loss.
- **§2 Related Work.** One-shot learning via generative/metric methods (Lake 2015 BPL; Koch Siamese;
  Vinyals Matching Nets; Santoro MANN/NTM; Ravi&Larochelle LSTM optimizer). Meta-optimizers
  (Andrychowicz 2016). Fast weights: Hinton&Plaut 1987, Ba 2016, Schmidhuber 1992/1993. Weight
  generation: Gomez&Schmidhuber (RNN→fast weights), De Brabandere dynamic filters, Ha et al. HyperNetworks
  (slow weights for RNNs). MetaNet generates fast weights at *two* time-scales; novel **layer
  augmentation** for integration. Ties to memory-augmented NNs.
- **§3 Meta Networks.** Two modules + memory; three procedures: meta-info acquisition, fast-weight
  generation, slow-weight optimization. Algorithm 1 (one-shot SL). Task = support set + training set,
  labels consistent within task but vary across tasks.
  - **§3.1 Meta Learner.** Representation net $u$ (slow $Q$, task-fast $Q^{*}$); generators $m$
    (Eq.1, gradient→example fast weight $W^{*}_i$, MLP with weights $Z$) and $d$ (Eqs.2–4, gradient
    summary→task fast weight $Q^{*}$, LSTM with weights $G$). Contrastive-loss variant (Eqs.6–7) for
    >1 example/class. Memory read via cosine attention + softmax (Eqs.8–9).
  - **§3.2 Base Learner.** $b$ with slow $W$ + example-fast $W^{*}$; produces gradient meta-info
    (Eqs.10–11); prediction $P(\hat y_i|x_i,W,W^{*}_i)=b(W,W^{*}_i,x_i)$ (Eq.12). Can consume $r_i$
    instead of $x_i$ to share representations.
  - **§3.3 Layer Augmentation.** $\phi(Wh)+\phi(W^{*}h)$; fast-only base learner fails to converge.
- **§4 Results.** Omniglot previous split (Table 1): MetaNet 98.95/98.67/97.11/97.0 (5/10/15/20-way),
  +0.5–2% over SOTA; MetaNet− (no task-fast $Q^{*}$) close but worse; MetaNet+ (extra task-fast in base)
  hurts. Mini-ImageNet (Table 2): 49.21% 5-way 1-shot, +6%. Omniglot standard split (Table 3): 95.92%
  20-way, ~human. Generalization: N-way train / K-way test (Table 4) — trained-harder-tested-easier helps,
  10-way→100-way ≈65%; rapid parameterization of a *fixed-weight* new CNN works; meta-level continual
  learning (Omniglot→MNIST) shows reverse transfer up to ~2400 MNIST trials then mild forgetting (−1.7%).
- **§5 Discussion.** Loss gradients are a promising, problem-independent meta-info; layer augmentation is
  a bottleneck when many weight time-scales coexist; future: discover own augmentation schema, apply to RL
  policies and sequence models.
- **Appendices A–B.** Training details (CNN backends, LSTM $d$ 20 units, MLP $m$ 20 units, Adam lr 1e−3);
  MNIST out-of-domain (MetaNet 74.8% vs Matching Net 72.0%).

---
