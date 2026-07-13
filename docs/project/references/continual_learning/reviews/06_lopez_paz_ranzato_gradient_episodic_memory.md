> **Per-paper review — continual-learning corpus, paper 6 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§6); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 6. Lopez-Paz & Ranzato (2017) — Gradient Episodic Memory (GEM)

**PDF:** `docs/project/references/continual_learning/sources/Lopez-Paz & Ranzato 2017 - Gradient Episodic Memory (GEM).pdf`
**Venue:** NIPS 2017 (Facebook AI Research). **Type:** algorithm + evaluation framework (supervised continuum).
**Primer link:** the **replay** family "with a twist" — store old examples, but use them to *constrain the gradient* so old-task loss never increases; uniquely allows *positive backward transfer*.

## Phase 1: Foundational Overview

**Introduction (plain language).** GEM contributes two things. First, a **measurement framework**: how should we even score a continual learner? Beyond average accuracy, GEM defines **backward transfer** (does learning a new task *help or hurt* old tasks?) and **forward transfer** (does having learned earlier tasks *help* a new one before you train on it?). Catastrophic forgetting is just *large negative backward transfer* in this language. Second, an **algorithm**: GEM stores a small episodic memory of examples from each past task, and at every gradient step it *checks* whether the proposed update would increase the loss on any stored past task. If it would, GEM **rotates** the gradient to the nearest direction that doesn't — so old-task losses are never allowed to rise, but *can* fall (which is *positive* backward transfer). The setting is deliberately harsh and human-like: many tasks, few examples each, **each example seen only once**.

**Key finding.** On MNIST-permutations, MNIST-rotations, and incremental CIFAR-100 (each with 20 tasks, single pass), GEM minimizes forgetting (near-zero or *positive* backward transfer) and matches or beats EWC and iCaRL — GEM even reaches the "oracle" accuracy of iid multi-task training on MNIST-rotations while EWC lags. Its cost advantage: the constrained-optimization step is solved in the space of *tasks-so-far* ($t-1$ variables), not parameters (millions), via a small quadratic program.

**Initial takeaway.** GEM reframes continual learning as **constrained optimization**: minimize the current task's loss *subject to not increasing any past task's loss*. That inequality-constraint stance is what distinguishes it from iCaRL/LwF (which enforce output *invariance* via distillation, forbidding backward transfer) and from EWC/SI (which softly penalize *weight* movement). GEM's episodic-memory-as-constraint is the direct realization of French's (§1) rehearsal idea, but expressed on gradients rather than on the loss itself.

## Phase 2: Graduate-Level Deep Dive

**The continuum and its challenges.** Instead of an iid training set, the learner sees an ordered stream of triples

$$
(x_1,t_1,y_1),\dots,(x_i,t_i,y_i),\dots,(x_n,t_n,y_n),
$$

where $t_i \in \mathcal{T}$ is a **task descriptor**, and $(x_i,y_i)\sim P_{t_i}$ is drawn iid *only locally* (a whole run of examples from one task precedes the switch to the next). Goal: learn $f:\mathcal{X}\times\mathcal{T}\to\mathcal{Y}$. Three challenges absent from ERM: non-iid input (tasks switch in blocks), catastrophic forgetting (new tasks may hurt old), and the *opportunity* for transfer (related tasks could help each other, forward and backward).

**The evaluation framework.** After the model finishes training on task $t_i$, evaluate its test accuracy on *all* $T$ tasks, filling a matrix $R \in \mathbb{R}^{T\times T}$ where $R_{i,j}$ = accuracy on task $t_j$ after finishing task $t_i$. Let $\bar{b}_j$ be the accuracy on task $j$ at random initialization. Three scalar metrics:

$$
\text{ACC} = \frac{1}{T}\sum_{i=1}^{T} R_{T,i}, \tag{2}
$$

$$
\text{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1} \big(R_{T,i} - R_{i,i}\big), \tag{3}
$$

$$
\text{FWT} = \frac{1}{T-1}\sum_{i=2}^{T} \big(R_{i-1,i} - \bar{b}_i\big). \tag{4}
$$

ACC is final average accuracy. **BWT** compares each task's accuracy *at the end* ($R_{T,i}$) to its accuracy *right after it was learned* ($R_{i,i}$): negative BWT = forgetting, positive BWT = learning later tasks *improved* earlier ones. **FWT** compares accuracy on a not-yet-trained task ($R_{i-1,i}$, before task $i$ is seen) to chance ($\bar{b}_i$): positive FWT = zero-shot benefit from earlier tasks. Larger is better on all three; ties on ACC are broken by BWT/FWT. Catastrophic forgetting is precisely *large negative BWT*.

**The episodic memory and the constrained objective.** GEM keeps a memory $\mathcal{M}_k$ of examples from task $k$ (a budget of $M$ locations total; $m = M/T$ per task if $T$ is known; last-$m$ examples by default). The memory loss for task $k$ is

$$
\ell(f_\theta, \mathcal{M}_k) = \frac{1}{|\mathcal{M}_k|}\sum_{(x_i,k,y_i)\in \mathcal{M}_k} \ell\big(f_\theta(x_i,k), y_i\big). \tag{5}
$$

Naively minimizing the current loss plus (5) would *overfit* the stored examples; distillation (à la iCaRL) would keep old predictions *invariant* but forbid positive backward transfer. GEM instead uses (5) as **inequality constraints** — don't let old-task loss *increase*, but allow it to *decrease*. When observing $(x,t,y)$:

$$
\min_\theta \; \ell(f_\theta(x,t),y) \quad \text{s.t.} \quad \ell(f_\theta, \mathcal{M}_k) \le \ell(f_\theta^{t-1}, \mathcal{M}_k) \;\; \forall\, k<t, \tag{6}
$$

where $f_\theta^{t-1}$ is the predictor at the end of task $t-1$.

**Two observations that make (6) cheap.** (1) We needn't store old predictors $f_\theta^{t-1}$ — it suffices to guarantee the memory losses *don't increase* at each parameter update $g$. (2) Assuming local linearity (valid for small steps) and that the memory represents past tasks, an increase in past-task loss is detectable by the **angle between the proposed update and the past-task gradient**. Define the proposed gradient $g = \partial\ell(f_\theta(x,t),y)/\partial\theta$ and each past-task memory gradient $g_k = \partial\ell(f_\theta,\mathcal{M}_k)/\partial\theta$. The constraints (6) become inner-product constraints:

$$
\langle g, g_k\rangle = \left\langle \frac{\partial\ell(f_\theta(x,t),y)}{\partial\theta}, \frac{\partial\ell(f_\theta,\mathcal{M}_k)}{\partial\theta}\right\rangle \ge 0, \quad \forall\, k<t. \tag{7}
$$

If all inner products are $\ge 0$ (the update points "with", not "against", each past gradient), then $g$ does not increase past-task loss, and we apply it unchanged. If any is violated, GEM **projects** $g$ onto the closest gradient $\tilde{g}$ (in squared $\ell_2$) that satisfies all constraints:

$$
\min_{\tilde{g}} \; \tfrac{1}{2}\|g - \tilde{g}\|_2^2 \quad \text{s.t.} \quad \langle \tilde{g}, g_k\rangle \ge 0 \;\; \forall\, k<t. \tag{8}
$$

**Solving (8) via its dual — the derivation.** Problem (8) is a quadratic program on $p$ variables ($p$ = #parameters, potentially millions). The generic QP primal and its dual (Dorn 1960) are

$$
\min_z \; \tfrac{1}{2} z^\top C z + p^\top z \;\; \text{s.t.}\; Az\ge b, \qquad\qquad \min_{u,v}\; \tfrac{1}{2}u^\top C u - b^\top v \;\; \text{s.t.}\; A^\top v - C u = p,\; v\ge 0. \tag{9,10}
$$

Write (8) in this form. Expanding $\tfrac12\|g-\tilde g\|^2 = \tfrac12\tilde g^\top\tilde g - g^\top \tilde g + \tfrac12 g^\top g$ and dropping the constant $g^\top g$, the GEM primal is

$$
\min_z \; \tfrac{1}{2} z^\top z - g^\top z \quad \text{s.t.}\quad G z \ge 0, \qquad G = -(g_1,\dots,g_{t-1}),
$$

i.e. $C=I$, $p=-g$, $A=G$, $b=0$, with the optimizer $z=\tilde g$. Substituting $C=I$, $b=0$, $p=-g$ into the dual (10): the equality constraint $A^\top v - Cu = p$ becomes $G^\top v - u = -g$, i.e. $u = G^\top v + g$; the objective $\tfrac12 u^\top u$ (since $b=0$) becomes $\tfrac12(G^\top v + g)^\top(G^\top v + g) = \tfrac12 v^\top GG^\top v + g^\top G^\top v + \tfrac12 g^\top g$. Dropping the constant, the **GEM dual** is a QP in only $t-1$ variables:

$$
\min_v \; \tfrac{1}{2} v^\top G G^\top v + g^\top G^\top v \quad \text{s.t.}\quad v \ge 0. \tag{11}
$$

Solve (11) for $v^\star$ (only $t-1 \ll p$ variables — the number of past tasks), then **recover the projected gradient** as

$$
\tilde{g} = G^\top v^\star + g,
$$

and take the SGD step $\theta \leftarrow \theta - \alpha\tilde{g}$. In practice a small constant $\gamma \ge 0$ added to $v^\star$ biases the projection toward *beneficial backward transfer*. This dual reduction is GEM's efficiency trick: the projection costs a QP the size of the task count, not the parameter count.

**Algorithm (per continuum).** For each task in sequence, for each incoming $(x,y)$: add to memory $\mathcal{M}_t$; compute $g=\nabla_\theta\ell(f_\theta(x,t),y)$ and $g_k=\nabla_\theta\ell(f_\theta,\mathcal{M}_k)$ for all $k<t$; project via (11) to get $\tilde g$; step $\theta\leftarrow\theta-\alpha\tilde g$. After finishing a task, evaluate on all test tasks to fill a row of $R$.

**A causal-compression reading.** The authors note GEM learns the subset of correlations *common across* the task distributions — invariant/causal structure — and can even predict without task descriptors (used in their MNIST experiments). Causal predictions are invariant across environments, hence the most compressed representation of a set of distributions.

**Experiments.** 20 tasks each, single pass: MNIST-permutations (unrelated input per task), MNIST-rotations (fixed rotation 0–180°), incremental CIFAR-100 (disjoint class subsets, shared input distribution, per-task output head). Architectures: MLP 2×100 ReLU (MNIST); reduced ResNet-18 (CIFAR). Baselines: single (one net all tasks), independent (per-task net), multimodal (per-task input layer), EWC, iCaRL (CIFAR only). Results (Fig. 1): GEM has the least negative — sometimes *positive* — BWT, with negligible/positive FWT, and ACC ≥ competitors. GEM beats EWC while using less compute (Table 1: 77s vs 179s on MNIST-permutations) because its QP is over 20 task-variables not ~1.1M parameter-variables; its bottleneck is computing per-task gradients each iteration. GEM's ACC grows with memory size and beats iCaRL across memory budgets (Table 2: at 5120 memories GEM 0.654 vs iCaRL 0.508 on CIFAR). Critically (Table 3, MNIST-rotations), as passes-per-task increase, memory-less methods forget *more* (BWT more negative) while GEM stays high and even matches the iid "oracle upper bound" (single-shuffled-data) — GEM: 0.86/+0.05 (1 epoch) vs oracle 0.83/-0.00.

*Relevance note.* GEM is the most directly **RL-relevant algorithmic idea** in this shard for a gradient-based agent: it is a *drop-in modification of the SGD step* (project the gradient against stored past-task gradients) with no architecture change and no distillation, and its BWT/FWT metrics are a clean way to *quantify* forgetting vs. transfer in any sequential-task or curriculum setting — exactly the kind of measurement the primer's curriculum critique ([[curriculum_underperformed_baseline_plasticity_vs_budget]]) needs. The inequality-constraint stance ("never increase old loss, but allow it to fall") is philosophically distinct from every other method here and is what lets GEM show *positive* backward transfer.

## Appendix: Section-by-Section Backbone

- **Abstract.** Continual learning where examples are seen once, one-by-one; proposes evaluation metrics (transfer + forgetting) and GEM, which alleviates forgetting while allowing beneficial backward transfer; strong on MNIST/CIFAR variants.
- **§1 Introduction.** ERM assumes iid; human learning is ordered, single-pass, finite memory, multi-task ⇒ ERM breaks ⇒ catastrophic forgetting. Continuum of data Eq. (1) with task descriptors $t_i$. Three ERM-unknown challenges: non-iid input, catastrophic forgetting, transfer opportunity.
- **§2 A Framework for Continual Learning.** Locally-iid continuum; predictor $f:\mathcal{X}\times\mathcal{T}\to\mathcal{Y}$. Task descriptors (integers or structured; enable zero-shot; disambiguate same-input-different-target). Training protocol ("more human-like": many tasks, few examples, single pass, report transfer + forgetting). Backward/forward transfer definitions; matrix $R$; ACC/BWT/FWT Eqs. (2)–(4); fine-grained learning curves via extra rows.
- **§3 GEM.** Episodic memory $\mathcal{M}_t$; budget $M$, $m=M/T$. Memory loss Eq. (5). Why naive min / distillation are wrong (overfit / forbid backward transfer). Constrained problem Eq. (6); two observations ⇒ inner-product constraints Eq. (7); projection QP Eq. (8); generic primal/dual Eqs. (9)–(10); GEM primal ⇒ dual Eq. (11) in $t-1$ variables; recover $\tilde g = G^\top v^\star + g$; slack $\gamma$ for backward transfer. Causal-compression view. Algorithm 1 (train + evaluate, fills $R$).
- **§4 Experiments.** 4.1 Datasets (MNIST-permutations/rotations, incremental CIFAR-100; 20 tasks, single pass). 4.2 Architectures (MLP 2×100; reduced ResNet-18 + per-task head; plain SGD, grid-searched). 4.3 Methods (single/independent/multimodal/EWC/iCaRL). 4.4 Results: GEM least forgetting + positive BWT (Fig. 1); beats EWC with less compute (Table 1); 4.4.1 memory-size (Table 2, GEM>iCaRL), multi-pass (Table 3, GEM robust, matches iid oracle), task-order robustness.
- **§5 Related work.** Continual/lifelong learning history; modular/freeze-finetune methods (hard to scale); regularization ("synaptic" EWC/SI) vs. "episodic" memory (LwF/iCaRL distillation) — GEM in the latter but uniquely allows positive backward transfer; multitask/transfer/domain-adaptation/zero-shot/one-shot/curriculum contrasts.
- **§6 Conclusion.** Formalized continual learning + metrics + GEM. Three improvement points: leverage structured task descriptors for FWT; advanced memory (coresets); per-iteration per-task backward pass cost.

# Phase 2 — The Learner Degrades (2020–2022)

**The cracks appear.** Around 2020–2022 the field realized that even with catastrophic forgetting handled, the everyday machinery of deep RL — warm-starting, bootstrapping, high replay ratios — quietly damages the learner itself. These five papers isolate the facets: a warm-start generalization gap (7), feature-rank collapse from bootstrapping (8), plasticity as its own measurable quantity (9), capacity loss and its regularizer fix (10), and overfitting to early experience with a reset cure (11). This is the hinge of the corpus: the *forward* failure is a separate disease from forgetting.

---
