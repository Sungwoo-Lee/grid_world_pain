> **Per-paper review — continual-learning corpus, paper 5 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§5); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 5. Rebuffi et al. (2017) — iCaRL: Incremental Classifier and Representation Learning

**PDF:** `docs/project/references/continual_learning/sources/Rebuffi et al. 2017 - iCaRL.pdf`
**Venue:** CVPR 2017 (Oxford / IST Austria). **Type:** algorithm / empirical (class-incremental image classification).
**Primer link:** the **replay** family template for *class-incremental* learning — exemplar storage + distillation + nearest-mean classification.

## Phase 1: Foundational Overview

**Introduction (plain language).** The earlier methods here (EWC, SI, progressive nets) mostly assume you know which task an input belongs to, and you keep learning the *same set of classes*. iCaRL tackles a harder, more realistic setting: **class-incremental learning**. Classes arrive over time in batches — first you learn to recognize cats and dogs, later birds and fish, later still lizards — and *at any moment* the system must classify an image into *any* class seen so far, without being told which batch it came from. A child at the zoo learns new animals without forgetting the pet at home; iCaRL wants that. The constraints: it must (i) learn from a stream where classes appear at different times, (ii) always give a competitive single classifier over *all* classes so far, and (iii) keep memory bounded (not store all data).

**Key finding.** Naively finetuning a network on each new batch causes accuracy to collapse (the network ends up predicting only the most recent classes — its confusion matrix has all mass on the last batch). iCaRL avoids this by combining **three** ingredients: (1) classify by **nearest-mean-of-exemplars** rather than the network's own output layer; (2) store a small, fixed budget of **exemplar images** per class, chosen by a **herding** procedure that best approximates each class's mean feature; (3) train the representation with a **classification + distillation loss** that rehearses old exemplars and preserves old outputs. On iCIFAR-100 and iImageNet, iCaRL learns 100–1000 classes incrementally where finetuning, fixed-representation, and distillation-only (LwF) baselines fail.

**Initial takeaway.** iCaRL is the canonical *exemplar-replay* method for growing class sets, and the paper is unusually clear about *why each of its three parts matters* (via ablations). Its central insight: when the feature representation $\varphi$ keeps changing, a network's learned output weights $w_y$ go stale — but a *nearest-mean* classifier that recomputes prototypes from stored images automatically tracks the changing representation, so it is robust to exactly the drift that causes forgetting.

## Phase 2: Graduate-Level Deep Dive

**Architecture and outputs.** iCaRL uses a CNN as a trainable **feature extractor** $\varphi: \mathcal{X} \to \mathbb{R}^d$ (feature vectors L2-normalized) followed by a single layer of sigmoid outputs, one per class seen so far. For class $y \in \{1,\dots,t\}$ ($t$ = classes observed so far):

$$
g_y(x) = \frac{1}{1 + \exp(-a_y(x))}, \qquad a_y(x) = w_y^\top \varphi(x). \tag{1}
$$

Crucially, iCaRL uses the network **only for representation learning**, *not* for the final classification decision.

**(1) Nearest-mean-of-exemplars classification.** For each class $y$, a prototype is the mean feature over its stored exemplar set $P_y$:

$$
\mu_y = \frac{1}{|P_y|} \sum_{p \in P_y} \varphi(p).
$$

An image $x$ is assigned the class whose prototype is nearest:

$$
y^* = \operatorname*{arg\,min}_{y=1,\dots,t} \; \big\| \varphi(x) - \mu_y \big\|. \tag{2}
$$

*Why this beats the network's own softmax/argmax.* The usual rule $y^* = \arg\max_y g_y(x) = \arg\max_y w_y^\top \varphi(x)$ has the output weights $w_y$ **decoupled** from the feature map $\varphi$: whenever $\varphi$ changes (as it must, since iCaRL keeps learning the representation), *all* the $w_y$ become inconsistent with it unless separately retrained — and that inconsistency *is* observable as catastrophic forgetting. The nearest-mean rule has no decoupled weights: the prototypes $\mu_y$ are recomputed from $\varphi$, so they **automatically move with the representation**. Because features are normalized, Eq. (2) is equivalent to $y^* = \arg\max_y \mu_y^\top \varphi(x)$ — a linear classifier whose "weights" $\mu_y$ are *tied* to the data representation rather than free parameters. This is the conceptual heart of iCaRL, and it directly echoes French's (§1) "shallow vs. deep forgetting": the knowledge isn't destroyed, it was just being read out through a stale head.

**(2) Representation learning — classification + distillation loss.** When data $X_s,\dots,X_t$ for new classes arrives, iCaRL builds a combined training set $D$ of the new images plus the stored exemplars of old classes. *Before* updating, it records the current network's outputs $q_i^y = g_y(x_i)$ for every old class $y \in \{1,\dots,s-1\}$ and every example (these are the distillation targets). It then minimizes a binary-cross-entropy loss with **two blocks** — a classification term for new classes and a distillation term for old classes:

$$
\ell(\Theta) = -\sum_{(x_i,y_i)\in D} \Bigg[ \underbrace{\sum_{y=s}^{t} \Big( \delta_{y=y_i}\log g_y(x_i) + \delta_{y\neq y_i}\log(1-g_y(x_i)) \Big)}_{\text{classification (new classes)}} + \underbrace{\sum_{y=1}^{s-1} \Big( q_i^y \log g_y(x_i) + (1-q_i^y)\log(1-g_y(x_i)) \Big)}_{\text{distillation (old classes)}} \Bigg].
$$

The classification block pushes the network to predict the correct new-class indicator; the distillation block (Hinton-style, but *within one network across time* rather than teacher→student) pulls each old-class output back toward its pre-update value $q_i^y$, so previously-learned discriminative information is not lost. Two modifications to plain finetuning make this work: the training set is *augmented* with exemplars (so some old-class data is present — and importantly exemplars are stored **as images**, not features, since features go stale), and the loss is *augmented* with distillation.

**(3) Exemplar management — herding selection.** With $t$ classes seen and a global budget of $K$ exemplars, each class keeps $m = K/t$ exemplars (the budget is always fully used, never exceeded; adding classes *shrinks* per-class $m$). Selection (Algorithm 4) is a greedy **herding** that builds a *prioritized list* $p_1,\dots,p_m$ so that the running mean of the first $k$ exemplars best approximates the true class mean $\mu = \tfrac{1}{n}\sum_{x\in X}\varphi(x)$:

$$
p_k = \operatorname*{arg\,min}_{x \in X} \; \left\| \mu - \frac{1}{k}\Big[ \varphi(x) + \sum_{j=1}^{k-1} \varphi(p_j) \Big] \right\|.
$$

The list is *ordered by importance* — earlier exemplars matter more. This makes **reduction** trivial and data-independent (Algorithm 5): to shrink from $m'$ to $m$, just keep the first $m$ and drop the rest. The design goal: the first-$k$ prefix of the list is always a good class-mean approximation, *even after future reductions*, without ever needing the (now-unavailable) full class data again. Herding (Welling 2009) reaches a target approximation quality with fewer samples than random subsampling.

**Resource profile.** Memory = feature-extractor parameters + $K$ exemplar images + $t$ weight vectors. If a class upper bound is known, pre-allocate weight vectors and spend the rest on exemplars; otherwise grow weight vectors and shrink $m$ over time. Selection runs once per class (when first seen); thereafter only the cheap reduction runs — no access to old training data required. This is the property that satisfies criterion (iii) that finetuning-with-full-replay violates.

**Experiments and ablations.** Protocol: fix a random class order, train incrementally in batches, after each batch test on all classes seen so far; report per-batch accuracy curves and their average ("average incremental accuracy"). Benchmarks: iCIFAR-100 (batches of 2/5/10/20/50 classes; 32-layer ResNet; $K=2000$) and iILSVRC (100 or 1000 ImageNet classes; 18-layer ResNet; $K=20000$). iCaRL clearly beats LwF.MC (distillation, no exemplars), fixed-representation, and finetuning — and the gap widens the more incremental the setting (smaller batches). The **confusion-matrix diagnostic** (Fig. 3) is memorable: finetuning predicts *only* last-batch classes; fixed-representation is biased toward the *first* batch (which set the frozen features); LwF.MC over-predicts recent batches; **iCaRL's confusion matrix is nearly uniform** across classes — no early/late bias, i.e. no catastrophic forgetting.

*Component ablation (Table 1a, iCIFAR-100 average accuracy):*

| batch | iCaRL | hybrid1 (net-output classify) | hybrid2 (no distillation) | hybrid3 (exemplars in repr only) | LwF.MC (distill, no exemplars) |
|---|---|---|---|---|---|
| 2 classes | 57.0 | 36.6 | 57.6 | 57.0 | 11.7 |
| 5 | 61.2 | 50.9 | 57.9 | 56.7 | 32.6 |
| 10 | 64.1 | 59.3 | 59.9 | 58.1 | 44.4 |
| 20 | 67.2 | 65.6 | 63.2 | 60.5 | 54.4 |
| 50 | 68.6 | 68.2 | 65.3 | 61.5 | 64.5 |

Readings: nearest-mean matters most at small batch sizes (iCaRL ≫ hybrid1 at 2 classes: 57.0 vs 36.6 — many representation updates make the network head go stale fastest); distillation can even *hurt* at tiny batches but helps at large ones (compare iCaRL vs hybrid2); exemplars-in-representation alone (hybrid3) already beats distillation-alone (LwF.MC), showing stored exemplars are the dominant anti-forgetting force. Replacing herding-means with the true nearest-class-mean (Table 1b, requires storing all data) changes accuracy only marginally — herding picks representative exemplars well. Larger memory budget $K$ helps all exemplar-based variants; given ≥1000 prototypes, mean-of-exemplars ≈ NCM.

*Relevance note.* iCaRL is the exemplar-replay reference and the direct predecessor to GEM (§6) in the replay lineage — both store a small per-task/per-class memory, but iCaRL *rehearses + distills* to keep outputs invariant, whereas GEM uses the stored gradients only as *inequality constraints* (allowing positive backward transfer, which iCaRL's invariance-preserving distillation forbids). The nearest-mean-tracks-the-representation trick is a clean idea for any setting where a learned feature space drifts under a fixed/stale readout.

## Appendix: Section-by-Section Backbone

- **Abstract.** iCaRL learns classifiers + representation together in class-incremental fashion; only a few classes present at once; CIFAR-100 and ImageNet experiments show it succeeds where others fail.
- **§1 Introduction.** Class-incremental definition (criteria i–iii: stream, competitive multi-class classifier any time, bounded memory). Naive SGD ⇒ catastrophic forgetting; existing methods limited to fixed representations. Three components: nearest-mean-of-exemplars, herding-based exemplar selection, representation learning via distillation + prototype rehearsal.
- **§2 Method.** 2.1 Architecture: CNN feature extractor $\varphi$ + sigmoid outputs Eq. (1); network for representation only. Algorithms 1 (classify) & 2 (incremental train). 2.2 Nearest-mean classification Eq. (2); decoupled-weights argument (why network-output classification forgets; prototypes track $\varphi$). 2.3 Representation learning (Algorithm 3): combined set of new data + exemplars; store pre-update outputs $q_i^y$; classification + distillation BCE loss; store exemplars as images not features. 2.4 Exemplar management: $m=K/t$; herding construction (Algorithm 4, greedy mean-approximation, prioritized list); reduction (Algorithm 5, keep first $m$); herding vs random subsampling.
- **§3 Related work.** Fixed-representation methods (NCM, Mensink et al.; open-set; ensembles; zero-shot). Representation-learning methods; McCloskey catastrophic forgetting; two classical strategies (freeze/grow vs. rehearsal). Freeze/grow (progressive nets, tree-structured) violates bounded memory; iCaRL uses rehearsal + within-network distillation (LwF connection).
- **§4 Experiments.** Benchmark protocol (fixed random order, per-batch test on seen classes, average incremental accuracy). iCIFAR-100 (2/5/10/20/50 per batch, 32-layer ResNet, $K=2000$) and iILSVRC-small/full (18-layer ResNet, $K=20000$). 4.1 Results: iCaRL > LwF.MC > fixed-repr > finetuning; gap grows with incrementality; confusion matrices (Fig. 3) — iCaRL uniform, finetune last-batch-only, fixed-repr first-batch bias, LwF.MC recency bias. 4.2 Differential analysis: hybrid1/2/3 ablation (Table 1a); NCM comparison (Table 1b); memory-budget curve (Fig. 4).
- **§5 Conclusion.** Three components recap; exemplars are the main driver; still below batch (joint) training; future: exemplar-free (autoencoder-encoded features), privacy settings.

---
