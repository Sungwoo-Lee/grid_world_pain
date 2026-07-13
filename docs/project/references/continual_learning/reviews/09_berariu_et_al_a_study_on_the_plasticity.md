> **Per-paper review — continual-learning corpus, paper 9 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§9); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 9. Berariu et al. (2021) — A Study on the Plasticity of Neural Networks

**PDF:** `docs/project/references/continual_learning/sources/Berariu et al. 2021 - A Study on the Plasticity of Neural Networks (preprint).pdf`
**Venue:** arXiv preprint 2021 (v2, Oct 2023). **Authors:** Tudor Berariu, Wojciech Czarnecki, Soham De, Jörg Bornschein, Samuel Smith, Razvan Pascanu, Claudia Clopath (Imperial College London / DeepMind).

## Phase 1 — Foundational Overview

**The problem in one sentence.** This paper takes the Ash & Adams warm-start gap and asks: is losing the ability to reach a fresh network's generalization a *general property of continual learning*, and what governs it? It isolates **plasticity** — the model's continuing ability to *learn well* on a new task — as its own quantity, separate from forgetting.

**A crucial definitional clarification (from the authors' Oct-2023 note).** "Loss of plasticity" has been used for *two different phenomena*, and this paper studies only the first:
1. **Generalization-type plasticity loss (this paper).** The warm-started model can still drive *training* error to zero just as well as a fresh model — but it *generalizes worse*. The failure is in the *quality of the minimum reached*, not the ability to optimize.
2. **Optimization-type plasticity loss (e.g. Dohare et al., continual backprop).** The system loses the ability to reduce training error at all.

The authors explicitly say these should be treated as *separate problems* until shown otherwise. This is an important guardrail for the project: not every "plasticity" citation is about the same failure.

**Key findings (all on ResNet-18 / CIFAR-10, "generalisation gap" = last-100-epoch mean test accuracy of pretrained vs. fresh).**
- The gap reproduces across optimizers (Adam, RMSprop, SGD, SGD+momentum) — so it is about the *minimum's quality*, not the descent trajectory.
- It appears after very few pretraining epochs (5–10 for Adam, before reaching 100% train accuracy) — early stopping won't remove it.
- It persists even when the data distribution *slides smoothly* from the pretrain subset to the full set — profound for RL, where the policy's data distribution drifts continuously.
- **It aggravates with more pretraining stages** and with **larger inter-stage distribution shift** (class-imbalanced splits) — the multi-stage setup is the continual-learning setting, so continual learning genuinely compounds the damage.
- **Bigger models don't fix it** — the gap survives across depths and widths.
- **Only resetting the *top* layers recovers the gap;** keeping the first (few) layers is fine, but the top of the network must be reinitialized, and keeping early layers gives *no* training-speed advantage.
- **A 10× larger tuning learning rate substantially reduces the gap** — the key supporting evidence for their mechanistic hypothesis.

**Initial takeaway.** Pretraining on a compact model that later gets fine-tuned is a hidden cost in continual learning: "when saving data from the past is feasible, retraining models seems a better strategy than using pretrained models." The paper reframes tracking the generalization gap as a *new, under-measured facet of forward transfer*.

## Phase 2 — Graduate-Level Deep Dive

**The generalization gap, formally.** Define the gap as the difference in final generalization between a *pretrained* model (already trained on some tasks) and a *freshly initialized* model, *without constraining the number of updates*. This resembles Chaudhry et al.'s "intransigence" but compares against a model trained *only on the new data* (not a multi-task solution). The focus is the *existence and sign* of the gap under identical training budgets.

**Experimental protocol.** ResNet-18 on CIFAR-10, batch 128, Adam lr $10^{-3}$ (unless noted), optimizer statistics reset between phases. Phase 1: 350 epochs on half the data; Phase 2: 500 epochs on all data. Both phases reach 100% train accuracy. Metric: mean test accuracy over the last 100 tuning epochs (green box, Fig. 1).

**Smooth-transition experiment (RL-relevant).** Instead of a hard 50%→100% switch, each example is sampled in a *single* training run with probability

$$p(n) = 1 - \gamma^{50 n / N}$$

from the full training set (and $1-p$ from the pretrain subset), where $n$ is the current step and $N$ the total steps ($\approx$500 epochs). Varying $\gamma\in\{0.1,\dots,0.9\}$ interpolates from near-instant to very slow distribution shift. Result (Fig. 3): even a *transient*, *slowly-sliding* bias in the data distribution induces the gap — directly analogizing the continuously-drifting data distribution of an RL agent whose policy keeps changing.

**Multi-stage aggravation.** Split the training set into $n$ incremental stages; all but the last are trained to 100% train accuracy. Final test accuracy *degrades monotonically with the number of splits* (Fig. 4, left). Adding *class imbalance* between stages (parameterized by $r$ = ratio of data drawn uniformly from all classes vs. from stage-designated classes) makes it worse: larger inter-stage discrepancy ⇒ larger gap. The class-split methodology (Appendix A.3): partition classes $\{C_1,\dots,C_{n+1}\}$, split $D$ into $D_c, D_u$ with $|D_u|/|D| = r$, and define stage datasets $D_i = D_{i-1}\cup D_{u,i}\cup\{(x,c)\in D_c : c\in C_i\}$.

**Layer-reset probe.** After pretraining, re-sample subsets of layers from the init distribution (Fig. 6, layers 1=first conv, 2–5=four residual modules, 6=FC output). Finding: resetting a *small* subset is insufficient; the gap is recovered only when the *top* of the model (last 4–5 modules) is reinitialized. Keeping the first 1–2 modules is fine but yields *no* training-speed benefit (Appendix Fig. 11) — so warm-starting's promised advantage evaporates once you reset enough to fix generalization. This is the supervised precursor of Nikishin's "reset the last layers" finding (§5).

**Mechanistic hypothesis — two phases of learning.** Building on flat-vs-sharp-minima generalization theory (Hochreiter & Schmidhuber): wide, low-curvature minima generalize better (shorter minimum-description-length; more robust to perturbation). SGD training traverses two phases:
1. **Exploration phase** — parameters "bounce" between critical points, driven by *noise* (mini-batch stochasticity, learning-rate scale, data noise), until captured by a wide basin.
2. **Refinement phase** — parameters follow the gradient flow to the critical point.

**The conjecture.** Pretraining *reduces the gradient noise available during tuning*, weakening the exploration phase and driving convergence to a *narrower* minimum ⇒ the generalization gap. Why less noise? Neural nets *early in training* become insensitive to non-discriminative directions of variation (background patterns, irrelevant features). Those irrelevant features are a *source of exploration noise* in the initial phase. A pretrained model has already filtered them out, so at the start of tuning there is less noise ⇒ less exploration ⇒ narrower minimum ⇒ worse generalization. The strong data overlap between phases plus the early-filtering property (Gur-Ari et al.'s "gradients live in a tiny fixed subspace after an early stage") explain why.

**The test.** If lost noise is the cause, *amplifying* the remaining noise should help. Raising the tuning-stage learning rate by 10× (which scales the noise) *substantially reduces the gap* (Fig. 7). Residual gap is attributed to a high constant LR forgoing the refinement phase. This is consistent with — and mechanistically distinct from — Ash & Adams' *gradient-imbalance* diagnosis; Berariu frames it as a *noise/exploration* diagnosis at the level of minimum flatness.

**Supporting evidence for two phases (Appendix B).** Achille et al. (critical learning periods): early memorization phase (information absorbed) then reorganization (pruning, redistribution); if data statistics change after the initial phase, the net stays trapped in the valley the memorization phase chose. Golatkar et al.: regularization (weight decay, augmentation) matters *only* early. Gur-Ari et al.: gradients confined to a small constant subspace after an early stage. Li et al. / Jastrzebski et al.: high learning-rate-to-batch-size ratio lands the net in flatter, better-generalizing minima.

**Project relevance.** This paper is the primer's designated "plasticity as its own quantity" reference (primer §2 Phase 2). Its warning that *smooth, slow* distribution shift alone reproduces the gap is the tightest analogy to the project's curriculum setup. Its layer-reset result prefigures the reset family; its LR-noise mechanism is a candidate lever distinct from resets. Crucially, its two-phenomena disambiguation note tells the project to be careful *which* plasticity loss any given citation addresses.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Continual learning desiderata (no forgetting, structural transfer, backward transfer, enduring capacity to acquire). Focuses on *plasticity* = ability to keep learning. Three "can't learn" nuances: (i) can't minimize train loss (PackNet frozen neurons, over-constrained EWC); (ii) less data-efficient (negative forward transfer, still reaches 0 train error); (iii) **can reach 0 train error but converges to a poorer-generalizing minimum** — *this* is the paper's focus. Defines generalization gap vs. intransigence. Oct-2023 footnote: two distinct "plasticity loss" phenomena (generalization vs. optimization; e.g. Dohare); treat separately.
- **§2 Generalisation Gap — Experiments.** Reproduces Ash & Adams (Fig. 1). Optimizer-invariance (Adam/RMSprop/SGD/mSGD) ⇒ minimum-quality problem not trajectory problem. Few pretraining epochs suffice. Smooth transition $p=1-\gamma^{50n/N}$ still induces gap (Fig. 3) — RL implication. Multi-stage aggravation + class imbalance (Fig. 4). Width/depth don't fix it (Fig. 5). Top-layer reset needed to recover; no speed benefit from keeping early layers (Fig. 6).
- **§3 Two Phases of Learning (hypothesis).** Flat-vs-sharp minima; exploration (noise-driven) then refinement (gradient-flow) phases. Conjecture: pretraining reduces tuning-phase gradient noise ⇒ weaker exploration ⇒ narrower minimum ⇒ gap. Test: 10× tuning LR reduces gap (Fig. 7).
- **§4 Conclusions.** Gap is robust (smooth transition, multi-stage, model size, partial reset). Continual learning may be hurt by fine-tuning compact models; retraining from stored data can beat warm-starting; tracking the gap is a new facet of forward transfer.
- **Appendix A.** Experimental details; other-optimizer figures; smooth-transition sampling; class-imbalance split methodology; depth/width architectures; layer-reset details.
- **Appendix B.** Supporting two-phase evidence (Achille, Golatkar, Gur-Ari, Li, Jastrzebski, Ghorbani).

---
