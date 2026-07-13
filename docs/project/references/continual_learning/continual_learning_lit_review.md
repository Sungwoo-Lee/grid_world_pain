# Continual Learning — Master Literature Review

**Corpus:** 22 papers on continual learning in deep reinforcement learning (RL), spanning two linked failure modes — *catastrophic forgetting* (the network loses old skills when it learns new ones) and *loss of plasticity* (the network gradually loses the very ability to learn anything new). Processed one paper at a time by `literature-reviewer`, then reorganized by theme and cross-connected here by `literature-curator`.

**Curated:** 2026-07-13 · literature-curator

---

## Purpose — what this document is

A neural-network agent that keeps training on a *changing* stream of tasks runs into two distinct troubles. The first, **catastrophic forgetting**, is the classic one: teach the network task B and it silently overwrites what it knew about task A. The second, discovered later and the real center of gravity of this corpus, is **loss of plasticity**: after enough training under change, the network stops being able to learn *at all* — a fresh, randomly-initialized network of the same size would now learn the next task faster than this "experienced" one can. Loss of plasticity is not about forgetting the past; it is about losing the future.

This corpus assembles 22 papers that trace the field from the 1989 discovery of catastrophic forgetting through the 2024 *Nature* capstone showing that standard deep learning, run continually, decays until it is no better than a shallow linear model. It exists to give the project a rigorous, self-contained reference for **why a network carried across a sequence of RL tasks degrades** — the exact situation the project's own difficulty-ladder curriculum agent hit when it underperformed a from-scratch baseline (see the null-result diagnosis series [NMN_PERFORMANCE_DIAGNOSIS](../../../develop/INDEX.md)).

## How to read this

Each of the 22 entries has the same three-part shape:

1. **Phase 1 — Foundational Overview**: a plain-language, undergraduate-level explanation of the paper's problem, finding, and takeaway. Start here.
2. **Phase 2 — Graduate-Level Deep Dive**: the full mathematics — every equation, every derivation — for readers who need the mechanism, not just the story.
3. **Appendix: Section-by-Section Backbone**: a faithful, original-section-order summary of the paper. This is the completeness guard; it is never abridged.

The document is ordered into four phases that mirror how the field actually developed:

- **Phase 1 — Catastrophic forgetting as *the* problem** (1999–2017): the forgetting era and its three escape families — *regularization* (anchor important weights), *replay* (rehearse stored old data), and *parameter isolation* (freeze old weights, grow new).
- **Phase 2 — The learner degrades** (2020–2022): the realization that even with forgetting handled, sequential / bootstrapped / warm-started training quietly damages the learner itself — rank collapse, capacity loss, primacy bias.
- **Phase 3 — Loss of plasticity: mechanisms and fixes** (2023–2024): the maturation into a named subfield, with a mechanistic account (dormant units, rank collapse, loss-landscape sharpening, empirical-NTK degeneracy) and a menu of cures (CReLU, ReDo, plasticity injection, resets, layer-norm + weight-decay, continual backprop).
- **Adjacent threads**: four bridge papers connecting the plasticity/forgetting literature to the project's specific concerns — activation design (CReLU), policy-entropy collapse in PPO, curriculum-learning theory, and carrying a recurrent belief-state across task switches.

Papers are numbered **1–22** sequentially across all phases; the number in each heading matches its entry in the Table of Contents below.

## Relationship to the field-evolution primer

This document has a sibling: the annotated bibliography / field-evolution primer, [[continual_learning_field_evolution]]. **Division of labor:** the primer is the *map and bibliography* — a compact narrative of how the field evolved, with one-line-per-paper placements into families and phases, meant to be read cover-to-cover in one sitting. **This document is the per-paper *deep-dive*** — the full Phase 1 / Phase 2 / backbone treatment of each paper, meant to be consulted one entry at a time. When the primer says "EWC is the canonical regularization method," this document is where you go to see EWC's Bayesian derivation worked line by line. The primer is not duplicated or edited here; per-paper "project relevance" notes are retained as written.

## Verification notes (read before lifting anything verbatim)

- **Dohare et al. 2024 (Nature) equations — reconstructed, not extracted.** The continual-backprop **contribution-utility (Eq. 1)**, **effective-rank (Eq. 2)**, and **stable-rank** formulas in entry 18 were pulled from a two-column PDF whose tokens fragmented on extraction and were **reconstructed from the surrounding prose**. Treat every symbol in that entry's Phase 2 as provisional and route it through a `math-reviewer` pass before copying any equation into project code or docs. The caveat is repeated inline at the head of entry 18.
- **Cui et al. 2025 (entry 20) is a preprint.** arXiv:2505.22617v1; any conference acceptance is not independently confirmed here.
- **D'Oro et al. 2023 (entry 15)** has no arXiv version; the version of record is OpenReview `OpC-9aBBVJe` (ICLR 2023 oral).

## Table of Contents

**Phase 1 — Catastrophic Forgetting as the Problem (1999–2017)**

1. [French (1999) — Catastrophic Forgetting in Connectionist Networks](#1-french-1999--catastrophic-forgetting-in-connectionist-networks)
2. [Rusu et al. (2016) — Progressive Neural Networks](#2-rusu-et-al-2016--progressive-neural-networks)
3. [Kirkpatrick et al. (2017) — Overcoming Catastrophic Forgetting (EWC)](#3-kirkpatrick-et-al-2017--overcoming-catastrophic-forgetting-ewc)
4. [Zenke et al. (2017) — Continual Learning Through Synaptic Intelligence (SI)](#4-zenke-et-al-2017--continual-learning-through-synaptic-intelligence-si)
5. [Rebuffi et al. (2017) — iCaRL: Incremental Classifier and Representation Learning](#5-rebuffi-et-al-2017--icarl-incremental-classifier-and-representation-learning)
6. [Lopez-Paz & Ranzato (2017) — Gradient Episodic Memory (GEM)](#6-lopez-paz--ranzato-2017--gradient-episodic-memory-gem)

**Phase 2 — The Learner Degrades (2020–2022)**

7. [Ash & Adams (2020) — On Warm-Starting Neural Network Training](#7-ash--adams-2020--on-warm-starting-neural-network-training)
8. [Kumar et al. (2021) — Implicit Under-Parameterization Inhibits Data-Efficient Deep RL](#8-kumar-et-al-2021--implicit-under-parameterization-inhibits-data-efficient-deep-rl)
9. [Berariu et al. (2021) — A Study on the Plasticity of Neural Networks](#9-berariu-et-al-2021--a-study-on-the-plasticity-of-neural-networks)
10. [Lyle, Rowland & Dabney (2022) — Understanding and Preventing Capacity Loss in RL](#10-lyle-rowland--dabney-2022--understanding-and-preventing-capacity-loss-in-rl)
11. [Nikishin et al. (2022) — The Primacy Bias in Deep RL](#11-nikishin-et-al-2022--the-primacy-bias-in-deep-rl)

**Phase 3 — Loss of Plasticity: Mechanisms and Fixes (2023–2024)**

12. [Abbas et al. 2023 — Loss of Plasticity in Continual Deep Reinforcement Learning](#12-abbas-et-al-2023--loss-of-plasticity-in-continual-deep-reinforcement-learning)
13. [Sokar et al. 2023 — The Dormant Neuron Phenomenon in Deep RL (ReDo)](#13-sokar-et-al-2023--the-dormant-neuron-phenomenon-in-deep-rl-redo)
14. [Nikishin et al. 2023 — Deep RL with Plasticity Injection](#14-nikishin-et-al-2023--deep-rl-with-plasticity-injection)
15. [D'Oro et al. 2023 — Sample-Efficient RL by Breaking the Replay Ratio Barrier](#15-doro-et-al-2023--sample-efficient-rl-by-breaking-the-replay-ratio-barrier)
16. [Lyle et al. 2023 — Understanding Plasticity in Neural Networks](#16-lyle-et-al-2023--understanding-plasticity-in-neural-networks)
17. [Lyle et al. 2024 — Disentangling the Causes of Plasticity Loss in Neural Networks](#17-lyle-et-al-2024--disentangling-the-causes-of-plasticity-loss-in-neural-networks)
18. [Dohare et al. 2024 — Loss of Plasticity in Deep Continual Learning (Nature)](#18-dohare-et-al-2024--loss-of-plasticity-in-deep-continual-learning-nature)

**Adjacent Threads**

19. [Shang et al. 2016 — Understanding and Improving CNNs via Concatenated ReLU (CReLU)](#19-shang-et-al-2016--understanding-and-improving-cnns-via-concatenated-relu-crelu)
20. [Cui et al. 2025 — The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](#20-cui-et-al-2025--the-entropy-mechanism-of-reinforcement-learning-for-reasoning-language-models)
21. [Narvekar et al. 2020 — Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](#21-narvekar-et-al-2020--curriculum-learning-for-reinforcement-learning-domains-a-framework-and-survey)
22. [Caccia et al. 2022 — Task-Agnostic Continual Reinforcement Learning](#22-caccia-et-al-2022--task-agnostic-continual-reinforcement-learning)

**Synthesis**

- [Cross-Paper Synthesis](#cross-paper-synthesis)

---

# Phase 1 — Catastrophic Forgetting as the Problem (1999–2017)

**The founding problem.** These six papers span the 1989 discovery of catastrophic forgetting through its three classic escape families. French (1) states the stability–plasticity dilemma and diagnoses forgetting as the price of overlapping distributed representations; progressive networks (2) make forgetting structurally impossible by freezing old weights and growing new capacity; EWC (3) and SI (4) softly anchor the weights that mattered for old tasks; iCaRL (5) and GEM (6) rehearse a small store of old data. Together they define the *backward* axis — protecting the past — and set up the later discovery that protecting the past does nothing for the *forward* ability to keep learning.

---


## 1. French (1999) — Catastrophic Forgetting in Connectionist Networks

**PDF:** `docs/project/references/continual_learning/sources/French 1999 - Catastrophic forgetting in connectionist networks.pdf`
**Venue:** *Trends in Cognitive Sciences* 3(4), 128–135. **Type:** review / conceptual survey.
**Primer link:** This is the "conceptual framing" entry in Phase 1 of the primer — the paper that named the stability–plasticity dilemma as the organizing problem.

### Phase 1: Foundational Overview

**Introduction (plain language).** Imagine teaching a network addition facts about the number one (1+1, 1+2, …), and then, once it has mastered them, teaching it facts about the number two. In humans, learning the "twos" would only gradually blur the "ones". In a standard neural network of the late 1980s, the "ones" knowledge is wiped out almost instantly — within a handful of training passes on the twos, the network's accuracy on the ones drops from 100% to 1%. That abrupt, near-total erasure is what McCloskey & Cohen (1989) called **catastrophic interference** (equivalently, **catastrophic forgetting**). French's 1999 review is the field's stock-taking of the first decade of work on this problem: what causes it, how to measure it, and every solution proposed up to that point.

**Key finding (the central diagnosis).** Catastrophic forgetting is not a bug in one learning rule; it is the flip side of the very property that makes distributed networks powerful. A distributed network stores every pattern in a *single shared set of weights*, and *reuses* those weights across all patterns (that overlap is what gives generalization and graceful degradation). But because the representations of different patterns **overlap**, adjusting the weights to fit new patterns necessarily disturbs the weights that encoded old ones. French's organizing claim: **reduce representational overlap and you reduce forgetting** — but you pay for it in generalization. This is the stability–plasticity dilemma stated concretely: a system must be plastic enough to absorb new input, yet stable enough not to have old memories overwritten.

**Initial takeaway.** The paper crystallizes the whole field's problem statement and a taxonomy of escapes that later work (EWC, progressive nets, replay) formalizes. Its three enduring contributions: (1) the diagnosis "forgetting = overlap of distributed representations"; (2) the two measurement conventions still used today — *exact recognition* vs. *savings/relearning*; (3) the argument that the brain likely solves this with **two interacting memory systems** (hippocampus for fast new learning, neocortex for slow consolidation), a proposal that seeds every later "complementary learning systems" and replay method.

### Phase 2: Graduate-Level Deep Dive

French's article is qualitative (no formal equations in the main text), so the "mathematical rigor" here is a careful reconstruction of the *conceptual mechanics* the paper argues, plus the one formal object it does supply — the weight-space picture in Box 1.

**The weight-space account of forgetting (Box 1).** Let a network's weights live in $\mathbb{R}^p$. Learning task A (the "ones") means finding a point $W_{\text{initial}}$ such that the network correctly maps all task-A patterns. Learning task B (the "twos") drives gradient descent to a new point $W_{\text{new}}$ that solves B. Forgetting is *catastrophic* precisely when $W_{\text{new}}$ is a poor solution for A, i.e. when

$$
\mathcal{L}_A(W_{\text{new}}) \gg \mathcal{L}_A(W_{\text{initial}}),
$$

even though $\|W_{\text{new}} - W_{\text{initial}}\|$ may be small. French emphasizes (citing Kolen & Pollack 1990) that weight-space is *not* smooth: it contains "weight cliffs" where a tiny displacement radically changes the function computed. For a 2-2-1 network solving XOR, sweeping just two of the nine weights over a grid produces a fractal-like map of convergence times — adjacent initial conditions can differ enormously. This non-smoothness is *why* a short move $W_{\text{initial}} \to W_{\text{new}}$ can be catastrophic for A. Note the contrast with EWC (§3), which *assumes* enough local smoothness that a quadratic (Laplace) bowl around $W_{\text{initial}}$ captures task A — French's Box 1 is precisely the caveat EWC's approximation trades away.

**The overlap principle.** French's mechanistic claim can be written informally. Let $r_A, r_B \in \mathbb{R}^h$ be the hidden-layer activation vectors evoked by task-A and task-B inputs. Weight updates for B are (by backprop) proportional to outer products involving $r_B$; they perturb exactly the weights that $r_B$ activates. If $r_A$ and $r_B$ have small overlap — quantified by something like $\langle r_A, r_B\rangle / (\|r_A\|\|r_B\|) \approx 0$ — then B's updates land on weights A does not use, and A is preserved. Hence the entire family of early fixes aims to **orthogonalize / sparsify** internal representations:

- **Activation sharpening** (French 1991/92): after the normal forward pass, sharpen the hidden layer — push the most-active hidden unit(s) higher and suppress the rest — producing sparse "semi-distributed" codes. Fewer active units ⇒ smaller $\langle r_A, r_B\rangle$ ⇒ less interference. Cost: sparser codes generalize and discriminate worse (an early statement of the stability–plasticity trade-off as a quantitative dial).
- **Novelty vectors** (Kortge 1990): weight each backprop delta by how *novel* the pattern is (the input–output mismatch of an auto-associator), so error is "blamed" only on the units responsible, reducing collateral interference.
- **Orthogonal input recoding** (bipolar $-1/+1$ instead of $0/1$): easier to make inputs mutually orthogonal, which propagates to less hidden overlap.
- **Localized receptive fields** — ALCOVE (Kruschke 1992): hidden-unit activation is an inverse-exponential function of distance from the input in feature space, $a_j = \exp(-c\, \|x - \mu_j\|)$. Widening the receptive field (large "covering") ⇒ more distributed ⇒ more overlap ⇒ more interference; narrowing it ⇒ localist ⇒ no interference but no generalization. ALCOVE makes the overlap dial explicit and continuous.
- **Modular / resonance architectures** — CALM (Murre 1992) and the ART family (Carpenter & Grossberg): new input is *recognized as new* and routed to uncommitted nodes via top-down connections, structurally separating new from old. French notes ART's argument that the deeper culprit is the *multiplicative* synaptic transfer function itself.

**Two measurement conventions (formalized).** French insists forgetting must be measured two ways because they can disagree:

1. *Exact recognition.* After learning A then B, present each A-input and check whether every output unit is within tolerance $\tau$ (French uses $\tau = 0.5$) of the target: pattern retained iff $\max_o |y_o - t_o| < \tau$.
2. *Savings / relearning* (Hetherington & Seidenberg 1989, after Ebbinghaus): measure the number of epochs $n_{\text{re}}$ to re-reach criterion on A after B. A network can score 0% on exact recognition yet relearn A in a couple of epochs — "shallow" forgetting where the trace is buried, not destroyed. Modern continual-learning papers report accuracy (≈ exact recognition); French's point that *savings* can reveal retained structure prefigures the backward-transfer diagnostics in GEM (§6).

**Rehearsal and pseudopatterns (Box 2) — the replay ancestor.** The most durable idea in the review, and the direct ancestor of every replay method in this shard (iCaRL, GEM). *Rehearsal*: interleave some stored old patterns with the new ones so the joint loss $\mathcal{L}_A + \mathcal{L}_B$ is minimized rather than $\mathcal{L}_B$ alone. But what if the old patterns are gone? Robins' (1995) **pseudopattern** trick: after learning function $f$, feed random inputs $\hat{I}_i$ through the *current* network to read off outputs $\hat{O}_i = f(\hat{I}_i)$; the pairs $\psi_i = (\hat{I}_i, \hat{O}_i)$ are "pseudo-examples" that approximate $f$. Interleaving the $\{\psi_i\}$ with new data rehearses the old function *without storing any real old data* — a self-generated, generative replay. This is exactly the modern "generative replay" idea (and conceptually the teacher signal iCaRL later replaces with stored exemplars + distillation).

**Dual-network / complementary learning systems (the brain's answer).** French's synthesis, following McClelland, McNaughton & O'Reilly (1995): the mammalian brain sidesteps the dilemma with *two* systems — a **hippocampus** for rapid, pattern-separated acquisition of new episodes, and a **neocortex** for slow interleaved consolidation of shared structure. The hippocampus later "teaches" the neocortex by replaying (biological pseudopatterns, perhaps during sleep/REM). French's own *pseudo-recurrent network* (1997) and Ans & Rousset's coupled reverberating networks (1997) implement this: an early-processing area and a long-term store trade information via pseudopatterns, yielding *gradual* (human-like) rather than catastrophic forgetting, and reproducing the human list-length effect and absence of a list-strength effect.

**Two cautions the paper raises.** (a) *Box 3 — animals.* Rats that learn a 40-s then an 8-s interval *sequentially* show no savings on the return to 40 s (catastrophic), but rats that learn both *concurrently* switch back rapidly (spared). Sequential vs. concurrent training changes the representation — the animal analogue of interleaving. (b) *Box 4 — catastrophic remembering.* The dual of forgetting: an auto-associator trained on so many patterns that it approximates the identity function will "recognize" *any* input as familiar, losing the ability to discriminate seen from unseen. Over-generalization destroys episodic discrimination — a reminder that pushing stability too far has its own failure mode.

### Appendix: Section-by-Section Backbone

- **Abstract / Introduction.** Natural cognitive systems forget gradually, not catastrophically; distributed connectionist nets do the opposite. The features enabling generalization and graceful degradation (shared, overlapping weights) are the *root cause* of forgetting. Catastrophic interference is a radical form of the general stability–plasticity problem (Grossberg). Scope: networks with a single set of shared multiplicative weights.
- **Catastrophic vs. gradual interference.** Barnes & Underwood (1959) A-B / A-C human paradigm shows gradual retroactive interference. McCloskey & Cohen (1989): backprop net learning "ones" then "twos" addition drops from 100%→1% on ones within ~5 trials. Ratcliff (1990) replicates across sizes/architectures. Opens the theoretical questions (inherent to all distributed nets? avoidable? why don't humans show it? can lower animals?).
- **Measuring interference.** Exact-recognition (within 0.5 of target) vs. savings/relearning (Ebbinghaus, Hetherington & Seidenberg). A net can look fully forgotten by recognition yet relearn fast — but not all forgetting is this "shallow" kind. Modern practice reports both.
- **Early solutions (~1990).** Kortge's novelty vectors; French's semi-distributed representations & activation sharpening (sparsify hidden codes to cut overlap). Brousse & Smolensky / McRae & Hetherington: in combinatorially structured domains, pretraining on random samples makes interference vanish (regularities captured; new items resemble old). Trade-off: sparse coding hurts generalization/discrimination.
- **Reducing representational overlap.** Orthogonal input recoding (bipolar coding); hidden-layer orthogonalization; emergent semi-distributed / localist codes. CALM (modular competing R/V/A nodes) and ART (resonance, top-down routing of novel input); ALCOVE (distance-based localized receptive fields; overlap tunable via field width). Claim (Carpenter): multiplicative path-weights are the underlying culprit.
- **Distributed-but-stable models.** Convolution-correlation memories (CHARM, TODAM) and Sparse Distributed Memory (Kanerva) forget gradually; SDM ≅ Hopfield, has a hard storage limit but is semi-distributed below it; bimodal zero-mean coding gives near-orthogonality.
- **Rehearsal.** Interleave old items with new (transforms catastrophic → gradual). Robins (1995) pseudopatterns for rehearsal when originals are unavailable — self-generated replay.
- **Separating new from old.** McClelland/McNaughton/O'Reilly hippocampus–neocortex proposal. Jumpnet (control net modulates processing net). French's pseudo-recurrent net (1997) & Ans–Rousset coupled reverberating nets (1997): dual areas exchanging pseudopatterns; reproduce list-length effect and no list-strength effect; representational compression can model category-specific amnesia.
- **Other techniques.** Chappell & Humphreys auto-associator + sparse codes; Hinton & Plaut fast/slow weights; dual-weight architectures; cascade-correlation.
- **Conclusion / Outstanding questions.** Best current answer: two permanently interacting processing areas. Open: is representational separation necessary? are dual nets required? role of episodic memory? neural reality of pseudopatterns (REM sleep)? which animals forget catastrophically?
- **Boxes.** Box 1 weight-space "cliffs" (Kolen & Pollack). Box 2 pseudopatterns. Box 3 catastrophic forgetting of temporal intervals in rats (sequential vs concurrent). Box 4 catastrophic *remembering* (over-generalization → loss of discrimination).

---

## 2. Rusu et al. (2016) — Progressive Neural Networks

**PDF:** `docs/project/references/continual_learning/sources/Rusu et al. 2016 - Progressive Neural Networks.pdf`
**Venue:** arXiv:1606.04671 (DeepMind). **Type:** architecture / empirical (deep RL).
**Primer link:** the canonical **parameter-isolation** family member in Phase 1 — forgetting made *structurally impossible* by freezing old weights and growing new capacity.

### Phase 1: Foundational Overview

**Introduction (plain language).** The standard way to reuse a trained network on a new task is *finetuning*: copy the weights, swap the output layer, keep training. But finetuning is *destructive* — the new task overwrites the old function, so you cannot go back to the first task, and you must guess which prior model to start from. Progressive Neural Networks refuse to overwrite anything. For each new task they **freeze** the entire previous network (a "column") and **add a brand-new column** trained from scratch, wiring in **lateral connections** so the new column can *read* the frozen features of every previous column. Because old weights never change, the old tasks are remembered perfectly — forgetting is impossible *by construction* — while the new column still benefits from prior knowledge through those lateral links.

**Key finding.** On deep-RL benchmarks — synthetic Pong variants ("Pong Soup"), random sequences of Atari games, and 3D maze ("Labyrinth") foraging — progressive nets beat the standard transfer baselines (finetuning) on both mean and median transfer score, *and* avoid finetuning's failure mode of *negative* transfer on incompatible tasks (they can simply ignore useless prior features). A companion **Average Fisher Sensitivity (AFS)** analysis shows transfer really does route through the lateral connections, and reveals *where*: low-level vision often transfers, task-specific control layers get relearned.

**Initial takeaway.** Progressive nets are a clean existence proof that transfer *and* zero-forgetting can coexist in deep RL. The catch is scaling: parameters grow with the number of tasks (linear in width, quadratic in parameters), and at inference you must know the task label to pick the column. But the AFS analysis shows each added column uses only a *fraction* of its capacity, pointing to pruning/compression as the fix.

### Phase 2: Graduate-Level Deep Dive

**The core recurrence.** A progressive net starts as a single column: an $L$-layer network with hidden activations $h_i^{(1)} \in \mathbb{R}^{n_i}$ and parameters $\Theta^{(1)}$, trained to convergence on task 1. For task 2, $\Theta^{(1)}$ is **frozen** and a fresh column $\Theta^{(2)}$ is added whose layer $i$ receives input from *both* its own previous layer $h_{i-1}^{(2)}$ *and* the frozen column's $h_{i-1}^{(1)}$. Generalized to $K$ columns, layer $i$ of column $k$ computes:

$$
h_i^{(k)} = f\!\left( W_i^{(k)} h_{i-1}^{(k)} + \sum_{j<k} U_i^{(k:j)}\, h_{i-1}^{(j)} \right),
$$

where $W_i^{(k)} \in \mathbb{R}^{n_i \times n_{i-1}}$ is column $k$'s own weight matrix at layer $i$, $U_i^{(k:j)} \in \mathbb{R}^{n_i \times n_j}$ are the **lateral connections** from layer $i-1$ of an earlier column $j<k$ into layer $i$ of column $k$, $f(x) = \max(0,x)$ is the ReLU nonlinearity, and $h_0$ is the network input.

**Why forgetting is structurally impossible.** Two facts jointly guarantee it. (1) Lateral connections run *only* from earlier columns into later ones ($j < k$): in the forward pass, later columns never feed back into earlier ones, so an earlier column's output is unchanged by anything added later. (2) When training column $k$, all $\{\Theta^{(j)} : j<k\}$ are **constants** for the optimizer (frozen). Therefore $\partial \mathcal{L}^{(k)} / \partial \Theta^{(j)} = 0$ for $j<k$ — no gradient ever touches an old column. The function computed for task $j$ is literally invariant. This is the sharpest possible contrast with EWC (§3) and SI (§4), which only *softly* penalize movement of old-task weights; progressive nets set the penalty to $\infty$ (freeze) and add capacity instead.

**Adapters (the practical lateral connection).** A raw linear lateral connection has two problems: the anterior features $h_{i-1}^{(<k)} = [h_{i-1}^{(1)} \cdots h_{i-1}^{(k-1)}]$ (dimension $n_{i-1}^{(<k)}$) grow with $k$, and their scales differ across columns. The **adapter** replaces the linear lateral term with a scaled single-hidden-layer MLP that performs dimensionality reduction:

$$
h_i^{(k)} = \sigma\!\left( W_i^{(k)} h_{i-1}^{(k)} + U_i^{(k:j)}\, \sigma\!\big( V_i^{(k:j)}\, \alpha_{i-1}^{(<k)}\, h_{i-1}^{(<k)} \big) \right),
$$

where $\alpha_{i-1}^{(<k)}$ is a learned scalar (initialized small and random) that rescales the anterior features so different columns' magnitudes are comparable, and $V_i^{(k:j)} \in \mathbb{R}^{n_{i-1} \times n_{i-1}^{(<k)}}$ projects the concatenated anterior features down onto an $n_{i-1}$-dimensional subspace *before* the lateral weight $U$ is applied. The projection keeps the parameter count of the lateral pathway on the same order as $|\Theta^{(1)}|$ as $k$ grows. For convolutional layers the dimensionality reduction is a $1\times1$ convolution.

**RL specialization.** Each column solves one MDP; column $k$ defines a policy $\pi^{(k)}(a\mid s) := h_L^{(k)}(s)$, the softmax output layer. Training uses A3C (asynchronous advantage actor-critic) with 16 workers; scoring uses **area under the learning curve** (not final score), and the **transfer score** is a column's AUC relative to a single-column-from-scratch baseline (baseline 1).

**Transfer analysis — Average Fisher Sensitivity (AFS).** To measure *where* the policy relies on each column/feature, they compute a diagonal Fisher of the policy $\pi$ with respect to the *normalized activations* $\hat{h}_i^{(k)}$ (not the parameters — this is the paper's non-standard twist, making $\hat{F}$ comparable across layers/columns):

$$
\hat{F}_i^{(k)} = \mathbb{E}_{\rho(s,a)}\!\left[ \frac{\partial \log \pi}{\partial \hat{h}_i^{(k)}} \left(\frac{\partial \log \pi}{\partial \hat{h}_i^{(k)}}\right)^{\!\top} \right],
$$

with the expectation over the state–action distribution $\rho(s,a)$ induced by the trained network. The per-feature AFS is the normalized diagonal element,

$$
\mathrm{AFS}(i,k,m) = \frac{\hat{F}_i^{(k)}(m,m)}{\sum_{k} \hat{F}_i^{(k)}(m,m)},
$$

and the per-layer-per-column score sums over features $m$: $\mathrm{AFS}(i,k) = \sum_m \mathrm{AFS}(i,k,m)$. Intuitively $\hat{F}$ is a *local approximation to a perturbation sensitivity* — how much the policy output changes if you jiggle a feature.

**The perturbation cross-check (APS).** The appendix corroborates AFS with a slower but more intuitive **Average Perturbation Sensitivity**: inject Gaussian noise into a layer's activations (variance scaled to the activation variance, to be scale-invariant), find the noise precision $\Lambda_i^{(k)} = 1/\sigma_i^{2(k)}$ that causes a 50% score drop, and normalize across columns:

$$
\mathrm{APS}(i,k) = \frac{\Lambda_i^{(k)}}{\sum_k \Lambda_i^{(k)}}.
$$

AFS and APS agree closely, validating the fast Fisher-based measure.

**Empirical results (Table 1, transfer % vs. single-column baseline = 100).**

| Method | Pong mean/median | Atari mean/median | Labyrinth mean/median |
|---|---|---|---|
| Baseline 2 (finetune output only) | 35 / 7 | 41 / 21 | 88 / 85 |
| Baseline 3 (full finetune) | 181 / 160 | 133 / 110 | 235 / 112 |
| Baseline 4 (2-col, random+frozen) | 134 / 131 | 96 / 95 | 185 / 108 |
| Progressive 2-col | 209 / 169 | 132 / 112 | 491 / 115 |
| Progressive 3-col | 222 / 183 | 140 / 111 | — |
| Progressive 4-col | — | 141 / 116 | — |

Progressive nets beat full finetuning (baseline 3), positive transfer in 8/12 Atari targets (vs. 5/12 for baseline 3), and only 2 cases of negative transfer. Baseline 2's failure (can't relearn low-level vision) shows why finetuning-only-the-head breaks in RL where the visual statistics change per task.

**The "sweet spot" finding.** Counter-intuitively, the *most positive* transfer does not come from maximal reliance on source features. AFS across 72 three-column Atari nets shows a sweet spot: transfer is best when source features are *augmented* by some new mid-level vision in the new column, and *most negative* when the net leans entirely on frozen convolutional features and learns no new vision. Two hypotheses: (a) source features give fast convergence to a poor local optimum (an inductive-bias trap), or (b) an exploration failure where a "good-enough" transferred representation yields a functional but sub-optimal policy.

**Limitations (stated).** Parameters grow linearly in hidden units / quadratically in parameters with $K$; inference needs the task label to select the column; only a fraction of each new column's capacity is used (AFS spectra get sparser with more columns), so pruning/online-compression/distillation is the natural mitigation.

*Relevance note for our project.* Progressive nets are the "no compromise" pole of the stability–plasticity dilemma — perfect stability at unbounded parameter cost — and the Labyrinth foraging setup is a near-neighbour of this project's grid-world foraging environment. The AFS/APS machinery (Fisher-of-activations, perturbation sensitivity) is a reusable diagnostic for asking *which features a policy relies on* — orthogonal to the loss-of-plasticity metrics the primer's Phase 2 introduces.

### Appendix: Section-by-Section Backbone

- **Abstract.** Progressive nets: immune to forgetting, transfer via lateral connections to previously-learned features; evaluated on Atari + 3D maze RL; beat pretrain/finetune baselines; novel Fisher-based sensitivity measure shows transfer at both sensory and control layers.
- **§1 Introduction.** Finetuning is destructive and needs foreknowledge of which model to init from; distillation needs persistent data for all tasks. Progressive nets keep a pool of pretrained columns and learn lateral connections into them — prior knowledge is non-transient, compositional, immune to forgetting. Three contributions: novel combination for task sequences, extensive deep-RL evaluation, Fisher+perturbation transfer analysis.
- **§2 Progressive Networks.** Column definition; freeze old $\Theta^{(j)}$, add random-init column per task; Eq. (1) recurrence with $W$ (own) and $U$ (lateral) weights, ReLU. Design goals: solve K tasks, accelerate via transfer, avoid forgetting. No assumption of task overlap (may be orthogonal/adversarial). Forgetting impossibility argument (lateral only $j<k$; frozen params). RL application: column = policy for one MDP. Adapters Eq. (2): scaled MLP lateral connection with projection matrix $V$ and learned scalar $\alpha$; $1\times1$ conv for conv layers. Limitations: parameter growth; task label needed at inference.
- **§3 Transfer Analysis.** APS (perturbation, slow) and AFS (Fisher, fast). AFS: diagonal Fisher of policy w.r.t. normalized activations; per-feature and per-layer normalized scores.
- **§4 Related Literature.** Transfer/multitask RL (actor-mimic, policy distillation); constructive architectures (cascade-correlation, incremental autoencoders); multi-column nets. Progressive uses lateral connections for deep compositionality.
- **§5 Experiments.** Setup: A3C, 16 workers, top-3-of-25 jobs, AUC scoring, transfer score vs. baseline 1. Baselines 1–4 (Fig. 3). §5.2 Pong Soup (Noisy/Black/White/Zoom/flips): baseline 2 negative transfer, baseline 3 strong, progressive beats it; AFS shows conv reuse on H-flip, new mid-vision on Zoom. §5.3 Atari (Pong/River Raid/Seaquest → 12 targets): 2/3/4 columns; positive transfer 8/12; sweet-spot AFS finding. §5.4 Labyrinth 3D foraging (apples/strawberries + / mushrooms/lemons −): progressive best; baseline 2 negative even on easy levels (can't relearn changing reward-item vision).
- **§6 Conclusion.** First demonstration of positive transfer in deep-RL continual learning; robust to harmful features; transfer grows with columns; constructive not destructive.
- **Supplement.** A: Perturbation analysis details, $\Lambda = 1/\sigma^2$ at 50% drop, APS Eq. (3); APS≈AFS. B: Compressibility — AFS spectra sparsen with more columns; new columns' features less important; pruning feasible. C: Setup details (hyperparameter grid; 3 conv layers, 12 maps each; 256 FC units; RMSProp; $1.6\times10^8$ env steps / $4\times10^7$ agent steps with action-repeat 4). D: per-game learning curves. E: Labyrinth level descriptions.

---

## 3. Kirkpatrick et al. (2017) — Overcoming Catastrophic Forgetting (EWC)

**PDF:** `docs/project/references/continual_learning/sources/Kirkpatrick et al. 2017 - Overcoming Catastrophic Forgetting (EWC).pdf`
**Venue:** *PNAS* 114(13), 3521–3526 (DeepMind). **Type:** algorithm / empirical (supervised + RL).
**Primer link:** the canonical **regularization** family member in Phase 1 — the diagonal-Laplace / Fisher weight-anchoring method the whole family is named after.

### Phase 1: Foundational Overview

**Introduction (plain language).** Progressive nets (§2) avoid forgetting by never touching old weights and growing new ones — but that costs ever-more parameters. Elastic Weight Consolidation (EWC) keeps a *single, fixed-size* network and instead asks: which weights *mattered* for the old task? It then makes those specific weights *stiff* — reluctant to move — while leaving the rest free to learn the new task. The mental image is a spring ("elastic") anchoring each important weight to its old value, with a stiffness proportional to how important that weight was. Unimportant weights feel no spring and adapt freely. This is directly inspired by neuroscience: when a mouse learns a skill, certain dendritic spines enlarge and *persist* through later learning; erase them and the skill is lost. EWC is the artificial analogue of that *task-specific synaptic consolidation*.

**Key finding.** EWC lets one fixed-capacity network learn a long sequence of tasks with only modest error growth, where plain SGD forgets catastrophically and plain L2 regularization fails the opposite way (it protects *all* weights equally, so it can't learn the new task). Demonstrated on (a) **permuted MNIST** (each task = MNIST with a fixed random pixel permutation) with many sequential tasks, and (b) **ten sequential Atari 2600 games** with a DQN agent, where EWC agents learn to play multiple games while plain-SGD agents never exceed one.

**Initial takeaway.** EWC is the reference "regularization" method: cheap (linear in parameters and data), grounded in a Bayesian/Laplace argument, and biologically motivated. Its known weakness — which the paper is honest about — is that it *underestimates* parameter uncertainty (the diagonal-Fisher point estimate is over-confident), so it doesn't reach the score of ten independently trained networks.

### Phase 2: Graduate-Level Deep Dive

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

### Appendix: Section-by-Section Backbone

- **Abstract.** Selectively slow learning on weights important for old tasks; scalable; demonstrated on permuted-MNIST classification and sequential Atari.
- **§1 Introduction.** Continual learning = learn consecutive tasks without forgetting. Interleaved multitask training avoids forgetting but is impractical (memory ∝ #tasks). Biological motivation: dendritic-spine enlargement persists across learning (Yang, Hayashi-Takagi, Cichon & Gan); erasing spines erases skill ⇒ task-specific synaptic consolidation.
- **§2 Elastic Weight Consolidation.** Quadratic-penalty spring intuition; stiffness varies per weight. Bayesian derivation Eqs. (1)–(2); all task-A info in $p(\theta\mid D_A)$; Laplace/Fisher-diagonal approximation (MacKay); Fisher's three properties; EWC loss Eq. (3); extension to 3+ tasks via summed quadratics.
- **§2.1 Supervised (permuted MNIST).** SGD forgets, L2 over-protects, EWC succeeds (Fig. 2A). Scales past dropout-SGD (Fig. 2B). Fisher-overlap by depth shows representation sharing (Fig. 2C).
- **§2.2 RL (Atari).** DQN + EWC on 10 interleaved games. Fixed capacity vs. progressive-net capacity growth. Task-recognition via HMM with spawnable generative models (FMN process). Per-task replay buffers; per-layer task-specific gains/biases. EWC after 20M frames, $\lambda=400$. EWC learns multiple games; SGD stays <1 (Fig. 3B). True-label control only modest gain. Fig. 3C perturbation test reveals under-estimated uncertainty (nullspace perturbations hurt).
- **§3 Discussion.** EWC grounded in Bayesian learning (prior = previous posterior). vs. French & Chater (slow), ELLA (matrix inversion). Linear-time cost via factorized-Gaussian + diagonal-Fisher point estimate; point estimate is the key weakness. Parallels to synaptic-uncertainty theories (Aitchison & Latham): synapse stores weight + variance + mean.
- **Appendix.** 4.1 MNIST settings (FC ReLU nets, dropout 0.2/0.5, early stopping). 4.2 Atari details (image preprocessing, 3 conv + FC, Double-Q, RMSProp, $\lambda=400$, EWC start 20M frames; HMM task model $p(c,t{+}1)=\sum_{c'}p(c',t)\Gamma(c,c')$ with switch rate $\alpha=1/t$; Dirichlet-multinomial generative models, spawn on hold-out selection). 4.3 Fisher overlap = $1-$ Fréchet distance between normalized Fishers.

---

## 4. Zenke et al. (2017) — Continual Learning Through Synaptic Intelligence (SI)

**PDF:** `docs/project/references/continual_learning/sources/Zenke et al. 2017 - Synaptic Intelligence.pdf`
**Venue:** ICML 2017 (PMLR 70; Stanford — Zenke, Poole, Ganguli). **Type:** algorithm + theory (supervised).
**Primer link:** the **regularization** family, "EWC's online cousin" — importance accumulated *along the whole trajectory* instead of at task end.

### Phase 1: Foundational Overview

**Introduction (plain language).** SI shares EWC's goal — protect the weights that mattered for old tasks — but changes *how importance is measured*. EWC waits until a task is finished, then does a separate pass to compute the Fisher information at the final weights. SI instead watches each weight *while* it learns and keeps a running tally of how much that weight has been *pulling the loss down* over the entire training trajectory. A weight that repeatedly contributed to reducing the loss is deemed important and gets consolidated (made stiff) when the task ends. The paper frames this as giving each synapse a richer internal state — "intelligent synapses" that are 3-dimensional (they track current value, old value, and accumulated importance) rather than a single scalar — echoing the biological fact that real synapses carry complex molecular machinery, not one number.

**Key finding.** SI matches EWC's forgetting-resistance on **split MNIST**, **permuted MNIST**, and **split CIFAR-10/100**, but computes importance **online and locally** (no separate end-of-task Fisher pass, no summing over output classes — so it scales to high-dimensional outputs where EWC's exact Fisher is expensive). A bonus empirical result: on split CIFAR, SI-consolidated networks sometimes *generalize better on new tasks* than networks trained from scratch — consolidation acts as a helpful regularizer against overfitting.

**Initial takeaway.** SI is the "trajectory-integrated" regularizer. Same penalty *shape* as EWC (a per-weight-weighted quadratic anchoring to old values), different *importance estimator* — cheap, streaming, biologically flavoured. The theory section proves that in a tractable quadratic case the SI importance reduces to the Hessian, giving it the same curvature meaning EWC's Fisher has.

### Phase 2: Graduate-Level Deep Dive

**Setup and notation.** Training traces a trajectory $\theta(t)$ in parameter space. For an infinitesimal update $\delta(t)$, the change in loss is, to first order,

$$
\mathcal{L}(\theta(t) + \delta(t)) - \mathcal{L}(\theta(t)) \approx \sum_k g_k(t)\, \delta_k(t), \qquad g_k = \frac{\partial \mathcal{L}}{\partial \theta_k}. \tag{1}
$$

So each parameter's change $\delta_k = \theta_k'(t)$ contributes $g_k(t)\,\delta_k(t)$ to the total loss change. Summing (integrating) over the whole trajectory gives a **path integral of the gradient field**:

$$
\int_{\mathcal{C}} g(\theta(t))\, d\theta = \int_{t_0}^{t_1} g(\theta(t)) \cdot \theta'(t)\, dt. \tag{2}
$$

Because the gradient is a *conservative* field, this integral equals the *net* loss change $\mathcal{L}(\theta(t_1)) - \mathcal{L}(\theta(t_0))$ regardless of path. The key move is to **decompose it per-parameter**, defining the per-weight importance $\omega_k^\mu$ for task $\mu$:

$$
\int_{t_{\mu-1}}^{t_\mu} g(\theta(t)) \cdot \theta'(t)\, dt = \sum_k \int_{t_{\mu-1}}^{t_\mu} g_k(\theta(t))\, \theta_k'(t)\, dt \;\equiv\; -\sum_k \omega_k^\mu. \tag{3}
$$

The minus sign is a convention (we care about *decreasing* the loss, so a weight that reduces loss earns positive $\omega_k^\mu$). **$\omega_k^\mu$ is the total amount by which parameter $k$ drove the loss down over task $\mu$'s training.** In practice it's a cheap running sum: at each SGD step accumulate the product of the gradient $g_k$ and the actual update $\theta_k'$. (Because SGD is noisy, this over-estimates the true $\omega_k^\mu$ — a known bias, corrected empirically by the strength parameter $c<1$ below.)

**The surrogate loss and consolidation penalty.** The problem: we want to minimize $\mathcal{L} = \sum_\mu \mathcal{L}^\mu$ over all tasks but only ever see one $\mathcal{L}^\mu$ at a time. Catastrophic forgetting is when minimizing the current $\mathcal{L}^\mu$ inadvertently raises past losses $\mathcal{L}^\nu$ ($\nu<\mu$). SI replaces the inaccessible past losses with a **quadratic surrogate** anchored at the end-of-previous-task weights $\tilde{\theta}_k = \theta_k(t_{\mu-1})$. The modified objective is

$$
\tilde{\mathcal{L}}^\mu = \mathcal{L}^\mu + c \sum_k \Omega_k^\mu \big(\tilde{\theta}_k - \theta_k\big)^2, \tag{4}
$$

where $c$ is a dimensionless strength trading old vs. new memories, and the per-parameter regularization strength is

$$
\Omega_k^\mu = \sum_{\nu < \mu} \frac{\omega_k^\nu}{(\Delta_k^\nu)^2 + \xi}, \qquad \Delta_k^\nu \equiv \theta_k(t_\nu) - \theta_k(t_{\nu-1}). \tag{5}
$$

Here $\Delta_k^\nu$ is how far weight $k$ *moved* during task $\nu$, and $\xi$ is a small damping constant preventing blow-up when $\Delta_k^\nu \to 0$. **Read Eq. (5) carefully:** importance = (loss reduction the weight achieved) divided by (distance-squared it travelled). The denominator serves two roles: (i) it makes the term carry the same units as the loss (so the penalty is dimensionally a loss), and (ii) it normalizes out how far the weight happened to move, isolating *efficiency* of loss reduction. Note $\tilde{\mathcal{L}}^\mu$ has the **exact same form as EWC's Eq. (3)** — a per-weight-weighted quadratic pulling $\theta_k$ back to a reference value — only the weighting $\Omega_k^\mu$ is trajectory-derived rather than Fisher-derived. Bookkeeping: $\omega_k$ accrues continuously during training; $\Omega_k^\mu$ and the references $\tilde{\theta}$ update only at task boundaries; $\omega_k$ resets to zero after each consolidation.

**Interpretation of the surrogate (Fig. 2).** The quadratic surrogate is *not* the Hessian-at-the-minimum quadratic. It is chosen to match three properties of the actual descent on the old task: the total loss drop $\mathcal{L}(\theta(0)) - \mathcal{L}(\theta(T))$, the net parameter motion $\theta(0)-\theta(T)$, and having its minimum at the endpoint $\theta(T)$. Those three conditions uniquely fix the surrogate quadratic that "summarizes" the whole descent trajectory.

**Theory — the SI importance recovers the Hessian.** The paper's analytic core proves that in a clean case the path-integral importance $Q$ (the matrix whose diagonal is $\omega$) equals the Hessian, giving SI the same curvature-based meaning EWC's Fisher has. Consider a quadratic error

$$
E(\theta) = \tfrac{1}{2}(\theta - \theta^*)^\top H (\theta - \theta^*), \tag{6}
$$

with minimum $\theta^*$ and Hessian $H$. Continuous-time gradient descent obeys

$$
\tau \frac{d\theta}{dt} = -\frac{\partial E}{\partial\theta} = -H(\theta - \theta^*), \tag{7}
$$

whose exact solution from initial $\theta(0)$ is

$$
\theta(t) = \theta^* + e^{-H t/\tau}\big(\theta(0) - \theta^*\big), \tag{8}
$$

with update velocity

$$
\theta'(t) = \frac{d\theta}{dt} = -\frac{1}{\tau} H\, e^{-H t/\tau}\big(\theta(0) - \theta^*\big). \tag{9}
$$

Since $g = \tau\, d\theta/dt$, the importances (Eq. 3) are the diagonal of the time-integrated outer product of the velocity:

$$
Q = \tau \int_0^\infty dt\; \frac{d\theta}{dt}\,\frac{d\theta}{dt}^{\!\top}. \tag{10}
$$

Diagonalize $H$ with eigenpairs $(\lambda_\alpha, u_\alpha)$ and let $d_\alpha = u_\alpha \cdot (\theta(0)-\theta^*)$ be the projection of the total displacement onto eigenmode $\alpha$. Substituting (9) into (10), changing to the eigenbasis, and doing the Gaussian time integral $\int_0^\infty e^{-(\lambda_\alpha + \lambda_\beta)t/\tau}dt = \tau/(\lambda_\alpha+\lambda_\beta)$ yields

$$
Q_{ij} = \sum_{\alpha\beta} u_i^\alpha\, d_\alpha\, \frac{\lambda_\alpha \lambda_\beta}{\lambda_\alpha + \lambda_\beta}\, d_\beta\, u_j^\beta. \tag{11}
$$

Note $Q$ no longer depends on the descent speed $\tau$ (it is a steady-state, time-integrated quantity).

**Three cases where $Q$ reduces to $H$.**
1. *Averaged over random initial conditions.* If the displacements $d_\alpha$ are zero-mean iid with variance $\sigma^2$, then $\langle d_\alpha d_\beta\rangle = \sigma^2 \delta_{\alpha\beta}$, and the double sum in (11) collapses (using $\tfrac{\lambda_\alpha^2}{2\lambda_\alpha} = \tfrac{\lambda_\alpha}{2}$):

$$
\langle Q_{ij}\rangle = \tfrac{1}{2}\sigma^2 \sum_\alpha u_i^\alpha \lambda_\alpha u_j^\alpha = \tfrac{1}{2}\sigma^2 H_{ij}. \tag{12}
$$

So the correlation of parameter updates, integrated over time, *is the Hessian* up to the scale factor $\sigma^2$ — and the $(\Delta_k^\nu)^2$ denominator in Eq. (5) (which averages to $\sigma^2$ at zero damping) exactly removes that scale factor. This is the theoretical justification for the normalization in Eq. (5).

2. *Diagonal Hessian.* If $H$ is diagonal, $u_i^\alpha = \delta_{\alpha i}e_i$, so eigenvalues are the diagonal entries $\lambda_i = H_{ii}$ and (11) reduces to $Q_{ij} = \delta_{ij}(d_i)^2 H_{ii}$. Normalizing by $(d_i)^2$ recovers the diagonal Hessian.

3. *Rank-1 Hessian.* If only $\lambda_1 \neq 0$, then $Q_{ij} = \tfrac{1}{2}(d_1)^2 u_i^1 \lambda_1 u_j^1 = \tfrac{1}{2}(d_1)^2 H_{ij}$. The paper flags this as the *interesting* case for continual learning: a low-rank error leaves many weight-space directions unconstrained by the current task, i.e. spare capacity for future tasks — the geometric reason weight-anchoring can work without freezing.

**Caveat the theory states.** These exact correspondences hold because for a quadratic loss $H$ is constant along the trajectory. For general losses $H$ varies along the path, so no exact SI↔Hessian↔endpoint-Fisher identity holds; but empirically SI's importance *correlates* with endpoint measures (Fisher, etc.), which the authors offer as the explanation for why SI and EWC perform comparably despite computing importance so differently.

**Experiments.** *Split MNIST* (5 tasks, each a binary digit pair, multi-head, MLP 2×256 ReLU, $\xi=10^{-3}$): with consolidation ($c=1$) old-task accuracy stays near 1; without ($c=0$) it drops to chance (Fig. 3). *Permuted MNIST* (MLP 2×2000, $\xi=0.1$, $c=0.1$ via grid search, Adam state retained across tasks): SI (blue) tracks EWC and both stay high over 10 tasks while SGD and SGD+dropout collapse (Fig. 4); importance-correlation matrices (Fig. 5) show consolidation keeps per-task important-weight sets *uncorrelated* (different weights per task), whereas fine-tuning lets second-layer importances become correlated across tasks — the mechanistic signature of forgetting. *Split CIFAR-10/100* (CNN, 6 tasks): consolidation shows no age-dependent accuracy decline; and green (consolidation) ≥ gray (from-scratch) on validation while the reverse holds on *training* accuracy — i.e. consolidation reduces overfitting, generalizing better on new tasks with limited data.

*Relevance note.* SI is the online/streaming member of the regularization family and the cleanest theoretical bridge in this shard between "importance" and "loss curvature" (its Hessian result). For any project idea that wants a *cheap, per-step* estimate of which parameters matter — computed without a separate Fisher pass — SI is the template. The path-integral / conservative-field argument is also a reusable analytic tool.

### Appendix: Section-by-Section Backbone

- **Abstract.** Intelligent synapses accumulate task-relevant information online; store new memories without forgetting; reduces forgetting while staying computationally efficient.
- **§1 Introduction.** ANNs freeze after training; retraining on shifted distributions ⇒ overfitting/forgetting. Biological synapses are complex molecular machines, not scalars. Proposal: 3D synaptic state (past value, current value, importance $\omega$); consolidate important synapses at task switch; new tasks learned by unimportant synapses.
- **§2 Prior work.** Taxonomy: (1) architectural (freezing, reduced LR, ReLU/Maxout/LWTA, dropout, progressive nets — grows with tasks); (2) functional (LwF distillation, activation-$\ell_2$ — expensive, need old-net forward passes); (3) structural (EWC — diagonal Fisher, cost linear in #outputs, limits high-dim outputs). SI is structural but online.
- **§3 Synaptic framework.** Loss-change first order Eq. (1); path integral Eqs. (2)–(3) defining $\omega_k^\mu$; online running-sum approximation; SGD noise ⇒ over-estimate. Surrogate loss Eq. (4); regularization strength Eq. (5) with $\Delta_k^\nu$ and damping $\xi$; strength $c$ ($c=1$ ideal, <1 to compensate noise); update schedule ($\omega$ continuous, $\Omega$/reference at task end, $\omega$ reset). Fig. 1/Fig. 2 surrogate intuition (3 matching conditions).
- **§4 Theoretical analysis.** Quadratic error Eq. (6); continuous-time descent Eqs. (7)–(9); $Q$ matrix Eq. (10)–(11); reductions to Hessian: averaged over inits Eq. (12) (justifies Eq. 5 normalization), diagonal Hessian Eq. (13), rank-1 Hessian Eq. (14) (interesting low-rank/spare-capacity case). Caveat: exact only for constant $H$; general case only correlational.
- **§5 Experiments.** 5.1 Split MNIST (Fig. 3, $c=0$ vs $c=1$). 5.2 Permuted MNIST (Fig. 4 vs EWC/SGD/dropout; Fig. 5 importance-correlation matrices). 5.3 Split CIFAR-10/100 (Fig. 6; consolidation prevents age-dependent decline + reduces overfitting).
- **§6 Discussion.** Similar to EWC but online + trajectory-wide; needs higher-dimensional synapses; biology of complex synapses (state-dependent plasticity, decaying tags, reversible changes). "Add intelligence to synapses" as a research direction.

---

## 5. Rebuffi et al. (2017) — iCaRL: Incremental Classifier and Representation Learning

**PDF:** `docs/project/references/continual_learning/sources/Rebuffi et al. 2017 - iCaRL.pdf`
**Venue:** CVPR 2017 (Oxford / IST Austria). **Type:** algorithm / empirical (class-incremental image classification).
**Primer link:** the **replay** family template for *class-incremental* learning — exemplar storage + distillation + nearest-mean classification.

### Phase 1: Foundational Overview

**Introduction (plain language).** The earlier methods here (EWC, SI, progressive nets) mostly assume you know which task an input belongs to, and you keep learning the *same set of classes*. iCaRL tackles a harder, more realistic setting: **class-incremental learning**. Classes arrive over time in batches — first you learn to recognize cats and dogs, later birds and fish, later still lizards — and *at any moment* the system must classify an image into *any* class seen so far, without being told which batch it came from. A child at the zoo learns new animals without forgetting the pet at home; iCaRL wants that. The constraints: it must (i) learn from a stream where classes appear at different times, (ii) always give a competitive single classifier over *all* classes so far, and (iii) keep memory bounded (not store all data).

**Key finding.** Naively finetuning a network on each new batch causes accuracy to collapse (the network ends up predicting only the most recent classes — its confusion matrix has all mass on the last batch). iCaRL avoids this by combining **three** ingredients: (1) classify by **nearest-mean-of-exemplars** rather than the network's own output layer; (2) store a small, fixed budget of **exemplar images** per class, chosen by a **herding** procedure that best approximates each class's mean feature; (3) train the representation with a **classification + distillation loss** that rehearses old exemplars and preserves old outputs. On iCIFAR-100 and iImageNet, iCaRL learns 100–1000 classes incrementally where finetuning, fixed-representation, and distillation-only (LwF) baselines fail.

**Initial takeaway.** iCaRL is the canonical *exemplar-replay* method for growing class sets, and the paper is unusually clear about *why each of its three parts matters* (via ablations). Its central insight: when the feature representation $\varphi$ keeps changing, a network's learned output weights $w_y$ go stale — but a *nearest-mean* classifier that recomputes prototypes from stored images automatically tracks the changing representation, so it is robust to exactly the drift that causes forgetting.

### Phase 2: Graduate-Level Deep Dive

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

### Appendix: Section-by-Section Backbone

- **Abstract.** iCaRL learns classifiers + representation together in class-incremental fashion; only a few classes present at once; CIFAR-100 and ImageNet experiments show it succeeds where others fail.
- **§1 Introduction.** Class-incremental definition (criteria i–iii: stream, competitive multi-class classifier any time, bounded memory). Naive SGD ⇒ catastrophic forgetting; existing methods limited to fixed representations. Three components: nearest-mean-of-exemplars, herding-based exemplar selection, representation learning via distillation + prototype rehearsal.
- **§2 Method.** 2.1 Architecture: CNN feature extractor $\varphi$ + sigmoid outputs Eq. (1); network for representation only. Algorithms 1 (classify) & 2 (incremental train). 2.2 Nearest-mean classification Eq. (2); decoupled-weights argument (why network-output classification forgets; prototypes track $\varphi$). 2.3 Representation learning (Algorithm 3): combined set of new data + exemplars; store pre-update outputs $q_i^y$; classification + distillation BCE loss; store exemplars as images not features. 2.4 Exemplar management: $m=K/t$; herding construction (Algorithm 4, greedy mean-approximation, prioritized list); reduction (Algorithm 5, keep first $m$); herding vs random subsampling.
- **§3 Related work.** Fixed-representation methods (NCM, Mensink et al.; open-set; ensembles; zero-shot). Representation-learning methods; McCloskey catastrophic forgetting; two classical strategies (freeze/grow vs. rehearsal). Freeze/grow (progressive nets, tree-structured) violates bounded memory; iCaRL uses rehearsal + within-network distillation (LwF connection).
- **§4 Experiments.** Benchmark protocol (fixed random order, per-batch test on seen classes, average incremental accuracy). iCIFAR-100 (2/5/10/20/50 per batch, 32-layer ResNet, $K=2000$) and iILSVRC-small/full (18-layer ResNet, $K=20000$). 4.1 Results: iCaRL > LwF.MC > fixed-repr > finetuning; gap grows with incrementality; confusion matrices (Fig. 3) — iCaRL uniform, finetune last-batch-only, fixed-repr first-batch bias, LwF.MC recency bias. 4.2 Differential analysis: hybrid1/2/3 ablation (Table 1a); NCM comparison (Table 1b); memory-budget curve (Fig. 4).
- **§5 Conclusion.** Three components recap; exemplars are the main driver; still below batch (joint) training; future: exemplar-free (autoencoder-encoded features), privacy settings.

---

## 6. Lopez-Paz & Ranzato (2017) — Gradient Episodic Memory (GEM)

**PDF:** `docs/project/references/continual_learning/sources/Lopez-Paz & Ranzato 2017 - Gradient Episodic Memory (GEM).pdf`
**Venue:** NIPS 2017 (Facebook AI Research). **Type:** algorithm + evaluation framework (supervised continuum).
**Primer link:** the **replay** family "with a twist" — store old examples, but use them to *constrain the gradient* so old-task loss never increases; uniquely allows *positive backward transfer*.

### Phase 1: Foundational Overview

**Introduction (plain language).** GEM contributes two things. First, a **measurement framework**: how should we even score a continual learner? Beyond average accuracy, GEM defines **backward transfer** (does learning a new task *help or hurt* old tasks?) and **forward transfer** (does having learned earlier tasks *help* a new one before you train on it?). Catastrophic forgetting is just *large negative backward transfer* in this language. Second, an **algorithm**: GEM stores a small episodic memory of examples from each past task, and at every gradient step it *checks* whether the proposed update would increase the loss on any stored past task. If it would, GEM **rotates** the gradient to the nearest direction that doesn't — so old-task losses are never allowed to rise, but *can* fall (which is *positive* backward transfer). The setting is deliberately harsh and human-like: many tasks, few examples each, **each example seen only once**.

**Key finding.** On MNIST-permutations, MNIST-rotations, and incremental CIFAR-100 (each with 20 tasks, single pass), GEM minimizes forgetting (near-zero or *positive* backward transfer) and matches or beats EWC and iCaRL — GEM even reaches the "oracle" accuracy of iid multi-task training on MNIST-rotations while EWC lags. Its cost advantage: the constrained-optimization step is solved in the space of *tasks-so-far* ($t-1$ variables), not parameters (millions), via a small quadratic program.

**Initial takeaway.** GEM reframes continual learning as **constrained optimization**: minimize the current task's loss *subject to not increasing any past task's loss*. That inequality-constraint stance is what distinguishes it from iCaRL/LwF (which enforce output *invariance* via distillation, forbidding backward transfer) and from EWC/SI (which softly penalize *weight* movement). GEM's episodic-memory-as-constraint is the direct realization of French's (§1) rehearsal idea, but expressed on gradients rather than on the loss itself.

### Phase 2: Graduate-Level Deep Dive

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

### Appendix: Section-by-Section Backbone

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


## 7. Ash & Adams (2020) — On Warm-Starting Neural Network Training

**PDF:** `docs/project/references/continual_learning/sources/Ash & Adams 2020 - On Warm-Starting Neural Network Training.pdf`
**Venue:** NeurIPS 2020. **Authors:** Jordan T. Ash (Microsoft Research NYC), Ryan P. Adams (Princeton).

### Phase 1 — Foundational Overview

**The problem in one sentence.** When new data arrive and you want to update a neural network, the intuitive, cheap thing to do is to keep yesterday's weights and just keep training on the bigger dataset — a "warm start." This paper shows that warm-starting *hurts the model's test accuracy*, even though it reaches exactly the same training accuracy as a network trained from scratch.

**The concrete demonstration.** Train an 18-layer ResNet on 50% of CIFAR-10 to convergence, then continue training it on 100% of CIFAR-10. Compare it to a fresh randomly-initialized ResNet trained on 100% of CIFAR-10. Both reach ~100% *training* accuracy, but the warm-started one generalizes several percentage points *worse* on held-out test data (e.g., 51.7% vs. 56.2% for ResNet+SGD on CIFAR-10). The gap is robust across architectures (ResNet, MLP), optimizers (SGD, Adam), and datasets (CIFAR-10, CIFAR-100, SVHN). Logistic regression — a convex model — is *not* damaged, which is the tell that this is a non-convex-optimization pathology, not a statistical one.

**Key findings.**
- The damage appears after *very little* pretraining — a few epochs, even before the pretraining phase reaches 100% accuracy. Early stopping does not save you.
- No standard fix works: batch normalization, larger/smaller batch size, larger learning rate, L2 weight decay, confidence penalties, adversarial training — all fail to close the gap while preserving the warm start's speed benefit. Warm-started models that *do* generalize well only do so by essentially "forgetting" their initialization (their converged weights end up nearly uncorrelated with the warm-start weights), which erases the time savings.
- **The fix — shrink-and-perturb (SP).** Before the next round of training, replace each weight by a shrunken-plus-noised version: multiply it by a factor $\lambda<1$ and add small Gaussian noise. This one-line trick closes the generalization gap *and* keeps the training-speed benefit of warm-starting.

**Initial takeaway.** A "head start" from prior training can be a liability, not an asset, whenever the second-round dataset is large (data-rich). The mechanism is a **gradient imbalance**: for a warm-started network, gradients from new/unseen data are much larger in magnitude than gradients from already-fit data, so the optimizer's trajectory is biased and lands in a worse-generalizing minimum. Shrink-and-perturb re-balances those gradient magnitudes without destroying the learned function. This is the supervised-learning ancestor of the whole "resets" line in RL (Nikishin 2022, D'Oro 2023) — see the project primer §2 Phase 2.

### Phase 2 — Graduate-Level Deep Dive

**Setup and the empirical gap.** Let a dataset $\mathcal{D}$ be split into halves $\mathcal{D}_1, \mathcal{D}_2$. Warm-starting fits $\theta^{(1)} = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D}_1)$ to convergence, then initializes round two at $\theta^{(1)}$ and fits on $\mathcal{D}_1 \cup \mathcal{D}_2$. The random-init control fits the same union from a fresh $\theta_0 \sim \text{init}$. Both attain $\approx 100\%$ training accuracy (zero training loss on a modern over-parameterized ResNet), so the test-accuracy gap is entirely an implicit-bias / generalization phenomenon, not an optimization-failure-to-fit phenomenon. The gap is shown (Appendix Fig. 10) to be *inversely proportional to the fraction of data available in round one* — the more you pretrain, the worse the eventual generalization.

**The shrink-and-perturb operator.** At training round $t$, each learnable parameter is re-initialized as

$$\theta_i^{t} \;\leftarrow\; \lambda\, \theta_i^{t-1} \;+\; p_t, \qquad p_t \sim \mathcal{N}(0, \sigma^2), \quad 0 < \lambda < 1 .$$

In practice the perturbation $p_t$ is drawn as a *scaled freshly-initialized network* rather than i.i.d. Gaussian, so that per-layer variances match the architecture's initialization scheme. Two claims justify why this works: (i) shrinking preserves the learned hypothesis, and (ii) shrinking re-balances the gradients.

**(i) Shrinking preserves the hypothesis (Proposition 1).** Consider an $L$-layer ReLU network with no bias terms and no batch normalization. ReLU is positively homogeneous: $\text{ReLU}(\lambda z) = \lambda\,\text{ReLU}(z)$ for $\lambda>0$. Propagating a global weight-shrink by $\lambda$ through all $L$ layers, the pre-softmax logits $z(x) = f_\theta(x)$ scale as

$$f_{\lambda\theta}(x) \;=\; \lambda^{L}\, f_\theta(x) .$$

*Derivation.* Let layer $\ell$ compute $h_\ell = \text{ReLU}(W_\ell h_{\ell-1})$ with $h_0 = x$. Replacing every $W_\ell \to \lambda W_\ell$:
$h_1' = \text{ReLU}(\lambda W_1 x) = \lambda\,\text{ReLU}(W_1 x) = \lambda h_1$; inductively $h_\ell' = \text{ReLU}(\lambda W_\ell \cdot \lambda^{\ell-1} h_{\ell-1}) = \lambda^{\ell} h_\ell$. The final linear map (layer $L$) gives logits $z' = \lambda W_L h_{L-1}' = \lambda\cdot\lambda^{L-1} z = \lambda^L z$. $\square$

Because $\arg\max_c z_c = \arg\max_c \lambda^L z_c$ for $\lambda>0$, the **predicted class label is unchanged** — the hypothesis $\arg\max f_\theta(x)$ is preserved. What *does* change is the *confidence*: shrinking the logits by $\lambda^L$ pushes the softmax toward uniform, raising output entropy and hence raising the cross-entropy loss. For architectures with batch-norm (ResNet), BN's running mean/variance absorb the rescaling, so performance is essentially invariant to $\lambda$ except at extreme shrinkage (Fig. 6). For ReLU MLPs *with* bias, the property degrades gracefully — damage only appears for $\lambda < 0.6$.

**(ii) Shrinking re-balances gradients (the mechanism).** The paper's diagnosis is a gradient-magnitude imbalance. In warm-started round-two training, decompose the batch loss into contributions from already-fit data $\mathcal{D}_1$ and new data $\mathcal{D}_2$. Because the network already fits $\mathcal{D}_1$ (near-zero loss there), $\|\nabla_\theta \mathcal{L}(\theta;\mathcal{D}_1)\| \ll \|\nabla_\theta \mathcal{L}(\theta;\mathcal{D}_2)\|$. The aggregate gradient is dominated by $\mathcal{D}_2$, so the optimizer moves as if it were doing a biased single-task step — an imbalance known to be pathological in multi-task optimization (gradient surgery, Yu et al. 2020). Figure 5 measures the two gradient norms separately over round two and shows a drastic gap for warm-started models.

Shrink-and-perturb repairs this because shrinking *raises the loss on the already-fit data*: from Proposition 1, shrinking logits by $\lambda^L$ increases the cross-entropy on $\mathcal{D}_1$ from ~0 back to a nontrivial value, which restores $\|\nabla_\theta \mathcal{L}(\theta;\mathcal{D}_1)\|$ to a magnitude comparable with $\|\nabla_\theta \mathcal{L}(\theta;\mathcal{D}_2)\|$. The gradients are re-standardized *while the coarse learned structure (the argmax hypothesis) is preserved* — that combination is exactly what pure noise injection cannot achieve (noise re-balances gradients but destroys the function; Appendix Table 4).

**Relationship to weight decay.** Applying SP at *every* SGD step yields the update

$$\theta_i \;\leftarrow\; \lambda\!\left(\theta_i + \eta\,\frac{\partial \mathcal{L}}{\partial \theta_i}\right) + p ,$$

so the shrink factor $\lambda$ behaves like a weight-decay coefficient plus injected noise. But SP is *not* reducible to weight decay: Appendix Table 3 shows L2-regularized models are still vulnerable to the warm-start gap, and SP closes the gap even on top of aggressive weight decay. The regularization benefit is a small secondary effect (marginal improvement even on static data); the primary benefit is the sequential-training gradient re-balancing.

**Trade-off surface.** Figure 8 sweeps $(\lambda, \sigma)$: the bottom-left corner ($\lambda\to0$) is pure random init (best generalization, slowest), the top-right ($\lambda\to1$) is pure warm start (fast, worst generalization). Intermediate $\lambda\approx0.6$, noise scale $0.01$ recovers random-init test accuracy while retaining most of the speed; smaller $\lambda$ can even *outperform* random init. Adding the perturbation improves both time and generalization over shrink-alone.

**Pre-training / transfer connection.** SP also robustifies transfer: when pre-training on a source then fitting a fraction of a target, warm-starting helps when target data are *scarce* (few-shot regime) but crosses over to hurting when target data are *abundant* (Fig. 9). SP tracks the better of the two strategies automatically at $\lambda=0.3$, noise $10^{-4}$, removing the need to predict the crossover.

**Project relevance.** This is the canonical citation for "carrying weights forward can lose to a from-scratch baseline" — precisely the regime the project's curriculum agent hit (primer cross-links `curriculum_underperformed_baseline_plasticity_vs_budget`). SP ($\theta\leftarrow\lambda\theta+\epsilon$) is a *partial-reset* primitive whose RL descendants (Nikishin's last-layer resets, D'Oro's high-replay-ratio resets) are the corrective family in the primer §4.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Real ML systems ingest data piecemeal (finance, ads, recsys, active learning). Convex theory says warm-start; deep nets contradict it — warm-starting hurts generalization without hurting training accuracy. Motivates studying "when and how" plus a simple fix. Notes findings are *not* inconsistent with small-data pre-training / few-shot transfer; the claim is specifically about the *large-data* second round.
- **§2 Warm Starting Damages Generalization.** §2.1 Basic batch updating: 50%→100% two-phase protocol; Table 1 shows consistent, significant test-accuracy damage for ResNet & MLP across SGD/Adam and CIFAR-10/100/SVHN; convex LR unaffected; effect stronger on harder datasets. §2.2 Online learning: streaming CIFAR-10 in 1000-sample batches to a ResNet; warm-start trains far faster but generalization gap grows with more data (Fig. 2).
- **§3 Conventional Approaches.** §3.1 Batch size / learning rate sweeps (Fig. 3): warm-started models can match random-init accuracy only by giving up the speed benefit; well-generalizing warm-started models have low weight-correlation with their init (Appendix Fig. 11) — they "forgot." §3.2 Speed of damage (Fig. 4): only a few epochs of pretraining suffice to inflict the gap. §3.3 Regularization (L2, confidence penalty, adversarial): helps a little, does not close the gap.
- **§4 Shrink, Perturb, Repeat.** Defines SP $\theta_i^t \leftarrow \lambda\theta_i^{t-1}+p_t$. "Shrinking preserves hypotheses" (Prop 1, $\lambda^L$ logit-scaling, entropy increase). "Shrink-perturb balances gradients" (Fig. 5, the imbalance mechanism). Trade-off study $(\lambda,\sigma)$ (Figs. 7, 8). §4.1 relation to weight decay (per-step SP ≈ noisy L2, but not reducible to it). §4.2 relation to pre-training (crossover; SP tracks the better strategy, Fig. 9).
- **§5 Discussion / Related Work.** Warm-start understood for convex models only; connections to critical learning periods (Achille et al.), initialization theory, distance-from-init generalization (Nagarajan & Kolter), margin/flat-minima/complexity generalization literature, and pre-training.
- **§6 Broader Impact.** SP reduces the compute/energy cost of retraining-from-scratch ("Red AI"), democratizing online/active-learning research.

---

## 8. Kumar et al. (2021) — Implicit Under-Parameterization Inhibits Data-Efficient Deep RL

**PDF:** `docs/project/references/continual_learning/sources/Kumar et al. 2021 - Implicit Under-Parameterization.pdf`
**Venue:** ICLR 2021. **Authors:** Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, Sergey Levine (UC Berkeley / Google Research / MILA).

### Phase 1 — Foundational Overview

**The problem in one sentence.** In value-based deep RL (Q-learning, actor-critic), the network is trained to regress onto *targets it generated itself* one step earlier — this is called **bootstrapping**. Kumar et al. show that repeating this self-regression, with gradient descent, gradually *collapses the rank* of the value network's internal features: a 512-dimensional feature layer ends up using only 20–100 truly independent directions. The network behaves as if it had far fewer parameters than it does — hence "implicit under-parameterization" — and performance drops with it.

**What "rank collapse" means intuitively.** The value network maps each state (or state-action pair) to a feature vector, then a final linear layer reads out the value. If the network maps *different* states to *nearly-parallel* feature vectors ("aliasing"), the final layer can no longer tell those states apart, so it cannot represent value functions that need to distinguish them. Effective rank counts how many genuinely-distinct feature directions survive; when it collapses, expressivity collapses.

**Key findings.**
- Rank collapse is demonstrated on Atari (DQN), continuous-control Gym (SAC), and a tabular-comparable gridworld (neural fitted-Q), in *both* offline RL (fixed dataset) and data-efficient online RL.
- **Lower rank ⇒ worse performance.** Across domains, the rank curve and the return curve fall together; when rank collapses, the network can no longer fit its own TD targets (TD error rises) nor the true optimal values $Q^*$.
- **More data reuse makes it worse.** Increasing the number of gradient updates per environment step ($n$) — the lever you pull for sample efficiency — accelerates rank collapse and degrades performance. Offline RL (infinite reuse of a fixed dataset) is the worst case.
- **Bootstrapping is the cause, isolated by controls.** Rank collapse persists even when you re-initialize the network from scratch each fitting iteration (rules out bad init / non-stationarity) and in pure policy-evaluation (rules out the max operator). It *disappears* when you regress to Monte-Carlo returns instead of bootstrapped targets. So it is the *self-referential target*, not the data or the control problem, that drives the collapse.
- **Fix (partial).** A singular-value penalty $\mathcal{L}_p(\Phi) = \sigma_{\max}^2(\Phi) - \sigma_{\min}^2(\Phi)$ that balances the feature spectrum mitigates collapse and improves DQN on 16/16 and CQL on 11/16 offline Atari games. It treats the symptom, not the root cause.

**Initial takeaway.** This is the first paper to tie a concrete, measurable representational pathology (feature-rank collapse) to the interaction of *bootstrapping* + *the implicit regularization of gradient descent*. It is the RL-specific sibling of the warm-start / capacity-loss story: the learner degrades its own substrate through self-referential training.

### Phase 2 — Graduate-Level Deep Dive

**Preliminaries.** MDP $(\mathcal{S},\mathcal{A},R,P,\gamma)$. $Q^\pi$ is the fixed point of the Bellman operator $\mathcal{T}^\pi Q(s,a) = R(s,a) + \gamma\,\mathbb{E}_{s'\sim P,\,a'\sim\pi}[Q(s',a')]$; $Q^*$ the fixed point of $\mathcal{T}Q(s,a) = R(s,a) + \gamma\,\mathbb{E}_{s'}[\max_{a'}Q(s',a')]$. Practical deep Q-learning minimizes the mean-squared TD error

$$\mathcal{L}(\theta) = \sum_{s,a}\big(R(s,a) + \gamma\,\bar{Q}_\theta(s',a') - Q_\theta(s,a)\big)^2 ,$$

where $\bar{Q}_\theta$ is a delayed target network. Write the penultimate-layer features as $\Phi \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|\times d}$ so $Q(s,a) = w^\top \Phi(s,a)$. The abstraction studied is **fitted Q-iteration (FQI)**: at fitting iteration $k$, form targets $y_k = R + \gamma P^\pi Q_{k-1}$ and take $T$ gradient steps to minimize $(Q_\theta - y_k)^2$.

**The measurement — effective rank.** For threshold $\delta$ (they use $0.01$),

$$\mathrm{srank}_\delta(\Phi) = \min\Big\{ k : \frac{\sum_{i=1}^{k}\sigma_i(\Phi)}{\sum_{i=1}^{d}\sigma_i(\Phi)} \ge 1-\delta \Big\},$$

with singular values $\sigma_1 \ge \dots \ge \sigma_d \ge 0$. It counts the number of leading singular directions that carry $(1-\delta)$ of the spectral mass — the number of "effective" independent feature components. High $\mathrm{srank}\approx d$ means states map to near-orthogonal features; low $\mathrm{srank}$ means aliasing onto a small subspace.

**Definition 1 (Implicit under-parameterization).** A reduction in $\mathrm{srank}_\delta(\Phi)$ that occurs implicitly as a by-product of learning the deep Q-network. Note: rank reduction *also* occurs in supervised learning where it is *beneficial* (a generalization-friendly implicit bias); the claim is that bootstrapping drives it *further*, into a *harmful* collapse.

#### Theoretical analysis I — kernel-regression (NTK) view

Model each bootstrapping round as squared-TD regression with a universal-kernel regularizer (coefficient $c\ge0$) capturing the inductive bias of gradient descent under early stopping (following Mobahi et al.'s self-distillation analysis):

$$Q_{k+1} \leftarrow \arg\min_{Q\in\mathcal{Q}} \sum_{s_i,a_i\in\mathcal{D}} \big(Q(s_i,a_i) - y_k(s_i,a_i)\big)^2 \;+\; c\sum_{(s,a)}\sum_{(s',a')} u\big((s,a),(s',a')\big)\,Q(s,a)Q(s',a'). \tag{1}$$

The closed-form solution is $Q_{k+1}(s,a) = g_{(s,a)}^\top (cI+G)^{-1} y_k$, with Gram matrix $G$ of the induced positive-definite kernel and $g_{(s,a)}$ the corresponding row. Substituting the FQI target $y_k = R + \gamma P^\pi Q_{k-1}$ and defining $A = G(cI+G)^{-1}$ gives the recurrence (with $Q_0 = 0$):

$$Q_{k+1} = A\,[R + \gamma P^\pi Q_k] = A\sum_{i=1}^{k}\gamma^{k-i}(P^\pi A)^{k-i} R \;=:\; A\,M_k\,R. \tag{2}$$

Here $M_k$ linearly maps rewards to Q-values, so the *expressivity of $M_k$ bounds what value functions the learner can represent*.

**Theorem 4.1 (spectrum of $M_k$ sparsifies).** Let $S = \gamma P^\pi A$ be a normal matrix. Then there is a strictly increasing sequence of iterations $(k_l)_{l\ge1}$, $k_1=0$, such that for any two singular values $\sigma_i(S) < \sigma_j(S)$ and any $l' \ge l$,

$$\frac{\sigma_i(M_{k_{l'}})}{\sigma_j(M_{k_{l'}})} < \frac{\sigma_i(M_{k_l})}{\sigma_j(M_{k_l})} \le \frac{\sigma_i(S)}{\sigma_j(S)}.$$

Hence $\mathrm{srank}_\delta(M_{k_{l'}}) \le \mathrm{srank}_\delta(M_{k_l})$; if $S$ is PSD the decrease is monotone in *every* iteration.

*Proof sketch / intuition.* For a normal $S$, singular values equal $|\text{eigenvalues}|$, and $M_k = \sum_{i=1}^k \gamma^{k-i}(P^\pi A)^{k-i}R$ is a matrix polynomial in $S$. Raising $S$ to increasing powers along the sequence exponentially amplifies the *ratio* between any two distinct singular values (the larger one dominates), so the relative weight of the smaller singular directions shrinks toward zero. As the ratios $\sigma_i/\sigma_j$ collapse, the number of directions carrying $(1-\delta)$ of the mass — the effective rank — decreases. $\square$ The takeaway: bootstrapping's repeated composition drives a *generally decreasing* (not necessarily every-iteration) rank trend, unlike self-distillation which is monotone.

#### Theoretical analysis II — deep-linear-network view (pinpoints *when*)

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

#### Mitigation

Since $\mathrm{srank}_\delta$ is non-differentiable, use the surrogate penalty

$$\mathcal{L}_p(\Phi) = \sigma_{\max}^2(\Phi) - \sigma_{\min}^2(\Phi), \tag{6}$$

added to the TD loss with weight $\alpha=0.001$, computed via SVD on a minibatch feature matrix. Minimizing the top singular value while lifting the bottom one balances the spectrum (effective rank is maximized when singular values are equal-magnitude). Result: on 5%-replay offline Atari, DQN$+\mathcal{L}_p$ improves 16/16 games (median $+74.5\%$), CQL$+\mathcal{L}_p$ improves 11/16 (median $+14.1\%$). Online (Rainbow): median $+20.6\%$; but DQN online got *worse* ($-11.5\%$) — the penalty is a symptom-level fix that "does not address the root cause."

**Distinction from prior rank work.** Yang et al. (2019) studied the rank of the $Q^*$-*matrix* ($|\mathcal{S}|\times|\mathcal{A}|$, upper-bounded by $\#$actions) and argued low rank is good. Kumar et al. study a *different object* — the learned *features* $\Phi$ — and show *feature*-rank collapse *hurts*. Network re-initialization (proposed by Igl et al./Fedus et al. for non-stationarity) does *not* prevent the collapse (Fig. 4c), further isolating bootstrapping as cause.

**Project relevance.** "Rank collapse" is one of the named mechanisms in the primer's Phase-3 decomposition (Lyle 2024 lists rank as one of several independent causes). Kumar is the origin of the feature-rank diagnostic that Lyle 2022 (§4 below) adapts (Lyle drops the max-singular-value normalization to also capture *representation collapse toward zero*). The $n$-updates-per-step finding directly underpins the project's replay-ratio study (`replay_ratio_speed_vs_performance`).

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Names "implicit under-parameterization": bootstrapping + GD makes an expressive value net behave under-parameterized via feature over-aliasing; aggravated by higher data re-use (offline RL = worst). Contributions: identify, demonstrate (Atari/Gym, offline+online), analyze (kernel + deep-linear), mitigate (SVD penalty).
- **§2 Preliminaries.** MDP, Bellman operators $\mathcal{T}^\pi/\mathcal{T}$, mean-squared TD error, target network, feature matrix $\Phi$ ($Q=w^\top\Phi$). Abstracts practical methods into generic FQI (Alg. 1): targets $y_k = R+\gamma P^\pi Q_{k-1}$, $T$ inner gradient steps.
- **§3 Implicit Under-Parameterization in Deep Q-Learning.** Defines $\mathrm{srank}_\delta(\Phi)$. Offline RL (Fig. 2): rank drops after initial learning, final rank tiny (20–100 of 512); worse with more gradient steps; persists with 4× data and under CQL (rules out coverage / distribution mismatch). Online RL (Fig. 3): higher $n$ (updates/env-step) ⇒ faster collapse + worse return. §3.1 Mechanism: as rank falls, TD error and $Q^*$-fitting error rise (Fig. 4a,b). Controls: rank collapse survives per-iteration re-initialization (4c) and FQE policy-evaluation; *vanishes* with Monte-Carlo targets (4d) ⇒ bootstrapping is the cause.
- **§4 Theoretical Analysis.** §4.1 Kernel regression: regularized objective Eq. 1, recurrence Eq. 2 ($Q_{k+1}=AM_kR$), Theorem 4.1 (singular-value ratios shrink ⇒ $\mathrm{srank}(M_k)$ decreases). §4.2 Deep linear nets: singular-value ODE Eq. 4, Prop. 4.1 (larger $\sigma$ evolve faster ⇒ within-iteration rank drop), abstract penalized objective Eq. 5, Prop. 4.2 (bound across iterations), Theorem 4.2 (near-fixed-point $\Rightarrow$ monotone collapse, increases distance to fixed point; Fig. 5).
- **§5 Mitigation.** Penalty $\mathcal{L}_p = \sigma_{\max}^2-\sigma_{\min}^2$ (Eq. 6), $\alpha=0.001$; prevents collapse (Fig. 6) and improves offline DQN 16/16, CQL 11/16 (Fig. 7); online results mixed; explicitly a symptom-level fix.
- **§6 Related Work.** Tabular/linear Q-learning error-propagation & divergence; convergence guarantees under restrictive assumptions contradicted near the optimum by Thm 4.2; distinction from Yang et al.'s $Q^*$-matrix rank; re-init (Igl/Fedus) doesn't help.
- **§7 Discussion.** Root-cause is GD's implicit regularization on bootstrapped objectives; future work: auxiliary losses that preserve rank passively.

---

## 9. Berariu et al. (2021) — A Study on the Plasticity of Neural Networks

**PDF:** `docs/project/references/continual_learning/sources/Berariu et al. 2021 - A Study on the Plasticity of Neural Networks (preprint).pdf`
**Venue:** arXiv preprint 2021 (v2, Oct 2023). **Authors:** Tudor Berariu, Wojciech Czarnecki, Soham De, Jörg Bornschein, Samuel Smith, Razvan Pascanu, Claudia Clopath (Imperial College London / DeepMind).

### Phase 1 — Foundational Overview

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

### Phase 2 — Graduate-Level Deep Dive

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

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Continual learning desiderata (no forgetting, structural transfer, backward transfer, enduring capacity to acquire). Focuses on *plasticity* = ability to keep learning. Three "can't learn" nuances: (i) can't minimize train loss (PackNet frozen neurons, over-constrained EWC); (ii) less data-efficient (negative forward transfer, still reaches 0 train error); (iii) **can reach 0 train error but converges to a poorer-generalizing minimum** — *this* is the paper's focus. Defines generalization gap vs. intransigence. Oct-2023 footnote: two distinct "plasticity loss" phenomena (generalization vs. optimization; e.g. Dohare); treat separately.
- **§2 Generalisation Gap — Experiments.** Reproduces Ash & Adams (Fig. 1). Optimizer-invariance (Adam/RMSprop/SGD/mSGD) ⇒ minimum-quality problem not trajectory problem. Few pretraining epochs suffice. Smooth transition $p=1-\gamma^{50n/N}$ still induces gap (Fig. 3) — RL implication. Multi-stage aggravation + class imbalance (Fig. 4). Width/depth don't fix it (Fig. 5). Top-layer reset needed to recover; no speed benefit from keeping early layers (Fig. 6).
- **§3 Two Phases of Learning (hypothesis).** Flat-vs-sharp minima; exploration (noise-driven) then refinement (gradient-flow) phases. Conjecture: pretraining reduces tuning-phase gradient noise ⇒ weaker exploration ⇒ narrower minimum ⇒ gap. Test: 10× tuning LR reduces gap (Fig. 7).
- **§4 Conclusions.** Gap is robust (smooth transition, multi-stage, model size, partial reset). Continual learning may be hurt by fine-tuning compact models; retraining from stored data can beat warm-starting; tracking the gap is a new facet of forward transfer.
- **Appendix A.** Experimental details; other-optimizer figures; smooth-transition sampling; class-imbalance split methodology; depth/width architectures; layer-reset details.
- **Appendix B.** Supporting two-phase evidence (Achille, Golatkar, Gur-Ari, Li, Jastrzebski, Ghorbani).

---

## 10. Lyle, Rowland & Dabney (2022) — Understanding and Preventing Capacity Loss in RL

**PDF:** `docs/project/references/continual_learning/sources/Lyle et al. 2022 - Understanding and Preventing Capacity Loss in RL.pdf`
**Venue:** ICLR 2022. **Authors:** Clare Lyle (Oxford), Mark Rowland & Will Dabney (DeepMind).

### Phase 1 — Foundational Overview

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

### Phase 2 — Graduate-Level Deep Dive

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

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Deep RL is brittle in sparse-reward tasks (seed-to-seed variance) vs. robust supervised learning; blames non-stationary prediction targets. Thesis: agents lose capacity to quickly fit new targets; extreme = representation collapse = no progress. Proposes InFeR. Striking claim: Montezuma-type games improvable *without* smart exploration given the right representation objective ⇒ poor sparse-reward performance is partly a representation-learning, not exploration, failure.
- **§2 Background.** MDP, $Q^\pi$, Q-learning bootstrap target, deep-RL TD loss Eq. 2 (replay buffer, target net $\bar\theta$), features = penultimate layer $\phi_\theta$.
- **§3 Capacity Loss.** §3.1 Target-fitting capacity (Def. 1, Eq. 3); H1 (iterative MNIST, Fig. 1, worse for small nets, over- vs under-parameterized); H2 (Atari checkpoints fit random targets worse over time, Fig. 2). §3.2 Representation collapse & performance: Feature rank (Def. 2, Eqs. 4–5), contrast with Kumar srank (no $\sigma_{\max}$ normalization; online regime); denser signal → higher rank (Fig. 3); rank↔score clustering, necessary-not-sufficient (Fig. 4a); recovery-from-collapse Pong seed (Fig. 4b).
- **§4 InFeR.** §4.1 Method: $k$ auxiliary heads regressed to init outputs, loss Eq. 6, $\beta$ amplification; results on 57 Atari (Fig. 5), Montezuma DDQN/Rainbow, MNIST. §4.2 Mechanism: H1 (random-subspace) rejected via frozen-random-features control (Fig. 6 left); H2 (whole-network dynamics) supported via DoubleRainbow width-doubling (Fig. 6 right).
- **§5 Related Work.** Auxiliary tasks; value-function geometry/stability; implicit under-parameterization (Kumar 2021) and spectral normalization (Gogianu 2021); sub-task interference & catastrophic forgetting (EWC, GEM, distillation).

---

## 11. Nikishin et al. (2022) — The Primacy Bias in Deep RL

**PDF:** `docs/project/references/continual_learning/sources/Nikishin et al. 2022 - The Primacy Bias in Deep RL.pdf`
**Venue:** ICML 2022. **Authors:** Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, Aaron Courville (Mila, Université de Montréal).

### Phase 1 — Foundational Overview

**The problem in one sentence.** Deep RL agents tend to **overfit to their earliest experiences** and then can't leverage the better data they collect later — the paper borrows the cognitive-science term **primacy bias** (like a guitarist who learns a passage badly first and can never unlearn the bad fingering even after being shown a better way). Because RL trains on a *growing* replay buffer, the agent sees its initial samples far more often, and that early over-specialization poisons the rest of training.

**Two clean demonstrations.**
1. **Heavy priming.** Train Soft Actor-Critic (SAC) on quadruped-run, but after collecting only the *first 100 transitions*, hammer the agent with $10^5$ gradient updates on that tiny buffer, *then* resume normal training. The agent *never recovers* — even after ~1M fresh transitions it can't learn. Extreme overfitting to early data is essentially unrecoverable.
2. **The data is fine; the learner is broken.** Take the buffer collected by a primacy-biased agent (SAC at 9 updates/step, which fails) and hand it to a *fresh* agent as its starting buffer. The fresh agent learns rapidly to near-optimal. So the primacy bias is *not* a failure to collect good data — it is a failure to *learn from* good data.

**The fix — resets.** Periodically re-initialize the *last few layers* of the agent's networks (from scratch) *while keeping the replay buffer intact*. Counterintuitively, throwing away learned weights *improves* final performance. It works across discrete (Atari 100k, SPR) and continuous (DMC, SAC & DrQ) domains, image and state inputs, prioritized and uniform replay, at *no extra compute cost*.

**Key findings.**
- Resets consistently improve IQM performance (e.g., SPR+resets on Atari 100k: IQM 0.478 vs SPR 0.380; SAC+resets on DMC: IQM 656 vs 501; DrQ+resets: 762 vs 569).
- The **higher the replay ratio** (updates per environment step), the **bigger the benefit from resets** — because high replay ratio is exactly what amplifies overfitting to early data. Resets let SAC reach its *best* performance at replay ratio 32 (+100% over no-resets) and stay functional even at extreme ratios 128/256.
- **Longer n-step targets** (higher-variance value estimates) also make the agent more primacy-prone, and resets help more as $n$ grows.
- **Keeping the buffer is essential** — emptying it at each reset is highly detrimental. The buffer acts as a non-parametric world model preserved across the reset.
- Resets also rescue **TD failure modes** (critic collapse in sparse reward; TD divergence / value overestimation) by giving the optimizer a fresh start.

**Initial takeaway.** Resets become a *first-class RL tool*: a tailor-made regularization against early-data overfitting that plain L2/dropout can't match. This paper is the direct parent of D'Oro et al. (2023) "Breaking the Replay Ratio Barrier," which turns resets into a scaling lever — and the RL descendant of Ash & Adams' shrink-and-perturb and Berariu's top-layer-reset probe.

### Phase 2 — Graduate-Level Deep Dive

**Definition (the primacy bias).** *A tendency to overfit early experiences that damages the rest of the learning process.* Deliberately wide-ranging: it has multiple roots (replay's over-exposure to initial samples; high replay ratio; high-variance n-step targets; TD instability) and multiple effects, all tied to improper learning from early data.

**Why RL amplifies it.** Standard components magnify the bias: (i) **experience replay** exposes the agent to its earliest samples more often than recent ones (they've been in the buffer longest); (ii) **replay ratio** — for sample efficiency, agents take many gradient updates per env step, re-fitting the same (early-dominated) data repeatedly; (iii) **n-step targets** $\mathbb{E}_\pi[r_t + \gamma r_{t+1} + \dots + \gamma^n Q^\pi(s_{t+n},a_{t+n})]$ trade bias for variance, and higher variance makes early overfitting easier. Compounding loop: an overfitted agent collects *worse* data, which further degrades learning.

**Experiment 1 — heavy priming (unrecoverable overfitting).** SAC on quadruped-run, default 1 update/step. Experimental arm: after 100 collected transitions, perform $10^5$ updates on that buffer, then resume. Figure 1: primed agent flatlines even after ~$10^6$ new transitions. Demonstrates the *compounding, near-absorbing* nature of the failure — a small early perturbation is amplified irreversibly.

**Experiment 2 — the buffer is sufficient (locate the failure in the learner).** SAC at 9 updates/step fails (primacy bias). Re-initialize a *fresh* agent but seed it with the *failed agent's buffer*: it learns rapidly to near-optimal (Fig. 2). Conclusion: the data is adequate; the *overfitted network* is what can't distill it. Random-init networks are unaffected by primacy bias and can fully exploit the collected experience.

**The intervention — resetting.** *Given an agent's network, periodically re-initialize the parameters of its last few layers while preserving the replay buffer.* Only two hyperparameters: reset periodicity and how many layers to reset. Domain-specific instantiation:
- **SPR (Atari 100k):** reset only the final linear layer of the 5-layer Q-network, every $2\times10^4$ steps.
- **SAC (DMC, dense state):** reset the *entire* 3-layer networks, every $2\times10^5$ steps (both Q-nets + targets, due to double Q-learning).
- **DrQ (DMC, pixels):** reset the last 3 of 7 layers of policy and value nets, ~10 times over training; buffer holds only the most recent 100k transitions.
No pre-training of the fresh parameters; return directly to the normal interaction/update cycle. Optimizer statistics are also reset but this has "almost no impact" (Adam moments recover quickly).

**Why does it recover so fast?** Two complementary explanations:
1. **Model-based view of the buffer.** The replay buffer is a *non-parametric model of the world*. After a reset, the agent forgets its (over-specialized) *behavior/parameters* but *retains its world model* in the buffer as the core of its knowledge. Emptying the buffer is highly detrimental (Appendix B) — confirming the buffer, not the weights, holds the essential knowledge.
2. **Representation-recovery view.** Zhang et al. (2019): most of learning amounts to recovering the right *representations*; with the buffer preserved, re-learning a good policy/actuator from good features is comparatively fast.

Resets trigger a *virtuous circle* (the inverse of the primacy-bias vicious circle): freed from negative priming, the agent leverages accumulated data better → improves → collects higher-quality data → better future updates.

**Resets as regularization.** If primacy bias is a special form of overfitting, resets are a *tailor-made* regularizer. Table 5 (Appendix B): resets overcome the primacy bias even where standard L2 and dropout *fail*, because the pathology is a discrete, accumulated over-specialization that continuous shrinkage doesn't undo.

**Interaction with replay ratio (the key scaling result).** Replay ratio = gradient updates per env step. Figure 5: reset benefit grows with replay ratio — SPR +40% at 4 updates/step; SAC's best performance at replay ratio 32 where resets add +100%; SAC remains "reasonable" at extreme ratios 128/256 where learning is otherwise "barely possible." Resets *reshape the hyperparameter landscape*, creating a new optimum at higher replay ratio (higher sample efficiency). This is the seed of D'Oro 2023's "replay-ratio barrier."

**Interaction with n-step targets.** Figure 6: as $n$ grows (higher target variance), the agent is more primacy-prone; reset benefit grows — up to +40% for SPR at $n=20$ (vs. none at $n=3$), 50–60% for SAC at increased $n$ (vs. 40% at $n=1$). Same mechanism: more overfitting pressure ⇒ more to gain from resetting.

**TD failure modes rescued.** (i) *Sparse-reward critic collapse:* DrQ on cartpole-swingup_sparse collapses (bootstrapping mostly on its own outputs, Kumar 2020) even though ~2% of buffer trajectories reach the goal; resets give a second chance to find a non-degenerate critic (Fig. 7 left) — evidence the primacy fix addresses *optimization*, not *exploration*. (ii) *TD divergence:* even with double Q-learning, the critic can overestimate unrecoverably; predicted values fail to decay for hundreds of thousands of steps; resets fix it (Fig. 7 right).

**Ablations (what/how to reset).** Number of layers is domain-dependent: SAC (dense state) can reset entirely; SPR best with last-layer-only (most Atari knowledge is in representations, so resetting deep layers wastes the hard-won encoder); DrQ best with last 3 of 7 (critic reset slightly more important than actor, since DrQ's encoder learns from the critic loss). Optimizer reset ≈ no effect. Reset frequency should scale with how fast the algorithm recovers; even a single reset can help. Resetting a *random subnetwork* was comparable or worse than resetting the last layers.

**Project relevance.** Primacy bias + resets is the primer's Nikishin-2022 entry (§2 Phase 2) and the origin of "resets as a first-class RL tool" (primer §4 corrective family). The replay-ratio finding directly feeds the project's `replay_ratio_speed_vs_performance` study, and its "keep the buffer, reset the last layers" recipe is a concrete, cheap intervention the project could trial against its curriculum/plasticity deficit. Note it is a *corrective* (task-boundary / periodic) intervention, complementary to *preventive* bases like LayerNorm+weight-decay (Lyle 2024, other shards).

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Guitar-learning parable of primacy bias; cognitive-science origin. Central finding: deep RL overfits early interactions; compounding vicious circle. Standard components (replay, high replay ratio) magnify it. Remedy preview: periodically re-init last few layers, keep the buffer. Four contributions: demonstrate, expose causes, propose resets, evaluate.
- **§2 Preliminaries.** MDP, $Q^\pi$, TD learning, replay buffer, replay ratio (too low = sample-inefficient, too high = overfit), n-step targets bias-variance trade-off.
- **§3 The Primacy Bias.** Definition. §3.1 Heavy priming: SAC + $10^5$ updates on first 100 transitions ⇒ unrecoverable (Fig. 1). §3.2 Primed agent's buffer is sufficient: fresh agent + failed agent's buffer learns fast (Fig. 2) ⇒ failure is in the learner, not the data.
- **§4 Have You Tried Resetting It?** Statement of the reset technique (re-init last few layers, keep buffer).
- **§5 Experiments.** §5.1 Setup: SPR/Atari 100k, SAC & DrQ/DMC; per-algorithm reset schedules; buffer preserved; IQM evaluation (Agarwal 2021). §5.2 Consistent gains (Tables 1–2). §5.3 Learning dynamics: fast post-reset recovery (Fig. 4); buffer-as-world-model + representation-recovery explanations; resets as regularization beating L2/dropout. §5.4 Elements behind success: replay-ratio interaction (Fig. 5), n-step interaction (Fig. 6), TD failure modes (Fig. 7), what/how-to-reset ablations.

# Phase 3 — Loss of Plasticity: Mechanisms and Fixes (2023–2024)

**Named and cured.** By 2023–2024 the forward failure had matured into a subfield with a mechanistic account and a menu of fixes. Entries 12–15 demonstrate the failure at scale and repair its micro-cause — activation and dormancy collapse — with CReLU, ReDo, plasticity injection, and resets-promoted-to-a-scaling-lever. Entries 16–18 supply the loss-landscape and empirical-NTK theory, a three-mechanism decomposition, and the *Nature* capstone (continual backprop). The corpus's central live disagreement — layer-normalization as cure (Lyle) vs. normalization as harm (Dohare) — lives here; it is adjudicated in the Cross-Paper Synthesis.

---


## 12. Abbas et al. 2023 — Loss of Plasticity in Continual Deep Reinforcement Learning

**PDF:** `docs/project/references/continual_learning/sources/Abbas et al. 2023 - Loss of Plasticity in Continual Deep RL.pdf`
**Venue:** 2nd Conference on Lifelong Learning Agents (CoLLAs) 2023, PMLR 232:620–636 · **arXiv:** 2303.07507
**Authors:** Zaheer Abbas, Rosie Zhao, Joseph Modayil, Adam White, Marlos C. Machado (DeepMind / Harvard / U. Alberta / Amii)

### <a id="abbas-phase-1"></a>Phase 1 — Foundational Overview (undergraduate level)

**The plain-language question.** If you take one neural-network game-player and make it play a *sequence* of Atari games — game A for a while, then game B, then C, … and eventually back to A — does it keep getting better at learning, or does it slowly go dull? This paper's answer: it goes dull. The agent learns game A fine the first time, but every time it returns to a game it has seen before, it learns *more slowly* and reaches a *worse* final score than the last visit — sometimes collapsing to no learning at all. The network has not run out of memory; it has lost the *ability to learn*. This is "loss of plasticity."

**The setup in one paragraph.** The authors take **Rainbow** — a strong, standard value-based deep-RL agent (a heavily-tuned descendant of Deep Q-Networks / DQN) — and run it in a new benchmark they call **Switching-ALE (S-ALE)**: the Arcade Learning Environment, but the agent cycles through a fixed list of games (e.g. Alien → Atlantis → Boxing → Breakout → Centipede → back to Alien), spending 20 million frames on each before switching, *never resetting* its weights or its replay buffer in between. Some runs go up to 2 billion frames over ~50 days of wall-clock time. They compare against an idealized **reset agent** — an agent imagined to be wiped clean before each game, so it always relearns from scratch and always recovers to the same first-visit level. A *good* continual learner should beat the reset agent (it has extra experience). Rainbow instead does *worse* than the reset agent over time.

**Key findings.**
1. **The phenomenon is real and severe.** Across 5-game and 10-game cycles, and across 10M / 20M / 50M frames-per-visit, and for both Rainbow and plain DQN, revisit performance degrades — the "dotted line" of first-visit performance is a ceiling later visits fall below.
2. **A clean forensic chain for *why*.** As visits accumulate, the authors watch four internal signals and find a causal cascade: **activations collapse** (fewer and fewer ReLU units fire — eventually <1% are active) → **gradients collapse** (a dead ReLU passes zero gradient to its incoming weights) → **weights stop changing** (on the 10th visit, weight change is ~20% of the first visit) → **learning stalls even though the training loss stays large**. A learner whose weights barely move despite a big loss has, by definition, lost plasticity.
3. **A remarkably simple fix: change the activation function.** Swapping ReLU for **Concatenated ReLU (CReLU)** — which outputs `[ReLU(x), ReLU(−x)]`, guaranteeing at least one of the two channels is non-zero for any non-zero input — prevents the activation collapse and *maintains* plasticity across all games. Rainbow-CReLU matches or exceeds the reset agent on repeated visits.
4. **An honest caveat.** CReLU fixes *plasticity* (the forward problem: can it still learn?) but does **not** fix *catastrophic forgetting* (the backward problem: does it remember old games?). The agent still forgets each game between visits. The two failures are distinct and both must be solved.

**Initial takeaway.** This is the closest published precedent to the project's own curriculum setup: a single network carried across a sequence of RL tasks with no reset, degrading below a from-scratch baseline. Abbas provides the vocabulary (activation footprint sparsification, gradient collapse) and the single cheapest preventive lever (CReLU) — a *family (c)* architecture fix in the primer's taxonomy (primer §4c). It also draws the sharp line the project must respect: fixing plasticity is not the same as fixing forgetting.

### <a id="abbas-phase-2"></a>Phase 2 — Graduate-Level Deep Dive

**Problem formalization.** The agent maximizes discounted return $G_t = \sum_{i=0}^{\infty}\gamma^i R_{t+1+i}$, $\gamma\in[0,1)$, via a value function $\hat q_w(s,a)\approx \mathbb{E}[G_t\mid S_t=s, A_t=a]$ parameterized by network weights $w$. DQN's update follows the gradient of the temporal-difference (TD) squared error against a periodically-frozen target network $\bar w$:

$$
\nabla_w\, \mathbb{E}\Big[\big(R_{t+1} + \gamma\max_{a\in\mathcal A}\hat q_{\bar w}(S_{t+1},a) - \hat q_w(S_t,A_t)\big)^2\Big].
$$

Rainbow additionally uses distributional RL, noisy nets, prioritized replay, dueling networks (a value head $\hat v_{w_1}(s)$ + advantage head $\hat d_{w_2}(s,a)$ with $\hat v_{w_1}(s)+\hat d_{w_2}(s,a)\approx \mathbb{E}[G_t\mid s,a]$), and n-step returns. The shared convolutional stack feeds both heads.

**The four-signal forensic analysis (the core technical contribution).** All statistics are measured at the *halfway point* of each visit and normalized by the first-visit value so cross-layer, cross-visit comparison is meaningful (layer weight magnitudes are not scale-invariant; norms are aggregated across layers weighted by parameter count).

1. **Weight change.** Let $\Delta w^{(v)}$ be the $\ell_2$ norm of the weight change over the first 10M frames of visit $v$. Empirically the *normalized* $\Delta w^{(v)}/\Delta w^{(1)}$ falls to ≈ 0.20 by $v=10$ in S-ALE, while the from-scratch baseline stays at ≈ 0.75. Weights freeze.

2. **Loss.** The distributional loss is *not* small — averaged over 100 mini-batches at the halfway point, it *grows* across visits. This is the crux: large loss with frozen weights is the operational signature of lost plasticity (a healthy optimizer would move weights to reduce a large loss).

3. **Gradient collapse.** The mechanism connecting (2) and (1). At each visit's halfway point the authors compute layer-wise gradient norms over the next 100 updates in three flavors: $\ell_0$ (count of non-zero gradient entries), $\ell_1$, and $\ell_2$. The $\ell_0$ and $\ell_1$ norms of the continual agent's gradients **decay to near zero**, far faster than the scratch baseline. The $\ell_2$ decay is milder — expected, because squaring makes $\ell_2$ dominated by the few surviving large-magnitude outliers. Crucially, this is the *raw* gradient before the optimizer scales it; and since Adam tracks exponentially-decaying averages of $g$ and $g^2$, if $g\to0$ then Adam's rescaling cannot resurrect it — the collapse is not an optimizer artifact.

4. **Activation collapse (the root cause).** The output of the convolution stack is 3136 units; each of the value and advantage heads has one hidden layer of 512 units. They measure the $\ell_0$ norm of each layer's activations (fraction of units producing non-zero output), averaged over 100 mini-batches of size 32. In the value/advantage heads this **collapses to <1%** of units active. The chain-rule explanation: for a ReLU unit with pre-activation $z$, $\text{ReLU}(z)=\max(0,z)$ and $\frac{\partial \text{ReLU}}{\partial z}=\mathbb{1}[z>0]$. A dead unit ($z\le 0$) contributes **exactly zero** gradient to its *incoming* weights via the chain rule, so those weights cannot update. The convolutional activations themselves do *not* collapse, but they depend on gradients flowing *back* from the collapsed heads — so the conv layers are progressively starved of learning signal too.

**The full causal cascade** is therefore: non-stationary targets in S-ALE → progressive sparsification of the activation footprint (units die and stay dead) → gradient $\ell_0/\ell_1$ collapse → weight change collapse → learning stalls despite large loss → performance degrades on revisits.

**The CReLU mitigation — mechanism and derivation.** CReLU (Shang et al. 2016) is defined as

$$
\text{CReLU}(x) = \big[\,\text{ReLU}(x),\; \text{ReLU}(-x)\,\big],
$$

i.e. it concatenates the input with its negation and rectifies both. **Key invariant:** for any scalar input $x\ne 0$, exactly one of $\text{ReLU}(x)$, $\text{ReLU}(-x)$ is strictly positive (if $x>0$ the first, if $x<0$ the second; both zero only at the measure-zero point $x=0$, essentially never hit in 32-bit float). Therefore CReLU *cannot* have a fully-dead unit for a non-zero pre-activation — it structurally guarantees a live gradient path. This directly attacks the root of the cascade (activation collapse) rather than its symptoms.

**Capacity bookkeeping.** CReLU doubles the number of *activations* per layer while keeping the *parameters producing* those activations fixed. The doubling can double the next layer's parameter count. To isolate CReLU's plasticity effect from a mere capacity increase, the authors test two controls:
- **Invariant input dimension**: fix the number of signals *entering* the activation → CReLU doubles parameters in every layer except the first.
- **Invariant output dimension**: fix the number of signals *leaving* the activation → CReLU *halves* parameters in every layer except the last.
Both variants maintain plasticity, so the benefit is not a capacity artifact. Confirming this, Figure 6 shows CReLU and ReLU perform *comparably* on standard single-game (non-continual) ALE — CReLU is not simply a "better activation," it is specifically a plasticity-preserving one under non-stationarity.

**CReLU internals mirror-image the pathology.** Re-running the four-signal analysis on Rainbow-CReLU (Figure 8): ~half the activations stay non-zero; relative gradient $\ell_0$ does not diminish and $\ell_1$ decays much more slowly than scratch (and the decay rate itself shrinks over visits); relative weight change stays large all 10 visits — in fact *larger* than both Rainbow-ReLU and the scratch baseline, consistent with CReLU's larger observed loss still driving real updates.

**Dependence on task similarity (a nuance).** Cycling through *game modes* of a single game (milder non-stationarity) produced loss of plasticity in only 1 of 3 games — **Breakout**, and specifically in modes 4, 8, 20, 36 where the *dynamics* change most (extra catch/release/steer actions). In Freeway the agent recovered *faster* on revisits; in Space Invaders it held steady. Conclusion: loss of plasticity tracks the *degree of change in dynamics* between successive tasks, though this is hard to characterize a priori. **Project relevance:** a difficulty-ladder curriculum whose stages change the dynamics substantially is exactly the regime most at risk.

**The unresolved half — catastrophic forgetting.** §6 of the paper is explicit that CReLU addresses only plasticity. The stability–plasticity dilemma remains: on every revisit the CReLU agent still relearns from scratch (no retention). Comparing continual-CReLU to a *conventional* Rainbow trained 200M frames uninterrupted on each game, a *widening gulf* opens (except Atlantis/Boxing which are near-solvable in one visit) — the signature of catastrophic interference. The paper's closing prescription: real continual learning needs *both* a plasticity fix *and* a forgetting fix. Future-work leads it names — shrink-and-perturb (Ash & Adams), plasticity injection (Nikishin 2023, [reviewed below](#14-nikishin-et-al-2023--deep-rl-with-plasticity-injection)), utility-based reinitialization (Dohare; Sokar/ReDo, [reviewed below](#13-sokar-et-al-2023--the-dormant-neuron-phenomenon-in-deep-rl-redo)) — are precisely the other shard-3 papers.

### <a id="abbas-appendix-section-by-section-backbone"></a>Appendix: Section-by-Section Backbone

- **Abstract.** Characterizes value-based deep RL under varying non-stationarity; agents lose the ability to learn good policies when cycling through Atari games. Ties the phenomenon to prior aliases (loss of plasticity, implicit under-parameterization, primacy bias, capacity loss). Analyzes weights/gradients/activations at scale; activation footprint becomes sparser → diminishing gradients. Proposes CReLU as a simple mitigation.
- **§1 Introduction.** Most deployed RL seeks a *fixed* policy (offline/simulator search) — appropriate when the policy never needs to change. But many settings (cooling controllers, sensor drift, changing weather) demand *continual adaptation*. Deep RL borrows i.i.d.-assuming supervised-learning machinery (replay buffers approximate stationarity) but RL is intrinsically non-stationary. Reviews prior threads: warm-starting hurts (Ash & Adams 2020; Berariu 2021); capacity/rank loss from changing value targets (Kumar 2021; Lyle 2022); primacy bias (Nikishin 2022). Contribution: demonstrate catastrophic loss of plasticity in a *performant* system (Rainbow) on a continual ALE variant, up to 2B frames / 50 days; extensive weight/gradient/activation analysis; evaluate CReLU.
- **§2 Preliminaries.** MDP + return $G_t$; DQN value function $\hat q_w$; the TD target-network update; Rainbow's components (distributional RL, noisy nets, prioritized replay, dueling value+advantage heads, n-step returns); conv stack shared by both heads. ALE evaluation protocol (Machado 2018 recommendations: stochasticity, ignore lives, report training-average performance). Divergence from standard practice: one network trained on a *sequence* of games, not uniformly-mixed batches.
- **§3 Demonstrating loss of plasticity.**
  - *§3.1 Adapting the ALE for continual learning (S-ALE).* Cycle a fixed game sequence, 20M frames/visit, no weight/buffer reset between switches. Define *visit*; e.g. 1B frames, 10 games, 20M/visit → 5 visits/game, 180M frames between successive plays — far exceeding the 1M-frame replay capacity. Contrasted with Jelly Bean World, Continual World.
  - *§3.2 Learning performance in S-ALE.* Per-game plots (reward scales differ). First visit good; successive visits slower and worse, sometimes collapsing. Introduces the idealized **reset agent** (forgets everything, always relearns to the same level) as the bar a real continual learner must beat; Rainbow falls below it. Hypothesizes *both* catastrophic interference *and* loss of plasticity. Robustness across 5/10 games and 10M/20M/50M frames-per-visit; same for DQN (appendix).
  - *§3.3 Varying non-stationarity via game modes.* Milder non-stationarity. Loss of plasticity in Breakout (modes 4/8/20/36, biggest dynamics changes) but not Freeway (recovers faster) or Space Invaders (holds). Plasticity loss correlates with dynamics change.
- **§4 Characterizing loss of plasticity.** Summary of the forensic chain, then per-signal detail: §4.1 weight change diminishes (20% by visit 10 vs. 75% scratch) despite growing loss; §4.2 gradient collapse ($\ell_0/\ell_1$ decay to ~0, $\ell_2$ milder; raw pre-optimizer gradients; Adam cannot rescue $g\to0$); §4.3 activation collapse (<1% of value/advantage units active; chain-rule dead-ReLU explanation; conv layers starved of back-flowing gradient).
- **§5 Mitigating loss of plasticity with CReLU.** Definition $\text{CReLU}(x)=[\text{ReLU}(x),\text{ReLU}(-x)]$; non-zero-guarantee invariant. Two capacity controls (invariant input dim → 2× params; invariant output dim → ½ params); both maintain plasticity. CReLU≈ReLU on non-continual ALE (not a general superiority). Rainbow-CReLU internal statistics (Fig. 8) confirm restored activations/gradients/weight change. Same result on Breakout game modes.
- **§6 Catastrophic forgetting: an unresolved challenge.** CReLU does not address forgetting; agent still relearns each revisit; widening gulf vs. uninterrupted 200M-frame per-game Rainbow = catastrophic interference. Path forward: solve both halves of the stability–plasticity dilemma.
- **§7 Conclusions & future work.** Canonical value-based deep RL (DQN, Rainbow) fails at continual learning; activation collapse inhibits adaptation; CReLU mitigates. Future leads: Leaky-ReLU / SeLU alternatives; shrink-and-perturb; plasticity injection; utility-based reinitialization (Dohare, Sokar). Activation change won't fix interference.

---

## 13. Sokar et al. 2023 — The Dormant Neuron Phenomenon in Deep RL (ReDo)

**PDF:** `docs/project/references/continual_learning/sources/Sokar et al. 2023 - The Dormant Neuron Phenomenon (ReDo).pdf`
**Venue:** ICML 2023, PMLR 202 · **arXiv:** 2302.12902
**Authors:** Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci (Eindhoven / Google DeepMind / Mila)

### <a id="redo-phase-1"></a>Phase 1 — Foundational Overview (undergraduate level)

**The plain-language question.** Abbas showed that units in an RL network go silent over training. Sokar asks: what *exactly* is the atomic unit of that decay, can we measure it precisely, and can we reverse it *without* the drastic step of wiping the whole network? Their answer names a concrete object — the **dormant neuron** — and offers a surgical fix, **ReDo** ("Recycle Dormant neurons").

**What a dormant neuron is.** A neuron is *dormant* if its activation, averaged over inputs and normalized against the other neurons in its layer, is essentially zero — it has stopped contributing to the network's output. The paper's central empirical fact, the **dormant neuron phenomenon**: as an RL agent trains, the *number* of dormant neurons steadily *grows*, and once a neuron goes dormant it tends to *stay* dormant. The network is quietly shrinking its own usable size even though it is nominally over-parameterized.

**Three diagnostic findings.**
1. **It's caused by moving targets, not moving data.** RL has two kinds of non-stationarity: the *input* distribution shifts (the agent's own policy changes what it sees) and the *target* shifts (the network bootstraps off its own changing estimate). Using controlled CIFAR-10 and offline-RL experiments, Sokar shows the *target* non-stationarity is the primary culprit — dormancy grows when learning targets keep moving, and barely grows with fixed targets, even when the input data is fixed.
2. **More gradient updates → more dormant neurons.** Cranking up the **replay ratio** (updates per environment step) increases dormancy — which explains why naively training harder on the same data collapses performance. This is the link to D'Oro's replay-ratio work.
3. **Dormancy directly damages future learning.** A pre-trained network full of dormant neurons is measurably *worse* than a fresh random network at fitting a new target — dormancy is not cosmetic, it degrades the learner.

**The fix — ReDo.** Every so often during training, scan every layer; any neuron below a dormancy threshold $\tau$ gets its *incoming* weights re-randomized and its *outgoing* weights zeroed. Zeroing the outgoing weights means the recycled neuron initially changes the network's output *not at all* (a "do no harm" property) — but its incoming weights are now fresh, so it can start learning again. Result: dormancy stays low, network capacity is maintained, and performance improves — especially at high replay ratios, where ReDo *avoids the performance collapse* that normally caps how hard you can train.

**Initial takeaway.** ReDo is the *surgical* member of the primer's "reset / recycle capacity" family (primer §4a): rather than resetting whole layers (Nikishin 2022) or perpetually reinitializing the least-used units (Dohare's continual backprop), it recycles *only* the units that have actually gone dormant, and does so without disturbing the current output. Directly relevant to the project's replay-ratio study: ReDo is a concrete lever for pushing the replay ratio up without collapse.

### <a id="redo-phase-2"></a>Phase 2 — Graduate-Level Deep Dive

**Formal definition of a dormant neuron.** Given an input distribution $D$, let $h_i^\ell(x)$ be the activation of neuron $i$ in layer $\ell$ under input $x\in D$, and $H^\ell$ the number of neurons in layer $\ell$. The neuron's **score** is its expected absolute activation, normalized by the layer's average:

$$
s_i^\ell \;=\; \frac{\mathbb{E}_{x\in D}\,\lvert h_i^\ell(x)\rvert}{\tfrac{1}{H^\ell}\sum_{k\in h}\mathbb{E}_{x\in D}\,\lvert h_k^\ell(x)\rvert}.
$$

A neuron is **$\tau$-dormant** if $s_i^\ell \le \tau$. The denominator normalizes scores to sum (in expectation) to a constant within a layer, making neurons *across layers of different width* comparable — a key design choice, since a raw activation magnitude means different things in a 512-wide vs. 3136-wide layer.

**Definition of the phenomenon.** An algorithm *exhibits the dormant neuron phenomenon* if the count of $\tau$-dormant neurons increases steadily throughout training. Such a network under-utilizes its capacity, and the under-utilization worsens over time. (Early analyses use the strict $\tau=0$; benchmarking loosens to $\tau=0.1$.)

**Why $\tau$-dormancy matters even for small $\tau$.** Low-activation neurons could in principle still shape the learned function, but their contribution — and the disruption from recycling them — is bounded by their small activation magnitude. So recycling a $\tau$-dormant neuron with small $\tau$ perturbs the output only slightly; with $\tau=0$, the output is left *exactly* unchanged.

**Evidence chain for the cause (target non-stationarity):**
- *Baseline phenomenon.* DQN dormant-neuron fraction rises steadily across gradient steps (DemonAttack, Asterix; Fig. 2), consistent across algorithms (DrQ($\epsilon$), SAC) and domains (Atari, MuJoCo).
- *Supervised control (Fig. 3).* CIFAR-10 with **fixed** labels → dormancy *decreases* over time; with **shuffled/non-stationary** labels → dormancy *increases*, with sharp jumps exactly at label-shuffle points. Isolates target non-stationarity.
- *Offline-RL control (Fig. 4).* Fixed dataset (removes *input* non-stationarity) but standard moving TD targets → phenomenon persists. Ablating to fixed *random* targets → dormancy drops. Therefore **target** non-stationarity (bootstrapping off a moving estimate) is primary; *input* non-stationarity is not a major factor.
- *Persistence (Figs. 5–6).* The overlap coefficient $\text{overlap}(X,Y)=\frac{\lvert X\cap Y\rvert}{\min(\lvert X\rvert,\lvert Y\rvert)}$ between the current and historical dormant sets *rises* → dormant neurons rarely reactivate. Explicitly **pruning** all-time-dormant neurons does **not** hurt performance → confirms they are functionally inert.
- *Replay-ratio dependence (Fig. 7).* Higher replay ratio (RR ∈ {0.25, 0.5, 1, 2}) → strictly more dormant neurons, correlating with the known performance drop at high RR. This is the mechanistic bridge to [D'Oro's replay-ratio barrier](#15-doro-et-al-2023--sample-efficient-rl-by-breaking-the-replay-ratio-barrier).
- *Causal harm (Fig. 8).* Distilling a dormant-heavy pre-trained DQN toward a well-performing target *degrades* over training and its dormancy keeps climbing, while a randomly-initialized network improves and stays stable-dormancy. Dormancy is a *cause* of impaired new-task learning, not a mere correlate.

**The ReDo algorithm (Algorithm 1).**

```
Input: parameters θ, threshold τ, training steps T, frequency F
for t = 1 to T:
    Update θ with the regular RL loss
    if t mod F == 0:
        for each neuron i:
            if s_i^ℓ ≤ τ:
                Reinitialize incoming weights of neuron i   (from the original init distribution)
                Set outgoing weights of neuron i to 0
```

**Why the two-sided reinit is the right design.** Reinitializing *incoming* weights gives the neuron a fresh, non-degenerate learning direction. Zeroing *outgoing* weights guarantees the recycled neuron's *immediate* contribution to the next layer is zero, so at the moment of recycling the network's function is (for $\tau=0$) unchanged and (for small $\tau$) barely perturbed — ReDo restores plasticity *without* the abrupt performance drop and re-exploration that a full-layer reset (Nikishin 2022) causes. Ablations (App. C.2): scaling incoming weights by the mean non-dormant norm ≈ same as using the init distribution; randomizing outgoing weights ≈ same or worse. So the simple recipe is not fragile.

**Is it a ReLU-specific problem?** RL nets typically use ReLU, which saturates at zero output (zero gradient). App. C.1 measures dormancy under a *different* activation and finds a *mild* decrease but the phenomenon persists — so dormancy is not purely a ReLU artifact (contrast Abbas, who fixes it *at the activation level* with CReLU; ReDo instead fixes it *at the weight level* with recycling — the two are complementary members of families (c) and (a)).

**Empirical results.**
- **Setup.** DQN on 17 ALE games (default CNN + IMPALA ResNet); DrQ($\epsilon$) on the 26-game Atari 100k; SAC on 4 MuJoCo tasks. Dopamine framework. Default $\tau=0.1$ (beat $\tau=0$ and $\tau=0.025$). Metric: Interquartile Mean (IQM, Agarwal 2021) with 95% stratified-bootstrap CIs, 5–10 seeds.
- **§5.1 Sample efficiency.** Across RR ∈ {0.25, 0.5, 1, 2} (DQN default 0.25), ReDo *avoids the performance collapse* at high RR and even *benefits* from higher RR. Holds with n-step=3 returns, with the ResNet architecture, and for DrQ($\epsilon$) at RR ∈ {1,2,4,8} on Atari 100k.
- **§5.2 Learning-rate scaling.** A reduced LR at high RR partially mitigates but does not match ReDo — ReDo is not just "the LR was too high."
- **Headline (Fig. 1).** On 17 Atari games at RR=1, DQN+ReDo beats DQN, DQN+Reset (Nikishin 2022), and DQN+WeightDecay in IQM human-normalized score.

**Relation to the primer & sibling papers.** ReDo operationalizes the "dormant unit" that Abbas observed as a *fraction* into a per-neuron, thresholded, recyclable object; it targets the *dormant count* directly (primer §4a). Versus Nikishin 2022 resets: more surgical, output-preserving, buffer-independent. Versus Dohare continual-backprop: ReDo triggers on *dormancy* ($s_i^\ell\le\tau$), continual-backprop on *low utility* — different selection criteria for the same "recycle the least-useful units" idea. The replay-ratio finding is the direct handshake to D'Oro.

### <a id="redo-appendix-section-by-section-backbone"></a>Appendix: Section-by-Section Backbone

- **Abstract.** Identifies the dormant neuron phenomenon (growing count of inactive neurons hurting expressivity) across algorithms/environments; proposes ReDo to recycle dormant neurons, maintaining expressivity and improving performance.
- **§1 Introduction.** Deep nets as RL function approximators enable scaling but bring RL-specific training pathologies. Scaling laws in supervised learning: performance ↑ with parameters; but in RL, networks *lose* expressivity/target-fitting ability despite over-parameterization (Kumar 2021; Lyle 2021), partly mitigated by perturbing/resetting params (Igl 2020; Nikishin 2022) — but resets are drastic (forget + slow recovery). Central question: *do RL agents use their parameters to full potential?* Track dormant neurons → they grow with training (unlike supervised learning). Contributions: demonstrate the phenomenon; investigate causes + negative effect; propose ReDo; show effectiveness.
- **§2 Background.** MDP $\langle S,A,R,P,\gamma\rangle$; $Q^\pi$, $Q^*$; deep $Q_\theta$; TD loss $L_\theta = Q_\theta(s,a)-Q_\theta^T(s,a)$ with bootstrap target $Q^T(s,a)=[R(s,a)+\gamma\max_{a'}Q_{\tilde\theta}(s',a')]$ and target net $Q_{\tilde\theta}$. **Replay ratio** = gradient updates per env step; higher RR → sample efficiency but training instability/collapse (Nikishin 2022). Two non-stationarities: **input** (online data collection under changing $\pi$) and **target** (bootstrapping off changing $Q_{\tilde\theta}$).
- **§3 The Dormant Neuron Phenomenon.** Def. 3.1 (score $s_i^\ell$, $\tau$-dormant); Def. 3.2 (phenomenon = steadily growing $\tau$-dormant count). Evidence: present in DQN (Fig. 2); **target non-stationarity exacerbates** (CIFAR fixed vs shuffled, Fig. 3); **input non-stationarity not major** (offline RL still shows it, Fig. 4); **dormant neurons remain dormant** (overlap coefficient ↑, Fig. 5; pruning them harmless, Fig. 6); **more updates → more dormant** (RR sweep, Fig. 7); **dormancy makes new-task learning harder** (distillation experiment, Fig. 8).
- **§4 Recycling Dormant Neurons (ReDo).** Algorithm 1 (periodic check, reinit incoming, zero outgoing; $\tau=0$ leaves output unchanged, small $\tau$ slightly changed). Design discussion: alternate recycling strategies (mean-norm scaling ≈ init distribution); alternate init (random outgoing ≈ worse); "Are ReLUs to blame?" — different activation shows mild decrease but phenomenon persists.
- **§5 Empirical Evaluations.** Agents/architectures/environments (DQN 17 games CNN+ResNet; DrQ($\epsilon$) Atari 100k; SAC MuJoCo); Dopamine; $\tau=0.1$; IQM + 95% CIs. §5.1 Consequences for sample efficiency (RR sweep, avoids collapse, benefits from high RR; n-step; ResNet; DrQ). §5.2 Learning-rate scaling (low-LR partial, ReDo better). (Further: related-methods comparison, ablations in appendices.)
- **Appendices.** A: CIFAR non-stationary-target details. B: phenomenon in DrQ($\epsilon$), SAC. C.1: activation-function ablation. C.2: recycling/init-strategy ablations.

---

## 14. Nikishin et al. 2023 — Deep RL with Plasticity Injection

**PDF:** `docs/project/references/continual_learning/sources/Nikishin et al. 2023 - Deep RL with Plasticity Injection.pdf`
**Venue:** NeurIPS 2023 (also ICLR 2023 Reincarnating RL Workshop) · **arXiv:** 2305.15555
**Authors:** Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, André Barreto (DeepMind)

### <a id="inject-phase-1"></a>Phase 1 — Foundational Overview (undergraduate level)

**The plain-language question.** Suppose an RL agent has plateaued — it stopped improving. *Why?* It could be that its network lost plasticity (can't learn anymore), **or** it could be that it simply can't *explore* well enough to find better behavior. These two causes look identical from the outside (flat learning curve) but need opposite fixes. This paper introduces a single minimal trick — **plasticity injection** — that both (a) *diagnoses* which cause it is, and (b) if it's plasticity, *fixes* it, cheaply.

**The trick, intuitively.** At any moment you can *freeze* the current network (so it stops learning but keeps its knowledge) and bolt on a *fresh, randomly-initialized* network whose job is to learn a *correction* to the frozen network's outputs. It's engineered so that at the instant you attach it, the correction is exactly **zero** — the agent's predictions and behavior are completely unchanged. But now there are *fresh* trainable weights, full of plasticity, ready to keep improving. Critically, the total number of *trainable* parameters is kept the same (the old head is frozen), and the predictions are not disturbed — so any change in performance afterward is attributable *only* to the added plasticity, with exploration and capacity held constant.

**As a diagnostic.** Take a plateaued agent, inject plasticity, and compare the training curves with vs. without the injection:
- If performance **jumps** → the agent *was* plasticity-limited (Phoenix: injection doubles the final return; Space Invaders: post-injection learns faster).
- If performance **doesn't budge** → the plateau is *not* about plasticity — it's exploration (Assault: stuck because a new action becomes necessary at score ~2800) or the agent is simply healthy (Robotank: no pathology). Varying *when* you inject even pinpoints *when* plasticity was lost (Phoenix ~25M frames; Space Invaders ~100M frames).

**As a practical tool.** It also saves compute. You can start training with a *small* network (cheap) and, partway through, inject plasticity to effectively *grow* into a larger network — reaching the same score as training the big network from the start, but saving ~20 GPU-hours because the small net was used for the first 50M frames. It also "reincarnates" agents: improve a long-trained agent that's out of plasticity *without* retraining from scratch — beating shrink-and-perturb, resets, and naive width-scaling on aggregate Atari score.

**Initial takeaway.** Plasticity injection is the primer's family-(a) member that *adds* fresh capacity rather than *recycling* old (primer §4a) — a residual/boosting-flavored intervention. Its distinctive value to the project is the **clean diagnostic protocol**: it disentangles "can't learn" from "can't explore," a confound that plagues any interpretation of a flat RL curve — exactly the ambiguity the project's curriculum plateau raises.

### <a id="inject-phase-2"></a>Phase 2 — Graduate-Level Deep Dive

**Design desiderata (what makes it a clean intervention).** Two properties are demanded up front:
1. **Unaffected predictions** — the agent's outputs must be identical immediately after injection, so exploration is not perturbed and cannot confound the analysis.
2. **Preserved trainable-parameter count** — so representational *capacity* (in the trainable sense) is held fixed and cannot confound.

**The construction.** Let the approximator be $h_\theta(x)$ (e.g. an action-value head). At the injection moment, freeze $\theta$ and introduce a fresh random init $\theta'$, kept in **two copies** $\theta_1'$ (trainable) and $\theta_2'$ (permanently frozen), with $\theta_1'=\theta_2'$ at injection. The post-injection prediction is:

$$
\underbrace{h_\theta(x)}_{\text{frozen}} \;+\; \underbrace{h_{\theta_1'}(x)}_{\text{trained}} \;-\; \underbrace{h_{\theta_2'}(x)}_{\text{frozen}}. \tag{1}
$$

**Two properties by construction, derived:**
- **Zero initial change.** At $t=$ injection, $\theta_1'=\theta_2'\Rightarrow h_{\theta_1'}(x)=h_{\theta_2'}(x)$, so the last two terms cancel and (1) reduces to $h_\theta(x)$ — the *exact* pre-injection prediction. No abrupt jump, no induced exploration. (Contrast a hard reset, which *does* jump.)
- **Constant trainable count.** Only $\theta_1'$ is trainable; $\theta$ and $\theta_2'$ are frozen. If $\theta_1'$ has the same shape as the frozen head, the number of *trainable* params is unchanged (total params rise, which costs memory/time but not trainable capacity).

As training proceeds, $\theta_1'$ drifts from $\theta_2'$, and $h_\theta(x)-h_{\theta_2'}(x)$ becomes a *constant bias term* (both frozen) added to the freshly-learning $h_{\theta_1'}(x)$. So the trainable network $h_{\theta_1'}$ learns a *residual* on top of a fixed offset — a residual-learning / boosting view the authors make explicit.

**Applying to only part of the network.** Injecting into *all* parameters would force relearning the entire representation from scratch. Instead, split the net into an encoder $\phi(\cdot)$ (first $k$ layers) and a head $h_\theta(\cdot)$ (remaining layers), and apply (1) only to the head. The encoder $\phi$ is *shared* across the three heads and keeps learning — importantly, gradients from the *frozen* heads are **not** stopped; they still flow into $\phi$, so the encoder is refined by all three output paths. (In experiments: 5-layer conv net, $k=3$ encoder, last 2 layers = head. Target network gets the same intervention.)

**Why not just reset (Nikishin 2022)?** A hard reset of the head abruptly changes predictions → temporary performance drop + induced exploration effect. Analytically that abruptness makes it *impossible* to isolate the plasticity effect from the exploration effect; practically, reset relies on the *replay buffer* to relearn, whereas injection does *not* need the buffer (§5.3 shows this can be decisive).

**Diagnostic protocol (the counterfactual).** For a suboptimal/plateaued agent: save a checkpoint, inject plasticity, and compare curves with vs. without injection — answering the counterfactual *"what would performance be if the network had more plasticity?"* Four canonical outcomes (Fig. 3):
- **Phoenix** — flat baseline; injection *doubles* final return → **catastrophic plasticity loss** (extra interactions weren't translating to learning). Earlier injection helps → plasticity lost ~25M frames.
- **Space Invaders** — baseline still learning; injection accelerates late learning → *gradual* plasticity decline; injection timing doesn't matter until ~100M → onset ~100M.
- **Assault** — plateau *not* fixed by injection → **exploration** limit (a new action becomes necessary at ~2800; App. D).
- **Robotank** — healthy; injection does nothing → no pathology.
The argument is careful/nuanced: plasticity is broadly defined and hard to measure, so "the post-injection agent learns further *because* plasticity was the bottleneck" is the *most likely* interpretation under a design built to hold other factors fixed — not a proof.

**What controls the degree of plasticity loss (Fig. 5).** Measuring the IQM improvement from a 50M-frame injection across regimes, the effect size:
- **increases monotonically with replay ratio (RR)** — more updates burn plasticity faster (consistent with Sokar's RR→dormancy finding);
- **increases monotonically with learning rate (LR)**;
- **decreases with network size** — bigger nets retain plasticity longer;
- **is smaller but still positive with spectral normalization (SN)** applied to the penultimate layer.
These double as *recommendations* for controlling plasticity loss: lower RR/LR, larger nets, normalization.

**Computational-efficiency applications (§5.3).**
- **Reincarnating RL.** Improve an already-trained-out agent without retraining. Across 57 Atari games, injection beats shrink-and-perturb (Ash & Adams), resets (Nikishin 2022), and naive width-scaling in aggregate IQM (paper reports ~20% improvement over other dynamic methods).
- **Dynamic growth to save compute.** Start with a small network; inject plasticity at 50M frames to grow (matching the parameter budget of $\phi + h_\theta + h_{\theta_1'}$). Reaches the same IQM as using the large net from the start while saving ~20 hours of A100 wall-clock, since the small net trains cheaply up to 50M and fewer params are updated afterward — supporting the hypothesis that a large net's *full* capacity isn't needed early, only later for plasticity.

**Illustration of plasticity loss (§3).** A didactic supervised sequence: train a Double-DQN agent on Up'n'Down 200M frames, snapshot policies every 10M, then for each policy build a Monte-Carlo value-regression task (states + MC value estimates) — a sequence of related prediction problems mimicking an online RL agent (Dabney 2021). *"Reset every task"* (random init each) fits every task; *"reset never"* (init from previous task's final params) takes **longer and longer** to fit each subsequent task — the *opposite* of the transfer-learning intuition that related pre-training accelerates. Clean demonstration that warm-starting *degrades* future learnability.

**Limitations.** Extra memory/training time; benefit varies a lot per game (aggregate positive on Atari, individually mixed); can't handle parameter *divergence* (another loss-of-plasticity mode) without more drastic tools; does *not* identify the *causal* factors driving plasticity loss — only diagnoses and mitigates.

**Relation to primer & siblings.** Family-(a) *additive* recycling; conceptually a simplified progressive network (Rusu 2016) motivated by *within-task* plasticity without prediction change; kin to residual learning, boosting, mixtures-of-experts, LoRA (frozen backbone + trainable low-rank addition). Whereas Sokar/ReDo *recycles dormant* units and Nikishin 2022 *resets* layers, injection *appends* fresh trainable capacity while freezing the old. It cites Abbas and Sokar as the "saturation of neurons" line, and notes Lyle 2023's caveat that saturation alone can't fully characterize plasticity loss.

### <a id="inject-appendix-section-by-section-backbone"></a>Appendix: Section-by-Section Backbone

- **Abstract.** Neural nets in deep RL gradually lose plasticity; analysis is hampered by the plasticity–exploration–performance confound. Introduces plasticity injection: increases plasticity without changing trainable-parameter count or biasing predictions. Two uses: (1) diagnostic — if injection helps, the agent was plasticity-limited (identifies plateau-causing Atari envs); (2) efficiency — grow the net dynamically or reincarnate without retraining. Stronger + more efficient than alternatives on Atari.
- **§1 Introduction.** Biological loss of plasticity (aging) has no reason to apply to artificial agents, yet RL agents lose learning ability (Dohare 2021, Lyle 2022, Nikishin 2022). Mechanisms poorly understood; performance confounded by exploration; proxy measures (saturated ReLUs, feature rank) may not capture it (Gulcehre 2022). Contributions: the intervention; complementary existence evidence; a diagnostic protocol; a dynamic-growth efficiency method.
- **§2 Related Work.** *Plasticity in continual learning* (McCloskey-Cohen stability–plasticity dilemma; French forgetting; Ash & Adams warm-start damage; Berariu gradient-noise-reduction conjecture; Dohare reduced train-error-minimization). *Loss of plasticity in deep RL* (Lyle capacity loss; Kumar implicit under-parameterization/rank; Gulcehre weak rank–performance correlation; Sokar/Abbas neuron saturation; Lyle 2023 saturation-insufficient; Nikishin 2022 primacy bias + resets; Igl distillation). *Architectures* (progressive nets Rusu 2016 = closest; MoE/modular nets; growing nets Fahlman-Lebiere, Net2Net; LoRA; residual learning; boosting).
- **§3 An Illustration of Plasticity Loss.** Up'n'Down policy-evaluation sequence; "reset never" fits progressively slower vs "reset every task"; opposite of transfer intuition. Notes RL's distinctive **exploration confounder** (agent shapes its own future data) motivating the injection design.
- **§4 Plasticity Injection.** Desiderata (unaffected predictions; preserved trainable count). Construction Eq. (1) with $\theta$, $\theta_1'$ (trained), $\theta_2'$ (frozen); zero-initial-change + residual-bias derivation. Encoder/head split ($\phi$, $k$ layers) to avoid full representation relearning; gradients flow from frozen heads into $\phi$. Contrast with resets (abrupt change, buffer dependence).
- **§5 Experiments.** §5.1 Setup (Double-DQN, 200M frames, 57 Atari; 5-layer conv, $k=3$; single injection @50M default; IQM, 3 seeds). §5.2 Diagnostic tool (Phoenix/Space Invaders/Assault/Robotank; injection-timing pinpoints onset; Fig. 4 across 57 games; Fig. 5 sensitivity to RR↑/LR↑/size↓/SN). §5.3 Computational efficiency (Reincarnating RL — beats SnP/resets/width-scaling; dynamic growth — matches larger net, saves ~20 A100-hours).
- **§6 Limitations.** Memory/time overhead; per-game variance; parameter divergence unaddressed; no causal identification of plasticity-loss drivers.
- **§7 Discussion & Conclusion.** A clean study of the phenomenon; the proposed version prioritizes simplicity over optimality (a "blueprint"); architecture-agnostic (ResNet blocks, Transformer decoder blocks); open questions — can plasticity loss be solved completely? which properties of fresh nets give high plasticity?

---

## 15. D'Oro et al. 2023 — Sample-Efficient RL by Breaking the Replay Ratio Barrier

**PDF:** `docs/project/references/continual_learning/sources/D'Oro et al. 2023 - Sample-Efficient RL by Breaking the Replay Ratio Barrier.pdf`
**Venue:** ICLR 2023 (Oral, top ~5%) · **OpenReview:** `OpC-9aBBVJe` (no arXiv version — see primer §6.5 note #5)
**Authors:** Pierluca D'Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc G. Bellemare, Aaron Courville (Mila / Google Brain)

### <a id="rr-phase-1"></a>Phase 1 — Foundational Overview (undergraduate level)

**The plain-language question.** In RL, every real interaction with the world can be expensive, so you want to squeeze as much learning as possible out of each collected experience. The obvious lever is the **replay ratio** (a.k.a. update-to-data / UTD ratio): how many gradient updates you do per environment step. Crank it up and you learn more from the same data. The problem: for standard algorithms, cranking it up *stops helping and then collapses* — the network loses its ability to learn (the same plasticity loss the other three papers study). This is the **replay-ratio barrier**. D'Oro's finding: the barrier is not fundamental — it's an artifact of *not* countering plasticity loss. Add **periodic resets** and the barrier lifts, letting you push the replay ratio up by an *order of magnitude* and win large sample-efficiency gains.

**The recipe.** Take a standard off-policy algorithm (SAC for continuous control, SPR for discrete Atari). Periodically **reset** the agent's networks — fully or partially — *while keeping the replay buffer intact* so the agent can quickly relearn from stored experience. Crucially, tie the reset schedule to the *number of updates*, not the number of environment steps: at higher replay ratio, the network is updated more, so it should be reset more often. The reset-augmented versions are **SR-SAC** (Scaled-by-Resetting SAC) and **SR-SPR**. With this, SR-SAC scales usefully out to replay ratio **128** and SR-SPR to **16** — where the un-reset baselines plateau or collapse far earlier.

**Headline results.**
- On the **DeepMind Control Suite** (15-task benchmark), SR-SAC at replay ratio 128 sets a new state-of-the-art for model-free continuous control, beating even REDQ (which used replay ratio 20) — with a *simpler* algorithm.
- On **Atari 100k** (sample-efficiency benchmark), SR-SPR at replay ratio 16 sets a new model-free state-of-the-art, rivaling methods that pretrained on extra data.

**What makes it work (and its limits).** The paper dissects *why* resets enable scaling and probes the role of online data. Key insight: the small trickle of *online* interaction acts as an **implicit regularizer** that prevents the policy from degenerating when trained aggressively offline on a fixed buffer. And there are hard limits: you can't extract more than the information already in the buffer; you can't overcome intrinsic hard-exploration/credit-assignment; keeping the whole history in the buffer costs storage; and the scaling is *inherently sequential* — more GPUs don't speed it up.

**Initial takeaway.** This is the paper that turns loss-of-plasticity *avoidance* into a *scaling knob*. It is the primer's family-(a) reset method promoted from "pathology fix" to "sample-efficiency lever" (primer §4a), and it is **directly relevant to the project's own replay-ratio speed-vs-performance study** ([`replay_ratio_speed_vs_performance`](../../concepts/replay_ratio_speed_vs_performance.md)). The load-bearing design detail for the project: **reset frequency must be defined in *updates*, not env steps** — a fixed env-step interval breaks above replay ratio 4.

### <a id="rr-phase-2"></a>Phase 2 — Graduate-Level Deep Dive

**Definition (given a name in the paper):**

> **Replay Ratio Scaling** — the change in an agent's performance caused by doing more updates for a fixed number of environment interactions.

The definition is deliberately *sign-neutral*: every algorithm has *some* replay-ratio-scaling behavior; the goal is *favorable* scaling (performance rises with RR). Related quantity: update-to-data (UTD) ratio. Most standard algorithms use RR ≈ 1.

**Why RR scaling is special (coupled to the online loop).** Unlike model-size scaling laws (Kaplan 2020), RR scaling is entangled with the online RL loop: more training between interactions → a *better data-collection policy* → *different* next samples → *different* future buffer contents → *different* future learning dynamics. So it cannot be understood as pure offline optimization; it is a property of the agent–environment interaction. The framing "what is deep RL if not a long sequence of related but distinct tasks (Dabney 2021)?" motivates why plasticity loss is the binding constraint: each RR increase = more training = more task-switch-like degradation (Ash & Adams; Berariu — the more training on a prior task, the worse the eventual new-task performance).

**The central mechanism.** The main factor inhibiting RR scaling is the *progressive loss of ability to learn and generalize*. It *can be restored* by periodically resetting parameters — partially (Ash & Adams shrink-and-perturb) or totally (Nikishin 2022). The pivotal design principle:

> Set the reset frequency to depend **only on the number of updates** (hence *implicitly* on the replay ratio), not on environment steps. Then "the more an algorithm updates its networks, the more frequent the restoration of its ability to learn."

**Continuous control — SR-SAC (§4.1).**
- **Benchmark:** DMC15 (15 DeepMind Control tasks chosen to be neither trivially solvable nor unsolvable); specialized to DMC15-500k ($5\times10^5$ interactions) and DMC15-1M ($10^6$).
- **Reset strategy:** completely reset *all* agent parameters every $2.56\times10^6$ updates. Because the schedule is in updates, higher RR ⇒ more frequent resets in env-step terms (e.g. at RR=128, a reset every 20,000 env steps).
- **Result:** SR-SAC at RR=128 beats SAC, DDPG, and REDQ (the prior SOTA at RR=20) at *every* interaction budget on DMC15 — new model-free SOTA, simpler algorithm. (Table 1: SR-SAC IQM 740 vs REDQ 511 vs SAC 391 on DMC15-500k.)

**Discrete control — SR-SPR (§4.2).** SPR = a sample-efficient DQN/Rainbow variant with a model-based latent-dynamics-prediction auxiliary objective. Getting robust RR scaling here needs more care:
- **Reset strategy:** one reset every 40,000 updates (at RR=16 → every 2,500 env steps ≈ every 3 min). Nikishin 2022 only reset a *subset* of params, leaving the conv encoder untouched (fully resetting the encoder is impractical). As an intermediate, D'Oro applies **soft resets** to the encoder via a Shrink-and-Perturb variant that *interpolates* each parameter between its current value and a fresh random init on each reset:

$$
\theta_t = \alpha\,\theta_{t-1} + (1-\alpha)\,\phi,\qquad \phi\sim\text{initializer},\quad \alpha=0.8\ \text{(default)}.
$$

  $\alpha$ interpolates smoothly between leaving a layer unchanged ($\alpha=1$) and fully resetting it ($\alpha=0$). *But*: SP alone on *all* parameters is **insufficient** — it is essential that at least the network's *final layers* be **completely** reset (Berariu 2021: plasticity loss concentrates in the last layers but affects all layers; SP on the encoder gives roughly a constant +0.04 IQM past RR=4).
- **Target networks:** SPR by default has *no* separate target net; D'Oro adds an EMA target ($\tau=0.005$) and — following speedy Q-learning (Ghavamzadeh 2011) — uses the target net for *action selection*. This is the **single most important factor** allowing SR-SPR to keep scaling to RR=16; the effect is primarily better action selection (relatable to the policy-churn phenomenon, Schaul 2022), not just optimization stabilization.
- **Result:** SR-SPR at RR=16 sets a new model-free SOTA on Atari 100k (Table 2: IQM 0.632 vs SPR 0.380 vs IRIS 0.501), rivaling methods that aggressively pretrained on extra data.

**The role of online interaction (§5.1) — three probing experiments.**
1. **Iterated offline (§5.1.1).** Alternate *purely offline* update bursts with data-collection (all the RR's updates applied at once right after each reset), rather than uniformly interleaved. This has *different*, generally worse RR scaling: applying a huge number of updates to a fixed dataset with SAC risks a **degenerate policy** unable to beat the previous one; the cycle breaks only when enough new data is collected — feasible on easy tasks (hopper-stand, walker-run) at a sample-efficiency cost, impossible on hard tasks (humanoid-stand, quadruped-walk). Explains the sudden jump in Fig. 4 once update count (and thus reset frequency) is large enough.
2. **Tandem (§5.1.2).** Two identical agents (differ only in init) both train on the *active* agent's buffer, but the **passive** agent never interacts. Post-reset high-RR training improves *both* initially, but after a few thousand steps the active agent stays stable while the **passive agent collapses**. Demonstrates online interaction acts as an **implicit regularization** mechanism — the small online stream is what prevents aggressive-training degeneration. Overall RR-scaling *capability* survives even under extreme off-policyness, but online data slows the collapse.
3. **Alternative offline/online mixes (§5.1.3).** For SR-SPR, doing *half* the interval's updates offline immediately after each reset mitigates the post-reset performance drop (improves *training* return) but has essentially *no* effect on final *evaluation* — useful when cumulative regret matters.

**Ablations (§5.2, Fig. 7).** For SR-SPR: SP-on-encoder gives ~constant +IQM past RR=4; the target network (for action selection) is the *most* important; SP-alone-on-all-params is insufficient (final layers must be fully reset); removing both SP and target ≈ Nikishin 2022 with SR-SPR's reset intervals — yields *some* scaling but less efficient; and critically, a **fixed env-step reset interval** (Nikishin 2022's default) gives **poor performance above RR=4**. Notably, these modifications help *specifically at high RR* — at RR 1–2 they don't improve (nor much harm) performance, suggesting other latent modifications may exist that only pay off at high RR.

**Data/compute tradeoff (§5.3, Fig. 8).** Resets provide a *knob* trading data for computation: to reach a performance obtainable by collecting 800,000 more env samples, one can instead spend ~2 orders of magnitude more *agent updates*. But this scaling is **inherently sequential** — more hardware does *not* speed it up (unlike model-parallel scaling).

**Limits of RR scaling (§6).** (1) Always bounded — beyond a finite RR there's no information left in the buffer to extract, and current methods can't auto-detect this limit, so they still collapse if pushed too far. (2) Can't overcome intrinsic algorithm limits (hard exploration / credit assignment). (3) Requires keeping the *entire* interaction history in the buffer — feasible for these benchmarks, costly at larger scale (permanent-storage buffers → slower retrieval). (4) Sequential ⇒ time-consuming ⇒ limits applicability to high-frequency-interaction settings.

**Relation to primer & siblings.** This is the family-(a) reset method (primer §4a) elevated to a *scaling* lever. It is the practical payoff of the pathology all four shard-3 papers describe: Abbas (activation collapse), Sokar (dormancy — and Sokar's Fig. 7 showing RR↑→dormancy↑ is the *mechanistic* explanation of *why* the replay-ratio barrier exists), Nikishin (injection as an alternative restoration, Fig. 5 showing injection-benefit rises with RR). D'Oro cites Nikishin 2022 (primacy bias) as its reset backbone; Nikishin 2023 in turn cites D'Oro's finding that first layers benefit from partial resets. **Project handshake:** the update-indexed reset schedule and the "reset final layers fully, soft-reset the encoder" recipe are the two most transferable design rules for the project's replay-ratio study.

### <a id="rr-appendix-section-by-section-backbone"></a>Appendix: Section-by-Section Backbone

- **Abstract.** Fully/partially resetting agent parameters produces better replay-ratio-scaling; training with an order of magnitude more updates significantly improves Atari 100k and DMC performance; analysis of the required design choices, limits, and tradeoffs.
- **§1 Introduction.** Sample efficiency valuable when interaction is costly; replay ratio (updates per interaction) is an appealing lever but gives limited benefit on standard baselines. With minimal, careful reset-based modifications (SR-SAC, SR-SPR), break the replay-ratio barrier: orders of magnitude more updates raise performance for a fixed interaction budget. Deep RL nets face dynamic datasets whose training determines future inputs *and* targets; the progressive loss of ability to learn/generalize (against which most RL deploys no countermeasure) is the main roadblock to RR scaling.
- **§2 Related Work.** *Loss of ability to learn/generalize* — invisible on static tasks, appears under distribution shift; partial resets help in continual learning (Ash & Adams); Berariu quantifies unrecoverable-damage update counts; in deep RL identified as transient-non-stationarity (Igl, self-distillation), capacity loss (Lyle 2022a, auxiliary tasks), sparse-reward (Lyle 2022b, policy distillation), loss of plasticity (Dohare, continual backprop). This paper leverages periodic hard resets (Zhou 2022; Nikishin 2022 primacy bias). *Scaling in deep/RL* — empirical scaling laws (Hestness, Kaplan); RR scaling as a data-efficient baseline vs model-based RL; prior high-RR via value-function ensembles (REDQ) or normalization (Smith) — argues explicit plasticity restoration pushes RR scaling much further.
- **§3 Effective Replay Ratio Scaling with Resets.** Off-policy replay-buffer loop; defines replay ratio and the sign-neutral **Replay Ratio Scaling** box. RR scaling is coupled to the online loop (training changes future data). Main limiting factor = progressive loss of ability to learn/generalize; restorable via partial/total resets with an *update-indexed* frequency (⇒ more updates → more frequent restoration → better performance).
- **§4 Replay Ratio Scaling Drastically Improves Sample Efficiency.** SAC (actor-critic, entropy-regularized) and SPR (model-free DQN variant with latent-dynamics auxiliary). §4.1 Continuous control — DMC15-500k/1M; reset all params every $2.56\times10^6$ updates; SR-SAC RR=128 > REDQ (RR=20), SAC, DDPG (new SOTA). §4.2 Atari 100k — reset every 40,000 updates; soft-reset encoder via SP interpolation $\theta_t=\alpha\theta_{t-1}+(1-\alpha)\phi$, $\alpha=0.8$; add EMA target ($\tau=0.005$) used for action selection; SR-SPR RR=16 new model-free SOTA (IQM 0.632).
- **§5 Algorithm Design in Light of Replay Ratio Scaling.** §5.1 Importance of online interaction — §5.1.1 iterated offline (all updates offline post-reset → degeneration risk, task-dependent), §5.1.2 tandem (passive agent collapses → online data = implicit regularizer), §5.1.3 alternative offline/online mixes (half-offline post-reset improves training return, not eval). §5.2 What's required for discrete-control scaling — SP-on-encoder (+0.04 past RR4), target net for action selection (most important), final layers must be fully reset, fixed env-step interval poor above RR4, modifications help specifically at high RR. §5.3 Data/compute tradeoff — resets as a knob; ~2 orders of magnitude of updates ≈ 800k more samples; inherently sequential (hardware doesn't help).
- **§6 The Limits of Replay Ratio Scaling.** Finite buffer information (no auto-detection of the limit → still collapses if overpushed); can't beat intrinsic algorithm limits (exploration/credit assignment); requires full-history buffer (storage cost); sequential ⇒ time-consuming ⇒ limits high-frequency-interaction applicability.
- **§7 Conclusions.** Partial/full resets unlock favorable RR scaling and sample efficiency for model-free deep RL (SR-SAC, SR-SPR) with minimal added complexity; discussion of important design choices, tradeoffs, and the value of online data; advocates discovering-and-exploiting new empirical deep-RL phenomena as a design methodology.

## 16. Lyle et al. 2023 — Understanding Plasticity in Neural Networks

**PDF:** `docs/project/references/continual_learning/sources/Lyle et al. 2023 - Understanding Plasticity in Neural Networks.pdf`
**Venue:** ICML 2023 (PMLR 202). **Authors:** Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney (Google DeepMind).

### Phase 1: Foundational Overview (Lyle 2023)

**The plain-language question.** A neural network that has already trained on one stream of
data often becomes *worse at learning new things* than a fresh, randomly-initialized network
of the same size. This is "loss of plasticity" — the network's ability to quickly update its
predictions in response to new information decays over training. It is a *different* problem
from catastrophic forgetting (which is about losing *old* knowledge); here the network loses
the ability to acquire *new* knowledge. The problem is especially acute in reinforcement
learning (RL), where the prediction targets constantly change (the value-function "bootstrap"
target moves as the policy improves), so non-stationarity is baked in rather than incidental.

**What the paper does.** It is a *systematic empirical autopsy* of plasticity loss. The
authors ask three questions in sequence: (1) what does plasticity loss look like in simple,
interpretable settings? (2) which of the popular "culprit" quantities — weight norm, feature
rank, number of dead units, weight-matrix rank — actually *cause* it? (3) which interventions
fix it?

**Key findings.**
- **A falsification framework kills the popular single-number explanations.** For every
  candidate culprit (weight norm, feature rank, dead-unit count, weight rank), you can build a
  learning problem where it *positively* correlates with plasticity loss and another where it
  *negatively* correlates. A quantity that flips sign depending on the dataset cannot be the
  causal driver. So none of these simple statistics is a reliable explanation on its own.
- **The real signature is the curvature of the loss landscape.** Plasticity loss tracks
  changes in the *sharpness* (largest Hessian eigenvalue) and the *gradient interference*
  structure of the loss surface that new tasks induce on the trained parameters. Importantly,
  this happens *even when no units are saturated* — so "dead ReLUs" is not the whole story.
- **Gradient descent itself is part of the problem.** Compared to a random walk of the same
  step size, gradient-based optimization drives parameters into regions where the landscape is
  *sharper* and gradients *interfere* more — i.e., the inductive bias of SGD actively worsens
  future trainability, beyond what mere movement away from initialization would do.
- **Adaptive optimizers can catastrophically self-destruct under abrupt task changes.** When
  the loss suddenly jumps (e.g., re-randomized labels, or a target-network update), Adam's
  moment estimates become stale and the update explodes, killing most ReLU units. A larger
  `ε` and faster second-moment decay fixes it — which is why deep-RL practitioners already use
  a large Adam `ε` by folk wisdom.
- **The best fix is smoothing the loss landscape, not perturbing parameters.** Across MLP /
  CNN / ResNet / ViT, *layer normalization* (and categorical "two-hot" output encoding) beat
  resetting, weight decay, spectral norm, and shrink-and-perturb. Adding layer norm to a
  vanilla Double-DQN improves performance across the 57-game Atari benchmark with *no*
  hyperparameter retuning.

**Initial takeaway.** Loss of plasticity is not reducible to any one tidy scalar. The most
predictive lens is the *geometry of the loss surface* (curvature + gradient interference),
and the most reliable practical lever is an architectural choice — normalization — that keeps
that geometry well-behaved. This is the paper that reframed the field from "which statistic to
regularize" toward "keep the optimization landscape trainable."

*Primer connection:* this is a Phase-3 "mechanisms" pillar. It is the paper that establishes
**layer normalization** as the field's default first-line defense, a thread the primer traces
through to Lyle 2024's LN+weight-decay recipe.

### Phase 2: Graduate-Level Deep Dive (Lyle 2023)

#### 2.1 Formal setting: TD learning as a non-stationarity generator

The analysis is grounded in temporal-difference (TD) learning, chosen because it manufactures
non-stationarity even with a *fixed* data distribution. Given sampled transitions
$\tau_t = (s_{t-1}, a_t, r_t, s_t)$ and a network $f:\Theta\times\mathcal{S}\times\mathcal{A}\to\mathbb{R}$,
the TD loss is

$$
\ell(\theta, \tau_t) = \Big( f(\theta, s_{t-1}, a_t) - \boxed{\big(r_t + \gamma\, f(\theta', s_t, a')\big)} \Big)^2 ,
$$

where $\boxed{\;\cdot\;}$ denotes a stop-gradient and $\theta'$ are the (typically stale)
target-network parameters. The crucial point: the regression target
$r_t + \gamma f(\theta', s_t, a')$ **depends on parameters and changes as learning proceeds**,
so the objective is non-stationary *independent of the exploration policy*. This isolates
"target drift" as the non-stationarity source.

#### 2.2 Two loss-landscape probes: the Hessian spectrum and the gradient covariance

Two objects carry the entire mechanistic argument.

**(a) The Hessian and its spectrum.** For loss $\ell(\theta)$,

$$
H_\ell(\theta) = \nabla^2_\theta \ell(\theta) \in \mathbb{R}^{d\times d}, \qquad
\Lambda(H_\ell(\theta)) = (\lambda_1 \ge \cdots \ge \lambda_d).
$$

The top eigenvalue $\lambda_1$ measures **sharpness** (Dinh et al. 2017); the condition number
$\kappa = \lambda_1/\lambda_d$ governs gradient-descent convergence. A first-order intuition:
for a quadratic model $\ell(\theta)\approx \tfrac12(\theta-\theta^\*)^\top H (\theta-\theta^\*)$,
gradient descent with step $\alpha$ contracts error along eigen-direction $i$ by a factor
$(1-\alpha\lambda_i)$. Stability requires $\alpha < 2/\lambda_1$, so a growing $\lambda_1$
forces a smaller admissible learning rate; and convergence *speed* along the slowest direction
scales like $(1 - \alpha\lambda_d) = (1 - \alpha\lambda_1/\kappa)$, so a large $\kappa$ means
crawling progress. A sharpening landscape thus *directly* slows the fitting of new targets —
which is exactly what "loss of plasticity" is operationally.

**(b) The normalized gradient covariance (interference) matrix.** For sampled points
$x_1,\dots,x_k$,

$$
C_k[i,j] = \frac{\big\langle \nabla_\theta \ell(\theta, x_i),\ \nabla_\theta \ell(\theta, x_j)\big\rangle}
{\lVert \nabla_\theta \ell(\theta, x_i)\rVert\, \lVert \nabla_\theta \ell(\theta, x_j)\rVert}.
$$

This is the cosine similarity between per-example gradients. Negative off-diagonal entries
signal **interference**: reducing loss on $x_i$ *increases* it on $x_j$, so the network cannot
simultaneously satisfy both. A low-rank / block-structured $C_k$ means gradients are
near-collinear — helpful (generalization) when the dot product is positive, harmful
(interference) when negative.

#### 2.3 Defining plasticity operationally

Plasticity is defined (following Lyle et al. 2021) via an optimization operator
$\mathcal{O}:(\theta,\ell)\mapsto\theta^\*$ that runs a *fixed budget* of updates (2000 steps
in experiments). Over a distribution $\mathcal{L}$ of probe losses,

$$
\mathcal{P}(\theta_t) = b - \mathbb{E}_{\ell\sim\mathcal{L}}\big[\ell(\theta^\*_t)\big],
\qquad \theta^\*_t = \mathcal{O}(\theta_t, \ell),
$$

with $b$ a baseline (e.g., target variance). **Plasticity loss** over a trajectory is
$\mathcal{P}(\theta_t) - \mathcal{P}(\theta_0)$; it is *independent of the baseline* $b$, so it
measures relative degradation of a checkpoint as an optimization *starting point*. The probe
targets are

$$
g(x) = a + \sin\!\big(10^5\, f(x;\omega_0)\big),
$$

with $\omega_0$ a fresh initialization and offset $a$ set to the network's *current mean
prediction* (so random-init baselines are not unfairly favored). The high-frequency $\sin$
makes the targets effectively a uniform random direction in output space — a task-agnostic
probe of "can this network still be pushed in an arbitrary direction?"

#### 2.4 Case study 1 — Adam instability under abrupt non-stationarity

Adam's update is

$$
u_t = \alpha\,\frac{\hat m_t}{\sqrt{\hat v_t} + \bar\epsilon} + \epsilon,
$$

with $\hat m_t$ the first-moment (mean-gradient) and $\hat v_t$ the second-moment
(mean-squared-gradient) EMAs. Because gradient magnitude scales roughly with the loss, a sudden
loss jump (re-randomized labels; a target-network refresh) makes both estimates stale. Under
default hyperparameters $\hat m_t$ (decay $\beta_1=0.9$) updates faster than $\hat v_t$ (decay
$\beta_2=0.999$), so immediately after a task change the numerator has grown while the
denominator lags:

$$
u_t \;\sim\; \frac{\text{(large fresh gradient)}}{\sqrt{\text{(small stale second moment)}}}
\;\Rightarrow\; \text{huge step} \;\Rightarrow\; \text{ReLU death / divergence.}
$$

The fix is to raise $\bar\epsilon$/$\epsilon$ (damping the denominator when it is small) and
lower $\beta_2$ (making $\hat v_t$ track faster). The paper notes DQN's canonically large
Adam/RMSProp $\epsilon$ is exactly this stabilization, converged on empirically by the RL
community.

#### 2.5 Case study 2 — SGD's inductive bias sharpens the landscape

Controlled experiment: two trajectories from the *same* initialization, applying updates of
*equal norm* — one following the true gradient, the other a Gaussian random walk (Brownian
motion). To probe how the *local geometry for arbitrary new targets* evolves, they measure the
Hessian of a stop-gradient perturbation objective

$$
\ell(\theta) = \big[\, f_\theta(X) - \boxed{f_\theta(X)} + \epsilon \,\big]^2, \qquad \epsilon\sim\mathcal{N}(0,1),
$$

i.e., "how hard is it to nudge the current outputs by random noise?" — deliberately *not* the
primary objective (whose Hessian trivially differs between trajectories). Result: both
trajectories increase the Hessian spectral norm, but **the gradient-descent trajectory grows
the outlier eigenvalues far faster and develops negative gradient interference** absent in the
random walk. Conclusion: SGD's inductive bias, not mere displacement from initialization,
pushes parameters into less-trainable regions.

#### 2.6 The falsification framework (the paper's central methodological contribution)

The premise (after Bühlmann 2020, invariant/causal prediction): *a genuinely causal predictor
of plasticity loss must keep a consistent correlation sign across interventions*. They train
128 DQN agents across tasks × observation spaces (CIFAR-10 vs MNIST) × optimizers × seeds and
log, per checkpoint: weight norm, weight (matrix) rank, dead-unit count, feature rank, plus
plasticity. For **each** candidate, there exists one environment with positive and one with
negative correlation to plasticity loss — e.g., weight norm correlates *positively* with
plasticity loss under CIFAR-10 observations but *slightly negatively* under MNIST; feature rank
and sparsity reverse sign depending on the reward function. Sign-reversal under intervention
$\Rightarrow$ the quantity is **falsified as a universal causal explanation**. (Note the paper
is careful: these are still useful *diagnostics*, just not causal levers.)

#### 2.7 Learning-curve diagnosis: slower slopes, not higher plateaus

Probing checkpoints at training iterations 0, 10, 20, 50, 100: later checkpoints do **not**
plateau early at a high loss (which would indicate bad local minima). Instead their learning
curves have *shallower slopes* and *higher variance / non-monotonicity*. In full-batch terms,
non-monotonic loss under fixed step size is the signature of an over-sharp landscape
(edge-of-stability, Cohen et al. 2021); in mini-batch terms they additionally observe rising
inter-minibatch interference. So plasticity loss = *difficulty navigating the landscape*, not
*entrapment in a minimum*.

#### 2.8 Interventions and the scaling result

- **Scaling is insufficient.** Widening a CNN to the single-GPU memory limit reduces but does
  not eliminate plasticity loss on the toy classification MDPs, especially when the task is
  misaligned with the architecture's inductive bias (MLP on CIFAR-10) or the net is
  under-expressive.
- **Intervention ranking (Fig. 6).** Landscape-smoothing methods win: **layer normalization**
  and **two-hot categorical output** give the largest reductions, often exceeding
  last-layer resetting. Parameter-perturbing / regularizing methods (shrink-and-perturb,
  weight decay $10^{-5}$, spectral norm) help less. Caveat: two-hot destabilized the policy in
  some cases and needed different optimizer hyperparameters — not a drop-in.
- **Atari validation.** Adding layer norm after each hidden layer of Double-DQN (RMSProp,
  ε-greedy, frame stacking, 200M frames) robustly improves human-normalized score across the
  57 games with no retuning; the biggest wins occur where the *default* agent had degenerate
  gradient-covariance / ill-conditioned Hessian, and LN restores better-behaved gradient
  covariance — closing the mechanistic loop.

**Trade-off flagged for future work.** Two-hot / categorical encodings smooth the landscape but
change output scale/semantics; there is a genuine tension between *preserving a trainable
gradient structure* and *accurately representing an evolving value function*.

### Appendix: Section-by-Section Backbone (Lyle 2023)

- **Abstract.** Plasticity = ability to quickly change predictions on new info; essential for
  RL. Networks lose it even on simple problems; mechanism poorly understood. Systematic
  empirical analysis → loss of plasticity deeply connected to *loss-landscape curvature
  changes*, but *often occurs without saturated units*. Identify parameterization/optimization
  choices that preserve plasticity; validate layer norm on Atari (ALE).
- **§1 Introduction.** Non-stationary training → reduced ability to solve new tasks; worst when
  input→target relationship changes over time (network must "overwrite" priors) — rare in SL,
  baked into RL. Existing fixes (layer/unit resets, feature regularization) probably act via
  *different* mechanisms, so hard to improve. Contributions: two interpretable case studies;
  a falsification framework (after Dziugaite et al. 2020) showing no single property uniquely
  explains plasticity loss; evidence that *loss-landscape curvature* is the crucial factor;
  broad intervention study → landscape-smoothing architecture choices (categorical output,
  normalization) win; LN on DQN improves ALE.
- **§2 Background.** Distinguishes catastrophic forgetting (old-task performance) from
  plasticity loss (new-task performance falls below a fresh random net). §2.1 Preliminaries: TD
  learning (Eq. 1) as non-stationarity source; Hessian $H_\ell$ (Eq. 2) and its spectrum;
  normalized gradient covariance $C_k$ (Eq. 3) and interpretation (interference vs
  generalization; low rank ⇒ collinearity). §2.2 Defining plasticity: classical complexity
  (VC/Rademacher) is capacity-agnostic to trainability → unsuitable; adopt Lyle et al. 2021
  operator-based definition; optimization operator $\mathcal{O}$; probe-loss distribution
  (Eq. 4); plasticity $\mathcal{P}$ (Eq. 5); trajectory loss $\mathcal{P}(\theta_t)-\mathcal{P}(\theta_0)$, baseline-independent.
- **§3 Methodology & motivating questions.** §3.1 Measuring plasticity: uniform prior over
  future targets via $g(x)=a+\sin(10^5 f(x;\omega_0))$, offset $a$ = current mean prediction,
  2000-step budget, 10 sampled functions. §3.2 Environments: block-MDP analogue of image
  classification over 10 states/actions with CIFAR-10 or MNIST observations — three variants:
  *true-label* (reward $\delta_{a=s}$), *random-label* (labels randomized; inductive-bias
  misaligned), *sparse-reward* (reward $\delta_{a=s=9}$, policy affects visitation). §3.3
  Outline: §4 what happens; §5 what properties cause it; §6 how to mitigate.
- **§4 Two simple studies.** §4.1 Optimizer instability: MLP memorizing re-randomized MNIST
  labels → default Adam diverges, kills ReLUs (Fig. 1); mechanism via Adam moment staleness
  (Eq. 6); fix = larger $\epsilon$, faster $\beta_2$. §4.2 Loss-landscape evolution: GD vs
  equal-norm Brownian motion from same init; stop-gradient perturbation Hessian; GD grows
  spectral outliers faster and induces negative gradient interference (Fig. 2).
- **§5 Explaining plasticity loss.** §5.1 Setting: DQN on each MDP×observation combo; probe
  every 5000 steps. §5.2 Falsification: 128 agents; each of weight norm / weight rank / dead
  units / feature rank reverses correlation sign across environments (Fig. 3) → none is a
  universal cause. §5.3 Learning-curve evolution: later checkpoints have shallower, more
  non-monotone probe-task learning curves (Fig. 4) → slow navigation, not bad minima.
- **§6 Solutions.** §6.1 Scaling reduces but doesn't eliminate plasticity loss (Fig. 5). §6.2
  Interventions across MLP/CNN/ResNet/ViT (Fig. 6): landscape-smoothing (LN, two-hot) beats
  perturbation/regularization; two-hot caveats. §6.3 Larger benchmark: LN on Double-DQN
  improves ALE across 57 games, changes gradient-covariance structure (Fig. 7).
- **§7 Related work.** Trainability/initialization (Glorot, He, mean-field, deep kernel
  shaping); ResNets bias toward identity → better gradients; loss-landscape smoothness &
  generalization; edge-of-stability, catapult, linear mode connectivity (all stationary SL);
  continual learning; resetting/distillation literature.
- **§8 Conclusions.** Contrast with large-model pretraining (good objectives accelerate
  adaptation) vs plasticity loss (bad objectives hurt adaptation). Stabilizing the loss
  landscape is the crucial lever, with ancillary benefits (easier optimization, better
  generalization).
- **Appendix A.** Case-study details (MNIST memorization MLP width-1024, Adam lr 1e-3 default
  vs tuned $\beta_2{=}0.9,\bar\epsilon{=}10^{-3}$; Brownian Q-learning, SGD lr 1e-3, batch 512;
  Lanczos Hessian eigenvalue density; probe protocol; MLP/CNN/ResNet-18/ViT architectures;
  Double-DQN ALE protocol, LN after each hidden layer, replay 100k, ε=0.1).

---

## 17. Lyle et al. 2024 — Disentangling the Causes of Plasticity Loss in Neural Networks

**PDF:** `docs/project/references/continual_learning/sources/Lyle et al. 2024 - Disentangling the Causes of Plasticity Loss.pdf`
**Venue:** Preprint (arXiv:2402.18762), Mar 2024. **Authors:** Clare Lyle, Zeyu Zheng, Khimya Khetarpal, Will Dabney, Hado van Hasselt, Razvan Pascanu, James Martens (Google DeepMind).

### Phase 1: Foundational Overview (Lyle 2024)

**The plain-language question.** Lyle 2023 showed that *no single number* explains plasticity
loss. So a natural worry follows: if you fix one cause, the network can still lose plasticity
through a *different* cause you didn't address. This follow-up paper asks: **how many
independent mechanisms are there, do they overlap, and can we combine one fix per mechanism to
get a robustly plastic learner?**

**The "Swiss cheese" model.** The paper's organizing metaphor: each mitigation is a slice of
Swiss cheese with holes (failure modes it misses). Any single slice lets some plasticity loss
through its holes. Stack slices whose holes *don't line up* — each targeting an independent
mechanism — and almost nothing gets through.

**Key findings.**
- **Plasticity loss decomposes into (at least) three independent mechanisms:**
  1. **Preactivation distribution shift** — the inputs to nonlinearities drift, producing not
     only the known "dead units" but also a *newly identified* pathology: **unit linearization
     ("zombie units")**, where a unit collapses into an effectively linear map and stops
     contributing nonlinearity.
  2. **Regression-target magnitude** — simply regressing on targets with a *large mean* (even
     stationary!) destroys plasticity. This alone explains a large fraction of previously
     reported RL plasticity loss, because value targets grow like $(1-\gamma)^{-1}$.
  3. **Parameter-norm growth** — weights grow → loss-landscape sharpness rises → optimization
     slows; also saturates normalization/softmax components.
- **All three roads end at the same place: a degenerate empirical Neural Tangent Kernel
  (eNTK).** However plasticity is lost, the network's eNTK collapses toward a
  *diagonal-plus-rank-1* structure — a shared "fingerprint" that is *predictive* of training
  difficulty and serves as a unified diagnostic.
- **Divide-and-conquer works.** Find the best fix for each mechanism in isolation, then combine:
  **layer normalization** (fixes preactivation shift) **+ L2 weight decay** (fixes norm growth)
  **+ scale-invariant / categorical output** (fixes target magnitude) gives *additive* benefit
  and drives plasticity loss to near-zero across synthetic non-stationary benchmarks, Atari
  (C51/Rainbow), DeepMind Control (SAC humanoid), and natural distribution shift (iWildcam).

**Initial takeaway.** Plasticity loss is not one disease but several, each with its own cure;
the practical recipe that emerges — **layer norm + weight decay + scale-invariant output** — is
cheap, combinable, and near-complete, and the eNTK gives a single diagnostic that all failure
modes share.

*Primer connection:* this is the Phase-3 paper that turns the "mechanisms" pillar into a
*prescription*. The primer's headline recommendation — LN + weight decay as the default
plasticity guardrail — comes directly from here; the target-magnitude mechanism also links to
the distributional-RL / two-hot thread the primer tracks.

### Phase 2: Graduate-Level Deep Dive (Lyle 2024)

#### 2.1 What kinds of non-stationarity induce plasticity loss?

Two independent problem-side factors:

**(a) Regression-target scale (dose-response).** Construct a *stationary* pretraining task:
targets are random labels offset by a constant, $f(x) = c + \epsilon(x)$ on MNIST inputs. Then
fine-tune on fresh targets (offset $c$ or zero-mean). Result: **larger pretraining offset $c$
$\Rightarrow$ worse fine-tuning**, monotonically — a clean dose-response curve. This is
decisive because the pretraining task is stationary; it is the *target mean*, not the
*existence* of non-stationarity, that drove the earlier DQN contextual-bandit plasticity loss
(where the optimal value is a one-hot plus a bias converging to $(1-\gamma)^{-1}$).

**(b) Smoothness of the distribution shift.** Sequentially re-randomize a fraction
$\epsilon\in[0.01,1]$ of CIFAR-10 labels each iteration. Larger $\epsilon$ (more abrupt shift)
$\Rightarrow$ more severe, more precipitous plasticity loss; $\epsilon=0.01$ barely hurts,
$\epsilon=0.1$ already accelerates the decline. Sudden shifts induce large gradient-magnitude
spikes (cf. the Adam instability of Lyle 2023).

#### 2.2 Mechanism 1 — preactivation distribution shift, dead units, and unit *linearization*

Let $z = \phi(a)$ where $a$ is the preactivation and $\phi$ the nonlinearity. Two failure modes
of a shifting preactivation distribution:

- **Dead unit.** $a<0$ for *all* inputs (ReLU) or $|a|$ huge (tanh) ⇒ $\phi'(a)=0$ ⇒ no
  gradient flows to the incoming weights ⇒ they freeze permanently (unless the *input*
  distribution later shifts). Solution candidate: non-saturating activations (Leaky ReLU).
- **Unit linearization ("zombie unit") — the new contribution.** A ReLU with *only positive*
  preactivations acts as the identity $\phi(a)=a$; a smooth unit with very *low-variance*
  inputs behaves near-linearly. Unlike dead units, zombies propagate gradients perfectly (even
  "perfect" signal propagation) — but they contribute **no nonlinearity**, so the network's
  *effective expressive power* silently collapses (Montúfar et al. 2014, Raghu et al. 2017:
  count of linear regions). This is invisible to dead-unit counters, which is why it had been
  missed.

**Two-phase post-task-change dynamics.** Immediately after a task switch the network passes
through (i) an **erasing phase**: predictive entropy spikes, incorrect logits are pushed toward
zero, gradients on first-layer incoming weights push *all* preactivations the same direction
(uniform sign of $\langle \nabla \ell(\theta,X), x\rangle$), coupled with large step sizes ⇒ a
burst of units entering the linearized/saturated regime; then (ii) a **disentanglement phase**:
as the loss shifts to *raising correct* logits, gradient directions diversify and nonlinearity
partially recovers — *provided* enough units did not permanently saturate.

#### 2.3 Mechanism 2 — regression-target magnitude via the bias-encoding singular value

Why does a large target mean poison the features? SVD of the penultimate-layer feature matrix
$\Phi\in\mathbb{R}^{n\times d}$ (n sampled inputs) reveals the mechanism. Training on
$f(x)=100+\epsilon(x)$:

- Without normalization, the **maximal singular value explodes** from $\mathcal{O}(10^3)$
  (mean-zero targets) to $\mathcal{O}(10^8)$ (mean-100 targets).
- The network *does not* use its output-layer **bias weight** to represent the large mean
  (bias norm is non-monotone, stays small). Instead it **encodes the constant offset into a
  single feature direction** with which all embeddings have roughly constant dot product — a
  learned "pseudo-bias" living in feature space.
- Consequence: one singular value dwarfs the rest; the feature Gram matrix becomes
  **ill-conditioned** ($\sigma_1/\sigma_i \to \infty$), and lower-order singular values collapse
  relative to $\sigma_1$. Since the last-layer least-squares / gradient dynamics have condition
  number set by $\sigma_1^2/\sigma_i^2$, learning any *new* direction (which lives in the
  crushed subspace) becomes extremely slow. Layer norm bounds $\sigma_1$ (it caps feature norm)
  but does *not* fully fix the relative decay of lower singular values — hence LN alone is
  insufficient, motivating the combination with scale-invariant output encoding.

#### 2.4 Mechanism 3 — parameter-norm growth → sharpness → saturation

Two learning difficulties from growing $\lVert\theta\rVert$:
- **Sharpness coupling.** Empirically parameter norm tracks the top Hessian eigenvalue (Fig. 3);
  last-layer norm ↔ sharpness is formalized in edge-of-stability analyses (Damian et al. 2022).
  Recall from Lyle 2023 §2.2: growing $\lambda_1$ shrinks the stable learning rate and worsens
  conditioning ⇒ slower adaptation.
- **Uneven-layer growth & component saturation.** If layers grow at different rates, the
  per-layer "effective learning rate" (which infinite-width analyses, e.g. μP/Yang, want equal
  across layers scaled by fan-in/out) becomes imbalanced. Large magnitudes also saturate
  softmax-attention heads and normalization layers (Wortsman et al. 2023; Merrill et al. 2021),
  shrinking the output change per fixed step. **Non-monotone caveat:** a network whose units are
  *all* saturated stops propagating gradients, so its norm stops growing — this is why parameter
  norm can be *causal* yet *not monotonically* related to plasticity loss.

#### 2.5 The unifying signature: empirical NTK collapse

The empirical NTK is the matrix of *output* (not loss) gradient dot products:
$K_\theta(x,x') = \langle \nabla_\theta f(\theta,x), \nabla_\theta f(\theta,x')\rangle$.
Decompose $K_\theta = D_\theta + G_\theta$ (diagonal + off-diagonal). Intuition: $G_\theta=0$
⇒ no generalization between inputs; $G_\theta = c\mathbf{1}$, $D_\theta=0$ ⇒ rank-1, the net is
"multiply by zero, add a learned constant."

**Why the eNTK matters under non-stationarity (heuristic derivation).** To first order, one
gradient-descent step on $\ell(\theta)=\lVert f(\theta,X)-y\rVert^2$ changes the loss by

$$
\ell(\theta_t) - \ell(\theta_{t+1}) \;\approx\; \big(f(\theta_t,X)-y\big)^\top K_{\theta_t}(X,X)\,\big(f(\theta_t,X)-y\big) + \text{h.o.t.}
$$

*Derivation sketch.* With learning rate $\eta$, the update is
$\theta_{t+1}=\theta_t - \eta\,\nabla_\theta \ell(\theta_t)$, and
$\nabla_\theta \ell = 2\,J^\top r$ where $J = \nabla_\theta f(\theta_t,X)$ is the Jacobian and
$r = f(\theta_t,X)-y$ the residual. First-order Taylor of $\ell$ along the update:

$$
\ell(\theta_{t+1}) \approx \ell(\theta_t) + \nabla_\theta\ell^\top(\theta_{t+1}-\theta_t)
= \ell(\theta_t) - \eta\,\lVert \nabla_\theta \ell\rVert^2
= \ell(\theta_t) - 4\eta\, r^\top (J J^\top)\, r,
$$

and $JJ^\top = K_{\theta_t}(X,X)$ is precisely the eNTK. So the per-step loss reduction is the
residual passed through the eNTK quadratic form. **The problem under non-stationarity:** the
targets $y_t$ evolve, so an eNTK $K_{\theta_t}$ that was well-aligned with the residual
$y-f(\theta_t,X)$ at one moment need not remain aligned as $y_t$ moves. A **collapsed
(low-rank) eNTK** has few directions in which it can reduce *any* residual ⇒ it cannot chase a
moving target. Hence *maintaining a non-collapsed eNTK is critical*.

**Empirical fingerprint (Fig. 3).** A random init has a rich eNTK. All three
plasticity-losing conditions collapse toward *diagonal-plus-low-rank*: (i) large-target-mean
regression ⇒ eNTK ≈ diagonal + rank-1 within a few hundred steps; (ii) ReLU random-label
memorization ⇒ block-diagonal (dead-unit blocks); (iii) Leaky-ReLU version ⇒ same pathology,
milder. Strikingly *different learning dynamics, same eNTK endpoint*.

#### 2.6 Mitigation: one fix per mechanism, then combine

| Mechanism | Best intervention | Notes |
|---|---|---|
| Unbounded norm growth | **Layer/batch normalization** (hard) + **L2** | Hard norm *constraint* on features works; regularizing feature norm softly is weaker; rescaling weight norm to init hurt optimization speed. |
| Preactivation shift | **Layer/batch norm** of preactivations; ReDo (reset dead) | Normalization aids both plasticity *and* single-task convergence; ReDo helps but can slow single-task convergence and misses init-time signal-prop issues (excluded on ResNets). |
| Loss-landscape conditioning | gradient-norm penalty, InFeR, Shampoo | None consistently beats **LN + L2**. |
| Target scale | **Two-hot categorical (distributional) output** + label smoothing | Two-hot lets the net represent large outputs *without* ill-conditioned features; label smoothing (mixture with uniform) fixes the saturated-softmax loss spike for $\gamma=0$; scale-invariant output is *necessary* in value-based RL. |

**Headline recipe: layer normalization + L2 weight decay.** LN fixes the preactivation
distribution; L2 caps weight-norm growth. The two are independent and additive. Validated on:
20-task synthetic CIFAR-10 non-stationarity sweeps (continual / composite / growing modes;
MLP/CNN/ResNet-18, L2 $=10^{-5}$) where accuracy *improves* over tasks; C51 + Rainbow on Atari
(LN gives modest but consistent gains; BN and L2 *interfere* with RL and are excluded); SAC on
DeepMind Control (LN gives near-uniform gains, striking on humanoid); iWildcam natural
distribution shift (LN + L2 widen the final-accuracy gap over 20 location shifts).

**Distributional-RL insight (mechanistic).** Categorical/two-hot losses help plasticity for
*two* reasons: (1) smoother gradients (Imani & White 2018), *and* (2) they let the network
encode large output values without developing an ill-conditioned, bias-dominated feature
representation — a genuinely new explanation of a known empirical benefit.

**Limitation flagged.** L2 controls norm at the cost of convergence speed; better norm-control
strategies that don't slow single-task training are open.

### Appendix: Section-by-Section Backbone (Lyle 2024)

- **Abstract.** The stationarity assumption underlies NN design; violating it (deep RL) makes
  learning brittle. Loss of plasticity = updating predictions gets harder over training. Claim:
  plasticity loss decomposes into *multiple independent mechanisms*; single interventions are
  insufficient but *combined* interventions are highly robust. **LN + weight decay** maintains
  plasticity across synthetic non-stationarities and ALE.
- **§1 Introduction.** Single-task training is easy; real relationships are dynamic; RL changes
  its own data distribution. Resets are expensive. Plasticity loss observed independently in RL
  and SL. Lyle et al. 2023 gave negative results (no single quantity). Existing methods either
  target one pathology (risk leakage via others) or regularize abstract properties (risk
  interfering with objective). Goal: a model combining both → target several independent
  mechanisms. Three questions: what non-stationarity induces it; what structural changes occur;
  what do plasticity-lost networks share. Answers: mechanisms unified by *preactivation
  distribution shift*; degeneracies in *empirical NTK*; new mechanism *unit linearization*;
  *target magnitude* explains much RL plasticity loss. → "Swiss cheese" mitigation model with
  additive benefit.
- **§2 Background & related work.** §2.1 Training/signal propagation/preactivation
  distributions: layer norm forces preactivation sample mean 0 / variance 1; signal-propagation
  literature; good init keeps preactivations $\approx\mathcal{N}(\mu,\sigma^2)$; no universal
  $(\mu,\sigma^2)$ but $(0,1)$ usually fine; LN enforces 1st/2nd-order statistics exactly. §2.2
  Loss of plasticity: two usages of "plasticity" (generalization vs trainability) — adopt
  *trainability*; plasticity = quality of a point as an optimization start; can include
  optimizer state; not necessarily permanent but transient loss still harmful. Prior mechanisms;
  none causal in isolation (Lyle 2023).
- **§3 A deeper look into plasticity.** §3.1 What non-stationarity induces it: *regression
  target scale* (stationary large-mean pretraining reproduces plasticity loss; dose-response;
  Appendix E.1 bias-encoding); *smoothness of distribution shift* (larger reset fraction ⇒
  worse; Fig. 1). §3.2 Mechanisms: *linearization & preactivation shift* (dead units, degraded
  signal propagation, zombie/linearized units; two-phase erasing/disentanglement dynamics,
  Fig. 2); *parameter-norm growth* (instability, sharpness ↔ Hessian, uneven layer growth,
  component saturation; non-monotone relationship, Fig. 3). §3.3 Characterizing
  plasticity-lost networks: empirical NTK $= D_\theta+G_\theta$; first-order loss-decrease
  formula; moving-target argument; all conditions collapse eNTK toward diagonal+low-rank
  (Fig. 3) → eNTK as diagnostic.
- **§4 Mitigation strategies.** §4.1 Per-mechanism: unbounded norm growth (hard LN/BN + L2 beats
  soft feature-norm regularization; weight-norm rescaling hurts, Fig. 4 left); preactivation
  statistics (LN/BN + ReDo, Fig. 4 center); loss-landscape conditioning (grad-norm penalty,
  InFeR, Shampoo — none beats LN+L2, Fig. 4 right); target scale (two-hot categorical + label
  smoothing; scale-invariant output necessary in RL, Fig. 5). §4.2 Evaluation: SL (20-task
  CIFAR-10 modes; accuracy improves over tasks); RL (C51/Rainbow + LN on ALE; SAC + LN on DMC
  humanoid; BN/L2 interfere with RL); natural shift (iWildcam, LN+L2 widen the gap).
- **§5 Conclusions.** No single property explains all plasticity loss, but a handful of
  independent mechanisms cover most cases; some new (large target offsets, zombification).
  Divide-and-conquer reduces combinatorial search. L2 controls norm but slows convergence —
  open problem.
- **Appendix E (theory highlights).** E.1: SVD of penultimate features; max singular value
  $\mathcal{O}(10^3)\to\mathcal{O}(10^8)$ with offset; network encodes bias in a
  single feature direction rather than the bias weight; ill-conditioning follows (Figs. 13–15).
  E.2: uneven per-layer norm growth. E.4/E.11: RL per-game/DMC results.

---

## 18. Dohare et al. 2024 — Loss of Plasticity in Deep Continual Learning (Nature)

**PDF:** `docs/project/references/continual_learning/sources/Dohare et al. 2024 - Loss of Plasticity in Deep Continual Learning (Nature).pdf`
**Venue:** *Nature* 632, 768–774 (22 Aug 2024). **Authors:** Shibhansh Dohare, J. Fernando
Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton (University
of Alberta / Amii).

> **Verification caveat (equations) — read before reuse.** The equations in this entry's Phase 2 — the continual-backprop **contribution-utility (Eq. 1)**, the **effective-rank (Eq. 2)**, and the **stable-rank** formula — were extracted from a two-column PDF whose tokens fragmented, and were **reconstructed from the surrounding prose**. Treat every symbol below as provisional and route it through a `math-reviewer` pass before lifting any equation verbatim into project code or docs.

### Phase 1: Foundational Overview (Dohare 2024)

**The plain-language question.** Standard deep learning uses two phases: a training phase where
weights change, then a frozen phase where the net is deployed. Natural learning is *continual* —
always learning. Does deep learning actually work if you never stop training on new data? This
Nature paper answers, definitively and at scale: **no.** Standard deep-learning methods
*gradually lose plasticity* until, after enough new tasks, they learn no better than a *shallow
(linear) network* — a total erasure of the benefit of depth.

**The demonstration (breadth is the point).** To be convincing for Nature, the result must be
systematic: many architectures, many optimizers, many hyperparameters, run long enough to
expose *long-term* decay (3–4 orders of magnitude more compute than a single training run).
They show plasticity loss on:
- **Continual ImageNet** — 1000 classes paired into ~500,000 binary tasks; task difficulty is
  *constant*, so any accuracy drop is pure plasticity loss. Backprop peaks near 88% then falls
  *below the linear baseline* by task 2000, at *all* step sizes.
- **Class-incremental CIFAR-100** — add 5 classes at a time; an incrementally-trained
  18-layer ResNet ends up 5% *worse* than a network retrained from scratch (a drop equal to
  removing batch norm).
- **Reinforcement learning** — a simulated ant robot with friction changed every 2M steps;
  standard PPO collapses after the first friction change. Even with *constant* friction, PPO's
  reward rises for ~3M steps then *collapses* — the ant fails every episode by 20M steps.

**The cure — continual backpropagation.** The fix is almost trivially simple: keep doing
backprop, but on every step **reinitialize a tiny fraction of the least-used units** back to
the initial random distribution (typically fewer than one unit per step). This continually
*injects diversity/variability* that gradient descent alone erodes. Continual backprop
*maintains plasticity apparently indefinitely* across all three settings — and, unlike Shrink-
and-Perturb (which perturbs *all* weights), it perturbs *selectively*, minimizing disruption to
what the net already knows.

**The three correlates of plasticity loss.** As backprop loses plasticity, three things happen
together: (1) the fraction of **dead/dormant units** rises; (2) the **average weight magnitude**
grows (→ ill-conditioned Hessian → slow convergence); (3) the **effective/stable rank** of the
representation drops (units become redundant / non-diverse). Continual backprop keeps all three
healthy.

**The bold claim.** *"Methods based on gradient descent are not enough — sustained deep learning
requires a random, non-gradient component to maintain variability and plasticity."* This is a
philosophical thesis, not just an algorithm: gradient descent is inherently *variability-
destroying*, so a *variation-and-selection* (evolution-like) process must be layered on top.

**Initial takeaway.** This is the field's capstone existence proof: standard deep learning
provably fails at continual learning, and the failure is *not intrinsic* — a cheap
selective-reinitialization rule fixes it. It elevates "loss of plasticity" from an RL curiosity
to a fundamental limitation of gradient-descent-based deep learning, published at the highest
visibility.

*Primer connection:* this is the primer's Phase-3 **anchor / capstone**. The three correlates
(dead units, weight magnitude, effective rank) and the continual-backprop algorithm are the
reference points the primer uses to organize the whole loss-of-plasticity literature; Shrink-
and-Perturb and L2 appear as the "partial fixes" that continual backprop completes.

### Phase 2: Graduate-Level Deep Dive (Dohare 2024)

#### 3.1 The continual-backpropagation algorithm and its utility measure

Continual backprop = standard backprop + a *selective reinitialization* step per update. The
key design choice is *which* units to replace, governed by a **contribution utility**.

**Contribution utility (Eq. 1).** For the $i$-th hidden unit in layer $l$ at time $t$, the
utility is a running average (decay $\eta=0.99$) of the magnitude of its contribution to the
next layer:

$$
u_{l,i,t} \;=\; \eta\, u_{l,i,t-1} \;+\; (1-\eta)\,\lvert h_{l,i,t}\rvert \sum_{k=1}^{n_{l+1}} \lvert w_{l,i,k,t}\rvert ,
$$

where $h_{l,i,t}$ is the unit's activation and $w_{l,i,k,t}$ its outgoing weight to unit $k$ in
layer $l+1$ ($n_{l+1}$ = width of layer $l+1$). **Intuition / derivation of the form:** the
signal a unit injects into a downstream consumer is $h_{l,i}\, w_{l,i,k}$; summed over
consumers and taken in magnitude, $\lvert h_{l,i}\rvert \sum_k \lvert w_{l,i,k}\rvert$ measures
how much this unit *matters* to the layer above. A unit whose contribution is small can be
overwhelmed by other units' contributions, so it is a safe candidate for replacement. The EMA
smooths this over the data stream.

**The reinitialization rule and its safeguards.**
- **Zero out the outgoing weights.** When unit $r$ is reinitialized, its *incoming* weights are
  resampled from the initial distribution $d_l$, but its *outgoing* weights are set to **zero**.
  *Why:* zero outgoing weights mean the fresh unit does **not** perturb the currently-computed
  function — the net's output is unchanged the instant of replacement, so learned behavior is
  preserved and the unit re-earns its influence only through subsequent gradient descent.
- **Maturity threshold $m$.** A freshly reset unit has zero utility (its outputs are zeroed), so
  it would be *immediately* re-selected for replacement — a churn pathology. To prevent this,
  units are protected for $m$ updates (a unit is "mature" once its age exceeds $m$; $m=100$ in
  ImageNet, up to 1000 elsewhere).
- **Replacement rate $\rho$.** Each step, a fraction $\rho$ of *mature* units per layer is
  replaced. $\rho$ is tiny — e.g., $\rho=10^{-5}$ with a 512-unit layer gives
  $512\times10^{-5}=0.00512$ replacements/step ≈ one replacement per ~200 updates. The
  algorithm accumulates a fractional counter $c_l \mathrel{+}= n_{\text{eligible}}\times\rho$ and
  triggers a replacement whenever $c_l>1$.

**Algorithm 1 (per input $x_t$).** Forward pass → evaluate loss → backward pass (SGD or variant)
→ for each layer $l\in[1,L-1]$: increment ages; update utilities (Eq. 1); count eligible
(mature) units; accumulate $c_l \mathrel{+}= n_{\text{eligible}}\rho$; if $c_l>1$, find the
minimum-utility mature unit $r$, resample its incoming weights $w_{l-1}[:,r]\sim d_l$, zero its
outgoing weights $w_l[r,:]=0$, reset $u_l[r]=0$ and age$[r]=0$, decrement $c_l$. In mini-batch
settings the instantaneous per-batch utility can replace the running average to save compute.

This is the modern, deep-learning-compatible descendant of "generate-and-test" feature search
(Selfridge's Pandemonium 1959; Mahmood & Sutton 2013) — a **variation-and-selection process in
the space of units**, layered on continuing gradient descent.

#### 3.2 The three quantitative correlates (formal definitions)

**(a) Dead / dormant units.** For ReLU: count units whose output is *zero for all* examples in a
2000-image sample taken at the start of each task. For sigmoidal activations: count units
within $\epsilon$ of an extreme (saturated) value. A dead unit has $\phi'(a)=0$ everywhere ⇒
zero gradient to its incoming weights ⇒ **frozen forever** (in permuted-MNIST, where all inputs
are non-negative, a first-layer dead ReLU can never revive). Dead units directly reduce network
capacity. (Cross-reference: Sokar et al. 2023's "dormant neuron"/ReDo is the sibling notion.)

**(b) Average weight magnitude & the Hessian condition-number argument.**
$\bar w = \frac{1}{|\theta|}\sum |\theta_i|$. Plasticity loss co-occurs with steady growth of
$\bar w$. The mechanistic link: in the second-order Taylor expansion of the loss,
$\ell(\theta+\delta)\approx \ell(\theta)+\nabla\ell^\top\delta + \tfrac12\delta^\top H\delta$,
the weights are tied to the Hessian $H$; large weights tend to yield a large **condition
number** $\kappa(H)=\lambda_{\max}/\lambda_{\min}$. Gradient descent's convergence rate is
governed by $\kappa$ — for a convex quadratic, the error contracts per step by
$\big(\frac{\kappa-1}{\kappa+1}\big)^2$, so $\kappa\gg1$ ⇒ near-unit contraction ⇒ crawling
progress. Hence weight growth ⇒ ill-conditioned Hessian ⇒ slow learning. (This is the same
sharpness/conditioning story Lyle 2023/2024 tell from the Hessian side.)

**(c) Effective rank and stable rank of the representation.** Let $\Phi\in\mathbb{R}^{n\times m}$
be the representation matrix with singular values $\sigma_1\ge\cdots\ge\sigma_q$,
$q=\max(n,m)$.

- **Effective rank (Eq. 2, Roy & Vetterli).** Normalize singular values into a distribution
  $p_k = \sigma_k / \lVert\sigma\rVert_1$, and take the exponential of their Shannon entropy:

$$
\operatorname{erank}(\Phi) \;=\; \exp\{H(p_1,\dots,p_q)\}, \qquad
H(p_1,\dots,p_q) \;=\; -\sum_{k=1}^{q} p_k \log(p_k).
$$

  It is continuous in $[1, \operatorname{rank}(\Phi)]$. **Interpretation & limits:** if one
  singular value dominates ($p_1\to1$, all others $\to0$), entropy $\to0$ and $\operatorname{erank}\to e^0=1$
  (one effective dimension). If all $q$ singular values are equal ($p_k=1/q$), entropy $=\log q$
  and $\operatorname{erank}=e^{\log q}=q$ (full diversity). So a *low* effective rank means a
  few units suffice to produce the layer output — the rest are redundant, a bad starting point
  for new tasks.
- **Stable rank (used for Fig. 2d).** For the same singular values sorted descending, the stable
  rank is the smallest $k$ capturing 99% of the (squared-)singular-value mass:

$$
\operatorname{srank}_{0.99}(\Phi) \;=\; \min\Big\{ k : \frac{\sum_{i\le k}\sigma_i}{\sum_{j\le q}\sigma_j} > 0.99 \Big\}.
$$

**The rank-collapse mechanism of plasticity loss.** Gradient-based optimization has an
*implicit bias toward low-rank solutions*. After each task the optimizer finds a low-rank
solution, which then *initializes* the next task; iterating, the representation's effective rank
**ratchets down** task after task, progressively shrinking the set of functions the network can
immediately represent at the start of a new task — a compounding loss of plasticity.

#### 3.3 The problem suite (why each testbed exists)

- **Continual ImageNet.** 1000 classes × 700 images (600 train / 100 test); binary tasks from
  class pairs (~500k tasks); 3-conv + 3-FC network, 2-unit head reset to zero at each task
  boundary (privileged task-boundary info, used only here as it is standard for
  sequential-independent-task CL); SGD+momentum 0.9, cross-entropy. *Constant task difficulty*
  ⇒ any accuracy drop = plasticity loss. Linear baseline doesn't degrade (its whole net is reset
  each task).
- **Class-incremental CIFAR-100.** 18-layer ResNet with batch norm, data augmentation, L2, LR
  schedule ("base deep-learning system"); grow classes 5→100; early-stopping-style weight reset
  to best-validation checkpoint each increment; compared to *retrain-from-scratch* to factor out
  the intrinsically harder many-class problem. Correlates plotted (dormant %, stable rank).
- **Ant locomotion (RL).** PPO on a simulated ant; friction changed every 2M steps
  (non-stationary) or held constant 50M steps (stationary). Standard PPO collapses; tuned-Adam
  PPO less so but still degrades; PPO + L2 or + continual-backprop maintain plasticity. Under
  constant friction, correlates (Fig. 4) mirror supervised learning: dormant units rise, stable
  rank falls, weight magnitude grows under PPO; L2 fixes norm but drives weights *too* small
  (prevents committing to good behavior); continual backprop (with slight L2) keeps improving.
- **Online Permuted MNIST.** Cheap testbed: 800 random pixel-permutations as tasks, one online
  pass, no mini-batches; used for the in-depth correlate study; loss of plasticity robust across
  step sizes, network sizes (100–10,000 units — even the largest lose *some* plasticity), and
  task-change rates (10k–1M examples).
- **Slowly-Changing Regression (SCR).** Ultra-idealized CPU-scale problem (15 min/run). Input =
  binary vector of size $m{+}1$: $f$ slowly-changing bits (one flipped every $T$ examples),
  $m{-}f$ random bits, 1 constant bias bit. Target = fixed random *target network* with LTU
  (linear-threshold-unit) hidden layer, weights $\pm1$, threshold $\theta_i=(m{+}1)\beta - S_i$
  ($S_i$ = number of negative input weights). Target net (100 hidden units) is *more complex*
  than the learner (5 hidden units), forcing continual tracking of a moving best-approximation.
  Result: squared error rises for *all six* activations (sigmoid, tanh, ELU, leaky-ReLU, ReLU,
  Swish) — ReLU/tanh worst (to linear-baseline level), ELU less severe but still degrading ⇒
  plasticity loss is *not* an artifact of one activation.

#### 3.4 What the existing partial fixes do (and don't)

- **L2 regularization.** Penalty $\lambda\lVert\theta\rVert_2^2$ keeps weight magnitude from
  growing ⇒ reduces plasticity loss substantially. But it does *not* stop dead-unit growth or
  effective-rank collapse ⇒ *partial* fix. In RL it can shrink weights *too* far (Fig. 4d),
  hurting commitment to good behavior.
- **Shrink-and-Perturb (Ash & Adams 2020).** L2 shrink + Gaussian noise to *all* weights. Caps
  weight magnitude *and* reduces dead units (noise revives them) ⇒ *almost* fully mitigates
  plasticity loss in permuted MNIST — but effective rank stays lower than continual backprop's,
  and it is sensitive to noise variance (too high ⇒ worse loss of plasticity).
- **Adam, Dropout, normalization.** Surprisingly, these popular methods *increased* plasticity
  loss in Dohare's continual settings (Extended Data Fig. 4a) — a notable tension with Lyle
  2023/2024, who find *layer* normalization strongly *helps*. (Likely reconciliation: Dohare's
  "normalization" is *online/batch* normalization in a single-pass online regime, and the
  optimizer/objective regimes differ; the papers agree that *weight-norm control* and
  *variability injection* matter.)

**The synthesis (Discussion).** During continual training, units become *dormant, overcommitted,
and similar to each other*; the network irreversibly loses diversity and thus the ability to
learn. Continual backprop restores diversity *selectively* (least-used units only) — variation
and selection in unit-space plus continuing gradient descent. The thesis: **gradient descent is
variability-destroying; sustained learning needs a random, non-gradient component.**

### Appendix: Section-by-Section Backbone (Dohare 2024)

- **Abstract.** Deep learning uses train-then-freeze; natural learning is continual. Standard
  deep-learning methods *lose plasticity* in continual settings until they learn no better than
  a shallow net. Shown on ImageNet + RL across networks/algorithms. Plasticity maintained only
  by algorithms that *continually inject diversity* — e.g. continual backprop (reinit a small
  fraction of least-used units). Gradient descent alone insufficient; need a random non-gradient
  component.
- **Intro.** Deep learning's train/deploy split; ChatGPT example; continuing to train on new
  data is usually ineffective; retrain-from-scratch costs millions. Real-world change is
  ubiquitous. Loss of plasticity first shown ~2000 in psychology; visible in recent works;
  distinct from catastrophic forgetting. Continual backprop overview; roots in generate-and-test
  (Pandemonium 1959).
- **Plasticity loss in supervised learning.** Continual ImageNet setup; backprop peaks 88% then
  falls below linear baseline by task 2000 at all step sizes (Fig. 1b). Weight-shrinking methods
  (L2, Shrink-and-Perturb) are exceptions and maintain plasticity (Fig. 1c). Class-incremental
  CIFAR-100 with 18-layer ResNet base system; incremental training ends 5% below
  retrain-from-scratch after 100 classes (Fig. 2b); dormant units rise, stable rank falls
  (Fig. 2c,d); continual backprop eliminates the loss.
- **Plasticity loss in reinforcement learning.** RL needs continual learning more (agent changes
  its own data). Harder to demonstrate rigorously. Ant with changing friction: standard PPO
  fails catastrophically (sawtooth for others); tuned-PPO better but degrades; PPO + L2 /
  continual backprop maintain plasticity (Fig. 3). Constant-friction ant: PPO collapses after
  ~3M steps; correlates mirror SL (Fig. 4).
- **Maintaining plasticity.** Adam/Dropout/normalization *worsen* plasticity loss; L2 and
  Shrink-and-Perturb help. Continual backprop: reinit small number of least-used units, zero
  outgoing weights, maturity protection; maintains plasticity across all settings, with fewer
  dormant units, high stable rank, constant weight magnitude. Variation-and-selection framing.
- **Discussion.** Deep learning fails when learning must continue (learns no better than shallow
  nets). Problem not intrinsic — Shrink-and-Perturb and especially continual backprop maintain
  plasticity indefinitely by adding continuing variability (continual backprop restricts it to
  least-used units).
- **Methods.**
  - *Specifics of continual backprop:* contribution-utility Eq. 1; zero outgoing weights;
    maturity threshold $m$; replacement rate $\rho$; Algorithm 1; mini-batch instantaneous
    utility option.
  - *Continual ImageNet details:* 1000 classes, 700 img/class; 32×32 downsampled; 3-conv+3-FC;
    2-unit head reset to zero at task change; SGD+momentum 0.9; step sizes 0.01/0.001/0.0001;
    30 runs; grid search over L2/Shrink-and-Perturb/continual-backprop hyperparameters.
  - *Class-incremental CIFAR-100:* increments of 5 classes; 200 epochs/increment (4000 total);
    LR schedule resetting each increment; validation-best weight reset; 18-layer ResNet; stable
    rank Eq.; final continual-backprop accuracy 76.13%.
  - *Robust loss in permuted MNIST:* Online Permuted MNIST (800 tasks, one online pass, no
    mini-batches); robustness across step sizes / network sizes (100–10,000) / task-change rates.
  - *Slowly-Changing Regression:* binary input ($f$ slow + $m{-}f$ random + 1 bias bit), LTU
    target network, six activations; error rises for all activations.
  - *Understanding loss of plasticity:* only the weights change over time ⇒ initial distribution
    has special properties (diversity, non-saturation, small magnitude). Three correlates: dead
    units (measure via all-zero over 2000 samples), average weight magnitude (→ Hessian
    condition number → slow convergence), effective rank Eq. 2 (entropy-of-singular-value-
    distribution; low rank ⇒ redundant units ⇒ bad starting point; low-rank ratchet across
    tasks).
  - *Existing methods:* L2 (caps norm, partial — dead units/rank still worsen); Shrink-and-
    Perturb (caps norm + revives dead units, near-complete but noise-sensitive); Adam/Dropout/
    online-normalization worsen plasticity loss.

# Adjacent Threads

**Bridges to the project's concerns.** Four papers that are not "loss of plasticity" proper but connect the literature to the project's specifics: the activation scheme (19) that Abbas repurposed as a plasticity fix, the covariance mechanism (20) behind policy-entropy collapse in PPO, the curriculum-learning framework (21) that lets the project's negative curriculum result be diagnosed precisely, and the task-agnostic continual-RL result (22) showing a carried recurrent belief-state can beat task-aware agents. Read together in the synthesis, the last three converge on one diagnosis of the project's curriculum failure.

---


## 19. Shang et al. 2016 — Understanding and Improving CNNs via Concatenated ReLU (CReLU)

**PDF:** `docs/project/references/continual_learning/sources/Shang et al. 2016 - Concatenated ReLU (CReLU).pdf`
**Venue:** ICML 2016 (PMLR 48:2217–2225). **Authors:** Wenling Shang, Kihyuk Sohn, Diogo Almeida, Honglak Lee.

**Primer connection.** This is the activation scheme that Abbas et al. (2023) later repurposed as their *strongest single fix* for loss of plasticity in continual deep RL (primer §2 Phase 3, §4(c)). The original 2016 paper is a *supervised-vision* paper with no mention of plasticity — it argues CReLU on the grounds of parameter efficiency, regularization, and reconstruction. Reading it clarifies *why* the mechanism happens to preserve plasticity: CReLU makes it structurally impossible for a unit to become one-sided-dead, because every filter's negative phase is always represented by a live companion channel. That is the bridge from "better CIFAR/ImageNet features" (2016) to "keeps units responsive under non-stationary RL training" (2023).

<a id="p1-crelu"></a>
### Phase 1 — Foundational Overview

**The problem in plain terms.** A standard ReLU unit computes $\max(x, 0)$: it keeps positive signal and throws away everything negative. Shang et al. noticed something odd when they looked inside a trained AlexNet: in the first few convolution layers, the learned filters come in **near-opposite pairs** — for almost every filter there is another filter pointing in nearly the opposite direction. The network is spending two filters to represent one direction, one for the "positive phase" and one for the "negative phase," because ReLU erased the negative half of each and the network had to relearn it separately. That is wasted capacity (redundancy).

**The fix.** Instead of letting the network learn those mirror-image pairs the hard way, *build the mirroring into the activation*. **Concatenated ReLU (CReLU)** takes each linear response $x$, makes a negated copy $-x$, stacks them, and applies ReLU to both: output is the pair $(\max(x,0),\ \max(-x,0))$. Now a single filter automatically contributes both its positive and its negative half. Whichever sign the input has, *one of the two channels is always active* — so information is never destroyed, and no filter can be permanently silenced by an unlucky sign.

**Key findings.**
- Replacing ReLU with CReLU in the lower convolution layers *improves accuracy* on CIFAR-10, CIFAR-100 and ImageNet — often *while using fewer parameters* (because you can halve the filter count and still match or beat the baseline's number of live activations).
- The benefit is concentrated in the **lower layers**; deep layers show less "pairing," so CReLU there helps little (best ImageNet result: CReLU on conv1–4 only).
- CReLU acts as a **regularizer**: CReLU models show a much smaller train/test error gap than ReLU models with the same or more parameters.
- After adding CReLU, the mirror-pair phenomenon *disappears* (as intended) — each filter now uniquely spans its own direction.

**Initial takeaway.** A one-line change to the activation — "also keep the negative half, as its own channel" — recovers information ReLU throws away, removes learned redundancy, regularizes, and (crucially for our field) guarantees each unit always has a live channel. That last property is exactly what makes CReLU a plasticity-preserving activation in the RL continual-learning setting, even though this paper never uses that word.

<a id="p2-crelu"></a>
### Phase 2 — Graduate-Level Deep Dive

#### 2.1 The pairing observation, formalized

For a set of unit-length filters $\{\phi_i\}$, define the **pairing filter** of $\phi_i$ as the filter most anti-aligned with it:

$$\bar\phi_i = \arg\min_{\phi_j}\ \langle \phi_i, \phi_j\rangle,$$

and their cosine similarity

$$\mu^{\phi}_i = \langle \phi_i, \bar\phi_i\rangle.$$

Empirically, for AlexNet's `conv1` the histogram of $\mu^{w}_i$ (over learned weight filters $w$) is **strongly negatively centered** — i.e. $\mu^w_i \approx -1$ for many filters, meaning true near-opposite pairs — whereas for random Gaussian unit filters $r_i$ the histogram of $\mu^r_i$ centers near $0$ (random high-dimensional vectors are nearly orthogonal). Going deeper (`conv2`→`conv5`), the learned distribution's center drifts back toward $0$: the pairing is a *lower-layer* phenomenon. This is the empirical premise for restricting CReLU to lower layers.

The **conjecture**: despite ReLU erasing negative linear responses, the lower layers capture *both* phases by learning negatively correlated filter pairs — implying redundancy that a phase-preserving activation could remove.

#### 2.2 CReLU definition and the information-view of activations

Denote ReLU by $[\cdot]_+ \triangleq \max(\cdot, 0)$.

**Definition (CReLU).** The CReLU activation $\rho_c:\mathbb{R}\to\mathbb{R}^2$ is
$$\rho_c(x) \triangleq \big([x]_+,\ [-x]_+\big).$$

An information-theoretic reading of three activations clarifies what CReLU preserves:
- **ReLU** retains the *phase* but destroys the *modulus* when the response is negative (it maps all $x<0$ to $0$).
- **AVR** (absolute-value rectification, $|x|$) retains the *modulus* but destroys the *phase* (it cannot distinguish $x$ from $-x$).
- **CReLU** retains *both*: from $([x]_+, [-x]_+)$ you can reconstruct $x = [x]_+ - [-x]_+$ exactly.

This is why CReLU beats AVR empirically (Tables 1, 4): AVR discards phase, and phase turns out to be essential for state-of-the-art deep CNN features. It is also the seed of the plasticity argument: because $x$ is exactly recoverable and *one of the two channels is always $>0$ whenever $x\neq 0$*, a CReLU "unit" (the pair) can never be pushed into the permanently-zero-gradient regime that kills a lone ReLU unit.

#### 2.3 Reconstruction property (Proposition 2.1) — step by step

Let $x\in\mathbb{R}^D$ be an input and $W$ the $D\times K$ matrix whose columns are the filters $w_i\in\mathbb{R}^l$. Decompose $x$ orthogonally with respect to the column space of $W$:

$$x = x' + (x - x'),\qquad x'\in\operatorname{range}(W),\quad (x-x')\in\ker(W^\top).$$

**Proposition 2.1.** The component $x'$ (the part of the input spanned by the filters) is fully recoverable from
$$f_{\text{cnn}}(x) \triangleq \operatorname{CReLU}(W^\top x).$$

*Why this holds (sketch of the constructive proof).* Write the linear responses $y = W^\top x \in \mathbb{R}^K$. CReLU stores $([y]_+, [-y]_+)$, and since $y = [y]_+ - [-y]_+$, the full real vector $y=W^\top x$ is recovered **losslessly** from the CReLU output — no thresholding information is lost (contrast plain ReLU, which only gives $[y]_+$ and cannot recover the negative coordinates of $y$). Given $y = W^\top x = W^\top x'$ (because $x-x'\in\ker(W^\top)$ contributes nothing), and $x'\in\operatorname{range}(W)$ means $x' = W\alpha$ for some coefficient vector $\alpha$, we have $y = W^\top W \alpha$. On $\operatorname{range}(W)$ the Gram operator $W^\top W$ is invertible (restricted to the row space), so $\alpha = (W^\top W)^{+} y$ and

$$x' = W\,(W^\top W)^{+}\,W^\top x = W\,(W^\top W)^{+}\, y,$$

with $(\cdot)^{+}$ the pseudo-inverse. This is precisely the orthogonal projector onto $\operatorname{range}(W)$ applied to $x$. The paper's Algorithm 1 is a linear (no additional learning) reconstruction realizing this map; Figure 5 shows the recovered images. The component $x-x'\in\ker(W^\top)$ is genuinely unrepresented by the filters and is irrecoverable — as it must be for any filter bank. The max-pooling case needs extra input-space constraints for a non-trivial bound (supplementary §A.2).

#### 2.4 Regularization: the Rademacher-complexity argument (Theorem 4.1)

The striking empirical fact is that CReLU *doubles* the parameter count yet does **not** increase overfitting. The formal support:

**Theorem 4.1.** Let $\mathcal{G}$ be a class of real functions $\mathbb{R}^{d_{in}}\to\mathbb{R}$ with input dimension, $\mathcal{G}=[\mathcal{F}]^{d_{in}}_{j=1}$. Let $\mathcal{H}$ be a linear map from $\mathbb{R}^{2d_{in}}\to\mathbb{R}$ parameterized by $W$ with $\lVert W\rVert_2 \le B$. Then the empirical Rademacher complexity of the composite obeys

$$\hat{\mathfrak{R}}_L(\mathcal{H}\circ\rho_c\circ\mathcal{G}) \le \sqrt{d_{in}}\,B\,\hat{\mathfrak{R}}_L(\mathcal{F}).$$

*Interpretation.* This bound is **the same** as the known bound for ReLU + linear transformation (Wan et al. 2013). The CReLU doubling ($d_{in}\to 2d_{in}$ channels) does not enlarge the complexity bound because the two channels $[x]_+$ and $[-x]_+$ are *deterministic functions of the same pre-activation* — they carry no independent Rademacher degrees of freedom. The key contraction step is that $\rho_c$ is $1$-Lipschitz componentwise (both $[\cdot]_+$ and $[-\cdot]_+$ are), so the standard Ledoux–Talagrand contraction absorbs the concatenation without an extra factor. Hence "twice the parameters, same capacity bound" — the formal shape of the observed regularization.

To rule out that CReLU's two output channels are merely negations of each other (which would make it collapse to AVR), Table 6 measures the correlation between the *outgoing* weights of the positive-channel and negative-channel of each pair; the "pair" correlations are only marginally above the "non-pair" baseline and both are well below $1$ — i.e. the network genuinely learns *distinct* non-linear manipulations of the two phases.

#### 2.5 Bridge to plasticity (the reason this paper is in a continual-learning corpus)

Nothing in Shang et al. names plasticity, but the mechanism translates directly to the loss-of-plasticity vocabulary of the primer:
- A **dormant/dead ReLU unit** (Sokar et al. 2023; primer §2 Phase 3) is a unit whose pre-activation is negative on (almost) all inputs, so it outputs $0$ and receives (almost) no gradient. CReLU's construction makes this **impossible at the pair level**: if $x<0$ almost always, the companion channel $[-x]_+ > 0$ almost always and stays trainable.
- **Effective-rank collapse** (Kumar et al. 2021; Abbas et al. 2023) is mitigated because CReLU's lossless preservation of the pre-activation keeps feature directions distinct (§2.3), rather than collapsing negative-phase directions to zero.
- Abbas et al. (2023) found CReLU the *most effective* single intervention among activation changes, resets, and regularization in cycling-Atari continual RL. This paper supplies the mechanistic "why": phase preservation + guaranteed-live companion channel. Cost to note for any project use: CReLU **doubles the width** of every layer it touches (so downstream weight matrices double their input dimension) — the parameter/compute accounting Abbas inherits comes straight from §2.2 here.

<a id="bb-crelu"></a>
### Appendix: Section-by-Section Backbone

**§1 Introduction.** Motivates from a curious observation: AlexNet's lower conv layers learn negatively-correlated ("opposite-phase") filter pairs (Fig. 1). Hypothesizes lower layers capture both phases via redundant pairs; proposes CReLU to remove the redundancy while preserving both phases and non-saturated non-linearity. Claims parameter-efficiency + accuracy gains on CIFAR-10/100 and ImageNet.

**§2 CReLU and Reconstruction Property.**
- **§2.1 Conjecture on convolution layers.** Defines pairing filter $\bar\phi_i=\arg\min_{\phi_j}\langle\phi_i,\phi_j\rangle$ and $\mu^\phi_i=\langle\phi_i,\bar\phi_i\rangle$. Histograms (Fig. 2) show learned `conv1` filters are negatively centered (true pairs) vs. random filters near $0$; effect fades with depth. Information view: ReLU keeps phase/loses modulus; AVR keeps modulus/loses phase; scattering nets keep modulus. Defines **CReLU**: $\rho_c(x)=([x]_+,[-x]_+)$. Contrasts with Leaky ReLU (a *function* with small negative slope) — CReLU is an *activation scheme*, composable with other non-linearities.
- **§2.2 Reconstruction property.** Because CReLU preserves all post-convolution information, reconstruction analysis is clean. **Proposition 2.1**: the range-of-$W$ component $x'$ of input $x$ is recoverable from $\operatorname{CReLU}(W^\top x)$. Max-pooling case deferred to supplementary (needs extra constraints).

**§3 Benchmark Results.**
- **§3.1 CIFAR-10/100.** Baseline ConvPool-CNN-C / VGG. Replacing ReLU→CReLU (same filter count → doubles channels/params) improves accuracy; CReLU+half (halved filters → same #neurons, half the params of baseline) still beats baseline (Table 1). On deeper VGG, applying CReLU to conv1 / conv1,3 / conv1,3,5 while halving filters gives substantial gains (Table 2). CReLU shows smaller train/test gap (regularization). AVR sometimes beats baseline but is inferior to CReLU under averaging/voting.
- **§3.2 ImageNet.** Baseline All-CNN-B. CReLU on **conv1–4** gives best top-1/top-5 (Table 4); going deeper doesn't help (matches the depth-fading pairing observation). CReLU(all) with only 4.7M params beats FriedNet/PrunedNet parameter-reduction methods (Table 5).

**§4 Discussion.**
- **§4.1 Regularization view.** CReLU overfits less despite 2× params. **Theorem 4.1**: Rademacher complexity bound of CReLU+linear equals that of ReLU+linear — doubling params need not increase capacity.
- **§4.2 Invariant features.** CReLU models have consistently higher invariance scores (Fig. 4); local maxima at conv1/conv4/conv7 motivate the CReLU(conv1,4,7) architecture, which achieves best 10-patch ImageNet result with fewer params.
- **§4.3 Revisiting reconstruction.** After CReLU, the pairing phenomenon vanishes (Fig. 3: learned distribution aligns with random). Table 6: positive/negative outgoing-weight correlations well below 1 → the two phases are manipulated distinctly (not mere negation), separating CReLU from AVR. Linear (no-learning) reconstructions (Fig. 5) qualitatively confirm Proposition 2.1.

**§5 Conclusion.** CReLU conserves positive+negative linear responses so each filter efficiently spans its own direction; improves classification with fewer params; suggested extensions to structured prediction / generation.

---

## 20. Cui et al. 2025 — The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models

**PDF:** `docs/project/references/continual_learning/sources/Cui et al. 2025 - The Entropy Mechanism of RL for Reasoning LLMs (preprint).pdf`
**Venue:** arXiv preprint 2505.22617v1 (28 May 2025). **Lead authors:** Ganqu Cui, Yuchen Zhang, Jiacheng Chen, et al. (Shanghai AI Lab / Tsinghua / UIUC).

**Primer connection.** Primer §5 flags this as the *policy-entropy-collapse* thread: PPO-style training can drive a policy onto a near-deterministic point it cannot recover from, a *behavioural* near-absorbing failure that compounds with plasticity loss. The primer notes the project's "eat-once-then-starve" degeneration and the modulator-temperature-head as a candidate adaptive entropy floor. This review supplies the full mechanism: the entropy-performance exchange law $R\approx-a\,e^{H}+b$ and the covariance identity $-\mathrm{d}H\propto\operatorname{Cov}(\log\pi, \pi\cdot A)$ that explains monotone collapse and motivates the two fixes (Clip-Cov, KL-Cov). §6.5(3) of the primer's verification notes: treat as a **2025 preprint** (NeurIPS 2025 acceptance not independently confirmed).

<a id="p1-cui"></a>
### Phase 1 — Foundational Overview

**The problem.** When you fine-tune a large language model with reinforcement learning to make it reason better (reward it for correct math/code answers), a reliable and damaging pattern appears: the policy's **entropy** — a measure of how much randomness/uncertainty is left in its choice of next token — **collapses toward zero within the first few hundred training steps.** The model becomes overconfident and stops exploring alternative reasoning paths. At the same moment, validation accuracy stops improving. Over 95% of both the entropy drop and the performance gain happen in the first ~1/3 of training; the remaining 2/3 of compute yields almost nothing.

**The empirical law.** Cui et al. show that *without any entropy intervention*, downstream performance $R$ and policy entropy $H$ are tied by a simple, tight, two-parameter curve:

$$R = -a\,e^{H} + b.$$

This holds across 11 base models (0.5B–32B params), 4 model families, math and code tasks, and 4 RL algorithms. Consequences: (1) you can **predict** the final performance from the first ~15% of training; (2) the **ceiling is fixed** — when entropy is exhausted ($H=0$), $R = -a + b$, and no amount of extra RL compute gets past it. So scaling RL naively hits a wall set by the entropy mechanism.

**Why entropy falls (the mechanism).** They prove the change in entropy from one step to the next equals (the negative of) the **covariance** between how likely an action already is and how much its logit is being pushed up. Under policy-gradient updates that logit push is proportional to the action's **advantage** (how much better than average it is). So: a token that is *already high-probability AND high-advantage* gets reinforced, which lowers entropy. Early in training the model is well-calibrated (confident tokens really are good), so this covariance is large and positive — entropy plummets. It stays positive throughout, so entropy keeps falling.

**The fix.** Since a *tiny fraction* of "pivotal" high-covariance tokens drive the collapse, restrain *just those*. Two simple surgical methods:
- **Clip-Cov** — randomly detach (stop-gradient) a small fraction of high-covariance tokens so they don't contribute to the update.
- **KL-Cov** — apply a KL penalty to the top-covariance tokens, softly holding them near the old policy.

Both keep entropy an order of magnitude higher, sustain exploration, lengthen responses, and beat vanilla GRPO by +2.0% (7B) and +6.4% (32B) on math benchmarks. Naive entropy-bonus or reference-KL regularization *fails* (hyper-parameter-sensitive or performance-degrading).

**Initial takeaway.** RL for reasoning LLMs "trades entropy for reward" on a predictable exponential curve; the trade is driven by a small set of high-covariance tokens; controlling those tokens breaks the ceiling. For our project the transferable idea is that **entropy collapse is a covariance-driven near-absorbing state**, and the right lever is *targeted* control of the actions that dominate the covariance — not a blunt global entropy bonus.

<a id="p2-cui"></a>
### Phase 2 — Graduate-Level Deep Dive

#### 2.1 Setup and definitions

An LLM policy $\pi_\theta$ autoregressively generates $y=\{y_1,\dots,y_T\}$ from prompt $x$; RL maximizes verifier reward
$$\max_\theta\ J(\theta) := \mathbb{E}_{x\sim D,\,y\sim\pi_\theta(x)}\,[\,r(y)\,].$$
The policy-gradient estimator (Williams 1992):
$$\nabla_\theta J(\theta) = \mathbb{E}_{x\sim D,\,y\sim\pi_\theta(x)}\!\left[\sum_{t=0}^{T}\nabla_\theta \log\pi_\theta(y_t\mid y_{<t})\,A_t\right].$$
Advantage $A_t$ varies by algorithm: REINFORCE uses $A_t=r(y)$; **GRPO** normalizes within a group of $K$ samples,
$$A_t = \frac{r(y) - \operatorname{mean}\big(r(y^{1:K})\big)}{\operatorname{std}\big(r(y^{1:K})\big)};$$
**PPO** optimizes the clipped surrogate
$$L(\theta)=\mathbb{E}_t\!\left[\min\!\Big(\tfrac{\pi_\theta(y_t\mid y_{<t})}{\pi_{\theta_{old}}(y_t\mid y_{<t})}A_t,\ \operatorname{clip}\big(\tfrac{\pi_\theta}{\pi_{\theta_{old}}},1-\epsilon,1+\epsilon\big)A_t\Big)\right].$$

**Policy entropy** (token-level, averaged over data):
$$H(\pi_\theta, D) = -\mathbb{E}_{D,\pi_\theta}\big[\log\pi_\theta(y_t\mid y_{<t})\big] = -\frac{1}{|D|}\sum_{x\in D}\frac{1}{|y|}\sum_{t=1}^{|y|}\mathbb{E}_{y_t\sim\pi_\theta}\big[\log\pi_\theta(y_t\mid y_{<t},x)\big].$$

#### 2.2 The entropy–performance law and its corollaries

Empirically (over 200+ data points per curve, fit with just 2 coefficients):
$$\boxed{\,R = -a\,e^{H} + b\,}$$
Differentiating,
$$\frac{\mathrm{d}R}{\mathrm{d}H} = -a\,e^{H},$$
so $a$ is the **conversion rate** of entropy into performance (larger $a$ = trades more efficiently). At full exhaustion $H=0$:
$$R_{\max} = -a + b,$$
the deterministic ceiling. Both $a$ and $b$ are found to be **algorithm-irrelevant** (GRPO, RLOO, PRIME, REINFORCE++ collapse onto one curve → $a,b$ are intrinsic to *model + data*), and both vary **log-linearly with model size**, enabling extrapolation of a large model's ceiling from small-model runs.

#### 2.3 Entropy dynamics — the covariance identity (Lemma 1) with full derivation

Consider a **tabular softmax** policy, each state-action pair with its own logit $z_{s,a}=\theta_{s,a}$:
$$\pi_\theta(a\mid s)=\frac{\exp(z_{s,a})}{\sum_{a'\in A}\exp(z_{s,a'})}.$$

**Lemma 1 (entropy difference of softmax policy).** Under a first-order (small-$\eta$) update,
$$H(\pi_\theta^{k+1}\mid s) - H(\pi_\theta^{k}\mid s) \approx -\operatorname{Cov}_{a\sim\pi_\theta^k(\cdot\mid s)}\!\Big(\log\pi_\theta^k(a\mid s),\ z^{k+1}_{s,a}-z^{k}_{s,a}\Big).$$

*Derivation (step by step).* First-order Taylor expansion of entropy along the logit update $z^{k+1}=z^k+\eta\nabla J$:
$$H(\pi^{k+1}_\theta\mid s)\approx H(\pi^k_\theta\mid s) + \big\langle \nabla H(\pi^k_\theta\mid s),\ z^{k+1}-z^k\big\rangle.$$
Compute the entropy gradient. With $H=-\mathbb{E}_{a\sim\pi}[\log\pi]$,
$$\nabla_\theta H(\pi_\theta\mid s) = -\mathbb{E}_{a\sim\pi}\big[\nabla_\theta\log\pi_\theta(a\mid s) + \log\pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)\big].$$
The first term vanishes because $\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a\mid s)] = \nabla_\theta\!\sum_a\pi = \nabla_\theta 1 = 0$. Hence
$$\nabla_\theta H(\pi_\theta\mid s) = -\mathbb{E}_{a\sim\pi}\big[\log\pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)\big].$$
Insert the **softmax log-derivative** (Lemma 2): $\dfrac{\partial\log\pi_\theta(a\mid s)}{\partial\theta_{s,a'}} = \mathbf{1}\{a=a'\}-\pi_\theta(a'\mid s)$. Then
$$\big\langle\nabla_\theta H, z^{k+1}-z^k\big\rangle = -\mathbb{E}_{a\sim\pi}\!\left[\log\pi(a\mid s)\sum_{a'}\big(\mathbf{1}\{a=a'\}-\pi(a'\mid s)\big)\big(\theta^{k+1}_{s,a'}-\theta^{k}_{s,a'}\big)\right].$$
The inner sum equals $(\theta^{k+1}_{s,a}-\theta^k_{s,a}) - \sum_{a'}\pi(a'\mid s)(\theta^{k+1}_{s,a'}-\theta^k_{s,a'}) = (z^{k+1}_{s,a}-z^k_{s,a}) - \mathbb{E}_{a'\sim\pi}[z^{k+1}_{s,a'}-z^k_{s,a'}]$. Substituting and recognizing the structure $\mathbb{E}[\,X\cdot(Y-\mathbb{E}Y)\,]=\operatorname{Cov}(X,Y)$ with $X=\log\pi(a\mid s)$ (and using $\mathbb{E}[\log\pi\cdot\mathbb{E}(\Delta z)]=\mathbb{E}[\log\pi]\cdot\mathbb{E}[\Delta z]$):
$$\big\langle\nabla_\theta H, z^{k+1}-z^k\big\rangle = -\operatorname{Cov}_{a\sim\pi}\big(\log\pi(a\mid s),\ z^{k+1}_{s,a}-z^k_{s,a}\big).\qquad\blacksquare$$

Interpretation: entropy falls when actions that already have **high log-probability** get their **logits increased** — the covariance is positive.

#### 2.4 Coupling to advantage (Proposition 1, Theorem 1) with derivation

**Proposition 1 (logit change under vanilla PG).** With tabular softmax updated by $z^{k+1}_{s,a}=z^k_{s,a}+\eta\,\nabla_{\theta_{s,a}}J(\theta)$,
$$z^{k+1}_{s,a}-z^{k}_{s,a} = \eta\,\pi_\theta(a\mid s)\,A(s,a).$$

*Derivation.* $\nabla_{\theta_{s,a}}J = \mathbb{E}_{a'\sim\pi}\big[\nabla_{\theta_{s,a}}\log\pi(a'\mid s)\,A(s,a')\big] = \sum_{a'}\pi(a'\mid s)\big(\mathbf{1}\{a=a'\}-\pi(a\mid s)\big)A(s,a')$ (Lemma 2). Expand:
$$= \pi(a\mid s)\Big[(1-\pi(a\mid s))A(s,a) - \!\!\sum_{a'\neq a}\!\pi(a'\mid s)A(s,a')\Big] = \pi(a\mid s)\Big[A(s,a) - \sum_{a'}\pi(a'\mid s)A(s,a')\Big].$$
The bracketed baseline $\sum_{a'}\pi(a'\mid s)A(s,a') = \mathbb{E}_{a'\sim\pi}[A(s,a')] = 0$ (Lemma 3: advantage has zero mean under $\pi$). Thus $\nabla_{\theta_{s,a}}J=\pi(a\mid s)A(s,a)$ and the result follows. $\blacksquare$

**Theorem 1 (entropy change under PG).** Substituting Proposition 1 into Lemma 1:
$$H(\pi^{k+1}_\theta\mid s) - H(\pi^{k}_\theta\mid s) \approx -\eta\,\operatorname{Cov}_{a\sim\pi_\theta^k(\cdot\mid s)}\!\Big(\log\pi^k_\theta(a\mid s),\ \pi^k_\theta(a\mid s)\,A(s,a)\Big).$$

**Theorem 2 (entropy change under natural PG).** For NPG the logit change is simply $z^{k+1}_{s,a}-z^k_{s,a}=\eta\,A(s,a)$ (from Agarwal et al. 2021), giving the cleaner
$$H(\pi^{k+1}_\theta\mid s) - H(\pi^{k}_\theta\mid s) \approx -\eta\,\operatorname{Cov}_{a\sim\pi_\theta^k(\cdot\mid s)}\!\Big(\log\pi^k_\theta(a\mid s),\ A(s,a)\Big).$$

**Conclusion of the analysis:** a strong positive correlation between an action's probability $\pi(a)$ and its advantage $A(a)$ drives entropy *down*; a negative correlation drives it *up*. A rare (low-prob) high-advantage action would *raise* entropy — which is exactly what exploration needs.

#### 2.5 Empirical verification

On on-policy GRPO (Qwen2.5-7B, bandit view: prompt = state, whole response = action), the measured covariance $\operatorname{Cov}(\cdot)$ and the negative entropy difference $-\mathrm{d}H$ track each other almost exactly over 2000 steps (Fig. 8 left) — direct confirmation of Theorem 1. $\operatorname{Cov}(\cdot)$ stays positive throughout (→ monotone entropy decrease) and is *larger for easy/high-accuracy prompts* (well-calibrated → strong prob-advantage alignment) and *smaller for hard prompts* (Fig. 8 right). This difficulty-dependence of the covariance is the sharpest empirical handle in the paper.

#### 2.6 The two interventions (Clip-Cov, KL-Cov)

A small fraction of tokens carry outsized covariance (Table 1: top 0.02% of tokens have mean covariance $5.65$ vs. overall $0.003$). Define the token-wise centered cross-product estimator over a batch of $N$ rollout tokens:
$$\operatorname{Cov}(y_i) = \Big(\log\pi_\theta(y_i) - \tfrac{1}{N}\textstyle\sum_j \log\pi_\theta(y_j)\Big)\cdot\Big(A(y_i) - \tfrac{1}{N}\textstyle\sum_j A(y_j)\Big).$$

**Clip-Cov.** Uniformly sample a small fraction $r$ of tokens whose covariance lies in a high band $[\omega_{low},\omega_{high}]$ (both $\gg$ average, $>500\times$):
$$I_{clip} = I\sim\operatorname{Uniform}\big(\{i\mid \operatorname{Cov}(y_i)\in[\omega_{low},\omega_{high}]\},\ \lfloor r\cdot N\rfloor\big),$$
and **detach** those tokens from the gradient:
$$L_{\text{Clip-Cov}}(\theta) = \begin{cases}\mathbb{E}_t\big[\tfrac{\pi_\theta(y_t\mid y_{<t})}{\pi_{\theta_{old}}(y_t\mid y_{<t})}A_t\big], & t\notin I_{clip}\\[4pt] 0, & t\in I_{clip}\end{cases}$$

**KL-Cov.** Select the top-$k$ proportion by covariance, $I_{KL}=\{i\mid \operatorname{Rank}(\operatorname{Cov}(y_i))\le k\cdot N\}$, $k\ll 1$, and apply a KL penalty on those tokens:
$$L_{\text{KL-Cov}}(\theta) = \begin{cases}\mathbb{E}_t\big[\tfrac{\pi_\theta}{\pi_{\theta_{old}}}A_t\big], & t\notin I_{KL}\\[4pt] \mathbb{E}_t\big[\tfrac{\pi_\theta}{\pi_{\theta_{old}}}A_t - \beta\,D_{KL}(\pi_{\theta_{old}}\,\|\,\pi_\theta)\big], & t\in I_{KL}\end{cases}$$

**Results.** With $r=2\times10^{-4}$ (Clip-Cov) or $k=2\times10^{-3}$/$2\times10^{-4}$ and $\beta=1$ (KL-Cov), both beat GRPO and the clip-higher baseline: +2.0% avg (7B), +6.4% avg (32B), with +15.0%/+14.6% on the hardest AIME24/AIME25 for 32B. Entropy stays $>10\times$ higher than the collapsing baseline; response length grows; no plateau. Entropy is *tunable* by $r$ or $\beta$ (Fig. 12); KL-Cov gives stabler entropy curves than Clip-Cov. Naive entropy-loss and reference-KL both fail (Figs. 9–10). Connection to clip-higher: raising PPO's upper clip $\epsilon$ implicitly admits more *low-covariance* (low-prob, high-advantage) tokens — Cui et al. make the covariance the *explicit* control variable instead.

#### 2.7 Relevance to the project

The mechanism is architecture-agnostic (it holds for any softmax policy, which includes the project's categorical action heads). Two transferable points: (1) entropy collapse is not random decay but a *covariance-driven* near-absorbing dynamic, so an "eat-once-then-starve" degeneracy can be read as high-covariance reinforcement of an early-dominant action; (2) the effective remedy is *targeted* suppression of the pivotal high-covariance actions, which is a sharper instrument than a global entropy bonus (shown here to fail). The primer's proposed modulator-temperature-head as an adaptive entropy floor is consistent with this — but Cui et al.'s finding is that *where* you spend the entropy budget (which tokens/actions) matters more than the global level.

<a id="bb-cui"></a>
### Appendix: Section-by-Section Backbone

**§1 Introduction.** RL for reasoning LLMs faces policy-entropy collapse: entropy drops to ~0 in a few steps, performance saturates. Establishes empirical law $R=-a\,e^{H}+b$ (fully predictable, fixed ceiling at $H=0$). Two corollaries: (1) exploitation-exploration curve is predetermined (predictable like scaling laws); (2) upper bound deterministic → naive RL-compute scaling has marginal return. Naive entropy regularization fails. Motivates mechanistic analysis + covariance-based control.

**§2 The Predictable "Collapse" of Policy Entropy.**
- **§2.1 Preliminaries.** Objective $J(\theta)=\mathbb{E}[r(y)]$; PG estimator; GRPO advantage (group normalization); PPO clipped surrogate; token-level entropy definition.
- **§2.2 Settings.** 4 model families, 11 base models (0.5–32B), math+code (8 benchmarks), 4 RL algorithms (GRPO, REINFORCE++, PRIME, RLOO), veRL framework, "Zero" setting, KL coef 0 by default.
- **§2.3 First glance.** Entropy sharp-drops and monotonically →0; performance rises then saturates. 73% entropy consumption + 76% performance gain in first 200/2400 steps; 93%/94% by step 800.
- **§2.4 Fitting curves.** $R=-a\,e^{H}+b$ fits all runs with 2 coefficients. Can predict final performance from first 15% of steps (RMSE ~0.5–1.9%). At $H=0$, $R=-a+b$.
- **§2.5 Understanding coefficients.** $a,b$ are algorithm-irrelevant (GRPO/RLOO/PRIME/REINFORCE++ share a curve) → intrinsic to model+data. $\mathrm{d}R/\mathrm{d}H=-a\,e^H$ ($a$=conversion rate; $-a+b$=max). $a,b$ vary log-linearly with model size → extrapolate large from small.
- **§2.6 Discussion.** Predictability echoes scaling laws but is not universal (off-policy / different policies differ). Conditionally supports the "RL only elicits pretrained behaviors / ceiling exists" claim, but attributes the ceiling to the entropy mechanism, not an intrinsic RL limit.

**§3 Dynamics Analysis of Policy Entropy.**
- **§3.1 Softmax policy.** **Lemma 1**: $H^{k+1}-H^k\approx-\operatorname{Cov}(\log\pi, \Delta z)$.
- **§3.2 Under PG / NPG.** **Proposition 1**: $\Delta z_{s,a}=\eta\,\pi(a\mid s)A(s,a)$. **Theorem 1**: $\Delta H\approx-\eta\operatorname{Cov}(\log\pi,\pi A)$. **Theorem 2** (NPG): $\Delta H\approx-\eta\operatorname{Cov}(\log\pi,A)$. Positive prob-advantage correlation → entropy down.
- **§3.3 Empirical verification.** On-policy GRPO on Qwen2.5-7B; $\operatorname{Cov}(\cdot)$ and $-\mathrm{d}H$ track exactly (Fig. 8L); Cov stays positive; higher for easy prompts, lower for hard (Fig. 8R).

**§4 Entropy Control by Covariance Regularization.**
- **§4.1 Effect of entropy regularization.** Entropy loss hyper-parameter-sensitive (too small→no effect, too large→explosion); reference-KL stabilizes entropy but degrades performance. Naive methods insufficient.
- **§4.2 Suppressing high-covariance tokens.** Table 1: top 0.02% tokens dominate covariance. Token-wise covariance estimator (Eq. 10). **Clip-Cov** (detach random high-cov tokens, Eqs. 11–12) and **KL-Cov** (KL-penalize top-k cov tokens, Eqs. 13–14). Pseudocode Listing 1 (a few lines of change).
- **§4.3 Experiments.** Qwen2.5-7B/32B on DAPO-MATH; beat GRPO (+2.0%/+6.4% avg; +15%/+14.6% on AIME for 32B); entropy $>10\times$ higher; longer responses; no plateau; more stable than clip-higher (Table 2, Fig. 11).
- **§4.4 Controlled entropy.** Entropy tunable via clip ratio $r$ or KL coef $\beta$ (Fig. 12); KL-Cov stabler.
- **§4.5 Discussion.** Clip-higher = implicitly adding low-covariance tokens; Cov is the direct control. A few pivotal tokens ($10^{-4}$–$10^{-3}$) control entropy; optimal entropy value still open.

**§5 Related Work.** Maximum-entropy RL lineage; predictability / scaling-laws / reward-model overoptimization; RL for LLM post-training.

**§6 Conclusion.** Performance gains bought with exploratory capacity → foreseeable ceiling; covariance-based Clip-Cov / KL-Cov counteract collapse; scaling RL needs more than entropy minimization.

**Appendix E (proofs).** Lemma 2 (softmax log-derivative), Lemma 3 ($\mathbb{E}_\pi[A]=0$), full proofs of Lemma 1, Proposition 1, Theorem 2 (reproduced in Phase 2 above).

---

## 21. Narvekar et al. 2020 — Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey

**PDF:** `docs/project/references/continual_learning/sources/Narvekar et al. 2020 - Curriculum Learning for RL Domains.pdf`
**Venue:** JMLR 21(181):1–50, 2020. **Authors:** Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E. Taylor, Peter Stone.

**Primer connection.** Primer §5 uses this survey as the *reference frame for why a curriculum can yield zero or negative benefit*: a curriculum is a bet that the inter-task transfer mechanism is net-positive, and the project's curriculum agent transferred a *collapsed* policy and *discarded* the belief state — net-negative. This review extracts the survey's formal machinery (its curriculum definition, the seven classification dimensions, the transfer-metric vocabulary, and the five sequencing families) so those diagnoses can be stated precisely. Being a 50-page survey, the deep dive targets the taxonomy and the curriculum-MDP formalism rather than every surveyed paper.

<a id="p1-narvekar"></a>
### Phase 1 — Foundational Overview

**The idea.** Just as a human student learns arithmetic before calculus, an RL agent can often learn a hard task faster if it is first trained on a *sequence* of easier, related tasks — a **curriculum**. The motivating example is Quick Chess: a graded series of mini-chess variants (smaller boards, fewer pieces) that build up to full chess. The survey's job is to give the field a *common language* and a *classification framework* for the many scattered ways people have built such curricula for RL.

**Three parts of the method.** Any curriculum-learning approach decomposes into:
1. **Task generation** — where do the intermediate tasks come from? (hand-designed, or automatically generated).
2. **Sequencing** — in what order should tasks be presented? (the central, most-studied problem).
3. **Transfer learning** — how is knowledge carried from one task to the next? (transfer a policy, a value function, a model, options/skills, or a shaping reward).

**How curricula are evaluated.** Borrowed from transfer learning: **time-to-threshold** (how much faster you reach a target performance), **asymptotic performance** (final performance after convergence), **jumpstart** (initial boost from transfer), and **total reward** (area under the learning curve). A crucial distinction is **weak transfer** (time spent on source tasks treated as a free sunk cost) vs. **strong transfer** (all that source-task time is charged against the curriculum). A curriculum can *look* good under weak transfer and *lose* under strong transfer — exactly the accounting a project must do before claiming a curriculum "helped."

**When curricula help vs. hurt.** The survey repeatedly warns of **negative transfer**: a badly chosen intermediate task, or a badly chosen thing-to-transfer, can make the target task *harder* than learning from scratch. The whole point of task generation is to produce tasks "such that knowledge transfer through them is beneficial" and "avoid negative transfer." There is (as of 2020) very little *theory* on when curricula help — the open-problems section flags this gap explicitly.

**Initial takeaway.** A curriculum is not automatically beneficial; it is a *bet* with three independently-fallible components (task set, order, transfer mechanism) that must be evaluated under honest (strong-transfer) accounting. This framework is what lets the project say precisely which component of *its* curriculum failed.

<a id="p2-narvekar"></a>
### Phase 2 — Graduate-Level Deep Dive

#### 3.1 Formal definition of a curriculum (Definition 2)

A task is an MDP $m_i=(S_i,A_i,p_i,r_i)$. Let $\mathcal{T}$ be a set of tasks and $D_{\mathcal{T}}$ the set of all transition samples generable from them:
$$D_{\mathcal{T}} = \{(s,a,r,s')\mid \exists\, m_i\in\mathcal{T}\ \text{s.t.}\ s\in S_i, a\in A_i, s'\sim p_i(\cdot\mid s,a), r\leftarrow r_i(s,a,s')\}.$$

**Definition 2 (Curriculum).** A curriculum $C=(V,E,g,\mathcal{T})$ is a **directed acyclic graph** where $V$ is a vertex set, $E\subseteq\{(x,y)\mid (x,y)\in V\times V \wedge x\neq y\}$ the directed edges, and $g:V\to\mathcal{P}(D_{\mathcal{T}})$ maps each vertex to a subset of samples (with $\mathcal{P}$ the power set). A directed edge $\langle v_j,v_k\rangle$ means the samples at $v_j$ should be trained on before those at $v_k$. All paths terminate at a single sink node $v_t$ (the target task).

Three common **simplifications** reduce this general graph:
- **Single-task curriculum** (Def. 3): all samples come from one task — i.e. ordering experience within a task (e.g. prioritized replay).
- **Task-level curriculum** (Def. 4): each vertex is an entire intermediate task; the DAG is over tasks.
- **Sequence curriculum** (Def. 5): a linear chain $[m_1,m_2,\dots,m_n]$ — the simplest and most common form.

These compose: e.g. a *task-level sequence curriculum* is an ordered list of tasks. The project's difficulty-ladder curriculum is a **task-level sequence curriculum** in this taxonomy.

#### 3.2 Transfer-learning evaluation metrics (the accounting that matters)

From §2.3, four metrics compare a post-transfer learning curve on the target against a from-scratch learner:
- **Time to threshold**: episodes/steps/wall-clock to reach expected return $G_0\ge\delta$.
- **Asymptotic performance**: final converged performance.
- **Jumpstart**: initial performance improvement at the start of target-task learning.
- **Total reward ratio**: accumulated reward up to a fixed stop, transfer vs. no-transfer.

**Weak vs. strong transfer.** The transfer curve conventionally starts at time $0$ on the target *even though source-task time was already spent* — that is **weak transfer** (source time = sunk cost). **Strong transfer** charges source-task (and, most comprehensively, curriculum-generation) time by offsetting the curves. The survey notes that *achieving an asymptotic improvement implies strong transfer*, whereas time-to-threshold claims are only meaningful once you specify which accounting is used. This is the precise lever behind the primer's critique that the project's curriculum must be judged on a fair (strong-transfer) budget.

#### 3.3 The seven classification dimensions (the taxonomy)

Every surveyed method is placed on seven attribute axes (attribute: *values*):
1. **Intermediate task generation**: *target / automatic / domain experts / naive users* — who/what produces the source tasks.
2. **Curriculum representation**: *single / sequence / graph*.
3. **Transfer method**: *policies / value function / task model / partial policies / shaping reward / other / no transfer* — what knowledge crosses task boundaries. Low-level (full policy / value function / model → directly initialize the learner) vs. high-level (partial policies/options, shaping rewards → guide but not initialize).
4. **Curriculum sequencer**: *automatic / domain experts / naive users*.
5. **Curriculum adaptivity**: *static* (whole curriculum fixed before training) vs. *adaptive* (dynamically adjusted using in-training signals like learning progress).
6. **Evaluation metric**: *time to threshold / asymptotic / jumpstart / total reward* — bolded when strong transfer.
7. **Application area**: *toy / sim robotics / real robotics / video games / other*.

For the project's setup: intermediate tasks = *domain experts* (hand-graded difficulties); representation = *sequence*; transfer method = *policies* (weights carried) — and, per the primer, the belief/recurrent state was *discarded*, which in this taxonomy is a transfer-method choice that omitted the most valuable thing to transfer; adaptivity = *static*.

#### 3.4 The five sequencing families (§4.2)

Sequencing methods form a spectrum by *how much the intermediate tasks may differ from the target MDP*:
1. **Sample sequencing (§4.2.1)** — reorder samples from the *target* task without changing the domain (supervised-CL analog of Bengio 2009). Includes **Prioritized Experience Replay** (Schaul 2016; prioritize high-TD-error transitions), complexity-index sorting (Ren 2018), ScreenerNet learned weights (Kim & Choi 2018). "No transfer" needed (single task).
2. **Co-learning (§4.2.2)** — a curriculum *emerges* from multi-agent interaction (self-play, competition/cooperation): e.g. OpenAI hide-and-seek (Baker 2020), asymmetric self-play (Sukhbaatar 2018), AlphaStar (Vinyals 2019). Transfers *policies*, adaptive.
3. **Reward and initial/terminal-state distribution changes (§4.2.3)** — intermediate tasks keep the dynamics but vary the reward and/or start/goal distributions (e.g. reverse curriculum / goal generation: Florensa 2017, 2018; Riedmiller 2018 SAC-X).
4. **No restrictions (§4.2.4)** — intermediate tasks may differ *arbitrarily* from the target. Three sub-approaches:
   - **MDP-based sequencing** — the **curriculum-MDP (CMDP)** formalism (detailed below).
   - **Combinatorial optimization / search** — treat sequencing as finding the best permutation of a fixed task set; black-box metaheuristics (Foglino 2019a–c).
   - **Graph-based sequencing** — build a DAG of tasks by relations (Svetlik 2017 shaping-reward graph; MacAlpine & Stone 2018).
5. **Human-in-the-loop (§4.2.5)** — how (expert and non-expert) humans design/sequence curricula (Peng 2018, Khan 2011, Stanley 2005).

#### 3.5 The curriculum-MDP (CMDP) formalism — the deepest technical idea

Narvekar et al. (2017) formalize sequencing as a **meta-MDP over the learning agent's policy space**. Two nested MDPs:
- The **base MDP**: the learning agent ("student") interacting with a task.
- The **meta-MDP / CMDP**: the curriculum agent ("teacher"). Its **state space** $S$ = the set of policies the student can represent (parameterized by the student's weights). Its **action space** $A$ = the set of tasks the student can train on next. The **transition** $p$: training the student on the chosen task updates the student's policy → a state transition in the CMDP. The **reward** $r$ = the time (steps/episodes) it took to learn the selected task.

The teacher starts at the state corresponding to a random student policy and aims to reach a *terminal state* (a student policy meeting a target-task threshold) **as fast as possible** — i.e. minimizing time-to-threshold. Matiisen et al. (2017) recast this as a **POMDP** (no access to student internal weights; observation = current score per task; reward = change in score) with the different objective of *maximizing the sum of performance over all tasks* — its "Teacher-Student" heuristic picks tasks where the absolute slope of the learning curve is highest (most progress *or* most forgetting). Narvekar & Stone (2019) show one can *learn a curriculum policy* over the CMDP via function approximation on the transfer-representation weights, mapping "current learning progress → next task," and that training each intermediate task for only a few episodes (letting the curriculum policy re-select) beats training-to-plateau. Caveat repeatedly stated: learning a curriculum is often *more expensive* than just solving the target task, and is done per-agent/per-task.

#### 3.6 Open problems (§6) — directly relevant framing

The survey's own list of gaps sharpens what a project should be cautious about: (6.1) fully automated task creation is under-studied; (6.2) transferring *different types* of knowledge between different task pairs is essentially unexplored (almost all works fix one transfer type — relevant to the project's decision to transfer only policy weights and drop belief state); (6.3) amortizing curriculum-generation cost (reuse, sim-to-real); (6.4) combining task generation + sequencing end-to-end; (6.5) **lack of theory** on *when and why* curricula help (only initial supervised-learning results via Ideal/Local Difficulty Scores); (6.6) understanding general principles of (human) curriculum design.

#### 3.7 Relevance to the project

The framework gives the project's negative curriculum result a precise diagnosis: it was a **task-level sequence curriculum** transferring **policies** *statically*, evaluated as a bet on net-positive transfer. Two of the survey's warnings apply directly — **negative transfer** (transferring a collapsed policy, cf. the Cui entropy thread, is a canonical way to poison the target) and the **weak-vs-strong-transfer accounting** (a curriculum that carried weights across a ladder but lost to a from-scratch baseline fails even the *weak* transfer bar, which is the strongest possible refutation). The §6.2 open problem — that the *type* of knowledge transferred should perhaps differ per task, and that discarding some transferable knowledge (here, the recurrent belief state) is a design choice with consequences — is the exact seam that connects to Caccia et al. (entry 22), who show carrying the recurrent state across boundaries can *beat* task-aware agents.

<a id="bb-narvekar"></a>
### Appendix: Section-by-Section Backbone

**§1 Introduction.** Quick Chess motivating example (graded mini-games → full chess). Curriculum = ordering over experience (samples or tasks). Field is scattered and inconsistently defined; goal is a systematic framework + survey + open problems. Poses the guiding questions (what is a curriculum; how to represent/evaluate; how to generate tasks; how to sequence; how to transfer).

**§2 Background.** RL preliminaries (MDP, value/action-value functions, policy search, actor-critic). **§2.2 Transfer learning**: train on source MDP(s), transfer samples/options/policies/models/value functions to target; task mappings; risk of negative transfer. **§2.3 Evaluation metrics**: time-to-threshold, asymptotic, jumpstart, total reward; weak vs. strong transfer (sunk-cost accounting; offset curves, Fig. 2).

**§3 The Curriculum Learning Method.**
- **§3.1 Curricula.** **Def. 2 (Curriculum)** = DAG $(V,E,g,\mathcal{T})$ over sample subsets, single sink. Simplifications: **Def. 3 single-task**, **Def. 4 task-level** (DAG of tasks), **Def. 5 sequence** (linear chain). Composable (task-level sequence). Online (adaptive edges) vs. offline (pre-generated).
- **§3.2 Method components.** Three parts: task generation, sequencing, transfer learning.
- **§3.3 Evaluation.** Same metrics as transfer, applied after the full curriculum vs. no curriculum; curriculum-generation cost accounting.
- **§3.4 Dimensions.** The **seven classification dimensions** (task generation / representation / transfer method / sequencer / adaptivity / evaluation metric / application area) with their value sets.

**§4 Curriculum Learning for RL Agents.**
- **§4.1 Task generation.** Create intermediate tasks so transfer is beneficial; avoid negative transfer. Parameterized-domain / task-descriptor methods (Narvekar 2016): task simplification, promising initialization, mistake learning, etc. (Table 1).
- **§4.2 Sequencing** (core; Table 2). Five families: **§4.2.1 sample sequencing** (PER Schaul 2016, CI Ren 2018, ScreenerNet Kim&Choi 2018); **§4.2.2 co-learning** (self-play, Baker 2020, Sukhbaatar 2018, Vinyals 2019); **§4.2.3 reward/initial-terminal-state changes** (Florensa 2017/2018, Riedmiller 2018); **§4.2.4 no restrictions** — MDP-based (**CMDP** Narvekar 2017; POMDP Matiisen 2017; curriculum policy Narvekar & Stone 2019), combinatorial optimization/search (metaheuristics, Foglino 2019a–c), graph-based (Svetlik 2017, MacAlpine & Stone 2018); **§4.2.5 human-in-the-loop**.
- **§4.3 Knowledge transfer.** Which transfer mechanism between curriculum tasks; freeze-and-grow (e.g. progressive nets) prevents forgetting at parameter-count cost.

**§5 Related Areas.** §5.1 related RL paradigms; §5.2 curricula in supervised ML (Bengio 2009; self-paced learning); §5.3 algorithmically designed curricula in education.

**§6 Open Questions.** 6.1 fully automated task creation; 6.2 transferring different *types* of knowledge; 6.3 reusing curricula / sim-to-real; 6.4 combining generation + sequencing; 6.5 theoretical results (IDS/LDS in supervised learning; RL analog open); 6.6 general principles of (non-expert) curriculum design.

**§7 Conclusion.** Curriculum learning = task generation + sequencing + transfer; five sequencing families surveyed; open problems as future directions; call for common terminology.

---

## 22. Caccia et al. 2022 — Task-Agnostic Continual Reinforcement Learning

**PDF:** `docs/project/references/continual_learning/sources/Caccia et al. 2022 - Task-Agnostic Continual RL.pdf`
**Full title:** "Task-Agnostic Continual Reinforcement Learning: Gaining Insights and Overcoming Challenges." **Venue:** CoLLAs 2023 (arXiv 2205.14495v3, May 2023). **Authors:** Massimo Caccia, Jonas Mueller, Taesup Kim, Laurent Charlin, Rasool Fakoor. (The primer's §6.5(4) note: this is the same paper the project critique cites under the Amazon-Science title "In Praise of a Simple Baseline.")

**Primer connection.** Primer §5 uses this as the *recurrent-state-across-task-switches* thread: in a POMDP the recurrent hidden state *is* the belief, and carrying it across task boundaries can beat *task-aware* agents. The project hard-resets the recurrent state at every task boundary — the opposite default — plausibly a self-inflicted cost stacked on top of the plasticity and entropy problems. This review supplies the formal setting (task-agnostic CRL as a hidden-mode MDP), the 3RL method, the two hypotheses, and the headline result (a task-agnostic replay+RNN agent matching or beating its *multi-task* soft-upper-bound), so that "we discard the belief state" can be evaluated as a concrete design error.

<a id="p1-caccia"></a>
### Phase 1 — Foundational Overview

**The setup.** An agent must learn a *sequence* of tasks, one after another, seeing each only once, and — the hard part — it is **not told which task it is currently on** (no task ID) or even *when* the task changes. This is **task-agnostic continual RL (TACRL)**. It must infer the task from experience while not forgetting old tasks (catastrophic forgetting) and ideally getting *better* at learning new ones (forward transfer).

**The comparison baseline.** The usual "soft upper bound" for continual learning is a **multi-task (MTL)** agent that trains on *all* tasks simultaneously *and* is told each task's ID. Conventional wisdom: MTL should always beat a continual, task-agnostic agent, because MTL has no forgetting and knows the task.

**The method — 3RL.** Caccia et al. combine two simple, well-known ingredients:
- **Experience Replay (ER)** — keep a buffer of past-task data and rehearse it (fights forgetting).
- **A recurrent network (RNN)** — feed the agent the *history* of recent (state, action, reward) tuples through a GRU, so its hidden state can *implicitly infer which task it is on* without ever being told.

The combination is **replay-based recurrent RL (3RL)**, built on Soft Actor-Critic.

**The two hypotheses.**
1. **H1** — When tasks share structure, a task-*agnostic* agent that learns to *adapt fast* can *beat* a task-*aware* agent that *memorizes* per-task solutions — especially when memorization is hard (high dimensionality, many tasks, limited data/compute).
2. **H2** — This advantage is *amplified* in continual learning, because fast-adapting agents suffer less from catastrophic forgetting than memorizing agents.

**Key findings.**
- 3RL beats other continual-RL baselines on a synthetic quadratic-optimization benchmark and on Meta-World (50 manipulation tasks; the CW10 and a new, harder MW20 subset).
- Strikingly, 3RL **matches its own multi-task soft-upper-bound** and even **surpasses the MTL equivalent** in high-dimensional settings — which the authors believe is a first for a continual method. The "upper bound" wasn't an upper bound.
- The mechanism: the RNN **reduces gradient conflict** between tasks (the dynamic task representation lets updates for different tasks interfere less), improving stability and performance.
- The **recurrent** agent matches or beats a **transformer**-based history encoder.

**Initial takeaway.** In a partially-observed, task-agnostic continual setting, the recurrent hidden state *is the belief about which task you're in*, and letting it flow across tasks (rather than resetting it) is what enables fast adaptation and reduced gradient conflict — enough to rival agents that are *told* the task. For the project, this is the direct argument that hard-resetting the recurrent state at each curriculum boundary throws away the very mechanism that would have made cross-task transfer positive.

<a id="p2-caccia"></a>
### Phase 2 — Graduate-Level Deep Dive

#### 4.1 The formal setting: TACRL as a hidden-mode MDP

Start from a **POMDP** $\langle S,A,T,X,O,r,\gamma\rangle$: an MDP augmented with observation space $X$ and emission $O(x'\mid s)$; the agent cannot read the true state $s_t$ from observation $x_t$. Split the state into an observable part $s^o_t$ and a hidden part $s^h_t$. The optimal policy must then condition on *history*:
$$\pi\big(a_t\mid s^o_{1:t},\,a_{1:t-1},\,r_{1:t-1}\big),$$
naturally parameterized by an RNN. The POMDP objective is
$$\mathbb{E}_{s^h}\Big[\,\mathbb{E}_\pi\big[\textstyle\sum_{t=0}^\infty \gamma^t r_t\big]\ \big|\ s^h\Big].$$

**TACRL is a structured special case — a hidden-mode MDP (HM-MDP)** (Choi et al. 2000): the agent has **no causal effect on $s^h$** (it cannot change which task it's in), and the hidden mode evolves as $p(s^h_{t+1}\mid s^h_t)$ independently of actions. The joint transition factorizes:
$$p\big(s^o_{t+1}\mid s^h_{t+1}, s^o_t, a_t\big)\,p\big(s^h_{t+1}\mid s^h_t\big).$$
TACRL further assumes $s^h$ follows a **non-backtracking chain**: once the chain enters a hidden state (task) it stays for a fixed number of steps, and previously visited tasks are never revisited. The hidden mode $s^h$ *is the task/context* (each context = a specific MDP).

**Evaluation — global return.** Unlike the "current return" (performance on the immediate task), TACRL evaluates the **global return**: performance averaged over *all* hidden states/tasks,
$$\mathbb{E}_{\tilde s^h}\Big[\,\mathbb{E}_\pi\big[\textstyle\sum_t\gamma^t r_t\big]\ \big|\ s^h\Big],$$
with $\tilde s^h$ the joint over all tasks. This measures how much knowledge about *all* tasks survives at the end of training — i.e. it *penalizes forgetting* by construction.

**The settings ladder (Table 1).** POMDP (most general) ⊃ HM-MDP (non-stationary hidden mode, no agent control over it) ⊃ **TACRL** (HM-MDP + non-backtracking chain + global-return evaluation). **Task-aware CRL** is TACRL made fully observable — the policy conditions on $s^h_t$ directly, $\pi(a_t\mid s^h_t, s^o_t)$, which turns the POMDP into an MDP. **MTRL** is task-aware CRL with a *stationary* hidden mode ($p(s^h_{t+1})$ independent of $s^h_t$ — all tasks always available). This ladder is the paper's cleanest contribution: it places "task ID given," "tasks all available," and "task boundaries observed" as three separable relaxations.

#### 4.2 The algorithm and architectures

**Base algorithm: Soft Actor-Critic (SAC)** — off-policy, chosen because (i) off-policy is more sample-efficient (agents spend little time per task and see each once), and (ii) off-policy *decouples the learning policy from the acting policy*, which is what makes **replay** possible. SAC learns a stochastic max-entropy policy $\pi_\phi$ and critic $Q_\theta$.

Architectures compared:
- **TaskID** — feed task ID $\tau$ as extra input: $Q_\theta(s,a,\tau)$, $\pi_\phi(a\mid s,\tau)$ (task-aware).
- **Multi-head (MH)** — shared feature extractor + one head per task; task-aware. **TAMH** (task-agnostic MH) picks the most-confident actor head (by policy entropy) and most-optimistic critic head — isolates the value of *task information* from the value of *extra capacity*.
- **RNN (task-agnostic)** — GRU history encoder producing $z_t = \mathrm{RNN}(\{(s_i,a_i,r_i)\})$, fed to $\pi_\phi(a\mid s,z)$ and $Q_\theta(s,a,z)$. Actor and critic have *their own* RNNs (as in Meta-Q-Learning / Ni et al. 2021).
- **TX (task-agnostic)** — a Transformer history encoder as an alternative to the RNN.

**3RL = ER + RNN.** As an episode unfolds, $z_t=\mathrm{RNN}(\{(s_i,a_i,r_i)\}_{i=1}^{t-1})$ should capture the task identity, helping actor and critic. Pseudocode (Algorithm 1): for each task, sample with the current RNN-conditioned policy, store transitions in buffer $D$; each update samples a batch mixing current-buffer trajectories (fraction $\approx b\cdot\min(1/n, 1-\beta)$) with old-buffer trajectories (fraction $\approx b\cdot\min((n-1)/n,\beta)$), capped by replay-cap $\beta$ (robotic experiments cap at 80% → always ≥20% compute on the current task); then flush $D$ into $D_{old}$. A key implementation nuance: capping replay strictly needs two buffers (a priori task-aware), but the same effect is obtained task-agnostically by *oversampling recently collected data*.

#### 4.3 The two hypotheses, precisely

**Hypothesis 1.** *When the reward and transition functions share structure across tasks, task-agnostic approaches can outperform task-aware ones where task memorization is difficult — high dimensionality, many tasks, or limited data/compute.* Intuition: a task-aware agent leans on the task ID to *memorize* a per-task solution (needs more data/capacity as tasks/dims grow); a task-agnostic RNN learns one *general* policy that *adapts* via task inference.

**Hypothesis 2.** *In continual learning the H1 effect is amplified, because algorithms that continually learn to adapt suffer less from catastrophic forgetting than algorithms that memorize each task.*

#### 4.4 Empirical results and the gradient-conflict mechanism

**Benchmarks.** (i) *Quadratic Optimization* — synthetic, controllable dims: reward $r(s^o,\tau) = s^{o\top}A_\tau s^o + b_\tau s^o + c_\tau$ ($A_\tau$ negative-definite → unique global max; $c_\tau$ set so all tasks share the same max reward), transition $s^o_{t+1}=s^o_t + a_t$, $a_t\in[-1,1]^d$. Lets them dial dimensionality/#tasks/#timesteps. 80,000-run random search; robustness reported as interquartile mean (IQM). (ii) *Meta-World* — 50 manipulation tasks in a shared state/action space with a shared reward structure (reaching/grasping/pushing); CW10 subset (forward-transfer-focused, 1M steps/task) and a new harder **MW20** (first 20 tasks, 500k steps/task — twice as long, half the data/compute).

**Findings.**
- **H1 supported** (Fig. 2): 3RL is more robust than ER and ER-TaskID, and its edge *grows with observation dimensionality* (memorization gets harder as dims grow), while being roughly independent of the number of tasks. TX is always ≤ RNN.
- **Mechanism — gradient conflict** (Figs. 3–4): using the *standard deviation of gradients across the minibatch* as a proxy for gradient conflict / task interference (Yu et al. 2020's PCGrad angle proxy is argued against in App. L), the RNN achieves the highest correlation with global return and the largest reduction in gradient conflict, especially in the challenging 32-task/32-dim/1M-step scenario. On Meta-World, performance correlates **-0.75** with gradient conflict and **-0.81** with training instability (Q-value variance) — both significant. Hypothesis: 3RL improves performance by *reducing gradient conflict via dynamic task representations*, which also tames the **deadly triad** (function approximation + bootstrapping + off-policy) that non-stationarity aggravates.
- **H2 supported + the headline** (Figs. 5–7): 3RL outperforms all baselines on CW10 and MW20, and **matches its MTL soft-upper-bound** — the first method the authors know of to equal a stationary-regime multi-task agent *despite* the non-stationary continual setting — and in high-dim synthetic settings *surpasses* the MTL equivalent. TaskID's relative performance *drops* with dimensionality in CL, exactly as H1/H2 predict.
- **Controls** (appendices): the gain is *not* from parameter count (App. G), *not* from task-awareness+RNN combination (App. N — adding task-awareness to the RNN does not help), *not* single-task improvement (App. H — RNN doesn't help single-task), *not* parameter stability (App. I). Support that the RNN *places new tasks in the context of previous ones* (forward transfer), backed by the PCA of RNN representations in Fig. 1 (task-invariant initialization + richer, evolving representations).

#### 4.5 Relevance to the project

The paper's core message is mechanistic and directly transferable: in a partially-observed, non-stationary (continual) setting, **the recurrent hidden state is the belief over the current task**, and letting it carry information *across* task boundaries (a) performs implicit task inference without a task ID and (b) *reduces gradient conflict*, which is what lets a continual agent rival a task-aware multi-task agent. The project's decision to **hard-reset the recurrent state at each curriculum boundary** is, in this light, discarding the belief precisely at the moment it is most informative (the task just changed) — throwing away both the task-inference signal and the gradient-conflict-reduction benefit. Combined with the Narvekar framing (entry 21: the project transferred *policy weights* but omitted the belief state — a §6.2 "wrong knowledge type transferred" error) and the Cui thread (entry 20: a transferred *collapsed* policy is a canonical negative-transfer poison), the three adjacent threads converge on a coherent diagnosis of the project's curriculum failure. Caveat for transfer: Caccia et al. use *off-policy SAC + replay*; the project's on-policy recurrent PPO differs, though the authors note (footnote 4) their findings extend to any method with a replay buffer, and the belief-state argument is algorithm-independent.

<a id="bb-caccia"></a>
### Appendix: Section-by-Section Backbone

**Abstract.** Investigates why task-agnostic CL differs in performance from multi-task (MTL) agents. Two hypotheses (task-agnosticity helps under limited data/compute/high-dim; fast adaptation mitigates forgetting). Introduces **3RL** (replay-based recurrent RL). Tested on synthetic + Meta-World (50 tasks); 3RL beats baselines and even surpasses its MTL equivalent in high-dim; recurrent ≥ transformer.

**§1 Introduction.** CL agents learn a task sequence without forgetting; MTL (task-aware, all tasks jointly) is the usual soft upper bound; task-agnostic CL (no task ID) is the practical, harder setting. Reasoning: task-agnostic methods learn to *adapt* (generalize), task-aware methods *memorize* (need more data/compute). States H1 and H2; instantiates task-aware and task-agnostic methods; adds RNN memory + replay = 3RL. Evaluated on synthetic quadratic + Meta-World.

**§2 Background & TACRL.** MDP and POMDP definitions; history-conditioned policy $\pi(a_t\mid s^o_{1:t},a_{1:t-1},r_{1:t-1})$ (→ RNN). **TACRL** = HM-MDP (agent has no causal effect on hidden mode $s^h$; $p(s^h_{t+1}\mid s^h_t)$) with a **non-backtracking chain** and **global-return** evaluation. Task awareness = full observability (POMDP→MDP). MTL trains on all tasks jointly (stationary distribution) — soft upper bound but often impractical. **Table 1** ladders MDP / POMDP / HM-MDP / TACRL / Task-Aware CRL / MTRL by transition, policy conditioning, objective, evaluation.

**§3 Methods & Hypotheses.**
- **§3.1 Algorithms.** Off-policy chosen (sample efficiency + replay support). Base = **SAC** (actor $\pi_\phi$, critic $Q_\theta$).
- **§3.2 Models.** **TaskID** (feed $\tau$); **Multi-head MH** (one head/task; task-aware) and **TAMH** (task-agnostic head selection by entropy/optimism — isolates task-info from capacity); **RNN** (GRU history encoder → $z$, task-agnostic; separate actor/critic RNNs); **TX** (transformer history encoder). **Hypothesis 1** stated. Robot-manipulation analogy (memorize per object vs. adapt).
- **§3.3 Baselines.** FineTuning (no forgetting-prevention), **ER** (replay; capped by oversampling current task, Alg. 1 L8-9), **MTL** (soft upper bound), **Independent** (separate model/task, no transfer). Combos (MTL-TaskID, FineTuning-MH, ...). **3RL = ER + RNN** (Algorithm 1). **Hypothesis 2** stated.

**§4 Empirical Findings.** Benchmarks: **Quadratic Optimization** (synthetic, $r=s^{o\top}A_\tau s^o+b_\tau s^o+c_\tau$, controllable dims) and **Meta-World** (CW10 1M steps/task; new **MW20** 500k steps/task, harder). Metrics: global vs current return/success; IQM over 80k runs; top-10% for maximal.
- **§4.1 Hypothesis 1.** 3RL most robust, edge grows with dimensionality (Fig. 2); TX ≤ RNN. Gradient-conflict proxy = std of gradients across minibatch; RNN reduces conflict most in the challenging 32-task/32-dim/1M scenario (Fig. 3). Meta-World: performance vs conflict corr -0.75, vs instability -0.81 (Fig. 4). Deadly-triad framing. 3RL beats all on CW10/MW20 (Fig. 5).
- **§4.2 CL vs MTL.** TaskID's relative performance drops with dims in CL; 3RL *surpasses* MTL analog in high-dim synthetic (Fig. 6); on MW20, **3RL matches its MTRL soft-upper-bound** — claimed first (Fig. 7). Appendix controls: not parameters (G), not task-aware+RNN (N), not single-task gain (H), not parameter stability (I); RNN contextualizes new tasks (J, Fig. 1 PCA).

**§5 Related Work.** CRL (Continual World: forgetting-reduction methods lose transfer); task-agnostic CL upper-bound comparisons; TACRL methods (GP mixtures, bandit policy retrieval, meta-learning); RNNs in continual supervised learning and in POMDP RL; transformers in RL. Novelty: RNN within TACRL combined with ER.

**§6 Conclusion.** Task-agnosticity can beat task-aware MTL in resource-constrained / high-dim / multi-task regimes; 3RL matches or surpasses MTL; mechanism partly = reduced gradient conflict; challenges the assumption that task-agnostic CL is inherently harder.

---

## Cross-Paper Synthesis

This section folds the five per-shard syntheses into one corpus-wide reading. It states where the 22 papers **agree** (consensus), where they genuinely **disagree** (and why), and which **gaps** the project's own experiments could fill. Entry numbers refer to the sections above.

### 1. The organizing map: two failure axes, not one

Every paper here lives on a 2×2 map with two orthogonal axes:

- **Backward axis — retention (catastrophic forgetting):** does learning something new destroy what was already known?
- **Forward axis — plasticity (loss of plasticity):** can the network still learn *anything* new, as fast as a fresh network would?

French (1) named the backward axis and diagnosed forgetting as the price of *overlapping distributed representations* — the same shared weights that give generalization are what get overwritten. The whole Phase-1 corpus (entries 1–6) attacks that backward axis. The pivotal conceptual move of the corpus is the discovery (entries 7–18) that the **forward axis is a separate disease**: Abbas (12) shows a network can still have a *large* training loss yet be unable to reduce it — its weights have frozen. Berariu (9) sharpens the warning with a definitional split the whole project must respect: "loss of plasticity" names *two* phenomena — a **generalization-type** loss (the network still drives training error to zero but converges to a worse-generalizing minimum) and an **optimization-type** loss (it can no longer reduce training error at all) — and a citation about one is not evidence about the other. The consensus: forgetting and plasticity are distinct, and a real continual learner must solve both (Abbas §6 is explicit that its plasticity cure, CReLU, does nothing for forgetting).

### 2. Phase 1 consensus — three forgetting families, one dilemma

The six Phase-1 papers are six points on French's stability–plasticity dial, cleanly separable into three families that recur through the rest of the corpus:

| Entry | Family | Escape mechanism | Old-task protection | Cost / limit |
|---|---|---|---|---|
| 1 French 1999 | (framing) | orthogonalize / sparsify / dual-nets / rehearsal | — (survey) | trade-off with generalization |
| 2 Rusu (Prog. nets) | parameter isolation | freeze old column, grow new + lateral links | **exact** ($\infty$ stiffness) | parameters grow with #tasks; need task label |
| 3 Kirkpatrick (EWC) | regularization | quadratic Fisher-weighted anchor to $\theta_A^*$ | soft (stiffness $\lambda F_i$) | Laplace point estimate under-estimates uncertainty |
| 4 Zenke (SI) | regularization | quadratic path-integral-weighted anchor | soft (stiffness $\Omega_k^\mu$) | trajectory importance = Hessian only for quadratic loss |
| 5 Rebuffi (iCaRL) | replay | exemplar rehearsal + distillation + nearest-mean | invariance (distillation) | stores images; below joint-training accuracy |
| 6 Lopez-Paz (GEM) | replay | gradient projection vs. stored past gradients | inequality (loss non-increasing) | per-step per-task gradient computation |

Two lineages inside this table matter downstream. **Regularization (EWC → SI):** identical penalty *form* — a per-weight quadratic pulling weights toward their old values — differing only in the importance weight (endpoint Fisher vs. trajectory path-integral); SI's Hessian theorem proves the two importances coincide in the tractable quadratic case, explaining their comparable performance. **Replay (French's rehearsal/pseudopatterns → iCaRL → GEM):** store a little old data and reuse it; iCaRL rehearses+distills to keep outputs *invariant* (forbidding backward transfer), whereas GEM uses the same stored data only as *gradient constraints* ("never increase old loss, but allow it to fall"), which uniquely permits *positive* backward transfer. GEM's backward-/forward-transfer metrics (BWT/FWT) are the field's tool for measuring exactly what these methods trade, and are the natural instrument for the project's own curriculum accounting. Ash & Adams' shrink-and-perturb (7) — $\theta \leftarrow \lambda\theta + \epsilon$ — is the supervised *partial-reset* primitive that the reset family (§4 below) descends from.

### 3. Phase 2→3 consensus — the plasticity mechanism, assembled

The Phase-2/3 papers, read together, assemble a single causal chain from "what the optimizer does" to "the agent stops learning":

**Non-stationary bootstrap targets** (the value target moves as the policy improves — present in every value-based agent) drive an **activation-footprint collapse** (Abbas 12: eventually <1% of ReLU units fire), whose atomic unit is the **dormant neuron** (Sokar 13: a unit whose normalized mean activation falls below a threshold $\tau$, and which — critically — *stays* dormant). A dead ReLU passes exactly zero gradient to its incoming weights (the chain-rule dead-unit argument, Abbas §4.3), so **gradients collapse** ($\ell_0/\ell_1$ norms → 0), **weight change collapses** (Abbas: ~20% of first-visit magnitude by visit 10), and the learner stalls *despite a large loss*. Kumar (8) supplies the parallel representational story — **feature-rank collapse** from bootstrapping's repeated self-regression (a 512-d feature layer ends up using 20–100 directions) — and Lyle 2022 (10) the **capacity-loss** operationalization (measure how well a checkpoint can fit a *fresh random target*; it declines over training) plus the sparse-reward danger zone. The rate of this whole collapse **rises with the replay ratio** (updates per environment step): Sokar Fig. 7 and Nikishin-2023 Fig. 5 both show more updates burn plasticity faster — which is *why* cranking the replay ratio collapses standard agents (the **replay-ratio barrier**, D'Oro 15).

The three theory-heavy papers (Lyle 2023/16, Lyle 2024/17, Dohare 2024/18) converge on the **same set of correlates**: (1) dead/dormant/saturated units, (2) growing weight/parameter norm → rising loss-landscape sharpness (top Hessian eigenvalue) → shrinking stable learning rate → crawling progress, (3) representation rank collapse (feature rank / stable rank / effective rank / empirical-NTK degeneracy toward diagonal-plus-rank-1). Lyle 2023 frames these as correlates that *individually fail a causal (sign-consistency) test*; Dohare treats them as *jointly explanatory*; Lyle 2024 reconciles by making each a distinct *mechanism* (adding the newly-named **unit linearization / "zombie units"** — units that collapse to an effectively linear map, invisible to dead-unit counters — and the **regression-target-magnitude** mechanism, whereby simply regressing on large-mean targets, even stationary ones, poisons the features via a bias-encoding singular-value blowup) and showing all mechanisms terminate in the same eNTK collapse.

### 4. The reset lineage: Nikishin 2022 → D'Oro 2023 → Dohare 2024

The single most load-bearing thread for the project is the evolution of **resetting** from a pathology-avoidance trick into a general principle of sustained learning:

- **Nikishin et al. 2022 (11) — resets as a first-class RL tool.** Diagnoses the *primacy bias* (deep RL overfits its earliest experiences and can't leverage better data collected later) and shows the data is fine but the *learner* is broken (a fresh agent handed the failed agent's buffer learns rapidly). Fix: periodically **re-initialize the last few layers while keeping the replay buffer intact**. Crucially, the benefit *grows with the replay ratio* — explicitly "the seed of D'Oro's replay-ratio barrier."
- **D'Oro et al. 2023 (15) — resets as a scaling lever.** Promotes the reset from "fix" to "sample-efficiency knob": with periodic resets, the replay-ratio barrier lifts and you can push the replay ratio up an order of magnitude (SR-SAC to 128, SR-SPR to 16, both new model-free SOTA). The load-bearing design rule: **tie the reset schedule to the number of *updates*, not environment steps** — a fixed env-step interval breaks above replay ratio 4. D'Oro cites Nikishin 2022 as its reset backbone.
- **Dohare et al. 2024 (18) — the reset made perpetual and selective.** *Continual backpropagation* is the limiting case: instead of periodically wiping whole layers, on *every* step reinitialize a *tiny fraction of the least-used units* (by a contribution-utility measure), zeroing their outgoing weights so the output is unperturbed at the moment of replacement. This maintains plasticity apparently indefinitely, and reframes the whole idea philosophically: "gradient descent is inherently variability-destroying; sustained learning needs a random, non-gradient (variation-and-selection) component."

Read as a lineage: **periodic partial reset (Nikishin) → update-indexed reset as a scaling knob (D'Oro) → perpetual, selective, output-preserving reset (Dohare).** Ash & Adams' shrink-and-perturb (7) is the supervised ancestor of all three; Sokar's ReDo (13) is the *dormancy-triggered* surgical sibling; Nikishin 2023's plasticity injection (14) is the *additive* relative (freeze the old head, append a zero-initialized fresh trainable one — also doubling as a clean **diagnostic** that separates "can't learn" from "can't explore").

### 5. Abbas's four-signal forensic chain and CReLU as its activation-level fix

Abbas (12) is the closest published precedent to the project's own curriculum setup — a single network carried across a sequence of RL tasks with no reset, degrading *below* a from-scratch baseline — and it supplies the corpus's clearest forensic method: a **four-signal causal chain** measured across task revisits, (i) **activation collapse** (fewer ReLU units fire, → <1% in the value/advantage heads) → (ii) **gradient collapse** (dead ReLUs pass zero gradient; $\ell_0/\ell_1$ decay to near zero) → (iii) **weight-change collapse** (~20% of first-visit magnitude) → (iv) **learning stalls despite a large, even growing, loss**. This is the operational signature the project should instrument on any network trained under non-stationarity.

Abbas's own fix acts at the *activation level*: **Concatenated ReLU (CReLU)**, $\mathrm{CReLU}(x) = [\mathrm{ReLU}(x),\ \mathrm{ReLU}(-x)]$, guarantees that for any non-zero pre-activation exactly one of the two channels is live — so a unit *cannot* become one-sided-dead, and dormancy never accumulates. Shang et al. 2016 (19) is the original CReLU paper (a supervised-vision paper that never mentions plasticity) and it supplies the *mechanistic why*: CReLU losslessly preserves the pre-activation ($x = [x]_+ - [-x]_+$), so it (a) keeps a guaranteed-live companion channel per unit and (b) keeps negative-phase feature directions distinct instead of collapsing them to zero — attacking both the dormancy and the rank-collapse ends of the mechanism. Sokar's ReDo (13) is the complementary *weight-level* fix for the same dormant unit: measure it precisely (the score $s_i^\ell \le \tau$) and recycle only the units that have actually gone dormant. So the three connect as **Abbas (measures the collapse) → Shang (explains why the activation-level fix works) → Sokar (names the atomic unit and recycles it surgically)**.

### 6. The genuine disagreement: layer-norm as cure vs. normalization as harm

The corpus is not unanimous, and the sharpest live disagreement is worth stating without papering over it:

- **Lyle et al. 2024 (17)** makes **layer normalization + L2 weight decay** (plus a scale-invariant / categorical output when target magnitudes grow) the headline, near-complete cure — LN fixes the preactivation-distribution-shift mechanism, L2 caps the weight-norm-growth mechanism, and the two are independent and additive. Lyle 2023 (16) had already established LN as the field's first-line defense, beating resetting, weight decay, spectral norm, and shrink-and-perturb across MLP/CNN/ResNet/ViT, and improving Double-DQN across all 57 Atari games with no retuning.
- **Dohare et al. 2024 (18)** reports the opposite in its regime: Adam, Dropout, and **(online/batch) normalization *increased* plasticity loss** in its single-pass online continual settings, and it champions **selective reinitialization (continual backprop)** instead.

**Why they diverge (and it is not a contradiction so much as a regime boundary):** the two camps are testing different things under the same word. Dohare's "normalization" is *online/batch* normalization applied in a *single-pass, no-minibatch, online* regime; Lyle's is *layer* normalization in a *minibatch, value-based-RL* regime. They **agree** on the two things that most matter — *weight-norm control* (L2 helps in both) and *preserving representation diversity/rank* — and differ only on whether *normalization* or *reinitialization* is the better primary lever for the second. **Recommendation for the project:** treat this as an empirical question in the project's *own* setting rather than assuming either recipe transfers. Because the project runs minibatch, value-/actor-based RL (recurrent PPO), Lyle's regime is the closer analogue, so **layer-norm + weight-decay is the sensible cheap first-line guardrail** — but the project should *instrument* dormant-unit fraction, effective rank, and weight norm (the shared diagnostics) so it can detect if Dohare's online-regime caveat bites, in which case selective reinitialization (ReDo / continual backprop) is the fallback.

### 7. The corrective menu compared

All the Phase-3 fixes restore trainable diversity; they differ in *what* they touch and *when*:

| Method | Entry | Family | What it touches | When | Output preserved at intervention? | Also fixes forgetting? |
|---|---|---|---|---|---|---|
| CReLU | 12 / 19 | architectural (preventive) | activation function (guarantees a live channel) | always-on | n/a (design choice) | No (Abbas §6 explicit) |
| ReDo | 13 | recycle (corrective, surgical) | incoming weights of *dormant* units; zero outgoing | periodic ($t \bmod F$) | Yes (zeroed outgoing) | No |
| Plasticity injection | 14 | additive (corrective + diagnostic) | append fresh frozen/trainable head pair | once / on plateau | Yes (by construction) | No |
| SR resets | 15 | reset (corrective → scaling knob) | whole net (final layers full; encoder soft) | update-indexed periodic | No (abrupt; buffer relearns) | No |
| LayerNorm + L2 | 16 / 17 | landscape-smoothing (preventive) | preactivation stats + weight norm | always-on | n/a | No |
| Continual backprop | 18 | perpetual selective reset (preventive) | least-utility units; zero outgoing | every step, tiny fraction | Yes (zeroed outgoing) | No |

The recurring design principle across the output-preserving methods (ReDo, injection, continual backprop): **zero the outgoing weights of any freshly-reset/added unit** so the network's function is unchanged at the instant of intervention, and the unit re-earns influence only through subsequent gradient descent. Note the whole menu addresses *plasticity only* — none of these fixes forgetting, consistent with the Phase-1/Phase-3 division of §1.

### 8. The project's curriculum-failure convergence (Narvekar × Cui × Caccia)

The three "adjacent threads" papers were reviewed for a reason: read together, they converge from three independent directions on a single, coherent diagnosis of *why the project's difficulty-ladder curriculum agent underperformed a from-scratch baseline*.

- **Narvekar 2020 (21) — the wrong *knowledge type* was transferred.** A curriculum is a bet with three independently-fallible parts (task generation, sequencing, transfer). In this survey's taxonomy the project ran a *task-level sequence curriculum* transferring *policy weights* *statically*. Its §6.2 open problem — that the *type* of knowledge transferred should perhaps differ per task, and that discarding transferable knowledge is a design choice with consequences — pinpoints the error: the project carried the policy weights but *discarded the recurrent belief-state*. It also fails even the *weak-transfer* accounting bar (a curriculum that carried weights across a ladder but lost to a from-scratch learner), the strongest possible refutation.
- **Cui 2025 (20) — the thing transferred was a *collapsed* policy.** Policy-entropy collapse is a *covariance-driven near-absorbing* dynamic ($-\mathrm{d}H \propto \mathrm{Cov}(\log\pi,\ \pi\!\cdot\!A)$): a token/action that is already high-probability *and* high-advantage gets reinforced, driving entropy monotonically to zero. Transferring such a collapsed, near-deterministic policy into a new curriculum stage is a **canonical negative transfer** — the "eat-once-then-starve" degeneration read as high-covariance reinforcement of an early-dominant action. The remedy is *targeted* suppression of the pivotal high-covariance actions, not a blunt global entropy bonus (which is shown to fail).
- **Caccia 2022 (22) — hard-resetting the belief-state discards the mechanism that makes transfer positive.** In a partially-observed, non-stationary setting the recurrent hidden state *is the belief over which task you are in*; carrying it *across* task boundaries performs implicit task inference (no task ID needed) **and** reduces gradient conflict between tasks — enough that a task-*agnostic* replay+RNN agent (3RL) matches or even surpasses its *task-aware multi-task* soft-upper-bound. The project's decision to **hard-reset the recurrent state at every curriculum boundary** throws away the belief precisely when it is most informative (the task just changed), discarding both the task-inference signal and the gradient-conflict-reduction benefit.

**The convergence:** Narvekar says *the wrong knowledge type was transferred* (weights kept, belief dropped); Cui says *what was transferred was poison* (a collapsed policy); Caccia says *the discarded thing was exactly the mechanism that would have made transfer positive* (the recurrent belief-state). Three angles, one diagnosis.

### 9. Gaps, and handshakes to the project's own work

What the corpus does *not* settle — and where the project's experiments could contribute:

- **The normalization regime boundary (§6) is unresolved for on-policy recurrent PPO.** Both Lyle and Dohare validated on off-policy value-based agents (DQN/SAC) or single-pass online supervised learning; neither tested the project's on-policy, recurrent, minibatch regime. Whether LN+weight-decay or selective reinitialization is the better first-line lever there is a genuinely open, project-answerable question.
- **The reset design rules (§4) transfer as concrete, testable levers.** The two most transplantable are D'Oro's **update-indexed** reset schedule (not env-step-indexed) and the "**fully reset final layers, soft-reset (shrink-and-perturb, $\alpha \approx 0.8$) the encoder**" recipe — directly relevant to the project's replay-ratio speed-vs-performance study ([replay_ratio_speed_vs_performance](../../concepts/replay_ratio_speed_vs_performance.md)); Sokar's ReDo is the surgical alternative that avoids the post-reset re-exploration dip.
- **The forensic instrumentation (§3, §5) is cheap and should be standing.** Dormant-unit fraction (Sokar), effective/stable rank (Kumar, Lyle, Dohare), target-fitting capacity (Lyle 2022), Fisher-of-activations sensitivity (Rusu's AFS/APS), and the empirical-NTK collapse fingerprint (Lyle 2024) are all inexpensive diagnostics to log whenever a network trains under non-stationarity — which any RL agent does.
- **Project pointers.** The corpus exists in part to explain the project's null-result diagnosis series ([NMN_PERFORMANCE_DIAGNOSIS](../../../develop/INDEX.md)) and its curriculum-underperformed-baseline finding; the plasticity mechanism (§3) is a leading candidate explanation for a network that trains but does not improve. Where a synthesis result implies a specific project gate (G1/G2 in [project_plan.md](../../project_plan.md) §4) or hypothesis (H1–H5 in [NEUROMODULATION_ALGORITHM.md](../../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) §1.4), those docs own the definitions; this review supplies the literature backing, not the gate criteria.

*Recommendations that touch code or config — adding a ReDo hook, a reset schedule, layer-norm, or belief-state carry-over to the trainer — are out of scope for this review and should be handed to `senior-developer` for an `issue_plan` or to `experiment-designer` for a characterization plan.*

---

*End of master review. All 22 per-paper reviews above retain their original Phase 1 / Phase 2 / Section-by-Section Backbone as produced by `literature-reviewer`; this curator layer added the entry point, the unified phase-ordered TOC, and this corpus-wide synthesis, and removed the five now-redundant per-shard synthesis blocks.*
