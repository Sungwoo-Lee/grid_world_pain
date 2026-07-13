> **Per-paper review — continual-learning corpus, paper 2 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§2); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 2. Rusu et al. (2016) — Progressive Neural Networks

**PDF:** `docs/project/references/continual_learning/sources/Rusu et al. 2016 - Progressive Neural Networks.pdf`
**Venue:** arXiv:1606.04671 (DeepMind). **Type:** architecture / empirical (deep RL).
**Primer link:** the canonical **parameter-isolation** family member in Phase 1 — forgetting made *structurally impossible* by freezing old weights and growing new capacity.

## Phase 1: Foundational Overview

**Introduction (plain language).** The standard way to reuse a trained network on a new task is *finetuning*: copy the weights, swap the output layer, keep training. But finetuning is *destructive* — the new task overwrites the old function, so you cannot go back to the first task, and you must guess which prior model to start from. Progressive Neural Networks refuse to overwrite anything. For each new task they **freeze** the entire previous network (a "column") and **add a brand-new column** trained from scratch, wiring in **lateral connections** so the new column can *read* the frozen features of every previous column. Because old weights never change, the old tasks are remembered perfectly — forgetting is impossible *by construction* — while the new column still benefits from prior knowledge through those lateral links.

**Key finding.** On deep-RL benchmarks — synthetic Pong variants ("Pong Soup"), random sequences of Atari games, and 3D maze ("Labyrinth") foraging — progressive nets beat the standard transfer baselines (finetuning) on both mean and median transfer score, *and* avoid finetuning's failure mode of *negative* transfer on incompatible tasks (they can simply ignore useless prior features). A companion **Average Fisher Sensitivity (AFS)** analysis shows transfer really does route through the lateral connections, and reveals *where*: low-level vision often transfers, task-specific control layers get relearned.

**Initial takeaway.** Progressive nets are a clean existence proof that transfer *and* zero-forgetting can coexist in deep RL. The catch is scaling: parameters grow with the number of tasks (linear in width, quadratic in parameters), and at inference you must know the task label to pick the column. But the AFS analysis shows each added column uses only a *fraction* of its capacity, pointing to pruning/compression as the fix.

## Phase 2: Graduate-Level Deep Dive

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

## Appendix: Section-by-Section Backbone

- **Abstract.** Progressive nets: immune to forgetting, transfer via lateral connections to previously-learned features; evaluated on Atari + 3D maze RL; beat pretrain/finetune baselines; novel Fisher-based sensitivity measure shows transfer at both sensory and control layers.
- **§1 Introduction.** Finetuning is destructive and needs foreknowledge of which model to init from; distillation needs persistent data for all tasks. Progressive nets keep a pool of pretrained columns and learn lateral connections into them — prior knowledge is non-transient, compositional, immune to forgetting. Three contributions: novel combination for task sequences, extensive deep-RL evaluation, Fisher+perturbation transfer analysis.
- **§2 Progressive Networks.** Column definition; freeze old $\Theta^{(j)}$, add random-init column per task; Eq. (1) recurrence with $W$ (own) and $U$ (lateral) weights, ReLU. Design goals: solve K tasks, accelerate via transfer, avoid forgetting. No assumption of task overlap (may be orthogonal/adversarial). Forgetting impossibility argument (lateral only $j<k$; frozen params). RL application: column = policy for one MDP. Adapters Eq. (2): scaled MLP lateral connection with projection matrix $V$ and learned scalar $\alpha$; $1\times1$ conv for conv layers. Limitations: parameter growth; task label needed at inference.
- **§3 Transfer Analysis.** APS (perturbation, slow) and AFS (Fisher, fast). AFS: diagonal Fisher of policy w.r.t. normalized activations; per-feature and per-layer normalized scores.
- **§4 Related Literature.** Transfer/multitask RL (actor-mimic, policy distillation); constructive architectures (cascade-correlation, incremental autoencoders); multi-column nets. Progressive uses lateral connections for deep compositionality.
- **§5 Experiments.** Setup: A3C, 16 workers, top-3-of-25 jobs, AUC scoring, transfer score vs. baseline 1. Baselines 1–4 (Fig. 3). §5.2 Pong Soup (Noisy/Black/White/Zoom/flips): baseline 2 negative transfer, baseline 3 strong, progressive beats it; AFS shows conv reuse on H-flip, new mid-vision on Zoom. §5.3 Atari (Pong/River Raid/Seaquest → 12 targets): 2/3/4 columns; positive transfer 8/12; sweet-spot AFS finding. §5.4 Labyrinth 3D foraging (apples/strawberries + / mushrooms/lemons −): progressive best; baseline 2 negative even on easy levels (can't relearn changing reward-item vision).
- **§6 Conclusion.** First demonstration of positive transfer in deep-RL continual learning; robust to harmful features; transfer grows with columns; constructive not destructive.
- **Supplement.** A: Perturbation analysis details, $\Lambda = 1/\sigma^2$ at 50% drop, APS Eq. (3); APS≈AFS. B: Compressibility — AFS spectra sparsen with more columns; new columns' features less important; pruning feasible. C: Setup details (hyperparameter grid; 3 conv layers, 12 maps each; 256 FC units; RMSProp; $1.6\times10^8$ env steps / $4\times10^7$ agent steps with action-repeat 4). D: per-game learning curves. E: Labyrinth level descriptions.

---
