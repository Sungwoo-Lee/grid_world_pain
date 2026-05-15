---
title: "NEST: A Neuromodulated Small-world Hypergraph Trajectory Prediction Model for Autonomous Driving"
authors: "Chengyue Wang, Haicheng Liao, Bonan Wang, Yanchen Guan, Bin Rao, Ziyuan Pu, Zhiyong Cui, Cheng-Zhong Xu, Zhenning Li"
year: 2025
venue: "AAAI-25 (Thirty-Ninth AAAI Conference on Artificial Intelligence)"
slug: "wang_2025_nest_hypergraph"
source_pdf: "sources/Wang et al. 2025 - NEST - A Neuromodulated Small-world Hypergraph Trajectory Prediction model for autonomous driving.pdf"
topic: "neuromodulatory_algorithms"
---

## Plain-English entry point

This paper is the most applied of the corpus. It addresses **trajectory prediction for autonomous driving** — given the past few seconds of how vehicles, pedestrians, and cyclists were moving, predict where each of them will be over the next few seconds. The authors note that traffic interactions are not just pair-to-pair (vehicle A reacts to vehicle B); they are *group-wise* (a convoy merges into a lane, a cluster of pedestrians crosses) and *long-range* (a braking lead vehicle can affect cars several positions back through a chain). They propose **NEST** (Neuromodulated Small-world Hypergraph Trajectory Prediction), which models traffic agents as nodes in a **hypergraph** — a graph where a single edge ("hyperedge") can connect more than two nodes, naturally representing a group — and uses a **small-world** topology to allow long-range "shortcut" connections. The "neuromodulation" part is the lightest of the seven papers in this corpus: it's just two small MLPs that compute two parameters, $\alpha$ (a clustering threshold) and $\beta$ (a long-range connection probability), from the agent features. The two parameters control the hypergraph structure on each prediction step, so the graph dynamically reshapes to current traffic conditions.

Why it matters as part of this corpus: NEST cites Vecoven 2020 directly, claiming inspiration from "cellular neuroregulation mechanisms" for its adaptive component. The mechanism is much shallower than Vecoven's — there's no parametric activation function modulation, no shared latent context vector $z$ propagated through layers; it's just two scalar/vector gating parameters $\alpha, \beta$ that reshape the graph. Conceptually the link is "we use a small adaptive subnetwork to make a downstream computation context-dependent", which is the broad shape of neuromodulation. On trajectory prediction benchmarks (nuScenes, MoCAD, HighD), NEST beats prior state-of-the-art (BAT, STDAN, WSiP, BAT-25%, Trajectron++, MultiPath, etc.) on standard metrics (minADE, minFDE, RMSE).

## Section-by-section backbone

### Abstract
Trajectory prediction is essential for autonomous-driving safety. Existing models struggle with real-time processing, non-linearity, dense traffic, and temporal interaction dynamics. NEST integrates **Small-world Networks** and **hypergraphs** for interaction modelling, with a **Neuromodulator** component that adapts the graph dynamically to traffic conditions. Validated on nuScenes, MoCAD, and HighD.

### Introduction
Trajectory prediction failure modes in existing models: (1) static snapshots of traffic miss temporal evolution; (2) non-linearity and chaos (sudden stops, erratic driving, pedestrians) are oversimplified; (3) inefficient in dense heterogeneous traffic; (4) predefined relationships fail to capture diverse interactions. NEST addresses all four. Three contributions: (i) Small-world Network with Neuromodulator captures local + long-range interactions; (ii) novel Hypergraph Neural Network (HGNN) for interaction learning; (iii) extensive validation on real-world datasets.

### Related work
Trajectory prediction has moved through: RNN/LSTM time-series approaches (Social LSTM, Alahi 2016); spatial+temporal attention (Nettraj); attention-based models (BAT, STDAN, MHA-LSTM); GNN-based models (SFEM-GCN, MTP-GO, Social Soft Attention GCN). GNN limitations: focus on pair-wise relations, miss group dynamics, computationally constrained in dense urban traffic.

### Methodology — Problem formulation
Given historical data $X = [X_0, X_1, \ldots, X_n]$ (positions/velocities/accelerations for target agent $X_0$ + surrounding agents $X_{1..n}$ over past $t_h$ time-steps) and HD map $M$, predict future trajectory $Y$ of target agent over horizon $t_f$ as $K$ multi-modal predictions $Y = [Y_1, \ldots, Y_K]$ each with probability $P_i$. Each $Y_i^t = [x_i^t, y_i^t, b_{i,x}^t, b_{i,y}^t]$ are Laplace-distribution parameters (mean coordinate and scale).

### Model overview (Figure 2)
Four modules:
1. **Hypergraph Forming** — builds interaction hypergraph $G$ with Neuromodulator + Small-world Network.
2. **Hypergraph Pooling** — extracts interaction features $F_i$ via vertex-to-hyperedge and hyperedge-to-vertex pooling.
3. **Context Fusion** — combines lane features $F_l$ (from HD maps) with $F_i$ → context feature $F_c$.
4. **Multi-modal Predictor** — $K$ generators produce $K$ candidate trajectories with intention probabilities.

### Hypergraph Forming
The interaction hypergraph $G = (V, E)$ has vertex feature set $V = F_a \in \mathbb{R}^{(n+1) \times d}$ (agent features from a Transformer encoder, including the target) and hyperedge set $E$ (group-wise interactions).

**Small-world Network (Newman–Watts model).** First, compute clustering coefficient $C_{i,j}$ between vertex $i$ and hyperedge $j$. Apply threshold:

$$C_{i,j} = \begin{cases} 1 & \text{if } C_{i,j} \ge \alpha \\ 0 & \text{otherwise} \end{cases} \tag{1}$$

Then add long-range shortcut connections via a small-world rewiring rule:

$$E_{i,j} = \begin{cases} 1 & \text{if } C_{i,j} = 1, \text{or } (C_{i,j} = 0 \text{ and } \eta \le \beta) \\ 0 & \text{otherwise} \end{cases} \tag{2}$$

where $\eta \sim U[0, 1]$.

**Neuromodulator.** Both $\alpha$ and $\beta$ are produced by small MLPs:

$$\alpha = \Pi_\alpha(C), \qquad \beta = \Pi_\beta(F_a) \tag{4, 5}$$

with $\Pi_\alpha$ a two-MLP-layer module projecting the clustering-coefficient matrix $C$ to a threshold value in $[0,1]$ (via sigmoid), and $\Pi_\beta$ projecting the agent feature $F_a$ to a connection probability in $[0,1]$.

The hyperedge set is then $E = \Omega(V, \alpha, \beta)$ where $\Omega$ is the Newman–Watts small-world generator.

### Hypergraph Pooling
Iterative pooling between vertices and hyperedges:
- **Vertex-to-Hyperedge** aggregates: personality $I_p = M_p(\sum \lambda_i V_i)$, intention $I_i = \sigma((M_i(I_a) + \xi) / \tau)$ (Gumbel-softmax over $K$ intentions), willingness $I_w = M_w(I_a)$. Group feature $F_{g,j} = $ concat($I_p, I_i, I_w$).
- **Hyperedge-to-Vertex** uses $F_{g,j}$ to update vertex features. After $H$ iterations, average across vertices to get interaction feature $F_i$.

### Context Fusion
Lane encoder produces $F_l$ from HD map $M$. Attention:

$$F_c = \text{Attn}(Q = F_i, K = F_l, V = F_l) \tag{10}$$

### Multi-modal Predictor
$K$ generators (one per intention mode) produce future Laplace-distribution parameters.

### Experiments
- **nuScenes** (urban): NEST beats DLow-AF, Trajectron++, MultiPath, LaPred, LDS-AF, AgentFormer, GoHome, STGM, EMSIN, SeFlow on minADE5 (1.18), minADE1 (2.97), minFDE1 (6.87).
- **MoCAD** (Macau): NEST beats CS-LSTM, NLS-LSTM, MHA-LSTM, CF-LSTM, STDAN, WSiP, BAT(25 %), BAT on 1–5 s RMSE.
- **HighD** (German highway): similar dominance on longer horizons.

Ablations confirm: (1) hypergraph beats pairwise graph; (2) small-world rewiring adds value; (3) neuromodulator (adaptive $\alpha, \beta$) is necessary — fixed-threshold variants are worse.

## Phase 1 — Undergraduate-level synthesis

**The key idea.** Predicting where cars and pedestrians will go is hard because real traffic involves *groups* (a platoon of cars, a cluster of pedestrians) and *long-range chains* (a brake light five cars ahead). Most deep learning models for this task represent each pair of agents as an edge in a graph — but that misses the group structure. The authors use a **hypergraph** instead, where a single "edge" can connect many agents at once (the convoy *is* the hyperedge). Then they add **small-world** shortcuts to capture long-range influence — most connections are local, but a few jump across the scene. The shape of the hypergraph isn't fixed: two parameters $\alpha$ and $\beta$ control which agents end up grouped and how many long-range shortcuts there are. The "neuromodulation" trick is to *learn* those two parameters from the agent features themselves, using two tiny MLPs. So the graph reconfigures dynamically depending on traffic density and conditions — analogous, the authors argue, to how human drivers adapt to context.

**The experimental setup.** Standard autonomous-driving benchmarks: nuScenes (urban Boston/Singapore), MoCAD (Macau), HighD (German highway). Metrics: minimum ADE / FDE (average and final displacement error over the predicted trajectory) and RMSE over 1–5 second horizons. Compared against 15+ prior trajectory predictors.

**The result.** NEST wins across all three benchmarks, especially at longer prediction horizons (4–5 s) where capturing group dynamics matters most. The ablations confirm both the hypergraph structure and the neuromodulator are necessary.

**Worked example (MoCAD, RMSE at 5 s):** NEST 2.42 m vs. BAT (the prior SOTA) 2.88 m vs. CS-LSTM 4.49 m. The neuromodulator-controlled hypergraph is roughly half the prediction error of a recurrent baseline at the 5-second horizon.

## Phase 2 — Graduate-level deep dive

### Small-world Network construction

The Newman–Watts (NW) small-world model starts with a regular ring lattice and *adds* (rather than rewires) random long-range edges. NEST's analogue:

1. Start with a clustering-coefficient matrix $C \in \mathbb{R}^{(n+1) \times s}$ where $n+1$ is the number of agents (target + surroundings) and $s$ is the predefined number of hyperedges. $C_{i,j}$ measures the tendency of vertex $i$ to cluster with hyperedge $j$.

2. **Threshold step (regular network):** apply threshold $\alpha$ to extract definite connections.

$$C_{i,j}^{\text{thresh}} = \mathbb{1}[\,C_{i,j} \ge \alpha\,]$$

3. **Random shortcut step (small-world):** for cells where $C_{i,j}^{\text{thresh}} = 0$, add a connection with probability $\beta$ via uniform random draw $\eta$:

$$E_{i,j} = \begin{cases} 1 & \text{if } C_{i,j}^{\text{thresh}} = 1 \\ 1 & \text{if } C_{i,j}^{\text{thresh}} = 0 \text{ and } \eta \sim U[0,1] \le \beta \\ 0 & \text{otherwise} \end{cases}$$

The final hyperedge set $E$ has the small-world structure: high clustering (from the threshold step) + short average path length (from the shortcut step). The whole process is summarised compactly as

$$E = \Omega(V, \alpha, \beta) \tag{3}$$

where $\Omega$ is the NW generator.

### The Neuromodulator

The two parameters are themselves *learned functions* of the data:

$$\alpha = \Pi_\alpha(C) \tag{4}$$
$$\beta = \Pi_\beta(F_a) \tag{5}$$

$\Pi_\alpha$: takes $C \in \mathbb{R}^{(n+1) \times s}$ → projects to $\mathbb{R}^{(n+1) \times 1}$ via two MLP layers → sigmoid → $\alpha \in [0,1]$.
$\Pi_\beta$: takes the agent feature $F_a$ (encoded by a Transformer) → MLP layers → sigmoid → $\beta \in [0,1]$.

The biological inspiration argued by the authors: $\alpha$ "determines which agents are included in an interaction group" (sensitivity to grouping); $\beta$ "represents the likelihood of involving additional agents in interaction groups, particularly in high-density traffic" (long-range coupling). In neuromodulation terms, $\alpha$ is acetylcholine-like (attentional threshold) and $\beta$ is noradrenaline-like (broad coupling vs. focal), although the authors don't make these specific mappings.

### Hypergraph Pooling formalisation

Let $V = [V_0, V_1, \ldots, V_n]$ and $E_{i,j}$ from above. **Vertex-to-Hyperedge** aggregation for hyperedge $j$:

Personality information ($V$ filtered by hyperedge $j$, then MLP-encoded):

$$I_p = M_p\!\left(\sum_{V_i \in V_j} \lambda_i\, V_i\right) \tag{6}$$

with $V_j = \{V_i \mid E_{i,j} = 1\}$ and $\lambda_i$ learnable weights.

Intention information (Gumbel-softmax over $K$ modes):

$$I_i = \sigma\!\left(\frac{M_i(I_a) + \xi}{\tau}\right) \tag{7}$$

with $\xi$ Gumbel noise, $\tau$ temperature, $I_a = \sum_{V_i \in V_j} \lambda_i V_i$ aggregated info, and $\sigma$ softmax.

Willingness: $I_w = M_w(I_a)$.

Group feature: $F_{g,j} = \text{concat}(I_p, I_i, I_w)$.

**Hyperedge-to-Vertex** update: aggregates hyperedge group features back to vertices. After $H$ iterations the interaction feature is

$$F_i = \frac{1}{n+1} \sum_{k=0}^{n} V_k^{(H)}$$

### Context Fusion

Lane encoder maps HD map $M$ to lane features $F_l$. Cross-attention with interaction features as queries:

$$F_c = \text{Attn}(Q = F_i, K = F_l, V = F_l) \tag{10}$$

### Multi-modal Laplace predictor

For each intention mode $k \in [1, K]$, a generator predicts Laplace parameters per time-step:

$$Y_k^t = [x_k^t, y_k^t, b_{k,x}^t, b_{k,y}^t]$$

with associated probability $P_k$. Loss is mode-selection (winner-takes-all) + Laplace NLL.

### Why the "neuromodulation" link to Vecoven 2020 is loose

The paper cites Vecoven et al. 2020 as motivation, but the mechanism is fundamentally different. Vecoven's NMN modulates the *slope* and *bias* of every activation function via a *shared latent context vector* $z$ that propagates through the network. NEST's neuromodulator modulates *two scalars* ($\alpha, \beta$) that control the *graph structure*, not the activations. There is no continuous slope/bias modulation, no parametric activation function, no shared context propagated through layers.

The conceptual analogy is real but coarse: "a small auxiliary network adapts a downstream computation to context". In the four-scale framework of Mei et al. 2022, this fits Scale 1 (hyperparameter reconfiguration) — $\alpha$ and $\beta$ are effectively learned hyperparameters of the graph-construction step. Scales 2/3/4 (cell-type modulation, weight scaling, compartmental neurons) are not addressed.

### Where the benchmarks land

| Dataset | Metric | Prior SOTA | NEST | Gain |
|---|---|---|---|---|
| nuScenes | minADE5 | 1.38 (SeFlow) | 1.18 | −14 % |
| nuScenes | minFDE1 | 6.99 (GoHome) | 6.87 | −2 % |
| MoCAD | RMSE 5s | 2.88 (BAT) | 2.42 | −16 % |
| MoCAD | RMSE 3s | 1.39 (BAT) | 1.27 | −9 % |

The improvements are most pronounced at long horizons (5 s) where group dynamics and long-range chains dominate — consistent with the hypergraph + small-world hypothesis.

### Critical scrutiny

1. The "neuromodulation" label is marketing-heavy; the actual modulator is two tiny MLPs producing two scalars. Compared to Vecoven 2020's per-neuron $(w_s, w_b)$ vectors and shared $z$, or Ben-Iwhiwhu's per-layer modulator branch, this is minimal.
2. The Newman–Watts small-world generator uses random shortcuts ($\eta \sim U[0,1]$). This introduces stochasticity into the forward pass that the paper does not analyse rigorously (no Gumbel-Softmax relaxation here — the shortcut decision is hard).
3. Ablation tables (in the full paper appendix) compare against fixed-$\alpha$/fixed-$\beta$ variants but do not isolate the contribution of the neuromodulator from the small-world topology vs. the hypergraph structure.
4. The biological inspiration cited (cellular neuroregulation, Vecoven 2020) is not the closest match; if anything, the dynamic-graph-rewiring story is closer to *Aston-Jones & Cohen 2005 network reset theory* (where NA reconfigures cortical activity under context shifts) than to slope/bias activation modulation.

## Connections

- **[vecoven_2020_neuromod_dnn](vecoven_2020_neuromod_dnn.md)** — explicitly cited as the inspiration for the Neuromodulator component. The mechanism is much shallower in NEST (two scalars vs. shared context vector + per-neuron scale/bias), but the *idea* of using a small adaptive subnetwork to make a downstream computation context-dependent is genuine.
- **[ben-iwhiwhu_2022_context_meta_rl](beniwhiwhu_2022_context_meta_rl.md)** — sister application of "neuromodulation as adaptive subnetwork" but for meta-RL representation learning rather than graph structure.
- **[mei_2022_multiscale_neuromod](mei_2022_multiscale_neuromod.md)** — NEST sits at Scale 1 (hyperparameter reconfiguration) of Mei's four-scale framework. The small-world threshold $\alpha$ and shortcut probability $\beta$ are graph-construction hyperparameters.
- **[wang_2024_neuromod_meta](wang_2024_neuromod_meta.md)** — both papers use the term "neuromodulation" loosely. Wang 2024 means *per-task structural mask*; Wang 2025 means *graph-structure hyperparameters*. Neither matches the activation-modulation tradition of Vecoven 2020 / Ben-Iwhiwhu 2022.
- **[ferguson_cardin_2020_gain_modulation](ferguson_cardin_2020_gain_modulation.md)** — biological substrate for gain modulation. The link to NEST is weak: NEST modulates *graph structure*, not neuronal gain.
- **Newman & Watts 1999** — the small-world network model; this is the structural backbone NEST uses.
- **Watts & Strogatz 1998** — original small-world paper; cited but not directly used (NW model is the variant).
- **Gao et al. 2022** — hypergraph neural networks foundational work; NEST's HGNN derives from this lineage.
- **Xu et al. 2022 (group-wise multi-modal trajectory prediction)** — most direct prior in hypergraph-based trajectory prediction; NEST extends this with the small-world + neuromodulator additions.
- **Grossman & Cohen 2022** — cited as a neuroscience reference for dynamic adaptation to traffic conditions.
