---
title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
authors: ["Noam Shazeer", "Azalia Mirhoseini", "Krzysztof Maziarz", "Andy Davis", "Quoc Le", "Geoffrey Hinton", "Jeff Dean"]
year: 2017
venue: "ICLR 2017"
slug: shazeer_2017_sparse_moe
source_pdf: "sources/Shazeer et al. 2017 - Outrageously large neural networks - The Sparsely-Gated Mixture-of-experts layer.pdf"
topic: FiLM
---

# Shazeer et al. 2017 — Sparsely-Gated Mixture-of-Experts Layer

## Plain-English entry point

A standard deep neural network uses *all* of its parameters on *every* example. That makes very large networks expensive: doubling the number of parameters roughly doubles the compute. The Sparsely-Gated **Mixture-of-Experts (MoE) layer** breaks this coupling.

The idea: replace a dense feedforward layer with a bank of $n$ separate small subnetworks called **experts** $E_1, \dots, E_n$, plus a tiny **gating network** $G$ that, given the input $x$, outputs a sparse weighting over which experts to use. For each example, only a small handful of experts (say, $k = 2$ or $4$) are activated; the rest are skipped entirely. The MoE output is $y = \sum_i G(x)_i E_i(x)$, with most $G(x)_i$ exactly zero. So the *total parameter count* of the layer can be enormous (the paper goes up to 137 billion) while the *per-example compute* stays small.

To make this work in practice the authors solve four problems: (1) **how to make gating sparse**: a "noisy top-$k$" softmax adds Gaussian noise then keeps only the $k$ largest logits, zeroing the rest; (2) **how to keep experts balanced** so the gate doesn't collapse onto a few favourites: two auxiliary losses — an "importance" loss and a "load" loss — penalize uneven expert utilization across a batch; (3) **how to make this efficient on GPUs**, which prefer big batches: combine data parallelism (gating, standard layers) with model parallelism (experts shared across devices), so each expert sees an aggregated batch; (4) **bandwidth**: choose experts large enough that compute-per-example dominates the cost of shipping activations.

On the 1-billion-word language modelling benchmark, an MoE-augmented LSTM with up to 4 billion parameters beats the previous best LSTM (151M params) at 6% of the compute. On WMT'14 En→Fr, an MoE-GNMT achieves 40.56 BLEU vs the previous best 39.92.

## Section-ordered backbone

**1. Introduction and related work.**

**1.1 Conditional computation.** Increasing capacity helps but causes quadratic cost blow-up. Conditional computation (turn off parts of the network per-example) has been proposed for a decade (Davis & Arel 2013, Bengio et al. 2013/2015, Cho & Bengio 2014, Eigen et al. 2013) but has not yet delivered massive capacity gains in practice. Five challenges: (i) GPUs prefer arithmetic to branching; (ii) per-expert batch size shrinks under conditional routing; (iii) network bandwidth across distributed devices is a bottleneck; (iv) sparsity-inducing loss terms can hurt quality and balance; (v) capacity gains require very large datasets to be useful.

**1.2 The Sparsely-Gated MoE layer.** General-purpose neural-net component: a bank of feedforward experts + a trainable gating network selecting a sparse combination. Applied convolutionally between stacked LSTM layers in this paper, the MoE chooses (potentially different) experts at every text position.

**1.3 Related work on MoE.** Classical MoE (Jacobs et al. 1991, Jordan & Jacobs 1994); various expert types (SVMs, GPs, Dirichlet processes, deep nets); hierarchical, infinite-experts, sequentially-added. Eigen et al. 2013 introduced stacked MoEs as deep-network components. This paper extends Eigen's idea to (a) sparse gating, (b) convolutional application across timesteps, (c) hierarchical multi-level MoE for very many experts.

**2. Structure of the MoE layer.** $n$ experts $E_i$ with identical feedforward architecture but separate parameters; one gating network $G$ outputting a sparse $n$-vector. Output $y = \sum_i G(x)_i E_i(x)$. Hierarchical option: primary gating selects among groups of secondary MoEs, each with its own gating.

**2.1 Gating network.** Softmax over $x \cdot W_g$ for non-sparse gating. **Noisy top-$k$ gating** adds tunable Gaussian noise scaled by a softplus-modulated learned $W_{noise}$, then keeps top-$k$, zeroing the rest. Trained by standard backprop (no REINFORCE). The noise serves both balancing (smooths the discrete-top-$k$ decision into a probability) and exploration purposes.

**3. Addressing performance challenges.**

**3.1 Shrinking batch.** Per-expert batch is $\sim kb/n \ll b$ for $n \gg k$. Solution: mix data parallelism (replicate gating/standard layers across $d$ devices) with model parallelism (one shared copy of each expert). Each expert then receives a batch of size $\sim kbd/n$, scaling proportionally with the cluster. For hierarchical MoE: data-parallel primary gate, model-parallel secondary MoEs (one per device). Also exploit convolutional/timestep application — combine all unrolled timesteps into one big MoE batch.

**3.2 Network bandwidth.** Experts are stationary on their devices, so the only data shipped across the network is expert inputs/outputs. For GPUs with thousand-to-one compute-to-bandwidth ratios, set expert hidden layer size $\gtrsim 1000$ to be bandwidth-bound-safe.

**4. Balancing expert utilization.** Without intervention, gating collapses onto a few favourite experts (self-reinforcing). Two soft constraints (Appendix A):
- **Importance loss**: $L_{imp} = w_{imp} \cdot \mathrm{CV}(\mathrm{Importance}(X))^2$ with $\mathrm{Importance}(X)_i = \sum_x G(x)_i$. Penalizes high coefficient-of-variation across expert importances.
- **Load loss**: smooth differentiable estimator $\mathrm{Load}(X)_i = \sum_x P(x, i)$ where $P(x, i)$ is the probability under fresh noise that $G(x)_i$ would be nonzero. $L_{load} = w_{load} \cdot \mathrm{CV}(\mathrm{Load}(X))^2$. Importance loss alone can leave actual example counts uneven; load loss ensures balanced *counts*.

**5. Experiments.**

**5.1 1-Billion-Word LM.** MoE between two stacked LSTM layers. Series with matched compute (~8M ops/timestep): flat MoE with 4, 32, 256 experts; hierarchical with 256, 1024, 4096 experts. Each expert ~1M params, $k=4$ active. 4096-expert model: 24% lower perplexity than computational-budget-matched baseline. High-budget MoE (4.37B params, 142.7M ops/timestep): test perplexity 28.0 after 100 epochs vs best published 30.6 (Jozefowicz et al. 2016 with 151M params). Even low-budget MoE beats best published with 6% of the compute.

**5.2 100-Billion-Word Google News corpus.** MoE up to 131072 experts (137B parameters). At 65536 experts (68B params, 99.994% layer sparsity), test perplexity drops 39% vs matched-compute baseline. Degrades slightly at 131072, possibly too much sparsity.

**5.3 Machine translation (single language pair).** Modified GNMT (3-layer encoder, 2-layer decoder) with MoE inserted between LSTM layers. 2048 experts × 2M params = 8B params. WMT'14 En→Fr: 40.56 BLEU vs GNMT+RL 39.92. WMT'14 En→De: 26.03 BLEU vs GNMT 24.91. Production En→Fr: 36.57 BLEU vs GNMT 35.56.

**5.4 Multilingual MT.** Single MoE model jointly handles 12 language pairs, outperforming 12 separately-trained GNMTs on most pairs.

**6. Conclusion.** Sparsely-gated MoE delivers >1000× model-capacity gains at fractional compute cost, opening the door to trillion-parameter models. The auxiliary balancing losses, plus the data+model parallelism scheme, are the key engineering enablers.

## Phase 1 — undergraduate-level synthesis

**Key idea.** Instead of one giant feedforward layer, use *many small specialist layers* (experts) plus a *router* (gating network) that picks just a few experts to run on each input. The rest of the experts cost zero compute. Total parameters can be billions; per-example compute stays small.

**Why this matters.** Capacity (parameter count) helps deep networks absorb information from huge datasets, but uniformly-active networks become prohibitively expensive. Sparse activation breaks the link between *capacity* and *cost-per-example*. You pay only for what you use.

**Setup.**
1. Bank of $n$ feedforward experts $E_1, \dots, E_n$, each a small two-layer MLP with its own weights.
2. A tiny gating network $G$ takes the input $x$ and produces an $n$-dim score vector.
3. **Noisy top-$k$ gating**: add Gaussian noise to the scores, keep the $k$ largest, set the rest to $-\infty$, softmax. Now only $k$ experts have non-zero weight.
4. Output: $y = \sum_i G(x)_i \, E_i(x)$. Skip $E_i$ if $G(x)_i = 0$.

**Two engineering nuisances.**
- **Load balancing.** If left alone, the gate picks the same few experts every time (the favoured ones get trained more, get picked more, etc.). Fix: add two penalty losses. One penalizes the variance of *gate sums per expert across the batch* (importance); the other penalizes the variance of *example-counts per expert* (load).
- **GPU efficiency.** Per-expert batch shrinks as $n$ grows. Fix: replicate gating across devices via data parallelism but place experts on different devices via model parallelism, so each expert collects examples from *all* data-parallel replicas.

**Headline result.** On 1-Billion-Word language modelling, an MoE-LSTM with 4 billion parameters beats the previous best 151M-parameter LSTM, using 6% of its compute. On WMT'14 En→Fr translation, MoE-GNMT (8 billion params) reaches 40.56 BLEU, beating GNMT+RL's 39.92. On a 100-billion-word internal corpus, a 68-billion-parameter MoE (65536 experts) cuts perplexity 39% vs compute-matched baseline.

**Initial takeaway.** MoE is one half of the "many small specialists, one router" family — the *discrete*, *bank-selection* variant. The other half is the hypernetwork family (Ha 2016, FiLM, etc.), which generates a *single context-dependent weight set* by continuous synthesis. MoE makes the **effective parameter count** much larger than the **compute per example**; hypernets make the **trained parameter count** much smaller than the **effective behavioural repertoire**. Both reshape the capacity-vs-compute trade-off.

## Phase 2 — graduate-level deep dive

### MoE layer definition

Given $n$ expert subnetworks $E_1, \dots, E_n: \mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out}}$ (here, two-layer ReLU MLPs with identical architectures but separate parameters), and a gating network $G: \mathbb{R}^{d_{in}} \to \Delta^{n-1}$ outputting a *sparse* probability simplex vector, the MoE layer computes:

$$
y = \sum_{i=1}^{n} G(x)_i \, E_i(x).
$$

Computational saving: whenever $G(x)_i = 0$, $E_i(x)$ is not computed.

### Gating: softmax baseline and noisy top-$k$

**Softmax gating** (Jordan & Jacobs 1994):

$$
G_\sigma(x) = \mathrm{Softmax}(x \cdot W_g), \qquad W_g \in \mathbb{R}^{d_{in} \times n}.
$$

This is dense — every expert is activated. To enforce sparsity and exploration, the authors define **Noisy Top-$k$ Gating**:

$$
\boxed{\,G(x) = \mathrm{Softmax}\!\Big(\mathrm{KeepTopK}\big(H(x), k\big)\Big),\,}
$$

where

$$
H(x)_i = (x \cdot W_g)_i + \mathcal{N}(0, 1) \cdot \mathrm{Softplus}\!\big((x \cdot W_{noise})_i\big),
$$

and

$$
\mathrm{KeepTopK}(v, k)_i = \begin{cases} v_i & \text{if } v_i \text{ is in the top-}k \text{ of } v, \\ -\infty & \text{otherwise.} \end{cases}
$$

After the $-\infty$ mask, softmax produces exactly $k$ non-zero entries summing to 1. Two trainable matrices: $W_g \in \mathbb{R}^{d_{in} \times n}$ for the mean logit; $W_{noise} \in \mathbb{R}^{d_{in} \times n}$ for the per-expert noise scale.

**Why this is differentiable.** Even with the hard top-$k$, the $k$ surviving entries pass gradients through both the softmax and the (continuous) sum $x \cdot W_g + \text{noise} \cdot \text{Softplus}(\cdot)$. The discontinuities at the boundary between "in top-$k$" and "not" are not differentiable, but empirically not problematic. The Gaussian noise smooths the boundary into a probabilistic selection, enabling the load-balancing loss below.

### Load-balancing losses

**Importance loss.** The "importance" of expert $i$ across a batch $X$ is

$$
\mathrm{Importance}(X)_i = \sum_{x \in X} G(x)_i.
$$

The importance penalty is

$$
\boxed{\,L_{importance}(X) = w_{importance} \cdot \mathrm{CV}\!\big(\mathrm{Importance}(X)\big)^2,\,}
$$

where $\mathrm{CV}(v) = \mathrm{std}(v) / \mathrm{mean}(v)$ is the coefficient of variation. Driving $\mathrm{CV} \to 0$ equalizes the expected sum of gate values per expert.

**Load loss.** Importance equalization is not enough: one expert might receive few examples with high gate values, another many examples with low gate values. Define the **smooth load**:

$$
\mathrm{Load}(X)_i = \sum_{x \in X} P(x, i),
$$

where $P(x, i) = \Pr\big[ G(x)_i > 0 \mid \text{resample noise on coordinate } i \big]$. Because of the Gaussian-noise structure of $H(x)_i$,

$$
P(x, i) = \Phi\!\Bigg( \frac{(x \cdot W_g)_i - \mathrm{kth\_excluding}(H(x), k, i)}{\mathrm{Softplus}\!\big((x \cdot W_{noise})_i\big)} \Bigg),
$$

where $\Phi$ is the standard-normal CDF and $\mathrm{kth\_excluding}(v, k, i)$ is the $k$-th largest entry of $v$ omitting index $i$. This is a smooth differentiable estimator of the number of examples that would assign expert $i$ as their top-$k$ winner. The load loss is then

$$
\boxed{\,L_{load}(X) = w_{load} \cdot \mathrm{CV}\!\big(\mathrm{Load}(X)\big)^2.\,}
$$

Total auxiliary loss: $L_{aux} = L_{importance} + L_{load}$ added to the standard task loss. Ablation (Table 6) shows either loss alone improves quality dramatically over no balancing (39.8 → 35.6 ppl on MoE-256), but the load loss specifically controls the *worst-overloaded* expert — `max(Load)/mean(Load)` drops from 17.80 (no loss) to 1.07–1.47 with balancing.

**Initial-balance trick.** Set $W_g = W_{noise} = 0$ at init so all experts start with equal expected load.

### Hierarchical MoE

For very large $n$, replace flat MoE with a two-level structure: primary gating $G_{primary}$ selects from $a$ groups, each group's secondary gating $G_j$ selects from $b$ experts in that group:

$$
y_H = \sum_{j=1}^a G_{primary}(x)_j \sum_{i=1}^b G_j(x)_i E_{j,i}(x).
$$

Effective expert count $a \cdot b$; gate branching factor $\max(a, b)$ instead of $a \cdot b$. Primary gating uses data parallelism; secondary MoEs use model parallelism (each on one device).

### Distributed-training analysis

Per-expert batch with naive distribution: $\sim kb/n$. With $d$ devices in a mixed-parallel scheme: $\sim kbd/n$. To keep this fixed (say $\sim$ baseline batch $b_0$), need $d / n$ constant — i.e., add devices proportionally as you add experts. Total cluster batch grows linearly with $d$; per-device memory/bandwidth stays constant. Conclusion: trillion-parameter MoE training requires only proportionally more devices, not exponentially more.

### Empirical anchors

| Task | Setup | Result |
|---|---|---|
| 1B-Word LM (low-budget MoE) | 4.3B params, 8.9M ops/timestep | 34.1 ppl (10 ep) vs 34.7 best published (151M, 32×K40s) |
| 1B-Word LM (high-budget MoE) | 4.37B params, 142.7M ops/timestep | 28.0 ppl (100 ep) vs 30.6 best published |
| 100B-Word LM | 68B params (65536 experts) | 39% perplexity drop vs matched-compute baseline |
| WMT'14 En→Fr | 8.7B params MoE-GNMT (longer training) | 40.56 BLEU vs GNMT+RL 39.92 |
| WMT'14 En→De | 8.7B params MoE-GNMT | 26.03 BLEU vs GNMT 24.91 |

## Connections

- **`ha_2016_hypernetworks.md`** (this batch): MoE and hypernetworks share the **"many small specialists, one router"** architectural motif. Hypernet routes by *generating* expert weights continuously from a low-dim embedding; MoE routes by *selecting* a sparse subset of pre-trained experts. Both decouple effective capacity from per-example compute. The relationship is mentioned in §2 of the MoE paper ("A MoE whose experts are simple weight matrices is similar to the parameterized weight matrix proposed in Cho & Bengio 2014").
- **`galanti_wolf_2020_hypernet_modularity.md`** (this batch): the modularity argument applies in spirit to MoE — the *expert* is small, capacity goes into the bank and the gate. Modularity is achieved by *discrete decomposition* in MoE vs. *continuous decomposition* in hypernets.
- **`krueger_2017_bayesian_hypernets.md`** (this batch): MoE is not Bayesian, but a "Bayesian MoE" could place a posterior over the gate distribution. The load-balancing entropy loss in this paper is conceptually adjacent to the entropy term in the BHN ELBO — both prevent collapse to a single mode.
- **`perez_2018_film.md`** (other batch — FiLM): FiLM is a *dense, continuous* conditional layer (every channel gets a non-zero $\gamma, \beta$). MoE is a *sparse, discrete* one (most experts contribute nothing). They sit on opposite ends of a sparsity spectrum within the same "context-conditioned routing" family.
- **`andreas_2016_neural_module_networks.md` / `hu_2017_e2e_module_networks.md`** (this batch): Module Networks also choose a sparse subset of computation units (modules) per example, but the routing is driven by an external program/layout *parser* rather than a learned gating network operating directly on the activations. Module networks are MoE's symbolic / compositional cousin.
- **Connection to the project's NMN-as-modulator direction**: MoE's "top-$k$ + load-balancing" recipe is the cleanest existing template for making *biological-system-shaped specialists* (e.g., 4-channel neuromodulators) coexist in a single architecture without one dominating. The CV-of-importance loss may be directly portable to the project's modulator-utilization-balance question.
