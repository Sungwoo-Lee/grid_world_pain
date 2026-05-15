---
title: "Attention-Based Selective Plasticity"
authors: ["Soheil Kolouri", "Nicholas Ketz", "Xinyun Zou", "Jeffrey Krichmar", "Praveen Pilly"]
year: 2019
venue: "arXiv:1903.06070 (preprint)"
slug: kolouri_2019_attention_plasticity
source_pdf: "sources/Kolouri et al. 2019 - Attention-Based Structural-Plasticity.pdf"
topic: neuromodulatory_algorithms
---

# Kolouri et al. 2019 — Attention-Based Selective Plasticity

## Plain-English entry point

This short paper attacks **catastrophic forgetting** — the well-known failing of standard neural networks that when you train them sequentially on task A then task B, they forget A. The biological brain mostly does not have this problem, and one ingredient is **selective plasticity**: not every synapse should be equally easy to change. The brain solves this in part using a chemical called acetylcholine, released by the cholinergic system, which biases attention to task-relevant features (Avery, Dutt & Krichmar 2014). Kolouri et al. translate this idea into a deep-learning trick: alongside each synaptic weight $\theta_k$, store an **importance** scalar $\gamma_k$. After learning task A, future training penalizes changes to weights with high $\gamma_k$ (so they stay near their task-A values), letting low-$\gamma$ weights freely adapt to task B. The novelty over prior work in this family (Elastic Weight Consolidation — Kirkpatrick 2017; Synaptic Intelligence — Zenke 2017) is *how* $\gamma_k$ is computed: instead of Fisher information or path integral, they use a top-down attention map computed via **contrastive Excitation Backpropagation (c-EB)** combined with a **Hebbian Oja's rule** to grow $\gamma$ for synapses whose pre- and post-synaptic neurons are both lit up by the attention map. The result: a biologically-motivated, online, label-aware importance estimator that matches EWC and SI on Permuted MNIST and Split MNIST benchmarks. The paper matters because it is the cleanest articulation in this corpus of "neuromodulator-as-attention" → "attention controls plasticity" → "selective plasticity defeats forgetting" — a thread that runs through the broader neuromodulatory-RL and lifelong-learning literature.

## Section-ordered backbone

**Introduction.** Catastrophic forgetting (McCloskey & Cohen 1989) hits all uniformly plastic networks. An ideal network balances plasticity (acquire new) and stability (preserve old). The "selective plasticity" family of solutions: modulate which weights are allowed to change. Cholinergic neuromodulation biologically drives both stimulus-driven and goal-directed attention (Avery et al. 2014); contrastive Excitation Backpropagation (c-EB, Zhang et al. 2018) gives a top-down attention map; combine the two.

**Relevant work.** Three families: (1) selective synaptic plasticity (EWC, SI, MAS); (2) neurogenesis (allocate new neurons); (3) complementary learning systems (rehearsal, generative replay). This paper extends family (1). EWC: $\gamma_k = F_{kk}$ from Fisher information matrix, computed offline at task end. SI (Zenke): $\gamma_k$ from cumulative weight change during training. MAS (Aljundi): $\gamma_k$ from gradient of last-layer output norm, decoupling from labels. All share the loss form:

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \lambda \sum_k \gamma_k\, (\theta_k - \theta_{A,k}^*)^2,
$$

where $\theta_{A,k}^*$ is the optimum from task A; $\mathcal{L}_B$ is the new task loss.

**Method (Section 3).**
- *Excitation Backpropagation* (Section 3.1, eqs. 2–3): Recursively defines a top-down "Marginal Winning Probability" $P(f_i^l)$ on neuron $i$ in layer $l$, propagated backward from $P(f_i^L = y) = 1$ at the output. Uses positive weights only: $P(f_j^{l-1} \mid f_i^l) = Z_i^{(l-1)} f_j^{(l-1)} \theta_{ji}^l$ if $\theta_{ji}^l \geq 0$ else $0$. Contrastive EB introduces a "negative" target node with negated weights and computes the ReLU-normalized *difference* between target and contrastive attention maps.
- *Attention-based importance* (Section 3.2, eq. 4): Synapse $\gamma_{ji}^l$ between neurons $f_j^{l-1}$ and $f_i^l$ grows when *both* are highlighted by c-EB. Oja's rule (Oja 1982) is used to keep $\gamma$ bounded:

$$
\gamma_{ji}^l \leftarrow \gamma_{ji}^l + \varepsilon\!\left( P_c(f_j^{(l-1)})\, P_c(f_i^{(l)}) - \left[P_c(f_i^{(l)})\right]^2 \gamma_{ji}^l \right).
$$

- *Updated loss* (Section 3.3): Same quadratic regularizer as EWC/SI, but with the new $\gamma$ values; importance is updated *online* during training, so no offline post-task computation is needed.

**Experiments (Section 4).** Two benchmarks:
- *Permuted MNIST*: 5 sequential tasks, each a fixed random permutation of MNIST pixels. MLP 784-400-400-10, ReLU, ADAM ($lr=10^{-3}$). The method preserves earlier-task accuracy as new tasks are learned, on par with EWC and SI. c-EB attention maps visually pick out the correct digit-shape pixels.
- *Split MNIST*: 5 pairs of digits learned sequentially (more realistic; transfer possible). Method matches Synaptic Intelligence performance.

**Conclusion.** Biologically-motivated online selective-plasticity rule, comparable to SOA. Discusses future extensions to deep RL and other top-down attention methods (Grad-CAM).

## Phase 1 — Undergraduate-level synthesis

**The key idea.** Neural networks forget previously-learned tasks because every weight is equally free to be updated by gradient descent on the new task. The brain doesn't forget so easily — partly because acetylcholine helps direct attention to what matters, and attention can lock down the synapses that encode important things. Kolouri et al. add a "memory tag" $\gamma_k$ to each weight; weights with high tags become hard to change. They compute the tag using a backward-attention algorithm (c-EB) that highlights which neurons mattered for the correct answer, and Hebbian learning grows the tag whenever both ends of a synapse were highlighted.

**The experimental setup.** Take a standard multilayer perceptron. For each training example, do two passes: forward to compute the prediction and loss, and a special "contrastive excitation backprop" pass to compute, layer-by-layer, which neurons were responsible for the correct class versus the runner-up class. For each synapse, if both its pre- and post-synaptic neurons are highlighted by this attention computation, increase the synapse's importance tag $\gamma$ via Hebbian Oja's rule. When training on a new task, add a regularizer $\lambda \sum_k \gamma_k (\theta_k - \theta^*_k)^2$ to keep important weights anchored. Test on Permuted MNIST and Split MNIST.

**The result.** The method preserves accuracy on old tasks roughly as well as Elastic Weight Consolidation and Synaptic Intelligence (the two leading 2017-era selective-plasticity methods), but uses a biologically-motivated attention mechanism (c-EB) instead of Fisher information or path integrals. Importantly, $\gamma$ is updated *online* during training — no offline phase needed.

## Phase 2 — Graduate-level deep dive

### 2.1 The general selective-plasticity objective (eq. 1)

Given a network $f(\cdot; \theta)$ with weights $\theta$, optimized to $\theta_A^*$ on task A, when learning task B minimize:

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \lambda \sum_k \gamma_k\, (\theta_k - \theta_{A,k}^*)^2,
$$

where $\gamma_k \geq 0$ is the **importance** of synapse $k$, $\lambda \geq 0$ controls the stability/plasticity trade-off, and $\mathcal{L}_B$ is e.g. cross-entropy for task B. Different methods differ only in how $\gamma_k$ is defined.

### 2.2 EWC vs SI vs MAS vs Kolouri (importance definitions)

- **EWC** (Kirkpatrick 2017): $\gamma_k = F_{kk}$, the $k$-th diagonal of the Fisher information matrix
$$F = \mathbb{E}_{p_A(x, y)}\!\left[\nabla_\theta \log p_\theta(y \mid x)\, \nabla_\theta \log p_\theta(y \mid x)^\top\right],$$
computed offline at the end of task A.
- **SI** (Zenke 2017): $\gamma_k$ = cumulative $|\Delta \theta_k|$ along the training trajectory, weighted by loss decrease.
- **MAS** (Aljundi 2018): $\gamma_k = \mathbb{E}_{p_A(x)}\!\left[ |\partial \|f(x; \theta)\|_2^2 / \partial \theta_k| \right]$, label-free.
- **Kolouri (this paper)**: $\gamma_k$ from Hebbian product of attention probabilities on pre- and post-synaptic neurons, updated online via Oja's rule.

### 2.3 Excitation Backpropagation (Section 3.1)

For a network $f_i^l = \sigma\!\big(\sum_j \theta_{ji}^l f_j^{l-1}\big)$, define the **Marginal Winning Probability (MWP)** distribution $P(f_j^{(l-1)})$ over layer-$(l-1)$ neurons given the target output, by recursive marginalization:

$$
P(f_j^{(l-1)}) = \sum_i P(f_j^{(l-1)} \mid f_i^l)\, P(f_i^l), \qquad P(f_j^{(l-1)} \mid f_i^l) = \begin{cases} Z_i^{(l-1)}\, f_j^{(l-1)}\, \theta_{ji}^l, & \theta_{ji}^l \geq 0, \\ 0, & \text{otherwise}, \end{cases}
$$

with normalization

$$
Z_i^{(l-1)} = \left( \sum_{j: \theta_{ji}^l \geq 0} f_j^{(l-1)}\, \theta_{ji}^l \right)^{-1}.
$$

Initialize at the output: $P(f_i^L = y) = 1$ for the correct class $y$ and $0$ for others. Recurse backward.

### 2.4 Contrastive EB (c-EB)

Introduce a hypothetical "negative" output node $\bar f_i^L$ with negated weights $\bar\theta_{ji}^L = -\theta_{ji}^L$ and compute $\bar P(f_j^{l-1} \mid f_i^l)$ analogously. The **contrastive attention** at layer $l-1$ is

$$
P_c(f_j^{(l-1)} \mid f_i^l) = \frac{\text{ReLU}\!\left( P(f_j^{(l-1)} \mid f_i^l) - \bar P(f_j^{(l-1)} \mid f_i^l) \right)}{\sum_j \text{ReLU}\!\left( P(f_j^{(l-1)} \mid f_i^l) - \bar P(f_j^{(l-1)} \mid f_i^l) \right)}.
$$

This subtracts off the attention the negative target would have produced — analogous to a contrastive top-down "differential" attention used in computer-vision visualization.

### 2.5 Hebbian Oja's-rule importance update (eq. 4)

For a synapse $\theta_{ji}^l$ between $f_j^{(l-1)}$ and $f_i^l$, the importance grows according to:

$$
\gamma_{ji}^l \;\leftarrow\; \gamma_{ji}^l + \varepsilon\!\left( P_c\!\big(f_j^{(l-1)}\big)\, P_c\!\big(f_i^l\big) - \left[P_c\!\big(f_i^l\big)\right]^2 \gamma_{ji}^l \right),
$$

where $\varepsilon > 0$ is the Hebbian learning rate. The first term is the **Hebbian product** of pre- and post-synaptic attention probabilities — bigger when both neurons are highly attended for the correct class. The second term is **Oja's stabilization** — it prevents unbounded growth by subtracting a quadratic decay proportional to $\gamma$. At steady state $\gamma_{ji}^l \to P_c(f_j^{l-1}) / P_c(f_i^l)$.

The rule has classical neuromodulatory interpretation:
- Pre-synaptic attention $P_c(f_j^{l-1})$ → cholinergic salience signal on the input neuron.
- Post-synaptic attention $P_c(f_i^l)$ → cholinergic salience signal on the output neuron.
- Hebbian product → "neurons that fire together (under attention) wire together (in importance)".
- Oja stabilization → biological synaptic homeostasis.

### 2.6 Full algorithm

```
For each training example (x, y) on task t:
    1. Forward pass: compute f^l(x; theta) for all layers l, get prediction.
    2. c-EB backward pass: compute P_c(f_j^l) for all neurons in all layers.
    3. Update gamma_{ji}^l via Hebbian Oja's rule (eq. above).
    4. Compute task loss L_B and regularizer lambda * sum_k gamma_k (theta_k - theta*_k)^2.
    5. Backprop gradient through L = L_B + regularizer, update theta with ADAM.
```

Online — no offline phase between tasks. At task transition, snapshot $\theta_A^* \leftarrow \theta$.

### 2.7 Why attention-based importance is principled

The attention map $P_c$ identifies which neurons are causally important for producing the correct (vs. confusable) output. Synapses connecting two highly-attended neurons are the ones whose weights actually carry task-relevant information; freezing them via large $\gamma$ preserves task knowledge. Synapses between low-attention neurons are likely encoding spurious / task-irrelevant features; leaving them low-$\gamma$ enables free adaptation to new tasks. This is the formal articulation of **"acetylcholine boosts the learning rate at attention-relevant sites and suppresses it elsewhere"** (Avery, Dutt & Krichmar 2014).

### 2.8 Empirical results summary

- **Permuted MNIST** (Fig. 5): After 5 tasks, average accuracy on first task with method ~85%; vanilla MLP ~30%. Comparable to EWC (~85%) and SI ($c$-dependent).
- **Split MNIST** (Fig. 7): Matches SI; both ~95% retained.
- c-EB attention maps visualize meaningfully: for digit "7", attention picks out the digit-shape pixels (Fig. 3).

### 2.9 What this paper is NOT

It is *not* a "neuromodulator-in-the-network" model (the modulator does not have its own state or dynamics). It is a *learning-rule* paper where the cholinergic system is taken as inspiration for a particular form of importance estimator. Contrast with Tsuda 2021 (this batch), where the modulator is a literal scalar applied to weights at inference time, and Costacurta 2024, where the modulator is a separate subnetwork.

## Connections to other corpus papers

- **`avery_krichmar_2017_neuromodulation_models.md`** (other batch) and the broader Krichmar cluster (Cox & Krichmar 2009, Avery, Dutt & Krichmar 2014 [ref. cited as Avery et al.]) — the foundational biological-modeling work that this paper draws on for "cholinergic system = attention control over plasticity". Kolouri's co-authors Zou and Krichmar are direct Krichmar-lab collaborators.
- **`kirkpatrick_2017_ewc.md`** (other batch, ref. Kirkpatrick et al. 2017 — EWC) — the canonical competitor; same loss form, different importance.
- **`zenke_2017_synaptic_intelligence.md`** (other batch, Zenke, Poole & Ganguli 2017) — closest competitor; online cumulative-change importance.
- **`aljundi_2018_memory_aware_synapses.md`** (other batch) — label-free importance via output-norm gradient.
- **`zhang_2018_top_down_attention_excitation.md`** (other batch, Zhang et al. 2018) — provides c-EB, the attention engine Kolouri uses.
- **`oros_chiba_nitz_krichmar_2014_learning_to_ignore.md`** (cited; Oros et al. 2014) — biological backing for "ACh increases attention to relevant, decreases to distractors".
- **`hebb_1961.md`** / **`oja_1982.md`** — Hebbian and stabilized-Hebbian learning rules underlying eq. 4.
- **`kudithipudi_2022_lifelong_learning.md`** (other batch) — broad lifelong-learning synthesis where this style of selective-plasticity sits as one of the canonical mechanisms.
- **`lee_2024_lifelong_rl_neuromodulation.md`** (other batch) — extends attention-based selective plasticity to reinforcement learning, which Kolouri explicitly flags as future work.
- **`mei_2022_multiscale_principles.md`** (other batch) — argues DNN designers should incorporate neuromodulatory multiscale principles; Kolouri 2019 is an early concrete instance for the plasticity axis.
- **`alonso_krichmar_2023_hopfield_continual.md`** (other batch, this same lab cluster) — Hopfield-based continual memory, complementary approach.
- **`tsuda_2021_activity_hypertubes.md`** / **`costacurta_2024_structured_flexibility.md`** (this batch) — orthogonal mechanism: in those papers the modulator scales *weights at inference*; Kolouri uses cholinergic-style attention to scale *the learning rate of weights*. Both fit under the "neuromodulator as low-dimensional control variable" umbrella but target different dimensions (plasticity vs. dynamics).
- **`vecoven_2020_neuromodulation_dnn.md`** (other batch) — modulator as inference-time gain; useful contrast.
- **`miconi_2019_backpropamine.md`** (other batch) — differentiable neuromodulated plasticity; closest in spirit, integrates modulation into the learning rule.
- **`shine_2021_cellular_to_dynamics.md`** (this batch) — Shine catalogues "cholinergic system enhances precision of bottom-up sensory transmission" (Box 3); Kolouri's c-EB attention map is exactly such a precision signal in DNN form.
