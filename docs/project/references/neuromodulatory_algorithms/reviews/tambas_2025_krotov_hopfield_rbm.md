---
title: "Neuromodulation via Krotov-Hopfield Improves Accuracy and Robustness of RBMs"
authors:
  - Baser Tambas
  - A. Levent Subasi
  - Alkan Kabakcioglu
year: 2025
venue: "arXiv:2505.06902 (cond-mat.dis-nn), 11 May 2025"
slug: tambas_2025_krotov_hopfield_rbm
source_pdf: "sources/Tambaş et al. 2025 - Neuromodulation via Krotov-Hopfield Improves Accuracy and Robustness of RBMs.pdf"
topic: neuromodulatory_algorithms
---

# Tambaş, Subaşı & Kabakçıoğlu (2025) — Neuromodulation via Krotov-Hopfield improves accuracy and robustness of RBMs

## Plain-English entry point

A **Restricted Boltzmann Machine (RBM)** is a classic generative neural network: two layers (visible pixels + hidden features) connected by weights, no within-layer connections, trained by an energy-based learning rule called Contrastive Divergence. RBMs are popular because they have an elegant statistical-physics description, but they have a known weakness — without lateral within-layer connections, hidden units **co-adapt** (multiple hidden units end up learning the same redundant feature), wasting model capacity.

This paper from physicists at Istanbul Technical / Koç University asks: **can we cure RBM hidden-unit redundancy by injecting a biology-inspired *neuromodulatory* signal during training?** Their inspiration is **biological neuromodulation** — diffusely transmitted chemical signals (think dopamine, noradrenaline, acetylcholine) that *globally tune synaptic plasticity* based on the organism's internal state, instead of working through point-to-point neural wiring. They use the Krotov-Hopfield (KH) algorithm — a recent biologically plausible unsupervised learning rule built on Bienenstock-Cooper-Munro theory — as the neuromodulatory signal. In each training step, **the hidden unit that fires most strongly gets full Hebbian reinforcement, the next few strongest get anti-Hebbian suppression, and the rest get nothing**. This is a winner-take-all-with-runner-up-penalty competition that does not require explicit lateral wires — just a global "modulator" function $g_\nu(I)$ that knows the rank-ordering of activations.

Result on MNIST: a KH-modulated RBM (KH-RBM) consistently beats a vanilla RBM on both reconstruction MSE and classification accuracy, is **robust to weight initialisation** (vanilla RBMs are notoriously sensitive), and **eliminates overfitting** during long training runs. With 500 hidden units, KH-RBM hits 97% on MNIST — matching a vanilla shallow cRBM that needs 10× more hidden units. The paper matters to this corpus because it is the cleanest demonstration that a single, simple, biologically motivated **neuromodulatory rule** can compensate for the missing lateral connectivity that plagues both biological microcircuits and machine-learning models.

## Section-ordered backbone

**Introduction.** Backpropagation, the workhorse of deep learning, is biologically implausible because it broadcasts a top-down error signal incompatible with local synaptic plasticity. Recent alternatives — feedback alignment, target propagation, equilibrium propagation, Hebbian methods including BCM theory — try to fix this. Boltzmann machines are energy-based generative networks built from binary stochastic units. Their fully connected version is intractable; **Restricted Boltzmann Machines (RBMs)** remove lateral connections to recover tractability but lose representational capacity. Workarounds (Deep Boltzmann Machines, convolutional RBMs, semi-RBMs, mean-field reintroductions) add complexity. The paper proposes a different route: borrow biology's **neuromodulation** — diffusively broadcast chemical signals that modulate synaptic plasticity globally — and implement it as the **Krotov-Hopfield (KH) algorithm** acting as a *three-factor learning rule* alongside the RBM's gradient descent.

**RBM essentials.** Standard derivation. Binary visible $v$ and hidden $h$ units, parameters $\theta = \{W, a, b\}$, energy

$$E_{\theta}(v, h) = -v^{\top} W h - a^{\top} v - b^{\top} h,$$

Boltzmann distribution $p_{\theta}(v, h) = e^{-E_{\theta}(v,h)} / Z_{\theta}$ with intractable partition function. Negative log-likelihood (free-energy form) is minimised by gradient descent, giving the canonical RBM weight update $\delta W_{ij} = \eta [\langle v_i h_j \rangle_d - \langle v_i h_j \rangle_m]$, the difference of data and model expectations. Model expectations approximated by **Contrastive Divergence** (CD$k$); $k=1$ and $k=10$ both reported.

**Krotov-Hopfield updates.** The local update for synapse $W_{\mu\nu}$:

- Input current to post-synaptic node $\nu$: $I_{r_\nu} = \langle W_\nu, x \rangle$, with rank $r_\nu$ in *descending* order.
- KH weight increment:

$$\delta W^{\text{KH}}_{\mu\nu} = \varepsilon \cdot \frac{g_\nu(I) \, \Phi_{\mu\nu}(x, W)}{\max_{\mu\nu}\bigl|g_\nu(I) \, \Phi_{\mu\nu}(x, W)\bigr|},$$

with local update function $\Phi_{\mu\nu}(x, W) = R^2 x_\mu - \langle W_\nu, x \rangle W_{\mu\nu}$ — first term is Hebbian, second is a homeostatic regulariser pulling weights to a sphere $\|W_\nu\|^2 = R^2$ (Oja-style).

The **global modulator** $g_\nu(I)$ is the key piece:

$$g_\nu(I) = \begin{cases} 1 & \text{if } r_\nu = K, \\ -\Delta & \text{if } r_\nu = K-\ell, \\ 0 & \text{otherwise.} \end{cases}$$

i.e. the **single most strongly driven** post-synaptic unit gets +1 Hebbian update, the *next $\ell$* most-driven get $-\Delta$ anti-Hebbian penalty, the rest get nothing. Authors use $\ell = 1$, $\Delta = 0.4$. This is the *neuromodulatory* layer: a global signal that selectively applies Hebbian / anti-Hebbian strategies across units that are not directly wired together.

**KH-modulated RBM.** Combine: at each step $t$, first apply a KH update to produce intermediate weights $\theta^{\text{KH}}_t$, then apply the standard RBM gradient step:

$$\theta^{\text{KH}}_t = \theta_t + \delta\theta^{\text{KH}}_t, \qquad \theta_{t+1} = \theta^{\text{KH}}_t - \eta \nabla_\theta \mathcal{L}(\theta^{\text{KH}}_t).$$

Two variants by choice of input $x$ to $\Phi$:
- **KH-TD** (top-down, cognition-driven): $x = h \sim p_\theta(h|v)$ — the hidden activations sampled given visible input.
- **KH-BU** (bottom-up, sensory-driven): $x = v \sim p_d(v)$ — the visible data directly.

The combined step can be expanded as a Langevin-like update $\theta_{t+1} = \theta_t - \eta[\nabla_\theta \mathcal{L} + \xi_t]$, where $\xi_t$ contains the KH contribution, Hessian curvature, and gated Oja terms. The authors anneal $\varepsilon$ down to zero with a $\varepsilon = \varepsilon_0 (1 - n/S)^{3/2}$ schedule, so KH is influential early in training and phases out by epoch $S$.

**Empirical analysis (Fig. 2 of the paper).** Cosine similarity between $\nabla_\theta \mathcal{L}$ and $\xi_t$ is *negative* during training (~-0.06), meaning KH modulation steers training **slightly opposite** to the RBM gradient — but **not random**: shuffling $\xi_t$ destroys the performance boost (Supplementary A). KH modulation is therefore not noise injection (à la SGLD); it systematically pushes the optimisation toward a different minimum where receptive fields are less redundant. Empirically, average max-cosine-similarity of hidden receptive fields with their most-overlapping neighbour drops from 0.41 (vanilla RBM) to 0.37 (KH-TD) and 0.35 (KH-BU) — quantifiable diversification.

**MNIST reconstruction results.** $M \in \{100, 500\}$ hidden units, CD1 / CD10, $\eta = 0.1$, batch 100, Std-Normal vs LeCun init, $R^{\text{std}}_\nu = 1$, $R^{\text{LeCun}}_\nu = 0.1$. KH-TD-RBM shows (i) **initialisation robustness** — both inits converge to the same accuracy, unlike vanilla RBM; (ii) slower convergence than LeCun-vanilla but faster than Std-Normal-vanilla; (iii) **better final accuracy** on both reconstruction MSE and cross-entropy. Annealing variants with shorter $S$ trade convergence speed for slightly worse but still better-than-vanilla accuracy.

**MNIST classification results (Table 1).** Using cRBMs (Larochelle et al. 2012). At $M = 500$:

| Method | LeCun | Std | Time per epoch |
|---|---|---|---|
| CD1 (vanilla) | 94.6% | 92.0% | 2.5 s |
| KH-TD-CD1 | 95.6% | **96.5%** | 4.5 s |
| CD10 (vanilla) | 95.8% | 93.8% | 9.0 s |
| KH-TD-CD10 | 96.9% | **97.0%** | 11.1 s |

KH-RBM with 500 units matches a vanilla shallow cRBM at 6000 units (10× reduction). Overfitting is virtually eliminated — the LeCun-init vanilla RBM's classic over-training dip is gone with KH-TD. Repeated on Kuzushiji-MNIST (harder) with consistent gains (Supplementary D).

**Conclusion.** KH-as-neuromodulation gives RBMs (a) initialisation robustness, (b) overfitting resilience, (c) more diverse receptive fields, (d) better accuracy or 10× parameter reduction at fixed accuracy. The mechanism is interpreted as a *biologically plausible attention/competition step* injected into vanilla RBM training. Future: same hybrid framework in deeper, more modern architectures.

## Phase 1 — Undergraduate-level synthesis

Here is the picture.

An **RBM** is a small two-layer generative network: a layer of pixel-like *visible* units and a layer of feature-like *hidden* units, fully connected between layers but with no within-layer wires. The lack of within-layer connections makes the network easy to train and analyse mathematically, but it has an annoying side effect — many hidden units end up learning the same feature, because there is no mechanism telling them to differentiate. (In biology this is solved by lateral inhibition: a strongly firing neuron inhibits its neighbours so they cannot grab the same feature.)

The paper's idea is to add lateral inhibition *without lateral wires*, using the brain's other trick for global coordination — **neuromodulation**. The brain's neuromodulators (dopamine, noradrenaline, acetylcholine) are chemicals broadcast diffusely; they do not respect the wiring graph, they just bathe many neurons in a state-dependent signal that tunes how strongly their synapses change.

The specific neuromodulatory rule is **Krotov-Hopfield (KH)**: every training step, look at which hidden unit is currently most strongly driven by the input. Strengthen its incoming weights (Hebbian "fire-together-wire-together"). Look at the next most strongly driven unit — *weaken* its incoming weights (anti-Hebbian). All other units, leave alone. The signal that decides "you are #1, you are #2, you are nobody" is the **global modulator** $g_\nu(I)$ — it is computed from the rank ordering of activations, not from connectivity.

Inject this KH update *in addition to* the standard RBM gradient update each step, anneal it away over a schedule, and you get:

1. **Initialisation no longer matters.** Both bad and good initial weights converge to similar final accuracy.
2. **No overfitting.** Training and validation curves stay parallel.
3. **Diverse features.** Hidden units learn less-redundant patterns (measured by lower cosine similarity).
4. **Better numbers.** 97% on MNIST classification at 500 hidden units, vs 96.5% for vanilla RBM at 6000 hidden units.

The take-home: a single global, rank-based "who fires loudest gets to learn" signal can substitute for the lateral connections that biology builds into cortical microcircuits — at a small training-time cost (~2× per epoch) and zero extra parameters.

## Phase 2 — Graduate-level deep dive

### RBM energy and learning rule

For a bipartite graph of $N$ binary visible units $v \in \{0,1\}^N$ and $M$ binary hidden units $h \in \{0,1\}^M$ with weights $W \in \mathbb{R}^{N \times M}$ and biases $(a, b)$, the **energy** is

$$
E_\theta(v, h) \;=\; -\, v^{\top} W h \;-\; a^{\top} v \;-\; b^{\top} h.
$$

The associated **Boltzmann distribution** factorises conditionally:

$$
p_\theta(v, h) \;=\; \frac{e^{-E_\theta(v, h)}}{Z_\theta}, \qquad Z_\theta \;=\; \sum_{v, h} e^{-E_\theta(v, h)}.
$$

Marginalising hidden units gives the **clamped free energy** $F_\theta(v)$ via $p_\theta(v) = e^{-F_\theta(v)} / Z_\theta$. Training minimises the **negative log-likelihood**

$$
\mathcal{L} \;=\; \langle -\log p_\theta(v) \rangle_{v \sim p_d(v)} \;=\; \langle F_\theta(v) \rangle_{v \sim p_d} + \log Z_\theta,
$$

whose weight gradient reduces to the textbook RBM rule

$$
\delta W_{ij} \;=\; \eta \bigl[ \langle v_i h_j \rangle_d \;-\; \langle v_i h_j \rangle_m \bigr],
$$

with $\langle \cdot \rangle_d$ the data expectation and $\langle \cdot \rangle_m$ the model expectation, the latter approximated by **CD-$k$** with $k$ Gibbs sweeps from a data sample.

### The Krotov-Hopfield energy and update

The **Krotov-Hopfield energy** lineage is the dense associative-memory line: starting from Hopfield's quadratic energy

$$
E_{\text{Hop}}(\mathbf{s}) \;=\; -\tfrac{1}{2}\, \mathbf{s}^{\top}\, J\, \mathbf{s},
$$

Krotov & Hopfield 2016 generalised to an interaction polynomial of higher order,

$$
E_{\text{KH-mem}}(\mathbf{s}) \;=\; -\sum_{\mu} F\!\bigl(\langle \xi^{\mu}, \mathbf{s} \rangle\bigr),
$$

with $F(\cdot)$ a fast-growing (e.g. polynomial $z^n$ or exponential) interaction. Tambaş et al. do **not** use the dense associative-memory energy directly — instead they use the **2019 PNAS Krotov-Hopfield biologically plausible *learning rule*** built on Bienenstock-Cooper-Munro (BCM) theory.

Per-synapse local update function (Eq. 7 of the paper):

$$
\boxed{\quad \Phi_{\mu\nu}(x, W) \;=\; R^{2}\, x_{\mu} \;-\; \langle W_\nu, x \rangle\, W_{\mu\nu} \quad}
$$

with the input current $I_\nu = \langle W_\nu, x \rangle$. The first term is the **Hebbian** kernel $x_\mu I_\nu$ (post-times-pre activity); the second term is a **homeostatic normalisation** that pulls each post-synaptic neuron's incoming weight vector onto the sphere $\sum_\mu W_{\mu\nu}^2 = R^2$. With $g_\nu = I_\nu$ uniformly, $\Phi$ recovers **Oja's rule**.

The KH weight update (Eq. 6):

$$
\boxed{ \quad \delta W^{\text{KH}}_{\mu\nu} \;=\; \varepsilon \cdot \frac{ g_\nu(I)\, \Phi_{\mu\nu}(x, W) }{ \max_{\mu\nu}\bigl| g_\nu(I)\, \Phi_{\mu\nu}(x, W) \bigr| } \quad }
$$

The denominator normalises the update so the largest entry has unit magnitude, making $\varepsilon$ a clean step-size hyperparameter.

### The neuromodulation-augmented variant — the $g_\nu(I)$ global modulator

The central novelty is the **rank-based modulator** (Eq. 8 of the paper):

$$
\boxed{ \quad g_\nu(I) \;=\; \begin{cases} +1, & r_\nu = K \quad\text{(rank-1, strongest input)} \\ -\Delta, & r_\nu = K - \ell \quad\text{(next $\ell$ strongest)} \\ 0, & \text{otherwise.} \end{cases} \quad }
$$

with $K$ the number of post-synaptic units, $\ell$ the number of suppressed runners-up ($\ell = 1$), and $\Delta > 0$ the suppression strength ($\Delta = 0.4$). Three properties:

1. **Winner-take-all + runner-up penalty.** Only the most-driven unit learns positively; the second-most-driven is actively pushed away from the current pattern.
2. **Global computation, local update.** $r_\nu$ is computed from the *rank-ordering* of activations across the entire layer — a global view — but applied as a per-synapse multiplier. This is the formal embodiment of a "three-factor" rule $(W, x, g)$.
3. **No lateral wires required.** $g_\nu$ is a function of activations, not of any explicit lateral connection between units.

This is the paper's interpretation of the rule as **neuromodulation**: $g_\nu$ is the broadcast modulatory signal, $\Phi_{\mu\nu}$ is the local Hebbian update, and the product is the gated three-factor learning rule.

### Combining KH neuromodulation with RBM training

At each time step (Eq. 9):

$$
\theta^{\text{KH}}_{t} \;=\; \theta_t + \delta\theta^{\text{KH}}_{t}, \qquad
\theta_{t+1} \;=\; \theta^{\text{KH}}_{t} - \eta\, \nabla_\theta \mathcal{L}\bigl(\theta^{\text{KH}}_{t}\bigr).
$$

A Taylor expansion casts the combined step as a noise-augmented gradient descent

$$
\theta_{t+1} \;=\; \theta_t - \eta\, \bigl[ \nabla_\theta \mathcal{L}(\theta_t) + \xi_t \bigr],
$$

where

$$
\xi_t \;\simeq\; \bigl(\delta\theta^{\text{KH}}_{t}\bigr) \cdot \nabla_\theta\bigl(\nabla_\theta \mathcal{L}(\theta_t)\bigr) \;-\; \eta^{-1}\, \delta\theta^{\text{KH}}_{t}
$$

couples the KH update to the local Hessian of the RBM loss. Empirically Figure 2 of the paper shows $|\xi_t| / |\nabla\mathcal{L}|$ decaying from ~0.4 at start to ~0 by epoch 300 (annealing), with a small but persistent **negative cosine similarity** ~-0.06 between $\xi_t$ and $\nabla\mathcal{L}$. The shuffling-control rules out SGLD-style noise as an explanation: shuffled $\xi_t$ does not yield the boost. Therefore KH modulation *systematically* steers training to an alternative minimum where receptive fields are more diverse.

### Annealing schedule

$$
\varepsilon(n) \;=\; \varepsilon_0 \, \bigl(1 - n / S\bigr)^{3/2},
$$

with $n$ the current epoch and $S$ the total annealing length. Super-linear decay was chosen so that, after KH-modulation fully phases out, pure RBM-gradient steps have enough time to converge to a minimum *in the basin selected by KH*. Across $S \in \{50, 100, 200, 300, 400, 500\}$, KH-TD-RBM beats vanilla-RBM at the end of the modulation window and stays above for the rest of training.

### Computational mechanism summary

The KH-modulated RBM is governed by three coupled equations:
1. Energy / Boltzmann distribution $p_\theta(v, h) \propto e^{-E_\theta(v, h)}$.
2. CD-$k$ gradient $\delta W_{ij}^{\text{RBM}} = \eta(\langle v_i h_j \rangle_d - \langle v_i h_j \rangle_m)$.
3. Neuromodulator-gated three-factor KH update $\delta W^{\text{KH}}_{\mu\nu} = \varepsilon \cdot g_\nu(I) \cdot \Phi_{\mu\nu}(x, W) / Z(I, W, x)$, with rank-based modulator $g_\nu$.

The two updates are interleaved per step. The neuromodulator $g_\nu$ provides a winner-take-all-with-runner-up-penalty competition that substitutes for missing lateral connectivity in the hidden layer, diversifies receptive fields, and converts the RBM training landscape into one with better attractors. Cost: ~2× training-time per epoch; no extra parameters.

### Empirical findings (recap)

- Final receptive-field overlap drops from 0.41 (vanilla) to 0.35–0.37 (KH variants) — quantitative diversification.
- Reconstruction MSE on validation strictly lower for KH-TD-RBM across all annealing schedules tested.
- Classification accuracy at $M = 500$ units, Std init, CD-10: **KH-TD-CD10 reaches 97.0%**, matching a 6000-hidden-unit vanilla shallow cRBM — a 10-fold parameter reduction.
- Overfitting eliminated (validation curve no longer dips below training curve, characteristic of LeCun-init vanilla RBMs).
- All results reproduced on Kuzushiji-MNIST (Supplementary D).

## Connections to other papers in this corpus

- **Hopfield / associative-memory cluster.** This paper is in the energy-based, associative-memory lineage that includes:
  - **Alonso & Krichmar 2023** (`alonso_krichmar_2023_sparse_quantized_hopfield`, other batch) — sparse quantised Hopfield for online-continual memory.
  - **Osman et al. 2024** (`osman_2024_hopfield_arousal`, other batch) — Hopfield model of neuromodulatory arousal state.
  - The dense-associative-memory line (Krotov & Hopfield 2016 *NeurIPS* and the 2019 PNAS biologically plausible rule [ref. 1 of this paper]) is the upstream methodological seed.
- **Three-factor / neuromodulation-as-gating.** The paper explicitly cites *three-factor learning rules* (Fremaux & Gerstner 2016, ref. 21) and Shine 2019 (ref. 20) as biological context. The rank-based $g_\nu$ aligns with:
  - **Doya 2002** (`doya_2002_metalearning_neuromodulation`, other batch) — neuromodulators as global hyperparameter controllers.
  - **Mei et al. 2023 iScience** (ref. 22 here; related to `mei_2022_multiscale_neuromodulation`, other batch) — neuromodulation for dynamic hyperparameter adjustment.
  - **Vecoven et al. 2020** (`vecoven_2020_neuromodulation_deep_nets`, other batch) — explicit neuromodulator-as-gate in deep nets.
- **Kudithipudi 2022 mechanism row.** Sits in the "metaplasticity" + "neuromodulation" rows of Kudithipudi et al. 2022 (`kudithipudi_2022_lifelong_learning`, this batch), supporting *overcoming catastrophic forgetting* and *task-similarity exploitation*.
- **Durstewitz 2025 BTSP analogue.** The "rapid one-shot encoding" framing of Durstewitz et al. 2025 (`durstewitz_2025_neuroscience_continual_learning`, this batch) on BTSP-based content-addressable memory is conceptually adjacent — both papers argue that a single, biologically plausible, *competitive* learning rule can match deep-net capacity at a fraction of the cost.
- **Gain modulation vs. attentional gating.** This paper interprets KH modulation as a "rudimentary attention mechanism without backpropagation"; the broader attention-via-neuromodulation framing connects to:
  - **Rodriguez-Garcia et al. 2026** (`rodriguezgarcia_2026_ne_stability_gap`, this batch) — gain modulation as a continuous-valued NA-inspired neuromodulator.
  - **Ferguson & Cardin 2020** (`ferguson_cardin_2020_gain_modulation`, other batch) — cortical gain-modulation review.
  - **Wainstein et al. 2025** (`wainstein_2025_gain_perceptual_switches`, other batch) — gain modulation in perceptual switches.
- **Kolouri et al. 2019** (`kolouri_2019_attention_based_structural_plasticity`, other batch) — attention-based structural plasticity is methodologically akin (selective protection of important units).
- **Spiking / neuromorphic SNN siblings.** Espino et al. 2024 (`espino_2024_snn_path_planning`, this batch) and AlKilany & Goodman 2025 (`alkilany_goodman_2025_snn_dynamic_sensory`, this batch) share the local-learning-rule + neuromorphic-deployability bias. Together with Tambaş et al. they form the "biologically plausible, backprop-free" wing of the corpus.
