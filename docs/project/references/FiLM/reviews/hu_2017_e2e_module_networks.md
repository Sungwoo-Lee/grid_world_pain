---
title: "Learning to Reason: End-to-End Module Networks for Visual Question Answering"
authors: ["Ronghang Hu", "Jacob Andreas", "Marcus Rohrbach", "Trevor Darrell", "Kate Saenko"]
year: 2017
venue: "ICCV 2017"
slug: hu_2017_e2e_module_networks
source_pdf: "sources/Hu et al. 2017 - Learning to reason - End-to-End Module Networks for visual question answering.pdf"
topic: FiLM
---

# Hu et al. 2017 — End-to-End Module Networks (N2NMN)

## Plain-English entry point

This paper is the direct successor to **Neural Module Networks (Andreas et al. 2016)**. NMN's recipe was: parse the question with an off-the-shelf grammar parser, map the parse tree to a tree of typed neural-network "modules", and execute the assembled network. The catch: off-the-shelf parsers are brittle for vision-language questions (they're trained on news text, not VQA), and the module layout is therefore not *learned* — it's externally imposed.

**End-to-End Module Networks (N2NMNs)** replace the parser with a **learned layout policy**. A sequence-to-sequence RNN reads the question and emits two parallel streams:
1. A **structural stream** — a sequence of module tokens (in reverse-Polish notation) specifying the tree shape, e.g. `[find, find, find, and, eq_count]`.
2. A **textual-attention stream** — for each module token, a soft attention over the question's words, producing a textual parameter vector for that module instance.

These two streams together let the network builder assemble a question-specific computation tree. The whole pipeline is trained end-to-end: the structural prediction is discrete, so it's optimized by **policy gradient (REINFORCE)** using the final answer accuracy as the reward; the module weights and the textual-attention weights are trained by backprop. Training is bootstrapped with **behavioral cloning** — first imitate expert layouts (constructed from CLEVR's ground-truth functional programs), then continue with reinforcement learning to refine.

The model achieves **83.7%** on the CLEVR benchmark (vs. 72.1% for NMN with expert parser layouts at test time, and 68.5% for the strongest CNN-LSTM+spatial-attention baseline) — roughly a 50% error reduction. On the SHAPES dataset, behavioral cloning hits 100%. N2NMN is the bridge between symbolic module composition (Andreas 2016) and fully learned compositional reasoning, and a key reference for FiLM (which arrived at similar CLEVR results via a continuous-modulation route).

## Section-ordered backbone

**1. Introduction.** VQA datasets like CLEVR demand compositional reasoning (multi-step queries: locate object → check property → compare across objects). Monolithic CNN-LSTM models discover statistical biases rather than reason. NMN [Andreas 2016] proposed compositional modules but relies on external parsers. N2NMN learns *both* the layout policy *and* the module parameters jointly. Soft-attention textual parameterization replaces hard-coded module instances (`find[cat]` etc.).

**2. Related work.**
- **NMN family**: Andreas et al. 2016 (fixed parse layouts); D-NMN Andreas 2016b (reranking 3–10 parser candidates). This paper searches the *full* layout space and needs no parser at test time.
- **Compositional modular networks** (Hu et al. 2017 for referring expressions): fixed (subject, relationship, object) layout — restricted compared to N2NMN's open search.
- **NN architecture search**: evolutionary (Stanley), Bayesian (Bergstra), reinforcement learning (Zoph & Le 2017 — closest analog). Difference: Zoph–Le learns *one fixed architecture per dataset*; N2NMN learns *one architecture per example*.
- **VQA approaches**: differentiable memory (Sukhbaatar et al.), question-specific computations (Andreas et al.), multi-modal pooling (Fukui et al.). Concurrent work (Johnson et al. 2017 "Inferring and Executing Programs") uses generic-typed modules and hard-coded instantiation; N2NMN differs by using specialized modules + soft textual attention.

**3. End-to-End Module Networks.**

**3.1 Attentional neural modules.** Each module is $y = f_m(a_1, a_2, \dots; x_{vis}, x_{txt}, \theta_m)$, taking 0–2 attention map inputs, visual features $x_{vis}$ (VGG-16 pool5 + spatial coords, $15 \times 10 \times 514$ on CLEVR), and a textual vector $x_{txt}^{(m)}$. The full module set (Table 1):

| Module | Att-inputs | Output | Implementation |
|---|---|---|---|
| `find` | (none) | att | $a_{out} = \mathrm{conv}_2(\mathrm{conv}_1(x_{vis}) \odot W x_{txt})$ |
| `relocate` | $a$ | att | $a_{out} = \mathrm{conv}_2(\mathrm{conv}_1(x_{vis}) \odot W_1 \, \mathrm{sum}(a \odot x_{vis}) \odot W_2 x_{txt})$ |
| `and` | $a_1, a_2$ | att | $\min(a_1, a_2)$ |
| `or` | $a_1, a_2$ | att | $\max(a_1, a_2)$ |
| `filter` | $a$ | att | $\mathrm{and}(a, \mathrm{find}[x_{vis}, x_{txt}]())$ |
| `exist`, `count` | $a$ | answer | $y = W^\top \mathrm{vec}(a)$ |
| `describe` | $a$ | answer | $y = W_1^\top (W_2 \, \mathrm{sum}(a \odot x_{vis}) \odot W_3 x_{txt})$ |
| `eq_count`, `more`, `less` | $a_1, a_2$ | answer | $y = W_1^\top \mathrm{vec}(a_1) + W_2^\top \mathrm{vec}(a_2)$ |
| `compare` | $a_1, a_2$ | answer | $y = W_1^\top (W_2 \, \mathrm{sum}(a_1 \odot x_{vis}) \odot W_3 \, \mathrm{sum}(a_2 \odot x_{vis}) \odot W_4 x_{txt})$ |

Each module $m$ obtains its textual feature by a soft attention $\alpha^{(m)}$ over the $T$ question words: $x_{txt}^{(m)} = \sum_i \alpha^{(m)}_i w_i$ where $w_i$ is the embedding for word $i$.

**3.2 Layout policy.** A sequence-to-sequence attentional RNN. Encoder LSTM reads the question $q$ producing hidden states $[h_1, \dots, h_T]$. Decoder LSTM emits module-token sequence $\{m^{(t)}\}$ in reverse-Polish notation. At each decoder step $t$, an attention map $\alpha_{ti}$ over input words is computed (Bahdanau attention); $\alpha_{ti}$ doubles as the textual attention $\alpha^{(m^{(t)})}$ for module $m^{(t)}$. Probability of layout $l$: $p(l|q) = \prod_t p(m^{(t)} | m^{(1)}, \dots, m^{(t-1)}, q)$. Test-time: beam search.

**3.3 End-to-end training.** Joint loss

$$
L(\theta) = \mathbb{E}_{l \sim p(l|q;\theta)}\!\big[\tilde L(\theta, l; q, I)\big]
$$

where $\tilde L$ is the softmax cross-entropy on answer prediction. Layout $l$ is discrete; use REINFORCE:

$$
\nabla_\theta L = \mathbb{E}_{l \sim p}\!\big[\tilde L(\theta, l) \nabla_\theta \log p(l|q;\theta) + \nabla_\theta \tilde L(\theta, l)\big].
$$

MC estimated with $M = 1$ sample. Variance reduction via baseline $b$ (exponential moving average of recent $\tilde L$). Entropy regularization $\alpha = 0.005$ over $p(l|q)$ encourages layout exploration.

**Behavioral cloning bootstrap.** Pre-train the policy by KL-minimization $D_{KL}(p_e \| p)$ against an expert policy $p_e$ constructed from ground-truth parses (SHAPES) or CLEVR functional programs. The expert is *not* used at test time. After cloning, switch to policy-gradient fine-tuning.

**4. Experiments.**

**4.1 SHAPES.** 15,616 image-question pairs, 244 unique questions. Settings: (i) behavioral cloning from expert achieves 100% (NMN was 90.8%); (ii) policy search from scratch achieves 96.19%. Two interpretive findings: the soft-attention modules outperform NMN's hard-coded ones even with the same layouts; pure policy search works on this small dataset.

**4.2 CLEVR.** 100K images, 853K questions. Photorealistic scenes, multi-step reasoning chains. VGG-16 pool5 visual features. Results (Table 3):
- CNN+BoW: 48.4%
- CNN+LSTM+SA: 68.5%
- NMN (expert layout at test): 72.1%
- N2NMN policy search from scratch: 69.0%
- N2NMN cloning expert: 78.9%
- **N2NMN policy search after cloning: 83.7%** (state-of-the-art at submission)

Largest gains on "compare color" (52.5 → 82.8%). Predicted layouts are *interpretable* — Figure 5 shows the model attending to "right" in a `find` module that supplies the spatial cue for a downstream `relocate`.

**4.3 VQA.** Real images. Smaller margins because VQA questions are mostly non-compositional and statistical baselines do well; N2NMN matches state-of-the-art (64.9% on test-dev).

**5. Conclusion.** Learned layout + soft-attention module parameterization beats both monolithic networks and parser-driven NMNs on compositional benchmarks. N2NMN's wins are largest precisely where compositional reasoning is most required (CLEVR).

## Phase 1 — undergraduate-level synthesis

**Key idea.** Neural Module Networks (Andreas 2016) showed that *compositional reasoning* works if you assemble small typed modules into question-specific networks. But they relied on an external grammar parser to decide which modules to use and how to wire them. That's brittle. **N2NMN replaces the parser with a neural network that learns the layout** alongside everything else.

**Setup.**
1. **Modules** are a small library: `find` (attention from text+image), `relocate` (shift attention by a textual cue), `and`/`or` (combine attentions), `filter`, `compare`, `describe`, `count`, `exist`, etc. Each has its own learnable parameters.
2. A **seq2seq RNN** reads the question and outputs two things in parallel: (a) a sequence of module tokens — the *structure* of the computation tree, in reverse-Polish notation — and (b) for each module, a soft attention over the question's words that supplies that module's "textual parameter" (e.g., "matte ball" for a `find` module).
3. A **network builder** turns the token sequence into an actual neural network, and runs it on the image features to produce an answer.

**Training.** The structure prediction is discrete (you sample tokens from a softmax), so you can't backprop through it. Use **reinforcement learning** (REINFORCE): reward = answer correctness, gradient = $\tilde L \cdot \nabla \log p(l|q)$. The module weights are continuous, so they train by ordinary backprop. **Bootstrap** with *behavioral cloning* — first train the layout RNN to imitate expert parses, then continue with RL.

**Headline result.** On CLEVR (the hard compositional VQA benchmark), N2NMN reaches 83.7% vs. the previous best 72.1% (NMN with expert parser at test) and 68.5% (CNN-LSTM+attention). On the small SHAPES dataset, behavioral cloning alone reaches 100%, and pure policy search from scratch reaches 96.2%.

**Initial takeaway.** N2NMN is the **fully-learned** version of NMN — both *what to compute* (the module layout) and *how each module behaves* are trained from data, not hard-coded by a parser. It's the closest module-network counterpart to FiLM (Perez et al. 2018, in another batch), which arrives at the same CLEVR-level reasoning by a completely different route: a single CNN with question-conditioned feature-wise affine modulation. The two methods are roughly competitive on CLEVR; N2NMN's edge is interpretability of the predicted layout, FiLM's edge is simplicity (no RL, no policy gradient).

## Phase 2 — graduate-level deep dive

### Module-composition formulation

A **layout** $l$ is a tree of module instances, equivalently a sequence of module tokens $\{m^{(t)}\}_{t=1}^L$ in **reverse-Polish (post-order) notation**. The network builder consumes the token sequence, popping the required number of attention-map arguments from a stack at each token and pushing the output. Final result: a single answer-distribution tensor.

For each module $m$ in the assembled tree, its **textual feature** is a soft attention over the question words:

$$
x_{txt}^{(m)} = \sum_{i=1}^{T} \alpha^{(m)}_i \, w_i,
$$

where $w_i \in \mathbb{R}^{300}$ is the GloVe-style embedding of word $i$ and $\alpha^{(m)} \in \Delta^{T-1}$ is a learned attention emitted by the layout policy at the time-step when $m$ is emitted.

The full module signature is:

$$
y = f_m\!\big(a_1, a_2, \dots; x_{vis},\, x_{txt}^{(m)},\, \theta_m\big),
$$

with $x_{vis} \in \mathbb{R}^{H \times W \times C}$ a spatial feature map (15×10×514 on CLEVR — VGG-16 pool5 plus 2 spatial-coordinate channels).

### Worked module examples

**`find`** (zero attention inputs, produces an attention map):

$$
a_{out} = \mathrm{conv}_2\!\Big(\mathrm{conv}_1(x_{vis}) \odot W x_{txt}\Big),
$$

where $\odot$ is broadcast element-wise multiplication and $W$ projects the textual vector to the channel dimension of $\mathrm{conv}_1(x_{vis})$. This is *structurally identical to a single FiLM layer* with $\gamma = W x_{txt}$ and $\beta = 0$ — modulation by the textual feature followed by a conv.

**`relocate`** (transforms one attention, takes both visual and textual context):

$$
a_{out} = \mathrm{conv}_2\!\Big(\mathrm{conv}_1(x_{vis}) \odot W_1 \, \mathrm{sum}(a \odot x_{vis}) \odot W_2 x_{txt}\Big).
$$

Here $\mathrm{sum}(a \odot x_{vis}) \in \mathbb{R}^C$ pools image features under the current attention to produce a "subject vector"; multiplying it into the conv lets `relocate` re-direct attention conditioned on what was just attended to plus a textual cue.

**`compare`** (binary, both visual and textual):

$$
y = W_1^\top\!\Big( W_2 \, \mathrm{sum}(a_1 \odot x_{vis}) \odot W_3 \, \mathrm{sum}(a_2 \odot x_{vis}) \odot W_4 x_{txt} \Big).
$$

### Layout-prediction step (sequence-to-sequence with attention)

**Encoder.** Multi-layer LSTM on word embeddings $w_1, \dots, w_T$ producing $h_1, \dots, h_T$.

**Decoder.** LSTM producing module tokens $\{m^{(t)}\}$. At decoder step $t$ with hidden state $h_t^{dec}$:

$$
u_{ti} = v^\top \tanh(W_1 h_i + W_2 h_t^{dec}), \qquad \alpha_{ti} = \frac{\exp(u_{ti})}{\sum_{j=1}^T \exp(u_{tj})},
$$

$$
c_t = \sum_{i=1}^T \alpha_{ti} h_i, \qquad p\!\big(m^{(t)} \mid m^{(1:t-1)}, q\big) = \mathrm{softmax}(W_3 h_t^{dec} + W_4 c_t).
$$

Sample $m^{(t)} \sim p(m^{(t)} \mid \dots, q)$ to obtain the discrete next token. Crucially, the same $\alpha_{ti}$ is reused as the **textual attention** $\alpha^{(m^{(t)})}_i$ for module $m^{(t)}$'s textual feature $x_{txt}^{(m^{(t)})} = \sum_i \alpha_{ti} w_i$. The layout policy thus emits *both* the structural action (which module type) and the parametric action (which words to attend to) at every step.

The layout probability factorizes:

$$
p(l \mid q) = \prod_{t=1}^{L} p\!\big(m^{(t)} \mid m^{(1:t-1)}, q\big).
$$

At test time, the maximum-probability layout is recovered by beam search.

### REINFORCE training

Final loss for the joint model:

$$
L(\theta) = \mathbb{E}_{l \sim p(l|q;\theta)}\!\Big[ \tilde L\big(\theta, l; q, I\big) \Big],
$$

where $\tilde L$ is the softmax cross-entropy on answer prediction given the network assembled from $l$. The gradient:

$$
\nabla_\theta L = \mathbb{E}_{l \sim p(\cdot|q;\theta)}\!\Big[ \tilde L(\theta, l) \, \nabla_\theta \log p(l|q;\theta) + \nabla_\theta \tilde L(\theta, l) \Big].
$$

The first term is the **REINFORCE** policy-gradient term — non-differentiable in $l$, so the score-function estimator is used. The second term is the **standard backprop** through the module parameters and the textual-attention pathway. Monte Carlo estimate with $M = 1$ sample per question:

$$
\nabla_\theta L \approx \big(\tilde L(\theta, l_m) - b\big)\, \nabla_\theta \log p(l_m | q; \theta) + \nabla_\theta \tilde L(\theta, l_m),
$$

with baseline $b$ an exponential moving average of recent $\tilde L$ for variance reduction. Entropy regularization $\alpha = 0.005 \cdot H[p(l|q)]$ encourages exploration.

### Behavioral-cloning bootstrap

REINFORCE from scratch is hard because the reward signal is sparse and the policy has to simultaneously learn structure and parameters. Bootstrap with an expert policy $p_e$ constructed from ground-truth parses (SHAPES) or CLEVR's functional programs. First stage loss:

$$
L_{clone}(\theta) = D_{KL}\!\big(p_e \| p_\theta\big) + \tilde L\big(\theta, l_e; q, I\big),
$$

i.e., minimize KL to the expert layout *and* the answer loss using the expert's layout for the rollout. After clone-pretraining, discard $p_e$ and switch to the REINFORCE objective above.

### Empirical anchor (CLEVR)

| Method | Overall | Compare-color | Count |
|---|---|---|---|
| CNN+BoW | 48.4% | 51% | 38.9% |
| CNN+LSTM+SA | 68.5% | 51% | 52.2% |
| NMN (expert layout at test) | 72.1% | 74.4% | 52.5% |
| N2NMN policy search from scratch | 69.0% | 53.9% | 55.1% |
| N2NMN cloning expert | 78.9% | 52.5% | 63.3% |
| **N2NMN policy search after cloning** | **83.7%** | **82.8%** | **68.5%** |

Two-stage training (clone → RL refine) gives the biggest gains on *compare* questions, which require the most multi-step reasoning.

## Connections

- **`andreas_2016_neural_module_networks.md`** (this batch): direct parent. Same module-library philosophy and typed-tree composition; N2NMN replaces the parser-based hard-coded layout with a learned RNN policy and the per-instance module weights (`find[dog]`, `find[cat]`) with a single per-type module taking a textual-attention vector.
- **`shazeer_2017_sparse_moe.md`** (this batch): both N2NMN and Sparse-MoE select a sparse subset of computation units per example. MoE routes by a *continuous gating softmax* trained by backprop; N2NMN routes by a *discrete RL policy*. MoE's "all-experts-feedforward-bank" is replaced in N2NMN by a "tree of typed modules". Both face the same fundamental "discrete-routing-in-a-differentiable-architecture" challenge — MoE solves it by smoothing, N2NMN solves it by REINFORCE.
- **`ha_2016_hypernetworks.md`** (this batch): N2NMN's `find` module is structurally a single FiLM/hypernet layer ($\mathrm{conv}_2(\mathrm{conv}_1(x_{vis}) \odot W x_{txt})$). The whole N2NMN can be viewed as a *hypernetwork tree*: each module is a tiny hypernet conditioned on its textual attention, and the layout policy is a hypernet over the topology itself.
- **`galanti_wolf_2020_hypernet_modularity.md`** (this batch): N2NMN's modules are the "primary networks" (small, shared across questions); the layout policy + textual-attention RNN is the analogue of the "hypernetwork" (large, where capacity lives). The modularity argument applies in spirit — capacity goes into the layout/composer.
- **`perez_2018_film.md`** (other batch — FiLM): direct competitor on CLEVR. FiLM uses a *fixed CNN topology* with question-conditioned per-channel affine modulation at every layer to reach ~97% on CLEVR (vs N2NMN's 83.7%). The lesson of the two papers together: dense continuous conditional modulation (FiLM) actually beats discrete tree composition (N2NMN) on CLEVR, *but* N2NMN gives interpretable layouts that FiLM does not.
- **`krueger_2017_bayesian_hypernets.md`** (this batch): a "Bayesian N2NMN" would place a posterior over the layout policy $p(l|q)$ itself, sampling alternative layouts and propagating predictive uncertainty across them. The combinatorial layout space is a natural setting for posterior sampling, though no one has reported this.
