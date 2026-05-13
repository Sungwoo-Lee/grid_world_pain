# Hypernetwork Literature Review

## Scope

This review covers eight papers that trace hypernetworks from their initial
formalisation as end-to-end gradient-trained weight generators (Ha, Dai & Le,
2016) through theoretical analysis (Galanti & Wolf, 2020), neuroscience-flavoured
generative modelling (Jiang, Gklezakos & Rao, 2021), Bayesian meta-learning
(Borycki et al., 2022), continual and meta-RL (Schöpf et al., 2022; Beck et al.,
2023), zero-shot RL transfer (Rezaei-Shoshtari et al., 2023), and a 2023 survey
(Chauhan et al., 2023). All sources are local PDFs in
`docs/project/references/Hypernetwork/`.

The review serves the project's question of whether hypernet-style weight
generation is a competitive alternative to FiLM-style affine modulation as the
substrate for context- / precision-dependent control in an interoceptive
grid-world agent. Connections back to the FiLM corpus
(`docs/project/references/FiLM/`), the Bayesian / uncertainty corpus
(`docs/project/references/Bayesian_Neural_net/`,
`docs/project/references/uncertainty/`) and the neuromodulatory corpus
(`docs/project/references/neuromodulatory_algorithms/`) are surfaced *only* where
the paper itself supports them.

## Reading order

- **Paper 1 — HyperNetworks (Ha, Dai & Le, 2016).** The foundational paper.
  Defines the static (convolutional) and dynamic (recurrent) hypernetwork in a
  form that is end-to-end trainable, and shows that a small "embedding-driven"
  network can generate competitive convolutional or recurrent weights with
  drastically fewer learnable parameters.
- **Paper 2 — On the Modularity of Hypernetworks (Galanti & Wolf, 2020).**
  Approximation-theoretic comparison between hypernetwork and embedding-based
  conditioning. Proves that, for smooth target functions, the inner network of a
  hypernetwork can be smaller than an embedding-method's primary network by
  orders of magnitude — the *modularity* property.
- **Paper 3 — Dynamic Predictive Coding with Hypernetworks (Jiang, Gklezakos &
  Rao, 2021).** Embeds a hypernetwork inside a Rao–Ballard predictive-coding
  generative model so that a higher cortical level controls the *transition
  dynamics* of a lower level. Recovers V1-like simple-cell space–time receptive
  fields and a slow–fast cortical timescale hierarchy as emergent properties.
- **Paper 4 — Hypernetwork approach to Bayesian MAML (Borycki et al., 2022).**
  Replaces MAML's per-task gradient-based adaptation with a hypernetwork that
  *samples* posterior task-specific parameters from a Gaussian. Yields a
  closed-form Bayesian variant of MAML with explicit posterior uncertainty.
- **Paper 5 — Hypernetwork-PPO for Continual Reinforcement Learning (Schöpf et
  al., 2022).** Demonstrates that a task-conditioned hypernetwork plus a
  regularisation term that stabilises previously generated weights enables a PPO
  agent to learn a sequence of Procgen tasks without catastrophic forgetting.
- **Paper 6 — Hypernetworks in Meta-Reinforcement Learning (Beck et al., 2023).**
  Empirical study of which design choices (hypernet vs. concatenation, generated
  vs. shared parameters, supervised pre-training) are necessary to make
  hypernetworks beat the standard meta-RL baseline.
- **Paper 7 — Hypernetworks for Zero-Shot Transfer in Reinforcement Learning
  (Rezaei-Shoshtari et al., 2023).** Generates entire actor and critic networks
  from a low-dimensional task descriptor, with a Lipschitz-bounded hypernet
  trained jointly with model-based imagined rollouts. Achieves zero-shot
  transfer to unseen continuous-control tasks.
- **Paper 8 — A Brief Review of Hypernetworks in Deep Learning (Chauhan et al.,
  2023).** Survey paper consolidating the field. Synthesised here last so the
  reader can use it as a map back into Papers 1–7.

## Table of Contents

1. [Paper 1 — HyperNetworks (Ha, Dai & Le, 2016)](#paper-1) — written
2. [Paper 2 — On the Modularity of Hypernetworks (Galanti & Wolf, 2020)](#paper-2) — written
3. [Paper 3 — Dynamic Predictive Coding with Hypernetworks (Jiang, Gklezakos & Rao, 2021)](#paper-3) — written
4. [Paper 4 — Hypernetwork approach to Bayesian MAML (Borycki et al., 2022)](#paper-4) — written
5. [Paper 5 — Hypernetwork-PPO for Continual RL (Schöpf et al., 2022)](#paper-5) — written
6. [Paper 6 — Hypernetworks in Meta-RL (Beck et al., 2023)](#paper-6) — written
7. [Paper 7 — Hypernetworks for Zero-Shot Transfer in RL (Rezaei-Shoshtari et al., 2023)](#paper-7) — written
8. [Paper 8 — A Brief Review of Hypernetworks in Deep Learning (Chauhan et al., 2023)](#paper-8) — written
9. [Cross-paper synthesis & relevance to interoceptive-pain RL](#synthesis) — written

---

## <a id="paper-1"></a>Paper 1 — HyperNetworks (Ha, Dai & Le, 2016)

**Citation:** Ha, D., Dai, A., & Le, Q. V. (2016). *HyperNetworks.* ICLR 2017
(arXiv:1609.09106v4).

**PDF:** `docs/project/references/Hypernetwork/sources/Ha et al. 2016 - HyperNetworks.pdf` (29 pages incl. appendix)

### Backbone

- **Problem.** Convolutional networks waste parameters by giving each layer its
  own kernel; recurrent networks share a single weight matrix across all
  timesteps and so are inflexible. The authors propose *hypernetworks* — small
  networks that *generate* the weights of a larger primary network — as a
  *relaxed* form of weight sharing that interpolates between these extremes,
  trained end-to-end with backprop rather than evolution as in HyperNEAT.
- **Static hypernetwork (Section 3.1).** For a convolutional layer with kernel
  $K^j \in \mathbb{R}^{N_{\mathrm{in}} f_{\mathrm{size}} \times N_{\mathrm{out}}
  f_{\mathrm{size}}}$, a learnable layer-embedding $z^j \in \mathbb{R}^{N_z}$
  is mapped through a *two-layer linear* hypernetwork
  $g(z^j)$ that produces the kernel slice-by-slice. The first layer linearly
  projects $z^j$ into $N_{\mathrm{in}}$ intermediate vectors $a^j_i \in
  \mathbb{R}^d$ via per-slice matrices $W_i, B_i$; a *shared* tensor
  $W_{\mathrm{out}}$ contracts each $a^j_i$ to a kernel slice $K^j_i$ which is
  then concatenated into the full $K^j$. The trainable parameter count is
  $N_z D + d (N_z+1) N_{\mathrm{in}} + f_{\mathrm{size}} N_{\mathrm{out}}
  f_{\mathrm{size}} (d+1)$, asymptotically much smaller than $D \cdot
  N_{\mathrm{in}} f_{\mathrm{size}} N_{\mathrm{out}} f_{\mathrm{size}}$ for a
  conventional ConvNet of depth $D$. Larger kernels are tiled from a
  *base-size* (16-channel, 3×3) brick.
- **Dynamic hypernetwork — HyperRNN (Section 3.2).** A small recurrent
  hypernetwork ("HyperRNN cell") is run alongside the main RNN. At every
  timestep it consumes the concatenation $\hat x_t = [h_{t-1};\,x_t]$, updates
  its own hidden state $\hat h_t$, and emits three small embedding vectors
  $z_h, z_x, z_b \in \mathbb{R}^{N_z}$. These embeddings are then expanded into
  *time-varying* main-RNN parameters. The naïve form
  $W_h(z_h) = \langle W_{hz}, z_h \rangle$ with $W_{hz} \in
  \mathbb{R}^{N_h \times N_h \times N_z}$ is impractically memory-hungry
  (Equation 5 in the paper); the *memory-efficient* form replaces matrix
  generation by *row-wise scaling* (Equation 7), turning $z$ into an
  element-wise scaling vector $d(z) = W_{hz} z$ acting row-by-row on a single
  fixed weight matrix.
- **HyperLSTM (Appendix A.2.2).** The same row-scaling idea is applied to the
  four LSTM gates $i, g, f, o$, with one set of $z^y_h, z^y_x, z^y_b$
  embeddings per gate $y \in \{i,g,f,o\}$. The HyperLSTM cell is a *Layer-Norm
  LSTM* whose own hidden state is linearly projected to those embeddings each
  timestep.
- **Results (Sections 4.1–4.6).**
  - **MNIST static.** A 4-parameter embedding driving a 4,240-parameter
    hypernet generates a 12,544-parameter conv kernel; 99.24% accuracy versus
    99.28% for the unhypered baseline.
  - **CIFAR-10 wide-resnet.** HyperResNet-40-2 reaches 7.23% test error with
    0.148M parameters, versus 5.66% with 2.236M for the conventional
    WRN-40-2 — a 15× parameter reduction at the cost of ~1.5% accuracy.
  - **Penn Treebank character-level LM.** HyperLSTM beats vanilla LSTM and
    matches Layer-Norm LSTM; combining the two (LayerNorm-HyperLSTM) gives
    1.250 bits-per-character, then 1.219 with a 2-layer stack — near
    state-of-the-art at the time.
  - **enwik8 (Hutter Prize).** LayerNorm-HyperLSTM-2048 reaches 1.340 bpc, near
    SOTA; converges faster per training step than LSTM/LayerNorm-LSTM.
  - **IAM handwriting.** HyperLSTM (no LayerNorm) is best on log-loss
    ($-1162$), and qualitatively produces less noisy strokes.
  - **WMT En→Fr (GNMT).** Replacing LSTM cells with HyperLSTM in Google's
    production GNMT yields BLEU 40.03 vs 38.95 baseline, demonstrating
    industrial-scale viability.
- **Interpretation (Sections 4.4, 4.5).** Visualising the per-timestep weight
  changes shows *regime-like* dynamics: $W_h$ stays nearly static during
  the body of a word and undergoes large jumps at word boundaries or
  brackets — i.e. the HyperRNN learns an *implicit segmenter*. Histograms of
  $\phi(c_t)$ show that LayerNorm and HyperLSTM produce *different* hidden
  distributions even when test losses are similar: LayerNorm avoids
  saturation; HyperLSTM tends to drive cells into saturation regions but
  rescues them via the next-step weight change. The two methods can compose
  for additional gains.

### Phase 1 — Foundational synthesis

A standard convolutional network has one set of weights per layer; a standard
recurrent network has one set of weights for all timesteps. Both are extreme
choices. *HyperNetworks* asks: what if a small "weight-generator network"
produced the weights instead, conditioning on a tiny handle that says *which*
layer or *which* timestep we are at? The handle, called an *embedding*, is just
a few-dimensional vector that is itself learned during training. The
weight-generator — the hypernetwork — is a couple of linear (or shallow MLP)
layers that maps the handle into a full weight tensor.

For convolutional networks this becomes a kind of compression. Each layer of a
CIFAR-10 wide-residual-net has a kernel with millions of weights. Replace it
with a 64-dimensional embedding $z^j$ and a hypernetwork that produces a
$3\times 3 \times 16 \times 16$ "base brick"; tile copies of the brick to make
the larger kernels. The whole hyper-version of WRN-40-2 has 0.148M
parameters instead of 2.2M, with about 1.5% extra test error. The interesting
question is *not* compression for its own sake — it is that this 64-dim
embedding becomes a *layer descriptor* that the network can interpolate or
manipulate.

For recurrent networks the same idea has a different consequence: it lets a
single RNN's effective weights *change with time*. The HyperRNN cell is a
small companion RNN that, every timestep, looks at the input and the main
RNN's hidden state and produces three short vectors $z_h, z_x, z_b$. Those
vectors get expanded into row-wise scaling factors that multiplicatively
modulate the main RNN's $W_h, W_x, b$. Effectively the main RNN's parameters
are now a *function of time and input*, while still being driven by a tiny
amount of new computation each step.

Why does this matter? Because it is a sharp empirical demonstration that the
hard-weight-sharing assumption of recurrent networks is unnecessary, and a
soft, input-driven alternative wins on language modelling, handwriting and
neural machine translation. Visualising the per-timestep weight change is
striking: weights are essentially constant during the body of a word and
change abruptly at word boundaries — the network learned a *segmenter for
free* by being allowed to retune itself.

For the present project, the most important takeaway is that the paper
distinguishes two *flavours* of hypernet:
1. **Static**: the embedding $z$ is a *learned constant per slot* (per layer,
   per task), so the generated weights are fixed at inference. Used for
   parameter compression and architectural study. Essentially a way to *share
   knowledge across slots*.
2. **Dynamic**: the embedding is a *function of the input* via a recurrent or
   feedforward hypernetwork. The generated weights vary at every step. This is
   the conditioning / modulation route — the version that competing FiLM and
   neuromodulation papers will need to be compared against.

Both flavours are end-to-end differentiable. That single point — that a
small generator network can produce another network's weights and the whole
thing trains by backprop — is what every later paper in this review is
building on.

### Phase 2 — Graduate-level deep dive

#### Static hypernetwork — kernel factorisation (Section 3.1, Eqs. 1–3)

Let the main convolutional network have depth $D$, with kernel
$K^j \in \mathbb{R}^{N_{\mathrm{in}} f_{\mathrm{size}} \times N_{\mathrm{out}}
f_{\mathrm{size}}}$ at layer $j$. Each slice $K^j_i \in
\mathbb{R}^{f_{\mathrm{size}} \times N_{\mathrm{out}} f_{\mathrm{size}}}$
($i = 1,\ldots,N_{\mathrm{in}}$) is the chunk of the kernel responsible for
input channel $i$. The hypernetwork is

$$
a^j_i \;=\; W_i\, z^j + B_i,
\qquad
K^j_i \;=\; \langle W_{\mathrm{out}},\, a^j_i \rangle + B_{\mathrm{out}},
\qquad
K^j \;=\; (K^j_1, K^j_2, \ldots, K^j_{N_{\mathrm{in}}}),
$$

with $W_i \in \mathbb{R}^{d \times N_z}$, $B_i \in \mathbb{R}^d$, the shared
tensor $W_{\mathrm{out}} \in \mathbb{R}^{f_{\mathrm{size}} \times
N_{\mathrm{out}} f_{\mathrm{size}} \times d}$, and the shared bias matrix
$B_{\mathrm{out}} \in \mathbb{R}^{f_{\mathrm{size}} \times N_{\mathrm{out}}
f_{\mathrm{size}}}$. The tensor contraction $\langle W_{\mathrm{out}},
a^j_i \rangle$ is the standard tensor-dot — for $W \in \mathbb{R}^{m \times n
\times d}$ and $a \in \mathbb{R}^d$ the result is $\langle W, a\rangle \in
\mathbb{R}^{m \times n}$ via $\sum_k W_{:,:,k} a_k$.

**Why two layers?** A one-layer hypernetwork would need
$N_z \cdot N_{\mathrm{in}} \cdot f_{\mathrm{size}} \cdot N_{\mathrm{out}}
\cdot f_{\mathrm{size}}$ parameters to map directly from $z$ to $K^j$. The
two-layer factorisation has total parameter count

$$
N_{\mathrm{params}} \;=\;
\underbrace{N_z D}_{\text{embeddings}}
\;+\;
\underbrace{d(N_z+1)\,N_{\mathrm{in}}}_{W_i, B_i}
\;+\;
\underbrace{f_{\mathrm{size}}\, N_{\mathrm{out}} f_{\mathrm{size}}\,(d+1)}_{W_{\mathrm{out}}, B_{\mathrm{out}}},
$$

which the paper recommends with $d = N_z$. The third term — the shared output
tensor — is what makes the two-layer form smaller: it is a *common feature
basis* over all $N_{\mathrm{in}}$ slices of all $D$ layers, with each slice
selecting a $d$-dimensional point in that basis via $a^j_i$. This is
isomorphic to a hierarchically semiseparable (HSS) matrix
factorisation in the sense of Xia et al. (2010): the kernel tensor is
expressed as a low-rank "global atlas" $W_{\mathrm{out}}$ plus per-slot
"local addresses" $a^j_i$.

For multi-scale ResNet kernels (e.g. $32\times 32 \times 3 \times 3$ layers in
WRN-40-2), the paper tiles eight base bricks of size $16\times 16\times 3
\times 3$ and uses *eight* embedding vectors $z^{j,k}$ ($k=1,\ldots,8$):

$$
K_{32 \times 64}
=
\begin{pmatrix}
K_1 & K_2 & K_3 & K_4 \\
K_5 & K_6 & K_7 & K_8
\end{pmatrix},
\qquad K_k = g(z^{j,k}).
$$

#### Dynamic hypernetwork — HyperRNN (Eqs. 4–8)

The standard Basic-RNN evolution is

$$
h_t \;=\; \phi\!\left(W_h h_{t-1} + W_x x_t + b\right),
$$

with $W_h \in \mathbb{R}^{N_h \times N_h}$, $W_x \in \mathbb{R}^{N_h \times
N_x}$, $b \in \mathbb{R}^{N_h}$ *fixed* across timesteps. HyperRNN allows
$W_h, W_x, b$ to depend on a small embedding triple $(z_h, z_x, z_b) \in
\mathbb{R}^{3 N_z}$ that is itself produced by a HyperRNN cell of hidden size
$N_{\hat h} \ll N_h$:

$$
\hat x_t = \begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix},
\qquad
\hat h_t = \phi\!\left( W_{\hat h} \hat h_{t-1} + W_{\hat x}\hat x_t + \hat b \right),
$$

$$
z_h = W_{\hat h h} \hat h_{t-1} + b_{\hat h h},
\quad
z_x = W_{\hat h x} \hat h_{t-1} + b_{\hat h x},
\quad
z_b = W_{\hat h b} \hat h_{t-1}.
$$

The *naïve* expansion $W_h(z_h) = \langle W_{hz}, z_h\rangle$ with $W_{hz} \in
\mathbb{R}^{N_h \times N_h \times N_z}$ requires $\mathcal{O}(N_z N_h^2)$
storage — $N_z$ times a Basic-RNN. To make this practical, the paper replaces
matrix generation with *row-wise multiplicative modulation*: a single
master matrix $W \in \mathbb{R}^{N_h \times N_h}$ is reweighted row-by-row by
a *scaling vector* $d(z) \in \mathbb{R}^{N_h}$:

$$
W(z) \;=\; W \odot d(z) \;=\;
\begin{pmatrix}
d_0(z)\, W_0 \\
d_1(z)\, W_1 \\
\vdots \\
d_{N_h-1}(z)\, W_{N_h - 1}
\end{pmatrix},
\qquad
d(z) = W_{dz} z.
$$

Substituting back gives the memory-efficient HyperRNN cell:

$$
h_t \;=\; \phi\!\left( d_h(z_h) \odot W_h\, h_{t-1}
  \;+\; d_x(z_x) \odot W_x\, x_t
  \;+\; b(z_b) \right),
$$

$$
d_h(z_h) = W_{hz} z_h, \quad
d_x(z_x) = W_{xz} z_x, \quad
b(z_b)   = W_{bz} z_b + b_0.
$$

The memory cost is now $\mathcal{O}(N_z N_h)$ — only a constant factor over a
Basic-RNN — and the row-wise multiplication is implementable as an
element-wise broadcast.

**Connection to FiLM and Layer-Norm.** Equation 8 is structurally identical
to a feature-wise affine modulation: the scaling vector $d_h(z_h)$ acts as a
*gain* on each row of $W_h h_{t-1}$ (i.e. on each pre-activation neuron), and
$b(z_b)$ is a *shift*. This is exactly the FiLM operation
$\text{FiLM}(F \mid \gamma, \beta) = \gamma \odot F + \beta$ of Perez et al.
(2018), with the crucial difference that here $(\gamma, \beta) = (d, b)$ are
generated from a *recurrent* embedding produced by a separate small RNN, not
from a per-task one-shot conditioning input. Recurrent BatchNorm (Cooijmans
et al. 2016) and LayerNorm (Ba et al. 2016) are essentially the same
multiplicative-plus-additive operation but with *fixed* learned scales/shifts
applied after a moments-based normalisation step. The HyperRNN cell can be
viewed as a Layer-Norm whose scale and shift are *predicted on the fly from
context*. Empirically, HyperLSTM and LayerNorm-LSTM produce different hidden
distributions (Figure 5: LayerNorm avoids saturation; HyperLSTM tends to
saturate), but combining them gives additional gains, suggesting they
exploit complementary degrees of freedom.

#### HyperLSTM (Eqs. 9–12)

The standard LSTM with gates $\{i, g, f, o\}$ is

$$
y_t = W^y_h h_{t-1} + W^y_x x_t + b^y, \qquad y \in \{i,g,f,o\},
$$

$$
c_t = \sigma(f_t)\odot c_{t-1} + \sigma(i_t) \odot \phi(g_t),
\qquad
h_t = \sigma(o_t) \odot \phi(c_t).
$$

In HyperLSTM each gate $y \in \{i,g,f,o\}$ has its *own* triple of embeddings
$(z^y_h, z^y_x, z^y_b)$ produced by a single shared HyperLSTM cell whose
hidden state is $\hat h_{t-1}$. Each gate's pre-activation is then computed
with row-wise scaling and a Layer-Norm wrapper:

$$
y_t = \mathrm{LN}\!\left(
   d^y_h(z^y_h) \odot W^y_h h_{t-1}
   + d^y_x(z^y_x) \odot W^y_x x_t
   + b^y(z^y_b)
\right),
$$

with $d^y_h, d^y_x, b^y$ defined as in Equation 8. The memory cost over a
plain Layer-Norm LSTM is $\mathcal{O}(N_{\hat h}^2 + 4 N_z N_h)$ — an
$N_{\hat h} = 128$ HyperLSTM cell adds $\sim$0.7M parameters to a 1000-unit
main LSTM. Initialisation matters: the weight-scaling matrices $W_{hz},
W_{xz}$ are initialised to $0.1/N_z$ rather than $1/N_z$, following Recurrent
BatchNorm's heuristic that initialising the gain to $0.1$ stabilises the
optimisation (Section A.2.3).

#### What the paper does *not* do

- It treats the embeddings $z^j$ as *learnable parameters*, not as a function
  of an external task or context label. Continual-learning and meta-RL
  applications (Papers 5–7) will close this gap by making $z$ a deterministic
  function of a task ID or task descriptor.
- It does not place a prior on $z$ or $\theta_I = f(z)$ and does not consider
  posterior uncertainty over generated weights. Bayesian extensions
  (Paper 4, and the Bayesian Hypernetworks of Krueger et al. 2017 cited by
  Paper 2) are deferred.
- The dynamic hypernetwork is *internally recurrent* but has no explicit notion
  of *task* — the time-varying weights track the input sequence, not an
  externally selected task or precision signal. The interoceptive-RL question
  (whether to drive $z$ from an internal precision signal vs. from external
  task ID) is therefore *outside* the paper's scope but is something every
  downstream paper has to choose.

### Relevance to interoceptive-pain RL

- **Static-hypernet ↔ FiLM equivalence.** The memory-efficient HyperRNN
  (Equation 8) reduces *exactly* to a FiLM-style affine modulation
  $\gamma \odot W h + \beta$, with $\gamma, \beta$ generated from a recurrent
  embedding. For the project's FiLM-vs-hypernet comparison this is the right
  *minimal* hypernet baseline: it shows that, at the row-scaling level of
  expressiveness, hypernet and FiLM are the same operation, and the empirical
  difference must come from how the embedding is computed (recurrent
  context-tracker for HyperRNN vs. one-shot conditioning vector for FiLM).
- **Time-varying weights as a precision-modulation analogue.** Figure 4 of the
  paper shows that HyperLSTM weights make *regime-shift* changes at semantic
  boundaries (between words, between brackets). Reading that picture through
  the project's predictive-coding framing: the HyperRNN spontaneously builds
  an internal *event-boundary detector* that retunes the recurrent dynamics.
  An interoceptive precision signal that retuned the policy network's weights
  on detection of pain or threat would have the same shape — though the paper
  itself does not make this claim and does not connect to neuromodulation.
- **Compression as biological plausibility.** The static-hypernet result that a
  64-dim embedding plus a shared output tensor can specify a 36-layer ResNet
  is an existence proof that *cortex-sized* parameter counts are not required
  to express a flexible function class — relevant to the
  `professor-neuromodulation` agent's guard against biologically implausible
  parameter budgets, but the paper does not make a neuroscience argument and
  this connection is exploratory.

---

## <a id="paper-2"></a>Paper 2 — On the Modularity of Hypernetworks (Galanti & Wolf, 2020)

**Citation:** Galanti, T., & Wolf, L. (2020). *On the Modularity of
Hypernetworks.* NeurIPS 2020.

**PDF:** `docs/project/references/Hypernetwork/sources/Galanti and Wolf 2020 - On the modularity of hypernetworks.pdf` (11 pages)

### Backbone

- **Problem.** Hypernetworks empirically beat embedding-based conditioning on
  numerous benchmarks, often using a *small* primary network $g$ driven by a
  *large* hypernetwork $f$. Why? The paper asks whether hypernetworks have a
  formal expressivity advantage and quantifies it as a property they call
  *modularity*: the ability of $g$ to be as small as if it had been fitted
  separately to each task $I$.
- **Setup (Section 2).** A target function $y : \mathcal X \times
  \mathcal I \to \mathbb R$ is to be approximated. Two approaches are
  compared:
  - **Embedding method**: $h(x,I) = q(x, e(I))$, with embedding $e:\mathcal I
    \to \mathbb R^k$ and primary $q$ taking the *concatenation*
    $[x; e(I)]$ as input. Parametric class $\mathcal E_{\mathfrak e,
    \mathfrak q}$.
  - **Hypernetwork**: $h(x,I) = g(x;\, f(I; \theta_f))$, where $f$ generates
    *all* weights of $g$. Parametric class $\mathcal H_{\mathfrak f,
    \mathfrak g}$.
- **Smoothness class.** The target $y$ lives in the Sobolev ball
  $\mathcal W_{r,n}$: $r$-times continuously differentiable
  $[-1,1]^n \to \mathbb R$ with Sobolev norm $\le 1$. The two-input case has
  $n = m_1 + m_2$ where $m_1 = \dim(\mathcal X)$, $m_2 = \dim(\mathcal I)$.
- **Theorem 1 — non-robust lower bound.** Extends the classical result of
  DeVore, Howard & Micchelli (1989). For piecewise-$C^1$ activations $\sigma$
  with $\sigma' \in BV(\mathbb R)$ (so clipped-ReLU, sigmoid, tanh, arctan all
  qualify), any neural-network class $\mathcal F$ that approximates
  $\mathcal W_{r,m}$ to error $\epsilon$ must have
  $N_{\mathcal F} = \Omega(\epsilon^{-m/r})$ trainable parameters.
  Critically the bound *does not* require the parameter-selection mapping
  $S : y \mapsto \theta$ to be continuous (the previous result of [6]
  required *robust* approximation).
- **Theorem 2 — embedding lower bound (constant $k$).** When the embedding
  dimension $k$ is constant, an embedding method achieving error $\epsilon$
  requires the *primary* $q$ to have
  $N_{\mathfrak q} = \Omega(\epsilon^{-(m_1 + m_2)})$ parameters.
- **Theorem 3 — embedding lower bound (variable $k$).** Even if $k$ is
  allowed to grow with $\epsilon$, but the first-layer norm of $q$ is
  bounded, $N_{\mathfrak q} = \Omega(\epsilon^{-\min(m, 2 m_1)})$. So an
  embedding method's primary network *cannot* match the per-task optimum.
- **Theorem 4 — modularity of hypernetworks.** For any $y \in \mathcal W_{r,m}$
  and a sufficiently large ReLU hypernet $f$, there is a class $\mathfrak g$
  with $N_{\mathfrak g} = O(\epsilon^{-m_1/r})$ such that the combined
  $h(x,I) = g(x;\, f(I;\theta_f))$ achieves error $\le \epsilon$. The primary
  network $g$ is therefore as small as the *per-instance* optimum, even
  though it must work for *every* $I$.
- **Theorem 5 — total parameter count for structured selectors.** If the
  optimal weight-selector $S(I)$ is a low-rank linear projection
  $W \cdot h(I)$ with $h$ having $w$-dimensional features (a structural
  assumption that holds for typical hypernet designs), the *overall*
  hypernetwork has $N_{\mathfrak f} = O(\epsilon^{-m_2/r} +
  \epsilon^{-m_1/r})$ parameters versus $\Omega(\epsilon^{-(m_1+m_2)/r})$ for
  generic neural-network or embedding-method approximators.
- **Empirical validation (Section 5).** Synthetic data
  $y(x,I) = \langle x, h(I)\rangle$ with $h$ a deep MLP, varying number of
  layers in the conditioning network: hypernet error keeps decreasing with
  depth/width, embedding error plateaus. Same trend on self-supervised
  rotation prediction with MNIST and CIFAR-10.

### Phase 1 — Foundational synthesis

This is a theory paper answering a simple-sounding question: when you want one
network to compute *different functions for different inputs $I$*, is it
better to (a) hand it a side-channel embedding and let it figure things out,
or (b) actually swap out its weights for each $I$? The first option is the
*embedding method* — concatenate $e(I)$ to the main input $x$ and let the
primary $q$ work it out. The second is the *hypernetwork* — let a separate
$f(I)$ compute new weights $\theta_I$ and run a fixed-architecture $g(\cdot;
\theta_I)$.

The intuition for why hypernetworks help is roughly this: in option (a), the
primary $q$ has to be expressive enough to compute *every* function in the
target family at once (it just gets a hint via $e(I)$ about which one).
In option (b), the primary $g$ only needs to be expressive enough to compute
*one* function — the function for the current $I$ — because its weights have
already been swapped out. So $g$ can be much smaller than $q$.

The paper's contribution is making this rigorous. They prove:

- For smooth target functions, *any* neural network needs at least
  $\Omega(\epsilon^{-n/r})$ parameters to fit them to accuracy $\epsilon$
  (Theorem 1; here $n$ is input dimension, $r$ is smoothness order). The
  proof tightens earlier work by removing a "robust selection" assumption.
- An embedding method's primary network $q$ is stuck at
  $\Omega(\epsilon^{-(m_1+m_2)})$ for the smoothness order $r=1$ — i.e., it
  pays the full *joint* dimension cost, no matter how big the embedding (Thms.
  2–3).
- A hypernetwork's primary network $g$ can be as small as
  $O(\epsilon^{-m_1/r})$ — the cost of fitting *one* function over $\mathcal X$
  (Thm. 4). And the total parameter count of the hypernet is the *sum*
  $O(\epsilon^{-m_2/r} + \epsilon^{-m_1/r})$ rather than the *product*
  (Thm. 5), because the structure of the selector $S(I) = W \cdot h(I)$ is
  itself low-dimensional.

The headline is the asymmetry: embedding methods can be viewed as a *special
case* of hypernetworks (where only the first layer's biases are generated),
but the converse is not true. So hypernetworks are at least as expressive as
embedding methods, and the modularity proof shows they are *strictly* more
parameter-efficient for smooth target families.

For a project comparing FiLM vs hypernet, the paper is encouraging in two
ways: (i) FiLM is essentially a "row-scaling embedding method", so the lower
bounds in Thms. 2–3 apply qualitatively — its primary network must absorb the
joint task complexity; (ii) a hypernet that generates *all* weights of a small
$g$ can beat that, so long as the selector $f$ is itself reasonably
parameterised. The caveat is that the bounds are *worst-case* over
$\mathcal W_{r,m}$ — a particular task may still be solvable by an embedding
method just fine, and SGD optimisation may not realise the theoretical
modular solution.

### Phase 2 — Graduate-level deep dive

#### Function-space framework (Section 2)

Define classes parametrised by neural-network weights:

$$
\mathfrak f := \{ f(\cdot; \theta_f) : \mathbb R^{m_2} \to \mathbb R^{N_{\mathfrak g}} \mid \theta_f \in \Theta_{\mathfrak f} \},
\quad
\mathfrak g := \{ g(\cdot; \theta_g) : \mathbb R^{m_1} \to \mathbb R \mid \theta_g \in \Theta_{\mathfrak g} \},
$$

$$
\mathfrak e := \{ e(\cdot; \theta_e) \},
\quad
\mathfrak q := \{ q(\cdot; \theta_q) \}.
$$

The hypernetwork class is

$$
\mathcal H_{\mathfrak f, \mathfrak g} \;=\;
\{\, x \mapsto g(x; f(I; \theta_f)) \mid \theta_f \in \Theta_{\mathfrak f}\,\}.
$$

Because $f(I; \theta_f) \in \Theta_{\mathfrak g}$, the hypernetwork picks a
*point* in $g$'s parameter space *as a function of $I$*. The key quantity is
the parameter count $N_{\mathfrak g}$ — the dimensionality of the codomain of
$f$ — which is also the number of weights $g$ has.

The approximation error of a class $\mathcal P$ on a target family
$\mathcal Y$ is

$$
d(\mathcal P; \mathcal Y) \;:=\; \sup_{y \in \mathcal Y}\,
\inf_{p \in \mathcal P} \, \|y - p\|_\infty.
$$

DeVore–Howard–Micchelli's $N$-width is the *robust* version, requiring the
selection $S : y \mapsto \theta$ to be continuous:

$$
\tilde d_N(\mathcal Y) := \inf_{\mathcal F, S \text{ continuous}}
\sup_{y \in \mathcal Y} \|f(\cdot; S(y)) - y\|_\infty,
\qquad
\tilde d_N(\mathcal W_{r,m}) = \Omega(N^{-m/r}).
$$

#### Theorem 1 — non-robust lower bound

Let $\sigma$ be piecewise-$C^1(\mathbb R)$ with $\sigma' \in BV(\mathbb R)$:

$$
BV(\mathbb R) := \Big\{ f \in L^1(\mathbb R) \;\Big|\;
\|f\|_{BV} := \sup_{\substack{\phi \in C^1_c(\mathbb R) \\ \|\phi\|_\infty \le 1}}
\int_{\mathbb R} f(x)\, \phi(x)\, dx < \infty \Big\}.
$$

The class of activations satisfying this includes clipped-ReLU, sigmoid, tanh
and arctan but *not* unbounded ReLU directly (though their Lemma 18 in the
appendix relaxes this).

**Statement.** Let $\mathfrak f$ be any class of neural networks with such
$\sigma$, $\mathcal Y = \mathcal W_{r,m}$, and assume any non-constant $y \in
\mathcal Y$ is not exactly representable by $\mathfrak f$. If $d(\mathfrak f;
\mathcal Y) \le \epsilon$, then $N_{\mathfrak f} = \Omega(\epsilon^{-m/r})$.

**Proof sketch.** The crux is constructing a "wide subclass"
$\mathcal Y' \subset \mathcal Y$ on which a continuous selector
$S : \mathcal Y' \to \Theta_{\mathfrak f}$ exists with bounded approximation
distortion factor $\alpha$:

$$
\forall y \in \mathcal Y':\;
\|f(\cdot; S(y)) - y\|_\infty \le
\alpha \cdot \inf_{\theta \in \Theta_{\mathfrak f}} \|f(\cdot; \theta) - y\|_\infty.
$$

By the classical $N$-width result $\tilde d_N(\mathcal Y') = \Omega(N^{-m/r})$.
Since $d(\mathfrak f; \mathcal Y) \ge d(\mathfrak f; \mathcal Y') \ge
\frac{1}{\alpha} \tilde d(\mathfrak f; \mathcal Y')$, the same lower bound
transfers to the *non-robust* setting. The continuous-selector existence is
the technical core (App. Sec. 3.2–3.3), relying on
Assumptions 1 (uniqueness of best approximator) and 2 (adding a neuron
strictly improves approximation if $y \notin \mathfrak f$).

#### Theorem 2 — Embedding lower bound (constant $k$)

Let $\sigma$ be a universal piecewise-$C^1$ activation with $\sigma' \in BV$
and $\sigma(0)=0$. Let $\mathfrak e$ be a class of zero-bias continuously
differentiable neural networks with output dimension $k = O(1)$ and Lipschitz
bound $C(e) \le \ell_1$. Let $\mathfrak q$ be a class with $C(q) \le \ell_2$.
Let $\mathcal Y = \mathcal W_{1,m}$. Assume any non-constant $y \in \mathcal Y$
is not representable as a $\sigma$-activation neural net.

**Statement.** If $d(\mathcal E_{\mathfrak e, \mathfrak q}, \mathcal Y) \le
\epsilon$, then $N_{\mathfrak q} = \Omega(\epsilon^{-(m_1 + m_2)})$.

**Why this is bad for embedding methods.** No matter how large you make the
embedding $e$ (its parameter count $N_{\mathfrak e}$), the *primary* $q$ —
the network that does the actual work at inference time on each new
$(x, I)$ — must scale with the *joint* input dimension. Doubling $m_2$
quadruples the required $q$ parameters when $m_1$ is fixed.

#### Theorem 4 — Modularity of hypernetworks

Let $\sigma$ be as in Thm. 2, and $y \in \mathcal Y = \mathcal W_{r,m}$ such
that $y_I = y(\cdot, I)$ is not exactly representable in any class of
$\sigma$-networks for any $I$. Then there exists a class $\mathfrak g$ of
$\sigma$-networks and a ReLU hypernet $f(I; \theta_f)$ such that

$$
\sup_I \|g(\cdot; f(I;\theta_f)) - y_I\|_\infty \le \epsilon
\quad\text{and}\quad
N_{\mathfrak g} = O(\epsilon^{-m_1/r}).
$$

**Proof sketch.** Three steps.
1. Treat $y$ as a parametric family $\mathcal Y_I := \{y_I\}_{I \in \mathcal
   I}$ and pick $\mathfrak g$ so that the per-instance error is bounded:
   universality of $\sigma$ gives $d(\mathfrak g; \mathcal Y_I) \le \epsilon$
   with $N_{\mathfrak g} = O(\epsilon^{-m_1/r})$.
2. *Continuous selector existence*. Construct a continuous map
   $S : \mathcal Y_I \to \Theta_{\mathfrak g}$ with
   $\sup_I \|g(\cdot; S(y_I)) - y_I\|_\infty \le 2 d(\mathfrak g; \mathcal
   Y_I) \le 2\epsilon$. The composition $\hat S(I) := S(y_I)$ is itself
   continuous because $I \mapsto y_I$ is.
3. *Replace the selector by a ReLU network*. Approximation theory for ReLU
   networks (Hanin & Sellke 2018) lets us pick $\theta_f$ so that
   $\|f(\cdot; \theta_f) - \hat S(\cdot)\|_\infty$ is arbitrarily small.
   Uniform continuity of $g$ in $\theta_g$ then gives
   $\sup_I \|g(\cdot; f(I;\theta_f)) - g(\cdot; \hat S(I))\|_\infty \le
   \epsilon$, and the triangle inequality completes the proof:

$$
\sup_I \|g(\cdot; f(I;\theta_f)) - y_I\|_\infty
\le 2\epsilon + \epsilon = 3\epsilon.
$$

The factor of 3 is absorbed into the asymptotic $\epsilon$.

#### Theorem 5 — Total parameter count

Suppose the optimal selector $S$ lives in a *structured* class
$\mathcal P_{r,w,c}$: $S(I) = W \cdot h(I)$ where each output coordinate
$h_i$ of $h: \mathcal I \to \mathbb R^w$ lies in $\mathcal W_{r, m_2}$ and
$\|W\|_1 \le c$. Then

$$
N_{\mathfrak f} = O\big( w^{1 + m_2/r} \cdot \epsilon^{-m_2/r} + w \cdot N_{\mathfrak g} \big)
= O\big( \epsilon^{-m_2/r} + \epsilon^{-m_1/r} \big),
$$

a *sum* rather than a product. The intuition: $w$ shared "feature heads" each
need only fit a $\mathcal W_{r, m_2}$ function, paying $\epsilon^{-m_2/r}$
each. The $w$ heads are then linearly recombined into the
$N_{\mathfrak g}$-dim weight space, paying $w \cdot N_{\mathfrak g}$. As long
as $w$ and $N_{\mathfrak g}$ are not too large, the total beats the
$\Omega(\epsilon^{-(m_1+m_2)/r})$ lower bound for arbitrary neural networks
(and a fortiori for embedding methods).

#### Where the paper draws the line

- **Worst-case theory**. The bounds are over *all* $y \in \mathcal W_{r,m}$.
  Particular target functions (e.g. linear in $I$) may be solved by
  embedding methods at much lower cost.
- **Realisability via SGD**. The proofs are existential — they exhibit
  parameter settings, not optimisation procedures. The authors note (Sec. 6)
  that whether SGD finds the modular solution is an empirical question; their
  experiments validate that it usually does.
- **Smooth target functions only**. Discontinuous targets (classification with
  hard boundaries) and high-frequency targets fall outside
  $\mathcal W_{r,m}$. The bounds give little information for those.
- **No noise / no Bayesian treatment**. The setting is deterministic
  approximation of fixed $y$. Bayesian hypernetworks (Paper 4) operate in a
  different regime where the *posterior* over $\theta_I$ is what matters.

### Relevance to interoceptive-pain RL

- **FiLM is an embedding method in this taxonomy.** FiLM's $\gamma, \beta$
  conditioning vectors enter the primary network as scaling/shift parameters
  on a *fixed* main architecture — formally an embedding method with a
  structured first layer. By Theorems 2–3 the FiLM primary must scale with
  the *joint* dimension of state+context, while a hypernet primary can be much
  smaller. *However*, the bounds are worst-case in the smoothness class — a
  task-specific empirical comparison is still needed and is what
  Papers 5–7 carry out.
- **Reduces "FiLM vs hypernet" to "what is generated, not how it is
  injected".** The paper's symmetric framing makes clear that the meaningful
  axis is whether *all* weights of the primary are generated (hypernet) vs.
  only a small subset (embedding/FiLM). For the project's RPPO and Dreamer
  baselines this is actionable: a "minimal" hypernet baseline that generates
  only the policy MLP's weights would already realise modularity, whereas
  FiLM-on-the-policy realises only the embedding bound.
- **No explicit pain or precision content.** The paper does not connect to
  predictive coding, Bayesian inference, or neuromodulation. Any bridge to
  the interoceptive-pain framing is exploratory.

---

## <a id="paper-3"></a>Paper 3 — Dynamic Predictive Coding with Hypernetworks (Jiang, Gklezakos & Rao, 2021)

**Citation:** Jiang, L. P., Gklezakos, D. C., & Rao, R. P. N. (2021).
*Dynamic Predictive Coding with Hypernetworks.* bioRxiv
2021.02.22.432194.

**PDF:** `docs/project/references/Hypernetwork/sources/Jiang et al. 2021 - Dynamic Predictive Coding with Hypernetworks.pdf` (19 pages)

### Backbone

- **Problem.** The classical Rao–Ballard predictive-coding model captures
  *spatial* receptive fields but treats temporal transitions either as static
  (Kalman-filter style) or absent. The cortex shows a *hierarchy of
  timescales* with higher areas responding more slowly and stably than lower
  areas (Hasson et al. 2008, Murray et al. 2014). The paper asks whether a
  hierarchical predictive-coding network with *time-varying* transition
  matrices generated by a hypernetwork can reproduce both V1-like simple-cell
  spatiotemporal receptive fields and the cortical timescale hierarchy.
- **Single-level model (Section 2).** The generative model factorises as

$$
p(\mathbf I_{1:T},\, \mathbf r_{1:T})
= p(\mathbf r_1) \prod_{t=1}^T p(\mathbf I_t \mid \mathbf r_t)
\prod_{t=2}^T p(\mathbf r_t \mid \mathbf r_{1:t-1}).
$$

with a sparse-coding emission $\mathbf I_t = U \mathbf r_t + \mathbf n$
(Olshausen–Field) and a *time-varying* linear transition
$\mathbf r_{t+1} = \mathrm{ReLU}(V_{t+1} \mathbf r_t) + \mathbf m$. The
transition matrix is generated by a recurrent hypernetwork $H$ as a *convex
combination* of $K$ learned basis matrices $\{V^k\}_{k=1}^K$:

$$
(\mathbf w_{t+1}, \mathbf h_{t+1}) = H(\mathbf r_t, \mathbf h_t),
\qquad
V_{t+1} = \sum_{k=1}^K w^k_{t+1} V^k.
$$

- **Hierarchical (two-level) model (Section 3).** A second-level state
  $\mathbf r^{(2)}$ replaces the recurrent hypernet with a *feedforward*
  hypernetwork $H : \mathbf r^{(2)} \mapsto \mathbf w$. The transition
  matrix at the first level is then $V = \sum_k w^k V^k$, with $\mathbf w =
  H(\mathbf r^{(2)})$. The entire first-level sequence shares one $\mathbf
  r^{(2)}$, which is itself updated by gradient descent on the prediction
  error. This recovers a *continuous-state hierarchical hidden Markov
  model*: a single higher-level state generates an entire lower-level
  sub-HMM (cf. Murphy & Paskin 2002).
- **Inference & learning (Eq. 7, Eq. 14, Algorithm 1).** Both inference of
  $\mathbf r_{1:T}, \mathbf r^{(2)}$ and learning of $U, V, H$ minimise a
  single prediction-error loss

$$
\mathcal L = \sum_{t=1}^T \big( \|\mathbf I_t - U \mathbf r_t\|_2^2 + \lambda
\|\mathbf r_t\|_1 \big) + \sum_{t=1}^{T-1} \|\mathbf r_{t+1} - \mathrm{ReLU}(V_{t+1} \mathbf r_t)\|_2^2,
$$

with the hierarchical version adding $\lambda_2 \|\mathbf r^{(2)}\|_2^2$. At
each timestep, $\mathbf r_t$ is updated by gradient descent on this loss
(Bayesian-filtering style); after a full sequence pass, the parameters
$U, V^k, H$ are updated by gradient descent.
- **Results (Section 4).** Trained on 5,000 natural video segments (forest
  trail, 10×10 patches, 20 frames each), $K=5$ basis matrices.
  - *Long-horizon prediction.* Dynamic-transition model beats static-Kalman
    baseline on both training and test MSE; on input-suppressed rollouts
    (input turned off after step 10) the dynamic model continues to
    *correctly* predict input motion direction while the static model
    predicts blurry stagnation.
  - *V1-like receptive fields.* Reverse-correlation analysis on the
    first-level neurons recovers both *separable* (oriented spatial filter,
    sign-flipping over time) and *inseparable* (direction-selective) STRFs,
    qualitatively matching cat V1 simple cells (DeAngelis et al. 1995).
  - *Timescale hierarchy.* On both natural-video and Moving-MNIST sequences,
    $\mathbf r^{(2)}$ remains *stable* over the full sequence except at
    motion-regime changes (e.g. when a digit bounces off a boundary), where
    it adapts to generate a new transition matrix. No smoothness penalty or
    explicit time constant was imposed — the slow-fast hierarchy emerges from
    minimising prediction error.
  - *Generative sampling (Figure 8).* Sampling $\mathbf r^{(2)} \sim
    \mathcal N(0,1)$ and applying the hypernet-generated $V$ to a fixed
    initial $\mathbf r^{(1)}_1$ yields a variety of plausible motion
    trajectories from the same starting frame, demonstrating that
    $\mathbf r^{(2)}$ acts as a low-dimensional embedding of motion type.
- **Discussion / interpretation (Section 5).** The hypernetwork is offered
  as a candidate *neural mechanism* for top-down modulation of synaptic
  efficacy: rather than rewriting the entire weight matrix (biologically
  implausible), the network outputs a low-dimensional *mixture vector* over
  a fixed basis of transition matrices, consistent with proposals about
  context-dependent population dynamics (Mante et al. 2013; Stroud et al.
  2018) and motor control. The continuous-state HHMM picture suggests that
  cortical timescale hierarchies could be emergent from a single
  prediction-error principle.

### Phase 1 — Foundational synthesis

Predictive coding is an old idea: the brain models the world by predicting
its sensory input and routing only the *errors* upward. Rao & Ballard's
1999 model showed this could reproduce V1's spatial receptive fields. But
brains do not just predict the next pixel — they predict the next *moment*.
And the dynamics of the world are not the same in every situation: a ball
rolling and a ball bouncing follow different rules. Vanilla Kalman filtering
assumes one fixed transition matrix; that is too rigid.

Jiang, Gklezakos & Rao's idea is to make the transition matrix *itself* a
function of context, generated on-the-fly by a hypernetwork. To keep the
mechanism biologically plausible, the hypernetwork does not produce raw
synaptic weights; it produces a small *mixture vector* over a fixed bank of
basis transition matrices. The actual transition matrix is then a weighted
sum of basis matrices. This is a clean instance of relaxed weight sharing in
the Ha-et-al sense (Paper 1), specialised to a generative setting.

Two architectures are compared:

1. **Recurrent single-level model.** A small RNN consumes the current state
   and emits next-step mixture weights. The transition matrix changes every
   step. The model can predict natural-video sequences, learns V1-like
   space-time receptive fields at level 1, and continues predicting
   sensible motion when the input is removed.
2. **Two-level hierarchical model.** A higher-level state vector
   $\mathbf r^{(2)}$ controls an *entire* lower-level sequence by feeding the
   hypernetwork. When the input dynamics are smooth, $\mathbf r^{(2)}$ stays
   put — small prediction errors do not drive its update. When the dynamics
   change abruptly, the resulting prediction-error spike forces a new
   $\mathbf r^{(2)}$, which generates new transition mixture weights and
   restarts a new lower-level "regime". The slow-fast cortical timescale
   hierarchy that has been measured empirically falls out automatically:
   higher-level activities are slower because they only update on regime
   changes.

The paper's key contribution to this review is conceptual. It is the first
to put a hypernetwork *inside* a generative model rather than inside a
discriminative classifier, and it does so in a Bayesian / predictive-coding
frame. For the project's interoceptive-pain agent that *also* aims to use
predictive coding plus precision modulation, this is a near-direct template:
the higher-level state plays the role of a precision (or task-context)
controller, and its job is to retune the lower-level dynamics by emitting a
small mixture vector. Whether to read $\mathbf r^{(2)}$ as a "neuromodulator"
or as a "task descriptor" or as an "interoceptive precision signal" is left
open — the paper itself only makes the cortical-timescale and V1-RF
arguments.

### Phase 2 — Graduate-level deep dive

#### Spatial generative model — sparse coding

Identical to Olshausen–Field 1996 with overcomplete basis $U \in
\mathbb R^{D \times d}$, $d > D$:

$$
\mathbf I_t = U \mathbf r_t + \mathbf n,
\qquad \mathbf n \sim \mathcal N(0, \sigma_I^2 I_D),
\qquad p(\mathbf r_t) \propto \exp(-\lambda \|\mathbf r_t\|_1).
$$

The MAP inference at the first frame solves the standard LASSO,

$$
\mathbf r_1 = \arg\min_{\mathbf r} \|\mathbf I_1 - U \mathbf r\|_2^2 + \lambda \|\mathbf r\|_1.
$$

#### Recurrent dynamic transition (Eqs. 4–6)

The transition operator at time $t+1$ is a *convex combination* of $K$ basis
matrices $\{V^k\}_{k=1}^K$:

$$
V_{t+1} = \sum_{k=1}^K w^k_{t+1} V^k,
\qquad
\mathbf w_{t+1} \in \Delta^{K-1}\;(\text{or}\; \in \mathbb R^K\;\text{if not constrained}),
$$

with $\mathbf w_{t+1}$ produced by a hypernetwork $H$ that is itself a
GRU/RNN with hidden state $\mathbf h_t$ and feedforward output head:

$$
(\mathbf w_{t+1}, \mathbf h_{t+1}) = H(\mathbf r_t, \mathbf h_t).
$$

The next latent state is then sampled (or MAP-inferred) from

$$
\mathbf r_{t+1} = \mathrm{ReLU}(V_{t+1} \mathbf r_t) + \mathbf m,
\qquad \mathbf m \sim \mathcal N(0, \sigma_r^2 I_d).
$$

The ReLU non-linearity is the only departure from a strict linear-Gaussian
state-space model and is needed to enforce non-negativity of cortical
firing rates. Crucially $V_{t+1}$ depends only on $(\mathbf r_t,
\mathbf h_t)$, so the parameter count of the entire dynamics module is
$K \cdot d \cdot d$ for the basis plus the (small) RNN parameters of $H$
— independent of sequence length $T$.

#### Hierarchical (top-down) parameterisation (Eqs. 11–13)

The two-level generative model factorises as

$$
p(\mathbf I_{1:T}, \mathbf r^{(1)}_{1:T}, \mathbf r^{(2)})
= p(\mathbf r^{(2)}) \, p(\mathbf r^{(1)}_1 \mid \mathbf r^{(2)})
\prod_{t=1}^T p(\mathbf I_t \mid \mathbf r^{(1)}_t)
\prod_{t=2}^T p(\mathbf r^{(1)}_t \mid \mathbf r^{(1)}_{t-1}, \mathbf r^{(2)}).
$$

The conditional independence structure is the key innovation: given
$\mathbf r^{(2)}$, the next first-level state depends only on the previous
first-level state, *not* on the entire history $\mathbf r^{(1)}_{1:t-1}$ —
because the higher-level state has absorbed that history. The hypernet maps
the higher level once per sequence:

$$
\mathbf w = H(\mathbf r^{(2)}),
\qquad
V = \sum_{k=1}^K w^k V^k,
\qquad
\mathbf r^{(1)}_{t+1} = \mathrm{ReLU}(V \mathbf r^{(1)}_t) + \mathbf m.
$$

This is the *continuous-state HHMM*: a single higher-level latent generates
an entire lower-level "episode" with its own transition matrix.

#### Loss and inference (Eqs. 7, 14, 15)

The single-level loss is

$$
\mathcal L = \sum_{t=1}^T \big( \|\mathbf I_t - U \mathbf r_t\|_2^2
  + \lambda \|\mathbf r_t\|_1 \big)
+ \sum_{t=1}^{T-1} \|\mathbf r_{t+1} - \mathrm{ReLU}(V_{t+1} \mathbf r_t)\|_2^2.
$$

The hierarchical version adds a Gaussian prior penalty on $\mathbf r^{(2)}$:

$$
\mathcal L^{(2)} = \mathcal L + \lambda_2 \|\mathbf r^{(2)}\|_2^2,
\qquad V = \sum_k (H(\mathbf r^{(2)}))^k V^k.
$$

**MAP filtering for $\mathbf r_t$.** At each timestep the filtering MAP
update minimises the *causal* loss:

$$
\mathbf r_t \leftarrow
\arg\min_{\mathbf r}
\|\mathbf I_t - U \mathbf r\|_2^2
+ \|\mathbf r - \mathrm{ReLU}(V_t \mathbf r_{t-1})\|_2^2
+ \lambda \|\mathbf r\|_1.
$$

**Inference of $\mathbf r^{(2)}$.** Since $\mathbf r^{(2)}$ shapes the full
lower-level sequence, it is updated by gradient descent on the
prediction-error term it most affects:

$$
\mathbf r^{(2)} \leftarrow \mathbf r^{(2)} - \eta \nabla_{\mathbf r^{(2)}}
\Big[ \|\mathbf r^{(1)}_{t+1} - \mathrm{ReLU}(V \mathbf r^{(1)}_t)\|_2^2
  + \lambda_2 \|\mathbf r^{(2)}\|_2^2 \Big],
\qquad V = \sum_k (H(\mathbf r^{(2)}))^k V^k.
$$

When the lower-level dynamics is well-predicted, the gradient is small and
$\mathbf r^{(2)}$ stays nearly constant — *that* is the mechanism that
produces the slow-fast timescale separation. The mathematical statement is
that $\mathbf r^{(2)}$ is updated proportionally to the *prediction error
gradient*, so a sequence that is well-predicted produces a near-zero update,
exactly mirroring the way evidence in Bayesian filtering only moves the
posterior when the likelihood is informative.

**Parameter learning.** After a full sequence pass with $\{\mathbf r_t\}$
inferred, the parameters update as

$$
U \leftarrow U - \eta_U \nabla_U \mathcal L,
\quad
V^k \leftarrow V^k - \eta_V \nabla_{V^k} \mathcal L,\;\forall k,
\quad
\theta_H \leftarrow \theta_H - \eta_H \nabla_{\theta_H} \mathcal L.
$$

(Algorithm 1.)

#### Why mixture-of-bases-matrices, not raw weight generation?

The paper explicitly notes (Sec. 1) that producing the *entire* set of
high-dimensional synaptic weights would not be neurally plausible. Instead,
the hypernetwork outputs a *low-dimensional* mixture vector $\mathbf w \in
\mathbb R^K$ ($K = 5$ in the experiments), and the basis $\{V^k\}$ is fixed
once trained. This is structurally identical to a *gated combination* of
fixed dynamical primitives — the kind of "motor primitive" or "neural
ensemble" picture used in motor-control modelling (Stroud et al. 2018) and
context-dependent computation in PFC (Mante et al. 2013).

Compared to the row-scaling form of HyperRNN in Paper 1, the mixture-vector
form is *less expressive* (a $K$-dim simplex vs. a full $N_h$-dim row-scaling
vector) but *more interpretable*: each $V^k$ can be inspected as an
elementary motion primitive. The mixture weights $\mathbf w$ play the role
of a *context-dependent gating* variable, which is the clearest bridge to
the project's neuromodulation / FiLM line of inquiry.

#### What the paper does *not* do

- **No Bayesian uncertainty.** $\mathbf r^{(2)}$ is treated by MAP, not by
  approximating a posterior; its update rule is plain gradient descent.
- **No precision parameter on the dynamics.** The Gaussian noise variance
  $\sigma_r^2$ is fixed; there is no per-step or per-context precision
  modulation, even though that would naturally fit the hierarchical
  predictive-coding frame. This is the obvious extension for the
  interoceptive-pain agent.
- **No reinforcement learning.** The model is fitted by unsupervised
  prediction-error minimisation; there is no reward signal and no policy.
- **Two levels only.** A deeper hierarchy is left as future work.

### Relevance to interoceptive-pain RL

- **Direct template for hypernet-modulated predictive coding.** The
  two-level model is essentially the architecture
  `professor-bayesian-brain` would propose: a higher level $\mathbf
  r^{(2)}$ that *retunes the lower-level dynamics* via a small mixture
  vector. For the interoceptive agent, $\mathbf r^{(2)}$ becomes a candidate
  *precision* or *threat* state that re-weights perceptual transition
  primitives. This is one of the cleanest published instances of the
  architecture the project is exploring.
- **Mixture-over-basis as a "neuromodulator" abstraction.** The choice to
  generate a $K$-dim mixture vector rather than raw weights is the key
  design move: it sidesteps the biological-plausibility critique
  (`professor-neuromodulation`'s domain) while preserving most of the
  flexibility of full hypernet weight generation.
- **Slow-fast timescale emergence.** The empirical result that
  $\mathbf r^{(2)}$ stays stable until prediction errors *force* it to
  change is the right shape for a "tonic" interoceptive precision signal
  with phasic updates on threat detection. A direct port to the project
  would replace the natural-video reverse-correlation analysis with a
  pain/no-pain regime change.
- **Recommendation.** If the project wants a *first* hypernet experiment, a
  closer model than Paper 1's HyperRNN is this paper's mixture-of-V variant
  — it is more parameter-efficient, easier to interpret, and has an
  existing predictive-coding formalism to inherit from. Suggest a brief
  `issue_plan` from `senior-developer` to scope a $K$-mixture transition
  module on top of the current Dreamer-V3 RSSM.

---

## <a id="paper-4"></a>Paper 4 — Hypernetwork approach to Bayesian MAML (Borycki et al., 2022)

**Citation:** Borycki, P., Kubacki, P., Przewięźlikowski, M., Kuśmierczyk,
T., Tabor, J., & Spurek, P. (2022). *Hypernetwork approach to Bayesian
MAML.* arXiv:2210.02796.

**PDF:** `docs/project/references/Hypernetwork/sources/Borycki et al. 2022 - Hypernetwork approach to Bayesian MAML.pdf` (14 pages)

### Backbone

- **Problem.** Model-Agnostic Meta-Learning (MAML; Finn et al. 2017) learns
  shared universal weights $\theta$ that adapt to a new task by a few
  gradient steps, but (i) it overfits with deep architectures and (ii) it
  produces *point* estimates with no uncertainty quantification. Bayesian
  variants (BayesianMAML, FO-MAML, MLAP-M, PACOH) exist but inherit MAML's
  hierarchical $\theta \to \theta'_i \to T_i$ structure, force a Gaussian
  posterior, and require coupled priors. The paper proposes
  *BayesianHMAML*: replace MAML's gradient-based adaptation with a
  *hypernetwork* that maps a support set $S_i$ and the universal weights
  $\theta$ to *posterior parameters* (mean+covariance for a Gaussian or
  conditioning vector for a continuous normalising flow), so the
  task-specific posterior is sampled directly rather than walked into via
  SGD.
- **Background — MAML and HyperMAML (Section 2).** MAML's adaptation step:

$$
\theta'_i = \theta - \alpha \nabla_\theta \mathcal L_{T_i}(f_\theta).
$$

HyperMAML replaces this with $\theta'_i = \theta + H_\phi(S_i, \theta)$.
- **BayesianHMAML (Section 3).** *Universal* weights $\theta$ are still
  point-wise (no distribution), but the *task-specialised* update is a
  *posterior distribution*:

$$
\theta'_i \sim q(\theta \mid \lambda_i(\theta, S_i)),
\quad
\lambda_i(\theta, S_i) := H_\phi(\theta, S_i).
$$

- **Two variants (Section 3 cont.).**
  - *BayesianHMAML-G* (Gaussian): $H_\phi$ outputs $(\mu_\theta(S_i),
    \sigma_\theta(S_i))$ and $\theta'_i \sim \mathcal N(\theta +
    \mu_\theta(S_i), \mathrm{diag}(\sigma_\theta^2(S_i)))$.
  - *BayesianHMAML-CNF* (Continuous Normalising Flow): $H_\phi$ outputs a
    conditioning vector $C(\theta, S_i)$ that conditions a continuous
    normalising flow $F_{\eta, C}$, allowing arbitrary non-Gaussian
    posteriors.
- **Loss (Section 3, "Learned objective").** The variational lower bound
  uses standard normal priors $p(\theta'_i) = \mathcal N(0, I)$ and drops
  the coupling between $\theta$ and $\theta'_i$:

$$
\mathcal L_{\mathcal D}^{\text{ours}} = \sum_i \mathbb E_{q(\theta'_i \mid
\lambda_i(\theta, S_i))} \big[ \log p(T_i \mid \theta'_i)
- \gamma\, KL\big(q(\theta'_i \mid \lambda_i)\,\|\, p(\theta'_i)\big) \big].
$$

The hyperparameter $\gamma$ explicitly trades likelihood vs. prior weight.
- **Architecture (Figure 2).** Encoder $E$ embeds support inputs $\to$ matrix
  $E_S$; concatenated with true labels $Y_S$ and predictions
  $\hat Y_S = f_\theta(e_{S,k})$; permutation-invariantly aggregated; passed
  through MLP head $H_\phi$ to produce either Gaussian parameters
  $(\mu, \sigma)$ or CNF conditioning vector $C$. Universal weights
  $\theta = (\theta_E, \theta_H)$ split into encoder + classification head;
  only $\theta_H$ is updated per task.
- **Results.** Cross-domain few-shot benchmarks (mini-ImageNet, CUB,
  cross-domain mini-ImageNet → CUB) on 1-shot and 5-shot. BayesianHMAML-CNF
  is competitive with or beats BayesianMAML, MAML, ABML (Ravi & Beatson),
  Reptile, and HyperMAML on accuracy. Calibration metrics (ECE, NLL) show
  the Bayesian variants quantify uncertainty better than vanilla MAML;
  BayesianHMAML-CNF is the most flexible posterior, capturing non-Gaussian
  shapes when needed.
- **Discussion.** The contribution is methodological: (i) a Bayesian
  treatment of *only* the per-task update, sidestepping the hierarchical
  coupling; (ii) using a hypernetwork as a *posterior-parameter generator*
  rather than a weight generator; (iii) plug-in flexibility — Gaussian for
  speed, CNF for expressivity.

### Phase 1 — Foundational synthesis

MAML is the reigning recipe for few-shot learning. You learn one set of
"universal" weights so that, given a tiny new training set $S_i$, a few
gradient steps adapt the network to the task. It works but has two
problems: it overfits when the network is deep, and it gives only a
single answer per task with no uncertainty estimate.

The Bayesian fix has been to put a distribution on both the universal
weights $\theta$ and the task-specific weights $\theta'_i$. That sounds
clean but introduces a coupled hierarchical prior $\theta \to \theta'_i$
that complicates optimisation, forces both distributions to be the same
shape (usually Gaussian), and limits how much $\theta'_i$ can drift from
$\theta$.

Borycki et al.'s contribution is small but impactful: keep the universal
weights $\theta$ *point-wise* — same as plain MAML — and use a
*hypernetwork* to produce the *posterior over $\theta'_i$* directly from
the support set $S_i$. The hypernetwork eats the support set (after a small
encoder) and outputs the parameters of $q(\theta'_i)$. In the simplest
case those are mean and standard-deviation vectors of a Gaussian; in the
more flexible case the hypernet conditions a normalising flow, so the
posterior can be any shape.

The mechanism is essentially "Bayesian conditioning by amortisation":
instead of approximating the posterior by gradient descent (the
variational-inference school) you train a network to output it directly.
The novelty is plugging that into the meta-learning frame, where the
posterior must adapt to a *task* rather than to a single dataset.

For the project, this paper matters because it establishes a clean
template for *uncertainty-aware* hypernetwork conditioning. Any
interoceptive-pain agent that wants the precision signal to drive *not just
a point estimate but a distribution over policy weights* — i.e. an agent
that *knows when it does not know how to act* — has a precedent here. The
Gaussian variant requires only a tweak to a standard hypernet head
(predict $\mu$ *and* $\log \sigma$ instead of just $\mu$), so it is also
the cheapest route to a Bayesian-flavoured hypernet baseline in the
project's RPPO/Dreamer stack.

### Phase 2 — Graduate-level deep dive

#### MAML and HyperMAML recap

MAML's bilevel optimisation: inner loop adapts $\theta$ to task $T_i$ by
$N$ gradient steps,

$$
\theta'_i = \theta - \alpha \sum_{j=1}^{N} \nabla_{\theta_{j-1}}
\mathcal L_{T_i}(f_{\theta_{j-1}}),
\quad \theta_0 = \theta,
$$

and outer loop minimises the post-adaptation loss across tasks,

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{T_i \sim p(T)}
\mathcal L_{T_i}(f_{\theta'_i}).
$$

HyperMAML (Przewięźlikowski et al. 2022) replaces the inner loop:

$$
\theta'_i = \theta + H_\phi(S_i, \theta),
$$

where $H_\phi$ is a hypernet trained jointly with the universal weights.
This removes the need for inner-loop differentiation through a chain of
gradient steps.

#### Hierarchical Bayesian MAML — what BayesianHMAML avoids

The standard hierarchical Bayesian formulation has

$$
\mathcal L_{\mathcal D} = \sum_i
\mathbb E_{q(\theta\mid\psi)} \mathbb E_{q(\theta'_i\mid\lambda_i)}\!\left[
\log p(T_i \mid \theta'_i) - KL\big(q(\theta'_i\mid\lambda_i) \,\|\, p(\theta'_i \mid \theta)\big)
\right]
- KL\big(q(\theta\mid\psi)\,\|\,p(\theta)\big).
$$

The coupling prior $p(\theta'_i \mid \theta)$ — typically Gaussian centred
at $\theta$ — is the source of the problem: the per-task posterior cannot
deviate too far without paying a large KL penalty, and the universal
posterior $q(\theta\mid\psi)$ has to be the same shape (Gaussian) to make
$KL$ tractable. BayesianMAML (Yoon et al. 2018) uses Stein Variational
Gradient Descent to relax the second constraint; PACOH replaces the
coupling with a learned prior; but all keep the hierarchical structure.

#### BayesianHMAML — non-hierarchical objective

BayesianHMAML drops the hierarchical structure entirely. The universal
weights are point-wise; only the per-task posterior is distributional. The
prior is a fixed standard normal on $\theta'_i$:

$$
\mathcal L_{\mathcal D}^{\text{ours}}
= \sum_{i=1}^{N} \mathbb E_{q(\theta'_i \mid \lambda_i(\theta, S_i))}
\Big[ \log p(T_i \mid \theta'_i)
  \;-\; \gamma \cdot KL\!\big( q(\theta'_i \mid \lambda_i) \,\|\, p(\theta'_i) \big) \Big],
\quad p(\theta'_i) = \mathcal N(0, I).
$$

The $\gamma$ scalar ($0 < \gamma \le 1$ in practice) is a tempering knob
that absorbs prior misspecification — small $\gamma$ trusts the data, large
$\gamma$ trusts the prior. Because $p(\theta'_i)$ is fixed, there is no
hyper-prior to learn and no chain $\theta \to \theta'_i \to T_i$.

#### Gaussian variant — closed-form KL

The hypernet outputs

$$
(\mu_\theta(S_i),\, \sigma_\theta(S_i)) := H_\phi(S_i, \theta),
$$

and samples are drawn via reparameterisation,

$$
\theta'_i = \theta + \mu_\theta(S_i) + \sigma_\theta(S_i) \odot \varepsilon,
\quad \varepsilon \sim \mathcal N(0, I).
$$

For $q = \mathcal N(\theta + \mu, \mathrm{diag}(\sigma^2))$ and $p = \mathcal N(0, I)$
the KL has the standard closed form

$$
KL(q \,\|\, p) = \tfrac{1}{2} \sum_{k=1}^{|\theta'_i|}
\Big( (\theta_k + \mu_k)^2 + \sigma_k^2 - 1 - \log \sigma_k^2 \Big),
$$

so the ELBO is differentiable end-to-end. The mean-field assumption
(diagonal covariance) is partially relaxed because $\sigma$ is itself
produced by the same hypernet that produces $\mu$, so the per-coordinate
standard deviations are *coupled* through the shared hypernet weights $\phi$
even though the formal posterior is diagonal — the paper explicitly notes
this as an *amortisation-induced regularisation*.

#### CNF variant — flow-conditioned posterior

A normalising flow transforms a base random variable $z \sim p_Z$ through
an invertible map $F: Z \to Y$ to get $y = F(z)$ with density

$$
\log p_Y(y) = \log p_Z(F^{-1}(y))
- \sum_{k=1}^K \log \big| \det \partial_z F_k \big|.
$$

In the *continuous* version (Chen et al. 2018), $F$ is the time-1 flow of
an ODE $\dot z(t) = g_\eta(z(t), t)$, and the change-of-variables term
becomes

$$
\log p_Y(y) = \log p_Z(F^{-1}(y))
- \int_{t_0}^{t_1} \mathrm{Tr}\!\left( \frac{\partial g_\eta}{\partial z(t)} \right) dt.
$$

For BayesianHMAML-CNF, the hypernet emits a conditioning vector
$C(\theta, S_i)$ that is concatenated to each layer of $g_\eta$:

$$
\dot z(t) = g_{\eta, C(\theta, S_i)}(z(t), t),
\quad \Delta\theta'_i \sim F_{\eta, C(\theta, S_i)}(z),
\quad z \sim \mathcal N(0, t \cdot I)\;(t = 0.1\text{ in experiments}),
$$

giving

$$
\theta'_i = \theta + \Delta\theta'_i.
$$

The KL term in the ELBO is no longer closed-form but is estimated by
Monte-Carlo with the trace-Jacobian formula above. The posterior shape is
no longer Gaussian; the paper shows this matters when the true posterior
is multimodal or skewed.

#### What is generated by the hypernetwork — comparison to other papers

| Method | What $H_\phi$ outputs | Posterior shape |
|---|---|---|
| Ha et al. 2016 (Paper 1, dynamic) | weight-scaling vector $d(z)$ | none (point) |
| Jiang et al. 2021 (Paper 3) | mixture vector $\mathbf w \in \Delta^{K-1}$ | none (point) |
| HyperMAML | weight delta $\Delta\theta$ | none (point) |
| BayesianHMAML-G | $(\mu, \sigma)$ for $\Delta\theta$ | diagonal Gaussian |
| BayesianHMAML-CNF | conditioning vector $C$ for an ODE flow | flow-induced |

Two design dimensions emerge: *what the hypernet conditions on* (input
sequence vs. support set vs. learned embedding) and *what its output
parameterises* (raw weights vs. modulation vector vs. posterior
parameters). Papers 5–7 will explore further combinations.

#### Limitations the authors flag

- *Mean-field diagonal posterior* in the Gaussian variant is still a
  significant restriction; the CNF variant relaxes it but is more
  expensive.
- *Only the classification head* $\theta_H$ is generated by the
  hypernetwork; the encoder $\theta_E$ is shared and point-wise. This is
  a design choice for tractability — generating the entire encoder would
  blow up the hypernet output dimension.
- *Standard-normal prior $p(\theta'_i) = \mathcal N(0, I)$* is the
  simplest choice but may be overly tight; the $\gamma$ knob is the
  practical compensation.

### Relevance to interoceptive-pain RL

- **Direct route to a Bayesian-flavoured precision-modulation hypernet.**
  Replacing the project's planned hypernet head with a BayesianHMAML-G
  head — predict $(\mu, \sigma)$ instead of $\mu$ alone — is a one-line
  architectural change. The posterior over policy weights becomes the
  natural carrier of "uncertainty about how to act" given a precision
  signal. This is the cheapest Bayesian extension to the project's
  hypernet baseline.
- **Caveat: support-set conditioning, not sequence conditioning.** The
  paper's hypernet eats a *batch of labelled examples* $S_i$, which is
  natural for few-shot but not for online RL. Adapting the architecture to
  a streaming interoceptive setting needs a small redesign — replace the
  permutation-invariant set encoder with a recurrent encoder over the
  agent's recent experience. This is the same "static vs. dynamic"
  distinction Paper 1 makes; the project would need the dynamic version.
- **CNF posterior is overkill for a first experiment.** The CNF variant is
  a fine endpoint but adds an ODE solver to the training loop and is
  expensive. Recommend starting with BayesianHMAML-G as the
  Bayesian-hypernet baseline, then upgrading to CNF only if posterior
  multimodality is empirically observed (e.g. bimodal "approach vs. avoid"
  policies in the predator gridworld).
- **No connection to predictive coding or neuromodulation in the paper.**
  Any bridge to the project's framing must come from Papers 3 (predictive
  coding), 5–7 (RL), and the project's own neuromodulation corpus.

---

## <a id="paper-5"></a>Paper 5 — Hypernetwork-PPO for Continual Reinforcement Learning (Schöpf et al., 2022)

**Citation:** Schöpf, P., Auddy, S., Hollenstein, J., & Rodriguez-Sanchez,
A. (2022). *Hypernetwork-PPO for Continual Reinforcement Learning.* Deep
RL Workshop @ NeurIPS 2022.

**PDF:** `docs/project/references/Hypernetwork/sources/Schöpf et al. 2022 - Hypernetwork-ppo for continual reinforcement learning.pdf` (13 pages)

### Backbone

- **Problem.** Modern policy-gradient RL (PPO, SAC) excels on single tasks
  but undergoes catastrophic forgetting when fine-tuned on a sequence of
  related tasks. Continual-RL benchmarks like DoorGym (Urakami et al. 2019)
  expose this clearly: a PPO agent that perfectly opens pull doors collapses
  on lever doors after fine-tuning. Hypernetworks have already been shown
  effective for continual *supervised* learning (von Oswald et al. 2020) and
  for continual *model-based* RL via CEM (Huang et al. 2021,
  "HyperCRL"); this paper extends the technique to *model-free* RL with
  PPO.
- **Method — HN-PPO (Section 4).** A task-conditioned hypernetwork
  $h: C \to \theta_t$ generates the parameters of the PPO actor (and
  optionally critic) network from a low-dimensional learnable task
  embedding $t \in \mathbb R^8$. Two architectural variants:
  - **HN-PPO**: hypernetwork generates *both* actor and critic weights,
    one shared embedding per task.
  - **HN-PPO+fc**: hypernetwork generates *only* the actor; the critic is
    a plain MLP that is *re-initialised* for each new task.
- **Anti-forgetting regularisation (Eq. 2).** At the start of training task
  $T$, the network records $\Theta_t = h(t, \Theta_h)$ for every previous
  embedding $t = 0,\ldots,T-1$. During training of $T$, an L2 penalty
  pushes the new hypernetwork's outputs back toward those recorded targets:

$$
\Theta_t = h(t, \Theta_h),
\qquad
\mathcal L_{\mathrm{reg}} = \frac{\beta}{T-1} \sum_{t=0}^{T-1}
\| \Theta_t - \Theta_{t,\mathrm{new}}\|_2.
$$

A two-stage update routine is used (von Oswald et al. 2020): first a
candidate change $\Delta\Theta_h$ is computed from $\mathcal L_{\mathrm
{task}}$ (PPO clipped objective), then the *true* update is computed using
both $\mathcal L_{\mathrm{task}}$ and $\mathcal L_{\mathrm{reg}}$
evaluated *after* the candidate change has been applied.
- **PPO objective (Eqs. 3–5).** Standard clipped surrogate

$$
\mathcal L^{\mathrm{clip}}_t(\theta) = \mathbb E_t \Big[ \min\big(
\rho_t(\theta) \hat A_t,\, \mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t \big) \Big],
\quad
\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)},
$$

augmented with a value-loss $c_v \mathcal L^{\mathrm{vf}}$ and entropy
bonus $c_e S$ to give $\mathcal L^{\mathrm{total}}_t$.
- **Setup (Section 4.1).** DoorGym 6-task sequence (3 handle types × 2
  hinge sides × 2 push/pull directions, simplified to 6 tasks): pull-right,
  pull-left, lever-right-pull, lever-left-pull, lever-right-push,
  lever-left-push. Floating "blue hook" robot with 6D continuous control.
- **Continual-learning metrics (Section 4.2).** Standard accuracy matrix
  $R_{i,j}$ = success rate on task $j$ after training on $i$. From this:
  *Accuracy* $A$, *Forward Transfer* $FT$, *Backward Transfer* $BWT$,
  *Remembering* $REM = 1 - |\min(BWT, 0)|$.
- **Results (Tables 2–5).**

  | Method | Accuracy | FT | BWT+ | REM |
  |---|---|---|---|---|
  | PPO (per-task fresh) | 0.24 | 0.10 | 0.00 | 0.29 |
  | PPO-finetuning | 0.20 | 0.10 | 0.00 | 0.47 |
  | HN-PPO+fresh | 0.21 | 0.03 | 0.00 | 0.32 |
  | **HN-PPO** | **0.81** | 0.05 | 0.00 | **1.00** |
  | **HN-PPO+fc** | 0.73 | 0.04 | 0.00 | **1.00** |
  | HN-PPO+fc, no regulariser ($\beta=0$) | 0.24 | 0.03 | 0.00 | 0.24 |

  HN-PPO and HN-PPO+fc both achieve perfect remembering ($REM = 1.00$);
  removing the regulariser collapses remembering to 0.24, *worse than
  plain PPO fine-tuning*. So the hypernet alone does not prevent
  forgetting — the L2 anchor on previous outputs does.
- **Limitations (Section 7).** PPO's training instability is *amplified*
  in the continual setting; the authors had to reduce the gradient-norm
  clip from 0.5 to $10^{-4}$ to prevent gradient explosions. Training a
  single task takes up to a day. Performance is highly seed-dependent.
  The metric *return* turns out to be a poor proxy for the metric *success
  rate* — three runs with similar reward curves can have wildly different
  door-opening rates (Figure 4).

### Phase 1 — Foundational synthesis

A robot that learns to open one type of door and is then trained to open a
different type forgets the first. This is catastrophic forgetting, and PPO
— like every other gradient-based method — suffers from it badly. The
naïve fix is to keep all the old training data and replay it; that does
not scale.

The cleaner fix this paper studies is to *swap out* the policy's weights
for each task. A small hypernetwork takes a task ID (encoded as an
8-dimensional learnable vector) and outputs an entire policy. To learn
task 0 you train the hypernet to output good policy weights for embedding
$t_0$; to learn task 1 you train it to output good policy weights for
$t_1$, *while keeping its outputs at $t_0$ unchanged*. The "unchanged"
constraint is enforced by an L2 anchor: every gradient step on the new
task is paired with a gradient step that pushes the hypernet's output at
each old embedding back toward what it was before training started.

The clever bit is that this is cheap. To remember $T-1$ previous tasks you
need to store $T-1$ low-dimensional embeddings (8 numbers each here), not
$T-1$ networks or $T-1$ replay buffers. The hypernetwork itself does not
grow — only the embedding vector is added per task, so the model size
scales as $\mathcal O(1) + \mathcal O(T)$ rather than $\mathcal O(T)$ in
the per-task-network sense.

The empirical result is dramatic. Plain PPO fine-tuning across the 6-task
sequence collapses (24% average accuracy, 47% remembering); HN-PPO with
the regulariser holds 81% accuracy and 100% remembering — perfect
preservation of skills across the entire sequence.

The paper's most important sanity check is the ablation that turns the
regulariser off ($\beta = 0$): performance collapses to 24% accuracy and
24% remembering, *worse than plain PPO fine-tuning*. So the
hypernetwork architecture by itself does not prevent forgetting — the L2
anchor on previous outputs is what does the work. The hypernetwork merely
*provides the architectural slot* for that anchor: a low-dimensional
input $t_i$ whose pre-image in weight space the regulariser can pin.

For the project's interoceptive-pain agent, the most relevant takeaways
are:
1. *Continual learning of multiple tasks (e.g., food vs. predator
   contexts) is feasible with a hypernet-conditioned PPO at low memory
   cost*.
2. *The L2-anchor regulariser is the mechanism, not the hypernet*.
   Replacing the project's recent RPPO continual-learning demo (commit
   `de102b8`, NoPred → PredInt3) with HN-PPO would only help if the
   regulariser is also used.
3. *The regulariser is computed via a forward pass through all previous
   embeddings each gradient step, so cost grows $\mathcal O(T)$ in number
   of past tasks*. For long curricula this becomes expensive.

### Phase 2 — Graduate-level deep dive

#### Architecture (Section 4)

For a task-incremental setup $X \times C \to Y$ (van de Ven & Tolias 2019),
the task ID $C$ is encoded as a learnable embedding $t_i \in \mathbb R^8$.
The hypernet $h: C \to \mathbb R^{|\theta_{\mathrm{policy}}|}$ produces a
*flat* parameter vector $\Theta_{t_i}$ that is reshaped into the actor /
critic MLP. In *HN-PPO* both networks share the embedding output; in
*HN-PPO+fc* only the actor's parameters come from the hypernet, the
critic is reset.

A single forward pass at task $i$ at simulation time $\tau$ thus has two
phases:
1. **Weight realisation**: $\Theta_{t_i} = h(t_i; \Theta_h)$, computed
   *once* at the start of an episode (or batch).
2. **Policy execution**: $a_\tau \sim \pi_{\Theta_{t_i}}(\cdot | s_\tau)$,
   $V_{\Theta_{t_i}}(s_\tau)$, the standard PPO inner loop.

#### Anti-forgetting via L2 anchor on hypernet outputs

Let $\Theta_h^{\mathrm{old}}$ denote the hypernet weights at the *start*
of training on task $T$. Record the targets

$$
\Theta_t^{\star} := h(t; \Theta_h^{\mathrm{old}}),\quad t = 0, 1, \ldots, T-1.
$$

These are *frozen* throughout the training of task $T$. The
regularisation loss penalises drift away from these targets, evaluated on
the *current* hypernet weights $\Theta_h$:

$$
\mathcal L_{\mathrm{reg}}(\Theta_h) =
\frac{\beta}{T-1} \sum_{t=0}^{T-1} \| h(t; \Theta_h) - \Theta_t^\star \|_2.
$$

The full update on the hypernet at task $T$ is

$$
\Theta_h \leftarrow \Theta_h - \eta\, \nabla_{\Theta_h}\!\left(
\mathcal L_{\mathrm{task}}^{\mathrm{PPO}}(\Theta_h) + \mathcal L_{\mathrm{reg}}(\Theta_h)
\right),
$$

with the **two-stage** subtlety from von Oswald et al. (2020): the
gradient $\nabla_{\Theta_h} \mathcal L_{\mathrm{reg}}$ is evaluated at the
*candidate* point $\Theta_h - \eta\, \nabla_{\Theta_h} \mathcal L_{\mathrm
{task}}$, not at the current $\Theta_h$. This makes the regulariser act
*as a counter-step* against the task gradient that would have been
applied, so it can fully cancel forgetting-inducing components of
$\nabla_{\Theta_h} \mathcal L_{\mathrm{task}}$.

Per-step computational cost:
- Standard PPO step: one forward / backward pass through actor + critic
  (cost $\mathcal O(|\theta_{\mathrm{policy}}|)$).
- HN-PPO step: $1 + (T-1)$ forward passes through the *hypernet* (one for
  the current task, $T-1$ for previous task targets), plus a backward pass
  on the combined loss. Cost is $\mathcal O(T \cdot |\Theta_h|)$.

For DoorGym's 6 tasks this is manageable; for the 100+ task curricula in
Beck et al. 2023 (Paper 6), it becomes a bottleneck.

#### Why the L2 anchor works (informal argument)

The hypernet implements a function $\Theta_t = h(t; \Theta_h)$, smooth in
both arguments. The Lipschitz constant in the embedding $t$ controls how
"orthogonal" different task slots are: nearby embeddings produce similar
weights, distant embeddings can produce quite different weights. The L2
anchor *exactly preserves the values at $T-1$ specific embedding points*.
Between those points the hypernet is free to adapt, so a task $T$ whose
embedding $t_T$ is far enough from $\{t_0, \ldots, t_{T-1}\}$ in
$\mathbb R^8$ can be fitted without disturbing the anchors.

The 8-dimensional embedding space is large enough that 6 task points sit
nearly orthogonally; the regulariser then only constrains a 6-dimensional
slice of the hypernet's output manifold. This is the geometric reason a
*single* hypernet plus L2 anchor can hold 6 distinct policies, whereas a
shared MLP (effectively a hypernet whose only "embedding" is the empty
prefix) cannot.

#### Connection to Elastic Weight Consolidation (EWC)

EWC (Kirkpatrick et al. 2017) regularises *parameter changes* weighted by
the Fisher information. Schöpf's regulariser regularises *output changes*
on a held-out anchor set. The key difference: EWC's anchor is implicit
(the Fisher) and depends on the curvature of the per-task loss; HN-PPO's
anchor is explicit (the embedding set) and depends only on the hypernet
architecture. For a hypernet that has many more weights than tasks,
output anchoring is *much* tighter than parameter anchoring — the L2
penalty exactly preserves the function on the anchor set, but EWC only
preserves it approximately.

#### Empirical observations not predicted by theory

- **Stability problems.** PPO's policy gradient is noisy; combining it with
  a hypernet output amplifies the noise. The authors had to drop the
  gradient-norm clip from 0.5 to $10^{-4}$ — three orders of magnitude —
  to avoid explosions. This is a practical warning for any
  hypernet-modulated RL implementation.
- **Stop-time sensitivity.** The exact training step at which one task
  "ends" and the next "begins" affects downstream tasks even when the
  per-task convergence metric is flat. The authors hypothesise that the
  hypernet has hidden internal dynamics that depend on training-state
  history beyond the loss curve.
- **Reward ≠ success rate.** Three runs with identical reward learning
  curves can have wildly different door-opening rates. This is a benchmark
  warning specific to DoorGym (sparse-reward terminal success metric vs.
  dense shaping reward) and matches the project's own emphasis on
  *survival steps* over cumulative reward.

### Relevance to interoceptive-pain RL

- **Drop-in template for the project's continual-learning demo.** The
  RPPO continual-learning demo (commit `de102b8`, NoPred → PredInt3) is
  exactly the setup where HN-PPO would be the natural upgrade: two tasks,
  task-incremental, task ID known at train and eval. Expected gain is
  clearly bounded (this paper shows up to 0.81 vs 0.20 PPO-finetuning on
  6 tasks). For the project's 2-task setting the gain is smaller but the
  forgetting risk is what the paper *robustly* mitigates.
- **The anchor is the active ingredient.** Recommend that any
  `senior-developer` `issue_plan` for HN-PPO explicitly include the
  output-anchor regulariser. Without it, performance is *worse* than
  plain PPO fine-tuning — the paper's ablation makes this unambiguous.
- **Cost grows with curriculum length.** For long curricula the
  $\mathcal O(T)$ regulariser pass becomes the bottleneck; consider only
  if the project's planned curricula are $\le 10$ tasks.
- **No predictive-coding or pain-modelling content.** The paper is purely
  algorithmic; bridges to the project's framing must come from Papers 3
  (predictive coding) and 7 (RL transfer with continuous task descriptors,
  closer to a precision signal).
- **Caveat on "task ID = precision signal".** A precision or
  interoceptive signal is *continuous* and varies *within* an episode;
  HN-PPO's task ID is *discrete* and fixed *across* an episode.
  Generalising from task-incremental CL to within-episode precision
  modulation requires the dynamic-hypernet variant (Paper 1, HyperRNN, or
  Paper 7's RL hypernet), not the static-task-ID HN-PPO formulation.

---

## <a id="paper-6"></a>Paper 6 — Hypernetworks in Meta-Reinforcement Learning (Beck et al., 2023)

**Citation:** Beck, J., Jackson, M., Vuorio, R., & Whiteson, S. (2023).
*Hypernetworks in Meta-Reinforcement Learning.* CoRL 2022.

**PDF:** `docs/project/references/Hypernetwork/sources/Beck et al. 2023 - Hypernetworks in Meta-Reinforcement Learning.pdf` (10 pages)

### Backbone

- **Problem.** Multi-task RL methods often fail to outperform a *degenerate*
  baseline that simply trains a separate policy per task (Yu et al. 2020;
  Meta-World). Hypernetworks should be the natural fix: a hypernet
  conditioned on a task ID can *replicate* per-task policies *and*
  generalise across tasks. But hypernetworks have a hidden gotcha:
  performance is extremely sensitive to *initialisation*, as Chang, Flokas
  & Lipson (2020) showed for supervised learning. The paper asks (i) does
  this sensitivity also occur in meta-RL? (ii) is there a simpler, more
  general initialisation than Chang et al.'s Hyperfan-In/Out (HFI/HFO)? and
  (iii) given a good init, do hypernets actually beat state-of-the-art
  meta-RL baselines?
- **Setup (Sections 3–4).** Meta-RL with VariBAD (Zintgraf et al. 2021) as
  the baseline. The standard architecture has a task encoder
  $g: \tau \mapsto e$ (a recurrent VAE for VariBAD) and a policy
  $\pi_\theta(a \mid s, e)$ that takes $e$ as a *concatenated input*. The
  hypernet variant replaces this with $\phi = h_\theta(e)$ generating *all*
  policy parameters, then $\pi_\phi(a \mid s)$.
- **Theoretical motivation (Section 4.1).** A linear hypernetwork without
  bias, conditioned on a one-hot task ID $e = \mathbf 1_i$, *exactly
  reproduces* per-task policies — the columns of the hypernet's weight
  matrix $W$ are the per-task policy parameters $\phi^i$. Adding a bias and
  relaxing the one-hot assumption introduces shared parameters but does
  not lose the ability to learn distinct per-task policies when needed.
- **Bias-HyperInit (Section 4.2).** Let $W, b$ be the *last layer* of the
  hypernet; let $f$ be any default base-network initialisation scheme.
  Bias-HyperInit sets

$$
W_{i,j} := 0 \quad \forall i,j,
\qquad
b := \phi_{\text{shared}} \sim f(\phi).
$$

  At initialisation the hypernet outputs *the same* base policy $\phi_{\text
  {shared}}$ for every task, regardless of the embedding. As training
  proceeds, $W$ moves away from zero and the per-task policies diversify.
  This makes *no assumptions* about the input distribution or base
  architecture.
- **Weight-HyperInit (Section 4.2).** Let $W_{:,i}$ be the $i$-th column of
  the last layer's weight matrix:

$$
W_{:,i} := \phi^i \sim f(\phi)\quad \forall i,
\qquad b := 0.
$$

  Each column is an independent draw from the base initialisation. Under
  the assumption that the embedding is *one-hot*, the resulting
  $\phi_{\text{init}} = W e + b = \phi^i$ is exactly an independent
  initialisation per task. Outside that assumption, performance is still
  better than default neural-net inits but worse than Bias-HyperInit.
- **Comparison to Hyperfan-In (HFI; Chang et al. 2020).** HFI sets the
  initial parameter variance to maintain constant activation variance
  through the *combined* hypernet+base forward pass. It is tightly tied to
  Kaiming initialisation, requires per-architecture variance derivations,
  and assumes either weights-only or biases-only generation. Bias-HyperInit
  is simpler (zero out $W$, use the base init for $b$) and works for any
  base initialisation function $f$.
- **Results (Tables 1, 2; Figure 3).**
  - Default initialisations (Kaiming, Normc, Orthogonal) *fail* on
    hypernet-VariBAD: Cheetah-Dir return drops from ~2100 (Standard) to
    378 (Kaiming), 356 (Normc), 1379 (Orthogonal); similarly across
    Walker, Ant-Dir, Humanoid.
  - **Bias-HyperInit recovers full performance** and matches HFI: 2300
    Cheetah-Dir return, 1994 Walker, 1328 Ant-Dir, 1678 Humanoid — all
    within statistical insignificance of HFI and the standard architecture
    on easy tasks.
  - On *hard* meta-RL benchmarks (ML1 Pick-Place, ML10), where the
    standard architecture has more headroom, hypernet-VariBAD with
    Bias-HyperInit gives a *9-fold* improvement on Pick-Place (4.4% →
    42.9% test success) and a *2-fold* improvement on ML10 (10.2% →
    23.9%). RL2 with Bias-HyperInit also doubles ML10 success (7.2% →
    14.2%).
  - **FiLM is also improved by Bias-HyperInit** (Pick-Place 5.5% → 25.5%),
    showing the init scheme generalises to any conditional architecture
    where a hypernet outputs scaling/bias parameters.
  - **Network-size scan (Figure 3).** Hypernet matches or beats standard
    architecture across base-policy sizes from XS to XL on Cheetah-Dir
    and Walker, *and* matches or beats it for equivalent total
    parameter counts when the model is large enough.
- **Limitations (Section 6).** Bias-HyperInit's improvement is
  empirical — no convergence guarantee, no claim that it works on real
  robots, and the cost of the hypernet (more parameters total) is real.

### Phase 1 — Foundational synthesis

Meta-reinforcement-learning aims to learn a *learning algorithm* — given a
new task drawn from a distribution, produce a competent policy after only a
few episodes. The state-of-the-art recipe (VariBAD) extracts a task
embedding from the agent's recent trajectory and feeds it into a
fixed-architecture policy as a side input. Beck et al. ask: should the
embedding instead *generate the policy weights* via a hypernetwork?

In principle, hypernets should be at least as good. A linear hypernet with
no bias, conditioned on a one-hot task ID, literally *is* "train one policy
per task" — the columns of its weight matrix become the per-task policies.
And once you allow the embedding to be continuous and the hypernet to be
nonlinear, you get generalisation across tasks for free. So why has nobody
made this work in meta-RL?

The answer is a sharp empirical observation: hypernet performance depends
catastrophically on initialisation. Plain Kaiming initialisation —
universally used for ordinary neural networks — *fails* on hypernet-VariBAD
across every benchmark tested. Cheetah-Dir return drops from 2100 to 378.
The reason, traced back to Chang et al. (2020) for supervised learning, is
that a Kaiming-initialised hypernet produces base-network weights with
*exploding or vanishing* activation variance — the base policy is broken
before the first gradient step.

Chang et al.'s fix (HFI / HFO) is technically beautiful but specific to
Kaiming initialisation and to particular base-architecture choices.
Beck et al.'s fix is much simpler: zero out the hypernet's last-layer
weight matrix and set its bias to a sample from the *base* network's
ordinary init. At initialisation the hypernet ignores the embedding and
produces a single shared base policy initialised exactly as if you had
trained one base network. As training proceeds, the weight matrix moves
away from zero and the per-task policies differentiate. They call this
*Bias-HyperInit*.

The empirical story is then clean: with Bias-HyperInit the hypernet
matches the standard architecture on easy meta-RL benchmarks (where the
ceiling is already saturated) and *substantially* beats it on hard ones
(9× improvement on Meta-World Pick-Place; 2× on ML10).

A separate result is that Bias-HyperInit also fixes *FiLM* under hypernet
conditioning: when FiLM's $\gamma, \beta$ are themselves generated by a
hypernet, the same init pathology applies and the same fix works. This is
the most direct link to the project's FiLM-vs-hypernet question — it
suggests that a fair comparison must control for initialisation, not just
architecture.

For the project the takeaways are:
1. *Naïve init breaks hypernets in RL*. Any baseline that uses Kaiming /
   Orthogonal / Normc on a hypernet-conditioned policy will report
   spuriously bad numbers.
2. *Bias-HyperInit is the fix*. It is one line of code (zero out the last
   layer's $W$, leave $b$ to be initialised by the base scheme) and
   works for any base architecture and any base init.
3. *Where hypernets pay off*. On easy tasks the standard architecture
   already saturates and the hypernet matches but does not beat. On hard
   tasks (Meta-World) the hypernet *substantially* beats the standard
   architecture. This is consistent with the modularity story of Paper 2:
   tasks that genuinely need different policies are exactly where
   hypernets should win.

### Phase 2 — Graduate-level deep dive

#### Linear hypernet replicates per-task training (Section 4.1)

Consider a linear hypernet $h_\theta(e) = W e + b$ with $\theta = (W, b)$.
The base policy parameters are $\phi = h_\theta(e)$. If $e = \mathbf 1_i$
(one-hot task ID) and $b = 0$, then

$$
\phi^i = h_\theta(\mathbf 1_i) = W \mathbf 1_i = W_{:,i},
$$

so the $i$-th column of $W$ *is* the policy for task $i$. Training the
hypernet via the meta-RL loss $\sum_i \mathcal L_i(\pi_{\phi^i})$ is
*identical* to training $T$ separate networks (with parameter
$\phi^i = W_{:,i}$). Adding a non-zero bias introduces a shared component:

$$
\phi^i = W_{:,i} + b,
$$

so all per-task policies share $b$ and differ only in the task-specific
column. Relaxing the one-hot assumption replaces the column-selection by a
linear combination $\phi = \sum_j e_j W_{:,j}$, which is the standard
*linear* hypernet. So the standard architecture is recovered as a special
case where only the bias is task-dependent (FiLM style) and the
"per-task-policy" extreme is recovered as another special case (no
shared bias).

#### Why default initialisation fails

In a Kaiming-initialised hypernet, the last layer's outputs are
distributed as $\phi \sim \mathcal N(0, \sigma_h^2 I)$, where $\sigma_h^2$
is set to maintain activation variance through the *hypernet*. But the
*base network's* activation variance depends on the variance of $\phi$,
not on $\sigma_h$. The two are typically mismatched by an order of
magnitude or more, so the base network's activations either vanish
($\sigma_\phi^2 \ll$ base-Kaiming target) or explode
($\sigma_\phi^2 \gg$ base-Kaiming target). Either way the gradient signal
through the policy is broken before training can correct it.

#### Hyperfan-In (Chang et al. 2020) — the fix in supervised learning

HFI re-derives the hypernet last-layer variance so that the *base*
network's activations are correctly scaled. For a hypernet last layer with
input dimension $d_e$ and output dimension $d_\phi$, where the base
network has fan-in $d_{\text{base-in}}$ and uses Kaiming with target
variance $2/d_{\text{base-in}}$, HFI sets

$$
\mathrm{Var}(W_{\text{hyp},ij}) = \frac{2}{d_e \cdot d_{\text{base-in}}},
\quad
b_{\text{hyp},i} = 0.
$$

The derivation requires per-architecture variance analysis (especially when
the base network has biases generated separately) and is tied to Kaiming
initialisation specifically. It works but is brittle.

#### Bias-HyperInit derivation (Section 4.2)

Let the hypernet head be linear with output $\phi = W x + b$, where $x$ is
the final hidden layer of the hypernet. Choose

$$
W_{ij} = 0 \quad \forall i,j,
\qquad
b = \phi_{\text{shared}} \sim f(\phi),
$$

where $f$ is the *base* network's initialiser (Kaiming, Orthogonal, etc.).
Then at the start of training,

$$
\phi_{\text{init}} = W x + b = 0 \cdot x + \phi_{\text{shared}} = \phi_{\text{shared}}
$$

regardless of the input $x$. Every task gets the *same* base-policy
initialisation, drawn exactly from $f$ — so the base network's activation
statistics are guaranteed correct *for every task*, regardless of base
architecture or base init scheme. As training proceeds, gradient updates
move $W$ away from zero, and the per-task policies start to differentiate.

The mathematical guarantee is that the base network's *forward-pass
behaviour at $t=0$* is *exactly* what it would be under standalone $f$
initialisation. No assumptions about embedding distribution, base
activation function, or hypernet depth are required.

#### Weight-HyperInit derivation

Set

$$
W_{:,i} = \phi^i \sim f(\phi)\quad \forall i,
\qquad b = 0.
$$

Under the *one-hot* assumption $x = \mathbf 1_i$ (e.g. discrete task IDs
with a learned linear embedding that has converged to one-hot at
initialisation), the produced policy is $\phi_{\text{init}} = W \mathbf 1_i =
\phi^i \sim f(\phi)$ — each task gets an *independent* base init. This
is the "train T networks separately" extreme, valid only under one-hot
embeddings. Outside this regime (continuous or low-dimensional embeddings,
which is the meta-RL norm), the activations are not exactly
$f$-distributed but the variance is approximately preserved.

The empirical comparison (Table 1) shows Weight-HyperInit is competitive
but generally beaten by Bias-HyperInit, consistent with the analysis: in
practice the embedding is rarely close to one-hot.

#### Hypernet vs. FiLM under Bias-HyperInit (Table 2)

The most concrete result for the project is the FiLM ablation. Under
default init, FiLM-VariBAD on Pick-Place gets 5.5% test success. Under
Bias-HyperInit, it gets 25.5%. So even in FiLM — where only the
$(\gamma, \beta)$ parameters are generated — the same init pathology
applies and the same fix works. *Hypernet-VariBAD with Bias-HyperInit
beats FiLM-VariBAD with Bias-HyperInit (42.9% vs. 25.5%)*, supporting
Galanti & Wolf's modularity argument: generating *all* parameters beats
generating only $(\gamma, \beta)$.

#### Network-size analysis (Figure 3)

The hypernet introduces extra parameters (the hypernet's own weights plus
the embedding). For a fair comparison the authors plot return vs. base-
policy size *and* return vs. total parameter count. The hypernet
consistently equals or beats the standard architecture for equivalent
*base-policy* size, and beats it for equivalent *total* parameter count
once the model is large enough. So the cost of the hypernet (extra total
parameters) does not buy the gain — the gain comes from the architectural
modularity.

### Relevance to interoceptive-pain RL

- **Init is the make-or-break detail.** If the project plans an ablation
  comparing hypernet-PPO against FiLM-PPO on the gridworld, *both*
  architectures must use Bias-HyperInit (or HFI). Default Kaiming /
  Orthogonal init will silently kill the hypernet variant and the
  comparison will be uninformative.
- **The hypernet→FiLM pipeline.** If Bias-HyperInit also rescues FiLM
  (and the paper shows it does), then the same fix applies to the
  project's FiLM RPPO baseline if its FiLM parameters are themselves
  generated by a small embedding network. Worth double-checking the
  current FiLM init in `configs/models/`.
- **Where hypernets actually pay off.** Easy tasks saturate and hypernets
  match the standard arch; hard tasks (Meta-World scale) diverge and
  hypernets win. The project's predator gridworld has ~6 environment
  configurations — likely "easy" by Meta-World standards. Expect
  hypernet to *match* but not necessarily *beat* a well-tuned FiLM
  baseline; the upgrade story should be evaluated on genuinely hard
  tasks (e.g. cross-arena transfer or large food/predator-pattern
  variation).
- **No predictive coding, no Bayesian, no neuroscience.** The paper is
  empirical engineering of meta-RL. Bridges to the project's framing
  must come from Papers 3 (predictive coding) and 7 (RL transfer with
  continuous task descriptors).

---

## <a id="paper-7"></a>Paper 7 — Hypernetworks for Zero-Shot Transfer in Reinforcement Learning (Rezaei-Shoshtari et al., 2023)

**Citation:** Rezaei-Shoshtari, S., Morissette, C., Hogan, F. R., Dudek,
G., & Meger, D. (2023). *Hypernetworks for Zero-Shot Transfer in
Reinforcement Learning.* AAAI-23.

**PDF:** `docs/project/references/Hypernetwork/sources/Rezaei-Shoshtari et al. 2023 - Hypernetworks for Zero-Shot Transfer in Reinforcement Learning.pdf` (9 pages)

### Backbone

- **Problem.** Zero-shot transfer in RL: at test time the agent is given a
  *task descriptor* (reward parameters $\psi$ and/or dynamics parameters
  $\mu$ — a "context") for a previously *unseen* MDP and must produce a
  near-optimal policy with no further interaction. The paper frames this as
  *approximating* the RL algorithm itself: an RL solver is a mapping
  $\mathrm{Algorithm}: M_i \to (\pi^\star_i, Q^\star_i)$, and a hypernetwork
  $H_\Theta(\psi, \mu)$ can be trained to approximate that mapping.
- **Setup (Section 3.1).** A *parameterised MDP family*

$$
\mathcal M = \{ M_i \mid M_i = (\mathcal S, \mathcal A, T_{\mu_i}, R_{\psi_i}, \gamma) \},
\qquad \psi_i \sim p(\psi),\; \mu_i \sim p(\mu).
$$

  Each MDP shares state and action spaces but has a parameterised reward
  $R_\psi$ and parameterised dynamics $T_\mu$. The hypernet conditions on
  $(\psi, \mu)$ and outputs the parameters of an actor $\hat\pi_\theta$
  and a critic $\hat Q_\phi$:

$$
[\theta_i;\; \phi_i] = H_\Theta(\psi_i, \mu_i).
$$

- **Assumptions (Section 3.2).**
  - *Assumption 1*: $\psi_i, \mu_i$ are sampled i.i.d. from priors
    $p(\psi), p(\mu)$. (Standard meta-learning assumption.)
  - *Assumption 2*: The RL solver is *converged* on each training MDP — the
    training data is near-optimal. (Imitation-learning-style.) The authors
    note this can be relaxed in practice.
- **Method — HyperZero (Section 3.2; Algorithm 1).**
  1. For each training MDP $M_i$, run a standard RL solver (TD3 in
     experiments) for 1M steps to obtain $(\pi^\star_i, Q^\star_i)$.
  2. Roll out $\pi^\star_i$ to collect a dataset $\mathcal D = \{(\psi_i,
     \mu_i, s, a^\star, s', r, q^\star)\}$.
  3. Train the hypernet by minimising the *prediction* loss

$$
\mathcal L_{\mathrm{pred}}(\Theta) =
\mathbb E_{\mathcal D}\!\left[
(\hat Q_{\phi_i}(s, a^\star) - q^\star)^2
+ (\hat\pi_{\theta_i}(s) - a^\star)^2
\right],
\qquad [\theta_i; \phi_i] = H_\Theta(\psi_i, \mu_i).
$$

  4. Add a *temporal-difference regulariser* (Section 3.3) that pushes the
     *target* value toward the *current* value estimate (note the
     direction is opposite to standard DQN/DDPG):

$$
\mathcal L_{\mathrm{TD}}(\Theta) =
\mathbb E_{\mathcal D}\!\left[
\big( r + \gamma \hat Q_{\phi_i}(s', a') - q^\star \big)^2
\right],\qquad
a' = \mathrm{stop\_grad}(\hat\pi_{\theta_i}(s')).
$$

  5. Total loss $\mathcal L = \mathcal L_{\mathrm{pred}} + \mathcal L_{\mathrm{TD}}$.
- **Evaluation (Section 4).** Tasks are continuous-control (DeepMind
  Control Suite) cheetah, walker, finger, with reward parameter
  $\psi$ = desired speed (negative = backward, positive = forward) and
  dynamics parameter $\mu$ = body / torso length (changes weight, inertia).
  85% / 15% train/test split over $(\psi, \mu)$. Compared against
  context-conditioned policy (CCP, an "embedding-method" baseline),
  CCP+UVFA, MAML, and PEARL (zero-shot and few-shot variants).
- **Results.**
  - **Reward generalisation (Figure 3).** HyperZero matches the converged
    TD3 solution across nearly the entire $\psi$ range on cheetah and
    walker; CCP performs well only near the training mean and degrades on
    extrapolation; MAML and PEARL zero-shot are far worse, *few-shot*
    PEARL recovers some of the gap.
  - **Dynamics generalisation (Figure 4).** Same picture for varying
    torso length: HyperZero stays close to TD3, baselines fall off.
  - **Joint reward + dynamics (Figure 5).** 3D return surface over
    $(\psi, \mu)$ shows HyperZero is the only method to recover
    near-optimal returns across the joint test grid.
  - **TD-loss ablation.** Removing $\mathcal L_{\mathrm{TD}}$ degrades
    out-of-training-distribution performance — the prediction loss alone
    is insufficient. The TD term enforces Bellman consistency on the
    *generated* policy / value pair, regularising it against arbitrary
    extrapolation.
- **Discussion / interpretation.** Three takeaways the authors emphasise:
  (i) hypernetworks can approximate *converged RL solvers* well enough for
  zero-shot transfer, given a smooth task family; (ii) modularity in the
  Galanti–Wolf sense (Paper 2) is the right framework — a hypernet
  generates *all* policy and critic weights; (iii) Bellman consistency
  must be added as a regulariser because supervised prediction alone does
  not capture the dynamics.

### Phase 1 — Foundational synthesis

Imagine you have trained a strong RL agent on cheetah-running at desired
speed 3 m/s with torso length 0.4. You have also trained agents at speed
−5, 0, +6 with torso lengths 0.3, 0.5, 0.6. Now you encounter a new
combination: speed +4.5 with torso 0.55. Can you produce a competent
policy *without* any further training time on this new combination?

Rezaei-Shoshtari et al. answer yes, by training a hypernetwork to
approximate the *RL solver itself*. The hypernet takes the task
parameters $(\psi, \mu)$ as input and outputs the entire weight vector of
both the actor and the critic for that task. To train it, the authors
collect for each training MDP (i) the converged TD3 actor and critic, and
(ii) rollouts from that actor. The hypernet is then trained to *match*
those actor / critic outputs everywhere on the rollout data:
$\hat\pi(s) \approx a^\star$ and $\hat Q(s, a^\star) \approx q^\star$.

This sounds like imitation learning, and it almost is — but the authors
add one crucial twist. Pure prediction loss is not enough: it overfits
to the visited states and breaks at the edges. The fix is a *temporal-
difference regulariser* that enforces Bellman consistency between the
*generated* policy and the *generated* critic. Concretely, for any
transition $(s, a^\star, s', r)$ in the data, the generated $\hat Q$
should satisfy $r + \gamma \hat Q(s', \hat\pi(s')) \approx q^\star$. The
direction is unusual: instead of moving the *current* value toward the
*target* (as in DQN), they move the target toward the current — because
the current $q^\star$ is the *known* near-optimal value from the training
solver, and the generated $\hat Q$ is the unknown that needs to match.

The empirical results are strong. On cheetah, walker, finger, HyperZero
recovers near-converged TD3 performance across the entire test slice of
$(\psi, \mu)$, and beats every baseline (context-conditioned policy,
CCP+UVFA, MAML, PEARL). The 3D test-return surface over $(\psi, \mu)$
shows the qualitative picture: HyperZero produces a smooth optimal
surface that interpolates between training points; CCP and MAML have
sharp drops away from the training mean.

For the project's interoceptive-pain agent, this is the most directly
analogous architecture in the corpus to "the policy is conditioned on a
continuous task descriptor". The descriptor in this paper is
$(\psi, \mu)$ — reward shape and dynamics shape; for the project,
the analogue is the *interoceptive precision* signal (or a more general
context). The training recipe — train per-context near-optimal policies
first, then train a hypernet to *imitate* them — is also a clean recipe
for the project's longer-term plan if it ever has access to a bank of
per-context optimal solutions (e.g., from offline pre-training on each
context separately).

The Bellman-consistency regulariser is the original methodological
contribution. It is the bridge between *supervised hypernetwork training*
(papers 1, 2, 4) and *RL hypernetwork training* (papers 5, 6) — it lets
you treat the problem as supervised but injects RL structure to prevent
overfitting on the supervised data.

### Phase 2 — Graduate-level deep dive

#### Parameterised MDP family (Definition 1, Section 3.1)

The authors are explicit about the assumptions on the MDP family. Let

$$
\mathcal M = \{ M_i = (\mathcal S, \mathcal A, T_{\mu_i}, R_{\psi_i}, \gamma) \mid \psi_i \sim p(\psi),\; \mu_i \sim p(\mu) \}.
$$

Crucially $\mathcal S, \mathcal A$ are *shared* across the family — only
$T_\mu$ and $R_\psi$ vary smoothly with the parameters. This is the same
structural assumption as Galanti & Wolf's modularity setting (Paper 2):
the target functions form a parametric family indexed by a low-
dimensional descriptor.

The authors view an RL algorithm as a *mapping*

$$
\mathrm{RL}: M_i \;\mapsto\; \big(\pi^\star_i, Q^\star_i\big).
$$

Under $M_i \equiv M(\psi_i, \mu_i)$ (the MDP is fully characterised by its
parameters and the functional forms of $T_\mu, R_\psi$) this becomes

$$
\mathrm{RL}(\psi, \mu) \;\mapsto\; \big(\pi^\star(\cdot \mid \cdot, \psi, \mu),\; Q^\star(\cdot, \cdot \mid \psi, \mu)\big),
$$

which is the form the hypernet $H_\Theta$ approximates.

#### Why hypernetwork rather than concatenated input

The authors invoke modularity (Paper 2) explicitly: a context-conditioned
policy that takes $[\psi, \mu]$ as a side input has a primary network that
must absorb the *joint* complexity of state + context. A hypernet
generates the *entire* primary, so the primary need only model one task
at a time. The empirical comparison against CCP and CCP+UVFA confirms
this: CCP's primary is the same architecture but conditioned by
concatenation; HyperZero generates it entirely. The latter wins the
out-of-training-distribution test.

#### Prediction loss

The supervised regression target for the hypernet is

$$
\mathcal L_{\mathrm{pred}}(\Theta) =
\mathbb E_{(\psi_i, \mu_i, s, a^\star, q^\star) \sim \mathcal D}
\Big[ (\hat Q_{\phi_i}(s, a^\star) - q^\star)^2 + (\hat\pi_{\theta_i}(s) - a^\star)^2 \Big],
\quad [\theta_i; \phi_i] = H_\Theta(\psi_i, \mu_i).
$$

For a deterministic optimal policy (Puterman 2014, Theorem 6.2.10) the
target $a^\star$ is well-defined, so the actor is parameterised as a
deterministic function $\hat\pi_\theta : \mathcal S \to \mathcal A$
without loss of optimality.

The data distribution is $d^{\pi^\star_i}$, the *stationary state
distribution under the optimal policy*. So the hypernet is trained to
match optimal actions on states the optimal agent visits — but it is
*not* told what to do at off-policy states. The TD regulariser fills this
gap.

#### Temporal-difference regulariser (Section 3.3)

Standard deep RL applications of TD are of the form

$$
y = r + \gamma Q_{\bar\phi}(s', \pi_{\bar\theta}(s'))
\quad \text{(target, frozen)},
\quad
\mathcal L = (Q_\phi(s, a) - y)^2,
$$

where $\bar\phi, \bar\theta$ are slow-moving target networks. The current
estimate $Q_\phi$ is moved toward the bootstrapped target $y$.

HyperZero *reverses* this. The "true" near-optimal $q^\star$ is *known*
from the training solver, so the *target* $r + \gamma \hat Q(s', a')$ is
the unknown; it should be moved toward $q^\star$:

$$
\mathcal L_{\mathrm{TD}}(\Theta) =
\mathbb E_{\mathcal D}\!\Big[
\big( r + \gamma \hat Q_{\phi_i}(s', a') - q^\star \big)^2
\Big],
\qquad a' = \mathrm{stop\_grad}\big(\hat\pi_{\theta_i}(s')\big).
$$

The stop-gradient on $a'$ prevents the actor from being penalised for
the critic's bootstrap error — only the critic is regularised.
Algebraically, this is enforcing

$$
\hat Q_{\phi_i}(s', \hat\pi_{\theta_i}(s')) \;\approx\;
\frac{q^\star - r}{\gamma},
$$

i.e. that the generated critic at $s'$ predicts the *correct
discounted-return correction* implied by the Bellman equation, evaluated
at the generated policy's action. Without this, $\hat Q(s', \cdot)$ for
$s'$ outside the visited states is unconstrained.

The total loss $\mathcal L = \mathcal L_{\mathrm{pred}} + \mathcal L_{\mathrm{TD}}$
combines (i) imitation of optimal action and value on visited states with
(ii) Bellman consistency on all transition tuples. The ablation shows
both terms are necessary for OOD generalisation.

#### Algorithm 1 — full training loop

```
Inputs: parameterised reward R_ψ, dynamics T_μ, priors p(ψ), p(μ),
        hypernet H_Θ, base actor π̂_θ, base critic Q̂_φ, RL solver, N tasks
1: D ← ∅
2: for i = 1..N:
3:   sample M_i with ψ_i ∼ p(ψ), μ_i ∼ p(μ)
4:   train standard RL on M_i to convergence → (π_i*, Q_i*)
5:   collect 10 rollouts of π_i* on M_i; store (ψ_i, μ_i, s, a*, s', r, q*) tuples
6:   D ← D ∪ {τ_i*}
7: while not converged:
8:   sample minibatch from D
9:   compute [θ_i; φ_i] = H_Θ(ψ_i, μ_i)
10:  Θ ← Θ - α ∇_Θ (L_pred + L_TD)
```

Step 4 is expensive (1M TD3 steps per task), but the hypernet training
itself (steps 7–10) is cheap supervised learning.

#### Network sizing / Lipschitz

The paper uses a 3-layer MLP hypernet with embedding dimension
$d_e = 32$ for $(\psi, \mu)$. Notably they do *not* report explicit
Lipschitz constraints on $H_\Theta$, but the smoothness of $T_\mu, R_\psi$
in $\mu, \psi$ together with the TD regulariser implicitly bounds the
extrapolation error.

#### What the paper does *not* do

- **No on-policy adaptation at test time.** Test-time inference is purely
  forward — given $(\psi, \mu)$, run $H_\Theta$ once to get $\theta_i$,
  then roll out $\hat\pi_{\theta_i}$. There is no online correction. So
  the method cannot recover from a misspecified context.
- **No Bayesian treatment.** The hypernet is point-wise; Borycki et al.
  2022 (Paper 4) is the natural Bayesian extension.
- **No predictive coding or neuroscience.** The framing is purely RL.
- **No within-episode dynamic context.** The context $(\psi, \mu)$ is
  *fixed* per episode; this is the same restriction as Schöpf et al.
  2022 (Paper 5) and is the gap to a precision-modulation interpretation.

### Relevance to interoceptive-pain RL

- **Direct architectural analogue for "policy = function of precision".**
  HyperZero with $(\psi, \mu)$ replaced by an interoceptive precision
  signal would be a near-direct port. The training recipe — train a bank
  of per-context near-optimal agents first, then train a hypernet to
  approximate them — is feasible if the project's RPPO baseline is
  cheap enough to instantiate per context (the project's gridworld is
  much cheaper than DeepMind Control). Recommend this as the
  *upper-bound* hypernet experiment: how well can a hypernet approximate
  a bank of per-precision-state agents?
- **TD-loss recipe is portable.** The reverse-direction TD regulariser
  is independent of the hypernet specifics and could be added to
  HN-PPO (Paper 5) to improve out-of-training-distribution performance.
  This is a methodological recipe worth flagging to `senior-developer`.
- **Static context limit.** The paper conditions the hypernet on a
  *fixed* per-episode descriptor. For an interoceptive precision signal
  that changes within an episode (e.g., a pain pulse), the dynamic
  hypernet of Paper 1 / Paper 3 is the closer match. A cleanest
  combination would be: HyperZero's TD-regulariser objective on top of
  a Paper-3-style mixture-of-V dynamic hypernet.
- **No predictive coding or Bayesian content in the paper itself.**
  Bridges to the project's pain-modelling framing must come from
  Papers 3 (predictive coding) and 4 (Bayesian).

---

## <a id="paper-8"></a>Paper 8 — A Brief Review of Hypernetworks in Deep Learning (Chauhan et al., 2023)

**Citation:** Chauhan, V. K., Zhou, J., Lu, P., Molaei, S., & Clifton, D.
A. (2023). *A Brief Review of Hypernetworks in Deep Learning.*
arXiv:2306.06955.

**PDF:** `docs/project/references/Hypernetwork/sources/Chauhan et al. 2023 - A Brief Review of Hypernetworks in Deep Learning.pdf` (19 pages)

### Backbone

- **Problem.** Hypernetworks have been applied to dozens of distinct
  problems (continual learning, causal inference, federated learning,
  uncertainty quantification, NAS, NLP, RL, etc.) but no review unifies
  the design space. The paper is the first survey of hypernet techniques
  in deep learning; it proposes a five-axis taxonomy and catalogues 50+
  applications.
- **HyperDNN definition (Section 2; Figure 1).** Standard DNN: $F(X;
  \Theta)$ with $\Theta$ learned. HyperDNN: $F(X; \Theta) = F(X;
  H(C; \Phi))$, where $H$ is a hypernet with parameters $\Phi$ that maps a
  context vector $C$ to the target weights $\Theta$. Only $\Phi$ is
  learnable; $\Theta$ is generated each forward pass.
- **Key advantages (Section 1).** (a) *Soft weight sharing* — multiple
  related target DNNs are generated by one hypernet, sharing structure
  across tasks. (b) *Dynamic architectures* — target architecture can
  change per input. (c) *Data-adaptive DNNs* — weights generated as a
  function of the input. (d) *Uncertainty quantification* — sampling
  from a noise-conditioned hypernet gives an ensemble of target weights.
  (e) *Parameter efficiency* — the hypernet often has fewer learnable
  parameters than the target DNN.
- **Five-axis taxonomy (Section 3, Figure 2).**
  1. **Input-based:** task-conditioned, data-conditioned,
     noise-conditioned.
  2. **Output-based (weight generation):** generate-once,
     generate-multiple (multi-head), generate-chunk-wise,
     generate-component-wise.
  3. **Variability of inputs:** static (fixed embeddings) vs. dynamic
     (input-dependent).
  4. **Variability of outputs:** static target architecture vs. dynamic.
  5. **Architecture of hypernetwork:** MLP, CNN, RNN, attention.
- **Application catalogue (Section 4; Table 2 lists 50 papers).**
  Continual learning (von Oswald et al. 2020; Schöpf et al. 2022 —
  Paper 5), Federated learning (Shamsian et al. 2021), Few-shot learning
  (Zhao et al. 2020), Manifold learning (Deutsch et al. 2019), AutoML /
  NAS (Brock et al. 2018; Zhang et al. 2019), Pareto-front learning
  (Navon et al. 2021), Domain adaptation (Volk et al. 2022), Causal
  inference (Chauhan et al. 2023), Uncertainty quantification (Krueger
  et al. 2017 Bayesian Hypernets; Ratzlaff & Fuxin 2020), Adversarial
  defence (Sun et al. 2020), Multitasking (Mahabadi et al. 2021), RL
  (Sarafian et al. 2021; Beck et al. 2023 — Paper 6;
  Rezaei-Shoshtari et al. 2023 — Paper 7), NLP (Mahabadi et al. 2021),
  Computer vision (Ha 2016 — Paper 1; Alaluf et al. 2022).
- **Decision questions for adoption (Section 5).** "Are there related
  components in the problem?" → task-conditioned hypernet for soft
  weight sharing. "Do we need a data-adaptive network?" →
  data-conditioned. "Do we need a dynamic architecture?" →
  output-dynamic hypernet. "Do we need faster training / parameter
  efficiency?" → hypernet for compression. "Do we need uncertainty
  quantification?" → noise-conditioned hypernet or hypernet+dropout.
- **Open challenges (Section 6).**
  - *Initialisation*: classical Xavier / Kaiming inits do not work for
    hypernets — Chang et al. 2020's HFI/HFO and Beck et al. 2023's
    Bias-HyperInit (Paper 6) are partial fixes but the problem is not
    fully resolved.
  - *Scalability*: the output layer of the hypernet must be at least as
    large as the target's flat parameter count, so hypernets do not
    scale to very large target DNNs without chunked or component-wise
    weight generation.
  - *Numerical stability*: vanishing/exploding gradients in the combined
    hypernet+target system; gradient clipping, spectral norm, careful
    inits all help.
  - *Theoretical understanding*: limited so far — Galanti & Wolf 2020
    (Paper 2) is one of the few principled results; infinitely-wide
    hypernets do not converge to a global minimum under gradient
    descent (Littwin et al. 2020).
  - *Uncertainty-aware DL*: hypernets are a promising route but UQ via
    hypernet ensembles is still an active area.
  - *Interpretability*: how to visualise / explain the generated weights
    is largely unsolved.
  - *Usage guidelines*: practitioners lack systematic comparisons of the
    five-axis design choices for specific problem settings.

### Phase 1 — Foundational synthesis

This is a survey paper, so the right way to read it is as a *map*
through the rest of the literature. The authors take roughly seven years
of hypernetwork research and organise it on five axes:

1. *What goes in?* The hypernetwork takes a context vector. That
   context can be (a) a discrete *task ID* (typed as a learned
   embedding), (b) the actual *input data* itself (so the hypernet
   adapts the policy/classifier to each input), or (c) *noise* sampled
   from a prior (used for uncertainty estimation).
2. *What comes out, and how?* The hypernet has to produce a flat vector
   of target weights, but the target network typically has millions of
   weights. Four strategies trade off scalability against expressivity:
   (a) generate everything in one shot; (b) split the output across
   multiple heads; (c) generate one chunk at a time, conditioned on a
   chunk-ID embedding; (d) generate one component (layer / channel) at
   a time.
3. *Are the inputs static or dynamic?* Static = fixed embedding per
   task; dynamic = input-dependent. Static is most common in continual
   learning and multi-task RL; dynamic is the basis of data-adaptive
   models.
4. *Are the outputs static or dynamic?* Static = target architecture is
   fixed; dynamic = target architecture changes (RNN unrolled to
   variable length, NAS-generated graphs).
5. *What architecture is the hypernet?* MLP, CNN, RNN, or attention.
   The choice usually mirrors the structure of the input context.

The survey's Table 2 then plots 50 papers in this 5-axis grid. The most
useful part of the paper for a researcher entering the field is exactly
this table — it shows which combinations of axes have been tried, where
the gaps are, and which methods are closest to a given application.

The survey also makes the field's *open problems* explicit:

- *Initialisation is brittle*. Chang et al. 2020 (HFI), Beck et al. 2023
  (Bias-HyperInit, Paper 6) are partial answers; classical inits fail.
- *Scalability* is a real concern: a hypernet's output layer is big.
  Multi-head, chunk-wise, and component-wise generation are the
  workarounds, but choosing among them is empirical art.
- *Theory* is limited. Galanti & Wolf 2020 (Paper 2) is one of the
  rare formal results; the dynamics of training, the geometry of the
  generated-weight manifold, and the convergence of infinitely-wide
  hypernets are mostly open.

For the project, the survey is most useful as a *retrieval index*. If
the project ever needs federated learning, causal inference, or
uncertainty quantification — all areas with strong existing hypernet
results — Table 2 points to the right starting paper. The five-axis
taxonomy also makes the project's own design choices explicit:

- *Input axis*: the planned hypernet would be either *task-conditioned*
  (one ID per environment configuration) or *data-conditioned* (the
  precision signal is part of the input). The choice between these is a
  modelling question worth surfacing to the
  `professor-bayesian-brain` and `professor-neuromodulation` agents.
- *Output axis*: most likely *generate-multiple* (separate heads for
  actor and critic) given the typical PPO/Dreamer architecture. Generate-
  once would be infeasible for a Dreamer-scale target.
- *Input variability*: dynamic (precision is a continuous,
  within-episode signal), so the hypernet *must* be input-dependent
  rather than just task-ID-keyed. This rules out the simplest HN-PPO
  template (Paper 5) and points toward the dynamic hypernets of
  Paper 1 / Paper 3 / Paper 7.

### Phase 2 — Graduate-level deep dive

#### Formal HyperDNN definition (Section 2)

Given task $T$, dataset $\{X, Y\}$:

$$
\mathrm{DNN:}\quad \min_\Theta \mathcal L\big(Y, F(X; \Theta)\big),
\qquad
\mathrm{HyperDNN:}\quad \min_\Phi \mathcal L\big(Y, F(X; H(C; \Phi))\big).
$$

In a HyperDNN, only $\Phi$ is learnable; $\Theta = H(C; \Phi)$ is
*generated*. The gradient flows are: in DNN, $\nabla_\Theta \mathcal L$
updates $\Theta$; in HyperDNN, $\nabla_\Phi \mathcal L$ updates $\Phi$
via the chain rule

$$
\nabla_\Phi \mathcal L = \frac{\partial \mathcal L}{\partial \Theta} \frac{\partial \Theta}{\partial \Phi}.
$$

The Jacobian $\partial \Theta / \partial \Phi$ is the new object — its
conditioning controls the hypernet's training stability.

#### Five-axis taxonomy (Section 3)

| Axis | Choices |
|---|---|
| 1. Input | task-conditioned, data-conditioned, noise-conditioned |
| 2. Output (weight generation) | generate-once, generate-multiple (multi-head), chunk-wise, component-wise |
| 3. Input variability | static (fixed task embeddings), dynamic (input-dependent) |
| 4. Output variability | static target architecture, dynamic |
| 5. Hypernet architecture | MLP, CNN, RNN, attention |

The taxonomy is somewhat redundant — input/output variability is a
"super-categorisation" over input/output axes — but it makes the design
space explicit. Table 1 in the paper compares the four weight-generation
strategies on cost vs. coverage:
- *Generate-once* uses all weights, lowest input complexity, *highest*
  output complexity (output layer is enormous).
- *Generate-component-wise* needs one embedding per component, may leave
  some weights unused, lower output complexity.
- *Generate-chunk-wise* is the most modular — chunks of fixed size, with
  per-chunk embeddings. Lowest output complexity, highest input
  complexity (more embeddings to learn).
- *Generate-multiple* (multi-head) is orthogonal — splits the output
  across multiple heads, complementing any of the above.

#### Why classical initialisation fails (Section 6, "Initialisation")

The fundamental issue: the hypernet's output layer produces flat target
weights, but the variance of those weights must respect the target's
*per-layer* fan-in / fan-out structure. A single output layer with a
single output variance cannot satisfy multiple layer-specific
constraints simultaneously. Chang et al. (2020) HFI/HFO is the
principled fix for Kaiming-style targets. Bias-HyperInit (Beck et al.
2023, Paper 6) sidesteps the problem entirely by zeroing the
output-layer weights and letting the bias carry the base initialisation.

#### Scalability ceiling (Section 6, "Complexity")

For a target DNN with $n$ flat parameters, the hypernet's penultimate
layer of size $m$ produces an output layer of size $m \times n$. For a
ResNet-50-scale target, $n \approx 2.5 \times 10^7$, so even a small
hypernet with $m = 128$ has a $3.2 \times 10^9$-parameter output
layer — infeasible. This is why most hypernet applications use either
small targets (small MLPs in RL) or chunk/component-wise generation.

#### Theoretical landscape (Section 6, "Theoretical Understanding")

The paper notes only two formal results in the field:

1. **Galanti & Wolf 2020 (Paper 2)**: hypernet's primary network can be
   asymptotically smaller than embedding-method's primary network for
   smooth target families.
2. **Littwin et al. 2020**: infinitely wide hypernetworks do *not*
   converge to a global minimum under SGD, but convexity can be
   recovered by *increasing the dimensionality of the hypernet's
   output*.

Beyond these, the field has empirical guidelines but no convergence
theory, no PAC-Bayes bounds, no kernel-regime characterisation.

#### Uncertainty-aware deep learning (Section 6)

Three flavours of UQ via hypernet:
- *Bayesian Hypernets* (Krueger et al. 2017): noise-conditioned,
  $\Theta \sim H(\epsilon)$ with $\epsilon \sim p(\epsilon)$ — produces a
  distribution over target weights at inference time.
- *Hypernet+dropout* (Chauhan et al. 2023 — same authors): use dropout
  inside the hypernet to get an ensemble for free.
- *BayesianHMAML* (Borycki et al. 2022, Paper 4): hypernet outputs
  posterior parameters $(\mu, \sigma)$ or flow-conditioning vector.

### Relevance to interoceptive-pain RL

- **Locating the project's hypernet on the taxonomy.** A precision-
  conditioned hypernet for an interoceptive RL agent is:
  *(input)* data-conditioned (precision signal is part of input);
  *(output)* generate-multiple (separate actor / critic / world-model
  heads); *(input variability)* dynamic (precision changes within
  episode); *(output variability)* static target architecture;
  *(architecture)* RNN or attention (to integrate temporal precision
  context). This combination is empirically rare — Table 2 shows
  "data-conditioned + dynamic-input + RNN" is the slot occupied
  primarily by Paper 1's HyperRNN, with little RL-specific work.
- **The five-axis taxonomy is a direct prompt for `senior-developer`'s
  `issue_plan`**. Any plan to add a hypernet baseline must explicitly
  pick one option per axis, and the choice has implications for both
  scalability (output-axis) and biological plausibility (input
  variability + architecture).
- **Open problem alignment.** The project's interoceptive-RL agenda
  intersects with Chauhan et al.'s open problem of *uncertainty-aware
  DL* (precision = uncertainty) and *interpretability* (the project
  needs to read off the generated weights and tie them to interoceptive
  features). Both are listed as open and active.
- **The survey itself does not introduce new methods or empirical
  results.** All methodological content is in Papers 1–7. The survey's
  value is the *index*: when the project's literature questions broaden
  beyond hypernets-for-RL, Table 2 is the right starting point.

---

## <a id="synthesis"></a>Cross-paper synthesis & relevance to interoceptive-pain RL

This synthesis is intentionally short. Detailed per-paper relevance
sections appear above; here we draw together the design moves that
*recur* across the eight papers and what they imply for the project's
FiLM-vs-hypernet comparison.

### Five recurring design choices

1. **Static vs. dynamic conditioning.** Paper 1 introduces both: a
   *static* hypernet uses a learned per-layer or per-task embedding;
   a *dynamic* hypernet has the embedding produced by a recurrent or
   feedforward network from the *current input or history*. Papers 5–6
   are essentially static (task ID); Paper 7 is static at episode-start
   but uses a continuous task descriptor; Paper 3 is dynamic (recurrent
   or top-down hypernet); Paper 1 covers both. For an interoceptive
   precision signal that varies *within an episode*, only the dynamic
   variant fits.

2. **What is generated.** The hypernet output can be (a) full target
   weights (Papers 1 static, 5, 6, 7); (b) a low-dimensional mixture
   vector over a fixed basis (Paper 3); (c) a row-scaling vector
   (Paper 1 dynamic, structurally identical to FiLM); (d) posterior
   parameters $(\mu, \sigma)$ rather than weights themselves (Paper 4).
   Galanti & Wolf 2020 (Paper 2) argues theoretically that *full*
   weight generation gives the best parameter efficiency for smooth
   target families, but the row-scaling and mixture variants are more
   biologically plausible and easier to train.

3. **What conditioning is on.** Three choices in the survey (Paper 8):
   *task* (discrete ID), *data* (input-dependent), *noise* (for
   uncertainty). Hybrid forms exist — Paper 4 conditions on a *support
   set* of labelled examples; Paper 7 conditions on a *task descriptor*
   $(\psi, \mu)$ that is a continuous summary of the MDP.

4. **Anti-forgetting / regularisation mechanism.** Two recurring
   patterns: (a) *output anchoring* on previous-task embeddings (Paper
   5, von Oswald et al. 2020 lineage) — the hypernet output at fixed
   anchors is L2-pinned across training; (b) *Bellman / TD consistency*
   on generated policy-critic pairs (Paper 7) — supervised generation
   plus an RL-style structural regulariser. These are largely
   independent and can be combined.

5. **Initialisation.** Almost universally fragile (Papers 6, 8). The
   project must use either Hyperfan-In (Chang et al. 2020) or
   Bias-HyperInit (Beck et al. 2023, Paper 6) — naïve Kaiming /
   Orthogonal init silently breaks hypernet training in RL.

### Specific implications for the FiLM-vs-hypernet comparison

The project is comparing FiLM-style affine modulation against
hypernet-style weight generation as the substrate for context- /
precision-dependent control. The literature gives a clear shape to the
empirical question.

- **FiLM is a special case of a hypernet (Paper 1, Paper 2).** Paper 1's
  memory-efficient HyperRNN has the row-scaling form
  $\gamma \odot W h + \beta$, which is FiLM with a recurrently-generated
  $(\gamma, \beta)$. Paper 2's modularity argument shows formally that
  FiLM (an "embedding method that scales the first layer") is dominated
  in the worst-case parameter complexity by a hypernet that generates
  *all* layers' weights.
- **Modularity advantage is asymptotic, not pointwise.** Papers 6 and 7
  empirically confirm that on *easy* benchmarks FiLM-style and hypernet
  perform similarly, while on *hard* benchmarks (high-dimensional task
  variation, out-of-training-distribution test) the hypernet wins by a
  wide margin (9× on Meta-World Pick-Place; consistent gains on DM
  Control with $\psi, \mu$ variation).
- **Init must be controlled.** Paper 6 explicitly shows that
  Bias-HyperInit improves *both* the hypernet variant *and* the FiLM
  variant. A fair project comparison must apply Bias-HyperInit to both
  arms; otherwise the hypernet arm will look spuriously bad.
- **Regulariser asymmetry.** Hypernet variants benefit from output-
  anchoring (Paper 5) and Bellman-consistency (Paper 7) regularisers;
  these have no clean FiLM analogue because FiLM does not produce a
  "weight anchor" point. This is a pro-hypernet point that is rarely
  flagged in the FiLM literature.
- **Bayesian extension is asymmetric.** Paper 4's BayesianHMAML
  generalises straightforwardly to "hypernet outputs $(\mu, \sigma)$
  for a Gaussian over weights"; the analogous extension for FiLM —
  Gaussian over $(\gamma, \beta)$ — is much shallower because the
  posterior is over a small affine modulation, not the full weight
  space. The project's interest in *precision = uncertainty* points to
  the Bayesian-hypernet route as the more natural fit.

### Cross-references to the project's existing master reviews

- **Predictive coding (`docs/project/foraging_for_cognitive_evolution_lit_review.md`,
  `docs/project/computational_pain_models_lit_review.md`).** Paper 3 is
  the direct architectural template for hypernet-modulated predictive
  coding and is the *only* paper in this corpus that explicitly couples
  hypernets with predictive coding.
- **Dreamer (`docs/project/dreamer_lit_review.md`).** A hypernet-modulated
  RSSM is a direct extension of Paper 3's mixture-of-V transition
  model: replace Dreamer-V3's single recurrent transition with a
  $K$-mixture transition whose weights are generated by an interoceptive
  hypernet. This is consistent with the World-Models temperature-
  modulation idea in the Dreamer review.
- **FiLM corpus (`docs/project/references/FiLM/`, no master review yet).**
  Papers 1, 2, 6 give the most direct hypernet–FiLM bridges (row-scaling
  equivalence, modularity bound, shared init pathology). Recommend a
  short FiLM master review by `literature-reviewer` to make this
  comparison tractable.

### Recommended next steps (handoffs)

- **For `senior-developer`:** an `issue_plan` for a *minimal* hypernet
  baseline. Architecture: dynamic hypernet conditioned on the current
  precision signal, generate-multiple over actor/critic, Bias-HyperInit
  init (Paper 6), output-anchor regulariser if the project includes a
  multi-context curriculum (Paper 5). Recommend Paper 3's mixture-of-V
  variant as the *first* design rather than full-weight generation, on
  biological-plausibility grounds.
- **For `professor-bayesian-brain`:** a concept memo on the
  predictive-coding interpretation of Paper 3's mixture-of-V. The slow-
  fast emergence in Paper 3 is the cleanest published version of the
  cortical-timescale hierarchy that the project is implicitly modelling.
- **For `professor-neuromodulation`:** a critique on the
  neural-plausibility implications of full-weight generation (Paper 1
  static, Papers 5–7) vs. mixture-vector generation (Paper 3). Paper 3
  explicitly cites context-dependent population-dynamics neuroscience
  (Mante et al. 2013; Stroud et al. 2018) as motivation.
- **For `experiment-designer`:** if the project ever runs a *bank of
  per-context optimal agents* (e.g. one RPPO per food/predator
  configuration) and wants to distil them into a single
  precision-conditioned policy, Paper 7's HyperZero framework + TD
  regulariser is the right starting recipe.
