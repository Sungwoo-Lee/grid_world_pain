---
title: "Modulation Granularity and Placement — Cross-Paper Synthesis (FiLM + Hypernetwork corpora)"
topic: FiLM
status: curated
created: 2026-07-28
last_updated: 2026-07-28
related:
  - film_lit_review.md
  - film_synthesis.md
  - film_neuromod_integration_synthesis.md
  - ../Hypernetwork/hypernetwork_lit_review.md
  - archive/film_conditional_modulation_review.md
scope: |
  Cross-paper synthesis over the 23-paper FiLM corpus plus the 8-paper
  Hypernetwork corpus, answering a single architectural question: at what
  granularity, and at which sites, does the conditional-architecture
  literature actually apply its modulation parameters? Organised along four
  axes (granularity of the signal; placement / number of sites;
  parameterisation across sites; grouping and parameter-sharing schemes),
  with a verdict positioning this project's neuromodulator against the
  field's convention.
---

# Modulation Granularity and Placement — Cross-Paper Synthesis

## 1. Plain-English entry point

Many neural networks are built so that one signal can *steer* another network's
computation. The most common way to do this is called **FiLM** — "feature-wise
linear modulation" — which means: take an intermediate layer's activity, multiply
it by a number called **gamma** (a gain) and add a number called **beta** (an
offset). A small side-network reads the steering signal and produces those numbers.

This document asks two mechanical questions the reviews in this corpus answer only
piecemeal. **How many distinct gain/offset numbers exist?** One per individual
neuron? One per feature channel? One shared by a block of neurons? One for a whole
layer? One for the whole network? And **where are they applied** — at every layer,
at one layer, early or late?

The answer from 31 papers is remarkably uniform. The field's convention is: **one
gain and one offset per feature channel** (which, in a plain fully-connected layer,
means one per hidden unit), **applied at many layers throughout the network**, with
**one shared generator network feeding a separate small output head per site**. Two
papers sweep *where* to inject and both find "inject at many depths, including early
ones" wins. Almost nobody makes the modulation coarser than per-channel — the
literature's pressure runs the other way, toward *finer* control (one number per
individual weight).

This project's agent sits partly inside and partly outside that convention. Our
shared-generator-plus-per-site-heads layout is textbook. Our decision to make the
modulation *coarser* by tying blocks of neurons together — the `grouping_size` sweep
— has exactly **one** precedent anywhere in the reference library, and it is not in
the FiLM or hypernetwork literature at all; it is in a 2025 spiking-neural-network
paper from the neuromodulation corpus. That paper is the one our code cites, and our
implementation matches its scheme. Details, tables, and four explicit judgments
follow.

---

## 2. The taxonomy used here

These labels are shared with a parallel synthesis running over the
`neuromodulatory_algorithms` corpus, so results are directly comparable.

**Axis 1 — Granularity of the modulation signal.** How many distinct
$(\gamma, \beta)$ values exist per modulated tensor.

| Cell | Name | Definition |
|---|---|---|
| **(a)** | Node-wise / per-unit (element-wise) | one $(\gamma,\beta)$ per individual neuron / activation |
| **(b)** | Per-channel | one $(\gamma,\beta)$ per conv feature map, broadcast over spatial positions — the canonical FiLM |
| **(c)** | Grouped | one $(\gamma,\beta)$ shared across a *group* of units/channels; anything strictly between (a)/(b) and (e) |
| **(d)** | Layer-scalar | a single $(\gamma,\beta)$ for an entire layer |
| **(e)** | Network-wise / global | one modulation signal for the whole network |

**Axis 2 — Placement / number of sites.** Which layers are modulated, and how many
injection points exist in total.

**Axis 3 — Parameterisation across sites.** Whether each modulated layer gets its
own parameter set, and whether the generator is shared.

**Axis 4 — Grouping / parameter-sharing / low-rank schemes**, plus any paper that
empirically *ablates granularity*.

### 2.1 A load-bearing caveat: "per-channel" collapses in dense layers

FiLM's canonical cell (b) is defined on a convolutional feature map
$F_{i,c} \in \mathbb{R}^{H\times W}$: one scalar $\gamma_{i,c}$ multiplies an entire
2-D map, so the modulation is *spatially uniform but channel-specific*
([Perez 2018 review §Phase 2](reviews/perez_2018_film.md)). The "sharing" in
per-channel FiLM is sharing **across spatial positions**, not across features.

In a fully-connected layer of width $N$ there are no spatial positions. A "channel"
*is* a single hidden unit. **Cells (a) and (b) therefore coincide in a dense
network.** Any claim of the form "the field uses per-channel, we use per-neuron, so
we differ" is a category error: for our 128-unit dense layers, per-channel FiLM *is*
per-neuron FiLM. What genuinely distinguishes our design is cell **(c)**, grouping —
which has no spatial analogue in the conv literature at all.

### 2.2 A second caveat: "layer-wise" is ambiguous

Papers say "layer-wise modulation" to mean two different things:

- **Axis 3 sense** — "each layer has its own distinct parameter set" (the
  overwhelmingly common meaning; e.g. Perez's four ResBlocks each with their own
  $W^n$).
- **Axis 1d sense** — "one scalar per layer" (a genuinely coarse signal).

Every paper in these two corpora that says "layer-wise" means the **Axis 3** sense.
Not one paper in either corpus uses an Axis 1d layer-scalar modulation signal. The
tables below always state which sense applies.

---

## 3. Axis 1 — Granularity, per paper

### 3.1 FiLM corpus

| Paper | Granularity cell | What the $(\gamma,\beta)$ index runs over | Host layer type | Notes / evidence |
|---|---|---|---|---|
| [Perez et al. 2018 — FiLM](reviews/perez_2018_film.md) | **(b)** | 128 conv channels per ResBlock | conv | $\gamma,\beta$ are scalars, not maps; parameter cost $2C$, independent of $H,W$ |
| De Vries et al. 2017 — CBN / MODERN ([archive §2](archive/film_conditional_modulation_review.md)) | **(b)** | every BN channel of ResNet-50, $\sum_\ell C_\ell \approx 2\times10^4$ | conv (post-BN) | residual-offset form $\gamma_c(z)=\gamma_c+\Delta\gamma_c(z)$ |
| [Dumoulin et al. 2017 — CIN](reviews/dumoulin_2017_cond_instance_norm.md) | **(b)** | an $N_{\text{styles}} \times C$ lookup table | conv (post-IN) | 0.2 % of parameters are style-specific |
| [Huang & Belongie 2017 — AdaIN](reviews/huang_belongie_2017_adain.md) | **(b)** | per-channel mean/std of a style image's deep features | conv (post-IN) | zero learnable parameters in the modulation layer |
| [Birnbaum et al. 2019 — TFiLM](reviews/birnbaum_2019_temporal_film.md) | **(b)** in features, **grouped in time** | per-channel × per time-block | 1-D conv | the corpus's only *grouping* hyperparameter — see §6.1 |
| [Wisnu et al. 2025 — STSM-FiLM](reviews/wisnu_2025_stsm_film.md) | **(b)** | per-channel; conditioner is a single scalar $\alpha$ | conv / decoder | scalar *input*, per-channel *output* |
| [Takeda et al. 2021](reviews/takeda_2021_multi_task_feature_mod.md) | **(b)** | per-channel at every conv layer but the last | conv (post-IN) | < 2 % task-specific parameters over 6 tasks |
| [Turkoglu et al. 2022 — FiLM-Ensemble](reviews/turkoglu_2022_film_ensemble.md) | **(b)** | per-channel × per-BN-layer × per-member $m$ | conv (post-BN) | $\Delta P = (M-1)\cdot 2 \sum_n D_n$ |
| [Jang et al. 2022 — BC-Z](reviews/jang_2022_bcz.md) | **(b)** | per-channel × 4 ResNet blocks | conv | $2C_k \times (512{+}1)$ params per block |
| [Moon et al. 2023](reviews/moon_2023_hierarchical_achievements.md) | **(b)** | per-channel of a CNN state embedding | conv embedding | a single fusion site |
| [Nikulin et al. 2023 — SAC-RND](reviews/nikulin_2023_anti_exploration_rnd.md) | **(a) ≡ (b)** | per hidden unit, width 256 | **dense MLP** | the corpus's cleanest dense-layer FiLM; see §7 judgment 4 |
| [Yan & Guo 2025 — CaFiLM](reviews/yan_guo_2025_context_aware_dg.md) | **(b)** signal, **tied** generator | per-channel $(\gamma_c,\beta_c)$ from a 2×2 matrix shared over all $C$ | dense feature vector | 6 total trainable parameters — an Axis-4 tying scheme, not a coarsening |
| [Abdollahzadeh et al. 2021 — KML](reviews/abdollahzadeh_2021_multimodal_meta.md) | **finer than (a)** | one modulation entry per *weight* of the conv kernel | conv weights | argues explicitly that (b) is *too coarse* |
| [Gorishniy et al. 2025 — TabM / BatchEnsemble](reviews/gorishniy_2025_tabm.md) | **(a)** | $r_i$ per input unit, $s_i, b_i$ per output unit, per member | **dense MLP weights** | $W \odot (s_i r_i^{\top})$ — rank-1 per member |
| [Shazeer et al. 2017 — Sparse MoE](reviews/shazeer_2017_sparse_moe.md) | *not on this axis* | expert-level discrete routing | FFN block | different kind, not different degree |
| [Andreas 2016](reviews/andreas_2016_neural_module_networks.md) / [Hu 2017](reviews/hu_2017_e2e_module_networks.md) | *not on this axis* | module-level composition | whole subnetworks | — |
| [Vaswani et al. 2017](reviews/vaswani_2017_attention.md) | *orthogonal* | modulates *positions*, not channels | attention | see [`film_synthesis.md` Cluster E](film_synthesis.md#cluster-e--attention-as-a-parallel-conditioning-primitive) |
| [Kendall & Gal 2017](reviews/kendall_gal_2017_uncertainties.md), [Gawlikowski 2023](reviews/gawlikowski_2023_uncertainty_survey.md), [Santurkar 2018](reviews/santurkar_2018_batchnorm_optimization.md) | n/a | not modulation-placement papers | — | cited for loss / taxonomy / optimisation only |

### 3.2 Hypernetwork corpus

| Paper | Granularity cell | What is generated | Notes |
|---|---|---|---|
| Ha et al. 2016 — static HyperNet ([review](reviews/ha_2016_hypernetworks.md)) | full-weight | entire conv kernel $K^j = g(z^j)$ from a per-layer embedding | granularity axis does not bind: everything is generated |
| Ha et al. 2016 — **HyperRNN scaling trick** | **(a)** | $d_h(z_h) \in \mathbb{R}^{N_h}$, one scalar per *row* of the recurrent matrix | per-row = per-hidden-unit; the review calls this "structurally identical to FiLM at per-row granularity" |
| Jiang et al. 2021 — Dynamic Predictive Coding | **(e)** | a $K$-simplex mixture vector, $K=5$, over a fixed bank of basis transition matrices | the only near-global modulation signal in either corpus |
| Beck et al. 2023 — Hypernets in meta-RL | full-weight (hypernet) vs **(a)** (FiLM baseline) | all policy parameters, vs $(\gamma,\beta)$ on the policy MLP | direct head-to-head; see §7 judgment 2 |
| Schöpf et al. 2022 — HN-PPO | full-weight | flat actor (+critic) parameter vector from an 8-dim task embedding | — |
| Borycki et al. 2022 — BayesianHMAML | full-weight, **one site** | only the classification head $\theta_H$; encoder shared | "generating the entire encoder would blow up the hypernet output dimension" |
| Rezaei-Shoshtari et al. 2023 | full-weight | entire actor and critic networks | — |
| Galanti & Wolf 2020 | theory | — | proves the parameter-efficiency gap; silent on granularity |
| Chauhan et al. 2023 — survey | taxonomy | generate-once / multi-head / chunk-wise / component-wise | the field's only explicit statement of an Axis-3/4 design space |

### 3.3 Verdict on Axis 1

**The field's convention is cell (b) — per-channel — and in dense layers that is
cell (a), per-unit.** 14 of the 15 FiLM-corpus papers that apply modulation at all
use it. The single deviation is Abdollahzadeh's KML, which moves *finer* (per-weight)
and argues the per-channel cell is too coarse for heterogeneous task mixtures.

**Cell (c) — grouped — is unattested in both corpora.** A regex scan of all 31 source
PDFs for grouping language (`group(ed|ing) of channels/neurons/units`, `channel group`,
`group size`, `granularity`, `shared across channels`, `per-neuron`) returned exactly
two hits, both irrelevant: a bibliography entry for Wu & He's GroupNorm in Santurkar's
reference list, and the word "granularity" used about image resolution in AdaIN's
related-work section. No FiLM or hypernetwork paper in this library ties a contiguous
block of channels to one shared $(\gamma,\beta)$.

**Cells (d) and (e) are essentially unattested too.** No paper uses a layer-scalar
signal. The closest thing to a global signal anywhere is Jiang et al. 2021's 5-dim
mixture vector over basis transition matrices — a whole-network low-dimensional
control signal, but structurally a mixture-of-bases, not an affine gain.

---

## 4. Axis 2 — Placement and number of sites

| Paper | Sites | Where | Relative to normalisation | Sweep / ablation? |
|---|---|---|---|---|
| Perez et al. 2018 | **4** | one per FiLM-ed ResBlock | after BN-without-affine | **Yes** — moving FiLM within the block, including after the post-ReLU, barely changes accuracy (97.7 %); removing BN entirely costs ~4 pp (97.4 → 93.7) |
| De Vries et al. 2017 | **all BN layers** of ResNet-50 | every residual stage | *is* the BN affine | **Yes, the key result** — stage-4 only → 3+4 → 2+3+4 → all four; monotonic improvement, largest marginal gain from *adding stage 2*. The corpus's primary "early modulation helps" evidence |
| Dumoulin et al. 2017 | every IN layer | throughout the transfer network | *is* the IN affine | no |
| Huang & Belongie 2017 | **1** | a single bottleneck between VGG encoder and decoder | *is* the IN step | **Yes, indirectly** — adding BN or IN *in the decoder* hurts badly (it re-normalises away the injected style) |
| Birnbaum et al. 2019 | after **each conv layer** in a symmetric encoder–decoder | every down/up block | standalone | block-length $B$ swept as a hyperparameter |
| Wisnu et al. 2025 | **multiple decoder layers** (HiFi-GAN variants); **1** for EnCodec (between encoder and quantizer) | decoder | standalone | no |
| Takeda et al. 2021 | **every conv layer except the last** | encoder *and* decoder | after IN | **Yes** — `EncIn` (FiLM in decoder only) is consistently worse. Conclusion: "IN+FiLM should be applied to every conv layer except the last" |
| Turkoglu et al. 2022 | **every BN layer** | throughout the backbone | *is* the BN affine | no layer-subset ablation |
| Jang et al. 2022 (BC-Z) | **4** | each ResNet-18 block | standalone | no ablation; design follows Perez's "inject throughout the depth" prescription |
| Moon et al. 2023 | **1** | inside an auxiliary contrastive head (action → state embedding) | standalone | no |
| Nikulin et al. 2023 | **1** by default (penultimate layer, pre-nonlinearity) of a 4-layer MLP | RND prior; predictor uses bilinear at the first layer | standalone | **Yes** — `film_first` / `film_last` / `film_full` swept. **All layers wins for the prior; last layer only wins for the predictor.** The authors flag it as domain-dependent |
| Yan & Guo 2025 | **1** | between frozen feature extractor and frozen classifier | standalone | no |
| Abdollahzadeh et al. 2021 | **all 4 conv layers** of the base network | on the kernels | standalone | rank sweep (1 vs 2 vs full), not a site sweep |
| Gorishniy et al. 2025 | **every linear layer** | around $W$ | n/a (tabular MLPs have no norm layers) | ablates which of the three adapters $R,S,B$ to keep |

### 4.1 Verdict on Axis 2

**The convention is "many sites, spread through the depth, including early ones."**
Both papers that actually sweep depth agree on the direction: De Vries's stage-wise
sweep (late-only → all stages, monotonic gain, biggest jump from adding an *early*
stage) and Nikulin's `film_first/last/full` sweep (all layers best for the prior).
Takeda's `EncIn` ablation is a third, weaker vote in the same direction.

Two dissents worth naming. **AdaIN uses exactly one site**, and *adding* modulation
downstream actively hurts, because downstream normalisation destroys the injected
statistics. **Nikulin's predictor** prefers a single late site. Both dissents share a
structural cause: when a *downstream consumer is frozen or must be left alone*, more
injection sites are worse, not better. Yan & Guo's CASA (one site, frozen classifier
downstream) is a third instance of the same pattern.

**Relative to normalisation, placement is nearly free.** Perez's within-block sweep
shows FiLM works before or after the ReLU and with BN removed entirely (−4 pp). This
is the licence for FiLM in LayerNorm-based recurrent RL backbones like ours.

---

## 5. Axis 3 — Parameterisation across sites

The recurring pattern, stated precisely, is: **one shared conditioner network whose
output is projected by a separate small linear head per modulated site.**

| Paper | Generator | Per-site parameters | Form |
|---|---|---|---|
| Perez et al. 2018 | **one** 4096-unit GRU over the question | **yes** — one $W^n \in \mathbb{R}^{2C \times d_z}$ per ResBlock $n$ | $[\gamma^n;\beta^n] = W^n g_\phi(x) + b^n$ |
| De Vries et al. 2017 | **one** MLP over the LSTM's language embedding | **yes** — but as one wide output of size $2\sum_\ell C_\ell$, sliced per layer | $[\Delta\gamma;\Delta\beta] = W^{(2)}\tanh(W^{(1)}z + b^{(1)}) + b^{(2)}$ |
| Dumoulin et al. 2017 | none (a lookup table) | **yes** — one $(\gamma,\beta)$ row per style *per layer* | table indexed by style |
| Huang & Belongie 2017 | frozen VGG + per-channel mean/std | n/a (one site) | parameter-free |
| Birnbaum et al. 2019 | **one LSTM per TFiLM layer**, over that layer's own block-pooled activations | **yes** — each site has its own generator | self-conditioning, so a shared generator would be meaningless |
| Jang et al. 2022 (BC-Z) | **one** frozen sentence encoder → 512-dim $z$ | **yes** — $W_\gamma^{(k)}, W_\beta^{(k)}$ per ResNet block | $2C_k \times 513$ per block |
| Turkoglu et al. 2022 | none (per-member lookup) | **yes** — $\gamma^m_n, \beta^m_n$ per layer $n$ per member $m$ | stored directly as parameters |
| Nikulin et al. 2023 | **one** linear layer of double hidden width, split into $\gamma$ and $\beta$ halves | n/a (one site by default) | $2\times256$ output split |
| Yan & Guo 2025 | **one** 2×2 matrix, **shared across channels and sites** | **no** | the corpus's extreme tying case |
| Ha et al. 2016 (HyperLSTM) | **one** HyperLSTM cell | **yes** — separate $z_h, z_x, z_b$ projections **per LSTM gate** $\{i,g,f,o\}$ | the closest structural analogue to our six heads |
| Abdollahzadeh et al. 2021 | **one** task encoder $h_\phi$ | **yes** — three small MLPs $g^l_{\phi_1}, g^l_{\phi_2}, g^l_{\phi_3}$ per layer $l$ | outer-product factorised |
| Chauhan et al. 2023 (survey) | — | names this design choice **"generate-multiple (multi-head)"** and calls it *orthogonal* to the other three output strategies | — |

### 5.1 Verdict on Axis 3

**"One shared generator + a separate head per site" is the field's default, by a wide
margin.** Perez, De Vries, BC-Z, Abdollahzadeh, and Ha's HyperLSTM all use it; the
hypernetwork survey names it as a first-class taxonomy option (`generate-multiple`).
The only clear exceptions are (i) TFiLM, whose generator is *self*-conditioned so it
must be per-site, and (ii) CaFiLM, whose whole point is extreme tying (6 parameters
total).

Note the sub-variant. Perez uses a distinct weight matrix $W^n$ per site; De Vries
uses a single wide output vector that is *sliced* per site. Functionally equivalent;
Perez's form is what our code implements.

---

## 6. Axis 4 — Grouping, tying, and low-rank schemes

### 6.1 Every parameter-sharing scheme in the two corpora

| Paper | Sharing unit | How the size is chosen | Stated motivation | Empirical finding |
|---|---|---|---|---|
| **Birnbaum et al. 2019 (TFiLM)** | **contiguous blocks along *time*** — one $(\gamma_b,\beta_b)$ per block of $T/B$ samples | hyperparameter; ~0.1 s ≈ a phoneme, giving $B\approx 50$ blocks for an 80 000-sample clip | **compute**: an LSTM over 50 pooled blocks instead of 80 000 samples (1600× fewer RNN steps); the review also names the trade-off as modulation *granularity* | overhead < 1 %; block length explicitly flagged as trading compute against "granularity of the temporal modulation" |
| **Abdollahzadeh et al. 2021 (KML)** | **rank-1 factorisation** of a per-weight modulation matrix, $M^l = g^l_{\phi_1}(\upsilon) \otimes g^l_{\phi_2}(\upsilon)$ | rank fixed at 1; ranks 2+ tested in the appendix | **parameter efficiency** — ~150× smaller generator than a single MLP emitting $|W^l|$ outputs | rank-1 *beats* full-rank at meta-test; higher ranks do not help |
| **Yan & Guo 2025 (CaFiLM)** | **generator tied across all channels** — one $A \in \mathbb{R}^{2\times2}$, $b\in\mathbb{R}^2$ | fixed by construction | **capacity control / overfitting**: the adapter must not overwrite a frozen downstream classifier | 6 parameters reach DomainBed SOTA (68.8 % avg); replacing CaFiLM with a free MLP loses 1.1 pp |
| **Gorishniy et al. 2025 (BatchEnsemble in TabM)** | **rank-1 per-member adapter** $s_i r_i^\top$ around a shared $W$ | rank fixed at 1 | **parameter efficiency**, then found to also **regularise** | weight sharing "acts as regularisation" — `TabM_naive` (shared $W$) beats `TabM_packed` (no sharing) |
| **Ha et al. 2016 (static hypernet)** | **kernel tiling** — a basic 16×16 kernel replicated with its own embedding per tile | tile size fixed at 16 to match ResNet channel multiples | **scalability** of the hypernet output layer | costs 1.25–1.5 pp CIFAR-10 accuracy for a large parameter reduction |
| **Chauhan et al. 2023 (survey)** | **chunk-wise generation** — fixed-size chunks of the flat parameter vector, each with a chunk-ID embedding | "empirical art"; the survey declines to give a rule | **scalability**: the hypernet output layer must otherwise be as large as the target's parameter count | named "the most modular" strategy; lowest output complexity, highest input complexity |
| **Turkoglu et al. 2022** | none (per-member, per-layer, per-channel, all free) | — | — | diversity is instead controlled by the *initialisation gain* $\rho$ |

### 6.2 Papers that empirically ablate granularity

Only three, and none of them sweep our axis.

1. **Abdollahzadeh et al. 2021** — the only true *granularity* ablation in the FiLM
   corpus, and it runs **the opposite direction from ours**: per-channel (b) versus
   per-weight (finer). Per-weight wins by ~5 pp on 5-mode few-shot. The paper's
   mechanism claim is that per-channel FiLM leaves "one degree of freedom per channel
   per task", so heterogeneous tasks "fight for capacity". They also sweep the *rank*
   of the generator factorisation (1 vs 2 vs full) and find rank-1 best — a
   capacity-*reduction* result, but in the generator, not in the signal.
2. **Beck et al. 2023** — full-weight hypernet versus FiLM $(\gamma,\beta)$-only on the
   same meta-RL policy, both with Bias-HyperInit: **42.9 % vs 25.5 %** test success on
   Meta-World Pick-Place. Generating everything beats generating only the affine.
   Again: finer/richer wins.
3. **Nikulin et al. 2023** — sweeps conditioning *type* (concat / gating / bilinear /
   FiLM) and *depth* (first / last / all), not granularity. Notes that FiLM is
   formally "a special case of a bilinear layer with low-rank weight matrices" — so
   the FiLM-vs-bilinear axis is itself a rank/capacity axis, and the coarser (FiLM)
   option wins on cost while bilinear-full matches it on quality.

**Net reading:** every granularity comparison in these two corpora finds that *more*
modulation capacity performs at least as well as less, and usually better. The
corpora contain **no experiment in which coarsening the modulation signal improved
performance.** The one place coarsening wins is the *generator* (KML's rank-1
factorisation, TabM's shared $W$), not the signal.

---

## 7. Where this project sits — verdict

### 7.1 Our design, in taxonomy terms

Our modulator (`src/models/neuromodulator.py`, class `NeuromodulatorRNN`) is:

- **Generator:** one shared GRU cell of 16 hidden units, reading the observation
  vector. This is the "affective inertia" core.
- **Sites (Axis 2): six heads, five distinct injection points.** Gain $\gamma$ and
  offset $\beta$ for the unimodal sensory-encoder stage; $\gamma$ and $\beta$ for the
  multimodal-hub stage; a bias injected into the task GRU's update gate; and a scalar
  action temperature.
- **Parameterisation (Axis 3):** one shared generator, **six separate `nnx.Linear`
  heads**. Textbook `generate-multiple`.
- **Granularity (Axis 1):** cell **(c), grouped**. Each $\gamma$/$\beta$/memory head
  emits $\lceil 128/g \rceil$ raw values, expanded to the 128-unit target by
  `jnp.repeat(raw, g)[:128]` — contiguous blocks of $g$ neurons share one dynamic
  value. `grouping_size` $g$ is swept over $\{1,2,4,8,16,32,64,128\}$, so the sweep
  spans cell (a) at $g=1$ through cell (e) at $g=128$.
- **Decomposition:** a full 128-dim **per-neuron learned baseline** (`nnx.Param`,
  zero-initialised) is added to every expanded signal. So the *static* component is
  always per-neuron; only the *dynamic* component is grouped.

### 7.2 Does the cited paper match? AlKilany & Goodman 2025 — **yes, exactly**

The code comment cites "Branched output heads with spatial grouping (AlKilany &
Goodman, 2025)". That paper **is** in the reference library, in the sister corpus:
[`../neuromodulatory_algorithms/reviews/alkilany_goodman_2025_snn_dynamic_sensory.md`](../neuromodulatory_algorithms/reviews/alkilany_goodman_2025_snn_dynamic_sensory.md)
(bioRxiv 2025.07.25.666748, Goodman lab, Imperial College).

Their scheme, from the review's Phase 2 "Spatial grouping" section:

> Group size $G$: each modulator output is broadcast to a contiguous block of $G$
> neurons. With $G = 1$, every neuron is independently modulated; with $G$ large,
> modulation is shared across many cells.

**This is our `jnp.repeat(raw, grouping_size)` verbatim.** Their primary SNN has 200
hidden neurons and they sweep $G$ from 1 to 200 — the same 1-to-full-width ladder we
sweep from 1 to 128. Three further correspondences and one divergence:

| Element | AlKilany & Goodman 2025 | This project |
|---|---|---|
| Grouping mechanism | contiguous block broadcast, $G$ swept 1 → 200 | contiguous block broadcast, $g$ swept 1 → 128 |
| Generator | one small MLP (or SNN) reading recent hidden activity | one GRU (16 units) reading the observation |
| Multiple outputs | one output group per modulated biophysical parameter (threshold, reset, rest, $\tau_m$, $\tau_s$) | six heads (two $\gamma$, two $\beta$, memory bias, temperature) |
| Coupling rule | **substitution** ($p \leftarrow m$) or **addition** ($p \leftarrow p + m$) | addition onto a **learned per-neuron baseline** |
| Second grouping axis | **temporal** grouping $K$ — the modulator fires every $K$ steps | none; our modulator fires every step ($K=1$) |
| Empirical finding on $G$ | "performance roughly flat in $G$"; broad neuromodulation is *almost* as effective as targeted | our screen is testing whether coarsening *helps* (as a regulariser) |

Two honest caveats. First, their motivation for grouping is **biological plausibility
plus neuromorphic-hardware flexibility** — diffuse neuromodulator anatomy, and letting
chip designers trade granularity for wiring cost. Our motivation is
**regularisation**: per-neuron ($g{=}1$) modulation destabilised late training (see
[`NMN_FILM_GROUPING_SCREEN.md`](../../../experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md)).
Same mechanism, different rationale. Second, their reported result is that
performance is *flat* in $G$ — it does not predict that coarsening will *improve*
survival, only that it should not cost much. Our screen's pre-registered hypothesis is
stronger than anything their data supports.

### 7.3 The four judgments

**(1) Is a grouped granularity something the literature does at all — or is it
rare/novel?**

**Rare, and outside the conditional-architecture literature entirely.** Cell (c) does
not appear in any of the 23 FiLM-corpus or 8 Hypernetwork-corpus papers. A full-text
scan of all 31 PDFs for grouping vocabulary produced no substantive hit. The only
attestation in the entire reference library is AlKilany & Goodman 2025 in the
neuromodulation corpus — which our implementation matches exactly.

The nearest structural cousins in the FiLM corpus are all *different objects*:
TFiLM's contiguous **time**-blocks (grouping along the sequence axis, motivated by
compute); KML's and BatchEnsemble's **rank-1 factorisations** (grouping in the
*generator*, motivated by parameter count); and CaFiLM's **generator tied across
channels** (a shared 2×2 projection, so the signal stays per-channel while the
mapping is tied). None of these ties a *contiguous block of units* to one shared
dynamic value.

**Verdict: our sweep axis is genuinely unusual for a FiLM-family architecture.** That
is not automatically bad — it is a defensible, biologically-motivated, cheap
regulariser with one clean precedent — but it should be presented as a *transfer from
the spiking-neuromodulation literature into the FiLM family*, not as a standard FiLM
variant, and the paper framing should cite AlKilany & Goodman for it rather than
Perez.

**(2) Is "one shared generator + per-site heads" the norm, or do most papers use
per-layer parameter sets?**

**It is the norm, and the question contains a false dichotomy.** "Per-layer parameter
sets" and "shared generator with per-site heads" are the *same thing* in nearly every
paper: the generator is shared, the head is per-site, and the resulting
$(\gamma^n, \beta^n)$ *are* a per-layer parameter set. Perez (one GRU → per-block
$W^n$), De Vries (one MLP → sliced per BN layer), BC-Z (one sentence encoder →
per-block projections), Abdollahzadeh (one task encoder → per-layer generator triple),
and Ha's HyperLSTM (one hyper-cell → per-gate projections) all instantiate it, and
Chauhan's survey names it as the `generate-multiple` output strategy.

**Verdict: our layout is conventional — arguably the single most conventional thing
about our architecture.** The one genuine outlier direction in the corpus is CaFiLM,
which ties the generator across channels *and* sites down to 6 parameters, and Ha's
HyperLSTM is the closest precedent for our specific choice of *branching heads by
functional role* (per-gate there, per-injection-site here).

**(3) Is our per-neuron-baseline-plus-grouped-dynamic decomposition attested
anywhere?**

**The decomposition as a whole: no. Each half separately: yes.**

- The **additive-baseline half** is well attested and is a recognised good practice.
  De Vries's residual-offset CBN, $\gamma_c(z) = \gamma_c + \Delta\gamma_c(z)$ with
  $W^{(2)}$ zero-initialised, is exactly "a static per-channel baseline plus a dynamic
  conditional offset". STSM-FiLM's $(1+\gamma_\alpha)$ trick, FiLM-Ensemble's
  identity-style initialisation, and Beck et al.'s Bias-HyperInit (zero the generator's
  output weights, put the base initialisation in the bias) are all the same idea:
  *start at the identity / pretrained behaviour and make the modulator earn its
  deviation*. AlKilany & Goodman's "addition" coupling mode is the neuromodulation
  version, and they note it is the biologically plausible one — though in their hands
  **substitution beat addition** empirically.
- The **grouped-dynamic half** is attested only in AlKilany & Goodman (§7.2).
- **The combination** — per-neuron static resolution with deliberately coarser dynamic
  resolution — appears nowhere in either corpus, and I did not find it in the
  neuromodulation corpus either.

**Verdict: novel as a composite, but each ingredient is standard.** Worth flagging as
a design contribution if it works, with one caveat worth stating in any write-up: the
decomposition means `grouping_size` does **not** cleanly control total modulation
capacity, because the per-neuron baseline restores full per-neuron degrees of freedom
to the *static* part regardless of $g$. What $g$ controls is specifically the
*resolution of the state-dependent component*. A reader of the grouping screen could
easily mis-read the sweep as a capacity ladder when it is really a
dynamic-resolution ladder.

**(4) Does the literature modulate an MLP's hidden units at all, or is FiLM
overwhelmingly a CONV/per-channel technique whose semantics do not transfer?**

**It does, and the transfer is clean — but the dense-layer evidence is thin and lives
almost entirely in the RL and tabular corners.**

Five dense-layer instances across the two corpora:

| Paper | Dense host | Width | Granularity |
|---|---|---|---|
| Nikulin et al. 2023 (SAC-RND prior) | 4-layer MLP | 256 | per hidden unit; $\gamma,\beta$ from one linear layer of double width, split in half |
| Gorishniy et al. 2025 (TabM) | tabular MLP linear layers | varies | per input unit ($r$) and per output unit ($s$) |
| Beck et al. 2023 (FiLM-VariBAD baseline) | policy MLP | XS–XL scan | per unit |
| Ha et al. 2016 (HyperRNN/HyperLSTM row scaling) | recurrent weight matrix | $N_h$ | per row = per hidden unit |
| Moon et al. 2023 | CNN embedding → MLP | — | per channel of the embedding, then MLP |

The mechanics carry over without modification: in a dense layer, "per-channel" *is*
"per-unit" (§2.1), so the FiLM equation is unchanged. What does **not** carry over is
the *spatial-sharing intuition* that motivates per-channel FiLM in vision — the idea
that a channel is a coherent feature detector whose response should be scaled
uniformly across positions. A dense hidden unit has no such internal structure, so
there is no principled reason from the conv literature to expect any particular
granularity to be right for our 128-unit layers. **The conv corpus gives us the
operator; it does not give us the granularity prior.**

Two further transfer caveats specific to us:

- **Nikulin's finding is the closest match to our setting and it points at placement,
  not granularity.** He conditions a dense 4-layer MLP in an RL context and finds
  conditioning *depth* is the decisive variable (all layers for the prior, last layer
  for the predictor), with the direction flipping depending on what the modulated
  network is *for*. We have five injection points chosen by function, never swept. That
  is an untested degree of freedom sitting right next to the one we are sweeping.
- **The corpus's dense-layer results are all feed-forward or single-step.** TFiLM and
  HyperRNN are the only recurrent modulation precedents, and both self-condition on
  the modulated network's own activity rather than on a raw observation as we do.

---

## 8. What the corpus does NOT settle

1. **Whether coarsening ever helps.** No experiment in either corpus reduces the
   granularity of the modulation *signal* and measures the effect. The three
   granularity comparisons that exist (KML, Beck's hypernet-vs-FiLM, Nikulin's
   FiLM-vs-bilinear) all move along the *finer/richer* direction. AlKilany & Goodman's
   $G$ sweep is the only coarsening evidence anywhere in the library, and it reports
   *flatness*, not improvement. Our grouping screen is therefore testing a hypothesis
   the literature does not support or contradict — it is genuinely open.
2. **Whether a modulator can be over-powered.** Our motivating observation is that
   $g{=}1$ led early then crashed at ~34 M episodes with the temperature knob railed.
   No paper in either corpus reports a modulation-capacity-induced *instability*; they
   report capacity-induced *gains*. The closest thing is Beck et al.'s finding that
   hypernetworks fail catastrophically under default initialisation — an
   *initialisation* pathology, not a capacity one, and one with a known fix
   (Bias-HyperInit). Our per-neuron baseline is structurally the same fix; whether it
   is applied correctly is a question for `code-reviewer`, not for this document.
3. **How many injection sites are right for a recurrent RL policy.** Nikulin is the
   only relevant sweep, is on a feed-forward RND network, and returns
   direction-dependent answers. Nobody has swept injection sites in a recurrent policy.
4. **Whether modulating a recurrent cell's gate is a good idea.** Ha's HyperLSTM
   modulates the recurrent matrix rows per gate; we bias the GRU update gate. These
   are different operations, and no paper in either corpus compares them.
5. **Whether the per-neuron baseline and the grouped dynamic signal interact
   badly.** Untested anywhere. In particular, if the baseline absorbs most of the
   useful modulation, the grouping sweep would show a flat null for reasons that have
   nothing to do with granularity — a confound the screen's current design cannot
   distinguish from a genuine "grouping is inert" result.
6. **Anything about the modulation of *value* heads.** The corpus's RL papers modulate
   priors (Nikulin), auxiliary heads (Moon), and policies (BC-Z, Beck). None modulate
   a critic. See [`film_synthesis.md` §5 Question 6](film_synthesis.md#5-open-questions-for-this-project).

---

## 9. Connections to project gates, hypotheses, and phases

- **Phase 2 (FiLM variant characterisation)** is the phase this document serves
  directly. The grouping screen is a Phase 2 experiment; §7.3 judgment 1 says its axis
  is novel, and §8 item 1 says the literature offers no prior on the outcome.
- **H1 (perceptual amplification)** predicts that $\gamma$ should rise on
  nociceptive-relevant features after injury — a *feature-selective* prediction. Note
  the tension: at $g = 128$ the dynamic component is a single global scalar and
  **H1's feature-selectivity claim becomes untestable in the dynamic component**,
  surviving only in the static per-neuron baseline. Any grouping-screen readout of H1
  needs to say which component it is reading.
- **H2 (memory persistence)** rides on the memory gate-bias head. Ha's HyperLSTM is
  the only precedent for per-gate modulation of a recurrent cell (§7.3 judgment 2),
  and it does not evaluate the analogue of H2.
- **G1 / G2 gates** ([project_plan.md §4](../../project_plan.md)) are survival-step
  thresholds and are not directly informed by this synthesis.
- **Null-result diagnosis series v1–v8** ([`docs/develop/INDEX.md`](../../../develop/INDEX.md)):
  this document supplies one candidate explanation the series has not fully exploited —
  the corpus's strongest, most-replicated placement finding is *"inject at many depths,
  including early ones"* (De Vries's stage sweep; Nikulin's `film_full` prior; Takeda's
  `EncIn` ablation), and our five injection points have never been ablated. If the
  grouping screen returns a null, **placement is the better-supported next axis than
  granularity**, because it is the axis where the literature actually has a consistent
  signal.

---

## 10. Cross-references

- Master index over the 23 FiLM per-paper reviews:
  [`film_lit_review.md`](film_lit_review.md)
- Thematic + historical synthesis (clusters A–G, timeline, cross-cluster matrix):
  [`film_synthesis.md`](film_synthesis.md)
- FiLM ↔ neuromodulation mechanism-mapping synthesis:
  [`film_neuromod_integration_synthesis.md`](film_neuromod_integration_synthesis.md)
- Hypernetwork corpus master review (8 papers, including the Chauhan 2023 taxonomy):
  [`../Hypernetwork/hypernetwork_lit_review.md`](../Hypernetwork/hypernetwork_lit_review.md)
- De Vries et al. 2017 CBN / MODERN (no fresh per-paper review; stage-wise sweep in §2
  and §3.5): [`archive/film_conditional_modulation_review.md`](archive/film_conditional_modulation_review.md)
- AlKilany & Goodman 2025 (the grouping precedent):
  [`../neuromodulatory_algorithms/reviews/alkilany_goodman_2025_snn_dynamic_sensory.md`](../neuromodulatory_algorithms/reviews/alkilany_goodman_2025_snn_dynamic_sensory.md)
- Project algorithm spec (H1–H5, injection points):
  [`docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md`](../../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)
- The live grouping experiment:
  [`docs/experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md`](../../../experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md)

### Recommended follow-ups

- **`experiment-designer`** — an injection-site ablation for the recurrent policy
  (unimodal only / hub only / memory only / all), which is the axis with the corpus's
  strongest and most replicated evidence (§4.1) and is currently unswept.
- **`plan-reviewer`** — the confound named in §8 item 5: the per-neuron baseline may
  make a grouping null uninterpretable. Worth raising against the grouping screen's
  pre-registration before its verdict is believed.
- **`literature-reviewer`** — no new pull is needed for this question. The corpus is
  saturated on Axes 1–3; the gap on Axis 4 is a gap in the *field*, not in our
  coverage of it.

---

## Update from `modulation_in_rl` — literature-reviewer, 2026-08-05

A new topic folder,
[`../modulation_in_rl/modulation_in_rl_lit_review.md`](../modulation_in_rl/modulation_in_rl_lit_review.md),
full-text reviews ten 2024–2026 papers implementing FiLM-style scale-and-shift
modulation in RL / robot policy learning. Two findings bear directly on this document.

**§3.3 and §7.3 are superseded on cell (c).** This synthesis concluded that grouped
modulation — one $(\gamma,\beta)$ shared across a block of units — is *"unattested in
both corpora"*. It is now attested once: **EquAct's `iFiLM` layer** (Zhu et al. 2025,
§4.3, Eqs. 7–9) scales an entire $(2l+1)$-dimensional irreducible-representation block
with a single scalar $\alpha_l$, giving block sizes 1, 3, 5, 7 at $L_{\max}=3$.

**The essential qualifier:** the grouping is **forced by Schur's lemma** under an
SE(3)-equivariance constraint, not chosen as a coarsening. Group boundaries are fixed
by representation theory and cannot be swept, and the paper never frames it as grouping
or parameter saving. A second paper (GEAR, Guo et al. 2026) substitutes iFiLM for
ordinary FiLM in an RL policy and finds the constrained version **worse** (mean success
95.46 % vs 98.85 %) where the symmetry is only approximate. Net rule: *grouping pays
if and only if the tied units are genuinely interchangeable under a structure the task
respects.*

**Axis 3 gains a counter-instance, with a confound.** FLOWER's Global-AdaLN-Zero
(CoRL 2025) shares one modulation weight set across all 18 transformer layers, against
this document's "shared generator + per-site heads" convention. But full text shows the
lost per-layer capacity is **restored by per-layer LoRA adapters that are never
ablated** — there is no `− LoRA` row anywhere — so it is evidence about
*parameterisation*, not about *capacity redundancy*. See §7.3 of the new review.
