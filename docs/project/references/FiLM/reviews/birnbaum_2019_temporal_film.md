---
title: "Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulations"
authors: Sawyer Birnbaum, Volodymyr Kuleshov, S. Zayd Enam, Pang Wei Koh, Stefano Ermon
year: 2019
venue: NeurIPS 2019
slug: birnbaum_2019_temporal_film
source_pdf: sources/Birnbaum et al. 2019 - Temporal FiLM - Capturing long-range sequence dependencies with feature-wise modulations.pdf
topic: FiLM
---

# Birnbaum et al. 2019 — Temporal FiLM (TFiLM): Capturing Long-Range Sequence Dependencies with Feature-Wise Modulations

## Plain-English entry point

This paper introduces **Temporal FiLM (TFiLM)** — a hybrid neural-network building block that combines the strengths of 1-D **convolutional networks** (fast to train, parallel, but with a *fixed and finite receptive field* — they only "see" a short window of the input) with the strengths of **recurrent neural networks (RNNs)** (can in principle remember information from far in the past, but slow and prone to the vanishing-gradient problem). The idea is to let an RNN observe the *whole* sequence at a coarse scale, and use what it learns to dynamically re-scale and re-shift the *fine-grained* activations of a 1-D CNN, channel by channel, block of time by block of time.

Concretely, take a 1-D conv layer's output — a tensor of shape `[time, channels]`. Chop the time axis into **blocks** (e.g., 0.1-second audio chunks that roughly correspond to phonemes). Pool each block down to a single per-channel vector. Run an **LSTM** along the sequence of block-pooled vectors. At each block, the LSTM's hidden state produces a (gamma, beta) modulation pair per channel. Multiply and shift the original convolutional activations within that block by those (gamma, beta). The result: a "self-modulating" convolution whose every feature channel is dynamically rescaled by a globally-aware RNN, achieved with only ~50 RNN steps for an 80,000-sample audio signal.

The paper applies TFiLM to two new application areas: **text classification** (sentiment analysis on Yelp reviews) and a new task they define called **time-series super-resolution** — reconstructing a high-resolution time series (e.g., 16 kHz audio) from a coarsely subsampled version (e.g., one in four samples). The latter has applications in audio bandwidth extension (cheap phone-quality audio → broadcast quality) and genomics (reducing the cost of ChIP-seq experiments by reconstructing high-sequencing-depth signals from low-depth ones). Across all tasks TFiLM beats the corresponding CNN-only baseline by ~1 dB SNR in audio and improves text-classification accuracy from 78.1% to 95.6% on Yelp-2.

## Section-ordered backbone

### Abstract
Proposes TFiLM, an architectural component inspired by adaptive (conditional) batch normalization, that uses an RNN to modulate the activations of a 1-D CNN. Expands the receptive field of convolutional sequence models with minimal computational overhead. Demonstrated on text classification and audio/genomic super-resolution.

### 1. Introduction
Frames the problem: long-range sequential dependencies (in speech, genomics, text) are crucial but hard. RNNs are the standard but suffer from vanishing gradients. 1-D CNNs are fast but have a finite receptive field. TFiLM is the proposed hybrid. Defines a new generative task — time-series super-resolution — and lists potential applications (speech compression, reducing genomics experiment cost, text classification).

### 2. Background
Recaps batch normalization (per-channel mean-subtraction, std-division, learned affine) and FiLM (replace the input-independent affine with a function of an auxiliary input). Table 1 organizes recent FiLM work by base modality (images, audio, text), conditioning modality, and conditioning architecture; positions TFiLM as "Self (Sequence)" conditioning with an RNN conditioning architecture.

### 3. Temporal Feature-Wise Linear Modulation

Algorithm 1 (the TFiLM layer):
1. Take 1-D conv activations $F \in \mathbb{R}^{T \times C}$, a block length $B$.
2. Reshape to $F^{\text{blk}} \in \mathbb{R}^{B \times T/B \times C}$ — split the time axis into $B$ blocks of length $T/B$ each. (Note: the paper's notation here is a bit unconventional; from context and Fig. 1, $B$ is the *number of blocks*, the block length is $T/B$, and the second axis indexes positions within a block.)
3. Pool each block down to a single per-channel vector: $F^{\text{pool}} \in \mathbb{R}^{B \times C}$, where $F^{\text{pool}}_{b,c} = \mathrm{Pool}(F^{\text{blk}}_{b,:,c})$. The paper uses max pooling.
4. Run an LSTM along the sequence of $B$ pooled blocks. The LSTM hidden state at block $b$ produces a $(\gamma_b, \beta_b) \in \mathbb{R}^{2C}$ pair: $(\gamma_b, \beta_b), h_b = \mathrm{RNN}(F^{\text{pool}}_{b,:}; h_{b-1})$, starting with $h_0 = \vec{0}$.
5. Modulate each block: $F^{\text{norm}}_{b,t,c} = \gamma_{b,c} \cdot F^{\text{blk}}_{b,t,c} + \beta_{b,c}$.
6. Reshape back: $F'_{\ell,c} = F^{\text{norm}}_{\lfloor \ell/B \rfloor, \ell \bmod B, c}$.

Key design properties: (a) each $(\gamma_b, \beta_b)$ depends on the *entire history* of pooled blocks 1 through $b$ via the LSTM, so the modulation is globally context-aware; (b) the LSTM is small (dimensionality $O(C)$) and is invoked only $B$ times, so the overhead is small — a 5-second 16 kHz audio recording yields 80,000 samples but only ~50 RNN steps if blocks correspond to ~0.1-second phonemes; (c) the convolutional weights remain spatially shared and locally efficient.

### 4. Time Series Super-Resolution

Defines time-series super-resolution by analogy with image super-resolution: reconstruct a high-resolution signal $y$ from a low-resolution measurement $x$ (e.g., recovering 16 kHz audio from a 2-8 kHz subsampled version).
- **Audio super-resolution / bandwidth extension**: given $x$ at rate $R_1$ and target $y$ at rate $R_2$, with upsampling ratio $r = R_2/R_1$.
- **Genomics super-resolution (ChIP-seq)**: given a low-depth experimental measurement (1M sequencing reads), reconstruct what the high-depth measurement (100M+ reads) would look like.

**4.1 Architecture (Figure 2).** A symmetric encoder-decoder with $K$ downsampling blocks (halving time, doubling channels), a bottleneck, $K$ upsampling blocks (doubling time, halving channels), and symmetric residual skip connections. Each block contains 1-D convolutions, dilation, and TFiLM. Max pooling reduces the LSTM input dimensionality. Subpixel shuffling avoids checkerboard artifacts during upsampling. Trained with mean-squared error.

### 5. Experiments

**5.1 Text Classification (Yelp-2, Yelp-5).** A Small CNN model (3 conv layers, no region embeddings, simplified DPCNN) augmented with TFiLM layers reaches Yelp-2 = 95.6% (vs. 78.1% without TFiLM) and Yelp-5 = 62.3% (vs. 61.5%). Comparable to state-of-the-art VDCNN/DenseCNN despite using fewer parameters; the methods that beat TFiLM (BERT, XLNet, DPCNN) use unsupervised pre-training on much more data. TFiLM models train several times faster than a comparable LSTM-only model.

**5.2 Audio Super-Resolution.** Three tasks: SINGLESPEAKER (1 VCTK speaker), MULTISPEAKER (99 VCTK speakers, generalize to 8 new ones), PIANO (Beethoven sonatas). Baselines: cubic B-spline, Li et al. DNN, Kuleshov et al. 2017 CNN. TFiLM improves over the CNN baseline by ~1.0 dB SNR and ~0.2 dB LSD on average, with the biggest gains on the hardest task (MULTISPEAKER) where long-range context (speaker identity, prosody) helps most. Human raters rank TFiLM-upscaled audio best among the upscaling methods.

**5.3 ChIP-seq Genomics.** Five histone marks (H3K4me1, H3K4me3, H3K27ac, H3K27me3, H3K36me3). Pearson correlation with the high-depth ground truth: TFiLM matches or beats Koh et al.'s specialized CNN baseline on all but one mark, achieving the equivalent of 10–20M sequencing reads from only 1M reads — a substantial cost reduction.

**5.4 Additional analyses.** Visualizing the TFiLM normalizer parameters in audio super-resolution shows clustering by speaker gender, suggesting the modulation learns meaningful long-range features. Ablations confirm TFiLM and skip connections are both significant; cross-domain generalization (speech→piano) suffers without diverse training data. Missing-value imputation on grocery sales also benefits.

### 6. Previous Work and Discussion
Most similar to Kim et al. 2017 (Dynamic Layer Norm) which conditions a feed-forward layer-norm on an audio sequence; TFiLM differs by conditioning *batch-norm* parameters using a *recurrent* network conditioned on the entire sequence. Also related to feed-forward FiLM in question answering, style transfer, speech recognition; this paper extends FiLM along the temporal axis.

### 7. Conclusion
TFiLM is a temporal adaptive normalization layer that bridges convolutional efficiency and recurrent long-range modeling. Effective on text, audio, and genomics tasks.

## Phase 1 — Undergraduate-level synthesis

**The key idea in one sentence.** Take a 1-D CNN that processes a long signal in short, overlapping windows. After each conv layer, group the time axis into blocks (e.g., 0.1-second audio chunks), run a small LSTM along the *sequence of blocks*, and use the LSTM's per-block output as a per-channel (gamma, beta) modulation that re-scales and re-shifts the conv activations *within each block*. The LSTM provides long-range context; the CNN provides local feature extraction.

**Worked example: audio super-resolution.** Goal: turn 4 kHz phone-quality audio into 16 kHz broadcast-quality audio.
1. Take a 4 kHz, 5-second waveform: 20,000 samples.
2. Upsample crudely (cubic interpolation) to 16 kHz, 80,000 samples.
3. Feed this into a 1-D conv encoder with K=4 downsampling blocks, ending at a low-resolution feature representation.
4. After each conv layer, apply a TFiLM layer: divide the time axis into blocks of length B (say 256 samples, ~16 ms), max-pool each block per channel down to a single vector, run an LSTM along the sequence of pooled vectors. Each block emits its own (gamma, beta) per channel, which scales and shifts that block's conv activations.
5. Decode through K upsampling blocks (also TFiLM-augmented) with symmetric skip connections from the encoder.
6. Output a 16 kHz waveform; train against the true 16 kHz signal with MSE.

The LSTM in step 4 lets the model say things like "this block is in the middle of a vowel formed by a female speaker — emphasize the high-frequency harmonics" or "this block continues a sustained piano note — interpolate smoothly with the previous block".

**Why the result is non-trivial.** A pure 1-D CNN with the same number of parameters has a finite receptive field — say 200 samples. It cannot "know" what happened 5 seconds ago. A pure RNN can in principle remember the past but is slow and unstable on 80,000-sample signals. TFiLM's clever trick is that the RNN only operates on the *coarse* (block-pooled) representation — 50 steps instead of 80,000 — and conveys the long-range information back into the local conv processing through the (gamma, beta) channel that FiLM already proved is highly expressive. The MULTISPEAKER experiment is the cleanest demonstration: it's the task that *needs* long-range context (speaker identity changes utterance to utterance) and it's where TFiLM gains the most over a pure CNN.

## Phase 2 — Graduate-level deep dive

### The TFiLM algorithm in equations

Let $F \in \mathbb{R}^{T \times C}$ be the activation tensor of a 1-D conv layer at one example (batch size dropped for clarity), with time dimension $T$ and channels $C$. Let $B$ be the number of blocks; assume $B \mid T$ for clarity, so the block length is $T/B$. The TFiLM layer computes:

**Step 1 — Reshape.**
$$
F^{\text{blk}}_{b, t, c} = F_{b \cdot (T/B) + t,\, c}, \qquad b \in \{0, \ldots, B-1\},\; t \in \{0, \ldots, T/B - 1\}.
$$

**Step 2 — Pool.** Channel-wise pooling over the within-block time axis:
$$
F^{\text{pool}}_{b, c} = \mathrm{Pool}\!\left(F^{\text{blk}}_{b, :, c}\right) = \max_{t \in \{0, \ldots, T/B - 1\}} F^{\text{blk}}_{b, t, c}.
$$

**Step 3 — RNN along the block sequence.**
$$
\bigl( (\gamma_b, \beta_b),\; h_b \bigr) = \mathrm{LSTM}\bigl(F^{\text{pool}}_{b, :}; \, h_{b-1}\bigr), \qquad h_0 = \vec{0}.
$$
The LSTM input dimension is $C$; the hidden state dimension is also $O(C)$; the output is a linear projection from the hidden state to $\mathbb{R}^{2C}$, giving $(\gamma_b, \beta_b) \in \mathbb{R}^C \times \mathbb{R}^C$.

**Step 4 — Per-block FiLM modulation.**
$$
F^{\text{norm}}_{b, t, c} = \gamma_{b, c} \cdot F^{\text{blk}}_{b, t, c} + \beta_{b, c}.
$$
Within a single block, $\gamma_{b, c}$ and $\beta_{b, c}$ are constant over the within-block time axis — i.e., the modulation is *block-piecewise-constant in time and per-channel*. Across blocks, the modulation varies according to the LSTM dynamics.

**Step 5 — Reshape back.**
$$
F'_{\ell, c} = F^{\text{norm}}_{\lfloor \ell / (T/B) \rfloor, \; \ell \bmod (T/B), \; c}.
$$

### Receptive-field interpretation

For a pure 1-D conv stack, the receptive field at depth $L$ with kernel size $k$ and stride 1 grows as $L(k-1) + 1$ — linear in depth. TFiLM augments this with a *global* receptive field at every block: because $\gamma_b, \beta_b$ depend on $F^{\text{pool}}_{1:b}$ via the LSTM, each output position has effective dependence on all preceding blocks' summary statistics. With $B = T/(T/B)$ blocks of length $T/B$, the LSTM context covers $T$ time steps at the cost of $B$ LSTM forward steps. For audio with $T = 80{,}000$ samples and $T/B = 1600$ (matching ~0.1 s phonemes), $B \approx 50$, so the LSTM is invoked 50 times — a 1600x reduction relative to running an LSTM at sample resolution.

### Cost comparison

| Component | Cost (per layer) |
|---|---|
| 1-D conv (kernel $k$, channels $C$) | $O(T \cdot k \cdot C^2)$ |
| TFiLM RNN forward | $O(B \cdot C^2)$ |
| TFiLM modulation | $O(T \cdot C)$ |

Total TFiLM overhead is $O(B \cdot C^2 + T \cdot C)$, dominated by the RNN term, which is $T/B$ times smaller than the conv cost. For typical $T/B = 1000-2000$, TFiLM adds well under 1% overhead. In contrast, a full RNN at sample resolution would cost $O(T \cdot C^2)$ — comparable to the conv stack itself but vastly slower in practice due to serial dependency.

### Relation to FiLM (Perez et al. 2018)

FiLM (`perez_2018_film.md`) generates a single $(\gamma, \beta)$ per layer from a global conditioning input — e.g., a question encoded by a GRU. TFiLM generates a *sequence* of $(\gamma_b, \beta_b)$ vectors, one per time block, where the conditioning input is the model's own activations at a coarser scale. So TFiLM is:
- A *self-conditioning* FiLM (the conditioning input is the network's own internal activations, not an external modality), and
- A *time-varying* FiLM (the modulation parameters change across the time axis).

In the FiLM equation $\mathrm{FiLM}(F_{i,c} | \gamma_{i,c}, \beta_{i,c}) = \gamma_{i,c} F_{i,c} + \beta_{i,c}$, the index $i$ in TFiLM ranges over *time blocks*, not examples. The conditioning generator $f$ that produces (gamma, beta) is the LSTM applied to the pooled activations.

### Relation to Conditional Batch Norm

TFiLM can be viewed as conditional batch norm where:
- The (gamma, beta) parameters are conditioned on a *summary of the input sequence*.
- The conditioning is *time-varying* (different blocks get different (gamma, beta)).
- The conditioning is *causal* (block $b$'s (gamma, beta) depends only on blocks 1 through $b$).

This causality (achieved by using a unidirectional LSTM with $h_0 = 0$) is important for generative tasks like audio super-resolution where future samples shouldn't influence past predictions, and is a natural fit for streaming inference.

### Visualization of learned (gamma, beta)

Section 5.4 mentions that visualizing the TFiLM (gamma, beta) parameters in audio super-resolution shows clustering by speaker gender. This is the temporal-domain analog of Perez et al.'s t-SNE of FiLM parameters clustering by reasoning function: a learned latent structure emerges in the modulation parameter space, indicating the modulation is doing semantically-meaningful work rather than acting as noise.

### Practical design recipe (from §5.2)

- $K = 4$ encoder/decoder blocks.
- Train on patches of length 8192 in high-resolution space.
- Adam optimizer, learning rate $3 \times 10^{-4}$.
- Adjust $B$ so that $T/B \approx 32$ (i.e., 32 samples per block).
- Pooling stride and spatial extent $= 8$.
- Dilated convolutions with dilation factor 2 to further expand local receptive field.
- Adjust the number of filters per layer to normalize parameter counts between baseline and TFiLM-augmented models — TFiLM adds parameters via the LSTM, so a fair comparison requires reducing the conv channel count.

### Limitations and design subtleties

- **Block length is a hyperparameter** that trades off the LSTM's compute cost (smaller blocks → more RNN steps) against the granularity of the temporal modulation (larger blocks → coarser modulation, less ability to capture rapid context changes).
- **Pooling choice matters.** Max pooling is used; mean pooling or attention pooling might be appropriate for different signals.
- **The LSTM is small.** Despite operating on a long sequence (via the block reduction), the LSTM hidden state dimensionality is only $O(C)$, so it cannot store unbounded long-range information — only enough to compute the next block's modulation.

## Connections to other papers in this corpus

- **`perez_2018_film.md`** — Foundational FiLM paper. TFiLM is a direct temporal extension: the FiLM generator is an LSTM operating on block-pooled self-activations, and the modulation varies along the time axis.
- **`dumoulin_2017_cond_instance_norm.md`** — Conditional Instance Norm is the prototype that originally specialized per-channel (gamma, beta) by index; TFiLM specializes them by time block via an LSTM.
- **`huang_belongie_2017_adain.md`** — AdaIN computes (gamma, beta) directly from data statistics; TFiLM uses a learned LSTM, but the conceptual structure ("(gamma, beta) is the load-bearing modulation channel") is the same.
- **`santurkar_2018_batchnorm_optimization.md`** — Provides the underlying optimization justification for why FiLM-style layers atop batch-norm train well. TFiLM inherits this directly.
- **`wisnu_2025_stsm_film.md`** — Applies FiLM-style conditioning in audio (specifically time-scale modification), continuing the line of work TFiLM opens up of FiLM-on-1-D-signals.
