---
title: "STSM-FiLM: A FiLM-conditioned neural architecture for Time-Scale Modification of speech"
authors: Dyah A. M. G. Wisnu, Ryandhimas E. Zezario, Stefano Rini, Fo-Rui Li, Yan-Tsung Peng, Hsin-Min Wang, Yu Tsao
year: 2025
venue: "arXiv preprint (Oct 2025)"
slug: wisnu_2025_stsm_film
source_pdf: sources/Wisnu et al. 2025 - STSM-FiLM - A FiLM-conditioned neural architecture for time-Scale Modification of speech.pdf
topic: FiLM
---

# Wisnu et al. 2025 — STSM-FiLM: A FiLM-conditioned Neural Architecture for Time-Scale Modification of Speech

## Plain-English entry point

This paper applies FiLM (Feature-wise Linear Modulation) — the per-channel scale-and-shift trick from `perez_2018_film.md` — to a long-standing audio-processing task called **Time-Scale Modification (TSM)**. TSM means changing the speed of an audio recording **without changing its pitch**: a recorded sentence can be played back 1.5x faster or 0.7x slower, and it should still sound like the same speaker saying the same words, just at a different rate. (If you naively resampled the audio at 1.5x the rate, pitch would rise too — voices would sound like chipmunks.)

Classical TSM methods like WSOLA (Waveform Similarity Overlap-Add) and PSOLA (Pitch-Synchronous Overlap-Add) use handcrafted heuristics — chunk the audio, find similar boundaries, overlap-add them — and work well in mild stretching cases but introduce audible artifacts (transient smearing, phase glitches) at extreme speed factors. Recent neural TSM methods (TSM-Net, DiffATSM) trained models for specific stretch ratios but lacked an explicit, continuous knob to control playback speed.

The authors' contribution is the first FiLM-conditioned neural TSM model. Architecture: encode the input waveform into latent features (using one of four encoders: STFT-based, WavLM, Whisper, or EnCodec), then **inject the desired speed factor α as a FiLM conditioning signal** that scales and shifts intermediate decoder features. A small MLP turns α into (gamma_α, beta_α), and the FiLM layer applies `(1 + gamma_α) * f + beta_α` (the `1 +` is a stability trick — at initialization the modulation is the identity). The model is trained against WSOLA's outputs as pseudo-targets, so it learns to mimic WSOLA's behavior on easy speeds while smoothing over WSOLA's artifacts at extreme speeds. The best variant (WavLM-HiFiGAN with FiLM) achieves human-rated MOS = 4.40, slightly above WSOLA itself at 4.33 — a substantial result given that WSOLA was the supervision target.

## Section-ordered backbone

### Abstract
Proposes STSM-FiLM: a fully neural TSM architecture using FiLM to condition the network on a continuous speed factor. Supervised against WSOLA outputs. Four encoder-decoder variants explored: STFT-HiFiGAN, WavLM-HiFiGAN, Whisper-HiFiGAN, EnCodec. Demonstrates perceptual consistency across a wide range of time-scaling factors. To the authors' knowledge, the first integration of FiLM into a neural TSM framework.

### 1. Introduction
TSM is fundamental for speech synthesis, audio editing, and intelligibility improvement in noisy environments. Classical methods (Griffin-Lim, WSOLA, PSOLA) preserve perceptual quality reasonably well but use handcrafted heuristics, degrade under non-stationary dynamics, and suffer at extreme factors. Neural methods (TSM-Net, DiffATSM, FastSpeech) have emerged but mostly train on discrete stretch ratios and lack continuous speed control. STSM-FiLM closes this gap by integrating FiLM-based conditioning that takes a continuous speed factor as input.

### 2. Proposed Methods

Goal formalized: given a speech signal $x$ and a desired speed factor $\alpha \in (0, \infty)$, produce $\hat x_\alpha$ resembling a natural utterance of $x$ at slower ($\alpha > 1$) or faster ($\alpha < 1$) playback rate without pitch change.

The model has three components (Figure 1):
- **Feature encoder** transforms raw audio into high-level representations.
- **FiLM conditioning module** injects $\alpha$ as a conditioning signal.
- **Feature decoder** reconstructs the time-scaled waveform.

**2.1 Feature Encoder.** Four variants:
1. **STFT Encoder** — log-mel spectrograms, 1024-dim, fixed resolution.
2. **WavLM Encoder** — 1024-dim features from layer 6 of WavLM-Large (self-supervised transformer).
3. **Whisper Encoder** — 1024-dim features from the last encoder layer of Whisper Medium (multilingual ASR system).
4. **EnCodec Encoder** — encoder of a learned neural audio codec.

Each maps the input audio to a sequence of frame-level features $\{f_1, f_2, \ldots, f_T\}$.

**2.2 FiLM Conditioning Module.** A small MLP maps the speed factor $\alpha$ to FiLM parameters:
$$(\gamma_\alpha, \beta_\alpha) = \mathrm{MLP}(\alpha). \quad (1)$$
These then modulate the features as:
$$\hat f_t = (1 + \gamma_\alpha) \cdot f_t + \beta_\alpha. \quad (2)$$

The `1 +` offset is a stability trick — at initialization (when $\gamma_\alpha \approx 0$), the modulation is the identity, so the model starts as a "speed-1.0 reproduction" baseline before learning the speed-conditional adjustments. For HiFi-GAN-based variants, FiLM is applied at multiple decoder layers; for EnCodec, FiLM is injected between the encoder and the quantizer (modulating latent codes before quantization).

FiLM does not change sequence length. To match the target output duration $T_\alpha = T / \alpha$, the FiLM-modulated feature sequence is **linearly interpolated** along the temporal axis. This is the "duration handle" that turns FiLM's content modulation into actual time stretching.

**2.3 Feature Decoder.**
- For STFT, WavLM, and Whisper encoders: a HiFi-GAN decoder reconstructs waveforms from modulated features.
- For EnCodec: the original EnCodec quantizer + decoder retained.

**2.4 Training and Inference.** Loss combines L1 waveform reconstruction loss against WSOLA-generated pseudo-ground-truth, plus adversarial and feature-matching losses for HiFi-GAN variants:
$$L = \lambda_{L_1} \cdot \|\hat x_\alpha - x_\alpha^{\text{WSOLA}}\|_1 + \lambda_{\text{adv}} \cdot L_{\text{GAN}} + \lambda_{\text{fm}} \cdot L_{\text{FM}}. \quad (3)$$
Training samples $\alpha$ uniformly from $[0.5, 2.0]$ in steps of $0.1$. At inference, the model takes arbitrary $x$ and $\alpha$ and produces $\hat x_\alpha$ — supporting *continuous* $\alpha$ thanks to the FiLM conditioning (no retraining for new speeds).

### 3. Experimental Results and Analysis

**Datasets/Setup.** VCTK (English) and TMHINT-QI (Mandarin), both clean speech. 16 kHz sampling. 200K Adam steps, learning rate $8 \times 10^{-4}$, 32-bit precision. Evaluation metrics: PESQ (quality), STOI (intelligibility), DNSMOS (naturalness), WER/CER (ASR transcription accuracy), and human MOS.

**Average objective performance (Table 1, 2).** STFT-HiFiGAN: best PESQ (2.034) and STOI (0.894). WavLM-HiFiGAN: best DNSMOS (2.986) and lowest WER (0.103) / CER (0.055). Whisper-HiFiGAN: middling. EnCodec: worst overall (quantization artifacts). TSM-Net baseline lies between EnCodec and the HiFi-GAN variants.

**FiLM ablation (Table 3).** WavLM-HiFiGAN with vs. without FiLM at selected speeds. PESQ improves by ~0.5–0.6 across all tested $\alpha \in \{0.7, 1.2, 1.5\}$, with the largest gains at the more extreme factors ($\alpha = 0.7$ and $1.5$). STOI improves by +0.02–0.03. FiLM clearly enhances generalization across speeds.

**Trends across speeds (Figure 2).** STFT-HiFiGAN and WavLM-HiFiGAN are consistently top across $\alpha \in [0.5, 2.0]$; TSM-Net and EnCodec degrade visibly at faster speeds, especially in WER/CER.

**Subjective listening test (Table 4).** 4 utterances × 4 speakers (2M, 2F) × 6 speed factors. WSOLA: MOS 4.33 average. WavLM-HiFiGAN: MOS 4.40 average — slightly *exceeds* its own supervision target. STFT-HiFiGAN: 3.57. TSM-Net: 1.89. Listeners prefer WavLM-HiFiGAN even over the system used as the training target — strong evidence that neural FiLM conditioning learns perceptually-favorable behaviors beyond what WSOLA itself produces.

### 4. Conclusion
STSM-FiLM is the first FiLM-conditioned neural framework for speech time-scale modification. FiLM enables smooth adaptation across a continuous range of speed factors and lets the model improve over its WSOLA-processed training targets. Future work: investigate Whisper-HiFiGAN's padding issues and EnCodec's quantization artifacts; explore other base encoder-decoders.

## Phase 1 — Undergraduate-level synthesis

**The key idea in one sentence.** Take a speech-encoder (e.g., WavLM) that turns audio into a sequence of frame-level feature vectors. Multiply and shift those vectors with two small numbers per channel that are produced from a desired speed factor α (e.g., 0.7 for "play 30% slower"). Then time-stretch the resulting feature sequence by linear interpolation and decode it back to audio with a vocoder (HiFi-GAN). Train this system against the output of a classical method (WSOLA) and it ends up sounding better than WSOLA itself.

**Worked example.** Suppose you have a 3-second VCTK English recording and want to play it back at 1.5x normal speed without pitch change:
1. Encode the waveform with WavLM-Large layer 6, producing a sequence of 1024-dimensional feature vectors at, say, 50 frames per second (so a 3-second clip → 150 vectors).
2. Pass α = 1.5 through a small MLP. Output: a (gamma_α, beta_α) pair, each of dimension 1024.
3. For each of the 150 feature vectors $f_t$, compute $\hat f_t = (1 + \gamma_\alpha) \cdot f_t + \beta_\alpha$.
4. Linearly interpolate the 150-vector sequence down to 100 vectors (since 150 / 1.5 = 100) — that's the time-stretching step.
5. Feed the 100-vector sequence into HiFi-GAN, producing a 2-second waveform that sounds like the same speaker saying the same words 50% faster.

During training, you do this with α uniformly sampled from $[0.5, 2.0]$ and supervise against what WSOLA would have produced for the same $(x, \alpha)$ pair. At inference, you can plug in any α in (0.5, 2.0) — including values never seen exactly during training — and the FiLM conditioning interpolates naturally.

**Why the result is non-trivial.** Earlier neural TSM (TSM-Net) had to be trained per-stretch-ratio. STSM-FiLM uses a single trained model for the entire continuous range $\alpha \in [0.5, 2.0]$. The "1+" trick in the FiLM equation ensures the model starts as the identity at init — a meaningful pre-training prior. The 0.5–0.6 PESQ gain from FiLM ablation (Table 3) demonstrates that the (gamma, beta) modulation is doing real work, not just acting as a learnable bias. And the fact that human listeners rate WavLM-HiFiGAN with FiLM (4.40) above its own teacher WSOLA (4.33) is a clear "student exceeds teacher" demonstration — the model has learned to fix some of WSOLA's heuristic-induced artifacts.

## Phase 2 — Graduate-level deep dive

### Problem formulation

Given a speech signal $x \in \mathbb{R}^L$ (length $L$ samples at 16 kHz) and a speed factor $\alpha \in (0, \infty)$, the goal is to produce $\hat x_\alpha \in \mathbb{R}^{L/\alpha}$ (or $\lfloor L/\alpha \rfloor$, depending on conventions) such that $\hat x_\alpha$:
- has the same perceived pitch as $x$ (no formant shift),
- has the same linguistic content and speaker identity as $x$,
- has duration $L / \alpha$ samples (or sounds like that duration at constant playback rate).

### The STSM-FiLM equations

**Encoder.** A learned or pre-trained encoder $E$ produces a sequence of frame-level features:
$$
F = E(x) = \{f_1, f_2, \ldots, f_T\}, \qquad f_t \in \mathbb{R}^C,
$$
with $T$ frames at the encoder's frame rate (e.g., $T = L / \text{hop\_length}$) and $C$ feature channels (1024 for STFT, WavLM, Whisper).

**FiLM conditioner.** A small MLP $g_\theta : \mathbb{R} \to \mathbb{R}^{2C}$ maps $\alpha$ to FiLM parameters:
$$
(\gamma_\alpha, \beta_\alpha) = g_\theta(\alpha) \in \mathbb{R}^C \times \mathbb{R}^C.
$$

**FiLM modulation (Eq. 2 of the paper).**
$$
\boxed{\;\hat f_t = (1 + \gamma_\alpha) \cdot f_t + \beta_\alpha.\;}
$$

In the Perez 2018 FiLM notation, this is FiLM with $\gamma^{\text{Perez}} = 1 + \gamma_\alpha$ and $\beta^{\text{Perez}} = \beta_\alpha$. The $1+$ offset means $\gamma^{\text{Perez}}|_{\alpha = \text{init}} \approx 1$ and $\beta^{\text{Perez}}|_{\alpha = \text{init}} \approx 0$, so the modulation initializes to the identity transformation. This is the "identity-mapping at init" stability trick that mirrors the well-known initialization choice in residual networks (initialize new layers to zero so the network starts at the identity).

Crucially, the (gamma_α, beta_α) are *spatially uniform* over the time axis $t$ — every frame in the sequence gets the same modulation. This contrasts with TFiLM (`birnbaum_2019_temporal_film.md`), where (gamma_b, beta_b) varies block by block. Time variation in STSM-FiLM is handled separately, via the next step.

**Temporal interpolation.** The FiLM modulation alone does not change the sequence length. To match the desired output duration $T_\alpha = \lfloor T / \alpha \rfloor$, the modulated sequence $\{\hat f_1, \ldots, \hat f_T\}$ is **linearly interpolated** along the time axis:
$$
\tilde f_s = \mathrm{LinInterp}\!\left(\{\hat f_t\}_{t=1}^{T}; \, t_s = s \cdot \frac{T-1}{T_\alpha - 1}\right), \qquad s = 1, \ldots, T_\alpha.
$$

This separates the "what the features look like at speed α" decision (handled by FiLM) from the "how long the feature sequence is" decision (handled by linear interpolation). The FiLM conditioning lets the same model produce subtly different feature dynamics for different α — for example, suppressing transient-heavy components when stretching slowly to avoid smearing artifacts.

**Decoder.** A HiFi-GAN $D$ (or EnCodec quantizer+decoder, for that variant) maps $\tilde F = \{\tilde f_s\}$ back to a waveform:
$$
\hat x_\alpha = D(\tilde F).
$$

### Loss function (Eq. 3)

The total training loss combines three terms:
$$
L = \lambda_{L_1} \cdot \|\hat x_\alpha - x_\alpha^{\text{WSOLA}}\|_1 + \lambda_{\text{adv}} \cdot L_{\text{GAN}} + \lambda_{\text{fm}} \cdot L_{\text{FM}},
$$

where:
- $L_1$ reconstruction against the WSOLA-generated pseudo-target $x_\alpha^{\text{WSOLA}}$ provides the primary supervision signal.
- $L_{\text{GAN}}$ is the standard HiFi-GAN least-squares discriminator loss promoting perceptual quality:
  $$
  L_{\text{GAN}}(\hat x, x) = \mathbb{E}_{\hat x}[(D_{\text{disc}}(\hat x) - 1)^2],
  $$
  averaged over multiple discriminators (multi-period and multi-scale, per HiFi-GAN).
- $L_{\text{FM}}$ is the feature-matching loss between intermediate discriminator features of real and generated speech, encouraging the generator to produce signals whose discriminator-perceived features match the real ones.

### Why FiLM is the right inductive bias for TSM

TSM is fundamentally a *single-knob continuous-control* problem: there is one number, $\alpha$, that should smoothly govern the network's behavior. FiLM's strengths fit this exactly:
1. **Low-dimensional conditioning input.** $\alpha \in \mathbb{R}$ is the simplest possible conditioning signal; a small MLP $\mathbb{R} \to \mathbb{R}^{2C}$ suffices.
2. **Continuous conditioning.** FiLM's (gamma, beta) are continuous functions of $\alpha$, so the model interpolates between speed factors naturally. The "linear (gamma, beta) algebra" demonstrated in Perez 2018 (zero-shot generalization via linear combination of (gamma, beta) parameters) supports this directly — speed factors that the model has never seen during training still produce sensible outputs.
3. **Per-channel modulation matches per-feature semantics.** Different feature channels in WavLM's representation correspond to different phonetic/acoustic concepts; FiLM lets the speed conditioning selectively rescale them, suppressing those that are speed-sensitive (e.g., transient detectors that would smear at slow $\alpha$) while preserving the rest.
4. **Decoupling from the temporal stretching mechanism.** FiLM does not need to handle the "make the sequence longer/shorter" job — that's left to linear interpolation. FiLM only adjusts *what* the features represent for the chosen speed, not *how many* frames to produce.

### Comparison with other FiLM variants in this corpus

| Variant | Conditioning input | Conditioning generator | Time variation |
|---|---|---|---|
| `perez_2018_film.md` (FiLM) | Question (sequence of tokens) | GRU + linear projection | None (single (gamma, beta) per layer) |
| `dumoulin_2017_cond_instance_norm.md` (CIN) | Style index (discrete) | Embedding lookup table | None |
| `huang_belongie_2017_adain.md` (AdaIN) | Style image | VGG-19 + per-channel statistics | None |
| `birnbaum_2019_temporal_film.md` (TFiLM) | Block-pooled self-activations | LSTM over block sequence | Per-block |
| STSM-FiLM (this paper) | Speed factor $\alpha \in \mathbb{R}$ | Small MLP | None per-FiLM (uniform across time) — temporal stretching handled separately |

STSM-FiLM is unusual in its *scalar-conditioning* setup — most prior FiLM work uses high-dimensional conditioning (questions, style images, sequences). The success demonstrates that even a 1-D conditioning input, properly mapped through an MLP into (gamma, beta) space, can encode rich continuous control.

### Ablation evidence (Table 3)

| Condition | $\alpha = 0.7$ PESQ/STOI | $\alpha = 1.2$ | $\alpha = 1.5$ |
|---|---|---|---|
| WavLM-HiFiGAN, no FiLM | 1.55 / 0.84 | 1.50 / 0.88 | 1.46 / 0.88 |
| WavLM-HiFiGAN, + FiLM | 2.10 / 0.87 | 2.03 / 0.90 | 2.04 / 0.90 |

FiLM improvement is +0.55, +0.53, +0.58 in PESQ across the three speeds — remarkably uniform. STOI gains are smaller (+0.02–0.03) since the intelligibility was already high without FiLM. The biggest absolute gains are at the more extreme stretch factors, consistent with the hypothesis that FiLM helps most where the baseline (and WSOLA) struggle the most.

### Subjective MOS as evidence of generalization

The headline result is in Table 4. WSOLA, the supervision target, scores MOS = 4.33. WavLM-HiFiGAN with FiLM scores MOS = 4.40 — *higher than its teacher*. The student does not just copy WSOLA; it learns a smoothed-over version that listeners find more natural. This is consistent with the "neural conditioning can reduce artifacts of classical methods" claim and is the cleanest evidence that the model is not merely memorizing WSOLA outputs.

### Open issues acknowledged

- **Whisper-HiFiGAN's weak intelligibility.** The authors hypothesize "padding effects in the conditioning pipeline" — likely related to how Whisper's fixed-input-length pre-processing interacts with variable-rate FiLM modulation. Not investigated in this paper.
- **EnCodec's quantization artifacts.** The discrete quantization codebook bottleneck appears to interact poorly with FiLM's continuous conditioning — the FiLM signal can be quantized away. This is consistent with the general challenge of combining differentiable conditioning with discrete latent spaces.

## Connections to other papers in this corpus

- **`perez_2018_film.md`** — Direct foundational reference (cited as Perez et al. 2018, ref [11] in the paper). STSM-FiLM uses the FiLM equation with a stability-promoting `1 +` offset on the multiplicative term.
- **`birnbaum_2019_temporal_film.md`** — Closely related: also uses FiLM for audio sequence modeling. TFiLM modulates *block-by-block* using an LSTM on self-activations; STSM-FiLM modulates *globally* using an MLP on a scalar speed factor. The two represent complementary points in the FiLM design space — TFiLM is self-conditioning and time-varying; STSM-FiLM is externally-conditioned and time-uniform.
- **`dumoulin_2017_cond_instance_norm.md`** — STSM-FiLM's MLP $g_\theta(\alpha)$ is the continuous-conditioning analog of Dumoulin et al.'s discrete embedding-table lookup. The interpolation properties of CIN's learned (gamma, beta) carry over: the MLP automatically interpolates between speed factors seen during training.
- **`huang_belongie_2017_adain.md`** — Conceptually related: both use FiLM with a low-overhead, statistics-based conditioning generator (here, an MLP on a scalar; in AdaIN, deterministic statistics). Both demonstrate that the *generator* of (gamma, beta) can be simple as long as the modulation target is rich.
- **`santurkar_2018_batchnorm_optimization.md`** — The smoothing-based justification for normalization-affine architectures supports STSM-FiLM's HiFi-GAN backbone, which contains batch normalization throughout.
