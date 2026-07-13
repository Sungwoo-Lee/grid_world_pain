> **Per-paper review — continual-learning corpus, paper 19 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§19); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 19. Shang et al. 2016 — Understanding and Improving CNNs via Concatenated ReLU (CReLU)

**PDF:** `docs/project/references/continual_learning/sources/Shang et al. 2016 - Concatenated ReLU (CReLU).pdf`
**Venue:** ICML 2016 (PMLR 48:2217–2225). **Authors:** Wenling Shang, Kihyuk Sohn, Diogo Almeida, Honglak Lee.

**Primer connection.** This is the activation scheme that Abbas et al. (2023) later repurposed as their *strongest single fix* for loss of plasticity in continual deep RL (primer §2 Phase 3, §4(c)). The original 2016 paper is a *supervised-vision* paper with no mention of plasticity — it argues CReLU on the grounds of parameter efficiency, regularization, and reconstruction. Reading it clarifies *why* the mechanism happens to preserve plasticity: CReLU makes it structurally impossible for a unit to become one-sided-dead, because every filter's negative phase is always represented by a live companion channel. That is the bridge from "better CIFAR/ImageNet features" (2016) to "keeps units responsive under non-stationary RL training" (2023).

<a id="p1-crelu"></a>
## Phase 1 — Foundational Overview

**The problem in plain terms.** A standard ReLU unit computes $\max(x, 0)$: it keeps positive signal and throws away everything negative. Shang et al. noticed something odd when they looked inside a trained AlexNet: in the first few convolution layers, the learned filters come in **near-opposite pairs** — for almost every filter there is another filter pointing in nearly the opposite direction. The network is spending two filters to represent one direction, one for the "positive phase" and one for the "negative phase," because ReLU erased the negative half of each and the network had to relearn it separately. That is wasted capacity (redundancy).

**The fix.** Instead of letting the network learn those mirror-image pairs the hard way, *build the mirroring into the activation*. **Concatenated ReLU (CReLU)** takes each linear response $x$, makes a negated copy $-x$, stacks them, and applies ReLU to both: output is the pair $(\max(x,0),\ \max(-x,0))$. Now a single filter automatically contributes both its positive and its negative half. Whichever sign the input has, *one of the two channels is always active* — so information is never destroyed, and no filter can be permanently silenced by an unlucky sign.

**Key findings.**
- Replacing ReLU with CReLU in the lower convolution layers *improves accuracy* on CIFAR-10, CIFAR-100 and ImageNet — often *while using fewer parameters* (because you can halve the filter count and still match or beat the baseline's number of live activations).
- The benefit is concentrated in the **lower layers**; deep layers show less "pairing," so CReLU there helps little (best ImageNet result: CReLU on conv1–4 only).
- CReLU acts as a **regularizer**: CReLU models show a much smaller train/test error gap than ReLU models with the same or more parameters.
- After adding CReLU, the mirror-pair phenomenon *disappears* (as intended) — each filter now uniquely spans its own direction.

**Initial takeaway.** A one-line change to the activation — "also keep the negative half, as its own channel" — recovers information ReLU throws away, removes learned redundancy, regularizes, and (crucially for our field) guarantees each unit always has a live channel. That last property is exactly what makes CReLU a plasticity-preserving activation in the RL continual-learning setting, even though this paper never uses that word.

<a id="p2-crelu"></a>
## Phase 2 — Graduate-Level Deep Dive

### 2.1 The pairing observation, formalized

For a set of unit-length filters $\{\phi_i\}$, define the **pairing filter** of $\phi_i$ as the filter most anti-aligned with it:

$$\bar\phi_i = \arg\min_{\phi_j}\ \langle \phi_i, \phi_j\rangle,$$

and their cosine similarity

$$\mu^{\phi}_i = \langle \phi_i, \bar\phi_i\rangle.$$

Empirically, for AlexNet's `conv1` the histogram of $\mu^{w}_i$ (over learned weight filters $w$) is **strongly negatively centered** — i.e. $\mu^w_i \approx -1$ for many filters, meaning true near-opposite pairs — whereas for random Gaussian unit filters $r_i$ the histogram of $\mu^r_i$ centers near $0$ (random high-dimensional vectors are nearly orthogonal). Going deeper (`conv2`→`conv5`), the learned distribution's center drifts back toward $0$: the pairing is a *lower-layer* phenomenon. This is the empirical premise for restricting CReLU to lower layers.

The **conjecture**: despite ReLU erasing negative linear responses, the lower layers capture *both* phases by learning negatively correlated filter pairs — implying redundancy that a phase-preserving activation could remove.

### 2.2 CReLU definition and the information-view of activations

Denote ReLU by $[\cdot]_+ \triangleq \max(\cdot, 0)$.

**Definition (CReLU).** The CReLU activation $\rho_c:\mathbb{R}\to\mathbb{R}^2$ is
$$\rho_c(x) \triangleq \big([x]_+,\ [-x]_+\big).$$

An information-theoretic reading of three activations clarifies what CReLU preserves:
- **ReLU** retains the *phase* but destroys the *modulus* when the response is negative (it maps all $x<0$ to $0$).
- **AVR** (absolute-value rectification, $|x|$) retains the *modulus* but destroys the *phase* (it cannot distinguish $x$ from $-x$).
- **CReLU** retains *both*: from $([x]_+, [-x]_+)$ you can reconstruct $x = [x]_+ - [-x]_+$ exactly.

This is why CReLU beats AVR empirically (Tables 1, 4): AVR discards phase, and phase turns out to be essential for state-of-the-art deep CNN features. It is also the seed of the plasticity argument: because $x$ is exactly recoverable and *one of the two channels is always $>0$ whenever $x\neq 0$*, a CReLU "unit" (the pair) can never be pushed into the permanently-zero-gradient regime that kills a lone ReLU unit.

### 2.3 Reconstruction property (Proposition 2.1) — step by step

Let $x\in\mathbb{R}^D$ be an input and $W$ the $D\times K$ matrix whose columns are the filters $w_i\in\mathbb{R}^l$. Decompose $x$ orthogonally with respect to the column space of $W$:

$$x = x' + (x - x'),\qquad x'\in\operatorname{range}(W),\quad (x-x')\in\ker(W^\top).$$

**Proposition 2.1.** The component $x'$ (the part of the input spanned by the filters) is fully recoverable from
$$f_{\text{cnn}}(x) \triangleq \operatorname{CReLU}(W^\top x).$$

*Why this holds (sketch of the constructive proof).* Write the linear responses $y = W^\top x \in \mathbb{R}^K$. CReLU stores $([y]_+, [-y]_+)$, and since $y = [y]_+ - [-y]_+$, the full real vector $y=W^\top x$ is recovered **losslessly** from the CReLU output — no thresholding information is lost (contrast plain ReLU, which only gives $[y]_+$ and cannot recover the negative coordinates of $y$). Given $y = W^\top x = W^\top x'$ (because $x-x'\in\ker(W^\top)$ contributes nothing), and $x'\in\operatorname{range}(W)$ means $x' = W\alpha$ for some coefficient vector $\alpha$, we have $y = W^\top W \alpha$. On $\operatorname{range}(W)$ the Gram operator $W^\top W$ is invertible (restricted to the row space), so $\alpha = (W^\top W)^{+} y$ and

$$x' = W\,(W^\top W)^{+}\,W^\top x = W\,(W^\top W)^{+}\, y,$$

with $(\cdot)^{+}$ the pseudo-inverse. This is precisely the orthogonal projector onto $\operatorname{range}(W)$ applied to $x$. The paper's Algorithm 1 is a linear (no additional learning) reconstruction realizing this map; Figure 5 shows the recovered images. The component $x-x'\in\ker(W^\top)$ is genuinely unrepresented by the filters and is irrecoverable — as it must be for any filter bank. The max-pooling case needs extra input-space constraints for a non-trivial bound (supplementary §A.2).

### 2.4 Regularization: the Rademacher-complexity argument (Theorem 4.1)

The striking empirical fact is that CReLU *doubles* the parameter count yet does **not** increase overfitting. The formal support:

**Theorem 4.1.** Let $\mathcal{G}$ be a class of real functions $\mathbb{R}^{d_{in}}\to\mathbb{R}$ with input dimension, $\mathcal{G}=[\mathcal{F}]^{d_{in}}_{j=1}$. Let $\mathcal{H}$ be a linear map from $\mathbb{R}^{2d_{in}}\to\mathbb{R}$ parameterized by $W$ with $\lVert W\rVert_2 \le B$. Then the empirical Rademacher complexity of the composite obeys

$$\hat{\mathfrak{R}}_L(\mathcal{H}\circ\rho_c\circ\mathcal{G}) \le \sqrt{d_{in}}\,B\,\hat{\mathfrak{R}}_L(\mathcal{F}).$$

*Interpretation.* This bound is **the same** as the known bound for ReLU + linear transformation (Wan et al. 2013). The CReLU doubling ($d_{in}\to 2d_{in}$ channels) does not enlarge the complexity bound because the two channels $[x]_+$ and $[-x]_+$ are *deterministic functions of the same pre-activation* — they carry no independent Rademacher degrees of freedom. The key contraction step is that $\rho_c$ is $1$-Lipschitz componentwise (both $[\cdot]_+$ and $[-\cdot]_+$ are), so the standard Ledoux–Talagrand contraction absorbs the concatenation without an extra factor. Hence "twice the parameters, same capacity bound" — the formal shape of the observed regularization.

To rule out that CReLU's two output channels are merely negations of each other (which would make it collapse to AVR), Table 6 measures the correlation between the *outgoing* weights of the positive-channel and negative-channel of each pair; the "pair" correlations are only marginally above the "non-pair" baseline and both are well below $1$ — i.e. the network genuinely learns *distinct* non-linear manipulations of the two phases.

### 2.5 Bridge to plasticity (the reason this paper is in a continual-learning corpus)

Nothing in Shang et al. names plasticity, but the mechanism translates directly to the loss-of-plasticity vocabulary of the primer:
- A **dormant/dead ReLU unit** (Sokar et al. 2023; primer §2 Phase 3) is a unit whose pre-activation is negative on (almost) all inputs, so it outputs $0$ and receives (almost) no gradient. CReLU's construction makes this **impossible at the pair level**: if $x<0$ almost always, the companion channel $[-x]_+ > 0$ almost always and stays trainable.
- **Effective-rank collapse** (Kumar et al. 2021; Abbas et al. 2023) is mitigated because CReLU's lossless preservation of the pre-activation keeps feature directions distinct (§2.3), rather than collapsing negative-phase directions to zero.
- Abbas et al. (2023) found CReLU the *most effective* single intervention among activation changes, resets, and regularization in cycling-Atari continual RL. This paper supplies the mechanistic "why": phase preservation + guaranteed-live companion channel. Cost to note for any project use: CReLU **doubles the width** of every layer it touches (so downstream weight matrices double their input dimension) — the parameter/compute accounting Abbas inherits comes straight from §2.2 here.

<a id="bb-crelu"></a>
## Appendix: Section-by-Section Backbone

**§1 Introduction.** Motivates from a curious observation: AlexNet's lower conv layers learn negatively-correlated ("opposite-phase") filter pairs (Fig. 1). Hypothesizes lower layers capture both phases via redundant pairs; proposes CReLU to remove the redundancy while preserving both phases and non-saturated non-linearity. Claims parameter-efficiency + accuracy gains on CIFAR-10/100 and ImageNet.

**§2 CReLU and Reconstruction Property.**
- **§2.1 Conjecture on convolution layers.** Defines pairing filter $\bar\phi_i=\arg\min_{\phi_j}\langle\phi_i,\phi_j\rangle$ and $\mu^\phi_i=\langle\phi_i,\bar\phi_i\rangle$. Histograms (Fig. 2) show learned `conv1` filters are negatively centered (true pairs) vs. random filters near $0$; effect fades with depth. Information view: ReLU keeps phase/loses modulus; AVR keeps modulus/loses phase; scattering nets keep modulus. Defines **CReLU**: $\rho_c(x)=([x]_+,[-x]_+)$. Contrasts with Leaky ReLU (a *function* with small negative slope) — CReLU is an *activation scheme*, composable with other non-linearities.
- **§2.2 Reconstruction property.** Because CReLU preserves all post-convolution information, reconstruction analysis is clean. **Proposition 2.1**: the range-of-$W$ component $x'$ of input $x$ is recoverable from $\operatorname{CReLU}(W^\top x)$. Max-pooling case deferred to supplementary (needs extra constraints).

**§3 Benchmark Results.**
- **§3.1 CIFAR-10/100.** Baseline ConvPool-CNN-C / VGG. Replacing ReLU→CReLU (same filter count → doubles channels/params) improves accuracy; CReLU+half (halved filters → same #neurons, half the params of baseline) still beats baseline (Table 1). On deeper VGG, applying CReLU to conv1 / conv1,3 / conv1,3,5 while halving filters gives substantial gains (Table 2). CReLU shows smaller train/test gap (regularization). AVR sometimes beats baseline but is inferior to CReLU under averaging/voting.
- **§3.2 ImageNet.** Baseline All-CNN-B. CReLU on **conv1–4** gives best top-1/top-5 (Table 4); going deeper doesn't help (matches the depth-fading pairing observation). CReLU(all) with only 4.7M params beats FriedNet/PrunedNet parameter-reduction methods (Table 5).

**§4 Discussion.**
- **§4.1 Regularization view.** CReLU overfits less despite 2× params. **Theorem 4.1**: Rademacher complexity bound of CReLU+linear equals that of ReLU+linear — doubling params need not increase capacity.
- **§4.2 Invariant features.** CReLU models have consistently higher invariance scores (Fig. 4); local maxima at conv1/conv4/conv7 motivate the CReLU(conv1,4,7) architecture, which achieves best 10-patch ImageNet result with fewer params.
- **§4.3 Revisiting reconstruction.** After CReLU, the pairing phenomenon vanishes (Fig. 3: learned distribution aligns with random). Table 6: positive/negative outgoing-weight correlations well below 1 → the two phases are manipulated distinctly (not mere negation), separating CReLU from AVR. Linear (no-learning) reconstructions (Fig. 5) qualitatively confirm Proposition 2.1.

**§5 Conclusion.** CReLU conserves positive+negative linear responses so each filter efficiently spans its own direction; improves classification with fewer params; suggested extensions to structured prediction / generation.

---
