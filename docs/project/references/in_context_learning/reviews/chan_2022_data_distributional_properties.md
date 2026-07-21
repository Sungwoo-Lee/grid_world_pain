> **Per-paper review — in-context-learning corpus, paper 26 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§26); content is identical. Manifest: [[in_context_learning_sources]].

# 26. Chan et al. 2022 — Data Distributional Properties Drive Emergent In-Context Learning

**PDF:** `docs/project/references/in_context_learning/sources/Chan et al. 2022 - Data Distributional Properties Drive Emergent In-Context Learning.pdf`
**Authors / venue:** Stephanie C.Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya K. Singh, Pierre H. Richemond, James L. McClelland, Felix Hill (DeepMind / UCL / Stanford). arXiv:2205.05055.

## Phase 1: Foundational Overview (Undergraduate-Level)

**The question.** Large transformer language models can do "in-context learning" (ICL) — solve a brand-new task from a few examples typed into the prompt, with **no weight updates**. Nobody trained them explicitly to do this; it just *appeared*. Why? This paper's answer: it is not just the transformer architecture — it is a property of the **training data itself**. Certain statistical shapes of the data (which natural language happens to have) *cause* ICL to emerge.

**The setup in plain terms.** The authors build a clean toy world from the Omniglot handwritten-character dataset (1,623 character classes, 20 images each). The model sees sequences of (image, label) pairs — 8 pairs of "context" followed by one "query" image — and must predict the query's label. They then deliberately dial specific statistical properties of the training data up and down and watch whether ICL emerges. They contrast two learning modes:
- **In-context learning (ICL):** answer the query by looking at the examples in *this* prompt (works even for characters never seen in training, whose label is randomly reassigned each sequence).
- **In-weights learning (IWL):** answer by recalling a fixed image→label mapping baked into the weights during training (works only for classes seen in training, and fails when the query class is absent from the context).

**Key findings — which data properties turn ICL on.**
1. **Burstiness.** In natural data an item tends to appear in temporal *clusters* rather than uniformly. They make a fraction of training sequences "bursty" — the query class appears 3 times in the context. More burstiness → more ICL, less IWL.
2. **Many rare classes (large vocabulary + long tail).** Going from 100 → 1,600 → 12,800 training classes (rotations/flips manufacture extra classes) increases ICL. You need **both** burstiness **and** many classes; neither alone suffices.
3. **Dynamic meaning.** (a) *Label multiplicity* — giving each class several possible labels (chosen randomly per sequence, consistent within a sequence) increases ICL. (b) *Within-class variation* — more visual variety per class (full 20-exemplar Omniglot, or added pixel noise) increases ICL. Making the generalization problem harder preferentially *hurt IWL more than ICL*, so ICL won by comparison.
4. **A tradeoff, and how to break it.** In most conditions ICL and IWL *trade off* — a single model could not hold both. The fix: train on a **Zipfian (power-law) marginal** over classes (few very-common classes, long tail of rare ones — exactly like word frequencies). At **Zipf exponent ≈ 1** (the value real languages sit at) the model keeps **both** high ICL (driven by the rare tail) **and** high IWL of the common classes. Uniform data (exponent 0) gives only ICL; very skewed data (exponent 3) gives only IWL.
5. **Architecture also matters.** Under identical data and matched parameter counts, **vanilla RNNs and LSTMs never achieved ICL** — only the transformer did. But a transformer on the *wrong* data distribution also failed. So: "attention is *not* all you need" — architecture **and** data are both necessary.

**Initial takeaway.** ICL is not a free gift of the transformer. It emerges from the *interaction* of the transformer architecture with naturalistic data statistics — burstiness, a heavy-tailed vocabulary of rare items, and dynamic/contextual meaning. Because language natively has all of these (and the magic Zipf-≈1 skew), the paper offers a mechanistic story for *why* ICL "just appears" in LLMs, and a recipe for eliciting it in non-language domains (which are usually deliberately uniformized). Cognitive tie-in: maps onto complementary-learning-systems theory (neocortex ≈ slow weights, hippocampus ≈ transformer context window).

## Phase 2: Graduate-Level Deep Dive

**Task and model formalism.** Each training sequence is a length-17 token stream: 8 image-label pairs (context) plus a query image,
$$ \big(\text{img}_1, \ell_1, \text{img}_2, \ell_2, \dots, \text{img}_8, \ell_8, \text{img}_{\text{query}}\big), $$
and the target is the query label. Images pass through a (non-pretrained) 2-block-per-group ResNet embedder; integer labels through a standard embedding table; both augmented with sinusoidal positional encodings; then fed to a **causal transformer** (12 layers, embedding dim 64, 8 heads by default). The objective is softmax cross-entropy on the *query* prediction only:
$$ \mathcal{L}(\theta) = -\,\mathbb{E}_{\text{seq}}\Big[ \log p_\theta\big(\ell_{\text{query}} \mid \text{img}_1,\ell_1,\dots,\text{img}_8,\ell_8,\text{img}_{\text{query}}\big)\Big]. $$
Crucially — unlike meta-training — **image→label mappings are fixed across training sequences**, and images/classes recur. This is the deliberate "middle ground" between standard supervised learning (fixed mappings, uniform recurrence) and few-shot meta-training (novel mappings every episode).

**Evaluation probes (design of the ICL vs IWL dissociation).**
- *ICL probe (Fig 1c):* a 4-shot 2-way few-shot sequence on **holdout** classes never seen in training; the two classes are randomly assigned to labels $\{0,1\}$ *per sequence*. Because labels are re-randomized, the only way to score above chance $=\tfrac12$ is to read the current context. Accuracy is computed restricted to the two in-context labels (so the model cannot cheat by copying any context label).
- *IWL probe (Fig 1d):* classes carry their **training labels**, but the query class is forced to be **absent from the context** (classes sampled uniformly without replacement within the sequence). No contextual support → the model must retrieve a weight-stored mapping; chance $\approx 1/1600$.

This construction is what makes the paper's claims clean: for the *training* data the two strategies give identical answers (labels are fixed), so the model's behavior on the *ambiguous* probes reveals a learned **bias**, not a correctness difference.

**Burstiness — formal instantiation.** A sequence is "bursty" iff the query class appears exactly 3 times in the context; to prevent a majority-label shortcut, a *second* class also appears 3 times. Non-bursty sequences draw the 8 context pairs i.i.d. uniformly over the class pool. The scalar control knob is $p(\text{bursty})\in\{0,0.5,0.9,1.0\}$ = fraction of bursty sequences. Empirically ICL accuracy is monotone increasing in $p(\text{bursty})$ and IWL accuracy monotone decreasing (Fig 2). A notable dynamical wrinkle: some runs *start* ICL-biased and drift toward IWL over training — ICL can be transient.

**Number of classes.** Holding $p(\text{bursty})=0.9$, ICL rises with the class count $\in\{100, 1600, 12800\}$. The 12,800-class regime is manufactured by applying the 8-element group of rotations $\{0^\circ,90^\circ,180^\circ,270^\circ\}\times$ horizontal flip, with holdout transforms excluded from training. Caveat the authors flag: Omniglot's rotational/mirror symmetries mean the ×8 augmentation *also* injects a label-multiplicity effect (same image → multiple labels), so the 12,800 result partly confounds "more classes" with "dynamic meaning."

**Dynamic meaning — two instantiations.**
- *Label multiplicity* $m\in\{1,2,5,10\}$: each class has $m$ candidate labels; the shown label is sampled per sequence but held consistent within a sequence ("one sense per discourse"). ICL increases with $m$.
- *Within-class variation:* single fixed exemplar (lowest), + resampled Gaussian pixel noise ($\sigma\in\{0.1,0.5\}$), full 20-exemplar Omniglot (highest). More variation → more ICL, bounded above by within-class generalization difficulty.

**The Zipfian sweet spot (the central quantitative result).** The marginal class distribution is
$$ p(X = x) \propto \frac{1}{x^{\alpha}}, \qquad x = \text{class rank},\ \alpha \in [0,\infty). $$
Interpretation of the limits and the interior:
- $\alpha = 0$ (uniform): every class rare → **high ICL, ~zero IWL**.
- $\alpha \to$ large (e.g. 3): a handful of classes dominate (top-3 ≈ 97% of data) → **high IWL of common classes, ICL collapses**, and even IWL of rare classes stays at chance (rare items are *never* memorized, Fig 6e).
- $\alpha \approx 1$: **sweet spot** — both ICL (on holdout) and IWL (on the 10 most-common classes) coexist at high accuracy. The long rare tail induces ICL while the common head is memorized into weights. Real languages sit at $\alpha\approx 1$ (Piantadosi 2014), a striking coincidence.

The mechanistic reading: the rare tail acts like a stream of quasi-meta-training episodes (rare + bursty ⇒ disproportionately likely to recur *within* a context window), while the common head behaves like a stationary supervised dataset the weights can memorize. Skew lets one dataset serve both regimes.

**Architecture ablation.** Matching depth, hidden size, and parameter count (Transformer-12L ≈ 831k params; LSTM-12L ≈ 628k; a 15-run log-uniform LR/warmup sweep per architecture, 90 runs total), vanilla RNN and LSTM **never exceed chance** on the ICL probe (Fig 7), while the transformer succeeds. The transformer even matches-or-beats the recurrent nets on **IWL** (Fig 8) — so the recurrent failure is *not* explainable as "RNNs are merely more IWL-biased." The authors connect the transformer's advantage to modern Hopfield / associative-memory equivalences (Ramsauer et al. 2021; Krotov & Hopfield 2021): attention over a context window is computationally akin to query-based associative retrieval, which recurrent state compression lacks.

**Why this matters for the shard's theme.** Chan et al. give the *data-side* necessary conditions for ICL: (i) burstiness, (ii) a large long-tailed vocabulary of rare classes, (iii) dynamic meaning, and (iv) — for ICL+IWL coexistence — a Zipf-≈1 marginal. These are exactly the levers a designer of *non-language* ICL curricula (e.g. RL environments, which are usually uniformized) would pull. The paper explicitly notes RL environments are typically uniform (citing Chan et al. 2022 "Zipfian environments for RL") and that this may forfeit an ICL capability.

## Appendix: Section-by-Section Backbone

- **Abstract.** ICL emerges from *data distributions*: burstiness + many rare classes + dynamic meaning. Initially ICL/IWL trade off, but a skewed Zipfian marginal lets both coexist. Naturalistic distributions elicit ICL only in transformers, not RNN/LSTM. Data **and** architecture both matter.
- **§1 Introduction.** ICL = rapid generalization from few in-context examples, no gradient updates; contrasts IWL (slow, gradient-based). Meta-learning achieved few-shot via *explicit* meta-training; here ICL is *emergent*. Two candidate causes: novel architecture (transformer) vs distributional qualities of data. Natural data is bursty, Zipfian, with dynamic/context-dependent meaning (polysemy/homonymy/synonymy) — a middle ground between supervised and meta-training data.
- **§2 Experimental Design.** §2.1 Data: Omniglot, 16-token context (8 image-label pairs) + query; labels fixed & recurring (departs from few-shot). Bursty = query class ×3 (+ a distractor class ×3); non-bursty = i.i.d. uniform. §2.2 Model: ResNet + label-embedder → causal transformer (12L, dim 64, 8 heads), softmax CE on query. §2.3 Eval: ICL probe = 4-shot-2-way on holdout classes with randomized $\{0,1\}$ labels (chance ½, scored on 2 labels); IWL probe = training labels, query class absent from context (chance 1/1600).
- **§3 Results.** §3.1 *What promotes ICL:* Burstiness↑ ⇒ ICL↑, IWL↓ (Fig 2); #classes 100→1600→12800 ⇒ ICL↑ (Fig 3, need burstiness too); label multiplicity 1→10 ⇒ ICL↑ (Fig 4); within-class variation↑ ⇒ ICL↑ (Fig 5, harder generalization hurts IWL more). §3.2 *Coexistence:* Zipfian marginal $p\propto x^{-\alpha}$ (Eq 1); sweet spot $\alpha\approx1$ holds both ICL + IWL(common); $\alpha=0$ only ICL; $\alpha$ large only IWL; rare classes never memorized (Fig 6). §3.3 *Architecture:* RNN/LSTM never reach ICL under matched params (Fig 7); transformer also ≥ on IWL (Fig 8).
- **§4 Discussion.** Data properties (burstiness, class number/rarity, dynamic meaning) promote ICL; architecture matters (transformer ≫ recurrent) but is insufficient alone. ICL/IWL "bias" framing (neither is "correct" on the ambiguous probes). Implications: mechanistic account of LLM ICL; counters "LLMs don't do genuine ICL" narrative (success on holdout classes); design non-language datasets with structured/non-uniform distributions; ties to complementary-learning-systems theory (neocortex=weights, hippocampus=context window) and infant statistical learning. Non-uniformity is dual: it hurts supervised/RL but *induces* ICL.
- **Appendix A** training details (500k steps, 16 TPU cores, Adam, warmup→inv-sqrt LR, 3–5 seeds). **B** could extend to novel labels (currently only novel classes). **C** recurrent-vs-transformer details, parameter counts, in-context eval on trained classes (~similar, slightly higher), multi-class eval (same patterns; Zipf-1 models output context labels less often but still above chance).

---
