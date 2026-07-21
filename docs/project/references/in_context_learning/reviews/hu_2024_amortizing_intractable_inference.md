> **Per-paper review — in-context-learning corpus, paper 7 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§7); content is identical. Manifest: [[in_context_learning_sources]].

# 7. Hu et al. 2024 — Amortizing Intractable Inference in Large Language Models

**Venue:** ICLR 2024. **Group:** Mila – Quebec AI Institute / Université de Montréal (+ Oxford). Authors: Edward J. Hu*, Moksh Jain*, Eric Elmoznino, Younesse Kaddar, Guillaume Lajoie, Yoshua Bengio, Nikolay Malkin. (Same neighbourhood as papers 1–2 via Elmoznino/Lajoie; adds the Bengio/Malkin GFlowNet line.)
**PDF:** `docs/project/references/in_context_learning/sources/Hu et al. 2024 - Amortizing intractable inference in large language models.pdf`
**Code:** https://github.com/GFNOrg/gfn-lm-tuning

## Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — the core problem.** A large language model (LLM) is trained to do one thing: predict the next token given the tokens before it (left-to-right). That makes *forward* sampling easy — start with a prompt, generate a continuation. But an enormous number of genuinely useful tasks are *not* left-to-right. **Infilling** (fill the gap between a known beginning and a known ending), **constrained generation** (produce text that satisfies some property), and — the paper's headline case — **chain-of-thought reasoning** (find the intermediate reasoning steps that best explain how a question leads to its answer) all require sampling from a *different* conditional distribution than the one the LLM was trained on. These other conditionals are **intractable**: you can *score* any candidate (evaluate its probability under the LLM), but you cannot cheaply *sample* from the posterior one token at a time. This paper's goal is to give LLMs a principled way to sample from these intractable posteriors.

**Key findings — the main result.** The authors frame chain-of-thought reasoning as a **latent-variable model**: an input `X` (question) generates an output `Y` (answer) through a hidden reasoning chain `Z`. Answering well means sampling `Z` from the **Bayesian posterior** `p(Z | X, Y) ∝ p_LM(X Z Y)` — the reasoning chains that make the correct answer likely. Rather than the usual tools (MCMC, which is slow and hard to tune for text; or reward-maximizing RL like PPO, which collapses onto a few high-reward modes), they **fine-tune the LLM itself to become an amortized sampler** of this posterior, using **GFlowNets** (Generative Flow Networks) — a class of diversity-seeking RL algorithms that train a policy to generate objects with probability *proportional to a reward*, rather than to maximize that reward. The reward here is just the LLM's own joint score `p_LM(X Z Y)`, so no external labels for `Z` are needed. A striking motivating demo: asked to "generate a random integer 0–100," a base LLM is wildly non-uniform and PPO barely fixes it (because rewarding all valid numbers equally gives zero policy gradient), while GFlowNet fine-tuning matches the uniform target almost exactly (KL from 3.37 down to ~1e-4). On real tasks the payoff is large: **+10.9%** over supervised fine-tuning on subjectivity classification with only 10 labelled examples, and it **outperforms supervised fine-tuning and PPO by 63%** on multi-step integer arithmetic with tool use — with especially big gains **out-of-distribution** (longer arithmetic expressions than seen in training).

**Initial takeaway.** The connection to in-context learning is conceptual and important: this paper shows that the *same* reasoning an LLM does "in context" (produce a chain of thought, then answer) can be recast as **Bayesian posterior inference over a latent variable**, and that you can **amortize** that inference into the weights of the model itself. The GFlowNet-tuned policy `q(Z | X)` is literally a learned, reusable inference network — it generalizes to unseen inputs `X` and, because it captures the *full diversity* of the posterior (not one mode), it enables **Bayesian model averaging**: sample several reasoning chains, answer with each, take a majority vote. Diversity is not a nicety here — it is what drives the out-of-distribution robustness and defends against reward misspecification (the "overoptimization" failure mode of RLHF-style tuning). This is the corpus's clearest statement of "ICL / reasoning = amortized Bayesian inference," and it operationalizes it as a training algorithm rather than a post-hoc interpretation.

## Phase 2: Graduate-Level Deep Dive

### 3.1 Intractable conditionals in autoregressive LLMs

An autoregressive LLM factorizes a sequence distribution as ordered conditionals,

$$
p(w_{1:N}) = p(w_1)\,p(w_2\mid w_1)\cdots p(w_N\mid w_{1:N-1}),
$$

which makes *left-to-right* sampling trivial but *any other* conditional intractable. The paper catalogs three intractable families:
- **Tempered / contrastive sampling:** $q(Z\mid X)\propto p_{LM}(XZ)^{1/T}$ with $T<1$ (peaky, high-likelihood continuations), or the contrastive $q(Z\mid X)\propto p_{LM}(XZ)^\alpha p_{LM}(Z)^\beta$ ($\beta<0$) that down-weights generic continuations.
- **Infilling / reverse generation:** $q(Z\mid X,Y)\propto p_{LM}(XZY)$ with $X,Y$ fixed (reverse generation is the special case $X=\varnothing$).
- **Constrained generation:** $q(Z)\propto p_{LM}(Z)\,c(Z)$ for an external constraint/classifier $c$.

All are "score-easy, sample-hard": evaluating $p_{LM}$ on a full sequence is one forward pass, but the token-wise conditionals needed to *sample* from the posterior are not available.

### 3.2 Chain-of-thought as latent-variable inference

The core reframing. For a question–answer pair $(X,Y)$, the marginal likelihood decomposes by summing over latent reasoning chains $Z$ (Eq. 1):

$$
p(Y\mid X) = \sum_{Z} p_{LM}(ZY\mid X) = \sum_{Z} p_{LM}(Y\mid XZ)\,p_{LM}(Z\mid X),
$$

with $p_{LM}(Z\mid X)$ the **conditional prior** over reasoning chains and $p_{LM}(Y\mid XZ)$ the **likelihood** of the answer given the chain. Reasoning = sampling from the **posterior** (Eq. 2):

$$
p_{LM}(Z\mid X,Y) = \frac{p_{LM}(XZY)}{\sum_{Z'} p_{LM}(XZ'Y)} \;\propto\; p_{LM}(XZY).
$$

The LVM is useful precisely when the marginal $p_{LM}(Y\mid X)$ is *harder* to model than the pieces $p_{LM}(Z\mid X)$ and $p_{LM}(Y\mid XZ)$ — i.e. a hard inference is broken into a chain of easier ones. The normalizing sum over $Z'$ (all possible reasoning chains) is what makes the posterior intractable.

**Learning the model too — variational EM.** Beyond sampling $Z$, one can fine-tune $p_{LM}$ to maximize the data likelihood of $(X,Y)$ under the LVM. Because $p_{LM}(X,Y)=\sum_Z p_{LM}(XZY)$ is intractable, use **variational EM**:
- **E-step:** draw $Z\sim p_{LM}(Z\mid X,Y)$ from the amortized posterior sampler (the GFlowNet).
- **M-step:** maximize $\mathbb{E}_{Z\sim p_{LM}(Z\mid X,Y)}\big[\log p_{LM}(XZY)\big]$ w.r.t. the LLM parameters.

Iterating = amortized inference (learn to sample the chain of thought) + supervised fine-tuning on self-generated chains. This is exactly the "+ Supervised fine-tuning" row in the subjectivity experiment.

### 3.3 GFlowNets as amortized samplers — the SubTB objective and its derivation

A GFlowNet learns a policy that samples terminal objects $Z=z_1z_2\cdots z_n\top$ (with stop symbol $\top$) with probability **proportional to a reward** $R:\mathcal{Z}\to\mathbb{R}_{>0}$. The generative process mirrors autoregressive generation: at step $i$ sample $z_i\sim q_{GFN}(z_i\mid z_{1:i-1})$, append, repeat until $\top$. The marginal likelihood of terminating at $Z$ is

$$
q^\top_{GFN}(Z) = \prod_{i=1}^{n} q_{GFN}(z_i\mid z_{1:i-1})\, q_{GFN}(\top\mid z),
$$

and the training goal is $q^\top_{GFN}(Z)\propto R(Z)$. **Setting the reward $R(Z)=p_{LM}(XZY)\propto p_{LM}(Z\mid X,Y)$ makes the converged GFlowNet a sampler of the target posterior** — this is the crux that ties GFlowNets to the LVM of §3.2.

**Derivation of the objective (Appendix A.2 → Eq. 3).** Start from the general **subtrajectory balance (SubTB)** constraint (Madan et al. 2023) for a partial trajectory $\tau = s_m\to\cdots\to s_n$, with forward policy $P_F$, backward policy $P_B$, and state-flow function $F$ (Eq. 4):

$$
\mathcal{L}_{SubTB}(Z;\theta) = \left( \log \frac{F(s_m)\prod_{i=m}^{n-1} P_F(s_{i+1}\mid s_i)}{F(s_n)\prod_{i=m}^{n-1} P_B(s_i\mid s_{i+1})} \right)^2 .
$$

Two simplifications specialize this to autoregressive text generation:
1. **Tree-structured state space.** Left-to-right generation in a fixed order means each state has a *unique* parent, so the backward policy is trivial: $P_B(s\mid s')=1$. The denominator product drops out.
2. **Every state is terminable** (Deleu et al. 2022). At convergence $R(s_n^\top)=F(s_n)\,P_F(\top\mid s_n)$, so substitute $F(s_n)=R(s_n^\top)/P_F(\top\mid s_n)$. This **eliminates the separately-parameterized flow function** $F$ — the only learned object is the forward policy $P_F \equiv q_{GFN}$.

Summing the resulting balance condition over all partial trajectories $0\le i<j\le n$ with equal weight ($\lambda=1$) gives the final per-sequence objective (Eq. 3):

$$
\mathcal{L}(Z;\theta) = \sum_{0\le i<j\le n} \left( \log \frac{R(z_{1:i}\top)\prod_{k=i+1}^{j} q_{GFN}(z_k\mid z_{1:k-1})\, q_{GFN}(\top\mid z_{1:j})}{R(z_{1:j}\top)\, q_{GFN}(\top\mid z_{1:i})} \right)^2 .
$$

For sequence generation this SubTB objective is **equivalent to the path-consistency objective** (Nachum et al. 2017) in maximum-entropy RL (Haarnoja et al. 2017) — placing GFlowNet fine-tuning squarely in the max-ent-RL family, but with the distinguishing goal of *matching* the full reward distribution rather than maximizing expected reward.

**Off-policy training.** Because Eq. 3 can be driven to zero for *all* trajectories simultaneously (given capacity), gradients may use trajectories from *any* full-support behaviour policy. The mini-batch mixes three sources: (1) the current policy $q_{GFN}$, (2) a tempered version of it, and (3) a **replay buffer** of past trajectories — crucial for exploring the combinatorially large sequence space. Fine-tuning is done with **LoRA** (low-rank adaptation) for hardware efficiency, not full fine-tuning.

### 3.4 Amortization, conditioning, and Bayesian model averaging (Table 1)

The GFlowNet policy is parameterized as an autoregressive LM that generates $Z$ token-by-token. Making $X$ (and optionally $Y$) an *input* to the policy is what turns per-instance inference into **amortized** inference that generalizes to unseen inputs. Two conditioning regimes:
- **Condition on $X$ only** — for reasoning/classification where each $X$ has a single correct $Y$ and $Y$ is unknown at test time. The policy $q_{GFN}(Z\mid X)$ is initialized as a copy of $p_{LM}$ conditioned on prefix $X$, then GFlowNet-tuned. Sampling $Z$ is an **inverse problem**: infer $Z$ given conditional prior $p_{LM}(Z\mid X)$ and observation $Y$ under likelihood $p_{LM}(Y\mid XZ)$. Prediction for a new $X$ is **Bayesian model averaging**: draw $Z\sim q_{GFN}(Z\mid X)$, then $Y\sim p_{LM}(Y\mid XZ)$, aggregate (majority vote). In this view the GFlowNet is a Bayesian model where $Z$ are conditionally-sampled "parameters" transforming $X$ into $Y$ — analogous to an LM cascade / deep language network.
- **Condition on both $X$ and $Y$** — for infilling, where the map $X\to Y$ is one-to-many and $Y$ is available at test time; here $Z$ itself is the object of interest, so the policy $q_{GFN}(Z\mid X,Y)$ conditions on a prompt containing both.

### 3.5 Empirical results — why distribution-matching beats mode-seeking

- **Random-number demo (§2, Fig. 2):** the sharpest illustration. Rewarding all valid integers equally gives an *expected policy gradient of zero*, so PPO cannot un-skew the pretraining bias (Benford-law-like preference for numbers starting with '1'); it only learns validity (95.8% valid, still skewed). GFlowNet matches the uniform target (100% valid, KL $3.37\to 9.75\times10^{-5}$). Distribution-matching succeeds exactly where reward-maximization is degenerate.
- **Sentence continuation (§4.1, Fig. 3):** reward $R(Z)=p_{LM}(Z\mid X)^{1/T}$, $0<T<1$. GFlowNet samples higher-max-likelihood *and* more diverse continuations than diverse beam search — even when beam search is given **5× the compute** — because it amortizes over prompts in a single learned pass.
- **Story infilling (§4.2, Table 2):** ROCStories, generate the 4th sentence (the turning point) given sentences 1–3 and 5; $q_{GFN}(Z\mid X,Y)$ beats prompting and supervised fine-tuning on BERTScore/BLEU-4/GLEU-4/GPT-4-eval by accounting for the ending while generating.
- **Subjectivity classification (§4.3, Table 3):** low-data regime; latent rationale $Z$. GFlowNet fine-tuning beats supervised fine-tuning by large margins with few labels (e.g. **71.4% vs 64.3% at 10 examples**), and one EM step ("+ supervised fine-tuning") often helps further. At test time: sample 10 rationales, answer with each, majority-vote.
- **Arithmetic with tool use (§4.4, Table 4):** GPT-J 6B, calculator restricted to two-term expressions (planning under a limited tool). Trained on 3–4 operands; evaluated in-distribution (3–4) and OOD (5 operands). GFlowNet: **95.2% / 75.4% / 40.7%** vs supervised fine-tuning 72.1 / 19.6 / 12.8 and PPO 30.6 / 13.7 / 5.6. PPO fails via **reward over-optimization** — high-reward-but-spurious sequences that aren't even valid tool calls (the misspecified-reward pathology). Distribution-matching avoids mode collapse and is robust to reward misspecification.

### 3.6 Significance and relation to the corpus

This paper is the corpus's **amortized-inference** anchor. Where paper 1 says next-token prediction *is* compression/Occam and paper 2 says clean latent inference isn't enough for OOD prediction, paper 3 provides a *constructive training method* for the latent-variable / Bayesian view: amortize the intractable posterior over reasoning chains into the LLM's weights via a diversity-seeking objective. The throughline "ICL / reasoning = (amortized) Bayesian inference over latents" is here made an algorithm with measurable wins, and it explicitly connects to Bayesian model averaging — the same posterior-predictive machinery invoked in Groups A/B. For the project, the transferable ideas are: (i) treat intractable posterior sampling as amortizable into a network; (ii) prefer distribution-matching (GFlowNet/max-ent) over reward-maximizing RL when *diversity* and *robustness to reward misspecification* matter; (iii) diversity of latent samples directly buys OOD generalization via averaging. **Limitations** the authors flag: experiments ≤6B params; on-policy exploration in complex-latent problems remains open; the method improves *inference*, not the LLM's underlying knowledge — so hallucination/miscalibration tied to knowledge representation are not addressed.

## Appendix: Section-by-Section Backbone

- **Abstract.** LLMs' next-token training limits tractable querying to left-to-right sampling; many tasks (continuation, infilling, constrained generation) need intractable-posterior sampling. Solution: amortized Bayesian inference via GFlowNet fine-tuning (diversity-seeking RL). Recast chain-of-thought as latent-variable modeling → data-efficient adaptation to multi-step reasoning + tool use.
- **§1 Introduction.** LLMs as knowledge stores queryable only by prefix-conditioned sampling. Reasoning as probabilistic inference; CoT as intractable posterior inference: $p(Y|X)=\sum_Z p_{LM}(Y|XZ)p_{LM}(Z|X)$ (Eq. 1); posterior $p_{LM}(Z|X,Y)\propto p_{LM}(XZY)$ intractable. MCMC (slow, hard proposals) and reward-max RL/PPO (mode collapse, overoptimization) inadequate. Amortized inference via GFlowNets: sample $\propto$ reward $p_{LM}(XZY)$. Contributions: (1) general amortized-sampling algorithm for intractable LLM posteriors; (2) probabilistic CoT fine-tuning; (3) results on continuation, NL reasoning, arithmetic+tools, infilling.
- **§2 Motivating example — random numbers.** Sample from a target given an unnormalized density. Base GPT-J skewed (50.5% valid); PPO learns validity but stays skewed (equal reward → zero gradient); GFlowNet matches uniform (KL $3.37\to 9.75\text{e-}5$). Distribution-matching vs reward-maximization.
- **§3 Fine-tuning LLMs to sample from intractable distributions.**
  - **§3.1 Problem.** Ordered-conditional factorization; intractable families — tempered/contrastive $q(Z|X)\propto p_{LM}(XZ)^{1/T}$; infilling/reverse $q(Z|X,Y)\propto p_{LM}(XZY)$; constrained $q(Z)\propto p_{LM}(Z)c(Z)$. Table 1 object glossary.
  - **§3.2 Reasoning through latent variables.** Posterior Eq. 2; LVM useful when marginal harder than pieces; variational EM (E-step: sample $Z\sim$ amortized posterior; M-step: maximize $\mathbb{E}_Z\log p_{LM}(XZY)$).
  - **§3.3 Amortized inference with GFlowNet objectives.** GFlowNet basics; $q^\top_{GFN}(Z)=\prod q_{GFN}(z_i|z_{1:i-1})q_{GFN}(\top|z)$; goal $q^\top_{GFN}\propto R$; SubTB objective (Eq. 3) ≡ path-consistency/max-ent RL; off-policy training (policy + tempered + replay buffer); reward $R(Z)=p_{LM}(XZY)$; conditioning on $X$ (reasoning, BMA prediction) vs $X,Y$ (infilling); LoRA fine-tuning.
- **§4 Empirical results.**
  - **§4.1 Sentence continuation** — OpenWebText, GPT-2 XL 1.5B; reward $p_{LM}(Z|X)^{1/T}$; beats diverse beam search on likelihood+diversity even at 5× compute (Fig. 3).
  - **§4.2 Infilling stories** — ROCStories, GPT-2 Large; $q_{GFN}(Z|X,Y)$; beats prompting + supervised FT on BERTScore/BLEU/GLEU/GPT-4 (Table 2).
  - **§4.3 Subjectivity classification** — SUBJ, instruct-GPT-J 6B, low-data; latent rationale; 71.4% vs 64.3% (SFT) at 10 examples; +EM step (Table 3).
  - **§4.4 Arithmetic step-by-step** — GPT-J 6B + calculator tool (2-term limit); train 3–4 operands, test OOD 5; GFlowNet 95.2/75.4/40.7 vs SFT and PPO; PPO overoptimizes misspecified reward (Table 4).
- **§5 Further related work.** Sampling intractable marginals (MCMC, SMC, Gibbs via masked LMs); GFlowNets as variational inference for structured Bayesian posteriors; CoT reasoning (self-consistency aggregation, STaR fine-tuning on successful chains, MCMC CoT).
- **§6 Conclusion.** GFlowNet fine-tuning gives better fidelity–diversity trade-off, sample efficiency, and generalization than MLE or reward-max RL; converts compute into test-time performance without extra data. Future: universal reasoner $q(Z|X)$ shared across tasks; better base LLM; epistemic-uncertainty quantification via multiple samples; richer (non-left-to-right) latent generative processes. **Limitations:** ≤6B params; on-policy exploration open; improves inference not knowledge (hallucination/miscalibration untouched).
- **Appendix A.1 — Glossary of RL for LLMs.** RL, policy, reward (= unnormalized posterior for GFlowNets), distribution-matching, policy-gradient methods.
- **Appendix A.2 — Learning objective.** General SubTB (Eq. 4) with $P_F$, $P_B$, flow $F$; for left-to-right generation $P_B=1$ (tree) and terminable-state substitution $F(s_n)=R(s_n^\top)/P_F(\top|s_n)$ eliminate the flow function → only $q_{GFN}$ learned; sum over subtrajectories ($\lambda=1$) → Eq. 3.
- **Appendices B–E.** Per-task details: sentence-continuation granularity rationale; infilling setup; subjectivity prompts (Table D.1/D.2); arithmetic replay-buffer seeding (50 $(X,Z,Y)$ demos + 1000 $(X,Y)$), tool-use mechanics, and PPO failure illustrations.

---
