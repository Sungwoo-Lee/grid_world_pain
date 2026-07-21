> **Per-paper review — in-context-learning corpus, paper 25 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§25); content is identical. Manifest: [[in_context_learning_sources]].

# 25. Mittal et al. 2025 — Iterative Amortized Inference: Unifying ICL and Learned Optimizers

**PDF:** `docs/project/references/in_context_learning/sources/Mittal et al. 2025 - Iterative Amortized Inference - Unifying ICL and Learned Optimizers.pdf`
**Venue:** arXiv (Oct 2025). **Authors:** Sarthak Mittal, Divyat Mahajan, Guillaume Lajoie, Mohammad Pezeshki (Mila / Université de Montréal / FAIR at Meta).

## Phase 1: Foundational Overview (Undergraduate-Level)

**The question in plain terms.** Many modern methods "learn to solve new tasks fast": meta-learning (MAML), in-context learning (ICL), prompt tuning, hypernetworks, and learned optimizers. They *look* different but share a goal — reuse structure across tasks so a new task needs only a little task-specific adaptation. This paper gives **one equation that contains all of them** and a **taxonomy** with three regimes, then proposes a scalable extension.

**The running example.** Model the motion of an object on many planets. Newton's laws are shared across all planets (amortize once); only the gravitational constant differs per planet (adapt per task). Every method is just a choice of *what you amortize* (the shared bit) versus *how you read the task data at inference* (the adapt bit).

**Key findings.**
1. **Unified formula.** Every method is $f_\gamma\big(x, g_\phi(\mathcal D_T)\big)$: a **shared predictor** $f_\gamma$ (task-invariant) applied to a query $x$, using a **task-adaptation function** $g_\phi$ that maps the task's support data $\mathcal D_T$ into task-specific information $\theta_T$ (weights, prompts, latents, or context tokens).
2. **Three regimes** (Table 1):
   - **Parametric** — $f$ is fixed (a known likelihood form), $g_\phi$ learns to infer its parameters. Hypernetworks and learned optimizers live here. Task info is *externalized* into explicit parameters.
   - **Implicit** — $g$ is trivial (identity/subsample), $f_\gamma$ is a transformer that jointly ingests context + query. **This is ICL / PFNs.** Task adaptation is *internalized* in the forward pass.
   - **Explicit** — both $f_\gamma$ and $g_\phi$ learned: $g_\phi$ compresses the whole dataset into a low-dim embedding (the "gravitational constant"), $f_\gamma$ is the learned likelihood ("laws of physics"). Neural processes live here.
3. **The limitation.** Implicit/ICL is capped by context length; parametric/explicit pooling throws away fine detail. Neither scales to *large* task datasets.
4. **The fix — iterative amortized inference.** Borrow SGD's trick: don't cram the whole dataset into one forward pass; instead **refine a solution step-by-step over mini-batches**. A learned sequence model $h_\phi$ takes the current state + a fresh mini-batch and outputs a better state. This unifies learned optimizers (which refine over gradient mini-batches) with ICL/hypernetworks (which refine over observation mini-batches), and lets either or both signals drive the update.
5. **Empirics.** More refinement steps consistently help across regression, MNIST/FMNIST/ImageNet classification (incl. OOD → CIFAR), topological-order prediction, and generative modeling (GMM, alphabets). Using **observations directly often beats gradients-only** at low dimension; gradients help more at high dimension / few observations. Parametric beats explicit (joint learning is hard). For implicit, keeping the recurrent state at the **logits** (pre-softmax) works best.

**Initial takeaway.** This is the *organizing map* of the whole shard. It places "attention/ICL" (implicit), "hypernetworks" (parametric), and "neural processes" (explicit) as three faces of one $f_\gamma(x, g_\phi(\mathcal D_T))$ template, and shows the in-context adaptation is best understood as **iterative optimization**. **Bridge to the project:** it tells you exactly where a FiLM/hypernetwork conditioning module sits (parametric amortization: a learned $g_\phi$ generating the conditioning parameters $\theta_T$) versus where in-context conditioning sits (implicit), and frames both as instances of amortized inference — useful when deciding whether the project's conditioning should externalize a task code (interpretable, low-dim) or internalize it in a forward pass (expressive, expensive).

## Phase 2: Graduate-Level Deep Dive

### 3.1 The single-task baseline and its two failures

Standard learning minimizes true risk, approximated by empirical risk over an i.i.d. dataset $\mathcal D_T$:

$$
\theta_T^* = \arg\min_\theta\ \mathbb E_{x,y\sim p_T}\big[\mathcal L(y, f(x,\theta))\big],\qquad
\hat\theta_T = \arg\min_\theta\ \sum_{(x,y)\in\mathcal D_T}\mathcal L(y,f(x,\theta)). \tag{1–2}
$$

Two failures: (a) no systematic transfer to a *new* task; (b) it ignores cross-task shared structure — data from task $T_2$ carries inductive signal for $T_1$ but is discarded.

### 3.2 Meta-learning, hypernetworks, learned optimizers, PFNs as one family

**MAML** learns a global initialization $\hat\theta_0$ then adapts via an inner optimizer $g_{\theta_0}$:

$$
\hat\theta_0 = \arg\min_{\theta_0}\ \mathbb E_T\,\mathbb E_{x,y,\mathcal D_T}\big[\mathcal L(y,f(x,\theta_T))\big],\quad \theta_T = g_{\theta_0}(\mathcal D_T). \tag{3}
$$

**Hypernetworks / learned optimizers** replace $\theta_0$ by learnable $\phi$: $g_\phi$ is a sequence model producing task parameters — from *observations* (hypernetwork) or from *gradients* (learned optimizer). **PFNs / example-based ICL** instead directly model the conditional predictive distribution without exposing $\theta_T$:

$$
\hat\gamma = \arg\min_\gamma\ \mathbb E_T\,\mathbb E_{x,y,\mathcal D_T}\big[\mathcal L(y, f_\gamma(x,\mathcal D_T))\big]. \tag{4}
$$

**The unification.** All of the above are

$$
\boxed{\;\min_{\gamma,\phi}\ \mathbb E_T\,\mathbb E_{x,y,\mathcal D_T}\Big[\mathcal L\big(y,\ f_\gamma\big(x,\ g_\phi(\mathcal D_T)\big)\big)\Big]\;} \tag{5}
$$

with the empirical support/query split (support $\mathcal D_T^{\text{train}}$ feeds $g_\phi$; loss on $\mathcal D_T^{\text{valid}}$):

$$
\min_{\gamma,\phi}\ \mathbb E_T\Bigg[\sum_{(x,y)\in\mathcal D_T^{\text{valid}}}\mathcal L\big(y, f_\gamma(x,\theta_T)\big)\Bigg],\quad \theta_T=g_\phi(\mathcal D_T^{\text{train}}). \tag{6}
$$

Special cases by choice of $(f,g,\text{split of }\phi\text{ vs }\gamma)$:
- **Supervised learning:** $f$ fixed, $g=$ SGD (solves Eq. 2).
- **MAML:** $f$ fixed, $g_\phi=$ SGD with learnable init $\theta_0\in\phi$.
- **Learned optimizers / hypernetworks:** $f$ fixed, $g_\phi$ a neural net. Learned optimizer is the *iterative* form $g_\phi := h_\phi^{(k)}\circ\cdots\circ h_\phi^{(1)}$, where each $h_\phi^{(i)}$ is a sequence model taking past parameters $\theta^{(0)},\dots,\theta^{(i-1)}$ and their gradients $\nabla^{(j)}=\nabla_\theta\sum_{(x,y)\in B^{(j)}}\mathcal L(y,f(x,\theta))\big|_{\theta^{(j)}}$; hypernetwork is the one-shot observation-to-parameter map.
- **ICL:** $g$ samples observations, $f_\gamma$ is a predictive sequence model with the context as observations (prompt-based ICL is the emergent LLM special case).

**Non-unique decomposition (important subtlety).** $f_\gamma(x, g_\phi(\mathcal D_T))$ does not uniquely split computation between $f$ and $g$: anything $g_\phi$ computes can be folded into $f_\gamma$. The *converse fails* because $g_\phi$ has no access to the query $x$. The convention that makes the split meaningful: **assign all query-independent computation to $g_\phi$, and query-dependent computation to $f_\gamma$.** So $f_\gamma$ sees task info $\mathcal D_T$ only through the structure $g_\phi$ imposes (pooled representation, gradients, or natural language).

### 3.3 The taxonomy (parametric / implicit / explicit)

- **Parametric** ($f$ fixed, $g_\phi$ learnable): likelihood form is *known* (a simulator, or a categorical likelihood with a linear map); $g_\phi$ is the inference procedure discovering optimal parameters (e.g. linear coefficients or a posterior over them). Advantages: can use **gradient information** in addition to $\mathcal D_T$; yields a low-dim, interpretable task representation.
- **Implicit** ($f_\gamma$ trainable, $g$ fixed): a single model ingests query + observations and learns the output form directly (ICL, some PFNs/CNPs). $g$ is identity or subsampling. Hard to localize which activations are task vs. prediction; under assumptions the forward pass recovers algorithmic behavior (gradient descent — von Oswald 2023; causal discovery — Nichani 2024), but this is the posterior-predictive view.
- **Explicit** (both $f_\gamma,g_\phi$ trainable): $g_\phi$ gives a low-dim dataset embedding ("gravity constant"), $f_\gamma$ the shared likelihood ("physics"). Neural processes. Gets dimensionality reduction + gradient use, but pays **non-stationarity** — $f_\gamma$ and $g_\phi$ are coupled and co-adapting.

### 3.4 Iterative amortized inference (Eqs. 7–9)

Motivation: implicit/ICL is context-length bounded; parametric/explicit compress via pooling or gradients (lossy). SGD scales by iterating over mini-batches. So make $g_\phi$ (or $f_\gamma$) an iterative, Markovian, greedy refinement.

**Parametric / explicit — refine the state $\theta$.** With subsampled batches $B^{(i)}_{\text{train}}\subset\mathcal D_T^{\text{train}}$ and learned init $\theta^{(0)}$:

$$
\theta^{(0)}\ \xrightarrow{\ h_\phi(\cdot,\,B^{(0)}_{\text{train}})\ }\ \theta^{(1)}\ \xrightarrow{\ h_\phi(\cdot,\,B^{(1)}_{\text{train}})\ }\ \cdots\ \xrightarrow{\ h_\phi(\cdot,\,B^{(k-1)}_{\text{train}})\ }\ \theta^{(k)}=:\theta_T. \tag{7}
$$

Learned optimizers are the restricted case where $h_\phi$ sees observations *only through gradients*: $h_\phi(\theta,B):=\text{Transformer}_\phi\big(\theta,\ \nabla_\theta\sum_{(x,y)\in B}\mathcal L(y,f_\gamma(x,\theta))\big)$. Because the observation→gradient map is **non-invertible**, gradient-only optimizers cannot recover general schemes (e.g. the closed-form linear-regression solution) that using raw observations can. Here $\theta$ acts as a **recurrent memory** compressing the task; prediction for $x$ given $\theta_T$ is conditionally independent of $\mathcal D_T$.

**Implicit — refine the prediction $\hat y$.** No task parameters are exposed; the recurrent state is tied to the query. $g_\phi$ just subsamples, $f_\gamma=r_\gamma$ a transformer:

$$
\hat y^{(0)}\ \xrightarrow{\ r_\gamma([x,\hat y^{(0)}],\,B^{(0)}_{\text{train}})\ }\ \hat y^{(1)}\ \xrightarrow{\ }\ \cdots\ \xrightarrow{\ r_\gamma([x,\hat y^{(k-1)}],\,B^{(k-1)}_{\text{train}})\ }\ \hat y^{(k)}. \tag{8}
$$

$\hat y^{(0)}$ is a learned init (ideally the marginal/unconditional prediction). Since $\theta_T$ is never exposed, gradient signal is hard to inject.

**Greedy single-step training (Eq. 9).** Following SGD's greediness, train $\phi,\gamma$ on single-step improvements only, stopping gradients from flowing back through prior states $\theta^{(i)}$ or $\hat y^{(i)}$:

$$
\theta^{(i)}\ \xrightarrow{h_\phi(\cdot,B)}\ \theta^{(i+1)}\to\mathcal L\big(y,f_\gamma(x,\theta^{(i+1)})\big)
\quad\text{or}\quad
\hat y^{(i)}\ \xrightarrow{r_\gamma([x,\cdot],B)}\ \hat y^{(i+1)}\to\mathcal L\big(y,\hat y^{(i+1)}\big). \tag{9}
$$

An outer loop produces the $i$-th states; $(x,y)\in\mathcal D_T^{\text{valid}}$ while $B\subseteq\mathcal D_T^{\text{train}}$ (amortizing a learner to minimize *validation* loss). All three paradigms modify causal masking for efficient parallel computation over variable-size mini-batches.

**Optional non-Markovian variant.** Relax to a history window: $h_\phi:\theta_{t-k:t},\nabla_{t-k:t},B_t\to\theta_{t+1}$. Empirically extra history barely helps (except FMNIST pretraining).

### 3.5 Connection to the rest of the shard

This paper is the abstraction layer over Schlag (paper 2) and Schug (paper 1): Schlag's delta-rule fast-weight update $\mathbf W^{(i)}=\mathbf W^{(i-1)}+\beta^{(i)}(\mathbf v^{(i)}-\bar{\mathbf v}^{(i)})\otimes\phi(\mathbf k^{(i)})$ is precisely a *single greedy refinement step* $h_\phi(\theta^{(i)},B^{(i)})$ in the parametric regime, where the "state" $\theta$ is the fast-weight matrix and $\beta$ is a learned learning rate. Schug's per-token hypernetwork code $\theta_T=g_\phi(\mathcal D_T)$ is a *one-step parametric* amortization (generate weights from context). Chen (paper 4) is the extreme parametric endpoint: convert the implicit ICL adaptation into literal explicit weights.

### 3.6 Empirical findings (Tables 2–6, §5)

- **All three regimes:** iterative refinement (more steps) monotonically improves ID and OOD performance across linear regression, MNIST/FMNIST/ImageNet classification, topological order, and generative (GMM/alphabet) tasks. E.g. pretrain on ImageNet → good CIFAR-10 from in-context examples alone.
- **Observations vs. gradients:** at low dimension, raw observations beat gradient-only (learned-optimizer) inference; at high dimension / few observations, gradients give a more reliable search signal. Combining both often best.
- **Parametric > explicit** even though explicit's solution class *contains* parametric's — attributed to the coupled, non-stationary joint optimization of $f_\gamma,g_\phi$.
- **Implicit state choice:** keep the recurrent state at the **logits** (pre-softmax), not the pre-MLP latents or post-softmax probs — being close to prediction helps greedy refinement.
- **Runtime:** a $K$-step method with mini-batch $B$ costs $O(KB^2)$; a one-step model over the same $KB$ data costs $O((KB)^2)$, so iterative amortization is $\sim K\times$ more efficient (and more memory-scalable) at equal data.

## Appendix: Section-by-Section Backbone

- **Abstract.** Amortized learning (meta-learning, ICL, prompt tuning, learned optimizers) reuses cross-task computation. Unified framework: methods differ in *what they amortize* (initializations / learned updates / predictive mappings) and *how they use task data at inference*. Taxonomy: parametric / implicit / explicit. Limitation: poor scaling to large datasets (context length / lossy pooling). Proposal: iterative amortized inference — step-by-step refinement over mini-batches (SGD-inspired), bridging optimization-based meta-learning and forward-pass amortization.
- **§1 Introduction.** Planet/gravity running example. Meta-learners (learned init), LLM ICL (conditioning on observations/instructions), learned optimizers (amortize optimization via gradient-conditioned updates). Two components: (1) shared task-invariant mechanism, (2) task-adaptation function. Three regimes by inductive bias. Contributions: unified framework (§2), taxonomy (§3), stochastic iterative amortization (§4).
- **§2 Unified perspective.** Eqs. 1–2 (true/empirical risk); Eq. 3 (MAML / hypernetwork / learned optimizer as $\theta_T=g_{\theta_0/\phi}(\mathcal D_T)$); Eq. 4 (PFN/ICL conditional predictive); Eq. 5–6 (unified $f_\gamma(x,g_\phi(\mathcal D_T))$). §2.1 special cases (supervised/MAML/learned-opt/hypernet/ICL). Non-unique decomposition (query-independent → $g_\phi$; query-dependent → $f_\gamma$). "Learning the learner": higher-order gradients, first-order approx, BPTT, RL, ES.
- **§3 Taxonomy.** Parametric (fixed $f$, learnable $g_\phi$; hypernets, learned optimizers, parametric-ICL; can use gradients; interpretable low-dim params). Implicit (trainable $f_\gamma$, fixed $g$; ICL/PFN/CNP; forward-pass conditioning; hard to localize task activations). Explicit (both trainable; dataset embedding + learned likelihood; neural processes; non-stationarity cost).
- **§4 Iterative amortized inference.** Motivation (context-length / pooling limits vs. SGD mini-batch scaling). Parametric/explicit refinement of $\theta$ (Eq. 7); learned-optimizer restriction (gradient-only, non-invertible obs→grad map); $\theta$ as recurrent task memory. Implicit refinement of $\hat y$ (Eq. 8), state tied to query. Greedy single-step training with stop-gradient (Eq. 9); validation-loss amortization; modified causal masking. Non-Markovian history variant.
- **§5 Experiments.** Tasks: linear regression, MNIST/FMNIST/ImageNet classification (+CIFAR OOD, Dino-v2 embeddings), topological order (SCM), MoG + alphabet generation (flow-matching). §5.1 consistent gains with steps across parametric (Table 2) / explicit (Table 3) / implicit (Tables 4–6). §5.2 analysis: gradients sub-optimal at low-dim; history barely helps; explicit < parametric (non-stationarity); best implicit state = logits; runtime $O(KB^2)$ vs $O((KB)^2)$ → $K\times$ efficiency.

---
