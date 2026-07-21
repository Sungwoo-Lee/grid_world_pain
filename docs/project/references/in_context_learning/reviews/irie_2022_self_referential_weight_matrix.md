> **Per-paper review — in-context-learning corpus, paper 33 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§33); content is identical. Manifest: [[in_context_learning_sources]].

# 33. Irie et al. 2022 — A Modern Self-Referential Weight Matrix That Learns to Modify Itself

**PDF:** `docs/project/references/in_context_learning/sources/Irie et al. 2022 - A Modern Self-Referential Weight Matrix That Learns to Modify Itself.pdf`
· ICML 2022 · arXiv 2202.05780

## Phase 1: Foundational Overview (Undergraduate-Level)

**The premise.** "A network's weight matrix is its program." Ordinary training freezes that program after
training. But the world keeps changing after training stops. A **self-referential weight matrix (SRWM)**
is a single weight matrix that keeps *rewriting itself* at runtime — in principle it can meta-learn to
learn, and even meta-meta-learn. The 1990s proposed such nets but no one made them work at scale; this
paper builds a practical version using the modern fast-weight-programmer machinery (outer products +
delta rule).

**The key move.** In a normal FWP (previous paper) there are two separate matrices: a fixed *slow* one
that decides how to update, and a *fast* one that gets updated. The SRWM **collapses these into one
matrix** $W_t$ that produces the outputs *and* produces the very key/value/query/learning-rate signals
that it then uses to modify itself. The only thing trained by gradient descent is the *initial* matrix
$W_0$; after that, $W_0$ has to encode its own self-modification algorithm.

**Key findings.** (i) On standard few-shot image classification (Omniglot, Mini-ImageNet) the SRWM is
competitive with the plain Delta Net and with generic sequence-model baselines like SNAIL — a "single
self-modifying matrix works about as well as separate slow+fast nets." (ii) In a **sequential multi-task**
setting — feed a stream of Omniglot images, then suddenly switch to Mini-ImageNet — the SRWM *adapts
itself on the fly* and beats the Delta Net (53.3 vs 50.4% total on the second task), showing genuine
runtime self-adaptation. (iii) In **multi-task RL on ProcGen** (6 procedurally-generated games trained
jointly) the SRWM is competitive/strong, and a "Fake SR" ablation (SRWM with the self-modification turned
off) isolates the benefit of self-modification.

**Initial takeaway.** The SRWM is the most extreme point on the shard's spectrum: not "generate a target
network's weights" but "a network that programs *itself*." For the project it is the strongest existing
demonstration that a *single self-modifying recurrent weight memory* can serve as the adaptive core of an
IMPALA-style RL agent in a procedurally-varying environment — the closest published analog to an agent
that must re-tune its own behaviour to shifting contexts without external gradient steps.

## Phase 2: Graduate-Level Deep Dive

**Recap: the DeltaNet backbone.** The SRWM is derived by making the DeltaNet's slow weight matrix fast /
self-referential. DeltaNet transforms $x_t\in\mathbb R^{d_{\text{in}}}$ to $y_t\in\mathbb R^{d_{\text{out}}}$:
$$
k_t,v_t,q_t,\beta_t=W_{\text{slow}}x_t,\qquad
\bar v_t=W_{t-1}\phi(k_t),
$$
$$
W_t=W_{t-1}+\sigma(\beta_t)\big(v_t-\bar v_t\big)\otimes\phi(k_t),\qquad
y_t=W_t\phi(q_t),
$$
where $\phi$ is a positive, sum-to-one map (softmax) applied on both write and read for stability, and
$W_{\text{slow}}$ is fixed (gradient-descent trained). Here the programmer $W_{\text{slow}}$ is *separate*
from the programmed $W_t$.

**The self-referential weight matrix.** Now let a *single* matrix $W_{t-1}$ play both roles. Given input
$x_t$, it emits the output **and** its own modification signals:
$$
y_t,k_t,q_t,\beta_t=W_{t-1}\phi(x_t),
$$
$$
\bar v_t=W_{t-1}\phi(k_t),\qquad
v_t=W_{t-1}\phi(q_t),
$$
$$
W_t=W_{t-1}+\sigma(\beta_t)\big(v_t-\bar v_t\big)\otimes\phi(k_t).
$$
The terminology (from Schmidhuber's original SRWM papers): $k_t$ is the **modifier-key** — the key whose
stored value must change; $q_t$ is the **analyser-query** — fed back into $W_{t-1}$ to retrieve the new
value $v_t$ to associate with the modifier-key. So the matrix *interrogates itself* ($v_t=W_{t-1}\phi(q_t)$)
to decide what to write, then delta-corrects itself. The output block $W^y_{t-1}$ has $d_{\text{out}}$ rows;
the key/query/beta blocks have $d_{\text{in}}$ (and 1) rows, so
$W_t\in\mathbb R^{(d_{\text{out}}+2 d_{\text{in}}+1)\times d_{\text{in}}}$, and the value vectors
$v_t,\bar v_t$ inherit that same row dimension so the outer-product update is shape-consistent.

**Only $W_0$ is trained.** This is the defining property: "the initial values of the SRWM $W_0$ are the
only parameters in this layer trained by gradient descent." Everything downstream — how it updates, how
fast ($\beta_t$), what it writes — is *emergent* from $W_0$ acting on the input stream. In practice they
expand the output to generate **four separate learning rates** $\beta_t\in\mathbb R^4$, one for each
sub-matrix $[W^y,W^q,W^k,W^\beta]$, and use multi-head computation.

**Why "self-referential" ⊃ "fast-weight."** Because the SRWM adapts *the way it adapts itself* per task
(the update-generating weights are themselves modified), it can specialize its learning algorithm to each
task in a multi-task stream — something the plain DeltaNet (which uses one fixed $W_{\text{slow}}$ for all
tasks) structurally cannot. This is the mechanistic reason the SRWM beats DeltaNet in the sequential
multi-task experiment but merely ties it in single-task few-shot.

**Trainability.** Despite the recursion, training uses ordinary truncated backprop-through-time — the
additive structure of the delta update ($W_t=W_{t-1}+\Delta_t$) keeps gradients tractable over a 50-step
span (the same span used for the RL agents). No second-order / bilevel optimization (contrast MAML).

**Experiments, mechanism-first.**
- *Few-shot (Table 1).* Synchronous-label episode (image+label fed together for the first $NK$ tokens,
  query fed without label). SRWM 97.4% Omniglot 1-shot, 47.0/61.4% Mini-ImageNet 1/5-shot — ≈ DeltaNet;
  the HyperTransformer (paper #5) is stronger (53.8/67.1) but is few-shot-specialized, whereas SRWM is a
  generic sequence model usable in RL.
- *Sequential multi-task (Fig 4, Table 2).* Delayed-label setting (label of input $t$ arrives at step
  $t{+}1$); concatenate an Omniglot segment then a Mini-ImageNet segment. Accuracy drops at the switch
  (step ~74) then recovers as the SRWM re-programs itself. SRWM > DeltaNet on the harder second task,
  demonstrating runtime adaptation. Large batch size was the critical hyperparameter (else the model
  collapses to a zero-shot heuristic).
- *Multi-task RL on ProcGen (Fig 5, §4.3).* Jointly train 6 easy-distribution games (Bigfish, Fruitbot,
  Maze, Leaper, Plunder, Starpilot), IMPALA/Torchbeast, 48 actors, 15-layer conv stem, memory module =
  SRWM (2 layers, hidden 128) vs LSTM / DeltaNet / feed-forward / **Fake SR** (self-modification removed —
  only the $y$ output kept). The shared $W_0$ is common to all tasks/episodes; the *effective* weight
  matrix is a function of the episode-specific input stream (Fig 5 caption). SRWM states reset only at
  episode boundaries. A memory-distribution 4-game partial-observability experiment (App C.1) confirms the
  effect.

**Project relevance.** This is the shard's headline for RL: a self-modifying weight memory as the adaptive
core of an actor-critic in a procedurally-generated environment with clean train/test level splits — the
methodological template one would copy to test "does an agent whose internal weights re-tune themselves to
context outperform a fixed-recurrent baseline" in the grid world. The **Fake-SR ablation is the exact
control design** the project should emulate (modulation-on vs modulation-off, same parameter budget) when
claiming a self-modification / modulation benefit.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** WM = program (Schmidhuber 1990); frozen after training; environments keep evolving
  → want autonomous self-updating programs. Revisit the '90s SRWM using modern FWP/delta-rule/linear-
  Transformer tools. Evaluate in (a) few-shot SL, (b) sequential multi-task few-shot, (c) multi-task RL
  (ProcGen).
- **§2 Background on FWPs.** DeltaNet (Eqs.1–4): $W_{\text{slow}}$ emits $k,v,q,\beta$; delta update with
  read-back $\bar v_t=W_{t-1}\phi(k_t)$; $\phi$ = softmax for stability; multi-head; used as self-attention
  replacement in a Transformer with FF/layernorm/residuals. Slow programmer separate from fast programmed
  net.
- **§3 A Modern SRWM.** Self-training via self-invented key/value patterns + learning rates. SRWM
  $W_{t-1}\in\mathbb R^{(d_{\text{out}}+2d_{\text{in}}+1)\times d_{\text{in}}}$ produces
  $[y_t,q_t,k_t,\beta_t]$ (Eqs.5–8). Modifier-key $k_t$, analyser-query $q_t$. Only $W_0$ trained by GD.
  Extend to "3D+4" for four $\beta_t$; multi-head; App A full spec. Can replace any WM; main model =
  DeltaNet with Eqs.1–4 replaced by Eqs.5–8; App C.1 variant replaces only DeltaNet's slow matrix by SRWM.
- **§4 Experiments.**
  - **§4.1 Standard few-shot.** N-way K-shot episodes, sequential (Santoro/Hochreiter) approach;
    synchronous-label setting (Mishra/SNAIL, Fig 2). Table 1: SRWM ≈ DeltaNet; competitive with SNAIL,
    MAML, fwCNN-Hebb; HyperTransformer strongest. Shared Conv-4-32 / Conv-64 backends.
  - **§4.2 Sequential multi-task adaptation.** Delayed-label setting (Fig 3). Concatenate
    Omniglot+Mini-ImageNet segments, alternate order, randomized lengths. Harder to train; large batch
    size critical. Fig 4: recovery after task switch. Table 2: SRWM > DeltaNet (53.3 vs 50.4% total on
    Mini-ImageNet-second).
  - **§4.3 Multi-task RL (ProcGen).** 6 easy games jointly (Fig 5), IMPALA/Torchbeast, 48 actors,
    15-layer conv, memory module swap; baselines LSTM/DeltaNet/FF/Fake-SR; 200 train levels, 3×200 test
    splits; 300M steps; BPTT span 50; states reset at episode boundary. App C.1: 4 memory-distribution
    games (partial observability).
- **§5/§6 Related work & discussion.** Self-modifying nets lineage (Schmidhuber '92/'93; Finn MAML;
  Hochreiter learning-to-learn); recursive self-improvement framing.

---
