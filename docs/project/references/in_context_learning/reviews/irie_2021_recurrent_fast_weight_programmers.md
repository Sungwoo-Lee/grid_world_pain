> **Per-paper review — in-context-learning corpus, paper 32 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§32); content is identical. Manifest: [[in_context_learning_sources]].

# 32. Irie et al. 2021 — Going Beyond Linear Transformers with Recurrent Fast Weight Programmers

**PDF:** `docs/project/references/in_context_learning/sources/Irie et al. 2021 - Going Beyond Linear Transformers with Recurrent Fast Weight Programmers.pdf`
· NeurIPS 2021 · arXiv 2106.06295

## Phase 1: Foundational Overview (Undergraduate-Level)

**The setup.** A "linear Transformer" is a Transformer whose softmax attention has been replaced by a
cheaper kernel, giving linear-time, constant-memory sequence processing. A 2021 insight (Schlag et al.,
Katharopoulos et al.) is that a linear Transformer is *exactly* an old idea from the '90s: a **Fast Weight
Programmer (FWP)** — a slow network that continually rewrites the weights of a fast network by adding
rank-one outer products. In existing linear Transformers, both the slow and fast nets are trivially
simple: single feedforward layers. This paper asks: *what if we make them recurrent?*

**The idea in plain terms.** Think of a small "fast" network whose weight matrix $W$ acts like a piece of
scratch memory. At every time step the "slow" network reads the input, produces a key–value pair, and
*writes* it into $W$ via an outer product; it also produces a query that *reads* from $W$ to make the
output. The authors' contribution is to add **recurrence** in two places: (i) make the fast network
itself an RNN or LSTM (so it has its own recurrent hidden state on top of the fast-weight memory), and
(ii) let the slow network's key/value/query depend on the fast net's previous output. These give
**Recurrent Fast Weight Programmers (RFWPs)** — models that behave like both Transformers and RNNs.

**Key findings.** On WikiText-103 language modelling the recurrent variants match or beat the linear
Transformer and even, when run without context truncation, beat a regular Transformer restricted to a
256-token window. On synthetic algorithmic tasks (code execution — track the values of several variables;
sequential ListOps — evaluate nested list operations) the recurrent fast nets are markedly more robust
than the plain Delta Net, especially at higher difficulty. **The result the project should care about:**
dropped in as a replacement for the LSTM in an IMPALA actor-critic on Atari 2600, the small RFWPs deliver
*large* score improvements over the LSTM baseline across many games.

**Initial takeaway.** This paper is the cleanest statement of the "linear attention ⇔ fast-weight
programming" equivalence *and* the one that pushes it into RL. For a project weighing recurrent modulated
architectures for a partially-observable grid world, the Delta RNN / Recurrent Delta Net are directly
relevant: they are constant-memory recurrent sequence models with a *content-addressable, continually
rewritten weight memory* — a plausible substrate for context-dependent behaviour.

## Phase 2: Graduate-Level Deep Dive

**General FWP formulation.** An FWP with trainable slow parameters $\theta_{\text{slow}}$ maps an input
sequence $\{x^{(t)}\}$ to outputs $\{y^{(t)}\}$ through a two-network system:
$$
\theta^{(t)}_{\text{fast}},\,q^{(t)} = \mathrm{SlowNet}\big(\{x^{(j)}\}_{j\le t},\{y^{(j)}\}_{j<t},
\{\theta^{(j)}_{\text{fast}}\}_{j<t},\{q^{(j)}\}_{j<t};\theta_{\text{slow}}\big),
$$
$$
y^{(t)} = \mathrm{FastNet}\big(\{q^{(j)}\}_{j\le t},\{y^{(j)}\}_{j<t};\theta^{(t)}_{\text{fast}}\big).
$$
The slow weights change only by gradient descent (fixed at test time); the fast weights
$\theta^{(t)}_{\text{fast}}$ can change *every step*. Because a weight matrix is too large to emit
directly, the fast weights are generated **incrementally** via an update rule:
$$
z^{(t)},q^{(t)}=\mathrm{SlowSubnet}(\cdots;\theta_{\text{slow}}),\qquad
\theta^{(t)}_{\text{fast}}=\mathrm{UpdateRule}\big(\theta^{(t-1)}_{\text{fast}},z^{(t)}\big).
$$
The update rule is the FWP's "elementary programming instruction."

**Linear Transformer as the additive-outer-product FWP.** With a kernel feature map $\phi$ (absorbed into
$k,q$ notation) the slow net emits key/value/query by fixed projections, writes a rank-one outer product,
and reads with the query:
$$
k^{(t)},v^{(t)},q^{(t)}=W_k x^{(t)},\,W_v x^{(t)},\,W_q x^{(t)},
$$
$$
W^{(t)}=W^{(t-1)}+v^{(t)}\otimes k^{(t)},\qquad
y^{(t)}=W^{(t)}q^{(t)}.
$$
This is the **sum update rule** — purely additive. It is *exactly* linearized self-attention: the fast
weight matrix $W^{(t)}=\sum_{j\le t} v^{(j)}\otimes k^{(j)}$ accumulates the key–value outer products, and
$W^{(t)}q^{(t)}=\sum_{j} v^{(j)}(k^{(j)\top}q^{(t)})$ is attention with unnormalized kernel scores. **This
equivalence is the conceptual keystone of the whole shard.**

**Delta Net — the delta-rule update.** Pure addition saturates memory capacity (keys collide). The Delta
Net (Schlag et al. 2021) replaces the write with a delta-rule correction: first *read back* the value
currently stored at key $k^{(t)}$, then write only the *residual*, scaled by a generated learning rate
$\beta^{(t)}=\sigma(W_\beta x^{(t)})$:
$$
\bar v^{(t)}=W^{(t-1)}k^{(t)},\qquad
W^{(t)}=W^{(t-1)}+\beta^{(t)}\big(v^{(t)}-\bar v^{(t)}\big)\otimes k^{(t)}.
$$
Interpretation: this is one step of online gradient descent on $\tfrac12\|W k^{(t)}-v^{(t)}\|^2$ with
step size $\beta^{(t)}$ — the model *learns to write* by error-correction, not blind accumulation. (This
"delta rule ≈ inner-loop gradient step" reading is the bridge to the [[Hypernetwork]]/ICL-as-GD papers in
§2 of the corpus.)

**Contribution 1 — recurrent fast nets.** Any architecture can be "made fast." A plain RNN layer
$h^{(t)}=\sigma(Wx^{(t)}+Rh^{(t-1)})$ becomes fast by replacing $W,R$ with delta-updated fast matrices
$W^{(t)},R^{(t)}$. The paper's concrete **Delta RNN** keeps the linear-Transformer read but adds a
recurrent fast term:
$$
y^{(t)}=W^{(t)}q^{(t)}+R^{(t)}f\big(y^{(t-1)}\big),\qquad f=\mathrm{softmax},
$$
where $R^{(t)}\in\mathbb R^{d_{\text{out}}\times d_{\text{out}}}$ is a *second* fast weight matrix,
generated by the same delta rule with its own slow weights. The softmax on the recurrent query is
required so keys/queries stay non-negative and sum to one — the stability condition the delta rule
demands. A **Delta LSTM** extends this to six fast weight matrices (best LM perplexity, but expensive:
14K words/s vs 41K for Delta RNN). A **Delta MLP** stacks $K$ fast layers
$h^{(t)}_k=W^{(t)}_k f(h^{(t)}_{k-1})$; it *underperforms* — evidence that generating all layers'
weights at once is worse than letting the slow net see intermediate activations.

**Contribution 2 — recurrent slow nets (Recurrent Delta Net, RDN).** Make the slow net's projections
depend on the fast net's previous output $y^{(t-1)}$:
$$
\begin{aligned}
k^{(t)}&=W_k x^{(t)}+R_k\tanh(y^{(t-1)}), &
v^{(t)}&=W_v x^{(t)}+R_v\tanh(y^{(t-1)}),\\
q^{(t)}&=W_q x^{(t)}+R_q\tanh(y^{(t-1)}), &
\beta^{(t)}&=\sigma\big(W_\beta x^{(t)}+R_\beta\tanh(y^{(t-1)})\big).
\end{aligned}
$$
With these four extra recurrent connections the whole system becomes a proper RNN while retaining
constant memory and linear time.

**Complexity.** All variants keep the linear Transformer's constant state size and linear time in
sequence length; the extra cost is per-step compute (custom CUDA kernels). Delta RNN and RDN are the
practical sweet spot.

**RL result (project-relevant).** Setup: IMPALA / V-trace actor-critic (Torchbeast), the large 15-layer
residual conv stem, with the single 256-node LSTM replaced by an RDN (2 layers, hidden 128, 4 heads,
FF 512) or Delta RNN. Trained per-game on standard Atari 2600. RFWPs give large *relative improvements
over the linear Transformer* and beat LSTM in most of 20 games. **Takeaway for us:** a fast-weight
recurrent memory is a viable, constant-memory drop-in for an LSTM in an on-policy actor-critic —
architecturally close to the recurrent policies used in the project's grid world, and the delta-rule
write gives a principled "context-dependent weight" mechanism that is stronger than plain additive
attention.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Transformers are quadratic-time and grow state linearly; problematic for
  auto-regressive / long / infinite sequences and for RL in POMDPs (still LSTM-dominated). Linearized
  attention → linear Transformers (Katharopoulos, Performer, Peng). This work: adopt the FWP view; add
  recurrence to slow and fast nets → RFWPs. Contributions: (1) study novel powerful FWPs; (2) augment
  linear Transformers with recurrence.
- **§2 Background on FWPs.**
  - **§2.1 General formulation** (Eqs.1–4): SlowNet/FastNet; incremental UpdateRule decomposition.
  - **§2.2 Linear Transformers as FWPs.** Sum update rule (Eqs.5–7, rank-one outer product = linearized
    attention). Delta Net (Eqs.8–10): delta rule with generated $\beta^{(t)}$ and read-back $\bar v^{(t)}$;
    fixes capacity problem. Multi-head version. "Other approaches": Hypernetworks scale rows of a slow
    matrix; weight compression; dynamic conv / LambdaNets / plasticity.
- **§3 FWPs with slow or fast RNNs.**
  - **§3.1 Fast-net extensions.** Making any net fast (RNN with fast $W^{(t)},R^{(t)}$). **Delta RNN**
    (Eq.12). **Delta LSTM** (6 fast matrices, App.A.2). **Delta MLP** (Eqs.13–14, deep fast net) and
    **Delta Delta Net** (Delta Net as fast net — hierarchical FWP).
  - **§3.2 Slow-net extensions.** **Recurrent Delta Net (RDN)** (Eqs.15–18): recurrent connections into
    $k,v,q,\beta$ from $y^{(t-1)}$.
  - **§3.3 Related models.** RFWPs as memory-augmented RNNs; Schmidhuber 1993 / Ba et al. 2016 recurrent
    FWP; Fast Weight Memory; Metalearned Neural Memory; residual/layernorm/multi-head deep Transformer
    scaffolding.
- **§4 Experiments.**
  - **§4.1 Language modelling (WikiText-103, Table 1).** Transformer 34.1 test ppl; Linear Transformer
    38.3; Delta Net 35.2; Delta MLP worse (36.8); Delta RNN 35.0; Delta LSTM best (33.8, 47.3M params);
    RDN 35.2. Full-context Delta RNN 32.8 / RDN 33.6 beat the 256-window Transformer. Speeds: LT 66K,
    Delta Net 63K, Delta RNN 41K, RDN 35K, Delta LSTM 14K words/s (Transformer 33K).
  - **§4.2 Code execution.** Track 3 or 5 variables. LSTM best (99.0/93.2%); Linear Transformer fails
    (0%); Delta RNN stable at 5 vars (85.1%) vs Delta Net unstable (61.4±20).
  - **§4.3 Sequential ListOps.** Depth 10/15. LSTM best at depth 10 (88.5%) but collapses at depth 15
    (24.4%); mutable-memory Transformer variants beat regular + linear Transformers; RDN 79.2% at depth 15.
  - **§4.4 RL on Atari 2600.** IMPALA/V-trace; replace LSTM with RDN or Delta RNN; large relative
    improvements over LT across ~20 games; small RDN already strong.
- **Appendices.** A: model/ablation details (incl. Delta LSTM equations, activation-function ablation,
  Delta-Delta dimensionality). B: task descriptions + extra results. C: RL hyperparameters.

---
