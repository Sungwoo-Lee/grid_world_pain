> **Per-paper review — in-context-learning corpus, paper 34 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§34); content is identical. Manifest: [[in_context_learning_sources]].

# 34. von Oswald et al. 2020 — Continual Learning with Hypernetworks

**PDF:** `docs/project/references/in_context_learning/sources/von Oswald et al. 2020 - Continual learning with hypernetworks.pdf`
· ICLR 2020 · arXiv 1906.00695

## Phase 1: Foundational Overview (Undergraduate-Level)

**The problem.** Train a network on task 1, then task 2, then task 3 — and it *forgets* task 1
("catastrophic forgetting"). The obvious fix, keeping all past data around and rehearsing it, violates
the spirit of online/continual learning. This paper's insight: **move the forgetting problem up one
level.** Don't try to protect the network's weights directly; instead learn a small **hypernetwork** — a
"weight factory" — that takes a short **task embedding** vector $e^{(t)}$ and outputs the full weight set
$\Theta_{\text{trgt}}$ of the task's network.

**Why that helps.** To remember a task you no longer need to store its data — you only need to remember
one low-dimensional embedding $e^{(t)}$ and make sure the hypernetwork still maps that embedding to the
same good weights it used to. So "don't forget task $t$" becomes a single, cheap constraint: *keep
$f_h(e^{(t)})$ fixed.* This is enforced with a simple regularizer that requires **no past data at all** —
only the stored embeddings.

**Key findings.** Task-conditioned hypernetworks reach state-of-the-art on standard continual-learning
benchmarks (permuted/split MNIST, split CIFAR-10/100) and — remarkably — retain performance on very long
task sequences (up to ~100 tasks) with essentially no degradation, even in a **compressive regime** where
the hypernetwork has *fewer* trainable weights than the target network it generates (thanks to
"chunking" — reusing a small hypernetwork to emit the target weights piece by piece). The learned
task-embedding space shows structure and supports forward transfer. The framework also protects a
generative-replay model (the same regularizer shields a VAE's hypernetwork).

**Initial takeaway.** This is the canonical "task-conditioned hypernetwork" paper and the most direct
[[Hypernetwork]]-topic bridge in the shard. Its central lesson for the project: **conditioning weight
generation on a compact task/context embedding turns "remember a behaviour" into "remember a vector,"**
and a single output-space regularizer prevents interference between contexts. If the project ever wants
one agent to hold several context-specific policies without them bleeding into each other, this is the
reference architecture and the reference regularizer.

## Phase 2: Graduate-Level Deep Dive

**Task-conditioned hypernetwork.** Instead of learning target weights $\Theta_{\text{trgt}}$ directly,
learn a metamodel $f_h(\cdot,\Theta_h)$ whose output *is* $\Theta_{\text{trgt}}$:
$$
\Theta_{\text{trgt}}^{(t)} = f_h\big(e^{(t)},\Theta_h\big),
$$
where $e^{(t)}$ is a learned, differentiable **task embedding** (an ordinary parameter vector optimized by
backprop alongside $\Theta_h$). The target network then computes $y=f_{\text{trgt}}(x,\Theta^{(t)}_{\text{trgt}})$.

**The forgetting problem, stated.** The ideal but forbidden rehearsal loss fixes past input–output maps:
$$
\mathcal{L}_{\text{output}}^{\text{data}}=\sum_{t=1}^{T-1}\sum_{i=1}^{|X^{(t)}|}
\big\|f(x^{(t,i)},\Theta^{*})-f(x^{(t,i)},\Theta)\big\|^2 ,
$$
which requires storing all past inputs $x^{(t,i)}$. The hypernetwork replaces this with a **weight-space**
constraint that needs *no data*.

**Two-step, data-free output regularizer (the core contribution).** When learning task $T$:
1. Compute a candidate task-loss step $\Delta\Theta_h$ (one Adam step) minimizing the current task loss
   $\mathcal{L}^{(T)}_{\text{task}}=\mathcal{L}_{\text{task}}(\Theta_h,e^{(T)},X^{(T)},Y^{(T)})$.
2. Take the actual step by minimizing the total loss
$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{task}}\big(\Theta_h,e^{(T)},X^{(T)},Y^{(T)}\big)
+\frac{\beta_{\text{output}}}{T-1}\sum_{t=1}^{T-1}
\big\|f_h(e^{(t)},\Theta_h^{*})-f_h\big(e^{(t)},\Theta_h+\Delta\Theta_h\big)\big\|^2 .
$$
Here $\Theta_h^{*}$ = hypernetwork parameters *before* learning task $T$ (a frozen snapshot), $\Delta\Theta_h$
is treated as a constant "lookahead," and $\beta_{\text{output}}$ trades plasticity vs stability. The
regularizer says: *for every past task embedding $e^{(t)}$, the weights the hypernetwork emits must not
drift from what they were.* Memory of the past enters **only** through the stored embeddings
$\{e^{(t)}\}_{t=1}^{T-1}$ — never through past data. (A stochastic version averages over a random subset
of past tasks for efficiency; a sensitivity scan on $\beta_{\text{output}}$ is in App D.)

**Model compression via chunking.** Emitting a full deep-net weight tensor in one shot is high-dimensional.
Instead invoke a *small* hypernetwork iteratively, once per **chunk** (e.g. one layer at a time). To break
the unwanted weight-sharing this induces, introduce a learned set of **chunk embeddings**
$C=\{c_i\}_{i=1}^{N_C}$; the full target weights are the concatenation
$$
\Theta_{\text{trgt}}=\big[f_h(e,c_1),\,f_h(e,c_2),\,\dots,\,f_h(e,c_{N_C})\big],
$$
iterating over chunk embeddings with the task embedding $e$ fixed. Chunk embeddings are ordinary learned
parameters shared across tasks. This yields the **compressive regime**: the number of trainable
hypernetwork parameters can be *smaller* than the target-network size, yet the model still solves the
task (App E states a universal-approximation result for chunked nets).

**Context-free inference (unknown task identity).** The hypernetwork needs a task embedding as input, but
sometimes the task ID is unknown at test time. Three strategies:
- **HNET+ENT** — pick the embedding $e^{(t)}$ giving the lowest predictive-entropy output for the input
  (in-distribution data → peaked output). No extra learning.
- **HNET+R** — hypernetwork parameterizes a *replay* generator (VAE); mix current data with synthetic
  past data, protecting the target classifier by soft targets.
- **HNET+TIR** — add an auxiliary task-inference classifier, itself protected by hypernetwork-regularized
  synthetic replay, that predicts task ID from the input.

**CL scenarios (van de Ven & Tolias taxonomy).** CL1: task ID given. CL2: task ID unknown but not needed
(shared head). CL3: task ID must be inferred (hardest).

**Project relevance.** Two takeaways. (1) *Architecturally*, "one shared weight-generator + per-context
embedding" is exactly the design pattern behind a context-modulated policy; the project's FiLM modulator
is the affine special case ($f_h$ emits scale/shift instead of full weights), and this paper is the full
generalization with the memory analysis to justify it. (2) *The output regularizer* is the mechanism that
lets a single generator hold *multiple* context-specific behaviours without interference — a directly
importable idea if the grid-world agent must support several noise/context regimes from one network.
Caveat: this is supervised continual learning, not RL; adapting the two-step regularizer to a non-
stationary RL objective is non-trivial and would be a `senior-developer` `issue_plan` if pursued.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** CL desiderata: no forgetting / positive backward transfer / positive forward
  transfer. Thought experiment: rehearsal with self-generated targets = multi-task upper bound but stores
  data. Change of perspective: maintain *sets of parameters* $\{\Theta^{(t)}\}$ via a metamodel
  $f_h(e,\Theta_h)$ mapping task embedding → weights; memorize one point per task. Generic w.r.t. target
  architecture; also improves generative replay.
- **§2 Model.**
  - **§2.1 Task-conditioned hypernetworks.** Weight generators (Ha 2017, Schmidhuber 1992). Output
    regularizer (Eq.1 = data-dependent baseline; Eq.2 = data-free two-step version with lookahead
    $\Delta\Theta_h$). Learned task embeddings updated by task loss, saved after each task.
  - **§2.2 Chunked hypernetworks.** Iterative chunk generation; chunk embeddings $C$; compressive regime;
    App E approximation result.
  - **§2.3 Context-free inference.** HNET+ENT (entropy), HNET+R (replay), HNET+TIR (task-inference
    classifier). CL1/CL2/CL3.
- **§3 Results.** Permuted/split MNIST, split CIFAR-10/100 (ResNet-32 target). SOTA on standard
  benchmarks; near-zero degradation on long (~100-task) sequences; compressive regime works; task-embedding
  space structure + forward transfer; hypernetwork-protected replay improves generative-replay CL.
- **Appendices B–F.** Architectures, loss functions, $\beta_{\text{output}}$ sensitivity (D),
  approximation theorem (E), GAN replay variant (F).

---
