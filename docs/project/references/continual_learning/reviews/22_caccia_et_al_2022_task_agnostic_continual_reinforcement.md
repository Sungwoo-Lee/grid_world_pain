> **Per-paper review — continual-learning corpus, paper 22 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§22); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 22. Caccia et al. 2022 — Task-Agnostic Continual Reinforcement Learning

**PDF:** `docs/project/references/continual_learning/sources/Caccia et al. 2022 - Task-Agnostic Continual RL.pdf`
**Full title:** "Task-Agnostic Continual Reinforcement Learning: Gaining Insights and Overcoming Challenges." **Venue:** CoLLAs 2023 (arXiv 2205.14495v3, May 2023). **Authors:** Massimo Caccia, Jonas Mueller, Taesup Kim, Laurent Charlin, Rasool Fakoor. (The primer's §6.5(4) note: this is the same paper the project critique cites under the Amazon-Science title "In Praise of a Simple Baseline.")

**Primer connection.** Primer §5 uses this as the *recurrent-state-across-task-switches* thread: in a POMDP the recurrent hidden state *is* the belief, and carrying it across task boundaries can beat *task-aware* agents. The project hard-resets the recurrent state at every task boundary — the opposite default — plausibly a self-inflicted cost stacked on top of the plasticity and entropy problems. This review supplies the formal setting (task-agnostic CRL as a hidden-mode MDP), the 3RL method, the two hypotheses, and the headline result (a task-agnostic replay+RNN agent matching or beating its *multi-task* soft-upper-bound), so that "we discard the belief state" can be evaluated as a concrete design error.

<a id="p1-caccia"></a>
## Phase 1 — Foundational Overview

**The setup.** An agent must learn a *sequence* of tasks, one after another, seeing each only once, and — the hard part — it is **not told which task it is currently on** (no task ID) or even *when* the task changes. This is **task-agnostic continual RL (TACRL)**. It must infer the task from experience while not forgetting old tasks (catastrophic forgetting) and ideally getting *better* at learning new ones (forward transfer).

**The comparison baseline.** The usual "soft upper bound" for continual learning is a **multi-task (MTL)** agent that trains on *all* tasks simultaneously *and* is told each task's ID. Conventional wisdom: MTL should always beat a continual, task-agnostic agent, because MTL has no forgetting and knows the task.

**The method — 3RL.** Caccia et al. combine two simple, well-known ingredients:
- **Experience Replay (ER)** — keep a buffer of past-task data and rehearse it (fights forgetting).
- **A recurrent network (RNN)** — feed the agent the *history* of recent (state, action, reward) tuples through a GRU, so its hidden state can *implicitly infer which task it is on* without ever being told.

The combination is **replay-based recurrent RL (3RL)**, built on Soft Actor-Critic.

**The two hypotheses.**
1. **H1** — When tasks share structure, a task-*agnostic* agent that learns to *adapt fast* can *beat* a task-*aware* agent that *memorizes* per-task solutions — especially when memorization is hard (high dimensionality, many tasks, limited data/compute).
2. **H2** — This advantage is *amplified* in continual learning, because fast-adapting agents suffer less from catastrophic forgetting than memorizing agents.

**Key findings.**
- 3RL beats other continual-RL baselines on a synthetic quadratic-optimization benchmark and on Meta-World (50 manipulation tasks; the CW10 and a new, harder MW20 subset).
- Strikingly, 3RL **matches its own multi-task soft-upper-bound** and even **surpasses the MTL equivalent** in high-dimensional settings — which the authors believe is a first for a continual method. The "upper bound" wasn't an upper bound.
- The mechanism: the RNN **reduces gradient conflict** between tasks (the dynamic task representation lets updates for different tasks interfere less), improving stability and performance.
- The **recurrent** agent matches or beats a **transformer**-based history encoder.

**Initial takeaway.** In a partially-observed, task-agnostic continual setting, the recurrent hidden state *is the belief about which task you're in*, and letting it flow across tasks (rather than resetting it) is what enables fast adaptation and reduced gradient conflict — enough to rival agents that are *told* the task. For the project, this is the direct argument that hard-resetting the recurrent state at each curriculum boundary throws away the very mechanism that would have made cross-task transfer positive.

<a id="p2-caccia"></a>
## Phase 2 — Graduate-Level Deep Dive

### 4.1 The formal setting: TACRL as a hidden-mode MDP

Start from a **POMDP** $\langle S,A,T,X,O,r,\gamma\rangle$: an MDP augmented with observation space $X$ and emission $O(x'\mid s)$; the agent cannot read the true state $s_t$ from observation $x_t$. Split the state into an observable part $s^o_t$ and a hidden part $s^h_t$. The optimal policy must then condition on *history*:
$$\pi\big(a_t\mid s^o_{1:t},\,a_{1:t-1},\,r_{1:t-1}\big),$$
naturally parameterized by an RNN. The POMDP objective is
$$\mathbb{E}_{s^h}\Big[\,\mathbb{E}_\pi\big[\textstyle\sum_{t=0}^\infty \gamma^t r_t\big]\ \big|\ s^h\Big].$$

**TACRL is a structured special case — a hidden-mode MDP (HM-MDP)** (Choi et al. 2000): the agent has **no causal effect on $s^h$** (it cannot change which task it's in), and the hidden mode evolves as $p(s^h_{t+1}\mid s^h_t)$ independently of actions. The joint transition factorizes:
$$p\big(s^o_{t+1}\mid s^h_{t+1}, s^o_t, a_t\big)\,p\big(s^h_{t+1}\mid s^h_t\big).$$
TACRL further assumes $s^h$ follows a **non-backtracking chain**: once the chain enters a hidden state (task) it stays for a fixed number of steps, and previously visited tasks are never revisited. The hidden mode $s^h$ *is the task/context* (each context = a specific MDP).

**Evaluation — global return.** Unlike the "current return" (performance on the immediate task), TACRL evaluates the **global return**: performance averaged over *all* hidden states/tasks,
$$\mathbb{E}_{\tilde s^h}\Big[\,\mathbb{E}_\pi\big[\textstyle\sum_t\gamma^t r_t\big]\ \big|\ s^h\Big],$$
with $\tilde s^h$ the joint over all tasks. This measures how much knowledge about *all* tasks survives at the end of training — i.e. it *penalizes forgetting* by construction.

**The settings ladder (Table 1).** POMDP (most general) ⊃ HM-MDP (non-stationary hidden mode, no agent control over it) ⊃ **TACRL** (HM-MDP + non-backtracking chain + global-return evaluation). **Task-aware CRL** is TACRL made fully observable — the policy conditions on $s^h_t$ directly, $\pi(a_t\mid s^h_t, s^o_t)$, which turns the POMDP into an MDP. **MTRL** is task-aware CRL with a *stationary* hidden mode ($p(s^h_{t+1})$ independent of $s^h_t$ — all tasks always available). This ladder is the paper's cleanest contribution: it places "task ID given," "tasks all available," and "task boundaries observed" as three separable relaxations.

### 4.2 The algorithm and architectures

**Base algorithm: Soft Actor-Critic (SAC)** — off-policy, chosen because (i) off-policy is more sample-efficient (agents spend little time per task and see each once), and (ii) off-policy *decouples the learning policy from the acting policy*, which is what makes **replay** possible. SAC learns a stochastic max-entropy policy $\pi_\phi$ and critic $Q_\theta$.

Architectures compared:
- **TaskID** — feed task ID $\tau$ as extra input: $Q_\theta(s,a,\tau)$, $\pi_\phi(a\mid s,\tau)$ (task-aware).
- **Multi-head (MH)** — shared feature extractor + one head per task; task-aware. **TAMH** (task-agnostic MH) picks the most-confident actor head (by policy entropy) and most-optimistic critic head — isolates the value of *task information* from the value of *extra capacity*.
- **RNN (task-agnostic)** — GRU history encoder producing $z_t = \mathrm{RNN}(\{(s_i,a_i,r_i)\})$, fed to $\pi_\phi(a\mid s,z)$ and $Q_\theta(s,a,z)$. Actor and critic have *their own* RNNs (as in Meta-Q-Learning / Ni et al. 2021).
- **TX (task-agnostic)** — a Transformer history encoder as an alternative to the RNN.

**3RL = ER + RNN.** As an episode unfolds, $z_t=\mathrm{RNN}(\{(s_i,a_i,r_i)\}_{i=1}^{t-1})$ should capture the task identity, helping actor and critic. Pseudocode (Algorithm 1): for each task, sample with the current RNN-conditioned policy, store transitions in buffer $D$; each update samples a batch mixing current-buffer trajectories (fraction $\approx b\cdot\min(1/n, 1-\beta)$) with old-buffer trajectories (fraction $\approx b\cdot\min((n-1)/n,\beta)$), capped by replay-cap $\beta$ (robotic experiments cap at 80% → always ≥20% compute on the current task); then flush $D$ into $D_{old}$. A key implementation nuance: capping replay strictly needs two buffers (a priori task-aware), but the same effect is obtained task-agnostically by *oversampling recently collected data*.

### 4.3 The two hypotheses, precisely

**Hypothesis 1.** *When the reward and transition functions share structure across tasks, task-agnostic approaches can outperform task-aware ones where task memorization is difficult — high dimensionality, many tasks, or limited data/compute.* Intuition: a task-aware agent leans on the task ID to *memorize* a per-task solution (needs more data/capacity as tasks/dims grow); a task-agnostic RNN learns one *general* policy that *adapts* via task inference.

**Hypothesis 2.** *In continual learning the H1 effect is amplified, because algorithms that continually learn to adapt suffer less from catastrophic forgetting than algorithms that memorize each task.*

### 4.4 Empirical results and the gradient-conflict mechanism

**Benchmarks.** (i) *Quadratic Optimization* — synthetic, controllable dims: reward $r(s^o,\tau) = s^{o\top}A_\tau s^o + b_\tau s^o + c_\tau$ ($A_\tau$ negative-definite → unique global max; $c_\tau$ set so all tasks share the same max reward), transition $s^o_{t+1}=s^o_t + a_t$, $a_t\in[-1,1]^d$. Lets them dial dimensionality/#tasks/#timesteps. 80,000-run random search; robustness reported as interquartile mean (IQM). (ii) *Meta-World* — 50 manipulation tasks in a shared state/action space with a shared reward structure (reaching/grasping/pushing); CW10 subset (forward-transfer-focused, 1M steps/task) and a new harder **MW20** (first 20 tasks, 500k steps/task — twice as long, half the data/compute).

**Findings.**
- **H1 supported** (Fig. 2): 3RL is more robust than ER and ER-TaskID, and its edge *grows with observation dimensionality* (memorization gets harder as dims grow), while being roughly independent of the number of tasks. TX is always ≤ RNN.
- **Mechanism — gradient conflict** (Figs. 3–4): using the *standard deviation of gradients across the minibatch* as a proxy for gradient conflict / task interference (Yu et al. 2020's PCGrad angle proxy is argued against in App. L), the RNN achieves the highest correlation with global return and the largest reduction in gradient conflict, especially in the challenging 32-task/32-dim/1M-step scenario. On Meta-World, performance correlates **-0.75** with gradient conflict and **-0.81** with training instability (Q-value variance) — both significant. Hypothesis: 3RL improves performance by *reducing gradient conflict via dynamic task representations*, which also tames the **deadly triad** (function approximation + bootstrapping + off-policy) that non-stationarity aggravates.
- **H2 supported + the headline** (Figs. 5–7): 3RL outperforms all baselines on CW10 and MW20, and **matches its MTL soft-upper-bound** — the first method the authors know of to equal a stationary-regime multi-task agent *despite* the non-stationary continual setting — and in high-dim synthetic settings *surpasses* the MTL equivalent. TaskID's relative performance *drops* with dimensionality in CL, exactly as H1/H2 predict.
- **Controls** (appendices): the gain is *not* from parameter count (App. G), *not* from task-awareness+RNN combination (App. N — adding task-awareness to the RNN does not help), *not* single-task improvement (App. H — RNN doesn't help single-task), *not* parameter stability (App. I). Support that the RNN *places new tasks in the context of previous ones* (forward transfer), backed by the PCA of RNN representations in Fig. 1 (task-invariant initialization + richer, evolving representations).

### 4.5 Relevance to the project

The paper's core message is mechanistic and directly transferable: in a partially-observed, non-stationary (continual) setting, **the recurrent hidden state is the belief over the current task**, and letting it carry information *across* task boundaries (a) performs implicit task inference without a task ID and (b) *reduces gradient conflict*, which is what lets a continual agent rival a task-aware multi-task agent. The project's decision to **hard-reset the recurrent state at each curriculum boundary** is, in this light, discarding the belief precisely at the moment it is most informative (the task just changed) — throwing away both the task-inference signal and the gradient-conflict-reduction benefit. Combined with the Narvekar framing (entry 21: the project transferred *policy weights* but omitted the belief state — a §6.2 "wrong knowledge type transferred" error) and the Cui thread (entry 20: a transferred *collapsed* policy is a canonical negative-transfer poison), the three adjacent threads converge on a coherent diagnosis of the project's curriculum failure. Caveat for transfer: Caccia et al. use *off-policy SAC + replay*; the project's on-policy recurrent PPO differs, though the authors note (footnote 4) their findings extend to any method with a replay buffer, and the belief-state argument is algorithm-independent.

<a id="bb-caccia"></a>
## Appendix: Section-by-Section Backbone

**Abstract.** Investigates why task-agnostic CL differs in performance from multi-task (MTL) agents. Two hypotheses (task-agnosticity helps under limited data/compute/high-dim; fast adaptation mitigates forgetting). Introduces **3RL** (replay-based recurrent RL). Tested on synthetic + Meta-World (50 tasks); 3RL beats baselines and even surpasses its MTL equivalent in high-dim; recurrent ≥ transformer.

**§1 Introduction.** CL agents learn a task sequence without forgetting; MTL (task-aware, all tasks jointly) is the usual soft upper bound; task-agnostic CL (no task ID) is the practical, harder setting. Reasoning: task-agnostic methods learn to *adapt* (generalize), task-aware methods *memorize* (need more data/compute). States H1 and H2; instantiates task-aware and task-agnostic methods; adds RNN memory + replay = 3RL. Evaluated on synthetic quadratic + Meta-World.

**§2 Background & TACRL.** MDP and POMDP definitions; history-conditioned policy $\pi(a_t\mid s^o_{1:t},a_{1:t-1},r_{1:t-1})$ (→ RNN). **TACRL** = HM-MDP (agent has no causal effect on hidden mode $s^h$; $p(s^h_{t+1}\mid s^h_t)$) with a **non-backtracking chain** and **global-return** evaluation. Task awareness = full observability (POMDP→MDP). MTL trains on all tasks jointly (stationary distribution) — soft upper bound but often impractical. **Table 1** ladders MDP / POMDP / HM-MDP / TACRL / Task-Aware CRL / MTRL by transition, policy conditioning, objective, evaluation.

**§3 Methods & Hypotheses.**
- **§3.1 Algorithms.** Off-policy chosen (sample efficiency + replay support). Base = **SAC** (actor $\pi_\phi$, critic $Q_\theta$).
- **§3.2 Models.** **TaskID** (feed $\tau$); **Multi-head MH** (one head/task; task-aware) and **TAMH** (task-agnostic head selection by entropy/optimism — isolates task-info from capacity); **RNN** (GRU history encoder → $z$, task-agnostic; separate actor/critic RNNs); **TX** (transformer history encoder). **Hypothesis 1** stated. Robot-manipulation analogy (memorize per object vs. adapt).
- **§3.3 Baselines.** FineTuning (no forgetting-prevention), **ER** (replay; capped by oversampling current task, Alg. 1 L8-9), **MTL** (soft upper bound), **Independent** (separate model/task, no transfer). Combos (MTL-TaskID, FineTuning-MH, ...). **3RL = ER + RNN** (Algorithm 1). **Hypothesis 2** stated.

**§4 Empirical Findings.** Benchmarks: **Quadratic Optimization** (synthetic, $r=s^{o\top}A_\tau s^o+b_\tau s^o+c_\tau$, controllable dims) and **Meta-World** (CW10 1M steps/task; new **MW20** 500k steps/task, harder). Metrics: global vs current return/success; IQM over 80k runs; top-10% for maximal.
- **§4.1 Hypothesis 1.** 3RL most robust, edge grows with dimensionality (Fig. 2); TX ≤ RNN. Gradient-conflict proxy = std of gradients across minibatch; RNN reduces conflict most in the challenging 32-task/32-dim/1M scenario (Fig. 3). Meta-World: performance vs conflict corr -0.75, vs instability -0.81 (Fig. 4). Deadly-triad framing. 3RL beats all on CW10/MW20 (Fig. 5).
- **§4.2 CL vs MTL.** TaskID's relative performance drops with dims in CL; 3RL *surpasses* MTL analog in high-dim synthetic (Fig. 6); on MW20, **3RL matches its MTRL soft-upper-bound** — claimed first (Fig. 7). Appendix controls: not parameters (G), not task-aware+RNN (N), not single-task gain (H), not parameter stability (I); RNN contextualizes new tasks (J, Fig. 1 PCA).

**§5 Related Work.** CRL (Continual World: forgetting-reduction methods lose transfer); task-agnostic CL upper-bound comparisons; TACRL methods (GP mixtures, bandit policy retrieval, meta-learning); RNNs in continual supervised learning and in POMDP RL; transformers in RL. Novelty: RNN within TACRL combined with ER.

**§6 Conclusion.** Task-agnosticity can beat task-aware MTL in resource-constrained / high-dim / multi-task regimes; 3RL matches or surpasses MTL; mechanism partly = reduced gradient conflict; challenges the assumption that task-agnostic CL is inherently harder.

---
