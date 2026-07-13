> **Per-paper review — continual-learning corpus, paper 21 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§21); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 21. Narvekar et al. 2020 — Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey

**PDF:** `docs/project/references/continual_learning/sources/Narvekar et al. 2020 - Curriculum Learning for RL Domains.pdf`
**Venue:** JMLR 21(181):1–50, 2020. **Authors:** Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E. Taylor, Peter Stone.

**Primer connection.** Primer §5 uses this survey as the *reference frame for why a curriculum can yield zero or negative benefit*: a curriculum is a bet that the inter-task transfer mechanism is net-positive, and the project's curriculum agent transferred a *collapsed* policy and *discarded* the belief state — net-negative. This review extracts the survey's formal machinery (its curriculum definition, the seven classification dimensions, the transfer-metric vocabulary, and the five sequencing families) so those diagnoses can be stated precisely. Being a 50-page survey, the deep dive targets the taxonomy and the curriculum-MDP formalism rather than every surveyed paper.

<a id="p1-narvekar"></a>
## Phase 1 — Foundational Overview

**The idea.** Just as a human student learns arithmetic before calculus, an RL agent can often learn a hard task faster if it is first trained on a *sequence* of easier, related tasks — a **curriculum**. The motivating example is Quick Chess: a graded series of mini-chess variants (smaller boards, fewer pieces) that build up to full chess. The survey's job is to give the field a *common language* and a *classification framework* for the many scattered ways people have built such curricula for RL.

**Three parts of the method.** Any curriculum-learning approach decomposes into:
1. **Task generation** — where do the intermediate tasks come from? (hand-designed, or automatically generated).
2. **Sequencing** — in what order should tasks be presented? (the central, most-studied problem).
3. **Transfer learning** — how is knowledge carried from one task to the next? (transfer a policy, a value function, a model, options/skills, or a shaping reward).

**How curricula are evaluated.** Borrowed from transfer learning: **time-to-threshold** (how much faster you reach a target performance), **asymptotic performance** (final performance after convergence), **jumpstart** (initial boost from transfer), and **total reward** (area under the learning curve). A crucial distinction is **weak transfer** (time spent on source tasks treated as a free sunk cost) vs. **strong transfer** (all that source-task time is charged against the curriculum). A curriculum can *look* good under weak transfer and *lose* under strong transfer — exactly the accounting a project must do before claiming a curriculum "helped."

**When curricula help vs. hurt.** The survey repeatedly warns of **negative transfer**: a badly chosen intermediate task, or a badly chosen thing-to-transfer, can make the target task *harder* than learning from scratch. The whole point of task generation is to produce tasks "such that knowledge transfer through them is beneficial" and "avoid negative transfer." There is (as of 2020) very little *theory* on when curricula help — the open-problems section flags this gap explicitly.

**Initial takeaway.** A curriculum is not automatically beneficial; it is a *bet* with three independently-fallible components (task set, order, transfer mechanism) that must be evaluated under honest (strong-transfer) accounting. This framework is what lets the project say precisely which component of *its* curriculum failed.

<a id="p2-narvekar"></a>
## Phase 2 — Graduate-Level Deep Dive

### 3.1 Formal definition of a curriculum (Definition 2)

A task is an MDP $m_i=(S_i,A_i,p_i,r_i)$. Let $\mathcal{T}$ be a set of tasks and $D_{\mathcal{T}}$ the set of all transition samples generable from them:
$$D_{\mathcal{T}} = \{(s,a,r,s')\mid \exists\, m_i\in\mathcal{T}\ \text{s.t.}\ s\in S_i, a\in A_i, s'\sim p_i(\cdot\mid s,a), r\leftarrow r_i(s,a,s')\}.$$

**Definition 2 (Curriculum).** A curriculum $C=(V,E,g,\mathcal{T})$ is a **directed acyclic graph** where $V$ is a vertex set, $E\subseteq\{(x,y)\mid (x,y)\in V\times V \wedge x\neq y\}$ the directed edges, and $g:V\to\mathcal{P}(D_{\mathcal{T}})$ maps each vertex to a subset of samples (with $\mathcal{P}$ the power set). A directed edge $\langle v_j,v_k\rangle$ means the samples at $v_j$ should be trained on before those at $v_k$. All paths terminate at a single sink node $v_t$ (the target task).

Three common **simplifications** reduce this general graph:
- **Single-task curriculum** (Def. 3): all samples come from one task — i.e. ordering experience within a task (e.g. prioritized replay).
- **Task-level curriculum** (Def. 4): each vertex is an entire intermediate task; the DAG is over tasks.
- **Sequence curriculum** (Def. 5): a linear chain $[m_1,m_2,\dots,m_n]$ — the simplest and most common form.

These compose: e.g. a *task-level sequence curriculum* is an ordered list of tasks. The project's difficulty-ladder curriculum is a **task-level sequence curriculum** in this taxonomy.

### 3.2 Transfer-learning evaluation metrics (the accounting that matters)

From §2.3, four metrics compare a post-transfer learning curve on the target against a from-scratch learner:
- **Time to threshold**: episodes/steps/wall-clock to reach expected return $G_0\ge\delta$.
- **Asymptotic performance**: final converged performance.
- **Jumpstart**: initial performance improvement at the start of target-task learning.
- **Total reward ratio**: accumulated reward up to a fixed stop, transfer vs. no-transfer.

**Weak vs. strong transfer.** The transfer curve conventionally starts at time $0$ on the target *even though source-task time was already spent* — that is **weak transfer** (source time = sunk cost). **Strong transfer** charges source-task (and, most comprehensively, curriculum-generation) time by offsetting the curves. The survey notes that *achieving an asymptotic improvement implies strong transfer*, whereas time-to-threshold claims are only meaningful once you specify which accounting is used. This is the precise lever behind the primer's critique that the project's curriculum must be judged on a fair (strong-transfer) budget.

### 3.3 The seven classification dimensions (the taxonomy)

Every surveyed method is placed on seven attribute axes (attribute: *values*):
1. **Intermediate task generation**: *target / automatic / domain experts / naive users* — who/what produces the source tasks.
2. **Curriculum representation**: *single / sequence / graph*.
3. **Transfer method**: *policies / value function / task model / partial policies / shaping reward / other / no transfer* — what knowledge crosses task boundaries. Low-level (full policy / value function / model → directly initialize the learner) vs. high-level (partial policies/options, shaping rewards → guide but not initialize).
4. **Curriculum sequencer**: *automatic / domain experts / naive users*.
5. **Curriculum adaptivity**: *static* (whole curriculum fixed before training) vs. *adaptive* (dynamically adjusted using in-training signals like learning progress).
6. **Evaluation metric**: *time to threshold / asymptotic / jumpstart / total reward* — bolded when strong transfer.
7. **Application area**: *toy / sim robotics / real robotics / video games / other*.

For the project's setup: intermediate tasks = *domain experts* (hand-graded difficulties); representation = *sequence*; transfer method = *policies* (weights carried) — and, per the primer, the belief/recurrent state was *discarded*, which in this taxonomy is a transfer-method choice that omitted the most valuable thing to transfer; adaptivity = *static*.

### 3.4 The five sequencing families (§4.2)

Sequencing methods form a spectrum by *how much the intermediate tasks may differ from the target MDP*:
1. **Sample sequencing (§4.2.1)** — reorder samples from the *target* task without changing the domain (supervised-CL analog of Bengio 2009). Includes **Prioritized Experience Replay** (Schaul 2016; prioritize high-TD-error transitions), complexity-index sorting (Ren 2018), ScreenerNet learned weights (Kim & Choi 2018). "No transfer" needed (single task).
2. **Co-learning (§4.2.2)** — a curriculum *emerges* from multi-agent interaction (self-play, competition/cooperation): e.g. OpenAI hide-and-seek (Baker 2020), asymmetric self-play (Sukhbaatar 2018), AlphaStar (Vinyals 2019). Transfers *policies*, adaptive.
3. **Reward and initial/terminal-state distribution changes (§4.2.3)** — intermediate tasks keep the dynamics but vary the reward and/or start/goal distributions (e.g. reverse curriculum / goal generation: Florensa 2017, 2018; Riedmiller 2018 SAC-X).
4. **No restrictions (§4.2.4)** — intermediate tasks may differ *arbitrarily* from the target. Three sub-approaches:
   - **MDP-based sequencing** — the **curriculum-MDP (CMDP)** formalism (detailed below).
   - **Combinatorial optimization / search** — treat sequencing as finding the best permutation of a fixed task set; black-box metaheuristics (Foglino 2019a–c).
   - **Graph-based sequencing** — build a DAG of tasks by relations (Svetlik 2017 shaping-reward graph; MacAlpine & Stone 2018).
5. **Human-in-the-loop (§4.2.5)** — how (expert and non-expert) humans design/sequence curricula (Peng 2018, Khan 2011, Stanley 2005).

### 3.5 The curriculum-MDP (CMDP) formalism — the deepest technical idea

Narvekar et al. (2017) formalize sequencing as a **meta-MDP over the learning agent's policy space**. Two nested MDPs:
- The **base MDP**: the learning agent ("student") interacting with a task.
- The **meta-MDP / CMDP**: the curriculum agent ("teacher"). Its **state space** $S$ = the set of policies the student can represent (parameterized by the student's weights). Its **action space** $A$ = the set of tasks the student can train on next. The **transition** $p$: training the student on the chosen task updates the student's policy → a state transition in the CMDP. The **reward** $r$ = the time (steps/episodes) it took to learn the selected task.

The teacher starts at the state corresponding to a random student policy and aims to reach a *terminal state* (a student policy meeting a target-task threshold) **as fast as possible** — i.e. minimizing time-to-threshold. Matiisen et al. (2017) recast this as a **POMDP** (no access to student internal weights; observation = current score per task; reward = change in score) with the different objective of *maximizing the sum of performance over all tasks* — its "Teacher-Student" heuristic picks tasks where the absolute slope of the learning curve is highest (most progress *or* most forgetting). Narvekar & Stone (2019) show one can *learn a curriculum policy* over the CMDP via function approximation on the transfer-representation weights, mapping "current learning progress → next task," and that training each intermediate task for only a few episodes (letting the curriculum policy re-select) beats training-to-plateau. Caveat repeatedly stated: learning a curriculum is often *more expensive* than just solving the target task, and is done per-agent/per-task.

### 3.6 Open problems (§6) — directly relevant framing

The survey's own list of gaps sharpens what a project should be cautious about: (6.1) fully automated task creation is under-studied; (6.2) transferring *different types* of knowledge between different task pairs is essentially unexplored (almost all works fix one transfer type — relevant to the project's decision to transfer only policy weights and drop belief state); (6.3) amortizing curriculum-generation cost (reuse, sim-to-real); (6.4) combining task generation + sequencing end-to-end; (6.5) **lack of theory** on *when and why* curricula help (only initial supervised-learning results via Ideal/Local Difficulty Scores); (6.6) understanding general principles of (human) curriculum design.

### 3.7 Relevance to the project

The framework gives the project's negative curriculum result a precise diagnosis: it was a **task-level sequence curriculum** transferring **policies** *statically*, evaluated as a bet on net-positive transfer. Two of the survey's warnings apply directly — **negative transfer** (transferring a collapsed policy, cf. the Cui entropy thread, is a canonical way to poison the target) and the **weak-vs-strong-transfer accounting** (a curriculum that carried weights across a ladder but lost to a from-scratch baseline fails even the *weak* transfer bar, which is the strongest possible refutation). The §6.2 open problem — that the *type* of knowledge transferred should perhaps differ per task, and that discarding some transferable knowledge (here, the recurrent belief state) is a design choice with consequences — is the exact seam that connects to Caccia et al. (entry 22), who show carrying the recurrent state across boundaries can *beat* task-aware agents.

<a id="bb-narvekar"></a>
## Appendix: Section-by-Section Backbone

**§1 Introduction.** Quick Chess motivating example (graded mini-games → full chess). Curriculum = ordering over experience (samples or tasks). Field is scattered and inconsistently defined; goal is a systematic framework + survey + open problems. Poses the guiding questions (what is a curriculum; how to represent/evaluate; how to generate tasks; how to sequence; how to transfer).

**§2 Background.** RL preliminaries (MDP, value/action-value functions, policy search, actor-critic). **§2.2 Transfer learning**: train on source MDP(s), transfer samples/options/policies/models/value functions to target; task mappings; risk of negative transfer. **§2.3 Evaluation metrics**: time-to-threshold, asymptotic, jumpstart, total reward; weak vs. strong transfer (sunk-cost accounting; offset curves, Fig. 2).

**§3 The Curriculum Learning Method.**
- **§3.1 Curricula.** **Def. 2 (Curriculum)** = DAG $(V,E,g,\mathcal{T})$ over sample subsets, single sink. Simplifications: **Def. 3 single-task**, **Def. 4 task-level** (DAG of tasks), **Def. 5 sequence** (linear chain). Composable (task-level sequence). Online (adaptive edges) vs. offline (pre-generated).
- **§3.2 Method components.** Three parts: task generation, sequencing, transfer learning.
- **§3.3 Evaluation.** Same metrics as transfer, applied after the full curriculum vs. no curriculum; curriculum-generation cost accounting.
- **§3.4 Dimensions.** The **seven classification dimensions** (task generation / representation / transfer method / sequencer / adaptivity / evaluation metric / application area) with their value sets.

**§4 Curriculum Learning for RL Agents.**
- **§4.1 Task generation.** Create intermediate tasks so transfer is beneficial; avoid negative transfer. Parameterized-domain / task-descriptor methods (Narvekar 2016): task simplification, promising initialization, mistake learning, etc. (Table 1).
- **§4.2 Sequencing** (core; Table 2). Five families: **§4.2.1 sample sequencing** (PER Schaul 2016, CI Ren 2018, ScreenerNet Kim&Choi 2018); **§4.2.2 co-learning** (self-play, Baker 2020, Sukhbaatar 2018, Vinyals 2019); **§4.2.3 reward/initial-terminal-state changes** (Florensa 2017/2018, Riedmiller 2018); **§4.2.4 no restrictions** — MDP-based (**CMDP** Narvekar 2017; POMDP Matiisen 2017; curriculum policy Narvekar & Stone 2019), combinatorial optimization/search (metaheuristics, Foglino 2019a–c), graph-based (Svetlik 2017, MacAlpine & Stone 2018); **§4.2.5 human-in-the-loop**.
- **§4.3 Knowledge transfer.** Which transfer mechanism between curriculum tasks; freeze-and-grow (e.g. progressive nets) prevents forgetting at parameter-count cost.

**§5 Related Areas.** §5.1 related RL paradigms; §5.2 curricula in supervised ML (Bengio 2009; self-paced learning); §5.3 algorithmically designed curricula in education.

**§6 Open Questions.** 6.1 fully automated task creation; 6.2 transferring different *types* of knowledge; 6.3 reusing curricula / sim-to-real; 6.4 combining generation + sequencing; 6.5 theoretical results (IDS/LDS in supervised learning; RL analog open); 6.6 general principles of (non-expert) curriculum design.

**§7 Conclusion.** Curriculum learning = task generation + sequencing + transfer; five sequencing families surveyed; open problems as future directions; call for common terminology.

---
