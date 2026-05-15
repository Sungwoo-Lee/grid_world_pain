---
title: "Adapting to Environment Changes Through Neuromodulation of Reinforcement Learning"
authors: ["Jinwei Xing", "Xinyun Zou", "Praveen K. Pilly", "Nicholas A. Ketz", "Jeffrey L. Krichmar"]
year: 2022
venue: "SAB 2022 — From Animals to Animats 16 (LNAI 13499, pp. 115–126, Springer)"
slug: xing_2022_neuromodulation_rl_environment_changes
source_pdf: "sources/Xing et al. 2022 - Adapting to Environment Changes Through Neuromodulation of Reinforcement Learning.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This paper asks a continual-RL question: an agent has been trained on one task, the environment silently changes to a new task, then later changes back. How can the agent (a) notice the switch as soon as possible, (b) decide whether the new task is *novel* (needs a fresh policy) or *familiar* (recall a stored policy), and (c) avoid corrupting what it already knew? Catastrophic forgetting is the trap: if you keep training your current policy with the wrong reward signal, you destroy the policy you had.

The authors' answer is a brain-inspired meta-controller that sits on top of a standard deep-RL agent (PPO for the gridworld, TD3 for the MuJoCo walker). Two neuromodulator-like signals do the work:
- **Acetylcholine (ACh)** — a vector with one entry per *known task*. Each entry is a running confidence "am I in this task?" updated from **reward prediction error (RPE)** between an internal *reward predictor* and the actual reward.
- **Norepinephrine (NE)** — a single scalar that rises when no stored task matches and signals "this is genuinely novel — open a new task slot."

The key trick is that the agent learns one *reward predictor* per task in addition to the policy. When the environment changes, the previously winning task's predictor stops being accurate, the corresponding ACh drops, all ACh entries are low, **task change is detected**. A softmax over ACh then attempts to **match** the new task against stored tasks; if the softmax fails to put $> p_{\text{match}}$ on any one of them, NE rises; once NE crosses a threshold, the agent creates a new task slot with a fresh policy and a fresh predictor.

The headline claim, demonstrated in two environments — a 3-task gridworld pickup task and a 3-task MuJoCo walker (stand / walk / run) with each task sequence visited twice: **with the neuromodulator, returning to a previously learned task recovers 90% performance in ~150 timesteps (gridworld) or ~800–1800 timesteps (walker); without it, recovery takes 25,000–600,000+ timesteps.** Two-to-four orders of magnitude faster. The paper is the explicit extension of Zou et al. 2020 (perception domain) into RL (action domain).

## Section-ordered backbone

### Abstract
Standard RL assumes a fixed MDP. Real environments change. Two challenges: rapid change detection and retention of prior knowledge. The authors develop an ACh+NE neuromodulator that enables RL agents to detect changes and selectively recall or learn. Evaluated in gridworld and on a MuJoCo bipedal walker.

### 1. Introduction
- Deep RL successes in games, robotics, autonomous driving — but fragile under dynamic environments.
- Two requirements: detect the change fast (otherwise reward feedback corrupts the existing policy); identify whether the changed environment is familiar (recall) or novel (learn).
- Animals solve this via the neuromodulatory system; Yu & Dayan (2005) decompose uncertainty into expected (ACh) and unexpected (NE).
- Prior work: Zou et al. 2020 applied this to classification / attention; this paper extends to RL.

### 2. Problem
**2.1 RL recap.** Standard MDP $(S, A, T, R)$, reward $r_t = R(s_t, a_t)$, transition $s_{t+1} = T(s_t, a_t)$, discounted return $r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$ with $\gamma \in [0, 1]$.

**2.2 Environment changes.** A "task" is an MDP $\text{Task}_i = \langle S, A, T, R_i \rangle$. The environment is a sequence $\text{Env} = [\text{Task}_1, \text{Task}_2, \dots, \text{Task}_N]$. **Only $R$ varies across tasks; $S$, $A$, $T$ are shared.** Tasks may recur (some $R_i = R_j$), so the agent must store and recall.

### 3. Method
**3.1 ACh and NE neuromodulation.**
- ACh is a vector with $K$ entries (one per detected task); initialized with $K = 1$ at start; $K$ grows.
- **Task-change detection**:
  $$\text{Task\_Change} = (\max(\text{ACh}) < ch_{\text{change}}). \tag{5}$$
- **Task match** via softmax confidence:
  $$P_i = \frac{\exp(\beta \cdot \text{ACh}_i)}{\sum_j \exp(\beta \cdot \text{ACh}_j)}, \qquad \text{Task\_Match}_i = (P_i > p_{\text{match}}). \tag{6, 7}$$
- **NE-driven novelty detection**:
  $$\text{Task\_Novel} = (\text{NE} > ne_{\text{threshold}}). \tag{8}$$

**3.2 ACh / NE update from reward prediction error.**
Each task carries a learned reward predictor $\tilde R_i(s, a)$ approximating $R_i$. Per transition:
$$\text{RPE}_i = \tilde R_i(s, a) - r. \tag{9}$$
Running mean / std of RPE for the *current* task are maintained. The "Expected" flag:
$$\text{Expected}_i = (|\text{RPE}_i - \text{RPE}_{\text{mean}}| < k \cdot \text{RPE}_{\text{std}}). \tag{10}$$
ACh multiplicative update:
$$\text{ACh}_i = \begin{cases} \min(ch_{\max},\, \text{ACh}_i \cdot ch_{\text{expected}}) & \text{if Expected}_i \\ \max(ch_{\min},\, \text{ACh}_i \cdot ch_{\text{unexpected}}) & \text{otherwise} \end{cases} \tag{11}$$
with $ch_{\text{expected}} > 1$, $ch_{\text{unexpected}} < 1$. NE update fires only at a task-change event:
$$\text{NE} = \begin{cases} \min(ne_{\max},\, \text{NE} \cdot ne_{\text{unmatched}}) & \text{if no Task\_Match}_i \\ ne_{\text{init}} & \text{otherwise} \end{cases} \tag{12}$$

**3.3 Complete system (Algorithm 1).** Each step:
1. Agent acts, gets reward.
2. Compute RPE, update ACh, check task change.
3. If no task change: train RL agent + reward predictor; update RPE statistics.
4. If task change: save current task's knowledge; for each known task $i$ check Task_Match; on match, reactivate that task's stored policy & predictor and reset modulators.
5. If still no match: update NE.
6. If NE > threshold: create a new policy + new reward predictor; $K \leftarrow K + 1$.

### 4. Experiments
**Gridworld** (`gym-minigrid`): three pickup tasks (pickup-green / pickup-blue / pickup-yellow). Reward +1 for correct pickup, −1 for wrong pickup. Task sequence: [green, blue, yellow, green, blue, yellow]. RL: PPO (Schulman et al. 2017).

**MuJoCo walker** (DeepMind control suite): three tasks (walker-stand / walker-walk / walker-run). Continuous control. Task sequence: [stand, walk, run, stand, walk, run, stand, walk, run]. RL: TD3 (Fujimoto et al. 2018).

**Hyperparameters** (Table 1, same for both): $ch_{\text{init}}=0.5$, $ch_{\max}=1.0$, $ch_{\min}=0.1$, $ch_{\text{change}}=0.2$, $ch_{\text{expected}}=1.1$, $ch_{\text{unexpected}}=0.9$, $p_{\text{match}}=0.5$, $\beta=2$, $ne_{\max}=1.0$, $ne_{\text{init}}=0.1$, $ne_{\text{unmatched}}=1.05$, $ne_{\text{threshold}}=0.9$.

### 5. Results
**5.1 RL performance (Fig. 3).** With neuromodulation, immediate high score on recurring tasks; without, learning from scratch each task. Ablations:
- **No NE**: still detects change, but cannot recognize prior tasks → treats every new task as novel → learns from scratch every time.
- **No NE, no ACh**: cannot detect change at all → one big policy contaminated by all tasks. Gridworld: negative transfer (right object for one task = wrong for next). Walker: some positive transfer (walk benefits from stand) but still inferior.

**Performance recovery (Table 2): timesteps to recover 90% performance on a recurring task.**

| Task | Ours | No NE | No NE, No ACh |
|---|---|---|---|
| pickup-green | 150.3 | 45,060.5 | 59,753.3 |
| pickup-blue | 130.7 | 25,510.7 | 69,535.2 |
| pickup-yellow | 143.3 | 29,428.7 | 87,605.6 |
| walker-stand | 796.0 | 311,000 | 354,500 |
| walker-walk | 1843.0 | 411,500 | 598,000 |
| walker-run | 1542.2 | 501,000 | 585,333.3 |

**5.2 Neuromodulator activity (Figs. 4, 5).** ACh: the entry for the *correct* task is always highest. NE: spikes on the first and second task switches (when the second and third tasks are genuinely novel); spikes are subthreshold for subsequent switches (because they re-match a stored task).

### 6. Conclusion
A neuromodulator-inspired system over standard deep RL enables rapid detection of and adaptation to environment changes. Demonstrated in gridworld + walker robot. Bridges Yu–Dayan uncertainty decomposition to RL.

## Phase 1 — Undergraduate-level synthesis

**Key idea.** When your RL agent is trained on Task A and the world quietly switches to Task B, the standard gradient-descent training loop is dangerous: every reward it gets from B updates the policy that was good for A, eroding A. The fix is **not** to make the underlying RL algorithm fancier — it's to put a tiny watchdog *on top* of any standard RL algorithm. The watchdog has two jobs:

1. **Notice that something changed.** The agent runs a small side network that predicts what reward it expects in the current state-action. If that prediction error suddenly stops being normal-sized, "we're not in the same task anymore". This is the **acetylcholine (ACh)** vector — one entry per task the agent has seen, each entry rising when reward predictions are accurate and falling when they're not. If *every* entry is low, "I'm lost — new task".
2. **Decide: have I seen this task before?** Take a softmax over the ACh vector. If one entry is clearly the winner, reactivate that task's stored policy and predictor — *don't retrain from scratch*. If no entry wins, the **norepinephrine (NE)** scalar rises. Once NE crosses a threshold, the watchdog says "okay, this is genuinely new, allocate a new policy slot" and the RL algorithm trains from scratch on a *fresh* network, leaving the old ones untouched.

**Setup.** Run two test environments. In gridworld, the agent has to pick up one of three colored objects; the "correct" color cycles green → blue → yellow → green → blue → yellow. In MuJoCo, a simulated 2D bipedal walker has to stand, then walk, then run, then re-do each. The standard PPO/TD3 trainers don't know the task boundaries — they just see state, action, reward.

**Result.** First time the agent meets each task, the watchdog correctly notices it's novel (NE crosses threshold), allocates a new slot, learns. Second time the agent meets a task, the watchdog correctly recognizes the task (an ACh entry softmaxes above 0.5), recalls the stored policy, and reaches 90% performance within ~150 timesteps in gridworld, ~800–1800 in walker. Without the NE half: the agent fails to recognize anything, learns from scratch every time, takes 25,000–600,000 timesteps. Without ACh either: the agent doesn't even know the task changed; everything gets tangled.

**Concrete instantiation.** The first time green is the target the agent learns "go to green". Then yellow becomes correct; the agent's predictor for "Task 1" starts saying "+1" but the world hands out 0 or −1; ACh[1] tumbles below threshold; no other task slot has high confidence; NE rises; new policy allocated. Five minutes later the world switches back to green. The agent acts (probably wrongly at first using its current policy); reward feedback comes; ACh[1] (still stored from before) rises above 0.5 in the softmax; the watchdog says "this is Task 1, swap in Policy 1"; the agent immediately resumes pickup-green at near-full performance.

## Phase 2 — Graduate-level deep dive

### Architecture in one figure (verbal)

For each of the $K$ known tasks the agent maintains:
- A policy $\pi_i(a|s)$ (PPO actor-critic networks in gridworld, TD3 actor + twin critics in walker).
- A reward predictor $\tilde R_i(s, a) \approx R_i(s, a)$, learned by regression.
- A scalar $\text{ACh}_i \in [ch_{\min}, ch_{\max}]$.

Plus one global scalar $\text{NE} \in [ne_{\text{init}}, ne_{\max}]$.

Plus per-task running statistics $\text{RPE}_{\text{mean}}, \text{RPE}_{\text{std}}$ of recent reward-prediction errors for the *currently active* task.

### Reward prediction error as the modulator's input

At each timestep, given the currently active task $i^\star$:

$$
\text{RPE}_{i^\star,t} = \tilde R_{i^\star}(s_t, a_t) - r_t. \tag{9}
$$

The predictor is trained online to minimize $\mathbb{E}[(\tilde R_i(s,a) - r)^2]$ via standard supervised regression on a replay buffer. Two regimes:
- **In-distribution**: $|\text{RPE}_{i^\star} - \text{RPE}_{\text{mean}}| < k \cdot \text{RPE}_{\text{std}}$; the predictor is right within the chronic noise level of the current task — *expected uncertainty*.
- **Out-of-distribution**: $|\text{RPE}_{i^\star} - \text{RPE}_{\text{mean}}| > k \cdot \text{RPE}_{\text{std}}$; the predictor's regret is structurally too big — *unexpected uncertainty*.

The flag $\text{Expected}_{i^\star}$ drives the ACh multiplicative dynamics (Eq. 11). Note that $k$ is a hyperparameter; the paper doesn't quote a numerical value in the main text, but a typical Yu–Dayan-style $k \in [2, 3]$ gives a per-step false-alarm rate of ~5% under Gaussian RPE.

### ACh dynamics as a stochastic geometric process

Let $g_t = ch_{\text{expected}} = 1.1$ if expected, $ch_{\text{unexpected}} = 0.9$ otherwise. Then:

$$
\text{ACh}_{i^\star,t+1} = \min(ch_{\max},\, \max(ch_{\min},\, g_t \cdot \text{ACh}_{i^\star,t})). \tag{11}
$$

Ignoring saturation, $\log \text{ACh}_{i^\star}$ executes a random walk with per-step expected drift $\log g_t$. With per-step expected-event probability $p_e$ (the predictor's hit rate in the *correct* task), the expected drift is $p_e \log 1.1 + (1 - p_e) \log 0.9$. For the random walk to move *up* on average requires:

$$
p_e \log 1.1 + (1 - p_e) \log 0.9 > 0 \quad\Longrightarrow\quad p_e > \frac{|\log 0.9|}{\log 1.1 + |\log 0.9|} \approx 0.525.
$$

In other words, the predictor only needs to be marginally better than chance — ~53% expected events — to push ACh of the correct task to the ceiling. Conversely, a predictor that's wrong > 47% of the time on the current task collapses to the floor. This explains why the system works robustly across vastly different RL algorithms: as long as the reward predictor is good enough to clear 53% expected events on the correct task, the modulator latches.

**Task-change threshold.** $ch_{\text{change}} = 0.2$, $ch_{\min} = 0.1$. So Task_Change fires only after several consecutive unexpected events drag ACh from the ceiling 1.0 below 0.2. The number of steps required is $\lceil \log(0.2/1.0)/\log 0.9 \rceil = \lceil 1.61/0.105 \rceil = 16$ unexpected events in a row. With a 47%-correct predictor, ~16 misses out of $\sim 16/0.47 \approx 34$ trials triggers detection — sufficiently fast.

### Task-match softmax

Once Task_Change fires, ACh is re-evaluated under softmax with $\beta = 2$ (sharp). For a single task to win with $P_i > p_{\text{match}} = 0.5$:

$$
\frac{\exp(2 \cdot \text{ACh}_i)}{\sum_j \exp(2 \cdot \text{ACh}_j)} > 0.5.
$$

With $K=3$ and other ACh entries at $ch_{\min}=0.1$, $\exp(2 \cdot 0.1) = 1.22$; the winning entry needs $\exp(2 \cdot \text{ACh}_i) > 2 \times 1.22 = 2.44$, so $\text{ACh}_i > 0.45$. This is feasible because the agent's reward predictor for the recalled task is still calibrated — initial probes with the candidate stored policy generate accurate predictions, raising that task's ACh quickly past 0.45.

### NE dynamics

NE only updates on task-change events (Eq. 12). If no task matches: $\text{NE} \leftarrow \min(ne_{\max}, \text{NE} \cdot 1.05)$. If some task matches: $\text{NE} \leftarrow ne_{\text{init}} = 0.1$ (full reset). Starting from $ne_{\text{init}} = 0.1$, hitting $ne_{\text{threshold}} = 0.9$ requires $\lceil \log(0.9/0.1)/\log 1.05 \rceil = \lceil 2.20/0.0488 \rceil = 46$ consecutive unmatched events. The 46 is divided across multiple Task_Change probings — when initial recall attempts fail, the system spends several Task_Change cycles ruling out each known task before NE finally crosses, at which point novel-task allocation fires.

**Why this is hysteretic.** First task switch ever: $K=1$, no other tasks to try, NE rises immediately. Second task switch: $K=2$, must rule out one stored task before NE rises. Third task: $K=3$, must rule out two stored tasks. By the fourth task switch (returning to Task 1), the system rules out Task_2 and Task_3 quickly because their ACh stay low (their predictors aren't tracking the current reward); Task_1's ACh rapidly rises; match fires; NE never crosses; no new task allocation.

### Comparison with Zou et al. 2020

| Aspect | Zou 2020 (perception) | Xing 2022 (RL) |
|---|---|---|
| ACh "signal" | correctness of goal-driven prediction | reward prediction error vs running stats |
| NE "signal" | wrong/right counter | task-change unmatched counter |
| Goal/Task count $K$ | fixed at 4 (predefined goals) | grows dynamically (new tasks allocated on the fly) |
| Memory of past | none — single network | per-task policy + predictor saved |
| Reset behavior | reset ACh & NE to baseline | reactivate stored module instead |
| Decision step | softmax over ACh chooses guessed goal | softmax over ACh decides match vs novel |

The key advance in 2022 is **memory** — the agent stores policies and predictors, so recall is genuinely free (no relearning) instead of merely "fast adaptation". This is the leap from goal *discovery* (Zou 2020) to lifelong *retention* (Xing 2022).

### Mapping to Doya 2002

- DA = TD error: this is what the underlying RL algorithm (PPO, TD3) already uses to update the policy.
- ACh = expected uncertainty / signal-to-noise on the current task model: implemented as the ACh vector.
- NE = unexpected uncertainty / novelty: implemented as the NE scalar.
- 5-HT = temporal discounting: not modeled; $\gamma$ is fixed per task.

The Doya framework is operationalized but only partially — Xing & colleagues focus on ACh + NE because the task-change-and-recall problem is the natural target for those two; the patience/discount knob is left to Xing 2020 (their earlier paper).

### Parameter table

| Symbol | Value | Role |
|---|---|---|
| $ch_{\text{init}}$ | 0.5 | Initial ACh for a fresh task slot |
| $ch_{\max}$ | 1.0 | Upper bound on ACh |
| $ch_{\min}$ | 0.1 | Lower bound on ACh |
| $ch_{\text{change}}$ | 0.2 | Task_Change threshold (max ACh below this) |
| $ch_{\text{expected}}$ | 1.1 | ACh up-gain on expected RPE |
| $ch_{\text{unexpected}}$ | 0.9 | ACh down-gain on unexpected RPE |
| $p_{\text{match}}$ | 0.5 | Softmax threshold for Task_Match |
| $\beta$ | 2 | Softmax inverse temperature |
| $ne_{\max}$ | 1.0 | NE ceiling |
| $ne_{\text{init}}$ | 0.1 | NE reset value |
| $ne_{\text{unmatched}}$ | 1.05 | NE up-gain on no match |
| $ne_{\text{threshold}}$ | 0.9 | NE threshold for Task_Novel |
| $k$ | (paper unspecified) | RPE std-multiplier for Expected flag |

### Limitations the paper does not flag

- The shared state-action space ($S$, $A$, $T$ identical across tasks) is restrictive — a generalization that changes dynamics would require also storing transition models.
- Number of detected tasks $K$ grows unboundedly; no consolidation/merging mechanism. The paper assumes $K$ stays small in practice.
- The reward predictor needs the *agent's* current actions as input; if the recalled policy hasn't been activated yet, the system probes briefly with the current policy before deciding match.
- $\beta=2$ and $p_{\text{match}}=0.5$ together are critical — too small $\beta$ blurs tasks; too large $\beta$ over-commits early.

## Connections

**Direct references inside this corpus:**

- **[xing_2020_neuromodulated_patience](xing_2020_neuromodulated_patience.md)** — Ref [6]; same first author. Xing 2020 puts 5-HT in the outer navigation loop; Xing 2022 puts ACh+NE in the RL inner loop. Sibling papers in the same lab's neuromodulator-for-autonomous-agents program.
- **[zou_2020_neuromodulated_attention](zou_2020_neuromodulated_attention.md)** — Ref [12]; same lab. **This paper is the explicit RL-domain extension of Zou 2020's perception-domain ACh+NE framework**. The update rules are isomorphic; the difference is reward-prediction-error vs goal-correctness as the modulator signal, and dynamic per-task memory.
- **[avery_krichmar_2017_models_neuromodulation](avery_krichmar_2017_models_neuromodulation.md)** — implicit ancestor. Same lab's authoritative neuromodulation review.
- **[hwu_krichmar_2020_schemas_memory](hwu_krichmar_2020_schemas_memory.md)** — same lab, same DARPA L2M contract. Both address catastrophic forgetting via context-aware indexing + neuromodulated rapid encoding. Hwu's model uses CHL + HPC indexing; Xing's uses RL + ACh/NE softmax. Same conceptual family.
- **[doya_2002_metalearning_neuromodulation](doya_2002_metalearning_neuromodulation.md)** — foundational. Doya's ACh/NE/DA/5-HT meta-parameter framework is the implicit scaffold. Not cited by name but the lineage runs through Yu & Dayan (Ref [11]) which is the modern Bayesian heir.
- **[krichmar_hwu_2022_design_principles_neurorobotics](krichmar_hwu_2022_design_principles_neurorobotics.md)** — Krichmar & Hwu's 2022 design-principles paper post-dates this one and uses Xing 2022 as a worked example of "principle: neuromodulation for lifelong RL".

**Expected forward citations.** Lee et al. 2024 ("Lifelong reinforcement learning via neuromodulation") and Ben-Iwhiwhu et al. 2022 ("Context meta-reinforcement learning via neuromodulation") sit directly downstream of this paper conceptually. Vecoven et al. 2020 and Wang et al. 2024 ("Neuromodulated meta-learning") share the same agenda from different angles.

**External anchors.** Yu & Dayan 2005 (uncertainty + neuromodulation); Bouret & Sara 2005 (NE-driven network reset); Grella et al. 2019 (LC phasic activation); Schulman et al. 2017 (PPO); Fujimoto et al. 2018 (TD3); Tassa et al. 2018 (DeepMind control suite); Chevalier-Boisvert et al. 2018 (gym-minigrid).
