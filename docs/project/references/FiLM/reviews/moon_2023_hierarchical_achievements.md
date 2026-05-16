---
title: "Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning"
authors: ["Seungyong Moon", "Junyoung Yeom", "Bumsoo Park", "Hyun Oh Song"]
year: 2023
venue: "NeurIPS 2023"
slug: moon_2023_hierarchical_achievements
source_pdf: docs/project/references/FiLM/sources/Moon et al. 2023 - Discovering hierarchical achievements in reinforcement learning via contrastive learning.pdf
topic: FiLM
---

# Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning

## Plain-English entry point

The Crafter benchmark is a 2-D Minecraft-style survival game. Each episode procedurally generates a new world, and the agent unlocks "achievements" — sub-goals like *collect wood*, *make wood pickaxe*, *collect iron*, *collect diamond* — that form a dependency tree (a wood pickaxe requires wood; an iron pickaxe requires wood, stone, and a furnace; a diamond requires the iron pickaxe). Solving Crafter therefore requires **long-horizon planning** and **generalization across procedurally-generated maps**, which is why prior work has favoured heavyweight model-based agents like DreamerV3 or hierarchical planners that explicitly reconstruct the achievement graph.

This paper makes two contributions. **(1)** With a few modern implementation tricks — bigger network, layer normalisation, value normalisation — plain **PPO** (Proximal Policy Optimization, a standard model-free policy-gradient method) outperforms DreamerV3 on Crafter at 1 M environment steps using 50× fewer parameters. **(2)** A new auxiliary objective called **achievement distillation** further boosts PPO via **contrastive learning**: the encoder is trained so that the latent of "current state-action" lies close to the latent of "the achievement that will be unlocked next" and far from random non-achievement states. A second cross-trajectory objective uses **partial optimal transport** to match achievement sequences across two episodes, so that the same achievement (e.g. *collect wood*) gets a consistent representation regardless of the procedurally-generated map. Together these give a state-of-the-art Crafter score of 21.8 % with 9 M parameters versus DreamerV3's 14.8 % at 201 M.

For this FiLM-corpus review, the load-bearing piece is **§4.1**: to fuse a discrete action vector $a_t$ into a CNN state representation $\phi_\theta(s_t)$, the authors **explicitly use a FiLM layer** ("a FiLM layer [Perez et al. 2018]"). FiLM here is the action-on-state conditioner inside the auxiliary representation head, not on the policy itself — a smaller, more targeted use than BC-Z, but a direct citation of the canonical FiLM mechanism.

## Section-ordered backbone

**1. Introduction.** Generalization and long-term reasoning in procedurally generated environments are the open problem. Existing approaches are either model-based (latent world models — DreamerV3, MuZero+SPR) or hierarchical (explicit graph + high-level planner — HAL, SEA). Both demand huge networks or large pretraining datasets. The paper asks: can a clean PPO baseline reach this regime? And can a *contrastive auxiliary task on achievements* close the rest of the gap without an explicit planner?

**2. Preliminaries.** Defines an MDP with hierarchical achievements as $\mathcal{M}_i = (\mathcal{S}_i, \mathcal{A}, \mathcal{G}, p, r, \rho_i, \gamma)$ where $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is a DAG of achievements. The achievement reward is 1 the first time each achievement is unlocked in an episode, 0 otherwise. The agent has *no* access to the graph topology — it must infer achievement structure from the reward signal alone. Crafter is the benchmark, with 22 achievements organised as the tree in Fig. 1b. The PPO preliminaries section reviews the clipped surrogate objective and the shared-encoder image-based RL setup.

**3. Motivation.** Two empirical findings drive the method. *(3.1)* A "modernised" PPO (channels $[16,32,32] \to [64,128,128]$, hidden $256 \to 1024$, layer norm, value normalisation) jumps from Crafter score 8.17 to 15.60, beating DreamerV3 (14.77) with a fraction of the parameters. *(3.2)* A linear probe on PPO's frozen encoder predicts the next achievement at 44.9 % top-1 accuracy in the 22-way task — non-trivial — but the *confidence* of those predictions is poor (median 0.24). PPO's encoder *contains* achievement-related information, but it is not strongly enough organised in the latent space.

**4. Contrastive learning for achievement distillation.** The core method, split into:
- **§4.1 Intra-trajectory achievement prediction.** Inside a single episode, treat the next-unlocked achievement $g_t^+$ as the *anchor*, $(s_t, a_t)$ as the *positive*, and a randomly sampled state-action pair from the same episode as the *negative*. Maximise cosine similarity between anchor and positive, minimise it for the anchor and negative, via an InfoNCE-style contrastive loss with temperature $\lambda$. **The state-action representation $\psi_\theta(s_t, a_t)$ is computed by feeding the state embedding $\phi_\theta(s_t)$ through a FiLM layer modulated by the action $a_t$, then through an MLP, then normalised.** The achievement representation $\nu_\theta(g)$ is computed as the *normalised residual* between the encoder embeddings of the two consecutive states bracketing the achievement, following Nair et al.
- **§4.2 Cross-trajectory achievement matching.** Take two episodes' achievement sequences $g = (g_i)_{i=1}^m$ and $g' = (g'_j)_{j=1}^n$, build a cost matrix $M_{ij} = 1 - \nu_\theta(g_i)^\top \nu_\theta(g'_j)$ of cosine distances, solve a *partial entropic optimal transport* problem to soft-match achievements across episodes, threshold at 0.5 to get hard matches, and run a second contrastive loss that pulls together representations of matched achievements across procedurally-distinct worlds.
- **§4.3 Achievement representation as memory.** The previous achievement embedding $\nu_\theta(g_t^-)$ is concatenated with the state embedding $\phi_\theta(s_t)$ and fed to the policy and value heads — so the policy is *conditioned on which achievement was last unlocked*. The same previous-achievement embedding is also fed into the prediction head as a "forward dynamics in achievement space" signal.
- **§4.4 Integration with PPO.** Two alternating phases: a *policy phase* runs standard PPO for $N_\pi$ iterations, filling a buffer; an *auxiliary phase* optimises $\mathcal{L}_{\text{pred}}$ and $\mathcal{L}_{\text{match}}$ over the buffer for $E_{\text{aux}}$ iterations, with policy- and value-output regularisers to prevent encoder drift from destabilising the policy.

**5. Experiments.** Crafter, 1 M environment steps, 10 seeds. The method beats every from-scratch baseline (DreamerV3, LSTM-SPCNN, MuZero+SPR, SEA) by a wide margin: score 21.79 % vs. PPO 15.60 % vs. DreamerV3 14.77 %, while using 9 M parameters (vs. 201 M for DreamerV3, 135 M for LSTM-SPCNN, 54 M for MuZero+SPR). Ablations show intra-trajectory prediction is the biggest contributor (+3.4 pts over PPO), cross-trajectory matching adds another +1.3, memory adds another +1.4. Linear-probe accuracy on the trained encoder jumps from PPO's 44.9 % to 73.6 %, and confidence median from 0.24 to 0.75 — confirming that the contrastive objective actually changes representation quality, not just downstream performance. Extensions to QR-DQN (off-policy, value-based) and to ProcgenHeist + MiniGrid environments both show large gains, demonstrating the recipe is not Crafter-specific.

**6. Related Work.** Positioned against model-based (DreamerV3, MuZero+SPR), hierarchical (HAL, SEA), and pure-representation-learning (SPR) prior work. The novelty is using *partial optimal transport* over discovered achievements as a cross-episode regulariser in an *online* setting (most prior OT-in-RL is imitation learning).

**7. Conclusion.** A self-supervised auxiliary objective can substitute for explicit long-term planning in hierarchical-achievement RL, at a fraction of the parameter and data budget of model-based methods. Limitation: still requires the per-achievement reward signal; truly unsupervised version remains future work.

## Phase 1 — Undergraduate-level synthesis

**The problem.** Crafter is a procedurally-generated 2-D survival game with 22 hierarchical achievements. To collect a diamond you must first make an iron pickaxe; to make that you need iron, which requires a stone pickaxe, which requires a wood pickaxe, which requires wood. The agent only learns from sparse rewards (+1 when an achievement is unlocked for the first time in an episode). The challenge is *long-horizon planning* without any explicit knowledge of the achievement graph.

**The PPO surprise.** Plain PPO with a few standard improvements (bigger network + layer norm + value normalisation) already beats the previous state-of-the-art (model-based DreamerV3) on Crafter while being 22× smaller.

**The new auxiliary objective.** PPO's encoder *partially* knows what the next achievement will be (it predicts it 45 % of the time when probed), but its predictions are unconfident — the latent space isn't well-organised. The fix is *contrastive learning on achievements*:

- **Intra-trajectory** — train the encoder so that the state-action pair $(s_t, a_t)$ sits close in latent space to "the next achievement to be unlocked", and far from a random other state-action pair from the same episode. A **FiLM layer** is used here to fuse the action into the state embedding before the contrastive loss is computed.

- **Cross-trajectory** — across two different episodes, find which achievements correspond to each other using *optimal transport* (a way to "align" two sequences of items), then pull matched achievements together in latent space. This forces the representation of, e.g. "collect wood" to be consistent across procedurally different worlds.

- **Memory** — feed the encoder embedding of the *previously unlocked* achievement back into the policy and value heads.

**The result.** SOTA on Crafter (21.79 % vs. DreamerV3's 14.77 %) at 9 M parameters vs. 201 M. The contrastive auxiliary task does the work that an explicit planner used to do.

## Phase 2 — Graduate-level deep dive

### 2.1 Hierarchical-achievement MDP and reward structure

For each $\mathcal{M}_i \in \mathcal{M}$, with achievement graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ and binary unlock vector $b \in \{0,1\}^{|\mathcal{V}|}$, the reward is

$$
r(s_t, a_t, s_{t+1}) \;=\; \begin{cases} 1 & \exists\, v_i \in \mathcal{V} :\; b[i] = 0 \text{ and } c(s_t, a_t, s_{t+1}) = v_i, \\ 0 & \text{otherwise}, \end{cases}
$$

where $c: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathcal{V} \cup \{\emptyset\}$ identifies which (if any) achievement is unlocked by the transition. Critically, the agent has no access to $\mathcal{V}$, $\mathcal{E}$, or $b$ — only to the resulting $\{0,1\}$ reward.

### 2.2 PPO baseline

The clipped surrogate (paper's notation):

$$
J_\pi(\theta) \;=\; \mathbb{E}_{(s_t, a_t) \sim \mathcal{T}}\!\left[ \min\!\left( \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat A_t,\; \mathrm{clip}\!\left( \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)},\, 1 - \epsilon,\, 1 + \epsilon \right) \hat A_t \right) \right],
$$

paired with the value objective

$$
J_V(\theta) \;=\; \mathbb{E}_{s_t \sim \mathcal{T}}\!\left[ \tfrac{1}{2}\bigl(V_\theta(s_t) - \hat V_t\bigr)^2 \right], \qquad \hat V_t \;=\; \hat A_t + V_{\theta_{\text{old}}}(s_t).
$$

The encoder $\phi_\theta : \mathcal{S} \to \mathbb{R}^h$ is shared between $\pi$ and $V$ via linear heads.

### 2.3 Intra-trajectory contrastive objective (with FiLM-fused action)

Let $g_t^+ = g_u$ where $u = \min\{i : t \le t_i\}$ — the index of the next achievement unlocked after time $t$. The achievement representation is

$$
\nu_\theta(g_i) \;=\; \frac{\phi_\theta(s_{t_{i+1}}) - \phi_\theta(s_{t_i})}{\bigl\|\phi_\theta(s_{t_{i+1}}) - \phi_\theta(s_{t_i})\bigr\|_2},
$$

i.e. the normalised *latent residual* across the transition that completed the achievement (motivated by Nair et al.). The state-action representation is

$$
\psi_\theta(s_t, a_t) \;=\; \mathrm{normalize}\Bigl( \mathrm{MLP}\bigl( \mathrm{FiLM}(\phi_\theta(s_t);\, a_t) \bigr) \Bigr),
$$

with $\mathrm{FiLM}(\phi(s); a)$ producing channel-wise scales $\gamma(a)$ and shifts $\beta(a)$ from the action vector and applying $\widetilde\phi_c = \gamma_c(a)\,\phi_c(s) + \beta_c(a)$. The InfoNCE contrastive loss with temperature $\lambda > 0$:

$$
\mathcal{L}_{\text{pred}}(\theta) \;=\; -\,\mathbb{E}_{(s_{t'},a_{t'}) \sim \tau,\, (s_t,a_t) \sim \tau}\!\left[ \log \frac{\exp\!\bigl(\psi_\theta(s_t, a_t)^\top \nu_\theta(g_t^+) / \lambda\bigr)}{\exp\!\bigl(\psi_\theta(s_t, a_t)^\top \nu_\theta(g_t^+) / \lambda\bigr) + \exp\!\bigl(\psi_\theta(s_{t'}, a_{t'})^\top \nu_\theta(g_t^+) / \lambda\bigr)} \right].
$$

This is the canonical contrastive-prediction objective for achievement discovery the curator should cite when grouping with other auxiliary-task RL papers.

### 2.4 Cross-trajectory achievement matching via partial optimal transport

Given two achievement sequences $g = (g_i)_{i=1}^m$ and $g' = (g'_j)_{j=1}^n$, the cost matrix is

$$
M_{ij} \;=\; 1 - \nu_\theta(g_i)^\top \nu_\theta(g'_j) \;\in\; \mathbb{R}^{m \times n}.
$$

Treating both sequences as uniform discrete distributions, the entropic partial-OT problem is

$$
T \;=\; \arg\min_{T \ge 0}\; \langle T, M \rangle + \alpha \sum_{i,j} T_{ij} \log T_{ij},
$$

subject to the partial-marginal constraints

$$
T\,\mathbf 1 \;\le\; \tfrac{1}{m}\mathbf 1, \qquad T^\top \mathbf 1 \;\le\; \tfrac{1}{n}\mathbf 1, \qquad \mathbf 1^\top T \mathbf 1 \;=\; \tfrac{\min(m,n)}{m\,n},
$$

solved by iterative Bregman projection (Benamou et al. 2015). The total transported mass equals the shorter sequence length — but because not every achievement in one episode appears in the other, a hard matching is obtained by

$$
T^\star_{ij} \;=\; \mathbf 1[T_{ij} > 0.5],
$$

which enforces at-most-one matching per achievement. The contrastive loss on matched achievements is

$$
\mathcal{L}_{\text{match}}(\theta) \;=\; -\,\mathbb{E}_{g_i \sim g,\, g'_j \sim g'}\!\left[ \log \frac{\exp\!\bigl(\nu_\theta(g_i)^\top \nu_\theta(g'_k) / \lambda\bigr)}{\exp\!\bigl(\nu_\theta(g_i)^\top \nu_\theta(g'_k) / \lambda\bigr) + \exp\!\bigl(\nu_\theta(g_i)^\top \nu_\theta(g'_j) / \lambda\bigr)} \right],
$$

where $g'_k$ is the achievement matched to $g_i$ by $T^\star$ and $g'_j$ is a random negative drawn from the target sequence.

### 2.5 Policy and value regularisers

To stop the encoder updates of the auxiliary phase from destabilising the policy:

$$
R_\pi(\theta) \;=\; \mathbb{E}_{s_t \sim \tau}\!\bigl[D_{\mathrm{KL}}\bigl(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \,\|\, \pi_\theta(\cdot \mid s_t)\bigr)\bigr], \qquad R_V(\theta) \;=\; \mathbb{E}_{s_t \sim \tau}\!\left[ \tfrac{1}{2}\bigl(V_\theta(s_t) - V_{\theta_{\text{old}}}(s_t)\bigr)^2 \right],
$$

so the encoder representations are sculpted by $\mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{match}}$ while $\pi$ and $V$ remain anchored to their values at the end of the policy phase.

### 2.6 Memory: policy conditioning on the previous achievement

Given state $s_t$ and previous-achievement label $g_t^- = g_\ell$ with $\ell = \max\{i : t > t_i\}$, the policy and value inputs become

$$
\bigl[\phi_\theta(s_t);\, \nu_\theta(g_t^-)\bigr] \;\longrightarrow\; \pi_\theta(\cdot \mid s_t, g_t^-),\; V_\theta(s_t, g_t^-).
$$

The same $\nu_\theta(g_t^-)$ is also concatenated with $a_t$ and the FiLM-modulated state embedding to form the state-action representation used in $\mathcal{L}_{\text{pred}}$ — making that loss learn *forward dynamics in achievement space conditioned on the previous achievement*.

### 2.7 Why this is FiLM-corpus-relevant

The paper does **not** use FiLM as the global conditioning mechanism the way BC-Z does. FiLM appears as a *small, local* design choice — fusing a low-dim discrete action vector into a CNN-derived state embedding inside an auxiliary representation head — and the authors explicitly cite Perez et al. 2018. The interesting comparison for the curator: in BC-Z, FiLM is *the* conditioning channel (task → policy); in Moon 2023, FiLM is a *micro-tool* for action conditioning inside a contrastive-prediction module. The same operator, deployed at different scales of the architecture.

## Connections

- **`perez_2018_film.md` (B1, canonical FiLM)** — Moon 2023 cites Perez et al. 2018 directly in §4.1 for the action-conditioning FiLM layer. The use here is local (action $\to$ state-embedding) rather than global (task $\to$ entire network), illustrating how FiLM scales down to small fusion modules.
- **`jang_2022_bcz.md` (B4, this batch)** — Different deployment scale of the same operator. BC-Z uses FiLM as the *primary* conditioning channel of a 7-DoF visuomotor policy; Moon 2023 uses FiLM as an auxiliary fusion gadget inside a contrastive head. The curator should note that the FiLM corpus contains both extremes.
- **Project-side relevance (NMN-as-modulator)** — Moon 2023 is the corpus's cleanest example of a low-bandwidth signal (a discrete action) modulating a high-dim image embedding *via FiLM* inside an auxiliary self-supervised head, with the resulting representation feeding back into the policy. This is the architectural template for "an interoceptive variable modulates a self-supervised representation head whose output is fed into the policy" — directionally relevant to NMN-style modulators.
- **Hand-off** — If the project wants to test "achievement distillation as a way to evaluate hierarchical generalization in our grid-world", that is a `senior-developer` issue plan, not within scope for this reviewer.
