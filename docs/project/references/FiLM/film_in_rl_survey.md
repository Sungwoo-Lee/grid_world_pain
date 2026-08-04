---
title: "FiLM and Conditional Modulation in Reinforcement Learning — Focused Survey + Gap List"
topic: FiLM
status: curated
created: 2026-08-04
last_updated: 2026-08-04
related:
  - film_modulation_granularity_synthesis.md
  - film_synthesis.md
  - film_lit_review.md
  - ../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md
  - ../Hypernetwork/hypernetwork_lit_review.md
  - ../TD/TD_lit_review_A_distributional_rl.md
  - ../TD/TD_lit_review_C_hypernet_rl.md
  - ../Gamma/Gamma_lit_review_B_multi_horizon.md
  - ../Temperature/Temperature_lit_review_B_sac_variants.md
scope: |
  Cross-corpus survey of how feature-wise / conditional modulation is actually
  used inside reinforcement-learning agents, drawn from RL papers scattered
  across eleven reference folders (FiLM, neuromodulatory_algorithms,
  Hypernetwork, TD, Temperature, continual_learning, in_context_learning,
  Dreamer, Gamma, Learning_Rate, uncertainty). Reuses the granularity /
  placement / parameterisation / grouping taxonomy defined in
  film_modulation_granularity_synthesis.md and
  neuromod_modulation_scope_synthesis.md. Second deliverable: a prioritised
  list of canonical FiLM-in-RL papers the corpus does not contain.
---

# FiLM and Conditional Modulation in Reinforcement Learning

## 1. Plain-English entry point

Many neural networks are built so one signal can **steer** another network's
computation. The most common recipe is called **FiLM** — "feature-wise linear
modulation": take a hidden layer's activity, multiply it by a learned gain and
add a learned offset, where both numbers come from a small side-network reading
a *steering signal*. In the original image-and-language work the steering signal
is a question ("what colour is the cube left of the sphere?"). This document
asks a narrower question: **when reinforcement-learning agents use this trick,
what actually plays the role of the question, and which part of the agent gets
steered?**

The short answer, from every RL paper in our reference library: **the steering
signal is almost always something the agent's main network cannot see for
itself** — a language instruction, a task identifier, an inferred summary of
"which task am I in", a risk level, a discount factor, a novelty score. The one
design principle the RL modulation literature agrees on, sometimes explicitly,
is that the modulator should carry *extra* information, not a copy of what the
policy already receives. The clearest statement is in a 2020 meta-RL paper
(Vecoven et al.) that deliberately feeds its modulator the interaction history
**with the current observation removed**.

Our agent does the opposite. Our modulator is a small recurrent network that
reads the **raw current observation** — the same vector the main encoder reads —
and then modulates that same encoder. Exactly one paper in the library does
anything similar (Ben-Iwhiwhu et al. 2022, whose modulator neurons read their own
layer's input), and it modulates only the action-selection network, having found
that modulating the value estimator was unstable. That divergence, and five
others, are what the rest of this survey documents. The final section lists the
canonical RL-modulation papers we do not yet hold, ranked by how much a reviewer
would expect to see them cited.

---

## 2. Method, scope, and how this composes with the two syntheses

**How the corpus was searched.** Eleven reference folders were scanned for RL
vocabulary (policy, actor, critic, PPO, SAC, Q-learning, meta-RL, POMDP,
task/goal conditioning, exploration, world model) intersected with modulation
vocabulary (FiLM, gain, gate, modulate, condition, hypernetwork). Papers already
summarised in existing per-paper reviews were read from those reviews; source
PDFs were opened only where the RL-specific implementation detail was missing.
The project's own architecture was read from `src/models/neuromodulator.py` and
`src/models/recurrent_ppo_network.py` (read-only).

**Taxonomy — reused verbatim, not redefined.** Labels come from
[`film_modulation_granularity_synthesis.md` §2](film_modulation_granularity_synthesis.md)
and its sister
[`../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md` §2](../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md):

| Axis | Question | Cells / values |
|---|---|---|
| **Axis 1 — granularity** | how many distinct $(\gamma,\beta)$ values per modulated tensor | **(a)** per-unit · **(b)** per-channel (collapses to (a) in dense layers) · **(c)** grouped · **(d)** layer-scalar · **(e)** global |
| **Axis 2 — placement** | which layers / how many injection sites | early/late, one/many |
| **Axis 3 — parameterisation** | shared generator + per-site heads, vs. per-site generators, vs. broadcast unchanged | — |
| **Axis 4 — grouping / tying** | explicit grouping, low-rank tying, population partitioning | — |

Two refinements from the neuromodulation half are load-bearing here as well:
**support** (which units are reached) is not **resolution** (how many distinct
values exist), and **what is modulated** largely determines granularity — a
temperature is a scalar because there is only one action distribution, not
because anyone made a granularity choice.

**One new axis this survey adds.** RL modulation papers differ on a question the
vision literature never has to ask: **is the conditioner the same information
source as the modulated stream?** In vision-language FiLM the two are always
different (a question modulates an image). In RL they are usually different too —
but not always, and the exception matters for us. Call this the
**self-conditioning** axis; it is Q1's second column throughout.

---

## 3. Q1 — What is the conditioning signal in RL?

### 3.1 Per-paper table

"Generator input" is what the $(\gamma,\beta)$-producing side-network actually
reads. "Modulated stream" is what gets scaled/shifted. "Same source?" flags
whether the two are the same information.

| Paper (corpus) | Generator input (the "question") | Modulated stream | Same source? |
|---|---|---|---|
| [Jang et al. 2022 — BC-Z](reviews/jang_2022_bcz.md) (FiLM) | frozen 512-d sentence embedding of a **language instruction**, or a human-video embedding | ResNet-18 image features in a visuomotor policy | no |
| [Nikulin et al. 2023 — SAC-RND](reviews/nikulin_2023_anti_exploration_rnd.md) (FiLM) | the **raw state** $s$ (D4RL proprioceptive vector) | the **action** feature stream inside an RND novelty network | no |
| [Moon et al. 2023](reviews/moon_2023_hierarchical_achievements.md) (FiLM) | the **discrete action** $a_t$ | the CNN **state** embedding $\phi_\theta(s_t)$ in an auxiliary contrastive head | no |
| Dabney et al. 2018 — IQN ([TD shard A §3.3](../TD/TD_lit_review_A_distributional_rl.md#33-iqn-as-a-film-mechanism--the-load-bearing-project-connection)) | a **sampled quantile level** $\tau \sim U[0,1]$ (a risk knob), via a 64-term cosine basis | the conv **state embedding** $\psi(s)$ inside the Q-network | no |
| Vecoven et al. 2020 ([neuromod](../neuromodulatory_algorithms/reviews/vecoven_2020_neuromod_dnn.md)) | **context $c_t = h_t \setminus x_t$** — interaction history with the current observation *explicitly removed* | activation slope + bias of every hidden neuron, actor and critic | **no, by construction** |
| Ben-Iwhiwhu et al. 2022 ([neuromod](../neuromodulatory_algorithms/reviews/beniwhiwhu_2022_context_meta_rl.md)) | the **layer's own input** $x$ (which carries CAVIA's context $\phi$ or PEARL's latent $z$) | that same layer's standard-neuron pre-activations, policy only | **yes** |
| Beck et al. 2023 ([Hypernetwork §6](../Hypernetwork/hypernetwork_lit_review.md#paper-6)) | VariBAD **task embedding** $e$ from a recurrent VAE over the trajectory | FiLM baseline: policy-MLP activations. Hypernet: all policy weights | no |
| Schöpf et al. 2022 — HN-PPO ([Hypernetwork §5](../Hypernetwork/hypernetwork_lit_review.md#paper-5)) | learnable 8-d **task-ID embedding** | generated actor (and optionally critic) weights | no |
| Rezaei-Shoshtari et al. 2023 — HyperZero ([Hypernetwork §7](../Hypernetwork/hypernetwork_lit_review.md#paper-7)) | explicit **MDP descriptor** $(\psi,\mu)$ = reward parameters + dynamics parameters | generated actor **and** critic weights | no |
| Sarafian et al. 2021 ([TD shard C §1](../TD/TD_lit_review_C_hypernet_rl.md)) | the **state** $s$ as "meta-variable" | a small dynamic net consuming the **action** — i.e. the critic $Q(s,a)$ | no |
| Sherstan et al. 2020 — Γ-nets ([Gamma shard B](../Gamma/Gamma_lit_review_B_multi_horizon.md)) | the **discount factor** $\gamma$ (and horizon $\tau=1/(1-\gamma)$) | the value head's feature vector $\phi$ | no |
| Schaul et al. 2015 — UVFA / Borsa et al. 2018 — USFA ([Gamma shard A](../Gamma/Gamma_lit_review_A_universal_vf.md)) | **goal** $g$ / **policy descriptor** $\mathbf z$ | concatenated late, after the LSTM — not multiplicative | no |
| Lin et al. 2020 — CAT-SAC ([Temperature shard B](../Temperature/Temperature_lit_review_B_sac_variants.md)) | an **RND curiosity score** $c(s)$ — per-state novelty | the entropy temperature $\alpha_\delta(s)$ | derived from $s$, applied to the action distribution |
| Asadi & Littman 2017 — mellowmax ([Temperature shard C](../Temperature/Temperature_lit_review_C_softmax_operators.md)) | the **current $Q$-values** at that state | inverse temperature $\beta(s)$, solved by root-finding | yes (Q-values are internal) |
| Lee et al. 2024 ([neuromod](../neuromodulatory_algorithms/reviews/lee_2024_lifelong_rl.md)) | an **uncertainty decomposition** (expected vs. unexpected) | learning rate $\alpha$, softmax temperature $\beta$ | no |
| Doya 2002 ([neuromod](../neuromodulatory_algorithms/reviews/doya_2002_metalearning_neuromodulation.md)) | conceptual; each modulator keyed to a behavioural signature | $\alpha,\beta,\gamma,\delta$ (learning rate, temperature, discount, TD error) | n/a |
| Xu, van Hasselt & Silver 2018 ([Learning_Rate shard C](../Learning_Rate/Learning_Rate_lit_review_C_meta_gradient_rl.md)) | a **meta-gradient** on a separate policy-gradient meta-objective | $\gamma, \lambda$ — the definition of the return | n/a |
| Zou et al. 2020 / Xing et al. 2022 ([neuromod](../neuromodulatory_algorithms/reviews/xing_2022_neuromodulation_rl_environment_changes.md)) | ACh = a $K$-vector over **goals / stored tasks**; NE = a scalar surprise/reset detector | a goal prior, or which frozen policy slot is selected | no |
| Chen et al. 2021 — Decision Transformer ([TD shard D](../TD/TD_lit_review_D_decision_transformer.md)) | **return-to-go** scalar | attention over tokens — token-wise, not feature-wise | no |
| Hafner et al. 2020–2025 — Dreamer family ([Dreamer review](../Dreamer/dreamer_lit_review.md)) | — | **no feature-wise modulation anywhere**; conditioning is concatenation into the recurrent state-space model, and token insertion in Dreamer 4 | n/a |

### 3.2 Verdict on Q1

**Six kinds of conditioner exist in the RL corpus, and none of them is the raw
current observation feeding the observation pathway.**

1. **Language / instruction** — BC-Z. The closest thing to the vision-language
   convention: a frozen pretrained encoder produces $z$, FiLM heads translate it
   into per-channel gains.
2. **Task / MDP descriptor** — Beck (learned VariBAD embedding), Schöpf (learned
   8-d task ID), Rezaei-Shoshtari (explicit reward + dynamics parameters),
   Ben-Iwhiwhu (CAVIA context / PEARL latent). This is the largest family.
3. **A control knob with an algebraic meaning** — IQN's quantile level $\tau$,
   Γ-nets' discount $\gamma$, UVFA's goal $g$. These are "what do you want me to
   predict", not "where am I".
4. **An internal statistic** — mellowmax's $\beta(s)$ from the current
   $Q$-values, CAT-SAC's curiosity score, Lee's uncertainty decomposition. This
   is the family our modulator is *conceptually* closest to, though ours is
   learned end-to-end rather than computed by a fixed rule.
5. **The other half of the state–action pair** — Nikulin (state modulates action
   features), Moon (action modulates state features), Sarafian (state generates
   the weights that consume the action). All three are inside auxiliary or value
   machinery, never the policy trunk.
6. **History minus the present** — Vecoven. This is the only paper that argues
   the point explicitly, and it argues *against* what we do.

**Does anything feed the raw current observation to the generator?** Yes —
Nikulin and Sarafian both use the raw state as the conditioner. But in both, the
*modulated* stream is the action side, so the modulator is genuinely fusing two
different information sources. **No paper in the corpus uses the current
observation both as the modulator's input and as the modulated pathway.**
Ben-Iwhiwhu is the sole self-conditioning precedent, and its "input" is a
per-layer pre-activation, not a raw sensor vector routed through a separate
recurrent side-network.

This is the sharpest single finding of the survey and it is corroborated
independently in
[`../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md` §8.2](../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md),
which reaches the same verdict from the neuroscience side ("nothing in the corpus
feeds a modulator the same input the main network already sees").

---

## 4. Q2 — What part of the RL architecture gets modulated?

### 4.1 Per-paper table

| Paper | Encoder | Trunk / recurrent core | Actor head | Critic / Q | World model | Auxiliary net |
|---|---|---|---|---|---|---|
| BC-Z (Jang 2022) | ✔ ResNet-18, 4 blocks | — | (heads read the modulated torso) | n/a (behaviour cloning) | — | — |
| SAC-RND (Nikulin 2023) | — | — | — | indirect: bonus enters the TD target | — | ✔ **the RND prior and predictor** |
| Moon 2023 | — | — | — | — | — | ✔ contrastive achievement head |
| IQN (Dabney 2018) | — | — | — | ✔ **the Q-network itself** | — | — |
| Vecoven 2020 | ✔ | ✔ every hidden neuron | ✔ | ✔ **separate, independent modulator** | — | — |
| Ben-Iwhiwhu 2022 | ✔ | ✔ every FC layer | ✔ | ✘ **tried, found unstable, dropped** | — | — |
| Beck 2023 | — | — | ✔ policy MLP (FiLM or full hypernet) | — | — | — |
| Schöpf 2022 (HN-PPO) | — | — | ✔ generated | ✔ generated in one variant; **reset instead in the other** | — | — |
| Rezaei-Shoshtari 2023 | — | — | ✔ generated | ✔ generated | — | — |
| Sarafian 2021 | — | — | — | ✔ **$Q(s,a)$ re-parameterised by $s$** | — | — |
| Sherstan 2020 (Γ-nets) | — | — | — | ✔ value head input fusion | — | — |
| CAT-SAC (Lin 2020) | — | — | ✔ via $\alpha_\delta(s)$ in the actor loss | ✔ via $\alpha_\delta(s')$ in the TD target | — | — |
| Dreamer family | — | — | — | — | ✘ **no modulation; concatenation / tokens only** | — |
| **This project** | ✔ unimodal encoder $(\gamma,\beta)$; multimodal hub $(\gamma,\beta)$ | ✔ GRU update-gate bias | ✔ logit temperature | indirect — the critic head reads the modulated trunk but has no $(\gamma,\beta)$ of its own | n/a (rPPO) | — |

### 4.2 The critic question, verified

The brief asked for verification of two specific claims. Both hold, and there are
four further data points.

**Ben-Iwhiwhu et al. 2022 — critic modulation was unstable. VERIFIED.** The
per-paper review's Phase 2 states it directly: *"The neuromodulator only enters
$\pi_\theta$ (and not $Q_\theta$) because PEARL's critic is shared across tasks
and the authors found modulating both unstable."* Note the mechanism they give:
the instability is attributed to the critic being **shared across tasks** while
the modulator is task-specific — a *task-interference* argument, not a
gain-blow-up argument. That is a narrower claim than "modulating critics is
dangerous", and it does not obviously transfer to a single-task agent like ours.
No quantitative table accompanies the claim; it is a prose report of a negative
result.

**Vecoven et al. 2020 — separate actor and critic modulators. VERIFIED.** *"Both
actor and critic are modelled as NMNs (one each, no parameter sharing — the
authors argue the modulatory signals for policy vs. value may differ)."* The
review flags the justification as *"intuitive but unproven"*. So the corpus's two
meta-RL modulation papers land on opposite designs — one modulates the critic
with a dedicated modulator, the other refuses to modulate it at all — and neither
runs the controlled comparison.

**Four further critic data points, all previously unrecorded in our syntheses:**

- **IQN (Dabney et al. 2018) modulates the Q-network successfully, at Atari-57
  scale.** The Hadamard fusion $\psi(s) \odot \phi(\tau)$ is FiLM with the shift
  term set to zero and a cosine-basis generator. This is the strongest existing
  counterexample to "value-head modulation does not work" — a value network
  modulated at every state, every step, with a state-independent conditioner.
- **Sarafian et al. 2021 re-parameterises the critic entirely** and reports the
  *point* of doing so is that the critic's action-gradient $\nabla_a Q$ becomes
  better shaped. 10–70 % gains on MuJoCo under TD3/SAC. A FiLM-style gain term
  $(1+g^l(z))$ is retained on top of the generated weights and is documented as
  **necessary for stability**.
- **Schöpf et al. 2022 ran the actual ablation.** HN-PPO generates both actor and
  critic from the task embedding; HN-PPO+fc generates only the actor and
  *re-initialises* the critic per task. The existence of the second variant is the
  corpus's clearest signal that practitioners treat critic generation as the
  risky half.
- **Rezaei-Shoshtari et al. 2023 generates both** actor and critic weights and
  reports no critic-specific pathology — but it trains by supervised regression
  onto a converged solver's outputs, so its critic never bootstraps. That
  difference is very likely why it escapes the instability others report.

**Verdict on Q2.** Modulating a **value estimator is common and works when the
conditioner is exogenous and fixed-meaning** (IQN's $\tau$, Sarafian's $s$,
HyperZero's MDP descriptor). It is reported unstable in exactly one setting —
**a shared critic under a task-varying modulator with bootstrapped targets**
(Ben-Iwhiwhu / PEARL). Our design sits in neither camp: our critic head is not
directly modulated, but it consumes a trunk whose gains move every step, which
means our value targets are non-stationary in a way none of these papers
studied.

---

## 5. Q3 — What is the modulation *for*?

| Purpose | Papers | Mechanism | Result |
|---|---|---|---|
| **Task / instruction conditioning** (multi-task, goal-conditioned, zero-shot) | BC-Z; Beck 2023; Schöpf 2022; Rezaei-Shoshtari 2023; UVFA/USFA | FiLM on the vision torso; generated policy weights; late concatenation | BC-Z 44 % zero-shot on 24 unseen tasks; HyperZero beats concatenation out-of-distribution |
| **Meta-RL adaptation** (infer the task, reshape the policy) | Vecoven 2020; Ben-Iwhiwhu 2022; Beck 2023; Sarafian 2021 (Hyper-MAML / Hyper-PEARL) | per-neuron slope/bias; per-layer multiplicative gating; weight generation | Ben-Iwhiwhu wins on hard benchmarks (Meta-World ML45, CT-graph depth-4), **ties on easy ones**; Hyper-MAML beats Context-MAML ~50 % |
| **Exploration control** | CAT-SAC; Meta-SAC; mellowmax; Doya's $\beta$=noradrenaline | state-dependent or meta-learned entropy temperature | CAT-SAC is the closest published precedent for "temperature conditioned on a modulator signal" |
| **Anti-exploration / OOD suppression** (offline RL) | SAC-RND (Nikulin 2023) | FiLM inside the RND prior so the bonus has followable gradients | matches ensemble SOTA on D4RL with no ensemble |
| **Continual learning / forgetting resistance** | Schöpf 2022 (HN-PPO + L2 output anchor); Xing 2022; Zou 2020; Rusu 2016 (progressive nets, lateral adapters) | task-conditioned weight generation; frozen backbone + modulator layer | HN-PPO retains earlier DoorGym tasks where PPO fine-tuning collapses |
| **Representation gating** (auxiliary objectives) | Moon 2023 | FiLM fuses action into the state embedding inside a contrastive head | SOTA on Crafter, 9 M params vs. DreamerV3's 201 M |
| **Risk / horizon control** | IQN; Γ-nets; Fedus 2019; Romoff 2019 | multiplicative $\tau$- or $\gamma$-conditioning of a value head | IQN halves the QR-DQN→Rainbow gap; Γ-nets find fusion choice barely matters |
| **Hyperparameter control** (learning rate, discount, temperature) | Doya 2002; Lee 2024; Xu 2018; Kearney 2018 (TIDBD); Wang & Ni 2020 | scalars, always | granularity is forced by the target, not chosen (see [neuromod synthesis §7](../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md)) |

**Verdict on Q3.** The dominant purpose is **task conditioning** — telling an
agent *which* of many jobs it is doing. The second is **meta-RL adaptation** —
letting a policy reshape itself once it infers the job. Both presuppose a task
distribution. **Our agent has one task.** The purposes that do not presuppose a
task distribution are exploration control (CAT-SAC, mellowmax), risk/horizon
control (IQN, Γ-nets), and representation gating (Moon) — and those are the three
places our design has the strongest literature support. This is worth stating
plainly, because it means most of the FiLM-in-RL evidence base is answering a
different question than ours.

---

## 6. Q4 — Granularity and placement, RL-specifically

### 6.1 Granularity

| Paper | Cell | Detail |
|---|---|---|
| BC-Z | **(b)** | per-channel × 4 ResNet blocks; $2C_k \times 513$ parameters per block |
| SAC-RND | **(a) ≡ (b)** | per hidden unit, width 256, dense MLP; one linear layer of double width split into $\gamma$/$\beta$ halves |
| Moon 2023 | **(b)** | per-channel of a CNN state embedding |
| IQN | **(a)** | per-unit gain, **no shift** ($\beta \equiv 0$), from a 64-term cosine basis |
| Vecoven 2020 | **(a)** carried by an **(e)** generator | a global $z \in \mathbb R^k$ decoded by per-neuron learned weights $(w_s,w_b)$ — a rank-$k$ factorisation of a per-neuron $(\gamma,\beta)$ |
| Ben-Iwhiwhu 2022 | **(a)** | element-wise, $\tanh$-bounded so gains can flip sign; a sign-only variant for discrete control |
| Γ-nets Hadamard variant | **(a)** | per-unit scale on the value head's features |
| Beck 2023 FiLM baseline | **(a)** | $(\gamma,\beta)$ on the policy MLP |
| CAT-SAC / Meta-SAC / mellowmax / Doya / Lee | **(e)** | temperature and learning rate are scalars by construction |
| Schöpf / Rezaei-Shoshtari / Sarafian | finer than (a) | full weight generation |

**Verdict.** RL follows the **vision** convention on granularity — per-unit /
per-channel wherever *activations* are the target — and the **neuro** convention
wherever a *scalar algorithmic quantity* is the target. There is no genuine
disagreement here; the two literatures are modulating different objects. Notably:
**no RL paper in the corpus uses grouped (cell c) modulation**, consistent with
the granularity synthesis's finding that AlKilany & Goodman 2025 is the sole
precedent library-wide. And **no paper anywhere emits a per-unit temperature** —
there is one action distribution, so there is one temperature.

### 6.2 Placement

| Paper | Sites | Where | Swept? |
|---|---|---|---|
| BC-Z | 4 | each ResNet block, throughout depth | no — follows Perez's prescription |
| SAC-RND | 1 (default) | penultimate layer of the prior | **yes** — `film_first`/`film_last`/`film_full`. **All layers wins for the prior; last layer only wins for the predictor.** Authors flag it as domain-dependent |
| IQN | 1 | after the conv encoder, before the value MLP — *late* | **yes**, indirectly: Hadamard vs. concatenation vs. residual $(1+\phi(\tau))$, six Atari games. Hadamard robust and slightly best |
| Γ-nets | 1 | at the value head's input — *late* | **yes** — direct concat / linear-embed / Hadamard-FiLM / matrix. **No clear winner; the paper recommends plain concatenation** |
| USFA | 1 | after the LSTM — *deliberately late*, so one LSTM unroll serves $n_z{=}30$ sampled policies | reasoned, not swept |
| Vecoven 2020 | all | every hidden neuron of actor and critic | main-network depth swept (0/1/4 layers); scope not swept |
| Ben-Iwhiwhu 2022 | all | every FC layer of the policy, pre-activation | actor-vs-actor+critic tried; result reported as unstable |
| Moon 2023 | 1 | inside the auxiliary head | no |
| **This project** | **5** | unimodal encoder $\gamma$, unimodal $\beta$, hub $\gamma$, hub $\beta$, GRU gate bias (+ temperature) | **never swept** |

**Verdict on placement — and a correction to the FiLM half.**
[`film_modulation_granularity_synthesis.md` §4.1](film_modulation_granularity_synthesis.md)
concludes the convention is *"many sites, spread through the depth, including
early ones."* **That conclusion does not survive restriction to RL.** Its two
supporting sweeps are De Vries's conv-vision stage sweep and Nikulin's *prior*
network. Within RL specifically:

- The RL papers that condition a **value or policy output** overwhelmingly
  inject **once, late** — IQN after the encoder, Γ-nets at the value head, USFA
  after the LSTM, and USFA gives an explicit efficiency reason for it.
- The RL papers that inject **everywhere** are the two neuromodulation-lineage
  meta-RL papers (Vecoven, Ben-Iwhiwhu), and neither ablates depth.
- Nikulin's own result is *split by function*: all layers for the prior, one late
  layer for the predictor. He explicitly declines to generalise.

So the honest RL statement is: **injection depth is unswept in every RL paper
except Nikulin's, whose answer is "it depends what the modulated network is
for."** Our five unswept injection points therefore sit on the axis with the
*least* RL evidence, not the most.

### 6.3 Does the answer differ by algorithm family?

| Family | Convention | Evidence |
|---|---|---|
| **On-policy (PPO/A2C)** | modulation is rare and, where present, acts on the shared trunk or on generated weights | Vecoven (A2C+GAE+PPO clipping), Schöpf (HN-PPO), Moon (PPO + auxiliary head). **No on-policy paper in the corpus FiLM-conditions a policy head directly** |
| **Off-policy value-based (DQN family)** | multiplicative conditioning of the Q-network is **standard practice** | IQN's Hadamard fusion is a default in distributional RL; Γ-nets test the same operator |
| **Off-policy actor–critic (SAC/TD3)** | conditioning appears in *auxiliary* machinery (RND prior) and in *critic re-parameterisation*, rarely in the actor | Nikulin, Sarafian, CAT-SAC (temperature only) |
| **Model-based / world models (Dreamer family)** | **no feature-wise modulation at all.** Conditioning is concatenation into the recurrent state-space model; Dreamer 4 uses token insertion into a transformer | [Dreamer review](../Dreamer/dreamer_lit_review.md) — a full-text pass returns no FiLM, no adaptive normalisation |

The model-based finding is worth stating loudly because our project has a live
Dreamer thread: **there is no precedent in the Dreamer corpus for FiLM-modulating
a world model.** The nearest thing the Dreamer review itself proposes is
token-insertion conditioning in Dreamer 4, which is the attention primitive, not
the affine one.

---

## 7. Q5 — Reported failure modes and instabilities

Six distinct failure modes appear. None is a clean match for our observation
(per-neuron run led early, then collapsed at ~34 M episodes with the temperature
head pinned at its ceiling and gain variance growing), but three are adjacent.

| # | Failure mode | Where reported | Mechanism given | Fix |
|---|---|---|---|---|
| 1 | **Critic modulation unstable** | Ben-Iwhiwhu 2022 (PEARL) | critic is shared across tasks; a task-varying modulator fights the shared bootstrapped target | modulate the actor only |
| 2 | **Initialisation pathology in generated parameters** | Beck 2023 | Kaiming-initialised generators produce exploding/vanishing base-network activations — *the base policy is broken before the first gradient step*. Cheetah-Dir return 2100 → 378 | **Bias-HyperInit**: zero the generator's output weights, put the base init in the bias. Also rescues the FiLM baseline (5.5 % → 25.5 % on Pick-Place) |
| 3 | **Conditioning operator produces unfollowable gradients** | Nikulin 2023 | concatenation gives the actor a bumpy bonus landscape with local minima; the actor cannot descend it even though the bonus is discriminative | replace concat with FiLM (or bilinear) |
| 4 | **Generated weights need their own learning rate** | Sarafian 2021 | hypernet training dynamics differ from the base algorithm's | re-sweep the LR; the FiLM baseline's LR is not portable |
| 5 | **Hypernet gain term required for stability** | Sarafian 2021 (citing Littwin & Wolf 2019) | plain generated weights are unstable without the extra $(1+g^l(z))$ FiLM gain | keep a FiLM gain on top of generated weights |
| 6 | **Entropy collapse into a near-absorbing state** | Cui et al. 2025 ([continual_learning](../continual_learning/reviews/cui_2025_entropy_mechanism.md)) | policy entropy falls monotonically because $-\mathrm dH \propto \mathrm{Cov}(\log\pi, \pi A)$ stays positive; a small set of high-covariance actions drives it. Performance is pinned to entropy by $R = -a e^{H} + b$, so the ceiling is fixed | targeted (not global) restraint: Clip-Cov / KL-Cov. A blunt entropy bonus **fails** |

**Explicitly absent: gain blow-up.** No paper in either the FiLM or the RL
corpora reports a *modulation-capacity-induced* instability — a modulator that
worked and then destabilised because its gains grew. The granularity synthesis
already flagged this ([§8 item 2](film_modulation_granularity_synthesis.md)); the
RL-restricted search confirms it. Every reported modulation failure is about
**initialisation** (Beck), **operator choice** (Nikulin), **what is modulated**
(Ben-Iwhiwhu), or **optimiser settings** (Sarafian).

**The closest analogues to our crash, ranked.**

1. **Cui et al. 2025's entropy collapse is the best structural match for the
   temperature symptom.** Our temperature head railing at its ceiling is, in
   entropy terms, the modulator demanding maximum stochasticity — the mirror
   image of Cui's collapse, but with the same signature: a monotone one-way drift
   into a state training cannot leave. Cui's diagnosis (a covariance between
   action probability and advantage, dominated by a small set of actions) is
   directly measurable and nobody has measured it for us. Cui's negative result
   also matters: a *global* entropy bonus does not fix a targeted problem.
2. **The RL plasticity literature explains "led early then crashed" without
   invoking modulation at all.** Abbas et al. 2023 documents ReLU activation
   collapse in continual RL (<1 % of units firing) cascading into gradient
   collapse and frozen weights; Sokar et al. 2023 formalises the *dormant neuron*
   and shows dormancy grows monotonically, persists, and causally damages future
   learning; Lyle et al. 2022/2023/2024 tie the same phenomenon to feature-rank
   and capacity loss. **A modulator that scales activations is an obvious
   accelerant for exactly this cascade** — a shrinking $\gamma$ silences units
   the same way a dying ReLU does — and no paper has looked. This is a testable
   confound for our g=1 collapse that the corpus supports and our diagnosis series
   has not used.
3. **Ben-Iwhiwhu's shared-substrate instability** is the only modulation-specific
   report, but its mechanism (task interference) does not apply to a single-task
   agent.

---

## 8. Q6 — Evidence quality: which claims are ablated?

| Claim | Ablated? | Strength |
|---|---|---|
| FiLM beats concatenation for an RND prior's gradient landscape | **Yes** — Table 1 (conditioning-operator sweep: FiLM 95–100 vs. concat 89.6 vs. gating 67.3 D4RL score) plus a gradient-field visualisation | **Strong.** The corpus's single best FiLM-vs-alternatives ablation in RL |
| FiLM-style Hadamard fusion beats concatenation in a Q-network | **Yes** — IQN ablates Hadamard vs. concat vs. residual over six Atari games | **Moderate.** "Robust and slightly best", not a large margin |
| FiLM-style Hadamard fusion beats concatenation in a *value head* for $\gamma$-conditioning | **Yes** — Γ-nets sweep four fusion operators on Atari | **Negative/neutral: no clear winner; the paper recommends plain concatenation.** The corpus's one head-to-head that does *not* favour FiLM |
| Generating all weights beats generating only $(\gamma,\beta)$ | **Yes** — Beck Table 2, matched initialisation: 42.9 % vs. 25.5 % Meta-World Pick-Place | **Strong**, but confined to meta-RL |
| Per-layer multiplicative gating helps hard meta-RL | **Yes** — parameter-matched wider/deeper baselines fail to close the gap; CKA analysis localises the effect | **Strong on hard tasks; explicit null on easy ones** (2-D navigation, half-cheetah: SPN and NPN tie) |
| A modulated critic is unstable | **No** — prose report of a discarded configuration | **Weak.** Single setting, no numbers |
| Separate actor and critic modulators are correct | **No** — the review calls the justification "intuitive but unproven" | **Weak** |
| Conditioning depth matters | **Yes, once** — Nikulin's `film_first`/`last`/`full` | **Moderate, and direction-dependent** |
| FiLM on a robot policy generalises zero-shot | **Partly** — conditioner type is ablated (one-hot 42 % vs. language 40 % vs. video), the *FiLM mechanism* is not | **Architecture description**, not a mechanism ablation |
| Moon 2023's FiLM fusion contributes to the result | **No** — the FiLM layer is one line of a method whose ablations target the contrastive losses | **Architecture description only** |
| State-dependent temperature helps | **Yes** — CAT-SAC vs. SAC-v2, plus a contraction argument for state-dependent $\alpha(s)$ | **Moderate** |

**Verdict on Q6.** Exactly **four** genuine mechanism ablations of feature-wise
conditioning exist in the RL corpus: Nikulin (operator + depth), IQN (operator),
Γ-nets (operator), Beck (what is generated). Two favour multiplicative
conditioning, one is neutral-to-negative, and one says "generate more than the
affine". Everything else is an architecture description. **The evidence base for
"FiLM helps in RL" is much thinner than the frequency of FiLM in RL papers
suggests.**

---

## 9. How our implementation compares

### 9.1 Our design in one paragraph

A shared 16-unit GRU (`NeuromodulatorRNN`) reads the **raw observation vector**
$x$ and emits six heads: gain and offset for the unimodal sensory-encoder stage,
gain and offset for the multimodal-hub stage, a bias added into the task GRU's
update gate, and a scalar action temperature that divides the policy logits. Each
$\gamma/\beta$/memory head emits $\lceil 128/g \rceil$ values expanded by
`jnp.repeat(raw, g)[:128]` (cell **c**, grouped) and added to a full 128-dim
per-neuron learned baseline. In `recurrent_ppo_network.py` the actor and critic
heads read the modulated trunk `x_h`; **neither head carries its own
$(\gamma,\beta)$.**

### 9.2 Five comparisons

- **Our conditioner is unprecedented in RL, and the one paper that argues the
  point argues against it.** We feed the modulator the same raw observation the
  encoder receives, then modulate that encoder. Every RL paper in the corpus
  gives its modulator something the main pathway lacks — a language string, a
  task code, a risk level, a discount, a novelty score, or (Vecoven) the history
  with the current observation *removed*. Ben-Iwhiwhu is the only self-conditioning
  precedent, and its modulator reads a per-layer pre-activation, not a raw sensor
  vector through a separate recurrent net. **If the modulator has no information
  the policy lacks, a null result is the expected outcome** — and this is cheap to
  test by swapping the modulator's input.

- **Our multi-target head layout is legitimate but our temperature head is
  unprecedented.** One shared generator with a per-site head is the field default
  (see [granularity synthesis §5.1](film_modulation_granularity_synthesis.md)).
  Driving an **action-selection temperature from the same trunk that drives
  perceptual gains** is not. Temperature modulation exists in RL — CAT-SAC's
  $\alpha_\delta(s)$, mellowmax's $\beta(s)$, Meta-SAC's meta-learned $\alpha$,
  Doya's noradrenaline — but always as a stand-alone scalar computed from an
  uncertainty or value statistic, never as one head among several on a shared
  low-dimensional recurrent state. Our six heads share a 16-unit GRU and therefore
  **cannot vary independently**, which is a design claim, not a convention.

- **Our injection depth sits on the axis with the least RL evidence.** Five
  injection points, never ablated. The RL papers that condition an output
  overwhelmingly inject once and late (IQN, Γ-nets, USFA); the ones that inject
  everywhere never sweep it; the only sweep in the corpus (Nikulin) returns
  opposite answers for two networks in the same paper. The "inject at many depths"
  prescription in the FiLM half comes from conv-vision and does not transfer.

- **Our critic is in an untested position.** We do not modulate the critic head,
  which superficially matches Ben-Iwhiwhu's caution — but our critic reads a trunk
  whose gains change every step, so its input distribution is non-stationary in a
  way that neither Ben-Iwhiwhu (actor-only, unmodulated critic input) nor Vecoven
  (dedicated critic modulator) nor IQN (fixed-meaning exogenous conditioner)
  studied. If we want literature cover for value-side modulation, IQN and Sarafian
  are the precedents to cite; if we want cover for *not* modulating it,
  Ben-Iwhiwhu is the only one, and it is weak.

- **Our regime predicts flatness, not a large effect.** Our agent has one task and
  one environment. The RL modulation results that are large — Ben-Iwhiwhu's
  Meta-World and CT-graph wins, Beck's hypernet gains, HyperZero's transfer — all
  come from *many-task* settings, and Ben-Iwhiwhu explicitly **ties** on the easy
  single-behaviour benchmarks. This corroborates the regime argument in
  [`../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md` §8.5](../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md).

### 9.3 Connections to project hypotheses, phases, and the null-result series

- **Phase 2 (FiLM variant characterisation)** is the phase this survey serves.
  Its verdict: the *variant* axis currently being swept (grouping size) is the
  axis with no RL evidence either way, while the axes with actual RL evidence —
  **conditioner content** and **injection depth** — are unswept. See
  [`../../../experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md`](../../../experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md).
- **H1 (perceptual amplification — gain rises on threat-relevant features after
  injury)** presumes the modulator can be feature-selective about the
  observation. But the modulator's only input *is* the observation, so H1 asks
  the modulator to extract from $x$ a signal the encoder reading $x$ could compute
  itself. Vecoven's design principle is precisely a rejection of this
  arrangement. H1 is testable, but the corpus predicts the modulator will be
  redundant rather than amplifying.
- **H3 (exploration suppression via the temperature head)** has the corpus's best
  RL support of any of our hypotheses — CAT-SAC is a direct precedent for a
  state-derived temperature, with a contraction argument showing a
  state-dependent $\alpha(s)$ keeps the soft Bellman operator well-behaved. It
  also supplies a design detail we do not use: CAT-SAC adds a **zero-mean**
  curiosity shift to the target entropy, preserving the baseline in expectation.
  A zero-mean constraint on our temperature head is a cheap guard against the
  ceiling-railing we observed.
- **H2 (memory persistence via the GRU update-gate bias)** remains without any RL
  precedent. Nothing in the RL corpus modulates a recurrent gate bias.
- **Null-result diagnosis series v1–v8**
  ([`docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md`](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)):
  this survey supplies two candidate explanations the series has not exploited.
  **(a) Modulator-input redundancy** — every RL modulator in the literature is
  fed something the policy cannot see; ours is not. **(b) The
  modulation-accelerates-dormancy hypothesis** — the RL plasticity literature
  (Abbas 2023, Sokar 2023, Lyle 2022–2024) says activation collapse is the
  dominant late-training failure in continual RL, and a multiplicative gain head
  is a mechanism that can drive units dormant faster than a plain ReLU would.
  Dormant-neuron fraction is a cheap metric and would separate "the modulator is
  inert" from "the modulator is actively silencing the network".

---

## 10. What the corpus does NOT settle

1. **Whether a modulator conditioned on the raw observation can work at all.**
   No RL paper has tried it; no RL paper has tried and rejected it. Vecoven
   *asserts* the modulator should not see the current observation but runs no
   ablation. This is the single most consequential unanswered question for our
   architecture, and the cheapest to answer.
2. **How many injection sites a recurrent RL policy should have.** Nikulin is the
   only sweep, on a feed-forward auxiliary network, with function-dependent
   answers. Nobody has swept injection depth in a recurrent policy.
3. **Whether modulating a bootstrapped critic is safe.** One negative prose report
   (Ben-Iwhiwhu, in a shared-critic multi-task setting), one positive at scale
   with an exogenous conditioner (IQN), one positive under supervised regression
   with no bootstrapping (HyperZero). The three do not compose into an answer for
   a single-task on-policy agent.
4. **Whether yoking an action temperature to the same trunk as perceptual gains
   creates cross-head interference.** Doya 2002 predicts a full cross-effect
   matrix among modulators precisely because their algebraic roles interact; our
   six heads share a 16-dim state and cannot vary independently. Untested
   anywhere.
5. **Whether multiplicative modulation accelerates plasticity loss.** The RL
   plasticity literature and the RL modulation literature do not cite each other
   anywhere in this library. Nobody has measured dormant-neuron fraction in a
   FiLM-modulated agent.
6. **Whether feature-wise modulation belongs in a world model.** The Dreamer
   corpus contains zero instances. Our Dreamer thread has no precedent to lean on
   and would be doing something new — worth knowing before it is scoped.
7. **Whether the temperature head should be constrained.** CAT-SAC's zero-mean
   target-entropy shift is the only published discipline on a state-dependent
   temperature in this corpus. We have a clip, not a constraint, and we observed
   the clip binding.

---

## 11. Prioritised gap list — canonical FiLM-in-RL papers we do not hold

Absence was verified by searching every `sources/` folder and every review under
`docs/project/references/`. Papers that appear only as *citations inside* our
existing reviews (Miconi, Beaulieu, VariBAD, PEARL) are listed here because we do
not hold the papers themselves. **Do not download anything from this list — the
parent session owns downloading.**

### 11.1 High priority

| # | Citation | ID | Why it matters here |
|---|---|---|---|
| 1 | Dumoulin, V., Perez, E., Schucher, N., Strub, F., de Vries, H., Courville, A., Bengio, Y. (2018). *Feature-wise transformations.* **Distill** | DOI [10.23915/distill.00011](https://doi.org/10.23915/distill.00011) | The canonical survey of the whole FiLM family, written by FiLM's authors, **with an RL section**. A referee will expect it as the umbrella citation for "conditional modulation"; we currently cite only Perez 2018 |
| 2 | Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., Song, S. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* **RSS 2023 / IJRR** | arXiv [2303.04137](https://arxiv.org/abs/2303.04137) | Its CNN variant FiLM-conditions the action-denoising network **on the observation embedding** — the closest published thing to our "observation drives the modulator". Would either supply the missing precedent for our conditioner or sharpen exactly how we differ |
| 3 | Chevalier-Boisvert, M., Bahdanau, D., Lahlou, S., Willems, L., Saharia, C., Nguyen, T. H., Bengio, Y. (2019). *BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning.* **ICLR 2019** | arXiv [1810.08272](https://arxiv.org/abs/1810.08272) | The standard FiLM-conditioned RL baseline architecture in gridworlds — the environment class closest to ours. Establishes what "FiLM in a small gridworld agent" is supposed to look like |
| 4 | Chaplot, D. S., Sathyendra, K. M., Pasumarthi, R. K., Rajagopal, D., Salakhutdinov, R. (2018). *Gated-Attention Architectures for Task-Oriented Language Grounding.* **AAAI 2018** | arXiv [1706.07230](https://arxiv.org/abs/1706.07230) | Gain-only multiplicative modulation of a visual stream by an instruction, **with an explicit ablation against concatenation** in an RL agent. One of the few RL papers that actually tests the modulation operator |
| 5 | Brohan, A., Brown, N., Carbajal, J., et al. (2022). *RT-1: Robotics Transformer for Real-World Control at Scale.* **RSS 2023** | arXiv [2212.06817](https://arxiv.org/abs/2212.06817) | FiLM-conditioned EfficientNet driven by a frozen sentence encoder, at production scale — the successor to BC-Z (which we do hold) and the modern reference point for "FiLM on a policy" |
| 6 | Miconi, T., Rawal, A., Clune, J., Stanley, K. O. (2019). *Backpropamine: Training Self-Modifying Neural Networks with Differentiable Neuromodulated Plasticity.* **ICLR 2019** | arXiv [2002.10585](https://arxiv.org/abs/2002.10585) | Cited by Ben-Iwhiwhu as the *plasticity-gating* alternative to activity-gating, and by three other papers we hold. It is the branch of neuromodulated RL our corpus discusses but does not contain |
| 7 | Beaulieu, S., Frati, L., Miconi, T., Lehman, J., Stanley, K. O., Clune, J., Cheney, N. (2020). *Learning to Continually Learn (ANML).* **ECAI 2020** | arXiv [2002.09571](https://arxiv.org/abs/2002.09571) | A neuromodulatory network that **gates the forward pass** of a learner — the direct ancestor of Ben-Iwhiwhu's design and the canonical modulation-for-continual-learning citation |
| 8 | Badia, A. P., Piot, B., Kapturowski, S., Sprechmann, P., Vitvitskyi, A., Guo, D., Blundell, C. (2020). *Agent57: Outperforming the Atari Human Benchmark.* **ICML 2020** — with Badia et al. (2020), *Never Give Up: Learning Directed Exploration Strategies*, **ICLR 2020** | arXiv [2003.13350](https://arxiv.org/abs/2003.13350); arXiv [2002.06038](https://arxiv.org/abs/2002.06038) | The canonical **hyperparameter-conditioned policy**: one network parameterises a family of policies indexed by an exploration/discount pair, with a bandit meta-controller choosing the index. Our temperature head is a continuous relative of this and we cite nothing for it |
| 9 | Zintgraf, L., Shiarlis, K., Igl, M., Schulze, S., Gal, Y., Hofmann, K., Whiteson, S. (2020). *VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning.* **ICLR 2020** | arXiv [1910.08348](https://arxiv.org/abs/1910.08348) | We quote Beck 2023's FiLM-vs-hypernet numbers, which are all measured *on VariBAD*, without holding the method paper. The belief-latent-as-conditioner design is also the cleanest published alternative to our observation-as-conditioner |
| 10 | Yang, R., Xu, H., Wu, Y., Wang, X. (2020). *Multi-Task Reinforcement Learning with Soft Modularization.* **NeurIPS 2020** | arXiv [2003.13661](https://arxiv.org/abs/2003.13661) | Per-task **routing** modulation of a shared policy — the discrete-composition alternative to FiLM, inside RL. Our corpus holds the vision version (mixture-of-experts, neural module networks) but not the RL version |
| 11 | Sodhani, S., Zhang, A., Pineau, J. (2021). *Multi-Task Reinforcement Learning with Context-based Representations (CARE).* **ICML 2021** | arXiv [2102.06177](https://arxiv.org/abs/2102.06177) | Context-conditioned mixture-of-encoders for multi-task RL; the standard baseline any per-task-modulation claim is measured against |

### 11.2 Medium priority

| # | Citation | ID | Why it matters here |
|---|---|---|---|
| 12 | Prasanna, S., Farid, K., Rajan, R., Biedenkapp, A. (2024). *Dreaming of Many Worlds: Learning Contextual World Models Aids Zero-Shot Generalization.* **RLC 2024** | arXiv [2403.10967](https://arxiv.org/abs/2403.10967) | A **context-conditioned DreamerV3** (cRSSM). The only candidate precedent for conditioning a world model, which §6.3 shows our Dreamer corpus completely lacks |
| 13 | Rakelly, K., Zhou, A., Quillen, D., Finn, C., Levine, S. (2019). *Efficient Off-Policy Meta-RL via Probabilistic Context Variables (PEARL).* **ICML 2019** | arXiv [1903.08254](https://arxiv.org/abs/1903.08254) | The base algorithm in which Ben-Iwhiwhu's critic-modulation instability occurred. Without it we cannot judge how far that negative result generalises |
| 14 | Beck, J., Vuorio, R., Liu, E. Z., Xiong, Z., Zintgraf, L., Finn, C., Whiteson, S. (2025). *A Tutorial on Meta-Reinforcement Learning.* **Foundations and Trends in ML 18(2-3), 224–384** | arXiv [2301.08028](https://arxiv.org/abs/2301.08028) | The survey a referee expects to see cited for the whole "conditioner as task belief" framing that dominates §3 |
| 15 | Perez, C. F., Such, F. P., Karaletsos, T. (2020). *Generalized Hidden Parameter MDPs: Transferable Model-Based RL in a Handful of Trials.* **AAAI 2020** | arXiv [2002.03072](https://arxiv.org/abs/2002.03072) | Latent-variable conditioning of *dynamics*, the model-based counterpart to our conditioning of perception. Names the hidden-parameter formalism our "interoceptive state modulates behaviour" framing implicitly assumes |
| 16 | Benjamins, C., Eimer, T., Schubert, F., Mohan, A., Döhler, S., Biedenkapp, A., Rosenhahn, B., Hutter, F., Lindauer, M. (2023). *Contextualize Me — The Case for Context in Reinforcement Learning.* **TMLR** | arXiv [2202.04500](https://arxiv.org/abs/2202.04500) | The contextual-RL formalism plus the CARL benchmark. Supplies the vocabulary for saying precisely what our modulator's input *is* in MDP terms |
| 17 | Sun, L., Zhang, H., Xu, W., Tomizuka, M. (2022). *PaCo: Parameter-Compositional Multi-Task Reinforcement Learning.* **NeurIPS 2022** | arXiv [2210.11653](https://arxiv.org/abs/2210.11653) | Per-task **parameter composition** — a third rung between FiLM and full hypernetworks, and directly relevant to the "how much capacity should the modulator have" question our grouping screen is asking |
| 18 | Obando-Ceron, J., Sokar, G., Willi, T., Lyle, C., Farebrother, J., Foerster, J., Dziugaite, G. K., Precup, D., Castro, P. S. (2024). *Mixtures of Experts Unlock Parameter Scaling for Deep RL.* **ICML 2024** | arXiv [2402.08609](https://arxiv.org/abs/2402.08609) | The only paper that joins **conditional computation** to **RL plasticity loss** — precisely the junction §7 identifies as unexamined and directly relevant to our late-training collapse |
| 19 | Doshi-Velez, F., Konidaris, G. (2016). *Hidden Parameter Markov Decision Processes: A Semiparametric Regression Approach for Discovering Latent Task Parametrizations.* **IJCAI 2016** | arXiv [1308.3513](https://arxiv.org/abs/1308.3513) | The original HiP-MDP. Foundational citation for §11.2 #15; low reading cost |

### 11.3 Low priority

| # | Citation | ID | Why it matters here |
|---|---|---|---|
| 20 | Lynch, C., Sermanet, P. (2021). *Language Conditioned Imitation Learning over Unstructured Data.* **RSS 2021** | arXiv [2005.07648](https://arxiv.org/abs/2005.07648) | Another FiLM-conditioned language-to-control policy; largely redundant with BC-Z, which we hold |
| 21 | Bahdanau, D., Hill, F., Leike, J., Hughes, E., Hosseini, A., Kohli, P., Grefenstette, E. (2019). *Learning to Understand Goal Specifications by Modelling Reward.* **ICLR 2019** | arXiv [1806.01946](https://arxiv.org/abs/1806.01946) | FiLM-conditioned reward modelling in gridworlds; useful only if we pursue instruction-following |
| 22 | Killian, T., Daulton, S., Konidaris, G., Doshi-Velez, F. (2017). *Robust and Efficient Transfer Learning with Hidden-Parameter MDPs.* **NeurIPS 2017** | arXiv [1706.06544](https://arxiv.org/abs/1706.06544) | The BNN-based HiP-MDP follow-up; only needed if #15/#19 turn out to be load-bearing |
| 23 | Sutton, R. S., Modayil, J., Delp, M., Degris, T., Pilarski, P. M., White, A., Precup, D. (2011). *Horde: A Scalable Real-Time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction.* **AAMAS 2011** | no arXiv; AAMAS 2011 proceedings | The GVF foundation cited throughout our Gamma and TD shards but never held. Needed only if the multi-horizon / GVF thread is revived |

### 11.4 Directions checked and found already covered

- **Goal/task-conditioned value functions** — covered by Schaul et al. 2015 (UVFA)
  and Borsa et al. 2018 (USFA) in `Gamma/sources/`.
- **Successor features** — covered by Barreto et al. 2017/2019 in `TD/sources/`.
- **Hypernetworks in meta-RL and continual RL** — covered by Beck 2023,
  Schöpf 2022, Rezaei-Shoshtari 2023, Sarafian 2021.
- **FiLM in offline RL** — covered by Nikulin et al. 2023 (SAC-RND), which is the
  canonical instance.
- **Neuromodulated meta-RL** — covered by Vecoven 2020, Ben-Iwhiwhu 2022,
  Wang 2024.
- **World models** — the Dreamer lineage is complete (2018–2025); the *gap* is
  conditional world models, filled by #12.

---

## 12. Cross-references and recommended follow-ups

**Composes with:**

- [`film_modulation_granularity_synthesis.md`](film_modulation_granularity_synthesis.md)
  — taxonomy source; this survey's §6.2 **narrows** its "inject at many depths"
  verdict to conv-vision, since the RL papers inject once and late.
- [`../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md`](../neuromodulatory_algorithms/neuromod_modulation_scope_synthesis.md)
  — independently reaches the same conclusion about our modulator's input
  (its §8.2, §10 item 6); this survey confirms it from the RL side and adds the
  self-conditioning axis.
- [`film_synthesis.md`](film_synthesis.md) — thematic clusters A–G; Cluster G is
  the applications cluster this survey re-cuts by RL role.
- [`film_lit_review.md`](film_lit_review.md) — the 23-paper FiLM index.

**Per-paper sources used:**
[BC-Z](reviews/jang_2022_bcz.md) ·
[SAC-RND](reviews/nikulin_2023_anti_exploration_rnd.md) ·
[Moon 2023](reviews/moon_2023_hierarchical_achievements.md) ·
[Vecoven 2020](../neuromodulatory_algorithms/reviews/vecoven_2020_neuromod_dnn.md) ·
[Ben-Iwhiwhu 2022](../neuromodulatory_algorithms/reviews/beniwhiwhu_2022_context_meta_rl.md) ·
[Hypernetwork corpus §5–7](../Hypernetwork/hypernetwork_lit_review.md) ·
[TD shard A (IQN)](../TD/TD_lit_review_A_distributional_rl.md) ·
[TD shard C (Sarafian, HyperZero)](../TD/TD_lit_review_C_hypernet_rl.md) ·
[TD shard D (Decision Transformer)](../TD/TD_lit_review_D_decision_transformer.md) ·
[Gamma shard A (UVFA/USFA)](../Gamma/Gamma_lit_review_A_universal_vf.md) ·
[Gamma shard B (Γ-nets)](../Gamma/Gamma_lit_review_B_multi_horizon.md) ·
[Temperature shard B (CAT-SAC, Meta-SAC)](../Temperature/Temperature_lit_review_B_sac_variants.md) ·
[Temperature shard C (mellowmax)](../Temperature/Temperature_lit_review_C_softmax_operators.md) ·
[Learning_Rate shard C (meta-gradient RL)](../Learning_Rate/Learning_Rate_lit_review_C_meta_gradient_rl.md) ·
[Learning_Rate shard D (gating primitives)](../Learning_Rate/Learning_Rate_lit_review_D_gated_conditional_arch.md) ·
[continual_learning reviews](../continual_learning/continual_learning_lit_review.md) ·
[Dreamer review](../Dreamer/dreamer_lit_review.md)

**Project docs:**
[NEUROMODULATION_ALGORITHM.md](../../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) (H1–H5, injection points) ·
[NMN_FILM_GROUPING_SCREEN.md](../../../experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md) ·
[NMN_PERFORMANCE_DIAGNOSIS_v8.md](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)

### Recommended follow-ups

- **`experiment-designer` — a modulator-input ablation.** §3.2 and §9.2 identify
  this as the one design choice the entire RL modulation literature agrees on and
  we depart from. Three arms: raw observation (current), observation history minus
  the current step (Vecoven's rule), and the task GRU's own hidden state
  (AlKilany's rule). Cheaper than the grouping screen and better motivated.
- **`experiment-designer` — an injection-site ablation** (unimodal only / hub only
  / memory only / all), since §6.2 shows five unswept sites on an axis where the
  only RL sweep returns function-dependent answers.
- **`experiment-analyzer` — measure dormant-neuron fraction and the entropy
  covariance on the g=1 collapse run.** §7 argues the late-training crash has two
  literature-supported explanations nobody has tested: activation-collapse
  acceleration (Sokar's dormant-neuron score) and covariance-driven entropy drift
  (Cui's identity). Both are computable from existing checkpoints.
- **`literature-reviewer` — pull items 1–11 of §11.1** once the parent session has
  downloaded them. Items 1–5 are the FiLM-in-RL citation backbone; 6–8 are the
  modulation-in-RL backbone.
- **`plan-reviewer`** — before any positive grouping-screen result is believed,
  §9.2's regime argument (all large modulation gains in RL come from many-task
  settings; the one paper that tests an easy single-behaviour benchmark ties)
  should be weighed against the screen's pre-registration.
