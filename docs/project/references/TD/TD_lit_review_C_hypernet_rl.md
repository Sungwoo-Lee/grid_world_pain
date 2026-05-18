# TD Corpus — Shard C: Hypernetwork-Based Reinforcement Learning

**Reviewer:** literature-reviewer
**Date:** 2026-05-19
**Shard scope:** the two RL-specific hypernetwork papers in the TD reference corpus. The rest of the TD corpus (distributional RL, successor features, Decision Transformer) is reviewed in sibling shards.

---

## Purpose (entry point — read this first)

This review covers **two papers that use one neural network to generate the weights of another network in the reinforcement-learning loop**. To understand why this matters for our project, picture a *ladder of conditioning richness* for how a "modulator" signal can change an agent's behaviour:

1. **FiLM (Feature-wise Linear Modulation)** — the modulator outputs a per-channel *gain* and *bias* that affine-shifts the agent's hidden activations. The agent's weights stay fixed; only the activations get re-scaled. Cheapest rung.
2. **Row-scaling hypernetwork** — the modulator outputs one scalar per *row* of a weight matrix, multiplicatively scaling that row. Strictly more expressive than FiLM, strictly less than a full hypernetwork. Middle rung.
3. **Full hypernetwork** — the modulator (here called the *primary network*) outputs the *entire weight matrix* (and bias) of one or more layers of the agent (the *dynamic network*). Top rung. The agent is, in effect, re-parameterised on every step / task by the modulator.

These two papers sit at rung 3. Together they answer two distinct questions:

- **Sarafian, Keynan, Kraus (ICML 2021) — "Recomposing the RL Building Blocks with Hypernetworks"** asks: in a *single-task* off-policy actor-critic loop, does it help to treat the Q-function as a hypernetwork in which the state generates the weights of a small dynamic network that consumes only the action (their "SA-Hyper" variant)? Their answer: yes, the *action-gradient* $\nabla_a Q^\pi(s,a)$ — which is what the policy needs in TD3/SAC — becomes more accurate, and TD3/SAC training is faster and final scores are 10%–70% better on MuJoCo.
- **Rezaei-Shoshtari et al. (AAAI 2023) — "HyperZero: Hypernetworks for Zero-Shot Transfer in RL"** asks: if we have a *family* of MDPs that differ only in reward / dynamics parameters $(\psi_i, \mu_i)$, can a hypernetwork map $(\psi_i, \mu_i) \mapsto$ *(near-optimal policy weights, near-optimal Q-function weights)* well enough that we can drop the trained hypernetwork onto a brand-new task in the family and execute zero-shot with no fine-tuning? Their answer: yes, beating context-conditioned policies, UVFA, MAML, and PEARL on DeepMind Control Suite at zero-shot.

**Zero-shot transfer in RL** means: the agent is told the task parameters $(\psi, \mu)$ of a never-seen test task and must execute its *first* policy rollout on that task with no exploration / fine-tuning steps allowed. It is the hardest end of the transfer spectrum (vs. few-shot, vs. meta-RL adaptation).

**Why a fresh reader should care for this project.** Our project's modulator-conditioned-agent thread (the FiLM and row-scaling-hypernet ablations under the modulator-grouped-norm and ALSTM architectures) is exactly this ladder. These two papers are the strongest existing precedent for the *top* rung — "let the modulator entirely re-parameterise the agent." Sarafian provides the *gradient-quality* argument for why hypernet-style conditioning can beat plain concatenation even on a single task; Rezaei-Shoshtari provides the *transfer* argument for why mapping context → weights generalises better than mapping (state, context) → action. Both papers are RL-native and use TD3-family losses, which is closer to our PPO / recurrent-PPO setting than the supervised-learning hypernet literature.

**Cross-reference.** The generic hypernetwork foundation (Ha, Dai, Le 2016) and the broader gated/conditional-architecture landscape (FiLM, hypernet, mixture-of-experts) are reviewed in [`../Learning_Rate/Learning_Rate_lit_review_D_gated_conditional_arch.md`](../Learning_Rate/Learning_Rate_lit_review_D_gated_conditional_arch.md). Read that document for the rung-1 (FiLM) and rung-2 baselines this shard sits on top of.

---

## Table of Contents

1. [Purpose (entry point — read this first)](#purpose-entry-point--read-this-first)
2. [Sarafian, Keynan, Kraus (ICML 2021) — Recomposing the RL Building Blocks with Hypernetworks](#1-sarafian-keynan-kraus-icml-2021--recomposing-the-rl-building-blocks-with-hypernetworks)
   - [Phase 1 — Foundational overview](#phase-1--foundational-overview-sarafian)
   - [Phase 2 — Graduate-level deep dive](#phase-2--graduate-level-deep-dive-sarafian)
   - [Appendix A.1 — Section-by-section backbone (Sarafian)](#appendix-a1--section-by-section-backbone-sarafian)
3. [Rezaei-Shoshtari et al. (AAAI 2023) — HyperZero: Hypernetworks for Zero-Shot Transfer in RL](#2-rezaei-shoshtari-et-al-aaai-2023--hyperzero-hypernetworks-for-zero-shot-transfer-in-rl)
   - [Phase 1 — Foundational overview](#phase-1--foundational-overview-hyperzero)
   - [Phase 2 — Graduate-level deep dive](#phase-2--graduate-level-deep-dive-hyperzero)
   - [Appendix A.2 — Section-by-section backbone (HyperZero)](#appendix-a2--section-by-section-backbone-hyperzero)
4. [Cross-paper synthesis and carry-over to this project](#cross-paper-synthesis-and-carry-over-to-this-project)

---

## 1. Sarafian, Keynan, Kraus (ICML 2021) — Recomposing the RL Building Blocks with Hypernetworks

**PDF:** `docs/project/references/TD/sources/Sarafian et al. 2021 - Recomposing the Reinforcement Learning Building Blocks with Hypernetworks.pdf`
**Venue:** ICML 2021 (PMLR 139).
**Code:** https://github.com/keynans/HypeRL
**Tags:** off-policy actor-critic, TD3, SAC, MAML, PEARL, MuJoCo, gradient quality.

### Phase 1 — Foundational overview (Sarafian)

**What problem are they solving?** In off-policy actor-critic algorithms like TD3 and SAC, the actor is updated by following the *parametric gradient* of the critic with respect to the action: $\nabla_a Q^\pi_\theta(s,a)$. The critic itself is a neural network — typically a 2-layer MLP that takes $\mathrm{concat}(s, a)$ as input. The actor's improvement therefore depends entirely on how *well-shaped* that critic's $\nabla_a$ is — not on the critic's TD error alone. Ilyas et al. (2019) already showed that this gradient has near-zero cosine similarity to the "true" gradient in standard implementations, which is shocking given how well RL works in practice.

**Their proposal.** Stop concatenating $s$ and $a$. Instead, treat the Q-function as a *hypernetwork*:

- A large **primary network** $w_\theta(\cdot)$ takes one of the two inputs (say $s$) and outputs the weights of a small **dynamic network** $f_{w_\theta(s)}$.
- The dynamic network takes the *other* input ($a$) and outputs the Q-value.
- The action gradient $\nabla_a Q$ now flows only through the small dynamic network, whose weights $w(s)$ are explicitly state-dependent.

They show this configuration — calling $s$ the "meta-variable" and $a$ the "base variable" — is called **SA-Hyper** (State→Action hypernet). Reversing the roles (action as meta, state as base) is **AS-Hyper**. SA-Hyper wins.

**What does each paper add beyond Ha et al. 2016 (generic HyperNetworks)?** Sarafian's contribution is *not* a new architectural primitive — they reuse Ha-style hypernets — but a **gradient-quality argument** for *which input should be the meta-variable in an RL setting*. They prove (Proposition 1) that monotonic policy improvement is guaranteed when the parametric gradient is close enough to the true gradient (in cosine), and they empirically demonstrate that SA-Hyper gives the cleanest gradient. The paper is a "use hypernets like *this*, not like *that*, for actor-critic" paper.

**Empirical regime tested.**
- **Single-task RL:** TD3 and SAC on Hopper-v2, Walker2D-v2, Ant-v2, HalfCheetah-v2 (4 MuJoCo continuous control environments).
- **Meta-RL:** MAML and PEARL on Half-Cheetah-Fwd-Back, Ant-Fwd-Back, Half-Cheetah-Vel, Ant-Vel, and Half-Cheetah-Vel-Medium (the out-of-distribution variant from Fakoor et al. 2019).
- All averaged over 5 seeds. Hypernet always uses the same baseline algorithm hyperparameters except learning rate.

**Headline result.** On single-task RL the hypernet critic gives 10%–70% gain over MLP-Standard at convergence and reaches the baseline score with 20%–70% of the training steps. On Meta-RL, Hyper-MAML beats Context-MAML by ~50% and — remarkably — **does not require the MAML adaptation step at all** (pre- and post-adaptation scores are indistinguishable). Hyper-PEARL beats PEARL by 15% in-distribution and generalises significantly better out-of-distribution on Half-Cheetah-Vel-Medium.

**Initial takeaway.** The action-gradient quality is the binding constraint on off-policy actor-critic, and putting the state in the "meta" slot (so the state-dependent weights determine the action-gradient configuration directly via $W_\ell(s)$, not just via ReLU active-set masks $\Lambda_\ell$) gives strictly more expressive gradient maps. For Meta-RL, putting the task context in the meta slot disentangles task-gradients from state-gradients and reduces learning-step variance.

### Phase 2 — Graduate-level deep dive (Sarafian)

#### 2.1 The hypernetwork model and which blocks are generated

The hypernetwork architecture (Figure 1 of the paper) processes a tuple $(z, x) \in \mathcal{Z} \times \mathcal{X}$ and outputs $y \in \mathcal{Y}$. It consists of:

- a **primary network** $w_\theta: \mathcal{Z} \to \mathbb{R}^{n_w}$ producing weights $w_\theta(z)$ for a
- **dynamic network** $f_{w_\theta(z)}: \mathcal{X} \to \mathcal{Y}$.

In the Sarafian implementation:

- The primary network $w_\theta(z)$ is a **deep ResNet stack** that maps $z$ through residual blocks into a $1024$-dim latent, then a series of parallel linear "heads" each emitting one set of dynamic weights. The ResNet contains Block-256, Block-512, and Block-1024 residual blocks plus head layers.
- The dynamic network $f_{w_\theta(z)}$ has only **one hidden layer of 256 units** — smaller than the 2-hidden-layer MLP-Standard baseline. This is deliberate, to isolate the gain from dynamic weights vs. extra parameters.

Each dynamic-network layer is implemented in a FiLM-augmented form:

$$
x^{l+1} = \sigma_{\text{ReLU}}\!\big(\,(\,1 + g^l(z)\,) \odot x^l W^l(z) + b^l(z)\,\big) \qquad (\text{Sarafian Eq. 1})
$$

where $W^l(z), b^l(z)$ are the generated weight and bias, and $g^l(z)$ is an *additional gain parameter* (FiLM-style) found necessary for hypernetwork stability following Littwin & Wolf (2019). Note that the non-linearity is applied only over the hidden layer; the output is linear. **This means the dynamic net is conceptually one hidden layer of full hypernet-generated weights *with* an extra FiLM-like gain — i.e. it's the top rung of the ladder, but with a FiLM "boost" included.** This is an architectural detail that is directly load-bearing for our project's design space.

**Which blocks are hypernet-generated vs. fixed.** Only the dynamic network's weights $W^l, b^l, g^l$ are state-conditioned. The primary network's parameters $\theta$ are the only *learnable* parameters that get updated by the optimizer; at inference time, $\theta$ is fixed and $z$ is what produces $w_\theta(z)$. There is no separate "trunk" — the state's only path into the Q-value is through generating the dynamic weights.

#### 2.2 The three configurations and the parameter-generation map in RL

Sarafian compares three Q-function architectures (Figure 2 of the paper):

| Name | Input to primary $w$ (meta $z$) | Input to dynamic $f$ (base $x$) | Output |
|---|---|---|---|
| MLP | — | concat($s$, $a$) | $Q^\pi(s,a)$ |
| AS-Hyper | $a$ | $s$ (features) | $Q^\pi(s,a) = f_{w_\theta(a)}(\xi(s))$ |
| SA-Hyper | $s$ | $a$ | $Q^\pi(s,a) = f_{w_\theta(s)}(a)$ |

The parameter-generation map in RL context is therefore literally

$$
\theta_{\text{main}} \;=\; w_\theta(z), \qquad z \in \{s, a\}
$$

with the "main" / "dynamic" net being the part that takes the *other* RL variable as input. The choice of which variable is meta-vs-base is the crux of Section 3.

#### 2.3 The gradient-quality argument (Proposition 1)

Sarafian's central theoretical claim is:

**Proposition 1 (paraphrased).** Let $\pi(a|s) = \mu_\phi(\varepsilon|s)$ be a stochastic parametric policy with $\varepsilon \sim p_\varepsilon$, $\mu_\phi(\cdot|s)$ Lipschitz-gradient with constant $\kappa_\mu$, and $Q^\pi(s,a)$ Lipschitz-gradient in $a$ with constant $\kappa_q$. Define the empirical advantage operator $\nabla_\phi \cdot f \;=\; \mathbb{E}_{s\sim\mathcal{D}}\mathbb{E}_{\varepsilon\sim p_\varepsilon}[\,\nabla_\phi \mu_\phi(\varepsilon|s)\cdot f(s, \mu_\phi(\varepsilon|s))\,]$. If there exists a gradient estimator $g(s,a)$ and a constant $0 < \alpha < 1$ such that

$$
\| \nabla_\phi \cdot g - \nabla_\phi \cdot \nabla_a Q^\pi \| \;\le\; \alpha \, \| \nabla_\phi \cdot \nabla_a Q^\pi \| \qquad (\text{Sarafian Eq. 6})
$$

then the ascent step $\phi' \leftarrow \phi + \eta\,\nabla_\phi\cdot g$ with $\eta \le \tfrac{1}{\tilde{k}}\,\tfrac{1-\alpha}{(1+\alpha)^2}$ yields a *positive-empirical-advantage* policy, where $\tilde k$ aggregates the Lipschitz constants (full proof in appendix).

**Reading.** Better critic gradient $\Rightarrow$ smaller $\alpha$ $\Rightarrow$ larger learning rate $\eta$ is allowed $\Rightarrow$ guaranteed monotonic policy improvement. This converts "more accurate $\nabla_a Q$" into "you can take bigger ascent steps." Critically, this proposition is about the **gradient**, not the **value MSE** — and Sarafian explicitly notes that lower-MSE Q-models do not necessarily give better $\nabla_a Q$. This is the theoretical justification for switching from value-fit-quality to gradient-fit-quality as the architecture-selection criterion.

#### 2.4 Why SA-Hyper has structurally better $\nabla_a Q$

For a single-layer linear dynamic network the comparison is illuminating.

**Linear / MLP model.**

$$
Q^\pi_\theta(s,a) = [w_s, w_a] \cdot [\xi(s), a], \qquad \nabla_a Q^\pi_\theta(s,a) = w_a \qquad (\text{Eq. 7})
$$

The gradient is a constant — *not a function of the state at all*. Useless for actor-critic.

**AS-Hyper (action is meta).**

$$
Q^\pi_\theta(s,a) = w_\theta(a)\cdot \xi(s), \qquad \nabla_a Q^\pi_\theta(s,a) = \nabla_a w_\theta(a)\,\xi(s) \qquad (\text{Eq. 8})
$$

The gradient now depends on $s$, but $\nabla_a w_\theta(a)$ is an $n_a \times n_w$ matrix with a large null-space — many state directions $\xi(s)$ produce near-zero gradients. The pathology is exacerbated in the multi-layer case (Sarafian §3.2).

**SA-Hyper (state is meta).**

$$
Q^\pi_\theta(s,a) = w_\theta(s)\cdot a, \qquad \nabla_a Q^\pi_\theta(s,a) = w_\theta(s) \qquad (\text{Eq. 9})
$$

This is a *state-conditioned constant model of the action-gradient*. Sufficient for localised policies with low action-variance because it captures the local tangent hyperplane around the policy mean.

**Multi-layer comparison.** For the MLP with $L$ layers using ReLU-like activations $\sigma$:

$$
x^{l+1} = f^l(x^l) = \sigma(x^l W^l + b^l), \qquad \nabla_a x^{l+1} = (\nabla_a x^l)\, W^l\,\Lambda^l(x^l)
$$

$$
\Lambda^l(x^l) = \mathrm{diag}\bigl(\sigma'(x^l W^l + b^l)\bigr) \qquad (\text{Eqs. 10–12})
$$

so by the chain rule the MLP action-gradient is

$$
\nabla_a Q^\pi_\theta(s,a) = W^a\,\Lambda^1(s,a)\,\prod_{\ell=2}^{L-1} W^\ell\,\Lambda^\ell(s,a)\, W^L \qquad (\text{Eq. 13})
$$

State-dependence enters *only* through the ReLU-active-set masks $\Lambda^\ell(s,a)$, which are piecewise-constant and bounded ($\le 1$). Sarafian argues these diagonal masks have severely limited expressive power — for two pairs $(s_1, a_1), (s_2, a_2)$ with identical ReLU active sets, the *estimated gradient is identical*.

In contrast, for SA-Hyper:

$$
\nabla_a Q^\pi_\theta(s,a) = W^1(s)\,\Lambda^1(s,a)\,\prod_{\ell=2}^{L-1} W^\ell(s)\,\Lambda^\ell(s,a)\, W^L(s) \qquad (\text{Eq. 14})
$$

Now every weight matrix is state-dependent through the primary net. The gradient configuration is controlled by $W^\ell(s)$, not just by ReLU masks. The claim is that this gives genuinely richer state-conditioned gradient maps.

#### 2.5 Empirical gradient-quality measurement

Sarafian estimates the "true" $\nabla_a Q^\pi$ at $a_\mu = \mathbb{E}_{\varepsilon}[\mu_\phi(\varepsilon|s)]$ by generating $N_r = 15$ independent trajectories with $a = a_\mu + \Delta_a$ and fitting a local linear LMS model. The "true" gradient is the slope of that local linear model. They then measure **cosine similarity** (CS):

$$
\mathrm{cs}(Q^\pi_\theta) = \mathbb{E}_{s\sim\mathcal{D}}\!\left[\,\frac{\nabla_a Q^\pi_\theta(s,a_\mu)\cdot \nabla_a Q^\pi(s,a_\mu)}{\|\nabla_a Q^\pi_\theta(s,a_\mu)\|\,\|\nabla_a Q^\pi(s,a_\mu)\|}\right]
$$

Findings: across all 4 MuJoCo environments and over the training trajectory, SA-Hyper has substantially higher cosine-similarity gradient than MLP or AS-Hyper. The advantage is most pronounced in the *first 100K learning steps*, which is consistent with the *learning-curve* result that SA-Hyper learns faster early.

#### 2.6 Meta-RL: context as meta-variable disentangles task- and state-gradients

The Meta-RL extension is the deeper conceptual point. A meta-policy

$$
\pi_\phi(a|s,c) = \mu_{w(c)}(\varepsilon|s) \qquad \text{with } \varepsilon \sim p_\varepsilon \qquad (\text{Eq. 16})
$$

puts the task context $c$ in the **meta** slot. Plugging this into the MAML objective:

$$
J(\phi) = \sum_{T_i}\sum_{s_j\in T_i} \hat A_{i,j}\,\frac{\nabla_\phi \mu_{\phi_i}(\varepsilon_j|s_j, c_i)}{\mu_{\phi_i}(\varepsilon_j|s_j, c_i)} \qquad (\text{Eq. 17})
$$

becomes, after the hypernetwork substitution,

$$
J(\phi) = \sum_{T_i} \nabla_\phi w(c_i)\,\cdot\,\sum_{s_j\in T_i} \hat A_{i,j}\,\frac{\nabla_w \mu_{w(c_i)}(\varepsilon_j|s_j)}{\mu_{w(c_i)}(\varepsilon_j|s_j)} \qquad (\text{Eq. 18})
$$

**Reading.** The state-dependent dynamic-weight gradients $\nabla_w \mu_{w(c_i)}$ are averaged *independently for each task*, and the task-dependent primary-weight gradients $\nabla_\phi w(c_i)$ are averaged *only over the task distribution*, not over the joint (task, state) distribution as in standard MAML (Eq. 17). The factorisation reduces gradient variance for the same sample count, particularly important for on-policy MAML where the per-step sample count is small. This is the *Meta-RL variance-reduction* argument.

For PEARL (off-policy), Sarafian instead uses the context as the *base* variable (input to the dynamic network), not the meta variable, because (i) generalisation benefited empirically from a constant weight set per state, and (ii) when the context is itself learnable, back-propagating its gradient through three networks (primary + dynamic + context-encoder) hurt training. This is a notable asymmetry — the optimal placement of the context depends on whether the context encoder is itself learnable.

#### 2.7 Training-stability tricks

- **Weight initialization.** Hypernetworks are sensitive to primary-net initialization because bad primary outputs can amplify into catastrophic dynamic weights (citing Chang et al. 2019). Sarafian initialises the primary net so that the *average initial distribution of generated dynamic weights resembles Kaiming-uniform initialization* (He et al. 2015). This is a load-bearing trick.
- **Extra gain $g^l(z)$ (FiLM-style).** Required for stability (Littwin & Wolf 2019). The dynamic-layer form $(1 + g^l(z))\odot x^l W^l(z) + b^l(z)$ is hypernet + FiLM combined.
- **ResNet primary.** A high-performance ResNet primary (Srivastava et al. 2015) was found necessary to generate good dynamic weights.
- **Learning-rate retune.** Hypernet uses the baseline algorithm's loss unchanged; only the architecture and the learning rate are modified.

#### 2.8 Result summary

| Setting | Algorithm | Hypernet gain vs. MLP-Standard |
|---|---|---|
| Single-task RL | TD3 | 10%–70% final-performance gain; reaches baseline with 20%–70% of steps |
| Single-task RL | SAC | similar gain across envs |
| Meta-RL | Hyper-MAML vs. Context-MAML | ~50% gain; **no adaptation step needed** |
| Meta-RL | Hyper-PEARL vs. PEARL | 15% in-distribution gain, 70% fewer steps to baseline, *better OOD* on Vel-Medium |

Ablations against same-parameter-count baselines (MLP-Large with 9M weights matching the hypernet, ResNet35 with 4.5M weights, Q-D2RL deep-dense, ResNet-Features which concats ResNet features with action) confirm that **the gain is not due to parameter count or ResNet expressiveness**: only the structural difference (state-conditioned dynamic weights) explains the gap.

### Appendix A.1 — Section-by-section backbone (Sarafian)

> Raw backbone, in the paper's original section order. Phase 1/2 above is the reorganised synthesis; this appendix preserves traceability to the source.

**Abstract.** Q-functions and meta-policies take inputs from the Cartesian product of two domains (state×action; state×context). Standard architectures concatenate features and feed to an MLP, ignoring the underlying input semantics. They argue this causes (a) poor critic-gradient estimation in actor-critic, and (b) high learning-step variance in Meta-RL. Their proposal: hypernetworks where a primary network determines the weights of a conditional dynamic network. Shown to improve gradient approximation, reduce learning-step variance, accelerate learning, improve final performance across MuJoCo locomotion tasks, in TD3, SAC, MAML, PEARL.

**§1 Introduction.** Motivates state-action reciprocity in Q-functions, points out that off-policy actor-critic uses $\nabla_a Q^\pi(s,a)$ for actor updates (unlike REINFORCE), cites Ilyas et al. (2019) showing near-zero cosine similarity in standard implementations, and motivates hypernetworks as the structurally right model. Contributions: (1) theoretical link between gradient quality and learning rate via Proposition 1; (2) empirical gradient-quality improvement and faster learning; (3) Meta-RL variance reduction, OOD generalisation gain, and elimination of MAML adaptation step.

**§2 Hypernetworks.** Reviews Ha-et-al hypernetworks; tracks history (McClelland 1985, Schmidhuber 1992); RL precedents (QMIX, continual model-based RL). Describes their model (Figure 1): ResNet primary $\to$ 1024-d latent $\to$ parallel linear heads $\to$ dynamic weights; dynamic net is 1-hidden-layer of 256 with FiLM-augmented form $x^{l+1}=\sigma((1+g^l(z))\odot x^l W^l(z) + b^l(z))$ (Eq. 1).

**§3 Recomposing the Actor-Critic's Q-function.**
- §3.1 Background — MDP formalism, $J(\pi)$ as expectation over Q (Eq. 3), stochastic policies as $\pi_\phi(a|s)=\mu_\phi(\varepsilon|s)$ (Eq. 4), actor-critic with critic loss (Eq. 4 in their numbering) and actor update $\nabla_\phi J_{\text{actor}} = \mathbb{E}[\nabla_\phi \mu_\phi(\varepsilon|s)\,\nabla_a Q^\pi_\theta(s,\mu_\phi(\varepsilon|s))]$ (Eq. 5). Distinguishes TD3 (deterministic + noise + double-Q) vs. SAC (stochastic + entropy bonus).
- §3.2 Our approach — Proposition 1 (gradient-CS-to-monotonic-improvement); three architectures (MLP, AS-Hyper, SA-Hyper); single-layer analysis (Eqs. 7, 8, 9); multi-layer analysis (Eqs. 10–14) showing that SA-Hyper has state-conditioned weight matrices $W^\ell(s)$ while MLP only has ReLU-active-set state-dependence; empirical CS measurement via local LMS at policy mean.

**§4 Recomposing the Policy in Meta-RL.**
- §4.1 Background — MAML (Eq. 15), PEARL with learnable probabilistic context, oracle-context option.
- §4.2 Our approach — Eq. 16 ($\pi(a|s,c) = \mu_{w(c)}(\varepsilon|s)$ with context as meta), then Eqs. 17–18 deriving the disentangled state-/task-gradient factorisation.

**§5 Experiments.**
- §5.1 Setup — MuJoCo + OpenAI Gym, 4 RL envs, 5 Meta-RL envs, baseline implementations preserved, learning-rate retuned, 5 seeds.
- §5.2 Architecture — MLP-Standard (2 layers × 256), MLP-Small (1 layer × 256), MLP-Large (2 layers × 2900 ≈ 9M params), ResNet-Features (ResNet trunk concatenated with action), Res35 (35-block ResNet trunk, ≈ 4.5M params), Q-D2RL (Sinha et al. 2020 deep-dense). Stability: Kaiming-uniform-matching primary init.
- §5.3 Results — Hyper consistently best across TD3, SAC, MAML, PEARL.

**§6 Conclusions.** Hypernets improve $\nabla_a Q^\pi$, reduce Meta-RL learning-step variance, improve OOD generalisation, eliminate MAML adaptation step. Code at https://github.com/keynans/HypeRL.

---

## 2. Rezaei-Shoshtari et al. (AAAI 2023) — HyperZero: Hypernetworks for Zero-Shot Transfer in RL

**PDF:** `docs/project/references/TD/sources/Rezaei-Shoshtari et al. 2023 - Hypernetworks for Zero-Shot Transfer in Reinforcement Learning.pdf`
**Venue:** AAAI 2023.
**Code / data:** https://sites.google.com/view/hyperzero-rl
**Tags:** zero-shot transfer, contextual RL, parameterized MDP family, supervised RL, TD3, DeepMind Control Suite.

### Phase 1 — Foundational overview (HyperZero)

**What problem are they solving?** Take a *family* of MDPs $\mathcal{M} = \{M_i = (\mathcal{S}, \mathcal{A}, T_{\mu_i}, R_{\psi_i}, \gamma)\}$ that all share state and action spaces but differ in reward parameters $\psi_i$ and dynamics parameters $\mu_i$ (e.g., desired walking speed; torso length of a cheetah body). For some "training" tasks $i$ we can afford to run a full RL solver (TD3) and get a near-optimal policy $\pi_i^*$ and Q-function $Q_i^*$. We are then handed a brand-new test task $(\psi, \mu)$ — never seen, but the *parameters* are known — and asked to roll out a good policy *on the first timestep*. No fine-tuning, no exploration, no adaptation. This is **zero-shot contextual RL**.

**Their proposal.** View an RL algorithm as a *mapping*: $\text{RL}: M_i = M(\psi_i, \mu_i) \mapsto \pi_i^*, Q_i^*$. Approximate this mapping with a hypernetwork $H_\Theta$ that takes the task parameters and emits the *full weights* of a near-optimal policy network $\hat\pi_\theta$ and Q-network $\hat Q_\varphi$. If the approximation is good, the policy can be rolled out immediately on a new $(\psi, \mu)$. Train $H_\Theta$ with a supervised regression loss on training-task near-optimal $(s, a^*, q^*)$ samples, plus a TD regularisation loss that enforces Bellman consistency between the generated $\hat\pi$ and $\hat Q$.

**What does this paper add beyond Ha 2016?** Ha-et-al hypernets were demonstrated on supervised learning (recurrent and convnet weight generation). The contribution here is **RL-native**:

1. **A new use case** — hypernet output is a complete policy + Q for a *family of MDPs*, not a per-sample weight.
2. **A novel TD regularisation** (Eq. 5 below) that flips the standard deep-RL TD-loss direction: it moves the *target* toward the *current* value estimate (since they trust the near-optimal $q^*$ as ground truth).
3. **A scalable supervised-learning framing** for zero-shot transfer, with explicit assumptions (Section 3.1) and empirical evidence that the strong assumption (RL solver fully converged) can be relaxed.

**Empirical regime tested.** Zero-shot transfer across reward, dynamics, and joint (reward+dynamics) shifts on three DeepMind Control Suite environments: **Cheetah** (reward = desired speed, dynamics = torso length affecting weight/inertia), **Walker** (same axes), **Finger** (finger length). Train/test split is random 85/15 over task parameters; 5 seeds.

**Baselines.**
1. Context-conditioned policy (imitation-learning style: predict $a^*$ from $(s, \psi, \mu)$).
2. Context-conditioned policy + UVFA (also predict $q^*$, with the same TD loss as HyperZero) — this baseline shares *every* learning signal with HyperZero, so the comparison isolates the hypernet contribution.
3. MAML (zero-shot and few-shot).
4. PEARL (zero-shot and few-shot, but PEARL infers the context from states/actions instead of being given ground truth).

**Headline result.** HyperZero significantly outperforms all baselines across reward shifts, dynamics shifts, and joint shifts. Crucially, the gap over **Context+UVFA** — which has every component of HyperZero except the hypernet — is large, demonstrating that the *modularity of the hypernet (mapping context → policy parameters)* is the load-bearing component, not the auxiliary value-prediction or TD-regularisation losses.

**Initial takeaway.** Hypernets give a structural advantage for *transfer*: generalising in *policy-parameter space* ($\psi_i, \mu_i \mapsto \pi_i^*$) generalises better than generalising in *action space* ($(s, \psi_i, \mu_i) \mapsto a^*$). This is the "modularity argument" of Galanti & Wolf (2020) cashed in for RL transfer. The framework also produces a paired $\hat Q$ that can be used for offline-to-online fine-tuning or task visualisation.

### Phase 2 — Graduate-level deep dive (HyperZero)

#### 2.1 The parameterised-MDP family and the supervised-learning reduction

The paper assumes a **parameterised MDP family**:

$$
\mathcal{M} = \{M_i \mid M_i = (\mathcal{S}, \mathcal{A}, T_{\mu_i}, R_{\psi_i}, \gamma)\}, \quad \psi_i \sim p(\psi), \quad \mu_i \sim p(\mu)
$$

with $p(\psi), p(\mu)$ fixed prior distributions over reward and dynamics parameters. The state and action spaces are shared. This is closely related to *contextual MDPs* (Hallak, DiCastro, Mannor 2015) where the learner sees the context.

**The key conceptual step.** View an RL algorithm, once converged, as a *deterministic mapping*

$$
M_i \;\xrightarrow{\;\text{RL Algorithm}\;}\; \pi_i^*(a|s),\, Q_i^*(s,a) \qquad (\text{Eq. 1})
$$

Since $M_i$ shares everything except $(\psi_i, \mu_i)$ across the family, we can write this more compactly as

$$
M(\psi_i, \mu_i)\;\xrightarrow{\;\text{RL Algorithm}\;}\; \pi_i^*(a|s, \psi_i, \mu_i),\, Q_i^*(s, a |\psi_i, \mu_i) \qquad (\text{Eq. 2})
$$

The HyperZero claim is that, under Assumptions 1 (parameters $\psi_i, \mu_i$ are i.i.d. from priors) and 2 (the RL solver converges to the near-optimum), approximating this map is a **supervised-learning problem**: given offline rollouts from each training task, fit $H_\Theta$ to predict $(\hat\pi_\theta, \hat Q_\varphi)$.

Note: Puterman (2014) guarantees the optimal policy of a given MDP is *deterministic*, so $\hat\pi_\theta: \mathcal{S} \to \mathcal{A}$ is parameterised deterministically with no loss of optimality. This matches the TD3 deterministic policy class.

#### 2.2 The hypernetwork architecture

For each MDP $M_i$:

$$
[\theta_i;\, \varphi_i] \;=\; H_\Theta(\psi_i, \mu_i)
$$

where $\theta_i, \varphi_i$ are the *full weights* of the policy network $\hat\pi_{\theta_i}$ and Q-network $\hat Q_{\varphi_i}$ respectively. $H_\Theta$ is the only network with learnable parameters; $\hat\pi, \hat Q$ have no independent learnable parameters — they are entirely generated.

The "main networks" $\hat\pi_\theta, \hat Q_\varphi$ are the standard architecture used in TD3 (full implementation details in their Appendix C). All baselines share the same main-network architecture for fair comparison; the only difference is whether the weights come from a hypernet $H_\Theta(\psi, \mu)$ or are directly parameterised (with context as an input).

#### 2.3 Dataset generation and the two-stage training paradigm

**Stage 1 (per-task RL).** For each of $N$ training tasks $M_i$:
1. Run standard TD3 on $M_i$ for $10^6$ steps to obtain $\pi_i^*, Q_i^*$.
2. Roll out $\pi_i^*$ on $M_i$ for 10 episodes; store the trajectories.

The full dataset $\mathcal{D}$ contains tuples $\langle \psi_i, \mu_i, s, a^*, s', r, q^* \rangle$ where $a^* = \pi_i^*(s)$ and $q^* = Q_i^*(s, a^*)$.

**Stage 2 (hypernet training).** Train $H_\Theta$ with the combined loss $L(\Theta) = L_{\text{pred.}}(\Theta) + L_{\text{TD}}(\Theta)$.

#### 2.4 The prediction loss

$$
L_{\text{pred.}}(\Theta) = \mathbb{E}_{(\psi_i, \mu_i, s, a^*, q^*)\sim\mathcal{D}}\!\left[\,(\hat Q_{\varphi_i}(s, a^*) - q^*)^2\,\right] \;+\; \mathbb{E}_{(\psi_i, \mu_i, s, a^*)\sim\mathcal{D}}\!\left[\,(\hat\pi_{\theta_i}(s) - a^*)^2\,\right]
$$
$$
\text{where}\quad [\theta_i;\,\varphi_i] = H_\Theta(\psi_i, \mu_i) \qquad (\text{Eq. 4})
$$

This is straightforward imitation-style regression on the policy and value heads.

#### 2.5 The TD regularisation loss — direction is flipped

This is the most distinctive piece of machinery:

$$
L_{\text{TD}}(\Theta) = \mathbb{E}_{(\psi_i, \mu_i, s, a^*, s', r, q^*)\sim\mathcal{D}}\!\left[\,\bigl(\,r + \gamma\,\hat Q_{\varphi_i}(s', a')\;-\;q^*\,\bigr)^2\,\right] \qquad (\text{Eq. 5})
$$

where $a' = \hat\pi_{\theta_i}(s')$ with stopped gradients on $a'$ (the policy gradient flows through the prediction loss, not this term's policy term).

**Reading.** In standard deep-RL TD learning (Mnih 2013, Lillicrap 2015), the *current* value estimate is moved toward the *target* estimate $r + \gamma\,Q_{\text{target}}(s', a')$. HyperZero **reverses the direction**: it moves the *target estimate $r + \gamma\,\hat Q(s', a')$ toward the current $q^*$**, treating $q^*$ as the ground truth (justified by Assumption 2 that the source RL solver converged). The role of this loss is to *regularise the hypernet's generated $\hat Q$ to be Bellman-consistent with its generated $\hat\pi$*, not to learn the value function from scratch.

This is conceptually a *consistency regulariser* between the two generated networks — a property that standard regression-only training (Eq. 4 alone) does not enforce.

#### 2.6 Algorithm pseudocode

| Stage | Step |
|---|---|
| **Inputs** | parameterised $R_\psi, T_\mu$; priors $p(\psi), p(\mu)$; hypernet $H_\Theta$; main networks $\hat\pi_\theta, \hat Q_\varphi$; learning rate $\alpha$; number of tasks $N$. |
| **Stage 1** | For $i = 1, \dots, N$: sample $\psi_i \sim p(\psi), \mu_i \sim p(\mu)$; build $M_i$; run RL solver to obtain $\pi_i^*, Q_i^*$; store rollouts $\tau_i^*$ in $\mathcal{D}$. |
| **Stage 2** | While not done: sample mini-batch $\langle \psi_i, \mu_i, s, a^*, s', r, q^* \rangle \sim \mathcal{D}$; generate $\hat\pi_{\theta_i}, \hat Q_{\varphi_i}$ via $H_\Theta(\psi_i, \mu_i)$; compute $L = L_{\text{pred.}} + L_{\text{TD}}$; gradient step $\Theta \leftarrow \Theta - \alpha\,\nabla_\Theta L$. |

Only $\Theta$ is learnable. The main networks have no independent parameters.

#### 2.7 The zero-shot rollout

Test-time inference on a new $(\psi, \mu)$:

$$
\hat\pi_{\theta}(a|s, \psi, \mu) \;\xrightarrow{\;\text{Policy rollout in env}\;}\; \hat\tau(\psi, \mu) \qquad (\text{Eq. 3})
$$

No further training. The hypernet emits $\theta = H_\Theta(\psi, \mu)$ once at the start of the episode (or once per step if desired), and the policy executes.

#### 2.8 Comparison to MAML / meta-RL / context-conditioned baselines

The conceptual contrast they draw with **context-conditioned policy + UVFA** is the cleanest:

| Method | What is learned | Generalisation space |
|---|---|---|
| Context + UVFA | $(s, \psi, \mu) \mapsto a^*, q^*$ | **action space** |
| HyperZero | $(\psi, \mu) \mapsto \theta^*$, then $\hat\pi_\theta(s) \mapsto a$ | **policy / weight space** |

The paper hypothesises (and the ablation confirms) that generalising in *policy space* exploits more structure than generalising in *action space*. This is precisely the modularity-and-abstraction argument from Galanti & Wolf (2020) and von Oswald et al. (2019) applied to the RL transfer setting.

**Comparison to MAML.** MAML (Finn et al. 2017) requires a *gradient adaptation step* on the target task — i.e., it is fundamentally a few-shot method. HyperZero requires no gradient updates on the target task because the hypernet has learned the mapping $(\psi, \mu) \to \theta$ directly. They report MAML zero-shot as a baseline and it underperforms.

**Comparison to PEARL.** PEARL (Rakelly et al. 2019) does not assume the context is observable; it *infers* the context from collected transitions via a probabilistic encoder. In the HyperZero setting, the context is given by assumption, so PEARL is at an information disadvantage — yet HyperZero also outperforms few-shot PEARL with explicit context substitution.

#### 2.9 What generalisation argument the paper relies on

The paper does **not** prove a formal generalisation bound. The structural arguments it relies on are:

1. **Modularity / abstraction (citing Galanti & Wolf 2020, Ha 2016, von Oswald 2019).** Hypernets factor the learning problem into "context → policy weights" and "policy → action," which gives more expressive abstraction than a single $(s, \text{context}) \to a$ map.
2. **Policy space vs. action space generalisation.** The hypernet generalises in the lower-dimensional policy-parameter space (one $\theta$ per task), where similar tasks should have similar $\theta$, whereas a context-conditioned policy must generalise in action space at every state $\times$ task pair.
3. **Bellman-consistency regularisation (the novel TD loss).** Forces the generated $(\hat\pi, \hat Q)$ to be self-consistent, which empirically helps (ablation: removing the TD loss hurts; removing the value-prediction altogether hurts more).

The ablation (Figure 7) decomposes HyperZero into HyperZero, HyperZero-without-TD, and HyperZero-without-value-and-TD; both ablation removals hurt, with the value-+TD combo providing additional gain. Note their honest observation that the gain from TD-loss is modest in proprioceptive control and they expect it to grow in visual control where value-fit provides representation-learning signal.

#### 2.10 Result summary

| Transfer setting | Environment | HyperZero vs. best baseline |
|---|---|---|
| Reward shift (zero-shot) | Cheetah | Significant improvement (Fig. 3a) |
| Reward shift (zero-shot) | Walker | Significant improvement (Fig. 3b) |
| Dynamics shift (zero-shot) | Cheetah | Significant improvement (Fig. 4a) |
| Dynamics shift (zero-shot) | Walker | Significant improvement (Fig. 4b) |
| Joint reward+dynamics shift | Cheetah, Walker | Significant improvement across the 2-D surface (Fig. 5) |

In several settings HyperZero recovers "nearly full" performance vs. a fresh TD3 trained from scratch on the test task — i.e., the zero-shot rollout is roughly as good as a fully retrained agent.

### Appendix A.2 — Section-by-section backbone (HyperZero)

> Raw backbone, in the paper's original section order.

**Abstract.** Train hypernets to generate behaviors across unseen task conditions, via a novel TD-based training objective and data from a set of near-optimal RL solutions for training tasks. Related to meta-RL, contextual RL, transfer learning; focus on zero-shot at test time enabled by known task parameters (context). Approach: view each RL algorithm as a mapping from MDP specifics to near-optimal value function and policy; approximate with hypernet given MDP parameters. Show this is a supervised-learning problem under certain conditions. Evaluate on DeepMind Control Suite continuous control with parameterised reward and dynamics. Outperforms multi-task and meta-RL baselines.

**§1 Introduction.** Human zero-shot behavior generalisation as motivation. Hypernet (Ha 2016) outputs all parameters of a target neural network. Train on full solutions of numerous RL problems in a family of MDPs where reward or dynamics changes. Hypernet outputs parameters of a fully-formed policy with no experience on a related but unseen task. Differences between tasks lead to large changes in optimal policy; need powerful learners with helpful loss functions. Modularity and abstraction (citing Galanti & Wolf 2020) is the structural argument. Three contributions: (1) hypernet as scalable approximator of RL algorithm as a mapping $\mathcal{M} \to \Pi^*$; (2) TD-based regularisation loss for Bellman consistency; (3) custom continuous control envs.

**§2 Background.**
- §2.1 MDPs — standard 5-tuple, $V^\pi$, $Q^\pi$, Bellman operator $B^\pi$.
- §2.2 GVF — General Value Functions extend $Q^\pi$ to include reward, dynamics, discount; UVFA (Schaul 2015) generalises across goals. Notes that HyperZero is related to GVFs and General Policy Improvement but does not seek to improve an existing generalised policy.
- §2.3 Hypernetworks — definition; only hypernet weights are learnable; main network used at inference; "relaxed form of weight sharing across layers"; modularity and abstraction enable efficient learning.

**§3 HyperZero.**
- §3.1 Problem formulation — Definition 1 (parameterised MDP family); Eq. 1 (RL algorithm as mapping); Eq. 2 (compact form); Eq. 3 (rollout in env). Assumption 1 (i.i.d. parameter sampling), Assumption 2 (RL solver converged). Both assumptions discussed; Assumption 2 relaxed empirically.
- §3.2 Generating optimal policies and value functions — Architecture (Figure 2). Eq. 4 ($L_{\text{pred.}}$).
- §3.3 TD regularisation — Eq. 5 ($L_{\text{TD}}$); explains the direction-flip vs. standard deep-RL TD.

Algorithm 1 (HyperZero pseudocode) appears in §3.

**§4 Evaluation.**
- §4.1 Experimental setup — DM Control Suite cheetah, walker, finger; reward params = desired speed (positive and negative); dynamics params = body size / weight / inertia. TD3 (Fujimoto et al. 2018) as the source RL algorithm; 1M training steps per source MDP; 10 rollouts per task. Random 85/15 train/test split; 5 seeds. Four baselines.
- §4.2 Results — Zero-shot transfer in reward (Fig. 3), dynamics (Fig. 4), joint (Fig. 5). Sample trajectories (Fig. 6). Ablation (Fig. 7) on value generation and TD loss.

**§5 Related Work.** Transfer / Contextual / Meta RL; Hypernets in RL (Sarafian et al. 2021 cited explicitly; Faccio et al. 2022a for goal-conditioned hypernets; Rashid et al. 2018 QMIX; Huang et al. 2021 continual model-based); Upside Down RL.

**§6 Conclusion.** Train on full RL solutions; hypernet outputs complete network policies for unseen tasks; nearly full performance at zero-shot. Practical for deployment in live systems.

---

## Cross-paper synthesis and carry-over to this project

### What the two papers agree on
- **Hypernet structure beats concatenation for RL conditioning.** Sarafian shows it for $s$-as-meta in single-task Q-functions; HyperZero shows it for $(\psi, \mu)$-as-meta across an MDP family.
- **The modularity / abstraction argument** (Galanti & Wolf 2020; von Oswald 2019) carries over from supervised learning to RL: factoring "context → weights" + "weights → output" beats a single concatenated $(s, c) \to y$ map.
- **Practical sensitivity to primary-net initialization.** Sarafian devotes a paragraph to Kaiming-uniform-matching primary init; the HyperZero paper inherits but does not re-justify this. Both papers cite Chang et al. 2019 as the canonical reference for hypernet weight init.
- **The dynamic network is kept small.** Sarafian explicitly uses a 1-hidden-layer × 256 dynamic net vs. the standard 2 × 256 MLP. HyperZero's main networks match standard TD3 sizes; the hypernet itself bears the parameter cost. This is consistent — the hypernet is large; the network whose weights it generates is modest.

### Where they diverge
- **Single-task vs. transfer.** Sarafian's hypernet is regenerated at every state for a single task ($w(s)$ varies with $s$, but the task is fixed). HyperZero's hypernet is regenerated once per task ($H(\psi, \mu)$ varies with $\psi, \mu$, but the same generated $\hat\pi_\theta, \hat Q_\varphi$ are used for every state in that task). These are two distinct uses of the same primitive.
- **Loss directions.** Sarafian uses standard TD3/SAC losses unchanged — only the architecture differs. HyperZero introduces a *flipped* TD loss that treats the source-RL $q^*$ as ground truth and regularises the *target estimate* toward it.
- **Meta-variable choice.** Sarafian *recommends* state-as-meta (SA-Hyper) for the critic, but in PEARL deliberately uses *context-as-base* because the context encoder is also learnable — back-propagating through three networks hurt. HyperZero uses $(\psi, \mu)$-as-meta because the parameters are observed scalars, not learned latents.

### Carry-over to a hypernetwork-extended modulator-conditioned agent (this project)

The project's modulator-conditioned-agent thread currently sits at rungs 1–2 of the conditioning ladder (FiLM and row-scaling hypernet). If we wanted to put a full-hypernet version on rung 3, the directly load-bearing details from these two papers are:

1. **Dynamic-layer form is FiLM + hypernet, not pure hypernet.** Sarafian Eq. 1 — $x^{l+1} = \sigma((1 + g^l(z)) \odot x^l W^l(z) + b^l(z))$ — combines a generated weight matrix $W^l(z)$ with a generated FiLM gain $g^l(z)$. The gain term is documented as *necessary* for stability (Littwin & Wolf 2019). This means rung 3 in our project should probably retain a FiLM-style gain term on top of the generated weight matrix, not replace FiLM with hypernet.

2. **Where to place the modulator: as meta-variable.** Sarafian's empirical conclusion is that the *state-like* input (the high-dimensional one — the agent's observation) should be the **base** variable and the *context-like* input (lower-dim, semantically slow-varying — in our case the modulator) should be the **meta** variable. This matches HyperZero's choice. The project's modulator is the natural meta-variable; the agent's recurrent / proprioceptive state is the natural base.

3. **Sub-architecture for the primary net.** Both papers use a deep, residual primary net (Sarafian: explicit ResNet; HyperZero: deep MLP with task-conditioning), and a small dynamic net (one hidden layer × 256). The asymmetry matters — the primary net needs the expressive capacity to learn the (modulator → weights) map; the dynamic net just needs to compute the policy or Q. For a recurrent-PPO-with-modulator design, this would translate to a ResNet-style modulator-MLP outputting the weights of *one* dense layer (or one head) of the agent.

4. **Weight initialization is load-bearing.** Both papers warn that bad primary-net init catastrophically amplifies into bad dynamic weights. Sarafian explicitly initialises the primary net so the *expected dynamic weight distribution* matches Kaiming-uniform. Any project-side implementation should include this exact trick or risk training instability. The canonical reference is Chang, Flokas, Lipson (2019) "Principled Weight Initialization for Hypernetworks."

5. **Learning-rate retuning is required.** Sarafian explicitly says they used the baseline algorithm's loss unchanged but had to retune the learning rate for the hypernet. A project-side hypernet ablation should expect to *re-sweep* the learning rate; the FiLM ablation's LR is not portable.

6. **Critic vs. policy generation.** Sarafian generates the *critic* (Q-function). HyperZero generates *both* the policy and the critic and adds a TD-consistency loss between them. For PPO (on-policy actor-critic, not deterministic-actor + Q), the analogue is less clear: PPO has a value head $V$, not a $Q(s,a)$ — so the SA-Hyper "$\nabla_a Q$" argument does not directly apply. But the *meta-policy* argument (Sarafian §4) does: if the modulator is treated as a "context" in the Meta-RL sense, putting the modulator in the meta slot of a hypernet-policy still gives the disentangled state-gradient / task-gradient factorisation in Eq. 18, which should reduce policy-gradient variance for the same sample count.

7. **Adaptation step elimination.** Sarafian's most striking Meta-RL result is that Hyper-MAML has no observable pre-vs-post-adaptation gap — the hypernet generalises without gradient updates on the new task. For our project, the analogue is: if the modulator is given as ground-truth input (which it is in our setup), a hypernet-conditioned agent may not need any modulator-specific fine-tuning to handle a new modulator value. The downstream implication is that a hypernet-conditioned agent could be a candidate for *zero-shot transfer across modulator settings*, not just for in-distribution performance gains.

8. **Empirical regime is continuous control, not gridworld.** Both papers use MuJoCo / DM Control Suite continuous-control benchmarks with continuous actions and TD3 / SAC (deterministic + double-Q) or MAML / PEARL. Our project uses a gridworld with discrete actions and recurrent PPO. The transfer is conceptually clean but the engineering (e.g. categorical policy head vs. deterministic continuous head, recurrent state vs. proprioceptive state) requires adaptation. None of the cited training-stability tricks (Kaiming-matching primary init, ResNet primary, FiLM gain) are PPO-specific or continuous-action-specific; they should port.

### Items that do NOT carry over directly
- **HyperZero's offline two-stage paradigm (RL-solve N tasks, then supervised-train the hypernet).** Our project is online; we are not solving N copies of the gridworld with different modulator values and then doing supervised distillation. A direct port would be expensive. The *online* analogue is closer to Sarafian — generate weights on the fly during PPO training.
- **HyperZero's flipped TD loss.** Only applies if you have a near-optimal $q^*$ to regress against. In our setting we do not.
- **Sarafian's $\nabla_a Q$ analysis.** Does not directly apply to discrete actions or to PPO's value head $V(s)$, although the SA-Hyper / state-as-meta heuristic still carries.

### Hand-off
If the project decides to prototype a rung-3 hypernet modulator, the design / implementation handoff to `senior-developer` should reference (i) Sarafian Eq. 1 for the dynamic-layer form (hypernet + FiLM gain), (ii) Chang et al. 2019 for primary-net init, (iii) the meta-variable placement rule (modulator = meta, agent observation = base), and (iv) the small-dynamic-net-large-primary-net asymmetry. The HyperZero offline distillation pipeline is *not* a near-term blueprint for the project but is a candidate route if we later want zero-shot transfer to unseen modulator values without re-training.

---

## Files written

- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/docs/project/references/TD/TD_lit_review_C_hypernet_rl.md` (this file)

Working extraction notes (kept for traceability, not consumed downstream): `tmp/sarafian_full.txt`, `tmp/rezaei_full.txt`.
