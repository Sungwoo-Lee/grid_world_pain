# Multi-Target Bayesian Inference and the Context-vs-Observation Design Choice in Pain RL

> **Purpose**: Follow-up to [`computational_pain_hypervigilance_tutorial.md`](./computational_pain_hypervigilance_tutorial.md), answering two questions that arise once the reader understands the standard single-target Bayesian pain model:
>
> 1. What happens when the *prior*, *likelihood*, and *"posterior"* of interest are **not over the same variable**? Concretely: prior over damage $x$, likelihood over an olfactory cue of a predator $o$, and the quantity we actually want is an avoidance *policy* $\pi$.
> 2. In conventional deep RL with neural-network function approximators, what is the formal difference between (a) using the damage level directly as **context** (e.g. FiLM-conditioned actor–critic), (b) feeding only olfactory observations and learning a policy **end-to-end**, and (c) doing proper **Bayesian / active-inference** style decision-making?
>
> **Audience**: Graduate-level — assumes familiarity with Bayesian inference, POMDPs, variational methods, and actor–critic RL.
> **Last updated**: 2026-04-17.
> **Style**: Deep dive, reorganized around the user's question. Backed by targeted NotebookLM queries against the project's 25-paper pain notebook.

---

## Table of Contents

1. [The Question Restated](#1-the-question-restated)
2. [Why the Textbook Bayes Rule Looks Like It Only Handles One Target](#2-why-the-textbook-bayes-rule-looks-like-it-only-handles-one-target)
3. [The POMDP Generative Model — Three Targets, One Joint](#3-the-pomdp-generative-model--three-targets-one-joint)
4. [Step 1 — Inference Over Hidden States (Perception)](#4-step-1--inference-over-hidden-states-perception)
5. [Step 2 — Inference Over Policies (Action Selection as Inference)](#5-step-2--inference-over-policies-action-selection-as-inference)
6. [Worked Example — Damage Prior, Olfactory Likelihood, Avoidance "Posterior"](#6-worked-example--damage-prior-olfactory-likelihood-avoidance-posterior)
7. [Three RL Regimes Compared — Observation-Only, Context-Conditioned, Bayesian/AIF](#7-three-rl-regimes-compared--observation-only-context-conditioned-bayesianaif)
8. [Formal Comparison Table](#8-formal-comparison-table)
9. [Failure Modes and Biological Realism](#9-failure-modes-and-biological-realism)
10. [Implications for the GridWorld Pain Project](#10-implications-for-the-gridworld-pain-project)
11. [References](#references)

---

## 1. The Question Restated

The base tutorial (§4) introduces Bayesian pain inference with the familiar one-target form

$$
P(x \mid o) = \frac{P(o \mid x)\, P(x)}{P(o)} \tag{1}
$$

where $x$ is the latent pain/damage state, $o$ is the nociceptive observation, and all three distributions — $P(x)$, $P(o\mid x)$, and $P(x\mid o)$ — concern **the same random variable $x$**. The "multi-target" scenario in the user's question breaks this symmetry:

| Object | Target | Concrete example (grid-world) |
|---|---|---|
| Prior | a **latent body state** $x_b$ | tissue damage / injury level |
| Likelihood | an **exteroceptive observation** $o_e$ of a **different** latent variable $x_e$ | olfactory cue of a predator |
| "Posterior" we actually want | an **action / policy** $\pi$ | avoidance vs. approach behavior |

The textbook equation (1) cannot be applied directly: the three objects are not distributions over the same space. This is *not* a deficiency of Bayesian theory — it is the standard **Partially Observable Markov Decision Process (POMDP)** / active-inference setup, in which the three objects live in a single joint generative model and are connected by two distinct inference steps. The purpose of this note is to spell that out formally and then contrast it with the two most common RL alternatives.

---

## 2. Why the Textbook Bayes Rule Looks Like It Only Handles One Target

The pedagogical Gaussian example in §4.2 of the base tutorial is a **filtering** problem in a degenerate sense: one hidden variable, one observation channel, no action. It is the minimal decision-theoretic unit, useful for building intuition about precision-weighting, but it is **not** the full computation an embodied agent performs. In that example the posterior $P(x\mid o)$ is itself the "decision" — there is no subsequent policy selection.

As soon as the agent must **act**, two things change:

1. The **prior** over the hidden state at time $t$ depends on what action was taken at $t-1$: $P(s_t \mid s_{t-1}, a_{t-1})$. The prior is action-conditioned.
2. The object of ultimate interest is not $P(s\mid o)$ but a **policy** $\pi$ — a mapping from beliefs to actions that optimizes some criterion. The policy lives in a different space than the state.

Bayesian decision theory handles this by adding a **utility** or **cost** layer on top of inference (classical Savage / Bayes-risk framing):

$$
a^* = \arg\min_a\ \mathbb{E}_{P(s\mid o)}\!\left[L(s, a)\right]\tag{2}
$$

This cleanly separates (i) inference over states — the posterior $P(s\mid o)$ — from (ii) selection over actions — the argmin of expected loss. Two different targets, two different operations.

Active inference (Friston, 2010; Smith, Friston & Whyte, 2022) unifies the two under a single objective, but still maintains the **same two-step factorization** inside the variational machinery. The rest of this document follows that structure.

---

## 3. The POMDP Generative Model — Three Targets, One Joint

The formal object that makes "multi-target" well-posed is the **joint generative model** of a POMDP. Smith, Friston & Whyte (2022) write it as

$$
\boxed{\;
p(o, s, \pi) \;=\; p(o\mid s, \pi)\, p(s\mid \pi)\, p(\pi)
\;}\tag{3}
$$

This single factorization *contains all three "targets" simultaneously*:

- $p(\pi)$ — a prior over **policies** (habits, innate biases, prior preferences).
- $p(s\mid \pi)$ — a prior over **latent states** *conditional on a policy*. This is where the "prior over damage level" lives: what does the agent expect its body state to be if it continues its current behavioral program?
- $p(o\mid s, \pi)$ — the **likelihood**, which maps hidden states (and implicitly the policy's effects on sensors) to observations. This is where "the olfactory cue of the predator" lives: $o$ is an exteroceptive signal whose probability depends on the latent threat state and the agent's own pose/action.

The three distributions occupy their own spaces — a state space $\mathcal{S}$, an observation space $\mathcal{O}$, a policy space $\Pi$ — but are glued together by the chain rule of probability. Bayesian inference on this joint model produces two conceptually distinct posteriors: $q(s\mid o, \pi)$ (perception, §4) and $q(\pi \mid o)$ (action selection, §5).

> **Note on the control-theoretic restatement.** Seymour, Crook & Chen (2023) write the same structure in a continuous, linear-Gaussian special case — LQG control. The state-space equations
> $$
> \dot{x}(t) = A\, x(t) + B\, u(t), \qquad y(t) = C\, x(t) + D\, u(t) \tag{4}
> $$
> separate the latent state $x(t)$ (e.g. injury), the exteroceptive/afferent observation $y(t)$ (e.g. nociception, olfaction), and the control input $u(t)$ (e.g. avoidance) into three distinct variables. The brain must invert these equations to estimate $x(t)$ from $y(t)$ — then choose $u(t)$ based on the estimate. This is the LQG incarnation of the two-step separation.

---

## 4. Step 1 — Inference Over Hidden States (Perception)

Given observations $o_{1:t}$ and the generative model (3), the perceptual posterior over states at time $\tau$ under a candidate policy $\pi$ is $s_{\pi,\tau} \triangleq q(s_\tau \mid o_{1:t}, \pi)$. Smith et al. (2022) give the marginal message-passing update

$$
\boxed{\;
s_{\pi,\tau} \;=\; \sigma\!\Big(\tfrac{1}{2}\bigl(\ln B_{\pi,\tau-1}\, s_{\pi,\tau-1} + \ln B_{\pi,\tau}^{\dagger}\, s_{\pi,\tau+1}\bigr) + \ln A^{\top} o_\tau\Big)
\;}\tag{5}
$$

where $A$ is the observation model (likelihood $p(o\mid s)$), $B_{\pi,\tau}$ is the state-transition model under policy $\pi$, and $\sigma$ is the softmax. **Equation (5) is where the heterogeneity of targets is resolved**: the likelihood $A^{\top} o_\tau$ mixes an *observation-space quantity* ($o_\tau$, e.g. olfaction) with a *state-space quantity* (the log-likelihood entries of $A$), producing a message in state space that is combined with the state-space prior $B_{\pi,\tau-1}\, s_{\pi,\tau-1}$. The different-target problem disappears because $A$ is precisely the map between the spaces.

Friston (2010) states the general free-energy form of the same step:

$$
\mu = \arg\min_\mu F(s, \mu), \tag{6}
$$

where $\mu$ are the variational parameters of $q(s)$ and $F$ is the variational free energy — an upper bound on $-\ln p(o)$ that becomes tight as $q$ approximates the true posterior. Concretely, $F$ decomposes as

$$
F = \underbrace{D_{\mathrm{KL}}\!\bigl[q(s)\,\|\,p(s)\bigr]}_{\text{complexity}} - \underbrace{\mathbb{E}_{q(s)}\bigl[\ln p(o\mid s)\bigr]}_{\text{accuracy}} \tag{7}
$$

(Smith et al., 2022). Minimizing complexity keeps $q$ close to the prior (no overfitting); maximizing accuracy fits the likelihood. This is the Bayesian brain's native bias-variance tradeoff, built into the objective.

### 4.1 Mapping the user's example to Step 1

- **Prior over damage** $p(x_b \mid \pi)$: the agent's expectation about tissue state under the current behavioral program.
- **Likelihood of olfactory cue** $p(o_{\mathrm{olf}} \mid x_e)$: where $x_e$ is the latent predator proximity state. Crucially, the likelihood target can be **a different hidden variable than the prior target** — the generative model has room for multiple hidden states $s = (x_b, x_e, \ldots)$, each with its own prior factor and each observed through its own likelihood factor. The factorization is $p(s,o) = p(o \mid s) \prod_i p(x_i \mid \mathrm{pa}(x_i))$, and message passing takes care of the cross-coupling.
- **Joint posterior** $q(x_b, x_e \mid o_{\mathrm{olf}}, o_{\mathrm{noc}}, \ldots)$: the object that enters policy selection.

So the apparent "different target" problem in the question — prior on damage, likelihood on olfaction — is just a **multivariate latent state** inside the same joint model. Eq. (5) generalizes verbatim; the softmax is taken over the joint state.

---

## 5. Step 2 — Inference Over Policies (Action Selection as Inference)

Once the state posterior is in hand, active inference treats **policy selection as another Bayesian posterior**, this time over $\pi$. Smith et al. (2022) give the closed form

$$
\boxed{\;
\pi \;=\; \sigma\!\bigl(\ln E \;-\; F \;-\; \gamma\, G\bigr)
\;}\tag{8}
$$

where the four terms encode four conceptually distinct contributions to the posterior over policies:

| Symbol | Meaning | Interpretation |
|---|---|---|
| $E$ | prior over policies $p(\pi)$ | habits, innate biases |
| $F$ | variational free energy (past/current) | evidence from observations already received |
| $G$ | expected free energy (prospective) | evaluation of each policy's future consequences |
| $\gamma$ | precision over $G$ | confidence / vigor of the prospective evaluation |

The **expected free energy** $G_\pi$ is the bridge that lets a distribution over policies be scored by expected future observations:

$$
\boxed{\;
G_\pi \;=\; \underbrace{D_{\mathrm{KL}}\!\bigl[\,q(o\mid\pi)\,\|\,p(o\mid C)\,\bigr]}_{\text{risk (goal-directed)}} \;+\; \underbrace{\mathbb{E}_{q(s\mid\pi)}\!\bigl[H[p(o\mid s)]\bigr]}_{\text{ambiguity (epistemic)}}
\;}\tag{9}
$$

Two things are worth noticing:

1. $p(o\mid C)$ is a **prior over preferred observations** — the agent's goals or homeostatic set-points, expressed in observation space. This is yet another target, and it lives in the same space as the likelihood, so the KL divergence is well-defined.
2. The ambiguity term $H[p(o\mid s)]$ is the expected **entropy of the likelihood** — precisely the *sensory precision* term. It is the formal handle by which state-dependent sensory precision (central sensitization, noise scaling with injury) enters action selection. A policy that leads to states with high likelihood entropy (noisy observations) is penalized even if its risk is low.

### 5.1 Why this resolves the user's "posterior over policy" puzzle

The posterior $q(\pi \mid o)$ is a genuine Bayesian posterior — over policies, not states. The trick is that the generative model $p(o, s, \pi)$ already has $\pi$ as a random variable. Bayes' theorem applied to this joint gives

$$
q(\pi\mid o) \;\propto\; p(o\mid \pi)\, p(\pi), \quad \text{with} \quad p(o\mid \pi) = \int p(o\mid s, \pi)\, p(s\mid \pi)\, ds. \tag{10}
$$

Active inference *approximates* the intractable $\ln p(o\mid \pi)$ by $-F$ (past) and $-\gamma G$ (future), yielding (8). So the "posterior over avoidance behavior" in the user's example is formally $q(\pi\mid o_{\mathrm{olf}}, o_{\mathrm{noc}}, \ldots)$, obtained by treating $\pi$ as a hidden variable jointly with $s$, then marginalizing — no new machinery required.

### 5.2 LQG and model-based RL as special cases

- **LQG (Seymour et al., 2023).** When dynamics are linear-Gaussian and costs are quadratic, (8)–(9) reduce to the separation principle: optimal control uses $u^*(t) = -K\, \hat{x}(t)$, where $\hat{x}$ is the Kalman estimate. The action-selection step reduces to matrix multiplication — a deterministic argmax collapse of (8) under zero noise in the EFE.
- **Model-based RL (LeDoux & Daw, 2018).** Level 4–5 "goal-directed" behavior corresponds to computing a policy by tree search / Bellman backup over a learned model — a model-based special case of (8) where the prospective term $G$ is replaced by sum-of-future-rewards. LeDoux & Daw explicitly separate this from **model-free habits** (level 3), where the mapping $o \mapsto a$ is cached and no state inference is performed. The model-free pathway is what conventional end-to-end deep RL primarily implements.
- **Model-free RL (Gershman, Uchida & Bhatt, 2024).** The classical TD update
  $$
  \Delta w = \alpha\, \delta_t\, \nabla_w \hat V_t, \quad \delta_t = r_t + \gamma \hat V_{t+1} - \hat V_t \tag{11}
  $$
  treats the value function $\hat V$ as a direct function of observation features, with no explicit posterior over states. Without a belief state, it suffers "biased value estimates" under partial observability and perceptual aliasing (Gershman et al., 2024).

---

## 6. Worked Example — Damage Prior, Olfactory Likelihood, Avoidance "Posterior"

To make (3)–(9) concrete, instantiate them on the user's scenario. Let

- $x_b \in \{\mathrm{healthy}, \mathrm{injured}\}$ — body damage latent.
- $x_e \in \{\mathrm{absent}, \mathrm{near}, \mathrm{adjacent}\}$ — predator proximity latent.
- $o_{\mathrm{olf}} \in \mathbb{R}^5$ — 5-d olfactory observation vector (the grid-world's actual sensor; §11.3 of the base tutorial).
- $o_{\mathrm{noc}} \in \mathbb{R}$ — nociceptive observation.
- $\pi \in \{\mathrm{approach\text{-}food}, \mathrm{flee}, \mathrm{rest}, \mathrm{scan}\}$ — candidate policies.

**Generative model factorization.** The joint decomposes as

$$
p(o_{\mathrm{olf}}, o_{\mathrm{noc}}, x_b, x_e, \pi) = p(o_{\mathrm{olf}}\mid x_e)\, p(o_{\mathrm{noc}}\mid x_b)\, p(x_b\mid \pi)\, p(x_e \mid \pi)\, p(\pi). \tag{12}
$$

Notice what this buys us:

- The **likelihood** $p(o_{\mathrm{olf}}\mid x_e)$ is over the olfactory observation given the *predator* latent — "different target from the damage prior," exactly as the user posed it. But that is fine: there is a **separate** likelihood factor $p(o_{\mathrm{noc}}\mid x_b)$ for nociception given damage. Each observation channel gets its own likelihood mapped to its own latent variable.
- The **prior** $p(x_b\mid \pi)$ is over damage. It is policy-conditioned because the damage state evolves under the agent's actions (approach increases expected damage via encounters with predators; rest decreases it via healing).
- The **"posterior over avoidance"** $q(\pi\mid o)$ is obtained by plugging (12) into (8). No category error — each object lives in its own space, and (8) scores policies by the expected consequences they induce on every one of those spaces.

**What the EFE looks like for "flee."** Under $\pi = \mathrm{flee}$, the predicted future states have $x_e \to \mathrm{absent}$ (escape successful) and $x_b \to \mathrm{healthy}$ (no further injury), predicting observations $o_{\mathrm{olf}} \to$ weak gradient and $o_{\mathrm{noc}} \to$ low. If the agent's preference prior $p(o \mid C)$ puts mass on low nociception and low olfactory threat, the **risk** term of (9) is small. The **ambiguity** term is moderate — fleeing reduces predator uncertainty but also reduces foraging information.

**What the EFE looks like for "approach-food."** Predicted future states have $x_e \to \mathrm{adjacent}$ (encounter predator), $x_b \to \mathrm{injured}$, with $o_{\mathrm{noc}}$ spiking. The risk term of (9) is **large** (preferred observations are violated). Action selection via (8) thus assigns low posterior probability to approach-food when the olfactory signal is strong — even though the damage prior $p(x_b\mid \pi)$ and the olfactory likelihood $p(o_{\mathrm{olf}}\mid x_e)$ are over *different targets*. The EFE integrates across them.

This is the core answer to the first question: **Bayesian / perceptual-decision theory handles mismatched targets by embedding all of them in a single joint generative model and performing two nested inferences — one over states, one over policies — linked by the expected free energy.**

---

## 7. Three RL Regimes Compared — Observation-Only, Context-Conditioned, Bayesian/AIF

We now turn to the second question. Given that in practice we implement agents with neural-network function approximators, what is the formal difference between the three regimes?

### 7.1 Regime A — Observation-only end-to-end deep RL

The agent receives only exteroceptive observations (e.g. olfaction, vision) and learns a parameterized policy $\pi_\theta(a\mid o)$ via policy-gradient or actor-critic updates. In the partial-observability regime, $o$ alone does not satisfy the Markov property, so the agent uses a recurrent encoder $h_t = f_\theta(h_{t-1}, o_t, a_{t-1})$ as an *implicit* belief state. The value function and policy are learned end-to-end by maximizing return:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_t \gamma^t r_t\right]. \tag{13}
$$

In Gershman et al. (2024)'s taxonomy this is model-free with a learned state representation — the neural network has to discover the body-state structure on its own, purely from the reward signal. No explicit $p(o\mid s)$, no explicit posterior, no explicit preference prior $C$.

### 7.2 Regime B — Privileged-context conditioning (FiLM / side-channel)

The agent still acts on observations $o$, but its internal pathways are **modulated** by a side-channel carrying ground-truth latent variables — in the user's example, the *actual* damage level $x_b$. Concretely (FiLM; Perez et al., 2018):

$$
h_\ell' \;=\; \gamma_\ell(x_b) \odot h_\ell \;+\; \beta_\ell(x_b), \tag{14}
$$

where $(\gamma_\ell, \beta_\ell)$ are affine transforms applied to the $\ell$-th layer's activations as functions of $x_b$. Formally this converts the problem from a POMDP over $s$ into an **MDP over $(o, x_b)$** — the agent has been handed a subset of the hidden state.

Mahajan & Seymour (2025) sharpen this distinction: the biological control problem is a "difficult control problem under uncertainty" — full POMDP — whereas privileged-context conditioning corresponds to "control with full observability." Seymour et al. (2023) frame it explicitly via the state-space separation (4): feeding $x(t)$ directly into the controller bypasses the Kalman-Bucy filtering step entirely.

### 7.3 Regime C — Bayesian / active-inference with a learned generative model

The agent maintains an explicit generative model $p_\theta(o, s, \pi)$, performs variational inference to obtain $q(s)$ via (5), and selects policies via (8)–(9). Neural-network function approximators can still do the heavy lifting: the likelihood $A$ and transition $B$ become amortized neural networks (e.g. the RSSM in DreamerV3 or the encoder/decoder of a VAE), but the *structure* — two-step inference, explicit preference prior $C$, explicit EFE — is preserved.

This is the regime Mahajan & Seymour (2025) call "forward engineering": mechanistic models whose parameters have interpretable meanings (likelihood matrix, transition matrix, preference vector), contrasted with the "purely data-driven approach, which aims to discover structure in data alone."

---

## 8. Formal Comparison Table

| Dimension | A. Observation-only end-to-end RL | B. Damage as FiLM context | C. Bayesian / active inference |
|---|---|---|---|
| **Underlying problem class** | POMDP treated with recurrent policy | MDP over $(o, x_b)$ — oracle reveals a subset of the latent | POMDP, solved explicitly |
| **Who infers $x_b$?** | The recurrent encoder, implicitly, driven by reward gradients | Nobody — it is given | The variational posterior $q(s\mid o)$ via eq. (5) |
| **Prior over latent state $p(x_b\mid \pi)$** | Entangled in network weights | Absent (replaced by ground truth) | Explicit, part of the generative model |
| **Likelihood $p(o\mid x_b)$** | Entangled in network weights | Implicit (policy bypasses it) | Explicit ($A$ matrix or learned decoder) |
| **Preference / goal representation $p(o\mid C)$** | Collapsed into the scalar reward | Collapsed into the scalar reward | Explicit as a vector in observation space |
| **State-dependent sensory precision** | Not natively handled — would require reward-shaping hacks | Not natively handled | Handled by the ambiguity term $H[p(o\mid s)]$ in eq. (9) |
| **Bias-variance / complexity control** | Ad-hoc regularization (weight decay, entropy bonus) | Ad-hoc regularization | Structural, via $F = \mathrm{Complexity} - \mathrm{Accuracy}$ (eq. 7) |
| **Sample efficiency** | Low — must sample every reward gradient | Medium — gradient has direct access to damage | High (in principle) — model-based planning on generative model |
| **Interpretability of latent** | Distributed activations; requires probing | Partially interpretable (one slot is $x_b$) | Explicitly decomposed: $A, B, C$ matrices have fixed roles |
| **Central sensitization modeling** | Requires retraining under new noise regime | Requires retraining; context does not tell the agent its sensors became noisier | Captured by changing the likelihood entropy; no retraining needed |
| **Placebo / phantom limb** | Not captured without extra machinery | **Impossible by construction** — the agent is told the true damage, so a zero-damage phantom limb cannot hurt | Captured natively via precision mismatches between prior and likelihood |
| **Cognates in the literature** | Gershman et al. (2024) model-free TD; LeDoux & Daw (2018) level 3 habits | LQG with full observability (special case of Seymour et al., 2023, eq. 4) | LeDoux & Daw (2018) level 4–6; Smith et al. (2022); Friston (2010) |

---

## 9. Failure Modes and Biological Realism

### 9.1 Regime A fails under partial observability and non-stationarity

Gershman et al. (2024) note that TD learning without an explicit belief state produces "biased value estimates" under aliasing. End-to-end policies can in principle recover — GRU/LSTM can learn a sufficient statistic — but sample complexity is large and the representation is opaque. If the observation noise statistics drift (e.g. $\kappa$ in the state-dependent perceptual noise of §11.4 of the base tutorial changes), the agent does not *know* that its sensors became less precise; it can only rediscover this through degraded returns.

### 9.2 Regime B fails as a biological model

This is the most important failure to flag for the project. Seymour et al. (2023) and the chronic-pain literature hinge on **computational decoupling between physical damage and experienced pain**. If the agent's policy is conditioned on the ground-truth damage level, then by construction:

- Placebo analgesia (pain suppression despite damage) cannot arise.
- Nocebo hyperalgesia (pain in the absence of damage) cannot arise.
- Phantom-limb pain cannot arise — Seymour et al.'s "persistent incongruent multisensory integration" requires a generative model that disagrees with the physical substrate.
- Hypervigilance as precision dysregulation (§9 of the base tutorial) has no substrate: there is no prior whose precision can be inflated.

In short, **Regime B is an engineering simplification that removes the phenomena the project exists to study**. It may still be useful as a control / upper-bound baseline ("what would the agent achieve if it had oracle damage information?"), but it cannot stand in as a model of interoceptive inference.

### 9.3 Regime A and C both admit hypervigilance — but differently

- In Regime A, hypervigilance must emerge as a learned *behavioral policy* shaped by reward. It is a distributed property of the network, difficult to localize.
- In Regime C, hypervigilance is an explicit *parameter configuration* of the generative model: elevated $\Pi_{\mathrm{prior}}$ on injury, elevated $\Pi_{\mathrm{obs}}$ on nociceptive channels, risk-dominated EFE (§9.4 of the base tutorial). This matches the "computational phenotyping" program of Mahajan & Seymour (2025) and is the reason the base tutorial argues for an active-inference / generative-model substrate.

### 9.4 Regime C's own failure mode: the active-inference trap

For completeness: Regime C is not immune to pathology. Seymour et al. (2023) formalize the chronic-pain failure mode as **information restriction** — protective policies suppress the observations needed to update the injury posterior, so $\hat x_b$ gets stuck at "injured." This is a local minimum of EFE, not a modeling limitation: it is the *point* of the account (§8.4–8.5 of the base tutorial).

---

## 10. Implications for the GridWorld Pain Project

Tying this back to the grid-world architecture (§11–13 of the base tutorial, and the current `feature/dreamer-opt` branch):

1. **Using damage as FiLM context (Regime B) buys speed at the cost of the research question.** If the modulation pathway receives $x_b$ directly — rather than the agent's *estimate* of $x_b$ — the agent cannot exhibit the precision-dysregulation phenotypes the project is designed to study. The neuromodulator's `percept pathway` (§11.5 of the base tutorial) takes $(\mathrm{sat}, \mathrm{nut}, \mathrm{inj}, \mathrm{collision})$ as input; if these are the **observed** (noisy, normalized) versions of the latent state, the architecture sits between Regime A and Regime C. If they are the raw latent values, it collapses to Regime B for those channels.

2. **DreamerV3's RSSM is the natural substrate for Regime C.** The posterior $q(z\mid h, o)$ and prior $p(z\mid h)$, with KL divergence $D_{\mathrm{KL}}[q\|p]$ as prediction error, are direct analogues of eqs. (5) and (7). Policy selection via imagination rollouts is an amortized approximation of (8)–(9), with the critic standing in for $-G_\pi$ and the entropy bonus standing in for the ambiguity term.

3. **For actor–critic with context conditioning, the principled upgrade is to feed the *estimated* body state, not the ground truth.** Concretely: route the body-state features through the recurrent encoder (so they are filtered through the partial-observability assumption), and allow the network to form its own posterior estimate $\hat x_b$ before FiLM-modulating downstream pathways. This preserves the sample-efficiency benefits of conditioning while keeping the inference problem intact.

4. **The asymmetric-$\kappa$ gap (Gap 2 in §13.2 of the base tutorial) is well-motivated only in Regime C.** Differential precision for nociceptive vs. non-nociceptive channels is a statement about the *likelihood* $p(o\mid s)$, which has no explicit role in Regime A and is bypassed in Regime B. In Regime C, $\kappa$ directly parameterizes the entropy term in (9), and the agent's policy will shift as a consequence of inference, not just as a consequence of reward-shaping.

---

## References

Citations below are drawn from the tutorial's annotated bibliography. The equations in §3–§9 are quoted directly from the NotebookLM-indexed sources (Smith, Friston & Whyte, 2022; Seymour, Crook & Chen, 2023; Friston, 2010; Gershman, Uchida & Bhatt, 2024; LeDoux & Daw, 2018; Tashjian, Zbozinek & Mobbs, 2021; Mahajan & Seymour, 2025).

- Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Gershman, S. J., Uchida, N., & Bhatt, M. (2024). Explaining dopamine through prediction errors and beyond. *Nature Neuroscience*.
- LeDoux, J. E., & Daw, N. D. (2018). Surviving threats: Neural circuit and computational implications of a new taxonomy of defensive behaviour. *Nature Reviews Neuroscience*, 19(5), 269–282.
- Mahajan, P., & Seymour, B. (2025). Forward and reverse engineering the pain system. *PAIN*.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI*. (Source for the FiLM formulation in eq. 14; not in the notebook.)
- Seymour, B., Crook, R. J., & Chen, Z. S. (2023). Post-injury pain and behaviour: A control theory perspective. *Nature Reviews Neuroscience*.
- Smith, R., Friston, K. J., & Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. *Journal of Mathematical Psychology*, 107, 102632.
- Tashjian, S. M., Zbozinek, T. D., & Mobbs, D. (2021). A decision architecture for safety computations. *Trends in Cognitive Sciences*, 25(5), 342–354.
- Wiech, K. (2016). Deconstructing the sensation of pain. *Science*, 354(6312), 584–587.

### Cross-references within this project

- Base tutorial: [`computational_pain_hypervigilance_tutorial.md`](./computational_pain_hypervigilance_tutorial.md) — §4 (single-target Bayes), §7 (LQG), §8 (active inference), §11–13 (grid-world mapping and gaps).
- FiLM / conditional modulation reference review: see `docs/project/references/` (referenced in project commits e0cb2bd, c596bcc).
