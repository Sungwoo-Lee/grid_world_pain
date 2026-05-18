# Decision Transformer — Literature Review (TD corpus, paper D)

**Paper:** Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling.* NeurIPS 2021.
**PDF:** `docs/project/references/TD/sources/Chen et al. 2021 - Decision transformer - Reinforcement learning via sequence modeling.pdf`
**Role in the TD reading list:** the **paradigm-contrast** paper. Every other paper in this batch (distributional RL, successor features, hypernetwork-RL) preserves the Bellman / value-function machinery — Decision Transformer (DT) discards it entirely and recasts the offline RL problem as autoregressive sequence modeling.

---

## Table of Contents

1. [Why this paper is on the TD reading list](#why-this-paper-is-on-the-td-reading-list)
2. [Phase 1 — Foundational Overview (undergrad-level)](#phase-1--foundational-overview-undergrad-level)
   - [What is offline RL?](#what-is-offline-rl)
   - [What does "RL as sequence modeling" mean?](#what-does-rl-as-sequence-modeling-mean)
   - [What is "return-to-go conditioning"?](#what-is-return-to-go-conditioning)
   - [Why does this work at all?](#why-does-this-work-at-all)
   - [Key findings, plain English](#key-findings-plain-english)
3. [Phase 2 — Graduate-Level Deep Dive](#phase-2--graduate-level-deep-dive)
   - [Trajectory representation and the return-to-go object](#trajectory-representation-and-the-return-to-go-object)
   - [Architecture: GPT over interleaved $(\hat R, s, a)$ tokens](#architecture-gpt-over-interleaved-hat-r-s-a-tokens)
   - [Training objective: autoregressive next-action likelihood](#training-objective-autoregressive-next-action-likelihood)
   - [Evaluation: return-to-go as the runtime controller](#evaluation-return-to-go-as-the-runtime-controller)
   - [Credit assignment via attention vs. via Bellman](#credit-assignment-via-attention-vs-via-bellman)
   - [Why offline? — what DT does and does not guarantee](#why-offline--what-dt-does-and-does-not-guarantee)
   - [The return-to-go token as an *implicit hypernetwork / conditioning signal*](#the-return-to-go-token-as-an-implicit-hypernetwork--conditioning-signal)
4. [Connections to the rest of the TD reading list](#connections-to-the-rest-of-the-td-reading-list)
   - [DT vs. distributional RL — does scalar $\hat R$ lose distributional information?](#dt-vs-distributional-rl--does-scalar-hat-r-lose-distributional-information)
   - [DT vs. successor features — bypassing the dynamics–reward decomposition](#dt-vs-successor-features--bypassing-the-dynamicsreward-decomposition)
   - [DT vs. hypernetwork-RL — return-conditioning as implicit hypernetwork](#dt-vs-hypernetwork-rl--return-conditioning-as-implicit-hypernetwork)
5. [Project-relevant point: the return-to-go token IS a learnable scalar modulator](#project-relevant-point-the-return-to-go-token-is-a-learnable-scalar-modulator)
6. [Limitations called out by the authors](#limitations-called-out-by-the-authors)
7. [Appendix: Section-by-Section Backbone](#appendix-section-by-section-backbone)

---

## Why this paper is on the TD reading list

**Question this doc answers.** What does it mean to do RL *without* TD learning, *without* a value function, and *without* a Bellman backup — and what trade-offs do you pay for that paradigm shift?

**Purpose for the project.** The TD corpus exists to map the design space around the project's "modulator-conditioned policy" thread. Most of that corpus (distributional RL, successor features, hypernetwork-RL) lives squarely inside the value-function tradition: a critic learns $Q(s,a)$ or $\psi(s,a)$, a Bellman backup propagates rewards, and the "modulator" enters as a conditioning input to the *value head*. Decision Transformer is included as a deliberate counter-example: an offline RL method where the entire pipeline is rebuilt around an autoregressive transformer over $(\hat R_t, s_t, a_t)$ tuples, with **no critic, no Bellman backup, no TD error**, and where the role of "modulator" is played by a single scalar token — the desired *return-to-go* — fed in alongside the state.

**Headline framing.** Reading DT against the rest of the TD batch makes one structural observation sharp: DT is structurally identical to a "modulator-conditioned policy" architecture, except (a) the modulator is the desired return rather than an interoceptive signal, and (b) the conditioning mechanism is transformer self-attention rather than FiLM or a hypernetwork. The same FiLM / cross-attention conditioning machinery the project uses elsewhere applies directly, and DT's success is evidence that a *single learned scalar token* is a competent re-parameterization knob for a policy.

**Verdict / what to take away.** DT is the cleanest demonstration in the corpus that "policy = $\pi_\theta(a \mid s, \text{context})$, trained by maximum likelihood" is a viable alternative to the Bellman pipeline when (i) the dataset is fixed and offline, (ii) the context signal is well-aligned with the desired behavior, and (iii) there is enough data to support a transformer. It does not provide policy-improvement guarantees, and the conditioning signal it uses (return-to-go) is interpretable but lossy — see the Phase 2 derivation of why.

---

## Phase 1 — Foundational Overview (undergrad-level)

### What is offline RL?

In **online** RL the agent collects its own data: act in the environment, see a reward, update the policy, repeat. In **offline** RL (also called batch RL) you do not get to act. Someone hands you a fixed dataset of past trajectories — sequences of $(s_t, a_t, r_t)$ tuples collected by some other policy, possibly a mix of policies of varying quality. Your job is to find the best policy you can extract from that dataset, knowing you will only be allowed to deploy it later — you cannot try things, you cannot ask for more data.

Why is this hard? The standard RL machinery (Q-learning, policy gradients) is built on the assumption that you can probe the environment to correct your value estimates. With a fixed dataset, any region of state-action space that the dataset does not cover is essentially unknown — and the standard estimators will happily extrapolate confidently into that region, producing what is called *value overestimation* and *distribution shift*. Most of the offline-RL literature is about constraining the policy to stay near states-and-actions the dataset actually covers (Conservative Q-Learning, BEAR, BRAC, behavior-regularized methods).

### What does "RL as sequence modeling" mean?

Imagine you ignore that this is RL at all. Just think of each trajectory as a sequence of tokens — like a sentence in language modeling. A trajectory of length $T$ becomes a sequence

$$\text{trajectory} = (\text{return-info}_1, s_1, a_1, \text{return-info}_2, s_2, a_2, \dots, \text{return-info}_T, s_T, a_T).$$

Now you train a GPT-style transformer on these sequences with the standard next-token prediction loss, but only ask it to predict the *action* tokens. After training, the transformer has learned: "given this past context, what action token comes next?"

That is it. **No Bellman backup. No value function. No TD error. No policy gradient. No reward maximization in the loss.** Just supervised next-token prediction on trajectory data. This is the "paradigm shift": the entire optimization machinery of RL — bootstrapping, critic networks, target networks, double-Q tricks, importance sampling — is gone, replaced by a single autoregressive likelihood.

### What is "return-to-go conditioning"?

The trick that makes "sequence-model the dataset" actually *do RL* lives in what goes in the "return-info" slot.

The naive option is to put the immediate reward $r_t$ there. But that does not help the model generate *good* trajectories — the model would just learn the distribution of trajectories the dataset already contains, with their usual rewards. To steer the model toward high-return behavior, DT uses a quantity called the **return-to-go**:

$$\hat R_t = r_t + r_{t+1} + \dots + r_T = \sum_{t'=t}^{T} r_{t'}.$$

That is, at each timestep, instead of the *past* reward, you store the *sum of all future rewards in this trajectory*. The model sees, in effect: "from this state, this many rewards will be collected." It learns the conditional distribution of next actions given the past states, the past actions, **and the desired return that should still be collected from here onward**.

At test time, you reverse the trick. You tell the model: "I want a trajectory that achieves return $R^*$." You feed in the start state $s_1$ and the **desired** return $\hat R_1 = R^*$. The model generates an action $a_1$. You execute that action in the environment, observe a reward $r_1$, and **decrement** the return-to-go: $\hat R_2 = \hat R_1 - r_1$. You feed the new state $s_2$ and the updated $\hat R_2$ back in. Repeat. The model is now "running" a policy conditioned on the desired return, generating actions that — if its training distribution covered comparable returns — should achieve $R^*$.

### Why does this work at all?

Three reasons the paper invokes:

1. **Hindsight relabeling at scale.** Every trajectory in the offline dataset is a positive example of *how to achieve the return it happened to achieve*. By conditioning on return-to-go (a hindsight quantity — you can only compute it because you already see the whole trajectory), every trajectory becomes a labeled training point of the form "from $s_t$ with desired-return $\hat R_t$, the right action is $a_t$." The model learns the conditional $p(a_t \mid s_t, \hat R_t, \text{history})$, which can then be queried with a high desired return at test time.

2. **Attention does credit assignment.** Standard TD methods propagate reward backwards in time one step at a time, through Bellman bootstraps. This is slow over long horizons, suffers from the "deadly triad" (bootstrapping + function approximation + off-policy), and gets distracted by spurious correlations. Self-attention, by contrast, can directly connect an action token at time $t$ to a reward-influencing event at time $t'$ many steps later, in a single forward pass — there is no temporal propagation, only direct association.

3. **No bootstrapping, no discount, no deadly triad.** Because the loss is supervised likelihood, none of TD-learning's known instabilities (value overestimation, bootstrap divergence under function approximation, target-network coupling) appear.

### Key findings, plain English

- **Benchmark performance.** On standard offline RL benchmarks (Atari with 1% of the DQN replay buffer, D4RL continuous control on HalfCheetah / Hopper / Walker / Reacher, and the Key-to-Door long-horizon credit-assignment task), DT matches or beats Conservative Q-Learning (CQL — the strongest TD baseline at the time) on a majority of tasks, and beats behavior-cloning baselines and TD baselines like BEAR, BRAC, REM, QR-DQN.
- **Return-conditioning actually works.** When you tell DT to achieve return $R^*$, the realized return at test time tracks $R^*$ closely — there is a near-linear relationship across most tasks. On some tasks (e.g., Seaquest) DT extrapolates *beyond* the best return seen in the dataset.
- **Long-horizon credit assignment is a strength.** On Key-to-Door, where the reward is delayed to the very end and only granted if the agent picked up a key early, CQL collapses to ~13% success while DT reaches ~95% (with 10K trajectories). Attention does the credit assignment.
- **Sparse / delayed rewards.** When the D4RL benchmarks are modified so that all rewards are zeroed out except for one cumulative reward at the end of the episode, CQL collapses to near-zero performance, while DT is *minimally affected* — because DT does not need dense rewards at all, only return-to-go labels.
- **Not just imitation on a subset.** A natural concern is "DT is just behavior cloning on the top-X% of the dataset." The authors run *percentile behavior cloning* baselines (10%/25%/40%/100% BC) and show DT consistently matches or beats the best of these in the low-data regime, suggesting DT genuinely benefits from training on the *full* distribution and using return-conditioning to query a slice at test time.

### Initial takeaway

Decision Transformer is the proof-of-concept that you can build an offline RL method whose policy update is *literally just maximum-likelihood next-token prediction*, and where the "what return do you want?" lever is a single scalar token. For the project, this is interesting because it strips the policy-conditioning problem down to its bare bones: a scalar input modulates a deep policy, and the modulation is implemented by feeding the scalar in as a token to a transformer. That is the same conditioning problem the project's interoceptive modulator solves, just with a different scalar (return vs. interoceptive signal) and possibly a different conditioning mechanism (attention vs. FiLM / hypernet).

---

## Phase 2 — Graduate-Level Deep Dive

### Trajectory representation and the return-to-go object

The trajectory format is

$$\tau = \big(\hat R_1, s_1, a_1, \hat R_2, s_2, a_2, \dots, \hat R_T, s_T, a_T\big), \tag{1}$$

with the return-to-go at time $t$ defined as

$$\hat R_t \;=\; \sum_{t'=t}^{T} r_{t'}, \qquad r_{t'} = R(s_{t'}, a_{t'}). \tag{2}$$

Two things to notice immediately:

1. **$\hat R_t$ is a hindsight quantity at training time.** It is computable only because the entire trajectory $\tau$ is observed. This is what makes the training-time supervision well-defined: every trajectory in the offline dataset, regardless of quality, is a fully-labeled training example under the return-to-go convention.

2. **$\hat R_t$ is a control-knob quantity at inference time.** During rollout, $\hat R_1$ is *specified by the user* as the desired return $R^*$, and subsequent $\hat R_t$ are computed by the autoregressive consistency rule

   $$\hat R_{t+1} \;=\; \hat R_t - r_t, \tag{3}$$

   i.e., subtract the realized reward from the remaining "return budget." Equations (2) and (3) are consistent — under (2), $\hat R_{t+1} = \sum_{t'=t+1}^T r_{t'} = \hat R_t - r_t$ holds by inspection — so the inference-time update rule is exactly the time-difference of the training-time definition.

3. **Choice not to discount.** The paper deliberately uses the *undiscounted* sum $\sum_{t' \ge t} r_{t'}$ rather than $\sum_{t' \ge t} \gamma^{t'-t} r_{t'}$. This matters because the standard TD machinery requires $\gamma < 1$ for the Bellman operator to be a contraction; DT, having no Bellman operator, has no such requirement. The cost is that very long horizons can make $\hat R_t$ numerically large; the authors address this by normalizing $\hat R$ per environment.

### Architecture: GPT over interleaved $(\hat R, s, a)$ tokens

Given a context window of length $K$, DT processes $3K$ tokens (one per modality at each of the last $K$ timesteps). Each modality gets its own learned linear projection into the transformer embedding space:

$$e_t^{\hat R} = W_{\hat R}\,\hat R_t + b_{\hat R} + \text{pos}(t), \tag{4a}$$
$$e_t^{s} = \phi_s(s_t) + \text{pos}(t), \tag{4b}$$
$$e_t^{a} = W_a\,a_t + b_a + \text{pos}(t), \tag{4c}$$

where $\phi_s$ is a linear projection for vector states and a small CNN for image states (Atari), and $\text{pos}(t)$ is a **learned per-timestep embedding** added to *all three* tokens belonging to timestep $t$. (Critically this is not a standard sinusoidal per-token positional encoding — one "position" in DT corresponds to three transformer tokens. The model has to learn the within-triple modality ordering implicitly.)

The interleaved sequence

$$x = \big(e_1^{\hat R}, e_1^{s}, e_1^{a}, e_2^{\hat R}, e_2^{s}, e_2^{a}, \dots\big) \tag{5}$$

is fed through a stack of causal-masked self-attention blocks. The causal mask means token $i$ attends only to tokens $1, \dots, i$. Each block computes, for each token,

$$z_i \;=\; \sum_{j \le i} \mathrm{softmax}\big(\{\langle q_i, k_{j'}\rangle / \sqrt{d}\}_{j' \le i}\big)_{\!j}\; v_j, \tag{6}$$

with $q_i = W_Q x_i$, $k_i = W_K x_i$, $v_i = W_V x_i$. The action prediction head $\pi_\theta(\cdot \mid \cdot)$ is a linear (continuous) or affine + softmax (discrete) layer applied to the hidden state at *action-prediction positions* — i.e., the positions right after each $s_t$ token in the sequence. The hidden state $h_t^{(a)}$ at that position has, by causal masking, attended to

$$\hat R_1, s_1, a_1, \hat R_2, s_2, a_2, \dots, \hat R_t, s_t,$$

i.e., all of history up to and including the current return-to-go and state, but not the current action (which is what we are predicting).

### Training objective: autoregressive next-action likelihood

The model is trained only to predict the action tokens; the paper notes that adding state-prediction or return-prediction auxiliary losses was tried and found unnecessary on the benchmarks considered (Section 3, "Training"). The loss is

$$\mathcal{L}(\theta) \;=\; \mathbb{E}_{\tau \sim \mathcal{D}}\!\left[\,\sum_{t=1}^{T} \ell\!\big(a_t,\, \pi_\theta(\,\cdot \mid s_{\le t},\, \hat R_{\le t},\, a_{< t})\big)\,\right], \tag{7}$$

with $\ell$ being cross-entropy for discrete actions and squared error for continuous actions. Equivalently, with the continuous-action MSE realization,

$$\mathcal{L}_{\text{cont}}(\theta) \;=\; \mathbb{E}_{\tau \sim \mathcal{D}}\!\left[\,\frac{1}{T}\sum_{t=1}^{T} \big\| a_t - \pi_\theta(s_{\le t}, \hat R_{\le t}, a_{<t}) \big\|_2^2\,\right]. \tag{8}$$

**Derivation of the standard "supervised hindsight relabeling" reading.** Consider the joint distribution over trajectories induced by the data-collection process: $p_{\mathcal{D}}(\tau)$ is a mixture of the policies that generated the dataset. By Bayes' rule, the conditional next-action distribution under this mixture, conditioned on the *future*-defined quantity $\hat R_t$, is

$$p_{\mathcal{D}}(a_t \mid s_{\le t}, \hat R_{\le t}, a_{<t}) \;=\; \frac{p_{\mathcal{D}}(a_t, \hat R_{\ge t} \mid \cdot)}{p_{\mathcal{D}}(\hat R_{\ge t} \mid \cdot)}, \tag{9}$$

i.e., the probability that the data-generating mixture would have produced action $a_t$ *given* that the remainder of the trajectory from $t$ accumulates total reward exactly $\hat R_t$. Maximizing the log-likelihood in (7) is a consistent estimator of this conditional. The "policy improvement" trick is then purely a *querying* trick at inference time: by setting the conditioning value $\hat R_1$ to a return larger than the *empirical* mean return in $\mathcal{D}$, you are querying the conditional at a high-return slice — and *if* the dataset contains enough mass at that return for the conditional to be estimable, you recover behavior consistent with that high return. **This is the entire policy-improvement mechanism. There is no Bellman backup. There is no off-policy correction. There is only conditional querying.** The implication is also the chief limitation: DT cannot reliably extrapolate to return targets the dataset does not support. The paper observes some Seaquest extrapolation (Section 5.2) but does not provide a guarantee.

### Evaluation: return-to-go as the runtime controller

At rollout time, the algorithm is (paraphrasing the paper's Algorithm 1):

```
R_target ← desired-return    # user-specified scalar
s_1 ← env.reset()
context ← (R_target, s_1)
for t = 1, 2, ..., until done:
    a_t ← π_θ(context)[-1]            # action at last position
    s_{t+1}, r_t, done ← env.step(a_t)
    R_{t+1} ← R_t - r_t                # return-to-go update, Eq. (3)
    context ← context ⊕ (a_t, R_{t+1}, s_{t+1})
    context ← context[-K:]             # truncate to context length K
```

**Notes on this loop that matter for the project's use case:**

- The model is given the past $K$ timesteps as context; the policy is therefore $\pi_\theta(a_t \mid s_{t-K+1:t}, \hat R_{t-K+1:t}, a_{t-K+1:t-1})$. This is a *fully observed* policy in the trajectory-history sense — there is no recurrent state to carry, the past-$K$-window does all the work.
- The conditioning signal $\hat R_t$ changes every timestep — it is *not* a static task descriptor. The transformer learns to behave conditionally on this dynamically-changing scalar.
- The decrement rule (3) is the only place the agent's realized reward enters the model's runtime computation. There is no critic update, no policy gradient. Everything else is fixed at training time.

### Credit assignment via attention vs. via Bellman

This is the most theoretically interesting claim of the paper. In Bellman-based methods, credit for a reward $r_T$ obtained at step $T$ propagates to an action $a_1$ at step $1$ only by being bootstrapped $T-1$ times through the Bellman equation

$$Q(s_t, a_t) \;\leftarrow\; r_t + \gamma\, \max_{a'} Q(s_{t+1}, a'). \tag{10}$$

Each bootstrap step contracts by $\gamma$, accumulates function-approximation error, and can be destabilized by the "deadly triad" of bootstrapping + function approximation + off-policy data. The number of training updates needed to assign credit across $T$ steps is at least $T$ (it takes that many Bellman sweeps for the reward to reach $s_1$).

In DT, the attention pattern of (6) lets a hidden state at position $t$ attend *directly* to any earlier position in the context window — including ones containing reward / return information. The attention weight $\mathrm{softmax}(\langle q_i, k_j \rangle / \sqrt d)$ is a *single forward-pass* association between event $i$ and event $j$, independent of $|i - j|$. In particular, the credit for a delayed reward at position $T$ can be assigned to an early action by the model placing a high attention weight on the relevant earlier $(\hat R, s, a)$ triple.

The empirical evidence the paper gives for this:

- **Key-to-Door, Section 5.3, Table 5.** The agent must pick up a key in phase 1 to receive a reward at the door in phase 3, with phase 2 being a distractor empty room. Training on **random walks** (so the dataset itself does not solve the task), DT reaches ~95% success with 10K random trajectories; CQL reaches 13.3%. The relabeling-by-hindsight + attention pipeline solves a problem that Bellman backups cannot.
- **Section 5.4, Figure 5.** When DT is augmented to predict return tokens as well as action tokens, the attention pattern from late states attends sharply to "pivotal" earlier events (picking up the key, reaching the door). The attention is doing the state-event association.

**Caveat on the comparison.** The attention argument relies on the context window $K$ being large enough to span the relevant horizon. For Key-to-Door the authors set $K$ to the entire episode length. The credit-assignment argument breaks if the relevant cause lies outside the context window — and in that case DT and TD methods are on more equal footing. This caveat matters for the project: any modulator-conditioned policy using a transformer over short contexts inherits this same limitation.

### Why offline? — what DT does and does not guarantee

DT lives naturally in the offline regime because **all of its supervision is hindsight-relabeled data**. There is no exploration mechanism in the algorithm — the policy is never asked to "try something new and see what happens." The conditional distribution $p(a_t \mid s_t, \hat R_t, \dots)$ is estimated from whatever the dataset provides.

What this gives:

- **No policy-evaluation step.** No critic, no bootstrap, no value overestimation.
- **No explicit policy-constraint machinery.** Methods like CQL/BEAR/BRAC have to explicitly penalize the policy for deviating from the dataset's behavior policy. DT does not — the maximum-likelihood loss *automatically* concentrates on the dataset's action distribution. There is no notion of an "out-of-distribution action" because the model can only produce actions it has been trained to produce.

What this does not give:

- **No policy-improvement guarantee beyond the data.** The conditional $p(a_t \mid s_t, \hat R_t, \dots)$ is only well-estimated at $\hat R_t$ values supported by the dataset. Querying at $\hat R_t$ far above the empirical return distribution is extrapolation, and the model has no built-in mechanism to do this safely. The paper's Seaquest extrapolation is observational, not theoretical.
- **No off-policy correction.** Importance-weighted estimators that other offline RL methods use are absent. The dataset distribution is taken at face value.
- **No exploration when extended online.** Naively applying DT online would just generate the conditional behavior at the asked return; it would not explore the state space.

### The return-to-go token as an *implicit hypernetwork / conditioning signal*

Here is the framing that matters most for the project. From the transformer's perspective, the return-to-go token $\hat R_t$ is *a single scalar input that materially changes the model's output distribution over actions*. Concretely, the action distribution depends on $\hat R_t$ through

$$\pi_\theta(a_t \mid s_{\le t}, \hat R_{\le t}, a_{<t}). \tag{11}$$

Now consider the partial derivative

$$\frac{\partial \log \pi_\theta(a_t \mid \cdot)}{\partial \hat R_t}. \tag{12}$$

A non-zero (12) is precisely what makes DT work: changing the desired-return scalar must change the action distribution. The model has learned a *continuous family of policies indexed by $\hat R_t$*, all sharing the same parameters $\theta$ — i.e.,

$$\{\pi_\theta(\cdot \mid s, R, \text{hist}) : R \in \mathbb{R}\}$$

is a one-parameter family of conditional policies. Structurally this is **identical** to:

- a **FiLM**-conditioned policy where the scalar conditioning signal is $R$ and the FiLM parameters $\gamma, \beta$ are produced as $\gamma = f_\gamma(R, \text{hist})$, $\beta = f_\beta(R, \text{hist})$;
- a **hypernetwork**-conditioned policy where a small network $g(R, \text{hist})$ outputs the policy weights;
- a **cross-attention** policy where $R$ is one element of an attended-over context.

DT uses the third option (cross-attention by virtue of the attention mechanism: $R$ is a token, the action position attends to it). But it is the same conditioning problem the project's modulator-conditioned policy is solving, with the conditioning mechanism swapped for self-attention. This connection is made precise in §[DT vs. hypernetwork-RL — return-conditioning as implicit hypernetwork](#dt-vs-hypernetwork-rl--return-conditioning-as-implicit-hypernetwork) below.

---

## Connections to the rest of the TD reading list

### DT vs. distributional RL — does scalar $\hat R$ lose distributional information?

**Yes — and this is the central comparison to make.** Distributional RL (Bellemare et al. 2017 / C51, Dabney et al. 2018 / QR-DQN and IQN) learns the *full distribution* $Z^\pi(s, a)$ over returns rather than just its mean $Q^\pi(s, a) = \mathbb{E}[Z^\pi(s,a)]$. The motivation is that the return-distribution carries information the mean throws away: variance for risk-sensitive policies (Lim & Malik 2022), quantiles for robust decision-making, multi-modality for stochastic-outcome environments.

DT, by contrast, uses a **single scalar $\hat R_t$** as its conditioning signal. Compare the information content:

- **Distributional RL** conditions implicitly on the full distribution $Z^\pi(s, a)$ — the critic outputs a quantile function or categorical distribution, and decisions can be made under any risk preference.
- **DT** conditions explicitly on one realization-tail quantity, $\hat R_t = \sum_{t' \ge t} r_{t'}$, a single sample from the return-distribution of the dataset trajectory containing this transition.

This has two consequences:

1. **The same $(s_t, \hat R_t)$ pair can correspond to many different return *distributions* depending on the trajectory context.** In a stochastic environment, two trajectories passing through $s_t$ with the same realized $\hat R_t$ might have come from very different policies and induce very different futures. The DT conditional $p(a_t \mid s_t, \hat R_t, \text{hist})$ marginalizes over this uncertainty — it does not represent it.
2. **DT cannot do risk-sensitive policy selection.** Distributional RL methods that derive risk-averse policies from quantiles (CVaR, conditional VaR; see Dabney et al. IQN, Lim & Malik 2022) have no analogue in DT, because there is no learned *return distribution* — only conditional next-action distributions given a deterministic return scalar.

**However**, DT does have an attractive *complementary* property: the conditioning signal is **specifiable at inference time**, not learned implicitly. A user can request a specific return target, which is exactly the kind of explicit-control knob distributional RL has to back out via post-hoc quantile selection.

**A natural hybrid not pursued in this paper:** condition on a *quantile-of-the-return-distribution* token instead of a return-scalar token. This would unify DT's explicit-control property with distributional RL's information-preserving representation. The contemporaneous Trajectory Transformer (Janner et al. 2021, ref [44] in DT's bibliography) discretizes the return for similar reasons.

### DT vs. successor features — bypassing the dynamics–reward decomposition

Successor features (SF; Barreto et al. 2017, 2019; Borsa et al. 2018 / USFAs) decompose the value function as

$$Q^\pi(s, a) \;=\; \psi^\pi(s, a)^\top w, \tag{13}$$

where $\psi^\pi$ encodes the *dynamics-driven* discounted feature-occupancy under $\pi$, and $w$ is the *reward-mapping* vector (assuming the reward decomposes as $r(s,a) = \phi(s,a)^\top w$). The architectural payoff is **transfer**: change the task to a new reward vector $w'$, and you reuse the same $\psi^\pi$ to instantly evaluate the new task's $Q$-value, then run Generalized Policy Improvement (GPI) over a library of source-task policies.

DT does not attempt this decomposition at all. The return-to-go $\hat R_t$ entangles dynamics and reward into a single scalar at training time. The architectural consequences:

- **DT has no zero-shot reward transfer story.** If the reward function changes, DT must be retrained on the new $\hat R$ labels (which depend on the new reward).
- **DT has no analogue of $\psi$ that you can "reuse with a different head."** Everything is end-to-end conditioned on the realized return.
- **However**, DT trivially handles *return-target* transfer — you can ask for any return target you like at inference time, with the same model. SF handles *reward-vector* transfer; DT handles *return-target* transfer. These are different transfer surfaces.

The deeper point: SF and DT make opposite bets about where to put the structure. SF says "factor the dynamics from the reward, so reward changes are cheap." DT says "do not factor at all — let the transformer learn the joint $(s, \hat R) \to a$ map directly, and exploit the scaling of supervised learning." On large offline datasets DT's bet pays off; in transfer-heavy settings SF's bet pays off.

A non-trivial hybrid: replace the scalar $\hat R$ in DT with a learned *task vector* $w$ analogous to SF's reward decomposition. This would let DT inherit SF's zero-shot reward transfer. Borsa et al.'s USFAs already explore the conditional-on-task-vector flavor of this; DT-style sequence modeling has not (in this paper) been applied to it.

### DT vs. hypernetwork-RL — return-conditioning as implicit hypernetwork

Hypernetwork-RL (Sarafian et al. 2021; Rezaei-Shoshtari et al. 2023) explicitly factors a policy / value network into the form $\pi_{\theta(\cdot)}(a \mid s)$ where the parameter set $\theta(\cdot) = h_\eta(c)$ is produced by a hypernetwork $h_\eta$ as a function of a context vector $c$ (a task descriptor, embedding, or modulator signal). The architectural promise is sharp factorization: the *primary network* implements the function "given parameters, map state to action," and the *hypernetwork* implements the function "given context, choose parameters." This gives clean compositional control over context-dependent behavior.

DT's mechanism is structurally analogous but **implicit**. There is no explicit hypernetwork in DT; instead, the conditioning happens through the self-attention pattern. Let us make this precise.

For a transformer block with input tokens $\{x_j\}$, the output at the action-prediction position $i$ is (specializing (6)),

$$z_i = \sum_j \alpha_{ij}(R, s, a)\, v_j, \qquad \alpha_{ij}(R,s,a) = \mathrm{softmax}_j \!\big(\langle q_i, k_j\rangle/\sqrt d\big), \tag{14}$$

where the dependence of $\alpha_{ij}$ on the return-to-go tokens $R$ is implicit through the queries and keys (which are linear projections of the input tokens, *which include the return-to-go tokens*). The function $\alpha_{ij}(R, s, a) v_j$ is, viewed as a function of $R$ with $(s, a)$ held fixed, a *content-dependent re-weighting of the value tokens by $R$*. In effect, the return-to-go token re-parameterizes the linear combination that produces the action representation.

Compare this to FiLM, where a scalar context $c$ modulates a feature $h$ via $h \leftarrow \gamma(c) \odot h + \beta(c)$. FiLM is a *channel-wise affine* re-parameterization of a feature; attention-based conditioning on $R$ is a *token-wise* re-parameterization of the attention-weighted sum. Both are forms of parameter-free conditional computation — neither materializes a separate weight matrix per context — but both implement the same logical operation: "given context $c$, modify the computation."

A useful way to state the equivalence: **the function $f_\theta(\cdot, R) : s \mapsto a$ implemented by DT, viewed as a function of $R$, is an implicit hypernetwork.** It is implicit because no separate $h_\eta(R) \to \theta$ map is materialized, but the effective behavior — a continuous family of policies indexed by $R$ — is the same. Hypernetwork-RL makes this explicit; DT keeps it implicit and folded into the attention machinery.

For the project, this matters because the question "should the modulator condition the policy through FiLM, a hypernetwork, or cross-attention?" is now visible as a *single design question with three implementations*, not three separate architectural choices. DT is the cross-attention answer; Rezaei-Shoshtari and Sarafian are the explicit-hypernetwork answer; the project's existing FiLM work is the affine-modulation answer. All three are conditioning the policy on a scalar / vector context, just with different parameterizations of "how does the context enter the network."

---

## Project-relevant point: the return-to-go token IS a learnable scalar modulator

State the structural mapping crisply.

| Decision Transformer | Project's modulator-conditioned policy |
|---|---|
| Return-to-go scalar $\hat R_t$ | Interoceptive modulator signal (e.g., a scalar from a small modulator network) |
| Linear embedding $e^{\hat R}_t = W_{\hat R} \hat R_t + b_{\hat R}$ | Linear embedding / FiLM-generator MLP of the modulator value |
| Causal-attention mixing with state and action tokens | FiLM affine modulation / hypernetwork weight generation / cross-attention conditioning |
| Inference-time control: pick $\hat R_1 = R^*$ | Inference-time control: modulator value is read off from interoceptive state |
| Training-time supervision: hindsight $\hat R_t$ from offline trajectories | Training-time supervision: PPO / RL gradient on policy outputs, with modulator-conditioning baked into the network |
| No explicit policy-improvement, only conditional querying | Online policy improvement via PPO/RL; modulator-conditioning serves as a per-context re-parameterization |

The two architectures answer different questions (DT: how do we do offline RL without TD? Project: how does an interoceptive signal modulate online policy behavior?), but they share the same *conditioning primitive*: **a scalar input that changes the policy's output distribution by entering the network as a token / FiLM coefficient / hypernetwork input.**

This means three concrete things for the project:

1. **DT is empirical evidence that a single learned scalar token is a competent re-parameterization knob for a deep policy.** Whatever loss is being used (in DT it is supervised hindsight likelihood; in the project it is PPO), the conditioning mechanism is a well-established building block.

2. **The choice between FiLM, hypernetwork, and cross-attention for the project's modulator is a real choice with empirical evidence on each side.** DT's success with cross-attention is one data point; Rezaei-Shoshtari's hypernetwork transfer results are another; the project's FiLM results are a third. The TD corpus collectively contains evidence on all three.

3. **DT's hindsight relabeling trick is not directly applicable to the project's online setting, but its inference-time control trick is.** The project does not relabel trajectories with hindsight — it has a real modulator signal at runtime. But the runtime mechanic of "feed a scalar at every step; the policy is conditional on it" is exactly the same.

---

## Limitations called out by the authors

From Section 7 of the paper:

- **Context length and return-conditioning are hyperparameters** that require tuning per environment. The paper uses $K \in \{30, 50\}$ for Atari and full-episode contexts for Key-to-Door; the right value is not principled.
- **Results are on standard RL benchmarks** (Atari, D4RL, Minigrid Key-to-Door). The paper does not demonstrate at language-model scale.
- **A simple supervised loss is used.** Self-supervised pretraining tasks (mask-and-predict, etc.) on trajectory data are flagged as future work.
- **Return embeddings are simple linear projections.** More sophisticated embeddings (e.g., distributional, learned-quantile) are flagged as future work.
- **Scaling and generalization are not directly studied.** The paper appeals to the known scaling behavior of transformers but does not show RL-specific scaling laws.

Additional limitations a careful reader should note:

- **No policy-improvement guarantee beyond the dataset return support.** Extrapolation to high $\hat R_1$ is empirical, not theoretical.
- **Information loss in scalar $\hat R$.** See [DT vs. distributional RL](#dt-vs-distributional-rl--does-scalar-hat-r-lose-distributional-information).
- **No online / exploration story.** A naive online extension would not explore — the model can only produce conditioned behavior, not seek novel states.
- **Deterministic-vs-stochastic environments.** In highly stochastic environments, the same $\hat R_t$ condition can correspond to wildly different future paths; the conditional $p(a_t \mid s_t, \hat R_t)$ is then high-variance.

---

## Appendix: Section-by-Section Backbone

The original-section-ordered extraction, preserved so the Phase 1/2 synthesis above remains traceable to the source.

### Abstract
- Framework: cast RL as a sequence modeling problem; draw on transformer simplicity / scalability and language-model advances (GPT-x, BERT).
- Architecture: Decision Transformer — conditional sequence modeling over $(\hat R, s, a)$. No value-function fitting, no policy-gradient computation.
- Mechanism: causal-masked transformer outputs optimal actions conditioned on desired return, past states, past actions.
- Result: matches or exceeds state-of-the-art model-free offline RL on Atari, OpenAI Gym (D4RL), and Key-to-Door.

### Section 1 — Introduction
- Transformers have shown large-scale generalization in language [GPT-3] and image [DALL-E]; contrasts with RL methods that learn a single narrow-policy.
- Paradigm shift: instead of TD learning, train transformer on collected experience with a sequence-modeling objective.
- Benefits invoked: bypass bootstrapping (avoid "deadly triad"); avoid discounting (avoid short-sightedness); reuse stable transformer training; credit assignment directly via self-attention (vs. slow Bellman backups subject to distractors); model wide behavior distributions for transfer.
- Distinction from "upside-down RL" (UDRL; Srivastava et al. 2019, Kumar et al. 2019): motivation is sequence modeling (long contexts, language-model objective), not pure supervised learning.
- Hypothesis tested in offline RL: train autoregressive model on offline trajectories of $(s, a, r)$; "prompt" with return tokens to specify desired policy expertise.
- Illustrative example: shortest-path on a directed graph as RL. Reward $0$ at goal, $-1$ else. Train GPT on random-walk trajectories with returns-to-go; at test time prompt with high-return and generate actions; recover optimal paths despite training on random data.

### Section 2 — Preliminaries

**Section 2.1 — Offline RL.** MDP $(S, A, P, R)$ standard. Goal: maximize $\mathbb{E}[\sum_{t=1}^T r_t]$. Offline = fixed dataset of trajectories, no environment access. Hard due to error propagation / value overestimation.

**Section 2.2 — Transformers.** Vaswani et al. self-attention layer: $z_i = \sum_j \mathrm{softmax}(\{\langle q_i, k_{j'}\rangle\}_{j'}) \cdot v_j$. Implicit credit-assignment by query-key similarity. DT uses GPT (causal mask, $j \le i$).

### Section 3 — Method

**Trajectory representation.** Modeling returns-to-go (sum of future rewards), not rewards directly, because the model must condition on *future desired return*. Format: $\tau = (\hat R_1, s_1, a_1, \hat R_2, s_2, a_2, \dots, \hat R_T, s_T, a_T)$.

**Architecture.** Last $K$ timesteps $\to$ $3K$ tokens. Modality-specific linear embedding for each of $\{\hat R, s, a\}$ (CNN for image states), followed by layer normalization. **Learned per-timestep embedding** (one timestep = three tokens; differs from standard token-level positional encoding) added to each token. GPT processes tokens; predicts action tokens autoregressively.

**Training.** Minibatch sequences of length $K$. Predict $a_t$ from input tokens; cross-entropy (discrete) or MSE (continuous). State/return prediction is optional (not needed for benchmarks).

**Evaluation.** Specify target return $\hat R_1 = R^*$ at start. Generate $a_t$, execute, decrement: $\hat R_{t+1} = \hat R_t - r_t$, obtain $s_{t+1}$. Continue until termination.

**Algorithm 1.** Pseudocode for continuous-action case: linear embeddings, per-timestep positional embedding, interleave $(R, s, a)$, transformer, linear action head, MSE loss, autoregressive rollout with return-decrement.

### Section 4 — Evaluations

**Section 4.1 — Atari.** 1% of DQN-replay dataset (~500K transitions). Baselines: CQL, REM, QR-DQN, behavior cloning. Context $K = 30$ (Pong: $K = 50$). Results in Table 1: DT competitive with CQL on 3/4 games (Breakout 267.5±97.5 vs CQL 211.1; Qbert 15.1±11.4 vs CQL 104.2; Pong 106.1±8.1 vs CQL 111.9; Seaquest 2.4±0.7 vs CQL 1.7). Outperforms REM, QR-DQN, BC in most games.

**Section 4.2 — OpenAI Gym (D4RL + a Reacher variant).** Three dataset settings: Medium, Medium-Replay, Medium-Expert. Baselines: CQL, BEAR, BRAC, AWR, BC. Results in Table 2: DT achieves highest scores in majority of tasks; average without Reacher = 74.7 (DT) vs 63.9 (CQL); all settings 69.2 (DT) vs 54.2 (CQL).

### Section 5 — Discussion

**Section 5.1 — Is DT just BC on a return-filtered subset?** Introduces Percentile Behavior Cloning (%BC) — BC on top $X\%$ of timesteps by episode return. Sweep $X \in \{10, 25, 40, 100\}$%. In high-data D4RL regime, %BC matches or beats other offline RL methods, and DT matches best %BC. In low-data Atari regime, %BC is weak and DT beats all %BC variants — suggests DT benefits from training on the full dataset, not just a filtered subset. Table 3 (D4RL) and Table 4 (Atari).

**Section 5.2 — Does DT model the return distribution?** Vary target return at inference. Figure 4 shows realized return strongly correlated with desired return across tasks. Some Atari tasks (Seaquest) show extrapolation beyond max dataset return.

**Section 5.3 — Long-term credit assignment.** Key-to-Door (Mesnard et al. 2020): three-phase grid task, binary reward at door iff key picked up. Train on random-walk trajectories. Table 5: DT 71.8% / 94.6% (1K / 10K trajectories) vs CQL 13.1% / 13.3% vs BC 1.4% / 1.6% vs %BC 69.9% / 95.1%. DT (and %BC on success-filtered) work; CQL fails. Full-episode context $K = T$ used.

**Section 5.4 — Can transformers be accurate critics?** Modify DT to predict return tokens in addition to action tokens on Key-to-Door. Predicted return-probability tracks events (key, door) during episode. Attention weights concentrate on critical events. Figure 5.

**Section 5.5 — Sparse-reward robustness.** Delayed-return variant of D4RL: all rewards zero except cumulative reward at last step. Table 6: DT minimally affected (e.g., Hopper-Medium-Expert: 107.3 delayed vs 107.6 dense); CQL collapses (9.0 delayed vs 111.0 dense).

### Section 6 — Related work
- Offline RL: action-space constraints (BCQ), value pessimism (CQL, BEAR), pessimistic dynamics models (MOReL, MOPO). DT is closest to likelihood-based skill-discovery approaches but uses sequence modeling, not variational methods.
- Supervised learning in RL: UDRL (Srivastava 2019, Kumar 2019, Ogma 2019). Key difference: DT motivated by sequence modeling (long contexts + scaling), not just supervised learning. Kumar et al. 2019 corresponds to $K = 1$; DT shows longer contexts outperform.
- Concurrent: Trajectory Transformer (Janner et al. 2021) — similar in spirit but adds state and return prediction, plus discretization, making it more model-based.
- Credit assignment via state-association: Ferret 2019, Harutyunyan 2019, Hung 2019, RUDDER (Arjona-Medina 2018), Liu 2019, Raposo 2021. DT lets credit assignment emerge from transformer architecture without explicit reward decomposition.
- Conditional language generation: many works on controllable text generation (PPLM, CTRL, GeDi, etc.). DT differs in that reward is time-varying and continuously decremented during generation.
- Transformers in RL: prior work (Parisotto 2020 stable-T, Zambaldi 2018 relational, Ritter 2020 episodic memory) used transformers as architectural augmentations to actor-critic; DT replaces the actor-critic entirely.

### Section 7 — Conclusion
- DT unifies language modeling and offline RL.
- Matches / outperforms strong offline RL baselines with minimal modification from standard language-modeling architecture.
- Societal impact: error-mode and dataset-bias considerations for transformers in MDPs.
- Limitations: context-length and return-to-go hyperparameter sensitivity; benchmarks are standard RL only; simple supervised loss (self-supervised pretraining is future work); simple linear return embeddings; scaling not directly studied.

### Section 8 — Acknowledgements
- Funding: Berkeley DeepDrive, Open Philanthropy, NSF NRI #2024675.
- Notable individuals: Justin Fu (D4RL setup), Aviral Kumar (CQL baselines).

### References (notable cross-corpus links)
- [1] Vaswani et al. 2017 — Attention is all you need.
- [12] Radford et al. — GPT-1.
- [17] Kumar et al. 2020 — **CQL** (DT's primary TD-learning baseline).
- [23] **Dabney et al. 2018 — QR-DQN** (also in this TD reading list).
- [24] Fu et al. 2020 — D4RL benchmark.
- [44] Janner et al. 2021 — **Trajectory Transformer** (concurrent sequence-modeling RL).
- [8, 9, 10] UDRL family — closest predecessors to DT's return-conditioning idea.
- [15, 26, 47, 48] — credit-assignment via state-association literature (Mesnard, Raposo, RUDDER, Liu).
