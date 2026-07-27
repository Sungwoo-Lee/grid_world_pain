# Why the same two algorithms invert across EVAAA and our grid world

> **One line:** There is no single inversion — there are two, and they have different causes. "Recurrent PPO fails in EVAAA but succeeds here" is real and is driven mostly by *what the observation is* (mandatory first-person pixels there, a 27-number hand-factored vector here) and *what death costs* (nothing there, −100 here). "DreamerV3 wins there but loses here" is largely an **accounting artifact**: on a matched-experience axis DreamerV3 still wins in our grid world too; only the wall-clock verdict flips, because our JAX environment makes environment steps essentially free while Unity makes them the dominant cost.

---

## 1. The question, in plain language

Our lab built two environments in which an artificial animal must keep itself alive by managing internal bodily variables — hunger, injury, temperature. **EVAAA** is a 3D world built in the Unity game engine, published at NeurIPS 2025; the agent sees the world through a first-person camera. **Our grid world** is a 10×10 board written in JAX, where the agent receives 27 numbers describing its body and its immediate surroundings.

Two families of learning algorithm were tried on both. One is **recurrent PPO** ("model-free": it learns a policy directly from experience, with a memory unit so it can remember what it has seen). The other is **DreamerV3** ("model-based": it first learns an internal simulator of the world, then trains its policy by dreaming inside that simulator).

The puzzle is that the two families appear to swap places. In EVAAA, recurrent PPO never got off the ground while DreamerV3 learned to survive. In our grid world, recurrent PPO reaches roughly 177 survival steps while DreamerV3 looks slower on the clock.

**Headline answer.** The swap is not a single phenomenon. DreamerV3's *experience efficiency* advantage survives in our grid world (it beats recurrent PPO by 2.4× when both have seen the same number of episodes) — what flips is only the *wall-clock* ranking, and that flip is caused by the environment's cost per step, not by anything about world models. The genuinely algorithmic half of the puzzle is why recurrent PPO fails in EVAAA, and the two strongest candidates are (i) EVAAA's task cannot be solved without vision, and a policy-gradient-only learner starves for supervision on a 12,288-pixel input, and (ii) EVAAA charges nothing for dying, which creates a reward landscape in which ending the episode is a rational move — a trap that on-policy PPO walks into and DreamerV3 is structurally shielded from.

**Also, honestly:** our grid world does not need a world model to reach its current performance. §9 says what that should mean for how the two papers are framed together.

---

## 2. What the EVAAA paper actually reports (verified from the PDF)

Source: `docs/project/references/InteroceptiveAI/sources/Lee et al. 2025 - EVAAA - ...pdf`, 42 pp. Section/appendix references below are to that PDF.

### 2.1 Corrections to the starting hypothesis

The brief's recollection was *"recurrent PPO completely failed to train; DreamerV3 succeeded, with curriculum learning."* Three corrections, all load-bearing:

| Recollection | What the paper says | Where |
|---|---|---|
| "recurrent PPO completely failed" | **Correct, and stronger than stated:** PPO+LSTM "failed to demonstrate successful learning… survival steps remained minimal throughout training in **both Level 1-1 and Level 1-2**" — i.e. it failed on the two *easiest* levels, the ones with no obstacles and no predator. | App. J.1; Fig. 20 |
| implicitly, "feedforward PPO was fine" | **No.** Feedforward PPO "demonstrated successful survival only in Level 1-1 and failed to scale." DQN likewise: "converged faster than DreamerV3 in Level 1-1, but failed to generalize." **Both model-free baselines mostly failed.** The recurrent one failed harder. | §8 |
| "DreamerV3 succeeded, with curriculum learning" | Curriculum helped *every* algorithm ("Agents that undergo the curriculum learning outperform the ones without"), and **DreamerV3 was the best in both the with- and without-curriculum conditions.** The paper never claims Dreamer *required* the curriculum. DreamerV3 itself "drops sharply from Level 3-2 onward." | §8; Fig. 4a right |

**Consequence for hypothesis 5 (curriculum asymmetry as a confound): rejected.** The curriculum cannot explain the algorithm inversion, because (a) Dreamer led with and without it, and (b) PPO+LSTM was trained on Levels 1-1 and 1-2 — the first two curriculum stages — and still failed there.

### 2.2 The two structural facts that matter most

**Vision is mandatory in EVAAA.** The paper's own modality-ablation study (App. J.2, Figs. 25–26): agents with **'No Vision' fail to learn**, as do agents with 'No Essential Variables'. Removing olfaction is nearly harmless; removing thermoception or collision is intermediate. So EVAAA is not a task where a 33-dimensional vector suffices — the decisive information arrives as a 64×64×3 egocentric RGB image, under walls that occlude, fog, and a five-phase day/night cycle that modulates visibility (§4.3, App. E.2, Fig. 17).

**EVAAA appears to charge nothing for dying — and the paper reports agents exploiting that.** Two problems here, and the first is an internal inconsistency in the paper:

- **Main text §2** gives the *drive-reduction* (difference) form, citing Keramati & Gutkin:
  $$R\big(s^{\text{int}}(t), s^{\text{int}}(t{+}1)\big) = \sum_i \beta_i\Big[D_i\big(s^{\text{int}}_i(t)\big) - D_i\big(s^{\text{int}}_i(t{+}1)\big)\Big], \qquad D_i(s) = (s - s^*_i)^2 .$$
- **Appendix D** gives the *level* form, which is a different function:
  $$r_t = -\sqrt{\sum_{i=1}^{4}\left(\frac{EV_i(t)-\mu_i}{\sigma_i}\right)^2} \;\le\; 0 .$$

These are not equivalent. **This is an erratum-level discrepancy the first author can settle from the repository in minutes, and several conclusions below depend on which one shipped.** Under either form, however, the paper describes **no terminal penalty**: "An episode is terminated if any EV falls outside its defined range" (App. E.1 II), with no negative reward attached to the termination event itself.

And the paper reports the exact behavioural fingerprint this predicts:

> "some agents adopt a strategy of repeatedly colliding with allies to deplete their damage EV and terminate the episode. This behavior reflects a learned policy that favors resetting the environment over persisting in low-reward or difficult-to-recover states." (App. J.2, *Self-Termination*)

and, in the main text, that this suicide policy **generalised** out of Level 3-2 into the Y-maze test (§9). That is not a quirk; it is the optimal policy of the objective as specified. See §5.3.

### 2.3 The reported configurations

| | EVAAA PPO | EVAAA PPO+LSTM | EVAAA DreamerV3-S |
|---|---|---|---|
| Framework | sheeprl | sheeprl (implied) | sheeprl, `dreamerv3S.yaml` |
| Discount $\gamma$ | 0.99 | **not reported** | 0.997 |
| $\lambda$ | 0.95 | not reported | 0.95 |
| Clip $\epsilon$ | 0.1 | not reported | — |
| Entropy coef. | 0.01 | not reported | 0.0003 |
| LR | 2.5e-4 | not reported | 1e-4 |
| Batch × unroll | 256 × 1024 | not reported | 16 × 64 |
| Visual input | 4-frame stack, 12ch 64×64 | not reported | single 64×64×3 frame |
| Actor grad | "Analytical (GAE)" | not reported | REINFORCE |
| Critic | scalar | not reported | 255-bin two-hot |

(Table 2, App. I.2–I.3.)

**Two red flags in the PPO description.**

1. **No hyperparameters at all are reported for PPO+LSTM.** It appears once, as Fig. 20 in Appendix J, with a two-sentence caption justifying its exclusion. For a benchmark paper whose contribution is the environment, that is entirely reasonable editorially — but it means the evidential weight of "recurrent PPO fails in EVAAA" as an *algorithmic* claim is low. See §5.4.
2. **The PPO actor is described as producing "the parameters of a diagonal Gaussian distribution over actions"** (App. I.2) — but EVAAA's action space is **5 discrete actions** (§4.2, Fig. 2f). Taken literally, that is a continuous-action head on a discrete-action environment. It is almost certainly boilerplate inherited from sheeprl's continuous-control template rather than the shipped config, but if it *is* the shipped config it is a first-order confound that would explain both PPO results by itself. **Checkable in the repo.**

---

## 3. Dimension-by-dimension comparison

Legend: **[PDF-]** = the EVAAA paper does not report enough to compare on this dimension.

| # | Dimension | EVAAA | Our grid world | Comparable? |
|---|---|---|---|---|
| 1 | Engine / substrate | Unity + ML-Agents, PyTorch | JAX, in-process | ✅ |
| 2 | World geometry | continuous 3D, 100×100 unit field, NavMesh navigation | discrete 2D, 10×10 cells | ✅ |
| 3 | Primary observation | **64×64×3 egocentric RGB, first-person, occluded by walls, modulated by fog + 5-phase day/night** | 27-d vector; `visual_sensor_range: 0` → the agent's own cell only | ✅ |
| 4 | Vector observation | 33-d = 4 EV + 10 olfaction + 9 thermal (3×3 grid) + 10 collision (100 rays → 10 sectors) | 27-d = 1 satiation + 1 intero-nocic + 1 extero-nocic + 5 olfaction + 5 collision + 6 proprio + 8 "vision" | ✅ |
| 5 | Is vision required? | **Yes** — 'No Vision' ablation fails to learn (Figs. 25–26) | **No** — there is no vision to speak of | ✅ |
| 6 | Distal sensing | olfaction = distance-weighted sum of fixed 10-d resource vectors within a sphere | olfaction = $\sum_e \mathbf p_e/d_e$ within radius 20 > grid diagonal 12.7 → **global** superposition | ✅ |
| 7 | Actions | 5 discrete (none, forward, turn L, turn R, eat) | 6 discrete | ✅ |
| 8 | Internal variables | 4: satiation, hydration, temperature, damage. Ranges: first three $[-15,15]$ setpoint 0; damage $[0,100]$ setpoint 0 | 2: nutrition $[0,100]$, injury $[0,100]$ | ✅ |
| 9 | Interoceptive observability | EVs directly observed (the 'No EV' ablation fails) | **`injury_observable: false`** — only a 3-step-lagged alpha-kernel proxy | ✅ (and we are strictly harder) |
| 10 | Reward form | **ambiguous** — §2 difference-of-squared-deviations vs App. D negative-norm-of-level | difference of the norm: $r_t = D(s_t)-D(s_{t+1})$, $D=\lVert (S,I)-(100,0)\rVert_2$ | ⚠️ inconsistent in source |
| 11 | Terminal penalty | **none stated**; termination = any EV out of range | **$-100$** on real death (not on the 500-step timeout) | ✅ — and this is the single biggest reward-side difference |
| 12 | Reward sign | non-positive throughout (under App. D) | mixed sign; positive when reducing drive | ✅ |
| 13 | Episode cap | testbeds 300–1000 steps. **[PDF-] training-level caps not reported** | 500 steps; empirical mean ≈120 | ⚠️ partial |
| 14 | Hidden per-episode dynamics | predator view angle / speed / state-transition limits are JSON-configured and fixed per level (Fig. 16) — **not redrawn per episode as far as reported**. **[PDF-]** | **Hidden-Parameter MDP**: predator count $\mathcal U\{0,1,2\}$, detect range $\mathcal U\{1..7\}$, damage $[15,120]$, all unobserved and redrawn each episode | ⚠️ we are harder; EVAAA side under-reported |
| 15 | Unforecastable hazards | predators are visible + raycast-detected; restricted to pre-defined nav zones "to prevent unavoidable deaths" (App. G Lv4-1) | 2–12 `hiding_predator` with **zero** olfactory signature, detectable only on contact | ✅ — we are much harder |
| 16 | Initial-state randomisation | **[PDF-]** not reported | $N_0, I_0 \sim \mathcal U[0,100]$ — some episodes unwinnable at $t{=}0$ | ❌ |
| 17 | Exploration demand | real: Lv3-2 relocates food to one random tree region per episode; Lv3-1 requires path planning around walls | essentially none; the helpful action is 1–5 steps away | ✅ — EVAAA is harder |
| 18 | Env-step budget per algorithm | **[PDF-] not reported.** Only "most environments requiring approximately 1.5 to 2 days to reach convergence" on **a single RTX 3090** (App. I.1) | rPPO 1.7–2.6 B; Dreamer ~13 M to plateau | ❌ **the critical missing number** |
| 19 | Environment throughput | **[PDF-] not reported.** No Unity time-scale, decision period, action repeat, or parallel-env count given | rPPO ~35,250 env-steps/s; Dreamer ~220–390 | ❌ **the second critical missing number** |
| 20 | Seeds | "standard error across random seeds" — **[PDF-] count not given** | 3–5 typical | ❌ |
| 21 | Model size | DreamerV3-**S** (RSSM: MLP 1029→512, LayerNorm-GRU 1024→1536) | DreamerV3 XS (3.17 M) ≈ M (20.7 M) at matched episodes | ✅ |
| 22 | rPPO discount | **[PDF-]** for PPO+LSTM; feedforward PPO used $\gamma=0.99$ | $\gamma = 0.95$, `return_mode: MC` | ⚠️ partial |
| 23 | Dreamer discount / horizon | $\gamma=0.997$, $\lambda=0.95$; **[PDF-]** imagination horizon and replay ratio not reported | $\gamma=0.997$, horizon 15, replay ratio 0.0625 | ⚠️ partial |
| 24 | rPPO encoder inductive bias | **[PDF-]** for PPO+LSTM; feedforward PPO used a generic CNN + 2-layer MLP | **hierarchical per-sensor MLPs** then a fusion hub (`encoding_mode: hierarchical`) — a strong hand-designed factorisation | ⚠️ we gave our rPPO a large advantage the EVAAA baseline did not have |
| 25 | Gradient steps per env-step | **[PDF-]** derivable only if the budget were reported | rPPO 1/4096; Dreamer 1/16 → **256×** | ❌ |

**Summary of what cannot be compared from the PDF:** the per-algorithm environment-step budget (#18), environment throughput (#19), seed count (#20), training-level episode caps (#13), initial-state randomisation (#16), and *every* hyperparameter of the recurrent PPO baseline (#22, #24). Dimensions #18 and #19 are exactly the two numbers needed to test hypothesis 1 directly, so that hypothesis has to be argued from architecture rather than measured.

---

## 4. Restating the puzzle correctly

The brief treats this as one inversion. It is two, and separating them is most of the work:

**Inversion A — "recurrent PPO learns here, does not learn there."** Real. Requires an algorithmic or task-structural explanation. Causes R1, R3, R4 below.

**Inversion B — "DreamerV3 wins there, loses here."** *Mostly does not exist.* From [[SYNTHESIS_20260727]]: at matched episodes DreamerV3 beats recurrent PPO **2.4×** in our grid world (123 vs 52 survival steps at 170k episodes) and is still improving while rPPO has plateaued. DreamerV3's sample-efficiency advantage is intact here. What flipped is the *wall-clock* ranking. Cause R2.

Keeping these apart immediately kills a tempting but wrong story ("world models only help on hard environments"), and it sharpens the real one.

---

## 5. Ranked causal account

Ranked by **explanatory share** (how much of the observed pattern each accounts for), not by confidence. Confidence is stated separately. Tags: **[ALGORITHMIC]** = a real property of task × algorithm; **[ARTIFACT]** = a property of how the comparison was measured.

### R1 — Supervision bandwidth into the encoder **[ALGORITHMIC]** · explains Inversion A · confidence ~85%

EVAAA's decisive information arrives as a $64\times64\times3 = 12{,}288$-dimensional image, and the paper's own ablation proves the task is unsolvable without it. Consider how much supervision each learner delivers to the encoder per environment step.

A model-free actor-critic trains its encoder $f_\theta$ only through the policy-gradient and value losses. Per transition, the total gradient signal reaching $f_\theta$ is

$$
\nabla_\theta \mathcal L \;=\; \underbrace{-\hat A_t\,\nabla_\theta \log \pi_\theta(a_t\mid f_\theta(o_t))}_{\text{one scalar } \hat A_t} \;+\; \underbrace{c_v\big(V_\theta(f_\theta(o_t)) - \hat R_t\big)\nabla_\theta V_\theta}_{\text{one scalar residual}} ,
$$

i.e. **$O(1)$ scalars of supervision per step**, and $\hat A_t$ is itself a high-variance Monte-Carlo estimate. DreamerV3 additionally trains its encoder through the world-model loss

$$
\mathcal L(\phi) = \mathbb E_{q_\phi}\Big[\sum_t \beta_{\text{pred}}\big(-\ln p_\phi(x_t\mid z_t,h_t) - \ln p_\phi(r_t\mid\cdot) - \ln p_\phi(c_t\mid\cdot)\big) + \beta_{\text{dyn}}\mathcal L_{\text{dyn}} + \beta_{\text{rep}}\mathcal L_{\text{rep}}\Big],
$$

whose reconstruction term $-\ln p_\phi(x_t\mid z_t,h_t)$ supplies **$O(12{,}288)$ targets per step**, every one of them low-variance. Under a data budget that Unity makes expensive (#18/#19), the model-free encoder is starved by roughly four orders of magnitude in supervision density. This is not speculation about EVAAA specifically — it is the mechanism the entire pixel-based self-supervised-RL literature was built to address (CURL, DrQ, SPR, BYOL-Explore all close most of the model-free/model-based gap on pixels *by adding an auxiliary representation loss*, without adding a world model).

Our grid world removes this entirely. The observation is 27 numbers, already factored by the designers into semantically meaningful blocks — and our rPPO is additionally given a **hierarchical per-sensor encoder** (dimension #24) that hard-codes that factorisation. There is no representation-learning problem left for a world model to solve.

**This is the strongest single real cause, and the paper's vision ablation is direct evidence for it.** It also refines the brief's hypothesis 2 from "representation learning vanished" to the sharper and more testable "supervision *bandwidth per environment step* collapsed from ~$10^4$ to ~$10^0$-and-it-didn't-matter."

### R2 — The scarce resource moved; and "model-based" is confounded with "more gradient steps" **[ARTIFACT]** · explains Inversion B · confidence ~85%

Two separate points, both artifacts, both large.

**(a) Which axis the comparison is drawn on.** Let $c_{\text{env}}$ be seconds per environment step and $c_{\text{grad}}$ seconds per gradient step, with $\rho$ gradient steps per environment step. Wall-clock per environment step is $c_{\text{env}} + \rho\,c_{\text{grad}}$. In our JAX grid world $c_{\text{env}} \approx 2.8\times10^{-5}\,$s (35,250 steps/s) — essentially free — so wall-clock is dominated entirely by $\rho\,c_{\text{grad}}$, and Dreamer's $\rho = 1/16$ versus rPPO's $\rho = 1/4096$ (**256×**) decides the race before any algorithmic property is consulted. In Unity with camera rendering, $c_{\text{env}}$ is milliseconds and *dominates*; then $c_{\text{env}} + \rho c_{\text{grad}} \approx c_{\text{env}}$ for both learners, wall-clock efficiency collapses onto sample efficiency, and Dreamer's extra compute per datum is very nearly free. **The brief's hypothesis 1 is correct and I would rank it as the dominant explanation of Inversion B.**

**(b) The deeper confound.** Our comparison does not isolate "has a world model." It compares a world-model agent at $\rho = 1/16$ against a model-free agent at $\rho = 1/4096$. Sample efficiency in deep RL is bought largely with **gradient steps per datum**, whether or not a model is present — this is the central finding of van Hasselt, Hessel & Aslanides (2019), *"When to use parametric models in reinforcement learning?"*, which showed that Rainbow with a raised replay ratio matches SimPLe's model-based sample efficiency, and of Schwarzer et al. (2023), *BBF*, which reached model-based-class Atari-100k efficiency with a purely model-free agent at replay ratio 8. **Until we match $\rho$, we cannot attribute any part of Dreamer's remaining 2.4× matched-episode edge to the world model.** This is the highest-value cheap experiment we can run (§7, E2).

### R3 — Terminal-state reward geometry: EVAAA charges nothing for dying **[ALGORITHMIC]** · explains Inversion A · confidence ~60% (limited by the §2/App. D ambiguity)

Suppose EVAAA's per-step reward is non-positive ($r_t \le 0$, App. D form) and termination carries no penalty. Then in the learner's arithmetic, death is an absorbing state of value exactly $0$, while every surviving trajectory has value $V^\pi(s) = \mathbb E[\sum \gamma^t r_t] \le 0$. **Self-termination is weakly optimal from every state**, and strictly optimal from any state where the agent expects future deviation from setpoint. This is the reason Gym's MuJoCo locomotion tasks add an explicit `healthy_reward` "alive bonus" — without it, the classic failure is an agent that dives to the ground immediately.

Now ask *which learner falls in.* At a terminal transition, PPO's TD residual is

$$
\delta_T = r_T + \gamma\,\underbrace{V(s_{T+1})}_{=\,0\ \text{at termination}} - V(s_T) = r_T - V(s_T),
$$

and with $V(s_T) < 0$ and $r_T \gtrsim V(s_T)$ this is **positive**. GAE propagates that positive residual backwards along the trajectory with weight $(\gamma\lambda)^{T-t}$. So the clipped surrogate

$$
L^{\text{CLIP}} = \mathbb E\Big[\min\big(\varrho_t \hat A_t,\ \mathrm{clip}(\varrho_t, 1{-}\epsilon, 1{+}\epsilon)\hat A_t\big)\Big], \quad \varrho_t = \tfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

*increases* the probability of the action sequence that ended the episode. On-policy PPO is a direct ascent on this quantity and will find the attractor within a few tens of updates. Entropy regularisation at coefficient 0.01 does not help, because this is not an exploration failure — it is a genuine optimum of the specified objective. And once the agent dies quickly, episodes shorten, which for a *recurrent* policy means the BPTT windows contain almost no long-horizon structure to learn from; the failure compounds. **That is a specific mechanism for why the recurrent variant fails worse than the feedforward one, which is exactly what Fig. 20 vs §8 shows.**

DreamerV3 is structurally shielded on three counts:
1. **Its actor is never trained on real terminal transitions.** It is trained on imagined rollouts of horizon $H$ from replayed latent states, where continuation is handled by a *soft* continue predictor $\hat c_t \in (0,1)$ inside $R^\lambda_t = r_t + \gamma \hat c_t\big[(1-\lambda)v_\psi(s_{t+1}) + \lambda R^\lambda_{t+1}\big]$. There is no hard $V=0$ bootstrap for the actor gradient to exploit.
2. **Return normalisation.** Dividing imagined returns by the percentile spread $S = \mathrm{Per}(R^\lambda,95) - \mathrm{Per}(R^\lambda,5)$ compresses the advantage of the suicide branch relative to the bulk.
3. **Replay.** The buffer keeps early, long, diverse episodes alive; the critic cannot collapse onto a suicide-consistent fixed point as fast as an on-policy critic that only ever sees the current (short) policy's data.

**The paper reports exactly the predicted fingerprint** — deliberate self-termination emerging at Level 3-2 and generalising to the Y-maze (§9, App. J.2). That the *best* algorithm still shows the pathology, while the weakest one is destroyed by it, is precisely the ordering this mechanism predicts.

Our grid world has the opposite geometry: a mixed-sign dense reward plus a **$-100$ terminal penalty gated on real death only**. Death is the most expensive thing that can happen, so the suicide attractor does not exist, and our rPPO can succeed even at $\gamma=0.95$.

**Confidence caveat.** If §2's difference form is what shipped, the argument weakens (returns telescope and are not uniformly negative) but does *not* vanish, because the absence of a terminal penalty is stated independently of the reward form. Resolving this from the repo would move confidence to ~85% or down to ~30%.

### R4 — Baseline-effort asymmetry **[ARTIFACT]** · explains Inversion A · confidence: ~70% that it contributes; ~25% that it explains everything

The EVAAA PPO+LSTM baseline is one figure in Appendix J with **zero reported hyperparameters** and no tuning narrative. Our recurrent PPO is the project's primary baseline, tuned over the whole diagnosis series, with a bespoke hierarchical per-sensor encoder, `return_mode: MC`, and a discount chosen for this task. On top of that, the *feedforward* PPO's actor is described as emitting a diagonal Gaussian over a discrete action space (§2.3, red flag 2) — if that description is faithful to the shipped config, both PPO results are explained by a configuration bug and nothing algorithmic is at stake.

This is not a criticism of the paper — a benchmark paper is not obliged to tune every baseline to exhaustion, and the paper is transparent that PPO+LSTM was excluded. It *is* a caution about how much weight the "recurrent PPO fails" datum can carry in a cross-paper algorithmic argument. **The cheapest explanation for a failed baseline is usually an untuned baseline, and this one is checkable in an afternoon from the EVAAA repo.**

### R5 — Horizon requirement, but asymmetrically **[ALGORITHMIC]** · confidence ~65%, with a direction correction

The brief's hypothesis 3 needs splitting, because it is only half right.

**The half that holds:** our rPPO succeeds at $\gamma = 0.95$, an effective horizon of $1/(1-\gamma) = 20$ steps. A death 40 steps away is discounted to $0.95^{40} \approx 0.13$ of its value. That an agent optimising a ~20-step surrogate reaches 177 survival steps is strong evidence that **our task is myopically solvable** — and if it is, DreamerV3's long-horizon imagination and $\gamma = 0.997$ (333-step horizon) are buying something the task does not need. This correctly predicts a *small* Dreamer advantage here.

**The half that fails:** it does not explain EVAAA. Their PPO ran at $\gamma = 0.99$ — a **longer** horizon than ours — and still failed. So "EVAAA needs a longer credit chain than PPO could support" is not supported by the reported configuration; if anything the horizon was set appropriately and something else broke.

**Net:** R5 explains why Dreamer's edge is small in our environment. It does **not** explain rPPO's failure in EVAAA. Rank the hypothesis accordingly.

### R6 — Aleatoric unforecastability penalising the world model **[ALGORITHMIC]** · confidence ~40% as a *ceiling limiter*, ~15% as a cause of the inversion

The brief's hypothesis 4 (our HiP-MDP + zero-smell ambushers make the decisive event structurally unforecastable) is a real effect, argued in detail at [[gridworld_vs_dreamerv3_benchmarks_difficulty]] §5.2. But it should be **down-ranked from its position in the brief**, for three reasons:

1. **The empirics contradict it as a cause of failure.** Our Dreamer's world-model loss plateaus (−0.7% over the last 8M steps) *while behaviour keeps improving monotonically*. If irreducible model error were the binding constraint, behaviour would plateau with it. It hasn't.
2. **Dreamer is winning at matched experience here (2.4×).** An effect that supposedly explains why Dreamer under-performs cannot be invoked while Dreamer out-performs on the experience axis.
3. **DreamerV3's latents are stochastic (categorical) by design.** Aleatoric noise can be absorbed into the latent distribution rather than appearing as model *bias*; that is precisely the argument for stochastic latents over deterministic ones. The MBPO-style compounding-error bound — return gap $\lesssim \tfrac{2 r_{\max}\gamma\,\epsilon_m}{(1-\gamma)^2}$ — punishes *systematic* model error, not well-calibrated stochasticity.

What survives is the narrower, sharper claim from the earlier memo: the **continue predictor** $\hat c_t$ regresses to a smooth base-rate hazard, so imagined rollouts under-represent the sharp state-contingent lethality of the real world — and with $H = 15 \ll 120$, the $-100$ almost never appears inside an imagined rollout at all. That caps Dreamer's ceiling. It does not cause an inversion. Keep it as a ceiling explanation, not a ranking explanation.

### R7 — Curriculum asymmetry **[REJECTED]** · confidence ~85% that it is *not* a confound

Rejected on the EVAAA side in §2.1 (Dreamer led with and without curriculum; PPO+LSTM failed on the two curriculum stages it was given). Independently rejected on our side: our prior three-stage size-curriculum experiment reported the *same* plateau with ~45% fewer episodes ([`DREAMER_SRL_3STAGE_SIZE_CURRICULUM.md`](../../experiments/active/continual_learning/DREAMER_SRL_3STAGE_SIZE_CURRICULUM.md)). **Recommendation: do not spend compute on a curriculum arm to close this gap.**

---

## 6. The ranking, condensed

| Rank | Cause | Tag | Explains | Confidence |
|---|---|---|---|---|
| R1 | Supervision bandwidth into the encoder (pixels vs 27-d hand-factored vector) | ALGORITHMIC | Inversion A | ~85% |
| R2 | Comparison axis moved (env-step cost) **+** model-based confounded with 256× update ratio | ARTIFACT | Inversion B | ~85% |
| R3 | No terminal penalty in EVAAA → suicide attractor that on-policy PPO ascends and Dreamer resists | ALGORITHMIC | Inversion A | ~60% |
| R4 | Baseline-effort asymmetry (untuned rPPO there, heavily tuned here; possible Gaussian-head bug) | ARTIFACT | Inversion A | ~70% contributes |
| R5 | Our task is myopically solvable (γ=0.95 suffices) → long-horizon imagination worth ~0 **here**; does *not* explain EVAAA | ALGORITHMIC | Dreamer's small edge here | ~65% |
| R6 | Aleatoric unforecastability of the decisive event | ALGORITHMIC | Dreamer's *ceiling* here, not the ranking | ~40% |
| R7 | Curriculum asymmetry | — | nothing | rejected, ~85% |

---

## 7. Falsifiable predictions and discriminating experiments, ranked by cost

All are runnable in **our** environment. Costs assume rPPO ≈ 1.2 h per 200 M-step run and Dreamer ≈ 1–2 days per run.

### E0 — Re-plot existing runs on a matched-**environment-step** axis · **cost: zero, no training**
*Tests R2.* We currently have matched-*episode* (2.4× Dreamer) and matched-*wall-clock* (rPPO wins) comparisons. Matched-env-step is the third and fairest axis, and Dreamer's shorter episodes mean it is *not* the same as matched-episode.
- **Predicts:** Dreamer ≥ rPPO across the whole matched-env-step curve. If so, "the inversion is an accounting artifact" is confirmed for Inversion B and should be stated that way in any paper.
- **Refuted if:** rPPO leads at matched env-steps, in which case R2 is wrong and Dreamer genuinely under-performs here.
- Offline from existing WandB history. **Do this before anything else.**

### E1 — rPPO death-penalty ablation: set the terminal penalty to 0 · **cost: 3 rPPO seeds ≈ 4 GPU-hours**
*Tests R3 — and does so by attempting to reproduce EVAAA's rPPO failure inside a 10×10 grid.*
- **Predicts:** with the $-100$ removed, our rPPO's survival collapses toward minimal, and (diagnostically) the fraction of episodes ending in early self-inflicted death rises sharply. Dreamer, run in the same condition, degrades far less.
- **Refuted if:** rPPO's survival is roughly unchanged — in which case the suicide-attractor mechanism is not operative and R3 drops out of the ranking.
- Strengthen by adding a second cell that also switches the dense reward to EVAAA's App. D level form $r_t = -\lVert (S,I) - (100,0)\rVert$, giving a full EVAAA-reward-geometry replication.
- **This is the highest-value-per-hour experiment in the list**, because a positive result is a genuine cross-environment causal claim: *the algorithm inversion is caused by the reward's terminal geometry, not by the algorithms.*

### E2 — Match gradient steps per datum · **cost: 3–4 rPPO seeds ≈ 5 GPU-hours**
*Tests R2(b) — the confound between "model-based" and "256× more updates per datum."*
- Raise rPPO's $\rho$ toward Dreamer's by **shrinking the rollout** (`num_envs × sequence_length`) at fixed `K_epochs`, which increases updates per environment step while leaving the trust region intact. Do **not** raise `K_epochs` alone — that increases off-policy drift and saturates the clip rather than adding effective updates.
- **Log as guards:** clip fraction, approximate KL, explained variance. If clip fraction exceeds ~0.3 the trust region has broken and the cell is uninterpretable.
- **Predicts:** if much of Dreamer's 2.4× matched-episode edge closes, then that edge was compute-per-datum, not the world model, and no paper should attribute it to model-based learning.
- The rigorous version is an off-policy model-free baseline (discrete SAC or R2D2-style) at Dreamer's replay ratio — a larger lift, worth it only if the cheap version is suggestive.

### E3 — rPPO discount sweep, $\gamma \in \{0.95, 0.99, 0.997\}$ · **cost: 3 configs × 3 seeds ≈ 11 GPU-hours**
*Tests R5.* Already pre-registered as P5 in [[gridworld_vs_dreamerv3_benchmarks_difficulty]].
- **Predicts:** if survival does **not** improve at $\gamma = 0.997$, our task is confirmed myopically solvable and DreamerV3's long-horizon machinery is worth ~0 here — which is the technical core of the "we don't need a world model" conclusion (§9).
- If survival **does** improve materially, our rPPO baseline was myopic and every published comparison against it is unfair to Dreamer in a second, independent way. **Run this before citing the rPPO number anywhere.**

### E4 — Destroy the observation's hand-designed factorisation · **cost: 3–4 rPPO seeds ≈ 5 GPU-hours**
*Cheap proxy for R1, without needing pixels.* Two cells: (a) `encoding_mode: flat`; (b) apply a **fixed random orthogonal mixing** $\tilde o = Q o$, $Q \in O(27)$, to the observation before the encoder — semantics-preserving in information content, but destroying the per-sensor block structure the hierarchical encoder exploits.
- **Predicts (R1):** rPPO degrades substantially under (b) while Dreamer — which trains its encoder through a reconstruction loss and is therefore indifferent to basis — degrades little. That is the pixel/vector asymmetry reproduced at 1/1000th the cost.
- **Refuted if:** both degrade equally, in which case the encoder-supervision story is not the mechanism and R1 should drop.

### E5 — Determinise the environment · **cost: config-only, but requires Dreamer runs ≈ 4–8 GPU-days**
*Tests R6.* Fixed predator count 1; collapse `detection_range` / `damage` / `attack_delay` to point masses; `hiding_predator` count 0; `injury_observable: true`. (These are P2/P3/P4 in the earlier memo.)
- **Predicts (R6):** Dreamer's plateau rises materially while rPPO's rises much less. If both rise equally, the aleatoric noise was hurting *both* learners and R6 loses its Dreamer-specific status.
- Expensive because it needs Dreamer arms. Run only after E0–E4.

### E6 — Pixel-observation variant · **cost: invasive code change + 4 Dreamer runs ≈ 8+ GPU-days**
*The direct test of R1.* Render the grid as a 10×10×C tensor (or a small RGB image) and give both agents only that, removing the 27-d vector.
- **Predicts:** Dreamer's relative advantage grows sharply; rPPO degrades far more than Dreamer. This is the most informative experiment for the two-paper story and the most expensive. Only run it if E4 is suggestive and the two-paper framing (§9) is being pursued for publication.

### Do not run
**A curriculum arm for Dreamer.** Refuted twice (§R7). The compute is better spent on E1 and E2.

---

## 8. Empirical signatures to watch for

Independent of the experiments above, these are logging-level tells that cost nothing:

- **R3 (suicide attractor):** in any zero-death-penalty cell, log the distribution of `EpisodeEndType`. A rising share of injury-death at low elapsed time, with *falling* mean episode length and *rising* mean per-step reward, is the signature. Mean per-step reward improving while survival collapses is diagnostic and impossible to explain any other way.
- **R2 (update-ratio confound):** plot survival against **gradient steps** rather than environment steps or wall-clock. If the two algorithms lie on a common curve in that coordinate, the world model contributes ~nothing beyond compute.
- **R6 (continue-predictor smoothing):** log $\mathbb E[1-\hat c_t]$ averaged over the $H=15$ imagination horizon against the empirical per-step death rate in the buffer. Systematic under-estimation confirms the ceiling mechanism (this is P7 in the earlier memo).
- **Dreamer actor-gradient spikes:** if pre-death states are ~1% of the imagined batch they fall outside the 5th percentile used for the return-scaling denominator $S$, so $-100$-scale returns enter the actor gradient nearly unattenuated. Expect spiky actor grad-norm correlated with death events in the batch.

---

## 9. "Does our environment need a world model?" — the honest answer

**No — not for survival performance on the current task.**

The case is straightforward and every clause is already established or cheaply testable. The observation is 27 hand-factored numbers, so there is no representation-learning problem (R1). The task is myopically solvable — a $\gamma = 0.95$ agent reaches 177 survival steps — so long-horizon imagination buys little (R5, pending E3). The decisive event is partly unforecastable by construction, so the world model's forward predictions are least reliable exactly where they matter (R6). And Dreamer's remaining matched-episode advantage is confounded with a 256× update-ratio difference that has nothing to do with world models (R2b, pending E2).

**What that does *not* mean.** It does not mean the Dreamer branch of this project is unjustified. It means the justification cannot be *survival steps*. The defensible reasons to keep a world model here are scientific, not performance-based:

1. It supplies an **explicit latent belief state** over the hidden per-episode threat parameters — which is the object the project's precision/modulator hypotheses are actually about (see [[PRECISION_MODULATION_ARCHITECTURE]]).
2. Its **decoder-side reconstruction loss** is a candidate substrate for sensory precision, which a model-free agent simply does not have.
3. Imagined rollouts permit **counterfactual probes** ("what would this agent have predicted if the predator were absent") that no model-free agent can answer.

Those are the grounds on which the Dreamer arm should be argued, and they should be argued explicitly rather than left to be inferred from a survival curve that will not support them.

### How the two papers should frame model-based vs model-free together

The most defensible joint claim is **not** "DreamerV3 is better" or "it depends on the task." It is a specific, mechanistic, two-axis statement, and the two environments are near-ideal anchors for it because they were built by the same lab around the same homeostatic premise:

> **Model-based RL's advantage is purchased, not intrinsic. It is bought with (i) high-dimensional observations that starve a policy-gradient encoder of supervision, and (ii) expensive environment steps that make extra gradient compute per datum effectively free. Remove both and a well-tuned recurrent PPO is Pareto-superior in wall-clock while remaining a modest 2.4× less sample-efficient — and even that residual is confounded with a 256× difference in updates per datum.**

Add the terminal-geometry axis (R3) and it becomes a genuinely novel methodological contribution rather than a re-derivation of van Hasselt et al. (2019):

> **Homeostatic reward designs with an all-negative per-step reward and a zero-cost absorbing death state create a suicide attractor. On-policy model-free methods ascend into it directly, because terminating produces a positive TD residual. Imagination-based methods resist it, because their actor is trained through a soft continue predictor rather than a hard terminal bootstrap. This is not a statement about representation learning, and it inverts the usual model-based/model-free ranking for a reason nobody has previously isolated.**

That framing is testable in a day (E1), it explains the EVAAA result and our result with one mechanism, and it turns an awkward "our results disagree with our own benchmark paper" into a controlled two-point study. **If E1 comes back positive, that is a paper.**

**One caveat to state in any such paper:** EVAAA's own emergent-behaviour section already documents self-termination as a learned strategy (§9, App. J.2), so this is not a criticism the benchmark is unaware of — it is a mechanism the benchmark surfaced and did not yet name.

---

## 10. Closest published precedents

| Precedent | Relation |
|---|---|
| **van Hasselt, Hessel & Aslanides (2019), "When to use parametric models in RL?"** | **The single closest precedent to this memo's question.** Shows Rainbow with a raised replay ratio matches SimPLe's model-based sample efficiency on Atari-100k — i.e. that the model-based advantage is partly a replay-ratio advantage. Directly underwrites R2(b). |
| **Schwarzer et al. (2023), BBF** | Model-free at replay ratio 8 reaching model-based-class Atari-100k efficiency. The modern confirmation of the same point. |
| **Wang et al. (2019), "Benchmarking Model-Based RL"** | Finds MBRL gains are strongly task-dependent and often vanish on proprioceptive control — the closest published statement of "on vector inputs the world model buys little." |
| **Schwarzer et al. (2021) SPR; Laskin et al. (2020) CURL; Yarats et al. (2021) DrQ-v2** | Auxiliary self-supervised losses close most of the model-free/model-based gap **on pixels**, without a world model. Underwrites R1: the gain is representational, not planning-based. |
| **Hafner et al. (2023), DreamerV3** | Its own suite makes the point: DMC-Proprio (~24-d vector) is solved with model **S** in 500 K steps and <1 GPU-day — the cheapest cell in the whole table. Our task is the proprio cell, not the vision cell. |
| **Janner et al. (2019), MBPO** | The compounding-model-error bound that R6 is calibrated against. |
| **Pardo et al. (2018), "Time Limits in RL"** | Correct handling of termination vs truncation in bootstrapping — the formal frame for R3's $\delta_T$ argument. Our env already gets this right (death penalty gated on real death, not the 500-step cap). |
| **Gym/MuJoCo `healthy_reward` (alive bonus)** | The standard engineering fix for exactly EVAAA's reward geometry; its existence is evidence the suicide attractor is a well-known failure and not a novel speculation. |
| **Keramati & Gutkin (2014); Yoshida (2017); Laurençon et al. (2021)** | Homeostatic RL — the shared theoretical home of both environments. Note that Keramati & Gutkin's formulation is the *drive-reduction* (difference) form, which is what EVAAA §2 cites and what our grid world implements; EVAAA's Appendix D form is not that. |

**Missing from `docs/project/references/`:** van Hasselt et al. (2019), Schwarzer et al. (2023) BBF, Wang et al. (2019) benchmarking-MBRL, Pardo et al. (2018). The first is the one a reviewer will name, and it should be fetched before any two-paper framing is drafted.

---

## 11. Two items for the EVAAA first author (repo-checkable, not inferable from the PDF)

1. **Which reward function shipped?** Main text §2 gives the drive-reduction difference form; Appendix D gives the negative-norm-of-level form. They are different functions with different optimal policies. This is erratum-level and it determines whether R3 is at ~85% or ~30% confidence.
2. **Is there any terminal penalty**, and **does PPO's actor really emit a diagonal Gaussian over the 5 discrete actions** (App. I.2)? Either answer materially changes the weight the "recurrent PPO fails in EVAAA" datum can carry.

Also worth reporting in any follow-up or camera-ready: the per-algorithm environment-step budget and the environment's steps-per-second (dimensions #18/#19). Without them, no reader can tell whether the DreamerV3 win is a sample-efficiency result or a wall-clock result — and this memo shows those can point in opposite directions in the same lab's other environment.

---

## 12. Cross-links

- [[SYNTHESIS_20260727]] — the five-perspective Dreamer investigation; source of the throughput, update-ratio, matched-episode, world-model-plateau, and capacity facts used throughout.
- [[gridworld_vs_dreamerv3_benchmarks_difficulty]] — the task-difficulty companion; §5.1 (potential-based shaping), §5.2 (continue-predictor smoothing, return-scaling percentile), §8 (probes P1–P8, of which E3/E5 here are P5/P2–P4).
- [[dreamer_srl_settings_regime_critique]] — hyperparameter-regime analysis; the source of the "M is over-sized, bins too coarse" prescriptions.
- [[dreamer_srl_h1_speed_investigation]] — where the wall-clock numbers come from, and the ~2× recoverable headroom that would partly (not wholly) offset R2(a).
- [[DREAMER_SRL_3STAGE_SIZE_CURRICULUM]] — the prior curriculum experiment that independently refutes R7 on our side.
- [[PRECISION_MODULATION_ARCHITECTURE]] — the canonical project notation, and the home of the non-performance justifications for the Dreamer branch given in §9.

---

## 13. Next steps

**→ `experiment-analyzer`** — **E0 first, it is free.** Re-plot the existing dsrl / rppo runs on a matched-environment-step x-axis, and additionally on a *gradient-step* x-axis (§8, R2 signature). This alone determines whether Inversion B is real or an artifact, and it needs no training.

**→ `experiment-designer`** — **E1 (death-penalty → 0, rPPO, 3 seeds) is the highest-value-per-hour experiment and should be pre-registered.** E2 (matched updates-per-datum via shortened rollouts, with clip-fraction and approx-KL guards) and E4 (flat encoder + random orthogonal observation mixing) are each 3–4 rPPO cells and can ride in the same sweep. E3 (γ sweep) is already pre-registered as P5. **Do not schedule a curriculum arm** (§R7).

**→ `senior-developer`** — E1 needs the terminal death penalty to be config-exposed as a scalar (check whether it already is; if not, this is a one-key change, not invasive). E4(b) needs a fixed random orthogonal mixing applied to the observation vector at the encoder boundary — additive and config-gated, no change to the 27-d layout or the dimension fingerprint. E6 (pixel observations) is genuinely invasive and should not be scoped until E4 reports.

**→ `pi`** — §9 is a portfolio-level call: the honest conclusion is that our environment does not need a world model *for performance*, and the Dreamer branch should be re-justified on the three scientific grounds in §9 or wound down. §9's two-axis framing is also a candidate publication direction in its own right, contingent on E1.

**→ `professor-dl-theory`** — R1 is stated here as an RL-side supervision-bandwidth argument. The architectural-identifiability version ("what does a reconstruction loss give the encoder that a policy gradient cannot, on a $d$-dimensional input") is theirs, and it is the part that would need to be rigorous for E6 to be worth its cost.

**→ `literature-reviewer` / `academic-pdf-fetch`** — van Hasselt, Hessel & Aslanides (2019); Schwarzer et al. (2023) BBF; Wang et al. (2019); Pardo et al. (2018). The first is required reading before the §9 framing is drafted.

---

*Memo by professor-rl — 2026-07-28. Scope: RL-algorithmic comparison only. EVAAA facts verified page-by-page against the local PDF (§2, §4.2–4.3, §7, §8, §9, App. C, D, E.1–E.5, G, I.1–I.3 incl. Table 2, J.1–J.2 incl. Figs. 20, 25, 26); every claim the PDF does not support is marked **[PDF-]** in §3. Grid-world facts taken from [[SYNTHESIS_20260727]], [[gridworld_vs_dreamerv3_benchmarks_difficulty]], and `configs/models/recurrent_ppo/recurrent_ppo_M.yaml` / `configs/models/dreamer_srl/01_food_only_M.yaml` (read-only). No code, config, or script was modified; no lab node was touched.*
