# Is our grid world harder or simpler than DreamerV3's original benchmarks?

> **One line:** Our task is *different in kind* — far simpler than the DreamerV3 suite in everything that makes benchmarks hard (exploration depth, action space, planning horizon, input dimensionality), and harder than everything in the suite except Minecraft in the two things the suite barely tests (irreducible per-step lethal noise, and a genuinely deep partial-observability of the agent's own death boundary). The verdict is *provisional* because nobody has measured the task's ceiling.

---

## 1. Question, in plain language

**The question.** Our agent lives in a 10×10 grid: it gets hungry, it can be hurt by predators and rocks, and it dies if either its food reserve hits zero or its accumulated injury hits the maximum. We measure how many time-steps it survives before dying. Two learners have been tried on it — a model-free recurrent PPO agent, and DreamerV3 (a "world-model" agent that learns an internal simulator and trains its policy inside it). Both take a *lot* of experience to get good, and neither reaches the 500-step episode cap. So: **is our little grid world actually a hard problem, or is it an easy problem that we are attacking badly?**

**Why it matters.** DreamerV3 was published with a fixed set of hyperparameters that worked across seven benchmark families — robot-control tasks, Atari games, a survival game called Crafter, 3D maze tasks, and Minecraft. If our world sits inside the region those settings were tuned for, then slow learning is our bug. If our world sits *outside* it, then slow learning is a property of the task and the right response is to change the algorithm, not the knobs.

**What this memo does.** It argues both sides at full strength, dimension by dimension, using our actual environment code and configuration files (not the description in the brief) and the actual DreamerV3 paper (task list, data budgets, benchmark table). Then it gives a verdict, states the confidence honestly, and names the concrete experiment that would settle it.

**Headline.** The strongest simplicity argument is that the *useful* state of our world is tiny — a handful of numbers — and the agent has only six buttons to press; nothing in it requires the multi-step tool-crafting reasoning that makes Minecraft and Crafter hard. The strongest difficulty argument is that our agent is effectively **blind beyond its own square**, its **injury level is hidden from it**, a substantial fraction of the grid is seeded with **hazards it cannot detect until it steps on them**, and its **starting health is randomised all the way down to nearly-dead**. That combination — cheap to describe, expensive to *infer* — is not represented anywhere in DreamerV3's benchmark suite.

**Verdict, one sentence.** *Different in kind, probably harder for a world-model agent specifically, and we cannot yet say "harder than the suite" as a bare claim because the achievable ceiling of our task has never been measured.* (Confidence: see §7.)

---

## 2. What our environment actually is (verified from code)

Verified against `src/environment/sensor.py`, `src/environment/core.py`, `configs/environment/default.yaml`, and `configs/environment/experiment/basic/03-random_init_10x10.yaml` / `04-jump_attack_10x10.yaml`. Several properties in the briefing understate the situation.

**Observation — 27 dimensions, and much narrower than "27-dim sensor vector" suggests.**

| Block | Dims | What it actually contains |
|---|---|---|
| Satiation | 1 | normalised food reserve — *observable* |
| Interoceptive nociception | 1 | injury history convolved with an alpha kernel, $\tau=3$, peak at 3 steps, $k[0]=0$ — a **delayed, smoothed proxy** for injury |
| Extero nociception | 1 | phasic contact pain this step |
| Olfaction | 5 | $\sum_e \mathbf{p}_e / d_e^{1.0}$ over **all active entities within radius 20** |
| Collision | 5 | Manhattan diamond, range 1 — fires only on grid edges (rocks are `blocking: false`) |
| Proprioception | 6 | one-hot of previous action |
| Vision | 8 | `visual_sensor_range: 0` → **the agent's own cell only** |

Three consequences the brief did not state:

1. **`visual_sensor_range: 0`.** The "8-channel vision" is a readout of the single cell the agent is standing on. The agent has *no* spatially resolved exteroception at all. Its only distal sense is olfaction.
2. **Olfaction radius 20 exceeds the grid diagonal ($\sqrt{2}\cdot 9 \approx 12.7$).** So every active entity contributes to every reading — the olfactory vector is a **global superposition**, not a local field. Recovering "where is the predator" from it is an inverse problem over a 5-dimensional aggregate.
3. **`injury_observable: false` and `nutrition_observable: false`.** The agent never sees how close it is to injury-death. It sees only the alpha-kernel-smoothed proxy, which lags by ~3 steps and integrates history.

**Sensory aliasing is designed in, and it is severe.**

- Predator olfactory signature $\mathbf p_{\text{pred}} \sim \mathcal N([0,\,0.7,\,0.5,\,0,\,0],\,0.3)$; rabbit (harmless) $\mathbf p_{\text{rab}} \sim \mathcal N([0,\,0.5,\,0.7,\,0,\,0],\,0.3)$, **redrawn per episode**. With $\sigma = 0.3$ against a mean separation of $0.2$ per channel, predator-vs-rabbit identification from smell is Bayes-limited even before the $1/d$ distance confound and the summation over up to 4 animals.
- Rock (`damage: [1,5]`, `nociception 0.9`, **zero smell**) and bush (harmless, hides the agent, smell channel 3) share the *same* visual channel 6. At vision range 0 they are distinguishable only after the agent has already stepped on one.
- `hiding_predator`: 2–12 per episode, **zero olfactory signature**, visible only in the vision channel of the cell the agent occupies, `damage: [15,45]`. On a 100-cell grid, up to 12% of cells are undetectable-until-contact hazards.

**Hidden per-episode dynamics parameters.** In basic-03/04 the predator's `detection_range` is drawn from $\{1,\dots,7\}$, `max_stamina` from $[30,150]$, `attack_delay` from $\{1,2,3\}$, `damage` from $[15,120]$, and in basic-04 `attack_range` from $\{2,3\}$ with `attack_success_rate: 0.5`. **None of these are observable.** The number of predators is drawn $K \sim \mathcal U\{0,1,2\}$, so ~33% of episodes contain no hunter at all. This makes the task a **Hidden-Parameter MDP** (Doshi-Velez & Konidaris 2016): the agent must perform online system identification of the current episode's threat model from within the episode.

**Initial condition randomised across the whole survivable range.** `random_start_nutrition: true`, `start_nutrition ~ U[0,100]`; `random_start_injury: true`, `start_injury ~ U[0,100]`. Nutrition decays 1/step, so an episode drawn at nutrition 4 has at most 4 steps to reach and eat food. Injury at 90 means a single predator contact ($15$–$120$ damage) is lethal. **A non-trivial fraction of episodes is unwinnable at $t=0$, and the agent cannot see which ones.**

**Reward.** With `use_homeostatic_reward: true` and drive $D(s) = \lVert (S,I) - (100,0)\rVert_2$:

$$r_t \;=\; D(s_t) - D(s_{t+1}) \;-\; 100\cdot\mathbb 1[\text{real death at } t{+}1]$$

Death penalty is gated on *real* death only (starvation / injury), never on the 500-step timeout — surviving to the cap is the success outcome and is not punished (`core.py:696`, `core.py:734`).

**Food is effectively infinite.** `max_consumption: 12`, `regeneration_delay: 0` → a depleted food cell respawns in place the next step. Net nutrition per eat is $6-1 = 5$; metabolic cost is 1/step. A policy that camps on food and eats once every five steps is **indefinitely sustainable**. The 500-step cap is therefore reachable in principle; the binding constraints are predators and the randomised start.

**Rest heals fast.** Recovery $= 0.1 \cdot 1.5^{(\text{streak}-1)}$ — ten consecutive rests heal at $\approx 3.8$/step. So injury is recoverable if the agent can find a safe interval.

---

## 3. What the DreamerV3 suite actually is (verified from the paper)

From Table A.1 of Hafner et al. 2023, plus §"Benchmarks" (p. 9):

| Benchmark | Tasks | Env steps | Action repeat | Model | GPU-days | Character |
|---|---|---|---|---|---|---|
| DMC Proprio | 18 | 500 K | 2 | S | <1 | ~20–24-d **full** state, deterministic, dense reward, no death |
| DMC Vision | 20 | 1 M | 2 | S | <1 | 64×64×3, deterministic, dense reward, no death |
| Crafter | 1 | 1 M | 1 | XL | 2 | procgen survival, health/food/drink/energy, **death**, 22 achievements |
| BSuite | 23 (468 cfgs) | — | 1 | XL | <1 | toy unit-tests: memory, credit assignment, reward scale, stochasticity, exploration |
| Atari 100 K | 26 | 400 K | 4 | S | <1 | near-deterministic, dense score deltas, full-frame observation |
| Atari 200 M | 55 | 200 M | 4 | XL | 16 | sticky actions ($p=0.25$) as the only stochasticity |
| DMLab | 8 | 50 M | 4 | XL | 4 | 3D first-person, spatial+temporal reasoning |
| Minecraft Diamond | 1 | 100 M | 1 | XL | 17 | procgen 3D open world, 12 sparse milestones, first diamond at 29 M steps, 24/40 seeds ever succeed |

The paper is explicit that the suite is chosen to span "continuous and discrete actions, visual and low-dimensional inputs, dense and sparse rewards, different reward scales, 2D and 3D worlds, and procedural generation" (p. 9). It says nothing about **irreversible catastrophe avoidance** or **hidden episode-level dynamics parameters**, and the only environments in the suite that terminate on agent death are Crafter, Minecraft, and (trivially) some Atari games where losing a life is a score event rather than an episodic catastrophe with a large negative reward.

---

## 4. Dimension-by-dimension comparison

| # | Dimension | Ours | Closest suite member | Simpler / Harder | Confidence |
|---|---|---|---|---|---|
| 1 | Input dimensionality | 27-d vector, MLP encoder/decoder | DMC Proprio (~24-d) | **Simpler** — representation learning is trivial; no CNN, no pixel reconstruction | High |
| 1b | Input *informativeness* | vision = own cell; smell = global 5-d superposition; injury hidden | Atari (full frame, near-Markov with frame stack); DMC Proprio (full state) | **Harder** — the observation is a lossy, aliased, non-invertible projection of the state | High |
| 2a | Reward density | non-zero essentially every step | Atari dense score deltas; Crafter ±0.1 health | **Simpler on paper** | High |
| 2b | Reward *informativeness* | dense term is $\approx$ potential-based shaping (see §5.1) → nearly policy-invariant; true objective is the terminal $-100$ | Minecraft (12 sparse milestones) | **Harder** — the density is a decoy; the task is effectively sparse-catastrophe | Medium-high |
| 2c | Rare-catastrophe structure | $-100$ once per episode ($\sim$1 per 120 steps), $50$–$200\times$ the per-step reward magnitude | **Nothing in the suite.** Crafter is the nearest (death terminates, health loss is $-0.1$/point) but its benchmark score is achievement-based and does not reward survival time | **Different in kind** | High |
| 3a | Stochasticity — dynamics | predator tie-breaks, patrol jitter, 50% pounce success, per-episode entity counts and positions | Atari sticky actions ($p=0.25$); Crafter procgen | **Harder** | High |
| 3b | Stochasticity — *hidden episode parameters* | detect range $\mathcal U\{1..7\}$, stamina $[30,150]$, damage $[15,120]$, predator count $\mathcal U\{0,1,2\}$, all unobserved | **Nothing in the suite** except Minecraft biome/terrain variation | **Different in kind** — HiP-MDP requiring in-episode system ID | High |
| 3c | Irreducible aleatoric reward | 2–12 invisible hiding predators; contact damage is unpredictable from any observation history | BSuite `bandit_noise` (toy) | **Harder** — the reward predictor has a non-zero loss floor by construction | High |
| 4 | POMDP depth | belief over predator positions must be maintained from a 5-d aggregate; injury boundary never observed directly | DMLab (3D, occlusion); BSuite `memory_len` (toy) | **Harder** than Atari/DMC, **comparable to** DMLab, **simpler than** Minecraft | Medium |
| 5 | Multi-objective / homeostatic | coupled Euclidean drive $\lVert(S,I)-(100,0)\rVert$; the marginal value of eating depends on current injury | Crafter (food/drink/energy/health, but *additive*) | **Different in kind** — coupling is multiplicative-in-effect, not additive | Medium |
| 6 | Action space | 6 discrete | Atari 18, Minecraft 25, DMC continuous | **Simpler** | High |
| 7 | Episode length / planning horizon | ≤500 steps; useful plan depth ~5–20 steps (reach food, dodge, hide) | Crafter ~10 K steps; Minecraft ~36 K steps with a 12-deep tech tree | **Much simpler** | High |
| 8 | Exploration difficulty | none — reward is immediate and the action that helps is always within 1–5 steps | Minecraft (first diamond at 29 M steps, 24/40 seeds), Crafter (22-achievement tree), BSuite `deep_sea` | **Much simpler** | High |
| 9 | Compositional structure | none — no crafting, no tools, no subgoal chain | Minecraft, Crafter | **Much simpler** | High |
| 10 | Data budget to plateau | Dreamer $\sim$5.8 M steps → 113 survival; 13 M → 117 plateau. rPPO 1.7 B → 182 | Atari-100k 400 K; DMC 0.5–1 M; Crafter 1 M; Minecraft 100 M | **Harder by budget** (13–32× Crafter/Atari-100k), *but see §5.3* | Low — confounded |
| 11 | Model capacity needed | task-relevant state is $O(10)$ numbers | Crafter/Minecraft use the XL (200 M-param) model | **Simpler** — and possibly *mismatched*: an over-parameterised world model on a high-noise, low-complexity task will spend capacity fitting noise | Medium |

---

## 5. The three technical arguments that actually carry the verdict

### 5.1 The dense homeostatic reward is (almost) policy-invariant shaping

Let $\Phi(s) = -D(s)$ with $D(s)=\lVert (S,I)-(100,0)\rVert_2$. Our per-step reward is $r_t = \Phi(s_{t+1}) - \Phi(s_t) - 100\,\mathbb 1[\text{death}]$. Expanding the discounted return over an episode of length $T$:

$$
G_\gamma \;=\; \sum_{t=0}^{T-1}\gamma^t r_t
\;=\; \underbrace{D(s_0)}_{\text{policy-independent}} \;-\; (1-\gamma)\!\!\sum_{t=1}^{T-1}\!\gamma^{t-1} D(s_t) \;-\; \gamma^{T-1}\Big[D(s_T) + 100\,\mathbb 1[\text{death}]\Big].
$$

Ng, Harada & Russell (1999) tell us the exactly-potential-based form $\gamma\Phi(s') - \Phi(s)$ leaves the optimal policy unchanged. Ours differs from that form by exactly $(1-\gamma)\Phi(s')$, which is the middle term above. **So the entire "dense" reward stream contributes to the objective only through a term of weight $(1-\gamma)$, plus the terminal bracket.** Two regimes follow:

- **Dreamer, $\gamma = 0.997$** (`configs/models/dreamer_srl/*.yaml`, `gamma: 0.996996...`). $(1-\gamma)\sum \gamma^{t-1}D_t \approx 0.003 \times 120 \times \bar D \approx 14$ for $\bar D\approx 40$; the death term is $100\gamma^{119}\approx 70$. **Death dominates ≈5:1.** Dreamer is solving a sparse-catastrophe-avoidance problem wearing a dense-reward costume.
- **rPPO, $\gamma = 0.95$** (`configs/models/recurrent_ppo/recurrent_ppo_M.yaml:6`, `gae_lambda: 0.95`). Effective horizon $1/(1-\gamma) = 20$ steps. The death term is $100\times 0.95^{k}$ for death $k$ steps away: $0.36$ at $k=20$, $0.13$ at $k=40$, $0.046$ at $k=60$. **A death more than ~40 steps in the future is, in the return, worth less than four ordinary steps of homeostatic drift.** Meanwhile the homeostatic term carries weight $(1-\gamma)=0.05$ over a 20-step window $\approx \bar D \approx 40$.

**This is a first-order finding for H3.** At $\gamma=0.95$, rPPO is not solving "survive"; it is solving "greedily reduce homeostatic drive over the next ~20 steps, with a mild aversion to imminent death." That it took 1.7–2.6 B env steps to reach 182 survival is therefore **weak evidence about task difficulty** and **strong evidence about objective mis-specification**. It is exactly the shape of result you get when the surrogate objective's optimum is not the metric's optimum: the learner converges fast to the surrogate optimum, then grinds.

### 5.2 An irreducible aleatoric floor sits inside the world model's loss

DreamerV3's world model minimises
$$
\mathcal L(\phi)=\mathbb E_{q_\phi}\Big[\textstyle\sum_t \beta_{\text{pred}}\mathcal L_{\text{pred}} + \beta_{\text{dyn}}\mathcal L_{\text{dyn}} + \beta_{\text{rep}}\mathcal L_{\text{rep}}\Big],\qquad
\mathcal L_{\text{pred}} = -\ln p_\phi(x_t\mid z_t,h_t) - \ln p_\phi(r_t\mid \cdot) - \ln p_\phi(c_t\mid \cdot).
$$

In our environment, three components of $-\ln p_\phi(r_t\mid\cdot)$ and $-\ln p_\phi(c_t\mid\cdot)$ are **unlearnable in principle**:

1. Hiding-predator contact (2–12 per episode, no olfactory or distal visual signature) — no function of the observation history predicts it better than the base rate.
2. Per-episode predator parameters (detection range, damage, pounce success) — identifiable only *after* enough interaction, and 33% of episodes have no predator to identify.
3. The injury-death boundary — the state variable that determines termination is hidden, and the observed proxy is a 3-step-lagged alpha-kernel convolution.

The consequence is **not** that the loss is high; it is that the *continue predictor* $\hat c_t$ regresses to a smooth base-rate hazard. Imagined rollouts of horizon $H=15$ (`horizon: 15`) then systematically **under-represent** the sharp, state-contingent lethality of the real world, and the imagined $\lambda$-return

$$
R^\lambda_t = r_t + \gamma c_t\big[(1-\lambda)v_\psi(s_{t+1}) + \lambda R^\lambda_{t+1}\big],\qquad R^\lambda_H = v_\psi(s_H)
$$

inherits that smoothing through $c_t$. Since $H=15 \ll$ the ~120-step episode, the $-100$ almost never appears *inside* an imagined rollout; it reaches the actor only via the bootstrap $v_\psi(s_H)$, i.e. through many rounds of critic self-consistency. **A hazard that is (a) rare, (b) partly unpredictable, and (c) mostly outside the imagination horizon is the worst case for imagination-based credit assignment.** This is a genuine "harder for Dreamer specifically" claim and it is testable (§8, P7).

A second, separate concern: DreamerV3's actor scales returns by $S=\mathrm{Per}(R^\lambda,95)-\mathrm{Per}(R^\lambda,5)$ with denominator $\max(1,S)$. If pre-death states are ~1% of the imagined batch, they sit *outside* the 5th percentile, so $S$ measures the harmless bulk; the surviving $-100$-scale returns then enter the actor gradient essentially unattenuated. Expect **spiky actor gradient norms correlated with death events in the imagination batch** — a concrete empirical signature.

By contrast, the symlog/two-hot machinery handles our reward scale *fine*: $\mathrm{symlog}(100)=\ln 101 = 4.615$, comfortably inside the $[-20,20]$ bin range, with 255 bins giving ~8% relative resolution near $-100$ and ~0.17 absolute resolution near the $\pm 1$ bulk. **Reward-scale robustness is not our problem.** That defuses one of the more obvious "our task is out of distribution for DreamerV3" claims.

### 5.3 The budget comparison is uninterpretable without a ceiling

Our Dreamer plateaus at 117 (basic-03) and ~142 (basic-04) survival steps against a 500-step cap; rPPO at 182 and 169 (see [[dreamer_srl_vs_rppo_survival_speed]], §4.1). At the **Atari-100k budget of 400 K env steps**, our Dreamer had not yet reached 33 survival steps — roughly 6.6% of the cap, after ~3,300 episodes. At the **Crafter budget of 1 M steps**, it is in the 50–60 range. It needs ~13 M — **13× Crafter, 32× Atari-100k, 26× DMC-proprio** — to reach its plateau.

That looks damning until you ask: *plateau at what fraction of achievable?* Nobody knows. Two readings are consistent with the same curves:

- **Reading A (task is easy, ceiling is low).** The randomised initial body state ($N_0, I_0 \sim \mathcal U[0,100]$) plus 2–12 undetectable hiding predators impose an aleatoric floor. If the achievable mean survival under an *omniscient* policy is ~200, then rPPO at 182 is at 91% of optimal and Dreamer at 117 is at 59%. The task is near-saturated and the whole difficulty story is "irreducible noise", not "hard learning".
- **Reading B (task is hard, ceiling is high).** Food is infinite and regenerates in place; rest heals exponentially; bushes block predator detection *and* pounces. An omniscient camper should reach ~400–450. Then rPPO's 182 leaves a **250-step learnable gap** and the task is genuinely, deeply unsolved.

**These two readings are separated by one cheap experiment (§8, P1).** Until it is run, "harder than the suite" is not a defensible claim, and neither is "simpler".

---

## 6. The two cases, stated at full strength

### 6.1 Strongest honest case for "SIMPLER than the suite"

1. **No exploration problem whatsoever.** Minecraft's difficulty is a 12-deep sparse-milestone chain where the first diamond appears at 29 M steps and 16 of 40 seeds never find one. Crafter's is a 22-achievement tech tree. Our agent gets a non-zero reward every single step and the action that helps is never more than a few steps away. On the axis that dominates the suite's hardest members, **we are trivially easy.**
2. **The task-relevant state is minute.** Position relative to nearest food, position relative to nearest threat, satiation, injury. That is $O(10)$ numbers. DreamerV3 uses a 200 M-parameter XL model for Crafter and Minecraft; our task should be within reach of a model two orders of magnitude smaller. An RSSM with 1024 GRU units on a 27-d MLP-encoded input is *enormously* over-provisioned.
3. **Six actions, ≤500 steps, 2D, discrete, fully synthetic dynamics.** Movement is deterministic. There is no physics, no contact dynamics, no 3D geometry, no visual generalisation problem.
4. **DMC-Proprio precedent.** DreamerV3 achieves state-of-the-art on 18 continuous-control tasks with ~24-dimensional proprioceptive input in **500 K steps**. That is the suite member closest to our input format, and it is the *cheapest* one in the whole table (<1 GPU-day, model size S).
5. **The compositional depth is zero.** Nothing in our task requires holding a subgoal across more than ~20 steps. Crafter requires "collect wood → craft table → craft pickaxe → mine stone → …". We require "walk to smell maximum, press eat".
6. **The performance numbers are consistent with a low ceiling.** Both learners plateau within 25% of each other (117–182 out of 500) despite a 130× difference in data consumed. Two very different algorithms converging to a similar band is what you expect when the band is set by the *environment's* noise floor, not by either algorithm's capacity.

### 6.2 Strongest honest case for "HARDER than the suite / different in kind"

1. **The agent is blind.** `visual_sensor_range: 0` means the only spatially-resolved sense is a 5-cell wall detector. Everything distal arrives as a 5-dimensional *sum* $\sum_e \mathbf p_e/d_e$ over up to ~40 entities. Atari gives the agent a complete, unambiguous frame; DMC-proprio gives it the exact state. **We give it a scalar potential field and ask it to invert a many-to-one map.** No suite member does this.
2. **The lethal boundary is unobservable.** `injury_observable: false`. The variable whose crossing terminates the episode with a $-100$ is never shown; the agent sees a 3-step-lagged, kernel-smoothed proxy. Compare: Crafter shows health in the HUD as rendered pixels; Atari shows lives on screen; DMC has no death.
3. **Up to 12% of the grid is an undetectable trap.** `hiding_predator`, count 2–12, zero smell, visible only from the cell you are standing on, damage 15–45. This is an **irreducible aleatoric reward**, and it is a designed property of the world, not an accident. Nothing in the suite has an unlearnable per-step lethal hazard.
4. **The episode's dynamics parameters are hidden and redrawn every episode.** Detection range $\mathcal U\{1..7\}$ alone changes the safe standoff distance by 7×. Predator count $\mathcal U\{0,1,2\}$ means the RSSM cannot answer "will there be a predator this episode" until it has evidence, and 33% of episodes have no evidence to find. This is a **Hidden-Parameter MDP** requiring in-episode system identification — a capability the suite tests only in BSuite's toy `memory_*` tasks.
5. **The starting state is drawn across the full survivable range, including unsurvivable draws.** $N_0, I_0 \sim \mathcal U[0,100]$. There is no benchmark in the DreamerV3 suite that randomises the agent's initial distance-to-death, let alone hides it.
6. **The dense reward is a decoy (§5.1).** The genuinely policy-relevant signal is the terminal $-100$, occurring once per ~120 steps. In effect this is a **sparse-catastrophe** task, i.e. the *safe-RL* problem family (Safety Gym, AI Safety Gridworlds), which DreamerV3 was never evaluated on.
7. **The multi-objective structure is coupled, not additive.** $D=\lVert(S,I)-(100,0)\rVert_2$ means $\partial D/\partial S$ depends on $I$ and vice versa. The value of eating literally changes with how injured you are. Crafter's health/food/drink/energy are additive; ours are on a shared 2-D manifold with a Euclidean metric. This is the closest thing to a genuine *homeostatic* objective in the RL literature (Keramati & Gutkin 2014; Laurençon et al. 2021), and it is absent from the suite.
8. **The empirical budget datum, taken at face value.** 13 M steps to plateau is 13× Crafter's entire budget; rPPO's 1.7–2.6 B is 8.5–13× the Atari-200M budget and 17–26× the Minecraft budget. If the ceiling turns out to be high (Reading B), this is straightforwardly a harder task than everything in the suite except DMLab-at-IMPALA-scale.

---

## 7. Verdict

**Different in kind, with a directional lean toward "harder for a world-model agent, easier for anything that just needs to explore."**

The DreamerV3 suite's hard members are hard for **epistemic** reasons: you must *discover* a deep chain of behaviours before any reward arrives, and you must build rich representations of high-dimensional inputs. Our task is hard for **aleatoric and inferential** reasons: the reward arrives constantly but is nearly uninformative, the truly decisive event is rare and partly unpredictable, and the state you need to condition on is systematically hidden behind a lossy aggregate sensor. **These load on different algorithmic machinery.** Better exploration and bigger encoders — DreamerV3's strengths, and the axes along which its suite discriminates — buy you almost nothing here. Better belief-state inference, risk-sensitive value estimation, and longer-horizon credit assignment for rare catastrophes would.

**Confidence.**

| Claim | Confidence |
|---|---|
| "Simpler than the suite on exploration depth, action space, planning horizon, compositional structure" | **High (~90%)** |
| "Harder than the suite on partial observability of the termination-relevant state, hidden per-episode dynamics, and irreducible lethal aleatoric noise" | **High (~85%)** |
| "Different in kind — the suite does not contain a task of this type" | **Moderate-high (~70%)**; Crafter is the honest counter-example and it is closer than one would like (procgen + hunger/thirst/health drives + death + partial view) |
| "Harder *overall* than the suite" | **Low (~35%)** — unmeasurable without the ceiling (§5.3) |
| "The rPPO 1.7–2.6 B datum is evidence of task difficulty" | **Low (~25%)** — at $\gamma=0.95$ that agent is optimising a 20-step-horizon surrogate in which death is nearly invisible (§5.1). It is better evidence of objective mis-specification. |
| "Our task is out of DreamerV3's reward-scale robustness envelope" | **Very low (~10%)** — symlog/two-hot handles $-100$ against a $\pm1$ bulk comfortably (§5.2) |

**The honest one-paragraph summary for a paper.** *Our environment occupies a corner of the difficulty space that standard model-based RL benchmarks under-sample: low state complexity and no exploration challenge, combined with deep partial observability of the termination-relevant state, hidden per-episode dynamics parameters, and an irreducible per-step lethal hazard. Benchmark suites optimised for exploration depth and representation richness do not discriminate on these axes, which is why a general-purpose agent tuned on them (DreamerV3) needs an order of magnitude more experience here than on Crafter while still leaving an unquantified performance gap.* That framing is publishable **provided** we measure the ceiling first.

---

## 8. What evidence would discriminate — concrete, runnable probes

Ordered by decisiveness. All are environment-config or logging changes; none require new algorithms. Hand off to `experiment-designer` for configs and seeds and to `senior-developer` for the oracle-observation plumbing (P1 is the only one that touches `src/`).

### P1 — Oracle-observation ceiling **(the decisive probe)**

Train the *same* rPPO agent on the *same* basic-03 environment with the observation vector augmented by privileged state: true `injury_level`, true `nutrition`, all active predator positions + their per-episode `detection_range`/`attack_range`, all active hiding-predator positions, all active food positions. Everything else — reward, dynamics, initial-state randomisation — unchanged.

- **Ladder (4 cells):** (a) baseline 27-d; (b) + true injury & nutrition only; (c) + full exteroceptive map only; (d) full oracle.
- **Reads:** if (d) plateaus near 180–220, the ceiling is low, the task is aleatoric-limited, and "harder than the suite" is **refuted** — both learners are near-optimal and the interesting science is the noise floor. If (d) plateaus at 400+, there is a 200+-step learnable gap and "harder in kind" is **supported**, with the (b)-vs-(c) contrast attributing the gap to interoceptive vs exteroceptive hiddenness.
- **Cost:** ~4 rPPO runs. rPPO collects ~47 K steps/s, so a 200 M-step run is ~1.2 h. Cheapest decisive experiment available.
- **Caveat to pre-register:** run (d) at $\gamma \ge 0.99$, not 0.95, otherwise §5.1 caps it artificially.

### P2 — Fixed initial body state

`random_start_nutrition: false`, `random_start_injury: false`, start at $(N,I)=(100,0)$. Measures how much of the survival deficit is the randomised initial distance-to-death (an irreducible per-episode floor) rather than policy quality. Pairs with P1 to decompose the ceiling.

### P3 — Fixed predator count and fixed per-episode parameters

Set `count_low: 1, count_high: 1`; collapse `detection_range`, `max_stamina`, `attack_delay`, `damage` to degenerate ranges. Removes the Hidden-Parameter-MDP structure. If Dreamer's plateau jumps materially, the world-model bottleneck is **latent-context identification**, and the right response is recurrent-policy / belief-state work, not more data.

### P4 — Hiding-predator count zero

`hiding_predator: count_low: 0, count_high: 0`. Removes the irreducible invisible-hazard term. Directly tests §5.2: if the world model's reward/continue loss drops sharply *and* the plateau rises, the unlearnable component was the binding constraint.

### P5 — rPPO discount sweep

$\gamma \in \{0.95, 0.99, 0.997\}$ at fixed `gae_lambda`. This is the cheapest test of §5.1. If survival rises materially at $\gamma=0.997$, then the headline "rPPO needed 2.6 B steps" ceases to be evidence about the environment and becomes evidence about our configuration — and the whole H3 framing shifts. **Run this before citing the rPPO datum in any paper.**

### P6 — Return decomposition logging (zero-cost, offline)

On existing eval rollouts, log per episode: $D(s_0)$, $\gamma^{T-1}D(s_T)$, $(1-\gamma)\sum\gamma^{t-1}D(s_t)$, and $100\gamma^{T-1}\mathbb 1[\text{death}]$. Establishes **empirically** which term dominates the return at each $\gamma$, converting §5.1's algebra into a measured fact. No training required.

### P7 — Imagined-vs-real hazard calibration (Dreamer-specific)

During world-model training, log $\mathbb E[1-\hat c_t]$ averaged over the $H=15$ imagination horizon, alongside the empirical per-step death rate in the replay buffer. If the imagined hazard is systematically below the real hazard, §5.2's mechanism is confirmed and the fix is on the value side (distributional critic, risk-sensitive objective, longer imagination horizon, or explicit hazard-head upweighting), not on the data side.

### P8 — Olfactory decay-power / aliasing sweep

`decay_power` $\in \{1.0, 2.0\}$ at fixed everything else. Higher decay sharpens the olfactory field (nearest entity dominates the sum), directly reducing superposition aliasing. If survival rises, the binding constraint is **sensor aliasing**, which is a genuinely novel axis relative to the suite and is a publishable finding in its own right.

---

## 9. Closest published precedents

| Precedent | Relation to our task |
|---|---|
| **Crafter** (Hafner 2022) | *Closest overall.* Procgen survival with hunger/thirst/energy/health drives, death termination, local view, discrete actions. **We differ in:** aliased aggregate sensing rather than a rendered local view; hidden injury; hidden per-episode dynamics parameters; survival-time as the metric rather than achievement coverage; no compositional tech tree. |
| **NetHack Learning Environment** (Küttler et al. 2020) / MiniHack | Permadeath, hunger clock, extreme procedural variation, notoriously requires billions of steps. **Closest precedent for "a small-looking task that eats billions of steps."** We differ in scale and in the absence of a symbolic knowledge prior. |
| **Safety Gym** (Ray, Achiam & Amodei 2019) / **AI Safety Gridworlds** (Leike et al. 2017) | The rare-irreversible-catastrophe family. Our $-100$ death penalty is a soft-constraint version of their cost signal. **This, not the DreamerV3 suite, is the literature our reward structure belongs to.** |
| **LAMBDA / safe model-based RL** (As et al., ICLR 2022) | Dreamer-style latent world model + Lagrangian constraint on Safety-Gym. The closest thing to "world model meets rare catastrophe" and the natural comparison for any risk-sensitive critic we build. |
| **Hidden-Parameter MDPs** (Doshi-Velez & Konidaris 2016); **RL²** (Duan et al. 2016) | The formal home for our per-episode hidden predator parameters. |
| **Homeostatic RL** (Keramati & Gutkin 2014; Laurençon et al. 2021) | The formal home for our coupled drive-reduction reward. |
| **Ng, Harada & Russell (1999)** | The policy-invariance-under-shaping result that §5.1 leans on. |

**Missing from `docs/project/references/`:** Crafter (Hafner 2022, "Benchmarking the Spectrum of Agent Capabilities"), Safety Gym (Ray et al. 2019), LAMBDA (As et al. 2022), NLE (Küttler et al. 2020), Doshi-Velez & Konidaris (2016), Ng/Harada/Russell (1999). Worth fetching the first three at minimum — they are the papers a reviewer will name.

---

## 10. Cross-links

Sibling investigations in this five-way parallel study (H1–H5):

- [[dreamer_srl_h1_speed_investigation]] — why the Dreamer stack is slow in wall-clock; owns the training curves this memo cross-references in §5.3 rather than re-fetching.
- [[dreamer_srl_faithfulness_review]] — whether our Dreamer implementation matches the published algorithm; §5.2's claims about $\hat c_t$, the $\lambda$-return and the percentile return scaling assume faithfulness and should be read against it.
- [[DREAMER_SRL_INVESTIGATION]] — the umbrella investigation doc.
- [[dreamer_srl_settings_regime_critique]] — whether our hyperparameter regime is inside DreamerV3's tuned envelope; §5.2's "symlog handles our scale fine" and the $H=15$ / $\gamma=0.997$ discussion are direct inputs to it.

Project-side:

- [[dreamer_srl_vs_rppo_survival_speed]] — the survival/sample-efficiency numbers quoted in §5.3.
- [[dreamer_conventional_failure_modes_for_our_setup]] — prior RL critique of Dreamer-on-this-project.
- [[threat_discrimination_assays]] — the designed predator/rabbit olfactory ambiguity discussed in §2.
- [[dreamer_v3_implementation_critique]] — implementation-level counterpart.

---

## 11. Next steps

**→ `experiment-designer`** — Turn P1 (oracle-observation ladder, 4 cells, $\gamma\ge0.99$) into configs and a seed plan; it is the single decisive experiment and should be pre-registered. P2/P3/P4 are three one-line config variants that can ride along in the same sweep. P5 (rPPO $\gamma$ sweep) is independently cheap and should be run before the rPPO datum is cited anywhere.

**→ `senior-developer`** — P1 requires an oracle-observation channel in the environment's observation assembly (an additive, config-gated block; the sensor pipeline already assembles blocks conditionally, so this is contained rather than invasive). P6 and P7 are logging-only additions. **No changes to reward, dynamics, or the existing 27-d layout** — the oracle block must be strictly additive so the existing dimension/fingerprint guard still distinguishes the variants.

**→ `experiment-analyzer`** — P6 is runnable today on existing eval rollouts with no new training: decompose logged episode returns into the four terms of §5.1 and report which dominates at $\gamma=0.95$ vs $\gamma=0.997$.

**→ `professor-bayesian-brain`** — §5.2's "the continue predictor regresses to a base-rate hazard under irreducible aleatoric lethality" is the RL-side statement of a precision-weighting problem; the cognitive-side framing (should the agent hold a *belief* over episode-level threat parameters, and is that belief what the modulator should carry?) is theirs.

**→ `professor-dl-theory`** — the observation-side claim in §2 ("olfaction is a non-invertible superposition; the encoder must solve an inverse problem") is an architectural-identifiability question about the encoder, not an RL question.

**→ `literature-reviewer` / `academic-pdf-fetch`** — the six missing references named in §9.

---

*Feedback from professor-rl — 2026-07-27. Scope: RL-algorithmic and task-structural comparison only. Environment facts verified against `src/environment/sensor.py`, `src/environment/core.py`, `configs/environment/default.yaml`, `configs/environment/experiment/basic/03-random_init_10x10.yaml`, `configs/environment/experiment/basic/04-jump_attack_10x10.yaml`, `configs/models/recurrent_ppo/recurrent_ppo_M.yaml`, `configs/models/dreamer_srl/01_food_only_M.yaml`. Benchmark facts verified against Hafner et al. 2023, Table A.1 and §Benchmarks. No code, config, or script was modified.*
