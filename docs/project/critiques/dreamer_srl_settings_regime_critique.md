# Are our DreamerV3 settings in the wrong operating regime?

*Critique memo — professor-rl — 2026-07-27*

> **One-line verdict:** Our replay ratio is defensible and roughly where the paper would put it for a 20M-step budget — but our *model size* is inverted relative to the paper's only vector-observation reference point, and the cheapest available win is **XS at replay_ratio 0.25**, which buys 4× the updates-per-datum at (predicted) near-zero wall-clock cost. The one setting I think is genuinely mis-specified is the **death penalty of 100**, which lands the two-hot critic's value discrimination inside a single bin.

---

## 1. What this memo is and what it claims

We run an in-house re-implementation of DreamerV3 — a model-based reinforcement-learning agent that first learns an internal simulator ("world model") of our gridworld and then improves its policy mostly by *imagining* rollouts inside that simulator. It is currently beating our model-free baseline on survival (120 vs. 52 steps at matched experience) but takes roughly 100× longer in real time per episode.

A sibling reviewer is checking whether our code *faithfully implements* the published algorithm. This memo asks a different question: **assume the code is correct — are the numbers we chose putting the algorithm in a regime where it works well?** I re-read the DreamerV3 paper's actual configuration tables (both the 2023 arXiv version and the 2025 *Nature* version, which revised them) and compared them to our saved configs and to what our environment actually looks like.

Three claims, in order of confidence:

1. **The replay ratio is not the scandal it looks like.** We do 1 gradient update per 16 environment steps. The paper's large-budget runs (Atari 200M, Minecraft, DMLab) do 1 per 32 — *fewer* than us. Measured per parameter, our model is better-fed than the paper's biggest runs. The premise "we're 16× below the paper's tuned value" compares against the small-data benchmarks (Atari-100k, BSuite), which is the wrong reference class for a 20M-step budget.
2. **But the model size is inverted.** The paper's *only* low-dimensional-vector benchmark (DeepMind Control from joint angles) uses its **smallest** network — 1 million parameters — paired with its **highest** update rate. We use a 20.7-million-parameter network with a low update rate. We are on the opposite corner of the same trade-off.
3. **The death penalty of 100 is the setting most likely to be actively hurting.** The critic represents values on a log-compressed scale; near a magnitude of 100, its resolution is about 16 value-units per bin, and the improvement we are asking it to detect (surviving 52 steps vs. 120 steps) is worth about 15 value-units. That is a one-bin difference. Shrinking the penalty to ~5–10 (rescaling food proportionally) would move the same decision into a 20× finer part of the scale, at zero algorithmic cost.

Everything below is the arithmetic. Section 7 is a falsifiable prediction set the sibling empirical doc can confirm or refute.

**Siblings:** [[DREAMER_SRL_INVESTIGATION]] (empirical axis) · [[dreamer_srl_faithfulness_review]] (is the code the algorithm?) · [[dreamer_srl_h1_speed_investigation]] (why is it slow?) · [[gridworld_vs_dreamerv3_benchmarks_difficulty]] (is our task hard?)
**Prior in-project memo this supersedes in part:** [[replay_ratio_speed_vs_performance]] — that memo used the 2023 arXiv Table A.1 only, and concluded we sit near the "data-scarce → high ratio" end. The 2025 *Nature* Extended Data Table 2 revises those numbers substantially and changes the conclusion; see §3.

---

## 2. Notation: one unit for "how much training per datum"

Three different papers/versions use three different normalisations. I fix one:

$$
g \;\triangleq\; \frac{\text{gradient steps}}{\text{policy step}} \;=\; \frac{\text{RR}}{B \cdot T}
$$

where $\mathrm{RR}$ is the paper's "replay ratio" (replayed time steps per policy step) and $B \cdot T = 16 \times 64 = 1024$ is the number of transitions consumed per gradient step. This is invariant across paper versions. Our config sets sheeprl's `algo.replay_ratio` $= 0.0625$, and the code computes gradient steps per iteration as `max(1, replay_ratio × num_envs)` = 8 per 128 environment steps (`src/algorithms/dreamer_srl/dreamer_srl_main.py:1264`), so

$$
g_{\text{ours}} = \tfrac{8}{128} = \tfrac{1}{16} = 0.0625, \qquad \mathrm{RR}_{\text{ours}} = 64 .
$$

Action repeat is 1 in our environment, so policy steps = environment steps throughout.

Also define the **replayed-transitions-per-parameter** ratio $\rho = (\text{total replayed transitions}) / (\text{parameters})$, the model-based analogue of a tokens-per-parameter budget.

---

## 3. Q1 — Replay-ratio regime

### 3.1 What the paper actually prescribes

**Nature 2025, Extended Data Table 2** (the authoritative, revised benchmark overview; the 2023 arXiv Table A.1 numbers differ and are internally inconsistent with its own Figure 6a legend — see §3.4):

| Benchmark | Env steps | Action repeat | Env instances | Replay ratio RR | GPU-days | Model size | $g$ | $\rho$ (replayed/param) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Minecraft | 100M | 1 | 64 | 32 | 8.9 | 200M | 1/32 | 16 |
| DMLab | 100M | 4 | 16 | 32 | 2.9 | 200M | 1/32 | 4 |
| ProcGen | 50M | 1 | 16 | 32 | 8.3 | 200M | 1/32 | 8 |
| Atari (200M) | 200M | 4 | 16 | 32 | 7.7 | 200M | 1/32 | 8 |
| Atari100K | 400K | 4 | 1 | 128 | 0.1 | 200M | 1/8 | 0.06 |
| BSuite | <1M | 1 | 1 | 1024 | 0.5 | 200M | 1 | ~5 |
| DMC Vision | 1M | 1 | 16 | 256 | 1.2 | 200M | 1/4 | 1.3 |
| **DMC Proprio** | **1M** | **1** | **16** | **1024** | **1.5** | **1M** | **1** | **1024** |
| **Ours (M)** | **~20M** | **1** | **128** | **64** | **~0.75/run** | **20.7M** | **1/16** | **~50** |
| **Ours (XS)** | ~20M | 1 | 128 | 64 | — | 3.17M | 1/16 | ~325 |

*(GPU-days for us: ~18 h at 313 env-steps/s. $\rho$ computed as env steps × RR / params.)*

Two things fall out immediately.

**(a) On the update-rate axis alone, RR = 64 is unremarkable.** Ordered by $g$: Atari-100k 1/8 → **us 1/16** → the four large-budget pixel benchmarks 1/32. Our budget (~20M) sits between DMC's 1M and Atari's 200M, and our $g$ sits between DMC-Vision's 1/4 and Atari's 1/32. That is exactly the interpolation the paper's own budget→RR schedule would produce. The brief's framing ("Atari-100k uses 1024 = 16× more updates/datum") compares us to a 400K-step benchmark; the correct comparison at a 20M-step budget is Atari-200M/Minecraft, which use **half** our update rate.

**(b) The paper is explicit that these are compute choices, not optima.** Nature Methods, "Computational choices": *"The replay ratio controls the trade-off between computational cost and data efficiency as analysed in Fig. 6 and is chosen to fit the step budget of each benchmark."* Data efficiency is **monotone increasing** in RR across the entire swept range (arXiv Fig. 6a; Nature Fig. 6d: *"Higher replay ratios predictably increase the data efficiency"*). So RR = 64 is not "correct" — it is a wall-clock decision with a quantifiable data-efficiency cost, exactly as we chose it. The question is whether we picked the right point.

### 3.2 The wall-clock frontier, computed

From the measured operating point (313 env-steps/s at $g = 1/16$, 128 envs, 10.7 ms/gradient step), decompose one *iteration* = 128 environment steps:

- $t_{\text{iter}} = 128/313 = 0.409$ s
- gradient time $= 8 \times 10.7\,\text{ms} = 0.086$ s
- environment + policy time $t_e = 0.409 - 0.086 = 0.323$ s

So the general iteration cost is $t_{\text{iter}}(g) = t_e + 128\,g \cdot t_{\text{grad}} = 0.323 + 1.370\,g$ seconds.

| $g$ (=`replay_ratio`) | RR | grad steps/iter | env-steps/s | slowdown | hours to 20M steps |
|---:|---:|---:|---:|---:|---:|
| 0.0625 (current) | 64 | 8 | 313 | 1.00× | 17.8 |
| 0.125 | 128 | 16 | 259 | 1.21× | 21.5 |
| **0.25** | **256** | **32** | **192** | **1.63×** | **28.9** |
| 0.5 | 512 | 64 | 127 | 2.47× | 43.7 |
| 1.0 | 1024 | 128 | 76 | 4.14× | 73.5 |

At the pessimistic end of the observed throughput band (209 env-steps/s, $t_e = 0.527$ s) the slowdowns compress to 1.42× at $g=0.25$ and 3.10× at $g=1.0$ — the frontier is *flatter* when the environment is the bottleneck, which favours raising $g$.

### 3.3 Break-even: how steep must the data-efficiency gain be?

Model the environment steps needed to reach a fixed survival level as $N(g) = N_0\, g^{-\alpha}$. Total wall-clock to that level is $W(g) \propto N(g)\,(t_e + 1.370\,g)$. Raising $g$ from 0.0625 to $g'$ is wall-clock-neutral when

$$
\left(\frac{g'}{0.0625}\right)^{\alpha} \;=\; \frac{t_e + 1.370\,g'}{t_e + 0.0856}.
$$

| Move | Break-even $\alpha$ (fast case, $t_e{=}0.323$) | Break-even $\alpha$ (slow case, $t_e{=}0.527$) |
|---|---:|---:|
| $g:0.0625 \to 0.25$ | **0.35** | **0.25** |
| $g:0.0625 \to 1.0$ | **0.51** | **0.41** |

**What is $\alpha$ empirically?** Reading arXiv Figure 6a (four panels: Breakout, MsPacman, Crafter, DMLab-Goals-Small; seven curves at training ratios 1→64 on log-$x$ axes spanning three decades): the curves are near-parallel horizontal shifts, and the full 64× sweep displaces each family by roughly **1.2–1.4 decades** of environment steps. That gives

$$
\alpha \;=\; \frac{\log_{10}(\text{data reduction})}{\log_{10} 64} \;\approx\; \frac{1.2\text{–}1.4}{1.806} \;\approx\; 0.66\text{–}0.78 .
$$

Both break-even thresholds are comfortably exceeded. Even at a heavily discounted $\alpha = 0.4$ (accounting for our task being different and for the visible bunching of the TR-32 and TR-64 curves in the Breakout/MsPacman panels), **$g = 0.25$ still wins on wall-clock** in both throughput cases, while $g = 1.0$ is marginal.

> **Prescription (Q1): move to `replay_ratio` 0.25 (RR = 256, 32 gradient steps per 128-env-step iteration).** It requires only $\alpha > 0.35$ to be *wall-clock-free*, and if $\alpha$ is anywhere near the paper's 0.7 it is wall-clock-*positive* by a factor of ~1.6–1.8. `replay_ratio` 1.0 needs $\alpha > 0.51$ and is the riskier bet.

### 3.4 A caveat about the 2023 table

The arXiv Table A.1 numbers used in [[replay_ratio_speed_vs_performance]] (Atari-100k 1024, DMC 512, Crafter 512, Atari-200M 64, Minecraft 16) are not consistent with that paper's own Figure 6a, whose training-ratio legend tops out at 64 — yet Crafter's tabled value is 512, eight times above the top of its own sweep. The units drift between "per policy step" (Table A.1) and "per environment step" (Fig. 6 caption). The 2025 *Nature* Extended Data Table 2 is internally consistent and carries a worked example in Methods (*"a replay ratio of 32 on Atari with action repeat of 4 and batch shape 16 × 64 corresponds to 1 gradient step every 128 environment steps, or 1.5 million gradient steps over 200 million environment steps"*). **Anchor on the Nature table.** This is why my recommendation here is milder than the earlier memo's.

---

## 4. Q2 — Model size: is M under-trained?

### 4.1 The gradient-steps-per-parameter arithmetic says no

At $g = 1/16$ and ~20M environment steps, M has taken ~1.25M gradient steps, consuming $1.25\text{M} \times 1024 = 1.28$ G replayed transitions across 20.7M parameters:

$$
\rho_{\text{M}} \;=\; \frac{1.28 \times 10^9}{2.07 \times 10^7} \;\approx\; 62 .
$$

Compare the paper (from the table in §3.1): Atari-200M $\rho = 8$, Minecraft $\rho = 16$, DMLab $\rho = 4$, BSuite $\rho \approx 5$, DMC-Vision $\rho = 1.3$. **Our M model receives 4–50× more replayed data per parameter than any of the paper's 200M-parameter runs.** By this measure M is not under-trained; it is comparatively over-fed.

The single exception is the one benchmark that resembles ours in observation type:

$$
\rho_{\text{DMC-Proprio}} \;=\; \frac{1\text{M} \times 1024}{1\text{M}} \;=\; 1024 .
$$

**That is the finding.** For low-dimensional vector observations, the authors chose a **1-million-parameter** model and pushed $\rho$ to 1024 — 20× smaller and 16× better-fed per parameter than our M. Our XS (3.17M params, $\rho \approx 325$) is within 3× of that point; our M is 20× off it.

Read as a compute-optimal-allocation question: our total replayed-transition budget of 1.28 G, allocated at DMC-Proprio's $\rho \approx 1024$, prescribes a model of **~1.25M parameters** — *smaller than XS*.

### 4.2 The counter-argument, stated fairly

arXiv Figure 6b (Breakout, MsPacman, Crafter, DMLab across XS→XL) shows monotone gains with model size in *both* final performance and data efficiency, with no size-induced degradation anywhere. The caption is explicit: *"larger models achieve not only higher final performance but also higher data-efficiency."* The paper never shows bigger hurting.

But: every panel in Fig. 6b is a **pixel** task with a convolutional encoder, run at 50M–500M environment steps. Our XS/M presets have `cnn_keys` empty — the entire convolutional stack that Fig. 6b's size axis is mostly scaling is absent. Our "M" is 20.7M rather than the paper's 37M precisely because the CNN is stripped. So Fig. 6b's evidence transfers weakly, and DMC-Proprio's 1M-parameter choice is the only direct evidence in the vector-observation regime — where it points the other way.

### 4.3 The actually-interesting move: trade size for update rate at fixed wall-clock

The gradient-step cost is dominated by the model, so shrinking the model buys back the budget needed to raise $g$. If XS's gradient step costs $\beta \cdot 10.7$ ms, then XS at $g'$ matches M's current wall-clock when

$$
128\,g' \cdot \beta \cdot 10.7\,\text{ms} \;=\; 0.0856\,\text{s} \quad\Longrightarrow\quad g' = \frac{0.0625}{\beta}.
$$

XS has 15% of M's parameters, but a gradient step includes fixed overheads (imagination rollout control flow, host-device sync, optimizer bookkeeping), so $\beta$ will be well above 0.15. Plausible values:

| assumed $\beta$ | wall-clock-neutral $g'$ | vs. current | XS $\rho$ at 20M steps |
|---:|---:|---:|---:|
| 0.60 | 0.104 | 1.7× more updates | 542 |
| 0.40 | 0.156 | 2.5× | 813 |
| 0.25 | 0.25 | **4×** | 1300 |

> **Prescription (Q2): measure $t_{\text{grad}}(\text{XS})$ first, then run XS at the wall-clock-neutral $g'$ against M at $g = 0.0625$.** If $\beta \le 0.4$ this is a *free* 2.5–4× increase in updates-per-datum, and it lands XS near DMC-Proprio's $\rho \approx 1024$ — the paper's own vector-observation operating point. That single measurement (one number, ~10 minutes of profiling) is the highest-information-per-cost action in this memo.

### 4.4 Falsifiable prediction for the sibling analyzer

Plot survival vs. **gradient steps** (not environment steps, not wall-clock) for the existing M and XS runs at matched `replay_ratio`.

- **If capacity binds (M is right):** the curves separate — M above XS at every matched gradient-step count, gap widening.
- **If optimisation binds (updates, not capacity):** the curves are near-superimposed against gradient steps, and separate only against environment steps because M's gradient steps are more expensive. This is the outcome that licenses the XS-plus-higher-$g$ trade.

---

## 5. Q3 — Horizon, discount, and rare catastrophic events

### 5.1 The discount is well matched; the imagination horizon is not the constraint

$\gamma = 1 - 1/333$ gives a 333-step effective horizon against ~120-step episodes (cap 500). Because every episode ends in death or timeout, the value of a state decomposes roughly as

$$
V(s) \;\approx\; \underbrace{\textstyle\sum_k \gamma^k c_k\, r^{\text{food}}_k}_{\text{dense homeostatic stream}} \;-\; 100\cdot \mathbb{E}\!\left[\gamma^{\,T_{\text{death}}}\right].
$$

With $\gamma^{52} = 0.855$ and $\gamma^{120} = 0.697$, improving survival from the model-free baseline's 52 steps to Dreamer's 120 steps moves the death term from $-85.5$ to $-69.7$ — a **+15.8 value gain**. So $\gamma$ is chosen such that survival time is the dominant driver of value, which is exactly what we want. A shorter horizon (say 30 steps) would make a death 100 steps away nearly invisible. **No mismatch here; leave $\gamma$ alone.**

**The 15-step imagination horizon is also not the binding constraint.** Our grid is 10×10; predator `move_interval` is `[1,1]` (full speed, 1 cell/step) and the agent likewise. The Chebyshev diameter of the reachable area is 9 steps. **Any predator anywhere on the map can reach the agent in ≤ 9 steps < H = 15.** So $H$ spans the entire causal reach of the threat. Further, with a per-step death hazard of roughly $1/120$, the fraction of 15-step imagined rollouts containing a terminal event is $1 - (1-1/120)^{15} \approx 11.8\%$ — about 121 of the 1024 imagination starts per gradient step. The critic sees deaths at an ample rate. **$H = 15$ is fine; what limits threat anticipation is world-model fidelity on an aliased olfactory channel (§6), not horizon length.**

### 5.2 The death penalty is in the wrong part of the symlog scale — this is my top concern

The critic predicts a categorical distribution over $K = 255$ equally spaced bins on $\mathcal{B} = [-20, +20]$ in symlog space (arXiv Eq. 8), with

$$
\mathrm{symlog}(x) = \mathrm{sign}(x)\ln(1+|x|), \qquad v_\psi(s) = \mathrm{symexp}\!\left(\mathbb{E}_{p_\psi}[b_i]\right).
$$

Bin width in symlog space: $\Delta u = 40/254 = 0.1575$. Because $\mathrm{d}\,\mathrm{symexp}(u)/\mathrm{d}u = e^{|u|}$, the **relative** resolution is scale-free but the **absolute** resolution degrades with magnitude:

$$
\Delta v \;\approx\; \Delta u \cdot |v| \;=\; 0.1575\,|v| \quad \text{for } |v| \gg 1 .
$$

At $|v| \approx 85$ (our typical state value, dominated by the discounted death term), one bin spans **≈ 13.4 value units**. The discrimination the critic must make to drive the improvement from 52-step to 120-step survival is worth **15.8 value units** (§5.1) — **1.2 bins.**

Two-hot encoding makes the *expectation* exact between bins, so this is not a hard quantisation floor. But the cross-entropy gradient is supported on only two adjacent bins, and the entire survival-improvement signal must be carried by shifting mass between two neighbours of a 255-way softmax. Every other source of value variance (food timing, injury, episode-to-episode predator count 0–2) also lives in that same one-to-two-bin window. **The signal-to-bin ratio is ~1, where the paper's benchmarks typically operate at 10–100.**

The fix is a reward rescale, not an algorithm change. If `death_penalty` were 5 with food rewards scaled by the same factor 1/20, the same decision would sit near $|v| \approx 4.3$, where one bin spans $0.1575 \times 4.3 = 0.68$ units against a discrimination of $15.8/20 = 0.79$ units — same ratio. **So a naive uniform rescale does not help**; symlog's relative resolution is scale-invariant by construction, which is the whole point of the design.

What *would* help, in descending order of cleanliness:

1. **Narrow the bin range.** $\mathcal{B} = [-20,20]$ is sized for Atari returns up to $\mathrm{symexp}(20) \approx 4.9\times10^8$. Our returns never leave $[-110, +50]$, i.e. $\mathrm{symlog} \in [-4.71, +3.93]$. **We are using ~22% of the bin range and wasting 78% of the critic's 255 bins on values that can never occur.** Setting $\mathcal{B} = [-6, +6]$ recovers a **3.3× finer** resolution at zero cost. This is a one-line config change (`bins` range) if the implementation exposes it, and it is the highest-leverage single change in this memo after the replay ratio.
2. **Raise `bins` from 255 to 511** — 2× finer, costs one extra output layer width. Weaker than (1) and less faithful.
3. **Reduce the ratio of the terminal penalty to the dense reward stream** (not a uniform rescale — a *ratio* change), so that state value is less dominated by "when will I die" and more by the dense homeostatic term. This changes the task, so it is a `professor-pain-modeling` / `experiment-designer` decision, not mine.

Note (1) is a deviation from paper defaults, so it must be routed through the deviation log, and the faithfulness sibling should be told it is a *deliberate* deviation rather than a parity bug.

### 5.3 Return normalisation: the percentile window probably does see the tail

The actor scales returns by $S = \mathrm{Per}(R^\lambda, 95) - \mathrm{Per}(R^\lambda, 5)$, applied as $\max(1, S)$ (arXiv Eq. 11–12; `moments` decay 0.99, max 1.0, percentiles 5/95 in our config). The concern I went in with — that a ~1%-frequency catastrophe would fall below the 5th percentile and leave $S$ blind to the death tail — **does not hold here**, because our returns are not "mostly zero with a rare −100 spike". Every state's value already carries $-100\gamma^{T_{\text{death}}}$, so the imagined-return distribution is a broad band roughly spanning $[-100, -50]$, and the 5–95 range genuinely covers it. $S$ should sit in the tens.

That is the benign case. The two failure signatures to watch, both cheap to log:

- **$S$ pinned at or below 1** (so $\max(1,S)$ clamps and no scaling occurs). Then advantages are small in absolute terms and the entropy term at $\eta = 3\times10^{-4}$ is comparatively strong → slow, exploration-dominated learning. Signature: high, non-decaying actor entropy with a flat survival curve.
- **$S$ swinging by more than ~2× within a few thousand gradient steps.** With `moments.decay = 0.99` the EMA half-life is ~69 gradient steps — very fast. A rapidly-moving denominator makes the effective actor learning rate non-stationary. Signature: actor gradient-norm spikes anti-correlated with $S$.

### 5.4 Truncation vs. termination — flag for the faithfulness sibling

With a 500-step cap and $\gamma$'s 333-step horizon, the two are *not* separable by discounting. If a cap-induced truncation is written into the buffer as `terminated` (continue flag 0) rather than `truncated` (continue flag 1, bootstrap), the critic learns that surviving to step 500 is worth zero future value — a large negative bias applied precisely to the best trajectories, i.e. anti-correlated with the thing we are optimising. At our current 120-step mean this bites rarely; **it becomes a first-order problem exactly as the agent gets good**, which is where we are heading. Worth an explicit check in [[dreamer_srl_faithfulness_review]].

---

## 6. Q4 — The vector-observation regime

### 6.1 What the paper says about vector observations

- **Nature Methods, "Networks":** *"Vector inputs are symlog transformed and then encoded and decoded"* by MLPs — no special handling beyond symlog. Our MLP-only encoder/decoder with `cnn_keys` empty is the intended configuration.
- **arXiv Figure D.1** — the world-model ablation panel — is run on **Reacher Hard (Proprio)**, i.e. a vector-observation task. Caption: *"Free bits avoids overfitting in simple environments. Symlog encoding and predictions for proprioceptive observations speeds up learning."* Both of the ablations that matter most for us were validated on a vector task, and both come out in favour of our current settings (`kl_free_nats: 1.0`, obs symlog on).
- **Nature ED Table 2:** the vector benchmark (DMC Proprio) is the *only* one not run at 200M parameters — it uses 1M. See §4.
- **arXiv Table L.1 (BSuite, also vector observations):** DreamerV3 scores well on credit assignment (`umbrella_length` 0.783 vs. Muesli 0.173; `umbrella_distract` 0.957 vs. 0.217) but **0.000 on both `deep_sea` exploration configurations** and 0.478 on `memory_len`. DreamerV3 has strong credit assignment and **no directed exploration**.

**On symlog for already-small vector observations:** for $|x| \ll 1$, $\mathrm{symlog}(x) \approx x$, so it is nearly the identity on our normalised body/olfactory channels (at $x = 1$ it compresses to $0.693$). Harmless. It earns its keep on the one channel that genuinely has a wide range — `extero_nociception`, configured with `clip_max: 100` — which symlog maps to $\le 4.6$ instead of letting it dominate the 27-dimensional reconstruction loss by four orders of magnitude in squared error. **Keep symlog on.** This is the case it was designed for.

### 6.2 The regime problem our observation actually creates

Reading `configs/environment/default.yaml` and the basic-03/04 experiment configs, the agent's exteroception is **not a spatial map**: `visual_sensor_range: 0` (no visual channel), and the only exteroceptive signal is a 5-dimensional olfactory vector with `sensor_radius: 20` and `decay_power: 1.0`. Since the sensor radius exceeds the grid diagonal (~13), *every* entity on the map contributes to that 5-vector simultaneously, distance-weighted as $1/d$.

That has three consequences the paper's benchmarks do not share:

1. **Predator and rabbit are near-aliased.** The hunting predator's property vector is `[0, 0.7, 0.5, 0, 0]` and the wandering rabbit's is `[0, 0.5, 0.7, 0, 0]` — the same two channels, near-swapped, with per-episode jitter `properties_std = [0, 0.3, 0.3, 0, 0]` that is comparable to the 0.2 separation itself. The agent must disambiguate threat from distractor **by dynamics over time** (hunters approach, wanderers do not), not by instantaneous smell.
2. **The decision-relevant signal is a tiny fraction of the reconstruction loss.** DreamerV3's world model is trained predominantly by reconstruction ($\beta_{\text{pred}} = 1.0$; the Nature ablation section reports the model *"rests predominantly on the unsupervised objective"*). Our decoder reconstructs 27 dimensions dominated by smooth, easily-predicted homeostatic and background-olfactory channels. Gradient is allocated by *variance*, not by *decision relevance*. In pixel benchmarks a predator sprite occupies many pixels and moves distinctively; here "a hunter is closing from 3 cells" is a sub-0.2 perturbation on two of 27 channels, superposed with two rabbits and up to twelve hiding predators.

   The one mitigating factor: reward-head cross-entropy is 1 term against a 27-term decoder, so the reward signal is ~3.5% of the prediction loss — versus ~0.008% against a $64\times64\times3$ image decoder. **Reward shaping of the representation is ~400× stronger for us than in the pixel benchmarks.** That is a genuine advantage of our regime and one reason Dreamer is beating the model-free baseline at all.
3. **Latent capacity vastly exceeds observation content.** The stochastic latent is 32 categoricals × 32 classes = $32\ln 32 = 111$ nats of capacity, sitting on top of a 1024-unit deterministic GRU (M) — against a 27-dimensional observation whose usable information content is a few nats. This is the regime where `kl_free_nats` earns its keep (Fig. D.1: *"free bits avoids overfitting in simple environments"*), and it is also the regime where **the exact semantics of the free-nats clamp matter most**: applied to the *summed* KL across latents (≈1 nat floor, danijar's implementation) versus *per latent* (32 nats floor) differ by 32×, and with our low-information observation the per-latent reading would clamp the dynamics loss for essentially all of training — meaning the prior (dynamics predictor) is never trained to match the posterior, and imagination becomes untethered. In pixel benchmarks the natural KL is high enough that the two readings rarely differ in practice, so this is a bug class that **only manifests in our regime**. → explicit check for [[dreamer_srl_faithfulness_review]].

### 6.3 A 2023-vs-2025 gap that hurts us specifically

The Nature version lists among its replay-buffer changes: *"storing and updating latent states"* in the buffer, so that the world model is initialised from a stored recurrent state on replayed sequences rather than from zeros. sheeprl implements the 2023 algorithm, which re-infers the belief from a zero recurrent state at the start of each sampled $T = 64$ sequence.

In a fully-observed task (DMC proprio) that costs almost nothing. In an **aliased POMDP where predator-vs-rabbit identity is only recoverable by temporal integration**, every sampled sequence spends its first several steps with a cold, uninformative belief, and those steps contribute gradient as if they were well-posed. With ~120-step episodes and $T = 64$, roughly half of every episode is trained from a cold start. **This is a real, regime-specific handicap of the 2023 formulation for our task** — and it is *not* a faithfulness bug, since we target the 2023 algorithm. Flag it as a known, principled deviation-of-the-paper-from-us rather than of-us-from-the-paper. Fixing it is invasive (buffer schema + world-model init), so it is a Phase-3 candidate, not a now-fix.

### 6.4 A smaller 128-env artefact

`learning_starts: 1024` is interpreted as environment steps and divided by `num_envs` (`dreamer_srl_main.py:582`, `derive_prefill`), so at 128 envs the prefill is **8 iterations = 8 time steps per environment stream** — far short of the $T = 64$ sequence length, and 128× shallower per-stream than the sheeprl XS reference (which prefills 1024 steps into a single stream). Whatever the buffer's readiness guard does, the first gradient updates operate on a buffer with almost no *temporal* diversity. This is second-order relative to §5.2 and §3, but it is free to fix (`learning_starts: 8192` at 128 envs restores parity in per-stream depth) and it interacts with the primacy-bias literature cited in [[replay_ratio_speed_vs_performance]] — the shallower the prefill, the more the early updates overfit a near-degenerate dataset, and the *more* that matters as we raise the replay ratio.

---

## 7. Q5 — Falsifiable prediction set

Written so [[DREAMER_SRL_INVESTIGATION]] can mark each **CONFIRMED / REFUTED / INCONCLUSIVE**. Each states what settings-are-the-problem predicts (**X**) and what settings-are-fine predicts (**Y**).

**P1 — Replay-ratio data efficiency (the load-bearing test).**
Run M at `replay_ratio` ∈ {0.0625, 0.25} on basic-04, matched seeds, plot survival vs. **environment steps**.
- **X (settings bind):** the 0.25 curve reaches any fixed survival level in **2.3–3.0× fewer** environment steps (implied $\alpha \in [0.6, 0.8]$), and therefore also in **1.4–1.8× less wall-clock** despite the 1.63× throughput loss.
- **Y (settings fine):** the shift is **< 1.5×** in environment steps ($\alpha < 0.3$), and 0.25 loses on wall-clock. Our task is then in the diminishing-returns part of the replay-ratio curve and RR = 64 is correct.
- *Decision rule:* if the env-step shift is ≥ 1.9× (i.e. $\alpha \ge 0.46$), adopt 0.25 as the production default.

**P2 — Size vs. update-rate, at matched gradient steps.**
Replot existing M and XS runs against **gradient steps**.
- **X (M is capacity-bound and correct):** M sits clearly above XS at every matched gradient-step count, gap widening late.
- **Y (optimisation-bound, XS is the better frontier point):** the two curves are within seed noise against gradient steps. Then M's only advantage is per-gradient-step quality, which the wall-clock arithmetic in §4.3 converts into a loss.
- *Corollary prediction:* if **Y**, then XS at the wall-clock-neutral $g'$ from §4.3 will **beat M at $g = 0.0625$ on survival-vs-wall-clock** by a factor $(g'/0.0625)^{\alpha}$ ≈ 1.7–2.6× fewer environment steps at equal wall-clock.

**P3 — The XS gradient-step cost (prerequisite measurement, not a prediction).**
Measure $t_{\text{grad}}$ for XS at $B{=}16, T{=}64, H{=}15$.
- If $\beta = t_{\text{grad}}(\text{XS})/t_{\text{grad}}(\text{M}) \le 0.4$, P2's corollary is live and cheap.
- If $\beta > 0.7$, the gradient step is overhead-bound rather than parameter-bound, the size-for-updates trade evaporates, and **the entire Q2 line of attack is dead** — which is itself a clean finding (and would point [[dreamer_srl_h1_speed_investigation]] at per-step overhead rather than model size).

**P4 — Critic bin resolution (the death-penalty claim).**
Log the critic's predicted **bin-index distribution** (not just the scalar value) over a batch, plus the empirical range of $\lambda$-returns.
- **X (bin range is mis-specified):** ≥ 90% of predicted mass sits within bins corresponding to $\mathrm{symlog} \in [-5, +4]$, i.e. **≤ 25% of the 255 bins are ever used**, and the value spread between the best and worst 10% of states spans **≤ 3 bins**. Under X, narrowing $\mathcal{B}$ to $[-6, +6]$ should improve survival-per-env-step measurably.
- **Y (fine):** predicted mass spans ≥ 60 bins and the best-vs-worst-state value gap spans ≥ 10 bins.
- *This is the cheapest diagnostic in the set — one histogram from one existing checkpoint, no new training.*

**P5 — Return-normalisation health.**
Log $S = \mathrm{Per}(R^\lambda,95)-\mathrm{Per}(R^\lambda,5)$ and actor entropy throughout training.
- **X (normalisation pathological):** either $S$ pinned at $\le 1$ (clamp active, entropy dominates, survival flat with entropy near $\ln 6 = 1.79$), or $S$ oscillating > 2× within 5k gradient steps with anti-correlated actor-gradient-norm spikes.
- **Y (healthy):** $S$ in the range 20–80, drifting smoothly downward as the critic improves; actor entropy declining monotonically from ~1.7 toward 0.5–1.0 nats without collapsing below ~0.3.

**P6 — Free-nats clamp binding (vector-regime-specific).**
Log the raw (pre-clamp) dynamics KL, summed over the 32 latents.
- **X (clamp binding, imagination untethered):** the raw KL sits **at or below the free-nats floor for > 50% of training**, meaning the dynamics loss contributes no gradient and the prior is never fit to the posterior. Corroborating signature: open-loop imagined rollouts diverge from replayed ground truth within ~5 steps.
- **Y (fine):** raw KL sits comfortably above the floor (a few nats) for most of training, and the clamp is active only in the first few thousand gradient steps.

**P7 — Truncation handling.**
Inspect buffer records at the 500-step cap.
- **X (bug):** cap-truncated episodes carry continue flag 0. Predicted signature: survival improvement decelerates sharply as mean episode length approaches the cap, and value estimates for long-lived states are systematically depressed relative to realised returns.
- **Y (fine):** cap-truncated episodes carry continue flag 1 and bootstrap.

---

## 8. Prescribed operating point, and what I am *not* recommending

**Recommended, in priority order:**

| # | Change | Cost | Expected effect | Confidence |
|---|---|---|---|---|
| 1 | Measure $t_{\text{grad}}(\text{XS})$ (P3) | ~10 min | gates #2 | — |
| 2 | **XS at the wall-clock-neutral $g'$** (likely 0.15–0.25) | config only | 2.5–4× updates/datum, ~free | medium |
| 3 | **`replay_ratio` 0.0625 → 0.25 on M** if XS proves capacity-limited (P2 = **X**) | 1.63× wall-clock | 2.3–3.0× fewer env steps to target | medium-high |
| 4 | **Critic bin range $[-20,20] \to [-6,+6]$** (P4) | one config key; **deviation-log entry required** | 3.3× finer value resolution exactly where our decision lives | medium |
| 5 | `learning_starts` 1024 → 8192 at 128 envs | config only | restores per-stream prefill depth | low-stakes, free |

**Explicitly *not* recommending:**

- **Do not change $\gamma$.** The 333-step horizon is well matched to a survival task where value is dominated by expected time-to-death (§5.1).
- **Do not change $H = 15$.** The grid diameter is 9 steps; $H$ already spans the full causal reach of the threat (§5.1). Raising $H$ costs imagination compute for no coverage gain.
- **Do not uniformly rescale rewards** to "fix" the two-hot resolution — symlog's relative resolution is scale-invariant, so a uniform rescale is a no-op (§5.2). Narrow the bin range instead.
- **Do not add intrinsic motivation yet.** DreamerV3 scores 0.000 on BSuite's `deep_sea` (arXiv Table L.1), so it genuinely lacks directed exploration — but our environment has dense homeostatic reward and 1–4 food items on a 10×10 grid. Exploration is not obviously the binding constraint, and an intrinsic bonus would confound the pain-signal coupling that is the project's actual object of study.
- **Do not port the Nature "stored latent states in buffer" change now** (§6.3). It is the right long-run fix for our aliased POMDP but it is invasive (buffer schema + world-model initialisation + checkpoint compatibility) and belongs in a Phase-3 plan, not this sweep.

---

## 9. Relation to the project's standing questions

**On the four causes of the v8 null** ([project_plan.md §4](../project_plan.md)): this memo speaks to the "was the agent simply under-trained?" cause on the world-model branch. My answer is **partially, and not the way we assumed** — the update rate is roughly where the paper would put it for our budget, but the *size × update-rate allocation* is off the paper's own vector-observation frontier, and the critic's value resolution (§5.2) is a distinct, previously-unnamed candidate confound. If P4 comes back **X**, then every Dreamer-side comparison run to date has been made by a critic that can barely resolve the survival improvement it is being asked to optimise — which would be a genuine, algorithm-level explanation for a flat comparison, independent of any modulator question.

**On H5 (modulator timescale)** ([NEUROMODULATION_ALGORITHM.md §1.4](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)): the $T = 64$ sequence length with a zero-initialised recurrent state (§6.3) sets a hard ceiling on the timescale any modulator riding the RSSM can express — no learned signal can integrate over more than 64 steps of context, and effectively fewer given the cold start. Any H5 claim about slow modulator dynamics on the Dreamer branch must either raise $T$ or adopt the stored-latent-state fix first. This is a stronger constraint on the Dreamer branch than on the recurrent-PPO branch and should be recorded before any modulator work is planned there.

---

## Next steps

- **`experiment-analyzer`** — P1, P2, P4, P5, P6 are all answerable from existing runs plus (for P1/P2) one new pair. P4 is a single histogram from one checkpoint and should be done **first** — highest information per unit cost in the set. Record verdicts into [[DREAMER_SRL_INVESTIGATION]].
- **`experiment-designer`** — the P1 sweep (`replay_ratio` ∈ {0.0625, 0.25} on M, basic-04, ≥3 seeds) and the P2/P3 XS-vs-M wall-clock-matched pair. Pre-register **survival-vs-environment-steps** *and* **survival-vs-wall-clock** as the two verdict axes; do not co-vary `num_envs` with `replay_ratio` (gradient steps = `replay_ratio × num_envs`, so varying both confounds the compute budget).
- **`senior-developer`** — changes #2, #3, #5 are config-only and contained. Change #4 (critic bin range $[-20,20] \to [-6,+6]$) touches the critic head and the two-hot encode/decode and **must be logged as a deliberate deviation** so the faithfulness track does not flag it as a parity bug. The §6.3 stored-latent-state port is invasive (buffer schema, world-model init, checkpoint compatibility) — Phase-3 scope, plan separately.
- **`code-reviewer` / faithfulness sibling** ([[dreamer_srl_faithfulness_review]]) — two regime-specific checks that only bite in our vector-observation setting: (a) is `kl_free_nats` applied to the **summed** KL across the 32 latents or **per latent**? (§6.2 — a 32× difference that is invisible on pixel benchmarks); (b) is a 500-step cap written as `terminated` or `truncated`? (§5.4).
- **`professor-bayesian-brain` / `professor-pain-modeling`** — §6.2's olfactory aliasing (predator `[0,0.7,0.5,0,0]` vs. rabbit `[0,0.5,0.7,0,0]` with per-episode jitter of comparable magnitude) makes threat identification a *temporal-inference* problem, not a detection problem. Whether that is the intended construct is a task-design question outside my scope.
- **`pi`** — recommendations #2/#3 constitute a change of production default for the Dreamer branch and gate the timeline for every downstream Dreamer comparison. That is a focus-vs-explore call.

## References read for this memo

- **Hafner, Pasukonis, Ba, Lillicrap (2023), *Mastering Diverse Domains through World Models*** — `docs/project/references/Dreamer/sources/Hafner et al. 2023 - Mastering Diverse Domains through World Models.pdf`. Cited: Eq. 8 (two-hot symlog critic, $K = 255$, $\mathcal{B} = [-20,20]$), Eq. 11–12 (return normalisation), Table A.1 (benchmark overview — but see §3.4), Table B.1 (model sizes XS 8M → XL 200M), Fig. 6a/6b (training-ratio and model-size scaling), Fig. D.1 + App. E (world-model ablations, run on Reacher Hard Proprio), Table L.1 (BSuite per-environment scores), Table O.1 (DMC proprio scores), Table W.1 (hyperparameters).
- **Hafner, Pasukonis, Ba, Lillicrap (2025), *Mastering diverse control tasks through world models*, Nature** — `docs/project/references/Dreamer/sources/Hafner et al. 2025 - Mastering diverse control tasks through world models.pdf`. Cited: Extended Data Table 2 (revised benchmark overview — the authoritative replay-ratio / model-size table), Methods §"Computational choices", §"Networks", §"Implementation — replay ratio" (worked Atari example), §"Previous generations" (2023→2025 change list including stored latent states).
- **Prior in-project memo:** [[replay_ratio_speed_vs_performance]] (professor-rl, 2026-07-13) — the replay-ratio-barrier literature (D'Oro et al. 2023 SR-SPR; Nikishin et al. 2022 primacy bias; Schwarzer et al. 2023 BBF; Lyle et al. 2023; Sokar et al. 2023; Hussing et al. 2024) is surveyed there and not repeated here. Its per-benchmark table is superseded by §3.1 above.
