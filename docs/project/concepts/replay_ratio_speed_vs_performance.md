# Replay ratio vs. speed vs. performance in DreamerV3 — a sourced briefing

*Concept memo / literature briefing — professor-rl — 2026-07-13*

> **One-line answer:** In our data-scarce gridworld regime, the literature predicts that cutting the replay ratio to buy throughput **will cost final performance (survival steps)** unless we compensate — a low replay ratio is exactly what large-scale, data-*rich* DreamerV3 runs use, not small-benchmark runs. The good news: the same literature says a *high* replay ratio is only worth its cost when paired with plasticity guardrails (normalization, occasional resets), so the sweet spot is likely an *intermediate* ratio, not the extreme in either direction.

---

## Purpose and plain-language entry point

We run an in-house JAX re-implementation of DreamerV3 (called "dreamer_srl", a faithful port of the sheeprl DreamerV3). DreamerV3 is a *model-based* reinforcement-learning agent: it learns a compact "world model" of the environment and then trains its policy mostly inside imagined rollouts of that model. Because it is model-based and off-policy (it re-uses stored past experience from a replay buffer), we get to choose **how many gradient updates to perform per step of real experience collected**. That number is the *replay ratio* (also called train ratio or update-to-data / UTD ratio). A replay ratio of 1.0 means one gradient update per environment step; 0.0625 means one update per 16 environment steps.

We just measured, on a small gridworld survival task with 16 parallel environments, that **lowering the replay ratio from 1.0 to 0.0625 gives a 3.2× throughput speedup** (69 → 221 environment-steps per second). The reason is mechanical: in our code the number of gradient steps per iteration is `replay_ratio × num_envs`, each gradient step costs ~10.7 ms, and those gradient steps — not the environment simulation — saturate the GPU. Fewer updates per step ⇒ proportionally less compute ⇒ faster wall-clock.

The open question this memo answers from the published literature: **does running leaner (a low replay ratio, for speed) cost final task performance, measured in survival steps — or can we go fast and keep performance?** We have a prior in-project observation that lowering the replay ratio *hurt* performance, but the algorithm has since been overhauled to match sheeprl, so we are re-checking against what the field knows.

**Bottom line for the impatient:** the replay ratio is a sample-efficiency knob, and our task is a *low-data* task, so cutting it should be expected to hurt — but the effect is not free-fall, and a moderate cut (say 1.0 → 0.25–0.5) may be nearly free while an aggressive cut (→ 0.0625) is the riskiest. Watch survival steps *as a function of environment steps*, not wall-clock, to separate the two effects. Details, citations, and the guardrails are below.

---

## Scope note

This is an RL-algorithm briefing: which knob, what the field found, what to expect. It touches Phase-2/Phase-3 world-model work and does not, by itself, adjudicate any of the [v8 null-result causes](../../develop/INDEX.md) — but it *does* speak to one of them indirectly: if any prior world-model run was quietly under-trained per environment step because of a low replay ratio, that is a policy-gradient-side confound worth ruling out. Every claim below is tagged **[established]** (supported by a cited paper) or **[inference]** (my reasoning from those papers applied to our setup). I actually fetched the sources; a couple of exact numbers I could not verify against primary text are flagged **[unverified]**.

---

## 1. What the replay / UTD / train ratio does — the "replay ratio barrier" literature

### The core trade-off

The replay ratio (updates per unit of collected experience) is the primary sample-efficiency-vs-compute dial in off-policy and model-based RL. Raising it means the agent extracts more learning signal from each transition, which **improves sample efficiency (return per environment step)** but **costs proportionally more compute per environment step** and, past a point, causes learning pathologies. **[established]**

- **D'Oro, Schwarzer, Nikishin, Bacon, Bellemare, Courville (2023), "Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier" (SR-SPR), ICLR 2023 oral.** Expanded in the box below — this is the memo's load-bearing citation.

- **Nikishin, Schwarzer, D'Oro, Bacon, Courville (2022), "The Primacy Bias in Deep Reinforcement Learning," ICML 2022.** ([arXiv:2205.07802](https://arxiv.org/abs/2205.07802) · [code](https://github.com/evgenii-nikishin/rl_with_resets)). Names the mechanism behind the barrier: deep RL agents **overfit to their early experience** and thereafter cannot exploit later data. High replay ratios *amplify* this (more updates on the small early dataset before it grows). Their fix — periodically resetting the last few layers while preserving the buffer — consistently improves SAC, DrQ, and SPR on DMC and Atari 100k. **[established]**

### SR-SPR in depth — the memo's load-bearing citation

> **Paper:** Pierluca D'Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc G. Bellemare, Aaron Courville, *"Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier."* Venue: **ICLR 2023, In-Person Oral, "Notable-top-5%"** ([iclr.cc/virtual/2023/oral/12655](https://iclr.cc/virtual/2023/oral/12655) · [OpenReview id OpC-9aBBVJe](https://openreview.net/forum?id=OpC-9aBBVJe) · [code](https://github.com/proceduralia/high_replay_ratio_continuous_control)). **There is no arXiv version — it is OpenReview-only.** A non-archival **NeurIPS 2022 Deep RL Workshop poster** (same title and authors) is the precursor; the ICLR 2023 oral is the paper of record.

**The problem it names.** In off-policy / model-based RL you can do many gradient updates per collected environment step — the update-to-data (UTD) or *replay ratio*. Intuitively, more updates should mean more learning per transition and thus better sample efficiency. The paper shows this **plateaus and then *degrades* performance** past a point: that is the **"replay ratio barrier."** Simply cranking the ratio does not buy you sample efficiency; it buys you a pathology. **[established]**

**The fix — one recipe, "Scaled-by-Resetting."** Periodically **reset the agent's network parameters (fully or partially) while PRESERVING the replay buffer.** Throwing away the weights but keeping the data lets the freshly-reinitialized network re-learn from the full accumulated buffer without carrying forward the accumulated overfitting/ill-conditioning. This restores *favorable* replay-ratio scaling — the agent can absorb roughly **an order of magnitude more updates per step and keep improving.** The same recipe was instantiated two ways: **[established]**
- **SR-SAC** = SAC + resets, on the DeepMind Control Suite (DMC15, 1M-step budget), continuous control.
- **SR-SPR** = SPR + resets, on Atari 100k, discrete control, with the replay ratio scaled up to **16**.

**Headline result.** **SR-SPR at replay ratio 16 reaches an Atari-100k interquartile-mean (IQM) human-normalized score of 0.631** — a new state of the art for model-free control on Atari 100k at the time. *Sourcing note:* the exact 0.631 figure is quoted here from the comparison table in the successor BBF paper (Schwarzer et al. 2023, verified via [ar5iv of arXiv:2305.19452](https://ar5iv.labs.arxiv.org/html/2305.19452)), because the original OpenReview PDF is bot-walled and could not be re-fetched. **[established, via BBF's table]**

**Why it works — the diagnostic that matters for us.** With resets, runs at *different* replay ratios converge to **similar TD errors, gradient norms, and parameter norms**; without resets, plain SPR's these metrics **diverge sharply across ratios.** In plain language: resets keep the network *well-conditioned and plastic* regardless of how hard you push the ratio, whereas an un-reset network progressively loses the ability to keep learning as the ratio climbs. This is the concrete, measurable signature of the barrier — and it tells us *what to log* if we ever push our own ratio up (see §4 guardrails and the implications section). **[established]**

**The successor sharpens the lesson.** **BBF** (Schwarzer et al. 2023, ICML) built directly on SR-SPR, swapping full resets for soft **"shrink-and-perturb"** resets and adding a larger network, and reached **IQM 1.045 at replay ratio 8** — i.e., *higher* performance at a *lower* replay ratio once the plasticity machinery is stronger. The arc SR-SPR → BBF is the whole thesis in miniature: better plasticity protection lets you get *more* out of *fewer* updates. **[established]**

**What this means for our decision.** A high replay ratio is only worth its compute cost **when it is paired with plasticity protection** (resets and/or normalization); the ratio is **not independently "good."** Two consequences for us: (i) if we ever wanted to push our ratio *up* to chase sample efficiency, we would need to add resets/shrink-and-perturb, not just turn the dial; (ii) going *lean* (lowering the ratio for speed, our actual proposal) safely **avoids** the barrier entirely — there is no plasticity pathology at low UTD — but it **forfeits sample efficiency**, which on our data-scarce task is the cost we must weigh. **[inference]**

### Why raising it eventually backfires — plasticity / capacity loss

The primacy bias is one face of a broader phenomenon: networks trained with many updates per sample **lose plasticity** (the ability to fit new targets) and **capacity** (effective rank / usable units).

- **Lyle, Zheng, Nikishin, Pires, Pascanu, Dabney (2023), "Understanding Plasticity in Neural Networks," ICML 2023.** ([PMLR](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf)). Plasticity loss intensifies with high replay ratios and small/fixed batches; **layer normalization** is a strong, cheap mitigation. **[established]**
- **Sokar, Agarwal, Castro, Evci (2023), "The Dormant Neuron Phenomenon in Deep RL,"** ([arXiv:2302.12902](https://arxiv.org/pdf/2302.12902)). High-UTD training drives a growing fraction of units to become inactive ("dormant"), throttling learning. **[established]**
- **Abbas, Zhao, Modayil, White, Machado (2023), "Loss of Plasticity in Continual Deep RL,"** ([PMLR v232](https://proceedings.mlr.press/v232/abbas23a/abbas23a.pdf)). Documents the progressive decline directly. **[established]**
- **Survey: Klein et al. (2024), "Plasticity Loss in Deep Reinforcement Learning: A Survey"** ([arXiv:2411.04832](https://arxiv.org/html/2411.04832v3)) — consolidates primacy bias, capacity loss, dormant neurons under one umbrella and catalogs mitigations (resets, LayerNorm, weight decay, regenerative regularization, larger nets). Good single entry point. **[established]**

### High replay ratio without resets — an important dissent

- **Hussing, Voelcker, Gilitschenski, Farahmand, Eaton (2024), "Dissecting Deep RL with High Update Ratios: Combatting Value Divergence," RLC 2024.** ([arXiv:2403.05996](https://arxiv.org/abs/2403.05996)). Argues the dominant failure at high UTD is not primacy bias per se but **value-function divergence** (Q/critic targets blowing up, propelled by optimizer momentum on unseen actions). They show you can train at high update ratios **without resets** if you control the divergence via **normalization** (unit-ball normalization on the penultimate features). Relevant to us because DreamerV3's critic uses a two-hot/symlog target and heavy normalization already — so DreamerV3 may sit on the *more forgiving* side of the barrier than a raw SAC critic. **[established]**

### And the state of the art bundles all of this

- **Schwarzer, Ceron, Courville, Bellemare, Agarwal, Castro (2023), "Bigger, Better, Faster: Human-level Atari with Human-level Efficiency" (BBF), ICML 2023.** ([arXiv:2305.19452](https://arxiv.org/abs/2305.19452)). The Atari-100k SOTA runs at **replay ratio 8** and only works because it *simultaneously* uses periodic soft resets (shrink-and-perturb), a larger CNN with residual connections, weight decay, and annealed n-step returns. Ablating the resets collapses the high-replay-ratio benefit. The lesson we care about: **the value of a high replay ratio is contingent on the accompanying plasticity machinery; the ratio is not independently "good."** **[established]**

**Synthesis of §1.** Higher replay ratio → better sample efficiency (fewer environment steps to reach a given return), *up to a barrier*, beyond which primacy bias / plasticity loss / value divergence dominate; the barrier can be pushed out with resets, normalization, weight decay, and bigger nets. Lower replay ratio → cheaper per environment step (our 3.2× speedup) but **less learning extracted per transition**, i.e., worse sample efficiency — the exact quantity we care about on a data-limited task. **[established]**

---

## 2. DreamerV3 specifically — what `train_ratio` is and how it is set

**Definition.** In DreamerV3, `train_ratio` = "the number of replayed steps per policy step" — deliberately defined against *policy* steps (i.e., unaware of action-repeat), not raw environment frames. Source: Hafner, Pasukonis, Ba, Lillicrap (2023/2024), "Mastering Diverse Domains through World Models," Appendix A ([ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2301.04104) · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) · [code](https://github.com/danijar/dreamerv3)). **[established]**

**This is the same quantity our `replay_ratio` measures**, up to units: sheeprl (which we port) defines `replay_ratio` as "the ratio between the gradient steps and the policy steps played by the agent," and computes gradient steps proportional to the number of parallel processes/envs ([sheeprl howto/work_with_steps.md](https://github.com/Eclectic-Sheep/sheeprl/blob/main/howto/work_with_steps.md)). That matches our `gradient-steps-per-iteration = replay_ratio × num_envs`. One caveat sheeprl flags: the replay ratio "does not account for the environment's action-repeat and `algo.learning_starts`" — so if we ever add action-repeat, the effective ratio shifts. **[established]**

**Per-benchmark values (Table A.1 of the paper, quoted via ar5iv):** **[established]**

| Benchmark | Data regime | Env instances | Train ratio |
|---|---|---|---|
| Atari 100k | tiny (400k frames) | 1 | **1024** |
| BSuite | tiny | 1 | 1024 |
| DMC proprio | small (500k) | 4 | 512 |
| DMC vision | small | 4 | 512 |
| Crafter | small | 1 | 512 |
| Atari 200M | large | 8 | **64** |
| DMLab | large | 8 | 64 |
| Minecraft | very large | 16 | **16** |

Batch size 16, batch length 64, held constant across all benchmarks.

**The single most important pattern for us is in that table.** Train ratio and data budget move in *opposite* directions: **data-scarce benchmarks use train ratios of 512–1024 with 1–4 envs; data-rich benchmarks use 16–64 with 8–16 envs.** DreamerV3's headline robustness claim ("150+ tasks with fixed hyperparameters") holds *the rest* of the config fixed but **explicitly tunes train ratio per benchmark by data regime.** So the paper itself treats replay ratio as the knob you set by how precious your data is — not as something to minimize for speed. **[established]**

Our gridworld survival task is a **small, data-scarce** task with a cheap simulator. By the paper's own placement it belongs near the **high** train-ratio end (the Crafter/DMC/Atari-100k neighborhood, 512–1024 in DreamerV3 units), *not* the Minecraft end (16). Note our absolute numbers are not directly comparable to DreamerV3's because our batch construction differs from theirs; what transfers is the *direction*, not the literal value. **[inference]**

**On follow-up ablations of `train_ratio`:** third-party work confirms the monotone-then-unstable shape. A DreamerV3-for-traffic-signal-control study ([arXiv:2503.02279](https://arxiv.org/pdf/2503.02279)) reports higher train ratios generally raise data efficiency but that "excessively high or low" values introduce instability, and that the viable window *narrows for larger models* (their L model tolerated only ~128). I did not find a clean, isolated `train_ratio` sweep-vs-final-return curve in the original DreamerV3 paper itself; the per-benchmark table and the traffic-control ablation are the closest published evidence. **[established, with the caveat that the cleanest ablation is third-party not first-party]**

---

## 3. Parallel environments in Dreamer / MBRL

Two distinct effects, often conflated:

**(a) Throughput.** More parallel envs amortize per-step Python/host overhead and feed the GPU larger collection batches; standard for maximizing wall-clock throughput ([sheeprl distributed docs](https://eclecticsheep.ai/page2/)). But — as we observed — **once the gradient-step compute dominates, adding envs stops helping wall-clock unless it also lets you *lower* the ratio.** In our code the coupling is explicit: gradient steps = `replay_ratio × num_envs`, so at fixed `replay_ratio`, adding envs adds *both* data and an equal share of gradient compute. That is precisely why we saw the GPU pinned by gradient steps and why dropping `replay_ratio` (not `num_envs`) is what bought the 3.2× speedup. **[established coupling in sheeprl/our port; the specific 3.2× is our in-house measurement]**

**(b) Sample efficiency and data quality.** More envs at a *fixed* replay ratio means more transitions collected before each update and a faster-refreshing buffer. General off-policy findings:
- There is a **non-monotone buffer-size / staleness trade-off**: too-recent data overfits, too-stale data (from outdated policies) slows convergence (Korniak, Czarnecki, As, Miłoś, Abbeel, Nauman 2026, *"When Does Non-Uniform Replay Matter in Reinforcement Learning?"* — a method paper on the recency/staleness question, [arXiv:2605.10236](https://arxiv.org/html/2605.10236v2)). **[established]**
- DreamerV3 sidesteps some of this because most gradient signal comes from **imagined rollouts in the world model**, not from raw replayed transitions — the world model is a learned interpolator over the buffer, which buffers (pun intended) the agent against replay staleness relative to a model-free SAC. **[inference]**

**Guidance on choosing `num_envs`.** DreamerV3's own scaling: data-rich benchmarks use more envs *and* lower train ratios together (8–16 envs at ratio 16–64); data-poor ones use 1–4 envs at ratio 512–1024. There is no free lunch from cranking envs alone. For a cheap simulator like ours, a **moderate `num_envs` (our 16 is reasonable) with the replay ratio set by data-hunger** is the configuration the literature supports. Adding envs beyond the point where the GPU is gradient-bound only helps if you *also* cut the ratio — which returns us to the §4 trade-off. **[inference from the DreamerV3 table + our own throughput measurement]**

---

## 4. Direct answer: should we expect a lower replay ratio to cost performance here?

**Expectation: yes, an aggressive cut should be expected to cost final survival-step performance, because our task is data-scarce and the replay ratio is the sample-efficiency knob.** This is consistent with (i) DreamerV3's own choice of *high* train ratios for small-data benchmarks, and (ii) the whole replay-ratio-barrier literature treating the ratio as the lever that *raises* sample efficiency (D'Oro 2023; Nikishin 2022; BBF 2023). Our prior in-project observation that lowering the ratio hurt is therefore the *expected* direction, not an artifact — and the sheeprl-parity overhaul does not change the direction of that effect, only (possibly) its magnitude. **[established direction + inference for our task]**

**But the magnitude is regime-dependent, and there are three reasons the penalty may be smaller than a naive reading suggests:** **[inference]**
1. **Diminishing returns near the top of the ratio.** The barrier literature says the *marginal* value of each extra update falls and eventually turns negative. If our current `replay_ratio = 1.0` is already past the knee for this small task, a cut toward 0.25–0.5 could be **nearly free** while still giving most of the throughput win.
2. **DreamerV3's imagination + normalization cushion.** Because DreamerV3 trains the actor-critic largely in imagination and already uses symlog/two-hot targets, LayerNorm-style normalization, and free-bits KL balancing, it is structurally closer to the "high-UTD-is-safe-with-normalization" regime of Hussing et al. (2024) than a bare SAC. That cushions the *pathology* side of high ratios — but it does **not** manufacture sample efficiency at low ratios. So the risk of going lean is under-training, not instability.
3. **Cheap simulator changes the economics.** The replay-ratio-barrier papers optimize *sample* efficiency because their environments are expensive. Our simulator is cheap, so we can partly compensate for a lower ratio by **collecting more environment steps** — trading the sample efficiency we lose for wall-clock we gained. The question becomes whether total wall-clock to a target survival-step level is lower at (low ratio, more steps) than at (high ratio, fewer steps). That is empirical and is exactly what the sweep should measure.

**Guardrails the literature recommends if we go lean or, conversely, want to safely push the ratio *up*:** **[established]**
- **Layer normalization** in the actor/critic/world-model MLPs — cheapest, most consistent plasticity protection (Lyle 2023; Hussing 2024). DreamerV3 already uses normalization heavily; confirm ours matches sheeprl.
- **Periodic soft resets** (shrink-and-perturb of last layers, buffer preserved) *if* we ever want to raise the ratio past its knee (Nikishin 2022; D'Oro 2023; BBF 2023). Not needed to go *leaner* — needed to go *higher* safely.
- **Weight decay** and **adequate network width** (BBF): larger nets tolerate higher ratios and get more data-efficient — but cost compute, cutting against our speed goal.
- **Watch value-target scale / critic divergence** at high ratios (Hussing 2024); watch **dormant-neuron fraction** as a plasticity early-warning (Sokar 2023).

---

## Implications for our 4-run replay_ratio sweep

1. **Measure performance against environment steps, not wall-clock.** The whole point is to separate the two effects the ratio bundles: sample efficiency (survival steps *per environment step*) vs. throughput (environment steps *per second*). Plot survival steps vs. environment steps for all four ratios on one axis, and survival steps vs. wall-clock on another. A ratio that looks bad per-env-step can still win per-wall-clock on a cheap sim. **[inference]**
2. **Expected shape:** survival-step-vs-env-step curves should **fan out by ratio**, with the highest ratio most sample-efficient and `0.0625` clearly worst per env step. If they *don't* separate — if `0.0625` matches `1.0` per env step — that is strong evidence we are already past the ratio knee and can safely run lean. That is the single most informative outcome to look for. **[inference]**
3. **Best-guess default:** an **intermediate ratio (~0.25–0.5)**, not either extreme. It keeps most of the throughput win (the per-gradient-step cost still dominates, so cutting from 1.0 → 0.5 already roughly halves gradient compute) while staying far from the aggressive-underfit regime `0.0625` risks on a data-scarce task. Treat `1.0` as the sample-efficiency ceiling reference and `0.0625` as the speed-ceiling reference; the production default should be whichever intermediate point sits at the elbow of the survival-vs-wall-clock curve. **[inference]**
4. **Confound to rule out (ties to the v8-null-cause hygiene):** if any prior world-model run used a low ratio "for speed," re-examine whether it was simply **under-trained per environment step** — a policy-gradient-side confound that could masquerade as a modeling failure. Flag it to whoever owns the v8 diagnosis. **[inference]**
5. **Plasticity instrumentation (cheap, high-value):** log **dormant-neuron fraction** and **critic/value-target scale** across the sweep. If the high-ratio runs show rising dormant fractions or drifting value scale, that both explains any high-ratio plateau and tells us whether adding LayerNorm/resets would let us push the ratio further. **[established metrics, inference on our benefit]**
6. **Don't co-vary `num_envs` and `replay_ratio` in the same sweep** unless deliberately studying the interaction — because gradient steps = `replay_ratio × num_envs`, changing both at once confounds the compute budget. Hold `num_envs = 16` fixed for the ratio sweep. **[inference]**

---

## Open questions

- Where is *our* task's replay-ratio knee? Only the sweep answers this; the literature only tells us the knee exists and its direction.
- Does our DreamerV3 port's normalization match sheeprl/danijar closely enough to inherit the "high-UTD-is-safe" cushion (Hussing 2024)? A parity check on the actor/critic/world-model norm layers would tell us whether resets would ever be needed. → route to `code-reviewer` / `math-reviewer`.
- Is the DreamerV3 imagination-horizon (λ-return in latent space) itself sensitive to replay ratio? Plausibly the model quality (which the ratio affects) matters more than the actor-critic update count on top of a fixed model — worth a targeted look if the sweep is ambiguous. **[inference]**

---

## Next steps

- **experiment-designer** — the 4-run `replay_ratio ∈ {1.0, 0.5, 0.25, 0.0625}` sweep at fixed `num_envs = 16`; pre-register survival-steps-vs-env-steps *and* -vs-wall-clock as the two verdict axes; add dormant-neuron-fraction and value-target-scale logging if cheap.
- **code-reviewer / math-reviewer** — confirm our port's normalization (LayerNorm placement, symlog/two-hot critic) matches sheeprl/danijar, since that governs how far up the ratio we could safely push.
- **senior-developer** — the change to run the sweep is contained (a config-level ratio swap); no cross-cutting code needed. Any dormant-neuron logging is an additive metric hook.
- **pi** — if the sweep is a pre-launch of a multi-run comparison, this is a focus-vs-explore decision point (lean-for-speed vs. sample-efficient-default).

## References named (add PDFs under `docs/project/references/`)

- Hafner et al. 2023/24, *Mastering Diverse Domains through World Models* (DreamerV3) — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- D'Oro et al. 2023, *Breaking the Replay Ratio Barrier* (SR-SPR) — **ICLR 2023 oral (Notable-top-5%), OpenReview-only, no arXiv version** — [OpenReview id OpC-9aBBVJe](https://openreview.net/forum?id=OpC-9aBBVJe) · [iclr.cc oral](https://iclr.cc/virtual/2023/oral/12655)
- Nikishin et al. 2022, *The Primacy Bias in Deep RL* — [arXiv:2205.07802](https://arxiv.org/abs/2205.07802)
- Schwarzer et al. 2023, *Bigger, Better, Faster* (BBF) — [arXiv:2305.19452](https://arxiv.org/abs/2305.19452)
- Lyle et al. 2023, *Understanding Plasticity in Neural Networks* — [PMLR v202](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf) · [arXiv:2303.01486](https://arxiv.org/abs/2303.01486)
- Sokar et al. 2023, *The Dormant Neuron Phenomenon in Deep RL* — [arXiv:2302.12902](https://arxiv.org/abs/2302.12902)
- Abbas et al. 2023, *Loss of Plasticity in Continual Deep RL* — [PMLR v232](https://proceedings.mlr.press/v232/abbas23a/abbas23a.pdf) · [arXiv:2303.07507](https://arxiv.org/abs/2303.07507)
- Klein et al. 2024, *Plasticity Loss in Deep RL: A Survey* — [arXiv:2411.04832](https://arxiv.org/abs/2411.04832)
- Hussing et al. 2024, *Dissecting Deep RL with High Update Ratios: Combatting Value Divergence* — [arXiv:2403.05996](https://arxiv.org/abs/2403.05996)
- sheeprl steps semantics — [howto/work_with_steps.md](https://github.com/Eclectic-Sheep/sheeprl/blob/main/howto/work_with_steps.md)
