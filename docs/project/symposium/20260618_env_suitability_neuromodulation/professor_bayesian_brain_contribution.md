---
title: "Symposium contribution — Bayesian-brain / precision lens on whether the env can test the four Doya-2002 neuromodulator knobs"
session: 20260618_env_suitability_neuromodulation
contributor: professor-bayesian-brain
date: 2026-06-18
focus_knobs:
  - "ACh → effective learning rate (expected uncertainty / volatility) — primary"
  - "NA → effective temperature (unexpected uncertainty / regime change) — primary"
  - "DA → effective TD-gain (precision-coding / variance) — brief"
  - "5-HT → effective discount (precision of prior over future states) — brief"
inputs:
  - docs/project/symposium/20260618_env_suitability_neuromodulation/00_env_capability_grounding.md
canonical_framework: "Yu & Dayan 2005 (expected vs unexpected uncertainty); Behrens et al. 2007 (volatility learning)"
---

# Symposium contribution — Bayesian-brain / precision lens

## Plain-English entry point (read first)

The symposium asks whether our pain grid world can test four classic ideas, each pairing a brain chemical with a learning "knob": noradrenaline (NA) with how exploratory the agent is, acetylcholine (ACh) with how fast it updates its beliefs, dopamine (DA) with how strongly reward-surprises drive learning, and serotonin (5-HT) with how patient it is. My job is the **uncertainty** half of this picture, because two of those four chemicals are, in the theory I work in, fundamentally about *uncertainty*.

The key idea is a distinction made by Yu & Dayan in 2005 between two kinds of uncertainty:

- **Expected uncertainty** — the *known, familiar* noisiness of a situation you understand well. You know the slot machine pays out roughly one time in three; the outcome of any single pull is uncertain, but the *rule* is stable and you trust your model of it. In the theory, **acetylcholine (ACh)** signals this kind of uncertainty.
- **Unexpected uncertainty** — *surprise that your model itself is wrong*. The slot machine that paid one-in-three suddenly pays one-in-ten: the world's rules changed, and your old model no longer applies. This is a **regime change** or **change-point**. In the theory, **noradrenaline (NA)** signals this kind of uncertainty.

A third recurring term is **precision** — just the inverse of uncertainty (how much you should *trust* a given stream of information). High precision = low noise = listen closely; low precision = noisy = discount it. Neuromodulators, in the Bayesian-brain tradition, are read as setting precisions. A related term is **volatility** — *how often the rules change*. When volatility is high, you should learn fast (each new observation matters because the world keeps moving); when volatility is low, you should learn slowly (average over many observations because the world is stable). Behrens et al. 2007 showed humans track volatility and raise their learning rate when it rises — this is the empirical heart of the ACh-as-learning-rate idea.

**My headline, arguing strictly from the shared grounding memo's facts:**

1. **ACh (expected uncertainty / learning rate) — the grounding memo's PARTIAL is, through my lens, closer to NOT SUPPORTED *as a clean test of ACh's actual construct*.** The env only has i.i.d. per-episode re-sampling of predator/resource parameters. That is *not* an expected-uncertainty manipulation and it is *not* a volatility manipulation — it is unstructured noise with no learnable rate-of-change. ACh's construct *requires* a structured volatility schedule (stable blocks vs. volatile blocks, or scheduled change-points) so the agent can *estimate volatility itself*. The env lacks this, and I specify exactly what it would need.

2. **NA (unexpected uncertainty / temperature) — the grounding memo's SUPPORTED is correct for the *arousal* reading but only PARTIAL for the *Yu-Dayan construct*.** Graded predator threat gives graded arousal, which is enough for an inverse-U exploration sweep. But NA in Yu & Dayan is specifically *unexpected uncertainty* — model-violating surprise — and continuous proximity-driven threat is *expected* once the agent has learned the predator. The env *can* deliver genuine unexpected uncertainty, but only via **predator regime-change events** that the env does not currently schedule.

3. **The crucial dissociation question:** ACh and NA are **confounded, not dissociable, in the env as it stands** — because the only mechanism (per-episode re-sampling) muddles "this context is reliably noisy" (ACh) with "the context just changed" (NA). Cleanly dissociating them requires the *same* missing machinery: a controllable volatility schedule with labelled change-points. Build that one thing and you get both knobs as distinct constructs; without it, neither is clean.

4. **DA and 5-HT (brief):** DA — per-context return-distribution *shape* is a genuine precision-coding signature, and a cost-variance task gives a usable (if imperfect) aleatoric-vs-epistemic separation. 5-HT — the active-inference reading offers a **precision-based route that sidesteps the env's missing reward-delay mechanism**: effective horizon is set by the *precision of the prior over future states*, which the interoceptive homeostatic prior can modulate without any reward-delay queue.

The formal arguments, the volatility-schedule specification, and the dissociation analysis are below.

---

## §A — Why "i.i.d. per-episode re-sampling" is not an ACh manipulation (the primary argument)

This is my most important contribution, so I will be precise. The grounding memo (§4.2) rates ACh → learning-rate **PARTIAL**, on the grounds that the *ingredients* of variability are config-reachable (per-episode re-sampling of five predator parameters from uniform ranges; per-respawn re-draw of resource chemical signatures) even though a *designed* volatility schedule is absent. I agree with every fact in that section. I disagree with the implied reading that "more variability ≈ closer to an ACh test." Through the precision / volatility lens, **adding i.i.d. variability moves the env in the wrong direction for ACh** and, if anything, contaminates the NA construct. Here is why.

### A.1 — What ACh's construct formally requires

In the Yu & Dayan 2005 framework, and made quantitative by Behrens et al. 2007, the agent runs a *hierarchical* generative model with (at least) three levels:

$$
\underbrace{v}_{\text{volatility}} \;\to\; \underbrace{r}_{\text{current rate / contingency}} \;\to\; \underbrace{o_t}_{\text{observation}}.
$$

The contingency $r$ (e.g. "probability this tile is dangerous", "predator detection radius") drifts over time, and the *speed of that drift* is governed by the volatility $v$:

$$
r_{t} \mid r_{t-1}, v_t \;\sim\; \mathcal{N}\!\big(r_{t-1},\, \exp(v_t)\big), \qquad
v_t \mid v_{t-1} \;\sim\; \mathcal{N}\!\big(v_{t-1},\, k\big).
$$

The Bayesian-optimal learning rate $\alpha_t$ for updating the belief about $r$ is an **increasing function of the inferred volatility** $\hat v_t$:

$$
\alpha_t \;=\; g\!\big(\hat v_t\big), \qquad \frac{\partial \alpha_t}{\partial \hat v_t} > 0.
$$

ACh is hypothesised to *report* $\hat v_t$ (expected uncertainty), and the learning-rate-up-when-volatile behaviour is the signature. **The entire construct depends on $v$ being a real, inferable, time-structured quantity.** The agent must be able to *estimate how volatile the world is* by watching how often the contingency changes, and then *act on that estimate* by changing its learning rate.

### A.2 — Why i.i.d. per-episode re-sampling has no inferable volatility

Per the grounding memo (§4.2, citing `07_predator_ai.md:89-100`), at every reset the five predator parameters are drawn fresh and independently from fixed uniform ranges $[\text{lo},\text{hi}]$. Formally, this is

$$
\theta^{(e)} \;\overset{\text{i.i.d.}}{\sim}\; \mathrm{Unif}[\text{lo}, \text{hi}] \quad \text{for each episode } e,
$$

with **no carry-over** $\theta^{(e)}$ from one episode to the next and **no latent process** governing the draws. This breaks the ACh construct in three distinct ways:

1. **There is no volatility level to infer.** Volatility $v$ is *the variance of the step-to-step drift of the contingency*. Under i.i.d. resampling there is no drift — each episode is an independent draw — so $v$ is undefined; the only inferable quantity is the *width* of the fixed sampling range, which is a constant the agent learns once and never needs to re-estimate. Once the agent has learned $[\text{lo},\text{hi}]$, the Bayes-optimal learning rate for *that range* is **constant**, not time-varying. There is no $\hat v_t$ trajectory for ACh to track.

2. **It is expected uncertainty *of the wrong shape*, and unestimable across episodes.** Within a single episode the predator parameters are *fixed* (drawn once at reset), so there is no within-episode contingency drift at all. Across episodes the agent never carries belief state (episodes are independent in the resampling), so it cannot accumulate the "how often does this change" estimate that defines volatility tracking. The variability is real but it is **structureless** — closer to fixed observation noise than to either expected *or* unexpected uncertainty in the Yu-Dayan sense.

3. **It confounds with NA (see §C).** From inside a single episode, an agent that *had* learned a predator model in a previous episode and is now in a fresh episode with re-drawn parameters experiences this as a *regime change* — which is the NA / unexpected-uncertainty construct, not the ACh one. So the one mechanism the env offers loads on *both* knobs at once and cleanly separates neither.

**Verdict (my lens, sharper than the grounding memo's PARTIAL):** for testing ACh's actual construct — volatility-driven learning-rate modulation — the env is effectively **NOT SUPPORTED**. The PARTIAL rating is correct about config-reachability of *variability*; it is too generous about whether that variability instantiates *expected uncertainty / volatility*. I would relabel this knob, in the symposium's summary, as **"PARTIAL ingredients, NOT SUPPORTED construct"** to avoid the trap of someone sweeping `detection_range: [lo, hi]` ranges wider and believing they have built an ACh paradigm. They will have built a noisier i.i.d. env, which is the opposite of what ACh needs.

### A.3 — What a Behrens-style volatility schedule would need in *this* env

The env needs a **latent contingency that drifts at a controllable, schedulable rate**, with the rate itself being the manipulated variable. Concretely, the minimal new machinery (hand-off to `senior-developer` / `experiment-designer` — I do not write code):

- **A persistent, slowly-drifting contingency parameter on `EnvState`.** Pick one ecologically meaningful contingency the agent must track — the cleanest candidate given the env's primitives is the **smell→danger or smell→food mapping** (the resource/predator *chemical signature*, already re-drawn on respawn per `08_resources_and_obstacles.md:165-167`). Instead of an i.i.d. re-draw, make it a **random walk that persists across steps/episodes**: $c_t = c_{t-1} + \eta_t$, $\eta_t \sim \mathcal N(0,\sigma^2_{\text{drift}})$.
- **A schedulable drift variance (the volatility knob).** Expose $\sigma^2_{\text{drift}}$ as a value that switches on a **block schedule** keyed on an episode counter or `current_step`: a **stable block** ($\sigma^2_{\text{drift}}$ small, contingency nearly fixed for, say, 20 episodes) followed by a **volatile block** ($\sigma^2_{\text{drift}}$ large, contingency reshuffles every few episodes). This is the exact Behrens 2007 design.
- **Optionally, scheduled change-points** as the discrete-jump variant: at known episodes, flip the smell→danger contingency (e.g. the previously-safe chemical signature becomes the predator's). Discrete change-points are the simplest instantiation and double as the NA probe (§B).
- **Logged ground truth.** The env must log the true $\sigma^2_{\text{drift}}$ schedule and the change-point times so the analyst can test "is the modulator's effective learning rate higher in volatile blocks than stable blocks?" — the falsifiable ACh signature.

The grounding memo already names the architectural template (a scheduled parameter switch keyed on `current_step` or an episode counter, §4.2). My contribution is the *theoretical specification of what makes it an ACh test rather than just a switch*: it must be a **two-level process** (a contingency that drifts, and a drift-rate that is itself scheduled), with the **drift-rate as the manipulated variable**, because ACh tracks the rate-of-change, not the change itself. A single one-off switch tests NA (unexpected uncertainty); a *schedule of how-fast-things-change* tests ACh (expected uncertainty / volatility). This distinction is the whole game and it is why the two knobs need related-but-distinct probes (§C).

---

## §B — NA → temperature: arousal is supported, but the Yu-Dayan construct needs regime-change events

### B.1 — Two readings of NA, and which one the env supports

The grounding memo (§4.1) rates NA → temperature **SUPPORTED** via graded predator threat, and as an *arousal* manipulation this is correct: predator proximity varies continuously, is perceptible (smell, vision, contact pain), and a Yerkes-Dodson inverse-U exploration sweep across detection-radius configs is genuinely buildable. **For the weak reading of NA — arousal sets exploration temperature — I concur: SUPPORTED.**

But Yu & Dayan 2005 give NA a *specific* job that is narrower than arousal: NA reports **unexpected uncertainty** — the signal that *the current generative model has been violated* and should be discarded in favour of relearning. Formally, NA fires when the likelihood of the observation under the current model collapses:

$$
\text{NA}_t \;\propto\; -\log p\big(o_t \mid \hat r_{t}, \text{current model}\big) \;\;\text{above a surprise threshold,}
$$

triggering a *reset* of the belief about $r$ (a large effective learning-rate transient) and an exploration burst. The defining property is **model violation**, not stimulus intensity.

### B.2 — Is proximity-driven threat "unexpected uncertainty"? Mostly no.

Here is the subtlety the grounding memo's SUPPORTED rating glosses. A predator that approaches along a learned pursuit FSM (`07_predator_ai.md`) produces threat that, once the agent has learned the predator's behaviour, is **expected** — the agent *predicts* the threat from the predator's position and its own. Rising threat as the predator nears is high *arousal* but low *surprise*: the agent's model is being *confirmed*, not violated. So the steady-state threat gradient drives the *arousal/temperature* reading but **not** the Yu-Dayan unexpected-uncertainty reading. Treating proximity threat as an NA-construct probe would mislabel an expected, model-consistent signal as model-violating surprise.

### B.3 — Where the env *can* deliver genuine unexpected uncertainty

The env has exactly the right raw material for true NA probes — it just does not currently *schedule* them as labelled events:

- **Predator regime-change events.** A predator whose behaviour *changes* mid-episode in a way the agent's learned model does not predict — e.g. detection radius jumps, or a previously-patrolling neutral animal *switches* into a predator (the env has both neutral animals and predators; `08`/`07`). The *first* encounter after the switch is genuine model violation: the agent's "this entity is safe / this predator hunts to range 3" prediction collapses. **This is the cleanest in-env NA-construct probe**, and it is the *same* discrete change-point machinery §A.3 needs.
- **Smell→danger contingency flips** (the §A.3 change-point variant): the chemical signature that meant "food" now precedes a trap. The first few post-flip contacts are unexpected uncertainty.

So NA's full construct is **PARTIAL → reachable**, on the *same* missing primitive as ACh: a scheduled, labelled change-point. The difference is in *how you schedule it and what you measure*:

| | ACh probe (§A.3) | NA probe (§B.3) |
|---|---|---|
| Schedule | **many** change-points at a controllable *rate* (volatility blocks) | **isolated, surprising** change-points after a stable model has formed |
| Manipulated variable | drift-rate $\sigma^2_{\text{drift}}$ / change frequency | model-violation magnitude (surprise) of a single event |
| Signature | effective learning rate higher in volatile blocks | transient exploration burst + learning-rate spike *at* the violation, scaling with surprise not arousal |
| What it isolates | expected uncertainty (stable noise structure) | unexpected uncertainty (model breakdown) |

### B.4 — A discriminator that distinguishes NA-construct from mere arousal

To show the modulator tracks *unexpected* uncertainty and not just arousal, construct two events with **matched arousal but mismatched surprise**: (a) a predator that approaches predictably to contact (high arousal, low surprise — the agent's model holds); (b) a neutral animal that *suddenly* turns predatory at the same distance (matched arousal at contact, high surprise — model violation). The NA-construct prediction is that the modulator burst is **larger for (b)**, and that the burst at (b) carries a learning-rate transient (faster post-event belief update) absent in (a). A pure arousal/temperature reading predicts equal bursts. This is a one-task-variant experiment that rides on the same regime-change machinery and gives the symposium a clean NA-construct test. (Hand-off: `experiment-designer`.)

---

## §C — Are ACh and NA dissociable in this env? (the crucial question)

**As the env stands today: no — they are confounded.** As the env *could* be, with the §A.3 machinery: **yes — cleanly dissociable.** The argument:

### C.1 — Why they are confounded now

The env's only source of changing statistics is i.i.d. per-episode re-sampling (§A.2). This single mechanism is **simultaneously**:

- an (ill-formed) expected-uncertainty signal — "predator parameters vary within a known range" (ACh-flavoured), and
- an unexpected-uncertainty signal — "this episode's predator differs from last episode's, model violated at reset" (NA-flavoured).

Because the two uncertainties ride on the *same* generative event (the reset re-draw) with **no parameter that moves one without the other**, any modulator response to it is unidentifiable: you cannot tell whether the modulator is reporting "this context is reliably noisy" (ACh) or "the context just changed" (NA). Yu & Dayan's whole contribution was that these are *separable* signals with *opposite optimal responses* in one regime (high expected uncertainty → *don't* reset your model; high unexpected uncertainty → *do* reset it). An env that fuses them into one knob cannot test the dissociation that defines the framework.

### C.2 — Why the §A.3 machinery dissociates them — by construction

The two-level volatility process $v \to r \to o$ has **two independent dials**, and that is exactly what dissociation requires:

- **Expected uncertainty (ACh) = the noise within a stable model** — controlled by the *observation noise / likelihood width* around the current contingency $r$. Crank this up with the contingency *held stable* (stable block, $\sigma^2_{\text{drift}} \approx 0$): the world is reliably noisy, model intact. Optimal response: low learning rate, integrate over noise.
- **Unexpected uncertainty (NA) = the rate/size of contingency change** — controlled by $\sigma^2_{\text{drift}}$ / the change-point magnitude. Crank this up: the model breaks. Optimal response: high transient learning rate, reset.

A $2 \times 2$ design — {stable, volatile} × {low-noise, high-noise} — orthogonalises them. The dissociation signatures:

$$
\frac{\partial(\text{eff. learning rate})}{\partial\,\sigma^2_{\text{drift}}} > 0 \quad (\text{NA / unexpected}), \qquad
\frac{\partial(\text{eff. learning rate})}{\partial\,\sigma^2_{\text{obs-noise}}} \le 0 \quad (\text{ACh / expected, opposite sign}).
$$

**The opposite-sign prediction is the killer test:** expected uncertainty should *lower* the learning rate (more noise to average out under a stable model), while unexpected uncertainty should *raise* it (the model needs replacing). If the modulator's interoceptive-conditioned effective learning rate shows this **sign reversal** across the two dials, the env has dissociated ACh from NA. If it responds identically (or monotonically) to both, it has not — and the modulator is a single arousal/surprise scalar with two labels, the same single-channel risk flagged in the prior symposium round.

### C.3 — Consequence for the symposium

The dissociability of ACh and NA is **gated on one build**: the controllable volatility schedule with a separately-tunable observation-noise width. This is a single coherent piece of new env code (the §A.3 spec), and it *simultaneously* unlocks the proper ACh construct, the proper NA construct, AND their dissociation. That is a strong return on one investment and I would rank it the **highest-value env extension** the symposium could recommend — above the 5-HT reward-delay queue, because it converts *two* knobs from "ingredients only" to "construct-testable and mutually-dissociable," and because the dissociation is the part of the Yu-Dayan framework with genuine theoretical and clinical (computational-psychiatry) bite.

---

## §D — DA → TD-gain (brief): return-distribution shape is a precision signature

The grounding memo (§4.3) rates DA → TD-gain **PARTIAL**: reward-cost variance is reachable via per-contact `damage [min,max]` ranges and spatial layout, but food reward itself is fixed and there is no single "reward σ" knob. Two precision-lens comments:

1. **Per-context return-distribution *shape* is a genuine precision-coding signature**, and the env can express it. In distributional / precision terms, a region with `damage:[10,10]` has a sharp (high-precision) return distribution; `damage:[0,40]` with matched mean has a flat (low-precision) one. If the modulator's effective TD-gain (or, in a distributional-RL reading, the spread of the value-head's predicted return distribution) *tracks the precision of the realised return distribution per context*, that is the DA-as-precision signature — and it is measurable on already-constructible cost-variance layouts. (This overlaps `professor-rl`'s distributional-RL territory; I am only flagging the precision interpretation.)

2. **Aleatoric vs. epistemic separation is *partial but usable*.** A cost-variance task built on `damage [min,max]` gives clean **aleatoric** uncertainty (irreducible per-contact randomness — the damage genuinely is a random draw, so no amount of learning sharpens it). It does *not* natively give **epistemic** uncertainty (reducible, model-ignorance uncertainty) unless you *also* withhold information — e.g. hide which region is high-variance behind an initially-ambiguous smell signature, so the agent's uncertainty about *which distribution it faces* shrinks with experience. Combine the cost-variance layout with the partially-latent-`c` trick the grounding memo notes (injury/nutrition gateable out of obs, §5) and you get a task where aleatoric (fixed damage spread) and epistemic (learnable region identity) are separable. This is a config-design exercise, not new code, but it is non-trivial — hand-off to `experiment-designer`, with `professor-bayesian-nn` for the aleatoric/epistemic decomposition machinery on the value head.

DA verdict from my lens: the grounding memo's **PARTIAL** stands; the precision reading adds that the *shape* (not just mean) of the per-context return distribution is the load-bearing measurable, and that a deliberately ambiguous-then-resolved region identity is what buys the epistemic half of the decomposition.

---

## §E — 5-HT → discount (brief): a precision-of-prior route that sidesteps reward delay

The grounding memo (§4.4) rates 5-HT → discount **NOT SUPPORTED**: reward is paid on the same step the food is eaten, and the two delay-named mechanisms (respawn timer, nociception kernel) are not reward delays. Building a reward-delay queue is the proposed fix. The active-inference lens offers a **different route that does not need the queue.**

In active inference, the agent does not discount a temporally-delayed scalar reward; it acts to fulfil a **prior preference over future states** $p(o_{\tau} \mid C)$ extending over a horizon $\tau \in \{t{+}1, \dots, t{+}H\}$. The **effective horizon $H$ is set by the precision of that prior over future states** — how sharply the agent's generative model commits to predictions about the far future — not by a discount applied to a literal reward latency:

$$
G(\pi) \;=\; \sum_{\tau=t+1}^{t+H} \mathbb{E}_{q(o_\tau\mid\pi)}\big[\ln q(o_\tau\mid\pi) - \ln p(o_\tau\mid C)\big], \qquad H \;\text{effective} \;\propto\; \text{precision of }p(o_{\tau}\mid C).
$$

When the prior over future states is **imprecise** (the agent cannot confidently predict far-future states), distant terms wash out and the effective horizon is *short* — behaviourally identical to a low discount factor. When it is **precise**, distant states carry weight and the effective horizon is *long*. **This gives a precision-based 5-HT signature that the env can express without a reward-delay primitive:** the interoceptive homeostatic prior is exactly a prior over future body states. An agent near a homeostatic boundary (low satiation, high injury — death imminent) *should* have a short effective horizon (the far future is moot if you die soon); an agent in homeostatic comfort can afford a long horizon. If a modulator conditioned on interoceptive state $c$ sharpens/flattens the precision of the prior over future *body* states as a function of homeostatic urgency, that is a 5-HT-discount signature realised through **prior precision**, not reward latency.

**5-HT verdict from my lens:** the grounding memo's **NOT SUPPORTED** is correct *for the literal-reward-delay operationalisation*. The active-inference reframing offers a **precision-of-prior route that is config-reachable today** — it rides on the existing interoceptive channels (satiation, injury, the nociception trace; grounding memo §5) and the existing death-by-homeostasis dynamics, with no reward-delay queue. The cost is that it is a *different* operationalisation of "discount" (prior-precision-driven effective horizon vs. temporal reward discounting); the symposium should decide whether that counts as testing the 5-HT function or as testing a *related* construct. My recommendation: present it as a **partial, construct-shifted route** — honestly flagged as not the Doya temporal-discount operationalisation, but a defensible Bayesian-brain alternative that needs no new env code. Cross-reference `professor-rl` for whether the effective-horizon-from-prior-precision reading maps onto the project's actual return estimator.

---

## §F — Summary verdict table (my lens vs. grounding memo)

| Knob | Grounding memo | My (precision/uncertainty) verdict | Key reason |
|---|---|---|---|
| **ACh → learning rate** (expected uncertainty) | PARTIAL | **PARTIAL ingredients, NOT SUPPORTED construct** | i.i.d. resampling has no inferable volatility; ACh needs a two-level drift-rate-scheduled process (§A) |
| **NA → temperature** (unexpected uncertainty) | SUPPORTED | **SUPPORTED (arousal) / PARTIAL (Yu-Dayan construct)** | proximity threat is *expected*; true unexpected uncertainty needs scheduled regime-change events (§B) |
| **ACh ⊥ NA dissociation** | (not rated) | **CONFOUNDED now / DISSOCIABLE with §A.3 build** | one mechanism fuses both; a 2×2 volatility×noise design with opposite-sign learning-rate predictions separates them (§C) |
| **DA → TD-gain** (precision/variance) | PARTIAL | **PARTIAL (concur)** | return-distribution *shape* is the precision signature; epistemic half needs ambiguous-then-resolved region identity (§D) |
| **5-HT → discount** (prior precision) | NOT SUPPORTED | **NOT SUPPORTED (literal delay) / PARTIAL (prior-precision route, no new code)** | effective horizon ∝ precision of prior over future body states; rides on existing interoceptive channels (§E) |

---

## §G — School / framework caveat

This memo reads from the **Yu & Dayan 2005 two-uncertainty** framework (the natural home for ACh/NA) and the **Behrens 2007 volatility-learning** model, with the 5-HT section drawing on **active inference** (Friston-tradition expected-free-energy). These are related but not identical formal systems. A reviewer committed to a pure Doya-2002 reading (where ACh→α and NA→β are *fixed functional assignments* rather than uncertainty-derived) would not require the volatility-schedule machinery — but would also not be able to claim the env tests the *uncertainty-theoretic* content that gives ACh and NA their construct validity. My position is that the uncertainty-theoretic content is the part worth testing (it is what carries the computational-psychiatry impact), and it is precisely the part the env most lacks. State the framework choice explicitly in any paper that uses these knobs.

---

## Hand-offs

- **`research-postdoc`** — owns the symposium synthesis. The load-bearing inputs from my side: (1) ACh's construct is effectively NOT SUPPORTED, not PARTIAL — do not let "wider uniform ranges" be mistaken for a volatility manipulation; (2) ACh and NA are confounded now but dissociable with one shared build; (3) that shared build (§A.3 controllable volatility schedule + separable observation-noise width) is the **single highest-value env extension** because it unlocks two construct tests plus their dissociation.
- **`senior-developer`** — the §A.3 spec: a persistent slowly-drifting contingency on `EnvState` (random walk on the smell→danger/food chemical mapping), a **schedulable drift-variance** (volatility knob) keyed on episode/step counter with stable→volatile blocks, optional discrete scheduled change-points, plus logged ground-truth schedule. Injury ring buffer (`05_body_homeostasis.md:118-138`) is a structural template per the grounding memo. I do not write code; this is the recommendation.
- **`experiment-designer`** — three designs ride on the §A.3 build: the ACh volatility-block learning-rate test (§A.3), the NA matched-arousal-mismatched-surprise discriminator (§B.4), and the 2×2 volatility×noise dissociation with the opposite-sign learning-rate prediction (§C.2). The DA aleatoric/epistemic separation (§D) and the 5-HT prior-precision route (§E) are config-only, no new code.
- **`professor-rl`** — whether the effective-horizon-from-prior-precision reading (§E) maps onto the project's actual return estimator; and the distributional-RL reading of per-context return-distribution shape (§D).
- **`professor-bayesian-nn`** — the aleatoric/epistemic decomposition machinery on the value head for the §D DA cost-variance task.
- **`professor-neuromodulation`** — likely agrees on the *variable* (ACh ↔ expected uncertainty / gain on stable contingencies; NA ↔ unexpected-uncertainty-driven network reset) but from receptor-kinetics grounds; cross-reference on whether the biological NA-reset literature supports the discrete-change-point NA probe over the graded-arousal one.

---

*Contribution by `professor-bayesian-brain`, 2026-06-18. Arguments built on the shared grounding memo `00_env_capability_grounding.md`; no architecture authored here.*
