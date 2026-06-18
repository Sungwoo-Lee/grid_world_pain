---
title: "Symposium contribution — pain / interoception construct-validity verdict on the four-neuromodulator env-suitability question"
symposium_session: 20260618_env_suitability_neuromodulation
contributor: professor-pain-modeling
date: 2026-06-18
inputs:
  - docs/project/symposium/20260618_env_suitability_neuromodulation/00_env_capability_grounding.md
  - docs/project/concepts/pain_vs_nociception_construct.md
  - docs/project/symposium/20260516_impact_vs_reasonable/professor_pain_modeling_contribution.md
knobs:
  - "NA → effective temperature (arousal / vigilance)"
  - "ACh → effective learning rate (expected uncertainty)"
  - "DA → effective TD-gain (wanting / risk)"
  - "5-HT → effective discount (horizon / myopia)"
role: pain-construct-validity guardian for the bridge "interoceptive state c drives neuromodulation"
---

# Symposium contribution — does the pain grid world's interoceptive state validly drive the four neuromodulator knobs?

## Plain-English entry point

This symposium asks whether our existing pain grid world can test four classic links between brain chemicals and learning settings (Doya 2002): noradrenaline setting how exploratory the agent is, acetylcholine setting how fast it updates beliefs, dopamine setting how strongly reward-surprises drive learning, and serotonin setting how far ahead the agent plans. A shared grounding memo already gave the engineering verdict: the noradrenaline knob is testable today, the acetylcholine and dopamine knobs are reachable with config work, and the serotonin knob is not testable without new code because the environment pays reward on the same step the food is eaten — there is no way to delay the reward.

My job is different from the other three professors'. The whole premise of this project is a **bridge**: the agent's *internal bodily state* (how full it is, how recently it was hurt) is what drives the chemical knobs. I am the guardian of whether that bridge is honest **specifically for pain**. The environment gives the modulator four internal channels to read: how injured the agent is, how nourished it is, how full it feels, and a delayed "recent pain" trace that smears past injuries forward in time. The question I answer, knob by knob, is: when we wire one of these channels into a knob and call the result "pain-like", is that a faithful claim from the pain-science literature, or are we dressing up generic aversion in clinical language?

My headline verdicts, in plain terms:

1. **Noradrenaline / vigilance — VALID, and this is the project's strongest pain claim.** "Recent pain raises arousal and sharpens vigilance" is one of the most robust findings in the clinical pain literature. The environment has a real threat source (predators), a real injury signal, and a delayed "recent pain" channel — exactly the ingredients the pain-arousal literature requires. I make the construct-validity case in full, and flag the one place it overreaches.
2. **Acetylcholine / belief-updating-speed — CONSTRUCT-MISMATCH as currently built.** The environment's randomness is reshuffled fresh every episode like a deck of cards, with no stable-then-volatile structure. That is not the kind of *pain-relevant* uncertainty (is this place dangerous? is my body healing or getting worse?) that the pain literature ties to belief-updating. The ingredients are too impoverished to be a pain manipulation.
3. **Dopamine / risk — VALID-WITH-CAVEATS, and more pain-relevant than the standard version.** The standard dopamine test uses *reward* gambles. But our environment's tunable variance is on the *damage* side — risky versus safe paths that differ in how much pain they expose you to. "Risk of harm" is a genuinely pain-relevant construct (it is the engine of the fear-avoidance model). The caveat is real and important: we must not conflate the appetitive dopamine system with the aversive system, and I spell out the trap.
4. **Serotonin / planning-horizon — the red verdict is SOFTENABLE on pain-construct grounds.** The grounding memo correctly says we cannot *delay reward*. But the clinical finding is not "pain delays reward" — it is "**pain makes you myopic**". Chronic pain robustly collapses people's planning horizon. That is an *interoceptive-state → horizon* link, and it does not require a reward-delay primitive at all. I propose a pain-faithful way to elicit and measure it that sidesteps the missing primitive entirely.

The rest of this memo is the per-knob construct-validity argument, each ending in an explicit verdict and a statement of the pain-faithful elicitor the environment actually offers.

---

## §0. The construct-validity stance I bring to the room

Three principles govern every verdict below. They are the discipline I uniquely hold.

**(i) Nociception is not pain.** The environment's nociception channels are *peripheral signals* — a contact-pain spike, a delayed injury trace. Pain is a *perceptual / inferential construct* built on top of those signals under a prior over bodily threat. A knob driven by a nociception channel tests pain *only if* the behavioural read-out depends on more than the raw signal magnitude — only if the agent's *inference about its body* moves the behaviour, not just the input itself. I apply this test to each knob.

**(ii) Aversion is not pain.** A predator is aversive. Starvation is aversive. Damage is aversive. The clinical pain literature exists precisely because pain has *specific* signatures that generic aversion does not (Eccleston & Crombez 1999, the interruption hypothesis). A knob that any aversive stimulus would drive equally well is testing aversion, not pain. The discriminating question for each knob: does the *injury/nociception* channel do something the generic *threat* channel does not?

**(iii) The modulator input `c` is interoceptive by design — this is the project's pain-relevant asset.** The four channels (`05_body_homeostasis.md` / `09_sensors_and_observation.md`, catalogued in grounding §5) are injury `[0]`, nutrition `[1]`, satiation `[2]`, and the delayed interoceptive-nociception trace `[3]`. The trace at `[3]` is the single most pain-relevant input the environment has: it is a *temporally-integrated injury percept*, an alpha-kernel convolution of past injury (peak lag at `interoceptive_kernel_tau`). It is, structurally, the closest thing the environment has to "tonic pain" — a slow bodily state that outlives the phasic insult. Every pain-faithful elicitor I propose below leans on channel `[3]` rather than on the raw injury level `[0]`, because `[3]` is where inference-over-bodily-state lives and `[0]` is closer to the raw signal.

With these in hand, the per-knob verdicts.

---

## §1. NA → effective temperature (arousal / vigilance) — VALID

**Grounding-memo engineering verdict:** SUPPORTED. Graded threat exists via the predator hunt FSM with config-tunable detection/patrol/damage; threat is observable through smell, vision, and contact pain.

**The construct I am guarding.** The Doya-NA knob, read through the pain lens, is the *arousal/vigilance* construct. The clinical pain literature ties this together tightly: nociceptive input and threat of harm drive autonomic arousal and a *narrowing, sharpening* of attention onto threat-relevant information. Eccleston & Crombez 1999 frame pain as an *interrupt* that captures attentional priority; Crombez, Van Damme & Eccleston 2005 frame hypervigilance as a *concern-dependent* attentional bias — vigilance scales with how much the body's threat-state matters right now. The arousal half (NE/locus-coeruleus) and the attentional-narrowing half are the same axis the NA-temperature knob touches: lower effective temperature = greedier, more exploitative, more locked-onto-the-threat policy.

### 1.1 The construct-validity case (why this is VALID)

The environment supplies the three things the pain-arousal construct requires, and supplies them in the *right causal order*.

1. **A genuine threat source, not a mere cost.** The predator FSM (`07_predator_ai.md:180-185`) is an *agent* that pursues — threat that *approaches*, that the agent must track and flee. This is qualitatively closer to the clinical elicitor (an impending bodily harm) than a static damage tile would be. The arousal literature is about *anticipated* harm; an approaching predator is anticipated harm.

2. **A real nociceptive consequence.** Contact with the predator produces injury (`07_predator_ai.md:275-276`), which feeds the injury channel `[0]` and, through the convolution buffer, the delayed "recent pain" trace `[3]`. The threat is not abstract — it cashes out in a bodily-state change the modulator reads.

3. **A `c`-channel that carries "recent pain" forward in time.** This is the linchpin. The pain-arousal signature the project wants is *not* "arousal spikes during the predator encounter" — every animal does that, and it is pure threat-reactivity, not pain. The *pain*-specific signature is "**arousal/vigilance stays elevated after the encounter, scaled by recent nociceptive history**". Channel `[3]` is exactly the substrate for this: it is high *because the agent was recently hurt*, and it decays slowly (alpha kernel, lag `interoceptive_kernel_tau`). So "recent pain exposure → channel `[3]` elevated → modulator lowers effective temperature → agent stays vigilant/exploitative after the threat has passed" is a *faithful* pain-arousal signature, and it is **dissociable from raw threat-reactivity** precisely because `[3]` outlasts the predator's presence.

That dissociation is what makes this VALID rather than merely "aversion-shaped". Run the discriminating test from §0(ii): does the injury/nociception channel do something the generic threat channel does not? Yes — channel `[3]` keeps vigilance up *after* the predator-vision and predator-smell channels have gone quiet. A pure threat-reactivity account predicts vigilance tracks predator presence; the pain account predicts vigilance tracks *recent nociceptive history*. The environment can express the difference.

### 1.2 The pain-faithful elicitor the env offers

> **Post-injury vigilance persistence.** Construct an episode where the agent is injured by a predator, then the predator leaves / loses interest (the FSM's `lose_interest_multiplier` already supports disengagement, grounding §3). Measure effective temperature (policy entropy / exploitation rate) in the window *after* predator disengagement, as a function of the `[3]` trace level. A pain-valid NA signature is: effective temperature stays *depressed* in proportion to channel `[3]`, even though the exteroceptive threat channels have returned to baseline. The decay of this depression should track `interoceptive_kernel_tau`, not predator-presence.

This is the cleanest pain-construct test in the whole symposium, and it falls out of the environment as-built.

### 1.3 Where it overreaches (the honest flag)

Two cautions, neither fatal.

- **Threat is emergent, not imposed (inherited from grounding §4.1).** The agent can flee to lower its own threat, so "arousal level" is a *measured* quantity confounded with the agent's own policy. For the *post-injury* signature in §1.2 this is actually fine — the injury has already happened and is logged in `[3]`; the agent cannot un-injure itself. So the pain-relevant elicitor is *less* exposed to this confound than a generic during-encounter arousal sweep would be. Prefer the post-injury window.
- **Do not claim Yerkes-Dodson inverse-U as a *pain* result.** The grounding memo frames the NA knob as an inverse-U-of-arousal sweep. The inverse-U is an *arousal-performance* law, not a pain-specific one — a startle or a loud noise would produce it too. The inverse-U is fine as a *demonstration that the temperature knob works*, but the *pain* claim must rest on §1.2's nociception-history dependence, not on the inverse-U shape. Conflating the two would be a construct overreach: it would let any arousing stimulus stand in for pain.

**Construct-validity verdict — NA: VALID.** The environment offers a faithful pain-arousal elicitor (post-injury vigilance persistence scaled by the `[3]` recent-pain trace, dissociable from threat-reactivity by outlasting predator presence). This is the project's strongest knob on construct grounds. The pain claim must rest on the nociception-history dependence, not on the generic arousal-performance inverse-U.

---

## §2. ACh → effective learning rate (expected uncertainty) — CONSTRUCT-MISMATCH

**Grounding-memo engineering verdict:** PARTIAL. Per-episode i.i.d. re-sampling of predator parameters and resource chemistry exists, but no controlled volatility schedule (no within-episode change-point, no stable-then-volatile block structure).

**The construct I am guarding.** The Doya-ACh knob is *expected uncertainty* — the learner's estimate of how noisy/volatile the world's contingencies are, which sets how fast it should update. The pain-relevant version of this is specific and clinically grounded: **uncertainty about bodily threat and bodily state**. "Is this location dangerous or safe?" "Is my body healing or deteriorating?" "Will this movement hurt?" The pain literature treats *intolerance of uncertainty* about pain as a core driver of pain-related fear and persistence (Vlaeyen & Linton 2000; Tabor & Burr 2019 on Bayesian learning of pain contingencies). The expected-uncertainty construct, in pain, is about the *predictability of the nociceptive contingency*.

### 2.1 Why this is a CONSTRUCT-MISMATCH (not merely PARTIAL)

The grounding memo's PARTIAL verdict is an *engineering* verdict — the variability ingredients exist, a controlled schedule does not. My *construct-validity* verdict is harsher, and on a different axis. Even if a senior-developer added a clean stable-then-volatile schedule, two problems would remain that make this a poor *pain* manipulation:

1. **The volatility is in the wrong place.** The re-sampling reshuffles *predator behavioural parameters* (detection, stamina, recovery) and *resource chemistry* (grounding §4.2). That is volatility in the *exteroceptive threat contingency* and in the *foraging contingency* — not in the *nociceptive contingency the agent infers about its own body*. The pain-relevant uncertainty construct is about the agent's uncertainty over **the mapping from situations to bodily harm** and over **its own injury/healing dynamics**. The environment's injury dynamics are *deterministic* given a contact (`damage [min,max]` is the only stochasticity, and that is reward/cost variance — the DA knob, §3 — not contingency volatility). The agent has nothing to be *expected-uncertain* about regarding its body's response to harm.

2. **i.i.d. resampling is not volatility, it is noise — and pain-uncertainty is about *structured* contingency change.** Per §0(i), pain is inferential. Expected-uncertainty in the pain literature is the agent's belief about whether the *rule* connecting cues to harm is stable or shifting (a place that *was* safe is now dangerous; a movement that *was* fine now hurts). i.i.d. per-episode resampling gives the agent no within-episode rule to track — the contingency is redrawn before any inference about its stability could matter. This is the same flaw the placebo/expectation literature warns about (grounding's cross-reference to "noise landscape does not reward precision", project_plan §4 cause 3): without an *expectation contrast* — a period of stable contingency against which a change registers — there is nothing for an uncertainty-estimate to do.

Run the §0(ii) discriminating test: does the injury/nociception channel do something here that the generic threat channel does not? **No.** The volatility, such as it is, lives entirely in the exteroceptive predator/resource parameters. The interoceptive channels `[0]`/`[3]` are *passive read-outs of a deterministic injury process*; they carry no contingency-uncertainty for the ACh construct to track. This is the signature of a construct-mismatch: the knob, if wired up, would be learning about predator-statistics volatility and calling it pain-uncertainty.

### 2.2 The pain-faithful elicitor the env *could* offer (what would fix it)

This is a "not as built" verdict, so I name what would make it pain-valid — handed to `senior-developer` / `experiment-designer`, not claimed as present:

> **Stochastic, change-pointed injury contingency.** Make the *nociceptive contingency itself* the thing that is stable-then-volatile: a cue (a smell, a tile colour) that predicts injury with probability `p`, where `p` is held stable for a block and then switches at a change-point. The pain-valid ACh signature is then a learning-rate increase localised to the *injury-predicting* cue after the change-point, faster than to a yoked food-predicting cue. This makes the uncertainty *about bodily harm specifically*, gives an expectation contrast, and dissociates pain-contingency-learning from foraging-contingency-learning. None of this exists today; both the probabilistic-injury-cue mechanism and the blocked change-point schedule are new env code.

Until that exists, wiring channel `[3]` into a learning-rate knob and calling the result "pain-relevant expected-uncertainty modulation" would be a construct overclaim. It would be measuring the agent's adaptation to *predator-parameter* shuffling.

**Construct-validity verdict — ACh: CONSTRUCT-MISMATCH (as built).** The environment's volatility is exteroceptive (predator/resource parameters) and i.i.d.-resampled, not a structured contingency-change in the agent's *bodily-harm* mapping. There is no pain-relevant uncertainty for the expected-uncertainty construct to track, because injury dynamics are deterministic given contact. A probabilistic, change-pointed *injury-cue* contingency would make it pain-valid; it requires new env code and does not exist today. Do not claim this knob tests *pain* uncertainty on the current testbed.

---

## §3. DA → effective TD-gain (wanting / risk) — VALID-WITH-CAVEATS

**Grounding-memo engineering verdict:** PARTIAL. Reward-cost variance is reachable via `damage [min,max]` and spatial layout; food reward is fixed at `+1`, so there is no single reward-σ knob; a risky-vs-safe two-path task needs careful config design but no new code.

**The construct I am guarding.** The Doya-DA knob is the TD-error gain — sensitivity to prediction-error magnitude, which behaviourally reads out as *risk sensitivity* and *wanting*. The grounding memo makes a sharp observation I want to build on: because food reward is fixed, the environment's *tunable* variance is on the **damage/pain** side. A risky-vs-safe task here is naturally "a path with high variance in *how much it hurts*" versus "a path with low variance in how much it hurts". The pain-science name for this is **risk of harm**, and it is the engine of the **fear-avoidance model** (Vlaeyen & Linton 2000; Vlaeyen, Crombez & Linton 2016): pain-related fear drives avoidance of activities whose *harm outcome is uncertain*, and pain catastrophising amplifies sensitivity to the *worst-case* harm outcome.

So the grounding memo's point is correct and stronger than it states: **a cost-variance task is not a degraded substitute for a reward-variance task — for a *pain* project it is the more construct-valid version.** A reward-variance gamble tests appetitive risk; a harm-variance gamble tests *fear of pain*, which is the project's actual target.

### 3.1 The caveat that determines validity — appetitive vs. aversive systems must not be conflated

Here is the construct trap, and it is the reason the verdict is VALID-*WITH-CAVEATS* and not simply VALID.

The Doya-DA mapping is, in its original form, about the *dopaminergic reward-prediction-error* system — an *appetitive* system. Pain/harm prediction-errors are handled by a *partly distinct* aversive-learning circuitry (the literature is explicit that appetitive and aversive prediction errors are not a single signed axis on one system — there are dedicated aversive-PE signals, and the dopamine response to aversive outcomes is heterogeneous). If the project wires a *harm-variance* task into a knob and calls it "the DA / TD-gain knob", it is **importing the aversive-learning construct into a slot the Doya framework defined for the appetitive system**. That is a category move, and it must be made *explicitly* or it becomes a construct error.

Two ways to handle it honestly:

- **Honest framing A (recommended):** State that in this environment the TD-gain knob is exercised on the *aversive* (harm-prediction-error) channel, that this is a *deliberate* pain-relevant repurposing of the Doya-DA function rather than a faithful instantiation of the dopaminergic-reward-PE story, and cite the fear-avoidance literature as the construct authority instead of the dopamine-RPE literature. This is defensible and is arguably the *more interesting* claim — but it must be flagged, not smuggled.
- **Honest framing B:** Keep food reward as the appetitive variance source and accept the grounding memo's note that this requires homeostatic mode (where drive-reduction reward carries continuous magnitude) rather than survival mode's `{0,+1}`. This stays faithful to the dopamine-RPE construct but is *less* pain-relevant — it tests appetitive risk, not fear of harm.

I recommend Framing A, with the flag stated in the paper, because it is the pain-relevant one and the project is a pain project.

### 3.2 The pain-faithful elicitor the env offers

> **Harm-variance two-path foraging (fear-avoidance task).** Two routes to equivalent food, matched on expected damage, differing in *damage variance*: a "safe" route with `damage: [10,10]` (deterministic small hits) and a "risky" route with `damage: [0,40]` (same mean, high variance), per grounding §4.3. A risk-*averse* agent (high effective harm-PE gain, fear-avoidance signature) prefers the safe route despite equal expected cost. The pain-valid manipulation is to make this route-preference *depend on the agent's recent-pain state*: when channel `[3]` is high (recently hurt), the agent should shift further toward the safe route — clinically, **prior pain experience amplifies pain-related fear and avoidance** (the fear-avoidance loop). That `[3]`-dependence is what makes it a *pain* result rather than a static risk-attitude result.

Run the §0(ii) discriminating test: does the nociception channel do something distinctive? **Yes, under the `[3]`-dependent version.** A static risk-attitude account predicts route-preference is constant; the fear-avoidance account predicts route-preference shifts with recent nociceptive history. The environment can express that shift, and the shift is the pain content.

### 3.3 The overreach to guard against

- **Do not call mere harm-aversion "wanting".** The Doya-DA "wanting" framing is appetitive incentive salience. Avoiding the risky-harm path is *not* "wanting" in that sense — it is fear-avoidance. Use the harm/fear vocabulary, not the appetitive-wanting vocabulary, for the harm-variance task. Mixing them would be the §0(ii) failure dressed as a feature.
- **Pre-register that the variance, not the mean, drives the effect.** Because the safe and risky routes are mean-matched, a preference shift *must* be a variance-sensitivity (TD-gain) effect and cannot be explained by expected-cost minimisation. This is the analog of the DA construct's defining test and it is what keeps the claim falsifiable.

**Construct-validity verdict — DA: VALID-WITH-CAVEATS.** A harm-variance (risk-of-pain) two-path task is *more* pain-relevant than a reward-variance task and is config-reachable. The binding caveat: this repurposes the Doya-DA *appetitive* TD-gain construct onto the *aversive* harm-prediction system, which must be flagged explicitly (Framing A) and anchored to the fear-avoidance literature, not the dopamine-RPE literature. The pain content lives in making route-preference *depend on the `[3]` recent-pain trace*; without that dependence it is a static risk-attitude result, not a pain result.

---

## §4. 5-HT → effective discount (horizon / myopia) — RED VERDICT SOFTENABLE

**Grounding-memo engineering verdict:** NOT SUPPORTED. Reward is paid the same step the action earns it; the two "delay"-named mechanisms (`res_reg_delay` respawn timer, the interoceptive nociception kernel) are respawn-timing and sensor-delay, neither a reward delay. A reward-delay queue would need new env code.

**This is the knob where I most directly disagree with the framing of the engineering verdict — not its facts, but its scope.**

### 4.1 The grounding memo asked the wrong question for *pain*

The grounding memo (correctly, on its own terms) asked: *can the environment deliver reward at a tunable delay?* That is the canonical *intertemporal-choice* operationalisation of the discount parameter — sooner-smaller vs. later-larger reward. The answer is genuinely NO without new code.

But the **clinical pain finding is not about reward delay.** The robust, replicated finding is: **chronic pain shifts delay-discounting toward myopia** — pain patients discount the future more steeply, choose smaller-sooner over larger-later, show shortened effective planning horizons, and this myopia tracks pain chronicity and intensity. The construct is **"pain makes you myopic"** — an *interoceptive-state → effective-horizon* causal link. It is *measured* with intertemporal-choice tasks in humans, but the *construct* is that a bodily-state variable (ongoing pain) compresses the horizon over which the agent optimises. That is precisely the kind of interoceptive→horizon coupling this project's whole architecture is built to express: `c` drives a knob, the knob here being effective discount.

So the right question for *this symposium* is not "can the env delay reward?" but: **"can the env elicit a recent-pain-driven shift in the agent's effective planning horizon, and can we measure that shift?"** And the answer to *that* is materially better than red.

### 4.2 Why no reward-delay primitive is needed

The intertemporal-choice task is one *instrument* for reading out the discount parameter. It is not the *only* one, and for an RL agent with a value critic it is not even the most direct one. The effective horizon is a property of the *value function itself*: it is governed by $\gamma$, and the critic's value estimate $V(s)$ already encodes how far into the future the agent is integrating expected return. We can read the effective horizon **directly off the critic**, without ever constructing a sooner-vs-later reward choice. Concretely:

- The **effective horizon** under discount $\gamma$ is $H_{\text{eff}} = 1/(1-\gamma)$. A myopia shift is a drop in $H_{\text{eff}}$, i.e. an increase in effective discounting.
- For a FiLM/hypernet-conditioned critic where the discount is *effective* (induced by conditioning on `c`, not literally set), $H_{\text{eff}}$ is recoverable behaviourally and representationally: the **planning-distance over which the critic's value is sensitive to future events**. Two read-outs, neither needing reward delay:
  - **Representational:** perturb a future hazard/food at horizon $k$ steps ahead and measure $|\Delta V|$ as a function of $k$. The slope of $\log|\Delta V|$ against $k$ *is* the effective discount rate (this is exactly the slope the grounding memo named for the reward-delay probe — but it is computed by varying *where the future event sits*, not by *delaying the reward*). The hazard can be a predator placement; the food can be placed at distance $k$. No reward-delay buffer is touched.
  - **Behavioural:** the agent's willingness to traverse a longer safe path for a larger delayed nutritional gain vs. grab a nearer smaller food — this *is* a sooner-smaller-vs-later-larger choice, but the "later" is implemented as *spatial distance* (more steps to reach), which the environment already supports natively (food placement + path length). Distance-to-reward is a temporal-delay proxy that needs *zero* new code: a food $k$ tiles away is a reward $\ge k$ steps in the future.

The second read-out is important: **the environment already has a delay primitive — it is called distance.** In a stepwise gridworld, spatial separation *is* temporal delay. A food 8 tiles away is reward at least 8 steps later. The grounding memo's "NOT SUPPORTED" is correct *only* for a reward-queue-that-defers-an-already-earned-reward; it is *not* correct for "can the agent face a sooner-near vs. later-far reward trade-off", which is the actual intertemporal-choice structure and which distance implements for free.

### 4.3 The pain-faithful elicitor the env offers

> **Recent-pain-driven myopia, read off the critic.** Inject recent pain (predator encounter → injury → channel `[3]` elevated), then in the post-injury window measure the effective horizon by *either* the representational slope ($\log|\Delta V|$ vs. event-distance $k$) *or* the behavioural near-small-vs-far-large food choice (delay implemented as spatial distance). The pain-valid 5-HT signature is: when channel `[3]` is high, $H_{\text{eff}}$ *contracts* — the agent's value sensitivity to far-future events drops, and it prefers near-small food over far-large food, relative to its uninjured baseline. This is the direct analog of the clinical "pain → steeper delay-discounting" finding, with `[3]` (tonic recent-pain) as the driving interoceptive state.

This passes every test in §0. It is inferential (the *trace* `[3]`, not raw injury, drives it — §0(i)). It is pain-specific, not generic-aversion: the discriminating §0(ii) test is whether *predator presence without injury* (threat alone) compresses the horizon as much as *recent injury* (channel `[3]`) does — the pain account predicts the horizon contracts with the *nociceptive trace*, not with mere threat exposure. And it uses the project's actual interoceptive→knob bridge as designed.

### 4.4 What is still genuinely missing (the honest residue)

I am softening the verdict, not erasing it.

- **The representational read-out ($\log|\Delta V|$ vs. distance) needs an analysis harness, not env code** — a value-probe that perturbs events at varying distance and records critic sensitivity. Handed to `experiment-analyzer` / `experiment-designer`.
- **The behavioural near-small-vs-far-large task needs config design** — a layout with a near small food and a far large food, mean-controlled so the choice reflects horizon not just value. Config work, no new env code.
- **The literal reward-delay queue remains absent and remains the *only* way to do classic temporal-discounting with reward held constant and time varied.** If a reviewer demands the textbook intertemporal-choice instrument specifically, the grounding memo's red stands for *that instrument*. My point is that the textbook instrument is not the only construct-valid read-out, and for a *pain* project the recent-pain-driven-myopia framing is *more* faithful to the clinical literature than a reward-delay gamble would be — because the clinical construct is "pain compresses the horizon", and that is an interoceptive-state effect, exactly what distance-as-delay + `[3]`-conditioning expresses.

**Construct-validity verdict — 5-HT: RED SOFTENABLE → VALID-WITH-CAVEATS on pain-construct grounds.** The grounding memo's NOT-SUPPORTED is correct for a literal reward-delay buffer but mis-scoped for the *pain* construct. The clinical finding is "recent/chronic pain → myopia" (an interoceptive-state → effective-horizon link), not "pain delays reward". The environment offers a pain-faithful elicitor: recent-pain (`[3]`)-driven contraction of the critic's effective horizon, with delay implemented as *spatial distance* (which the gridworld supports natively) and horizon read directly off the critic. This needs an analysis harness and config design, not a reward-delay primitive. The textbook reward-delay instrument remains absent; the *construct* is testable without it.

---

## §5. Cross-knob summary table

| Knob | Engineering verdict (grounding memo) | Pain construct-validity verdict (this memo) | Pain-faithful elicitor the env offers | Driving `c` channel |
|---|---|---|---|---|
| **NA → temperature** (arousal/vigilance) | SUPPORTED | **VALID** | Post-injury vigilance *persistence*, scaled by recent-pain trace, dissociable from threat-reactivity by outlasting predator presence | `[3]` recent-pain trace |
| **ACh → learning rate** (expected uncertainty) | PARTIAL | **CONSTRUCT-MISMATCH (as built)** | None as built; would need a probabilistic, change-pointed *injury-cue* contingency (new env code) | `[3]` — but only if injury contingency is made stochastic |
| **DA → TD-gain** (wanting/risk) | PARTIAL | **VALID-WITH-CAVEATS** | Harm-variance (risk-of-pain) two-path fear-avoidance task; route-preference shift *with* recent-pain state | `[3]` modulating risk attitude; harm-PE (aversive system, flagged) |
| **5-HT → discount** (horizon/myopia) | NOT SUPPORTED | **RED SOFTENABLE → VALID-WITH-CAVEATS** | Recent-pain-driven myopia: `[3]`-conditioned contraction of critic effective horizon; delay = spatial distance | `[3]` recent-pain trace |

The pattern is worth naming: **every pain-faithful elicitor I found leans on channel `[3]`, the delayed interoceptive-nociception trace.** That channel is the project's pain-relevant crown jewel. It is the one input that carries *bodily-state-over-time* rather than *instantaneous signal*, and it is therefore the only channel that lets a knob's behaviour depend on *inference about the body* (§0(i)) rather than on raw input magnitude. If the project does one thing to strengthen its pain-construct validity across all four knobs, it is to ensure `[3]` is the modulator input and to design probes that dissociate `[3]`-driven behaviour from raw-threat-driven behaviour.

---

## §6. Construct-validity guardian's bottom line

What I am willing to let the project claim, and what I am not:

**I will sign off on:**
- "Recent pain elevates arousal/vigilance" (NA) — the strongest, cleanest pain claim, fully supported, *provided* the claim rests on nociception-history persistence (§1.2) and not on the generic arousal inverse-U.
- "Risk of pain drives fear-avoidance route choice, amplified by recent pain" (DA) — *provided* it is flagged as a deliberate repurposing of the appetitive TD-gain construct onto the aversive harm-prediction system, anchored to fear-avoidance not dopamine-RPE.
- "Recent pain compresses the agent's effective planning horizon" (5-HT) — *provided* it is read off the critic / distance-as-delay, framed as the clinical pain-myopia construct, and *not* claimed as the textbook reward-delay intertemporal-choice instrument (which the env lacks).

**I will not sign off on:**
- "The environment tests pain-relevant expected uncertainty" (ACh) — as built it tests adaptation to *predator-parameter* shuffling; the injury process is deterministic, so there is no bodily-harm contingency-uncertainty for the construct to track. This needs a probabilistic injury-cue with a change-point before it is a pain claim.
- Any claim that survives the §0(ii) test failing — i.e., any knob whose behaviour the *generic threat channel* would drive as well as the *nociception channel* does. Aversion is not pain. The recurring discriminator across all four knobs is whether channel `[3]` (recent pain) does something predator-presence alone does not.

The bridge — "interoceptive state drives neuromodulation" — is construct-valid in this environment **for three of the four knobs**, and the one that fails (ACh) fails for a *fixable* reason (deterministic injury dynamics), not a fundamental one. The project's pain-relevance does not rest on dressing aversion in clinical language; it rests on the delayed interoceptive-nociception trace `[3]` being a genuine bodily-state-over-time signal that can drive behaviour the raw threat signal cannot. That is a real, defensible foundation — *if* the probes are designed to demonstrate the dissociation rather than assume it.

---

## §7. Hand-offs

- **`experiment-designer`:** the three pain-faithful tasks — post-injury vigilance-persistence window (NA, §1.2), harm-variance two-path fear-avoidance task with `[3]`-dependent route preference (DA, §3.2), near-small-vs-far-large food choice with distance-as-delay (5-HT, §4.3). All three are config-reachable per the grounding memo. Each must include the §0(ii) dissociation control (threat-without-injury arm).
- **`experiment-analyzer`:** the critic effective-horizon probe ($\log|\Delta V|$ vs. event-distance $k$, conditioned on channel `[3]`) for the 5-HT myopia read-out (§4.2). Analysis harness, not env code.
- **`senior-developer`:** the *only* genuine env-code gaps the pain lens flags are (a) the probabilistic, change-pointed injury-cue contingency that would make the ACh knob pain-valid (§2.2), and (b) — optional, lower priority — a literal reward-delay queue if a reviewer insists on the textbook intertemporal-choice instrument (§4.4). Neither is required for three of the four knobs.
- **`professor-neuromodulation` (cross-review):** the NA post-injury-persistence signature (§1.2) maps at the substrate level onto tonic locus-coeruleus tone driven by descending nociceptive input; the DA harm-variance caveat (§3.1) maps onto the appetitive-vs-aversive prediction-error circuitry split — please confirm the substrate-level story for both.
- **`professor-bayesian-brain` (cross-review):** the 5-HT myopia-via-critic-horizon read-out (§4.2) overlaps your effective-horizon / planning-as-inference machinery; the `[3]`-conditioned $H_{\text{eff}}$ contraction is where our domains meet.

---

## Cross-references

- Shared grounding: [`docs/project/symposium/20260618_env_suitability_neuromodulation/00_env_capability_grounding.md`](00_env_capability_grounding.md) — §3 capability table, §4 per-knob detail, §5 interoceptive-`c` inventory.
- Construct-validity foundation: [`docs/project/concepts/pain_vs_nociception_construct.md`](../../concepts/pain_vs_nociception_construct.md).
- Prior pain-modeling symposium position (the four-dissociation gate, behavioural-effect re-statement): [`docs/project/symposium/20260516_impact_vs_reasonable/professor_pain_modeling_contribution.md`](../20260516_impact_vs_reasonable/professor_pain_modeling_contribution.md).

**Canonical pain literature anchors:**
- Eccleston & Crombez 1999 — interruption hypothesis (pain as attentional interrupt; arousal/vigilance; pain ≠ generic aversion).
- Crombez, Van Damme & Eccleston 2005 — hypervigilance as concern-dependent attentional bias.
- Vlaeyen & Linton 2000; Vlaeyen, Crombez & Linton 2016 — fear-avoidance model (the DA risk-of-harm / route-choice construct authority).
- Tabor & Burr 2019 — Bayesian learning of pain contingencies (the ACh expected-uncertainty-of-bodily-harm construct).
- Büchel et al. 2014; Wiech 2016 — predictive-coding-of-pain (inference-over-bodily-state; §0(i) nociception ≠ pain).
- Apkarian, Baliki & Geha 2009 — chronic pain as adaptive-controller failure (the persistence framing behind §1.2 and §4).
- Pain → delay-discounting / myopia literature (clinical finding behind §4): chronic-pain steepened temporal discounting and shortened effective horizon. *Not yet in `docs/project/references/` — flag for the user to add a delay-discounting-in-chronic-pain source (e.g. work tying pain intensity/chronicity to intertemporal-choice myopia) so the 5-HT softening has a citable anchor.*

---

*Contribution by `professor-pain-modeling`, 2026-06-18. Construct-validity guardian for the symposium `20260618_env_suitability_neuromodulation`. Verdicts: NA VALID, ACh CONSTRUCT-MISMATCH (as built, fixable), DA VALID-WITH-CAVEATS, 5-HT RED-SOFTENABLE-to-VALID-WITH-CAVEATS on pain-construct grounds.*
