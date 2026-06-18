---
title: "Postdoc synthesis — can the pain grid world test the four Doya-2002 neuromodulator knobs?"
session: 20260618_env_suitability_neuromodulation
synthesiser: research-postdoc
date: 2026-06-18
inputs:
  - docs/project/symposium/20260618_env_suitability_neuromodulation/00_env_capability_grounding.md
  - docs/project/symposium/20260618_env_suitability_neuromodulation/professor_neuromodulation_contribution.md
  - docs/project/symposium/20260618_env_suitability_neuromodulation/professor_rl_contribution.md
  - docs/project/symposium/20260618_env_suitability_neuromodulation/professor_pain_modeling_contribution.md
  - docs/project/symposium/20260618_env_suitability_neuromodulation/professor_bayesian_brain_contribution.md
verdict_axis: "SUITABLE-AS-IS | SUITABLE-WITH-CONFIG | SUITABLE-WITH-ANALYSIS-ONLY | NEEDS-ENV-WORK"
downstream: "named hand-offs only (experiment-designer / senior-developer / pi) — nothing executed here; the PI focus-vs-explore call is the natural next gate before GPU-weeks"
---

# Postdoc synthesis — can the pain grid world test the four Doya-2002 neuromodulator knobs?

## §1 — Plain-English entry point

This symposium asked one question: **can our existing pain grid world environment actually test the four classic "a brain chemical sets a learning knob" ideas from Doya (2002)?** The four ideas are noradrenaline (NA, the alarm/arousal chemical) setting *how exploratory* the agent is; acetylcholine (ACh, the "the world changed, pay attention" chemical) setting *how fast it relearns*; dopamine (DA, the reward chemical) setting *how strongly risky-vs-safe payoffs drive learning*; and serotonin (5-HT, the patience chemical) setting *how far into the future the agent plans*. We do not literally set these knobs — a small "modulator" network reads the agent's bodily state (how full it is, how recently it was hurt) and conditions the policy so that we can *measure* behaviour as if it had those knobs.

A shared grounding memo gave a first engineering verdict: NA testable now, ACh and DA reachable with config work, and 5-HT not testable (the environment pays reward the instant food is eaten, with no way to delay it). The four professors then largely converged but **revised two of those four verdicts in important ways.** The headline a reader should leave with: **NA is testable now** (with the right reading of "arousal"); **5-HT is testable now too, by reading the critic's planning horizon** — the original "not supported" was mis-scoped; **DA is testable with a config-built risky-vs-safe task**, honestly relabelled because the only variance our world offers is on the *harm* side; and **ACh is the one knob that genuinely needs new environment code** — and that same code also unlocks the deepest version of NA. So: not "one green, two amber, one red", but **one green, two now-reachable, and one firm build-this-first.**

---

## §2 — The load-bearing convergence (four lenses, and what each revised)

Reading the four contributions side by side, the professors agree on the substantive picture and each revised the grounding memo on a specific knob. When four independently-reasoned lenses converge on the same per-knob direction, that direction is load-bearing.

| Lens | Core question it answered | What it revised vs. the grounding memo |
|---|---|---|
| **Neuromodulation** (biological-plausibility guardian) | Does the env *wake up* the brain system the knob stands for, or just look like it on paper? | **Downgraded ACh** to *construct-not-engaged* (i.i.d. resampling is the one regime where neither ACh nor NA should respond). **Softened 5-HT** from red to faithful-with-caveats (internal-state-driven horizon is the *primary* 5-HT construct, not literal delay). Flagged **DA** as partly an aversive-system task, not pure dopamine. |
| **RL** (measurability + cleanest probe) | Even where the env can express it, can we *measure* the fingerprint with a clean RL probe? | **Split ACh** — agent-internal gradient probe works today but only proves context-conditioned steering, *not* volatility-tracking; behavioural signature needs a block scheduler. **Reclaimed 5-HT** — effective horizon reads off the critic with *no env work*. Pinned **DA** to homeostatic mode + IQN; cost-side variance → downside/left-tail reweighting. |
| **Pain-modeling** (construct-validity guardian) | When we wire a body channel into a knob and call it "pain-like", is that honest? | **Downgraded ACh** to *construct-mismatch* (injury dynamics are deterministic → no bodily-harm uncertainty to track). **Softened 5-HT** (clinical finding is "pain makes you myopic" — an interoceptive→horizon link, no reward-delay needed). Gave **NA** its load-bearing pain signature: post-injury vigilance *persistence*. Argued the **DA** harm-variance version is *more* pain-relevant. |
| **Bayesian-brain** (precision / uncertainty lens) | Does the env engage the *uncertainty-theoretic* content (Yu-Dayan expected vs. unexpected uncertainty)? | **Downgraded ACh** to "PARTIAL ingredients, NOT-SUPPORTED construct" (no inferable volatility). **Split NA** into *arousal* (supported) vs. the true *unexpected-uncertainty* construct (needs regime-change events). Offered a **5-HT** prior-precision route needing no env code. Elevated the **ACh⊥NA dissociation** as the single highest-value build. |

**The four-way agreement worth naming:** all four professors independently downgrade ACh (each in their own vocabulary — no learnable volatility / no volatility-tracking / deterministic injury / no inferable drift-rate), and all four push back on the 5-HT red. These two revisions are the symposium's most load-bearing outputs and they drive the verdict table below.

---

## §3 — The per-knob verdict table (the deliverable)

Verdict scale: **SUITABLE-AS-IS** (run today, no code, no new config) · **SUITABLE-WITH-CONFIG** (YAML/task design only, no new env code) · **SUITABLE-WITH-ANALYSIS-ONLY** (no env or config change; needs a measurement/analysis harness) · **NEEDS-ENV-WORK** (requires new code in `src/environment/`).

| Knob | Verdict | Cleanest probe | Reward mode | What's missing |
|---|---|---|---|---|
| **NA → effective temperature** (exploration / arousal) | **SUITABLE-AS-IS** for the arousal reading; **NEEDS-ENV-WORK** for the deeper unexpected-uncertainty reading | Across a threat-config grid, regress policy entropy on the modulator's effective-temperature output (primary); survival-steps inverse-U (corroboration); **post-injury vigilance-persistence window** scaled by the recent-pain trace (the pain-faithful signature) | Survival | Nothing for the arousal/inverse-U sweep. The *unexpected-uncertainty* construct needs the same change-point machinery ACh needs. |
| **ACh → effective learning rate** (volatility) | **NEEDS-ENV-WORK** — the firmest "not testable as-is" verdict in the symposium | Behavioural learning-rate jump after a scheduled regime change, faster in volatile blocks than stable blocks; agent-internal gradient probe runs today but only proves context-conditioned steering | Survival | A controllable volatility schedule: a persistent drifting contingency on `EnvState` with a *schedulable drift-rate* and labelled change-points (the §5 ACh⊥NA build). i.i.d. per-episode resampling is not volatility. |
| **DA → effective TD-error gain** (risk sensitivity) | **SUITABLE-WITH-CONFIG**, honestly relabelled | Two-zone, mean-matched-cost foraging (safe `damage:[c,c]` vs. risky `damage:[c−δ,c+δ]`); IQN distributional critic reads the per-context return-distribution shape; behavioural read-out = zone-occupancy + survival steps, shifting with the recent-pain channel | **Homeostatic** (survival mode's `{0,+1}` reward carries no variance) | A clean *appetitive* DA story needs stochastic food gain (new config/code). As-built the variance lives in *harm* → relabel as aversive/risk prediction-error gain. |
| **5-HT → effective discount** (planning horizon) | **SUITABLE-WITH-ANALYSIS-ONLY** — the red verdict was mis-scoped; literal-delay behavioural version is optional **NEEDS-ENV-WORK** | Regress `log\|V\|` on graph-distance-to-food across states, stratified by interoceptive context; γ_eff = exp(∂log\|V\|/∂k); test whether γ_eff contracts when the recent-pain channel is high ("pain makes you myopic") | Either (the critic probe reads the value function, not behaviour) | Nothing for the critic-horizon probe. The *behavioural* sooner-small-vs-later-large delay-discounting task needs a reward-delay ring buffer on `EnvState`. |

**The one cross-knob constraint the table cannot show in a single row:** the DA probe *requires* homeostatic reward mode and the optional 5-HT behavioural-delay probe *requires* survival mode. They cannot be co-tested. Plan two reward-mode tracks (§5).

---

## §4 — Per-knob narrative (resolving the tensions)

### 4.1 NA → temperature — the clearly-suitable knob, but state which reading

The grounding memo said SUPPORTED and all four professors agree NA is the strongest knob. The synthesis must be precise about *which* NA, because the professors split it cleanly into two readings.

**Reading 1 — arousal as effective temperature (SUITABLE-AS-IS).** Graded predator threat is the right elicitor; the RL voice gives the clean probe (entropy regressed on the modulator's temperature output across a threat-config grid, corroborated by the survival-steps inverse-U). Two honest caveats both professors raise: arousal is **measured/emergent, not experimenter-imposed** — the agent can flee to lower its own threat — so the relationship is *correlational*, and the curve must be **tiled across a threat-config sweep** rather than set within one episode. This is adequate and runnable today.

**The pain-faithful signature that makes it more than generic arousal (load-bearing).** The pain lens contributes the signature that dissociates pain from mere threat-reactivity: **post-injury vigilance persistence.** After a predator injures the agent and then disengages, effective temperature should stay *depressed in proportion to the recent-pain trace* (the delayed interoceptive-nociception channel, `c[3]`), even though the predator's smell and vision channels have returned to baseline, decaying on the kernel's timescale rather than predator-presence. This is the project's single cleanest pain-construct test and it falls out of the env as-built — and it is *less* exposed to the emergent-arousal confound, because the injury has already happened and is logged in `c[3]`; the agent cannot un-injure itself. The NA claim should rest on this nociception-history dependence, **not** on the generic arousal inverse-U (which a loud noise would also produce).

**Reading 2 — NA's true Yu-Dayan construct (NEEDS-ENV-WORK).** The Bayesian-brain lens sharpens the verdict: NA in Yu & Dayan is specifically *unexpected uncertainty* — model-violating surprise. Proximity threat from a predator the agent has already learned is *expected* (the model is being confirmed, not violated), so it drives the arousal axis but not the unexpected-uncertainty construct. The deeper NA reading needs scheduled, labelled **regime-change events** (a neutral animal suddenly turning predatory; a smell→danger contingency flip) — which is the *same* machinery ACh needs (§5). **Resolution: NA is the clearly-suitable knob via the arousal + post-injury-persistence reading (run now); the unexpected-uncertainty reading is deferred onto the shared change-point build.**

### 4.2 ACh → learning rate — the firm "needs env work"

This is the symposium's firmest verdict, and it is firm precisely because **three professors independently downgraded the grounding memo's PARTIAL to construct-not-engaged**, each from a different direction:

- **Bayesian-brain:** i.i.d. per-episode resampling has no inferable volatility — there is no step-to-step drift whose *rate* the agent can estimate. Once the agent learns the fixed sampling range, the Bayes-optimal learning rate is **constant**. There is no `v̂_t` trajectory for ACh to track.
- **Neuromodulation:** i.i.d. resampling sits in the one corner of the uncertainty plane where *neither* ACh nor its NA opponent should modulate anything — observing no learning-rate signature would be uninformative, and observing one would be hard to attribute to the construct.
- **Pain-modeling:** the env's injury dynamics are *deterministic given a contact*, so there is no bodily-harm contingency uncertainty for the pain-relevant version of the construct to track — the volatility, such as it is, lives entirely in exteroceptive predator parameters.

The RL voice adds the measurability nuance that keeps this honest: the **agent-internal gradient probe** (does the network rotate its update direction across post-hoc strata of sampled predator parameters?) *can* run today — but it proves only **context-conditioned gradient steering**, not **volatility-driven learning-rate adaptation**, because there is no volatility to track. Calling the i.i.d. gradient result a "volatility" result would be the underspecified-claim trap.

**Resolution: ACh genuinely NEEDS-ENV-WORK.** The construct does not merely lack a clean probe — without structured volatility, *the construct does not exist in the env to be probed at all.* The required build is a structured / change-point volatility mechanism, specified in §5.1.

### 4.3 DA → TD-gain — resolve the appetitive-vs-aversive labelling honestly

All four professors converge on one fact: **food reward is fixed at +1, so the only tunable variance is on the damage/cost side.** This forces an honest labelling decision.

- Survival mode's `{0,+1}` reward carries no magnitude variance — the DA knob is **not testable in survival mode** (RL voice, firmly).
- In **homeostatic mode**, the per-step drive-reduction reward carries continuous magnitude, and per-contact `damage:[min,max]` injects genuine variance into the *injury component* of that reward. A two-zone mean-matched-cost task (safe `[c,c]` vs. risky `[c−δ,c+δ]`) is then config-constructible. An IQN distributional critic reads the per-context return-distribution shape correctly **even though the variance lives in the cost** — the distributional Bellman operator is agnostic to the sign of the reward. The phenotype is **downside / left-tail reweighting** (avoidance of the high-variance harm zone), which is the *more pain-relevant* direction.

**The labelling honesty (the tension to resolve):** the Doya-DA construct is *appetitive* — Dabney's optimistic/pessimistic asymmetry is about *reward*. A task whose variance is entirely on *harm* engages the **aversive opponent system** (the neuromodulation lens names the lateral-habenula / 5-HT / D2-NoGo circuitry; the pain lens names the fear-avoidance model). Two honest fixes, and the professors split on which is better:

- **Fix (a) — add appetitive variance.** Make food gain stochastic (`food_gain:[min,max]`) for a clean Dabney/distributional-DA story. This is the faithful-to-Doya route but needs a config-schema addition (likely new code) and is *less* pain-relevant.
- **Fix (b) — relabel honestly.** Keep cost-variance and rename the knob "aversive / risk prediction-error gain (DA–5-HT opponent)", anchored to the fear-avoidance literature. The pain lens argues this is the *more* construct-valid version for a pain project, and the pain content lives in making zone-preference *depend on the recent-pain channel* `c[3]` (recent pain → more harm-averse, the fear-avoidance loop).

**Resolution: SUITABLE-WITH-CONFIG in homeostatic mode via cost-side variance, relabelled as aversive/risk prediction-error gain (Fix b).** This is honest, config-only, and the more pain-relevant story. The appetitive `food_gain` variance (Fix a) is the cleaner *dopamine* story and is the recommended add *if* the project wants a faithful Doya-DA result rather than a fear-avoidance one — a small env-code item, not on the first-pass critical path.

### 4.4 5-HT → discount — downgrade red to amber via critic-horizon analysis

The grounding memo's NOT-SUPPORTED rested on one premise: that **literal reward delay is the unique operationalisation of the discount construct.** All four professors reject that premise, and from three independent directions they converge on the same probe.

- **Neuromodulation:** the biologically primary 5-HT construct is *internal-state-driven timescale gating* (Tanaka 2007; dorsal-raphe patience) — a starving or threatened animal behaviourally discounts the future more steeply. This is driven by internal state, **not** by an experimenter inserting latency. A reward-delay buffer would actually *bypass* the interoceptive-modulation story, which is the project's whole thesis.
- **Pain-modeling:** the clinical finding is not "pain delays reward" but "**pain makes you myopic**" — chronic pain robustly collapses the planning horizon. That is an interoceptive-state → effective-horizon link, exactly what `c[3]` → discount-knob expresses.
- **RL:** the effective horizon is a property of the *critic*, readable directly. **Probe A:** regress `log|V|` on graph-distance-to-food across states, stratified by interoceptive context; γ_eff = exp(∂log|V|/∂k); the modulator-attributable quantity is the *differential* ∂γ_eff/∂c with `γ_train` and food layout held fixed. No env code.
- **Bayesian-brain:** an active-inference route gives the same conclusion — effective horizon ∝ precision of the prior over future body states, modulable by the existing interoceptive channels with no reward-delay queue.

Two further points make the route native rather than a workaround: in a stepwise gridworld **distance-to-food *is* delay-to-reward** (a food k tiles away is reward ≥ k steps in the future), and the slope the grounding memo named for the reward-delay probe is computed by varying *where the future event sits*, not by *delaying the reward*.

**Resolution: SUITABLE-WITH-ANALYSIS-ONLY.** The critic-horizon probe (Probe A) is on the critical path, needs no env or config change, and aligns *better* with the interoceptive-modulation thesis than a reward-delay buffer would. The honest residue: inferring γ_eff from spatial foraging is **confounded** with metabolic cost, threat exposure, and exploration (the neuromodulation and RL lenses both flag this) — control it with matched-distance designs or model-based discount/risk separation. The **literal-delay behavioural version (Probe B)** — a sooner-small-vs-later-large *choice* with the indifference point as read-out — remains optional **NEEDS-ENV-WORK** (a reward-delay ring buffer, survival mode), worth building only if a cognitive-behavioural venue demands the textbook intertemporal-choice instrument.

---

## §5 — Two cross-cutting findings to elevate

These two findings span more than one knob and are the synthesis's most valuable outputs. They should drive the next decision more than any single per-knob verdict.

### 5.1 The ACh⊥NA dissociation build — the single highest-value env extension

**The problem (Bayesian-brain, §C).** ACh (expected uncertainty — reliable noise *inside* a stable regime) and NA (unexpected uncertainty — the regime *broke*) are **confounded in the env today**: the only source of changing statistics is the per-episode i.i.d. resampling, and that one event loads on *both* constructs at once (it is both "predator params vary within a known range" and "this episode differs from last, model violated at reset"). No single parameter moves one without the other, so any modulator response is unidentifiable.

**The fix is one shared build.** A persistent, slowly-drifting contingency on `EnvState` (a random walk on the smell→danger/food chemical mapping) with a **schedulable drift-rate** (the volatility knob) keyed on an episode/step counter — stable blocks vs. volatile blocks, plus optional discrete labelled change-points and logged ground-truth. This is one coherent piece of new env code, and it unlocks **three things at once**:

1. **The proper ACh construct** — effective learning rate higher in volatile blocks than stable blocks (§4.2).
2. **The deeper NA construct** — isolated surprising change-points after a stable model has formed, giving NA's true unexpected-uncertainty reading (§4.1 Reading 2).
3. **Their dissociation** — a 2×2 {stable, volatile} × {low-noise, high-noise} design with a **killer opposite-sign prediction**: expected uncertainty should *lower* the effective learning rate (average out reliable noise under an intact model) while unexpected uncertainty should *raise* it (the model needs replacing). A sign reversal across the two dials *is* the dissociation; identical responses mean the modulator is a single arousal/surprise scalar with two labels.

The pain lens adds the construct-valid refinement: make the drifting contingency an **injury-predicting cue** (a smell/tile that predicts harm with probability p, p stable-then-volatile), so the uncertainty is about *bodily harm specifically* — this is what converts ACh from construct-mismatch to pain-valid. The RL lens adds the binding constraint: **block length must exceed the credit-assignment horizon** ~1/(1−λγ), or the agent cannot track the regime for reasons unrelated to the modulator (the project's H5 modulator-timescale interaction).

**This is the highest-return investment the symposium can recommend:** one build converts *two* knobs from not-testable to construct-testable and delivers their mutual dissociation — the part of the Yu-Dayan framework with the genuine computational-psychiatry bite.

### 5.2 The reward-mode two-track conflict

The RL lens surfaces an unavoidable structural fact: the **DA knob requires homeostatic reward mode** (only the continuous drive-reduction reward can carry variance) while the **5-HT literal-delay behavioural probe requires survival mode** (the discrete `+1` food bonus is the clean quantity to delay; delaying a continuous drive-delta is semantically muddy). **They cannot be co-tested in one experiment.** Any experiment plan must split into two reward-mode tracks — a homeostatic track (DA risk task; NA and the critic-horizon 5-HT Probe A also run here since the critic probe is mode-agnostic) and a survival track (NA inverse-U; ACh behavioural volatility; optional 5-HT Probe B). This is a planning constraint to hand to `experiment-designer` up front, not a defect.

---

## §6 — What's next (named hand-offs — nothing executed here)

The synthesis routes, it does not run. Below are the named hand-offs implied by the verdicts; **the natural next gate is a PI focus-vs-explore call before any of them commits GPU-weeks**, because the two cross-cutting findings (§5) imply a real focus-vs-explore choice (run the three now-reachable knobs cheaply, or invest in the ACh⊥NA build first).

**Imply an `experiment-designer` config (config-only / analysis-only, no env code):**
- NA inverse-U + post-injury vigilance-persistence window (survival track; log `dist_to_pred`, policy entropy, survival steps, `c[3]` trace; include a threat-without-injury dissociation arm).
- DA two-zone mean-matched-cost risk task (homeostatic track; tuned so injury-variance is a non-trivial share of return variance; zone-preference conditioned on `c[3]`).
- 5-HT critic-horizon Probe A (γ_eff from `log|V|` vs. graph-distance, stratified by interoceptive context; fixed `γ_train` + food layout, vary only `c`).
- **Binding constraint to carry on every design:** any ACh volatility schedule must use block length > credit-assignment horizon 1/(1−λγ) (H5 interaction); and the DA/5-HT-behavioural reward-mode split (§5.2) means two tracks, not one.

**Imply `senior-developer` env code (genuine new code in `src/environment/`):**
- **The ACh⊥NA change-point / drift build (§5.1) — the priority item.** A persistent drifting contingency on `EnvState` with a schedulable drift-rate, blocked stable/volatile schedule, optional labelled change-points, logged ground truth; the injury ring buffer is the structural template. This is the one build that unlocks ACh + the deeper NA reading + their dissociation.
- **Optional, queued:** the 5-HT literal reward-delay ring buffer (survival mode) — build only if a cognitive-behavioural venue demands the behavioural intertemporal-choice instrument. **Optional:** stochastic `food_gain:[min,max]` if the project wants the clean *appetitive* DA story rather than the relabelled aversive one.

**Imply a `pi` call (the recommended next step):**
- A focus-vs-explore call before GPU-weeks. The cheap path: run the three now-reachable knobs (NA, DA-with-config, 5-HT-critic-probe) on existing env code and let the results inform whether the ACh⊥NA build is worth it. The invest-first path: build the §5.1 change-point machinery now because it is the highest-value extension and unlocks the dissociation that carries the computational-psychiatry impact. This is a genuine portfolio choice the user should own.

**Imply professor cross-review (not new contributions):**
- `professor-bayesian-nn` — aleatoric/epistemic decomposition on the DA/IQN value head (separating cost-variance from model uncertainty).
- `professor-dl-theory` — how the IQN distributional head attaches to the FiLM/hypernet modulator architecturally.

---

## §7 — Links

- Shared grounding (the empirical ground truth): [[00_env_capability_grounding]] — §3 capability table, §4 per-knob detail, §5 interoceptive-`c` inventory.
- [[professor_neuromodulation_contribution]] — biological-plausibility verdicts; the 5-HT softening and the DA aversive-relabelling.
- [[professor_rl_contribution]] — measurability + cleanest probes; the critic-horizon Probe A; the reward-mode two-track conflict.
- [[professor_pain_modeling_contribution]] — construct-validity verdicts; the post-injury vigilance-persistence signature; "pain makes you myopic".
- [[professor_bayesian_brain_contribution]] — precision/uncertainty lens; the ACh⊥NA dissociation build (§A–§C); the opposite-sign learning-rate prediction.
- Format-template prior synthesis: [`docs/project/symposium/20260516_impact_vs_reasonable/postdoc_synthesis.md`](../20260516_impact_vs_reasonable/postdoc_synthesis.md).
- Project anchors: G1/G2 and the four v8-null causes in [`docs/project/project_plan.md`](../../project_plan.md) §4; hypotheses H1–H5 (esp. **H5 modulator timescale**, which binds the ACh block-length constraint) in [`docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md`](../../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) §1.4.

---

*Synthesis by `research-postdoc`, 2026-06-18, for symposium `20260618_env_suitability_neuromodulation`. Adjudicates four professor contributions plus the grounding memo. No architecture, code, or experiment authored here; all hand-offs are named, not executed.*
