---
title: "Symposium contribution — does the env engage the BIOLOGICAL construct behind each Doya-2002 knob?"
session: 20260618_env_suitability_neuromodulation
contributor: professor-neuromodulation
date: 2026-06-18
grounding: docs/project/symposium/20260618_env_suitability_neuromodulation/00_env_capability_grounding.md
knobs:
  - "NA → effective temperature (Aston-Jones & Cohen 2005 adaptive gain)"
  - "ACh → effective learning rate (Yu & Dayan 2005 expected uncertainty)"
  - "DA → effective TD-error gain (Dabney 2020; Collins & Frank OpAL; Niv 2007)"
  - "5-HT → effective discount (Tanaka 2007 timescale gating)"
verdict_axis: "biologically-faithful-as-is | faithful-with-caveats | construct-not-engaged"
---

# Symposium contribution — does the environment engage the BIOLOGICAL construct behind each Doya knob?

## 1. Plain-English entry point

The shared grounding memo (the postdoc's "can the env do it at all?" pass) already told us **what the environment is mechanically capable of**: it can vary threat (so it can express a noradrenaline-style exploration signature), it can shuffle its statistics between episodes (a partial acetylcholine-style "learn faster when the world changes" signature), it can make some paths riskier than others (a partial dopamine-style signature), and it **cannot** pay reward after a delay (which the memo flags as a blocker for the serotonin-style "patience" signature). Those verdicts are about **mechanics**.

My job is the **biology check** — a different question. A manipulation can be mechanically present and still fail to engage the brain system the Doya (2002) knob actually stands for. "Doya (2002)" here is Kenji Doya's proposal that four chemical messengers in the brain — **noradrenaline** (NA, the brain's alarm/arousal chemical), **acetylcholine** (ACh, the "the world just changed, pay attention" chemical), **dopamine** (DA, the reward-and-motivation chemical), and **serotonin** (5-HT, the patience/restraint chemical) — each set one tuning parameter of a learning agent (exploration, learning rate, reward sensitivity, patience respectively). The question I answer for each knob is: **does the specific thing the env does actually wake up that brain system the way the biology requires, or does it merely look like it on paper?**

The headline, knob by knob:

- **NA / exploration — faithful-with-caveats.** Graded threat is the *right kind* of stimulus for the locus coeruleus (LC, the brainstem nucleus that releases NA). But the env's threat is something the agent itself controls by moving toward or away from a predator. The single most diagnostic LC signature — a sharp **phasic burst at the moment the situation changes** (predator appears, regime flips) — is present in principle but is **not isolated or measurable** the way the canonical experiments isolate it. The construct is engaged; the cleanest signature is not yet readable.
- **ACh / learning rate — construct largely not engaged.** ACh's "expected uncertainty" job is specifically about *structured, learnable* volatility — stable stretches punctuated by change-points. The env only re-shuffles its parameters **independently every episode** (statisticians call this i.i.d.). That is unstructured noise, not volatility-with-structure, and it is exactly the regime where the ACh-vs-NA division of labour says ACh should *not* be the system that responds. So the env as-described engages the construct only weakly.
- **DA / reward sensitivity — engaged, but it is partly the wrong system.** Food reward is fixed at +1; the only thing that varies is **damage** (cost). A risk task built out of cost-variance does engage a prediction-error system — but in the brain, **cost/aversive** prediction errors are substantially a **serotonin and lateral-habenula** story, not purely a dopamine one. So a damage-variance task is a real distributional-prediction-error task, but labelling it cleanly "dopamine" overclaims; it is partly an aversive-system task.
- **5-HT / patience — the red verdict is softenable, and this is my most important point.** The grounding memo marks reward-delay "NOT SUPPORTED" and therefore the serotonin knob untestable. But **literal reward delay is not the only — or even the most biologically faithful — way to engage serotonin's role.** Tanaka (2007) and the dorsal-raphe patience literature describe 5-HT as gating **which timescale of valuation dominates**, and that timescale shift is driven in real animals by **internal state — threat, pain, energy** — not by an experimenter inserting literal latency. The env already varies threat, injury, and satiation. So there is a biologically faithful 5-HT manipulation the env **can** do today, without the reward-delay buffer. I argue the red should become amber.

Section 2 takes each knob in turn. Section 3 gives the per-knob verdict block. Section 4 lists the specific biologically faithful manipulations that would close each gap.

---

## 2. Per-knob biological analysis

### 2.1 NA → effective temperature — is emergent graded threat a faithful LC elicitor?

**What the brain system actually is.** Noradrenaline is released by the locus coeruleus, a tiny brainstem nucleus with brain-wide projections. Aston-Jones & Cohen's (2005) adaptive-gain theory describes the LC operating in two modes that the project's "temperature" knob conflates at its peril:

- a **phasic** mode — a brief, sharp burst time-locked to a behaviourally significant, often surprising, event (target detection, threat onset, a change in task contingency). Phasic LC transiently *sharpens* the policy (raises gain, *lowers* effective temperature, promotes exploitation of the just-detected signal).
- a **tonic** mode — a sustained elevation of baseline firing. High tonic LC *disengages* from the current task, *raises* effective temperature, promotes exploration / task-switching. The relationship between tonic level and performance is the Yerkes-Dodson inverted-U.

The key biological fact for this symposium: **the canonical LC signature is the phasic burst at a regime change**, and the canonical exploration effect is the *tonic* shift. These are two timescales doing two things. A single scalar "effective temperature" only captures the tonic axis.

**Is the env's threat a faithful elicitor?** Graded proximity-driven threat is genuinely the right *category* of stimulus — predators are aversive, salient, brain-wide-arousing in exactly the way LC responds to. The smell/vision/contact-pain channels give the agent the sensory evidence an LC-analog would integrate. On the **tonic** axis (sustained threat → sustained arousal → exploration/exploitation shift), the env's config sweep (near vs. far patrol, large vs. small detection radius) is a defensible way to set a graded *tonic* arousal level across conditions. For the inverted-U sweep the postdoc green-lit, this is fine.

**Where it falls short biologically.** Two gaps.

1. **No experimenter-set arousal dial; arousal is agent-controlled and emergent.** In the canonical LC paradigms arousal is *imposed* (task utility manipulated, threat onset scheduled) so that the LC response can be cleanly attributed. In the env the agent **regulates its own threat by fleeing** — it is a closed loop. This is not fatal (it is arguably *more* ecological), but it means "tonic arousal level" is a **measured, endogenous** quantity, not a manipulated independent variable. Construct-validity-wise, you cannot say "we set arousal to X and observed temperature Y"; you can only say "across configs with different threat affordances, inferred arousal and inferred temperature co-vary." That is a correlational, not interventional, engagement of the construct.

2. **The phasic-burst-at-regime-change signature is present but not isolated.** A predator flipping from patrol into pursuit (`dist <= hunt_detect`, grounding §4.1) is *exactly* the kind of discrete, salient regime change that should drive a phasic LC burst. So the env *contains* the eliciting event. But nothing in the described setup **isolates** it: there is no event-locked marker, no separation of the phasic transient from the tonic background, and the "temperature" readout is a single scalar with no timescale decomposition. The most diagnostic NA signature — *transient gain increase locked to threat onset* — is therefore latent in the dynamics but not measurable as-described.

**Verdict: faithful-with-caveats.** The tonic-arousal → temperature axis is faithfully elicited by graded threat and adequate for an inverted-U sweep. The phasic axis — the *signature* that would most convincingly say "this is NA, not just generic adaptive gating" — requires event-locking the predator-onset transition and decomposing the modulator's response into phasic and tonic components. Without that, the project can claim adaptive-gain-like behaviour but not specifically LC-NA-like dynamics.

### 2.2 ACh → effective learning rate — does i.i.d. resampling engage expected uncertainty at all?

**What the brain system actually is.** Yu & Dayan (2005) is the load-bearing theory here, and it is precise about what ACh signals: **expected uncertainty** — the *known, learnable* unreliability of cues within an otherwise stable context. Its opponent partner is NA, which signals **unexpected uncertainty** — a gross, surprising violation indicating the context itself has changed (a change-point). The whole point of the Yu-Dayan account is the **division of labour between two structured forms of uncertainty**:

- ACh tracks the slow, estimable noise level *inside a stable regime* and tunes how much you trust the prior vs. the current likelihood → effective learning rate within a regime.
- NA detects when the regime has *broken* (a change-point) and triggers a reset / fast relearning.

Crucially, **both constructs presuppose regime structure.** Expected uncertainty is only definable relative to a stable generative context whose noise level can be learned. Unexpected uncertainty is only definable as a *departure* from that context.

**Does i.i.d. per-episode resampling engage this?** Largely **no**, and this is a sharper "no" than the grounding memo's mechanical "PARTIAL." The env re-draws the five predator parameters independently at every reset (grounding §4.2). Independent re-draw each episode is **unstructured stochasticity, not volatility-with-structure**. From the agent's standpoint there is no stable regime whose noise level can be *learned* and then *exploited* — every episode is a fresh independent sample. In Yu-Dayan terms:

- There is no **expected uncertainty** to estimate, because there is no persistent regime over which "expected" is defined. The optimal strategy under i.i.d. re-draw is a *fixed* Bayes-optimal prior, with **no episode-to-episode learning-rate modulation** — which is precisely the behaviour ACh is *not* needed for.
- There is no **change-point**, so the NA/unexpected-uncertainty partner has nothing to detect either.

In other words, i.i.d. resampling sits in the one corner of the uncertainty plane where *neither* ACh nor NA should modulate anything — the agent should converge to a static optimal policy. An effective-learning-rate signature is not predicted to appear, so observing none would be uninformative and observing one would be hard to attribute to the ACh construct.

**The within-episode drift caveat.** The grounding memo notes resource respawn re-draws position + chemical signature mid-episode if `res_property_std > 0` (§4.2). This is closer — a *within-episode* drift in the smell→location map is a genuine (slow) non-stationarity an ACh-analog could track. But it is unstructured drift, not the stable-then-volatile **block structure** or scheduled **change-point** that the expected-vs-unexpected-uncertainty dissociation requires. Drift alone engages a weak, undifferentiated "the world is a bit noisy" signal — not the specific expected-uncertainty construct.

**Verdict: construct-not-engaged (as-described); reachable only with structured volatility.** The ACh knob is the one where the biology is most demanding and the env is furthest from meeting it. To engage the construct you need **regime structure the agent can learn** — stable blocks vs. volatile blocks, or scheduled change-points — so that expected uncertainty (ACh) is dissociable from unexpected uncertainty (NA). i.i.d. resampling provides neither.

### 2.3 DA → effective TD-error gain — is a damage-variance task a dopamine story or an aversive-system story?

**What the brain system actually is.** Three threads matter:

- **Dabney et al. (2020, Nature)** — distributional reinforcement learning in dopamine: a *population* of DA neurons with heterogeneous optimism/pessimism (asymmetric scaling of positive vs. negative RPEs) collectively encodes the *distribution* of reward, not just its mean. This is the cleanest neural evidence for distributional codes and the strongest modern anchor for "TD-error gain sensitive to reward variance."
- **Collins & Frank OpAL** — D1 ("Go") and D2 ("NoGo") striatal pathways learn from positive and negative prediction errors with separable gains; tonic DA sets the balance, controlling risk/vigour asymmetry.
- **Niv (2007)** — tonic DA sets response *vigour* (how fast/hard you act), an opportunity-cost signal.

The decisive biological caveat: **DA's prediction-error code is strongly appetitive/asymmetric.** Dabney's optimistic/pessimistic asymmetry is *about reward*. The classic RPE is a *reward* prediction error. When the prediction error is about **punishment / cost / aversive outcome**, the neural division of labour shifts substantially: aversive prediction errors are carried prominently by the **lateral habenula → RMTg → DA dip** pathway and by the **serotonin** system (Daw, Kakade & Dayan 2002's appetitive-aversive opponency, with DA appetitive and 5-HT aversive). DA does respond to aversive events, but a task whose *only* variance is on the **cost/damage** side is not a clean dopamine paradigm.

**Does the env's damage-variance task engage the DA construct?** Partly — and partly the wrong system.

- It *does* engage a **distributional** prediction-error code in the abstract: a `damage: [0, 40]` region vs. a `damage: [10, 10]` region with matched mean is a genuine variance manipulation, and an agent sensitive to it must represent more than the mean — the Dabney distributional claim in spirit.
- But because the variance lives on **cost**, the construct it most directly engages is the **aversive** opponent system (5-HT / habenula), not the appetitive DA-RPE system. Food reward is fixed at +1 (grounding §4.3), so there is **no appetitive reward-variance** at all — the one thing the Dabney optimistic/pessimistic asymmetry is specifically about.
- The OpAL D1/D2 reading survives better: D2/"NoGo" learning *is* the punishment-sensitive pathway, so a cost-variance task plausibly probes the D2 side of the OpAL balance. But probing only the D2/aversive side is a partial, asymmetric engagement of the OpAL construct, not the full Go/NoGo balance.

**Where it falls short biologically.** The env can build an asymmetric *risk* task, and risk-sensitivity is genuinely DA/5-HT-relevant — but attributing the resulting signature to **dopamine** specifically overclaims, because (a) the variance is aversive not appetitive, and (b) the appetitive RPE-variance that DA's distributional code is built for is absent (fixed +1). A cleaner DA story needs **reward-magnitude variance on the appetitive side** (variable food gain), which the grounding memo confirms is absent.

**Verdict: faithful-with-caveats, with a labelling correction.** The task engages a distributional / risk-sensitive prediction-error construct — real and publishable — but it is **as much an aversive-system (5-HT / habenula / D2-NoGo) probe as a dopamine probe.** Calling it cleanly "DA → TD-error gain" mislabels the substrate. Either add appetitive reward-variance (variable food gain) to make it a DA story, or rename the knob to "aversive-prediction-error gain (DA/5-HT opponent)" to match what the cost-variance task actually engages.

### 2.4 5-HT → effective discount — is literal reward delay the only way? (the red is softenable)

**This is the verdict I most want the symposium to revisit.** The grounding memo marks 5-HT/discount NOT SUPPORTED because no reward-delay buffer exists. That conclusion follows only if **literal reward latency is the unique operationalisation of the serotonin/discount construct.** Biologically, it is not — and arguably it is not even the most faithful one.

**What the brain system actually is.** Doya (2002 §3.2) maps 5-HT to the discount factor γ. The mechanism, as refined by:

- **Tanaka et al. (2007)** — serotonergic manipulation shifts which *timescale of valuation* dominates: low 5-HT biases toward immediate/short-horizon value (steep discounting); high 5-HT toward delayed/long-horizon value (shallow discounting). The fMRI signature is a **rostral-caudal gradient of striatal/cortical representations of value at different timescales**, with 5-HT gating which part of the gradient drives behaviour. The construct is **timescale gating**, not literal latency.
- **Dorsal-raphe patience (Miyazaki, Cohen et al.)** — dorsal-raphe 5-HT neuron activity *sustains waiting for a delayed reward*; it controls the *willingness to wait*, i.e. the effective horizon, behaviourally.
- **Dayan & Huys / Daw** — 5-HT as behavioural inhibition and aversive opponent: under threat/aversive load, behaviour shifts toward inhibition and (relevant here) toward **short-horizon, defensive valuation**.

The unifying read: **5-HT sets the effective horizon / which timescale of value dominates, and in real animals this is driven by internal state — energy, threat, pain, aversive load — not by an experimenter inserting latency between action and reward.** A starving or threatened animal *behaviourally* discounts the future more steeply (short horizon, grab-the-immediate); a sated, safe animal can afford a long horizon. That internal-state-driven horizon shift **is** the 5-HT construct in its ecological form.

**What the env CAN do that engages this faithfully.** The env varies exactly the internal states that drive the biological horizon shift:

- **Threat / predator proximity** — a high-threat config should, under the 5-HT/Daw account, push the agent toward a **short effective horizon** (defensive, immediate-survival valuation). A low-threat forage config affords a **long horizon** (invest in distant food, explore).
- **Injury / accumulated pain** (the interoceptive nociception channel `c[3]`) — accumulated aversive load is precisely the signal Dayan-Huys behavioural-inhibition 5-HT integrates; high injury should shorten the effective horizon.
- **Satiation / nutrition** (`c[2]`, `c[1]`) — energy state is the canonical driver of intertemporal-choice horizon in foraging theory; low satiation → steep discounting (urgency), high satiation → shallow.

The **measurable signature** does not require literal reward delay. It is the **effective discount factor γ_eff inferred from behaviour as a function of internal state** — e.g. fit the agent's revealed time-preference (willingness to traverse distance / accept near-term cost for distant food) across satiation/threat/injury levels, and test whether γ_eff *decreases* as threat/injury rise and satiation falls. Tanaka's construct is about *which timescale dominates*; in a spatial foraging task, the timescale of value is naturally expressed as **spatial-temporal reach** — how far the agent will plan/travel for reward. The env expresses this natively: distance-to-food *is* a delay-to-reward in a gridworld (every step toward distant food is a step of deferred reward and accumulated metabolic/threat cost).

**Why this is more faithful than a reward-delay buffer, not just a workaround.** A reward-delay buffer would implement the *RL textbook* γ (literal temporal discounting of a scalar) but would be **biologically the weaker analog** — it would test whether the agent discounts an artificially-deferred scalar, with no internal-state driver. The biological 5-HT construct is *state-dependent horizon modulation*: the interesting claim is "the modulator reads internal aversive/energy state and shifts the effective horizon," which is exactly the project's thesis (interoceptive `c` → behavioural-hyperparameter signature). The reward-delay buffer would actually **bypass** the interoceptive-modulation story; the internal-state-driven horizon shift **is** the interoceptive-modulation story.

**The honest caveat.** Inferring γ_eff from spatial foraging behaviour is **confounded** in a way literal reward delay is not: distance-to-food co-varies with metabolic cost, threat exposure, and exploration, so a measured horizon shift could reflect risk-sensitivity or energy-budgeting rather than discounting per se. Disentangling γ_eff from these requires careful task design (e.g. matched-distance paths differing only in a deferral structure, or model-based fits that separate discount from risk). This is a real construct-validity burden — but it is the *normal* burden of inferring discounting from foraging, well-trodden in behavioural ecology, **not** an unsupported capability.

**Verdict: construct-engaged-with-caveats — the red should become amber.** Literal reward delay is one operationalisation of the discount construct and the env lacks it. But the **biologically primary** form of the 5-HT/discount construct — internal-state-driven effective-horizon modulation — is **directly expressible** in the env via threat/injury/satiation-driven shifts in revealed time-preference, and it aligns *better* with the project's interoceptive-modulation thesis than a reward-delay buffer would. I recommend the symposium downgrade the 5-HT verdict from NOT-SUPPORTED to **faithful-with-caveats via the state-dependent-horizon route**, with literal reward delay noted as a cleaner-but-less-biological supplement.

---

## 3. Per-knob verdict block

| Knob | Doya construct | Biological elicitor required | Does the env's manipulation engage it? | Verdict |
|---|---|---|---|---|
| **NA → temperature** | LC tonic/phasic adaptive gain (Aston-Jones & Cohen 2005) | Salient/aversive arousal, ideally with isolable phasic-burst-at-regime-change + tonic baseline | Graded threat is the right elicitor and sets a tonic-arousal axis; but arousal is agent-controlled/emergent (correlational, not interventional) and the diagnostic phasic burst is present-but-not-isolated | **faithful-with-caveats** |
| **ACh → learning rate** | Expected uncertainty within a stable regime; opponent to NA's unexpected uncertainty (Yu & Dayan 2005) | *Structured* volatility — stable regime with learnable noise, vs. change-points | i.i.d. per-episode resampling is unstructured noise, the one regime where neither ACh nor NA should modulate; no learnable expected-uncertainty exists | **construct-not-engaged** (as-described) |
| **DA → TD-error gain** | Appetitive, asymmetric (optimistic/pessimistic) reward-distribution code (Dabney 2020; OpAL; Niv 2007) | Variance on the **appetitive reward** side; Go/NoGo balance | Variance lives on **cost/damage** (food fixed at +1); engages the **aversive** opponent (5-HT/habenula/D2-NoGo) as much as DA; no appetitive reward-variance at all | **faithful-with-caveats** (mislabelled — it is partly an aversive-system task) |
| **5-HT → discount** | Effective-horizon / timescale gating driven by internal state (Tanaka 2007; dorsal-raphe patience; Dayan-Huys) | A driver of horizon shift — biologically, internal aversive/energy state, **not** literal latency | Threat/injury/satiation (the interoceptive `c` channels) directly drive horizon shifts; revealed time-preference inferrable from spatial foraging; literal reward delay absent but not required | **faithful-with-caveats** (the NOT-SUPPORTED red is softenable) |

---

## 4. What biologically faithful manipulation would close each gap

Handoffs named for `experiment-designer` (task/config) and `senior-developer` (any new env code). Nothing here edits `src/`/`configs/` — these are specifications.

**NA (close the phasic gap).** No new env code needed; the eliciting event (predator patrol→pursuit transition) already exists. What is needed is **event-locked measurement**: mark the step at which `dist <= hunt_detect` first fires, and decompose the modulator's response into a **phasic** (transient, event-locked) and **tonic** (slow baseline) component, then test phasic↔threat-onset and tonic↔sustained-threat separately. This is an analysis/instrumentation task (`experiment-designer` + whatever logs `mod_h` time-courses), not an env change. Without it the project can only claim adaptive-gating, not LC-NA dynamics specifically.

**ACh (the construct genuinely needs new structure).** This is the one gap config alone cannot close. Engaging expected uncertainty requires **regime structure the agent can learn**: either (i) a **blocked design** — a stable block of episodes (narrow predator-parameter ranges) followed by a volatile block (wide ranges or scheduled mid-episode switches), so learning rate before vs. after a change-point is comparable; or (ii) a **within-episode scheduled change-point** keyed on `current_step` (predator behaviour flips at step k). Both need new env code (a schedule/block controller). Hand to `senior-developer`; the grounding memo already flags this as the new-code item. Critically — and this is the biological point the grounding memo did not make — without regime structure there is **no expected-uncertainty construct to engage at all**, so this is not "PARTIAL/nice-to-have," it is "required for the construct to exist."

**DA (close the appetitive-variance gap, or relabel).** Two options. (i) **Add appetitive reward-magnitude variance** — make food gain stochastic (`food_gain: [min, max]`), so a low-variance steady-food region vs. a high-variance jackpot-food region with matched mean becomes constructible. *This* is the clean Dabney/distributional-DA task and it is a modest config-schema addition (hand to `senior-developer` if `food_gain` is currently scalar-only). (ii) If only cost-variance is available, **rename the knob** to "aversive/risk prediction-error gain (DA-5-HT opponent)" and frame results as probing the D2/NoGo + aversive-opponent system, citing Daw-Kakade-Dayan 2002 opponency — honest, and still publishable.

**5-HT (use the state-dependent-horizon route; no env change).** Do **not** build the reward-delay buffer first. Instead: run a **config sweep over the internal-state drivers the env already has** — threat level (predator proximity/count), accumulated injury, baseline satiation — and **infer γ_eff from revealed time-preference** (willingness to traverse distance / accept near-term metabolic-and-threat cost for distant food) at each level. Pre-register the directional prediction: γ_eff *falls* as threat and injury rise and satiation falls (Tanaka/Daw-Huys short-horizon-under-aversive-load). Control the distance↔risk↔energy confound with matched-distance path designs or model-based discount/risk separation. Hand to `experiment-designer`. A literal reward-delay buffer (the grounding memo's ring-buffer template) remains a valid **supplementary, cleaner-but-less-biological** check — build it only if the state-driven route's confounds prove intractable.

---

## 5. Cross-references to the sibling professors

- **`professor-bayesian-brain`** owns the precision-as-inference reading. The NA phasic/tonic decomposition I ask for (§4) is the same quantity they would read as *precision dynamics*; we should share the instrumentation. My ACh "no learnable regime → no expected uncertainty" point is the substrate-level statement of their "no volatility to do inference over" point — we should converge on flagging ACh as the hardest gap.
- **`professor-pain-modeling`** is directly implicated by my 5-HT argument: the state-dependent-horizon route uses **injury / accumulated nociception** (`c[3]`) as a horizon driver, and short-horizon-under-aversive-load is a pain-relevant behavioural signature (fear-avoidance, defensive foreshortening of planning). The DA relabelling (cost-variance = aversive-system probe) is also their territory — descending aversive modulation (5-HT, NA, opioidergic) is the substrate for the cost-variance task. Coordinate the 5-HT and DA verdicts with them.
- **`professor-rl`** owns whether γ_eff is identifiable from the foraging behaviour and what estimator separates discount from risk/energy — the confound I flag in §2.4 is theirs to resolve. The DA appetitive-variance task design (matched-mean risky-vs-safe arms) is also a return-estimator question for them.

---

## 6. Biological-plausibility verdict (summary block)

- **NA → temperature:** **PARTIAL / faithful-with-caveats.** Right elicitor (graded threat), tonic axis adequate for an inverted-U; but arousal is emergent/agent-controlled (correlational only) and the diagnostic phasic-burst signature is present-but-not-isolated. Fix = event-locked phasic/tonic decomposition (analysis, no env change).
- **ACh → learning rate:** **OVERCLAIMED if asserted on i.i.d. resampling / construct-not-engaged.** i.i.d. per-episode re-draw is the one regime where neither ACh nor its NA opponent should modulate anything; expected uncertainty is undefined without a learnable stable regime. Fix = blocked or change-point volatility structure (new env code — genuinely required, not optional).
- **DA → TD-error gain:** **PARTIAL / faithful-with-caveats and mislabelled.** A cost-variance task engages a distributional/risk prediction-error code, but on the **aversive** side (5-HT/habenula/D2-NoGo), not the appetitive DA-RPE the Dabney code is built for; appetitive reward-variance is absent (food fixed +1). Fix = add stochastic food gain (DA story) OR relabel as aversive-opponent prediction-error gain.
- **5-HT → discount:** **DEFENSIBLE via the state-dependent-horizon route — the red verdict is softenable.** Literal reward delay is absent but is the *weaker* biological analog; the *primary* 5-HT construct (internal-state-driven effective-horizon / timescale gating, Tanaka 2007) is directly expressible via threat/injury/satiation-driven shifts in revealed time-preference, and it aligns better with the project's interoceptive-modulation thesis than a reward-delay buffer would. Fix = config sweep over internal-state drivers + γ_eff inference (no env change); reward-delay buffer optional supplement.

---

*Contribution by `professor-neuromodulation`, 2026-06-18. Arguments built on the shared grounding memo `00_env_capability_grounding.md`; biological anchors: Aston-Jones & Cohen 2005 (LC adaptive gain), Yu & Dayan 2005 (expected vs. unexpected uncertainty), Dabney et al. 2020 + Collins-Frank OpAL + Niv 2007 (DA distributional/vigour), Tanaka et al. 2007 + Daw-Kakade-Dayan 2002 + Dayan-Huys (5-HT timescale gating / aversive opponency). Named-missing key papers the references corpus should hold for these claims: Tanaka et al. 2007 (Nat Neurosci, serotonin timescale gating), Miyazaki/Cohen dorsal-raphe-patience series, and Matsumoto & Hikosaka lateral-habenula aversive-RPE — these underwrite the 5-HT softening and the DA relabelling respectively.*
