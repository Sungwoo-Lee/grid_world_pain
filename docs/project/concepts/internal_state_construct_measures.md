---
title: "Internal state → behaviour: four construct measures, with a signal-detection design for hypervigilance"
topic: construct-validity / pain-modeling / measurement design
status: draft
author: professor-pain-modeling
created: 2026-08-25
last_updated: 2026-08-25
closes_gaps: measurement spec for project_plan §3.1 four categories; sharpens G2 (hypervigilance)
---

# Internal state → behaviour: four construct measures, with a signal-detection design for hypervigilance

> One-line summary: **Of the project's four "pain-like" categories, this environment cleanly supports one (competing-needs conflict), strongly supports a signal-detection form of hypervigilance that no ordinary pain dataset can — because the *ambiguity of the threat cue is randomised per episode* — partially supports avoidance, and only weakly supports recovery. The whole content of a hypervigilance claim is whether injury lowers the agent's decision *criterion* for calling "threat" rather than changing its perceptual *sensitivity*; a single false-alarm rate cannot tell those apart, but the full psychometric curve along the recorded, randomised smell axis can, and this dataset has it.**

## Purpose (plain-language entry point)

The project wants to show a simulated animal behaves "as if in pain" in four ways: it **avoids** danger, it **recovers** after being hurt, it **trades off** hunger against safety, and it becomes **hypervigilant** — jumpy, over-reacting to anything that might be a threat — after injury. This memo designs how to *measure* each of those four from data we already have: 14 trained agents each re-lived one million randomised lives, with every step recorded.

The one I treat most carefully is **hypervigilance**, in the specific form the user asked for: does the agent get more trigger-happy about *ambiguous* signals when it is hurt? The environment gives an unusually clean way to ask this. Predators and harmless rabbits were built to *smell alike*, and how predator-like each rabbit smells is a fresh dice-roll every life. So a "predator-smelling rabbit" is a genuinely ambiguous alarm, and its ambiguity is randomised — the cleanest possible handle. The question becomes: does the agent **hide from a harmless rabbit more often when it is injured**? A hiding response to a rabbit is a *false alarm*. If false alarms rise with injury, the agent has lowered its internal threshold for shouting "threat" under bodily threat — which is close to what the clinical literature calls hypervigilance.

Crucially, "rising false alarms" alone is ambiguous between two very different things: the agent may have **dropped its decision threshold** (true hypervigilance — a *criterion* shift) or it may have **got worse at telling rabbits from predators** (a *sensitivity* loss — a different claim). Separating those two is the entire scientific content of the hypervigilance claim, and this memo shows the dataset *can* separate them, where a naive single-number false-alarm rate could not.

For each of the four constructs below I give: the measure (from recorded columns), whether it is causally identified or merely associational, the specific confound it defeats, a pre-registered prediction that can fail, and what result would *falsify* the construct rather than support it. Then a blunt construct-validity verdict. Everything marked **[existing data]** is runnable now; two items marked **[new collection]** are flagged separately with what they would buy.

## The confound this whole project keeps hitting

Every measure here is built to defeat one specific trap that has burned the project repeatedly: **endogenous internal state**. The agent's own choices produce the injury and hunger you would like to condition on. Hiding blocks eating, so "low nutrition" is partly a *footprint* of hiding, not a cause of it. Being attacked produces injury, so "high injury" marks "a predator was near" — conditioning on injury silently conditions on threat. In pain terms: an agent that defends more when it hurts may just be doing *correct threat inference* ("I was hit, a predator is near"), which is rational caution, **not** hypervigilance.

The escape is the same throughout: **use the reset dice-rolls as instruments.** `start_injury` (0–100) and `start_nutrition` (0–100) are drawn *before the agent acts*, so they are exogenous. A wound the agent *woke up carrying* was not inflicted by any predator, so it carries no threat evidence — and in this run family injury is not directly observable and there is no perceptual-noise route, so a randomised start wound reaches behaviour **only through the lagged nociception percept.** That single fact is what makes an honest hypervigilance test possible here (see §4.4).

> **Assumption to verify in `src/` before trusting any injury contrast (hand to `senior-developer`/`code-reviewer`):** that `start_injury` has *no mechanical effect on action execution* (movement speed, stamina, action masking) — i.e. its only route to the policy is the nociceptor. If injury also slows the agent, the injury→behaviour path is not purely perceptual and every injury contrast below is confounded by a motor effect. The established hiding-drivers work assumed the perception-only route; it should be confirmed, not inherited.

---

## 4. HYPERVIGILANCE — the priority construct **[existing data]**

### 4.1 The signal-detection frame

Treat the agent as running an implicit detection task at every step: *"is there a predator near me that warrants hiding?"*

| | agent hides (`agent_in_bush`) | agent does not hide |
|---|---|---|
| **predator actually near** | hit | miss |
| **only a rabbit near (no predator)** | **false alarm** | correct rejection |

Signal-detection theory decomposes the hide-rate into two independent quantities:

- **sensitivity** `d′` — how well the agent can tell a predator from a rabbit given the smell it receives;
- **criterion** `c` — the amount of threat-evidence at which it decides to hide.

**Clinical hypervigilance is a criterion drop, not a sensitivity change.** The hypervigilant patient is not *better* at detecting threat; they set a *lower bar* for calling something a threat, so both hits and false alarms rise. A pure sensitivity change (better or worse discrimination) is a different phenomenon. So the hypervigilance claim lives or dies on whether injury moves `c`, holding `d′`.

### 4.2 Why a single false-alarm number is not enough — and what fixes it

A single false-alarm rate — one number, "P(hide | rabbit near)" — is one point on a curve. One point cannot separate a criterion shift (curve slides sideways) from a sensitivity change (curve changes slope). **You need the whole curve.** This dataset supplies it, because the thing the agent responds to — the rabbit's predator-likeness — is *recorded and continuously randomised*.

Define, from the run's config, a scalar **predator-likeness** `x` of an animal's smell (the difference between the two odour channels that separate the classes; the established work already derives this and refits it, z ≈ 13.8). Because both olfactory channels of every animal are drawn at reset, `x` is exogenous and recorded per episode. Now build the **psychometric function**

$$
P(\text{hide} \mid x,\ \text{injury bin},\ \text{animal near}) \;=\; \Phi\!\big(k(\text{injury})\,[\,x - c(\text{injury})\,]\big),
$$

fit as a probit for each `start_injury` bin, separately for **rabbit-only** episodes (`n_rabbit=1, n_predator=0`) and **predator-only** episodes (`n_predator=1, n_rabbit=0`).

- **criterion** `c(injury)` = the smell value at which the agent hides half the time (the curve's horizontal position). **Hypervigilance = `c` moves left as injury rises** — the agent hides at *weaker* smell evidence.
- **sensitivity** `k(injury)` = the curve's slope (discriminability). A **sensitivity change** = `k` changes with injury.

A criterion shift slides the curve sideways at fixed slope; a sensitivity change tilts it. **This is exactly the separation the clinical claim requires, and the continuous randomised smell axis is what makes it estimable.**

### 4.3 The identifying check that makes the SDT story clean

The agent **cannot see the animal's label** — it has only the smell (visual range 0). So if the smell is the agent's *sole* decision axis, then at matched `x` the rabbit curve and the predator curve **must coincide**:

$$
P(\text{hide}\mid x, \text{rabbit}) \;\overset{?}{=}\; P(\text{hide}\mid x, \text{predator}).
$$

This is *testable on the data*, and it is the linchpin:

- **If they coincide** → the agent decides on one olfactory evidence axis, and any injury effect is unambiguously a `c`/`k` shift on that axis. Clean SDT.
- **If they diverge** (more hiding to a predator than a rabbit at identical smell) → the agent uses some *non-olfactory* cue correlated with the label — animal movement dynamics, or memory of a prior contact — so the "sensitivity" has a component that is not olfactory perception, and the clean two-parameter SDT reading is compromised. You would then report the criterion shift as *conditional on* this check and flag the residual.

### 4.4 Causal identification, and what "hypervigilance to internal state" can mean here

**Handle:** `start_injury` × `x`, both randomised at reset — a 2-D exogenous design against all reset confounds.

**The killer confound, and how the design defeats it.** The established work already found that when the agent is hurt *by an attack*, defending more is *rational* ("pain ⇒ a predator is near"), not hypervigilance — and that the recurrent policy could carry "I was attacked" in memory with no use of the pain channel at all, so the interoceptive contribution was **not identified** for earned wounds. The randomised start wound breaks both problems at once:

1. a wound woken up with was **not** inflicted by a predator → it carries **no threat evidence**, so extra hiding to a rabbit under a start wound cannot be rational threat-inference;
2. with injury unobservable and no perceptual-noise route in this family, `start_injury` reaches the policy **only through the nociceptor** → for the start-wound handle specifically, the interoceptive channel *is* the identified mediator, unlike the earned-wound case.

So the honest reading of "hypervigilance **to internal state**": the criterion is set by the **lagged, smoothed nociceptive percept**, not by true injury (the agent has no readout of true injury). The buffer's zeroing-at-reset is a *gift* — a start wound is unfelt on step 1 and fades in over ~10 steps — giving a within-episode staircase of *perceived* pain at *declining* true injury, which dissociates percept from state. Index the psychometric curves by `start_injury` (the instrument) and read the reconstructed percept as the dose/mediator.

**Sample restriction (all from recorded columns):** exactly-one-animal episodes; early window (≈ steps 5–20, where a start wound is perceptually present); **zero prior hits** (cumulative `damage`/`hit_predator` = 0, so the *only* injury is the randomised start wound); animal-near defined by **lagged** per-animal position (predator/rabbit position at `t−1` within a small radius) so the agent's own flight cannot manufacture the proximity; `defend = agent_in_bush` at `t`. Truncation check on the window (established recipe).

### 4.5 Pre-registered prediction (can fail)

Restricting to rabbit-only, zero-prior-hit, windowed, near-moments: as `start_injury` rises, the rabbit psychometric curve **shifts left (criterion `c` drops)** — false alarms to predator-like-smelling rabbits rise — **with no change in slope `k`** (no sensitivity change), and the shift is **channel-selective**: concentrated where the rabbit is near, not a uniform lift of baseline hiding. This is consistent with the established interaction result (start-injury barely moves *mean* hiding, −1.0 pp, but *amplifies* the predator-proximity response, +0.83…+1.92 pp/bin) — the SDT design converts that interaction into a criterion estimate.

Run every contrast as a **modulated-vs-unmodulated paired comparison** on the 4 matched FiLM pairs: the project's actual thesis is that the *cognitive layer* produces the signature, so the criterion shift should be **present in the modulated agent and absent/attenuated in its unmodulated twin.** That paired contrast, not the within-agent effect, is the headline for Paper 1.

### 4.6 What FALSIFIES hypervigilance (as opposed to supporting it)

- **No curve movement in the zero-hit window** → the earlier injury×proximity interaction was carried by *earned* injury, i.e. rational threat-inference, and there is no internal-state hypervigilance. Construct fails.
- **Slope change instead of lateral shift** (`k` rises with injury) → injury *sharpens discrimination*. That is a sensitivity change, the opposite of the clinical claim — "injury improves perception", not hypervigilance.
- **Uniform lift across all `x` including the least predator-like rabbit** → a criterion/bias shift that is *non-selective* (general freezing), weaker than channel-selective hypervigilance; report as "global response bias", not hypervigilance.
- **Rabbit and predator curves diverge at matched `x` (§4.3 fails)** → the decision uses a non-olfactory cue; the clean SDT criterion reading does not hold and the claim must be downgraded.
- **Curve moves the *wrong way*** (criterion rises with injury; injured agent *more* tolerant of ambiguous smell) → refutes outright.

### 4.7 Verdict for hypervigilance

**Supported, and this is the best hypervigilance handle in the dataset — arguably better than a standard clinical paradigm can offer, because the cue ambiguity is randomised exogenously.** But it is subtle: the effect is an *interaction*, small in magnitude, windowed, and only clean in the zero-prior-hit sample. It must be run as a criterion-vs-sensitivity psychometric decomposition, **not** as a single false-alarm number, and gated on the §4.3 label-coincidence check. Deterministic argmax actions cost us the graded decision variable (§7) — a real limitation the psychometric-over-recorded-smell approach works around but does not fully replace.

---

## 3. COMPETING NEEDS / CONFLICT — the best-supported construct **[existing data]**

### 3.1 Measure

The forage-vs-safety trade-off, as a **randomised 2×2(×2)**: `start_nutrition` (hunger) × predator presence × *optionally* `start_injury`. Outcome: **P(leave/avoid cover to forage under threat)** — e.g. `P(not agent_in_bush ∧ moving toward food | predator near)`, or directly `ate_food` rate in threatened states, over the early window.

### 3.2 Identification and the confound it defeats

Both drives are **randomised at reset**, so the interaction is causally identified. This **defeats the reverse-causation confound** that wrecks the observational hunger→hiding gradient: contemporaneous low nutrition is a footprint of hiding, but `start_nutrition` is pre-behaviour. The established work already shows randomised hunger acts *opposite* to the observational gradient (hungry agents hide *less*, forage *more*, from the first steps) — the conflict design sharpens this from a main effect into the **hunger × threat interaction** that *is* the trade-off.

### 3.3 Pre-registered prediction (can fail)

The suppression of foraging by threat is **attenuated under randomised hunger** — i.e. a hungry agent accepts more predator risk to eat (a genuine interaction, not two additive main effects). Adding the third axis: **`start_injury` tips the balance toward safety even when hungry** (injured-and-hungry forages *less* than healthy-and-hungry under matched threat) — internal damage state reweighting the trade-off, which is the pain-relevant claim.

### 3.4 Falsification

Hunger and threat act **additively** (no interaction) → there is no "conflict resolution", only two independent reflexes, and "managing competing needs" is not demonstrated. Or hunger does not move foraging-under-threat at all → no motivational competition. Or injury does not reweight the hungry-and-threatened choice → the *internal-state* modulation of conflict (the pain-relevant part) fails even if a hunger×threat interaction exists.

### 3.5 Verdict

**Strongly supported.** Two randomised drives plus randomised threat = a clean causal interaction on existing data, mapping directly onto fear-avoidance / motivational-competition accounts. This is the construct the environment supports best and the one I would lead the "managing competing needs" panel with.

---

## 1. AVOIDANCE (distinct from hiding) **[existing data, with a caveat]**

### 1.1 The measure and the distinction it must earn

The project defines avoidance as *context-conditioned movement away from threat*, explicitly **distinct from bush hiding**. This distinction is load-bearing: the established work shows bush hiding dominates everything, and an agent *pinned* in a bush next to a predator is not "avoiding" in the locomotor sense. So avoidance needs its own measure:

**Smell-gradient descent.** Following lagged high predator-smell exposure, in the *open* (not in a bush), does the agent's next-`k`-step trajectory **reduce the predator-smell intensity** it receives (from the recorded noise-free obs) faster than a movement-null? Avoidance = net decrease in received predator-smell over the next `k` steps, conditional on lagged high intensity and not-in-bush.

### 1.2 The construct-validity caveat (be honest up front)

With visual range 0, the agent's only distal sense is olfaction over a radius-20 field covering the whole map. **Whether the agent can do directional avoidance at all depends on whether that smell carries usable gradient/direction.** If the distal signal is a near-scalar intensity, the agent can still gradient-descend on intensity (move until the smell weakens) — that is a legitimate avoidance measure — but locomotor avoidance may partly *collapse into hiding*, because hiding is the cheaper protective action. Report the smell-intensity-reduction measure, and explicitly test whether it carries variance *beyond* what bush entry explains (partial it out).

### 1.3 Identification, prediction, falsification

**Handle:** the predator's randomised olfactory channels (smell strength/predator-likeness) and `detection_range`; and, for the internal-state question, `start_injury`. Proximity must be **lagged** (endogenous otherwise). **Prediction:** stronger randomised predator-smell → more smell-reducing movement in the open; and injured agents (randomised `start_injury`, zero prior hits) avoid *more* per fear-avoidance. **Falsification:** avoidance measure carries no variance beyond bush entry (construct collapses into hiding); or no `start_injury` effect / injured agent *approaches* more (foraging drive dominates — which would actually be evidence for the *conflict* construct, not avoidance).

### 1.4 Verdict

**Partially supported.** The honest measure (smell-intensity reduction in the open) is computable now, but the construct is at risk of not being separable from hiding given the sensory setup. Run it; if it adds nothing over bush entry, say so and fold avoidance into the hiding story rather than claim a separate signature.

---

## 2. RECOVERY — the weakest construct; say so **[existing data]**

### 2.1 What "recovery" means in pain science, and why the fit is poor

In the pain literature, **recovery is the *resolution* phase** — the protective repertoire winding *down* as tissue heals — and its pathological mirror is *failure to recover* (chronic pain; the project's H5). A recovery construct therefore needs the protective state to **track the healing trajectory**. This environment supports that only weakly:

1. **Healing is fast** (~20 steps) — little separation between "wound present" and "recovered", so the resolution curve is compressed.
2. **The healing-rate manipulation is confounded.** The ten agents differ in `recovery_accel_rate`, but the established analysis shows this is *not causally attributable* — one run per level, node collinear with parameter, and the mechanistic chain broken (resting does not even trend with the bonus, ρ = +0.25, ns). So the cross-agent recovery contrast is **associational and confounded**, not identified.
3. **There may be no distinct "recovery behaviour."** If resting does not respond to the healing incentive, the agent may lack a dedicated recovery mode at all.

### 2.2 The honest weaker claim, and its measure

Drop "recovery" and claim **"post-damage behavioural relaxation time"**: align defensive behaviour (bush hiding) on a damage event and measure the **decay time back to baseline**. The established work already has this curve (+0.9 pp all-events, returning within ~6 steps; +4.6 pp on a collider-selected subset that stays elevated — and the collider inflation is itself a warning). The *internal-state* version: does the relaxation time **scale with randomised `start_injury` magnitude**? A bigger randomised wound fades perceptually on a *known* buffer schedule; if the defensive state follows the percept's fade rather than true injury, that is "the protective state tracks the interoceptive signal" — the strongest recovery-flavoured statement the data will bear.

### 2.3 Prediction / falsification

**Prediction:** post-damage defensive relaxation time increases with randomised wound magnitude and follows the nociception buffer's decay schedule. **Falsification:** relaxation time is independent of wound magnitude, or follows true injury rather than the percept → no interoceptively-driven recovery; report only a fixed transient response to a transient cue.

### 2.4 Verdict

**Weakest of the four; do not call it "recovery" in the pain-science sense.** The clinical meaning (resolution-of-protection, failure = chronicity) is not cleanly supported: healing is fast, the rate manipulation is confounded, and the agent may lack a recovery mode. The defensible claim is **behavioural relaxation time after damage**, optionally scaled by randomised wound size. The chronic-pain persistence claim (H5) needs the clamp experiment below, not this data.

---

## Construct-validity verdict (summary)

| Construct | Verdict | Cleanest handle | Honest claim if the strong one fails |
|---|---|---|---|
| **Competing-needs conflict** | **Validated** | randomised `start_nutrition` × predator (× `start_injury`), 2×2(×2) | — (this one holds) |
| **Hypervigilance** | **Supported, subtle** | randomised `start_injury` × recorded rabbit predator-likeness `x`; psychometric criterion-vs-sensitivity decomposition | "channel-selective response-bias shift" if slope-clean criterion shift not isolable |
| **Avoidance** | **Partial** | randomised predator smell / `detection_range`, lagged proximity | fold into hiding if no variance beyond bush entry |
| **Recovery** | **Weak — being stretched** | (rate manipulation confounded) randomised `start_injury` magnitude only | "post-damage behavioural relaxation time", not "recovery" |

**Direct answers to the three guardian questions:**

1. **Which are legitimate, which stretched?** Conflict is legitimately supported and clean. Hypervigilance is legitimately supported *as a signal-detection criterion shift* and is arguably better here than in a standard paradigm because the cue ambiguity is randomised — but it is subtle and must be run as a psychometric decomposition. Avoidance is at risk of collapsing into hiding given the visual-range-0 sensory setup; run it, but be prepared to fold it in. **Recovery is being stretched**: the pain-science meaning (resolution of protection as tissue heals; chronicity as failure) is not cleanly supported, because healing is fast, the healing-rate manipulation is confounded, and resting does not respond to the healing incentive. The honest weaker claim is "post-damage behavioural relaxation time."

2. **Is SDT the right frame, and can we separate criterion from sensitivity?** Yes, SDT is exactly right for the rabbit false-alarm measure. **A single false-alarm rate cannot** separate criterion from sensitivity — that distinction is the whole content of the hypervigilance claim. The **full psychometric curve along the recorded, randomised smell axis can**: criterion = lateral position, sensitivity = slope. This separation is valid **conditional on** the §4.3 check (rabbit and predator hide-curves coinciding at matched smell, which they must if smell is the agent's only decision axis — testable). If that check fails, a non-olfactory cue contributes and the separation is compromised; report accordingly.

3. **No direct injury readout — only lagged smoothed nociception. What can "hypervigilance to internal state" mean?** It means hypervigilance to the **nociceptive percept**, not to injury — the agent has no readout of true injury, so the criterion can only be set by the smoothed, lagged buffer. This is not a weakness but a *sharpening*: because a randomised start wound was not inflicted by any predator and (in this family) reaches the policy *only* through the nociceptor, `start_injury` is a clean instrument for "perceived pain", and the buffer's fade-in dissociates percept from true state within an episode. The claim must be stated at the percept level, and it becomes the cleanest interoceptive identification in the dataset — unlike earned wounds, where memory-of-attack cannot be ruled out.

---

## New collection — flagged separately

- **[new collection] Nociception clamp / counterfactual replay.** Replay the trained agent with the interoceptive input **clamped to a constant or set to a falsified value**, holding the world fixed. This is the only way to move from "the criterion shift is *coupled to* the pain percept" to "the pain percept *causes* it", ruling out the recurrent memory-of-attack alternative for *earned* wounds and settling H5 (chronic persistence). The established hiding-drivers doc already names this as the recommended next experiment; it is the definitive hypervigilance and recovery test. Buys: causal isolation of the interoceptive channel. Hand to `experiment-designer` + `senior-developer`.
- **[new collection] Record the pre-argmax hide/defend logit (or value), not just the deterministic action.** Deterministic argmax discards the graded internal decision variable. Logging the policy's continuous decision variable would let criterion and sensitivity be separated *within a single state*, rather than reconstructed across the randomised smell axis, and would sharpen every SDT estimate here. Buys: a direct decision-variable readout for SDT. Small pipeline change. Hand to `senior-developer`.
- **[new collection, cheap] Perceptual-noise contrast.** The `nmn_tempceil10` family has `perceptual_noise.enabled: True` (injury degrades the senses) — a *second* route from injury to behaviour. Comparing criterion shifts with the noise route on vs off would test whether hypervigilance needs the interoceptive channel specifically. Requires the legacy config branch noted in the hiding-drivers doc. Hand to `experiment-designer`.

## Anchors

- [project_plan.md §§2–4, four categories, G2](../project_plan.md)
- [NEUROMODULATION_ALGORITHM.md §1.4 H1 (perceptual amplification), H4 (coherence), H5 (chronicity)](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)
- [hiding-drivers factor analysis](../../experiments/active/trajectory_factors/a01_hiding_drivers.md) — the established confound-cleared base this builds on
- [pain_vs_nociception_construct.md](pain_vs_nociception_construct.md); [active_inference_hypervigilance.md](active_inference_hypervigilance.md) — the channel-selective-gain fingerprint this operationalises

## Next steps

- **`experiment-analyzer`** — implement §§3, 4, and 1 on existing data (conflict 2×2×2, rabbit psychometric criterion-vs-sensitivity, avoidance-vs-hiding partialling), as **modulated-vs-unmodulated paired contrasts** on the 4 FiLM pairs. Run the §4.3 label-coincidence check *first*; it gates the whole SDT reading.
- **`senior-developer` / `code-reviewer`** — verify the §"confound" assumption that `start_injury` reaches the policy only through the nociceptor (no mechanical motor effect); scope the two [new collection] items.
- **`experiment-designer`** — design the nociception-clamp replay (the definitive hypervigilance/recovery/H5 test).
