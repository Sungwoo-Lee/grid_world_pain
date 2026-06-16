# Assay-as-measure: gridworld test environments that *force* a predator-vs-rabbit discrimination read-out

> **One-line summary.** The predator-vs-rabbit study kept asking detection-theory questions (d′, criterion, efficiency, mutual information) of *naturalistic* rollouts — where the agent wanders freely and the experimenter computes statistics after the fact. That is fragile: the signal hides in conditional sub-states, the encounter geometry is uncontrolled, and a null is unreadable. This memo flips it. It designs **held-out TEST environments whose layout is itself the measuring instrument** — the agent is *forced* to commit to an action that reveals which class it inferred, the cue is *titrated* on a single axis so a psychometric slope falls out, and the available information is *known by construction* so agent behaviour can be reported as efficiency against a Bayesian ceiling. Five reusable assays, each instantiable in the 2D gridworld today.

## Purpose — what this memo is for, in plain English

The project spent June asking: *does the trained agent treat a harmful animal (a "predator", whose touch injures it) differently from a harmless one (a "rabbit", whose touch does nothing) — before contact?* The two animals were made identical on everything sensable at a distance (same smell, same chase, same speed, same strike-and-retreat), so any pre-contact difference could only come from the agent *inferring* danger. The verdict came back **"class-blind"** — the agent reacts to being hurt, it does not anticipate. But two sibling memos ([behavioural_discrimination_detectability_fable.md](behavioural_discrimination_detectability_fable.md), [behavioral_discrimination_measures.md](behavioral_discrimination_measures.md)) argued that verdict is *unfinished*: a "class-blind" reading from averaged naturalistic rollouts cannot tell apart three very different worlds — (i) the agent **can't** discriminate, (ii) it **can** but its policy doesn't act on it ("knows but won't act"), and (iii) **no information ever existed** to discriminate.

Those memos fixed the *statistics*. This memo fixes the *task*. The core idea, borrowed from the project's own two-tier EVAAA design (naturalistic training → controlled test-beds; [Lee et al. 2025](../references/InteroceptiveAI/sources/)), is to build **separate held-out test environments where the environment geometry forces the read-out** — a behavioural *assay*, the way a psychophysics rig forces a yes/no decision rather than hoping one shows up in free behaviour. In an assay:

- the agent must **commit** to an action that *only makes sense* under one class inference (forced choice), so a clean hit / false-alarm table exists with **no averaging**;
- a single cue (distance, lethality, the vision-count gate, smell separation) is **titrated** along one axis, so the read-out is a **slope / threshold**, robust to wherever the mean sits;
- the available class information is **computable in closed form**, so the agent's score is an **efficiency** against a known Bayesian ceiling — and a null cleanly separates "no information existed" from "information ignored";
- **catch trials** (predator-absent / rabbit-absent probes) pull **sensitivity apart from response bias**, so a jumpy "avoid-everything" agent does not score as discriminating.

This is a **concept memo** in the perceptual-decision / Bayesian-brain domain. It touches no `src/`, `configs/`, or `scripts/`; every config or code step is flagged as a hand-off to `experiment-designer` / `senior-developer`. It assumes the gridworld's existing knobs: entities configurable on class / behaviour / detection-range / damage / smell (a 5-vector) / spawn area; bushes that conceal the agent (`hides_agent: true`); food; a 27-dim observation whose **vision channel reports a per-class COUNT of animals on the agent's own cell** (predator → channel 5, rabbit → channel 7), which is the only distal "elimination" cue in the current design.

---

## §1 The reframe — from *post-hoc statistic on free behaviour* to *designed measurement*

A naturalistic rollout is an **observational study**: the agent chooses where to go, so predator-encounters and rabbit-encounters happen at uncontrolled distances, energies, and gating states, and the experimenter must *reconstruct* a contrast from whatever happened. Three pathologies follow, all visible in the June work:

1. **Encounter confounding.** The agent's own policy decides the encounter geometry, so predator and rabbit are *not* met under matched conditions — exactly the "predator encounters sampled at different distances than rabbit ones" caveat the detectability memo flagged for its AUC.
2. **Conditional cancellation.** The real signal lived in a sub-state (rabbits "accounted-for" in the vision count), and marginalizing over the gate averaged it to zero — the "76% vs 76%" failure.
3. **Unreadable null.** A flat result on free behaviour cannot distinguish *can't*, *won't*, and *no-information-existed*.

An **assay** removes all three by *taking the choice of encounter geometry away from the agent and giving it to the experimenter*. The environment is built so that (a) the agent is funneled into a matched encounter, (b) it must emit a discrete committing action, and (c) the information content is fixed and known. The statistic is then **designed-in**, not reconstructed. Every measure in the sibling memos becomes a *direct read-out of a forced response*, not a salvage operation on free behaviour.

> **Design contract shared by all five assays.** (i) The agent is **transferred** to the test env from its naturalistic checkpoint (held-out; never trained on the assay — or the assay measures memorization, not inference). (ii) Each assay yields a **per-trial discrete or scalar response** with a class label, so a 2×2 table or a psychometric vector falls out with no episode-averaging. (iii) Every assay ships its **catch / matched-noise trials** in the same block, so sensitivity and bias are estimated jointly. (iv) Where the information is computable, the assay reports **efficiency** $\eta=(d'_{\text{agent}}/d'_{\text{ideal}})^2$ against the closed-form ceiling, not just raw $d'$.

---

## §2 The five assays

Notation reused from the detectability memo: $C\in\{\text{pred},\text{rab}\}$ is class; $H=P(\text{defend}\mid\text{pred})$ the hit rate; $F=P(\text{defend}\mid\text{rab})$ the false-alarm rate; $d'=z(H)-z(F)$ sensitivity; $c=-\tfrac12(z(H)+z(F))$ criterion; $z=\Phi^{-1}$ the inverse-normal.

---

### Assay A1 — Two-corridor forced choice (the clean d′ / criterion rig)

**Plain-English question.** *Put a predator at the end of one corridor and a behaviourally identical rabbit at the end of the other. Which one does the agent walk toward, and which does it refuse?* The direction it commits to **is** its class inference, read off a single action — no averaging.

**Task structure.** A symmetric Y- or T-layout (or two mirrored vertical corridors flanking the start cell at `start_pos: [5,5]`). One corridor terminates in a single **predator** (`class: predator`, damage band as in cell 08), the other in a single **rabbit** (`class: neutral`, harmless) — byte-identical on smell `[0,1,0,0,0]`, detection range, chase, stamina, disengage-on-contact, exactly as the cell-08 manifest already guarantees. **Food is placed beyond, or symmetrically between, both animals**, so the agent has a *reason* to traverse a corridor (forage pressure) and the only question is *which* it dares. Critically, the **class-revealing cue must be present at choice time**: with vision-range 0 the agent can only resolve class on-contact, so A1 in its base form tests whether the agent commits *toward the harmless one* once it has been down each corridor once (a within-episode learning read) **or** A1 is paired with a distal cue (Assay A4's smell-separation, or a non-zero vision range as a `senior-developer` knob) so the choice is genuinely pre-contact. Left/right predator assignment is **randomized per trial** to kill any spatial-side bias.

**Detection-theoretic read-out.** Each trial yields a binary commit: *entered-predator-corridor* vs *entered-rabbit-corridor* (or *avoided* vs *approached* per corridor in the catch variant). Define $H=P(\text{approach}\mid\text{rab corridor})$, $F=P(\text{approach}\mid\text{pred corridor})$ — or equivalently avoid-rates — and read **behavioural $d'$ and criterion $c$ directly from the 2×2 table**. No gating, no marginalization: every trial is a matched, forced commit.

**What failure-of-the-mean it fixes.** Kills *encounter confounding* (both animals met at identical distance, the corridor length) and *conditional cancellation* (there is no free wandering to average over — one trial, one commit). It is the assay incarnation of Measure **D1**.

**Sample-size / estimation caveat.** $d'$ explodes when $H$ or $F$ hit 0/1 — apply the **log-linear (Hautus) +0.5 correction** and bootstrap CIs over trials. Need $\gtrsim 30$ trials per class-side for a stable $d'$; randomize predator side and start orientation so the commit reflects class, not geometry. **Side-bias is the dominant confound** — always report the left/right-balanced $d'$.

---

### Assay A2 — Approach-titration psychometric (the slope/threshold rig)

**Plain-English question.** *Vary one cue — distance, lethality, or smell-separation — by fine steps, and ask at each level: how often does the agent treat the animal as dangerous?* Fitting that curve gives a **threshold and a slope**, which are the discrimination measure — and they don't care where the agent's overall caution sits.

**Task structure.** A single animal approaches the foraging agent along a controlled track, but one parameter is **titrated across trial blocks**. Three natural axes, all configurable today:
- **Lethality titration.** Hold everything identical; sweep the predator's damage band from harmless `[0,0]` → mild `[5,15]` → … → lethal `[5,120]` (the cell-08 band). Read how the agent's pre-contact avoidance scales with *true* danger it cannot see — a pure test of whether anything in its behaviour tracks lethality.
- **Distance / time-to-contact titration.** Vary the spawn distance (`spawn_area`) or approach speed (`move_interval`) so the pre-contact window length is the swept variable — the **psychophysical** axis: how much evidence-time does the agent need before it commits to defend?
- **Vision-count gate titration.** Add $k\in\{0,1,2,\dots\}$ rabbits that are forced onto the agent's cell (so the vision count channel-7 reads $k$), with one extra approaching smell that is *either* the $(k{+}1)$-th rabbit *or* the predator. Sweep $k$. This titrates the **elimination cue** the June trajectory read identified — the agent should discriminate better as more rabbits are "accounted-for."

**Detection-theoretic read-out.** For each cue level $x$, estimate the defend-probability $\psi(x)=P(\text{defend}\mid x)$ and **fit a psychometric function** (logistic / Weibull): $\psi(x)=\gamma+(1-\gamma-\lambda)\,F(x;\alpha,\beta)$, where $\alpha$ is the **threshold** (the cue level at 50% defend), $\beta$ the **slope** (discrimination acuity), $\gamma$ the guess/floor rate, $\lambda$ the lapse/ceiling rate. **The slope $\beta$ is the discrimination measure**; the threshold $\alpha$ is where the agent's criterion crosses.

**What failure-of-the-mean it fixes.** A slope is invariant to *where the mean sits* — a globally cautious agent and a globally reckless one can have the *same* slope (same acuity, different criterion), so this disentangles acuity from baseline caution that a single mean conflates. It also turns "is there a difference" into "**how graded** is the agent's sensitivity," which is far more informative than a binary verdict.

**Sample-size / estimation caveat.** Psychometric fits are hungry: $\gtrsim 6$–$8$ cue levels × $\gtrsim 40$ trials each for a stable slope (use bootstrap or a Bayesian fit, e.g. a Gamma-prior logistic). **Explicitly fit $\gamma$ and $\lambda$** — a non-zero lapse rate otherwise masquerades as a shallow slope (the classic psychophysics trap). The lethality-titration axis has a subtle confound: if the agent *learned* damage statistics during naturalistic training, the curve reflects learned association, not online inference — state which you are claiming.

---

### Assay A3 — Ideal-observer-matched gauntlet (the efficiency rig that makes a null readable)

**Plain-English question.** *Build the test so that we know exactly how much class information is physically in the agent's sensory stream — then report not "did it discriminate" but "what fraction of the available information did its behaviour express."* A null then means one of two precise things, not an ambiguous shrug.

**Task structure.** Constrain the encounter so the class-evidence is a **small, enumerable quantity** whose generative model is known. The cleanest instantiation uses the existing **vision-count elimination cue**: fix a world with exactly $n$ harmless rabbits + 1 predator, all sharing smell `[0,1,0,0,0]`. At a given step the agent observes (a) its channel-7 count $k$ = rabbits currently on its cell, and (b) one *other* smell approaching from distance. The class of that approaching smell is decidable by the closed-form posterior
$$
P(C{=}\text{pred}\mid k,\,\text{1 approaching smell}) \;=\; \frac{P(k\mid \text{pred approaching})\,\pi_{\text{pred}}}{\sum_{c}P(k\mid c)\,\pi_c},
$$
which, because the approaching entity is predator **iff** all $n$ rabbits are accounted-for, collapses to a step function in $k$ — and that gives a **per-gate ideal discriminability** $d'_{\text{ideal}}(k)$ in closed form. The gauntlet forces the agent through encounters at each $k$ (controlled by how many rabbits are spawned onto its cell), with a committing defend/forage choice.

**Detection-theoretic read-out.** Report **SDT efficiency** per gate:
$$
\eta(k)=\left(\frac{d'_{\text{agent}}(k)}{d'_{\text{ideal}}(k)}\right)^{2}.
$$
The verdict structure (straight from Measure **B5**):
- $d'_{\text{ideal}}(k)\approx 0$ for all $k$ ⇒ **the information was never there** — "class-blind" is a *task* fact, an honest statement about the environment, not an agent failing.
- $d'_{\text{ideal}}(k)\gg 0$ but $\eta(k)\approx 0$ ⇒ **information available, agent ignored it** — the "knows-but-won't-act" / "can't-extract" claim, far stronger and more publishable.
- $\eta(k)$ rising toward 1 as $k\to n$ ⇒ the agent **is** doing elimination inference — the positive result the June trajectory read hinted at.

**What failure-of-the-mean it fixes.** It is the *only* design here that makes a **null result interpretable**: it benchmarks against the physical limit, not against chance. It converts the project's ambiguous "class-blind" into a precise, falsifiable claim.

**Sample-size / estimation caveat.** The ceiling is *computed*, not estimated — it costs derivation, not data (hand-off: I derive the closed form on request). The only sampling noise is in $d'_{\text{agent}}(k)$, so per-gate trial counts follow A1 ($\gtrsim 30$/cell). **The fair ceiling is the *bounded-memory* ideal observer** if the agent's recurrent memory is shorter than the evidence window — state which ceiling (omniscient vs memory-matched) you report, or the efficiency is not honest.

---

### Assay A4 — Catch-trial bias-control block (the "is it actually discriminating, or just jumpy?" rig)

**Plain-English question.** *Mix in trials where there is no predator at all (and trials with no rabbit at all). An agent that flees everything will "hit" on every predator trial but also false-alarm on every empty/rabbit trial — so its high avoidance score collapses to zero sensitivity once the false-alarm rate is in the table.*

**Task structure.** Not a standalone geometry but a **trial-composition wrapper** layered onto A1 or A2. The block interleaves four trial types in randomized order:
- **Signal trials** — predator present (target).
- **Matched-noise trials** — behaviourally identical rabbit present (the noise distribution; this is the false-alarm denominator the naive "predator-avoidance-only" seed lacks).
- **Predator-absent catch** — empty approach (smell present, no animal, or an inert prop) — measures pure response bias / startle.
- **Rabbit-absent catch** — confirms the agent *will* forage freely when genuinely safe (guards against a frozen "never-approach" policy that trivially scores high avoidance).

**Detection-theoretic read-out.** The full 2×2 (or 2×$k$) confusion table → **$d'$ AND $c$ jointly**. The key product is the **criterion $c$ and the false-alarm rate $F$**: a jumpy avoid-everything agent shows $H\approx F\approx 1$ ⇒ $d'\approx 0$ with a **lax criterion** ($c\ll 0$), which is exactly the diagnosis "high avoidance, zero discrimination." Sensitivity is reported *net of bias*; bias is reported as its own number.

**What failure-of-the-mean it fixes.** Directly fixes the **bias-as-discrimination** error the detectability memo's §3 critique named: "predator-avoidance alone" is a hit rate with no false-alarm rate, so a pure-bias agent scores maximally. The catch block *supplies the false-alarm rate by construction*, making any avoidance number interpretable.

**Sample-size / estimation caveat.** Catch trials must be **frequent enough to estimate $F$ precisely** (rare catch trials give a noisy false-alarm rate and an unstable $c$) — budget $\gtrsim 25\%$ of the block as catch/noise trials. Randomize trial type so the agent cannot condition on block structure. If the agent's policy is near-deterministic (argmax eval), $H,F$ are near 0/1 and you again need the Hautus correction; a **soft/stochastic eval** (logit-logging) gives graded rates and tighter $c$ — hand-off to `senior-developer` (current evals log argmax only).

---

### Assay A5 — Slow-approach anticipation window (the "measure the pre-contact behaviour before any contact" rig)

**Plain-English question.** *Make the run-up to contact long, slow, and fully observable, so that "what the agent does in the seconds before the animal reaches it" is a rich, measurable trace — and anticipation, if it exists, has room to show.* The June nulls partly came from contact happening at step ~11, leaving almost no pre-contact window to read.

**Task structure.** A **titrated safe-vs-threat slow-approach** layout: the animal spawns far (large `spawn_area` offset) and approaches slowly (`move_interval` ≥ 2, or a reduced chase aggression) along an open, bush-flanked lane so the agent has a **long, observable pre-contact window** with cover available. Two matched conditions are interleaved: a **threat** run (predator approaching) and a **safe** run (rabbit approaching), identical in kinematics. Optionally combine with A4 catch trials. The point is to *stretch the anticipation window* so graded readouts (closest-approach-allowed, latency-to-first-cover, eat-suppression-onset-time) have dynamic range.

**Detection-theoretic read-out.** A **graded ROC / AUC on the anticipatory readouts** (Measure **D2**), now well-sampled because the window is long: $\mathrm{AUC}=P(B_{\text{pred}}>B_{\text{rab}})$ on $B=$ {min-distance-allowed, latency-to-cover, eat-suppression magnitude}. The slow window also enables a **time-resolved $d'(t)$** — when *before* contact does the agent's behaviour first separate the two classes? That onset-time is itself an anticipation measure: pre-contact onset = genuine anticipation; onset only at $t=\text{contact}$ = reactive.

**What failure-of-the-mean it fixes.** Fixes the **window-truncation** failure — a fast approach gives almost no pre-contact samples, so any anticipatory signal is undersampled to noise. A long observable window makes the *anticipatory* part of the trajectory (the part the whole study cares about) the *measured* part, and time-resolved $d'(t)$ turns "anticipatory vs reactive" from a verbal claim into a measured onset latency.

**Sample-size / estimation caveat.** AUC standard error follows Hanley–McNeil; $\gtrsim 20$–$30$ trials/class for a usable CI. The **slow approach changes the task distribution** — a slowly-approaching predator is out-of-distribution relative to fast naturalistic training, so a null here may reflect distribution shift, not inability; pair with an in-distribution-speed control or note the OOD caveat (the same caveat the predator-only control carried).

---

## §3 How the five compose into a test battery

The assays are not alternatives — they answer different questions and are **strongest run as one battery on the same held-out checkpoint**:

| Order | Assay | Answers | If it comes back null… |
|---|---|---|---|
| 1 | **A3 ideal-observer gauntlet** | Was the information even there? | If $d'_{\text{ideal}}\approx0$, stop — "no distal cue exists" is the honest verdict; the rest is moot. |
| 2 | **A4 catch-block** (wrapping A1) | Is apparent avoidance real sensitivity or just bias? | A lax-criterion, $d'\approx0$ result = "jumpy, not discriminating." |
| 3 | **A1 two-corridor** | Forced commit: does the choice reveal class inference? | Clean $d'$/$c$ with no averaging. |
| 4 | **A2 titration** | How *graded* is sensitivity along the cue? | Slope $\beta$; flat slope with high ceiling = pure bias (cross-checks A4). |
| 5 | **A5 slow-window** | *When* before contact does behaviour separate? | Time-resolved $d'(t)$ onset = anticipatory vs reactive. |

The decisive pair is **A3 + A4**: A3 says whether information existed, A4 says whether an avoidance number is sensitivity or bias. Together they convert the project's ambiguous "class-blind" into exactly one of {*no-information*, *information-ignored*, *bias-not-discrimination*, *genuine-discrimination*}.

---

## §4 Project anchors

- **The "class-blind" verdict is reopened, not overturned.** The predator-vs-rabbit summary's §3.1 verdict table reads "class-blind" from naturalistic rollouts; this memo gives the *forced* tests that would tell whether that null is *can't*, *won't*, or *no-information* — the same gap the sibling detectability memo named, now operationalized as buildable environments.
- **Phase plan.** These assays are the **Phase 4 (hypervigilance read-out)** instrument — the standard way any "does the agent recognize the threat" claim should be scored from here on, for any two-class contrast.
- **G1/G2 gates.** This adds **no new mechanism hypothesis**; it changes *how the discrimination verdict is measured*. A gate that rests on danger-recognition should be scored by an assay's forced $d'$/efficiency, not by a mean-difference on free behaviour. If a gate currently passes/fails on naturalistic means, re-scoring it through A1+A3 could flip the verdict — flag those gates.
- **H1–H5.** Hypothesis-agnostic (a measurement upgrade), but it **sharpens** any H asserting anticipatory danger-recognition: such an H should be pre-registered as "**per-gate behavioural $d'_z>0$ with efficiency $\eta$ bounded away from 0 in Assay A3, and a non-flat psychometric slope $\beta$ in A2**," not as a mean shift. *Proposed (this memo):* a candidate **H6** — *"discrimination is gated, not absent": efficiency $\eta(k)$ rises with the number of accounted-for rabbits* — is directly testable by A3 and not in the current H1–H5.
- **Two-tier EVAAA lineage.** This is the project's own naturalistic-train → controlled-test-bed structure ([Lee et al. 2025](../references/InteroceptiveAI/sources/)) applied to the discrimination question: train in the rich foraging world, *measure* in purpose-built assays.

---

## §5 Ranked shortlist

| Rank | Assay | Task structure (one line) | Detection-theoretic read-out | What it fixes |
|---|---|---|---|---|
| 1 | **A3 — Ideal-observer gauntlet** | Force encounters at each known vision-count gate $k$ where the optimal class posterior is closed-form | **Efficiency** $\eta(k)=(d'_{\text{agent}}/d'_{\text{ideal}})^2$ per gate | Makes a NULL readable — separates *no-information-existed* from *information-ignored*; the verdict-critical disambiguation |
| 2 | **A4 — Catch-trial block** | Interleave predator-absent / rabbit-absent / matched-rabbit probes into A1 or A2 | **$d'$ AND criterion $c$** jointly; false-alarm rate $F$ supplied by construction | Stops a jumpy "avoid-everything" agent scoring as discriminating — splits sensitivity from response bias |
| 3 | **A1 — Two-corridor forced choice** | Symmetric corridors, predator down one, matched rabbit down the other; agent must commit | **Behavioural $d'$ + criterion** off a clean 2×2, **no averaging** | Kills encounter-confounding and conditional-cancellation — one trial, one forced commit |
| 4 | **A2 — Approach titration** | Sweep one cue (lethality / distance / vision-count $k$) and fit the defend-probability curve | **Psychometric slope $\beta$ + threshold $\alpha$** | Slope is invariant to where the mean sits — separates discrimination *acuity* from baseline caution |
| 5 | **A5 — Slow-approach window** | Long, observable pre-contact run-up; matched threat vs safe approaches | **AUC** + **time-resolved $d'(t)$ onset** | Fixes window-truncation — gives anticipation room to show and dates it (anticipatory vs reactive) |

**Unifying principle.** *Don't compute discrimination statistics on free behaviour and hope the signal survives averaging — build the environment so the agent is **forced** to emit the statistic: a committing choice that reveals its class inference, a titrated cue that yields a slope, and a known information ceiling that makes a null mean something.*

---

## Next steps

- `experiment-designer` — instantiate A1 (two-corridor) and A4 (catch-block) first as `configs/experiment/hypervigilance/` test-bed configs reusing the cell-08 matched predator/rabbit entity spec; define the per-trial response logging and randomized predator-side / trial-type schedule. A3 needs the per-gate forced-encounter schedule; A2 needs the titration sweep grid.
- `senior-developer` — two enabling code changes the assays want: (1) a **soft / stochastic eval with logit logging** (current evals log argmax only) so $H,F$ are graded and criterion $c$ is stable; (2) optionally a non-zero vision-range or a distal class-cue knob so A1's choice is genuinely pre-contact rather than post-first-traversal. Scope minimal.
- `professor-bayesian-brain` (me) — derive the closed-form $P(C\mid k,\text{smell})$ and per-gate $d'_{\text{ideal}}(k)$ for A3 (omniscient and bounded-memory ceilings), on request.
- `professor-rl` — if A3 shows information present but unused ($\eta\approx0$ with $d'_{\text{ideal}}\gg0$), the follow-up is policy-side: does the return signal reward acting on the distinction at all? Route there.
- `professor-pain-modeling` — construct-validity check that a forced-choice "approach the harmless one" genuinely indexes danger-recognition rather than a foraging-optimality artifact.
