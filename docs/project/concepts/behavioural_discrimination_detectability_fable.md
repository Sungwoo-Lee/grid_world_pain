# Is there discriminative behaviour? A detection-theoretic answer (signal detection · information · ideal observer)

> **One-line summary.** Stop asking "did the agent's *average* behaviour differ between the harmful predator and the harmless rabbit?" and start asking "*is class information detectable in the agent's behaviour at all, and is the agent extracting as much of it as physically possible?*" This memo builds that out as four reusable measures — a **conditional behavioural d′** that splits "can it tell them apart" from "does it act on the difference", a **bias-free ROC/AUC** on graded anticipatory readouts, a **conditional mutual information** that is the most general "is there ANY signal" test, and an **ideal-observer efficiency** that says whether a null result means *no information* or *information left unused*. It closes by dissecting the naive seed idea ("just measure predator-avoidance alone").

## Purpose — what this memo is for, in plain English

The project's predator-vs-rabbit study made a harmful animal (a **predator**, whose touch injures the agent) and a harmless one (a **rabbit**, whose touch does nothing) identical on everything sensable from a distance — same smell, same chase, same speed — leaving danger as the only difference, then asked whether the trained agent behaves differently toward them *before* contact. The machinery that answered was a set of **averages**: episode-mean distance to each animal (gap ≈ 0), a distance-matched flee rate that read **76% for the predator and 76% for the rabbit** (i.e. identical), an interrupted-feeding rate of exactly **0.000**. Every number said "no difference." Then a step-by-step read of individual trajectories found a *real* difference the averages had buried: the discrimination was **conditional** — it switched on only in certain states (when the agent's vision count showed the rabbits accounted-for, so the only other approaching smell *had* to be the predator) — and mixing the "switched-on" and "switched-off" states together averaged the effect to zero. (Full post-mortem: the 2026-06-09 memory insight on aggregate stats hiding conditional behaviour.)

This memo fixes the *measurement* at its root, from the signal-detection / Bayesian / information-theoretic angle. The thesis is that "does behaviour differ?" is the wrong question because a *difference of means* is the wrong quantity. The right quantity is **detectability**: is there any extractable information about animal-class in the agent's behaviour, by any route, in any state — and how does the amount the agent expresses compare to the amount that was *available to extract*? Detection theory, mutual information, behavioural decoding, and ideal-observer analysis are exactly the tools built for that question. Below, each measure is named, with the plain-English question it answers, its formula, the failure-of-the-mean it repairs, an estimation recipe with small-sample caveats, and its pros/cons. The measures are deliberately **general** — they apply to any two-context behavioural-discrimination study, not just this gridworld.

This is a **concept memo** in the perceptual-decision / Bayesian-brain domain. It touches no `src/`, `configs/`, or `scripts/`; every code-needing step is flagged as a hand-off. It is a sibling to two existing concept memos ([behavioural_discrimination_detectability.md](behavioural_discrimination_detectability.md), [behavioral_discrimination_measures.md](behavioral_discrimination_measures.md)); where those emphasize the conditional/matched/distributional toolkit, this one leads with the **sensitivity-vs-bias split** and the **ideal-observer ceiling** as the two things the project's current "class-blind" verdict has not yet separated.

---

## §1 The reframe — three nested questions, and the axis the mean destroys

Let $C\in\{\text{pred},\text{rab}\}$ be an animal's latent class in an encounter, and let $B$ be some behavioural readout the agent emits (a scalar like "closest approach allowed", a vector like "action histogram over the approach window", or a whole trajectory). Three questions stack in strict order of generality:

$$
\underbrace{\mathbb{E}[B\mid \text{pred}]\neq\mathbb{E}[B\mid\text{rab}]}_{\text{(1) mean shift — what we did}}
\;\Rightarrow\;
\underbrace{p(B\mid\text{pred})\neq p(B\mid\text{rab})}_{\text{(2) distributions differ — SDT}}
\;\Rightarrow\;
\underbrace{I(C;X)>0}_{\text{(3) information — most general}}
$$

The implications run **one way only**. Equal means with different spread, effects that flip sign across states and cancel, or discrimination expressed in a variable whose mean nobody computed — all of these break (1) while leaving (2) and (3) true. The project lives precisely in that gap: $\mathbb{E}[B\mid\text{pred}]\approx\mathbb{E}[B\mid\text{rab}]$ yet the trajectory read suggests $p(B\mid\text{pred})\neq p(B\mid\text{rab})$ in some conditioning state. So we must climb from means to distributions to information.

There is a **second, orthogonal axis** the mean collapses entirely: **sensitivity vs. bias.** An agent can (i) be *able* to tell predator from rabbit yet apply the same defensive action to both because its policy doesn't act on the distinction, or (ii) be *unable* to tell them apart at all. These are different scientific claims — "class-blind (no information)" vs. "class-aware but acts the same (information present, policy ignores it)" — and a difference-of-means test cannot tell them apart, because it folds discriminability and decision criterion into one number. Signal detection theory exists to pull these two apart. **The project's current "class-blind" verdict has not made this split, and it is the highest-value thing this memo adds.**

---

## §2 The measures

Notation: $N$ = encounters; $N_{\text{pred}},N_{\text{rab}}$ = per-class counts; $B$ = a scalar readout; $X$ = a (possibly high-dim) behavioural/state summary; $Z$ = a discrete *gating state* we condition on (e.g. "rabbits accounted-for in the vision count"). $\Phi^{-1}=z(\cdot)$ is the inverse standard-normal CDF (the "z-score").

### Measure D1 — Conditional behavioural d′ and criterion (the sensitivity/bias splitter)

**Plain-English question.** *In matched situations, how reliably can the agent tell predator from rabbit (sensitivity), and separately, how trigger-happy is its defensive response (bias)?*

**Setup.** Reduce each encounter to a binary response $R\in\{\text{defend},\,\neg\text{defend}\}$ (e.g. "did the agent flee/dive before contact?"). Form the $2\times2$ table and define, **within each gating state $Z=z$**:

$$
H_z = P(R{=}\text{defend}\mid \text{pred}, Z{=}z),\qquad
F_z = P(R{=}\text{defend}\mid \text{rab}, Z{=}z),
$$
$$
\boxed{\,d'_z = z(H_z) - z(F_z)\,}\qquad
\boxed{\,c_z = -\tfrac{1}{2}\big(z(H_z)+z(F_z)\big)\,}
$$

$d'_z$ is **discriminability** (signal-to-noise of the class evidence the behaviour reflects); $c_z$ is **response bias** (how much evidence the agent demands before defending). A flat marginal flee rate of "76% vs 76%" is exactly $H\approx F\Rightarrow d'\approx 0$ — *but only marginalized over $Z$*. The fix is to compute $d'_z$ within each gate and report the profile.

**What failure of the mean it overcomes.** The mean flee rate gave one number that was simultaneously blind to (a) the state-gating ($H_z\!\neq\!F_z$ inside a gate, equal across the marginal) and (b) the sensitivity/bias confound. $d'_z$ recovers both.

**Estimation + sample-size.** Each cell needs enough encounters that $H_z,F_z$ aren't 0 or 1 (extreme rates send $z(\cdot)\to\pm\infty$). Use a **log-linear / Hautus correction** (add 0.5 to each cell) for small $N$; report bootstrap CIs over encounters. Rule of thumb: $\gtrsim 30$ encounters per (class × gate) cell for a stable $d'_z$. Hand-off to `experiment-designer` to ensure eval rollouts emit enough matched encounters per gate.

**Pros.** The single most diagnostic measure for the project's open ambiguity — it is the only one here that *separates* "can't discriminate" from "won't act differentially." **Cons.** Needs a defensible binary $R$ and a defensible gate $Z$; the equal-variance Gaussian assumption behind $d'$ can fail (then use the AUC of D2 as the assumption-free sensitivity index).

### Measure D2 — ROC / AUC on a graded anticipatory readout (bias-free, distribution-level)

**Plain-English question.** *If I pick a random predator encounter and a random rabbit encounter, how often does the agent behave more defensively toward the predator?*

**Setup.** Pick a graded readout $B$ — ranked by how *anticipatory* it is (the project cares most about pre-contact signals): $B_1=$ minimum distance allowed before contact; $B_2=$ latency from "animal enters radius" to first flee/cover; $B_3=$ eat-suppression magnitude while the animal is near. Sweep a threshold $\theta$, plot $H(\theta)$ vs $F(\theta)$ — the ROC curve — and take its area:

$$
\mathrm{AUC} = \int_0^1 H\big(F^{-1}(u)\big)\,du = P\big(B_{\text{pred}} > B_{\text{rab}}\big),
$$

the probability that a random predator encounter outranks a random rabbit encounter (the Mann–Whitney $U$ statistic, normalized). $\mathrm{AUC}=0.5$ ⇔ indistinguishable; deviation either way ⇔ separable distributions.

**What failure of the mean it overcomes.** AUC is **invariant to any monotone reparameterization of $B$ and free of response bias by construction** — it reads the *whole distribution overlap*, so it catches equal-mean/different-shape separations the mean cannot.

**Estimation + sample-size.** Nonparametric; AUC's standard error is well-approximated by Hanley–McNeil. $\gtrsim 20$–$30$ encounters per class gives a usable CI. Compute a **per-gate** $\mathrm{AUC}_z$ as well, for the same conditional reason as D1. Confound: if predator encounters are sampled at systematically different distances/energies than rabbit ones, AUC inherits that — match the encounter sets (hand-off shared with the existing matched-state memo).

**Pros.** The cleanest *single* assumption-free number for "are the two behaviour distributions separable"; pairs naturally with D1 (AUC is the nonparametric sensitivity index). **Cons.** A scalar — it does not say *which* feature carries the signal (that's D3/D4); and a near-0.5 AUC on one readout does not rule out separability on another.

### Measure I3 — Conditional mutual information $I(C;B\mid Z)$ (the most general "is there ANY signal")

**Plain-English question.** *Once we control for the situation, does knowing the agent's behaviour tell us anything at all about which animal it faced?*

**Setup.**
$$
I(C;B\mid Z) = \sum_z p(z)\sum_{c,b} p(c,b\mid z)\,\log\frac{p(c,b\mid z)}{p(c\mid z)\,p(b\mid z)} \;\ge 0,
$$
positive **iff** the class-conditional behaviour distributions differ *for some function of $B$* in some state $z$ — agnostic to which moment, which variable, and how the dependence is shaped. The marginal $I(C;B)\approx 0$ alongside $I(C;B\mid Z)>0$ is the **exact information-theoretic statement of the project's failure mode**: a conditional dependence cancelled by marginalizing over the gate.

**What failure of the mean it overcomes.** All of them — MI is the most general dependence test; a zero mean-shift, a sign-flip cancellation, and a variance-only difference all still register as $I>0$.

**Estimation + sample-size (the part to be careful about).** MI estimators are **positively biased** at small $N$, so a raw $\hat I>0$ is *not* evidence of dependence. Three practical routes:
- **Discrete binning** (bin $B$): bias $\approx \tfrac{(|B|-1)(|C|-1)}{2N\ln 2}$ bits; apply the **Miller–Madow** correction and keep bins coarse.
- **k-NN (Kraskov–Stögbauer–Grassberger)** for continuous $B$: lower variance, no binning, but sensitive to $k$ and to ties.
- **MINE** (neural lower bound) for high-dim $X$ — only worth it when $X$ is a trajectory/observation window; overkill for a scalar.

In **all** cases, certify significance with a **label-permutation null**: shuffle $C$ within each $Z=z$ stratum, recompute $\hat I$ many times, and report $\hat I - \mathbb{E}[\hat I_{\text{shuffled}}]$ with the permutation $p$-value. This subtracts the finite-sample bias empirically and is the single most important safeguard. Hand-off to `experiment-designer` for the estimator + permutation harness.

**Pros.** Maximally general; the conditional form is a precise, citable formalization of "the discrimination was hidden in a conditional." **Cons.** Hardest to estimate well; gives a *yes/no* (and a noisy bits value), not a directly interpretable effect size; needs the most data.

### Measure C4 — Behavioural decodability (operational MI lower bound, cheapest to run)

**Plain-English question.** *Can a simple model read the agent's behaviour and guess which animal it faced, better than chance?*

**Setup.** Train a cross-validated classifier $g:X_{\text{window}}\mapsto\hat C$ on a window of agent behaviour (action sequence, distance trace, observation slice) around the encounter; report **balanced accuracy** or AUC against a **permutation null** (shuffle labels, re-train). Above-chance decoding *certifies* discrimination.

**Relation to MI (why this is principled, not just ML).** Any decoder beating chance is a constructive proof that $I(C;X)>0$, and via **Fano's inequality** the decoder's error $P_e$ lower-bounds the information:
$$
I(C;X)\;\ge\; H(C) - H_b(P_e) - P_e\log(|\mathcal{C}|-1),
$$
with $H_b$ the binary entropy. So decoding accuracy is an **operational lower bound** on Measure I3's MI — easy to estimate, hard to make *tight*. By the data-processing inequality $I(C;\hat C)\le I(C;X)$, the decoder can only *under*-report, never inflate, the true information (modulo leakage, below).

**What failure of the mean it overcomes.** A decoder finds *any* linear or nonlinear combination of behavioural features that separates the classes — it does not need the experimenter to have guessed the right scalar.

**Estimation + sample-size.** Strict train/test separation **by episode** (no window leakage across the split); permutation null for the chance baseline; keep $g$ simple (logistic / shallow) so it doesn't memorize. The dominant confound is **nuisance leakage** — the decoder may exploit distance/energy covariates correlated with class rather than class-driven behaviour; defend with the same matched-state discipline as D1/D2, or include the nuisances as controls and test for *incremental* decodability. $\gtrsim 50$ encounters per class for a believable cross-validated estimate.

**Pros.** Lowest-effort first move on the existing `.rec.gz` rollouts; a positive result immediately refutes "class-blind"; naturally handles high-dim windows. **Cons.** Only a *lower bound* on information; leakage-prone; an above-chance decoder shows information is *present*, not that the *policy acts on it* (that's D1).

### Measure B5 — Ideal-observer efficiency (the Bayesian ceiling that disambiguates the verdict)

**Plain-English question.** *Of the class information that was physically available in the agent's sensory stream, what fraction did its behaviour actually express?*

**Setup.** Construct the **Bayes-optimal observer** that has the *same information the agent had* — its observation history $o_{1:t}$ — and computes the exact posterior over class:
$$
P(C\mid o_{1:t}) \propto P(o_{1:t}\mid C)\,P(C),
$$
yielding an upper-bound discriminability $d'_{\text{ideal}}$. Compare against the agent's behavioural $d'_{\text{agent}}$ (Measure D1) via the **SDT efficiency**:
$$
\boxed{\;\eta = \left(\frac{d'_{\text{agent}}}{d'_{\text{ideal}}}\right)^{2}\;}
$$

**Why this is the verdict-critical measure.** A null result ($d'_{\text{agent}}\approx0$) has two utterly different causes, and only the ideal observer separates them:
- $d'_{\text{ideal}}\approx 0$ too ⇒ **the information was never there** — "class-blind" is a *sensory/environment* fact, not an agent failing. The honest claim becomes "no distal cue exists to discriminate", which is a statement about the task design, not the policy.
- $d'_{\text{ideal}}\gg 0$ but $\eta\approx 0$ ⇒ **the information was available and the agent failed to use it** — a genuine *policy/learning* limitation, and a much stronger, more publishable claim.

This gridworld makes $d'_{\text{ideal}}$ **tractable**, not merely conceptual: pre-contact, the only class evidence is the elimination signal (the integer vision-count of accounted-for rabbits) plus the shared smell, so the optimal posterior $P(C\mid \text{count}, \text{smell})$ can be written in closed form and turned into $d'_{\text{ideal}}$ per gate. The current "class-blind" verdict is incomplete until this ceiling is computed.

**Estimation + sample-size.** The ideal observer is *computed*, not estimated, from the generative model of the environment — so it costs derivation, not data (hand-off to me / `experiment-designer` for the closed form, `senior-developer` if it needs an eval-time probe). The only sampling noise is in $d'_{\text{agent}}$ (inherited from D1).

**Pros.** Turns an ambiguous negative into a precise claim; benchmarks the agent against the *physical limit* rather than against chance. **Cons.** Requires a faithful generative model of the available information; if the agent's effective memory is shorter than $o_{1:t}$, the fair ceiling is the *bounded-memory* ideal observer, not the omniscient one — state which ceiling you're using.

---

## §3 Critique of the naive seed idea

**The seed.** *"Just measure how much the agent avoided the predator only — not the mean."*

From a detection-theory standpoint this is insufficient on two independent counts, and the fixes are exactly the structure of §2.

1. **No contrast ⇒ no detectability, only response level.** Detection is intrinsically a *two-distribution* problem: a signal (predator) read against a noise baseline (rabbit). "Predator-avoidance alone" is a **hit rate with no false-alarm rate** — and with only $H$ you cannot compute $d'$, cannot separate sensitivity from bias, and cannot distinguish "avoids the predator *because* it recognizes danger" from "avoids *any* approaching animal." A high avoidance number could be **pure bias** (an agent that flees everything scores maximally and discriminates nothing). The matched rabbit is not optional garnish; it *is* the noise distribution that makes "avoidance" interpretable.

2. **A single statistic is still a mean ⇒ same blindness.** Collapsing the encounter to one avoidance scalar discards the distribution and the gating state — the very move that produced "76% vs 76%." Even with the contrast restored, a point estimate hides conditional and distribution-shape effects.

**The minimal principled fix** is the smallest object that has all three missing ingredients at once: **contrast + distribution + conditioning.** Concretely, the smallest sufficient upgrade is **Measure D2 (per-gate AUC)** — two classes (contrast), full rank-overlap (distribution), computed within the gating state (conditioning) — or **Measure D1** if a bias/sensitivity split is wanted. Either is a strict superset of the seed and costs no extra data. The seed's instinct ("don't average over the episode") is right; it just stops one step short of the structure that makes the number mean anything.

---

## §4 Project anchors

- **Methodology lesson it operationalizes.** This is the measurement-side answer to the 2026-06-09 aggregate-stats post-mortem: that note diagnosed *why* the means lied (conditional effect marginalized away); this memo supplies the *replacement* measures, with the conditional MI of I3 as the formal statement of that failure.
- **G1/G2 gates.** This memo does **not** add a new mechanism hypothesis; it changes **how the discrimination verdict is tested and interpreted**. Specifically, the "class-blind" reading in the predator-vs-rabbit summary (§3.1 verdict table) is, in detection-theoretic terms, *unfinished*: it has not separated sensitivity from bias (D1) nor checked the information ceiling (B5). Re-running the verdict through D1+B5 could flip "class-blind" into either "no distal information existed" or "information present, policy unused" — materially different conclusions for any gate that rests on danger-recognition.
- **Phase plan.** These measures are the readout layer for **Phase 4 (hypervigilance readout)** — they are how a "does the agent recognize the threat" claim should be scored going forward, for any two-class contrast.
- **H1–H5.** The framing is hypothesis-agnostic (a measurement upgrade), but it *sharpens* any H that asserts anticipatory danger-recognition: such a hypothesis should be stated as "behavioural $d'_z>0$ with efficiency $\eta$ bounded away from 0", not as a mean-difference.

---

## §5 Ranked shortlist

| Rank | Measure | One-line rationale |
|---|---|---|
| 1 | **B5 — Ideal-observer efficiency $\eta=(d'_{\text{agent}}/d'_{\text{ideal}})^2$** | The only measure that says whether a null means *no information existed* or *information went unused* — the verdict-critical disambiguation, and tractable here in closed form. |
| 2 | **D1 — Conditional behavioural $d'_z$ + criterion $c_z$** | Splits "can't tell them apart" from "won't act differentially" within each gating state — the split the current "class-blind" verdict skipped; directly dissolves "76% vs 76%". |
| 3 | **I3 — Conditional mutual information $I(C;B\mid Z)$** | The most general "is there ANY signal" test; its marginal-vs-conditional gap is the exact formal statement of the project's averaging failure. |
| 4 | **C4 — Behavioural decoder accuracy** | Cheapest first move on existing rollouts; a constructive lower bound on I3's MI via Fano; above-chance instantly refutes "class-blind". |
| 5 | **D2 — Per-gate ROC/AUC** | The clean, assumption-free single number for distribution separability and the minimal principled fix to the naive seed (contrast + distribution + conditioning). |

**Unifying principle.** *Discrimination is not a difference of means but the **detectability** of class information in behaviour — measured as the separation between two distributions (signal vs. noise), **conditioned on the gating state** rather than averaged over it, with sensitivity split from bias, and benchmarked against what a Bayesian ideal observer could have extracted from the same information.*

---

## Next steps

- `experiment-designer` — build the per-gate encounter-matched evaluation that emits enough matched encounters per (class × gate) cell for stable $d'_z$ / AUC$_z$, plus the label-permutation null harness for I3/C4.
- `senior-developer` — if a soft (logit-logging) eval or an eval-time ideal-observer probe is wanted, scope the minimal recording change (current evals log argmax actions only).
- `professor-bayesian-brain` (me) — derive the closed-form gridworld ideal observer $P(C\mid\text{vision-count},\text{smell})$ and $d'_{\text{ideal}}$ per gate for B5, on request.
- `professor-rl` — if D1 shows information present but unused ($d'_{\text{agent}}\ll d'_{\text{ideal}}$), the follow-up is a policy-side question (does the return signal even reward acting on the distinction?) — route there.
