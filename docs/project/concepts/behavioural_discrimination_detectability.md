# Behavioural discrimination as a detectability problem: signal-detection, information, and ideal-observer measures

> **One-line summary.** Replace "did the agent's *mean* behaviour differ between predator and rabbit?" with "*is there any detectable information* about which animal is which in the agent's behaviour?" — a family of distribution-level and information-level measures (behavioural d′ / ROC, mutual information, a trained class-decoder, and an ideal-observer reference) that catch discrimination the mean averages to zero.

## Purpose — what this memo is for and why it exists

The project keeps asking one question: when a harmful animal (a **predator**, whose touch injures the agent) and a harmless one (a **rabbit**, whose touch does nothing) are made identical on everything sensable at a distance, does the trained agent *behave differently* toward them? Our current answer-machinery is a set of **averaged statistics** — episode-mean distance to each animal, a distance-matched flee rate (it read 76% for the predator and 76% for the rabbit, i.e. identical), an interrupted-feeding rate that came out at exactly 0.000. Those numbers all said "no difference." But a step-by-step read of individual trajectories then found a *real* difference that the averages had hidden: the discrimination was **conditional** — it only fired in certain states — and mixing the "fires" and "doesn't-fire" states together averaged the effect to zero. (The full post-mortem is the memory insight on aggregate stats hiding conditional behaviour, dated 2026-06-09.)

This memo fixes the *measurement* problem at its root. A difference of means is the **wrong question**. The right question is about **information and detectability**: *is there any signal about animal-class expressed in the agent's behaviour, by any route, in any state?* That is the question signal-detection theory, information theory, behavioural decoding, and ideal-observer analysis were built to answer. Below I give a ranked, reusable toolbox — each measure named, with the plain-English question it answers, its formula, the failure-mode of the mean it overcomes, how to estimate it from our per-step rollouts, and its sample-size caveats. The measures are deliberately **general**: they apply to any "does behaviour differ between two contexts" study, not just this gridworld.

This is a **concept memo** in the Bayesian-brain / perceptual-decision-making domain. It does not change `src/`; where a measure needs code it is flagged as a hand-off.

---

## §1 The core reframe — means vs. distributions vs. information

Let $C \in \{\text{pred}, \text{rab}\}$ be the latent class of an animal in a given encounter, and let $B$ be **some behavioural readout** the agent emits in that encounter (a scalar like "minimum distance reached", a vector like "action histogram over the approach window", or a whole trajectory window). The three questions stack in strict generality:

1. **Difference of means (what we did).** Is $\mathbb{E}[B \mid C=\text{pred}] \neq \mathbb{E}[B \mid C=\text{rab}]$? This is true only if the *first moment* of the readout shifts. It is **blind** to (a) equal-mean-different-variance differences, (b) effects that flip sign across states and cancel, (c) any discrimination expressed in a variable whose mean you didn't happen to compute.

2. **Difference of distributions (signal detection).** Is $p(B \mid C=\text{pred}) \neq p(B \mid C=\text{rab})$ *as distributions*? This is what d′ / ROC test. It catches any separation in the readout you chose, mean-shift or not — but only in *that* readout.

3. **Information (the most general).** Is $I(C; X) > 0$ for $X$ = some observable summary of behaviour-and-state? Mutual information is positive **iff** the conditional distributions differ *for some function of $X$* — it is agnostic to *which* moment, *which* variable, and *how* the dependence is shaped. This is the cleanest possible "is there ANY discrimination, by any route?" test.

The hierarchy: **mean-shift $\Rightarrow$ distribution-shift $\Rightarrow$ positive MI**, but none of the reverse implications hold. Our project lives in exactly the gap the arrows leave open — a case with $\mathbb{E}[B\mid\text{pred}] \approx \mathbb{E}[B\mid\text{rab}]$ but (the trajectory read suggests) $p(B\mid\text{pred}) \neq p(B\mid\text{rab})$ in some conditioning state. So we must climb the ladder from means to distributions to information.

A second, orthogonal axis matters and the mean conflates it away entirely: **sensitivity vs. bias.** An agent can (i) be *able* to tell predator from rabbit (high sensitivity) yet apply the same defensive action to both because its policy doesn't act on the distinction (a *bias*/criterion choice), or (ii) be *unable* to tell them apart at all. Signal detection theory's whole reason to exist is to separate these two. A difference-of-means test cannot; it collapses sensitivity and bias into one number. This distinction is **directly load-bearing for the project's verdict**: "the agent is class-blind" (cannot discriminate, no information) is a very different scientific claim from "the agent discriminates but doesn't act differentially" (information present, policy doesn't use it). The current summary's "class-blind" verdict has not separated these.

---

## §2 The measures

Notation used throughout: $N$ = number of encounters/episodes; $N_{\text{pred}}, N_{\text{rab}}$ = per-class counts; $B$ = a chosen scalar behavioural readout; $X$ = a (possibly high-dimensional) behavioural/state summary; $Y$ = the agent's action; $\hat{C}$ = a decoder's predicted class.

### Measure 1 — Behavioural d′ (and the full ROC) on a chosen response variable

**Plain-English question.** *On some specific behavioural response, are the predator-encounter and rabbit-encounter distributions separable — and how much of any separation is genuine sensitivity vs. a response bias?*

**Setup.** Pick a scalar readout $B$ per encounter. Candidates, in rough order of how anticipatory they are (the project cares most about *pre-contact* signals):
- $B_1$ = **minimum distance reached before contact** (does the agent let the predator get as close as the rabbit?).
- $B_2$ = **flee latency** — steps from "animal within radius $r$" to the agent's first away-step.
- $B_3$ = **fraction of the approach window spent moving away** from the animal.
- $B_4$ = **bush-dive indicator** in the pre-contact window (the May event-level signal lives here).
- $B_5$ = **eat-suppression** — eat-rate while the animal is within $r$ (the M5 readout).

**Formula (parametric / equal-variance Gaussian form).**
$$
d' = \frac{\mu_{\text{pred}} - \mu_{\text{rab}}}{\sigma_{\text{pooled}}}, \qquad
\sigma_{\text{pooled}} = \sqrt{\tfrac{1}{2}\!\left(\sigma_{\text{pred}}^2 + \sigma_{\text{rab}}^2\right)}.
$$
Equivalently and more robustly, take a decision threshold $\theta$ on $B$, define hit rate $H = P(B > \theta \mid \text{pred})$ and false-alarm rate $F = P(B > \theta \mid \text{rab})$, and read sensitivity and bias off:
$$
d' = \Phi^{-1}(H) - \Phi^{-1}(F), \qquad
c = -\tfrac{1}{2}\big(\Phi^{-1}(H) + \Phi^{-1}(F)\big),
$$
with $\Phi^{-1}$ the inverse-normal (probit). **$d'$ is sensitivity (how separable the two distributions are); $c$ is bias (the agent's tendency to emit the "threat" response regardless of class).** Sweeping $\theta$ traces the **ROC curve**; its area, $\mathrm{AUC} = P(B_{\text{pred}} > B_{\text{rab}})$ over independent draws, is a **non-parametric, distribution-free** sensitivity index identical to the Mann–Whitney $U$ statistic. AUC = 0.5 means indistinguishable; AUC $\to 1$ (or $\to 0$) means perfectly separable.

**What failure of the mean it overcomes.** The probit/AUC form is sensitive to *any* stochastic dominance or distributional separation, including equal-mean / unequal-spread cases the mean misses. And $d'$/$c$ separate *can it discriminate* from *does it act on it* — the sensitivity-vs-bias confound the mean cannot touch. If the matched flee rates of 76% vs 76% were measured at a *single* operating point, that is **one point on the ROC**; the rest of the curve can differ even when that point coincides. Reading off one $(H,F)$ pair and calling it "no difference" is exactly the trap AUC avoids.

**Estimation + sample-size caveats.** AUC is the cleanest estimator (rank-based, no distributional assumption, exact small-sample null via the $U$ distribution). Get a confidence interval by bootstrap over encounters. The probit $d'$ needs a correction when $H$ or $F$ hits 0 or 1 — use the **log-linear / loglinear correction** (add $0.5$ to each cell, $1$ to each total) before applying $\Phi^{-1}$, or you get infinite $d'$. Rule of thumb: $\gtrsim 30$–$50$ encounters *per class* for a stable AUC CI; $d'$ via probit is noisier in the tails and wants more. **Crucially, compute these *conditionally* on the state that gates the effect** (e.g. "rabbits accounted-for in the vision count" vs. not) — pooling across the gating state is the original sin that produced the 76%/76% null.

**Pros.** Interpretable, decomposes sensitivity from bias, distribution-free in the AUC form, standard in the perceptual-decision literature so reviewers know it. **Cons.** Requires *choosing* a readout $B$ — if the discrimination lives in a variable you didn't pick, d′ on your chosen $B$ is blind to it. That blindness is exactly what Measures 2–3 remove.

### Measure 2 — Mutual information $I(C; X)$ between class and behaviour

**Plain-English question.** *Is there ANY information about animal-class in the agent's behaviour (or state) — by any route, in any moment, however shaped — and how many bits?*

**Formula.**
$$
I(C; X) = \sum_{c}\int p(c, x)\,\log\frac{p(c,x)}{p(c)\,p(x)}\,dx
        = H(C) - H(C \mid X).
$$
$I(C;X) = 0$ **iff** $X$ carries no class information; any positive value is a witness that the conditional behaviour distributions differ. With balanced classes $H(C) = 1$ bit, so $I$ is bounded in $[0, 1]$ bit and reads directly as "fraction of the class label recoverable from behaviour." $X$ can be a scalar, the action histogram over the approach window, a trajectory-window embedding, or even the agent's *internal state* (recurrent hidden state / 27-dim observation) — giving $I(C; \text{action})$, $I(C; \text{trajectory})$, $I(C; \text{hidden state})$ as a graded battery.

**What failure of the mean it overcomes.** All of them at once. MI is the **maximally general** discrimination test — it is positive for mean-shifts, variance-shifts, sign-flipping conditional effects, multimodal separations, and nonlinear dependencies alike. It is the formal object the difference-of-means is a crude lower-bound proxy for.

**Estimation + sample-size caveats.** This is where MI earns its caveats — **MI is systematically *over*-estimated at finite sample**, and the bias grows with the dimension/cardinality of $X$. Options, in increasing sophistication:
- **Discrete / binned $X$.** Plug-in entropy estimate, then apply the **Miller–Madow** bias correction or the **NSB** (Nemenman–Shafer–Bialek) Bayesian estimator. Keep the number of bins small relative to $N$ (bias $\approx \frac{(\,|\mathcal{X}|-1)(|\mathcal{C}|-1)}{2N\ln 2}$ bits — explicitly $N$-dependent).
- **Continuous low-dim $X$.** The **Kraskov–Stögbauer–Grassberger (KSG)** $k$-NN estimator — no binning, handles a few continuous dims well.
- **High-dim $X$ (trajectory windows, embeddings).** Neural estimators — **MINE** (Donsker–Varadhan lower bound) or **InfoNCE / CPC** (a *lower-bound* capped at $\log(\text{batch size})$, so it under-reads when MI is large but is bias-safe and the cap is benign here since $I \le 1$ bit). For our setting MINE/InfoNCE is overkill unless we go to raw trajectory windows; prefer binned+NSB or KSG on a hand-built summary first.
- **Always compute a null.** Permute the class labels and re-estimate; the permutation distribution of $\hat I$ gives both the finite-sample bias floor and a significance test. **Report $\hat I - \hat I_{\text{perm}}$, never raw $\hat I$** — the corrected, debiased estimate is the only honest number when $N$ is modest.

Sample size: binned MI with the permutation null is usable at $N \sim$ a few hundred encounters; neural estimators want thousands. **The permutation null is mandatory** here — without it a positive $\hat I$ is uninterpretable because the estimator is biased up.

**Pros.** Assumption-free about *how* discrimination is expressed; a single scalar; bounds the decodability of Measure 3 (see §3). **Cons.** Biased at finite sample (mitigated by permutation null + NSB), tells you *that* information exists but not *what behaviour* carries it (pair it with d′ to localize), and high-dim estimation is finicky.

### Measure 3 — Behavioural decodability: train a classifier $\hat{C} = f(X)$

**Plain-English question.** *Can a simple model predict which animal it was, just from watching what the agent did? If yes, the behaviour discriminated — full stop.*

**Setup.** For each encounter form a feature vector $X$ (action histogram over the approach window; or the agent's trajectory relative to the animal; or summary stats: min-distance, flee-latency, dive-flag, eat-suppression — i.e. the Measure-1 readouts *stacked*). Train a cross-validated classifier (logistic regression first — it keeps the result interpretable and the capacity low; a small random forest or gradient-boost as a nonlinear check). The held-out **balanced accuracy** (or AUC of the decoder) above the chance level is the discrimination signal.

**Formula / decision rule.** Chance balanced accuracy is $0.5$. Test against it with the exact binomial, or — far better at small $N$ — a **label-permutation null**: retrain on $\sim 10^3$ shuffles of the class labels, build the null accuracy distribution, and place the true accuracy as a percentile (this null absorbs cross-validation optimism and class-imbalance quirks automatically). **Decoding accuracy reliably above the permutation null = the behaviour carries class information** — the operational, finite-sample-honest cousin of $I(C;X)>0$.

**Relation to MI (why this and Measure 2 are two faces of one coin).** A decoder *is* an MI lower-bound machine. By **Fano's inequality**, decoder error $P_e$ caps the information: $H(C\mid X) \le H_b(P_e) + P_e\log(|\mathcal{C}|-1)$, so for two balanced classes
$$
I(C;X) \;\ge\; 1 - H_b(P_e), \qquad H_b(p) = -p\log_2 p - (1-p)\log_2(1-p).
$$
Thus **decoding accuracy puts a *floor* under the mutual information**: a decoder that hits $80\%$ guarantees $I(C;X) \ge 1 - H_b(0.2) \approx 0.28$ bits. Decoder accuracy is the *achievable* discrimination of a *specific* model; MI is the *best-possible* discrimination of *any* model. Run both: MI says "there are $X$ bits up for grabs," the decoder says "a simple policy-readable model recovers $Y$ of them." If MI $\gg$ decoder, the information is there but nonlinearly buried (the agent's recurrent policy might still use it); if MI $\approx$ decoder, a linear readout already exhausts it.

**What failure of the mean it overcomes.** Same as MI — catches any distributional difference, not just first-moment — but with a *concrete, interpretable* output (an accuracy you can put a CI on and a permutation null you can defend) and, with a linear decoder, a **weight vector that tells you which behavioural feature carried the class signal**. That feature-importance read is the bridge back to mechanism.

**Estimation + sample-size caveats.** Use stratified $k$-fold or nested CV; **never** report training accuracy. Keep the decoder *low-capacity* relative to $N$ — a deep net on 200 encounters will memorize and the permutation null will (correctly) show the "signal" is fake, but you waste effort; start with logistic regression. With $N_{\text{pred}}, N_{\text{rab}} \sim 30$–$100$ the permutation null is the right inference tool (the binomial CI is too optimistic). Balance the classes or use balanced accuracy / per-class recall, never raw accuracy on imbalanced data.

**Pros.** Operational, finite-sample-honest (permutation null), interpretable feature weights, lower-bounds MI for free. **Cons.** A *negative* decoder result is "*this* model in *this* feature space found nothing," not proof of zero information — pair a negative decode with a direct MI estimate before concluding "class-blind."

### Measure 4 — Ideal-observer / Bayesian reference: how much *could* be discriminated

**Plain-English question.** *Against what yardstick is the agent's discrimination "low" or "high"? What would an agent that optimally used every scrap of available class-information do — and how close is ours to that ceiling?*

**Why it's needed.** Measures 1–3 tell you whether and how much the agent's *behaviour* discriminates. They do not tell you whether a low number means **the agent failed** or **the task is genuinely near-impossible**. The ideal observer supplies the missing denominator: it is the Bayes-optimal use of the information actually present in the agent's input stream.

**Formalism.** The agent receives, at each step, an observation $o_t$ (the 27-dim vector) and carries history $h_t$. The optimal posterior over the approaching animal's class given everything sensed up to $t$ is
$$
p^\star(C \mid o_{1:t}) = \frac{p(o_{1:t}\mid C)\,p(C)}{\sum_{c'} p(o_{1:t}\mid c')\,p(c')}.
$$
The **ideal-observer discriminability** is the information this posterior carries about the true class, $I^\star_t = I(C; o_{1:t})$, or equivalently the Bayes-optimal classification accuracy from $o_{1:t}$. This is a **ceiling**: no policy can express more class-discrimination in its behaviour than is present in its inputs (data-processing inequality — $I(C; \text{behaviour}) \le I(C; o_{1:t})$, since behaviour is a function of the inputs). Estimate $I^\star_t$ the same way as Measure 2/3 but with $X = o_{1:t}$ (the *input stream*) rather than the behaviour — i.e. **train the best decoder you can from the raw observation/history to class**, and read its accuracy as the ceiling.

**The diagnostic contrast.** Put three numbers side by side:
$$
\underbrace{I(C; o_{1:t})}_{\text{ceiling — what's knowable}} \;\;\ge\;\; \underbrace{I(C; h_t^{\text{agent}})}_{\text{what the agent's state encodes}} \;\;\ge\;\; \underbrace{I(C; B)}_{\text{what its behaviour expresses}}.
$$
The two inequalities **localize the failure**, and this is the payoff of the whole memo:
- **Ceiling $\approx 0$** → the task is genuinely undiscriminable from the inputs; "class-blind" is the *correct and only possible* verdict, and no architecture change can help (this is the project's strong negative result, if it holds).
- **Ceiling $> 0$ but agent-state $\approx 0$** → the information is *present in the inputs* but the agent never *encodes* it (a representation-learning failure — a candidate for the "give the agent a learnable route" direction).
- **Agent-state $> 0$ but behaviour $\approx 0$** → the agent *knows* but does not *act* differentially. This is **sensitivity-present / bias-driven** in d′ terms — the agent can tell them apart but its policy applies the same response. A profoundly different (and more interesting) finding than "cannot discriminate," and one the current verdict has not ruled out.

**What failure of the mean it overcomes.** The mean has no notion of a ceiling at all — it cannot tell "didn't discriminate" from "couldn't." The ideal observer turns a bare null into a *calibrated* statement: discrimination relative to what was achievable.

**Estimation + caveats.** The ceiling estimate is itself a decoder (Measure 3 applied to inputs), so it inherits those caveats. Note the visual-count elimination route from the methodology post-mortem makes the input-side information *history-dependent* (the predator is identifiable only once both rabbits are "accounted-for") — so $I^\star_t$ should be computed *conditionally on the accounting state*, and will be near-zero when rabbits are unaccounted and positive when they are. That conditional ceiling is the principled version of the project's "in-distribution confirmation" follow-up.

**Pros.** Converts a null into a calibrated null; cleanly separates *task-impossible* from *agent-failed* from *agent-knows-but-won't-act*; ties directly to the predictive-coding view (the ideal observer *is* the optimal generative-model posterior). **Cons.** The most work; requires a good input-side decoder; "best decoder you can build" is a soft ceiling, not the true Bayes optimum unless the generative model is known (here it partly is — we control the env, so a near-exact ideal observer is buildable, which is a luxury most studies lack).

---

## §3 Critique of the seed idea — "just measure how much the agent avoided the predator only"

The user's instinct — drop the mean-of-the-difference, just look at predator-avoidance alone — moves in the right direction (away from a confounded contrast) but lands on a measure that is **weaker, not stronger**, for three detection-theoretic reasons:

1. **No contrast → no discrimination claim is even *defined*.** "The agent avoided the predator a lot" is a statement about *one* conditional distribution, $p(B \mid \text{pred})$. Discrimination is intrinsically a *relational* property — it is a statement about $p(B\mid\text{pred})$ **versus** $p(B\mid\text{rab})$. A high predator-avoidance number is fully consistent with the agent avoiding the *rabbit* exactly as much (the actual finding — robust avoidance applied equally). Without the rabbit baseline you cannot distinguish "recognises the predator" from "is just generally jumpy around any animal." The single-class number has **no discriminating power by construction**; it measures the agent's overall defensiveness, not its class-sensitivity. This is precisely the predator-only-world ambiguity: the agent avoided, but we couldn't call it recognition because there was no rabbit to contrast against.

2. **A point estimate discards the distribution → it cannot see the conditional structure that broke us last time.** A single avoidance statistic is one number; the discrimination we *missed* was **conditional** (present only when rabbits were accounted-for, absent otherwise), so it lives in the *shape* and *state-dependence* of the distribution, not its level. Any point estimate re-commits the original error of collapsing a state-dependent distribution to a scalar. The fix is not a different point estimate — it is to **keep the distribution** (ROC/AUC, MI, decoder) and to **condition on the gating state**.

3. **No sensitivity/bias split → it cannot answer the question the project actually cares about.** Avoidance magnitude alone conflates "can the agent tell?" with "how strongly does it react?". The project's real question — anticipatory *recognition* — is a **sensitivity** question ($d'$, MI, ceiling), and a single-class avoidance number measures something closer to **bias/response-magnitude**. You can max out avoidance with zero sensitivity (jumpy-at-everything) and you can have high sensitivity with low avoidance (knows but doesn't act). The seed measure cannot tell these apart.

**The minimal principled fix:** keep a **contrast** (predator *vs.* rabbit, the relational object), keep a **distribution/sensitivity** view rather than a point estimate (AUC or $d'$ at minimum, MI for the general case), and **condition on the gating state** (accounted-for vs. not) so the conditional effect isn't averaged away again. In one line: *replace "how much did it avoid the predator" (one class, one number, pooled) with "how separable are the predator and rabbit behaviour distributions, in the state where separation is possible" (two classes, a distribution, conditioned).* That is the smallest change that turns the seed idea into a valid discrimination measure.

---

## §4 Project mapping and anchors

- **The v8-style "null" reframed.** The project's pattern — aggregate means reading ≈ 0 while a conditional effect exists — is the canonical case where the mean fails and §1's information view is required. These measures are the operational instrument for *every* future "does behaviour differ between two contexts" question in the hypervigilance line, not just this study.
- **Sensitivity vs. bias is a new hypothesis the current verdict hasn't tested.** The summary's "class-blind" verdict assumes *low sensitivity*. Measure 4's three-way contrast ($I(C;\text{input}) \ge I(C;\text{agent-state}) \ge I(C;\text{behaviour})$) can instead reveal *high sensitivity, policy doesn't act* — a different and arguably more publishable finding. **Flagging this as an addition to the project's hypothesis set:** the existing controls cannot distinguish "can't discriminate" from "discriminates but won't act differentially"; these measures can.
- **The in-distribution confirmation, formalized.** The pending follow-up ("condition pre-contact avoidance on whether rabbits are accounted-for") is *exactly* a conditional AUC/MI computed on the accounting state — Measure 1 or 2 conditioned on the visual-count regime. This memo supplies the formal object that follow-up should report.
- **Predictive-coding reading (domain tie).** The ideal observer of Measure 4 *is* the optimal generative-model posterior $p^\star(C\mid o_{1:t})$ — the precision-weighted inference a predictive-coding agent would perform. "Behaviour expresses less class-info than the inputs carry" is, in that language, a *failure to propagate a precise prediction-error about class to action* — which connects this measurement memo to the precision/active-inference line should the project want to pursue *why* the policy doesn't act on available information.

---

## §5 Open questions (not fixed by the formalism)

- **Which behavioural readout $B$ for d′?** The formalism is agnostic; the project must choose pre-contact-weighted readouts (min-distance, flee-latency) to keep the *anticipatory* framing. A design decision, not a math one.
- **What is the unit of analysis** — per-encounter, per-episode, or per-step? d′/MI/decoder all need i.i.d.-ish units; per-step samples are autocorrelated and will inflate every estimate. Recommend per-encounter with block-bootstrap, but this needs an explicit definition of "an encounter."
- **How exact an ideal observer?** Because we control the env we *can* build a near-exact Bayes-optimal observer (rare luxury), but it requires writing the generative model out. Worth it for the ceiling, but it's real work.

---

## Next steps (hand-offs)

- **`experiment-designer`** — turn Measures 1–4 into a concrete re-analysis protocol over the existing recorded rollouts (`results/eval/matchedAggression_traj/...` and the cell-08 lethal-contact eval): define "an encounter," pick the readout battery $B_1$–$B_5$, specify the per-class $N$ and the permutation-null procedure, and pre-register the conditional (accounted-for vs. not) split. No new training needed for a first pass.
- **`senior-developer`** — the MI / decoder / ideal-observer estimators are new analysis code (not `src/` model code): a small `scripts/` module taking the `.rec.gz` rollouts and emitting AUC, debiased MI (binned+NSB with permutation null), cross-validated decoder accuracy, and the input-side ceiling. Flagging as a hand-off; this memo does not write it.
- **`professor-rl`** — if Measure 4 shows *agent-state carries class-info but behaviour doesn't* (knows-but-won't-act), the question of *why the policy gradient doesn't act on a discriminable state* is an RL-update question.
- **`professor-bayesian-nn`** — the MI estimators (MINE/InfoNCE) and the decoder-as-MI-lower-bound (Fano) overlap their calibration/uncertainty-disentanglement domain; pull them in if we go to neural MI estimation on raw trajectory windows.

**Cross-links:** [[active_inference_hypervigilance]] (the precision/active-inference framing of the same line) · the predator-rabbit study summary (2026-06-12) · the aggregate-stats methodology post-mortem (2026-06-09). Suggested reference home for the SDT/ideal-observer papers: `docs/project/references/perceptual_decision_making/`.
