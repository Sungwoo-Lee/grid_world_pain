# Critique — Why the continual curriculum lost to a from-scratch baseline, and whether 100× more target-level training will fix it

**Author**: professor-rl
**Date**: 2026-06-23
**Type**: RL-focused critique + literature synthesis + decision verdict
**Anchors**: [[basic_curriculum_continual_result]] (the empirical result this memo explains); [[nmn_continual_lifelong_probe]] (the project's own continual-probe direction, whose stability-gap framing this memo extends with a plasticity-loss lens); H5 (modulator timescale) in [NEUROMODULATION_ALGORITHM.md §1.4](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md).

---

## Plain-language entry point

**The question.** We trained one RecurrentPPO agent straight through a five-world difficulty
ladder, carrying its network weights forward from each world to the next (but hard-resetting
its short-term memory and rebuilding the world at every boundary). The bet was that an agent
that already knows an easy world should learn the next, harder world *faster* than an agent
starting from a blank slate. The bet failed: no world was learned faster, the fast-predator
world was learned *worse* than from scratch (negative transfer), the slow-predator world
collapsed into a degenerate "eat once then starve" policy *earlier* inside the curriculum than
it ever did from scratch, and the final hardest world ended no better than a fresh agent
despite six million extra episodes of "head start." (Full numbers and metric definitions —
survival steps, never reward — in [[basic_curriculum_continual_result]].)

**The decision this memo informs.** A follow-up run has already been launched that keeps the
early stages identical but gives the *target* world about 100× more episodes (~994 million),
on the hypothesis that the curriculum just needs more training on the final level to overtake
the baseline. **Is that likely to work?** In one sentence: **probably not, and the published
evidence says the deficit is mostly a damaged-network problem, not a too-little-training
problem — so more episodes alone will mostly be wasted unless a plasticity-restoring
intervention is added first.** This memo (a) synthesises the current RL literature on the four
mechanisms that explain our result, (b) maps each mechanism onto the specific things we saw,
and (c) gives a concrete, evidence-backed list of fixes ranked by expected payoff.

**The headline.** Three forces, all documented in recent literature, jointly produce exactly
the pattern we observed: **(1) loss of plasticity** — sequential/warm-started training silently
erodes a network's ability to keep learning, the dominant suspect for the target-level
under-performance; **(2) policy-entropy collapse** — PPO drives the action distribution to a
near-single-action point that, once reached, *resists* recovery and is the direct cause of the
"eat-once-then-starve" degeneration; and **(3) negative transfer from a bad warm start** — a
pretrained init from a *collapsed* policy is provably a worse starting point than random init.
None of these is cured by simply adding episodes. Each has a known, cheap intervention.

---

## 1. The four mechanisms, with current citations

### 1.1 Loss of plasticity in continual / warm-started deep RL — the dominant suspect

The single most relevant body of work. **Dohare et al. (Nature 2024, "Loss of plasticity in
deep continual learning")** show that standard deep-learning methods, trained on a sequence of
tasks, *progressively and silently lose the ability to learn new things* — eventually learning
no better than a shallow linear network — across both supervised and RL settings. The loss is
not visible in the current task's loss; it shows up as failure to adapt to the *next* task. The
proposed remedy, **continual backprop**, continually re-initialises a small fraction of the
least-used units to keep injecting fresh, trainable capacity.

**Abbas et al. (CoLLAs 2023, "Loss of Plasticity in Continual Deep RL")** is the closest
published precedent to *our exact setup* — a single network trained sequentially across a
series of RL tasks. They show sequential training substantially impairs learning capacity on
later tasks versus fresh init; the mechanism is **dead/dormant ReLU units proliferating** and
**effective feature rank collapsing**, which contracts the representational space available for
the next task. Their most effective fix was an **activation-function change (Concatenated ReLU /
CReLU)** that keeps units responsive; resets and regularisation helped less but still helped.

**Sokar et al. (ICML 2023, "The Dormant Neuron Phenomenon in Deep RL")** name the micro-cause:
neurons drift into an inactive state and stop contributing. Their **ReDo** ("Recycle Dormant
neurons") periodically detects τ-dormant units and reinitialises their incoming weights (zeroing
outgoing), restoring plasticity without a full reset.

**Lyle et al. (2022–2025)** tie the phenomenon to **loss of useful curvature**, parameter-norm
growth, and reduced effective rank — i.e. the optimiser loses the well-conditioned directions it
needs to keep moving. **Nikishin et al. (ICML 2022, "The Primacy Bias in Deep RL")** show the
complementary failure: agents over-fit to *early* experience and that early over-fit poisons all
later learning; their fix is **periodically resetting the last few layers** while keeping the
replay buffer / data. Recent work (Sokar; Lyle 2025; Forget-and-Grow 2025) frames all of these
as facets of one problem: **non-stationary training drives networks toward a low-rank,
low-plasticity state that more gradient steps cannot escape.**

> **Why this is the dominant suspect for our L4 under-performance.** Stage 4 (far-sight) did
> *not* fail by entropy collapse — its policy stayed stochastic (entropy-loss -0.53 → -0.62,
> healthy). It failed by *never relearning the world above ~200 despite 4M episodes on a warm
> start*. "Stochastic policy + plenty of data + still can't climb" is the textbook signature of
> plasticity loss, not of under-training. This is the regime where Dohare/Abbas/Lyle predict
> that **adding episodes does not help** — the network has lost the capacity to use them.

### 1.2 Policy-entropy collapse in PPO — the direct cause of the easy-stage degeneration

Our stage-0/stage-1 "eat once, then starve, one action chosen ~55/75 steps, entropy → 0"
degeneration is **policy-entropy collapse**: the PPO action distribution concentrates onto a
near-deterministic point, exploration dies, and performance saturates then degrades.

**Cui et al. (NeurIPS 2025, "The Entropy Mechanism of RL for Reasoning Language Models")** give
the cleanest current theory: entropy change is driven by the **covariance between an action's
advantage and its log-probability** — high-advantage, high-probability actions get reinforced,
which *lowers* entropy in a self-amplifying loop. They establish an empirical law
$R \approx -a\,e^{H} + b$: performance is *bought* by spending entropy, and once $H \to 0$ the
ceiling is hit and there is **no exploration budget left to recover**. Their fix is to clip /
KL-penalise the high-covariance updates that drive the collapse.

This matters for our verdict in two ways. First, it explains why **PPO's clip alone does not
prevent collapse** — clipping bounds the per-step ratio but does not stop the slow,
self-reinforcing entropy drain. Second — and this is the crucial point for the long-L4 plan —
**collapse is a near-absorbing state**: once entropy is at the floor, the policy gradient
samples almost one action, so it gets almost no signal about the alternatives it would need to
climb back out. More episodes *in the collapsed regime* mostly re-confirm the collapse. The
remedies in the current literature are an **entropy floor / adaptive entropy coefficient**
(CE-GPPO 2025, EPO 2025, "Rediscovering Entropy Regularization" 2025 all show adaptive
coefficients beat a fixed one), **KL-to-reference / trust-region constraints**, and the
**clip-higher trick** (DAPO) that preserves gradient on low-probability actions.

### 1.3 Negative transfer from a bad warm start — why stage 2 went *backwards*

Stage 2 (fast predator) reached only ~221 inside the curriculum vs ~418 from scratch. It was
warm-started from the *collapsed, near-deterministic* stage-1 policy and then had its recurrent
state hard-reset. This is negative transfer with a clean mechanistic explanation.

**Ash & Adams (NeurIPS 2020, "On Warm-Starting Neural Network Training")** show that a network
initialised from prior training **generalises worse than a fresh random init even when the
training loss matches**. The mechanism is **gradient imbalance**: gradients from newly-seen data
are much larger in magnitude than from already-fit data, so the warm-started optimiser takes a
biased, poorly-conditioned path. Their remedy is **shrink-and-perturb**:
$$\theta \leftarrow \lambda\,\theta + \epsilon,\qquad \epsilon \sim \mathcal{N}(0,\sigma^2 I),\quad 0<\lambda<1,$$
which shrinks the inherited weights toward zero and re-injects noise — partially "freshening"
the init while keeping coarse structure. (The paper is supervised-only; the RL analogue is
exactly the primacy-bias / reset line of §1.1.) Warm-starting from a *collapsed* policy is the
worst case: we inherit not a useful prior but a low-entropy, low-rank, dead-unit-laden network —
strictly worse than random init, which is what the ~221-vs-~418 gap shows.

### 1.4 Curriculum learning in RL — when a curriculum buys nothing

**Narvekar et al. (JMLR 2020, "Curriculum Learning for RL Domains: A Framework and Survey")** is
the reference frame. A curriculum is only justified when (a) source tasks are genuinely easier,
(b) the inter-task *transfer mechanism* preserves something reusable, and (c) the task *ordering*
is favourable. Critically, the survey is explicit that **curricula can produce no benefit or
negative transfer** when the transferred representation is specialised to the source in a way the
target penalises. Our curriculum violated (b): the thing carried forward across the boundary was
a *collapsed* policy + a *hard-reset* recurrent state — i.e. we transferred the liability (a
low-plasticity, low-entropy network) and discarded the asset (the belief state the recurrent core
had built). Recent curriculum-transfer work (Proximal Curriculum w/ Task Correlations 2024;
CADENT 2026) reinforces that the *form* of knowledge transfer, not merely the task order,
determines whether a curriculum helps. **A curriculum is not a free lunch; it is a bet that the
transfer mechanism is net-positive, and our transfer mechanism was net-negative.**

### 1.5 Recurrent-state handling across task switches — a self-inflicted cost

Each of our boundaries **hard-reset the recurrent hidden state**. In a POMDP, the recurrent
state *is* the agent's belief / history-feature encoding; resetting it throws away the one thing
that could carry cross-task competence and forces a cold re-acquisition of belief at every
switch — directly producing the ≥100-step transition dips. The continual-RL literature
(Caccia et al. 2022, "Task-Agnostic Continual RL: In Praise of a Simple Baseline") finds that
**recurrent memory carried across tasks can outperform task-aware agents** — i.e. *keeping* the
recurrent state is often the better default, and task-boundary resets should be justified, not
assumed. Our hard reset is the opposite choice and is a plausible secondary contributor to both
the transition cost and the negative transfer (stage 2 started from collapsed weights *and* a
blank belief state).

---

## 2. Mechanism-by-mechanism mapping to our result

| Observed phenomenon (from [[basic_curriculum_continual_result]]) | Dominant mechanism | Citation | Is "more episodes" a fix? |
|---|---|---|---|
| **S1 collapsed to ~94 survival, entropy-loss → -0.14, one action 55/75 steps**, *earlier* than from-scratch | Policy-entropy collapse, accelerated by warm-start into an already-low-entropy basin | Cui 2025; CE-GPPO 2025 | **No** — collapse is near-absorbing; more episodes re-confirm it |
| **S2 fast: 221 vs 418 from-scratch (negative transfer)**; inherited collapsed S1 weights + recurrent reset | Bad warm start (gradient imbalance) + dead-unit/rank inheritance + lost belief state | Ash & Adams 2020; Abbas 2023; Caccia 2022 | **No** — the init is strictly worse than random; needs perturb/reset |
| **S4 far-sight: stays stochastic but never climbs above ~200 over 4M episodes** | **Loss of plasticity** (dormant units, rank collapse, lost curvature) | Dohare 2024; Abbas 2023; Sokar 2023; Lyle 2025 | **No** — capacity to learn is the bottleneck, not data volume |
| **No stage learned faster than from-scratch** | Net-negative transfer mechanism; curriculum bet failed | Narvekar 2020 | n/a — curriculum gave no speed-up to extend |
| **≥100-step dip at every boundary, slow recovery** | Recurrent-state hard reset discards belief; cold re-acquisition | Caccia 2022 | partially — but the reset is the cause, not the budget |
| **Carry-forward *accelerated* S1 collapse** | Warm start lands in a lower-entropy, lower-rank basin sooner | Nikishin 2022 (primacy bias); Lyle 2025 | **No** — accelerant, not curable by more steps |

The table's right column is the whole answer to the decision question: **in five of six rows,
"more episodes" is not the fix, and in the one partial case the fix is to stop resetting, not to
train longer.**

---

## 3. Verdict on the long-L4 hypothesis (~994M episodes on far-sight)

**Best-evidence judgement: the long-L4 run is unlikely to surpass the from-scratch baseline,
and the literature predicts most of the 100× extra compute will be wasted, because the deficit
is primarily a plasticity / over-specialisation / entropy problem — a partially "frozen" network
— not a budget problem.**

The reasoning, made explicit:

1. **The S4 failure signature is the plasticity signature, not the under-training signature.**
   Under-training looks like "policy still exploring, loss still descending, curve still
   climbing." S4 instead showed a *stochastic* policy that *erodes* a warm start and oscillates
   around ~200 for 4M episodes. Dohare 2024 / Abbas 2023 / Lyle 2025 specifically characterise
   this as the regime where additional gradient steps do not recover performance, because the
   network's effective rank / live-unit count / curvature has degraded. 100× more of the same
   gradient is 100× more steps with a damaged optimiser substrate.

2. **The warm start the long-L4 run inherits is the *same* collapsed-then-eroded network.**
   Nothing in "more L4 episodes" repairs the dead units accumulated over stages 0–3, restores
   effective rank, or re-injects exploration. Ash & Adams 2020 show even a *clean* warm start
   underperforms fresh init; ours is a *degraded* warm start.

3. **One genuine caveat that could partially rescue it.** If far-sight's 4M-episode budget in
   the original run was *itself* below the from-scratch convergence horizon (from-scratch L4 was
   reported to converge by ~3.6M), then *some* of the S4 gap is plausibly under-training, and a
   *modest* multiple (say 2–4×, not 100×) might close part of it. But the magnitude here (100×)
   is far past any plausible under-training horizon — from-scratch L4 plateaued at ~261 by
   ~3.6M, so the marginal value of episodes 4M → 994M is, on the from-scratch evidence,
   essentially flat. The extra 990M episodes are buying compute against a flat marginal curve
   *and* a degrading substrate.

4. **Single-seed caveat cuts both ways.** Everything here is one seed; the *magnitude* of each
   effect is uncertain. But the *direction* (stochastic-but-stuck S4, collapsed S1, negative-
   transfer S2) is corroborated across two independent data sources in the original run and is
   exactly what the literature predicts, so the qualitative verdict is robust even if the
   numbers move.

**Bottom line: do not expect the long-L4 run to beat the baseline on its own. Treat it as a
control that will most likely *confirm* the plasticity diagnosis (a near-flat or eroding L4
curve out to hundreds of millions of episodes is strong evidence FOR loss-of-plasticity and
AGAINST under-training). The informative outcome is the *shape*, not a win.**

---

## 4. Recommended interventions, ranked by expected payoff

These are RL-algorithm-level recommendations. Code routing is in §5. Ranked by
evidence-strength × cheapness for our setup.

1. **Entropy floor / adaptive entropy coefficient (highest payoff, cheapest).** Prevent the
   absorbing collapse that sank S0/S1 in the first place. A fixed entropy bonus is known to be
   brittle; the 2025 literature (CE-GPPO, EPO, "Rediscovering Entropy Regularization") favours an
   **adaptive coefficient that targets a minimum entropy** rather than a fixed weight. This alone
   addresses the §1.2 mechanism and likely prevents the warm-start-accelerated collapse of §1.1's
   primacy-bias variant. *This is the single change most likely to change the outcome.*

2. **Do NOT hard-reset the recurrent state at boundaries (cheap, high-value).** Carry the belief
   state across the boundary (or warm it with a short burn-in), per Caccia 2022. This directly
   attacks the ≥100-step transition cost and part of the S2 negative transfer. Justify a reset;
   do not default to it.

3. **ReDo / periodic dormant-neuron recycling (directly targets the L4 plasticity failure).**
   Sokar 2023's ReDo periodically reinitialises dormant units mid-training. This is the
   intervention most specifically matched to the "stochastic-but-stuck S4" signature. *If the
   long-L4 run is going to run anyway, run it WITH ReDo enabled* — that converts a likely-null
   compute burn into a plasticity-restoration test.

4. **Shrink-and-perturb at each boundary (targets the bad warm start).**
   $\theta \leftarrow \lambda\theta + \epsilon$ (Ash & Adams 2020) re-freshens the inherited init
   without discarding all structure — a middle ground between full reset (loses transfer) and
   naive carry-forward (inherits the damage). Apply at the stage-1→2 boundary especially, where
   negative transfer was worst.

5. **Activation-function change to CReLU (architectural, most invasive, strongest in Abbas).**
   Abbas 2023 found CReLU the most consistent plasticity preserver. This is a cross-cutting
   change to the encoder/recurrent/heads and should be weighed against simplicity; flag for
   `professor-dl-theory` on the architectural side and `senior-developer` on the engineering
   side before committing.

6. **Last-layer resets à la Nikishin 2022 (primacy-bias control).** Periodically reset the final
   policy/value layers while keeping the representation. Cheaper than CReLU, complementary to
   ReDo.

**Minimum viable intervention set if only one thing changes:** an **entropy floor (1)** — it is
the cheapest and attacks the mechanism (entropy collapse) that is both the most clearly
demonstrated in our data and the upstream cause of the warm-start damage. **If the long-L4 run
is treated as a diagnostic rather than a fix, add ReDo (3)** so the run tests the plasticity
hypothesis directly instead of merely re-confirming the null.

---

## 5. Hand-offs

- **`experiment-designer`** — design the controls that disambiguate budget-vs-plasticity:
  (a) long-L4 *with* an entropy floor; (b) long-L4 *with* ReDo; (c) a no-recurrent-reset
  variant of the original curriculum; (d) a from-scratch L4 run extended to the *same* episode
  budget as long-L4 (the essential control — if from-scratch-L4-at-994M ≈ from-scratch-L4-at-4M,
  that proves the marginal curve is flat and kills the budget hypothesis directly). Pre-register
  the diagnostic: a flat/eroding warm-started L4 curve = plasticity confirmed.
- **`senior-developer`** — scope the code changes. Entropy-floor / adaptive-coefficient and
  recurrent-state-carry are contained (loss-term + state-handling at the boundary). ReDo and
  shrink-and-perturb are contained-but-new (a periodic hook over parameters). CReLU is
  invasive (cross-cuts encoder + recurrent core + heads, doubles some activation widths).
- **`professor-dl-theory`** — owns the CReLU / activation-function architectural question and the
  effective-rank / curvature framing if the project wants to *measure* plasticity loss (rank,
  dormant-unit fraction) as a diagnostic.
- **`professor-neuromodulation`** — note the convergence with [[nmn_continual_lifelong_probe]]:
  the modulator's temperature head is *already* a candidate entropy-floor mechanism (a
  boundary-triggered τ excursion = renewed exploration). This memo's §4.1 entropy-floor
  recommendation and the NMN's temperature head are the same lever from two directions; worth a
  joint look at whether the modulator can *be* the adaptive entropy controller.

---

## 6. Missing references to add under `docs/project/references/`

- Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton (2024). *Loss of plasticity in deep
  continual learning.* Nature 632:768–774.
- Abbas, Zhao, Modayil, White, Machado (2023). *Loss of Plasticity in Continual Deep RL.* CoLLAs.
- Sokar, Agarwal, Castro, Evci (2023). *The Dormant Neuron Phenomenon in Deep RL (ReDo).* ICML.
- Nikishin, Schwarzer, D'Oro, Bacon, Courville (2022). *The Primacy Bias in Deep RL.* ICML.
- Lyle et al. (2022–2025). Plasticity / curvature / effective-rank series.
- Ash & Adams (2020). *On Warm-Starting Neural Network Training.* NeurIPS.
- Cui et al. (2025). *The Entropy Mechanism of RL for Reasoning Language Models.* NeurIPS.
- Narvekar, Peng, Leonetti, Sinapov, Taylor, Stone (2020). *Curriculum Learning for RL Domains:
  A Framework and Survey.* JMLR 21.
- Caccia, Mueller, Kim, Charlin, Fakoor (2022). *Task-Agnostic Continual RL: In Praise of a
  Simple Baseline.*
