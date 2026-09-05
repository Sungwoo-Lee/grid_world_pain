# Critique: severity analysis and repair of the MC-return units bug in the recurrent-PPO trainer

> One-line summary: the planned repair is right and standard; the revised "0.16 sigma" severity number is honest about one defect but is measured in the wrong denominator for policy-gradient harm, cannot see the two dynamic pathologies (moving affine target, advantage-scale decay) that are plausibly the larger damage, and the explainer's gradient-clipping arithmetic ignores Adam — which also weakens the pre-registered Gate's instrument.

> **Update 2026-09-04 (professor-rl):** the endorsed repair was run and lost 3.5× to the historical code it replaced. The endorsement is retracted as stated; the reassessment, measurements, and corrected recommendation are in [§10](#10-update--the-repair-was-run-and-the-result-refutes-the-endorsement). Sections 1–9 are left unchanged.

**Author**: professor-rl · 2026-09-01
**Reviews**: the diagnosed units mismatch in `src/models/recurrent_ppo_trainer.py` (MC branch, lines ~369–386), the fix plan [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]], and the user-facing explainer (`tmp/units_bug.html`, session job 96e71c7b).

---

## Verdict (plain language)

The project's recurrent PPO trainer, in its Monte-Carlo-returns mode (the mode every live training run uses), rescales its learning targets to a standard 0-to-1 spread every training window, then feeds the resulting rescaled value estimate back in as the starting point of a sum over *unrescaled* rewards — so a window-edge correction added in July delivers only ~4% of its intended effect. The proposed repair — keep returns in raw reward units as the critic's target and rescale only the advantages, exactly as the neighbouring GAE code path already does — **is the correct, standard fix, and I endorse it**. The severity analysis under review went through two framings: first "the sign of the learning signal is inverted" (retracted after user pushback), then "an average target distortion of 0.16 standard deviations". My assessment: the retraction was correct, but the revised number is **still not the right severity statement**. It measures a one-iteration snapshot of one of *five* things the bug changes, in units (return spread) that are the wrong denominator for harm to the policy update (advantage spread, which is much smaller). The two defects a snapshot cannot see — the critic chasing a target whose zero point and scale move every iteration, and the policy's effective step size silently shrinking as the critic fits — are plausibly the larger damage. Separately, the explainer's "actor update scaled to 2%" clipping arithmetic is wrong under Adam, and the same objection reaches the fix plan's safety gate. Details, corrections, and literature below.

---

## 1. What the analysis gets right

Credit where due, because the core of this diagnosis is sound and better-instrumented than most:

1. **The bug is real and correctly localised.** `compute_mc_returns` is seeded with the raw value-head output (`:311–314`), the scan accumulates raw rewards (`:127`), and the accumulated returns are z-scored and used as the critic target (`:378–379`). At convergence `V ≈ (G − μ)/σ`, so the seed arrives at ~1/σ ≈ 4% strength. The GAE branch (`:391–396`) is the standard formulation and is unaffected. Verified against the code.
2. **The self-correction was good epistemics.** The original "good moments were relabelled bad" framing compared raw-unit columns across a normalisation the critic never sees; retracting it after checking invariance was right. The raw-units table in explainer §4 *would* mislead an RL-literate reader if the correction box were removed.
3. **The chosen repair is the field's default.** Raw value targets + normalised advantages is what reference PPO implementations do (Schulman et al. 2017; Stable-Baselines3; CleanRL), what this file's own GAE branch does, and what the 2026-07-23 diagnosis (Findings 1 and 3) already recommended. Rejecting seed de-normalisation (`μ + σ·V`) is right — it patches the symptom, keeps the non-stationary target alive, and adds lagged cross-iteration statistics.
4. **The "irreducible structured noise" insight is genuinely the sharpest observation in the analysis** — with one correction (§4 below): part of the artefact *is* absorbable, and the unabsorbable residual is noise, not bias.

---

## 2. Q1 — Is the z-scored comparison the right counterfactual?

**Partly. It is the right invariance class for the critic, applied at the wrong scope, and it answers a narrower question than the one being asked.**

The defensible core: a regression target's affine frame is (approximately) absorbable by the critic — the network can represent `aV + b` about as easily as `V` — so comparing `z(correct)` against `z(broken)` is the fair way to isolate *shape* distortion of one iteration's critic target. That the post-fix code emits raw targets does **not** invalidate this: the comparison is modulo an affine map, and the post-fix target is in that same equivalence class *up to* the caveats below. So no, the setup is not wrong on its own terms.

Four problems with its terms:

**(a) The affine-absorbability premise requires the affine map to be *stationary*, and it is not.** `μ_w`, `σ_w` are recomputed from each iteration's batch (`:378`) and drift with the policy (death frequency, survival length). The z-vs-z comparison is exact for one frozen iteration and, by construction, assigns *zero* cost to the frame moving between iterations. That moving frame is Finding 3 of the July diagnosis and is the enabling defect. So the counterfactual is fine for "how distorted is this iteration's target shape" and structurally blind to "what does this do to training".

**(b) The trainer normalises over the whole batch, not per window.** `jnp.mean(returns)` / `jnp.std(returns)` at `:378` pool all 128 envs × 128 steps (~16k values). The analysis z-scored with the single window's σ = 12.5. In actual trainer units (batch σ ≈ 23–25) the same absolute errors are ≈ **0.08σ**, half the reported figure. The explainer's footnote ("this window is a mild case, the typical distortion is roughly twice as severe") has this **backwards**: a larger normalising σ makes the distortion in target units *smaller*, not larger. What *does* make typical harm worse than this window is (c).

**(c) n = 1 window, and Δ is state-dependent.** The bootstrap gap Δ = `G_future − seed ≈ G_future·(1 − 1/σ)` scales with the true future return at that particular edge state. Eventful windows — the ones containing approaches to predators, injuries, recoveries, i.e. the windows that carry this project's scientific signal — have larger |Δ|. One window of one episode of one run, with Δ ≈ 11.9 against a batch σ of 24, is an anecdote about a mild-to-typical case, not a measurement of the distribution. The 0.16σ (properly ~0.08σ) figure should carry an explicit "per-window Δ varies; unmeasured over windows" caveat or be re-measured over the recorded four-run corpus, which exists.

**(d) The "correct" column conflates bug error with critic estimation error.** The correct seed used is the *realised* discounted future return (13.18), which no critic — even one trained in the right units — would emit; a correct-units critic outputs `E[G_future | s]`, and 13.18 contains on-trajectory MC luck. For isolating *the bug*, the counterfactual seed should be the same critic's estimate expressed in raw units. Since the broken seed is ~4% strength, nearly all of Δ is bug in this instance, so this is second-order here — but it slightly overstates what the fix will recover per window, and it should be stated.

**Bottom line for Q1**: the comparison legitimately kills the "sign inversion" claim. It does not license "the honest magnitude of the defect is 0.16σ", because the fix changes five coupled things — (i) target shape (measured), (ii) target stationarity (not measured), (iii) advantage normalisation (not measured), (iv) bootstrap strength 4% → 100% (partially measured), (v) target scale ×σ (deferred to the Gate) — and the metric quantifies only (i).

---

## 3. Q2 — Does "0.16σ of target distortion" understate the harm?

**Yes, in three specific ways — and it also *overstates* it against one baseline that matters for the back-catalogue. Both directions should be reported.**

### 3.1 The wrong denominator for policy-gradient harm

The policy gradient consumes advantages, not returns. In the MC branch:

$$
A_t = z(G_t) - V_t, \qquad \text{no re-normalisation (line 380)}
$$

Once the critic fits the (normalised) target reasonably well, `A_t` is a *residual* — its standard deviation is well below 1 and shrinks as the critic improves. The bootstrap error at edge steps, `Δ·γ^k/σ_batch` ≈ up to ~0.5 in normalised units at the edge, must be compared to **that** residual scale, not to the return spread. An error worth 0.08–0.16 return-σ can be **order-1 advantage-σ at the steps it touches**. Since PPO's per-step update direction is set by the sign and magnitude of `A_t`, the correct statement is: *the average target distortion is small in return units, but the induced advantage distortion at late-window steps is of the same order as the advantage signal itself.* That is a materially stronger claim than 0.16σ, and it is the policy-relevant one.

### 3.2 Two dynamic pathologies the snapshot metric cannot see

These are, in my judgment, the larger live defects, and neither appears in the severity analysis:

**(i) The moving affine target.** The critic regresses on `(G − μ_t)/σ_t` with `μ_t, σ_t` re-estimated per iteration. Even for a state whose true value never changes, the required output drifts with batch composition. This is exactly the pathology PopArt was built for — van Hasselt et al. 2016 (*Learning values across many orders of magnitude*) showed that rescaling value targets without inversely correcting the output head ("preserving outputs precisely") injects spurious, scale-driven learning signal into every parameter of the value function. The MC branch here is target normalisation *without* the PopArt correction — the documented anti-pattern. Cost: unquantified by any per-window metric, paid every iteration, for the whole history of the project.

**(ii) The implicit, decaying policy learning rate.** Because MC-branch advantages are un-normalised residuals, the policy gradient's magnitude is proportional to the critic's current fitting error. As the critic converges, advantages shrink toward the noise floor, the policy update shrinks with them, and the *relative* weight of the entropy bonus (`entropy_coef: 0.01` against a shrinking policy-gradient term) grows without anyone choosing it. This is an uncontrolled, drifting effective step size on the actor and a drifting exploration–exploitation balance. The GAE branch does not have this (its advantages are normalised to σ = 1 every iteration). If any historically flat or high-entropy-plateau training curves are in the diagnosis series' suspect list, this mechanism is a candidate and should be flagged to whoever owns that post-mortem — it is a policy-gradient-side defect that the units fix repairs *for free*, and it belongs in the fix plan's list of expected behavioural changes (post-fix policy gradients will be *larger and stable-scaled*, independent of the clipping question).

### 3.3 The baseline question — where the number *overstates*

Severity depends on the counterfactual baseline, and the analysis mixes two:

- **Against intended semantics** (what the July H4 fix was supposed to deliver): the distortion is as analysed — last ~45 steps of every window materially truncated, ~35% of targets.
- **Against historical behaviour** (the code as it ran before H4): the pre-H4 code seeded the scan with 0.0 *and already z-scored returns*. The units bug delivers a seed of ~`G/24` instead of 0 — a marginal change of ~4%. So the *marginal* effect of the units mismatch, relative to what the project actually trained with for its whole history, is near zero: **the honest headline is "the H4 fix never worked", not "a new distortion was introduced".** Pathologies (i) and (ii) above predate H4 and have been degrading every run all along.

This matters for §9 of the explainer (are old conclusions safe?). The conclusion is right but the argument given is wrong: "the bias is uniform — same magnitude, every run, every arm" is false in general, because Δ, σ_t, and critic quality are policy- and environment-dependent, and the study arms differ in exactly those (a sensory-degraded arm has a different return distribution than its control). The *correct* defence of the cross-arm comparisons is: every arm ran with effectively no bootstrap and the same moving-target handicap, i.e. the same (wrong) estimator semantics — a level playing field, not an identical bias. For survival-step endpoints that defence holds; the explainer should swap the argument, keep the verdict.

### 3.4 My severity statement, if I had to write one

> The units mismatch per se reverts the July window-edge fix to ~4% strength; measured against intent, the last ~45 steps of every 128-step window (~35% of targets) are trained toward a truncated horizon, with edge-step advantage distortion of order the advantage scale itself, unpredictable from anything the agent observes. Measured against history, it changed almost nothing — which is precisely the finding: the fix was inert, and the two defects that were live all along (per-iteration affine re-targeting of the critic, and a policy step size that decays with critic residual) are repaired by the same one-line change. "0.16σ average target distortion" is a fair summary of the least important of these.

---

## 4. Q3 — "The critic cannot learn it because window position is unobservable"

**Sound in its conclusion, imprecise in two places, and it misses the sharpest supporting fact.**

**The missed supporting fact**: window boundaries are invisible not only to the observation but to the *recurrent state*. The GRU hidden state is reset on episode `done` only (`_h_reset_on_done`), never at window edges, and is carried across iterations (`last_h_state` threading in `collect_trajectories`). Window phase is a property of the trainer's global step counter, `t mod 128`; episode resets occur asynchronously across the 128 envs at phases uncorrelated with it (after the first iteration's common start decorrelates). What a GRU *can* count — time since episode start, plus weak clocks in the observation (satiation decay) — carries essentially no mutual information with window phase. So the claim survives, and it survives for a stronger reason than "it's not in the 27 obs dims": it is not in the *history* either, which for a POMDP agent is the relevant sufficient statistic. (Formally: the artefact is a function of a variable outside the agent's belief-state sigma-algebra.)

**Correction 1 — mean/variance decomposition.** "The critic cannot learn to predict it, however long it trains" is true of the *positional* structure but not of the whole artefact. For a fixed state `s`, its window position on any given visit is ~uniform; the artefact's *conditional mean* over positions,

$$
\mathbb{E}_k\!\left[\gamma^{k}\right]\cdot \Delta(s)\cdot(1-1/\sigma) \;\approx\; 0.15\,\Delta(s),
$$

is a function of the state and **is** absorbable — the critic converges to a value function mildly biased toward myopia (roughly 15% of the beyond-edge future term deleted on average). What is irreducible is the position-conditional *residual* around that mean: zero-mean, structured, never-shrinking target noise, worst near edges. So the correct characterisation is "a learnable myopia bias plus unlearnable positional noise", not "wholly unlearnable". The unlearnable part is smaller than the full ramp; the harm channel is inflated critic residual → noisier advantages → slower, noisier policy learning, not persistent value bias at the edge magnitude.

**Correction 2 — "would it be better or worse if the GRU could fit it?"** Worse. This architecture shares a trunk (encoder + GRU) between actor and critic, so any feature the critic recruited to fit the artefact would sit in the actor's representation too, and — worse — a critic that partially encoded window phase would emit phase-contaminated bootstrap seeds, feeding the artefact back into target construction. Unlearnability is the milder pathology here. The shared trunk also breaks the implicit "only the critic suffers" framing throughout the explainer: with `vf_coef: 0.5`, critic-target noise reaches the actor's representation through shared-trunk gradients regardless of the clipping question.

**Literature**: this whole artefact class — truncating at a time limit that is not part of the state — is formalised in Pardo et al. 2018, *Time Limits in Reinforcement Learning* (ICML): either make the truncation variable observable ("time-awareness") or bootstrap the value at truncation ("partial-episode bootstrapping"). The project correctly chose the second (H4); the units bug silently disabled it, leaving the exact biased estimator Pardo et al. warn against for the last segment of every window. Cite this in the fix plan; it settles that the repair target is standard, not bespoke.

---

## 5. Q4 — Why the field normalises advantages, not returns

The user's statement to their user was correct but incomplete. The complete argument has three parts, and the third is the one this bug illustrates:

1. **Advantages are scale-free by role.** The policy gradient's direction (and PPO's clipped surrogate's fixed points, approximately) are invariant to positive rescaling of advantages; per-batch advantage normalisation is a step-size/conditioning device, not a change of objective. Empirically standard and mildly beneficial (Andrychowicz et al. 2021, *What Matters for On-Policy Deep Actor-Critic Methods?*, ICLR — which also documents that per-batch normalisation introduces small biases the field tolerates).
2. **Value targets define a function, not a step.** The critic target is a regression label whose *meaning* downstream code relies on: `V` must estimate `E[G]` in reward units because it is consumed as a Bellman/bootstrap quantity. An affine per-batch map on labels changes the learned function every iteration.
3. **Value targets close a loop; advantages don't.** Advantages are consumed once and discarded. The value estimate re-enters the return recursion as a bootstrap. Any unit inconsistency in a quantity inside a recursion **compounds and becomes self-referential** — which is precisely this bug. Where the field does want normalised value learning, it uses machinery that keeps the recursion's units consistent: PopArt (van Hasselt et al. 2016; deployed at scale in Hessel et al. 2019, *Multi-task Deep RL with PopArt*), reward normalisation by running return std *upstream* of the recursion (the OpenAI-Baselines `VecNormalize` convention; its load-bearing role documented in Engstrom et al. 2020, *Implementation Matters in Deep Policy Gradients*, ICLR), return-based scaling (van Hasselt et al. 2021), or symlog targets (Hafner et al. 2023, DreamerV3).

**One archaeological note worth adding to the fix plan**: the MC branch is commented "PyTorch parity" (`:370`), and normalise-the-returns is the idiom of the classic `pytorch/examples` REINFORCE script — where it is *harmless*, because in critic-free REINFORCE the normalised return *is* the advantage estimate and nothing bootstraps from it. The pattern became a bug at the exact moment H4 closed the loop through a critic. This is the clean answer to "how did this survive review": the pattern is legitimate in the lineage it was copied from; the July fix changed the invariant that made it legitimate. That is a reusable lesson for the bug registry (a correctness invariant can be destroyed by a *different, correct* change elsewhere).

---

## 6. Q5 — Shared global clip: standard or not? And the Adam problem

**Two corrections here, one to the explainer/user-statement, one that reaches the fix plan's Gate.**

**(a) Shared global clipping IS the standard for shared-trunk PPO.** OpenAI Baselines (PPO2), Stable-Baselines3, and CleanRL all compute one total loss `L_pg + c_v L_v + c_e L_ent` and clip the global norm over all parameters (typically 0.5) under one optimizer. Fully separate actor/critic optimizers with separate clips are common **only where the networks are separate** — SpinningUp-style PPO with disjoint π/V MLPs (which typically uses no clipping at all), and off-policy families (SAC, TD3) where separation is forced by the objectives. With a shared trunk, "separate clips" is not even well-defined for the trunk parameters: the trunk gradient is the *sum* of the actor's and critic's contributions, and no head-level clip isolates the actor from a value-gradient explosion arriving through the trunk. So the statement relayed to the user — separate optimizers/clips are common practice and would dissolve this problem — needs a retraction-grade qualification: *common for separate-network architectures; neither common nor problem-dissolving for this shared-trunk one.* The value-scale problem must be solved at the source (target scale / vf_coef / loss form), not by clip partitioning. Also worth noting: the standard shared-clip regime was tuned on scale-controlled returns (Atari reward clipping, MuJoCo reward normalisation) — this environment with raw return σ ≈ 24 and no reward normalisation sits outside the regime the defaults come from, which is the honest way to frame why the Gate exists at all.

**(b) The explainer's clipping arithmetic ignores Adam, and so does the Gate's instrument.** Explainer §8 ("actor push 1 + critic push 24 → scaled to 2%") computes as if the clipped gradient were applied directly. The actual chain is `clip_by_global_norm(0.5) → scale_by_adam() → lr`. Adam's update `m̂/(√v̂+ε)` is invariant to any *time-constant* rescaling of a parameter's gradient — a uniform clip factor `f` scales `m` by `f` and `√v` by `f`, cancelling. So:

- If the clip binds *persistently and uniformly* (the early-training regime: critic near 0 vs targets of ±50–150), the actor-head parameters' effective step size largely **recovers after Adam's moment warm-up** (~1/(1−β₂) updates); the persistent harm is not an 18× actor slowdown but **direction contamination of the shared trunk**, whose gradient becomes value-term-dominated regardless of clipping.
- If the clip binds *intermittently* (the death-window regime the plan rightly worries about), Adam does not cancel it — `E[f g]/\sqrt{E[f^2 g^2]}` degrades relative to the unclipped ratio — and this is the real mechanism behind the plan's "the lesson is throttled exactly at deaths" scenario. The plan's tail-sensitive Gate (G2/G3/G4) is therefore pointed at the right regime.
- **But the Gate reads pre-Adam clip factors and infers a post-Adam consequence.** A median clip factor of 0.3 that is *constant* is nearly harmless under Adam; a clip factor of 0.9 that fires only on death iterations is harmful. The four conditions partially distinguish these (G3's binding fraction helps), but the clean instrument is cheap and direct: log the **post-Adam, pre-lr per-group *update* norms** (the output of `scale_by_adam`, or equivalently `‖Δθ_g‖/lr_g` per group) alongside the gradient norms already planned. The stop-gate statistic that actually corresponds to "the actor's applied step shrank" is the actor-group update norm, not the actor-group gradient norm times the global clip factor. One extra `optax.global_norm` per group on an already-materialised pytree; I recommend the plan add it before Stage A-obs, since the "before" run cannot be retro-instrumented. Everything else about the Gate design (per-update statistics, unsmoothed CSV, tail conditions, pre-registration) is sound and unusually careful.

---

## 7. Q6 — vf_coef, and the right fallbacks

**First, a framing correction the plan should absorb**: "we refuse to touch `vf_coef` inside a bug fix" preserves a YAML number, not a behaviour. The actor:critic gradient *ratio* is the real hyperparameter, and the fix moves it by a factor of ~σ ≈ 24 no matter what — the only choice available is *which* quantity stays fixed: the config value (chosen) or the gradient balance (rejected). That is a defensible choice, and pre-registering a gate is the right response to it, but the plan should state it as a choice between two hyperparameter changes rather than as declining one. A reader who believes "nothing was retuned" will misattribute any post-fix learning-speed change.

**On the fallback menu if the Gate trips.** The plan names (a) stationary-scale value loss (divide the value term by a slowly-tracked return scale — essentially van Hasselt et al. 2021's return-based scaling applied to the loss) and (b) symlog value targets with a symexp'd bootstrap. Both are legitimate. Two more standard candidates are missing, and I would rank one of them above both named fallbacks:

1. **Reward normalisation upstream (VecNormalize convention): divide rewards by a slow EMA of the *discounted return* standard deviation, before the recursion.** This is the field's most-used lever for exactly this profile (large-scale dense homeostatic rewards, occasional ±100 events), it shrinks return σ from ~24 to ~1 — dissolving the clip/vf_coef concern rather than gating it — and it keeps every unit in the pipeline consistent because the scaling is applied *before* the scan, not after. Engstrom et al. 2020 document it as one of the implementation details PPO's reported performance actually rests on. Under this project's rules it is safe on the evaluation side (performance is survival steps, never reward) and it is a smaller conceptual change than symlog. Its cost: one slow cross-iteration statistic (mild non-stationarity, far gentler than per-batch), and it must be excluded from any analysis that reads raw reward magnitudes.
2. **PopArt on the value head** (van Hasselt et al. 2016): normalised critic training with the output-preserving affine correction and a de-normalised read-out for bootstrapping. This is the *principled* completion of what the current broken code was half-doing; heavier to implement in the NNX/optax stack (head-weight surgery per update), so I would hold it behind option 1.
3. If symlog is ever reached for, note the stronger recent precedent for the whole family: Farebrother et al. 2024, *Stop Regressing: Training Value Functions via Classification for Scalable Deep RL* — two-hot/HL-Gauss classification targets robustify value learning against exactly this scale/outlier profile, beyond what symlog-MSE gives. If the project later wants the distributional-critic direction (Phase 2/3 temperature-head territory), that fallback doubles as a stepping stone.

My recommendation ordering if the Gate trips: **reward normalisation → stationary-scale value loss → symlog/two-hot → PopArt**, with the first named in the plan now so the "which fallback" conversation (plan's Open question 1) has the standard option on the table. None of this blocks the current fix, which should land as specified.

---

## 8. Itemised corrections to the explainer (`units_bug.html`)

For the record, what would mislead an RL-literate reader as the page stands:

| Location | Issue | Correction |
|---|---|---|
| §4 footnote ("this window is a mild case… roughly twice as severe") | Confuses per-window σ (12.5) with the batch σ (~24) the trainer actually divides by; the inference runs backwards | In trainer units the distortion is ~half the reported 0.157σ, not double; what makes typical windows worse is larger Δ in eventful windows, not smaller σ |
| §4 "the critic cannot learn to predict this error, however long it trains" | Overclaims; the positional *mean* is a state function and is absorbed (as a mild myopia bias); only the positional residual is unlearnable | State the mean/variance split (§4 above) |
| §5 "the honest size of this defect is ~0.16σ" | Return-σ is the wrong denominator for policy harm; advantages in this branch are unnormalised residuals with σ ≪ 1 | Add the advantage-scale statement (§3.1 above) |
| §8 clipping arithmetic ("scaled to 2%… falls 18-fold") | Ignores Adam's scale invariance; a persistent uniform clip factor is largely cancelled after moment warm-up; the persistent harm is trunk-direction contamination, the transient harm is intermittent (death-window) binding | Reframe per §6(b) |
| §9 "the bias is uniform — same magnitude, every run, every arm" | False in general (Δ, σ_t, critic quality are arm-dependent) | Replace with the level-playing-field argument: every arm ran the same no-bootstrap semantics |
| (relayed verbally) "separate actor/critic optimizers with separate clips are common and would dissolve this" | True only for separate-network architectures; undefined for the shared trunk, which still transmits value gradients to the actor's representation | Retract/qualify per §6(a) |
| §5 "the July fix is about 96% inert… runs behave close to how they behaved before it" | Correct, and under-used | Promote this to the headline severity statement; it is the most accurate single sentence in the document |

---

## 9. Empirical signatures to watch post-fix

Beyond the plan's Gate, three cheap confirmations that the repair is doing the intended work (all readable from the already-planned instrumentation plus standard WandB curves):

1. **`loss/value_target_mean` and `_std` become slowly-varying** (state-visitation-driven) rather than pinned at (0, 1) — direct evidence the moving affine target is gone.
2. **`loss/advantage_std` post-normalisation is 1 by construction; the *pre-normalisation* advantage std should be logged once** and should track critic residual — its historical counterpart (the un-normalised MC advantage scale) is the implicit-lr defect of §3.2; a one-off comparison run quantifies how much policy-gradient magnitude the old branch was leaking away. This is the cheapest way to settle whether §3.2 materially contributed to any historical null result, and it costs one diagnostic column.
3. **Edge-step target error**: on any recorded episode, the post-fix critic's `V` at window-edge states should sit near the realised discounted future (up to MC noise), where today it sits at ~1/24 of it. A ten-line offline check against the existing trajectory store; it verifies the loop the seven synthetic-input regression tests cannot close.

---

## References named (for `docs/project/references/` acquisition if not held)

- Pardo, Tavakoli, Levdik, Kormushev, 2018. *Time Limits in Reinforcement Learning.* ICML. — settles Q3/Q4's truncation-bootstrap point.
- van Hasselt, Guez, Hessel, Mnih, Silver, 2016. *Learning values across many orders of magnitude* (PopArt). NeurIPS.
- Hessel et al., 2019. *Multi-task Deep RL with PopArt.* AAAI.
- Engstrom et al., 2020. *Implementation Matters in Deep Policy Gradients.* ICLR. — reward-normalisation load-bearing evidence.
- Andrychowicz et al., 2021. *What Matters for On-Policy Deep Actor-Critic Methods?* ICLR. — advantage-normalisation practice.
- van Hasselt et al., 2021. *Return-based Scaling: Yet Another Normalisation Trick for Deep RL.* arXiv. — the plan's "stationary-scale value loss" fallback, named.
- Hafner et al., 2023. *Mastering Diverse Domains through World Models* (DreamerV3). — symlog fallback.
- Farebrother et al., 2024. *Stop Regressing: Training Value Functions via Classification for Scalable Deep RL.* ICML. — stronger form of the symlog fallback.

---

## Next steps

- **User / top-level session**: relay §3.4 (revised severity statement), §6 corrections (retract the "separate optimizers would dissolve this" claim; fix explainer §8), and the explainer correction table (§8) before the page is shown further.
- **senior-developer**: consider two contained amendments to [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] before Stage A-obs (both are append-level, neither reopens the design): (1) add per-group **post-Adam update norms** to the instrument — the Gate's clip-factor statistics are pre-Adam and can both over- and under-read the actor throttle (§6b); (2) name upstream reward normalisation as a third Gate fallback (§7) so Open question 1 has the standard option. Also consider logging pre-normalisation advantage std (§9.2).
- **experiment-analyzer**: after the first post-fix run, the §9 signatures are the confirmation checklist; §9.2's one-off comparison quantifies the implicit-lr defect for the diagnosis series.
- **bug-curator**: the §5 archaeology ("a correct change elsewhere destroyed the invariant that made a copied idiom safe") is a candidate pattern note for the registry row; also the explainer's §9 correction to the "pre/post-H4 runs not comparable" warning (~4% true) matches what the plan already records.
- **professor-bayesian-nn** (only if the symlog/two-hot fallback is ever activated): the two-hot critic head's calibration interacts with their remit; not needed for the current fix.

---

## 10. Update — the repair was run and the result refutes the endorsement

**Feedback from professor-rl — 2026-09-04.** Appended after the 1M-episode, 5-seed, three-arm comparison on the jump-attack environment. The original sections above are left as written so the reasoning that was wrong stays legible next to the result that refuted it.

### 10.1 What happened (plain language)

The repair I endorsed in §1–§7 — keep the Monte-Carlo return in raw reward units as the critic's target, normalise only the advantages — was implemented as a third return mode (`MC_FIXED`) and run against the historical mode (`MC`, returns rescaled per batch) and the conventional GAE mode. Everything else was held identical. The historical mode survived **139 steps**; the repaired mode survived **40**; GAE survived **42**. The repaired mode behaves exactly like GAE, which is what it was designed to do — and both are 3.5× worse than the code they were meant to fix. Five seeds each, spread of 1–3 steps. There is no reading of this in which the endorsement, as written, was adequate.

The one-sentence diagnosis, from data the runs already logged: **the repair multiplied the value-loss gradient by several hundred, the shared network's gradient norm went from 0.4 to 125–154 against a clip ceiling of 0.5, and the shared encoder–GRU trunk stopped receiving any usable policy-gradient signal.** The original memo named this mechanism (§4 Correction 2, §6b "direction contamination of the shared trunk"), ranked upstream reward normalisation as the first fallback (§7), and then endorsed shipping without it. That ordering was the error.

### 10.2 The measurements (from the fifteen finished 1M runs, WandB group `return_mode_cmp`)

Steady-state (last 20 % of training) per arm; every number is consistent across all five seeds.

| Quantity | MC (historical) | MC_FIXED (repair) | GAE |
|---|---|---|---|
| Pre-clip global gradient norm (`loss/grad_norm`; clip ceiling 0.5) | **0.36–0.48** | **125–154** | **77–86** |
| Implied global clip factor `0.5 / ‖g‖` | ≈ 1 (clip rarely binds; per-iteration max ≈ 0.8) | **≈ 0.003–0.004, every update** | ≈ 0.006, every update |
| Value loss `0.5·MSE` (units differ by arm) | 0.23–0.24 (z-units → residual σ ≈ 0.69) | 260–287 (raw → residual RMS ≈ 23) | 159–170 (raw → residual RMS ≈ 18) |
| Policy loss magnitude | 0.002 | 0.003 | 0.003 |
| Policy entropy, nats (5 actions, max 1.61) | −0.56 … −0.70 | −0.55 … −0.75 | −0.64 … −0.67 |
| Food eaten per episode | 24–30 | 1.3–2.0 | 2.0–2.2 |
| Episodes reaching the 500-step cap | 15–19 % | 0 % | 0 % |

Two things to read off this table. First, the policy loss and entropy terms are O(10⁻²–10⁻³) in every arm, so a global norm of 125–154 is, to within a percent, the value term alone; the ratio of value-gradient to policy-gradient magnitude in the shared parameters moved from order 1 to order 10²–10³. Second, entropy did **not** collapse in the failing arms — they are not stuck in a deterministic bad policy; they are exploring at the same rate as MC and failing to convert exploration into a foraging policy.

The learning-curve shape is the other decisive fact. In ten equal windows of the MC run, survival went 29 → 38 → 40 → 41 → 46 → 50 → **76 → 115** → 123 → 131 steps, and food eaten went 0.9 → 1.6 → 1.5 → 1.8 → 3.0 → 3.7 → **10 → 20** → 23 → 26. There is a **take-off between roughly 550k and 750k episodes**. MC_FIXED and GAE creep linearly 26 → 41 with no knee. Before the knee (window 6), MC leads by only 1.3× in survival and 2.5× in food — the mild lead a gradient-balance story predicts. The 3.5× endpoint is one arm crossing a discovery threshold inside the budget and the other two not.

### 10.3 Q1 — Is the gradient-swamping hypothesis right, and how to test it cheaply

**Right at the level of gradient balance, and already confirmed by logged data** — no new instrument is needed to establish that the clip binds at factor ≈ 0.004 on every update in the repaired arm and essentially never in the historical arm. What the logged data cannot separate is *which of two channels* turns that imbalance into a failure to learn:

- **Channel (a), the clip.** The actor head's gradient is scaled by the global clip factor. Under Adam a *constant* factor cancels (§6b); the measured factor varies ~2–3× across 100-iteration means and ~10× on spikes (per-iteration maxima 250–1400 against means 125–150), so it is partly intermittent, and death-heavy windows are the spikes. This predicts a moderate slowdown plus a down-weighting of exactly the informative updates.
- **Channel (b), the trunk direction.** The shared encoder + GRU receives `g_π + 0.5·g_V` with `‖g_V‖/‖g_π‖ ~ 10²–10³`. Adam is per-parameter; it cannot separate two contributions summed inside the same parameter. The trunk is therefore trained by the critic alone and the actor is a two-layer readout of features chosen to predict returns *under the current, non-foraging policy* — which do not include "where is food", because under a policy that does not eat, food position does not predict return. This is independent of clipping and it predicts a threshold failure (see 10.4).

**The cheap test is two config-only arms, 1M episodes × 5 seeds, identical to the runs that just finished (ten GPUs, about a day):**

1. **`MC_FIXED` with `max_grad_norm` raised to effectively off (e.g. 10⁴).** Recovery ⇒ channel (a); still ≈ 40 steps ⇒ channel (b).
2. **`MC_FIXED` with `vf_coef` reduced from 0.5 to ≈ 10⁻³.** This is the algebraic proxy for reward scaling: if rewards are divided by `c` the critic learns `V/c`, and the value-loss gradient becomes

$$
\nabla_\theta\,\tfrac12\bigl(V'-G'\bigr)^2 \;=\; \frac{1}{c^{2}}\,(V-G)\,\nabla_\theta V ,
$$

so scaling rewards by `1/c` and multiplying `vf_coef` by `1/c²` are the same change to the critic's gradient at the same function. With the measured return spread `c ≈ 24`, `0.5/24² ≈ 9×10⁻⁴`. Adam still trains the critic *head* at learning-rate speed regardless (per-parameter normalisation), so this arm changes only the mix inside the shared trunk and the global norm. **Take-off by ~700k episodes and final survival ≥ 100 ⇒ scale is the whole story and upstream reward normalisation will work. Final survival ≈ 40 ⇒ scale is not the story, and the raw-unit offset / non-stationarity the critic must track is the next suspect.**

The plan's per-group gradient instrument (Stage A of [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]]) would give the direct number `‖g_V,trunk‖ / ‖g_π,trunk‖`; it was never implemented, and the two ablations answer the question without it. If it is built later, the confirming value is a share ≥ 0.99 for the value term in the trunk group of the repaired arm.

### 10.4 Q2 — Does a scaling problem predict this effect size and shape?

**A uniform actor-step throttle predicts something milder — a proportional slowdown. Channel (b) predicts exactly this shape.** Foraging discovery here is a positive-feedback process: an eating event raises satiation, extends the episode, and creates more eating opportunities per episode, which multiplies the advantage signal for approaching food. Any process with that structure has a knee, and a constant-factor reduction in the policy's ability to sculpt trunk features moves the knee out by roughly that factor. A 1M budget that sits just past MC's knee and well before the repaired arm's knee produces a 3.5× gap that looks qualitative.

The project already holds the long-horizon data that adjudicates "slowdown vs. never": [[NMN_PERFORMANCE_DIAGNOSIS_v8]] §4.1.1, on the noise environment, at ~19M episodes, had unmodulated GAE at 265 ± 11 survival steps against MC at 284 ± 10, with GAE "still improving". GAE arrives, an order of magnitude later. So the honest characterisation is **an order-of-magnitude slowdown of the discovery phase, presenting as a qualitative failure at a 1M budget** — not a permanent inability. The running 10M comparison is the direct test: the repaired arm and GAE should take off somewhere between roughly 3M and 10M episodes. If they have not by 10M, channel (b) is stronger than "slow" — the critic-trained representation has locked in — and the fix has to change the architecture's coupling, not just the scale.

What else could produce the shape, and why the data argue against it: **entropy collapse** — no, entropies are matched across arms; **estimator variance** — no, MC_FIXED and GAE use different estimators and match each other to within a step, while MC and MC_FIXED share the same estimator and differ 3.5×, so the variable is the target bookkeeping, not λ; **the H4 bootstrap now working** — the seed is 4 % of one edge value per 128-step window and cannot move survival from 139 to 40. One secondary contributor that is real but small: the raw-unit critic must first learn an offset of roughly −50 (mean discounted return under a policy that always dies within ~40 steps) and then track its drift as survival improves, at Adam's per-parameter learning-rate speed; the z-scored critic never pays this. It is a few hundred updates, not the whole run.

### 10.5 Q3 — Was the endorsement wrong, or the implementation incomplete?

**The endorsement was wrong in structure, and the implementation was incomplete in a way the endorsement licensed.** Specifically:

- §6a of this memo stated that "this environment with raw return σ ≈ 24 and no reward normalisation sits outside the regime the defaults come from". §7 stated that the fix "moves the actor:critic gradient ratio by a factor of ~σ no matter what — the only choice available is which quantity stays fixed", and then accepted holding the YAML `vf_coef` fixed. That is the error in one sentence: I identified which hyperparameter moves, and endorsed holding the wrong one still. The ratio moves by `σ`–`σ²` (measured: ≈ 300×), not by a factor a 2× gate tolerance could absorb.
- §7 ranked upstream reward normalisation as the *first fallback if a gate trips*. It should have been a **precondition**. Every one of the nine library precedents in the survey ([ppo_return_normalization_survey](../references/modulation_in_rl/ppo_return_normalization_survey.md)) regresses the critic on a raw target **at a return scale of order one**: Baselines / SB3 / CleanRL-continuous via `VecNormalize`-style scaling by a running discounted-return standard deviation, Atari via reward clipping to ±1, MuJoCo via rewards that are already O(1). I read the survey's column "raw target" as the convention and missed that the *pair* — raw target at unit scale — is the convention. "Raw critic target + normalised advantages" is not invalid without scale control; it is the correct target semantics. It is **incomplete**: it is one-and-a-half of a three-part package, and the missing part is the one that keeps the shared-trunk gradient balanced.
- The literature that settles this was in my reference list and I under-weighted it. Engstrom et al. (2020) identify reward scaling as one of the implementation details PPO's reported results actually rest on. Andrychowicz et al. (2021), the largest on-policy ablation, put "check whether value-function normalisation improves performance" in their top-level recommendations, alongside observation normalisation. Cobbe et al. (2021, *Phasic Policy Gradient*) document the shared-trunk interference channel directly: with shared parameters the relative weight of the value objective is a sensitive hyperparameter, and value-gradient scale degrades policy learning — their motivation for decoupling. Raileanu & Fergus (2021, IDAAC) report the same. van Hasselt et al. (2016, PopArt) is the canonical statement that value-target scale must be controlled *before* it reaches the shared parameters.
- The Gate as designed measured channel (a) and carried my own Adam caveat that a constant clip factor is mostly harmless to the actor head. Channel (b), which I named and called "the persistent harm", had no stop condition. Had the Gate run it would have tripped anyway (G4 fires below 0.25; the measured factor is 0.004), so the plan's safety net was adequate to the plan; it was skipped. But a memo that names the dominant channel and builds the gate around the other one is not a memo that endorsed correctly.

The honest recommendation now: **all three together or none.** Raw target, normalised advantages, and return-scale control upstream (running discounted-return standard deviation, scale only, no mean subtraction, SB3/Baselines semantics) — or, equivalently for the gradient balance, a `vf_coef` reduced by `σ²`, or PopArt. Shipping the first two alone is not a smaller change than the historical code; it is a ~300× hyperparameter change disguised as a bookkeeping fix.

### 10.6 Q4 — Does this vindicate the historical MC branch?

**No. It shows the historical branch contains one correct ingredient, by accident, wrapped in the defects the memo already listed.** Decompose what per-batch z-scoring of returns does:

1. **Scale** — divides the target by `σ_batch ≈ 24`, which divides the value-loss gradient by ≈ σ and keeps the shared trunk's gradient balanced at order 1. This is the ingredient that wins, and it is exactly what `VecNormalize` reward scaling, Tianshou's `return_scaling`, PopArt, and return-based scaling (van Hasselt et al. 2021) deliver — with a running statistic instead of a one-batch one. The closest published precedent for what the MC branch does is *return scaling with a one-iteration window plus mean subtraction*.
2. **Centre** — subtracts `μ_batch`, so the critic never learns the −50 offset or its drift. Mild help here; Tianshou's and Baselines' authors report mean subtraction as harmful in their settings, and it is what destroys the critic's absolute-value information (§3.2(i)).
3. **Per-batch statistics** — the moving affine target. Still a defect.
4. **Mixed-unit bootstrap seed** — the H4 fix at ~4 % strength. Still a defect.
5. **Un-normalised residual advantages** — the "implicit decaying learning rate" I flagged in §3.2(ii). **Overstated at this operating point**: the value loss plateaus at 0.24 from the third window onward, so the residual standard deviation sits at ≈ 0.69 and does not decay; the actor's effective step is stable and within 30 % of the normalised arm's. Retract the "uncontrolled, drifting step size" framing for this environment.

There *is* a principled, environment-specific reason scale control matters more here than in the survey's reference environments. The reward is potential-shaped: `r_t = D(s_{t−1}) − D(s_t) − 100·[death]`, with `D` the homeostatic drive (0–141). Summing by parts,

$$
G_t \;=\; D(s_{t-1}) \;-\; (1-\gamma)\sum_{k\ge 0}\gamma^{k} D(s_{t+k}) \;-\; \gamma^{\tau}\bigl(D(s_\tau)+100\bigr)\,\mathbb{1}[\text{death at }\tau] .
$$

The current drive enters the return at full weight while the decision-relevant per-step cost is `(1−γ)·D ≈ 0.05·D`. The return's spread is therefore set by the potential and by the 100-point death penalty, not by the differences between actions; σ ≈ 24 is large relative to the policy-relevant signal. This is why an O(1) rescale is load-bearing here and why an Atari-style pipeline with |r| ≤ 1 never meets the problem. It is an argument for **running-statistic scale control**, not for per-batch z-scoring.

Is MC winning for a reason unrelated to normalisation? The comparison holds the estimator fixed, so the win lives entirely in the bookkeeping, and the only bookkeeping difference of large magnitude is the scale. Prediction, falsifiable by the arms in 10.3: `MC_FIXED` with scale control ≥ `MC`. If `MC` still wins after scale is equalised, then per-batch centring or the non-stationarity itself is doing something beneficial in this environment — that would be genuinely surprising and would earn its own memo.

A project-wide belief needs re-attributing on the same evidence. The v8 diagnosis and the live config comments ("MC: cleaner per v8 — smaller seed variance + lower critic loss") attributed GAE's underperformance to the *estimator*. This experiment says the estimator is not the variable: MC_FIXED and GAE differ in estimator and match; MC and MC_FIXED share the estimator and differ 3.5×. The "lower critic loss" comparison in v8 was z-units against raw units and is not a comparison. GAE(λ = 0.95) was never given a fair trial in this project; it has only ever run without scale control.

### 10.7 Q5 — The next experiment

- **Leave the 10M comparison running.** It is the direct test of "slowdown vs. lock-in" (10.4) and the MC arm is the baseline every later arm is judged against. Do not add arms to it yet.
- **Launch now, config-only, 1M × 5 seeds, same environment**: (i) `MC_FIXED` + `vf_coef ≈ 10⁻³`; (ii) `MC_FIXED` + `max_grad_norm` effectively off. Pre-registered readings in 10.3. One day on ten GPUs; these two arms decide whether reward normalisation is the fix and which channel the failure runs through.
- **Queue, needs code (senior-developer)**: (iii) `MC_FIXED` + running discounted-return standard-deviation reward scaling, SB3/Baselines semantics — scale only, no mean subtraction, clip ±10, statistic updated from a per-environment discounted-return accumulator carried through the collection scan, applied *before* `compute_mc_returns` so the bootstrap seed stays in consistent units. This is the arm that becomes the default if (i) confirms. Add it to the 10M comparison only after (i) reads positive.
- **Do not** revert to per-batch z-scoring as a considered design, and do not treat the current `MC` branch as validated by this result. It is the best-performing configuration the project has, and it is also the one with the moving target and the inert H4 seed.

### 10.8 Retractions and corrections to the sections above, itemised

| Section | Claim | Status |
|---|---|---|
| §1.3, Verdict | "The proposed repair … is the correct, standard fix, and I endorse it" | **Retracted as stated.** Correct target semantics; incomplete without return-scale control; a ~300× change to the effective `vf_coef`. |
| §7 | Reward normalisation is "the leading fallback if the Gate trips" | **Corrected**: it is a precondition of the raw-target convention, not a fallback. |
| §7 | "the fix moves [the ratio] by a factor of ~σ ≈ 24" | **Understated**: measured ≈ 300× on the global norm, consistent with σ-to-σ² depending on critic residual. |
| §6b | Persistent uniform clipping is "largely cancelled after Adam's moment warm-up"; the persistent harm is trunk-direction contamination | Mechanism stands; **magnitude was not stated** — the trunk channel can prevent discovery, not merely slow it. |
| §3.2(ii) | MC's un-normalised advantages give an "uncontrolled, drifting effective step size" | **Overstated here**: residual σ ≈ 0.69, stable from the third window. |
| §3.3 / §9 | Cross-arm historical comparisons on MC are on a level playing field | Stands, and gains a clause: every historical modulator comparison ran under *accidentally correct* gradient balance; any future switch of return mode must carry scale control or every modulator comparison will be confounded by the switch. |

### References added

- Cobbe, Hilton, Klimov, Schulman, 2021. *Phasic Policy Gradient.* ICML. — shared-trunk value/policy interference; the precedent for channel (b).
- Raileanu & Fergus, 2021. *Decoupling Value and Policy for Generalization in Reinforcement Learning* (IDAAC). ICML.
- Ng, Harada, Russell, 1999. *Policy invariance under reward transformations.* ICML. — the potential-shaping decomposition in 10.6.

### Next steps

- **experiment-designer**: the two config-only arms in 10.7 (i)–(ii), 1M × 5 seeds, same environment and seeds as the finished comparison; pre-registered readings are in 10.3.
- **senior-developer**: arm (iii), running discounted-return-std reward scaling upstream of the MC scan; keep the estimator untouched. Also consider whether the Stage A per-group gradient instrument is still worth building given the ablations answer the question.
- **experiment-analyzer**: on the 10M comparison, report the episode at which each arm's survival first exceeds 80 steps (the knee), not just the endpoint.
- **bug-curator**: the KNOWN_BUGS row for "H4 bootstraps in the wrong units" should record that the raw-target repair is unsafe without scale control, so the row is not closed by `MC_FIXED` alone.
- **pi**: the v8-era attribution "GAE underperforms because of the estimator" is refuted; any roadmap item that inherited it should be re-read.
