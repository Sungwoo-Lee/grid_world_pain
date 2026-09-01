# Critique: severity analysis and repair of the MC-return units bug in the recurrent-PPO trainer

> One-line summary: the planned repair is right and standard; the revised "0.16 sigma" severity number is honest about one defect but is measured in the wrong denominator for policy-gradient harm, cannot see the two dynamic pathologies (moving affine target, advantage-scale decay) that are plausibly the larger damage, and the explainer's gradient-clipping arithmetic ignores Adam — which also weakens the pre-registered Gate's instrument.

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
