---
title: "FiLM-as-RL-hyperparameter-modulator — the audit, in story form"
status: draft
author: top-level-claude (synthesis)
audience: user + pi + research-postdoc
date: 2026-05-18
scope: "Reader-facing companion to the technical theoretical audit. Tells the whole arc — the initial verdict, the two pushbacks the user raised, the two softenings the professor agreed to — in plain English, with every symbol and equation translated. The technical memo at `20260518_film_as_hyperparameter_modulator_theoretical_audit.md` holds the equations; this doc holds the story."
one_line_summary: "The first-pass audit said FiLM can only reach 2 of Doya's 4 neuromodulator-hyperparameters; after two rounds of user pushback the verdict softened to 'all 4 are reachable, but 2 of them require FiLM to be coupled with a separate loss-side mechanism — so the unifying claim is two-headed, not single-FiLM.'"
companion_to: "20260518_film_as_hyperparameter_modulator_theoretical_audit.md"
---

## Headline

The project asked an outside theoretical-deep-learning expert a clean math question: can the FiLM trick — a small network that outputs *scaling* and *shifting* numbers, which are then applied to another network's activations — substitute for *all four* of the classical neuromodulator-as-RL-hyperparameter assignments that Doya (2002) proposed?

The first answer came back **"partial, with a category warning"** — two of the four (the policy temperature and the reward-prediction-error gain) are clean FiLM targets, but the other two (the learning rate and the time-discount) live in *other* parts of the algorithm where a forward-pass scaling operator cannot reach them.

The user then pushed back twice. The first pushback observed that FiLM is in the *gradient* path, not just the activation path, so it can gate the learning rate by a chain-rule argument the audit had missed. The second pushback observed that the time-discount question can be solved by training the critic network *as a family* indexed by context, with the discount applied separately in the training target — a known move in the RL literature called a γ-conditional UVFA.

After both rounds, the verdict has shifted. All four assignments are now reachable in principle, but for two of them (learning rate, discount) the route is not "FiLM alone" — it is "FiLM as a conditioning channel, coupled with a separate mechanism on the loss / return side". The architecture is structurally **two-headed**, not single-FiLM-as-all-four. The audit's own original recommendation for a two-headed design has been re-derived from a second direction.

## Why we asked the question

The project's research direction memo (currently in version 5) has spent several iterations trying to find the cleanest way to motivate a *single* shared modulator architecture — one small network whose outputs control the rest of the agent's behaviour. The motivating analogy throughout has been that the brain has a handful of ascending neuromodulator systems (noradrenaline, acetylcholine, dopamine, serotonin) that each tune a different operating parameter of the rest of the brain, and the project's modulator is a computational analogue.

The clearest formal version of that analogy was given by Doya in 2002: each neuromodulator is mapped to a specific reinforcement-learning hyperparameter.

- **Noradrenaline → inverse temperature.** Sets how greedy versus exploratory the policy is. A high inverse temperature means the agent almost always picks the action it currently thinks is best; a low inverse temperature means the agent samples more broadly.
- **Acetylcholine → learning rate.** Sets how aggressively the agent updates its weights from each piece of training signal.
- **Dopamine → reward-prediction-error gain.** Multiplies the reward-prediction error (the difference between what reward the agent expected and what it actually got) in the update.
- **Serotonin → time-discount.** Sets how far into the future the agent looks when valuing states — a high discount means the agent cares about distant rewards almost as much as immediate ones; a low discount means the agent cares mostly about the near term.

The framing question for the project: if we have one small "modulator" network that emits FiLM-style scales and shifts — that is, it outputs a multiplier and an offset that are applied to another network's hidden activations — can that one network play *all four* of Doya's roles? If yes, the analogy to a single shared neuromodulator system is theoretically clean. If no, the analogy is decorative and needs reframing.

That is the question the audit was asked.

## Audit Round 1 — the first verdict

The audit's first-pass answer was **partial, with a category warning**. The metaphor that the professor used to organise the result: a FiLM operator is a "section of a forward-pass parameter bundle." In plainer language: FiLM lives in the *forward pass* of the network — it modifies activations as data flows forward — and the question for each of Doya's four hyperparameters is whether that hyperparameter's effect also lives in the forward pass, or whether it lives somewhere else.

Per hyperparameter:

**Noradrenaline (temperature) — clean.** This is the easy case. The policy temperature is just a multiplier on the policy logits before the softmax. A FiLM layer at the logit layer, with its scaling factor constrained to be a single shared number broadcast across all actions, *is exactly* a context-conditioned temperature. The professor flagged one nuance: if the FiLM scale is allowed to vary per-action (not broadcast), then the FiLM layer is silently also learning *context-conditioned action priors*, not just a temperature. That extra capacity is theoretically useful but architecturally dishonest if the project's banner is "this FiLM block IS the noradrenaline system."

**Dopamine (reward-prediction-error gain) — clean up to a gauge.** The reward-prediction error is multiplied by some gain factor in the actor / critic gradient. Multiplying the prediction error by a context-dependent gain is the same thing (mathematically) as multiplying the policy gradient by that gain — and that is, in turn, the same thing as scaling the learning rate for the policy parameters by the same factor. So the gain and the learning rate are not separately identifiable from training dynamics alone. There is a *gauge symmetry*: doubling the gain and halving the learning rate produces identical training. This means the project cannot empirically tell apart "dopamine modulation" from "acetylcholine modulation" using only on-policy training curves. It is a real identifiability hazard, not just a theoretical curiosity.

**Acetylcholine (learning rate) — original verdict: not reachable.** The learning rate is a property of the *optimiser step*, not the forward pass. A forward-pass operator like FiLM, the professor argued, cannot by construction change the optimiser's step size. The professor noted one *side channel* — when an Adam-style optimiser is in use, FiLM scales change activation magnitudes, which change the per-parameter normaliser inside Adam, which changes effective step sizes — but called this "a real effect but a side channel, not the structure Doya specified." The clean route to acetylcholine-style modulation, the audit said, is meta-gradient RL or a learned optimiser, not FiLM.

**Serotonin (discount) — original verdict: not reachable, and the obstruction is deeper.** The discount sits inside the *Bellman equation* — the recursive equation that defines the value function. The professor argued that scaling the critic's output post-hoc does not give a clean change to the Bellman fixed point: if you rescale the value-head output by some factor that depends on context, the algebra of the Bellman residual does not close into the form of "the same MDP with a different discount." The professor also flagged a worry: a context-conditioned discount might break the Bellman fixed-point's *well-definedness* unless context is absorbed into state.

**The first-round summary.** Out of Doya's four hyperparameters, two were judged clean FiLM targets (noradrenaline and dopamine) and two were judged out of reach (acetylcholine and serotonin). The unifying claim — "one FiLM block is all four neuromodulators at once" — was therefore a category error, because the two unreachable hyperparameters live in parts of the algorithm (the optimiser step; the Bellman operator) that a forward-pass scaling cannot touch.

At the end of the original audit, the professor proposed a constructive alternative: an architecturally honest **two-headed** design, where one head emits FiLM-style scales and shifts to activations (the fast, forward-pass channel) and a separate head emits scalar coefficients to the loss / return estimator (the slow, loss-side channel). The two-headed shape was the audit's exit recommendation.

## Pushback 1 — acetylcholine via the chain rule

The user's first pushback was sharp and short: *if FiLM is in the forward path, it is also in the backward path by the chain rule. At its multiplicative scale equals zero, the gradient flowing back to upstream parameters is exactly zero. That is a context-conditioned learning rate of zero for those parameters. The professor's "side channel" framing missed this.*

The math is clean. When the FiLM operator multiplies an activation by a context-dependent factor, that same factor sits between the loss and any parameter *upstream* of the FiLM layer in the gradient calculation. Doubling the FiLM scale doubles the gradient to every upstream parameter; halving it halves them; setting it to zero blocks them entirely. This is a direct, optimiser-independent (it works for plain SGD, not just Adam), per-parameter, context-conditioned modulation of effective learning rates.

The professor agreed and reclassified. Three points worth highlighting:

**The chain-rule channel is the principal route, not a side channel.** The Adam-denominator effect the original audit named is real but secondary — it is an additional optimiser-specific modulation that sits *on top of* the chain-rule channel. The principal channel is the chain-rule channel, and it works for any optimiser.

**FiLM is a generalisation of Doya's learning-rate scalar, not a substitute for it.** Doya's learning rate is one number applied to every parameter. The chain-rule channel gives a *per-channel, structured* multiplier — different parameters in the network get different effective learning rates depending on which FiLM channel they predominantly feed. The clean formal name the professor offered: "per-channel context-conditioned gradient gating at a layer cut", equivalent to "context-conditioned diagonal Jacobian re-weighting" of the loss in the FiLM-channel basis. Structurally, this is the same kind of object as the gating mechanisms in LSTM gates, gated linear units, and highway networks — they all have this same forward-multiplier-equals-backward-multiplier structure. Doya's scalar is recovered as a special case: FiLM placed at the network output, with the scale broadcast uniformly across channels.

**The coupling caveat — forward and backward are not separable.** The chain-rule channel is *the same operator* as the forward-feature-gating that FiLM is more commonly thought of as doing. They are two views of one operation, not two separate channels. You cannot have one without the other. So the audit's softened verdict says: yes, FiLM reaches acetylcholine, but only in *coupled* form — you cannot modulate the learning rate without simultaneously modulating which features the network is paying attention to.

**The empirical test that distinguishes a real chain-rule channel from a decorative one.** Stratify the upstream parameters by which FiLM channel they feed into (using, for example, the row-norms of the weight matrix into the FiLM layer). Then check: across two contexts, does the gradient direction in parameter space stay the same *within* a stratum but rotate *across* strata? If yes, the chain-rule reading is load-bearing. If the gradient direction rotates uniformly across all strata, the system is doing pure feature-conditioning and acetylcholine is decorative. Pure Doya-style scalar learning-rate modulation would predict identical direction preservation across all strata, not the stratified pattern.

## Pushback 2 — serotonin via functional equivalence

The user's second pushback was structurally similar but aimed at a different argument: *the original 5-HT objection assumed a particular interpretation — "rescale a trained critic's output" — and showed the algebra doesn't close. But there's another interpretation the original audit missed. If the critic is FiLM-conditioned, and we train it with a context-dependent discount in the target, then one critic network can store the family of value functions corresponding to different discounts, indexed by context.*

This move is well-known in the RL literature. Schaul et al. (2015) introduced Universal Value Function Approximators, which condition the value function on a goal; the natural variant here is to condition on the discount factor itself. Sherstan et al. (2018), in a paper called γ-Nets, did exactly this — one network producing different value estimates for different discounts. Fedus et al. (2019) used a related construction for hyperbolic discounting over multiple horizons.

The architecture has two pieces:
- **The FiLM conditioning channel** sits on the critic's hidden activations and shapes the critic's output as a function of context.
- **A separate context-dependent discount** sits in the *target construction* — when computing the Bellman target that the critic is regressing toward, the discount applied to the next-state value is itself a function of context.

The professor agreed that under this two-piece reading, the original "algebra doesn't close" objection does not apply. That objection was about taking a critic *trained with a fixed discount* and post-hoc rescaling its output; the residual of that operation is not the residual of any standard MDP. But that is not the operation the user is describing. The user's operation is to train a critic from scratch *with* a context-dependent discount in the target — and under that training procedure, the critic learns a different value function for each context, each one a valid fixed point of its own Bellman operator. The function approximator absorbs the family; FiLM is the indexing mechanism.

The well-definedness worry the original audit raised also dissolves. If the context is constant within a trajectory (varying only across trajectories), then for each context there is a well-defined fixed point — the standard UVFA setting, no obstruction. If the context varies *within* a trajectory, the right object is a "time-inhomogeneous discounted return", which is well-defined as long as the product of discounts goes to zero in the limit. Mathematically, that latter case is the value function of an enlarged MDP whose state is the pair (state, context) — which is the same as the "context absorbed into state" caveat the original audit had named, but treated as a bookkeeping convention rather than a real obstruction.

The softened serotonin verdict matches the user's framing exactly: **FiLM alone cannot implement a context-conditioned discount, but FiLM as a conditioning channel coupled with an explicit context-dependent discount applied in the return estimator can — as a γ-conditional UVFA.** And crucially: this is structurally *the same two-headed architecture* the original audit had recommended at its conclusion. The discount itself still lives in the return estimator, not on FiLM; the project cannot pretend FiLM "is" the discount. But the conditioning needed to make the family work is exactly what FiLM provides.

**The empirical test for this case.** The professor proposed a calibrated-effective-horizon experiment: for each context, perturb a reward at some delay and measure how much the critic's value changes. The change should scale like the discount to the power of the delay; the slope (on a log scale) recovers the effective discount; that recovered discount should match the loss-side discount used in training. Three competing readings give three different predicted patterns: a true γ-conditional UVFA shows the slope varying with context and matching the training-side discount; a pure FiLM-as-feature-conditioning with fixed discount shows the same slope across all contexts; a post-hoc rescale shows the same slope but with magnitudes that differ by the rescale factor. The slope-vs-context pattern is the discriminator.

## Where we landed

The full verdict, after both rounds of pushback:

| Neuromodulator | Hyperparameter | Verdict | Mechanism |
|---|---|---|---|
| Noradrenaline | inverse temperature | **clean** | rank-1 FiLM at policy logits, scale broadcast across actions |
| Dopamine | reward-prediction-error gain | **clean up to a gauge** | loss-side scalar coefficient; not separately identifiable from the learning rate from on-policy training alone |
| Acetylcholine | learning rate | **clean as a strict generalisation of Doya's scalar** | chain-rule channel: FiLM in the forward path is also in the backward path; per-pathway, upstream-only, coupled with feature-gating; LSTM/GLU/Highway gate flavour |
| Serotonin | time-discount | **clean as a γ-conditional UVFA** | FiLM conditioning on critic + explicit context-dependent discount in the return estimator; same two-headed (forward-pass + loss-side) shape the audit recommended |

The net effect of the two pushbacks is that the verdict has shifted from "partial, with a category warning" to closer to **"all four are reachable, but two of them require coupling FiLM with a separate loss-side mechanism."** That is, the unifying claim is no longer a category error — but it is structurally **two-headed**, not single-FiLM-as-all-four.

The architectural shape that comes out of all three rounds — original audit, ACh pushback, 5-HT pushback — converges on the *same* two-headed design from three independent directions:

1. The original audit proposed a two-headed shape (FiLM head + scalar-coefficient head to loss / return) as a constructive alternative to single-FiLM, on the grounds that the four hyperparameters live in four different parts of the computation graph.
2. The acetylcholine pushback showed that FiLM-as-learning-rate works via the chain-rule channel — but is coupled with feature-gating, so the "learning-rate-knob-only" interpretation needs a separate scalar coefficient if you want it cleanly identifiable.
3. The serotonin pushback showed that FiLM-as-discount works only as a γ-conditional UVFA — and that architecture has a separate context-dependent discount in the target, on top of FiLM on the critic.

All three roads end at "FiLM is the conditioning channel, a scalar-coefficient head is the loss-side / return-estimator-side mechanism, and the two together do the work of Doya's four neuromodulators." That is the audit's converged architectural recommendation.

## Implications and open decisions

The math-first pivot the user was considering — moving the direction memo away from "the modulator is the noradrenergic system" framing toward "the modulator is a hyperparameter-modulation operator that happens to admit a neuromodulator interpretation" — is no longer blocked by a clean theoretical refutation. The audit's verdict has softened enough that the math-first framing is *theoretically defensible*. The original "category error" was specific to a single-FiLM unifying claim; under a two-headed framing, the category error dissolves.

But the audit also does not give the user a free pass. Two important constraints survive:

**Architectural honesty.** The project cannot claim that "one FiLM block is all four neuromodulators." The structurally honest claim is that FiLM is the conditioning channel and a second head (scalar coefficients to the loss / return) is needed to reach the loss-side and return-estimator-side hyperparameters. If the project implements only the FiLM head, two of the four Doya assignments (acetylcholine and serotonin) are only partly there — acetylcholine is reachable only as a *coupled* gate (you also change the features), serotonin requires the return-estimator-side discount to be implemented separately.

**Identifiability.** The dopamine-versus-learning-rate gauge remains real. On-policy training curves alone cannot tell apart "doubling the dopamine gain" from "doubling the learning rate." Any empirical claim that the project's modulator is *doing* dopamine modulation (as opposed to acetylcholine modulation) needs a second-order signal — either a curvature / Fisher / natural-gradient term that breaks the gauge, or a controlled experiment that separates them.

These are not stoppers; they are honesty constraints on the rhetorical framing of the direction memo.

The open decisions the user still owes are unchanged in number but shifted in content. Originally — before the two pushbacks — the four options were (1) cross-check with a second discipline (professor-rl), (2) commission a v6 in math-first framing with narrowed scope, (3) commission a v6 around a two-headed architecture, (4) stay on v5's paths A/B/C/E. Option (3) — the two-headed architecture pivot — has gained the most support from this audit and its softenings, because all three rounds converged on that same architectural shape. Option (2) — the narrowed math-first framing — is now less compelling because the scope no longer needs to be narrowed to two hyperparameters; the verdict reaches all four. Option (1) — a professor-rl cross-check — is still available and would target the dopamine-learning-rate gauge specifically. Option (4) — stay on v5 paths — is still available but loses some of its appeal because the math-first reframing is now theoretically defensible.

## Pointers

- The technical audit, with full LaTeX equations and identifiability conditions: [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md`](20260518_film_as_hyperparameter_modulator_theoretical_audit.md). Both addenda are inside that file, attached to their respective subsections.
- Commits: `2b279dd` (original audit), `7eb4dc0` (acetylcholine addendum), `792a4b8` (serotonin addendum).
- The earlier neuromodulator-attribution audit that triggered the math-first pivot question: [`docs/project/critiques/20260518_lc_na_basket_attribution_audit.md`](20260518_lc_na_basket_attribution_audit.md). That audit looked at whether the noradrenergic system carries all five "hypervigilance" effects in the project's basket — and concluded that one effect (action-prior shift) is canonically dopamine-flavoured, not noradrenaline-flavoured. That audit is what raised the question of whether biology-first framing was the right move at all.
- Direction memo currently being weighed: [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md`](../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md), with its plain-English companion at [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5_guide.md`](../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5_guide.md).
