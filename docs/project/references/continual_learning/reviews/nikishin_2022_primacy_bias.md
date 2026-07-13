> **Per-paper review — continual-learning corpus, paper 11 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§11); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 11. Nikishin et al. (2022) — The Primacy Bias in Deep RL

**PDF:** `docs/project/references/continual_learning/sources/Nikishin et al. 2022 - The Primacy Bias in Deep RL.pdf`
**Venue:** ICML 2022. **Authors:** Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, Aaron Courville (Mila, Université de Montréal).

## Phase 1 — Foundational Overview

**The problem in one sentence.** Deep RL agents tend to **overfit to their earliest experiences** and then can't leverage the better data they collect later — the paper borrows the cognitive-science term **primacy bias** (like a guitarist who learns a passage badly first and can never unlearn the bad fingering even after being shown a better way). Because RL trains on a *growing* replay buffer, the agent sees its initial samples far more often, and that early over-specialization poisons the rest of training.

**Two clean demonstrations.**
1. **Heavy priming.** Train Soft Actor-Critic (SAC) on quadruped-run, but after collecting only the *first 100 transitions*, hammer the agent with $10^5$ gradient updates on that tiny buffer, *then* resume normal training. The agent *never recovers* — even after ~1M fresh transitions it can't learn. Extreme overfitting to early data is essentially unrecoverable.
2. **The data is fine; the learner is broken.** Take the buffer collected by a primacy-biased agent (SAC at 9 updates/step, which fails) and hand it to a *fresh* agent as its starting buffer. The fresh agent learns rapidly to near-optimal. So the primacy bias is *not* a failure to collect good data — it is a failure to *learn from* good data.

**The fix — resets.** Periodically re-initialize the *last few layers* of the agent's networks (from scratch) *while keeping the replay buffer intact*. Counterintuitively, throwing away learned weights *improves* final performance. It works across discrete (Atari 100k, SPR) and continuous (DMC, SAC & DrQ) domains, image and state inputs, prioritized and uniform replay, at *no extra compute cost*.

**Key findings.**
- Resets consistently improve IQM performance (e.g., SPR+resets on Atari 100k: IQM 0.478 vs SPR 0.380; SAC+resets on DMC: IQM 656 vs 501; DrQ+resets: 762 vs 569).
- The **higher the replay ratio** (updates per environment step), the **bigger the benefit from resets** — because high replay ratio is exactly what amplifies overfitting to early data. Resets let SAC reach its *best* performance at replay ratio 32 (+100% over no-resets) and stay functional even at extreme ratios 128/256.
- **Longer n-step targets** (higher-variance value estimates) also make the agent more primacy-prone, and resets help more as $n$ grows.
- **Keeping the buffer is essential** — emptying it at each reset is highly detrimental. The buffer acts as a non-parametric world model preserved across the reset.
- Resets also rescue **TD failure modes** (critic collapse in sparse reward; TD divergence / value overestimation) by giving the optimizer a fresh start.

**Initial takeaway.** Resets become a *first-class RL tool*: a tailor-made regularization against early-data overfitting that plain L2/dropout can't match. This paper is the direct parent of D'Oro et al. (2023) "Breaking the Replay Ratio Barrier," which turns resets into a scaling lever — and the RL descendant of Ash & Adams' shrink-and-perturb and Berariu's top-layer-reset probe.

## Phase 2 — Graduate-Level Deep Dive

**Definition (the primacy bias).** *A tendency to overfit early experiences that damages the rest of the learning process.* Deliberately wide-ranging: it has multiple roots (replay's over-exposure to initial samples; high replay ratio; high-variance n-step targets; TD instability) and multiple effects, all tied to improper learning from early data.

**Why RL amplifies it.** Standard components magnify the bias: (i) **experience replay** exposes the agent to its earliest samples more often than recent ones (they've been in the buffer longest); (ii) **replay ratio** — for sample efficiency, agents take many gradient updates per env step, re-fitting the same (early-dominated) data repeatedly; (iii) **n-step targets** $\mathbb{E}_\pi[r_t + \gamma r_{t+1} + \dots + \gamma^n Q^\pi(s_{t+n},a_{t+n})]$ trade bias for variance, and higher variance makes early overfitting easier. Compounding loop: an overfitted agent collects *worse* data, which further degrades learning.

**Experiment 1 — heavy priming (unrecoverable overfitting).** SAC on quadruped-run, default 1 update/step. Experimental arm: after 100 collected transitions, perform $10^5$ updates on that buffer, then resume. Figure 1: primed agent flatlines even after ~$10^6$ new transitions. Demonstrates the *compounding, near-absorbing* nature of the failure — a small early perturbation is amplified irreversibly.

**Experiment 2 — the buffer is sufficient (locate the failure in the learner).** SAC at 9 updates/step fails (primacy bias). Re-initialize a *fresh* agent but seed it with the *failed agent's buffer*: it learns rapidly to near-optimal (Fig. 2). Conclusion: the data is adequate; the *overfitted network* is what can't distill it. Random-init networks are unaffected by primacy bias and can fully exploit the collected experience.

**The intervention — resetting.** *Given an agent's network, periodically re-initialize the parameters of its last few layers while preserving the replay buffer.* Only two hyperparameters: reset periodicity and how many layers to reset. Domain-specific instantiation:
- **SPR (Atari 100k):** reset only the final linear layer of the 5-layer Q-network, every $2\times10^4$ steps.
- **SAC (DMC, dense state):** reset the *entire* 3-layer networks, every $2\times10^5$ steps (both Q-nets + targets, due to double Q-learning).
- **DrQ (DMC, pixels):** reset the last 3 of 7 layers of policy and value nets, ~10 times over training; buffer holds only the most recent 100k transitions.
No pre-training of the fresh parameters; return directly to the normal interaction/update cycle. Optimizer statistics are also reset but this has "almost no impact" (Adam moments recover quickly).

**Why does it recover so fast?** Two complementary explanations:
1. **Model-based view of the buffer.** The replay buffer is a *non-parametric model of the world*. After a reset, the agent forgets its (over-specialized) *behavior/parameters* but *retains its world model* in the buffer as the core of its knowledge. Emptying the buffer is highly detrimental (Appendix B) — confirming the buffer, not the weights, holds the essential knowledge.
2. **Representation-recovery view.** Zhang et al. (2019): most of learning amounts to recovering the right *representations*; with the buffer preserved, re-learning a good policy/actuator from good features is comparatively fast.

Resets trigger a *virtuous circle* (the inverse of the primacy-bias vicious circle): freed from negative priming, the agent leverages accumulated data better → improves → collects higher-quality data → better future updates.

**Resets as regularization.** If primacy bias is a special form of overfitting, resets are a *tailor-made* regularizer. Table 5 (Appendix B): resets overcome the primacy bias even where standard L2 and dropout *fail*, because the pathology is a discrete, accumulated over-specialization that continuous shrinkage doesn't undo.

**Interaction with replay ratio (the key scaling result).** Replay ratio = gradient updates per env step. Figure 5: reset benefit grows with replay ratio — SPR +40% at 4 updates/step; SAC's best performance at replay ratio 32 where resets add +100%; SAC remains "reasonable" at extreme ratios 128/256 where learning is otherwise "barely possible." Resets *reshape the hyperparameter landscape*, creating a new optimum at higher replay ratio (higher sample efficiency). This is the seed of D'Oro 2023's "replay-ratio barrier."

**Interaction with n-step targets.** Figure 6: as $n$ grows (higher target variance), the agent is more primacy-prone; reset benefit grows — up to +40% for SPR at $n=20$ (vs. none at $n=3$), 50–60% for SAC at increased $n$ (vs. 40% at $n=1$). Same mechanism: more overfitting pressure ⇒ more to gain from resetting.

**TD failure modes rescued.** (i) *Sparse-reward critic collapse:* DrQ on cartpole-swingup_sparse collapses (bootstrapping mostly on its own outputs, Kumar 2020) even though ~2% of buffer trajectories reach the goal; resets give a second chance to find a non-degenerate critic (Fig. 7 left) — evidence the primacy fix addresses *optimization*, not *exploration*. (ii) *TD divergence:* even with double Q-learning, the critic can overestimate unrecoverably; predicted values fail to decay for hundreds of thousands of steps; resets fix it (Fig. 7 right).

**Ablations (what/how to reset).** Number of layers is domain-dependent: SAC (dense state) can reset entirely; SPR best with last-layer-only (most Atari knowledge is in representations, so resetting deep layers wastes the hard-won encoder); DrQ best with last 3 of 7 (critic reset slightly more important than actor, since DrQ's encoder learns from the critic loss). Optimizer reset ≈ no effect. Reset frequency should scale with how fast the algorithm recovers; even a single reset can help. Resetting a *random subnetwork* was comparable or worse than resetting the last layers.

**Project relevance.** Primacy bias + resets is the primer's Nikishin-2022 entry (§2 Phase 2) and the origin of "resets as a first-class RL tool" (primer §4 corrective family). The replay-ratio finding directly feeds the project's `replay_ratio_speed_vs_performance` study, and its "keep the buffer, reset the last layers" recipe is a concrete, cheap intervention the project could trial against its curriculum/plasticity deficit. Note it is a *corrective* (task-boundary / periodic) intervention, complementary to *preventive* bases like LayerNorm+weight-decay (Lyle 2024, other shards).

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Guitar-learning parable of primacy bias; cognitive-science origin. Central finding: deep RL overfits early interactions; compounding vicious circle. Standard components (replay, high replay ratio) magnify it. Remedy preview: periodically re-init last few layers, keep the buffer. Four contributions: demonstrate, expose causes, propose resets, evaluate.
- **§2 Preliminaries.** MDP, $Q^\pi$, TD learning, replay buffer, replay ratio (too low = sample-inefficient, too high = overfit), n-step targets bias-variance trade-off.
- **§3 The Primacy Bias.** Definition. §3.1 Heavy priming: SAC + $10^5$ updates on first 100 transitions ⇒ unrecoverable (Fig. 1). §3.2 Primed agent's buffer is sufficient: fresh agent + failed agent's buffer learns fast (Fig. 2) ⇒ failure is in the learner, not the data.
- **§4 Have You Tried Resetting It?** Statement of the reset technique (re-init last few layers, keep buffer).
- **§5 Experiments.** §5.1 Setup: SPR/Atari 100k, SAC & DrQ/DMC; per-algorithm reset schedules; buffer preserved; IQM evaluation (Agarwal 2021). §5.2 Consistent gains (Tables 1–2). §5.3 Learning dynamics: fast post-reset recovery (Fig. 4); buffer-as-world-model + representation-recovery explanations; resets as regularization beating L2/dropout. §5.4 Elements behind success: replay-ratio interaction (Fig. 5), n-step interaction (Fig. 6), TD failure modes (Fig. 7), what/how-to-reset ablations.

# Phase 3 — Loss of Plasticity: Mechanisms and Fixes (2023–2024)

**Named and cured.** By 2023–2024 the forward failure had matured into a subfield with a mechanistic account and a menu of fixes. Entries 12–15 demonstrate the failure at scale and repair its micro-cause — activation and dormancy collapse — with CReLU, ReDo, plasticity injection, and resets-promoted-to-a-scaling-lever. Entries 16–18 supply the loss-landscape and empirical-NTK theory, a three-mechanism decomposition, and the *Nature* capstone (continual backprop). The corpus's central live disagreement — layer-normalization as cure (Lyle) vs. normalization as harm (Dohare) — lives here; it is adjudicated in the Cross-Paper Synthesis.

---
