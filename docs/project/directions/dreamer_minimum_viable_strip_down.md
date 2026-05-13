---
title: "DreamerV3 minimum-viable strip-down — joint no-homeostatic + no-predator task as the first diagnostic"
topic: direction
status: active
created: 2026-05-09
last_updated: 2026-05-09
---

# DreamerV3 minimum-viable strip-down — joint no-homeostatic + no-predator task

> **One-line summary.** The joint strip (no-homeostatic + no-predator) **isolates whether DreamerV3 can learn the locomotion-on-a-5×5-grid problem at all** with the only remaining reward being a sparse food-eaten signal. It is qualitatively different from the already-refuted "no-homeostatic-only" strip because the latter still had a $-100$ death penalty driving the reward distribution, which is exactly the conventional failure mode the team has not yet addressed (see companion critique). The joint strip is the cheapest first diagnostic *if* it is run alongside one config-knob fix from §4 of the critique, not before.

> **Status.** Direction memo. Not an implementation plan. Math first, then decision tree, then the better-strip-down argument.

---

## 1. Falsifiable question and target venue

**Question.** Under DreamerV3's canonical hyperparameters and our small-grid vector-observation setup, is there *any* reward landscape on which the algorithm reliably reaches non-trivial competence and stays there asymptotically?

**Target venue.** This is internal diagnostic work, not yet a publication target. If the answer is "yes, DreamerV3 learns the strip-down robustly," the project's downstream interoceptive-modulation work proceeds; if "no, DreamerV3 is structurally unsuited even to this," the project pivots to TD-MPC2 / IRIS / pure-PPO, and the publication direction shifts (CCN / COSYNE workshop on "limits of model-based RL with sparse-discontinuous rewards"). At this stage we are answering an **algorithm-fitness question**, not a science question.

---

## 2. The reward landscape under each strip

The reward function in the codebase is:

$$r_t = \underbrace{(D_{t-1} - D_t)}_{\text{homeostatic, dense}} \cdot \mathbb{1}[\text{use\_homeostatic\_reward}] + \underbrace{\Delta r_{\text{eat}}}_{\text{sparse positive}} - \underbrace{100 \cdot \mathbb{1}[\text{terminal,injury}]}_{\text{sparse large negative}}$$

where $D_t = \sqrt{(S_t - S^*)^2 + I_t^2}$, the homeostatic drive.

| Strip | Homeostatic reward | Predator | Death penalty | Reward distribution | Symlog-bin support |
|---|---|---|---|---|---|
| **Hypervigilance (`01-5X5_PredInterval3_NutGain18`)** | yes | yes (live + hiding) | $-100$ | mixed: $\sim \pm 0.5$ dense + $-100$ terminal sparse | bins at $\sim$ 0.4 (modal) and $-4.6$ (terminal) — bimodal |
| **NoPred only (`00-5X5_NoPred`)** | yes | only hiding_predator (sparse hazard) | $-100$ | mostly $\sim \pm 0.5$ dense + $-100$ rare terminal | weakly bimodal, terminal sparse |
| **No-homeostatic only (Cell C of prior battery)** | **no** | yes | $-100$ | sparse positive (eat) + sparse $-100$ terminal | very sparse, mostly zero |
| **Joint strip (no-homeostatic + no-predator)** | **no** | **no** | only via injury from hiding_predator (rare) | very sparse positive (eat) + occasional zero | almost-everywhere zero |

The joint strip's reward distribution is **the closest thing to canonical Atari** in our environment: sparse positive event-rewards, no per-step shaping, no large-magnitude terminal. **This is exactly the regime DreamerV3 was tuned for.**

---

## 3. What a positive result on the joint strip would prove

| Outcome on joint strip | Conclusion |
|---|---|
| **Dreamer learns it** (survives 500 steps, eats food reliably, no asymptotic collapse) | The vector-observation + small-grid + DreamerV3 stack is fundamentally workable. The hypervigilance failure is **caused by the reward landscape**, not by the architecture. Cleanly localizes the next investigation to **reward-scale fixes** (item 1 in the critique's ranked list). |
| **Dreamer learns it transiently then collapses** (the `3zjhap9w` shape on NoPred + replay_ratio=0.5 — survival climbs to high, then falls back) | The replay-ratio over-training pathology is real and **independent of reward scale**. Lowering replay_ratio is the priority intervention. |
| **Dreamer never learns it** (survival stays at ~28-step random-walk floor) | The architecture has a fundamental issue with **vector observations + small state space**, independent of reward landscape. This is the doomsday outcome and would justify pivoting algorithm. |

The joint strip is therefore a **three-way diagnostic**: it sorts the failure into "reward-scale," "replay-ratio over-training," or "structural mismatch," and each branch has a different downstream plan.

---

## 4. What a negative result would prove

A "Dreamer never learns the joint strip" outcome would refute the most-charitable hypothesis (that everything except the reward landscape is fine). It would put substantial weight on the structural-mismatch branch — that **vector observations into a Dreamer encoder, with a 19-dim reconstruction objective, do not provide enough representational signal for the WM to bootstrap from in the early training regime**. Hafner-2023 explicitly notes the encoder/decoder reconstruction objective is the "primary signal" for the WM's representation; if our 19-dim vector reconstruction is too easy (loss converges in a few thousand steps to $\sim 0.008$, leaving nothing for the WM to do), the WM does not develop a useful latent.

This branch would push toward (a) augmenting the WM with **auxiliary self-supervised objectives** (SPR, BYOL-on-latents), or (b) switching to an algorithm that does not depend on a reconstruction objective for representation learning (TD-MPC2's reward-conditioned latent, IRIS's discrete-token-decoder, or simply going back to recurrent PPO which already works on hypervigilance).

---

## 5. Difference from the already-refuted "no-homeostatic-only" strip

The prior diagnostic battery (Cell C of `20260508_1431`) ran **no-homeostatic + predator on**. Survival doubled from 30 to 44 (the only cell where actor entropy dropped to 0.55 and advantage went *positive* to +0.014), but predator-deaths still happened on 99% of episodes. The cell was scored as "refuted as a sole fix" because survival did not approach 500.

**Why no-homeostatic-only does not isolate the right thing.** Cell C still has the death penalty $-100$ at terminal, and the predator on. So the reward distribution is **sparse positive (eat) + sparse $-100$ (terminal)** — the death penalty is still 100× the eat signal in absolute magnitude. The conventional reward-scale failure mode (item 1 in the critique) is **not addressed** by removing only the homeostatic shaping; it remains the dominant feature of the reward distribution.

**The joint strip removes the predator**, which in our env drops the dominant source of $-100$ terminals. The remaining $-100$ source is **injury death from the hiding_predator hazard**, which fires only rarely (regen_delay = 20, and the agent has many turns to step away). Reward distribution is now genuinely sparse-positive-dominated, much closer to Atari.

The joint strip is therefore **not redundant with Cell C**: it isolates a strictly different question. Cell C asked "is dense homeostatic shaping the issue?" — answer: no, but the residual reward distribution still failed. The joint strip asks "is the heavy-tailed reward distribution the issue?" — a question Cell C did not answer.

---

## 6. Caveat from `3zjhap9w`: this strip has already partially been run

`3zjhap9w` is the NoPred-only run with `replay_ratio = 0.5` and `cont_loss_weight = 1` (Cell E3 of the probe battery is its near-twin). The `3zjhap9w` post-hoc analysis (`tmp/20260508_replayRatio05_NoPred_analysis.md`) shows:

- 0–600k env steps: agent learns. Survival climbs from 196 to 330. Advantage drops to $\sim 0$. Food intake rises to 30+/episode.
- 600–800k env steps: peak competence.
- 800k–1.8M env steps: **collapse**. Survival falls to 142, food intake to 7.6, starvation rises to 80%.

**This is not the "joint strip" because it still has homeostatic reward enabled.** The dense homeostatic reward is plausibly what enables the early-training competence (a dense gradient signal carries the WM through the bootstrapping phase) and is also potentially what enables the collapse (the over-fitting target is rich enough for the WM to lock in on).

**The joint strip removes both shaping rewards**, leaving only the sparse eat-event signal. The expected dynamics:

- **Slower early learning** (no dense gradient → WM has only sparse positive events to learn from).
- **More resistant to collapse** (no rich modal-reward target for the WM to over-fit to — every step's expected reward is near zero, so the value function's pessimistic basin has shallow slope).
- **Bounded survival** since there is no longer a "die soon if I don't eat" signal — the agent can wander indefinitely until episode timeout.

**This is genuinely diagnostic.** If the joint strip shows the same `3zjhap9w` learn-then-collapse trajectory, the over-training mechanism is **independent of reward density** and replay_ratio is the dominant cause. If the joint strip is monotonically learning (slow but stable), the collapse is *caused* by the dense-reward over-fitting and the conventional fix is to keep dense reward but lower replay_ratio.

---

## 7. Is there a better strip-down?

The joint strip is the simplest *task* strip-down within our existing env code. **A more conservative strip would require changing the env** — for instance:

- **Single-state navigation (no body, no homeostasis, no predator, fixed start, fixed food location)** — would test pure WM-encoder competence on a deterministic problem. But this requires removing the body state from the observation, which is a non-trivial code change and does not reflect the project's interoceptive-AI scope.
- **Deterministic predator (predator moves on fixed schedule, deterministic)** — would test whether stochastic dynamics is the issue. But our predator is already mostly deterministic (`move_interval=3`, fixed patrol area), so the gain is small.
- **Reduce action space from 6 to 4 (drop rest+eat, only directional moves)** — would test whether the action-space structure is the issue. But rest+eat are central to the body-state interaction the project cares about, and dropping them changes what the project is studying.

**Recommendation: stick with the joint strip as the no-code-change minimum-viable task.** It is reachable by a single-config copy of `00-5X5_NoPred.yaml` with `use_homeostatic_reward: false`. No env code changes.

A *complementary* simpler strip would be to also set `predator_enabled: false` AND `count: 0` on the `hiding_predator` resource, removing the last source of injury death entirely. The reward distribution then becomes "sparse positive (eat) + zero everywhere else + episode timeout at 500 steps with no terminal reward." This is **the closest analog to a sparse-reward Atari task in our env** — call it the "pure foraging" strip. Worth running alongside the joint strip if the budget allows; the comparison `joint vs. pure-foraging` localizes whether the rare hiding_predator terminal is itself a problem.

---

## 8. Should the joint strip run with default hyperparameters or with one of the cheapest-fix interventions?

**Strong recommendation: run with `replay_ratio = 0.0625` (Hafner-2023 Atari default), not with our `replay_ratio = 0.5`.** Here is why.

The `3zjhap9w` evidence is that **NoPred + replay_ratio = 0.5 already collapses**. If we run the joint strip at the same replay_ratio, we are likely to see the same collapse pattern in roughly the same env-step regime, and the experiment will not have isolated the joint-strip variable from the over-training variable.

The conservative diagnostic is a **2×2 mini-battery**:

| | replay_ratio = 0.0625 | replay_ratio = 0.5 |
|---|---|---|
| **NoPred + homeostatic (already-known: `3zjhap9w` shape)** | (new run) | already done — collapse |
| **Joint strip (new task)** | (new run) | (new run) |

**Three new runs**, each ~3h on a single node. The 2×2 design factorially separates the two variables. After this battery, we know:

1. Does lowering replay_ratio prevent the `3zjhap9w` collapse on NoPred? (top-right vs. top-left)
2. Does the joint strip behave like NoPred when both are run at low replay_ratio? (top-left vs. bottom-left)
3. Is the joint strip robust to high replay_ratio (the published Atari-default condition where collapse should be minimal)? (bottom-right vs. bottom-left)

This is the **conventional first cut**. Any further bespoke instrumentation (per-action cont, fork-rollouts) should be conditional on the outcome of this 2×2.

---

## 9. Predicted outcomes

I will commit to predictions for accountability:

| Run | My prediction | Confidence |
|---|---|---|
| NoPred + homeostatic + `replay_ratio=0.0625` | Slow climb to ~250–350 step survival, no collapse through 1.5M env steps. Demonstrates that low replay_ratio rescues `3zjhap9w`. | Medium-high |
| Joint strip + `replay_ratio=0.0625` | Slow climb to ~150–250 step survival (dense reward removed → less signal). No collapse. | Medium |
| Joint strip + `replay_ratio=0.5` | One of two outcomes. (a) Same shape as `3zjhap9w` — climb + collapse — demonstrating over-training is reward-density-independent. (b) Flat at random-walk floor because the sparse signal is too weak under over-training pressure. | Medium-low confidence on which |

If the predictions are right, the project goes back to the hypervigilance task with `replay_ratio = 0.0625` and **death_penalty = -1** as the joint conventional intervention. If the predictions fail in a structured way, we have new evidence for the next round of memos.

---

## 10. Required experiments (handoff to `experiment-designer`)

- **3 runs**: the three new cells of the 2×2 in §8. Seed 0, num_envs 16 (faster wall-clock and matches the prior battery), 1.5M env steps each (past the `3zjhap9w` collapse boundary).
- **WandB group**: `dreamer_minimum_viable_strip_down`.
- **Tags**: `dreamer_strip_NoPred_homeo_rr06`, `dreamer_strip_joint_rr06`, `dreamer_strip_joint_rr05`. Lock at design time per the user's auto-memory rule.
- **Required new config**: a copy of `configs/experiment/basic/00-5X5_NoPred.yaml` with `body.use_homeostatic_reward: false` (call it `00-5X5_NoPred_NoHomeo.yaml`); two model configs derived from `dreamer_v3.yaml` with `replay_ratio: 0.0625` (call it `dreamer_v3_rr06.yaml`).
- **Probe**: keep `imagined_rollout_probe: false` for the strip-down. The probe is now thoroughly understood and not the bottleneck. Adding probe overhead does not change the diagnostic value of the strip-down.

The Launch Manifest fields and seed grid are out of scope for this memo — `experiment-designer` owns them.

---

## 11. Required architecture changes (handoff to `senior-developer`)

**None.** The strip-down is a config-only intervention. No code changes.

The companion critique memo flags two read-only audits that `senior-developer` (or `code-reviewer`) should do *in parallel* with the strip-down runs, since they are cheap and might rule out additional confounders:

1. Verify the KL-weight implementation in `src/models/dreamer_v3_trainer.py` matches Hafner-2023 §B.1 ($\alpha_{\text{dyn}} = 0.5, \alpha_{\text{rep}} = 0.1$, free bits = 1.0).
2. Verify the sensor normalization in `src/env/sensor.py` produces values in $[0, 1]$ before the encoder, *especially* for `extero_nociception` and `olfaction` which the env config clips to $[0, 100]$.

If either audit finds a deviation from canonical, fix it before running the 2×2 and rerun the matrix.

---

## 12. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Joint strip too sparse to bootstrap → all three runs fail at random-walk floor | Medium | Add the dense homeostatic reward back as a fourth row of the 2×2 if needed; or add a small +1 eat-bonus as a sparse positive shaper |
| `replay_ratio = 0.0625` makes runs 8× slower per env-step (more env interaction per gradient update) → wall-clock budget overruns | High | Allocate parallel nodes; this is a real cost — three sequential 1.5M-step runs at 0.0625 RR is ~24h+ on one node |
| The 2×2 outcomes are ambiguous (e.g., joint strip + low RR climbs slowly, no clear collapse, but does not reach high competence either) | Medium | Pre-register a quantitative criterion: "competence" = mean survival > 250 steps over the last 200k env steps, "collapse" = survival drops by > 30% from peak across any 200k window |
| Wall-clock pressure pushes us back to bespoke probes before the 2×2 finishes | Medium-high | This memo + the critique are the formal record that the priority should be the conventional 2×2; user discretion to override |

---

## 13. Closest precedents

I cannot cite a paper that ran this exact 2×2 — it is task-specific to our env. But the design pattern is conventional:

- **Atari ablation pattern (Mnih 2013, Bellemare 2013)**: strip the env to its sparse-reward core to test algorithmic competence before adding reward shaping. The joint strip is the analog of "Atari with no reward shaping" for our env.
- **DMC sparse-reward variants (Tassa et al., 2018)**: DMC tasks come in dense and sparse versions; published results show that DreamerV3 is **competitive on the dense variants and loses to TD-MPC on sparse variants**. This is directly relevant — our hypervigilance task is more "sparse" than the paper's DMC sparse, due to the asymmetric magnitude.
- **Crafter (Hafner, 2022)**: Crafter's reward is sparse-positive ($+1$ per achievement) and sparse-negative ($-1$ on death). DreamerV3's Crafter benchmark is the **closest published match** to the joint strip's target reward distribution, and is what motivates the prediction "joint strip + low replay_ratio: Dreamer learns slowly but stably."

If the joint strip + low replay_ratio fails to learn even slowly, **the discrepancy with Crafter is the right thing to investigate next** — Crafter has image observations and richer reconstruction signal than our 19-dim vector. That would put us back at item 2.5 of the critique (vector-input mismatch), which would be a structural finding.

---

## 14. Next steps (by agent)

- **`senior-developer`** — read this memo and the companion critique. Write the implementation plan for the strip-down 2×2 + config audits. Plan should specify the four config files to add, the launch manifest, and the seed/num_envs/env-step budget.
- **`experiment-designer`** — generate the 3 configs, lock the launch manifest with WandB tags pinned at design time per the auto-memory rule.
- **`training-runner`** — collect node + GPU upfront from the user, then launch the 3 runs in parallel where possible.
- **`experiment-analyzer`** — post-run, read the 2×2 outcome through the §9 prediction grid; verdict on which of the three branches in §3 the project lands in.
- **No new probe instrumentation; no per-action conditional cont; no fork-rollouts. Those are conditional on this 2×2's outcome.**
