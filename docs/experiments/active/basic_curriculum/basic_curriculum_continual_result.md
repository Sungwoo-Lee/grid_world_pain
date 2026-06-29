---
title: "Basic curriculum — continual 5-stage run: did the schedule actually train?"
topic: basic_curriculum
status: active
created: 2026-06-23
last_updated: 2026-06-23
phase: 1
wandb_tag: "rppo_basic_curriculum_n106"
---

# Basic curriculum — continual 5-stage run: did the schedule actually train?

## Headline finding (plain language)

We took one agent and trained it straight through a five-world difficulty ladder
(easy 5×5 room with non-moving predators → up to a hard 10×10 world with a fast,
far-seeing predator and scarce food), letting its network weights carry forward from
each world to the next. The point of building the ladder was a bet: **if the agent
already knows how to survive an easier world, it should learn the next, harder world
faster than an agent starting from scratch** — that is the "carry-forward benefit"
the whole curriculum exists to buy. Performance is **survival steps** (how many steps
the agent lives before it starves or is killed), never reward — the project rule.

**The bet did not pay off.** Three things went wrong and only two stages came out clean:

1. **The carry-forward speed-up did not appear — and on the fast-predator world it
   looked like the opposite (negative transfer).** The fast-predator world (stage 2)
   from scratch climbs to ~418 survival steps; trained inside the curriculum it only
   reached ~220 by the end of its 2-million-episode budget. The far-sight world
   (stage 4) ended around 200-300 vs ~261 from scratch — equal at best, not faster.
   No stage clearly beat its from-scratch learning speed.

2. **The "collapse-avoidance" trick failed on stage 1 (slow predator).** We
   deliberately capped the easy stages at 1 million episodes to stay below the point
   where, from scratch, the easy worlds catastrophically forget their policy and decay
   to a degenerate "eat once, then starve" loop. Stage 0 (static) stayed healthy. But
   **stage 1 collapsed anyway — and much earlier than from scratch** (within its
   1M-episode window, vs ~9M from scratch). It crashed from a 501-step capped policy
   to the ~101-step starve floor and never recovered inside the stage.

3. **Every world-switch costs a big, slow-to-recover survival dip.** Each transition
   hard-resets the agent's short-term memory (the recurrent state) and rebuilds the
   world, so survival drops by at least one full feed-starve cycle (~100 steps) right
   after each boundary, and on the harder stages takes a long time to climb back.

**Bottom line: the curriculum, as scheduled, did NOT work** — it neither bought faster
learning nor reliably avoided the over-training collapse, and it ended the hardest world
no better than a from-scratch agent. Two stages (0 static, 3 rabbit) trained cleanly;
the other three did not. This is a single-seed run, so the *magnitude* of each effect is
uncertain, but the direction (no carry-forward win, stage-1 collapse, large transition
costs) is consistent across two independent data sources.

> **Post-hoc framing (Mode B).** There was no pre-registered hypothesis with numeric
> pass/fail thresholds for this continual run; the questions ("did each stage learn?
> did carry-forward speed things up? was collapse avoided?") were posed after the run
> existed. The from-scratch numbers we compare against are themselves single-seed point
> estimates (see [[basic_curriculum_convergence]]). Conclusions are directional, not
> precise.

---

## 1. Research question

A single RecurrentPPO agent was trained straight through the 5-stage basic curriculum
(stage budgets 1M/1M/2M/2M/4M episodes, cumulative boundaries [1M, 2M, 4M, 6M, 10M]),
weights carrying forward across stages, recurrent hidden state reset at each boundary.
Did the schedule **work**? Concretely:

1. **Per-stage survival** — for each stage, did survival climb after the reset, and where
   did it end just before the next boundary?
2. **Carry-forward benefit (the core question)** — did later stages reach competence in
   *fewer* episodes than their from-scratch converged-by numbers (1.4M / 1.2M / 3.6M for
   stages 2 / 3 / 4), because the weights were warm?
3. **Collapse avoidance** — did the 1M caps on the easy stages keep the agent out of the
   entropy→0 "eat-once-then-starve" degenerate policy that sank the from-scratch easy
   worlds?
4. **Final competence** — end-of-stage-4 survival vs the from-scratch far-sight plateau
   (~261 steps)?
5. **Transition cost** — how big is the post-boundary dip and how fast does it recover?

## 2. Methods — which metric, which source

**Metric.** Survival steps = episode length, the project headline. Reward is ignored.

**Two sources, both local (the WandB web API was not queried):**

- **Training-mean survival** — the `Episode/Steps` key from the local WandB datastore
  (`wandb/run-20260622_170154-mpql5i25/run-mpql5i25.wandb`). **Unlike the from-scratch
  runs, this datastore was NOT pruned** (51 MB, 268 dense survival points across the
  10M-episode run). This is the 128-environment rolling training average — the
  *primary*, less-noisy source.
- **Eval-recording survival** — mean `len(actions)` over the 3 greedy eval episodes per
  checkpoint, 60 checkpoints in
  `results/JAX_RecurrentPPO/20260622-170153_rppo_basic_curriculum_n106/recordings/<train_episode>/episode_*.rec.gz`.
  Noisier (3 episodes) but confirms direction and resolves the transition dips.

The survival cap is 500 steps (recorded ~501); **~101 steps** is the degenerate
"eat once, satiate, then starve" floor. Stage transitions (from the training log) fired at
episodes 1,000,021 → 2,000,054 → 4,000,040 → 6,000,035.

Entropy is read from `loss/entropy` (the entropy-loss term; **magnitude shrinking toward
0 = a more deterministic, lower-entropy policy** — the collapse signature).

Extraction + working numbers: `tmp/20260623_103000_basic_curriculum_continual.md`,
`tmp/20260623_curriculum_history.json`, `tmp/20260623_eval_curve.json`.

## 3. Results

### 3.1 Per-stage survival table (primary = training-mean `Episode/Steps`)

| Stage | World | Survival right after reset | Survival at END of stage | From-scratch plateau (single seed) | Did it learn within the stage? |
|---|---|---|---|---|---|
| 0 static | `00-static_predator_5x5` | 186 (ep 44k) → 466 by 0.2M | **473** | ~448 | **Yes** — reaches the cap fast, stays high, no collapse |
| 1 slow | `01-slow_predator_5x5` | 224 (ep 1.02M) | **94** | ~361 | **No** — crashed to the ~101 floor and stayed there |
| 2 fast | `02-fast_predator_8x8` | 85 (ep 2.07M) | **221** | ~418 | **Partial** — climbed steadily but only to ~half the from-scratch plateau |
| 3 rabbit | `03-predator_and_rabbit_10x10` | 245 (ep 4.03M) | **383** | ~428 | **Yes** — climbed to near from-scratch level |
| 4 farsight | `04-far_sight_predator_10x10` | 311 (ep 6.01M, carry-over high) | **203** | ~261 | **Yes, but** — drops after the carry-over spike, ends below from-scratch |

Final checkpoint (ep 10,000,045) **eval** survival = 297 (noisy, 3 episodes, min 191 /
max 501); training-mean at that point = 203. Both are at-or-below the from-scratch
far-sight plateau of ~261.

### 3.2 Temporal shape per stage (the part that matters)

Training-mean survival sampled across each stage (steps):

- **S0 static [0–1M]:** 44k→186, 0.2M→466, then flat **~470 through 0.98M.** Reaches the
  cap by ~0.2M and **holds** — no collapse. The clean stage.
- **S1 slow [1M–2M]:** starts 1.02M→**224**, then **drops to ~90–107 and stays there**
  for the rest of the stage; END 1.97M→**94.** A within-stage collapse to the ~101
  "eat-once-then-starve" floor. The greedy-eval recordings agree (1.1M→92, oscillating
  66–126). Action-distribution probe at ep 1.10M: one action fired 55 of 75 steps — a
  near-single-action degenerate policy, the same collapse signature as from scratch.
- **S2 fast [2M–4M]:** post-switch dip to **85** (eval: 36), then a **slow steady climb**
  — 2.5M→129, 3.0M→160, 3.5M→195, END 3.99M→**221.** It learns, but the climb is slow and
  tops out at roughly **half** the from-scratch ~418 plateau, even with 2M episodes.
- **S3 rabbit [4M–6M]:** starts **245** (carry-over keeps it well above the floor), climbs
  to END 5.98M→**383** (eval end ~353–382). The **healthiest** within-curriculum stage —
  it both starts warm and finishes near the from-scratch ~428.
- **S4 farsight [6M–10M]:** post-switch carry-over spike to **311** (eval 385), then
  **decays** to ~150–200 and oscillates noisily for the rest of the 4M-episode budget;
  END 9.998M→**203** (eval final 297). It never builds on the warm start; it erodes it.

### 3.3 Entropy (collapse corroboration)

Entropy-loss magnitude (closer to 0 = more deterministic):

| Stage | entropy-loss start → end | reading |
|---|---|---|
| S0 | -0.81 → -0.72 | healthy, retains stochasticity |
| S1 | -0.44 → **-0.14** (min -0.09) | **collapsing toward deterministic** — matches the survival crash |
| S2 | -0.09 → -0.43 | recovers spread as it relearns |
| S3 | -0.39 → -0.51 | healthy |
| S4 | -0.53 → -0.62 | healthy spread, but survival still erodes (not an entropy-collapse failure — a learning/transfer failure) |

So S1's failure **is** an over-training/entropy collapse (the thing the 1M cap was meant
to dodge); S4's underperformance is **not** — its policy stays stochastic but simply never
relearns the far-sight world above ~200.

## 4. Analysis — holding the predictions against the data

**Q1 — did each stage learn?** Two clean (S0 static, S3 rabbit), one partial (S2 fast,
learned to ~half strength), one warm-start-then-erode (S4 farsight), one outright failure
(S1 slow collapsed). **2 of 5 stages came out clean.**

**Q2 — carry-forward benefit (core question): NOT realized; on S2 it looks negative.**
The curriculum's reason for existing is faster learning on later stages. It did not
happen:
- **S2 fast** had 2M episodes of warm-started training and still only reached ~220, vs
  ~418 reached from scratch by ~1.4M. That is **worse, not faster** — consistent with
  *negative* transfer: S2 inherited weights from the collapsed, near-deterministic S1
  policy plus a hard recurrent reset, and started from a worse place than random-init did.
- **S4 farsight** ended ~200–300 vs ~261 from scratch by ~3.6M — **equal at best**, with
  no speed-up despite the warm start (it briefly spiked to 311 on carry-over, then lost
  it).
- **S3 rabbit** is the one stage where the warm start visibly helped the *starting point*
  (it began at 245, well above the floor, where from-scratch L3 started ~146–239), but it
  still took most of its 2M budget to reach ~383, so even here "faster convergence" is not
  clearly demonstrated.

The carry-forward assumption flagged as untested in [[basic_curriculum_convergence]] §7 has
now been tested once, and it **did not hold** on this seed.

**Q3 — collapse avoidance: FAILED on S1, succeeded on S0.** S0 stayed capped and healthy —
the 1M cap worked there. But **S1 collapsed inside its 1M budget**, *earlier* than the
from-scratch L1 collapse (~9M). The cap was set assuming the curriculum delays collapse;
on S1 it **accelerated** it — plausibly because S1 inherited an already-converged,
low-entropy static policy and tipped into the degenerate attractor almost immediately. The
collapse-avoidance design rests on the from-scratch collapse onsets, but **carry-forward
changes the onset**, and not in the safe direction.

**Q4 — final competence:** end-of-stage-4 training-mean 203 / eval 297 vs from-scratch
~261. **Equal-to-slightly-worse.** The curriculum did not produce a better far-sight
agent than training far-sight from scratch — it just spent 6M extra episodes getting
there.

**Q5 — transition cost:** every boundary costs a large dip:
- S0→S1: 501 → ~92, and **never recovered** (became the S1 collapse).
- S1→S2: ~94 → 36/85, recovered slowly over ~1M episodes to ~220.
- S2→S3: ~221 → 245 (mildest — carry-over actually held above the floor here).
- S3→S4: 383 → 101 (eval), then a noisy partial recovery.
The recurrent-state reset alone costs ≥ one feed-starve cycle (~100 steps); on the hard
stages recovery takes 0.5–1M+ episodes.

**Why it underperformed (mechanistic read, single seed).** The two failure modes compound:
(a) the easy stages drive the policy toward low entropy / near-determinism, and (b) each
boundary hard-resets the recurrent state and hands the next stage a policy that is both
over-specialized to the *previous* world and stripped of its working memory. The result is
that the harder stages do not start from "a competent survivor" so much as from "a
brittle, low-entropy policy mid-collapse," which is a worse-than-random-init starting point
on S2.

## 5. Conclusions — did the curriculum work?

**No, not as scheduled.** On the three decision-relevant axes:

- **Faster learning (the payoff)?** **No.** No stage demonstrably beat its from-scratch
  learning speed; S2 fast was markedly worse (negative transfer).
- **Collapse avoided?** **Partially — and the part that failed is the dangerous one.** S0
  stayed healthy; **S1 collapsed inside its budget, earlier than from scratch.**
- **Better final agent?** **No.** End-of-curriculum far-sight competence (~200–300) is
  equal-to-slightly-worse than from-scratch (~261).

**What this means for the schedule.** The current schedule's two load-bearing assumptions —
positive transfer and that the 1M caps dodge collapse — both broke on this run. Before
re-running, the design should address: (i) an **entropy floor / KL guard** so the easy
stages cannot collapse the policy before handoff (this is exactly the
[[basic_curriculum_convergence]] §9 "Related Issues" item, now confirmed to bite the
curriculum, not just the from-scratch runs); (ii) **not resetting the recurrent state** at
each boundary, or warming it up, to reduce the transition tax; and (iii) re-examining
whether the easy stages help at all, or whether starting nearer the hard worlds is better.

**Caveat — single seed.** Everything above is one seed (42) compared against single-seed
from-scratch baselines. The *directions* (no carry-forward win, S1 collapse, large
transition dips, S4 ≤ from-scratch) are corroborated across the training-mean curve and
the greedy-eval recordings independently, so they are unlikely to be pure noise — but the
*magnitudes* (e.g. "S2 reached 221 not 418") carry unknown run-to-run variance. A
2–3-seed repeat is needed to claim effect sizes. **TODO:** if this schedule is revised and
re-run, pre-register the carry-forward prediction (e.g. "stage 2 reaches ≥90% plateau in
< 1.4M episodes") so the next analysis is Mode A, not post-hoc.

## 6. Metrics Requested

| Subfield | Content |
|---|---|
| **Metric** | `eval/mean_episode_length` (+ std) logged as a WandB scalar at each eval checkpoint, on the `Episode/Number` axis. |
| **Why now** | The greedy-eval survival curve still had to be reconstructed from 60×3 `.rec.gz` files. A logged scalar would make per-stage survival a one-line read. (Same request as [[basic_curriculum_convergence]] §8 — re-raised because it bit again.) |
| **Where it'd live** | The RecurrentPPO eval/checkpoint hook in `src/` that writes `recordings/<train_episode>/`. |
| **Cost** | Cheap (one scalar + std per checkpoint). |

| Subfield | Content |
|---|---|
| **Metric** | Raw policy entropy (nats), not just the `loss/entropy` term, logged on `Episode/Number`. |
| **Why now** | Collapse detection currently leans on the entropy-*loss* sign/magnitude, which is easy to misread. A raw entropy scalar would make the entropy-floor decision (§5-i) unambiguous and give a clean collapse alarm. |
| **Where it'd live** | The PPO loss/logging path in `src/` where `loss/entropy` is already computed. |
| **Cost** | Cheap (one scalar per step). |

## 7. Related Issues

- **Over-training / entropy collapse now confirmed to break a curriculum stage (S1), not
  just from-scratch easy worlds.** An entropy floor / KL guard / eval-based early stop is
  now a higher-priority candidate (`feature-workflow`). Surfaced, not patched.
- **Recurrent-state reset at stage boundaries as a transition-cost driver** — candidate
  design change (do not reset, or warm-start the hidden state). Surfaced for
  `experiment-designer` / `senior-developer`, not patched here.

## 8. Links

- From-scratch per-level convergence + budgets (the baseline this compares against):
  [[basic_curriculum_convergence]]
- Curriculum design / world ladder rationale: [[basic_curriculum]]
- Memory insight (over-training collapse + interval rationale):
  `docs/memory/memories/curriculum_learning/20260622_1748_basic_curriculum_overtraining_collapse_and_intervals.md`
- Working files: `tmp/20260623_103000_basic_curriculum_continual.md`,
  `tmp/20260623_curriculum_history.json`, `tmp/20260623_eval_curve.json`
- Run: WandB `mpql5i25` (`rppo_basic_curriculum_n106`, group `basic_curriculum`); local
  `results/JAX_RecurrentPPO/20260622-170153_rppo_basic_curriculum_n106/`
