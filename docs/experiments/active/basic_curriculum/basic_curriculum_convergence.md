---
title: "Basic curriculum — per-level survival-step convergence & episode budgets"
topic: basic_curriculum
status: active
created: 2026-06-20
last_updated: 2026-06-20
phase: 1
wandb_tag: "rppo_basic0*_n11*"
supersedes:
superseded_by:
---

# Basic curriculum — per-level survival-step convergence & episode budgets

## Headline finding (plain language)

We ran the agent **from scratch** on each of the five basic-curriculum worlds (a difficulty
ladder from an easy 5×5 room with non-moving predators up to a hard 10×10 world with a
fast, far-seeing predator and scarce food). Each run lasted 10 million episodes. The goal
of *this* analysis is narrow: **for each world, at what episode count does the agent stop
getting better at staying alive?** — so we can hand a continual-learning curriculum a
sensible per-stage time budget. Performance here is **survival steps** (how many steps the
agent lives before it starves or is killed), never reward — that is the project rule.

Two things came out of the data, one expected and one not:

1. **The convergence numbers we wanted exist and are clean for the three harder worlds.**
   The fast-predator world (Level 2) and the rabbit world (Level 3) reach ~90% of their
   best survival level by roughly **1.2–1.4 million episodes**; the hardest world
   (Level 4, far-sighted predator + scarce food) is the slowest and only gets there by
   roughly **3.6 million episodes**, and it tops out lower (it is genuinely harder).

2. **The two *easiest* worlds (Level 0 static, Level 1 slow) did not plateau — they
   collapsed.** Both learned a strong policy early (Level 0 hit the 500-step survival cap
   by ~0.2 million episodes), then **catastrophically forgot it mid-training** and decayed
   to a degenerate "eat once, then starve at step ~100" policy. The agent's action choice
   at the end is literally walking into a wall (Level 0: 90% of its moves were a single
   repeated direction). This is over-training instability, **not** a plateau, and it
   directly shapes the budget recommendation: the easy stages must be **stopped early**,
   long before the from-scratch run would have run them into the ground.

The practical output is at the bottom: a recommended **per-stage episode budget** and the
**cumulative `episode_boundaries`** list a curriculum schedule YAML needs.

> This is a **post-hoc** analysis (Mode B): there was no pre-registered hypothesis about
> *where* each level would converge, so the framing ("when does each level plateau?") was
> set after seeing the runs exist. Conclusions about convergence points are descriptive
> and rest on a **noisy, low-sample eval signal** (3 greedy eval episodes per checkpoint).
> They are strong enough to set budgets with margin, not to make fine claims.

---

## 1. Research question

For each of the 5 basic-curriculum worlds, trained **from scratch** with RecurrentPPO for
10M episodes: at what episode count does the **mean survival-step (episode-length) curve**
reach ~90–95% of its plateau ("converged-by"), does the level learn meaningfully above the
random-policy floor, and how do the levels rank in difficulty / learning speed? Then:
recommend a per-stage episode budget and the cumulative `episode_boundaries` for a 5-stage
continual curriculum.

## 2. The 5 runs

| Level | World (config) | What changes vs. previous | WandB id | Tag | Training run dir |
|---|---|---|---|---|---|
| 0 static | `basic/00-static_predator_5x5.yaml` | 5×5, food + non-moving predators, no chaser | `7ha3jzr3` | `rppo_basic00_static_n113` | `20260619-172721_rppo_basic00_static_n113` |
| 1 slow | `basic/01-slow_predator_5x5.yaml` | + slow chaser, + cover | `wg0x8j34` | `rppo_basic01_slow_n113` | `20260619-172832_rppo_basic01_slow_n113` |
| 2 fast | `basic/02-fast_predator_8x8.yaml` | 8×8, fast chaser | `707u3kt6` | `rppo_basic02_fast_n114` | `20260619-172921_rppo_basic02_fast_n114` |
| 3 rabbit | `basic/03-predator_and_rabbit_10x10.yaml` | 10×10, + rabbit (alt food) | `macskfvn` | `rppo_basic03_rabbit_n114` | `20260619-173018_rppo_basic03_rabbit_n114` |
| 4 farsight | `basic/04-far_sight_predator_10x10.yaml` | 10×10, detection-range 5, scarce food | `c8cd77ft` | `rppo_basic04_farsight_n114` | `20260619-173115_rppo_basic04_farsight_n114` |

All: 10,000,000 episodes, `num_envs=128`, RecurrentPPO (`configs/models/recurrent_ppo/recurrent_ppo.yaml`), seed 42, `environment.max_steps=500`. **One seed per level** — see Caveats.

## 3. Methods — which metric, which source

**Metric.** Survival steps = episode length. The canonical logged key is **`Episode/Steps`**
(mean episode length over the rolling window), plotted against **`Episode/Number`**
(episode count). Cumulative reward (`Episode/Reward`) is used only as a secondary
sanity check.

**Data source — local files only (project rule; the WandB web API was NOT queried).** The
local `wandb/run-*/run-*.wandb` datastore turned out to be **pruned**: it retains only the
first 3–16 history points of each 10M-episode run, so it cannot reconstruct the plateau.
The full, dense, *local* survival-step curve was instead reconstructed from the **eval
recordings** the trainer writes to disk:

```
results/JAX_RecurrentPPO/<run>/recordings/<train_episode>/episode_*.rec.gz
```

There are **50 checkpoints per run** (one every ~200k episodes — the directory name equals
the `train_episode` field, i.e. the `Episode/Number` axis), each with **3 greedy eval
episodes**. Per episode, survival steps = `len(actions)`. Per checkpoint we take the mean
of the 3 eval episodes, then smooth with a width-7 moving average. (Extraction: working
file `tmp/20260620_basic_curriculum_convergence.md`, scripts `tmp/_eval_curve.py`,
`tmp/_analyze2.py`.)

**Floors and cap.** The survival cap is 500 steps (recorded as 501 incl. terminal frame).
The recurring degenerate value **~101 steps** is the "eat once, then starve" floor (one
satiation cycle then death); a truly random policy lives less than that. We treat a plateau
**> ~150 steps** as "learned meaningfully above the floor".

## 4. Results

### 4.1 Per-level survival-step curve (eval, smoothed)

| Level | start (≈0.2M ep) | best sustained plateau | final (last ~6 ckpts) | converged-by (sustained ≥90% plateau) | collapse onset | learned above floor? |
|---|---|---|---|---|---|---|
| 0 static | 501 | ~448 | **~92** | 0.2M | **3.8M (collapse)** | learned, then LOST |
| 1 slow | 191 | ~361 | **~129** | 0.2M | **9.0M (collapse)** | learned, then LOST |
| 2 fast | 101 | ~418 | ~347 | **1.4M** | none | yes, stable |
| 3 rabbit | 146 | ~428 | ~338 | **1.2M** | none | yes, stable |
| 4 farsight | 116 | ~261 | ~260 | **3.6M** | none | yes, stable |

Cross-check — WandB final-summary mean survival (128-env training average, less noisy than
the 3-episode eval): L0 = 88, L1 = 118, L2 = 116, L3 = 414, L4 = 291 steps. The training
mean confirms the eval read on direction for every level (L0/L1 low, L3/L4 high), with the
usual training-vs-greedy gaps.

### 4.2 Temporal shape (the part that matters)

Smoothed survival curve sampled every ~1M episodes (steps):

- **L0 static:** 0.2M→473, 1.2M→497, 2.2M→409, 3.2M→425, **4.2M→103**, 5.2M→108, 6.2M→101, … 9.2M→90. **Cap by 0.2M, cliff at ~3.8M, flat-dead thereafter.**
- **L1 slow:** 0.2M→351, 1.2M→349, 2.2M→349, 3.2M→359, 4.2M→346, 5.2M→350, 6.2M→301, 7.2M→283, 8.2M→339, **9.2M→120.** **Stable ~350 for most of training, then a late collapse near 9M.**
- **L2 fast:** 0.2M→148, 1.2M→357, **2.2M→456**, 3.2M→408, 4.2M→345, 5.2M→415, 6.2M→347, 7.2M→425, 8.2M→418, 9.2M→411. **Rises to working level by ~1.4M, then noisy-stable around 350–450. No collapse.**
- **L3 rabbit:** 0.2M→239, 1.2M→384, 2.2M→458, 3.2M→430, 4.2M→429, 5.2M→334, 6.2M→391, 7.2M→394, 8.2M→364, 9.2M→311. **Working level by ~1.2M; noisy-stable; healthiest terminations (61% reach the max-step cap).**
- **L4 farsight:** 0.2M→106, 1.2M→98, 2.2M→231, 3.2M→202, **4.2M→282**, 5.2M→260, 6.2M→240, 7.2M→234, 8.2M→230, 9.2M→274. **Slowest learner — sits at the floor until ~1.8M, climbs, stabilizes ~260 by ~3.6M. Lowest plateau = hardest world.**

### 4.3 The collapse, concretely (Level 0)

At its peak (1.2M episodes) the Level-0 agent ran a rich survival policy — eval episode:
`Eat`×131, `Rest`×298, plus movement → survives the full 500 steps. At 10M episodes the
same checkpoint's eval episode is `Right`×91, `Eat`×9, length 101 — it walks into a wall
and starves. The policy entropy at the end is ≈ `-2.5e-5` (effectively deterministic) and
`RestCount` fell from 107 to 0. This is genuine **catastrophic policy collapse from
over-training**, corroborated across the eval recordings, the training-summary
(86% starvation terminations), and the entropy metric — it is not eval-seed noise.

## 5. Analysis

**Which levels actually learned.** All five learned a competent policy *at some point*
(every level cleared the ~150-step floor, three of them reaching the 500-step cap). **No
level failed to learn.** But two levels **did not retain** what they learned:

- **L0 static** and **L1 slow** — the *easiest* two worlds — collapsed. This is the
  classic pattern for an easy task trained far too long with no entropy floor: the policy
  over-specializes, entropy collapses, and a degenerate attractor (here: spam one action,
  starve) takes over. L0 collapsed early (~3.8M); L1 held on much longer and collapsed late
  (~9M).

**Difficulty / learning-speed ranking.** From-scratch convergence (stable levels) and final
plateau height together rank difficulty:

- **Fastest / easiest to *learn*:** L3 rabbit (~1.2M) and L2 fast (~1.4M) — both climb to
  a high (~420–450) plateau quickly. (L0/L1 reach the cap even faster, ~0.2M, but are
  unstable — speed of learning ≠ stability.)
- **Slowest / hardest:** L4 farsight — needs ~3.6M to converge and tops out lowest (~261).
  The far-sighted predator + scarce food is the genuinely hard world.
- Middle: L2/L3 are comparable; L3's terminations are the healthiest (most episodes hit the
  step cap rather than dying), consistent with the alternate rabbit food source.

**Why the easy levels are the dangerous ones for a curriculum.** The naive reading — "easy
levels converge instantly, give them tiny budgets" — is *right* on budget size but for a
subtler reason: they converge fast **and then rot if you keep training**. The budget must be
set to **stop inside the stable window**, not just "long enough to converge".

## 6. Recommendation — per-stage episode budget

**Principle.** Anchor each stage on its from-scratch converged-by episode; use the collapse
onset as a hard upper bound (never train a stage past it); add ~30–50% margin for the
converged stable levels; and **fold in curriculum carry-forward** — because the agent
carries weights forward, later stages start from a competent recurrent policy and should
need **less** than their from-scratch budget. The numbers below are *from-scratch-measured*
anchors, then trimmed for carry-forward.

| Stage | Level | From-scratch converged-by | Collapse upper bound | Recommended stage budget (episodes) | Rationale |
|---|---|---|---|---|---|
| 1 | 0 static | 0.2M | 3.8M | **1.0M** | Converges almost instantly; cap at 1M keeps a wide safety margin below the 3.8M collapse. |
| 2 | 1 slow | 0.2M | 9.0M | **1.0M** | Same — fast to learn, collapses if over-trained; 1M is safe and carry-forward helps. |
| 3 | 2 fast | 1.4M | none | **2.0M** | ~1.4M from scratch; carry-forward from stages 1–2 should cover it; 2M with margin. |
| 4 | 3 rabbit | 1.2M | none | **2.0M** | Comparable to L2; new rabbit dynamic but warm start helps; 2M with margin. |
| 5 | 4 farsight | 3.6M | none | **4.0M** | Hardest, slowest; keep close to from-scratch budget — carry-forward helps least here. |

**Total curriculum length:** ~10.0M episodes.

### Cumulative `episode_boundaries` for the schedule YAML

The boundaries are the cumulative episode counts at which the curriculum advances to the
next stage (the agent trains stage *i* until the episode count crosses boundary *i*):

```yaml
# stage budgets (episodes):  [1.0M, 1.0M, 2.0M, 2.0M, 4.0M]
episode_boundaries: [1000000, 2000000, 4000000, 6000000, 10000000]
#                    end-L0   end-L1   end-L2   end-L3   end-L4
```

**A leaner alternative** if you want to bias toward the measured fast-convergence and lean
harder on carry-forward (riskier on L4):

```yaml
# stage budgets:  [0.75M, 0.75M, 1.5M, 1.5M, 3.5M]  -> total 8.0M
episode_boundaries: [750000, 1500000, 3000000, 4500000, 8000000]
```

I recommend the **first (10M) schedule** as the default — the margin on L2–L4 is cheap
insurance and the easy stages are already short enough to dodge the collapse window.

## 7. Caveats (read before trusting the exact numbers)

- **One seed per level.** No seed dispersion is available, so converged-by episodes are
  point estimates with unknown run-to-run variance. The collapse onset, in particular, is
  likely seed-sensitive (L0 at 3.8M, L1 at 9.0M from a single seed each).
- **Low-sample eval signal.** 3 greedy eval episodes per checkpoint ⇒ noisy per-point
  values (visible min/max spread is large). Smoothing (w=7) and the cross-check against
  the 128-env training mean mitigate this, but treat converged-by figures as ±~0.5M.
- **Post-hoc framing.** No pre-registered prediction of convergence points; this is
  descriptive (Mode B).
- **Curriculum carry-forward is an assumption, not a measurement.** The trim from
  from-scratch budgets to stage budgets assumes positive transfer across stages. That has
  not been measured here — the first actual curriculum run should be instrumented to check
  it, and budgets revised if a stage starts cold.

## 8. Metrics Requested

| Subfield | Content |
|---|---|
| **Metric** | `eval/mean_episode_length` logged as a WandB scalar at each eval checkpoint (mean + std over the eval episodes), on the `Episode/Number` axis. |
| **Why now** | The full survival curve had to be reconstructed by decompressing 50×3 `.rec.gz` files per run because the local `.wandb` datastore is pruned to a few early points. A scalar logged per checkpoint would make convergence/collapse analysis a one-line read instead of a recording crawl. |
| **Where it'd live** | The RecurrentPPO eval/checkpoint hook in `src/` that already writes `recordings/<train_episode>/` and the `eval/checkpoint_episode` summary key. |
| **Cost** | Cheap (one scalar + std per checkpoint, ~50 points per run). |

| Subfield | Content |
|---|---|
| **Metric** | More eval episodes per checkpoint (e.g. 16–32 instead of 3), or the std already computed over them. |
| **Why now** | 3 episodes/checkpoint is the dominant source of noise in the converged-by estimate; more episodes would let us state convergence to ±0.1M instead of ±0.5M. |
| **Where it'd live** | The eval-rollout count parameter in the checkpoint hook. |
| **Cost** | Moderate (more eval rollouts per checkpoint; eval is cheap relative to training). |

## 9. Related Issues

- **Policy collapse on easy levels (over-training instability).** L0 and L1 lose a learned
  survival policy mid-training, entropy → ~0. If the curriculum or any standalone run is
  expected to train an easy world past ~2M episodes, an entropy floor / KL guard / early
  stopping is needed. This is a candidate `bug-fix-workflow` or `feature-workflow` item
  (entropy-coefficient floor or eval-based early stop) — surfaced here, not patched.

## 10. Links

- Curriculum design doc (the worlds, the ladder rationale): [[basic_curriculum]]
- Working file (extraction + scripts): `tmp/20260620_basic_curriculum_convergence.md`
- Run dirs: `results/JAX_RecurrentPPO/20260619-17{2721,2832,2921,3018,3115}_rppo_basic0*`
