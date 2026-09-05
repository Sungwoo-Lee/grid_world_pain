# GAE_NORM — pre-registered prediction (fourth arm of the return-mode comparison)

- **created:** 2026-09-04
- **status:** launched, predictions fixed BEFORE any result was seen
- **parent analysis:** [[return_mode_cmp_1M]]

## Question

Recurrent PPO learns two numbers from every batch of experience: a **critic target** (what
the value head *should* have predicted for the rest of the episode) and an **advantage**
(how much better an action turned out than the critic expected). Two independent choices
produce that pair — **which estimator** builds it, and **how the two numbers are rescaled**
before training. Those choices form a 2×2, and only three cells had ever been run:

| | **matched scale** (target rescaled to spread ≈ 1; advantage = target − critic, left alone) | **split scale** (raw target, spread ≈ 24; advantage separately rescaled to spread ≈ 1) |
|---|---|---|
| **Monte-Carlo estimator** | `MC` — **138.8** survival steps | `MC_FIXED` — **40.0** |
| **GAE estimator** | `GAE_NORM` — **this run** | `GAE` — **41.9** |

Those numbers are five-seed means on the 10×10 jump-attack world (`basic/04`), one million
episodes per run. The two arms that follow the textbook convention are ~3.5× worse than the
one that does not, and never once reach the 500-step episode cap.

The 1M analysis established that the *estimator* cannot be what separates them: `MC` and
`MC_FIXED` consume **byte-identical returns from identical code** and differ by 98 steps,
while `MC_FIXED` and `GAE` use **different estimators under the same scale convention** and
differ by 1.7. **Scale predicted the outcome; estimator did not.** This arm fills the empty
cell, which is the only way to tell those two explanations apart.

## Pre-registered predictions

Stated before any `GAE_NORM` number exists. The decision metric is **mean survival steps**
over the final evaluation window, five seeds.

- **If GAE_NORM lands near 139** (say, above ~100) → **scale-matching is the mechanism.**
  Keeping the critic target and the advantage in the same units is what rescues learning,
  and the estimator choice is close to irrelevant on this task. The project's real defect is
  then a **missing upstream reward normaliser** (VecNormalize / PopArt), which all nine
  surveyed mainstream PPO libraries have and this codebase lacks — `MC`'s z-scoring has been
  doing that job by accident.
- **If GAE_NORM lands near 42** (say, below ~60) → **the estimator is the mechanism**, the
  scale story is wrong, and something specific to the Monte-Carlo return — not its
  normalisation — is carrying `MC`.
- **Anything between ~60 and ~100** → neither explanation is sufficient alone; both
  contribute, and the comparison does not resolve which dominates.

Secondary, already-measured quantities that should move **with** the survival result if the
scale account is right: gradient norm falling below the 0.5 clipping ceiling (`MC` 0.36–0.48
and never clipped; `MC_FIXED` 125–154 and clipped every step), the policy term's share of the
total loss rising from ~0.002 % toward ~2 %, and the critic's explained variance rising from
~9 % toward ~53 %.

## Known confound, recorded in advance

`GAE_NORM` mirrors `MC` faithfully, and that **includes inheriting MC's units quirk**: the
GAE recursion mixes raw rewards with a bootstrap value `V` that, in this mode, is trained on
z-scored targets — so the bootstrap enters in normalised units while the rewards are raw
(same class as the open Known Bugs row on the H4 window-edge fix). Mirroring `MC` exactly was
the point, but it means that **if `GAE_NORM` behaves like `MC`, "matched scale" and "carries
MC's units quirk" are confounded** and a further arm would be needed to separate them.

## Manifest

- Config: `configs/models/recurrent_ppo/recurrent_ppo_cmp_gaenorm.yaml` (differs from
  `recurrent_ppo.yaml` in exactly one key, `agent.return_mode`; neuromodulation off)
- Seeds 42–46, 10M episodes, WandB group `return_mode_cmp_10m`
- Nodes 110:0, 110:1, 111:0, 111:1, 112:0

---

## Outcome (added 2026-09-05, after the runs finished)

*Revised 2026-09-05 after an adversarial review of the analysis
([[plan_return_mode_cmp_10M]]) and a ten-times-larger re-evaluation; the first version of this
section reported the greedy-evaluation tie as a real disagreement.*

**All four predictions above were correct.** `GAE_NORM` reached **170.4 ± 0.5** survival steps at
10M episodes (threshold was "above ~100"), its gradient norm stayed at 0.17-0.39 and was never
clipped, its policy term took 1.0-1.4% of the loss budget, and its critic explained **58.3%** of
its target's variance. Scale-matching is supported as the mechanism.

*A note on the decision metric.* This pre-registration named "mean survival over the final
**evaluation** window". No evaluation-time survival metric is logged by the trainer, so the
scoring above uses the final window of the **training** survival series, with an offline greedy
evaluation reported alongside it. The substitution does not change any verdict here — the
threshold was ~100 and the observed value ~170 on both — but it should be recorded.

Two things the pre-registration did not anticipate. First, `GAE_NORM` finished **above** `MC`
by about 5 survival steps, with all five seeds above all five during training (exact *p* = 0.0079)
**and the same gap under greedy-policy evaluation once that evaluation was large enough to see
it**: +4.47 steps, 95% CI [+0.75, +8.19], at 2,000 episodes per seed. (A first 200-episode
evaluation read a tie at 177.55 vs 177.59, but its margin of error was ±6.4 steps — larger than
the effect — so it never bore on the question. The "it is only a cheaper exploration bill"
explanation is dead: the greedy evaluation takes no exploratory actions in either arm and the gap
persists. Why `GAE_NORM` is better is not established.) Second, the recorded confound is **worse**
than assumed: `GAE_NORM` applies the raw-reward/z-scored-bootstrap units mismatch at *every step*,
where `MC` applies it only at rollout-window edges. Full analysis: [[return_mode_cmp_10M]].
