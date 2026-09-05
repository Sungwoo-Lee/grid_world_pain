# MC_RAW — pre-registered prediction (fifth arm: matched scale, but matched at the LARGE scale)

- **created:** 2026-09-04
- **status:** predictions fixed BEFORE the arm was run
- **siblings:** [[gae_norm_prereg]] · [[return_mode_cmp_1M]]

## The question, in plain language

Recurrent PPO learns two numbers per batch: a **critic target** (what the value head should have
predicted for the rest of the episode) and an **advantage** (how much better an action turned out than
the critic expected). Four arms have now been run, and three of them agree that something about
*scale* — not about which estimator builds the numbers — decides whether the agent learns to forage at
all.

But "matched scale" has been doing two jobs at once in that story, and nothing run so far separates
them:

- **The relational reading.** What matters is that the critic target and the advantage are expressed in
  the *same units*, so the two halves of the network are learning about the same quantity.
- **The absolute reading.** What matters is that the advantage lands near a **spread of 1**, which is
  what keeps the policy's gradient well-conditioned against a fixed gradient-clipping ceiling and a
  fixed learning rate.

Every arm run so far has these two properties glued together. `MC` and `GAE_NORM` are matched **and**
near 1. `MC_FIXED` and `GAE` are neither. `MC_RAW` breaks the glue: it is **matched, but at the large
scale**.

| arm | critic target | advantage | same units? | that scale | result |
|---|---|---|---|---|---|
| `MC`, `GAE_NORM` | z-scored, spread ≈1 | the residual `target − V` | yes | ≈1 | 138.8 / tracking it |
| `MC_FIXED`, `GAE` | raw, spread ≈24 | separately z-scored to ≈1 | no | split | 40.0 / 41.9 |
| **`MC_RAW`** | raw, ≈24 | **raw residual, ≈20** | **yes** | **≈24** | this run |

It is the `MC_FIXED` branch with exactly one line deleted — the advantage normalisation — so the
contrast against `MC_FIXED` is a single-variable one.

## Pre-registered predictions

Decision metric: **mean survival steps**, five seeds, judged at matched *environment steps* (not
episodes — the arms consume experience at very different rates).

- **Near 139 (say above ~100) ⟹ the relational reading is right.** Sharing units is what matters, and
  the absolute magnitude is close to irrelevant; the field's convention of pinning advantages to spread
  1 would then be solving a problem this environment does not have.
- **Near 42 (say below ~60) ⟹ the relational reading is wrong.** Sharing units is not sufficient, and
  what `MC` and `GAE_NORM` were really buying is *absolute* conditioning — the critic able to fit a
  target of spread ≈1, and the policy stepping on an advantage of spread ≈1.
- **Between ~60 and ~100 ⟹ both contribute** and this design cannot rank them.

**My own prediction, recorded so it can be wrong: closer to `MC_FIXED` than to `MC`.** The reasoning:
the critic's target is still raw, and the raw-target arms' critics never fit it (≈9% of variance
explained, against `MC`'s ≈53%). Un-normalising the advantage multiplies the policy's contribution to
the summed gradient by roughly 20×, which lifts its share from about 0.3% to about 6% — better, but
still a network shaped overwhelmingly by value regression. If that is right, `MC_RAW` should land
*above* `MC_FIXED` but well short of `MC`, which is the "between" band and the least informative
outcome. I would rather say so in advance than claim afterwards that I expected whatever happens.

## A second thing this arm tests, which the literature cannot answer

The mechanism in that prediction — that in a shared trunk the value loss dominates the combined
gradient, leaving the actor to read out features shaped only by value regression — is, per the
[[ppo_implementation_details_lit_review]], **measured by nobody**. It appears once, as a one-sentence
conjecture in a practitioner's blog post, and the same literature contains a counterweight: the
reference PPO implementation deliberately *shares* a convolutional trunk on Atari, and the only
recurrent PPO in that corpus is built on it.

So `MC_RAW` is not only a control for our own tuning question. Together with the per-arm gradient-norm
and loss-share logs already being recorded, it is a direct test of a mechanism this project has been
invoking on no evidence.

## Manifest

- Config: `configs/models/recurrent_ppo/recurrent_ppo_cmp_mcraw.yaml` — differs from
  `recurrent_ppo_cmp_mcfixed.yaml` in exactly one key, `agent.return_mode`; neuromodulation off
- Environment `basic/04`, seeds 42–46, 10M episodes, WandB group `return_mode_cmp_10m`
- Nodes 105:0, 106:1, 109:1, 112:1, 102:0 — single free GPUs on otherwise-occupied nodes, chosen to
  leave nodes 113 and 114 whole for heavier jobs

---

## Outcome (added 2026-09-05, after the runs finished)

*Revised 2026-09-05 after an adversarial review of the analysis
([[plan_return_mode_cmp_10M]]); the first version of this section scored a post-hoc criterion
before the pre-registered one and stated two hypotheses as results.*

**The pre-registered decision rule returns "cannot rank".** At the pre-registered decision point
(matched environment steps) `MC_RAW` reached **70.4 ± 39.4** against `MC_FIXED` **69.3** and `MC`
**156.8**. The banding written in advance was: above ~100 ⟹ relational reading right; below ~60 ⟹
relational reading wrong; **between ~60 and ~100 ⟹ both contribute, this design cannot rank
them**. 70.4 is in the middle band. **That is the outcome of record.**

**The recorded personal prediction — "closer to `MC_FIXED` than to `MC`" — was correct** (a
distance of 1.1 against 86.4).

**The banding rule could not do its job, because the arm is bimodal**: two seeds at 108-119 and
three at 35-45, nothing in between. The mean of 70.4 describes no seed. That is a limitation of
the rule, not a result.

**Post-hoc, and labelled as such.** The *strong* relational reading — sharing units is what
matters whatever the magnitude, so `MC_RAW` should behave like `MC` — **is refuted**, on a fact
that needs no test: every one of `MC_RAW`'s five seeds finished below `MC`'s worst seed at every
budget checked. The **absolute** reading (the advantage must land near spread 1) is correspondingly
supported. But the *weaker* claim that matching units at scale 24 buys nothing **relative to the
split convention** is **not shown**: that comparison's *p* = 0.95 has a minimal detectable
difference of about 50 survival steps, and rests on a mean the analysis elsewhere says should
never be quoted for this arm.

**The prediction was right for the wrong reason, and the mechanism this arm was built to test
turned out to be untestable from these logs.** The policy's share of the loss budget rose 20x as
expected (0.0017% → 0.0348%, though ~170x lower in absolute terms than the pre-registration
guessed) with no measurable performance change — but **loss shares cannot test the shared-trunk
mechanism at all**, because the policy surrogate's *value* is ≈ 0 by construction for mean-zero
advantages however large its *gradient* is. The mechanism remains **unmeasured in both
directions** until per-term gradient norms are logged. Separately, the ratio that moved by two
orders of magnitude was policy-versus-*entropy* (0.34 → 42.25, ~120x) and `MC_RAW`'s policy
entropy fell to 0.095 nats; **"it failed through loss of exploration" is the leading hypothesis,
not a finding** — it is one arm, five seeds, observational, and the reverse direction (a policy
that has settled into resting becomes deterministic) is not excluded. The test is `MC_RAW` re-run
with the entropy coefficient raised ~20x. Full analysis: [[return_mode_cmp_10M]].
