---
title: "Grounding memo: what the offline eval recordings let us measure for predator-vs-rabbit discrimination"
topic: hypervigilance
status: active
created: 2026-06-12
last_updated: 2026-06-12
---

# Grounding memo — what we can actually measure from the recordings

## Purpose (plain-language entry point)

We want to add new behaviour measures that detect whether the agent treats a harmful
"predator" differently from a harmless "rabbit" — and we want them to work on the data
we *already record* during offline evaluation, with as little new code as possible.

The motivation is a past mistake. Our old measures (how often the agent stops eating
when a threat is near, how often it dives into cover, whether it eats less under threat,
plus mean distance and flee rate) are all **averages over the whole episode**. They
averaged away a real *conditional* behaviour: the agent discriminates predator from
rabbit **only in some situations** (specifically, only when both rabbits are already
sitting on the agent's cell, so its vision "accounts for" them and the only remaining
approaching smell must be the predator). Mixing the "rabbits accounted for" steps with
the "rabbits scattered" steps and averaging drove the effect to zero. We only caught
this by reading individual step-by-step trajectories and the agent's raw 27-number
observation vector.

So this memo answers three grounding questions for the new-measure design:

1. **What is physically recorded per step** — the raw material any new measure must be
   built from.
2. **For each family of measure the experts are likely to propose**, is it computable
   from today's recordings, does it need a new field logged, and where in the pipeline
   would it slot in?
3. **Which single cheapest measure should be implemented first**, and why.

The headline recommendation: the cheapest high-value first move is **not a new metric
family at all — it is a *conditional re-binning* of the measures we already have**, split
by whether the rabbits are currently "accounted for" in the vision count. That is the
exact axis the old aggregates collapsed, and every field it needs is already on disk.

---

## 1. Per-step recorded-field inventory

Two artifacts are written per eval run. Both come out of `scripts/eval_rollout.py`.

### 1a. The `.npz` per-episode arrays (always written)

Path: `results/eval/<run>/<ckpt>/episodes/<NNNN>.npz`. One row per step. These are the
arrays the *online-replay* M1/M2/M5 cross-check is computed from.

| Field | Shape | Meaning |
|---|---|---|
| `agent_pos` | (T, 2) | Agent grid coordinates. |
| `action` | (T,) int | Action index (Up/Right/Down/Left[/Rest][/Eat]). |
| `ate_food` | (T,) bool | Did the agent eat this step. |
| `agent_in_bush` | (T,) bool | Is the agent on a concealing bush this step. |
| `dist_per_predator` | (T, n_pred) | Manhattan distance to **each** predator (padded). |
| `dist_per_neutral` | (T, n_neut) | Manhattan distance to **each** rabbit (padded). |
| `hit_predator` | (T,) bool | Contact with a predator this step. |
| `hit_neutral` | (T,) bool | Contact with a rabbit this step. |
| `nociception` | (T,) float | Exteroceptive nociception (pain) signal. |
| `termination_reason` | scalar int | Why the episode ended. |
| `length` | scalar int | Episode length T (survival steps). |
| `seed` | scalar int | Per-episode seed. |

### 1b. The `.rec.gz` recordings (only with `--record`)

Path: `results/eval/<run>/<ckpt>/recordings/<pct>/episode_<NNNNNN>.rec.gz`, plus one
shared `run_meta.pkl`. This is the **richer** artifact and the one `trajectory-story`
reads. Per step it holds:

| Field | Content |
|---|---|
| `snapshots[t]` | Full env-state snapshot: `agent_pos`, `satiation`, `nutrition`, `injury_level`, `rest_streak`, `res_pos`, `res_active`, **`animal_pos` (all animals, unified)**, `obs_pos` (bush/obstacle positions). |
| `obs[t]` | The **full 27-dim observation the agent received** (with training noise applied). |
| `true_obs[t]` | The same observation **without noise** — the ground-truth sensory vector. |
| `actions[t]` | Action index. |
| `rewards[t]` | Step reward. |
| `run_meta.params` | `animal_classes` (predator/neutral per animal), `animal_tags`, `obs_hides_agent` (which obstacles are bushes), action map, icon config. |

**The 27-dim obs decomposes** (from `get_observation_breakdown`) into: satiation(1),
intero-nociception(1), extero-nociception(1), olfaction(5), collision(5),
proprioception(6), visual(8). The visual block is the load-bearing one: with vision range
0 it reports a **per-class COUNT of animals on the agent's own cell** — channel 5 =
predator count, channel 7 = rabbit count. This count is exactly the "accounted-for"
signal the conditional discrimination is gated on.

**What is recorded but NOT separately broken out**: animal *class* per position lives in
`run_meta.params.animal_classes`, indexed parallel to `animal_pos` — so predator vs rabbit
identity is fully recoverable, it just has to be joined from meta, not read off a per-step
field. Animal stamina / disengage state is **not** recorded per step (only its effect on
`animal_pos` is visible).

**Bottom line for measure design**: from the `.rec.gz` recordings alone we have, per step,
the agent position, every animal's position *and class*, the full distance vector to each
class, every action, the eat/bush/contact/pain flags, injury and nutrition, AND the exact
observation vector including the visual on-cell count. This is a rich enough substrate that
almost every conditional/distributional/divergence/decodability measure is computable
*offline with zero new logging* — the gating field (visual count) is already there.

---

## 2. Feasibility table — measure family → computable now?

"Computable now" = derivable from `.rec.gz` (or `.npz`) fields already on disk. "New
field" = needs something not currently logged. "Where it slots in" names the natural home.

| # | Measure family (what the experts will likely propose) | Computable now? | New log field? | Where it slots in |
|---|---|---|---|---|
| A | **Conditional / per-context binning of existing M1/M2/M5** — recompute interrupted-feeding / bush-dive / eat-under-threat *separately* for "rabbits accounted for" (visual ch7 == n_rabbits) vs "not accounted for" | **Yes** | None — visual count is in `obs[t]` | New offline analysis script (or extend `_compute_online_replay`); reads `.rec.gz` |
| B | **Full distance DISTRIBUTION (not the mean)** — histogram / quantiles of nearest-predator vs nearest-rabbit distance, optionally split by the accounted-for bin | **Yes** | None — `dist_per_*` already per-step | New offline script over `.npz` or `.rec.gz` |
| C | **Action-distribution divergence between matched near-predator vs near-rabbit states** — KL / total-variation between P(action ∣ predator at distance d) and P(action ∣ rabbit at distance d), matched on distance (and on accounted-for state) | **Yes** | None — action + per-class distance + visual count all present | New offline script; this is the cleanest "matched-context contrast" |
| D | **Behavioural d-prime (signal-detection)** — treat a defensive action (flee/bush-dive) as the "response", predator-present as "signal"; compute hit-rate vs false-alarm-rate → d′, matched on distance bin | **Yes** | None | New offline script; built on the same matched bins as C |
| E | **Occupancy / spatial measures** — fraction of time within radius r of predator vs rabbit, dwell-time distribution, approach/retreat asymmetry | **Yes** | None — positions + classes present | New offline script; partly overlaps `trajectory-story summary`'s distance bins |
| F | **Mutual information I(action ; class ∣ distance bin)** — how much the agent's action choice tells you about which class is near, holding distance fixed | **Yes** (plug-in / binned estimator; small action + small class space makes this tractable) | None | New offline script; same data join as C/D |
| G | **Decodability — train a small classifier to predict class from the agent's behaviour window** (action sequence, distance-change, bush use over a K-step window) | **Yes** (windows are constructible; `motif_window_K` windows already conceptualised in eval_rollout) | None for the *behaviour-side* features. If the decoder is meant to read the agent's *internal state* (RNN hidden), that **is a new field** — hidden state is not recorded | New offline script for behaviour-side; **needs new log field (RNN carry)** for internal-state decoding |
| H | **Pre- vs post-contact split of any of the above** — anticipation is the whole question; every measure should be computable on the pre-first-contact window only | **Yes** | None — first-contact step derivable from `hit_*` / position match (see `_first_contact`) | Filter applied inside any of the above scripts |

### The one genuine new-logging request

Only **family G's internal-state variant** needs new logging: the recurrent policy's
hidden state (`carry`) is computed every step inside `policy_fn` but thrown away. If the
experts want "is danger *decodable from the agent's brain* before contact" (the strongest
form of the anticipation claim), the RNN carry must be appended to the recorder. Captured
in §Metrics Requested below. Everything behaviour-side (A–F, H, and the behaviour-feature
form of G) is computable today.

---

## 3. Duplication / overlap with existing M1/M2/M5

The existing measures are not redundant with the proposals — but several proposals are
**cheap conditional reformulations of an existing measure**, and those are the best value:

- **Family A literally IS M1/M2/M5**, just recomputed inside the bin the aggregate
  collapsed. M5 (eat-under-threat ratio) split by accounted-for-vs-not is the single most
  direct fix for the documented failure: the post-mortem showed M5 < 1 (suppression) was
  real but only in one regime, and the whole-episode M5 washed it out. Re-binning M5 costs
  almost nothing — the per-step `ate_food`, `dist_per_*`, and the visual count are all
  already in the loop in `_compute_online_replay`; you add one `if accounted_for:` split
  of the existing accumulators.
- **Family E (occupancy) partly overlaps** `trajectory-story summary`, which already prints
  on-cell / adjacent / near / far distance bins per class. The new contribution there would
  be the *accounted-for split* and the *distribution shape*, not the bins themselves.
- **Families C, D, F, G are genuinely new** — none of M1/M2/M5 is a matched-context
  action-distribution contrast. These are the measures that would catch discrimination the
  *event-level* M-metrics can miss, because they compare behaviour at **matched distance and
  matched accounted-for state**, removing the confounds (aggression, animal count) that the
  June controls showed were driving the apparent discrimination.

The cheapest high-value add is therefore the **conditional reformulation (A)** — same
formula, one extra binning dimension — because it directly repairs the known failure and
needs zero new fields.

---

## 4. Single recommended first-implement measure

**Implement first: a matched-context action-distribution contrast, conditioned on the
accounted-for state.** Concretely — bin every pre-first-contact step by (nearest-animal
class, nearest-animal distance d, accounted-for flag), then compare the agent's action
distribution between the predator-near and rabbit-near bins at matched (d, accounted-for).
Report total-variation (or KL) distance per bin, plus the n in each bin.

Why this one first:

- **It is the direct operationalisation of the study's actual question** — "does the agent
  *act differently* toward predator vs rabbit when everything else is matched?" — at the
  level (matched distance, matched accounted-for) where the June controls proved the naive
  comparison is confounded.
- **Zero new logging.** Action, per-class distance, the visual on-cell count, and the
  first-contact step are all already in `.rec.gz`.
- **It subsumes the cheap fix.** The same (class, distance, accounted-for) binning is the
  exact scaffold family A needs; once the binning loop exists, re-binned M5/M1 and the
  d-prime / mutual-information measures (D, F) are a few extra lines over the same bins.
- **It cannot be averaged into nothing** the way the old aggregates were — the conditioning
  is built into the measure, and the per-bin n is reported so a thin bin can't masquerade as
  a signal (the exact trap the post-mortem flagged).

It would live in a **new offline analysis script** (e.g. a `contrast` subcommand alongside
`trajectory-story`'s `flee`/`obs`, or a standalone script reading the same `.rec.gz`), not
in `eval_rollout.py`'s hot loop — the recordings already contain everything, so this is
pure post-hoc analysis.

---

## Metrics Requested

| Subfield | Content |
|---|---|
| **Metric** | Per-step recurrent hidden state (`carry`) of the policy, recorded alongside `obs` in the eval recorder. Unit: float vector, length = `agent.hidden_size`. |
| **Why now** | Needed only for the *internal-state decodability* form of measure family G ("is danger decodable from the agent's representation before contact"). All behaviour-side measures (A–F, H) are computable without it; this unlocks the strongest anticipation claim. |
| **Where it'd live** | `src/utils/eval_recording.py` (`EpisodeRecorder.append` gains a `carry` arg) and the `--record` branch of `scripts/eval_rollout.py` (`_run_episode_with_recording`), where `carry` already exists in scope but is discarded. |
| **Cost** | Moderate — one hidden-size float vector per step per recorded episode (e.g. 128–256 floats × T). Only the first `--record-n-episodes` episodes carry it, so storage impact is bounded. |

This is the only proposed measure that is *not* satisfiable from existing recordings. Do
not implement it pre-emptively — it should be added only if the expert synthesis decides
the internal-state decoder is worth running.

## Related Issues

- The behaviour-side measures (families A–H minus G-internal) imply a **new offline
  analysis script**, not a code change to the training or env path. That is a
  `feature-workflow` item if the user wants it productionised, but nothing here requires
  touching `src/`, `configs/`, or the training loop.

## Links

- Study summary (incl. Appendix A behaviour-measure definitions): `docs/experiments/summaries/20260612_1625_predator_rabbit_discrimination.md`
- The methodology post-mortem this memo exists to honour: `docs/llm_wiki/entries/hypervigilance/20260609_1721_aggregate_stats_hide_conditional_behavior.md`
- Recording format: `src/utils/eval_recording.py`; eval driver: `scripts/eval_rollout.py`; trajectory microscope: `scripts/trajectory_story.py`.
