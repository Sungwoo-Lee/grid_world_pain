---
title: "Discrimination measures: what the eval recordings can actually support (codebase grounding)"
topic: hypervigilance
status: active
created: 2026-06-12
last_updated: 2026-06-12
---

## Purpose (plain-language entry point)

We want to ADD new ways of measuring whether the trained agent treats a harmful
"predator" differently from a harmless "rabbit" — keeping the existing measures,
but catching the kind of **conditional** behaviour the old ones missed. The old
measures (how often the agent stops eating near a threat, how often it dives into
a bush, whether it eats more or less when a threat is near, plus average distance
and flee rate) are all **averages**. On 2026-06-09 we learned the hard way that an
average can hide a real behaviour: the agent only discriminated **when both rabbits
were sitting on its own cell** (its vision channel "counted" them), and mixing that
situation with all the others washed the effect to zero
([memory: aggregate stats hid a conditional behaviour](../../../memory/memories/hypervigilance/20260609_1721_aggregate_stats_hide_conditional_behavior.md)).

Two professors are proposing the *concepts* for new measures (action-distribution
divergence, occupancy, signal-detection d-prime, mutual information, decoders).
**This memo is the grounding layer**: it says, for each measure family, exactly which
raw fields the eval pipeline already records, which measures are computable from
those fields **today**, which need one extra thing logged, and where each would slot
in. It ends with the single cheapest "matched-context contrast" I'd build first.

The one fact that governs everything below: the offline pipeline writes **two
different files per eval**, and they do **not** carry the same fields.

- The **always-on per-episode dump** (`episodes/NNNN.npz`, written for *every* eval
  episode) carries event flags — did the agent eat, get hit, sit in a bush — and
  per-class distances, but **not the raw 27-number observation vector**.
- The **opt-in recording** (`recordings/.../episode_NNNNNN.rec.gz`, written only for
  the first N episodes when you pass `--record`) carries full positions plus the
  **raw observation vector** — including the "vision count" that was the hidden
  gating variable — but **not** the eat/hit/in-bush event flags directly.

So the variable that hid the conditional effect (the vision count) lives **only** in
the opt-in recording, and only for a handful of episodes. That single split is the
most important thing for the professors to know, and it shapes the cheapest fix.

---

## 1. Per-step recorded-field inventory

### 1a. `episodes/NNNN.npz` — written for EVERY eval episode

Source: `scripts/eval_rollout.py::_run_episode` / `_run_episode_with_recording`.
Arrays are length-T (one row per step) unless noted.

| Field | Shape | Meaning |
|---|---|---|
| `agent_pos` | (T, 2) | Agent grid cell each step. |
| `action` | (T,) int | Action index taken (argmax — deterministic eval). |
| `ate_food` | (T,) bool | Did an eat succeed this step. |
| `agent_in_bush` | (T,) bool | Was the agent inside a concealing bush. |
| `dist_per_predator` | (T, n_pred) | Manhattan distance to each predator (padded). |
| `dist_per_neutral` | (T, n_neut) | Manhattan distance to each rabbit (padded). |
| `hit_predator` | (T,) bool | Contact with a predator this step. |
| `hit_neutral` | (T,) bool | Contact with a rabbit this step. |
| `nociception` | (T,) float | Exteroceptive pain signal this step. |
| `termination_reason` | scalar int | Why the episode ended. |
| `length` | scalar int | T. |
| `seed` | scalar int | Episode seed. |

Not present here: the raw observation vector, internal/proprio state (hunger,
injury, nutrition as continuous values), the policy's action probabilities, the RNN
hidden state.

### 1b. `recordings/.../episode_NNNNNN.rec.gz` — written only with `--record`, first N episodes

Source: `src/utils/eval_recording.py::EpisodeRecorder`. Per-step `snapshots[t]` dict
plus stacked arrays.

| Field | Shape | Meaning |
|---|---|---|
| `snapshots[t]['agent_pos']` | (2,) | Agent cell. |
| `snapshots[t]['satiation']` | float | Hunger/fullness state. |
| `snapshots[t]['nutrition']` | float | Nutrition store. |
| `snapshots[t]['injury_level']` | float | Accumulated injury (0–100; ~100 = death). |
| `snapshots[t]['rest_streak']` | int | Consecutive rest steps. |
| `snapshots[t]['res_pos']` / `res_active` | (n_res,2)/(n_res,) | Food positions + alive flags. |
| `snapshots[t]['animal_pos']` | (n_animals, 2) | **Every animal's position** (class via `run_meta`). |
| `snapshots[t]['obs_pos']` | (n_obs, 2) | Bush/obstacle positions. |
| `obs` | (T, 27) | **Raw observation the agent received** (noisy). |
| `true_obs` | (T, 27) | Same, noise removed. |
| `actions` | (T,) int | Action index (incl. initial -1). |
| `rewards` | (T,) float | Per-step reward. |

`run_meta.pkl` (one per recorded run) carries `params` — which gives
`animal_classes`, `animal_tags`, `animal_visual_channel` (predator→5, rabbit→7),
and `obs_hides_agent` (which obstacle indices are concealing bushes).

The 27-dim `obs` block order (from `get_observation_breakdown`): satiation(1),
intero-nociception(1), extero-nociception(1), olfaction(5), collision(5),
proprioception(6), **visual(8)**. The visual block is a **per-class COUNT of
on-cell animals**: `obs[vis0+5]` = number of predators on the agent's cell,
`obs[vis0+7]` = number of rabbits on the agent's cell. **That `obs[vis0+7]` count
is the gating variable from the 2026-06-09 lesson** — and it exists only here, in
the `.rec.gz`.

### 1c. The coverage asymmetry that matters

`--record-n-episodes` defaults to 10; `--eval-n-episodes` is often 200. So today the
gating variable (vision count) and the raw obs are available for ~10 episodes, while
the event flags + distances are available for all ~200. Any **regime-conditional**
measure that needs the vision count is currently limited to the recorded subset
unless we either record all episodes or copy the two count scalars into the `.npz`.

---

## 2. Feasibility table (measure family → computable now? / new field? / where it slots in)

"Now (rec.gz)" = computable today but only over the recorded subset. "Now (npz)" =
computable today over all eval episodes. "Now (both)" = inputs exist in both.

| Measure family (likely proposer) | Computable now? | Needs a new logged field? | Where it slots in |
|---|---|---|---|
| **Full distance DISTRIBUTION / quantiles** (replace mean dist + flee-rate) | **Now (both)** — `dist_per_*` in `.npz`; positions in `.rec.gz` | No | New offline analysis script over `episodes/*.npz`; trivial histogram/ECDF. |
| **Per-context (conditional) binned M1/M2/M5** — e.g. eat-under-threat split by whether rabbits are "accounted for" in the vision count | **Now (rec.gz only)** — needs eat/in-bush AND the vision count; eat/in-bush derivable from `obs`/positions/satiation in `.rec.gz`, vision count is `obs[vis0+7]` | **Cheap field would help**: copy `obs[vis0+5]`, `obs[vis0+7]` per step into the `.npz` records dict so it runs over ALL episodes, not just the recorded 10 | Gate added inside the `_compute_online_replay` loop (the M5 branch), OR a new offline script over `.rec.gz`. |
| **Distance-matched action-distribution divergence** (JS/KL between action histogram near-predator vs near-rabbit, same distance bin) — *empirical* version | **Now (npz)** — `action` + `dist_per_*` per step | No (for empirical histograms) | New offline script; reuses the distance-binning logic already in `trajectory_story.py::cmd_flee`. |
| **Action-distribution divergence at the policy level** — divergence of π(a\|s), the policy's *probability* vector, at matched states | **No** — deterministic eval records only the argmax action; no probability vector | **Yes**: log per-step action **logits / probs**, or run a stochastic eval pass | `policy_fn` in `eval_rollout.py` already gets `_log_prob`/`_value` from `get_action_and_value_nnx`; would record the full logits. |
| **Behavioural d-prime** (signal-detection discriminability: predator = signal, rabbit = noise; response = flee / bush-dive / eat-suppress) | **Now (npz)** — built from `in_bush`/`action`/`ate_food` + per-class distance | No | New offline script; it is a re-aggregation of M2/M5-style events into a discriminability index. |
| **Mutual information** between "near-animal class" and a behavioural feature (action, bush-dive, eat) | **Now (npz)** — discrete histograms of `action`/`ate_food`/`in_bush` vs class-near label | No | New offline script; discrete plug-in MI estimator. |
| **Decodability from observable BEHAVIOUR** (classifier predicts "predator nearby" from action/position/internal-state window) | **Now (both)** — features from `action`+positions (`.npz`), richer with `satiation`/`injury` (`.rec.gz`) | No | New offline script; small logistic/MLP probe. |
| **Decodability from the agent's INTERNAL representation** (probe the RNN hidden state for class) | **No** — the recurrent hidden state (`carry`/`h_new`) is computed but never saved | **Yes**: log the per-step RNN hidden state | `policy_fn` returns `h_new` each step; record it alongside `obs`. Moderate cost (per-step hidden-size vector). |
| **Occupancy / spatial maps** (where the agent dwells relative to each animal) | **Now (both)** — agent + animal positions | No | New offline script; 2-D occupancy histogram. |

---

## 3. Duplication with existing M1/M2/M5 — and the cheap reformulation

The proposed families overlap heavily with the existing event measures; the
high-value moves are **reformulations**, not brand-new instruments:

- **M5 (eat-under-threat ratio)** already splits eat-rate by "threat in radius" vs
  "safe", per class. A *conditional* M5 = add **one more conditioning axis**: the
  vision-count regime (rabbits accounted-for vs not). This is the cheapest
  high-value add because it (a) reuses M5's exact machinery in
  `_compute_online_replay`, (b) targets the precise variable that hid the effect on
  2026-06-09, and (c) needs only the two count scalars logged into the `.npz` to run
  at full episode coverage.
- **M2 (bush-dive rate)** is already per-class. A "behavioural d-prime" or
  "MI(class; bush-dive)" built on bush-dive is **M2-predator vs M2-rabbit repackaged
  as a discriminability index** — same inputs, different summary. Worth having as a
  single bounded number, but it is not new data.
- **M1 (interrupted-feeding)** likewise is per-class; any divergence/d-prime over it
  is a re-aggregation.
- **Mean distance + flee-rate** are exactly what the "full distribution" proposal
  **supersedes** — keep the mean as a scalar but report the distribution.

The reformulation principle: the old measures collapse a per-class event into one
average. The new value comes from (1) **not averaging** (distribution instead of
mean) and (2) **adding the gating axis** (split by the vision-count regime) — both of
which are reformulations of measures we already compute.

---

## 4. The single cheapest "matched-context contrast" to implement FIRST

**Regime-split, distance-matched action-distribution divergence.**

Concretely: for each distance bin `d` to the nearest animal, compute the agent's
empirical action histogram in "**nearest animal is a predator, at distance d**" states
versus "**nearest animal is a rabbit, at distance d**" states, and report the
Jensen-Shannon divergence between them — computed **separately** within the two
regimes (rabbits accounted-for in the vision count vs not).

Why this one first:

1. **Zero new logging for the first cut.** Every input — agent/animal positions →
   distance, action index, and the vision count `obs[vis0+7]` — already lives in the
   `.rec.gz`. It runs today over the recorded episodes.
2. **It reuses code we already have.** The distance-binning loop in
   `trajectory_story.py::cmd_flee` is 90% of the scaffolding; this swaps "flee
   magnitude" for "action histogram + JS divergence" and adds the regime split.
3. **It is genuinely matched.** Holding the distance bin fixed removes the
   aggression/number confound that the summary showed was driving the old distance
   gap; comparing histograms (not means) keeps conditional structure.
4. **The regime split is exactly the variable that averaged the 2026-06-09 effect to
   zero.** A single JS number per `(distance, regime)` cell makes "does the agent act
   differently toward predator vs rabbit, holding context fixed?" a directly readable
   discriminability value — and will reveal the effect in the accounted-for regime
   even if it is invisible in the pooled average.

The natural **second** step, once this confirms the gating, is the cheap `.npz`
field (log `obs[vis0+5]`/`obs[vis0+7]` per step) so the same regime-split runs over
all ~200 eval episodes rather than the recorded ~10 — see Metrics Requested.

---

## 5. Metrics Requested

| Subfield | Content |
|---|---|
| **Metric** | Per-step on-cell visual counts `vis_count_predator` (`obs[vis0+5]`) and `vis_count_neutral` (`obs[vis0+7]`), unitless counts. |
| **Why now** | The vision count is the gating variable for every regime-conditional measure, but it lives only in the `.rec.gz` (≈10 episodes). Copying it into the `.npz` lets the conditional M5 / action-divergence run over all ~200 eval episodes. |
| **Where it'd live** | `scripts/eval_rollout.py` — add two entries to the `records` dict in `_run_episode` / `_run_episode_with_recording`, populated from the `obs` already computed by `policy_fn`. |
| **Cost** | Cheap (two scalars per step). |

| Subfield | Content |
|---|---|
| **Metric** | Per-step policy action distribution `action_probs` (length = action_dim) and/or the RNN hidden state `rnn_hidden` (length = hidden_size). |
| **Why now** | Needed for *policy-level* action-distribution divergence (not just empirical histograms) and for *internal-representation* decodability — two professor proposals that are otherwise not computable. |
| **Where it'd live** | `scripts/eval_rollout.py::policy_fn` — `get_action_and_value_nnx` already returns log-prob/value and `h_new`; record them. |
| **Cost** | Moderate (per-step vector; `action_probs` cheap, `rnn_hidden` larger). |

> Note: `action_probs` is also confounded by deterministic eval — under best-action
> rollout the recorded action is the argmax, so a clean π(a\|s) comparison wants either
> logged logits or a stochastic eval pass. Flag for the experiment-designer.

## 6. Related notes

- This is a **grounding memo**, not an experiment analysis — no runs were analyzed.
  It feeds the measure-toolkit design that `experiment-designer` will assemble from
  the three expert inputs (professor-rl, professor-bayesian-brain, this memo).
- Methodology anchor: [memory — aggregate stats hid a conditional behaviour](../../../memory/memories/hypervigilance/20260609_1721_aggregate_stats_hide_conditional_behavior.md).
- Study context + existing measure definitions: [predator-vs-rabbit discrimination summary, Appendix A](../../summaries/20260612_1625_predator_rabbit_discrimination.md).
- Existing trajectory tooling to reuse: `scripts/trajectory_story.py` (distance
  binning, obs decode) and `scripts/eval_rollout.py` (`_compute_online_replay` for
  M1/M2/M5).
