---
title: "Diagnosis — v3.0 pipeline math correctness (behavior measures / returns / neuromodulation)"
topic: diagnosis
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# v3.0 pipeline — mathematical correctness audit

## Summary (plain language)

This is a math-correctness sweep of the code that the large v3.0 update wave
touched, run in the same spirit as the config-loader bug hunt: the question is
"do the equations still compute what their names claim, and do they match the
formulas the code cites?" The good news up front: **no wrong-sign, wrong-loss,
or dimensional-mismatch bug was found.** The advantage/return math (both the
Monte-Carlo return path and the Generalized Advantage Estimation path — "GAE",
the standard actor-critic advantage estimator), the neuromodulated GRU, and the
FiLM feature-modulation formulas are all faithful to their cited forms, and the
new behavior-measure accumulators (the M1 "interrupted feeding", M2 "bush dive",
and M5 "eat-under-threat" counters wired at the five training sites) compute
what their docstrings say — including the recently-fixed divide-by-zero guard
for the eat-under-threat ratio, which is mathematically correct.

The findings are all **second-order accuracy biases**, not crashes or sign
flips. The two worth acting on: (1) the advantage estimator treats "the agent
ran out of time" (max-step truncation) identically to "the agent died" — it
throws away the value bootstrap in both cases, which biases value targets low at
every episode that ends by timeout, and in a survival task most episodes end
that way; and (2) the interrupted-feeding / bush-dive rates silently under-count
events that are still "in flight" when an episode ends, which disproportionately
drops the death-by-predator case — the single most behaviorally interesting
interrupted-feeding event. Neither is a v3.0 regression (both predate the
branch), but both distort numbers the project reads. Details, line references,
and the verified-correct list are below.

---

## Scope and what is / isn't a v3.0 change

| Surface | Files | v3.0 status |
|---|---|---|
| Behavior measures M1/M2/M5 | `src/behavior/accumulators.py`, `episode_metrics.py`, `distance_aggregator.py` | **NEW in v3.0** (added, then train.py delegated to them) |
| Return / advantage estimation | `src/models/recurrent_ppo_trainer.py` `compute_gae` / `compute_mc_returns` | logic **unchanged** vs `main` (only wrapped in a `named_scope`) |
| Neuromodulated GRU + FiLM | `recurrent_ppo_network.py`, `neuromodulator.py`, `modulated_gru_cell.py` | **unchanged** vs `main` (byte-identical) |

So the only genuinely *new* math in the audited surface is `src/behavior/`.
The return and neuromodulation math were re-verified per the task but any issue
there is pre-existing, not a v3.0 regression.

---

## Findings table (ranked)

| # | Sev | Location | Equation as written | What it should be | Failure scenario | Conf. |
|---|-----|----------|---------------------|-------------------|------------------|-------|
| 1 | 🟡 | `recurrent_ppo_trainer.py:63` + `src/environment/core.py:706` | `delta = r + γ·V(s') ·(1-done) - V(s)` with `done = terminated OR truncated` | bootstrap should be zeroed only on **true termination**; on max-step **truncation** keep `γ·V(s')` | Every episode that ends by timeout gets its value target set to the truncated Monte-Carlo tail with no bootstrap → value targets biased low; in a survival task the majority of episodes end by timeout, so the bias is systematic, not rare | High |
| 2 | 🟡 | `accumulators.py:241` vs `:229` (M1); `:263` vs `:292` (M2) | per-class denominator incremented at **event-record** time; per-tag denominator at **resolution** (age==K) time | both counted at the same instant | Per-class `InterruptedFeedingRate` / `BushDiveRate` and their per-tag variants are computed over *different* candidate populations; they diverge whenever an event is overwritten within K steps or is still pending at episode end | High |
| 3 | 🟡 | `accumulators.py:126-144` (`bm_reset_env`) | pending candidate/onset (age < K) is zeroed at episode end; its per-class denominator was already incremented at record time | resolve pending events at episode boundary (or exclude from denominator) | Per-class interrupted-feeding & bush-dive **rates biased downward**: the denominator counts the last-K-steps events but the numerator can never fire for them. This preferentially drops death-by-predator interruptions — the most behaviorally salient case | High |
| 4 | 🟢 | `neuromodulator.py:171` | `temperature = clip(softplus(z)+0.5, temp_clip[0]=0.1, temp_clip[1])` | — | `softplus(z)+0.5 ≥ 0.5 > 0.1`, so the lower clip bound is **dead code**; effective floor is 0.5. Harmless unless a config sets `temp_clip[0] > 0.5` expecting it to bind | High |
| 5 | 🟢 | `recurrent_ppo_trainer.py:138-139` | comment: "clipped for GAE"; code: plain `0.5·mean((V−target)²)` for both MC and GAE | comment should match (no value-clipping is implemented) | Documentation only — the value loss is a correct un-clipped MSE; no numeric effect | High |
| 6 | 🟡 | `recurrent_ppo_network.py:164, 168, 171` | Phase-1 "unimodal" gain/bias applied as `γ[..., None, :] * encoded_all` — one hidden-dim vector broadcast across **all** modality groups | if per-modality modulation is intended, head must output `[modality, hidden]` | The "unimodal" head applies an **identical** gain to every modality, i.e. it is not per-modality. May be intended (target-dim is `hidden`), but the naming implies otherwise — confirm against the modulation design doc | Med |

Adjacent (out of this surface, flagged for `env-config-auditor`): `core.py:722/725`
applies `death_penalty` on `done`, and `done` includes truncation — so surviving
to `max_steps` incurs the death penalty. This is reward shaping, not return/BM
math, but it compounds finding #1.

---

## Sub-surfaces verified CORRECT

**Return / advantage estimation** (`recurrent_ppo_trainer.py`)
- GAE reverse `lax.scan`: `δ_t = r_t + γ V(s_{t+1})(1-d_t) - V(s_t)`,
  `A_t = δ_t + γλ(1-d_t) A_{t+1}` — matches Schulman et al. (2016) Eq. 11-12.
- `values_with_next[1:]` alignment: interior `next_value[t] = V(s_{t+1})`,
  terminal `next_value[T-1] = final_v` bootstrap — correct off-by-one handling.
- MC return (`compute_mc_returns`): reverse scan with `ret = where(done, 0, ret)`
  **before** `ret = r + γ·ret` correctly severs episodes with **no cross-episode
  leakage** (verified by tracing a two-episode boundary).
- Advantage normalization: batch mean/std with `+1e-8` (GAE) / `+1e-7` (MC) — standard.
- Targets: `targets = A + V` (GAE), `targets = returns` (MC) — correct.
- PPO clipped surrogate, entropy `-Σ p·log p`, value `0.5·MSE` — all correct.
- Hidden-state reset on `done` in **both** `collect_trajectories` and the loss
  `scan_fn` (PyTree `_h_reset_on_done`) — prevents cross-episode BPTT leakage; correct.

**Behavior measures M5** (`accumulators.py`)
- `EatUnderThreatRatio` NaN guard `if threat_steps>0 and eat_safe>0` is
  mathematically sound: `eat_safe>0 ⇒ safe_steps>0`, so both `p_eat_threat` and
  the divisor `p_eat_safe` are well-defined and non-zero — no false NaN, no
  div-by-zero. Matches the "EatUnderThreatRatio NaN when eat_safe=0" fix commit.
- Threat/safe/eat step counters use consistent `< bm_R` radius test and correct
  `~under_threat` complement; per-class and per-tag use identical logic.

**Episode metrics** (`episode_metrics.py`)
- Reward sum/min/max, damage sums, termination one-hot (codes 1-4), and the
  cumulative `episode_counter` (correctly **not** reset in `episode_reset_env`)
  are all consistent with the documented JAX→WandB key mapping.

**Distance aggregator** (`distance_aggregator.py`)
- Per-episode mean = `sum / max(step_count, 1)` — correct running mean with a
  sound zero-step guard; per-tag slicing bounds-checked against `num_predator/neutral`.

**Neuromodulated GRU + FiLM** (unchanged vs `main`, re-verified)
- FiLM: `relu(γ⊙x + β)` — matches Perez et al. (2018) FiLM(x)=γ⊙x+β. Pass-through
  init is correct: γ head `bias_init=1.0`, β head `bias_init=0.0`, baselines zero
  ⇒ γ≈1, β≈0 at start (identity), avoiding the v8 "γ collapses to identity"
  *failure* by instead *starting* at identity and learning away — the intended design.
- PreActivation: `relu(σ(z)·x + β)` and Multiplicative: `relu(x)·σ(z)` match the
  neuromodulator docstring (Ferguson & Cardin / Ben-Iwhiwhu styles).
- `ModulatedGRUCell`: standard GRU (`r,u,n,h`) with `gate_bias` added to the
  update-gate pre-activation — identical to `nnx.GRUCell` when `gate_bias=0`; the
  injection point matches NEUROMODULATION_ALGORITHM.md §5.1a.
- Dimensional consistency: `repeat(raw, G)[..., :target]` gives exact target dim
  for any `grouping_size`; `z_memory` (hidden-dim) matches the GRU gate width;
  temperature is a scalar dividing logits (`logits/temp`), correct direction
  (higher temp → flatter policy). Symlog obs compression `sign(x)·log(|x|+1)` correct.

---

## Derivation note — finding #1 (truncation bootstrap)

For a time-limit truncation at step `T` the trajectory is cut, not terminal.
The correct target is `G_t = r_t + γ V(s_{t+1})` (bootstrap retained), because
the environment would have continued. The code computes, at the truncated step,
`δ = r + γ V(s') (1-done) - V = r - V` (bootstrap dropped, since `done=1`), so
`target = A + V = r`. The value head is thereby trained to regress the *single
truncated reward* instead of the discounted continuation — an underestimate of
`γ V(s')`. The standard fix (e.g. SB3, CleanRL) is to carry a separate
`terminated` mask (true death only) into the `(1-·)` bootstrap factor while the
`done` mask still drives episode/hidden resets. `termination_reason` already
distinguishes code `1` (max_steps) from `2/3/4` (death), so the signal exists.

---

## Verdict

No sign, loss-form, or dimensional errors; the audited math is faithful to its
cited formulations. Two accuracy biases are worth fixing before drawing
quantitative conclusions — truncation-vs-termination in the advantage bootstrap
(#1) and episode-end under-counting of interrupted-feeding / bush-dive events
(#2, #3) — but none block training.

Reviewed by: math-reviewer
