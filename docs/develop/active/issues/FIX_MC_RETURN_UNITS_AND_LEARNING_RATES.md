---
title: "Fix: MC return units + wiring the critic and modulator learning rates"
topic: issues
status: active
created: 2026-09-01
last_updated: 2026-09-01
---

# Fix: MC return units + wiring the critic and modulator learning rates

> ## ⛔ Fix 1 WAS RUN AND REFUTED — DO NOT APPLY IT (added 2026-09-07)
>
> **Fix 1 of this plan — replace the MC branch's z-scored critic target with a raw
> target plus normalised advantages — was implemented as the separate `MC_FIXED`
> return mode, run at scale, and lost decisively to the code it was meant to
> replace.** It is not a latent improvement waiting to be applied; it is a measured
> regression.
>
> Evidence: 25 runs, 5 arms × 5 seeds, `basic/04`, 10M episodes
> ([[return_mode_cmp_10M]]). At matched environment steps the z-scored-target arms
> reached ~156–160 mean survival steps while the raw-target arms reached ~67–98 —
> roughly an **order of magnitude more experience** to reach any given survival
> level, and far worse seed-to-seed reproducibility (spread 22–81 steps against 2–5).
> The convention this plan proposed is the one nine mainstream PPO libraries use,
> but all nine also carry an upstream **reward normaliser** that this codebase does
> not have; MC's return-rescaling has been standing in for it, and removing the
> stand-in without installing the real thing is what `MC_FIXED` measured.
>
> **Consequence for anyone editing the trainer:** do not change the semantics of the
> `MC` branch. `MC_FIXED`, `GAE_NORM` and `MC_RAW` already exist as separate modes,
> so any further comparison is a config change, not a code change. Live experiments
> depend on `MC` continuing to mean what it means today — see
> [[NMN_INPUT_SITE_GRID]], whose entire budget is sized on it.
>
> **Fix 2 (wiring `lr_critic` and `lr_modulator`) is unaffected and still stands.**
> `lr_critic` is read by nothing today, so actor and critic already share one
> learning rate; wiring it to the same 0.0005 is behaviour-preserving.
>
> The genuinely untested repair is an upstream reward normaliser plus the textbook
> convention — see [[return_mode_cmp_10M]] §"what the missing piece is".

> **Status**: PLANNED — Fix 1 **WITHDRAWN** (refuted 2026-09-05, see above); Fix 2 still planned. Amended 2026-09-01 after adversarial plan review
> **Opened**: 2026-09-01
> **Related**: [[MODULATION_SITE_REFACTOR]] (must land **before** its fixture capture) · [[KNOWN_BUGS]] rows "P1 #5" and "A1" · `docs/reviews/diagnosis_20260723/findings_rppo_trainer.md` Findings 1 and 3 · `docs/reviews/diagnosis_20260723/review_full_diagnosis_20260723.md` P1 #5

> 🚧 **Blocked on one user decision.** Everything is specified and ready except **Open question 5** — how much slowdown in the policy's learning is acceptable while still calling this a clean bug fix. The safety-check thresholds are *pre-registered*, meaning they are fixed before the measurement is taken and cannot be adjusted afterwards, so the answer is needed **before the "before" measurement is taken** (the stage called Stage A-obs below). The implementer must not pick a number.

### Amendment log

| Date | Change | Trigger |
|---|---|---|
| 2026-09-01 | **The safety check now reads the tail, not the middle.** The check that guards against the fix accidentally slowing the policy down used to look at a typical (median) update. That statistic cannot see the failure that matters here — the brake binding only on the death-containing updates this project studies — so it was replaced by a four-condition rule reading per-update, unsmoothed statistics. | plan-reviewer M1 |
| 2026-09-01 | **The measurement window moved from 200 to ~1000 iterations, judged on its last fifth, with the trajectory reported** — and a standing obligation added to re-run the same check on the first real training run. 200 iterations is simultaneously the moment the risk looks worst and a window that says nothing about later training. | plan-reviewer M2 |
| 2026-09-01 | **The environment is now pinned**: `configs/environment/experiment/basic/04-jump_attack_10x10.yaml`, single seed (the user's decision). Every diagnostic and speed measurement records environment config, node, GPU, seed and commit alongside the result. | plan-reviewer M3 |
| 2026-09-01 | **The tolerance a "pass" grants is surfaced as Open question 5** rather than left as an implementer default. | plan-reviewer M4 |
| 2026-09-01 | Four specification tightenings: corrected a false claim about the optimizer's saved state keeping its shape; specified the test critic as tabular; required the freeze tests to assert both directions with a nonzero-gradient precheck; corrected two transposed grep counts. Plus an attribution step for a bit-identity mismatch. | plan-reviewer L1–L4, O1 |

---

## Context

Two defects in how the recurrent-PPO agent learns are being fixed together, in one change, before any further training runs are launched.

**The first defect is about units.** The agent estimates "how much reward is still coming" for each step of a 128-step training window. At the window's edge it has to guess, and it asks its own value estimator (the "critic") for that guess. But the critic has been taught to answer in a *rescaled* currency — every window, its answers are squashed to average 0 and spread 1 — while the guess is then added onto raw, unrescaled reward numbers whose spread is about 25. The guess therefore arrives about **25 times too small**: the window-edge correction that was added in an earlier fix delivers roughly **4 % of its intended effect**, and the bias it was meant to remove is materially back for roughly the last 45 steps of every 128-step window — about **35 % of all training targets**. The same mismatch also leaks into the policy update, because in this return mode the resulting advantage numbers are never rescaled afterwards.

**The second defect is about learning rates.** Every recurrent-PPO configuration file advertises a separate learning rate for the critic (`lr_critic`), and *nothing in the codebase reads it*. One optimizer covers the shared trunk, the action head, the value head and the neuromodulator alike, all at the actor's rate. The neuromodulator — the object of study in this project — trains at whatever rate it happened to inherit. The fix wires the critic's rate for real, adds a new explicit rate for the modulator, and sets all of them to the value everything actually trains at today, so that **training dynamics do not change at all**. Nothing about the numbers moves; what changes is that the file now describes reality and the modulator's speed becomes a knob rather than an accident.

They are fixed and evaluated **together, never A/B'd apart**, because they interact through the same object: the critic's learning problem. The first defect corrupts what the critic is asked to learn; the second controls how fast it learns it. Splitting them into separate training comparisons would attribute a joint effect to whichever landed second.

**All existing runs are retrained from scratch after this lands.** No old checkpoint needs to keep working, and no pre-fix run will be compared against a post-fix run. Do not build migration or compatibility machinery.

---

## Analysis

### Part 1 — the units mismatch

#### The code path

| Step | Location | What happens |
|---|---|---|
| a | `src/models/recurrent_ppo_trainer.py:311-314` | `bootstrap_value = v_boot.squeeze(-1)` — the raw value-head output, no rescaling. |
| b | `:374-376` | that value seeds `compute_mc_returns`, whose reverse scan does `ret = reward + gamma * ret` over **raw env rewards** (`:127`). |
| c | `:378` | `returns = (returns - mean) / (std + 1e-7)` — the accumulated returns are z-scored using **this window's** batch statistics. |
| d | `:379` | `targets = returns` — the z-scored numbers become the critic's regression target. |
| e | `:184` (`ppo_loss_fn`) | `value_loss = 0.5 * mean((new_values - targets)^2)` — the only signal the critic ever gets. So at convergence `V ≈ (G - μ_w) / σ_w`. |
| f | back to (a) | the next iteration feeds that normalised-scale output back in as the seed for a raw-reward accumulation. |

The seed is supposed to contribute the future raw return `G_future`. It contributes `(G_future - μ_w) / σ_w`.

#### Magnitude, from recorded data

From four recorded rollouts (`results/trajectories/20260810-*_rppo_restprem_a07..a10`):

| Quantity | Value |
|---|---|
| per-step reward standard deviation | ≈ 7.4 |
| raw Monte-Carlo return σ (whole-run) | ≈ 23–25, consistent across runs |
| per-128-step-window σ | median 11, p10 3.4, p90 29 |

So the seed arrives at roughly **1/25 of correct magnitude**, and the window-edge correction delivers ≈ 4 % of its intended value. The error's reach into the window is `gamma^k * (1 - 1/σ)`; at `gamma: 0.95` it exceeds 10 % of the target standard deviation for roughly the **last 45 steps of each 128-step window**, i.e. about **35 % of all training targets**.

Why the reward is large-scale, not small: `calculate_drive` (`src/environment/core.py:49-53`) is a Euclidean distance in **raw** units — satiation and injury are each on a 0–100 scale, so the drive spans roughly 0–141 — and `body.death_penalty` is 100. The normalised `drive_hunger` / `drive_injury` quantities at `core.py:725-726` are **logging-only** and must not be used to reason about reward scale.

#### Two aggravating facts

1. **The artefact reaches the policy gradient, not only the critic.** In this return mode `advantages = returns_norm - trajectories.value` (`:380`) and is **not** re-normalised afterwards. The GAE branch does normalise its advantages (`:396`). So the MC branch feeds an unrescaled, mis-seeded advantage straight into the clipped policy objective.
2. **The seven existing regression tests cannot catch this.** `tests/models/test_mc_window_bootstrap.py` feeds synthetic numbers into `compute_mc_returns` in isolation and never closes the loop through a critic that was trained on normalised targets. They correctly guard the H4 window-edge semantics and must keep passing; they are simply blind to the units question.

#### Root cause, and the repair chosen

The 2026-07-23 diagnosis records this as **Finding 1** and separately records **Finding 3**: the critic chases a *per-window affine target that moves every iteration* (`μ_w`, `σ_w` drift as the policy improves and death frequency changes), so it can never converge to a stationary function. Finding 3 is what makes Finding 1 possible — normalising the target is exactly what puts the critic's output in a different currency from the rewards.

Both findings propose the same one-line repair: **normalise advantages, not returns.**

**This plan adopts that repair.** Concretely, in the MC branch: keep returns in raw units, use them directly as the critic's target, form the advantage as `returns - value`, and normalise *that*. Reasons:

1. **It removes the defect at the root.** The critic is trained in raw units, so its output is already in the currency the reverse scan accumulates. There is nothing left to convert, no tracked statistics to maintain, no one-iteration lag.
2. **It fixes Finding 3 in the same stroke.** The regression target becomes a stationary function of the state. The alternative repair — de-normalising the seed with tracked `μ`, `σ` — leaves the moving target in place, i.e. leaves the enabling condition for the very bug being fixed, and introduces a stale-statistics error of its own (the seed would be calibrated to the *previous* window's affine transform).
3. **It fixes aggravating fact (1) for free.** Advantages become normalised, exactly as in the GAE branch.
4. **It makes the two return modes structurally identical.** Today the file carries two different conventions for what "target" and "advantage" mean; after the change the MC branch mirrors the GAE branch line for line. That is a real maintenance benefit and removes a class of future confusion.
5. **It is the standard PPO formulation.** Raw value targets plus normalised advantages is what essentially every reference PPO implementation does, and what this file's own GAE branch already does. Deviating from it is what needs justification, not adopting it.

Rejected alternatives, and why:

| Alternative | Why not |
|---|---|
| De-normalise the seed: `bootstrap_raw = μ + σ·V` using previous-window or running statistics | Leaves Finding 3 (moving target) alive; adds lagged statistics that must be threaded through a jitted function and checkpointed; leaves MC advantages unnormalised. Compensates for the defect instead of removing it. |
| Keep normalised targets but with EMA statistics instead of per-window | Same objections, plus new cross-iteration state. |
| Symlog-transformed value target (DreamerV3-style) | Stationary and invertible, and would keep magnitudes O(1) — but it is a **design change to the value objective**, not a bug fix, and belongs in a feature plan if it is ever wanted. Named here only as a fallback (see the gate below). |
| Reduce `vf_coef` to compensate for the larger value loss | A hyperparameter change smuggled into a bug fix. It would confound every downstream comparison. Explicitly out of scope. |

#### The one consequence that must be measured, not assumed

⚠️ **Read this before implementing Stage B.**

Raw value targets have σ ≈ 23–25 instead of ≈ 1. So:

- `loss/value` will jump by roughly σ² — from order 0.1–1 to order 10²–10³. **This is expected, not a regression.** Anyone reading the WandB curve must be told in advance.
- Gradients from the value term grow by roughly σ (≈ 25×).
- Gradient clipping is **global**: `optax.clip_by_global_norm(0.5)` over *all* parameters (`train.py:1167-1170`, `max_grad_norm: 0.5` in every config). When clipping binds, every parameter's update is scaled by the same factor `min(1, c/‖g‖)`. If the value term comes to dominate `‖g‖`, the **actor's effective step shrinks by that same factor** — a large, unintended change to the policy's learning speed.

⚠️ **And the throttle need not be uniform.** The clip may bind only on the *subset* of update steps whose window contained a death — which are the highest-magnitude rewards in this environment, and precisely the steps carrying the nociceptive lesson this project exists to study. In that regime a typical (median) update is untouched while the informative updates are crushed. Any statistic that summarises the run with a middle value is blind to it. This is why the Gate below reads **per-update, unsmoothed tail statistics** and not an average of averages, and why the instrument writes a per-iteration CSV rather than relying on the 100-iteration WandB means.

This is a genuine risk of the chosen repair, and it cannot be settled by reading the code. It is settled by measurement: Stage A adds per-parameter-group gradient-norm logging **and per-update clip statistics**, Stage A-obs records the pre-fix composition, Stage B records it again, and a **pre-registered four-condition decision rule** (below) says what to do. The GAE branch already lives in this regime, but the registry records that every live config uses MC returns, so the GAE path is not evidence that the regime is safe here.

#### Scope: plain PPO is excluded from Part 1

`src/models/ppo_trainer.py:216-221` carries the same normalise-the-returns pattern, but it has **no window-edge bootstrap at all** (recorded OPEN/Low in the registry as "Plain-PPO MC returns also lack the window-edge bootstrap"), so there is no critic output being folded into a raw-reward accumulation and therefore no units bug. It does inherit Finding 3 and the unnormalised-advantage issue. Both are out of scope: no live config uses plain PPO, and touching it would enlarge a fix that must land quickly and cleanly ahead of the refactor. **Do not modify `src/models/ppo_trainer.py` in this plan.**

---

### Part 2 — the learning rates

#### What is actually true today

- `lr_critic` has **zero consumers** anywhere in `src/`, `train.py`, or `scripts/`. 23 YAML files declare it; nothing reads it.
- `train.py:762`: `lr = args.lr or config.get_mandatory('agent.lr_actor')`.
- `train.py:1165-1172`: one optimizer — `nnx.Optimizer(model, optax.chain(clip_by_global_norm(max_grad_norm), optax.adam(lr)), wrt=nnx.Param)` — covering trunk, actor head, critic head and modulator alike.
- `train.py:1259-1274` (plain PPO): the comment reads *"For now, let's use lr_actor as primary"*; `optax.adam(lr_actor)`, no clipping. `lr_critic` is dead there too.
- This affects **all** recurrent PPO, plain and neuromodulated — not only the NMN configs.

#### The user's decisions (final)

1. **Wire `lr_critic`** rather than delete it. A declared rate that controls nothing is a defect.
2. **Add a new `lr_modulator` key**, so the modulator's training speed is a controlled variable rather than a side effect of which optimizer it joined. This matters now because the pending [[MODULATION_SITE_REFACTOR]] gives the modulator **both** actor and critic modulation heads.
3. **Zero change to training dynamics.**

#### The migration rule — per file, not a single number

> **Rule: `lr_critic` and the new `lr_modulator` are each set to THAT FILE'S OWN `lr_actor` value.**

This is what preserves current behaviour, because every config trains everything at its own `lr_actor` today. Quoting one number would be wrong: `configs/models/ppo/ppo.yaml` uses `lr_actor: 0.0003`, and its declared `lr_critic: 0.001` is *higher* than its actor rate — the reverse of the recurrent-PPO case. The "critic is 5× slower than advertised" framing holds for recurrent PPO only, not universally.

If the code starts reading `lr_critic` while the files still declare their current values, every recurrent-PPO critic silently drops to 1/5 speed and plain PPO's critic jumps to 3.3× — precisely the outcome this plan exists to avoid. **The config migration and the code change must land in the same commit.**

#### Design decision — one optimizer with per-leaf learning-rate scaling

Two candidate shapes were considered.

**Candidate A — `optax.multi_transform`** with one `optax.adam(lr_g)` per group. Rejected. `multi_transform` is implemented with `optax.masked`, which replaces out-of-group leaves with `MaskedNode()` sentinels. That has two costs: (i) any `clip_by_global_norm` placed *inside* a branch would compute a **per-group** norm, silently turning global clipping into per-group clipping — a behaviour change; and (ii) the optimizer state pytree gains masked sentinels, changing the structure that `nnx.state(optimizer)` serialises through `src/utils/checkpoint_restore.py`.

**Candidate B — one shared Adam, per-leaf learning-rate scaling applied afterwards. CHOSEN.**

The key algebraic fact: `optax.adam(lr)` is exactly `chain(scale_by_adam(), scale(-lr))`, and every operation in `scale_by_adam` is **elementwise per leaf** — there is no cross-leaf reduction anywhere in Adam. Therefore "one shared `scale_by_adam` followed by a per-group `-lr` multiply" is **mathematically identical** to "a separate `optax.adam(lr_g)` per group", not merely similar. It is also simpler and has no masked sentinels.

⚠️ **Correction (plan-reviewer L1): the optimizer-state pytree does NOT keep the same structure.** An earlier draft of this plan claimed it did; that claim is false and was verified false against optax 0.2.6. Because `optax.adam` is itself a chain, today's state nests as `(EmptyState, (ScaleByAdamState, EmptyState))`, while the new chain is a **flat** `(EmptyState, ScaleByAdamState, EmptyState)`. The *contents* (the Adam moment pair) are identical; only the nesting depth differs.

Consequences:

- **For this plan: none.** Every run is retrained from scratch (see Context), and V6 round-trips the new shape through `src/utils/checkpoint_restore.py`.
- **For anyone else: warm-resuming a pre-Stage-A checkpoint's optimizer state will fail structurally**, with a pytree-structure mismatch, not silently. That is the correct failure mode, and it is another reason no migration shim is built. Say so in the Implementation Report so the failure is recognised rather than debugged.

```
optax.chain(
    optax.clip_by_global_norm(max_grad_norm),   # UNCHANGED — still global, still first
    optax.scale_by_adam(),                      # one shared moment pair, elementwise
    scale_by_group_lr({'trunk': ..., 'actor': ..., 'critic': ..., 'modulator': ...}),
)
```

**Effect on gradient clipping: none.** The clip stays outside and first, over the full parameter tree, in the same traversal order, with the same threshold. It is not made per-group. Making it per-group would itself be a behaviour change and is explicitly *not* done.

**Bit-identity claim.** When all group rates are equal, the final stage is `u * (-lr)` per leaf — exactly what `scale(-lr)` does elementwise. Adam's moments, bias correction and epsilon are untouched. There is no reordering of any floating-point summation: the only reduction in the whole chain is `global_norm` inside the clip, which operates on the same pytree as before. **Bit-identical output is therefore claimed and must be demonstrated, not asserted** (see Verification). If the implementer measures any discrepancy at all, that is a signal something structural changed — investigate, do not paper over it with a tolerance.

#### The trunk question

The recurrent model has a **shared trunk** (observation encoder, RNN cell, and — in modulated hierarchical mode — the encoder LayerNorms) that feeds both heads. Splitting learning rates by head forces a decision about which rate the trunk gets.

**Decision: the trunk trains at `lr_actor`.** Rationale: `lr_actor` is the rate everything trains at today, so this is the choice that makes the zero-change claim exactly true for the largest share of parameters; and the policy is the thing whose learning speed the actor rate has always described. This is numerically moot while all three rates are equal, but it becomes load-bearing the moment someone sets them apart, so it is pinned by a test rather than left implicit.

#### Labelling must fail loudly, not fall back

The group label is derived from each parameter's **top-level attribute name**. Two traps:

- `mod_unimodal_ln`, `mod_multimodal_ln`, `mod_flat_ln` (`recurrent_ppo_network.py:242-245`) begin with `mod` but are **task-side** LayerNorms on the encoder output, not modulator parameters. They belong to **trunk**. A `startswith('mod')` rule would silently mislabel them.
- The pending [[MODULATION_SITE_REFACTOR]] adds new head attributes. If the labeller defaults unknown names to `trunk`, those heads would be silently swept into the actor's rate — reintroducing exactly the "the modulator's rate is an accident" defect this fix removes.

**Therefore the labeller carries an explicit allowlist and raises `ValueError` on any unrecognised top-level name.** This is the no-fallback-defaults rule applied to parameter labelling. It means the refactor *cannot* add a head without consciously assigning it a group.

⚠️ **A circular-verification hazard, stated so it is not walked into.** Because all three rates are equal, a labeller that put *everything* in one group would still produce bit-identical training. **The bit-identity test cannot detect mislabelling.** Mislabelling is caught only by the deliberately-unequal-rate tests described in Verification, which observe *which parameters moved* and do not depend on the labeller being right.

#### Scope: plain PPO is INCLUDED in Part 2

`ActorCriticMLP` (`src/models/ppo_network.py:6-29`) has fully separate actor and critic stacks (`actor_layers`, `actor_head`, `critic_layers`, `critic_head`) — **no shared trunk and no modulator**. The same labeller covers it unchanged. Wiring `lr_critic` there is two lines plus two config values, and it removes the worse trap of a key that is live in one algorithm and dead in another. So:

- **In scope:** the optimizer construction at `train.py:1274` and the `lr_critic` values in `configs/models/ppo/ppo.yaml` and `configs/models/ppo/neuromodulated_ppo.yaml`.
- **Out of scope:** `lr_modulator` for plain PPO — `ActorCriticMLP` has no modulator, and a mandatory key for a component that does not exist would be a fabricated requirement. Plain PPO reads `lr_actor` and `lr_critic` only. If a leaf ever lands in the `trunk` or `modulator` group under plain PPO, the labeller raises — the desired loud failure.
- **Out of scope, flagged for `bug-curator` as a separate row:** `configs/models/ppo/neuromodulated_ppo.yaml` builds an **unmodulated** `ActorCriticMLP` — `train.py`'s `elif algorithm == "PPO"` branch never passes a `modulation_config`. The file's name promises something the code does not do. Do not fix it here; report it.

#### The archived environment config — deliberately out of scope

`configs/environment/experiment/archive/hypervigilance/testbed_cellC_native_v2.yaml:35-36` declares `lr_actor: 0.0005` / `lr_critic: 0.0001` inside an `agent:` block. It is **not missed — it is deliberately left alone.** Reasons:

- It is an **auto-dumped snapshot** of a past run's resolved config (alphabetically sorted keys, full `agent:` + env blocks), living under `archive/`, referenced from nowhere in the repo.
- Editing it would rewrite a historical record of what that run was configured with. The record is already misleading (the run trained everything at 0.0005 regardless of the declared 0.0001), and the honest remedy for that is the change-log entry, not a retroactive edit.
- If anyone ever revives it, `config.get_mandatory('agent.lr_modulator')` raises at startup — an immediate, loud failure, which is strictly safer than a silently-migrated file running at rates nobody chose. Whoever revives it must then set `lr_critic` and `lr_modulator` to `0.0005` to reproduce the original run.

---

## Implementation Plan

### Sequencing — this must not be reordered

```
  HEAD  ──►  Stage A  ──►  Stage A-obs  ──►  Stage B  ──►  Gate  ──►  [ MODULATION_SITE_REFACTOR C0a / C0b ]
            (Fix 2 +      (measure the      (Fix 1)      (decide)
             logging)      "before")
```

1. **Stage A first, Stage B second.** Stage A's zero-change claim is cheapest and sharpest to prove against an unmodified tree. If Stage B landed first, Stage A's "before" half would have to be regenerated from an intermediate commit, and a bit-identity failure could not be attributed.
2. **Stage A-obs sits between them** because the diagnostic instrumentation is added in Stage A, so the only tree that has the instrument but not Fix 1 is the post-Stage-A tree. That is the "before" for Fix 1.
3. **Both stages land strictly before** the [[MODULATION_SITE_REFACTOR]]'s Checkpoint **C0a** (golden-fixture capture) and **C0b** (the "before" losses for its end-to-end parity check). That refactor's own sequencing rule (its C0b note) forbids this fix landing between C0b and its "after" half; landing both stages entirely before C0a satisfies it with margin, and means the refactor's before/after pair is taken on already-fixed code.
4. **Do not start the refactor's fixture capture until the Gate below has been passed and recorded.**
5. **Open question 5 must be answered before Stage A-obs runs.** The Gate's four thresholds are pre-registered — fixed before the measurement, unchangeable after it. If the user has not confirmed them, Stage A can be implemented and committed, but the diagnostic run waits.

### Stage A — wire the learning rates, add the instrument (zero behaviour change)

#### New file: `src/models/lr_groups.py`

Placed under `src/`, **not** `scripts/`, so the [[SCRIPTS_DEPENDENCY_MAP]] maintenance contract is not triggered. Do not relocate it.

```python
"""Parameter-group learning rates for the PPO / recurrent-PPO optimizers.

Four groups: trunk, actor, critic, modulator.

The SHARED TRUNK (observation encoder, RNN cell, and the modulated-mode encoder
LayerNorms) trains at the ACTOR's rate. That is the choice that keeps behaviour
identical to the single-optimizer setup this replaced, in which everything trained
at `lr_actor`. It becomes load-bearing only when the rates are set apart; a test
pins it.

Note `mod_unimodal_ln` / `mod_multimodal_ln` / `mod_flat_ln` are TASK-side
LayerNorms on the encoder output, not modulator parameters -> trunk. A
`startswith('mod')` rule would mislabel them.

There is no fallback group: an unrecognised top-level attribute name raises. A new
head added by a future refactor must be assigned deliberately.
"""
import jax
import jax.numpy as jnp
import optax

_GROUP_BY_TOP_LEVEL_NAME = {
    # --- recurrent PPO (ActorCriticRNN) ---
    'obs_encoder':       'trunk',
    'rnn_cell':          'trunk',
    'mod_unimodal_ln':   'trunk',
    'mod_multimodal_ln': 'trunk',
    'mod_flat_ln':       'trunk',
    'actor_fc1':         'actor',
    'actor_fc2':         'actor',
    'critic_fc1':        'critic',
    'critic_fc2':        'critic',
    'modulator':         'modulator',
    # --- plain PPO (ActorCriticMLP) ---
    'actor_layers':      'actor',
    'actor_head':        'actor',
    'critic_layers':     'critic',
    'critic_head':       'critic',
}

LR_GROUPS = ('trunk', 'actor', 'critic', 'modulator')


def _top_level_name(path):
    """First path element of a jax key path, as a plain string."""
    k = path[0]
    return str(getattr(k, 'key', getattr(k, 'name', k)))


def param_group_labels(params):
    """Pytree of group-name strings, same structure as `params`.

    Raises ValueError on any top-level attribute not in the allowlist.
    """
    def label(path, _leaf):
        name = _top_level_name(path)
        if name not in _GROUP_BY_TOP_LEVEL_NAME:
            raise ValueError(
                f"lr_groups: unlabelled top-level parameter '{name}'. Add it to "
                f"_GROUP_BY_TOP_LEVEL_NAME with a deliberate group choice — there is "
                f"no fallback group."
            )
        return _GROUP_BY_TOP_LEVEL_NAME[name]
    return jax.tree_util.tree_map_with_path(label, params)


def scale_by_group_lr(lr_by_group):
    """optax transformation: multiply each leaf by -lr[group(leaf)].

    Exactly replaces `optax.scale(-lr)` when every group carries the same rate, so
    `chain(scale_by_adam(), scale_by_group_lr({...: lr}))` is bit-identical to
    `optax.adam(lr)`. A group present in the model but missing from `lr_by_group`
    raises (no fallback).
    """
    def init_fn(_params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        def scale(path, u):
            name = _top_level_name(path)
            if name not in _GROUP_BY_TOP_LEVEL_NAME:
                raise ValueError(f"lr_groups: unlabelled top-level parameter '{name}'.")
            group = _GROUP_BY_TOP_LEVEL_NAME[name]
            if group not in lr_by_group:
                raise ValueError(
                    f"lr_groups: parameter '{name}' is in group '{group}', which has no "
                    f"learning rate. Provided: {sorted(lr_by_group)}."
                )
            return u * (-lr_by_group[group])
        return jax.tree_util.tree_map_with_path(scale, updates), state

    return optax.GradientTransformation(init_fn, update_fn)


def group_grad_norms(grads):
    """{group: global_norm(grads in that group)} for every group in LR_GROUPS.

    Groups with no parameters report 0.0 (e.g. `modulator` on a baseline model).
    Diagnostic only — never feeds an update.
    """
    leaves = {g: [] for g in LR_GROUPS}
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        leaves[_GROUP_BY_TOP_LEVEL_NAME[_top_level_name(path)]].append(leaf)
    return {g: (optax.global_norm(v) if v else jnp.float32(0.0)) for g, v in leaves.items()}
```

#### `train.py` (around lines 754-765) — read the new keys

```python
# BEFORE:
        hidden_size = args.hidden_size or config.get_mandatory('agent.hidden_size')
        lr = args.lr or config.get_mandatory('agent.lr_actor')
        config.set('agent.hidden_size', hidden_size)
        config.set('agent.lr_actor', lr)

# AFTER:
        hidden_size = args.hidden_size or config.get_mandatory('agent.hidden_size')
        # `--lr` keeps its current meaning: it overrides THE rate, i.e. all of them.
        # Before this change one optimizer covered every parameter, so `--lr X` set
        # actor, critic and modulator alike; overriding all three preserves that exactly.
        lr           = args.lr or config.get_mandatory('agent.lr_actor')
        lr_critic    = args.lr or config.get_mandatory('agent.lr_critic')
        config.set('agent.hidden_size', hidden_size)
        config.set('agent.lr_actor', lr)
        config.set('agent.lr_critic', lr_critic)
        if algorithm == "RecurrentPPO":
            # Modulator rate exists only where a modulator can exist (ActorCriticRNN).
            lr_modulator = args.lr or config.get_mandatory('agent.lr_modulator')
            config.set('agent.lr_modulator', lr_modulator)
```

The `config.set(...)` calls matter beyond the local variable: the dumped `models/config.yaml` is the ground truth for what a run used, so all three resolved rates must appear there.

#### `train.py:1163-1172` — the recurrent-PPO optimizer

```python
# BEFORE:
        # Use optax.chain for gradient clipping (Option A)
        max_grad_norm = config.get_mandatory('agent.max_grad_norm')
        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr),
            ),
            wrt=nnx.Param,
        )

# AFTER:
        # Gradient clipping stays GLOBAL and stays FIRST — it is applied over the whole
        # parameter tree exactly as before. Only the final per-leaf learning-rate multiply
        # is split by group. `optax.adam(lr) == chain(scale_by_adam(), scale(-lr))` and
        # every Adam operation is elementwise, so at equal rates this is bit-identical.
        from src.models.lr_groups import scale_by_group_lr
        max_grad_norm = config.get_mandatory('agent.max_grad_norm')
        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.scale_by_adam(),
                scale_by_group_lr({
                    'trunk':     lr,            # shared trunk -> actor's rate (documented choice)
                    'actor':     lr,
                    'critic':    lr_critic,
                    'modulator': lr_modulator,
                }),
            ),
            wrt=nnx.Param,
        )
```

#### `train.py:1274` — the plain-PPO optimizer

```python
# BEFORE:
        optimizer = nnx.Optimizer(model, optax.adam(lr_actor), wrt=nnx.Param)

# AFTER:
        # No gradient clipping here today — do not add any.
        # ActorCriticMLP has separate actor/critic stacks and no trunk or modulator,
        # so only those two groups are supplied; anything else raises.
        from src.models.lr_groups import scale_by_group_lr
        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.scale_by_adam(),
                scale_by_group_lr({'actor': lr_actor, 'critic': lr_critic}),
            ),
            wrt=nnx.Param,
        )
```

Also update the stale comment at `train.py:1258-1260` (*"We can support dual LR by choosing one or using a complex optimizer / For now, let's use lr_actor as primary"*) — dual LR is now supported.

#### `src/models/recurrent_ppo_trainer.py:332-354` — the diagnostic instrument

Append seven scalars to the aux tuple. **Append at the end** so the existing indices `l[1][0..4]` used in `train.py` keep working.

```python
# BEFORE (:332-354, abridged):
    grad_norm = optax.global_norm(grads)
    ...
    ppo_loss, v_loss, ent_loss = aux
    return loss, (ppo_loss, v_loss, ent_loss, grad_norm, mod_grad_norm)

# AFTER:
    grad_norm = optax.global_norm(grads)
    ...
    # Per-group gradient norms. Diagnostic only — never feeds an update. These are the
    # instrument the Fix-1 gate reads: with raw value targets the value term's gradient
    # grows ~sigma, and clipping is global, so the actor's share of the clipped step can
    # shrink. Reuses the same labelling as the optimizer split, so the two cannot drift.
    from src.models.lr_groups import group_grad_norms
    gnorms = group_grad_norms(grads)
    # Target/advantage scale, read straight off the batch the critic is trained on.
    target_std  = jnp.std(batch.targets)
    target_mean = jnp.mean(batch.targets)
    adv_std     = jnp.std(batch.advantages)

    ppo_loss, v_loss, ent_loss = aux
    return loss, (ppo_loss, v_loss, ent_loss, grad_norm, mod_grad_norm,
                  gnorms['trunk'], gnorms['actor'], gnorms['critic'], gnorms['modulator'],
                  target_std, target_mean, adv_std)
```

Note: `mod_grad_norm` (the existing `'modulator' in grads` probe at `:336-347`) is **left in place unchanged**. If `Grad/norm_modulator` turns out non-zero while `modulator/grad_norm` logs `0.0`, that probe is broken — report it as a finding; do not silently retire it in this change.

#### `train.py:1779-1804` — log the new scalars

```python
# AFTER (added alongside the existing avg_* lines):
                    avg_gn_trunk     = jnp.mean(jnp.array([l[1][5] for l in losses]))
                    avg_gn_actor     = jnp.mean(jnp.array([l[1][6] for l in losses]))
                    avg_gn_critic    = jnp.mean(jnp.array([l[1][7] for l in losses]))
                    avg_gn_modulator = jnp.mean(jnp.array([l[1][8] for l in losses]))
                    avg_target_std   = jnp.mean(jnp.array([l[1][9] for l in losses]))
                    avg_target_mean  = jnp.mean(jnp.array([l[1][10] for l in losses]))
                    avg_adv_std      = jnp.mean(jnp.array([l[1][11] for l in losses]))
```

and extend `_loss_sample` with the seven keys `loss/grad_norm_trunk`, `loss/grad_norm_actor`, `loss/grad_norm_critic`, `loss/grad_norm_modulator`, `loss/value_target_std`, `loss/value_target_mean`, `loss/advantage_std`. They join the windowed set and go through `spread()` like the existing five; keep them as JAX scalars (no `float()` on the hot path), matching the existing pattern documented in the block comment there.

#### `train.py:1779-1804` — the clip statistics (added per plan-reviewer M1)

**Why this is not optional.** The Gate is supposed to detect the value gradient eating the shared clip budget. The quantity it reads, `loss/grad_norm`, is currently averaged twice before anyone sees it, and both averages destroy exactly the signal the Gate needs:

- **First smoothing — within the iteration.** `train.py:1783` computes `jnp.mean` over the iteration's `K_epochs` update steps (`K_epochs: 4` in `recurrent_ppo_nmn_het_film_g1.yaml`; `src/models/recurrent_ppo_trainer.py:414-417` appends one aux tuple per epoch). The clip factor is a **nonlinear** function of the norm — `f = min(1, c/‖g‖)` — so `f(mean ‖g‖) ≠ mean f(‖g‖)`, and neither equals `min f(‖g‖)`. The factor must be computed **per update** and only then reduced.
- **Second smoothing — across iterations.** `configs/train/recurrent_ppo.yaml:56-63` sets `logging.step.smoothing_iters: 100` / `interval_iters: 50`, so each WandB row is a mean over 100 iterations emitted every 50. A 1000-iteration diagnostic yields ~20 rows, each already an average of 100 iterations. A percentile computed over 20 pre-averaged rows cannot see a tail that lives in individual iterations.

The failure mode this hides is the one that matters here: if the clip binds only on **death-containing** update steps — the exact steps carrying the nociceptive lesson this project studies — then the median clip factor is 1.0 before and after, the Gate passes, and the policy's update is crushed at precisely those steps.

**Two changes, both diagnostic-only, neither touching an update.**

**(1) Per-update clip statistics, computed correctly and logged to WandB.** In the same block at `train.py:1780-1785`, alongside the existing `avg_*` lines. `max_grad_norm` is already a local in `main()` (assigned at `train.py:1164`, same function, RecurrentPPO branch — the loop at 1780 is inside that branch, so it is in scope; if the implementer finds otherwise, read it from `train_config.max_grad_norm` rather than re-reading the config on the hot path).

```python
                    # Per-UPDATE clip statistics. `f = min(1, c/||g||)` is nonlinear, so it
                    # must be computed per update step and only then reduced — averaging the
                    # norms first and clipping the average is a different, blinder number.
                    # Diagnostic only; nothing here feeds an update.
                    _gn = jnp.stack([l[1][3] for l in losses])          # (K_epochs,)
                    _f  = jnp.minimum(1.0, max_grad_norm / (_gn + 1e-12))
                    avg_grad_norm_max   = jnp.max(_gn)
                    avg_clip_factor     = jnp.mean(_f)                  # mean over updates
                    avg_clip_factor_min = jnp.min(_f)                   # worst update
                    avg_clip_binding    = jnp.mean((_gn > max_grad_norm).astype(jnp.float32))
```

Add four more keys to `_loss_sample`: `loss/grad_norm_max`, `loss/clip_factor`, `loss/clip_factor_min`, `loss/clip_binding_frac`. All stay JAX scalars — no `float()` on the hot path.

⚠️ **Naming.** Do **not** call any of these `clip_fraction` / `clipfrac`. In PPO that name conventionally means the *policy-ratio* clip fraction, which is a different quantity; grep confirms neither name is currently used in `train.py` or `src/`, and this change must not create the collision.

**(2) A per-iteration, unsmoothed CSV — the tail-visible record.** WandB rows are window-averaged and cannot be un-averaged. Write one raw row per iteration to `os.path.join(results_dir, "grad_diagnostics.csv")`, always on, for every recurrent-PPO run.

Header:

```
iteration,grad_norm_mean,grad_norm_max,clip_factor,clip_factor_min,clip_binding_frac,gn_trunk,gn_actor,gn_critic,gn_modulator,value_target_std,value_target_mean,advantage_std,value_loss
```

**It must not add a host sync.** The block comment already in `train.py:1786-1799` explains why nothing on this path is `float()`-converted per iteration. Respect it: append the iteration's JAX scalars to a plain Python list `_diag_rows`, and convert + write the whole buffer inside the **existing** emission branch, reusing the sync that already happens there.

- Flush trigger: the two-level path's `_emit` (`train.py:1811`) — **not** `_do_step_log`, which is additionally gated on `wandb_enabled`, so a `--no-wandb` diagnostic would never flush. On the legacy path (`logging_cfg is None`), flush every `log_interval` iterations.
- Convert with one batched `np.asarray(...)` over the buffered rows, exactly as `spread()` does, then `clear()` the buffer.
- Flush once more at the end of training so the last partial buffer is not lost.
- Buffer cost is ~100 iterations × 14 tiny device scalars — negligible; V10 is the check that this is true in practice.

**This instrument is permanent, not diagnostic scaffolding.** It stays in the tree after the Gate is passed, because the Gate answers a question about iterations 1–1000 and the same question has to be re-asked at iteration 50k (see "The post-launch obligation" below).

#### Config migration — `lr_critic` set to the file's own `lr_actor`, plus a new `lr_modulator`

**Twenty recurrent-PPO files.** Each currently has `lr_actor: 0.0005` and `lr_critic: 0.0001`. In each: set `lr_critic: 0.0005` and add `lr_modulator: 0.0005` immediately after it. None of these files use `extends:` — they are all standalone, so every one needs the edit.

| # | File | `lr_actor` | `lr_critic` before → after | add `lr_modulator` |
|---|---|---|---|---|
| 1 | `configs/models/recurrent_ppo/recurrent_ppo.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 2 | `configs/models/recurrent_ppo/recurrent_ppo_XS.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 3 | `configs/models/recurrent_ppo/recurrent_ppo_S.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 4 | `configs/models/recurrent_ppo/recurrent_ppo_M.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 5 | `configs/models/recurrent_ppo/recurrent_ppo_L.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 6 | `configs/models/recurrent_ppo/recurrent_ppo_XL.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 7 | `configs/models/recurrent_ppo/recurrent_ppo_gae.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 8 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_screen.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 9 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g2_screen.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 10 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g4_screen.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 11 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g8_screen.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 12 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g16_screen.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 13 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 14 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen_gae.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 15 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g64_screen.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 16 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g128_screen.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 17 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_tempceil5.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 18 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 19 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |
| 20 | `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_unmod.yaml` | 0.0005 | 0.0001 → 0.0005 | 0.0005 |

**Two plain-PPO files.** `lr_critic` only; **do not add `lr_modulator`**.

| # | File | `lr_actor` | `lr_critic` before → after |
|---|---|---|---|
| 21 | `configs/models/ppo/ppo.yaml` | **0.0003** | **0.001 → 0.0003** ← the trap: a blanket 0.0005 here would change behaviour |
| 22 | `configs/models/ppo/neuromodulated_ppo.yaml` | 0.0005 | 0.0001 → 0.0005 |

**One file deliberately NOT changed.** `configs/environment/experiment/archive/hypervigilance/testbed_cellC_native_v2.yaml` — see "The archived environment config" above.

**Verification that the inventory is complete** (run before and after; the "after" run must show zero non-`0.0005`/`0.0003` values and 20 `lr_modulator` keys):

```bash
grep -rn "lr_critic\|lr_modulator\|lr_actor" configs/ | sort
```

#### Docs to update in the same commit

| Doc | Required? | What |
|---|---|---|
| `docs/environment/CONFIG_CRITICAL_SETTINGS.md` | **Yes** | Three registry rows (`agent.lr_actor`, `agent.lr_critic`, `agent.lr_modulator`) + a dated change-log entry. Warranted by exactly this history: a declared rate that controlled nothing for the project's entire recurrent-PPO history. The registry's "Set in" column says `configs/environment/default.yaml` for env keys; for these, write `all 20 configs/models/recurrent_ppo/*.yaml` and note that agent rates are per-model-config, not inherited. Change-log entry must state: `agent.lr_critic` 0.0001 → 0.0005 in 20 rPPO configs and 0.001 → 0.0003 in `ppo.yaml`, 0.0001 → 0.0005 in `neuromodulated_ppo.yaml`; `agent.lr_modulator` added at 0.0005 in 20 rPPO configs; reason = the key was dead and is now live; **blast radius = zero, verified bit-identical** (cite the Stage-A evidence); and that the archived `testbed_cellC_native_v2.yaml` was deliberately left unmigrated and will now raise if revived. |
| `docs/environment/CONFIG_GUIDE.md` and `docs/environment/02_config_schema.md` | **No — but prove it** | Both document the **environment** schema (`load_env_params`, `EnvParams`, the layering system). Neither documents the `agent:` block: `grep -c "agent\."` returns **2** for `CONFIG_GUIDE.md` and **1** for `02_config_schema.md`, both incidental. (An earlier draft transposed these two counts — corrected per plan-reviewer L4; the exemption conclusion is unchanged.) `config_loader` / `state.py` are untouched. **The implementer must re-run that grep and record the counts in the Implementation Report**; if either doc has since gained an agent-key section, the Maintenance Contract activates and both must be updated. |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | **No** | Nothing under `scripts/` is added, moved, renamed or deleted. `src/models/lr_groups.py` is under `src/` precisely to keep it that way. **If the implementer relocates it to `scripts/`, this contract activates.** |
| `docs/develop/active/issues/KNOWN_BUGS.md` | **Do not edit** | The registry is owned by `bug-curator`, and the file currently carries another session's uncommitted edits. After the fix lands, ask `bug-curator` to close rows **A1** and **P1 #5** and to open a new Low row for the misleadingly-named `neuromodulated_ppo.yaml`. |
| `docs/develop/INDEX.md` | Regenerate, leave unstaged | `python scripts/claude/regen_dev_index.py`. The file is dirty from a parallel session — **do not stage it**. |

### Stage A-obs — record the "before" for Fix 1

On the Stage-A tree (Fix 2 landed, Fix 1 not yet), run a fixed-seed diagnostic and save the result to `tmp/`. This is a measurement step, not a code step.

#### The run — every setting pinned

Both halves of the comparison (this run and the Gate's "after" run) must be identical in **all** of the following. An earlier draft named only the model config; that is not enough to make "identical settings" checkable (plan-reviewer M3).

| Setting | Value | Why pinned |
|---|---|---|
| Model config | `configs/models/recurrent_ppo/recurrent_ppo_nmn_het_film_g1.yaml` | the hierarchical + LayerNorm + modulator path real runs use |
| **Environment config** | **`configs/environment/experiment/basic/04-jump_attack_10x10.yaml`** | **the user's decision — this is the environment the next experiment runs on.** Reward composition (death frequency, drive scale) is environment-dependent, so the Gate's verdict only transfers if it is measured on the environment that will actually be used. The σ ≈ 23–25 figures in §Magnitude come from rest-premium recordings, **not** from this environment (plan-reviewer O3 — see the note below) |
| Seed | one fixed seed, **single seed** (the user's decision) | |
| Iterations | **~1000** (see below) | |
| Node, GPU | one named node + one named GPU, **the same on both halves** | the cluster is heterogeneous (`docs/environment/LAB_NODE_GPU_SPEC.md`); a card swap invalidates the speed half of the comparison and can perturb the numeric half |
| Commit SHA | recorded for each half | |

**Record all six in the results file and in the Implementation Report, for each half.** A comparison whose two halves cannot be shown to differ in exactly one thing is not evidence.

#### Why ~1000 iterations, and why the verdict reads the tail of it (plan-reviewer M2)

200 iterations is the least representative window available, and it is unrepresentative in **both** directions:

- **It over-states the risk.** Immediately after Fix 1, the critic starts at ≈ 0 against targets of σ ≈ 20–25, so the value term's share of the gradient is at its **lifetime maximum** in the first few hundred iterations and then falls as the critic fits. A Gate reading only that window can trip on a transient that resolves itself — blocking a correct fix.
- **It under-states the risk.** A pass over iterations 1–200 says nothing about iteration 50 000, by which point the policy has changed, death frequency has changed, and the reward composition with it.

So: run **~1000 iterations**, and evaluate the decision rule on the **final 200** (iterations 801–1000), while **reporting the trajectory** across the whole run in 100-iteration blocks. The final segment is the closest available proxy for the settled regime; the trajectory is what shows whether the segment is settled or still moving. If the per-block clip factor is still trending downward at iteration 1000, say so explicitly — a still-falling trajectory is itself a finding and is reported to the user regardless of whether the numeric rule passes.

#### What to extract

Read the **per-iteration CSV** `results/JAX_RecurrentPPO/<run>/grad_diagnostics.csv`, not the WandB rows. WandB rows are 100-iteration means (`logging.step.smoothing_iters: 100`) and cannot show a tail.

Compute over the final 200 iterations, and per 100-iteration block for the trajectory:

- `clip_factor` — **median**, **p10**, and **p01**
- `clip_binding_frac` — mean, and the **fraction of iterations with `clip_factor < 1`** (i.e. the clip bound on at least one update in that iteration)
- `grad_norm_mean`, `grad_norm_max` — median and p90
- `gn_trunk`, `gn_actor`, `gn_critic`, `gn_modulator` — medians, and each group's **share of the total** (`gn_g / Σ gn`), median over iterations
- `value_target_std`, `value_target_mean`, `advantage_std` — medians

Write everything to `tmp/YYYYMMDD_HHMMSS_lr_and_mc_fix_before.md`.

#### The instrument's own sanity check

Expected "before" shape: `value_target_std ≈ 1.0` and `value_target_mean ≈ 0.0` (targets are z-scored today), and `advantage_std` some arbitrary value that is **not** 1.0 (MC advantages are not normalised today). If `value_target_std` does not come out at ≈ 1.0, the instrument is wired wrong — **stop and fix that before proceeding**; every number downstream depends on it.

Second check, on the clip statistics specifically: confirm `clip_factor_min ≤ clip_factor ≤ 1` in every row and that `clip_binding_frac > 0` exactly when `clip_factor < 1`. These are algebraic identities; if they do not hold, the statistic is miswired.

📌 **Open item O3, carried from the review.** Return σ on `basic/04` is currently **unmeasured** — the ≈ 23–25 figure comes from rest-premium recordings. `value_target_std` in the Gate's "after" run is the first in-situ measurement of it. Report the number even if it is far from 25; the plan's argument depends on σ ≫ 1, not on σ = 25 specifically, but a σ near 1 would mean the units defect is far smaller on this environment than estimated and the user should be told before Stage B is trusted.

### Stage B — fix the units

#### `src/models/recurrent_ppo_trainer.py:369-380` — extract and repair

Extract the target computation into a pure function so it is testable without an environment, then repair it.

```python
# NEW, module level, next to compute_mc_returns:
def compute_mc_targets_and_advantages(rewards, dones, terminateds, values,
                                      bootstrap_value, gamma):
    """MC value targets and advantages for a (T, B) rollout window.

    UNITS. Returns stay in RAW reward units and are used directly as the critic's
    regression target. That is what makes the window-edge bootstrap correct: the
    critic's output is the seed for a raw-reward reverse scan, so the critic must be
    trained in raw units. The previous code z-scored the returns before using them as
    targets, so the critic learned a per-window affine rescaling of the return and the
    seed came back ~1/sigma too small (sigma ~= 23-25 on recorded rollouts).

    Advantages are normalised instead — the same convention as the GAE branch.
    """
    returns = jax.vmap(compute_mc_returns, in_axes=(1, 1, 1, 0, None), out_axes=1)(
        rewards, dones, terminateds, bootstrap_value, gamma
    )
    advantages = returns - values
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    return returns, advantages
```

```python
# BEFORE (:369-380):
        if return_mode.upper() == "MC":
            terminateds = (trajectories.step_info.termination_reason >= 2).astype(trajectories.done.dtype)
            returns = jax.vmap(compute_mc_returns, in_axes=(1, 1, 1, 0, None), out_axes=1)(
                trajectories.reward, trajectories.done, terminateds, bootstrap_value, config.gamma
            )
            # Normalize returns
            returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-7)
            targets = returns
            advantages = returns - trajectories.value

# AFTER:
        if return_mode.upper() == "MC":
            terminateds = (trajectories.step_info.termination_reason >= 2).astype(trajectories.done.dtype)
            targets, advantages = compute_mc_targets_and_advantages(
                trajectories.reward, trajectories.done, terminateds,
                trajectories.value, bootstrap_value, config.gamma
            )
```

The GAE branch is **not** touched — it is already internally consistent.

#### New file: `tests/models/test_mc_return_units.py`

Three tests. **All three must be shown to FAIL on the Stage-A tree and pass after Stage B.** Capture the failing output.

To demonstrate the failure without a branch switch (this repo's git-safety rule forbids switching with untracked data present):

```bash
git worktree add /tmp/pre_fix_units <STAGE_A_SHA>
cp tests/models/test_mc_return_units.py /tmp/pre_fix_units/tests/models/
cd /tmp/pre_fix_units && /home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/models/test_mc_return_units.py -v
# paste the failures into the Implementation Report, then:
git worktree remove /tmp/pre_fix_units
```

**Test 1 — `test_targets_are_in_raw_reward_units`.** Build a synthetic `(T=128, B=8)` reward stream at realistic scale (baseline reward ≈ 2 with noise, a −100 death every ~60 steps, `gamma=0.95`), plus a `dones`/`terminateds` pair consistent with those deaths. Compute the expected raw returns with a **plain NumPy reverse loop written inside the test file** — not by calling `compute_mc_returns`, so the oracle is independent of the code under test. Assert `targets` equals that oracle within `1e-4`, and assert `std(advantages)` is within `1e-3` of `1.0`.
*Pre-fix:* fails on the first assertion — targets are z-scored, so `std(targets) ≈ 1` against an oracle with `std ≈ 20`+ — and on the second, since advantages were never normalised.

**Test 2 — `test_a_correct_critic_is_a_fixed_point` (the loop-closer).** This is the test the seven existing H4 tests cannot be: it closes the loop through the critic's own output.
Take the *oracle* critic — the one that is already exactly right, i.e. `V*(s_t)` = the independently-computed raw discounted future return. Feed `values = V*` and `bootstrap_value = V*(s_T)`. Assert `targets == V*` elementwise (within `1e-4`) and `max|advantages_unnormalised| < 1e-4` before normalisation.
This is the self-consistency property the defect destroys: a critic that is already correct must be trained toward what it already predicts, so its value loss must be zero. *Pre-fix:* the "already correct" critic (which outputs z-scored returns) is handed a target in a different currency, the value loss is large and never reaches zero, and the assertion fails.

**Test 3 — `test_critic_trained_on_targets_converges_to_raw_returns` (end-to-end fixed point).** Iterate the actual loop, five rounds: (i) compute `targets` from the current value estimate and bootstrap; (ii) fit the critic to `targets` with the **same value-loss expression as `ppo_loss_fn`** (`0.5 * mean((V - target)^2)`) for a few hundred Adam steps; (iii) feed its window-edge output back as `bootstrap_value`; repeat. Assert the fixed point matches the independently-computed raw return within 5 %.

⚠️ **The critic here must be TABULAR, not a function approximator** (plan-reviewer L2). "A small value function" was under-specified: a `Dense` net over synthetic features can miss a 5 % tolerance for reasons that have nothing to do with the units bug, making the test flaky and its failure uninterpretable. Specify instead: a free parameter per state — a single `jnp` array of shape `(T + 1, B)` holding `V[t, b]` for each of the `T` in-window steps **plus the window-edge boot state at index `T`** — initialised to zeros and optimised directly. A tabular critic has zero approximation error, so the only thing the 5 % tolerance can be measuring is the fixed point of the target/bootstrap recursion, which is exactly the property under test. Index `T` is the entry that gets fed back as `bootstrap_value`, so the loop is genuinely closed through the critic's own output.
*Pre-fix:* the fixed point lands near the z-scored scale (order 1), not the raw scale (order hundreds), and the assertion fails by two orders of magnitude — which is the measured 1/25-magnitude defect, observed end to end.

The seven existing tests in `tests/models/test_mc_window_bootstrap.py` must keep passing unchanged — `compute_mc_returns` itself is not modified, and their continued passing is the guard that Stage B did not disturb the H4 window-edge semantics.

### The Gate — the one thing this plan cannot pre-decide

**What the Gate is for, in one paragraph.** Fixing the units makes the critic's targets ~25× larger, so the gradient coming from the value term gets ~25× larger too. Gradient clipping in this codebase is **global** — one budget shared by every parameter — so a bigger value gradient can crowd out the policy's share of that budget and quietly slow the policy's learning. Nobody can predict from the code whether that happens; it is measured. The Gate is the pre-registered rule that says, before the measurement is taken, what result means "proceed" and what result means "stop and ask the user".

After Stage B, re-run the **identical** diagnostic from Stage A-obs — same model config, same environment config (`basic/04-jump_attack_10x10.yaml`), same seed, same node, same GPU, same ~1000 iterations — and write `tmp/YYYYMMDD_HHMMSS_lr_and_mc_fix_after.md`.

#### Expected and fine

| Metric | Before | After |
|---|---|---|
| `value_target_std` | ≈ 1.0 | raw scale — expected ≫ 1; the actual number is the first in-situ measurement on `basic/04` (open item O3) |
| `value_target_mean` | ≈ 0.0 | raw-scale, non-zero |
| `advantage_std` | arbitrary, ≠ 1 | ≈ 1.0 |
| `loss/value` | ≈ 0.1–1 | larger by roughly σ² — **expected, not a regression** |

#### The amended pre-registered decision rule

> **These thresholds are pre-registered: they are fixed in this document BEFORE the "before" run is taken, and must not be adjusted after seeing either result.** If the user wants different numbers, they change them here first (see Open question 5).

**Definitions.** All statistics are computed from the per-iteration CSV `grad_diagnostics.csv`, over the **final 200 iterations** (801–1000) of each half. Nothing is read from WandB rows, which are 100-iteration means.

Per iteration `i`, the instrument records `K_epochs` per-update clip factors and reduces them to two numbers:

- `f_i` = **mean** clip factor over that iteration's updates (`clip_factor`)
- `w_i` = **worst-update** clip factor in that iteration (`clip_factor_min`) — the tail-visible one

From those, over the 200-iteration segment:

| Symbol | Plain meaning | Definition |
|---|---|---|
| `M` | typical throttle | `median_i(f_i)` |
| `P10` | throttle on a bad-but-not-rare iteration | `10th percentile over i of w_i` |
| `P01` | throttle on the worst 1 % of iterations | `1st percentile over i of w_i` (reported, not gated) |
| `B` | how often the clip binds at all | fraction of iterations with `w_i < 1` |

**STOP and escalate if ANY of these four fires.** They are OR-ed, not averaged; each covers a failure the others cannot see.

| # | Trip condition | The failure it catches |
|---|---|---|
| **G1** | `M_after < M_before / 2` | The whole run is throttled ≥ 2×. (This was the original rule — kept, but it is now the *least* sensitive of the four.) |
| **G2** | `P10_after < P10_before / 2` | **Tail binding.** The clip binds ≥ 2× harder on the worst tenth of iterations while the median stays at 1.0 on both sides. This is the case the original median-only rule was blind to, and it is the case that matters: if the clip binds only on death-containing updates, the median sees nothing while the policy's lesson is crushed at exactly the steps this project studies. |
| **G3** | `B_after − B_before > 0.10` (ten percentage points) | **The clip starts binding in a regime where it previously did not.** A statistic that stayed at 1.0 before and dips below 1.0 on 30 % of iterations after is a regime change even if the dips are mild. |
| **G4** | `P10_after < 0.25` (absolute floor, independent of the "before" value) | **A severe absolute throttle that the ratio rules structurally cannot see.** If the clip essentially never bound before (`P10_before = 1.0`), G2 only fires below 0.5 — so a post-fix `P10` of 0.3, a 3.3× throttle on a tenth of all iterations, would pass every ratio test. G4 is the backstop. |

**If none fires — proceed**, and record all eight numbers (`M`, `P10`, `P01`, `B`, before and after) in the Implementation Report. A pass is not silent: see Open question 5 for what a pass still permits.

**If any fires — STOP.** Do not launch training and do not proceed to the [[MODULATION_SITE_REFACTOR]]. Present to the user: which condition(s) fired, all eight numbers, the per-group gradient shares, the trajectory table, and the two named fallbacks — (a) a stationary-scale value loss that divides the value term by a slowly-tracked return scale, keeping raw targets and raw critic outputs; (b) a symlog value target with a symexp'd bootstrap. Both are **design changes**, not bug fixes; neither is pre-approved here and the user decides.

#### Reported alongside the verdict — always, pass or fail

1. **The trajectory, not just the endpoint.** A table of `M`, `P10` and `B` per 100-iteration block across all ~1000 iterations, for both halves. The final-segment verdict is only trustworthy if the trajectory has settled. **If the post-fix `P10` is still trending downward at iteration 1000, report that to the user even on a numeric pass** — a still-falling tail means the 200-iteration segment is not the settled regime and the Gate is extrapolating.
2. **Per-group gradient shares.** Median of `gn_critic / (gn_trunk + gn_actor + gn_critic + gn_modulator)` and the same for each other group, before and after. If the critic goes from a minority to an overwhelming majority of the total, that is the mechanism made visible — and it is the number that tells the user *why* a trip happened, not just that it did.
3. **The `mod_grad_norm` cross-check.** If the legacy `Grad/norm_modulator` probe disagrees with the new `gn_modulator`, report it as a finding (see the note in the instrument section).

#### The post-launch obligation — pre-registered here, not optional later

The Gate answers a question about iterations 1–1000 of one diagnostic. The same question has to be re-asked on the real run, because reward composition drifts as the policy improves and death frequency changes. That is why the instrument is permanent.

**Pre-registered:** the analysis of the **first real training run** launched after this fix must report, from that run's own `grad_diagnostics.csv`, over the **full run**:

- `M`, `P10`, `P01`, `B` — under the same definitions above;
- the four per-group gradient shares;
- the same 100-iteration-block trajectory, so a late-onset regime change is visible;
- an explicit statement of whether the full-run numbers would have tripped **G1–G4** against the Stage A-obs "before" baseline.

If they would have tripped, that is a finding to raise **before** the run's behavioural results are interpreted — because a throttled actor makes "the agent failed to learn nociceptive avoidance" indistinguishable from "the optimizer clipped the lesson away", and the second reading is not a scientific result.

⚠️ **Hand-off note.** `senior-developer` does not write under `docs/experiments/`. This obligation is recorded here; it must be **carried into the experiment design doc** by `experiment-designer` when that run is designed, and executed by `experiment-analyzer` when it is analysed. Whoever launches the first post-fix run is responsible for naming this requirement to both.

---

## Verification

Every row names evidence that can actually come out negative.

| # | Claim under test | Evidence | How it could fail |
|---|---|---|---|
| V1 | Stage A changes nothing numerically | `tests/models/test_lr_param_groups.py::test_equal_rates_match_single_adam` — two identical models, one stepped with the **old** `chain(clip_by_global_norm, adam(lr))` constructed inline in the test, one with the new chain, on the same synthetic batch for 10 steps; assert every parameter is **bitwise** equal | A non-elementwise operation sneaking into the chain; a different Adam epsilon/b1/b2 |
| V2 | Stage A changes nothing end-to-end | 20 iterations of `recurrent_ppo_nmn_het_film_g1.yaml` **on `configs/environment/experiment/basic/04-jump_attack_10x10.yaml`** at a fixed seed, run on a `git worktree` at the **pre-Stage-A** SHA and on the Stage-A tree; the five existing `loss/*` scalars must match to **float equality**. If it mismatches, run the attribution step below before concluding anything | Anything V1 missed that only appears through the real model, the real env and the real optimizer state |
| V3 | The labeller is right — not merely consistent | Deliberately-unequal-rate tests (below). **V1 and V2 cannot detect mislabelling**, because all rates are equal there; a labeller that put every parameter in one group would pass both | A prefix rule capturing `mod_*_ln`; a new head defaulting into `trunk` |
| V4 | No parameter is silently unlabelled | `test_all_params_labelled` constructs a baseline `ActorCriticRNN`, a modulated `ActorCriticRNN` (hierarchical + LayerNorm) and an `ActorCriticMLP`, and asserts `param_group_labels` raises for none of them and that each expected group is non-empty | A renamed attribute; a new module |
| V5 | The new keys cannot be silently defaulted | `test_missing_lr_modulator_raises` / `test_missing_lr_critic_raises` — a config dict without the key must raise `ValueError` from `get_mandatory` | `.get(key, default)` sneaking in |
| V6 | New checkpoints round-trip | Save and restore through `src/utils/checkpoint_restore.py` on the new optimizer, assert model **and** optimizer state come back bitwise equal, and that one further update step from the restored state matches one from the unsaved state | An optimizer-state pytree the serializer cannot represent |
| V7 | Fix 1 is correct (not bit-identical — it deliberately changes behaviour) | The three tests in `test_mc_return_units.py`, **shown failing on the Stage-A tree** and passing after; plus the before/after diagnostic table | An oracle computed with the code under test (guarded: the oracle is a NumPy loop written in the test) |
| V8 | H4's window-edge semantics survive Fix 1 | The seven existing tests in `tests/models/test_mc_window_bootstrap.py` pass unchanged | An edit to `compute_mc_returns` that should not have happened |
| V9 | The clip budget has not silently retuned the actor — **including on the tail** | The Gate's four-condition rule (G1 median, G2 p10 tail, G3 binding frequency, G4 absolute floor), computed **per update** and read from the unsmoothed per-iteration `grad_diagnostics.csv`, over the final 200 of ~1000 iterations | Value gradients dominating everywhere (G1); dominating only on death-containing steps while the median stays at 1.0 (G2/G3); a severe throttle that no ratio rule can see because the clip never bound before (G4). Each is caught, not hidden |
| V9b | The instrument itself is not lying | Algebraic identities on every CSV row: `clip_factor_min ≤ clip_factor ≤ 1`, and `clip_binding_frac > 0` iff `clip_factor_min < 1`; plus `value_target_std ≈ 1.0` pre-fix (targets are z-scored today) | A clip factor computed from the epoch-**mean** norm instead of per update; a wrongly-scoped `max_grad_norm`; the CSV flushing from a stale buffer |
| V10 | No speed regression | `Time/sps_env` over the same ~1000-iteration diagnostic, same model config, **same environment config (`basic/04-jump_attack_10x10.yaml`)**, same node, GPU and seed, at HEAD and after both stages. Per project protocol: > 5 % slowdown warrants discussion, > 15 % blocks merge | Per-leaf `tree_map_with_path` in the update path; the per-group norm reduction on the hot path; **the per-iteration CSV forcing a host sync** — the buffered-flush design exists to prevent exactly this, and V10 is what proves it worked |
| V11 | Full suite green | `pytest tests/` | Anything |

**Deliberately-unequal-rate tests** — `tests/models/test_lr_param_groups.py`:

**Two requirements bind every one of the freeze tests below** (plan-reviewer L3 — write them into the shared helper, do not leave them implicit):

1. **Assert both directions.** Every frozen leaf must be **bitwise unchanged** AND every leaf in the complement must have **changed**. Asserting only the frozen half would pass for a labeller that froze everything.
2. **Assert the gradient is nonzero first.** Before the update, take the gradient of the same loss and assert `optax.global_norm(grads[leaf]) > 0` for every leaf that the test expects to *change*. Without this, "changed" can fail for a reason that has nothing to do with labelling — a saturated or clipped modulator output produces an exactly-zero gradient, so the parameter would not move even under a correct label and a nonzero rate, and the test would report a labelling bug that does not exist. If a leaf legitimately has zero gradient on the synthetic batch, the test must **fail loudly with that diagnosis** ("zero gradient — change the batch"), not silently reinterpret it.

Shared helper sketch: `_assert_only_group_moved(model, rates, expected_frozen_prefixes)` — snapshot params, check grads nonzero on the complement, apply one update, assert bitwise-equal on the frozen set and not-equal on the complement.

- `test_zero_actor_rate_freezes_actor_and_trunk_only` — rates `{trunk: 0, actor: 0, critic: 1e-3, modulator: 1e-3}`; after one update assert `actor_fc1`, `actor_fc2`, `obs_encoder`, `rnn_cell` are **bitwise unchanged** and `critic_fc1`, `critic_fc2`, `modulator` **changed** (with the nonzero-gradient precheck on the latter three). This also pins the documented trunk→`lr_actor` decision.
- `test_zero_critic_rate_freezes_critic_head_only` — the mirror image; assert `critic_fc1`, `critic_fc2` frozen AND trunk + actor + modulator all changed.
- `test_zero_modulator_rate_freezes_modulator_only` — on a modulated model; assert `modulator` frozen AND trunk + actor + critic all changed.
- `test_encoder_layernorms_are_trunk_not_modulator` — asserts `mod_unimodal_ln`, `mod_multimodal_ln`, `mod_flat_ln` label as `trunk`, so the `startswith('mod')` trap is caught if anyone reintroduces it. Strengthen it behaviourally too: under `{modulator: 0, everything else: 1e-3}` the three LayerNorms must **change** (they are trunk), and under `{trunk: 0, everything else: 1e-3}` they must be **frozen**.
- `test_unknown_parameter_name_raises` — a stub module with an unrecognised attribute must raise, proving there is no fallback group.

These observe *which parameters moved*. They do not depend on the labeller being correct, so they are not circular with it.

**If V2 mismatches — the attribution step (plan-reviewer O1).** Bitwise equality of the five losses across two *differently compiled* graphs is an empirical bet, not a theorem: Stage A adds seven outputs to the aux tuple, and XLA is free to fuse the surrounding computation differently, which can change floating-point results at the last bit even when the algebra is identical. The STOP-on-mismatch rule stands — do **not** substitute a tolerance — but before escalating, separate the two candidate causes:

1. Build a third worktree carrying **Stage A minus the instrument**: the learning-rate wiring and the config migration, but *not* the seven new aux outputs and *not* the new logging.
2. Re-run the identical 20-iteration comparison against the pre-Stage-A tree.
3. If Stage-A-minus-instrument **matches** and full Stage A does not, the cause is the instrument changing XLA fusion — a compilation artefact, not an optimizer change. Report it with both numbers and the diff magnitude; the user decides whether to accept it.
4. If Stage-A-minus-instrument **also mismatches**, the optimizer change is implicated and V1's bitwise claim is contradicted at the whole-model level. That is a genuine STOP: report and do not proceed to Stage B.

This step costs one extra worktree build and turns an ambiguous failure into a diagnosed one.

---

## Checkpoints

Ordered. A1–A8 must all be green before B1 begins.

- [ ] **A0** — Record the HEAD commit SHA in the Implementation Report. Run `grep -rn "lr_critic\|lr_modulator\|lr_actor" configs/ | sort` and paste the "before" inventory.
- [ ] **A1** — `src/models/lr_groups.py` created. Print `param_group_labels(nnx.state(model, nnx.Param))` for a baseline model, a modulated hierarchical model and an `ActorCriticMLP`; paste the label→leaf-count table into the report. Confirm the key-path type returned by `tree_map_with_path` on an `nnx.State` matches what `_top_level_name` expects.
- [ ] **A2** — All five tests in `tests/models/test_lr_param_groups.py` pass, including the unequal-rate ones (V3) and the unknown-name raise (V4).
- [ ] **A3** — V1: the bitwise old-vs-new optimizer equivalence test passes.
- [ ] **A4** — 22 config files migrated per the table. Re-run the grep; paste the "after" inventory. Confirm: no `lr_critic: 0.0001` anywhere in `configs/models/`; `ppo.yaml` shows `0.0003`; exactly 20 `lr_modulator` keys, all in `configs/models/recurrent_ppo/`; the archived `testbed_cellC_native_v2.yaml` untouched.
- [ ] **A5** — V5: both missing-key tests raise `ValueError`.
- [ ] **A6** — V6: the checkpoint save/restore round-trip test passes.
- [ ] **A7** — V2: the 20-iteration fixed-seed before/after loss comparison is **exactly equal**. State model config, **environment config (`basic/04-jump_attack_10x10.yaml`)**, node, GPU, seed and both SHAs. If it is not exactly equal, **stop and report** — do not substitute a tolerance — and first run the **attribution step** (Stage-A-minus-instrument worktree, §Verification) so the report says *which* change caused it.
- [ ] **A8** — `CONFIG_CRITICAL_SETTINGS.md` updated (3 registry rows + change-log entry) in the same commit. Re-run `grep -c "agent\." docs/environment/CONFIG_GUIDE.md docs/environment/02_config_schema.md` and record the counts, confirming those two docs still have no `agent:`-block section.
- [ ] **A-obs** — the **~1000-iteration** diagnostic run on `recurrent_ppo_nmn_het_film_g1.yaml` × **`configs/environment/experiment/basic/04-jump_attack_10x10.yaml`**, single fixed seed, one named node + GPU, SHA recorded. `grad_diagnostics.csv` exists and has ~1000 rows. `tmp/..._before.md` written with: `M`, `P10`, `P01`, `B` over the final 200 iterations; the 100-iteration-block trajectory; the four per-group gradient shares; and the seven scale metrics. `value_target_std ≈ 1.0` and the V9b algebraic identities both hold — these confirm the instrument is wired correctly. If either fails, **stop and fix the instrument before measuring anything**.
- [ ] **B1** — `compute_mc_targets_and_advantages` extracted; `train_iteration`'s MC branch calls it; the GAE branch untouched (`git diff` on lines 381-396 must be empty).
- [ ] **B2** — `tests/models/test_mc_return_units.py` written and shown **FAILING on the Stage-A tree via a `git worktree`**. Paste the failure output. A test that cannot fail proves nothing.
- [ ] **B3** — the same three tests pass on the fixed tree.
- [ ] **B4** — V8: all seven existing `test_mc_window_bootstrap.py` tests still pass, unchanged.
- [ ] **B5** — the "after" diagnostic run, **identical in all six pinned settings** (model config, environment config, seed, iterations, node, GPU); `tmp/..._after.md` written; the before/after table filled in. Record `value_target_std` — this is the first in-situ measurement of return σ on `basic/04` (open item O3).
- [ ] **B6** — **The Gate.** Compute `M`, `P10`, `P01`, `B` for both halves over the final 200 iterations. Evaluate all four trip conditions **G1, G2, G3, G4**. If **any** fires, STOP and escalate to the user with the numbers, the shares and the trajectory. Record the verdict, all eight numbers and the trajectory table either way — **including on a pass**. If the post-fix `P10` trajectory is still falling at iteration 1000, say so explicitly even on a pass.
- [ ] **B7** — V10: speed numbers at HEAD vs after both stages, the same ~1000-iteration diagnostic, same node/GPU/seed/model config/environment config. Confirm specifically that the per-iteration CSV did **not** introduce a per-iteration host sync (that is the most likely source of a slowdown here).
- [ ] **B8** — V11: `pytest tests/` green.
- [ ] **B9** — `python scripts/claude/regen_dev_index.py` exits 0. **Leave `docs/develop/INDEX.md` unstaged** — it is dirty from a parallel session. Do not touch or stage `docs/develop/active/issues/KNOWN_BUGS.md`, which carries another session's uncommitted hunks; hand the registry updates to `bug-curator` instead.
- [ ] **B10** — Confirm in the report that the [[MODULATION_SITE_REFACTOR]]'s C0a/C0b fixture capture has **not** started, and that both stages are committed before it does.

**Commits:** two, not one — Stage A (code + 22 configs + `CONFIG_CRITICAL_SETTINGS.md`) and Stage B (trainer + tests). Stage with explicit pathspecs; `git diff` each shared file first and leave any hunk that is not yours unstaged.

---

## Open questions for the user

1. **The Gate's fallback, if it trips.** If the policy's effective update turns out to be throttled by the larger value gradients, which fallback: a stationary-scale value loss, or a symlog value target? Both are design changes; neither is pre-approved here.
2. **The trunk's rate.** The shared trunk is assigned `lr_actor` (documented and test-pinned). Confirm — it is the choice that keeps behaviour identical today, but it is a genuine modelling decision the moment the rates diverge.
3. **`--lr` semantics.** It currently overrides the single rate, so it is specified to override all three. Confirm that is wanted, rather than `--lr` meaning "actor only".
4. **Plain PPO.** Included for `lr_critic` (2 lines of code, 2 config values), excluded for `lr_modulator` and excluded entirely from Fix 1. Confirm.
5. **⚠️ How much policy-throttling is acceptable on a PASS?** *(Raised by plan-reviewer M4 — this one needs an answer before Stage A-obs runs, because the thresholds are pre-registered and cannot be changed after the measurement.)*

   Plain English: this change is billed as a **bug fix**, which normally means "training behaves the same, only correctly". But the Gate as specified lets a real slowdown through. Condition **G1** only stops the work when the typical clip factor *halves* — so a run where every policy update is scaled down by, say, 1.6× would **pass the Gate and be treated as a clean bug fix**. Nobody would be told. That tolerance is a decision the user should own, not a number an implementer picks.

   The same question applies to the other three thresholds, which are equally arbitrary until confirmed: **G2** (tail, halving of the 10th-percentile worst-update clip factor), **G3** (the clip starts binding on 10 percentage points more iterations than before), **G4** (an absolute floor — stop if the worst tenth of iterations is throttled below 0.25, i.e. more than 4×, regardless of the "before" value).

   Three options:

   | Option | What it means |
   |---|---|
   | **(a) Accept as written** | Up to a 2× typical throttle, and up to a 4× tail throttle, pass silently as "no behaviour change". Fastest; the risk is a real change in policy learning speed shipped inside a bug fix. |
   | **(b) Tighten** | Require the clip regime to be *materially unchanged* — e.g. G1 trips at `M_after < 0.9 · M_before`, G2 at `P10_after < 0.8 · P10_before`, G4 floor raised to 0.5. Safest scientifically; most likely to trip and force the design-change conversation in Open question 1. |
   | **(c) Accept explicitly, and say so downstream** | Keep the 2× tolerance, but require the measured throttle to be written into `CONFIG_CRITICAL_SETTINGS.md`'s change-log entry and into the first post-fix run's analysis, so no later reader mistakes "passed the Gate" for "provably identical". |

   If the user does not answer, the implementer does **not** pick — Stage A-obs waits.

**Already decided by the user, recorded here so it is not re-asked:** the diagnostic environment is `configs/environment/experiment/basic/04-jump_attack_10x10.yaml`, single seed (this closes plan-reviewer M3).

---

## Implementation Report

> **Implemented by**: [agent]
> **Date**: [date]

<!-- Must include:
     - HEAD SHA at A0 and the Stage-A SHA used for the B2 worktree
     - before/after `grep` inventories of lr_actor / lr_critic / lr_modulator
     - the A1 label -> leaf-count table for all three model variants
     - the A7 exact-equality result, naming ALL SIX pinned settings (model config,
       environment config, seed, iterations, node, GPU) plus both SHAs; if it mismatched,
       the attribution-step result from the Stage-A-minus-instrument worktree
     - the B2 pre-fix FAILURE output for all three new tests
     - the before/after diagnostic table, with ALL EIGHT gate numbers
       (M, P10, P01, B for each half), the four per-group gradient shares for each half,
       and the 100-iteration-block trajectory table for both halves
     - the B6 Gate verdict: which of G1/G2/G3/G4 fired (or that none did), and an explicit
       statement of whether the post-fix P10 trajectory had settled by iteration 1000
     - the measured `value_target_std` on basic/04 (first in-situ measurement of return
       sigma on this environment — open item O3)
     - the V9b instrument identities (clip_factor_min <= clip_factor <= 1; binding_frac > 0
       iff clip_factor_min < 1) confirmed on the raw CSV
     - speed numbers (B7) with node, GPU, model config, environment config, seed, iteration
       count, and a statement that the per-iteration CSV added no host sync
     - the L1 note: the optimizer-state pytree structure CHANGED (flat 3-tuple vs nested),
       and warm-resuming a pre-Stage-A checkpoint will fail structurally
     - the `grep -c "agent\."` counts proving CONFIG_GUIDE / 02_config_schema stayed out of
       contract (expected 2 and 1 respectively)
     - confirmation that KNOWN_BUGS.md and INDEX.md were not staged -->

## Verification Report

> **Verified by**: [agent]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [one-line summary]

---

## Feedback from plan-reviewer

> **Reviewed by**: plan-reviewer · 2026-09-01 · plan at commit `c62e0eef`, code at `549db684` (branch v3.0)
> **Verdict**: **SOUND WITH CONCERNS** — no Critical findings; four Moderate findings, all on the Gate and the diagnostic-run specification, resolvable by editing this doc before implementation.

**Plain-language summary.** The two repairs themselves are sound: training the value estimator in raw reward units (matching the file's own GAE branch) is the standard formulation and removes the defect at its root, and the learning-rate wiring was verified to be exactly equivalent to today's optimizer. The concerns are all about the safety net around the one acknowledged risk — that the much larger value-gradient could eat the shared gradient-clipping budget and silently slow the policy's learning. The pre-registered STOP gate is the right idea, but as specified it uses a statistic (the median) that cannot see the failure mode it guards against (clipping that binds only on the death-heavy iterations that matter most to this project), and it measures the least representative 200 iterations of a run that will train orders of magnitude longer.

### Independently verified (not just read)

- **The load-bearing optax identity holds.** Installed optax 0.2.6: `adam(lr)` is literally `chain(scale_by_adam(...), scale_by_learning_rate(lr))`, and `scale_by_learning_rate(lr)` with a static float is `scale(-lr)` (`optax/_src/transform.py::scale_by_learning_rate`). Numerically confirmed: 10 update steps of `chain(clip_by_global_norm(0.5), adam(5e-4))` vs `chain(clip_by_global_norm(0.5), scale_by_adam(), per-leaf * -lr)` produce **bitwise-identical parameters**.
- **Plain PPO genuinely has no units bug**: `src/models/ppo_trainer.py:216-221` calls its `compute_mc_returns` with `(rewards, dones, gamma)` only — no critic seed folds into the accumulation. The Fix-1 exclusion is correct.
- **`mod_unimodal_ln` / `mod_multimodal_ln` / `mod_flat_ln` → trunk is correct**: `src/models/recurrent_ppo_network.py:242-245` constructs them as LayerNorms on the task-side encoder pre-activations, independent of the modulator.
- **Config inventory correct**: 23 files declare `lr_critic` (20 rPPO + 2 PPO + the archived testbed), zero `extends:` among the rPPO files, zero existing `lr_modulator`, and the per-file values match the migration table (including the `ppo.yaml` 0.0003 trap).
- **Aux-tuple append is safe**: the only index consumers are `train.py:1780-1784` (`l[1][0..4]`); nothing else in `src/` or `scripts/` reads the tuple.
- **No downstream consumer assumes z-scored critic output**: grep over `scripts/eval/` and analysis code found nothing reading the value head's numeric scale, so Fix 1's side-effect surface is as small as claimed.
- **Sequencing satisfied**: `MODULATION_SITE_REFACTOR.md` (C0b note, ~line 590) requires this fix strictly before C0b or after V4; landing both stages before C0a satisfies it, and that doc already cross-links here.
- **Registry check clean**: KNOWN_BUGS rows A1 (line 86, "wire or remove") and P1 #5 (line 94) are both OPEN and match this plan's premises; no prior fix collides. `compute_mc_returns` is untouched, so the seven `test_mc_window_bootstrap.py` tests (7 confirmed) pass trivially — V8 is sound.
- **The circular-verification escape is genuine**: with one group's rate zeroed, a mislabelled leaf either moves when it must be frozen or freezes when it must move; either way an assertion fails without relying on the labeller. See L3 for a specification tightening.

### Findings

| Sev | Location | Issue | Suggested fix |
|---|---|---|---|
| 🟡 M1 | §The Gate (decision rule) | **The median clip factor cannot see tail binding.** If the clip binds only on death-containing update steps (the learning signal this project studies), median `f` stays 1.0 before and after and the gate passes while the actor's update at exactly those steps is crushed ~25×. Also, the logged `loss/grad_norm` used to compute `f` is an epoch-**mean** per iteration (`train.py:1783`), which smooths the tail further. | Gate on tail statistics too: STOP if p10(`f_after`) < p10(`f_before`)/2 **or** the fraction of iterations with `f < 1` grows by more than a stated amount. State the rule numerically before Stage A-obs runs. |
| 🟡 M2 | §Stage A-obs / §The Gate | **200 iterations is the least representative window, in both directions.** Post-fix, iterations 1–200 are the lifetime maximum of critic error (V≈0 vs targets σ≈23–25), so the value-grad share is transiently maximal — the gate can trip on a self-resolving transient. Conversely, a pass says nothing about iteration 50k after reward composition drifts. Nothing watches after the gate. | Lengthen to ~1000 iterations and evaluate `f` on the final segment, reporting its trajectory; pre-register that the first real run's analysis must report full-run median+p10 clip factor and the per-group norm shares under the same halving rule (the instrument is permanent — use it). |
| 🟡 M3 | §Stage A-obs, V2, V10 | **The diagnostic and speed runs never name the environment config.** Only the model config is pinned; "identical settings" is implicit. The σ≈23–25 magnitudes come from restpremium-era trajectories, while the next experiment runs on `basic/04` — reward composition (death frequency, drive scale) is env-dependent, so the gate's verdict transfers only if measured on the env that will actually be used. | Pin the env config explicitly in Stage A-obs (recommend the coming experiment's `basic/04`), and record it alongside node/GPU/seed/SHA. |
| 🟡 M4 | §The Gate | **A gate pass silently licenses up to a 2× actor-step throttle** (`f_after ≥ f_before/2` proceeds without further ceremony) inside a change billed as a bug fix. That tolerance is a user decision, not an implementer default. | Add to §Open questions: "a pass may still mean the actor's effective step shrank up to 2× — accept?" |
| 🟢 L1 | §Candidate B | "keeps the optimizer-state pytree structurally the same as today" is **false**: old state is `(EmptyState, (ScaleByAdamState, EmptyState))` (adam is itself a chain), new is flat `(EmptyState, ScaleByAdamState, EmptyState)` — verified against optax 0.2.6. Harmless here (all runs retrained; V6 round-trips the new shape), but correct the sentence, and note that warm-resuming any pre-Stage-A checkpoint's optimizer state will fail structurally. | Reword; keep V6 as-is. |
| 🟢 L2 | §Test 3 | "Fit a small value function" is under-specified — a function approximator over unspecified features could miss the 5 % tolerance flakily. | Specify a tabular critic: one free parameter per state, including the window-edge boot state. |
| 🟢 L3 | §Deliberately-unequal-rate tests | Two implicit requirements should be explicit: each test asserts **both** directions (frozen set bitwise unchanged AND complement changed), and asserts every leaf's gradient is nonzero before the update — a zero gradient (e.g. a saturated/clipped modulator output) makes "changed" fail for the wrong reason. | Write both into the test spec. |
| 🟢 L4 | §Docs table, A8 | The recorded grep counts are transposed: actual `grep -c "agent\."` is **2** for `CONFIG_GUIDE.md` and **1** for `02_config_schema.md` (plan says 1 and 2). The re-run instruction already covers it; the exemption conclusion stands. | Fix the numbers. |
| ❓ O1 | V2 / A7 | Bitwise equality of the five losses across **differently compiled graphs** (Stage A adds outputs; XLA may fuse differently) is an empirical bet. The STOP-on-mismatch handling is right; what's missing is attribution. | If V2 mismatches, first build Stage A *minus the instrument* in a worktree to separate "optimizer change" from "instrument changed the graph". |
| ❓ O2 | A1 | The nnx `tree_map_with_path` key-path shape (`.key` vs `.name`) is assumed; the plan itself pins it with checkpoint A1. No action — noted as the one structural assumption Stage A rests on. | — |
| ❓ O3 | §Magnitude | Return σ on `basic/04` is unmeasured (magnitudes come from restpremium recordings). Mitigated: `loss/value_target_std` measures it in situ post-fix; M3 makes the measurement land on the right env. | — |

### Decisions the user should make before implementation (not before launch)

1. The plan's own four open questions (gate fallback, trunk rate, `--lr` semantics, plain-PPO scope) — all well-posed; no objection to the defaults chosen.
2. M4: accept (or tighten) the up-to-2× actor-throttle tolerance implied by a gate pass.
3. M1/M2 gate amendments: tail statistic + longer window + pre-registered post-launch check.
4. M3: which environment config the diagnostic runs use (recommend `basic/04`).

### Cost of being wrong

If the gate's median-statistic hole lets a tail-binding clip regime through, every post-fix run — including the coming single-seed `basic/04` experiment — trains with its policy update crushed at exactly the death-heavy steps this project studies, and "the agent fails to learn nociceptive avoidance" becomes indistinguishable from "the optimizer clipped the lesson away": weeks of runs and a wrong scientific reading. Everything else found here costs at most a rerun of a 20-minute diagnostic or a one-line doc correction.

*Reviewed by: plan-reviewer*

---

## Response to plan-reviewer — disposition of every finding

> **Amended by**: senior-developer · 2026-09-01 · plan-only amendments, no code exists yet

All ten findings are **accepted**. None was disputed. Summary of where each was addressed:

| Finding | Disposition | Where in this doc |
|---|---|---|
| 🟡 **M1** — median cannot see tail binding; `loss/grad_norm` is a pre-smoothed epoch mean | **Accepted, largest amendment.** The Gate now has four OR-ed trip conditions (G1 median, G2 tenth-percentile tail, G3 binding frequency, G4 absolute floor). The clip factor is computed **per update** (`f(mean ‖g‖) ≠ mean f(‖g‖)`), and a new **per-iteration, unsmoothed CSV** replaces the 100-iteration WandB means as the Gate's data source | §The Gate → "The amended pre-registered decision rule"; §`train.py:1779-1804` — the clip statistics; V9 / V9b |
| 🟡 **M2** — window unrepresentative in both directions; nothing watches after the gate | **Accepted.** ~1000 iterations, verdict on the final 200, trajectory reported in 100-iteration blocks, "still falling → tell the user even on a pass". Plus a pre-registered **post-launch obligation** on the first real run, with an explicit hand-off to `experiment-designer` / `experiment-analyzer` since `senior-developer` cannot write under `docs/experiments/` | §Stage A-obs → "Why ~1000 iterations…"; §The Gate → "The post-launch obligation" |
| 🟡 **M3** — environment config unpinned | **Accepted; answered.** `configs/environment/experiment/basic/04-jump_attack_10x10.yaml`, single seed (user's decision). A six-row pinned-settings table now governs both halves of every comparison, and env config + node + GPU + seed + SHA are recorded per result. Open item O3 (σ unmeasured on this env) is carried as an explicit reporting requirement | §Stage A-obs → "The run — every setting pinned"; V2, V10, checkpoints A7 / A-obs / B5 / B7 |
| 🟡 **M4** — a pass silently licenses a 2× throttle | **Accepted.** Surfaced as **Open question 5** with three named options (accept / tighten / accept-and-disclose), stated in plain language, and made **blocking**: the thresholds are pre-registered, so the user answers before the "before" run | §Open questions 5; §Sequencing item 5; the blocked-status callout at the top |
| 🟢 **L1** — optimizer-state pytree structure claim false | **Accepted, claim retracted.** The false sentence is replaced by the verified structures (`(EmptyState, (ScaleByAdamState, EmptyState))` → flat 3-tuple) plus the warm-resume consequence, and the Implementation Report must restate it | §Design decision → Candidate B correction block |
| 🟢 **L2** — "small value function" under-specified | **Accepted.** Test 3's critic is now specified as **tabular**: one free parameter per state in a `(T+1, B)` array, index `T` being the window-edge boot state that closes the loop | §Test 3 |
| 🟢 **L3** — freeze tests need both directions + nonzero-gradient precheck | **Accepted.** Two binding requirements written into a shared helper, with an explicit "fail loudly with that diagnosis" rule for a legitimately zero gradient. The LayerNorm test is strengthened behaviourally as well as by label | §Deliberately-unequal-rate tests |
| 🟢 **L4** — grep counts transposed | **Accepted.** Corrected to CONFIG_GUIDE = 2, 02_config_schema = 1; the exemption conclusion is unchanged and the re-run instruction stands | §Docs to update; checkpoint A8; Implementation Report checklist |
| ❓ **O1** — bitwise equality across differently-compiled graphs is a bet | **Accepted.** STOP-on-mismatch retained; a four-step **attribution procedure** added (build Stage-A-minus-instrument in a worktree) so a mismatch is diagnosed as "instrument changed XLA fusion" vs "optimizer change broke it" rather than escalated ambiguously | §Verification → "If V2 mismatches — the attribution step"; checkpoint A7 |
| ❓ **O2** — nnx key-path shape assumed | **No change needed**, as the reviewer noted: checkpoint A1 already pins it before anything depends on it | checkpoint A1 |
| ❓ **O3** — return σ unmeasured on `basic/04` | **Accepted as a reporting requirement.** `value_target_std` from the post-fix run is the first in-situ measurement; the plan now states that a σ near 1 would materially weaken the units argument and must be raised to the user | §Stage A-obs (O3 callout); checkpoint B5; Implementation Report checklist |

Not revisited, per the reviewer's own independent verification: the optax identity, the plain-PPO exclusion from Fix 1, the `mod_*_ln` → trunk labelling, the stage sequencing, the config inventory, and the Fix-1 test design.

*Amended by: senior-developer*
