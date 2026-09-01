---
title: "Modulation Site Refactor — uniform FiLM at selectable sites (RecurrentPPO)"
topic: neuromodulation
status: active
created: 2026-08-31
last_updated: 2026-09-01
---

# Modulation Site Refactor — uniform FiLM at selectable sites (RecurrentPPO)

> **Status**: PLANNED — **Part A approved and plan-review-cleared** (all five `plan-reviewer` findings addressed, incl. the 🔴 Critical; see [Response to plan-reviewer](#response-to-plan-reviewer-2026-08-31)). **Part B remains a proposal, NOT approved.**
> **Opened**: 2026-08-31 · **Revised**: 2026-08-31 (post-review)
> **Related**: [[NEUROMODULATION_ALGORITHM]] · [[NMN_METRICS_REFERENCE]] · [[NMN_ARCHITECTURE_REVIEW]] · [[FILM_MODULATION_PLAN]] · [`docs/project/ideas/20260805_film_rl_context_dependent_policy_discussion.md`](../../../project/ideas/20260805_film_rl_context_dependent_policy_discussion.md) §6 · [`modulation_in_rl_lit_review.md`](../../../project/references/modulation_in_rl/modulation_in_rl_lit_review.md) §8, §12

---

## Context

The agent in this project carries a small second network — the **neuromodulator** — that reads the agent's senses and continuously re-tunes the main policy network, the way a brain chemical like acetylcholine re-tunes cortex. Today that re-tuning happens through **four different mechanisms** at four different places: a multiply-and-add on the sensory encoder, a bias slipped into the memory cell's forget/keep gate, a scalar that divides the policy's action scores, and nothing at all on the decision-making layers. Four mechanisms means every experimental comparison confounds *where* the modulator acts with *how* it acts.

This plan makes all of them **the same operation** — feature-wise scale-and-shift, i.e. multiply each neuron by a learned gain and add a learned offset (the technique called **FiLM**) — and makes each place it can act **switchable from config**. It also adds two places the modulator currently cannot reach at all: the **actor** (the layers that choose the action) and the **critic** (the layers that estimate how good the situation is). That follows the closest published precedent, a quadruped-skateboarding controller called **PAPL**, which modulates every layer of both actor and critic; and a fixed-wing-aircraft study (**Marquis & Farhood**) which is the only controlled actor-vs-critic comparison in our corpus and found that FiLM-modulating the critic cut tracking error by roughly half.

The point of the change is a planned **two-factor experiment**: *what the modulator reads* × *what the modulator writes to*. This plan delivers the "writes to" half (**Part A**, approved). The "reads" half is sketched as a separable, **not-yet-approved** **Part B** at the end for the user to accept or defer.

Two hard promises constrain everything below. The **unmodulated baseline must stay bit-identical** — it is the project's true control, and if it drifts, every past comparison silently becomes invalid. And **no behaviour of an existing config may change silently**: where this refactor would flip a default, the config must be edited to say what it wants out loud, or loading it must fail.

---

## Analysis

### A. What exists today

| Site | Mechanism today | Where in code | Uniform with the others? |
|---|---|---|---|
| Encoder — unimodal stage | FiLM `relu(γ⊙z + β)` on pre-activations | `recurrent_ppo_network.py:161-171` | ✅ yes (reference operator) |
| Encoder — multimodal hub | FiLM `relu(γ⊙z + β)` on pre-activations | `recurrent_ppo_network.py:179-189` | ✅ yes |
| Task GRU (memory) | **bespoke** — additive bias into the update-gate pre-activation | `modulated_gru_cell.py:66-70`, called at `recurrent_ppo_network.py:338` | ❌ no |
| Policy logits | **bespoke** — divide by a bounded scalar temperature | `recurrent_ppo_network.py:345`, `neuromodulator.py:172-173` | ❌ no |
| **Actor MLP** | **absent** | — | ❌ not modulated |
| **Critic MLP** | **absent** | — | ❌ not modulated |

The modulator itself (`neuromodulator.py:41-186` `NeuromodulatorRNN`) is one GRU over the full observation feeding six linear heads. Every head already shares one machinery: emit `ceil(target_dim / grouping_size)` values, expand with `jnp.repeat` across contiguous neuron blocks, add a full per-neuron learned baseline (`neuromodulator.py:145-155`). **That machinery is exactly what the new sites need** — the refactor reuses it unchanged, it does not invent a second path.

### B. One convenient fact that makes this cheap

Every modulation target in `ActorCriticRNN` is the **same width**, `hidden_size` (128 in every current config):

- encoder unimodal output → `hidden_size` (`recurrent_ppo_network.py:92`)
- multimodal hub output → `hidden_size` (`:97`)
- task GRU hidden → `hidden_size` (`:252-254`)
- `actor_fc1` output → `hidden_size` (`:257`)
- `critic_fc1` output → `hidden_size` (`:261`)

So a **single global `grouping_size`** applies cleanly to every head, `num_groups_hidden = ceil(hidden_size / grouping_size)` is reused verbatim, and no per-site grouping key is needed. **Recommendation: keep `grouping_size` global** (the user's default), and this is not a compromise — it is exactly right here. If a future config ever gives the actor a different width, that assumption breaks; the plan adds a construction-time assertion so it breaks loudly rather than silently mis-broadcasting.

### C. "Every hidden layer" is currently exactly one layer per head

`ActorCriticRNN` hard-codes a two-layer actor (`actor_fc1` → `actor_fc2`) and a two-layer critic. `actor_fc2` produces the logits and `critic_fc2` produces the value scalar, so **the only hidden layer in each head is `actor_fc1` / `critic_fc1`**. A1 and A2 therefore add **one FiLM site each**, not N.

Related, worth stating but **out of scope**: the config keys `agent.actor_fc_layers` / `agent.critic_fc_layers` (present in all 12 NMN configs, e.g. `recurrent_ppo_nmn_film_g1_tempceil5.yaml:37-38`) are **read by nobody** on the RecurrentPPO path — `ActorCriticRNN` ignores them entirely. This is already recorded as a known config-boundary trap (B1–B4 in the bug registry). Do not fix it here; mentioning it so the implementer is not surprised that "every hidden layer" resolves to one.

### D. The RNN "activation" site must not touch the carry

The task GRU produces `h_new` and emits `x_h`, which are **the same array** (`modulated_gru_cell.py:78` returns `h_new, h_new`). Applying FiLM to both would multiply the recurrent carry state — precisely the 🔴 Critical "double-gating the GRU carry state" pathology that [[NEUROMODULATION_ALGORITHM]] §5.1 identifies and that the gate-bias mechanism was invented to avoid (it destroys the GRU's additive gradient highway).

**Therefore: the `"activation"` mechanism modulates the emitted output only; the carry passed to the next timestep is left untouched.** This is a load-bearing constraint, not a style preference, and it gets its own regression test.

It also means one small, deliberate deviation from "the same operator as the encoder sites": the encoder applies `relu(γ⊙z + β)` because `z` is a pre-activation, whereas the GRU output is already the cell's own activation, so the RNN site applies `γ⊙x_h + β` **with no following nonlinearity**. Adding a `relu` there would change what the downstream layers receive relative to the baseline and is not what any FiLM paper does at a recurrent output.

### E. Pre-activation vs post-activation for actor/critic — resolved to pre-activation

The user's wording ("at the output of every hidden layer") admits both readings. Resolved to **pre-activation** — `a_h = activate(γ_a ⊙ actor_fc1(x_h) + β_a)` — for two independent reasons: it is what the existing encoder sites do (`:164`, `:168`, `:182`, `:186`), which the user made a hard constraint; and it is what PAPL does (`h_ℓ = σ(γ_ℓ ⊙ z_ℓ + β_ℓ)`, ideas doc §6). Flagged here so the user can overrule before code is written.

### F. Evidence base for adding actor + critic modulation

- **PAPL** (ideas doc §6; lit review §8) modulates every layer of both actor and critic, pre-activation, with no normalisation in the modulation path. Direct precedent for A1/A2.
- **Marquis & Farhood** (lit review §12.3) is the corpus's only controlled actor-vs-critic ablation under PPO. **FiLM + conditioned critic cut errors ~42–57 % across four metrics**; LoRA + conditioned critic roughly *doubled* them. Two conclusions the plan inherits: critic modulation is mechanism-dependent, and **for FiLM specifically the evidence is positive**.
- **PAPL's precondition for critic modulation IS met here** (ideas doc §6.2, corrected in `c36a6355`). PAPL's stated rule is *modulate the critic when the reward itself is conditioned on the modulating variable*. Our reward is exactly that: with `use_homeostatic_reward: true` (`configs/environment/default.yaml:181`, live in every current config) the per-step reward is the **reduction in homeostatic drive**, `r_t = D(s_{t-1}) − D(s_t)`, where `D` is the L2 distance of `(satiation, injury)` from `(setpoint, 0)` (`src/environment/core.py:49-53`, `:728-731`). Injury is an argument of the reward and healing is rewarded directly. The condition is met *more* strongly than a merely additive injury term would give, because `∂D/∂injury = injury/D` — the marginal value of healing rises with injury **and** depends on satiation, so hunger and injury are coupled inside the value function rather than separable. **Critic modulation is licensed on PAPL's own criterion.**

  > ⚠️ **An earlier revision of this plan claimed the opposite** ("our reward is not a function of injury"). That was wrong; it was inherited from a since-corrected source doc and caught by `plan-reviewer`. It is recorded here so the false claim is not re-derived from an older copy of this file.

- **The residual caveats that DO survive** — these, not the retracted one, are what the experiment write-up should carry:
  1. **Conditioner form.** PAPL's modulating variable is an **open-loop clock** — perfectly predictable, so its modulator has a trivially learnable signal. Ours is **contingent sensed injury**, and it arrives through an alpha-kernel smoothing delay (`injury_smoothing_duration: 3`, no instantaneous leak), so the modulator **cannot react on the step the damage occurs**. That is a real disanalogy and it is about *observability of the conditioner*, not about the reward.
  2. **Sign is mechanism-dependent.** Marquis shows conditioning the critic helps or hurts depending on *how* it is conditioned — positive for FiLM under PPO (error cut ~40–50 %), negative for LoRA. So "critic modulation works" is not transferable as a bare claim; it transfers only for the FiLM mechanism this plan implements.

- **Reader's footgun in the reward code** (confirmed independently by `plan-reviewer`): two different "drive" quantities live in the same function. `calculate_drive` (`core.py:49-53`) is the **unnormalised L2 norm** that actually feeds the reward. `drive_hunger` / `drive_injury` (`core.py:725-726`) are **normalised, logging-only** quantities (sum-of-squares form), and their inline comment sits *directly above* the reward computation. Anyone skimming that block to learn what the reward is will read the wrong function. Do not "fix" this here — it is noted so the next reader of §F does not repeat the retracted claim.
- **Marquis §V-C** additionally offers a Lipschitz-bound diagnostic (product of the modulator's weight-matrix spectral norms) that tracked stability across their variants, with spectral normalisation as the remedy. **Explicitly out of scope here** — noted as a candidate follow-up metric so it is not silently forgotten.

### G. Known-bug collisions (from the bug registry)

Four recorded items bear directly on this work:

1. **Finding A — hand-built `modulation_config` whitelists (FIXED once, at one site).** The deprecated `evaluation.py` used to hand-build `modulation_config` from a stale key list and crashed `KeyError('memory_clip')` on every `recurrent_ppo_nmn_*` checkpoint; fixed in `2ad9104`, pinned by `tests/scripts/test_evaluation_model_rebuild.py`. **Adding any new mandatory key repeats this bug at every hand-built rebuild site.** This plan therefore enumerates all six construction sites and moves validation *into* `ActorCriticRNN`.
2. **`save_snapshot.py` is an unrecorded escapee of that same fix.** At repo root, it hand-builds a whitelist missing `memory_clip` **and** passes no `encoding_config`, so it raises `ValueError("Strict Config: encoding_config is required…")` on *every* config today, modulated or not. **Pre-existing and already broken — this plan does not fix it** (out of scope, per "don't fix pre-existing dead code unless asked"); it is flagged for the user and should get a registry row from `bug-curator`.
3. **The `ModulatedGRUCell` init confound (OPEN, Med).** With modulation on, the memory cell is swapped for a hand-built `ModulatedGRUCell` with different initialisation and gate polarity than the baseline's Flax `nnx.GRUCell` (`recurrent_ppo_network.py:249-254`), so *every* modulated-vs-baseline comparison carries an initialisation confound. **This refactor incidentally removes that confound for most new arms** — see decision D6 below.
4. **Unknown `modulation.type` strings silently build a Multiplicative modulator** (no whitelist, config-boundary trap B1). The new `rnn_mechanism` selector must **not** repeat this: it gets an explicit whitelist that raises on anything else.

---

## Implementation Plan

### Design

#### Decision table

| # | Decision | Rationale |
|---|---|---|
| **D1** | Site selection is four independent booleans under `modulation.sites`: `encoder`, `rnn`, `actor`, `critic`. | The experiment needs each site alone and in combination; four booleans give all 15 non-empty combinations with no enum explosion. `encoder` stays **one** flag covering both the unimodal and multimodal stages — that pair is the existing atomic "Injection A", and splitting it doubles the factor levels with no hypothesis attached. Splitting later is a one-line change. |
| **D2** | All sites reuse the existing `_get_signal` machinery verbatim: `ceil(hidden_size/grouping_size)` groups → `jnp.repeat` → `+ per-neuron baseline`. | The user's hard constraint; also means no new numerical behaviour to validate. |
| **D3** | `grouping_size` stays a single global value. | All five targets are exactly `hidden_size` wide (Analysis §B). No reason found to split it. Guarded by an assertion so a future width change fails loudly. |
| **D4** | Actor/critic FiLM is applied **pre-activation**. | Matches the encoder sites (the stated constraint) and PAPL (Analysis §E). |
| **D5** | The RNN `"activation"` site modulates the **emitted output only**, never the carry, and applies **no nonlinearity** after FiLM. | Avoids the 🔴 double-gating pathology of §5.1 (Analysis §D). |
| **D6** | `ModulatedGRUCell` is constructed **only** when `sites.rnn and rnn_mechanism == "gate_bias"`. Otherwise the modulated path uses the same plain `nnx.GRUCell` as the baseline. | Today *any* modulation swaps the cell, importing a known initialisation confound into every comparison (Analysis §G3). Under this rule, encoder-only / actor-only / critic-only / `activation`-mechanism arms become architecturally identical to the baseline except for the modulation itself — which is exactly what a clean 2-factor design requires. Legacy `gate_bias` configs are unaffected. |
| **D7** | Disabled sites emit `None` in `ModulatorOutput`, not zeros/ones. | `mod_info` is stored per-step per-env inside the rollout buffer (`recurrent_ppo_trainer.py:292`), so always-on arrays would add six `(T, num_envs, 128)` float arrays of pure padding. `None` is a valid empty pytree under `vmap`/`scan` — the trainer already relies on exactly this for `mod_info=None` on the unmodulated path. It also keeps fake constants out of WandB. |
| **D8** | The **field set** of `ModulatorOutput` is fixed at construction and never varies at runtime; the model keeps returning a **4-tuple on both paths**. | The trainer's structure-agnostic helpers (`_h_vmap_axes`, `_h_reset_on_done`, `_h_get_first_timestep`) and all three call sites depend on this symmetry; a config-varying dict would break them and reintroduce a recompile trigger. |
| **D9** | Temperature moves to a nested `temperature: {enabled, clip}` block; the legacy flat `temp_clip` key becomes a **hard error**. | A3 flips a default, and the project forbids silent behaviour change. A config still carrying `temp_clip` would otherwise load happily with temperature *off* — including files literally named `…_tempceil5.yaml`. Failing loudly with a migration message is the only acceptable option. |
| **D10** | Existing NMN configs are migrated with `temperature.enabled: **true**` and their current ceiling. Default-off applies to **new** configs. | Those 12 files back completed and in-flight studies with published analyses; turning their temperature off would change what the file means without changing its name. Explicit key everywhere, no silent change in either direction. |
| **D11** | New head *attribute names and construction order* are chosen so the legacy-equivalent setting produces a **byte-identical parameter tree and an identical RNG draw sequence**. | Makes existing NMN checkpoints restore exactly, and makes a fresh run from a fixed seed bit-identical pre/post refactor — which is the strongest available evidence that the refactor is behaviour-preserving. |
| **D12** | `rnn_mechanism` gets an explicit whitelist; anything outside `{"activation", "gate_bias"}` raises. | Do not repeat config-boundary trap B1 (unknown `modulation.type` silently building a Multiplicative modulator). |
| **D13** | A config with `type` set but **all four sites false and temperature disabled** raises at construction. | Otherwise the modulator GRU runs every step, consumes memory and receives no gradient — dead weight with no error, the same class of failure as the fixed LSTM+`z_memory` bug. |
| **D14** | The blanket `modulation + rnn_type == "LSTM"` guard (`recurrent_ppo_network.py:220-226`) is **left exactly as-is**. | Narrowing it to "only when `gate_bias`" is defensible but is unrequested scope, and no config in the repo uses LSTM with modulation. Explicit non-goal. |
| **D15** | `DreamerNeuromodulatorRNN` (same file, `neuromodulator.py:203-379`) is **not touched**. | Dreamer is a separate, archived-config stack. Shared file, separate class. |
| **D16** | **Archived NMN runs stop being re-evaluatable, and that is intended.** No compatibility shim, no translation layer at the eval boundary, no manual saved-config migration procedure, no extension of `--agent-config` to the other tools. The D9 hard error is the whole response. | **User decision, 2026-08-31** (see "Archived-run evaluation" below for the reasoning and the blast radius). |

#### The config surface

```yaml
agent:
  modulation:
    # --- unchanged keys ---
    type: "FiLM"
    mod_hidden_size: 16
    grouping_size: 1
    percept_bias_init: 3.0
    memory_bias_init: 0.0
    memory_clip: [-2.0, 2.0]

    # --- NEW: what the modulator WRITES TO (A1, A2, site selection) ---
    sites:
      encoder: true      # unimodal encoder + multimodal hub (existing Injection A)
      rnn:     true      # the task GRU
      actor:   false     # NEW — FiLM on actor_fc1's pre-activation
      critic:  false     # NEW — FiLM on critic_fc1's pre-activation

    # --- NEW: HOW the rnn site is modulated (A4) ---
    rnn_mechanism: "activation"   # "activation" (default for new configs) | "gate_bias"

    # --- NEW: temperature is now opt-in (A3) ---
    temperature:
      enabled: false
      # clip: [0.5, 5.0]     # required ONLY when enabled: true
```

Semantics, stated so the implementer has no room to guess:

- `sites.*` — four booleans, **all four mandatory** whenever `type` is non-null. `false` is a legal value, so `get_mandatory`'s null check behaves correctly (it rejects `None`, not `False`).
- `rnn_mechanism` — mandatory whenever `type` is non-null (read even when `sites.rnn` is false, so that flipping the site on cannot silently pick a mechanism nobody chose). Whitelisted (D12).
- `temperature.enabled` — mandatory whenever `type` is non-null.
- `temperature.clip` — mandatory **iff** `temperature.enabled` is true; must be absent-or-ignored otherwise. Conditional-mandatory is deliberate: forcing a dead `clip` into every temperature-off config would be noise.
- `memory_bias_init` / `memory_clip` — still mandatory (unchanged), but now only *used* when `sites.rnn and rnn_mechanism == "gate_bias"`. Left mandatory rather than made conditional, to keep this refactor's config diff to strictly additive-plus-temperature.

#### New mandatory config keys

| YAML path | Type | Legal values | Required when |
|---|---|---|---|
| `agent.modulation.sites.encoder` | bool | `true` / `false` | `agent.modulation.type` non-null |
| `agent.modulation.sites.rnn` | bool | `true` / `false` | `agent.modulation.type` non-null |
| `agent.modulation.sites.actor` | bool | `true` / `false` | `agent.modulation.type` non-null |
| `agent.modulation.sites.critic` | bool | `true` / `false` | `agent.modulation.type` non-null |
| `agent.modulation.rnn_mechanism` | str | `"activation"` \| `"gate_bias"` | `agent.modulation.type` non-null |
| `agent.modulation.temperature.enabled` | bool | `true` / `false` | `agent.modulation.type` non-null |
| `agent.modulation.temperature.clip` | `[float, float]` | `[lo, hi]`, `lo < hi` | `temperature.enabled: true` |

**Removed key:** `agent.modulation.temp_clip` — its presence is now a **hard `ValueError`** with a migration message (D9).

**Configs with `type: null` need no edit at all.** That is: `recurrent_ppo.yaml`, `recurrent_ppo_gae.yaml`, `recurrent_ppo_{XS,S,M,L,XL}.yaml`, `recurrent_ppo_nmn_het_unmod.yaml`. `train.py:1138-1140` collapses them to `modulation_config = None` before construction, so no new key is ever read.

#### Migration for the 12 existing NMN configs

Every file matching `configs/models/recurrent_ppo/recurrent_ppo_nmn_*film*.yaml` gets the same edit — replace the flat `temp_clip` line with the new block, preserving the file's own ceiling:

```yaml
  modulation:
    type: "FiLM"
    mod_hidden_size: 16
    grouping_size: 1                 # unchanged per file (1/2/4/8/16/32/64/128)
    percept_bias_init: 3.0
    memory_bias_init: 0.0
    memory_clip: [-2.0, 2.0]

    # MIGRATION 2026-08-31 (MODULATION_SITE_REFACTOR): sites + rnn_mechanism became
    # mandatory; `temp_clip` became `temperature.{enabled,clip}`. Values below reproduce
    # this file's pre-refactor behaviour EXACTLY — this is a no-op migration, not a retune.
    sites:
      encoder: true
      rnn:     true
      actor:   false
      critic:  false
    rnn_mechanism: "gate_bias"
    temperature:
      enabled: true
      clip: [0.5, 5.0]               # <- the file's OWN former temp_clip; do not homogenise
```

Per-file ceilings to carry across (read each file, do not assume):

| Config | former `temp_clip` |
|---|---|
| `recurrent_ppo_nmn_het_film_g1.yaml` | `[0.5, 3.0]` |
| `recurrent_ppo_nmn_het_film_g1_tempceil10.yaml` | `[0.5, 10.0]` |
| `recurrent_ppo_nmn_film_g1_tempceil5.yaml` | `[0.5, 5.0]` |
| `recurrent_ppo_nmn_film_g{1,2,4,8,16,32,64,128}_screen.yaml` | `[0.5, 10.0]` |
| `recurrent_ppo_nmn_film_g32_screen_gae.yaml` | `[0.5, 10.0]` |

(Values above were read off the files on 2026-08-31 via `grep -m1 '^    temp_clip'`; re-read rather than trusting this table if the files have moved since.)

Each edited file's header comment must gain a one-line migration note so a future reader does not mistake the new block for an experimental change.

#### Checkpoint compatibility — stated plainly

- **Unmodulated baseline checkpoints**: unaffected. No modulator exists in either the old or the new tree.
- **Existing NMN checkpoints, loaded with their migrated config**: **restore exactly**, by construction (D11). The legacy-equivalent setting builds precisely `head_unimodal(+_add)`, `head_multimodal(+_add)`, `head_memory`, `head_action` and the same five baselines, in the same order, with the same shapes and the same `nnx.Rngs` consumption — and keeps `ModulatedGRUCell` because `rnn_mechanism` is `gate_bias`. This is a *verified claim, not a hope*: it is Checkpoint C4 below.
- **Any new configuration** — actor or critic enabled, `rnn_mechanism: "activation"`, a site turned off, or temperature disabled — produces a **different parameter tree, and old checkpoints will not restore into it.**
- **Mitigation: none warranted, and here is why.** Those configurations are new architectures; there is no prior run to resume, and no scientific claim depends on continuity with one. Writing a checkpoint-remapping shim would be code with exactly zero callers. The one thing that *is* warranted is a loud failure rather than a silent partial restore — and that already exists: the Finding-L3 fix (`2ad9104`) made restore assert completeness instead of quietly keeping randomly-initialised weights. Confirm that assertion still fires (Checkpoint C5).

#### Archived-run evaluation — a decided policy, not an open risk (D16)

**The problem, stated plainly.** All three evaluation tools rebuild the model from the **run's own saved copy** of the agent config (`models/config.yaml` inside the run directory) — not from the maintained files under `configs/models/`. See `evaluation.py:235`, `scripts/eval/eval_rollout.py:1064-1071`, `scripts/eval/traj_collect/collect_trajectories.py:693`. Every archived NMN run's saved copy still carries the flat `temp_clip` key and none of the new mandatory keys. So after this refactor, re-running an evaluation, a dwell-history pass, or a trajectory collection over an archived NMN run **fails immediately** with the D9 migration error. Only `eval_rollout.py` has an `--agent-config` override; the other two have none.

**Decision (user, 2026-08-31): accept it. The hard failure is correct and intended.**

**Rationale — this is a guard rail, not a defect.** Once the modulation architecture has changed underneath a checkpoint, re-evaluating that checkpoint produces numbers that are **not comparable to anything**: not to the run's own original results (different architecture), and not to post-refactor runs (different training). A refusal to load is therefore the *desired* behaviour — it makes it structurally impossible to silently mix pre- and post-refactor results in one table. The correct response to the error is **re-training under the new architecture, not re-evaluating the old checkpoint.**

**Blast radius — narrower than it first looks.** Only archived **NMN** runs are affected. Every run whose agent config has `modulation.type: null` re-evaluates completely untouched, because all new keys are gated on `type` being non-null and `train.py:1138-1140` collapses null to `modulation_config = None` before construction, so not one new key is ever read. Concretely, that covers **the sensor-ladder, directional-sensor, rest-premium and dwell studies** — the bulk of what actually gets re-analysed. (Same reasoning as "Configs with `type: null` need no edit at all" above.)

**Implementation requirement that follows.** The D9 error must be **self-explaining about the cause and the remedy**, not merely name a missing key. When validation rejects a saved config carrying legacy flat `temp_clip`, the message must say that the run **predates the modulation-site refactor**, that its architecture no longer exists in the code, and that **re-training — not re-evaluation — is the correct response**. The exact wording is in File Changes §2.

**Explicitly NOT done** (all rejected by the same decision): no `--agent-config` on `evaluation.py` or `collect_trajectories.py`; no read-only flat-`temp_clip` → legacy-block translation at the eval boundary; no documented manual saved-config migration procedure. Adding any of them would re-open the door this decision deliberately closes.

#### JAX / Flax NNX staticness — why nothing recompiles per step

The site enables and the mechanism selector are **plain Python attributes on an `nnx.Module`, never traced values**. Under `nnx.split`, non-`Variable` attributes land in the **graphdef**, which is the static half of the jit cache key. They are therefore resolved once at trace time and are constant for the lifetime of a model instance.

This is not a new pattern — it is exactly how `self.modulation_type` (a string, branched on at `:139/:143/:161/:165/:179/:183`), `self.rnn_type`, and `self.mode` already behave, and how `return_mode` is handled in the trainer with an explicit comment saying so (`recurrent_ppo_trainer.py:196-198`). One compile per distinct model structure; zero per-step recompiles.

Two concrete requirements follow:

1. **Store the enables as four separate scalar bool attributes** (`self.site_encoder`, `self.site_rnn`, `self.site_actor`, `self.site_critic`), **not** as a Python `dict` attribute. A dict on an `nnx.Module` is traversed by the graph machinery as a container; a bare bool is unambiguously static. Same for `self.rnn_mechanism` (a `str`) and `self.temperature_enabled` (a `bool`).
2. **`initial_state` structure is invariant under site toggling.** It stays `(task_h, mod_h)` whenever `modulation_enabled`, and a bare array otherwise — regardless of which sites are on, because the modulator GRU always runs. Do not make `mod_h` conditional. Verified by test (C7).

### File Changes

#### 1. `src/models/neuromodulator.py` — `NeuromodulatorRNN` only (lines 31–186)

**`ModulatorOutput` (lines 31–38)** — extend. Field order below is chosen for **readability only** (sites grouped in data-flow order: encoder → rnn → actor → critic → temperature); nothing depends on it. An earlier draft justified the ordering as protecting positional readers — that rationale was wrong and is withdrawn: `z_memory` and `temperature` both move index regardless, and `plan-reviewer` grepped the tree and confirmed **every** access is by attribute name. The implementer is free to reorder if a different grouping reads better.

```python
# AFTER:
class ModulatorOutput(NamedTuple):
    """Output from the neuromodulator's branched heads.

    A field is None when its site is disabled. None is a valid empty pytree,
    so vmap/scan carry it without allocating anything (see plan D7/D8).
    The field SET is fixed at construction and never varies at runtime.
    """
    z_unimodal: Any          # encoder stage 1 gain (γ);  None if sites.encoder is False
    z_unimodal_add: Any      # encoder stage 1 shift (β)
    z_multimodal: Any        # encoder stage 2 gain (γ)
    z_multimodal_add: Any    # encoder stage 2 shift (β)
    z_rnn: Any               # NEW: task-GRU output gain (γ); None unless rnn_mechanism == "activation"
    z_rnn_add: Any           # NEW: task-GRU output shift (β)
    z_memory: Any            # gate-bias signal; None unless rnn_mechanism == "gate_bias"
    z_actor: Any             # NEW: actor hidden gain (γ);  None if sites.actor is False
    z_actor_add: Any         # NEW: actor hidden shift (β)
    z_critic: Any            # NEW: critic hidden gain (γ); None if sites.critic is False
    z_critic_add: Any        # NEW: critic hidden shift (β)
    temperature: Any         # None unless temperature.enabled
```

**`__init__` (lines 59–127)** — new keyword-only args `sites: dict`, `rnn_mechanism: str`, `temperature_enabled: bool`; `temp_clip` becomes optional-when-disabled. Head construction becomes conditional. **Construction order is load-bearing (D11)** — existing heads keep their exact current positions in the `nnx.Rngs` stream, and every new head is created strictly *after* all of them:

```
1. self.gru                                            (unchanged, always)
2. head_unimodal   / head_unimodal_add                 iff sites["encoder"]
3. head_multimodal / head_multimodal_add               iff sites["encoder"]
4. head_memory                                         iff sites["rnn"] and rnn_mechanism == "gate_bias"
5. head_action                                         iff temperature_enabled
--- everything below is NEW; it must come after (5) ---
6. head_rnn    / head_rnn_add                          iff sites["rnn"] and rnn_mechanism == "activation"
7. head_actor  / head_actor_add                        iff sites["actor"]
8. head_critic / head_critic_add                       iff sites["critic"]
--- baselines, same rule ---
9.  z_unimodal_baseline, z_hidden_baseline             (existing positions, now conditional on sites["encoder"])
10. z_mem_baseline                                     (existing position, now conditional)
11. z_unimodal_add_baseline, z_hidden_add_baseline     (existing positions)
12. z_rnn_baseline, z_rnn_add_baseline                 NEW
13. z_actor_baseline,  z_actor_add_baseline            NEW
14. z_critic_baseline, z_critic_add_baseline           NEW
```

`nnx.Param(jnp.zeros(...))` consumes no RNG, so the baselines' relative order does not affect the RNG stream — but keeping existing ones in place keeps the state dict's key set identical, which is what the restore assertion checks.

New-head bias initialisation follows the existing FiLM convention exactly (`neuromodulator.py:92-95`): γ heads get `bias_init = constant(1.0)` under `type == "FiLM"` (identity in linear space) or `percept_bias_init` otherwise; β heads get `constant(percept_add_bias_init)`. **The modulator therefore starts as a no-op at every new site**, per §5.2 pass-through initialisation — which is also what makes the "adding sites cannot hurt at step 0" argument true.

**`__call__` (lines 129–180)** — same `_get_signal` helper, now called once per enabled site; each disabled site yields `None`. Temperature block wrapped in `if self.temperature_enabled: … else: temperature = None`.

**Add** a construction-time assertion that `target_hidden_size` is the single width all sites share (Analysis §B / D3), so a future divergence fails loudly.

#### 2. `src/models/recurrent_ppo_network.py` — `ActorCriticRNN`

**`__init__` (lines 202–288).** Insert a **strict-validation helper** immediately after `self.modulation_enabled` is computed (`:211`). This is the single most important structural change in the plan: validation lives *here*, in the one place all six construction sites funnel through, so a hand-built dict at any call site fails with a clear `ValueError` instead of a bare `KeyError` — which is exactly how Finding A (`2ad9104`) escaped to a second site.

```python
def _mod_required(cfg: dict, key: str):
    """Mirror Config.get_mandatory for a plain modulation dict.

    ActorCriticRNN is built from a hand-assembled dict at several call sites
    (save_snapshot.py, and historically evaluation.py — see Finding A / 2ad9104),
    so a missing key must raise a NAMED ValueError here rather than a bare
    KeyError deep in the modulator.
    """
    if key not in cfg or cfg[key] is None:
        raise ValueError(
            f"Strict Config: modulation key '{key}' is required but missing. "
            f"See docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md"
        )
    return cfg[key]
```

Then, inside `if self.modulation_enabled:`, before anything else:

```python
# --- A3 migration guard: the flat temp_clip key was replaced by temperature.{enabled,clip}
# The message must be self-explaining for the TWO distinct readers who will hit it (D16):
#   (a) someone editing a live config    -> tell them the new key spelling;
#   (b) someone re-evaluating an ARCHIVED run from its saved models/config.yaml
#       -> tell them the architecture changed and RE-TRAINING, not re-evaluation,
#          is the correct response. Do not shorten this to "missing key".
if 'temp_clip' in modulation_config:
    raise ValueError(
        "modulation.temp_clip is no longer supported: it was replaced by "
        "modulation.temperature.{enabled, clip} in the modulation-site refactor "
        "(2026-08-31). Temperature modulation is now OPT-IN and defaults to disabled.\n"
        "\n"
        "If you are EDITING A LIVE CONFIG, replace the flat key with:\n"
        "    temperature:\n      enabled: true\n      clip: <the old temp_clip value>\n"
        "(plus the now-mandatory `sites:` block and `rnn_mechanism:`).\n"
        "\n"
        "If you are RE-EVALUATING AN ARCHIVED RUN, this config was saved BEFORE the "
        "modulation-site refactor, so the architecture it describes no longer exists in "
        "this code. Its checkpoint cannot be restored, and results produced from it would "
        "not be comparable with either its own original results or any post-refactor run. "
        "The correct response is to RE-TRAIN under the current architecture, NOT to "
        "re-evaluate this checkpoint. This refusal is intentional (plan decision D16).\n"
        "\n"
        "See docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md."
    )

sites = _mod_required(modulation_config, 'sites')
for _s in ('encoder', 'rnn', 'actor', 'critic'):
    if _s not in sites or sites[_s] is None:
        raise ValueError(f"Strict Config: modulation.sites.{_s} is required but missing.")
self.site_encoder = bool(sites['encoder'])
self.site_rnn     = bool(sites['rnn'])
self.site_actor   = bool(sites['actor'])
self.site_critic  = bool(sites['critic'])

self.rnn_mechanism = _mod_required(modulation_config, 'rnn_mechanism')
if self.rnn_mechanism not in ("activation", "gate_bias"):        # D12 — explicit whitelist
    raise ValueError(
        f"modulation.rnn_mechanism must be 'activation' or 'gate_bias', "
        f"got {self.rnn_mechanism!r}."
    )

_temp = _mod_required(modulation_config, 'temperature')
self.temperature_enabled = bool(_mod_required(_temp, 'enabled'))
temp_clip = tuple(_mod_required(_temp, 'clip')) if self.temperature_enabled else None

if not (self.site_encoder or self.site_rnn or self.site_actor
        or self.site_critic or self.temperature_enabled):        # D13 — no dead modulator
    raise ValueError(
        "modulation.type is set but every site is disabled and temperature is off: "
        "the modulator GRU would run every step and receive no gradient (dead weight). "
        "Enable at least one of sites.{encoder,rnn,actor,critic} or temperature.enabled."
    )
```

**RNN cell selection (lines 247–254)** — implement D6:

```python
# BEFORE:
if self.modulation_enabled:
    self.rnn_cell = ModulatedGRUCell(hidden_size, hidden_size, rngs=rngs)
else:
    self.rnn_cell = nnx.GRUCell(hidden_size, hidden_size, rngs=rngs)

# AFTER:
# ModulatedGRUCell differs from nnx.GRUCell in init scheme and gate polarity
# (KNOWN_BUGS: modulated/baseline init confound). Only pay that cost when the
# gate-bias mechanism actually needs it.
self._uses_gate_bias = (self.modulation_enabled and self.site_rnn
                        and self.rnn_mechanism == "gate_bias")
if self._uses_gate_bias:
    self.rnn_cell = ModulatedGRUCell(hidden_size, hidden_size, rngs=rngs)
else:
    self.rnn_cell = nnx.GRUCell(hidden_size, hidden_size, rngs=rngs)
```

⚠️ **RNG-order note**: `ModulatedGRUCell` and `nnx.GRUCell` consume different numbers of RNG draws, so this branch shifts the stream for any *new* configuration. Legacy-equivalent configs take the `ModulatedGRUCell` branch exactly as before, so D11's bit-identity claim is unaffected.

**Modulator construction (lines 265–288)** — pass the three new args through; pass `temp_clip=temp_clip` (possibly `None`).

**`__call__`, modulated branch (lines 318–351)** — rewrite:

```python
# Encoder (A: existing) — only when the site is on
if self.site_encoder:
    x_proj = self.obs_encoder.forward_with_modulation(
        x, mod_output, self.modulation_type,
        unimodal_ln=getattr(self, 'mod_unimodal_ln', None),
        multimodal_ln=getattr(self, 'mod_multimodal_ln', None),
        flat_ln=getattr(self, 'mod_flat_ln', None))
else:
    x_proj = self.obs_encoder(
        x,
        unimodal_ln=getattr(self, 'mod_unimodal_ln', None),
        multimodal_ln=getattr(self, 'mod_multimodal_ln', None),
        flat_ln=getattr(self, 'mod_flat_ln', None))

# RNN (A4)
if self._uses_gate_bias:
    h_new, x_h = self.rnn_cell(task_h, x_proj, gate_bias=mod_output.z_memory)
else:
    h_new, x_h = self.rnn_cell(task_h, x_proj)
    if self.site_rnn:      # rnn_mechanism == "activation"
        # D5: modulate the EMITTED output only. `h_new` (the carry) is deliberately
        # left untouched — scaling the carry re-creates the double-gating pathology
        # that §5.1 flags as Critical. No nonlinearity: x_h is already an activation.
        x_h = mod_output.z_rnn * x_h + mod_output.z_rnn_add

# Actor (A1) — pre-activation FiLM on the single hidden layer
a_pre = self.actor_fc1(x_h)
if self.site_actor:
    a_pre = mod_output.z_actor * a_pre + mod_output.z_actor_add
a_h = self._activate(a_pre)
logits = self.actor_fc2(a_h)

# Temperature (C) — A3: only when enabled
if self.temperature_enabled:
    logits = logits / mod_output.temperature

# Critic (A2)
c_pre = self.critic_fc1(x_h)
if self.site_critic:
    c_pre = mod_output.z_critic * c_pre + mod_output.z_critic_add
c_h = self._activate(c_pre)
value = self.critic_fc2(c_h)
```

**`__call__`, unmodulated branch (lines 353–373)** — **not one character changes.** This is the bit-identity guarantee, and it is guaranteed structurally (the branch is untouched) as well as empirically (C1).

**`initial_state` (lines 375–392)** — unchanged.

#### 3. `train.py` — modulation logging (lines 1145–1147, 1667–1670, 1820–1858)

Every `mod_info.<field>` read must be guarded, because disabled sites now carry `None` and `jnp.mean(None)` raises. Add the new sites' metrics.

- `:1145-1147` — the startup print gains the site set, mechanism and temperature state, e.g.
  `Neuromodulation: ENABLED (type=FiLM, mod_hidden=16, grouping=1, sites=[encoder,rnn], rnn_mechanism=gate_bias, temperature=on)`.
- `:1667-1670` — guard `z_uni` / `z_multi` / `z_mem` / `temp` with `is not None`; keep the existing `1.0` fallback for the progress-bar display only (it is cosmetic, not a config default).
- `:1820-1841` — build the metric dict conditionally. Name the new series consistently with the existing ones:

| Site enabled | Metrics logged |
|---|---|
| `encoder` | `modulator/gamma_uni_{mean,std}`, `gamma_multi_{mean,std}` (+ `beta_*` for FiLM/PreActivation, as today) |
| `rnn`, `gate_bias` | `modulator/z_memory_{mean,std}` (**unchanged name** — preserves continuity with every archived run) |
| `rnn`, `activation` | `modulator/gamma_rnn_{mean,std}`, `modulator/beta_rnn_{mean,std}` (**new names** — a different quantity must not reuse `z_memory`) |
| `actor` | `modulator/gamma_actor_{mean,std}`, `modulator/beta_actor_{mean,std}` |
| `critic` | `modulator/gamma_critic_{mean,std}`, `modulator/beta_critic_{mean,std}` |
| `temperature.enabled` | `modulator/temperature_{mean,min,max}` (as today) |

- `:1858` — guard the `postfix["T"]` temperature display.

#### 4. Configs — the 12 files listed in the Migration section

Exactly the block shown above, per-file ceiling preserved, plus a header migration note. **No other key in these files changes.**

#### 5. Tests

**New file `tests/models/test_modulation_sites.py`:**

| Test | What it pins |
|---|---|
| `test_baseline_forward_matches_golden` | Unmodulated path bit-identity vs a pre-change golden fixture. **Parametrised over both fixture modes** (flat/no-LN and hierarchical/LN). |
| `test_baseline_param_tree_matches_golden` | Unmodulated parameter tree: identical key set, shapes **and values** from a fixed seed. **Both modes.** |
| `test_legacy_equivalent_forward_matches_golden` | `sites={encoder,rnn}`, `rnn_mechanism="gate_bias"`, `temperature.enabled=true` reproduces the pre-change modulated forward pass bitwise. **Both modes** — the hierarchical/LN case is the one that covers what real runs execute. |
| `test_legacy_equivalent_param_tree_matches_golden` | Same, for the parameter tree (this is what makes old checkpoints restorable). **Both modes.** |
| `test_site_toggles_build_expected_heads` | Parametrised over all 15 non-empty site subsets × both mechanisms: asserts exactly the expected `head_*` / `z_*_baseline` attributes exist and no others. |
| `test_all_sites_off_and_temperature_off_raises` | D13. |
| `test_legacy_temp_clip_key_raises` | D9 — must match on the migration message, **including the archived-run clause**: assert the message mentions re-training (not re-evaluation) as the remedy, so a future message edit cannot quietly strip the D16 guidance. |
| `test_missing_site_key_raises` / `test_missing_rnn_mechanism_raises` / `test_missing_temperature_enabled_raises` | No-fallback-defaults contract, one test per new mandatory key. |
| `test_unknown_rnn_mechanism_raises` | D12. |
| `test_temperature_disabled_leaves_logits_untouched` | Builds two models sharing weights, temp on vs off; asserts `head_action` absent and logits differ only by the division. |
| `test_rnn_activation_does_not_modulate_carry` | D5 — run one step with a deliberately non-identity `z_rnn`; assert the returned carry equals the unmodulated cell's carry while `x_h`'s downstream effect differs. |
| `test_initial_state_structure_invariant_across_sites` | `jax.tree_util.tree_structure(model.initial_state(4))` identical across all site combinations. |
| `test_plain_gru_cell_used_when_not_gate_bias` | D6 — `isinstance(model.rnn_cell, nnx.GRUCell)` for `activation` and for encoder/actor/critic-only. |
| `test_no_retrace_across_steps` | Jit a stepping function with a module-level trace counter incremented in the function body (executed at trace time only); run 5 steps with all four sites on; assert exactly one trace. |

**Modify `tests/models/test_network_construction.py:19-27`** — the `MODULATION_CONFIG` fixture gains `sites`, `rnn_mechanism`, `temperature`, and drops `temp_clip`. All three existing tests must still pass unchanged.

**Modify `tests/scripts/test_evaluation_model_rebuild.py:142-144`** — extend the key-presence assertion (currently `memory_clip`) to the full new mandatory set, so the Finding-A regression guard covers the new keys too.

#### 6. Golden fixtures — `tests/fixtures/modulation/`

**Four** `.npz` files — a (baseline, legacy-equivalent) **pair in each of two encoding modes** — each holding the flattened parameter tree and the forward-pass outputs for a fixed seed and one fixed observation:

| Fixture | Encoding | LayerNorm | Why it exists |
|---|---|---|---|
| `flat_baseline.npz` | flat | off | Minimal, fast, isolates the unmodulated path |
| `flat_legacy.npz` | flat | off | Minimal, isolates the legacy-equivalent modulated path |
| `hier_ln_baseline.npz` | **hierarchical** | **on** | **The path every real run uses** |
| `hier_ln_legacy.npz` | **hierarchical** | **on** | Same, modulated |

**Why the second pair is mandatory, not optional** (`plan-reviewer` finding 4): all 12 real NMN configs are `encoding_mode: "hierarchical"` with `use_layer_norm: true` (e.g. `recurrent_ppo_nmn_het_film_g1.yaml:30,37`). A flat/no-LN fixture never enters `forward_with_modulation`'s hierarchical branch and never exercises the LayerNorm placement — so with only the flat pair, bit-identity of the path the project actually runs would rest entirely on V4 and C5, i.e. on one expensive end-to-end run rather than on a cheap unit test. The hierarchical pair needs a realistic `observation_breakdown` (a small multi-sensor dict, not a single flat block) so the unimodal-per-sensor stage and the multimodal hub are both non-trivial.

Dimensions for the flat pair stay tiny (`input_dim=8, hidden_size=16, action_dim=4`); the hierarchical pair uses whatever minimal breakdown makes both encoder stages real — still small, still one fixed observation.

Generator: `tests/fixtures/modulation/generate_golden.py`. **Deliberately NOT under `scripts/`** — placing it there would trigger the [[SCRIPTS_DEPENDENCY_MAP]] maintenance contract for a test-only helper.

⚠️ **Ordering hazard, and the plan's single most skippable step:** the fixtures must be generated on the **pre-change working tree**, before `src/` is edited. Generating them afterwards makes every parity test a tautology that passes no matter what broke. This is Checkpoint C0a and it comes first. **The same hazard applies to V4's "before" losses** — see C0b.

#### 7. Docs

- `docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md` — §2A head list (≈ lines 587–604) and the rPPO pseudocode block (≈ lines 672–684): add the actor/critic sites, the `rnn_mechanism` fork, the opt-in temperature, and a note that the `activation` mechanism modulates the GRU's emitted output only (with the §5.1 cross-reference).
- `docs/develop/active/neuromodulation/NMN_METRICS_REFERENCE.md` — new §2.x entries for `gamma_rnn_*`, `beta_rnn_*`, `gamma_actor_*`, `beta_actor_*`, `gamma_critic_*`, `beta_critic_*`, plus a note on when `z_memory_*` and `temperature_*` are absent.

#### Explicitly NOT changed (verify, do not edit)

| File | Why it needs no change |
|---|---|
| `evaluation.py:378-393` | Passes the whole `agent.modulation` dict through; new keys flow automatically. **Confirm by reading, do not edit** (C6). |
| `scripts/eval/eval_rollout.py:1093-1113` | Same pass-through pattern. |
| `scripts/eval/traj_collect/collect_trajectories.py:199-210` | Same pass-through pattern. |
| `src/models/modulated_gru_cell.py` | Untouched; still used for the `gate_bias` mechanism. |
| `src/models/neuromodulator.py:193-379` (`DreamerNeuromodulatorRNN`) | D15 — Dreamer stack, separate class. |
| `src/models/recurrent_ppo_trainer.py` | Structure-agnostic; `None` fields cost it nothing. **Confirm the three 4-tuple unpack sites still hold** (C8). |
| `save_snapshot.py` | **Already broken before this change** (Analysis §G2). Out of scope. Flag to the user; `bug-curator` should record it. |
| `configs/models/ppo/neuromodulated_ppo.yaml` | For `ppo_network.py`, which has no modulation code at all. Stale; do not touch. |

#### Maintenance-contract audit

| Contract | Triggered? | Reasoning |
|---|---|---|
| [[CONFIG_GUIDE]] + `02_config_schema.md` | **No, for items 1–2.** | Both documents are scoped to the **environment** config (`configs/environment/default.yaml`, `EnvParams`, `config_loader.py`). This change touches only the **agent** config under `configs/models/`, adds no `EnvParams` field, and does not modify `config_loader.py`. Confirmed by grep: neither doc mentions `agent.modulation`, `grouping_size` or `temp_clip`. **Items 3–4 of that contract DO apply** and are satisfied: a test per new mandatory key (§5), and the env parity gate re-run green (C9). If the implementer finds this reading wrong, stop and ask rather than guessing. |
| [[CONFIG_CRITICAL_SETTINGS]] | **No.** | The registry contains only `sensory.*` settings and `body.death_penalty`. No registry value moves. No change-log entry required. |
| [[SCRIPTS_DEPENDENCY_MAP]] | **No.** | Nothing under `scripts/` is added, moved, renamed or deleted; the fixture generator is placed under `tests/` precisely to keep it that way. If the implementer relocates it, this contract activates. |
| `docs/develop/INDEX.md` | **Yes.** | Run `python scripts/claude/regen_dev_index.py` after this doc lands and after any status change. |

---

## Verification — what could fail, and what would prove it did not

Every row names evidence that could actually come out negative.

| # | Claim under test | Evidence that proves it | How it could fail |
|---|---|---|---|
| V1 | Unmodulated baseline is bit-identical | `test_baseline_forward_matches_golden` + `test_baseline_param_tree_matches_golden` against fixtures captured **pre-change** | An accidental edit inside the `else:` branch, or a reordered `nnx.Linear` shifting the RNG stream |
| V2 | Legacy-equivalent modulated config is bit-identical | The two `test_legacy_equivalent_*` tests | Any new head constructed before `head_action`; the RNN-cell branch picking `nnx.GRUCell` |
| V3 | Existing NMN checkpoints still restore | Restore a real archived `recurrent_ppo_nmn_*` checkpoint with its migrated config; the completeness assertion from `2ad9104` must not fire | Renamed head attributes; changed baseline shapes |
| V4 | End-to-end training is unchanged for a migrated config | Run ~20 iterations of `recurrent_ppo_nmn_het_film_g1.yaml` at a fixed seed **before** and **after**; per-iteration losses must match to float equality. **The "before" half is subject to the same ordering discipline as the fixtures — see C0b**; and the whole before/after pair must be bracketed by the sequencing rule below | Anything V1–V2 missed that only shows up through the optimiser; **or a silently non-comparable "before" half captured after `src/` was already touched** |
| V5 | Each new site actually does something | For each of actor / critic / rnn-activation alone: perturb that head's bias and assert the model's output changes; assert `optax.global_norm` of that head's gradient is non-zero after one PPO update | A head that is constructed but never read — the exact failure mode of the fixed LSTM/`z_memory` bug |
| V6 | Disabling a site really disables it | With `sites.actor=false`, assert `head_actor` is absent **and** that the forward output equals the same-weights model with the actor path bypassed | A stale `getattr(..., None)` making a disabled site silently active |
| V7 | No new mandatory key can be silently defaulted | One raising test per key (§5) | `.get(key, default)` sneaking in |
| V8 | No per-step recompilation | `test_no_retrace_across_steps`; independently, `JAX_LOG_COMPILES=1` on a 200-step smoke run shows compiles only in the first iterations | Storing enables in a mutable container that becomes a traced leaf |
| V9 | Speed does not regress | `Time/sps_env` over ≥ 200 iterations, same node/GPU/seed/config, before vs after. **Two measurements**: (a) migrated legacy config — must be within run-to-run noise, since the graph is unchanged; (b) all-four-sites-on — record the number, expected small | Extra `jnp.repeat`/gather per step; unnecessary buffer growth |
| V10 | Rollout memory does not balloon | Peak device memory on the all-sites-on config vs baseline | D7 violated — zeros emitted for disabled sites and stored per step |

**On V9's threshold**, per the project's verification protocol: > 5 % slowdown warrants discussion, > 15 % blocks merge. Measurement (a) is the sharp one — the legacy-equivalent config executes an identical computation graph, so *any* measurable slowdown there means something structural changed and should be investigated rather than accepted.

---

## Checkpoints

Ordered. C0a/C0b are genuinely first — doing either late invalidates C1–C4 and V4 respectively.

- [ ] **C0a — Capture all four golden fixtures on the UNMODIFIED tree.** Run `tests/fixtures/modulation/generate_golden.py` before editing any file under `src/`; it must emit the flat pair **and** the hierarchical+LayerNorm pair. Confirm via `git status` that `src/` is clean at capture time and record the commit SHA in the Implementation Report.
- [ ] **C0b — Capture V4's "before" losses, also on the UNMODIFIED tree, in the same sitting as C0a.** V4 is the only bitwise check that exercises the real hierarchical-encoder + LayerNorm path *through the optimiser*, and its "before" half is only meaningful if produced pre-change. Run ~20 iterations of `recurrent_ppo_nmn_het_film_g1.yaml` at a fixed seed and save the per-iteration losses to a file under `tmp/`, recording seed, node, GPU and the same SHA as C0a.
  **Fallback if C0b was skipped or the run is lost:** do **not** improvise a "before" from the modified tree. Regenerate it from the SHA recorded in C0a via a throwaway git worktree — `git worktree add /tmp/pre_refactor <SHA>` — run the 20 iterations there with the identical seed/node/GPU, then `git worktree remove`. Note in the Implementation Report which route was used. (A worktree is used rather than a branch switch because this repo's git-safety rule forbids switching with untracked data present.)

  ⚠️ **Sequencing rule — the MC-return-bootstrap fix.** The open Monte-Carlo return-bootstrap units bug (Risks item 2, registry row P1 #5) changes the loss values V4 compares. If that fix lands during this work, it must land **strictly before C0b** or **strictly after V4's "after" half** — **never between the two halves**. Landing it in between makes the two sides differ for a reason that has nothing to do with this refactor, and V4 then fails (or, worse, appears to fail) uninterpretably. If the fix lands between, both halves must be re-run. **Plan for both fixes: [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]]** — it is sequenced to land entirely BEFORE C0a/C0b, so the before/after pair here is taken on already-fixed code.
- [ ] **C1** — After editing `neuromodulator.py` + `recurrent_ppo_network.py`, the baseline golden tests pass.
- [ ] **C2** — All three existing tests in `tests/models/test_network_construction.py` pass with the updated fixture dict.
- [ ] **C3** — `tests/scripts/test_evaluation_model_rebuild.py` passes, including its unmodulated round-trip case.
- [ ] **C4** — Legacy-equivalent golden tests pass (forward **and** parameter tree).
- [ ] **C5** — A real archived NMN checkpoint restores under its migrated config with no completeness-assertion failure.
- [ ] **C6** — Read (do not edit) `evaluation.py`, `scripts/eval/eval_rollout.py`, `scripts/eval/traj_collect/collect_trajectories.py`; confirm each passes the whole `agent.modulation` dict through. Report if any hand-builds a whitelist.
- [ ] **C7** — `initial_state` pytree structure identical across all 15 site combinations.
- [ ] **C8** — All three trainer call sites still unpack a 4-tuple on both paths; a smoke train with `mod_info` containing `None` fields survives `vmap` + `scan`.
- [ ] **C9** — Full test suite green, including `tests/env/test_unified_parity.py` and `tests/env/test_visual_parity.py` (these must be untouched by an agent-side change; if they move, something is wrong).
- [ ] **C10** — Short smoke train (~200 iterations) on **three** configs: a baseline, a migrated legacy NMN config, and a new all-four-sites config. No NaN; WandB shows exactly the metric series the site table predicts, and none of the others.
- [ ] **C11** — Speed + memory numbers for V9/V10 recorded in the Implementation Report, with node, GPU, config and seed stated.
- [ ] **C12** — `NEUROMODULATION_ALGORITHM.md` and `NMN_METRICS_REFERENCE.md` updated in the same change; `python scripts/claude/regen_dev_index.py` exits 0.

---

## Risks the user should decide on

1. **`lr_critic` is dead — the modulator and critic train at `lr_actor` (5× the advertised critic rate).** Recorded open bug (registry row A1). A2 adds FiLM heads that feed the critic, so those heads will also train at 5× the intended critic rate. This does not block the refactor, but it will confound any critic-modulation arm of the 2-factor experiment. **Decide before launching, not before merging.** **Now planned** (wire `lr_critic`, add a new `lr_modulator`, all rates set to each file's own `lr_actor` so behaviour is unchanged): [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]].
2. **The Monte-Carlo return bootstrap is in the wrong units** (registry row P1 #5, severity **High**, `recurrent_ppo_trainer.py:374-380`): the window-edge seed is added onto un-rescaled rewards while the critic trains on within-window-normalised returns. It is live in every config, so it contaminates the *baseline* that any modulated arm would be compared against. **This is a pre-experiment blocker, not a refactor blocker** — but it should be fixed before the 2-factor runs launch, or the comparison inherits it. ⚠️ **It does carry one hard constraint on *this* work**: because it changes loss values, it must not land between V4's "before" and "after" halves. See the sequencing rule under Checkpoint C0b. **Plan: [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]]**, which lands both this and item 1 before C0a.
3. **~~PAPL's precondition for critic modulation is not met~~ — RETRACTED 2026-08-31, the precondition IS met.** The earlier text here claimed our reward is not a function of injury. That is false: the live reward is the step-to-step reduction in homeostatic drive, and injury is one of the two coordinates of that drive (Analysis §F). Nothing about the refactor changes; what changes is what the experiment write-up must say. **The write-up must claim PAPL's licence, not disclaim it**, and must carry instead the two residual caveats in §F: PAPL's conditioner is an open-loop clock while ours is contingent sensed injury delayed by a 3-step smoothing kernel, and Marquis shows the sign of critic conditioning is mechanism-dependent (positive for FiLM, negative for LoRA). This item is left in place rather than deleted so a reader of an older copy of this plan can see it was withdrawn.
4. **`save_snapshot.py` is broken today and stays broken.** It is a second, never-swept instance of the Finding-A hand-built-whitelist bug, plus a missing `encoding_config`. Recommend asking `bug-curator` to record it (Status OPEN, Severity Low) and fixing it in a separate one-file change.
5. **Pre-activation vs post-activation for actor/critic** (Analysis §E) — resolved to pre-activation on the strength of the encoder-parity constraint and PAPL. Overrule now if that reading is wrong; changing it later invalidates any runs launched in the meantime.
6. **Whether to land Part B in the same change.** If Part B is approved now, the same 12 configs get one edit instead of two, and the 2-factor experiment becomes runnable in one step. If deferred, the 12 files are edited twice.

---

## Part B — configurable modulator input slice (PROPOSAL — NOT REQUESTED, NOT APPROVED)

> ⚠️ **The user has not approved this section.** It is presented so it can be accepted or deferred. Part A is complete and landable without any of it. Nothing in Part A depends on Part B.

### What it is, in plain terms

The 2-factor experiment varies **what the modulator reads** as well as what it writes to. Today the modulator's GRU reads the **entire** observation vector — every sense the agent has. The experiment needs it to read a **subset**: only the body's internal signals (satiation plus internal pain), only the external senses, only the pain-related channels, or everything.

### Design

The observation layout is name-keyed and built by `get_observation_breakdown(params)` (`src/environment/sensor.py:482-524`), which returns an **ordered** `{sensor_name: dim}` dict. `ActorCriticRNN` already receives it as `observation_breakdown`. So the slice is specified **by sensor name** and resolved to indices at construction — never hard-coded, because the breakdown is config-dependent (Injury, Nutrition, both nociception channels, olfaction, proprioception, vision and location are each individually gated, and several widths scale with sensor range).

```yaml
agent:
  modulation:
    input_sensors: "all"     # or an explicit list of sensor names
    # e.g. interoceptive only:  ["Satiation", "Interoceptive Nociception"]
    # e.g. nociceptive only:    ["Interoceptive Nociception", "Extero Nociception"]
```

Resolution, in `ActorCriticRNN.__init__`:

1. Walk `observation_breakdown` accumulating offsets → `{name: (start, stop)}`.
2. `"all"` → the full range, and the modulator's `obs_dim` is unchanged.
3. A list → validate **every** name against the breakdown; on a miss raise `ValueError` naming the unknown sensor **and listing the available ones** (the breakdown is config-dependent, so a plain "unknown key" message would send the reader on a hunt).
4. Build the flat index tuple, store as `self.mod_input_idx: tuple[int, ...]` — a plain tuple of Python ints, so it stays in the graphdef as static data and never becomes a trained or checkpointed leaf.
5. Pass `obs_dim = len(self.mod_input_idx)` to `NeuromodulatorRNN`.
6. In `__call__`, gather **after** the symlog compression (`recurrent_ppo_network.py:316`) so the modulator sees the same scaling as the task network: `mod_in = x[..., jnp.array(self.mod_input_idx)]`.

A contiguous fast path (a plain slice when the selected names are adjacent) is **not** worth adding: interoceptive-only and nociceptive-only happen to be contiguous, but exteroceptive-only is not, and a single gather is negligible against a GRU.

### Rejected alternative — named presets

A `input_sensors: "interoceptive"` preset would need a hard-coded name→sensor-set map inside the code. That map would be a hidden default that silently mis-resolves whenever a sensor is gated off, and it would put an experimental grouping decision in `src/` where nobody reviewing a config could see it. **Explicit lists in the config; presets live in the experiment configs**, which is `experiment-designer`'s territory.

### Consequences

- **New mandatory key**: `agent.modulation.input_sensors` (`"all"` or a list of strings), required whenever `type` is non-null. All 12 NMN configs get `input_sensors: "all"` — a no-op that preserves current behaviour exactly.
- **Checkpoints**: `"all"` keeps `modulator.gru`'s input width and therefore full restore compatibility. Any real slice changes that width, so old checkpoints will not restore into it — same reasoning and same non-mitigation as Part A.
- **Interaction with observation gating**: the same config text resolves to different indices under different environment configs. That is correct behaviour, but it means a modulator-input slice is only meaningful relative to a stated environment config. The experiment design must record both.
- **Tests** (`tests/models/test_modulation_input_slice.py`): `"all"` is byte-identical to today; a named subset selects exactly the right columns (checked against a hand-computed breakdown); an unknown sensor name raises with the available names listed; the resolved width changes the modulator GRU's input dimension; a sensor gated off in the env config makes a config naming it fail loudly rather than silently shifting indices.

---

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- developer: fill this in. Must include:
     - the commit SHA the golden fixtures were captured at (C0a), and confirmation
       that all FOUR fixtures (flat pair + hierarchical/LN pair) were captured there
     - where V4's "before" losses live and whether they came from C0b directly or
       from a worktree regeneration at the recorded SHA
     - whether the MC-return-bootstrap fix landed during this work, and if so on which
       side of V4's before/after pair (it must never land between them — see C0b)
     - before/after speed numbers with node, GPU, config, seed (C11)
     - any deviation from the plan and why -->

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [one-line summary]

---

## Feedback from plan-reviewer (2026-08-31)

**Verdict: SOUND WITH CONCERNS — one 🔴 Critical text correction required before approval.** Full review: [`docs/reviews/plan_modulation_site_refactor.md`](../../../reviews/plan_modulation_site_refactor.md).

1. 🔴 **Analysis §F bullet 3 and Risks item 3 are factually wrong and must be rewritten.** They claim "our reward is not a function of injury", so PAPL's precondition for critic modulation is unmet. Verified false against the code: `src/environment/core.py:49-53` + `:728-731` — with `use_homeostatic_reward: true` (`configs/environment/default.yaml:181`) the reward is the reduction in homeostatic drive, and the drive is the L2 distance of `(satiation, injury)` from the setpoint. Injury is in the reward; healing is rewarded; **PAPL's precondition IS met.** The source doc was corrected in `c36a6355` after this plan was committed. The honest residual caveats: PAPL's conditioner is an open-loop clock vs. our contingent sensed injury; Marquis shows critic modulation is mechanism-dependent (positive for FiLM).
2. 🟡 **Archived-run evaluation breaks by default after migration.** `evaluation.py:235`, `scripts/eval/eval_rollout.py:1064-1071`, and `scripts/eval/traj_collect/collect_trajectories.py:693` all rebuild models from the run's **saved** `models/config.yaml`, which still carries flat `temp_clip` — so the D9 hard error fires on every re-analysis of an archived NMN run. Loud, not silent, but the plan's checkpoint-compatibility section assumes the *migrated* config is loaded, which is not the tools' default. Needs a decided policy (documented manual migration + override flags, or a read-only translation at the eval boundary) before implementation.
3. 🟡 **V4's "before" half needs the same ordering discipline as C0** (capture pre-change, or pin the SHA for a worktree regen), and the golden fixtures cover only flat encoding without LayerNorm while every real NMN config is hierarchical + LN — consider a second fixture pair.

Everything else checked out: the golden-fixture verification is genuinely non-circular (fixtures are pre-change ground truth), the RNG-order/bit-identity argument holds structurally for this Flax version, the 12-file migration inventory and ceilings are exact, D6's confound-removal claim is correct (the modulator is constructed last, so the task network's init draws match the baseline), the maintenance-contract audit claims were re-verified, and the cited bug-registry rows exist as described.

— Reviewed by: plan-reviewer

---

## Response to plan-reviewer (2026-08-31)

All five findings are addressed. Nothing was declined.

| # | Sev | Finding | Disposition | Where in this doc |
|---|:--:|---|---|---|
| 1 | 🔴 | "Our reward is not a function of injury" is false; PAPL's precondition IS met | **Fixed — claim retracted in both locations.** §F bullet 3 now states the precondition is met, shows the reward is `r_t = D(s_{t-1}) − D(s_t)` with `D` the L2 distance of `(satiation, injury)` from `(setpoint, 0)`, and notes the coupling `∂D/∂injury = injury/D`. Risks item 3 is struck through and rewritten to instruct the write-up to *claim* PAPL's licence, not disclaim it. The two honest residual caveats (open-loop clock vs. delayed contingent sensed injury; FiLM-vs-LoRA mechanism dependence) replace it. A retraction marker is left in both places so an older copy of this file cannot silently win. The `calculate_drive` vs `drive_hunger`/`drive_injury` reader-footgun is recorded in §F. | Analysis §F; Risks 3 |
| 2 | 🟡 | Archived NMN runs break on re-evaluation because the eval tools load the run's *saved* config | **Resolved by user decision — converted from open risk to recorded decision D16.** Accept the hard failure; it is a guard rail against mixing pre- and post-refactor results, since re-evaluating a checkpoint whose architecture has changed yields numbers comparable to nothing. No shim, no translation layer, no manual migration procedure, no `--agent-config` on the other two tools. Two follow-ons added: the error message must say *re-train, not re-evaluate*, and the blast radius is stated explicitly (NMN runs only — all `type: null` runs, i.e. the sensor-ladder, directional-sensor, rest-premium and dwell work, are untouched). | D16; §"Archived-run evaluation"; File Changes §2 error text; `test_legacy_temp_clip_key_raises` |
| 3 | 🟡 | V4's "before" half needs C0's ordering discipline | **Fixed — C0 split into C0a (fixtures) and C0b (V4 before-side losses), both on the unmodified tree.** A worktree-regeneration fallback at the recorded SHA is documented for the case where C0b is missed. The MC-return-bootstrap sequencing constraint is recorded alongside it and cross-linked from Risks item 2. | C0a / C0b; V4 |
| 4 | 🟡 | Fixtures are flat + no LayerNorm; all real NMN configs are hierarchical + LN | **Fixed — a second fixture pair in hierarchical + LayerNorm mode is now mandatory**, and the four parity tests are parametrised over both modes. Without it, bit-identity of the path real runs execute would rest solely on V4/C5. | §6 Golden fixtures; §5 test table |
| 5 | 🟢 | The "keep temperature last for positional readers" rationale is moot | **Fixed — rationale deleted.** Field order is now stated as readability-only, with an explicit note that all access is by attribute and the implementer may reorder. | File Changes §1 |

**Not addressed, by design:** Part B (modulator input slicing) is unchanged and remains **unapproved** — the user has not ruled on it. The reviewer raised no Part B findings.

— Revised by: senior-developer
