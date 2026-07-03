# v3.0 Pipeline Correctness Audit — Surface 1 (JAX/Flax)

## Plain-language summary

This is a JAX/Flax correctness audit of the v3.0 train-and-evaluate pipeline,
prompted by an earlier silent bug (the trainer loaded env configs without
resolving the `extends:` inheritance chain, dropping inherited layers). The
worry: the large v3.0 change wave may have introduced other silent regressions.

**Headline:** The environment itself (random-number handling, the parallel-env
batching, the immutable-state update discipline) is **JAX-correct** — I found no
regression in `core.py`, `wrapper.py`, or `sensor.py`. The **training** side
(`train.py`) builds the model and saves checkpoints correctly. The defects are
all on the **evaluation** side: `evaluation.py` rebuilds the model from the saved
config **by hand**, and that hand-rebuild has drifted out of sync with how
`train.py` builds the same model. Concretely, **any agent that uses
neuromodulation cannot be evaluated** — `evaluation.py` forgets to pass one
required setting (`memory_clip`) and crashes with `KeyError`. Separately, the
**DreamerV3 evaluation path is broken** — it hands the model builder a plain
5-key dictionary where the builder expects a full config object, so it crashes,
and it also omits the observation-layout and modulation settings, so even if it
didn't crash it would rebuild a differently-shaped network than training saved.
Plain (non-modulated) RecurrentPPO agents — the current mainstream runs — round-trip
**correctly**. Both broken paths fail **loudly** (a crash, not wrong numbers), so
no silently-wrong evaluation results have been produced; the risk is that a whole
class of runs simply can't be evaluated.

**Verdict: 2 confirmed defects (both eval-side), 2 latent-robustness concerns.
Environment + training + non-modulated rPPO round-trip verified correct.**

---

## Findings

| # | Severity | File:line | What's wrong | Failure scenario | Confidence |
|---|----------|-----------|--------------|------------------|------------|
| 1 | 🔴 blocker | `evaluation.py:252-259` | Eval hand-reconstructs `modulation_config` from 6 keys and **omits `memory_clip`** (and `percept_add_bias_init`). `ActorCriticRNN.__init__` reads `modulation_config['memory_clip']` with **no default** (`recurrent_ppo_network.py:265`). Train instead passes the whole dict via `config.get('agent.modulation')` (`train.py:745`), which includes `memory_clip`. | Evaluating **any** neuromodulation-enabled RecurrentPPO checkpoint (every `recurrent_ppo_nmn_*` config — all carry `memory_clip: [-2.0, 2.0]`) raises `KeyError: 'memory_clip'` at model construction, before restore. No modulated agent can be evaluated. | **High** — confirmed by reading both call sites + config keys. |
| 2 | 🔴 blocker | `evaluation.py:293-300` | Eval builds `dreamer_config` as a **plain dict** of 5 keys and passes it as the `config` arg to `DreamerTrainer`. But `DreamerTrainer.__init__` calls `config.get_mandatory('agent.encoder_dim', ...)` (`dreamer_v3_trainer.py:65-82`) — a plain dict has no `.get_mandatory`. It also passes **no `obs_breakdown` and no `modulation_config`**, whereas train passes both (`train.py:815-817`). | Evaluating a DreamerV3 checkpoint raises `AttributeError: 'dict' object has no attribute 'get_mandatory'`. Even if that were fixed, the missing `obs_breakdown`/`modulation_config` would rebuild a **differently-shaped** world model than training saved → silent partial restore (see #3). | **High** for the crash (signature-level); **High** that shape would diverge. Needs one functional eval run to confirm end-to-end. |
| 3 | 🟡 concern | `evaluation.py:77-94` (`_merge_restored_into_module_state`) | The merge iterates only over the **current module's** Param keys and, for any key **absent in the restored checkpoint**, silently keeps the freshly-initialised (random) weight — **no warning, no error**. Faithful restore is therefore guaranteed *only* when the eval-rebuilt structure exactly matches the train-saved structure. | If train/eval model shapes ever diverge (exactly the failure mode behind #1/#2, or any future config key that changes architecture), eval loads a **half-random** model and reports meaningless survival numbers with no signal that anything is wrong. This is the silent-regression class the audit exists to catch. | **High** that the behavior is silent; the trigger is latent (needs a structural mismatch). |
| 4 | 🟡 concern | `evaluation.py:250-261` (design) | Root cause of #1/#2: eval **re-derives** the model-construction config by hand instead of reusing train's exact expression (`config.get('agent.modulation')`, then null-if-`type`-is-None). Every future key added to `agent.modulation` that `ActorCriticRNN` reads without a default will silently reintroduce this drift. | Standing divergence risk between the two model builders; guarantees recurrence of #1-type bugs. | **High** (structural). |
| 5 | 🟢 nit | `wrapper.py:35-61` (`auto_reset_step`) | Dead code — **no caller** in the repo (train.py handles auto-reset inline). Its reset key `jax.random.split(state.key)[0]` derives from the pre-step key, and it returns the **terminal** obs alongside the **reset** state, which would be a subtle bug if it were ever wired in. | None today (unused). Flagged so it isn't adopted as-is. | **High** (unused confirmed by grep). |

---

## Sub-surfaces verified correct (explicit negatives)

These were checked against the v3.0 diff and found **JAX-correct** — reported so the audit is not just a defect list:

- **PRNG threading in `jax_step` (`core.py:505`, `825`).** 6-way split (`key, respawn_key, hunt_key, wander_key, damage_key, property_key`); the main `key` is advanced and stored back on the new state (`core.py:825`). `res_visual_property` noise is decorrelated from `res_property` via `jax.random.fold_in(property_key, 0x7150A1)` (`core.py:534`). Reproducibility property (same key + same actions → same episode) holds.
- **Per-episode sampling in `jax_reset` (`core.py:951-953`).** `animal_episode_key = jax.random.fold_in(property_key, 0xAE1)` derives the new per-episode animal-parameter stream **without disturbing** the existing agent/placement/body/property streams — byte-stable, correctly independent.
- **vmap axis discipline (`wrapper.py:13-15`).** `reset` = `in_axes=(None, 0)` (params broadcast, per-env keys batched on axis 0); `step` = `in_axes=(0, 0, None)` (states + actions batched, params broadcast); per-env keys are distinct via `jax.random.split(key, num_envs)` (`wrapper.py:18`). Matches the "axis 0 = env index, EnvParams broadcast not batched" convention exactly.
- **Immutability (`core.py:787`).** `jax_step` produces the new state via `state._replace(...)`; no in-place pytree mutation in traced code.
- **obs_dim / action_dim parity.** Train `input_dim = obs.shape[-1]` (`train.py:712`) vs eval `input_dim = obs.shape[0]` (`evaluation.py:234`) — both are the observation-vector length from `get_observation` under the *same* `load_env_params(config)`, so they cannot diverge. `action_dim = 4 + int(rest) + int(eat)` is identical in both (`train.py:716`, `evaluation.py:239`).
- **encoding_config parity.** Both pass `config.to_dict().get('agent', {})` (`train.py:767`, `evaluation.py:272`), so `encoding_mode`, `hierarchical_params`, and `use_layer_norm` (and thus the LayerNorm submodules and hierarchical grouping) are built identically. **For non-modulated RecurrentPPO agents, `modulation_config` is `None` in both paths → structurally identical model → byte-faithful checkpoint round-trip.**
- **Inference parity.** Eval's `generic_inference` (`evaluation_core.py:29-48`) uses `argmax` under `eval_mode`; the observation `symlog` compression lives *inside* the model (`recurrent_ppo_network.py:308`), so both train and eval feed raw obs and get identical preprocessing. Consistent.
- **Behavior modules (`src/behavior/*`, new in v3.0).** Pure host-side numpy operating on already-transferred arrays — **no JAX tracing, vmap, or PRNG surface**, so no JIT/pytree/key pitfalls. (Metric-logic correctness, e.g. M1/M2 counting, is a numpy-semantics question outside this JAX lens and not reviewed here.)
- **`record_true_obs` change (`evaluation_core.py:186`, uncommitted).** The widened predicate (`record_stats or (render_video and params.perceptual_noise_enabled)`) is a host-side Python boolean gating optional CSV/recording columns; `get_observation(..., apply_noise=False)` is a supported signature (`sensor.py:291`). No JAX-correctness impact.

---

## Pre-existing (not a v3.0 regression) — noted, not counted

- **`damage_key` reused across three draws (`core.py:571, 622, 643`).** The same sub-key feeds `res`, `predator`, and `obstacle` damage `uniform` draws, making those rolls correlated. This is **present verbatim on `main`** (`git show main:src/environment/core.py` lines 345/394/406) and the surrounding comments show it is a deliberate byte-preserving choice. Not introduced by v3.0; flagged only for awareness (if damage rolls should be statistically independent, decorrelate with `fold_in`).

---

## Conventions audit checklist

- pytree / `_replace` immutability: ✅
- JIT / static-field discipline: ✅ (no traced→static leakage found; `get_observation` correctly marks `apply_noise` static)
- vmap axis correctness: ✅
- PRNG threading (split / fold_in / tail-key): ✅
- Sensor ↔ observation-breakdown sync: ✅ (not touched in a way that desyncs; obs layout stable)
- Config protocol (`get_mandatory`): ⚠️ — train side clean; **eval side violates parity** by hand-rebuilding config instead of reusing the trainer's expression (findings #1, #4)
- Checkpoint round-trip fidelity: ⚠️ — faithful for non-modulated rPPO; **broken for modulated rPPO (#1) and DreamerV3 (#2)**; silent-merge robustness gap (#3)

---

## Recommended direction (for `developer`, not applied here)

- **#1 / #4:** Replace the hand-built dict in `evaluation.py:250-261` with the *same* expression train uses — `modulation_config = config.get('agent.modulation'); if modulation_config is not None and modulation_config.get('type') is None: modulation_config = None` — so eval and train share one source of truth and no future key can drift.
- **#2:** Pass the `Config` object (not a 5-key dict) plus `obs_breakdown=get_observation_breakdown(params)` and the null-normalised `modulation_config` into `DreamerTrainer`, mirroring `train.py:815-817`.
- **#3:** After `nnx.update`, assert that every current-module Param leaf was covered by the restored checkpoint (or at least warn on uncovered leaves), so a structural mismatch fails loudly instead of loading a half-random model.

**Verdict: pipeline has 2 confirmed eval-side defects (modulated-rPPO eval, DreamerV3 eval) + 2 latent-robustness concerns; environment, training, and non-modulated RecurrentPPO checkpoint round-trip are JAX-correct.**

Reviewed by: code-reviewer
