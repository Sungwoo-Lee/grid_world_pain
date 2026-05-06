# Config Audit — Continual Learning Config Schedule

**Scope:** New feature — multi-config sweep with stage-rotation logic in `train.py`; new schedule YAML
**Files audited:**
- `train.py` (new code: `ContinualSchedule`, `_build_continual_schedule`, stage-transition block, checkpoint freq lookup)
- `configs/continual/example_schedule.yaml` (new file)
**Audited by:** env-config-auditor
**Date:** 2026-05-06
**Plan:** [CONTINUAL_LEARNING_CONFIG_SCHEDULE.md](../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md)

---

## Summary

The implementation is largely correct and mechanically faithful to the plan. All six explicitly-planned validations fire as `ValueError` at startup. The mandatory-key discipline is sound for the schedule YAML itself. However, three concerns are worth resolving before a long training run: (1) `get_mandatory` has a silent pass-through for empty lists `[]`, so a malformed schedule with `episode_boundaries: []` would not raise at the mandatory-key gate and instead crash inside `_build_continual_schedule` with a less legible error; (2) `boundaries[0] = 0` passes the strictly-increasing check but silently skips stage 0 entirely (the first iteration fires a stage transition before any episode runs); and (3) the PPO and DQN/DRQN algorithms do not write a `stage` key into their checkpoint payloads, so resume on those algorithms silently defaults to `current_stage = 0` via `restored.get('stage', 0)`. None of these are immediate crash risks on the planned RecurrentPPO/DreamerV3 use case, but item (3) is a silent data-corruption risk for any non-RecurrentPPO, non-DreamerV3 curriculum run. The perceptual-noise modality system is sound per-stage; the obs_dim probe catches total-dim changes but cannot catch same-total modality swaps.

---

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| CONCERN | `src/utils/config.py:46` | `get_mandatory` checks `val is None` — an empty list `[]` is falsy in Python but `None`-check passes it through. `episode_boundaries: []` or `checkpoint_frequencies: []` would return `[]` without raising, then fail opaquely at `len(boundaries) != len(paths)` with a length message instead of a "missing mandatory key" message. Not a crash blocker on well-formed YAML, but degrades diagnostics. | Add `if val is None or (isinstance(val, (list, dict)) and len(val) == 0)` check in `get_mandatory`, or add an explicit `if not paths` guard after the mandatory-key reads. |
| CONCERN | `train.py:166-175` (`_build_continual_schedule`) | `boundaries[0] = 0` passes the strictly-increasing check (`sorted([0, 1000]) == [0, 1000]` and `len(set(...)) == 2`). At the first training iteration, `stage_for_episode(0)` returns `1` (because `0 < 0` is False), and `new_stage (1) != current_stage (0)` fires the transition immediately — stage 0 is used only to build the initial env params but never runs any episodes. The user would observe `[STAGE] 0 -> 1 at ep=0` in the first log line with no indication that stage 0 was skipped. | Add `if boundaries[0] <= 0: raise ValueError(f"episode_boundaries[0] must be > 0 (got {boundaries[0]}); stage 0 must run for at least one episode.")` after the strictly-increasing check at `train.py:174`. |
| CONCERN | `train.py:1961-1983` | The `ckpt_data` block only handles `algorithm == "RecurrentPPO"` and `algorithm == "DreamerV3"`. PPO, DQN, and DRQN all fall through to `ckpt_data = {}`, which means `if ckpt_data:` is False and no checkpoint is saved for those algorithms at all in the current code. More critically for this feature: even if those branches were filled in later, they would not include `'stage': current_stage`, so `restored.get('stage', 0)` in the resume block would silently default to stage 0 regardless of where in the curriculum the run was when interrupted. The resume block (`train.py:979`) only restores `current_stage` from checkpoint for RecurrentPPO but also only has one `restored.get('stage', 0)` call which defaults to 0 — on DQN/DRQN this would cause a multi-stage jump at iteration 1 post-resume. | For now the plan explicitly targets RecurrentPPO and DreamerV3 — document this limitation explicitly in a code comment at `train.py:1961`. If PPO/DQN/DRQN continual mode is ever added, the `stage` key must be included and the resume block extended. |
| CONCERN | `train.py:461-478` (obs/action dim probe) | The probe measures total `obs_dim` via `probe_obs.shape[-1]`. This catches cases where a sensor is added or removed (changing total dims). It does NOT catch modality swaps where the total dim happens to stay the same — e.g., stage 0 has `olfactory_enabled: true` with `vector_size: 5` and stage 1 disables olfaction but enables visual with `visual_sensor_range: 0` (visual dim = 8 ≠ 5, caught) — OR if `vector_size: 8` and visual also = 8, the probe passes and the model silently trains on a different sensor layout at stage 1. This is not a crash, but the trained policy's earlier feature weights now receive a semantically different signal without any warning. | Add a secondary check that compares `noise_modality_order` tuples (or the full `get_observation_breakdown(params)` dict) across stages, not just total dim. A name-order mismatch with equal total dim is a silent curriculum-corruption risk. |
| NIT | `train.py:162` (`_build_continual_schedule`) | The deep-copy strategy `Config(yaml.safe_load(yaml.dump(base_config.to_dict())))` round-trips through YAML serialization. Python `yaml.dump` does not guarantee round-trip fidelity for all Python types (e.g., tuples become sequences, non-string keys may be stringified, `None` vs `null` edge cases). For the types actually used in these configs (nested dicts of scalars and lists) this is safe, but it is fragile if any config value contains a tuple. | Use `copy.deepcopy(base_config.to_dict())` instead: `import copy; stage_cfg = Config(copy.deepcopy(base_config.to_dict()))`. |
| NIT | `configs/continual/example_schedule.yaml` | The file has three-element arrays for a 3-stage schedule. The `configs/experiment/curriculum_basic/` example in the plan has 3 YAML files. Consistent — no issue. | N/A — example is correct as written. |
| NIT | `train.py:991-996` (`_stage_tag`) | `_stage_tag` is a closure over `schedule` and `current_stage`. `current_stage` is a plain Python int, not a reference — in Python closures, re-binding `current_stage = new_stage` at `train.py:1062` updates the enclosing scope's variable, and the closure reads the current value at call time. This works correctly because `_stage_tag` reads `current_stage` at call time, not capture time. No bug, but worth a comment for reviewers unfamiliar with Python closure semantics. | Add a brief comment: `# current_stage is read at call time from the enclosing scope`. |

---

## Checklist

- [x] (1) Observation/Noise Modality Consistency — Per-stage noise system is self-contained: each stage's `params.noise_modality_order` is built from that stage's YAML. After env rebuild, `apply_perceptual_noise` uses the new params. Within-stage mismatch (enabled sensor with no noise entry) would KeyError at JIT trace — same as single-config mode. No new risk introduced. Same-total-dim modality swap across stages is a concern (Finding 4 above) but is not a noise system crash.
- [x] (2) Mandatory-Key Discipline — `continual.episode_boundaries` and `continual.checkpoint_frequencies` both loaded via `get_mandatory`. Six validation checks fire as `ValueError` before training. Empty-list edge case degrades diagnostics but does not silently proceed to training. No `config.get(..., default)` calls on critical schedule keys.
- [x] (3) Static-Field JIT Recompile Risk — `load_env_params` is called only inside `if new_stage != current_stage:` (stage transition gate), which fires at most `N_stages - 1` times per run. No per-iteration recompile risk. The `stage_for_episode` function is pure Python with no JAX interaction. Stage transition correctly rebuilds `ParallelEnv(params)` which triggers one recompile per transition — accepted per plan.
- [x] (4) Known Latent-Bug Recurrences — N/A for the schedule YAML itself. The stage YAML files authored by the user inherit all latent-bug risks (overeating_death, random_start_pos, etc.) from the base config. No new latent bugs introduced by the schedule mechanism. `random_start_pos: true` in any stage config still carries the step-0 contact risk.
- [x] (5) Schema Padding and Modality-Count Quirks — No new sensors added. Padding length (13) unchanged. Modality count per stage is controlled entirely by each stage's YAML; the 13-slot pad is stable.
- [N/A] (6) Cross-Config Coherence (sweep) — Not a parameter sweep in the traditional sense. Stages are sequential, not parallel. Seed is a single scalar shared across all stages (initialized once, threaded via `key`). Stage configs are validated for obs/action dim consistency at startup. Non-swept fields (num_envs, architecture, seed) are shared because the model and env are rebuilt only for env params, not the agent.

---

## Additional Finding: Checkpoint Resume Stage Reconstruction

When `--load-checkpoint` is used on a continual run and the checkpoint was saved by a build that did not include `'stage'` in the payload (e.g., first deployment, or PPO/DQN algorithms), `restored.get('stage', 0)` silently returns 0. On the first iteration, `stage_for_episode(total_episodes_completed)` will return the correct stage (e.g., stage 2 if `total_episodes_completed = 2500`), and `new_stage (2) != current_stage (0)` will fire a multi-stage jump, silently skipping stages 0 and 1's env rebuild sequence and going directly to stage 2. The env gets rebuilt correctly for stage 2, but the log will show `[STAGE] 0:stage_a -> 2:stage_c` with no indication that stage 1 was bypassed. This is a one-iteration correction and not a crash, but it produces a misleading log line. Logged as a documentation/diagnostic concern, not a blocker.

---

## Conclusion

WARNINGS — three CONCERN-level findings. No crash blockers for the primary RecurrentPPO/DreamerV3 use case. Fix `boundaries[0] > 0` guard (Finding 2) before committing, as it is a one-line addition that prevents a silent stage-skip that would be difficult to diagnose during a 7-day run. The other findings are lower priority.

Audited by: env-config-auditor
