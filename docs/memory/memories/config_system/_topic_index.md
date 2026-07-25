# _topic_index.md — `config_system` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `config_system` topic.

**Folder definition**: Config loader/layering/schema decisions
**Insights**: 9
**Last updated**: 2026-07-26

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-07-26 | 04:19 | [20260726_0419_experiment_eval_config_layer_selector](20260726_0419_experiment_eval_config_layer_selector.md) | experiment-eval opt-in moved from a --experiment-eval CLI flag to a config-layer --eval-config <preset> selector: default evaluation/default.yaml (off) documents every field with notes; experiment_on.yaml extends it and flips enabled:true. Config-selector (like --config) vs behavioral-flag; self-documenting defaults. rPPO-only. |
| 2026-07-26 | 04:16 | [20260726_0416_critical_settings_registry_changelog_guardrail](20260726_0416_critical_settings_registry_changelog_guardrail.md) | Created docs/environment/CONFIG_CRITICAL_SETTINGS.md: a registry of high-impact settings (canonical value + meaning) + a dated change log with a same-commit logging protocol, to prevent silent config drift. Wired as required reading into 5 config/training agents + CLAUDE.md; context-efficient (compact 28-line doc, 1-line CLAUDE.md bullet, conditional reads). |
| 2026-07-26 | 04:15 | [20260726_0415_decay_power_silent_drift_reverted_1p0](20260726_0415_decay_power_silent_drift_reverted_1p0.md) | sensory.decay_power silently drifted 1.0->2.0 inside an unrelated 2026-02-27 'scale to 10x10' commit (def81c1); found via the SAVED config (first lookup returned spurious None — key is under sensory. not olfaction.); reverted to 1.0. Training+probes were consistent at 2.0, just unintended. |
| 2026-07-23 | 19:15 | [20260723_1915_two_level_logging_and_config_layering](20260723_1915_two_level_logging_and_config_layering.md) | Two-level logging: smoothing(window/noise) split from interval(cadence/volume), unit-bearing knobs; Buffer uncapped for rPPO's 128-step rollout. Added dreamer_srl.yaml (unconditional merge). Fixed Dreamer num_envs (was CLI-only, silently 1). Regression I caused: moving checkpoint keys out of default.yaml broke DQN/DRQN/PPO. Principle: default documents EVERY field. |
| 2026-07-03 | 15:07 | `20260703_1507_train_py_ignores_extends_drops_layers` | SEVERE: train.py loads --config via Config.load_yaml (no extends resolution) -> training gets default.yaml + only the file's OWN keys; inherited layers (noise, random-init, all-combined) silently dropped. basic/07 trained noise-off/random-init-off/hiding-4. Latent since v3.0 (2026-06-19); tooling uses load_env_config so it looked fine. Fix: load --config via load_env_config. |
| 2026-06-22 | 17:46 | [20260622_1746_start_injury_dead_needs_random_range](20260622_1746_start_injury_dead_needs_random_range.md) | A plain body.start_injury is DEAD in the reset path — injury starts at 0 unless random_start_injury:true (drawn from [low,high]); pin a fixed non-zero injury via a degenerate range (low==high). |
| 2026-06-19 | 01:14 | [20260619_0114_config_guide_maintenance_contract](20260619_0114_config_guide_maintenance_contract.md) | docs/environment/CONFIG_GUIDE.md (collaborator-facing) + a Maintenance Contract wired into the 5 config-caring agent profiles (read-before / update-on-change) — the anti-rot mechanism for the v3.0 config system, enforced via profiles not memory. |
| 2026-06-19 | 01:13 | [20260619_0113_configurable_initial_state_ranges](20260619_0113_configurable_initial_state_ranges.md) | body.start_{nutrition,injury}_{low,high} expose the start-randomization bounds (replacing hardcoded max/2); conditional-mandatory (read only when random_start_* is true → zero config ripple); unlocks genuinely hungry/injured starts. |
| 2026-06-19 | 01:11 | [20260619_0111_config_v3_extends_layering_default_base](20260619_0111_config_v3_extends_layering_default_base.md) | v3.0 makes default.yaml the canonical base; experiment configs are sparse extends: environment/default overrides deep-merged at load_env_config(); configs/experiment moved to configs/environment/experiment/archive (91 configs). Opt-in, parity-safe. |

---

## Related indexes

| Index | Path | Holds |
|---|---|---|
| Topic registry | [../../ROOT_INDEX.md](../../ROOT_INDEX.md) | All topic-folder metadata |
| Tag dictionary | [../_global_tags.md](../_global_tags.md) | All active tags |
