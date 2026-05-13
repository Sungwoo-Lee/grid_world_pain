# Config Audit — DREAMER_CONVENTIONAL_FIXES_BATTERY

> Pre-flight audit by `env-config-auditor` for the 2-cell DreamerV3 conventional-fixes mini-battery (Cell A1 NoPred + rr=0.0625, Cell A2 Predator + rr=0.0625 + death_penalty=1) on node 113.
>
> Audited: 2026-05-09. Configs: `configs/models/dreamer_v3_rr06.yaml`, `configs/experiment/dreamer_diagnostic/01-PredInterval3_NutGain18_DeathPenalty1.yaml`. Design doc: `docs/experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md`.

## Items

- **1. get_mandatory discipline: PASS** — `body.death_penalty` is loaded via `config.get_mandatory('body.death_penalty')` at `src/environment/config_loader.py:369`. `agent.replay_ratio` is loaded via `config.get_mandatory('agent.replay_ratio')` at `train.py:789` inside `Ratio(...)`. No fallback defaults for either key.

- **2. Single-line delta: PASS (with note)** — Model config `dreamer_v3_rr06.yaml` vs `dreamer_v3.yaml`: exactly one functional delta, `replay_ratio: 0.5 -> 0.0625`. Additional differences are comment additions, trailing-whitespace normalization, and removal of a commented-out XL profile block that was already dead code in the base file — no live key values changed. Experiment config `01-PredInterval3_NutGain18_DeathPenalty1.yaml` vs `01-5X5_PredInterval3_NutGain18.yaml`: exactly one functional delta, `body.death_penalty: 100 -> 1`. Additional differences are comment additions and trailing-whitespace normalization. Both are clean.

- **3. Tag uniqueness: PASS** — Grep across `docs/diary/`, `wandb/`, and `results/` for both `dreamer_conv_NoPred_rr06_s0_n113` and `dreamer_conv_Pred_rr06_dp1_s0_n113` returned no matches. Neither tag exists yet.

- **4. Modality fingerprint: PASS** — Programmatic live comparison confirms byte-identical observation breakdowns across both cells: `{'Satiation': 1, 'Interoceptive Nociception': 1, 'Extero Nociception': 1, 'Olfaction': 5, 'Collision': 5, 'Proprioception': 6}`. Note that `00-5X5_NoPred.yaml` uses `predator_enabled: true` with `count: 0` (not `predator_enabled: false`), and the Pred config has a live predator. Despite this structural difference the observation dimensionality is identical — the predator sensor is not a separate modality, it shares the olfaction/collision channels. No recompile risk from modality-count mismatch.

## Recommendation

Proceed-to-runner. All four checks pass. No blockers.
