# Config-layer diagnosis — findings (2026-07-23)

**Scope**: the non-archived `configs/` tree (environment, models, train, continual, eval_sweeps, evaluation, logger, verification, visualization), prioritized by what recent launches actually use (`train_command-agent.sh` tail: basic/03, basic/04, basic04_variants 01–08, rPPO default/GAE/size presets, rPPO+Dreamer train layers).
**Method**: read CONFIG_GUIDE + known-bugs context first; bulk-loaded **all 170 non-archived env + verification configs through the project loader (`load_env_config` → `load_env_params`) — 170 OK / 0 FAIL**, so every `extends:` chain resolves and every mandatory env key is satisfied post-merge. Sweep families verified by *loaded-value* flatten-diffs, not by eyeballing YAML. Report-only; nothing modified.

Known rows honored (not re-reported as new): dreamer gamma typo (verified fixed — all 18 files carry `0.996996996996997`), `*_het_*` = noise heterogeneity (not a bug), fractional attack/detection inclusive-int fix, the 4 known config-boundary silent-failure traps, dead `lr_critic`.

---

## Findings

### P1-1 — `configs/environment/experiment/basic/05-sensory_noise_10x10.yaml`: "interoception kept CLEAN" is false in the merged config

**Claim**: the hardest live level's header (and its whole hypervigilance rationale) says satiation / interoceptive-nociception / extero-nociception are kept clean so the injury-gating signal stays reliable — but the merged config leaves **constant Gaussian noise sigma=0.1 on all three channels**.

**Evidence (loaded with the project loader)**:
```
satiation                 {'mode': 'constant', 'sigma': 0.1, 'injury_noise_scale': 0.0, ...}
interoceptive_nociception {'mode': 'constant', 'sigma': 0.1, 'injury_noise_scale': 0.0, ...}
extero_nociception        {'mode': 'constant', 'sigma': 0.1, 'injury_noise_scale': 0.0, ...}
```
Mechanism: the child overrides only `mode` + `injury_noise_scale`; `sigma` deep-merges through from `configs/environment/default.yaml` (0.1 on those three modalities). sigma=0.1 on a [0,1]-clipped interoceptive channel is 10x the "tiny" collision noise (0.01) the file's own footnote uses as its benchmark for acceptable residual noise. The header's parenthetical "(no injury coupling)" makes intent ambiguous, but the design argument ("if injury perception got noisy, the gating trigger would be corrupted") reads as clean-clean.

**Exposure**: this file's lineage (as the then-`06-sensory_noise_10x10`) trained `rppo_basic06_noise_n110` (2026-07-03 re-launch, 10M eps), and it is the designated next level after the currently-running basic/04 sweeps.

**Fix direction**: add explicit `sigma: 0.0` to the three interoception overrides (if clean is the intent) or rewrite the header to say "no injury coupling; baseline sigma=0.1 retained" — decide before the next noise-level launch.

### P1-2 — `--wandb-job-type` / `wandb.job_type` is silently dropped: the documented WandB filter convention is not honored

**Claim**: every agent launch passes `--wandb-job-type prod` (the launch script documents job-type as the operational-category filter field), and `configs/logger/wandb.yaml` declares `wandb.job_type` — but `train.py` never passes `job_type` to `wandb.init`.

**Evidence**: `train.py:403` stores the CLI flag into config (`config.set('wandb.job_type', ...)`), yet the single `wandb.init(**wandb_kwargs)` call site (train.py:668-699) builds kwargs with project/entity/group/name only — no `job_type`. It survives only as a config-payload field (via `**config.to_dict()`). `dreamer_srl_main.py` has zero `job_type` references. So the native WandB job-type facet the convention promises ("filter prod vs debug/pilot/ablation") is unset on every run; filtering works only via the buried `Config.wandb.job_type` field.

**Fix direction**: add `"job_type": ...` to `wandb_kwargs` (code change; hand to `senior-developer` — a config-boundary silent failure *adjacent to but not among* the 4 known traps).

### P2-1 — `configs/continual/basic_curriculum_schedule.yaml` + `_longL4.yaml`: embedded run commands point at the wrong stage directory

**Claim**: both schedules' "Run with:" blocks say `--configs-dir configs/environment/experiment/basic`, but `basic/` now holds **6** stage files whose contents no longer match the schedule's own stage list (schedule names `02-fast_predator_8x8`, `04-far_sight_predator_10x10` — those live in `basic_curriculum/`; `basic/` has `02-predator_and_rabbit_10x10`, `04-jump_attack_10x10`, `05-sensory_noise_10x10`). The frozen 5-stage dir `configs/environment/experiment/basic_curriculum/` exists precisely for this (per the 2026-06-27 relaunch note in train_command-agent.sh).

**Evidence**: `ls configs/environment/experiment/basic/` = 00…05 (6 files); schedules carry 5 boundaries. `train.py:173-180` hard-fails on the length mismatch (loud today), but if `basic/` ever returns to 5 files the alphabetic pickup would silently bind the schedule to the *wrong* difficulty ladder.

**Fix direction**: edit both schedule headers to `--configs-dir configs/environment/experiment/basic_curriculum`.

### P2-2 — `configs/continual/dreamer_srl_3stage_curric_T1..T4.yaml` (+ `dreamer_srl_3stage_size_curriculum.yaml`): stage-source path never existed and the stages are archived

**Claim**: the T1-T4 schedules describe their stage configs as living in `configs/experiment/dreamer_srl_curriculum/` — a path that does not exist under `configs/` (wrong prefix), and the actual stage dir has been archived to `configs/environment/experiment/archive/dreamer_srl_curriculum/`.

**Evidence**: `ls configs/experiment` → No such file or directory; `find configs -type d -name "*dreamer*"` → only `models/dreamer_srl` + three `archive/` dirs.

**Fix direction**: update the header path (noting the archive location), or archive the schedules alongside their stages if the T-budget study is closed.

### P2-3 — `configs/logger/wandb.yaml`: `job_type: "defualt"` typo

**Claim**: the fallback job type is misspelled ("defualt"). Currently masked by P1-2 (job_type never reaches WandB natively), but the moment P1-2 is fixed, every run launched without `--wandb-job-type` would be labeled `defualt`. **Fix direction**: `"default"`.

### P2-4 — Dead keys in all 12 active rPPO agent configs: `frame_stack`, `actor_fc_layers`, `critic_fc_layers` (+ depth on the known dead-key trap)

**Claim**: beyond the known-OPEN dead keys (`fc_layers`, `unimodal_overrides`), every config in `configs/models/recurrent_ppo/` also declares `frame_stack: 1`, `actor_fc_layers: [128,128]`, `critic_fc_layers: [128,128]` — none is read anywhere: `src/models/recurrent_ppo_network.py:257-262` hardcodes actor/critic heads as `hidden_size → hidden_size → out`; no reader for `frame_stack` exists in the rPPO path.

**Depth on the known rows** (not new findings): (a) in the **actively-running basic04 size sweep** (XS/S/M/L/XL), `hierarchical_params.unimodal_overrides.visual/olfaction: [128,128]` is silently ignored — all unimodal encoders follow `default_mlp` (network.py:85-92 reads only `default_mlp` + `multimodal_hub`), so the presets' "visual/olfaction stay [128,128]" annotation is not what runs; (b) the known-bugs row listing `percept_bias_init` as dead is **stale** — it is now read (network.py:269, hard-keyed, mandatory when modulation is on), as are `mod_hidden_size`/`grouping_size`/`memory_bias_init`/`temp_clip`/`memory_clip` (network.py:266-273). Suggest `bug-curator` update that row.

**Fix direction**: delete the three dead keys from the 12 configs (or implement readers); correct the size-preset comments about unimodal overrides.

### P2-5 — Latent: perceptual-noise parser silently drops typo'd modality names and unknown mode strings

**Claim**: `_parse_noise_config` (`src/environment/config_loader.py:1506-1541`) filters modality keys through `_YAML_KEY_TO_SENSOR_NAME` with `if k in ...` — a misspelled modality (e.g. `olfactory:` for `olfaction:`) is dropped without error; `_parse_mode` maps any unknown mode string (e.g. `state-dependent`) to `0` = noise off, silently. The block also uses `.get('sigma', 0.0)`-style fallbacks, contra the no-fallback rule. **No current config is affected** (all 170 load with valid names/modes) — same genus as the 4 known config-boundary traps; belongs on that list. **Fix direction**: raise on unknown modality keys and unknown mode strings.

### P2-6 — `configs/environment/default.yaml:266`: stale noise-index comment

"Key order here is the single source of truth for noise array indices (0-8)" — the block now has **10** modalities (0-9; `interoceptive_nociception` was added). Inline per-modality comments are correct; only the header range is stale. **Fix direction**: "(0-9)".

### P2-7 — `train_command-agent.sh`: two *uncommented* (active) launch blocks reference renamed configs

Lines 685 and 709 (the 2026-07-03 extends-fix re-launches) point at `configs/environment/experiment/basic/06-sensory_noise_10x10.yaml` and `.../07-jump_attack_10x10.yaml`, since renamed to `05-`/`04-`. With `set -euo pipefail`, re-executing the script aborts at the first FileNotFoundError (loud, and the file is an audit record not a runbook) — but the active lines no longer describe reproducible commands. **Fix direction**: comment the stale blocks out or update paths (training-runner's file).

---

## Reviewed but clean

- **basic/00–04** — chains resolve; `attack_range: [0,0]` / `attack_success_rate: 0.0` declared explicitly at pre-jump levels per the no-implicit-defaults policy; body random-start flags + ranges coherent; `overeating_death` stays false everywhere (no recurrence); no `property:`-for-`properties:` misspelling anywhere.
- **basic04_variants/01–08** (live sweep) — loaded flatten-diff vs `basic/04` proves each variant differs in *exactly* its claimed knobs and nothing else (v01 move_interval; v02 attack_range; v03 damage; v04 all three; v05/06 asr only; v07/08 range+damage+asr with move fixed). Exemplary sweep hygiene.
- **basic_curriculum/00–04** — 5 files matching the 5-boundary schedules (the frozen dir does its job; only the schedule *comments* are stale, P2-1).
- **olfactory_ambiguity + olfactory_ambiguity_lindecay** — within-family axes minimal (s → smell-vector separation on properties channels 1/2; sig → properties_std); each lindecay cell differs from its matched cell **only** in `sensory.decay_power` 2.0→1.0 (loaded proof).
- **behavior_probes (core + explore, all sub-families)** — all load; spot-checked pairs differ only in the intended factor (e.g. avoid_pred inj00 vs inj70: only `start_injury_low/high` 0→70).
- **hypervig_scarcity** — scarce vs abundant is a deliberate composite (food count / max_consumption / regeneration_delay / layout); noted, not a defect.
- **models/dreamer_srl (18 files)** — gamma fix intact everywhere; all scientific-notation values are dot-form (`1.0e-4`) which PyYAML parses as float — **no string-typed-number trap in configs/** (bare `1e-4` would be a string; none found).
- **models/recurrent_ppo NMN family** — every `modulation.*` key is consumed (network.py:266-273); `temp_clip` sweep annotations match the `neuromodulator.py:71` default-envelope claim.
- **train/{default,recurrent_ppo,dreamer_srl}** — three-layer merge story coherent; intentional duplication documented; `num_envs` / `checkpoint_frequency` / `max_checkpoints_to_keep` / logging blocks consistent with the get_mandatory call sites their comments quote.
- **continual/nmn_double_return_stages/01–05** — all 5 standalone stage configs build EnvParams cleanly.
- **eval_sweeps, evaluation/default, visualization/default, verification (6 files)** — load clean; no dead-key or type issues found.
- **Type traps / booleans** — no `yes/no/on/off` booleans, no quoted numerics in the non-archived tree.
- **JIT static-field check** — sweep members run as separate processes (no cross-config recompile risk); the only static-field variation inside a single run is curriculum grid-size change at stage boundaries, which recompiles by design.
- **get_mandatory coverage** — every rPPO-path mandatory agent key (algorithm, sequence_length, hidden_size, lr_actor, rnn_type, activation, return_mode, K_epochs, gamma, gae_lambda, eps_clip, entropy_coef, vf_coef, max_grad_norm, use_layer_norm) present in all 12 active rPPO configs; env mandatory keys proven satisfied by the 170/170 EnvParams bulk build.

## Deliberately skipped as inactive

`configs/environment/experiment/archive/**` (~91 frozen standalone configs; load-verified only), `configs/models/archive/**` (NNX Dreamer, abandoned 2026-07-09), `configs/models/{dqn,drqn,ppo,q_learning}/` (no recent launches), `configs/continual/{example_schedule,basic_5x5_NoPred_to_PredInt3,dreamer_curriculum_*}.yaml` (legacy schedules), `configs/models/dreamer_srl/agent_xs.yaml` + smoke variants beyond structural scan.
