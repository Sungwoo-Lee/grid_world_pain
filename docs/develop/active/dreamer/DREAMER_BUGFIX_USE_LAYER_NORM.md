---
title: "Bugfix: DreamerV3 training crashes with missing agent.use_layer_norm"
topic: dreamer
status: active
created: 2026-05-06
last_updated: 2026-05-06
---

# Bugfix: DreamerV3 training crashes with `ValueError: Configuration key 'agent.use_layer_norm' is required but missing.`

> **Status**: PLANNED
> **Opened**: 2026-05-06
> **Related**:
> - [DREAMER_IMPLEMENTATION_AUDIT.md](DREAMER_IMPLEMENTATION_AUDIT.md) (§7 — recent dreamer additions)
> - Reproducer logs:
>   - `tmp/20260506_170049_dreamer_repro_cuda3.log` (failure on `cuda:3` — environment-level)
>   - `tmp/20260506_170049_dreamer_repro_cuda0.log` (failure on `cuda:0` — code-level, addressed here)
> - Originating commit: `f1db0f0 refactor: Update modulation handling …` (Mar 18 2026)

---

## Context

The active command in `train_command-new.sh`:

```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/labmeeting/basic-01-PredInterval3_NutGain18.yaml \
  --agent_config configs/models/dreamer_v3/dreamer_v3.yaml \
  --num-envs 16 \
  --episodes 10000000 \
  --checkpoint-frequency 100000 \
  --device cuda:3 \
  --log-interval 10 \
  --tag "dreamer_v3_basic01_PredInterval3_NutGain18"
```

errors out at trainer construction. There are **two distinct failure modes** depending on the device argument; only one of them is a code bug, which is what this plan fixes.

### Failure A (environment / not a code bug): `--device cuda:3`

`nvidia-smi` on this box reports GPUs **0 and 1 only** (2× RTX 4090). `train.py` lines 47–51 set `CUDA_VISIBLE_DEVICES="3"` *before* JAX is imported, then JAX's CUDA plugin init fails:

```
RuntimeError: jaxlib/cuda/versions_helpers.cc:113: operation cuInit(0) failed: CUDA_ERROR_NO_DEVICE
…
RuntimeError: Backend 'cuda' is not in the list of known backends: ['cpu', 'tpu'].
```

This is a user-side configuration error — `cuda:3` does not exist on this machine. The user should change `--device` in `train_command-new.sh` to `cuda:0` or `cuda:1`. **No code fix required**, but worth noting that the `except` block at `train.py:179–181` then crashes again because its fallback `jax.devices()[0]` re-triggers the same dead `cuda` backend (small UX nit, out of scope here).

### Failure B (code bug — root cause this plan fixes): `--device cuda:0`

Re-running the same command with `--device cuda:0` (real GPU) gets past JAX init, loads all configs, builds the env, prints the CONFIGURATION banner, then crashes at trainer construction:

```
File ".../train.py", line 558, in main
  trainer = DreamerTrainer(input_dim, action_dim, agent_config, rngs=nnx.Rngs(init_key), …)
File ".../src/models/dreamer_v3_trainer.py", line 75, in __init__
  'use_layer_norm': config.get_mandatory('agent.use_layer_norm', bool),
File ".../src/utils/config.py", line 47, in get_mandatory
  raise ValueError(f"Strict Config: Configuration key '{key}' is required but missing.")
ValueError: Strict Config: Configuration key 'agent.use_layer_norm' is required but missing.
```

This is the actionable bug.

## Analysis

### Root cause (one-liner)

`configs/models/dreamer_v3/dreamer_v3.yaml` is missing the `agent.use_layer_norm` key, which became mandatory in commit `f1db0f0` but was only added to the *PPO* configs (`recurrent_ppo.yaml:20`, `neuromodulated_ppo.yaml:20`) — the two DreamerV3 configs were left out of that commit.

### Evidence

`git show --stat f1db0f0`:

```
configs/models/ppo/neuromodulated_ppo.yaml     |  3 +-
configs/models/recurrent_ppo/recurrent_ppo.yaml          |  1 +
src/models/dreamer_v3_nnx.py               | 86 +++++++++++++++++-------------
src/models/dreamer_v3_trainer.py           | 32 +++++++----
src/models/recurrent_ppo_network.py        | 77 ++++++++++++++------------
…
```

The commit added `use_layer_norm: true` to the PPO YAMLs and added `config.get_mandatory('agent.use_layer_norm', bool)` to **both** PPO and DreamerV3 trainers — but did not touch `configs/models/dreamer_v3/dreamer_v3.yaml` or `configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml`.

Verification:

```
$ grep -n "use_layer_norm" configs/models/recurrent_ppo/recurrent_ppo.yaml configs/models/ppo/neuromodulated_ppo.yaml
configs/models/recurrent_ppo/recurrent_ppo.yaml:20:  use_layer_norm: true
configs/models/ppo/neuromodulated_ppo.yaml:20:  use_layer_norm: true

$ grep -n "use_layer_norm" configs/models/dreamer_v3/dreamer_v3.yaml configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml
(no matches)
```

### Where the key is consumed

`src/models/dreamer_v3_trainer.py:75`:
```python
'use_layer_norm': config.get_mandatory('agent.use_layer_norm', bool),
```
…packed into `agent_config` and forwarded to the world model. Used at `src/models/dreamer_v3_nnx.py:515–521`:
```python
self.use_layer_norm = config['use_layer_norm']  # mandatory — no fallback default
if self.use_layer_norm:
    if self.encoder.mode == 'hierarchical':
        self.mod_unimodal_ln = nnx.LayerNorm(encoder_dim, rngs=rngs)
        self.mod_multimodal_ln = nnx.LayerNorm(encoder_dim, rngs=rngs)
    else:  # flat mode
        self.mod_flat_ln = nnx.LayerNorm(encoder_dim, rngs=rngs)
```

`use_layer_norm` is orthogonal to `modulation.type` (per the originating refactor description), so it must be declared in **every** agent YAML — including the plain DreamerV3 ones — regardless of whether modulation is enabled.

### Hypotheses ruled out

Following the senior-developer's hypothesis list:

- **Hierarchical encoder/decoder mismatch** — Not the cause. The error fires at trainer construction (`get_mandatory` for `use_layer_norm`), strictly *before* any encoder build. The CONFIGURATION banner in the cuda:0 reproducer log shows `Observation Dim: 19 (Satiation=1, Interoceptive Nociception=1, Extero Nociception=1, Olfaction=5, Collision=5, Proprioception=6)`, i.e. obs breakdown is fine.
- **Three-pool mixture sampler keys missing** — All five keys (`sampling_mode`, `mixture_positive_slots`, `mixture_recent_slots`, `mixture_recent_window`, `positive_buffer_capacity`) are present in `configs/models/dreamer_v3/dreamer_v3.yaml:18–22`. Not the cause for this run. **Note**: the *neuromodulated_dreamer_v3.yaml* file is missing all five — see "Adjacent issue" below.
- **Neuromodulation tap points** — Confirmed not the cause; `dreamer_v3.yaml:74–75` has `modulation.type: null`, and `dreamer_v3_nnx.py:491–493` correctly treats null as disabled.
- **Pre-computed encoder embeddings shape mismatch** — Not the cause; never reached.
- **Experiment YAML existence** — `configs/experiment/labmeeting/basic-01-PredInterval3_NutGain18.yaml` exists and loads fine.

### Default value choice (`true` vs `false`)

Both PPO YAMLs set `use_layer_norm: true`. The originating commit message says "ensured LayerNorm integration is consistent with the new architecture", and the code paths that activate when `use_layer_norm: true` (`src/models/dreamer_v3_nnx.py:516–521`) are clearly the intended baseline (modulation-orthogonal LayerNorm on encoder pre-activations). We therefore mirror the PPO default and use `true`.

### Adjacent issue (flagged but **not fixed** by this plan)

`configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml` has the same problem **plus** is missing the entire mixture-sampling block:

```
$ grep -nE "sampling_mode|mixture_positive_slots|mixture_recent_slots|mixture_recent_window|positive_buffer_capacity|use_layer_norm" \
    configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml
(no matches)
```

Any user invoking that config will hit `get_mandatory` on `agent.sampling_mode` (line 742 of the trainer) before even reaching `agent.use_layer_norm`. Since this plan is scoped to the user's reported failure (the `dreamer_v3` config), I am **not** modifying `neuromodulated_dreamer_v3.yaml` here. If/when the user wants neuromodulated runs, that needs a separate small bugfix plan — recommend adding both blocks in one go using `dreamer_v3.yaml` as the template. Cross-reference back to this plan when that one is opened.

## Implementation Plan

### Design

Single-line YAML insert. Mirror the placement and value used by the PPO configs (`recurrent_ppo.yaml:20`, `neuromodulated_ppo.yaml:20`): `use_layer_norm: true`, placed inside the top-level `agent:` block, immediately *before* the "Encoding Configuration" comment. This keeps the key adjacent to the other architecture flags (`unimix`, `entropy_scale`) and matches the structural ordering reviewers expect by analogy with the PPO configs.

No code changes. No Python changes. The `config.get_mandatory` call at `dreamer_v3_trainer.py:75` is the **correct** behaviour per the project's "no fallback defaults" rule (`CLAUDE.md` → "Configuration Protocol"), so the fix is at the YAML, not at the call site.

### File Changes

#### `configs/models/dreamer_v3/dreamer_v3.yaml` (insert one line at line 42)

Add the single key `use_layer_norm: true` between `unimix` (line 41) and the "Encoding Configuration" comment (line 43).

```yaml
# BEFORE (lines 40–44):
  entropy_scale: 3e-4  # Hafner 2023 standard value
  unimix: 0.01

  # Encoding Configuration
  encoding_mode: "hierarchical"  # "flat" or "hierarchical"

# AFTER (lines 40–46):
  entropy_scale: 3e-4  # Hafner 2023 standard value
  unimix: 0.01
  use_layer_norm: true  # Orthogonal to modulation.type — applies LayerNorm to encoder pre-activations.
                         # Mirrors recurrent_ppo.yaml:20 / neuromodulated_ppo.yaml:20. Mandatory since commit f1db0f0.

  # Encoding Configuration
  encoding_mode: "hierarchical"  # "flat" or "hierarchical"
```

That is the **only** code/config change required to make the user's command run.

### Regression test

#### `scratch/test_dreamer_v3_config_smoke.py` (new file)

A minimal, fast (no GPU, no env, no model-build) smoke test that asserts every mandatory `agent.*` key referenced in `dreamer_v3_trainer.py` is declared in `configs/models/dreamer_v3/dreamer_v3.yaml`. Catches future "added a `get_mandatory` call but forgot to update the YAML" regressions for this exact config.

Place under `scratch/` (matches existing project convention — see `scratch/test_fallback.py`). Runnable with the project python:

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scratch/test_dreamer_v3_config_smoke.py
```

Test body (developer agent to implement — sketch):

```python
"""Smoke test: every mandatory agent.* key consumed by DreamerTrainer.__init__
is present in configs/models/dreamer_v3/dreamer_v3.yaml. No GPU, no env, no model build.

Run:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python scratch/test_dreamer_v3_config_smoke.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config

# Hard-coded list of mandatory agent.* keys read in DreamerTrainer.__init__
# (src/models/dreamer_v3_trainer.py:64–80). Update when that constructor changes.
MANDATORY_KEYS = [
    'agent.encoder_dim',
    'agent.encoder_fc_layers',
    'agent.rssm_deter_dim',
    'agent.rssm_stoch_dim',
    'agent.rssm_classes',
    'agent.decoder_fc_layers',
    'agent.reward_fc_layers',
    'agent.continue_fc_layers',
    'agent.actor_fc_layers',
    'agent.critic_fc_layers',
    'agent.use_layer_norm',
    'agent.encoding_mode',
    'agent.model_lr',
    'agent.actor_lr',
    'agent.value_lr',
    # Mixture sampler (mandatory whenever sampling_mode is read at line 742)
    'agent.sampling_mode',
]

cfg = Config.load_yaml('configs/models/dreamer_v3/dreamer_v3.yaml')
missing = []
for key in MANDATORY_KEYS:
    try:
        cfg.get_mandatory(key)
    except ValueError as e:
        missing.append((key, str(e)))

if missing:
    print('FAIL: dreamer_v3.yaml is missing mandatory keys:')
    for k, msg in missing:
        print(f'  - {k}')
    sys.exit(1)

print(f'OK: all {len(MANDATORY_KEYS)} mandatory keys present in dreamer_v3.yaml')
```

(The developer agent should adjust the imports / path-fix idiom to match other `scratch/` scripts and may shorten the docstring. The key list above is the contract — keep it complete.)

### Test plan

After the YAML edit:

1. Run the smoke test:
   ```bash
   /home/vncuser/miniconda3/envs/grid_world_pain/bin/python scratch/test_dreamer_v3_config_smoke.py
   ```
   Expect: `OK: all 16 mandatory keys present in dreamer_v3.yaml`, exit 0.

2. Re-run the user's exact command on a **valid** GPU (cuda:0 — change in train_command-new.sh OR pass on CLI for testing) with a short timeout, asserting it gets past trainer construction and into the training loop. The previous failure point was `train.py:558` → `dreamer_v3_trainer.py:75`. After the fix, the command should print the iteration progress bar within ~30–60s, at which point we kill it.

   ```bash
   timeout 90 /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
     --config configs/experiment/labmeeting/basic-01-PredInterval3_NutGain18.yaml \
     --agent_config configs/models/dreamer_v3/dreamer_v3.yaml \
     --num-envs 16 --episodes 10000 --checkpoint-frequency 100000 \
     --device cuda:0 --log-interval 10 --no-wandb \
     --tag "smoke_dreamer_v3_post_fix" 2>&1 | tail -40
   ```

   Pass criterion: stdout shows the tqdm iteration bar (e.g. `it/s` rate) and no `ValueError` traceback. The process being killed by `timeout` after starting to iterate is the success signal.

3. **Do NOT** test with `--device cuda:3` — that fails for unrelated environment reasons (no such GPU on this box). The user should fix the `--device` argument in `train_command-new.sh` separately to one of the available GPUs (0 or 1).

### Checkpoints

- [x] **Checkpoint 1** — After the YAML edit, `grep -n "use_layer_norm" configs/models/dreamer_v3/dreamer_v3.yaml` returns exactly one match at line 42, value `true`. `neuromodulated_dreamer_v3.yaml` confirmed untouched.
- [x] **Checkpoint 2** — Smoke test (`scratch/test_dreamer_v3_config_smoke.py`) prints `OK: all 16 mandatory keys present in dreamer_v3.yaml` and exits 0.
- [x] **Checkpoint 3** — `timeout 90 … --device cuda:0 …` produces tqdm iteration bar at 2.64it/s, iteration 1 logs Loss/Rew/R_MAE/Ent, no `ValueError`. Process killed by signal 15 (timeout), exit code 0.

## Implementation Report

> **Implemented by**: developer (claude-sonnet-4-6)
> **Date**: 2026-05-06

### Summary

Two files changed:

1. **`scratch/test_dreamer_v3_config_smoke.py`** (new file) — Minimal smoke test asserting all 16 mandatory `agent.*` keys from `DreamerTrainer.__init__` are present in `configs/models/dreamer_v3/dreamer_v3.yaml`. Uses absolute path derivation (`PROJECT_ROOT = dirname(dirname(abspath(__file__)))`) to resolve the config file correctly regardless of cwd. Created before the YAML fix per plan instructions.

2. **`configs/models/dreamer_v3/dreamer_v3.yaml`** (+2 lines at line 42) — Inserted `use_layer_norm: true` with its explanatory comment between `unimix: 0.01` (line 41) and the `# Encoding Configuration` comment (line 43), exactly as specified in the plan's File Changes section.

No other files were modified. `configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml` was intentionally left untouched (out of scope per plan).

### Pre-fix Regression Test Output (FAIL — expected)

```
FAIL: dreamer_v3.yaml is missing mandatory keys:
  - agent.use_layer_norm
```
Exit code: 1

### Post-fix Regression Test Output (OK)

```
OK: all 16 mandatory keys present in dreamer_v3.yaml
```
Exit code: 0

### Smoke Training Tail (last ~30 lines after `timeout 90`)

```
Config saved to: results/JAX_DreamerV3/20260506-171145_smoke_dreamer_v3_post_fix/models/config.yaml

============================================================
================ JAX/FLAX RL CONFIGURATION =================
============================================================

[Environment]
  ● Grid Size................ 5x5
  ● Max Steps................ 500
  ● Mode..................... Conventional (Goal-driven)
  ● Injury System............ ENABLED

[Training]
  ● Framework................ JAX/Flax NNX
  ● Total Timesteps.......... 80,000,000
  ● Parallel Envs............ 16
  ● Steps/Iteration.......... 128
  ● Seed..................... 42
  ● WandB.................... Disabled

Action Dim: 6
Observation Dim: 19 (Satiation=1, Interoceptive Nociception=1, Extero Nociception=1, Olfaction=5, Collision=5, Proprioception=6)

Training:   0%|          | 0/10000 [00:00<?, ?it/s]
Training:   1%|          | 73/10000 [00:27<1:02:42,  2.64it/s]
Training:   1%|          | 73/10000 [01:46<4:00:41,  1.45s/it, Iter=1, Loss=L: 7.80, Rew=-200.00, R_MAE=7.457, Ent=1.73]

⚠️  Shutdown signal received (signal 15). Finishing current iteration...
Training complete. Results saved to results/JAX_DreamerV3/20260506-171145_smoke_dreamer_v3_post_fix
```
Exit code: 0 (killed by `timeout`, not by exception).

The training loop reached `dreamer_v3_trainer.py:75` and beyond — iteration 1 completed with Loss, Reward, R_MAE, Entropy all reported. No `ValueError` traceback.

### `git diff --stat HEAD` (changed files)

```
configs/models/dreamer_v3/dreamer_v3.yaml           |  2 ++
scratch/test_dreamer_v3_config_smoke.py  |  (untracked, ?? in git status)
```
All other modified files in the diff stat are pre-existing uncommitted changes unrelated to this plan.

### Deviations from Plan

None. Implemented exactly as specified.

### Blockers / Follow-up Items

- `configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml` is missing both `use_layer_norm` and the full mixture-sampler block. A separate plan should be opened when neuromodulated runs are needed (as flagged in the plan's "Adjacent issue" section).
- The user should update `--device cuda:3` to `cuda:0` or `cuda:1` in `train_command-new.sh` (no such GPU on this box).

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-05-06

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `configs/models/dreamer_v3/dreamer_v3.yaml` | +2 lines (`use_layer_norm: true` + continuation comment) inserted at line 42, between `unimix: 0.01` (L41) and `# Encoding Configuration` (L45) | ✅ | Value `true` matches PPO YAMLs (`recurrent_ppo.yaml:20`, `neuromodulated_ppo.yaml:20`). Placement matches plan exactly. Diff is `+2/-0`. |
| `scratch/test_dreamer_v3_config_smoke.py` | new file — smoke test enumerating all 16 mandatory `agent.*` keys consumed by `DreamerTrainer.__init__` (10 architecture keys at L65–78, 3 LR keys at L96/104/112, plus `use_layer_norm`, `encoding_mode`, `sampling_mode`) | ✅ | Re-confirmed pre-fix behavior via `git stash`: test correctly FAILs with exit 1 reporting `agent.use_layer_norm` missing. Post-fix: exit 0, "OK: all 16 mandatory keys present". Test self-documents the bug it guards against (clear docstring + explicit key list). Does not swallow exceptions. |

**Out-of-scope guardrails (all clean)**: `configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml`, `train_command-new.sh`, `train.py`, and all of `src/` show `0` lines of diff vs HEAD — no scope creep.

**Root-cause vs symptom check**: Fix adds the missing YAML key (root-cause) rather than relaxing `get_mandatory` at `dreamer_v3_trainer.py:75` (symptom). Confirmed: no `src/` files modified; the strict-config call site is preserved. ✅

**Conclusion**: ✅ Verified — minimal scoped fix; regression test reproduces the bug pre-fix and passes post-fix; no out-of-scope changes; ready for the user to commit.

<!-- For ⚠️/❌ items, add detailed sections below the table with:
     root cause, affected lines, and recommended fix. -->

---

<!--
Adjacent issue (NOT fixed by this plan, see Analysis):
  configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml is missing both
  `use_layer_norm` and the entire mixture-sampler block
  (sampling_mode, mixture_positive_slots, mixture_recent_slots,
   mixture_recent_window, positive_buffer_capacity).
  Recommend opening a separate plan when neuromodulated runs are needed,
  cross-referencing this doc.
-->
