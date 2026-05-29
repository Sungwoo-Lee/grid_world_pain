---
title: "dreamer-srl WandB filter + config-save parity with rPPO/JAX-Dreamer"
topic: dreamer
status: active
created: 2026-05-29
last_updated: 2026-05-29
---

# dreamer-srl WandB filter + config-save parity with rPPO/JAX-Dreamer

> **Status**: PLANNED (frontmatter `status: active` per [FRONTMATTER_CONTRACT.md](../meta/FRONTMATTER_CONTRACT.md) enum — this label tracks plan lifecycle)
> **Opened**: 2026-05-29
> **Related**: [docs/develop/active/dreamer_srl_v2/MIGRATION_PLAN.md](MIGRATION_PLAN.md)

---

## Context

Two small parity defects in `src/algorithms/dreamer_srl/dreamer_srl_main.py` are degrading day-to-day usability of dreamer-srl runs and need a clean fix.

**Bug 1 — WandB filter is broken.** When you open the WandB UI and try to filter for "only dreamer-srl runs" with the obvious query `agent.algorithm = DreamerV3`, the filter behaves inconsistently. The rPPO trainer (`train.py`) writes a clean payload where `wandb.config.agent` equals `{"algorithm": "RecurrentPPO"}`, and the equivalent filter works. dreamer-srl writes a doubly-nested payload — the algorithm identifier ends up under both `wandb.config.agent.algorithm` AND `wandb.config.agent.agent.algorithm` — because the agent YAML's own `agent:` block is wrapped one level too deep. This was confirmed empirically against the running dreamer-srl run `w9as1qe3` (n114, cuda:0) versus the recent rPPO run `0ikqqpvc`.

**Bug 2 — config files are never saved with the results.** rPPO writes the full merged YAML to `<results_dir>/models/config.yaml` so future analysis can recover exactly which config produced any given checkpoint. dreamer-srl creates the same `results_dir` and an Orbax checkpoint manager, then silently skips the YAML dump — so a `results/JAX_DreamerSRL/<ts>_<name>/` directory carries the weights but no record of the env or agent YAML that produced them.

**What success looks like.** After the fix: (1) the WandB filter `agent.algorithm = DreamerV3` returns exactly the dreamer-srl runs (no nesting weirdness in the config payload); (2) every new run leaves both the env and agent YAML next to its checkpoints, so any future analysis can be rerun deterministically. The scope is exactly these two fixes — nothing else.

## Analysis

### Bug 1: doubly-nested `agent` block in `wandb_config`

The current dreamer-srl code at `src/algorithms/dreamer_srl/dreamer_srl_main.py:391-419` builds the WandB config payload like this:

```python
env_config_dict = env_cfg.to_dict()      # full env YAML tree
agent_config_dict = agent_cfg.to_dict()  # full agent YAML tree, INCLUDING its top-level `agent:` block
wandb_config = {
    "algorithm": "DreamerV3",
    ...
    "env":   env_config_dict,             # wraps env YAML under "env"
    "agent": agent_config_dict,           # wraps the WHOLE agent YAML under "agent"  ← root cause
}
wandb_config.setdefault("agent", {})["algorithm"] = "DreamerV3"   # line 419 "defense"
```

The agent YAML (`configs/models/dreamer_srl/01_food_only_buf256k.yaml`) starts with:

```yaml
agent:
  algorithm: "DreamerV3"
algo: { ... }
buffer: { ... }
training: { ... }
env:    { num_envs: 1 }
```

So `agent_config_dict["agent"] == {"algorithm": "DreamerV3"}`. When the code then does `wandb_config["agent"] = agent_config_dict`, the resulting structure is:

```
wandb_config["agent"] = {
    "algorithm": "DreamerV3",            # injected by line 419 defense
    "agent":     {"algorithm": "DreamerV3"},   # original YAML block — NESTED one level too deep
    "algo":      {...},
    "buffer":    {...},
    "training":  {...},
    "env":       {num_envs: 1},
}
```

Empirically verified by querying `wandb.Api().run("w9as1qe3").config["agent"]` on the live run.

The rPPO trainer at `train.py:599-617` avoids this by **spreading** the merged config dict into the payload:

```python
wandb_config_payload = {
    "algorithm": algorithm,
    "framework": "JAX/Flax NNX",
    ...
    **config.to_dict(),     # top-level YAML keys (agent, training, env, …) land at top level
}
```

Because rPPO's config has a top-level `agent:` block in YAML, `**config.to_dict()` lands that block directly at `wandb_config["agent"]` — clean, single level, filterable.

### Bug 2: configs not persisted to results_dir

`src/algorithms/dreamer_srl/dreamer_srl_main.py:454-477` creates `results_dir` and the Orbax checkpoint manager but never writes the YAML files. rPPO's pattern at `train.py:561-580`:

```python
config_save_path = os.path.join(models_dir, "config.yaml")
with open(config_save_path, 'w') as f:
    yaml.dump(config.to_dict(), f, default_flow_style=False)
```

This is missing in dreamer-srl. Reproducibility cost: any future analysis of `results/JAX_DreamerSRL/<ts>_<name>/` has to chase the matching commit + the launch command in the diary, which is brittle (configs evolve, the live YAML may have changed since).

### Key-collision survey (for the spread-style merge)

dreamer-srl has **two** configs (`--env-config` + `--agent-config`), unlike rPPO (one merged config). Before adopting a `**env_cfg.to_dict(), **agent_cfg.to_dict()` spread, the collision space needs to be known.

**Current top-level keys in `configs/experiment/hypervigilance/01-interoNocicept.yaml`** (env):
`environment, body, sensory, visualization, perceptual_noise, behavior_measures`

**Current top-level keys in `configs/models/dreamer_srl/01_food_only_buf256k.yaml`** (agent):
`agent, algo, buffer, training, env`

**Collisions in the current canonical configs**: none — the names are disjoint. (Note: env-config uses `environment:` for its world block; agent-config uses `env:` only for `num_envs`. Different keys.)

**Future-proofing**: a second config could one day introduce a collision (e.g. an env config that adds a `training:` block, or both growing a `seed:`). The merge rule below picks a deterministic winner so a future collision degrades gracefully instead of silently overwriting.

## Implementation Plan

### Design

**Bug 1 fix (parity with rPPO).** Replace the `"env": …, "agent": …` wrapping with a spread of both configs, mirroring `train.py:599-617`. Use a deterministic merge rule: **agent-config wins on collision**, because in dreamer-srl the agent config carries the algorithm-identity and training-cadence keys that the WandB UI must filter on. Remove the line-419 "defense" — once the spread lands the YAML's `agent:` block at the top level cleanly, the defense becomes redundant.

**Bug 2 fix.** After `os.makedirs(results_dir, exist_ok=True)` at line 467, write the two YAMLs to a `models/` subdirectory (mirrors rPPO's `<results_dir>/models/config.yaml` layout). Keep the two YAMLs **separate** (`env_config.yaml` + `agent_config.yaml`) rather than merging — the CLI source-of-truth distinction (`--env-config` vs `--agent-config`) is part of the contract, and keeping them separate makes it obvious which file came from which flag. The single-file merge offers no benefit here and obscures provenance.

### File Changes

#### `src/algorithms/dreamer_srl/dreamer_srl_main.py` — Bug 1 (lines 386-419)

Replace the current WandB config block with a spread-style payload that mirrors `train.py:599-617`.

```python
# BEFORE (lines 386-419):
use_wandb = not args.no_wandb
if use_wandb:
    try:
        import wandb
        # Build flat config payload mirroring train.py:L599-L617
        env_config_dict = env_cfg.to_dict()
        agent_config_dict = agent_cfg.to_dict()
        wandb_config = {
            # Explicit top-level identifiers (mirrors train.py:L600-L601)
            "algorithm": "DreamerV3",
            "framework": "JAX/Flax NNX",
            # Backward-compat scalars (previously hand-picked subset)
            "env_config": args.env_config,
            "agent_config": args.agent_config,
            "total_steps":      total_timesteps,
            "total_timesteps":  total_timesteps,
            "episodes":         episodes,
            "num_envs": num_envs,
            "seed": args.seed,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "horizon": horizon,
            "gamma": gamma,
            "lmbda": lmbda,
            "learning_starts": learning_starts,
            "seq_len": seq_len,
            "batch_size": batch_size,
            # Full config trees — agent.algorithm becomes filterable in WandB UI
            "env": env_config_dict,
            "agent": agent_config_dict,
        }
        # Defense-in-depth: force agent.algorithm even if a variant YAML omits the field
        # Ported from train.py:L600 + 189f0df defense
        wandb_config.setdefault("agent", {})["algorithm"] = "DreamerV3"
```

```python
# AFTER:
use_wandb = not args.no_wandb
if use_wandb:
    try:
        import wandb
        # Build flat config payload mirroring train.py:L599-L617.
        # Spread both YAMLs at top level so the agent YAML's `agent:` block lands
        # at wandb.config.agent (clean, single level) — makes
        # `agent.algorithm = DreamerV3` filterable in the WandB UI.
        # Merge rule: agent-config wins on key collision (algorithm-identity and
        # training-cadence live there). Current canonical configs are disjoint at
        # top level (env:  environment/body/sensory/visualization/perceptual_noise/
        # behavior_measures; agent: agent/algo/buffer/training/env), so no actual
        # overrides today — the rule only matters for future configs.
        env_config_dict = env_cfg.to_dict()
        agent_config_dict = agent_cfg.to_dict()
        wandb_config = {
            # Explicit top-level identifiers (mirrors train.py:L600-L601)
            "algorithm": "DreamerV3",
            "framework": "JAX/Flax NNX",
            # Backward-compat scalars (previously hand-picked subset)
            "env_config": args.env_config,
            "agent_config": args.agent_config,
            "total_steps":      total_timesteps,
            "total_timesteps":  total_timesteps,
            "episodes":         episodes,
            "num_envs": num_envs,
            "seed": args.seed,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "horizon": horizon,
            "gamma": gamma,
            "lmbda": lmbda,
            "learning_starts": learning_starts,
            "seq_len": seq_len,
            "batch_size": batch_size,
            # Spread env YAML first, then agent YAML — agent wins on collision.
            **env_config_dict,
            **agent_config_dict,
        }
        # Defense-in-depth: if a variant YAML ever omits agent.algorithm,
        # the WandB filter still works. (Cheap, idempotent.)
        wandb_config.setdefault("agent", {})["algorithm"] = "DreamerV3"
```

**Diff summary of the load-bearing change**:
```diff
-            "env":   env_config_dict,
-            "agent": agent_config_dict,
+            **env_config_dict,
+            **agent_config_dict,
```

Everything else above the spread is preserved exactly. The line-419 defense is kept (it's a one-liner with no downside and protects against future YAMLs that omit `agent.algorithm`).

#### `src/algorithms/dreamer_srl/dreamer_srl_main.py` — Bug 2 (after line 467)

After `_os.makedirs(results_dir, exist_ok=True)`, create `<results_dir>/models/` and write both YAMLs there. Mirrors `train.py:561-566`.

```python
# BEFORE (lines 459-468):
from datetime import datetime as _dt
_timestamp = _dt.now().strftime("%Y%m%d-%H%M%S")
if args.results_dir:
    results_dir = args.results_dir
elif use_wandb and wandb.run is not None:
    results_dir = _os.path.join(_project_root, 'results', 'JAX_DreamerSRL', f"{_timestamp}_{wandb.run.name}")
else:
    results_dir = _os.path.join(_project_root, 'tmp', f'JAX_DreamerSRL_{_timestamp}')
_os.makedirs(results_dir, exist_ok=True)
print(f"[dreamer-srl] results_dir={results_dir}")
```

```python
# AFTER:
from datetime import datetime as _dt
_timestamp = _dt.now().strftime("%Y%m%d-%H%M%S")
if args.results_dir:
    results_dir = args.results_dir
elif use_wandb and wandb.run is not None:
    results_dir = _os.path.join(_project_root, 'results', 'JAX_DreamerSRL', f"{_timestamp}_{wandb.run.name}")
else:
    results_dir = _os.path.join(_project_root, 'tmp', f'JAX_DreamerSRL_{_timestamp}')
_os.makedirs(results_dir, exist_ok=True)
print(f"[dreamer-srl] results_dir={results_dir}")

# Persist both source YAMLs next to checkpoints for reproducibility.
# Mirrors train.py:L561-L566 — but kept SEPARATE (env_config.yaml + agent_config.yaml)
# instead of merged, to preserve the --env-config / --agent-config CLI provenance.
import yaml as _yaml
_models_dir = _os.path.join(results_dir, 'models')
_os.makedirs(_models_dir, exist_ok=True)
_env_save = _os.path.join(_models_dir, 'env_config.yaml')
_agent_save = _os.path.join(_models_dir, 'agent_config.yaml')
with open(_env_save, 'w') as _f:
    _yaml.dump(env_cfg.to_dict(), _f, default_flow_style=False, sort_keys=False)
with open(_agent_save, 'w') as _f:
    _yaml.dump(agent_cfg.to_dict(), _f, default_flow_style=False, sort_keys=False)
print(f"[dreamer-srl] saved env_config → {_env_save}")
print(f"[dreamer-srl] saved agent_config → {_agent_save}")
```

**Notes for the developer**:
- The `models/` subdir mirrors rPPO's layout — Orbax's existing `<results_dir>/checkpoints/` subdir at line 477 is untouched.
- `sort_keys=False` preserves the source YAML's key order in the dump (rPPO's `default_flow_style=False` is the only knob it uses; the order-preservation here is a small extra so the saved YAML reads like the original).
- `import yaml` is local to this block to avoid adding a module-level import unrelated to the rest of the file.

### Merge-rule decision (Bug 1)

**Rule**: `agent_config_dict` is spread **second**, so it wins on any top-level key collision with `env_config_dict`.

**Rationale**:
- The agent YAML owns algorithm-identity (`agent.algorithm`) and training-cadence (`training.*`) — the keys that gate WandB filtering and run inspection. If a future env YAML ever declares an overlapping `training:` block, the agent's value is the one the run actually uses, so it should win in the WandB record too.
- Current canonical configs are disjoint at top level (see Analysis above), so this rule has no effect today — it is forward-compatible protection.

**Single-line diff preview** of the spread block:
```diff
-            "env":   env_config_dict,
-            "agent": agent_config_dict,
+            **env_config_dict,        # env YAML keys (environment, body, sensory, …) at top level
+            **agent_config_dict,      # agent YAML keys (agent, algo, buffer, training, env) — wins on collision
```

### Save-config decision (Bug 2)

**Rule**: write **two separate YAMLs** (`env_config.yaml` + `agent_config.yaml`) under `<results_dir>/models/`, not one merged file.

**Rationale**:
- dreamer-srl's CLI surface is `--env-config` + `--agent-config` — two flags, two source files. Preserving that 1:1 mapping on disk makes the provenance obvious: anyone reading the saved YAMLs sees exactly which flag value produced which file.
- A merged file would force a merge convention (env-first? agent-first? alphabetised? collision-warning?) that has to be documented and remembered. Two files have no such convention to remember.
- rPPO writes one file because rPPO has one config. The right parity move is "write each config the trainer was given," not "produce one file regardless of how many configs went in."

## Checkpoints

What the implementing agent should verify **during** implementation:

- [ ] Checkpoint 1 — after the Bug 1 edit, run `python -c "import ast; ast.parse(open('src/algorithms/dreamer_srl/dreamer_srl_main.py').read())"` to catch syntax errors before launching anything.
- [ ] Checkpoint 2 — launch a 1-iteration smoke (`--total-steps 200 --num-envs 1 --seed 0 --wandb-name smoke_bug1_2`) and confirm both stdout lines appear:
  - `[dreamer-srl] saved env_config → …/models/env_config.yaml`
  - `[dreamer-srl] saved agent_config → …/models/agent_config.yaml`
- [ ] Checkpoint 3 — while the smoke is still running (or after), pull the WandB config via `wandb.Api().run("<smoke_run_id>").config` and assert:
  - `config["agent"]["algorithm"] == "DreamerV3"`
  - `"agent" not in config["agent"]` (no inner-`agent` nesting)
  - `"algo" in config and "buffer" in config and "training" in config` (top-level spread landed)

## Verification Steps

For the senior-developer verification pass after `developer` finishes:

1. **Re-launch a smoke run** (~200 env-steps is enough): `python src/algorithms/dreamer_srl/dreamer_srl_main.py --env-config configs/experiment/hypervigilance/01-interoNocicept.yaml --agent-config configs/models/dreamer_srl/01_food_only_buf256k.yaml --total-steps 200 --num-envs 1 --seed 0 --wandb-project grid_world_pain --wandb-name parity_fix_smoke_$(date +%s)`.

2. **Verify clean WandB config** (Bug 1):
   ```python
   import wandb
   run = wandb.Api().run("sungwoolee/grid_world_pain/<smoke_run_id>")
   assert run.config["agent"] == {"algorithm": "DreamerV3"}, run.config["agent"]
   assert "algo" in run.config and "training" in run.config, "spread didn't land top-level"
   assert "agent" not in run.config["agent"], "inner-agent nesting still present"
   print("OK: wandb.config.agent is clean")
   ```

3. **Verify saved YAMLs** (Bug 2):
   ```bash
   ls -la results/JAX_DreamerSRL/$(ls -t results/JAX_DreamerSRL | head -1)/models/
   # expect: env_config.yaml, agent_config.yaml, plus the existing Orbax checkpoint subdirs
   ```

4. **WandB UI filter test** (user-side):
   - Open `https://wandb.ai/sungwoolee/grid_world_pain/`.
   - Add filter: `Config → agent.algorithm = DreamerV3`.
   - Expected: returns the new smoke run plus all post-fix dreamer-srl runs; does NOT return rPPO runs (whose `agent.algorithm = RecurrentPPO`).
   - Pre-fix dreamer-srl runs (e.g. `w9as1qe3`) may or may not appear depending on whether WandB resolved them via the line-419 defense — note the boundary in the verification report but don't try to back-fill them.

## Out of Scope

- No refactor of anything else in `dreamer_srl_main.py` (the existing scalar payload keys above the spread are preserved verbatim).
- No new CLI flags (`--save-config`, `--no-save-config`, etc.).
- No changes to `log_interval`, `checkpoint_frequency`, or any other cadence keys.
- No changes to checkpoint logic, Orbax wiring, or the `results_dir` naming convention.
- No back-fill of saved configs into pre-fix runs.
- No changes to `train.py` (rPPO) — it is the reference; only dreamer-srl moves toward parity.

## Implementation Report

> **Implemented by**: TBD
> **Date**: TBD

## Verification Report

> **Verified by**: TBD
> **Date**: TBD

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` (lines 386-419) | Bug 1: spread-style WandB config | | |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` (after line 467) | Bug 2: save env+agent YAMLs to `models/` | | |

**Conclusion**: TBD

---

## Implementation Findings (2026-05-29)

Implementation completed by the `developer` agent (stream timed out mid-task; verification + commit handled by top-level Claude).

### Edits landed
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` lines 386-419 (Bug 1): `"env": env_config_dict, "agent": agent_config_dict` → `**env_config_dict, **agent_config_dict`. Line-419 defense kept.
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` lines 469-481 (Bug 2): added `env_config.yaml` + `agent_config.yaml` write inside a `<results_dir>/models/` directory, immediately after `os.makedirs(results_dir)`. Uses `yaml.dump(..., default_flow_style=False, sort_keys=False)` to preserve original key order for diff-friendliness.

### Smoke run evidence
- WandB run: [`mo1pkvm5`](https://wandb.ai/sungwoolee/grid_world_pain/runs/mo1pkvm5) — `parity_fix_smoke_1780041740`, 200 env-steps, num_envs=1, seed=0, 26.3s wall-clock, exited cleanly.
- **Bug 1 — wandb.config.agent is clean**:
  ```
  config["agent"] == {"algorithm": "DreamerV3"}    # ✓ no inner-agent nesting
  ```
  Top-level keys of `wandb.config` now include `agent, algo, buffer, training, environment, body, sensory, perceptual_noise, behavior_measures, visualization, …` — the spread landed correctly for both env and agent YAMLs.
- **Bug 2 — config files saved**:
  ```
  results/JAX_DreamerSRL/20260529-170251_parity_fix_smoke_1780041740/models/
    agent_config.yaml   (1623 B, 5 top-level keys)
    env_config.yaml    (15001 B, 11 top-level keys)
  ```
  `agent_config.yaml` opens with `agent: { algorithm: DreamerV3}` — round-trips the source YAML cleanly.

### Boundary
Pre-fix runs (the four `_log2k` cells launched earlier this session: `w9as1qe3`, `x0ahp61j`, `jx3obafg`, `h7aa37na`, and the four crashed `_logfix` cells from 2026-05-28 03:30) still carry the legacy doubly-nested `wandb.config.agent`. They are NOT back-filled. The WandB UI filter `agent.algorithm = DreamerV3` will now resolve for the smoke run and all future launches.
