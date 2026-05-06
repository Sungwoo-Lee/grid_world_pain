---
title: "Evaluation Video Comparison: Current Branch vs `feature/tuningEnv`"
topic: refactors
status: archive
created: 2026-03-04
last_updated: 2026-04-12
---

# Evaluation Video Comparison: Current Branch vs `feature/tuningEnv`

**Date:** 2026-02-21  
**Current branch:** `feature/fixSensorFlag`  
**Reference branch:** `feature/tuningEnv`  
**Goal:** Recover the same evaluation video rendering as `feature/tuningEnv` (arrangement, colors, behavior) and fix the issue where the agent appears not to move in generated videos.

---

## 1. Summary of Differences

| Area | tuningEnv | Current (fixSensorFlag) | Impact on Video |
|------|-----------|--------------------------|-----------------|
| **evaluation.py** | Merges visualization config; device handling with explicit print | No vis config merge; device handling without print | Icons/layout may differ if saved config lacks vis |
| **evaluation_core.py** | RecurrentPPO-only inference (`get_action_and_value_nnx`); no `info` to renderer | Generic inference for RPPO + DreamerV3; passes `info` to renderer | DreamerV3 eval supported; damage pod can be shown |
| **renderer.py** | No `info`; no bush/agent_bush; no damage segment; no obstacle-hides-agent logic | `info` param; bush/agent_bush; damage pod; obstacle-hides-agent | Current has more UI (damage, bush); layout same |
| **State in render** | Same step loop | Same step loop; `state` may stay as JAX DeviceArray | **Possible cause of “agent not moving”** if not materialized |

---

## 2. evaluation.py

### 2.1 Device selection

- **tuningEnv:** Handles `--device` as digit (e.g. `0`) or `gpu:0`; prints device to stdout.
- **Current:** Handles `cpu`, `gpu`, `cuda:0`; no device print. Logic is unified with `startswith("cpu")` and `":" in device_str`.

### 2.2 Config merge

- **tuningEnv:** After merging evaluation defaults, **also merges** `configs/visualization/visualization.yaml` (“Required for obstacle icons”).
- **Current:** Only merges `configs/evaluation/default.yaml`. **Does not** merge visualization config.

**Recommendation:** Merge visualization config in evaluation (like tuningEnv) so icon names (e.g. bush, agent_bush) and layout options are present when running `evaluation.py` with a saved run that might not have full vis in the saved config.

### 2.3 Observation spec logging

- **tuningEnv:** Prints detailed observation spec (breakdown, action dim, hidden size, learning rate) before evaluation.
- **Current:** Removed (cleaner stdout). Can be re-added behind a `--verbose` flag if needed.

---

## 3. evaluation_core.py

### 3.1 Model inference

- **tuningEnv:** Uses `get_action_and_value_nnx(model, obs_batch, h_state, eval_mode=True)` (RecurrentPPO only). If `h_state is None`, calls with `None` for hidden state.
- **Current:** Uses `generic_inference(model, obs_batch, h_state, eval_mode=True)` so the same path supports both RecurrentPPO and DreamerV3. Always passes `h_state` (can be `None`).

Behavior for RecurrentPPO should be equivalent. For DreamerV3, current branch supports evaluation; tuningEnv did not.

### 3.2 Reset and initial state

- **tuningEnv:** `state = jax_reset(params, reset_key)` (comment says “CRITICAL FIX” in current). `initial_state(batch_size=1)`.
- **Current:** Same reset. `initial_state(batch_size=None)` for compatibility with both RPPO and DreamerV3.

### 3.3 Renderer call

- **tuningEnv:** `render_jax_state(..., sensory_data=..., icon_config=icon_config)` — **no `info`**.
- **Current:** `render_jax_state(..., sensory_data=..., info=jax.device_get(info), icon_config=icon_config)`.

So current can show damage breakdown in the right panel; tuningEnv did not.

### 3.4 Stats and sensory viz

- **tuningEnv:** Stat headers include explicit obs labels (e.g. visual channels from `obstacle_names`); sensory viz uses `'type': 'spectrum'` for Olfaction without `labels`; Visual uses `num_features` from `len(obstacle_names)` and dynamic labels.
- **Current:** Stat headers use generic names (e.g. `res_{i}_x`); Olfaction has `labels: ['GRS', 'SND', ...]`; Visual uses fixed `num_features: 8` and fixed `labels: ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']`.

For video, the main difference is whether the right panel shows dynamic vs fixed channel names; layout is the same.

---

## 4. renderer.py

### 4.1 Signature

- **tuningEnv:** `render_jax_state(state, params, episode=None, step=None, train_episode=None, dpi=100, icon_scale=1.0, action=None, sensory_data=None, icon_config=None)`.
- **Current:** Same, plus **`info=None`** (used for damage pod).

### 4.2 Icons and colors

- **tuningEnv:** Icons: agent, food, danger, predator, agent_food, agent_danger, agent_predator, rock, neutral. **No bush, agent_bush.** Colors: no `dmg_danger`, `dmg_predator`, `dmg_obstacle`.
- **Current:** Adds **bush**, **agent_bush**; adds **dmg_danger**, **dmg_predator**, **dmg_obstacle** for the damage segment.

### 4.3 Obstacle and agent drawing

- **tuningEnv:** For each obstacle cell: if inside view and **not** (agent cell), draw obstacle icon. Then choose agent icon (predator/danger/food/neutral/default).
- **Current:** If obstacle cell **is** agent cell and obstacle `obs_hides_agent`, add `'bush'` to `at_agent` and use **agent_bush** icon. Otherwise same “inside view then draw” logic. So current supports “agent hidden in bush”; tuningEnv did not.

### 4.4 Right panel (telemetry)

- **tuningEnv:** No damage segment. Rest of layout (vitals, sensors, action) same.
- **Current:** If `info is not None` and `'damage' in info`, draws **“Acute Damage”** pod with segmented bar (DNG / PRD / OBS) and total. Same colors and layout otherwise.

---

## 5. “Agent Not Moving” in Video

Reported behavior: in the generated evaluation video, the agent appears not to move.

Possible causes:

1. **State not materialized (JAX DeviceArray)**  
   The step loop does `state = next_state` and passes `state` to `render_jax_state`. If `state` is never transferred to host, `np.array(state.agent_pos)` might not see updated values in time (e.g. async execution).  
   **Fix:** Before calling `render_jax_state` when `render_video` is True, materialize the state used for rendering, e.g.  
   `state_for_render = jax.device_get(state)` and pass `state_for_render` (or at least ensure `state.agent_pos` is fetched with `jax.device_get(state.agent_pos)` and the renderer uses that).

2. **Model always outputting same action**  
   If the policy always outputs the same action (e.g. Rest), the agent would not move. This can be checked by logging `action_idx` per step or inspecting stats CSVs. Not a rendering bug.

3. **Wrong state passed to render**  
   Current code passes the updated `state` (after `state = next_state`) to `render_jax_state`, so the drawn position should be correct. No change needed unless a copy-paste error is found elsewhere.

**Recommendation:** In `evaluation_core.py`, when appending a frame for video, use a materialized state for rendering, e.g.:

```python
if render_video:
    state_for_render = jax.device_get(state)  # ensure host-side values for matplotlib
    all_frames.append(render_jax_state(
        state_for_render, params, ...
    ))
```

---

## 6. Recovering tuningEnv-Style Rendering

To make current evaluation videos match tuningEnv’s look and behavior:

1. **Merge visualization config in evaluation**  
   In `evaluation.py`, after merging evaluation defaults, merge `configs/visualization/visualization.yaml` (same as tuningEnv) so icon keys and layout options are defined.

2. **Optional: match tuningEnv renderer exactly**  
   - Remove or gate the damage pod (e.g. only draw if `info is not None` and a config flag is set; or remove for “tuningEnv mode”).  
   - Remove or gate bush/agent_bush and obstacle-hides-agent logic if you want strict tuningEnv parity.  
   (Current additions are backward-compatible; tuningEnv simply didn’t have them.)

3. **Icons**  
   tuningEnv’s visualization config does not list `bush` or `agent_bush`; current config does. For tuningEnv-identical videos, use a vis config without bush/agent_bush and ensure the renderer doesn’t require them when those features are disabled.

4. **Fix “agent not moving”**  
   Apply the materialization fix above (use `jax.device_get(state)` for the state passed to `render_jax_state` when generating video frames).

---

## 7. File-Level Diff Summary

| File | tuningEnv → Current (summary) |
|------|-------------------------------|
| **evaluation.py** | No vis config merge; device handling simplified; no obs spec print; DreamerV3 uses `sequence_length`. |
| **evaluation_core.py** | Generic inference; `info` passed to renderer; stats/viz headers and labels changed; quiet logic for tqdm; `jax_reset(params, key)`; `initial_state(batch_size=None)`. |
| **renderer.py** | Added `info`, bush/agent_bush, damage colors, damage pod, obstacle-hides-agent logic. |
| **configs/visualization/visualization.yaml** | Current adds `bush`, `agent_bush` in icons. |

---

## 8. Next Steps

1. ~~**Implement state materialization** in `evaluation_core.py`~~ **Done:** When rendering video, `state` is now materialized with `jax.device_get(state)` before passing to `render_jax_state` (initial frame and every step). Fix for “agent not moving” when it is due to JAX async/device.
2. ~~**Re-add visualization config merge** in `evaluation.py`~~ **Done:** Evaluation now merges `configs/visualization/visualization.yaml` after evaluation defaults, so icon and layout config match training/tuningEnv.
3. **Optionally** add a “tuningEnv-style” mode (e.g. env or config flag) that disables damage pod and bush/agent_bush so videos match tuningEnv exactly.
4. **Verify** with a short evaluation run and visual inspection of the saved video that the agent moves and that layout/colors match the chosen mode.
5. **DreamerV3 evaluation:** Restore and state-merge are fixed. `from_twohot` in the agent forward now uses `value_logits.shape[-1]` so both 128- and 255-bucket checkpoints work. Recurrent PPO and DreamerV3 evaluation both run successfully.
