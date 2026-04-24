# Evaluation Recording Stats — Review

This document describes how the **recording stats** mechanism works during JAX evaluation (`evaluation.py` → `evaluate_jax_checkpoint` in `src/utils/evaluation_core.py`).

---

## 1. Enable and output location

- **Config key:** `testing.record_stats`  
  Read via `config.get('testing.record_stats', False)` in `evaluation_core.py` (default `False` in code; evaluation defaults set it to `true` in `configs/evaluation/default.yaml`).
- **Output directory:**  
  `results_dir/stats/<checkpoint_iteration>/`  
  Example: `results/JAX_RecurrentPPO/my_run/stats/9000030/`
- If `record_stats` is true, that directory is created with `os.makedirs(stats_dir, exist_ok=True)`.

---

## 2. CSV structure (headers)

Headers are built once when `record_stats` is true (lines 66–111 in `evaluation_core.py`):

- **Fixed columns:**  
  `step`, `pos_r`, `pos_c`, `action`, `reward`, `satiation`, `nutrition`, `injury`, `rest_streak`,  
  `event_ate`, `event_collided`, `event_rested`,  
  `damage_total`, `damage_hiding_predator`, `damage_predator`, `damage_obstacle`
- **Observation columns:**  
  From `get_observation_breakdown(params)` and `get_visual_offsets(params)`: e.g. `obs_olf_*`, `obs_noc`, `obs_coll_*`, `obs_loc_r`, `obs_loc_c`
- **World state (per step):**  
  `res_i_r`, `res_i_c`, `res_i_active` (for each resource);  
  `pred_i_r`, `pred_i_c` (for each predator);  
  `neutral_i_r`, `neutral_i_c` (for each neutral);  
  `obs_entity_i_r`, `obs_entity_i_c` (for each obstacle).
- **End of row:**  
  `termination_reason`, `max_satiation`, `max_injury`

So each row is one timestep: state, action, reward, events, damage, full observation vector, and world entity positions.

---

## 3. Per-episode in-memory collection (deferred, no sync in loop)

During the episode loop, **no** `jax.device_get` is done for stats; only Python lists are appended (lines 142–146, 302–318):

- **`ep_jax_states`:** Dicts of JAX arrays: `agent_pos`, `satiation`, `nutrition`, `injury_level`, `rest_streak`, `res_pos`, `res_active`, `pred_pos`, `neutral_pos`, `obs_pos`
- **`ep_jax_infos`:** The `info` dict from `jax_step` (e.g. `ate_food`, `event_collided`, `rested`, `damage*`, `termination_reason`)
- **`ep_actions`:** Action index per step (with `-1` for “step 0 / no action”)
- **`ep_rewards`:** Reward per step
- **`ep_obs`:** Observation vector per step

Step 0 is recorded once at reset (lines 149–166); then after each `jax_step` the new state, info, action, reward, and next observation are appended. So “recording” during evaluation is just appending to these lists.

---

## 4. Batch transfer and CSV write (once per episode)

After the episode loop, **if** `record_stats` and there is data (lines 325–412):

1. **Single GPU→CPU sync:**  
   All `ep_jax_states` entries are stacked into batched arrays and transferred with one `jax.device_get` (batched state dict). Same for `ep_jax_infos` (selected keys) and `ep_obs` (stacked then `device_get`). So there is exactly one sync per episode for stats, not per step.

2. **File:**  
   One CSV per episode:  
   `stats_dir / f"{ep+1:06d}ep_stats.csv"`  
   e.g. `000001ep_stats.csv`, `000002ep_stats.csv`.

3. **Row construction:**  
   For each step index `t`, a row is built from:
   - `batched_state` (agent pos, satiation, nutrition, injury, rest_streak, res/pred/neutral/obs positions and active flags),
   - `ep_actions[t]`, `ep_rewards[t]`,
   - `batched_info` (events and damage),
   - `batched_obs[t]` (observation vector, mapped to the precomputed `obs_*` header indices),
   - then `termination_reason`, `max_satiation`, `max_injury`.

So the “recording stats” mechanism = **collect in Python lists during the loop, then one batched transfer + one CSV write per episode**.

---

## 5. Flow summary

| Stage | What happens |
|--------|----------------|
| **Config** | `testing.record_stats` (e.g. from `configs/evaluation/default.yaml`) turns recording on. |
| **Setup** | `stats_dir = results_dir/stats/<checkpoint_pct>`, headers built from env/sensor breakdown. |
| **Per step** | Append to `ep_jax_states`, `ep_jax_infos`, `ep_actions`, `ep_rewards`, `ep_obs` (no device transfer). |
| **After episode** | One batched `jax.device_get` for states, infos, obs → NumPy; write one CSV (`{ep:06d}ep_stats.csv`) with all columns. |

---

## 6. Related code

- **Entry:** `evaluation.py` → `evaluate_jax_checkpoint(...)` in `src/utils/evaluation_core.py`
- **Config:** `configs/evaluation/default.yaml` → `testing.record_stats: true`
- **Usage:** `plot_physiology.py` expects enriched stats from runs with `record_stats: true`

---

## 7. Parallel environment number (e.g. 4 envs) — implemented

**Behavior:**

- **Config / CLI:** `testing.num_envs` in config (e.g. `configs/evaluation/default.yaml`) and `--num_envs` on the command line. Default is 1.
- **Effective parallelism:** The number of envs that actually run is `effective_num_envs = min(num_episodes, num_envs)` (episode-ticket design; see §9).
- **When `num_envs == 1` (or `effective_num_envs == 1`):** Same as before: one env at a time, sequential episodes, same CSV and video behavior.
- **When `num_envs > 1` and `num_episodes > 1`:** Evaluation uses `ParallelEnv` from `src/environment/wrapper.py`; `effective_num_envs` envs step in parallel. Per-slot stats buffers are kept; when an env’s episode finishes, its CSV is written (via `_write_episode_stats`) and, if tickets remain, that env is reset and runs again. Same CSV format and one file per completed episode (`000001ep_stats.csv`, …). Video is only produced in the single-env path (parallel path does not fill `all_frames`).

---

## 8. Motivation: selection bias when using parallel envs

When we add parallel envs, a naive approach is: run N envs, and **stop as soon as** we have collected `num_episodes` completed episodes. That creates **selection bias**:

- The **first** episodes to finish are often the **shortest** (e.g. agent died early, hit a terminal condition quickly). The envs that are still running might be the ones where the agent is surviving longer.
- If we stop the whole evaluation as soon as we have 3 episodes (e.g. 3 episodes, 5 envs), we record only those 3 “first to finish” trajectories. Our stats and CSVs then **over-represent short / “died early”** runs and under-represent longer, better survival. Mean reward and physiology plots would be biased.

So we need a design that avoids “whoever finishes first” determining which episodes get recorded.

---

## 9. Episode-ticket design (for parallel eval)

To avoid that bias and to keep semantics clear, we use an **episode-ticket** model:

- There are exactly **`num_episodes` tickets**. Each ticket corresponds to one episode we will record (one CSV, one entry in episode_rewards/episode_lengths).
- **Only envs that hold a ticket run.** We never run more envs than we have tickets to assign. So the number of envs that actually execute is at most `min(num_episodes, num_envs)`.
- **When an env finishes an episode:** It “uses” one ticket (we write that trajectory as the next episode number and increment the completed count). If there are still **unused** tickets, we give that env a **new** ticket: reset it and let it run another episode. If no tickets remain, we do **not** start a new episode in that env.
- **When all tickets are used:** We stop giving new tickets. Any env that is still running is finishing an episode that already had a ticket; we wait for those to finish and write them. When all tickets are used and all in-flight episodes are written, evaluation ends.

**Concrete cases:**

| Episodes | Envs | Tickets | What happens |
|----------|------|--------|----------------|
| 3 | 5 | 3 | Only **3 envs** run; the other **2 never execute** (no ticket). When one of the 3 finishes, we write that episode; if tickets remain we give that env a new ticket (reset). When 3 episodes are written, we’re done. No bias: all 3 recorded episodes are from envs that ran from the start. |
| 10 | 5 | 10 | **5 envs** run in parallel. When an env finishes, we write an episode (use one ticket) and, if tickets remain, give that env a new ticket (reset). When 10 episodes are written (all tickets used), we stop giving new tickets and wait for any still-running env to finish; then we’re done. |

**Implementation takeaway:** Use **effective parallelism** `effective_num_envs = min(num_episodes, num_envs)`. Only that many envs are ever stepped. When an env finishes, we write its episode and, if `completed_episodes < num_episodes`, we give it a new ticket (reset and continue). When all tickets are used, we no longer refill; we only wait for in-flight episodes to finish. This gives unbiased stats (no “first to finish” bias) and no wasted work (we don’t start envs that would never get a ticket).

**Code:** `evaluate_jax_checkpoint(..., num_envs=1)` in `src/utils/evaluation_core.py`; single-env path `_run_single_env_eval`, parallel path `_run_parallel_env_eval`. Episode stats written via `_write_episode_stats`.

---

## 10. Conversation / implementation summary (for LLM context)

This section summarizes the design and implementation decisions from the conversation that added parallel-env evaluation and the episode-ticket mechanism. Use it as context when editing evaluation or stats code.

### What was implemented

- **Parallel-env evaluation:** When `num_envs > 1` and `num_episodes > 1`, evaluation runs `effective_num_envs = min(num_episodes, num_envs)` envs in parallel via `ParallelEnv` (`src/environment/wrapper.py`). Only that many envs are ever reset or stepped (episode-ticket design).
- **Episode-ticket logic:** Exactly `num_episodes` “tickets”; each finished episode uses one ticket and gets one CSV. When an env finishes, we only give it a new ticket (reset) if `completed_episodes < num_episodes`. When all tickets are used we stop refilling and exit the step loop after writing the last episode(s).
- **Stats:** One CSV per completed episode, same format as single-env. Written via `_write_episode_stats(stats_dir, episode_number, ...)` so both single-env and parallel paths share the same write logic.
- **Single-env path unchanged:** When `effective_num_envs == 1`, the original sequential loop runs (`_run_single_env_eval`); it now calls `_write_episode_stats` instead of inline CSV code.

### What was done (concrete changes)

**`src/utils/evaluation_core.py`**

- **`generic_inference`:** Support batched input: `action = jnp.argmax(logits, axis=-1)`; when `logits.ndim > 1` set `log_prob = jnp.zeros(logits.shape[0])` so batch inference works.
- **`_write_episode_stats`:** New helper that takes one episode’s lists (states, infos, actions, rewards, obs), does one batched `jax.device_get`, and writes one CSV with the same row layout as before. Used by both single-env and parallel paths.
- **`evaluate_jax_checkpoint`:** Added parameter `num_envs=1`. Compute `effective_num_envs = min(num_episodes, num_envs)`. If `effective_num_envs == 1` call `_run_single_env_eval`, else call `_run_parallel_env_eval`. Moved video save and return (mean_reward, mean_length, etc.) to after the branch so both paths share it. Guard `mean_reward` / `mean_length` with `if episode_rewards else 0.0` to avoid NaNs when no episodes complete.
- **Stats setup:** Build `stat_headers` and `action_map` so they exist even when `record_stats` is False (minimal `action_map` always; `stat_headers` only when `record_stats`), so parallel path can be called safely.
- **`_run_single_env_eval`:** New function containing the original per-episode loop. Uses `params_ref` for env and stats. Replaced the inline CSV block (batch device_get + row writing) with a call to `_write_episode_stats(..., ep+1, ...)`. Added `render_jax_state` import when `render_video`. **Fix:** `action_idx = int(jnp.squeeze(action))` so single-env (batch size 1) action is a Python int.
- **`_run_parallel_env_eval`:** New function. Uses `ParallelEnv`; resets `effective_num_envs` envs; keeps per-slot buffers (slot_states, slot_infos, slot_actions, slot_rewards, slot_obs). Step loop: batched inference, `penv.step(states, actions)`, append to each slot, then `states = next_states`, `obs = next_obs`. For each `i` with `dones[i]`: write episode via `_write_episode_stats`, append to episode_rewards/episode_lengths; if `completed_episodes >= num_episodes` break (no refill); else reset slot `i` (new_state, obs[i], h_state[i], clear slot buffers). **Fix:** `obs = obs.at[i].set(new_obs)` instead of nonexistent `jax.lax.dynamic_update_index`. Index vmapped infos with `v[i]` when `hasattr(v, 'ndim') and v.ndim > 0`. **Final message:** If `completed_episodes >= num_episodes` print “all tickets used”; else “X/Y episodes (safety cap or early exit).”
- **Ticket debug prints:** In `_run_parallel_env_eval`, when not quiet print start (“X envs, Y tickets”) and end (“Done: Z episodes…”). When `debug` print per-env finish (env index, episode K/Y, reward, steps, tickets_left), “Giving new ticket to slot i”, “All tickets used; not refilling slot i”, and “Exiting step loop”.

**`evaluation.py`**

- Added `--num_envs` (int, optional). Resolve `num_envs = args.num_envs if args.num_envs is not None else config.get('testing.num_envs', 1)`. Print “Num envs: X (effective: Y)” in the summary. Pass `num_envs=num_envs` into `evaluate_jax_checkpoint`.
- Added `--debug` (store_true). Pass `debug=args.debug` into `evaluate_jax_checkpoint`.

**`configs/evaluation/default.yaml`**

- Added `num_envs: 1` under `testing`.

**`docs/EVALUATION_RECORDING_STATS.md`**

- §7: Updated to state parallel env is implemented; described `testing.num_envs`, `--num_envs`, effective_num_envs, and that video is only in single-env path.
- §9: Added one sentence pointing to the code (evaluate_jax_checkpoint, _run_single_env_eval, _run_parallel_env_eval, _write_episode_stats).
- §10: Added this conversation/implementation summary (design, what was done, key locations, config/CLI, bugs fixed, debug messages, verification notes).

### Key code locations

| Item | Location |
|------|----------|
| Entry, effective_num_envs, branch to single vs parallel | `evaluation_core.py`: `evaluate_jax_checkpoint`, ~lines 123–193 |
| Single-env loop | `_run_single_env_eval` |
| Parallel loop, ticket logic, refill only when tickets left | `_run_parallel_env_eval` (~452–578); refill guarded by `if completed_episodes >= num_episodes: break` before reset |
| One-episode CSV write | `_write_episode_stats` (~42–119) |
| Batched inference (batch size 1 or N) | `generic_inference`: `argmax(logits, axis=-1)`; single-env call uses `jnp.squeeze(action)` for scalar |

### Config and CLI

- **Config:** `configs/evaluation/default.yaml` → `testing.num_envs: 1`, `testing.record_stats: true`.
- **CLI:** `evaluation.py` → `--num_envs`, `--debug`. `--debug` enables episode-ticket debug prints in the parallel path.

### Bugs fixed during implementation

1. **Single-env action shape:** With `obs_batch` shape `(1, obs_dim)`, `generic_inference` returns `action` of shape `(1,)`. Using `int(action)` raised (only scalar arrays convertible to Python scalars). **Fix:** `action_idx = int(jnp.squeeze(action))` in `_run_single_env_eval`.
2. **Parallel obs update:** Code used `jax.lax.dynamic_update_index` (does not exist in JAX). **Fix:** `obs = obs.at[i].set(new_obs)` when resetting slot `i`.
3. **Final message:** When the step loop exits due to safety cap before completing all episodes, the message must not say “all tickets used”. **Fix:** If `completed_episodes >= num_episodes` print “all tickets used”; else print “X/Y episodes (safety cap or early exit).”
4. **Ticket over-issuing (Refined):** The initial parallel loop used `completed_episodes` to guard refills. This over-issued tickets because `completed_episodes` only increments *after* an episode finishes, while multiple envs might be running. **Fix:** Introduced `issued_tickets` counter. Refills only happen if `issued_tickets < num_episodes`.
5. **Multiple `dones` data loss:** The `dones` loop contained a `break` that could trigger before all finished episodes in a single step were recorded. **Fix:** Removed premature `break` and replaced with `continue` for slot deactivation.
6. **Ghost episodes/Double counting:** Finished environments that were not reset continued to report `done=True`, leading to "ghost" completions. **Fix:** Introduced `slot_active` mask to track which slots hold a valid ticket and ignore `dones` from inactive slots.

### Verification (Episode-Ticket Design)

The fix was verified using a simulation script that modeled parallel environments finishing at different rates and simultaneously.

**Reproduction results:**
- **Goal**: Exactly 10 episodes from 5 environments.
- **Outcome**: Exactly 10 tickets issued, 10 episodes recorded. No over-issuing, no double-counting.
- **Exhaustion**: Environments correctly deactivate once `issued_tickets == num_episodes`, and the loop waits for the remaining active environments to finish.
