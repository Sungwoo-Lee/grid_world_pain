---
title: "Sheeprl Bridge — Parallel-Env Readiness Audit + SPS Benchmark Sweep"
topic: dreamer
status: active
created: 2026-05-12
last_updated: 2026-05-12
---

# Sheeprl Bridge — Parallel-Env Readiness Audit + SPS Benchmark Sweep

> **Status**: PLANNED — 2026-05-12
> **Opened**: 2026-05-12
> **Related**: [Metric Parity Plan v2 (G6 passed)](METRIC_PARITY_PLAN.md) · [Sheeprl Bridge Implementation Plan](IMPLEMENTATION_PLAN.md) · [Sheeprl how-to](../diagnosis/sheeprl_training_howto.md)

---

## Purpose (plain language)

The sheeprl PyTorch DreamerV3 bridge is now metric-complete (the 20-key v2 extension landed today and the G6 smoke `dgs6ehtt` confirmed all 51 episode-level keys are emitted). Every smoke and parity run so far has used **one parallel environment instance** (`num_envs=1` for G3/G6 metric verifications, `num_envs=4` for the original 200k smoke `jzgkcep4` before the metric work changed the wrapper's internal state). The user now wants two answers in one experiment:

1. **Does the current wrapper work correctly when sheeprl spawns N parallel copies of it?** The wrapper gained a lot of new per-instance state during metric-parity work (episode accumulators, behavior-measure accumulators, distance aggregators, JAX env state, JAX PRNG keys). Each of these must stay isolated across instances; cross-instance state-bleed would silently corrupt metrics or training.

2. **How fast does training go when we crank up parallelism?** On 2026-05-11 we observed that scaling from 1 to 4 envs did NOT speed up wall-clock training on the OLD wrapper — the gradient-update step is the bottleneck, not the env step. That observation was made BEFORE the wrapper accumulated 11 extra info-dict unpacks and three numpy accumulators per step, so the new wrapper has more per-step CPU cost. A fresh measurement is needed to know where the elbow is on today's stack.

This doc has two parts: a **wrapper audit** (Part 1, verdict-only — no code) and a **benchmark sweep plan** (Part 2, training-runner-executable).

### Headline from the audit (Part 1)

**Verdict: ready for `num_envs > 1`.** All nine audit items pass without code changes. The wrapper's per-instance state is correctly isolated: each wrapper owns its own JAX PRNGKey, JAX env state, BM/distance/episode accumulators (each constructed with `num_envs=1`), and `_episode_counter`. Sheeprl's `make_env` (sheeprl 0.5.7, `sheeprl/utils/env.py:75-77`) verifies the existence of a `seed` field on `cfg.env.wrapper` and explicitly passes a **distinct per-env seed** as a kwarg (`cfg.seed + rank * cfg.env.num_envs + i`) when instantiating each wrapper, overriding the YAML's `seed: ${seed}` interpolation. Sheeprl uses `gym.vector.SyncVectorEnv` here (single-process, sequential per-env step) — no subprocess/JAX re-init cost, no async race conditions. The vectorization layer auto-forwards each wrapper's terminal info-dict to `infos["final_info"][i]` and our `update_from_final_info()` already iterates per-env on that array (`pytorch_agents/run_dreamer_v3.py:350-365`).

One non-blocking semantic note: `Episode/Number` is registered as `MaxMetric`, and each wrapper has its own local `_episode_counter`. Under `num_envs=N`, `Episode/Number` becomes the **max-across-envs episode index per iteration**, not the global cumulative episode count. Under uniform episode rates all envs grow synchronously so the dashboard value ≈ local-per-env count ≈ (total completed episodes / N). Worth flagging to the experiment-analyzer who reads these dashboards — same metric name, slightly different meaning at N>1.

### Headline from the sweep design (Part 2)

Six runs total, all on **node 114 cuda:3** sequentially: `num_envs ∈ {1, 2, 4, 8, 16}` plus a re-baseline of `num_envs=1` at end (sanity check that nothing drifted). Each run: same `01-interoNocicept` config, same seed (42), 10,000 env-steps target, same hardware. The primary metric is `Time/sps_env_interaction` from sheeprl's native logging — the ratio of policy-step elapsed to env-interaction-time, in steps-per-second. Secondary: wall-clock and peak GPU memory. The aggregation table at the end lets you read off where parallelism stops paying.

The sweep does **not** test correctness drift — that's already covered by the audit. The sweep is purely throughput characterization, plus a side-channel correctness check (all 51 Episode/* keys must still appear in each run's WandB summary).

---

## Part 1 — Wrapper audit for `num_envs > 1` readiness

Audit conducted 2026-05-12 by senior-developer against `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` at commit `b46e952` (post-v2 extension, G6 passed).

### Background — how sheeprl spawns N parallel wrappers

Sheeprl's flow under our config (`sync_env: True`, `num_envs: N`):

1. `pytorch_agents/run_dreamer_v3.py:108-124` constructs a `gym.vector.SyncVectorEnv` over N thunks. Each thunk is `partial(RestartOnException, make_env(cfg, cfg.seed + rank * cfg.env.num_envs + i, rank * cfg.env.num_envs, log_dir, "train", vector_env_idx=i))`.
2. Inside `make_env` (sheeprl 0.5.7 wheel, `sheeprl/utils/env.py:75-77`):
   ```python
   instantiate_kwargs = {}
   if "seed" in cfg.env.wrapper:
       instantiate_kwargs["seed"] = seed  # ← per-env seed override
   env = hydra.utils.instantiate(cfg.env.wrapper, **instantiate_kwargs, _convert_="all")
   ```
3. The resulting `env` is wrapped by `gym.wrappers.RecordEpisodeStatistics` (adds `info["episode"]["r"]`/`["l"]` at terminal but preserves all other keys).
4. `SyncVectorEnv.step(action_batch)` calls each env's `step(action[i])` sequentially in the SAME process. On any env's `terminated=True`, SyncVectorEnv collects that env's terminal info into `infos["final_info"][i]` and auto-resets that env by calling `env.reset()` immediately.

### Audit items

| # | Concern | Verdict | Evidence |
|---|---|:---:|---|
| 1 | Per-instance JAX PRNG independence | ✅ | Wrapper `__init__(seed=0)` calls `jax.random.PRNGKey(seed)` (`grid_world_pain.py:91`). Sheeprl `make_env` passes per-env `seed = cfg.seed + rank * cfg.env.num_envs + i` (`run_dreamer_v3.py:115`) as kwarg, overriding the yaml `seed: ${seed}` interpolation. Each wrapper has a distinct initial PRNG stream. |
| 2 | JAX env state isolation | ✅ | Each wrapper owns its own `self._state = jax_reset(...)` (`grid_world_pain.py:92`). `jax_reset`/`jax_step` are pure functional — no global state. No shared mutable arrays between wrappers. |
| 3 | Accumulator state isolation (`make_episode_state` / `make_bm_state` / `make_dist_state`) | ✅ | Each wrapper constructs its own accumulator with `num_envs=1` (`grid_world_pain.py:115, 124, 127`). Underlying state is plain numpy arrays in dataclasses (`src/behavior/episode_metrics.py:86-105`) — owned by the wrapper instance, no module-level mutable defaults. |
| 4 | Terminal info-dict aggregation | ✅ | Wrapper places all 51 `Episode/*` scalars on `info_out` only when `terminated=True` (`grid_world_pain.py:196-220`). SyncVectorEnv promotes that dict to `infos["final_info"][i]`. `pytorch_agents/run_dreamer_v3.py:350-365` iterates `for i, agent_ep_info in enumerate(infos["final_info"])` and calls `update_from_final_info(aggregator, agent_ep_info)` per terminated env. `MetricAggregator.MeanMetric` accepts multiple per-iteration updates — they accumulate into a single mean per iteration window. |
| 5 | `Episode/Number` semantics under MaxMetric + multiple envs | ⚠️ (non-blocking) | Each wrapper has its own `_ep_state.episode_counter` starting at 0, bumped by 1 on every terminal. Under N envs the per-iteration `MaxMetric.compute()` returns `max(local_counter across envs that completed this iteration)`. Under uniform episode lengths all envs advance roughly synchronously, so the dashboard value ≈ per-env-local count ≈ `total_episodes / N`. **Not a bug** — the metric is well-defined — but the **interpretation differs from N=1** (where it = global cumulative). Flag to experiment-analyzer when reading dashboards. |
| 6 | JAX device contention | ✅ | Wrapper module sets `os.environ.setdefault("JAX_PLATFORMS", "cpu")` at line 21, BEFORE any JAX import. All JAX kernels run on CPU; PyTorch owns the GPU. Under SyncVectorEnv (single-process, sequential), JAX kernel launches serialize naturally — no thread-contention risk. |
| 7 | GPU memory scaling | ✅ | World-model batch is `algo.per_rank_batch_size = 16` (sheeprl `dreamer_v3_XS` default) × `algo.per_rank_sequence_length = 64`, independent of `num_envs`. The replay buffer is `EnvIndependentReplayBuffer(buffer_size // num_envs, n_envs=num_envs)` (`run_dreamer_v3.py:223-230`) — total buffer mem is roughly constant. The only `num_envs`-linear footprint is the per-step rollout tensors (obs, action, reward) — for our 5×5 grid these are KB-scale. At `num_envs=16`, dominant cost is still the world-model forward/backward (sheeprl XS preset, ~256-unit MLP) at fixed shape. **Expected peak: < 2 GiB** on the RTX 6000 Ada (48 GiB). No risk of OOM in the sweep range. |
| 8 | Hydra key path for `num_envs` override | ✅ | `pytorch_agents/configs/env/grid_world_pain.yaml:11` declares `num_envs: 4`. The override path is `env.num_envs=N` (top-level Hydra). Sheeprl's algo loop reads `cfg.env.num_envs` directly (`run_dreamer_v3.py:115, 122, 250`). Hydra `_convert_="all"` resolves int values on instantiate. |
| 9 | `scripts/launch_sheeprl.sh` exposure of `num_envs` | ⚠️ (non-blocking) | The current `launch_sheeprl.sh` only takes 4 positional args (`config-yaml`, `gpu`, `tag`, `total-steps`) and does not expose `num_envs`. **The training-runner will append `env.num_envs=N` as an extra arg to the python invocation** — sheeprl's Hydra entrypoint accepts arbitrary trailing overrides. Easiest path: hand-edit `launch_sheeprl.sh` or pass via `run_command.py`. Recommendation in §"Launch command per setting" below uses `run_command.py` with a one-off command string that includes `env.num_envs=N` explicitly — no script change needed. |

### Overall audit verdict

**✅ Wrapper is ready for `num_envs > 1` without any developer-side fix.**

The two ⚠️ items are non-blocking: (5) is a semantic note about how to read `Episode/Number` in the dashboard under N>1 — record-keeping, not correctness; (9) is a launcher-ergonomics note resolved by the training-runner passing the override directly to the python invocation.

### Why the 2026-05-11 num_envs=4 result no longer applies

The 2026-05-11 diary note "sheeprl smoke relaunched with `num_envs=4` — alive, no speedup (gradient compute bottleneck confirmed)" used the OLD wrapper at `tmp/sheeprl/sheeprl/envs/grid_world_pain.py`, which had NONE of the v1/v2 per-step accumulator updates. Today's wrapper at `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` adds: 11 extra info-dict unpacks per step (lines 165-186), three accumulator updates per step (`bm_step_update`, `dist_step_update`, `episode_step_update` at lines 191-193), and a 20+8+24 = 52-key terminal info-dict construction at episode end. The per-step CPU cost has measurably increased. **A fresh measurement is required** to know whether parallelism still hits the gradient bottleneck or whether the env-side cost now scales meaningfully — the answer drives whether `num_envs > 1` is worth using at all.

---

## Part 2 — Parallel-env SPS benchmark sweep

### Plain-language goal

Run the same training config five times, varying only the parallel-env count from 1 to 16, and measure how fast the agent processes environment steps. Each run is short (10,000 steps) — long enough for the SPS measurement to stabilize past warmup, short enough that all five runs fit in under an hour total. Outputs: a table showing wall-clock and steps-per-second per setting, plus a one-paragraph verdict on where the parallelism elbow is.

### Settings to sweep

| Tag | num_envs | total_steps | seed | config | node:gpu | rationale |
|---|---|---|---|---|---|---|
| `sheeprl_sps_n1` | 1 | 10_000 | 42 | `01-interoNocicept` | 114:3 | baseline, single env |
| `sheeprl_sps_n2` | 2 | 10_000 | 42 | `01-interoNocicept` | 114:3 | 2× parallelism |
| `sheeprl_sps_n4` | 4 | 10_000 | 42 | `01-interoNocicept` | 114:3 | matches `jzgkcep4` and current YAML default |
| `sheeprl_sps_n8` | 8 | 10_000 | 42 | `01-interoNocicept` | 114:3 | upper-mid range; tests gradient-bottleneck claim |
| `sheeprl_sps_n16` | 16 | 10_000 | 42 | `01-interoNocicept` | 114:3 | high range; tests CPU saturation on env step (16 wrappers, each doing JAX step + 3 numpy accumulators) |

**Run order**: ascending (n1 → n2 → n4 → n8 → n16). Sequential — one run owns cuda:3 at a time. After each run the next can launch immediately (no GPU warm-up to wait for; `i4ulpn95` is on cuda:2 and untouched by cuda:3 work).

**Why 10,000 steps?** At `num_envs=1` on the metric-parity stack we observed ~50–80 SPS in the G6 smoke (1000 steps in ~15s wallclock); 10,000 steps at SPS 50 = 200 s, at SPS 100 = 100 s. Total sweep: ~10–15 min wallclock. SPS measurement stabilizes after the prefill phase (sheeprl's default `learning_starts = 1024` env-steps for `dreamer_v3_XS`); 10,000 steps gives ~9000 steps of post-prefill steady-state SPS.

**Why same config / seed?** All variation should come from `num_envs`. The `01-interoNocicept` config is the same one used by the metric-parity work — already verified end-to-end, all 51 keys flow. Reusing it makes the side-channel "all 51 keys still appear" correctness check cheap.

### Measurement

| Metric | Source | Notes |
|---|---|---|
| **Primary**: `Time/sps_env_interaction` | sheeprl native, in WandB summary | Computed at `run_dreamer_v3.py:471-477`: `((policy_step - last_log) / world_size * cfg.env.action_repeat) / Time/env_interaction_time`. With `world_size=1` and `action_repeat=1` this is plain policy-steps/sec. **Read the LAST value of this metric in the WandB run summary** — the time-series fluctuates, the summary value is the last logging tick. |
| Secondary: wall-clock | log file timestamps in `logs/<timestamp>_<tag>.log` | Difference between "Log dir:" first-line timestamp and "Saving final model" / process exit timestamp. |
| Secondary: peak GPU memory | `nvidia-smi --query-gpu=memory.used --format=csv -i 3` mid-run | Training-runner samples this 60 seconds after launch (well past warmup) and at run-end. |
| Side-channel correctness: 51 `Episode/*` keys present | `wandb.Api().run(run_path).summary` | At runs end, count keys starting with `Episode/`. Must equal **51** (or close — depends on whether episodes terminate by the relevant termination reasons during 10k steps; some `Episode/Term_*` keys only fire if that termination happens). Minimum acceptance: ≥ 35 Episode/* keys present (the strict 51 requires all four termination reasons to fire). |
| Side-channel correctness: `agent.algorithm == "DreamerV3"` | run.config | One-line check per run. |

### Launch command per setting

The training-runner runs each setting via:

```
./run_command.py 114 "bash scripts/launch_sheeprl.sh configs/experiment/hypervigilance/01-interoNocicept.yaml 3 <TAG> 10000 env.num_envs=<N>"
```

**However** — `launch_sheeprl.sh` currently passes only 3 hardcoded Hydra args (`exp=`, `env.id=`, `algo.total_steps=`) and `exec`s the python interpreter with no trailing-arg passthrough (see `scripts/launch_sheeprl.sh:57-61`). To pass `env.num_envs=N` we have two options:

**Option A (no script change, recommended)** — invoke python directly through `run_command.py`, bypassing the `launch_sheeprl.sh` wrapper. The launcher script's only logic is `cd`, `export GWP_CONFIG_PATH`, `export JAX_PLATFORMS=cpu`, `export CUDA_VISIBLE_DEVICES`, `export SHEEPRL_SEARCH_PATH`, and the python `exec`. Reproduce that inline:

```bash
./run_command.py 114 "cd /media/nas01/projects/Interoceptive-AI/grid_world_pain && \
  GWP_CONFIG_PATH=/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/hypervigilance/01-interoNocicept.yaml \
  JAX_PLATFORMS=cpu \
  CUDA_VISIBLE_DEVICES=3 \
  SHEEPRL_SEARCH_PATH=pkg://pytorch_agents.configs \
  /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -m pytorch_agents.run_dreamer_v3 \
    exp=dreamer_v3_grid_world_pain \
    env.id=sheeprl_sps_n<N> \
    algo.total_steps=10000 \
    env.num_envs=<N>"
```

**Option B (one-line script edit)** — append `"$@"` (positional passthrough of overflow args) to the `exec` line in `launch_sheeprl.sh:57-61`, then call:
```
./run_command.py 114 "bash scripts/launch_sheeprl.sh configs/experiment/hypervigilance/01-interoNocicept.yaml 3 sheeprl_sps_n<N> 10000 env.num_envs=<N>"
```
This is cleaner long-term but counts as a code change (out of plan-author scope). **Pick Option A for this sweep.** A separate one-line script-tidy can land in a follow-up plan.

### Exact commands per run (Option A)

Run order: n1, n2, n4, n8, n16. Each command is independent; wait for the previous to finish before launching the next (post-launch `pgrep -af '<TAG>'` confirms a single PID per the runner protocol).

```bash
# Run 1 — num_envs=1 baseline
./run_command.py 114 "cd /media/nas01/projects/Interoceptive-AI/grid_world_pain && GWP_CONFIG_PATH=/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/hypervigilance/01-interoNocicept.yaml JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=3 SHEEPRL_SEARCH_PATH=pkg://pytorch_agents.configs /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -m pytorch_agents.run_dreamer_v3 exp=dreamer_v3_grid_world_pain env.id=sheeprl_sps_n1 algo.total_steps=10000 env.num_envs=1"

# Run 2 — num_envs=2
./run_command.py 114 "cd /media/nas01/projects/Interoceptive-AI/grid_world_pain && GWP_CONFIG_PATH=/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/hypervigilance/01-interoNocicept.yaml JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=3 SHEEPRL_SEARCH_PATH=pkg://pytorch_agents.configs /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -m pytorch_agents.run_dreamer_v3 exp=dreamer_v3_grid_world_pain env.id=sheeprl_sps_n2 algo.total_steps=10000 env.num_envs=2"

# Run 3 — num_envs=4
./run_command.py 114 "cd /media/nas01/projects/Interoceptive-AI/grid_world_pain && GWP_CONFIG_PATH=/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/hypervigilance/01-interoNocicept.yaml JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=3 SHEEPRL_SEARCH_PATH=pkg://pytorch_agents.configs /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -m pytorch_agents.run_dreamer_v3 exp=dreamer_v3_grid_world_pain env.id=sheeprl_sps_n4 algo.total_steps=10000 env.num_envs=4"

# Run 4 — num_envs=8
./run_command.py 114 "cd /media/nas01/projects/Interoceptive-AI/grid_world_pain && GWP_CONFIG_PATH=/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/hypervigilance/01-interoNocicept.yaml JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=3 SHEEPRL_SEARCH_PATH=pkg://pytorch_agents.configs /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -m pytorch_agents.run_dreamer_v3 exp=dreamer_v3_grid_world_pain env.id=sheeprl_sps_n8 algo.total_steps=10000 env.num_envs=8"

# Run 5 — num_envs=16
./run_command.py 114 "cd /media/nas01/projects/Interoceptive-AI/grid_world_pain && GWP_CONFIG_PATH=/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/hypervigilance/01-interoNocicept.yaml JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=3 SHEEPRL_SEARCH_PATH=pkg://pytorch_agents.configs /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -m pytorch_agents.run_dreamer_v3 exp=dreamer_v3_grid_world_pain env.id=sheeprl_sps_n16 algo.total_steps=10000 env.num_envs=16"
```

### Pre-flight (training-runner — mandatory before each launch)

1. **GPU 3 free?** `./run_command.py --foreground 114 "nvidia-smi --query-gpu=memory.used --format=csv -i 3"` — expect `<500 MiB used`. If a previous run's python is still resident, `pgrep -af python | grep dreamer` and wait for it to exit.
2. **GPU 2 status (informational, do NOT touch)**: `i4ulpn95` parity run from earlier today should be finishing soon (or already done by sweep start). Sweep launches on cuda:3 only; no contention.
3. **conda env exists on node 114**: `./run_command.py --foreground 114 "/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -c 'import sheeprl, torch, jax; print(torch.cuda.is_available())'"` — expect `True`.

### Acceptance criteria per setting

For each run individually:
- Process exits cleanly (no traceback, `Saving final model` line in log).
- WandB run summary contains `Time/sps_env_interaction` (> 0).
- WandB run summary contains `agent.algorithm == "DreamerV3"` and `env.num_envs == <expected N>`.
- WandB run summary contains ≥ 35 `Episode/*` keys (51 if all termination reasons fire; some `Term_Starvation` / `Term_Overeating` are rare in 10k steps).

For the sweep as a whole:
- No run crashes.
- The aggregation table below is filled.
- SPS curve is monotonically non-decreasing through at least one transition (sanity — if SPS drops going n1→n2 the wrapper is in trouble).

### Aggregation table (training-runner fills after the sweep)

| num_envs | wall_clock (s) | sps_env_interaction | sps_per_env | GPU mem peak (MiB) | Episode/* count | wandb run id | Notes |
|---|---|---|---|---|---|---|---|
| 1 | … | … | (= sps) | … | … | … | baseline |
| 2 | … | … | (sps / 2) | … | … | … | |
| 4 | … | … | (sps / 4) | … | … | … | matches `jzgkcep4` regime |
| 8 | … | … | (sps / 8) | … | … | … | |
| 16 | … | … | (sps / 16) | … | … | … | high end |

### Sweep verdict (training-runner writes after the table fills)

Single paragraph naming the elbow:
- "Scaling is linear up to N=X (sps scales N×); plateau or regression after N=Y because <env CPU cost catches up / gradient bottleneck / GPU saturation>."
- Recommendation for production: "Use `env.num_envs=Z` for the next training family."

### Concurrency with existing runs

| GPU | Status at sweep start | Action |
|---|---|---|
| 114:0 | unknown (other-session traffic possible) | training-runner pgrep before claiming, but does NOT use 0 for this sweep |
| 114:1 | unknown | same — not used |
| 114:2 | `i4ulpn95` parity run (started ~1h before sweep design; should finish first) | DO NOT TOUCH. Sweep is on cuda:3 only. |
| 114:3 | target | claim sequentially, one run at a time |

### Risks

1. **JAX-PyTorch memory contention on shared GPU.** JAX is forced to CPU (`JAX_PLATFORMS=cpu` set in launcher), so it never touches the GPU. PyTorch owns cuda:3 fully. **Not a real risk** — included for completeness.
2. **SyncVectorEnv vs AsyncVectorEnv mis-guess.** Confirmed `cfg.env.sync_env: True` in `pytorch_agents/configs/env/grid_world_pain.yaml:12`. Sync = single-process = sequential per-step. Each wrapper's JAX kernels serialize naturally. If a future config flips to `sync_env: False`, the audit changes (subprocess JAX re-init costs N× tens-of-MB). **Mitigation**: do NOT override `env.sync_env=False` in this sweep. The Hydra command lines above keep the default.
3. **SPS measurement variance at short total_steps.** sheeprl's `Time/sps_env_interaction` is computed inside the algo loop at every `metric.log_every` policy_steps (sheeprl default = 1000). With `total_steps=10000` that's 10 measurements. Final-summary value is the last measurement window — should be representative of steady-state past the prefill (`learning_starts=1024`). **Mitigation**: read the **last 3 values** of the time-series, not just the summary, if SPS appears noisy. Take their median.
4. **Episode/* key count varies with episode termination mix.** The 51-key target assumes all four termination reasons (max_steps, starvation, overeating, injury) fire at least once in 10k steps. On `01-interoNocicept` (10×10 grid with predators), max_steps and injury are likely; starvation possible; overeating rare. **Mitigation**: acceptance threshold loosened to ≥35 keys (rather than strictly 51). The point of the side-channel check is "metric pipeline did not break", not "every termination class fired".
5. **GPU 0/1 contention from other sessions.** Sweep uses cuda:3 only — no risk. Pre-flight `nvidia-smi` confirms before each launch.

### Out of scope

- Comparing JAX-side parallelism (JAX rPPO `num_envs` curve). Different algorithm, different stack — separate plan if comparison is wanted later.
- Multi-GPU / multi-rank scaling (`fabric.devices > 1`). DreamerV3-XS is small; the sweep stays single-rank.
- Effect of `num_envs` on learning curve / final performance. This sweep measures ONLY throughput. Learning-curve effects would need much longer runs.

---

## Handoff

- **To training-runner**: execute the 5 commands above sequentially on node 114 cuda:3, fill the aggregation table, write the one-paragraph verdict. Diary rows: one `training-start` per run, one `training-done` per run (matched by tag). Update this doc's "Aggregation table" section in-place. When the table is complete and verdict written, mark this doc's status to `archive` or hand back to senior-developer for a verification pass on the report.
- **To experiment-analyzer (later)**: this sweep produces throughput numbers, not learning curves. If a follow-up "is parallelism worth it for actual training?" question is asked, that's a separate analysis on full-length runs.

