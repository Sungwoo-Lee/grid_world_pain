---
title: "H1 — dreamer_srl throughput investigation: where the 0.61 s/iteration actually goes"
topic: diagnosis
status: active
created: 2026-07-27
last_updated: 2026-07-27
---

# H1 — Where does dreamer_srl's wall-clock go, and what would make it faster?

## Question (plain-language entry point)

Our in-house DreamerV3 trainer ("dreamer_srl") trains at roughly 209 environment
steps per second on its medium-size model, while the recurrent-PPO trainer on the
same grid-world environment does ~35,000. The GPU sits at 15–70% utilization, not
100%. This document asks: **is the trainer wasting time on something fixable, and
if so, where exactly?**

**Headline answer: yes, on two distinct things.** First, roughly **a third of all
wall-clock time (34% on the medium model, 44% on the extra-small model) is spent
not training at all** — every ~1,000 episodes the training loop stops dead while
it saves a checkpoint, plays 3 evaluation episodes one step at a time, and then
**waits for a subprocess to render those episodes into a 300-DPI MP4 video**
(~2–4 minutes of blocking per event, ~8 events per hour). Second, within a normal
training iteration, the acting path (the network choosing actions each step) runs
**un-jitted** — it issues ~436 separate GPU operations from Python every
iteration instead of one compiled call — and episode resets loop over finished
environments one at a time in Python. Fixing the first is a config/async change
worth ~1.5×; fixing the second is a code change worth another ~1.3–1.5×; together
they plausibly take the medium model from ~209 to ~450+ env-steps/s without
touching training semantics.

All numbers below are from ground truth: the live runs' own logs and saved
configs (node 114, 2026-07-26 `dp1` runs), the current v3.0 code, and local
CPU-side measurements of the exact code paths at the live shapes. No GPU
micro-benchmarks were run (prohibited for this investigation).

Sibling investigations running in parallel:
[[dreamer_srl_faithfulness_review]] ·
[[DREAMER_SRL_INVESTIGATION]] ·
[[dreamer_srl_settings_regime_critique]] ·
[[gridworld_vs_dreamerv3_benchmarks_difficulty]]

---

## 1. Ground-truth throughput of the live runs

Extracted from the live runs' stdout logs (`logs/20260726_042547_dsrl_*.log`,
tqdm lines parsed over the last ~21 h window; ~125k–187k iterations each):

| Run | cumulative SPS (log postfix) | **steady-state s/iter** (median of 100-iter segments) | steady SPS | mean s/iter (incl. stalls) | wall-clock lost to periodic stalls |
|---|---|---|---|---|---|
| b03 **M** (`20260726-042618_dsrl_b03_M_dp1`) | 209 | **0.40** (p10 0.38, p75 0.42) | **320** | 0.607 | **34%** (25,772 s of 76,253 s) |
| b03 **XS** (`20260726-042618_dsrl_b03_XS_dp1`) | 314 | **0.22** (p10 0.20) | **582** | 0.408 | **44%** (33,614 s of 76,337 s) |

Two immediate corrections to the working picture:

1. **The "209 SPS" figure is not the trainer's steady-state speed.** Steady-state
   is 320 SPS (M) / 582 SPS (XS). The gap is periodic multi-minute stalls: 156
   slow segments (M) in the window, i.e. one per ~800 iterations, each costing
   +60 to +247 s (mean ≈ 165 s) over the 0.40 s/iter baseline. Slow-segment rate
   matches the checkpoint cadence exactly (175 `[CHECKPOINT] Saved.` + video
   events in the same window; `checkpoint_frequency: 1000` episodes ≈ 745
   iterations at the observed ~1.34 episodes finished per iteration).
2. **The GPU-utilization pattern (15–70%, never pinned) is explained**, not
   mysterious: during the 34–44% stall time the GPU is idle (checkpoint save,
   single-env eval, CPU matplotlib render in a subprocess the trainer *blocks
   on*); during steady iterations only the 8-grad-step jitted scan keeps the GPU
   busy (~90–240 ms of the 400 ms), the rest is host-side Python/dispatch.

### What one stall event contains (code-verified)

`dreamer_srl_main.py:1768–1841`: Orbax checkpoint save → `dreamer_srl_eval_rollout`
(**the single-env, un-jitted, Python-per-step rollout** — `eval.py:101–192`; the
jitted batched variant `dreamer_srl_eval_rollout_batched` exists at
`eval.py:296` but is *not* called by the driver) for `eval_video_episodes: 3`
episodes of up to 500 steps → `_render_and_upload` (`eval.py:508–582`) which
runs `scripts/eval/render_recordings.py` via a **blocking `subprocess.run`**,
rendering up to ~1,500 matplotlib frames at `video_dpi: 300` on CPU, then
consolidates and uploads the MP4. The saved run config
(`results/JAX_DreamerSRL/20260726-042618_dsrl_b03_M_dp1/models/env_config.yaml`)
confirms `video_during_training: true`, `checkpoint_frequency: 1000`,
`eval_video_episodes: 3`, `visualization.video_dpi: 300`, `fps: 5`;
`stats_during_training: false` (so the separately-known sequential-stats finding,
[[findings_dreamer_main]] Finding 2, is dormant in these runs but the video pass
has the same shape).

---

## 2. Iteration time budget (steady-state 0.40 s, M-size, 128 envs, 8 grad steps)

One driver iteration = 1 vectorized env step (128 envs) + buffer add + (train
gate) sample + 1 jitted scan of 8 gradient steps + logging. Components below are
measured on the dev box CPU at the exact live shapes where the cost is
host-side (Python/dispatch — these carry over to the GPU runs since they never
touch the accelerator), and estimated where GPU compute is involved:

| Component | Cost / iteration | How obtained |
|---|---|---|
| **8 gradient steps** (jitted `_scan_grad_steps`, B=16 T=64) | **~90–240 ms** | Lower bound: prior 10.7 ms/step × 8. The M-vs-XS steady delta (0.40 − 0.22 = 0.18 s) is mostly model-size-dependent work, implying in-situ grad-step cost nearer 25–30 ms on the shared node — needs the (dead) timers to pin down, see §5 |
| **Acting path — un-jitted** (`Player.get_actions`, `dreamer_srl_main.py:304–352`) | **~50–80 ms** | Traced: **436 top-level eager op dispatches** (533 nested primitives) per call — encoder `jax.vmap` + RSSM `dynamic` + actor, all eager NNX. CPU measurement: 74 ms eager vs 22 ms jitted → ≥52 ms is pure host dispatch overhead that jitting removes |
| **Per-done-env autoreset loop** (`dreamer_srl_main.py:1644–1655`) | **~22 ms avg** (≈1.34 dones/iter × 16.3 ms) | Measured warm per done env: `jax_reset` 1.0 ms + `get_observation` 0.1 ms + `tree_map` `.at[idx].set` over the **39-leaf** EnvState **15.1 ms** (39 dispatches, each copying the full [128,...] leaf). Worse early in training when episodes are short |
| **`env.step`** (`ParallelEnv` = un-jitted `vmap` over jitted `jax_step`) | **~10 ms** | Measured: 9.5 ms/call at num_envs=1 and 10.0 ms at 128 → the cost is per-call host machinery (vmap re-trace + pytree flatten of 39-leaf state + EnvParams), not compute |
| **Replay buffer, CPU** (`EnvIndependentSequentialReplayBuffer`) | **~9 ms** | Measured at live shapes (128 sub-buffers × 7,812 rows, full): `add` 1.3 ms (Python loop over 128 envs), done-mask add 0.03 ms, `sample(B=16, n=8, T=64)` 7.3–8.0 ms |
| **H2D transfer** (one `jax.tree.map` per iteration — Option S, verified live at `dreamer_srl_main.py:1951`) | ~1 ms | 1.21 MB/iteration measured |
| **`nnx.split`×7 + `nnx.update`×7** at the scan boundary | ~10–15 ms | Measured 4 modules (92 leaves): split 6.3 ms + update 1.4 ms; 3 optimizer states extra |
| **Infos → numpy** (24 keys), rewards/dones sync, episode bookkeeping, logging pushes | ~5–10 ms | 24-key conversion 0.06 ms on CPU; on GPU ~24 small D2H transfers ≈ 2–3 ms; Python loops small |
| **Sum of identified** | **~200–390 ms** | vs 400 ms measured steady iteration |

The residual uncertainty sits almost entirely in the true in-situ grad-step cost
(the four live runs also share node 114's host CPUs with each other *and* with
the periodic render subprocesses, which inflates every host-side number above
during co-residency). The decisive instrument already exists and is **dead
code**: `t_env_total` / `t_train_total` are accumulated at
`dreamer_srl_main.py:1373–1374, 1516, 2042` **but never logged anywhere** — not
in `log_dict`, not in the final print.

### Answers to the specific questions posed

- **Q1 (budget):** table above. Steady-state ≈ grad scan (largest single item,
  90–240 ms) + ~120–150 ms of host-side overhead (acting path, autoreset, env
  dispatch, buffer, NNX graph ops) + periodic stalls worth 34–44% of wall-clock
  on top.
- **Q2 (10.7 ms/grad-step still current?):** cannot be re-verified without a GPU
  measurement (out of scope here); the M-vs-XS steady-state arithmetic suggests
  the in-situ figure on the shared node is higher (~25–30 ms). It is
  GPU-compute + scan-internal; per-iteration *dispatch* of the scan is one cached
  jit call (compile-once verified: constant `_G`, fixed shapes). Restructuring
  the 8 steps into fewer/larger batches is **not** a pure speed knob — it changes
  the number of parameter updates per env step (replay-ratio semantics) and is
  not recommended.
- **Q3 (CPU buffer):** efficient enough — fully vectorized numpy gathers; ~9
  ms/iteration total at 1M/128 shapes; 2% of the iteration. The known
  1M-element-Python-list hazard in `SequentialReplayBuffer.sample`
  (`buffers.py:422–424` builds `np.array(list(range(...)))` over the whole
  capacity when full) is defused at 128 envs because per-env capacity is only
  7,812 — but it *would* bite in single-env GPU-buffer mode at 1M capacity.
- **Q4 (env host-device ping-pong):** envs are JAX on the default (GPU) device;
  per iteration there is one D2H of obs/rewards/dones/24 infos (~few ms) and one
  H2D inside the acting path. The ping-pong is real but cheap (~13 KB obs); the
  expensive part is not the transfer, it's that env stepping, acting, and resets
  are all *driven* from Python per iteration (rPPO amortizes all of this inside
  one jitted scan of 128 steps).
- **Q5 (more work than sheeprl?):** three additions beyond the sheeprl port
  surface: (a) 24-key per-step info computation + conversion and behavior/BM
  accounting (small, ~5 ms); (b) the **blocking in-loop video render** — sheeprl
  logs metrics but never renders matplotlib MP4s inside the training loop (this
  is the 34–44% item); (c) per-done-env `tree_map` state surgery for autoreset
  (sheeprl's gym vector env resets in the env process). The double buffer-row
  write at dones matches sheeprl and is not extra.
- **Q6 (num_envs 256):** at 128 we are past the knee only for the *jitted* part.
  Host overhead (~120–150 ms) is constant per iteration, so doubling envs
  amortizes it 2× while grad-step count doubles (`_G` = 16) to keep the replay
  ratio. Buffer total memory is **constant** (driver divides: `buffer.size //
  num_envs`, `dreamer_srl_main.py:789` — 256 envs → 3,906 rows/env, still ≫
  seq_len 64). Rough projection for M: steady ~0.49 s per 256-step iteration →
  ~520 SPS (+60%). Caveats: doubled dones/iter makes the Python autoreset loop
  worse (do candidate #3 first), and per-env history shortens.

---

## 3. Ranked optimization candidates

| # | Change | Est. payoff (M) | Effort | Risk | Notes |
|---|---|---|---|---|---|
| 1 | **Stop blocking training on video renders** — make `_render_and_upload` async (`subprocess.Popen` + poll/upload on completion or in the child), and/or config-only: raise `checkpoint_frequency` 1000→5000, drop `video_dpi` 300→100 | **×1.5** (209→~310 SPS; XS ×1.8) | Low (config-only variant: zero code) | Low — no training semantics touched; async needs care that renders don't pile up (cadence 7 min vs ~2–4 min render: OK) and WandB upload ordering | Render already runs in a separate CPU-pinned process; the trainer just *waits* for it. Also route the video-pass rollout through the existing jitted `dreamer_srl_eval_rollout_batched` — same fix direction as [[findings_dreamer_main]] Finding 2 |
| 2 | **Jit the acting path** — one jitted `(params, h, z, prev_a, obs, is_first, key) → (h', z', action)` function; keep player state + obs on device | **~50–80 ms/iter** → with #1, ~209→~400 SPS combined | Medium | Low-medium — same ops, same numerics; needs the `is_first` gating and PRNG threading inside the jit; `test_grad_parity`-style check advisable | Removes 436 eager dispatches/iter. Also fixes eval rollout speed (same eager path, `eval.py:131–171`) |
| 3 | **Batch the autoreset** — replace per-done-env Python loop + 39-leaf `.at[idx].set` tree_map with one fixed-width masked reset (vmapped `jax_reset` over all envs + `jnp.where(done_mask, ...)`, the rPPO idiom already used for the *player* reset) | ~15–25 ms/iter avg; more early-training and at 256 envs | Medium | Low — same values; fixed shapes (no recompile storm; the existing constant-shape key split at `1642–1643` already anticipates this) | Prerequisite to making candidate #5 pay fully |
| 4 | **Log the dead timers + add `t_policy`/`t_eval`** — `t_env_total`/`t_train_total` are computed but never emitted | Enables exact budget; trivial | Trivial | None | Do this first in the same PR as anything else; it converts §2's bands into measurements |
| 5 | **num_envs 128→256** (config-only) | +~60% steady (with #3 in place) | Trivial | Low-medium — shorter per-env replay history (7,812→3,906 rows); doubled grad steps per iteration keep replay ratio, VRAM for activations grows modestly | Buffer RAM constant by construction |
| 6 | **K-step collection per outer iteration** — fuse policy+env+autoreset into a jitted `lax.scan` over K steps (rPPO architecture), bulk-add K rows to the buffer, run K×`_G` grad steps | Potential further ~2× (host overhead amortized K×) | High | Medium — buffer write order, done-boundary double-rows, and Ratio semantics must be preserved exactly; this is the "faithful port" boundary | Only worth it after #1–#3; at that point steady-state is grad-bound and the ceiling is ~128/(8×t_grad) |

**Explicit verdict:** this is **not** a "no further wins" situation. Candidates
1–2 alone are a plausible ~2× on the medium model (209 → ~400 SPS) with low
risk, and #1's config-only form costs nothing to try. The trainer will remain
1–2 orders of magnitude slower than rPPO regardless — DreamerV3 does ~8 full
world-model+actor+critic updates per 128 env steps by design (replay ratio),
where rPPO does 4 cheap updates per 16,384 env steps; that gap is algorithmic,
not implementational.

---

## 4. Prior "already fixed" claims — re-verified in current v3.0 code

Per the fresh-skepticism directive, each was checked against the current source,
not the fix-era docs:

| Claim | Status in current code |
|---|---|
| One H2D per iteration (Option S) | ✅ single `jax.tree.map(jnp.asarray, local_data)` at `dreamer_srl_main.py:1951` |
| Compile-once grad scan, constant length (`_G`) | ✅ `@jax.jit` `_scan_grad_steps` (only jit in the driver, line 1161); `n_grad_steps_scan = _G` at 1906 |
| CPU buffer for num_envs>1 (GPU-buffer bincount storm) | ✅ hard error at `dreamer_srl_main.py:800–805`; GPU mode restricted to num_envs==1 |
| Fixed-width masked player resets (reset-storm) | ✅ `Player.init_states(done_mask=…)` path, lines 251–271 |
| Constant-shape autoreset key split | ✅ `jax.random.split(k_autoreset, num_envs)` at 1642–1643 (the *loop that consumes it* is candidate #3's target — correctness fixed, cost remains) |
| Prefill/train-gate (P6/P8) | ✅ inclusive prefill gate at 1452; `ratio_steps = policy_step − prefill_steps × num_envs` at 1900; `buffer.ready_to_sample` at 1913 |

## 5. Proposed follow-ups requiring user approval

1. **One instrumented GPU run** (any free non-114 node, ~30 min, M-size,
   128 envs) with the §3-#4 timers logged, to convert the §2 bands into exact
   numbers — in particular the in-situ grad-step cost (10.7 vs ~25–30 ms
   question) and the acting-path share on GPU.
2. If #1 (async render) is approved as a code change, hand a plan to
   `developer` via the normal senior-developer plan flow; the config-only
   variant (`checkpoint_frequency`, `video_dpi`) can be adopted by
   `experiment-designer` for the *next* launches without code.

## Links & provenance

- Live logs parsed: `logs/20260726_042547_dsrl_b03_M_dp1.log`, `..._b03_XS_dp1.log` (segment stats: `/tmp/seg2.txt`, `/tmp/seg3.txt` method — tqdm `iter=` postfix lines, monotonic filter, 100-iter granularity)
- Saved live config: `results/JAX_DreamerSRL/20260726-042618_dsrl_b03_M_dp1/models/env_config.yaml`, `models/agent_config.yaml`
- Code: `src/algorithms/dreamer_srl/dreamer_srl_main.py` (driver), `buffers.py`, `eval.py`, `src/environment/wrapper.py`
- CPU measurements: this doc §2 (conda interpreter, `JAX_PLATFORMS=cpu`, live shapes)
- Untriaged-findings source: `docs/reviews/diagnosis_20260723/findings_dreamer_main.md` (Finding 2 — sequential stats eval — is the dormant sibling of the video-pass cost)
