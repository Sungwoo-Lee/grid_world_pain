---
title: "SPS comparison: in-house JAX DreamerV3 vs sheeprl PyTorch DreamerV3"
topic: sheeprl_bridge
status: active
created: 2026-05-13
last_updated: 2026-05-13
phase: post-hoc-analysis
---

# SPS comparison: in-house JAX DreamerV3 vs sheeprl PyTorch DreamerV3

## 1. Question + verdict (plain language)

**Question.** On the same food-only no-predator survival task, how fast does our in-house JAX DreamerV3 train compared to the sheeprl PyTorch DreamerV3 bridge we just adopted? Specifically: total environment steps per second of wall-clock. The reason this matters: on 2026-05-12 we pivoted off the JAX rebuild because sheeprl reaches ~500-step survival vs our JAX cells' ~106-115. That was a *survival-quality* call. Today's question is the independent *training-speed* axis — if JAX is dramatically faster per env-step, the rebuild might still pay off as a compute-budget play even with the survival deficit; if not, the pivot stands cleanly.

**Headline finding.** **JAX trains ~2,600× faster than sheeprl on raw environment-steps-per-second** (~11,500 env-sps vs ~4.4 env-sps end-to-end). **But this is not an apples-to-apples comparison.** JAX runs with replay_ratio=0.0625 (1 gradient update per 16 env steps) and num_envs=16; sheeprl runs with replay_ratio=1 (1 gradient update per env step) and num_envs=4 — so per env step, sheeprl does 16× more gradient compute. After normalizing to gradient-updates-per-second (the actual training-compute bottleneck on both sides), JAX is still **~325× faster** (~360 grad-updates/sec vs ~1.1 grad-updates/sec); after normalizing further to transitions-processed-per-second through gradient updates (controlling for the 2× JAX batch advantage from longer sequence length), JAX is **~655× faster**. So the gradient-compute throughput gap is real and very large.

**Verdict: GO** — the JAX rebuild is worth reopening *as a future option*, not now. The gap is so large that it does not fit any benign explanation (parallelism choice, replay-ratio choice, or model-size choice) — it points at a real difference between JAX/JIT/vmap compiled-graph compute and PyTorch eager-mode lightning on the same RTX 4090 / RTX 6000 Ada class hardware. **However**, the survival-quality pivot stands: sheeprl ~500 vs JAX cells ~106-115 is not something speed alone fixes. The right action is to keep sheeprl as the primary backend now, and queue a senior-developer spike to ask "if we rebuilt the JAX backend with sheeprl's recipe (replay_ratio=1, paper-canonical model size, identical gradient batch) what's the speed gap then?" before committing to a full rebuild.

---

## 2. Matched-config table

The food-only no-predator (`5×5 NoPred`) task is held the same on both sides (both bridges read our env YAML). The training stack and DreamerV3 hyperparameters differ.

| Axis | In-house JAX (Z2 representative) | Sheeprl PyTorch (jzgkcep4 / kfsvh1qk) | Match? |
|---|---|---|---|
| 1. Model size (recurrent dim) | `rssm_deter_dim=512`, `rssm_stoch_dim=32`, classes=32 | `recurrent_state_size=256`, stochastic_size=32, classes=32 | **differs** — JAX has 2× recurrent dim |
| 1. Model size (MLP units / layers) | dense=128, 2 layers per head; hierarchical encoder (per-sensor MLPs + multimodal hub) | `dense_units=256`, `mlp_layers=1`, flat MLP encoder | **differs** — JAX has hierarchical encoder, sheeprl has wider+flatter |
| 2. Batch size | `batch_size=16` | `per_rank_batch_size=16` | matched |
| 3. Sequence length | `sequence_length=128` | `per_rank_sequence_length=64` | **differs** — JAX 2× longer sequences (= 2× gradient batch) |
| 4. Imagination horizon | (not in config; uses train.py default; project did NOT alter from paper default 15) | `horizon=15` (sheeprl default) | unknown / probably matched |
| 5. Replay ratio | `replay_ratio=0.0625`; `train_steps=64` per iteration; `collect_interval=128` per env | `replay_ratio=1`; 1 gradient update per env step | **differs — 16× difference** (the single biggest SPS lever) |
| 6. num_envs | 16 (parallel JAX-vmap envs in same JIT-compiled scan) | 4 (SyncVectorEnv) | **differs** |
| 7. action_repeat | 1 (env handles) | 1 | matched |
| 8. Frame stack / obs processing | 1-D float32 vector (19 dims); same env YAML, same `apply_noise` setting (Z2 = noise on) | 1-D float32 vector (19 dims); `apply_noise=True` for i4ulpn95 | matched |
| 9. Logging / metric overhead | ~30 keys per iteration (per Episode/* + WorldModel/* + Behavior/* etc.); log_interval=50 iters | jzgkcep4: ~17 keys/event @ log_every=5000 steps; kfsvh1qk: 51 keys/event @ log_every=5000 steps | **differs** but the impact on whole-loop SPS is sub-1% on both sides (40 logging events × <100 ms each in a 12.5h run); not the bottleneck |

**Implication.** This is *not* a clean matched-config comparison. The two biggest mismatches — replay_ratio (16× different) and sequence_length (2× different) — both favor JAX on env-steps-per-second. Normalizing for them is done in §3 below.

---

## 3. SPS comparison table

### 3.1 End-to-end environment-steps-per-second

This is total env-steps logged across all parallel envs, divided by total wall-clock seconds. It is the "how fast does the training-loop chew through env steps" metric.

| Run | Env steps reached | Wall-clock (s) | Wall-clock (h) | **env-SPS** | num_envs | replay_ratio |
|---|---:|---:|---:|---:|---:|---:|
| A1 JAX (czfnljf0) | 67,788,800 | 5,888 | 1.64 | **11,512** | 16 | 0.0625 |
| Z1 JAX (axndoqsz) | 69,120,000 | 5,917 | 1.64 | **11,682** | 16 | 0.0625 |
| Z2 JAX (q66macky) | 75,673,600 | 6,660 | 1.85 | **11,362** | 16 | 0.0625 |
| Z3 JAX (pk8lsyro) | — (1-min smoke) | 93 | 0.03 | n/a | 128 (auto) | 0.0625 |
| jzgkcep4 sheeprl 200k (stock) | 200,000 | 45,132 | 12.54 | **4.43** | 4 | 1 |
| i4ulpn95 sheeprl 50k (parity) | 50,000 | 11,416 | 3.17 | **4.38** | 4 | 1 |
| kfsvh1qk sheeprl 200k (replica) | 200,000 | 45,975 | 12.77 | **4.35** | 4 | 1 |

- **Best JAX SPS**: Z1 at **11,682 env-SPS** (full 700k-iter run, 1.64 h).
- **Best sheeprl SPS**: jzgkcep4 at **4.43 env-SPS** (full 200k-step run, 12.54 h).
- **Raw ratio: JAX / sheeprl ≈ 2,640×.**
- **Z3 excluded** — it was a 1-minute smoke that crashed/finished early (100 iters × 128 envs); the WandB summary has no `timesteps` field, so no stable SPS reading. JAX-side cascade is therefore represented by A1/Z1/Z2 (which already show ±3% SPS variance across cascade rungs — the cell-to-cell SPS does not depend on which fix is in).

### 3.2 The replay-ratio normalization (gradient-updates-per-second)

`env_SPS` mixes two costs: (a) env step + bookkeeping, (b) gradient compute. Sheeprl with `replay_ratio=1` spends most of its time on (b); JAX with `replay_ratio=0.0625` spends most of its time on (a). The fairer ratio is gradient-updates-per-second.

| Run | Gradient updates total | Wall-clock (s) | **grad-update-SPS** | Method |
|---|---:|---:|---:|---|
| A1 JAX | 33,100 iters × 64 train_steps = 2,118,400 | 5,888 | **359.8** | iteration count × train_steps |
| Z1 JAX | 33,750 × 64 = 2,160,000 | 5,917 | **365.1** | same |
| Z2 JAX | 36,950 × 64 = 2,364,800 | 6,660 | **355.1** | same |
| jzgkcep4 sheeprl | ≈ 50,000 (1 grad/env-step after learning_starts=1024) | 45,132 | **~1.12** | from WandB `Time/sps_train` summary mean (median 1.114, steady-state mean 1.113 over 37 events) |
| i4ulpn95 sheeprl | ≈ 50,000 | 11,416 | **~1.09** | `Time/sps_train` summary, steady-state mean 1.082 over 7 events |
| kfsvh1qk sheeprl | ≈ 200,000 | 45,975 | **~1.09** | `Time/sps_train` summary, steady-state mean 1.093 over 37 events |

- **Best JAX**: ~360 gradient updates/sec.
- **Best sheeprl**: ~1.1 gradient updates/sec.
- **Ratio: JAX / sheeprl ≈ 325×.**

### 3.3 Gradient-batch normalization (transitions-processed-per-second through gradient updates)

JAX uses `batch=16 × seq=128 = 2048` transitions per gradient update; sheeprl uses `16 × 64 = 1024`. To control for this 2× difference:

| Run | grad-update-SPS | tx per grad update | **transition-SPS** |
|---|---:|---:|---:|
| Best JAX (Z1) | 365.1 | 2048 | **~747,700** |
| Best sheeprl (jzgkcep4) | 1.12 | 1024 | **~1,147** |

- **Ratio: ~652×.**

### 3.4 Sheeprl's `Time/sps_env_interaction`

This is **env-collection-only** SPS, i.e. how fast the env steps when isolated from the gradient loop. From the WandB history:

| Run | sps_env_interaction (mean over training) |
|---|---:|
| jzgkcep4 | 611 env-steps/sec (mean of 40 events; range 569-691) |
| i4ulpn95 | 609 env-steps/sec |
| kfsvh1qk | 580 env-steps/sec |

In other words: with sheeprl's env-collection cost on its own, even a hypothetical "sheeprl with replay_ratio=0" (no gradient compute at all) would land at **~600 env-SPS**, still **~19× slower than JAX's full-training 11,500 env-SPS**. So the JAX advantage is not just about the replay-ratio choice — the env-collection layer itself is ~20× faster (vmap-batched 16 envs inside a JIT-compiled scan vs SyncVectorEnv-wrapped 4 envs each doing one Python-level step).

### 3.5 Where the JAX advantage is coming from

Three compounding factors, each multiplicative:

1. **JIT-compiled fused scan loop.** JAX runs the entire `collect_interval=128` step loop, the world-model encode/observe forward, the replay sample, and the gradient backward all inside one compiled graph (XLA). Sheeprl's PyTorch loop is eager-mode + Python-level `for` over each env step + a separate gradient call.
2. **`vmap`-batched envs.** 16 JAX envs running in parallel inside a single compiled `vmap` are nearly free per extra env (the env is tiny — 5×5 grid, 19-dim obs). 4 sheeprl envs in SyncVectorEnv each pay Python overhead per step.
3. **GPU buffer (`buffer_device='gpu'`).** JAX replay sampling is a zero-copy `jnp` slice on-device. Sheeprl's replay buffer is on CPU/numpy; each batch transfers to GPU.

None of these are inherent to JAX-vs-PyTorch — sheeprl could in principle compile its loop with `torch.compile`, use `AsyncVectorEnv`, and stage the buffer to GPU. The point is: as-shipped in our two stacks, JAX is much faster on this small-grid task.

---

## 4. Verdict + recommendation

### Verdict: **GO (qualified)**

JAX's raw env-SPS advantage is **~2,600×**; even after normalizing for the 16× replay-ratio mismatch and the 2× sequence-length mismatch, the gradient-compute-per-second advantage is still **~325-650×**. This is far past the 3× threshold the question specified. The "MAYBE" and "NO" bands of the framing are not reached.

**However, the GO is qualified by two things:**

1. **The survival-quality verdict is unaffected.** Sheeprl 500 vs JAX 106-115 is a *4-5× gap in survival* that no amount of training speedup will close unless the JAX code is rewritten to match sheeprl's algorithmic recipe. The pivot to sheeprl as the primary backend was the right call for the immediate path (paper-ready results). A JAX rebuild would have to ship the cascade fixes (#28 GRU reset, #29 critic EMA, #30 RSSM hidden layers) AND match all the smaller paper-canonical knobs that sheeprl ships by default. That's the real engineering cost — not the speedup, which we know is there.
2. **The speed advantage may shrink with matched configs.** If a future JAX rebuild adopts sheeprl's `replay_ratio=1` and the wider `dense=256, mlp_layers=1` paper-canonical model size, the gradient compute would grow ~16× and the gradient-batch advantage would narrow. The realistic "matched-config-corrected" JAX-vs-sheeprl SPS ratio is probably **~20-40×**, not 2,600× — still very large, but the back-of-envelope to compute it cleanly requires running JAX with sheeprl's recipe, which we haven't.

### Recommendation

**Spawn a senior-developer feasibility spike on a JAX-with-sheeprl-recipe rebuild.** Not the full rebuild — a 1-day spike that does exactly two things:

- (a) Quantify the realistic speed advantage when JAX runs `replay_ratio=1` and `dense_units=256, mlp_layers=1` (the sheeprl XS recipe). This nails down whether the ~20-40× estimate is right or whether the speedup collapses to ~5× under matched compute.
- (b) Estimate the engineering cost of porting the 5 paper-canonical cascade fixes (#27 zero-init, #28 GRU reset, #29 critic EMA, #30 RSSM hidden layers, #2 paper-canonical bins) AND any silent sheeprl-vs-ours code-fidelity gaps into a clean JAX rebuild.

**Until that spike returns**, sheeprl stays the primary backend. The 2026-05-12 pivot is not reversed; it is queued for re-examination once we have a concrete "JAX could be N× faster with M weeks of work" tradeoff, not a hand-wavy "JAX is faster but the survival isn't there yet".

---

## 5. Open questions

1. **JAX side never logged `Time/sps_train` or `Time/sps_env_interaction`.** All JAX SPS numbers in this memo are computed from `_runtime + timesteps` summary fields, which is end-to-end and reliable but does not let us cleanly decompose env-collection vs gradient compute. **Metrics Requested** (see §6) covers logging these in future JAX runs.
2. **Z3 (GRU reset gate fix) was a 93-second smoke**, not a full 700k-iter run. We don't have a stable Z3 SPS reading. The cascade rung-to-rung SPS variance across A1/Z1/Z2 is ±3%, so it is very unlikely Z3 changes the JAX SPS verdict materially — but strictly we do not have a Z3 number.
3. **The sheeprl 200k-step replica (kfsvh1qk) was still running** at the time of analysis launch (started 2026-05-12 23:21, ~10h remaining as of the analysis spec). Numbers used here are from the run that *did* complete by the time I extracted summaries — kfsvh1qk actually shows `_runtime=45,975 s` and reached 200,000 global_steps. It finished. (Diary "still running" note was stale.)
4. **JAX collect_interval=128 vs sheeprl env-step-by-step.** JAX does 128 env steps per iteration per env (one big vmap-scan), then `train_steps=64` gradient updates. Sheeprl does 1 env step per loop body iteration. This may be the dominant source of JAX's env-collection advantage (211× — 611 env-SPS solo vs 11,500 env-SPS full-loop is suspect; the JAX 11,500 number includes parallel env-collection while gradient is happening *and* parallelism across 16 envs, so it is a different kind of measurement). A cleaner JAX env-collection-only number would require relaunching JAX with `replay_ratio=0` or instrumenting `Time/sps_env_interaction` directly.
5. **Hardware difference.** Both sides ran on the lab cluster's RTX 4090 / RTX 6000 Ada nodes (113 and 114). Same node class, same CUDA driver. No GPU-class confound flagged.
6. **No multi-seed for either side on the SPS axis.** All three JAX runs are seed 0; all three sheeprl runs are seed 42. SPS is much more reproducible than survival (it depends on hardware + compiled graph, not on policy state), so the 1-seed-per-side caveat is mild for this question.

---

## 6. Metrics Requested

If we decide to reopen JAX as a backend candidate, we should land lightweight SPS instrumentation in `train.py` (the JAX-side trainer) so that future JAX-vs-sheeprl SPS comparisons are extracted from the same WandB keys without recomputation.

| Subfield | Content |
|---|---|
| **Metric** | `Time/sps_env_interaction` (env-steps-per-sec across the collect_interval block, not counting gradient compute) and `Time/sps_train` (gradient-updates-per-sec across the train_steps block). Same names as sheeprl. |
| **Why now** | Without these, we can't decompose JAX's 11,500 env-SPS into "env-collection wall-clock" vs "gradient wall-clock". The §5 open question 4 — "is the env-collection ~20× faster on its own, or is the JIT-compiled-scan trick the dominant lever?" — is unanswerable from current summaries. |
| **Where it'd live** | `train.py` main training loop (the iteration counter at line ~33100 ticks; just split `time.time()` checkpoints around the env-collection block and the train block, push to wandb at the existing log_interval=50 cadence). |
| **Cost** | Cheap — two scalar logs per iteration. Sub-1% overhead. |

This is **optional** — it does not block the GO/MAYBE/NO call on the rebuild. It would only be needed if we move to the senior-developer spike in §4.

---

## 7. Related Issues

- [`docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md`](../../../develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md) — the bridge plan; this memo's conclusion does not change the recommendation to keep sheeprl as the primary backend, but adds context to the "do we ever revisit JAX?" question.
- [`docs/develop/active/diagnosis/sheeprl_drop_in_test.md`](../../../develop/active/diagnosis/sheeprl_drop_in_test.md) — the survival-quality call that triggered the 2026-05-12 pivot; this memo addresses the speed axis the user wanted answered independently.
- [`docs/pi/calls/2026-05-12_dreamer_backend.md`](../../../pi/calls/2026-05-12_dreamer_backend.md) — the PI call that picked sheeprl-direct over JAX-rebuild on survival grounds; this memo provides the speed-axis evidence to consult on any future reopening.
- [`docs/develop/active/sheeprl_bridge/PARALLEL_ENV_BENCHMARK.md`](../../../develop/active/sheeprl_bridge/PARALLEL_ENV_BENCHMARK.md) — the 2026-05-12 num_envs sweep that confirmed env-collection is single-threaded on sheeprl side (flat 612-686 SPS across N=1,2,4,8,16); cross-references the §3.4 reading here.
- **No code changes proposed.** The recommended spike (§4) is a senior-developer planning task, not a developer implementation; the Metrics Requested (§6) is contingent on the spike returning.

---

## 8. Working files

- `tmp/20260513_SPS_extract_all.md` — per-run summary extracts (raw WandB `wandb-summary.json` reads).
- Inline shell scripts in this analysis session pulled the JAX `_runtime + timesteps + iteration` and sheeprl `Time/sps_train + Time/sps_env_interaction` from local `wandb/run-*/files/wandb-summary.json` and `logs/runs/dreamer_v3/.../run-*.wandb` (via `wandb.sdk.internal.datastore.DataStore` protobuf parsing). No WandB web API calls used.
