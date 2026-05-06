---
title: "Positive-Buffer Host-Loop Vectorization"
topic: refactors
status: active
created: 2026-05-06
last_updated: 2026-05-06
---

# Positive-Buffer Host-Loop Vectorization

> **Status**: PROPOSED (stub — no implementation)
> **Date**: 2026-05-06
> **Author**: senior-developer (planning)
> **Related**:
>   - [dreamer_v3_vs_rppo_speed_profile.md](../diagnosis/dreamer_v3_vs_rppo_speed_profile.md) — speed profile that surfaced this
>   - [MIXTURE_SAMPLING_PLAN.md](MIXTURE_SAMPLING_PLAN.md) — the original two-buffer architecture this code came from

---

## 1. Problem

`train.py:1048-1073` (GPU path) and `train.py:1092-1114` (CPU path) contain a Python for-loop that walks each `sequence_length`-sized block of newly collected transitions, checks whether the block contains any positive reward, and copies it to the dedicated positive buffer:

```python
for b in range(num_written_blocks):       # num_written_blocks = num_envs (= 16)
    blk_rewards = rew_flat[blk_start:blk_end]
    if buffer._on_gpu:
        has_positive = bool(jnp.any(blk_rewards > 0.0))   # <-- device→host sync
    else:
        has_positive = bool(np.any(blk_rewards > 0.0))
    if has_positive:
        positive_buffer.add_batch(...)
```

The `bool(jnp.any(...))` cast forces a per-block device→host transfer on the GPU path, blocking the Python thread until the small reduction completes. At `num_envs=16`, this fires **16 syncs per training iteration**.

## 2. Speed-profile evidence

From [dreamer_v3_vs_rppo_speed_profile.md](../diagnosis/dreamer_v3_vs_rppo_speed_profile.md), §4 (XSpace analysis of the 50-iter Dreamer trace):

- Host-side `PjitFunction(_reduce_any)` calls: 1600 over 50 iters = **32 per iter** (16 from this loop + 16 from a downstream check).
- Total `PjitFunction(_reduce_any)` host wall-clock under CUPTI: 132.6 ms over 50 iters = **2.65 ms/iter**.
- Other small JIT calls clearly attributable to this host loop (`add`, `dynamic_slice`, `less`, `select_n`, `_squeeze`, `_broadcast_arrays`, `scatter`, `broadcast_in_dim`): each fires 8230 times over 50 iters ≈ 165 calls/iter, summing to **~12 ms/iter** of host-side small-JIT scheduling overhead.

Under CUPTI overhead this nets to ~15 ms/iter; without CUPTI (baseline) the host loop should cost roughly the same since the device-sync stalls dominate the wall-clock and CUPTI overhead is per-kernel-launch-callback, which has *less* relative impact on tiny ops than on bulk training kernels.

**Estimated baseline cost: 5-15 ms/iter out of 6028 ms/iter = ~0.1-0.25% of Dreamer's iter time.**

This is **not** a top bottleneck — the speed profile refutes the "host-sync dominates" hypothesis. But it is a clean, low-risk vectorization opportunity that removes 16 device-syncs per iter, useful for future scaling to higher `num_envs` (where the loop count grows linearly).

## 3. Proposed fix (sketch — not a full plan)

Vectorize the per-block check on-device:

```python
# Replace the for-loop with one device-side reduction over all blocks at once.
blk_rewards = rew_flat.reshape(num_written_blocks, seq_len)
has_positive_per_block = jnp.any(blk_rewards > 0.0, axis=1)   # (num_blocks,)
# Single host sync — all 16 booleans in one transfer:
has_positive_np = np.asarray(has_positive_per_block)
for b in np.where(has_positive_np)[0]:
    positive_buffer.add_batch(
        obs_flat[b*seq_len:(b+1)*seq_len], ... )
```

This collapses 16 sync barriers into 1.

A more aggressive version would push the entire selection + copy into a single JIT'd `add_batch_filtered(obs_flat, ..., mask)` kernel that drops zero-reward blocks on-device, eliminating the host-side conditional altogether. That's a larger refactor and would intersect with `MIXTURE_SAMPLING_PLAN.md`'s buffer interface.

## 4. When to take this on

- After larger Dreamer optimizations (imagination horizon trim, replay-ratio reduction) have been evaluated — those have ~20-50× the impact.
- Before scaling experiments to `num_envs=64+` where the host-loop cost grows linearly.

## 5. Status

Not scheduled. Open this plan if/when revisiting the buffer-add path or scaling `num_envs` significantly.
