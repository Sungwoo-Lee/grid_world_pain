#!/usr/bin/env python3
import numpy as np
import time
from src.models.dreamer_v3_trainer import ReplayBuffer

def test_sampling_performance():
    print("=" * 80)
    print("PHASE 4 PERFORMANCE TEST: Vectorized Replay Sampling")
    print("=" * 80)

    # Setup
    capacity = 100_000
    seq_len = 16
    batch_size = 16
    obs_dim = 33
    act_dim = 6
    
    buffer = ReplayBuffer(capacity, seq_len, obs_dim, act_dim)
    
    # Fill buffer with some dummy data
    print(f"Filling buffer with {capacity} steps...")
    for i in range(capacity):
        buffer.add(
            np.random.randn(obs_dim).astype(np.float32),
            np.zeros(act_dim).astype(np.float32),
            0.0,
            False,
            False
        )
    
    print(f"\nBenchmarking batch_size={batch_size}, seq_len={seq_len}...")

    # 1. Baseline: Python loop (old logic)
    # We'll use the NEW buffer but manually implement the OLD sampling logic
    def old_sample(buf, b_size):
        obs_b, act_b, rew_b, done_b, first_b = [], [], [], [], []
        for _ in range(b_size):
            start = np.random.randint(0, buf.size - buf.sequence_length)
            indices = np.arange(start, start + buf.sequence_length) % buf.capacity
            obs_b.append(buf.obs[indices])
            act_b.append(buf.actions[indices])
            rew_b.append(buf.rewards[indices])
            done_b.append(buf.dones[indices])
            first_b.append(buf.is_first[indices])
        return {
            'obs': np.array(obs_b),
            'action': np.array(act_b),
            'reward': np.array(rew_b),
            'terminal': np.array(done_b),
            'is_first': np.array(first_b)
        }

    print("\n[Baseline] Running Sequential Sampling (Python loop)...")
    start_time = time.time()
    num_iterations = 100
    for _ in range(num_iterations):
        _ = old_sample(buffer, batch_size)
    baseline_time = (time.time() - start_time) / num_iterations
    print(f"  Baseline time: {baseline_time*1000:.4f}ms")

    # 2. Optimized: Vectorized sampling
    print("\n[Optimized] Running Vectorized Sampling...")
    start_time = time.time()
    for _ in range(num_iterations):
        _ = buffer.sample(batch_size)
    opt_time = (time.time() - start_time) / num_iterations
    print(f"  Optimized time: {opt_time*1000:.4f}ms")

    speedup = baseline_time / opt_time
    print(f"\n{'─'*80}")
    print(f"SPEEDUP: {speedup:.1f}x")
    print(f"{'─'*80}")

    # Correctness check
    batch = buffer.sample(batch_size)
    assert batch['obs'].shape == (batch_size, seq_len, obs_dim)
    assert batch['action'].shape == (batch_size, seq_len, act_dim)
    assert batch['is_first'].dtype == np.float32
    print("\n✓ Correctness check passed (Shapes and Dtypes)")

    return speedup

if __name__ == "__main__":
    test_sampling_performance()
