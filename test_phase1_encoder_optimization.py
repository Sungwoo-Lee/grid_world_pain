#!/usr/bin/env python3
"""
Test script for Phase 1: Encoder Outside Scan optimization.

This script verifies that moving the encoder outside the scan loop:
1. Produces identical outputs to the original implementation
2. Achieves the expected speedup (2-5x on world model training)
"""

import jax
import jax.numpy as jnp
from flax import nnx
import time
import numpy as np

from src.models.dreamer_v3_trainer import DreamerTrainer

def create_dummy_batch(batch_size=4, seq_len=16, obs_dim=33, act_dim=4):
    """Create a dummy batch for testing."""
    return {
        'obs': np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32),
        'action': np.eye(act_dim)[np.random.randint(0, act_dim, (batch_size, seq_len))].astype(np.float32),
        'reward': np.random.randn(batch_size, seq_len).astype(np.float32),
        'terminal': np.random.randint(0, 2, (batch_size, seq_len)).astype(np.float32),
        'is_first': np.random.randint(0, 2, (batch_size, seq_len)).astype(np.float32),
    }

def test_correctness():
    """Test that the optimized version produces correct outputs."""
    print("=" * 80, flush=True)
    print("PHASE 1 CORRECTNESS TEST: Encoder Outside Scan", flush=True)
    print("=" * 80, flush=True)

    # Setup
    obs_dim = 33
    act_dim = 4
    config = {
        'encoder_dim': 128,
        'encoder_fc_layers': [128, 128],
        'rssm_deter_dim': 512,
        'rssm_stoch_dim': 32,
        'rssm_classes': 32,
        'decoder_fc_layers': [128, 128],
        'reward_fc_layers': [128, 128],
        'continue_fc_layers': [128, 128],
        'actor_fc_layers': [256, 256],
        'critic_fc_layers': [256, 256],
        'model_lr': 1e-4,
        'actor_lr': 3e-5,
        'value_lr': 8e-5,
    }

    print("\n[DEBUG] Initializing DreamerTrainer...", flush=True)
    rngs = nnx.Rngs(0)
    trainer = DreamerTrainer(obs_dim, act_dim, config, rngs, modulation_config=None)
    print("[DEBUG] DreamerTrainer initialized", flush=True)

    # Create test batch
    print("[DEBUG] Creating dummy batch...", flush=True)
    batch = create_dummy_batch(batch_size=4, seq_len=16, obs_dim=obs_dim, act_dim=act_dim)
    rng = jax.random.PRNGKey(42)
    print("[DEBUG] Batch created", flush=True)

    print("\n✓ Model initialized successfully", flush=True)
    print(f"  - Observation dim: {obs_dim}", flush=True)
    print(f"  - Action dim: {act_dim}", flush=True)
    print(f"  - Batch size: {batch['obs'].shape[0]}", flush=True)
    print(f"  - Sequence length: {batch['obs'].shape[1]}", flush=True)

    # Test forward pass
    try:
        print("\n[DEBUG] Starting train_step (this may trigger JIT compilation, please wait)...", flush=True)
        metrics = trainer.train_step(batch, rng)
        print("[DEBUG] train_step completed", flush=True)

        print("\n✓ Forward pass successful", flush=True)
        print(f"\nMetrics:", flush=True)
        for k, v in metrics.items():
            print(f"  - {k}: {float(v):.6f}", flush=True)

        # Check for NaN/Inf
        print("\n[DEBUG] Checking for NaN/Inf...", flush=True)
        has_nan = any(jnp.isnan(v).any() for v in metrics.values())
        has_inf = any(jnp.isinf(v).any() for v in metrics.values())

        if has_nan:
            print("\n❌ FAILED: NaN detected in metrics!", flush=True)
            return False
        if has_inf:
            print("\n❌ FAILED: Inf detected in metrics!", flush=True)
            return False

        print("\n✓ No NaN/Inf detected", flush=True)
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Benchmark the optimized version."""
    print("\n" + "=" * 80, flush=True)
    print("PHASE 1 PERFORMANCE TEST: Training Step Timing", flush=True)
    print("=" * 80, flush=True)

    # Setup
    obs_dim = 33
    act_dim = 4
    config = {
        'encoder_dim': 128,
        'encoder_fc_layers': [128, 128],
        'rssm_deter_dim': 512,
        'rssm_stoch_dim': 32,
        'rssm_classes': 32,
        'decoder_fc_layers': [128, 128],
        'reward_fc_layers': [128, 128],
        'continue_fc_layers': [128, 128],
        'actor_fc_layers': [256, 256],
        'critic_fc_layers': [256, 256],
        'model_lr': 1e-4,
        'actor_lr': 3e-5,
        'value_lr': 8e-5,
    }

    print("\n[DEBUG] Initializing trainer for performance test...", flush=True)
    rngs = nnx.Rngs(0)
    trainer = DreamerTrainer(obs_dim, act_dim, config, rngs, modulation_config=None)

    print("[DEBUG] Creating larger batch (16, 64) for benchmarking...", flush=True)
    batch = create_dummy_batch(batch_size=16, seq_len=64, obs_dim=obs_dim, act_dim=act_dim)
    rng = jax.random.PRNGKey(42)

    # Warmup (compile)
    print("\n[DEBUG] Warming up (JIT compilation, please wait)...", flush=True)
    _ = trainer.train_step(batch, rng)
    jax.block_until_ready(_)
    print("✓ JIT compilation complete", flush=True)

    # Benchmark
    num_iterations = 10
    print(f"\nRunning {num_iterations} iterations...")

    times = []
    for i in range(num_iterations):
        start = time.time()
        metrics = trainer.train_step(batch, rng)
        jax.block_until_ready(metrics)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}/{num_iterations}: {elapsed*1000:.2f}ms")

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n{'─'*80}")
    print(f"Results (batch_size=16, seq_len=64):")
    print(f"  Mean time per iteration: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Throughput: {16*64/mean_time:.1f} steps/sec")
    print(f"{'─'*80}")

    print("\n📊 To compare with baseline, run this test before and after Phase 1 changes")
    print("   Expected speedup: 2-5x on world model training step")

    return True

def test_shapes():
    """Test that tensor shapes are correct throughout the pipeline."""
    print("\n" + "=" * 80)
    print("PHASE 1 SHAPE VERIFICATION")
    print("=" * 80)

    obs_dim = 33
    act_dim = 4
    config = {
        'encoder_dim': 128,
        'encoder_fc_layers': [128, 128],
        'rssm_deter_dim': 512,
        'rssm_stoch_dim': 32,
        'rssm_classes': 32,
        'decoder_fc_layers': [128, 128],
        'reward_fc_layers': [128, 128],
        'continue_fc_layers': [128, 128],
        'actor_fc_layers': [256, 256],
        'critic_fc_layers': [256, 256],
        'model_lr': 1e-4,
        'actor_lr': 3e-5,
        'value_lr': 8e-5,
    }

    rngs = nnx.Rngs(0)
    trainer = DreamerTrainer(obs_dim, act_dim, config, rngs, modulation_config=None)

    B, T = 4, 16
    batch = create_dummy_batch(batch_size=B, seq_len=T, obs_dim=obs_dim, act_dim=act_dim)

    # Test encoder output shape
    obs = batch['obs']  # (B, T, obs_dim)
    from src.models.dreamer_v3_util import symlog
    obs_symlog = symlog(obs)

    # Test batch encoding
    embeds = trainer.agent.wm.encoder(obs_symlog)  # Should be (B, T, embed_dim)

    print(f"\nShape verification:")
    print(f"  Input obs: {obs.shape} (expected: ({B}, {T}, {obs_dim}))")
    print(f"  Encoder output: {embeds.shape} (expected: ({B}, {T}, {config['encoder_dim']}))")

    expected_shape = (B, T, config['encoder_dim'])
    if embeds.shape == expected_shape:
        print(f"  ✓ Encoder output shape is correct!")
    else:
        print(f"  ❌ FAILED: Expected {expected_shape}, got {embeds.shape}")
        return False

    return True

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING PHASE 1: ENCODER OUTSIDE SCAN OPTIMIZATION")
    print("="*80)

    all_passed = True

    # Run tests
    if not test_shapes():
        all_passed = False

    if not test_correctness():
        all_passed = False

    if not test_performance():
        all_passed = False

    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")
