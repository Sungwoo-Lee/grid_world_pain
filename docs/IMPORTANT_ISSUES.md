# Important Issues & Lessons Learned

This document records critical performance regressions, architecture pivots, and technical "gotchas" encountered during development to prevent future regressions.

## 1. JAX Performance Regression (Phase 3 implementation)

### Issue Description
During the transition from Phase 2 to Phase 3 (mid-February 2026), the training speed dropped significantly from **~1.9s/iter** to **~4.0s/iter** (and even slower when all features were enabled).

### Root Causes
1. **Inefficient Visual Sensor (Main Bottleneck)**:
    - Initial implementation used `jax.lax.scan` and nested `jax.vmap` to iterate over every entity (resources, predators, obstacles) for every visual cell.
    - CPU-like sequential looping on a GPU is extremely slow in JAX.
2. **Entity Complexity**:
    - Increasing resource count from 7 to 24 multiplied the cost of sequential checks.
3. **Graph Bloat**:
    - Redundant `PRNGKey` splitting and non-vectorized interaction logic increased JIT compilation depth and execution time.

### Solution: Matmul Rendering & Unified Vectorization
- **Matmul Rendering**: Replaced all loops in `sensor.py` with a single high-performance `jnp.matmul` that matches targets against visual cells in one dense parallel operation.
- **Unified State**: Grouped all dynamic entities into single arrays for vectorized processing in `core.py`.
- **Boilerplate Reduction**: Minimized branching and redundant assembly in `get_observation`.

### Key Lesson
**NEVER use `lax.scan` or nested `vmap` for spatial/entity interaction checks if a matrix operation (`matmul`) or a single pooled `vmap` is possible.** JAX is optimized for dense linear algebra, not for processing entities one by one.

---

## 2. Terminology Consolidation
- **Update**: Replaced all instances of "Interoceptive Nociception" and "Health" with **"Injury"**.
- **Reason**: Simplifies user understanding and aligns with biological homeostasis literature where "injury" is the state being avoided.
