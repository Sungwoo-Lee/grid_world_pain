---
id: 20260519_1509_nnx_lax_scan_split_merge_pattern
date: 2026-05-19
time: "15:09"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, design, decision, meta]
summary: "To run a flax NNX Module inside jax.lax.scan without re-tracing on every call, split the module via nnx.split (graphdef in closure, state in carry), reconstruct via nnx.merge inside the body, and pass mutable storage (buffers etc.) as explicit JAX-array arguments rather than via the object. Generalisable beyond Dreamer."
related: ["20260519_1507_dreamer_srl_v2_cpu_buffer_regression", "20260519_1508_dreamer_jax_perf_retrofit_4_phases"]
session_origin: claude_code
session_label: "dreamer-srl v2 perf diagnosis + M-cell wakeup chain"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# NNX Module inside `jax.lax.scan`: the `nnx.split` / `nnx.merge` pattern

## Key conclusion
Putting a flax `nnx.Module` directly into a `jax.lax.scan` carry will JIT-retrace on every call because the module is a Python object with non-hashable Python state (mutable submodules, etc.). The working pattern, validated in the original-JAX-Dreamer trainer (`src/models/dreamer_v3_trainer.py:791-833`, commit `0dec9e0` + `6705735`), is to **split the module into `(graphdef, state)` via `nnx.split`, capture `graphdef` in the enclosing closure (it's hashable static metadata), carry only the `state` pytree through the scan, and reconstruct the module inside the scan body via `nnx.merge(graphdef, state)`**. A second rule: any mutable storage the scan reads (replay buffer arrays, etc.) must be passed as **explicit JAX-array arguments**, not via the Python object that owns them — otherwise the object's identity instability causes the same retracing problem. Together these two rules are what makes Phase 3's 596× speedup ([[20260519_1508_dreamer_jax_perf_retrofit_4_phases]]) actually compile cleanly under NNX.

## Evidence, measurements, facts
**Source pattern (working reference)**: `src/models/dreamer_v3_trainer.py:791-833` (`train_multiple_gpu`) and `:685-790` (`_scan_train_gpu`). The pattern, in compressible form:

```python
def train_multiple_gpu(self, buffer, num_steps, rng):
    # ---- OUTSIDE THE JIT BOUNDARY ----
    graphdef, _ = nnx.split(self)               # (1) split; graphdef is hashable
    buffer_arrays = (
        buffer.obs, buffer.actions,             # (2) extract arrays from Python object
        buffer.rewards, buffer.dones,
        buffer.is_first,
    )
    final_state, metrics, rng = self._scan_train_gpu(
        graphdef,                               # (3) hashable static → goes in closure
        int(num_steps), rng,
        buffer_arrays,                          # (4) explicit JAX-array tuple
        buffer.size, buffer.capacity,           # (5) Python ints (hashable, traceable)
        buffer.sequence_length,
        # ... other static args
    )
    nnx.update(self, final_state)              # (6) apply scanned state back to module
    return metrics, rng

@nnx.jit  # or just @jax.jit on _scan_train_gpu
def _scan_train_gpu(self, graphdef, num_steps, rng, main_arrays, ...):
    obs, actions, rewards, dones, is_first = main_arrays
    _, state = nnx.split(self)                 # (7) extract starting state

    def scan_body(carry, _):
        current_state, rng = carry
        trainer = nnx.merge(graphdef, current_state)   # (8) reconstruct module
        ...                                            # do the work
        metrics = trainer.train_step(batch, key)       # (9) call methods normally
        new_state = nnx.state(trainer)                 # (10) extract updated state
        return (new_state, rng), metrics

    final_carry, all_metrics = jax.lax.scan(
        scan_body, (state, rng), None, length=num_steps,    # (11) carry is pure pytree
    )
    return final_carry[0], jax.tree.map(jnp.mean, all_metrics), final_carry[1]
```

**Why each rule matters:**

1. **`nnx.split` separates graphdef (hashable static) from state (pytree of arrays).** A flax `nnx.Module` holds both. JIT can hash and cache on the graphdef but cannot trace on raw module references — they look "different" every Python call.
2. **`buffer_arrays` is an explicit tuple of `jnp.ndarray`.** The `buffer` Python object itself has identity that changes (or appears to change) each call from JIT's perspective; passing the arrays directly makes them traced values.
3. **`graphdef` is captured in the enclosing closure** (or passed as a static argument with `static_argnames`). Either works; closure is cleaner.
4. The explicit tuple form (vs `**kwargs`) prevents accidental dict-iteration order differences from triggering retraces.
5. **Python ints like `buffer.size`** are fine to pass — they're hashable and become trace constants. If they CHANGE between calls (e.g., growing as the buffer fills), they DO trigger retracing — solution: pass them as `jnp.array(buffer.size)` if dynamic, or accept the recompile cost if growth is monotonic and saturates.
6. **`nnx.update(self, final_state)`** writes the post-scan state back to the original module. Without this, the JIT-internal state changes are lost.
7. **`_, state = nnx.split(self)`** inside the JIT scope is fine because the `self` is the same module the outer code already split.
8. **`nnx.merge(graphdef, current_state)` inside scan body** reconstructs a working module instance from the static graphdef + the per-step state. This is cheap (no array copies; it's a Python-side reconstruction over pytree structure).
9. After merge, the module behaves normally — methods, attribute access, etc.
10. **`nnx.state(trainer)`** at end of scan body extracts the updated state pytree for the next iteration's carry.
11. **`(state, rng)` is a pure pytree of arrays** — exactly what `lax.scan` needs for its carry argument.

**The follow-up retrace bug (commit `6705735`):** before this pattern was settled, the `lax.scan` body would retrace each call because the buffer was passed as a Python object whose attribute access JIT couldn't trust. The fix was rule 2 — pull the arrays out explicitly. Doc message from the commit: *"Update GPU training loop to prevent retracing and optimize buffer sampling."*

**Anti-patterns this rule explicitly prevents:**
- ❌ Putting `nnx.Module` instance directly in `lax.scan` carry → retraces on every call.
- ❌ Reading `buffer.obs` inside the scan body via the Python object → retraces because the object's identity isn't stable.
- ❌ Calling `nnx.split` inside the scan body → infinite recursion or repeated work.
- ❌ Mutating `self` (the outer module) inside the scan body via `nnx.update` — should be done ONCE at the end, after `lax.scan` returns.

## Decisions and actions
- This pattern is the canonical answer for *any* "I need an NNX module inside a JAX-functional loop construct" question in this project. Applies to `lax.scan`, `lax.fori_loop`, `lax.while_loop`, and even `jax.vmap` of methods.
- **Upcoming application**: dreamer-srl v2 option L (full lax.scan training loop) per [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]] will need this pattern. The original Dreamer's `train_multiple_gpu` is a near-direct template; the mechanical changes are renaming buffer-field accessors and dropping the mixture-sampling branches (dreamer-srl uses sheeprl's uniform sampling, not the original Dreamer's three-pool mixture).
- **Add a tests/reviews flag**: if a future code review sees `lax.scan(..., (self_module, ...), ...)` in any agent's training loop, flag it as a retracing risk and apply this pattern.

## Open questions and follow-ups
- **Does `nnx.cached_partial` (newer NNX API) supersede the `split / merge` pattern?** Worth checking against current flax-nnx documentation. The split/merge pattern is what was working at the time of the original retrofit; later NNX versions may have added higher-level helpers. If yes, the rule above can be simplified — but until verified, the explicit pattern is still correct.
- **`nnx.jit` vs `jax.jit` on the wrapper function** — both work; minor differences in how they handle NNX state are documented in the flax-nnx repo but don't affect the pattern above.

## References
- See [[20260519_1508_dreamer_jax_perf_retrofit_4_phases]] for the 4-phase retrofit context (this pattern enabled Phase 3's 596× speedup).
- See [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]] for the dreamer-srl v2 perf regression that will be fixed with this pattern.
- Working reference in-repo: `src/models/dreamer_v3_trainer.py:791-833` (`train_multiple_gpu`) and `:685-790` (`_scan_train_gpu`).
- Commits: `6705735` (retracing fix that codified rule 2), `0dec9e0` (the broader 4-phase retrofit).
- Flax NNX official docs: `https://flax.readthedocs.io/en/latest/nnx/index.html` (general reference; specific scan/loop guidance evolves with the API).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260519_1509_nnx_lax_scan_split_merge_pattern.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]] (dreamer_diagnosis, 2026-05-19) — dreamer-srl v2's replay buffer is 100% numpy/CPU (faithful to sheeprl's PyTorch 
- [[20260519_1508_dreamer_jax_perf_retrofit_4_phases]] (dreamer_diagnosis, 2026-05-19) — On 2026-02-21, the original JAX Dreamer trainer was retrofitted in 4 measured ph
- [[20260521_0151_xla_scan_body_compile_dominates_module_count]] (dreamer_diagnosis, 2026-05-21) — Earlier claim — that dreamer-srl's 7-module decomposition CAUSED the 70× lax.sca
- [[20260529_1823_unified_animal_entity_v2_0_arch]] (env_entities, 2026-05-29) — v2.0 env merges predator + neutral animals into one unified entity class with a 
<!-- END BACKLINKS -->
