# DreamerV3 implementation reference — code-fidelity review

> Review target: `docs/project/concepts/dreamer_v3_implementation.md` (778 lines, sections §1–§8)
> Reviewer: code-reviewer (Phase 4 of pilot)
> Reviewed at HEAD (branch `v1.3`, commit `d623c5a`)

## Summary

- **Forward pass**: ~95 distinct file:line / behavior / hyperparameter claims checked. **3 mismatches found**, of which 1 is high-impact (free-nats axis-of-application) and 1 is medium-impact (`_scan_train_gpu` static-args list). Two minor off-by-one line numbers and one slightly imprecise `ValueError` site cite are listed as nits. The vast majority of citations (encoder/decoder line spans, RSSM step body, two-hot bin code, replay buffer code, mixture-sampling pools, EMA decay, optimiser eps split, reward MAE thresholds, `is_first` propagation, IMG_PROBE keys, hierarchical config flow) are accurate at HEAD.
- **Reverse pass**: Independent re-inventory walked `src/models/dreamer_v3_{nnx,trainer,util,network}.py`, `configs/models/dreamer_v3*.yaml`, `train.py:773-1665`, and `scripts/dreamer_offline_wm_test.py`. The doc's §3 component map is materially complete for the **active training path**. **5 components found in code that the doc does not mention**, all of them either dead code in active path (`Moments.normalize`, `OneHotDist.mode`, `DreamerV3Agent.__call__`, `DreamerV3Agent.initial_state`, `ReplayBuffer.sample_multiple`'s `NotImplementedError` for GPU mode), 1 instrumentation pattern (`jax.named_scope`), and 1 environmental-coupling detail (`is_first.ndim==3` branch in buffer write). **No fabrications** (zero items mentioned in doc that don't exist in code). Six DreamerV3 config variants (`_curriculum`, `_curriculum_probe`, `_probe`, `_probe_cont10`, `neuromodulated_dreamer_v3`) exist alongside the canonical config; the doc only cites `dreamer_v3.yaml` and `dreamer_v3_rr06.yaml`. This is acceptable scope for an algorithmic reference but should be noted.
- **Recommendation**: ACCEPT-WITH-FIXES. The free-nats finding (forward F1 below) is the only blocker — it changes both a behavior claim and a paper-deviation classification, and propagates between §3.5.4 and §3.7.5 of the doc. Everything else is line-number polish or scope expansion.

---

## Forward-pass findings (claim-checking)

### F1 — BLOCKER. Free-nats applied per-(B,T) state, NOT per stochastic group

**Doc claims** (§3.5.4 paragraph after the code excerpt at L260-269 of the doc, plus §3.7.5):
> "Free-nats is applied **per-stoch-group AFTER summing over the 32 discrete classes**, then clipped, then averaged across batch+time. So the clipping floor is 1.0 nat per `(B, T, stoch_dim=32)` entry — i.e. the lower bound of `dyn_kl + rep_kl` summed over groups would be `(0.5 + 0.1) * 32 = 19.2` nats per `(B, T)`."

> §3.7.5 "Code spec ... `dyn_kl/rep_kl` going into `max(., 1.0)` is shape `(B, T, stoch_dim)` after the per-class sum, so the floor is per stoch group."

> §3.5.4 deviation flag: "free-nats applied per-group rather than per-step. `MATCHES PAPER`."

**Code at `src/models/dreamer_v3_trainer.py:230–249`**:
- `kl_div_categ` returns `jnp.sum(... axis=-1)` over `discrete=32` (the LAST axis of `(B, T, stoch=32, discrete=32)` logits) → output shape `(B, T, stoch=32)`.
- L243-244: `dyn_kl = jnp.sum(dyn_kl, axis=-1)` and `rep_kl = jnp.sum(rep_kl, axis=-1)` then sum over **`stoch=32`** (now the last axis) → output shape `(B, T)`.
- L246-247: `jnp.maximum(dyn_kl, FREE_NATS)` clamps a `(B, T)` tensor — i.e., the floor is **1 nat per (B, T) entry**, NOT per stoch group.

**Consequences**:
1. The "lower bound" arithmetic is wrong. With per-(B,T) clamping and `FREE_NATS=1.0`, `0.5 * mean(max(dyn_kl, 1)) + 0.1 * mean(max(rep_kl, 1)) ≥ 0.5 + 0.1 = 0.6` nats (averaged), not 19.2.
2. The deviation classification flips. Hafner's paper (Eq. 4–5, official-implementation `dreamer/world_model.py`) applies free-bits per stochastic group → lower bound `1 nat × stoch_dim = 32` nats per state. Our code applies it per state, which is a **stricter** clamp (the regulariser is "alive" in fewer cases). This should be flagged at minimum `MINOR DEVIATION (justified-or-not)` or `MAJOR DEVIATION (deliberate)` depending on intent — not `MATCHES PAPER`.
3. The inline source comment at `trainer.py:239` ("# (B, T, stoch, discrete) -> (B, T, stoch)") describes only the kl_div_categ output before line 243-244 collapses the `stoch` axis. The doc appears to have stopped reading at the comment and missed the second `jnp.sum(..., axis=-1)` two lines later.

**Suggested fix**: rewrite §3.5.4 description ("Free-nats is applied per-state after summing over both discrete classes and stochastic groups; floor is 1 nat per (B,T) entry, lower bound on the loss term `0.5 + 0.1 = 0.6` averaged"), correct §3.7.5 shape claim to `(B, T)`, and reclassify the deviation flag from `MATCHES PAPER` to e.g. `MAJOR DEVIATION (intent unclear — stricter than paper)`. Also propagate the change to §6 — this currently appears nowhere in the §6 ranked list, which it should once reclassified.

---

### F2 — CONCERN. `_scan_train_gpu` static-args list misstates which args are static

**Doc claims** (§3.8.3): "Static args: `(graphdef, num_steps, b_size, b_cap, b_seq_len, pos_size, pos_cap, pos_slots, recent_slots, recent_window, buf_idx)`."

**Code at `src/models/dreamer_v3_trainer.py:677`**: `@nnx.jit(static_argnums=(1, 2, 7, 8, 10, 11, 12, 13))`

Counting the function signature (`self`=0, `graphdef`=1, `num_steps`=2, `rng`=3, `main_arrays`=4, `pos_arrays`=5, `b_size`=6, `b_cap`=7, `b_seq_len`=8, `pos_size`=9, `pos_cap`=10, `pos_slots`=11, `recent_slots`=12, `recent_window`=13, `buf_idx`=14), the actual static args are:

`graphdef, num_steps, b_cap, b_seq_len, pos_cap, pos_slots, recent_slots, recent_window`.

The doc's list incorrectly includes **`b_size` (6), `pos_size` (9), and `buf_idx` (14)** as static — they are traced. This matters because: `b_size`, `pos_size`, and `buf_idx` change every iteration as the buffers fill, which is precisely why they cannot be static (would force constant retracing). Stating them as static gives a misleading picture of the JIT's stability profile.

**Suggested fix**: §3.8.3 — "Static args: `(graphdef, num_steps, b_cap, b_seq_len, pos_cap, pos_slots, recent_slots, recent_window)`. Traced (per-call): `(rng, main_arrays, pos_arrays, b_size, pos_size, buf_idx)`."

---

### F3 — NIT. ValueError site for missing `encoding_mode`

**Doc claims** (§3.2): "Selected via the mandatory key `agent.encoding_mode` ∈ `{"hierarchical", "flat"}`; missing the key raises `ValueError` (`nnx.py:204–205`)."

**Code at `src/models/dreamer_v3_nnx.py:204–205`**:
```python
if config is None:
    raise ValueError("Strict Config: Hierarchical config is required for DreamerObservationEncoder.")
```
The `ValueError` at L204-205 fires when `config is None`, not when the `encoding_mode` key is missing. The actual `encoding_mode` read at L207 (`self.mode = config['encoding_mode']`) raises `KeyError` if missing. The mandatory-key check that produces a `ValueError` for a missing `agent.encoding_mode` lives in the trainer at `trainer.py:78` (`config.get_mandatory('agent.encoding_mode', str)`).

The doc's claim is true in spirit (a missing config does raise `ValueError`) but the cited line is the wrong site. Recommend: cite `trainer.py:78` instead, or expand the description to acknowledge two failure modes.

---

### F4 — NIT. Off-by-one line ranges

Two minor citations point to the line ABOVE the actual content:
- §3.5.5 "Caller (`trainer.py:413–416`)" — span begins at L413 (`all_vals = ...`), L414-415 are blank, L416 is `lambda_returns = ...`. Acceptable; reads as a 4-line span with whitespace.
- §3.3 decoder code excerpt cites `nnx.py:432–441` — actual `for h in fc_layers:` loop body is L433–441; L432 is the docstring close. Excerpt content matches; line range starts one line early.

Neither is load-bearing; recording for completeness.

---

## Reverse-pass findings (completeness — load-bearing)

Independent re-inventory of `src/models/dreamer_v3_*.py`, `configs/models/dreamer_v3*.yaml`, `scripts/dreamer_offline_wm_test.py`, and the `train.py` DreamerV3 branch (L773-1665). For each component found in code, I verified whether the doc covers it.

### Components in code but missing from doc

#### R1 — `Moments.normalize()` method is dead code

`util.py:154–159` defines `Moments.normalize(x)` returning `(x - low) / max(1/max_, high - low)`. **Never called anywhere** in `src/`, `train.py`, or `scripts/`. The actual normalisation in the actor loss at `trainer.py:418, 434` is done inline: `(x - moments_low) / moments_invscale`. The doc lists `Moments` as a numerical-stability trick (§3.7.4) but does not flag the dead method. Recommend: add to §6 "Mid-tier" deviations alongside §6 #9 (`KL_SCALE`) and §6 #11 (`agent.train_steps`) as another stranded refactor artefact.

#### R2 — `OneHotDist.mode()` method is dead code in the active training path

`util.py:108–110` defines `mode()` returning `argmax`-based one-hot. **Never called** in `src/`, `train.py`, or `scripts/`. The doc mentions `OneHotDist` extensively in §3.7.3 but doesn't flag this. Lower priority than R1 because it's a legitimate API method that an offline diagnostic might use; still, current-state is dead.

#### R3 — `DreamerV3Agent.__call__` is dead code

`nnx.py:614–668` defines `DreamerV3Agent.__call__(self, x, h, key)` doing one inference step (encode → RSSM step → actor → critic → returns logits/value/h_new/mod_info). **Never called** anywhere — the actual inference path used by the trainer at training time is `DreamerTrainer.get_action` (`trainer.py:509-582`), and at evaluation time it is also `get_action` plus `collect_sequence`. The 55-line `__call__` is unused. Doc doesn't flag this. Recommend: §6 mid-tier item, similar in spirit to "`dreamer_v3_network.py` is unused legacy file" (§6 #14).

#### R4 — `DreamerV3Agent.initial_state` is dead code

`nnx.py:602–612`. Trainer code constructs initial state manually (`trainer.py:591-601` and `train.py:821-829, 1146-1151`), bypassing this helper. Same severity as R3.

#### R5 — `ReplayBuffer.sample_multiple` raises NotImplementedError on GPU

`trainer.py:1008–1017`. The method exists for the CPU path but raises `NotImplementedError` for the GPU buffer device. The active GPU path uses `_scan_train_gpu`'s in-JIT mixture sampling instead. The doc does mention `sample_multiple` once (§3.8.4 "pre-samples `num_batches` batches with `_sample_mixture_cpu` or `buffer.sample_multiple`") but doesn't flag the GPU constraint. Minor — would prevent confusion if a reader were to try the GPU path through this method.

#### R6 — `jax.named_scope(...)` instrumentation throughout the trainer

13 named scopes are scattered through `trainer.py` (`wm_encoder`, `wm_rssm_scan`, `wm_losses`, `dreamer_optim` ×2, `ac_imagine_scan`, `ac_losses`, `dreamer_sense`, `dreamer_act`, `dreamer_env_step`, `dreamer_env_reset`, `replay_mixture_sample`, `replay_concat`). They are JAX trace annotations used for profiling. **The doc says nothing about this**. Probably out of scope for an algorithmic reference, but a one-line mention in §4 would make `dreamer_buffer_add` / `dreamer_positive_buffer_copy` / `dreamer_train_multiple` (in `train.py:1385, 1397, 1614`) — visible in chrome:// traces — discoverable.

#### R7 — Six DreamerV3 config variants exist; doc cites only two

Files in `configs/models/`: `dreamer_v3.yaml`, `dreamer_v3_rr06.yaml`, `dreamer_v3_curriculum.yaml`, `dreamer_v3_curriculum_probe.yaml`, `dreamer_v3_probe.yaml`, `dreamer_v3_probe_cont10.yaml`, `neuromodulated_dreamer_v3.yaml` (7 total). Doc cites only the canonical and `_rr06.yaml` (§5.3) plus an indirect mention of `neuromodulated_dreamer_v3.yaml` in the YAML comment quoted at §3.4 / §4.4. The other four variants exist for active experiments and may diverge from canonical on additional knobs. Recommend: `dreamer_v3_curriculum.yaml` and `dreamer_v3_probe.yaml` get one-line summaries in §5.3 alongside `_rr06.yaml`, since they are part of the active experimental branch.

#### R8 — `is_first` arrives shape-polymorphic to the buffer write

`train.py:1390-1394` (GPU path) and `train.py:1434-1438` (CPU path) carry an `if is_first_arr.ndim == 3:` branch that handles both `(T, B, 1)` and `(T, B)` shapes. The doc's §3.6.3 description of `is_first` flow does not mention this polymorphism. Minor — the trainer guarantees `(B, 1)` per-step shape (`trainer.py:642, 597, 650`), so the `ndim==3` branch is the canonical path. Worth one sentence for completeness.

#### R9 — `FiLMNoNorm` modulation type is explicitly removed with a hard error

`nnx.py:495-499`: `if self.modulation_enabled and self.modulation_type == "FiLMNoNorm": raise ValueError(...)`. The doc lists allowed `modulation.type` values as `null / "FiLM" / "PreActivation" / "Multiplicative"` (§4.7) but doesn't flag the explicit-rejection path for `"FiLMNoNorm"`. Out of scope for algorithmic correctness; flag here for completeness.

### Components mentioned in doc but not found in code

**None.** Every file:line / function name / behavior claim verified except those flagged in F1-F4. No fabrications detected.

---

## Spot-checks on known sharp edges (from review brief)

| Item | Status | Notes |
|---|---|---|
| `dreamer_v3_network.py` flagged as unused | OK | Doc §3.0 introduction says "**NOT imported** anywhere — confirmed orphaned by `grep -rln`" + §6 #14. Independent grep at HEAD reproduces zero references except `egg-info/SOURCES.txt` and the doc itself. |
| `KL_SCALE = 1.0` flagged as dead | OK | `grep -rn KL_SCALE` returns only the declaration at `trainer.py:18`. Doc §3.5.4 deviation flag and §6 #9 both correct. |
| Mixture sampling pools (5 / 5 / 6 of 16, threshold `> 0.0`, capacity 100k) | OK | All numbers verified at HEAD. `dreamer_v3.yaml:18-22`. Threshold `> 0.0` literal at `train.py:1411` (GPU) and `train.py:1454` (CPU). Implicit uniform_slots = 16 - 5 - 5 = 6 (`trainer.py:704`). Positive-buffer reward gate uses `bool(jnp.any(...))` host-side, **outside** the JIT — confirmed at `train.py:1397-1463`. |
| Adam eps asymmetry 1e-8 (WM) / 1e-5 (actor, critic) | OK | `trainer.py:96` (`eps=1e-8`), `trainer.py:104` (`eps=1e-5`), `trainer.py:112` (`eps=1e-5`). Doc §3.8.1 correct. |
| `imagined_rollout_probe` 7 metrics | OK | All 7 key strings match exactly: `imagined_termination_fraction_h8`, `imagined_termination_fraction_h15`, `imagined_first_term_step_mean`, `imagined_term_step_p10`, `imagined_term_step_p50`, `imagined_term_step_p90` (behavior side, `trainer.py:478-484`) + `imagined_real_term_step_mean` (WM side, `trainer.py:294`). |
| `Ratio(...)` math at `dreamer_v3_util.py:178-191` and use at `train.py:1620` | OK | Class spans L162-192 (doc cites L162-192 for class, L178-191 in user's brief for `__call__`); both correct. Caller `train_steps = ratio_scaled_updates(global_step // num_steps)` at `train.py:1620` confirmed. Cold-start gate at `train.py:1615` correctly cited. |
| `> 0.0` reward threshold for positive-buffer admission | OK | Doc cites `train.py:1411, 1454` — both confirmed (one for GPU path with `jnp.any`, one for CPU path with `np.any`). Strict-greater (positive-only; zero rewards excluded). |
| Slow / target critic EMA factor (0.98 / 0.02) | OK | `trainer.py:504`: `0.98 * t + 0.02 * c`. No periodic reset; doc §3.4.5 correct that only buffer (not target critic) is cleared at stage transitions (`train.py:1116-1136`). |
| Deviations §6 (25 items) — every code citation exists | OK with one nuance | All 25 items have valid file:line citations. The nuance: §6 item #8 ("WM grad-clip 1000.0") is correctly cited at `trainer.py:95` and §6 item #16 (`is_first` reset both fields) at `nnx.py:81-83`; both verified. **F1 (free-nats per-state) is NOT in the §6 list** because the doc currently classifies it as `MATCHES PAPER` — once reclassified, it should be added to §6. |

---

## Verdict

- **Forward-pass mismatches resolved?** **FAIL** until F1 (free-nats axis) is fixed. F2 (`_scan_train_gpu` static args) is medium impact; F3-F4 are nits. Most claims pass.
- **Code components covered in doc?** **PASS for the active training path** (RSSM, encoder, decoder, heads, losses, buffer, mixture sampling, optimisers, target critic, λ-returns, free-nats, Moments, IMG_PROBE — all present). Misses listed in §R1-R9 are mostly dead-code-in-active-path or instrumentation, not algorithmic functionality. Recommend adding `Moments.normalize`, `DreamerV3Agent.__call__`, `DreamerV3Agent.initial_state`, and `OneHotDist.mode` to §6 "Mid-tier" alongside the existing `KL_SCALE` / `train_steps` / `unimix-yaml-key` dead-code triplet.
- **Recommendation: ACCEPT-WITH-FIXES**. Fix F1 (blocking), F2 (high-priority), R1/R3/R4 (dead-code completeness for §6). The other reverse-pass items are scope expansion the author can defer or merge into a future revision.

---

## Files audited

Forward-pass and reverse-pass walked the following files at HEAD (`v1.3` branch, top-of-tree commit `d623c5a`):

- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/src/models/dreamer_v3_nnx.py` (670 lines)
- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/src/models/dreamer_v3_trainer.py` (1017 lines)
- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/src/models/dreamer_v3_util.py` (205 lines)
- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/src/models/dreamer_v3_network.py` (80 lines, confirmed orphan)
- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/models/dreamer_v3.yaml` (78 lines)
- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/models/dreamer_v3_rr06.yaml` (referenced)
- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/models/dreamer_v3_curriculum.yaml`, `dreamer_v3_curriculum_probe.yaml`, `dreamer_v3_probe.yaml`, `dreamer_v3_probe_cont10.yaml`, `neuromodulated_dreamer_v3.yaml` (existence confirmed; not deeply audited)
- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/train.py` — DreamerV3 branch (L440-505 dispatch, L773-832 setup, L1100-1160 stage transition, L1380-1465 buffer write + positive-buffer copy, L1605-1665 train + WandB log)
- `/media/nas01/projects/Interoceptive-AI/grid_world_pain/scripts/dreamer_offline_wm_test.py` (only L250-279 spot-checked for the doc's two cited line refs)

Cross-checks executed:
- `grep -rln "DreamerV3\|dreamer_v3\|from_twohot\|to_twohot\|symlog\|symexp\|FREE_NATS\|HORIZON" src/` (verified no other files touch these symbols outside the four `dreamer_v3_*.py` files plus `evaluation_core.py` for the public API).
- `grep -rln "dreamer_v3_network"` (confirmed orphan claim — only `egg-info/SOURCES.txt` and the doc reference it).
- `grep -rln "ModulatedLayerNormGRUCell"` (confirmed used inside `dreamer_v3_nnx.py:54` only when `modulation_enabled`).
- `grep -rn "unimix"` (confirmed YAML key never threaded — only `OneHotDist.__init__` constructor default at `util.py:83`).
- `grep -rn "KL_SCALE"` (confirmed dead — only declaration at `trainer.py:18`).
- `grep -rn "agent.train_steps\|train_steps"` (confirmed YAML key never read; only the local variable `train_steps = ratio_scaled_updates(...)` at `train.py:1620` is used).
- `grep -rn "Moments.normalize"` (confirmed `Moments.normalize` is never called).
- `grep -rn "DreamerV3Agent.__call__"` (confirmed unused).

---

Reviewed by: code-reviewer
