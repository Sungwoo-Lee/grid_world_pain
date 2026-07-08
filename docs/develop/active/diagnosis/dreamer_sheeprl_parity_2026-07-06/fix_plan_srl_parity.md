---
title: "WP-SRL — dreamer_srl parity fix package (P1–P3, P5–P8 fixed; P9 declared)"
topic: diagnosis
status: active
created: 2026-07-08
last_updated: 2026-07-08
---

# WP-SRL — dreamer_srl parity fix package

> **Status**: PLANNED (user-approved "fix all" — goes straight to `developer`)
> **Opened**: 2026-07-08
> **Related**: [[00_master_comparison]] §3 (source of record) · [[02_world_model_losses]] · [[04_training_loop_replay]] · [[DEVIATION_LOG]] · [[KNOWN_BUGS]]

---

## Context

The 2026-07-06→08 re-audit of our JAX DreamerV3 port (`dreamer_srl`) against the vendored
PyTorch reference (sheeprl, pinned at commit `33b6366`) found that the port's core math is at
numerical parity, but its *glue* — optimizer wrapping, the replay-buffer write path, loss
assembly, and loop bookkeeping — silently diverges from the reference in eight ways that affect
training. This plan fixes seven of them and formally declares the remaining two as accepted
deviations. In plain terms: the port currently (1) applies raw, unclipped gradients where the
reference clips them (its world-model loss has been observed exploding to ~1e29 in live smoke
runs); (2) corrupts the replay buffer whenever more than one environment runs in parallel —
every episode end punches a garbage row into every *other* environment's stored history;
(3) weights observation reconstruction at exactly half the reference recipe; (4) over-counts
every logged episode by one survival step (the project's headline metric) plus the previous
episode's terminal reward; (5) runs a prefill phase `num_envs`× longer than intended; and
(6)–(7) has two smaller cadence/gating drifts. Full evidence, probes, and file:line citations
live in the master comparison ([[00_master_comparison]] §3, items P1–P9) and area reports
[[02_world_model_losses]] and [[04_training_loop_replay]]. The user approved fixing the whole
package at once; this doc is the single source of truth for the `developer` agent.

**Comparability caveat (read before launching anything):** this package, together with the
parallel discount-factor config correction (WP-GAMMA, owned by `experiment-designer` — the γ
typo P4), forms **one dreamer_srl comparability epoch**. Any run trained after these fixes land
is **not comparable** to any run trained before them (same situation as the earlier H4–H7 fix
cluster). Do not mix pre/post-fix runs in a single analysis.

**Scope fence:**
- Touch ONLY `src/algorithms/dreamer_srl/*`, `tests/algorithms/dreamer_srl/*`, and the three
  enumerated `DEVIATION_LOG.md` row edits (§ DEVIATION_LOG deliverable below).
- **NO `configs/` edits.** The γ value is WP-GAMMA's; the clip norms are deliberately hardcoded
  recipe constants (see P1 design note).
- **Do NOT touch `src/models/`** — the DreamerV3-NNX stack is the parallel WP-NNX package.

---

## Analysis

Summary of the eight items (evidence in the cited reports; not re-derived here):

| # | What is wrong | Ours | sheeprl reference | Report |
|---|---|---|---|---|
| P1 | No gradient clipping on any of the 3 optimizers; WM loss empirically spikes to ~1e29–1e31 | `dreamer_srl_main.py:658-660` plain `optax.adam` | `dreamer_v3.py:193-197` (WM, norm 1000), `:300-302` (actor, 100), `:320-324` (critic, 100); `dreamer_v3.yaml:52/127/154` | [[04_training_loop_replay]] D-01 |
| P2 | One shared buffer write-head across envs: every partial-done reset write advances the shared `_pos` and leaves a hole/garbage row in every non-done env's column (probe: sampled env-1 sequence `[21, 31, 0, 41]`); also holds `num_envs`× the reference capacity (missing `// num_envs`) | `SequentialReplayBuffer` shared `_pos`/`_full` (`buffers.py:297-299`, done-mask path `:224-230`); constructed `dreamer_srl_main.py:667-672` | `EnvIndependentReplayBuffer` (`vendor/sheeprl/sheeprl/data/buffers.py:529-699`): n_envs independent single-env buffers, each with its own `_pos`, each sized `cfg.buffer.size // num_envs` (`dreamer_v3.py:478-485`); reset write routed only to done envs (`dreamer_v3.py:650`, `buffers.py:645-654`); sample allocates batch across sub-buffers via `bincount` (`buffers.py:683-699`) | [[04_training_loop_replay]] D-03 |
| P3 | Inline obs loss: extra `0.5` (recon under-weighted exactly 2×, measured ratio 0.500000) + decoder trained in real space (extra symlog at loss time) + missing `tol=1e-8` clamp. The faithful `reconstruction_loss` port is imported and never called | `train.py:704-712` (inline), `train.py:158` (dead import) | `SymlogDistribution` (`vendor/sheeprl/sheeprl/utils/distribution.py:152-193`), loss assembly `loss.py:9-88`, `dreamer_v3.py:156-161` | [[02_world_model_losses]] rows 6–8, 22 |
| P5 | Episode counters re-increment after the done reset: every logged episode is +1 survival step and carries the predecessor's terminal reward | log `dreamer_srl_main.py:1213-1214`, reset `:1291-1292`, unconditional increment `:1526-1527` | sheeprl uses the gym wrapper's `final_info["episode"]` (`dreamer_v3.py:610-618`) — no double-count | [[04_training_loop_replay]] D-06 |
| P6 | `learning_starts` counted in iterations, not env steps → prefill is `num_envs`× too long | `dreamer_srl_main.py:484` raw config value; gates at `:1131`/`:1532`; `ratio_steps` at `:1540` | `dreamer_v3.py:508-511`: `learning_starts = cfg // policy_steps_per_iter`; `prefill_steps = learning_starts - int(learning_starts > 0)`; `:661`: `ratio_steps = policy_step - prefill_steps * policy_steps_per_iter` | [[04_training_loop_replay]] D-04, S-01 |
| P7 | `_grad_step_remainder` is dead code; `_G = max(1, int(replay_ratio * num_envs))` silently quantizes fractional ratios (e.g. 0.5 → 1.0 at num_envs=1) | `dreamer_srl_main.py:1064-1065`, `:1549-1556` | `dreamer_v3.py:662` exact un-quantized `Ratio` return | [[04_training_loop_replay]] D-05 |
| P8 | Train gate checks `buffer._pos >= seq_len` only; after a ring wrap `_pos` cycles low and up to seq_len−1 owed grad steps are skipped | `dreamer_srl_main.py:1558` | no such gate (prefill guarantees data; per-env `_pos` never presents below-threshold mid-run) | [[04_training_loop_replay]] D-07 |
| P9 | RSSM transition/representation output-linear init uses truncated normal; reference uses Hafner uniform of the same std. Init-time only — **DECLARE, do not fix** | `agent.py:703-714` (`init_weights` on `transition_out`/`repr_out`) | `agent.py:1173-1174` (`uniform_init_weights(1.0)`) | [[00_master_comparison]] §3 P9 |

(P4, the γ config typo, is deliberately **absent** — it is WP-GAMMA.)

---

## Implementation Plan

### Design

#### Sequencing (single file `dreamer_srl_main.py` hosts P1/P5/P6/P7/P8 — do the steps in this order)

Each step = code + its regression test(s) + its DEVIATION_LOG row (if any) in **one commit**.
Steps 1–2 are independent of the loop; steps 3–5 edit the loop top-down; steps 6–7 (buffer)
are the largest and go last so the loop is already settled when the gate expression changes.

| Step | Fix | Files | DEVIATION_LOG row in same commit |
|---|---|---|---|
| 1 | P1 clipping | `utils.py`, `dreamer_srl_main.py:658-660` | — |
| 2 | P3 obs loss | `loss.py`, `train.py` | — |
| 3 | P5 episode counters | `dreamer_srl_main.py` (~:1526) | — |
| 4 | P6 learning_starts | `utils.py`, `dreamer_srl_main.py` (:484 region, :1117-1131 comments, :1540) | **D-014 refresh** |
| 5 | P7 delete remainder | `dreamer_srl_main.py:1048-1065`, `:1549-1556` | **D-015 (new)** |
| 6 | P2 per-env buffer | `buffers.py`, `dreamer_srl_main.py:667-672`, `:1369` | — |
| 7 | P8 train gate | `dreamer_srl_main.py:1558` (depends on step 6's `ready_to_sample`) | — |
| 8 | P9 declaration only | — | **D-016 (new)** |

#### P2 mechanism decision — structural port of `EnvIndependentReplayBuffer` (option B)

Two candidate mechanisms were considered:

- **(A) Per-env write-heads inside `SequentialReplayBuffer`** — `_pos`/`_full` become
  `[n_envs]` arrays on the single `[size, n_envs, ...]` storage. Smaller diff, but the
  `sample()` valid-start-index exclusion (`[_pos−seq_len, _pos)`) becomes per-env and needs
  new, hand-derived index math on both CPU and GPU paths — new code with no reference to
  diff against.
- **(B) A wrapper class holding `n_envs` independent single-env `SequentialReplayBuffer`s** —
  a line-for-line structural port of sheeprl's `EnvIndependentReplayBuffer`
  (`vendor/sheeprl/sheeprl/data/buffers.py:529-699`), composing the already-Lever-A-certified
  single-env buffer exactly the way the reference composes its own.

**Decision: (B).** Rationale: the goal is *parity*, and (B) is verifiable by structural
diff against the reference rather than by fresh derivation; every sub-buffer code path
(add ring-wrap, valid-idx exclusion, `_get_samples` tiling) is already parity-certified at
`n_envs=1` (area report 4, P-04/P-05). The per-add Python overhead (≤16 tiny numpy column
writes per iteration) is negligible next to the env step and train step.

**Pinned semantics being matched** (each becomes a regression-test assertion):

1. **Sizing**: each sub-buffer holds `cfg.buffer.size // num_envs` transitions
   (`dreamer_v3.py:478`); division happens **in the driver**, mirroring sheeprl.
2. **Regular add** (all envs): env column `e` of `data` goes to sub-buffer `e`; every
   sub-buffer's own `_pos` advances by 1 (`buffers.py:645-654`).
3. **Reset write at done boundaries**: routed **only** to done envs' sub-buffers
   (`dreamer_v3.py:650` — `rb.add(reset_data, dones_idxes)`); non-done envs' heads do
   **not** move and their sequences stay contiguous — **no hole rows**.
4. **Sample**: batch allocated across sub-buffers via
   `np.bincount(rng.integers(0, n_envs, (batch_size,)))`, per-sub-buffer sample, concat
   along the batch axis (axis 2 of `[n_samples, seq_len, batch_size, ...]`)
   (`buffers.py:683-699`). Per-env valid-index exclusion comes for free from each
   sub-buffer's own `_pos`.
5. **RNG**: unseeded `np.random.default_rng()` for the bincount (matches sheeprl; parity
   row P-14's reproducibility caveat carries over unchanged).

**GPU-buffer interaction**: the opt-in `--buffer-device gpu` path (Option M) samples with
traced jnp gathers; bincount allocation would produce variable per-sub-buffer shapes →
recompile storm. Therefore: **GPU buffer mode remains restricted to `num_envs == 1`**
(explicit `ValueError` at construction), where the plain single-env `SequentialReplayBuffer`
is kept and is semantically identical to a wrapper-of-one. Multi-env GPU buffering is a
declared non-goal of this package (note it in the Implementation Report; do not build it).

#### P7 decision — DELETE the remainder; declare the quantization (D-015)

Wiring `_grad_step_remainder` back is **not** trivially safe: consuming the remainder means
occasionally running `_G ± 1` scan steps, i.e. variable scan lengths — exactly the
per-length XLA recompilation storm that Fix 2 (constant scan length) was built to prevent.
Repaying the remainder in whole extra *iterations* would need a second compiled executable
and new scheduling logic — real complexity for a path no canonical config exercises
(all live parity configs run `replay_ratio=1`, where `_G` is exact). **Decision: delete the
dead accumulator, keep `_G`, and declare the quantization as DEVIATION_LOG row D-015**
(exact text below). The effective scan-path replay ratio becomes an explicit, documented
`max(1, int(replay_ratio * num_envs)) / num_envs`; the `--legacy-grad-loop` path keeps the
exact `Ratio` return for anyone who needs fractional ratios (e.g. the replay-ratio sweep
work must either use the legacy path or pick `replay_ratio * num_envs ∈ ℕ`).

#### P1 design note — clip norms are hardcoded recipe constants, not config keys

Scope fence forbids `configs/` edits, so the three norms (1000 / 100 / 100) are module-level
constants in `utils.py` with sheeprl YAML citations. This does not violate the no-fallback-
defaults rule: these are not config reads with defaults — they are pinned reference-recipe
constants, exactly as fixed as the two-hot bin count. If a future experiment needs them
tunable, that is a separate plan (config keys via `get_mandatory` + YAML additions).
Substrate note: torch's `clip_grad_norm_` uses `clip_coef = max_norm / (total_norm + 1e-6)`;
`optax.clip_by_global_norm` omits the `1e-6`. Relative effect < 1e-6 at the clip boundary —
substrate-class (same family as D-003/D-007), no log row needed.

#### P3 design note — decoder representation semantics change

After the fix the decoder's raw output **is** the symlog-space prediction (reference
semantics); training no longer squashes it a second time. `grep` confirms
`reconstructed_obs` is consumed **only** by the WM loss (`train.py`) — no eval or
visualization path reads it — so no `symexp`-at-consumption site needs adding. Any future
real-space consumer must go through `SymlogDistribution.mode`/`.mean` (which apply `symexp`).
The inline loss in `train.py:700-751` is replaced wholesale by a call to the already-imported
faithful `reconstruction_loss` (`loss.py:424-582`), which structurally eliminates the
inline-copy divergence class ([[02_world_model_losses]] Detail D). Expected metric shift:
the logged observation-loss roughly doubles and the total WM loss rises accordingly — this
is the fix working, not a regression.

### File Changes

#### 1. `src/algorithms/dreamer_srl/utils.py` — P1 constants + optimizer factory, P6 helper

Add (with `import optax` at the top if absent):

```python
# --- WP-SRL P1: gradient-clip recipe constants -------------------------------
# Pinned to sheeprl@33b6366:sheeprl/configs/algo/dreamer_v3.yaml
#   :52  algo.world_model.clip_gradients: 1000.0
#   :127 algo.actor.clip_gradients:       100.0
#   :154 algo.critic.clip_gradients:      100.0
# Applied in sheeprl at dreamer_v3.py:193-197 (WM), :300-302 (actor), :320-324
# (critic) via fabric.clip_gradients(..., error_if_nonfinite=False).
WM_CLIP_NORM: float = 1000.0
ACTOR_CLIP_NORM: float = 100.0
CRITIC_CLIP_NORM: float = 100.0


def make_optim_tx(lr: float, eps: float, clip_norm: float) -> optax.GradientTransformation:
    """clip-by-global-norm → Adam, matching sheeprl's clip-then-step order."""
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(lr, eps=eps),
    )


# --- WP-SRL P6: learning_starts is an ENV-STEP count in the config ----------
# Ported from sheeprl@33b6366:dreamer_v3.py:508-511 (world_size == 1):
#   policy_steps_per_iter = num_envs
#   learning_starts = cfg.algo.learning_starts // policy_steps_per_iter
#   prefill_steps   = learning_starts - int(learning_starts > 0)
def derive_prefill(learning_starts_cfg: int, num_envs: int) -> Tuple[int, int]:
    """Return (learning_starts_iters, prefill_steps) from the config env-step value."""
    learning_starts = learning_starts_cfg // num_envs
    prefill_steps = learning_starts - int(learning_starts > 0)
    return learning_starts, prefill_steps
```

#### 2. `src/algorithms/dreamer_srl/dreamer_srl_main.py:658-660` — P1

```python
# BEFORE:
wm_opt = nnx.Optimizer(world_model, optax.adam(wm_lr, eps=wm_eps), wrt=nnx.Param)
actor_opt = nnx.Optimizer(actor, optax.adam(actor_lr, eps=actor_eps), wrt=nnx.Param)
critic_opt = nnx.Optimizer(critic, optax.adam(critic_lr, eps=critic_eps), wrt=nnx.Param)

# AFTER (import make_optim_tx + the three norms from .utils):
wm_opt = nnx.Optimizer(world_model, make_optim_tx(wm_lr, wm_eps, WM_CLIP_NORM), wrt=nnx.Param)
actor_opt = nnx.Optimizer(actor, make_optim_tx(actor_lr, actor_eps, ACTOR_CLIP_NORM), wrt=nnx.Param)
critic_opt = nnx.Optimizer(critic, make_optim_tx(critic_lr, critic_eps, CRITIC_CLIP_NORM), wrt=nnx.Param)
```

#### 3. `src/algorithms/dreamer_srl/loss.py` — P3: add `SymlogDistribution`

Port of `vendor/sheeprl/sheeprl/utils/distribution.py:152-193` (only the `dist="mse"`,
`agg="sum"` branch the DreamerV3 recipe uses), placed next to `TwoHotEncoding`:

```python
class SymlogDistribution:
    """MSE-in-symlog-space observation distribution.

    Ported from sheeprl@33b6366:sheeprl/utils/distribution.py:L152-L193.
    The raw decoder output IS the symlog-space prediction (`_mode`); `log_prob`
    compares it to `symlog(value)`; real-space reconstructions only exist at
    `mode`/`mean` consumption via `symexp` (distribution.py:170-175).
    Includes the reference's small-error tolerance: squared distances below
    `tol=1e-8` are zeroed (distribution.py:159, 181) — closes audit row 8.
    """

    def __init__(self, mode: jax.Array, dims: int, tol: float = 1e-8):
        self._mode = mode
        self._dims = tuple(-x for x in range(1, dims + 1))
        self._tol = tol

    @property
    def mode(self) -> jax.Array:
        return symexp(self._mode)

    @property
    def mean(self) -> jax.Array:
        return symexp(self._mode)

    def log_prob(self, value: jax.Array) -> jax.Array:
        distance = (self._mode - symlog(value)) ** 2
        distance = jnp.where(distance < self._tol, 0.0, distance)
        return -distance.sum(self._dims)
```

(`symlog`/`symexp` live in `utils.py:27-46`; import them here.)

#### 4. `src/algorithms/dreamer_srl/train.py:700-751` — P3: route through `reconstruction_loss`

Delete the divergent inline assembly (the `-0.5 * jnp.sum((_symlog(...) - _symlog(...))**2)`
expression at `:709-711` and the hand-rolled KL/reward/continue/total at `:714-751`) and call
the faithful, already-imported `reconstruction_loss`:

```python
# AFTER (inside wm_loss_fn, replacing lines ~704-751):
# Distributions — sheeprl dreamer_v3.py:156-172
po = {"obs": SymlogDistribution(wm_outputs["reconstructed_obs"], dims=1)}
pr = TwoHotEncoding(wm_outputs["reward_logits"], dims=1)
pc = IndependentBernoulli(wm_outputs["continue_logits"])
continue_targets = 1.0 - batch["terminated"]          # §S10, sheeprl L168

num_cat = wm.rssm.num_categoricals
num_cls = wm.rssm.num_classes
post_logits = wm_outputs["posterior_logits"].reshape(
    *wm_outputs["posterior_logits"].shape[:2], num_cat, num_cls)
prior_logits = wm_outputs["prior_logits"].reshape(
    *wm_outputs["prior_logits"].shape[:2], num_cat, num_cls)

total, kl_mean, kl_loss_mean, reward_loss_mean, obs_loss_mean, cont_loss_mean = (
    reconstruction_loss(
        po, {"obs": batch["obs"]}, pr, batch["rewards"],
        prior_logits, post_logits,
        kl_dynamic=kl_dynamic, kl_representation=kl_representation,
        kl_free_nats=kl_free_nats, kl_regularizer=kl_regularizer,
        pc=pc, continue_targets=continue_targets,
        continue_scale_factor=continue_scale_factor,
    )
)
```

- Keep the WM-quality aux probes (`:753-790`) unchanged (they read `wm_outputs` directly).
- Re-wire the aux/logging dict (`:775-783`) to the six returned scalars, **keeping the
  existing WandB key names** (`kl` = `kl_mean`, the pre-floor dynamic KL, matching current
  semantics; per-term means map 1:1). Values will shift (see P3 design note) — names must not.
- Update the now-wrong comment at `:704` ("Normal(symlog(pred), 1)") — that mis-model was the
  root cause of the 2× bug ([[02_world_model_losses]] Detail B); the new comment should cite
  `SymlogDistribution` + `loss.py`.

#### 5. `src/algorithms/dreamer_srl/dreamer_srl_main.py` (~:1525-1527) — P5

```python
# BEFORE:
        # Update episode tracking
        episode_lengths += 1
        episode_rewards += rewards.astype(np.float32)

# AFTER:
        # Update episode tracking — WP-SRL P5: skip envs that finished THIS
        # iteration. Their terminal transition was already counted into the
        # logged episode (ep_len = counters+1, ep_rew = counters+rewards[i] at
        # the done block above); the unconditional increment previously leaked
        # +1 step and the terminal reward into the SUCCESSOR episode's counters.
        # sheeprl needs no counters (gym wrapper final_info, dreamer_v3.py:610-618).
        _alive = ~dones.astype(bool)
        episode_lengths[_alive] += 1
        episode_rewards[_alive] += rewards[_alive].astype(np.float32)
```

For testability, implement the three AFTER lines as a module-level helper and call it here:

```python
def _advance_episode_counters(episode_lengths, episode_rewards, rewards, dones) -> None:
    """WP-SRL P5: advance per-env episode counters, excluding envs done this iter."""
    alive = ~np.asarray(dones, dtype=bool)
    episode_lengths[alive] += 1
    episode_rewards[alive] += rewards[alive].astype(np.float32)
```

The done-block logging (`:1213-1214`) and the zeroing (`:1291-1292`) stay **unchanged** —
with the leak removed, `episode_lengths[i] + 1` / `episode_rewards[i] + rewards[i]` are
exactly right (the +1/+reward accounts for the in-flight terminal transition, which is
never counter-incremented for a done env).

#### 6. `src/algorithms/dreamer_srl/dreamer_srl_main.py` (:484 region, :1117-1131, :1540) — P6

At the config-read block (after `num_envs = args.num_envs`, `:488`):

```python
# BEFORE (:484):
    learning_starts = agent_cfg.get_mandatory("algo.learning_starts", int)

# AFTER (:484 + new lines after :488 once num_envs is known):
    learning_starts_cfg = agent_cfg.get_mandatory("algo.learning_starts", int)
    ...
    num_envs = args.num_envs
    # WP-SRL P6: the config value is an ENV-STEP count (sheeprl semantics).
    # Ported from sheeprl@33b6366:dreamer_v3.py:508-511.
    learning_starts, prefill_steps = derive_prefill(learning_starts_cfg, num_envs)
```

At the train gate (`:1540`):

```python
# BEFORE:
            ratio_steps = policy_step - learning_starts * num_envs

# AFTER (line-for-line sheeprl dreamer_v3.py:661, world_size == 1):
            ratio_steps = policy_step - prefill_steps * num_envs
```

Also required:
- Rewrite the stale comment blocks at `:1117-1129` and `:1534-1539` (both describe the
  pre-fix D-014 semantics; after this change the derivation and subtraction match sheeprl
  line-for-line — say so, cite `dreamer_v3.py:508-511, 660-661`).
- The startup print (`:580`) and the WandB config entry (`:726`) now carry the derived
  iteration count — log both: `learning_starts` (iters) and `learning_starts_env_steps`
  (= `learning_starts_cfg`).
- Behaviour is bit-identical at `num_envs=1` (÷1) and at `learning_starts_cfg=0`
  (D-012 smoke configs: `derive_prefill(0, n) == (0, 0)`).
- **DEVIATION_LOG D-014 refresh in the same commit** (exact text in the DEVIATION_LOG
  deliverable section).

#### 7. `src/algorithms/dreamer_srl/dreamer_srl_main.py` (:1048-1065, :1549-1556) — P7

```python
# BEFORE (:1064-1065):
    _G: int = max(1, int(replay_ratio * num_envs))  # steady-state grad steps per iter
    _grad_step_remainder: float = 0.0               # fractional carry across iters

# AFTER:
    # WP-SRL P7 / D-015: the scan path runs a CONSTANT _G gradient steps per
    # training iteration (one XLA executable — Fix 2, recompile-storm). The
    # former fractional remainder carry was dead code and has been deleted;
    # fractional replay ratios are QUANTIZED on this path (declared deviation
    # D-015 in DEVIATION_LOG.md). Exact Ratio cadence: use --legacy-grad-loop.
    _G: int = max(1, int(replay_ratio * num_envs))  # steady-state grad steps per iter
```

```python
# BEFORE (:1549-1556):
            if not args.legacy_grad_loop and n_grad_steps > 0:
                _grad_step_remainder += n_grad_steps - _G
                # If remainder has grown to >= 1, we owe an extra step.
                # If remainder has fallen to <= -1, we skip a step.
                # In practice at steady state n_grad_steps ≈ _G so remainder stays ~0.
                n_grad_steps_scan = _G
            else:
                n_grad_steps_scan = n_grad_steps  # legacy path: exact Ratio value

# AFTER:
            # D-015: constant _G on the scan path (quantized ratio, declared);
            # legacy path keeps the exact Ratio return.
            if not args.legacy_grad_loop and n_grad_steps > 0:
                n_grad_steps_scan = _G
            else:
                n_grad_steps_scan = n_grad_steps  # legacy path: exact Ratio value
```

Also trim the now-misleading half of the `:1048-1063` comment block (the "track a fractional
remainder so the long-run total equals Ratio" sentences — that contract is NOT preserved for
fractional ratios; D-015 says so honestly).

#### 8. `src/algorithms/dreamer_srl/buffers.py` — P2: `EnvIndependentSequentialReplayBuffer`

New class (bottom of the file), a structural port of
`vendor/sheeprl/sheeprl/data/buffers.py:529-699` specialized to
`buffer_cls=SequentialReplayBuffer` (CPU only — see design note):

```python
class EnvIndependentSequentialReplayBuffer:
    """n_envs independent single-env SequentialReplayBuffers.

    Structural port of sheeprl@33b6366:sheeprl/data/buffers.py:L529-L699
    (EnvIndependentReplayBuffer with buffer_cls=SequentialReplayBuffer),
    minus the memmap trio (D-004) and torch tensor methods. WP-SRL P2:
    replaces the shared-write-head buffer whose done-mask reset writes
    punched hole rows into non-done envs' columns.
    """

    def __init__(self, buffer_size: int, n_envs: int = 1,
                 obs_keys: Sequence[str] = ("obs",)):
        if buffer_size <= 0 or n_envs <= 0:
            raise ValueError(...)
        self._buf = [
            SequentialReplayBuffer(buffer_size=buffer_size, n_envs=1,
                                   obs_keys=obs_keys, device="cpu")
            for _ in range(n_envs)
        ]
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        self._rng = np.random.default_rng()   # bincount RNG — unseeded like sheeprl (P-14)
        self._on_gpu = False                  # driver's :1567 branch reads this

    def add(self, data, validate_args: bool = False, done_mask=None) -> None:
        # sheeprl buffers.py:627-654. done_mask replaces sheeprl's `indices`
        # (Fix-3 fixed-width convention): data is full-width [seq, n_envs, ...],
        # only truthy columns are routed — non-done sub-buffers are untouched
        # (dreamer_v3.py:650 — the no-hole-rows property).
        if done_mask is not None:
            indices = np.where(np.asarray(done_mask, dtype=bool))[0]
        else:
            indices = range(self._n_envs)
        for env_idx in indices:
            env_data = {k: v[:, env_idx:env_idx + 1] for k, v in data.items()}
            self._buf[env_idx].add(env_data, validate_args=validate_args)

    def sample(self, batch_size: int, sequence_length: int, n_samples: int = 1,
               sample_next_obs: bool = False, clone: bool = False):
        # sheeprl buffers.py:656-699: bincount allocation + concat on batch axis.
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(...)
        bs_per_buf = np.bincount(
            self._rng.integers(0, self._n_envs, (batch_size,)),
            minlength=self._n_envs,
        )
        per_buf = [
            b.sample(batch_size=bs, sequence_length=sequence_length,
                     n_samples=n_samples, sample_next_obs=sample_next_obs, clone=clone)
            for b, bs in zip(self._buf, bs_per_buf) if bs > 0
        ]
        # sub-buffer output [n_samples, seq_len, bs, ...] → concat batch axis 2
        return {k: np.concatenate([s[k] for s in per_buf], axis=2)
                for k in per_buf[0]}

    def ready_to_sample(self, sequence_length: int) -> bool:
        # WP-SRL P8 gate: every env column has >= sequence_length valid rows,
        # counting wrapped buffers as full (b._full OR b._pos >= seq_len).
        return all(b._full or b._pos >= sequence_length for b in self._buf)

    @property
    def filled_size(self) -> int:
        # total stored transitions (continual-stage-swap logging, main.py:1369)
        return int(sum(b._buffer_size if b._full else b._pos for b in self._buf))

    def reset(self) -> None:
        for b in self._buf:
            b.reset()
```

Additionally give the plain `SequentialReplayBuffer` the same two conveniences so the driver
is class-agnostic:

```python
    def ready_to_sample(self, sequence_length: int) -> bool:
        return bool(self._full or self._pos >= sequence_length)

    @property
    def filled_size(self) -> int:
        return int(self._buffer_size if self._full else self._pos)
```

(Adapt signatures to the real `SequentialReplayBuffer.__init__`/`sample` signatures — the
snippets above pin semantics, not final keyword lists. The existing shared-head
`done_mask` branch in `SequentialReplayBuffer.add` (`:219-230`) stays — it is still used by
nothing in the driver after this change but is covered by existing tests; leave it, note it
as superseded in its comment.)

#### 9. `src/algorithms/dreamer_srl/dreamer_srl_main.py:667-672` — P2 construction + sizing

```python
# BEFORE:
    buffer = SequentialReplayBuffer(
        buffer_size=buffer_size,
        n_envs=num_envs,
        obs_keys=("obs",),
        device=args.buffer_device,
    )

# AFTER:
    # WP-SRL P2: per-env independent buffers + sheeprl capacity semantics.
    # Sizing ported from sheeprl@33b6366:dreamer_v3.py:478
    # (buffer_size = cfg.buffer.size // (num_envs * world_size), world_size == 1).
    per_env_buffer_size = buffer_size // num_envs
    if args.buffer_device == "gpu":
        if num_envs != 1:
            raise ValueError(
                "--buffer-device gpu supports num_envs == 1 only (WP-SRL P2: "
                "bincount sampling across per-env buffers would recompile per "
                "allocation pattern on the traced GPU path)."
            )
        buffer = SequentialReplayBuffer(
            buffer_size=per_env_buffer_size, n_envs=1, obs_keys=("obs",), device="gpu",
        )
    else:
        buffer = EnvIndependentSequentialReplayBuffer(
            buffer_size=per_env_buffer_size, n_envs=num_envs, obs_keys=("obs",),
        )
```

The two existing call sites need no shape changes: `buffer.add(step_data, validate_args=False)`
(`:1145`) and `buffer.add(reset_data, done_mask=dones, validate_args=False)` (`:1272`) keep
their signatures. `:1369` becomes `_pre_size = buffer.filled_size`.

#### 10. `src/algorithms/dreamer_srl/dreamer_srl_main.py:1558` — P8

```python
# BEFORE:
            if n_grad_steps > 0 and buffer._pos >= seq_len:

# AFTER (WP-SRL P8: a wrapped-full buffer is sampleable; the old expression
# skipped up to seq_len-1 owed grad steps after every ring wrap):
            if n_grad_steps > 0 and buffer.ready_to_sample(seq_len):
```

### DEVIATION_LOG deliverable — exact row edits

File: `docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md`. The `developer` writes each row
in the **same commit** as its code step (table above). Never renumber; append D-015/D-016 at
the bottom of the table; the D-014 refresh is **additive** (append to the existing cells —
no silent rewrite, per the log's own convention).

**(a) D-014 — refresh (with Step 4 / P6).** Append to the row's "What we did instead" cell:

> **UPDATE 2026-07-08 (WP-SRL P6, [[fix_plan_srl_parity]]):** this row is now historical.
> The driver no longer omits the subtraction, and the intermediate variant it acquired later
> (`ratio_steps = policy_step - learning_starts * num_envs`, the "D-014 fix" comment at
> `dreamer_srl_main.py:1534-1540`, flagged stale as S-01 in [[04_training_loop_replay]]) has
> also been replaced. As of WP-SRL the driver derives
> `learning_starts, prefill_steps = derive_prefill(cfg_value, num_envs)`
> (= `cfg // num_envs` and `learning_starts - int(learning_starts > 0)`, sheeprl
> `dreamer_v3.py:508-511`) and computes
> `ratio_steps = policy_step - prefill_steps * num_envs` — a line-for-line match of sheeprl
> `dreamer_v3.py:661` at `world_size=1`, including the intentional off-by-one in
> `prefill_steps`. The one-shot debt-repayment burst described in this row no longer occurs;
> boundary behaviour now matches sheeprl's smeared per-iter rate. Regression guard:
> `tests/algorithms/dreamer_srl/test_prefill.py` (extended, WP-SRL P6 cases).

And append to its "PI verdict" cell: `— row historical as of 2026-07-08 (WP-SRL P6)`.

**(b) D-015 — new row (with Step 5 / P7).**

| ID | CP | Function | Sheeprl source line | What we did instead | Why | Bit-identity test result | PI verdict | Resolution link |
|---|---|---|---|---|---|---|---|---|
| D-015 | 2026-07-06 re-audit (P7) | `dreamer_srl_main.py` driver — scan-path gradient-step count (`_G`) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L662` (`per_rank_gradient_steps = ratio(ratio_steps / world_size)` — exact, un-quantized `Ratio` return each iteration) | The default (non-`--legacy-grad-loop`) scan path runs a **constant** `_G = max(1, int(replay_ratio * num_envs))` gradient steps per training iteration instead of the exact per-iteration `Ratio` return. The dead fractional-remainder carry (`_grad_step_remainder`) was **deleted** (WP-SRL P7, 2026-07-08) rather than wired: consuming it would reintroduce variable scan lengths and the per-length XLA recompile storm that the constant-length design (Fix 2) exists to prevent. Effective scan-path replay ratio = `max(1, int(replay_ratio * num_envs)) / num_envs` — e.g. `replay_ratio=0.5, num_envs=1` trains at an effective 1.0. Exact at every config where `replay_ratio * num_envs` is a positive integer (all canonical parity configs: `replay_ratio=1`). `--legacy-grad-loop` retains the exact `Ratio` cadence for fractional-ratio work (e.g. replay-ratio sweeps). | XLA constant-shape constraint (substrate-class, same family as the D-001/D-011 mechanism deviations): one compiled executable per run requires a compile-time-constant scan length; sheeprl (eager PyTorch) has no such constraint. The `Ratio` class itself remains a line-for-line port (P-03); only the consumption of its return is quantized. | n/a — cadence deviation, no numerical threshold applies; regression guard `tests/algorithms/dreamer_srl/test_lax_scan_train.py` (scan path runs exactly `_G` steps) | ✅ APPROVED — 2026-07-08 (user blanket "fix all" disposition for the WP-SRL parity package, [[fix_plan_srl_parity]]; declaration pre-approved as part of the approved package) | [[fix_plan_srl_parity]] |

**(c) D-016 — new row (with Step 8 / P9).**

| ID | CP | Function | Sheeprl source line | What we did instead | Why | Bit-identity test result | PI verdict | Resolution link |
|---|---|---|---|---|---|---|---|---|
| D-016 | 2026-07-06 re-audit (P9) | `RSSM` output-linear init — `transition_out`, `repr_out` (`agent.py:703-714`) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1173-L1174` (`rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))`, same for `representation_model`, under `hafner_initialization`) | Our RSSM initializes the transition/representation **output** linear kernels with `init_weights` (truncated normal, Hafner fan-avg std) instead of the reference's `uniform_init_weights(1.0)` (Hafner uniform, gain 1.0). Both draw zero-mean with the same theoretical std; the distribution **shape** differs (truncated Gaussian vs uniform). **KEPT, not fixed** (WP-SRL P9): the deviation is init-time only — it perturbs the starting point, not the objective, gradients, or any per-step computation; after the first optimizer steps the parameter distributions are training-dominated. Cross-platform init streams are already non-bit-comparable (D-002 precedent). | Fixing would touch init plumbing for zero expected training effect; declared for record completeness per the 2026-07-06 re-audit ([[00_master_comparison]] §3 P9). Same class as D-002 (distribution-property, not bit-identity). | n/a — init-distribution class; D-002-style property test already covers std/bounds of `init_weights` | ✅ APPROVED — 2026-07-08 (user blanket "fix all" disposition for the WP-SRL parity package explicitly scoped P9 as declare-not-fix, [[fix_plan_srl_parity]]) | [[fix_plan_srl_parity]] |

No other DEVIATION_LOG edits. (Fixed items P1/P2/P3/P5/P8 need no rows — they are no longer
deviations; their record is this plan + the master comparison.) Post-merge, ask
`bug-curator` to update the open KNOWN_BUGS gradient-clipping row (close as fixed, all three
optimizers) — registry edits are bug-curator's, not the developer's.

### Regression tests

All run with `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest <path> -q`.
Full suite: `tests/algorithms/dreamer_srl/`. "Red pre-fix" = the assertion fails on current
code; "red by absence" = the test imports a symbol the fix introduces (fails collection
pre-fix — acceptable where the buggy logic is inline in the driver loop and not importable).

| Fix | Test (file :: name) | Fixture / assertion | Red pre-fix? |
|---|---|---|---|
| P1 | `tests/algorithms/dreamer_srl/test_grad_clip.py::test_clip_norm_constants` (NEW file) | `WM_CLIP_NORM/ACTOR_CLIP_NORM/CRITIC_CLIP_NORM == 1000/100/100` (pins to sheeprl yaml:52/127/154) | red by absence |
| P1 | `test_grad_clip.py::test_spike_grad_does_not_poison_adam` | `tx = make_optim_tx(1e-3, 1e-8, 100.0)`; step 1: synthetic exploding grad, global norm 1e8; step 2: unit-norm grad. Assert step-2 update global norm > 1e-5. Discriminating: WITH the clip, Adam's second moment stays 100-scale and step 2 moves; WITHOUT it, the 1e8 spike poisons `ν` and step 2 is ~1e-10 (the real post-spike learning-freeze pathology). Also assert step-1 post-clip behaviour: updates finite. | red by absence; the assertion itself fails if the chain ever loses the clip (guards regression) |
| P2 | `tests/algorithms/dreamer_srl/test_env_independent_buffer.py::test_no_hole_rows_on_partial_done` (NEW file) | The area-report-4 probe as a fixture: `n_envs=2`, per-step obs values env0=`[20,30,40]`, env1=`[21,31,41]`; after step 2 env0 is done → `add(reset_data, done_mask=[True, False])`; then step 3 row. Assert env1's sub-buffer contents are contiguous `[21, 31, 41]` — **no `0` hole** — and env0's contain its reset row. On the old shared-head buffer this sampled `[21, 31, 0, 41]`. | **YES** (fails on `SequentialReplayBuffer` shared head; passes on the wrapper) |
| P2 | `test_env_independent_buffer.py::test_per_env_heads_advance_independently` | After the partial-done write: env0 sub-buffer `_pos` == env1 `_pos` + 1 | red by absence |
| P2 | `test_env_independent_buffer.py::test_capacity_division` | Driver-mirroring construction at `buffer_size=256000, num_envs=4` → each sub-buffer `_buffer_size == 64000` (sheeprl `dreamer_v3.py:478`) | red by absence |
| P2 | `test_env_independent_buffer.py::test_sample_shape_and_env_isolation` | Fill 2 envs with disjoint value ranges; `sample(batch_size=8, sequence_length=3, n_samples=2)` → shape `[2, 3, 8, ...]`; every sampled sequence lies entirely in one env's value range (no cross-env sequences) | red by absence |
| P3 | `tests/algorithms/dreamer_srl/test_loss.py::test_symlog_distribution_matches_reference_formula` (extend file) | `SymlogDistribution(pred, dims=1).log_prob(target) == -Σ where((pred-symlog(target))² < 1e-8, 0, ·)` on a random fixture incl. sub-tol elements | red by absence |
| P3 | `test_loss.py::test_obs_loss_exactly_2x_old_inline` | The measured-ratio regression: `pred` ~ N(0,1) `[T,B,D]` as the symlog-space prediction, `target` real-space. `old = 0.5 * Σ(pred - symlog(target))²` (the deleted inline formula fed identical symlog-space residuals, i.e. evaluated at `reconstructed_obs = symexp(pred)`); `new = -SymlogDistribution(pred, dims=1).log_prob(target)`. Assert `new / old == 2.0` within 1e-5 rel (area report 2 measured 0.500000 the other way). | encodes the audited ratio; guards re-introduction of the ½ |
| P3 | `test_loss.py::test_reconstruction_loss_with_symlog_po` | `reconstruction_loss` end-to-end with `po={"obs": SymlogDistribution(...)}`: total == manual sum of the four terms' means; obs term matches `-po["obs"].log_prob` mean | extends existing coverage to the newly-live call path |
| P5 | `tests/algorithms/dreamer_srl/test_episode_metrics.py::test_second_episode_counters_unbiased` (NEW file) | Import `_advance_episode_counters`; script two back-to-back episodes for one env replaying the driver's exact order (log `len=counters+1`, `rew=counters+reward` at done → zero → advance): ep1 = 3 steps, rewards `[1, 2, 10]`; ep2 = 2 steps, rewards `[5, 7]`. Assert **both counters, both episodes**: logged `(l, r)` == `(3, 13.0)` and `(2, 12.0)`. Under the old unconditional increment ep2 logs `(3, 22.0)` — the +1 step / +terminal-reward bleed. | red by absence (helper new); the two-episode fixture is the discriminator — document the old-code values in the test docstring |
| P6 | `tests/algorithms/dreamer_srl/test_prefill.py::test_derive_prefill` (extend file) | `derive_prefill(1024, 1) == (1024, 1023)`; `(1024, 4) == (256, 255)`; `(1024, 16) == (64, 63)`; `(0, 4) == (0, 0)` — prefill env-step count == config value at every env count (sheeprl `dreamer_v3.py:508-511`), and the D-012 zero-prefill smoke path unchanged | red by absence |
| P6 | `test_prefill.py` — existing `test_no_gradient_step_before_learning_starts` | Must still pass (the §S3 hard invariant survives the re-derivation); extend with a `num_envs>1` parametrization if the harness allows | guards the invariant |
| P7 | `tests/algorithms/dreamer_srl/test_lax_scan_train.py` (extend) | Scan path runs exactly `_G` grad steps per gated iteration; `_grad_step_remainder` no longer exists (`not hasattr` / grep-style assertion optional) | trivially green; documents D-015 |
| P8 | `test_env_independent_buffer.py::test_ready_to_sample_after_wrap` | Tiny buffer (`buffer_size=8`), `seq_len=4`: fill 10 rows so `_full=True, _pos=2 < seq_len`. Assert `ready_to_sample(4) is True`. The old gate expression `_pos >= seq_len` is False in this exact state → assert that inequality too, as documentation of what was wrong. Cover both classes (plain + wrapper). | **YES** in spirit (the old expression evaluates False on the fixture; the new method returns True) |

**Existing tests to update, and the known-red baseline.** Pre-existing failures the developer
must NOT chase (record them verbatim in the Implementation Report before starting):
**4× A1 parity gates, 3× stale-config (`b093023`), 1× dreamer_srl offline-WM smoke.**
Beyond those: P3 changes WM-loss *values*, so any test pinning the old inline obs-loss
numbers (`test_loss.py`, `test_train.py`, and the already-red `test_grad_parity.py` /
`test_end_to_end_parity.py`) needs fixture updates — re-derive expected values from the
faithful formula, never relax tolerances to make old numbers pass. P2 changes buffer
internals, so `test_buffers.py` / `test_buffer_reset.py` / `test_gpu_buffer.py` assertions on
scalar `_pos` semantics stay valid for the plain class (still used at `num_envs=1` GPU and as
the sub-buffer) but any driver-level assumptions must move to the wrapper. Diff the suite's
red set before vs after: the ONLY acceptable delta is fixture-updated tests going green.

### Checkpoints

- [x] **Step 0** — ran the full `tests/algorithms/dreamer_srl/` suite pre-change: **110 passed,
      2 skipped, 0 failed** — clean in-scope, nothing to reconcile. (The 4× A1 parity gates /
      3× stale-config / 1× offline-WM-smoke known-red baseline was later located precisely
      during the full-project run below: `tests/env/test_unified_parity.py` ×4
      [`observability_gates_S1`–`S4`], `tests/env/test_inactive_animal_offgrid.py` ×1 +
      `tests/env/test_truncation_not_death.py` ×2 [`FileNotFoundError` on an archived config
      path], and `tests/scripts/test_dreamer_srl_offline_wm_test.py::test_offline_wm_smoke`
      ×1 [`RuntimeError: Only 36 valid starting states`] — 8 total, exactly matching the
      brief's count. Confirmed pre-existing and unrelated to WP-SRL by re-running all 8 with
      the dreamer_srl fixes stashed out: identical 8 failures, same error messages.)
- [x] **Step 1 (P1)** — `test_grad_clip.py` green (2 passed). Discriminator confirmed
      numerically: same spike+200-unit-grad fixture gives clipped-Adam final update norm
      4.67e-4 vs unclipped 4.75e-10 (5+ orders of magnitude — the post-spike learning-freeze
      pathology, reproduced standalone before writing the assertion).
- [x] **Step 2 (P3)** — `grep -n "0.5 \* jnp.sum" src/algorithms/dreamer_srl/train.py` → no
      match. `grep -n "reconstruction_loss(" src/algorithms/dreamer_srl/train.py` → live call
      at the WM-loss site. Sanity smoke (num_envs=1, same seed, first post-prefill log at iter
      1024/1025): pre-fix `world_model_loss=13.5681`; post-fix `world_model_loss=24.4894` —
      consistent direction with the fix (recon under-weighting removed raises the total WM
      loss); the exact 2× ratio is pinned on the synthetic fixture in
      `test_obs_loss_exactly_2x_old_inline`, not expected bit-exact on live network state
      (KL/reward/continue terms and the RSSM trajectory differ between the two runs).
- [x] **Step 3 (P5)** — `test_episode_metrics.py` green (2 passed). Live-smoke cross-check
      (num_envs=1, same seed 0, same first 12 episodes): pre-fix ep 2 logged
      `ep_len=11 ep_rew=-325.005`; post-fix same ep 2 logged `ep_len=10 ep_rew=-200.499` —
      exactly 1 fewer survival step and the leaked terminal reward removed, matching the
      predicted P5 correction pattern.
- [x] **Step 4 (P6)** — `test_prefill.py` green (4 passed: 2 new + 2 existing, including the
      `num_envs>1` parametrized invariant). Startup print confirms the derivation live:
      `learning_starts=1024 iters (= 1024 env steps / 1 envs)` at num_envs=1. D-014 row
      refreshed (historical) in the same change — see DEVIATION_LOG section.
- [x] **Step 5 (P7)** — `grep -rn "_grad_step_remainder" src/` → no matches (confirmed via
      `test_grad_step_remainder_deleted`, which was RED pre-fix — see red-evidence table).
      D-015 row appended in the same change.
- [x] **Steps 6–7 (P2+P8)** — `test_env_independent_buffer.py` all green (8 passed), including
      the `[21, 31, 41]` no-hole fixture (`test_no_hole_rows_on_partial_done`) and the
      `ready_to_sample` post-wrap fixture (`test_ready_to_sample_after_wrap`, parametrized
      plain+wrapper). `num_envs=4` CPU smoke ran cleanly to completion (see SPS section).
      `--buffer-device gpu` at `num_envs=1` still runs (exercised inside `test_gpu_buffer.py`,
      unaffected — 100% pass in the post-fix full-module run). The
      `--buffer-device gpu --num-envs 4` ValueError path is asserted at construction in the
      driver (`args.buffer_device == "gpu" and num_envs != 1`) but was not separately
      re-verified with a fresh end-to-end CLI invocation — flagged as a minor follow-up gap,
      not a blocker (the guard is a single un-conditional `raise` with no data dependency to
      mis-fire on).
- [x] **Step 8 (P9)** — D-016 row appended; no code change (declare-only).
- [x] **Speed check (mandatory)** — see "Speed Check" section below. **No regression**: clean
      back-to-back (stash/pop) comparison on the same idle node/GPU/seed/config shows
      num_envs=4 steady-state SPS 28.75 → 30.21 (+5.1%, noise) and num_envs=1 parity-config
      steady-state SPS 6.21 → 6.12 (−1.4%, noise). An earlier temporally-separated measurement
      (not stash-based) showed a spurious ~30% apparent regression at num_envs=4 that traced to
      an unrelated third-party job occupying the same shared GPU during the "before" window —
      resolved by re-measuring stash/pop back-to-back under confirmed-idle GPU state (see
      Speed Check section for both raw measurement sets and the confound explanation).
- [x] **Final** — full `tests/algorithms/dreamer_srl/` suite: **129 passed, 2 skipped, 0
      failed** (the +19 over the 110-passed baseline are exactly the new/extended WP-SRL
      tests: 2 grad_clip + 2 episode_metrics + 8 env_independent_buffer + 3 loss + 2 prefill +
      2 lax_scan_train = 19). Zero tests turned red; zero baseline tests needed fixture
      updates in this module (the plan's warned `test_grad_parity.py` /
      `test_end_to_end_parity.py` / `test_train.py` / `test_loss.py` bit-identity risk did not
      materialize — all passed as-is; see Deviations below for why). Full-project suite
      (`tests/`, excluding `tests/models/`/parallel-WP-NNX and re-running
      `tests/algorithms/dreamer_srl/` separately above): **269 passed, 8 failed, 492 skipped**
      — the 8 failures are exactly the known-red baseline located above (verified identical
      with WP-SRL stashed out); zero new failures introduced outside `dreamer_srl`.
      `git diff --stat` confirms zero edits outside `src/algorithms/dreamer_srl/`,
      `tests/algorithms/dreamer_srl/`, and
      `docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md` (the `configs/` and `docs/diary/`
      diffs visible in `git status` predate this session — WP-GAMMA and other parallel work,
      confirmed untouched by this implementation). `src/models/` and `tests/models/` untouched.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-08 / 2026-07-09

### Summary — what a fresh reader needs to know

All nine WP-SRL parity items (P1, P3, P5, P6, P7, P2, P8 fixed; P9 declared) were implemented
in the plan's sequence, each with a new or extended regression test confirmed red before the
fix and green after. The two numerically-live fixes (P1 gradient clipping, P3 the faithful
observation-loss formula) were additionally cross-checked against short live training smokes
run at the same seed before and after, not just synthetic unit fixtures. A clean back-to-back
speed comparison (see Speed Check below) found **no throughput regression** — the buffer
restructure (P2) and the added per-optimizer clip reduction (P1) do not measurably slow
training; an initial ~30% apparent slowdown traced to an unrelated third-party job sharing the
GPU node, not to this package's changes. The full project test suite (minus the parallel
WP-NNX package under `tests/models/`, which this plan explicitly does not touch) shows zero
new failures — the 8 pre-existing failures match the plan's declared known-red baseline
exactly and were confirmed identical with WP-SRL's code stashed out.

### File-by-file

- **`src/algorithms/dreamer_srl/utils.py`** — added `WM_CLIP_NORM`/`ACTOR_CLIP_NORM`/
  `CRITIC_CLIP_NORM` (1000/100/100, sheeprl YAML-pinned) + `make_optim_tx(lr, eps, clip_norm)`
  (P1: `optax.chain(clip_by_global_norm, adam)`), and `derive_prefill(learning_starts_cfg,
  num_envs) -> (learning_starts, prefill_steps)` (P6, sheeprl `dreamer_v3.py:508-511`
  line-for-line at `world_size=1`).
- **`src/algorithms/dreamer_srl/loss.py`** — added `SymlogDistribution` (P3: MSE-in-symlog-space
  observation distribution, ported from sheeprl `distribution.py:152-193`, `dims=1`/`agg="sum"`
  branch only, with the `tol=1e-8` clamp).
- **`src/algorithms/dreamer_srl/train.py`** — P3: `wm_loss_fn` rewired to build
  `po={"obs": SymlogDistribution(...)}` and call the already-imported, previously-dead
  `reconstruction_loss` instead of the inline `-0.5 * sum((symlog(recon)-symlog(target))**2)`
  assembly; deleted the dead `_symlog` import this made unnecessary; the aux/logging dict keeps
  its pre-fix key names (`kl_mean`, `obs_loss_mean`, etc.) mapped 1:1 onto
  `reconstruction_loss`'s six returned scalars, with `dyn_kl_mean`/`rep_kl_mean` recomputed
  logging-only (the combined function only returns the merged `kl_loss_mean`).
- **`src/algorithms/dreamer_srl/dreamer_srl_main.py`** —
  - P1: optimizer construction now calls `make_optim_tx(...)` for all three optimizers (import
    added).
  - P5: new module-level `_advance_episode_counters(episode_lengths, episode_rewards, rewards,
    dones)` helper; the unconditional `episode_lengths += 1; episode_rewards += rewards` line
    replaced with a call to it.
  - P6: `learning_starts` renamed to `learning_starts_cfg` at the config-read site;
    `learning_starts, prefill_steps = derive_prefill(...)` added once `num_envs` is known;
    downstream `ratio_steps = policy_step - learning_starts * num_envs` corrected to
    `policy_step - prefill_steps * num_envs`; the two stale comment blocks (§S3 prefill-gate
    docstring, train-gate D-014 comment) rewritten; startup print and WandB config payload
    extended with `learning_starts_env_steps`.
  - P7: deleted `_grad_step_remainder` (both the declaration and the accumulate-then-discard
    usage); trimmed the now-inaccurate half of the surrounding comment block.
  - P2: buffer construction replaced with `EnvIndependentSequentialReplayBuffer` (CPU path,
    sized `buffer_size // num_envs`) or the plain `SequentialReplayBuffer` at `num_envs=1` GPU
    mode (`--buffer-device gpu` now raises `ValueError` for `num_envs != 1`); `_pre_size =
    buffer._pos if not buffer._full else buffer._buffer_size` replaced with the class-agnostic
    `buffer.filled_size`.
  - P8: train gate `buffer._pos >= seq_len` replaced with `buffer.ready_to_sample(seq_len)`.
- **`src/algorithms/dreamer_srl/buffers.py`** — added `ready_to_sample(seq_len)` and
  `filled_size` to `SequentialReplayBuffer` (P8, class-agnostic driver API); added the new
  `EnvIndependentSequentialReplayBuffer` class (P2: structural port of sheeprl's
  `EnvIndependentReplayBuffer` specialized to `SequentialReplayBuffer`, CPU-only, bincount
  sampling); annotated the existing shared-head `done_mask` branch in
  `SequentialReplayBuffer.add` as superseded-but-retained (still valid at `n_envs=1`).
- **Tests** — new: `test_grad_clip.py`, `test_episode_metrics.py`,
  `test_env_independent_buffer.py`. Extended: `test_loss.py` (+3 P3 tests),
  `test_prefill.py` (+2 P6 tests), `test_lax_scan_train.py` (+2 P7 tests).
- **`docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md`** — D-014 row refreshed
  (additive "UPDATE 2026-07-08" note + PI-verdict-cell append marking it historical); D-015
  and D-016 rows appended verbatim per the plan; frontmatter `last_updated` bumped with a
  summary note.

### Red → green evidence (one line each)

| Fix | Test | Pre-fix | Post-fix |
|---|---|---|---|
| P1 | `test_grad_clip.py` (2 tests) | Collection `ImportError` (constants/`make_optim_tx` didn't exist) | 2 passed; discriminator numeric: clipped final-update-norm 4.67e-4 vs unclipped 4.75e-10 |
| P3 | `test_loss.py` (3 new tests) | Collection `ImportError` (`SymlogDistribution` didn't exist) | 8/8 `test_loss.py` passed (5 pre-existing + 3 new) |
| P5 | `test_episode_metrics.py` (2 tests) | Collection `ImportError` (`_advance_episode_counters` didn't exist) | 2 passed |
| P6 | `test_prefill.py::test_derive_prefill` + multi-env invariant | Collection `ImportError` (`derive_prefill` didn't exist) | 4/4 `test_prefill.py` passed (2 pre-existing + 2 new) |
| P7 | `test_lax_scan_train.py::test_grad_step_remainder_deleted` | **True red** (`AssertionError`: `_grad_step_remainder` found in source) — the one genuinely-red-first assertion in the package, not red-by-absence | Passed alongside `test_scan_path_runs_constant_G_steps` |
| P2 | `test_env_independent_buffer.py` (8 tests) | Collection `ImportError`; the no-hole fixture additionally reproduced as a **true behavioural red** against the OLD `SequentialReplayBuffer(n_envs=2)` directly (`tmp/20260708_wp_srl_p2_probe_red.log`: env-1 stored `[21.0, 31.0, 0.0, 41.0]` — the exact hole-row pathology) | 8/8 passed, including the no-hole fixture now storing `[21, 31, 41]` |
| P8 | `test_env_independent_buffer.py::test_ready_to_sample_after_wrap` | Collection `ImportError`; fixture also asserts the OLD gate expression (`_pos >= seq_len`) evaluates `False` in the reproduced post-wrap state, documenting the skipped-grad-steps bug | Passed (parametrized plain + wrapper) |
| P9 | — (declare-only) | n/a | D-016 row appended |

Live-smoke cross-checks (same seed 0, `num_envs=1`, `configs/models/dreamer_srl/01_food_only.yaml`):
- **P3**: first post-prefill logged `world_model_loss` at iter ~1024/1025 rose `13.5681` (pre)
  → `24.4894` (post) — consistent direction with removing the 2× recon under-weighting (not
  expected bit-exact on live network state; the exact 2× ratio is pinned on the synthetic
  fixture instead).
- **P5**: episode 2 of the same run logged `ep_len=11, ep_rew=-325.005` (pre) →
  `ep_len=10, ep_rew=-200.499` (post) — exactly 1 fewer survival step and the leaked
  predecessor terminal reward removed.

### Speed Check

**Commands** (`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
src/algorithms/dreamer_srl/dreamer_srl_main.py --env-config
configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml --agent-config
configs/models/dreamer_srl/01_food_only.yaml --seed 0 --log-interval 50 --no-wandb`, plus
`--num-envs 1 --total-steps 4000` or `--num-envs 4 --total-steps 12000`), node 102, GPU 1
(RTX 4090), `CUDA_VISIBLE_DEVICES=1`. SPS parsed from the trainer's own `sps=` log field via
`tmp/20260708_wp_srl_sps_parse.py` (cumulative + mean instantaneous SPS over the last 25% of
logged windows — steady-state).

**Confound discovered and corrected.** The first before/after pair (temporally separated, run
sequentially over ~2 hours as other steps landed) showed num_envs=4 steady-state SPS
**41.5 → 29.3 (−29%)** — an apparent blocker-level regression. Investigating before reporting
it: `nvidia-smi` showed a **third-party process holding 18.6–23.8 GB on the same GPU** during
parts of this window (not ours — no `CUDA_VISIBLE_DEVICES` isolation on a shared lab node
protects against a co-resident job's compute contention, only its device *visibility*). This
fully explains an artifact in either direction depending on which run overlapped the other
job's phase.

**Clean re-measurement**: `git stash push -- <5 dreamer_srl src files>` immediately before,
`git stash pop` immediately after, running the identical command back-to-back under a
confirmed-idle GPU (`nvidia-smi` checked immediately before each launch: 2 MiB used). This
isolates the code-path change from all node-contention confounds.

| Config | Pre-fix steady-state SPS | Post-fix steady-state SPS | Δ |
|---|---|---|---|
| `num_envs=1` (parity config, `01_food_only.yaml`, `--total-steps 4000`) | 6.21 | 6.12 | −1.4% (noise) |
| `num_envs=4` (same config, `--total-steps 12000`) | 28.75 | 30.21 | +5.1% (noise) |

**Conclusion: no regression.** Both deltas are within normal run-to-run noise (a single
JIT-warmup / GC-timing / OS-scheduler jitter easily produces ±5% on a 10-15 minute smoke); the
P2 buffer restructure and P1 clip do not measurably affect end-to-end throughput at this scale.
A standalone CPU-only micro-benchmark of *just* the buffer hot path
(`tmp/20260708_wp_srl_p2_microbench.py`, no GPU/JIT involved) did show the per-env wrapper at
+156% buffer-only time (0.229 ms/iter → 0.585 ms/iter at `num_envs=4`) — but this cost is
sub-millisecond and fully absorbed by the multi-hundred-millisecond GPU-bound env-step +
train-step cost per iteration, consistent with the measured end-to-end noise-floor delta.

### Deviations from the plan

1. **P7 red-evidence classification** — the plan's test-contract table marks P7's
   `test_grad_step_remainder_deleted` implicitly alongside the other "red by absence" tests.
   In practice it is the one **true pre-fix-failing assertion** in the package (source-grep
   finds `_grad_step_remainder` present, so the assertion fails, not the import) — flagging
   this because it is the strongest kind of red evidence in the set and worth distinguishing
   from the seven collection-time `ImportError`s.
2. **P8 GPU-mode `ValueError` end-to-end check** — the plan's checkpoint asks to confirm
   `--buffer-device gpu --num-envs 4` raises. I verified this by code inspection (a single
   unconditional `raise ValueError` gated only on `args.buffer_device == "gpu" and num_envs !=
   1`, no data dependency) rather than a fresh CLI invocation, to avoid an extra ~10-minute GPU
   smoke for a guard with no branch to mis-fire on. Flagging as a minor scope note, not a
   blocker — `senior-developer` may want a 30-second CLI confirmation during verification.
3. **Speed-check methodology addition** — the plan asks for "before/after SPS on the same
   node/GPU/seed/config" but does not anticipate shared-node contention. I added a stash/pop
   back-to-back protocol beyond what's written, specifically because the first (plan-literal)
   measurement produced a false-alarm 29% apparent regression that would have been reported as
   a blocker without the contention check. This is a methodology addition, not a deviation from
   scope.
4. **Session interruption / stash recovery** — a connection error interrupted this session
   mid-comparison, at a point where dreamer_srl fixes were stashed out for a clean before/after
   measurement. On resuming, `git stash pop` initially failed ("local changes would be
   overwritten") because of stale index state left by an intervening `git stash push`+run+(pop
   attempted twice); recovered cleanly via `git checkout stash@{0} -- <paths>` (byte-identical
   restore, confirmed via `git diff stash@{0} -- <paths>` returning empty) followed by `git
   restore --staged` and `git stash drop`. Final working tree confirmed to exactly match all
   fixes (spot-checked symbols + full `tests/algorithms/dreamer_srl/` re-run, 129 passed).
   No commits were made at any point; no destructive git operations (`clean -x`, `reset
   --hard`, force-checkout) were used.
5. **No file-list surprises** — every file touched (`utils.py`, `dreamer_srl_main.py`,
   `train.py`, `loss.py`, `buffers.py`, plus the six test files and `DEVIATION_LOG.md`) appears
   in the plan's File Changes section. No out-of-plan files were modified.

### Known-red baseline (confirmed, located precisely)

Full-project suite (`tests/`, excluding `tests/models/` — the parallel WP-NNX package):
**269 passed, 8 failed, 492 skipped** in 429s. The 8 failures:
- `tests/env/test_unified_parity.py::test_parity[configs__verification__observability_gates_S{1,2,3,4}]` (4× "A1 parity gates")
- `tests/env/test_inactive_animal_offgrid.py::test_allactive_config_no_offgrid_parking` +
  `tests/env/test_truncation_not_death.py::test_timeout_no_death_penalty` +
  `tests/env/test_truncation_not_death.py::test_starvation_applies_death_penalty` (3× stale-config `FileNotFoundError`)
- `tests/scripts/test_dreamer_srl_offline_wm_test.py::test_offline_wm_smoke` (1× offline-WM smoke, `RuntimeError: Only 36 valid starting states`)

Confirmed pre-existing and unrelated to WP-SRL: re-ran all 8 with the 5 dreamer_srl source
files stashed out (`git stash push -- <paths>`) — identical 8 failures, identical error
messages, then restored via `git checkout stash@{0} -- <paths>` + `git stash drop`.

### Blockers / follow-ups for senior-developer

- None blocking. The one minor gap (P8 GPU-mode `ValueError` — code-inspected, not
  CLI-re-verified) is noted above for optional spot-check during verification.
- Per the plan: "Post-merge, ask `bug-curator` to update the open KNOWN_BUGS gradient-clipping
  row (close as fixed, all three optimizers)" — registry edit is `bug-curator`'s, not mine;
  flagging so `senior-developer` remembers to route it after this package is verified.

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-08

All nine items verified against the plan by independent diff review, independent re-runs of
both test suites, an independent reproduction of the P2 hole-row pathology on the old buffer
class, and a fresh CLI check of the GPU-mode guard the developer had only code-inspected.

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/algorithms/dreamer_srl/utils.py` | P1 constants + `make_optim_tx`, P6 `derive_prefill` | ✅ | Matches plan snippets exactly: norms 1000/100/100, `optax.chain(clip_by_global_norm, adam)` clip-then-step order, sheeprl `dreamer_v3.py:508-511` derivation incl. the intentional off-by-one |
| `src/algorithms/dreamer_srl/loss.py` | P3 `SymlogDistribution` | ✅ | Verbatim port per plan (`dims` event-sum, `tol=1e-8` clamp, `mode`/`mean` = `symexp` at consumption). `reconstruction_loss` return order re-checked against the train.py unpack — matches |
| `src/algorithms/dreamer_srl/train.py` | P3 rewire to `reconstruction_loss` | ✅ | Inline `-0.5·Σ(symlog−symlog)²` assembly and hand-rolled KL/reward/continue/total DELETED; dead `_symlog` import removed; faithful call live with all six scalars mapped to the pre-fix WandB key names; `dyn_kl/rep_kl` recomputed logging-only under `stop_gradient` (value-identical to old semantics — dyn/rep KL values coincide, only grad routing differed). `grep` re-confirmed `reconstructed_obs` has no consumer outside this loss — the 0.5 factor and the extra symlog are fully out of the training path |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | P1 opt wiring, P5 helper, P6 derivation + `ratio_steps`, P7 deletion, P2 construction + `filled_size`, P8 gate | ✅ | All six hunks match the plan line-for-line. `grep -rn "_grad_step_remainder" src/ scripts/ configs/` → zero hits (no orphans); `--legacy-grad-loop` escape hatch documented at the D-015 comment per plan; stale D-014/S-01 comment blocks rewritten; startup print + WandB payload carry both `learning_starts` (iters) and `learning_starts_env_steps` |
| `src/algorithms/dreamer_srl/buffers.py` | P2 `EnvIndependentSequentialReplayBuffer`, P8 `ready_to_sample`/`filled_size`, superseded-branch annotation | ✅ | Structural port matches the plan's pinned semantics 1–5 (driver-side `//num_envs`, per-env heads, done-only routing, unseeded-RNG bincount + axis-2 concat, per-env valid-idx exclusion for free); GPU mode restricted to `num_envs==1` with the plain class |
| `tests/.../test_grad_clip.py` (new) | P1 | ✅ | Re-run verbose: 2 passed. Regression-discriminating in BOTH directions: asserts the clipped chain moves (>1e-5) AND that plain Adam freezes (<1e-6) on the same spike fixture — losing the clip from the chain fails the first assertion, fixture drift fails the second |
| `tests/.../test_episode_metrics.py` (new) | P5 | ✅ | 2 passed. Two-episode fixture replays the driver's exact log→zero→advance order; old-code value (3, 22.0) documented — a regression of the helper to unconditional increment fails it |
| `tests/.../test_env_independent_buffer.py` (new) | P2 + P8 | ✅ | Re-run verbose: 8 passed. **True red independently reproduced**: I re-ran the probe fixture against the OLD `SequentialReplayBuffer(n_envs=2)` (still importable) — env-1 stored `[21.0, 31.0, 1.9e-06, 41.0]`, i.e. the hole row punched with uninitialized-memory garbage (the developer's log shows `0.0` at that slot; the value is `np.empty` garbage, the pathology is the punched row itself, confirmed either way). P8 wrap fixture asserts the old gate expression is False in the reproduced state — true red in spirit as planned |
| `tests/.../test_loss.py` (+3) | P3 | ✅ | The exact-2× ratio test discriminates the deleted 0.5 formula; the reference-formula test additionally forces a full sub-tol ROW so the tol-clamp check cannot be absorbed by float32 summation (a nice strengthening over the plan's sketch); end-to-end `reconstruction_loss` test covers the newly-live call path |
| `tests/.../test_prefill.py` (+2) | P6 | ✅ | `derive_prefill` pinned at (1024,1)/(1024,4)/(1024,16)/(0,4) incl. env-step-coverage invariant; §S3 hard invariant re-proved at num_envs∈{1,4} with the new `prefill_steps` subtraction |
| `tests/.../test_lax_scan_train.py` (+2) | P7 | ✅ | `test_grad_step_remainder_deleted` was the package's one true red-first assertion (pre-fix source demonstrably contained the symbol — visible in the diff's BEFORE hunks); source-level `_G` quantization guard pins D-015's declared cadence |
| `docs/.../DEVIATION_LOG.md` | D-014 refresh, D-015, D-016 | ✅ | Row text verbatim per the plan's DEVIATION_LOG deliverable; D-014 refresh is additive (original cells intact) + PI-verdict-cell append; D-014's historical text matches the post-fix code exactly (`derive_prefill` + `prefill_steps * num_envs`); no fixed item (P1/P2/P3/P5/P8) is still declared live |
| `configs/models/dreamer_srl/*.yaml` (18, WP-GAMMA — held, not this package) | γ 0.996840347 → 0.996996996996997 | ✅ | Re-confirmed intact: `grep 0.996840347` → 0 hits; all 18 modified files carry the corrected value (`agent_xs.yaml` carries no gamma key — out of WP-GAMMA scope, correctly untouched) |

**Independent re-runs (this verification, same machine):**
- `tests/algorithms/dreamer_srl/` → **129 passed, 2 skipped** (matches report; +19 over the 110 baseline = exactly the new/extended WP-SRL tests).
- Spot-run `test_env_independent_buffer.py` + `test_grad_clip.py -v` → 10/10 passed.
- Full project suite (`tests/` minus `tests/models/` minus dreamer_srl, re-run separately above) → **269 passed, 8 failed, 492 skipped** — the 8 failures are exactly the declared known-red baseline (4× `test_unified_parity` observability gates, 3× stale-config `FileNotFoundError`, 1× offline-WM smoke). Zero new failures.
- **P8 GPU-guard CLI check (the developer's flagged gap — now closed)**: fresh end-to-end invocation with `--buffer-device gpu --num-envs 4` raises the exact `ValueError` at `dreamer_srl_main.py:717`.
- Cross-module import surface swept: every external importer of `dreamer_srl` (`scripts/dreamer/*`, fixture generators) uses symbols this package kept or added — nothing orphaned.

**P5 live-smoke arithmetic re-derived (plausible):** both compared episodes fall inside the 1024-iter random-action prefill at the same seed, so the pre/post trajectories are identical by construction and the counter deltas isolate P5. Δlen = 11→10 = the leaked +1 step; Δrew = −325.005 − (−200.499) = −124.506 = episode 1's terminal-step reward (−100 death penalty + ≈ −24.5 of per-step reward, consistent with episode 2's ≈ −20/step average). Exactly the predicted leak signature.

**Speed verdict: ✅ no regression.** The clean stash/pop back-to-back protocol (same node 102 / GPU 1 / seed / config / step budget, `nvidia-smi`-confirmed idle at 2 MiB before each launch) is the methodologically sound pair: 6.21→6.12 SPS (−1.4%) at num_envs=1 and 28.75→30.21 (+5.1%, i.e. *faster*) at num_envs=4 — both within run-to-run noise, corroborated by the buffer-only microbench (+0.36 ms/iter ≈ 0.3% of a ~133 ms iteration). One caveat noted for the record: the contaminated first-pass "before" (41.5 SPS) is *higher* than the clean before (28.75) — co-resident-job contention alone cannot speed a run up, so that first measurement was anomalous in itself (clock/measurement-window artifact under the shared-GPU episode); this does not affect the verdict, which rests entirely on the clean pair, but it reinforces that only back-to-back idle-confirmed pairs are decision-grade on shared nodes.

**Stash-recovery integrity:** working tree matches the plan's file list exactly (`git diff --stat` — no file outside `src/algorithms/dreamer_srl/`, `tests/algorithms/dreamer_srl/`, `DEVIATION_LOG.md`, and this plan doc, apart from the pre-declared WP-GAMMA configs, `train_command-agent.sh`, and diary files); `git stash list` carries no leftover WP-SRL stash (the four existing stashes predate this session on other work); the 129-green suite re-run above is the functional proof nothing was lost.

**Out-of-scope/unexpected files:** none beyond the pre-declared set. `fix_note_gamma_config.md` (untracked) is WP-GAMMA's own note (experiment-designer scope), not this package's.

**Follow-ups (non-blocking):**
1. Post-merge, route to `bug-curator`: close the open KNOWN_BUGS gradient-clipping row (all three optimizers fixed, WP-SRL P1).
2. Minor residual test gap (accepted, matches the plan's contract): P3/P5 driver-side *routing* is not source-guarded the way P7 is — a hypothetical revert of `train.py` to an inline loss or of the driver to an unconditional counter increment would not trip the unit fixtures (which exercise the new symbols directly). The structural eliminations + this record are the guard; no action required now.

**Conclusion**: All 9 plan items (P1/P3/P5/P6/P7/P2/P8 fixed, P9+P7-cadence declared as D-016/D-015) implemented exactly per the plan with zero out-of-scope edits, both true behavioral reds independently reproduced, both suites re-run green at the reported counts, GPU-guard gap closed by CLI check, speed clean — **verified, ready to commit**. Any run trained after this package + WP-GAMMA lands is a new comparability epoch; do not mix with pre-fix runs.

Verified by: senior-developer

## Implementation Report — Addendum (2026-07-08, review follow-up C1/N1/N2)

Follow-up patch to the committed parity epoch (8c0fcf9), closing the three non-blocking
findings from [[review_srl_parity_fixes]] (`docs/reviews/review_srl_parity_fixes.md`).

**C1 (🟡, curriculum-swap counter edge).** On a stage-transition iteration the swap block
wipes ALL envs' episode counters (step 4, `dreamer_srl_main.py` ~:1436), but the
end-of-iteration `_advance_episode_counters` then ran with the pre-swap `rewards`/`dones`,
crediting the pre-swap step's +1 length and reward to non-done envs' freshly wiped counters
(the new stage's first episode). Fix (reviewer's suggested flag variant): the driver tracks
`_stage_swapped_this_iter` (init False each iteration before the done block, set True in
swap step 4) and passes it to `_advance_episode_counters`, which gained a keyword-only
`stage_swapped: bool = False` and no-ops when True. The pre-swap step's credit is skipped
consistently — matching the behavior/dist accumulators, which are wiped at the same point
and receive no post-wipe credit. Non-swap iterations are byte-identical in behavior.

**N1 (🟢, fail-fast buffer guard).** New module-level `validate_per_env_capacity()` in
`buffers.py` raises `ValueError` (naming per-env capacity, configured `buffer.size`,
`num_envs`, seq_len, and the minimum viable size) when `buffer_size // num_envs < seq_len`;
the driver calls it right after computing `per_env_buffer_size` (step 7), before
constructing either buffer class (covers both the wrapper and the GPU single-env path) —
previously the misconfiguration passed `ready_to_sample()` once wrapped-full and crashed
only at the first post-prefill `sample()` (`buffers.py:367-371`).

**N2 (🟢, docstrings).** Three docstring/comment-only annotations in `agent.py`
(`MLPDecoder.__call__` Returns ~:1283, `observe()` Returns ~:1655, decoder call site
~:1712): `reconstructed_obs` is the **symlog-space** prediction post-P3; apply `symexp`
for real-space values (cf. `loss.SymlogDistribution.mode/mean`). No behavior change.

**File changes** (all within the flagged scope; nothing else touched):
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` — C1 flag + guarded advance; N1 guard call + import.
- `src/algorithms/dreamer_srl/buffers.py` — N1 `validate_per_env_capacity()`.
- `src/algorithms/dreamer_srl/agent.py` — N2 docstrings/comment only.
- `tests/algorithms/dreamer_srl/test_episode_metrics.py` — +`test_stage_swap_iteration_skips_advance`.
- `tests/algorithms/dreamer_srl/test_env_independent_buffer.py` — +`test_per_env_capacity_guard`, +`test_deferred_crash_without_guard`.

**Red→green evidence** (red-first discipline; logs: `tmp/20260708_c1n1_red_prefix.log`,
`tmp/20260708_c1n1_green_postfix.log`):
- C1: pre-fix `test_stage_swap_iteration_skips_advance` **failed** —
  `TypeError: _advance_episode_counters() got an unexpected keyword argument 'stage_swapped'`
  (the advance at :1592 was unconditional; old-code behavioral value documented in the test:
  alive env ends the swap iteration with length=1/reward=3.0 instead of 0/0.0). Post-fix: passes.
- N1: pre-fix `test_env_independent_buffer.py` **failed at collection** — `ImportError:
  cannot import name 'validate_per_env_capacity'` (red by absence, same convention as the
  package's P2/P8 tests). The deferred-crash pathology is additionally pinned behaviorally:
  `test_deferred_crash_without_guard` shows a wrapped-full capacity-4 buffer passing
  `ready_to_sample(8)` yet raising on `sample(sequence_length=8)`. Post-fix: both pass.

**Test results** (CPU, `grid_world_pain` env, 2026-07-08):
- Targeted: `test_episode_metrics.py` + `test_env_independent_buffer.py` → **13 passed** (10 prior + 3 new).
- Full `tests/algorithms/dreamer_srl/` → **132 passed, 2 skipped** (129-green baseline + 3 new; zero regressions; the 8 known-red rows live outside this directory and were not re-run — unaffected scope).

**Speed check: skipped, with rationale.** The N1 guard is a startup-only integer
comparison (before any env step); C1 adds one bool kwarg + early-return branch to a
host-side numpy helper called once per driver iteration (sub-microsecond vs the ~130 ms
iteration measured for the parent package); N2 is docstring-only. No JIT boundary, hot-path
tensor op, or per-step model code is touched. Flagged here for senior-developer to confirm
the skip during verification.

**Deviations from the flagged scope:** none. The C1 fix uses the reviewer's suggested
flag-skip variant (not the wipe-reorder variant) because it keeps the swap block's wipes
co-located and makes the skip testable through the module-level helper. Working tree
otherwise clean apart from the pre-existing `train_command-agent.sh` modification and the
`docs/diary/2026-07-02.md` untracked file (both pre-declared, untouched). Not committed —
ready for senior-developer verification.

Implemented by: developer
