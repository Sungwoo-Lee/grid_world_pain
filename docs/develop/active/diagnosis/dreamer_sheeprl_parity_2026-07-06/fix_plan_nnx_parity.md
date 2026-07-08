---
title: "WP-NNX fix plan — align DreamerV3-NNX to the canonical DreamerV3 recipe (U1–U6 + is_first collection fix)"
topic: diagnosis
status: active
created: 2026-07-08
last_updated: 2026-07-08
---

# WP-NNX: DreamerV3-NNX recipe-alignment fix plan (U1–U6 + is_first)

> **Status**: PLANNED
> **Opened**: 2026-07-08
> **Related**: [[00_master_comparison]] (§4, §6 item WP-nnx) · [[05_dreamer_v3_nnx_conventions]] (full audit detail + probe evidence) · [[KNOWN_BUGS]] (is_first row = K1) · parallel work package: WP-SRL (dreamer_srl port fixes, `src/algorithms/` — separate plan, separate comparability epoch)

---

## Context

The project's **in-house DreamerV3 agent** (the JAX/Flax-NNX world-model agent trained by `train.py`, living in `src/models/dreamer_v3_*`) was re-audited against the canonical DreamerV3 recipe (the vendored sheeprl reference + Hafner 2023) on 2026-07-06→08. The audit found the loss *arithmetic* healthy but confirmed **six undeclared deviations with plausible training impact** plus one long-known-open bug. In plain terms: (1) the actor is not trained with the pure REINFORCE gradient the recipe prescribes — critic-loss and world-model-dynamics gradients leak into it through undetached imagined features (U1, empirically probed); (2) the observation-reconstruction loss is under-weighted by roughly the observation dimension, ~40–60× (U3); (3) the agent never resets its belief state at episode boundaries during data collection — the `is_first` flag is staged but consumed as a hardcoded zero (known-open K1); (4) value learning is DreamerV2-style — λ-returns bootstrap from the slow EMA critic instead of the live one, and the recipe's slow-critic regularizer is missing (U2); (5) the `replay_ratio` knob trains ~1/128 of what the same number means in sheeprl, and there is no random-action prefill (U4); (6) rollouts imagined from death rows are fully weighted (U5); and (7) the decoder ends in a nonstandard LayerNorm (U6).

This plan packages all seven fixes for the `developer` agent, in two internal phases, with a per-fix regression test that must be **red before the fix and green after**. The user gave blanket approval, but **two items carry consequences the parent must rule on at the gate**: U4's compute cost (this plan recommends a cost-neutral semantic fix + config rescale, §"U4 recommendation") and U6's checkpoint break (this plan recommends fix-now, §"U6 recommendation"). Fixing any of these changes training: post-fix runs open a **new NNX comparability epoch** (see §Comparability).

## Analysis

Root causes, probe evidence, and reference-line citations are already fully derived in [[05_dreamer_v3_nnx_conventions]] (items U1–U6, K1) — this plan does not re-derive them. Summary table with fix IDs used throughout this plan:

| Fix | Audit item | Phase | One-line problem | Ours | sheeprl ref |
|---|---|---|---|---|---|
| F1 | K1 (is_first) | 1 | Collection never resets belief state; `get_action` hardcodes `is_first = 0`, carries stale `prev_action`/`mod_h` across resets | `dreamer_v3_trainer.py:566` (zeros), `:562`, `:594-596`; staged-but-unread flag at `:697` | `agent.py:643-659` (`player.init_states`), `dreamer_v3.py:657` |
| F2 | U1 | 1 | Missing stop-gradients at 3 imagined feat/action sites → critic-loss + dynamics grads contaminate the actor | `trainer:379-381, 411-413, 460, 470-472` | `dreamer_v3.py:219,240,273,286,307` |
| F3 | U3 | 1 | Obs loss `mean` over feature dims (recipe: **sum**) → recon under-weighted ≈ obs_dim | `trainer:242` | `loss.py:61` |
| F4 | U2 | 2 | λ-return bootstrap from **target** critic (recipe: online) + slow-critic regularizer term absent | `trainer:400/421/444`, `:459-463` | `dreamer_v3.py:244, 307-316` |
| F5 | U5 | 2 | Imagination discount weights start at 1; no true-continue override for the source row | `trainer:455` | `dreamer_v3.py:247-248, 260` |
| F6 | U6 | 2 | Decoder trailing LayerNorm (recipe: bare Linear). **Changes the param tree** | `dreamer_v3_nnx.py:224-227` | `agent.py:274-278` |
| F7 | U4 | 2 | `replay_ratio` counted per **sequence** (1/128 of sheeprl's per-env-step meaning) + no random-action prefill | `train.py:1804-1809`; `train.py:844` | sheeprl yaml:16-17, `dreamer_v3.py:510-511, 563` |
| F8 | — | 2 | This stack has no deviation log (it was never a port); kept-not-fixed items need a home | — | — |

All line numbers are pre-change; the developer should locate by the code snippets in File Changes, not raw line numbers (earlier edits shift later lines).

### Phase-2 keep/fix/defer recommendations (decision items for the parent)

#### U2 (F4) — **FIX**
Small, local to `behavior_loss_fn`, and the recipe is unambiguous (online critic for all imagination values; slow critic only as a regularizer). No cost, no interface change. One interaction to note: after this fix the target critic's *only* job is the regularizer, which makes the known-open K6 (target critic initialized fresh-random, never τ=1-copied from the online critic) slightly more relevant early in training — benign while `zero_init_reward_critic: true` (the live setting); K6 stays open in the registry, not expanded here.

#### U4 (F7) — **FIX SEMANTICS, RESCALE CONFIG TO COST-NEUTRAL (option a) — recommended**
Two options were weighed:

- **(a) Recommended: fix the semantics AND rescale the config value so today's effective intensity is preserved.** Change the counting so `replay_ratio` means what it means in sheeprl (gradient steps per env step), and hand off a config change that divides every live `replay_ratio` value by `collect_interval` (128). **Zero compute-cost change; the knob becomes honest**; recipe intensity becomes reachable by simply raising the (now honestly-scaled) value. Also add the missing random-action prefill (`learning_starts`), which is cheap (1024 env steps once, ~1 collect iteration).
- **(b) Full recipe intensity at current config values: rejected on cost.** At `collect_interval=128`, `batch=16`, `seq=128`, today's `replay_ratio: 0.5` performs ≈ 0.0039 gradient steps per env step (≈ 8 replayed transitions/env-step; the WandB metric `Params/effective_replay_ratio` confirms ~0.004 on live runs). sheeprl's meaning of 0.5 is 0.5 grad steps per env step (≈ 512 replayed transitions/env-step) — **128× more gradient steps**. Each gradient step pushes 2048 timesteps through world-model + imagination; gradient work already dominates iteration wall-time on the lab 3090s. Lower bound on the slowdown: if a fraction *f* of current iteration time is gradient work, wall time multiplies by ≈ (1−f) + 128·f — even at f = 0.3 that is ~39×, at f = 0.7 it is ~90×. A 2-day run becomes multi-month. Not launchable on this cluster as a default.

Under (a), one config deserves an explicit exception: `dreamer_v3_sheeprl_matched.yaml` exists precisely to match sheeprl's intensity — its `replay_ratio: 1.0` should **stay 1.0** (after the semantic fix it finally does what its comments claim; anyone launching it must accept the ~128× gradient-step cost knowingly). **All config edits route to `experiment-designer`, not the developer** — exact key/value table in §Config handoff.

#### U5 (F5) — **FIX**
Three-line change plus threading one array; recipe unambiguous; crisp regression test (all-terminal start rows ⇒ zero behavior loss).

#### U6 (F6) — **FIX NOW — recommended (accepting the checkpoint break)**
Removing the decoder's trailing LayerNorm deletes two parameters from the NNX param tree, so **checkpoints saved before this fix will not restore into post-fix code** (tree-structure mismatch at restore). Recommendation: fix now, because (i) Phase 1 (U1/U3/F1) already breaks run-to-run comparability this epoch — deferring U6 would spend a *second* comparability epoch later for a two-line change; (ii) the checkpoint break costs less than it sounds: per known-open K4, NNX checkpoints already omit all 3 optimizer states, the target critic, and the Moments EMA, so restores were never faithful resumes; (iii) batching all recipe fixes into one epoch is exactly the master doc's guidance (§5 item 3). **Follow-ups if fixed now (enumerated, mandatory):** (1) after landing, the parent asks `bug-curator` to record a registry note "NNX DreamerV3 checkpoints saved before <commit> cannot be restored by post-<commit> code (decoder param-tree change, WP-NNX F6)"; (2) row U6 = FIXED in the F8 deviation register; (3) append a line to the checkpoint-compat caveat in `docs/develop/active/dreamer/dreamer_v3_implementation.md` §6 (the doc the live YAML comments cite) if that section exists — otherwise note in the Implementation Report where the caveat landed.

#### F8 — deviation register: **new area-style doc in this folder — recommended**
This stack never had a DEVIATION_LOG (it was never a port). Rather than bloating the master comparison doc, create **`06_nnx_recipe_deviation_register.md`** next to the five area reports (fits the folder's numbering; the master doc §7 links it). Exact rows in File Changes.

## Implementation Plan

### Design

- **Internal sequencing** (both `dreamer_v3_trainer.py` and `dreamer_v3_nnx.py` are shared across phases — one fix at a time, one commit per fix, test authored first and shown red):
  **F1 → F2 → F3** (Phase 1) **→ F4 → F5 → F6 → F7 → F8** (Phase 2).
  F2 and F4/F5 all edit `behavior_loss_fn`; doing F2 first keeps the stop-gradient baseline in place when F4 moves the value source to the online critic. F1 is independent (get_action / collect_sequence). F3 is one line in `model_loss_fn`. F6 touches only `dreamer_v3_nnx.py`. F7 touches only the DreamerV3 branch of `train.py` + `collect_sequence`.
- **F1 mechanism (the established 2-line shape + mod_h decision):** `get_action` consumes the staged `prev_state['is_first']` instead of hardcoded zeros, and zeroes `prev_action` at flagged rows — the analog of sheeprl `player.init_states` (vendor `agent.py:643-659`), which zeroes actions and resets latents per done env. The existing `RSSM.step` mask (`dreamer_v3_nnx.py:105-119`, `deter/stoch × (1 − is_first)`) then handles the latent reset. **mod_h decision:** reset the modulator hidden state the same way, via `jnp.where(is_first > 0.5, modulator.initial_state(B), mod_h)` *before* `forward_obs` consumes it — the `where` form is used (rather than multiplying by the mask) so it stays correct even if `initial_state` ever becomes non-zero; today both modulator classes return zeros, so it is equivalent to masking. The staging side (`collect_sequence:697`) is already correct and stays untouched; the stored `transition['is_first']` (`:705`) is already correct.
- **F2 mechanism:** detach imagined features **at creation** inside both `scan_imag` branches (one `stop_gradient` per branch severs actor input, `step_info['feat']` storage, and the modulator's imagination input in one place), and detach the ST action sample where the log-prob consumes it. This reproduces sheeprl's three detach sites while keeping the ST gradient alive *inside* the RSSM/imagination itself, exactly as sheeprl does. `behavior_loss_fn` grads are w.r.t. (actor, critic) only, so detaching feat costs nothing elsewhere.
- **Scope fence (hard):** the developer may edit **only** `src/models/dreamer_v3_trainer.py`, `src/models/dreamer_v3_nnx.py`, `src/models/dreamer_v3_util.py` (if needed for helpers), the **DreamerV3-only sections of `train.py`** named in F7, new/changed tests under `tests/models/`, and the F8 register doc (+ the two doc follow-ups named in U6). **No `configs/` edits** (→ `experiment-designer`, §Config handoff). **No `src/algorithms/`** (parallel WP-SRL owns it). Nothing else.
- **Config keys:** F7 introduces `agent.learning_starts` read via `config.get_mandatory('agent.learning_starts', int)` — no fallback default, per the Configuration Protocol. The key lands in YAML via the `experiment-designer` handoff **before** the F7 commit is exercised (coordinate: the developer's F7 smoke run needs the key present; if the handoff hasn't landed, the developer smokes with a local *uncommitted* config edit and says so in the Implementation Report — committed config changes remain experiment-designer's).

### File Changes

#### F1 — `src/models/dreamer_v3_trainer.py`, `get_action` (pre-change lines 558–596)

```python
# BEFORE (lines 558-566):
        if 'prev_action' not in prev_state:
            prev_state['prev_action'] = jnp.zeros(
                (B, self.agent.ac.actor.net.layers[-1].out_features))

        prev_action = prev_state['prev_action']
        obs_symlog = symlog(obs)

        key = random.split(rng)[0] if rng is not None else random.PRNGKey(0)
        is_first = jnp.zeros((B, 1))

# AFTER:
        if 'prev_action' not in prev_state:
            prev_state['prev_action'] = jnp.zeros(
                (B, self.agent.ac.actor.net.layers[-1].out_features))

        # Episode-boundary reset (WP-NNX F1 / registry K1): consume the is_first
        # flag staged by collect_sequence (analog of sheeprl player.init_states,
        # vendor agent.py:643-659, which zeroes actions and resets latents per
        # done env). RSSM.step masks deter/stoch itself (dreamer_v3_nnx.py:117-119);
        # here we zero the stale prev_action (and reset mod_h below).
        is_first = prev_state.get('is_first', jnp.zeros((B, 1)))
        prev_action = prev_state['prev_action'] * (1.0 - is_first)
        obs_symlog = symlog(obs)

        key = random.split(rng)[0] if rng is not None else random.PRNGKey(0)
```

(The hardcoded `is_first = jnp.zeros((B, 1))` at old line 566 is removed — `is_first` now flows into `rssm.step(...)` at old line 594-596 unchanged in call shape.)

```python
# BEFORE (lines 568-572, modulated branch):
        if modulation_enabled:
            if 'mod_h' not in prev_state:
                mod_h = self.agent.wm.modulator.initial_state(B)
            else:
                mod_h = prev_state['mod_h']

# AFTER:
        if modulation_enabled:
            if 'mod_h' not in prev_state:
                mod_h = self.agent.wm.modulator.initial_state(B)
            else:
                # WP-NNX F1: modulator state resets at episode boundaries too.
                # jnp.where (not mask-multiply) so this stays correct if
                # initial_state ever becomes non-zero. is_first is (B, 1) and
                # broadcasts against (B, mod_hidden).
                mod_h = jnp.where(is_first > 0.5,
                                  self.agent.wm.modulator.initial_state(B),
                                  prev_state['mod_h'])
```

Also update the stale comment block at `collect_sequence` old lines 692-696 ("but here get_action handles it" — after F1 it actually does; say so).

#### F2 — `src/models/dreamer_v3_trainer.py`, `behavior_loss_fn` (pre-change lines 375–478)

Three detach sites, mirroring sheeprl `dreamer_v3.py:219,240,273` (actor input), `:307` (critic input), `:286` (log-prob action):

```python
# BEFORE (modulated scan_imag, lines 378-380 — same shape in the plain branch at 410-412):
                    prev_state, h_mod = carry
                    feat = self.agent.wm.get_feat(prev_state)
                    actor_out = actor(feat)

# AFTER (BOTH branches):
                    prev_state, h_mod = carry
                    # WP-NNX F2 (U1): actor, critic, modulator, and step_info
                    # consume imagined features DETACHED (sheeprl dreamer_v3.py:
                    # 219,240,273,307) — discrete-action DreamerV3 trains the
                    # actor by REINFORCE only; no dynamics backprop, no
                    # critic-loss leak. The ST gradient stays alive inside the
                    # RSSM imagination chain itself, as in sheeprl.
                    feat = jax.lax.stop_gradient(self.agent.wm.get_feat(prev_state))
                    actor_out = actor(feat)
```

```python
# BEFORE (line 460):
                v_pred_logits = critic(rollouts['feat'])
# AFTER (redundant with the creation-site detach, kept explicit to mirror D:307):
                v_pred_logits = critic(jax.lax.stop_gradient(rollouts['feat']))
```

```python
# BEFORE (line 470):
                actions = rollouts['action']
# AFTER (sheeprl D:286 — log_prob of the DETACHED action; kills the spurious
# ∇probs term riding the ST sample):
                actions = jax.lax.stop_gradient(rollouts['action'])
```

#### F3 — `src/models/dreamer_v3_trainer.py`, `model_loss_fn` (pre-change line 242)

```python
# BEFORE:
                loss_recon = jnp.mean(jnp.square(recon - obs))
# AFTER (recipe: SUM over the feature/event dim, mean over batch & time —
# sheeprl loss.py:61 via SymlogDistribution; WP-NNX F3 / U3):
                loss_recon = jnp.mean(jnp.sum(jnp.square(recon - obs), axis=-1))
```

#### F4 — `src/models/dreamer_v3_trainer.py`, `behavior_loss_fn` (pre-change lines 400, 421, 444, 459-463)

λ-bootstrap values from the **online** critic (sheeprl D:244); slow critic becomes a **regularizer** (D:307-316):

```python
# BEFORE (lines 400 and 421 — both scan_imag branches):
                    val = from_twohot(self.target_critic(next_feat), paper_canonical_bins=self._paper_canonical_twohot_bins)
# AFTER:
                    val = from_twohot(critic(next_feat), paper_canonical_bins=self._paper_canonical_twohot_bins)

# BEFORE (line 444):
                v_start = from_twohot(self.target_critic(start_feat), paper_canonical_bins=self._paper_canonical_twohot_bins)
# AFTER:
                v_start = from_twohot(critic(start_feat), paper_canonical_bins=self._paper_canonical_twohot_bins)
```

(All consumers of these values — critic target, advantage, discount weights — are already stop-gradient-protected, so routing them through the differentiable `critic` arg adds no live gradient path; this matches sheeprl, which also computes them undetached and detaches at consumption.)

```python
# BEFORE (lines 459-463):
                # Critic Loss — train on RAW lambda_returns (canonical DreamerV3)
                v_pred_logits = critic(jax.lax.stop_gradient(rollouts['feat']))
                target_twohot = to_twohot(jax.lax.stop_gradient(lambda_returns), paper_canonical_bins=self._paper_canonical_twohot_bins)
                loss_critic_step = -jnp.sum(target_twohot * jax.nn.log_softmax(v_pred_logits), axis=-1)
                loss_critic = jnp.mean(loss_critic_step * discount_weights)

# AFTER:
                # Critic Loss — RAW lambda_returns + slow-critic regularizer
                # (WP-NNX F4 / U2; sheeprl dreamer_v3.py:307-316: the EMA critic
                # no longer supplies bootstrap values — it regularizes the online
                # critic toward its own predictions instead).
                v_pred_logits = critic(jax.lax.stop_gradient(rollouts['feat']))
                target_twohot = to_twohot(jax.lax.stop_gradient(lambda_returns), paper_canonical_bins=self._paper_canonical_twohot_bins)
                slow_vals = from_twohot(self.target_critic(jax.lax.stop_gradient(rollouts['feat'])), paper_canonical_bins=self._paper_canonical_twohot_bins)
                slow_twohot = to_twohot(jax.lax.stop_gradient(slow_vals), paper_canonical_bins=self._paper_canonical_twohot_bins)
                loss_critic_lambda = -jnp.sum(target_twohot * jax.nn.log_softmax(v_pred_logits), axis=-1)
                loss_critic_slow_reg = -jnp.sum(slow_twohot * jax.nn.log_softmax(v_pred_logits), axis=-1)
                loss_critic_step = loss_critic_lambda + loss_critic_slow_reg
                loss_critic = jnp.mean(loss_critic_step * discount_weights)
```

Add to the behavior metrics dict: `'loss_critic_slow_reg': jnp.mean(loss_critic_slow_reg * discount_weights)`.

#### F5 — `src/models/dreamer_v3_trainer.py`, `train_step` + `behavior_loss_fn` (pre-change lines ~361 and 455)

In `train_step`, next to the `start_state` construction (old line 361-362), build the source rows' **true continue** from the replay batch (`term_reason` is already in scope — `model_loss_fn` closes over it at old lines 246/257) and flatten it with the same `(B, T) → (B·T,)` reshape as `start_state`:

```python
        # WP-NNX F5 (U5): true continue of each imagination source row
        # (sheeprl dreamer_v3.py:247-248 — continues[0] = 1 - terminated).
        true_cont0 = jax.lax.stop_gradient(
            compute_continue_target(term_reason).reshape(-1))   # (B*T,)
```

```python
# BEFORE (line 455):
                discount_weights = jnp.concatenate([jnp.ones_like(conts[:1]), conts[:-1] * GAMMA], axis=0)
# AFTER (weight row 0 = the source row's TRUE continue, not 1 — rollouts
# imagined from death rows carry zero weight; sheeprl D:247-248,260):
                discount_weights = jnp.concatenate([true_cont0[None] * jnp.ones_like(conts[:1]), conts[:-1] * GAMMA], axis=0)
```

`true_cont0` reaches `behavior_loss_fn` by closure (it is stop-gradiented and non-differentiable — no interface change needed).

#### F6 — `src/models/dreamer_v3_nnx.py`, `DreamerGroupedMLP.__init__` (pre-change lines 224-227)

```python
# BEFORE:
        layers.append(DreamerGroupedLinear(num_groups, in_d, output_dim, rngs=rngs))
        layers.append(nnx.LayerNorm(output_dim, rngs=rngs))
        # Note: No final SiLU here, similar to Encoder.body
        self.net = nnx.Sequential(*layers)

# AFTER (WP-NNX F6 / U6: recipe decoders end in a bare Linear — sheeprl
# agent.py:274-278. The trailing LayerNorm was copied from the ENCODER body,
# where it belongs; on a reconstruction head it pins the pre-affine output
# scale and couples features within a sensor group.
# ⚠ PARAM-TREE CHANGE: checkpoints saved before this commit will NOT restore.):
        layers.append(DreamerGroupedLinear(num_groups, in_d, output_dim, rngs=rngs))
        self.net = nnx.Sequential(*layers)
```

`DreamerGroupedMLP` is used by the hierarchical decoder (`:434`) **and** — verify before assuming — possibly by the hierarchical *encoder* path. **Checkpoint the actual use sites first** (`grep -n "DreamerGroupedMLP" src/models/dreamer_v3_nnx.py`): if the encoder also instantiates this class, the trailing LN must become a constructor flag (`final_layer_norm: bool`) so the **encoder keeps it** (recipe encoders DO end their body in LN+act) and only the decoder instantiation passes `False`. Do not silently change the encoder.

#### F7 — `train.py` DreamerV3 branch (pre-change lines 844, 1524-1527, 1750, 1804-1809) + `src/models/dreamer_v3_trainer.py` `collect_sequence`

1. **Ratio semantics** (the U4 core):

```python
# BEFORE (line 1809, with the comment block at 1805-1808):
                            train_steps = ratio_scaled_updates(global_step // num_steps)
# AFTER (WP-NNX F7 / U4: replay_ratio now means what it means in sheeprl —
# gradient steps per ENV STEP (global, matching sheeprl's per-policy-step
# Ratio; train.py increments global_step by num_envs * num_steps per iter).
# Config values were rescaled ÷ collect_interval in the same change window
# to keep effective intensity unchanged — see fix_plan_nnx_parity.md §Config handoff):
                            train_steps = ratio_scaled_updates(global_step)
```

Rewrite the 1805-1808 comment accordingly. `Ratio` (`dreamer_v3_util.py:187-217`) already carries fractional remainders correctly (`_prev += repeats / ratio`), so values like 0.00390625 accumulate exactly. **Confirm `agent.train_steps` (yaml:7) has no live interaction with this call site** (the sheeprl_matched yaml comment suggests it is display-only); report the finding.

2. **Random-action prefill / learning_starts:** near line 844 read `learning_starts = config.get_mandatory('agent.learning_starts', int)` (env steps, global — sheeprl Y:17 uses 1024, prefilled with `action_space.sample()`, D:510-511,563). In the Dreamer iteration loop: while `global_step < learning_starts`, collect with random actions and skip training; also add `global_step >= learning_starts` to the train gate at 1804. In `collect_sequence` (trainer, pre-change line 617-731) add a `random_actions: bool = False` **static** flag (extend `@nnx.jit(static_argnums=...)` — one extra compile for the prefill variant, then never again). Inside `scan_fn`, after `get_action`:

```python
            if random_actions:   # static flag — resolved at trace time
                current_key, rand_key = jax.random.split(current_key)
                action_idx = jax.random.randint(rand_key, (B,), 0, act_dim)
                # keep the RSSM carry's prev_action consistent with the action
                # actually executed (sheeprl's player pairs likewise):
                next_d_state['prev_action'] = jax.nn.one_hot(action_idx, act_dim)
```

where `act_dim = self.agent.ac.actor.net.layers[-1].out_features`. The rest of the transition dict is unchanged (it already derives from `action_idx`).

#### F8 — NEW DOC `docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/06_nnx_recipe_deviation_register.md`

Frontmatter (`topic: diagnosis`), plain-language entry point ("what this register is: the NNX stack's analog of the port's DEVIATION_LOG — every audited recipe deviation that we consciously KEEP, plus the fixed rows' disposition"), then one table:

| Row | Item (from [[05_dreamer_v3_nnx_conventions]]) | Status after WP-NNX |
|---|---|---|
| U1, U3, K1 | stop-gradients / obs-loss sum / is_first collection reset | FIXED — commit refs (developer fills) |
| U2, U4, U5, U6 | online bootstrap + slow reg / replay-ratio semantics + prefill / true-continue weights / decoder LN | FIXED — commit refs (developer fills; U4 notes the config rescale) |
| C1 | LayerNorm eps 1e-6 vs recipe 1e-3 | KEPT — cosmetic; flax default |
| C2 | `hafner_init` effective std ≈ 0.77× recipe; trunc-normal on heads sheeprl gives uniform | KEPT — init-time only |
| C3 | zeros initial recurrent state vs learnable `tanh(param)` | KEPT — interacts with E2 equivalence; revisit only if learnable init is ever added |
| C4 | no pre-GRU projection LayerNorm | KEPT — cosmetic |
| C5 | dead `agent.unimix` YAML knob (hardcoded 0.01) | KEPT — flag to experiment-designer for eventual config cleanup |
| C6 | actor/critic lr 3e-5 vs 8e-5; seq_len 128 vs 64 | KEPT — declared knobs, comparative-alignment choice |
| C7 | block-aligned sequence sampling + mixture pools | KEPT — declared DreamerV4-inspired extension |
| C8 | HORIZON/GAMMA/LAMBDA/FREE_NATS hardcoded constants | KEPT — recipe values; config-exposure is a separate refactor |
| K2–K7 | eval path, unimix placement, checkpoint omissions, collect_interval validation, target-critic init, PRNG hygiene | OPEN — owned by [[KNOWN_BUGS]]; referenced, not duplicated (K6 note: U2 makes the regularizer the target critic's only role) |

After creating it, run `python scripts/claude/regen_dev_index.py` and add the `[[06_nnx_recipe_deviation_register]]` link to [[00_master_comparison]] §7.

### Config handoff (parent → `experiment-designer`, NOT the developer)

Required for F7; should land in the same change window (before any post-fix launch):

| File (`configs/models/dreamer_v3/`) | Key | Before | After | Why |
|---|---|---|---|---|
| `dreamer_v3.yaml` | `agent.replay_ratio` | `0.5` | `0.00390625` | = 0.5/128 — cost-neutral rescale under the new per-env-step semantics |
| `dreamer_v3.yaml` | `agent.learning_starts` | — (new) | `1024` | sheeprl Y:17 random-action prefill; **mandatory key** (`get_mandatory`) |
| `neuromodulated_dreamer_v3.yaml` | `agent.replay_ratio` | `1` | `0.0078125` | = 1/128 |
| `dreamer_v3_probe.yaml`, `dreamer_v3_curriculum.yaml`, `dreamer_v3_curriculum_probe.yaml`, `dreamer_v3_probe_cont10.yaml` | `agent.replay_ratio` | `0.5` | `0.00390625` | same rescale |
| `dreamer_v3_rr06.yaml` | `agent.replay_ratio` | `0.0625` | `0.00048828125` | = 0.0625/128 |
| `dreamer_v3_sheeprl_matched.yaml` | `agent.replay_ratio` | `1.0` | **keep `1.0`** | this config's declared intent IS recipe intensity; post-fix it finally delivers it (⚠ ~128× more grad steps than it performed before — launching it becomes a deliberate, expensive act) |
| all of the above that do not `extends:` a file already carrying it | `agent.learning_starts` | — | `1024` | every live dreamer_v3 config must resolve the mandatory key |

Also update the stale comments in these YAMLs that describe the old per-sequence semantics (e.g. `sheeprl_matched` lines 22-63), and the `collect_interval: 1 = sheeprl-style (canonical)` comment (`dreamer_v3.yaml:6`) which K5 already flagged as misleading.

#### Config changes applied (2026-07-08, `experiment-designer`)

The handoff table above has been executed in full. The directory was enumerated independently: `configs/models/dreamer_v3/` contains exactly 8 YAMLs, all standalone (no `extends:` chains within the tree), all covered by the table — **nothing missed, nothing extra**. Because there is no shared base config, `agent.learning_starts: 1024` was added to every file individually.

| File (`configs/models/dreamer_v3/`) | Key | Old | New |
|---|---|---|---|
| `dreamer_v3.yaml` | `agent.replay_ratio` | `0.5` | `0.00390625` |
| `dreamer_v3_probe.yaml` | `agent.replay_ratio` | `0.5` | `0.00390625` |
| `dreamer_v3_probe_cont10.yaml` | `agent.replay_ratio` | `0.5` | `0.00390625` |
| `dreamer_v3_curriculum.yaml` | `agent.replay_ratio` | `0.5` | `0.00390625` |
| `dreamer_v3_curriculum_probe.yaml` | `agent.replay_ratio` | `0.5` | `0.00390625` |
| `dreamer_v3_rr06.yaml` | `agent.replay_ratio` | `0.0625` | `0.00048828125` |
| `neuromodulated_dreamer_v3.yaml` | `agent.replay_ratio` | `1` | `0.0078125` |
| `dreamer_v3_sheeprl_matched.yaml` | `agent.replay_ratio` | `1.0` | `1.0` (kept — declared intent is full recipe intensity; inline ⚠ comment added about the ~128× grad-step cost) |
| all 8 files | `agent.learning_starts` | — (new) | `1024` (mandatory key; inline comment notes `get_mandatory` crash-on-missing) |

Every changed `replay_ratio` line carries the comment: *semantics changed to per-ENV-STEP (sheeprl); value rescaled ÷128 to preserve prior effective training intensity.* Stale-comment cleanup also applied per the paragraph above: the misleading `collect_interval` comment (`1 = sheeprl-style (canonical)`, K5) replaced in the 6 files that carried it; `sheeprl_matched` header delta-block and `rr06` header both annotated that their prose/cost-model values refer to the old per-sequence counting.

Resolution check — every leaf config resolves both keys through `src.environment.config_loader.load_env_config` + `get_mandatory` (run 2026-07-08):

```
dreamer_v3.yaml                          replay_ratio=0.00390625  learning_starts=1024
dreamer_v3_curriculum.yaml               replay_ratio=0.00390625  learning_starts=1024
dreamer_v3_curriculum_probe.yaml         replay_ratio=0.00390625  learning_starts=1024
dreamer_v3_probe.yaml                    replay_ratio=0.00390625  learning_starts=1024
dreamer_v3_probe_cont10.yaml             replay_ratio=0.00390625  learning_starts=1024
dreamer_v3_rr06.yaml                     replay_ratio=0.00048828125  learning_starts=1024
dreamer_v3_sheeprl_matched.yaml          replay_ratio=1.0  learning_starts=1024
neuromodulated_dreamer_v3.yaml           replay_ratio=0.0078125  learning_starts=1024
```

### Regression tests (one file per fix, under `tests/models/`; reuse the tiny-trainer/env builder pattern from `tests/models/test_dreamer_collect_arrival_alignment.py:57-97`)

Contract for every test: **must fail on pre-fix code** (the developer demonstrates red by running the new test before applying the fix commit, and records the red output in the Implementation Report), then pass post-fix. Run command template:
`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/models/<file> -v`

| # | File | Red-pre-fix contract |
|---|---|---|
| T1 (F1) | `test_dreamer_nnx_is_first_reset.py` | (a) `get_action(obs, state_A)` where `state_A` has `is_first=1` and **garbage** `deter/stoch/prev_action` (and `mod_h` if modulated) returns bit-identical `(action, next_state)` to `get_action(obs, state_B)` where `state_B` is a fresh `rssm.initial(B)` state with `is_first=1`, same rng. Pre-fix: differs (stale `prev_action` and `mod_h` leak through; deter/stoch masked only if is_first were consumed — it isn't). (b) integration: `collect_sequence` across a forced episode boundary (timeout fixture as in the H6 tests) — the first post-done row's action distribution is invariant to the pre-done trajectory's actions. |
| T2 (F2) | `test_dreamer_nnx_stop_gradients.py` | Actor-gradient isolation, the area-report ST-chain probe as a test: on a tiny trainer, (a) `nnx.grad` of the **critic-loss component alone** w.r.t. **actor** params is exactly zero on every leaf (pre-fix: nonzero via the undetached `rollouts['feat']` → ST-action chain); (b) the actor-loss gradient w.r.t. actor params equals a reference computation in which imagined feats and sampled actions are explicitly detached (pre-fix: differs — the audit's probe numbers, e.g. `[-0.2486,…]` vs `[-0.2138,…]`, demonstrate the shape). To expose the components, the developer may extract `behavior_loss_fn` into a testable method (`_behavior_loss(self, actor, critic, rng, start_state, …)`) — mechanical, no numerical change; assert one pre-existing train_step smoke still passes bit-identically pre/post extraction. |
| T3 (F3) | `test_dreamer_nnx_obs_loss_sum.py` | On a fixture batch with hand-built `recon`/`obs` arrays of feature dim D (extract the recon-loss expression into a tiny helper, or recompute via the trainer's metrics on a controlled decoder), `loss_recon == mean_over_(B,T)(sum_over_D(sq_err))` — numerically = D × the old mean-form. Pre-fix: off by exactly ×D. |
| T4 (F4) | `test_dreamer_nnx_online_bootstrap.py` | (a) Bootstrap-source assertion: corrupt the target critic (add +1e3 to every param), rerun the behavior loss with identical rng — `mean_return`/`mean_value` metrics (λ-returns, baseline) are **bit-identical** to the uncorrupted run (post-fix they depend only on the online critic); pre-fix they shift. (b) Regularizer present: `loss_critic_slow_reg` metric exists, is ≥ 0, and equals a hand-computed two-hot CE between the online critic's logits and the (corrupted) target critic's symexp'd predictions on the same rollout feats — so it DOES move when the target critic moves. Pre-fix: KeyError. |
| T5 (F5) | `test_dreamer_nnx_terminal_start_weights.py` | Feed a replay batch whose every row has `termination_reason = 2` (real death): post-fix `loss_actor == 0` and `loss_critic == 0` exactly (row-0 weight 0 ⇒ cumprod ⇒ all weights 0); pre-fix both nonzero. Companion positive control: all-alive batch (`term_reason = 0`) gives nonzero losses post-fix. |
| T6 (F6) | `test_dreamer_nnx_decoder_head.py` | Structural: the hierarchical decoder's `unimodal_grouped_decoder.net.layers[-1]` is a `DreamerGroupedLinear` (pre-fix: `nnx.LayerNorm`); if the shared-class flag route is taken, additionally assert the **encoder** instantiation still ends in LN. |
| T7 (F7) | `test_dreamer_nnx_replay_ratio_semantics.py` | (a) Semantics: simulate the call-site loop (Ratio(r), `global_step += num_envs*num_steps` per iter, `collect_interval=128`, `num_envs=1`): cumulative `train_steps` after N iters ≈ `r × global_step` (±1) — pre-fix it returns `r × global_step / 128`. Test the *actual* call-site arithmetic (extract the one-line gate+ratio call into a helper if needed for importability from `train.py`, or replicate the exact expression and pin it with a source-comment cross-ref). (b) Prefill: `collect_sequence(..., random_actions=True)` with an actor whose logits are forced degenerate (huge logit on action 0) produces >1 distinct action in 128 steps, and the carry's `prev_action` matches the executed actions; `random_actions=False` produces all-0 actions. Pre-fix: `TypeError` (flag doesn't exist). |

**Full-suite gate:** `…/python -m pytest tests/ -q --continue-on-collection-errors` after the last commit. **Known-red baseline (pre-existing, do NOT chase, do NOT count against this WP): 8 rows** — 4× `test_unified_parity[observability_gates_S1–S4]` (A1 parity fixtures), 3× stale-config `FileNotFoundError` in `test_inactive_animal_offgrid`/`test_truncation_not_death` (config retirement `b093023`), 1× `test_dreamer_srl_offline_wm_test::test_offline_wm_smoke`. Any *new* red must be investigated before reporting. ⚠ WP-SRL is editing `src/algorithms/dreamer_srl` in parallel — transient collection errors or flips in the srl smoke row may come from *their* uncommitted work tree; attribute before chasing (same situation occurred during H6/H7).

### Speed check (mandatory, per Verification Protocol)

Same hardware, same config, same seed, ≥300 iterations of the standard dreamer smoke, s/it before vs after the full stack of commits. Expectations: Phase 1 ≈ free; F4 adds one extra target-critic forward + CE per grad step (small); F7 under recommendation (a) is **compute-neutral by construction** — verify `Params/effective_replay_ratio` reads ≈ the (rescaled) config value and cumulative gradient steps per env step are unchanged within rounding. >5% slowdown ⇒ discuss; >15% ⇒ blocker.

### Comparability caveat

Landing this WP opens a **new comparability epoch for the NNX stack** ("NNX parity epoch, 2026-07 / WP-NNX"): every post-fix `dreamer_v3_nnx` run is incomparable with every pre-fix run (actor gradient composition, world-model loss balance, value-learning style, replay accounting, discount weighting, and — via F6 — the parameter tree itself all change). This epoch is **separate from and independent of the dreamer_srl epoch** that WP-SRL opens on `src/algorithms/dreamer_srl` — do not conflate the two when labeling runs. Additionally, F6 makes pre-fix NNX checkpoints **unrestorable** by post-fix code (see U6 recommendation follow-ups). Ongoing/queued NNX runs should finish or be abandoned before the merge; post-merge baselines must be re-established before any new NNX-vs-anything comparison.

## Checkpoints

- [x] 1. F1: T1(a) unit red → green (unmodulated + modulated `mod_h` variants); T1(b) boundary integration red → green; stale comment at `collect_sequence` updated.
- [x] 2. F2: T2(a) critic→actor gradient exactly zero on all leaves (red: max |g| ≈ 1.9e-4 → green: exact 0); T2(b) matches detached reference (red: 100% of elements mismatched → green); `_behavior_loss` extraction verified bit-identical on 2 train_steps × fixed batch+rng (pre/post `.npz` compare, all metrics `array_equal`). Note: the leak is invisible at init under the live `zero_init_reward_critic: true` — tests force random heads (see Implementation Report §F2).
- [x] 3. F3: T3 red (×40 / ×13 discrepancy) → green; eyeballed train_step: `loss_recon` 0.581 → 11.04 (≈ ×19 = fixture obs_dim), total WM loss finite, no NaN.
- [x] 4. F4: T4(a) target-corruption invariance bit-exact (red: `mean_return` −0.476 → −2.9e6 under corruption; green: bit-identical); T4(b) `loss_critic_slow_reg` in the metrics dict — auto-fans-out to WandB via the existing `mk.startswith('loss_critic')` match in train.py.
- [x] 5. F5: T5 all-terminal ⇒ behavior losses exactly 0 (red: loss_actor = −0.0515, loss_critic > 0); positive control nonzero pre+post.
- [x] 6. F6: **STOPPED per parent gate ruling.** Guard grep (`grep -n "DreamerGroupedMLP" src/models/dreamer_v3_nnx.py`) → instantiated by BOTH the encoder (`:259`) and the decoder (`:434`) ⇒ shared ⇒ parent's contingency instruction says halt and flag (the plan's constructor-flag route NOT taken). No code change, no T6, no checkpoint break. U6 recorded OPEN in the F8 register.
- [x] 7. F7: T7 both parts red → green; config handoff had already landed (all 8 dreamer_v3 YAMLs); offline-WandB smoke: `Params/effective_replay_ratio` ≈ rescaled config value, prefill + gate behavior verified (Implementation Report §F7).
- [x] 8. F8: register doc created with frontmatter + all rows (U6 row = OPEN/stopped, deviation from plan table per parent ruling); `regen_dev_index.py` run (168 docs); master doc §7 link added.
- [x] 9. Full suite: no new red beyond the known-red baseline (details + WP-SRL attribution in Implementation Report).
- [x] 10. Speed: before/after recorded — same GPU (local RTX 4090 cuda:0, back-to-back), same env config/seed/step budget, 300 iterations: BEFORE (pristine HEAD code + `replay_ratio` 0.5) 199 s; AFTER (post-fix code + 0.00390625 + prefill) 188 s ⇒ ~5.5% FASTER, no regression; effective replay intensity unchanged by construction. *(Verification note, senior-developer 2026-07-08: these wall-second figures do not match the valid A/B — the log artifacts `tmp/20260708_wpnnx_speedB2.log`/`speedA2.log` record `WALL_SECONDS=535` (BEFORE) and `WALL_SECONDS=448` (AFTER), i.e. −16%, as in the Implementation Report's Speed check table. The table is authoritative; the direction (post-fix faster, no regression) is unaffected.)*

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-08

### Summary

Six of the seven fixes landed (F1–F5, F7, F8); **F6 was STOPPED at its guard per the parent's gate ruling** (see §F6). Sequencing F1→F2→F3→F4→F5→(F6 stopped)→F7→F8 was followed, each fix with its regression test authored first and shown red on pre-fix code. **No commits were made** — the parent overrode the plan's one-commit-per-fix instruction and commits the package after verification; per-fix red→green evidence is recorded below instead.

Files changed:
- `src/models/dreamer_v3_trainer.py` — F1 (get_action is_first consumption + prev_action/mod_h reset, stale comment update), F2 scaffolding (`behavior_loss_fn` extracted verbatim into `_behavior_loss`) + 3 detach sites, F3 (`dreamer_obs_recon_loss` helper, sum-over-feature-dim), F4 (online-critic bootstrap + slow-critic regularizer + `loss_critic_slow_reg` metric), F5 (`true_cont0` threading + row-0 discount weight), F7 (`random_actions` static flag on `collect_sequence`, `static_argnums=(3, 6)`).
- `train.py` (DreamerV3 branch only) — F7: `learning_starts = config.get_mandatory('agent.learning_starts', int)`; collect call passes `bool(global_step < learning_starts)` as the static `random_actions` arg; train gate gains `global_step >= learning_starts`; ratio call site `ratio_scaled_updates(global_step // num_steps)` → `ratio_scaled_updates(global_step)` with rewritten comment.
- NEW `tests/models/`: `dreamer_nnx_fixtures.py` + `dreamer_nnx_rollout_replica.py` (shared helpers, not test files), `test_dreamer_nnx_is_first_reset.py` (T1), `test_dreamer_nnx_stop_gradients.py` (T2), `test_dreamer_nnx_obs_loss_sum.py` (T3), `test_dreamer_nnx_online_bootstrap.py` (T4), `test_dreamer_nnx_terminal_start_weights.py` (T5), `test_dreamer_nnx_replay_ratio_semantics.py` (T7).
- NEW `docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/06_nnx_recipe_deviation_register.md` (F8) + master doc §7 link + `regen_dev_index.py` run.
- `configs/` untouched by the developer — the experiment-designer's handoff had already landed in the tree (verified: all 8 dreamer_v3 YAMLs carry the rescaled `replay_ratio` + `learning_starts: 1024`; the F7 smoke used the live configs directly, no local YAML edit needed).

### Per-fix red → green evidence

| Fix | Test | Pre-fix (red) | Post-fix (green) |
|---|---|---|---|
| F1 | T1 | 3 failed: garbage-carry vs fresh-carry `next_state['deter']` differs despite `is_first=1` (unmodulated AND modulated `mod_h` variants); integration: final carry `deter` after a timeout boundary depends on the pre-done trajectory | 4 passed (incl. the is_first=0 non-vacuity control) |
| F2 | T2 | 3 failed: critic-loss→actor max |g| = 1.9e-4 (exact-zero asserted); actor-loss→WM max |g| = 4.6e-3; actor grad vs detached reference: 100 % of elements mismatched | 3 passed; critic→actor and actor→WM grads exactly 0 on every leaf; actor grad == detached replica (rtol 1e-4) |
| F3 | T3 | 2 failed: off by exactly ×40 (B,T,D fixture) and ×13 (flat fixture) | 2 passed; train_step eyeball: `loss_recon` 0.581 → 11.04 ≈ ×19 = fixture obs_dim; total WM loss finite, no NaN |
| F4 | T4 | 2 failed: `mean_return` −0.476 → −2,931,764 when target critic corrupted (+1e3/param, same rng); `loss_critic_slow_reg` missing (KeyError-style) | 2 passed: `mean_return`/`mean_value` bit-identical under corruption; regularizer ≥ 0, == hand-computed two-hot CE on replica rollout feats, and moves when the target critic moves |
| F5 | T5 | 1 failed: all-death batch (`term_reason=2`) gave `loss_actor` = −0.0515 (≠ 0) | 2 passed: all-death ⇒ both behavior losses exactly 0.0; all-alive positive control nonzero |
| F6 | — | **STOPPED** (see below) | — |
| F7 | T7 | 3 failed: source pins found `ratio_scaled_updates(global_step // num_steps)` and no `learning_starts` read/gate; `collect_sequence(..., random_actions=True)` → TypeError | 4 passed: per-env-step call site pinned; `get_mandatory('agent.learning_starts')` + gate pinned; Ratio ±1 accumulation at 0.00390625/0.5 across geometries; prefill executes uniform-random actions vs degenerate policy, carry `prev_action` == executed action |

Full T-suite + pre-existing model tests: `pytest tests/models/` → **45 passed, 0 failed**.

### F2 scaffolding — extraction bit-identity

`behavior_loss_fn` was extracted verbatim into `DreamerTrainer._behavior_loss(self, actor, critic, rng, start_state, h_mod_start, moments_low, moments_invscale, true_cont0=None)` (the plan-sanctioned testability extraction). Verified bit-identical BEFORE applying any F2 detach: 2 consecutive `train_step` calls on a fixed batch + fixed rngs, all 2×23 metrics compared `array_equal` pre- vs post-extraction (`tmp/20260708_f2_pre_extraction.npz` vs `tmp/20260708_f2_post_extraction.npz` → BIT-IDENTICAL).

**Finding worth flagging (affects how the U1 audit reads):** with the live `zero_init_reward_critic: true`, the U1 leak paths carry ~zero gradient AT INITIALIZATION — the critic/reward output heads start all-zero, so ∂critic/∂feat = 0 and advantage ≈ 0; the leaks only become material once training moves the heads (or on any non-zero-init net). The audit's probe used random weights. T2/T4/T5 therefore build their fixture trainer with `zero_init_reward_critic: false` (documented in the test files). The fix itself is unconditional.

### F6 — STOPPED (parent gate ruling)

Guard ran BEFORE any edit, as instructed:

```
$ grep -n "DreamerGroupedMLP" src/models/dreamer_v3_nnx.py
207:class DreamerGroupedMLP(nnx.Module):
259:            self.unimodal_grouped = DreamerGroupedMLP(...)          # ENCODER
434:            self.unimodal_grouped_decoder = DreamerGroupedMLP(...)  # DECODER
```

The class IS shared with the hierarchical encoder. The parent's ruling for this contingency was to **stop the fix and flag** (not take the plan's constructor-flag route). Consequences: no code change, no T6, **no param-tree change, no checkpoint break** — the plan's U6 follow-ups (bug-curator checkpoint-compat note, `dreamer_v3_implementation.md` §6 caveat) are moot. The F8 register records U6 as OPEN with the guard evidence and the future fix shape (per-instantiation `final_layer_norm` flag; encoder keeps its LN).

### F7 — findings

- **`agent.train_steps` interaction check (plan-mandated):** grep of `train.py` and `src/` finds NO reader of the `agent.train_steps` config key — it appears only in YAML files. Display-only/dead; no interaction with the ratio call site. (Candidate for the same config cleanup as C5/`agent.unimix`.)
- **Config handoff coupling:** the experiment-designer's changes were already in the working tree before F7 was exercised (their applied-record is in §Config handoff above); the smoke ran against the live `configs/models/dreamer_v3/dreamer_v3.yaml` unmodified.
- **Prefill/gate smoke** (offline-WandB, live config, 16 envs, `--episodes 0 --total-timesteps 204800` = 100 true iterations, cuda:0): `Params/effective_replay_ratio` = **0.00390625 — exactly the rescaled config value — at every logged iteration (10, 20, …, 100)**. The prefill iteration collects randomly (`global_step` 0 < 1024), then the gate opens (2048 ≥ 1024) and `Ratio` grants `int(2048 × 0.00390625)` = 8 catch-up grad steps — identical to the old code's first-iteration count, so cumulative intensity matches from iteration 1 on. `Behavior/loss_critic_slow_reg` (F4) confirmed logging live (≈1.14).
- **One extra collect compile** for the prefill variant (static flag), as planned; observed once, then never again.

### Speed check

**Gotcha discovered (cost half the speed-check time, worth recording):** `train.py`'s loop bounds on the config's `episodes: 100` (from `configs/train/default.yaml`) whenever `episodes > 0` — `--total-timesteps` alone does NOT extend the run (same class as the memory note on `dreamer_srl` single-config budgets). First-round "300-iteration" measurements actually ran ~1 iteration + compile and were discarded. Valid runs pass `--episodes 0` so the loop uses the timestep budget.

Valid A/B (same box, RTX 4090 `cuda:0`, back-to-back, same env config/seed/budget: `01_food_only`, 16 envs, seed 42, `--episodes 0 --total-timesteps 614400` = **300 iterations**, `--no-wandb`, JIT compile included in both):

| Leg | Code | Config | Wall | s/it | env SPS |
|---|---|---|---|---|---|
| BEFORE | pristine `HEAD` (`git show HEAD:` swap-in of `train.py` + `dreamer_v3_trainer.py`) | `replay_ratio: 0.5` (old per-sequence semantics) | 535 s | 1.78 | 1148 |
| AFTER | full WP-NNX stack | live `dreamer_v3.yaml` (`0.00390625` + `learning_starts: 1024`) | 448 s | 1.49 | 1371 |

**Δ = −16 % wall time (post-fix FASTER); no regression.** Gradient-step counts are identical by construction (both legs: 8 grad steps/iteration, 2400 total — BEFORE `Ratio(0.5)(global/128)`, AFTER `Ratio(0.00390625)(global)`), so the speedup is code-side — consistent with F2's stop-gradients pruning the backprop graph through the 15-step imagination scan, partially offset by F4's extra target-critic forward. Cost-neutrality of the U4 rescale confirmed: `Params/effective_replay_ratio` reads ≈ the rescaled config value on the live run.

### Full-suite gate

`pytest tests/ -q --continue-on-collection-errors` → **435 passed, 16 failed, 494 skipped** (19 m 53 s). Accounting:

- **8 = the known-red baseline, exactly as listed in the plan**: 4× `test_unified_parity[observability_gates_S1–S4]`, 3× stale-config (`test_inactive_animal_offgrid` ×1 + `test_truncation_not_death` ×2), 1× `test_dreamer_srl_offline_wm_test::test_offline_wm_smoke`. Not chased.
- **8 transient, self-inflicted and re-verified green**: the suite ran concurrently with the speed-check A/B, which (i) temporarily swapped `train.py` to HEAD (→ my 2 T7 source-pin tests read the pre-fix text mid-window) and (ii) saturated the GPU (→ 3× `test_evaluation_model_rebuild` + 3× `test_checkpoint_restore_roundtrip` failed with XLA `INTERNAL: Failed to materialize symbols`). Isolated re-run after the A/B finished: **all 14 tests in those 3 files pass**.
- **No new red attributable to WP-NNX. No WP-SRL collection errors observed** (no `--ignore` needed).

### Deviations from plan (all recorded, none silent)

1. **F6 stopped** per parent gate ruling (plan recommended the constructor-flag route when shared; parent overrode). Register row U6 = OPEN instead of FIXED.
2. **No per-fix commits** (parent override of the plan's one-commit-per-fix; parent commits the package post-verification).
3. **`_behavior_loss` extraction** passes `start_state/h_mod_start/moments_*/true_cont0` as parameters rather than closures (the plan's F5 said "reaches by closure"; the F2 extraction it sanctioned makes them parameters). `true_cont0=None` default reproduces pre-F5 row0=1 for direct callers; `train_step` always passes the real value.
4. **T1(b)** implemented as bit-equality of ALL post-boundary transition rows + final carry (stronger than the plan's "action distribution invariant"), with the non-vacuity guard on carry divergence at the boundary — the naive "pre-done actions differ" guard proved vacuous (tiny action space; same-key sampling coincided).
5. **T2/T4/T5 fixtures** set `zero_init_reward_critic: false` (see §F2 finding) — without it the pre-fix tests are NOT red at initialization.
6. **T2(b) reference** (`dreamer_nnx_rollout_replica.py`) tracks the production bootstrap source; it was flipped target→online in lockstep with F4 (in-file MAINTENANCE note added).
7. **T7(a)** pins the call-site arithmetic via source-text assertions on `train.py` (the plan's sanctioned fallback; `train.py` is not importable) plus a Ratio-accumulation simulation.
8. **T4(b)** hand-computes the regularizer on the replica rollout and additionally asserts it MOVES under target-critic corruption while `mean_return` does not.

### Follow-ups / blockers for the parent

- **K1 registry row**: F1 closes the long-open is_first collection reset — ask `bug-curator` to flip K1 to fixed with the package commit ref (also fill the commit refs in the F8 register's FIXED rows).
- **U6**: remains open (register row carries the future fix shape). Decide whether to schedule a follow-up WP.
- **Stale config comment** (experiment-designer scope, not mine): `dreamer_v3_sheeprl_matched.yaml:71` still says "Effective per-iter grad updates = Ratio(replay_ratio)(global_step / collect_interval)" — the formula is now `Ratio(replay_ratio)(global_step)`.
- `scripts/dreamer/dreamer_offline_wm_test.py` calls `get_action` without staging `is_first`; the `prev_state.get('is_first', zeros)` default keeps it at old behavior (no breakage), but it could now stage the flag for correct boundary resets — outside my scope fence.
- Comparability epoch: per plan §Comparability, post-fix NNX runs open the "NNX parity epoch, 2026-07 / WP-NNX" (no F6 param-tree break though — old checkpoints still restore structurally).

Scratch artifacts (not for commit): `tmp/20260708_*` (extraction smokes, U1 probes, speed logs `20260708_wpnnx_speedB2.log`/`speedA2.log`, ratio-smoke offline WandB dir).

> Implemented by: developer

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-08
> **Verdict**: **PASS** — all six landed fixes (F1–F5, F7) match the plan's mechanism sections; F6 correctly stopped at the guard with zero code diff; F8 register consistent with the code's actual state; tests, configs, suite accounting, and speed all check out. Ready for the parent to commit.

### Scope note

The working tree also carries the parallel WP-SRL package (`src/algorithms/dreamer_srl/*`, `tests/algorithms/*`, `DEVIATION_LOG.md`, `fix_plan_srl_parity.md`), the held WP-GAMMA diffs (`configs/models/dreamer_srl/*.yaml`), and a pre-existing `train_command-agent.sh` edit (present at session start). None of these are WP-NNX scope and none were reviewed here. Within the WP-NNX file fence, **no out-of-scope changes were found**: the diff touches exactly `src/models/dreamer_v3_trainer.py`, the DreamerV3 branch of `train.py`, 8 new files under `tests/models/`, the 8 `configs/models/dreamer_v3/` YAMLs (experiment-designer's parallel handoff, verified below), the F8 register, the two parity-doc cross-link updates, `docs/develop/INDEX.md` (regen), and the diary.

### Per-file verification

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/models/dreamer_v3_trainer.py` | F1, F2, F3, F4, F5, F7 (`collect_sequence` flag) + `_behavior_loss` extraction | ✅ | Diff matches every plan mechanism: F1 consumes staged `is_first` via `prev_state.get`, zeroes `prev_action`, `jnp.where`-resets `mod_h` *before* `forward_obs`, and `is_first` flows into `rssm.step` (which masks deter/stoch at `dreamer_v3_nnx.py:117-119`) — the sheeprl `player.init_states` analog; stale `collect_sequence` comment updated. F2's three detach sites present in both scan branches + explicit critic-input detach + detached action in the log-prob. F3 sum-over-feature-dim via the new `dreamer_obs_recon_loss` helper. F4 online-critic bootstrap at all three sites + slow-critic two-hot-CE regularizer + `loss_critic_slow_reg` metric. F5 `true_cont0` stop-gradiented, flattened like `start_state`, row-0 weight. F7 `random_actions` static flag (`static_argnums=(3, 6)`), carry `prev_action` kept consistent with the executed random action. |
| `src/models/dreamer_v3_nnx.py` | — (F6 STOPPED) | ✅ | **Zero diff confirmed** (`git diff --stat` empty); decoder trailing `nnx.LayerNorm` intact at the `DreamerGroupedMLP` head (`:225`). Guard result (class shared by encoder `:259` and decoder `:434`) matches my own grep. No param-tree change, no checkpoint break. |
| `train.py` | F7 only | ✅ | Exactly three touch points, all in the DreamerV3 branch: `get_mandatory('agent.learning_starts', int)` near :847; static `bool(global_step < learning_starts)` passed to `collect_sequence`; train gate gains `global_step >= learning_starts` and the ratio call becomes `ratio_scaled_updates(global_step)` with the rewritten comment. No other `train.py` changes. |
| `tests/models/` (6 test files + 2 helpers) | T1–T5, T7 + fixtures/replica | ✅ | Read in full. T1 has non-vacuity controls in both directions (is_first=0 keeps history; garbage carry propagates to the boundary). T2 asserts exact-zero critic→actor and actor→WM grads plus equality with a detached-reference replica (forward-value agreement guards PRNG drift). T4 corruption-invariance is bit-exact + hand-computed regularizer + moves-with-target assertion. T5 all-death ⇒ exactly-zero losses + all-alive positive control. T7 = plan-sanctioned source pins + Ratio accumulation simulation + prefill behavior test. Replica helper carries an explicit MAINTENANCE lockstep note. |
| `configs/models/dreamer_v3/*.yaml` (8 files) | U4 rescale + `learning_starts` (experiment-designer) | ✅ | Diff matches the handoff table value-for-value (0.5→0.00390625 ×5, 0.0625→0.00048828125, 1→0.0078125, `sheeprl_matched` kept 1.0 with the ⚠ ~128× cost warning). **Independently re-resolved all 8 through `load_env_config` + `get_mandatory`: every file yields `learning_starts=1024` and the expected `replay_ratio`** — nothing crashes at launch. Misleading K5 `collect_interval` comment replaced. Residual: `dreamer_v3_sheeprl_matched.yaml` `train_steps: 64` line still carries the stale formula `Ratio(replay_ratio)(global_step / collect_interval)` — comment-only, already flagged to experiment-designer in the Implementation Report follow-ups. |
| `06_nnx_recipe_deviation_register.md` (new) | F8 | ✅ | Frontmatter + plain-language entry point present. Register ↔ code consistent: U1/U3/K1 and U2/U4/U5 FIXED (all verified in the diff), **U6 OPEN with the guard evidence and the future fix shape** (per-instantiation flag, encoder keeps LN), C1–C8 KEPT, K2–K7 pointed at KNOWN_BUGS without duplication. No fixed item listed open; no open item silently fixed. |
| `00_master_comparison.md` / `05_dreamer_v3_nnx_conventions.md` | §7 link + routing notes | ✅ | Cross-links both directions; `docs/develop/INDEX.md` regenerated (in diff). |

### Judgment-call rulings (parent-requested)

1. **T2/T4/T5 `zero_init_reward_critic: false` fixtures — CORRECT, and the right way to make the tests discriminative.** Reasoning: under the live zero-init, the critic/reward heads' final-layer kernels are zero, so ∂(head output)/∂feat = 0 and advantage ≈ 0 — the U1/U2 leak paths exist in the graph but carry numerically zero gradient *at initialization only*. After the first optimizer step the heads move off zero and the leaks become live on every live run. The tests assert a **structural graph property** (a path is severed by `stop_gradient`), and asserting exact-zero gradients at a *random* point in parameter space proves the path is severed for **all** parameter values — including trained live-config ones. Risk that the fix is a no-op under live configs: none — every `stop_gradient` added by F2/F4/F5 is unconditional in the code (no config dependence). The finding is properly documented in the test files and the report.
2. **`_behavior_loss` extraction bit-identity — independently re-verified.** I reloaded `tmp/20260708_f2_pre_extraction.npz` vs `_post_extraction.npz` myself: 44 arrays (2 train_steps × 22 metrics), key sets equal, `np.array_equal` on every key — bit-identical. Methodology sound: the extraction was validated in isolation *before* any semantic (F2) change was applied.
3. **T7 source-text pins — ACCEPTABLE.** The plan explicitly sanctioned this fallback (`train.py` is not importable), and the pins are paired with a behavioral `Ratio` accumulation simulation covering three geometries, so the semantics are tested, not just the text. Standard caveat: text pins break on innocent refactors of the pinned lines — acceptable for a regression pin whose job is exactly to make silent reversions loud.

### Test & suite verification (re-run by verifier)

- `pytest tests/models/` → **45 passed, 0 failed** (2 m 18 s) — matches the expected count.
- The 2 T7 source-pin tests re-run by name in isolation → **green** (their mid-window failure was the A/B's temporary `train.py` HEAD swap, as claimed).
- The formerly XLA-OOM files `tests/scripts/test_evaluation_model_rebuild.py` + `tests/training/test_checkpoint_restore_roundtrip.py` re-run on an idle GPU → **10 passed** (+ 4 T7 = the 14 the report claims). GPU-contention attribution confirmed; not regressions.
- Known-red baseline spot-check: `tests/env/test_inactive_animal_offgrid.py` + `tests/env/test_truncation_not_death.py` → exactly the 3 pre-existing stale-config `FileNotFoundError` failures (config retirement `b093023`), unrelated to WP-NNX. The remaining baseline rows (4× unified-parity S1–S4, 1× dreamer_srl smoke) are long-documented and the srl row belongs to the parallel WP-SRL tree. Accounting: 16 = 8 baseline + 8 re-verified transients, **no new red attributable to WP-NNX** — implementer's accounting confirmed.

### Speed verdict

**✅ no regression — post-fix is ~16% FASTER** (535 s → 448 s wall, 300 iterations, same box/GPU/config/seed, back-to-back, JIT compile included in both legs). Corroborated against the raw log artifacts (`tmp/20260708_wpnnx_speedB2.log` `WALL_SECONDS=535`, `speedA2.log` `WALL_SECONDS=448`, both reaching ~291–294 iterations). Cost-neutrality of the U4 rescale holds by construction (both legs 8 grad steps/iter, 2400 total) and was confirmed on the live smoke (`Params/effective_replay_ratio` = 0.00390625 at every logged iteration). The `--episodes 0` budget gotcha was caught and documented. One doc inconsistency found and annotated in place: Checkpoint 10 cited discarded first-round numbers (199/188 s) — the Implementation Report table is authoritative.

### Deviations review

All 8 recorded deviations are either parent-ruled (F6 stop, no per-fix commits), plan-sanctioned (extraction parameters, T7 pins), or strengthenings (T1(b) bit-equality form, T4(b) extra assertion, replica lockstep note, zero-init fixtures). None silent, none objectionable.

### Follow-ups for the parent (unchanged from the Implementation Report, endorsed)

1. Commit the package, then ask `bug-curator` to flip **K1 → fixed** with the commit ref and fill the F8 register's commit refs.
2. **U6 stays open** — decide whether to schedule a follow-up WP (register row carries the fix shape).
3. Hand the stale `sheeprl_matched.yaml` `train_steps` formula comment to `experiment-designer` (comment-only).
4. Optional, out of WP scope: `scripts/dreamer/dreamer_offline_wm_test.py` could stage `is_first` for correct boundary resets (currently safe via the `.get` default).
5. New NNX comparability epoch ("NNX parity epoch, 2026-07 / WP-NNX") applies to all post-fix runs.

**Conclusion**: PASS. Implementation is faithful to the plan, the F6 stop is clean and correctly registered, the parallel config handoff is complete and launch-safe, the test suite is green with the known-red baseline exactly accounted for, and the change is a speed improvement. Approved for commit by the parent.

> Verified by: senior-developer
