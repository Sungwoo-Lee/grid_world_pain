---
title: "Live-path audit step 4/4 — dreamer_srl remaining surface (eval/checkpoint) + NMN-port readiness"
topic: diagnosis
status: active
created: 2026-07-10
last_updated: 2026-07-10
---

# dreamer_srl eval/checkpoint gaps + neuromodulation-port readiness

## Purpose (plain-language entry point)

This is the last step of a four-step audit of the code paths the project actually trains
and evaluates with. The live Dreamer implementation (`dreamer_srl`) had its training loop,
replay buffer, losses, world-model core, and actor-critic deeply audited and fixed earlier
this week — those areas are **not** re-audited here. What remained un-audited was the
**evaluation and checkpointing side**: the in-training evaluation that runs whenever a
checkpoint is saved, the checkpoint save/load helpers themselves, and whether a finished
Dreamer run can be evaluated offline at all. Part A walks that surface.

Part B is a **readiness assessment** (not a plan) for the neuromodulation port that was
promised when the project pivoted to this Dreamer implementation in May 2026 and never
happened: the live Dreamer has zero modulation hooks today. The assessment maps where the
hooks would go, what the two existing neuromodulated implementations (the live
recurrent-PPO variant and the archived old Dreamer) already provide for reuse, the design
decisions a plan would have to settle, and an honest size estimate.

**Headline, part A:** the in-driver eval path is correct and faithful to training — same
environment object, same observation function (noise included), deterministic action
choice, sound random-number handling, honest survival-step counting. The gaps are at the
edges: evaluation metrics get **double-logged** under one name when both eval passes run
(unlike the rPPO trainer it claims to mirror), a config knob for parallel eval
environments is silently ignored, and — the big picture item — **there is no offline
evaluation path for the live Dreamer at all**: the standard offline eval script refuses
Dreamer checkpoints, so the only behavioral evaluation that exists is the one inside the
training process. 6 new findings (0 blocker, 2 concern, 4 nit), 2 known items
re-confirmed, plus an explicit sound-list.

**Headline, part B:** the port is tractable and well-precedented — roughly 600–900 lines
across ~6 files, with the Dreamer-specific modulator network already living in the live
codebase (it was written for the old Dreamer and survives in `src/models/neuromodulator.py`)
and a drop-in modulated GRU cell in the archive. The three decisions that must be made
before any plan is written: **which loss trains the modulator** (dreamer_srl's three
separate optimizers make this structurally harder than rPPO's single shared loss),
**which components get modulated** (encoder only, vs encoder + RSSM memory gate, vs also
imagination), and **how the modulator's hidden state is reset** at episode boundaries in
all three places state lives (collection, training sequences, eval). Every hook is a
deviation from the sheeprl reference and needs a deviation-log row.

Scope fence honored: training loop / replay / losses / RSSM / actor-critic
([[00_master_comparison]] areas 01–04, fixed in `8c0fcf9`, `867ec51`, `b1dd90a`) are cited,
not re-derived.

---

## Part A — the un-audited dreamer_srl surface

### A.1 What the eval path actually is

`src/algorithms/dreamer_srl/eval.py` (`dreamer_srl_eval_rollout`) is an **in-process**
eval: it receives the **live** `world_model` / `actor` module objects from the training
driver — it never loads a checkpoint. It is called from the driver's checkpoint-save block
(`dreamer_srl_main.py:1538-1605`) in up to two passes (video pass with recordings, stats
pass without), mirroring the rPPO trainer's pattern (`train.py:1995-2040`).

**Env rebuild fidelity: sound (better than a rebuild).** Eval reuses the driver's live
`env_params` object (`dreamer_srl_main.py:1549,1589`) — by construction identical to the
training environment, including the current curriculum stage (the stage-swap block
reassigns `env_params` at `dreamer_srl_main.py:1428` *before* the checkpoint/eval block at
`:1512` runs, and both live in the same `if dones_idxes:` body in the correct order). The
old "eval uses wrong stage's environment" bug class (registry Finding E) does not recur
here. Observations go through the same `get_observation(state, env_params)`
(`eval.py:104,171`) the training wrapper uses (`src/environment/wrapper.py:15,20,31`),
with `apply_noise=True` by default (`src/environment/sensor.py:291,346-347`) — eval sees
the same noise regime as training.

**Checkpoint restore usage: none.** Eval-during-training uses in-memory modules. The
restore story is Part A.3.

**Deterministic-vs-stochastic policy: correct.** Action = `argmax` over post-unimix
logits via `actor.forward_logits` (`eval.py:156-157`, head at `agent.py:1487`), no Gumbel
sampling — while training collection samples (`Player.get_actions`,
`dreamer_srl_main.py:340`). The RSSM posterior is still *sampled* per step with a fresh
sub-key (`eval.py:139-147`), which matches sheeprl's test-time behavior (greedy actor,
stochastic latent) and is fully seeded — same `seed` → same episode.

**PRNG: sound.** Eval builds its own `jax.random.PRNGKey(seed)` (`eval.py:97`) and never
consumes the training key stream, so running eval does not perturb training
reproducibility. The key is split before every reset (`:102`) and every RSSM step
(`:139`); no sub-key reuse. Using the *same* fixed seed at every checkpoint means every
checkpoint is evaluated on the same episode set — good for cross-checkpoint
comparability (deliberate; same as rPPO).

**Survival-step counting: honest.** `step_count` increments once per `jax_step`,
including the terminal transition (`eval.py:167-169`); `done` comes from the env
(truncation at `max_steps` is `termination_reason==1` → `done`), and the redundant
`step_count < max_steps` loop guard (`:131`) can never bind first. `mean_length` is
therefore the survival-steps metric with the same +0/+1 convention as the driver's
post-P5-fix episode logging.

**Recording outputs and consumers.** The video pass writes standard
`episode_NNNNNN.rec.gz` files under `<results_dir>/recordings/<episode>/` plus
`run_meta.pkl` (`eval.py:74-90,182-185`) — the same format `render_recordings.py`,
`trajectory_story.py`, and `motif_cluster.py` consume, so qualitative/trajectory analysis
of dreamer_srl runs works today. Rendering runs in a CPU-pinned subprocess with WandB
video upload (`eval.py:246,261-270`), mirroring `evaluation_core.py`.

### A.2 Findings — driver eval/checkpoint cadence + eval.py

| # | Severity | Class | Where | Issue |
|---|---|---|---|---|
| D1 | 🟡 concern | NEW | `dreamer_srl_main.py:1573-1580` + `:1598-1605` | **`Eval/MeanReward` / `Eval/MeanLength` double-logged per checkpoint.** When both `training.video_during_training` and `training.stats_during_training` are on, the video pass (small sample, `eval_video_episodes`) and the stats pass (`eval_stats_episodes`) each log the *same* WandB keys with the same `timesteps` value. The rPPO trainer this block claims to mirror logs Eval/* **only from the stats pass** (`train.py:2038`). Effect: the Eval curves interleave two estimates of different sample size at identical x — noisy small-N points pollute the headline eval survival-steps curve. Fix shape: drop the Eval/* log from the video pass (or namespace it `Eval/Video*`). |
| D2 | 🟡 concern | NEW (documented in code comments, no registry row) | `scripts/eval/eval_rollout.py:1010-1011` (`NotImplementedError`), detection `:857-862` | **No offline behavioral evaluation exists for the live Dreamer.** `eval_rollout.py` — the standard offline eval/behavior-measures/recording pipeline — hard-raises `NotImplementedError` for `agent_type == "dreamer"`. The only checkpoint consumers are (a) the in-driver eval, which doesn't restore at all, and (b) `scripts/dreamer/visualize_dream.py:806-841`, which restores world-model + actor (its own `PyTreeRestore` with CPU sharding, `partial_restore=True`) for *dream visualization*, not behavioral eval. Plainly: **a trained dreamer_srl run can be evaluated today only by the eval that ran inside its own training process.** Post-hoc re-evaluation with new seeds, more episodes, behavior measures, or threat-onset analysis is impossible without new code. (The `_resolve_continual_stage_config` machinery at `:633-732` even reads the Dreamer-style `stage` checkpoint field — the plumbing is half-ready; the agent-build branch is the missing piece.) |
| D3 | 🟢 nit | NEW (dead-key class) | `dreamer_srl_main.py:593`, `eval.py:101-192` | `training.eval_stats_num_envs` is read via `get_mandatory` but **never used** — the stats pass runs single-env, sequentially, in eager (unjitted) Python, unlike rPPO's batched stats pass (`train.py:2030-2036`, `num_envs=stats_envs`). Config silently no-ops, and stats eval is ~N× slower than the config advertises (relevant when `eval_stats_episodes` is large — training stalls for the duration). |
| D4 | 🟢 nit | KNOWN-class, NEW instance | `eval.py:122,177` (`true_obs=None`) | Eval recordings never carry the clean (pre-noise) observation, so the noise-contrast side panel that was restored for rPPO eval videos (registry "Noise invisible in eval video", fixed `80d3b70`) is structurally absent for dreamer_srl. Only matters for noise-enabled envs; recordings remain valid otherwise. |
| D5 | 🟢 nit | NEW | `dreamer_srl_main.py:1550,1592` vs `:1428` | In curriculum mode, eval passes `config=env_cfg`, which stays bound to **stage 0's** config after stage swaps (only `env_params` is rebound). Consumers are `visualization.icons` and the `source_path` provenance string in `run_meta.pkl` (`eval.py:82-90`) — so later-stage recordings carry stage-0 icon config/provenance. Cosmetic unless stages change icon sets; env physics/obs are correct (live `env_params`). |
| D6 | 🟢 nit | NEW | `dreamer_srl_main.py:1189-1935` (loop end), compare cadence gate `:1512-1514` | **No end-of-training checkpoint save.** If the episode budget is not a multiple of the active checkpoint frequency, the final weights are silently discarded — the last save can be up to `checkpoint_frequency − 1` episodes stale. Same gap exists in rPPO `train.py` (only the cadence save at `:1996`; nothing after the loop), so this is a platform-wide pattern, not an srl regression — but it becomes load-bearing the day a resume path lands. |
| D7 | 🟢 nit | NEW (dead code) | `dreamer_srl_main.py:1511,1532,1608`; `:854`; `checkpoint.py:113-123` | Three dead ends: (a) `_just_saved_ckpt` is assigned and never read; (b) `wandb.define_metric("eval/checkpoint_episode")` defines a key that is never logged (the mirrored train.py key); (c) `load_checkpoint()` in `checkpoint.py` has **zero live callers** — `visualize_dream.py:770` imports it but restores through its own manager — and its `manager.restore(episode)` with no restore-args against a `StandardCheckpointer` item is untested and would likely fail without an abstract target. Anyone "resuming" via this helper would be debugging dead code. |

### A.3 Known items re-confirmed

| Known item | Where confirmed | Note |
|---|---|---|
| **Checkpoints save weights only — no optimizer state, no restore path** (registry: "Dreamer-srl checkpoints drop the optimizer's momentum on save", OPEN) | `checkpoint.py:84-96` — payload is `nnx.Param` states of the four modules + moments + PRNG key + five bookkeeping scalars (incl. `stage`, explicitly documented write-only at `:80-82`). No `wm_opt`/`actor_opt`/`critic_opt` state, no `Ratio` scheduler state, no replay buffer. The driver has **no resume/`--load-checkpoint` flag at all** (`dreamer_srl_main.py` argparse `:410-464`). | Unchanged; D7(c) adds that the nominal load helper is dead code. Any future resume plan must add optimizer + Ratio state to the payload or accept a momentum restart. |
| **Dreamer neuromodulation does not exist in the live stack** (registry OPEN — informational) | Re-grepped: zero hits for `modulat` in `src/algorithms/dreamer_srl/` and `configs/models/dreamer_srl/` (19 config files, none with a `modulation` block). | Ground truth for Part B. |

### A.4 Explicitly checked and found sound (non-findings)

- **Checkpoint cadence math** — the `//`-crossing trigger (`dreamer_srl_main.py:1512-1514`)
  fires exactly once per crossed multiple, handles multi-episode iterations (several envs
  finishing at once) and per-stage frequencies (`checkpoint_frequency_active`,
  `:1004-1007,1494`); saving inside the `dones` block is complete because
  `total_episodes_completed` only advances there.
- **Eval does not corrupt training state** — eval touches neither the training `key`, the
  env `states`, `step_data`, nor the Player's recurrent state; it builds its own key,
  resets its own env states, and initializes fresh RSSM state per episode
  (`eval.py:97-112`).
- **Eval RSSM stepping parity with collection** — `eval.py:136-161` is a faithful
  single-env transcription of `Player.get_actions` (`dreamer_srl_main.py:299-347`):
  same `vmap(encoder)`, same `rssm.dynamic` argument order, same
  `cat(posterior_flat, recurrent)` latent, `is_first=1` on the first step then 0.
- **WandB step-collision handling** — the eval block deliberately logs without an explicit
  `step=` (comment `:1568-1572`) because the render subprocess advances WandB's internal
  counter; subsequent `wandb.log(..., step=policy_step)` calls stay monotonic because
  `policy_step` outruns the +1 internal bumps.
- **Checkpoint payload round-trips through the scan path** — Orbax `StandardSave` of
  `nnx.state(module, nnx.Param)` pytrees plus 0-d arrays; `moments` NamedTuple stored
  field-by-field (`checkpoint.py:97-106`); `manager.wait_until_finished()` prevents the
  async-save/next-iteration race.
- **`terminated`/`truncated` split at eval parity** — eval doesn't need the split (no
  bootstrapping), and the driver's split (`termination_reason>=2` vs `==1`,
  `dreamer_srl_main.py:1244-1246`) was area-04 audited; nothing on the eval side
  re-derives it inconsistently.

---

## Part B — NMN-port readiness assessment (memo, not a plan)

### B.1 What exists today — the two reference implementations

**Live rPPO-NMN** (audited yesterday, [[02_rppo_nmn_wiring]] — verdict: forward path
correct and train/eval-consistent; debt at the config boundary):

- `src/models/neuromodulator.py:41-186` `NeuromodulatorRNN` — GRU core over raw obs;
  branched heads → per-neuron γ (gain), β (bias), memory gate-bias, action temperature;
  grouping via `repeat/slice`; FiLM forces γ-bias 1.0 (pass-through init).
- `src/models/recurrent_ppo_network.py:140-190` — injection sites: FiLM/PreActivation/
  Multiplicative applied between encoder pre-activation and activation, in both flat and
  hierarchical (Phase-1 unimodal / Phase-2 hub) encoders; `:326-340` gate-bias into the
  GRU update gate (`src/models/modulated_gru_cell.py`, 78 lines) and temperature on the
  logits.
- **Loss signal: none of its own.** The modulator trains purely through the shared PPO
  loss under **one Adam over all `nnx.Param`** — the 02 report's headline correction (no
  heteroscedastic aux loss exists). Modulator state `mod_h` is everywhere the task-RNN
  state's pytree sibling, so every reset/checkpoint/eval mechanism covers it for free.

**Archived NNX-Dreamer NMN** (`src/models/archive/dreamer_v3_nnx/`, archived `fd7af84`):

- `src/models/neuromodulator.py:203-379` `DreamerNeuromodulatorRNN` — **lives in the live
  tree, not the archive** (shared file). Dual input modes: `forward_obs` (raw obs, during
  collection/WM-observe) and `forward_imagine` (`cat(feat, action)`, during imagination —
  perceptual heads return zeros there since the encoder doesn't run). Heads target
  `embed_dim` (encoder output), `deter_dim` (RSSM recurrent state), plus a sigmoid
  `z_reward` scale.
- Encoder hook: `dreamer_v3_nnx.py:301-410` `forward_with_modulation` — γ/β between
  pre-activation and activation, flat + hierarchical variants.
- RSSM hook: `modulated_layer_norm_gru_cell.py` (90 lines) — the archive's LayerNorm GRU
  cell with an optional `gate_bias` added to the **update gate** pre-activation
  (`update = sigmoid(update + gate_bias)`, `:79-82`); `gate_bias=None` is bit-identical
  to the unmodulated cell. RSSM `step`/`img_step` accept and thread it
  (`dreamer_v3_nnx.py:106-165`).
- Reward hook: imagination rewards scaled `rew * z_reward`
  (`dreamer_v3_trainer.py:461-472`).
- State threading: `mod_h` joins the agent state — eval tuple `(rssm_h, mod_h)`
  (`dreamer_v3_nnx.py:668-676,689-691,731-733`); collection dict key `'mod_h'` with
  **is-first-gated reset** `jnp.where(is_first > 0.5, initial, prev)`
  (`dreamer_v3_trainer.py:653-673,711,746`); WM-training loop calls `forward_obs` per
  timestep inside the observe scan (`:178`).
- Config shape: `configs/models/archive/dreamer_v3_nnx/neuromodulated_dreamer_v3.yaml`
  `agent.modulation` block — `type, mod_hidden_size, grouping_size, percept_bias_init,
  percept_add_bias_init, memory_bias_init, reward_bias_init` (7 keys).

**Measured NNX-NMN footprint** (what "the port" cost last time): ~550 modulation-specific
lines across 4 code files + 1 config — `DreamerNeuromodulatorRNN` 187 lines (reusable
as-is), `ModulatedLayerNormGRUCell` 90 lines (drop-in-shaped), encoder/WM/eval wiring in
`dreamer_v3_nnx.py` ~150 lines, trainer threading (collection reset, observe loop,
imagination + z_reward, metrics) ~120 lines.

### B.2 Hook map — where equivalents go in dreamer_srl

| NNX-Dreamer hook | dreamer_srl equivalent location | Notes |
|---|---|---|
| Modulator module | reuse `src/models/neuromodulator.py:203` `DreamerNeuromodulatorRNN` | Already live code; dims fit (`embed_dim` = encoder `dense_units`, `deter_dim` = `recurrent_state_size`). Its `obs_breakdown` arg is currently unused — fine for the flat encoder. |
| Encoder `forward_with_modulation` | `agent.py:1115-1197` `MLPEncoder` | **dreamer_srl's encoder is a flat sheeprl MLP** (symlog → N×(Linear+LN+SiLU)); there is no hierarchical unimodal/multimodal split like rPPO/NNX. A γ/β hook goes between the last Linear+LN and its SiLU (or on every layer) — see decision 2. |
| Modulated GRU cell | `agent.py:~60-135` `LayerNormGRUCell` | Structurally the same LN-GRU as the archive's cell (combined 3-way gates); the archive's 90-line `gate_bias` variant ports nearly verbatim. `gate_bias=None` must keep `test_agent.py::test_layernorm_gru_cell_matches_sheeprl` green. |
| RSSM threading | `agent.py:371-1090` — `dynamic` (`:975`), `_transition` (`:720`), `WorldModel.observe` (WM-loss forward, called at `train.py:706`), `imagine` (`:1732`) | `dynamic` resets h/z internally on `is_first` (Lever-A parity); the mod path must add the **same is-first gating for `mod_h` inside the observe scan**, or mod state leaks across episode boundaries within replayed training sequences (the NNX trainer did exactly this at `dreamer_v3_trainer.py:668`). |
| Collection state | `dreamer_srl_main.py:207-347` `Player` | `_mod_h` joins `_recurrent_state/_posterior_state/_prev_action`; the **fixed-width `done_mask` masked-reset path** (`init_states`, `:246-266` — the recompile-storm Fix 1 idiom) must cover it; `get_actions` calls `modulator.forward_obs` before the encoder. The H5 `step_data`/`is_first` machinery needs no change (mod_h is inference state, not buffer data). |
| Config plumbing | `configs/models/dreamer_srl/*.yaml` + `dreamer_srl_main.py:522-582` mandatory reads + `build_agent(cfg=agent_cfg.to_dict())` (`:694-699`) | New `algo.modulation` (or `agent.modulation`) block; driver reads with `get_mandatory` per protocol. WandB `modulator/*` metrics need adding (rPPO logs collection-time mod signals). |
| Checkpoint payload | `checkpoint.py:84-96` — automatic **if** the modulator is a `WorldModel` submodule (`nnx.state(world_model, nnx.Param)` picks it up); `_graphdef_wm` (`dreamer_srl_main.py:1032`) and the jitted scan carry it transparently | Old (unmodulated) checkpoints become structurally incompatible with a modulated model — fine, since no restore path exists anyway (A.3); `visualize_dream.py`'s `partial_restore=True` keeps working. |
| Eval path | `eval.py:106-161` | Same threading as Player: init `mod_h` per episode, `forward_obs` + modulated encoder per step. Small (~20 lines) once Player is done. |

### B.3 Design decisions a plan must settle (ranked)

1. **Which loss trains the modulator.** This is the structural fork. rPPO's answer —
   "the shared loss, one optimizer" — has no direct analog: dreamer_srl runs **three
   disjoint optimizers** over disjoint modules (`dreamer_srl_main.py:710-712`,
   `wrt=nnx.Param` per module). If the modulator is a `WorldModel` submodule, it trains
   through the **WM loss only** (reconstruction/KL/reward/continue via the modulated
   encoder + gate-bias path). Consequences: (a) the actor/critic losses cannot shape it —
   imagination-side hooks like the NNX `z_reward` reward scale would be **gradient-dead**
   under dreamer_srl's optimizer split (sheeprl's actor grads flow only into actor
   params); (b) an action-temperature head (rPPO-style) would likewise be untrainable
   unless the actor optimizer is widened to include modulator params — a bigger deviation.
   The honest options: (i) WM-loss-only modulator (encoder + RSSM gate hooks, drop
   z_reward/temperature) — smallest, cleanest; (ii) widen `actor_opt` to a
   `(actor, modulator)` pair — restructures the optimizer contract the parity audit just
   verified; (iii) a fourth optimizer with an explicit modulator objective — new research
   surface, out of port scope. The pivot memo and the rPPO finding both point to (i) as
   the defensible default; a plan must state the choice and its DEVIATION_LOG rows.
2. **What gets modulated.** Encoder-only (one γ/β pair on the flat `dense_units` embed) is
   the minimum viable hook; encoder + RSSM update-gate bias matches the NNX precedent at
   modest extra cost (the 90-line cell + `dynamic`/`observe`/`imagine` threading);
   imagination-mode modulation (`forward_imagine`) is only meaningful if decision 1
   gives it a gradient. Related sub-decision: dreamer_srl's flat encoder has **no
   per-sense structure** — accept sense-agnostic modulation (loses the Phase-1/Phase-2
   story the rPPO experiments are built on), or port the hierarchical encoder into
   dreamer_srl first (a large, sheeprl-visible deviation that would itself need parity
   review). This choice decides whether Dreamer-NMN results are comparable to the
   rPPO-NMN sweep.
3. **`mod_h` reset semantics in all three habitats.** (a) Collection: fixed-shape
   `done_mask` reset in `Player.init_states` (recompile-safe idiom already in place);
   (b) training sequences: is-first-gated reset **inside** the observe scan, mirroring
   `rssm.dynamic`'s internal h/z reset — forgetting this is the port's most likely silent
   bug (cross-episode mod-state leakage in every training batch); (c) imagination: seed
   `mod_h` from the observe-scan's per-timestep values (NNX precedent) or zeros — must be
   stated. Also curriculum: stage-swap `player.init_states()` full reset covers `mod_h`
   for free once it's a Player field.
4. **Reuse vs re-port the modulator class.** `DreamerNeuromodulatorRNN` is live, tested
   shape-wise, and dimensionally right — reuse as-is and keep `neuromodulator.py` shared.
   The cost of reuse: it drags the archive's design (z_reward head, dual projections)
   into a stack that may not train those heads (decision 1); dead heads still consume
   params and would sit in checkpoints. Trimming = diverging from the rPPO-shared file.
5. **Parity bookkeeping.** Every hook is a deviation from sheeprl@33b6366: the plan must
   pre-declare D-rows (encoder hook, cell hook, threading, config) in DEVIATION_LOG, keep
   the `type: null` / absent-modulation path **bit-identical** (regression: existing
   grad-parity + GRU-cell tests), and note that modulated runs are a new comparability
   class vs the just-fixed baseline runs.

### B.4 Port preconditions from the rPPO-NMN audit ([[02_rppo_nmn_wiring]] F1–F5)

Carry these into the port as hard requirements, not repeats of the debt:

- **F1**: whitelist `modulation.type` at construction (`raise` on anything outside
  `{"Multiplicative", "PreActivation", "FiLM"}`) — do not copy the silent
  else-is-Multiplicative fallback into `build_agent`.
- **F3**: require the modulation block's presence in dreamer_srl agent configs; only an
  explicit `type: null` means baseline (per the no-fallback protocol, and unlike the
  rPPO `.get()` None-conflation).
- **F2/F4**: no dead or silently-defaulted keys — don't carry `percept_bias_init` into
  FiLM configs (the code discards it), hard-index every modulation key (no
  `.get(..., default)`).
- **F5 lesson**: state explicitly which optimizer owns the modulator (decision 1) in both
  the config comments and the plan — don't leave a `lr_modulator`-style key that nothing
  reads.

### B.5 Size estimate

| Piece | Estimate | Basis |
|---|---|---|
| Modulator network | ~0 new lines (reuse) — or ~50 if trimmed | `neuromodulator.py:203-379` (187 lines) already live |
| Modulated GRU cell | ~90 lines | archive drop-in, near-verbatim |
| `agent.py` (encoder hook, RSSM threading, `build_agent`) | ~150–250 | NNX encoder+WM wiring was ~150; dreamer_srl adds observe/imagine scans |
| `train.py` (observe/imagine mod threading in `one_train_step`) | ~80–150 | NNX trainer threading ~120 |
| `dreamer_srl_main.py` (Player, config reads, WandB metrics) | ~100–150 | Player state + done_mask reset + metrics fan-out |
| `eval.py` | ~20–30 | mirrors Player |
| Configs (1–2 NMN variants) + DEVIATION_LOG rows | ~60 | 7-key block + comments |
| Tests (cell parity `gate_bias=None`, mod_h reset gating, baseline bit-identity) | ~150–250 | matches the parity-fix test style |
| **Total** | **~600–900 lines across ~6 code files + configs + tests** | NNX-NMN measured footprint ~550 lines / 4 files + config |

This is a 1-plan, single-developer-cycle port **after** the three B.3 decisions are made
(they are design calls, not implementation effort). The checkpoint payload, jitted scan
path, and curriculum machinery absorb the new module transparently; the offline-eval gap
(D2) is orthogonal but worth closing in the same era — an NMN study on dreamer_srl with
no offline eval would repeat the rPPO history of leaning entirely on in-training curves.

---

## Verdict

**Part A:** the live Dreamer's in-training eval is faithful — right environment, right
observations, deterministic policy, sound PRNG, honest survival steps, reusable
recordings. The debt is peripheral but real: doubled Eval metrics when both passes run
(D1, worth fixing before the next long run), a silently ignored eval-parallelism knob
(D3), and — the strategic gap — **no offline evaluation or resume path exists for
dreamer_srl checkpoints at all** (D2 + known optimizer-momentum row): today the training
process is the only judge of its own runs, plus a dream-visualizer.

**Part B:** the NMN port is ready to be planned, not ready to be coded: ~600–900 lines
with both reference implementations providing most parts (the Dreamer modulator class is
already live code; the modulated GRU cell is a 90-line drop-in), but three design
decisions gate everything — which loss trains the modulator under dreamer_srl's
three-optimizer split (rPPO's shared-loss answer does not transfer), whether modulation
is encoder-only or reaches the RSSM/imagination, and where `mod_h` gets reset in each of
collection / training sequences / eval. Every hook needs a pre-declared deviation-log row,
and the rPPO audit's config-boundary hardening (type whitelist, mandatory keys, no dead
knobs) should be built in from the first commit.

Reviewed by: code-reviewer (live-path audit step 4/4, Fable 5, 2026-07-10)
