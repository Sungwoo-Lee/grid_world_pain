---
title: "Live-path audit step 3/4 — rPPO-NMN (neuromodulated) variant end-to-end wiring"
topic: diagnosis
status: active
created: 2026-07-10
last_updated: 2026-07-10
---

# rPPO-NMN wiring — config-to-gradient audit of the neuromodulated variant

## Purpose (plain-language entry point)

This is step 3 of a four-step audit of the code paths the project actually trains and
evaluates with. It walks the **neuromodulated ("NMN") variant** of the main agent
end-to-end for the first time: the YAML knobs under `agent.modulation.*`, how they build
the modulator network (a small recurrent net that produces per-neuron gains/biases, a
memory-gate bias, and an action temperature), whether the modulator behaves identically
during training, loss replay, and evaluation, and whether its numbers reach the logs and
checkpoints faithfully.

**Headline:** the modulated forward path itself is correct and train/eval-consistent
(verified by code walk **and** an empirical probe). What the audit found instead lives in
the **config layer**: an unknown modulation-type string silently builds a *different*
architecture instead of erroring, and one advertised modulation knob
(`percept_bias_init`) is silently ignored in every live FiLM config. Also corrected: the
step-3 brief assumed a heteroscedastic ("Kendall & Gal") auxiliary loss exists — **it does
not**; the "het" in config names means perceptual-noise *heterogeneity* environments, and
the modulator is trained solely by the shared PPO loss. 8 new findings (0 High, 2 Med-Low,
5 Low, 1 nit), 8 known items re-confirmed, and an explicit sound-list below.

Scope fence honored: base-h_state lifecycle, checkpoint/H1-restore internals, curriculum
roster staleness (step-2 N1/N2), GAE/MC math (H4), and the collector reset-key PRNG reuse
(known, scheduled) were **not** re-derived; they are cited where NMN state rides on them.

---

## Correction to the brief (read first)

**There is no NMN auxiliary loss.** Grep of the live stack
(`src/models/recurrent_ppo_trainer.py`, `neuromodulator.py`, `recurrent_ppo_network.py`)
finds no heteroscedastic / precision / log-variance term anywhere. The total loss is
exactly `policy + vf_coef·value + ent_coef·entropy` (`recurrent_ppo_trainer.py:184`). The
modulator's parameters receive gradients only *through* those three terms, via the
modulated forward (FiLM γ/β on the encoder, gate-bias into the GRU, temperature on the
logits). Consequences: (a) there is no aux-loss coefficient key to audit; (b) "does the
aux loss leak into the trunk" is moot — trunk and modulator deliberately share one loss
and one Adam optimizer; (c) the `*_het_*` config filenames
(`recurrent_ppo_nmn_het_film_g1.yaml`, `..._tempceil10.yaml`, `..._het_unmod.yaml`)
denote the **noise-heterogeneity sweep** (P5 heterogeneous perceptual-noise environments;
see the config headers), not heteroscedastic regression.

## NEW findings

### F1 — Unknown `modulation.type` strings silently build a *Multiplicative* network instead of raising (Med-Low, latent)

**Where:** `src/models/recurrent_ppo_network.py:211-218` (only `"FiLMNoNorm"` is
explicitly rejected; no whitelist), `src/models/neuromodulator.py:92-95, 99, 106, 125`
(`FiLM`/`PreActivation` string equality gates head construction and γ-bias),
`recurrent_ppo_network.py:145-146, 169-171, 187-189` (`else: # Multiplicative` fallback
in every injection site).

**What happens.** Every branch on the type string treats "not FiLM, not PreActivation"
as Multiplicative. A typo'd config (`type: "Film"`, `"film"`, `"FILM"`, or any future
name) therefore constructs without error and trains a sigmoid-gated Multiplicative
modulator with γ-head bias = the config's `percept_bias_init` (3.0 → gates ≈ 0.95) and
**no β heads at all**. Empirically verified (probe, 2026-07-10): `type: "Film"`
constructs cleanly, builds no `head_unimodal_add`, keeps γ bias 3.0, and runs forward.
The run's WandB would even log `modulator/gamma_*` plausibly; only the missing
`modulator/beta_*` metrics (gated at `train.py:1388` on the same exact-string match)
would hint at the swap. This is the registry's noise-mode-typo class (silent
spelling-variant feature swap). **No live config is affected** — all 11 modulated configs
spell `"FiLM"` exactly — hence Med-Low latent, not High.

**Suggested:** a whitelist raise in `ActorCriticRNN.__init__` (and/or
`NeuromodulatorRNN.__init__`): `if mod_type not in ("Multiplicative", "PreActivation",
"FiLM"): raise ValueError(...)` — the `FiLMNoNorm` guard already sets the pattern.

### F2 — `modulation.percept_bias_init` is a dead key in every live NMN config (Low, dead-key class)

**Where:** all 11 live FiLM configs set `percept_bias_init: 3.0`
(`configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g{1,2,4,8,16,32,64,128}_screen.yaml`,
`recurrent_ppo_nmn_film_g1_tempceil5.yaml`, `recurrent_ppo_nmn_het_film_g1.yaml`,
`recurrent_ppo_nmn_het_film_g1_tempceil10.yaml`); the code overrides it for FiLM:
`src/models/neuromodulator.py:92-95` (`_percept_bias = 1.0` when `type == "FiLM"`),
consumed at `:97-98, 104-105`.

**What happens.** The FiLM γ head's bias is forced to 1.0 (identity/pass-through init in
linear space) regardless of the YAML value — probe-confirmed (config 3.0 → actual bias
1.0). The override is **deliberate design** ([[FILM_MODULATION_PLAN]] specifies exactly
this), so no run trained wrong. But the configs carry 3.0 with no "ignored under FiLM"
comment, and `NMN_PERFORMANCE_DIAGNOSIS_v8` §2.2 lists `percept_bias_init: 3.0` among
the anchored hyperparameters of the controlled config — a reader auditing the sweep's
controlled variables is told a knob is set that the code discards. Under FiLM the key is
live *nowhere* (it only feeds sigmoid-mode biases). **Suggested:** drop the key from FiLM
configs or annotate it `# ignored under FiLM (code forces 1.0 identity init)`; a
one-line warning print at construction would close the class.

### F3 — A missing/typo'd `agent.modulation` block silently trains the unmodulated baseline (Low, latent)

**Where:** `train.py:767-771` — the comment says the key "MUST be defined in config,
even if empty/null", but the read is `config.get('agent.modulation')`, which returns
`None` for an absent key exactly as for `type: null`. Mirrored at `evaluation.py:382-384`
and `scripts/eval/eval_rollout.py:900-902`.

**What happens.** An NMN launch whose agent config misspells the block (`modulations:`,
wrong indentation under `agent:`) constructs the plain baseline with no error — the only
tell is the `Neuromodulation: DISABLED (baseline)` console line (`train.py:780`), absent
under `--quiet`. Because eval sides read the same saved config the same way, eval stays
consistent with training (no train/eval divergence — that is why this is Low, not High).
Distinct from F1: F1 is a bad *value*, F3 is a missing *key*. **Suggested:** require the
key's presence (fail if `'modulation' not in config['agent']`) while still allowing
`type: null` as the explicit baseline switch — matching the comment's stated intent.

### F4 — `percept_add_bias_init` is a silent `.get(..., 0.0)` default, contradicting the no-fallback protocol (Low)

**Where:** `src/models/recurrent_ppo_network.py:262`
(`modulation_config.get('percept_add_bias_init', 0.0)`); every other modulation key uses
hard indexing (`['mod_hidden_size']` etc., `:258-265`) that raises loudly when absent.

**What happens.** No live config sets the key, so every FiLM β head initializes at bias
0.0 — which *is* the design's pass-through value, so behavior is correct today. But the
one modulation knob with a silent default is also the one a sweep would most plausibly
add; a typo'd attempt (`percept_add_bias: 0.5`) would silently no-op. Project rule is
`get_mandatory`/hard-index for critical params. **Suggested:** hard-index it and add the
key (value 0.0) to the live configs, or document the default in the config guide.

### F5 — `agent.lr_critic` is a dead key on the live rPPO path; critic and modulator actually train at `lr_actor` (Med-Low)

**Where:** every NMN config sets `lr_actor: 0.0005` / `lr_critic: 0.0001`; `train.py:509`
reads **only** `agent.lr_actor` into `lr`, and `train.py:796-803` builds a **single**
Adam over all `nnx.Param` — trunk, critic head, and modulator alike. `lr_critic` has zero
consumers in live code (repo-wide grep).

**What happens.** The YAML advertises a 5×-lower critic learning rate (a PyTorch-stack
inheritance where actor/critic had separate optimizers), but the live JAX path trains
critic *and modulator* at 0.0005. No silent distortion of a declared-and-working feature
— the code never claimed to split — but every NMN experiment doc that lists
`lr_critic: 0.0001` as a controlled hyperparameter (e.g. v8 §2.2's anchored table)
describes a setting that does not exist. Uniform across all runs, so comparability
holds. **Status: NEW** (no registry row; area-01 did not flag it; v4–v8 diagnosis docs
list the key without noting deadness). **Suggested:** delete the key from rPPO configs
or register it as a known dead key; decide explicitly whether the modulator should share
`lr_actor` (today's implicit choice).

### F6 — `hierarchical_params.unimodal_overrides` is dead (Low)

**Where:** set in every NMN config (`unimodal_overrides: {visual: [128,128], olfaction:
[128,128]}`); zero consumers — `ObservationEncoder` reads only `default_mlp` and
`multimodal_hub` (`src/models/recurrent_ppo_network.py:84-97`); the grouped Phase-1
encoder (`GroupedMLP`) structurally cannot give per-sense layer sizes (one shared
`[num_groups, in, out]` weight tensor, `:38-53`).

**What happens.** Today the override values equal `default_mlp`, so there is no
behavioral gap — but a config that *changes* them (e.g. a bigger visual encoder) would
silently get the default. Dead-key class. **Suggested:** remove from configs or raise on
presence. (Sibling nits, same class, out of modulation scope but in the same files:
`fc_layers`, `actor_fc_layers`, `critic_fc_layers`, `frame_stack` are also unread on the
rPPO path — the actor/critic heads are hard-coded one hidden layer of `hidden_size`,
`recurrent_ppo_network.py:249-254`.)

### F7 — `modulator/gamma_*` metric names are only literally "gamma" under FiLM (nit)

**Where:** `train.py:1378-1381` logs raw `mod_info.z_unimodal` / `z_multimodal` as
`gamma_uni_*` / `gamma_multi_*`.

**What happens.** Under FiLM the applied gain *is* the raw signal
(`recurrent_ppo_network.py:166-168`), so for **all live configs the logged numbers are
exactly what the network applied** — faithful. Under Multiplicative/PreActivation the
applied gain would be `sigmoid(z)` while the raw pre-sigmoid `z` is logged under the
"gamma" name (a reader would see "gamma_mean 3.0" for a gate of 0.95). Informational
only while FiLM is the only live type.

### F8 — eval_rollout's stochastic policy mode would reuse one PRNG key for every step of an episode (Low, latent; eval path)

**Where:** `scripts/eval/eval_rollout.py` `_run_episode` (`:68-118`): the loop calls
`policy_fn(obs, carry, rng_key, ...)` with the **same** `rng_key` every step; under
`eval_policy_mode: "stochastic"` the policy closure (`:1000-1008`) passes that key to
`jax.random.categorical` each step. Same pattern in `_run_episode_with_recording`.

**What happens.** In stochastic mode, each step's action is sampled with an identical
key — the "randomness" across steps comes only from logits changing, collapsing the
intended sampling distribution (e.g. near-uniform logits would repeat the same
pseudo-draw). **Dormant:** the default is `"deterministic"` (`:789`), the batched path
hard-rejects stochastic (`:813-817`), and no live study uses stochastic eval — argmax is
temperature-invariant, so NMN eval is unaffected today. Fix is the standard
`key, sub = jax.random.split(key)` per step. Not modulation-specific, but recorded here
because this audit owned the NMN eval walk.

## KNOWN items re-confirmed (cited, not re-counted)

| Known item | Where confirmed | Note |
|---|---|---|
| FiLM shared gain across senses = INTENDED | `recurrent_ppo_network.py:164-168` (γ broadcast `[..., None, :]` over the sense axis) | Registry "Confirmed NOT a bug" row; unchanged |
| LSTM + modulation silently drops gate-bias injection | `recurrent_ppo_network.py:326-330` (LSTM branch ignores `z_memory`) | 02_rppo_stack 4b; all live NMN configs set `rnn_type: "GRU"` — still Low |
| Entropy computed on temperature-scaled logits = intended | `recurrent_ppo_trainer.py:156-160` + temp division inside `model.__call__` (`recurrent_ppo_network.py:337`) | 02_rppo_stack 4d; consistent with tempceil study designs |
| `memory_clip` enforced | `neuromodulator.py:167` (`jnp.clip(z_mem, ...)`) | Registry historical row (`014a195`→`ea4bb6e`); post-clip value is both injected and logged |
| `mod_grad_norm` bare-`except: pass` | `recurrent_ppo_trainer.py:326-335` | 02_rppo_stack 4c; **probe-verified the extraction WORKS today** (`'modulator' in grads` → True; norm 0.16 ≠ 0 on flax 0.12.4) — the metric is live, the fragility remains |
| Temperature floor: `temp_clip[0]` below 0.5 is dead (`softplus+0.5`) | `neuromodulator.py:169-173` | `5caa0df` cosmetic note; live configs set floor exactly 0.5, ceilings 3/5/10 all binding |
| Collector reset-key PRNG reuse | `recurrent_ppo_trainer.py:244` | 02_rppo_stack Finding 2; known, scheduled with A1; out of scope here |
| WandB `define_metric` case mismatch — `modulator/*` falls to the catch-all axis | `train.py:680-683` vs `:1377-1394` | Area-01 F9 / step-2 known row; modulator curves plot against the default axis |

## Config-key coverage table

Every key in the live `agent.modulation` blocks, plus the code-side keys configs omit:

| Key | Present in live configs | Read where | Status |
|---|---|---|---|
| `type` | all (FiLM ×11; `null` ×2 baselines) | `train.py:769-771`; `recurrent_ppo_network.py:211-212, 257-259`; `train.py:1388` (β-metric gate); `evaluation.py:382-384`; `eval_rollout.py:900-902` | **LIVE** — but no whitelist (F1); exact-string `"FiLM"`/`"PreActivation"` matches everywhere |
| `mod_hidden_size` | all (16) | `recurrent_ppo_network.py:258` (hard index) → `NeuromodulatorRNN` GRU + heads | LIVE |
| `grouping_size` | all (sweep axis 1–128) | `recurrent_ppo_network.py:260` → `neuromodulator.py:82` (`ceil(hidden/g)` groups), repeat/slice `:147` | LIVE — g128→1 group, g1→128 groups, matches config comments |
| `percept_bias_init` | all (3.0) | `recurrent_ppo_network.py:261` → `neuromodulator.py:92-95` | **DEAD under FiLM** (F2) — overridden to 1.0 in every live modulated config |
| `percept_add_bias_init` | **none** | `recurrent_ppo_network.py:262` `.get(..., 0.0)` | **SILENTLY DEFAULTED** to 0.0 (F4) — benign value, protocol violation |
| `memory_bias_init` | all (0.0) | `recurrent_ppo_network.py:263` → memory-head bias `neuromodulator.py:111-115` | LIVE |
| `temp_clip` | all (`[0.5, 3/5/10]`) | `recurrent_ppo_network.py:264` → `neuromodulator.py:173` clip after `softplus+0.5` | LIVE — floor 0.5 coincides with the natural softplus floor (known); ceilings binding |
| `memory_clip` | all (`[-2, 2]`) | `recurrent_ppo_network.py:265` → `neuromodulator.py:167` | LIVE |
| *(block absent / `type: null`)* | 2 baseline configs | None-gate at all three read sites | LIVE baseline switch — but absence of the whole block is indistinguishable from it (F3) |

Adjacent dead keys carried by the same NMN config files (not `modulation.*`):
`lr_critic` (F5), `hierarchical_params.unimodal_overrides` (F6), `fc_layers` /
`actor_fc_layers` / `critic_fc_layers` / `frame_stack` (F6 sibling note).

## Empirical probe (run 2026-07-10, `grid_world_pain` env, flax 0.12.4 / jax 0.9.0.1)

Tiny modulated net (FiLM, hidden 16, mod_hidden 8, g=4, hierarchical 2-sense breakdown,
LayerNorm on):

1. **Train-shape vs eval-shape forward parity:** two-step rollout, batched
   `model(x[N], h[N])` (the eval/batched-rollout shape) vs per-env unbatched
   `model(x_i, h_i)` (the train-collection shape under vmap). Max |Δ| — logits 5.1e-4,
   value 1.1e-4, temperature 6.2e-5: pure f32 reduction-order noise from batched-vs-vector
   matmuls across 5+ layers, no semantic divergence. Greedy eval is argmax (and
   temperature-scaling-invariant), so this cannot flip evaluated behavior.
2. **F2 confirmed:** constructing with `percept_bias_init=3.0`, FiLM γ-head bias is 1.0.
3. **F1 confirmed:** `type="Film"` constructs with no error, no β heads, γ bias 3.0.
4. **mod_grad_norm live:** `'modulator' in grads` → `True`; `optax.global_norm` of the
   modulator sub-tree = 0.16 (nonzero) under the current flax — the known bare-except has
   not silently zeroed the metric as of today.

## Explicitly checked and found sound (non-findings)

- **Modulator state lifecycle is everywhere the task-RNN state's sibling** — one pytree
  `(task_h, mod_h)` from `model.initial_state` (`recurrent_ppo_network.py:367-384`), so
  every audited base-h_state mechanism covers it automatically: per-done reset in
  collection (`recurrent_ppo_trainer.py:258` `_h_reset_on_done`, generic tree_map) and in
  loss replay (`:164`); stage-transition/resume re-inits in `train.py` (step-2 audit);
  eval batched stats-pass per-done slot reset (`evaluation_core.py:690-692`, tree_map
  with a fresh `initial_state(1)`); eval video pass fresh state per episode
  (`evaluation_core.py:400-401`); `eval_rollout` fresh carry per episode (`:1002-1003`)
  and the batched runner's post-hoc truncation at first done (h continues past done but
  those steps are discarded, `eval_rollout.py:~410-425`). The step-2 N2 caveat
  (single-config resume pairs stale h with fresh envs) applies to the modulator hidden
  identically — same pytree, same row, not re-counted.
- **Loss replay reproduces the modulated forward exactly.** The collector stores the
  *pre-forward* `(task_h, mod_h)` carry per step (`recurrent_ppo_trainer.py:286`);
  `h_init = h_states[0]` (`:387`) seeds the loss scan, which calls the same
  `model(obs, h)` — modulator GRU, γ/β, gate-bias, and temperature are all recomputed
  identically (first epoch bit-comparable; later epochs differ only via updated params,
  which is standard PPO ratio semantics).
- **Bootstrap value forwards handle the modulated 4-tuple correctly** — GAE per-step
  (`recurrent_ppo_trainer.py:237`) and MC window-edge (`:303`) unpack
  `_, value, _, _` and use the un-reset `(task_h, mod_h)` `h_new`/`boot_h` — correct
  modulated continuation semantics.
- **Temperature math (post-`5caa0df`).** `clip(softplus(raw) + 0.5, lo, hi)`
  (`neuromodulator.py:172-173`); applied as `logits / temperature` with shape `(...,1)`
  broadcasting over actions (`recurrent_ppo_network.py:337`); collection sampling,
  replay log-probs, and entropy all see the same scaled logits — internally consistent;
  greedy eval invariant.
- **Grouping arithmetic** — `num_groups = ceil(128/g)`, `repeat(g)[:hidden]` maps group
  *i* to neurons `[i·g,(i+1)·g)`; config comments (g1→128 groups … g128→1 group) match
  the code; probe re-verified construction at g=4.
- **Logging faithfulness (beyond F7).** `z_memory` is logged post-clip = the injected
  gate-bias; `temperature` logged post-clip = the applied divisor; β metrics gated
  exactly on the types that emit nonzero β (`train.py:1388-1394`); `mod_info` comes from
  collection-time trajectories (`train.py:1222`), i.e. the signals the behavior-generating
  policy actually applied that iteration.
- **Checkpoint round-trip.** Modulator heads, GRU, and the `nnx.Param` baselines are
  ordinary members of `nnx.state(model, nnx.Param)` (`train.py:1984`); the H1 restore
  module round-trips them with fatal mismatch checks (step-2 sound-list);
  `eval_rollout.py:965-990` independently does a strict missing-key/shape check that
  would fail loudly on a modulated-vs-baseline mismatch.
- **Eval-side rebuild applies modulation, not just constructs it.** All three eval paths
  read the **whole** `agent.modulation` dict from the saved config (post-`2ad9104`
  whole-dict read: `evaluation.py:382-396`, `eval_rollout.py:900-923`) with the same
  None-gating as `train.py:769-771`, build the same `ActorCriticRNN`, and their only
  forward entry points are `model.__call__` itself (`evaluation_core.py:35` via
  `generic_inference`, `eval_rollout.py` via `get_action_and_value_nnx` /
  `_rollout_scan_jit`) — there is no separate eval forward that could skip the
  injections. During-training eval reuses the in-memory model object outright
  (`train.py:2015-2036`).
- **No dead-reckoned type fall-through in construction vs forward** — the same
  exact-string gates decide head construction and forward branch, so a given (valid)
  type is self-consistent between what is built and what is applied.
- **PPOConfig/static-argnum discipline** — modulation influences only the model object,
  never the static `PPOConfig`; no recompile hazard added by NMN.

## Verdict

**The live NMN path trains, replays, logs, saves, and re-evaluates the same modulated
computation.** The forward is train/eval-consistent (probe: differences at f32 noise
level), the modulator's hidden state rides the audited h_state lifecycle everywhere, its
parameters ride the audited checkpoint contract, and its WandB metrics are the applied
tensors for every live (FiLM) config. Nothing found distorts a live run. The debt is at
the **config boundary**: the type string has no whitelist (a typo silently swaps the
architecture, F1), one advertised knob is silently ignored under FiLM (F2), a missing
modulation block silently means "baseline" against the code's own comment (F3), and the
config files carry a small museum of dead keys (`lr_critic`, `unimodal_overrides`,
`percept_bias_init`) that misdescribe what live runs actually do — worth one small
validation-hardening pass before the next NMN sweep is designed off these YAMLs. Also on
record: the assumed heteroscedastic auxiliary loss does not exist; "het" configs are
noise-heterogeneity environments.

Reviewed by: code-reviewer (live-path audit step 3/4, Fable 5, 2026-07-10)
