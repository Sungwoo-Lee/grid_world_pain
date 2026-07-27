---
title: "Modality-Hierarchical Encoder/Decoder for dreamer_srl (3 designs, flat default)"
topic: dreamer
status: active
created: 2026-07-28
last_updated: 2026-07-28
aliases: [dreamer_hier_encoder_plan]
---

# Modality-Hierarchical Encoder/Decoder for dreamer_srl — three designs

> **Status**: PLANNED
> **Opened**: 2026-07-28
> **Related**: [[evaaa_vs_gridworld_algorithm_inversion]] (dimension #24, cause R1), [[SYNTHESIS_20260727]] (investigation synthesis), [dreamer_multimodal_obs_encoding_and_recon_loss](../../../project/concepts/dreamer_multimodal_obs_encoding_and_recon_loss.md) (professor-rl memo — resolves symlog + loss form), [[DEVIATION_LOG]] (D-018 to be appended), [[dreamer_srl_settings_regime_critique]]

---

## Context

Our recurrent-PPO baseline and our in-house DreamerV3 (`dreamer_srl`) do not see the world the same way. rPPO's encoder is **modality-aware**: the environment tells it the observation is made of 7 named sensor blocks (satiation, two nociception channels, smell, touch, its own last movement, vision), and each block gets its own small network before a fusion hub combines them. Dreamer's encoder is **modality-blind**: one shared MLP over the same 27 numbers. A comparative analysis against the EVAAA benchmark flagged this as a hand-designed advantage we gave rPPO and never gave Dreamer (the critique's dimension #24, cause R1).

Whether copying that bias into Dreamer helps is genuinely contested: this plan's initial reasoning favored a depth-symmetric mirror, a `professor-rl` memo argues for branching the encoder but keeping a shared decoder trunk with thin per-sensor output heads (a deep per-modality decoder branch for a 1-dim sensor trains on 1/27 of the reconstruction signal), and the field convention (official DreamerV3) is one shared MLP for *all* vector inputs — i.e. exactly what we do today. The **user's decision (2026-07-28, final) is to implement THREE designs**: (1) today's flat architecture, unchanged, as the default; (2) the branched encoder with the memo-recommended thin per-key decoder heads; (3) the same branched encoder with the depth-symmetric per-modality decoder branches. Designs 2 and 3 share an identical encoder and differ **only in decoder depth**, so the decoder question is settled empirically. The success criterion is finding one configuration that learns better, not a controlled attribution. Declared deviation D-018; implementation is the `developer`'s; nothing touches the 12 live training jobs.

## Where this stands relative to convention (do not oversell)

Checked directly (memo §2.2 + coordinator's convention check of `danijar/dreamerv3@main`): the official DreamerV3 has **one** `enc:` and **one** `dec:` config block, no per-key architecture settings, and a single global `loss_scales.rec: 1.0`. Its only observation split is by **data type** — `cnn_keys` (images → CNN) vs `mlp_keys` (vectors → MLP) — and *all vector keys share one MLP* (`DictConcat` → shared net, `rssm.py:216-219`). Multimodal Dreamer derivatives (Sense-Imagine-Act LiDAR+RGB; CLIP-text + MAE-visual + conv-proprio stacks) branch encoders only across heterogeneous data types. The one true per-modality-encoder-with-fusion precedent is the Global Workspace world model (arXiv:2502.21142), whose claimed benefit is robustness to a *missing* modality, not raw performance. **Our 27 dims are all one data type, so canonical DreamerV3 would use one shared MLP — i.e. today's flat encoder. The branched encoder is a bespoke import of rPPO's inductive bias, not convention-following**, and the memo's confidence that it materially improves survival is 0.35. Designs 2/3 exist because the argument cannot be settled on priors.

## Resolved design inputs (professor-rl memo, 2026-07-28)

[dreamer_multimodal_obs_encoding_and_recon_loss](../../../project/concepts/dreamer_multimodal_obs_encoding_and_recon_loss.md) closes the two slots left open in the previous revision:

- **Symlog (was SLOT-A) — RESOLVED: uniform symlog on all 27 dims before any split; exempt nothing** (memo §3, confidence 0.9). Symlog ≈ identity near the origin, so one-hot {0,1} → {0, 0.693} stays perfectly separated; and symlog already mildly *down-weights* the large-count visual channels (~4.7× worst-case gradient-share shrink vs 2.1× for [0,1] scalars) — the direction we want, so exempting bounded modalities would be backwards. One precedented refinement — declaring genuinely discrete keys (our proprioception one-hot) as discrete → skip symlog → categorical head, as official DreamerV3 does — is **OPTIONAL / future, not part of this change**.
- **Loss form (was SLOT-B) — RESOLVED: we ship the single concatenated `[.., 27]` tensor and the single `po = {"obs": ...}` key; `loss.py` is untouched.** The memo proves (§4) that a per-key split is *exactly* loss-neutral for any factorised likelihood aggregated by sum (sum over a partition = sum over the whole; the τ=1e-8 clamp is elementwise and survives partitioning) — but it is **not bit-identical**: fp32 re-association gives ~1e-6 relative discrepancy, so any obs-loss check across decoder variants needs a *tolerance*, not a fix. Per-modality diagnostics are computed as logging-only slices of the concatenated tensor (File Change 7), which buys the memo's §8 instrumentation without touching the loss.
- **Reweighting — documented NON-GOAL** (memo §5, confidence 0.85). Precedent: DreamerV3 trains Minecraft's 1-dim health/hunger scalars against a 12,288-dim image at equal per-number weight (~10⁴:1 vs our worst-case 8:1) and solves the task; danijar's code has a per-key loss-scale dict deliberately overwritten with uniform `rec: 1.0`. If reweighting is *ever* added it must be **magnitude-preserving** (`Σ w_k d_k = 27`), because free bits clip the KL at an ABSOLUTE 1-nat floor — changing ‖L_obs‖ silently re-tunes the KL/reconstruction balance (memo §5.3, §10.1). It would be its own experiment arm, never bundled with this change (memo §10.6).
- Also load-bearing (memo §6.1): **per-key linear decoder heads are an exact reparameterisation of the single 27-dim head** — same function class, same parameter count (27×256+27), same gradients (a dense layer's output rows already receive independent gradients); only the RNG draw *order* differs. This is what makes the reparameterisation unit test possible (§Test Plan).

## Analysis

### The two encoders today (verified 2026-07-28)

| | rPPO (`src/models/recurrent_ppo_network.py:72-128`) | dreamer_srl (`src/algorithms/dreamer_srl/agent.py:1115-1195`) |
|---|---|---|
| Input handling | Receives `observation_breakdown` (ordered dict {modality → width}) from `train.py:1021` via `get_observation_breakdown(params)` (`src/environment/sensor.py:350`) | Receives only `obs_dim=int` probed from a reset (`dreamer_srl_main.py:707`); **breakdown never plumbed in** |
| Phase 1 | `GroupedMLP`: each modality zero-padded to `max_in`, own MLP branch → `hidden_size`, single einsum kernel (`GroupedLinear`, lines 11-30) | none — single block |
| Phase 2 | `multimodal_hub`: concat(7 × hidden_size) → MLP → `hidden_size` | none |
| Block idiom | Linear (+bias) → optional LayerNorm → ReLU | Linear(bias=False) → LayerNorm(eps=1e-3) → SiLU, Hafner truncated-normal init (sheeprl parity) |
| Input transform | none | `symlog(obs)` before the MLP (`agent.py:1190`) — kept uniform per memo ruling |
| Config switch | `agent.encoding_mode` + `agent.hierarchical_params` (`configs/models/recurrent_ppo/recurrent_ppo_XS.yaml:29-41`) | none — architecture fixed |
| Decoder | n/a | `MLPDecoder` (`agent.py:1203-1293`): latent(1280 at XS) → [Linear→LN→SiLU]×`mlp_layers` → single `Linear(256→27)`, head init `uniform_init_weights(1.0)` |
| Modulation hooks | `forward_with_modulation` (FiLM/gating) | none — dreamer_srl has no NMN hooks; the port deliberately excludes the modulation path |

Breakdown for basic03/04: {Satiation 1, Interoceptive Nociception 1, Extero Nociception 1, Olfaction 5, Collision 5, Proprioception 6, Visual 8} = 27, `max_in` = 8.

**rPPO YAML footgun (matters for the mirror):** `hierarchical_params.unimodal_overrides` exists in the YAML but the unified `ObservationEncoder` **does not read it** — only `default_mlp` and `multimodal_hub` are consumed (`recurrent_ppo_network.py:84-97`). The Dreamer port copies what the code *does*: all 7 branches share `default_mlp`. Do not port the vestigial key.

### Native multi-key support in sheeprl (verified; corroborated by memo §2.1)

- **Encoder — native multi-key is still flat.** `MLPEncoder.forward` (`vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:149-151`): concat all keys → **one shared MLP**. Declaring 7 `mlp_keys` gives the encoder zero per-modality structure. The encoder hierarchy must be bespoke.
- **Decoder — native multi-key is per-key heads.** `MLPDecoder` (L251-278): shared body → `nn.ModuleList([nn.Linear(dense_units, d_k) ...])` (L274). Design 2's `heads` decoder is therefore the **upstream idiom**, and (memo §6.1) an exact reparameterisation of today's single head.

Full dict-obs plumbing (per-key buffer/batch/`po` dict) was evaluated and **rejected**: the entire data plane is built on a single `[.., 27]` tensor pinned by 141 parity tests, and the per-key `po` dict is loss-equivalent to the single key (memo §4) — all designs keep single-tensor plumbing, with modality structure internal to the encoder/decoder modules.

### Call sites that constrain the design (all verified)

- `WorldModel.observe` (`agent.py:1671`): `jax.vmap(self.encoder)(obs.reshape(T*B, -1))` — the new encoder must be **vmap-safe**, signature `__call__(obs[..., 27]) → [..., output_dim]` with an `output_dim` attribute; `build_agent` passes `encoder.output_dim` into the RSSM (line 2051; with `hidden_size: 256` at XS it coincides with today's 256, so the representation model is unchanged in practice).
- Acting path (`dreamer_srl_main.py:332`): same vmap call.
- Loss (`train.py:730`, `loss.py:546`): single `po` key, untouched in all designs (memo ruling above). Decoder contract in every design: one `[..., 27]` symlog-space tensor.
- `build_agent` (`agent.py:1939-2063`) is also called by `scripts/eval/eval_rollout.py:1455` (already computes `d_obs_breakdown` ~1427), `scripts/eval/dreamer_srl_probe_eval.py`, `scripts/dreamer/visualize_dream.py`, `scripts/dreamer/dreamer_srl_offline_wm_test.py` (both already import `get_observation_breakdown`), `scripts/fixtures/gen_cp3_fixtures.py`, and ~10 test files. Flat-mode callers keep working because the new argument is optional.
- Pre-existing quirk (out of scope, flat path untouched): `build_agent:2060` builds the flat decoder from the **encoder's** `dense_units`/`mlp_layers`; the `world_model.decoder` YAML keys are currently ignored.

### Precedent for the config mechanism

D-017 (two-hot bin range, `DEVIATION_LOG.md` row 106): **optional agent-config keys read with `.get(default=parity value)`** — not `get_mandatory` — because existing configs must keep running bit-identically and the parity tests pin the default path. Within an enabled feature, sub-keys are read with hard indexing (missing key → `KeyError`).

## The three designs (user decision, 2026-07-28, final)

| # | Design | Encoder | Decoder | WM enc+dec params (XS) | Total model |
|---|---|---|---|---|---|
| 1 | **Current (default)** | shared MLP over 27 (today; sheeprl parity; DreamerV3 convention) | single `Linear(256→27)` head (today) | ≈ 7 k + 335 k | ≈ 3.17 M (= today) |
| 2 | **Hier + heads** (`mirror_encoder_in_decoder: false`) | rPPO-style per-modality branches → fusion hub | shared trunk → thin per-key `Linear` heads (sheeprl-native idiom; memo recommendation §7) | ≈ 0.76 M + 335 k | ≈ 3.92 M |
| 3 | **Hier + mirror** (`mirror_encoder_in_decoder: true`) | *identical to design 2* | hub-mirror → deep per-modality branch MLPs (depth-symmetric; user's original design) | ≈ 0.76 M + 0.89 M | ≈ 4.47 M |

Designs 2 and 3 share an identical encoder and differ **only in decoder depth** — the memo predicts 3's 1-dim branches starve (§6.2); design 3 tests that empirically. There is **no** flat-encoder + per-modality-decoder combination. (Param counts at rPPO-XS sizing — `default_mlp [256]`, hub `[128,128]`, `hidden_size 256`, latent 1280; exact per-module counts printed by `build_agent` at startup for run manifests. No param-parity framework, per the earlier user decision — the spread 3.17–4.47 M is accepted.)

### Config surface (final key spec)

```yaml
algo:
  world_model:
    encoding_mode: hierarchical   # OPTIONAL; "flat" (default = today, bit-identical) | "hierarchical"
    hierarchical_params:          # required iff encoding_mode == hierarchical (KeyError if absent)
      mirror_encoder_in_decoder: false  # true  -> deep per-modality decoder branches (design 3)
                                        # false -> shared trunk + thin per-key output heads (design 2)
                                        # read ONLY under hierarchical; REQUIRED there (no default)
      default_mlp: [256]          # per-modality branch hidden layers (rPPO XS recipe)
      multimodal_hub: [128,128]   # fusion hub (encoder); reversed as the hub-mirror (design 3 decoder)
      hidden_size: 256            # branch output width = hub output width = encoder output_dim
    encoder: { dense_units: 256, mlp_layers: 1 }   # existing keys — flat + shared-trunk sizing, untouched
    decoder: { dense_units: 256, mlp_layers: 1 }   # existing keys — flat path, untouched
```

- `encoding_mode` read with `.get(..., 'flat')` (D-017 pattern) — existing configs construct byte-identical modules. Under `hierarchical`, `hierarchical_params` and its sub-keys (incl. `mirror_encoder_in_decoder`) are hard-indexed (no fallback — the boolean is REQUIRED under hierarchical, so each of the three designs has exactly one config expression).
- **`mirror_encoder_in_decoder` has no effect under `encoding_mode: flat` and gets the same fail-loud treatment:** it lives inside `hierarchical_params`, so setting it under `flat` necessarily means the `hierarchical_params` block is present — which is the rejected combination below (`ValueError`). No silently-ignored key exists.
- **Meaningless combination is REJECTED, not ignored:** if `encoding_mode` is `flat` (explicit or defaulted) and the config nevertheless contains `algo.world_model.hierarchical_params`, `build_agent` raises `ValueError` at config-resolution time. Rationale: a silently inert block is exactly the class of config drift the project's no-fallback rule exists to prevent; failing loudly costs one deleted block, while a logged note gets lost in training output.
- Not duplicating an existing key (`grep -rn "encoding_mode\|mirror_encoder_in_decoder" configs/models/dreamer_srl/` is empty). No `configs/train/default.yaml` change (agent keys, no mandatory read — D-017 disposition). Not a critical-settings registry entry. No `scripts/` file added/moved/renamed/deleted ⇒ `SCRIPTS_DEPENDENCY_MAP.md` untouched.

## Design decisions (carried, updated)

- **Sizing** — resolved (user): rPPO XS recipe as the code consumes it (`recurrent_ppo_XS.yaml:29-41`); no param-parity framework; counts recorded above and printed at startup.
- **Symlog** — resolved (memo): uniform on all 27 dims at encoder entry in every design; no per-modality exemptions; discrete-key categorical heads optional/future.
- **Loss** — resolved (memo): single concatenated tensor + single `po` key shipped; `loss.py` untouched; reweighting a documented non-goal (magnitude-preserving `Σ w_k d_k = 27` constraint recorded for any future revisit).
- **Checkpoint compatibility — broken across designs, new runs only.** The Orbax/nnx tree follows module structure; even the function-equivalent single-head↔per-key-head pair differ structurally (1 head vs 7), so designs 2/3 cannot restore a design-1 checkpoint and vice versa (`PyTreeRestore(partial_restore=True)` tolerates extra keys, not renamed/missing subtrees). Flat-default runs — including the 4 live node-114 jobs and their future resumes — are unaffected. Eval scripts rebuild from the run's saved agent config, so design-2/3 checkpoints are handled automatically once callers pass the breakdown.

## Implementation Plan

### Design

New classes in `src/algorithms/dreamer_srl/agent.py` (no imports from `recurrent_ppo_network.py` — stacks stay decoupled; all new blocks use **Dreamer's idiom**: Linear(bias=False) → LayerNorm(eps=1e-3) → SiLU, Hafner `init_weights`):

1. **`HierarchicalMLPEncoder`** (designs 2+3) — `__call__(obs[..., 27]) → [..., hidden_size]`, `output_dim = hidden_size`.
   `symlog(obs)` (uniform, memo ruling) → static split by breakdown widths (`jnp.split` at `np.cumsum(widths)[:-1]`, vmap-safe) → zero-pad to `max_in`, stack `[..., 7, 8]` → branches: `default_mlp` layers of [GroupedLinear(bias-free) → per-group LN → SiLU] ending in a grouped projection to `hidden_size` → concat `[..., 7*hidden_size]` → hub: `multimodal_hub` layers ending at `hidden_size`. Local bias-free Hafner-init `GroupedLinear` einsum helper (~15 lines; pattern from `recurrent_ppo_network.py:11-30`).
2. **`HeadsMLPDecoder`** (design 2) — flat `MLPDecoder`'s shared trunk unchanged (`decoder.dense_units`/`mlp_layers`); output = 7 per-key `Linear(dense_units → d_k)` heads, kernels `uniform_init_weights(1.0)` (sheeprl L1178 idiom), concatenated in breakdown order to `[..., 27]`.
3. **`MirrorMLPDecoder`** (design 3) — latent → hub-mirror (`multimodal_hub` reversed) → grouped expansion to 7 branches → per-modality `default_mlp` branch MLPs → grouped output projection to each modality's width (output kernels `uniform_init_weights(1.0)`) → concat `[..., 27]`.
4. **`build_agent`** — keyword-only `observation_breakdown: Optional[dict] = None`; switch on `wm_cfg.get('encoding_mode', 'flat')`; under `hierarchical`, `hierarchical_params['mirror_encoder_in_decoder']` (required boolean) selects `MirrorMLPDecoder` (true) vs `HeadsMLPDecoder` (false). Validation: hierarchical + `observation_breakdown is None` → `ValueError` naming the caller fix; `sum(breakdown.values()) != obs_dim` → `ValueError`; flat + `hierarchical_params` present (which subsumes any `mirror_encoder_in_decoder` setting) → `ValueError` (rejected combination, above); `mirror_encoder_in_decoder` missing → `KeyError`, non-boolean → `ValueError`. **The flat branch must be textually untouched and consume the `rngs` stream identically** — mode selection precedes all constructor calls, so default-path init bits cannot move; the fixture-pinned parity suite then proves it.

### File Changes

1. **`src/algorithms/dreamer_srl/agent.py`** (new classes after ~1293; `build_agent` 1939–2063) — as above + per-module param-count print; pass `encoder.output_dim` to RSSM in hierarchical mode.
2. **`src/algorithms/dreamer_srl/dreamer_srl_main.py`** (~707, 781) — compute `get_observation_breakdown(env_params)` after `load_env_params`, assert sum == probed `obs_dim`, pass to `build_agent`; print the modality table when hierarchical (mirror `train.py:1024-1031`).
3. **`scripts/eval/eval_rollout.py`** (dreamer branch ~1455) — pass `observation_breakdown=dreamer_envs[0]["obs_breakdown"]` (already computed ~1427).
4. **`scripts/eval/dreamer_srl_probe_eval.py`** — compute breakdown from its env params; pass through.
5. **`scripts/dreamer/visualize_dream.py`** (~768-800) and 6. **`scripts/dreamer/dreamer_srl_offline_wm_test.py`** (~659) — already import `get_observation_breakdown`; pass it.
7. **`src/algorithms/dreamer_srl/train.py`** (additive, logging-only — memo §8 instrumentation) — in `wm_loss_fn`'s aux dict, per-modality `stop_gradient` diagnostics from slices of the same `[T,B,27]` tensors: symlog-space MSE per key (`wm/recon_mse/<slug>`) **and** per-key symlog-target second moment (`wm/target_var/<slug>`), so the memo's variance-normalised starved-modality signature `L̃_k` is computable at analysis time. Breakdown widths threaded as a static tuple (D-017 mechanism, no new jit boundary). Runs in all designs incl. the flat baseline. If a parity fixture pins the aux-key set, update the fixture, not the metric.
8. **`docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md`** — append **D-018** (same commit as code): opt-in hierarchical encoder (`algo.world_model.encoding_mode`) with two decoder variants selected by the boolean `hierarchical_params.mirror_encoder_in_decoder` (false = shared trunk + thin per-key heads, true = deep per-modality branches) vs sheeprl's flat encoder + single-head decoder; note the per-key-heads variant is sheeprl's own multi-key idiom (L274) and an exact reparameterisation of the single head (memo §6.1), while the branched encoder is a bespoke rPPO-bias import **against** field convention (this doc §convention); default flat = parity, pinned by the bit-identity suite; loss code unchanged; reweighting recorded as a non-goal; status `☐ pending`, flip at senior-developer verification. Frontmatter `last_updated` + header note per that file's convention.
9. **New `tests/algorithms/dreamer_srl/test_hierarchical_encoder.py`** — see Test Plan.

The `developer` may add one smoke config per non-default design (e.g. `configs/models/dreamer_srl/xs_hier_heads_smoke.yaml`, `xs_hier_mirror_smoke.yaml`); experiment configs are `experiment-designer`'s.

### Test Plan

1. **Flat default stays bit-identical (the gate):** full existing dreamer_srl suite green — 141 tests incl. `test_agent.py`, `test_end_to_end_parity.py`, `test_grad_parity.py`, `test_checkpoint.py`, `test_lax_scan_train.py`. No fixture changes except (possibly) the aux-key list from File Change 7.
2. **Reparameterisation unit test (replaces any training-run verification — no GPU cost, same assurance):** construct a flat `MLPDecoder` and a `HeadsMLPDecoder` with the same trunk sizing; copy the flat decoder's trunk weights across, and transplant the flat head's `(W, b)` into the 7 per-key heads as row-blocks (rows partitioned by breakdown order). Assert, on a batch of random latents: (a) decoder outputs identical within `1e-6` relative; (b) `reconstruction_loss` values identical within `1e-6` relative; (c) per-block gradients w.r.t. head weights match the corresponding row-blocks of the flat head's gradient within `1e-6`. Tolerance, never exact-bit — fp32 re-association bounds the discrepancy at ~`27ε ≈ 3e-6` relative (memo §4 point 3, §10.3). This proves the per-key refactor is an exact reparameterisation; if it fails, the plumbing is wrong and neither design 2 nor 3 can be trusted.
3. **New unit tests (same file):** shapes (`[..., 256]` / `[..., 27]`) and finiteness for designs 2+3; vmap-compat as used (`jax.vmap(encoder)` on `[B,27]`); `ValueError` on hierarchical + `observation_breakdown=None`, on breakdown-sum ≠ obs_dim, on flat + `hierarchical_params` present (the rejected combination — including when only `mirror_encoder_in_decoder` is set), and on non-boolean `mirror_encoder_in_decoder`; `KeyError` on missing `hierarchical_params` sub-keys (incl. the required `mirror_encoder_in_decoder`); gradient reaches every branch (encoder branches, mirror-decoder branches — the memo's starvation warning makes this check non-optional); `build_agent` flat with the kwarg supplied ≡ without it (identical param tree); checkpoint save→restore roundtrip per design 2 and 3; flat-checkpoint-into-design-2/3 raises.
4. **End-to-end smoke** (local CPU/free GPU): ~200 steps per design 2 and 3; losses finite; exactly one compile of `one_train_step`.
5. **Speed check** (developer, Implementation Report): flat SPS before/after unchanged (any delta is a red flag); SPS for designs 2 and 3 recorded. Same hardware/config/seed, ≥ 2 k steps past warm-up.

### Live-jobs safety

All edited files are imported (held in memory) by the 4 live node-114 dreamer jobs, not hot-read — NAS edits are safe. **`scripts/eval/render_recordings.py` is hot-read by live jobs and is not touched.** No node access; no live run's config modified; verification entirely local.

### Running it once implemented (hand-off to `experiment-designer`)

Three arms × ≥3 seeds, XS, current best-known cadence settings; design-1 baselines reusable from the in-flight bins±6/replay-ratio experiment (no rerun needed). Primary metric: survival steps at matched env-steps (wall-clock reported alongside; never reward). Per-key diagnostics (File Change 7) adjudicate the memo's predictions: starved-modality signature in design 3 (§6.2) vs interoceptive-reconstruction gains in designs 2/3 (§8 table). The magnitude-preserving reweighting of memo §5.3 is the pre-specified *follow-up* arm **only if** the starved-modality signature fires — never bundled. PI consultation before launch (multi-run experiment).

## Checkpoints

- [ ] After File Change 1: flat `build_agent` param tree byte-identical (`jax.tree_util.tree_map(shape)` compare on the XS config).
- [ ] Existing 141-test suite green **before** writing new tests (catches flat-path drift early).
- [ ] Reparameterisation unit test passes at 1e-6 (outputs, loss, gradients).
- [ ] After File Change 2: flat config smoke prints unchanged `build_agent OK`; design-2/3 smokes print modality table + param counts (expect ≈ 3.17 / 3.92 / 4.47 M).
- [ ] After File Change 7: `wm/recon_mse/*` + `wm/target_var/*` present in a flat smoke; loss values unchanged vs pre-change fixture.
- [ ] D-018 row present in the same commit as the code.

## Implementation Report

> **Implemented by**: —
> **Date**: —

## Verification Report

> **Verified by**: —
> **Date**: —

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: —
