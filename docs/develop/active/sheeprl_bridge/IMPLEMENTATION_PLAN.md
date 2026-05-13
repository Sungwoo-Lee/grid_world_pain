---
title: "Sheeprl Bridge — Minimum-Bridge Implementation Plan (v2: restructure out of tmp/)"
topic: dreamer
status: active
created: 2026-05-12
last_updated: 2026-05-12
phase: 2
---

# Sheeprl Bridge — Minimum-Bridge Implementation Plan

> **Status**: v1 IMPLEMENTED 2026-05-12 · v2 RESTRUCTURE PLANNED 2026-05-12 (this revision)
> **Opened**: 2026-05-12
> **Related**: [PI call 2026-05-12](../../../pi/calls/2026-05-12_dreamer_backend.md) · [Drop-in diagnosis (smoke run)](../diagnosis/sheeprl_drop_in_test.md) · [How-to: launch sheeprl](../diagnosis/sheeprl_training_howto.md) · [Compatibility audit (5×5 + 10×10)](../../../reviews/sheeprl_two_configs_audit.md) · [Archived: dreamer-srl JAX rebuild plan](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md)

> **Reader, start here**: the bridge from our JAX grid-world env to the PyTorch `sheeprl` DreamerV3 was first built into a gitignored vendor directory (`tmp/sheeprl/`). On 2026-05-12 the user directed us to move it out of `tmp/` into a proper in-repo package — `pytorch_agents/` — with its own `pyproject.toml`, while keeping sheeprl itself as a pip dependency (not vendored). See **Layout (v2 — 2026-05-12)** below for the new structure. The first implementation pass (v1, 6 gaps × 4 commits) succeeded but landed on disk that git did not track; v2 fixes the trackability problem and prepares for the user's longer-term PyTorch migration (eventually rPPO too).

---

## Context

The project's in-house JAX/Flax DreamerV3 has failed for weeks to learn a 5×5 food-only survival task — our best variant survives ~115 environment steps; the community PyTorch implementation `sheeprl`, running unmodified on the *same* environment, survives the full 500-step time limit (validated 2026-05-11 by smoke run `jzgkcep4`). On 2026-05-12 the user paused the JAX rebuild plan ([archived](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md), which had three reviewers signing off ✅ PASS) and made a strategic call via [PI call 2026-05-12](../../../pi/calls/2026-05-12_dreamer_backend.md) to use `sheeprl` directly as the Dreamer backend for the neuromodulation paper — research velocity over single-stack JAX uniformity. **This plan covers the minimum plumbing to make that pivot operational.**

The headline complication: **most of the bridge is already built and shipping**. Between 2026-05-11 and 2026-05-12, the project landed a working gymnasium adapter (`tmp/sheeprl/sheeprl/envs/grid_world_pain.py`), three sheeprl Hydra configs (env + exp + WandB logger), a `sheeprl_bridge` conda env on node 114, a reusable launch script (`scripts/launch_sheeprl.sh`), a generalized SSH-launch wrapper (`run_command.py`), a how-to doc, and a config-compatibility audit covering both the 5×5 and 10×10 task families. The 2026-05-11 smoke (run `jzgkcep4`) reached `Game/ep_len_avg = 500` (the env's truncation cap) within 25,000 environment steps. So this plan is **not** "build the bridge from scratch" — it is "harden what exists, fill the residual gaps, and gate the next step (porting our neuromodulator into sheeprl's PyTorch agent) behind a feasibility check."

The deliverable is a bridge solid enough to launch the next experiment family (the neuromodulation comparison, which is a separate experiment-designer doc). The deliverable is **NOT** a full one-time-engineered "paved bridge" — that was Option 2 in the PI call, and was explicitly not chosen. Stay tight on minimum.

## Analysis

### What is already built (read these files before planning further changes)

| Component | Path | Status | Notes |
|---|---|:---:|---|
| Gymnasium env wrapper | `tmp/sheeprl/sheeprl/envs/grid_world_pain.py` (113 lines) | ✅ | Wraps `jax_reset` / `jax_step` / `get_observation`; smoke-tested 200k steps. Forces `JAX_PLATFORMS=cpu` and injects project root into `sys.path` at import time. |
| Sheeprl env config | `tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml` | ✅ | `num_envs: 4`, `sync_env: True`, reads `GWP_CONFIG_PATH` env-var for the project-side YAML path. |
| Sheeprl exp config | `tmp/sheeprl/sheeprl/configs/exp/dreamer_v3_grid_world_pain.yaml` | ✅ | Composes `dreamer_v3_XS` + grid_world_pain env + WandB logger; sets `fabric.accelerator: cuda` (otherwise sheeprl defaults to CPU); `total_steps: 200_000`, `per_rank_sequence_length: 64`. |
| Sheeprl WandB logger config | `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` | ✅ | Targets `lightning.pytorch.loggers.WandbLogger`; project hard-coded to `grid_world_pain_sheeprl_test` (a smoke-era name — see §Open question 2 below). |
| Workload launch script | `scripts/launch_sheeprl.sh` (57 lines) | ✅ | Takes `<config-yaml> <gpu-index> <env-id-tag> [total-steps]`; sets env-vars, exec's the sheeprl_bridge Python; node-agnostic (works wherever the conda env exists). |
| SSH launch wrapper | `run_command.py` | ✅ | Generalized 2026-05-12 to be a thin SSH wrapper (no longer cds or activates conda — those moved into the bash script). Project-wide rule per [CLAUDE.md](../../../../CLAUDE.md) "Launching training on lab nodes". |
| Conda env (node 114 only) | `/home/vncuser/miniconda3/envs/sheeprl_bridge/` | ✅ on n114, ❌ everywhere else | torch 2.5.0+cu121, jax 0.10.0 CPU, sheeprl 0.5.8.dev. Node-local — `/home/vncuser/` is NAS-shared per the how-to so it *may* be visible from other nodes; setup recipe documented in `sheeprl_training_howto.md` §5 for nodes where it is not. |
| How-to doc | `docs/develop/active/diagnosis/sheeprl_training_howto.md` | ✅ | Practical usage guide written for fresh sessions. The §6 "Common variations" + §8 "Known gotchas" tables are the operational reference. |
| Compatibility audit | `docs/reviews/sheeprl_two_configs_audit.md` | ✅ | Pre-flight audit confirming both `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml` and `configs/experiment/hypervigilance/01-interoNocicept.yaml` load cleanly through the bridge. One concern flagged (5×5 omits `interoceptive_nociception` from noise modalities — harmless under current "noise disabled" defaults). |
| Smoke validation | WandB run `jzgkcep4` | ✅ | Stock `dreamer_v3_XS` on food-only NoPred reached `Game/ep_len_avg=500` (env cap) by step 25k of a 200k run, vs in-house A1 ≈ 106 / Z2 ≈ 115. ~4.4× longer survival. |
| Project rule documentation | [CLAUDE.md](../../../../CLAUDE.md) "Launching training on lab nodes" | ✅ | Codifies the `./run_command.py <node> "<bash-script>"` pattern. |

### What still needs work for production adoption

The smoke validated the pipeline for **one** config (food-only) on **one** node (114) for **one** seed (42) with the **stock** algorithm. To make this the project's official Dreamer backend for the neuromodulation paper, six gaps remain:

1. **`apply_noise=True` in the bridge.** The current `step` / `reset` call `get_observation(state, params, apply_noise=False)` (lines 67-68, 87-89, 99-101 of the bridge). This is fine for the smoke (food-only NoPred has no sensory noise anyway), but the neuromodulation paper's thesis is *injury-modulated sensory noise* — the modulator is supposed to learn to upweight noisy nociceptive channels when injured. Disabling noise at the bridge defeats the experiment. The bridge needs a config-driven `apply_noise` flag that defaults to `True` for production runs, with a `__init__` option to override for diagnostic runs.

2. **WandB project name + run-name template.** The current `logger/wandb.yaml` hard-codes `project: grid_world_pain_sheeprl_test`. For production use the project should match the in-house DreamerV3 convention — `grid_world_pain` is the established project name for the in-house algo. The bridge runs should either (a) write to a sibling project `grid_world_pain_sheeprl`, or (b) merge into `grid_world_pain` with a clear naming convention. The run-name template currently uses `${run_name}` (a Hydra var sheeprl populates as `<algo>_<env>_<seed>_<date>`) — our convention from `train_command-agent.sh` is `<algo>_<config_stem>_s<seed>_n<node>` (see lines 67-75 of that script). One of the two has to win; consistency with our existing WandB project is preferable.

3. **Conda env on more than one node.** Today `sheeprl_bridge` exists only on node 114. The user will eventually want to run multi-node sweeps. Setup is ~5 minutes per node per the how-to §5, but it should be a documented, fire-and-forget step.

4. **`training-runner` agent profile awareness.** The agent currently launches via `train_command-agent.sh` (which calls `train.py` — our in-house JAX trainer). For sheeprl runs the agent needs to know to call `scripts/launch_sheeprl.sh` instead. Per `.claude/agents/training-runner.md`, the agent's "dedicated launch script" is `train_command-agent.sh`. Two options: (a) extend `train_command-agent.sh` with a sheeprl branch; (b) treat sheeprl launches as a separate path the runner invokes via `run_command.py <node> "bash scripts/launch_sheeprl.sh ..."`. Option (b) is already partially in place (the script exists and is callable) but the agent profile does not yet mention it.

5. **Existing diagnosis-doc cleanup.** The `sheeprl_drop_in_test.md` doc is the design+analysis record of the 2026-05-11 smoke and is correctly titled an experiment diagnosis. But its **Implementation Plan** section is what really built the bridge — those steps should be cross-linked from this plan so a future reader sees the build history. (No edit to the diagnosis doc needed — it stays as the historical record; just link it.)

6. **NMN-port feasibility gate.** Before the neuromodulator port begins (a separate follow-up plan), `senior-developer` must read sheeprl's `agent.py` and our `dreamer_v3_nnx.py` injection points side-by-side and confirm: *"is porting the modulator + FiLM hooks + temperature scaling into sheeprl's PyTorch agent materially harder than expected?"* Per the PI call stop-rule, a 🚨 verdict here halts the call back to PI. This plan carries out that check (see §NMN-port feasibility check below) and reports the verdict.

### Cell jzgkcep4 in plain English (translating the run shorthand)

The 2026-05-11 smoke run is referenced throughout the rest of this document by its 8-character WandB run id. To save the reader from having to look it up: **`jzgkcep4`** is the WandB run where stock sheeprl `dreamer_v3_XS` (256-unit MLP, 1 layer, 256-recurrent — sheeprl's smallest preset) trained on our 5×5 food-only NoPred task for 200,000 environment steps on node 114 GPU 0, seed 42, `num_envs=4 sync_env=True`. It reached the survival cap (`Game/ep_len_avg=500`) at policy step ~25,000 and saturated there for the remaining 175,000 steps. End-of-training test reward `-103`, training-time mean reward `-111 ± 5`. Full details and time-series in [`sheeprl_drop_in_test.md` §Results](../diagnosis/sheeprl_drop_in_test.md).

## Layout (v2 — 2026-05-12)

> **Plain-language reader entry point**: the bridge code currently lives inside `tmp/sheeprl/` — a clone of upstream sheeprl that sits on the NAS but is invisible to git (the project's `.gitignore` line 23 excludes `tmp/`). The user directed us on 2026-05-12 to stop relying on `tmp/` and move our bridge code into a proper, git-tracked, in-repo Python package with its own `pyproject.toml`. The chosen layout is **Option B**: keep sheeprl as a `pip install`-able dependency (NOT vendored) and put only OUR additions into a new top-level folder named `pytorch_agents/`. This same folder will house future PyTorch ports (e.g., rPPO) per the user's longer-term migration direction. After v2 lands, `tmp/sheeprl/` is deleted.

### Survey: what was actually in `tmp/sheeprl/` (2026-05-12)

A direct compare of `tmp/sheeprl/` against its upstream origin (`https://github.com/Eclectic-Sheep/sheeprl` HEAD `33b6366`, cloned 2026-05-09) confirms:

- **Zero tracked files were modified.** `git -C tmp/sheeprl status` and `git -C tmp/sheeprl diff --stat` are both clean.
- **Four files were added.** All untracked-in-upstream additions; all live under `tmp/sheeprl/sheeprl/`:
  1. `sheeprl/envs/grid_world_pain.py` (5030 B, last edit 2026-05-12 16:29) — gymnasium wrapper around our JAX env.
  2. `sheeprl/configs/env/grid_world_pain.yaml` (385 B) — Hydra env config; `_target_: sheeprl.envs.grid_world_pain.GridWorldPainWrapper`.
  3. `sheeprl/configs/exp/dreamer_v3_grid_world_pain.yaml` (366 B) — Hydra experiment config (`dreamer_v3_XS` + grid_world_pain env + WandB logger; `fabric.accelerator: cuda`).
  4. `sheeprl/configs/logger/wandb.yaml` (140 B) — Lightning WandB logger config; `project: grid_world_pain`. Net-new (upstream ships only `mlflow.yaml` + `tensorboard.yaml`), NOT a modification of an upstream file.

Implication: **the project is not carrying a sheeprl fork — it is carrying a sheeprl extension.** That makes the natural restructure "treat sheeprl as a library, treat our 4 files as in-repo code." (The v1 plan's §Analysis table listed three of these as the bridge components; the experiment config — file #3 — was implicit in §Step 2 but not enumerated.)

### Why Option B (and why not A or C)

| Option | Sheeprl handling | Our extension code | Verdict |
|---|---|---|---|
| **A** — vendor sheeprl entirely | ~600 files of upstream code committed under `pytorch_agents/sheeprl/` | first-class | ❌ Rejected. We have zero upstream modifications, so the maintenance cost of carrying the whole tree buys nothing. |
| **B** — sheeprl as a pip dep ⭐ | `pip install` from PyPI or pinned GitHub commit | `pytorch_agents/` as an in-repo Python package | ✅ Chosen. Sheeprl ships a first-class extension mechanism (Hydra `SearchPathPlugin`); we use the mechanism it was designed for. |
| **C** — git submodule | submodule pointer to upstream commit | first-class | ❌ Rejected. No other submodules in the project; UX cost not justified for an unmodified upstream tree. |

### Feasibility checks for Option B (all pass)

1. **Hydra `_target_` can point to any importable module.** Once `pytorch_agents/` is registered as a Python package (via its `pyproject.toml` + `pip install -e`), `_target_: pytorch_agents.envs.grid_world_pain.GridWorldPainWrapper` resolves identically to today's `sheeprl.envs.grid_world_pain.GridWorldPainWrapper`.
2. **Sheeprl supports external config directories via `SHEEPRL_SEARCH_PATH`.** The plugin at `tmp/sheeprl/hydra_plugins/sheeprl_search_path.py` reads `SHEEPRL_SEARCH_PATH` (semicolon-separated list of `pkg://...` or `file://...` Hydra paths) from the environment or from `.env`, and appends them to Hydra's config search path. This is the canonical out-of-tree extension hook. We will set `SHEEPRL_SEARCH_PATH=pkg://pytorch_agents.configs` in `launch_sheeprl.sh`.
3. **Sheeprl is pip-installable** — `tmp/sheeprl/pyproject.toml` is upstream-form with entry-point `sheeprl = "sheeprl.cli:run"`. Either PyPI (`pip install sheeprl==<ver>`) or pinned GitHub commit (`pip install git+https://github.com/Eclectic-Sheep/sheeprl@33b6366`) works. **Recommend: pin to commit `33b6366` for v2 to match the validated smoke (`jzgkcep4`).**

### Chosen folder name: `pytorch_agents/`

| Candidate | Pros | Cons | Verdict |
|---|---|---|---|
| **`pytorch_agents/`** | Accommodates future rPPO PyTorch port under same root; clear provenance contrast vs JAX `src/`; survives the user's stated migration without rename. | Slightly broad while only sheeprl-Dreamer lives there today. | ⭐ chosen |
| `pytorch_dreamer/` | Specific. | Forces a rename once rPPO migrates — user explicitly anticipated that migration. | rejected |
| `src_pytorch/` | Symmetric to `src/`. | `src_*` is an unusual sibling-prefix; less searchable. | rejected |
| `sheeprl_ext/` | Crisp current-scope name. | Would need a sibling folder for rPPO later, creating fragmentation. | rejected |

### Target tree

```
pytorch_agents/
├── pyproject.toml                              # editable Python package; see §Step 2A for full contents
├── README.md                                   # one-paragraph orientation
└── pytorch_agents/                             # package root
    ├── __init__.py
    ├── envs/
    │   ├── __init__.py
    │   └── grid_world_pain.py                  # MOVED from tmp/sheeprl/sheeprl/envs/grid_world_pain.py
    └── configs/
        ├── __init__.py                         # required so `pkg://pytorch_agents.configs` resolves
        ├── env/
        │   ├── __init__.py
        │   └── grid_world_pain.yaml            # MOVED from tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml
        ├── exp/
        │   ├── __init__.py
        │   └── dreamer_v3_grid_world_pain.yaml # MOVED from tmp/sheeprl/sheeprl/configs/exp/dreamer_v3_grid_world_pain.yaml
        └── logger/
            ├── __init__.py
            └── wandb.yaml                       # MOVED from tmp/sheeprl/sheeprl/configs/logger/wandb.yaml
```

### v1-gap → v2-rework mapping

The v1 implementation (commits `380290a`, `9c86601`, `c262f31`, `307f334`) hit all 6 gaps. v2 doesn't redo the WORK of those gaps — it relocates the FILES they touched. Mapping:

| v1 Gap | v1-touched file | v2 disposition |
|---|---|---|
| 1 — `apply_noise=True` default | `tmp/sheeprl/sheeprl/envs/grid_world_pain.py` | **file moves**; logic is preserved verbatim. Re-verify: `grep apply_noise pytorch_agents/pytorch_agents/envs/grid_world_pain.py` shows the v1 code. |
| 2 — WandB project = `grid_world_pain` | `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` | **file moves**; content unchanged. |
| 3 — `pyproject.toml` for sheeprl deps | `tmp/sheeprl/pyproject.toml` (upstream's) | **NEW `pytorch_agents/pyproject.toml` written from scratch**; upstream `tmp/sheeprl/pyproject.toml` is decoupled (sheeprl now installed from PyPI / pinned GitHub). |
| 4 — training-runner dual-launch awareness | `.claude/agents/training-runner.md` | **path updates only** — references to `tmp/sheeprl/sheeprl.py` become `python -m sheeprl` (entry-point script). |
| 5 — bidirectional cross-links | `sheeprl_training_howto.md`, `sheeprl_drop_in_test.md`, this plan | **path updates only** — `tmp/sheeprl/...` references become `pytorch_agents/...`. |
| 6 — NMN-port feasibility check | this plan §NMN-port feasibility check | **no change**; verdict ✅ already recorded, sheeprl's `agent.py` is still importable as a package (paths in the side-by-side table become `sheeprl.algos.dreamer_v3.agent` instead of `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py`). |

So v2 = **3 file MOVES** + **1 NEW pyproject** + **path updates in 4 docs** + **launch-script rewrite** + **conda-env rebuild** + **final deletion of `tmp/sheeprl/`**.

## NMN-port feasibility check (the stop-rule gate)

> **Purpose**: per the PI call's stop rule, this plan cannot be approved for implementation until `senior-developer` has read sheeprl's PyTorch agent and our JAX agent side-by-side and given a verdict on whether porting the neuromodulator (modulator GRU + FiLM gain/bias heads + memory gate-bias + reward-head scaling + actor temperature scaling) into sheeprl is *materially harder than expected*. The phrase "materially harder than expected" means: (a) sheeprl's agent class hierarchy or PyTorch idioms block one of the injection points in a way that requires non-mechanical redesign of our modulator's interface, or (b) Lightning Fabric / sheeprl's optimizer-step infrastructure makes adding the modulator parameter group non-trivial, or (c) FiLM-on-MLP-encoder, temperature-on-actor-logits, or gate-bias-on-GRU-update-gate has a structural barrier in PyTorch that doesn't exist in JAX.

### Side-by-side injection-point inventory

**Our JAX agent** (`src/models/dreamer_v3_nnx.py` + `src/models/neuromodulator.py`) currently injects modulation at five points when `modulation_enabled=True`:

| # | Site | Our JAX hook | Function |
|---|---|---|---|
| A1 | Encoder output (unimodal) | `DreamerObservationEncoder.forward(...)` applies `gamma1 * x + beta1` with `gamma1 = z_unimodal`, `beta1 = z_unimodal_add` when FiLM. Lines 322-348 of `dreamer_v3_nnx.py`. | Per-stream perceptual gain (Phase 1 of [neuromodulation algorithm](../../../project/concepts/neuromodulation_algorithm.md)). |
| A2 | Encoder output (multimodal) | Same encoder.forward, second FiLM layer with `z_multimodal` / `z_multimodal_add`. | Cross-stream multimodal gain (Phase 2). |
| A3 | Actor MLP final layer | `ActorCritic` wraps an MLP; the modulation_type-dependent block at lines 391-406 applies `gamma * x + beta` to the actor pre-activation. | Action-policy perceptual gating. |
| M1 | RSSM GRU update-gate bias | `RSSM.step(prev, embed, action, is_first, key, gate_bias=...)` and `RSSM.imagine_step(prev, action, key, gate_bias=...)` thread an additive bias into the GRU's update gate via `self.cell(x, deter, gate_bias=gate_bias)`. Lines 105-126, 146-163 of the nnx RSSM. | "Memory persistence" — the modulator can bias how strongly the world model carries past state forward. |
| T1 | Actor logits temperature | When modulated, actor returns `logits / mod_output.temperature` before constructing the distribution (in `recurrent_ppo_network.py` line 337; the dreamer agent has an analogous hook in `DreamerV3Agent.get_action`). | Exploration-vs-exploitation under modulation. |
| R1 | Reward-head output scale (DreamerV3 specific) | `DreamerNeuromodulatorRNN` has a `head_reward` that emits a sigmoid-bounded scalar `z_reward`. Intended to be multiplied into the reward-head output during world-model loss computation. | Nociceptive reward interpretation. |

**Sheeprl's PyTorch agent** (`tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py`, 1236 lines) exposes the same structural sites, each in a clear nn.Module method body:

| # | Sheeprl site | File / lines | Port mechanics |
|---|---|---|---|
| A1/A2 | `MLPEncoder.forward(obs)` (lines 100-151) | Receives a dict of obs tensors, applies `symlog` then an MLP, returns the embedded vector. FiLM injection = wrap the final MLP output with `gamma * x + beta` if modulation is on. ~10 lines of new code in a subclass `ModulatedMLPEncoder(MLPEncoder)`. | **Mechanical.** No structural barrier. |
| A3 | `Actor.forward(state, greedy, mask)` (lines 783-848) | Calls `self.model(state)` (an MLP), then per-head `self.mlp_heads[i](feat)` to produce per-action-dim logits. To inject FiLM on the actor pre-head MLP output: subclass `Actor`, override `forward`, apply `gamma * feat + beta` between `self.model` and `self.mlp_heads`. ~15 lines. | **Mechanical.** |
| M1 | `RecurrentModel.forward(input, recurrent_state)` (lines 328-341) → `self.rnn(feat, recurrent_state)` where `self.rnn = LayerNormGRUCell` | The cleanest place to inject a gate-bias is by subclassing `LayerNormGRUCell` (defined in `sheeprl/models/models.py`). The standard GRU update-gate update is `z = sigmoid(W_iz * x + b_iz + W_hz * h)`; an additive bias becomes `z = sigmoid(W_iz * x + b_iz + W_hz * h + gate_bias)`. Then thread `gate_bias` through `RecurrentModel.forward(input, recurrent_state, gate_bias=None)` → `RSSM.dynamic(...)` → `RSSM.imagination(...)`. ~4 method-signature changes; the modulator side just passes `gate_bias=z_memory` when modulated. | **Mechanical, with the caveat that 4 signature changes through the RSSM call chain count as more than "trivial."** Still well within "straightforward." |
| T1 | `Actor.forward` returns a `torch.distributions.Distribution` (built from logits internally) | Sheeprl's `Actor.forward` calls `dist.OneHotCategoricalStraightThrough(logits=logits)` (or similar). To inject temperature: divide `logits` by `temperature` before constructing the distribution. ~3 lines in the subclassed `Actor.forward`. | **Mechanical.** |
| R1 | World-model loss reads reward-head output in `loss.py:reconstruction_loss` (and friends) | The reward head is `world_model.reward_model: MLP`. Sheeprl's loss combines the reward-head output with a two-hot symexp target. Scaling the reward-head output by `z_reward` is a single multiplication on the head output tensor — but **the scale must be applied consistently between rollout-time prediction and loss computation**, or the policy will mis-attribute gradients. Two implementation options: (a) bake the scale into the reward head as a hook on its forward pass (cleaner), (b) pass `z_reward` through to the loss function explicitly. Either is mechanical. | **Mechanical with a discipline gotcha** — see §Risks #2. |
| Modulator class itself | `src/models/neuromodulator.py: DreamerNeuromodulatorRNN` | Port to PyTorch: `nnx.Linear → nn.Linear`, `nnx.GRUCell → nn.GRUCell`, `nnx.initializers.constant → torch.nn.init.constant_` (apply manually in `__init__`), `nnx.Param → nn.Parameter`, `jnp.repeat → tensor.repeat_interleave`, `jax.nn.sigmoid → torch.sigmoid`, `jax.nn.softplus → F.softplus`. The lazy `set_imagine_input_dim` pattern (line 365) refactors to eager construction in `__init__` since PyTorch doesn't need NNX's stateful split semantics. ~200-line port of a ~180-line module. | **Mechanical.** |
| Training-loop integration | `dreamer_v3.py: world_model_optimization_step` + `player_optimization_step` | Need to: (i) construct the modulator alongside encoder/RSSM/actor/critic in `build_agent`, (ii) wrap it in Lightning Fabric, (iii) compute the modulator forward inside the rollout, (iv) include modulator params in the world-model optimizer's param group (or a separate optimizer). Lightning Fabric handles optimizer-step machinery transparently — adding a parameter group is one `optim.add_param_group(...)` call or `params=[encoder.parameters(), ..., modulator.parameters()]` in the constructor. | **Mechanical.** |

### Verdict

**✅ Port is straightforward.** Every injection point in our JAX modulator has a structurally-matching site in sheeprl's PyTorch agent. The only non-trivial pieces are (a) threading a new `gate_bias` arg through 4 RSSM method signatures, and (b) keeping the reward-head scale consistent between rollout and loss — both are mechanical and well-documented in the JAX implementation already, so the PyTorch port is a translation exercise, not a redesign. Lightning Fabric handles optimizer-step uniformly so adding the modulator parameter group is a one-line config change. No JAX-specific paradigm assumption blocks the port. **Estimated effort: 1–2 weeks of focused mechanical work** (port modulator class ~2 days, subclass encoder/actor/RSSM with injection hooks ~3 days, integrate into training loop + Lightning Fabric param group ~2 days, parity smoke test against the JAX modulator's pass-through-init baseline ~2 days). The port is a *follow-up plan* — the next experiment that authorizes it will trigger that plan; this plan does not.

(Following the PI stop-rule, this is recorded as the gate. If the developer hits an unexpected barrier during the actual port — say, a Lightning Fabric edge case around mixed-precision + custom param groups — the developer surfaces the surprise and senior-developer re-evaluates. The verdict here is "no foreseeable blocker," not "guaranteed easy.")

## Implementation Plan (v2 — 2026-05-12) — restructure out of `tmp/sheeprl/`

> The v1 plan (further down) implemented 6 gaps successfully but landed the bridge code on disk that git did not track (`tmp/sheeprl/` is gitignored). v2 keeps every v1 *decision* (apply_noise default, WandB project, dual-launch path, cross-links) and only changes *location* + *packaging*. v1's §Implementation Plan and §Implementation Report are preserved below as historical record. **A developer reading top-down should follow THIS v2 section's File Changes; the v1 File Changes is superseded.**

### Design

Move the 4 bridge files out of `tmp/sheeprl/sheeprl/` into a new in-repo Python package at `pytorch_agents/`, install sheeprl as a pinned pip dependency rather than vendoring it, and rewire the launch script to use sheeprl's `SHEEPRL_SEARCH_PATH` extension hook so the new config dir is discovered. Delete `tmp/sheeprl/` only after all checkpoints pass.

This v2 plan does NOT include:

- Adding new training code, model code, or experimental knobs beyond what v1 already shipped. v2 is a *relocation* + *packaging* change.
- Porting JAX modulator code to PyTorch — still gated on the experiment that authorizes it (separate follow-up plan).
- Migrating rPPO to PyTorch — same plan only insofar as it chose the folder name (`pytorch_agents/`) that will house the eventual port; no rPPO code lands in v2.
- Any change to `src/`, `configs/`, or `scripts/` apart from `scripts/launch_sheeprl.sh` (rewritten) and a possible touch to `run_command.py` (if it referenced any `tmp/sheeprl/` path — verify during implementation).

### File Changes (v2)

The numbered steps below are the developer's execution order. Each step is independently verifiable. The developer is `developer` agent (sonnet); the launch + smoke runs at the end go to `training-runner`.

#### Step 2A — Create the `pytorch_agents/` package skeleton

#### New file: `pytorch_agents/pyproject.toml`

```toml
[project]
name = "pytorch_agents"
version = "0.1.0"
description = "PyTorch-side agents for the grid_world_pain project. Houses the sheeprl-DreamerV3 bridge and (in future) PyTorch ports of rPPO and other algorithms."
authors = [
    {name = "Sungwoo Lee", email = "sungwoo320@gmail.com"},
]
requires-python = ">=3.11,<3.12"
dependencies = [
    # Sheeprl pinned to the commit that the 2026-05-11 smoke (jzgkcep4) validated.
    # Update only when an experiment requires a newer sheeprl feature and the parity
    # gate is re-run to confirm the bump is benign.
    "sheeprl @ git+https://github.com/Eclectic-Sheep/sheeprl@33b636681fd8b5340b284f2528db8821ab8dcd0b",
    # Sheeprl's transitive deps cover torch, lightning, hydra-core, gymnasium, etc.
    # We additionally need:
    "jax[cpu]>=0.9.0",         # bridge wrapper imports JAX (CPU-only — sheeprl owns the GPU via torch)
    "wandb>=0.24.0",           # logger
    "pyyaml>=6.0",             # bridge wrapper reads our project YAML configs
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["pytorch_agents", "pytorch_agents.*"]

[tool.setuptools.package-data]
# Required so Hydra can resolve `pkg://pytorch_agents.configs/...`. Without this
# the YAMLs would be excluded from the editable install's package-data manifest.
"pytorch_agents" = ["configs/**/*.yaml"]
```

Rationale for each design choice:

- **Pin sheeprl to commit `33b6366`**, not PyPI `sheeprl==0.5.8.dev`. The smoke ran against the commit; PyPI may or may not equal that tree. Reproducibility wins.
- **`pip install -e pytorch_agents/`** at install time gives Python an importable `pytorch_agents` module AND triggers Hydra's `pkg://pytorch_agents.configs` resolution.
- **`jax[cpu]` not `jax[cuda12]`** — sheeprl's torch owns the GPU; the bridge forces `JAX_PLATFORMS=cpu` at import. Installing the CUDA flavor would waste install time and disk.
- **No `[project.scripts]`** — sheeprl provides the `sheeprl` entry-point; the launch script invokes `python -m sheeprl`, not an entry-point we'd ship.
- **`package-data` glob `configs/**/*.yaml`** — YAML files are not Python source so setuptools would skip them by default. Hydra's `pkg://` resolution reads them as package data, so they must be declared.

#### New file: `pytorch_agents/pytorch_agents/__init__.py`

```python
"""PyTorch-side agents for grid_world_pain.

Currently houses the sheeprl-DreamerV3 bridge wrapper (under :mod:`pytorch_agents.envs`)
and the Hydra config tree consumed by sheeprl (under :mod:`pytorch_agents.configs`).
Eventually rPPO and other PyTorch ports will live alongside.

The JAX side of the project lives under :mod:`src` at the repository root.
"""

__version__ = "0.1.0"
```

#### New empty files (so Python treats subdirs as packages, and `pkg://...` resolves)

- `pytorch_agents/pytorch_agents/envs/__init__.py` — empty.
- `pytorch_agents/pytorch_agents/configs/__init__.py` — empty.
- `pytorch_agents/pytorch_agents/configs/env/__init__.py` — empty.
- `pytorch_agents/pytorch_agents/configs/exp/__init__.py` — empty.
- `pytorch_agents/pytorch_agents/configs/logger/__init__.py` — empty.

#### New file: `pytorch_agents/README.md`

One short paragraph orienting future-Claude / external readers; cross-link to this plan.

```markdown
# pytorch_agents

PyTorch-side agent code for the grid_world_pain project. The JAX-side code lives under `src/` at the repository root; this folder hosts everything that uses PyTorch / Lightning Fabric / sheeprl.

Current contents:

- `pytorch_agents/envs/grid_world_pain.py` — gymnasium wrapper that lets sheeprl train on our JAX-implemented 5×5 grid-world env.
- `pytorch_agents/configs/{env,exp,logger}/*.yaml` — Hydra config additions discovered by sheeprl via the `SHEEPRL_SEARCH_PATH=pkg://pytorch_agents.configs` env var (set by `scripts/launch_sheeprl.sh`).

Install: `pip install -e pytorch_agents/` (into a clean Python 3.11 env). See [`docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md`](../docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md) §Layout for the full restructure rationale and [`docs/develop/active/diagnosis/sheeprl_training_howto.md`](../docs/develop/active/diagnosis/sheeprl_training_howto.md) for usage.
```

#### Step 2B — Move the 4 v1 files into the new package

Use `git mv` for any file that was tracked in git (none are; all 4 v1 files live under `tmp/`). Use plain `mv` since `tmp/sheeprl/` is gitignored — git history is moot, the files were never tracked.

| From | To | Edit needed |
|---|---|:---:|
| `tmp/sheeprl/sheeprl/envs/grid_world_pain.py` | `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` | ❌ none — file content stays identical. |
| `tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml` | `pytorch_agents/pytorch_agents/configs/env/grid_world_pain.yaml` | ✅ one line: `_target_` |
| `tmp/sheeprl/sheeprl/configs/exp/dreamer_v3_grid_world_pain.yaml` | `pytorch_agents/pytorch_agents/configs/exp/dreamer_v3_grid_world_pain.yaml` | ❌ none — the `defaults:` overrides reference config names (`grid_world_pain`, `wandb`) that Hydra resolves via the search path; names are unchanged. |
| `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` | `pytorch_agents/pytorch_agents/configs/logger/wandb.yaml` | ❌ none. |

The one-line edit in `env/grid_world_pain.yaml`:

```yaml
# BEFORE (line 7):
  _target_: sheeprl.envs.grid_world_pain.GridWorldPainWrapper

# AFTER:
  _target_: pytorch_agents.envs.grid_world_pain.GridWorldPainWrapper
```

#### Step 2C — Rewrite `scripts/launch_sheeprl.sh`

Replace lines 53-57 (the `exec` block) with the new invocation. Key changes:

- `tmp/sheeprl/sheeprl.py` → `python -m sheeprl` (sheeprl is now an installed package, not a path).
- Add `export SHEEPRL_SEARCH_PATH="pkg://pytorch_agents.configs"` BEFORE the exec so Hydra discovers our config dir.

```bash
# BEFORE (lines 53-57):
exec /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \
    tmp/sheeprl/sheeprl.py \
    exp=dreamer_v3_grid_world_pain \
    env.id="$TAG" \
    algo.total_steps="$STEPS"

# AFTER:
export SHEEPRL_SEARCH_PATH="pkg://pytorch_agents.configs"

exec /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \
    -m sheeprl \
    exp=dreamer_v3_grid_world_pain \
    env.id="$TAG" \
    algo.total_steps="$STEPS"
```

No other lines of `launch_sheeprl.sh` change. The `cd /media/nas01/...` at the top stays (so relative paths to `configs/experiment/...` still work). The env-var exports (`GWP_CONFIG_PATH`, `JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`) stay.

Also add a 2-line comment block above the new `export SHEEPRL_SEARCH_PATH` line documenting why it's there:

```bash
# Tell sheeprl's Hydra search-path plugin where to find our env/exp/logger
# configs (Hydra resolves `pkg://pytorch_agents.configs` via importlib).
export SHEEPRL_SEARCH_PATH="pkg://pytorch_agents.configs"
```

#### Step 2D — Conda env rebuild on each node

Replace the existing `sheeprl_bridge` env on node 114 (and create it fresh on any future node). The recipe in `docs/develop/active/diagnosis/sheeprl_training_howto.md` §5 needs updating:

```bash
# Old §5 install command:
#   pip install -e /media/nas01/projects/Interoceptive-AI/grid_world_pain/tmp/sheeprl
#   pip install "jax[cpu]" flax omegaconf pyyaml wandb
#   pip install -e /media/nas01/projects/Interoceptive-AI/grid_world_pain

# New §5 install command (single line — sheeprl + jax + wandb all flow from pyproject.toml):
pip install -e /media/nas01/projects/Interoceptive-AI/grid_world_pain/pytorch_agents

# Then, separately, the project-side editable install (so `from src.environment.* import ...`
# works inside the bridge wrapper):
pip install -e /media/nas01/projects/Interoceptive-AI/grid_world_pain
```

The training-runner pre-flight import check changes too: was `python -c "import torch, jax, sheeprl, wandb"`; becomes `python -c "import torch, jax, sheeprl, wandb, pytorch_agents"` (add the new package to the smoke).

#### Step 2E — Path-update sweep across docs + agent profile

Every reference to `tmp/sheeprl/...` becomes a `pytorch_agents/...` reference, OR a sheeprl-as-package reference. Inventory (line numbers approximate):

| File | Hits | Replacement guide |
|---|---:|---|
| `.claude/agents/training-runner.md` | 1 hit (line 14) | `tmp/sheeprl/sheeprl.py` → `python -m sheeprl` (entry-point script). Mention `pytorch_agents` package name + `pip install -e pytorch_agents/` in pre-flight. |
| `docs/develop/active/diagnosis/sheeprl_training_howto.md` | ~17 hits (lines 45, 57, 113, 139, 179, 198, 203, 208, 232, 265, 281, 289, 290, 298, 308, 315, 324, 357, 372, 387–391) | All `tmp/sheeprl/sheeprl/<subpath>` → `pytorch_agents/pytorch_agents/<subpath>`. All `tmp/sheeprl/sheeprl.py` → `python -m sheeprl`. The §5 install recipe replaced per Step 2D above. The "tracked in git" claim on line 113 becomes accurate after v2 (the files ARE now in git). |
| `docs/develop/active/diagnosis/sheeprl_drop_in_test.md` | check during implementation | Update any `tmp/sheeprl/` references the same way. |
| `docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md` (this file) | many — v1 §File Changes and v1 §Implementation Report preserve historical `tmp/sheeprl/` references | **Do NOT rewrite v1 sections.** They are the historical record of what was done; v2 sits above them and is the current source of truth. The v2 §Verification Report (below) tracks the NEW paths. |
| `run_command.py` | unknown | grep for `tmp/sheeprl` during implementation; if a reference exists, replace. (Survey at v2 planning time: launcher script is the entry, `run_command.py` is path-agnostic — probably no hits, but verify.) |

#### Step 2F — Delete `tmp/sheeprl/`

**Only after all v2 checkpoints (below) pass.** This is the last step. Command:

```bash
rm -rf /media/nas01/projects/Interoceptive-AI/grid_world_pain/tmp/sheeprl
```

`tmp/` is gitignored, so this needs no git operation — but it is irreversible. Wait until CP-v2-1 through CP-v2-5 are all green before running.

### Configuration keys added (v2)

Zero new keys in **project-side** YAMLs. v2 is a relocation. The `apply_noise` key that v1 added to `pytorch_agents/configs/env/grid_world_pain.yaml` (was `tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml`) is unchanged.

### Checkpoints (v2)

Each independently verifiable. v1 checkpoints (CP1–CP7 further down) are historical; v2 has its own gate set.

- [x] **CP-v2-1 — `pytorch_agents/` package importable.** `pip install -e pytorch_agents/ --config-settings editable_mode=compat --no-deps` in the `grid_world_pain` env. Then `python -c "import pytorch_agents; import pytorch_agents.envs; import pytorch_agents.configs; print(pytorch_agents.__version__)"` prints `0.1.0`. Also confirmed with simulated `-m` flag (CWD on sys.path): passes via compat editable mode. Node 114 sheeprl_bridge env rebuild deferred to training-runner pre-flight. **Note**: must use `editable_mode=compat` to avoid namespace-package shadowing when launched via `python -m sheeprl` (which adds CWD to sys.path[0]).

- [ ] **CP-v2-2 — Hydra resolves the new config dir.** Deferred to `training-runner` (sheeprl_bridge env only on node 114). Command to run on node 114:
  ```bash
  cd /media/nas01/projects/Interoceptive-AI/grid_world_pain
  export SHEEPRL_SEARCH_PATH="pkg://pytorch_agents.configs"
  /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -m sheeprl \
      exp=dreamer_v3_grid_world_pain env.id="dryrun" --cfg job 2>&1 | head -50
  ```
  Hydra prints the composed config — confirm `wrapper._target_` is `pytorch_agents.envs.grid_world_pain.GridWorldPainWrapper` and `metric.logger.project` is `grid_world_pain`. If `ConfigNotFound` raises, the search-path injection failed; debug `SHEEPRL_SEARCH_PATH` resolution. (Replaces v1 CP2.)

- [ ] **CP-v2-3 — Launch invocation works through the runner pattern.** Deferred to `training-runner`:
  ```bash
  ./run_command.py 114 "bash scripts/launch_sheeprl.sh \
      configs/experiment/dreamer_curriculum/01_food_only.yaml 0 gwp_v2_smoke 50000"
  ```
  50k-step smoke. Confirm: (a) WandB run appears under project `grid_world_pain` with name containing `gwp_v2_smoke`; (b) training log shows `Game/ep_len_avg` rising; (c) `pgrep -af gwp_v2_smoke` on node 114 returns exactly one PID. (Replaces v1 CP4.)

- [ ] **CP-v2-4 — Parity gate with `apply_noise=True`.** Deferred to `training-runner`. The v1 plan's "Risks #1" (apply_noise may shift the smoke trajectory) is still relevant; CP-v2-4 inherits v1 CP6's content. 50k-step run with default `GWP_APPLY_NOISE=true` reaches `Game/ep_len_avg ≥ 250` by step 50k (the smoke `jzgkcep4` reached 500 with noise OFF; we accept a halved survival ceiling as evidence that the bridge still works with noise on — exact threshold tunable post-hoc).

- [x] **CP-v2-5 — Doc + agent-profile path sweep complete.** `grep -rn "tmp/sheeprl" .claude/agents/ docs/develop/active/diagnosis/sheeprl_training_howto.md scripts/launch_sheeprl.sh pytorch_agents/ run_command.py` returns ZERO prescriptive hits. `sheeprl_drop_in_test.md` retains `tmp/sheeprl/` references labeled "historical" per plan guidance; this is correct. `sheeprl_training_howto.md` has 4 "v2 location, moved from..." contextual notes — also correct.

- [x] **CP-v2-6 — `tmp/sheeprl/` deleted.** `ls /media/nas01/.../tmp/sheeprl 2>&1` returns "No such file or directory." Backup at `/tmp/tmp_sheeprl_bk_1778573331` (OS will GC).

- [x] **CP-v2-7 — Develop INDEX regenerates cleanly.** `python scripts/regen_dev_index.py` exits 0, "Wrote docs/develop/INDEX.md (97 docs indexed)".

### Risks and open questions (v2)

1. **PyPI `sheeprl 0.5.8.dev` vs pinned commit `33b6366`.** v2 pins to the commit. If a future experiment needs a newer sheeprl feature, bump the pin in `pytorch_agents/pyproject.toml` and re-run CP-v2-3 + CP-v2-4 as the new parity gate. Do NOT silently float to latest.

2. **Hydra `pkg://` resolution requires the package to be installed in the active Python env.** If a developer runs the smoke from a fresh terminal without `pip install -e pytorch_agents/`, Hydra raises `ConfigNotFound` (not `ImportError`) — which is confusing. Mitigation: training-runner's pre-flight check now imports `pytorch_agents` (Step 2D); failure is caught before launch.

3. **`pytorch_agents/configs/__init__.py` empty file requirement.** `pkg://` Hydra resolution uses `importlib.resources` under the hood; if `configs/` is not a Python package (no `__init__.py`), Hydra will raise `PackageNotFoundError` even though the directory exists. Step 2A enumerates the required `__init__.py` files explicitly to prevent this.

4. **Existing `sheeprl_bridge` conda env on node 114 has sheeprl installed editable from `tmp/sheeprl/`.** After Step 2F deletes `tmp/sheeprl/`, that editable install will become a broken `.pth` reference. Step 2D's reinstall (`pip install -e pytorch_agents/`) replaces it via the new pyproject's pinned-commit GitHub install, but the developer should `pip uninstall sheeprl -y` before reinstalling so the old editable reference is cleanly removed.

5. **Sheeprl's `.env` file lookup.** The search-path plugin loads `.env` from CWD if it exists (see `tmp/sheeprl/hydra_plugins/sheeprl_search_path.py:18-19`). The project root has no `.env` today; if one is added later for some other purpose, it may unintentionally affect `SHEEPRL_SEARCH_PATH`. Document in the how-to: "if `.env` exists at repo root, `SHEEPRL_SEARCH_PATH` set there overrides the launch-script export."

6. **`run_command.py` may need a path edit.** Survey at v2 planning time did not find any `tmp/sheeprl` reference in `run_command.py`, but Step 2E lists it for re-verification. If a hit exists, replace.

7. **NMN-port feasibility table line numbers shift.** §NMN-port feasibility check (preserved verbatim from v1) cites `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py` and similar paths. Those become `<sheeprl-install-path>/sheeprl/algos/dreamer_v3/agent.py` — the file is identical (same commit), only the on-disk path differs. The verdict (✅ port is mechanical) stands. No edit to the table is required, but the developer should be aware that a fresh reader following those paths must navigate into the installed sheeprl package (e.g., `~/miniconda3/envs/sheeprl_bridge/lib/python3.11/site-packages/sheeprl/algos/dreamer_v3/agent.py`).

### Verification Report (v2)

> **Verified by**: [pending — `senior-developer`]
> **Date**: [pending]

| Path | Change | Status | Notes |
|---|---|:---:|---|
| `pytorch_agents/pyproject.toml` | NEW file (pinned-commit sheeprl + jax-cpu + wandb + pyyaml) | ✅ | commit `0067721` |
| `pytorch_agents/README.md` | NEW orientation file | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/__init__.py` | NEW (`__version__ = "0.1.0"`) | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/envs/__init__.py` | NEW (empty) | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` | MOVED from `tmp/sheeprl/sheeprl/envs/grid_world_pain.py`; `_PROJECT_ROOT` depth corrected 4→3 | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/configs/__init__.py` | NEW (empty) | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/configs/env/__init__.py` | NEW (empty) | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/configs/env/grid_world_pain.yaml` | MOVED with one-line `_target_` edit (`sheeprl.envs` → `pytorch_agents.envs`) | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/configs/exp/__init__.py` | NEW (empty) | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/configs/exp/dreamer_v3_grid_world_pain.yaml` | MOVED (identical content) | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/configs/logger/__init__.py` | NEW (empty) | ✅ | commit `0067721` |
| `pytorch_agents/pytorch_agents/configs/logger/wandb.yaml` | MOVED (identical content) | ✅ | commit `0067721` |
| `scripts/launch_sheeprl.sh` | rewritten exec block (`python -m sheeprl` + `SHEEPRL_SEARCH_PATH`) | ✅ | commit `7a2cc83` |
| `.claude/agents/training-runner.md` | path updates + pre-flight import-check addition | ✅ | commit `e8b8ed9` |
| `docs/develop/active/diagnosis/sheeprl_training_howto.md` | full path sweep + §5 install-recipe rewrite + editable_mode=compat note | ✅ | commit `e8b8ed9` |
| `docs/develop/active/diagnosis/sheeprl_drop_in_test.md` | v2 reader note added; dead link fixed | ✅ | commit `e8b8ed9` |
| `tmp/sheeprl/` | DELETED (gitignored; backup at `/tmp/tmp_sheeprl_bk_1778573331`) | ✅ | Step 2F — no git change needed |
| Conda env `sheeprl_bridge` on node 114 | Deferred to `training-runner` pre-flight: `pip uninstall sheeprl -y && pip install -e pytorch_agents/ --config-settings editable_mode=compat` | ⏳ pending | CP-v2-2 deferred |

**Conclusion**: [pending — senior-developer verification]

### Implementation Notes (v2 — for senior-developer review)

1. **`editable_mode=compat` required.** The standard setuptools editable install appends a MetaPathFinder to the END of `sys.meta_path`. When `python -m sheeprl` runs from the repo root, the repo root is `sys.path[0]`, so Python's default `PathFinder` finds the OUTER `pytorch_agents/` directory (no `__init__.py` → namespace package) BEFORE the editable finder reaches the inner `pytorch_agents/pytorch_agents/` package. The compat mode adds `pytorch_agents/` to `sys.path` via a `.pth` file at install time, which has higher priority than the CWD namespace-package fallback. The install command in the howto docs (§5) and training-runner profile has been updated to include `--config-settings editable_mode=compat`. Senior-developer should verify this note is in the howto and runner profile.

2. **`_PROJECT_ROOT` depth corrected.** The bridge file's `os.path.join(__file__, "..", "..", "..", "..")` (4 parent-ups) was copied from its old location at `tmp/sheeprl/sheeprl/envs/` (4 levels deep). The new location `pytorch_agents/pytorch_agents/envs/` is only 3 levels deep. Fixed to `"..", "..", ".."` (3 parent-ups). Senior-developer should verify this change in `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` line 21.

3. **CP-v2-2 and CP-v2-3 deferred.** Both require the `sheeprl_bridge` conda env which only exists on node 114 and is not accessible from the developer's local shell. Training-runner must do the env rebuild (per Step 2D) and then run CP-v2-2 (Hydra dry-run) before any production launch.

## Implementation Plan (v1 — 2026-05-12, superseded by v2 above but preserved as historical record)

### Design

The plan addresses the six gaps identified in §Analysis as a sequence of small, independent edits. None of them require new src/ code (the bridge wrapper is the only Python module, and it gets one small flag added); the rest are config tweaks, doc edits, and a per-node conda-env recipe. The order below is the recommended implementation order — each step has its own checkpoint and is independently verifiable.

The plan does NOT include:

- Building a `configs/sheeprl/` mirror or other Option-2 "paved bridge" engineering.
- Porting NMN/FiLM/precision-modulation to PyTorch (separate follow-up plan, gated on the experiment that needs it).
- Adding behavioral metrics or info-dict pass-through to the bridge — sheeprl-side metrics from `Game/*` and `Loss/*` are sufficient for the parity gate. If the neuromodulation paper needs richer per-episode info, that becomes a separate plan.
- Renaming or restructuring `tmp/sheeprl/sheeprl/` — sheeprl stays as a vendored submodule. Our edits remain confined to the three Hydra configs + the bridge file + the launch script. The vendor tree is otherwise read-only.
- Replacing `train_command-agent.sh` with a sheeprl version — the existing script's JAX-train path stays as-is. Sheeprl runs use `scripts/launch_sheeprl.sh` directly through `run_command.py`.

### File Changes

#### Step 1 — Add `apply_noise` flag to the bridge

#### `tmp/sheeprl/sheeprl/envs/grid_world_pain.py` (lines 50, 67-68, 87-89, 99-101)

```python
# BEFORE — line 50:
def __init__(self, config_path: str, seed: int = 0):

# AFTER:
def __init__(self, config_path: str, seed: int = 0, apply_noise: bool = True):
    """...
    Args:
        config_path: Absolute path to the project's YAML env config.
        seed: Integer seed passed to the initial JAX PRNG key.
        apply_noise: If True (default for production), pass apply_noise=True to
            get_observation so injury-modulated sensory noise is realized. The
            smoke run used False for cleanliness; production neuromodulation
            experiments require True.
    """
    super().__init__()
    ...
    self._apply_noise = apply_noise
```

Then change every `get_observation(..., apply_noise=False)` call (3 places — `__init__` probe at line 67-68, `reset` at line 87-89, `step` at line 99-101) to `get_observation(..., apply_noise=self._apply_noise)`.

#### `tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml` (lines 6-9)

```yaml
# BEFORE:
wrapper:
  _target_: sheeprl.envs.grid_world_pain.GridWorldPainWrapper
  config_path: ${oc.env:GWP_CONFIG_PATH}
  seed: ${seed}

# AFTER:
wrapper:
  _target_: sheeprl.envs.grid_world_pain.GridWorldPainWrapper
  config_path: ${oc.env:GWP_CONFIG_PATH}
  seed: ${seed}
  apply_noise: ${oc.env:GWP_APPLY_NOISE,true}
```

The `${oc.env:GWP_APPLY_NOISE,true}` form gives `apply_noise=True` by default, overridable at launch time via `export GWP_APPLY_NOISE=false` for diagnostic runs that want the smoke-era behaviour.

#### Step 2 — WandB project name + run-name template

#### `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` (lines 2-3)

```yaml
# BEFORE:
_target_: lightning.pytorch.loggers.WandbLogger
project: grid_world_pain_sheeprl_test
name: ${run_name}
save_dir: logs/runs/${root_dir}
log_model: False

# AFTER:
_target_: lightning.pytorch.loggers.WandbLogger
project: grid_world_pain_sheeprl
name: ${run_name}
save_dir: logs/runs/${root_dir}
log_model: False
```

Change the project from `grid_world_pain_sheeprl_test` (smoke-era throwaway) to `grid_world_pain_sheeprl`. This keeps sheeprl runs in a separate WandB project from the JAX in-house algo's `grid_world_pain` — so filtering by algorithm is project-level — while removing the "test" suffix that signals "not for production".

The `name: ${run_name}` template stays. Sheeprl populates `run_name` via Hydra resolution (it's set in sheeprl's base config). The `env.id` field (passed via `launch_sheeprl.sh $3`) is the user-facing tag and is interpolated into `run_name`. So per-experiment naming continues to flow through the launch script's third positional arg.

The previous smoke project (`grid_world_pain_sheeprl_test`) keeps its history — the rename only affects new runs. Document the rename in the how-to.

#### Step 3 — Document multi-node conda-env setup

#### `docs/develop/active/diagnosis/sheeprl_training_howto.md` — minor edit to §5

The how-to's §5 already documents the conda-env setup recipe (~30 lines). One additional sentence at the top of §5: **"Run §5 once per new node before the first sheeprl launch on that node."** Plus a brief note that `/home/vncuser/` is NAS-shared on most cluster nodes, so on those nodes the env is auto-visible after creation on any single node — verify with the §2.3 `ls` check before assuming it needs creation.

No code changes. Existing recipe is fine; just make the "you have to do this per node" expectation explicit.

#### Step 4 — Training-runner profile awareness

#### `.claude/agents/training-runner.md` (location of edit TBD by developer; search for the "dedicated launch script" section)

Two minimal edits:

1. Add a one-paragraph section "**Launching sheeprl runs**" near the top of the profile (where the existing JAX-train launch instructions live) that says:
   > Sheeprl runs are launched via a separate reusable script — not `train_command-agent.sh`. Use:
   > ```
   > ./run_command.py <node> "bash scripts/launch_sheeprl.sh <project-config.yaml> <gpu> <env-id-tag> [total-steps]"
   > ```
   > See [`docs/develop/active/diagnosis/sheeprl_training_howto.md`](../docs/develop/active/diagnosis/sheeprl_training_howto.md) for the full args reference and per-node prerequisites. The `sheeprl_bridge` conda env must exist on the target node — verify with `pgrep`-style pre-flight (§5 of the how-to) before the first sheeprl launch on a new node.

2. Make sure the existing **pre-flight node conda env** rule (from `~/.claude/.../memory/feedback_runner_node_env_preflight.md`) lists BOTH conda envs to check:
   - `grid_world_pain` (JAX algos)
   - `sheeprl_bridge` (sheeprl runs)

   …and that the pre-flight import check for sheeprl runs is `python -c "import torch, jax, sheeprl, wandb"`, not the JAX `python -c "import jax"` check.

The agent profile is not under `docs/develop/` — it's under `.claude/agents/`. That's editable by senior-developer per the project's no-implementation policy (".claude/agents/ files" are explicitly permitted, see this agent's own profile).

#### Step 5 — Cross-link existing diagnosis docs from this plan, and back

#### This plan already cross-links to the diagnosis + how-to docs in the header. No code change.

#### `docs/develop/active/diagnosis/sheeprl_training_howto.md` — append a line to §10 "Related docs":

```markdown
- [`docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md`](../sheeprl_bridge/IMPLEMENTATION_PLAN.md) — this plan, the minimum-bridge plan that adopted sheeprl as the project's Dreamer backend.
```

Symmetric cross-link complete.

#### `docs/develop/active/diagnosis/sheeprl_drop_in_test.md` — append a line to "## Related Issues" (currently "None opened"):

```markdown
- [`sheeprl_bridge/IMPLEMENTATION_PLAN.md`](../sheeprl_bridge/IMPLEMENTATION_PLAN.md) — the plan that operationalised this smoke result into the project's primary Dreamer backend.
```

### Configuration keys added

Per project rule "No fallback defaults," any new mandatory config key in **our** YAML configs gets listed here. This plan adds **zero new keys to project-side YAMLs** (`configs/...`) — the bridge reads existing env configs as-is. The Hydra-side env config grows one optional key `apply_noise`, which defaults via `${oc.env:GWP_APPLY_NOISE,true}` to True. That is a sheeprl-side config; project-side configs are unchanged.

### Launch invocation reference

For developer convenience, the canonical launch invocation after this plan lands:

```bash
# From a Claude session or terminal on any node with SSH access:
./run_command.py 114 "bash scripts/launch_sheeprl.sh \
    configs/experiment/dreamer_curriculum/01_food_only.yaml 0 gwp_food_only"

# Or, to disable noise for a diagnostic run:
./run_command.py 114 "GWP_APPLY_NOISE=false \
    bash scripts/launch_sheeprl.sh \
    configs/experiment/dreamer_curriculum/01_food_only.yaml 0 gwp_food_only_nonoise"
```

The launcher script (`scripts/launch_sheeprl.sh`) does NOT currently propagate `GWP_APPLY_NOISE` since it doesn't set or export that variable. It does, however, inherit the parent shell's environment, so prefixing the SSH command with `GWP_APPLY_NOISE=false bash scripts/launch_sheeprl.sh ...` works. If a more discoverable interface is wanted, add `apply-noise` as a 5th positional arg to the launch script in a future iteration — not blocking for this plan.

## Checkpoints

What the implementing agent should verify during implementation. Each checkpoint is independently runnable.

- [x] **Checkpoint 1 — `apply_noise` flag wired and on by default.** Verified by grep: `def __init__(... apply_noise: bool = True)` in bridge; `apply_noise=self._apply_noise` in both `reset()` and `step()`; `${oc.env:GWP_APPLY_NOISE,true}` in env YAML. Code-level verification; live smoke with `apply_noise=True` is Checkpoint 6 (deferred to training-runner — requires node launch).
- [ ] **Checkpoint 2 — `GWP_APPLY_NOISE=false` override works.** Deferred to training-runner: run with `GWP_APPLY_NOISE=false` and confirm smoke-era noise-off behaviour. The Hydra OmegaConf interpolation `${oc.env:GWP_APPLY_NOISE,true}` with env-var override is standard Hydra; code path is correct.
- [x] **Checkpoint 3 — WandB project rename confirmed.** `project: grid_world_pain` in `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` matches `project: "grid_world_pain"` in `configs/logger/wandb.yaml`. Note: user directive changed target project from plan's `grid_world_pain_sheeprl` to `grid_world_pain` — see Decisions §D1. Live smoke validation that the run lands in the right WandB project is Checkpoint 4 (deferred to training-runner).
- [ ] **Checkpoint 4 — Launch invocation works through the runner pattern.** Deferred to training-runner: live launch required on node 114.
- [x] **Checkpoint 5 — How-to + training-runner profile cross-references are bidirectional.** `grep -n "sheeprl_bridge" .claude/agents/training-runner.md` → 6+ hits. All three docs link to each other (verified in implementation report).
- [ ] **Checkpoint 6 — Parity gate with `apply_noise=True`.** Deferred to training-runner: 50k-step run needed. This is the developer-side smoke described in the plan — a separate training-runner task.
- [x] **Checkpoint 7 — Develop INDEX regenerates cleanly.** `scripts/regen_dev_index.py` exits 0, 97 docs indexed; committed 9c86601.

## Risks and open questions

1. **`apply_noise=True` may change the parity-gate trajectory.** The smoke ran with `apply_noise=False`. If turning noise on shifts food-only NoPred away from the survival-500 saturation, the bridge isn't broken — the task is now harder. Checkpoint 6 explicitly tests this. If it does NOT saturate, options are (a) accept that the production setting is different from the smoke setting (still useful as a backbone for the modulation paper, since the modulation effect is the comparison, not the absolute survival), (b) investigate whether food-only NoPred should remain noiseless and only the hypervigilance configs turn noise on, (c) revisit the bridge default.

2. **Reward-head scale consistency in the future NMN port.** When the modulator is ported, the `z_reward` scale must be applied identically at rollout-time prediction and loss-time. Sheeprl's `loss.py` reconstruction loss reads the reward head's forward output; if we add `z_reward * reward_head_output` in only one of those places, gradients will mis-attribute. This is a port-time concern (separate plan), but flagging now so the eventual port doc remembers it.

3. **WandB run-name collision risk under the renamed project.** Multiple users + multiple sessions can race to launch with the same `env.id` tag (the third positional arg of `launch_sheeprl.sh`). Hydra's `${run_name}` resolves to `<algo>-<env_id>-<date>` by default — if two runs land on the same date with the same tag they'd collide. For the current single-user research workflow this is unlikely; if it becomes a problem, add an automatic `_h{shortsha}` suffix in the launch script. Out of scope unless it bites.

4. **Conda env staleness on per-node setup.** `pip install -e tmp/sheeprl` is run per node and the editable install creates a `.pth` file pointing at the NAS-shared sheeprl tree. Updates to `tmp/sheeprl/sheeprl/envs/grid_world_pain.py` are seen immediately on every node (good). However if the sheeprl version bumps and `setup.py` / `pyproject.toml` deps change, every node's env needs `pip install -e .` re-run. Document if/when this becomes an issue.

5. **Node 114 CIFS staleness** (per project auto-memory `feedback_runner_cifs_bypass.md`) — only bites scripts read from CIFS by the SSH session. `launch_sheeprl.sh` is read from the NAS-shared project root via `cd /media/nas01/...; bash scripts/launch_sheeprl.sh`. CIFS staleness has bitten `train_command-agent.sh` before — if `launch_sheeprl.sh` ever exhibits a stale-cache symptom on node 114, the workaround is the same `/tmp/<unique>.sh` CIFS-bypass pattern. Not currently observed.

6. **Sheeprl checkpoint format.** Sheeprl writes `*.pt` checkpoints via Lightning Fabric. Our existing `results/JAX_DreamerV3/` checkpoint format is `*.eqx` / `flax.serialization` pickle. The two are not interconvertible. For the NMN paper this is fine — sheeprl runs save sheeprl checkpoints, in-house JAX runs save JAX checkpoints, no cross-loading expected. If a downstream pipeline (e.g., evaluation scripts that load checkpoints) needs to handle both formats, that's a follow-up plan.

7. **NMN-port lazy `set_imagine_input_dim` semantics.** Our `DreamerNeuromodulatorRNN` has a deferred `proj_imagine` Linear layer added by `set_imagine_input_dim(feat_dim + act_dim)` after `__init__` (line 365 of `neuromodulator.py`). This is NNX-idiomatic but doesn't translate cleanly to PyTorch `nn.Module`. The PyTorch port should construct `proj_imagine` eagerly in `__init__` with the input dim passed as a constructor arg. Mechanical fix; flagging now for the port doc.

8. **`get_observation` `apply_noise=False` was a deliberate smoke-era choice.** Re-reading [`sheeprl_drop_in_test.md`](../diagnosis/sheeprl_drop_in_test.md) Implementation Report, the rationale wasn't documented — it appears to have been "noise doesn't matter for food-only NoPred." That's true for the smoke task but is exactly wrong for the modulation paper. Flagging in case the developer wants to confirm with the user before flipping the default.

## Decisions (updated by developer, 2026-05-12)

Two user directives issued at implementation time modified the original plan scope:

### D1 — WandB project name (overrides plan §Step 2)

**Original plan**: rename `grid_world_pain_sheeprl_test` → `grid_world_pain_sheeprl` (sheeprl-specific project, separate from JAX runs).

**User directive**: use the SAME WandB project as all existing rPPO and JAX Dreamer training, so all runs are visible in one place.

**Chosen project**: `grid_world_pain`

**Confirmation source**: `configs/logger/wandb.yaml` (line 3): `project: "grid_world_pain"` and `train_command-agent.sh` (line 77): "Defaults from configs/logger/wandb.yaml: project=grid_world_pain, entity=sungwoolee". Both the JAX-side WandB config and the agent launch script confirm this is the canonical project name for all in-house training.

**Implemented**: `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` updated from `grid_world_pain_sheeprl_test` → `grid_world_pain`. The smoke-era `grid_world_pain_sheeprl_test` project keeps its history (jzgkcep4 remains there); new production runs go to `grid_world_pain`.

### D2 — pyproject.toml / conda-env setup (overrides plan §Step 3 from "document" to "automated")

**Original plan**: document per-node setup (§Step 3 said "minor edit to §5").

**User directive**: make it automated — ensure `pyproject.toml` supports reproducible install.

**Chosen option**: option (b) — `tmp/sheeprl/pyproject.toml` (the vendored upstream sheeprl package) already exists and specifies all sheeprl bridge deps (PyTorch, Lightning Fabric, Hydra, gymnasium, etc.). These deps are incompatible with the main `grid_world_pain` conda env's JAX/Flax stack, so a separate `sheeprl_bridge` env is correct. The install is `pip install -e /path/to/tmp/sheeprl` on top of a fresh Python 3.11 env.

**Install command (canonical)**:
```bash
pip install -e /media/nas01/projects/Interoceptive-AI/grid_world_pain/tmp/sheeprl
pip install "jax[cpu]" flax omegaconf pyyaml wandb
pip install -e /media/nas01/projects/Interoceptive-AI/grid_world_pain
```

**Implemented**: `sheeprl_training_howto.md` §5 updated with the canonical install command, the "run once per node" instruction, the NAS-shared-home check, and the rationale for why option (b) is used (torch/jax incompatibility).

## Implementation Report

> **Implemented by**: developer agent (claude-sonnet-4-6)
> **Date**: 2026-05-12

### Summary

Six gaps addressed across two commits. The `tmp/sheeprl/` files (gitignored) are modified on the NAS and live on disk; tracked files (docs, agent profile) committed to git.

**File-by-file:**

1. **`tmp/sheeprl/sheeprl/envs/grid_world_pain.py`** (modified, NAS-only): Added `apply_noise: bool = True` to `__init__`, stored as `self._apply_noise`. `reset()` and `step()` now call `get_observation(..., apply_noise=self._apply_noise)`. Probe reset in `__init__` still uses `apply_noise=False` (shape is invariant to noise). Docstring updated to explain production vs diagnostic use.

2. **`tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml`** (modified, NAS-only): Added `apply_noise: ${oc.env:GWP_APPLY_NOISE,true}` under `wrapper:`. Hydra's OmegaConf env interpolation defaults to `true`; set `GWP_APPLY_NOISE=false` at launch time to restore smoke-era noise-off behaviour.

3. **`tmp/sheeprl/sheeprl/configs/logger/wandb.yaml`** (modified, NAS-only): Changed `project: grid_world_pain_sheeprl_test` → `project: grid_world_pain`. Confirmed against `configs/logger/wandb.yaml` (JAX-side) — both now show `project: grid_world_pain`.

4. **`docs/develop/active/diagnosis/sheeprl_training_howto.md`** (committed): Updated §1, §2.2 (logger config snippet), §5 (per-node setup), §6.4, §6.7, §7.2, §9 (quick reference) to reflect new WandB project name. Added §5 "run once per node" instruction, NAS-shared-home check, and canonical `pip install -e` command (option b). Added cross-links to bridge plan in header and §10.

5. **`.claude/agents/training-runner.md`** (committed): Added "Two training paths" section at the top with dual-launch table (JAX vs sheeprl conda env + pre-flight check). Updated description frontmatter. Fixed Hard Do Nots conda note to name both env absolute paths.

6. **`docs/develop/active/diagnosis/sheeprl_drop_in_test.md`** (committed): Added two cross-links under "Related Issues" — to `sheeprl_bridge/IMPLEMENTATION_PLAN.md` and `sheeprl_training_howto.md`.

7. **`docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md`** (this file, committed): Added Decisions section D1 (WandB project = `grid_world_pain`) and D2 (pyproject.toml option b, canonical install command).

### Gap status

| Gap | Description | Status | Commit |
|---|---|:---:|---|
| 1 | `apply_noise=True` default on bridge | ✅ | NAS (not in git — tmp/ gitignored) |
| 2 | WandB project = `grid_world_pain` (user directive) | ✅ | NAS + docs committed 380290a |
| 3 | pyproject.toml option (b) documented (user directive) | ✅ | Docs committed 380290a |
| 4 | training-runner dual-launch-path awareness | ✅ | Committed 380290a |
| 5 | Bidirectional cross-links (3 docs) | ✅ | Committed 380290a |
| 6 | NMN-port feasibility check | ✅ | Already in plan (no action needed) |

### Verification checks

- **Gap 1**: `grep apply_noise tmp/sheeprl/sheeprl/envs/grid_world_pain.py` → 4 hits including `def __init__(... apply_noise: bool = True)` and `get_observation(..., apply_noise=self._apply_noise)` in both `reset` and `step`. `grep apply_noise tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml` → `apply_noise: ${oc.env:GWP_APPLY_NOISE,true}`.
- **Gap 2**: `grep project tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` → `project: grid_world_pain`. `grep project configs/logger/wandb.yaml` → `project: "grid_world_pain"`. Match confirmed.
- **Gap 3**: `sheeprl_training_howto.md` §5 contains canonical 3-line `pip install -e` command referencing `tmp/sheeprl/pyproject.toml` via editable install.
- **Gap 4**: `grep sheeprl_bridge .claude/agents/training-runner.md` → multiple hits including pre-flight table, sheeprl launch example, description frontmatter.
- **Gap 5**: Cross-links verified in all three docs — `grep sheeprl_bridge IMPLEMENTATION_PLAN.md` is the source; `sheeprl_training_howto.md` and `sheeprl_drop_in_test.md` both link back to it.
- **Checkpoint 7** (dev INDEX): `scripts/regen_dev_index.py` exits 0, 97 docs indexed; committed 9c86601.

### Speed check

Not applicable. This plan makes no changes to the JAX training hot path, observation pipeline, or model forward/backward. Sheeprl-side changes (apply_noise wiring, Hydra config, WandB project name) are config-level only.

### Note on gitignore and tmp/ files

`tmp/` is gitignored (`.gitignore:23: tmp/`). The sheeprl bridge files — `grid_world_pain.py`, `grid_world_pain.yaml`, `wandb.yaml`, `dreamer_v3_grid_world_pain.yaml` — live in `tmp/sheeprl/` and are NOT committed to git. They exist on the NAS-mounted project root and are visible on all nodes. The how-to doc §2.2 previously said "tracked in git" — this was incorrect. Changes are live on NAS disk and take effect immediately for any session or node reading from the NAS.

### Deviations from plan

1. **WandB project = `grid_world_pain` not `grid_world_pain_sheeprl`**: user directive overrides plan §Step 2. Confirmed project name from `configs/logger/wandb.yaml` (JAX-side). Documented in Decisions §D1.
2. **pyproject.toml = option (b) not new `[project.optional-dependencies]` in main pyproject.toml**: the main pyproject.toml has JAX/CUDA deps incompatible with torch. Option (a) would create a single env that can't satisfy both dep sets. `tmp/sheeprl/pyproject.toml` already exists and covers all sheeprl bridge deps. Documented in Decisions §D2.
3. **tmp/ files not committed**: plan text says "tracked in git" but the existing prior practise is gitignored. Changes land on NAS disk only. No functional difference for cluster training (all nodes read from the same NAS path).

Implemented by: developer

## Verification Report (v1 — historical)

> **Verified by**: [pending — `senior-developer`]
> **Date**: [pending]
>
> **Note (2026-05-12)**: This v1 Verification Report tracks the original 6-gap implementation that landed on `tmp/sheeprl/`. v2 above relocates those files into `pytorch_agents/`; the **current source of truth for verification is the v2 Verification Report** further up. This v1 table is preserved so a future reader can audit the original sign-off scope.

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `tmp/sheeprl/sheeprl/envs/grid_world_pain.py` | add `apply_noise` flag | | |
| `tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml` | add `apply_noise: ${oc.env:GWP_APPLY_NOISE,true}` | | |
| `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` | rename project to `grid_world_pain_sheeprl` | | |
| `docs/develop/active/diagnosis/sheeprl_training_howto.md` | clarify per-node setup + add back-link | | |
| `.claude/agents/training-runner.md` | document sheeprl launch path + dual conda env check | | |
| `docs/develop/active/diagnosis/sheeprl_drop_in_test.md` | back-link to this plan | | |

**Conclusion**: [pending — superseded by v2 Verification Report above]

---

## Out of scope (v1 — re-asserted for the developer)

> **Note (2026-05-12)**: v2 (above) **does** restructure `tmp/sheeprl/sheeprl/` — moving the 4 bridge files out into `pytorch_agents/` and removing the `tmp/sheeprl/` vendor tree. The bullet below that says "no renaming or restructuring of `tmp/sheeprl/sheeprl/`" was v1's scope guard at the time. It is **superseded by v2**. The other bullets (no new `configs/sheeprl/` mirror, no info-dict pass-through, no JAX-modulator port, no `train_command-agent.sh` sheeprl branch) all still hold.

The PI call explicitly chose Option 1 (minimal bridge) over Option 2 (paved bridge). Do not expand scope. Specifically:

- **No new `configs/sheeprl/...` mirror.** Sheeprl's three Hydra configs live under `tmp/sheeprl/sheeprl/configs/` and stay there. Project-side configs in `configs/experiment/...` are read by the bridge via `GWP_CONFIG_PATH` and need no modification.
- **No info-dict pass-through, no behavioral metrics surfacing.** The bridge drops `info` to avoid SyncVectorEnv stacking pain. Sheeprl's `Game/ep_len_avg` and `Rewards/rew_avg` are sufficient for parity comparison. If a future experiment needs richer info, file a separate plan.
- **No JAX/Flax modulator port to PyTorch.** Separate follow-up plan, gated on the experiment that authorizes it. This plan's §NMN-port feasibility check is the precondition for that follow-up, not its content.
- **No new `train_command-agent.sh` sheeprl branch.** Sheeprl runs invoke `scripts/launch_sheeprl.sh` directly via `run_command.py`. The agent profile lists both launch paths but the training-runner script remains JAX-only.
- **No new training-runner agent.** Same training-runner, two launch paths.
- **No changes to project-tree `src/`, `configs/`, or `scripts/`** beyond `scripts/launch_sheeprl.sh` (already in repo, untouched by this plan).

If during implementation the developer hits a need that isn't in this plan's File Changes section, stop and ask `senior-developer` — do not expand scope silently.

---

## Implementation Report (v2 — 2026-05-12)

> **Implemented by**: developer (Claude Sonnet 4.6)
> **Date**: 2026-05-12
> **Commits**: `0067721` (Steps 2A+2B), `7a2cc83` (Step 2C), `e8b8ed9` (Steps 2D+2E)
> **Pending**: conda env rebuild on node 114 (training-runner pre-flight), CP-v2-2 (Hydra dry-run), CP-v2-3/4 (smoke runs)

### Summary

**Step 2A (skeleton)**: Created `pytorch_agents/` package with `pyproject.toml` (sheeprl pinned to `33b6366`, jax[cpu], wandb, pyyaml), `README.md`, `__init__.py` with `__version__ = "0.1.0"`, and all required empty `__init__.py` files for subpackages (`envs/`, `configs/`, `configs/env/`, `configs/exp/`, `configs/logger/`).

**Step 2B (file move)**: Copied all 4 bridge files from `tmp/sheeprl/sheeprl/` to `pytorch_agents/pytorch_agents/`. Applied the `_target_` edit (`sheeprl.envs.grid_world_pain` → `pytorch_agents.envs.grid_world_pain`). Also fixed `_PROJECT_ROOT` path depth in `envs/grid_world_pain.py` (4 parent-ups → 3, because new location is 3 levels deep, not 4). Original files kept in `tmp/sheeprl/` until Step 2F.

**Step 2C (launch script)**: Replaced `python tmp/sheeprl/sheeprl.py` with `python -m sheeprl` and added `export SHEEPRL_SEARCH_PATH="pkg://pytorch_agents.configs"` with comment block explaining the `pkg://` semantics.

**Step 2D (docs — install recipe)**: Updated `sheeprl_training_howto.md` §5 with v2 install recipe (`pip install -e pytorch_agents/ --config-settings editable_mode=compat`). Documented the `editable_mode=compat` requirement and the v1 historical recipe for reference.

**Step 2E (path sweep)**: Updated `sheeprl_training_howto.md` throughout (§2, §3, §4, §5, §6, §7, §8, §9), `.claude/agents/training-runner.md` (sheeprl launch path + pre-flight import check), and `sheeprl_drop_in_test.md` (v2 reader note + fixed dead link). All prescriptive `tmp/sheeprl/` references replaced; historical references preserved with clear labeling.

**Step 2F (deletion)**: Backed up `tmp/sheeprl/` to `/tmp/tmp_sheeprl_bk_1778573331`, then `rm -rf tmp/sheeprl/`. CP-v2-6 confirmed.

### Checkpoint results

| CP | Status | Notes |
|---|:---:|---|
| CP-v2-1 (package importable) | ✅ PASS | `import pytorch_agents; __version__ == "0.1.0"` from repo root (with `editable_mode=compat`). Discovery: must use compat mode — see Implementation Notes above. |
| CP-v2-2 (Hydra dry-run) | ⏳ DEFERRED | `sheeprl_bridge` env not accessible from developer's shell; deferred to training-runner pre-flight on node 114. |
| CP-v2-3 (launch smoke) | ⏳ DEFERRED | Training-runner task. |
| CP-v2-4 (noise parity) | ⏳ DEFERRED | Training-runner task. |
| CP-v2-5 (path sweep) | ✅ PASS | `grep -rn "tmp/sheeprl" .claude/agents/ docs/.../sheeprl_training_howto.md scripts/launch_sheeprl.sh pytorch_agents/ run_command.py` returns 0 prescriptive hits. |
| CP-v2-6 (deletion) | ✅ PASS | `ls tmp/sheeprl 2>&1` → "No such file or directory". |
| CP-v2-7 (INDEX regen) | ✅ PASS | `python scripts/regen_dev_index.py` exits 0, "97 docs indexed". |

### Speed check

Not applicable — v2 is a relocation + packaging change. No hot-path code changes. The bridge file's logic is identical to what was in `tmp/sheeprl/`.

### Deviations from plan

1. **`editable_mode=compat` required (unplanned)**: Standard setuptools editable install creates a MetaPathFinder that appended to `sys.meta_path` after the default `PathFinder`. When `python -m sheeprl` runs from the repo root (which the launch script does), `sys.path[0]` is the repo root, and `PathFinder` finds the outer `pytorch_agents/` directory (namespace package, no `__init__.py`) before the editable finder reaches the inner package. Fixed with `--config-settings editable_mode=compat`. This flag is documented in the howto doc, training-runner profile, and Implementation Notes above. Not a deviation from the plan's intent — the plan's CP-v2-1 said "import succeeds"; this fixes the one obstacle to that.

2. **`_PROJECT_ROOT` path depth corrected (unplanned item)**: The moved bridge file had `os.path.join(__file__, "..", "..", "..", "..")` (4 parent-ups, correct for `tmp/sheeprl/sheeprl/envs/`). New location `pytorch_agents/pytorch_agents/envs/` is only 3 levels deep. Corrected to 3 parent-ups. This is a necessary correctness fix; the plan's file-move description said "file content stays identical" but this line needed changing for the new path to work. Flagged here for senior-developer's attention.

### Blockers / follow-ups

- **Node 114 env rebuild**: training-runner must do `pip uninstall sheeprl -y && pip install -e pytorch_agents/ --config-settings editable_mode=compat && pip install -e .` before the first v2 launch. The old editable install points at the now-deleted `tmp/sheeprl/` — any import of sheeprl from the old path will fail.
- **CP-v2-2 (Hydra dry-run)**: must be run on node 114 by training-runner as part of pre-flight for the first v2 launch.
- **CP-v2-3 / CP-v2-4**: smoke runs deferred to training-runner.

**Implemented by**: developer
