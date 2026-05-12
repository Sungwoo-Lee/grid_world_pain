---
title: "Sheeprl Bridge — Minimum-Bridge Implementation Plan"
topic: dreamer
status: active
created: 2026-05-12
last_updated: 2026-05-12
phase: 2
---

# Sheeprl Bridge — Minimum-Bridge Implementation Plan

> **Status**: PLANNED
> **Opened**: 2026-05-12
> **Related**: [PI call 2026-05-12](../../../pi/calls/2026-05-12_dreamer_backend.md) · [Drop-in diagnosis (smoke run)](../diagnosis/sheeprl_drop_in_test.md) · [How-to: launch sheeprl](../diagnosis/sheeprl_training_howto.md) · [Compatibility audit (5×5 + 10×10)](../../../reviews/sheeprl_two_configs_audit.md) · [Archived: dreamer-srl JAX rebuild plan](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md)

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

## Implementation Plan

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

- [ ] **Checkpoint 1 — `apply_noise` flag wired and on by default.** Run a 1k-step local smoke against the 5×5 food-only config with `apply_noise=True` (the new default) and confirm no exceptions. Verify by adding `print(self._apply_noise)` in `__init__` temporarily, or by checking that an obs vector pulled from a noise-enabled config (e.g. `configs/experiment/hypervigilance/01-interoNocicept.yaml`) has the expected noisy modalities. Remove the print before commit.
- [ ] **Checkpoint 2 — `GWP_APPLY_NOISE=false` override works.** Re-launch the same smoke with `GWP_APPLY_NOISE=false` and confirm the env's noise modalities are not realized (compare obs trajectories or check `apply_noise` is False inside the wrapper).
- [ ] **Checkpoint 3 — WandB project rename does not break run.** Launch a 5k-step sheeprl run with the new `project: grid_world_pain_sheeprl` and confirm the run lands in the new project at `https://wandb.ai/sungwoolee/grid_world_pain_sheeprl/`. Capture the URL.
- [ ] **Checkpoint 4 — Launch invocation works through the runner pattern.** Have the `training-runner` agent (or the developer simulating it) run `./run_command.py 114 "bash scripts/launch_sheeprl.sh configs/experiment/dreamer_curriculum/01_food_only.yaml 0 gwp_food_only_test_v2"` end-to-end. Verify (a) a `logs/YYYYMMDD_HHMMSS.log` is created locally with the SSH-multiplexed output, (b) the run shows up at `pgrep -af sheeprl.py` on node 114 with exactly one PID, (c) WandB run name reflects the tag `gwp_food_only_test_v2`.
- [ ] **Checkpoint 5 — How-to + training-runner profile cross-references are bidirectional.** `grep -l sheeprl_bridge docs/develop/active/diagnosis/*.md` and `grep -l sheeprl_bridge .claude/agents/training-runner.md` both return non-empty.
- [ ] **Checkpoint 6 — Parity gate from the smoke is reproducible.** Launch a 50k-step run (not the full 200k — that's 11+ hours; 50k = ~3 hours suffices for the gate) on the 5×5 food-only NoPred config with the bridge's new `apply_noise=True` default and confirm `Game/ep_len_avg` reaches 500 (the env cap) by ~25k steps as it did in `jzgkcep4`. If it does **not** reach 500 by 50k, halt — that signals the `apply_noise=True` change has shifted the task in a way that breaks parity, and the change needs investigation before merging.
- [ ] **Checkpoint 7 — Develop INDEX regenerates cleanly.** Run `python scripts/regen_dev_index.py` after creating this plan doc and confirm exit 0 with the new doc indexed.

## Risks and open questions

1. **`apply_noise=True` may change the parity-gate trajectory.** The smoke ran with `apply_noise=False`. If turning noise on shifts food-only NoPred away from the survival-500 saturation, the bridge isn't broken — the task is now harder. Checkpoint 6 explicitly tests this. If it does NOT saturate, options are (a) accept that the production setting is different from the smoke setting (still useful as a backbone for the modulation paper, since the modulation effect is the comparison, not the absolute survival), (b) investigate whether food-only NoPred should remain noiseless and only the hypervigilance configs turn noise on, (c) revisit the bridge default.

2. **Reward-head scale consistency in the future NMN port.** When the modulator is ported, the `z_reward` scale must be applied identically at rollout-time prediction and loss-time. Sheeprl's `loss.py` reconstruction loss reads the reward head's forward output; if we add `z_reward * reward_head_output` in only one of those places, gradients will mis-attribute. This is a port-time concern (separate plan), but flagging now so the eventual port doc remembers it.

3. **WandB run-name collision risk under the renamed project.** Multiple users + multiple sessions can race to launch with the same `env.id` tag (the third positional arg of `launch_sheeprl.sh`). Hydra's `${run_name}` resolves to `<algo>-<env_id>-<date>` by default — if two runs land on the same date with the same tag they'd collide. For the current single-user research workflow this is unlikely; if it becomes a problem, add an automatic `_h{shortsha}` suffix in the launch script. Out of scope unless it bites.

4. **Conda env staleness on per-node setup.** `pip install -e tmp/sheeprl` is run per node and the editable install creates a `.pth` file pointing at the NAS-shared sheeprl tree. Updates to `tmp/sheeprl/sheeprl/envs/grid_world_pain.py` are seen immediately on every node (good). However if the sheeprl version bumps and `setup.py` / `pyproject.toml` deps change, every node's env needs `pip install -e .` re-run. Document if/when this becomes an issue.

5. **Node 114 CIFS staleness** (per project auto-memory `feedback_runner_cifs_bypass.md`) — only bites scripts read from CIFS by the SSH session. `launch_sheeprl.sh` is read from the NAS-shared project root via `cd /media/nas01/...; bash scripts/launch_sheeprl.sh`. CIFS staleness has bitten `train_command-agent.sh` before — if `launch_sheeprl.sh` ever exhibits a stale-cache symptom on node 114, the workaround is the same `/tmp/<unique>.sh` CIFS-bypass pattern. Not currently observed.

6. **Sheeprl checkpoint format.** Sheeprl writes `*.pt` checkpoints via Lightning Fabric. Our existing `results/JAX_DreamerV3/` checkpoint format is `*.eqx` / `flax.serialization` pickle. The two are not interconvertible. For the NMN paper this is fine — sheeprl runs save sheeprl checkpoints, in-house JAX runs save JAX checkpoints, no cross-loading expected. If a downstream pipeline (e.g., evaluation scripts that load checkpoints) needs to handle both formats, that's a follow-up plan.

7. **NMN-port lazy `set_imagine_input_dim` semantics.** Our `DreamerNeuromodulatorRNN` has a deferred `proj_imagine` Linear layer added by `set_imagine_input_dim(feat_dim + act_dim)` after `__init__` (line 365 of `neuromodulator.py`). This is NNX-idiomatic but doesn't translate cleanly to PyTorch `nn.Module`. The PyTorch port should construct `proj_imagine` eagerly in `__init__` with the input dim passed as a constructor arg. Mechanical fix; flagging now for the port doc.

8. **`get_observation` `apply_noise=False` was a deliberate smoke-era choice.** Re-reading [`sheeprl_drop_in_test.md`](../diagnosis/sheeprl_drop_in_test.md) Implementation Report, the rationale wasn't documented — it appears to have been "noise doesn't matter for food-only NoPred." That's true for the smoke task but is exactly wrong for the modulation paper. Flagging in case the developer wants to confirm with the user before flipping the default.

## Implementation Report

> **Implemented by**: [pending — `developer` agent]
> **Date**: [pending]

<!-- developer fills this section after each step -->

## Verification Report

> **Verified by**: [pending — `senior-developer`]
> **Date**: [pending]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `tmp/sheeprl/sheeprl/envs/grid_world_pain.py` | add `apply_noise` flag | | |
| `tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml` | add `apply_noise: ${oc.env:GWP_APPLY_NOISE,true}` | | |
| `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` | rename project to `grid_world_pain_sheeprl` | | |
| `docs/develop/active/diagnosis/sheeprl_training_howto.md` | clarify per-node setup + add back-link | | |
| `.claude/agents/training-runner.md` | document sheeprl launch path + dual conda env check | | |
| `docs/develop/active/diagnosis/sheeprl_drop_in_test.md` | back-link to this plan | | |

**Conclusion**: [pending]

---

## Out of scope (re-asserted for the developer)

The PI call explicitly chose Option 1 (minimal bridge) over Option 2 (paved bridge). Do not expand scope. Specifically:

- **No new `configs/sheeprl/...` mirror.** Sheeprl's three Hydra configs live under `tmp/sheeprl/sheeprl/configs/` and stay there. Project-side configs in `configs/experiment/...` are read by the bridge via `GWP_CONFIG_PATH` and need no modification.
- **No info-dict pass-through, no behavioral metrics surfacing.** The bridge drops `info` to avoid SyncVectorEnv stacking pain. Sheeprl's `Game/ep_len_avg` and `Rewards/rew_avg` are sufficient for parity comparison. If a future experiment needs richer info, file a separate plan.
- **No JAX/Flax modulator port to PyTorch.** Separate follow-up plan, gated on the experiment that authorizes it. This plan's §NMN-port feasibility check is the precondition for that follow-up, not its content.
- **No new `train_command-agent.sh` sheeprl branch.** Sheeprl runs invoke `scripts/launch_sheeprl.sh` directly via `run_command.py`. The agent profile lists both launch paths but the training-runner script remains JAX-only.
- **No new training-runner agent.** Same training-runner, two launch paths.
- **No changes to project-tree `src/`, `configs/`, or `scripts/`** beyond `scripts/launch_sheeprl.sh` (already in repo, untouched by this plan).

If during implementation the developer hits a need that isn't in this plan's File Changes section, stop and ask `senior-developer` — do not expand scope silently.
