# Diagnosis findings — recurrent_ppo_network.py + neuromodulator.py (2026-07-23)

Reviewer: JAX/Flax correctness pass (deep diagnosis). Scope: `src/models/recurrent_ppo_network.py`,
`src/models/neuromodulator.py`, plus the directly-wired `src/models/modulated_gru_cell.py` and the
call sites in `src/models/recurrent_ppo_trainer.py` / `train.py` needed to trace end-to-end.
All claims below were verified by tracing code and/or running read-only probes in the
`grid_world_pain` conda env (dummy instantiation, gradient checks, flax source inspection).
Known OPEN bugs (4 config-boundary traps, dead `memory_clip` key history) are not re-reported;
the FiLM-gain-shared-across-senses design is respected as NOT-A-BUG.

---

## Findings

### F1 — P1 — Baseline vs. modulated arms use *different task-RNN cells with different initializers* (comparison confound)

- **Where**: `src/models/recurrent_ppo_network.py:249-254` (cell selection) and
  `src/models/modulated_gru_cell.py:29-40`.
- **Claim (plain English)**: When modulation is off, the task RNN is `nnx.GRUCell`; when modulation
  is on, it is the custom `ModulatedGRUCell`. These are *not* the same architecture at
  initialization: flax's `nnx.GRUCell` initializes its three recurrent (hidden-to-hidden) matrices
  with **orthogonal** init (verified from the installed flax: `recurrent_kernel_init=orthogonal`),
  while `ModulatedGRUCell` builds them from plain `nnx.Linear` layers → **lecun_normal**. The
  update-gate polarity is also flipped relative to flax (`h_new = (1-u)*h + u*n` here vs. flax's
  `new_h = (1-z)*n + z*h`, verified from flax source), and the modulated cell carries duplicate
  biases (both `W_i*` and `W_h*` have `use_bias=True`). The polarity flip is
  statistically neutral at init (symmetric weight init) and its sign semantics match
  NEUROMODULATION_ALGORITHM.md line 37 (negative z_memory → retention), so it is not itself wrong —
  but the **orthogonal-vs-lecun recurrent init difference is a systematic confound between the two
  comparison arms**. The docstring claim "When gate_bias is None or zeros, this is functionally
  identical to nnx.GRUCell" (`modulated_gru_cell.py:8`) is false at the initialization/
  parameterization level.
- **Failure scenario**: Every modulated-vs-baseline study (the project's core claim structure)
  attributes performance/behaviour differences to neuromodulation. Part of any observed gap can
  instead come from the baseline arm enjoying orthogonal recurrent init (better long-horizon
  gradient propagation over the 128-step BPTT windows) while the modulated arm gets lecun_normal.
  This biases *against* the modulated arm in a way that mimics "modulation hurts".
- **Evidence**: `inspect.signature(nnx.GRUCell.__init__)` → `recurrent_kernel_init=orthogonal(...)`,
  `kernel_init=variance_scaling(...)`; `inspect.getsource(nnx.GRUCell.__call__)` contains
  `new_h = (1.0 - z) * n + z * h`. `ModulatedGRUCell` uses six default `nnx.Linear` layers.
- **Fix direction**: Give `ModulatedGRUCell`'s `W_hr/W_hu/W_hn` orthogonal kernel init (and drop the
  duplicate biases) so the zero-gate-bias cell is init-distribution-identical to `nnx.GRUCell`;
  or make the baseline arm use `ModulatedGRUCell` with `gate_bias=None`. Either way, re-baseline
  any comparison that hinges on small effect sizes.

### F2 — P2 — Hierarchical encoder silently drops observation dims when `sum(breakdown) < input_dim`; the modulator still sees them

- **Where**: `src/models/recurrent_ppo_network.py:110-115` (`__call__`) and `:149-154`
  (`forward_with_modulation`); missing guard also at `train.py:771-772` (computes
  `total_dim = sum(obs_breakdown.values())` but never asserts `total_dim == input_dim`).
- **Claim**: The per-sense slicing loop copies `x[..., start:start+dim]` for each breakdown entry.
  If the breakdown sums to less than the actual observation dim, the trailing dims are **silently
  never encoded** — no error, no warning (JAX slice semantics clamp). Verified empirically: a
  12-dim obs with an 8-dim breakdown runs fine, and setting the last 4 dims to 999.0 leaves the
  logits bit-identical. The opposite direction (`sum > input_dim`) does raise (broadcast error), so
  only the under-sum direction is a trap. Extra asymmetry: the neuromodulator consumes the *full*
  flat `x` (`recurrent_ppo_network.py:322`), so the modulator would see sensory channels the task
  network is blind to.
- **Failure scenario**: A future sensor edit that appends dims in `get_observation()`
  (`src/environment/sensor.py:291-348`) without updating `get_observation_breakdown()`
  (`:350-390`) trains a hierarchical agent that is silently blind to the new sensor while the
  modulator reacts to it. Today the two functions are in-sync (ordering and dims verified
  entry-by-entry), so this is latent, not live.
- **Evidence**: Probe run: "MISMATCH ACCEPTED SILENTLY … trailing dims ignored: True";
  over-sum probe raised `ValueError: Incompatible shapes for broadcasting: (4,) and requested shape (8,)`.
- **Fix direction**: One-line assert in `ObservationEncoder.__init__` (hierarchical branch):
  `sum(breakdown.values()) == input_dim`, and/or the same assert at `train.py:772`.

### F3 — P2 — FiLM "pass-through init" is bias-only: γ has std ≈0.28 at step 1 and goes negative within ~20 steps

- **Where**: `src/models/neuromodulator.py:92-98` (γ bias = 1.0, kernel = default lecun_normal),
  `:104-108`, applied at `recurrent_ppo_network.py:165-168, 183-186`.
- **Claim**: §5.2's pass-through intent is implemented only through the head *bias* (γ bias 1.0,
  β bias 0.0). The head *kernels* keep default lecun_normal init, and the modulator GRU state is
  non-zero after the first observation, so at t=0 the FiLM gains are γ = 1 + W·h with substantial
  spread. Unlike the sigmoid modes (where kernel noise is squashed to ~[0.75, 0.95] around σ(2)),
  FiLM has no squashing — the noise passes straight through, and individual γ's cross zero
  (sign-flipping the corresponding encoder features).
- **Failure scenario**: The modulated agent starts training with a randomly feature-scrambled
  encoder relative to the baseline — a slow/noisy start that contaminates early-training
  comparisons and works against the §5.2 "fair baseline comparison" goal that motivated the bias
  init in the first place.
- **Evidence**: Probe on a realistic hierarchical FiLM net (hidden 128, mod_hidden 64):
  step-1 γ_uni mean 0.953, std 0.283, range [0.207, 1.585]; β std 0.255; over 20 random steps
  min γ = −0.645.
- **Fix direction**: `kernel_init=nnx.initializers.zeros` on the four FiLM heads (standard FiLM
  pass-through trick) — γ then starts *exactly* 1.0, β exactly 0.0, and gradients still flow
  (bias + baseline params break symmetry).

### F4 — P2 — Temperature is not pass-through at init: modulated policy starts ~20% flatter than baseline

- **Where**: `src/models/neuromodulator.py:118` (head_action, default bias 0) and `:172-173`
  (`softplus(z)+0.5`).
- **Claim**: At init, temperature = softplus(0 + W·h) + 0.5 ≈ 1.19–1.35 (probed: 1.22 at step 1,
  drifting 1.16–1.35 over 20 steps), never 1.0. The modulated arm therefore starts with
  systematically softened logits (`logits / T`, `recurrent_ppo_network.py:345`) — higher entropy,
  weaker early preferences — vs. the baseline arm's implicit T=1. The `temp = 0.5 + softplus(z)`
  form is per design (§5.3b), but the doc never addresses init parity, and the +0.5 floor also
  makes `temp_clip[0]` dead (already self-documented at `neuromodulator.py:169-171`).
- **Failure scenario**: Early-training entropy/behaviour differences between arms that are pure
  init artifacts get read as modulation effects (e.g., "modulated agent explores more early on").
- **Evidence**: Probe output "step1 temperature: 1.2226 … temp range 1.163..1.353".
- **Fix direction**: `bias_init=constant(softplus_inverse(0.5)) ≈ −0.433` on `head_action`
  (optionally zero kernel too, per F3) so T(init) = 1.0.

### F5 — P2 — Flat encoding + modulation silently makes the multimodal heads dead weight (zero gradient)

- **Where**: `src/models/recurrent_ppo_network.py:135-146` (flat branch of
  `forward_with_modulation` consumes only `z_unimodal`/`z_unimodal_add`);
  `src/models/neuromodulator.py:104-108, 122, 127` (heads/baselines still constructed).
- **Claim**: In flat mode there is no Phase-2 hub, so `head_multimodal`, `head_multimodal_add`,
  `z_hidden_baseline`, `z_hidden_add_baseline` are computed every step but never consumed — zero
  gradient, dead parameters, no warning. Same failure family that got LSTM+modulation promoted to
  a constructor `ValueError` (network:220-226, tests/models/test_network_construction.py).
- **Failure scenario**: A flat-encoder modulation ablation (a natural future config — and the
  exact combination the new construction test itself builds) silently trains with a third of the
  modulator's perceptual machinery inert, skewing grouping/capacity interpretations.
  Latent today: every live modulated rPPO config sets `encoding_mode: "hierarchical"` (grep over
  `configs/`; only archived dreamer-NNX and one sheeprl-matched config use flat).
- **Evidence**: `nnx.grad` probe on a flat FiLM net: grad_norm exactly 0.0 for all six
  multimodal-head/baseline leaves; non-zero for unimodal/memory/temperature leaves.
- **Fix direction**: Either raise at construction (flat + modulation → error, mirroring the LSTM
  guard), skip constructing the Phase-2 heads in flat mode, or apply `z_multimodal` to the flat
  projection as the single stage.

### F6 — P2 — `percept_add_bias_init` read with a silent fallback default, contra project rule

- **Where**: `src/models/recurrent_ppo_network.py:270`
  (`modulation_config.get('percept_add_bias_init', 0.0)`).
- **Claim**: Every sibling modulation key at `:266-273` is a mandatory `[]` lookup (loud KeyError),
  but the β-bias init silently defaults to 0.0. A config that misspells the key trains without
  complaint at the default — a fifth member of the known "config-boundary silent failures" family
  (this exact key is not among the four known traps). Note the mirror-image quirk:
  `percept_bias_init` (`:269`) is *mandatory* yet **dead** in FiLM mode
  (`neuromodulator.py:92-95` overrides to 1.0) — that half is already in KNOWN_BUGS;
  the `.get` default half is new.
- **Failure scenario**: A PreActivation experiment sweeping β init writes the key at the wrong
  nesting level; all sweep points silently run identical β=0.0 models.
- **Fix direction**: Make it a mandatory lookup when `type ∈ {PreActivation, FiLM}` (and, for the
  known dead-key half, stop requiring `percept_bias_init` under FiLM).

### F7 — P2 (minor) — `NeuromodulatorRNN` accepts `obs_breakdown` but never uses it

- **Where**: `src/models/neuromodulator.py:63` (parameter), stored nowhere, read nowhere;
  passed from `recurrent_ppo_network.py:278`.
- **Claim**: Dead constructor argument. Harmless today, but it advertises per-sense awareness the
  modulator does not have — a reader could assume the modulator's unimodal head is per-sense-
  grouped when it is actually `ceil(hidden/G)` over the *hidden* axis, broadcast identically to
  all senses (the intended NOT-A-BUG design).
- **Fix direction**: Drop the parameter (and call-site kwarg), or comment it as reserved.

---

## Fixed-bug regression check (unit-relevant FIXED rows)

| Fixed item | Status now |
|---|---|
| Dead `memory_clip` key never enforced (NeuromodulatorRNN) | **Fixed, no regression** — `neuromodulator.py:167` clips `z_mem` after the baseline add; key threaded as mandatory from `recurrent_ppo_network.py:273`. |
| LSTM + modulation silently drops Injection B (parity-review row 4) | **Fixed** — constructor raises `ValueError` (`recurrent_ppo_network.py:220-226`); `tests/models/test_network_construction.py` (3 tests) passes green in the project env. |
| `FiLMNoNorm` removal | Guarded loudly (`recurrent_ppo_network.py:214-218`). |
| PRNG reset-key aliasing (parity-review row 1, trainer) | **Fix landed** — `recurrent_ppo_trainer.py:254` now advances the carried key with an explanatory comment (matches commit `b8eb286`). |
| MC window-edge bootstrap (H4) / truncation-vs-death | Unchanged from the 07-22 parity review's CONFIRMED CORRECT verdict; network-side contract (`__call__` handles `(task_h, mod_h)` opaquely; symlog inside `__call__` at `:316` so all forwards share preprocessing) still holds. |

## Reviewed but clean

- **FiLM/PreActivation γ/β application order + broadcasting**: γ,β `(..., H)` broadcast over the
  sense axis via `[..., None, :]` (`network:164-171`) — correct, matches the intended
  shared-across-senses design; hierarchical Phase-2 and flat paths apply γ·x+β pre-ReLU with LN
  placement identical in modulated and unmodulated paths.
- **Observation layout**: `get_observation` part order/dims match `get_observation_breakdown`
  entry-by-entry (Injury→Nutrition→Satiation→InteroNoc→ExteroNoc→Olfaction→Collision→
  Proprioception→Visual→Location); encoder iteration order self-consistent (same dict). Only the
  missing sum guard (F2) is a gap.
- **Grouping semantics**: `repeat(raw, G)[..., :H]` gives contiguous neuron blocks; `ceil` handles
  non-divisible H; clip applied after baseline add (correct order).
- **Hidden-state init/reset**: `initial_state` zeros for task GRU/LSTM and modulator; matches the
  trainer's `_h_reset_on_done` zeros (fresh-init ≡ reset). LSTM carry tuple symmetric, order-safe.
- **Temperature parity across collection/loss/eval**: division happens inside `__call__` before
  return (`:345`), so both log-prob computations and eval argmax are consistent (argmax invariant
  to positive scalar T). No stop_gradient in the modulator path — intentional (BPTT trains it).
- **`get_action_and_value_nnx`**: unbatched per-env semantics match the trainer's
  `vmap(..., in_axes=(None, 0, h_axes, 0))`; `argmax`/`log_softmax[action]` correct for 1-D logits;
  `eval_mode` static under `nnx.jit`.
- **Symlog**: applied once, at the network boundary, to both task and modulator inputs; no double
  application found in trainer/eval paths.
- **GroupedLinear/GroupedMLP**: einsum `'...gi,gio->...go'` correct; zero-padding of shorter senses
  is inert through the weights. Nits: fixed `normal*0.1` init ignores fan-in (comparable to lecun
  at the fan-ins used); `b_key = rngs.params()` drawn but unused (harmless).
- **Shared trunk**: actor and critic heads both read the RNN output `x_h` — intended shared-trunk
  recurrent PPO design; no unintended leakage beyond that.
- **DreamerNeuromodulatorRNN**: currently dead code (dreamer_srl has no NMN hooks; NNX archived).
  `forward_imagine` before `set_imagine_input_dim` fails loudly (AttributeError). Not audited
  further per the Dreamer-stack status memory.
- **Cosmetics noted, not filed**: `ActorCriticRNN.__call__` docstring says 3-tuple but returns
  4-tuple; `NeuromodulatorRNN` docstring names `input_dim` for the `obs_dim` arg; unmodulated
  LSTM/GRU branch at `network:362-365` is a redundant if/else.

*Verification: /home/vncuser/miniconda3/envs/grid_world_pain/bin/python; probes run 2026-07-23. Report-only pass — no code modified.*
