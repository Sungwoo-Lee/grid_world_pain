---
title: "dreamer-srl v3 — JAX rebuild of sheeprl DreamerV3 with deviation-prevention guardrails"
topic: dreamer
status: active
created: 2026-05-13
last_updated: 2026-05-14  # CP3 — zero-init reward + critic heads implemented (cascade fix #27)
supersedes: IMPLEMENTATION_PLAN.md
phase: 2
---

<!--
NOTE on `supersedes:` — the archived v2 plan lives at
`docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md`. The frontmatter field
takes the bare filename per the develop frontmatter contract; the develop
INDEX resolver matches by filename across `active/` + `archive/`.
-->


# dreamer-srl v3 — JAX rebuild of sheeprl DreamerV3 with deviation-prevention guardrails

## Plain-language entry point

We are restarting the JAX/Flax rebuild of the community PyTorch DreamerV3 implementation
(`sheeprl`, pinned to commit `33b6366`). The previous attempt — v2, archived at
[`docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md`](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md) — was a
1056-line line-by-line spec that three reviewers signed off (code, math, RL/Bayesian-DL
professor). On 2026-05-12 the user shelved it and switched to running sheeprl directly
because of a 5× survival gap and a year of inconclusive cascade debugging. **Today
that decision is reversed for a specific reason**: a matched-config speed comparison
(documented in [`SPS_COMPARISON_JAX_VS_SHEEPRL.md`](../../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md))
measured JAX as 18–27× faster than sheeprl on training-compute throughput even with
every hyperparameter matched. That speed gap is large enough to justify the rebuild
as a research-productivity investment.

The user committed GO on the rebuild **and** simultaneously requested the maximum
safety net. The reason is concrete: during the previous cascade-debugging effort
we shipped a two-hot reward-bin encoding bug that took weeks to find — the bin grid
was implemented in real reward space when sheeprl stores it in symlog space. That
bug slipped past three reviewers reading the same plan, because reading a spec
does not actually run the function. The user's verbatim concern is *"nothing has
to be changed in the meaning of functions"* — i.e., bit-identical algorithm
semantics with sheeprl, no unjustified deviations.

This v3 plan is **the v2 algorithmic content (unchanged) wrapped in five guardrail
layers** that turn "we asked you to match exactly" into mechanical enforcement:
per-function bit-identity unit tests, mandatory source-line citations in docstrings,
a three-reviewer gate at every checkpoint, a vendored pinned copy of sheeprl plus
a diff tool, and a deviation log that the principal investigator signs off on
before any deviation merges. The deliverable is the same: a `src/algorithms/dreamer_srl/`
module that passes a parity gate of mean survival ≥ ~500 steps on the food-only
NoPred task across 3 seeds. The cost is roughly 2× the naive v2 effort estimate
because of the per-checkpoint review gates and bit-identity test fixtures — this
is the trade the user explicitly chose.

### Plain-language note on the 2026-05-13 CP3b revision

The original v3 sketch listed the replay buffer plus the training-cadence wiring as
*"no Lever-A gate; integration smoke only"* — meaning we would only check that
the buffer rounds data through without crashing. The user pushed back, citing
empirical evidence: in the matched-config speed measurement (see
[SPS_COMPARISON_JAX_VS_SHEEPRL.md §3.6](../../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md#36-where-the-jax-advantage-is-coming-from)),
we discovered that JAX and sheeprl have a 16× difference in how often gradient
updates fire per environment step, even though both YAMLs say `replay_ratio: 1.0`.
The difference is in the *meaning* of the knob (sheeprl: per env step; JAX:
per `collect_interval`-normalised macro-step). We found this **at the parity
gate**, which is the silent-divergence failure mode this whole v3 plan exists to
prevent. Three more details in the same layer — how the buffer stores N parallel
envs' transitions, whether sample windows are allowed to cross episode boundaries,
and the exact `learning_starts` off-by-one (`prefill_steps = learning_starts -
int(learning_starts > 0)`) — would be just as silent if they drifted.

The revision promotes the buffer + cadence layer to a real checkpoint, **CP3b**.
"State-evolution bit-identity" (drive both implementations with the same
deterministic input sequence; assert resulting state and sample-index sequences
match) is achievable here even though pure-function bit-identity is not (because
the PyTorch buffer's RNG differs from JAX's, declared deviation D-002 class). The
new CP slots into the build queue right after CP1 and before CP5 — historical-scar
gates fire early so the discipline is visible to all downstream CPs. See the
sibling spec doc [CP3B_SPEC.md](CP3B_SPEC.md) for the full scope (8 high-risk
items with sheeprl-source citations, 6 Lever-A tests, per-trap reviewer chain,
file-change spec for `developer`), §"Implementation order (revised)" for the new
build sequence, and [DEVIATION_LOG.md D-004](DEVIATION_LOG.md#deviation-table)
for the pre-declared `memmap` omission entry.

## How v3 relates to v2

**v2 is the algorithmic backbone.** Three reviewers read it line-by-line in 2026-05-12
and signed off PASS on (a) zero residual algorithm-level deviations from sheeprl
(11/11 v1 items resolved), (b) zero residual math deviations (7/7 v1 items resolved,
including the two-hot symlog-space bin-grid fix), (c) zero residual JAX/Flax-NNX
correctness errors (11/11 v1 items resolved). v3 does **not** re-derive any of that
content; it references v2 by section.

**v3 adds five guardrail layers** (A–E below) and rewrites the checkpoint table
to fold per-checkpoint reviewer gates and bit-identity tests into the flow. Nothing
algorithmic is added or removed.

What lives where:

| Content | Location |
|---|---|
| Module layout (5 files under `src/algorithms/dreamer_srl/`) | [v2 §"Design"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#design) |
| File-by-file change table (per-file rows for `utils.py`, `agent.py`, `loss.py`, `buffers.py`, `train.py`) | [v2 §"File Changes — summary table"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#file-changes--summary-table) |
| 10 silent training-loop semantics (S1–S10) | [v2 §"Training-loop semantics"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#training-loop-semantics-silent-omissions-called-out-2026-05-12) |
| Cascade items #2, #27, #28, #29, #30 (paper-canonical) | [v2 §"The five cascade items and where they live in dreamer-srl"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#the-five-cascade-items-and-where-they-live-in-dreamer-srl) |
| Non-goals (no NMN, no FiLM, no continuous actions, no Hydra, no memmap, no sweeps) | [v2 §"Non-goals"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#non-goals-out-of-scope-do-not-do-these) |
| Parity-verification protocol (3 seeds, ≥ 480 mean survival, ≤ 25 h wall-clock) | [v2 §"Parity-verification protocol"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#parity-verification-protocol-after-checkpoints-110-pass) |
| Risks and open questions (Flax-API resolution, `Ratio` semantics, GPU contention, Hafner-init constant precision, …) | [v2 §"Risks and open questions"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#risks-and-open-questions) |
| **Per-function bit-identity tests** | This doc §Lever A |
| **Source-citation discipline** | This doc §Lever B |
| **Per-checkpoint 3-reviewer gate** | This doc §Lever C |
| **Vendored sheeprl + diff tool** | This doc §Lever D |
| **DEVIATION_LOG.md + PI consultation** | This doc §Lever E + [DEVIATION_LOG.md](DEVIATION_LOG.md) |
| **Rewritten checkpoint table** (gates + tests + deviations per CP) | This doc §"Checkpoint table (v3)" |
| **Pre-CP0 setup** | This doc §"Pre-CP0 setup tasks" |

When a developer reads v3, the path is: (1) read this entry section; (2) follow §Lever
A–E for the guardrails; (3) for each checkpoint, follow the v3 checkpoint table
(which cites the v2 file-by-file rows for the algorithmic content). The developer
never has to re-read all 1056 lines of v2 — only the specific row referenced by
each checkpoint.

---

## Lever A — Per-function bit-identity test suite

### What it is

Every public function in `src/algorithms/dreamer_srl/*.py` (the new module from v2)
is paired with a `tests/algorithms/dreamer_srl/test_<file>.py::test_<fn>_matches_sheeprl()`
pytest. The test imports both sides — the JAX function from our new module and the
PyTorch function from the vendored sheeprl@33b6366 at `vendor/sheeprl/` — runs each
on a fixed-seed fixture, and asserts they agree to within `1e-6` max-absolute-difference.

### How it is enforced

The bit-identity test gate is **mandatory per function**. A function is not
marked done in the checkpoint table until its paired test passes. Concretely:

1. Developer ports the function (e.g. `twohot_encode`) from sheeprl PyTorch to JAX.
2. Developer writes the paired test at the canonical path
   `tests/algorithms/dreamer_srl/test_loss.py::test_twohot_encode_matches_sheeprl`.
3. Test fixture lives at `tests/fixtures/dreamer_srl/<fn>_input.npz` —
   fixed PRNG seed, shapes representative of what the function sees in training
   (e.g. for `twohot_encode`, a batch of `[T=16, B=4, 1]` reward targets).
4. Test body:
   ```python
   def test_twohot_encode_matches_sheeprl():
       import torch
       from vendor.sheeprl.sheeprl.utils.distribution import TwoHotEncodingDistribution
       from src.algorithms.dreamer_srl.loss import TwoHotEncoding
       fixture = np.load("tests/fixtures/dreamer_srl/twohot_encode_input.npz")
       logits = fixture["logits"]                                                  # [T,B,255]
       target = fixture["target"]                                                  # [T,B,1]
       # Torch side
       torch_dist = TwoHotEncodingDistribution(torch.tensor(logits), dims=1)
       torch_log_prob = torch_dist.log_prob(torch.tensor(target)).detach().numpy()
       # JAX side
       jax_dist = TwoHotEncoding(jnp.asarray(logits), dims=1)
       jax_log_prob = jax_dist.log_prob(jnp.asarray(target))
       # Bit-identity assert
       max_abs_diff = float(jnp.max(jnp.abs(jnp.asarray(torch_log_prob) - jax_log_prob)))
       assert max_abs_diff < 1e-6, f"twohot_encode deviates by {max_abs_diff}"
   ```
5. Threshold: `1e-6` strict. A 1e-3-deviation test does NOT pass. If the deviation
   is genuinely unavoidable (e.g. JAX defaults to float32 reductions where PyTorch
   uses float64 internally for a numerically sensitive op), the developer logs the
   deviation in `DEVIATION_LOG.md` (§Lever E) and the PI signs it off before merge.

### Skip semantics

Some sheeprl functions are too entangled with global state (Lightning Fabric, the
optimiser-instantiation registry, the Hydra config-resolution graph) to call
in isolation. In those cases the developer cannot directly call the sheeprl function;
instead the developer reads sheeprl's own unit-test suite (`vendor/sheeprl/tests/`)
and ports the equivalent assertion. Each such case is named in the function's
docstring header (§Lever B) — *"Sheeprl-side unit test cited: `vendor/sheeprl/tests/algos/test_dreamer_v3.py:test_xyz`"*.

### Example of what this would have caught

The two-hot bin-grid bug from the cascade debugging: v1 of the original plan
described `self.bins = symexp(linspace(-20, 20, 255))` as the storage form (bins in
real reward space). Sheeprl actually stores `self.bins = linspace(-20, 20, 255)` in
**symlog space** — `symexp` is applied only at consumption sites (`mean`, `mode`).
A `test_twohot_encode_matches_sheeprl` with a non-trivial fixture target (say
`target=0.5`, which maps to `symlog(0.5) ≈ 0.405` and lands between bins 127 and
128) would have failed at `log_prob` mismatch on the first run — long before the
function shipped into a training loop where the bug presents as "the reward head
just doesn't learn." That is the class of bug Lever A exists to catch.

### Test file layout

```
tests/
├── algorithms/
│   └── dreamer_srl/
│       ├── __init__.py
│       ├── test_utils.py          # paired with src/.../utils.py (symlog, init_weights, compute_lambda_values, Moments, Ratio, prepare_obs)
│       ├── test_buffers.py        # paired with src/.../buffers.py (SequentialReplayBuffer)
│       ├── test_agent.py          # paired with src/.../agent.py (LayerNormGRUCell, MLP, encoder/decoder, RSSM, Actor, Critic, Player, build_agent)
│       ├── test_loss.py           # paired with src/.../loss.py (TwoHotEncoding, Symlog, MSE, BernoulliSafeMode, reconstruction_loss)
│       └── test_train.py          # paired with src/.../train.py (one_train_step, collect_step, polyak_update)
└── fixtures/
    └── dreamer_srl/               # .npz files, fixed PRNG seed = 0xD3EAF
        ├── twohot_encode_input.npz
        ├── layernorm_gru_input.npz
        ├── compute_lambda_values_input.npz
        ├── ...
```

Every function listed in [v2 §"Design"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#design)
gets a corresponding test row. The checkpoint table (§"Checkpoint table (v3)") lists
exactly which test names must pass before each checkpoint closes.

---

## Lever B — Source-citation discipline

### What it is

Every public function in `src/algorithms/dreamer_srl/*.py` carries a docstring header
that names the exact sheeprl source file and line range it was ported from, the pinned
commit hash, and a one-line note about any non-obvious gotcha.

### The header template

```python
def twohot_encode(x: jnp.ndarray, bins: jnp.ndarray) -> jnp.ndarray:
    """Encode x into two-hot distribution over bins.

    Ported from sheeprl@33b6366:sheeprl/utils/distribution.py:L185-L260
    (TwoHotEncodingDistribution.__init__ + the log_prob internal two-hot encode).

    GOTCHA: the bin grid lives in **symlog space** — bins[0]=-20, bins[254]=+20
    in symlog units. symexp is applied only at consumption (`mean`/`mode`) via
    `transbwd`, NOT to the stored grid. See v2 cascade item #2 and the
    Checkpoint 5 spec at v3 §"Checkpoint table" for the original bug case.

    Bit-identity test: tests/algorithms/dreamer_srl/test_loss.py::test_twohot_encode_matches_sheeprl
    """
```

Three required elements:
1. **`Ported from sheeprl@<commit-hash>:<path>:<line-range>`** — the pinned commit
   `33b6366`, the file path inside `vendor/sheeprl/`, and the line range with `L`
   prefixes.
2. **`GOTCHA:` (optional but expected for any function with a known non-obvious detail)** —
   one paragraph naming what a literalist implementer would get wrong. The v2 plan's
   "Training-loop semantics" S1–S10 and Risks §1–§14 are the source for this
   paragraph.
3. **`Bit-identity test:`** — the exact pytest node id of the Lever-A test.

### How it is enforced

- `code-reviewer` (§Lever C) rejects any PR for `src/algorithms/dreamer_srl/` where
  a public function is missing this header or where the cited line range does not
  cover the function being ported.
- `code-reviewer` spot-checks the line range by running `cmp` or `diff` between
  the JAX implementation and `vendor/sheeprl/<cited-path>` lines, looking for the
  named structural pattern (e.g. the `1+1` fused-gate form of `LayerNormGRUCell`
  must be visible in the cited line range).
- The diff tool from §Lever D (`scripts/sheeprl_jax_diff.py`) reads the docstring
  header to auto-locate the sheeprl source — the tool's `--function <name>` flag
  parses the header and pulls the cited lines for the side-by-side diff.

### Example of what this would have caught

The Hafner-init constant precision issue (v2 Risks §5): sheeprl uses
`0.87962566103423978`; the existing in-house code at `src/models/dreamer_v3_util.py`
truncated it to `0.8796`. With Lever B, the docstring header on `init_weights`
would cite `vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L149` and the developer
copying the constant from that line would not have a chance to truncate it. The
code-reviewer's `cmp`-check of the cited line would catch a mismatch if it slipped
in anyway.

---

## Lever C — Per-checkpoint 3-reviewer gate

### What it is

The v2 plan has 8 numbered checkpoints (CP1–CP8) plus four sub-checkpoints (CP2b,
CP4b, CP9b, CP10). In v3, each checkpoint becomes a **three-reviewer gate** that
must close before the developer proceeds to the next checkpoint. The three
reviewers fire sequentially (NOT in parallel — they cite each other):

1. **`code-reviewer`** — JAX/Flax-NNX correctness review: `vmap` axis discipline,
   `lax.scan` carry signatures, pytree mutation rules, PRNG threading at JIT
   boundaries, recompilation triggers (static-vs-traced), and **source-citation
   accuracy** (Lever B header line-range checks via `cmp` against `vendor/sheeprl/`).
2. **`math-reviewer`** — equation-vs-sheeprl-source fidelity: every constant
   (KL-balanced coefficients, free-nats floor, lambda-return γ pre-multiplication,
   discount cumprod ordering, two-hot bin grid endpoints in symlog space, EMA
   target-critic decay) matches sheeprl's source line-for-line. Spot-checks
   the v2 Training-loop semantics S1–S10 are honored at the checkpoint's file.
3. **`professor-rl-bayesian-dl`** — algorithm-level fidelity: silent training-loop
   semantics (S1 `is_first[0]=1` force-set, S2 prepend-zero-action shift, S3
   `learning_starts` random-action prefill, S4 three-quantity arithmetic-mask reset,
   S5 true-continue splice, S6 discount weighting, S7 Moments offset cancellation,
   S8 per-element free-nats floor, S9 `Independent(BernoulliSafeMode, 1)` wrap,
   S10 continue target = `1 - terminated`) are wired correctly in the checkpoint's
   scope.

### How it is enforced

The developer:
1. Implements the checkpoint's scope (1 or more files, 1 or more functions per
   checkpoint — see the v3 checkpoint table for the breakdown).
2. Runs the Lever-A bit-identity tests for the checkpoint's functions; all must
   pass at the `1e-6` threshold.
3. Records the test pass/fail and any speed-change measurements in an
   "Implementation Report — CP<N>" block in this plan doc.
4. **Halts implementation.** Spawns `code-reviewer`. When `code-reviewer` writes
   its review at `docs/develop/active/dreamer_srl_v3/review_code_CP<N>.md` with
   verdict `✅ PASS`, spawns `math-reviewer`. When `math-reviewer` writes its
   review with `✅ PASS`, spawns `professor-rl-bayesian-dl`. When all three are
   `✅ PASS`, the developer may proceed to CP<N+1>.
5. If any reviewer flags `⚠️` or `❌` items, the developer either fixes them (preferred)
   or, for unavoidable deviations, logs them in `DEVIATION_LOG.md` and waits for
   PI sign-off (§Lever E) before proceeding.

### Reviewer-output layout

```
docs/develop/active/dreamer_srl_v3/
├── IMPLEMENTATION_PLAN.md                          # this doc
├── DEVIATION_LOG.md                                # §Lever E
├── review_code_CP1.md
├── review_math_CP1.md
├── review_professor_rl_bayesian_dl_CP1.md
├── review_code_CP2.md
├── review_math_CP2.md
├── review_professor_rl_bayesian_dl_CP2.md
... (24 review files total — 3 per CP × 8 CPs)
```

Each reviewer file carries the standard frontmatter (`title`, `topic: dreamer`,
`status: active`, `created`, `last_updated`, `phase: 2`), names the audited checkpoint,
lists `✅`/`⚠️`/`❌` per scope item, and ends with a verdict line. The `_v2`
reviewers' archive files
([review_code_v2.md](../../archive/dreamer_srl/review_code_v2.md),
[review_math_v2.md](../../archive/dreamer_srl/review_math_v2.md),
[review_professor_rl_bayesian_dl_v2.md](../../archive/dreamer_srl/review_professor_rl_bayesian_dl_v2.md))
are the template — they show what `✅ PASS` looks like for the same algorithmic
content.

### What the per-checkpoint focus should be (from v2-reviewer findings)

The v2 reviewers (in their plan-level audits) flagged the highest-risk
silent-pattern-match traps. The per-CP gates focus on the trap class active in
that CP's scope:

| Trap | Active in CP | What the reviewer trio looks for |
|---|---|---|
| `LayerNormGRUCell` 1+1 fused-gate (NOT 2+2); chunk order `(reset, cand, update)` | CP2 | `code-reviewer`: matmul structure 1:1 with cited sheeprl lines. `math-reviewer`: chunk-order ⇒ correct gate semantics. `professor`: confirms reset gate is inside `tanh(reset * cand_proj)`. |
| `is_first` arithmetic-mask reset (three quantities: action, recurrent, posterior with reshape-flatten) | CP4b | `code-reviewer`: form is `(1-is_first)*x + is_first*init`, NOT `jnp.where`. `math-reviewer`: numerical equivalence to sheeprl trace. `professor`: confirms three-quantity scope + S1 force-set. |
| `get_initial_states` uses transition mode, NOT a sample; no PRNG consumed | CP4 | `code-reviewer`: no `random.split` / `random.choice` in the function. `math-reviewer`: softmax-of-uniform-mixed-logits matches sheeprl. `professor`: confirms semantic — initial state is the prior's mode, NOT a draw. |
| `Moments` as pure-functional `flax.struct.dataclass` (NOT `nnx.Variable` mutation) | CP1 | `code-reviewer`: no in-place mutation; return signature `(new_state, offset, invscale)`. `math-reviewer`: `low`/`high` EMA decay matches sheeprl 0.99. `professor`: confirms `max_=1.0` floor + per-rank-local semantics. |
| Two-hot bins in symlog space; `target` symlog-encoded before bin lookup | CP5 | `code-reviewer`: `self.bins = linspace(-20, 20, 255)` literally — no `symexp` wrap. `math-reviewer`: re-checks the three-site consistency from v2 (cascade-table row, Checkpoint 5 spec, `loss.py` row). `professor`: confirms `log_prob` symlog-encodes target. |
| Polyak fires BEFORE `one_train_step`, not after; cumulative-gradient-step semantics | CP7 | `code-reviewer`: call-order in the gradient loop. `math-reviewer`: tau decay schedule. `professor`: confirms `train_step` = cumulative gradient steps applied. |
| Critic uses BOTH `−qv.log_prob(λ.detach())` AND `−qv.log_prob(target_critic_value.detach())` | CP6 | `code-reviewer`: both terms in the JAX loss expression. `math-reviewer`: discount weighting on both. `professor`: confirms target uses **un-normalised** lambda values. |

---

## Lever D — Vendored sheeprl + diff tool

### What it is

A pinned, in-repo copy of sheeprl@`33b6366` at `vendor/sheeprl/`, plus a
side-by-side diff tool at `scripts/sheeprl_jax_diff.py` that runs both the JAX
and sheeprl versions of a function on a fixture and reports max-absolute-difference.

### Why `vendor/` (not `tmp/`)

The previous attempt placed sheeprl at `tmp/sheeprl/` which is gitignored. That
caused a namespace-shadow trap during the cascade debugging session
(2026-05-12) — a `tmp/sheeprl/sheeprl/` import got shadowed when the path was
cleaned. With `vendor/sheeprl/` committed under version control, the source the
developer + reviewers + bit-identity tests + diff tool reference is identical to
what is in the merge artifact at all times.

### Vendoring approach (two equivalent options; developer picks at pre-CP0)

**Option (a) — git submodule:**
```bash
cd /media/nas01/projects/Interoceptive-AI/grid_world_pain
git submodule add https://github.com/Eclectic-Sheep/sheeprl.git vendor/sheeprl
cd vendor/sheeprl
git checkout 33b6366
cd ../..
git add .gitmodules vendor/sheeprl
git commit -m "deps: 📦 vendor sheeprl@33b6366 as submodule"
```

**Option (b) — pinned-commit subtree copy** (simpler if submodule complicates
CI / parallel-checkout):
```bash
cd /tmp && git clone https://github.com/Eclectic-Sheep/sheeprl.git
cd sheeprl && git checkout 33b6366 && rm -rf .git
cp -r /tmp/sheeprl /media/nas01/projects/Interoceptive-AI/grid_world_pain/vendor/
cd /media/nas01/projects/Interoceptive-AI/grid_world_pain
echo "33b6366" > vendor/sheeprl/.pinned_commit
git add vendor/sheeprl
git commit -m "deps: 📦 vendor sheeprl@33b6366 (subtree copy)"
```

Either way, the directory `vendor/sheeprl/` is **never edited** by us — it is
read-only reference source. If sheeprl evolves upstream, we deliberately bump
the pin via a separate PR with a vendor-bump rationale.

### The diff tool

`scripts/sheeprl_jax_diff.py` — a CLI utility used daily by the developer and at
every CP gate by the three reviewers.

```bash
# Single-function diff
python scripts/sheeprl_jax_diff.py \
  --function twohot_encode \
  --fixture tests/fixtures/dreamer_srl/twohot_encode_input.npz \
  --threshold 1e-6

# Output:
#   sheeprl: vendor/sheeprl/sheeprl/utils/distribution.py:L185-L260
#   jax:     src/algorithms/dreamer_srl/loss.py:TwoHotEncoding
#   fixture: shape=(16, 4, 255) logits + (16, 4, 1) target, seed=0xD3EAF
#   max_abs_diff = 1.2e-7
#   PASS  (< 1.0e-6 threshold)

# Whole-checkpoint diff (run by reviewers at each CP gate)
python scripts/sheeprl_jax_diff.py --checkpoint CP5
# Loads the CP5 function list from this plan doc and runs all of them.
# Output: a table with one row per function, max_abs_diff, PASS/FAIL.
```

The tool:
- Resolves `--function <name>` by reading the JAX function's Lever-B docstring
  header (the `Ported from sheeprl@33b6366:<path>:<lines>` line) to locate the
  sheeprl side automatically.
- Loads the named `.npz` fixture (or, with `--checkpoint`, the union of fixtures
  for that CP's functions).
- Runs both sides on the fixture (PyTorch side via `vendor/sheeprl/` imports;
  JAX side via `src/algorithms/dreamer_srl/`).
- Reports `max_abs_diff` and PASS/FAIL against the threshold (default `1e-6`).
- Exits non-zero on FAIL — usable in CI.

### Pre-CP0 deliverables for Lever D

1. `vendor/sheeprl/` populated at commit `33b6366`, committed to git (submodule
   or subtree per developer's pick).
2. `scripts/sheeprl_jax_diff.py` skeleton (CLI argument parsing + fixture loader +
   torch/jax import paths + PASS/FAIL formatting) — actual function-by-function
   logic grows incrementally as each function lands.
3. `tests/fixtures/dreamer_srl/` directory created with a `README.md` documenting
   the fixed PRNG seed (`0xD3EAF`) and the per-function fixture conventions.

---

## Lever E — DEVIATION_LOG.md + PI consultation

### What it is

A single document at `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` that
lists every place we deviated from sheeprl, with PI sign-off per entry. The
file template lives at [DEVIATION_LOG.md](DEVIATION_LOG.md) (created in
pre-CP0).

### Schema

| ID | Function | Sheeprl source line | What we did instead | Why | PI verdict |
|---|---|---|---|---|---|
| D-001 | `twohot_encode.log_prob` numerical precision | `vendor/sheeprl/sheeprl/utils/distribution.py:L237` | Reduction in float32 (JAX default) where sheeprl uses float64 internally for the `torch.nn.functional.one_hot` indexing | JAX has no efficient float64 path on this GPU; the deviation produces `max_abs_diff = 3.2e-7` against sheeprl (PASS at `1e-6` is impossible) | ☐ pending |
| D-002 | … | … | … | … | … |

### How it is enforced

1. **No silent deviations.** Any place the bit-identity test exceeds `1e-6`, OR any
   place the developer cannot match sheeprl line-for-line (e.g. a missing JAX API
   equivalent), the developer logs an entry in `DEVIATION_LOG.md` **before** marking
   the function done.
2. **PI consultation per entry.** The `pi` agent reads each new entry and either
   `✅ APPROVED` (with rationale captured in the log row) or `❌ REJECTED — fix
   required`. A rejected entry blocks the CP gate the function belongs to.
3. **The log is part of the merge artifact.** Before the parity-gate run (the
   3-seed launch at the end of CP8), the log must have zero pending entries —
   every row is either `✅ APPROVED` or has been resolved.

### What this would have caught

The Hafner-init `0.8796` truncation: the developer who shipped that change would
have run the bit-identity test on the affected layer, seen `max_abs_diff = 4e-5`
(or whatever), opened a deviation log entry "truncated the constant for
readability", and the PI would have rejected with "use the full-precision
constant; the truncation is the bug." The deviation never ships. Without Lever E
the same truncation slipped past three reviewers and lived in the codebase for
months.

### Frequency of PI sign-off

Deviations are batched at CP gates — the PI reviews the per-CP delta to the log
when the three-reviewer gate closes, not per-deviation in real time. This keeps
PI involvement bounded (one sign-off block per CP, ≤ ~5 deviations per CP
expected).

---

## Checkpoint table (v3)

Each row is a v2 checkpoint reframed as a v3 gate. The v2 checkpoint content
itself (the "what to test" spec) is unchanged — only the table columns are new.
For the algorithmic content of each CP, follow the link to the v2 row.

| CP | Scope (v2 link) | Lever-A tests required (must all pass at `1e-6`) | Lever-C reviewer chain | Deviation-log entries (target = 0) | Status |
|---|---|---|---|---|---|
| **CP1** | `utils.py` forward parity — [v2 Checkpoint 1](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_symlog_symexp_roundtrip`, `test_init_weights_matches_sheeprl`, `test_uniform_init_weights_matches_sheeprl`, `test_compute_lambda_values_matches_sheeprl`, `test_moments_update_matches_sheeprl`, `test_ratio_matches_sheeprl`, `test_prepare_obs_shape_contract` | code → math → professor | D-001 ✅, D-002 ✅, D-003 ✅ | **CP-PASS (2026-05-13)** — final code state at `77382f2` (F2 D-002 tighten); 8/8 Lever-A PASS at tightened bounds; math + professor reviews ✅ PASS on disk (`ba362e3`); PI sign-off on all 3 deviations at [`f653260`](../../../pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md). Code-reviewer audit fired and drove F1+F3+F2 fixes (commits `46a18cb` + `77382f2`); the agent's `review_code_CP1.md` write did not persist to disk — see Verification subsection. |
| **CP2** | `agent.py` `LayerNormGRUCell` cascade fix #28 — [v2 Checkpoint 2](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_layernorm_gru_cell_matches_sheeprl` (1+1 fused-gate form; chunk order `(reset, cand, update)`; reset gate inside `tanh`) | code → math → professor | D-007 ✅ | **CP-PASS (2026-05-14)** — implementation `df9c328` + LayerNorm eps fix `949f188`; 1/1 Lever-A PASS at `max_abs_diff = 2.947e-4` (< 5e-4 D-007 threshold) post-fix; diff-tool PASS; reset-before-tanh trap active at 336× margin; code + math + professor reviews ✅ PASS on disk (`ebff4d8`); PI sign-off on D-007 at [`6a8b878`](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp2_deviations.md). Code-reviewer audit caught LayerNorm eps mismatch (nnx default `1e-6` vs sheeprl production `1e-3`) — silent-bug class that would have degraded CP4 RSSM parity; fix landed before close-out, validating the guardrail discipline. |
| **CP2b** | Action-shift §S2 test | `test_action_shift_matches_sheeprl` (prepend-zero, drop-last; `[0] == 0`, `[1:] == actions[:-1]`) | code → math → professor | ☐ none | **CP-PASS (2026-05-14)** — implementation `df9c328`; 1/1 Lever-A PASS at `max_abs_diff = 0.000e+00` (exact equality, pure concat + zeros_like); diff-tool PASS; code + math + professor reviews ✅ PASS on disk (`ebff4d8`); no deviation. |
| **CP3** | `agent.py` `build_agent` cascade fix #27 — [v2 Checkpoint 3](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_zero_init_reward_head`, `test_zero_init_critic_head` (kernel + bias both exactly zero) | code → math → professor | ☐ none | NOT STARTED |
| **CP3b** | `buffers.py` `SequentialReplayBuffer` (state-evolution parity) + training-cadence wiring (`Ratio` × `replay_ratio` × `collect_interval` × `learning_starts` × `prefill_steps`) — see [CP3B_SPEC.md](CP3B_SPEC.md) | `test_buffer_storage_state_after_deterministic_adds`, `test_buffer_sample_at_indices_matches_sheeprl`, `test_buffer_is_first_marker_placement_in_straddling_window`, `test_buffer_parallel_env_lane_non_interference`, `test_cadence_yaml_key_parity_with_sheeprl_xs`, `test_cadence_env_grad_step_trace_5000_iters` | code → math → professor | D-004 ✅, D-005 ✅ | **CP-PASS (2026-05-14)** — implementation `9c57c06`; 6/6 Lever-A PASS at `max_abs_diff = 0.000e+00`; code + math + professor reviews ✅ PASS on disk (`ef36099`); PI sign-off on D-004 + D-005 at [`7007723`](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp3b_deviations.md). |
| **CP4** | `agent.py` RSSM cascade fix #30 + `get_initial_states` mode-not-sample — [v2 Checkpoint 4](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_rssm_transition_2layer_mlp`, `test_rssm_representation_2layer_mlp`, `test_get_initial_states_no_prng`, `test_get_initial_states_matches_sheeprl_mode` | code → math → professor | ☐ none | NOT STARTED |
| **CP4b** | RSSM `is_first` reset §S1+§S4 — [v2 Checkpoint 4b](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_is_first_force_set_step0`, `test_is_first_three_quantity_reset` (arithmetic-mask form, posterior reshape-flatten BEFORE masking) | code → math → professor | ☐ none | NOT STARTED |
| **CP5** | `loss.py` two-hot distribution cascade fix #2 (symlog space) — [v2 Checkpoint 5](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_twohot_bins_endpoints` (`bins[0]=-20, bins[127]≈0, bins[254]=+20`, in symlog space), `test_twohot_encode_matches_sheeprl`, `test_twohot_log_prob_target_symlog_encoded` | code → math → professor | D-006 ✅ | **CP-PASS (2026-05-14)** — implementation `fdb09da` (`src/algorithms/dreamer_srl/loss.py` `TwoHotEncoding`); 5/5 tests PASS (3 Lever-A + 2 structural) at D-006 relaxed threshold 3e-5; diff-tool PASS; code + math + professor reviews ✅ PASS on disk (`ff30e77`); PI sign-off on D-006 at [`b2dd5de`](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp5_deviations.md). Historical-scar bug class (`symexp(linspace)` in real-reward-space) structurally prevented at 14-OOM margin by `test_bins_not_symexp_at_storage`. |
| **CP6** | `train.py` critic loss cascade fix #29 — [v2 Checkpoint 6](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_critic_loss_two_terms`, `test_critic_target_un_normalised_lambda`, `test_discount_weighting_critic` (slice `[:-1].squeeze(-1)`) | code → math → professor | ☐ none | NOT STARTED |
| **CP7** | `train.py` Polyak update — [v2 Checkpoint 7](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_polyak_first_call_hard_copy` (tau=1), `test_polyak_subsequent_call_blend` (tau=0.02), `test_polyak_fires_before_train_step` | code → math → professor | ☐ none | NOT STARTED |
| **CP8** | End-to-end forward parity (`scripts/dreamer_srl_offline_check.py`) — [v2 Checkpoint 8](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | All previous Lever-A tests rerun + the offline forward-pass cross-framework parity check (PyTorch sheeprl-trained ckpt vs JAX dreamer-srl freshly initialised at same param count) | code → math → professor | ☐ none (CP8 is the merge-gate — log must be empty of pending entries) | NOT STARTED |
| **CP9** | 5,000-step dry-run on food-only NoPred — [v2 Checkpoint 9](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | No new Lever-A tests; integration smoke test only (no NaN, world-model loss decreasing, ep_len_avg logged) | (post-CP8; reviewer chain optional) | ☐ none | NOT STARTED |
| **CP9b** | Random-action prefill §S3 | `test_prefill_uniform_entropy_below_learning_starts`, `test_no_gradient_step_before_learning_starts` | code → professor (math not needed) | ☐ none | NOT STARTED |
| **CP10** | Wall-clock budget (≤ 2× sheeprl's 12.5 h) | No new tests; measured under speed-check protocol | (no reviewer chain; senior-developer judges per the standard ≤ 5% / ≤ 15% rule in the agent profile) | ☐ none | NOT STARTED |

After CP10 passes, the parity-gate launch follows [v2 §"Parity-verification protocol"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#parity-verification-protocol-after-checkpoints-110-pass)
unchanged: 3 seeds, mean survival ≥ ~480, ≤ 25 h wall-clock per seed.

### CP-id convention (canonical — read before editing CP references)

**CP-ids in this table are stable numerical labels, NOT implementation-order slots.** `CP5` means "two-hot symlog-space bins" for the lifetime of this plan, regardless of when it is built. Two consequences:

1. **Execution order ≠ CP-id order.** §"Implementation order (revised)" moves CP5 to slot #3 (built after CP1 and CP3b) without renumbering. The label `CP5` stays attached to two-hot bins; only the position in the build queue changes.
2. **New CP-ids may be introduced via the `Nb` suffix convention.** A `CP<N>b` row attaches a state-evolution or §S-rule mini-checkpoint to its closest base CP<N>; it does NOT renumber CP<N+1> or later rows. Existing examples: `CP2b` (action shift), `CP4b` (is_first three-quantity reset), `CP9b` (`learning_starts` prefill gating). On 2026-05-13 the convention extended to module-level state-evolution gates with **`CP3b`** (buffer + cadence), promoted from the original "no-CP sanity round-trip" status because of the empirical 16× `replay_ratio` semantic drift caught at the parity gate (see [SPS §3.6](../../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md#36-where-the-jax-advantage-is-coming-from)).

**Every downstream artifact** — `scripts/sheeprl_jax_diff.py:CHECKPOINT_REGISTRY`, `tests/algorithms/dreamer_srl/README.md` file-layout comments, the per-CP review filenames `review_{code,math,professor_rl_bayesian_dl}_CP<N>.md`, deviation-log IDs cross-referencing CPs, fixture filenames — uses **this table's CP-id**, not the implementation-order slot. If a checkpoint is later split, merged, or removed, the affected row's CP-id is retired; the others do **not** renumber.

When unsure: search for the function name in this table's "Scope" column; the row's CP-id is canonical.

### CP3b spec — see sibling doc

The full CP3b spec (8 high-risk items with sheeprl-source citations, 6 Lever-A
tests, per-trap reviewer chain, file-change spec for `developer`, what's
explicitly out of scope) lives in [CP3B_SPEC.md](CP3B_SPEC.md). One-paragraph
summary here so the plan stands on its own:

CP3b promotes the `buffers.py` + training-cadence wiring from "sanity round-trip
only" to a full Lever-A checkpoint, using **state-evolution bit-identity** (drive
both implementations with the same deterministic input; assert resulting state
matches byte-for-byte) instead of pure-function bit-identity (which is impossible
because PyTorch and JAX have different PRNG streams — D-002 class). The 6 Lever-A
tests cover: storage state after deterministic adds (test 1), sample at
pre-computed sheeprl indices (test 2), `is_first` marker placement in straddling
windows (test 3), parallel-env lane non-interference (test 4), `agent_xs.yaml`
key parity with sheeprl XS (cadence test 5), and a 5000-iteration `(env_step,
grad_step)` cadence trace bit-identity (cadence test 6). The whole CP exists
because of the 2026-05-13 SPS finding (16× `replay_ratio` semantic drift caught
at the parity gate); CP9b's `learning_starts` spot-check at the training-loop
level remains the downstream consumer-side complement to CP3b's upstream cadence-
trace bit-identity. Deviation [D-004](DEVIATION_LOG.md#deviation-table) (memmap
omission) is pre-declared for PI ratification alongside CP3b's launch.

### Updating the table

The developer updates the `Status` column at each step:
- `NOT STARTED` → `IN PROGRESS` (after implementation begins)
- `IN PROGRESS` → `CP-PASS` (after all three reviewers ✅ AND deviation-log entries
  for the CP are all `✅ APPROVED`)

The `Deviation-log entries` column links to specific D-### IDs in `DEVIATION_LOG.md`
once entries are filed.

---

## Pre-CP0 setup tasks

The developer runs these once, before CP1 begins. Each is verifiable.

### Pre-CP0.1 — Vendor sheeprl@33b6366

Per §Lever D. Either submodule or subtree-copy at `vendor/sheeprl/`. Verifier:
```bash
cd vendor/sheeprl && git log -1 --format=%H 2>/dev/null
# Must print: 33b6366... (or for subtree: `cat vendor/sheeprl/.pinned_commit` prints "33b6366")
```

### Pre-CP0.2 — Build the diff tool skeleton

Create `scripts/sheeprl_jax_diff.py` with:
- CLI argparse (`--function`, `--fixture`, `--checkpoint`, `--threshold`)
- `.npz` fixture loader
- Stub torch + jax import paths (no function-specific logic yet — that grows
  per-CP)
- PASS/FAIL formatting + non-zero exit on FAIL

Verifier: `python scripts/sheeprl_jax_diff.py --help` prints the expected usage
block without crashing.

### Pre-CP0.3 — Create the test directory + fixture conventions

```bash
mkdir -p tests/algorithms/dreamer_srl tests/fixtures/dreamer_srl
touch tests/algorithms/dreamer_srl/__init__.py
```

Write `tests/fixtures/dreamer_srl/README.md` documenting:
- Fixed PRNG seed: `0xD3EAF`
- Per-function fixture naming: `<function-name>_input.npz`
- Fixture-generation script convention: each CP gets a tiny
  `scripts/fixtures/gen_<cp>_fixtures.py` that re-creates the `.npz` files
  deterministically (so a reviewer can regenerate and verify).

### Pre-CP0.4 — Create DEVIATION_LOG.md from template

The file is created in pre-CP0 (by this plan-drafting step — already done; see
[DEVIATION_LOG.md](DEVIATION_LOG.md)). The developer confirms it exists and
contains the schema header.

### Pre-CP0.5 — Confirm Flax-NNX patterns from existing JAX Dreamer

Read `src/models/dreamer_v3_nnx.py` and `src/models/dreamer_v3_trainer.py` for
the project's NNX conventions (`nnx.Rngs` construction, `nnx.split` / `nnx.merge`
at JIT boundaries, EMA via `nnx.state` / `nnx.update`). The dreamer-srl module
follows these conventions but **does not import from** the existing files (v2
Risks §13). Verifier: a one-paragraph note added to the developer's CP1
Implementation Report confirming the conventions read and adopted.

### Pre-CP0.6 — Confirm `config.get_mandatory` audit for every new YAML key

Per v2 Risks §10 — every key in `configs/dreamer_srl/agent_xs.yaml` is read via
`Config.get_mandatory`, no `config.get('key', default)` allowed. Verifier: a one-paragraph
note in CP1 Implementation Report listing the loader pattern.

---

## Effort estimate (with guardrails)

A rough breakdown for the user to recalibrate against if needed. The naive v2
estimate was "2–6 weeks build, 1 week parity run = 3–7 weeks total." With the
five guardrail layers on, the per-CP cost roughly doubles because of the
three-reviewer gate and fixture generation. The fixture generation is the
single largest new cost — each Lever-A test needs a deterministic PyTorch-side
trace + JAX-side fixture that matches the trace.

| Phase | Naive v2 cost | v3 cost (with guardrails) | Drivers |
|---|---|---|---|
| Pre-CP0 setup | n/a | 2–3 days | Vendor sheeprl; diff-tool skeleton; fixture directory; NNX convention read |
| CP1 (utils.py — 7 functions) | 2 days | 4–5 days | 7 Lever-A tests + fixtures; 3-reviewer gate; ~2 deviation-log entries likely |
| CP3b (buffers.py + cadence — 6 tests) | n/a (was "integration smoke only" — promoted 2026-05-13) | 3–4 days | 4 storage/sample state-evolution tests + 2 cadence-trace tests + fixtures; 3-reviewer gate; D-004 (memmap omission) pre-declared for PI ratification |
| CP2 + CP2b (LayerNormGRUCell + action shift) | 1 day | 2–3 days | 2 Lever-A tests + fixtures; 3-reviewer gate (LayerNormGRU is the highest-risk silent-pattern-match item; review will be slow) |
| CP3 (build_agent zero-init heads) | 0.5 day | 1 day | 2 Lever-A tests; quick gate |
| CP4 + CP4b (RSSM hidden layers + is_first reset) | 2 days | 4–5 days | 4 Lever-A tests + fixtures; 3-reviewer gate; `is_first` three-quantity reset is the second-highest-risk item |
| CP5 (two-hot symlog-space bins) | 1 day | 2 days | 3 Lever-A tests + fixtures; 3-reviewer gate (the historical-bug case, so the review is detailed) |
| CP6 (critic two-term loss) | 1 day | 2 days | 3 Lever-A tests + fixtures; 3-reviewer gate |
| CP7 (Polyak update) | 0.5 day | 1 day | 3 Lever-A tests; quick gate |
| CP8 (end-to-end forward parity) | 2 days | 3–4 days | All previous tests rerun + offline forward-pass check; 3-reviewer gate is the merge-gate |
| CP9 / CP9b / CP10 (integration smokes + speed check) | 1–2 days | 2 days | Mostly unchanged — no new Lever-A overhead |
| Parity-gate launch (3 seeds) | 1 week wall-clock | 1 week wall-clock | Unchanged — only the launch waits |
| **Total** | **~3 weeks dev + 1 week run = 4 weeks** | **~5.5 weeks dev + 1 week run = 6.5 weeks** | ~1.6× overhead from guardrails (was 1.5× before CP3b promotion; +3–4 days) |

**Caveat to the user.** The 1.6× overhead is the rough estimate, NOT 2×. The
CP3b promotion (2026-05-13) added ~3–4 days for the buffer + cadence state-
evolution gate; the user chose this trade explicitly because the alternative is
finding cadence/storage drift at the parity gate (where the 16× `replay_ratio`
mismatch was actually found). If the bit-identity tests find more deviations
than expected (more than ~2 per CP on average, i.e. > ~16 total), the
deviation-log + PI-sign-off cycle adds more time. If they find fewer (i.e. the
v2 plan was as precise as the v2 reviewers claimed), the overhead is closer to
1.3×.

If the user wants to lower the cost: dropping Lever C (per-checkpoint 3-reviewer
gate) and keeping only Levers A + B + D + E reduces the overhead to ~1.2×,
trading the in-process review for a single end-of-plan review like v2 had. The
user did not pick that option; flagging it explicitly so the user can recalibrate.

---

## Implementation order (revised)

The v2 implementation order [v2 §"Implementation order"](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#implementation-order-step-by-step-for-developer)
remains the spine. The v3 changes:

0. **Pre-CP0 setup** (§"Pre-CP0 setup tasks" above) — vendor sheeprl, diff tool,
   test dirs, NNX-convention read, mandatory-key audit. **All five sub-steps must
   complete and be committed before CP1 begins.** **DONE** — `0bcf5c6` + `292dd3a`.
1. **CP1 — `utils.py`** → Lever-A tests → 3-reviewer gate → CP-PASS. **DONE 2026-05-13** — `77382f2` (final F2 state); math + professor on disk at `ba362e3`; PI sign-off `f653260`.
2. **CP3b — `buffers.py` (state-evolution parity) + training-cadence wiring** —
   moved into the build queue ahead of CP5 because the historical-scar argument
   for CP3b (the 2026-05-13 SPS finding of 16× `replay_ratio` semantic drift
   between JAX and sheeprl, caught at the parity gate) is structurally identical
   to the historical-scar argument for CP5 (twohot bins). Implement the
   buffer/cadence discipline early so the state-evolution-parity pattern is
   visible to all downstream CPs. Eligible to start immediately after CP1
   (no dependencies on CP2/CP3/CP4/CP5). D-004 (memmap omission) + D-005
   (test-scope unfilled-region exclusion) go to PI for ratification alongside
   this CP's launch. → Lever-A tests (4 storage/sample + 2 cadence) →
   3-reviewer gate → CP-PASS. **DONE 2026-05-14** — `9c57c06` (implementation);
   reviewer chain on disk at `ef36099`; PI sign-off on D-004 + D-005 at `7007723`.
3. **CP5 — `loss.py` two-hot distribution** (moved earlier; the two-hot bug is
   the historical scar — implement it third so the symlog-space discipline is
   set early and visible) → Lever-A tests → 3-reviewer gate → CP-PASS. **DONE 2026-05-14** —
   `fdb09da` (implementation: `TwoHotEncoding` class in `src/algorithms/dreamer_srl/loss.py`);
   reviewer chain on disk at `ff30e77`; PI sign-off on D-006 at `b2dd5de`. Historical-
   scar bug class structurally prevented at 14-OOM margin.
4. **CP2 + CP2b** → tests → gate → CP-PASS. **DONE 2026-05-14** — `df9c328`
   (implementation in `src/algorithms/dreamer_srl/agent.py`: `LayerNormGRUCell` +
   `action_shift`) + `949f188` (LayerNorm eps fix `1e-6 → 1e-3` landed by
   code-reviewer audit); reviewer chain on disk at `ebff4d8`; PI sign-off on D-007
   at `6a8b878`. Single developer task covered both since they share the same file.
5. **CP3** → tests → gate → CP-PASS. **← NEXT** (`build_agent` final init phase
   applies `uniform_init_weights(scale=0.0)` to reward-head + critic-head output
   linears; eligible once user authorizes; do not start without authorization).
6. **CP4 + CP4b** → tests → gate → CP-PASS.
7. **CP6** → tests → gate → CP-PASS.
8. **CP7** → tests → gate → CP-PASS.
9. **CP8 — end-to-end forward parity** → all previous tests rerun + offline
   forward-pass check → merge-gate review → CP-PASS.
10. **CP9 + CP9b** → integration smoke → CP-PASS. (CP9b's `learning_starts`
    spot-check at the training-loop level is now the consumer-side complement to
    CP3b's upstream cadence-trace bit-identity; the two checkpoints share no
    test functions.)
11. **CP10** → wall-clock budget → senior-developer speed verdict.
12. **Parity-gate launch** → 3 seeds via `training-runner` → `experiment-analyzer`
    writes the verdict report under `docs/experiments/active/diagnosis/`.

---

## Doc-framing rule (CLAUDE.md compliance)

Per [CLAUDE.md "Documentation framing"](../../../../CLAUDE.md), the first body section
of any develop doc is a plain-English entry point readable by a fresh reader. This
plan opens with §"Plain-language entry point" (~200 words, no bare WandB run IDs,
no bare config paths, no bare predicate / shorthand names) — for example, `H₀`,
`Δ_SS`, run `jzgkcep4`, file `configs/dreamer_srl/agent_xs.yaml` are introduced only
in later sections. The plan's symbolic / file-path / shorthand detail lives in §"How
v3 relates to v2" onward.

The plan's structure:
1. Plain-language entry point (~200 words)
2. How v3 relates to v2 (where the v2 content is + the new v3 deltas)
3. Lever A — bit-identity test suite
4. Lever B — source-citation discipline
5. Lever C — per-checkpoint 3-reviewer gate
6. Lever D — vendored sheeprl + diff tool
7. Lever E — DEVIATION_LOG.md + PI consultation
8. Checkpoint table (v3) — 8 CPs + 4 sub-CPs as gates
9. Pre-CP0 setup tasks (six concrete items)
10. Effort estimate (with guardrails)
11. Implementation order (revised)
12. Doc-framing rule (this section)
13. Links + Implementation Report stub

---

## Links

- [v2 plan (algorithmic backbone)](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md)
- [v2 review — code](../../archive/dreamer_srl/review_code_v2.md)
- [v2 review — math](../../archive/dreamer_srl/review_math_v2.md)
- [v2 review — RL/Bayesian-DL professor](../../archive/dreamer_srl/review_professor_rl_bayesian_dl_v2.md)
- [PI call 2026-05-12 — original Option-1 (sheeprl-direct) pivot](../../../pi/calls/2026-05-12_dreamer_backend.md)
- [SPS comparison memo — the empirical basis for re-opening](../../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md)
- [Sheeprl walkthrough INDEX](../../../project/references/sheeprl_dreamer_v3/INDEX.md)
- [DEVIATION_LOG.md](DEVIATION_LOG.md)
- [Issue plan template](../../../TEMPLATES/issue_plan.md)
- [CLAUDE.md — doc-framing rule](../../../../CLAUDE.md)
- [Develop frontmatter contract](../meta/FRONTMATTER_CONTRACT.md)

---

## Implementation Report

> **Implemented by**: developer agent
> **Date**: 2026-05-13

<!-- Filled by the developer per CP. Each CP gets its own sub-block. -->

### Pre-CP0
- [x] Pre-CP0.1 — vendor sheeprl@33b6366 at `vendor/sheeprl/` (subtree copy, `.pinned_commit` = `33b6366`; verified `wc -l dreamer_v3.py` = 780)
- [x] Pre-CP0.2 — `scripts/sheeprl_jax_diff.py` skeleton (argparse, fixture loader, compare(), FUNCTION_REGISTRY, CHECKPOINT_REGISTRY; `--help` prints without crash)
- [x] Pre-CP0.3 — `tests/algorithms/dreamer_srl/__init__.py` + README; `tests/fixtures/dreamer_srl/README.md` (seed 0xD3EAF, naming convention, generation script convention)
- [x] Pre-CP0.4 — `DEVIATION_LOG.md` confirmed present with schema header (7-column table + enforcement rules + approved/rejected sections)
- [x] Pre-CP0.5 — NNX-convention read complete; `docs/develop/active/dreamer_srl_v3/NNX_CONVENTIONS.md` written (~50 lines). Summary: `nnx.Rngs` passed to `__init__` only (never stored, never passed to `__call__`); forward-pass PRNG via explicit `jax.random.PRNGKey` args; JIT boundary via `@nnx.jit` (or `nnx.split`/`nnx.merge`/`nnx.state`/`nnx.update` for plain-`jax.jit` boundaries); EMA via `nnx.state` + arithmetic + `nnx.update`. Isolation confirmed: dreamer-srl will not import from `src.models.dreamer_v3_*`.
- [x] Pre-CP0.6 — `config.get_mandatory` smoke test (3 cases: present key, missing key raises `ValueError`, nested missing key raises `ValueError`); all PASS. CP1 loader pattern: every YAML key via `config.get_mandatory('key', type_converter)`; no `config.get('key', default)` anywhere in `src/algorithms/dreamer_srl/`.

#### Speed check
Pre-CP0 is infrastructure only (no hot-path code). No algorithm code created under `src/`. Speed check skipped per protocol.

#### Files created / changed (pre-CP0)
| File | Action |
|---|---|
| `vendor/sheeprl/` | Created (subtree copy, 780-line `dreamer_v3.py` verified) |
| `vendor/sheeprl/.pinned_commit` | Created (`33b6366`) |
| `.gitattributes` | Created (`vendor/sheeprl/** linguist-vendored`) |
| `scripts/sheeprl_jax_diff.py` | Created (skeleton: argparse + loader + compare + registries) |
| `tests/algorithms/dreamer_srl/__init__.py` | Created (empty) |
| `tests/algorithms/dreamer_srl/README.md` | Created (naming convention + Lever-A gate rule + template) |
| `tests/fixtures/dreamer_srl/README.md` | Created (seed 0xD3EAF + naming + shapes table) |
| `docs/develop/active/dreamer_srl_v3/NNX_CONVENTIONS.md` | Created (NNX pattern reference for CP1+) |
| `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` | Confirmed present (no changes; senior-dev created) |

### CP1 — `utils.py`

**Implemented by**: developer agent | **Date**: 2026-05-13

#### Files created / changed
| File | Action |
|---|---|
| `src/algorithms/dreamer_srl/__init__.py` | Created (package init) |
| `src/algorithms/dreamer_srl/utils.py` | Created — 7 functions: `symlog`, `symexp`, `init_weights`, `uniform_init_weights`, `compute_lambda_values`, `MomentsState` + `moments_init` + `moments_update`, `Ratio`, `prepare_obs` |
| `tests/algorithms/dreamer_srl/test_utils.py` | Created — 8 Lever-A bit-identity tests (one per function) |
| `tests/fixtures/dreamer_srl/symlog_input.npz` | Created (seed 0xD3EAF) |
| `tests/fixtures/dreamer_srl/symexp_input.npz` | Created |
| `tests/fixtures/dreamer_srl/init_weights_input.npz` | Created |
| `tests/fixtures/dreamer_srl/uniform_init_weights_input.npz` | Created |
| `tests/fixtures/dreamer_srl/compute_lambda_values_input.npz` | Created |
| `tests/fixtures/dreamer_srl/moments_update_input.npz` | Created |
| `tests/fixtures/dreamer_srl/ratio_input.npz` | Created |
| `tests/fixtures/dreamer_srl/prepare_obs_input.npz` | Created |
| `scripts/fixtures/gen_cp1_fixtures.py` | Created — deterministic fixture generator (sheeprl_bridge env) |
| `scripts/sheeprl_jax_diff.py` | Updated — CP1 runners added to FUNCTION_REGISTRY; FUNCTION_THRESHOLDS dict added for per-function threshold overrides (D-003 symexp) |
| `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` | Updated — D-001, D-002, D-003 logged |

#### Lever-A test results
```
pytest tests/algorithms/dreamer_srl/test_utils.py -v
8 passed in 4.35s
```

#### Diff tool sweep — `python scripts/sheeprl_jax_diff.py --checkpoint CP1`
```
symlog                  PASS  max_abs_diff=2.384e-07 (< 1.0e-06)
symexp                  PASS  max_abs_diff=1.526e-05 (< 2.0e-05 D-003 threshold)
init_weights            PASS  distribution test — std rel-err=12.1% < 15% (D-002)
uniform_init_weights    PASS  distribution test — max_abs_val <= limit (D-002)
compute_lambda_values   PASS  max_abs_diff=4.768e-07 (< 1.0e-06)
moments_update          PASS  max_abs_diff=8.196e-08 (< 1.0e-06, D-001 all_gather omitted)
ratio                   PASS  integer-exact (0 diff)
prepare_obs             PASS  max_abs_diff=0.000e+00 (exact)
```

#### Deviation-log entries for CP1
- **D-001**: `moments_update` — `fabric.all_gather` omitted (no-op on single-process). Numerical output bit-identical to single-process sheeprl. **PI verdict: ☐ pending**
- **D-002**: `init_weights` / `uniform_init_weights` — stochastic functions with different RNG implementations; distribution property test instead of bit-identity. Formula is identical to sheeprl. **PI verdict: ☐ pending**
- **D-003**: `symexp` — float32 GPU exp 1-ULP difference; max_abs_diff=1.526e-5; max_rel_diff=2.1e-7. Formula identical to sheeprl. Threshold relaxed to 2e-5. **PI verdict: ☐ pending**

#### Pre-CP0.5 NNX convention note
Read `src/models/dreamer_v3_nnx.py` (735 lines) and `src/models/dreamer_v3_trainer.py` (1023 lines). Key conventions adopted: `nnx.Rngs` in `__init__` only (not stored, not passed to `__call__`); JIT via `@nnx.jit`; EMA via `nnx.state` + arithmetic + `nnx.update`. `dreamer_srl` module does NOT import from `src.models.*` (isolation enforced, grep returns empty).

#### Pre-CP0.6 config.get_mandatory note
CP1 is infrastructure (`utils.py`) — no config key reads in this file. The loader pattern (`config.get_mandatory`) will be enforced in `train.py` and `agent.py` at CP2+.

#### Speed check
CP1 is pure utility functions (no training loop hot path). No speed check required per protocol.

#### Reviewer chain
- [x] `code-reviewer` ⚠️ PASS WITH FIX (F1+F3+F2 fixes landed, F4+F5 deferred) → audit fired but `review_code_CP1.md` did NOT persist to disk; findings live in commits `46a18cb` (F1+F3) and `77382f2` (F2). See **Verification** subsection below for the gap-disposition decision.
- [x] `math-reviewer` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp1_math_review.md`](../../../reviews/dreamer_srl_v3_cp1_math_review.md) (committed `ba362e3`).
- [x] `professor-rl-bayesian-dl` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp1_professor_rl_bayesian_dl_review.md`](../../../reviews/dreamer_srl_v3_cp1_professor_rl_bayesian_dl_review.md) (committed `ba362e3`).
- [x] PI sign-off on D-001, D-002, D-003 — all ✅ APPROVED at [`docs/pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md`](../../../pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md) (committed `f653260`).

Status: **CP-PASS (2026-05-13)** — final code state `77382f2`; all four gates closed; PI cleared deviation log; CP5 is next eligible.

---

#### Verification (senior-developer, 2026-05-13)

Plain-language summary of the CP-PASS decision. CP1 (the JAX port of sheeprl's `utils.py` — seven functions: `symlog`/`symexp`, weight-init helpers, lambda-return computer, the running-statistics `Moments` state, the `Ratio` integer ratio helper, and the obs-prep helper) has cleared all four gates that the v3 plan requires before a checkpoint may close: (a) every paired bit-identity test passes at the tightened thresholds — `symlog` and `compute_lambda_values` at the canonical `1e-6`; `symexp` at the relaxed `2e-5` justified by hardware float32 ULP drift; `init_weights` at the tightened distribution-property bound after F2 raised the fixture sample size 64×; `moments_update` at `8.2e-8`; `ratio` integer-exact; `prepare_obs` exact-zero. (b) The source-citation discipline (`Ported from sheeprl@33b6366:<path>:<line-range>` headers + GOTCHA paragraphs + Bit-identity-test references) is present on every public function and was line-checked by the math reviewer. (c) The three-reviewer chain closed — math and professor reviews are committed and readable; the code-reviewer audit also fired (its F1+F3+F2 findings are why those commits exist) but the agent's write to `review_code_CP1.md` did not land on disk. (d) The PI signed off on all three deviations (D-001 `all_gather` dropped under single-process Fabric, D-002 distribution-test-not-bit-identity for the cross-PRNG weight inits, D-003 ULP-drift threshold relaxation for `symexp`).

| Lever / gate | Verdict | Evidence |
|---|---|---|
| Lever A — bit-identity tests | ✅ | 8/8 PASS at the tightened bounds; diff-tool sweep at `77382f2`; pytest run `8 passed in 4.35s` recorded in CP1 Implementation Report |
| Lever B — source citations | ✅ | math-reviewer line-checked the docstring headers and cited line ranges as part of its audit; verdict `✅ PASS` in `docs/reviews/dreamer_srl_v3_cp1_math_review.md` |
| Lever C — code-reviewer | ⚠️ on-disk artifact missing — see follow-up; audit DID fire and drive the F1 (NamedTuple → `flax.struct.dataclass`), F3 (docstring-content drift), and F2 (D-002 fixture tighten) fixes that landed in `46a18cb` and `77382f2`. F4 (D-002 print format) and F5 (`compute_lambda_values` edge-case wording) deferred per code-reviewer recommendation. |
| Lever C — math-reviewer | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp1_math_review.md`](../../../reviews/dreamer_srl_v3_cp1_math_review.md) (`ba362e3`) |
| Lever C — professor-rl-bayesian-dl | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp1_professor_rl_bayesian_dl_review.md`](../../../reviews/dreamer_srl_v3_cp1_professor_rl_bayesian_dl_review.md) (`ba362e3`) |
| Lever D — diff tool | ✅ | `scripts/sheeprl_jax_diff.py --checkpoint CP1` runs all 7 functions + `symlog_symexp_roundtrip` (8 entries) and prints PASS for each at the tightened thresholds |
| Lever E — PI sign-off | ✅ | All 3 deviations APPROVED at [`docs/pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md`](../../../pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md) (`f653260`); DEVIATION_LOG.md table updated with approval anchors |
| Speed check | n/a | CP1 is pure utility code with no training-loop hot path; the per-CP speed protocol's "no-op" case applies (see CP1 Implementation Report § Speed check) |
| Scope drift | none flagged | All changed paths are inside the CP1-scoped set (`src/algorithms/dreamer_srl/utils.py`, paired tests, fixtures, the diff-tool registries, DEVIATION_LOG, this plan, two review files, one PI doc); no out-of-scope source modifications |

**Code-reviewer on-disk artifact — disposition.** The CP1 code-reviewer agent ran (its findings F1, F3, F2 are what drove `46a18cb` and `77382f2`; F4 and F5 are documented as deferred-nits in those commit messages and in this plan's reviewer-chain block) but its write to `docs/reviews/dreamer_srl_v3_cp1_code_review.md` did not persist. The senior-developer evaluated three closure paths — (A) reconstruct the doc from the session transcript, (B) skip the file and rely on the chain, (C) re-spawn the code-reviewer at the current state — and chose **B**: the audit was effective (the fixes are in the tree; the deferred nits are recorded; the math + professor + PI artifacts are persisted; the four-gate chain is verifiable end-to-end without the missing file). Reconstruction is editorial and risks drifting from what the agent actually said; re-spawning would re-litigate closed issues against the F2-landed state and create a structurally post-hoc audit. The gap is documented here (this subsection + the CP1 row note) rather than silently absorbed; if a future reader needs the per-finding detail, the commit messages on `46a18cb` and `77382f2` are the canonical source.

**Conclusion.** CP1 → **CP-PASS** at `77382f2`. CP5 is the next eligible checkpoint per the v3 implementation order (slot #2 — `loss.py` two-hot symlog-space distribution, the historical-scar function). The user authorizes the CP1 → CP5 transition; the senior-developer does not spawn `developer` for CP5 without that authorization.

**Verified by**: senior-developer
**Date**: 2026-05-13

### CP3b — `buffers.py` + training-cadence wiring

**Implemented by**: developer agent | **Date**: 2026-05-14

#### Files created / changed
| File | Action |
|---|---|
| `src/algorithms/dreamer_srl/buffers.py` | Created — `SequentialReplayBuffer` (ring-buffer, parallel-env-lane storage, `add` + `sample` + `_sample_at_indices` + `is_first` marker placement) + `Ratio` × `replay_ratio` cadence helpers |
| `configs/dreamer_srl/agent_xs.yaml` | Updated — 9 cadence keys added (`buffer_size`, `replay_ratio`, `collect_interval`, `learning_starts`, `prefill_steps`, etc.) matching sheeprl XS at `33b6366` |
| `tests/algorithms/dreamer_srl/test_buffers.py` | Created — 6 Lever-A state-evolution bit-identity tests |
| `tests/fixtures/dreamer_srl/test_buffer_storage_state_after_deterministic_adds_input.npz` | Created |
| `tests/fixtures/dreamer_srl/test_buffer_sample_at_indices_matches_sheeprl_input.npz` | Created |
| `tests/fixtures/dreamer_srl/test_buffer_is_first_marker_placement_in_straddling_window_input.npz` | Created |
| `tests/fixtures/dreamer_srl/test_buffer_parallel_env_lane_non_interference_input.npz` | Created |
| `tests/fixtures/dreamer_srl/test_cadence_yaml_key_parity_with_sheeprl_xs_input.npz` | Created |
| `tests/fixtures/dreamer_srl/test_cadence_env_grad_step_trace_5000_iters_input.npz` | Created |
| `scripts/fixtures/gen_cp3b_fixtures.py` | Created — deterministic CP3b fixture generator (sheeprl_bridge env) |
| `scripts/sheeprl_jax_diff.py` | Updated — CP3b runners added to FUNCTION_REGISTRY + CHECKPOINT_REGISTRY |
| `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` | Updated — D-004 + D-005 logged |
| `docs/develop/active/dreamer_srl_v3/CP3B_SPEC.md` | Updated — implementation notes appended |

#### Lever-A test results
```
pytest tests/algorithms/dreamer_srl/test_buffers.py -v
6 passed
```

#### Diff tool sweep — `python scripts/sheeprl_jax_diff.py --checkpoint CP3b`
```
buffer_storage_state_after_deterministic_adds          PASS  max_abs_diff=0.000e+00 (filled region; D-005)
buffer_sample_at_indices_matches_sheeprl               PASS  max_abs_diff=0.000e+00
buffer_is_first_marker_placement_in_straddling_window  PASS  is_first=1 at expected offset
buffer_parallel_env_lane_non_interference              PASS  4 env lanes, no cross-lane leakage
cadence_yaml_key_parity_with_sheeprl_xs                PASS  9 keys match sheeprl XS @ 33b6366
cadence_env_grad_step_trace_5000_iters                 PASS  5000-iter (env_step, grad_step) trace bit-identical
```

Exit code: 0.

#### Deviation-log entries for CP3b
- **D-004**: `SequentialReplayBuffer` — `memmap` / `memmap_dir` / `memmap_mode` argument trio omitted entirely; in-RAM storage only. Pre-declared at v2 design stage; substrate-mechanical (storage backend, not algorithm semantics). **PI verdict: ✅ APPROVED 2026-05-14**
- **D-005**: `test_buffer_state_evolution_matches_sheeprl` — bit-identity comparison restricted to filled region `[:_pos]`; unfilled tail excluded because `np.empty` allocation is uninitialised by contract. Filled region byte-identical at `0.000e+00`; `_pos` + `_full` independently asserted. **PI verdict: ✅ APPROVED 2026-05-14**

#### Speed check
CP3b code paths (buffer `add` / `sample` / `_sample_at_indices`) are CPU-side and live outside the JIT'd training step; the cadence helpers (`Ratio`, `replay_ratio`) are integer arithmetic. No training-loop hot path touched at this checkpoint. Speed check skipped per protocol; the wall-clock budget verdict lives at CP10.

#### Reviewer chain
- [x] `code-reviewer` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp3b_code_review.md`](../../../reviews/dreamer_srl_v3_cp3b_code_review.md) (committed `ef36099`).
- [x] `math-reviewer` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp3b_math_review.md`](../../../reviews/dreamer_srl_v3_cp3b_math_review.md) (committed `ef36099`). All 8 equations match sheeprl@33b6366 line-for-line.
- [x] `professor-rl-bayesian-dl` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp3b_professor_rl_bayesian_dl_review.md`](../../../reviews/dreamer_srl_v3_cp3b_professor_rl_bayesian_dl_review.md) (committed `ef36099`). All 8 critical algorithm-points covered; §S scope verified.
- [x] PI sign-off on D-004, D-005 — both ✅ APPROVED at [`docs/pi/calls/2026-05-14_dreamer_srl_v3_cp3b_deviations.md`](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp3b_deviations.md) (committed `7007723`).

Status: **CP-PASS (2026-05-14)** — implementation `9c57c06`; all four gates closed; PI cleared deviation log; CP5 is next eligible.

---

#### Verification (senior-developer, 2026-05-14)

Plain-language summary of the CP-PASS decision. CP3b (the JAX port of sheeprl's replay buffer `SequentialReplayBuffer` plus the training-cadence wiring that schedules environment-step vs. gradient-step ratios — the layer whose 16× semantic drift was caught at the v2 parity gate and motivated the CP3b promotion in the first place) has cleared all four gates the v3 plan requires. (a) Every state-evolution bit-identity test passes at `max_abs_diff = 0.000e+00` — the tightest possible floor: storage state after a deterministic add-sequence, `_sample_at_indices` at sheeprl-precomputed indices, `is_first` marker placement in a window straddling a done boundary, four-env-lane non-interference, the 9-key cadence YAML matches sheeprl XS at `33b6366`, and a 5000-iteration `(env_step, grad_step)` cadence trace is byte-for-byte identical. (b) The source-citation discipline (`Ported from sheeprl@33b6366:<path>:<line-range>` headers + GOTCHA paragraphs + Bit-identity-test references) is present on every public function in `buffers.py` and was line-checked by the math-reviewer. (c) The three-reviewer chain closed cleanly — code, math, and professor-rl-bayesian-dl audits all persisted to disk at `ef36099` (no missing-file gap this time, unlike CP1). (d) The PI signed off on both deviations (D-004 memmap omission as substrate-mechanical; D-005 unfilled-region exclusion as test-scope honesty about `np.empty`'s uninitialised-tail contract).

| Lever / gate | Verdict | Evidence |
|---|---|---|
| Lever A — state-evolution bit-identity tests | ✅ | 6/6 PASS at `max_abs_diff = 0.000e+00`; diff-tool sweep at `9c57c06` (`scripts/sheeprl_jax_diff.py --checkpoint CP3b` exits 0); pytest `6 passed` recorded in CP3b Implementation Report |
| Lever B — source citations | ✅ | math-reviewer line-checked the docstring headers and cited line ranges against `vendor/sheeprl/sheeprl/data/buffers.py` @ `33b6366`; verdict `✅ PASS` in `docs/reviews/dreamer_srl_v3_cp3b_math_review.md` |
| Lever C — code-reviewer | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp3b_code_review.md`](../../../reviews/dreamer_srl_v3_cp3b_code_review.md) (`ef36099`). 0.000e+00 results verified genuine (fixtures generated via vendored sheeprl side, JAX tested against stored bytes — no self-comparison); source-citation line ranges accurate; isolation rule + memmap rule honored. One nit (non-blocker): `SequentialReplayBuffer` is a plain Python class not a `flax.struct.dataclass`; rationale (CPU-side, never crosses JIT; matches sheeprl OOP + CP1's `Ratio` precedent) accepted. |
| Lever C — math-reviewer | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp3b_math_review.md`](../../../reviews/dreamer_srl_v3_cp3b_math_review.md) (`ef36099`). All 8 equations match sheeprl@`33b6366` line-for-line: ring-buffer wrap, modular sample-window, valid-start arithmetic, env-tiled flat-index time-major C-order, `is_first` offset, prefill off-by-one, ratio shift, `Ratio` scheduler. D-004 + D-005 math-invariant. |
| Lever C — professor-rl-bayesian-dl | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp3b_professor_rl_bayesian_dl_review.md`](../../../reviews/dreamer_srl_v3_cp3b_professor_rl_bayesian_dl_review.md) (`ef36099`). All 8 critical algorithm-points covered. §S-rules correctly scoped (CP3b substrates §S1, §S2-half, §S3-gate-arithmetic-half, §S4; §S2 call-site + §S3 random-action-prefill correctly deferred to CP9b). Hand-off notes for CP4b, CP2b, CP9b downstream reviewers included. |
| Lever D — diff tool | ✅ | `scripts/sheeprl_jax_diff.py --checkpoint CP3b` runs all 6 CP3b tests against `vendor/sheeprl/sheeprl/data/buffers.py` @ `33b6366` and prints PASS for each at `max_abs_diff = 0.000e+00` (Test 1 over the filled region per D-005) |
| Lever E — PI sign-off | ✅ | Both D-004 and D-005 ✅ APPROVED at [`docs/pi/calls/2026-05-14_dreamer_srl_v3_cp3b_deviations.md`](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp3b_deviations.md) (`7007723`); DEVIATION_LOG.md table updated with approval anchors |
| Speed check | n/a | CP3b code paths are CPU-side (buffer `add`/`sample`/`_sample_at_indices`) outside the JIT'd training step; cadence helpers are integer arithmetic. No training-loop hot path touched at this CP. Wall-clock budget verdict lives at CP10. |
| Scope drift | none flagged | All changed paths are inside the CP3b-scoped set: `src/algorithms/dreamer_srl/buffers.py`, `configs/dreamer_srl/agent_xs.yaml`, paired tests, six fixtures, the fixture generator, the diff-tool registries, DEVIATION_LOG, this plan, three review files, one PI doc. No out-of-scope source modifications. |

**Conclusion.** CP3b → **CP-PASS** at `9c57c06`. CP5 is the next eligible checkpoint per the v3 implementation order (slot #3 — `loss.py` two-hot symlog-space distribution, the historical-scar function whose previous bug took weeks to find because the bin grid was implemented in real reward space instead of symlog space). The user authorizes the CP3b → CP5 transition; the senior-developer does not spawn `developer` for CP5 without that authorization.

**Verified by**: senior-developer
**Date**: 2026-05-14

### CP5 — `loss.py` `TwoHotEncoding`

**Implemented by**: developer agent | **Date**: 2026-05-14

#### Files created / changed
| File | Action |
|---|---|
| `src/algorithms/dreamer_srl/loss.py` | Created — `TwoHotEncoding` class (one class, four methods: `__init__` storing `logits`/`probs`/`bins`/`dims`; `mean` and `mode` properties applying `symexp` at consumption time; `log_prob(x)` symlog-encoding the target before bin-lookup and reducing the soft cross-entropy) ported from `vendor/sheeprl/sheeprl/utils/distribution.py:L224-L276` (`TwoHotEncodingDistribution`) |
| `tests/algorithms/dreamer_srl/test_loss.py` | Created — 5 tests (3 Lever-A: `test_twohot_bins_endpoints`, `test_twohot_encode_matches_sheeprl`, `test_twohot_log_prob_target_symlog_encoded`; 2 structural: `test_bins_not_symexp_at_storage` historical-scar trap, `test_loss_module_does_not_import_from_src_models` isolation rule) |
| `tests/fixtures/dreamer_srl/twohot_bins_endpoints_input.npz` | Created (seed 0xD3EAF) |
| `tests/fixtures/dreamer_srl/twohot_encode_input.npz` | Created |
| `tests/fixtures/dreamer_srl/twohot_log_prob_input.npz` | Created |
| `scripts/fixtures/gen_cp5_fixtures.py` | Created — deterministic CP5 fixture generator (sheeprl_bridge env) |
| `scripts/sheeprl_jax_diff.py` | Updated — three CP5 runners (`_run_twohot_bins_endpoints`, `_run_twohot_encode`, `_run_twohot_log_prob`) added to FUNCTION_REGISTRY; CHECKPOINT_REGISTRY `CP5` slot populated; FUNCTION_THRESHOLDS updated with the D-006 3e-5 relaxation |
| `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` | Updated — D-006 logged + ✅ APPROVED 2026-05-14 |

#### Lever-A test results
```
pytest tests/algorithms/dreamer_srl/test_loss.py -v
5 passed (3 Lever-A bit-identity + 2 structural)

Full suite: pytest tests/algorithms/dreamer_srl/
19/19 PASS (CP1: 8 + CP3b: 6 + CP5: 5, no regressions)
```

#### Diff tool sweep — `python scripts/sheeprl_jax_diff.py --checkpoint CP5`
```
twohot_bins_endpoints   PASS  max_abs_diff=1.907e-06 (< 3.0e-05 D-006 threshold)
twohot_encode           PASS  max_abs_diff=6.080e-06 (< 3.0e-05 D-006 threshold)
twohot_log_prob         PASS  max_abs_diff=1.812e-05 (< 3.0e-05 D-006 threshold)
```

Exit code: 0.

#### Symlog-space discipline verification
```
grep "symexp(self.bins)" src/algorithms/dreamer_srl/loss.py      → empty (storage is bare linspace; symexp only at mean/mode consumption time)
grep "from src.models.dreamer_v3" src/algorithms/dreamer_srl/    → empty (isolation rule honored)
```
Triple-consistency contract (v2 plan cascade row #2, v3 plan CP5 row, `loss.py` class) all state "linspace in symlog space — NOT `symexp`'d". Historical-scar trap `test_bins_not_symexp_at_storage` would fire at 14 orders of magnitude (real-space `bins[0]` = `-4.85e8` vs symlog-space `bins[0]` = `-20.0`) if a future developer re-introduced the bug.

#### Deviation-log entries for CP5
- **D-006**: `TwoHotEncoding.__init__` + all three CP5 functions — JAX `jnp.linspace(-20, 20, 255)` produces `bins[127] = 0.0` exactly; PyTorch `torch.linspace(...)` produces `bins[127] = 7.45e-8` (1 float32 ULP). Cascades through bin-lookup + two-hot weights + log-prob reduction to `max_abs_diff = 1.812e-5` on `log_prob` (relative `2.4e-6`, < 0.25 ULP relative — platform float32 arithmetic drift, NOT semantic). Threshold relaxed to 3e-5 (same class as PI-approved D-003 for CP1 `symexp`). **PI verdict: ✅ APPROVED 2026-05-14** ([pi/calls/2026-05-14_dreamer_srl_v3_cp5_deviations.md](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp5_deviations.md)).

#### Speed check
CP5 code (`TwoHotEncoding` class constructed fresh per loss computation, methods are JAX array ops inside JIT-traced loss functions) is consumed by CP6 (critic loss) and downstream world-model loss; it does not itself contain a training-loop hot path. No isolated speed measurement possible at this CP. Wall-clock budget verdict lives at CP10 once the full training step is assembled.

#### Reviewer chain
- [x] `code-reviewer` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp5_code_review.md`](../../../reviews/dreamer_srl_v3_cp5_code_review.md) (committed `ff30e77`). Symlog-space discipline confirmed across 7 sites (storage, mean, mode, log_prob, v3 plan, v2 plan, sheeprl source). `test_bins_not_symexp_at_storage` confirmed to genuinely catch the recurrence at 14-OOM margin. All 12 inline sub-citations in `log_prob` verified line-by-line against sheeprl. One 🟢 nit (non-blocker): stale `L185-L260` citation in `scripts/sheeprl_jax_diff.py:21` example output (v2-plan era when sheeprl lived at `tmp/sheeprl/`) — folded into this CP-PASS commit as `L224-L276`.
- [x] `math-reviewer` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp5_math_review.md`](../../../reviews/dreamer_srl_v3_cp5_math_review.md) (committed `ff30e77`). All 5 equations match sheeprl `distribution.py:L224-L276` term-for-term. D-006 cascade re-derived analytically (10× amplification expected, observed within 1.5× of prediction). ±20 range confirmed against Hafner 2023 §B. Sheeprl mode == mean (both `symexp(E[bin])`), correctly followed by JAX code.
- [x] `professor-rl-bayesian-dl` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp5_professor_rl_bayesian_dl_review.md`](../../../reviews/dreamer_srl_v3_cp5_professor_rl_bayesian_dl_review.md) (committed `ff30e77`). All 8 algorithm-integration properties verified (TwoHotEncoding is a proper probability distribution; historical-scar contract intact bidirectionally; `sample()` correctly absent matching sheeprl; `Independent(TwoHotEncoding, 1)` correctly absent — `dims=1` ctor arg handles event reduction; CP6/CP7 consumption signatures match). D-006 gradient-flow analysis: forward drift invisible to optimization (depends on differences, not absolute log_prob).
- [x] PI sign-off on D-006 — ✅ APPROVED at [`docs/pi/calls/2026-05-14_dreamer_srl_v3_cp5_deviations.md`](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp5_deviations.md) (committed `b2dd5de`).

Status: **CP-PASS (2026-05-14)** — implementation `fdb09da`; all four gates closed; PI cleared deviation log; CP2/CP2b is next eligible.

---

#### Verification (senior-developer, 2026-05-14)

Plain-language summary of the CP-PASS decision. CP5 (the JAX port of sheeprl's `TwoHotEncodingDistribution` — the 255-bin grid the world-model's reward head and the critic head use to predict scalar values as a soft histogram) is the **historical-scar checkpoint**: the previous in-house DreamerV3 stored the bin grid in *real reward space* (linspace wrapped with `symexp`) while sheeprl stores it in *symlog space*, and the v1 of this project trained to a different basin because three reviewers reading a 1056-line static plan never caught the divergence. The whole v3 plan exists to mechanically prevent that bug class. CP5 has now cleared all four gates the plan requires. (a) Five tests pass — three Lever-A bit-identity tests (`twohot_bins_endpoints` `1.9e-6`, `twohot_encode` `6.1e-6`, `twohot_log_prob` `1.8e-5`, all within the D-006 3e-5 relaxed threshold; D-006 is the JAX-vs-PyTorch float32 ULP drift in `linspace` at the midpoint bin) and two structural tests (`test_bins_not_symexp_at_storage` would catch the historical-bug recurrence at 14 orders of magnitude; `test_loss_module_does_not_import_from_src_models` enforces module isolation from the legacy in-house Dreamer). (b) The source-citation discipline (`Ported from sheeprl@33b6366:<path>:<line-range>` headers, GOTCHA paragraphs, Bit-identity-test references) is present on the class and all four methods; math-reviewer line-checked all 12 inline sub-citations in `log_prob` against the vendored sheeprl file. (c) The three-reviewer chain closed cleanly — code, math, and professor-rl-bayesian-dl audits all persisted to disk at `ff30e77` (no missing-file gap, matching the CP3b clean pattern rather than the CP1 missing-write gap). (d) The PI signed off on the one deviation D-006 (substrate-mechanical platform float32 drift, same class as the PI-approved D-003 for CP1 `symexp`; gradient flow is invisible to the drift because `bins` is non-trainable and `log_prob` only enters the loss as a difference of logits). The historical-scar bug class is structurally prevented at three independent sites — the bare `jnp.linspace` at `loss.py:111` with no `symexp` wrap, the 14-OOM-margin trap test, and the seven cross-citation sites all agreeing on symlog-space storage.

| Lever / gate | Verdict | Evidence |
|---|---|---|
| Lever A — bit-identity tests | ✅ | 5/5 PASS (3 Lever-A bit-identity + 2 structural) at the D-006 relaxed threshold; diff-tool sweep at `fdb09da` (`scripts/sheeprl_jax_diff.py --checkpoint CP5` exits 0 with `1.9e-6` / `6.1e-6` / `1.8e-5`); full-suite pytest `19/19 PASS` (CP1+CP3b+CP5, no regressions) recorded in CP5 Implementation Report |
| Lever B — source citations | ✅ | math-reviewer line-checked the docstring headers and all 12 inline sub-citations against `vendor/sheeprl/sheeprl/utils/distribution.py:L224-L276` @ `33b6366`; verdict `✅ PASS` in `docs/reviews/dreamer_srl_v3_cp5_math_review.md` |
| Lever C — code-reviewer | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp5_code_review.md`](../../../reviews/dreamer_srl_v3_cp5_code_review.md) (`ff30e77`). Symlog-space discipline confirmed across 7 sites; `test_bins_not_symexp_at_storage` genuinely catches recurrence at 14-OOM margin; D-006 cascade plausibility checked. One 🟢 nit (stale `L185-L260` in `scripts/sheeprl_jax_diff.py:21` docstring example) folded into this CP-PASS commit. |
| Lever C — math-reviewer | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp5_math_review.md`](../../../reviews/dreamer_srl_v3_cp5_math_review.md) (`ff30e77`). All 5 equations match sheeprl `distribution.py:L224-L276` term-for-term; D-006 10× cascade re-derived analytically; ±20 range confirmed against Hafner 2023 §B. |
| Lever C — professor-rl-bayesian-dl | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp5_professor_rl_bayesian_dl_review.md`](../../../reviews/dreamer_srl_v3_cp5_professor_rl_bayesian_dl_review.md) (`ff30e77`). All 8 algorithm-integration properties verified; D-006 gradient-flow analysis confirms forward drift invisible to optimization. |
| Lever D — diff tool | ✅ | `scripts/sheeprl_jax_diff.py --checkpoint CP5` runs all 3 CP5 functions against `vendor/sheeprl/sheeprl/utils/distribution.py:L224-L276` @ `33b6366` and prints PASS for each at the D-006 3e-5 relaxed threshold |
| Lever E — PI sign-off | ✅ | D-006 ✅ APPROVED at [`docs/pi/calls/2026-05-14_dreamer_srl_v3_cp5_deviations.md`](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp5_deviations.md) (`b2dd5de`); DEVIATION_LOG.md table updated with approval anchor |
| Historical-scar prevention | ✅ | `grep "symexp(self.bins)"` returns empty; `bins` stored at `loss.py:111` as bare `jnp.linspace(low, high, n_bins)`; `symexp` calls only at lines 134 (mean) and 146 (mode), exactly matching sheeprl's consumption-time placement; `test_bins_not_symexp_at_storage` trap fires at 14 orders of magnitude (real-space `bins[0] = -4.85e8` vs symlog-space `bins[0] = -20.0`) — the v1 bug class is structurally prevented |
| Speed check | n/a | CP5 code is a probability-distribution helper class consumed by CP6 (critic loss) and the world-model loss; it does not itself contain a training-loop hot path. Wall-clock budget verdict lives at CP10 once the full training step is assembled. |
| Scope drift | none flagged | All changed paths in `fdb09da` + `ff30e77` + `b2dd5de` are inside the CP5-scoped set: `src/algorithms/dreamer_srl/loss.py`, paired tests, three fixtures, the fixture generator, the diff-tool registries, DEVIATION_LOG, this plan, three review files, one PI doc. No out-of-scope source modifications. The stale-docstring nit fix in `scripts/sheeprl_jax_diff.py:21` (L185-L260 → L224-L276) is folded into this CP-PASS commit. |

**Conclusion.** CP5 → **CP-PASS** at `fdb09da`. The historical-scar bug class (`symexp(linspace)` storing bins in real reward space) is structurally prevented at three independent sites, mechanically caught at 14-OOM margin if it ever recurs. CP2/CP2b is the next eligible checkpoint per the v3 implementation order (slot #4 — `agent.py` `LayerNormGRUCell` cascade fix #28 + the §S2 action-shift wiring; single developer task can cover both since they share the same file). The user authorizes the CP5 → CP2/CP2b transition; the senior-developer does not spawn `developer` for CP2/CP2b without that authorization.

**Verified by**: senior-developer
**Date**: 2026-05-14

### CP2 + CP2b — `LayerNormGRUCell` + `action_shift`

**Implemented by**: developer agent | **Date**: 2026-05-14

#### Files created / changed
| File | Action |
|---|---|
| `src/algorithms/dreamer_srl/agent.py` | Created — `LayerNormGRUCell` (nnx.Module, 1+1 fused gate, reset-before-tanh) + `action_shift` function |
| `tests/algorithms/dreamer_srl/test_agent.py` | Created — 2 Lever-A bit-identity tests (`test_layernorm_gru_cell_matches_sheeprl`, `test_action_shift_matches_sheeprl`) |
| `tests/fixtures/dreamer_srl/layernorm_gru_cell_input.npz` | Created (seed 0xD3EAF; fixture has reset ≈ 0.55 to make reset-before-tanh trap active at 336× margin) |
| `tests/fixtures/dreamer_srl/action_shift_input.npz` | Created (T=5, B=4, A=3; covers off-by-one in both directions) |
| `scripts/fixtures/gen_cp2_fixtures.py` | Created — deterministic CP2+CP2b fixture generator (sheeprl_bridge env) |
| `scripts/sheeprl_jax_diff.py` | Updated — `_run_layernorm_gru_cell` + `_run_action_shift` added to FUNCTION_REGISTRY; D-007 threshold (5e-4) added to FUNCTION_THRESHOLDS; CP2/CP2b already in CHECKPOINT_REGISTRY |
| `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` | Updated — D-007 logged (☐ pending PI sign-off at CP2 gate) |

#### Lever-A test results
```
pytest tests/algorithms/dreamer_srl/test_agent.py -v
test_layernorm_gru_cell_matches_sheeprl  PASSED  (max_abs_diff=2.975e-04 < 5.0e-04 D-007)
test_action_shift_matches_sheeprl        PASSED  (max_abs_diff=0.000e+00 exact equality)
2 passed in 4.88s
```

#### Full suite (no regressions)
```
pytest tests/algorithms/dreamer_srl/ -v
21 passed in 12.59s  (CP1: 8 + CP3b: 6 + CP5: 5 + CP2: 2 = 21, no regressions)
```

#### Diff tool sweep
```
python scripts/sheeprl_jax_diff.py --checkpoint CP2
layernorm_gru_cell   PASS  max_abs_diff=2.975e-04 (< 5.0e-04 D-007 threshold)

python scripts/sheeprl_jax_diff.py --checkpoint CP2b
action_shift         PASS  max_abs_diff=0.000e+00 (< 1.0e-06 threshold)
```

Both exit code: 0.

#### Isolation rule check
```
grep -r "from src.models.dreamer_v3" src/algorithms/dreamer_srl/
→ empty (docstring reference only — no actual import)
```

#### Deviation-log entries for CP2 + CP2b
- **D-007** (CP2 only): `LayerNormGRUCell.__call__` — JAX XLA float32 matmul accumulation order differs from PyTorch CPU for the 24-element fused projection `[hx; input] @ W.T`. Cascades through LayerNorm + gate nonlinearities. `max_abs_diff = 2.97e-4`. Float64 numpy reference gives `1.85e-7` vs PyTorch reference, confirming pure float32 accumulation order (not semantic). Reset-before-tanh trap (cascade fix #28) produces O(0.1) — 336× above this ULP drift — so the 5e-4 threshold still catches the structural trap loudly. Threshold relaxed to 5e-4. **PI verdict: ☐ pending — sign-off required at CP2 gate**.
- CP2b: no deviation. `action_shift` is pure concatenate + zeros_like — exact equality expected and confirmed.

#### Reset-before-tanh trap verification
The fixture has reset mean ≈ 0.55 (post-sigmoid) at `input_size=8, hidden_size=16, BATCH_SIZE=4`.  
A wrong-order implementation (`cand = reset * tanh(cand_proj)` instead of `cand = tanh(reset * cand_proj)`) would produce `max_abs_diff ≈ 0.15-0.30` — 336× above the 5e-4 threshold — and the test would fail loudly.  
The measured `max_abs_diff = 2.97e-4` (< 5e-4) is consistent with float32 accumulation order drift, NOT the cascade-fix-#28 trap.

#### Speed check
CP2 (`LayerNormGRUCell`, `action_shift`) are components of the world-model RSSM; they are not yet wired into a training loop at this checkpoint. No training-loop hot path is touched. Speed check skipped per protocol; wall-clock budget verdict lives at CP10 once the full training step is assembled.

#### Reviewer chain
- [x] `code-reviewer` ⚠ PASS WITH FIX → [`docs/reviews/dreamer_srl_v3_cp2_code_review.md`](../../../reviews/dreamer_srl_v3_cp2_code_review.md) (committed `ebff4d8`). Caught LayerNorm eps mismatch (JAX nnx default `1e-6` vs sheeprl production `1e-3`) — a silent-bug class that would have produced `1.72e-3` forward-pass drift and degraded CP4 RSSM parity at wire-up. Fix landed at `949f188` (eps surfaced as explicit ctor param defaulting to `1e-3`, fixture regenerated). Post-fix `max_abs_diff = 2.947e-4` (was `2.975e-4` pre-fix; D-007 5e-4 threshold still holds).
- [x] `math-reviewer` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp2_math_review.md`](../../../reviews/dreamer_srl_v3_cp2_math_review.md) (committed `ebff4d8`). All 6 governing equations match sheeprl `agent.py:L1170-L1180` line-for-line (fused projection with hidden-FIRST concat, chunk-split `(reset, cand, update)` order, reset INSIDE `tanh`, `update_proj-1` bias shift, convex-combination final update, `action_shift` dtype-preserved). Zero findings.
- [x] `professor-rl-bayesian-dl` ✅ PASS → [`docs/reviews/dreamer_srl_v3_cp2_professor_rl_bayesian_dl_review.md`](../../../reviews/dreamer_srl_v3_cp2_professor_rl_bayesian_dl_review.md) (committed `ebff4d8`). Convex-combination Lipschitz bound preserved, `update_proj-1` retain-factor math confirms Hafner convention is load-bearing (`5e-9` trainable vs `5e-20` dead without bias shift), D-007 gradient-flow analysis shows forward drift invisible to optimization (same precedent as D-006). Hand-off notes for CP4 / CP4b / CP9b included.
- [x] PI sign-off on D-007 ✅ APPROVED at [`docs/pi/calls/2026-05-14_dreamer_srl_v3_cp2_deviations.md`](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp2_deviations.md) (committed `6a8b878`). Autonomous approval under substrate-mechanical class precedent (same class as PI-approved D-003 for CP1 `symexp` and D-006 for CP5 `linspace`); unanimous reviewer concurrence; float64 numpy reference at `1.85e-7` confirms pure float32 accumulation order.

Status: **CP-PASS (2026-05-14)** — implementation `df9c328` + eps fix `949f188`; all four gates closed; PI cleared deviation log; CP3 is next eligible.

**Implemented by**: developer
**Date**: 2026-05-14

---

#### Verification (senior-developer, 2026-05-14)

Plain-language summary of the CP-PASS decision. CP2 + CP2b (the JAX port of sheeprl's `LayerNormGRUCell` — the recurrent core of the world-model's RSSM, the highest-risk silent-pattern-match item in the v3 plan — plus the §S2 `action_shift` wiring that prepends a zero-action and drops the last step so the model conditions on the *previous* action) is the fourth algorithmic checkpoint of the v3 rebuild to close. The LayerNormGRUCell is high-risk because the cascade fix #28 from the v2 plan teaches that a wrong implementation — `cand = reset * tanh(cand_proj)` instead of `cand = tanh(reset * cand_proj)` — produces an `O(0.1)` deviation that is structurally indistinguishable from "the model just learned slightly differently" without bit-identity tests. CP2 + CP2b have now cleared all four gates the v3 plan requires. (a) Two Lever-A bit-identity tests pass — `test_layernorm_gru_cell_matches_sheeprl` at `max_abs_diff = 2.947e-4` (within the D-007 5e-4 relaxed threshold, with the reset-before-tanh trap active at 336× margin above the float32 ULP drift so the cascade-fix-#28 bug would still fire loudly), and `test_action_shift_matches_sheeprl` at `max_abs_diff = 0.000e+00` (exact equality, pure concat + zeros_like). Full suite `21/21 PASS` (CP1+CP3b+CP5+CP2, no regressions). (b) The source-citation discipline (`Ported from sheeprl@33b6366:<path>:<line-range>` headers + GOTCHA paragraphs + Bit-identity-test references) is present on `LayerNormGRUCell` and `action_shift`, and the math-reviewer line-checked all 6 governing equations against `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1170-L1180` @ `33b6366`. (c) The three-reviewer chain closed cleanly — code, math, and professor-rl-bayesian-dl audits all persisted to disk at `ebff4d8`, and crucially the code-reviewer caught a real silent-bug: the LayerNorm `eps` defaulted to nnx's `1e-6` while sheeprl production uses `1e-3`, a `1000×` mismatch that produced `1.72e-3` forward-pass drift and would have silently degraded CP4 RSSM parity at wire-up. The fix landed at `949f188` (eps as explicit ctor param defaulting to `1e-3`, fixture regenerated, threshold still cleared) — **this is the guardrail discipline working as designed**: the cascade fix #28 trap caught a structural bug, then the parallel review chain caught a substrate-level bug, and both were closed before CP-PASS rather than detonating during downstream wire-up. (d) The PI signed off autonomously on D-007 under the substrate-mechanical class precedent (same class as the PI-approved D-003 for CP1 `symexp` and D-006 for CP5 `linspace` — platform float32 ULP drift on a fused-projection matmul, float64 numpy reference at `1.85e-7` confirms hardware accumulation order, not semantic). Gradient flow is invisible to the drift because the forward-pass deviation cascades through tanh + LayerNorm without affecting backward-pass gradient direction.

| Lever / gate | Verdict | Evidence |
|---|---|---|
| Lever A — bit-identity tests | ✅ | 2/2 PASS at the D-007 relaxed threshold post-eps-fix; `test_layernorm_gru_cell_matches_sheeprl` `2.947e-4` (< 5e-4 D-007); `test_action_shift_matches_sheeprl` `0.000e+00` exact; diff-tool sweep at `df9c328 + 949f188` (`scripts/sheeprl_jax_diff.py --checkpoint CP2` + `--checkpoint CP2b` both exit 0); full-suite pytest `21/21 PASS` (CP1+CP3b+CP5+CP2, no regressions) recorded in CP2 + CP2b Implementation Report |
| Lever B — source citations | ✅ | math-reviewer line-checked docstring headers and inline citations against `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1170-L1180` @ `33b6366`; verdict `✅ PASS` (0 findings) in `docs/reviews/dreamer_srl_v3_cp2_math_review.md` |
| Lever C — code-reviewer | ⚠ PASS WITH FIX → ✅ closed | [`docs/reviews/dreamer_srl_v3_cp2_code_review.md`](../../../reviews/dreamer_srl_v3_cp2_code_review.md) (`ebff4d8`). Caught LayerNorm eps mismatch (nnx default `1e-6` vs sheeprl production `1e-3`); fix landed at `949f188` before CP-PASS. Silent-bug class — would have degraded CP4 RSSM parity at wire-up. **Concrete evidence the guardrail discipline works**: parallel review chain caught a substrate-level bug that the bit-identity test on CP2 alone would not have flagged structurally (the eps drift sits below the reset-before-tanh trap signal at this fixture size). |
| Lever C — math-reviewer | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp2_math_review.md`](../../../reviews/dreamer_srl_v3_cp2_math_review.md) (`ebff4d8`). All 6 governing equations match sheeprl line-for-line: fused projection with hidden-FIRST concat (not input-first), chunk-split `(reset, cand, update)` order (not the more common `(reset, update, cand)`), reset INSIDE `tanh` (not outside), `update_proj-1` bias shift, convex-combination final update, `action_shift` dtype-preserved. 0 findings. |
| Lever C — professor-rl-bayesian-dl | ✅ PASS | [`docs/reviews/dreamer_srl_v3_cp2_professor_rl_bayesian_dl_review.md`](../../../reviews/dreamer_srl_v3_cp2_professor_rl_bayesian_dl_review.md) (`ebff4d8`). Convex-combination Lipschitz bound preserved; `update_proj-1` retain-factor math (`5e-9` trainable vs `5e-20` dead without bias shift) confirms Hafner convention is load-bearing, not stylistic. D-007 gradient-flow analysis: forward drift invisible to optimization (cascade through tanh + LayerNorm does not change backward-pass gradient direction). Hand-off notes for CP4 / CP4b / CP9b downstream reviewers included. |
| Lever D — diff tool | ✅ | `scripts/sheeprl_jax_diff.py --checkpoint CP2` runs `layernorm_gru_cell` against `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1170-L1180` @ `33b6366` and prints PASS at `2.947e-4` (< 5e-4 D-007); `--checkpoint CP2b` runs `action_shift` and prints PASS at `0.000e+00` (< 1e-6) |
| Lever E — PI sign-off | ✅ | D-007 ✅ APPROVED at [`docs/pi/calls/2026-05-14_dreamer_srl_v3_cp2_deviations.md`](../../../pi/calls/2026-05-14_dreamer_srl_v3_cp2_deviations.md) (`6a8b878`); autonomous approval under substrate-mechanical class precedent (same class as PI-approved D-003 for CP1 `symexp` + D-006 for CP5 `linspace`); unanimous reviewer concurrence to approve; DEVIATION_LOG.md table updated with approval anchor |
| Reset-before-tanh trap | ✅ | Fixture seeded with reset mean ≈ 0.55 post-sigmoid at `input_size=8, hidden_size=16, BATCH_SIZE=4`. A wrong-order implementation (`cand = reset * tanh(cand_proj)` instead of `cand = tanh(reset * cand_proj)`) would produce `max_abs_diff ≈ 0.15-0.30` — 336× above the 5e-4 threshold — and the test would fail loudly. The measured `2.947e-4` is consistent with float32 accumulation order drift, NOT the cascade-fix-#28 trap. Cascade fix #28 is structurally caught. |
| LayerNorm eps catch | ✅ | **Validates the four-gate guardrail design**: code-reviewer flagged the `1e-6` vs `1e-3` mismatch (`1.72e-3` forward-pass drift, `1000×` parameter mismatch); fix landed at `949f188` before CP-PASS. Without the parallel review chain the bug would have lain dormant until CP4 RSSM wire-up, where the drift would have either (a) been silently absorbed into "model just learned differently" — the exact bug class the v3 plan exists to prevent — or (b) detonated noisily under a much more entangled symptom surface. Cost-of-late-catch ratio: bug caught at the leaf, not at the cascade. |
| Speed check | n/a | CP2 (`LayerNormGRUCell`, `action_shift`) are RSSM components not yet wired into a training loop at this checkpoint. No training-loop hot path touched. Wall-clock budget verdict lives at CP10 once the full training step is assembled. |
| Scope drift | none flagged | All changed paths in `df9c328 + 949f188 + ebff4d8 + 6a8b878` are inside the CP2 + CP2b-scoped set: `src/algorithms/dreamer_srl/agent.py`, paired tests, two fixtures, the fixture generator, the diff-tool registries, DEVIATION_LOG, this plan, three review files, one PI doc, one diary row. No out-of-scope source modifications. |

**Conclusion.** CP2 + CP2b → **CP-PASS** at `df9c328` + `949f188`. The cascade-fix-#28 trap (reset-before-tanh wrong-order bug) is structurally caught at 336× margin; the LayerNorm eps silent-bug was caught by the parallel review chain before CP-PASS rather than at CP4 wire-up — concrete evidence that the four-gate guardrail design works as intended. CP3 is the next eligible checkpoint per the v3 implementation order (slot #5 — `build_agent` final init phase applies `uniform_init_weights(scale=0.0)` to the reward-head's output linear AND the critic-head's output linear; both become zero-init at construction per sheeprl `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1170-L1180` per the v2 plan cascade item #27). The user authorizes the CP2 + CP2b → CP3 transition; the senior-developer does not spawn `developer` for CP3 without that authorization.

**Verified by**: senior-developer
**Date**: 2026-05-14

### CP-by-CP speed-change measurements
Per the senior-developer agent profile's speed-change review protocol, record
the before/after speed numbers on the same hardware/config/seed for any change
that could plausibly affect runtime. ≤ 5% slowdown is no-regression, > 5%
warrants discussion, > 15% blocks merge unless the plan explicitly accepts it.

---

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [date]

| CP | Files | Lever-A tests | Reviewer chain | Deviation-log entries | Status | Notes |
|---|---|:---:|:---:|---|:---:|---|
| CP1 | `utils.py` | | | | | |
| CP2 | `agent.py` (`LayerNormGRUCell`) | | | | | |
| CP2b | `train.py` (action shift) | | | | | |
| CP3 | `agent.py` (`build_agent`) | | | | | |
| CP4 | `agent.py` (RSSM) | | | | | |
| CP4b | `agent.py` (RSSM `is_first` reset) | | | | | |
| CP5 | `loss.py` (`TwoHotEncoding`) | | | | | |
| CP6 | `train.py` (critic loss) | | | | | |
| CP7 | `train.py` (Polyak) | | | | | |
| CP8 | offline forward parity | | | | | |
| CP9 | dry-run integration smoke | n/a | optional | | | |
| CP9b | prefill behavior | | | | | |
| CP10 | wall-clock budget | n/a | n/a | | | |

**Conclusion**: [one-line summary]

**Parity-gate launch verdict**: [populated post-launch by `experiment-analyzer`,
linked from this row]

---

## Implementation Report — CP3 (developer)

**Date**: 2026-05-14
**Implemented by**: developer

### Summary

CP3 ports the `build_agent` final-init phase for the reward model and critic output
linear layers from `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1170-L1180`.
With `hafner_initialization=True`, sheeprl applies `uniform_init_weights(scale=0.0)`
to both heads, forcing all-zeros kernel + bias at construction (cascade fix #27).
This is a small, self-contained checkpoint — no deviations encountered.

### File-by-file changes

| File | Action | Notes |
|---|---|---|
| `src/algorithms/dreamer_srl/agent.py` | Extended | Added `RewardHead` + `CriticHead` classes; added `from src.algorithms.dreamer_srl.utils import uniform_init_weights` import; updated module docstring Contents section |
| `tests/algorithms/dreamer_srl/test_agent.py` | Extended | Added `test_zero_init_reward_head_matches_sheeprl` + `test_zero_init_critic_head_matches_sheeprl`; updated module docstring; extended import to include `RewardHead, CriticHead` |
| `scripts/sheeprl_jax_diff.py` | Extended | Added `_run_zero_init_reward_head` + `_run_zero_init_critic_head` runners; registered both in `FUNCTION_REGISTRY` |
| `scripts/fixtures/gen_cp3_fixtures.py` | Created | New fixture generator (sheeprl_bridge env); generates `zero_init_{reward,critic}_head_input.npz` |
| `tests/fixtures/dreamer_srl/zero_init_reward_head_input.npz` | Created | Fixture: in=512, out=255, torch_kernel all-zeros, torch_bias all-zeros |
| `tests/fixtures/dreamer_srl/zero_init_critic_head_input.npz` | Created | Fixture: in=512, out=255, torch_kernel all-zeros, torch_bias all-zeros |
| `docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md` | Extended | Appended Implementation Report; updated `last_updated` frontmatter |

All changed paths are in the CP3-scoped set. No out-of-scope source modifications.

### Test results

**Verification step 1 — diff tool:**

```
python scripts/sheeprl_jax_diff.py --checkpoint CP3
```
- `zero_init_reward_head`: max_abs_diff = 0.000e+00  PASS (< 1e-6)
- `zero_init_critic_head`: max_abs_diff = 0.000e+00  PASS (< 1e-6)
- Exit code: 0

**Verification step 2 — test_agent.py (4/4):**

```
pytest tests/algorithms/dreamer_srl/test_agent.py -v
```
- `test_layernorm_gru_cell_matches_sheeprl`    PASS (< 5e-4 D-007)
- `test_action_shift_matches_sheeprl`          PASS (0.000e+00)
- `test_zero_init_reward_head_matches_sheeprl` PASS (0.000e+00)
- `test_zero_init_critic_head_matches_sheeprl` PASS (0.000e+00)
- **4/4 PASS** in 6.27 s

**Verification step 3 — full dreamer_srl suite (23/23):**

```
pytest tests/algorithms/dreamer_srl/ -v
```
- **23/23 PASS** in 13.12 s — no regressions

**Verification step 4 — isolation check:**

```
grep -r "from src.models.dreamer_v3" src/algorithms/dreamer_srl/
```
Returns nothing (docstring mention only — no actual import).

### Speed check

CP3 adds module classes (`RewardHead`, `CriticHead`) with no training-loop hot-path
changes. These are instantiated once at build time; no vmap/jit/scan boundary is
modified. Speed check skipped per protocol (provably cannot affect runtime at this stage).

### Deviations

None. Zero is zero on both platforms — max_abs_diff = 0.000e+00, no D-### log entry required.
The CP3 risk profile was correctly assessed as "low" in the task spec.

### Commit

`21e7f50` — `feat(dreamer-srl): ✨ CP3 — zero-init reward + critic heads (cascade fix #27)`
Working tree clean after commit. Streak: 5 clean CP-PASS flips + 5 clean implementation commits.

### Checkpoints completed

- [x] CP3 Lever-A tests: 2/2 PASS at strict 1e-6 threshold (max_abs_diff = 0.0 exact)
- [x] Diff tool exit 0 for `--checkpoint CP3`
- [x] Full suite 23/23 PASS (no regressions)
- [x] Isolation check: no `src.models.dreamer_v3` imports in dreamer_srl module
- [x] Lever-B citation headers on both `RewardHead` and `CriticHead` constructors: `# Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1170-L1180`
- [x] CP1's `uniform_init_weights` used — no reinvention of zero-init

**Implemented by**: developer
