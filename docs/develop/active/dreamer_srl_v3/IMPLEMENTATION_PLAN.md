---
title: "dreamer-srl v3 — JAX rebuild of sheeprl DreamerV3 with deviation-prevention guardrails"
topic: dreamer
status: active
created: 2026-05-13
last_updated: 2026-05-13
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
| **CP1** | `utils.py` forward parity — [v2 Checkpoint 1](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_symlog_symexp_roundtrip`, `test_init_weights_matches_sheeprl`, `test_uniform_init_weights_matches_sheeprl`, `test_compute_lambda_values_matches_sheeprl`, `test_moments_update_matches_sheeprl`, `test_ratio_matches_sheeprl`, `test_prepare_obs_shape_contract` | code → math → professor | D-001, D-002, D-003 | IN PROGRESS — awaiting reviewer gate |
| **CP2** | `agent.py` `LayerNormGRUCell` cascade fix #28 — [v2 Checkpoint 2](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_layernorm_gru_cell_matches_sheeprl` (1+1 fused-gate form; chunk order `(reset, cand, update)`; reset gate inside `tanh`) | code → math → professor | ☐ none | NOT STARTED |
| **CP2b** | Action-shift §S2 test | `test_action_shift_matches_sheeprl` (prepend-zero, drop-last; `[0] == 0`, `[1:] == actions[:-1]`) | code → math → professor | ☐ none | NOT STARTED |
| **CP3** | `agent.py` `build_agent` cascade fix #27 — [v2 Checkpoint 3](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_zero_init_reward_head`, `test_zero_init_critic_head` (kernel + bias both exactly zero) | code → math → professor | ☐ none | NOT STARTED |
| **CP4** | `agent.py` RSSM cascade fix #30 + `get_initial_states` mode-not-sample — [v2 Checkpoint 4](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_rssm_transition_2layer_mlp`, `test_rssm_representation_2layer_mlp`, `test_get_initial_states_no_prng`, `test_get_initial_states_matches_sheeprl_mode` | code → math → professor | ☐ none | NOT STARTED |
| **CP4b** | RSSM `is_first` reset §S1+§S4 — [v2 Checkpoint 4b](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_is_first_force_set_step0`, `test_is_first_three_quantity_reset` (arithmetic-mask form, posterior reshape-flatten BEFORE masking) | code → math → professor | ☐ none | NOT STARTED |
| **CP5** | `loss.py` two-hot distribution cascade fix #2 (symlog space) — [v2 Checkpoint 5](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md#checkpoints-verification-checks-during-implementation) | `test_twohot_bins_endpoints` (`bins[0]=-20, bins[127]=0, bins[254]=+20`, in symlog space), `test_twohot_encode_matches_sheeprl`, `test_twohot_log_prob_target_symlog_encoded` | code → math → professor | ☐ none | NOT STARTED |
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

1. **Execution order ≠ CP-id order.** §"Implementation order (revised)" moves CP5 to slot #2 (built right after CP1) without renumbering. The label `CP5` stays attached to two-hot bins; only the position in the build queue changes.
2. **`buffers.py` (SequentialReplayBuffer) has no CP-id.** Per §"Implementation order" step 3 it is an inter-CP sanity round-trip with no Lever-A gate, no reviewer chain, and no row in this table. It must not be labelled `CP3` (or any other CP) in downstream artifacts.

**Every downstream artifact** — `scripts/sheeprl_jax_diff.py:CHECKPOINT_REGISTRY`, `tests/algorithms/dreamer_srl/README.md` file-layout comments, the per-CP review filenames `review_{code,math,professor_rl_bayesian_dl}_CP<N>.md`, deviation-log IDs cross-referencing CPs, fixture filenames — uses **this table's CP-id**, not the implementation-order slot. If a checkpoint is later split, merged, or removed, the affected row's CP-id is retired; the others do **not** renumber.

When unsure: search for the function name in this table's "Scope" column; the row's CP-id is canonical.

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
| CP2 + CP2b (LayerNormGRUCell + action shift) | 1 day | 2–3 days | 2 Lever-A tests + fixtures; 3-reviewer gate (LayerNormGRU is the highest-risk silent-pattern-match item; review will be slow) |
| CP3 (build_agent zero-init heads) | 0.5 day | 1 day | 2 Lever-A tests; quick gate |
| CP4 + CP4b (RSSM hidden layers + is_first reset) | 2 days | 4–5 days | 4 Lever-A tests + fixtures; 3-reviewer gate; `is_first` three-quantity reset is the second-highest-risk item |
| CP5 (two-hot symlog-space bins) | 1 day | 2 days | 3 Lever-A tests + fixtures; 3-reviewer gate (the historical-bug case, so the review is detailed) |
| CP6 (critic two-term loss) | 1 day | 2 days | 3 Lever-A tests + fixtures; 3-reviewer gate |
| CP7 (Polyak update) | 0.5 day | 1 day | 3 Lever-A tests; quick gate |
| CP8 (end-to-end forward parity) | 2 days | 3–4 days | All previous tests rerun + offline forward-pass check; 3-reviewer gate is the merge-gate |
| CP9 / CP9b / CP10 (integration smokes + speed check) | 1–2 days | 2 days | Mostly unchanged — no new Lever-A overhead |
| Parity-gate launch (3 seeds) | 1 week wall-clock | 1 week wall-clock | Unchanged — only the launch waits |
| **Total** | **~3 weeks dev + 1 week run = 4 weeks** | **~5 weeks dev + 1 week run = 6 weeks** | ~1.5× overhead from guardrails |

**Caveat to the user.** The 1.5× overhead is the rough estimate, NOT 2×. If
the bit-identity tests find more deviations than expected (more than ~2 per CP
on average, i.e. > ~16 total), the deviation-log + PI-sign-off cycle adds more
time. If they find fewer (i.e. the v2 plan was as precise as the v2 reviewers
claimed), the overhead is closer to 1.2×. The user explicitly chose maximum
safety net, so this overhead is the accepted cost of preventing another
weeks-long twohot-bug-style debugging session.

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
   complete and be committed before CP1 begins.**
1. **CP1 — `utils.py`** → Lever-A tests → 3-reviewer gate → CP-PASS.
2. **CP5 — `loss.py` two-hot distribution** (moved earlier; the two-hot bug is
   the historical scar — implement it second so the symlog-space discipline is
   set early and visible) → Lever-A tests → 3-reviewer gate → CP-PASS.
3. **(buffers.py + sanity round-trip — no Lever-A gate; integration smoke only)**.
4. **CP2 + CP2b** → tests → gate → CP-PASS.
5. **CP3** → tests → gate → CP-PASS.
6. **CP4 + CP4b** → tests → gate → CP-PASS.
7. **CP6** → tests → gate → CP-PASS.
8. **CP7** → tests → gate → CP-PASS.
9. **CP8 — end-to-end forward parity** → all previous tests rerun + offline
   forward-pass check → merge-gate review → CP-PASS.
10. **CP9 + CP9b** → integration smoke → CP-PASS.
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
- [ ] `code-reviewer` ✅ PASS → `review_code_CP1.md`
- [ ] `math-reviewer` ✅ PASS → `review_math_CP1.md`
- [ ] `professor-rl-bayesian-dl` ✅ PASS → `review_professor_rl_bayesian_dl_CP1.md`
- [ ] PI sign-off on D-001, D-002, D-003 (via senior-developer after reviewers close)

Status: **IN PROGRESS — implementation complete; awaiting 3-reviewer gate**

### CP2 — `LayerNormGRUCell`
... (one block per CP; filled by developer)

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
