---
title: "Adversarial review round 2 — dreamer_srl → train.py integration plan, rev 2"
topic: dreamer
status: active
reviewer: fresh-eyes adversarial reviewer (round 2)
created: 2026-07-25
last_updated: 2026-07-25
audited_doc: "docs/develop/active/dreamer/DREAMER_SRL_TRAIN_PY_INTEGRATION_PLAN.md (rev 2) + DREAMER_SRL_TRAIN_PY_DIVERGENCE_MATRIX.md (rev 2), verified against working tree v3.0 @ ed8b218 (train.py 2,472 lines @ fff6055; dreamer_srl_main.py 2,112 lines @ 39f851b)"
---

# Round-2 review: rev 2 of the dreamer_srl → train.py integration plan

## Verdict (plain language)

**APPROVED-WITH-NITS.** Rev 2 of the plan genuinely resolves all 12 findings from the
first adversarial review — not just in prose: every code-level claim the revision now
makes was re-checked against the files it cites, and they all hold (the two hardcoded
`"DreamerV3"` WandB-payload lines really are at `dreamer_srl_main.py:829` and `:865` and
really do need the spec-driven substitution; the Gate 1 harness's A-side flag list now
contains only flags the legacy argparse actually has, with the checkpoint cadence coming
from a fixture YAML key the loop demonstrably reads; the shim's `--episodes M
--total-steps N` rewrite reproduces the legacy resolver's semantics at `:584-586`
including the both-aliases precedence; and so on — full closure table below). The live
repo state still matches the plan's baseline: **no commit since `fff6055` touches
`train.py`, `dreamer_srl_main.py`, `configs/`, or the dreamer tests** (only the three
docs commits `e8edd10`/`f6d7a08`/`ed8b218`), so every line number the plan pins to
`fff6055` is still the line number on disk.

The fresh hunt over the machinery the revision itself added (Gate 1b's uniform-mask
rule, the three sanctioned seam substitutions, the deep-copy/override-propagation
mandate, the shim's `sys.argv`-before-`import train` ordering, the Gate 1 fixture)
found **no blocker and no major**: five minor findings, all of the
tighten-the-wording / close-a-small-gate-hole class. None of them lets a silent
training change through undetected — the worst case for each is a
fail-then-iterate loop during implementation, not a wrong run. The plan is fit to hand
to `developer` with the five nits folded in (or noted for the implementer).

Binding-decision compliance: thin dispatch ✅ (seam edits enumerated and verified
minimal) · unified CLI/config with every delta enumerated ✅ (C1–C19 spot-verified at
line level) · working deprecation shim ✅ (combo-flag semantics verified; `import
train` resolves because `dreamer_srl_main.py:39` puts the repo root on `sys.path`) ·
three gates ✅ (1a + 1b + 2 + flip-after-3) · feature scope ✅ (resume/NMN/batched
stats stay out — C13/C17/C18).

---

## Closure table — the 12 round-1 findings

| # | Round-1 finding | Status | Evidence (code-level) |
|---|---|---|---|
| 1 | BLOCKER: curriculum never gate-exercised; rPPO stage-0 pollution unenumerated | **RESOLVED** | Gate 1b added (2-stage fixture, per-stage `resolved.json`, swap-crossing telemetry); pollution enumerated in compat C19 + risk R11 + matrix 2.3. Code re-verified: aliasing at `train.py:563`, agent merge `:588`, CLI-override block `:591-604`, stage dumps `dreamer_srl_main.py:936-951`, from-scratch builder `:102-134` vs clone-builder `train.py:185-201`. Residual mechanism ambiguity → new finding N1 (MINOR). |
| 2 | MAJOR: C7 contradicted by hardcoded `"DreamerV3"` at :829/:865 | **RESOLVED** | File Change 1 now enumerates both edits as spec-driven (`spec.wandb_algorithm_label`); verified `:829` is the payload hardcode and `:865`'s `setdefault("agent", {})["algorithm"] = "DreamerV3"` unconditionally overwrites the agent dict spread at `:861` — the plan's characterization is exact. Legacy caller passes `"DreamerV3"` → byte-identical payload pre-flip. C7 + matrix 4.3 rewritten to match. |
| 3 | MAJOR: Gate 1 harness passes flags the legacy argparse lacks | **RESOLVED** | A-side list is now `--env-config/--agent-config/--episodes/--num-envs/--seed/--no-wandb/--quiet/--parity-dump` — all exist in argparse `:429-491` except `--parity-dump`, which is the single enumerated additive flag. Cadence moved to the fixture YAML (`training.checkpoint_frequency: 20`), which flows to the legacy side correctly: env config merges after `train/default.yaml`+`train/dreamer_srl.yaml` (`:542-553`, overriding their `10000`) and the loop reads exactly `env_cfg training.checkpoint_frequency` at `:632`. "Byte-for-byte argparse" wording retired. |
| 4 | MAJOR: not reconciled with landed `fff6055` | **RESOLVED** | `--eval-config` verified at `train.py:415` (merged via `load_env_config` at `:519-526`, missing path → ValueError at `:522`); gate key `experiment.during_training.enabled` verified at `:668` with the non-rPPO raise at `:669-673`. Remaining `--experiment-eval` mentions in both docs are explanatory negations only ("no such flag exists"). C12 enumerates `--eval-config` pass-through; matrix rows 1.16/5.5 reflect landed state. |
| 5 | MAJOR: shim dies on `--episodes M --total-steps N` | **RESOLVED** | New rule (always rewrite `--total-steps N` → `--total-timesteps N`; inject `--episodes 0` only when `--episodes` absent) reproduces `dreamer_srl_main.py:584-586` (`total_timesteps = env_step_override or product`) exactly, and the both-aliases precedence (`--total-timesteps` wins) matches `:578` (`args.total_timesteps or args.total_steps`). Shim translation test incl. combo case named in Phase 4. |
| 6 | MINOR: Gate 1 mask omits written-back `seed` | **RESOLVED** | Pass-criteria mask now lists "top-level `seed` written-back value" with the harness's `--seed 7` called out explicitly. |
| 7 | MINOR: pre-seam locals + imprecise delete range | **RESOLVED** | `_os` (`:498`) and `_project_root` (`:499`) named for re-derivation — verified consumed below the seam at `:908-950`. Delete range restated as line 567 + lines 578–594 with 577 retained — verified `env_max_steps` is used below the seam only at `:1323` (banner), and the `total_steps` alias defined at `:594` has **zero** consumers below the seam (its "kept for the final-log print" comment is stale), so the deletion is safe. |
| 8 | MINOR: `log_code` promised but not implemented | **RESOLVED** | `spec.wandb_log_code` field + call site inside the seam after `wandb.init` (File Change 1), semantics copied from `train.py:957` (`include_fn` = `.py`); legacy caller passes `False` → no behavior change pre-flip. C8 + matrix 4.6 updated. |
| 9 | MINOR: two more legacy-driver consumers + missing preserved import | **RESOLVED** | All three handled with named decisions: `test_eval_telemetry_wandb.py` (drives `main()` at `:102-104` → retarget at `run_dreamer_training`); `test_eval_video_smoke.py` (verified `_DRIVER` at `:26`, legacy flags `--env-config`/`--total-steps` at `:78-80`/`:165-167` → kept as post-flip shim-integration test, timeout re-check named); `bench_sps.py` (verified `TRAINER_SCRIPT` pin at `:57` → retargeted at train.py form). `_build_continual_schedule` in the preserved-import list (verified imported at `test_continual_schedule.py:38`). |
| 10 | MINOR: shim env-var timing unexamined | **RESOLVED** | `sys.argv`-before-`import train` mandated in File Change 7; verified `train.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `--device` platform vars at module import time (`:40-56` incl. the pre-parser that reads `sys.argv`); Phase 4 CPU smoke asserts the env var post-import, plus a named 2-min GPU shim smoke. `import train` resolves from the shim because `dreamer_srl_main.py:39` inserts the repo root into `sys.path`. |
| 11 | MINOR: absolute→relative results-dir delta unenumerated | **RESOLVED** | C9 now enumerates path form (legacy absolute under hardcoded root at `:913` — verified — vs relative `train.py:843` — verified) with the repo-root-CWD launch-convention justification, plus the no-more-`tmp/` diversion. |
| 12 | MINOR: Gate-3 flip target thinner than described | **RESOLVED** | §9 rewritten; re-verified: `grep -rln "dreamer_srl_main" --include="*.sh"` returns nothing repo-wide. Deliverables now correctly scoped to the command template + module docstring (`:10-18`, verified that is the usage block) + docs grep; `train_command-agent.sh` add-only rule noted (it is currently user-modified in the working tree). |

**Closure: 12/12 resolved.** None is prose-only; each is backed by matching code.

---

## New findings (fresh hunt on rev-2 machinery)

### MINOR

**N1. The stage-0 anti-pollution *mechanism* is unstated — the pollution happens in
shared pre-dispatch code, and the enumerated fix as literally written does not stop
it.** The plan mandates the outcome ("dreamer branch never aliases
`schedule.stage_configs[0]`", verification checklist) and Gate 1b enforces it. But the
polluting code is not on the dreamer branch: `config = schedule.stage_configs[0]`
(train.py:563), the agent merge (:588), and the CLI-override block (:591-604) all run
**before** the dispatch point (~:690) for every algorithm. File Change 2's instruction
— "deep-copy the merged config right before the agent-config merge (line ~588) into
`dreamer_env_cfg`" — protects `dreamer_env_cfg` but leaves the subsequent shared merges
still mutating the aliased `stage_configs[0]`, which the seam then dumps as
`stage_00_*.yaml` (dreamer_srl_main.py:936-951) with `agent.*` keys stages 1..N lack —
exactly the unmasked, non-uniform diff Gate 1b fails on. Not silent (the gate catches
it deterministically), but the plan should name the concrete edit: when the peeked
algorithm is `dreamer_srl`, line 563 itself becomes a deep copy (`config =
copy.deepcopy(schedule.stage_configs[0])`), dreamer-conditional so the rPPO path's
declared-out-of-scope aliasing is untouched.

**N2. Extending the override-propagation loop (train.py:555-560) is a shared-code
change — unconditional extension alters rPPO curriculum stage dumps.** File Change 2
says to propagate seed/body/checkpoint-frequency to every stage by "extending the
existing `--no-satiation`/`--no-overeating-death` propagation loop rather than adding a
second mechanism". That loop runs for **all** algorithms in continual mode; extended
unconditionally, an rPPO curriculum run passing `--seed`/`--checkpoint-frequency` would
gain written-back keys in its `stage_XX_*.yaml` dumps — an unenumerated rPPO-side
delta, and Checkpoint 2's "rPPO smoke unchanged" is single-config so it would not
catch it. Gate the extension on the dreamer peek (or explicitly enumerate the rPPO
dump-content change as accepted).

**N3. Gate 1's mask should be conditional per key, and the parity dump should carry a
checkpoint-cadence signal.** Answering the uniform-mask stress question directly: a
pollution bug confined to masked keys that hits **all** stages identically does slip
the uniformity rule — but every such key (`wandb.*`, `tag`, dump-only `seed`/
`episodes`) is loop-inert (seeding comes from `spec.seed` at :647-648; cadence in
curriculum mode from `schedule.checkpoint_frequencies`), so nothing training-relevant
hides there, **except** `training.checkpoint_frequency` in the single-config leg: the
loop reads it (dreamer_srl_main.py:632), the harness never passes the CLI flag (cadence
comes from the fixture YAML), yet the mask lists the key unconditionally — and the
telemetry (`iter_num`, `policy_step`, episodes, grad-steps, buffer fill, losses,
PRNG/param checksums) contains **no checkpoint-save events**, so a resolution bug in
exactly that key would evade both criteria. Two one-line hardenings: (a) mask
written-back keys only when the corresponding CLI flag was actually passed in that
harness invocation (the text already does this for `body.*` — apply the same
conditionality to `seed`/`episodes`/`training.checkpoint_frequency`); (b) add
`last_ckpt_episode` (final value, or the list of checkpoint-firing iterations) to
`final.json`.

**N4. Stale line cite for the signal-handler move.** File Change 2 says the
`signal.signal(SIGINT/SIGTERM, ...)` registration is at "lines 437–438"; at `fff6055`
it is **train.py:448-449** (the 437-438 cite is pre-`fff6055` residue, contradicting
the plan's own "line numbers reference `fff6055`" pledge). The code is unambiguous;
fix the number so the byte-diff verification checklist doesn't send the
senior-developer to the wrong lines.

**N5. The `args.seed` substitution site-count is wrong (says 3, is 6).** Below the
seam, `args.seed` appears at :647 (`np.random.seed`), :648 (`PRNGKey`), :1726 and
:1769 (the two eval passes' `seed=`) — already 4 — plus the WandB payload `"seed":
args.seed` (:841) and the console banner (:1334). All are covered in spirit by the
mechanical `args.X → spec.X` sweep, but the plan's precise-enumeration posture (and
the verification checklist's "only the enumerated mechanical substitutions" byte-diff
gate) means the count should be exhaustive or restated as "every remaining `args.*`
reference below the seam becomes `spec.*` — verified complete by grep".

### Observations (no action required)

- **C5 shim residual**: a legacy launch that passes neither `--episodes` nor
  `--total-steps` and relies on a custom `training.episodes` in its env config resolves
  through the shim to the top-level `episodes` key instead. Both defaults are 100
  (verified: `configs/train/default.yaml:27` and `:63`), and the only config in the
  repo setting a non-default `training.episodes` is one archived hypervigilance
  testbed — C5 enumerates the delta and real runs pass `--episodes`; acceptable as-is.
- The stale "kept for the final-log print" comment on `dreamer_srl_main.py:593-594`
  (`total_steps` alias) should be dropped along with the lines — it currently claims a
  consumer that does not exist.
- Via train.py, `--total-timesteps` combined with `--configs-dir` is not rejected the
  way the legacy guard (:517-522) rejects it; the plan's resolution formula would
  silently honor it as an env-step cap. Edge case worth a loud-fail line in
  `_run_dreamer_srl` step 2, at the implementer's discretion.

---

## Verified correct (fresh spot-checks beyond the closure table)

- **Repo-state consistency**: `fff6055..HEAD` = three docs-only commits; no drift in
  `train.py`, `dreamer_srl_main.py`, `configs/`, or `tests/algorithms/dreamer_srl/`.
- **Seam substitution completeness**: the full `args.*` inventory below line 557
  (~30 references) maps onto the plan's enumerated families with no orphan —
  `seed/quiet/debug/buffer_device/legacy_grad_loop/log_interval/no_wandb/
  results_dir/wandb_project/wandb_name` + provenance (:832-835) + the deleted budget
  block. Nothing consumed below the seam is missing from the spec (given N5's
  restatement).
- **Legacy path byte-equivalence of the three sanctioned substitutions**: with the
  legacy spec (`wandb_algorithm_label="DreamerV3"`, `wandb_log_code=False`,
  `results_dir=None`), :829/:865 produce today's payload verbatim, the log_code call
  is skipped, and the `wandb_kwargs` None-filter is behavior-identical to today's call
  (today passes `name=None` explicitly; filtered-out `None` and explicit `None` both
  yield a WandB auto-name, and :913 consumes `wandb.run.name` either way).
- **Gate 1 fixture cadence flow**: `training.checkpoint_frequency: 20` in the fixture
  env YAML overrides the `10000` in both `train/default.yaml:42` and
  `train/dreamer_srl.yaml:37` on **both** sides (legacy merge order :542-553; train.py
  peek-merge parallel to :507), landing on the exact key the loop reads (:632) and the
  checkpoint block consumes (`checkpoint_frequency_active`, ~:1686).
- **Gate 1b A-side viability**: the legacy mutual-exclusion guard (:517-522) permits
  the planned A-side invocation (`--configs-dir` + `--continual-schedule` + `--seed` +
  `--num-envs`, no budget flags); the budget comes from `episode_boundaries[-1]`
  (:582), matching the B-side formula in `_run_dreamer_srl` step 2.
- **train.py wandb-kwargs claims**: plan step 3 matches code exactly —
  `:903-907` (project/entity/group CLI-or-config, `job_type` config-mandatory,
  `name = args.wandb_name or tag`) and `:898` (`WANDB_AVAILABLE and not no_wandb and
  not wandb.disabled`), `wandb_login` at `:900`.
- **19-config inventory**: 19 YAMLs in `configs/models/dreamer_srl/`, exactly 17
  declaring `"DreamerV3"` — matches File Change 3's 17-edits-2-additions split.
- **Whitelist placement + DreamerV3 hint**: `get_mandatory('agent.algorithm')` at
  :649 with the archived-stack raise at :651-658 — the plan's insertion point and
  hint-extension are consistent with the code.
- **Uniform-mask rule soundness** (the stress question): every path by which a *real*
  training-relevant divergence could hide behind identical cross-stage masked diffs is
  closed — `body.*` is masked only when flagged (harness flags none), seeding and
  curriculum cadence bypass the dumped configs entirely — modulo the single
  `training.checkpoint_frequency` conditionality gap recorded as N3.

## Recommendation

Fold N1–N5 into the plan as rev-2.1 line edits (five small wording/instruction
changes; no design change), or hand rev 2 to `developer` with this review attached as
a binding rider. Either way the three-gate structure stands as designed.
