---
title: "Adversarial review — dreamer_srl → train.py integration plan (thin dispatch, 3 gates)"
topic: dreamer
status: active
reviewer: fresh-eyes adversarial reviewer
created: 2026-07-24
last_updated: 2026-07-24
audited_doc: "docs/develop/active/dreamer/DREAMER_SRL_TRAIN_PY_INTEGRATION_PLAN.md + DREAMER_SRL_TRAIN_PY_DIVERGENCE_MATRIX.md, verified against working tree v3.0 @ fff6055 (train.py 2,472 lines; dreamer_srl_main.py 2,112 lines)"
---

# Review: dreamer_srl → train.py integration plan — what it gets wrong, misses, or hand-waves

## Verdict (plain language)

**SOUND-WITH-CORRECTIONS.** The plan's core architecture is right and most of its
line-level claims check out against the code: the extraction seam at
`dreamer_srl_main.py:557` really is a clean waist, the "unknown algorithm hangs forever"
bug is real and the whitelist fix lands in the right place, the dispatch point genuinely
bypasses train.py's open episode-metrics bug, and the delegated loop contains no
wall-clock-dependent control flow — so CPU bit-identity is an achievable Gate 1 target.
But the review found **one gap where a silent training change in an in-scope feature
would pass all three gates (curriculum mode is never exercised by any gate)**, plus a
cluster of plan-vs-code contradictions: the WandB identity fields are hardcoded to
`"DreamerV3"` *inside* the code region the plan pledges to leave byte-untouched (so the
compatibility table's row C7 describes behavior the plan will not produce); the Gate 1
harness as written passes flags to the legacy entry point that do not exist there; and
the plan was not reconciled with the experiment-eval refactor (`fff6055`) that landed
after it was drafted — it rejects a CLI flag (`--experiment-eval`) that no longer exists
and says nothing about the new `--eval-config` flag. None of these break the thin-dispatch
design; all are fixable before implementation. The plan must be revised (not just
implemented carefully) on findings 1–5 before Phase 1 starts.

Binding-decision compliance: thin dispatch ✅ · unified CLI/config ✅ · shim ⚠️ (findings
5, 8) · three gates ⚠️ (findings 1, 3) · "every behavior delta enumerated" ❌ (findings
2, 4, 10, 11) · feature scope ✅ (curriculum in scope but unproven — finding 1).

---

## Findings

### BLOCKER

**1. Curriculum mode is in binding scope but no gate exercises it — a curriculum
config-resolution divergence would pass Gate 1, Gate 2, and Gate 3 silently.**
Gate 1's harness is single-config (`--episodes 40`, one env YAML; plan §Phase 2) and
Gate 2 is the single-config basic04 family (plan §Phase 3). Yet curriculum is exactly
where the two entry points resolve configs most differently, and the differences are
real in code, not hypothetical:
- Stage-config *construction* differs: dreamer rebuilds each stage from scratch
  (`dreamer_srl_main.py:102-134` — defaults + 4 fixed files + stage YAML), while
  train.py deep-copies the fully-merged base (`train.py:193-201`), which includes the
  logger layer and the selectable `--eval-config` layer.
- Worse, train.py **mutates the schedule's stage-0 config after building it**:
  `config = schedule.stage_configs[0]` (train.py:563) followed by
  `config.merge(agent_config)` (train.py:588) and the CLI-override block
  (train.py:591-604) writes the agent config, `wandb.*`, `tag`, and `seed` into
  `schedule.stage_configs[0]` *in place*. The schedule object the plan passes through
  the seam therefore has a stage 0 that is polluted relative to stages 1..N and
  relative to today's dreamer schedule — and the extracted body dumps all stage
  configs to the run dir (`dreamer_srl_main.py:936-951`), so `stage_00_*.yaml` changes
  content. Neither the compat table (C1–C18) nor the risk register mentions any of this.
- The existing `test_continual_schedule.py` tests dreamer's *own* builder
  (imports `_build_continual_schedule` from dreamer_srl_main —
  tests/algorithms/dreamer_srl/test_continual_schedule.py:38), which after Gate 3 is
  dead code; the *live* builder (train.py's) is never parity-tested for dreamer.

Required fix: either add a curriculum leg to the Gate 1 harness (a 2-stage toy
schedule, comparing `resolved.json` stage-config dicts and the telemetry across a
stage swap), or explicitly de-scope curriculum-via-train.py until proven and keep the
legacy path for curriculum — but the plan must pick one and enumerate the stage-0
mutation either way.

### MAJOR

**2. Compat row C7 is contradicted by code the plan pledges to keep byte-untouched:
the extracted body hardcodes the WandB identity to `"DreamerV3"`.**
`dreamer_srl_main.py:829` sets `"algorithm": "DreamerV3"` in the wandb config payload,
and line 865 — `wandb_config.setdefault("agent", {})["algorithm"] = "DreamerV3"` — is
not the "defense-in-depth default" its comment claims: `setdefault` returns the
*existing* `agent` dict (spread from the agent YAML at line 861) and then
**unconditionally overwrites** its `algorithm` to `"DreamerV3"`. File Change 1 keeps the
payload untouched except for the four provenance fields, so after the 19-config rename,
new runs launched via train.py would still report `agent.algorithm = "DreamerV3"` (and
top-level `algorithm: "DreamerV3"`) in wandb.config while their dumped configs and the
dispatch say `dreamer_srl` — the exact opposite of compat row C7's claim that "WandB
dashboard filters on `DreamerV3` won't match NEW runs". Gate 2's R7 checklist item
would likely catch the mislabel, but the plan itself asserts the wrong delta.
Required fix: File Change 1 must enumerate edits to lines 829/865 (spec-driven
algorithm string), and C7 re-verified.

**3. The Gate 1 harness, as specified, cannot run: it passes flags to the legacy
A-side that the plan simultaneously promises not to add.**
Phase 2 runs "both entry points … with `--no-wandb --parity-dump <tmpdir>`" and
"`--checkpoint-frequency 20`". But `dreamer_srl_main.py`'s argparse
(lines 429-491) has **no `--checkpoint-frequency` flag** (compat row C6 says so
itself) and no `--parity-dump` flag — while the Design section pledges `main()` "keeps
its argparse + config resolution byte-for-byte" / "exactly as today". As written, the
A-side subprocess exits with an argparse error and Gate 1 never produces a dump.
Required fix: state that `main()` gains exactly two additive flags (`--parity-dump`,
and either `--checkpoint-frequency` or — cleaner — put `training.checkpoint_frequency: 20`
in the harness env YAML so neither side needs the flag), and drop the "byte-for-byte
argparse" wording.

**4. The plan was not reconciled with the landed experiment-eval refactor (`fff6055`):
it rejects a flag that no longer exists and ignores the new one.**
C13 and File Change 2 step 1 say "`--experiment-eval` is already rejected by the
existing gate" — train.py has **no `--experiment-eval` flag** (grep of argparse,
train.py:388-445). The landed mechanism is `--eval-config` (train.py:415-420) selecting
the evaluation layer, plus the config gate `experiment.during_training.enabled`
(train.py:668-673, which does raise for non-rPPO when enabled — that part of the plan's
claim survives). But `--eval-config` itself is a **new, unenumerated behavior surface
for dreamer**: via train.py a dreamer run can swap its entire evaluation layer (which
feeds `testing.*`/`experiment.*` keys into the env_cfg the loop reads, e.g.
`testing.auto_render_after_eval` consumed at dreamer_srl_main.py:636), a capability
direct launches never had. It appears in neither the C12 additive list, the C13
reject list, nor the divergence matrix. Required fix: decide (pass-through as additive
C12 row, or reject for dreamer) and enumerate; delete the `--experiment-eval`
references. Note the plan predicted the refactor would move keys to
`evaluation.experiment.*`; the landed key is `experiment.during_training.enabled` —
File Change 2's gate-relocation instruction should name it.

**5. Shim translation is incomplete for a flag combination the legacy resolver
supports: `--episodes M --total-steps N`.**
Today that combo is meaningful — episode mode with an explicit env-step cap
(`dreamer_srl_main.py:584-586`: `total_timesteps = env_step_override or (episodes *
env_max_steps * num_envs)`). The plan's shim rule (C3 / File Change 7) only translates
a *bare* `--total-steps N` and otherwise "forwards everything else verbatim" — but
train.py has no `--total-steps` flag, so the combined form dies on an argparse error
through the shim. Loud, not silent, but it violates "deprecated-but-**working** shim"
for a supported invocation. Required fix: shim always rewrites `--total-steps N` →
`--total-timesteps N`, and additionally injects `--episodes 0` only when `--episodes`
is absent. (Same handling for the legacy `--total-timesteps` alias, which is already a
train.py flag with identical semantics — verify no clash when both aliases appear.)

### MINOR

**6. Gate 1's accepted-diff mask omits `seed`, which the plan's own dispatch writes
back into the dreamer env_cfg.** File Change 2 applies "seed, body flags,
checkpoint_frequency" to `dreamer_env_cfg`; C11 lists "seed" among the written-back CLI
overrides — but the Phase 2 pass-criteria mask enumerates only `wandb.*`, `tag`,
`episodes`, `training.checkpoint_frequency`, and conditional `body.*`. With the harness
passing `--seed 7`, the B-side env_cfg carries `seed: 7` vs the A-side's untouched
`seed: 42` → the gate fails on its own accepted delta. Add `seed` to the mask.

**7. The seam inventory is slightly incomplete: the extracted body consumes two
pre-seam locals beyond the enumerated set.** `import os as _os`
(dreamer_srl_main.py:498) and `_project_root` (line 499) are defined above the seam and
used below it (results dir + dumps, lines 908-960). Mechanical (re-import / re-derive
inside `run_dreamer_training`), but the divergence matrix §7 claim "everything below
consumes only those four plus a dozen scalar `args.*` fields" is not literally true, and
the senior-developer's "only the enumerated mechanical substitutions" byte-diff
checklist will trip on it. Also, File Change 1's "delete lines 567, 574–594" contains
line 577 (`env_max_steps = env_cfg.get_mandatory(...)`), which the same bullet says must
stay — restate the range precisely.

**8. C8/§4.6 claim `log_code` runs for dreamer via train.py, but no file change
implements it.** train.py's `wandb.run.log_code` (train.py:957) is in the rPPO wandb
block the dreamer dispatch never reaches; `DreamerRunSpec` has no log_code field and
File Change 1's wandb-init substitution list doesn't add the call inside the seam. As
planned, dreamer runs get **no** code upload — either fix the compat row or add the
spec field + call site to File Change 1. (Same class as finding 2: the compat table
promises a delta the file changes don't produce.)

**9. Phase 4 misses two legacy-driver consumers beyond `test_eval_telemetry_wandb.py`.**
`tests/algorithms/dreamer_srl/test_eval_video_smoke.py:26,79-86` subprocess-drives
`dreamer_srl_main.py` with legacy flags (`--env-config`, `--total-steps`, no
`--episodes`) — after the flip it silently becomes a shim-integration test running the
full train.py path (arguably desirable, but should be a named decision, and its 10-min
timeout budget re-checked); `tests/algorithms/dreamer_srl/bench_sps.py:57` pins
`TRAINER_SCRIPT` to dreamer_srl_main.py, so post-flip SPS benchmarks measure
shim+train.py, not the loop alone. Also the preserved-import list omits
`_build_continual_schedule` (test_continual_schedule.py:38) — it survives the
extraction (module-level), but list it so Phase 4's deletion of "the old
config-resolution body" doesn't overreach.

**10. Shim env-var timing is unexamined.** train.py sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and the `--device` platform vars at **module
import time** (train.py:40-55). In the shim path, `dreamer_srl_main.py` has already
imported jax and the full dreamer module chain before `import train` runs; if anything
in that chain ever initializes the JAX backend at import, the env vars silently no-op
and shim-path GPU behavior diverges from direct train.py (preallocation, device pick) —
unverifiable by Gate 1 (CPU) or Gate 2 (doesn't test the shim on GPU). Today the chain
appears not to trigger backend init (no module-level array creation found in
utils/loss/buffers), so this is a verify-at-implementation note: the shim must rewrite
`sys.argv` *before* `import train`, and Phase 4's smoke should assert
`XLA_PYTHON_CLIENT_PREALLOCATE` took effect (or C10 documents the shim exception).

**11. Results-dir path form quietly changes from absolute to relative.** Legacy builds
`results_dir` under the hard-coded project root (`dreamer_srl_main.py:913`); File
Change 2 step 4 uses relative `os.path.join("results", ...)` (matching rPPO,
train.py:843). Identical only when CWD is the repo root — true for run_command.py
launches per the launch convention, but it's an unenumerated delta; add it to C9 or pin
the dispatch to the absolute form.

**12. Plan's Gate-3 flip target is thinner than described.** No `.sh` file in the repo
currently invokes `dreamer_srl_main.py` at all (grep across `train_command-agent.sh`,
`train_command-new.sh`, `scripts/lab/`); `train_command-agent.sh`'s dreamer-named
blocks are archived NNX-era train.py invocations. Real dreamer launches are ad-hoc
`run_command.py` command lines. The flip step is therefore even lower-risk than
planned, but item 9's "future Dreamer blocks" template is the only real deliverable —
plus updating the two docs that teach the legacy CLI (module docstring, README/howto).

---

## Verified correct (credit where due)

- **Seam location and cleanliness** — the waist at `dreamer_srl_main.py:557` is real:
  no module-level mutable state, no atexit/signal handlers, no closures over pre-seam
  locals except the two noted in finding 7; `_emit_episode_row`'s closure hazard is
  internal to the extracted body and documented in-code (lines 1039-1046).
- **"Unknown algorithm hangs forever" is real and the fix placement is right** — no
  `else` raise after the PPO branch (train.py:1151); an unknown string reaches the
  no-op `while` at train.py:1442 and spins forever. The whitelist after
  `get_mandatory('agent.algorithm')` (train.py:649) kills it for every algorithm.
- **Dispatch really does bypass the open episode-metrics P1** — dispatch lands at
  ~line 690 (after the experiment gate at 668-690), far above the branch loop at 1442;
  and the dreamer loop always takes the two-level logging path under unified layering
  (`logging:` blocks in both configs/train/default.yaml and dreamer_srl.yaml →
  `resolve_logging_cfg` non-None), matching plan Analysis #4.
- **CPU bit-identity is an achievable Gate 1 target** — the loop has no wall-clock
  control flow (time feeds only logging/postfix), eval passes take `seed=` and never
  consume the training `key`, np.random is seeded inside the seam, and the compared
  quantities (per-iteration losses, `policy_step`, episode counters, grad-step/Ratio
  state, buffer fill, final PRNG key, per-module param checksums) genuinely cover risks
  R1/R2.
- **Compat rows spot-verified at line level**: C4 (dreamer `--seed` default 0 at :460
  vs config 42), C5 (both fallbacks are the same 100 — `training.episodes` in
  train/default.yaml and top-level `episodes`), C6 (flag absent in dreamer argparse),
  C14 (`--buffer-device`/`--legacy-grad-loop` at :481/:485), C15 (train.py's handler
  at :122-129 swallows the first SIGTERM — registering it around the dreamer loop
  would hang remote kills; the move below dispatch is correct), 1.14/6.2 logging
  defaults (5000/200/200/100 vs 5000/4000/100/50 — both match code), 4.4
  define_metric inventories, 2.1 layer orders, and the 19-config inventory
  (17 × `DreamerV3` + `agent_xs.yaml`/`01_food_only_smoke.yaml` with no agent block).
- **C9's pin to `JAX_DreamerSRL` is load-bearing and correctly identified** —
  `scripts/eval/dwell_sweep/run_sweep.py:101` hard-expects that parent name, and
  `scripts/eval/dreamer_srl_probe_eval.py` + `scripts/dreamer/*` all glob
  `results/JAX_DreamerSRL/<run>/models|checkpoints|recordings` — all preserved by the
  plan's layout decisions (R4), incl. `eval_rollout.py:1414`'s hard requirement for
  `models/agent_config.yaml`.
- **The two `ContinualSchedule` dataclasses are field-identical** (5 fields, identical
  `stage_for_episode`) — duck-typing through the seam is safe.
- **Scope call on "probe-eval"** — the offline dreamer probe battery
  (`scripts/eval/dreamer_srl_probe_eval.py`) reads only the unchanged run-dir layout,
  so the plan's reading of decision 5 ("probe-eval as dreamer does it today" = the
  checkpoint video/stats eval) holds without touching it.
