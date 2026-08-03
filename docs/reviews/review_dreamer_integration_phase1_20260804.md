---
title: "Code review — Dreamer→train.py integration Phase 1 + Gate 1 (worktree branch)"
topic: dreamer_srl / train.py integration
status: active
created: 2026-08-04
last_updated: 2026-08-04
---

# Code Review: Dreamer-SRL → train.py Integration, Phase 1 (seam + dispatch + parity harness)

## Verdict

This review checked the work that lets the Dreamer world-model agent launch through the project's
unified training front door (`train.py`) instead of only through its own standalone script. The
change extracts the Dreamer training loop behind a frozen "run spec" boundary object, adds a thin
dispatch in `train.py`, and proves — with a four-leg bit-identity harness — that both entry points
train identically. **The implementation is sound: no Critical findings.** Every spec field the old
script passes was verified value-identical to its pre-refactor behavior; the test-only random-number
pin that makes bit-identity achievable is genuinely dormant in production and cannot fake a pass;
the dispatch's copy discipline, loud-failure guards, and signal-handler placement are correct. Three
Moderate findings remain, all pre-merge-fixable or acceptable-with-a-note: a silently-ignored
`--total-timesteps` trap on the new path (a launch migrated by hand from the old script would train
only the 100-episode smoke placeholder), a confusing crash when the env config (rather than the
agent config) names the algorithm, and a self-description gap in the dumped env config's episode
budget (parity-equal with the legacy script, so not a regression).

Review scope: worktree `worktree-agent-a18eeb59fa7ffffeb`, commits `15fa01b` / `77ae13c` / `be89c97`
vs base `c00ffc5` (v3.0 tip). Plan: rev 5.1 of the integration plan
([[DREAMER_SRL_TRAIN_PY_INTEGRATION_PLAN]]), Implementation Report in the worktree's copy.

**Severity legend — reproduce it verbatim in every report so the labels never need looking up:** 🔴 Critical = fix before going further · 🟡 Moderate = likely costs a re-run · 🟢 Low = cosmetic · ❓ Open = an assumption nobody has verified yet.

## Scope note

`git diff --stat c00ffc5..HEAD`: 10 files, +1334/−87. The task brief listed
`src/algorithms/dreamer_srl/buffers.py` as changed — **it is not**: it has zero diff in this range.
The parity-guard sampler-RNG pin lives in `dreamer_srl_main.py` (the guarded block after buffer
construction), reaching into `buffer._rng` from outside. All findings below cite worktree paths.

## Findings

| # | Sev | Location (worktree) | Issue | Suggested fix |
|---|-----|---------------------|-------|---------------|
| 1 | 🟡 | `train.py:427-440` (`_run_dreamer_srl` step 2) | Bare `--total-timesteps N` (no `--episodes`) on the dreamer dispatch is **silently ignored as a budget**: `episodes` falls back to the config's root `episodes` key (`configs/train/default.yaml:82` — the 100-episode smoke placeholder), the loop runs in episode mode (`dreamer_srl_main.py:1833` ignores `total_timesteps` when `episodes > 0`), and no warning fires (the env-step WARNING is gated on `episodes == 0`). The legacy entry point forced env-step mode with a loud WARNING for the same flags. Documented as compat row C3 (env-step mode now requires `--episodes 0 --total-timesteps N`), and the planned Gate-3 shim will translate old commands — but until the shim lands, a hand-migrated legacy launch (e.g. the historical `--total-steps 50000000` style) trains 100 episodes and exits, wasting a node-day. The parity harness cannot catch this: it always passes `--episodes`. | Add a loud guard in `_run_dreamer_srl`: if `args.total_timesteps is not None and episodes > 0`, raise (or at minimum WARN) that the cap is inert in episode mode and env-step mode needs `--episodes 0`. |
| 2 | 🟡 | `train.py:633` vs `train.py:821` | The peek (`_peeked_algorithm`, read from the **agent YAML alone**) and the authoritative `algorithm` (read from the **merged config**) can disagree: an env `--config` that sets `agent.algorithm: dreamer_srl` while the agent YAML omits it (all 19 live dreamer agent configs omit it today, pending the deferred rename) yields `algorithm == "dreamer_srl"` but `_peeked_algorithm is None` → the dreamer train-defaults merge, stage-0 deepcopy, and env-cfg snapshot are all silently skipped, and `_run_dreamer_srl` receives `env_cfg=None` → bare `AttributeError: 'NoneType' object has no attribute 'get_mandatory'`. Loud, but wrong-shaped and undiagnosable from the message. | After `algorithm = config.get_mandatory('agent.algorithm')`, assert `algorithm != "dreamer_srl" or _peeked_algorithm == "dreamer_srl"` with a message naming the fix ("declare agent.algorithm in the agent config"). |
| 3 | 🟡 | `train.py:436` + `dreamer_srl_main.py` env-config dump | `config.set('episodes', episodes)` claims "Persist for the dump (G3/L4 convention)" but the dreamer path never dumps the merged `config` (deliberately — the run dir carries separate `models/env_config.yaml` / `models/agent_config.yaml`, asserted by the layout smoke test). The write is dead, and the dumped `env_config.yaml` still self-describes `training.episodes: 100000000` regardless of the actual `--episodes` budget. Parity-equal with the legacy entry point (same gap there), so not a regression — but it is exactly the KNOWN_BUGS "CLI overrides not saved" (L4) failure shape recurring on a new path, and compat row C11's self-description goal is not met for the budget. | Either mirror the resolved budget onto `dreamer_env_cfg` (`dreamer_env_cfg.set('training.episodes', episodes)`) before dispatch, or delete the dead `config.set` and its misleading comment. Flag to `bug-curator` if deferred. |
| 4 | 🟢 | `dreamer_srl_main.py:39`, `:654`, `:794` | Hardcoded main-checkout absolute path three times: `sys.path.insert(0, …)` at module import, `_project_root` in `main()`, and the re-derived `_project_root` inside `run_dreamer_training` (new code — the seam extraction *duplicated* the hazard rather than fixing it). Script-path invocation from a worktree imports the main checkout's `src/`; the legacy results-dir fallback writes into the main checkout. The harness sidesteps it via `python -m` + cwd; production train.py path is immune (results_dir always resolved by the caller). Already flagged as deviation 6 in the Implementation Report. | Replace with `Path(__file__).resolve().parents[3]` in a follow-up (out of Phase-1 byte-diff discipline; fine to defer, but the *new* line at :794 needn't have copied it). |
| 5 | 🟢 | `train.py:1037-1041` (new registration point) | Signal-handler move is correct for the dreamer path (default handlers → SIGTERM kills, matching legacy; avoids the pre-fix swallow-SIGTERM hang demonstrated in the regression test). Side effect on the rPPO path: during config load / stage validation / env probe (now *before* registration), SIGINT/SIGTERM hard-kill instead of setting the graceful `stop_requested` flag. Behavior change, but strictly an improvement (previously setup was un-interruptible by a single Ctrl-C). | None — note for the record. |
| 6 | 🟢 | `tests/algorithms/dreamer_srl/test_entry_parity.py:281-282` | `_SPEC_ACCEPTED_DIFF` filtering parses diff strings (`d.split(".")[0].split(":")[0].split(" ")[0]`) — correct for the three current keys, but a future spec field whose name prefixes an accepted key (or a diff-string format change in `_deep_diff`) would silently widen the mask. | Compare structurally: drop the accepted keys from both spec dicts before `_deep_diff`, as `_apply_mask` already does for `env_cfg`. |
| 7 | ❓ | `dreamer_srl_main.py:1102-1119` (parity pin) + Implementation Report deviation 1 | The underlying production bug the pin works around — dreamer_srl same-seed runs are **not bit-reproducible** (unseeded replay-sampler RNG, `buffers.py:76/:782`; measured grad-steps 404 vs 440 on identical invocations) — is real, unfixed, and not yet in the Known Bugs registry (my grep of `KNOWN_BUGS.md` found no covering row; reproducibility rows there concern other paths). The Implementation Report correctly names `bug-curator` as owner. | Parent should spawn `bug-curator` to record the row before this branch lands, so the "same seed = same episode" reproducibility property is not silently assumed by future experiment designs. |

## Per-target assessment

### T1 — DreamerRunSpec provenance ✅ (with findings 1–3)

Legacy `main()` spec construction (`dreamer_srl_main.py:751-774`) reproduces pre-seam behavior
field-for-field: `seed=args.seed` (argparse default 0, compat C4), hardcoded `entity="sungwoolee"`,
`"DreamerV3"` label (C7), `wandb_log_code=False` (C8), `results_dir=None` → legacy naming (C9),
`group`/`job_type` `None` → omitted from `wandb.init` (the `{k: v … if v is not None}` filter
reproduces the old 3-kwarg call exactly). The budget block moved verbatim into `main()` §2b —
including the `env_step_override` forcing of `episodes=0` and its WARNING; the dropped `total_steps`
alias verifiably had zero consumers (loop condition at :1833 reads only `episodes`/`total_timesteps`).
`run_dreamer_training` re-reads `env_max_steps` and `buffer_size`+ from configs identically on both
paths. train.py-side provenance: every `get_mandatory` traced to a populated layer (`training.num_envs`
from `configs/train/dreamer_srl.yaml` via the peek-gated merge for single-config, and via
`_build_continual_schedule`'s base-config clone for curriculum — the base already contains the
dreamer train defaults *and* the logger layer, which is why the harness masks `wandb`/`tag`
unconditionally). `agent_config_path` loaded fresh at dispatch — deterministic. Deltas vs legacy are
exactly the documented compat rows (C3/C4/C5/C7/C8/C9) plus finding 1's silent corner of C3.

### T2 — Parity-guard sampler-RNG pin ✅

- **Inert in production**: `--parity-dump` defaults to `None` in both argparse blocks; train.py
  rejects it loudly for every non-dreamer algorithm (tested); every hook call site — `_parity_init`,
  the pin, prologue/epilogue, checkpoint append, `_parity_finalize` — is guarded by
  `spec.parity_dump is not None`. No code path reaches the pin without the flag.
- **No PRNG-state sharing**: the pin creates fresh `np.random.Generator` (PCG64) objects; production
  seeding (`np.random.seed` → global MT19937, `jax.random.PRNGKey` → JAX stream) is untouched and
  algorithm-disjoint. The GPU-buffer sampling path uses the JAX key, not `_rng` — unaffected.
- **Per-sampler independence**: outer buffer seeded `spec.seed`, per-env sub-buffers
  `spec.seed + 1 + i` — distinct `SeedSequence`-decorrelated streams, no collision (no inner buffer
  ever receives the outer's seed). `buffer.reset()` at stage swaps only zeroes `_pos`/`_full`
  (verified `buffers.py:598-608`, `:901-904`), so the pin survives curriculum boundaries as claimed.
- **Cannot fake parity**: the pin removes only OS-entropy nondeterminism, which is *identical-run*
  noise and cannot encode an entry-point difference; both sides derive the same streams from the
  same `--seed 7`, so any genuine divergence (buffer content, control flow, counters, weights)
  still fails the bit-identity criteria — the same indices into different content give different
  losses/checksums, and control-flow drift breaks the counter/PRNG-key equality first.

### T3 — Dispatch ✅ (finding 2, 5)

Deepcopy placement verified against the stage-0 pollution mechanism: propagation loop
(`train.py:692-708`) runs **before** the stage-0 deepcopy (`:716`), so the copy carries the
propagated overrides; the agent merge + CLI-override block then mutate only the copy;
`dreamer_env_cfg` is a second deepcopy taken pre-agent-merge with its own CLI mirror
(`:747`, `:768-775`). The 1b harness criterion (per-stage pre-mask diff-set uniformity) is exactly
the pollution signature and passed. Propagation is peek-gated so rPPO curriculum dumps are
byte-unchanged (N2 honored). Loud-fail matrix: unknown algorithm → whitelist `ValueError`
(regression-tested against the pre-fix infinite loop, exit-124 demo archived); archived `DreamerV3`
→ hint added; rPPO-only flags on dreamer and dreamer-only flags on rPPO both raise (all
parametrized-tested); `--total-timesteps`+`--configs-dir` and `--episodes`+`--configs-dir` both
raise, matching legacy. `--device` is deliberately usable on the dreamer path (device setup runs
before dispatch) — a benign capability gain. Signal-handler ordering correct (finding 5).

### T4 — Seam edges ✅ (finding 4)

No module-level state or import-order change: `dreamer_srl_main` performs no `jax.config.update` at
import; train.py imports it lazily inside `_run_dreamer_srl`, after train.py's own device setup;
wandb is imported inside the loop's try-block on both paths. `import copy` is the only new train.py
top-level import. The pre-existing hardcoded-path hazard was duplicated, not worsened (finding 4).

### T5 — Parity-harness quality ✅ (finding 6)

The comparison cannot pass on trivially-equal dumps: A- and B-sides write to disjoint dump dirs
(`a1a` vs `b1a`, …) and disjoint results dirs, and `_load` reads each side's own files — there is no
shared path either side could accidentally read. The non-vacuity criterion (non-empty telemetry +
≥1 loss-updating iteration at/after `train_start_iter`, checked **per side**) blocks the
instant-exit vacuous pass the round-3 plan review identified. The GPU leg's bit-identity→rtol
fallback still enforces exact equality on integer arrays, counters, and the PRNG key inside the
rtol call. Worktree isolation of the A-side via `python -m` + cwd is sound (the `-m` package
resolution binds `src` to the worktree before the module's main-checkout `sys.path.insert` runs).
Checkpoint-cadence telemetry (`ckpt_episodes`) makes a cadence-resolution bug fail numerically, not
just in the config diff.

## Conventions audit

| Convention | Verdict | Note |
|---|---|---|
| Pytree / immutability | ✅ | No `EnvState`/`EnvParams` or pytree code touched. |
| JIT recompilation | ✅ | No static/traced field changes; parity hooks are host-side Python outside jit. |
| vmap / batch | ✅ | No vmap changes; `num_envs` plumbing value-identical. |
| PRNG threading | ✅ | Seed sweep `args.seed`→`spec.seed` exhaustive (4 sites verified); parity pin isolated (T2); eval seed stays `testing.seed`. |
| Sensor / obs sync | ✅ (n/a) | No sensor or observation code touched. |
| Config protocol | ✅ | All new reads use `get_mandatory`; no fallback defaults introduced (the harness fixtures carry their keys explicitly). Finding 3 is a dead write, not a fallback. |

## Prior-art pass

`KNOWN_BUGS.md` grepped for dreamer/dispatch/parity: the B1–B4 "unknown algorithm spins forever"
row is **fixed** by this branch (regression test in place) — `bug-curator` should mark it on
landing; the "CLI overrides not saved" (L4) shape recurs as finding 3; the unseeded-sampler
reproducibility bug (finding 7) has **no existing row** — `bug-curator` is the named owner.

## Conclusion

No Critical findings; three Moderates (silent `--total-timesteps` trap, peek/merged-algorithm
mismatch crash shape, dead budget write / env-config self-description) are cheap pre-merge fixes;
the seam, pin, dispatch, and harness are otherwise correct and well-guarded.

Reviewed by: code-reviewer
