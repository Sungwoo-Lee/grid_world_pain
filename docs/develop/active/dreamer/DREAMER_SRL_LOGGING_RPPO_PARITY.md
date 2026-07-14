---
title: "dreamer_srl logging — rPPO console/WandB parity (presentation-only)"
topic: dreamer
status: active
created: 2026-07-14
last_updated: 2026-07-14
---

# dreamer_srl logging — rPPO console/WandB parity (presentation-only)

> **Status**: PLANNED
> **Opened**: 2026-07-14
> **Related**: [[train.py]] (rPPO reference driver), `src/algorithms/dreamer_srl/dreamer_srl_main.py` (target)

---

## Context

The model-based Dreamer trainer (`dreamer_srl_main.py`, the "dreamer_srl" driver) floods
its logs. It prints one line **per finished episode** — `[iter N] episode done: env=X
ep_len=… ep_rew=…` — and with 16 parallel environments an episode ends every 2–3 loop
iterations, so a long run emits millions of these lines. Because WandB captures stdout
verbatim (no `\r` collapsing for plain `print`), the WandB "Logs" console for a Dreamer run
becomes unscannable.

The model-free trainer (`train.py`, which drives rPPO / DQN / DRQN / PPO) does not have this
problem. It wraps its main loop in a single **`tqdm` progress bar** — a one-line, in-place
updating status bar. WandB collapses tqdm's carriage-return (`\r`) rewrites, so a 21-hour rPPO
run shows only a config banner, a "JIT compiling…" line, and one `[CHECKPOINT] Saving model at
episode N (Iteration I)…` line per checkpoint (~44 lines total). Clean and scannable.

**The ask:** make dreamer_srl's console/WandB output look like rPPO's — a startup config
banner, a single tqdm progress bar carrying live metrics, and `[CHECKPOINT]`-style lines at
save points — and demote the per-episode line to `--debug` only (the user confirmed it was
debug scaffolding). **This is a presentation-only change: no training loop, loss, buffer,
checkpoint-cadence, or metric-computation logic is touched.** Every metric the new bar shows
(steps-per-second, world-model loss, etc.) is already computed by the existing code; we only
change how it is *displayed*.

The two drivers are architecturally independent: `dreamer_srl_main.py` imports nothing from
`train.py` (it hand-copied logic, marked with `# Ported from train.py` comments), and
`train.py` has a fail-fast stub that rejects Dreamer. So Dreamer cannot reuse `train.py` as a
driver — only its *presentation style* is being mirrored.

## Analysis

### Root cause of the flood

`dreamer_srl_main.py:1331-1332` — inside the done-envs loop, gated only by `not args.quiet`:

```python
iteration_episodes.append(ep_data)
if not args.quiet:
    print(f"[iter {iter_num}] episode done: env={i} ep_len={ep_len} ep_rew={ep_rew:.3f}")
```

This fires once per done environment per iteration. At `num_envs=16` that is a continuous
torrent of plain `print` lines that WandB cannot collapse.

### How rPPO stays clean (the pattern to mirror)

- **`from tqdm import tqdm`** at `train.py:59`.
- **Bar creation** at `train.py:1117`: `with tqdm(total=episodes, disable=args.quiet, desc="Training") as pbar:`.
- **Live metrics** via `pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0`
  → `pbar.set_postfix({...})` → `pbar.refresh()` (e.g. `train.py:1592-1599`).
- **Discrete events** (stage swaps, checkpoints) via `pbar.write(...)`, which prints a
  permanent line *without breaking the bar* — e.g. `train.py:1995`:
  `pbar.write(f"[CHECKPOINT] Saving model at episode {total_episodes_completed} (Iteration {iteration})...")`.
- **Startup banner** at `train.py:690-754`: a `not args.quiet` block printing `[Environment]`,
  `[Training]`, and `--- RL API Specifications ---` sections via a small `print_section` helper.

### What dreamer_srl already has (so the new bar is cheap)

- Main loop: `while` at `dreamer_srl_main.py:1189`, `iter_num += 1` at `:1190`.
- Periodic metric block: `dreamer_srl_main.py:1801-1919`. It already computes `sps_env`,
  `WorldModel/loss_model`, `Diagnostic/moments_invscale`, and (at `:1913-1919`) prints
  `[ep X/Y] policy_step=… world_model_loss=… moments_invscale=… sps=…`. These are exactly the
  values a `set_postfix` needs — no new computation required.
- Checkpoint save: `dreamer_srl_main.py:1515` (`[dreamer-srl] Saving checkpoint @ episode N...`)
  and `:1533` (`[dreamer-srl] Checkpoint saved.`).
- Eval/stage `print`s gated by `not args.quiet`: `:1544-1545` (video pass), `:1584-1585`
  (stats pass).
- Startup `print`s: `:624-628` (obs_dim / action_dim / num_envs / episodes / seq_len / …),
  and the `[dreamer-srl] Starting training loop…` message at `:1179`/`:1183`.
- `results_dir` is only known at `:873-879`, and `t_start` at `:1174` — so the banner must be
  emitted **after** `results_dir` exists (place it just before the loop, near `:1174`), not up
  at `:624`.

## Design decision: direct-copy vs. shared logging module

**Recommendation: direct-copy the presentation code into `dreamer_srl_main.py`. Do NOT extract
a shared `src/utils/train_logging.py`, and do NOT edit `train.py`.**

The alternative — factor the banner + tqdm helper + checkpoint-line formatter into a shared
module imported by both drivers — is DRY and drift-proof, but it is the wrong call here:

1. **It fights the established architecture.** `dreamer_srl_main.py` is *deliberately* an
   independent hand-copy of `train.py` (every ported block is marked `# Ported from train.py`),
   and `train.py:456-463` fail-fast-rejects Dreamer. The project has already decided these two
   drivers do not share a code path. Introducing a shared module for *just the log strings*
   would be an inconsistent one-off; the principled DRY move would be to share the whole
   driver, which the architecture explicitly forbids.
2. **Blast radius.** Direct-copy touches exactly one file (`dreamer_srl_main.py`). The shared
   option edits `train.py` — the project's primary, heavily-used driver — which is **running
   rPPO training right now** (basic03/basic04 size sweeps on nodes 107/108/110). Editing the
   source will not affect those already-running Python processes (the module is loaded in
   memory), but it needlessly puts the critical driver in the diff, forces re-verification of
   rPPO's own console/WandB output, and risks a cosmetic change silently altering rPPO logging
   that a parallel session depends on. Not touching `train.py` removes that coordination cost
   entirely.
3. **The shared surface is tiny and cosmetic.** A banner, a `set_postfix` call, and one
   format string. Drift risk is low and purely presentational — a mismatch would be "the two
   banners look slightly different", never a correctness bug. That does not justify coupling
   two intentionally-decoupled drivers or expanding the test surface.
4. **Simplicity-first.** Direct-copy is the minimum change that solves the stated problem.

**Caveat for the future:** if the project later wants one logging home, the right move is a
*deliberate, separately-scoped* refactor extracting a shared module for **both** drivers at
once (with `train.py` re-verified) — not a side-effect of this cosmetic change. Note that as a
follow-up option; do not do it here.

## Implementation Plan

All edits are in **`src/algorithms/dreamer_srl/dreamer_srl_main.py`** only. Presentation-only;
no logic changes.

### Change 0 — import tqdm

Add to the import block (after `import time` at `:27`, or alongside the other third-party
imports near `:31-35`):

```python
from tqdm import tqdm
```

### Change 1 — demote the per-episode line to `--debug`, route via `pbar.write`

At `dreamer_srl_main.py:1331-1332`, replace:

```python
                iteration_episodes.append(ep_data)
                if not args.quiet:
                    print(f"[iter {iter_num}] episode done: env={i} ep_len={ep_len} ep_rew={ep_rew:.3f}")
```

with:

```python
                iteration_episodes.append(ep_data)
                if args.debug:
                    pbar.write(f"[iter {iter_num}] episode done: env={i} ep_len={ep_len} ep_rew={ep_rew:.3f}")
```

Gate flips from `not args.quiet` → `args.debug` (per user: this was debug scaffolding), and the
`print` becomes `pbar.write` so it does not corrupt the bar when it does fire under `--debug`.

### Change 2 — add the tqdm progress bar around the main loop

**Use the manual instantiate/close form, NOT a `with` context manager.** The main loop body
spans ~730 lines (`:1189-1919`); wrapping it in `with tqdm(...) as pbar:` would force a full
reindent of every line in the loop — a massive, hard-to-verify diff that violates the
surgical-change principle. Instead:

- **Create the bar just before the loop**, after `t_start` / the "Starting training loop"
  message (around `:1184`, before the `while` at `:1189`):

  ```python
  pbar = tqdm(total=episodes if episodes > 0 else None, disable=args.quiet, desc="Training")
  ```

- **Drive the bar inside the existing periodic-log block.** At `dreamer_srl_main.py:1905-1919`,
  replace the `if not args.quiet: print("[ep …] …")` block with a `set_postfix` update (the
  bar replaces the periodic stdout line; keep the printed form only under `--debug`):

  ```python
              pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
              pbar.set_postfix({
                  "iter":  iter_num,
                  "step":  policy_step,
                  "wm":    f"{log_dict.get('WorldModel/loss_model', float('nan')):.4f}",
                  "invsc": f"{log_dict['Diagnostic/moments_invscale']:.3f}",
                  "sps":   f"{sps_env:.1f}",
              })
              pbar.refresh()
              if args.debug:
                  progress = f"ep {total_episodes_completed}/{episodes}" if episodes > 0 else f"iter {iter_num}"
                  pbar.write(
                      f"[{progress}] policy_step={policy_step} "
                      f"world_model_loss={log_dict.get('WorldModel/loss_model', float('nan')):.4f} "
                      f"moments_invscale={log_dict['Diagnostic/moments_invscale']:.4f} "
                      f"sps={sps_env:.1f}"
                  )
  ```

- **Close the bar after the loop**, before the "Done." final-log block (before `:1924`,
  `elapsed = time.time() - t_start`):

  ```python
  pbar.close()
  ```

Mirrors rPPO's `pbar.n = min(...); pbar.set_postfix(...); pbar.refresh()` pattern
(`train.py:1592-1599`). `total=None` when `episodes == 0` (env-step fallback mode) gives an
unbounded bar rather than a wrong denominator.

### Change 3 — convert checkpoint + eval prints to `pbar.write`, reformat the checkpoint line

- `dreamer_srl_main.py:1515` — replace:
  ```python
                  print(f"[dreamer-srl] Saving checkpoint @ episode {total_episodes_completed}...")
  ```
  with (rPPO format, `train.py:1995`):
  ```python
                  pbar.write(f"[CHECKPOINT] Saving model at episode {total_episodes_completed} (Iteration {iter_num})...")
  ```
- `dreamer_srl_main.py:1533` — `print(f"[dreamer-srl] Checkpoint saved.")` →
  `pbar.write("[CHECKPOINT] Saved.")` (or drop it — rPPO prints only the "Saving" line; keep a
  single confirmation line at most).
- `dreamer_srl_main.py:1544-1545` and `:1584-1585` — the eval `print`s gated by
  `not args.quiet`: keep the `not args.quiet` gate but change `print(...)` → `pbar.write(...)`
  so they don't break the bar. (These fire only at checkpoint boundaries, so they stay rare and
  scannable — matching rPPO.)

### Change 4 — startup config banner (rPPO style)

Add a `not args.quiet` banner block **just before the training loop** (near `:1174`, after
`results_dir`, `total_timesteps`, `seed`, `num_envs`, etc. are all resolved). Mirror
`train.py:690-754`:

```python
    if not args.quiet:
        width = 60
        print("\n" + "=" * width)
        print(" JAX/FLAX DREAMER-SRL CONFIGURATION ".center(width, "="))
        print("=" * width)

        def _print_section(title, data):
            print(f"\n[{title}]")
            for k, v in data.items():
                print(f"  ● {k:.<25} {v}")

        with_satiation = env_cfg.get_mandatory('body.with_satiation', bool)
        _print_section("Environment", {
            "Grid Size":  f"{env_params.height}x{env_params.width}",
            "Max Steps":  env_max_steps,
            "Mode":       "Interoceptive (Homeostasis)" if with_satiation else "Conventional (Goal-driven)",
        })
        _print_section("Training", {
            "Framework":       "JAX/Flax NNX (Dreamer-SRL)",
            "Total Timesteps": f"{total_timesteps:,}",
            "Episodes":        episodes if episodes > 0 else "(env-step mode)",
            "Parallel Envs":   num_envs,
            "Seq Length":      seq_len,
            "Batch Size":      batch_size,
            "Horizon":         horizon,
            "Seed":            args.seed,
            "Results":         results_dir,
            "WandB":           "Enabled" if use_wandb else "Disabled",
        })
        print("\n--- RL API Specifications ---")
        print(f"Action Dim: {action_dim}")
        print(f"Observation Dim: {obs_dim}")
        print("=" * width + "\n")
```

Use the exact variable names present in the file (`env_params`, `env_cfg`, `env_max_steps`,
`total_timesteps`, `episodes`, `num_envs`, `seq_len`, `batch_size`, `horizon`, `args.seed`,
`results_dir`, `use_wandb`, `action_dim`, `obs_dim`) — the developer should confirm each is in
scope at the chosen insertion point and adjust the `get_mandatory` type hints to match the
project's `Config` API. The existing plain startup prints at `:624-628` may be left as-is
(they precede WandB init and are terse) or folded into this banner at the developer's
discretion — not required for the fix.

### Files changed

| File | Change | New config keys |
|---|---|---|
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | Changes 0–4 (import tqdm; per-episode line → `--debug` + `pbar.write`; add tqdm bar; checkpoint/eval prints → `pbar.write` + reformat; startup banner) | none |

No config keys added, no files under `scripts/` added/moved/renamed, no config-system changes
→ CONFIG_GUIDE / SCRIPTS_DEPENDENCY_MAP maintenance contracts do **not** apply.

## Test / Verification Plan

This is presentation-only, so there is **no correctness regression test to add** — the
verification is a runtime smoke observation of the console.

1. **Baseline smoke (no `--debug`).** Launch a short single-config dreamer_srl run with a tiny
   episode budget and a small checkpoint cadence so at least one checkpoint fires. Per the
   single-config budget gotcha, pass the budget on the CLI, and lower
   `training.checkpoint_frequency` in the scratch config (read at `:594`) so a `[CHECKPOINT]`
   line is emitted within the smoke budget, e.g.:
   ```
   /home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m src.algorithms.dreamer_srl.dreamer_srl_main \
       --config <small env config> --episodes 40 --log-interval 5 --no-wandb
   ```
   (adjust to the project's actual dreamer_srl entry/args; keep `num_envs` at its default so the
   old flood would be obvious).
   **Expect:** startup banner (Environment / Training / RL API sections) → a single `Training:`
   tqdm bar updating in place with `sps`/`wm`/`invsc` in the postfix → one
   `[CHECKPOINT] Saving model at episode N (Iteration I)...` line at the checkpoint → **no**
   `[iter N] episode done: …` lines at all.
2. **Debug smoke (`--debug`).** Re-run the same command with `--debug` appended.
   **Expect:** the per-episode `[iter N] episode done: …` lines reappear (now via `pbar.write`,
   not breaking the bar), plus the periodic `[ep …]` line.
3. **No-op logic check.** Confirm final-log values (`grad_steps`, episodes completed, final
   losses) and any WandB scalar keys are unchanged vs. a pre-change run — the `wandb.log(...)`
   dicts at `:1857` and `:1903` must NOT be edited by this change.
4. **Speed check.** tqdm refreshes only inside the existing periodic-log block (once per
   `log_every` iters), so runtime impact should be negligible. Record before/after `sps` from
   the smoke runs (same config/seed/node); a >5% slowdown would be surprising and warrants a
   look.

---

## Implementation Report

<!-- developer fills this in -->

- **Branch / commit:**
- **Changes made (file:line):**
- **Deviations from plan:**
- **Smoke-run console excerpts (baseline + --debug):**
- **Speed before/after (sps, same config/seed/node):**
- **Blockers:**

## Verification Report

<!-- senior-developer fills this in after developer reports -->
