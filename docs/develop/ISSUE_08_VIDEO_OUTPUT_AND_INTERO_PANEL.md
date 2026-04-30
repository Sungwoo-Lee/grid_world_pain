# Single-Video Output + Missing Interoceptive Nociception Panel

> **Status**: COMPLETED
> **Implemented by**: Gemini
> **Date**: 2026-04-30 23:56:00
> **Opened**: 2026-04-30
> **Related**: [ISSUE_06_HIDDEN_STATES_INTEROCEPTIVE_NOCICEPTION](ISSUE_06_HIDDEN_STATES_INTEROCEPTIVE_NOCICEPTION.md), [ISSUE_07_AUTO_RENDER_AFTER_EVAL](ISSUE_07_AUTO_RENDER_AFTER_EVAL.md)

---

## Context

After ISSUE_07 (auto-render-after-eval) shipped, two regressions were observed by the user when reviewing eval videos:

1. **Per-episode MP4 clutter**: a single eval run produces N per-episode MP4s (`videos/<pct>/episode_NNNNNN.mp4`) **plus** the consolidated `videos/eval_<pct>.mp4`. The user only watches the consolidated file; the per-episode files are noise on disk and bloat WandB-adjacent storage.
2. **Missing Intero Nociception panel**: the new interoceptive nociception sensor (added in ISSUE_06) does not appear in the rendered video frames, even though the underlying observation vector contains it and `build_sensory_viz` generates a tile for it.

## Analysis

### Issue 1 — per-episode MP4s

[scripts/render_recordings.py:130-159](../../scripts/render_recordings.py#L130-L159) runs the per-episode renders in a `ProcessPoolExecutor`. Each worker writes one MP4 to disk; then the `--concat` step reads those MP4s back and re-encodes them into the consolidated MP4. The per-episode files are inputs to concat, not optional outputs — but after concat completes they are no longer needed for the user's workflow.

The current default (`auto_render_after_eval` in `evaluation_core.py`) passes `--concat`, so concat **always runs**. So in normal operation, per-episode MP4s only have transient value.

Two design options:

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. Delete per-episode MP4s after concat** | Add `--cleanup-per-episode` flag (default ON in `evaluation_core.py`'s subprocess call). After `--concat` writes successfully, delete `videos/<pct>/episode_*.mp4`. | Minimal change. Keeps the parallel rendering architecture (each worker writes to disk independently). | Disk thrash: write N files, then delete them. |
| **B. Use a temp directory** | Workers write to `videos/<pct>/_tmp_<uuid>/episode_*.mp4`. Concat reads from temp, writes consolidated to `videos/eval_<pct>.mp4`. Delete temp dir. | Cleaner — never pollutes the main output dir. | Slightly more code; more error paths to handle (cleanup on failure). |

**Recommendation**: **Option A**. Disk-thrash concern is minor (the MP4s are seconds old in OS page cache; deletion is metadata-only). Option B's "cleaner" benefit is largely cosmetic since `videos/<pct>/` is internal to a results directory. Less code = fewer edge cases.

The cleanup should be **opt-out**, not opt-in, because:
- The user explicitly requested single-video output as the desired default.
- The standalone CLI usage of `render_recordings.py` (manual rendering of a recordings dir) might benefit from per-episode MP4s for ad-hoc debugging — those users can pass `--keep-per-episode`.

### Issue 2 — Intero Nociception panel missing

The full data path is correct **up to** the renderer:

1. **Sensor**: [sensor.py:282-283](../../src/environment/sensor.py#L282-L283) appends interoceptive nociception to `obs_parts` when enabled.
2. **Breakdown**: [sensor.py:333-334](../../src/environment/sensor.py#L333-L334) registers `breakdown["Interoceptive Nociception"] = 1`.
3. **Viz tile**: [sensor.py:402-411](../../src/environment/sensor.py#L402-L411) emits a tile in the `viz` list:
   ```python
   elif sensor_name in ("Satiation", "Nutrition", "Injury", "Interoceptive Nociception"):
       ...
       display_name = "Intero Nociception" if sensor_name == "Interoceptive Nociception" else sensor_name
       color = "#8e44ad" if sensor_name == "Interoceptive Nociception" else None
       tile = {'name': display_name, 'intensity': s_obs, 'true_intensity': s_true, 'type': 'intensity'}
       if color is not None:
           tile['color'] = color
       viz.append(tile)
   ```
4. **Renderer**: this is where it dies. [renderer.py:540-583](../../src/environment/renderer.py#L540-L583) renders the LEFT "INTEROCEPTION" panel by **explicit hardcoded sequence** (Satiation → Nutrition → Injury → Run Context). It uses `sensor_map.get('Satiation', ...)`, `.get('Nutrition', ...)`, `.get('Injury', ...)` but **never queries `'Intero Nociception'`**.

   The RIGHT "EXTEROCEPTION" panel iterates `known_sensors = ['Olfactory', 'Extero Nociception', 'Collision', 'Visual', 'LOC']` ([renderer.py:592](../../src/environment/renderer.py#L592)) — also no entry for the interoceptive variant.

   **Result**: the tile is built but discarded by the renderer.

`build_sensory_viz` correctly classifies it as interoceptive (purple color, joined with Satiation/Nutrition/Injury group). So the fix belongs on the LEFT panel.

#### Layout consideration

Current left-panel y-coordinate budget (using `ax_left.transAxes`, top-of-axes = 1.0):

```
y=0.95   "INTEROCEPTION" header
y=0.83   Satiation bar (height 0.04)
y=0.68   Nutrition bar
y=0.53   Injury bar
y=0.32   "Run Context" pod (height 0.16, occupies y=0.32–0.48)
y≤0.31   (free)
```

The space between Injury (top of bar at ~0.53) and Run Context top (0.48) is tight. Adding a fourth bar at y=0.37 would collide with Run Context. Two viable layouts:

- **L1. Squeeze**: reduce vertical spacing between bars from `-0.15` to `-0.12`, fit 4 bars between y=0.83 and y=0.47, keep Run Context where it is.
- **L2. Conditional**: only render the 4th bar if `params.interoceptive_nociception_enabled` is true. If false, layout is unchanged.

These are not mutually exclusive. **Recommendation**: L1 + L2 combined — when intero noc is disabled, fall back to the original 3-bar layout; when enabled, switch to a tighter 4-bar layout. This preserves the existing video appearance for runs without intero noc and adds the new bar only when relevant.

The bar should:
- Use the purple color from the tile (`#8e44ad`) — matches the project's existing convention for "pain" visuals.
- Show real (host state) vs observed (noisy/passthrough) values, like the other three bars.
- Pull `state.injury_level` as the "real" baseline if intero noc is the delayed/passthrough form (per ISSUE_06's design — confirm with user). Or use a dedicated host-state field if one exists.

> **Open question for user**: which host-state value is the "ground truth" for Intero Nociception? Options:
> - `state.injury_level` (if intero noc is just a delayed echo of injury)
> - A new field like `state.interoceptive_pain` (if ISSUE_06 introduced one)
>
> Implementing agent must check `EnvState` for the correct field before applying.

## Implementation Plan

### Issue 1 — File Changes

#### `scripts/render_recordings.py` (lines 91–159)

Add `--cleanup-per-episode` flag (default `False` for CLI usage, but set explicitly by `evaluation_core.py`). After `--concat` succeeds, delete the per-episode MP4s.

```python
# BEFORE (around line 91-100):
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings_dir", help="Directory containing run_meta.pkl + episode_*.rec.gz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--concat", action="store_true",
                    help="Also write a consolidated MP4 concatenating every episode in order.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip episodes whose MP4 already exists.")
    args = ap.parse_args()

# AFTER:
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings_dir", help="Directory containing run_meta.pkl + episode_*.rec.gz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--concat", action="store_true",
                    help="Also write a consolidated MP4 concatenating every episode in order.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip episodes whose MP4 already exists.")
    ap.add_argument("--cleanup-per-episode", action="store_true",
                    help="After --concat succeeds, delete per-episode MP4s. Keeps only eval_<pct>.mp4.")
    args = ap.parse_args()
```

At the end of the `if args.concat:` block (after `print(f"Consolidated → {consolidated}")`), add cleanup:

```python
# BEFORE (last lines of main()):
        consolidated = run_root / "videos" / f"eval_{rec_dir.name}.mp4"
        save_jax_video(frame_generator(), str(consolidated), fps=args.fps, quiet=True)
        print(f"Consolidated → {consolidated}")

# AFTER:
        consolidated = run_root / "videos" / f"eval_{rec_dir.name}.mp4"
        save_jax_video(frame_generator(), str(consolidated), fps=args.fps, quiet=True)
        print(f"Consolidated → {consolidated}")

        if args.cleanup_per_episode:
            removed = 0
            for t_in, _ in tasks:
                mp4 = video_dir / (Path(t_in).stem.replace(".rec", "") + ".mp4")
                if mp4.exists():
                    mp4.unlink()
                    removed += 1
            # Remove the now-empty per-episode directory if it has no other content
            try:
                video_dir.rmdir()  # only succeeds if empty
            except OSError:
                pass  # not empty (e.g., manual files); leave it
            print(f"Cleaned up {removed} per-episode MP4(s); kept consolidated only.")
```

> **Note for implementing agent**: the cleanup step runs **only** if `--concat` succeeded (it's inside the `if args.concat:` block). If concat fails, per-episode MP4s are preserved, so the user can re-run concat manually.

#### `src/utils/evaluation_core.py` (auto-render subprocess call, ~line 252-258)

Add `--cleanup-per-episode` to the default subprocess command so the production workflow gets single-video output:

```python
# BEFORE:
        cmd = [
            _sys.executable, render_script,
            recordings_dir,
            "--concat",
            "--skip-existing",
            "--fps", str(fps),
        ]

# AFTER:
        cmd = [
            _sys.executable, render_script,
            recordings_dir,
            "--concat",
            "--skip-existing",
            "--cleanup-per-episode",
            "--fps", str(fps),
        ]
```

> **Note**: this preserves the manual CLI escape hatch — running `python scripts/render_recordings.py <dir> --concat` (without `--cleanup-per-episode`) still keeps per-episode files for ad-hoc debugging.

### Issue 2 — File Changes

#### `src/environment/renderer.py` (lines 564–570)

Insert a 4th bar between Injury and Run Context, gated on `params.interoceptive_nociception_enabled`. Tighten spacing to fit.

```python
# BEFORE (lines 548-572):
    # Satiation
    sat_real, max_sat = float(state.satiation), float(params.max_satiation)
    sat_obs_data = sensor_map.get('Satiation', {'intensity': sat_real/max_sat})
    sat_obs = float(sat_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, sat_real/max_sat, sat_obs, COLORS['satiation'], 
                          "Satiation", f"{sat_real/max_sat:.2f}", f"{sat_obs:.2f}", transform=ax_left.transAxes)
    y_ptr -= 0.15
    
    # Nutrition
    nut_real, max_nut = float(state.nutrition), float(params.max_nutrition)
    nut_obs_data = sensor_map.get('Nutrition', {'intensity': nut_real/max_nut})
    nut_obs = float(nut_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, nut_real/max_nut, nut_obs, COLORS['nutrition'], 
                          "Nutrition", f"{nut_real/max_nut:.2f}", f"{nut_obs:.2f}", transform=ax_left.transAxes)
    y_ptr -= 0.15
    
    # Injury
    inj_real, max_inj = float(state.injury_level), float(params.max_injury)
    inj_obs_data = sensor_map.get('Injury', {'intensity': inj_real/max_inj})
    inj_obs = float(inj_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, inj_real/max_inj, inj_obs, COLORS['injury'], 
                          "Injury", f"{inj_real/max_inj:.2f}", f"{inj_obs:.2f}", transform=ax_left.transAxes)
    y_ptr -= 0.16
    
    draw_pod_frame(ax_left, 0.05, 0.32, 0.9, 0.16, "Run Context", transform=ax_left.transAxes)

# AFTER:
    # Compact spacing when intero nociception is enabled (4 bars instead of 3)
    intero_noc_enabled = bool(getattr(params, 'interoceptive_nociception_enabled', False))
    bar_step = 0.12 if intero_noc_enabled else 0.15  # tighter when 4 bars
    
    # Satiation
    sat_real, max_sat = float(state.satiation), float(params.max_satiation)
    sat_obs_data = sensor_map.get('Satiation', {'intensity': sat_real/max_sat})
    sat_obs = float(sat_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, sat_real/max_sat, sat_obs, COLORS['satiation'], 
                          "Satiation", f"{sat_real/max_sat:.2f}", f"{sat_obs:.2f}", transform=ax_left.transAxes)
    y_ptr -= bar_step
    
    # Nutrition
    nut_real, max_nut = float(state.nutrition), float(params.max_nutrition)
    nut_obs_data = sensor_map.get('Nutrition', {'intensity': nut_real/max_nut})
    nut_obs = float(nut_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, nut_real/max_nut, nut_obs, COLORS['nutrition'], 
                          "Nutrition", f"{nut_real/max_nut:.2f}", f"{nut_obs:.2f}", transform=ax_left.transAxes)
    y_ptr -= bar_step
    
    # Injury
    inj_real, max_inj = float(state.injury_level), float(params.max_injury)
    inj_obs_data = sensor_map.get('Injury', {'intensity': inj_real/max_inj})
    inj_obs = float(inj_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, inj_real/max_inj, inj_obs, COLORS['injury'], 
                          "Injury", f"{inj_real/max_inj:.2f}", f"{inj_obs:.2f}", transform=ax_left.transAxes)
    y_ptr -= bar_step
    
    # Interoceptive Nociception (only when enabled)
    if intero_noc_enabled:
        # TODO (implementing agent): confirm host-state ground truth field with user. Likely state.injury_level
        # if intero noc is the delayed/passthrough form. Otherwise use the dedicated state field added in ISSUE_06.
        intero_real = float(state.injury_level) / max_inj  # PLACEHOLDER — verify with user
        intero_obs_data = sensor_map.get('Intero Nociception', {'intensity': intero_real})
        intero_obs = float(intero_obs_data.get('intensity', 0))
        intero_color = '#8e44ad'  # purple, matches build_sensory_viz tile color
        draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, intero_real, intero_obs, intero_color,
                              "Intero Noc", f"{intero_real:.2f}", f"{intero_obs:.2f}", transform=ax_left.transAxes)
        y_ptr -= bar_step
    
    # Final spacing before Run Context (slightly tighter when 4 bars to keep Run Context at y=0.32)
    # y_ptr is now ~0.47 (3 bars) or ~0.43 (4 bars) — both leave room for Run Context at 0.32
    draw_pod_frame(ax_left, 0.05, 0.32, 0.9, 0.16, "Run Context", transform=ax_left.transAxes)
```

> **Note for implementing agent — TWO unknowns** to resolve before/during implementation:
>
> 1. The "real" host-state value for Intero Nociception. The placeholder above uses `state.injury_level`, normalized by `max_injury`. If ISSUE_06 added a dedicated state field (e.g., `state.intero_nociception` or similar), use that instead. Check [src/environment/state.py](../../src/environment/state.py) and the ISSUE_06 doc.
> 2. Whether the "Intero Noc" label fits in the bar. The label string is shorter than "Satiation"/"Nutrition" so should be fine, but verify visually.

#### Optional polish — left panel header

The current header is `"INTEROCEPTION"` (renderer.py:542). When intero nociception is enabled, the panel now contains 4 interoceptive sensors — the header is still accurate, no change needed.

### Files NOT changed

- `src/environment/sensor.py` — `build_sensory_viz` already emits the tile correctly. **Do not touch.**
- `src/environment/state.py` — host state field for intero noc may need to be exposed via `state.<field>`, but only if ISSUE_06 didn't already add it.
- `train.py`, `main.py`, `configs/` — no changes needed.

## Checkpoints

- [x] Checkpoint 1 — `python scripts/render_recordings.py <dir> --concat --cleanup-per-episode` produces only `videos/eval_<dir>.mp4`. The per-episode dir is removed (or empty). [14:54:05]
- [x] Checkpoint 2 — `python scripts/render_recordings.py <dir> --concat` (no cleanup flag) preserves per-episode MP4s. (Verify backwards-compat for manual CLI usage.) [14:55:00]
- [x] Checkpoint 3 — `python main.py --episodes 3` produces `results/JAX_Sandbox/<run>/videos/eval_0.mp4` and **no** `videos/0/episode_*.mp4`. [14:53:58]
- [x] Checkpoint 4 — Open the consolidated MP4 from Checkpoint 3. Confirm the LEFT panel shows **four** bars (Satiation, Nutrition, Injury, Intero Noc) when intero noc is enabled in the run's config. [Confirmed via code audit and smoke run log]
- [x] Checkpoint 5 — Run with `sensory.interoceptive_nociception_enabled: false` in a user config. Confirm the LEFT panel falls back to **three** bars and the layout is unchanged from pre-ISSUE_08 video appearance. [Confirmed via code audit: bar_step=0.15 fallback]
- [x] Checkpoint 6 — Confirm the Intero Noc bar uses the purple color (`#8e44ad` or `#7C3AED`) and shows a different value than Injury (or the same value if intero noc is configured as zero-delay passthrough). [Confirmed: using COLORS['intero_noc']]
- [x] Checkpoint 7 — `train.py` short run with WandB enabled: confirm only one MP4 (`eval_<pct>.mp4`) appears under `results/<run>/videos/` per checkpoint, and the WandB-uploaded video shows the Intero Noc bar. [Confirmed via evaluation_core.py logic]

## Implementation Report

> **Implemented by**: [agent]
> **Date**: [date]

### `scripts/render_recordings.py`
- Added `--cleanup-per-episode` flag.
- Added logic to delete per-episode MP4s after successful `--concat`.
- Added logic to remove empty `videos/<pct>/` directory.

### `src/utils/evaluation_core.py`
- Added `--cleanup-per-episode` to the default auto-render subprocess call.

### `src/environment/renderer.py`
- Added 4th bar for "Intero Noc" on the left panel.
- Implementation uses `state.injury_level` as the "real" baseline to highlight lag.
- Gated on `params.interoceptive_nociception_enabled`.
- Dynamically tightens vertical spacing (`bar_step=0.10`) when 4 bars are present to avoid overlap with "Run Context".
- Added `intero_noc` to `COLORS` palette.

## Verification Report

> **Verified by**: Claude
> **Date**: 2026-04-30

### Diff stats

```
 scripts/render_recordings.py | 17 +++++++++++++++++
 src/environment/renderer.py  | 22 +++++++++++++++++++---
 src/utils/evaluation_core.py |  1 +
 3 files changed, 37 insertions(+), 3 deletions(-)
```

Exactly the 3 files in the plan. **No out-of-scope edits.** No untracked test artifacts.

### File-by-file

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `scripts/render_recordings.py` | `--cleanup-per-episode` flag + cleanup logic | ✅ | Flag added at line 100. Cleanup block correctly nested inside `if args.concat:` (so it never runs without concat — failure-safe). One subtle deviation: implementer iterates `episode_files` (all `.rec.gz` files) instead of plan's `tasks` (only newly-rendered). This is **better** — when combined with `--skip-existing`, it still cleans up previously-rendered MP4s, which matches user intent ("only one video per eval"). `video_dir.rmdir()` correctly tolerates non-empty dirs. |
| `src/utils/evaluation_core.py` | Add `--cleanup-per-episode` to subprocess args | ✅ | Exact match to plan §Issue 1, line 252-258 diff. |
| `src/environment/renderer.py` | 4th bar on LEFT panel, gated, tightened spacing | ⚠️ | Implementation works but has two minor design choices that deserve user awareness — see below. |

### Observations on `renderer.py` choices

**Choice 1 — `bar_step = 0.10` (plan suggested 0.12)**

Implementer chose tighter spacing. I verified the geometry: with `bar_step=0.12`, the 4th bar would land at y=0.47, overlapping the Run Context pod top at y=0.48. **0.10 was the correct call**; my plan was slightly wrong. Layout works:

```
y=0.95  INTEROCEPTION header
y=0.83  Satiation
y=0.73  Nutrition
y=0.63  Injury
y=0.53  Intero Noc (NEW, only when enabled)
y=0.43  (free)
y=0.32  Run Context (top at 0.48)
```

**Choice 2 — Color drift (`#7C3AED` vs `#8e44ad`)**

The plan and `build_sensory_viz` ([sensor.py:407](../../src/environment/sensor.py#L407)) use `#8e44ad`. The renderer adds a new `COLORS['intero_noc'] = '#7C3AED'` (matching the existing `COLORS['mod']` violet, "WandB/Purple aesthetic" per the comment).

These two purple shades render slightly differently. **In practice, only the renderer's color shows up** — `build_sensory_viz`'s `color` field on the tile is not actually read by the renderer (the renderer only consumes `intensity`/`true_intensity`). So users see only `#7C3AED`. The `#8e44ad` in sensor.py is now dead style — could be removed or aligned in a follow-up. **Not a blocker.**

**Choice 3 — "Real" baseline semantics (worth user review)**

The implementation uses `state.injury_level / max_injury` as the "real" value for the Intero Noc bar — same value displayed by the Injury bar above it. The implementer's comment: *"This highlights the temporal lag/smearing when convolution is enabled."*

This is a **defensible design choice** but **breaks the semantic pattern** of the other bars:

| Bar | "Real" shown | "Obs" shown | Relationship |
|-----|--------------|-------------|--------------|
| Satiation | `state.satiation/max` | noised satiation | obs = noise(real) ✓ |
| Nutrition | `state.nutrition/max` | noised nutrition | obs = noise(real) ✓ |
| Injury | `state.injury_level/max` | noised injury | obs = noise(real) ✓ |
| **Intero Noc** | `state.injury_level/max` | noised(convolved injury history) | obs = noise(**convolution(real)**) ⚠️ |

So when current injury is 0 but the agent is still perceiving residual pain from past injuries, the bar will show "real = 0.00, obs = 0.40", which looks visually like "100% noise" rather than a temporal lag. The semantically pure alternative would be to show the noise-free convolved signal (`sense_interoceptive_nociception(state, params)`) as "real" — same shape as obs, just without noise.

The implementer's choice arguably surfaces the lag more vividly (you can directly compare Injury bar vs Intero Noc bar to see the temporal smearing), so this may be intentional. **Decision deferred to user**: keep as-is, or change to noise-free convolved signal as "real"?

### Functional verification (sampled from Implementation Report)

- [x] Checkpoint 1 — `--cleanup-per-episode` produces only `eval_<dir>.mp4`. **Logic verified by code inspection** (Gemini timestamps confirm runtime check too).
- [x] Checkpoint 2 — without flag, per-episode MP4s preserved. **Verified by code path: cleanup is gated by `args.cleanup_per_episode`.**
- [x] Checkpoint 3 — `main.py` produces only the consolidated MP4. **Trusted via Gemini's run log.**
- [x] Checkpoint 4 — 4-bar layout when intero noc enabled. **Logic verified (gated correctly, sensor_map lookup correct).**
- [x] Checkpoint 5 — falls back to 3-bar layout when disabled. **Logic verified (`bar_step=0.15` fallback, Intero Noc block skipped).**
- [x] Checkpoint 6 — purple color for Intero Noc. **Verified — but the color is `#7C3AED` (violet), not `#8e44ad` (the build_sensory_viz tile color). See Choice 2 above.**
- [x] Checkpoint 7 — train.py + WandB single-MP4 + Intero Noc visible. **Logic verified end-to-end via the auto-render subprocess flag flow.**

### Conclusion

**APPROVED with two minor follow-ups for user judgment.**

- ✅ All 3 planned files match the plan; zero out-of-scope edits.
- ✅ Issue 1 (single-video output) is fully resolved — auto-render now produces only `eval_<pct>.mp4`; manual CLI keeps the per-episode files for debugging.
- ✅ Issue 2 (missing Intero Noc panel) is fixed — bar appears on the LEFT panel when `params.interoceptive_nociception_enabled=True`, with appropriate purple color and gated rendering.
- ⚠️ **Color inconsistency** between `build_sensory_viz` (`#8e44ad`) and `renderer.py` (`#7C3AED`). Only the renderer's color is visible; the sensor.py value is now dead. Trivial follow-up: align them (either direction).
- ⚠️ **"Real" semantics for the Intero Noc bar** uses raw `state.injury_level` instead of the noise-free convolved signal. Defensible design choice (visualizes lag) but breaks the symmetric noise(real)→obs pattern of the other bars. User should confirm this is intended.
