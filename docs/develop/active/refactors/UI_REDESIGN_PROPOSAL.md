---
title: Renderer UI Redesign — Dashboard
topic: refactors
status: active
created: 2026-04-24
last_updated: 2026-04-24
---

# Renderer UI Redesign — Dashboard

> **Status**: IN PROGRESS — first layout rendered, next iteration planned
> **Opened**: 2026-04-22
> **Updated**: 2026-04-22 (second design iteration)
> **Related**: [docs/environment/12_renderer.md](../environment/12_renderer.md), [NOISE_RENDERING_RENDERER_FIX.md](NOISE_RENDERING_RENDERER_FIX.md)
> **Reference frame**: [tmp/v2_frames/step_0010.png](../../tmp/v2_frames/step_0010.png)

---

## Context

A first redesign replaced the manual-coord layout with a declarative `subfigures` + `subplot_mosaic` engine, consolidated run-context into a header banner, and introduced per-card backgrounds. The live output is now in `tmp/v2_frames/`.

Review of the rendered frames surfaced concrete issues that were invisible at the mock-up stage. This iteration responds to those issues with a consolidated left panel, a larger minimap, simpler dual-view numerics, and removal of the dedicated action pod.

The brief remains unchanged:
1. Keep the central arena as the primary visual.
2. Preserve all sensor content.
3. Layout must be declarative — no manual coordinate arithmetic.

## Design Feedback (from rendered frames)

| # | Issue | Evidence frame | Root cause |
|---|-------|----------------|-----------|
| 1 | Minimap too small to read entity positions | `step_0010.png` | Minimap axes sits at ~13% vertical height of its column; dots are 3 px |
| 2 | REAL text (`real 0.43 · Δ +0.01`) too small, too far from OBS | any vital card | REAL is a 6 pt footnote at the bottom; the OBS value is an 18 pt readout at the top — eye must jump |
| 3 | Δ text is visual clutter | all vital + nociception cards | Redundant with OBS/REAL numbers; reader can subtract mentally |
| 4 | Collision direction labels (`C U R D L`) too small | `step_0010.png` right column | Inherited 5.5 pt font from `draw_categorical_visual`, placed inside bar frame |
| 5 | Visual / Olfactory bars touch pod ceiling | `step_0001.png` olfactory, `step_0010.png` visual | Bar drawing area `bar_h = 0.65` is too tall relative to card height; normalized max bar reaches top edge |
| 6 | Current Action pod wastes a full slot | bottom-right | Single word + arrow glyph in a pod the size of a sensor — action should be a badge |

Additional standing rule from the user: **the three interoception cards may be consolidated into a single bar widget** if that frees enough space to fix issue #1 (minimap).

## Revised Design

Design direction, one per feedback item.

### 1. Consolidated Vitals + large Minimap in left column

Fold Satiation / Nutrition / Injury into a single `VITALS` card with three compact rows. The space freed by merging three cards into one is given entirely to the minimap, which doubles its area and gains readable entity dots.

New left column proportions:
- `VITALS` card — ~45% of column height (was 3 × ~30% = ~90%)
- `MINIMAP` card — ~55% of column height (was ~25%)

Inside the VITALS card each row is a single horizontal strip:

```
SAT    0.90  (real 0.95)   ▓▓▓▓▓▓▓▓▓░░
NUT    0.75  (real 0.77)   ▓▓▓▓▓▓░░░░░
INJ    0.71  (real 0.45)   ▓▓▓▓▓▓▓▓░░░
```

Label left, numbers middle, bar right. REAL sits in parentheses directly next to OBS — a single horizontal scan reads both. No Δ text. The bar still carries a thin vertical tick marker at the REAL position for visual reference (this is a mark, not text, so it doesn't violate the "no Δ" rule — and it is what keeps the dual view readable at a glance).

### 2. REAL value adjacent to OBS — everywhere

Intensity and spectrum pods on the right column apply the same inline treatment:

```
EXTERO NOCICEPTION
OBS  0.66   REAL  0.90
▓▓▓▓▓▓▓░░░░░░  ┃   (tick at real)
```

OBS and REAL are the same font size (~10 pt), stacked side-by-side at the top of the card. OBS stays in the accent color (cobalt); REAL is in muted grey. No subtitle, no Δ.

### 3. Δ text removed project-wide

All `Δ ±X.XX` lines deleted from vital rows and intensity pods. Dual view is carried by:
- the bar (OBS fill),
- the tick marker (REAL position),
- the inline `(real X)` numeric.

### 4. Collision / Visual: legend above bars, larger font

Move the spatial direction labels (`C U R D L`) and feature labels out of the tiny rotated-90° slot inside the bar frame. New treatment:

- A horizontal legend strip sits at the top of the pod, below the title: `C    U    R    D    L` at 8 pt, horizontal, evenly spaced above the corresponding bar columns.
- Feature labels (for Visual pod) use the same 8 pt horizontal strip — one row of letters above the bars.
- The rotated-inside-bar labels are removed entirely.

### 5. Bar-height headroom in spectrum / visual pods

Reduce the normalized drawing area from `bar_h = 0.65` to `bar_h = 0.48` of the card height. The tallest bar now caps at ~60% of the card vertical, leaving clear air above. Applies to Olfactory, Collision, and Visual pods.

### 6. Action badge — integrated into arena corner

Delete the standalone `CURRENT ACTION` pod. The action moves to a pill badge overlaid on the **top-right corner of the Arena card**:

```
┌─────────────────────────────────────┐
│ ARENA — LOCAL VIEW         [ ↓ DOWN ]│   ← badge, rounded, cobalt accent
│                                      │
│              (grid)                  │
```

Rationale: the action is an action *about the arena* — in-context placement is more meaningful than a sidebar pod. The right column gains a row that can be redistributed to the four remaining sensor pods.

### Right column redistribution after action removed

Olfactory, Nociception, Collision, Visual — 4 pods share the full right-column height instead of 4.55 pods. Each pod gets ~25% taller, which:
- gives Olfactory more headroom for the new shorter `bar_h`,
- makes Collision directional labels legible without cramping,
- makes Visual's 8-feature bars individually readable.

## Layout Blueprint

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ GridWorld · Homeostasis Dashboard              Ep 1 · 10×10 · Step 23        │ HEADER
├──────────────────┬────────────────────────────────────┬───────────────────────┤
│ VITALS           │ ARENA — LOCAL VIEW      [↓ DOWN]   │ OLFACTORY             │
│ ┌──────────────┐ │ ┌────────────────────────────────┐ │ ┌───────────────────┐ │
│ │SAT 0.90 (0.95)│ │ │                                │ │ │ ▓ ▓▓ ▓ ░           │ │
│ │ ▓▓▓▓▓▓▓▓▓░░┃ │ │ │                                │ │ └───────────────────┘ │
│ │NUT 0.75 (0.77)│ │ │       LOCAL GRID               │ │ EXTERO NOCICEPTION    │
│ │ ▓▓▓▓▓▓░░░░┃  │ │ │       (5×5 around agent)       │ │ ┌───────────────────┐ │
│ │INJ 0.71 (0.45)│ │ │                                │ │ │ OBS 0.66  REAL 0.9│ │
│ │ ▓▓▓▓▓▓▓▓┃░░  │ │ │                                │ │ │ ▓▓▓▓▓▓▓░░ ┃        │ │
│ └──────────────┘ │ │                                │ │ └───────────────────┘ │
│                  │ │                                │ │ COLLISION             │
│ ┌──────────────┐ │ │                                │ │ ┌───────────────────┐ │
│ │   MINIMAP    │ │ │                                │ │ │ C   U   R   D   L │ │
│ │              │ │ └────────────────────────────────┘ │ │ ·   ·   ·   ·   ▓ │ │
│ │  (enlarged,  │ │                                    │ └───────────────────┘ │
│ │  readable)   │ │                                    │ VISUAL                │
│ │              │ │                                    │ ┌───────────────────┐ │
│ │  █ agent     │ │                                    │ │G S P F D R N Rk   │ │
│ │  ▫ viewport  │ │                                    │ │▓ ░ ░ ░ ░ ▓ ▓ ░    │ │
│ └──────────────┘ │                                    │ └───────────────────┘ │
└──────────────────┴────────────────────────────────────┴───────────────────────┘
```

### Named mosaic (implementation spec)

```python
left_axes = sf_left.subplot_mosaic(
    [['vitals'], ['minimap']],
    gridspec_kw={'hspace': 0.08, 'height_ratios': [1.0, 1.2]}
)

center_axes = sf_center.subplot_mosaic(
    [['arena']],
    gridspec_kw={}
)

right_axes = sf_right.subplot_mosaic(
    [['olfactory'],
     ['nociception'],
     ['collision'],
     ['visual']],
    gridspec_kw={'hspace': 0.35, 'height_ratios': [1, 1, 1.2, 1.2]}
)
```

No `action` key — the action badge is drawn directly onto `center_axes['arena']` at `(0.97, 0.97)` axes-fraction, as an overlay annotation.

## Typography & Color

Carry over the existing palette (`COLORS` dict) unchanged.

Type changes from current implementation:
- OBS value in vitals rows: 10 pt (was 14 pt — smaller because label + two numbers + bar share one row)
- REAL value in vitals rows: 10 pt muted grey, displayed as `(0.95)` next to OBS
- Legend letters (C U R D L, feature labels): **8 pt, horizontal** (was 5.5 pt rotated 90°)
- Action badge: 9 pt text + 12 pt arrow, rounded pill at `COLORS['card_bg']` with `COLORS['action']` text

## Implementation Plan

All changes live in `src/environment/renderer_v2.py`. No other files touched.

### A. New `draw_vitals_panel(ax, state, params, sensor_map)`

Replaces three calls to `draw_vital_card`. Draws a single card containing 3 row blocks, each with the `LABEL  OBS (REAL)  BAR` layout. Each bar still carries the REAL tick marker; no Δ text.

Row geometry inside the card (axes-fraction):
- Title strip: `y = 0.92–1.00` (card title: `VITALS`)
- Row 1 (SAT): `y = 0.62–0.88`
- Row 2 (NUT): `y = 0.33–0.59`
- Row 3 (INJ): `y = 0.04–0.30`

Each row draws: label at `x=0.04`, OBS text at `x=0.22`, REAL text at `x=0.42`, bar from `x=0.60` to `x=0.96`.

### B. Delete `draw_action_pod` and `action` mosaic entry

Remove the `action` row from the right-column mosaic. Delete the call to `draw_action_pod`. Add an overlay on `ax_arena` after arena rendering finishes:

```python
if action is not None:
    name = action_names.get(int(action), str(action))
    arrow = arrows.get(int(action), '•')
    ax_arena.text(
        0.97, 0.96, f' {arrow} {name} ',
        transform=ax_arena.transAxes,
        ha='right', va='top',
        fontsize=9, fontweight='bold',
        color=COLORS['action'],
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor=COLORS['card_bg'],
                  edgecolor=COLORS['arena_border'],
                  linewidth=0.8)
    )
```

### C. Rewrite `draw_intensity_pod` header row

Replace the current layout (label top-left, big OBS top-right, Δ bottom) with:

```
Title row (y=0.88):   LABEL (OBS ONLY if applicable)
Value row (y=0.72):   OBS 0.66    REAL 0.90
Bar row  (y=0.30):    [bar with tick]
(no subtitle row)
```

Both OBS and REAL numbers at 10 pt: OBS in `COLORS['action']`, REAL in `COLORS['text_label']`.

### D. Rewrite `draw_spectrum_pod` — reduce `bar_h`

```python
bar_h = 0.65   →   bar_h = 0.48
bar_y = 0.14   →   bar_y = 0.10
```

The normalized-max bar now tops out at `0.10 + 0.48 = 0.58` axes-fraction, leaving room for the title without cramping.

### E. New in-module categorical routine, replacing `draw_categorical_visual`

Stop calling `draw_categorical_visual` from `renderer.py` (which has the 5.5 pt rotated labels baked in). Write a small inline categorical routine in `renderer_v2.py` with:
- Horizontal legend strip at `y = 0.72–0.82` showing direction or feature labels at 8 pt, evenly spaced over bar columns.
- Bars drawn in `y = 0.08–0.68` (cap at 0.60 of card height).
- No rotated text inside bar frames.

### F. Left column: add MINIMAP row, move VITALS into single card

Left mosaic becomes `[['vitals'], ['minimap']]` with height ratio `[1.0, 1.2]`. Centre mosaic becomes a single-axes `[['arena']]`; the minimap is no longer in the centre column.

Minimap card inner dimensions are roughly 200 × 240 px at 14×8-inch figure / 100 DPI — more than 3× the pixel area of the current minimap. Scale the entity dots from 3 px to 6 px and bump the viewport rectangle line width from 1.0 to 1.5.

## Checkpoints

- [ ] **CP1** — Single frame renders without errors. Arena is full-height (centre column). Minimap is visibly larger (more than 2× previous area).
- [ ] **CP2** — Vitals card shows three rows in one card, each row reads `LABEL  OBS (REAL)  BAR` with tick marker at REAL. No Δ text anywhere.
- [ ] **CP3** — Action badge visible at top-right of arena. Shows `↓ DOWN` (or current action). Standalone action pod is gone.
- [ ] **CP4** — Collision and Visual pods show a horizontal legend strip (letters 8 pt, non-rotated) above the bars. No text inside or rotated below bar frame.
- [ ] **CP5** — Olfactory / Visual bars visibly stop below the card title (clear vertical air between tallest bar and title).
- [ ] **CP6** — Grep `src/environment/renderer_v2.py` for `Δ` and ` delta `. Expect zero matches outside comments.
- [ ] **CP7** — Re-run `tmp/test_v2_render.py`. Inspect `tmp/v2_frames/step_0010.png` against the layout blueprint above.

## Open Questions

1. **Minimap position**: below VITALS in left column (this proposal) vs keep below arena in centre column but make it wider. Left-column placement frees the centre for a full-height arena but visually separates the minimap from the arena it samples. Preference: left-column, because the arena already implies "where the agent is looking" while the minimap answers "where in the world is everything" — a different question that does not need spatial adjacency to the arena.
2. **Badge style**: cobalt-on-cream pill (proposed) vs accent-colored solid button vs transparent with border only. Cobalt-on-cream reads clearly on the pale-green arena background without competing with entity icons.
3. **REAL number display when OBS ONLY**: should the REAL slot show `—` or be omitted? Proposed: show a pill `(OBS ONLY)` in the title row and omit the REAL number entirely in the value row.
4. **Feature label set for Visual pod**: current code uses `['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']`. At 8 pt horizontal those may still clip — consider single letters `G S P F D R N K` if width is tight.

## Implementation Report

> **Implemented by**: [TBD]
> **Date**: [TBD]

## Verification Report

> **Verified by**: [TBD]
> **Date**: [TBD]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [one-line summary]
