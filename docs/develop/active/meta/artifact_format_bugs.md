---
title: Artifact format bugs — the register, and why reading the CSS never finds them
topic: meta
status: active
created: 2026-08-31
last_updated: 2026-08-31
---

# Artifact format bugs

## Why this document exists

The sensor-ladder artifact shipped with three numbered lists rendering **one word per line**, prose
spilling out of its column, and code chips overflowing their boxes. The page had been checked twice
before it shipped: once by me, once by a dedicated reviewing agent given a long, specific brief
about cascade collisions, grid containment, theme tokens and table overflow. That agent produced a
genuinely excellent report — it found a defect that made the whole page scroll sideways on every
phone — and it **did not find the one-word-per-line bug**. Neither did I.

The user found it in about one second, by looking at the page.

That is the lesson this document exists to encode. **Every format check before this one was static
analysis: reading HTML and CSS and reasoning about what they would produce. Chrome has been
installed in this container the whole time.** A defect that lives in the box tree rather than in the
stylesheet text is invisible to reading and obvious to rendering. The fix is not a better brief. It
is to render the page, measure it, and look at it.

## The protocol, from now on

Before publishing or republishing any artifact:

```bash
python scripts/claude/check_artifact_layout.py <page.html> --out tmp/artifact_layout
```

It wraps the page in the same skeleton the Artifact host injects at publish time, renders it in
headless Chrome at several viewport widths, and reports horizontal page overflow, boxes that escape
the viewport, text squeezed into an absurd column, elements with text but zero height, text
overlapping text, and image problems — then writes a full-page screenshot per width. **Exit code is
non-zero when it finds anything.**

Then **open the screenshots**. The measurements catch geometry; they do not catch ugly. Both steps
are required, and the second one is the one that found the bug that started this document.

Then hand the page to the [`artifact-format-reviewer`](../../../../.claude/agents/artifact-format-reviewer.md)
agent, which runs the tool, reads this register, and looks at the rendering.

### Known limits of the tool

- **Chrome headless floors the viewport at 500px.** Asking for 390 silently renders 500. Genuine
  phone-width layout is therefore not covered; the tool says so when you ask for less.
- It renders a *lean* copy with the inlined images swapped for same-aspect placeholders, so the run
  is fast. Text layout is identical; anything about the image content itself is not tested.
- It knows nothing about whether the design is good, whether a colour means two different things,
  or whether an axis label is wrong. Those need a reader.

---

## The register

Each entry is a defect that actually shipped or nearly shipped, what a reader saw, why the static
review missed it, and the rule that would have caught it.

### F1 — `display:grid` turns bare text into anonymous grid items

**Saw:** three numbered lists rendering one word per line in a ~40px column, with `<code>` chips
overflowing to the right and an `<em>` overlapping the word before it.

**Cause:** `.caveats li{display:grid;grid-template-columns:26px 1fr}` where the `<li>` content was
`<strong>…</strong> text <em>…</em> text <code>…</code> text`. In a grid container **every
contiguous run of bare text becomes its own anonymous grid item**. The author counted two items —
the `::before` counter and "the content" — but there were seven, so the prose alternated between the
26px counter column and the content column.

**Why reading missed it:** the CSS is correct in isolation and the HTML is valid. `<strong>` inside
`<li>` is fine; `display:grid` is fine. The defect exists only in the box tree, which requires
enumerating the element's *children* — including text nodes — and applying the anonymous-item rule.
Two sibling lists on the same page used the same CSS and looked fine, because their content happened
to be wrapped in a single child element, which made the pattern look proven.

**Rule:** never put `display:grid` or `display:flex` on an element whose children include bare text.
For a counter-and-content list, place the counter with `position:absolute` inside a
`position:relative` list item; it is immune to whatever inline markup the content contains.

### F2 — a grid item's automatic minimum size is its content minimum

**Saw:** the entire page scrolled sideways — 800px of drag on a phone, 110px even at 1300px.

**Cause:** `<main>` was a grid item in a `1fr` track. A grid item's automatic minimum size is its
*content* minimum, and `figure img{width:100%}` contributes the image's **intrinsic** width during
intrinsic sizing (a percentage width resolves to `auto` there). The inlined PNGs are 2000–3600px
wide, clamped by `.wide{max-width:1120px}`, so the track could never be narrower than 1120px.
`img{max-width:100%}` does not help — a percentage max-width is also ignored for intrinsic sizing.

**Rule:** every grid or flex item that contains content wider than the layout gets `min-width:0`.

### F3 — a utility class beats a type selector, silently

**Saw:** every figure and every section divider lost its vertical margin; figures butted against the
prose above them.

**Cause:** `.col{margin:0 auto}` and `.wide{margin:0 auto}` are class selectors (0,1,0) and beat
`figure{margin:44px 0}` and `hr{margin:70px 0}` (0,0,1) regardless of source order. The shorthand
claimed all four sides.

**Rule:** a centring utility sets `margin-left`/`margin-right` only, never the `margin` shorthand.

### F4 — `overflow:auto` clips the caption too

**Saw:** scrolling a wide table sideways carried its own title off the left edge, leaving a floating
border.

**Cause:** `<caption>` is a child of `<table>`, which was inside the `overflow-x:auto` box.

**Rule:** a scroll container holds only the thing that scrolls. Titles, captions and controls are
siblings of it.

### F5 — `position:sticky` constrains the margin box, so padding pushes the element down

**Saw:** the sticky table of contents floated 130px below the top of the window, permanently.

**Cause:** `.toc{position:sticky;top:34px;padding-top:96px}`. The sticky offset applies to the
margin box, so the content began at 34+96. Moving it to `margin-top` does not help either.

**Rule:** when a sticky element has leading padding, subtract it from `top`.

### F6 — `width:100%` inside a scroll box squeezes instead of scrolling

**Saw:** on a narrow screen, 50-character table headers crushed into 100px columns, five lines tall,
while the scroll container never scrolled.

**Cause:** `table{width:100%}` fills the `.scroll` box exactly, so there is never any overflow to
scroll.

**Rule:** a data table inside a scroll container is `width:max-content;min-width:100%` — natural
width, at least filling the box.

### F7 — `scroll-margin-top` has to be on the element the anchor targets

**Saw:** clicking a table-of-contents link parked the heading flush against the top of the window.

**Cause:** the property was on `h2`; every anchor targeted the enclosing `<section>`.

**Rule:** put it on `[id]`, or on whatever actually carries the anchor.

### F8 — a percentage `max-width` does not stop an unbreakable string

**Saw:** a 54-character file path overflowed the text column on a narrow screen.

**Cause:** inline `<code>` with no wrapping opt-in.

**Rule:** `code{overflow-wrap:anywhere}`.

### F9 — the host declares `color-scheme: light` and nothing else

**Saw:** bright light-mode scrollbars on dark panels in dark mode.

**Rule:** declare `color-scheme:light dark` on `:root` and pin it in each explicit theme block.

### F10 — panels a reader is asked to compare, drawn on different scales

**Saw:** a three-panel figure whose entire purpose was "these three answers differ" gave each panel
its own y-range, so the panel showing the *wrong* answer looked as strong as the right one.

**Rule:** panels presented as a comparison share a scale. If one then looks flat, that is the
finding.

### F11 — one colour, five meanings

**Saw:** red meant sign-of-change, predator, cause-of-death, heavy-wound, and
cannot-resolve-identity — twice within a single image.

**Rule:** fix the colour→meaning map once, in the shared plotting module, with a comment. A reader
who learns a colour on figure 3 must not be punished for carrying it to figure 8.

### F12 — more than about six series in one colour ramp

**Saw:** 14 and 28 lines in a continuous viridis ramp, where adjacent ramp colours were exactly the
series the text asked the reader to tell apart.

**Rule:** colour by the distinction the argument turns on, draw the rest thin, and label the
discussed series directly on the line.

### F13 — per-row annotations not reordered when the rows were

**Saw:** every value label in the report's headline figure sat on the wrong bar. The 166-step bar
was labelled 250; the longest bush-dwell bar was labelled with the smallest number.

**Cause:** fixing F-something-else (the y-axis said "poorest at the bottom" while plotting the
poorest at the top) meant changing the bar order from `np.arange(n)[::-1]` to `np.arange(n)`. The
bars moved. The separate annotation loop still placed its text at `len(arms) - 1 - i`, which was
correct only for the old order.

**Why nothing caught it:** the figure is internally plausible — the bars are right, the axis labels
are right, and every number printed is a real number from the data. Only cross-referencing a label
against the bar it sits on reveals it. The geometry checker cannot see it (nothing overlaps or
overflows), and a reader who does not already know the result will not notice.

**Rule:** draw an annotation from the same row object as its bar, never from a parallel index. Where
that is impractical, **read both back off the axes and assert they agree** — `lad01_ladder_overview.py`
now does exactly this, comparing every `axis.texts` entry against the `axis.patches` bar at the same
y, and refusing to write the figure if any disagree.

### F14 — an author `display` rule defeats the `hidden` attribute

**Saw:** the full-screen figure viewer rendered open on page load, dimming the whole article, in any
host that does not mark its `[hidden]` rule `!important`.

**Cause:** `.lb{display:flex}` on an element whose only closing mechanism is the `hidden` attribute.
The UA's `[hidden]{display:none}` loses to any author `display` rule. The layout checker's own
skeleton *does* use `!important`, which is precisely why its screenshots looked fine — the bug was
invisible to the tool that should have caught it.

**Rule:** every element toggled with `hidden` carries its own `.x[hidden]{display:none!important}`.
Never rely on the host's rule being `!important`.

### F15 — prose repeats a figure's number by hand, and the number moved

**Saw:** after the evaluation sample was tripled and every figure regenerated, about a dozen
sentences still quoted the old values — mostly 0.1–0.4 off the figure sitting beside them, but one
range wrong by 1.3 (a panel described as collapsing "to between &minus;0.0 and &minus;0.7" actually
ran &minus;2.0 to +0.0). One alt-text string still asserted a claim the correction box two paragraphs
below explicitly withdrew, so a screen-reader user got the retracted version.

**Why nothing caught it:** the tables and figures are generated, so they were all correct and
mutually consistent. Only the hand-written sentences drifted, and each one is individually
plausible — you cannot spot it without cross-reading every quoted number against the figure or table
it summarises. The geometry checker sees text, not meaning. A first review that fixed four instances
and stopped left ten behind.

**Rule:** a number that appears in both a figure and a sentence should be emitted by the analysis
code into both, or the sentence should quote a table cell verbatim. Where prose genuinely must
restate a value, a republish after any data regeneration re-scans **every** number in the prose and
in **all** alt text — not only the ones flagged last time. Beware of claiming a page is fully
generated when only its tables are: this report's Method section said "every number on this page is
generated, not transcribed" while a dozen transcribed numbers were stale.

### Tool note — a truncated screenshot is not a review

`check_artifact_layout.py --shot-height` defaulted to 24,000px while these pages run 32,000–37,000px
tall, so the bottom third was never rendered to an image, and a review that only looked at the
screenshots would have silently skipped Sections 6–8, the Method and the Limitations. The
measurements always covered the whole DOM; only the pictures were short. The tool now reports the
page height, and **counts a truncated capture as a problem** with the `--shot-height` value needed to
fix it.

---

### F16 — a regex that edits one of N repeated elements, and reaches into the next

**Saw:** one figure shipped with no "How it is computed" block, and no error anywhere.

**Cause:** a `re.DOTALL` pattern of the form `FIG:lad03.*?<div><h5>How it is computed</h5>...` was
used to replace that figure's method block. Figure 3 had no such block, so `.*?` ran forward into
figure 4 and replaced *its* block instead. Figure 4's own pass then rewrote it correctly, leaving no
trace except a silently missing block.

**Rule:** to edit one of N repeated structures, first slice out that element (`re.finditer` over
`<figure ...>.*?</figure>`, pick the one you want, edit inside it, splice back), so no pattern can
span two of them. And assert the postcondition: the build now fails if any figure lacks an axes
sentence or a method block.

### F17 — content inside a closed `<details>` is laid out but never painted

**Saw:** the layout checker reported 24 text-on-text overlaps on a clean page, all of them a table
cell inside a collapsed panel against the paragraph above it.

**Cause:** a closed `<details>` does not paint its content, but the content still has a layout box —
`getComputedStyle` reports `display: table-cell` and `getBoundingClientRect` returns real
coordinates. Any geometry check that walks the DOM will see it.

**Rule:** a renderer-based checker must skip content inside a closed `<details>`, the same way it
skips `display:none`. `check_artifact_layout.py` now does.

### F18 — an axis label wider than its panel, in a multi-panel figure

**Saw:** in a two-panel figure the two x-axis labels printed through each other mid-canvas, reading
`...below zero = hides LESS when o**aneDIsFFEeRE**NCE in percentage points...`, and the right one
ran off the canvas edge mid-word. Three figures were affected.

**Cause:** the labels had just been *lengthened* — to fix a different defect, F15's cousin, where
"percentage points" alone did not tell the reader an axis was a difference. Matplotlib places an
xlabel centred under its axes and neither wraps nor warns when it is wider than the panel; in a
multi-panel figure it simply overlaps the neighbour.

**Why nothing caught it:** the defect is inside the PNG. The DOM-based layout checker sees an image,
not the text drawn in it, and matplotlib emits nothing.

**Rule:** measure every axis label's rendered extent against its own panel at draw time and refuse
to write the figure on overflow — the same read-back-and-assert pattern F13 mandates for bar
annotations. `_plot.assert_labels_fit(fig, ax)` does this and is called by all fifteen figure
scripts; it immediately caught a fourth figure the human reviewer had not flagged.

### F19 — a collapsed panel is never geometry-checked

**Saw:** nothing, for a while — which is the problem. Once the layout checker was taught to skip
content inside a closed `<details>` (F17), that content stopped being checked at all, and a real
defect sat inside one undetected: at a 500px viewport the notes column of every data panel was
142px wide, crushing a 214-character sentence into a ribbon.

**Cause:** the F17 fix was correct but one-sided. Skipping unpainted content removes the false
positives and the true ones together.

**Rule:** a checker that skips collapsed content must also offer a pass that expands it.
`check_artifact_layout.py --open-details` forces every `<details>` open and re-runs the geometry;
run both passes before publishing. Relatedly, the `<summary>` element itself *is* painted when the
details is closed and must stay in the default pass.

## Checklist

- [ ] `check_artifact_layout.py` exits 0
- [ ] The screenshots have been **opened and looked at**, not just generated
- [ ] No `display:grid` / `display:flex` on any element with bare text children (F1)
- [ ] Every grid/flex item that can hold wide content has `min-width:0` (F2)
- [ ] No centring utility uses the `margin` shorthand (F3)
- [ ] Scroll containers hold only the scrolling thing (F4)
- [ ] Sticky offsets account for the element's own padding (F5)
- [ ] Data tables are `width:max-content;min-width:100%` (F6)
- [ ] `scroll-margin-top` is on the anchored element (F7)
- [ ] Long unbreakable strings can wrap (F8)
- [ ] `color-scheme` declared in all theme paths (F9)
- [ ] Compared panels share a scale (F10)
- [ ] One meaning per colour across the whole figure set (F11)
- [ ] No more than ~6 series distinguished by colour alone (F12)
- [ ] Every per-row annotation verified against the row it sits on (F13)
- [ ] Every `hidden`-toggled element carries its own `[hidden]{display:none!important}` (F14)
- [ ] The screenshots are as tall as the page — the tool now says so
- [ ] Every number quoted in prose AND in alt text re-checked against its figure (F15)
- [ ] Edits to one of N repeated elements are scoped to that element (F16)
- [ ] Geometry checks skip closed `<details>` content (F17)
- [ ] Every axis label measured against its own panel at draw time (F18)
- [ ] The checker run twice: default, and `--open-details` (F19)
- [ ] Every figure has exactly one generating script, and the page says which

## Related

- [`artifact_generation_guide`](artifact_generation_guide.md) — the wider guide: content, claims,
  jargon, figure legibility. This document is the *format* half and is narrower on purpose.
- `scripts/claude/check_artifact_layout.py` — the tool.
- [`artifact-format-reviewer`](../../../../.claude/agents/artifact-format-reviewer.md) — the agent
  that runs it.
