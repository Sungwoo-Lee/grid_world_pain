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

> **Amended after F24 — this rule holds only for tables whose columns are all short or numeric.**
> F6 and F24 are the two ends of one trade-off, and the register has to say which applies when.
> Under `max-content`, a cell containing a *sentence* claims that sentence's full unwrapped width, so
> the table outgrows its scroll box and the **last column disappears at desktop width** (F24) — a far
> worse failure than F6's, because F6 is visible and F24 is not. The split:
>
> | Table | Rule |
> |---|---|
> | every column short or numeric | `width:max-content; min-width:100%` (F6 as written) |
> | any column carries prose | `width:100%` **plus** a `min-width` floor (≈560px), `nowrap` on the numeric cells only |
>
> The `min-width` floor is what keeps the prose case from falling back into F6's ribboning: below the
> floor the box scrolls (signalled, expected on a phone) instead of crushing prose into one-to-two-word
> lines. Never put `white-space:nowrap` on `th` — a long header forces the same overflow F24 describes.

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
- [ ] Breakout wrappers are siblings of the column, not children (F20)
- [ ] Mono-block column alignment uses `&nbsp;` or `pre`, not plain spaces (F21)
- [ ] Numeric table columns are right-aligned, headers included (F22)
- [ ] Every figure has exactly one generating script, and the page says which

### F20 — a nested max-width silently caps a designed-wider element

**Saw:** a four-step pipeline diagram given its own `.wide` wrapper (`max-width:940px`) to break out
of the 720px prose column rendered its step columns at ~130px — three words per line — at *every*
desktop viewport. The layout checker flagged "squeezed column" four times at 834/1100/1440 but named
the symptom, not the cause.

**Cause:** the breakout wrapper was placed **inside** the column it was meant to escape. A child can
never exceed its parent's `max-width`, so the wrapper was inert and the diagram silently inherited
720px. Nothing in the stylesheet looks wrong; the defect lives entirely in the nesting.

**Rule:** a breakout wrapper must be a **sibling** of the column, never a child — close the column
element, emit the wide block, reopen the column. Verify by measuring the element's *rendered* width
against its own `max-width`; if they disagree, an ancestor is capping it. And because closing and
reopening a wrapper is exactly where a spacing artefact would appear, check the **margins at both
boundaries** after the fix, not just the width.

### F21 — column alignment built from collapsible spaces

**Saw:** a monospace block aligning two labelled values with runs of plain spaces
(`value  =  29.0` / `value   =   1.2`) rendered with its `=` signs and values off by a character at
every viewport width. A second block in the same page, built with `&nbsp;`, aligned correctly.

**Cause:** HTML collapses consecutive whitespace. Monospace makes the *glyphs* equal width, which is
easy to mistake for alignment being handled — but the spacing between them is still collapsed to one
space.

**Rule:** in a mono block, build column alignment from `&nbsp;` runs or `white-space: pre`, never
plain spaces. The symptom is subtle — it reads as sloppiness rather than as an error — so it survives
proofreading and only shows up on render.

### F22 — `tabular-nums` mistaken for column alignment

**Saw:** a worked-example table whose numeric columns held mixed signs and digit counts
(`0.58`, `13.95`, `−7.32`, `4.25`) rendered with the decimal points drifting down every column.
`font-variant-numeric: tabular-nums` was set, and the cells were monospace, so the alignment
looked handled.

**Cause:** tabular figures equalise the *width of each glyph*; they say nothing about where the
number starts. A **left-aligned** numeric column still begins every value at the same left edge, so
a minus sign or an extra integer digit shifts the decimal point. Monospace plus tabular-nums makes
the drift look deliberate rather than broken, which is why it survives proofreading.

**Rule:** numeric table columns are **right-aligned** (`td.num, th.num { text-align: right }`) — with
a fixed number of decimal places that right-alignment decimal-aligns for free. Pad with a figure
space (U+2007) only where right-alignment is not wanted. `tabular-nums` is necessary but not
sufficient; align the header cell too, or the column reads as detached from its label.

### F23 — a component's `p{margin:0}` reset, written when the component only ever held one paragraph

**Saw:** a `.correction` callout on the sensor-ladder page, holding a three-paragraph argument,
rendered as one unbroken wall of text. The two paragraph breaks got exactly the within-paragraph
line pitch — measured 0 px gap against a 23.6 px line-height — so the reader saw a block where the
author had written three steps. A pre-existing two-paragraph instance of the same component had the
defect too, unnoticed, because nobody had read it closely.

**Cause:** `.correction p{margin:0}` was written when every instance of the component held exactly
one paragraph, where the reset is correct and invisible. The first instance to hold several inherits
it silently. Nothing overlaps, nothing is clipped, nothing squeezes, and the CSS reads as deliberate
in source review — so **the geometry checker structurally cannot see this**, and neither can reading
the stylesheet. Only looking at the render finds it.

**Rule:** any component that resets `p{margin:0}` must also set `p + p{margin-top: …}`. More
generally: a reset written for a single-child case is a latent defect the first time the component
takes a second child. When adding a paragraph to an existing callout, card or note component, render
it — do not assume the component's spacing was designed for more than it had.

**Verifying a fix:** measure the gap in the box tree rather than trusting the rule was added — and
check the measurement fires by removing the rule and confirming it reports 0 px. A spacing assertion
that has never been seen to fail is not evidence.

### F24 — `width:max-content` on a table that has a prose column hides the last column, at every width

**Saw:** a new results section whose headline table listed three training arms and their scores. The
score column — `138.2 ± 3.5 / 40.3 ± 0.8 / 41.9 ± 0.7`, the entire point of the section — was off the
right edge of its scroll box at **1440 px**, not merely on a phone. The reader saw a header clipped to
`SURVIVAL` with nothing beneath it. A sweep of the same page then found **eight pre-existing tables**
with the identical defect, including one in an appendix whose third column had never been visible to
any reader at any viewport width since the page was first published.

**Cause:** `table{width:max-content;min-width:100%}` (the F6 rule) tells the table to take its natural
width, and a cell containing a sentence has a natural width of *that sentence on one line*. The rule is
right for the short numeric cells it was written for, and looks proven by the ten other tables on the
same page that use it correctly. Add one prose column and the table silently outgrows the 720 px text
column; the `.scroll` wrapper then does its job and hides the overflow. A long `th` under
`white-space:nowrap` causes the same thing on an all-numeric table.

**Why nothing caught it:** the geometry checker **deliberately exempts any element overflowing its own
scroll container** — that is normally correct behaviour, since a scroll box is supposed to scroll — so
it reports nothing at all. Reading the CSS finds nothing either: the rule is correct in isolation and
has many working instances above the broken one. And with `--hide-scrollbars` set for screenshots, the
render carries no visible cue; the column is simply absent.

**Rule:** `width:max-content` only for tables whose cells are **all short**. Any table with a prose
column is `width:100%`, with `white-space:nowrap` left on the numeric cells so the prose column absorbs
the wrapping. Do not put `white-space:nowrap` on `th` — let long headers wrap to two lines rather than
push a column off-screen.

**Verifying a fix:** the checker's silence is not evidence here. Measure `scrollWidth − clientWidth` on
every scroll container at the **widest** viewport and require it to be zero. Overflow at 1440 px means
the table was never designed to fit — it does not mean the scroll box is working. Some overflow at
phone width is acceptable and expected; overflow at desktop width is the defect.

### F25 — a legibility `min-width` floor carried from one diagram's viewBox to another's

**Saw:** a hand-authored SVG whose labels rendered at **6.6 px** on a phone, directly beneath a CSS
comment promising "a legible floor". The floor was doing nothing, and the comment made it look
handled.

**Cause:** `figure svg{min-width:660px}` had been tuned for an earlier diagram. Rendered label size
is not the floor — it is `floor x fontSize / viewBoxWidth`. The new diagram used a 1000-unit viewBox
with 10-unit labels, so the same 660 px floor produced 6.6 px text where the old diagram had
produced legible text. Nothing in the CSS records which viewBox the constant was derived from, so
the next diagram inherits a number that no longer means anything.

**Rule:** derive the floor per diagram from its smallest label:
`min-width >= 9px x viewBoxWidth / smallestFontSize`. Write the derivation into the comment beside
the rule, not just the result — a bare constant cannot be checked by the next reader. Measure the
rendered size (`svg.getBoundingClientRect().width x fontSize / viewBox.baseVal.width`) rather than
trusting the floor.

### F8 amendment — the 500 px floor in the checker hides F8

**Saw:** two full review passes called a page clean at 500 px; pinned to a real 390 px phone it
scrolled sideways by 35 px, with two monospace paths running off the right edge mid-path.

**Cause:** Chrome headless refuses to open a viewport below 500 px — `--window-size=390` silently
reports a `clientWidth` of 500. A 56-character monospace token at 11.5 px is about 386 px: it fits
the 448 px column a 500 px window produces and overflows the 338 px column a real phone produces. So
the tool's floor sits exactly above the width at which this defect appears, and every rendered pass
reports clean.

**Rule:** `check_artifact_layout.py` now runs a `--pin-width 390` pass by default, pinning
`html,body{width:390px}` inside the 500 px window so the document lays out at the true width, and
reporting the elements whose text cannot wrap. Media queries still see 500 px, so the pinned pass
checks the one thing it can check honestly: does the document overflow its own width. Do not treat a
clean 500 px pass as evidence about phones.

### F26 — a UA-default `<sup>` or `<sub>` widens the line box it sits in

**Saw:** one line inside a three-paragraph callout sat 4-5 px lower than every other line, wherever
the prose carried an exponent. Measured line pitch in that paragraph read `[24, 28, 25]` against a
24.3 px line-height.

**Cause:** the page declared no `sup` rule, so the browser default applied — `vertical-align: super`
with `font-size: smaller` and a non-zero line-height. A raised inline box with its own line-height
**grows the line it sits on**, so the text around it is pushed apart.

**Why neither review method finds it:** there is no rule to read. The markup is correct and the
defect is the *absence* of a declaration, so source review sees nothing wrong; and nothing overlaps,
clips or overflows, so the geometry checker stays silent. Only measuring line pitch, or looking
closely at the render, shows it.

**Rule:** any page using `<sup>` or `<sub>` declares them explicitly —
`sup,sub{font-size:.72em;line-height:0;vertical-align:baseline;position:relative}` with
`top:-.5em` / `top:.25em`. `line-height:0` is the load-bearing part: it stops the raised box
contributing to the line box at all.

**Verifying a fix:** collect the line rectangles across the paragraph with `Range.getClientRects()`
and require every gap to equal the line-height. Filter out sub-pixel rect boundaries first — a naive
version of this check reports 1-2 px "pitches" that are rect edges rather than lines, and those
false positives will hide the real 4 px one.

### F24 amendment — `width:max-content` fixes a numeric table and breaks a prose one

**Saw:** two four-column tables of prose hid **2,082 px and 1,950 px** of themselves at 1440 px, on a
page with room to spare. The rightmost column of each — the one carrying the argument — was absent at
every viewport width, silently, behind an overlay scrollbar.

**Cause:** `table{width:max-content;min-width:100%}`, copied verbatim from a page whose tables were
four columns of single digits. There it is correct: let the table size to its content and scroll
rather than squeeze numbers. On columns of prose, `max-content` means *as wide as the longest
sentence*, so the table grows to two or three thousand pixels and the scroll container dutifully
hides most of it.

**Why it is worth its own entry:** the two cases look identical in the stylesheet, and the fix for
one is the defect in the other. A rule carried between pages without re-asking what its columns
contain is how a fix becomes a bug.

**Rule:** numeric tables may use `width:max-content`; tables containing a prose column use
`width:100%` with a `min-width` around the point where the columns stop being readable (560 px works
for four columns). Verify by measuring `scrollWidth - clientWidth` on the scroll container at the
widest viewport — it must be **0**. The layout checker exempts scroll containers by design, so it
reports clean either way and cannot catch this.

### F11 amendment — page chrome must not borrow the data palette

**Saw:** a page whose legend read "colour carries one meaning only: purple is the body-only slice,
blue the world-only slice, green everything" drew the word **FINISHED** in that green — including
inside a blue *world-only* cell. The same three tokens were also styling tier badges, a callout
accent and the header eyebrow.

**Cause:** F11 as written is about plotting code, so a page that keeps its *figure* colours honest
can still contradict its own legend through chrome. The tokens were reused because they were the
nice colours already on the page.

**Rule:** a token that encodes a data category is a data colour. Nothing that is not a member of that
category may use it — not a status badge, not a tier label, not an accent border, not the eyebrow.
Give chrome its own tokens. Check by grepping every `var(--<data-token>)` outside the figure that
defines it; on a correct page the only hits are the encoded elements themselves.

### F1 amendment — the same defect wearing `display:flex`

**Saw:** eighteen `<summary>` elements, each holding a marker, a bold term, a **bare text run**
(`, and why it is not called pain`) and a muted span. At desktop every comma sat 10 px away from its
word. At 390 px the runs wrapped into *side-by-side columns* — one summary rendered as
`Neuromodulator | (the / "NMN") | — a small / network that / re-tunes the / big one`, and a two-word
run `, and` became a 25 px column with the comma on one line and "and" on the next.

**Cause:** `summary{display:flex; gap:10px}`. A bare text run inside a flex container becomes an
**anonymous flex item**, so it can no longer wrap as part of the surrounding sentence — it shrinks to
its own min-content and wraps independently. The `gap` then applies *between words*, which is what
tears the comma off.

**Why it recurs:** this is F1 — the founding entry, first seen as `display:grid` on a list item — and
it will keep coming back, because flex and grid are the natural way to place a marker beside a label.
The display value changes; the defect does not.

**Rule:** never make a text-bearing element a flex or grid container. Position the marker instead:
`position:relative` on the row, `position:absolute; left:0` on the `::before`, and padding to clear
it. Then term, bare text and span flow as one inline run.

**Verifying a fix:** collect the summary's rectangles with `Range.getClientRects()`, group them by
`top`, and require the gap between consecutive rects **on the same line** to be 0. A non-zero
same-line gap is a torn word.

### F3 amendment — a `margin` shorthand un-centres a component that also carries the layout class

**Saw:** one section sat **175 px** left of every other section at 1440 px, hanging out of the page
column.

**Cause:** `.terms{margin:26px 0}` on an element whose class list is `col terms`. `.col` centres
itself with `margin-left:auto; margin-right:auto`; the shorthand later in the sheet resets all four
sides, so the auto margins became zero. Nothing overlaps and nothing overflows, so the geometry
checker sees a correctly laid-out section that happens to be in the wrong place.

**Rule:** a component rule applied *alongside* a layout class sets `margin-top`/`margin-bottom`
individually, never the shorthand.

**Verifying a fix:** measure `getBoundingClientRect().left` for every top-level column child; they
must all be identical.

### F3, second amendment — two rules that TIE on specificity, where the later one silently wins

The original F3 is a rule that *beats* another on specificity. This is the flatter case, and it is
harder to see: two rules with **equal** specificity, where source order decides and the loser reads
as live code.

**Saw:** a table given its own wider floor, `table.repl{min-width:660px}`, rendering at 560 —
because `table.results{min-width:560px}` appears later in the sheet and both selectors score
(0,1,1). The consequence was not visual: the table simply had a smaller floor than intended, its
real overflow point moved, and the scroll cue derived from the intended 660 then announced a scroll
across a 76px band where nothing scrolled. A dead declaration produced a wrong cue two rules away.

**Why neither review method catches it:** the stylesheet reads correctly — both rules are present,
both are well-formed, and the intent is obvious. Nothing overlaps, clips or overflows, so the
geometry checker is silent, and the rendered table looks entirely normal at its unintended floor.
The defect is only visible by comparing the declared value against the computed one.

**Rule:** a per-instance floor must **out-score** the shared rule it is meant to override, not merely
follow it: write `table.results.repl`, not `table.repl`. And never trust a `min-width` you have only
read — confirm it with `getComputedStyle(el).minWidth` on the rendered page, then derive any
breakpoint from that number.

**Verifying a fix:** for every element carrying a floor, print the declared value beside
`getComputedStyle(el).minWidth`. Any disagreement is a rule that lost a tie you did not know it was
in.

### F27 — a diagram's legibility floor can push all of its data off a phone

**Saw:** a schematic with a 950 px minimum width inside a scrolling box. At 390 px the reader saw the
row labels, one control box, and none of the grid the diagram exists to show — with overlay
scrollbars, no indication anything was missing.

**Cause:** the floor keeps labels readable (correct, see F25) but says nothing about *what is in the
first screenful*. Here the data columns started a quarter of the way across the viewBox, so the
visible strip was entirely label gutter.

**Rule:** at the floor, `firstDataX × floor ÷ viewBoxWidth` must land inside the narrowest column
width supported, or the phone view opens on nothing. Fix by tightening the label gutter in the
viewBox; where the diagram genuinely cannot fit, **say so** — a one-line scroll cue shown under a
media query, because an overlay scrollbar is not a cue.

### F28 — a class that matches no rule renders as a bare block, silently

**Saw:** a `<div class="note">` intended as a bordered callout rendered with no border, no
background and no padding, beside seven correctly boxed callouts on the same page. Its mono
uppercase heading, designed to sit inside a box, read as an orphan sub-label under the preceding
paragraph.

**Cause:** the component is defined as a **compound selector**, `.callout.note`, and the markup
carried only the modifier. `class="note"` matches nothing, so the element inherits bare-`div`
styling. CSS has no error for this: an unmatched selector is indistinguishable from a deliberate
absence of styling.

**Why neither review method catches it:** the markup looks right — `class="note"` is exactly what a
reader expects for a note — and the stylesheet is right too; the defect lives in the mismatch
between them. Nothing overflows, overlaps or clips, so the geometry checker is silent, and the block
is still perfectly readable, so a screenshot scan can pass over it.

**Rule:** a modifier class never travels alone. Write `class="callout note"`, or define the modifier
as a standalone rule.

**Verifying a fix:** append a bare `<div>`, read its computed style, then walk every `[class]`
element and flag any whose border, padding and background are all identical to it. One pass, and it
catches every instance on the page rather than the one somebody noticed.

### F29 — a table whose header row has fewer cells than its body rows

**Saw:** a sixteen-row results table rendered with three column headers over seven columns of
numbers. Every cell after the third sat under no header at all, and the reader had no way to know
which quantity a column held. The page still looked orderly: the rows were aligned, the numbers were
right, and nothing overflowed.

**Cause:** the table was rebuilt by a script that located it with a regex on
`<table class="results">`, and the page had three tables with that class. The regex matched the
**first** in document order, so a newly written three-column header was written over the sixteen-row
seven-column table while the intended target kept its stale numbers. Two defects for the price of
one, and neither is visible unless you count.

**Why neither review method catches it:** the HTML reads correctly in isolation — a `<thead>` with
three `<th>` is valid markup, and a `<tbody>` row with seven `<td>` is valid markup. The browser
does not complain; it renders the extra columns headerless. Nothing overlaps, nothing clips, and a
screenshot scan reads the block as "a table", because the eye checks alignment, not arity.

**Rule:** any script that rewrites a table must address it by **position or a unique id**, never by
a class that repeats. And after any table edit, assert that every `<tbody>` row has exactly as many
cells as the `<thead>` has headers, for every table on the page.

**Verifying a fix:** parse the page and, per table, print `len(thead th)` against the set of
`len(tr td)` across body rows. A set with more than one member, or a member that differs from the
header count, is the defect. This is three lines and catches every table at once; counting by eye on
the rendered page does not scale past about five columns.

### F30 — a transparent raster figure carries one theme's ink onto both grounds

**Saw:** in dark mode, two plots with no visible title, no axis labels, no ticks, no spines and no
reference line — a nearly empty rectangle with two bright white legend boxes floating in it. The
same two files look perfect in light mode, in a file browser, and in every screenshot taken so far.

**Cause:** the figures were written with `savefig(transparent=True)` and `figure.facecolor: none`,
which was done deliberately so they would sit on the page's paper colour rather than a white card.
But transparency does not make a figure theme-aware — it makes it inherit whatever ground the
reader has, while its ink stays the single colour it was drawn in. Near-black ink on a `#141416`
ground is invisible. The legends survived only because a legend frame has its own opaque fill,
which is what makes the failure look bizarre rather than blank.

**Why neither review method catches it:** the PNG is correct in isolation and the CSS is correct in
isolation; the defect exists only in the composite. The layout checker substitutes placeholders for
images and renders light by default, so it reports nothing. A screenshot review that looks at the
light render — the natural one to take — sees a good figure.

**Rule:** a raster figure either carries an **opaque** background, or ships one variant per theme
switched by `[data-theme]` plus the guarded media query. Transparency is only safe when every mark
in the image is drawn in a colour legible on both grounds, which for a plot with axes and text it
never is. This project's convention is the opaque form: `_plot.finish()` has always passed
`facecolor="white"`, and a new figure script that departs from it is the defect.

**Verifying a fix:** read the PNG's corner pixel and assert alpha is 255; then composite the image
over both `--paper` values and look at each.

### F31 — a scroll cue keyed to the viewport, for an overflow keyed to the column

**Saw:** at 1440px — a desktop, with room to spare — a prose table's last column cut off mid-word
("not a location, so a freezi"), with an overlay scrollbar as the only indication that anything was
missing. At 500px the same table showed a correct scroll cue.

**Cause:** the cue was written as `.tbl-hint{display:none}` plus
`@media (max-width:820px){.tbl-hint{display:block}}` — the right form for a table that overflows
only on a narrow *viewport*. But the table had since been given a content floor (`min-width:860px`)
larger than the 730px reading column it sits in, so it overflows its scroll box at **every**
viewport width. The condition that hides the cue and the condition that creates the overflow are
different conditions, and they had drifted apart.

**Why neither review method catches it:** the `.scroll` container absorbs the overflow, so nothing
spills onto the page and a geometry checker sees a clean box. A reviewer checking the phone width —
the width where scroll cues are usually wrong — finds the cue present and correct.

**Rule:** compare the scroll box's content floor against the width of the **column** it lives in,
not the viewport. If the floor is larger, the cue is unconditional (`display:block`); the
media-query form is correct only when the box is as wide as the viewport.

**Verifying a fix:** for each `.scroll`, read `scrollWidth` and `clientWidth` at the widest
supported viewport. Any box where `scrollWidth > clientWidth` there must have a cue that is visible
at that width.

### F32 — one scroll cue serving two components with different content floors

**Saw:** the cue "the plot is wider than a phone screen — scroll it sideways" printed above seven
figures that fitted their column perfectly, at every tablet width from about 670px to 1010px. Below
670 it was correct; above 1010 it was correctly absent. The false band was the middle.

**Cause:** the page had one `.dia-hint` class and one breakpoint, `@media (max-width:1010px)`. That
breakpoint was derived correctly — for the SVG diagram, whose legibility floor is 950px. When raster
figures were later given the same scroll-with-a-cue treatment, they reused the class, and their
floor is 620px. One breakpoint cannot be right for two floors, so it was wrong for whichever
component did not own it.

**Why neither review method catches it:** the phone width is the one everybody checks, and there the
cue is present and true; the desktop width is the second, and there it is correctly absent. The
defect lives only in the band between the two floors, which is exactly the range a two-width review
skips. Nothing overflows, nothing clips — the page is merely lying to the reader about a scrollbar.

**Rule:** one cue per floor, and the breakpoint is `floor + the column's side padding`, written in a
comment beside the rule so the derivation can be checked rather than trusted. This is the mirror of
[F31](#f31--a-scroll-cue-keyed-to-the-viewport-for-an-overflow-keyed-to-the-column): there the cue
was absent where overflow existed, here it is present where none does. Both come from a cue whose
condition and an overflow whose condition were allowed to drift apart.

**Verifying a fix:** for each scroll box, at three widths spanning the floors, assert
`cueVisible == (scrollWidth > clientWidth)`.

### F11, second amendment — a figure that borrows the page's *status* palette

The original F11 and its first amendment forbid the page's chrome from using the data colours. This
is the same defect running the other way: a figure script choosing a categorical colour with no
knowledge of what the surrounding page has already spent.

**Saw:** a stacked-bar figure drew "starved" in `#8a6d1f` — byte-identical to the page's
`--pending` token, the ochre that borders every "still open" callout — and "survived" in a green a
short hop from `--both`, the colour that means "the modulator reads everything" in the figure 1,300
pixels above. A reader who has just learned that green means *everything* meets green bars on the
same rows meaning *survived*.

**Why the caption is not the fix:** the caption said, in as many words, "these colours mean an
outcome and nothing else — they are not the purple/blue/green that mean what the modulator reads."
A sentence telling the reader to ignore what they can see is an admission that the figure and the
page disagree, not a repair.

**Rule:** the figure module owns one colour→meaning map for the whole page, chrome tokens included,
and a new categorical set is checked against every token in `:root` before it ships — by perceptual
distance, not equality, since a near neighbour reads the same as an exact match. Where every hue is
already spent, the honest answer is a neutral ramp: it carries the ordering without claiming a
meaning the page has given away.

**Verifying a fix:** take a pixel census of the rendered PNG, take the token list from `:root`, and
report the closest token to each figure colour. The output is not pass/fail — an exact match is
*correct* when the figure element means what the token means, and this project's data figures should
hit `--intero` / `--extero` / `--both` dead on. The check earns its keep by forcing the author to
name, colour by colour, why each near match is intended; the one that cannot be named is the defect.
Running it on the fixed figure above returns only its own data colours and its text ink, and nothing
within reach of `--pending`.

### F33 — the legibility fix for an in-panel annotation deletes the data it was illegible against

**Saw:** in a panel whose entire claim is "these sixteen lines are flat", one of the sixteen ran
into a tidy label box two thirds of the way across and never came out. The label was perfectly
readable. So was every other line. Nothing looked wrong.

**Cause:** the annotation had previously been reported as hard to read, because three thin series
ran through its glyphs. The fix applied was an opaque `bbox` behind the text. But text-on-data and
box-on-data are the same collision over the same pixels: an opaque ground does not resolve it, it
only decides which of the two the reader loses. The first version lost the text; the second lost the
data, which is strictly worse, and is invisible because a missing line looks like a line that was
never plotted.

**Why neither review method catches it:** the geometry checker sees one PNG and no overflow. A
screenshot scan sees a clean, legible label — the defect is the *absence* of something, and absence
does not attract the eye. Finding it means tracing one specific line from one end of the panel to
the other and noticing it stops.

**Rule:** an annotation goes where there is no ink. Enlarge the axis margin until empty space
exists, or move the text into it. A filled `bbox` is permitted only over axes that are genuinely
empty there. Where the text must sit over data, use a halo
(`path_effects.withStroke(linewidth=…, foreground=<paper>)`) instead: the glyphs stay legible and
the line runs through the gaps between them, so neither is deleted.

**Verifying a fix:** for each annotation take its bounding box in data coordinates and assert that
no plotted series has a y-value inside it across the box's x-range. Failing that, crop the region at
full resolution and follow each line through it.

**Related:** this is the third defect in the family where a *fix* introduced the next one — the
opaque figure background of [F30](#f30) was the fix that made [F25](#f25)'s narrowing necessary, and
the narrowing then sheared an axis label off the canvas. A figure script that is edited to satisfy a
review finding should be re-rendered and re-read as a whole, not diffed.

### F34 — a `min-width` floor sized for the widest table, applied by a type selector

**Saw:** at a 390px phone width, a sixteen-row numeric table rendering its row-label column and
**neither of its two numbers** — a list of sixteen names with no data. At 500px the same table was
worse than useless: the clip edge fell exactly on the decimal point, so 28.62, 26.67 and 24.03 read
as "28", "26" and "24" — plausible integers, wrong values, with nothing to suggest anything was
missing.

**Cause:** `table{min-width:560px}` — a bare type selector. The 560 was derived honestly, for the
page's seven-column results table. Every other table on the page then inherited it, including a
three-column one whose own content needs about 350px and which would have fitted a phone with room
to spare. The floor did not describe that table's content; it described a different table's.

**Why neither review method catches it:** the floor is inside a `.scroll` container, so nothing
overflows the page and the geometry checker is silent. The table is not squeezed — it is at its
floor, which is the state the floor exists to produce. And a review at the common 500px checkpoint
sees numbers, because the clip lands mid-value rather than before it; only reading them against the
source shows they are truncations rather than values.

**Rule:** a `min-width` floor is a statement about one table's content, so it belongs on a class,
never on `table{}`. Every table gets the floor its own columns need, or none.

**A second trap in the same family:** a `.scroll` box inside a padded component is narrower than one
in a bare section — on this page, 52px narrower inside a `.callout`. A cue breakpoint derived as
`floor + column padding` is therefore wrong by exactly that much for the boxes inside callouts, and
the table's verdict column was cut with no cue across a 50px band. Derive each breakpoint from the
box's own measured `clientWidth`, not from the column's.

**Verifying a fix:** for every table, assert its declared floor is no greater than its `max-content`
width plus whatever headroom was intended; and for every `.scroll`, assert `cueVisible ==
(scrollWidth > clientWidth)` at a sweep of widths rather than at two checkpoints.

### F35 — a shared `min-width` floor SMALLER than a table's content, where a value takes the hit

The mirror of [F34](#f34). There an inherited floor was too large and pushed columns off a phone.
Here it is 14 pixels too small, and the failure is stranger: the table does not overflow at all.

**Saw:** across a 126px band of viewport widths, one row of a numeric table rendered as

```
bush hiding (% of steps)     14.45 ±      13.84 ± 0.53
                              0.57
```

and two column headers broke at their hyphens. On a 390px phone the clip fell after `14.` and the
wrapped `0.57` was off-screen, so the row showed a number over an empty line.

**Cause:** the table inherited `table.results{min-width:560px}`, sized for a different table. Its own
max-content width is 574px. A table asked to fit inside less than its content does not necessarily
overflow — it looks for something to break, and a numeric cell like `14.45 ± 0.57` contains a
breakable space. So the table met the floor by splitting a *value* across two lines.

**Why every check passes:** nothing overflows, so the scroll cue is correct at every width — and the
`cueVisible == (scrollWidth > clientWidth)` sweep that catches F31 and F32 confirms it. No text is
narrow, nothing overlaps, nothing sticks out, and the table is fully legible at desktop and at the
floor. The defect exists only between the two, and it is one deformed row.

**Rule:** a numeric table's floor is *its own* measured max-content width, never a number borrowed
from a wider table — and `td.num` carries `white-space:nowrap`, so no future floor can split a value
whatever else it does.

**Verifying a fix:** render each table at its declared floor and assert every body cell is one line
tall (`height − padding ≤ line-height`). F34's check — floor ≤ max-content — tests the other
direction and cannot see this one; both are needed, and together they say the floor must equal the
content, not merely bound it from one side.

## Related

- [`artifact_generation_guide`](artifact_generation_guide.md) — the wider guide: content, claims,
  jargon, figure legibility. This document is the *format* half and is narrower on purpose.
- `scripts/claude/check_artifact_layout.py` — the tool.
- [`artifact-format-reviewer`](../../../../.claude/agents/artifact-format-reviewer.md) — the agent
  that runs it.
