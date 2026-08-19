---
name: publish-page
description: "Publish a finished piece of work as an Artifact — a private web page on claude.ai with its own shareable URL — instead of leaving it as a file path or a wall of terminal text. Use for anything with figures to compare (matplotlib PNGs, kernel/parameter sweeps, before-after plots), a decision to be made from evidence, an experiment write-up, a mechanism study, or a doc a collaborator will read. Trigger on /publish-page, 'publish that', 'put that on a page', 'make it a page I can share', 'give me a link', 'artifact', 'share this with the lab', 'update the artifact/page', or any request to see several figures side by side. Also use proactively when a reply would otherwise be six file paths the user must open one at a time. Do NOT use for scratch output, a single number, a one-paragraph answer, or anything the user asked to keep local."
---

# publish-page — turn finished work into a shareable page

The `Artifact` tool renders a local HTML or Markdown file to a private page on claude.ai and
returns a URL. The page is **private to the user's account** until they share it from the
page's share menu.

> Lives at project level, not `~/.claude/skills/`, because Claude runs in a Docker container
> whose home directory does not persist — only this repo on the NAS survives a container
> rebuild. Anything that should outlive the session belongs in the repo.

Reach for this when the work has an audience and a shape: several figures that only mean
something next to each other, a decision that rests on evidence, an analysis a collaborator
or future-you will re-read. A file path makes the user do the opening; a page does not.

## When NOT to use
- Scratch, intermediate, or debug output — that goes to `tmp/YYYYMMDD_HHMMSS_<topic>/` per CLAUDE.md.
- A short answer that fits in the reply. Publishing it adds a click, not clarity.
- Anything the user asked to keep local, or anything containing credentials / unpublished data
  they have not agreed to put on a hosted page. **When in doubt about sensitivity, ask first** —
  publishing is outward-facing and the URL exists whether or not it is shared.

## How to do it

1. **Load the `artifact-design` skill first.** Required before writing the file, every time.
   It calibrates how much design the piece actually warrants.
2. Write a self-contained `.html` file (usually into the same `tmp/<timestamp>_<topic>/`
   directory as the scripts that produced the figures, so the page and its source sit together).
   Write page content only — no `<!DOCTYPE>`, `<html>`, `<head>` or `<body>` tags; those are
   added at publish time. Put a real `<title>` at the top.
3. Call `Artifact` with the file path, a one-sentence `description`, and a `favicon` emoji.

## Conventions that make these pages worth re-reading

Carry the project's documentation framing (CLAUDE.md) onto the page — it is a doc like any other:

- **Lead with a plain-language entry point.** First section says what the page is about, why it
  exists, and what it claims, in English a reader without context can follow. Translate every
  symbol, run ID, and config path on first mention.
- **Every figure gets a real caption** — what it shows *and* what to conclude from it. A figure
  with no caption is an unlabelled data dump.
- **Say where it came from.** Footer with the `tmp/` path and the command that regenerates the
  figures. A page nobody can reproduce is a screenshot.
- **Separate settled from open.** For a decision page, a two-column ledger of what is decided
  and what is still open beats prose — it is the thing the user actually came for.
- **State corrections.** If the measurements contradicted an earlier claim of yours, say so on
  the page rather than quietly shipping the corrected version.

## Mechanics and gotchas

- **Self-contained only.** A strict CSP blocks external hosts. Embed PNGs as
  `data:image/png;base64,...` (a `base64.b64encode` in the page-builder script), inline all CSS
  and JS. Google Fonts is the single exception. Page must stay under 16 MB.
- **Images live in the file.** Once published, the page does not depend on the NAS staying
  mounted or the `tmp/` directory surviving.
- **Republish the same file path to update in place** — same URL, no second link. Only use a
  new path when a genuinely separate page is intended.
- **From a different session**, pass the artifact's `url` to update it; publishing without `url`
  creates a duplicate. Find old ones with `action: "list"`.
- **Math.** Real equations go in a styled block; inline expressions with subscripts go in
  `<code>`; single symbols as Unicode (γ β τ σ ρ Δθ ∥ ⊥). Same reasoning as the chat renderer
  rule in CLAUDE.md, for the same reason.
- **Comments.** Viewers can leave comment threads; read them with `action: "comments"` and reply
  into any thread the user activates for Claude. Often the easiest way for the user to mark up a
  specific figure.

## Pattern: a figure-set page

Generate figures with a plotting script, then a small `build_page.py` that base64-embeds each PNG
and writes `index.html`. Keeps the page regenerable rather than hand-assembled — edit a caption or
a parameter, re-run two commands, republish to the same URL.
