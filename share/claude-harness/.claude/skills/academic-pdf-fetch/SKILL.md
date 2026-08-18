---
name: academic-pdf-fetch
description: "Download the PDF of ONE academic paper into the project's reference library. Fires on 'download this paper', 'get me the PDF of <title>', 'fetch this DOI/arXiv', a bare DOI (10.xxxx/...) or arXiv id (e.g. 2303.07507), 'add this to my references / sources folder', 'grab the published version of <paper>', or when a paywalled publisher is named (Springer, Elsevier/ScienceDirect, APS / Physical Review, Nature, Science/AAAS, Wiley, Cambridge, IEEE, ACM, PNAS, Oxford, Taylor & Francis). Runs INSIDE the lab container whose own egress IP is already a campus/institutional IP (115.145.189.x), so institutional journal access works with a plain LOCAL curl — no SSH, no tunnel, no scp, no jump host. Escalates lazily and stops at the first verified PDF: OA/Unpaywall resolve -> direct curl -> campus-authenticated curl (Tier 2) -> (only after you confirm) a real headed Chrome on a vncserver display to clear Cloudflare (Tier 3). Prefers the published version of record over a preprint. Legitimate access only — no Sci-Hub."
---

# academic-pdf-fetch — get the version-of-record PDF into `references/<topic>/sources/`

Download **one** paper's PDF and drop it in the project reference library. You are running
**inside the lab container**, and that changes everything about how this works vs. a laptop
version of the same skill (see [Why this is not a laptop skill](#why-this-is-not-a-laptop-skill)).

## This container (probed 2026-07-13 — re-verify at runtime, don't trust stale values)

| Fact | Value | Why it matters |
|---|---|---|
| Egress IP | `115.145.189.x` (was `.17`) — **verify with `curl -s https://api.ipify.org`** | Your OWN IP is the campus IP → institutional access from a plain local curl. |
| Login shell | **zsh** (`/usr/bin/zsh`) | zsh globbing/quoting applies; unquoted `*.pdf` in an empty dir errors — quote globs or use `setopt null_glob`. |
| Browser | `/usr/bin/google-chrome` (no chromium) | Tier 3 only. |
| Virtual display | **`vncserver` = TigerVNC** (`/usr/bin/tigervncserver`); **Xvfb is NOT installed** | Tier 3 must start a display with `vncserver`, not `Xvfb`. |
| `$DISPLAY` | empty at start | Tier 3 must start a display first. |
| PDF tools | no `pdfinfo`/`pdftotext`/`qpdf` | Verification is **magic-bytes only** (`head -c 4 | grep %PDF`). |

## Destination & naming

- **Default destination:** `docs/project/references/<topic>/sources/`. If the topic isn't obvious
  from the request, ask which topic folder (e.g. `continual_learning`). Create `sources/` if absent.
- **Filename:** `Author et al. YEAR - Short title.pdf` (e.g. `Abbas et al. 2023 - Loss of plasticity in continual deep RL.pdf`).
- **Dedup before downloading:** list the destination `sources/` first. A preprint and the published
  version are the **same work** — keep exactly ONE, preferring the version of record. If you obtain the
  final PDF and a preprint is already there, replace it. If only the preprint is obtainable, keep it and
  put `(preprint)` in the filename.

## Two locks — the mental model

Getting a PDF means clearing up to two independent locks:

1. **Paywall** — asks *"where are you connecting from?"* Opened by a campus IP. **You already have this
   natively** (your egress IP is institutional). No tunnelling needed.
2. **Bot check (Cloudflare)** — asks *"are you a real browser?"* Opened only by a **genuine headed
   browser** (real JS engine + real TLS/JA3 fingerprint). Headless Chrome and curl-cookie-replay both
   fail this. This is the ONLY reason to reach Tier 3.

**Version preference (always):** OA final PDF  >  paywalled final PDF via campus access  >  arXiv/preprint
(label it a preprint). Never hand back a preprint as if it were the version of record.

## The cascade — escalate lazily, STOP at the first verified PDF

### Tier 0 — Resolve the identifier, classify candidates (published vs preprint)
- **Unpaywall** (best OA resolver): `curl -s "https://api.unpaywall.org/v2/<DOI>?email=YOUR_EMAIL@example.com"`
  → prefer an `oa_location` with `"version":"publishedVersion"`; note its `url_for_pdf`.
- **Publisher landing page** = the version of record: `https://doi.org/<DOI>`.
- **Semantic Scholar**: `curl -s "https://api.semanticscholar.org/graph/v1/paper/DOI:<DOI>?fields=openAccessPdf,title,year,authors"`
  → `openAccessPdf.url` when present.
- **Life sciences:** PMC (`https://www.ncbi.nlm.nih.gov/pmc/`), bioRxiv/medRxiv.
- **arXiv** `https://arxiv.org/pdf/<id>` = **PREPRINT → fallback only.**

### Tier 1 — Direct fetch (the common case)
```zsh
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
curl -sL -A "$UA" -o out.pdf "<pdf_url>"
```
Then **verify magic bytes** (see below). If it's HTML instead of a PDF:
- a login / paywall page  → **Tier 2**
- a *"Just a moment…"* / *"Checking your browser"* / `cf-mitigated` page  → **Tier 3**

### Tier 2 — Paywalled, non-Cloudflare (e.g. Springer, ScienceDirect, Wiley)
The **same local curl already carries campus access** because your IP is institutional — there is
nothing extra to authenticate. Just add a publisher `Referer` if the CDN demands it, and write
**straight to the destination** (no copy-back):
```zsh
curl -sL -A "$UA" -e "https://link.springer.com/" \
  -o "docs/project/references/<topic>/sources/<name>.pdf" "<pdf_url>"
```
Verify. If you instead hit a *"checking your browser"* interstitial → Tier 3.

### Tier 3 — Paywalled **and** Cloudflare (e.g. APS / Physical Review, Cambridge) — CONFIRM FIRST
Headless anything fails here (headless Chrome is detected; curl/requests cookie-replay still 403s
because Cloudflare fingerprints the TLS handshake). The only robust method is a **real headed Chrome
on a vncserver display, on THIS machine**.

**This tier starts a VNC server and a browser — a visible, heavier side effect. Pause and ask the user
to confirm before running it.** Once confirmed, follow the full recipe in
[`references/local-fetch.md`](references/local-fetch.md). **Two kinds of Cloudflare challenge — know which you're facing (learned 2026-07-13):**

- *Managed / JS challenge* (e.g. OpenReview) — a real headed browser **auto-solves** it, but only on the
normal human-facing page, not on a raw file endpoint. **Warm the cookie first:** load the landing /
forum / abstract page, let the challenge clear (it writes a `cf_clearance` cookie into the profile),
then **kill Chrome but KEEP the profile** and **relaunch the SAME profile** at the `/pdf` endpoint — the
persisted cookie waves it through and the PDF downloads. Hitting `/pdf` cold **fails** (challenge fires,
no file). This is the single most important Tier-3 rule.
- *Interactive Turnstile captcha* (e.g. ScienceDirect) — a "Verify you are human" **checkbox** that does
NOT auto-solve; it needs a real click. This container has no `xdotool`/`wmctrl`, so it is a **dead end
for the automated path** — stop and report (the content is usually closed-access anyway).

The mechanics (start `vncserver :1` with `-SecurityTypes None`; pre-seed a throwaway profile that forces
PDF *download*; `DISPLAY=:1 google-chrome --no-sandbox …`; the two-phase warm-then-fetch launch; poll for
a `*.pdf` with no `*.crdownload`; verify; `mv`; kill Chrome) and the **failure-diagnosis routine**
(pipeline sanity-check + an `ffmpeg x11grab` screenshot you then read) are in the recipe.

Because you are **local**, there is NO ssh-wrapping and NO scp anywhere in this tier — run the
commands directly and `mv` the file to the destination path.

## Verification (ALWAYS — never trust the `.pdf` extension)
```zsh
f="<path>"
if head -c 4 "$f" | grep -q '%PDF'; then echo "OK: real PDF"; else
  echo "NOT A PDF — first 80 chars show which lock was hit:"; head -c 80 "$f"; echo
fi
```
A "login" / "sign in" body → paywall (retry Tier 2 headers, or you lack a subscription to that title).
A "Just a moment" / "cf-" body → Cloudflare → Tier 3.

## Guardrails (do not weaken these)
- **Only lawful access:** open-access copies, or content licensed through the user's own institutional
  subscription. This skill is **not** for circumventing access controls.
- **No Sci-Hub or similar.** Stop at legitimate sources.
- If a paper cannot be obtained within this cascade, **say so plainly and name which lock blocked it**
  — do not silently escalate to dubious sources.

## Why this is not a laptop skill
On a laptop this skill would SSH to a lab server, run curl there to borrow the campus IP, then `scp`
the PDF back — two hops, a tunnel, and a copy-back. **Here, you ARE the lab server.** Your egress IP is
already institutional, so Tiers 2 and 3 are plain **local** commands (`curl` / `google-chrome`) with
**no tunnel and no scp**; the PDF is simply written or `mv`-ed to its destination on the same
filesystem. The only machinery that survives is the browser tier (Tier 3), and even that runs directly
on this box against a local `vncserver` display.

## Shell note
This container's shell is **zsh**. Quote globs (`"*.pdf"`) or `setopt null_glob` before matching in a
possibly-empty download dir, or an unmatched glob will error out the command. The laptop version's
zsh-vs-ssh/scp warnings do not apply — there is no remote hop here.
