# Tier 3 — headed Chrome on a vncserver display (local, this container)

Use this **only** when Tier 1/2 returned a Cloudflare interstitial (*"Just a moment…"* / *"Checking your
browser"* / a `cf-mitigated` header), and **only after the user confirmed** — it starts a VNC server and
a visible browser on this machine.

Everything runs **directly on this box**: no ssh, no scp. The campus IP is already yours and the
destination filesystem is local, so the PDF is just `mv`-ed into place.

> **Run the whole flow as ONE script** (python or a single zsh script), not as many separate Bash calls.
> The poll loops use `sleep`, which must run *inside* a script — issuing bare foreground `sleep` as its
> own Bash call is blocked in this harness.

## Container facts (re-verify if the environment changed)
- `vncserver` = **TigerVNC** (`/usr/bin/tigervncserver`); **Xvfb is NOT installed**.
- Browser `/usr/bin/google-chrome`; **requires** `--no-sandbox --disable-gpu --disable-dev-shm-usage`.
  The `Failed to connect to the bus` / DBus errors in the log are **harmless** — ignore them.
- Screenshot tools present: **`ffmpeg`** (x11grab), `xwd`, `xdpyinfo`, `xset`.
  **Absent:** `scrot`, `import`/`convert` (ImageMagick), and crucially **`xdotool` / `wmctrl`** — so you
  **cannot programmatically click** anything.
- Shell is **zsh** — quote globs or `setopt null_glob`.

---

## THE key lesson (why the first Tier-3 run failed, 2026-07-13)

Pointing Chrome at a raw `…/pdf?id=…` endpoint **cold** fails: Cloudflare intercepts the *file* request
with a challenge and no file ever lands. The fix that worked:

> **Warm the cookie first.** Load the **human-facing** page (forum / abstract / landing), let the managed
> JS challenge auto-clear — it writes a `cf_clearance` cookie into the profile — then **kill Chrome but
> keep the profile** and **relaunch the same profile** straight at the `/pdf` endpoint. The persisted
> cookie now rides the request and the PDF downloads.

Two challenge types, two outcomes:

| Challenge | Looks like | Auto-solves in headed Chrome? | Action |
|---|---|---|---|
| **Managed / JS** (OpenReview) | briefly "checking…", then the real page renders | **Yes**, on the human page | warm-cookie-first, then fetch `/pdf` |
| **Interactive Turnstile** (ScienceDirect) | *"Verify you are human"* **checkbox** | **No** — needs a click | **dead end** here (no `xdotool`) → stop & report |

---

## Step 1 — Ensure an X display
```zsh
if ! vncserver -list 2>/dev/null | grep -q ':1'; then
  # -SecurityTypes None => no interactive vncpasswd prompt; -localhost yes binds VNC to loopback.
  vncserver -localhost yes -SecurityTypes None -geometry 1440x900 -depth 24 :1
  sleep 3
fi
export DISPLAY=:1
DISPLAY=:1 xdpyinfo >/dev/null 2>&1 && echo "display :1 up"
```

## Step 2 — Pre-seed a throwaway Chrome profile that DOWNLOADS PDFs
```zsh
PROFILE="/tmp/cpf_$$"; DLDIR="$PROFILE/dl"
mkdir -p "$PROFILE/Default" "$DLDIR"
cat > "$PROFILE/Default/Preferences" <<JSON
{"plugins":{"always_open_pdf_externally":true},
 "download":{"default_directory":"$DLDIR","prompt_for_download":false},
 "profile":{"default_content_setting_values":{"automatic_downloads":1}}}
JSON
FLAGS="--user-data-dir=$PROFILE --no-first-run --no-default-browser-check --no-sandbox --disable-gpu --disable-dev-shm-usage --disable-blink-features=AutomationControlled"
```

## Step 3 — Warm the cookie, THEN fetch the PDF (the two-phase pattern)
```zsh
LANDING="https://openreview.net/forum?id=OpC-9aBBVJe"   # human page, NOT /pdf
PDF_URL="https://openreview.net/pdf?id=OpC-9aBBVJe"

# Phase A: warm — clear the managed challenge on the human page (sets cf_clearance in $PROFILE)
DISPLAY=:1 google-chrome $FLAGS --new-window "$LANDING" >"$PROFILE/a.log" 2>&1 &
sleep 25
pkill -f "user-data-dir=$PROFILE"; sleep 2        # stop Chrome, KEEP the profile (cookie persists)

# Phase B: fetch — relaunch the SAME profile straight at the file endpoint
setopt local_options null_glob
rm -f "$DLDIR"/*
DISPLAY=:1 google-chrome $FLAGS --new-window "$PDF_URL" >"$PROFILE/b.log" 2>&1 &
for i in {1..30}; do          # ~90s
  sleep 3
  done_pdf=("$DLDIR"/*.pdf(N)); part=("$DLDIR"/*.crdownload(N))
  (( ${#done_pdf} && ${#part} == 0 )) && break
done
```
For a **single-page** site (no separate file endpoint) the warm page and the download are the same URL —
one launch is enough; keep the poll loop.

## Step 4 — Verify, move, clean up
```zsh
DEST="docs/project/references/<topic>/sources/<Author et al. YEAR - Short title>.pdf"
f="${done_pdf[1]}"
if [[ -n "$f" ]] && head -c 4 "$f" | grep -q '%PDF'; then
  mv "$f" "$DEST"; echo "OK -> $DEST"
else
  echo "no PDF — run the diagnosis below"
fi
pkill -f "user-data-dir=$PROFILE" 2>/dev/null    # kill ONLY the Chrome you started
rm -rf "$PROFILE"
```

---

## If it still produces nothing — diagnose, don't guess
Empty-with-no-error is ambiguous. Split "is my browser broken?" from "is this site blocking me?":

1. **Prove the pipeline** with a known OA PDF. If this downloads, the display/profile/download path is
   fine and the block is target-specific (challenge), not your setup:
   ```zsh
   DISPLAY=:1 google-chrome $FLAGS --new-window "https://arxiv.org/pdf/1606.04671" & sleep 12
   ls "$DLDIR"/*.pdf   # expect the arXiv PDF
   ```
2. **See the actual page** — screenshot the framebuffer and READ the PNG:
   ```zsh
   DISPLAY=:1 ffmpeg -y -f x11grab -video_size 1440x900 -i :1 -frames:v 1 /tmp/shot.png
   ```
   - Full paper rendered → cookie is warm; you aimed at the wrong URL → do Phase B (`/pdf`) now.
   - *"Verify you are human"* checkbox → **interactive Turnstile** → dead end here, stop & report.
   - *"Just a moment…"* spinner that never resolves → give it more time, or it's the interactive kind.

---

## Cleanup gotcha (bit me on 2026-07-13)
`pgrep -f Xtigervnc` **false-positives on your own command line** — your `pgrep`/`ps` invocation contains
the literal string "Xtigervnc", so it "finds" a VNC that isn't there and looks like it keeps respawning.
**`vncserver -list` is the source of truth** for live sessions. Tear down only what you started:
```zsh
vncserver -list          # authoritative; trust this, not pgrep
vncserver -kill :1        # this container had ZERO sessions at task start, so :1 is yours
```
