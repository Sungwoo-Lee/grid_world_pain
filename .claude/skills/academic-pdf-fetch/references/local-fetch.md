# Tier 3 — headed Chrome on a vncserver display (local, this container)

Use this **only** when Tier 1/2 returned a Cloudflare interstitial (*"Just a moment…"* /
*"Checking your browser"* / a `cf-mitigated` header), and **only after the user has confirmed** —
this starts a VNC server and a visible browser on this machine.

Everything below runs **directly on this box**. There is NO ssh and NO scp: the campus IP is
already yours, and the destination filesystem is local, so the PDF is just `mv`-ed into place.

Container facts this recipe depends on (re-verify if the environment changed):
- `vncserver` is **TigerVNC** (`/usr/bin/tigervncserver`) — Xvfb is **not** installed.
- Browser is `/usr/bin/google-chrome`.
- Shell is **zsh** — quote globs.

---

## 1. Ensure an X display

```zsh
# Reuse an existing display if one is already up; otherwise start :1.
if [ -z "$DISPLAY" ] && ! pgrep -f 'Xtigervnc .*:1' >/dev/null; then
  # -SecurityTypes None => no interactive vncpasswd prompt (we only need the X server,
  # not remote VNC access); -localhost yes keeps the VNC port bound to loopback.
  vncserver -localhost yes -SecurityTypes None -geometry 1920x1080 -depth 24 :1
fi
export DISPLAY=:1
```

If TigerVNC still complains about a missing password on your first ever run, create one once with
`vncpasswd` (any value) — subsequent starts with `-SecurityTypes None` won't need it.

## 2. Pre-seed a throwaway Chrome profile that DOWNLOADS PDFs (not in-viewer)

```zsh
PROFILE="/tmp/cpf_$$"          # unique per run
DLDIR="$PROFILE/dl"
mkdir -p "$PROFILE/Default" "$DLDIR"
cat > "$PROFILE/Default/Preferences" <<JSON
{"plugins":{"always_open_pdf_externally":true},
 "download":{"default_directory":"$DLDIR","prompt_for_download":false},
 "profile":{"default_content_setting_values":{"automatic_downloads":1}}}
JSON
```

## 3. Launch real (headed) Chrome at the PDF / article URL

```zsh
PDF_URL="<the publisher PDF or article URL>"
DISPLAY=:1 google-chrome \
  --user-data-dir="$PROFILE" \
  --no-first-run --no-default-browser-check \
  --disable-blink-features=AutomationControlled \
  --new-window "$PDF_URL" >/dev/null 2>&1 &
CHROME_PID=$!
```

Real Chrome presents a genuine TLS fingerprint and runs the challenge JS, so Cloudflare clears it;
the campus IP grants the paywall; Chrome then downloads the PDF into `$DLDIR`.

## 4. Poll for a completed download, verify, move, clean up

```zsh
DEST="docs/project/references/<topic>/sources/<Author et al. YEAR - Short title>.pdf"
setopt local_options null_glob
for i in {1..60}; do            # up to ~2 min
  done_pdf=("$DLDIR"/*.pdf(N))
  part=("$DLDIR"/*.crdownload(N))
  if (( ${#done_pdf} > 0 )) && (( ${#part} == 0 )); then break; fi
  sleep 2
done

f="${done_pdf[1]}"
if [[ -n "$f" ]] && head -c 4 "$f" | grep -q '%PDF'; then
  mv "$f" "$DEST"
  echo "OK -> $DEST"
else
  echo "Tier 3 did not produce a PDF. First 80 chars of what landed (if any):"
  [[ -n "$f" ]] && head -c 80 "$f"; echo
fi

# Always kill THIS Chrome instance (match on the unique profile dir) and remove the profile.
pkill -f "user-data-dir=$PROFILE" 2>/dev/null
rm -rf "$PROFILE"
```

## Notes
- Kill only the Chrome you started (`pkill -f "user-data-dir=$PROFILE"`) so you never touch the
  user's own interactive Chrome/VNC desktop session.
- Leave the `:1` display running if you started it and might fetch more papers this session; tear it
  down with `vncserver -kill :1` when done if you started it solely for this.
- If polling times out, the challenge may have shown a visible checkbox that needs a human — report
  that plainly rather than looping forever.
