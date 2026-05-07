#!/usr/bin/env bash
# Bootstrap SSH key auth from this docker container to lab nodes 101–114.
#
# Run this once per fresh container. Idempotent — safe to re-run.
# All work happens in ~/.ssh/ (docker overlay, POSIX perms honored).
# The CIFS-mounted project stays clean of credentials.
#
# Steps:
#   1. Verify sshpass is installed.
#   2. Generate ~/.ssh/id_ed25519_gridworld if missing.
#   3. Populate ~/.ssh/known_hosts with host keys for nodes 101–114.
#   4. Write ~/.ssh/config Host block (port 1800 + identity file).
#   5. Prompt for the lab password ONCE, push the public key to all 14 nodes
#      via sshpass + ssh-copy-id. Password stays in the SSHPASS env var
#      for the duration of the loop and is unset at the end.
#   6. Verify key auth works to node 101.
#
# Usage:
#   bash scripts/bootstrap_lab_ssh.sh

set -euo pipefail

KEY="$HOME/.ssh/id_ed25519_gridworld"
KNOWN_HOSTS="$HOME/.ssh/known_hosts"
SSH_CONFIG="$HOME/.ssh/config"
NODES=(101 102 103 104 105 106 107 108 109 110 111 112 113 114)
PORT=1800
USER_REMOTE=vncuser
NETWORK_PREFIX=192.168.0
CONFIG_SENTINEL="# lab-nodes 101-114 (managed by bootstrap_lab_ssh.sh)"

step() { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  ✓ %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*"; }
die()  { printf '  ✗ %s\n' "$*" >&2; exit 1; }

# ── 1. sshpass ────────────────────────────────────────────────────────────
step "1/6 Checking sshpass"
command -v sshpass >/dev/null || die "sshpass not installed. Run: sudo apt install -y sshpass"
ok "sshpass $(sshpass -V | awk 'NR==1{print $2}')"

# ── 2. SSH key ────────────────────────────────────────────────────────────
step "2/6 SSH key (~/.ssh/id_ed25519_gridworld)"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
if [[ -f "$KEY" ]]; then
    ok "key exists"
else
    ssh-keygen -t ed25519 -f "$KEY" -N '' -C "gridworld-lab-$(hostname)-$(date +%Y%m%d)"
    ok "key generated"
fi
chmod 600 "$KEY" "$KEY.pub" 2>/dev/null || true

# ── 3. known_hosts ────────────────────────────────────────────────────────
step "3/6 Populating ~/.ssh/known_hosts"
touch "$KNOWN_HOSTS" && chmod 600 "$KNOWN_HOSTS"
added=0; skipped=0; unreachable=0
for n in "${NODES[@]}"; do
    host="${NETWORK_PREFIX}.${n}"
    if ssh-keygen -F "[${host}]:${PORT}" >/dev/null 2>&1; then
        skipped=$((skipped+1))
        continue
    fi
    # `|| true` defends against pipefail killing the script when grep finds
    # no matches (i.e., ssh-keyscan returned nothing because the node is down).
    scan="$(ssh-keyscan -p "$PORT" -t ed25519,rsa -T 5 "$host" 2>/dev/null | grep '^\[' || true)"
    if [[ -z "$scan" ]]; then
        warn "node $n unreachable — skipped"
        unreachable=$((unreachable+1))
        continue
    fi
    printf '%s\n' "$scan" >> "$KNOWN_HOSTS"
    added=$((added+1))
done
ok "added=$added  already-known=$skipped  unreachable=$unreachable"

# ── 4. ~/.ssh/config ──────────────────────────────────────────────────────
step "4/6 ~/.ssh/config Host block"
touch "$SSH_CONFIG" && chmod 600 "$SSH_CONFIG"
if grep -qF "$CONFIG_SENTINEL" "$SSH_CONFIG"; then
    ok "config block already present"
else
    cat >> "$SSH_CONFIG" <<EOF

$CONFIG_SENTINEL
Host ${NETWORK_PREFIX}.10? ${NETWORK_PREFIX}.11?
  Port $PORT
  User $USER_REMOTE
  IdentityFile $KEY
  IdentitiesOnly yes
EOF
    ok "config block appended"
fi

# ── 5. ssh-copy-id to all nodes ───────────────────────────────────────────
step "5/6 Pushing public key to all 14 nodes"
read -rs -p "Lab SSH password (used 14× via sshpass, then discarded): " SSHPASS
echo
[[ -n "$SSHPASS" ]] || die "empty password"
export SSHPASS

copied=0; already=0; failed=0
for n in "${NODES[@]}"; do
    host="${NETWORK_PREFIX}.${n}"
    out="$(sshpass -e ssh-copy-id -i "${KEY}.pub" -p "$PORT" \
            -o StrictHostKeyChecking=accept-new \
            "${USER_REMOTE}@${host}" 2>&1 || true)"
    if grep -q 'added: 0' <<<"$out"; then
        already=$((already+1)); printf '  • node %3d: already authorized\n' "$n"
    elif grep -qE 'added: [1-9]' <<<"$out"; then
        copied=$((copied+1)); printf '  ✓ node %3d: key copied\n' "$n"
    else
        failed=$((failed+1)); printf '  ✗ node %3d: FAILED\n' "$n"
        echo "$out" | sed 's/^/      /' | tail -3
    fi
done
unset SSHPASS
ok "copied=$copied  already-authorized=$already  failed=$failed"
[[ $failed -eq 0 ]] || warn "Some nodes failed — re-run after fixing (wrong password? node down?)"

# ── 6. Verify ─────────────────────────────────────────────────────────────
step "6/6 Verifying key auth to node 101"
if ssh -o BatchMode=yes -o ConnectTimeout=5 "${USER_REMOTE}@${NETWORK_PREFIX}.101" \
        'echo "OK from $(hostname)"' 2>&1; then
    ok "key auth works"
else
    die "key auth failed — check the failed nodes above"
fi

echo
echo "Done. The training-runner agent can now launch via:"
echo "  python3 run_command.py <node> grid_world_pain \"bash train_command-new.sh\""
