#!/bin/bash

# ============================================================
# sync-agent-data.sh — Sync agent data between NAS and local
# Supports: antigravity (Gemini), claude
# ============================================================

NAS_PROJECT="/media/nas01/projects/Interoceptive-AI/grid_world_pain"

# Default values
ACTION=""
TARGET=""
SUBFOLDER=""
DRY_RUN=""
CLAUDE_MULTI_SYNC=false

# --- Helper: resolve paths and excludes per target ---
setup_target() {
    case "$TARGET" in
        antigravity)
            NAS_BASE="${NAS_PROJECT}/antigravity_data/"
            LOCAL_BASE="$HOME/.gemini/antigravity/"
            INCLUDES=""
            EXCLUDES="--exclude=installation_id --exclude=mcp_config.json --exclude=user_settings.pb --exclude=browserOnboardingStatus.txt"
            ;;
        claude)
            # NAS mirrors $HOME structure: claude_data/.claude/ and claude_data/.claude.json
            NAS_BASE="${NAS_PROJECT}/claude_data/"
            LOCAL_BASE="$HOME/"
            INCLUDES=""
            # Exclude machine-specific credentials, transient caches, and session state
            EXCLUDES="--exclude=.credentials.json --exclude=statsig/ --exclude=ide/ --exclude=session-env/ --exclude=shell-snapshots/ --exclude=debug/ --exclude=cache/ --exclude=file-history/ --exclude=mcp-needs-auth-cache.json --exclude=telemetry/"
            CLAUDE_MULTI_SYNC=true
            ;;
        *)
            echo "Error: Unknown target '$TARGET'. Must be 'antigravity' or 'claude'."
            exit 1
            ;;
    esac
}

# --- Usage ---
show_help() {
    echo "Usage: $0 {antigravity|claude} {pull|push} [subfolder] [--dry-run]"
    echo ""
    echo "Target (required):"
    echo "  antigravity   - Gemini Antigravity data (~/.gemini/antigravity/)"
    echo "  claude        - Claude Code data (~/.claude/)"
    echo ""
    echo "Action (required):"
    echo "  pull          - Update local from NAS (downloads newer files)"
    echo "  push          - Backup local to NAS (uploads newer files)"
    echo ""
    echo "Options:"
    echo "  subfolder     - Optional: specific directory (e.g., conversations, projects, config)"
    echo "  --dry-run, -d - Preview changes without executing"
    echo ""
    echo "Examples:"
    echo "  $0 antigravity pull                # Mirror Antigravity NAS → local"
    echo "  $0 antigravity push                # Backup Antigravity local → NAS"
    echo "  $0 claude pull                     # Mirror Claude NAS → local"
    echo "  $0 claude push projects            # Backup only Claude projects to NAS"
    echo "  $0 claude pull --dry-run           # Preview Claude pull"
    exit 1
}

# --- Parse arguments ---
if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        antigravity|claude)
            TARGET="$1"
            shift
            ;;
        pull|push)
            ACTION="$1"
            shift
            ;;
        --dry-run|-d)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            if [ -z "$SUBFOLDER" ]; then
                SUBFOLDER="$1"
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "Error: You must specify a target: 'antigravity' or 'claude'."
    echo "Run '$0 --help' for usage."
    exit 1
fi

if [ -z "$ACTION" ]; then
    echo "Error: You must specify an action: 'pull' or 'push'."
    echo "Run '$0 --help' for usage."
    exit 1
fi

# --- Resolve target-specific config ---
setup_target

# --- Define paths based on subfolder ---
if [ -n "$SUBFOLDER" ]; then
    NAS_PATH="${NAS_BASE}${SUBFOLDER%/}/"
    LOCAL_PATH="${LOCAL_BASE}${SUBFOLDER%/}/"
else
    NAS_PATH="$NAS_BASE"
    LOCAL_PATH="$LOCAL_BASE"
fi

# Ensure directories exist
mkdir -p "$LOCAL_PATH"
mkdir -p "$NAS_PATH"

# --- RSYNC ---
# -a: archive  -v: verbose  -z: compress  -u: skip newer on dest  -L: follow symlinks
RSYNC_OPTS="-avzuL $DRY_RUN"

echo "=== Syncing $TARGET ($ACTION) ==="
echo "  NAS:   $NAS_PATH"
echo "  Local: $LOCAL_PATH"
echo ""

if [ "$CLAUDE_MULTI_SYNC" = true ] && [ -z "$SUBFOLDER" ]; then
    # Claude: sync ~/.claude/ dir + ~/.claude.json file separately
    case "$ACTION" in
        pull)
            echo "Updating local from NAS..."
            rsync $RSYNC_OPTS --delete $EXCLUDES "${NAS_BASE}.claude/" "$HOME/.claude/"
            echo ""
            echo "--- Global config: .claude.json ---"
            [ -f "${NAS_BASE}.claude.json" ] && rsync -avzuL $DRY_RUN "${NAS_BASE}.claude.json" "$HOME/.claude.json"
            ;;
        push)
            echo "Backing up local to NAS..."
            rsync $RSYNC_OPTS $EXCLUDES "$HOME/.claude/" "${NAS_BASE}.claude/"
            echo ""
            echo "--- Global config: .claude.json ---"
            [ -f "$HOME/.claude.json" ] && rsync -avzuL $DRY_RUN "$HOME/.claude.json" "${NAS_BASE}.claude.json"
            ;;
    esac
else
    case "$ACTION" in
        pull)
            echo "Updating local from NAS..."
            [ -z "$SUBFOLDER" ] && RSYNC_OPTS="$RSYNC_OPTS --delete"
            rsync $RSYNC_OPTS $EXCLUDES "$NAS_PATH" "$LOCAL_PATH"
            ;;
        push)
            echo "Backing up local to NAS..."
            rsync $RSYNC_OPTS $EXCLUDES "$LOCAL_PATH" "$NAS_PATH"
            ;;
    esac
fi

echo ""
echo "Done!"
