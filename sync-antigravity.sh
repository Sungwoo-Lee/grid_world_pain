#!/bin/bash

# Configuration
NAS_BASE="/media/nas01/projects/Interoceptive-AI/grid_world_pain/antigravity_data/"
LOCAL_BASE="$HOME/.gemini/antigravity/"

# Default values
ACTION=""
SUBFOLDER=""
DRY_RUN=""

# Parse arguments
if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 {pull|push} [subfolder] [--dry-run]"
    echo ""
    echo "Mandatory Action:"
    echo "  pull          - Update local from NAS (downloads newer files)"
    echo "  push          - Backup local to NAS (uploads newer files)"
    echo ""
    echo "Options:"
    echo "  subfolder     - Optional: specific directory (e.g., conversations, brain, knowledge)"
    echo "  --dry-run, -d - Preview changes without executing"
    echo ""
    echo "Examples:"
    echo "  $0 pull                 # Mirror NAS to local (sync everything)"
    echo "  $0 push                 # Backup everything to NAS"
    echo "  $0 pull conversations   # Update only conversations"
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
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

if [ -z "$ACTION" ]; then
    echo "Error: You must specify either 'pull' or 'push' as the first argument."
    echo "Run '$0 --help' for detailed usage."
    exit 1
fi

# Define paths based on subfolder
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

# RSYNC Options
# --delete: Remove files from destination that don't exist in source (Syncing)
# --exclude: Avoid machine-specific or configuration files that might break other nodes
# --copy-links (-L): Transform symlinks into referent files/dirs (Crucial for NAS without symlink support)
RSYNC_OPTS="-avzuL $DRY_RUN"
EXCLUDES="--exclude=installation_id --exclude=mcp_config.json --exclude=user_settings.pb --exclude=browserOnboardingStatus.txt"

case "$ACTION" in
    pull)
        echo "Updating local from NAS ($SUBFOLDER)..."
        [ -z "$SUBFOLDER" ] && RSYNC_OPTS="$RSYNC_OPTS --delete"
        rsync $RSYNC_OPTS $EXCLUDES "$NAS_PATH" "$LOCAL_PATH"
        ;;
    push)
        echo "Backing up local to NAS ($SUBFOLDER)..."
        rsync $RSYNC_OPTS $EXCLUDES "$LOCAL_PATH" "$NAS_PATH"
        ;;
esac

echo "Done!"