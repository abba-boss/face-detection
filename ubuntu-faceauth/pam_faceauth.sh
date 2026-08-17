#!/usr/bin/env bash
# =============================================================================
# pam_faceauth.sh — PAM wrapper for Ubuntu FaceAuth
#
# Called by pam_exec.so during GDM login.
# PAM sets $PAM_USER to the Linux system username being authenticated.
#
# Exit codes (PAM contract):
#   0  = authentication SUCCESS  → PAM grants access
#   1  = authentication FAILURE  → PAM falls through to password
#
# Security notes:
#   • This script runs as root (invoked by PAM).
#   • It must be owned root:root, permissions 755.
#   • No user-writable paths are on the exec path.
# =============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────

# Absolute path to the FaceAuth project (main.py lives here)
FACEAUTH_DIR="/home/abba-boss/Desktop/face-detection/ubuntu-faceauth"

# Python interpreter — must be the conda env that has cv2, insightface, onnxruntime
PYTHON="/home/abba-boss/miniconda3/envs/face-detection/bin/python3"

# HOME override: when PAM calls us as root, HOME=/root and InsightFace
# can't find its model cache. Set HOME to the real user's home so that
# InsightFace resolves ~/.insightface/models correctly.
INSIGHTFACE_HOME="/home/abba-boss"

# Timeout (seconds) breakdown:
#   ~5 s  model load (InsightFace buffalo_sc, cached)
#   ~1 s  camera warmup
#   8 s   liveness challenge window (default)
#   ~1 s  post-liveness settle (15 frames @ 30fps)
#   ~4 s  capture 8 samples @ 30fps
#   5 s   headroom
# Total: 30 s. If the process hangs beyond this, kill it and fall through.
TIMEOUT_SECONDS=30

# Log file for wrapper-level events (not FaceAuth's own log)
# When running as root (PAM), this will be /var/log/faceauth_pam.log.
# When running as a regular user for testing, fall back to a writable location.
if [[ -w /var/log/faceauth_pam.log ]] || touch /var/log/faceauth_pam.log 2>/dev/null; then
    WRAPPER_LOG="/var/log/faceauth_pam.log"
else
    WRAPPER_LOG="/tmp/faceauth_pam.log"
fi

# ── PAM_USER → FaceAuth username mapping ──────────────────────────────────────
#
# PAM_USER is the Linux system account (e.g. "abba-boss").
# FaceAuth may enroll users under a different name (e.g. "abba").
# Add one entry per user: ["linux_username"]="faceauth_enrolled_name"
# If no mapping exists, FACEAUTH_USER falls back to PAM_USER unchanged.
#
declare -A USER_MAP=(
    ["abba-boss"]="abba"
)

# ── Sanity checks ─────────────────────────────────────────────────────────────

# PAM_USER must be set
if [[ -z "$PAM_USER" ]]; then
    echo "$(date '+%F %T') [faceauth] ERROR: PAM_USER not set" >> "$WRAPPER_LOG"
    exit 1   # fail closed → falls through to password
fi

# Resolve FaceAuth username via the map; default to PAM_USER if no entry
FACEAUTH_USER="${USER_MAP[$PAM_USER]:-$PAM_USER}"

# Only run face auth for real human users (skip root, system accounts)
# /etc/passwd: if shell is /usr/sbin/nologin or /bin/false → skip
USER_SHELL=$(getent passwd "$PAM_USER" | cut -d: -f7)
if [[ "$USER_SHELL" == */nologin || "$USER_SHELL" == */false ]]; then
    echo "$(date '+%F %T') [faceauth] SKIP: $PAM_USER is a system account" >> "$WRAPPER_LOG"
    exit 1   # fall through to password immediately
fi

# Check the FaceAuth user is enrolled — avoids opening camera for unenrolled users
# FaceAuth data_dir is 3 parents up from settings.py:
#   ubuntu-faceauth/app/config/settings.py → parents[3] = face-detection/data
FACEAUTH_DATA_DIR="/home/abba-boss/Desktop/face-detection/data"
SQLITE_DB="$FACEAUTH_DATA_DIR/faceauth.db"
EMBEDDINGS_DIR="$FACEAUTH_DATA_DIR/embeddings"

enrolled=0
# Check SQLite backend first
if [[ -f "$SQLITE_DB" ]]; then
    count=$(sqlite3 "$SQLITE_DB" "SELECT COUNT(*) FROM users WHERE username='$FACEAUTH_USER';" 2>/dev/null || echo 0)
    [[ "$count" -gt 0 ]] && enrolled=1
fi
# Fallback: check legacy npz files
if [[ $enrolled -eq 0 && -f "$EMBEDDINGS_DIR/${FACEAUTH_USER}.npz" ]]; then
    enrolled=1
fi

if [[ $enrolled -eq 0 ]]; then
    echo "$(date '+%F %T') [faceauth] SKIP: pam_user=$PAM_USER → faceauth_user=$FACEAUTH_USER not enrolled" >> "$WRAPPER_LOG"
    exit 1   # not enrolled → fall through to password
fi

# Camera device must exist and be accessible
if [[ ! -e /dev/video0 ]]; then
    echo "$(date '+%F %T') [faceauth] SKIP: /dev/video0 not found" >> "$WRAPPER_LOG"
    exit 1   # no camera → fall through to password
fi

# ── Run FaceAuth ──────────────────────────────────────────────────────────────

echo "$(date '+%F %T') [faceauth] START: pam_user=$PAM_USER → faceauth_user=$FACEAUTH_USER" >> "$WRAPPER_LOG"

# --headless: no cv2.imshow/waitKey, no DISPLAY required — works inside PAM/GDM.
# HOME override ensures InsightFace finds its model cache at ~/.insightface/models
# Run under timeout so a hung camera never blocks the login screen.
timeout "$TIMEOUT_SECONDS" \
    env HOME="$INSIGHTFACE_HOME" \
    "$PYTHON" "$FACEAUTH_DIR/main.py" authenticate --user "$FACEAUTH_USER" --headless \
    >> "$WRAPPER_LOG" 2>&1

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "$(date '+%F %T') [faceauth] SUCCESS: $PAM_USER (faceauth=$FACEAUTH_USER) authenticated via face" >> "$WRAPPER_LOG"
    exit 0   # ✓ face auth passed → PAM grants access, skips password
elif [[ $EXIT_CODE -eq 124 ]]; then
    echo "$(date '+%F %T') [faceauth] TIMEOUT: $PAM_USER — fell through to password" >> "$WRAPPER_LOG"
    exit 1   # timeout → fall through to password
else
    echo "$(date '+%F %T') [faceauth] FAIL: $PAM_USER — exit $EXIT_CODE — fell through to password" >> "$WRAPPER_LOG"
    exit 1   # face denied → fall through to password
fi
