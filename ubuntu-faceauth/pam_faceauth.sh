#!/usr/bin/env bash
# =============================================================================
# pam_faceauth.sh — PAM wrapper for Ubuntu FaceAuth
#
# Called by pam_exec.so during GDM login.
#
# PAM_USER       = Linux system username
# FACEAUTH_USER  = FaceAuth enrolled username
#
# Exit codes:
#   0 = Face authentication SUCCESS
#   1 = Face authentication failed / unavailable → password fallback
# =============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────

FACEAUTH_DIR="/home/abba-boss/Desktop/face-detection/ubuntu-faceauth"

PYTHON="/home/abba-boss/miniconda3/envs/face-detection/bin/python3"

INSIGHTFACE_HOME="/home/abba-boss"

TIMEOUT_SECONDS=30

# ── Logging ───────────────────────────────────────────────────────────────────

if [[ -w /var/log/faceauth_pam.log ]] || touch /var/log/faceauth_pam.log 2>/dev/null; then
    WRAPPER_LOG="/var/log/faceauth_pam.log"
else
    WRAPPER_LOG="/tmp/faceauth_pam.log"
fi

log() {
    echo "$(date '+%F %T') [faceauth] $*" >> "$WRAPPER_LOG"
}

# ── Runtime dependency checks ─────────────────────────────────────────────────

if [[ ! -d "$FACEAUTH_DIR" ]]; then
    log "ERROR: FaceAuth directory missing: $FACEAUTH_DIR"
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    log "ERROR: Python interpreter missing/not executable: $PYTHON"
    exit 1
fi

if [[ ! -f "$FACEAUTH_DIR/main.py" ]]; then
    log "ERROR: main.py missing: $FACEAUTH_DIR/main.py"
    exit 1
fi

# ── PAM_USER check ────────────────────────────────────────────────────────────

if [[ -z "$PAM_USER" ]]; then
    log "ERROR: PAM_USER not set"
    exit 1
fi

# ── PAM_USER → FaceAuth username mapping ──────────────────────────────────────
#
# Linux account:
#   abba-boss
#
# FaceAuth enrolled user:
#   abba
#
# Add additional mappings here when needed.
#
declare -A USER_MAP=(
    ["abba-boss"]="abba"
)

FACEAUTH_USER="${USER_MAP[$PAM_USER]:-$PAM_USER}"

log "PAM wrapper called: pam_user=$PAM_USER → faceauth_user=$FACEAUTH_USER"

# ── User sanity check ─────────────────────────────────────────────────────────

USER_SHELL=$(getent passwd "$PAM_USER" | cut -d: -f7)

if [[ -z "$USER_SHELL" ]]; then
    log "SKIP: Linux user '$PAM_USER' does not exist"
    exit 1
fi

if [[ "$USER_SHELL" == */nologin || "$USER_SHELL" == */false ]]; then
    log "SKIP: $PAM_USER is a system account"
    exit 1
fi

# ── FaceAuth data ─────────────────────────────────────────────────────────────

FACEAUTH_DATA_DIR="/home/abba-boss/Desktop/face-detection/data"

SQLITE_DB="$FACEAUTH_DATA_DIR/faceauth.db"

EMBEDDINGS_DIR="$FACEAUTH_DATA_DIR/embeddings"

# ── Check enrollment ──────────────────────────────────────────────────────────

enrolled=0

# SQLite backend
if [[ -f "$SQLITE_DB" ]]; then
    count=$(
        sqlite3 "$SQLITE_DB" \
        "SELECT COUNT(*) FROM users WHERE username='$FACEAUTH_USER';" \
        2>/dev/null || echo 0
    )

    if [[ "$count" -gt 0 ]]; then
        enrolled=1
    fi
fi

# Legacy NPZ backend
if [[ "$enrolled" -eq 0 &&
      -f "$EMBEDDINGS_DIR/${FACEAUTH_USER}.npz" ]]; then
    enrolled=1
fi

if [[ "$enrolled" -eq 0 ]]; then
    log "SKIP: pam_user=$PAM_USER → faceauth_user=$FACEAUTH_USER not enrolled"
    exit 1
fi

log "Enrollment confirmed for faceauth_user=$FACEAUTH_USER"

# ── Camera check ──────────────────────────────────────────────────────────────

if [[ ! -e /dev/video0 ]]; then
    log "SKIP: /dev/video0 not found"
    exit 1
fi

if [[ ! -r /dev/video0 || ! -w /dev/video0 ]]; then
    log "WARNING: /dev/video0 permissions may restrict camera access"
fi

# ── Run FaceAuth ──────────────────────────────────────────────────────────────

log "START: pam_user=$PAM_USER → faceauth_user=$FACEAUTH_USER"

timeout "$TIMEOUT_SECONDS" \
    env HOME="$INSIGHTFACE_HOME" \
    "$PYTHON" \
    "$FACEAUTH_DIR/main.py" \
    authenticate \
    --user "$FACEAUTH_USER" \
    --headless \
    >> "$WRAPPER_LOG" 2>&1

EXIT_CODE=$?

# ── PAM result ─────────────────────────────────────────────────────────────────

if [[ "$EXIT_CODE" -eq 0 ]]; then

    log "SUCCESS: $PAM_USER (faceauth=$FACEAUTH_USER) authenticated via face"

    exit 0

elif [[ "$EXIT_CODE" -eq 124 ]]; then

    log "TIMEOUT: $PAM_USER — falling through to password"

    exit 1

else

    log "FAIL: $PAM_USER — FaceAuth exit code $EXIT_CODE — falling through to password"

    exit 1

fi
