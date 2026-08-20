"""
FaceAuth log reader for `python main.py logs`.

Reads the existing FaceAuth log file and surfaces only meaningful
authentication events — filtering out per-frame noise (per-frame
similarity scores, camera open/close, model init lines).

Security contract:
  - Raw face embeddings are never written to the log file and
    therefore never appear here.
  - Passwords are never written to the log file.
  - No additional filtering of credentials is needed beyond what
    is already guaranteed by the logger.
  - The similarity score shown is an aggregate auth outcome value,
    not a stream of per-frame scores.

Log format produced by security/logger.py:
  YYYY-MM-DD HH:MM:SS | LEVEL    | module.path | message
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

# ── Line patterns that represent meaningful auth/enrollment events ────────

# Keep: lines whose module OR message contains one of these keywords.
_KEEP_MODULES = {
    "faceauth.main",
    "app.auth.authenticator",
    "app.auth.headless",
    "app.enrollment.enroll",
}

_KEEP_MESSAGE_PATTERNS = [
    re.compile(r"Authentication outcome"),
    re.compile(r"Authentication (SUCCESS|DENIED|started|error)", re.I),
    re.compile(r"Enrollment (completed|blocked|started|failed)", re.I),
    re.compile(r"Liveness challenge (passed|failed)", re.I),
    re.compile(r"\[headless\] (SUCCESS|FAIL|TIMEOUT|Authentication)", re.I),
    re.compile(r"Ubuntu FaceAuth started"),
]

# Drop: noisy lines even if they come from a kept module.
_DROP_MESSAGE_PATTERNS = [
    re.compile(r"recognition cache"),       # "Loaded N enrolled users into cache"
    re.compile(r"Known face —"),            # per-frame recognition scores
    re.compile(r"Unknown face —"),          # per-frame unknown scores
    re.compile(r"InsightFace model"),       # model init/download noise
    re.compile(r"Camera (ready|released|Opening)"),  # camera lifecycle noise
    re.compile(r"Initialising"),
    re.compile(r"download_path"),
    re.compile(r"Applied providers"),
    re.compile(r"find model"),
    re.compile(r"set det-size"),
]

# The structured log line format: timestamp | level | module | message
_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"\s*\|\s*(?P<level>\w+)\s*"
    r"\|\s*(?P<module>[\w.]+)\s*"
    r"\|\s*(?P<message>.+)$"
)

_DEFAULT_LIMIT = 20


def _is_interesting(module: str, message: str) -> bool:
    """Return True if this log line should be shown to the user."""
    # First: drop noisy lines unconditionally
    for pattern in _DROP_MESSAGE_PATTERNS:
        if pattern.search(message):
            return False

    # Keep: lines from key modules
    if module in _KEEP_MODULES:
        return True

    # Keep: lines whose message matches an interesting pattern
    for pattern in _KEEP_MESSAGE_PATTERNS:
        if pattern.search(message):
            return True

    return False


def _format_line(ts: str, level: str, module: str, message: str) -> str:
    """Format a log line for display."""
    # Shorten module path for readability: faceauth.main → main
    short_mod = module.split(".")[-1]
    level_short = level[:4].upper()
    return f"{ts}  {level_short:<4}  [{short_mod}]  {message}"


def read_logs(log_file: Path, limit: int = _DEFAULT_LIMIT) -> List[str]:
    """
    Read *log_file* and return the last *limit* interesting lines,
    formatted for display.

    Returns an empty list if the file does not exist or is empty.
    Never raises — errors produce an empty list.
    """
    if not log_file.exists():
        return []

    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    interesting: List[str] = []

    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        m = _LOG_LINE_RE.match(raw_line)
        if not m:
            continue

        ts      = m.group("ts")
        level   = m.group("level")
        module  = m.group("module")
        message = m.group("message").strip()

        if _is_interesting(module, message):
            interesting.append(_format_line(ts, level, module, message))

    # Return the most recent *limit* lines
    return interesting[-limit:]


def run_logs(log_file: Path, limit: int = _DEFAULT_LIMIT) -> int:
    """
    Print recent FaceAuth authentication events.

    Parameters
    ----------
    log_file : Path   Path to the faceauth.log file from Settings.
    limit    : int    Maximum number of lines to show.

    Returns exit code 0 always (read-only, informational command).
    """
    lines = read_logs(log_file, limit)

    if not lines:
        if not log_file.exists():
            print(f"No log file found at: {log_file}")
            print("FaceAuth has not been run yet.")
        else:
            print("No authentication events found in the log.")
        return 0

    print(f"Recent FaceAuth events (last {len(lines)}):")
    print()
    for line in lines:
        print(line)
    return 0
