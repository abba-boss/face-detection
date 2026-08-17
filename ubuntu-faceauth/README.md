# Ubuntu FaceAuth — V3

A standalone, local face-recognition engine for Ubuntu 24.04.

**V3 is completely independent of the Ubuntu login system — no PAM, GDM, sudoers, or any system file is touched.**

---

## Environment

| Item | Value |
|------|-------|
| OS | Ubuntu 24.04.4 LTS |
| Python | 3.11.x (conda env `face-detection`) |
| OpenCV | 5.0.0 |
| InsightFace | 0.7.3 |
| ONNX Runtime | 1.28.0 (CPU) |
| scikit-image | 0.26.0 |
| AI model | ArcFace via `buffalo_sc` (~14 MB, downloaded once) |

---

## Quick Start

```bash
conda activate face-detection
cd ~/Desktop/face-detection/ubuntu-faceauth

# 1 — verify your environment
python scripts/check_env.py

# 2 — enroll your face
python main.py enroll --user abba

# 3 — authenticate (liveness + face recognition)
python main.py authenticate --user abba

# 4 — or run continuous live recognition
python main.py recognize
```

Press **Q** or **ESC** to close any camera window.

---

## Commands

### Enroll
```bash
python main.py enroll --user <username>
```
- Opens webcam, shows a 3-second countdown, then collects 10 sharp face samples.
- Saves a single representative embedding to `data/embeddings/<username>.npz`.
- No raw images are stored.

Flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--user` | required | Identity label |
| `--re-enroll` | off | Overwrite an existing enrollment |
| `--samples N` | 10 | Number of samples to collect |
| `--camera N` | 0 | Camera device index (`/dev/videoN`) |
| `--model` | `buffalo_sc` | InsightFace model (`buffalo_sc` or `buffalo_l`) |

### Recognise (live)
```bash
python main.py recognize [--threshold 0.45] [--camera 0] [--debug] [--liveness] [--model buffalo_sc|buffalo_l]
```
Displays `AUTHORIZED` or `UNKNOWN` over each detected face in real time.

`--liveness` enables the **liveness gate**: the first time a face crosses the AUTHORIZED threshold, a head-turn challenge fires before access is confirmed. See [Phase 2B — Liveness Gate](#phase-2b--liveness-gate-in-recognition) below.

`--debug` prints the full per-user cosine similarity table in the terminal — useful for tuning the threshold. Never use in production.

### List enrolled users
```bash
python main.py list
```
Output:
```
Enrolled users (2):
  • abba                  enrolled: 2026-08-11T14:22:05Z
  • user2                 enrolled: 2026-08-10T18:55:03Z
```

### Delete an enrolled user
```bash
python main.py delete --user <username>
```

### Authenticate
```bash
python main.py authenticate --user <username> [--camera 0] [--threshold 0.45]
```
Single-attempt biometric authentication flow: **Liveness → Face Capture → Recognition → Identity Verification**.

- Exit code **0** = SUCCESS
- Exit code **1** = DENIED or ERROR

The user must be enrolled first. No passwords are created, read, or stored.

See [V3 — Authentication Layer](#v3--authentication-layer) for full details.

---

## Scripts

```bash
# Full environment and dependency check
python scripts/check_env.py

# CPU inference speed benchmark
python scripts/benchmark.py
```

---

## How It Works

### AI model — ArcFace (`buffalo_sc` / `buffalo_l`)

InsightFace's model bundles (downloaded once to `~/.insightface/`):
- **RetinaFace** — detects face bounding boxes and 5-point landmarks
- **ArcFace** — generates a 512-d unit-length embedding per face

| Model | Size | Speed | Accuracy | Best for |
|-------|------|-------|----------|----------|
| `buffalo_sc` | ~14 MB | ~32 FPS on i5 | Good | Default — fast CPU |
| `buffalo_l` | ~500 MB | slower | Higher | Strict environments |

Select with `--model`:
```bash
python main.py enroll --user abba --model buffalo_l
python main.py recognize --model buffalo_l
python main.py authenticate --user abba --model buffalo_l
```

`buffalo_l` triggers a one-time ~500 MB download on first use.  
**Re-enroll after switching models** — embeddings from different models are not compatible.

Everything runs on **CPU only**. Measured throughput on an i5-6300U: **~32 FPS** (`buffalo_sc`).

### Enrollment

1. Camera opens and discards 10 warmup frames (allows auto-exposure to settle).
2. A 3-second countdown gives you time to position your face.
3. Each frame is checked:
   - Exactly **one** face must be present.
   - Face width and height must be ≥ 80 px.
   - Laplacian variance of the face crop must be ≥ `enrollment_blur_threshold` (default 20.0).
4. Ten valid embeddings are averaged into one representative vector, then L2-normalised.
5. Saved to `data/embeddings/<username>.npz` with file permissions `0o600`.

### Recognition

1. Every frame: detect faces → generate ArcFace embedding per face.
2. Compute **cosine similarity** against every enrolled user (dot product — embeddings are unit-length).
3. Return the identity with the highest score. If below the threshold → `UNKNOWN`.
4. Results are **EMA-smoothed over 5 frames** (α=0.4 — recent frames weighted higher, majority vote for decision) to prevent display flicker while still tracking rapid changes within 2–3 frames.
5. Enrolled users are cached in memory — disk is read once per session.

### Similarity threshold

Default: **0.45** — override with `--threshold` or `settings.recognition_threshold`.

| Score | Meaning |
|-------|---------|
| 0.60 – 0.95 | Clearly the same person |
| 0.45 – 0.60 | Passes threshold — AUTHORIZED |
| < 0.45 | Different person — UNKNOWN |

### Blur threshold

Default: **20.0** (Laplacian variance of face crop) — `settings.enrollment_blur_threshold`.

Calibrated against a typical USB/built-in webcam (observed range 24–42, mean ~35).
- Raise to 40+ for a high-quality camera.
- Lower to 10–15 if enrollment keeps stalling due to soft optics.

---

## Storage Format

### SQLite (default)

All embeddings are stored in a single file `data/faceauth.db` (mode `0o600`):

```
data/
└── faceauth.db       (mode 0o600 — owner read/write only)
```

Schema:

| Column | Type | Description |
|--------|------|-------------|
| `username` | `TEXT PRIMARY KEY` | Identity label |
| `embedding` | `BLOB` | `float32 (512,)` packed as bytes, L2-normalised |
| `enrolled_at` | `TEXT` | ISO-8601 UTC timestamp |
| `version` | `INTEGER` | Format version (1) |

WAL journal mode is enabled for safe concurrent reads.

### Legacy .npz (backward compatible)

Set `settings.storage_backend = "npz"` to use the original per-user files:

```
data/embeddings/
├── abba.npz      (mode 0o600)
└── user2.npz
```

### Migration

To migrate existing `.npz` files to SQLite:

```bash
python scripts/migrate_to_sqlite.py
```

Idempotent — safe to run multiple times. Original `.npz` files are left in place.

---

## Running Tests

```bash
cd ~/Desktop/face-detection/ubuntu-faceauth
conda activate face-detection

# Unit tests (no camera required) — 222 tests
pytest tests/ --ignore=tests/test_realworld.py -v

# Real-world integration tests (requires /dev/video0)
FACEAUTH_REALWORLD=1 pytest tests/test_realworld.py -v
```

### Test coverage

| File | Tests | What it covers |
|------|-------|----------------|
| `test_config.py` | 11 | Settings defaults, overrides, path creation |
| `test_storage.py` | 18 | CRUD, normalisation, timestamps, permissions, v1 compat |
| `test_embedding_serialization.py` | 5 | Round-trip, dtype, shape, overwrite, cache invalidation |
| `test_recognition.py` | 6 | Threshold logic, multi-user matching, unknown detection |
| `test_quality.py` | 13 | Blur scoring, EMA smoothing (weighted, α=0, α=1, recent-higher), majority vote |
| `test_logger.py` | 12 | Singleton, level, file handler, retroactive handler, no-leak policy |
| `test_camera_unit.py` | 10 | Camera open/read/release with mocked VideoCapture |
| `test_main_cli.py` | 15 | list/delete/enroll-guard/re-enroll/liveness CLI, `--model` flag on all subcommands |
| `test_realworld.py` | 11 | Live camera open, detection, unit-norm embeddings, full enroll+recognise |
| `test_liveness.py` | 40+ | Liveness state machine, NOR calculation, session (all mocked) |
| `test_liveness_gate.py` | 15 | Liveness gate in RecognitionRunner: disabled, pass, fail, timeout, cooldown, expiry, face-disappear, CLI flag, `_run_liveness_inline` unit |
| `test_sqlite_store.py` | 27 | SQLite backend: CRUD, metadata, permissions, round-trip, WAL mode, factory |
| `test_authentication.py` | 22 | V3 AuthSession: not-enrolled, liveness fail/timeout, no face, blurry, below threshold, mismatch, success, error, Q-cancel, CLI exit codes |

---

## Phase 2A — Standalone Liveness Challenge

### Command
```bash
python main.py liveness [--camera 0] [--timeout 8.0]
```
Shows a camera window. Prompts you to turn your head LEFT. Returns exit code 0 on LIVE, 1 on FAILED/TIMEOUT.

### Developer demo (no camera)
```bash
python scripts/test_liveness_demo.py
```
Runs 5 synthetic scenarios through the state machine without opening any hardware.

### How it works

The **nose-offset ratio (NOR)** measures horizontal head pose from 5-point landmarks:

```
NOR = (nose_x − eye_midpoint_x) / eye_span
```

| NOR | Meaning |
|-----|---------|
| ≈ 0.0 | Looking straight ahead |
| < −0.18 | Left turn (passes challenge) |
| > +0.18 | Right turn |

**State flow:**
```
WAITING ──(frontal face)──▶ CHALLENGE_ACTIVE
                                   │
              ┌────────────────────┤
              │                    │
        (face lost)       (NOR drop ≥ threshold
              │            for min_confirm_frames)
              ▼                    │
           FAILED               LIVE

   (elapsed ≥ timeout)
              ▼
           TIMEOUT
```

Terminal states are **immutable** — call `reset()` for a fresh challenge.

### Module structure

```
app/liveness/
├── __init__.py       exports LivenessDetector, LivenessState,
│                             LivenessResult, LivenessConfig, LivenessSession
├── detector.py       state machine + NOR calculation (pure Python/numpy)
├── session.py        camera-driven wrapper (mirrors EnrollmentSession)
└── drawing.py        OpenCV overlay helpers
```

---

## Phase 2B — Liveness Gate in Recognition

The liveness gate integrates the Phase 2A challenge directly into the recognition flow.

### Usage
```bash
# Standard recognition — no liveness required
python main.py recognize

# With liveness gate — challenge fires before access is granted
python main.py recognize --liveness

# Combined with other flags
python main.py recognize --liveness --threshold 0.50 --debug
```

### How it works

```
Face detected → similarity ≥ threshold → AUTHORIZED?
                                              │
                              ┌───────────────┘
                              ▼
                   Liveness gate (--liveness only)
                              │
              ┌───────────────┴───────────────┐
              │                               │
         Challenge fires                Already passed
         (head-turn)                    this session
              │                               │
     ┌────────┴────────┐              Show AUTHORIZED
     │                 │
   LIVE            FAILED / TIMEOUT
     │                 │
 AUTHORIZED        UNKNOWN + 3s cooldown
                   (gate re-arms after cooldown)
```

Key behaviours:

- The gate fires **once per face appearance**. After passing, AUTHORIZED is shown for the rest of that continuous face-detection session without re-challenging.
- On failure or timeout, the face is shown as **UNKNOWN** with a **3-second cooldown** before the gate can re-trigger.
- If the face leaves the frame completely, gate state resets — the next appearance triggers a fresh challenge.
- Without `--liveness`, behaviour is identical to before (no gate, no challenge).

---

## Liveness Configuration

All liveness parameters are in `app/config/settings.py` and shared between Phase 2A and 2B:

| Setting | Default | Description |
|---------|---------|-------------|
| `liveness_timeout` | `8.0` s | Time limit for the head-turn challenge |
| `liveness_left_threshold` | `0.18` | NOR drop required to confirm left turn |
| `liveness_frontal_max` | `0.15` | Max \|NOR\| to lock the baseline |
| `liveness_min_frames` | `3` | Consecutive confirming frames required |

---

## Configuration Reference

All values in `app/config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `camera_device` | `0` | `/dev/videoN` index |
| `camera_width` | `640` | Capture width (px) |
| `camera_height` | `480` | Capture height (px) |
| `camera_fps` | `30` | Target frame rate |
| `camera_warmup_frames` | `10` | Frames discarded after open (auto-exposure) |
| `min_face_size` | `80` | Minimum face dimension (px) |
| `detection_threshold` | `0.5` | RetinaFace confidence cutoff |
| `enrollment_samples` | `10` | Frames to collect per enrollment |
| `enrollment_max_attempts` | `300` | Max frames before enrollment fails |
| `enrollment_blur_threshold` | `20.0` | Laplacian variance minimum |
| `recognition_threshold` | `0.45` | Cosine similarity to authorise |
| `insightface_model_name` | `buffalo_sc` | Model bundle (`buffalo_sc` = fast CPU, `buffalo_l` = higher accuracy) |
| `log_level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `log_to_file` | `True` | Write log to `data/faceauth.log` |
| `show_landmarks` | `False` | Overlay 5-point facial landmarks |
| `font_scale` | `0.7` | Label text size multiplier |
| `liveness_timeout` | `8.0` | Liveness challenge time limit (s) |
| `liveness_left_threshold` | `0.18` | NOR drop to confirm left turn |
| `liveness_frontal_max` | `0.15` | Max \|NOR\| for baseline lock |
| `liveness_min_frames` | `3` | Frames to confirm the turn |

---

## Security Notes

- Raw face photographs are **never stored**.
- Embeddings are stored **locally only** — nothing leaves the machine.
- Embedding files are set to **mode `0o600`** (owner read/write only).
- `data/` is in `.gitignore` — never commit it.
- Face recognition alone **is not sufficient** against sophisticated spoofing (printed photos, video replay).
- Phase 2A adds **standalone head-turn liveness detection** — significantly harder to spoof than static recognition alone.
- Phase 2B adds a **liveness gate in the recognition loop** (`--liveness`) — challenge fires automatically on first AUTHORIZED detection.
- V3 adds a **dedicated `authenticate` command** — single-attempt, user-specific, liveness-gated authentication with identity verification.
- Full anti-spoofing (IR liveness, depth sensing) is a later phase.

---

## V3 — Authentication Layer

Single-attempt, user-specific biometric authentication.  
**Does not touch passwords, PAM, GDM, `/etc/passwd`, or `/etc/shadow`.**

### Usage
```bash
# Basic authentication
python main.py authenticate --user abba

# Override camera or threshold
python main.py authenticate --user abba --camera 0 --threshold 0.50
```

### Flow

```
┌─────────────────────────────────────────────────────┐
│  python main.py authenticate --user abba            │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  1. Pre-flight check    │
          │  Is user enrolled?      │
          └────────────┬────────────┘
               No ─────┘ Yes
               │               │
        DENIED_NOT_ENROLLED     │
                       ┌────────▼────────┐
                       │  2. Liveness    │
                       │  Head-turn      │
                       │  challenge      │
                       └────────┬────────┘
              FAILED/TIMEOUT ───┘ LIVE
              │                        │
       DENIED_LIVENESS        ┌────────▼─────────┐
                               │  3. Face capture │
                               │  Single sharp    │
                               │  frame           │
                               └────────┬─────────┘
              No face ─────────┘ Captured
              │                         │
       DENIED_NO_FACE          ┌────────▼──────────┐
                                │  4. Recognition  │
                                │  Cosine sim vs   │
                                │  enrolled users  │
                                └────────┬──────────┘
              Below threshold ──┘ Authorized
              │                           │
       DENIED_BELOW_THRESHOLD    ┌────────▼──────────┐
                                  │  5. Verify ID    │
                                  │  matched == user?│
                                  └────────┬──────────┘
              Wrong person ───────┘ Match
              │                            │
       DENIED_MISMATCH              ✓ SUCCESS (exit 0)
```

### Outcomes

| Outcome | Exit | When |
|---------|------|------|
| `SUCCESS` | 0 | Liveness LIVE + face matches `--user` |
| `DENIED_NOT_ENROLLED` | 1 | User has no enrollment (no camera opened) |
| `DENIED_LIVENESS` | 1 | Liveness FAILED or TIMEOUT |
| `DENIED_NO_FACE` | 1 | Cannot capture a sharp face within 60 frames |
| `DENIED_BELOW_THRESHOLD` | 1 | Face similarity below threshold (unknown person) |
| `DENIED_MISMATCH` | 1 | Face recognised but belongs to a different user |
| `ERROR` | 1 | Camera failure or unexpected exception |

### Module structure

```
app/auth/
├── __init__.py        exports AuthSession, AuthResult, AuthOutcome
└── authenticator.py   full authentication flow
```

### Security properties

- Requires **liveness** before face capture — static photo attacks blocked.
- Requires **exact identity match** — passing as a different enrolled user → DENIED.
- No passwords read, stored, or modified.
- All existing Ubuntu authentication (password, PAM, GDM) is completely untouched.

---

## Roadmap

1. **Liveness / anti-spoofing** ✅ Phase 2A complete — standalone challenge-response module
2. **Liveness gate in recognition** ✅ Phase 2B complete — `--liveness` flag wires challenge into `recognize`
3. **Multi-frame EMA smoothing** ✅ Complete — α=0.4 weighted average replaces plain mean; majority vote unchanged
4. **Authentication layer** ✅ V3 complete — `authenticate` command: liveness + recognition + identity verification
5. **SQLite-backed storage** ✅ Complete — default backend, WAL mode, `0o600` permissions, migration script, npz fallback kept
6. **PAM module** (`pam_faceauth.so`) — only after thorough real-world testing of this engine.
7. **GDM face-unlock UI** — overlay at the lock screen.
8. **`buffalo_l` model option** ✅ Complete — `--model buffalo_sc|buffalo_l` on all camera commands
9. **Secure IPC** — replace the direct function-call architecture with a privilege-separated daemon.
