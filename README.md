# Intelligent Face Tracker with Auto-Registration and Visitor Counting

## Problem Statement

Retail stores, offices, and event venues need to know **how many unique
people** enter a space — not how many times a motion sensor fired. Existing
cheap counters over-count people who walk in and out repeatedly, and manual
counting does not scale. This project builds an AI-driven pipeline that
detects faces in a video stream, learns each visitor's identity on first
sight, re-identifies them on later visits, and maintains an accurate unique
visitor count with timestamped photographic evidence.

## Objective

Process a video stream (local file now, RTSP camera later) to:

1. **Detect** faces with a YOLO object-detection model.
2. **Recognize** faces with InsightFace/ArcFace embeddings.
3. **Automatically register** each new person with a unique Face ID.
4. **Track** people continuously across frames.
5. **Log exactly one ENTRY and one EXIT event** per visit, with cropped
   images and metadata stored in SQLite and on disk.
6. **Count unique visitors** accurately — never double-counting a person
   who is seen in many frames, and never re-registering someone who leaves
   and comes back.

## Features

- YOLOv8 face detection with configurable confidence and **frame skipping**
- InsightFace/ArcFace 512-d embeddings with configurable similarity threshold
- Automatic registration of new faces (unique Face ID + reference image)
- Duplicate-registration prevention via embedding matching
- IOU + centroid multi-object tracker (ByteTrack-style, zero extra deps)
- Exactly-one ENTRY / exactly-one EXIT logging per visit
- SQLite storage: `faces`, `embeddings`, `events` tables
- Mandatory `events.log` (Python `logging`, file + console handlers)
- Dated image folders: `logs/entries/YYYY-MM-DD/`, `logs/exits/YYYY-MM-DD/`
- `config.json`-driven — no hard-coded paths or thresholds
- Clean file/RTSP source abstraction with automatic RTSP reconnection
- Graceful failure for missing files, bad URLs, model errors, DB errors
- Unit tests covering config, DB, registration, matching, counting, events

## Architecture

```mermaid
flowchart TD
    A[Video File / RTSP Stream] --> B[Frame Capture<br/>src/video_source.py]
    B --> C[YOLO Face Detection<br/>src/detector.py<br/>every Nth frame]
    C --> D[Tracking<br/>src/tracker.py<br/>IOU + centroid association]
    D --> E[Face Crop<br/>main.py]
    E --> F[InsightFace / ArcFace Embedding<br/>src/recognizer.py]
    F --> G{Face Matching<br/>cosine similarity vs threshold}
    G -->|Match| H[Existing Face ID<br/>re-identify, no new count]
    G -->|No match| I[New Registration<br/>src/registration.py]
    H --> J[Face ID assigned to track]
    I --> J
    J --> K{Track lifecycle}
    K -->|First frame with Face ID| L[ENTRY Event<br/>exactly once]
    K -->|Disappears > N frames| M[EXIT Event<br/>exactly once]
    L --> N[(SQLite DB<br/>src/database.py)]
    M --> N
    L --> O[Local Images<br/>logs/entries, logs/exits]
    M --> O
    L --> P[events.log<br/>src/event_logger.py]
    M --> P
    N --> Q[Unique Visitor Count<br/>src/visitor_counter.py]
```

## Project Structure

```
face_tracker_project/
│
├── main.py                  # Pipeline orchestrator / entry point
├── config.json              # All configurable values
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── events.log               # Mandatory log file (created at runtime)
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Config loader + validation
│   ├── database.py          # SQLite layer (faces/embeddings/events)
│   ├── detector.py          # YOLO face detection wrapper
│   ├── recognizer.py        # InsightFace embeddings + matching
│   ├── tracker.py           # IOU/centroid multi-object tracker
│   ├── registration.py      # Auto-registration, duplicate prevention
│   ├── event_logger.py      # Logging setup + ENTRY/EXIT persistence
│   ├── visitor_counter.py   # Unique visitor counting
│   ├── video_source.py      # File/RTSP abstraction
│   └── utils.py             # Shared helpers (IoU, cosine sim, paths)
│
├── models/                  # YOLO weights go here
├── data/                    # Sample video, DB, registered faces
├── logs/
│   ├── entries/YYYY-MM-DD/  # ENTRY face crops
│   └── exits/YYYY-MM-DD/    # EXIT face crops
├── output/                  # Optional annotated output video
├── tests/                   # pytest suite
└── docs/
    ├── AI_PLANNING.md       # AI planning document
    └── INTERVIEW_GUIDE.md   # Interview preparation guide
```

## Technologies Used

| Concern | Technology |
|---|---|
| Language | Python 3.10+ (developed/tested on 3.13) |
| Face detection | YOLOv8 via Ultralytics |
| Face recognition | InsightFace / ArcFace (buffalo_l, ONNX Runtime) |
| Tracking | Custom IOU + centroid tracker (ByteTrack-style logic) |
| Video/frames | OpenCV (`cv2.VideoCapture`, `cv2.imwrite`) |
| Database | SQLite (stdlib `sqlite3`, parameterized queries) |
| Config | JSON (`config.json`) |
| Logging | Python `logging` → `events.log` + console |
| Tests | pytest |

## Installation / Setup

### Python version

- **Python 3.10 – 3.13** (this project was built and tested on 3.13.0)

### Dependency installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Model setup / download

**1. YOLO face model** (`models/yolov8n-face.pt`)

```bash
# Option A: download a community face-tuned YOLOv8 weights file and place
# it at models/yolov8n-face.pt (e.g. from face-detection model zoos).

# Option B (demo fallback): let Ultralytics download the default COCO
# model (detects persons; works for a demo, less precise on faces):
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Update `config.json -> detection.model_path` if you use a different file.

**2. InsightFace ArcFace model** (`buffalo_l`)

InsightFace downloads it automatically on first run into
`~/.insightface/models/buffalo_l`. No manual step is required if you have
internet access. To pre-download:

```bash
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0)"
```

### Sample video

Place any video containing faces at `data/sample_video.mp4` (path is
configurable via `config.json -> video_source.file_path`), or generate a
synthetic one:

```bash
python tests/make_sample_video.py
```

## Configuration Instructions

All behavior is controlled by `config.json`:

| Key | Meaning |
|---|---|
| `video_source.type` | `"file"` or `"rtsp"` |
| `video_source.file_path` | Local video path (relative to project root) |
| `video_source.rtsp_url` | RTSP URL used when type is `"rtsp"` |
| `video_source.reconnect_attempts` | RTSP reconnect tries before giving up |
| `detection.model_path` | YOLO weights location |
| `detection.confidence_threshold` | Minimum detection confidence |
| `detection.frame_skip` | **Run detection only every Nth frame** |
| `detection.device` | `"cpu"` or `"cuda"` / `"0"` for GPU |
| `recognition.similarity_threshold` | Cosine similarity needed for a match |
| `recognition.model_name` | InsightFace model pack (`buffalo_l`) |
| `recognition.crop_margin` | Padding around YOLO box before embedding (needs ≥0.3) |
| `recognition.min_crop_size_px` | Upscale small crops to this short-side size |
| `tracking.max_disappeared_frames` | Missed frames before an EXIT event |
| `tracking.iou_match_threshold` | Min IoU to associate boxes |
| `tracking.max_distance_px` | Fallback centroid match radius |
| `database.path` | SQLite file location |
| `storage.*_dir` | Image folders (entries/exits/reference faces) |
| `logging.log_file` | Log file name (`events.log`) |
| `processing.display_window` | Show live OpenCV window (`true`/`false`) |
| `processing.min_face_size_px` | Ignore tiny detections |

## How to Run with a Video

```bash
cd face_tracker_project
python main.py --config config.json
```

The default `config.json` runs headless (`display_window: false`) and ends
at the last frame, printing the unique visitor count, ENTRY and EXIT
totals, and writing everything to `events.log` and the database.

To watch the live annotated view (green boxes + Face IDs + counter overlay),
install the GUI build of OpenCV (`pip install opencv-python`) and set
`processing.display_window` to `true` — press `q` to stop early.

## How to Configure RTSP

1. Edit `config.json`:
   ```json
   "video_source": { "type": "rtsp", "rtsp_url": "rtsp://user:pass@192.168.1.10:554/stream" }
   ```
2. Run `python main.py` — no code changes needed.
3. If the connection drops, the source automatically retries
   `reconnect_attempts` times with `reconnect_delay_seconds` between tries.
   Invalid URLs fail fast with a clear error in `events.log`.

## Database Explanation

SQLite file at `data/face_tracker.db`, created automatically:

- **`faces`** — `face_id` (PK), `registered_at`, `reference_image_path`,
  `total_sightings`, `last_seen_at`. One row per unique person.
- **`embeddings`** — `id` (PK), `face_id` (FK), `embedding` (BLOB of
  float32 vector), `created_at`. Room for multiple embeddings per person.
- **`events`** — `id` (PK), `face_id` (FK), `track_id`, `event_type`
  (`ENTRY`/`EXIT` CHECK), `timestamp`, `image_path`, `similarity`.

All queries are parameterized (`?` placeholders); connections are closed
in `finally` blocks.

```bash
python -c "import sqlite3; c=sqlite3.connect('data/face_tracker.db'); \
print('faces:', c.execute('SELECT COUNT(*) FROM faces').fetchone()[0]); \
print('entries:', c.execute(\"SELECT COUNT(*) FROM events WHERE event_type='ENTRY'\").fetchone()[0]); \
print('exits:', c.execute(\"SELECT COUNT(*) FROM events WHERE event_type='EXIT'\").fetchone()[0])"
```

## Logging Explanation

`events.log` is written by `src/event_logger.py -> setup_logging()` with
format `timestamp | LEVEL | logger | message`. Covered events: detection
counts, embedding generation, recognition matches, registration, tracking
lifecycle (new/removed tracks), ENTRY/EXIT rows, DB operations, model
loading, video source problems, and all exceptions. Levels: `DEBUG`
(verbose), `INFO` (normal flow), `WARNING` (recoverable issues),
`ERROR`/`CRITICAL` (failures).

## Face Registration Explanation

When a track has no Face ID yet, its crop goes to InsightFace for a 512-d
embedding. That embedding is compared (cosine similarity) against every
known embedding. If **nothing** clears `similarity_threshold`, the face is
new: `registration.py` generates `FACE-<hex>`, saves the crop to
`data/registered_faces/`, inserts the `faces` + `embeddings` rows, and
updates the in-memory cache. Because every subsequent frame matches the
cache above threshold, the same person is never registered twice.

## Recognition Explanation

ArcFace maps each face to a point on a 512-d unit sphere: same person ⇒
points close together (cosine similarity ≈ 1), different people ⇒ lower
similarity. `recognizer.find_best_match()` scans the cache, returns the
highest-scoring Face ID **only if** `score >= threshold`, otherwise `None`
(unknown). The threshold trades false links against duplicate identities
and is tunable per deployment.

## Tracking Explanation

`tracker.py` keeps a dict of active `Track` objects keyed by a monotonic
`track_id`. Each frame it greedily associates detections to tracks by IoU
first, then by centroid distance (fallback for small/fast motion). Each
track stores its bound `face_id` once recognition succeeds, so identity
persists even while detection is skipped on intermediate frames.

## Entry/Exit Detection

- **ENTRY** fires exactly once when a track first obtains a Face ID
  (`entry_logged` flag on the track).
- **EXIT** fires exactly once when a track goes unmatched for more than
  `max_disappeared_frames` consecutive frames — i.e., the person left the
  view. Remaining active tracks at shutdown are flushed as EXIT events too.

## Unique Visitor Counting

Count = number of distinct registered Face IDs. A person seen in 1,000
frames has one Face ID → counted once. Someone who leaves and returns is
re-identified by embedding → same Face ID → still counted once. The count
is retrievable via `VisitorCounter.persisted_count()` (backed by
`SELECT COUNT(*) FROM faces`) and printed in the session summary and
`events.log`.

## Testing Instructions

```bash
cd face_tracker_project
python -m pytest tests -q
```

The suite covers: configuration loading/validation, database CRUD,
auto-registration (incl. duplicate prevention), recognition matching,
unique visitor counting, ENTRY/EXIT event creation (image + DB row), and
tracker lifecycle. Tests do **not** require model downloads — the heavy
models are only needed for the full `main.py` run.

## Sample Output

All numbers below are from **actual verified runs** of this project on
`data/sample_video.mp4` (Intel `face-demographics-walking.mp4`, 732 frames)
and `data/reid_sample.mp4` (a leave-and-return replay built by
`tests/make_reid_video.py`).

**events.log**
```
2026-10-03 00:00:43 | INFO | face_tracker | Registered new face FACE-f9d5291f (image=...data\registered_faces\FACE-f9d5291f_...jpg)
2026-10-03 00:00:43 | INFO | face_tracker | Unique visitor count incremented -> 1 (new: FACE-f9d5291f)
2026-10-03 00:00:43 | INFO | face_tracker | ENTRY event | face_id=FACE-f9d5291f track_id=1 ... image=...logs\entries\2026-10-03\FACE-f9d5291f_....jpg
2026-10-03 00:00:51 | INFO | face_tracker | EXIT event | face_id=FACE-f9d5291f track_id=1 image=...logs\exits\2026-10-03\FACE-f9d5291f_....jpg
2026-10-03 00:02:26 | INFO | face_tracker | === Session summary: unique_visitors=7 entries=7 exits=7 ===
```

**Console**
```
Unique visitors counted: 7
Total ENTRY events: 7
Total EXIT events: 7
```

**Image folders** — verified after the run:
7 files in `logs/entries/2026-10-03/`, 7 files in `logs/exits/2026-10-03/`,
7 reference images in `data/registered_faces/`.

**SQLite (main run)** — `faces: 7`, `embeddings: 7`,
`events: ENTRY 7, EXIT 7`.

**Re-identification test** (`python main.py --config config_reid.json`
on the leave-and-return replay video):
```
Re-identified existing face FACE-5070d27a (sim=0.768) on track 3
Re-identified existing face FACE-696977f9 (sim=0.890) on track 4
Unique visitors counted: 2      <- NOT 4: same people, second visit
Total ENTRY events: 4           <- re-entry still logs an ENTRY
Total EXIT events: 4
```
DB shows exactly `2` face rows with 4 ENTRY + 4 EXIT events — proving a
person who leaves and returns is **re-identified, not re-registered**.

**How to reproduce both runs**
```bash
python main.py --config config.json        # main demo (7 visitors on our sample)
python tests/make_reid_video.py            # build leave-and-return test video
python main.py --config config_reid.json   # re-id proof (2 visitors, 4 entries)
```
(`config_reid.json` is a copy of `config.json` pointing at
`data/reid_sample.mp4` and the separate `data/reid_test.db`.)

## Assumptions

- One camera view; people are visible enough for YOLO to detect a face.
- The sample video is a front-ish camera (faces not fully occluded).
- `buffalo_l` runs acceptably on CPU for a hackathon demo.
- The unique-visitor count is scoped to the database it runs against
  (delete `data/face_tracker.db` to reset).
- Correctness is validated against a sample video whose distinct-person
  count is known.

## Known Limitations

- IOU/centroid tracking is simpler than ByteTrack/DeepSORT; fast
  crossings of two people may swap identities.
- Matching uses a single reference embedding per person (multi-embedding
  support exists in the schema but is not yet populated).
- No age/pose filtering — side-profile faces may fail embedding and be
  re-registered later (mitigated by threshold tuning).
- Not stress-tested on real RTSP hardware (code path is implemented and
  reconnect logic exists, but was not exercised against a live camera).

## CPU/GPU Compute Requirements

| Component | CPU (i5/Ryzen 5 class) | GPU (optional) |
|---|---|---|
| YOLOv8n detection (every Nth frame) | ~10–30 ms/frame | ~2–5 ms/frame |
| InsightFace buffalo_l embedding | ~25–60 ms (only on new tracks) | ~5–10 ms |
| Tracker + DB (pure Python/SQLite) | < 1 ms/frame | n/a |

**Estimated load:** ~10–20 FPS end-to-end on a modern CPU with
`frame_skip=2`; 25–30+ FPS on GPU. Embedding generation runs only when a
track lacks a Face ID, so steady-state cost is dominated by detection.
Memory: < 500 MB with buffalo_l on CPU (models ~300 MB on disk).

## Future Improvements

- Swap in real ByteTrack/DeepSORT (supervision or boxmot).
- Store multiple embeddings per face and use a moving average.
- Re-identification embedding refresh on each sighting.
- Web dashboard (FastAPI + live count endpoint).
- PostgreSQL/MySQL migration behind the same `Database` interface.
- Face quality gating (blur/pose) before registration.
- Docker container + docker-compose for one-command demo.

## AI-Assisted Development Workflow

This project was developed with an AI coding agent (Cline) acting as a
senior CV engineer: the workflow was (1) inspect the environment and
dependencies, (2) produce a staged implementation plan, (3) implement
module-by-module with docstrings and type hints, (4) write unit tests for
every core behavior and run them after each stage, (5) fix failures before
moving on, and (6) document architecture (`docs/AI_PLANNING.md`) and
prepare interview explanations (`docs/INTERVIEW_GUIDE.md`). All claims in
this README about tested behavior are backed by the pytest suite.

---

This project is a part of a hackathon run by https://katomaran.com