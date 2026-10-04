# AI Planning Document — Intelligent Face Tracker

## 1. Problem Understanding

We need an AI system that watches a camera stream and answers one business
question accurately: **how many distinct people visited?** The naive
approach (count motion/detection frames) fails because one person appears
in hundreds of frames and may leave and re-enter. The system must
therefore solve four sub-problems in sequence: detect faces, keep identity
stable across frames (tracking), recognize who is who (embeddings), and
turn all of that into exactly-one ENTRY/EXIT events plus a durable unique
count.

## 2. Requirements

- **Mandatory stack:** Python, YOLO detection, InsightFace/ArcFace
  recognition (never `face_recognition`), OpenCV/ByteTrack-style tracking,
  SQLite, `config.json`, `events.log`, local image storage.
- Local video file input first; RTSP switchable purely via config.
- Configurable: frame skip, similarity threshold, model paths, DB/log/image
  paths, tracking and processing parameters.
- Exactly one ENTRY and one EXIT per visit; duplicate registration must be
  impossible for the same person; unique count retrievable from DB/logs.
- Graceful failure on missing video, bad RTSP URL, model/DB/image errors,
  invalid frames, and Ctrl-C.
- Modular, commented, testable, interview-explainable code.

## 3. Design Decisions

| Decision | Rationale |
|---|---|
| YOLOv8-nano via Ultralytics | Fast, well-maintained, single-line inference, configurable conf/imgsz, CPU-friendly |
| InsightFace `buffalo_l` (ArcFace) | State-of-the-art open embedding, 512-d vectors, ONNX Runtime on CPU, explicitly allowed |
| Custom IOU + centroid tracker instead of full DeepSORT | Zero extra heavy dependencies, trivially testable, sufficient for a single-camera visitor scenario; interface is swap-friendly |
| Detection only every Nth frame (`frame_skip`) | Detection is the dominant cost; tracker bridges skipped frames with the last detection set |
| Embedding only while a track has no Face ID | Avoids 25–60 ms embedding cost on every frame in steady state |
| In-memory embedding cache + SQLite persistence | DB is authoritative across runs; cache avoids per-frame SQLite reads |
| Greedy IoU matching with centroid fallback | Simple, deterministic, explainable; good enough when faces are small relative to motion |
| Events written once via flags on Track objects | Guarantees "exactly one ENTRY/EXIT" without DB-side dedup queries |
| Fail-fast constructor errors (custom exceptions) | Model/video/DB failures surface at startup with clear log messages instead of mid-run crashes |

## 4. Architecture

See the Mermaid diagram in `README.md`. In words:

`VideoSource` yields frames → `FaceDetector` (YOLO, every Nth frame) yields
boxes → `FaceTracker.update()` associates boxes into `Track` objects → for
any track without a Face ID, `main.py` crops the face, `FaceRecognizer`
extracts the ArcFace embedding, and either re-identifies (`recognizer.match`)
or `FaceRegistrar` creates a new ID → first association logs ENTRY, track
disappearance logs EXIT → both persist through `EventLogger` (image +
SQLite row + `events.log` line) → `VisitorCounter` reads the unique count
from the `faces` table.

## 5. Module Responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Orchestration, main loop, frame-skip policy, entry/exit triggers, shutdown flush |
| `src/config.py` | Load/validate `config.json`, dotted access, path resolution |
| `src/video_source.py` | File/RTSP abstraction, reconnect logic, FPS/size queries |
| `src/detector.py` | YOLO model load + inference → `Detection` list |
| `src/recognizer.py` | ArcFace embedding extraction + pure `find_best_match()` matching |
| `src/tracker.py` | IOU/centroid association, track lifecycle, exit detection |
| `src/registration.py` | Face ID generation, reference image save, embedding cache |
| `src/event_logger.py` | Logging setup; ENTRY/EXIT image + DB persistence |
| `src/database.py` | SQLite schema, parameterized CRUD, counts |
| `src/visitor_counter.py` | In-memory + persisted unique visitor count |
| `src/utils.py` | IoU, cosine similarity, path/date helpers, box clamping |

## 6. Data Flow

1. Frame (BGR `numpy` array) enters the loop.
2. Every `frame_skip` frames: YOLO → `[Detection(box, confidence)]`.
3. `tracker.update(detections)` → `(active_tracks, exited_tracks)`.
4. For each active track: clamp box → crop → (if unidentified) embed →
   match → `face_id` assigned or newly registered.
5. First `face_id` assignment → ENTRY event (image + DB + log).
6. Exited tracks with a `face_id` and a logged entry → EXIT event.
7. Session end → flush remaining active tracks as EXIT, print counts.

## 7. AI / Model Choices

- **Detection — YOLOv8n-face:** single-stage detector, ~3M parameters,
  runs real-time on CPU; face-tuned weights avoid the person-vs-face
  ambiguity of COCO models. Confidence threshold and input size configurable.
- **Recognition — ArcFace (InsightFace buffalo_l):** a metric-learning
  model trained so that same-identity embeddings cluster; produces a
  normalized 512-d vector. Runs via ONNX Runtime (`CPUExecutionProvider`
  by default; swap to CUDAExecutionProvider on GPU machines).
- **Why not `face_recognition` (dlib)?** Project constraint, plus
  InsightFace has better accuracy and an actively maintained ONNX stack.

## 8. Tracking Approach

ByteTrack-inspired but simplified: two-stage greedy association —
(a) maximum-IoU matching against `iou_match_threshold`, (b) centroid-distance
fallback within `max_distance_px` for non-overlapping/small motions.
Unmatched tracks increment `disappeared`; matched ones reset to 0.
Tracks exceeding `max_disappeared_frames` are retired → treated as exits.
The `Track` dataclass carries `track_id` (temporary), `face_id`
(persistent), `entry_logged`, and `best_similarity`.

## 9. Recognition Approach

Cosine similarity between L2-normalized 512-d vectors. The best score must
clear `recognition.similarity_threshold` or the face is considered new.
Matching runs only when a track lacks a Face ID (first appearance or
post-occlusion), never on every frame. The threshold is the single knob
balancing false links (merging two different people) vs. duplicate IDs
(splitting one person) — the failure modes are visible in tests.

## 10. Entry/Exit Detection Approach

- ENTRY: a one-shot boolean on the track (`entry_logged`) set when the
  Face ID is first known — structurally impossible to double-fire.
- EXIT: track retirement after `max_disappeared_frames` consecutive
  unmatched update calls, plus a shutdown sweep for still-active tracks.
- Both write: dated image (entry/exit folder) + `events` row + log line.

## 11. Database Design

Three normalized tables (see `src/database.py`): `faces` (identity +
metadata), `embeddings` (1..N vectors per face — schema ready for
multi-embedding matching), `events` (append-only ENTRY/EXIT log with
CHECK constraint on `event_type`). Foreign keys enabled, all statements
parameterized, one connection with cursor-per-operation + commit/rollback.

## 12. Logging Design

Single named logger `face_tracker`, two handlers (FileHandler → mandatory
`events.log`, StreamHandler → console), shared format with timestamps and
levels. Module loggers reuse the same name so all components interleave
chronologically in one file. Every stage logs its lifecycle and catches
its own exceptions with `logger.exception`.

## 13. Performance Considerations

- Detection every `frame_skip` frames (config).
- Embedding only for unidentified tracks.
- DB writes only at registration, sighting confirmation, and events —
  never per frame.
- O(#faces) cosine scans are negligible (hundreds of faces max).
- Tracker state is a flat dict — O(detections × tracks) per frame.

## 14. CPU/GPU Considerations

- **CPU-only demo (default):** YOLOv8n ~10–30 ms + ArcFace ~25–60 ms on
  first sight of each person; typical 10–20 FPS with `frame_skip=2`.
- **GPU:** set `detection.device` to `cuda`/`0` and add
  `CUDAExecutionProvider` to `recognition.providers` → ~25–40 FPS.
- Memory < 500 MB; disk: buffalo_l ~300 MB, images grow ~30–60 KB per event.

## 15. Testing Strategy

- **Unit (pytest, no models required):** config load/validate/errors;
  DB registration/sighting/events/counts + injection-safety; matching
  logic (exact, below-threshold, empty, best-of-multiple, duplicate
  prevention); visitor counting (unique-only, DB-seeded); ENTRY/EXIT
  creation incl. image files on disk; tracker lifecycle (id stability,
  distinct ids, exit-after-timeout).
- **Integration (manual/CI with models):** run `main.py` against
  `data/sample_video.mp4` and verify counts vs. known people; check
  `events.log`, image folders, and DB rows.
- Design for testability: pure functions (`find_best_match`), injectable
  config paths, side effects (disk/DB) confined to small classes.

## 16. Assumptions

- Camera angle shows faces clearly; no heavy occlusion.
- One stream per process; scale-out = multiple processes/DBs.
- Video FPS ~25; `max_disappeared_frames=30` ≈ 1.2 s grace period.
- First-run model downloads are acceptable (internet available once).

## 17. Known Limitations

- Simple tracker can swap IDs on close crossings.
- Single reference embedding per person (schema supports more).
- No face-quality/blur gating before registration.
- RTSP path implemented but not validated against physical hardware.
- Synthetic sample video may not exercise real-world occlusion/pose.

## 18. Future Improvements

- Real ByteTrack/DeepSORT or `supervision` tracker swap.
- Multi-embedding gallery + rolling centroid for robust re-ID.
- Embedding refresh on every confirmed sighting.
- Face quality (blur, pose, size) gating before registration.
- FastAPI dashboard with live counts and event feed.
- Postgres/MySQL behind the existing `Database` interface.
- Docker image + compose for one-command demos.
- Occupancy-style metrics: dwell time, flow direction (in vs. out zones).