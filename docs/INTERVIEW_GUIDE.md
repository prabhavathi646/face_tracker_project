# Interview Guide — Intelligent Face Tracker

This document explains the project in simple language so you can walk an
interviewer through every part confidently.

## What Every Major Module Does

| Module | One-sentence explanation |
|---|---|
| `main.py` | The conductor: reads frames, calls detection/tracking/recognition in order, and fires ENTRY/EXIT events. |
| `src/config.py` | Reads `config.json` once, validates required sections, and gives everyone else typed access to settings. |
| `src/video_source.py` | Hides whether frames come from a file or an RTSP camera; retries the connection if an RTSP stream drops. |
| `src/detector.py` | Wraps YOLOv8: "give me a frame, I return face boxes + confidence." |
| `src/recognizer.py` | Wraps InsightFace: crops → 512-d embedding; `find_best_match()` compares embeddings by cosine similarity. |
| `src/tracker.py` | Remembers who is who between frames using boxes (IoU + centroid), giving each person a temporary `track_id`. |
| `src/registration.py` | When nobody matches, invents a new `FACE-xxxx` ID, saves the crop, and writes the embedding to SQLite. |
| `src/event_logger.py` | Sets up `events.log` and persists ENTRY/EXIT events (dated image + DB row). |
| `src/database.py` | All SQLite access: `faces`, `embeddings`, `events` tables with parameterized queries. |
| `src/visitor_counter.py` | Unique visitor count = number of distinct registered Face IDs (in-memory + DB-backed). |
| `src/utils.py` | Pure helpers: IoU, cosine similarity, date folders, box clamping. |

## Why YOLO Is Used

YOLO ("You Only Look Once") is a single-stage object detector: one forward
pass gives all boxes and confidences, typically 10–30 ms on CPU with the
nano model. For video we need that speed every frame (or every Nth frame
thanks to `frame_skip`). We use face-tuned YOLOv8 weights so the model is
trained on faces specifically — more precise than asking a COCO
person-detector to guess where the face is.

## Why InsightFace/ArcFace Is Used

Detection only says "a face is here." To count *unique* visitors we need
identity. ArcFace is a metric-learning model trained so that faces of the
same person map close together on a 512-d unit sphere and different people
map far apart. InsightFace ships it as a ready-to-run ONNX model
(`buffalo_l`) that runs on CPU via ONNX Runtime — accurate, fast enough,
and permitted by the project constraints (the `face_recognition` library
is explicitly forbidden).

## How Embeddings Work

An embedding is just a list of 512 numbers describing the face. Think of
it as coordinates: identical-looking faces land at nearly the same point.
We L2-normalize the vector, so the dot product of two embeddings equals
their **cosine similarity**: 1.0 = identical direction, 0 = unrelated.
That single number is what we threshold on.

## How Face Matching Works

`find_best_match(new_embedding, known_embeddings, threshold)` scans every
registered embedding, computes cosine similarity, and keeps the highest
score. If the best score ≥ `similarity_threshold` (default 0.45), we
return that `face_id` — the person is known. Otherwise we return `None`,
which means "new person → register." The scan is O(#faces) with tiny
512-float vectors, so it's microseconds even for thousands of faces.

## How Tracking Works

The custom tracker is ByteTrack-inspired:

1. Each frame, existing tracks try to claim detections by **IoU**
   (how much boxes overlap) — best overlap above `iou_match_threshold` wins.
2. Unmatched tracks fall back to **centroid distance** within
   `max_distance_px` (handles small/fast motion where boxes don't overlap).
3. Matched tracks update their box and reset a `disappeared` counter;
   unmatched ones increment it.
4. New leftover detections spawn new tracks with fresh `track_id`s.
5. A track whose `disappeared` exceeds `max_disappeared_frames` is retired.

Crucially, each track also stores its `face_id` once recognition confirms
it, so the identity survives across frames — even on frames where YOLO
didn't run because of `frame_skip`.

## How a New Face Gets Registered

1. A track has no `face_id` yet.
2. Its box is cropped from the frame and passed to InsightFace → embedding.
3. `find_best_match` returns `None` (nobody close enough).
4. `FaceRegistrar.register_new_face()` generates `FACE-<8 hex>`, saves the
   crop to `data/registered_faces/`, inserts a `faces` row and an
   `embeddings` row (BLOB of float32), and appends to the in-memory cache.
5. The track gets this `face_id` and logs an ENTRY event.

## How the System Knows a Face Exited

The tracker only removes a track after it has been unmatched for more than
`max_disappeared_frames` consecutive update calls (default 30 ≈ 1.2 s at
25 FPS). When the person walks out of view, YOLO stops detecting them, the
counter climbs, and on the threshold-crossing frame the track is retired
and returned in `exited_tracks`. `main.py` then logs exactly one EXIT
(guarded by the track's `entry_logged` flag). On shutdown, any tracks still
active are flushed as EXIT events so counts stay consistent.

## How Duplicate Registration Is Prevented

- Registration only happens when **no** stored embedding clears the
  similarity threshold for the new crop.
- The moment a face is registered, its embedding lands in the cache that
  the matching step scans next frame — the same person now matches above
  threshold, so no second ID is created.
- The cache is seeded from SQLite on startup, so even across runs a known
  visitor is never re-registered.
- Tests `test_duplicate_registration_prevented` and
  `test_below_threshold_returns_none` pin this behavior.

## How Unique Visitors Are Counted

Unique visitors = number of rows in `faces` = number of distinct Face IDs
ever registered. A person seen in 1,000 frames has one ID → counted once.
A person who leaves and returns is re-identified by embedding → same ID →
still counted once. `VisitorCounter` keeps a fast in-memory set for the
live display and `persisted_count()` queries the database for the
authoritative number printed in the summary and `events.log`.

## Database Schema

```
faces      (face_id PK, registered_at, reference_image_path,
            total_sightings, last_seen_at)
embeddings (id PK, face_id FK → faces, embedding BLOB, created_at)
events     (id PK, face_id FK → faces, track_id,
            event_type CHECK IN ('ENTRY','EXIT'), timestamp,
            image_path, similarity)
```

Design notes: identity is separated from vectors (multiple embeddings per
person are possible later); `events` is append-only with a CHECK
constraint enforcing valid types; every statement is parameterized
(`?` placeholders) — SQL injection-safe by construction.

## Logging Mechanism

`setup_logging()` creates one named logger (`face_tracker`) with two
handlers: a `FileHandler` writing the mandatory `events.log` and a
`StreamHandler` echoing to the console. Format:
`%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`. Because every
module uses `logging.getLogger("face_tracker")`, all messages interleave
chronologically in one file. Lifecycle events use INFO, verbose details
DEBUG, recoverable issues WARNING, failures ERROR with
`logger.exception` (full traceback).

## config.json

Everything tunable lives here: video source (file vs. RTSP + reconnect
settings), YOLO model path/confidence/`frame_skip`/device, InsightFace
model name/similarity threshold/providers, tracking thresholds, DB path,
entry/exit/reference image directories, log file/levels, display toggle,
and minimum face size. `Config.get("detection", "frame_skip")` provides
safe dotted access with defaults; `resolve_path()` anchors relative paths
to the project root — nothing is hard-coded in the modules.

## CPU/GPU Requirements

- **CPU-only (default):** YOLOv8n ~10–30 ms/frame; ArcFace ~25–60 ms but
  only when a track still needs identity; overall ~10–20 FPS with
  `frame_skip=2`. RAM < 500 MB.
- **GPU:** set `detection.device="cuda"` and add `CUDAExecutionProvider`
  to recognition providers → ~25–40 FPS.
- Disk: ~300 MB for buffalo_l, ~30–60 KB per event image.

## Important Functions/Classes

- `Config.get()` / `Config.resolve_path()` — safe config access.
- `FaceDetector.detect()` — frame → `[Detection]`.
- `FaceRecognizer.get_embedding()` — crop → 512-d vector (None if no face).
- `prepare_face_crop()` — pad + upscale a YOLO box so InsightFace can
  detect the face inside the crop.
- `find_best_match()` — pure, testable matching core.
- `FaceTracker.update()` — detections → `(active, exited)` tracks.
- `FaceRegistrar.register_new_face()` — the auto-registration transaction.
- `EventLogger.log_entry()/log_exit()` — image + DB + log in one call.
- `Database.register_face()/insert_event()/get_unique_visitor_count()`.
- `VisitorCounter.register_new_visitor()/persisted_count()`.

## Real Debugging Story (great interview material)

During development the first full run registered **0 visitors** even
though YOLO found faces. I isolated the failing stage with a small
diagnostic script (`tests/diagnose_pipeline.py`) and found:

1. YOLO detection: OK (2 boxes).
2. InsightFace on the **full frame**: OK (2 faces, dim 512).
3. InsightFace on **tight YOLO crops**: always `None`.

A parameter sweep (`tests/diagnose_crops.py`) showed the root cause:
InsightFace's detector needs surrounding context — a crop with margin
< 0.3 of the box size fails, while ≥ 0.3 succeeds (we ship 0.4). The fix
was `prepare_face_crop()`: pad the box by `crop_margin`, clamp to the
frame, and upscale small crops to `min_crop_size_px`. Both values are in
`config.json`, and a unit test covers the helper. Lesson: always verify
each pipeline stage in isolation before blaming the model.

## Possible Interview Questions and Answers

**Q: Why not just count detections?**
A: One person appears in hundreds of frames; counting detections measures
frames, not people. We need identity (embeddings) + persistence (tracking).

**Q: What happens if the similarity threshold is too low?**
A: Two different people can be merged into one ID (false link) →
undercounting. Too high and one person gets multiple IDs → overcounting.
0.45 with ArcFace is a common starting point; we'd tune on validation data.

**Q: Why frame skipping? Does it hurt accuracy?**
A: Detection dominates cost; the tracker bridges skipped frames using the
last detections, so identity is preserved. Only if someone moves very far
between two detection cycles could a match be lost — the centroid
fallback radius (`max_distance_px`) covers typical motion.

**Q: What if two people cross each other?**
A: The simple tracker may swap identities in that instant (documented
limitation). Production systems use DeepSORT/ByteTrack with appearance
features; our interfaces allow that swap without touching other modules.

**Q: How do you guarantee exactly one ENTRY?**
A: The `entry_logged` boolean on the Track is set on the first frame the
track has a Face ID and checked before every log call — double-firing is
structurally impossible, not just unlikely.

**Q: Where does the unique count come from?**
A: `SELECT COUNT(*) FROM faces` — every distinct person is exactly one
row, because registration only happens when no embedding matches.

**Q: Is this production-ready?**
A: It's a hackathon-grade, well-tested core. Gaps: simpler tracker, single
embedding per person, no face-quality gating, RTSP untested on hardware.
The schema and interfaces were designed so those are drop-in upgrades.

**Q: How did you use AI in building this?**
A: With an AI coding agent (Cline) following a plan→implement→test→fix
cycle: environment inspection first, module-by-module commits, unit tests
after each stage, and AI-drafted docs — with every claim verified by the
pytest suite before it went into the README.

## Limitations and Possible Improvements

- **Tracker robustness** → swap in DeepSORT/ByteTrack with appearance
  embeddings (`supervision` or `boxmot` libraries).
- **Single reference embedding** → store N embeddings per face, match by
  max/mean similarity; refresh on each confirmed sighting.
- **No quality gating** → reject blurry/extreme-pose crops before
  registration using blur variance and pose estimation from InsightFace.
- **RTSP not hardware-validated** → test with an ONVIF camera; add
  hardware-accelerated decode if CPU becomes the bottleneck.
- **No privacy controls** → hash embeddings at rest, retention policies,
  GDPR-style delete-by-face-id tooling.
- **Single process** → shard by camera; move SQLite → Postgres for
  multi-writer deployments.