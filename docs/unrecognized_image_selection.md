# Image Selection & Filtering for Unrecognized Cases

This document describes the **implemented** approach for reducing dashboard noise
in unrecognized cases, and the finding that reshaped the original size+sharpness
plan. The system gates which unrecognized captures reach the dashboard based on
**face frontality**, measured from facial landmarks.

## 1. Implemented Architecture Workflow

```
            Track ends, person was never recognized
                          │
                          ▼
          ┌───────────────────────────────────┐
          │ Get all saved frames of this track │   (up to 30; frames where a
          │   {frame# → face, bbox, image,     │    face was seen, now also
          │    det_score, landmarks}           │    storing det_score+landmarks)
          └───────────────────┬────────────────┘
                              │
                              ▼
          ┌───────────────────────────────────────────┐
          │  Measure two independent signals across    │
          │  all frames (separate sibling methods —    │
          │  _get_best_person_image stays UNTOUCHED):  │
          │                                            │
          │   quality   = best size+sharpness  (log)   │
          │   frontality= best f(landmarks→yaw) ★GATE   │
          └───────────────────┬────────────────────────┘
                              │
                              ▼
              attach (quality, frontality) to the event
                              │
                              ▼
          ┌───────────────────────────────────┐
          │   Gate on frontality (config:      │   unrecognized_frontality_min
          │   unrecognized_frontality_min=0.6) │   (config.yaml, default 0.6)
          └───────┬───────────────────┬────────┘
                  │                   │
        frontality ≥ thr        frontality < thr
                  │                   │
                  ▼                   ▼
        ┌──────────────────┐   ┌──────────────────────────┐
        │ upload + persist │   │ UNRECOGNIZED_DROPPED log  │
        │ → DASHBOARD card │   │ no upload, no DB, no card │
        │ + UNRECOGNIZED_  │   │ (debug data lives in logs)│
        │   QUALITY log    │   └──────────────────────────┘
        └──────────────────┘
```

## 2. The Finding That Changed the Plan

The original plan assumed *low image quality* (small/blurry) explained the
unrecognized noise. **Live data disproved this.** Best-frame crop size+sharpness
turned out to be **uncorrelated with whether a usable face is present**:

| track | quality (size+sharp) | frontality | actual content |
|---|---|---|---|
| 25 | **58.3** | 0.000 | back of head / chair — no face |
| 22 | 34.9 | 0.000 | profile at monitor |
| 8  | 17.8 | 0.000 | bent over keyboard |
| **28** | 37.4 | **1.000** | genuine frontal face → real review case |

The highest *quality* scores were backs of heads and chairs (cameras pointed at
people working at desks). Ranking by quality would have promoted the **worst**
cards. **Frontality cleanly separates** non-faces (≈0.0) from real faces (≈1.0)
— so it, not blended quality, drives the gate.

## 3. Architectural Comparison: Original Proposal vs. Implemented

| Stage / Attribute | Original Proposal | Implemented |
|---|---|---|
| **`_get_best_person_image`** | Rewritten (soft-score, tuple return, fallback removed) | **Untouched** — protects the recognized/fire-once attendance path |
| **New logic location** | Inside the shared method | **Separate sibling methods** (`_best_person_image_quality`, `_best_face_signals`) |
| **Primary signal** | Blended `quality = w·size + w·sharp + w·frontality + w·det` | **Frontality alone** gates; quality is logged for context only |
| **Scoring inputs** | size + sharpness + frontality + det_score (one score) | size+sharpness (observability) and frontality (gate) kept **separate** |
| **Routing** | Review vs Debug buckets in the dashboard (read-time threshold) | **Hard gate at write-time**; dropped cases logged, not persisted |
| **"Debug tier"** | A second dashboard bucket | **Logs** (`UNRECOGNIZED_DROPPED`) — no dashboard/DB change required |
| **Threshold** | Tunable config | `unrecognized_frontality_min` in `config.yaml` (default 0.6), edit + restart |
| **Frontality** | Step 3 optimization (last) | **Core signal, implemented early** |

## 4. Frontality Definition

A yaw (left-right turn) proxy from 3 of the 5 InsightFace landmarks (eyes +
nose), x-coordinates only:

```
eye_dx     = |right_eye_x − left_eye_x|        # inter-eye distance
eye_mid_x  = (left_eye_x + right_eye_x) / 2
yaw        = |nose_x − eye_mid_x| / eye_dx      # 0 = frontal, grows with turn
frontality = max(0, 1 − 2·yaw)                  # [0,1], 1 = frontal
```

- Scale-invariant (normalized by inter-eye distance); no model needed.
- `0.0` = profile or **no landmarks** (no usable face); `1.0` = straight-on.
- **Known limitation — pitch blind spot:** only measures yaw, so a
  frontal-but-looking-down face (e.g. bent over a keyboard) still scores
  moderately and passes. Deferred refinement: add a pitch term from
  eye-line→mouth-line geometry.

## 5. Behavioral Shift

- **Before:** one flat pile of cards where backs of heads, profiles, and chairs
  (mostly *recognized* employees, captured by desk-facing cameras) polluted
  genuine unrecognized entries.
- **After:** only captures with a real frontal face (`frontality ≥ 0.6`) become
  dashboard cards. Everything else is recorded in logs with its scores and
  image-correlation, kept off the review list. Nothing is silently lost —
  dropped cases remain auditable in logs for tuning.

## 6. Deployment Roadmap & Status

| Step | Description | Status |
|---|---|---|
| **0 — Quality logging** | `_best_person_image_quality` (size+sharpness); log-only, no behavior change | ✅ Done |
| **0b — Frontality logging** | Store landmarks+det_score in crop history; `_frontality_from_landmarks` / `_best_face_signals`; log alongside quality + `image_url` | ✅ Done |
| **Gate — Frontality filter** | Drop cards below `unrecognized_frontality_min` (config 0.6) at write-time; dropped cases logged | ✅ Done |
| **1 — Persistence** | Persist `frontality`/`quality` to the `unrecognized_faces` row | ⏳ Deferred |
| **2 — Dashboard bucketing** | True Review vs Debug split in the dashboard (needs backend + DB column) | ⏳ Deferred |
| **3 — Pitch term** | Penalize looking-up/down faces; refine frontality beyond yaw | ⏳ Future |

> Observability discipline: Steps 0 / 0b were behavior-preserving (log-only). The
> size+sharpness finding came from that telemetry *before* any filtering, which is
> exactly why the gate uses frontality rather than the originally-assumed quality
> metric.

## 7. Relevant Code Map

| Concern | Location |
|---|---|
| Mark track unrecognized (`not identity_locked`) | `src/pipeline/camera_engine.py` (removed-tracks loop) |
| Recognized fire-once edge (calls `_get_best_person_image`) | `src/pipeline/camera_engine.py` |
| `_get_best_person_image` (unchanged; card/proof image) | `src/pipeline/camera_engine.py` |
| `_best_person_image_quality` (size+sharpness, soft) | `src/pipeline/camera_engine.py` |
| `_frontality_from_landmarks` / `_best_face_signals` | `src/pipeline/camera_engine.py` |
| Crop history stores face/bbox/frame/landmarks/det_score | `src/pipeline/camera_engine.py` |
| Forward quality+frontality on the event | `src/infrastructure/async_logger.py` |
| Frontality gate + `UNRECOGNIZED_QUALITY` / `UNRECOGNIZED_DROPPED` logs | `src/infrastructure/entry_logger.py` |
| Threshold plumbed from config via `args` | `src/pipeline/engine.py` (`_init_entry_logger`) |
| `unrecognized_frontality_min` config key | `configs/config.yaml` |
