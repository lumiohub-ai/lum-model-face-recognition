# Unrecognized Faces — Quality Filtering Plan

> Reducing dashboard noise from unrecognized cases by scoring face quality and
> splitting cases into an admin-review bucket and a debug bucket.

## Problem

The dashboard shows too many "unrecognized" cases, but most are actually
**registered employees** who simply weren't recognized due to bad face angle,
blur, or distance — not genuine strangers.

## Key Insight

An unrecognized card is only useful to an admin if the face is clear enough to
act on. A bad-quality face is noise to a human for the **same reason** it was
noise to the recognizer. So **face quality cleanly separates "actionable" from
"ignore."**

Bonus: a clear-but-unrecognized face is doubly valuable — it's either

- a genuine stranger (the real thing we want to catch), **or**
- an employee whose enrollment is weak — and we now have a clean crop to
  **re-enroll** them, fixing the root cause instead of generating more cards.

Bad-quality faces give neither.

## Direction

Don't drop anything. Compute a `face_quality` score per case and split into two
buckets:

- **Review** — clean faces, for admin action
- **Debug / all** — everything else, for engineers to audit and tune

Store the score; derive the bucket at **read-time** via a threshold, so it's
tunable without reprocessing history.

### Why score-and-filter beats hard-drop

- Tune the threshold from real data instead of guessing.
- False-negative safety net — a borderline genuine stranger is still recorded.
- One schema change; pipeline logic stays simple.

## Note on the current `_get_best_person_image` fallback

The existing "return most recent frame" fallback
([camera_engine.py:724](../src/pipeline/camera_engine.py#L724)) is the weakest
part:

- **Last ≠ best** — the most-recent frame is usually the worst (person
  leaving/turning).
- **No score** — it returns an image with no quality info, so it can't be
  bucketed.

In the new design it should be **replaced by soft-gating**: score every frame,
always return the best-scored frame **plus its score**; the score routes the
bucket. The old `max(crops.keys())` branch then disappears.

The only true fallback that remains is **empty crops** (face never detected):
`crop_history` is only populated on frames where a face was found
([camera_engine.py:451](../src/pipeline/camera_engine.py#L451)), so those tracks
produce **no card at all** today and stay invisible — unless we deliberately add
body-crop capture (probably wanted only for the debug bucket, not review).

## ⚠ Implementation safety — do NOT break the fire-once pattern

`_get_best_person_image` is called from **two** places:

| Call site | Line | Result used as |
|---|---|---|
| Recognized edge (fire-once attendance) | [camera_engine.py:393](../src/pipeline/camera_engine.py#L393) | `proof_image` → attendance proof |
| Unrecognized removed-track | [camera_engine.py:479](../src/pipeline/camera_engine.py#L479) | `person_image` → unrecognized card |

The fire-once / edge-detection logic (`prev_state and not prev_state.identity_locked`)
is driven **only by identity-lock state**, so changing *image selection logic*
cannot change whether/how often it fires. **But** changing the method to return a
tuple `(image, quality)` **in place** would corrupt the attendance `proof_image`
at the recognized call site.

**Rule:** keep `_get_best_person_image` untouched. Add a **separate sibling
method** (e.g. `_best_person_image_quality(track_id) -> float`) used only by the
unrecognized removed-track block. The recognized / fire-once path stays
byte-for-byte unchanged → zero risk.

Adding new fields (`det_score`, `landmarks`) to the crop-history entry at
[camera_engine.py:451](../src/pipeline/camera_engine.py#L451) is **additive** —
`_read_crop_image` and the scorer only read `face`/`bbox`/`frame`, so extra keys
disturb neither path.

| Change | Breaks fire-once? |
|---|---|
| New selection logic (different frame/score) | No |
| Add `det_score`/`landmarks` to crop history | No |
| Change `_get_best_person_image` to return a tuple **in place** | **Yes** — corrupts attendance proof_image |
| Add a **separate** quality method, leave the original | No — required approach |

## Action Plan (minimal, observable steps)

Each step is independently shippable and observable. **Steps 0–1 are
behavior-preserving** — nothing is bucketed or dropped until Step 0's logged
distribution shows where the threshold should sit, so each later change's effect
is measurable in isolation.

| Step | Change | Behavior change? | Files |
|---|---|---|---|
| **0** | Add a **separate** `_best_person_image_quality` method (size + sharpness, reusing the score `_get_best_person_image` already computes); call it in the removed-track block and **just log it** | None — measure only | `src/pipeline/camera_engine.py` |
| **1** | Persist the score: nullable `face_quality` DB column + event field | None | `entry_logger`, `detection_tasks`, `detection_repository`, `publisher` + migration |
| **2** | Dashboard splits review vs debug via read-time threshold | UI/query only | dashboard (Backend) |
| **3** | Add **frontality** (face landmarks → yaw) to the score — targets the "different angles" root cause | Pipeline | crop history + quality fn |
| **4** | Optional: de-dup repeated unknowns; "near-miss → possible match: `<name>`" routing | Later | — |

### Quality score composition

- **Step 0/1 minimum:** size + sharpness (reuse the score
  `_get_best_person_image` already computes but discards).
- **Step 3:** add **frontality** from the detector's 5-point landmarks
  (eye-symmetry / nose offset → yaw). This is the part that actually fixes the
  "employees at different angles" complaint; without it, sharp side-profiles
  still land in the review list. Also available: `det_score` (detector
  confidence).

## Immediate Next Action — Step 0

Add `_best_person_image_quality(track_id) -> float` (a **new sibling** of
`_get_best_person_image`, leaving the original untouched so the recognized /
fire-once path carries no risk), computing size + sharpness. Call it in the
removed-track block and **log it** next to the existing `UNRECOGNIZED` line.

No schema, no payload, no filtering — pure observability.

## Relevant Code Map

| Concern | Location |
|---|---|
| Mark track unrecognized (`not identity_locked`) | [camera_engine.py:478](../src/pipeline/camera_engine.py#L478) |
| Recognized fire-once edge (calls `_get_best_person_image`) | [camera_engine.py:393](../src/pipeline/camera_engine.py#L393) |
| Best-crop selection + scoring | [camera_engine.py:690](../src/pipeline/camera_engine.py#L690) |
| Crop history populated (face/bbox/frame) | [camera_engine.py:451](../src/pipeline/camera_engine.py#L451) |
| Event handed off (non-blocking) | [camera_worker.py:183](../src/pipeline/camera_worker.py#L183) |
| Route recognized vs unrecognized + `"unrecognized"` gate | [async_logger.py:170](../src/infrastructure/async_logger.py#L170) |
| Upload crop + queue Celery task | [entry_logger.py:202](../src/infrastructure/entry_logger.py#L202) |
| DB write (`save_unrecognized_face`) | [detection_tasks.py:101](../src/workers/detection_tasks.py#L101) |
| Publish live event (`publish_unrecognized_face_saved`) | [detection_tasks.py:111](../src/workers/detection_tasks.py#L111) |
