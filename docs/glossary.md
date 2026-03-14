# Evaluation Metrics Glossary

## Three-Layer Hierarchy

```
Frame-level       Probe/Track-level     Person-level
(FIR / FNMR)      (TPIR / FPIR)         (PLA / FNR / SIR)
per-frame         per-track             per-person appearance
```

---

## Person-Level Metrics

> Our paper contribution. Computed in `_evaluate()`.

| Term | Full Name | Formula |
|------|-----------|---------|
| **PLA** | Person-Level Accuracy | `correctly_identified / N` |
| **FNR** | False Negative Rate | `missed / N` |
| **SIR** | Substitution / Identity-swap Rate | `misidentified / N` |

**Invariant:** `PLA + FNR + SIR = 1.0`

**Per-person classification:**

| Outcome | Condition |
|---------|-----------|
| `correctly_identified` | ≥1 track where `true_gt_id == pid` AND `name == pid` |
| `missed` | No track spatially matched to this person |
| `misidentified` | Track exists but none have `name == pid` |

Matching is **spatial** via XML eye-coordinate GT (`true_gt_id`). No temporal ordering.
Falls back to `null` values with a warning if XML GT is not loaded.

---

## Probe-Level Metrics

> NIST FRVT / ISO 19795-1. Computed in `_compute_biometric_curves()`. Stored as `probe_curves`.

One **track** = one **probe transaction**.

| Term | Full Name | Formula |
|------|-----------|---------|
| **TPIR(T)** | True Positive Identification Rate | `correct probes accepted at T / all genuine probes` |
| **FPIR(T)** | False Positive Identification Rate | `impostor probes accepted at T / all impostor probes` |
| **FNIR(T)** | False Negative Identification Rate | `1 − TPIR(T)` *(not stored, derivable)* |

**Probe types:**

| Type | Condition |
|------|-----------|
| Genuine | `true_gt_id ∈ enrolled persons` |
| Genuine-Correct | genuine AND `name == true_gt_id` → counts in TPIR numerator |
| Genuine-Wrong | genuine AND `name != true_gt_id` → penalises TPIR, does NOT affect FPIR |
| Impostor | `true_gt_id is None` AND (`xml_checked ≥ 3` OR `n_embeddings == 0`) |
| FN injection | enrolled person with zero tracks → `(score=0.0, correct=False)` inserted |

---

## Frame-Level Metrics

> ISO 19795-1. Computed in `_compute_frame_metrics()`. Merged into `evaluation`.

Per XML-annotated frame `f` with person `p`: **correct** if a track covers frame `f` with `true_gt_id == p AND name == p`.

| Term | Full Name | Formula |
|------|-----------|---------|
| **FIR** | Frame Identification Rate | `n_frames_correct / n_frames` |
| **FNMR** | False Non-Match Rate | `1 − FIR` |

> FIR < 1.0 even when PLA = 1.0 — DeepOCSORT `min_hits=3` means the first ~3 frames of each person's GT window are uncovered before the track activates.

---

## Null Probe

A track that was detected and tracked but had **no processable face embeddings** (e.g. face width < `minimum_face_size = 50 px`). Emitted in `eval` mode as:

```
name=UNKNOWN  status=NO_EMBEDDINGS  similarity=0.0  true_gt_id=null
```

Classified as **IMPOSTOR** in probe-level evaluation. Score `0.0` → correctly rejected at any `T > 0`.

---

## Spatial Rejection

A result entry excluded from PLA evaluation because XML GT confirms the track's claimed identity is wrong, or the person is a confirmed bystander. Reported as `spatial_false_positives` in `results.json`.

---

## results.json Field Reference

### Per result entry

| Field | Meaning |
|-------|---------|
| `track_id` | DeepOCSORT tracker ID |
| `name` | System's predicted identity (`"0003"`, `"UNKNOWN"`) |
| `recognized` | `"recognized"` or `"unrecognized"` |
| `status` | `"RECOGNIZED"` or `"NO_EMBEDDINGS"` |
| `similarity` | Cosine similarity to nearest gallery embedding [0, 1] |
| `first_frame_num` | First frame where a valid embedding was stored |
| `last_frame_num` | Last frame where a valid embedding was stored |
| `true_gt_id` | XML-matched GT person ID (`null` = unannotated / impostor) |

### evaluation dict

| Field | Type | Meaning |
|-------|------|---------|
| `pla` | float | Person-Level Accuracy |
| `fnr` | float | False Negative Rate |
| `sir` | float | Substitution/Identity-swap Rate |
| `correctly_identified` | list | Person IDs correctly identified |
| `missed` | list | Person IDs with no matching track |
| `misidentified` | list | Person IDs tracked but wrongly labelled |
| `total_gt` | int | Total enrolled persons in sequence |
| `probe_curves` | object | TPIR/FPIR arrays + counts |
| `fir` | float | Frame Identification Rate |
| `fnmr` | float | False Non-Match Rate |
| `n_frames` | int | Total XML-annotated frames |
| `n_frames_correct` | int | Frames covered by a correct track |

---

## ChokePoint Dataset

| Item | Value |
|------|-------|
| Enrolled persons per sequence | 25 (P1 portal) |
| Total persons in P1E_S1 | 27 (25 enrolled + 2 unannotated impostors) |
| Impostors in P1E_S1_C1 | track 1 (bystander, frame 343) · track 12 (face 33 px wide, below threshold) |
| JSON annotation format | `{person_id: {start: frame, end: frame}}` |
| XML GT format | `{frame_num: {person_id: (eye_mid_x, eye_mid_y)}}` |
| XML GT path | `volumes/src/annotation/chokepoint/groundtruth/{camera_name}.xml` |
