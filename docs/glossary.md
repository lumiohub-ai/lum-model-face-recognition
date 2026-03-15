# Evaluation Metrics Glossary

## Four-Layer Hierarchy

```
Frame-level       Probe/Track-level     Person-level          Rank-based
(FIR / FNMR)      (TPIR / FPIR)         (PLA / FNR / SIR)     (CMC / mAP)
per-frame         per-track             per-person episode     per-probe ranked list
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

## Rank-Based Metrics

> Standard closed-set identification metrics. Computed in `_compute_cmc_map()`. Stored as `cmc_map`.

Each genuine probe provides a ranked candidate list of up to 10 unique persons (by max cosine similarity across their gallery embeddings), produced by `recognizer.get_top_k_candidates()`.

| Term | Full Name | Formula |
|------|-----------|---------|
| **CMC[k]** | Cumulative Match Characteristic | `fraction of genuine probes where correct person is in top-k` |
| **mAP** | mean Average Precision | `mean(1/rank)` across genuine probes; 0 if not in top-10 |

> In ChokePoint with one track per person and perfect identification, Rank-1 ≈ 1.0 and mAP ≈ 1.0.

---

## Person-Level TAR/FAR Curve

> Computed in `_compute_person_curves()`. Stored as `person_curves`. Required by paper abstract.

One **episode** per GT person: score = max similarity across their spatially-matched tracks.
One **impostor episode** per qualifying impostor track (same criterion as probe-level).

| Term | Formula |
|------|---------|
| **TAR(T)** | `persons accepted correctly at T / n_genuine_persons` |
| **FAR(T)** | `impostor episodes accepted at T / n_impostor_episodes` |

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
| `person_curves` | object | TAR/FAR arrays at person level |
| `cmc_map` | object | CMC curve + mAP |
| `det_curve` | object | FNMR/FMR arrays (derived from probe_curves) |
| `pla_fir_sweep` | object | PLA(T) and FIR(T) threshold sweep arrays |
| `mot_metrics` | object | MOTA + IDF1 (track-level approximation) |
| `pl_frr` | float | PL-FRR = missed / N (alias for `fnr`) |
| `pl_far` | float | PL-FAR = misidentified / N (alias for `sir`) |
| `swap_rate` | float | Swap Rate = misidentified / N (alias for `sir`) |
| `fir` | float | Frame Identification Rate |
| `fnmr` | float | False Non-Match Rate |
| `n_frames` | int | Total XML-annotated frames |
| `n_frames_correct` | int | Frames covered by a correct track |

---

## Reviewer-Added Metrics

### PL-FRR / PL-FAR / Swap Rate

Person-level error decomposition — aliases of existing fields:

| Term | Full Name | Formula | `evaluation` key |
|------|-----------|---------|-----------------|
| **PL-FRR** | Person-Level False Rejection Rate | `missed / N` | `pl_frr` (= `fnr`) |
| **PL-FAR** | Person-Level False Acceptance Rate | `misidentified / N` | `pl_far` (= `sir`) |
| **Swap Rate** | Identity Swap Rate | `misidentified / N` | `swap_rate` (= `sir`) |

### DET Curve

Detection Error Tradeoff curve — standard biometric plot on log–log axes.

- `FNMR(T) = 1 − probe_curves.tpir[T]`
- `FMR(T) = probe_curves.fpir[T]`

Stored as `det_curve = {thresholds, fnmr, fmr}`. Data is a strict derivation of `probe_curves` — no new computation.

### PLA(T) / FIR(T) Threshold Sweep

Stored as `pla_fir_sweep = {thresholds, pla, fir, mota, idf1, n_frames}`.

- `pla[T]` = `person_curves.tar[T]` — reused, no recomputation.
- `fir[T]` = fraction of XML-annotated frames correctly identified *if* threshold T applied post-hoc to similarity scores (track must have `true_gt_id==pid`, `name==pid`, `similarity >= T`).
- `mota[T]` = `1 − (FP + FN(T) + IDSW(T)) / GT_total` — person-level MOTA approximation at T. FP = spatial FP (fixed); FN(T) = GT persons with no accepted track at T; IDSW(T) = GT persons with accepted tracks but all wrong identity at T.
- `idf1[T]` = `2·IDTP(T) / (2·IDTP(T) + IDFP(T) + IDFN(T))` — track-level IDF1 approximation at T. Uses best correctly-accepted track per GT person; IDFP(T) from impostor track durations accepted at T.

At `T=0.0`, `fir[0] == existing fir` (threshold-free). As T rises, FIR(T) degrades faster than PLA(T) — this is the paper's core figure.

### MOTA / IDF1 (track-level approximation)

**Not standard frame-level MOT** — this pipeline lacks frame-level detection logs required for standard MOTA/IDF1 (TrackEval). These are defensible approximations stored as `mot_metrics` with `"approximation": "track-level"`.

| Metric | Formula |
|--------|---------|
| **MOTA** | `1 − (FP + FN + IDSW) / GT_total` — FP/FN/IDSW are person-level counts |
| **IDF1** | `2·IDTP / (2·IDTP + IDFP + IDFN)` — duration-weighted, track-level |

MOTA components: `mot_fp` = `n_spatial_false_positives`, `mot_fn` = `len(missed)`, `mot_idsw` = `len(misidentified)`.
IDF1 components: `idtp` = duration of correctly-labeled best tracks, `idfp` = impostor track duration, `idfn` = GT frames not covered by best tracks.

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
