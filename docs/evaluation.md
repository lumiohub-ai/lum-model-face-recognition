# R&D Evaluation Pipeline

## Overview

The R&D pipeline evaluates face recognition performance on the **ChokePoint** dataset across four complementary metric layers. All evaluation is driven by `src/face_recognition/rnd/rnd_runner.py` and configured via `configs/rnd_config.yaml`.

See [`docs/glossary.md`](glossary.md) for full term and field definitions.

---

## Running the Pipeline

```bash
# Start container and tail logs
./compose.sh start -l
```

Config is at `configs/rnd_config.yaml`. Key flags:

| Flag | Effect |
|------|--------|
| `eval: true` | Enables evaluation; bypasses production recognition filters |
| `debug: true` | Enables summary-level DEBUG logs (GT sequence, spatial rejections) |

Full video list (72 sequences) is auto-generated — do not hand-edit:

```bash
python scripts/build_experiment_config.py
```

---

## Ground Truth

Two annotation sources are required per sequence:

| Source | Format | Purpose |
|--------|--------|---------|
| JSON annotation | `{person_id: {start: frame, end: frame}}` | Enrolled persons + appearance windows |
| XML groundtruth | `{frame_num: {person_id: (eye_mid_x, eye_mid_y)}}` | Per-frame spatial eye coordinates |

**Paths:**
- JSON: `volumes/src/annotation/chokepoint/{portal}/{session}/{camera}/{camera}.json`
- XML: `volumes/src/annotation/chokepoint/groundtruth/{camera_name}.xml`

Spatial matching (`match_track_to_gt` in `gt_matcher.py`) checks whether the GT eye midpoint falls inside a track's bounding box across frames. A track is assigned `true_gt_id` if ≥ 30% of checked frames yield a hit.

---

## Four-Layer Evaluation

### 1. Frame-Level

**FIR** (Frame Identification Rate) and **FNMR** (False Non-Match Rate).

Per XML-annotated frame, did the active track correctly identify the person?

- `FIR = n_frames_correct / n_frames`
- `FNMR = 1 − FIR`

FIR is always slightly below 1.0 even for a perfect system — the DeepOCSORT tracker requires `min_hits=3` confirmed detections before activating, so the first ~3 frames of each person's window are uncovered.

### 2. Probe-Level

**TPIR** and **FPIR** per NIST FRVT / ISO 19795-1. Each track = one probe transaction.

- `TPIR(T) = correct probes at T / all genuine probes`
- `FPIR(T) = impostor probes at T / all impostor probes`

Stored as `probe_curves` in results.json (101-point threshold sweep from 0.0 to 1.0).

**Probe classification:**

| Probe type | Condition |
|-----------|-----------|
| Genuine-Correct | `true_gt_id ∈ enrolled` AND `name == true_gt_id` |
| Genuine-Wrong | `true_gt_id ∈ enrolled` AND `name != true_gt_id` — penalises TPIR only |
| Impostor | `true_gt_id is None` AND (XML-confirmed bystander OR null probe) |
| FN injection | Enrolled person with zero tracks → (score=0.0, correct=False) |

**Null probe:** a tracked person whose face was too small to process (width < 50 px). Emitted as `UNKNOWN / NO_EMBEDDINGS / similarity=0.0` in eval mode. Correctly rejected at any threshold > 0.

### 3. Person-Level

**PLA** (Person-Level Accuracy) — the paper's primary contribution.

- `PLA = correctly_identified / N`
- `FNR = missed / N`
- `SIR = misidentified / N`
- `PLA + FNR + SIR = 1.0`

Matching is purely spatial via `true_gt_id` — no temporal ordering. A person is:
- **correctly_identified** if ≥1 track has `true_gt_id == pid AND name == pid`
- **missed** if no track was spatially matched to them
- **misidentified** if tracked but no track had the correct name

**Person-level TAR/FAR curve** (`person_curves`): threshold sweep at person level — one episode per GT person (score = max similarity across their tracks), one impostor episode per qualifying track. Enables direct comparison against probe-level TPIR/FPIR.

### 4. Rank-Based

**CMC** (Cumulative Match Characteristic) and **mAP** (mean Average Precision). Computed in `_compute_cmc_map()`. Stored as `cmc_map`.

Each genuine probe contributes a ranked list of up to 10 unique persons (produced by `recognizer.get_top_k_candidates()` — aggregates max cosine similarity per unique gallery identity).

- `CMC[k] = fraction of genuine probes where correct person appears in top-k`
- `mAP = mean(1/rank)` — single relevant item per probe; 0 if not found in top-10

---

## Output

Results written to `volumes/rnd_results/{camera_name}_results.json`.

### Top-level structure

```json
{
  "source": "...",
  "annotation_file": "...",
  "total_frames": 2292,
  "results": [...],
  "evaluation": {...}
}
```

### Per result entry

```json
{
  "first_frame_num": 236,
  "last_frame_num": 299,
  "track_id": 2,
  "name": "0003",
  "recognized": "recognized",
  "status": "RECOGNIZED",
  "similarity": 0.8706,
  "true_gt_id": "0003"
}
```

### evaluation dict keys

```
pla, fnr, sir
correctly_identified, missed, misidentified
total_gt
probe_curves   →  thresholds, tpir, fpir, n_genuine_accepted, n_genuine, n_impostors
person_curves  →  thresholds, tar, far, n_genuine_persons, n_impostor_episodes
cmc_map        →  ranks, cmc, map, n_probes
fir, fnmr, n_frames, n_frames_correct
n_spatial_false_positives, spatial_false_positives
gt_sequence
```

---

## Log Output (INFO level)

```
INFO | Evaluation — PLA: 1.0000  identified: 25/25 (100.0%)  FNR: 0.0000  SIR: 0.0000
INFO |   [Probe-level]   genuine: 25/25  impostors: 2  @ T=0.50 — TPIR: 1.0000, FPIR: 0.0000
INFO |   [Person-level]  persons: 25/25  impostor episodes: 2  @ T=0.50 — TAR: 1.0000, FAR: 0.0000
INFO |   [Rank-based]    Rank-1: 1.0000  Rank-5: 1.0000  mAP: 1.0000  (25 probes)
INFO |   [Frame-level]   FIR: 0.9XXX  FNMR: 0.0XXX  frames: XXXX/1382
```

---

## Reviewer-Added Metrics

Four metrics added to address reviewer feedback. All are computed before `_STRIP` runs, so internal debug fields are available.

### PL-FRR / PL-FAR / Swap Rate

Zero-cost aliases — no new computation. `pl_frr = fnr`, `pl_far = sir`, `swap_rate = sir`. Present in `evaluation` dict alongside the originals.

### DET Curve (`det_curve`)

Derived from `probe_curves` in `_evaluate()`:

```
fnmr[i] = 1 − probe_curves.tpir[i]
fmr[i]  = probe_curves.fpir[i]
```

Plot on log–log axes for standard biometric DET presentation.

### PLA(T) / FIR(T) / MOTA(T) / IDF1(T) Threshold Sweep (`pla_fir_sweep`)

Computed in `_compute_pla_fir_sweep()`, called from `_process_source()` after `_evaluate()`.

- `pla` = `person_curves.tar` (reused, no recomputation)
- `fir` = at each T, count frames where a correctly-ID'd track (name==pid, true_gt_id==pid) has similarity ≥ T
- `mota` = `1 − (FP + FN(T) + IDSW(T)) / GT_total` — person-level MOTA at T (FP is fixed; FN/IDSW vary with T)
- `idf1` = `2·IDTP(T) / (2·IDTP(T) + IDFP(T) + IDFN(T))` — IDF1 at T using best correctly-accepted track per person

Sanity checks: `fir[0]` == scalar `fir`; `mota[50]` ≈ `mot_metrics.mota`; `idf1[50]` ≈ `mot_metrics.idf1`.

### MOTA / IDF1 (`mot_metrics`)

Computed in `_compute_mot_metrics()`, called from `_process_source()` after `_evaluate()`.

**Report with qualifier:** `"approximation": "track-level (not standard frame-level MOTA/IDF1)"` is stored in the dict and should be noted in the paper. Standard MOT evaluation (TrackEval) requires per-frame detection logs which this pipeline does not produce.

---

## Architecture Notes

- **Tracker:** DeepOCSORT — `max_age=30` frames, `min_hits=3`
- **Gallery:** 29 embeddings in `volumes/src/embeddings/main.pkl` (buffalo_l / InsightFace)
- **Eval mode bypass:** in `eval: true`, production filters (counting line, quality gates) are skipped so all tracks reach evaluation regardless of confidence
- **Spatial rejection:** tracks whose XML-confirmed identity disagrees with their claimed name are excluded from PLA but logged in `spatial_false_positives`
