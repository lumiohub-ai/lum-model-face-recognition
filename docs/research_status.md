# Research Status — PLA Paper
## lightfaceR / so.model-face-recognition · branch `rnd/liteFaceR`
*Last updated: 2026-03-15*

---

## 1. What Has Been Done

### Evaluation Pipeline

| Layer | Metric | Method | Status |
|-------|--------|--------|--------|
| Frame | FIR, FNMR | `_compute_frame_metrics` | ✅ |
| Probe/Track | TPIR, FPIR curves | `_compute_biometric_curves` | ✅ |
| Person | PLA, FNR, SIR | `_evaluate` | ✅ |
| Person | TAR/FAR curve | `_compute_person_curves` | ✅ |
| Rank | Rank-1, mAP | `_compute_cmc_map` | ✅ |

All metric definitions follow ISO 19795-1 / NIST FRVT.

### Ground Truth & Spatial Matching
- XML eye-coordinate GT used for all evaluation (JSON annotation dependency dropped)
- `match_track_to_gt` assigns `true_gt_id` via eye-midpoint-in-bbox spatial matching
- Impostor classification: XML-confirmed bystanders + null probes (face too small)
- FN injection for GT persons with zero tracks
- GENUINE-WRONG handling per NIST FRVT — does not inflate FPIR

### Dataset
- 72 ChokePoint sequences: P1E/P1L (12 each) + P2E/P2L (24 each)
- 6 sequences excluded (P2E_S5/P2L_S5) — no XML groundtruth exists
- Full 72-sequence run completed / in progress
- Visualization video per sequence (`volumes/rnd_results/videos/`)

### Output per Sequence
Each `volumes/rnd_results/{camera}_results.json` contains:
- Per-track result entries (identity, similarity, spatial GT match)
- `evaluation.probe_curves` — full TPIR/FPIR threshold sweep
- `evaluation.person_curves` — full TAR/FAR threshold sweep (person level)
- `evaluation.cmc_map` — CMC curve + mAP
- `evaluation.pla`, `fnr`, `sir` — person-level scalars
- `evaluation.fir`, `fnmr` — frame-level scalars

---

## 2. Core Claim — Already Demonstrated

**P2L_S3_C3.2** (leaving portal, 4 GT persons, 11 impostors):

```
PLA  = 1.0000   all 4 persons correctly identified at least once
TPIR = 1.0000   all genuine probes accepted at T=0.50
FIR  = 0.5704   only 57% of annotated frames correct
FNMR = 0.4296
```

This is the paper's key argument: **the system works perfectly by operational criteria (PLA=1.0) but looks mediocre at frame level (FIR=0.57)**. Frame-level evaluation would mislead a deployer; PLA would not.

---

## 3. What Still Needs to Be Done

### Critical — needed before submission

| Task | Description |
|------|-------------|
| **Aggregation script** | Collect all 72 `results.json` → single dataset-level summary table (PLA, TPIR, FIR, Rank-1, mAP per sequence + macro averages per portal/direction) |
| **Divergence analysis** | Systematically identify sequences where metrics disagree — e.g. PLA=1.0 but FIR<0.7, or TPIR<PLA. These are the paper's evidence. |
| **Figures** | DET curve (TPIR vs FPIR), person-level TAR vs FAR, CMC, and the cross-metric comparison table |
| **Multi-camera scope decision** | Abstract mentions "multi-camera episode" — currently single-camera only. Either implement cross-camera aggregation or narrow paper scope. |

### Important — strengthens the paper

| Task | Description |
|------|-------------|
| **Threshold sensitivity** | Show how PLA vs FIR/TPIR divergence changes as operating threshold varies |
| **Per-condition breakdown** | P1 vs P2 (portal/gallery size), entering vs leaving, session (illumination), camera angle |
| **Failure case examples** | Frames/tracks where FIR is low but PLA=1.0 — visualize why (pose, occlusion, blur) |

### Lower priority

| Task | Description |
|------|-------------|
| Remove Rank-5 from log | Only Rank-1 + mAP are meaningful at gallery size 25 — Rank-5 always equals Rank-1 |
| Commit current state | Push all evaluation code changes to remote `rnd/liteFaceR` |

---

## 4. Recommended Next Step

**Write the aggregation script.** Once all 72 results are in, produce a table like:

| Sequence | N | PLA | TPIR@0.5 | FPIR@0.5 | FIR | Rank-1 | mAP | Impostors |
|----------|---|-----|----------|----------|-----|--------|-----|-----------|
| P1E_S1_C1 | 25 | 1.00 | 1.00 | 0.00 | 0.93 | 1.00 | 1.00 | 2 |
| P2L_S3_C3.2 | 4 | 1.00 | 1.00 | 0.00 | 0.57 | 1.00 | 1.00 | 11 |
| ... | | | | | | | | |
| **Mean** | | | | | | | | |

From this table the divergence cases are immediately visible and the paper's argument is empirically grounded.

---

## 5. Metric Reference (quick lookup)

| Symbol | Name | Layer | Formula |
|--------|------|-------|---------|
| PLA | Person-Level Accuracy | Person | `correctly_identified / N` |
| FNR | False Negative Rate | Person | `missed / N` |
| SIR | Substitution/Identity-swap Rate | Person | `misidentified / N` |
| TPIR | True Positive Identification Rate | Probe | `correct probes at T / all genuine probes` |
| FPIR | False Positive Identification Rate | Probe | `impostor probes at T / all impostor probes` |
| TAR | True Accept Rate (person-level) | Person curve | `persons accepted correctly at T / N` |
| FAR | False Accept Rate (person-level) | Person curve | `impostor episodes at T / n_impostors` |
| FIR | Frame Identification Rate | Frame | `correct frames / total annotated frames` |
| FNMR | False Non-Match Rate | Frame | `1 − FIR` |
| Rank-1 | CMC at k=1 | Rank | `probes with correct person at top-1 / n_probes` |
| mAP | mean Average Precision | Rank | `mean(1/rank)` across genuine probes |

**Invariant:** `PLA + FNR + SIR = 1.0`
