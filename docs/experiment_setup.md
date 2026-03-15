# Experimental Setup

## Dataset

We evaluate on the **ChokePoint** dataset [cite], a controlled indoor surveillance benchmark recorded at two portal cameras (P1 and P2) capturing subjects entering and leaving a building. The dataset provides frame-level identity annotations in XML format, with per-frame eye-midpoint coordinates used for spatial ground-truth matching.

We use 72 sequences across four conditions:

| Condition | Description | Sequences | Enrolled persons/seq |
|-----------|-------------|-----------|---------------------|
| P1E | Portal 1, entering | 12 | 25 |
| P1L | Portal 1, leaving | 12 | 25 |
| P2E | Portal 2, entering | 24 | 5–24 (mean 13.9) |
| P2L | Portal 2, leaving | 24 | 4–25 (mean 14.5) |

Six sequences (P2E_S5, P2L_S5) are excluded due to missing ground-truth annotation. Each sequence contains unannotated persons treated as **impostors** (mean 7.5 per sequence; maximum 51). All evaluation is derived solely from XML ground truth; no supplementary JSON annotations are used.

---

## System Architecture

The recognition pipeline consists of three sequential stages.

### Face Detection and Embedding

Face detection and 512-dimensional embedding extraction use **InsightFace** (`buffalo_l`) [cite], a state-of-the-art ArcFace-based model. A minimum face size gate of **50 px** is applied; faces below this threshold are treated as null probes in the evaluation protocol (see below). Detected face crops are padded by 20% via border replication before embedding extraction.

### Multi-Object Tracking

Detected faces are tracked using **DeepOCSORT** [cite], an appearance-based multi-object tracker. Tracker parameters:

| Parameter | Value | Implication |
|-----------|-------|-------------|
| min\_hits | 3 | Track confirmed after 3 consecutive detections (~120 ms at 25 fps) |
| max\_age | 30 frames | Track retired after 30 missed frames (1.2 s) |
| Feature dim | 512-D | Shared with recognition embeddings |

Each confirmed track produces exactly one identity decision (probe), treating the video episode as the unit of recognition.

### Face Recognition

Identity decisions are made by comparing the best-frame embedding of each track against a pre-built gallery of 512-D embeddings using **cosine similarity**:

$$s = \frac{\mathbf{e}_{\text{probe}} \cdot \mathbf{e}_{\text{gallery}}}{\|\mathbf{e}_{\text{probe}}\| \, \|\mathbf{e}_{\text{gallery}}\|}$$

The gallery may contain multiple embeddings per enrolled identity; the maximum similarity across all gallery vectors for a given person is used as that person's score. Classification follows a three-zone scheme:

| Decision | Condition |
|----------|-----------|
| RECOGNIZED | $s \geq 0.40$ |
| UNCERTAIN | $0.22 \leq s < 0.40$ (with top-2 margin $\geq 0.08$) |
| UNKNOWN | $s < 0.22$ |

In evaluation mode, two production filters are disabled to ensure complete probe coverage: (1) a null-image gate that would otherwise discard low-quality tracks, and (2) an 8-stage dashboard filter. This enables full impostor track capture including faces below the minimum size threshold.

---

## Evaluation Protocol

We evaluate at four hierarchical levels, progressing from fine-grained frame observations to coarse set-level identity decisions. All metrics are computed against XML ground truth via spatial matching (eye-midpoint proximity).

### Probe definitions

- **Genuine probe**: a track whose ground-truth identity ($\texttt{true\_gt\_id}$) belongs to the enrolled gallery.
- **Impostor probe**: a track with no matching ground-truth identity ($\texttt{true\_gt\_id} = \text{None}$), provided at least 3 of its frames were checked against the XML annotation, or the track produced zero embeddings (face-too-small case).
- **False-negative injection**: a GT identity with zero associated tracks contributes a null probe $(s = 0.0, \text{correct} = \text{False})$ to prevent missed detections from being ignored.

### Layer 1 — Frame-level

$$\text{FIR} = \frac{|\text{frames correctly labeled}|}{|\text{annotated frames}|}, \qquad \text{FNMR} = 1 - \text{FIR}$$

A frame is correctly labeled if the active track covering it has both the correct identity assignment and a confirmed spatial match.

### Layer 2 — Probe / Track-level (per NIST FRVT / ISO 19795-1)

$$\text{TPIR}(T) = \frac{|\text{genuine probes accepted at } T|}{|\text{genuine probes}|}, \qquad \text{FPIR}(T) = \frac{|\text{impostor probes accepted at } T|}{|\text{impostor probes}|}$$

Each track constitutes one probe/transaction. Misidentified genuine probes (wrong identity accepted) remain in the genuine denominator and do not inflate FPIR, consistent with NIST FRVT convention.

### Layer 3 — Person-level

$$\text{PLA} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{p}_i = g_i]$$

where $N$ is the number of enrolled GT persons and $\hat{p}_i$ is the system's identity decision for person $i$, determined by the highest-scoring track spatially matched to that person. PLA partitions exactly with two complementary rates:

$$\text{PL-FRR} + \text{PL-FAR} + \text{PLA} = 1$$

where PL-FRR is the enrolled-to-UNKNOWN rejection rate and PL-FAR is the impostor-accepted-as-known rate, both computed at episode (person-encounter) granularity.

### Layer 4 — Rank-based

$$\text{CMC}[k] = \frac{|\{\text{genuine probes}: \text{rank of correct identity} \leq k\}|}{|\text{genuine probes}|}$$

$$\text{mAP} = \frac{1}{|\text{genuine probes}|} \sum_i \frac{1}{\text{rank}(i)}$$

Ranking is over the top-10 unique gallery persons by maximum cosine similarity. mAP under single-relevant-item retrieval reduces to mean reciprocal rank (MRR).

### Reporting

All metrics are evaluated at a fixed operating threshold $T = 0.50$ for scalar reporting, and swept over $T \in [0, 1]$ (step 0.01) for curve figures. Scalar results are macro-averaged (unweighted) across all 72 sequences. Per-condition (P1E/P1L/P2E/P2L) averages are reported separately to isolate the effect of portal direction and gallery size.
