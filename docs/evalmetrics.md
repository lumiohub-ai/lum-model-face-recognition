# Metrics for Face Recognition Evaluation

## Overview

This document describes the evaluation methodology used to assess face recognition systems. Unlike traditional frame-by-frame detection metrics, person-based metrics focus on whether each unique individual in the dataset was correctly identified at least once, making them more suitable for evaluating identity recognition in surveillance and security applications.

## Metric Definitions

### Basic Counts

- **True Positives (TP)**: Number of people who were correctly identified at least once.
- **False Positives (FP)**: Number of incorrect identifications (predicted identities that don't exist in ground truth or were incorrectly assigned).
- **False Negatives (FN)**: Number of people who were never correctly identified.

### Derived Metrics

- **Precision**: Proportion of identified people who were correctly identified.
  ```
  Precision = TP / (TP + FP)
  ```

- **Recall**: Proportion of ground truth people who were correctly identified at least once.
  ```
  Recall = TP / (TP + FN)
  ```

- **F1 Score**: Harmonic mean of precision and recall.
  ```
  F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
  ```

- **Accuracy**: Proportion of all decisions (identifications and non-identifications) that were correct.
  ```
  Accuracy = TP / (TP + FP + FN)
  ```

## Calculation Method

The person-based metrics are calculated from a confusion matrix that shows the predicted identity of each person:

1. **True Positives (TP)**: Count people on the diagonal (where predicted identity = ground truth identity) who have at least one detection.

2. **False Positives (FP)**: Count predicted identities that either:
   - Don't exist in the ground truth dataset, or
   - Were assigned to the wrong person (off-diagonal entries in the confusion matrix)

3. **False Negatives (FN)**: Count ground truth people who were never correctly identified (zero entries on the diagonal).

## Comparison with Detection-based Metrics

Person-based metrics differ from traditional detection-based metrics in several important ways:

| Aspect | Detection-based Metrics | Person-based Metrics |
|--------|------------------------|---------------------|
| Focus | Each detection is counted separately | Each person is counted once |
| When used | Evaluating frame-by-frame performance | Evaluating overall identity recognition |
| TP definition | Correct identity in a specific detection | Person correctly identified at least once |
| Strengths | Detailed analysis of system performance | Better assessment of real-world utility |
| Use case | Fine-tuning detection algorithms | Evaluating security/surveillance applications |

## Example Interpretation

For a system that achieves:
- TP = 6 (six people correctly identified)
- FP = 4 (four incorrect identifications)
- FN = 3 (three people never identified)

The metrics would be:
- Precision = 6/(6+4) = 0.6000 (60%)
- Recall = 6/(6+3) = 0.6667 (66.67%)
- F1 Score = 2*(0.6*0.6667)/(0.6+0.6667) = 0.6316 (63.16%)
- Accuracy = 6/(6+4+3) = 0.4615 (46.15%)

This means the system correctly identified 60% of the people it detected and found 66.67% of all people in the dataset.
