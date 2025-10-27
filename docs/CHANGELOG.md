# V2 Updates

### Technical Description, Methodology, and Module 



## Overview

This system is a robust, modular, and extensible solution for real-time or offline face recognition and entry logging, supporting both single and multi-camera environments. It integrates state-of-the-art models for face detection, recognition, and tracking, and provides flexible logging and visualization capabilities. The architecture is designed for clarity, maintainability, and scalability.

---

## System Architecture & Methodology

### 1. Configuration & Setup
- **Module:** `FaceSetup`
- **Function:** Handles camera stream initialization, configuration loading, and system parameter management.
- **Features:**  
  - Supports RTSP streams, video files, or camera devices.
  - Initializes per-camera processing engines and logging utilities.

### 2. Face Detection
- **Model:** RetinaFace  
- **Library:** [InsightFace](https://github.com/deepinsight/insightface)  
- **Function:** Detects faces in each video frame, providing bounding boxes and detection confidence scores.

### 3. Feature Extraction & Recognition
- **Model:** ArcFace (`w600k_r50.onnx`)  
- **Library:** [InsightFace](https://github.com/deepinsight/insightface)  
- **Function:** 
  - Loads a precomputed database of embeddings and identities.
  - Uses cosine similarity to match detected faces to known identities.
  - Applies a configurable similarity threshold for recognition.

### 4. Multi-Object Tracking
- **Algorithm:** DeepOCSORT  
- **Library:** [boxmot](https://github.com/levan92/boxmot)  
- **Function:**  
  - Tracks detected faces across frames, assigning unique IDs.
  - Maintains track histories (bounding boxes, embeddings, and trajectories).
  - Associates faces across frames for robust identity tracking.

### 5. Similarity Computation
- **Metric:** Cosine Similarity  
- **Library:** [scikit-learn](https://scikit-learn.org/)  
- **Function:**  
  - Compares aggregated track embeddings to the database.
  - Determines the best identity match for each track.

### 6. Entry Logging & Visualization
- **Logging Module:** `EntryLogger` (custom)
- **Visualization Library:** OpenCV (`cv2`)
- **Function:**  
  - Logs recognized entries (IN/OUT events) with metadata (name, track ID, status, timestamp).
  - Optionally sends events to a backend via GraphQL ([gql](https://gql.readthedocs.io/)).
  - Overlays recent recognized entries and system status on video frames.

### 7. Backend Integration
- **API:** GraphQL  
- **Library:** gql  
- **Function:**  
  - Sends recognized entry/exit events to a backend server for persistent storage and analytics.

---

## Processing Pipeline

1. **Frame Acquisition:**  
   Frames are read from each configured video stream in a synchronized loop, supporting both real-time and offline processing.

2. **Face Detection & Tracking:**  
   Each frame is processed by the `FaceEngine` to detect faces and extract embeddings. DeepOCSORT tracks faces across frames, assigning persistent IDs.

3. **Recognition Trigger:**  
   When a track is removed (person leaves the scene) or at the end of the video, embeddings for that track are aggregated and compared to the database using cosine similarity.

4. **Event Logging:**  
   If recognition confidence exceeds the threshold and the person has crossed a configured counting line, the event is logged and optionally sent to the backend.

5. **Visualization & Output:**  
   Annotated frames are displayed (if enabled) and/or saved to video files. Recent recognized entries are overlaid for operator awareness.

---

## Methodological Details

- **Face Embedding Aggregation:**  
  Multiple embeddings per track are aggregated for robust recognition, reducing the impact of outliers.

- **Track Filtering:**  
  Only tracks that cross a logical counting line (e.g., entrance/exit) are logged, reducing false positives.

- **Multi-Camera Support:**  
  The system processes multiple streams in parallel, maintaining independent pipelines for each camera.

- **Extensibility:**  
  - The face embedding database can be updated as needed.
  - Logging can be extended for new backends or notification systems.
  - Detection and recognition models can be swapped/upgraded easily.

---

## Module & Model Summary Table

| Functionality      | Module/Model           | Library/Framework         |
|--------------------|------------------------|--------------------------|
| Detection          | RetinaFace             | InsightFace              |
| Recognition        | ArcFace (w600k_r50)    | InsightFace              |
| Tracking           | DeepOCSORT             | boxmot                   |
| Similarity Metric  | Cosine Similarity      | scikit-learn             |
| Logging            | EntryLogger (custom)   | Python, gql (GraphQL)    |
| Visualization      | -                      | OpenCV                   |

---

## Conclusion

This face recognition and entry logging system leverages advanced models and a modular architecture to deliver accurate, real-time identification and event logging in multi-camera environments. Its design ensures high accuracy, flexibility, and ease of integration for security, attendance, and analytics applications.