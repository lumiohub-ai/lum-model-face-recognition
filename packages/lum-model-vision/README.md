# lum-model-vision

Face detection, person tracking, cross-camera re-identification and action
recognition — packaged for reuse outside the SmartOffice application.

## Install

Verified end-to-end in a clean venv on CPU. Order matters:

```bash
python -m venv .venv && source .venv/bin/activate

# 1. torch first, so the CPU build wins. GPU deployments use their own
#    --index-url (see the repo Dockerfile) instead of this line.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. the forked insightface — builds a Cython extension, needs gcc
pip install ./modules/insightface

# 3. the package itself; pulls the boxmot fork from git
pip install -e ./packages/lum-model-vision
```

`boxmot` *is* declared, pinned to the `humblebeeintel/yolo_tracking` fork, and
resolves from git — no submodule or `sys.path` manipulation required.

InsightFace is **not** declared. The copy in `modules/insightface` is a fork (it
drops `genderage.onnx` from the loaded model set), so PyPI's 0.7.3 is not a
substitute, and the fork has no git remote to pin. Hence step 2.

### Version constraints, and why

These are pinned because something breaks otherwise, not out of caution:

| Pin | Reason |
|---|---|
| `numpy<2` | The boxmot fork pins `numpy==1.24.4` exactly. |
| `opencv-python-headless~=4.11.0` | 4.12+ requires `numpy>=2`, which the above rules out. |
| `setuptools<81` | The boxmot fork imports `pkg_resources`, removed in setuptools 81. It does not declare this itself. |
| `onnxruntime` (CPU) | See below. |

There is deliberately **no `[gpu]` extra**: `onnxruntime` and `onnxruntime-gpu`
install into the same `onnxruntime/` directory, so an extra would leave both
present and the result undefined. GPU deployments uninstall the CPU build and
install the GPU one — the repo Dockerfile does exactly this, mirroring the swap
it already performs for opencv.

## Use

```python
from lum_vision import ModelFactory, VisionConfig, InMemoryEmbeddingProvider

config = VisionConfig(match_threshold=0.35, model_cache_dir=Path("volumes/models"))
models = ModelFactory(config, embedding_provider=my_store)

faces = models.face_detector.extract_face_features(frame)
sims = models.face_matcher.compute_similarities(embeddings)
idx, score = models.face_matcher.get_best_match(sims)
```

Models are built on first access, so constructing the factory is cheap and you
only pay for what you touch.

### Supplying embeddings

`FaceMatcher` reads known faces through the `EmbeddingProvider` protocol —
anything with `get_all_embeddings() -> (names, embeddings)`. A pgvector store, a
REST client, or the bundled `InMemoryEmbeddingProvider` all work:

```python
provider = InMemoryEmbeddingProvider(names=["alice"], embeddings=embs)
```

This is why the package needs no database driver of its own.

## What this package will not do

These are guarantees, not omissions — the host application owns each one:

- **No threads.** `ActionRecognizer.recognize()` blocks; drive it from your own
  pool. Nothing is started behind your back, so shutdown ordering stays yours.
- **No I/O beyond inference.** No database, queue, or object storage. Results are
  returned, never dispatched.
- **No writes outside `model_cache_dir`.** Every weight file — buffalo_l,
  yolo26s, OSNet — resolves under that one root. Set it somewhere persistent and
  mount it; the package neither knows nor cares that Docker exists.
- **No GPU setup.** A plain `pip install` pulls CPU torch. CUDA builds come from
  your own image (`torch==2.4.0+cu121` via PyTorch's index).

## Model weights

Roughly 370 MB, downloaded on first use rather than shipped:

| Model | Purpose | Lands in |
|---|---|---|
| buffalo_l | face detection + embeddings | `model_cache_dir/insightface` |
| yolo26s | person detection | `model_cache_dir/weights` |
| osnet_x0_25_msmt17 | body ReID | `model_cache_dir/weights` |

`LUM_VISION_MODEL_DIR` overrides the default (`~/.cache/lum-vision`), though
setting `VisionConfig.model_cache_dir` directly is clearer.

## Tests

```bash
pytest packages/lum-model-vision/tests -q
```

No GPU, database, or network required.
