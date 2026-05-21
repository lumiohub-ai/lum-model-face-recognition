#!/usr/bin/env python3
"""
Multi-embedding enrollment: CCTV crops + augmented portraits.

Adds multiple embeddings per person to pgvector so ArcFace has gallery
entries that match the actual CCTV domain (top-down, low-res, variable
lighting) instead of only studio portrait embeddings.

Run inside the container:
  docker cp scripts/enroll_multi.py somodel-face-recognition-person-tracking-1:/tmp/enroll_multi.py
  docker exec somodel-face-recognition-person-tracking-1 python3 /tmp/enroll_multi.py
"""

import os
import sys

sys.path.insert(0, '/app/src')
os.environ.setdefault('INSIGHTFACE_HOME', '/tmp/.insightface')

from dotenv import load_dotenv
load_dotenv('/app/.env')

import cv2
import numpy as np
from pathlib import Path

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from infrastructure.storage import PgVectorStore

# ── Config ────────────────────────────────────────────────────────────────────

CLIENT_SLUG = 'mebel'

CCTV_DIR  = Path('/app/enrollment_photos/cctv_temp')
LOCAL_DIR = Path('/app/enrollment_photos')

USERS = {
    "A'zamjon Xusanov": {
        'user_id': '1',
        'portrait': LOCAL_DIR / 'azamjon.jpeg',
        'cctv_crops': [
            CCTV_DIR / 'az_cctv_02.jpg',
            CCTV_DIR / 'az_cctv_04.jpg',
            CCTV_DIR / 'az_cctv_07.jpg',
            CCTV_DIR / 'az_cctv_09.jpg',
            CCTV_DIR / 'az_cctv_11.jpg',
        ],
    },
    'Muhammad Dolixonov': {
        'user_id': '2',
        'portrait': LOCAL_DIR / 'muhammad.jpg',
        'cctv_crops': [
            CCTV_DIR / 'muh_cctv_02.jpg',
            CCTV_DIR / 'muh_cctv_03.jpg',
            CCTV_DIR / 'muh_cctv_04.jpg',
            CCTV_DIR / 'muh_cctv_05.jpg',
            CCTV_DIR / 'muh_cctv_08.jpg',
        ],
    },
}

# ── InsightFace setup ─────────────────────────────────────────────────────────

def build_app() -> FaceAnalysis:
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_embedding(app: FaceAnalysis, img_bgr: np.ndarray):
    """Detect best face and return (embedding, aligned_112x112) or (None, None)."""
    faces = app.get(img_bgr)
    if not faces:
        return None, None
    face = max(faces, key=lambda f: f.det_score)
    aligned = face_align.norm_crop(img_bgr, landmark=face.kps, image_size=112)
    emb = face.embedding.copy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb, aligned


def embed_aligned(app: FaceAnalysis, aligned_112: np.ndarray) -> np.ndarray:
    """Embed a pre-aligned 112×112 BGR crop directly via the recognition model."""
    rec = app.models['recognition']
    feat = rec.get_feat(aligned_112)
    emb = np.array(feat).flatten()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb

# ── CCTV-style augmentations on 112×112 aligned face ─────────────────────────

def augment_aligned(aligned: np.ndarray):
    """Yield augmented versions of a 112×112 aligned face crop.

    Simulates CCTV conditions: low resolution, blur, noise, lighting variation,
    JPEG compression artifacts, and slight perspective changes.
    """
    variants = []

    # 1-3: Resolution degradation (simulates 20-35px CCTV face)
    for target_px in [22, 28, 35]:
        small = cv2.resize(aligned, (target_px, target_px), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(small, (112, 112), interpolation=cv2.INTER_LINEAR)
        variants.append(('res_{}'.format(target_px), upscaled))

    # 4-5: Motion / focus blur
    for sigma in [1.2, 2.0]:
        blurred = cv2.GaussianBlur(aligned, (0, 0), sigma)
        variants.append(('blur_{}'.format(int(sigma * 10)), blurred))

    # 6-7: Brightness (dim / bright lighting)
    for alpha in [0.65, 1.35]:
        bright = np.clip(aligned.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        variants.append(('bright_{}'.format(int(alpha * 100)), bright))

    # 8: Low contrast (flat CCTV sensor)
    lo = np.clip(aligned.astype(np.float32) * 0.6 + 50, 0, 255).astype(np.uint8)
    variants.append(('lowcon', lo))

    # 9: Gaussian noise
    noise = np.random.normal(0, 12, aligned.shape).astype(np.float32)
    noisy = np.clip(aligned.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    variants.append(('noise', noisy))

    # 10-11: JPEG compression artifacts
    for quality in [20, 40]:
        _, buf = cv2.imencode('.jpg', aligned, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        variants.append(('jpeg_{}'.format(quality), compressed))

    # 12: Combined: low-res + blur (most CCTV-like)
    small = cv2.resize(aligned, (28, 28), interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(small, (112, 112), interpolation=cv2.INTER_LINEAR)
    combined = cv2.GaussianBlur(upscaled, (0, 0), 1.2)
    variants.append(('res_blur', combined))

    return variants


# ── Main enrollment ───────────────────────────────────────────────────────────

def main():
    print("==> Initializing InsightFace (buffalo_l)...")
    app = build_app()

    print("==> Connecting to pgvector...")
    store = PgVectorStore(CLIENT_SLUG)

    total_inserted = 0
    total_skipped = 0

    for user_name, cfg in USERS.items():
        user_id = cfg['user_id']
        print(f"\n── {user_name} (id={user_id}) ──")

        # ── 1. CCTV crops ────────────────────────────────────────────────────
        print("  [CCTV crops]")
        for crop_path in cfg['cctv_crops']:
            if not crop_path.exists():
                print(f"    SKIP {crop_path.name} — file not found")
                continue

            img = cv2.imread(str(crop_path))
            if img is None:
                print(f"    SKIP {crop_path.name} — failed to load")
                continue

            emb, _ = get_embedding(app, img)
            if emb is None:
                print(f"    SKIP {crop_path.name} — no face detected")
                continue

            url_norm = f"cctv_{crop_path.stem}"
            result = store.add_embedding(
                user_id=user_id,
                user_name=user_name,
                image_url=f"cctv://{crop_path.name}",
                embedding=emb,
                metadata={'source': 'cctv_crop', 'file': crop_path.name},
                image_url_norm_override=url_norm,
            )
            if result:
                print(f"    OK   {crop_path.name} → id={result}")
                total_inserted += 1
            else:
                print(f"    DUP  {crop_path.name} — already in DB")
                total_skipped += 1

        # ── 2. Portrait + augmentations ──────────────────────────────────────
        print("  [Portrait + augmentations]")
        portrait_path = cfg['portrait']
        if not portrait_path.exists():
            print(f"    SKIP portrait — {portrait_path} not found")
            continue

        portrait_img = cv2.imread(str(portrait_path))
        if portrait_img is None:
            print(f"    SKIP portrait — failed to load")
            continue

        # Original portrait embedding
        emb, aligned = get_embedding(app, portrait_img)
        if emb is None:
            print(f"    SKIP portrait — no face detected")
            continue

        url_norm = f"portrait_{portrait_path.stem}_original"
        result = store.add_embedding(
            user_id=user_id,
            user_name=user_name,
            image_url=f"portrait://{portrait_path.name}",
            embedding=emb,
            metadata={'source': 'portrait_original', 'file': portrait_path.name},
            image_url_norm_override=url_norm,
        )
        if result:
            print(f"    OK   {portrait_path.name} original → id={result}")
            total_inserted += 1
        else:
            print(f"    DUP  {portrait_path.name} original — already in DB")
            total_skipped += 1

        # Augmented versions from aligned crop
        augmented = augment_aligned(aligned)
        for aug_name, aug_crop in augmented:
            aug_emb = embed_aligned(app, aug_crop)

            url_norm = f"portrait_{portrait_path.stem}_aug_{aug_name}"
            result = store.add_embedding(
                user_id=user_id,
                user_name=user_name,
                image_url=f"portrait_aug://{portrait_path.stem}_{aug_name}",
                embedding=aug_emb,
                metadata={'source': 'portrait_augmented', 'augmentation': aug_name},
                image_url_norm_override=url_norm,
            )
            if result:
                print(f"    OK   portrait aug={aug_name} → id={result}")
                total_inserted += 1
            else:
                print(f"    DUP  portrait aug={aug_name} — already in DB")
                total_skipped += 1

    print(f"\n==> Done. Inserted: {total_inserted}  Skipped (duplicates): {total_skipped}")
    print("\n==> Recommended config update after enrollment:")
    print("    match_threshold: 0.32  (was 0.22)")
    print("    match_margin:    0.06  (was 0.04)")


if __name__ == '__main__':
    np.random.seed(42)
    main()
