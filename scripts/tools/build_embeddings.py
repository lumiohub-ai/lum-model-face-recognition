"""Build face embeddings pickle file from a flat folder of labeled face images.

Image naming convention:  {name}_{index}.jpg
Example:
    data/faces/
    ├── oybek_1.jpg
    ├── oybek_2.jpg
    ├── john_1.jpg
    └── john_2.jpg

The part before the first '_' becomes the person's name in the database.
Multiple images per person are all embedded (more = better recall).

Local setup (without Docker):
    pip install -r requirements.rnd.txt
    pip install ./modules/insightface
    pip install -e .
    python scripts/tools/build_embeddings.py --input volumes/src/images/folder_1 --output volumes/src/embeddings/main.pkl

Inside Docker:
    python scripts/tools/build_embeddings.py --input volumes/src/images/folder_1 --output volumes/src/embeddings/main.pkl
"""

import argparse
import os
import pickle

import cv2
import numpy as np

from face_recognition.core.detector import FaceDetector


def build_embeddings(input_dir: str, output_path: str, gpu_id: int) -> None:
    print(f"Loading face detector on GPU {gpu_id}...")
    detector = FaceDetector(gpu_id=gpu_id)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".pgm"))
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} image(s)\n")

    names = []
    embeddings = []
    skipped = 0

    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]          # e.g. "oybek_1" or "0001"
        # Accept both {name}_{index} and plain {name} formats
        name = stem.split("_")[0] if "_" in stem else stem

        img_path = os.path.join(input_dir, img_file)
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            # Fallback for formats cv2 may read as grayscale (e.g. PGM)
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(f"  [SKIP] Cannot read: {img_path}")
                skipped += 1
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Upscale small face crops so the detector can find the face
        h, w = frame.shape[:2]
        if max(h, w) < 256:
            scale = 256 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        emb = detector.compute_embedding(frame)
        if emb is None:
            print(f"  [SKIP] No face detected: {img_file}")
            skipped += 1
            continue

        names.append(stem)           # store full stem e.g. "oybek_1" or "0001"
        embeddings.append(emb)
        print(f"  [OK]   {img_file}  →  name: '{name}'")

    if not embeddings:
        print("\nNo embeddings collected. Check your images.")
        return

    emb_array = np.array(embeddings)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"embeddings": emb_array, "names": names}, f)

    unique_persons = len(set(n.split("_")[0] for n in names))
    print(f"\nSaved {len(names)} embeddings ({unique_persons} person(s)) → {output_path}")
    if skipped:
        print(f"Skipped {skipped} image(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build face embeddings pickle from flat image folder")
    parser.add_argument("--input", default="volumes/src/images", help="Folder with {name}_{index}.jpg images")
    parser.add_argument("--output", default="volumes/src/embeddings/main.pkl", help="Output .pkl path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (-1 for CPU)")
    args = parser.parse_args()

    build_embeddings(args.input, args.output, args.gpu)
