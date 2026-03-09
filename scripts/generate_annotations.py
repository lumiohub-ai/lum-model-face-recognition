"""Regenerate ChokePoint annotation JSONs with explicit start+end frames.

Reads:  volumes/src/annotation/chokepoint/groundtruth/{P1,P2}*.xml
Writes: volumes/src/annotation/chokepoint/P1E/*.json  (overwrites)
        volumes/src/annotation/chokepoint/P1L/*.json  (overwrites)
        volumes/src/annotation/chokepoint/P2E/<base>/*.json  (new)
        volumes/src/annotation/chokepoint/P2L/<base>/*.json  (new)

Run from repo root:
    python scripts/generate_annotations.py
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

GROUNDTRUTH_DIR = Path("volumes/src/annotation/chokepoint/groundtruth")
ANNOTATION_ROOT = Path("volumes/src/annotation/chokepoint")


def xml_to_annotation(xml_path: Path) -> dict:
    """Return {person_id: {"start": str, "end": str}} ordered by first appearance."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    first_frame: dict[str, int] = {}
    last_frame: dict[str, int] = {}

    for frame in root.findall("frame"):
        frame_num = int(frame.get("number"))
        for person in frame.findall("person"):
            pid = person.get("id")
            if pid not in first_frame:
                first_frame[pid] = frame_num
            last_frame[pid] = frame_num  # always update -> tracks last seen

    ordered = sorted(first_frame.items(), key=lambda x: x[1])
    return {
        pid: {"start": str(first_frame[pid]), "end": str(last_frame[pid])}
        for pid, _ in ordered
    }


def annotation_output_path(seq_id: str) -> Path:
    """Mirror the logic in build_experiment_config.py:annotation_path().

    Mirrors the frames directory structure:
      P1: {portal_type}/{session}/{base_id}/{seq_id}.json
          e.g. P1E/P1E_S1/P1E_S1_C1/P1E_S1_C1.json
      P2: {portal_type}/{session}/{base_id}/{seq_id}.json
          e.g. P2E/P2E_S1/P2E_S1_C1/P2E_S1_C1.1.json
    """
    portal_type = seq_id[:3]           # P1E, P1L, P2E, P2L
    session = seq_id[:6]               # P1E_S1, P2E_S3, etc.
    base_id = seq_id.rsplit(".", 1)[0] if "." in seq_id else seq_id  # P2E_S1_C1 or P1E_S1_C1
    return ANNOTATION_ROOT / portal_type / session / base_id / f"{seq_id}.json"


def main() -> None:
    xmls = sorted(GROUNDTRUTH_DIR.glob("*.xml"))
    print(f"Found {len(xmls)} XML files")

    for xml_path in xmls:
        seq_id = xml_path.stem  # e.g. P1E_S1_C1 or P2E_S1_C1.1
        out_path = annotation_output_path(seq_id)
        annotation = xml_to_annotation(xml_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        action = "updated" if out_path.exists() else "created"
        with open(out_path, "w") as f:
            json.dump(annotation, f, indent=4)

        print(f"  [{action}] {seq_id}: {len(annotation)} persons -> {out_path}")


if __name__ == "__main__":
    main()
