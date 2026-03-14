"""XML groundtruth parser and spatial GT matcher for ChokePoint evaluation."""

import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple


def parse_xml_groundtruth(xml_path: str) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Parse ChokePoint per-frame XML groundtruth.

    Args:
        xml_path: path to e.g. P1E_S1_C1.xml

    Returns:
        {frame_num: {person_id: (eye_mid_x, eye_mid_y)}}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result: Dict[int, Dict[str, Tuple[float, float]]] = {}

    for frame_el in root.findall(".//frame"):
        fnum = int(frame_el.get("number"))
        persons = {}
        for person_el in frame_el.findall("person"):
            pid = person_el.get("id")
            left = person_el.find("leftEye")
            right = person_el.find("rightEye")
            if left is None or right is None:
                continue
            lx, ly = float(left.get("x")), float(left.get("y"))
            rx, ry = float(right.get("x")), float(right.get("y"))
            mx, my = (lx + rx) / 2, (ly + ry) / 2
            persons[pid] = (mx, my)
        if persons:
            result[fnum] = persons

    return result


def match_track_to_gt(
    track_boxes: Dict[int, list],
    xml_gt: Dict[int, Dict[str, Tuple[float, float]]],
    min_hit_ratio: float = 0.3,
) -> Tuple[Optional[str], int]:
    """Match a track to a GT person via eye-midpoint-in-bbox containment.

    Args:
        track_boxes: {frame_num: [x1, y1, x2, y2, conf, ...]} — all frames for this track
        xml_gt: output of parse_xml_groundtruth()
        min_hit_ratio: fraction of track frames that must match for assignment

    Returns:
        (GT person ID or None, total_checked) where total_checked is the number of track
        frames that overlapped with XML-annotated frames. total_checked=0 means no XML
        coverage for this track (no data). total_checked>0 with None means a bystander
        was confirmed: XML was present but no GT eye landed inside the bbox.
    """
    hit_counts: Dict[str, int] = {}
    total_checked = 0

    for frame_num, box in track_boxes.items():
        gt_persons = xml_gt.get(frame_num)
        if not gt_persons:
            continue
        total_checked += 1
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        for pid, (mx, my) in gt_persons.items():
            if x1 <= mx <= x2 and y1 <= my <= y2:
                hit_counts[pid] = hit_counts.get(pid, 0) + 1

    if not hit_counts or total_checked == 0:
        return None, total_checked

    best_pid = max(hit_counts, key=hit_counts.__getitem__)
    if hit_counts[best_pid] / total_checked >= min_hit_ratio:
        return best_pid, total_checked
    return None, total_checked
