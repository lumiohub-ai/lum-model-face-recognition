"""Aggregate all 72 ChokePoint results.json into a dataset-level summary.

Usage:
    python scripts/aggregate_results.py

Outputs:
    - Prints a markdown table to stdout
    - Writes volumes/rnd_results/summary.csv
    - Writes volumes/rnd_results/summary.md
"""

import json
import csv
from pathlib import Path

RESULTS_DIR = Path("volumes/rnd_results")
THRESHOLD_IDX = 50  # index for T=0.50 in 0.00..1.00 step 0.01 sweep


def load_result(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
        ev = d.get("evaluation", {})
        if not ev:
            return None

        seq = path.stem.replace("_results", "")
        portal = seq[:3]   # P1E, P1L, P2E, P2L
        direction = "E" if portal[2] == "E" else "L"
        portal_num = portal[1]

        pc = ev.get("probe_curves", {})
        cc = ev.get("cmc_map", {})
        persc = ev.get("person_curves", {})
        mot = ev.get("mot_metrics", {})

        tpir = pc.get("tpir", [None] * 101)[THRESHOLD_IDX]
        fpir = pc.get("fpir", [None] * 101)[THRESHOLD_IDX]
        tar = persc.get("tar", [None] * 101)[THRESHOLD_IDX]
        far = persc.get("far", [None] * 101)[THRESHOLD_IDX]
        cmc = cc.get("cmc", [])
        rank1 = cmc[0] if cmc else None

        return {
            "sequence": seq,
            "portal": portal,
            "direction": direction,
            "portal_num": portal_num,
            "n": ev.get("total_gt"),
            "pla": ev.get("pla"),
            "fnr": ev.get("fnr"),
            "sir": ev.get("sir"),
            "pl_frr": ev.get("pl_frr"),   # PL-FRR = fnr alias
            "pl_far": ev.get("pl_far"),   # PL-FAR = sir alias
            "tpir_50": tpir,
            "fpir_50": fpir,
            "tar_50": tar,
            "far_50": far,
            "fir": ev.get("fir"),
            "fnmr": ev.get("fnmr"),
            "rank1": rank1,
            "map": cc.get("map"),
            "idf1": mot.get("idf1"),
            "mota": mot.get("mota"),
            "n_impostors": pc.get("n_impostors"),
            "n_probes": cc.get("n_probes"),
        }
    except Exception as e:
        print(f"  ERROR loading {path.name}: {e}")
        return None


def fmt(v, decimals=4):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def divergence_flag(row: dict) -> str:
    """Mark sequences where PLA and FIR lead to different conclusions."""
    flags = []
    if row["pla"] is not None and row["fir"] is not None:
        diff = row["pla"] - row["fir"]
        if diff >= 0.15:
            flags.append(f"PLA-FIR↑{diff:.2f}")
    if row["pla"] is not None and row["tpir_50"] is not None:
        diff = row["pla"] - row["tpir_50"]
        if diff >= 0.10:
            flags.append(f"PLA-TPIR↑{diff:.2f}")
    if row.get("mota") is not None and row["pla"] is not None:
        diff = row["pla"] - row["mota"]
        if abs(diff) >= 0.10:
            flags.append(f"PLA-MOTA{diff:+.2f}")
    return " ".join(flags)


def main():
    paths = sorted(RESULTS_DIR.glob("*_results.json"))
    print(f"Found {len(paths)} result files\n")

    rows = []
    for p in paths:
        r = load_result(p)
        if r:
            rows.append(r)

    if not rows:
        print("No valid results found.")
        return

    # Sort by sequence name
    rows.sort(key=lambda r: r["sequence"])

    # --- CSV ---
    csv_path = RESULTS_DIR / "summary.csv"
    fieldnames = ["sequence", "portal", "n", "pla", "fnr", "sir", "pl_frr", "pl_far",
                  "tpir_50", "fpir_50", "tar_50", "far_50",
                  "fir", "fnmr", "rank1", "map", "idf1", "mota",
                  "n_impostors", "n_probes"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {csv_path}\n")

    # --- Markdown table ---
    header = (
        "| Sequence | N | PLA | TPIR@.5 | FPIR@.5 | TAR@.5 | FAR@.5 | FIR | Rank-1 | mAP | MOTA | IDF1 | Impostors | Note |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"

    lines = [header, sep]
    for r in rows:
        flag = divergence_flag(r)
        lines.append(
            f"| {r['sequence']} "
            f"| {r['n'] or '—'} "
            f"| {fmt(r['pla'])} "
            f"| {fmt(r['tpir_50'])} "
            f"| {fmt(r['fpir_50'])} "
            f"| {fmt(r['tar_50'])} "
            f"| {fmt(r['far_50'])} "
            f"| {fmt(r['fir'])} "
            f"| {fmt(r['rank1'])} "
            f"| {fmt(r['map'])} "
            f"| {fmt(r['mota'])} "
            f"| {fmt(r['idf1'])} "
            f"| {r['n_impostors'] if r['n_impostors'] is not None else '—'} "
            f"| {flag} |"
        )

    # Per-condition macro averages
    def macro_avg(subset, key):
        vals = [r[key] for r in subset if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    def section_avg(label, subset):
        return (
            f"| **{label}** "
            f"| {fmt(macro_avg(subset, 'n'), 1)} "
            f"| {fmt(macro_avg(subset, 'pla'))} "
            f"| {fmt(macro_avg(subset, 'tpir_50'))} "
            f"| {fmt(macro_avg(subset, 'fpir_50'))} "
            f"| {fmt(macro_avg(subset, 'tar_50'))} "
            f"| {fmt(macro_avg(subset, 'far_50'))} "
            f"| {fmt(macro_avg(subset, 'fir'))} "
            f"| {fmt(macro_avg(subset, 'rank1'))} "
            f"| {fmt(macro_avg(subset, 'map'))} "
            f"| {fmt(macro_avg(subset, 'mota'))} "
            f"| {fmt(macro_avg(subset, 'idf1'))} "
            f"| {fmt(macro_avg(subset, 'n_impostors'), 1)} "
            f"| |"
        )

    lines.append("| | | | | | | | | | | | |")
    for portal in ["P1E", "P1L", "P2E", "P2L"]:
        sub = [r for r in rows if r["portal"] == portal]
        if sub:
            lines.append(section_avg(f"Mean {portal} (n={len(sub)})", sub))
    lines.append(section_avg(f"**Mean ALL (n={len(rows)})**", rows))

    md_content = "\n".join(lines)

    md_path = RESULTS_DIR / "summary.md"
    md_path.write_text(
        "# ChokePoint Evaluation Summary\n\n"
        f"72 sequences · {len(rows)} with valid evaluation\n\n"
        "Metrics at T=0.50. **Note** column flags divergence cases "
        "(PLA−FIR ≥ 0.15, PLA−TPIR ≥ 0.10, or |PLA−MOTA| ≥ 0.10).\n\n"
        + md_content + "\n"
    )
    print(f"Written: {md_path}\n")

    # Print to stdout
    print(md_content)

    # --- Divergence summary ---
    divergent = [r for r in rows if divergence_flag(r)]
    print(f"\n\n## Divergence cases ({len(divergent)} sequences)\n")
    print("| Sequence | PLA | FIR | PLA-FIR | TPIR@.5 | Note |")
    print("|---|---|---|---|---|---|")
    for r in sorted(divergent, key=lambda x: -(x["pla"] or 0) + (x["fir"] or 0)):
        diff = (r["pla"] or 0) - (r["fir"] or 0)
        print(
            f"| {r['sequence']} "
            f"| {fmt(r['pla'])} "
            f"| {fmt(r['fir'])} "
            f"| {diff:+.4f} "
            f"| {fmt(r['tpir_50'])} "
            f"| {divergence_flag(r)} |"
        )


if __name__ == "__main__":
    main()
