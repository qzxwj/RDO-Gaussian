#!/usr/bin/env python3
"""Room AbsGS vs canon+stage and rate2. No next-arm launch."""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

BASE = os.path.join(ROOT, "output", "mipnerf360")
NEW_ARM = "absgs_canon_stage_l4096_rate2"
ARMS = (
    ("rate2", "rate2"),
    ("canon_stage_l4096_rate2", "canon_stage_l4096"),
    (NEW_ARM, "absgs_canon_stage"),
)


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def quality(results):
    if not results:
        return None
    block = results.get("ours_30000") or next(iter(results.values()))
    return {
        "PSNR": block.get("PSNR"),
        "SSIM": block.get("SSIM"),
        "LPIPS": block.get("LPIPS"),
    }


def n_from_curves(run_dir):
    path = os.path.join(run_dir, "curves.csv")
    if not os.path.isfile(path):
        return None, None
    peak_15k = None
    final_n = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "iter" not in row or "n_gaussians" not in row:
                continue
            it = int(float(row["iter"]))
            n = int(float(row["n_gaussians"]))
            final_n = n
            if it <= 15000:
                if peak_15k is None or n > peak_15k:
                    peak_15k = n
    return peak_15k, final_n


def arm_row(folder, label):
    run_dir = os.path.join(BASE, "room", folder)
    results = load_json(os.path.join(run_dir, "results.json"))
    stats = load_json(os.path.join(run_dir, "bitstream", "codec_stats.json"))
    if stats is None and os.path.isdir(os.path.join(run_dir, "bitstream")):
        from report_bitstream_stats import build_stats

        stats = build_stats(os.path.join(run_dir, "bitstream"))
    q = quality(results)
    peak_15k, curve_final = n_from_curves(run_dir)
    row = {
        "arm": label,
        "ready": results is not None,
        "PSNR": None if q is None else q["PSNR"],
        "SSIM": None if q is None else q["SSIM"],
        "LPIPS": None if q is None else q["LPIPS"],
        "peak_N_15k": peak_15k,
        "N": None,
        "total_MB": None,
        "scale_bit": None,
    }
    if stats:
        n = stats.get("num_gaussians")
        row["N"] = n if n is not None else curve_final
        row["total_MB"] = stats.get("total_bytes", 0) / (1024 * 1024)
        ib = stats.get("index_bytes", {})
        if n and ib.get("scale") is not None:
            row["scale_bit"] = ib["scale"] * 8 / n
    elif curve_final is not None:
        row["N"] = curve_final
    return row


def fmt(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def delta_line(name, a, b, keys=("PSNR", "SSIM", "LPIPS", "N", "total_MB")):
    parts = [name]
    for key in keys:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            parts.append(f"{key}=NA")
        else:
            digits = 4 if key in ("PSNR", "SSIM", "LPIPS") else 0 if key == "N" else 3
            parts.append(f"d_{key}={vb - va:+.{digits}f}")
    return " ".join(parts)


def main():
    rows = [arm_row(folder, label) for folder, label in ARMS]
    by_arm = {r["arm"]: r for r in rows}
    lines = [
        "# room AbsGS isolation: canon+stage vs AbsGS",
        "# primary: absgs_canon_stage - canon_stage_l4096",
        "",
        "arm ready PSNR SSIM LPIPS peak_N_15k N total_MB scale_bit",
    ]
    for row in rows:
        lines.append(
            f"{row['arm']} {row['ready']} "
            f"{fmt(row['PSNR'], 4)} {fmt(row['SSIM'], 4)} {fmt(row['LPIPS'], 4)} "
            f"{fmt(row['peak_N_15k'], 0)} {fmt(row['N'], 0)} "
            f"{fmt(row['total_MB'], 3)} {fmt(row['scale_bit'], 3)}"
        )

    new_row = by_arm["absgs_canon_stage"]
    if not new_row["ready"]:
        lines.extend(["", "STATUS WAIT", "未齐: room absgs_canon_stage", ""])
        rc = 2
    else:
        lines.append("")
        lines.append(
            delta_line(
                "PRIMARY AbsGS vs canon_stage",
                by_arm["canon_stage_l4096"],
                new_row,
            )
        )
        lines.extend(["", "STATUS DONE", "只对照 room，不自动开下一臂。", ""])
        rc = 0

    text = "\n".join(lines)
    out_path = os.path.join(ROOT, "output", "_logs", "absgs_compare.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(text, end="")
    print(f"wrote {out_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
