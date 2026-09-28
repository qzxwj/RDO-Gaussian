#!/usr/bin/env python3
"""Compare room canon+stage λ=4096 against stage and canon_l4096."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ROOM = os.path.join(ROOT, "output", "mipnerf360", "room")
ARMS = (
    ("rate2", "rate2"),
    ("canon_rate2", "canon_l32768"),
    ("canon_l4096_rate2", "canon_l4096"),
    ("stage_rate2", "stage"),
    ("canon_stage_l4096_rate2", "canon_stage_l4096"),
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


def softmax_entropy(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p = p / p.sum(axis=-1, keepdims=True)
    p = np.clip(p[0], 1e-12, 1)
    H = float(-(p * np.log2(p)).sum())
    ps = np.sort(p)[::-1]
    n50 = int(np.searchsorted(np.cumsum(ps), 0.5) + 1)
    return H, n50


def arm_row(folder, label):
    run_dir = os.path.join(ROOM, folder)
    results = load_json(os.path.join(run_dir, "results.json"))
    stats = load_json(os.path.join(run_dir, "bitstream", "codec_stats.json"))
    if stats is None and os.path.isdir(os.path.join(run_dir, "bitstream")):
        from report_bitstream_stats import build_stats

        stats = build_stats(os.path.join(run_dir, "bitstream"))
    q = quality(results)
    row = {
        "arm": label,
        "ready": results is not None and stats is not None,
        "PSNR": None if q is None else q["PSNR"],
        "SSIM": None if q is None else q["SSIM"],
        "LPIPS": None if q is None else q["LPIPS"],
        "N": None,
        "total_MB": None,
        "scale_bit": None,
        "rot_bit": None,
        "used_scale": None,
        "H_scale": None,
        "codes50": None,
    }
    if stats:
        n = stats.get("num_gaussians")
        row["N"] = n
        row["total_MB"] = stats.get("total_bytes", 0) / (1024 * 1024)
        ib = stats.get("index_bytes", {})
        if n and ib.get("scale") is not None:
            row["scale_bit"] = ib["scale"] * 8 / n
        if n and ib.get("rot") is not None:
            row["rot_bit"] = ib["rot"] * 8 / n
        used = stats.get("used_codewords", {})
        row["used_scale"] = used.get("scale")
    logits_path = os.path.join(run_dir, "bitstream", "logits.npz")
    if os.path.isfile(logits_path):
        z = np.load(logits_path)
        if "scale" in z.files:
            row["H_scale"], row["codes50"] = softmax_entropy(z["scale"])
    return row


def fmt(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    rows = [arm_row(folder, label) for folder, label in ARMS]
    lines = [
        "# canon+stage comparison (room)",
        "",
        "arm ready PSNR SSIM LPIPS N total_MB scale_bit rot_bit used_scale H_scale codes50",
    ]
    for row in rows:
        lines.append(
            f"{row['arm']} {row['ready']} "
            f"{fmt(row['PSNR'], 4)} {fmt(row['SSIM'], 4)} {fmt(row['LPIPS'], 4)} "
            f"{fmt(row['N'], 0)} {fmt(row['total_MB'], 3)} "
            f"{fmt(row['scale_bit'], 3)} {fmt(row['rot_bit'], 3)} "
            f"{fmt(row['used_scale'], 0)} {fmt(row['H_scale'], 3)} {fmt(row['codes50'], 0)}"
        )
    combo = next(r for r in rows if r["arm"] == "canon_stage_l4096")
    if not combo["ready"]:
        lines.extend(["", "STATUS WAIT", "canon_stage_l4096 尚未出齐", ""])
        rc = 2
    else:
        lines.extend(["", "STATUS DONE", "只对照，不自动开下一臂。", ""])
        rc = 0
    text = "\n".join(lines)
    out_path = os.path.join(ROOT, "output", "_logs", "canon_stage_compare.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(text, end="")
    print(f"wrote {out_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
