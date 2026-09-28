#!/usr/bin/env python3
"""Compare canon scale-tune arms against room rate2 and canon λ=32768."""
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
)
RATE2_PSNR = 30.54677391052246
RATE2_SSIM = 0.9048125147819519
RATE2_LPIPS = 0.24234050512313843


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
    c = np.cumsum(ps)
    n50 = int(np.searchsorted(c, 0.5) + 1)
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
        "folder": folder,
        "ready": results is not None and stats is not None,
        "PSNR": None if q is None else q["PSNR"],
        "SSIM": None if q is None else q["SSIM"],
        "LPIPS": None if q is None else q["LPIPS"],
        "N": None,
        "total_MB": None,
        "scale_bit": None,
        "rot_bit": None,
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


def decide_gate(rows):
    by = {r["arm"]: r for r in rows}
    new = by.get("canon_l4096")
    canon = by.get("canon_l32768")
    if new is None or not new["ready"]:
        return "WAIT", "canon_l4096 尚未出齐 results.json 与 codec_stats.json"
    psnr = new["PSNR"]
    ssim = new["SSIM"]
    lpips = new["LPIPS"]
    scale_bit = new["scale_bit"]
    total_mb = new["total_MB"]
    canon_mb = None if not canon["ready"] else canon["total_MB"]
    quality_ok = (
        psnr is not None
        and psnr >= RATE2_PSNR
        and ssim is not None
        and ssim + 1e-12 >= RATE2_SSIM - 0.002
        and lpips is not None
        and lpips <= RATE2_LPIPS + 0.005
    )
    quality_fail = psnr is None or psnr < RATE2_PSNR or (
        ssim is not None and ssim < RATE2_SSIM - 0.002
    ) or (lpips is not None and lpips > RATE2_LPIPS + 0.01)
    bits_down = (scale_bit is not None and scale_bit <= 12.70) or (
        total_mb is not None and total_mb <= 2.90
    )
    bits_flat = (
        scale_bit is not None
        and scale_bit >= 12.90
        and (
            canon_mb is None
            or total_mb is None
            or (canon_mb - total_mb) < 0.03
        )
    )
    if quality_fail:
        return (
            "NEXT_K4096",
            "质量低于 rate2，停止降 λ。若仍要压 scale 索引，下一臂 canon_k4096，λ 保持 32768。",
        )
    if quality_ok and bits_down:
        return "STOP", "质量不低于 rate2，且 scale 索引或总码流已压下来，不必立刻打 λ=1024。"
    if quality_ok and bits_flat:
        return "NEXT_L1024", "质量仍在，但 rate 几乎没动。下一臂 canon_l1024。"
    return (
        "REVIEW",
        "落在门控边界，需要人工看表后再决定，不自动开下一臂。",
    )


def main():
    rows = [arm_row(folder, label) for folder, label in ARMS]
    lines = [
        "# canon scale-tune comparison (room)",
        "",
        "arm ready PSNR SSIM LPIPS N total_MB scale_bit rot_bit H_scale codes50",
    ]
    for row in rows:
        lines.append(
            f"{row['arm']} {row['ready']} "
            f"{fmt(row['PSNR'], 4)} {fmt(row['SSIM'], 4)} {fmt(row['LPIPS'], 4)} "
            f"{fmt(row['N'], 0)} {fmt(row['total_MB'], 3)} "
            f"{fmt(row['scale_bit'], 3)} {fmt(row['rot_bit'], 3)} "
            f"{fmt(row['H_scale'], 3)} {fmt(row['codes50'], 0)}"
        )
    gate, reason = decide_gate(rows)
    lines.extend(["", f"GATE {gate}", reason, ""])
    text = "\n".join(lines)
    out_path = os.path.join(ROOT, "output", "_logs", "canon_scale_compare.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(text, end="")
    print(f"wrote {out_path}")
    return 0 if gate != "WAIT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
