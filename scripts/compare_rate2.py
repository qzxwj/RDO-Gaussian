#!/usr/bin/env python3
"""Compare canon/stage ablation runs against existing mipnerf360 rate2 baselines."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ("room", "garden")
ARMS = (
    ("rate2", "baseline"),
    ("canon_rate2", "canon"),
    ("stage_rate2", "stage"),
)


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def quality_row(results):
    if not results:
        return {"PSNR": None, "SSIM": None, "LPIPS": None}
    block = results.get("ours_30000") or next(iter(results.values()))
    return {
        "PSNR": block.get("PSNR"),
        "SSIM": block.get("SSIM"),
        "LPIPS": block.get("LPIPS"),
    }


def fmt(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    rows = []
    for scene in SCENES:
        for folder, arm in ARMS:
            run_dir = os.path.join(ROOT, "output", "mipnerf360", scene, folder)
            results = load_json(os.path.join(run_dir, "results.json"))
            stats = load_json(os.path.join(run_dir, "bitstream", "codec_stats.json"))
            q = quality_row(results)
            index_bytes = (stats or {}).get("index_bytes", {})
            used = (stats or {}).get("used_codewords", {})
            rows.append({
                "scene": scene,
                "arm": arm,
                "run_dir": run_dir,
                "ready": results is not None,
                "PSNR": q["PSNR"],
                "SSIM": q["SSIM"],
                "LPIPS": q["LPIPS"],
                "total_MB": None if stats is None else stats.get("total_bytes", 0) / (1024 * 1024),
                "scale_index_B": index_bytes.get("scale"),
                "rot_index_B": index_bytes.get("rot"),
                "dc_index_B": index_bytes.get("dc"),
                "used_scale": used.get("scale"),
                "used_rot": used.get("rot"),
                "used_dc": used.get("dc"),
            })

    lines = [
        "# rate2 comparison",
        "",
        "scene arm ready PSNR SSIM LPIPS total_MB scale_B rot_B dc_B used_scale used_rot used_dc",
    ]
    for row in rows:
        lines.append(
            f"{row['scene']} {row['arm']} {row['ready']} "
            f"{fmt(row['PSNR'], 4)} {fmt(row['SSIM'], 4)} {fmt(row['LPIPS'], 4)} "
            f"{fmt(row['total_MB'], 3)} {fmt(row['scale_index_B'], 0)} "
            f"{fmt(row['rot_index_B'], 0)} {fmt(row['dc_index_B'], 0)} "
            f"{fmt(row['used_scale'], 0)} {fmt(row['used_rot'], 0)} {fmt(row['used_dc'], 0)}"
        )
    text = "\n".join(lines) + "\n"
    out_path = os.path.join(ROOT, "output", "_logs", "rate2_compare.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(text, end="")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
