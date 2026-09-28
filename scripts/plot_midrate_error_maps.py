#!/usr/bin/env python3
"""Mip-NeRF360 midrate per-pixel error maps. PNG only."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

ROOT = Path("/home/lsh/Data/xwj/RDO-Gaussian")

# Match the reference 3x3 order.
SCENES = [
    ["bonsai", "counter", "kitchen"],
    ["room", "bicycle", "garden"],
    ["stump", "flowers", "treehill"],
]
TITLES = {
    "bonsai": "Bonsai",
    "counter": "Counter",
    "kitchen": "Kitchen",
    "room": "Room",
    "bicycle": "Bicycle",
    "garden": "Garden",
    "stump": "Stump",
    "flowers": "Flowers",
    "treehill": "Treehill",
}


def _load_rgb(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _median_psnr_view(per_view_path):
    with open(per_view_path) as f:
        psnr = json.load(f)["ours_30000"]["PSNR"]
    names = sorted(psnr, key=lambda k: psnr[k])
    return names[len(names) // 2]


def _almost_empty(img):
    return float(img.mean()) < 0.02 or float(img.std()) < 0.01


def _view_label(scene_dir, view_name):
    with open(scene_dir / "cameras.json") as f:
        cams = json.load(f)
    idx = int(Path(view_name).stem)
    img_name = cams[idx]["img_name"]
    if not Path(img_name).suffix:
        img_name = f"{img_name}.jpg"
    with open(scene_dir / "per_view.json") as f:
        psnr = float(json.load(f)["ours_30000"]["PSNR"][view_name])
    return f"{img_name}\nPSNR {psnr:.3f}dB"


def pick_view(scene, run):
    scene_dir = ROOT / "output" / "mipnerf360" / scene / run
    test_dir = scene_dir / "test" / "ours_30000"
    name = "00000.png"
    render = test_dir / "renders" / name
    gt = test_dir / "gt" / name
    if not render.is_file() or not gt.is_file() or _almost_empty(_load_rgb(render)):
        name = _median_psnr_view(scene_dir / "per_view.json")
        render = test_dir / "renders" / name
        gt = test_dir / "gt" / name
    label = _view_label(scene_dir, name)
    return name, _load_rgb(render), _load_rgb(gt), label


def add_corner_label(ax, text):
    ax.text(
        0.015,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        linespacing=1.15,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
    )


def error_map(render, gt):
    err = np.abs(render - gt).mean(axis=2)
    vmax = float(np.percentile(err, 99))
    if vmax <= 1e-8:
        vmax = 1e-3
    return err, vmax


def style_axes(ax, title):
    ax.set_title(title, fontsize=12, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_single(scene, err, vmax, label, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=150)
    im = ax.imshow(err, cmap="jet", vmin=0.0, vmax=vmax, interpolation="nearest")
    style_axes(ax, TITLES[scene])
    add_corner_label(ax, label)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Mip-NeRF360 midrate error maps")
    parser.add_argument("--run", default="rate2", help="per-scene output folder name, e.g. rate2 or shac_lst_rate2")
    return parser.parse_args()


def main():
    args = parse_args()
    run = args.run
    out_dir = ROOT / "output" / "mipnerf360" / f"error_maps_{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for row in SCENES:
        for scene in row:
            name, render, gt, label = pick_view(scene, run)
            err, vmax = error_map(render, gt)
            records[scene] = {"name": name, "err": err, "vmax": vmax, "label": label}
            out = out_dir / f"{scene}_error.png"
            save_single(scene, err, vmax, label, out)
            print(f"{scene}: view={name} {label.replace(chr(10), ' | ')} vmax={vmax:.4f} -> {out}")

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 10.5), dpi=150)
    for r, row in enumerate(SCENES):
        for c, scene in enumerate(row):
            rec = records[scene]
            ax = axes[r, c]
            im = ax.imshow(rec["err"], cmap="jet", vmin=0.0, vmax=rec["vmax"], interpolation="nearest")
            style_axes(ax, TITLES[scene])
            add_corner_label(ax, rec["label"])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.8)
    grid_path = out_dir / f"mip360_{run}_error_maps.png"
    fig.savefig(grid_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"grid -> {grid_path}")


if __name__ == "__main__":
    main()
