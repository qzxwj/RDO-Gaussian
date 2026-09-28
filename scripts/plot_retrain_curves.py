#!/usr/bin/env python3
"""Plot midrate retrain curves in the 1x3 faint+smooth reference style."""

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

ROOT = Path("/home/lsh/Data/xwj/RDO-Gaussian")
OUT_DIR = ROOT / "output" / "mipnerf360" / "rate2_retrain_curves"
SCENES = ("room", "garden")

ORANGE = "#E69F00"
PURPLE = "#6B4C9A"
CANON_GREEN = "#009E73"
STAGE_BLUE = "#0072B2"
WIN = 51
OVERLAY_RUNS = (
    ("rate2_retrain", "retrain", PURPLE, "-"),
    ("canon_rate2", "canon", CANON_GREEN, "-."),
    ("stage_rate2", "stage", STAGE_BLUE, "--"),
    ("absgs_canon_stage_l4096_rate2", "absgs", ORANGE, "-"),
)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _col(rows, key, cast=float):
    return np.asarray([cast(r[key]) for r in rows])


def rolling_mean(x, window=WIN):
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    xpad = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(xpad, kernel, mode="valid")[: len(x)]


def rolling_std(x, window=WIN):
    mean = rolling_mean(x, window)
    var = rolling_mean((x.astype(np.float64) - mean) ** 2, window)
    return np.sqrt(np.maximum(var, 0.0))


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 30000)


def _load_run(scene_dir):
    train = _read_csv(scene_dir / "curves.csv")
    data = {
        "it": _col(train, "iter", int),
        "loss": _col(train, "total_loss"),
        "psnr": _col(train, "train_psnr"),
        "n_a": _col(train, "n_active", int),
    }
    test_path = scene_dir / "curves_test.csv"
    if test_path.is_file():
        test = _read_csv(test_path)
        data["test_it"] = _col(test, "iter", int)
        data["test_psnr"] = _col(test, "test_psnr")
    return data


def _available_runs(scene):
    runs = []
    for dirname, label, color, ls in OVERLAY_RUNS:
        scene_dir = ROOT / "output" / "mipnerf360" / scene / dirname
        if (scene_dir / "curves.csv").is_file():
            runs.append((label, color, ls, _load_run(scene_dir)))
    return runs


def plot_scene(scene):
    runs = _available_runs(scene)
    if not runs:
        print(f"{scene}: no curves, skip")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), dpi=150)
    extra = [lab for lab, *_ in runs if lab != "retrain"]

    ax = axes[0]
    for label, color, ls, run in runs:
        ax.plot(run["it"], run["loss"], color=color, linewidth=0.35, alpha=0.14, linestyle=ls)
        ax.plot(run["it"], rolling_mean(run["loss"]), color=color, linewidth=1.8, linestyle=ls, label=label)
    ax.set_title("Training loss (raw faint + smoothed bold)", fontsize=10)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    ax = axes[1]
    for label, color, ls, run in runs:
        ax.plot(run["it"], run["n_a"], color=color, linewidth=1.8, linestyle=ls, label=label)
    ax.set_title("Number of Gaussians", fontsize=10)
    ax.set_xlabel("iteration")
    ax.set_ylabel("N gaussians")
    ax.legend(frameon=False, fontsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style_ax(ax)

    ax = axes[2]
    for label, color, ls, run in runs:
        ax.plot(run["it"], run["psnr"], color=color, linewidth=0.35, alpha=0.14, linestyle=ls)
        ax.plot(run["it"], rolling_mean(run["psnr"]), color=color, linewidth=1.6, linestyle=ls, label=label)
    ax.set_title("PSNR vs iteration", fontsize=10)
    ax.set_xlabel("iteration")
    ax.set_ylabel("PSNR (train)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    title = f"{scene.capitalize()} midrate retrain"
    if extra:
        title = f"{scene.capitalize()} midrate retrain vs {', '.join(extra)}"
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / f"{scene}_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{scene} -> {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        plot_scene(scene)


if __name__ == "__main__":
    main()
