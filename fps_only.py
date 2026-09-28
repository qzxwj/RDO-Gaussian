#!/usr/bin/env python3
"""Generic vanilla-3DGS FPS-only (HAC++ protocol: sync + skip-5)."""
import os
import sys
import time
import csv
from argparse import ArgumentParser
from datetime import datetime

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from utils.general_utils import safe_state

# Prefer package-local GaussianModel / render
try:
    from gaussian_renderer import GaussianModel, render
except Exception:
    from scene import GaussianModel
    from gaussian_renderer import render


def measure_fps(views, gaussians, pipeline, background, warmup=5, render_kwargs=None):
    render_kwargs = render_kwargs or {}
    t_list = []
    with torch.no_grad():
        for view in tqdm(views, desc="FPS"):
            torch.cuda.synchronize()
            t_start = time.time()
            _ = render(view, gaussians, pipeline, background, **render_kwargs)
            torch.cuda.synchronize()
            t_end = time.time()
            t_list.append(t_end - t_start)
    assert len(t_list) > warmup, f"need >{warmup} views, got {len(t_list)}"
    fps = 1.0 / (sum(t_list[warmup:]) / len(t_list[warmup:]))
    return fps, len(t_list)


def append_csv(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "scene", "fps_raw", "fps_int", "n_views", "gpu", "model_path", "timestamp"],
        )
        if write_header:
            w.writeheader()
        w.writerow(row)


def fix_source_path(source_path):
    if source_path and os.path.exists(source_path):
        return os.path.abspath(source_path)
    if not source_path:
        return source_path
    for old in (
        "/mnt/dataset/lsh/xwj/gaussian-splatting/data",
        "/mnt/dataset/lsh/xwj/Compact-3DGS/data",
        "/mnt/dataset/lsh/xwj/LightGaussian/data",
        "/mnt/dataset/lsh/xwj/RDO-Gaussian/data",
        "/mnt/dataset/lsh/xwj/Self-Organizing-Gaussians/data",
        "/mnt/dataset/lsh/xwj/reduced-3dgs/data",
        "/mnt/dataset/lsh/xwj/data",
    ):
        if old in source_path:
            cand = source_path.replace(old, "data")
            if os.path.exists(cand):
                return os.path.abspath(cand)
            # also try shared data root
            cand2 = source_path.replace(old, "/home/lsh/Data/xwj/data")
            if os.path.exists(cand2):
                return cand2
    base = os.path.basename(source_path.rstrip("/"))
    for root in ("data/tandt", "data/blending", "data/mipnerf360", "/home/lsh/Data/xwj/data/tandt", "/home/lsh/Data/xwj/data/blending"):
        cand = os.path.join(root, base)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return source_path


def main():
    parser = ArgumentParser(description="Vanilla-GS FPS-only")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--scene_name", type=str, default="")
    parser.add_argument("--csv_out", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="")
    args = get_combined_args(parser)

    if getattr(args, "gpu", "") not in ("", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    safe_state(False)
    dataset = model.extract(args)
    dataset.source_path = fix_source_path(dataset.source_path)
    pipe = pipeline.extract(args)

    print(f"method={args.method_name} model_path={dataset.model_path}")
    print(f"source_path={dataset.source_path}")

    # Construct GaussianModel with best-effort signature compatibility
    sh = getattr(dataset, "sh_degree", 3)
    try:
        gaussians = GaussianModel(sh)
    except TypeError:
        gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_kwargs = {}
    # gaussian-splatting newer API
    if hasattr(dataset, "train_test_exp"):
        render_kwargs["use_trained_exp"] = dataset.train_test_exp
        try:
            from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
            render_kwargs["separate_sh"] = True
        except Exception:
            render_kwargs["separate_sh"] = False

    views = scene.getTestCameras()
    fps, n_views = measure_fps(views, gaussians, pipe, background, render_kwargs=render_kwargs)
    scene_name = args.scene_name or os.path.basename(dataset.source_path.rstrip("/"))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    row = {
        "method": args.method_name,
        "scene": scene_name,
        "fps_raw": f"{fps:.5f}",
        "fps_int": int(round(fps)),
        "n_views": n_views,
        "gpu": gpu,
        "model_path": args.model_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"Test FPS: \033[1;35m{fps:.5f}\033[0m  ({row['fps_int']} int)  n_views={n_views}")
    append_csv(args.csv_out, row)


if __name__ == "__main__":
    main()
