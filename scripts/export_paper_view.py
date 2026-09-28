#!/usr/bin/env python3
"""Export one named COLMAP view from an RDO-Gaussian bitstream.

Writes render.npy (float RGB), xyz.npy (Gaussian means), and meta.json
for change-splatting/scripts/visualize_error_and_centers.py --precomputed.

Loads the compressed bitstream via Scene.decode, not a vanilla ply render.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _limit_cpu_threads():
    n = max(1, int(os.environ.get("CPU_THREADS", "4")))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = str(n)
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["CPU_THREADS"] = str(n)
    return n


def _bind_gpu():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gpu", type=int, default=3)
    args, _ = pre.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _limit_cpu_threads()
    return args.gpu


if __name__ == "__main__":
    _bind_gpu()

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("CPU_THREADS", "4")))
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state


def find_camera(scene, image_name):
    name = image_name.replace(".jpg", "").replace(".png", "")
    cams = list(scene.getTestCameras()) + list(scene.getTrainCameras())
    match = [c for c in cams if c.image_name == name or c.image_name.startswith(name)]
    if not match:
        sample = ", ".join(c.image_name for c in cams[:8])
        raise ValueError(
            "image_name {} not found. First names: {}".format(image_name, sample)
        )
    return match[0]


def psnr_np(rendered, gt):
    mse = np.mean((rendered.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return float(-10.0 * np.log10(mse + 1e-16))


def parse_args():
    parser = argparse.ArgumentParser(description="Export one RDO-Gaussian view for paper figures")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--image_name", default="00001", type=str)
    parser.add_argument(
        "--out",
        type=str,
        default="/home/lsh/Data/xwj/change-splatting/results/paper_viz/rdo_gaussian_train_00001",
    )
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    return args, model, pipeline


def main():
    args, model_cfg, pipe_cfg = parse_args()
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES")))
    print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    safe_state(getattr(args, "quiet", False))

    dataset = model_cfg.extract(args)
    pipeline = pipe_cfg.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cam = find_camera(scene, args.image_name)
    print(
        "[cam] {}  {}x{}  FoV={:.3f}x{:.3f}".format(
            cam.image_name, cam.image_width, cam.image_height, cam.FoVx, cam.FoVy
        )
    )

    bg = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    with torch.no_grad():
        rendered = render(cam, gaussians, pipeline, bg)["render"].detach().clamp(0.0, 1.0)
    rendered_np = rendered.cpu().numpy().transpose(1, 2, 0)
    gt = cam.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    gt = np.clip(gt, 0.0, 1.0)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    psnr = psnr_np(rendered_np, gt)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "render.npy"), rendered_np.astype(np.float32))
    np.save(os.path.join(args.out, "xyz.npy"), xyz)
    meta = {
        "image_name": cam.image_name,
        "width": int(cam.image_width),
        "height": int(cam.image_height),
        "psnr": psnr,
        "n_gaussians": int(xyz.shape[0]),
        "model_path": os.path.abspath(args.model_path),
        "iteration": int(scene.loaded_iter),
        "source_path": dataset.source_path,
        "bitstream": os.path.join(os.path.abspath(args.model_path), "bitstream"),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(
        "[export] PSNR={:.2f} dB  N={:,}  -> {}".format(
            psnr, xyz.shape[0], args.out
        )
    )


if __name__ == "__main__":
    main()
