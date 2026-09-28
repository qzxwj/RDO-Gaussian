#!/usr/bin/env python3
"""Write or print bitstream/codec_stats.json for a trained run."""
import argparse
import json
import os
import sys

ATTR_KEYS = ["scale", "rot", "dc", "sh1", "sh2", "sh3", "opa"]
FILE_NAMES = ["index_bitstream.bin", "header.bin", "codebook.npz", "logits.npz", "position.npz"]


def collect_file_bytes(bitstream_dir):
    file_bytes = {}
    for fname in FILE_NAMES:
        fpath = os.path.join(bitstream_dir, fname)
        file_bytes[fname] = int(os.path.getsize(fpath)) if os.path.isfile(fpath) else 0
    return file_bytes


def collect_index_bytes(bitstream_dir):
    index_path = os.path.join(bitstream_dir, "index_bitstream.bin")
    if not os.path.isfile(index_path):
        return {}
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.encode_utils import unpack_strings
    with open(index_path, "rb") as f:
        data = f.read()
    strings, _ = unpack_strings(data, len(ATTR_KEYS))
    return {k: int(len(s)) for k, s in zip(ATTR_KEYS, strings)}


def collect_used_codewords(bitstream_dir):
    codebook_path = os.path.join(bitstream_dir, "codebook.npz")
    if not os.path.isfile(codebook_path):
        return {}
    import numpy as np
    codebooks = np.load(codebook_path)
    used = {}
    for key in codebooks.files:
        cb = codebooks[key]
        used[key] = int(cb.shape[1])
    return used


def build_stats(bitstream_dir):
    file_bytes = collect_file_bytes(bitstream_dir)
    stats = {
        "index_bytes": collect_index_bytes(bitstream_dir),
        "used_codewords": collect_used_codewords(bitstream_dir),
        "file_bytes": file_bytes,
        "total_bytes": int(sum(file_bytes.values())),
    }
    pos_path = os.path.join(bitstream_dir, "position.npz")
    if os.path.isfile(pos_path):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.encode_utils import load_positions
        stats["num_gaussians"] = int(load_positions(pos_path).shape[0])
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="model directory that contains bitstream/")
    parser.add_argument("--write", action="store_true", help="write bitstream/codec_stats.json")
    args = parser.parse_args()
    bitstream_dir = os.path.join(args.run_dir, "bitstream")
    if not os.path.isdir(bitstream_dir):
        print(f"missing bitstream: {bitstream_dir}")
        sys.exit(1)
    stats = build_stats(bitstream_dir)
    text = json.dumps(stats, indent=2)
    print(text)
    if args.write:
        out = os.path.join(bitstream_dir, "codec_stats.json")
        with open(out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
