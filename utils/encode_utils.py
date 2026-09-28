from argparse import ArgumentParser
import io
import os
import struct
import zipfile
import numpy as np
import torch
from utils.entropy_model import DiscreteUnconditionalEntropyModel, Softmax

def remove_negatives(codebook):
    mask = codebook < 0
    codebook[mask] = 1e-8

def pack_string(string):
    bit_stream = struct.pack(f'>I', len(string))
    bit_stream += struct.pack(f'>{len(string)}s', string)
    return bit_stream

def unpack_string(bit_stream):
    s1 = struct.calcsize('I')
    s2 = struct.calcsize('s')
    length = struct.unpack(f'>I', bit_stream[:s1])[0]
    string = struct.unpack(f'>{length}s', bit_stream[s1:s1+s2*length])[0]
    return string, bit_stream[s1+s2*length:]

def pack_strings(strings):
    bit_stream = b''
    for string in strings:
        bit_stream += pack_string(string)
    return bit_stream

def unpack_strings(bit_stream, n):
    strings = []
    for i in range(n):
        string, bit_stream = unpack_string(bit_stream)
        strings.append(string)
    return strings, bit_stream

def pack_uints(uints):
    bit_stream = struct.pack(f'>{len(uints)}I', *uints)
    return bit_stream

def unpack_uints(bit_stream, n):
    s1 = struct.calcsize('I')
    uints = struct.unpack(f'>{n}I', bit_stream[:n*s1])
    return uints, bit_stream[n*s1:]

def pack_floats(floats):
    bit_stream = struct.pack(f'>{len(floats)}f', *floats)
    return bit_stream

def unpack_floats(bit_stream, n):
    s1 = struct.calcsize('f')
    floats = struct.unpack(f'>{n}f', bit_stream[:n*s1])
    return floats, bit_stream[n*s1:]


def shmask_sort(sh_bitmask:dict):
    shmask_sorted = np.zeros_like(sh_bitmask['sh1'], dtype=int)

    for key in sh_bitmask:
        shmask_sorted = (shmask_sorted << 1) | sh_bitmask[key].astype(int)
        
    sort_idx = np.argsort(shmask_sorted)
    num_gaussians = shmask_sorted.shape[0]
    shmask_sorted = shmask_sorted[sort_idx]
    boundaries = np.where(np.diff(shmask_sorted))[0] + 1
    boundaries = np.append(boundaries, num_gaussians)
    
    shmask_sorted = np.insert(shmask_sorted, 0, 0)
    shmask_sorted = np.append(shmask_sorted, 7)
    diff_array = np.diff(shmask_sorted)
    indices = np.flatnonzero(diff_array)
    boundaries = []

    for i in indices:
        value = diff_array[i]
        boundaries.extend([i] * value)
        
    boundaries = np.array(boundaries)
    if boundaries[-1] != shmask_sorted.shape[0]:
        boundaries = np.append(boundaries, num_gaussians)
                    
    return sort_idx, boundaries


def _part1by2_10bit(x):
    """Expand 10 bits so three coordinates can be interleaved."""
    x = x.astype(np.uint32, copy=False) & np.uint32(0x3FF)
    x = (x | (x << np.uint32(16))) & np.uint32(0x030000FF)
    x = (x | (x << np.uint32(8))) & np.uint32(0x0300F00F)
    x = (x | (x << np.uint32(4))) & np.uint32(0x030C30C3)
    x = (x | (x << np.uint32(2))) & np.uint32(0x09249249)
    return x


def spatial_sort_within_boundaries(xyz, sort_idx, boundaries):
    """Morton-sort each SH-mask group without changing group boundaries."""
    xyz32 = np.asarray(xyz, dtype=np.float32)
    xyz_min = xyz32.min(axis=0)
    xyz_max = xyz32.max(axis=0)
    quant = np.clip(
        np.rint((xyz32 - xyz_min) / (xyz_max - xyz_min + 1e-12) * 1023.0),
        0,
        1023,
    ).astype(np.uint32)
    morton = (
        _part1by2_10bit(quant[:, 0])
        | (_part1by2_10bit(quant[:, 1]) << np.uint32(1))
        | (_part1by2_10bit(quant[:, 2]) << np.uint32(2))
    )

    sorted_idx = np.asarray(sort_idx)
    local_order = []
    start = 0
    for end in boundaries:
        end = int(end)
        segment = np.arange(start, end)
        original_idx = sorted_idx[segment]
        local_order.append(segment[np.argsort(morton[original_idx], kind="stable")])
        start = end
    if not local_order:
        return sorted_idx
    return sorted_idx[np.concatenate(local_order)]


def save_positions_lossless(path, xyz):
    """Losslessly delta-code spatially sorted float16 positions."""
    xyz16 = np.ascontiguousarray(xyz, dtype=np.float16)
    bits = xyz16.view(np.uint16)
    ordered = np.where(
        (bits & np.uint16(0x8000)) != 0,
        np.bitwise_not(bits),
        bits ^ np.uint16(0x8000),
    ).astype(np.uint16)
    delta = np.empty_like(ordered)
    delta[0] = ordered[0]
    delta[1:] = (
        ordered[1:].astype(np.uint32) - ordered[:-1].astype(np.uint32)
    ).astype(np.uint16)
    # Coordinate-major layout gives the compressor longer homogeneous runs.
    # The LZMA-backed NPZ remains directly readable by numpy.load.
    save_npz_lzma(
        path,
        position_delta_flat_f=delta.reshape(-1, order="F"),
    )


def save_npz_lzma(path, **arrays):
    """Write a numpy-compatible NPZ using ZIP-LZMA compression."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_LZMA) as archive:
        for key, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asanyarray(value),
                allow_pickle=False,
            )
            archive.writestr(f"{key}.npy", buffer.getvalue())


def load_positions(path):
    """Load the new lossless position stream or a legacy position array."""
    payload = np.load(path)
    if "position" in payload.files:
        return payload["position"]
    if "position_delta_flat_f" in payload.files:
        flat = payload["position_delta_flat_f"]
        if flat.size % 3 != 0:
            raise ValueError(f"Invalid position delta length in {path}: {flat.size}")
        delta = flat.reshape((-1, 3), order="F")
    else:
        delta = payload["position_delta"]
    delta = delta.astype(np.uint64)
    ordered = (np.cumsum(delta, axis=0) & np.uint64(0xFFFF)).astype(np.uint16)
    bits = np.where(
        (ordered & np.uint16(0x8000)) != 0,
        ordered ^ np.uint16(0x8000),
        np.bitwise_not(ordered),
    ).astype(np.uint16)
    return np.ascontiguousarray(bits).view(np.float16)

def entropy_coding(attr_index, attr_logits, range_coder_precision=16):
    em = DiscreteUnconditionalEntropyModel(Softmax(attr_logits), range_coder_precision)
    em.get_ready_for_compression()
    bits_str = em.compress(attr_index)
    decode_results = em.decompress(bits_str, attr_index.shape)
    return bits_str

def entropy_decoding(bits_str, attr_index_shape, attr_logits, range_coder_precision=16):
    em = DiscreteUnconditionalEntropyModel(Softmax(attr_logits), range_coder_precision)
    em.get_ready_for_compression()
    decode_results = em.decompress(bits_str, attr_index_shape)
    return decode_results

def shrink_codebook(codebook_dict: dict, logits_dict: dict, index_dict: dict):
    for vq_key in codebook_dict.keys():
        codebook = codebook_dict[vq_key]
        logits = logits_dict[vq_key]
        index_vector = index_dict[vq_key].squeeze()

        used_indices = np.unique(index_vector)
        print(vq_key, 'num of used codewords', len(used_indices))

        new_codebook = codebook[:, used_indices]
        new_logits = logits[:, used_indices]
        new_index_vector = np.expand_dims(np.searchsorted(used_indices, index_vector), 1)
        
        codebook_dict[vq_key] = new_codebook
        logits_dict[vq_key] = new_logits
        index_dict[vq_key] = new_index_vector
        
    return codebook_dict, logits_dict, index_dict

def opacity_quant(opacities, precision=2**8):
    min_opacity = np.min(opacities)
    max_opacity = np.max(opacities)
    step_size = (max_opacity - min_opacity) / (precision - 1)
    
    quantized_opacities = np.round((opacities - min_opacity) / step_size).astype(np.uint8)
    dequantized_opacities = quantized_opacities * step_size + min_opacity
    
    hist, _ = np.histogram(quantized_opacities, bins=precision, density=True)
    log_prob = np.expand_dims(np.log(hist + 1e-10), 0)
    
    return quantized_opacities, dequantized_opacities, log_prob, step_size, min_opacity

def opacity_dequant(quantized_opacities, min_opacity, step_size):    
    dequantized_opacities = quantized_opacities * step_size + min_opacity
    return dequantized_opacities

def decode_header(header_path):
    with open(header_path, 'rb') as f:
        header_bitstream = f.read()
    if len(header_bitstream) == 12:
        boundaries, header_bitstream = unpack_uints(header_bitstream, 1)
    else:
        boundaries, header_bitstream = unpack_uints(header_bitstream, 8)
    step_size, header_bitstream = unpack_floats(header_bitstream, 1)
    min_opacity, header_bitstream = unpack_floats(header_bitstream, 1)
    boundaries = np.array(boundaries, dtype=np.int32)
    step_size = step_size[0]
    min_opacity = min_opacity[0]
    
    return boundaries, step_size, min_opacity

def get_index_shapes(boundaries):
    dec_num_gaussians = boundaries[-1]
    if len(boundaries) == 1:
        return {
            'scale': (dec_num_gaussians, ),
            'rot': (dec_num_gaussians, ),
            'dc': (dec_num_gaussians, ),
            'sh1': (dec_num_gaussians, ),
            'sh2': (dec_num_gaussians, ),
            'sh3': (dec_num_gaussians, ),
            'opa': (dec_num_gaussians, )
        }
    dec_sh1_len = boundaries[7] - boundaries[3]
    dec_sh2_len = boundaries[7] - boundaries[5] + boundaries[3] - boundaries[1]
    dec_sh3_len = boundaries[7] - boundaries[6] + boundaries[5] - boundaries[4] \
        + boundaries[3] - boundaries[2] + boundaries[1] - boundaries[0]

    return {
        'scale': (dec_num_gaussians, ),
        'rot': (dec_num_gaussians, ),
        'dc': (dec_num_gaussians, ),
        'sh1': (dec_sh1_len, ),
        'sh2': (dec_sh2_len, ),
        'sh3': (dec_sh3_len, ),
        'opa': (dec_num_gaussians, )
    }
    
def decode_indexes(index_path, logits_dict, shape_dict):
    attr_keys = ['scale', 'rot', 'dc', 'sh1', 'sh2', 'sh3', 'opa']
    with open(index_path, "rb") as f:
        index_bitstream = f.read()
        dec_indexes = {}
        for i, attr_key in enumerate(attr_keys):
            dec_indexes_string, index_bitstream = unpack_string(index_bitstream)
            dec_indexes[attr_key] = entropy_decoding(dec_indexes_string, shape_dict[attr_key], torch.Tensor(logits_dict[attr_key]))
                
    return dec_indexes

def get_sh_bitmask(boundaries):
    num_gaussians = boundaries[-1]
    if len(boundaries) == 1:
        return {sh_key: np.ones((num_gaussians, ), dtype=np.bool_) for sh_key in ['sh1', 'sh2', 'sh3']}
    shmask_sorted = np.zeros(num_gaussians, dtype=int)
    for i in range(7, -1, -1):
        shmask_sorted[:boundaries[i]] = i
    
    sh_bitmask = {sh_key: np.zeros(num_gaussians, dtype=bool) for sh_key in ['sh1', 'sh2', 'sh3']}

    sh_bitmask['sh1'] |= (shmask_sorted & (1 << 2)).astype(bool)
    sh_bitmask['sh2'] |= (shmask_sorted & (1 << 1)).astype(bool)
    sh_bitmask['sh3'] |= (shmask_sorted & (1 << 0)).astype(bool)
    
    return sh_bitmask
