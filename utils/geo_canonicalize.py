import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation

_EIGH_CHUNK = 16384


def _eigh_one_chunk(chunk: torch.Tensor):
    try:
        return torch.linalg.eigh(chunk)
    except RuntimeError:
        if chunk.shape[0] <= 256:
            w, v = torch.linalg.eigh(chunk.cpu())
            return w.to(device=chunk.device), v.to(device=chunk.device)
        mid = chunk.shape[0] // 2
        w0, v0 = _eigh_one_chunk(chunk[:mid])
        w1, v1 = _eigh_one_chunk(chunk[mid:])
        return torch.cat([w0, w1], dim=0), torch.cat([v0, v1], dim=0)


def _batched_eigh(sigma: torch.Tensor):
    """Chunked eigh to avoid cusolver failures on very large batches."""
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    eye = torch.eye(3, device=sigma.device, dtype=sigma.dtype)
    sigma = sigma + (1e-12 * eye)
    n = sigma.shape[0]
    if n <= _EIGH_CHUNK:
        return _eigh_one_chunk(sigma)
    evals = []
    evecs = []
    for start in range(0, n, _EIGH_CHUNK):
        w, v = _eigh_one_chunk(sigma[start : start + _EIGH_CHUNK])
        evals.append(w)
        evecs.append(v)
    return torch.cat(evals, dim=0), torch.cat(evecs, dim=0)


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 3) rotation matrices to (w, x, y, z) quaternions."""
    m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]

    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1), min=0.0))

    quat_by_rijk = torch.stack([
        torch.stack([q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[:, 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[:, 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[:, 3] ** 2], dim=-1),
    ], dim=-2)

    flr = torch.tensor(0.1, device=matrix.device, dtype=matrix.dtype)
    quat_candidates = quat_by_rijk / (2.0 * q_abs.unsqueeze(-1).clamp(min=flr))
    idx = q_abs.argmax(dim=-1)
    one_hot = F.one_hot(idx, num_classes=4).to(dtype=matrix.dtype)
    quat = (quat_candidates * one_hot.unsqueeze(-1)).sum(dim=-2)
    return F.normalize(quat, dim=-1)


@torch.no_grad()
def canonicalize_scale_rotation(scaling_log: torch.Tensor, rotation: torch.Tensor, aniso_ratio: float = 1.05):
    """Map (log-scale, raw quaternion) to a unique (s, q) with sx>=sy>=sz and w>=0."""
    s = torch.exp(scaling_log)
    q = F.normalize(rotation, dim=-1)
    frac_hemisphere = float((q[:, 0] < 0).float().mean().item())

    rot_mat = build_rotation(q)
    sigma = rot_mat @ torch.diag_embed(s * s) @ rot_mat.transpose(-1, -2)
    evals, evecs = _batched_eigh(sigma)
    evals = evals.flip(-1).clamp_min(1e-16)
    evecs = evecs.flip(-1)

    det = torch.det(evecs)
    evecs = evecs.clone()
    evecs[det < 0, :, 2] *= -1

    s_canon = evals.sqrt()
    aniso = s_canon[:, 0] / s_canon[:, 2].clamp_min(1e-8)
    skip_axis = aniso < aniso_ratio
    frac_skipped_iso = float(skip_axis.float().mean().item())

    q_from_r = rotation_matrix_to_quaternion(evecs)
    s_out = torch.where(skip_axis.unsqueeze(-1), s, s_canon)
    q_out = torch.where(skip_axis.unsqueeze(-1), q, q_from_r)
    q_out = torch.where(q_out[:, :1] < 0, -q_out, q_out)

    orig_sorted = (s[:, 0] >= s[:, 1]) & (s[:, 1] >= s[:, 2])
    frac_axis_perm = float(((~skip_axis) & (~orig_sorted)).float().mean().item())

    scaling_log_out = torch.log(s_out.clamp_min(1e-16))
    stats = {
        "frac_axis_perm": frac_axis_perm,
        "frac_hemisphere": frac_hemisphere,
        "frac_skipped_iso": frac_skipped_iso,
    }
    return scaling_log_out, q_out, stats
