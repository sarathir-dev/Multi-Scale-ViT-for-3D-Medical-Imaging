# Helper functions

import torch
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_3d(patches, temperature=10000, dtype=torch.float32):
    _, f, h, w, dim, device = *patches.shape, patches.device
    z, y, x = torch.meshgrid(
        torch.arange(f, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )

    fourier_dim = dim // 6
    omega = torch.arange(fourier_dim, device=device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(),
                   y.cos(), z.sin(), z.cos()), dim=1)
    pe = F.pad(pe, (0, dim - (fourier_dim * 6)))
    return pe.type(dtype)
