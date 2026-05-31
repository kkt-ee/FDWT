"""ॐ
FDWT: Fast Multidimensional Discrete Wavelet Transform Layers (PyTorch).
Copyright 2026 Kishore Kumar Tarafdar

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import torch
from dwt.DWT3D import DWT3D, IDWT3D


def dwt3(x: torch.Tensor, level: int = 3, wave: str = 'haar') -> list[torch.Tensor]:
    """Multilevel 3D DWT.

    Args:
        x:     (batch, D, H, W, C)
        level: decomposition depth
        wave:  wavelet name

    Returns:
        list of length (level + 1): [H1, H2, ..., H_level, L_level]
        H_k subbands: (batch, D/2^k, H/2^k, W/2^k, C*7),  L_level: (..., C).
    """
    analysis = DWT3D(wave=wave)
    subbands = []
    current = x
    channels_in = x.shape[-1]
    for _ in range(level):
        w = analysis(current)
        lowpass  = w[:, :, :, :, :channels_in]
        highpass = w[:, :, :, :, channels_in:]
        subbands.append(highpass)
        current = lowpass
    subbands.append(current)
    return subbands


def idwt3(subbands: list[torch.Tensor], wave: str = 'haar') -> torch.Tensor:
    """Multilevel 3D IDWT (inverse of dwt3).

    Args:
        subbands: list [H1, H2, ..., H_level, L_level]  (output of dwt3)
        wave:     wavelet name (must match dwt3 call)

    Returns:
        reconstructed tensor (batch, D, H, W, C)
    """
    synthesis = IDWT3D(wave=wave)
    *highpasses, current = subbands
    for H in reversed(highpasses):
        combined = torch.cat([current, H], dim=-1)
        current = synthesis(combined)
    return current


if __name__ == '__main__':
    x = torch.randn(1, 32, 32, 32, 2)
    sb = dwt3(x, level=2)
    print([s.shape for s in sb])
    xhat = idwt3(sb)
    print('Recon error:', (x - xhat).abs().max().item())
