"""ॐ
DWT: Fast Multidimensional Discrete Wavelet Transform Layers (PyTorch).
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
from dwt.DWT2D import DWT2D, IDWT2D


def dwt2(x: torch.Tensor, level: int = 3, wave: str = 'haar') -> list[torch.Tensor]:
    """Multilevel 2D DWT.

    Args:
        x:     (batch, H, W, C)
        level: decomposition depth
        wave:  wavelet name

    Returns:
        list of length (level + 1): [H1, H2, ..., H_level, L_level]
        H_k subbands: (batch, H/2^k, W/2^k, C*3),  L_level: (batch, H/2^level, W/2^level, C).
    """
    analysis = DWT2D(wave=wave)
    subbands = []
    current = x
    channels_in = x.shape[-1]
    for _ in range(level):
        w = analysis(current)
        lowpass  = w[:, :, :, :channels_in]
        highpass = w[:, :, :, channels_in:]
        subbands.append(highpass)
        current = lowpass
    subbands.append(current)
    return subbands


def idwt2(subbands: list[torch.Tensor], wave: str = 'haar') -> torch.Tensor:
    """Multilevel 2D IDWT (inverse of dwt2).

    Args:
        subbands: list [H1, H2, ..., H_level, L_level]  (output of dwt2)
        wave:     wavelet name (must match dwt2 call)

    Returns:
        reconstructed tensor (batch, H, W, C)
    """
    synthesis = IDWT2D(wave=wave)
    *highpasses, current = subbands
    for H in reversed(highpasses):
        combined = torch.cat([current, H], dim=-1)
        current = synthesis(combined)
    return current


if __name__ == '__main__':
    x = torch.randn(1, 256, 256, 1)
    sb = dwt2(x, level=3)
    print([s.shape for s in sb])
    xhat = idwt2(sb)
    print('Recon error:', (x - xhat).abs().max().item())
