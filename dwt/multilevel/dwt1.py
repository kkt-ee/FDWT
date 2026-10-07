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
from dwt.DWT1D import DWT1D, IDWT1D
from dwt.dwt_op import (
    analysis_filterbank_axis,
    make_dwt_operator_matrix_A,
    operator_matrix_axis,
    synthesis_filterbank_axis,
)
from dwt.filters import FetchAnalysisSynthesisFilters


_VALID_BACKENDS = ('matrix', 'filterbank')


def _validate_backend(backend: str) -> str:
    if backend not in _VALID_BACKENDS:
        choices = ', '.join(repr(value) for value in _VALID_BACKENDS)
        raise ValueError(f'backend must be one of: {choices}.')
    return backend


def _validate_level(level: int) -> int:
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError('level must be a non-negative integer.')
    return level


def _normalize_axis(x: torch.Tensor, axis: int) -> int:
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f'axis={axis} is invalid for rank-{x.ndim} input.')
    return axis


def _validate_packed_shape(
    x: torch.Tensor,
    level: int,
    filter_length: int,
    axis: int,
) -> tuple[int, int]:
    axis = _normalize_axis(x, axis)
    length = x.shape[axis]
    divisor = 2 ** level
    if length % divisor:
        raise ValueError(
            f'The transformed length must be divisible by 2**level={divisor}.'
        )
    if level and length // (2 ** (level - 1)) < filter_length:
        raise ValueError(
            'Every decomposition level must cover the wavelet-filter length.'
        )
    return axis, length


def _wavelet_filters(wave: str):
    filters = FetchAnalysisSynthesisFilters(wave)
    analysis = filters.analysis()
    synthesis = (
        filters.synthesis()
        if 'bior' in wave or 'rbio' in wave
        else analysis
    )
    return analysis, synthesis


def _analysis_axis(x, filters, axis, backend):
    if backend == 'filterbank':
        return analysis_filterbank_axis(x, *filters, axis=axis)
    operator = make_dwt_operator_matrix_A(*filters, x.shape[axis])
    return operator_matrix_axis(x, operator, axis=axis)


def _synthesis_axis(x, filters, axis, backend):
    if backend == 'filterbank':
        return synthesis_filterbank_axis(x, *filters, axis=axis)
    operator = make_dwt_operator_matrix_A(*filters, x.shape[axis]).T
    return operator_matrix_axis(x, operator, axis=axis)


def dwt(
    x: torch.Tensor,
    level: int = 3,
    wave: str = 'haar',
    backend: str = 'matrix',
) -> list[torch.Tensor]:
    """Multilevel 1D DWT.

    Args:
        x:     (batch, N, C)
        level: decomposition depth
        wave:  wavelet name
        backend: ``'matrix'`` or ``'filterbank'``

    Returns:
        list of length (level + 1): [H1, H2, ..., H_level, L_level]
        H_k has shape (batch, N/2^k, C),  L_level has shape (batch, N/2^level, C).
    """
    level = _validate_level(level)
    backend = _validate_backend(backend)
    analysis = DWT1D(wave=wave, backend=backend)
    subbands = []
    current = x
    channels_in = x.shape[-1]
    for _ in range(level):
        w = analysis(current)
        lowpass  = w[:, :, :channels_in]
        highpass = w[:, :, channels_in:]
        subbands.append(highpass)
        current = lowpass
    subbands.append(current)
    return subbands


def idwt(
    subbands: list[torch.Tensor],
    wave: str = 'haar',
    level: int | None = None,
    backend: str = 'matrix',
) -> torch.Tensor:
    """Multilevel 1D IDWT (inverse of dwt).

    Args:
        subbands: list [H1, H2, ..., H_level, L_level]  (output of dwt)
        wave:     wavelet name (must match dwt call)
        level:    decomposition depth; inferred when omitted
        backend:  ``'matrix'`` or ``'filterbank'``

    Returns:
        reconstructed tensor (batch, N, C)
    """
    if level is None:
        level = len(subbands) - 1
    level = _validate_level(level)
    backend = _validate_backend(backend)
    if len(subbands) != level + 1:
        raise ValueError(
            f'level={level} requires {level + 1} subbands, '
            f'but received {len(subbands)}.'
        )

    synthesis = IDWT1D(wave=wave, backend=backend)
    *highpasses, current = subbands
    for H in reversed(highpasses):
        combined = torch.cat([current, H], dim=-1)
        current = synthesis(combined)
    return current


def dwt_packed_axis(
    x: torch.Tensor,
    level: int = 3,
    wave: str = 'haar',
    axis: int = 1,
    backend: str = 'matrix',
) -> torch.Tensor:
    """Packed multilevel 1D DWT along an arbitrary tensor axis.

    The selected axis retains its length and is ordered as
    ``[L_level, H_level, H_(level-1), ..., H_1]``.
    """
    level = _validate_level(level)
    backend = _validate_backend(backend)
    analysis, _ = _wavelet_filters(wave)
    axis, _ = _validate_packed_shape(
        x,
        level,
        len(analysis[0]),
        axis,
    )
    if level == 0:
        return x

    highpasses = []
    current = x
    for _ in range(level):
        packed = _analysis_axis(current, analysis, axis, backend)
        current, highpass = torch.chunk(packed, 2, dim=axis)
        highpasses.append(highpass)
    return torch.cat([current] + list(reversed(highpasses)), dim=axis)


def idwt_packed_axis(
    x: torch.Tensor,
    level: int = 3,
    wave: str = 'haar',
    axis: int = 1,
    backend: str = 'matrix',
) -> torch.Tensor:
    """Inverse of :func:`dwt_packed_axis`."""
    level = _validate_level(level)
    backend = _validate_backend(backend)
    _, synthesis = _wavelet_filters(wave)
    axis, length = _validate_packed_shape(
        x,
        level,
        len(synthesis[0]),
        axis,
    )
    if level == 0:
        return x

    lowest_length = length // (2 ** level)
    sizes = [lowest_length, lowest_length]
    sizes.extend(
        lowest_length * (2 ** exponent)
        for exponent in range(1, level)
    )
    lowpass, *highpasses = torch.split(x, sizes, dim=axis)
    current = lowpass
    for highpass in highpasses:
        packed = torch.cat([current, highpass], dim=axis)
        current = _synthesis_axis(
            packed,
            synthesis,
            axis,
            backend,
        )
    return current


if __name__ == '__main__':
    x = torch.randn(1, 256, 2)
    sb = dwt(x, level=4)
    print([s.shape for s in sb])
    xhat = idwt(sb)
    print('Recon error:', (x - xhat).abs().max().item())
