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
import torch.nn.functional as F


def make_dwt_operator_matrix_A(h0, h1, N: int) -> torch.Tensor:
    """Build the (N x N) DWT operator matrix A from analysis filters h0, h1.

    Rows are circularly shifted and downsampled convolution rows.
    Upper half: lowpass (h0) rows.  Lower half: highpass (h1) rows.
    Synthesis operator S = A^T.
    """
    h0 = torch.tensor(h0, dtype=torch.float32)
    h1 = torch.tensor(h1, dtype=torch.float32)
    L = h0.shape[0]

    def _branch_row(h):
        return torch.cat([h, torch.zeros(N - L, dtype=h.dtype)])

    def _start_row(row):
        return torch.roll(row, shifts=-(L - 2))

    def _build_rows(row):
        return torch.stack([torch.roll(row, shifts=2 * k) for k in range(N // 2)], dim=0)

    H0 = _build_rows(_start_row(_branch_row(h0)))
    H1 = _build_rows(_start_row(_branch_row(h1)))
    return torch.cat([H0, H1], dim=0)  # (N, N)


def _normalize_axis(x: torch.Tensor, axis: int) -> int:
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f'axis={axis} is invalid for rank-{x.ndim} input.')
    return axis


def operator_matrix_axis(
    x: torch.Tensor,
    operator: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    """Apply a square linear operator along one tensor axis."""
    axis = _normalize_axis(x, axis)
    operator = torch.as_tensor(
        operator,
        dtype=x.dtype,
        device=x.device,
    )
    length = x.shape[axis]
    if operator.ndim != 2 or operator.shape != (length, length):
        raise ValueError(
            'The operator must be square and match the selected axis.'
        )

    if x.ndim == 1:
        return torch.einsum('ij,j->i', operator, x)

    moved = x.movedim(axis, -2)
    moved_shape = moved.shape
    fibres = moved.reshape(-1, length, moved_shape[-1])
    transformed = torch.einsum('ij,bjc->bic', operator, fibres)
    transformed = transformed.reshape(moved_shape)
    return transformed.movedim(-2, axis)


def _filter_tensors(x: torch.Tensor, h0, h1):
    h0 = torch.as_tensor(h0, dtype=x.dtype, device=x.device)
    h1 = torch.as_tensor(h1, dtype=x.dtype, device=x.device)
    filter_length = h0.numel()
    if filter_length < 2:
        raise ValueError('The wavelet-filter length must be at least two.')
    if h1.numel() != filter_length:
        raise ValueError('Lowpass and highpass filters must have equal length.')
    return h0, h1, filter_length


def analysis_filterbank_axis(
    x: torch.Tensor,
    h0,
    h1,
    axis: int,
) -> torch.Tensor:
    """Apply the packed periodic analysis bank along one tensor axis.

    This is the matrix-free equivalent of multiplying by the operator made by
    ``make_dwt_operator_matrix_A``. The selected axis retains its length and is
    packed as ``[lowpass | highpass]``.
    """
    axis = _normalize_axis(x, axis)
    h0, h1, filter_length = _filter_tensors(x, h0, h1)
    moved = x.movedim(axis, -1)
    moved_shape = moved.shape
    length = moved_shape[-1]
    if length % 2:
        raise ValueError('The transformed length must be even.')
    if length < filter_length:
        raise ValueError('The transformed length must cover the wavelet filter.')

    fibres = moved.reshape(-1, 1, length)
    pad_left = filter_length - 2
    if pad_left:
        fibres = torch.cat((fibres[..., -pad_left:], fibres), dim=-1)

    filters = torch.stack((h0, h1), dim=0).unsqueeze(1)
    subbands = F.conv1d(fibres, filters, stride=2)
    packed = torch.cat((subbands[:, 0, :], subbands[:, 1, :]), dim=-1)
    return packed.reshape(moved_shape).movedim(-1, axis)


def synthesis_filterbank_axis(
    x: torch.Tensor,
    g0,
    g1,
    axis: int,
) -> torch.Tensor:
    """Apply the packed periodic synthesis bank along one tensor axis.

    This is the matrix-free equivalent of multiplying by the transpose of an
    operator made from the synthesis filters. The selected input axis must be
    packed as ``[lowpass | highpass]``.
    """
    axis = _normalize_axis(x, axis)
    g0, g1, filter_length = _filter_tensors(x, g0, g1)
    moved = x.movedim(axis, -1)
    moved_shape = moved.shape
    length = moved_shape[-1]
    if length % 2:
        raise ValueError('The transformed length must be even.')
    if length < filter_length:
        raise ValueError('The transformed length must cover the wavelet filter.')

    fibres = moved.reshape(-1, length)
    half = length // 2
    coefficients = torch.stack(
        (fibres[:, :half], fibres[:, half:]),
        dim=1,
    )
    filters = torch.stack((g0, g1), dim=0).unsqueeze(1)
    padded = F.conv_transpose1d(coefficients, filters, stride=2)

    pad_left = filter_length - 2
    natural = padded[..., pad_left:]
    if pad_left:
        natural = torch.cat(
            (
                natural[..., :length - pad_left],
                natural[..., length - pad_left:] + padded[..., :pad_left],
            ),
            dim=-1,
        )

    restored = natural.squeeze(1).reshape(moved_shape)
    return restored.movedim(-1, axis)
