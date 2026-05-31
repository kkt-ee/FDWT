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
