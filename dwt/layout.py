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
import torch.nn as nn

from dwt.filters import FetchAnalysisSynthesisFilters
from dwt.dwt_op import make_dwt_operator_matrix_A


class DWTNDlayout(nn.Module):
    """Base analysis module.  Builds A lazily on first forward call (N unknown at init).

    Tensor layout: channels-last throughout.
      1D: (batch, N, C)
      2D: (batch, H, W, C)
      3D: (batch, D, H, W, C)
    """

    def __init__(self, wave: str = 'haar', clean: bool = True):
        super().__init__()
        self.wave = wave
        self.clean = clean
        w = FetchAnalysisSynthesisFilters(wave)
        self.h0, self.h1 = w.analysis()
        self._A_cache: torch.Tensor | None = None
        self._A_N: int = -1

    def _get_A(self, N: int, device: torch.device) -> torch.Tensor:
        if self._A_N != N or self._A_cache is None:
            self._A_cache = make_dwt_operator_matrix_A(self.h0, self.h1, N).to(device)
            self._A_N = N
        elif self._A_cache.device != device:
            self._A_cache = self._A_cache.to(device)
        return self._A_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IDWTNDlayout(nn.Module):
    """Base synthesis module.  Builds S = A^T lazily on first forward call.

    For biorthogonal wavelets (bior/rbio) synthesis filters differ from analysis.
    """

    def __init__(self, wave: str = 'haar', clean: bool = True):
        super().__init__()
        self.wave = wave
        self.clean = clean
        w = FetchAnalysisSynthesisFilters(wave)
        if 'bior' in wave or 'rbio' in wave:
            self.h0, self.h1 = w.synthesis()
        else:
            self.h0, self.h1 = w.analysis()
        self._S_cache: torch.Tensor | None = None
        self._S_N: int = -1

    def _get_S(self, N: int, device: torch.device) -> torch.Tensor:
        if self._S_N != N or self._S_cache is None:
            A = make_dwt_operator_matrix_A(self.h0, self.h1, N)
            self._S_cache = A.T.contiguous().to(device)
            self._S_N = N
        elif self._S_cache.device != device:
            self._S_cache = self._S_cache.to(device)
        return self._S_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
