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
from dwt.layout import DWTNDlayout, IDWTNDlayout


class DWT1D(DWTNDlayout):
    """1D DWT analysis module.

    clean=True:  (batch, N, C) -> (batch, N/2, C*2)   [L||H packed along channel axis]
    clean=False: (batch, N, C) -> (batch, N, C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[1]
        A = self._get_A(N, x.device)
        out = torch.einsum('ij,bjc->bic', A, x)
        if self.clean:
            return self._extract_2subbands(out)
        return out

    def _extract_2subbands(self, x: torch.Tensor) -> torch.Tensor:
        mid = x.shape[1] // 2
        L = x[:, :mid, :]
        H = x[:, mid:, :]
        return torch.cat([L, H], dim=-1)


class IDWT1D(IDWTNDlayout):
    """1D IDWT synthesis module.

    clean=True:  (batch, N/2, C*2) -> (batch, N, C)
    clean=False: (batch, N, C)     -> (batch, N, C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.clean:
            x = self._join_2subbands(x)
        N = x.shape[1]
        S = self._get_S(N, x.device)
        return torch.einsum('ij,bjc->bic', S, x)

    def _join_2subbands(self, x: torch.Tensor) -> torch.Tensor:
        L, H = torch.chunk(x, 2, dim=-1)
        return torch.cat([L, H], dim=1)


if __name__ == '__main__':
    wave = 'haar'
    dwt  = DWT1D(wave)
    idwt = IDWT1D(wave)
    x = torch.randn(2, 256, 1)
    lh = dwt(x)
    xhat = idwt(lh)
    print('DWT output:', lh.shape)
    print('Reconstruction error (max):', (x - xhat).abs().max().item())
