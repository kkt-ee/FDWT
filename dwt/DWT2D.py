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


class DWT2D(DWTNDlayout):
    """2D DWT analysis module.

    clean=True:  (batch, H, W, C) -> (batch, H/2, W/2, C*4)   [LL|LH|HL|HH along channel]
    clean=False: (batch, H, W, C) -> (batch, H, W, C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[1]
        A = self._get_A(N, x.device)
        # columns (swap H and W, apply A along H, swap back)
        x = x.permute(0, 2, 1, 3)
        x = torch.einsum('ij,bjkc->bikc', A, x)
        x = x.permute(0, 2, 1, 3)
        # rows
        x = torch.einsum('ij,bjkc->bikc', A, x)
        if self.clean:
            return self._extract_4subbands(x)
        return x

    def _extract_4subbands(self, x: torch.Tensor) -> torch.Tensor:
        mid = x.shape[1] // 2
        LL = x[:, :mid, :mid, :]
        LH = x[:, mid:, :mid, :]
        HL = x[:, :mid, mid:, :]
        HH = x[:, mid:, mid:, :]
        return torch.cat([LL, LH, HL, HH], dim=-1)


class IDWT2D(IDWTNDlayout):
    """2D IDWT synthesis module.

    clean=True:  (batch, H/2, W/2, C*4) -> (batch, H, W, C)
    clean=False: (batch, H, W, C)        -> (batch, H, W, C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.clean:
            x = self._join_quadrants(x)
        N = x.shape[1]
        S = self._get_S(N, x.device)
        # columns
        x = x.permute(0, 2, 1, 3)
        x = torch.einsum('ij,bjkc->bikc', S, x)
        x = x.permute(0, 2, 1, 3)
        # rows
        x = torch.einsum('ij,bjkc->bikc', S, x)
        return x

    def _join_quadrants(self, x: torch.Tensor) -> torch.Tensor:
        LL, LH, HL, HH = torch.chunk(x, 4, dim=-1)
        top    = torch.cat([LL, HL], dim=2)
        bottom = torch.cat([LH, HH], dim=2)
        return torch.cat([top, bottom], dim=1)


if __name__ == '__main__':
    wave = 'haar'
    dwt  = DWT2D(wave)
    idwt = IDWT2D(wave)
    x = torch.randn(2, 256, 256, 1)
    lh = dwt(x)
    xhat = idwt(lh)
    print('DWT output:', lh.shape)
    print('Reconstruction error (max):', (x - xhat).abs().max().item())
