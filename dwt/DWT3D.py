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


class DWT3D(DWTNDlayout):
    """3D DWT analysis module.

    clean=True:  (batch, D, H, W, C) -> (batch, D/2, H/2, W/2, C*8)
    clean=False: (batch, D, H, W, C) -> (batch, D, H, W, C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[1]
        A = self._get_A(N, x.device)
        # columns (axis 2): swap D and H
        x = x.permute(0, 2, 1, 3, 4)
        x = torch.einsum('ij,bjklc->biklc', A, x)
        x = x.permute(0, 2, 1, 3, 4)
        # rows (axis 1)
        x = torch.einsum('ij,bjklc->biklc', A, x)
        # depth (axis 3): bring W to front
        x = x.permute(0, 3, 1, 2, 4)
        x = torch.einsum('ij,bjklc->biklc', A, x)
        x = x.permute(0, 2, 3, 1, 4)
        if self.clean:
            return self._extract_8subbands(x)
        return x

    def _extract_8subbands(self, x: torch.Tensor) -> torch.Tensor:
        mid = x.shape[1] // 2
        LLL = x[:, :mid, :mid, :mid, :]
        LLH = x[:, :mid, :mid, mid:, :]
        LHL = x[:, :mid, mid:, :mid, :]
        LHH = x[:, :mid, mid:, mid:, :]
        HLL = x[:, mid:, :mid, :mid, :]
        HLH = x[:, mid:, :mid, mid:, :]
        HHL = x[:, mid:, mid:, :mid, :]
        HHH = x[:, mid:, mid:, mid:, :]
        return torch.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=-1)


class IDWT3D(IDWTNDlayout):
    """3D IDWT synthesis module.

    clean=True:  (batch, D/2, H/2, W/2, C*8) -> (batch, D, H, W, C)
    clean=False: (batch, D, H, W, C)          -> (batch, D, H, W, C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.clean:
            x = self._join_octants(x)
        N = x.shape[1]
        S = self._get_S(N, x.device)
        # columns
        x = x.permute(0, 2, 1, 3, 4)
        x = torch.einsum('ij,bjklc->biklc', S, x)
        x = x.permute(0, 2, 1, 3, 4)
        # rows
        x = torch.einsum('ij,bjklc->biklc', S, x)
        # depth
        x = x.permute(0, 3, 1, 2, 4)
        x = torch.einsum('ij,bjklc->biklc', S, x)
        x = x.permute(0, 2, 3, 1, 4)
        return x

    def _join_octants(self, x: torch.Tensor) -> torch.Tensor:
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = torch.chunk(x, 8, dim=-1)
        front_top = torch.cat([LLL, LLH], dim=3)
        front_bot = torch.cat([LHL, LHH], dim=3)
        back_top  = torch.cat([HLL, HLH], dim=3)
        back_bot  = torch.cat([HHL, HHH], dim=3)
        front = torch.cat([front_top, front_bot], dim=2)
        back  = torch.cat([back_top,  back_bot],  dim=2)
        return torch.cat([front, back], dim=1)


if __name__ == '__main__':
    wave = 'bior1.5'
    dwt  = DWT3D(wave)
    idwt = IDWT3D(wave)
    x = torch.randn(1, 32, 32, 32, 2)
    lh = dwt(x)
    xhat = idwt(lh)
    print('DWT output:', lh.shape)
    print('Reconstruction error (max):', (x - xhat).abs().max().item())
