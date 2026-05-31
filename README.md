# DWT: Fast Multidimensional Discrete Wavelet Transform Layers (PyTorch)

[![PyPI Version](https://img.shields.io/pypi/v/dwt?label=PyPI&color=gold)](https://pypi.org/project/dwt/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/dwt)](https://pypi.org/project/dwt/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-Apache%202.0-deepgreen.svg?style=flat)](LICENSE)

Fast 1D, 2D, and 3D Discrete Wavelet Transform (DWT) and Inverse DWT (IDWT) layers for backpropagation networks. Drop-in `nn.Module` layers with einsum-based fast computation. CPU and GPU ready.

**Supported wavelet families**

```
Haar (haar)          Daubechies (db)      Symlets (sym)
Coiflets (coif)      Biorthogonal (bior)  Reverse biorthogonal (rbio)
```

**Shape requirements**

- Single-level: dimensions must be even. 2D and 3D inputs must be square / cubic.
- Multilevel (L levels): each spatial side must be divisible by 2^L. Pad to the nearest multiple of 2^L when needed.

---

## Installation

```bash
pip install dwt
```

From source:

```bash
git clone https://github.com/kkt-ee/DWT.git
cd DWT-PyTorch
pip install .
```

---

## Quick start

### 1D — `(batch, N, channels)`

```python
from dwt import DWT1D, IDWT1D

lh   = DWT1D(wave='bior3.1')(x)   # (B, N, C) -> (B, N/2, C*2)
xhat = IDWT1D(wave='bior3.1')(lh) # (B, N/2, C*2) -> (B, N, C)
```

### 2D — `(batch, H, W, channels)`

```python
from dwt import DWT2D, IDWT2D

lh   = DWT2D(wave='bior1.3')(x)   # (B, H, W, C) -> (B, H/2, W/2, C*4)
xhat = IDWT2D(wave='bior1.3')(lh) # (B, H/2, W/2, C*4) -> (B, H, W, C)
```

### 3D — `(batch, D, H, W, channels)`

```python
from dwt import DWT3D, IDWT3D

lh   = DWT3D(wave='bior1.3')(x)   # (B, D, H, W, C) -> (B, D/2, H/2, W/2, C*8)
xhat = IDWT3D(wave='bior1.3')(lh) # (B, D/2, H/2, W/2, C*8) -> (B, D, H, W, C)
```

`clean=True` (default) packs subbands along the channel axis and halves spatial dims.
`clean=False` returns the raw operator output at full spatial size.

---

## Tensor shapes

| Module | Input | Output (clean=True) |
|--------|-------|---------------------|
| `DWT1D` | `(B, N, C)` | `(B, N/2, C×2)` — L \|\| H |
| `DWT2D` | `(B, H, W, C)` | `(B, H/2, W/2, C×4)` — LL \| LH \| HL \| HH |
| `DWT3D` | `(B, D, H, W, C)` | `(B, D/2, H/2, W/2, C×8)` — 8 subbands |
| `IDWT1D` | `(B, N/2, C×2)` | `(B, N, C)` |
| `IDWT2D` | `(B, H/2, W/2, C×4)` | `(B, H, W, C)` |
| `IDWT3D` | `(B, D/2, H/2, W/2, C×8)` | `(B, D, H, W, C)` |

All layouts are **channels-last**.

---

## Multilevel DWT

Returns a list `[H1, H2, ..., H_level, L_level]`. Each `Hi` contains all high-pass subbands at level `i` packed along the channel axis. The last element is the final low-pass residual.

**1D**

```python
from dwt.multilevel.dwt1 import dwt, idwt

subbands = dwt(x, level=3, wave='haar')   # [H1, H2, H3, L3]
xhat     = idwt(subbands, wave='haar')
```

**2D**

```python
from dwt.multilevel.dwt2 import dwt2, idwt2

subbands = dwt2(x, level=3, wave='haar')  # [H1, H2, H3, L3]
xhat     = idwt2(subbands, wave='haar')
```

**3D**

```python
from dwt.multilevel.dwt3 import dwt3, idwt3

subbands = dwt3(x, level=2, wave='haar')  # [H1, H2, L2]
xhat     = idwt3(subbands, wave='haar')
```

Using the single-level and multilevel transforms, arbitrary multilevel filter banks and Wavelet Packet Transform filter banks can be constructed.

---

## Use as a PyTorch layer in a model

```python
import torch.nn as nn
from dwt import DWT2D, IDWT2D

class WaveletAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = DWT2D(wave='db4')
        self.dec = IDWT2D(wave='db4')

    def forward(self, x):
        return self.dec(self.enc(x))
```

---

## Verified dependencies

```
Python   3.12+
PyTorch  2.0+  (verified on 2.7)
CUDA     12+
```

---

## Uninstall

```bash
pip uninstall dwt
```

---

*DWT (C) 2026 Kishore Kumar Tarafdar, भारत* 🇮🇳
