# FDWT: Fast Multidimensional Discrete Wavelet Transform Layers (PyTorch)

[![PyPI Version](https://img.shields.io/pypi/v/fdwt?label=PyPI&color=gold)](https://pypi.org/project/fdwt/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/fdwt)](https://pypi.org/project/fdwt/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-Apache%202.0-deepgreen.svg?style=flat)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2504.04168-b31b1b.svg)](https://doi.org/10.48550/arXiv.2504.04168)

Fast 1D, 2D, and 3D Discrete Wavelet Transform (DWT) and Inverse DWT (IDWT) layers for backpropagation networks. Drop-in `nn.Module` layers with dense-operator and matrix-free filter-bank computation. CPU and GPU ready.

> PyTorch port of [TFDWT](https://github.com/kkt-ee/TFDWT) — Fast Discrete Wavelet Transform TensorFlow Layers ([arXiv:2504.04168](https://doi.org/10.48550/arXiv.2504.04168)).

**Supported wavelet families**

```
Haar (haar)          Daubechies (db)      Symlets (sym)
Coiflets (coif)      Biorthogonal (bior)  Reverse biorthogonal (rbio)
```

**Shape requirements**

- Single-level: dimensions must be even. 2D and 3D inputs must be square / cubic.
- Multilevel (L levels): each transformed length must be divisible by 2^L,
  and every decomposition stage must cover the selected wavelet-filter length.

---

## Installation

```bash
pip install fdwt
```

From source:

```bash
git clone https://github.com/kkt-ee/FDWT.git
cd FDWT
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

The layers use `backend='matrix'` by default, preserving the existing dense
operator implementation. Set `backend='filterbank'` on both DWT and IDWT to
compute the same periodic transform with strided convolutions, without storing
an `N x N` operator matrix:

```python
lh = DWT1D(wave='bior3.1', backend='filterbank')(x)
xhat = IDWT1D(wave='bior3.1', backend='filterbank')(lh)
```

The same `backend` argument is available on the 2D and 3D layers.

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

subbands = dwt(                           # [H1, H2, H3, L3]
    x, level=3, wave='haar', backend='filterbank'
)
xhat = idwt(
    subbands, wave='haar', level=3, backend='filterbank'
)
```

For a length-preserving coefficient tensor, including transforms along an
arbitrary tensor axis, use the packed 1D helpers. Their coefficient order is
`[L_level, H_level, H_(level-1), ..., H1]`:

```python
from dwt.multilevel.dwt1 import dwt_packed_axis, idwt_packed_axis

coefficients = dwt_packed_axis(
    x, level=3, wave='bior2.2', axis=1, backend='filterbank'
)
xhat = idwt_packed_axis(
    coefficients, level=3, wave='bior2.2', axis=1,
    backend='filterbank',
)
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
pip uninstall fdwt
```


---

## Citation

Cite the original TFDWT paper:

```bibtex
@misc{tarafdar2025tfdwt,
  title   = {TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers},
  author  = {Kishore Kumar Tarafdar},
  year    = {2025},
  url     = {https://doi.org/10.48550/arXiv.2504.04168},
  note    = {arXiv:2504.04168}
}
```

---

*FDWT (C) 2026 Kishore Kumar Tarafdar, भारत* 🇮🇳
