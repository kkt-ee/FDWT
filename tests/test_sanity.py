"""Sanity checks for DWT-PyTorch.

Tests:
  1. Perfect reconstruction (PR): IDWT(DWT(x)) == x  within floating-point tolerance.
  2. Subband shapes for clean mode.
  3. clean=False round-trip.
  4. Multilevel PR (1D, 2D, 3D).
  5. GPU round-trip (skipped if CUDA unavailable).
  6. Orthogonal + biorthogonal wavelets.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from dwt import DWT1D, IDWT1D, DWT2D, IDWT2D, DWT3D, IDWT3D
from dwt.multilevel.dwt1 import dwt, idwt
from dwt.multilevel.dwt2 import dwt2, idwt2
from dwt.multilevel.dwt3 import dwt3, idwt3

ATOL = 1e-4  # float32 accumulation tolerance

WAVELETS_ORTHO   = ['haar', 'db1', 'db2', 'db4', 'db6', 'sym4']
WAVELETS_BIOR    = ['bior1.3', 'bior1.5', 'bior2.2', 'bior3.1']
WAVELETS_ALL     = WAVELETS_ORTHO + WAVELETS_BIOR

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'

results = []


def check(name, x, xhat):
    err = (x - xhat).abs().max().item()
    ok = err < ATOL
    tag = PASS if ok else FAIL
    print(f'  [{tag}] {name:55s}  max_err={err:.2e}')
    results.append(ok)
    return ok


# ─── 1D ─────────────────────────────────────────────────────────────────────
print('\n=== 1D DWT/IDWT  (batch=2, N=256, C=3) ===')
x1 = torch.randn(2, 256, 3)

for wave in WAVELETS_ALL:
    lh   = DWT1D(wave)(x1)
    xhat = IDWT1D(wave)(lh)
    check(f'1D PR  wave={wave}', x1, xhat)

# shape check clean mode
wave = 'db2'
lh = DWT1D(wave, clean=True)(x1)
assert lh.shape == (2, 128, 6),  f'Shape mismatch: {lh.shape}'
print(f'  [    ] 1D clean shape  (2,256,3)->(2,128,6): {lh.shape}  OK')

# clean=False round-trip
lh2   = DWT1D(wave, clean=False)(x1)
xhat2 = IDWT1D(wave, clean=False)(lh2)
check(f'1D PR  clean=False  wave={wave}', x1, xhat2)


# ─── 2D ─────────────────────────────────────────────────────────────────────
print('\n=== 2D DWT/IDWT  (batch=2, H=64, W=64, C=2) ===')
x2 = torch.randn(2, 64, 64, 2)

for wave in WAVELETS_ALL:
    lh   = DWT2D(wave)(x2)
    xhat = IDWT2D(wave)(lh)
    check(f'2D PR  wave={wave}', x2, xhat)

# shape check
wave = 'haar'
lh = DWT2D(wave, clean=True)(x2)
assert lh.shape == (2, 32, 32, 8),  f'Shape mismatch: {lh.shape}'
print(f'  [    ] 2D clean shape  (2,64,64,2)->(2,32,32,8): {lh.shape}  OK')

# clean=False
lh2   = DWT2D(wave, clean=False)(x2)
xhat2 = IDWT2D(wave, clean=False)(lh2)
check(f'2D PR  clean=False  wave={wave}', x2, xhat2)


# ─── 3D ─────────────────────────────────────────────────────────────────────
print('\n=== 3D DWT/IDWT  (batch=1, D=32, H=32, W=32, C=1) ===')
x3 = torch.randn(1, 32, 32, 32, 1)

for wave in WAVELETS_ALL:
    lh   = DWT3D(wave)(x3)
    xhat = IDWT3D(wave)(lh)
    check(f'3D PR  wave={wave}', x3, xhat)

# shape check
wave = 'haar'
lh = DWT3D(wave, clean=True)(x3)
assert lh.shape == (1, 16, 16, 16, 8),  f'Shape mismatch: {lh.shape}'
print(f'  [    ] 3D clean shape  (1,32,32,32,1)->(1,16,16,16,8): {lh.shape}  OK')


# ─── Multilevel 1D ──────────────────────────────────────────────────────────
print('\n=== Multilevel 1D  (batch=1, N=256, C=2, level=4) ===')
x = torch.randn(1, 256, 2)
for wave in ['haar', 'db4', 'bior3.1']:
    sb   = dwt(x, level=4, wave=wave)
    xhat = idwt(sb, wave=wave)
    check(f'Multilevel 1D  wave={wave}  level=4', x, xhat)
    shapes = [s.shape for s in sb]
    print(f'         subband shapes: {shapes}')


# ─── Multilevel 2D ──────────────────────────────────────────────────────────
print('\n=== Multilevel 2D  (batch=1, H=64, W=64, C=1, level=3) ===')
x = torch.randn(1, 64, 64, 1)
for wave in ['haar', 'db2', 'bior1.3']:
    sb   = dwt2(x, level=3, wave=wave)
    xhat = idwt2(sb, wave=wave)
    check(f'Multilevel 2D  wave={wave}  level=3', x, xhat)
    shapes = [s.shape for s in sb]
    print(f'         subband shapes: {shapes}')


# ─── Multilevel 3D ──────────────────────────────────────────────────────────
print('\n=== Multilevel 3D  (batch=1, D=32, H=32, W=32, C=1, level=2) ===')
x = torch.randn(1, 32, 32, 32, 1)
for wave in ['haar', 'db2', 'bior1.5']:
    sb   = dwt3(x, level=2, wave=wave)
    xhat = idwt3(sb, wave=wave)
    check(f'Multilevel 3D  wave={wave}  level=2', x, xhat)
    shapes = [s.shape for s in sb]
    print(f'         subband shapes: {shapes}')


# ─── GPU ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    print('\n=== GPU round-trip  (1D, 2D, 3D, wave=db4) ===')
    dev = torch.device('cuda')
    wave = 'db4'

    x1g = torch.randn(2, 256, 3, device=dev)
    check('GPU 1D PR', x1g, IDWT1D(wave)(DWT1D(wave)(x1g)))

    x2g = torch.randn(2, 64, 64, 2, device=dev)
    check('GPU 2D PR', x2g, IDWT2D(wave)(DWT2D(wave)(x2g)))

    x3g = torch.randn(1, 32, 32, 32, 1, device=dev)
    check('GPU 3D PR', x3g, IDWT3D(wave)(DWT3D(wave)(x3g)))
else:
    print('\n  [SKIP] CUDA not available — GPU tests skipped')


# ─── Subband energy check (1D, haar) ────────────────────────────────────────
print('\n=== Subband energy check (1D haar, orthogonal -> energy conserved) ===')
x = torch.randn(1, 256, 1)
lh = DWT1D('haar')(x)
L = lh[:, :, :1]
H = lh[:, :, 1:]
E_in  = (x ** 2).sum().item()
E_out = (L ** 2).sum().item() + (H ** 2).sum().item()
ratio = E_out / E_in
ok = abs(ratio - 1.0) < 1e-3
tag = PASS if ok else FAIL
print(f'  [{tag}] 1D haar energy ratio L+H / input = {ratio:.6f}  (expect 1.0)')
results.append(ok)


# ─── Summary ────────────────────────────────────────────────────────────────
n_pass = sum(results)
n_total = len(results)
print(f'\n{"="*60}')
print(f'  {n_pass}/{n_total} tests passed')
if n_pass == n_total:
    print('  All tests PASSED.')
else:
    print(f'  {n_total - n_pass} test(s) FAILED.')
    sys.exit(1)
