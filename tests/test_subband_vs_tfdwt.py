"""Cross-framework subband comparison: DWT-PyTorch vs TFDWT.

For each (wave, dim, clean), fixes a random seed, runs both frameworks on the
same numerical input, and compares subbands index-by-index.

Passes if max absolute difference < ATOL (float32 cross-framework tolerance).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/kishor/src/TFDWT')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # CPU only — deterministic, no TF GPU init race
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
import tensorflow as tf

from dwt.DWT1D import DWT1D as PT_DWT1D, IDWT1D as PT_IDWT1D
from dwt.DWT2D import DWT2D as PT_DWT2D, IDWT2D as PT_IDWT2D
from dwt.DWT3D import DWT3D as PT_DWT3D, IDWT3D as PT_IDWT3D

from TFDWT.DWT1DFB import DWT1D as TF_DWT1D, IDWT1D as TF_IDWT1D
from TFDWT.DWT2DFB import DWT2D as TF_DWT2D, IDWT2D as TF_IDWT2D
from TFDWT.DWT3DFB import DWT3D as TF_DWT3D, IDWT3D as TF_IDWT3D

ATOL = 1e-5
WAVELETS = ['haar', 'db2', 'db4', 'bior1.3', 'bior3.1']

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
results = []

np.random.seed(42)


def check(name, tf_out, pt_out):
    tf_np = tf_out.numpy() if hasattr(tf_out, 'numpy') else tf_out
    pt_np = pt_out.detach().numpy() if hasattr(pt_out, 'detach') else pt_out
    diff = np.abs(tf_np - pt_np)
    max_err = diff.max()
    ok = max_err < ATOL
    tag = PASS if ok else FAIL
    print(f'  [{tag}] {name:65s}  max_diff={max_err:.2e}')
    results.append(ok)


# ─── 1D ─────────────────────────────────────────────────────────────────────
print('\n=== 1D subband comparison  (batch=2, N=64, C=2) ===')
x_np = np.random.randn(2, 64, 2).astype(np.float32)
x_tf = tf.constant(x_np)
x_pt = torch.tensor(x_np)

for wave in WAVELETS:
    for clean in [True, False]:
        tf_out = TF_DWT1D(wave=wave, clean=clean)(x_tf)
        pt_out = PT_DWT1D(wave=wave, clean=clean)(x_pt)
        check(f'DWT1D  wave={wave:<10} clean={clean}', tf_out, pt_out)

    # IDWT: feed DWT output into both IDWT layers
    # IDWT clean=True
    lh_tf = TF_DWT1D(wave=wave)(x_tf)
    lh_pt = PT_DWT1D(wave=wave)(x_pt)
    tf_rec = TF_IDWT1D(wave=wave)(lh_tf)
    pt_rec = PT_IDWT1D(wave=wave)(lh_pt)
    check(f'IDWT1D wave={wave:<10} clean=True ', tf_rec, pt_rec)

    # IDWT clean=False (feed raw DWT clean=False output)
    raw_tf = TF_DWT1D(wave=wave, clean=False)(x_tf)
    raw_pt = PT_DWT1D(wave=wave, clean=False)(x_pt)
    tf_rec2 = TF_IDWT1D(wave=wave, clean=False)(raw_tf)
    pt_rec2 = PT_IDWT1D(wave=wave, clean=False)(raw_pt)
    check(f'IDWT1D wave={wave:<10} clean=False', tf_rec2, pt_rec2)


# ─── 2D ─────────────────────────────────────────────────────────────────────
print('\n=== 2D subband comparison  (batch=2, H=32, W=32, C=2) ===')
x_np = np.random.randn(2, 32, 32, 2).astype(np.float32)
x_tf = tf.constant(x_np)
x_pt = torch.tensor(x_np)

for wave in WAVELETS:
    for clean in [True, False]:
        tf_out = TF_DWT2D(wave=wave, clean=clean)(x_tf)
        pt_out = PT_DWT2D(wave=wave, clean=clean)(x_pt)
        check(f'DWT2D  wave={wave:<10} clean={clean}', tf_out, pt_out)

    # IDWT clean=True
    lh_tf = TF_DWT2D(wave=wave)(x_tf)
    lh_pt = PT_DWT2D(wave=wave)(x_pt)
    tf_rec = TF_IDWT2D(wave=wave)(lh_tf)
    pt_rec = PT_IDWT2D(wave=wave)(lh_pt)
    check(f'IDWT2D wave={wave:<10} clean=True ', tf_rec, pt_rec)

    # IDWT clean=False
    raw_tf = TF_DWT2D(wave=wave, clean=False)(x_tf)
    raw_pt = PT_DWT2D(wave=wave, clean=False)(x_pt)
    tf_rec2 = TF_IDWT2D(wave=wave, clean=False)(raw_tf)
    pt_rec2 = PT_IDWT2D(wave=wave, clean=False)(raw_pt)
    check(f'IDWT2D wave={wave:<10} clean=False', tf_rec2, pt_rec2)


# ─── 3D ─────────────────────────────────────────────────────────────────────
print('\n=== 3D subband comparison  (batch=1, D=16, H=16, W=16, C=1) ===')
x_np = np.random.randn(1, 16, 16, 16, 1).astype(np.float32)
x_tf = tf.constant(x_np)
x_pt = torch.tensor(x_np)

for wave in WAVELETS:
    for clean in [True, False]:
        tf_out = TF_DWT3D(wave=wave, clean=clean)(x_tf)
        pt_out = PT_DWT3D(wave=wave, clean=clean)(x_pt)
        check(f'DWT3D  wave={wave:<10} clean={clean}', tf_out, pt_out)

    # IDWT clean=True
    lh_tf = TF_DWT3D(wave=wave)(x_tf)
    lh_pt = PT_DWT3D(wave=wave)(x_pt)
    tf_rec = TF_IDWT3D(wave=wave)(lh_tf)
    pt_rec = PT_IDWT3D(wave=wave)(lh_pt)
    check(f'IDWT3D wave={wave:<10} clean=True ', tf_rec, pt_rec)

    # IDWT clean=False
    raw_tf = TF_DWT3D(wave=wave, clean=False)(x_tf)
    raw_pt = PT_DWT3D(wave=wave, clean=False)(x_pt)
    tf_rec2 = TF_IDWT3D(wave=wave, clean=False)(raw_tf)
    pt_rec2 = PT_IDWT3D(wave=wave, clean=False)(raw_pt)
    check(f'IDWT3D wave={wave:<10} clean=False', tf_rec2, pt_rec2)


# ─── Summary ─────────────────────────────────────────────────────────────────
n_pass = sum(results)
n_total = len(results)
print(f'\n{"="*70}')
print(f'  {n_pass}/{n_total} subband comparison tests passed  (ATOL={ATOL})')
if n_pass == n_total:
    print('  All subband values match TFDWT index-by-index.')
else:
    print(f'  {n_total - n_pass} mismatch(es) found.')
    sys.exit(1)
