"""Cross-framework multilevel subband comparison: DWT-PyTorch vs TFDWT.

Compares every subband tensor index-by-index for dwt/idwt, dwt2/idwt2, dwt3/idwt3.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/kishor/src/TFDWT')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
import tensorflow as tf

from dwt.multilevel.dwt1 import dwt  as PT_dwt,  idwt  as PT_idwt
from dwt.multilevel.dwt2 import dwt2 as PT_dwt2, idwt2 as PT_idwt2
from dwt.multilevel.dwt3 import dwt3 as PT_dwt3, idwt3 as PT_idwt3

from TFDWT.multilevel.dwt  import dwt  as TF_dwt,  idwt  as TF_idwt
from TFDWT.multilevel.dwt2 import dwt2 as TF_dwt2, idwt2 as TF_idwt2
from TFDWT.multilevel.dwt3 import dwt3 as TF_dwt3, idwt3 as TF_idwt3

ATOL = 1e-5
WAVELETS = ['haar', 'db2', 'db4', 'bior1.3', 'bior3.1']

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
results = []

np.random.seed(42)


def check(name, tf_t, pt_t):
    tf_np = tf_t.numpy() if hasattr(tf_t, 'numpy') else np.array(tf_t)
    pt_np = pt_t.detach().numpy()
    diff = np.abs(tf_np - pt_np).max()
    ok = diff < ATOL
    tag = PASS if ok else FAIL
    print(f'  [{tag}] {name:70s}  max_diff={diff:.2e}')
    results.append(ok)


def check_list(prefix, tf_sbs, pt_sbs):
    assert len(tf_sbs) == len(pt_sbs), f'Length mismatch: {len(tf_sbs)} vs {len(pt_sbs)}'
    for i, (tf_s, pt_s) in enumerate(zip(tf_sbs, pt_sbs)):
        label = 'L' if i == len(tf_sbs) - 1 else f'H{i+1}'
        check(f'{prefix}  subband[{label}]  shape={tuple(pt_s.shape)}', tf_s, pt_s)


# ─── Multilevel 1D ──────────────────────────────────────────────────────────
print('\n=== Multilevel 1D  (batch=1, N=64, C=2, level=3) ===')
x_np = np.random.randn(1, 64, 2).astype(np.float32)
x_tf = tf.constant(x_np)
x_pt = torch.tensor(x_np)

for wave in WAVELETS:
    tf_sbs = TF_dwt(x_tf, level=3, Ψ=wave)
    pt_sbs = PT_dwt(x_pt, level=3, wave=wave)
    check_list(f'dwt  wave={wave:<10} level=3', tf_sbs, pt_sbs)

    tf_rec = TF_idwt(tf_sbs, level=3, Ψ=wave)
    pt_rec = PT_idwt(pt_sbs, wave=wave)
    check(f'idwt wave={wave:<10} level=3  reconstruction', tf_rec, pt_rec)


# ─── Multilevel 2D ──────────────────────────────────────────────────────────
print('\n=== Multilevel 2D  (batch=1, H=32, W=32, C=1, level=3) ===')
x_np = np.random.randn(1, 32, 32, 1).astype(np.float32)
x_tf = tf.constant(x_np)
x_pt = torch.tensor(x_np)

for wave in WAVELETS:
    tf_sbs = TF_dwt2(x_tf, level=3, Ψ=wave)
    pt_sbs = PT_dwt2(x_pt, level=3, wave=wave)
    check_list(f'dwt2 wave={wave:<10} level=3', tf_sbs, pt_sbs)

    tf_rec = TF_idwt2(tf_sbs, level=3, Ψ=wave)
    pt_rec = PT_idwt2(pt_sbs, wave=wave)
    check(f'idwt2 wave={wave:<10} level=3  reconstruction', tf_rec, pt_rec)


# ─── Multilevel 3D ──────────────────────────────────────────────────────────
print('\n=== Multilevel 3D  (batch=1, D=16, H=16, W=16, C=1, level=2) ===')
x_np = np.random.randn(1, 16, 16, 16, 1).astype(np.float32)
x_tf = tf.constant(x_np)
x_pt = torch.tensor(x_np)

for wave in WAVELETS:
    tf_sbs = TF_dwt3(x_tf, level=2, Ψ=wave)
    pt_sbs = PT_dwt3(x_pt, level=2, wave=wave)
    check_list(f'dwt3 wave={wave:<10} level=2', tf_sbs, pt_sbs)

    tf_rec = TF_idwt3(tf_sbs, level=2, Ψ=wave)
    pt_rec = PT_idwt3(pt_sbs, wave=wave)
    check(f'idwt3 wave={wave:<10} level=2  reconstruction', tf_rec, pt_rec)


# ─── Summary ────────────────────────────────────────────────────────────────
n_pass = sum(results)
n_total = len(results)
print(f'\n{"="*70}')
print(f'  {n_pass}/{n_total} multilevel subband comparison tests passed  (ATOL={ATOL})')
if n_pass == n_total:
    print('  All multilevel subbands match TFDWT index-by-index.')
else:
    print(f'  {n_total - n_pass} mismatch(es) found.')
    sys.exit(1)
