import math
import os
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from dwt import DWT1D, DWT2D, DWT3D, IDWT1D, IDWT2D, IDWT3D
from dwt.dbFBimpulseResponse import FBimpulseResponses
from dwt.dwt_op import (
    analysis_filterbank_axis,
    make_dwt_operator_matrix_A,
    synthesis_filterbank_axis,
)
from dwt.filters import FetchAnalysisSynthesisFilters


def _assert_close(actual, expected, tolerance=2e-6):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize('wave', ['haar', 'db2', 'bior2.2', 'rbio2.2'])
@pytest.mark.parametrize('clean', [False, True])
def test_1d_filterbank_matches_matrix_and_reconstructs(wave, clean):
    torch.manual_seed(101)
    x = torch.randn(2, 32, 3)

    matrix_dwt = DWT1D(wave=wave, clean=clean, backend='matrix')
    filterbank_dwt = DWT1D(wave=wave, clean=clean, backend='filterbank')
    matrix_coefficients = matrix_dwt(x)
    filterbank_coefficients = filterbank_dwt(x)
    _assert_close(filterbank_coefficients, matrix_coefficients)

    matrix_idwt = IDWT1D(wave=wave, clean=clean, backend='matrix')
    filterbank_idwt = IDWT1D(wave=wave, clean=clean, backend='filterbank')
    matrix_reconstruction = matrix_idwt(matrix_coefficients)
    filterbank_reconstruction = filterbank_idwt(filterbank_coefficients)
    _assert_close(filterbank_reconstruction, matrix_reconstruction)
    _assert_close(filterbank_reconstruction, x)

    assert filterbank_dwt._A_cache is None
    assert filterbank_idwt._S_cache is None


def test_filterbank_matches_matrix_for_every_supported_wavelet():
    torch.manual_seed(102)
    for wave, banks in FBimpulseResponses.items():
        largest_filter = max(len(filt) for bank in banks for filt in bank)
        length = max(32, 2 ** math.ceil(math.log2(largest_filter)))
        x = torch.randn(1, length, 2)

        matrix_coefficients = DWT1D(
            wave=wave, clean=False, backend='matrix'
        )(x)
        filterbank_coefficients = DWT1D(
            wave=wave, clean=False, backend='filterbank'
        )(x)
        _assert_close(filterbank_coefficients, matrix_coefficients, tolerance=3e-6)

        matrix_reconstruction = IDWT1D(
            wave=wave, clean=False, backend='matrix'
        )(matrix_coefficients)
        filterbank_reconstruction = IDWT1D(
            wave=wave, clean=False, backend='filterbank'
        )(filterbank_coefficients)
        _assert_close(
            filterbank_reconstruction,
            matrix_reconstruction,
            tolerance=3e-6,
        )
        _assert_close(filterbank_reconstruction, x, tolerance=3e-6)


@pytest.mark.parametrize('wave', ['haar', 'bior2.2'])
@pytest.mark.parametrize('clean', [False, True])
def test_2d_filterbank_matches_matrix_and_reconstructs(wave, clean):
    torch.manual_seed(103)
    x = torch.randn(2, 16, 16, 2)

    matrix_coefficients = DWT2D(
        wave=wave, clean=clean, backend='matrix'
    )(x)
    filterbank_coefficients = DWT2D(
        wave=wave, clean=clean, backend='filterbank'
    )(x)
    _assert_close(filterbank_coefficients, matrix_coefficients, tolerance=5e-6)

    matrix_reconstruction = IDWT2D(
        wave=wave, clean=clean, backend='matrix'
    )(matrix_coefficients)
    filterbank_reconstruction = IDWT2D(
        wave=wave, clean=clean, backend='filterbank'
    )(filterbank_coefficients)
    _assert_close(filterbank_reconstruction, matrix_reconstruction, tolerance=5e-6)
    _assert_close(filterbank_reconstruction, x, tolerance=5e-6)


@pytest.mark.parametrize('wave', ['haar', 'bior2.2'])
@pytest.mark.parametrize('clean', [False, True])
def test_3d_filterbank_matches_matrix_and_reconstructs(wave, clean):
    torch.manual_seed(104)
    x = torch.randn(1, 16, 16, 16, 2)

    matrix_coefficients = DWT3D(
        wave=wave, clean=clean, backend='matrix'
    )(x)
    filterbank_coefficients = DWT3D(
        wave=wave, clean=clean, backend='filterbank'
    )(x)
    _assert_close(filterbank_coefficients, matrix_coefficients, tolerance=1e-5)

    matrix_reconstruction = IDWT3D(
        wave=wave, clean=clean, backend='matrix'
    )(matrix_coefficients)
    filterbank_reconstruction = IDWT3D(
        wave=wave, clean=clean, backend='filterbank'
    )(filterbank_coefficients)
    _assert_close(filterbank_reconstruction, matrix_reconstruction, tolerance=1e-5)
    _assert_close(filterbank_reconstruction, x, tolerance=1e-5)


@pytest.mark.parametrize('layer_class', [DWT1D, IDWT1D])
@pytest.mark.parametrize('wave', ['haar', 'bior2.2'])
def test_1d_filterbank_input_gradient_matches_matrix(layer_class, wave):
    torch.manual_seed(105)
    x = torch.randn(2, 32, 3)
    probe = torch.randn_like(x)

    def input_gradient(backend):
        variable = x.detach().clone().requires_grad_(True)
        output = layer_class(wave=wave, clean=False, backend=backend)(variable)
        gradient, = torch.autograd.grad((output * probe).sum(), variable)
        return gradient

    _assert_close(input_gradient('filterbank'), input_gradient('matrix'))


def test_generic_axis_filterbanks_match_matrix_operators():
    torch.manual_seed(106)
    x = torch.randn(2, 3, 4, 32, 5, requires_grad=True)
    wave = 'bior2.2'
    filters = FetchAnalysisSynthesisFilters(wave)
    h0, h1 = filters.analysis()
    g0, g1 = filters.synthesis()

    analysis_matrix = make_dwt_operator_matrix_A(h0, h1, 32)
    moved = x.movedim(3, -1)
    expected_analysis = torch.einsum(
        'ij,...j->...i', analysis_matrix, moved
    ).movedim(-1, 3)
    actual_analysis = analysis_filterbank_axis(x, h0, h1, axis=3)
    _assert_close(actual_analysis, expected_analysis)

    synthesis_matrix = make_dwt_operator_matrix_A(g0, g1, 32).T
    expected_synthesis = torch.einsum(
        'ij,...j->...i', synthesis_matrix, moved
    ).movedim(-1, 3)
    actual_synthesis = synthesis_filterbank_axis(x, g0, g1, axis=3)
    _assert_close(actual_synthesis, expected_synthesis)

    probe = torch.randn_like(x)
    expected_gradient, = torch.autograd.grad(
        (expected_analysis * probe).sum(), x, retain_graph=True
    )
    actual_gradient, = torch.autograd.grad(
        (actual_analysis * probe).sum(), x
    )
    _assert_close(actual_gradient, expected_gradient)


def test_backend_is_validated():
    with pytest.raises(ValueError, match='backend must be one of'):
        DWT1D(backend='unknown')


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason='MPS is unavailable',
)
def test_filterbank_round_trip_on_mps():
    device = torch.device('mps')
    x = torch.randn(2, 32, 3, device=device)
    coefficients = DWT1D(
        wave='bior2.2', clean=False, backend='filterbank'
    )(x)
    reconstruction = IDWT1D(
        wave='bior2.2', clean=False, backend='filterbank'
    )(coefficients)
    _assert_close(reconstruction, x, tolerance=1e-5)
