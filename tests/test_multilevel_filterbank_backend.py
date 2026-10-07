import math
import os
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from dwt import DWT1D, IDWT1D
from dwt.dbFBimpulseResponse import FBimpulseResponses
from dwt.multilevel.dwt1 import (
    dwt,
    dwt_packed_axis,
    idwt,
    idwt_packed_axis,
)


def _assert_close(actual, expected, tolerance=3e-5):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize('wave', ['haar', 'db2', 'bior2.2', 'rbio2.2'])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_list_api_filterbank_matches_matrix_and_reconstructs(wave, level):
    torch.manual_seed(401 + level)
    x = torch.randn(2, 128, 3)
    matrix = dwt(x, level=level, wave=wave, backend='matrix')
    filterbank = dwt(x, level=level, wave=wave, backend='filterbank')

    assert len(matrix) == level + 1
    for matrix_subband, filterbank_subband in zip(matrix, filterbank):
        _assert_close(filterbank_subband, matrix_subband)

    matrix_reconstruction = idwt(
        matrix,
        wave=wave,
        level=level,
        backend='matrix',
    )
    filterbank_reconstruction = idwt(
        filterbank,
        wave=wave,
        level=level,
        backend='filterbank',
    )
    _assert_close(filterbank_reconstruction, matrix_reconstruction)
    _assert_close(filterbank_reconstruction, x)


@pytest.mark.parametrize(
    'shape,axis',
    [
        ((32, 2, 3), 0),
        ((2, 32, 3), 1),
        ((2, 3, 32), 2),
        ((2, 3, 32), -1),
    ],
)
@pytest.mark.parametrize('wave', ['haar', 'bior2.2'])
def test_packed_arbitrary_axis_matches_matrix_and_reconstructs(
    shape,
    axis,
    wave,
):
    torch.manual_seed(405)
    x = torch.randn(shape)
    matrix = dwt_packed_axis(
        x,
        level=3,
        wave=wave,
        axis=axis,
        backend='matrix',
    )
    filterbank = dwt_packed_axis(
        x,
        level=3,
        wave=wave,
        axis=axis,
        backend='filterbank',
    )
    _assert_close(filterbank, matrix)
    _assert_close(
        idwt_packed_axis(
            filterbank,
            level=3,
            wave=wave,
            axis=axis,
            backend='filterbank',
        ),
        x,
    )
    _assert_close(
        idwt_packed_axis(
            matrix,
            level=3,
            wave=wave,
            axis=axis,
            backend='matrix',
        ),
        x,
    )


@pytest.mark.parametrize('backend', ['matrix', 'filterbank'])
def test_packed_order_matches_list_api_exactly(backend):
    torch.manual_seed(406)
    x = torch.randn(2, 64, 3)
    subbands = dwt(
        x,
        level=3,
        wave='bior2.2',
        backend=backend,
    )
    expected = torch.cat(
        [subbands[-1]] + list(reversed(subbands[:-1])),
        dim=1,
    )
    packed = dwt_packed_axis(
        x,
        level=3,
        wave='bior2.2',
        axis=1,
        backend=backend,
    )

    assert torch.equal(packed, expected)
    assert torch.equal(
        idwt_packed_axis(
            packed,
            level=3,
            wave='bior2.2',
            axis=1,
            backend=backend,
        ),
        idwt(
            subbands,
            wave='bior2.2',
            level=3,
            backend=backend,
        ),
    )


@pytest.mark.parametrize('backend', ['matrix', 'filterbank'])
def test_level_one_packed_is_raw_single_level_transform(backend):
    torch.manual_seed(407)
    x = torch.randn(2, 32, 3)
    packed = dwt_packed_axis(
        x,
        level=1,
        wave='bior2.2',
        axis=1,
        backend=backend,
    )
    expected = DWT1D(
        wave='bior2.2',
        clean=False,
        backend=backend,
    )(x)
    assert torch.equal(packed, expected)
    assert torch.equal(
        idwt_packed_axis(
            packed,
            level=1,
            wave='bior2.2',
            axis=1,
            backend=backend,
        ),
        IDWT1D(
            wave='bior2.2',
            clean=False,
            backend=backend,
        )(packed),
    )


def test_packed_filterbank_gradients_match_matrix():
    torch.manual_seed(408)
    x = torch.randn(2, 32, 3)
    probe = torch.randn_like(x)

    def gradient(backend):
        variable = x.detach().clone().requires_grad_(True)
        coefficients = dwt_packed_axis(
            variable,
            level=3,
            wave='bior2.2',
            axis=1,
            backend=backend,
        )
        reconstruction = idwt_packed_axis(
            coefficients,
            level=3,
            wave='bior2.2',
            axis=1,
            backend=backend,
        )
        gradient, = torch.autograd.grad(
            (reconstruction * probe).sum(),
            variable,
        )
        return gradient

    _assert_close(gradient('filterbank'), gradient('matrix'))


def test_level_two_filterbank_matches_matrix_for_every_wavelet():
    torch.manual_seed(409)
    for wave, banks in FBimpulseResponses.items():
        largest_filter = max(len(filt) for bank in banks for filt in bank)
        minimum = 2 * largest_filter
        length = max(16, 2 ** math.ceil(math.log2(minimum)))
        x = torch.randn(1, length, 1)
        matrix = dwt_packed_axis(
            x,
            level=2,
            wave=wave,
            axis=1,
            backend='matrix',
        )
        filterbank = dwt_packed_axis(
            x,
            level=2,
            wave=wave,
            axis=1,
            backend='filterbank',
        )
        _assert_close(filterbank, matrix)
        _assert_close(
            idwt_packed_axis(
                filterbank,
                level=2,
                wave=wave,
                axis=1,
                backend='filterbank',
            ),
            x,
        )


def test_level_zero_is_identity():
    x = torch.randn(2, 7, 3)
    subbands = dwt(x, level=0)
    assert len(subbands) == 1
    assert subbands[0] is x
    assert idwt([x], level=0) is x
    assert dwt_packed_axis(x, level=0) is x
    assert idwt_packed_axis(x, level=0) is x


def test_existing_positional_idwt_wave_argument_is_preserved():
    x = torch.randn(1, 32, 1)
    subbands = dwt(x, level=2, wave='bior2.2')
    _assert_close(idwt(subbands, 'bior2.2'), x)


def test_multilevel_validation():
    x = torch.zeros(1, 32, 1)
    with pytest.raises(ValueError, match='level'):
        dwt_packed_axis(x, level=-1)
    with pytest.raises(ValueError, match='level'):
        dwt(x, level=True)
    with pytest.raises(ValueError, match='divisible'):
        dwt_packed_axis(torch.zeros(1, 30, 1), level=2)
    with pytest.raises(ValueError, match='filter length'):
        dwt_packed_axis(x, level=3, wave='db10')
    with pytest.raises(ValueError, match='backend'):
        dwt(x, backend='unknown')
    subbands = dwt(x, level=2)
    with pytest.raises(
        ValueError,
        match='level=1 requires 2 subbands, but received 3',
    ):
        idwt(subbands, level=1)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason='MPS is unavailable',
)
def test_packed_multilevel_round_trip_on_mps():
    device = torch.device('mps')
    x = torch.randn(2, 32, 3, device=device)
    coefficients = dwt_packed_axis(
        x,
        level=3,
        wave='bior2.2',
        axis=1,
        backend='filterbank',
    )
    reconstruction = idwt_packed_axis(
        coefficients,
        level=3,
        wave='bior2.2',
        axis=1,
        backend='filterbank',
    )
    _assert_close(reconstruction, x)
