"""ॐ
FDWT: Fast Multidimensional Discrete Wavelet Transform Layers (PyTorch).
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

from dwt.filters import FetchAnalysisSynthesisFilters
from dwt.dwt_op import make_dwt_operator_matrix_A
from dwt.DWT1D import DWT1D, IDWT1D
from dwt.DWT2D import DWT2D, IDWT2D
from dwt.DWT3D import DWT3D, IDWT3D
from dwt.multilevel.dwt1 import dwt, dwt_packed_axis, idwt, idwt_packed_axis
from dwt.multilevel.dwt2 import dwt2, idwt2
from dwt.multilevel.dwt3 import dwt3, idwt3

__version__ = "0.1.2"
