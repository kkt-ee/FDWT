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

from dwt.dbFBimpulseResponse import FBimpulseResponses


class FetchAnalysisSynthesisFilters:
    """Get h0, h1, g0, g1 FIR filters (analysis + synthesis) for a PR DWT filter bank.

    Supports orthogonal (db, haar, sym, ...) and biorthogonal (bior, rbio) families.
    Filter coefficients sourced from the embedded FBimpulseResponses dict.
    """

    def __init__(self, wavelet: str):
        w = FBimpulseResponses[wavelet]
        self.h0n, self.h1n = w[0][0], w[0][1]
        self.g0n, self.g1n = w[1][0], w[1][1]

    def analysis(self):
        """Return (h0, h1) analysis filters."""
        return self.h0n, self.h1n

    def synthesis(self):
        """Return (g0, g1) synthesis filters."""
        return self.g0n, self.g1n
