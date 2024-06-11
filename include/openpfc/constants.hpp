/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#pragma once

#include <cmath>

namespace pfc {

namespace constants {

const double pi = std::atan(1.0) * 4.0;

// 'lattice' constants for the three ordered phases that exist in 2D/3D PFC
const double a1D = 2 * pi;               // stripes
const double a2D = 2 * pi * 2 / sqrt(3); // triangular
const double a3D = 2 * pi * sqrt(2);     // BCC

/*
If the input of an FFT transform consists of all real numbers,
 the output comes in conjugate pairs which can be exploited to reduce
 both the floating point operations and MPI communications.
 Given a global set of indexes, HeFFTe can compute the corresponding DFT
 and exploit the real-to-complex symmetry by selecting a dimension
 and reducing the indexes by roughly half (the exact formula is floor(n / 2)
+ 1).

Do not change this.
 */
const int r2c_direction = 0;

} // namespace constants
} // namespace pfc
