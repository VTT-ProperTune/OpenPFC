#include <cmath>

namespace PFC {

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
} // namespace PFC
