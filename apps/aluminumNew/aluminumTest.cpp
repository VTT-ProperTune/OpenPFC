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

#include "Aluminum.hpp"
#include "SeedGridFCC.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openpfc/openpfc.hpp>

using namespace pfc;
using namespace Catch::Matchers;

/* Parameters from aluminumNew.json:
{
    "n0": -0.0060,
    "alpha": 0.20,
    "n_sol": -0.036,
    "n_vap": -1.297,
    "T_const": 980,
    "T_min": 780,
    "T_max": 1280,
    "T0": 89285.0,
    "Bx": 0.817900686921996,
    "G_grid": 0,
    "V_grid": 0,
    "x_initial": 130,
    "alpha_farTol": 0.001,
    "alpha_highOrd": 0,
    "lambda": 0.22,
    "stabP": 0.0,
    "shift_u": 1.0,
    "shift_s": 0.0,
    "p2_bar": 0.8286531831,
    "p3_bar": -0.04204863,
    "p4_bar": 0.007533,
    "q20_bar": 0.016531729105214,
    "q21_bar": 5.467,
    "q30_bar": 1.7152418049986,
    "q31_bar": 0.45,
    "q40_bar": 0.787482
}
*/

TEST_CASE("Aluminum functionality", "[Aluminum]") {
  SECTION("Step model and calculate norm of the result") {
    MPI_Worker worker(0, nullptr);
    World world({32, 32, 32});
    Decomposition decomp(world);
    FFT fft(decomp);

    Aluminum aluminum;
    aluminum.set_n0(-0.0060);
    aluminum.set_alpha(0.20);
    aluminum.set_n_sol(-0.036);
    aluminum.set_n_vap(-1.297);
    aluminum.set_T_const(980);
    aluminum.set_T_min(780);
    aluminum.set_T_max(1280);
    aluminum.set_T0(89285.0);
    aluminum.set_Bx(0.817900686921996);
    aluminum.set_G_grid(0);
    aluminum.set_V_grid(0);
    aluminum.set_x_initial(130);
    aluminum.set_alpha_farTol(0.001);
    aluminum.set_alpha_highOrd(0);
    aluminum.set_lambda(0.22);
    aluminum.set_stabP(0.0);
    aluminum.set_shift_u(1.0);
    aluminum.set_shift_s(0.0);
    aluminum.set_p2_bar(0.8286531831);
    aluminum.set_p3_bar(-0.04204863);
    aluminum.set_p4_bar(0.007533);
    aluminum.set_q20_bar(0.016531729105214);
    aluminum.set_q21_bar(5.467);
    aluminum.set_q30_bar(1.7152418049986);
    aluminum.set_q31_bar(0.45);
    aluminum.set_q40_bar(0.787482);
    aluminum.set_fft(fft);
    double dt = 1.0e-2;
    aluminum.initialize(dt);

    SeedGridFCC ic;
    ic.set_Nx(1);
    ic.set_Ny(2);
    ic.set_Nz(2);
    ic.set_X0(8.0);
    ic.set_radius(4.0);
    ic.set_amplitude(0.4);
    ic.set_rho(-0.036);
    ic.set_rseed(42);

    std::vector<double> &psi = aluminum.get_real_field("psi");
    std::fill(psi.begin(), psi.end(), -0.0060);
    ic.apply(aluminum, 0.0);

    std::array<double, 5> expected_norms{1297.08, 1250.21, 1209.28, 1173.19, 1141.09};
    for (int i = 0; i < 5; ++i) {
      double norm2 = 0.0;
      for (auto &x : psi) norm2 += x * x;
      std::cout << "norm: " << norm2 << std::endl;
      REQUIRE_THAT(norm2, WithinAbs(expected_norms[i], 0.1));
      aluminum.step(1.0);
    }
  }
}
