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

#include <iostream>
#include <openpfc/model.hpp>
#include <openpfc/world.hpp>

using namespace std;
using namespace pfc;

class Diffusion : public Model {
  using Model::Model;

private:
  vector<double> opL, psi;
  vector<complex<double>> psi_F;
  const bool verbose = false;
  int m_midpoint_idx = -1;

public:
  void initialize(double dt) override {

    const World &w = get_world();
    FFT &fft = get_fft();
    const Decomposition &decomp = get_decomposition();

    psi.resize(fft.size_inbox());
    psi_F.resize(fft.size_outbox());
    opL.resize(fft.size_outbox());

    Vec3<int> i_low = decomp.inbox.low;
    Vec3<int> i_high = decomp.inbox.high;
    Vec3<int> o_low = decomp.outbox.low;
    Vec3<int> o_high = decomp.outbox.high;

    int idx = 0;
    double D = 1.0;
    for (int k = i_low[2]; k <= i_high[2]; k++) {
      for (int j = i_low[1]; j <= i_high[1]; j++) {
        for (int i = i_low[0]; i <= i_high[0]; i++) {
          double x = w.x0 + i * w.dx;
          double y = w.y0 + j * w.dy;
          double z = w.z0 + k * w.dz;
          psi[idx] = exp(-(x * x + y * y + z * z) / (4.0 * D));
          if (abs(x) < 1.0e-9 && abs(y) < 1.0e-9 && abs(z) < 1.0e-9) {
            cout << "Found midpoint from index " << idx << endl;
            m_midpoint_idx = idx;
          }
          idx += 1;
        }
      }
    }

    idx = 0;
    const double pi = std::atan(1.0) * 4.0;
    const double fx = 2.0 * pi / (w.dx * w.Lx);
    const double fy = 2.0 * pi / (w.dy * w.Ly);
    const double fz = 2.0 * pi / (w.dz * w.Lz);
    for (int k = o_low[2]; k <= o_high[2]; k++) {
      for (int j = o_low[1]; j <= o_high[1]; j++) {
        for (int i = o_low[0]; i <= o_high[0]; i++) {
          const double ki = (i <= w.Lx / 2) ? i * fx : (i - w.Lx) * fx;
          const double kj = (j <= w.Ly / 2) ? j * fy : (j - w.Ly) * fy;
          const double kk = (k <= w.Lz / 2) ? k * fz : (k - w.Lz) * fz;
          const double kLap = -(ki * ki + kj * kj + kk * kk);
          opL[idx++] = 1.0 / (1.0 - dt * kLap);
        }
      }
    }
  }

  void step(double) override {
    FFT &fft = get_fft();
    fft.forward(psi, psi_F);
    for (int k = 0, N = psi_F.size(); k < N; k++) {
      psi_F[k] = opL[k] * psi_F[k];
    }
    fft.backward(psi_F, psi);
  }

  Field &get_field() override { return psi; }

  int get_midpoint_idx() const { return m_midpoint_idx; }
};
