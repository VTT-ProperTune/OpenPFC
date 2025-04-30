// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <openpfc/core/world.hpp>
#include <openpfc/model.hpp>

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

    Vec3<int> i_low = decomp.get_inbox().low;
    Vec3<int> i_high = decomp.get_inbox().high;
    Vec3<int> o_low = decomp.get_outbox().low;
    Vec3<int> o_high = decomp.get_outbox().high;

    auto origin = get_origin(w);
    auto spacing = get_spacing(w);

    int idx = 0;
    double D = 1.0;
    for (int k = i_low[2]; k <= i_high[2]; k++) {
      for (int j = i_low[1]; j <= i_high[1]; j++) {
        for (int i = i_low[0]; i <= i_high[0]; i++) {
          double x = origin[0] + i * spacing[0];
          double y = origin[1] + j * spacing[1];
          double z = origin[2] + k * spacing[2];
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
    const double fx = 2.0 * pi / (spacing[0] * get_size(w, 0));
    const double fy = 2.0 * pi / (spacing[1] * get_size(w, 1));
    const double fz = 2.0 * pi / (spacing[2] * get_size(w, 2));
    for (int k = o_low[2]; k <= o_high[2]; k++) {
      for (int j = o_low[1]; j <= o_high[1]; j++) {
        for (int i = o_low[0]; i <= o_high[0]; i++) {
          const double ki =
              (i <= get_size(w, 0) / 2) ? i * fx : (i - get_size(w, 0)) * fx;
          const double kj =
              (j <= get_size(w, 1) / 2) ? j * fy : (j - get_size(w, 1)) * fy;
          const double kk =
              (k <= get_size(w, 2) / 2) ? k * fz : (k - get_size(w, 2)) * fz;
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
