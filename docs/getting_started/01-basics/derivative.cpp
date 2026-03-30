// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/openpfc.hpp>

#include <cmath>
#include <cstdio>
#include <mpi.h>
#include <vector>

using namespace pfc;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  const int Lx = 16;
  const double pi = 3.14159265358979323846;
  const double dx = 2 * pi / Lx;
  const double x0 = -0.5 * Lx * dx;

  auto world = world::create(GridSize({Lx, 1, 1}), PhysicalOrigin({x0, 0.0, 0.0}),
                             GridSpacing({dx, 1.0, 1.0}));
  auto decomp = decomposition::create(world, mpi::get_size());
  auto fft = fft::create(decomp);

  const size_t size_inbox = fft.size_inbox();
  const size_t size_outbox = fft.size_outbox();

  std::vector<double> y(size_inbox), dy(size_inbox);
  std::vector<double> op(size_outbox);
  std::vector<std::complex<double>> Y(size_outbox);

  auto outbox = fft::get_outbox(fft);
  const double fx = 2.0 * pi / (dx * Lx);
  size_t idx = 0;
  for (int i = outbox.low[0]; i <= outbox.high[0]; ++i) {
    op[idx++] = (i < Lx / 2) ? i * fx : (i - Lx) * fx;
  }

  auto inbox = fft::get_inbox(fft);
  idx = 0;
  for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
    double x = x0 + i * dx;
    y[idx++] = std::sin(x);
  }

  fft.forward(y, Y);
  const std::complex<double> im(0, 1);
  for (size_t i = 0; i < size_outbox; ++i) {
    Y[i] = im * op[i] * Y[i];
  }
  fft.backward(Y, dy);

  idx = 0;
  for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
    double x = x0 + i * dx;
    double diff = std::abs(dy[idx] - std::cos(x));
    std::printf("i=%2d, x(i)=%5.2f, dy(x)=%5.2f, |dy(x)-dy_true(x)|=%0.5f\n", i, x,
                dy[idx], diff);
    ++idx;
  }

  MPI_Finalize();
  return 0;
}
