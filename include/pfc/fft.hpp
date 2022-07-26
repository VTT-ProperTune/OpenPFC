#pragma once

#include "constants.hpp"
#include "decomposition.hpp"
#include <heffte.h>
#include <iostream>

namespace PFC {

class FFT {
private:
  const Decomposition decomp;
  const heffte::fft3d_r2c<heffte::backend::fftw> m_fft;
  std::vector<std::complex<double>> wrk;

public:
  FFT(const std::array<int, 3> &dims, MPI_Comm comm = MPI_COMM_WORLD)
      : decomp({dims, comm}),
        m_fft({decomp.inbox, decomp.outbox, constants::r2c_direction, comm}),
        wrk(std::vector<std::complex<double>>(m_fft.size_workspace())){};

  void forward(std::vector<double> &in,
               std::vector<std::complex<double>> &out) {
    m_fft.forward(in.data(), out.data(), wrk.data());
  };

  void backward(std::vector<std::complex<double>> &in,
                std::vector<double> &out) {
    m_fft.backward(in.data(), out.data(), wrk.data(), heffte::scale::full);
  };

  size_t size_inbox() const { return m_fft.size_inbox(); }
  size_t size_outbox() const { return m_fft.size_outbox(); }
  size_t size_workspace() const { return m_fft.size_workspace(); }
  Decomposition get_decomposition() const { return decomp; }
};

} // namespace PFC
