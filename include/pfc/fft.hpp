#pragma once

#include "constants.hpp"
#include "decomposition.hpp"
#include "types.hpp"
#include <heffte.h>
#include <iostream>

namespace pfc {

class FFT {

private:
  const Decomposition &m_decomposition;
  const heffte::fft3d_r2c<heffte::backend::fftw> m_fft;
  std::vector<std::complex<double>> m_wrk;

public:
  FFT(const Decomposition &decomposition, MPI_Comm comm)
      : m_decomposition(decomposition),
        m_fft({m_decomposition.inbox, m_decomposition.outbox,
               m_decomposition.r2c_direction, comm}),
        m_wrk(std::vector<std::complex<double>>(m_fft.size_workspace())){};

  void forward(std::vector<double> &in,
               std::vector<std::complex<double>> &out) {
    m_fft.forward(in.data(), out.data(), m_wrk.data());
  };

  void backward(std::vector<std::complex<double>> &in,
                std::vector<double> &out) {
    m_fft.backward(in.data(), out.data(), m_wrk.data(), heffte::scale::full);
  };

  const Decomposition &get_decomposition() { return m_decomposition; }
  auto size_inbox() const { return m_fft.size_inbox(); }
  auto size_outbox() const { return m_fft.size_outbox(); }
  auto size_workspace() const { return m_fft.size_workspace(); }
};

} // namespace pfc
