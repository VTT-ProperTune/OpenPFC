#pragma once

#include "constants.hpp"
#include "decomposition.hpp"
#include "types.hpp"
#include <heffte.h>
#include <iostream>

namespace PFC {

class FFT {

private:
  const Decomposition m_decomp;
  const heffte::box3d<int> m_inbox, m_outbox;
  const heffte::fft3d_r2c<heffte::backend::fftw> m_fft;
  std::vector<std::complex<double>> m_wrk;

public:
  FFT(const Vec3<int> &dims, MPI_Comm comm = MPI_COMM_WORLD)
      : m_decomp({dims, comm}), m_inbox(m_decomp.inbox),
        m_outbox(m_decomp.outbox),
        m_fft({m_inbox, m_outbox, constants::r2c_direction, comm}),
        m_wrk(std::vector<std::complex<double>>(m_fft.size_workspace())){};

  void forward(std::vector<double> &in,
               std::vector<std::complex<double>> &out) {
    m_fft.forward(in.data(), out.data(), m_wrk.data());
  };

  void backward(std::vector<std::complex<double>> &in,
                std::vector<double> &out) {
    m_fft.backward(in.data(), out.data(), m_wrk.data(), heffte::scale::full);
  };

  int get_id() const { return m_decomp.get_id(); }
  size_t size_inbox() const { return m_fft.size_inbox(); }
  size_t size_outbox() const { return m_fft.size_outbox(); }
  size_t size_workspace() const { return m_fft.size_workspace(); }
  Vec3<int> get_inbox_low() const { return m_inbox.low; }
  Vec3<int> get_inbox_high() const { return m_inbox.high; }
  Vec3<int> get_outbox_low() const { return m_outbox.low; }
  Vec3<int> get_outbox_high() const { return m_outbox.high; }
};

} // namespace PFC
