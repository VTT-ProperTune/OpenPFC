#pragma once
#include "fft.hpp"
#include "types.hpp"
#include "world.hpp"

namespace pfc {
class Model {

private:
  World m_world;
  FFT m_fft;

public:
  const int id;
  const bool master;

  Model(const Vec3<int> &dimensions, const Vec3<double> &origo,
        const Vec3<double> &discretization, MPI_Comm comm = MPI_COMM_WORLD)
      : m_world(dimensions, origo, discretization), m_fft(dimensions, comm),
        id(m_fft.get_id()), master(id == 0) {}

  void fft_r2c(std::vector<double> &A, std::vector<std::complex<double>> &B) {
    m_fft.forward(A, B);
  }

  void fft_c2r(std::vector<std::complex<double>> &A, std::vector<double> &B) {
    m_fft.backward(A, B);
  }

  World get_world() const { return m_world; }
  size_t size_inbox() const { return m_fft.size_inbox(); }
  size_t size_outbox() const { return m_fft.size_outbox(); }
  Vec3<int> get_inbox_low() const { return m_fft.get_inbox_low(); }
  Vec3<int> get_inbox_high() const { return m_fft.get_inbox_high(); }
  Vec3<int> get_outbox_low() const { return m_fft.get_outbox_low(); }
  Vec3<int> get_outbox_high() const { return m_fft.get_outbox_high(); }

protected:
  virtual void step(double dt) = 0;
  virtual void initialize(double dt) = 0;
  virtual std::vector<double> &get_field() = 0;
};
} // namespace pfc
