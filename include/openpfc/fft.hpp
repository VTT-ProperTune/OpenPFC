#ifndef PFC_FFT_HPP
#define PFC_FFT_HPP

#include "decomposition.hpp"

#include <heffte.h>
#include <iostream>
#include <mpi.h>

namespace pfc {

/**
 * @brief FFT class for performing forward and backward Fast Fourier Transformations.
 */
class FFT {

private:
  const Decomposition m_decomposition;                  /**< The Decomposition object. */
  const heffte::fft3d_r2c<heffte::backend::fftw> m_fft; /**< HeFFTe FFT object. */
  std::vector<std::complex<double>> m_wrk;              /**< Workspace vector for FFT computations. */
  double m_fft_time = 0.0;                              /**< Recorded FFT computation time. */

public:
  /**
   * @brief Constructs an FFT object with the given Decomposition and MPI communicator.
   *
   * @param decomposition The Decomposition object defining the domain decomposition.
   * @param comm The MPI communicator for parallel computations (default: MPI_COMM_WORLD).
   * @param plan_options Optional plan options for configuring the FFT behavior (default: HeFFTe default options).
   */
  FFT(const Decomposition &decomposition, MPI_Comm comm = MPI_COMM_WORLD,
      heffte::plan_options plan_options = heffte::default_options<heffte::backend::fftw>())
      : m_decomposition(decomposition),
        m_fft({m_decomposition.inbox, m_decomposition.outbox, m_decomposition.r2c_direction, comm, plan_options}),
        m_wrk(std::vector<std::complex<double>>(m_fft.size_workspace())){};

  /**
   * @brief Performs the forward FFT transformation.
   *
   * @param in Input vector of real values.
   * @param out Output vector of complex values.
   */
  void forward(std::vector<double> &in, std::vector<std::complex<double>> &out) {
    m_fft_time -= MPI_Wtime();
    m_fft.forward(in.data(), out.data(), m_wrk.data());
    m_fft_time += MPI_Wtime();
  };

  /**
   * @brief Performs the backward (inverse) FFT transformation.
   *
   * @param in Input vector of complex values.
   * @param out Output vector of real values.
   */
  void backward(std::vector<std::complex<double>> &in, std::vector<double> &out) {
    m_fft_time -= MPI_Wtime();
    m_fft.backward(in.data(), out.data(), m_wrk.data(), heffte::scale::full);
    m_fft_time += MPI_Wtime();
  };

  /**
   * @brief Resets the recorded FFT computation time to zero.
   */
  void reset_fft_time() { m_fft_time = 0.0; }

  /**
   * @brief Returns the recorded FFT computation time.
   *
   * @return The FFT computation time in seconds.
   */
  double get_fft_time() const { return m_fft_time; }

  /**
   * @brief Returns the associated Decomposition object.
   *
   * @return Reference to the Decomposition object.
   */
  const Decomposition &get_decomposition() { return m_decomposition; }

  /**
   * @brief Returns the size of the inbox used for FFT computations.
   *
   * @return Size of the inbox.
   */
  size_t size_inbox() const { return m_fft.size_inbox(); }

  /**
   * @brief Returns the size of the outbox used for FFT computations.
   *
   * @return Size of the outbox.
   */
  size_t size_outbox() const { return m_fft.size_outbox(); }

  /**
   * @brief Returns the size of the workspace used for FFT computations.
   *
   * @return Size of the workspace.
   */
  size_t size_workspace() const { return m_fft.size_workspace(); }
};

} // namespace pfc

#endif
