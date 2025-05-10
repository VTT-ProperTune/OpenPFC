// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_FFT_HPP
#define PFC_FFT_HPP

#include "core/decomposition.hpp"
#include "openpfc/backends/heffte_adapter.hpp" // Ensure this is included for the conversion operator
#include "openpfc/core/world.hpp"

#include <heffte.h>
#include <iostream>
#include <mpi.h>

namespace pfc {

using heffte::box3d;
using pfc::types::Bool3;
using pfc::types::Int3;
using pfc::types::Real3;
using Box3D = heffte::box3d<int>; ///< Type alias for 3D integer box.

inline heffte::fft3d_r2c<heffte::backend::fftw>
make_fft(const Box3D &inbox, const Box3D &outbox, const int &r2c_direction,
         MPI_Comm comm, heffte::plan_options plan_options) {
  return heffte::fft3d_r2c<heffte::backend::fftw>(inbox, outbox, r2c_direction, comm,
                                                  plan_options);
}

/**
 * @brief FFT class for performing forward and backward Fast Fourier
 * Transformations.
 */
struct FFT {

  using ComplexVector = std::vector<std::complex<double>>;

  const Decomposition m_decomposition;        /**< The Decomposition object. */
  const heffte::box3d<int> m_inbox, m_outbox; /**< Local inbox and outbox boxes. */
  const int m_r2c_direction; /**< Real-to-complex symmetry direction. */
  const heffte::fft3d_r2c<heffte::backend::fftw> m_fft; /**< HeFFTe FFT object. */
  ComplexVector m_wrk;       /**< Workspace vector for FFT computations. */
  double m_fft_time = 0.0;   /**< Recorded FFT computation time. */
  const World &m_world;      /**< Reference to the World object. */
  heffte::box3d<int> domain; /**< Domain converted from World object. */

  /**
   * @brief Constructs an FFT object with the given Decomposition and MPI
   * communicator.
   *
   * @param decomposition The Decomposition object defining the domain
   * decomposition.
   * @param comm The MPI communicator for parallel computations.
   * @param plan_options Optional plan options for configuring the FFT behavior.
   * @param world The World object providing the domain size information.
   */
  FFT(const Decomposition &decomposition, MPI_Comm comm,
      heffte::plan_options plan_options, const World &world)
      : m_decomposition(decomposition), m_inbox(decomposition.m_inbox),
        m_outbox(decomposition.m_outbox), m_r2c_direction(0),
        m_fft(make_fft(m_inbox, m_outbox, m_r2c_direction, comm, plan_options)),
        m_wrk(std::vector<std::complex<double>>(m_fft.size_workspace())),
        m_world(world), domain(to_heffte_box(world)){
                            // Use to_heffte_box for conversion
                            // Explicit conversion
                        };

  /**
   * @brief Performs the forward FFT transformation.
   *
   * @param in Input vector of real values.
   * @param out Output vector of complex values.
   */
  void forward(const std::vector<double> &in,
               std::vector<std::complex<double>> &out) {
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
  void backward(const std::vector<std::complex<double>> &in,
                std::vector<double> &out) {
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

inline const Box3D &get_inbox(const FFT &fft) noexcept { return fft.m_inbox; }
inline const Box3D &get_outbox(const FFT &fft) noexcept { return fft.m_outbox; }
inline const Int3 &get_inbox_size(const FFT &fft) noexcept {
  return get_inbox(fft).size;
}
inline const Int3 &get_inbox_offset(const FFT &fft) noexcept {
  return get_inbox(fft).low;
}
inline const Int3 &get_outbox_size(const FFT &fft) noexcept {
  return get_outbox(fft).size;
}
inline const Int3 &get_outbox_offset(const FFT &fft) noexcept {
  return get_outbox(fft).low;
}

} // namespace pfc

#endif
