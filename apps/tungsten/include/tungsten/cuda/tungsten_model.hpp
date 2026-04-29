// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_model.hpp
 * @brief Tungsten Phase Field Crystal (PFC) model implementation for CUDA
 *
 * @details
 * This file implements the CUDA version of the Tungsten PFC model.
 * It uses DataBuffer<CudaTag, T> for GPU memory management and CUDA kernels
 * for element-wise operations.
 *
 * This is a separate implementation from the CPU version
 * (tungsten/cpu/tungsten_model.hpp) to allow incremental development and testing.
 *
 * @see tungsten/cpu/tungsten_model.hpp for the CPU version
 * @see tungsten_params.hpp for model parameters
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef TUNGSTEN_CUDA_MODEL_HPP
#define TUNGSTEN_CUDA_MODEL_HPP

#if !defined(OpenPFC_ENABLE_CUDA)
#error                                                                              \
    "tungsten/cuda/tungsten_model.hpp requires CUDA support. Enable with -DOpenPFC_ENABLE_CUDA=ON"
#endif

#include <cuda.h> // For CUDA events
#include <cuda_runtime.h>
#include <memory>
#include <openpfc/frontend/utils/nancheck.hpp>
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/openpfc.hpp>
#include <openpfc/runtime/cuda/fft_cuda.hpp>
#include <tungsten/common/tungsten_ops.hpp>
#include <tungsten/common/tungsten_params.hpp>
#include <tungsten/common/tungsten_spectral.hpp>

/**
 * @brief Tungsten Phase Field Crystal model (CUDA version)
 *
 * Implements the Tungsten PFC model with mean-field filtering and quasi-Gaussian
 * correlation functions for GPU execution. Uses CUDA for all computations.
 *
 * @tparam RealType Real number type (float or double). Defaults to double.
 *                  Use float for better GPU performance and lower memory usage.
 *
 * @note All parameters are accessed via `params.get_*()` getters
 * @note Parameters are set via `params.set_*()` setters
 * @note This version uses GPU memory (DataBuffer<CudaTag, T>)
 *
 * @example Double precision
 * @code
 * TungstenCUDA<double> model(world);
 * @endcode
 *
 * @example Single precision (better GPU performance)
 * @code
 * TungstenCUDA<float> model(world);
 * @endcode
 */
template <typename RealType = double> class TungstenCUDA : public pfc::Model {
private:
  // CUDA FFT (precision determined by data types, not template parameter)
  // Stored as unique_ptr, but FFT_Impl cannot be moved due to const members
  // So we construct it in place via set_cuda_fft(decomp, rank)
  std::unique_ptr<pfc::fft::FFT_CUDA> m_cuda_fft;

  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>
      filterMF; ///< Mean-field filter in Fourier space
  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>
      opL; ///< Linear operator: exp(L·dt)
  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>
      opN; ///< Nonlinear operator: (exp(L·dt) - 1) / L
  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>
      psiMF; ///< Mean-field filtered density
  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType> psi;  ///< Density field
  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType> psiN; ///< Nonlinear term
  pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<RealType>>
      psiMF_F; ///< Mean-field in Fourier space
  pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<RealType>>
      psi_F; ///< Density in Fourier space
  pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<RealType>>
      psiN_F;               ///< Nonlinear term in Fourier space
  size_t mem_allocated = 0; ///< Memory allocated (for debugging)

  // CPU-side buffers for FieldModifiers and VTKWriter
  // These mirror GPU data and are synchronized when needed
  pfc::RealField m_psi_cpu; ///< CPU copy of psi for FieldModifiers/VTKWriter
  bool m_cpu_buffer_valid;  ///< Whether CPU buffer is up-to-date

  // CUDA events for non-blocking synchronization
  cudaEvent_t kernel_done_event; ///< Event to track kernel completion
  cudaEvent_t fft_ready_event;   ///< Event to track FFT readiness

public:
  /**
   * @brief Model parameters
   *
   * Access parameters using getters: `params.get_n0()`, `params.get_T()`, etc.
   * Set parameters using setters: `params.set_n0(value)`, `params.set_T(value)`,
   * etc.
   */
  TungstenParams params;

  /**
   * @brief Set the CUDA FFT object from decomposition and rank
   *
   * Constructs cuFFT-backed HeFFTe FFT state and wraps it in FFT_CUDA via
   * std::make_unique.
   *
   * @param decomp Decomposition object
   * @param rank MPI rank in `mpi_comm()` (same communicator as base `Model`)
   */
  void set_cuda_fft(const pfc::Decomposition &decomp, int rank) {
    auto options = heffte::default_options<heffte::backend::cufft>();
    auto r2c_dir = 0;
    auto fft_layout = pfc::fft::layout::create(decomp, r2c_dir);

    auto inbox = pfc::fft::layout::get_real_box(fft_layout, rank);
    auto outbox = pfc::fft::layout::get_complex_box(fft_layout, rank);
    auto r2c_direction = pfc::fft::layout::get_r2c_direction(fft_layout);
    const MPI_Comm comm = mpi_comm();

    // Create cuFFT-based FFT directly
    using fft_r2c_cuda = heffte::fft3d_r2c<heffte::backend::cufft>;
    fft_r2c_cuda fft_cuda(inbox, outbox, r2c_direction, comm, options);

    m_cuda_fft = std::make_unique<pfc::fft::FFT_CUDA>(std::move(fft_cuda));
  }

  /**
   * @brief Construct TungstenCUDA and set up CUDA FFT
   *
   * @param fft CPU FFT reference (used by base Model)
   * @param world Simulation domain
   */
  explicit TungstenCUDA(pfc::FFT &fft, const pfc::World &world,
                        MPI_Comm mpi_comm = MPI_COMM_WORLD)
      : pfc::Model(fft, world, mpi_comm), m_cpu_buffer_valid(false) {
    // Create CUDA events for non-blocking synchronization
    cudaEventCreate(&kernel_done_event);
    cudaEventCreate(&fft_ready_event);
    // Initialize CUDA FFT using the same communicator as the base Model / App
    // session
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(mpi_comm(), &rank);
    MPI_Comm_size(mpi_comm(), &size);
    auto decomp = pfc::decomposition::create(get_world(), size);
    set_cuda_fft(decomp, rank);
  }

  /**
   * @brief Get the CUDA FFT object
   *
   * @return Reference to the CUDA FFT object
   */
  pfc::fft::FFT_CUDA &get_cuda_fft() {
    if (!m_cuda_fft) {
      throw std::runtime_error("CUDA FFT not set. Call set_cuda_fft() first.");
    }
    return *m_cuda_fft;
  }

  /**
   * @brief Allocate memory for fields and operators
   *
   * Resizes all field arrays based on FFT inbox/outbox sizes.
   * Also allocates CPU-side buffers for FieldModifiers and VTKWriter.
   */
  void allocate() {
    auto &fft = get_cuda_fft();
    auto size_inbox = fft.size_inbox();
    auto size_outbox = fft.size_outbox();

    // Operators are only half size due to the symmetry of Fourier space
    filterMF = pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(size_outbox);
    opL = pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(size_outbox);
    opN = pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(size_outbox);

    // Real-space fields
    psi = pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(size_inbox);
    psiMF = pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(size_inbox);
    psiN = pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(size_inbox);

    // Fourier-space fields (suffix F means in Fourier space)
    psi_F = pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<RealType>>(
        size_outbox);
    psiMF_F = pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<RealType>>(
        size_outbox);
    psiN_F = pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<RealType>>(
        size_outbox);

    // Allocate CPU-side buffer for FieldModifiers and VTKWriter
    m_psi_cpu.resize(size_inbox);
    m_cpu_buffer_valid = false;

    // Register CPU buffer with Model base class for FieldModifier access
    // FieldModifiers will modify m_psi_cpu, then we sync back to GPU
    pfc::add_field(*this, "psi", m_psi_cpu);

    // Track memory usage
    mem_allocated = 0;
    mem_allocated += filterMF.size() * sizeof(RealType);
    mem_allocated += opL.size() * sizeof(RealType);
    mem_allocated += opN.size() * sizeof(RealType);
    mem_allocated += psi.size() * sizeof(RealType);
    mem_allocated += psiMF.size() * sizeof(RealType);
    mem_allocated += psiN.size() * sizeof(RealType);
    mem_allocated += psi_F.size() * sizeof(std::complex<RealType>);
    mem_allocated += psiMF_F.size() * sizeof(std::complex<RealType>);
    mem_allocated += psiN_F.size() * sizeof(std::complex<RealType>);
  }

  /**
   * @brief Sync GPU data to CPU buffer
   *
   * Copies psi from GPU to CPU buffer. Used before applying FieldModifiers
   * or writing VTK output.
   */
  void sync_gpu_to_cpu() {
    if (!m_cpu_buffer_valid) {
      m_psi_cpu = psi.to_host();
      m_cpu_buffer_valid = true;
    }
  }

  /**
   * @brief Sync CPU buffer to GPU
   *
   * Copies modified CPU buffer back to GPU. Called after FieldModifiers
   * have modified the CPU buffer.
   */
  void sync_cpu_to_gpu() {
    psi.copy_from_host(m_psi_cpu);
    m_cpu_buffer_valid = false; // GPU is now the source of truth
  }

  /**
   * @brief Precompute time integration operators in k-space
   *
   * Computes operators on CPU and transfers to GPU.
   * This is acceptable since operators are computed once during initialization.
   *
   * @param dt Time step size
   */
  void prepare_operators(double dt) {
    auto &fft = get_cuda_fft();
    auto &world = get_world();
    auto [Lx, Ly, Lz] = pfc::world::get_size(world);

    auto outbox = pfc::fft::get_outbox(fft);
    auto low = outbox.low;
    auto high = outbox.high;

    // Get frequency scaling factors using helper function
    auto [fx, fy, fz] = pfc::fft::kspace::k_frequency_scaling(world);

    const auto op_params = tungsten::spectral::make_operator_params(params);

    // Get FFT sizes
    auto size_outbox = fft.size_outbox();

    // Compute operators on CPU first
    pfc::core::DataBuffer<pfc::backend::CpuTag, RealType> filterMF_cpu(size_outbox);
    pfc::core::DataBuffer<pfc::backend::CpuTag, RealType> opL_cpu(size_outbox);
    pfc::core::DataBuffer<pfc::backend::CpuTag, RealType> opN_cpu(size_outbox);

    int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {

          // Compute wave vector components using helper function
          double ki = pfc::fft::kspace::k_component(i, Lx, fx);
          double kj = pfc::fft::kspace::k_component(j, Ly, fy);
          double kk = pfc::fft::kspace::k_component(k, Lz, fz);

          // Compute Laplacian operator -k² using helper function
          double kLap = pfc::fft::kspace::k_laplacian_value(ki, kj, kk);

          auto m = tungsten::spectral::operators_for_mode(kLap, dt, op_params);
          filterMF_cpu[idx] = static_cast<RealType>(m.filterMF);
          opL_cpu[idx] = static_cast<RealType>(m.opL);
          opN_cpu[idx] = static_cast<RealType>(m.opN);

          idx += 1;
        }
      }
    }

    CHECK_AND_ABORT_IF_NANS_MPI(opL_cpu.as_vector(), mpi_comm());
    CHECK_AND_ABORT_IF_NANS_MPI(opN_cpu.as_vector(), mpi_comm());

    // Transfer to GPU
    filterMF.copy_from_host(filterMF_cpu.to_host());
    opL.copy_from_host(opL_cpu.to_host());
    opN.copy_from_host(opN_cpu.to_host());
  }

  /**
   * @brief Initialize the model
   *
   * Allocates memory and precomputes operators. Must be called before time
   * stepping.
   *
   * @param dt Time step size
   */
  void initialize(double dt) override {
    allocate();
    prepare_operators(dt);
  }

  /**
   * @brief Perform one time step of the evolution equation
   *
   * Implements the exponential time integration scheme using GPU operations:
   * 1. Compute mean-field filtered density ψ_MF
   * 2. Calculate nonlinear term N[ψ, ψ_MF] in real space
   * 3. Transform nonlinear term to Fourier space
   * 4. Apply linear and nonlinear operators
   * 5. Transform back to real space
   *
   * @param t Current time (unused, kept for interface compatibility)
   */
  void step(double t) override {
    (void)t; // suppress compiler warning about unused parameter

    auto &fft = get_cuda_fft();

    // Step 1: Calculate mean-field density n_MF
    // Forward FFT: ψ → ψ̂
    // Wait for any previous operations using event (non-blocking if possible)
    cudaEventSynchronize(kernel_done_event);
    fft.forward(psi, psi_F);

    // Apply mean-field filter in Fourier space: ψ̂_MF = χ(k) · ψ̂
    // Uses GPU kernel via backend-agnostic operation (no sync - async launch)
    tungsten::ops::multiply_complex_real<pfc::backend::CudaTag, RealType>(
        psi_F, filterMF, psiMF_F);
    // Record event after kernel launch (kernels run on default stream)
    cudaEventRecord(kernel_done_event, 0);

    // Inverse FFT: ψ̂_MF → ψ_MF
    // Wait for kernel to complete using event
    cudaEventSynchronize(kernel_done_event);
    fft.backward(psiMF_F, psiMF);

    // Step 2: Calculate nonlinear part in real space
    // N[ψ, ψ_MF] = p̄₃ψ² + p̄₄ψ³ + q̄₃ψ_MF² + q̄₄ψ_MF³
    double p3_bar = params.get_p3_bar();
    double p4_bar = params.get_p4_bar();
    double q3_bar = params.get_q3_bar();
    double q4_bar = params.get_q4_bar();

    // Uses GPU kernel via backend-agnostic operation (no sync - async launch)
    tungsten::ops::compute_nonlinear<pfc::backend::CudaTag, RealType>(
        psi, psiMF, p3_bar, p4_bar, q3_bar, q4_bar, psiN);

    // Step 3: Apply stabilization factor if given
    double stabP = params.get_stabP();
    if (stabP != 0.0) {
      // Uses GPU kernel via backend-agnostic operation (no sync - async launch)
      tungsten::ops::apply_stabilization<pfc::backend::CudaTag, RealType>(
          psiN, psi, stabP, psiN);
    }
    // Record event after all kernels in this sequence complete
    cudaEventRecord(kernel_done_event, 0);

    // Step 4: Transform nonlinear term to Fourier space
    // Wait for kernels to complete using event
    cudaEventSynchronize(kernel_done_event);
    fft.forward(psiN, psiN_F);

    // Step 5: Apply exponential time integration in Fourier space
    // ψ̂(t+Δt) = L(k)·ψ̂(t) + N(k)·N̂[ψ, ψ_MF]
    // Uses GPU kernel via backend-agnostic operation (no sync - async launch)
    tungsten::ops::apply_time_integration<pfc::backend::CudaTag, RealType>(
        psi_F, psiN_F, opL, opN, psi_F);
    // Record event after kernel launch
    cudaEventRecord(kernel_done_event, 0);

    // Step 6: Transform back to real space
    // Wait for kernel to complete using event
    cudaEventSynchronize(kernel_done_event);
    fft.backward(psi_F, psi);

    // Note: NaN checking for GPU would require transferring data to CPU
    // Skipped for performance - can be enabled in debug builds if needed
  }

  // Note: Constructor defined earlier; avoid duplicate definitions.

  /**
   * @brief Destructor - cleans up CUDA events
   */
  ~TungstenCUDA() {
    cudaEventDestroy(kernel_done_event);
    cudaEventDestroy(fft_ready_event);
  }

  // Accessors for fields (for testing/debugging)
  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType> &get_psi() { return psi; }
  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType> &get_psiMF() {
    return psiMF;
  }

  /**
   * @brief Prepare for FieldModifier application
   *
   * Syncs GPU data to CPU buffer so FieldModifiers can access it.
   * Call this before applying initial/boundary conditions.
   */
  void prepare_for_field_modifiers() { sync_gpu_to_cpu(); }

  /**
   * @brief Finalize after FieldModifier application
   *
   * Syncs modified CPU buffer back to GPU.
   * Call this after applying initial/boundary conditions.
   */
  void finalize_after_field_modifiers() { sync_cpu_to_gpu(); }

  /**
   * @brief Get CPU copy of psi field for VTKWriter
   *
   * Syncs GPU to CPU and returns reference to CPU buffer.
   * Used by VTKWriter to write results.
   */
  pfc::RealField &get_psi_for_writer() {
    sync_gpu_to_cpu();
    return m_psi_cpu;
  }
};

// Type aliases for convenience
using TungstenCUDA_double = TungstenCUDA<double>;
using TungstenCUDA_float = TungstenCUDA<float>;

#endif // TUNGSTEN_CUDA_MODEL_HPP
