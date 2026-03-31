// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_model.hpp
 * @brief Tungsten Phase Field Crystal (PFC) model implementation for HIP
 *
 * @details
 * This file implements the HIP version of the Tungsten PFC model.
 * It uses DataBuffer<HipTag, T> for GPU memory management and HIP kernels
 * for element-wise operations.
 *
 * @see tungsten/cuda/tungsten_model.hpp for the CUDA version
 * @see tungsten_params.hpp for model parameters
 */

#ifndef TUNGSTEN_HIP_MODEL_HPP
#define TUNGSTEN_HIP_MODEL_HPP

#if !defined(OpenPFC_ENABLE_HIP)
#error                                                                              \
    "tungsten/hip/tungsten_model.hpp requires HIP support. Enable with -DOpenPFC_ENABLE_HIP=ON"
#endif

#include <hip/hip_runtime.h>
#include <memory>
#include <openpfc/frontend/utils/nancheck.hpp>
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/openpfc.hpp>
#include <openpfc/runtime/hip/fft_hip.hpp>
#include <tungsten/common/tungsten_ops.hpp>
#include <tungsten/common/tungsten_params.hpp>
#include <tungsten/common/tungsten_spectral.hpp>

/**
 * @brief Tungsten Phase Field Crystal model (HIP version)
 *
 * Implements the Tungsten PFC model with mean-field filtering and quasi-Gaussian
 * correlation functions for GPU execution on AMD GPUs via HIP/ROCm.
 *
 * @tparam RealType Real number type (float or double). Defaults to double.
 */
template <typename RealType = double> class TungstenHIP : public pfc::Model {
  using pfc::Model::Model;

private:
  std::unique_ptr<pfc::fft::FFT_HIP> m_hip_fft;

  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> filterMF;
  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> opL;
  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> opN;
  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> psiMF;
  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> psi;
  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> psiN;
  pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<RealType>> psiMF_F;
  pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<RealType>> psi_F;
  pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<RealType>> psiN_F;
  size_t mem_allocated = 0;

  pfc::RealField m_psi_cpu;
  bool m_cpu_buffer_valid;

  hipEvent_t kernel_done_event;
  hipEvent_t fft_ready_event;

public:
  TungstenParams params;

  void set_hip_fft(const pfc::Decomposition &decomp, int rank) {
    auto options = heffte::default_options<heffte::backend::rocfft>();
    auto r2c_dir = 0;
    auto fft_layout = pfc::fft::layout::create(decomp, r2c_dir);

    auto inbox = pfc::fft::layout::get_real_box(fft_layout, rank);
    auto outbox = pfc::fft::layout::get_complex_box(fft_layout, rank);
    auto r2c_direction = pfc::fft::layout::get_r2c_direction(fft_layout);
    auto comm = MPI_COMM_WORLD;

    using fft_r2c_hip = heffte::fft3d_r2c<heffte::backend::rocfft>;
    fft_r2c_hip fft_hip(inbox, outbox, r2c_direction, comm, options);

    m_hip_fft = std::unique_ptr<pfc::fft::FFT_HIP>(
        new pfc::fft::FFT_HIP(std::move(fft_hip)));
  }

  explicit TungstenHIP(pfc::FFT &fft, const pfc::World &world)
      : pfc::Model(fft, world), m_cpu_buffer_valid(false) {
    hipEventCreate(&kernel_done_event);
    hipEventCreate(&fft_ready_event);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto decomp = pfc::decomposition::create(get_world(), size);
    set_hip_fft(decomp, rank);
  }

  pfc::fft::FFT_HIP &get_hip_fft() {
    if (!m_hip_fft) {
      throw std::runtime_error("HIP FFT not set. Call set_hip_fft() first.");
    }
    return *m_hip_fft;
  }

  void allocate() {
    auto &fft = get_hip_fft();
    auto size_inbox = fft.size_inbox();
    auto size_outbox = fft.size_outbox();

    filterMF = pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(size_outbox);
    opL = pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(size_outbox);
    opN = pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(size_outbox);

    psi = pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(size_inbox);
    psiMF = pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(size_inbox);
    psiN = pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(size_inbox);

    psi_F = pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<RealType>>(
        size_outbox);
    psiMF_F = pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<RealType>>(
        size_outbox);
    psiN_F = pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<RealType>>(
        size_outbox);

    m_psi_cpu.resize(size_inbox);
    m_cpu_buffer_valid = false;

    pfc::add_field(*this, "psi", m_psi_cpu);

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

  void sync_gpu_to_cpu() {
    if (!m_cpu_buffer_valid) {
      m_psi_cpu = psi.to_host();
      m_cpu_buffer_valid = true;
    }
  }

  void sync_cpu_to_gpu() {
    psi.copy_from_host(m_psi_cpu);
    m_cpu_buffer_valid = false;
  }

  void prepare_operators(double dt) {
    auto &fft = get_hip_fft();
    auto &world = get_world();
    auto [Lx, Ly, Lz] = pfc::world::get_size(world);

    auto outbox = pfc::fft::get_outbox(fft);
    auto low = outbox.low;
    auto high = outbox.high;

    auto [fx, fy, fz] = pfc::fft::kspace::k_frequency_scaling(world);

    const auto op_params = tungsten::spectral::make_operator_params(params);

    auto size_outbox = fft.size_outbox();

    pfc::core::DataBuffer<pfc::backend::CpuTag, RealType> filterMF_cpu(size_outbox);
    pfc::core::DataBuffer<pfc::backend::CpuTag, RealType> opL_cpu(size_outbox);
    pfc::core::DataBuffer<pfc::backend::CpuTag, RealType> opN_cpu(size_outbox);

    int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {

          double ki = pfc::fft::kspace::k_component(i, Lx, fx);
          double kj = pfc::fft::kspace::k_component(j, Ly, fy);
          double kk = pfc::fft::kspace::k_component(k, Lz, fz);

          double kLap = pfc::fft::kspace::k_laplacian_value(ki, kj, kk);

          auto m = tungsten::spectral::operators_for_mode(kLap, dt, op_params);
          filterMF_cpu[idx] = static_cast<RealType>(m.filterMF);
          opL_cpu[idx] = static_cast<RealType>(m.opL);
          opN_cpu[idx] = static_cast<RealType>(m.opN);

          idx += 1;
        }
      }
    }

    CHECK_AND_ABORT_IF_NANS(opL_cpu.as_vector());
    CHECK_AND_ABORT_IF_NANS(opN_cpu.as_vector());

    filterMF.copy_from_host(filterMF_cpu.to_host());
    opL.copy_from_host(opL_cpu.to_host());
    opN.copy_from_host(opN_cpu.to_host());
  }

  void initialize(double dt) override {
    allocate();
    prepare_operators(dt);
  }

  void step(double t) override {
    (void)t;

    auto &fft = get_hip_fft();

    hipEventSynchronize(kernel_done_event);
    fft.forward(psi, psi_F);

    tungsten::ops::multiply_complex_real<pfc::backend::HipTag, RealType>(
        psi_F, filterMF, psiMF_F);
    hipEventRecord(kernel_done_event, 0);

    hipEventSynchronize(kernel_done_event);
    fft.backward(psiMF_F, psiMF);

    double p3_bar = params.get_p3_bar();
    double p4_bar = params.get_p4_bar();
    double q3_bar = params.get_q3_bar();
    double q4_bar = params.get_q4_bar();

    tungsten::ops::compute_nonlinear<pfc::backend::HipTag, RealType>(
        psi, psiMF, p3_bar, p4_bar, q3_bar, q4_bar, psiN);

    double stabP = params.get_stabP();
    if (stabP != 0.0) {
      tungsten::ops::apply_stabilization<pfc::backend::HipTag, RealType>(
          psiN, psi, stabP, psiN);
    }
    hipEventRecord(kernel_done_event, 0);

    hipEventSynchronize(kernel_done_event);
    fft.forward(psiN, psiN_F);

    tungsten::ops::apply_time_integration<pfc::backend::HipTag, RealType>(
        psi_F, psiN_F, opL, opN, psi_F);
    hipEventRecord(kernel_done_event, 0);

    hipEventSynchronize(kernel_done_event);
    fft.backward(psi_F, psi);
  }

  ~TungstenHIP() {
    hipEventDestroy(kernel_done_event);
    hipEventDestroy(fft_ready_event);
  }

  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> &get_psi() { return psi; }
  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> &get_psiMF() {
    return psiMF;
  }

  void prepare_for_field_modifiers() { sync_gpu_to_cpu(); }

  void finalize_after_field_modifiers() { sync_cpu_to_gpu(); }

  pfc::RealField &get_psi_for_writer() {
    sync_gpu_to_cpu();
    return m_psi_cpu;
  }
};

using TungstenHIP_double = TungstenHIP<double>;
using TungstenHIP_float = TungstenHIP<float>;

#endif // TUNGSTEN_HIP_MODEL_HPP
