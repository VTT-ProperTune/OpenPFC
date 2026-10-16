// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file device_advance.cpp
 * @brief Host-side owner of the selected HeFFTe device velocity step.
 *
 * CUDA and HIP share this owner; kernels remain in their installed library.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL) && !defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
#error "incompressible advance requires a spectral device backend"
#endif

#include <flow/device_session.hpp>
#include <flow/taylor_green.hpp>

#include <iostream>
#include <optional>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc/runtime/gpu/incompressible_device.hpp>

namespace flow {

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
using SessionSpace = pfc::HIPSpace;
#else
using SessionSpace = pfc::CUDASpace;
#endif

struct DeviceSession {
  // The stack is not movable. Construct it in place, and destroy the
  // velocity buffers before the FFT they were planned against.
  std::optional<pfc::sim::stacks::GPUSpectralStack<SessionSpace>> stack;
  pfc::incompressible::DeviceVelocity<SessionSpace> velocity;
};

void destroy_device_session(DeviceSession *session) { delete session; }

DeviceSessionPtr start_device_session(State &state, int rank, int nproc) {
  auto session = DeviceSessionPtr(new DeviceSession());
  auto domain = pfc::domain::create(
      pfc::GridSize({state.n[0], state.n[1], state.n[2]}),
      pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({state.spacing[0], state.spacing[1], state.spacing[2]}));
  session->stack.emplace(std::move(domain), rank, nproc, MPI_COMM_WORLD);
  pfc::incompressible::prepare_device_velocity(session->velocity,
                                               session->stack->fft(), state.n,
                                               state.spacing, state.nu, state.dt);
  pfc::incompressible::upload_device_velocity(session->velocity, state.u, state.v,
                                              state.w);
  GPU_CHECK(pfc::gpuDeviceSynchronize());

  unsigned long long local = static_cast<unsigned long long>(
      pfc::incompressible::device_velocity_bytes(session->velocity));
  unsigned long long total = 0;
  MPI_Reduce(&local, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "incompressible_flow device_bytes=" << total << std::endl;
  }
  return session;
}

void step_device_session(DeviceSession &session, DeviceProfile *profile) {
  pfc::incompressible::StepProfile step;
  pfc::incompressible::step_device_velocity(session.velocity, session.stack->fft(),
                                            true,
                                            profile == nullptr ? nullptr : &step);
  if (profile == nullptr) return;
  profile->spectral_s += step.spectral_s;
  profile->fft_s += step.fft_s;
  profile->nonlinear_s += step.nonlinear_s;
  profile->steps += 1;
}

void finish_device_session(DeviceSession &session, State &state) {
  GPU_CHECK(pfc::gpuDeviceSynchronize());
  pfc::incompressible::download_device_velocity(session.velocity, state.u, state.v,
                                                state.w);
}

} // namespace flow
