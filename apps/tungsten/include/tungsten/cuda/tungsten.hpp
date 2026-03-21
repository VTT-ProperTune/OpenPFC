// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_CUDA_HPP
#define TUNGSTEN_CUDA_HPP

#if !defined(OpenPFC_ENABLE_CUDA)
#error                                                                              \
    "tungsten/cuda/tungsten.hpp requires CUDA support. Enable with -DOpenPFC_ENABLE_CUDA=ON"
#endif

#include <openpfc/frontend/ui/ui.hpp>
#include <tungsten/common/tungsten_input.hpp>
#include <iostream>
#include <tungsten/cuda/tungsten_model.hpp>

/*
Sometimes, one need to get access to the simulator during stepping. This can be
done by overriding the following function. The default implementation just runs
m.step(t) so if there's no need to access the simulator, it's not necessary to
override this function.
*/
template <typename RealType>
void step(pfc::Simulator &s, TungstenCUDA<RealType> &m) {
#ifdef TUNGSTEN_DEBUG
  if (m.is_rank0())
    std::cout << "Performing Tungsten CUDA step" << std::endl;
#endif
  double t = s.get_time().get_current();
  m.step(t);
  // perform some extra logic after the step, which can access both simulator
  // and model
}

#endif // TUNGSTEN_CUDA_HPP
