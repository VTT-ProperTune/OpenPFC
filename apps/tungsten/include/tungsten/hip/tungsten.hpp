// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_HIP_HPP
#define TUNGSTEN_HIP_HPP

#if !defined(OpenPFC_ENABLE_HIP)
#error                                                                              \
    "tungsten/hip/tungsten.hpp requires HIP support. Enable with -DOpenPFC_ENABLE_HIP=ON"
#endif

#include <openpfc/frontend/ui/ui.hpp>
#include <tungsten/common/tungsten_input.hpp>
#include <iostream>
#include <tungsten/hip/tungsten_model.hpp>

/*
Sometimes, one need to get access to the simulator during stepping. This can be
done by overriding the following function. The default implementation just runs
m.step(t) so if there's no need to access the simulator, it's not necessary to
override this function.
*/
template <typename RealType>
void step(pfc::Simulator &s, TungstenHIP<RealType> &m) {
#ifdef TUNGSTEN_DEBUG
  if (m.is_rank0())
    std::cout << "Performing Tungsten HIP step" << std::endl;
#endif
  double t = s.get_time().get_current();
  m.step(t);
}

#endif // TUNGSTEN_HIP_HPP
