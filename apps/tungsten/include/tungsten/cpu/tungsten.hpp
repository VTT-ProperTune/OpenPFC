// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_HPP
#define TUNGSTEN_HPP

#include <iostream>
#include <openpfc/frontend/ui/ui.hpp>
#include <tungsten/common/tungsten_input.hpp>
#include <tungsten/cpu/tungsten_model.hpp>

/*
Sometimes, one need to get access to the simulator during stepping. This can be
done by overriding the following function. The default implementation just runs
m.step(t) so if there's no need to access the simulator, it's not necessary to
override this function.
*/
void step(pfc::Simulator &s, Tungsten &m) {
#ifdef TUNGSTEN_DEBUG
  if (m.is_rank0()) std::cout << "Performing Tungsten step" << std::endl;
#endif
  double t = pfc::time::current(pfc::get_time(s));
  m.step(t);
  // perform some extra logic after the step, which can access both simulator
  // and model
}

#endif // TUNGSTEN_HPP
