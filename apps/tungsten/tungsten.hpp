// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_HPP
#define TUNGSTEN_HPP

#include "tungsten_input.hpp"
#include "tungsten_model.hpp"
#include <openpfc/ui.hpp>

using namespace pfc::ui;
using namespace std;

/*
Sometimes, one need to get access to the simulator during stepping. This can be
done by overriding the following function. The default implementation just runs
m.step(t) so if there's no need to access the simulator, it's not necessary to
override this function.
*/
void step(Simulator &s, Tungsten &m) {
#ifdef TUNGSTEN_DEBUG
  if (m.is_rank0()) cout << "Performing Tungsten step" << endl;
#endif
  double t = s.get_time().get_current();
  m.step(t);
  // perform some extra logic after the step, which can access both simulator
  // and model
}

#endif // TUNGSTEN_HPP
