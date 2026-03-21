// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_fwd.hpp
 * @brief Forward declarations for simulation types (lighter includes)
 *
 * Use this header when you only need pointers/references to simulation types
 * (e.g. UI JSON stubs, field modifiers). For `Model` implementation details,
 * field registries, or `FFT` access, include `model.hpp` (or `openpfc.hpp`).
 */

#pragma once

namespace pfc {

class FieldModifier;
class Model;
class ResultsWriter;
class Simulator;
class Time;

void step(Simulator &s, Model &m);

} // namespace pfc
