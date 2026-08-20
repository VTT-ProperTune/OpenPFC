// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file simulation_driver.hpp
 * @brief Thin time loop over `Time` plus a physics `step` (M10).
 *
 * @details
 * Same ordering as `Simulator::step` / ETD session `run()`:
 *
 * 1. while not done
 * 2. if increment is 0: `on_start`, then `on_save` when `do_save()`
 * 3. `Time::next()`
 * 4. `apply_conditions`
 * 5. `step(current time)`
 * 6. `on_save` when `do_save()`
 *
 * Gen-1 `Simulator` stays for A1/A2 until M12. This driver does not own
 * writers or checkpoints; callers pass those as hooks.
 */

#include <utility>

#include <openpfc/kernel/simulation/time.hpp>

namespace pfc {
class SimulationState;
}

namespace pfc::sim {

struct NoopHook {
  template <class... Args> constexpr void operator()(Args &&...) const noexcept {}
};

/**
 * Drive @p time to completion. @p step is `void(double t)` (accepted time
 * after `next()`). Optional hooks take `Time &` / `const Time &`.
 */
template <class Step, class OnStart = NoopHook, class Apply = NoopHook,
          class OnSave = NoopHook>
void run(Time &time, Step &&step, OnStart &&on_start = {}, Apply &&apply = {},
         OnSave &&on_save = {}) {
  while (!pfc::time::done(time)) {
    if (pfc::time::increment(time) == 0) {
      on_start(time);
      if (pfc::time::do_save(time)) {
        on_save(time);
      }
    }
    pfc::time::next(time);
    apply(time);
    step(pfc::time::current(time));
    if (pfc::time::do_save(time)) {
      on_save(time);
    }
  }
}

/**
 * Non-owning bundle of `Time` plus optional `SimulationState`. Call `run`
 * with the same hook pack as the free function.
 */
class SimulationDriver {
public:
  explicit SimulationDriver(Time &time, SimulationState *state = nullptr) noexcept
      : m_time(&time), m_state(state) {}

  template <class Step, class OnStart = NoopHook, class Apply = NoopHook,
            class OnSave = NoopHook>
  void run(Step &&step, OnStart &&on_start = {}, Apply &&apply = {},
           OnSave &&on_save = {}) {
    pfc::sim::run(*m_time, std::forward<Step>(step), std::forward<OnStart>(on_start),
                  std::forward<Apply>(apply), std::forward<OnSave>(on_save));
  }

  [[nodiscard]] Time &time() noexcept { return *m_time; }
  [[nodiscard]] const Time &time() const noexcept { return *m_time; }
  [[nodiscard]] SimulationState *state() noexcept { return m_state; }
  [[nodiscard]] const SimulationState *state() const noexcept { return m_state; }

private:
  Time *m_time{};
  SimulationState *m_state{};
};

} // namespace pfc::sim
