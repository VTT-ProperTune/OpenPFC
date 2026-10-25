// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>
#include <openpfc/kernel/simulation/time.hpp>

int main(int argc, char **argv) {
  try {
    const auto input = lesson::input(argc, argv);
    const double dt = input.value("dt", .13), end = input.value("end", 1.0);
    const double saveat = input.value("saveat", .2);
    lesson::positive(dt, "dt");
    lesson::positive(end, "end");
    lesson::positive(saveat, "saveat");
    if (end / std::min(dt, saveat) > 10000)
      throw std::invalid_argument("Too many intervals");
    pfc::Time time({0, end, dt}, saveat);
    auto output = lesson::csv("clock.csv");
    output << "event,increment,time,attempted,save\n";
    output << "initial,0,0,0," << time.do_save() << '\n';
    // start snippet reject
    time.begin_attempt(dt);
    const double first = time.get_attempted_dt();
    time.reject_attempt();
    time.increment_step_rejection();
    output << "rejected,0," << time.get_current() << ',' << first << ','
           << time.do_save() << '\n';
    // end snippet reject
    // start snippet clock
    while (!time.done()) {
      time.begin_attempt(dt);
      const double attempted = time.get_attempted_dt();
      // Advance the model by `attempted` here. This lesson isolates the clock.
      time.commit_attempt();
      time.increment_step_success();
      output << "accepted," << time.get_increment() << ',' << time.get_current()
             << ',' << attempted << ',' << time.do_save() << '\n';
    }
    // end snippet clock
    std::cout << "committed=" << time.get_increment()
              << " final=" << time.get_current() << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
