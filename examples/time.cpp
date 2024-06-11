/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#include <iostream>
#include <openpfc/time.hpp>

using namespace std;
using namespace pfc;

void print_status(const Time &time) {
  cout << "Simulation " << time.get_current() << "/" << time.get_t1();
  if (time.do_save()) {
    cout << " (writing results every " << time.get_saveat() << ")";
  }
  cout << endl;
}

int main() {
  cout << std::fixed;
  cout.precision(3);
  Time time({0.0, 10.0, 1.0});
  time.set_saveat(2.0);
  cout << time << endl;
  cout << "Starting simulation at time " << time.get_t0() << ", with timesteps " << time.get_dt() << endl;
  print_status(time);
  while (!time.done()) {
    time.next();
    print_status(time);
  }
  return 0;
}
