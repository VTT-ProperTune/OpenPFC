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
  cout << "Starting simulation at time " << time.get_t0() << ", with timesteps "
       << time.get_dt() << endl;
  print_status(time);
  while (!time.done()) {
    time.next();
    print_status(time);
  }
  return 0;
}
