#include <array>
#include <iostream>
#include <openpfc/mpi/timer.hpp>
#include <unistd.h>

void fft() {
  sleep(1);
}

enum { Total = 0, FFT = 1 };

int main() {
  std::array<pfc::mpi::timer, 2> timers;
  timers[Total].description("Total program run time");
  timers[FFT].description("Time used to FFT");

  timers[Total].tic(); // tic is called also when initializing

  for (int i = 0; i < 3; i++) {
    timers[FFT].tic();
    std::cout << "Starting fourier transform #" << i << std::endl;
    fft();
    std::cout << "Fourier transform #" << i << " done" << std::endl;
    timers[FFT].toc();
  }

  std::cout << "Finalizing program..." << std::endl;
  sleep(1);
  std::cout << "All done!" << std::endl;

  timers[Total].toc();

  std::cout << timers[0] << std::endl;
  std::cout << timers[1] << std::endl;
  return 0;
}
