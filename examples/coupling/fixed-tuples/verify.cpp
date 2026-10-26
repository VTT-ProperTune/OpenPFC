// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "verify.hpp"
#include <cstdlib>
#include <iostream>
#include <new>
#if defined(_OPENMP)
#include <omp.h>
#endif
namespace {
std::atomic<bool> observing{false};
std::atomic<std::size_t> allocation_requests{0};
void observe(bool enabled) { observing.store(enabled); }
void count() {
  if (observing.load()) allocation_requests.fetch_add(1);
}
} // namespace
// Teaching-test observer: successful C++ new requests, not malloc/runtime pools.
void *operator new(std::size_t n) {
  if (void *p = std::malloc(n ? n : 1)) {
    count();
    return p;
  }
  throw std::bad_alloc();
}
void *operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
void *operator new(std::size_t n, std::align_val_t align) {
  void *p = nullptr;
  if (posix_memalign(&p, std::size_t(align), n ? n : 1)) throw std::bad_alloc();
  count();
  return p;
}
void *operator new[](std::size_t n, std::align_val_t a) {
  return ::operator new(n, a);
}
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
#if defined(_OPENMP)
  omp_set_num_threads(1); // deterministic invocation/allocation check
#endif
  try {
    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    coupled::require(ranks == 1, "teaching demonstrator requires one MPI rank");
    coupled::verify_model();
    auto proof = coupled::verify_fd(32, observe);
    coupled::require(allocation_requests == 0,
                     "C++ allocation in warmed stage loop");
    std::cout << "FD6 residual=" << proof.residual
              << " stage_error=" << proof.stage_error
              << " C++ allocation_requests=" << allocation_requests << '\n';
#if defined(OpenPFC_ENABLE_HEFFTE)
    std::cout << "spectral residual=" << coupled::verify_spectral() << '\n';
#endif
  } catch (const std::exception &e) {
    observe(false);
    std::cerr << e.what() << '\n';
    result = 1;
  }
  MPI_Finalize();
  return result;
}
