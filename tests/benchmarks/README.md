<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Benchmarks

This directory contains performance benchmarks for OpenPFC. Benchmarks measure execution time, memory usage, and scaling characteristics to prevent performance regressions.

## Purpose

- Performance tracking: Monitor performance over time
- Regression detection: Catch performance degradations early
- Optimization guidance: Identify bottlenecks and hot paths
- Scaling validation: Verify parallel scaling characteristics
- HPC readiness: Ensure performance suitable for HPC systems

## What Belongs Here

Benchmarks should measure:

- FFT performance (single/multi-rank)
- Time integration performance
- Field operations (transformations, arithmetic)
- Memory usage patterns
- Weak/strong scaling with MPI
- GPU kernel and device-halo microbenchmarks when those sources are enabled

## What Doesn't Belong Here

- Correctness tests → use `tests/unit/` or `tests/integration/`
- Functional validation → use other test directories
- Exploratory performance testing → use `examples/` or standalone scripts

## Running Benchmarks

Note: Benchmark sources are not built unless `OpenPFC_BUILD_BENCHMARKS=ON` (default is OFF in CMake). CI keeps build times down by excluding benchmarks from `ctest` via `--exclude-regex "benchmark"`; run them locally when you need timings.

Configure, build `openpfc-tests`, then run from the build directory:

```bash
cmake -S . -B build \
 -DCMAKE_BUILD_TYPE=Release \
 -DOpenPFC_BUILD_BENCHMARKS=ON
cmake --build build --target openpfc-tests
cd build

./tests/openpfc-tests "[benchmark]"

# Specific benchmark category
./tests/openpfc-tests "[world][benchmark]"

# With detailed output
./tests/openpfc-tests "[benchmark]" --reporter console

# Run all tests EXCEPT benchmarks (what CI does)
./tests/openpfc-tests '~[benchmark]'
```

For accurate results:

- Run on dedicated nodes (no interference)
- Use representative problem sizes
- Run multiple iterations for statistical significance

## Writing Benchmarks

1. Tag tests with `[benchmark]` (and component tags)
2. Use Catch2's `BENCHMARK` macro for microbenchmarks
3. Report timing results in test output
4. Use realistic problem sizes
5. Document expected performance characteristics
6. Consider both CPU and memory performance

Example:

```cpp
TEST_CASE("FFT performance", "[fft][benchmark]") {
 // Setup
 auto fft = create_fft(large_size);
 
 BENCHMARK("forward transform") {
 return fft.forward();
 };
}
```

## Current Benchmarks

### World and coordinate transforms (`bench_world_coords.cpp`)

Microbenchmarks for core coordinate transformation operations used in hot
paths (field initialization loops, spatial operations). Compare Release
builds on a quiet machine; do not treat checked-in comments or this README
as a performance baseline (see `tests/baselines/BASELINES.md` for that).

Benchmark categories:

1. Coordinate transformations — `to_coords()`, `to_indices()`, round-trip
2. World accessors — `get_spacing()`, `get_origin()`, `get_size()`
3. Loop-based usage — full-grid conversion and Gaussian initialization
4. Zero-cost abstraction check — World helpers vs a manual arithmetic baseline
5. Memory access — construction, copy, equality

```bash
./tests/openpfc-tests "[world][benchmark]"
./tests/openpfc-tests "[benchmark]" --reporter console
```

## Performance Expectations

For accurate performance measurements:

1. Use Release build:

 ```bash
 cmake -B build-release -DCMAKE_BUILD_TYPE=Release
 cmake --build build-release
 ./build-release/tests/openpfc-tests "[benchmark]"
 ```

2. Measure in Release (or RelWithDebInfo). Debug timings are not comparable
   to production.

3. Run on a dedicated system (no background processes)
4. Use representative problem sizes
5. Multiple iterations for statistical significance

## Adding New Benchmarks

When adding benchmarks:

1. Focus on hot paths (inner loops, frequently called functions)
2. Use realistic data (prevent compiler optimizations with `volatile`)
3. Document expected performance in comments
4. Compare to baseline (manual calculation)
5. Tag appropriately: `[component][benchmark]`

Example:

```cpp
TEST_CASE("FFT performance", "[fft][benchmark]") {
 auto fft = create_fft({128, 128, 128});
 
 BENCHMARK("Forward transform") {
 return fft.forward();
 };
}
```
