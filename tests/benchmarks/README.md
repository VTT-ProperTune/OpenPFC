<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Benchmarks

This directory contains **performance benchmarks** for OpenPFC. Benchmarks measure execution time, memory usage, and scaling characteristics to prevent performance regressions.

## Purpose

- **Performance tracking**: Monitor performance over time
- **Regression detection**: Catch performance degradations early
- **Optimization guidance**: Identify bottlenecks and hot paths
- **Scaling validation**: Verify parallel scaling characteristics
- **HPC readiness**: Ensure performance suitable for HPC systems

## What Belongs Here

Benchmarks should measure:

- FFT performance (single/multi-rank)
- Time integration performance
- Field operations (transformations, arithmetic)
- Memory usage patterns
- Weak/strong scaling with MPI
- GPU performance (when GPU support added)

## What Doesn't Belong Here

- Correctness tests → use `tests/unit/` or `tests/integration/`
- Functional validation → use other test directories
- Exploratory performance testing → use `examples/` or standalone scripts

## Running Benchmarks

```bash
# All benchmarks
./tests/openpfc-tests "[benchmark]"

# Specific benchmark
./tests/openpfc-tests "[fft][benchmark]"

# With performance output
./tests/openpfc-tests "[benchmark]" --reporter console
```

For accurate results:

- Use Release build: `cmake -B build -DCMAKE_BUILD_TYPE=Release`
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

## Current Status

**Empty** - Benchmarks will be added as performance-critical paths are identified and optimized. Initial focus is on correctness (unit and integration tests).
