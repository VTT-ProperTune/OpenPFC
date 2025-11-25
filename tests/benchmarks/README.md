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

**Note**: Benchmarks are **automatically excluded from CI** to keep build times fast. They must be run manually for performance validation.

```bash
# Using nix (recommended)
nix run .#benchmark

# Or directly with the test executable
./tests/openpfc-tests "[benchmark]"

# Specific benchmark category
./tests/openpfc-tests "[world][benchmark]"

# With detailed output
./tests/openpfc-tests "[benchmark]" --reporter console

# Run all tests EXCEPT benchmarks (what CI does)
./tests/openpfc-tests '~[benchmark]'
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

## Current Benchmarks

### World and Coordinate System (`bench_world_coords.cpp`)

Microbenchmarks for core coordinate transformation operations. These functions are used in hot paths (field initialization loops, spatial operations) and must be zero-cost abstractions.

**Benchmark Categories:**

1. **Coordinate Transformations** - Core mapping functions
   - `to_coords()`: Grid indices → physical coordinates (~400 ns)
   - `to_indices()`: Physical coordinates → grid indices (~400 ns)
   - Round-trip transformation (~750 ns)

2. **World Accessors** - Property access functions
   - `get_spacing()`, `get_origin()`, `get_size()` (~70-220 ns)
   - Verify these inline to direct member access

3. **CoordinateSystem Direct** - Bare coordinate system operations
   - Verifies no overhead from World wrapper
   - `to_coords()` on CartesianCS (~380 ns)
   - Direct member access (~70 ns)

4. **Loop-Based Realistic Usage** - Representative patterns
   - Full grid conversion (64³ grid): ~116 ms
   - Gaussian initialization with coordinates: ~139 ms
   - Sparse access (1000 points): ~500 μs

5. **Zero-Cost Abstraction Validation** - Compiler optimization check
   - Manual calculation (baseline): ~404 ns
   - World abstraction: ~446 ns (~10% overhead - acceptable)

6. **Memory Access Patterns** - Cache and copy performance
   - World construction/destruction: ~970 ns
   - World copy: ~220 ns
   - Equality comparison: ~1.5 μs

**Key Insights:**

- ✅ Coordinate transformations are fast (~400 ns)
- ✅ Accessors inline well (<100 ns)
- ✅ World wrapper has minimal overhead (~10%)
- ✅ Copy semantics are efficient (~220 ns)
- ⚠️ Debug build - Release build will be significantly faster

**Running:**

```bash
# All World/coordinate benchmarks
./tests/openpfc-tests "[world][benchmark]"

# CoordinateSystem only
./tests/openpfc-tests "[csys][benchmark]"

# With detailed output
./tests/openpfc-tests "[benchmark]" --reporter console
```

## Performance Expectations

For accurate performance measurements:

1. **Use Release build:**

   ```bash
   cmake -B build-release -DCMAKE_BUILD_TYPE=Release
   cmake --build build-release
   ./build-release/tests/openpfc-tests "[benchmark]"
   ```

2. **Expected improvements in Release:**
   - Coordinate transforms: 1-5 ns (vs ~400 ns in Debug)
   - Accessors: <1 ns (should completely inline)
   - Zero-cost abstraction overhead: <5%

3. **Run on dedicated system** (no background processes)
4. **Use representative problem sizes**
5. **Multiple iterations** for statistical significance

## Adding New Benchmarks

When adding benchmarks:

1. **Focus on hot paths** (inner loops, frequently called functions)
2. **Use realistic data** (prevent compiler optimizations with `volatile`)
3. **Document expected performance** in comments
4. **Compare to baseline** (manual calculation)
5. **Tag appropriately**: `[component][benchmark]`

Example:

```cpp
TEST_CASE("FFT performance", "[fft][benchmark]") {
    auto fft = create_fft({128, 128, 128});
    
    BENCHMARK("Forward transform") {
        return fft.forward();
    };
    
    INFO("Expected: <10 ms for 128³ grid");
}
```
