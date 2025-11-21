# OpenPFC Test Suite

This directory contains the comprehensive test suite for OpenPFC, organized to support different testing strategies and use cases.

## Testing Philosophy

OpenPFC follows a rigorous testing approach with these goals:

- **>90% code coverage** for all production code
- **Test-Driven Development (TDD)** for new features
- **Fast feedback** through isolated unit tests
- **Confidence** through integration and system tests
- **Performance validation** through benchmarks

## Directory Structure

```text
tests/
├── unit/                    # Fast, isolated unit tests
│   ├── core/               # Core data structures (World, Field, etc.)
│   ├── fft/                # FFT functionality
│   ├── models/             # Model base classes
│   ├── field_modifiers/    # Initial/boundary conditions
│   ├── simulator/          # Simulation orchestration
│   └── operators/          # Mathematical operators (not yet active)
│
├── integration/            # Multi-component integration tests
├── benchmarks/             # Performance benchmarks
└── fixtures/               # Shared test utilities and mocks
```

## Running Tests

### Build the Tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target openpfc-tests
```

### Run All Tests

```bash
cd build
./tests/openpfc-tests
```

### Run Tests with Different Reporters

```bash
# Compact output
./tests/openpfc-tests --reporter compact

# JUnit XML for CI
./tests/openpfc-tests --reporter junit --out results.xml

# Verbose output
./tests/openpfc-tests --reporter console
```

### Run Specific Tests

```bash
# By tag
./tests/openpfc-tests "[unit]"
./tests/openpfc-tests "[fft]"

# By test name pattern
./tests/openpfc-tests "World*"

# Specific test case
./tests/openpfc-tests "World - basic functionality"
```

### Run Tests with MPI

For parallel tests:

```bash
mpirun -np 4 ./tests/openpfc-tests
```

## Test Categories

### Unit Tests (`unit/`)

Fast, isolated tests that verify individual components in isolation. Each test should:

- Run in milliseconds
- Have no external dependencies
- Use mocks/fixtures for dependencies
- Test one logical concept per TEST_CASE

**Purpose**: Catch bugs early, enable rapid refactoring, document API usage

### Integration Tests (`integration/`)

Tests that verify multiple components working together. May involve:

- Real FFT operations across MPI ranks
- File I/O operations
- Complete simulation workflows (short runs)

**Purpose**: Verify component interactions, catch integration bugs

### Benchmarks (`benchmarks/`)

Performance-focused tests that measure:

- Execution time
- Memory usage
- Scaling characteristics
- Performance regressions

**Purpose**: Maintain performance standards, guide optimization

## Test Framework

OpenPFC uses **Catch2 v3** as its testing framework.

### Key Features

- BDD-style sections for test organization
- Rich assertion macros (`REQUIRE`, `CHECK`, etc.)
- Floating-point comparisons with `Approx()`
- Test tagging for selective execution
- MPI-aware test runner

### Common Patterns

```cpp
TEST_CASE("Component - what it does", "[module][tag]") {
    SECTION("Basic usage") {
        // Arrange
        auto input = create_input();
        
        // Act
        auto result = function_under_test(input);
        
        // Assert
        REQUIRE(result.is_valid());
        REQUIRE(result.value == Approx(expected).margin(1e-10));
    }
    
    SECTION("Edge case") {
        // Test boundary conditions
    }
    
    SECTION("Error handling") {
        REQUIRE_THROWS_AS(invalid_call(), std::invalid_argument);
    }
}
```

## Shared Test Utilities

The `fixtures/` directory contains shared test utilities:

- **`mock_model.hpp`**: Mock implementations of Model for testing
  - `MockModel`: Basic no-op model
  - `InstrumentedMockModel`: Tracks method calls for verification

Use these to avoid duplicating test infrastructure across test files.

## Coverage Reporting

Tests are built with coverage instrumentation enabled. To generate coverage reports:

```bash
# Run tests
cd build && ./tests/openpfc-tests

# Generate coverage report
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage-html

# View report
xdg-open coverage-html/index.html
```

## Writing New Tests

1. **Determine test type**: Unit, integration, or benchmark?
2. **Find the right directory**: Place tests near related code
3. **Follow naming convention**: `test_<component>.cpp`
4. **Use descriptive names**: `"Component - specific behavior"`
5. **Add appropriate tags**: `[unit]`, `[fft]`, `[model]`, etc.
6. **Update CMakeLists.txt**: Add test file to appropriate subdirectory
7. **Follow TDD**: Write failing test first, then implement

## Continuous Integration

All tests must pass before merging. The CI pipeline:

1. Builds tests in Debug and Release modes
2. Runs all unit tests
3. Runs integration tests with multiple MPI ranks
4. Checks code coverage (target: >90%)
5. Runs static analysis and linting

## Test Maintenance

- **Keep tests fast**: Unit tests should complete in <10ms each
- **Keep tests independent**: No shared state between tests
- **Keep tests readable**: Tests are documentation
- **Update tests with code**: Tests and code evolve together
- **Remove obsolete tests**: Delete tests for removed features

## Getting Help

- **Catch2 Documentation**: <https://github.com/catchorg/Catch2/tree/devel/docs>
- **Ask the Team**: Questions about testing strategy or specific tests

## Test Statistics

Current test suite status:

- **36 test cases**
- **5,298 assertions**
- **All passing** ✅
- Coverage: (to be measured)

---

**Last Updated**: 2025-11-21
