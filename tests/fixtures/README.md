# Test Fixtures

This directory contains **shared test utilities** used across the test suite. Fixtures eliminate code duplication and provide consistent test infrastructure.

## Purpose

- **Reusability**: Common test utilities available to all tests
- **Consistency**: Standardized mocks and helpers
- **Maintainability**: Update test infrastructure in one place
- **Simplicity**: Reduce boilerplate in individual test files

## Current Fixtures

### `mock_model.hpp`

Provides mock implementations of the `Model` interface for testing components that depend on models:

- **`MockModel`**: Basic no-op model implementation for simple tests
- **`InstrumentedMockModel`**: Tracks method calls (evolve, initialize) to verify behavior

**Usage**: Include in test files that need to test Simulator, FieldModifier, or other components that interact with models.

```cpp
#include "fixtures/mock_model.hpp"

TEST_CASE("Simulator - runs model", "[simulator][unit]") {
    InstrumentedMockModel model;
    // ... test using model
    REQUIRE(model.evolve_called());
}
```

## Adding New Fixtures

1. Create `.hpp` file with fixture implementation
2. Use clear, descriptive names (e.g., `mock_*.hpp`, `test_helpers.hpp`)
3. Document fixture purpose and usage in comments
4. Keep fixtures simple and focused
5. Include from test files using: `#include "fixtures/your_fixture.hpp"`

Note: The `tests/` directory is added to the include path, so fixtures can be included with `"fixtures/"` prefix.
