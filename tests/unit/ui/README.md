# UI Layer Unit Tests

This directory contains unit tests for the OpenPFC user interface layer, which handles JSON configuration parsing, validation, and error reporting.

## Test Files

- **`test_ui_errors.cpp`** - Tests for error message formatting helpers
  - `format_config_error()` - Multi-line error message formatting
  - `get_json_value_string()` - JSON value extraction with type info
  - `format_unknown_modifier_error()` - Field modifier type listing
  - `list_valid_field_modifiers()` - Valid modifier types enumeration

## Coverage

These tests ensure that:

- Error messages provide helpful context (field name, description, expected type, actual value)
- All JSON types are handled correctly (null, boolean, integer, float, string, array, object)
- Valid options are formatted clearly when provided
- Examples are included in error messages
- Missing fields are reported with clear guidance

## Related

- **Source**: `include/openpfc/ui_errors.hpp`, `src/openpfc/ui_errors.cpp`
- **Integration Tests**: `tests/integration/test_ui_validation.cpp` - Tests actual validation error paths
- **User Story**: #0038 - Improve UI Error Messages with Helpful Context
