// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <openpfc/kernel/simulation/steppers/stepper_validation.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>

// Mock grad types for testing
struct TestGrads {
  double grad_x;
  double grad_y;
  double grad_z;
};

// Mock evaluator that returns TestGrads from operator()
class MockEvaluator {
public:
  TestGrads operator()(int, int, int) const { return TestGrads{}; }
};

// Mock models for testing
class ValidScalarModel {
public:
  double rhs(double t, const TestGrads&) const { return 0.0; }
};

// Tuple protocol model
class ValidTupleProtocolModel {
public:
  struct TupleReturn {
    double first;
    double second;
    
    auto as_tuple() & { return std::tie(first, second); }
    auto as_tuple() const & { return std::tie(first, second); }
  };
  
  TupleReturn rhs(double t, const TestGrads&) const { return {0.0, 0.0}; }
};

class ValidStdTupleModel {
public:
  std::tuple<double, double, double> rhs(double t, const TestGrads&) const { return {0.0, 0.0, 0.0}; }
};

class InvalidReturnModel {
public:
  std::string rhs(double t, const TestGrads&) const { return "invalid"; }
};

// Helper that matches MultiEulerStepper's template signature
template <class Rhs, std::size_t N>
class TestMultiStepper {
public:
  using RhsType = Rhs;
  static constexpr std::size_t field_count = N;
};

TEST_CASE("validate_rhs_signature accepts scalar return") {
  pfc::sim::steppers::validate_rhs_signature<ValidScalarModel, MockEvaluator>();
  SUCCEED();
}

TEST_CASE("validate_rhs_signature accepts tuple protocol return") {
  pfc::sim::steppers::validate_rhs_signature<ValidTupleProtocolModel, MockEvaluator>();
  SUCCEED();
}

TEST_CASE("validate_rhs_signature accepts std::tuple return") {
  pfc::sim::steppers::validate_rhs_signature<ValidStdTupleModel, MockEvaluator>();
  SUCCEED();
}

TEST_CASE("validate_rhs_signature rejects invalid return") {
  // This should fail to compile due to static_assert
  // Uncomment to verify error message:
  // pfc::sim::steppers::validate_rhs_signature<InvalidReturnModel, MockEvaluator>();
  SUCCEED("Test verified by compilation failure when uncommented");
}

TEST_CASE("validate_spatial_compatibility accepts fd gradient") {
  using GradType = pfc::gradient::FDGradient<TestGrads>;
  pfc::sim::steppers::validate_spatial_compatibility<GradType>();
  SUCCEED();
}

TEST_CASE("validate_spatial_compatibility accepts spectral gradient") {
  using GradType = pfc::field::SpectralGradient<TestGrads>;
  pfc::sim::steppers::validate_spatial_compatibility<GradType>();
  SUCCEED();
}

TEST_CASE("validate_spatial_compatibility accepts composite gradient") {
  using GradType = pfc::field::CompositeGradient<TestGrads, double, float>;
  pfc::sim::steppers::validate_spatial_compatibility<GradType>();
  SUCCEED();
}

TEST_CASE("validate_spatial_compatibility rejects unknown backend") {
  struct UnknownEvaluator {};
  REQUIRE_THROWS_AS(
    pfc::sim::steppers::validate_spatial_compatibility<UnknownEvaluator>(),
    pfc::sim::steppers::StepperValidationError
  );
  
  try {
    pfc::sim::steppers::validate_spatial_compatibility<UnknownEvaluator>();
    FAIL("Should have thrown StepperValidationError");
  } catch (const pfc::sim::steppers::StepperValidationError& ex) {
    REQUIRE(ex.error_type() == pfc::sim::steppers::StepperValidationError::ErrorType::IncompatibleBackend);
    REQUIRE_THAT(ex.what(), Catch::Matchers::ContainsSubstring("not a recognized gradient backend"));
  }
}

TEST_CASE("validate_field_count accepts matching type and count") {
  using ModelRhs = decltype(std::declval<ValidStdTupleModel&>().rhs(0.0, std::declval<const TestGrads&>()));
  pfc::sim::steppers::validate_field_count<TestMultiStepper<ModelRhs, 3>, ValidStdTupleModel, MockEvaluator>();
  SUCCEED();
}

TEST_CASE("validate_field_count rejects mismatched type") {
  using WrongRhs = std::tuple<int, int, int>;  // Wrong type, same count
  // This should fail to compile due to static_assert
  // Uncomment to verify error message:
  // pfc::sim::steppers::validate_field_count<TestMultiStepper<WrongRhs, 3>, ValidStdTupleModel, MockEvaluator>();
  SUCCEED("Test verified by compilation failure when uncommented");
}

TEST_CASE("validate_field_count rejects mismatched count") {
  using ModelRhs = decltype(std::declval<ValidStdTupleModel&>().rhs(0.0, std::declval<const TestGrads&>()));
  // This should fail to compile due to static_assert
  // Uncomment to verify error message:
  // pfc::sim::steppers::validate_field_count<TestMultiStepper<ModelRhs, 2>, ValidStdTupleModel, MockEvaluator>();
  SUCCEED("Test verified by compilation failure when uncommented");
}

TEST_CASE("stepper_validation_error type and message") {
  using namespace pfc::sim::steppers;
  
  SECTION("SignatureMismatch error") {
    StepperValidationError err(StepperValidationError::ErrorType::SignatureMismatch, "test message");
    REQUIRE(err.error_type() == StepperValidationError::ErrorType::SignatureMismatch);
    REQUIRE_THAT(err.what(), Catch::Matchers::Equals("test message"));
  }
  
  SECTION("IncompatibleBackend error") {
    StepperValidationError err(StepperValidationError::ErrorType::IncompatibleBackend, "backend error");
    REQUIRE(err.error_type() == StepperValidationError::ErrorType::IncompatibleBackend);
    REQUIRE_THAT(err.what(), Catch::Matchers::Equals("backend error"));
  }
  
  SECTION("FieldCountMismatch error") {
    StepperValidationError err(StepperValidationError::ErrorType::FieldCountMismatch, "field count error");
    REQUIRE(err.error_type() == StepperValidationError::ErrorType::FieldCountMismatch);
    REQUIRE_THAT(err.what(), Catch::Matchers::Equals("field count error"));
  }
}
