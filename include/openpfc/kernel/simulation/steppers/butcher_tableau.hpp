#pragma once
#include <vector>
#include <string_view>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <string>

namespace pfc::sim::steppers {

class TableauValidationError : public std::runtime_error {
public:
  enum class ErrorType {
    StageCountMismatch,
    ExplicitStructureViolation,
    RowSumInconsistency,
    NonFiniteCoefficient,
    InvalidStageCount
  };
  
  TableauValidationError(ErrorType type, const std::string& message)
    : std::runtime_error(message), type_(type) {}
  
  ErrorType error_type() const noexcept { return type_; }
  
private:
  ErrorType type_;
};

template <typename T>
class ButcherTableau {
public:
  ButcherTableau(unsigned int s,
                 std::vector<T> a_ij,
                 std::vector<T> b_i,
                 std::vector<T> c_i,
                 std::vector<T> b_hat_i = {},
                 std::string_view name = "",
                 unsigned int order = 0,
                 unsigned int embedded_order = 0)
    : s_(s), a_ij_(std::move(a_ij)), b_i_(std::move(b_i)), c_i_(std::move(c_i)),
      b_hat_(std::move(b_hat_i)), name_(name), order_(order), embedded_order_(embedded_order)
  {
    validate();
  }
  
  unsigned int stage_count() const noexcept { return s_; }
  
  T a(unsigned int i, unsigned int j) const {
    if (i >= s_ || j >= s_) {
      throw std::out_of_range("a(" + std::to_string(i) + "," + std::to_string(j) + ") out of range [0," + std::to_string(s_) + ")");
    }
    return a_ij_[i * s_ + j];
  }
  
  T b(unsigned int i) const {
    if (i >= s_) {
      throw std::out_of_range("b(" + std::to_string(i) + ") out of range [0," + std::to_string(s_) + ")");
    }
    return b_i_[i];
  }
  
  T c(unsigned int i) const {
    if (i >= s_) {
      throw std::out_of_range("c(" + std::to_string(i) + ") out of range [0," + std::to_string(s_) + ")");
    }
    return c_i_[i];
  }
  
  T b_hat(unsigned int i) const {
    if (b_hat_.empty()) {
      throw std::runtime_error("b_hat not available for this tableau");
    }
    if (i >= s_) {
      throw std::out_of_range("b_hat(" + std::to_string(i) + ") out of range [0," + std::to_string(s_) + ")");
    }
    return b_hat_[i];
  }
  
  std::string_view name() const noexcept { return name_; }
  unsigned int order() const noexcept { return order_; }
  bool has_embedded() const noexcept { return !b_hat_.empty(); }
  
private:
  void validate() const {
    // Check minimum stage count
    if (s_ < 1) {
      throw TableauValidationError(TableauValidationError::ErrorType::InvalidStageCount,
        "stage_count " + std::to_string(s_) + " must be >= 1");
    }
    
    // Check stage count consistency
    if (a_ij_.size() != s_ * s_) {
      throw TableauValidationError(TableauValidationError::ErrorType::StageCountMismatch,
        "a_ij size " + std::to_string(a_ij_.size()) + " != stage_count^2 (" + std::to_string(s_ * s_) + ")");
    }
    if (b_i_.size() != s_) {
      throw TableauValidationError(TableauValidationError::ErrorType::StageCountMismatch,
        "b_i size " + std::to_string(b_i_.size()) + " != stage_count (" + std::to_string(s_) + ")");
    }
    if (c_i_.size() != s_) {
      throw TableauValidationError(TableauValidationError::ErrorType::StageCountMismatch,
        "c_i size " + std::to_string(c_i_.size()) + " != stage_count (" + std::to_string(s_) + ")");
    }
    if (!b_hat_.empty() && b_hat_.size() != s_) {
      throw TableauValidationError(TableauValidationError::ErrorType::StageCountMismatch,
        "b_hat_i size " + std::to_string(b_hat_.size()) + " != stage_count (" + std::to_string(s_) + ")");
    }
    
    // Check finite coefficients BEFORE other checks that use values
    for (std::size_t i = 0; i < a_ij_.size(); ++i) {
      check_finite(a_ij_[i], "a_ij", i);
    }
    for (std::size_t i = 0; i < b_i_.size(); ++i) {
      check_finite(b_i_[i], "b_i", i);
    }
    for (std::size_t i = 0; i < c_i_.size(); ++i) {
      check_finite(c_i_[i], "c_i", i);
    }
    for (std::size_t i = 0; i < b_hat_.size(); ++i) {
      check_finite(b_hat_[i], "b_hat_i", i);
    }
    
    // Check explicit lower-triangular structure: a_ij = 0 for i <= j
    const T tolerance = T(1e-12);
    for (unsigned int i = 0; i < s_; ++i) {
      for (unsigned int j = i; j < s_; ++j) {
        if (std::fabs(a(i, j)) > tolerance) {
          throw TableauValidationError(TableauValidationError::ErrorType::ExplicitStructureViolation,
            "a_ij[" + std::to_string(i) + "][" + std::to_string(j) + "] = " + std::to_string(a(i, j)) +
            " != 0 for explicit method (must be zero for i <= j)");
        }
      }
    }
    
    // Check row-sum condition
    const T row_sum_tol = T(1e-10);
    for (unsigned int i = 0; i < s_; ++i) {
      T row_sum = T(0);
      for (unsigned int j = 0; j < s_; ++j) {
        row_sum += a(i, j);
      }
      if (std::fabs(row_sum - c(i)) > row_sum_tol * (T(1) + std::fabs(c(i)))) {
        throw TableauValidationError(TableauValidationError::ErrorType::RowSumInconsistency,
          "sum_j a_ij[" + std::to_string(i) + "][j] = " + std::to_string(row_sum) +
          " != c_i[" + std::to_string(i) + "] = " + std::to_string(c(i)));
      }
    }
  }
  
  
  void check_finite(const T& value, const char* component, std::size_t index) const {
    if (!std::isfinite(value)) {
      throw TableauValidationError(TableauValidationError::ErrorType::NonFiniteCoefficient,
        std::string(component) + "[" + std::to_string(index) + "] = " + std::to_string(value) + " is not finite");
    }
  }
  
  const unsigned int s_;
  const std::vector<T> a_ij_;
  const std::vector<T> b_i_;
  const std::vector<T> c_i_;
  const std::vector<T> b_hat_;
  const std::string name_;
  const unsigned int order_;
  const unsigned int embedded_order_;
};

template <typename T>
ButcherTableau<T> make_rk2_midpoint() {
  return ButcherTableau<T>(
    2,
    {T(0), T(0), T(0.5), T(0)},
    {T(0), T(1)},
    {T(0), T(0.5)},
    {},
    "RK2 midpoint",
    2
  );
}

template <typename T>
ButcherTableau<T> make_rk2_heun() {
  return ButcherTableau<T>(
    2,
    {T(0), T(0), T(1), T(0)},
    {T(0.5), T(0.5)},
    {T(0), T(1)},
    {},
    "RK2 Heun",
    2
  );
}

template <typename T>
ButcherTableau<T> make_rk4_classical() {
  return ButcherTableau<T>(
    4,
    {T(0), T(0), T(0), T(0),
     T(0.5), T(0), T(0), T(0),
     T(0), T(0.5), T(0), T(0),
     T(0), T(0), T(1), T(0)},
    {T(1.0/6.0), T(1.0/3.0), T(1.0/3.0), T(1.0/6.0)},
    {T(0), T(0.5), T(0.5), T(1)},
    {},
    "RK4 classical",
    4
  );
}

template <typename T>
ButcherTableau<T> make_embedded_rk23() {
  return ButcherTableau<T>(
    4,
    {T(0), T(0), T(0), T(0),
     T(0.5), T(0), T(0), T(0),
     T(0), T(0.75), T(0), T(0),
     T(2.0/9.0), T(1.0/3.0), T(4.0/9.0), T(0)},
    {T(7.0/24.0), T(1.0/4.0), T(1.0/3.0), T(1.0/8.0)},
    {T(0), T(0.5), T(0.75), T(1)},
    {T(2.0/9.0), T(1.0/3.0), T(4.0/9.0), T(0)},
    "Bogacki-Shampine 2(3)",
    2,
    3
  );
}

} // namespace pfc::sim::steppers
