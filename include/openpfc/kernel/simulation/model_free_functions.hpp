// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file model_free_functions.hpp
 * @brief Non-member accessors and helpers for `Model`
 *
 * @details
 * Included from `model.hpp` immediately after the `Model` class definition so
 * callers keep a single `#include <openpfc/kernel/simulation/model.hpp>` entry
 * point (ISP: field/world/FFT helpers vs. the class definition).
 *
 * @see model.hpp
 */

#ifndef PFC_KERNEL_SIMULATION_MODEL_FREE_FUNCTIONS_HPP
#define PFC_KERNEL_SIMULATION_MODEL_FREE_FUNCTIONS_HPP

/**
 * @brief World associated with the model (free function; preferred over
 * Model::get_world()).
 */
[[nodiscard]] inline const World &get_world(const Model &model) noexcept {
  return model.get_world();
}

/**
 * @brief FFT instance used by the model (free function; preferred over
 * Model::get_fft()).
 */
[[nodiscard]] inline fft::IFFT &get_fft(Model &model) noexcept {
  return model.get_fft();
}

[[nodiscard]] inline bool is_rank0(const Model &model) noexcept {
  return model.is_rank0();
}

[[nodiscard]] inline bool has_field(const Model &model,
                                    std::string_view field_name) noexcept {
  return model.has_field(field_name);
}

[[nodiscard]] inline bool has_real_field(const Model &model,
                                         std::string_view field_name) noexcept {
  return model.has_real_field(field_name);
}

[[nodiscard]] inline bool has_complex_field(const Model &model,
                                            std::string_view field_name) noexcept {
  return model.has_complex_field(field_name);
}

[[nodiscard]] inline RealField &get_real_field(Model &model, std::string_view name) {
  return model.get_real_field(name);
}

[[nodiscard]] inline const RealField &get_real_field(const Model &model,
                                                     std::string_view name) {
  return model.get_real_field(name);
}

[[nodiscard]] inline ComplexField &get_complex_field(Model &model,
                                                     std::string_view name) {
  return model.get_complex_field(name);
}

[[nodiscard]] inline const ComplexField &get_complex_field(const Model &model,
                                                           std::string_view name) {
  return model.get_complex_field(name);
}

inline void initialize(Model &model, double dt) { model.initialize(dt); }

inline void step(Model &model, double t) { model.step(t); }

inline void add_real_field(Model &model, std::string_view name, RealField &field) {
  model.add_real_field(name, field);
}

inline void add_complex_field(Model &model, std::string_view name,
                              ComplexField &field) {
  model.add_complex_field(name, field);
}

/** @brief Register a real field (same as add_real_field; matches Model::add_field).
 */
inline void add_field(Model &model, std::string_view name, RealField &field) {
  model.add_real_field(name, field);
}

/** @brief Register a complex field (same as add_complex_field). */
inline void add_field(Model &model, std::string_view name, ComplexField &field) {
  model.add_complex_field(name, field);
}

[[nodiscard]] inline size_t get_allocated_memory_bytes(const Model &model) {
  return model.get_allocated_memory_bytes();
}

#endif // PFC_KERNEL_SIMULATION_MODEL_FREE_FUNCTIONS_HPP
