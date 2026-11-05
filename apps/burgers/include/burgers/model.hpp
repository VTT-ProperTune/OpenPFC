// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/** @file model.hpp
 * @brief Periodic viscous Burgers on one CPU rank.
 *
 * The PDE is ∂t u + ∂x(u²/2) = ν ∂xx u on [0, 2π), with u(x,0) = sin(x).
 * Nodal methods collocate at i Δx. The finite-volume unknown is the cell
 * average on [i Δx, (i+1) Δx). Those representations are not interchangeable.
 * The flux and the Rusanov solver stay in this application.
 */

#ifdef BURGERS_HAS_FFTW3
#include <openpfc/kernel/fft/complex_line_fft.hpp>
#endif
#include <openpfc/kernel/field/central_derivatives.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace burgers {

enum class Method {
  spectral,
  spectral_aliased,
  fdm2,
  fdm2_advective,
  fvm,
};

[[nodiscard]] inline const char *method_name(Method method) {
  switch (method) {
  case Method::spectral: return "spectral";
  case Method::spectral_aliased: return "spectral_aliased";
  case Method::fdm2: return "fdm2";
  case Method::fdm2_advective: return "fdm2_advective";
  case Method::fvm: return "fvm";
  }
  throw std::invalid_argument("unknown burgers method");
}

[[nodiscard]] inline Method parse_method(std::string_view name) {
  if (name == "spectral") return Method::spectral;
  if (name == "spectral_aliased") return Method::spectral_aliased;
  if (name == "fdm2") return Method::fdm2;
  if (name == "fdm2_advective") return Method::fdm2_advective;
  if (name == "fvm") return Method::fvm;
  throw std::invalid_argument("burgers method must be spectral, spectral_aliased, "
                              "fdm2, fdm2_advective or fvm");
}

/** Local Lax–Friedrichs flux for f(u) = u²/2. The wave speed is max(|u_L|, |u_R|).
 */
[[nodiscard]] inline double rusanov_flux(double left, double right) {
  const double flux_left = 0.5 * left * left;
  const double flux_right = 0.5 * right * right;
  const double speed = std::max(std::abs(left), std::abs(right));
  return 0.5 * (flux_left + flux_right) - 0.5 * speed * (right - left);
}

[[nodiscard]] inline double mass(double spacing, std::span<const double> state) {
  double sum = 0.0;
  for (double value : state) sum += value;
  return sum * spacing;
}

[[nodiscard]] inline double energy(double spacing, std::span<const double> state) {
  double sum = 0.0;
  for (double value : state) sum += value * value;
  return 0.5 * sum * spacing;
}

[[nodiscard]] inline double maximum_abs(std::span<const double> state) {
  double peak = 0.0;
  for (double value : state) peak = std::max(peak, std::abs(value));
  return peak;
}

[[nodiscard]] inline std::uint64_t checksum(std::span<const double> state) {
  std::uint64_t mixed = 14695981039346656037ull;
  for (double value : state) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    mixed ^= bits;
    mixed *= 1099511628211ull;
  }
  return mixed;
}

/** SSP-RK3 advance of one selectable Burgers discretization. Not reentrant. */
class Simulation {
  static int validated_count(int count) {
    if (count < 4) {
      throw std::invalid_argument("burgers needs at least four points");
    }
    return count;
  }

public:
  Simulation(Method method, int count, double viscosity)
      : method_(method), count_(validated_count(count)), viscosity_(viscosity),
        spacing_(2.0 * std::numbers::pi / static_cast<double>(count_)),
        state_(static_cast<std::size_t>(count_), 0.0) {
    if (!std::isfinite(viscosity) || viscosity < 0.0) {
      throw std::invalid_argument(
          "burgers viscosity must be finite and non-negative");
    }
    if (method_ == Method::spectral || method_ == Method::spectral_aliased) {
#ifdef BURGERS_HAS_FFTW3
      fft_ = std::make_unique<pfc::fft::ComplexLineFFT>(count);
      field_.resize(static_cast<std::size_t>(count));
      spectrum_.resize(static_cast<std::size_t>(count));
      viscous_.resize(static_cast<std::size_t>(count));
      viscous_back_.resize(static_cast<std::size_t>(count));
#else
      throw std::invalid_argument("burgers spectral methods need FFTW3");
#endif
    }
  }

  void set_initial_sine() {
    if (method_ == Method::fvm) {
      for (int i = 0; i < count_; ++i) {
        const double left = static_cast<double>(i) * spacing_;
        const double right = left + spacing_;
        state_[static_cast<std::size_t>(i)] =
            (std::cos(left) - std::cos(right)) / spacing_;
      }
      return;
    }
    for (int i = 0; i < count_; ++i) {
      state_[static_cast<std::size_t>(i)] =
          std::sin(static_cast<double>(i) * spacing_);
    }
  }

  void rate(std::span<const double> in, std::span<double> out) {
    if (static_cast<int>(in.size()) != count_ ||
        static_cast<int>(out.size()) != count_) {
      throw std::invalid_argument("burgers rate spans must match the grid");
    }
    switch (method_) {
    case Method::spectral:
    case Method::spectral_aliased:
#ifdef BURGERS_HAS_FFTW3
      spectral_rate(in, out, method_ == Method::spectral);
#else
      throw std::invalid_argument("burgers spectral methods need FFTW3");
#endif
      break;
    case Method::fdm2: fd_rate(in, out, false); break;
    case Method::fdm2_advective: fd_rate(in, out, true); break;
    case Method::fvm: fvm_rate(in, out); break;
    }
  }

  /// Explicit SSP-RK3 admissible-step screening, conservative for FV/FD2.
  /// For Fourier methods this is a conservative linearized estimate; it does
  /// not claim nonlinear post-shock stability for an arbitrary field.
  [[nodiscard]] double stability_limit() const {
    const double peak = maximum_abs(state_);
    if (!std::isfinite(peak)) {
      throw std::domain_error("burgers nonfinite state");
    }
    const double h = spacing_;
    const bool spectral = method_ != Method::fvm && method_ != Method::fdm2 &&
                          method_ != Method::fdm2_advective;
    double convection = peak / h;
    double diffusion = 2.0 * viscosity_ / (h * h);
    if (spectral) {
      const double divisor = method_ == Method::spectral ? 3.0 : 2.0;
      convection = peak * static_cast<double>(count_) / divisor;
      const double nyquist = std::numbers::pi / h;
      diffusion = viscosity_ * nyquist * nyquist;
    }
    const double rate = convection + diffusion;
    if (rate == 0.0) return std::numeric_limits<double>::infinity();
    return spectral ? 2.35 / rate : 1.0 / rate;
  }

  void advance(double dt, int steps) {
    if (!(dt > 0.0) || !std::isfinite(dt) || steps < 0) {
      throw std::invalid_argument(
          "burgers advance needs a positive step and a non-negative count");
    }
    std::vector<double> u0(static_cast<std::size_t>(count_));
    std::vector<double> u1(static_cast<std::size_t>(count_));
    std::vector<double> u2(static_cast<std::size_t>(count_));
    std::vector<double> stage_rate(static_cast<std::size_t>(count_));
    std::vector<double> candidate(static_cast<std::size_t>(count_));
    for (int step = 0; step < steps; ++step) {
      if (dt > stability_limit()) {
        throw std::invalid_argument("burgers dt exceeds stability limit");
      }
      u0 = state_;
      rate(u0, stage_rate);
      for (int i = 0; i < count_; ++i) {
        u1[static_cast<std::size_t>(i)] =
            u0[static_cast<std::size_t>(i)] +
            dt * stage_rate[static_cast<std::size_t>(i)];
      }
      for (double value : u1) {
        if (!std::isfinite(value)) {
          throw std::domain_error("burgers nonfinite RK stage 1");
        }
      }
      rate(u1, stage_rate);
      for (int i = 0; i < count_; ++i) {
        u2[static_cast<std::size_t>(i)] =
            0.75 * u0[static_cast<std::size_t>(i)] +
            0.25 * (u1[static_cast<std::size_t>(i)] +
                    dt * stage_rate[static_cast<std::size_t>(i)]);
      }
      for (double value : u2) {
        if (!std::isfinite(value)) {
          throw std::domain_error("burgers nonfinite RK stage 2");
        }
      }
      rate(u2, stage_rate);
      for (int i = 0; i < count_; ++i) {
        candidate[static_cast<std::size_t>(i)] =
            (1.0 / 3.0) * u0[static_cast<std::size_t>(i)] +
            (2.0 / 3.0) * (u2[static_cast<std::size_t>(i)] +
                           dt * stage_rate[static_cast<std::size_t>(i)]);
      }
      for (double value : candidate) {
        if (!std::isfinite(value)) {
          throw std::domain_error("burgers nonfinite RK candidate");
        }
      }
      state_.swap(candidate);
    }
  }

  [[nodiscard]] Method method() const { return method_; }
  [[nodiscard]] int count() const { return count_; }
  [[nodiscard]] double viscosity() const { return viscosity_; }
  [[nodiscard]] double spacing() const { return spacing_; }
  [[nodiscard]] std::vector<double> &state() { return state_; }
  [[nodiscard]] const std::vector<double> &state() const { return state_; }

private:
#ifdef BURGERS_HAS_FFTW3
  [[nodiscard]] int raw_mode(int bin) const {
    return bin <= count_ / 2 ? bin : bin - count_;
  }

  [[nodiscard]] bool drop_mode(int bin) const {
    return std::abs(raw_mode(bin)) * 3 >= count_;
  }

  [[nodiscard]] int first_mode(int bin) const {
    if (count_ % 2 == 0 && bin == count_ / 2) return 0;
    return raw_mode(bin);
  }

  void spectral_rate(std::span<const double> in, std::span<double> out,
                     bool dealias) {
    const double norm = static_cast<double>(count_);
    for (int i = 0; i < count_; ++i)
      field_[static_cast<std::size_t>(i)] = in[static_cast<std::size_t>(i)];
    fft_->forward(field_, spectrum_);
    for (int bin = 0; bin < count_; ++bin) {
      const double mode = static_cast<double>(std::abs(raw_mode(bin)));
      viscous_[static_cast<std::size_t>(bin)] =
          spectrum_[static_cast<std::size_t>(bin)] * (-mode * mode);
    }
    if (dealias) {
      for (int bin = 0; bin < count_; ++bin) {
        if (drop_mode(bin)) spectrum_[static_cast<std::size_t>(bin)] = 0.0;
      }
    }
    fft_->backward(spectrum_, field_);
    for (int i = 0; i < count_; ++i) {
      const double value = field_[static_cast<std::size_t>(i)].real() / norm;
      field_[static_cast<std::size_t>(i)] = 0.5 * value * value;
    }
    fft_->forward(field_, spectrum_);
    for (int bin = 0; bin < count_; ++bin) {
      if (dealias && drop_mode(bin)) {
        spectrum_[static_cast<std::size_t>(bin)] = 0.0;
        continue;
      }
      const double mode = static_cast<double>(first_mode(bin));
      spectrum_[static_cast<std::size_t>(bin)] *= std::complex<double>(0.0, mode);
    }
    fft_->backward(spectrum_, field_);
    fft_->backward(viscous_, viscous_back_);
    for (int i = 0; i < count_; ++i) {
      const double flux_derivative =
          field_[static_cast<std::size_t>(i)].real() / norm;
      const double diffusion =
          viscous_back_[static_cast<std::size_t>(i)].real() / norm;
      out[static_cast<std::size_t>(i)] = -flux_derivative + viscosity_ * diffusion;
    }
  }

#endif

  void fd_rate(std::span<const double> in, std::span<double> out,
               bool advective) const {
    const double spacing = spacing_;
    auto at = [&](int index) {
      int wrapped = index % count_;
      if (wrapped < 0) wrapped += count_;
      return in[static_cast<std::size_t>(wrapped)];
    };
    for (int i = 0; i < count_; ++i) {
      auto sample_u = [&](int dx, int, int) { return at(i + dx); };
      const auto velocity =
          pfc::field::fd::evaluate<2>(sample_u, spacing, spacing, spacing);
      if (advective) {
        out[static_cast<std::size_t>(i)] =
            -at(i) * velocity.x + viscosity_ * velocity.xx;
        continue;
      }
      auto sample_flux = [&](int dx, int, int) {
        const double value = at(i + dx);
        return 0.5 * value * value;
      };
      const auto flux =
          pfc::field::fd::evaluate<2>(sample_flux, spacing, spacing, spacing);
      out[static_cast<std::size_t>(i)] = -flux.x + viscosity_ * velocity.xx;
    }
  }

  void fvm_rate(std::span<const double> in, std::span<double> out) const {
    const double spacing = spacing_;
    auto at = [&](int index) {
      int wrapped = index % count_;
      if (wrapped < 0) wrapped += count_;
      return in[static_cast<std::size_t>(wrapped)];
    };
    for (int i = 0; i < count_; ++i) {
      const double left = at(i - 1);
      const double here = at(i);
      const double right = at(i + 1);
      const double flux_right =
          rusanov_flux(here, right) - viscosity_ * (right - here) / spacing;
      const double flux_left =
          rusanov_flux(left, here) - viscosity_ * (here - left) / spacing;
      out[static_cast<std::size_t>(i)] = -(flux_right - flux_left) / spacing;
    }
  }

  Method method_;
  int count_;
  double viscosity_;
  double spacing_;
  std::vector<double> state_;
#ifdef BURGERS_HAS_FFTW3
  std::unique_ptr<pfc::fft::ComplexLineFFT> fft_;
  std::vector<std::complex<double>> field_;
  std::vector<std::complex<double>> spectrum_;
  std::vector<std::complex<double>> viscous_;
  std::vector<std::complex<double>> viscous_back_;
#endif
};

} // namespace burgers
