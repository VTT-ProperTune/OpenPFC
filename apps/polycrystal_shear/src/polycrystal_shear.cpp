// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file polycrystal_shear.cpp
 * @brief Incremental finite-strain shear of a periodic FCC polycrystal.
 *
 * The grid is a cube of cubic grains. Each converged increment writes the
 * mid-plane scalar of the symmetric Piola stress. History is the existing
 * FCC slip state. The solver is unchanged apart from accepting the grid.
 */

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/mechanics/constitutive/crystal_plasticity.hpp>
#include <openpfc/solvers/finite_strain_fft/newton.hpp>

#include <mpi.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

using namespace pfc;
using namespace pfc::finite_strain;

namespace {

struct Options {
  int size = 0;
  int steps = 2;
  double shear = 0.002;
  int grains = 0;
  int seed = 1;
  int render = 0;
  int fps = 60;
  int krylov = 64;
  int newton = 8;
  std::string outdir;
  std::string commit = "unknown";
  std::string launch;
};

FccSlipParameters aluminum_parameters() noexcept {
  FccSlipParameters parameters;
  parameters.c11 = 106.75e9;
  parameters.c12 = 60.41e9;
  parameters.c44 = 28.34e9;
  parameters.reference_rate = 0.001;
  parameters.rate_exponent = 20.0;
  parameters.hardening_exponent = 2.25;
  parameters.hardening_coefficient = 75.0e6;
  parameters.initial_resistance = 31.0e6;
  parameters.saturation_resistance = 63.0e6;
  parameters.time_step = 1.0;
  parameters.coplanar = 1.0;
  parameters.latent = 1.4;
  return parameters;
}

Tensor2 simple_shear(double gamma) {
  Tensor2 value = identity2();
  value(0, 1) = gamma;
  return value;
}

Tensor2 rotation_from_quaternion(double w, double x, double y, double z) {
  Tensor2 orientation{};
  orientation(0, 0) = 1.0 - 2.0 * (y * y + z * z);
  orientation(0, 1) = 2.0 * (x * y - z * w);
  orientation(0, 2) = 2.0 * (x * z + y * w);
  orientation(1, 0) = 2.0 * (x * y + z * w);
  orientation(1, 1) = 1.0 - 2.0 * (x * x + z * z);
  orientation(1, 2) = 2.0 * (y * z - x * w);
  orientation(2, 0) = 2.0 * (x * z - y * w);
  orientation(2, 1) = 2.0 * (y * z + x * w);
  orientation(2, 2) = 1.0 - 2.0 * (x * x + y * y);
  return orientation;
}

int automatic_grains(int size) {
  int edge = 16;
  while (edge > 1 && size % edge != 0) {
    edge /= 2;
  }
  return edge;
}

/// Frobenius norm of dev(sym(P)). Not `von_mises`: no sqrt(3/2), and P
/// is symmetrized first.
[[nodiscard]] double symmetric_piola_scalar(const Tensor2 &piola) noexcept {
  const double s00 = piola(0, 0);
  const double s11 = piola(1, 1);
  const double s22 = piola(2, 2);
  const double s01 = 0.5 * (piola(0, 1) + piola(1, 0));
  const double s02 = 0.5 * (piola(0, 2) + piola(2, 0));
  const double s12 = 0.5 * (piola(1, 2) + piola(2, 1));
  const double mean = (s00 + s11 + s22) / 3.0;
  const double d00 = s00 - mean;
  const double d11 = s11 - mean;
  const double d22 = s22 - mean;
  return std::sqrt(d00 * d00 + d11 * d11 + d22 * d22 +
                   2.0 * (s01 * s01 + s02 * s02 + s12 * s12));
}

struct Layout {
  int origin_x = 0;
  int origin_y = 0;
  int origin_z = 0;
  int local_x = 0;
  int local_y = 0;
  int local_z = 0;
  int global_n = 0;
  int grains_per_edge = 1;
  int voxels_per_grain = 1;

  [[nodiscard]] int grain_at(std::size_t index) const {
    const int x =
        origin_x + static_cast<int>(index % static_cast<std::size_t>(local_x));
    const int plane = local_x * local_y;
    const int y =
        origin_y + static_cast<int>((index / static_cast<std::size_t>(local_x)) %
                                    static_cast<std::size_t>(local_y));
    const int z =
        origin_z + static_cast<int>(index / static_cast<std::size_t>(plane));
    const int ix = x / voxels_per_grain;
    const int iy = y / voxels_per_grain;
    const int iz = z / voxels_per_grain;
    return ix + grains_per_edge * (iy + grains_per_edge * iz);
  }
};

class PolycrystalFcc {
public:
  PolycrystalFcc(std::size_t local_count, Layout layout,
                 FccSlipParameters parameters, std::vector<Tensor2> orientations)
      : parameters_(parameters), layout_(layout),
        orientations_(std::move(orientations)),
        state_(local_count, fcc_identity_state(parameters_)), cache_(local_count) {}

  [[nodiscard]] std::size_t size() const noexcept { return state_.size(); }

  [[nodiscard]] Tensor2 stress(std::size_t index, const Tensor2 &deformation) const {
    return response(index, deformation).piola;
  }

  [[nodiscard]] Tensor2 tangent_action(std::size_t index, const Tensor2 &deformation,
                                       const Tensor2 &increment) const {
    return response(index, deformation).tangent.action(increment);
  }

  void accept(std::size_t index, const Tensor2 &deformation) {
    state_[index] = response(index, deformation).trial;
  }

  [[nodiscard]] bool failed() const noexcept { return failed_; }

  [[nodiscard]] const std::string &failure() const noexcept { return failure_; }

  [[nodiscard]] double stored_bytes() const noexcept {
    return static_cast<double>(state_.size()) *
           static_cast<double>(sizeof(FccSlipState) + sizeof(Slot));
  }

private:
  struct Slot {
    Tensor2 deformation{};
    Tensor2 piola{};
    PiolaTangent tangent{};
    FccSlipState trial{};
    bool valid = false;
  };

  [[nodiscard]] const Slot &response(std::size_t index,
                                     const Tensor2 &deformation) const {
    Slot &slot = cache_[index];
    if (slot.valid && crystal::same_tensor(slot.deformation, deformation)) {
      return slot;
    }
    try {
      const FccEvaluation evaluation = evaluate_fcc(
          parameters_,
          orientations_[static_cast<std::size_t>(layout_.grain_at(index))],
          state_[index], deformation);
      slot.deformation = deformation;
      slot.piola = evaluation.update.piola;
      slot.tangent = evaluation.tangent;
      slot.trial = evaluation.update.state;
      slot.valid = true;
    } catch (const std::exception &error) {
      failed_ = true;
      if (failure_.empty()) {
        failure_ = error.what();
      }
      slot = Slot{};
      slot.deformation = deformation;
      slot.valid = true;
    }
    return slot;
  }

  FccSlipParameters parameters_;
  Layout layout_;
  std::vector<Tensor2> orientations_;
  std::vector<FccSlipState> state_;
  mutable std::vector<Slot> cache_;
  mutable bool failed_ = false;
  mutable std::string failure_;
};

std::vector<Tensor2> make_orientations(int grains_per_edge, std::uint32_t seed) {
  const int count = grains_per_edge * grains_per_edge * grains_per_edge;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<Tensor2> orientations(static_cast<std::size_t>(count));
  constexpr double kPi = 3.14159265358979323846;
  for (Tensor2 &orientation : orientations) {
    const double u1 = unit(generator);
    const double u2 = unit(generator);
    const double u3 = unit(generator);
    const double amplitude = std::sqrt(std::max(0.0, 1.0 - u1));
    const double complement = std::sqrt(u1);
    double w = amplitude * std::sin(2.0 * kPi * u2);
    double x = amplitude * std::cos(2.0 * kPi * u2);
    double y = complement * std::sin(2.0 * kPi * u3);
    double z = complement * std::cos(2.0 * kPi * u3);
    const double norm = std::sqrt(w * w + x * x + y * y + z * z);
    w /= norm;
    x /= norm;
    y /= norm;
    z /= norm;
    orientation = rotation_from_quaternion(w, x, y, z);
    if (!crystal::proper_rotation(orientation)) {
      throw std::runtime_error("generated crystal orientation is not a rotation");
    }
  }
  if (orientations.empty()) {
    throw std::runtime_error("crystal orientation grid is empty");
  }
  orientations.front() = identity2();
  return orientations;
}

void broadcast_orientations(std::vector<Tensor2> &orientations, int rank,
                            MPI_Comm comm) {
  int count = static_cast<int>(orientations.size());
  MPI_Bcast(&count, 1, MPI_INT, 0, comm);
  if (rank != 0) {
    orientations.resize(static_cast<std::size_t>(count));
  }
  MPI_Bcast(orientations.empty() ? nullptr : &orientations.front()(0, 0), count * 9,
            MPI_DOUBLE, 0, comm);
}

bool parse_options(int argc, char **argv, Options &options, std::string &error) {
  std::ostringstream launch;
  for (int i = 0; i < argc; ++i) {
    if (i != 0) {
      launch << ' ';
    }
    launch << argv[i];
  }
  options.launch = launch.str();
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        error = std::string(name) + " needs a value";
        return nullptr;
      }
      return argv[++i];
    };
    if (flag == "--size") {
      const char *value = need("--size");
      if (value == nullptr) {
        return false;
      }
      options.size = std::atoi(value);
    } else if (flag == "--steps") {
      const char *value = need("--steps");
      if (value == nullptr) {
        return false;
      }
      options.steps = std::atoi(value);
    } else if (flag == "--shear") {
      const char *value = need("--shear");
      if (value == nullptr) {
        return false;
      }
      options.shear = std::atof(value);
    } else if (flag == "--grains") {
      const char *value = need("--grains");
      if (value == nullptr) {
        return false;
      }
      options.grains = std::atoi(value);
    } else if (flag == "--seed") {
      const char *value = need("--seed");
      if (value == nullptr) {
        return false;
      }
      options.seed = std::atoi(value);
    } else if (flag == "--render") {
      const char *value = need("--render");
      if (value == nullptr) {
        return false;
      }
      options.render = std::atoi(value);
    } else if (flag == "--fps") {
      const char *value = need("--fps");
      if (value == nullptr) {
        return false;
      }
      options.fps = std::atoi(value);
    } else if (flag == "--krylov") {
      const char *value = need("--krylov");
      if (value == nullptr) {
        return false;
      }
      options.krylov = std::atoi(value);
    } else if (flag == "--newton") {
      const char *value = need("--newton");
      if (value == nullptr) {
        return false;
      }
      options.newton = std::atoi(value);
    } else if (flag == "--outdir") {
      const char *value = need("--outdir");
      if (value == nullptr) {
        return false;
      }
      options.outdir = value;
    } else if (flag == "--commit") {
      const char *value = need("--commit");
      if (value == nullptr) {
        return false;
      }
      options.commit = value;
    } else {
      error = "unknown argument " + flag;
      return false;
    }
  }
  if (options.size < 2) {
    error = "--size must be at least 2";
    return false;
  }
  if (options.steps < 1 || options.shear <= 0.0 || options.fps < 1 ||
      options.krylov < 1 || options.newton < 1 || options.outdir.empty()) {
    error = "steps, shear, fps, krylov, newton, and outdir must be positive";
    return false;
  }
  if (options.grains == 0) {
    options.grains = automatic_grains(options.size);
  }
  if (options.grains < 1 || options.size % options.grains != 0) {
    error = "--grains must divide --size";
    return false;
  }
  return true;
}

void write_provenance(const Options &options, int steps_completed, int exit_status) {
  const std::filesystem::path path =
      std::filesystem::path(options.outdir) / "provenance.txt";
  std::ofstream out(path);
  out << "commit: " << options.commit << "\n";
  out << "launch: " << options.launch << "\n";
  out << "grid: " << options.size << " " << options.size << " " << options.size
      << "\n";
  out << "formulation: finite strain\n";
  out << "law: FCC crystal plasticity (DAMASK phenopowerlaw aluminum)\n";
  out << "steps_requested: " << options.steps << "\n";
  out << "steps_completed: " << steps_completed << "\n";
  out << "shear_final: " << options.shear << "\n";
  if (exit_status < 0) {
    out << "exit_status: running\n";
  } else {
    out << "exit_status: " << exit_status << "\n";
  }
}

[[nodiscard]] double resident_gib() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream values(line);
      std::string key;
      double kilobytes = 0.0;
      values >> key >> kilobytes;
      return kilobytes / (1024.0 * 1024.0);
    }
  }
  return 0.0;
}

void colorize(double unit, unsigned char &red, unsigned char &green,
              unsigned char &blue) {
  const double stops[5][3] = {
      {16, 18, 38}, {48, 92, 168}, {36, 168, 142}, {230, 190, 70}, {245, 244, 236}};
  const double clamped = std::min(1.0, std::max(0.0, unit));
  const double scaled = clamped * 4.0;
  const int bin = std::min(3, static_cast<int>(scaled));
  const double fraction = scaled - static_cast<double>(bin);
  red = static_cast<unsigned char>(stops[bin][0] +
                                   (stops[bin + 1][0] - stops[bin][0]) * fraction);
  green = static_cast<unsigned char>(stops[bin][1] +
                                     (stops[bin + 1][1] - stops[bin][1]) * fraction);
  blue = static_cast<unsigned char>(stops[bin][2] +
                                    (stops[bin + 1][2] - stops[bin][2]) * fraction);
}

[[nodiscard]] bool write_all(int fd, const unsigned char *data, std::size_t size) {
  while (size > 0) {
    const ssize_t written = ::write(fd, data, size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    data += written;
    size -= static_cast<std::size_t>(written);
  }
  return true;
}

/// Reap the encoder. A non-zero status is a failure. The write end is closed
/// so the child sees end-of-file.
[[nodiscard]] int encoder_status(pid_t child, int write_fd) {
  if (write_fd >= 0) {
    ::close(write_fd);
  }
  int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      std::fprintf(stderr, "ffmpeg wait failed: %s\n", std::strerror(errno));
      return 1;
    }
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::fprintf(stderr, "ffmpeg exit %d\n", status);
    return 1;
  }
  return 0;
}

int render_animation(const Options &options, double peak) {
  // FFMPEG names one executable. Otherwise search PATH for `ffmpeg`.
  // Neither value is passed through a shell.
  const char *from_env = std::getenv("FFMPEG");
  const std::string binary = (from_env != nullptr && from_env[0] != '\0')
                                 ? std::string(from_env)
                                 : std::string("ffmpeg");

  int pipefd[2] = {-1, -1};
  if (::pipe(pipefd) != 0) {
    std::fprintf(stderr, "ffmpeg pipe failed: %s\n", std::strerror(errno));
    return 1;
  }

  const std::filesystem::path output =
      std::filesystem::path(options.outdir) / "microstructure.mp4";
  const std::string size_arg =
      std::to_string(options.size) + "x" + std::to_string(options.size);
  const std::string fps_arg = std::to_string(options.fps);
  const std::string output_arg = output.string();
  const char *pixel = (options.size % 2 == 0) ? "yuv420p" : "yuv444p";
  std::vector<std::string> args = {
      binary,      "-y",         "-loglevel", "error",   "-f",       "rawvideo",
      "-pix_fmt",  "rgb24",      "-s",        size_arg,  "-r",       fps_arg,
      "-i",        "-",          "-c:v",      "libx264", "-pix_fmt", pixel,
      "-movflags", "+faststart", output_arg};
  std::vector<char *> argv;
  argv.reserve(args.size() + 1);
  for (std::string &arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions) != 0) {
    ::close(pipefd[0]);
    ::close(pipefd[1]);
    std::fprintf(stderr, "ffmpeg pipe failed\n");
    return 1;
  }
  const bool actions_ok =
      posix_spawn_file_actions_adddup2(&actions, pipefd[0], STDIN_FILENO) == 0 &&
      posix_spawn_file_actions_addclose(&actions, pipefd[0]) == 0 &&
      posix_spawn_file_actions_addclose(&actions, pipefd[1]) == 0;
  if (!actions_ok) {
    posix_spawn_file_actions_destroy(&actions);
    ::close(pipefd[0]);
    ::close(pipefd[1]);
    std::fprintf(stderr, "ffmpeg pipe failed\n");
    return 1;
  }

  pid_t child = 0;
  const int spawned = binary.find('/') != std::string::npos
                          ? posix_spawn(&child, binary.c_str(), &actions, nullptr,
                                        argv.data(), environ)
                          : posix_spawnp(&child, binary.c_str(), &actions, nullptr,
                                         argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  ::close(pipefd[0]);
  if (spawned != 0) {
    ::close(pipefd[1]);
    std::fprintf(stderr, "ffmpeg executable failed to start (%s): %s\n",
                 binary.c_str(), std::strerror(spawned));
    return 1;
  }

  const std::size_t pixels = static_cast<std::size_t>(options.size) *
                             static_cast<std::size_t>(options.size);
  std::vector<float> sample(pixels);
  std::vector<unsigned char> rgb(pixels * 3u);
  const double scale = peak > 0.0 ? peak : 1.0;
  for (int step = 1; step <= options.steps; ++step) {
    char name[64];
    std::snprintf(name, sizeof(name), "slice-%05d.f32", step);
    const std::filesystem::path file = std::filesystem::path(options.outdir) / name;
    std::ifstream in(file, std::ios::binary);
    in.read(reinterpret_cast<char *>(sample.data()),
            static_cast<std::streamsize>(sample.size() * sizeof(float)));
    if (!in) {
      std::fprintf(stderr, "missing slice %s\n", file.string().c_str());
      (void)encoder_status(child, pipefd[1]);
      return 1;
    }
    for (std::size_t pixel_index = 0; pixel_index < pixels; ++pixel_index) {
      unsigned char red = 0;
      unsigned char green = 0;
      unsigned char blue = 0;
      colorize(static_cast<double>(sample[pixel_index]) / scale, red, green, blue);
      rgb[3u * pixel_index] = red;
      rgb[3u * pixel_index + 1u] = green;
      rgb[3u * pixel_index + 2u] = blue;
    }
    if (!write_all(pipefd[1], rgb.data(), rgb.size())) {
      std::fprintf(stderr, "ffmpeg write failed at step %d\n", step);
      (void)encoder_status(child, pipefd[1]);
      return 1;
    }
  }
  if (encoder_status(child, pipefd[1]) != 0) {
    return 1;
  }
  std::printf("rendered %s\n", output.string().c_str());
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  // A failed fabric init can leave every process as its own world. Each
  // one would then allocate the full grid. Abort before that allocation.
  if (const char *launched = std::getenv("SLURM_STEP_NUM_TASKS")) {
    const int expected = std::atoi(launched);
    if (expected > 1 && ranks != expected) {
      std::fprintf(stderr,
                   "polycrystal_shear rank %d mpi_world=%d slurm_step_tasks=%d\n",
                   rank, ranks, expected);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
  }
  if (rank == 0) {
    std::fprintf(stderr, "polycrystal_shear mpi_world=%d\n", ranks);
    std::fflush(stderr);
  }

  Options options;
  std::string error;
  if (!parse_options(argc, argv, options, error)) {
    if (rank == 0) {
      std::fprintf(stderr, "%s\n", error.c_str());
    }
    MPI_Finalize();
    return 2;
  }

  // Rank 0 creates the directory. Every rank must observe that result
  // before the barrier: a rank-zero failure used to finalize alone and
  // leave the others waiting.
  int outdir_ok = 0;
  if (rank == 0) {
    std::error_code mkdir_error;
    std::filesystem::create_directories(options.outdir, mkdir_error);
    if (mkdir_error) {
      std::fprintf(stderr, "outdir: %s: %s\n", options.outdir.c_str(),
                   mkdir_error.message().c_str());
    } else {
      outdir_ok = 1;
    }
  }
  MPI_Bcast(&outdir_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (outdir_ok == 0) {
    MPI_Finalize();
    return 2;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  const FccSlipParameters parameters = aluminum_parameters();
  std::vector<Tensor2> orientations;
  if (rank == 0) {
    orientations =
        make_orientations(options.grains, static_cast<std::uint32_t>(options.seed));
  }
  broadcast_orientations(orientations, rank, MPI_COMM_WORLD);

  // HeFFTe frees a duplicated communicator from its destructor. That
  // object has to die before MPI_Finalize, or a multi-rank run aborts
  // after the solve has already succeeded.
  int render_status = 0;
  {
    Domain domain = domain::create(Int3{options.size, options.size, options.size});
    auto decomposition = decomposition::create(domain, ranks);
    auto transform = fft::create(decomposition, rank, MPI_COMM_WORLD);
    CompatibleProjector projector(transform, domain);
    const auto box = transform.get_inbox_bounds();
    Layout layout;
    layout.origin_x = box.low[0];
    layout.origin_y = box.low[1];
    layout.origin_z = box.low[2];
    layout.local_x = box.high[0] - box.low[0] + 1;
    layout.local_y = box.high[1] - box.low[1] + 1;
    layout.local_z = box.high[2] - box.low[2] + 1;
    layout.global_n = options.size;
    layout.grains_per_edge = options.grains;
    layout.voxels_per_grain = options.size / options.grains;
    const std::size_t local_count = projector.local_size();
    if (static_cast<std::size_t>(layout.local_x) *
            static_cast<std::size_t>(layout.local_y) *
            static_cast<std::size_t>(layout.local_z) !=
        local_count) {
      if (rank == 0) {
        std::fprintf(stderr, "FFT inbox is not a dense brick\n");
      }
      MPI_Abort(MPI_COMM_WORLD, 2);
    }

    PolycrystalFcc material(local_count, layout, parameters, orientations);
    if (rank == 0) {
      const double local_gib = material.stored_bytes() / (1024.0 * 1024.0 * 1024.0);
      std::printf("polycrystal_shear grid=%d grains=%d ranks=%d local_points=%zu "
                  "material_GiB_per_rank=%.3f\n",
                  options.size, options.grains, ranks, local_count, local_gib);
      std::fflush(stdout);
    }

    NewtonControls controls;
    controls.maximum_krylov_iterations = options.krylov;
    controls.maximum_newton_iterations = options.newton;
    IncrementStart start;
    start.deformation.assign(local_count, identity2());
    const int mid = options.size / 2;
    const std::size_t plane_count = static_cast<std::size_t>(options.size) *
                                    static_cast<std::size_t>(options.size);
    std::vector<float> local_plane(plane_count);
    std::vector<float> reduced_plane(plane_count);
    double peak = 0.0;
    int completed = 0;
    bool slices_differ = false;
    double first_sum = 0.0;
    bool step_failed = false;

    for (int step = 1; step <= options.steps; ++step) {
      const auto started = std::chrono::steady_clock::now();
      const double gamma = options.shear * static_cast<double>(step) /
                           static_cast<double>(options.steps);
      const Tensor2 macroscopic = simple_shear(gamma);
      NewtonResult solved;
      try {
        solved = solve_increment(projector, material, macroscopic, start, controls,
                                 MPI_COMM_WORLD);
      } catch (const std::exception &solve_error) {
        if (rank == 0) {
          std::fprintf(stderr, "step %d failed: %s\n", step, solve_error.what());
          write_provenance(options, completed, 1);
        }
        step_failed = true;
        render_status = 1;
        break;
      }
      int local_failed = material.failed() ? 1 : 0;
      int any_failed = 0;
      MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (any_failed != 0 || !solved.converged) {
        if (rank == 0) {
          std::fprintf(stderr, "step %d did not converge (material_failed=%d)\n",
                       step, any_failed);
          if (!material.failure().empty()) {
            std::fprintf(stderr, "material: %s\n", material.failure().c_str());
          }
          write_provenance(options, completed, 1);
        }
        step_failed = true;
        render_status = 1;
        break;
      }
      for (std::size_t index = 0; index < local_count; ++index) {
        material.accept(index, solved.deformation[index]);
      }
      std::fill(local_plane.begin(), local_plane.end(), 0.0f);
      double local_peak = 0.0;
      std::size_t index = 0;
      for (int z = layout.origin_z; z < layout.origin_z + layout.local_z; ++z) {
        for (int y = layout.origin_y; y < layout.origin_y + layout.local_y; ++y) {
          for (int x = layout.origin_x; x < layout.origin_x + layout.local_x;
               ++x, ++index) {
            if (z != mid) {
              continue;
            }
            const double scalar = symmetric_piola_scalar(solved.piola[index]);
            local_peak = std::max(local_peak, scalar);
            local_plane[static_cast<std::size_t>(x + options.size * y)] =
                static_cast<float>(scalar);
          }
        }
      }
      if (index != local_count) {
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
      MPI_Reduce(local_plane.data(), reduced_plane.data(),
                 static_cast<int>(plane_count), MPI_FLOAT, MPI_SUM, 0,
                 MPI_COMM_WORLD);
      double global_peak = 0.0;
      MPI_Reduce(&local_peak, &global_peak, 1, MPI_DOUBLE, MPI_MAX, 0,
                 MPI_COMM_WORLD);
      const double local_rss = resident_gib();
      double max_rss = 0.0;
      MPI_Reduce(&local_rss, &max_rss, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      const auto stopped = std::chrono::steady_clock::now();
      const double seconds =
          std::chrono::duration<double>(stopped - started).count();
      if (rank == 0) {
        peak = std::max(peak, global_peak);
        char name[64];
        std::snprintf(name, sizeof(name), "slice-%05d.f32", step);
        const std::filesystem::path file =
            std::filesystem::path(options.outdir) / name;
        std::ofstream slice(file, std::ios::binary);
        slice.write(
            reinterpret_cast<const char *>(reduced_plane.data()),
            static_cast<std::streamsize>(reduced_plane.size() * sizeof(float)));
        if (!slice) {
          std::fprintf(stderr, "slice write failed: %s\n", file.string().c_str());
          write_provenance(options, completed, 1);
          MPI_Abort(MPI_COMM_WORLD, 4);
        }
        double sum = 0.0;
        for (float value : reduced_plane) {
          sum += static_cast<double>(value);
        }
        if (step == 1) {
          first_sum = sum;
        } else if (sum != first_sum) {
          slices_differ = true;
        }
        int krylov = 0;
        for (const NewtonStep &item : solved.history) {
          krylov += item.krylov_iterations;
        }
        std::printf(
            "step %d/%d gamma %.8e converged 1 newton %zu krylov %d "
            "correction %.3e residual %.3e peak %.6e rss_GiB %.2f seconds %.3f\n",
            step, options.steps, gamma, solved.history.size(), krylov,
            solved.history.empty() ? 0.0
                                   : solved.history.back().correction_over_reference,
            solved.projected_residual, global_peak, max_rss, seconds);
        std::fflush(stdout);
        write_provenance(options, completed + 1, -1);
      }
      start.deformation = std::move(solved.deformation);
      start.macroscopic = macroscopic;
      ++completed;
    }

    if (!step_failed && rank == 0) {
      if (options.render != 0) {
        render_status = render_animation(options, peak);
      }
      const int exit_status = render_status == 0 ? 0 : 1;
      write_provenance(options, completed, exit_status);
      std::printf("polycrystal_shear status=%s grid=%dx%dx%d steps_completed=%d "
                  "slices_differ=%d law=finite-strain-fcc\n",
                  exit_status == 0 ? "ok" : "render-failed", options.size,
                  options.size, options.size, completed, slices_differ ? 1 : 0);
      std::fflush(stdout);
    }
    MPI_Bcast(&render_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return render_status == 0 ? 0 : 1;
}
