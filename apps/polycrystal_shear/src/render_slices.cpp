// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
/** @file render_slices.cpp
 * @brief Encode existing polycrystal stress slices without invoking a solver.
 */
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
extern char **environ;
namespace {
struct Options {
  int size = 0;
  int steps = 0;
  int fps = 60;
  std::string outdir;
};
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
  try {
    Options options;
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (i + 1 == argc) throw std::invalid_argument("missing option value");
      const std::string value = argv[++i];
      if (flag == "--size")
        options.size = std::stoi(value);
      else if (flag == "--steps")
        options.steps = std::stoi(value);
      else if (flag == "--fps")
        options.fps = std::stoi(value);
      else if (flag == "--outdir")
        options.outdir = value;
      else
        throw std::invalid_argument("unknown renderer option: " + flag);
    }
    if (options.size < 1 || options.steps < 1 || options.fps < 1 ||
        options.outdir.empty())
      throw std::invalid_argument(
          "--size, --steps, --fps and --outdir must be positive");
    const std::size_t count = static_cast<std::size_t>(options.size) * options.size;
    std::vector<float> sample(count);
    double peak = 0.0;
    for (int step = 1; step <= options.steps; ++step) {
      char name[64];
      std::snprintf(name, sizeof(name), "slice-%05d.f32", step);
      std::ifstream in(std::filesystem::path(options.outdir) / name,
                       std::ios::binary);
      in.read(reinterpret_cast<char *>(sample.data()),
              static_cast<std::streamsize>(sample.size() * sizeof(float)));
      if (!in) throw std::runtime_error(std::string("missing slice ") + name);
      for (float value : sample) peak = std::max(peak, static_cast<double>(value));
    }
    const int status = render_animation(options, peak);
    std::printf("polycrystal_renderer status=%s\n",
                status == 0 ? "ok" : "render-failed");
    return status;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "polycrystal renderer: %s\n", error.what());
    return 2;
  }
}
