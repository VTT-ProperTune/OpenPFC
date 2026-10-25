// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "cli.hpp"
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace openpfc_cli {
namespace {
constexpr int k_dt_null = 0;
constexpr int k_dt_needed = 1;
constexpr int k_dt_strtab = 5;
constexpr int k_dt_strsz = 10;
constexpr int k_dt_rpath = 15;
constexpr int k_dt_runpath = 29;
constexpr unsigned k_pt_load = 1;
constexpr unsigned k_pt_dynamic = 2;
constexpr unsigned k_pt_interp = 3;
constexpr unsigned k_sht_gnu_verneed = 0x6ffffffe;
constexpr std::uint16_t k_em_x86_64 = 62;
constexpr std::uint16_t k_em_aarch64 = 183;

class Image {
public:
  explicit Image(std::string bytes) : bytes_(std::move(bytes)) {}
  [[nodiscard]] std::size_t size() const { return bytes_.size(); }
  template <typename T> [[nodiscard]] T pod(std::size_t offset) const {
    if (offset > bytes_.size() || bytes_.size() - offset < sizeof(T))
      throw Error("ELF field extends past the end of the file");
    T value{};
    std::memcpy(&value, bytes_.data() + offset, sizeof(T));
    return value;
  }
  [[nodiscard]] std::string cstr(std::size_t offset) const {
    if (offset >= bytes_.size()) throw Error("ELF string extends past the file");
    const auto end = bytes_.find('\0', offset);
    if (end == std::string::npos) throw Error("ELF string is not NUL-terminated");
    return bytes_.substr(offset, end - offset);
  }
  [[nodiscard]] const std::string &bytes() const { return bytes_; }

private:
  std::string bytes_;
};

struct Segment {
  std::uint64_t offset = 0;
  std::uint64_t address = 0;
  std::uint64_t filesz = 0;
  std::uint64_t memsz = 0;
};

struct Inspected {
  std::uint16_t machine = 0;
  std::string interpreter;
  std::vector<std::string> needed;
  std::vector<std::string> rpath;
  std::vector<std::string> runpath;
  bool has_runpath = false;
  int glibc_major = 0;
  int glibc_minor = 0;
};

[[nodiscard]] std::uint64_t virtual_offset(const std::vector<Segment> &loads,
                                           std::uint64_t address) {
  for (const auto &segment : loads) {
    if (address < segment.address) continue;
    const auto delta = address - segment.address;
    if (delta < segment.filesz) return segment.offset + delta;
  }
  throw Error("ELF virtual address is outside the loadable file segments");
}

[[nodiscard]] std::vector<std::string> split_paths(const std::string &value) {
  std::vector<std::string> parts;
  std::string current;
  for (char character : value) {
    if (character == ':') {
      if (!current.empty()) parts.push_back(current);
      current.clear();
    } else if (character == '\n' || character == ',') {
      throw Error("runtime search path contains a newline or comma");
    } else {
      current += character;
    }
  }
  if (!current.empty()) parts.push_back(current);
  return parts;
}

[[nodiscard]] std::string expand_token(std::string token, const fs::path &origin) {
  auto replace = [&](const std::string &needle) {
    for (auto position = token.find(needle); position != std::string::npos;
         position = token.find(needle, position)) {
      token.replace(position, needle.size(), origin.string());
      position += origin.string().size();
    }
  };
  replace("${ORIGIN}");
  replace("$ORIGIN");
  if (token.find("$LIB") != std::string::npos ||
      token.find("${LIB}") != std::string::npos ||
      token.find("$PLATFORM") != std::string::npos ||
      token.find("${PLATFORM}") != std::string::npos)
    throw Error("RPATH token $LIB or $PLATFORM is not expanded: " + token);
  return token;
}

void note_glibc(int &major, int &minor, const std::string &symbol) {
  constexpr std::string_view prefix = "GLIBC_";
  if (!symbol.starts_with(prefix)) return;
  const auto body = std::string_view(symbol).substr(prefix.size());
  std::size_t consumed = 0;
  int first = 0;
  int second = 0;
  try {
    first = std::stoi(std::string(body), &consumed);
  } catch (const std::exception &) {
    return;
  }
  if (consumed >= body.size() || body[consumed] != '.') return;
  const auto rest = body.substr(consumed + 1);
  try {
    second = std::stoi(std::string(rest), &consumed);
  } catch (const std::exception &) {
    return;
  }
  if (first > major || (first == major && second > minor)) {
    major = first;
    minor = second;
  }
}

Inspected inspect(const Image &image) {
  if (image.size() < 64 || image.bytes().compare(0, 4,
                                                 "\x7f"
                                                 "ELF") != 0)
    throw Error("model is not an ELF file");
  if (image.pod<unsigned char>(4) != 2 || image.pod<unsigned char>(5) != 1)
    throw Error("only little-endian ELF64 models can be bundled");
  const auto type = image.pod<std::uint16_t>(16);
  if (type != 2 && type != 3)
    throw Error("ELF is neither an executable nor a shared object");
  Inspected result;
  result.machine = image.pod<std::uint16_t>(18);
  if (result.machine != k_em_x86_64 && result.machine != k_em_aarch64)
    throw Error("ELF machine is not x86_64 or aarch64");
  const auto phoff = image.pod<std::uint64_t>(32);
  const auto shoff = image.pod<std::uint64_t>(40);
  const auto phentsize = image.pod<std::uint16_t>(54);
  const auto phnum = image.pod<std::uint16_t>(56);
  const auto shentsize = image.pod<std::uint16_t>(58);
  auto shnum = image.pod<std::uint16_t>(60);
  if (phentsize < 56) throw Error("unexpected ELF program-header size");
  std::vector<Segment> loads;
  std::optional<std::uint64_t> dynamic_offset;
  std::optional<std::uint64_t> dynamic_size;
  for (unsigned index = 0; index < phnum; ++index) {
    const auto header = phoff + static_cast<std::uint64_t>(index) * phentsize;
    const auto kind = image.pod<std::uint32_t>(header);
    const auto offset = image.pod<std::uint64_t>(header + 8);
    const auto address = image.pod<std::uint64_t>(header + 16);
    const auto filesz = image.pod<std::uint64_t>(header + 32);
    const auto memsz = image.pod<std::uint64_t>(header + 40);
    if (kind == k_pt_load)
      loads.push_back(Segment{offset, address, filesz, memsz});
    else if (kind == k_pt_dynamic) {
      dynamic_offset = offset;
      dynamic_size = filesz;
    } else if (kind == k_pt_interp)
      result.interpreter = image.cstr(static_cast<std::size_t>(offset));
  }
  if (!dynamic_offset || !dynamic_size)
    return result; // No dynamic segment: a static executable has nothing to copy.
  std::optional<std::uint64_t> strtab;
  std::uint64_t strsz = 0;
  std::optional<std::uint64_t> rpath;
  std::optional<std::uint64_t> runpath;
  std::vector<std::uint64_t> needed;
  const auto end = *dynamic_offset + *dynamic_size;
  for (auto cursor = *dynamic_offset; cursor + 16 <= end; cursor += 16) {
    const auto tag = image.pod<std::int64_t>(static_cast<std::size_t>(cursor));
    const auto value =
        image.pod<std::uint64_t>(static_cast<std::size_t>(cursor + 8));
    if (tag == k_dt_null) break;
    if (tag == k_dt_needed)
      needed.push_back(value);
    else if (tag == k_dt_strtab)
      strtab = value;
    else if (tag == k_dt_strsz)
      strsz = value;
    else if (tag == k_dt_rpath)
      rpath = value;
    else if (tag == k_dt_runpath)
      runpath = value;
  }
  if (!needed.empty() && !strtab) throw Error("ELF dynamic string table is missing");
  const auto strings = strtab ? virtual_offset(loads, *strtab) : 0;
  auto text = [&](std::uint64_t offset) {
    if (strsz != 0 && offset >= strsz) throw Error("ELF string is outside DT_STRSZ");
    return image.cstr(static_cast<std::size_t>(strings + offset));
  };
  for (const auto offset : needed) result.needed.push_back(text(offset));
  if (runpath) {
    result.has_runpath = true;
    result.runpath = split_paths(text(*runpath));
  } else if (rpath) {
    result.rpath = split_paths(text(*rpath));
  }
  if (shoff == 0 || shentsize < 64) return result;
  if (shnum == 0)
    shnum = static_cast<std::uint16_t>(
        image.pod<std::uint64_t>(static_cast<std::size_t>(shoff + 32)));
  for (unsigned index = 0; index < shnum; ++index) {
    const auto section = shoff + static_cast<std::uint64_t>(index) * shentsize;
    if (image.pod<std::uint32_t>(static_cast<std::size_t>(section + 4)) !=
        k_sht_gnu_verneed)
      continue;
    const auto offset =
        image.pod<std::uint64_t>(static_cast<std::size_t>(section + 24));
    const auto size =
        image.pod<std::uint64_t>(static_cast<std::size_t>(section + 32));
    const auto link =
        image.pod<std::uint32_t>(static_cast<std::size_t>(section + 40));
    if (link >= shnum) continue;
    const auto strings_header = shoff + static_cast<std::uint64_t>(link) * shentsize;
    const auto string_offset =
        image.pod<std::uint64_t>(static_cast<std::size_t>(strings_header + 24));
    auto entry = offset;
    const auto limit = offset + size;
    while (entry + 16 <= limit) {
      const auto count =
          image.pod<std::uint16_t>(static_cast<std::size_t>(entry + 2));
      const auto auxiliary =
          entry + image.pod<std::uint32_t>(static_cast<std::size_t>(entry + 8));
      const auto next =
          image.pod<std::uint32_t>(static_cast<std::size_t>(entry + 12));
      auto item = auxiliary;
      for (unsigned child = 0; child < count && item + 16 <= limit; ++child) {
        const auto name =
            image.pod<std::uint32_t>(static_cast<std::size_t>(item + 8));
        const auto step =
            image.pod<std::uint32_t>(static_cast<std::size_t>(item + 12));
        note_glibc(result.glibc_major, result.glibc_minor,
                   image.cstr(static_cast<std::size_t>(string_offset + name)));
        if (step == 0) break;
        item += step;
      }
      if (next == 0) break;
      entry += next;
    }
  }
  return result;
}

struct CacheEntry {
  std::uint32_t flags = 0;
  std::uint64_t hwcap = 0;
  std::string soname;
  std::string path;
};

[[nodiscard]] bool plausible_string(const std::string &bytes, std::size_t offset) {
  return offset < bytes.size() && bytes[offset] > 32 && bytes[offset] < 127;
}

[[nodiscard]] std::vector<CacheEntry> parse_new_cache(const std::string &bytes,
                                                      std::size_t base) {
  std::vector<CacheEntry> entries;
  if (base > bytes.size() || bytes.size() - base < 48) return entries;
  if (bytes.compare(base, 20, "glibc-ld.so.cache1.1") != 0) return entries;
  Image image(bytes);
  const auto count = image.pod<std::uint32_t>(base + 20);
  const auto table = base + 48;
  if (count > (bytes.size() - std::min(table, bytes.size())) / 24) return entries;
  const auto probe = table;
  const auto key = image.pod<std::uint32_t>(probe + 4);
  const bool absolute = count == 0 || plausible_string(bytes, base + key);
  const auto relative_base = table + static_cast<std::uint64_t>(count) * 24;
  const bool relative = count == 0 || plausible_string(bytes, relative_base + key);
  if (!absolute && !relative) return entries;
  const auto origin = absolute ? base : relative_base;
  entries.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    const auto entry = table + static_cast<std::uint64_t>(index) * 24;
    CacheEntry parsed;
    parsed.flags = image.pod<std::uint32_t>(static_cast<std::size_t>(entry));
    const auto name = image.pod<std::uint32_t>(static_cast<std::size_t>(entry + 4));
    const auto value = image.pod<std::uint32_t>(static_cast<std::size_t>(entry + 8));
    parsed.hwcap = image.pod<std::uint64_t>(static_cast<std::size_t>(entry + 16));
    if (!plausible_string(bytes, origin + name) ||
        !plausible_string(bytes, origin + value))
      continue;
    entries.push_back(CacheEntry{parsed.flags, parsed.hwcap,
                                 image.cstr(origin + name),
                                 image.cstr(origin + value)});
  }
  return entries;
}

// glibc stores each cache key as an offset from the new-cache header
// (elf/dl-cache.c, cache_data + key). The bytes at that offset are the soname; the
// value offset is the absolute path. Older files embed this header after the legacy
// cache.
[[nodiscard]] std::unordered_map<std::string, std::string>
load_cache(std::uint16_t machine) {
  std::unordered_map<std::string, int> scores;
  std::unordered_map<std::string, std::string> paths;
  std::string bytes;
  try {
    bytes = read("/etc/ld.so.cache");
  } catch (const Error &) {
    return paths;
  }
  std::vector<CacheEntry> entries;
  if (bytes.starts_with("glibc-ld.so.cache1.1"))
    entries = parse_new_cache(bytes, 0);
  else if (bytes.starts_with("ld.so-1.7.0")) {
    const auto magic = bytes.find("glibc-ld.so.cache1.1");
    if (magic != std::string::npos) entries = parse_new_cache(bytes, magic);
  }
  const std::uint32_t preferred = machine == k_em_x86_64    ? 0x0300u
                                  : machine == k_em_aarch64 ? 0x0a00u
                                                            : 0u;
  for (const auto &entry : entries) {
    const auto kind = entry.flags & 0xffu;
    if (kind != 1 && kind != 3) continue;
    const auto arch = entry.flags & 0xff00u;
    if (preferred != 0 && arch != 0 && arch != preferred) continue;
    std::error_code error;
    if (!fs::is_regular_file(entry.path, error)) continue;
    int score = (arch == preferred ? 4 : 0) + (entry.hwcap == 0 ? 1 : 0);
    const auto found = scores.find(entry.soname);
    if (found != scores.end() && found->second >= score) continue;
    scores[entry.soname] = score;
    paths[entry.soname] = entry.path;
  }
  return paths;
}

[[nodiscard]] std::vector<fs::path> default_directories(std::uint16_t machine) {
  std::vector<fs::path> directories{"/lib64", "/usr/lib64", "/lib", "/usr/lib"};
  if (machine == k_em_x86_64) {
    directories.emplace_back("/lib/x86_64-linux-gnu");
    directories.emplace_back("/usr/lib/x86_64-linux-gnu");
  } else if (machine == k_em_aarch64) {
    directories.emplace_back("/lib/aarch64-linux-gnu");
    directories.emplace_back("/usr/lib/aarch64-linux-gnu");
  }
  return directories;
}

[[nodiscard]] std::optional<fs::path> existing_file(const fs::path &candidate) {
  std::error_code error;
  // Follow a symlink to a regular file. A directory, dangling link or special
  // node is not a library we can copy.
  if (!fs::is_regular_file(candidate, error)) return std::nullopt;
  return absolute_path(candidate);
}

[[nodiscard]] std::optional<fs::path>
search_dirs(const std::string &soname, const std::vector<fs::path> &directories) {
  for (const auto &directory : directories) {
    if (auto found = existing_file(directory / soname)) return found;
  }
  return std::nullopt;
}
} // namespace

RuntimeClosure dynamic_closure(const fs::path &executable) {
  const auto program = absolute_path(executable);
  require_executable(program);
  const auto main_image = inspect(Image(read(program)));
  RuntimeClosure closure;
  closure.executable = program;
  closure.interpreter = main_image.interpreter;
  closure.glibc_major = main_image.glibc_major;
  closure.glibc_minor = main_image.glibc_minor;
  const auto cache = load_cache(main_image.machine);
  const auto defaults = default_directories(main_image.machine);
  std::vector<std::string> environment_path;
  for (const auto &part : split_paths(env("LD_LIBRARY_PATH")))
    environment_path.push_back(part);
  struct Pending {
    fs::path path;
    Inspected elf;
  };
  // Copy an item out before appending. Appending can reallocate the queue.
  std::vector<Pending> queue{Pending{program, main_image}};
  std::unordered_set<std::string> walked;
  std::unordered_map<std::string, fs::path> by_soname;
  for (std::size_t index = 0; index < queue.size(); ++index) {
    const auto current = queue[index];
    std::vector<fs::path> directories;
    auto add = [&](const std::vector<std::string> &tokens, const fs::path &origin) {
      for (const auto &token : tokens) {
        const auto expanded = expand_token(token, origin);
        if (!expanded.empty()) directories.emplace_back(expanded);
      }
    };
    if (current.elf.has_runpath) {
      for (const auto &entry : environment_path) directories.emplace_back(entry);
      add(current.elf.runpath, current.path.parent_path());
    } else {
      add(current.elf.rpath, current.path.parent_path());
      if (!main_image.has_runpath && current.path != program)
        add(main_image.rpath, program.parent_path());
      for (const auto &entry : environment_path) directories.emplace_back(entry);
    }
    for (const auto &soname : current.elf.needed) {
      std::optional<fs::path> resolved = search_dirs(soname, directories);
      if (!resolved) {
        const auto cached = cache.find(soname);
        if (cached != cache.end()) resolved = existing_file(cached->second);
      }
      if (!resolved) resolved = search_dirs(soname, defaults);
      if (!resolved)
        throw Error("unresolved runtime library " + soname +
                    "; load the build's compiler/MPI modules first");
      fs::path canonical = *resolved;
      std::error_code error;
      if (auto real = fs::canonical(*resolved, error); !error) canonical = real;
      const auto inserted = by_soname.emplace(soname, canonical);
      if (!inserted.second && inserted.first->second != canonical)
        throw Error("runtime library " + soname +
                    " resolves to two different files");
      if (!inserted.second) continue;
      auto nested = inspect(Image(read(canonical)));
      if (nested.machine != main_image.machine)
        throw Error("runtime library machine does not match the model: " +
                    canonical.string());
      closure.libraries.push_back(
          RuntimeLibrary{soname, canonical, nested.glibc_major, nested.glibc_minor});
      if (walked.insert(canonical.string()).second)
        queue.push_back(Pending{canonical, std::move(nested)});
    }
  }
  return closure;
}
} // namespace openpfc_cli
