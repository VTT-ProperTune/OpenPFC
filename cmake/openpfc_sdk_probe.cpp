// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cstddef>
#include <iostream>
#include <mpi.h>
#include <string>

#ifdef OPENPFC_SDK_LINK_PROBE
#include <openpfc/kernel/utils/logging.hpp>
#endif

#define OPENPFC_STRING_IMPL(x) #x
#define OPENPFC_STRING(x) OPENPFC_STRING_IMPL(x)

// Emit JSON without depending on a provider's header-only JSON package.
static void field(const char *key, const std::string &value) {
  std::cout << '"' << key << "\":\"";
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      std::cout << '\\' << c;
    } else if (c < 32) {
      std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
    } else {
      std::cout << c;
    }
  }
  std::cout << '"';
}

int main() {
#ifdef OPENPFC_SDK_LINK_PROBE
  // Force a real reference into the installed archive, even with --as-needed.
  pfc::log(pfc::k_null_logger, pfc::LogLevel::Debug, "SDK compatibility probe");
#endif
  std::cout << '{';
#if defined(_LIBCPP_VERSION)
  field("stdlib", "libc++");
  std::cout << ',';
  field("stdlib_version", OPENPFC_STRING(_LIBCPP_VERSION));
  std::cout << ',';
  field("stdlib_abi", OPENPFC_STRING(_LIBCPP_ABI_VERSION));
#elif defined(__GLIBCXX__)
  field("stdlib", "libstdc++");
  std::cout << ',';
  field("stdlib_version", OPENPFC_STRING(__GLIBCXX__));
  std::cout << ',';
  field("stdlib_abi", OPENPFC_STRING(_GLIBCXX_USE_CXX11_ABI));
#else
  field("stdlib", "unknown");
  std::cout << ',';
  field("stdlib_version", "unknown");
  std::cout << ',';
  field("stdlib_abi", "unknown");
#endif
  std::cout << ',';
#ifdef _GLIBCXX_DEBUG
  field("stdlib_debug", "ON");
#else
  field("stdlib_debug", "OFF");
#endif
  std::cout << ',';
  field("pointer_size", std::to_string(sizeof(void *)));
  std::cout << ',';
#if defined(OMPI_MAJOR_VERSION)
  field("mpi_vendor", "Open MPI");
  std::cout << ',';
  field("mpi_header_version",
        OPENPFC_STRING(OMPI_MAJOR_VERSION) "." OPENPFC_STRING(
            OMPI_MINOR_VERSION) "." OPENPFC_STRING(OMPI_RELEASE_VERSION));
#elif defined(I_MPI_VERSION)
  field("mpi_vendor", "Intel MPI");
  std::cout << ',';
  field("mpi_header_version", OPENPFC_STRING(I_MPI_VERSION));
#elif defined(CRAY_MPICH_VERSION)
  field("mpi_vendor", "Cray MPICH");
  std::cout << ',';
  field("mpi_header_version", OPENPFC_STRING(CRAY_MPICH_VERSION));
#elif defined(MPICH_VERSION)
  field("mpi_vendor", "MPICH");
  std::cout << ',';
  field("mpi_header_version", MPICH_VERSION);
#else
  field("mpi_vendor", "unknown");
  std::cout << ',';
  field("mpi_header_version", "unknown");
#endif
  // MPI_Get_library_version is allowed before MPI_Init; no launcher or GPU
  // is involved. This is evidence about the loaded library, not an ABI proof.
  char version[MPI_MAX_LIBRARY_VERSION_STRING]{};
  int length = 0;
  if (MPI_Get_library_version(version, &length) != MPI_SUCCESS) return 1;
  while (length > 0 && version[length - 1] == '\0') --length;
  std::cout << ',';
  field("mpi_library_version",
        std::string(version, static_cast<std::size_t>(length)));
  std::cout << "}\n";
}
