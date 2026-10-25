// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "cli.hpp"
#include <iostream>

namespace openpfc_cli {
void print_help(const std::string &command) {
  if (command.empty() || command == "help") {
    std::cout << "OpenPFC native C++ CLI\n\n"
                 "  openpfc --version\n"
                 "  openpfc doctor [--sdk PREFIX]\n"
                 "  openpfc apps\n"
                 "  openpfc init DIRECTORY --model=diffusion\n"
                 "  openpfc init CASE [--app NAME] [--preset NAME]\n"
                 "  openpfc compile MODEL.cpp [--sdk PREFIX]\n"
                 "  openpfc run MODEL.cpp INPUT.json [--sdk PREFIX]\n"
                 "  openpfc compile CASE [--source CHECKOUT]\n"
                 "  openpfc run CASE [--ranks N]\n"
                 "  openpfc bundle MODEL.cpp -o MODEL.sif --base BASE.sif\n"
                 "  openpfc export CACHED_RECORD NEW_DIRECTORY\n\n"
                 "Use openpfc COMMAND --help or openpfc help COMMAND for details.\n"
                 "Source builds require a compatible installed SDK and compiler/MPI "
                 "environment.\n"
                 "Cache: $OPENPFC_CACHE, then $XDG_CACHE_HOME/openpfc or "
                 "~/.cache/openpfc.\n";
  } else if (command == "doctor") {
    std::cout
        << "Usage: openpfc doctor [--sdk PREFIX] [options]\n\n"
           "Check the CPU model build environment using the installed SDK "
           "compatibility checker.\n"
           "This compiles, links and runs a small host probe; it never fetches or "
           "builds the SDK.\n"
           "Reports the resolved SDK, compiler, MPI launcher and recorded SDK "
           "backends.\n"
           "GPU hardware and MPI launches are not tested. Run once outside "
           "mpirun/srun,\n"
           "in an allocation or permitted build environment on LUMI.\n\n"
           "Options: --cc COMPILER --cxx COMPILER --toolchain FILE\n"
           "         --generator NAME --build-type TYPE --backend cpu --profile "
           "local-cpu\n"
           "Defaults: Ninja, Release, CPU. Load matching compiler/MPI modules "
           "first.\n"
           "SDK discovery: --sdk, OpenPFC_DIR, SDK beside CLI, then "
           "CMAKE_PREFIX_PATH.\n"
           "Exit status: 0 compatible; 1 failed check; 2 invalid usage or execution "
           "context.\n"
           "Probe files and logs remain in the printed diagnostics directory.\n";
  } else if (command == "apps") {
    std::cout << "Usage: openpfc apps\n\n"
                 "List shipped applications, presets and build backends.\n"
                 "This catalog does not check which binaries are installed.\n";
  } else if (command == "init") {
    std::cout
        << "Usage: openpfc init DIRECTORY --model=diffusion\n"
           "       openpfc init CASE [--app NAME] [--preset NAME]\n\n"
           "--model=diffusion creates editable model.cpp, input.json and "
           "README.md.\n"
           "Then run openpfc doctor, compile model.cpp, and run model.cpp "
           "input.json.\n"
           "The starter uses periodic finite-difference diffusion on CPU.\n\n"
           "Create a new catalogued app case with input.json and openpfc.json.\n"
           "Default app: tungsten. List apps and presets with openpfc apps.\n"
           "The destination must not exist. No SDK or compiler is needed to "
           "initialize.\n"
           "--model cannot be combined with --app or --preset.\n";
  } else if (command == "compile" || command == "run") {
    if (command == "compile") {
      std::cout << "Usage: openpfc compile MODEL.cpp [options]\n"
                   "       openpfc compile CASE [case options]\n\n"
                   "Compile an editable C++20 model with main() against an "
                   "installed SDK.\n"
                   "Unchanged source and toolchain reuse a verified cached build.\n";
    } else {
      std::cout << "Usage: openpfc run MODEL.cpp INPUT.json [options]\n"
                   "       openpfc run CASE [case options]\n\n"
                   "Compile if needed, then run frozen source and inputs in a new "
                   "attempt.\n"
                   "The model receives the input pathname as argv[1]. Outputs "
                   "belong under results/.\n"
                   "Saved logs, outputs and status are in the printed attempt "
                   "directory.\n";
    }
    std::cout
        << "\nSource build options:\n"
           "  --sdk PREFIX           Installed SDK prefix or OpenPFCConfig.cmake "
           "directory\n"
           "  --file PATH            Additional local header or translation unit "
           "(repeatable)\n"
           "  --source-root DIR      Root for local sources and relative --file "
           "paths\n"
           "  --cc COMPILER --cxx COMPILER --toolchain FILE\n"
           "  --generator NAME       CMake generator (default Ninja)\n"
           "  --build-type TYPE      Release (default), Debug, RelWithDebInfo or "
           "MinSizeRel\n"
           "  --jobs N               Build parallelism (default 2)\n"
           "  --define NAME=VALUE    Model compile definition (repeatable)\n"
           "  --backend cpu          Qualified source backend\n"
           "  --profile local-cpu    CPU source profile (local also accepted)\n"
           "\nSDK discovery: --sdk, OpenPFC_DIR, SDK beside CLI, then "
           "CMAKE_PREFIX_PATH.\n"
           "Load matching compiler/MPI modules first. The CLI never fetches or "
           "rebuilds the SDK.\n";
    if (command == "run") {
      std::cout << "\nSource run options:\n"
                   "  --input-file PATH      Input data relative to INPUT.json "
                   "(repeatable)\n"
                   "  --ranks N              MPI ranks (default 1)\n"
                   "  --launcher NAME        auto (default), none, mpirun or srun\n"
                   "  --arg VALUE            Extra model argument (repeatable)\n"
                   "  --record-env NAME      Numerical runtime setting to record "
                   "(repeatable)\n"
                   "Invoke the source CLI once, outside mpirun/srun. "
                   "--launcher=none uses one rank.\n"
                   "Application failures propagate their exit code and retain "
                   "partial output.\n"
                   "\nCase options: --launcher NAME --ranks N --executable FILE "
                   "--dry-run\n";
    } else {
      std::cout << "\nCase options:\n"
                   "  --profile local|tohtori|lumi --source CHECKOUT\n"
                   "  --build-dir DIR --jobs N --no-test --dry-run\n"
                   "Catalogued cases build shipped apps through the checkout's "
                   "scripts/build.sh.\n";
    }
  } else if (command == "bundle") {
    std::cout
        << "Usage: openpfc bundle MODEL.cpp -o MODEL.sif --base BASE.sif "
           "[options]\n\n"
           "Package a compiled CPU model and its non-MPI runtime libraries into "
           "an Apptainer/Singularity image.\n"
           "The image runs that model. It does not contain a compiler, the "
           "OpenPFC sources, SDK headers or Python.\n"
           "MPI stays on the host. MODEL.sif.run binds the declared host stack "
           "and launches the image.\n"
           "Direct ./MODEL.sif INPUT.json works only when that environment "
           "already exposes the libraries the model needs.\n"
           "Under mpirun, MODEL.sif.run also bind-mounts TMPDIR so the "
           "process-manager session is visible.\n\n"
           "Qualified profile: --profile=local-cpu (local is accepted). "
           "--profile=tohtori-cuda and --profile=lumi-g are rejected as not "
           "qualified.\n"
           "--site=openmpi (default) or --site=cray selects the host MPI "
           "classification. Cray binds are not a ROCm qualification.\n"
           "Pass the same SDK and compiler options used to compile the model.\n\n"
           "Options:\n"
           "  -o, --output FILE      Destination image. Also writes FILE.sha256 "
           "and FILE.run\n"
           "  --base FILE            Pinned local SIF providing glibc and "
           "/bin/sh\n"
           "  --prepare-only         Stage the recipe under the cache and do not "
           "build an image\n"
           "  --runtime NAME         apptainer (default) or singularity\n"
           "  --host-prefix DIR      Host MPI/UCX prefix to bind read-only "
           "(repeatable)\n"
           "  --host-lib FILE        One fabric or MPI library that shares a "
           "directory with system libraries (repeatable)\n"
           "  --site NAME            openmpi or cray\n"
           "  --input FILE           Optional frozen input recorded in the image\n"
           "  Source build options are the same as openpfc compile.\n\n"
           "A system library directory such as /usr/lib64 cannot be a host prefix. "
           "libfabric, libibverbs and the other fabric libraries there need "
           "--host-lib. An adjacent provider directory is bind-mounted when it "
           "exists.\n"
           "Dependencies come from ELF NEEDED entries, not from executing the "
           "model. A build requires a host that can run apptainer/singularity "
           "build.\n"
           "--prepare-only needs no container runtime. The staged directory "
           "contains bundle.def, the payload and provenance.\n"
           "The image digest is the sidecar, not a field inside the image.\n"
           "Catalogued app cases continue to use scripts/bake_case.py.\n";
  } else if (command == "export") {
    std::cout
        << "Usage: openpfc export CACHED_RECORD NEW_DIRECTORY\n\n"
           "Copy a completed cached build or run to durable storage.\n"
           "Refuses existing destinations, active attempts and damaged artifacts.\n"
           "The model/ directory in a run contains a reusable CMake project.\n";
  } else {
    throw Error("unknown command: " + command + "; use openpfc --help");
  }
}
} // namespace openpfc_cli
