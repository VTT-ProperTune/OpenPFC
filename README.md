# OpenPFC

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/VTT-ProperTune/OpenPFC)](https://github.com/VTT-ProperTune/OpenPFC/releases/latest)
[![GitHub](https://img.shields.io/github/license/VTT-ProperTune/OpenPFC)](https://github.com/VTT-ProperTune/OpenPFC/blob/main/LICENSE)
![GitHub Repo stars](https://img.shields.io/github/stars/VTT-ProperTune/OpenPFC)

![Screenshot of OpenPFC simulation result](docs/img/screenshot.png)

Phase field crystal (PFC) is a semi-atomistic technique, containing atomic
resolution information of crystalline structures while operating on diffusive
time scales. PFC has an ability to simulate solidification and elastic-plastic
material response, coupled to a wide range of phenomena, including formation and
co-evolution of microstructural defects such as dislocations and stacking
faults, voids, defect formation in epitaxial growth, displacive phase
transitions, and electromigration.

OpenPFC is an open-source framework for high performance 3D phase field crystal
simulations. It is designed to scale up from a single laptop to exascale class
supercomputers. OpenPFC has succesfully used to simulate domain of size 8192 x
8192 x 4096 on CSC Mahti. 200 computing nodes were used, where each node
contains 128 cores, thus total 25600 cores were used. During the simulation, 25
TB of memory was utilized. The central part of the solver is Fast Fourier
Transform with time complexity of O(N log N), and there are no known limiting
bottlenecks, why larger models could not be calculated as well.

## Features

- scales up to tens of thousands of cores, demonstrably
- modern c++17 header only library, easy to use

## Installing

### Using singularity

### Compiling from source

Requirements:

- Compiler supporting C++17 standard. C++17 features [are
  available](https://gcc.gnu.org/projects/cxx-status.html) since GCC 5. Check
  your version number with `g++ --version`. The default compiler might be
  relatively old, and more recent version needs to be loaded with `module load
  gcc`. Do not try to compile with GCC 4.8.5. It will not work. At least GCC
  versions 9.4.0 (coming with Ubuntu 20.04) and 11.2 are working.
- [OpenMPI](https://www.open-mpi.org/). All recent versions should work. Tested
  with OpenMPI version 2.1.3. Again, you might need to load proper OpenMPI
  version with `module load openmpi/2.1.3`, for instance. Additionally, if cmake
  is not able to find proper OpenMPI installation, assistance might be needed by
  setting `MPI_ROOT`, e.g. `export MPI_ROOT=/share/apps/OpenMPI/2.1.3`.
- FFTW. Probably all versions will work. Tested with FFTW versions 3.3.2 and
  3.3.10. Again, cmake might need some assistance to find the libraries, which
  can be controlled with environment variable `FFTW_ROOT`. Depending how FFTW is
  installed to system, it might be in non-standard location and `module load
  fftw` is needed. You can use commands like `whereis fftw` or `ldconfig -p |
  grep fftw` to locate your FFTW installation, if needed.

Typically in clusters, these are already installed and can be loaded with an
on-liner

```bash
module load gcc openmpi fftw
```

For local Linux machines (or WSL2), packages usually can be installed from
repositories, e.g. in case of Ubuntu, the following should work:

```bash
sudo apt-get install -y gcc openmpi fftw
```

Some OpenPFC applications uses json files to provide initial data for
simulations. In principle, applications can also be built to receive initial
data in other ways, but as a widely known file format, we recommend to use json.
The choice for json package is [JSON for Modern C++](https://json.nlohmann.me/),
which can be installed either system-wide (e.g. `sudo apt-get install -y
nlohmann-json3-dev`), or by copying the [header
file](https://json.nlohmann.me/home/releases/) `json.hpp` to some appropriate
place in system, e.g. `$HOME/.local/include/nlohmann`. As this very likely is not
installed to system by administrators, a manual installation is needed.



OpenPFC uses [cmake](https://cmake.org/) to automate software building.
Compiling the software doesn't require a greater knowledge of cmake, but it
doesn't hurt either. First the source code must be downloaded to some
appropriate place:

```bash
git clone https://github.com/ProperTune-VTT/OpenPFC
```

Next step is to configure project. One might consider at least options
`CMAKE_BUILD_TYPE` and `CMAKE_INSTALL_PREFIX`

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/OpenPFC -S OpenPFC -B OpenPFC/build
```

Then, building:

```bash
cmake --build OpenPFC/build
```

And installation:

```bash
cmake --install OpenPFC/build
```

After that, one should find example codes from `<build_dir>/examples`.

OpenPFC depends from several other packages, where the somewhat standard ones
(OpenMP, FFTW, cmake) are assumed to be already installed. The uncommon
dependencies (HeFFTe, json) are downloaded during the configuration of the
project. 

# Examples

See our great article!

# Getting started

The basic concepts of the package are demonstrated with a simple diffusion model.

## Citing

```bibtex
@article{
  blaablaa
}
```
