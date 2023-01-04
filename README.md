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
- modern c++17 header only framework, easy to use

## Installing

### Using singularity

- Todo

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
The choice for json package is [JSON for Modern C++](https://json.nlohmann.me/).
There exists packages for certain Linux distributions (`nlohmann-json3-dev` for
Ubuntu, `json-devel` for Centos) for easy install. If the system-wide installation
is not found, the library is downloaded from GitHub during the configuration.

The last and most important dependency in order to use OpenPFC is
[HeFFTe](https://icl.utk.edu/fft/), which is our choice for parallel FFT
implementation. The instructions to install HeFFTe can be found from
[here](https://mkstoyanov.bitbucket.io/heffte/md_doxygen_installation.html).
HeFFTe can be downloaded from <https://bitbucket.org/icl/heffte/downloads/>.

If HeFFTe is installed to some non-standard location, cmake is unable to find it
when configuring OpenPFC. To overcome this problem, the install path of HeFFTe
can be set into environment variable `CMAKE_PREFIX_PATH`. For example, if HeFFe
is installed to `$HOME/opt/heffte/2.3`, the following is making cmake to find
HeFFTe succesfully:

```bash
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.3:$CMAKE_PREFIX_PATH
```

During the configuration, OpenPFC prefers local installations, thus if HeFFTe is
already installed and founded, it will be used. For convenience, there is a
fallback method to fetch HeFFTe sources from internet and build it concurrently
with OpenPFC. In general, however, it is better to build and install programs
one at a time. So, make sure you have HeFFTe installed and working on your
system before continuing.

OpenPFC uses [cmake](https://cmake.org/) to automate software building. First
the source code must be downloaded to some appropriate place:

```bash
git clone https://github.com/ProperTune-VTT/OpenPFC
cd OpenPFC
```

Next step is to configure project. One might consider at least setting option
`CMAKE_BUILD_TYPE` to `Debug` or `Release`. For large scale simulations, make
sure to use `Release` as it turns on compiler optimizations.

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
```

Keep on mind, that configuration will download HeFFTe if the local installation
is not found. To use local installation instead, add HeFFTe path to environment
variable `CMAKE_PREFIX_PATH` or add `Heffte_DIR` option to point where HeFFTe
configuration files are installed. Typical configuration command in cluster
environment is something like

```bash
module load gcc openmpi fftw
CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.3:$CMAKE_PREFIX_PATH
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/opt/openpfc \
      -S . -B build
```

Then, building can be done with command  `cmake --build build`. After build
finishes, one should find example codes from `./build/examples` and apps from
`./build/apps`. Installation to path defined by `CMAKE_INSTALL_PREFIX` can be
done with `cmake --install build`.

## Getting started

OpenPFC is a [software framework][software framework]. It doesn't give you
ready-made solutions, but a platform on which you can start building your own
scalable PFC code. We will familiarize ourselves with the construction of the
model with the help of a simple diffusion model in a later stage of the
documentation. However, let's give a tip already at this stage, how to start the
development work effectively. Our "hello world" code is as follows:

```cpp
#include <iostream>
#include <openpfc/openpfc.hpp>

using namespace std;
using namespace pfc;

int main() {
  World world({32, 32, 32});
  cout << world << endl;
}
```

To compile, `CMakeLists.txt` is needed. Minimal `CMakeLists.txt` is:

```cmake
cmake_minimum_required(VERSION 3.15)
project(helloworld)
find_package(OpenPFC REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main OpenPFC)
```

With the help of `CMakeLists.txt`, build and compilation of application is
straightforward:

```bash
cmake -S . -B build
cmake --build
./build/main
```

[software framework]: Software_framework

## Examples

A bigger application example is Tungsten model. Todo.

## Citing

- Todo

```bibtex
@article{
  blaablaa
}
```
