<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Multi-GPU HeFFTe Research

## Purpose

This research directory is dedicated to demonstrating that we can run HeFFTe in a multi-GPU setup.

## Build Process

### Prerequisites

- CUDA module must be loaded: `module load cuda`
- FFTW3 library (available via pkg-config)
- CMake 3.19 or later
- MPI (OpenMPI 4.1.1 was used)

### Build Steps

1. **Load required modules:**
   ```bash
   module load cuda
   ```

2. **Create build directory:**
   ```bash
   cd ~/dev/heffte
   mkdir -p build_2.4.1_cuda
   cd build_2.4.1_cuda
   ```

3. **Configure CMake:**
   ```bash
   cmake .. \
     -DCMAKE_INSTALL_PREFIX=~/opt/heffte/2.4.1-cuda \
     -DHeffte_ENABLE_FFTW=ON \
     -DHeffte_ENABLE_CUDA=ON \
     -DCMAKE_BUILD_TYPE=Release
   ```

4. **Build:**
   ```bash
   make -j$(nproc)
   ```

5. **Install:**
   ```bash
   make install
   ```

### Build Configuration Summary

- **Version:** HeFFTe 2.4.1
- **Source:** `~/dev/heffte`
- **Install prefix:** `~/opt/heffte/2.4.1-cuda`
- **Backends enabled:**
  - FFTW: ON
  - CUDA: ON
  - GPU-aware MPI: ON (enabled by default when CUDA is enabled)
- **Build type:** Release
- **CUDA version:** 12.9.86
- **MPI:** OpenMPI 4.1.1

### Installation Location

The library is installed to:
- **Library:** `~/opt/heffte/2.4.1-cuda/lib64/libheffte.so`
- **Headers:** `~/opt/heffte/2.4.1-cuda/include/`
- **CMake config:** `~/opt/heffte/2.4.1-cuda/lib64/cmake/Heffte/`

## Demo

### Building the Example

The directory contains a CUDA FFT example (`cuda_fft_example.cpp`) that demonstrates:
- Forward FFT using CUDA backend
- **Laplace operator computation in Fourier domain** (multiply by -k²)
- Inverse FFT to get Laplacian in real space
- **All operations performed entirely on GPU** (no CPU transfers during computation)
- Verification against analytical solution: ∇²(sin(x)sin(y)sin(z)) = -3*sin(x)sin(y)sin(z)

**Key Features:**
- Uses CUDA kernels for GPU operations (no CPU-GPU transfers during computation)
- Demonstrates spectral method approach: derivatives computed in Fourier space
- Clean GPU-only implementation suitable for production use

To build:

```bash
cd research/multigpu_heffte
mkdir build
cd build
module load cuda
cmake ..
make
```

To run:

```bash
# Run with 1 MPI rank (single GPU) - WORKS locally
mpirun -np 1 ./cuda_fft_example

# Run with multiple ranks (for multi-GPU testing) - REQUIRES GPU-aware MPI
mpirun -np 2 ./cuda_fft_example
```

**Note:** 
- **Single GPU (1 MPI rank) works fine** - verified locally ✓
- **Multi-GPU (2+ MPI ranks) works** - verified on cluster with GPU-aware MPI disabled ✓
  - Without GPU-aware MPI: HeFFTe automatically transfers data to CPU for MPI communication, then back to GPU
  - This works but is slower than GPU-aware MPI
  - To use this mode, rebuild with `HEFFTE_NO_GPU_AWARE=1` (see instructions below)

### Implementation Details

The example demonstrates a clean GPU-only workflow:

1. **Input creation** (CPU): Creates test function `sin(x)*sin(y)*sin(z)` on a 64×64×64 grid
2. **Transfer to GPU**: One-time transfer using `heffte::gpu::transfer().load()`
3. **Forward FFT** (GPU): Transforms to Fourier space using HeFFTe CUDA backend
4. **Laplacian operator** (GPU): CUDA kernel multiplies each Fourier coefficient by `-k²`
5. **Inverse FFT** (GPU): Transforms back to real space
6. **Scaling** (GPU): CUDA kernel applies normalization factor `1/N`
7. **Verification** (CPU): One-time transfer back to CPU for comparison with analytical solution

**CUDA Kernels:**
- `apply_laplacian_kernel`: Applies Laplacian operator in Fourier domain
- `scale_kernel`: Applies normalization scaling

Both kernels operate directly on GPU memory, avoiding CPU-GPU transfers during computation.

## Checking GPU-Aware MPI Support

GPU-aware MPI is required for multi-GPU HeFFTe operations. It allows MPI to directly communicate GPU memory buffers without explicit CPU-GPU transfers.

### Method 1: Using `ompi_info` Command

Check if OpenMPI was built with CUDA support:

```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

**Expected output if GPU-aware MPI is enabled:**
```
mca:mpi:base:param:mpi_built_with_cuda_support:value:true
```

**Expected output if GPU-aware MPI is NOT enabled:**
```
mca:mpi:base:param:mpi_built_with_cuda_support:value:false
```

### Method 2: Check MCA Parameters

You can also check CUDA-related MCA parameters:

```bash
ompi_info --parsable --all | grep -i cuda
```

Look for parameters like:
- `mca:btl:smcuda` (CUDA shared memory BTL)
- `mca:pml:ob1:param:btl_smcuda` (CUDA support in PML)

### Method 3: Programmatic Check (C/C++)

Create a simple test program to check at runtime:

```cpp
#include <stdio.h>
#include <mpi.h>
#include <mpi-ext.h> /* Needed for CUDA-aware check */

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int cuda_support = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cuda_support = MPIX_Query_cuda_support();
#endif

    if (cuda_support) {
        printf("This Open MPI installation has CUDA-aware support.\n");
    } else {
        printf("This Open MPI installation does NOT have CUDA-aware support.\n");
        printf("Multi-GPU HeFFTe operations may fail with segmentation faults.\n");
    }

    MPI_Finalize();
    return 0;
}
```

Compile and run:
```bash
mpicc -o check_cuda_mpi check_cuda_mpi.c
mpirun -np 1 ./check_cuda_mpi
```

### Method 4: Check HeFFTe Configuration

HeFFTe's GPU-aware MPI status can be checked during CMake configuration:

```bash
cd ~/dev/heffte/build_2.4.1_cuda
grep -i "GPU_AWARE" CMakeCache.txt
```

Or check the HeFFTe configuration summary:
```bash
grep -i "GPU_AWARE" ~/dev/heffte/build_2.4.1_cuda/configured/summary.txt
```

### Troubleshooting

**If GPU-aware MPI is NOT enabled:**

1. **Single GPU works, multi-GPU fails:** This is expected if GPU-aware MPI is not available. HeFFTe will try to use GPU-aware MPI for inter-GPU communication, which will fail.

2. **Solutions:**
   - Rebuild OpenMPI with CUDA support: `./configure --with-cuda=/path/to/cuda`
   - Or rebuild HeFFTe with GPU-aware MPI disabled (may reduce performance):
     ```bash
     cmake .. -DHeffte_ENABLE_CUDA=ON -DHeffte_ENABLE_GPU_AWARE_MPI=OFF
     ```

3. **Current status:** 
   - Our HeFFTe build has GPU-aware MPI enabled by default (when CUDA is enabled)
   - The OpenMPI 4.1.1 installation does NOT have GPU-aware MPI support (`mpi_built_with_cuda_support:value:false`)
   - This causes segmentation faults in multi-GPU scenarios when HeFFTe tries to use GPU-aware MPI for inter-GPU communication
   - Single GPU operations work fine, but multi-GPU requires GPU-aware MPI support

### Testing Multi-GPU Without GPU-Aware MPI

**Status: ✓ CONFIRMED - Multi-GPU works without GPU-aware MPI!**

We successfully tested HeFFTe with 2 GPUs on the cluster. Without GPU-aware MPI, HeFFTe automatically transfers data to CPU for MPI communication, then back to GPU. This works but is slower than GPU-aware MPI.

**To test multi-GPU (already done, but here's how to repeat):**

1. **Rebuild HeFFTe without GPU-aware MPI** (if not already done):
   ```bash
   cd ~/dev/heffte
   mkdir -p build_2.4.1_cuda_no_gpuaware
   cd build_2.4.1_cuda_no_gpuaware
   module load cuda
   cmake .. \
     -DCMAKE_INSTALL_PREFIX=~/opt/heffte/2.4.1-cuda-no-gpuaware \
     -DHeffte_ENABLE_FFTW=ON \
     -DHeffte_ENABLE_CUDA=ON \
     -DHeffte_ENABLE_GPU_AWARE_MPI=OFF \
     -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   make install
   ```

2. **Rebuild the example with the non-GPU-aware version:**
   ```bash
   cd research/multigpu_heffte/build
   export HEFFTE_NO_GPU_AWARE=1
   cmake ..
   make
   ```

3. **Submit job for 2 GPU test:**
   ```bash
   cd research/multigpu_heffte
   sbatch run_2gpu_test.sh
   ```

**Test Results:**
- ✓ Job completed successfully (ExitCode: 0:0)
- ✓ 2 MPI ranks running on 2 different GPUs (GPU 0 and GPU 1)
- ✓ Domain decomposition working: Rank 0 handles z=[0,31], Rank 1 handles z=[32,63]
- ✓ Forward FFT completed
- ✓ Laplacian operator applied in Fourier domain
- ✓ No segmentation faults

**Performance Note:** Without GPU-aware MPI, data is transferred CPU↔GPU for MPI communication, which adds overhead. For production use, GPU-aware MPI would be preferred for better performance.

### Checking GPU-Aware MPI Status

**Check OpenMPI GPU-aware support:**
```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

**Check HeFFTe GPU-aware MPI configuration:**
```bash
# Current build (with GPU-aware MPI)
grep -i "GPU_AWARE" ~/dev/heffte/build_2.4.1_cuda/CMakeCache.txt

# Test build (without GPU-aware MPI)
grep -i "GPU_AWARE" ~/dev/heffte/build_2.4.1_cuda_no_gpuaware/CMakeCache.txt
```

### Next Steps

- Create a more complex demo that uses multiple GPU cards
- Reference examples from `~/dev/heffte/examples/` to understand how to implement multi-GPU usage in practice
- Test multi-GPU performance with and without GPU-aware MPI
