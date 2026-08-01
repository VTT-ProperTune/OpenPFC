/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_CONFIG_H
#define HEFFTE_CONFIG_H

#define Heffte_VERSION_MAJOR 2
#define Heffte_VERSION_MINOR 4
#define Heffte_VERSION_PATCH 1

#define Heffte_GIT_HASH "6e16996e5a3eb65c60fd62fd4d93c4b2df833033"

/* #undef Heffte_ENABLE_AVX */
/* #undef Heffte_ENABLE_AVX512 */

/* #undef Heffte_ENABLE_FFTW */
/* #undef Heffte_ENABLE_MKL */
#define Heffte_ENABLE_CUDA
/* #undef Heffte_ENABLE_ROCM */
/* #undef Heffte_ENABLE_ONEAPI */

/* #undef Heffte_ENABLE_MAGMA */

/* #undef Heffte_ENABLE_TRACING */

/* #undef Heffte_ENABLE_GPU_AWARE_MPI */

#if defined(Heffte_ENABLE_CUDA) || defined(Heffte_ENABLE_ROCM) ||                   \
    defined(Heffte_ENABLE_ONEAPI)
#define Heffte_ENABLE_GPU
#endif

#endif /* HEFFTE_CONFIG_H */
