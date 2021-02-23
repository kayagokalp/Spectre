#pragma once
#include <cstdio>
#ifdef SMART
#define GPU
#endif

#ifdef GPU
#include "cublas_v2.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cusolverDn.h>
#endif

//#define DEBUG_CONSOLE
#ifdef DEBUG_CONSOLE
#  define DBG(x) x;
#else
#  define DBG(x) {}
#endif

const double pi = 3.14159265358979323846;
typedef double dtype;

//#define OMP_TIMER
#ifdef OMP_TIMER
#define TIME(str, s, e, x) s = omp_get_wtime(); x; e = omp_get_wtime(); cout << str << ": " << e - s << " secs\n";
//#define TIME(str, s, e, x) s = omp_get_wtime(); x; e = omp_get_wtime(); cout << e - s << " ";
#else
#define TIME(str, s, e, x) x
#endif

#ifdef GPU
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
#endif
