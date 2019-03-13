#ifndef __CUDA_UTILS__
#define __CUDA_UTILS__

#include "intrinsic_matrix.h"
#include <iostream>
#include <cuda_runtime.h>

#define MAX_THREAD 1024
#define WARP_SIZE 32

#if defined(__GNUC__)
#define safe_call(expr) ___SafeCall(expr, __FILE__, __LINE__, __func__)
#else
#define safe_call(expr) ___SafeCall(expr, __FILE__, __LINE__)
#endif

static inline void error(const char *error_string, const char *file, const int line, const char *func)
{
    std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

static inline void ___SafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

template <class T, class U>
static inline int div_up(T a, U b)
{
    return (int)((a + b - 1) / b);
}

struct DeviceIntrinsicMatrix
{
    inline DeviceIntrinsicMatrix() = default;
    inline DeviceIntrinsicMatrix(IntrinsicMatrixPtr K) : fx(K->fx), fy(K->fy), cx(K->cx), cy(K->cy), invfx(K->invfx), invfy(K->invfy)
    {
    }

    float fx, fy, cx, cy, invfx, invfy;
};

#endif