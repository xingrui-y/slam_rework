#include "image_ops.h"
#include "cuda_utils.h"
#include "vector_math.h"
#include "intrinsic_matrix.h"
#include <opencv2/cudawarping.hpp>

namespace slam
{
namespace cuda
{

void build_depth_pyramid(const cv::cuda::GpuMat &base_depth, std::vector<cv::cuda::GpuMat> &pyramid, const int &max_level)
{
    assert(max_level == pyramid.size());
    base_depth.copyTo(pyramid[0]);

    for (int level = 1; level < max_level; ++level)
    {
        cv::cuda::resize(pyramid[level - 1], pyramid[level], cv::Size(0, 0), 0.5, 0.5);
    }
}

void build_intensity_pyramid(const cv::cuda::GpuMat &base_intensity, std::vector<cv::cuda::GpuMat> &pyramid, const int &max_level)
{
    assert(max_level == pyramid.size());
    base_intensity.copyTo(pyramid[0]);

    for (int level = 1; level < max_level; ++level)
    {
        cv::cuda::pyrDown(pyramid[level - 1], pyramid[level]);
    }
}

__global__ void compute_intensity_derivative_kernel(cv::cuda::PtrStepSz<float> intensity, cv::cuda::PtrStep<float> intensity_dx, cv::cuda::PtrStep<float> intensity_dy)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x > intensity.cols - 1 || y > intensity.rows - 1)
        return;

    int x10 = max(x - 1, 0);
    int x01 = min(x + 1, intensity.cols);
    int y10 = max(y - 1, 0);
    int y01 = min(y + 1, intensity.rows);

    intensity_dx.ptr(y)[x] = (intensity.ptr(y)[x01] - intensity.ptr(y)[x10]) * 0.5;
    intensity_dy.ptr(y)[x] = (intensity.ptr(y01)[x] - intensity.ptr(y10)[x]) * 0.5;
}

void build_intensity_derivative_pyramid(const std::vector<cv::cuda::GpuMat> &intensity, std::vector<cv::cuda::GpuMat> &sobel_x, std::vector<cv::cuda::GpuMat> &sobel_y)
{
    const int max_level = intensity.size();

    assert(max_level == sobel_x.size());
    assert(max_level == sobel_y.size());

    for (int level = 0; level < max_level; ++level)
    {
        const int cols = intensity[level].cols;
        const int rows = intensity[level].rows;

        dim3 thread(8, 8);
        dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

        if (sobel_x[level].empty())
            sobel_x[level].create(rows, cols, CV_32FC1);
        if (sobel_y[level].empty())
            sobel_y[level].create(rows, cols, CV_32FC1);

        compute_intensity_derivative_kernel<<<block, thread>>>(intensity[level], sobel_x[level], sobel_y[level]);
    }
}

__global__ void back_project_kernel(const cv::cuda::PtrStepSz<float> depth, cv::cuda::PtrStep<float4> vmap, DeviceIntrinsicMatrix intrinsics)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x > depth.cols - 1 || y > depth.rows - 1)
        return;

    float z = depth.ptr(y)[x];
    z = (z == z) ? z : 0;

    vmap.ptr(y)[x] = make_float4(z * (x - intrinsics.cx) * intrinsics.invfx, z * (y - intrinsics.cy) * intrinsics.invfy, z, 1.0f);
}

void build_point_cloud_pyramid(const std::vector<cv::cuda::GpuMat> &depth_pyr, std::vector<cv::cuda::GpuMat> &point_cloud_pyr, const IntrinsicMatrixPyramidPtr intrinsics_pyr)
{
    assert(depth_pyr.size() == point_cloud_pyr.size());
    assert(intrinsics_pyr->get_max_level() == depth_pyr.size());

    for (int level = 0; level < depth_pyr.size(); ++level)
    {
        const cv::cuda::GpuMat &depth = depth_pyr[level];
        cv::cuda::GpuMat &point_cloud = point_cloud_pyr[level];
        IntrinsicMatrixPtr intrinsic_matrix = (*intrinsics_pyr)[level];

        const int cols = depth.cols;
        const int rows = depth.rows;

        if (point_cloud.empty())
            point_cloud.create(rows, cols, CV_32FC4);

        dim3 thread(8, 8);
        dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

        back_project_kernel<<<block, thread>>>(depth, point_cloud, *intrinsic_matrix);
    }
}

__global__ void compute_nmap_kernel(cv::cuda::PtrStepSz<float4> vmap, cv::cuda::PtrStep<float4> nmap)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x > vmap.cols - 1 || y > vmap.rows - 1)
        return;

    int x10 = max(x - 1, 0);
    int x01 = min(x + 1, vmap.cols);
    int y10 = max(y - 1, 0);
    int y01 = min(y + 1, vmap.rows);

    float4 v00 = vmap.ptr(y)[x10];
    float4 v01 = vmap.ptr(y)[x01];
    float4 v10 = vmap.ptr(y10)[x];
    float4 v11 = vmap.ptr(y01)[x];

    nmap.ptr(y)[x] = make_float4(normalised(cross(v01 - v00, v11 - v10)), 1.f);
}

void build_normal_pyramid(const std::vector<cv::cuda::GpuMat> &vmap_pyr, std::vector<cv::cuda::GpuMat> &nmap_pyr)
{
    assert(vmap_pyr.size() == nmap_pyr.size());
    for (int level = 0; level < vmap_pyr.size(); ++level)
    {
        const cv::cuda::GpuMat &vmap = vmap_pyr[level];
        cv::cuda::GpuMat &nmap = nmap_pyr[level];

        const int cols = vmap.cols;
        const int rows = vmap.rows;

        if (nmap.empty())
            nmap.create(rows, cols, CV_32FC4);

        dim3 thread(8, 8);
        dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

        compute_nmap_kernel<<<block, thread>>>(vmap, nmap);
    }

    safe_call(cudaDeviceSynchronize());
    safe_call(cudaGetLastError());
}

} // namespace cuda
} // namespace slam