#include "image_ops.h"
#include "cuda_utils.h"
#include "vector_math.h"
#include "intrinsic_matrix.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>

namespace slam
{
namespace cuda
{

void imshow(const char *name, const cv::cuda::GpuMat image)
{
    cv::Mat image_cpu;
    image.download(image_cpu);
    cv::imshow(name, image_cpu);
    cv::waitKey(0);
}

void build_depth_pyramid(const cv::cuda::GpuMat &base_depth, std::vector<cv::cuda::GpuMat> &pyramid, const int &max_level)
{
    assert(max_level == pyramid.size());
    // base_depth.copyTo(pyramid[0]);
    cv::cuda::bilateralFilter(base_depth, pyramid[0], 5, 1, 1);

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
}

void resize_device_map(std::vector<cv::cuda::GpuMat> &map_pyr)
{
    for (int level = 1; level < map_pyr.size(); ++level)
    {
        cv::cuda::resize(map_pyr[level - 1], map_pyr[level], cv::Size(), 0.5, 0.5);
    }
}

__global__ void image_rendering_phong_shading_kernel(const cv::cuda::PtrStep<float4> vmap, const cv::cuda::PtrStep<float4> nmap, const float3 light_pos, cv::cuda::PtrStepSz<uchar4> dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= dst.cols || y >= dst.rows)
        return;

    float3 color;
    float3 point = make_float3(vmap.ptr(y)[x]);
    if (isnan(point.x))
    {
        const float3 bgr1 = make_float3(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
        const float3 bgr2 = make_float3(236.f / 255.f, 120.f / 255.f, 120.f / 255.f);

        float w = static_cast<float>(y) / dst.rows;
        color = bgr1 * (1 - w) + bgr2 * w;
    }
    else
    {
        float3 P = point;
        float3 N = make_float3(nmap.ptr(y)[x]);

        const float Ka = 0.3f; //ambient coeff
        const float Kd = 0.5f; //diffuse coeff
        const float Ks = 0.2f; //specular coeff
        const float n = 20.f;  //specular power

        const float Ax = 1.f; //ambient color,  can be RGB
        const float Dx = 1.f; //diffuse color,  can be RGB
        const float Sx = 1.f; //specular color, can be RGB
        const float Lx = 1.f; //light color

        float3 L = normalised(light_pos - P);
        float3 V = normalised(make_float3(0.f, 0.f, 0.f) - P);
        float3 R = normalised(2 * N * (N * L) - L);

        float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, (N * L)) + Lx * Ks * Sx * pow(fmax(0.f, (R * V)), n);
        color = make_float3(Ix, Ix, Ix);
    }

    uchar4 out;
    out.x = static_cast<unsigned char>(__saturatef(color.x) * 255.f);
    out.y = static_cast<unsigned char>(__saturatef(color.y) * 255.f);
    out.z = static_cast<unsigned char>(__saturatef(color.z) * 255.f);
    out.w = 255.0;
    dst.ptr(y)[x] = out;
}

void image_rendering_phong_shading(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, cv::cuda::GpuMat &image)
{
    dim3 thread(8, 4);
    dim3 block(div_up(vmap.cols, thread.x), div_up(vmap.rows, thread.y));

    if (image.empty())
        image.create(vmap.rows, vmap.cols, CV_8UC4);

    image_rendering_phong_shading_kernel<<<block, thread>>>(vmap, nmap, make_float3(5, 5, 5), image);
}

} // namespace cuda
} // namespace slam