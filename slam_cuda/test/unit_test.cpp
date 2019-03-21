#include <cassert>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "image_ops.h"
#include "device_map_ops.h"
#include "cuda_utils.h"

int main(int argc, char **argv)
{
    // IntrinsicMatrix base_intrinsic_matrix(640, 480, 528.f, 528.f, 320.f, 240.f);
    // IntrinsicMatrixPyramidPtr intrinsic_pyr(new IntrinsicMatrixPyramid(base_intrinsic_matrix, 5));

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<float> ureal(0, 10);

    // cv::Mat depth(base_intrinsic_matrix.height, base_intrinsic_matrix.width, CV_32FC1);
    // depth.forEach<float>([&](float &z, const int *position) -> void { z = ureal(gen); });
    // cv::cuda::GpuMat depth_gpu(depth);

    // std::vector<cv::cuda::GpuMat> depth_pyr(5);
    // std::vector<cv::cuda::GpuMat> point_cloud_pyr(5);
    // std::vector<cv::cuda::GpuMat> normal_pyr(5);

    // depth_pyr.resize(5);
    // point_cloud_pyr.resize(5);
    // normal_pyr.resize(5);

    // slam::cuda::build_depth_pyramid(depth_gpu, depth_pyr, 5);
    // slam::cuda::build_point_cloud_pyramid(depth_pyr, point_cloud_pyr, intrinsic_pyr);

    // for (int i = 0; i < 5; ++i)
    // {
    //     cv::Mat depth, point_cloud;
    //     depth_pyr[i].download(depth);
    //     point_cloud_pyr[i].download(point_cloud);
    //     cv::imshow("depth", depth);
    //     cv::imshow("point_cloud", point_cloud);
    //     cv::waitKey(0);
    // }
}