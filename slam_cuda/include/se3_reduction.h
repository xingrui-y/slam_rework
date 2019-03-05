#ifndef __SE3_REDUCTION__
#define __SE3_REDUCTION__

#include "intrinsic_matrix.h"
#include "sophus/se3.hpp"
#include <opencv2/cudaarithm.hpp>

using DeviceImage = cv::cuda::GpuMat;

void rgb_reduce(const DeviceImage &curr_intensity,
                const DeviceImage &last_intensity,
                const DeviceImage &last_vmap,
                const DeviceImage &curr_vmap,
                const DeviceImage &intensity_dx,
                const DeviceImage &intensity_dy,
                DeviceImage &sum,
                DeviceImage &out,
                const Sophus::SE3d &pose,
                const IntrinsicMatrixPtr K,
                float *jtj, float *jtr,
                float *residual);

void icp_reduce(const DeviceImage &curr_vmap,
                const DeviceImage &curr_nmap,
                const DeviceImage &last_vmap,
                const DeviceImage &last_nmap,
                DeviceImage &sum,
                DeviceImage &out,
                const Sophus::SE3d &pose,
                const IntrinsicMatrixPtr K,
                float *jtj, float *jtr,
                float *residual);

#endif