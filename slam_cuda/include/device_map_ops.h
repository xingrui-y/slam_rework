#ifndef __DEVICE_MAP__
#define __DEVICE_MAP__

#include "map_struct.h"
#include "intrinsic_matrix.h"
#include <sophus/se3.hpp>
#include <opencv2/cudaarithm.hpp>

namespace slam
{
namespace map
{

void update(const cv::cuda::GpuMat depth,
            const cv::cuda::GpuMat image,
            MapStruct &map_struct,
            const Sophus::SE3d &frame_pose,
            const IntrinsicMatrixPtr intrinsic_matrix,
            uint &visible_block_count);

void create_rendering_blocks(MapStruct map_struct,
                             cv::cuda::GpuMat &zrange_x,
                             cv::cuda::GpuMat &zrange_y,
                             const Sophus::SE3d &frame_pose,
                             const IntrinsicMatrixPtr intrinsic_matrix);

} // namespace map
} // namespace slam

#endif