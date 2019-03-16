#include "slam_system.h"
#include "dense_odometry.h"
#include "intrinsic_matrix.h"
#include "stop_watch.h"
#include <memory>

class SlamSystem::SlamSystemImpl
{
public:
  SlamSystemImpl() = default;
  SlamSystemImpl(const SlamSystemImpl &) = delete;
  SlamSystemImpl(const IntrinsicMatrixPyramidPtr &intrinsic_pyr);
  void update(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp);

  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::unique_ptr<DenseOdometry> odometry_;
};

SlamSystem::SlamSystemImpl::SlamSystemImpl(const IntrinsicMatrixPyramidPtr &intrinsics_pyr)
    : intrinsics_pyr_(intrinsics_pyr), odometry_(new DenseOdometry(intrinsics_pyr))
{
}

void SlamSystem::SlamSystemImpl::update(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp)
{
  odometry_->track(image, depth_float, id, time_stamp);
}

SlamSystem::SlamSystem(const IntrinsicMatrixPyramidPtr &intrinsic_pyr) : impl(new SlamSystemImpl(intrinsic_pyr))
{
}

void SlamSystem::update(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp)
{
  impl->update(image, depth_float, id, time_stamp);
}

RgbdImagePtr SlamSystem::get_current_image() const
{
  return impl->odometry_->get_current_image();
}