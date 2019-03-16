#include "dense_odometry.h"
#include "se3_reduction.h"
#include "dense_tracking.h"

class DenseOdometry::DenseOdometryImpl
{
public:
  DenseOdometryImpl(const IntrinsicMatrixPyramidPtr intrinsics_pyr);
  void track(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp);

  RgbdFramePtr current_frame_;
  RgbdFramePtr current_keyframe_;
  RgbdFramePtr last_frame_;
  RgbdImagePtr current_;
  RgbdImagePtr reference_;
  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::unique_ptr<DenseTracking> tracker_;
};

DenseOdometry::DenseOdometryImpl::DenseOdometryImpl(const IntrinsicMatrixPyramidPtr intrinsics_pyr)
    : intrinsics_pyr_(intrinsics_pyr), tracker_(new DenseTracking()), current_(new RgbdImage()), reference_(new RgbdImage()),
      current_frame_(nullptr), current_keyframe_(nullptr), last_frame_(nullptr)
{
}

void DenseOdometry::DenseOdometryImpl::track(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp)
{
  current_frame_ = std::make_shared<RgbdFrame>(image, depth_float, id, time_stamp);
  current_->upload(current_frame_, intrinsics_pyr_);

  if (current_keyframe_ == nullptr)
  {
    current_keyframe_ = last_frame_ = current_frame_;
    current_.swap(current_);
    return;
  }

  TrackingContext c;
  c.use_initial_guess_ = false;
  c.intrinsics_pyr_ = intrinsics_pyr_;
  c.max_iterations_ = {10, 5, 3, 2, 2};
  TrackingResult result = tracker_->compute_transform(reference_, current_, c);

  if (result.sucess)
  {
  }
  else
  {
  }
}

DenseOdometry::DenseOdometry(const IntrinsicMatrixPyramidPtr intrinsics_pyr) : impl(new DenseOdometryImpl(intrinsics_pyr))
{
}

void DenseOdometry::track(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp)
{
  impl->track(image, depth_float, id, time_stamp);
}

RgbdImagePtr DenseOdometry::get_current_image() const
{
  return impl->current_;
}

RgbdImagePtr DenseOdometry::get_reference_image() const
{
  return impl->reference_;
}