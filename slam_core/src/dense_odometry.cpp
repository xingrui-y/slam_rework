#include "dense_odometry.h"
#include "se3_reduction.h"
#include "dense_tracking.h"
#include "point_struct.h"

class DenseOdometry::DenseOdometryImpl
{
public:
  DenseOdometryImpl(const IntrinsicMatrixPyramidPtr intrinsics_pyr);
  void track_frame(RgbdFramePtr current_frame);
  void track_frame_keyframe_based(RgbdFramePtr current_frame);
  bool keyframe_needed() const;
  void create_keyframe();

  RgbdFramePtr current_keyframe_;
  RgbdFramePtr last_frame_;

  RgbdImagePtr current_image_;
  RgbdImagePtr reference_image_;

  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::unique_ptr<DenseTracking> tracker_;

  bool keyframe_needed_;
  bool tracking_lost_;

  TrackingResult result_;
  TrackingContext context_;
};

DenseOdometry::DenseOdometryImpl::DenseOdometryImpl(const IntrinsicMatrixPyramidPtr intrinsics_pyr)
    : intrinsics_pyr_(intrinsics_pyr), tracker_(new DenseTracking()), current_image_(new RgbdImage()),
      reference_image_(new RgbdImage()), current_keyframe_(nullptr), last_frame_(nullptr),
      keyframe_needed_(false), tracking_lost_(false)
{
}

bool DenseOdometry::DenseOdometryImpl::keyframe_needed() const
{
  return false;
}

void DenseOdometry::DenseOdometryImpl::create_keyframe()
{
  keyframe_needed_ = false;
  current_keyframe_ = last_frame_;
  // current_image_.swap(reference_image_);
}

void DenseOdometry::DenseOdometryImpl::track_frame(RgbdFramePtr current_frame)
{
  current_image_->upload(current_frame, intrinsics_pyr_);

  if (current_keyframe_ != nullptr)
  {
    context_.use_initial_guess_ = true;
    context_.initial_estimate_ = Sophus::SE3d();
    context_.intrinsics_pyr_ = intrinsics_pyr_;
    context_.max_iterations_ = {10, 5, 3, 3, 3};

    result_ = tracker_->compute_transform(reference_image_, current_image_, context_);
  }
  else
  {
    last_frame_ = current_frame;
    keyframe_needed_ = true;
    current_image_.swap(reference_image_);
    return;
  }

  if (result_.sucess)
  {
    auto pose = last_frame_->get_pose() * result_.update;
    current_frame->set_reference_frame(current_keyframe_);
    current_frame->set_pose(pose);

    last_frame_ = current_frame;
    current_image_.swap(reference_image_);
  }
  else
  {
    tracking_lost_ = true;
  }
}

void DenseOdometry::DenseOdometryImpl::track_frame_keyframe_based(RgbdFramePtr current_frame)
{
  current_image_->upload(current_frame, intrinsics_pyr_);

  if (current_keyframe_ != nullptr)
  {
    context_.use_initial_guess_ = true;
    context_.initial_estimate_ = current_keyframe_->get_pose().inverse() * last_frame_->get_pose();
    context_.intrinsics_pyr_ = intrinsics_pyr_;
    context_.max_iterations_ = {10, 0, 0, 0, 0};

    result_ = tracker_->compute_transform(reference_image_, current_image_, context_);
  }
  else
  {
    last_frame_ = current_frame;
    keyframe_needed_ = true;
    return;
  }

  if (result_.sucess)
  {
    auto pose = current_keyframe_->get_pose() * result_.update;
    current_frame->set_reference_frame(current_keyframe_);
    current_frame->set_pose(pose);

    last_frame_ = current_frame;
  }
  else
  {
    tracking_lost_ = true;
  }
}

DenseOdometry::DenseOdometry(const IntrinsicMatrixPyramidPtr intrinsics_pyr)
    : impl(new DenseOdometryImpl(intrinsics_pyr))
{
}

void DenseOdometry::track_frame(RgbdFramePtr current_frame)
{
  impl->track_frame(current_frame);
}

RgbdImagePtr DenseOdometry::get_current_image() const
{
  return impl->current_image_;
}

RgbdImagePtr DenseOdometry::get_reference_image() const
{
  return impl->reference_image_;
}

RgbdFramePtr DenseOdometry::get_current_keyframe() const
{
  return impl->current_keyframe_;
}

bool DenseOdometry::keyframe_needed() const
{
  return impl->keyframe_needed_;
}

bool DenseOdometry::is_tracking_lost() const
{
  return impl->tracking_lost_;
}

void DenseOdometry::create_keyframe()
{
  impl->create_keyframe();
}