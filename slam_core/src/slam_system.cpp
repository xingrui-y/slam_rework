#include "slam_system.h"
#include "dense_odometry.h"
#include "dense_mapping.h"
#include "intrinsic_matrix.h"
#include "bundle_adjuster.h"
#include "point_struct.h"
#include "stop_watch.h"
#include <memory>

class SlamSystem::SlamSystemImpl
{
public:
  SlamSystemImpl() = default;
  SlamSystemImpl(const SlamSystemImpl &) = delete;

  SlamSystemImpl(const IntrinsicMatrixPyramidPtr &intrinsic_pyr);
  void update(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp);

  double compute_distance_score() const;
  bool keyframe_needed() const;
  void create_keyframe();

  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::unique_ptr<DenseOdometry> odometry_;
  std::unique_ptr<BundleAdjuster> bundler_;
  std::unique_ptr<DenseMapping> mapping_;

  RgbdFramePtr current_frame_;
  RgbdFramePtr current_keyframe_;
  Sophus::SE3d initial_pose_;
  std::vector<Sophus::SE3d> frame_poses_;
  std::vector<RgbdFramePtr> keyframes_;

  KeyPointStructPtr reference_point_struct_;
  KeyPointStructPtr current_point_struct_;
};

SlamSystem::SlamSystemImpl::SlamSystemImpl(const IntrinsicMatrixPyramidPtr &intrinsics_pyr)
    : intrinsics_pyr_(intrinsics_pyr), odometry_(new DenseOdometry(intrinsics_pyr)),
      bundler_(new BundleAdjuster()), mapping_(new DenseMapping(intrinsics_pyr)),
      reference_point_struct_(new KeyPointStruct()), current_point_struct_(new KeyPointStruct())
{
}

double SlamSystem::SlamSystemImpl::compute_distance_score() const
{
  auto z_keyframe = current_keyframe_->get_pose().rotationMatrix().rightCols<1>();
  auto z_frame = current_frame_->get_pose().rotationMatrix().rightCols<1>();
  double angle_diff = z_keyframe.dot(z_frame);
  return 0;
}

bool SlamSystem::SlamSystemImpl::keyframe_needed() const
{
  if (odometry_->keyframe_needed())
    return true;

  if (compute_distance_score())
    return true;

  return false;
}

void SlamSystem::SlamSystemImpl::create_keyframe()
{
  odometry_->create_keyframe();
  current_keyframe_ = odometry_->get_current_keyframe();
  reference_point_struct_->detect(current_keyframe_, intrinsics_pyr_->get_intrinsic_matrix_at(0));
}

void SlamSystem::SlamSystemImpl::update(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp)
{
  current_frame_ = std::make_shared<RgbdFrame>(image, depth_float, id, time_stamp);

  if (id == 0)
    current_frame_->set_pose(initial_pose_);

  odometry_->track_frame(current_frame_);

  if (!odometry_->is_tracking_lost())
  {
    RgbdImagePtr image = odometry_->get_reference_image();

    mapping_->update(image);
    mapping_->raycast(image);
    image->resize_device_map();

    cv::cuda::GpuMat vmap = image->get_vmap();
    cv::cuda::GpuMat nmap = image->get_nmap();
    cv::cuda::GpuMat rendered_image = image->get_rendered_image();

    cv::Mat img(rendered_image);
    cv::imshow("rendered image", img);
    cv::waitKey(1);

    if (current_keyframe_)
    {
      reference_point_struct_->project_and_show(current_frame_, current_keyframe_->get_pose(), intrinsics_pyr_->get_intrinsic_matrix_at(0));
      // auto pose_to_ref = current_keyframe_->get_pose().inverse() * current_frame_->get_pose();
      // current_point_struct_->detect(current_frame_, intrinsics_pyr_->get_intrinsic_matrix_at(0));
      // current_point_struct_->match(reference_point_struct_, pose_to_ref, intrinsics_pyr_->get_intrinsic_matrix_at(0));
    }
  }

  if (keyframe_needed())
    create_keyframe();

  frame_poses_.push_back(current_frame_->get_pose());
}

SlamSystem::SlamSystem(const IntrinsicMatrixPyramidPtr &intrinsic_pyr)
    : impl(new SlamSystemImpl(intrinsic_pyr))
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

RgbdImagePtr SlamSystem::get_reference_image() const
{
  return impl->odometry_->get_reference_image();
}

std::vector<Sophus::SE3d> SlamSystem::get_camera_trajectory() const
{
  return impl->frame_poses_;
}

std::vector<Sophus::SE3d> SlamSystem::get_keyframe_poses() const
{
  std::vector<Sophus::SE3d> tmp;
  std::transform(impl->keyframes_.begin(), impl->keyframes_.end(), std::back_inserter(tmp), [](RgbdFramePtr frame) -> Sophus::SE3d { return frame->get_pose(); });
  return tmp;
}

Sophus::SE3d SlamSystem::get_current_pose() const
{
  return impl->current_frame_->get_pose();
}

void SlamSystem::finish_pending_works()
{
}

void SlamSystem::set_initial_pose(const Sophus::SE3d pose)
{
  impl->initial_pose_ = pose;
}