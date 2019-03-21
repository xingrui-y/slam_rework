#include "slam_system.h"
#include "dense_odometry.h"
#include "dense_mapping.h"
#include "intrinsic_matrix.h"
#include "bundle_adjuster.h"
#include "point_struct.h"
#include "stop_watch.h"
#include "opencv_recorder.h"
#include <memory>

class SlamSystem::SlamSystemImpl
{
public:
  SlamSystemImpl() = default;
  SlamSystemImpl(const SlamSystemImpl &) = delete;

  SlamSystemImpl(const IntrinsicMatrixPyramidPtr &intrinsic_pyr);
  void update(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp);

  bool compute_distance_score() const;
  bool keyframe_needed() const;
  void create_keyframe();

  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::unique_ptr<DenseOdometry> odometry_;
  std::unique_ptr<BundleAdjuster> bundler_;
  std::unique_ptr<DenseMapping> mapping_;

  RgbdFramePtr current_frame_;
  RgbdFramePtr current_keyframe_;
  RgbdFramePtr last_keyframe_;
  Sophus::SE3d initial_pose_;
  std::vector<Sophus::SE3d> frame_poses_;
  std::vector<RgbdFramePtr> keyframes_;

  KeyPointStructPtr reference_point_struct_;
  KeyPointStructPtr current_point_struct_;
  bool system_initialised_;
};

SlamSystem::SlamSystemImpl::SlamSystemImpl(const IntrinsicMatrixPyramidPtr &intrinsics_pyr)
    : intrinsics_pyr_(intrinsics_pyr), odometry_(new DenseOdometry(intrinsics_pyr)), bundler_(new BundleAdjuster()),
      mapping_(new DenseMapping(intrinsics_pyr)), reference_point_struct_(new KeyPointStruct()),
      current_point_struct_(new KeyPointStruct()), system_initialised_(false), current_keyframe_(nullptr)
{
}

bool SlamSystem::SlamSystemImpl::compute_distance_score() const
{
  Eigen::Vector3d z_keyframe = current_keyframe_->get_pose().rotationMatrix().topRightCorner(3, 1);
  Eigen::Vector3d z_frame = current_frame_->get_pose().rotationMatrix().topRightCorner(3, 1);
  Eigen::Vector3d dist_keyframe = current_keyframe_->get_pose().translation();
  Eigen::Vector3d dist_frame = current_frame_->get_pose().translation();
  double angle_diff = z_keyframe.dot(z_frame);
  double dist_diff = (dist_frame - dist_keyframe).norm();

  if (angle_diff < 0.8 || dist_diff > 0.2)
    return true;

  return false;
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
  last_keyframe_ = current_keyframe_;
  current_keyframe_ = odometry_->get_current_keyframe();
  keyframes_.push_back(current_keyframe_);
  KeyPointStructPtr tmp_point_struct = reference_point_struct_;
  reference_point_struct_ = std::make_shared<KeyPointStruct>();
  reference_point_struct_->detect(current_keyframe_, intrinsics_pyr_->get_intrinsic_matrix_at(0));

  if (last_keyframe_ != nullptr)
  {
    // const Sophus::SE3d pose_to_ref = last_keyframe_->get_pose().inverse() * current_keyframe_->get_pose();
    // reference_point_struct_->match(tmp_point_struct, pose_to_ref, intrinsics_pyr_->get_intrinsic_matrix_at(0));
  }
}
slam::util::CVRecorder video(1280, 960, 10);
void SlamSystem::SlamSystemImpl::update(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp)
{
  current_frame_ = std::make_shared<RgbdFrame>(image, depth_float, id, time_stamp);

  if (!system_initialised_)
  {
    current_frame_->set_pose(initial_pose_);
    system_initialised_ = true;
  }

  odometry_->track_frame(current_frame_);

  if (!odometry_->is_tracking_lost())
  {
    RgbdImagePtr reference_image = odometry_->get_reference_image();

    mapping_->update(reference_image);
    mapping_->raycast(reference_image);
    reference_image->resize_device_map();

    cv::cuda::GpuMat vmap = reference_image->get_vmap();
    cv::cuda::GpuMat nmap = reference_image->get_nmap();
    cv::cuda::GpuMat rendered_image = reference_image->get_rendered_image();
    cv::Mat img(rendered_image);
    cv::imshow("rendered image", img);
    cv::waitKey(1);

    if (current_keyframe_ != nullptr)
    {
      cv::Mat out_image, out_image2;
      IntrinsicMatrix K_0 = intrinsics_pyr_->get_intrinsic_matrix_at(0);
      current_point_struct_ = std::make_shared<KeyPointStruct>();
      current_point_struct_->detect(current_frame_, K_0);
      reference_point_struct_->project_and_show(current_point_struct_, K_0, out_image2);
      auto pose_to_ref = current_keyframe_->get_pose().inverse() * current_frame_->get_pose();
      current_point_struct_->detect(current_frame_, K_0);
      current_point_struct_->match(reference_point_struct_, pose_to_ref, K_0, out_image);

      // cv::Mat temp(960, 1280, CV_8UC3);
      // cv::Mat rendered_image0(480, 640, CV_8UC3);
      // cv::cvtColor(out_image, out_image, cv::COLOR_RGB2BGR);
      // cv::cvtColor(out_image2, out_image2, cv::COLOR_RGB2BGR);
      // cv::cvtColor(img, rendered_image0, cv::COLOR_RGBA2BGR);

      // out_image.copyTo(temp(cv::Rect2i(cv::Point2i(0, 0), cv::Point2i(1280, 480))));
      // rendered_image0.copyTo(temp(cv::Rect2i(cv::Point2i(0, 480), cv::Point2i(640, 960))));
      // out_image2.copyTo(temp(cv::Rect2i(cv::Point2i(640, 480), cv::Point2i(1280, 960))));
      // cv::imshow("test", temp);
      // cv::waitKey(1);

      // if (!video.is_recording())
      //   video.create_video("1.avi");
      // video.add_frame(temp);
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