#include "slam_system.h"
#include "dense_odometry.h"
#include "dense_mapping.h"
#include "intrinsic_matrix.h"
#include "bundle_adjuster.h"
#include "point_struct.h"
#include "image_ops.h"
#include "stop_watch.h"
#include "opencv_recorder.h"
#include <memory>
#include <mutex>

class SlamSystem::SlamSystemImpl
{
public:
  SlamSystemImpl() = default;
  SlamSystemImpl(const SlamSystemImpl &) = delete;

  SlamSystemImpl(const IntrinsicMatrixPyramidPtr &intrinsic_pyr);
  void update(const cv::Mat &image, const cv::Mat &depth_float, const size_t &id, const double &time_stamp);

  bool compute_distance_score() const;
  bool keyframe_needed() const;
  void create_keyframe();
  void finish_pending_work();
  bool search_constraint();
  void run_bundle_adjustment();

  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::unique_ptr<DenseOdometry> odometry_;
  std::unique_ptr<BundleAdjuster> bundler_;
  std::unique_ptr<DenseMapping> mapping_;

  RgbdFramePtr current_frame_;
  RgbdFramePtr current_keyframe_;
  RgbdFramePtr last_keyframe_;
  RgbdFramePtr first_frame_;
  cv::Mat first_image_;
  Sophus::SE3d initial_pose_;
  std::vector<Sophus::SE3d> frame_poses_;
  std::vector<RgbdFramePtr> keyframes_;

  KeyPointStructPtr reference_point_struct_;
  KeyPointStructPtr current_point_struct_;
  std::vector<KeyPointStructPtr> key_struct_list_;
  std::queue<KeyPointStructPtr> key_struct_buffer_;
  std::mutex key_struct_mutex_guard_;
  bool system_initialised_;
  int num_points_matched_;
  int num_points_detected_;

  RgbdImagePtr vis;
  // slam::util::CVRecorder video(1280, 960, 10);
};

SlamSystem::SlamSystemImpl::SlamSystemImpl(const IntrinsicMatrixPyramidPtr &intrinsics_pyr)
    : intrinsics_pyr_(intrinsics_pyr), odometry_(new DenseOdometry(intrinsics_pyr)), bundler_(new BundleAdjuster()),
      mapping_(new DenseMapping(intrinsics_pyr)), reference_point_struct_(new KeyPointStruct()), num_points_matched_(0),
      current_point_struct_(new KeyPointStruct()), system_initialised_(false), current_keyframe_(nullptr),
      num_points_detected_(0), vis(new RgbdImage())
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

  if (angle_diff < 0.9 || dist_diff > 0.1)
    return true;

  return false;
}

bool SlamSystem::SlamSystemImpl::keyframe_needed() const
{
  if (odometry_->keyframe_needed())
    return true;

  if (compute_distance_score())
    return true;

  if (num_points_matched_ < 40)
    return true;

  return false;
}

void SlamSystem::SlamSystemImpl::create_keyframe()
{
  odometry_->create_keyframe();
  last_keyframe_ = current_keyframe_;
  current_keyframe_ = odometry_->get_current_keyframe();
  keyframes_.push_back(current_keyframe_);

  // if (last_keyframe_ == nullptr)
  // {
  //   reference_point_struct_ = std::make_shared<KeyPointStruct>();
  //   num_points_detected_ = reference_point_struct_->detect(current_keyframe_);

  //   if (num_points_detected_ == 0)
  //     return;

  //   reference_point_struct_->create_points(200, intrinsics_pyr_->get_intrinsic_matrix_at(0));
  // }
  // else
  // {
  //   std::shared_ptr<KeyPointStruct> temp_point_struct = std::make_shared<KeyPointStruct>();
  //   num_points_detected_ = temp_point_struct->detect(current_keyframe_);

  //   if (num_points_detected_ == 0)
  //     return;

  //   int num_matched = reference_point_struct_->match(temp_point_struct, intrinsics_pyr_->get_intrinsic_matrix_at(0), false);
  //   reference_point_struct_ = temp_point_struct;

  //   if (num_matched < 200)
  //     reference_point_struct_->create_points(200, intrinsics_pyr_->get_intrinsic_matrix_at(0));
  // }

  // key_struct_buffer_.push(reference_point_struct_);
}

double compute_se3_to_se3_dist(const Sophus::SE3d pose_1, const Sophus::SE3d pose_2)
{
  Sophus::Vector3d trans_1 = pose_1.translation();
  Sophus::Vector3d trans_2 = pose_2.translation();
  double dist = (trans_1 - trans_2).norm();

  // TODO : dist based on rotation ?

  return dist;
}

bool SlamSystem::SlamSystemImpl::search_constraint()
{
  std::lock_guard<std::mutex> lock(key_struct_mutex_guard_);
  if (key_struct_buffer_.size() > 0)
  {
    auto key_struct = key_struct_buffer_.front();
    auto keyframe = key_struct->get_reference_frame();
    auto pose_se3 = keyframe->get_pose();

    const double dist_threshold = 0.5;

    for (int i = 0; i < key_struct_list_.size(); ++i)
    {
      auto other_key_struct = key_struct_list_[i];
      auto other_keyframe = other_key_struct->get_reference_frame();
      auto other_pose_se3 = other_keyframe->get_pose();

      double dist = compute_se3_to_se3_dist(other_pose_se3, pose_se3);
      if (dist < dist_threshold)
      {
        auto num_matched = other_key_struct->match(key_struct, intrinsics_pyr_->get_intrinsic_matrix_at(0), false);
      }
    }

    key_struct_buffer_.pop();
    key_struct_list_.emplace_back(std::move(key_struct));
    return true;
  }

  return false;
}

void SlamSystem::SlamSystemImpl::run_bundle_adjustment()
{
  std::lock_guard<std::mutex> lock(key_struct_mutex_guard_);
  bundler_->set_up_bundler(key_struct_list_);
  bundler_->run_bundle_adjustment(intrinsics_pyr_->get_intrinsic_matrix_at(0));
}
// slam::util::CVRecorder recorder(1280, 960, 30);
void SlamSystem::SlamSystemImpl::update(const cv::Mat &image, const cv::Mat &depth_float, const size_t &id, const double &time_stamp)
{
  current_frame_ = std::make_shared<RgbdFrame>(image, depth_float, id, time_stamp);
  if (!system_initialised_ && id > 5)
  {
    current_frame_->set_pose(initial_pose_);
    // auto initpose = Sophus::SE3d(Sophus::SO3d(), Sophus::SE3d::Point(100, 100, 100));
    vis->upload(current_frame_, intrinsics_pyr_);
    first_frame_ = current_frame_;
    first_image_ = image.clone();
    system_initialised_ = true;
  }
  else if (!system_initialised_)
  {
    return;
  }

  odometry_->track_frame(current_frame_);

  if (!odometry_->is_tracking_lost())
  {
    auto reference_image = odometry_->get_reference_image();

    mapping_->update(reference_image);
    mapping_->raycast(reference_image);
    reference_image->resize_device_map();
    // cv::cuda::GpuMat rendered_image = reference_image->get_rendered_image();

    cv::cuda::GpuMat rendered_image = reference_image->get_rendered_scene_textured();

    // vis->upload(reference_image->get_reference_frame(), intrinsics_pyr_);
    // mapping_->raycast(vis);
    // vis->resize_device_map();
    // cv::cuda::GpuMat rendered_image = vis->get_rendered_image();
    // cv::cuda::GpuMat vmap = reference_image->get_vmap();
    // cv::cuda::GpuMat nmap = reference_image->get_nmap();
    cv::Mat img(rendered_image);
    cv::resize(img, img, cv::Size(0, 0), 2, 2);
    cv::imshow("rendered image", img);

    // auto vmap = reference_image->get_vmap(0);
    // cv::cuda::GpuMat dst_image;
    // cv::cuda::GpuMat src_image(first_image_);
    // auto pose = first_frame_->get_pose().inverse() * reference_image->get_reference_frame()->get_pose();
    // slam::cuda::warp_image(src_image, vmap, pose, intrinsics_pyr_->get_intrinsic_matrix_at(0), dst_image);
    // cv::cuda::GpuMat intensity;
    // cv::cuda::cvtColor(dst_image, intensity, cv::COLOR_RGB2BGR);

    // cv::Mat dx;
    cv::Mat img2(reference_image->get_image());
    cv::Mat intensity1;
    cv::cvtColor(img2, intensity1, cv::COLOR_RGB2GRAY);

    // cv::Sobel(img2, dx, CV_8UC1, 1, 0);
    cv::resize(intensity1, intensity1, cv::Size(0, 0), 2, 2);
    cv::imshow("img2", intensity1);

    // cv::cvtColor(img, img, cv::COLOR_RGBA2BGR);
    // recorder.add_frame(img);
    // auto reference = reference_image->get_image();
    // cv::Mat img2(reference);
    // cv::resize(img2, img2, cv::Size(0, 0), 2, 2);
    // cv::imshow("img2", img2);

    cv::Mat img3(current_frame_->get_image());
    cv::Mat intensity2;
    cv::cvtColor(img3, intensity2, cv::COLOR_RGB2GRAY);
    cv::resize(intensity2, intensity2, cv::Size(0, 0), 2, 2);
    cv::imshow("img3", intensity2);

    cv::waitKey(1);

    // if (current_point_struct_ != nullptr)
    // {
    //   current_point_struct_ = std::make_shared<KeyPointStruct>();
    //   current_point_struct_->detect(current_frame_);
    //   num_points_matched_ = reference_point_struct_->match(current_point_struct_, intrinsics_pyr_->get_intrinsic_matrix_at(0), true);
    // }
  }

  if (keyframe_needed())
    create_keyframe();

  frame_poses_.push_back(current_frame_->get_pose());
}

void SlamSystem::SlamSystemImpl::finish_pending_work()
{
  while (key_struct_buffer_.size() > 0)
    search_constraint();

  bundler_->set_up_bundler(key_struct_list_);
  bundler_->run_bundle_adjustment(intrinsics_pyr_->get_intrinsic_matrix_at(0));
}

SlamSystem::SlamSystem(const IntrinsicMatrixPyramidPtr &intrinsic_pyr)
    : impl(new SlamSystemImpl(intrinsic_pyr))
{
}

void SlamSystem::update(const cv::Mat &image, const cv::Mat &depth_float, const size_t &id, const double &time_stamp)
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

std::vector<Eigen::Vector3f> SlamSystem::get_current_key_points() const
{
  auto keypoints = impl->current_point_struct_->get_key_points();
  std::vector<Eigen::Vector3f> temp_keypoints;
  std::transform(keypoints.begin(), keypoints.end(), std::back_inserter(temp_keypoints), [](KeyPoint &pt) { if(pt.pt3d_) return pt.pt3d_->pos_.cast<float>(); });
  return temp_keypoints;
}

Sophus::SE3d SlamSystem::get_current_pose() const
{
  return impl->current_frame_->get_pose();
}

void SlamSystem::finish_pending_works()
{
  impl->finish_pending_work();
}

void SlamSystem::set_initial_pose(const Sophus::SE3d pose)
{
  impl->initial_pose_ = pose;
}

bool SlamSystem::search_constraint()
{
  return impl->search_constraint();
}

void SlamSystem::run_bundle_adjustment()
{
  impl->run_bundle_adjustment();
}