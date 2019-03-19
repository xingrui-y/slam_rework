#include "dense_odometry.h"
#include "se3_reduction.h"
#include "dense_tracking.h"
#include "point_struct.h"

class DenseOdometry::DenseOdometryImpl
{
public:
  DenseOdometryImpl(const IntrinsicMatrixPyramidPtr intrinsics_pyr);
  void track(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp);
  bool need_new_keyframe() const;
  void create_new_keyframe();

  RgbdFramePtr current_frame_;
  RgbdFramePtr current_keyframe_;
  RgbdFramePtr last_frame_;
  RgbdImagePtr current_;
  RgbdImagePtr reference_;
  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::unique_ptr<DenseTracking> tracker_;

  // KeyPoint detection
  std::vector<RgbdKeyPointStructPtr> key_point_chain_;
  RgbdKeyPointStructPtr current_point_struct_;

  // Public interface
  Sophus::SE3d initial_pose_;
  std::vector<Sophus::SE3d> camera_trajectory_;
};

DenseOdometry::DenseOdometryImpl::DenseOdometryImpl(const IntrinsicMatrixPyramidPtr intrinsics_pyr)
    : intrinsics_pyr_(intrinsics_pyr), tracker_(new DenseTracking()), current_(new RgbdImage()), reference_(new RgbdImage()),
      current_frame_(nullptr), current_keyframe_(nullptr), last_frame_(nullptr), current_point_struct_(new RgbdKeyPointStruct())
{
}

bool DenseOdometry::DenseOdometryImpl::need_new_keyframe() const
{
  IntrinsicMatrix K = intrinsics_pyr_->get_intrinsic_matrix_at(0);
  int visible_kp_count = current_point_struct_->count_visible_keypoints(current_frame_->get_pose().inverse() * current_keyframe_->get_pose(), K);
  return visible_kp_count < 800;
}

void DenseOdometry::DenseOdometryImpl::create_new_keyframe()
{
  current_keyframe_ = current_frame_;
  cv::Mat image = current_frame_->get_image();
  cv::Mat depth = current_frame_->get_depth();
  current_point_struct_ = std::make_shared<RgbdKeyPointStruct>();
  current_point_struct_->detect(image, depth, intrinsics_pyr_->get_intrinsic_matrix_at(0));
}

void DenseOdometry::DenseOdometryImpl::track(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp)
{
  current_frame_ = std::make_shared<RgbdFrame>(image, depth_float, id, time_stamp);
  current_->upload(current_frame_, intrinsics_pyr_);

  if (current_keyframe_ == nullptr)
  {
    current_frame_->set_pose(initial_pose_);
    current_keyframe_ = last_frame_ = current_frame_;
    current_.swap(reference_);

    current_point_struct_->detect(image, depth_float, intrinsics_pyr_->get_intrinsic_matrix_at(0));

    return;
  }

  TrackingContext c;
  c.use_initial_guess_ = true;
  c.initial_estimate_ = Sophus::SE3d();
  c.intrinsics_pyr_ = intrinsics_pyr_;
  c.max_iterations_ = {10, 5, 3, 3, 3};
  TrackingResult result = tracker_->compute_transform(reference_, current_, c);

  if (result.sucess)
  {
    Sophus::SE3d current_pose = last_frame_->get_pose() * result.update;
    current_frame_->set_pose(current_pose);
    camera_trajectory_.push_back(current_pose);
    current_.swap(reference_);
    last_frame_ = current_frame_;

    auto key_points = current_point_struct_->get_key_points();
    IntrinsicMatrix K = intrinsics_pyr_->get_intrinsic_matrix_at(0);

    /** use surf to search key point correspondences */
    // RgbdKeyPointStructPtr current_ref(new RgbdKeyPointStruct());
    // current_ref->detect(image, depth_float, K);
    // current_ref->match(current_point_struct_, current_keyframe_->get_pose().inverse() * current_frame_->get_pose(), K);
    // std::cout << "visible: " << current_point_struct_->count_visible_keypoints(current_pose.inverse() * current_keyframe_->get_pose(), K) << std::endl;

    if (need_new_keyframe())
      create_new_keyframe();

    std::vector<cv::KeyPoint> kp_list;
    Sophus::SE3f T = current_pose.cast<float>().inverse() * current_keyframe_->get_pose().cast<float>();
    for (auto key : key_points)
    {
      Eigen::Vector3f revec = T * key;
      float x = K.fx * revec(0) / revec(2) + K.cx;
      float y = K.fy * revec(1) / revec(2) + K.cy;
      cv::KeyPoint kp;
      kp.pt.x = x;
      kp.pt.y = y;
      kp_list.push_back(kp);
    }

    cv::drawKeypoints(image, kp_list, image);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imshow("iamge", image);
    cv::waitKey(1);
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

std::vector<Sophus::SE3d> DenseOdometry::get_camera_trajectory() const
{
  return impl->camera_trajectory_;
}

void DenseOdometry::set_initial_pose(const Sophus::SE3d pose)
{
  impl->initial_pose_ = pose;
}