#ifndef __SLAM_SYSTEM__
#define __SLAM_SYSTEM__

#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include "intrinsic_matrix.h"
#include "rgbd_image.h"

class SlamSystem
{
public:
  SlamSystem(const IntrinsicMatrixPyramidPtr &intrinsic_pyr);
  void update(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp);

  RgbdImagePtr get_current_image() const;
  RgbdImagePtr get_reference_image() const;
  Sophus::SE3d get_current_pose() const;
  std::vector<Sophus::SE3d> get_keyframe_poses() const;
  std::vector<Sophus::SE3d> get_camera_trajectory() const;

  void finish_pending_works();
  void restart_system(); // TODO
  void force_keyframe(); // TODO
  void set_initial_pose(const Sophus::SE3d pose);

private:
  class SlamSystemImpl;
  std::shared_ptr<SlamSystemImpl> impl;
};

#endif