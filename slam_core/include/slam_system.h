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
  SlamSystem(const IntrinsicMatrix &K);
  void update(const cv::Mat &intensity, const cv::Mat &depth, const unsigned long id, const double time_stamp);

  RgbdFramePtr get_current_frame() const;
  Sophus::SE3d get_current_pose() const;
  std::vector<Sophus::SE3d> get_camera_trajectory() const;
  void set_initial_pose(const Sophus::SE3d &pose);

private:
  class SlamSystemImpl;
  std::shared_ptr<SlamSystemImpl> impl;
};

#endif