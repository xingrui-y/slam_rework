#ifndef __DENSE_ODOMETRY__
#define __DENSE_ODOMETRY__

#include "rgbd_image.h"
#include <memory>

class DenseOdometry
{
public:
  DenseOdometry();
  void track(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid &K, const unsigned long id, const double time_stamp);
  void set_initial_pose(const Sophus::SE3d &pose);
  std::vector<Sophus::SE3d> get_camera_trajectory() const;
  RgbdFramePtr get_current_frame() const;
  RgbdFramePtr get_current_keyframe() const;

private:
  class DenseOdometryImpl;
  std::shared_ptr<DenseOdometryImpl> impl;
};

#endif