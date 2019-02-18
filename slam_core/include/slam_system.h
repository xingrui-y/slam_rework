#ifndef __SLAM_SYSTEM__
#define __SLAM_SYSTEM__

#include "dense_odometry.h"
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class SlamSystem
{
public:
  SlamSystem();
  void set_new_images(const cv::Mat &intensity, const cv::Mat &depth);
  void set_intrinsics(Eigen::Matrix<float, 3, 3> &parametr_matrix);

private:
  class SlamSystemImpl;
  std::shared_ptr<SlamSystemImpl> impl;
};

#endif