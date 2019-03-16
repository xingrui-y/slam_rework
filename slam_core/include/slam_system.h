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

private:
  class SlamSystemImpl;
  std::shared_ptr<SlamSystemImpl> impl;
};

#endif