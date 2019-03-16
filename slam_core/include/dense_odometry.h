#ifndef __DENSE_ODOMETRY__
#define __DENSE_ODOMETRY__

#include "rgbd_image.h"
#include <memory>

class DenseOdometry
{
public:
  DenseOdometry(const IntrinsicMatrixPyramidPtr intrinsics_pyr);
  void track(const cv::Mat &image, const cv::Mat &depth_float, const ulong &id, const double &time_stamp);

  RgbdImagePtr get_current_image() const;
  RgbdImagePtr get_reference_image() const;

private:
  class DenseOdometryImpl;
  std::shared_ptr<DenseOdometryImpl> impl;
};

#endif