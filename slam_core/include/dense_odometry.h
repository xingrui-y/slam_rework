#ifndef __DENSE_ODOMETRY__
#define __DENSE_ODOMETRY__

#include <memory>
#include "dense_tracking.h"

class DenseOdometry
{
public:

  DenseOdometry();
  void track(const cv::Mat &intensity, const cv::Mat &depth);

private:
  class DenseOdometryImpl;
  std::shared_ptr<DenseOdometryImpl> impl;
};

typedef std::shared_ptr<DenseOdometry> DenseOdometryPtr;

#endif