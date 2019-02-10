#ifndef __DENSE_ODOMETRY__
#define __DENSE_ODOMETRY__

#include <memory>
#include "dense_tracking.h"

class DenseOdometry
{
public:
  DenseOdometry();

private:
  class DenseOdometryImpl;
  std::unique_ptr<DenseOdometryImpl> impl;
};

typedef std::shared_ptr<DenseOdometry> DenseOdometryPtr;

#endif