#include "dense_odometry.h"

class DenseOdometry::DenseOdometryImpl
{
public:
  DenseOdometryImpl();
  DenseTrackerPtr tracker;

  void track(const cv::Mat &intensity, const cv::Mat &depth);
};

DenseOdometry::DenseOdometryImpl::DenseOdometryImpl() : tracker(new DenseTracker())
{
}

void DenseOdometry::DenseOdometryImpl::track(const cv::Mat &intensity, const cv::Mat &depth)
{
  RgbdImagePtr frame;
  frame->create(intensity, depth);
}

DenseOdometry::DenseOdometry() : impl(new DenseOdometryImpl())
{
}