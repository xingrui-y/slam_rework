#include "dense_odometry.h"

class DenseOdometry::DenseOdometryImpl
{
public:
  DenseOdometryImpl();
  void track(const cv::Mat &intensity, const cv::Mat &depth);
  DenseTrackerPtr tracker;
};

DenseOdometry::DenseOdometryImpl::DenseOdometryImpl() : tracker(new DenseTracker())
{
}

void DenseOdometry::DenseOdometryImpl::track(const cv::Mat &intensity, const cv::Mat &depth)
{
}

DenseOdometry::DenseOdometry() : impl(new DenseOdometryImpl())
{
}

void DenseOdometry::track(const cv::Mat &intensity, const cv::Mat &depth)
{
}