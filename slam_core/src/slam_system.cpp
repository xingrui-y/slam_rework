#include <fstream>
#include "slam_system.h"

class SlamSystem::SlamSystemImpl
{
public:
  SlamSystemImpl();

  RgbdCameraPyramidPtr camera;
  DenseOdometryPtr main_tracker;
  DenseTrackerPtr constraint_tracker;
};

SlamSystem::SlamSystemImpl::SlamSystemImpl()
{
}

SlamSystem::SlamSystem() : impl(new SlamSystemImpl())
{
}

void SlamSystem::set_new_images(const cv::Mat &intensity, const cv::Mat &depth)
{
}