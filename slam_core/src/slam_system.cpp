#include "slam_system.h"

class SlamSystem::SlamSystemImpl
{
public:
  SlamSystemImpl();
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