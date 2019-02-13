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