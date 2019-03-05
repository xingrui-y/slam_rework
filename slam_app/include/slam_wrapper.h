#ifndef __SLAM_WRAPPER__
#define __SLAM_WRAPPER__

#include <memory>
#include "data_source.h"

class SlamWrapper
{
public:
  SlamWrapper(DataSource *source);

private:
  class SlamWrapperImpl;
  std::shared_ptr<SlamWrapperImpl> impl;
};

#endif