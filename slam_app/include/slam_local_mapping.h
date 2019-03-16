#ifndef __SLAM_LOCAL_MAPPING__
#define __SLAM_LOCAL_MAPPING__

#include <memory>
#include "data_source.h"

class SlamLocalMapping
{
  public:
    SlamLocalMapping(DataSource *source);

  private:
    class SlamLocalMappingImpl;
    std::shared_ptr<SlamLocalMappingImpl> impl;
};

#endif