#ifndef __SLAM_LOCAL_MAPPING__
#define __SLAM_LOCAL_MAPPING__

#include <memory>
#include "data_source.h"
#include "simple_config_file_loader.h"

class SlamLocalMapping
{
public:
  SlamLocalMapping(DataSource *source, SimpleConfigStruct config_struct);

private:
  class SlamLocalMappingImpl;
  std::shared_ptr<SlamLocalMappingImpl> impl;
};

#endif