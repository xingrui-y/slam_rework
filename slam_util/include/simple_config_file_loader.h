#ifndef __SIMPLE_CONFIG_FILE_LOADER__
#define __SIMPLE_CONFIG_FILE_LOADER__

#include <fstream>

struct SimpleConfigStruct
{
  int width;
  int height;
  float fx;
  float fy;
  float cx;
  float cy;
  int pyramid_level;
  int num_bucket;
  int num_hash_entry;
  int num_voxel_block;
  float zmin_update;
  float zmax_update;
  float zmin_raycast;
  float zmax_raycast;
};

class SimpleConfigFileLoader
{
public:
  SimpleConfigFileLoader() = default;
  SimpleConfigStruct load_config_file(std::string file_name);
  void write_config_file(std::string file_name, SimpleConfigStruct config);

  std::ifstream file_in_;
  std::ofstream file_out_;
};

#endif