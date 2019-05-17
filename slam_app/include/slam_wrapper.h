#ifndef __SLAM_LOCAL_MAPPING__
#define __SLAM_LOCAL_MAPPING__

#include <memory>
#include "data_source.h"
#include "simple_config_file_loader.h"

class SlamWrapper
{
public:
  SlamWrapper();
  void set_data_source(DataSource *source);
  void set_configuration(SimpleConfigStruct config);
  void set_ground_truth_poses(std::vector<Sophus::SE3d> &gt);
  void set_initial_pose(const Sophus::SE3d pose);
  void set_image_time_stamp(std::vector<double> &ts);
  void set_depth_scale(const float &scale);
  void run_slam_system();

private:
  class SlamWrapperImpl;
  std::shared_ptr<SlamWrapperImpl> impl;
};

#endif