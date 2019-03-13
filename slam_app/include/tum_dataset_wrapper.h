#ifndef __TUM_DATASET_WRAPPER__
#define __TUM_DATASET_WRAPPER__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include "data_source.h"

class TUMDatasetWrapper : public DataSource
{
public:
  TUMDatasetWrapper(std::string dir);
  bool load_association_file(std::string file_name);
  void load_ground_truth(std::string file_name);
  bool read_next_images(cv::Mat &image, cv::Mat &depth);

  // interface implementations
  void save_full_trajectory(std::vector<Sophus::SE3d> full_trajectory, std::string file_name) const;
  Sophus::SE3d get_starting_pose() const;
  double get_current_timestamp() const;
  unsigned int get_current_id() const;
  std::vector<Sophus::SE3d> get_groundtruth() const;
  float get_depth_scale() const;

private:
  int find_closest_index(std::vector<double> list, double time) const;
  unsigned int id;
  std::string base_dir;
  std::vector<double> time_stamp;
  std::vector<std::string> image_list;
  std::vector<std::string> depth_list;
  std::vector<double> time_stamp_gt;
  std::vector<Sophus::SE3d> ground_truth;
};

#endif