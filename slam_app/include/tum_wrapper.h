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
  void save_full_trajectory(std::vector<Sophus::SE3d> full_trajectory) const;

  // overload
  bool read_next_images(cv::Mat &image, cv::Mat &depth);
  size_t get_current_id() const;
  double get_current_timestamp() const;
  Sophus::SE3d get_current_gt_pose() const;
  double get_current_gt_timestamp() const;
  std::vector<Sophus::SE3d> get_groundtruth() const;
  float get_depth_scale() const;
  Sophus::SE3d get_initial_pose() const;

private:
  size_t current_id_;
  std::string base_dir_;
  std::vector<double> data_ts_;
  std::vector<double> groundtruth_ts_;
  std::vector<std::string> image_name_;
  std::vector<std::string> depth_name_;
  std::vector<Sophus::SE3d> groundtruth_;
};

#endif