#ifndef __DATA_SOURCE__
#define __DATA_SOURCE__

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>

class DataSource
{
public:
  virtual bool read_next_images(cv::Mat &image, cv::Mat &depth) = 0;
  virtual Sophus::SE3d get_starting_pose() const = 0;
  virtual double get_current_timestamp() const = 0;
  virtual unsigned int get_current_id() const = 0;
  virtual std::vector<Sophus::SE3d> get_groundtruth() const = 0;
  virtual float get_depth_scale() const = 0;
};

#endif