#ifndef __OPENNI_CAMERA__
#define __OPENNI_CAMERA__

#include <memory>
#include <opencv2/opencv.hpp>
#include "data_source.h"

class OpenNICamera : public DataSource
{
public:
  OpenNICamera(int width, int height, int fps);
  ~OpenNICamera();

  bool read_next_images(cv::Mat &image, cv::Mat &depth);
  Sophus::SE3d get_starting_pose() const;
  double get_current_timestamp() const;
  unsigned int get_current_id() const;
  std::vector<Sophus::SE3d> get_groundtruth() const;
  float get_depth_scale() const;

private:
  class OpenNICameraImpl;
  std::shared_ptr<OpenNICameraImpl> impl;
};

#endif