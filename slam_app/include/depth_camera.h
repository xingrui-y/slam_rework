#ifndef __DEPTH_CAMERA__
#define __DEPTH_CAMERA__

#include <memory>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include "data_source.h"

class DepthCamera : public DataSource
{
public:
  DepthCamera(int width, int height, int fps);
  ~DepthCamera();
  void start_video_streaming();
  void stop_video_streaming();

  bool read_next_images(cv::Mat &image, cv::Mat &depth);
  size_t get_current_id() const;
  double get_current_timestamp() const;
  Sophus::SE3d get_current_gt_pose() const;
  double get_current_gt_timestamp() const;
  std::vector<Sophus::SE3d> get_groundtruth() const;
  float get_depth_scale() const;
  Sophus::SE3d get_initial_pose() const;

private:
  class DepthCameraImpl;
  std::shared_ptr<DepthCameraImpl> impl;
};

#endif