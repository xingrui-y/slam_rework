#ifndef __RGBD_FRAME__
#define __RGBD_FRAME__

#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include "intrinsic_matrix.h"
#include "pose_struct.h"

class RgbdImage;
class RgbdFrame;
class PoseStruct;
using RgbdImagePtr = std::shared_ptr<RgbdImage>;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;
using PoseStructPtr = std::shared_ptr<PoseStruct>;

class RgbdImage
{
public:
  RgbdImage() = default;
  RgbdImage(const RgbdImage &) = delete;
  RgbdImage(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera);

  cv::Mat get_depth_map(int level = 0) const;
  cv::Mat get_intensity_map(int level = 0) const;
  cv::Mat get_intensity_dx_map(int level = 0) const;
  cv::Mat get_intensity_dy_map(int level = 0) const;
  cv::Mat get_point_cloud(int level = 0) const;
  cv::Mat get_normal_map(int level = 0) const;

private:
  class RgbdImageImpl;
  std::shared_ptr<RgbdImageImpl> impl;
};

class RgbdFrame
{
public:
  RgbdFrame() = delete;
  RgbdFrame(const RgbdFrame &other) = delete;
  RgbdFrame(const cv::Mat &image, const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera, unsigned long id, double time_stamp);

  void set_pose(const Sophus::SE3d &pose);
  void set_reference(const RgbdFramePtr reference);

  unsigned long get_id() const;
  Sophus::SE3d get_pose() const;
  PoseStructPtr get_pose_struct() const;
  int get_pyramid_level() const;
  RgbdImagePtr get_image_pyramid() const;
  IntrinsicMatrixPyramid get_intrinsics() const;
  cv::Mat get_image() const;

private:
  class RgbdFrameImpl;
  std::shared_ptr<RgbdFrameImpl> impl;
};

#endif