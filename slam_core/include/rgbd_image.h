#ifndef __RGBD_FRAME__
#define __RGBD_FRAME__

#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "intrinsic_matrix.h"

class RgbdImage;
class RgbdFrame;
typedef std::shared_ptr<RgbdImage> RgbdImagePtr;
typedef std::shared_ptr<RgbdFrame> RgbdFramePtr;

class RgbdImage
{
public:
  RgbdImage();
  RgbdImage(const RgbdImage &) = delete;
  RgbdImage(const int &max_level);

  void upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr);
  RgbdFramePtr get_reference_frame() const;
  cv::cuda::GpuMat get_depth(const int &level = 0) const;
  cv::cuda::GpuMat get_image(const int &level = 0) const;

private:
  class RgbdImageImpl;
  std::shared_ptr<RgbdImageImpl> impl;
};

class RgbdFrame
{
public:
  RgbdFrame() = delete;
  RgbdFrame(const RgbdFrame &other) = delete;
  RgbdFrame(const cv::Mat &image, const cv::Mat &depth_float, ulong id, double time_stamp);

  cv::Mat get_image() const;
  cv::Mat get_depth() const;
  Sophus::SE3d get_pose() const;
  void set_pose(const Sophus::SE3d &pose);

private:
  class RgbdFrameImpl;
  std::shared_ptr<RgbdFrameImpl> impl;
};

#endif