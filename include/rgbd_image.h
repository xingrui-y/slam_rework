#ifndef __RGBD_IMAGE__
#define __RGBD_IMAGE__

#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class RgbdCamera
{
public:

private:
  class RgbdCameraImpl;
  std::shared_ptr<RgbdCameraImpl> impl;
};

typedef std::shared_ptr<RgbdCamera> RgbdCameraPtr;

class RgbdCameraPyramid
{
public:
  RgbdCameraPyramid();

private:
  std::vector<RgbdCameraPtr> camera;
};

class RgbdImage
{
public:
  RgbdImage();
  void create(const cv::Mat &image, const cv::Mat &depth);
  void compute_derivatives();
  void compute_intensity_derivatives();
  void compute_depth_derivatives();

private:
  class RgbdImageImpl;
  std::shared_ptr<RgbdImageImpl> impl;
};

typedef std::shared_ptr<RgbdImage> RgbdImagePtr;

class RgbdImagePyramid
{
public:
  RgbdImagePyramid(const cv::Mat &intensity, const cv::Mat &depth);

private:
  std::vector<RgbdImagePtr> levels_;
};

#endif