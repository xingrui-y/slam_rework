#ifndef __RGBD_IMAGE__
#define __RGBD_IMAGE__

#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class RgbdCamera
{
public:
private:
};

typedef std::shared_ptr<RgbdCamera> RgbdCameraPtr;

class RgbdCameraPyramid
{
public:
  RgbdCameraPyramid();

private:
  std::vector<RgbdCameraPtr> levels;
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
  cv::Mat intensity;
  cv::Mat depth;
  cv::Mat intensity_dx;
  cv::Mat intensity_dy;
  cv::Mat point_cloud;
};

typedef std::shared_ptr<RgbdImage> RgbdImagePtr;

class RgbdImagePyramid
{
public:
  RgbdImagePyramid(const cv::Mat &intensity, const cv::Mat &depth);

private:
  std::vector<RgbdImagePtr> levels;
};

typedef std::shared_ptr<RgbdImagePyramid> RgbdImagePyramidPtr;

#endif