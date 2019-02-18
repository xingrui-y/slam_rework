#ifndef __RGBD_IMAGE__
#define __RGBD_IMAGE__

#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class RgbdCamera
{
public:
  RgbdCamera(float fx, float fy, float cx, float cy);
  cv::Vec3f back_project_point(int x, int y, float z) const;

  int cols, rows;
  float cx, cy, fx, fy;
  float inv_fx, inv_fy;
};

typedef std::shared_ptr<RgbdCamera> RgbdCameraPtr;

class RgbdCameraPyramid
{
public:
  RgbdCameraPyramid(int total_level);
  int get_total_level() const;
  RgbdCameraPtr operator[](int level) const;

private:
  std::vector<RgbdCameraPtr> levels;
};

typedef std::shared_ptr<RgbdCameraPyramid> RgbdCameraPyramidPtr;

class RgbdImage
{
public:
  RgbdImage();
  void create(const cv::Mat &image, const cv::Mat &depth);
  void compute_intensity_derivatives();
  void back_project_points(float fx, float fy, float cx, float cy);

private:
  cv::Mat intensity;
  cv::Mat depth;
  cv::Mat intensity_dx;
  cv::Mat intensity_dy;
  cv::Mat point_cloud;
  cv::Mat estimated_normal;
  RgbdCameraPtr camera;
};

typedef std::shared_ptr<RgbdImage> RgbdImagePtr;

class RgbdImagePyramid
{
public:
  RgbdImagePyramid(cv::Mat &intensity, cv::Mat &depth, RgbdCameraPyramidPtr cameras);
  RgbdImagePtr operator[](int level);

private:
  std::vector<RgbdImagePtr> levels;
};

typedef std::shared_ptr<RgbdImagePyramid> RgbdImagePyramidPtr;

#endif