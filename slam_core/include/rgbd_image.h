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
using ImagePyramid = std::vector<cv::Mat>;
using RgbdImagePtr = std::shared_ptr<RgbdImage>;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;
using PoseStructPtr = std::shared_ptr<PoseStruct>;

class RgbdImage
{
public:
  RgbdImage() = default;
  RgbdImage(const RgbdImage &) = delete;
  RgbdImage(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera);
  ~RgbdImage();

  void build_pyramid_derivative(const ImagePyramid &intensity, ImagePyramid &pyramid_dx, ImagePyramid &pyramid_dy);
  void build_pyramid_gaussian(const cv::Mat &origin, ImagePyramid &pyramid, int max_level);
  void build_pyramid_subsample(const cv::Mat &origin, ImagePyramid &pyramid, int max_level);
  void compute_point_cloud(const cv::Mat &depth, cv::Mat &vmap, const IntrinsicMatrixPtr K);
  void compute_point_cloud_pyramid(const ImagePyramid &depth, ImagePyramid &vmap, const IntrinsicMatrixPyramid K);
  void compute_surface_normal_pyramid(const ImagePyramid &vmap, ImagePyramid &nmap);

  cv::Mat get_depth_map(int level = 0) const;
  cv::Mat get_intensity_map(int level = 0) const;
  cv::Mat get_intensity_dx_map(int level = 0) const;
  cv::Mat get_intensity_dy_map(int level = 0) const;
  cv::Mat get_point_cloud(int level = 0) const;
  cv::Mat get_normal_map(int level = 0) const;

private:
  ImagePyramid intensity_;
  ImagePyramid depth_;
  ImagePyramid point_cloud_;
  ImagePyramid intensity_dx_;
  ImagePyramid intensity_dy_;
  ImagePyramid normal_;
};

class RgbdFrame
{
public:
  RgbdFrame() = delete;
  RgbdFrame(const RgbdFrame &other) = delete;
  RgbdFrame(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera, unsigned long id, double time_stamp);

  unsigned long get_id() const;
  Sophus::SE3d get_pose() const;
  PoseStructPtr get_pose_struct() const;
  void set_pose(const Sophus::SE3d &pose);
  void set_reference(const RgbdFramePtr reference);
  int get_pyramid_level() const;
  RgbdImagePtr get_image_pyramid() const;
  IntrinsicMatrixPyramid get_intrinsics() const;

private:
  double time_stamp_;
  unsigned long id_;
  PoseStructPtr pose_;
  RgbdImagePtr data_;
  RgbdFramePtr reference_;
  IntrinsicMatrixPyramid intrinsics_;
};

#endif