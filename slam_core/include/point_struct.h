#ifndef __POINT_STRUCT__
#define __POINT_STRUCT__

#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "intrinsic_matrix.h"

class RgbdKeyPointStruct;
typedef std::shared_ptr<RgbdKeyPointStruct> RgbdKeyPointStructPtr;

struct Point3d
{
  float x, y, z;
};

class RgbdKeyPointStruct
{
public:
  RgbdKeyPointStruct();
  void detect(const cv::Mat image, const cv::Mat depth, const IntrinsicMatrix K);
  void match(RgbdKeyPointStructPtr reference, const Sophus::SE3d pose_curr_to_ref, IntrinsicMatrix K);
  int count_visible_keypoints(const Sophus::SE3d pose_update, IntrinsicMatrix K) const;
  void clear_struct();
  cv::Mat get_image() const;
  cv::Mat compute_surf() const;
  cv::Mat compute_brisk() const;
  std::vector<Eigen::Vector3f> get_key_points() const;
  std::vector<cv::KeyPoint> get_cv_keypoints() const;

private:
  class RgbdKeyPointStructImpl;
  std::shared_ptr<RgbdKeyPointStructImpl> impl;
};

#endif