#ifndef __POINT_STRUCT__
#define __POINT_STRUCT__

#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "rgbd_image.h"
#include "intrinsic_matrix.h"

class KeyPointStruct;
typedef std::shared_ptr<KeyPointStruct> KeyPointStructPtr;

struct Point3d
{
  float x_, y_;         // projection on the image
  Eigen::Vector3f pos_; // 3d positions
  cv::Mat descriptor_;  // surf
  std::map<RgbdFramePtr, int> observations_;
};

class KeyPointStruct
{
public:
  KeyPointStruct();

  cv::Mat compute();
  cv::Mat get_image() const;
  std::vector<cv::KeyPoint> get_key_points_cv() const;
  std::vector<Point3d> get_key_points_3d() const;
  void project_and_show(const RgbdFramePtr frame, const Sophus::SE3d pose, const IntrinsicMatrix K);
  void detect(const RgbdFramePtr frame, const IntrinsicMatrix K);
  int match(KeyPointStructPtr reference_struct, Sophus::SE3d pose_to_ref, IntrinsicMatrix K);

private:
  class KeyPointStructImpl;
  std::shared_ptr<KeyPointStructImpl> impl;
};

#endif