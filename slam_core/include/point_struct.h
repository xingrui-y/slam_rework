#ifndef __POINT_STRUCT__
#define __POINT_STRUCT__

#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "rgbd_image.h"
#include "intrinsic_matrix.h"

struct Point3d;
class KeyPointStruct;
typedef std::shared_ptr<Point3d> Point3dPtr;
typedef std::shared_ptr<KeyPointStruct> KeyPointStructPtr;

struct Point3d
{
  Eigen::Vector3f pos_; // 3d positions
  cv::Mat descriptor_;  // surf
  cv::KeyPoint kp_;     // cv key point
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
  RgbdFramePtr get_reference_frame() const;
  void detect(const RgbdFramePtr frame, const IntrinsicMatrix K);
  void project_and_show(const KeyPointStructPtr current, const IntrinsicMatrix K, cv::Mat &out_image);
  int match(KeyPointStructPtr reference_struct, Sophus::SE3d pose_to_ref, IntrinsicMatrix K, cv::Mat &out_image);

private:
  class KeyPointStructImpl;
  std::shared_ptr<KeyPointStructImpl> impl;
};

#endif