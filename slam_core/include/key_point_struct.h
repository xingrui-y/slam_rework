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
  Eigen::Vector3d pos_; // 3d positions
  cv::KeyPoint kp_;     // cv key point
  RgbdFramePtr reference_frame_;
  std::map<const size_t, int> observations_;
};

struct KeyPoint
{
  cv::KeyPoint kp_;
  float z_;
  Point3dPtr pt3d_;
  Eigen::Vector3f rgb_;
  cv::Mat descriptor_;
};

class KeyPointStruct
{
public:
  KeyPointStruct();

  int detect(const RgbdFramePtr frame);
  int match_by_descriptors(KeyPointStructPtr current_struct);
  int match_by_pose_constraint(KeyPointStructPtr current_struct, IntrinsicMatrix K, bool count_only);
  int match_by_pose_constraint(KeyPointStructPtr current_struct, IntrinsicMatrix K, bool count_only, const char *window_name);
  void track_key_points(const KeyPointStructPtr current, const IntrinsicMatrix K);
  int create_points(int maximum_number, const IntrinsicMatrix K);

  cv::Mat compute();
  cv::Mat get_image() const;
  RgbdFramePtr get_reference_frame() const;
  std::vector<cv::KeyPoint> get_key_points_cv() const;
  std::vector<KeyPoint> &get_key_points();

private:
  class KeyPointStructImpl;
  std::shared_ptr<KeyPointStructImpl> impl;
};

#endif