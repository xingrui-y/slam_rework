#ifndef __DENSE_ODOMETRY__
#define __DENSE_ODOMETRY__

#include "rgbd_image.h"
#include <memory>

class DenseOdometry
{
public:
  DenseOdometry(const IntrinsicMatrixPyramidPtr intrinsics_pyr);
  void track_frame(RgbdFramePtr current_frame);
  bool keyframe_needed() const;
  bool is_tracking_lost() const;
  void create_keyframe();

  std::vector<Sophus::SE3d> get_keyframe_poses() const;
  std::vector<Sophus::SE3d> get_camera_trajectory() const;

  RgbdImagePtr get_current_image() const;
  RgbdImagePtr get_reference_image() const;
  RgbdFramePtr get_current_frame() const;
  RgbdFramePtr get_current_keyframe() const;

private:
  class DenseOdometryImpl;
  std::shared_ptr<DenseOdometryImpl> impl;
};

#endif