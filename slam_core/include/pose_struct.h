#ifndef __POSE_STRUCT__
#define __POSE_STRUCT__

#include "rgbd_image.h"

class RgbdFrame;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;

class PoseStruct
{
public:
  PoseStruct();
  Sophus::SE3d world_pose_;
  Sophus::SE3d update_;
  RgbdFramePtr reference_;
};

#endif