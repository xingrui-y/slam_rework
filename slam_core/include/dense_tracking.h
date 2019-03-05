#ifndef __DENSE_TRACKING__
#define __DENSE_TRACKING__

#include <memory>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "rgbd_image.h"

struct TrackingResult
{
  bool sucess;
  Sophus::SE3d update;
};

struct TrackingContext
{
  bool use_initial_guess_;
  IntrinsicMatrixPyramid intrinsics_;
  std::vector<int> tracking_level_;
  Sophus::SE3d initial_estimate_;
};

class DenseTracking
{
public:
  DenseTracking();
  TrackingResult track(const RgbdFramePtr reference, const RgbdFramePtr current, const TrackingContext &c);

private:
  class DenseTrackingImpl;
  std::shared_ptr<DenseTrackingImpl> impl;
};

#endif