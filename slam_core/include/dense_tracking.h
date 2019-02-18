#ifndef __DENSE_TRACKING__
#define __DENSE_TRACKING__

#include <memory>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "rgbd_image.h"

class DenseTracker
{
public:
  struct Result
  {
    Sophus::SE3d result;
    float final_log_likelyhood;
    std::vector<int> iterations_per_level;
    float residual_sum;
  };

  struct Context
  {
    bool icp_only, rgb_only;
    unsigned int levels;
    bool use_initial_estimate;
    Sophus::SE3d initial_estimate;
    std::vector<int> max_iterations;
  };

  DenseTracker();
  Result match(RgbdImagePyramidPtr reference, RgbdImagePyramidPtr target, const Context &c);

private:
  class DenseTrackerImpl;
  std::shared_ptr<DenseTrackerImpl> impl;
};

typedef std::shared_ptr<DenseTracker> DenseTrackerPtr;

#endif