#ifndef __DENSE_TRACKING__
#define __DENSE_TRACKING__

#include <memory>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "rgbd_image.h"

class DenseTracker
{
public:
  struct Context
  {
    int level_start, level_end;
    std::vector<int> max_iteration_per_level;
    std::vector<double> stopping_criteria;
  };

  DenseTracker();
  Sophus::SE3d match(RgbdImagePyramid &reference, RgbdImagePyramid &target);

private:
  class DenseTrackerImpl;
  std::unique_ptr<DenseTrackerImpl> impl;
};

typedef std::shared_ptr<DenseTracker> DenseTrackerPtr;

#endif