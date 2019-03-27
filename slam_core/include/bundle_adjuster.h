#ifndef __BUNDLE_ADJUSTER__
#define __BUNDLE_ADJUSTER__

#include <memory>
#include "rgbd_image.h"
#include "point_struct.h"

class BundleAdjuster
{
public:
  BundleAdjuster();
  void set_up_bundler(std::vector<KeyPointStructPtr> keypoint_structs);
  void run_bundle_adjustment(const IntrinsicMatrix K);
  void run_unit_test();

private:
  class BundleAdjusterImpl;
  std::shared_ptr<BundleAdjusterImpl> impl;
};

#endif