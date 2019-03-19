#ifndef __BUNDLE_ADJUSTER__
#define __BUNDLE_ADJUSTER__

#include <memory>
#include "rgbd_image.h"

class BundleAdjuster
{
  public:
    BundleAdjuster();
    void set_up_bundler(std::vector<RgbdFramePtr> keyframe_list);
    void run_bundle_adjust(int iteration) const;

  private:
    class BundleAdjusterImpl;
    std::shared_ptr<BundleAdjusterImpl> impl;
};

#endif