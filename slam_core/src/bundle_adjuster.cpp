#include "bundle_adjuster.h"

class BundleAdjuster::BundleAdjusterImpl
{
public:
  BundleAdjusterImpl();
};

BundleAdjuster::BundleAdjusterImpl::BundleAdjusterImpl()
{
}

BundleAdjuster::BundleAdjuster() : impl(new BundleAdjusterImpl())
{
}

void BundleAdjuster::set_up_bundler(std::vector<RgbdFramePtr> keyframe_list)
{
}

void BundleAdjuster::run_bundle_adjustment(int iteration) const
{
}