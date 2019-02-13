#include "dense_tracking.h"

class DenseTracker::DenseTrackerImpl
{
public:
  DenseTrackerImpl();
  Result match(RgbdImagePyramidPtr reference, RgbdImagePyramidPtr target, const Context &c);
};

DenseTracker::DenseTrackerImpl::DenseTrackerImpl()
{
}

DenseTracker::Result DenseTracker::DenseTrackerImpl::match(RgbdImagePyramidPtr reference, RgbdImagePyramidPtr target, const Context &c)
{
}

DenseTracker::DenseTracker() : impl(new DenseTrackerImpl())
{
}

DenseTracker::Result DenseTracker::match(RgbdImagePyramidPtr reference, RgbdImagePyramidPtr target, const Context &c)
{
  return impl->match(reference, target, c);
}