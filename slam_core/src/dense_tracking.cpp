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
  Sophus::SE3d initial_estimate;
  if (c.use_initial_estimate)
    Sophus::SE3d initial_estimate = c.initial_estimate;

  for (int i = c.levels; i >= 0; --i)
  {
    for (int j = 0; j < c.max_iterations[i]; ++j)
    {
      
    }
  }
}

DenseTracker::DenseTracker() : impl(new DenseTrackerImpl())
{
}

DenseTracker::Result DenseTracker::match(RgbdImagePyramidPtr reference, RgbdImagePyramidPtr target, const Context &c)
{
  return impl->match(reference, target, c);
}