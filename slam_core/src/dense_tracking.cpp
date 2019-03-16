#include "dense_tracking.h"
#include "rgbd_image.h"
#include "se3_reduction.h"
#include "revertable.h"
#include "stop_watch.h"

class DenseTracking::DenseTrackingImpl
{
public:
  DenseTrackingImpl();
  TrackingResult compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c);
};

DenseTracking::DenseTrackingImpl::DenseTrackingImpl()
{
}

TrackingResult DenseTracking::DenseTrackingImpl::compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
{
  Revertable<Sophus::SE3d> estimate;

  if (c.use_initial_guess_)
    estimate = Revertable<Sophus::SE3d>(c.initial_estimate_);

  for (int level = c.max_iterations_.size() - 1; level >= 0; --level)
  {
    for (int iter = 0; iter < c.max_iterations_[level]; ++iter)
    {
    }
  }
}

DenseTracking::DenseTracking() : impl(new DenseTrackingImpl())
{
}

TrackingResult DenseTracking::compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
{
  return impl->compute_transform(reference, current, c);
}
