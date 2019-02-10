#include "dense_tracking.h"

class DenseTracker::DenseTrackerImpl
{
public:
  DenseTrackerImpl();
};

DenseTracker::DenseTrackerImpl::DenseTrackerImpl()
{
}

DenseTracker::DenseTracker() : impl(new DenseTrackerImpl())
{
}

