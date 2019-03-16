#ifndef __DENSE_MAPPING__
#define __DENSE_MAPPING__

#include <memory>
#include "rgbd_image.h"

class DenseMapping
{
public:
  DenseMapping(const IntrinsicMatrixPyramidPtr &intrinsics_pyr);
  void integrate_frame(RgbdImagePtr current_image);

private:
  class DenseMappingImpl;
  std::shared_ptr<DenseMappingImpl> impl;
};

#endif