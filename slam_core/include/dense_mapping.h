#ifndef __DENSE_MAPPING__
#define __DENSE_MAPPING__

#include <memory>
#include "rgbd_image.h"
#include "point_struct.h"

class DenseMapping
{
public:
  DenseMapping(const IntrinsicMatrixPyramidPtr &intrinsics_pyr);
  void update(RgbdImagePtr image);
  void raycast(RgbdImagePtr image);
  void raycast(KeyPointStructPtr reference);

private:
  class DenseMappingImpl;
  std::shared_ptr<DenseMappingImpl> impl;
};

#endif