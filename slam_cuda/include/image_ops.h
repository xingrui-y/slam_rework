#ifndef __IMAGE_PROCESSING__
#define __IMAGE_PROCESSING__

#include "device_array.h"

void compute_vmap(const DeviceArray2D<float> &depth, DeviceArray2D<float4> &vmap, float cx, float cy, float fx, float fy);

#endif