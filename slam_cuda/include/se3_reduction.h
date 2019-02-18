#ifndef __SE3_REDUCTION__
#define __SE3_REDUCTION__

#include "device_array.h"

void se3_reduction(const DeviceArray2D<float4> &vmap_curr, const DeviceArray2D<float4> &vmap_last,
                   const DeviceArray2D<float4> &nmap_curr, const DeviceArray2D<float4> &nmap_last);

#endif