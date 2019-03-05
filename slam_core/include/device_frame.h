#ifndef __DEVICE_FRAME__
#define __DEVICE_FRAME__

#include "rgbd_image.h"
#include "opencv2/cudaarithm.hpp"
#include <mutex>
#include <queue>

using DeviceImagePyramid = std::vector<cv::cuda::GpuMat>;
class DeviceFrame;
using DeviceFramePtr = std::shared_ptr<DeviceFrame>;
class RgbdImage;
using RgbdImagePtr = std::shared_ptr<RgbdImage>;
class RgbdFrame;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;

class DeviceFrame
{
public:
  DeviceFrame() = default;
  DeviceFrame(const RgbdFramePtr data);
  DeviceFrame(const DeviceFrame &) = delete;
  DeviceFrame &operator=(const DeviceFrame &) = delete;
  void upload(const RgbdFramePtr data);

  RgbdFramePtr owner_;
  DeviceImagePyramid intensity_;
  DeviceImagePyramid depth_;
  DeviceImagePyramid intensity_dx_;
  DeviceImagePyramid intensity_dy_;
  DeviceImagePyramid point_cloud_;
  DeviceImagePyramid normal_;
};

#endif