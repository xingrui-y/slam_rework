#ifndef __DEVICE_FRAME__
#define __DEVICE_FRAME__

#include "rgbd_image.h"
#include <opencv2/cudaarithm.hpp>

typedef std::vector<cv::cuda::GpuMat> DeviceImagePyramid;
class DeviceFrame;
typedef std::shared_ptr<DeviceFrame> DeviceFramePtr;
class RgbdImage;
typedef std::shared_ptr<RgbdImage> RgbdImagePtr;
class RgbdFrame;
typedef std::shared_ptr<RgbdFrame> RgbdFramePtr;

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