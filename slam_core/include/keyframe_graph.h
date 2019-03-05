#ifndef __KEY_FRAME_GRAPH__
#define __KEY_FRAME_GRAPH__

#include <memory>
#include "rgbd_image.h"

class KeyFrameGraph
{
public:
  KeyFrameGraph();

private:
  class KeyFrameGraphImpl;
  std::shared_ptr<KeyFrameGraphImpl> impl;
};

#endif