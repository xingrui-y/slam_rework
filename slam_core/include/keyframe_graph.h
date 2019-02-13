#ifndef __KEY_FRAME_GRAPH__
#define __KEY_FRAME_GRAPH__

#include <memory>

class KeyFrameGraph
{
  public:
    KeyFrameGraph();

  private:
    class KeyFrameGraphImpl;
    std::shared_ptr<KeyFrameGraph> impl;
};

#endif