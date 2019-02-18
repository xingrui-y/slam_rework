#ifndef __PANGOLIN_GUI__
#define __PANGOLIN_GUI__

#include <memory>
#include <sophus/se3.hpp>

class PangolinGUI
{
  public:
    PangolinGUI();
    PangolinGUI(int width, int height);

    bool should_quit() const;
    void draw_frame();
  
  private:
    class PangolinGUIImpl;
    std::shared_ptr<PangolinGUIImpl> impl;
};

#endif