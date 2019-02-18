#ifndef __SLAM_GUI_WRAPPER__
#define __SLAM_GUI_WRAPPER__

#include "slam_system.h"
#include "pangolin_gui.h"

class SlamGuiWrapper
{
public:
  SlamGuiWrapper();
  void grab_image(const cv::Mat &intensity, const cv::Mat &depth);
  bool initialise_system(std::string config_file_path);

private:
  std::unique_ptr<SlamSystem> slam;
  std::unique_ptr<PangolinGUI> ui;
};

#endif