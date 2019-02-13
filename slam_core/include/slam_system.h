#ifndef __SLAM_SYSTEM__
#define __SLAM_SYSTEM__

#include <memory>
#include <opencv2/opencv.hpp>

class SlamSystem
{
  public:
    SlamSystem();
    void set_new_images(const cv::Mat &intensity, const cv::Mat &depth);

  private:
    class SlamSystemImpl;
    std::shared_ptr<SlamSystemImpl> impl;
};

#endif 