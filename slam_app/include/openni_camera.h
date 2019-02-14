#ifndef __OPENNI_CAMERA__
#define __OPENNI_CAMERA__

#include <memory>
#include <opencv2/opencv.hpp>

class OpenNICamera
{
  public:
    OpenNICamera(int width, int height, int fps);
    ~OpenNICamera();

    void start_video_streaming();
    void stop_video_streaming();
    bool capture(cv::Mat &colour, cv::Mat &depth);

  private:
    class OpenNICameraImpl;
    std::shared_ptr<OpenNICameraImpl> impl;
};

#endif