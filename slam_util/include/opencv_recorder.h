#ifndef __OPENCV_RECORDER__
#define __OPENCV_RECORDER__

#include <memory>
#include <opencv2/opencv.hpp>

namespace slam
{
namespace util
{

class CVRecorder
{
public:
  CVRecorder(int width, int height, int frame_rate);
  void create_video(const char *name);
  void add_frame(const cv::Mat frame);
  bool is_recording() const;

private:
  class CVRecorderImpl;
  std::shared_ptr<CVRecorderImpl> impl;
};

} // namespace util
} // namespace slam

#endif