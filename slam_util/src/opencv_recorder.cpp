#include "opencv_recorder.h"

namespace slam
{
namespace util
{

class CVRecorder::CVRecorderImpl
{
  public:
    CVRecorderImpl() = default;
    CVRecorderImpl(int width, int height, int frame_rate);
    void create_video(const char *name);

    int cols, rows, fps;
    std::shared_ptr<cv::VideoWriter> video;
};

CVRecorder::CVRecorderImpl::CVRecorderImpl(int width, int height, int frame_rate)
    : cols(width), rows(height), fps(frame_rate)
{
}

void CVRecorder::CVRecorderImpl::create_video(const char *name)
{
    video = std::make_shared<cv::VideoWriter>(name, CV_FOURCC('M', 'J', 'P', 'G'), fps, cv::Size(cols, rows));
}

CVRecorder::CVRecorder(int width, int height, int frame_rate) : impl(new CVRecorderImpl(width, height, frame_rate))
{
}

void CVRecorder::create_video(const char *name)
{
    impl->create_video(name);
}

void CVRecorder::add_frame(const cv::Mat frame)
{
    if (impl->video == nullptr)
        create_video("1.avi");
    impl->video->write(frame);
}

bool CVRecorder::is_recording() const
{
    return impl->video != nullptr;
}

} // namespace util
} // namespace slam