#include "depth_camera.h"
#include <openni2/OpenNI.h>

using namespace openni;

class DepthCamera::DepthCameraImpl
{
  public:
    DepthCameraImpl(int width, int height, int fps);

    void start_video_streaming();
    void stop_video_streaming();
    bool capture(cv::Mat &image, cv::Mat &depth);

    std::shared_ptr<Device> device_;
    std::shared_ptr<VideoStream> colour_stream_;
    std::shared_ptr<VideoStream> depth_stream_;
    std::shared_ptr<VideoFrameRef> colour_frame_ref_;
    std::shared_ptr<VideoFrameRef> depth_frame_ref_;

    int image_width_;   // <= 640
    int image_height_;  // <= 480
    int frame_rate_;    // <= 30fps
    size_t current_id_; // starts from 0
};

DepthCamera::DepthCameraImpl::DepthCameraImpl(int width, int height, int fps) : image_width_(width), image_height_(height), frame_rate_(fps), current_id_(0)
{
    auto OK = STATUS_OK;
    device_ = std::make_shared<Device>();

    CHECK_EQ(OpenNI::initialize(), OK) << "initialisation failed: " << OpenNI::getExtendedError() << std::endl;
    CHECK_EQ(device_->open(ANY_DEVICE), OK) << "device open failed: " << OpenNI::getExtendedError() << std::endl;

    colour_stream_ = std::make_shared<VideoStream>();
    depth_stream_ = std::make_shared<VideoStream>();

    CHECK_EQ(depth_stream_->create(*device_, SENSOR_DEPTH), OK) << "depth initialisation failed :" << OpenNI::getExtendedError() << std::endl;
    CHECK_EQ(colour_stream_->create(*device_, SENSOR_COLOR), OK) << "colour initialisation failed :" << OpenNI::getExtendedError() << std::endl;

    VideoMode vm_depth = depth_stream_->getVideoMode();
    vm_depth.setResolution(image_width_, image_height_);
    vm_depth.setFps(frame_rate_);
    vm_depth.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);

    VideoMode vm_colour = colour_stream_->getVideoMode();
    vm_colour.setResolution(image_width_, image_height_);
    vm_colour.setFps(frame_rate_);
    vm_colour.setPixelFormat(PIXEL_FORMAT_RGB888);

    depth_stream_->setVideoMode(vm_depth);
    colour_stream_->setVideoMode(vm_colour);

    CHECK(device_->isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR)) << "depth registration is not supported.";
    CHECK_EQ(device_->setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR), OK) << "depth not registered.";

    LOG(INFO) << "initialisation finished." << std::endl;
}

void DepthCamera::DepthCameraImpl::start_video_streaming()
{
    depth_stream_->setMirroringEnabled(false);
    colour_stream_->setMirroringEnabled(false);

    CHECK_EQ(depth_stream_->start(), 0) << "start depth stream failed: " << OpenNI::getExtendedError() << std::endl;
    CHECK_EQ(colour_stream_->start(), 0) << "start colour stream failed: " << OpenNI::getExtendedError() << std::endl;

    depth_frame_ref_ = std::make_shared<VideoFrameRef>();
    colour_frame_ref_ = std::make_shared<VideoFrameRef>();

    LOG(INFO) << "camera stream started\n";
}

void DepthCamera::DepthCameraImpl::stop_video_streaming()
{
    depth_stream_->stop();
    colour_stream_->stop();

    depth_stream_->destroy();
    colour_stream_->destroy();

    device_->close();

    OpenNI::shutdown();
    LOG(INFO) << "camera stream stopped.\n";
}

bool DepthCamera::DepthCameraImpl::capture(cv::Mat &colour, cv::Mat &depth)
{
    VideoStream *dual_stream[] = {depth_stream_.get(), colour_stream_.get()};
    int stream_index = -1;
    auto state = STATUS_OK;
    while (state == STATUS_OK)
    {
        state = OpenNI::waitForAnyStream(dual_stream, 2, &stream_index, 0);

        if (state == STATUS_OK)
        {
            switch (stream_index)
            {
            case 0: // depth ready
            {
                CHECK_EQ(depth_stream_->readFrame(depth_frame_ref_.get()), 0);
                depth = cv::Mat(image_height_, image_width_, CV_16UC1, const_cast<void *>(depth_frame_ref_->getData()));
                break;
            }
            case 1: // colour ready
            {
                CHECK_EQ(colour_stream_->readFrame(colour_frame_ref_.get()), 0);
                colour = cv::Mat(image_height_, image_width_, CV_8UC3, const_cast<void *>(colour_frame_ref_->getData()));
                break;
            }
            default:
            {
                LOG(FATAL) << "unexpected stream index when reading images.\n";
                break;
            }
            }
        }
    }

    if (!depth_frame_ref_.get() ||
        !colour_frame_ref_.get() ||
        !depth_frame_ref_->isValid() ||
        !colour_frame_ref_->isValid())
        return false;

    current_id_ += 1;
    return true;
}

DepthCamera::DepthCamera(int width, int height, int fps) : impl(new DepthCameraImpl(width, height, fps)), DataSource()
{
}

DepthCamera::~DepthCamera()
{
    impl->stop_video_streaming();
}

void DepthCamera::start_video_streaming()
{
    impl->start_video_streaming();
}

void DepthCamera::stop_video_streaming()
{
    impl->stop_video_streaming();
}

bool DepthCamera::read_next_images(cv::Mat &image, cv::Mat &depth)
{
    impl->capture(image, depth);
    return true;
}

size_t DepthCamera::get_current_id() const
{
    return impl->current_id_;
}

double DepthCamera::get_current_timestamp() const
{
    return 0;
}

Sophus::SE3d DepthCamera::get_current_gt_pose() const
{
    return Sophus::SE3d();
}

double DepthCamera::get_current_gt_timestamp() const
{
    return 0;
}

std::vector<Sophus::SE3d> DepthCamera::get_groundtruth() const
{
    return std::vector<Sophus::SE3d>();
}

float DepthCamera::get_depth_scale() const
{
    return 1 / 1000.f;
}

Sophus::SE3d DepthCamera::get_initial_pose() const
{
    return Sophus::SE3d();
}