#include "openni_camera.h"
#include <openni2/OpenNI.h>

using namespace openni;

class OpenNICamera::OpenNICameraImpl
{
  public:
    OpenNICameraImpl(int width, int height, int fps);
    void start_video_streaming();
    void stop_video_streaming();
    bool capture(cv::Mat &colour, cv::Mat &depth);

    Device *device;
    VideoStream *colour_stream;
    VideoStream *depth_stream;
    VideoFrameRef *colour_frame;
    VideoFrameRef *depth_frame;

    int width, height, fps;
};

OpenNICamera::OpenNICameraImpl::OpenNICameraImpl(int width, int height, int fps)
    : width(width), height(height), fps(fps)
{
    if (OpenNI::initialize() != STATUS_OK)
    {
        printf("OpenNI Initialisation Failed with Error Message : %s\n", OpenNI::getExtendedError());
        exit(0);
    }

    device = new Device();
    if (device->open(ANY_DEVICE) != STATUS_OK)
    {
        printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
        exit(0);
    }

    depth_stream = new VideoStream();
    colour_stream = new VideoStream();
    if (depth_stream->create(*device, SENSOR_DEPTH) != STATUS_OK ||
        colour_stream->create(*device, SENSOR_COLOR) != STATUS_OK)
    {
        printf("Couldn't create streaming service\n%s\n", OpenNI::getExtendedError());
        exit(0);
    }

    VideoMode depth_video_mode = depth_stream->getVideoMode();
    depth_video_mode.setResolution(width, height);
    depth_video_mode.setFps(fps);
    depth_video_mode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);

    VideoMode colour_video_mode = colour_stream->getVideoMode();
    colour_video_mode.setResolution(width, height);
    colour_video_mode.setFps(fps);
    colour_video_mode.setPixelFormat(PIXEL_FORMAT_RGB888);

    depth_stream->setVideoMode(depth_video_mode);
    colour_stream->setVideoMode(colour_video_mode);

    // Note: Doing image registration earlier than this point seems to fail
    if (device->isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR))
    {
        if (device->setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR) == STATUS_OK)
        {
            printf("Depth To Colour Image Registration Set Success\n");
        }
        else
        {
            printf("Depth To Colour Image Registration Set FAILED\n");
        }
    }
    else
    {
        printf("Depth To Colour Image Registration is NOT Supported!!!\n");
    }

    printf("OpenNI Camera Initialisation Complete!\n");
}

void OpenNICamera::OpenNICameraImpl::start_video_streaming()
{
    depth_stream->setMirroringEnabled(false);
    colour_stream->setMirroringEnabled(false);

    if (depth_stream->start() != STATUS_OK)
    {
        printf("Couldn't start depth streaming service\n%s\n", OpenNI::getExtendedError());
        exit(0);
    }

    if (colour_stream->start() != STATUS_OK)
    {
        printf("Couldn't start colour streaming service\n%s\n", OpenNI::getExtendedError());
        exit(0);
    }

    depth_frame = new VideoFrameRef();
    colour_frame = new VideoFrameRef();

    printf("Camera Stream Started!\n");
}

void OpenNICamera::OpenNICameraImpl::stop_video_streaming()
{
    depth_stream->stop();
    colour_stream->stop();

    depth_stream->destroy();
    colour_stream->destroy();

    device->close();

    OpenNI::shutdown();
    printf("Camera Stream Successfully Stopped.\n");
}

bool OpenNICamera::OpenNICameraImpl::capture(cv::Mat &colour, cv::Mat &depth)
{
    VideoStream *streams[] = {depth_stream, colour_stream};
    int streamReady = -1;
    auto state = STATUS_OK;
    while (state == STATUS_OK)
    {
        state = OpenNI::waitForAnyStream(streams, 2, &streamReady, 0);
        if (state == STATUS_OK)
        {
            switch (streamReady)
            {
            case 0:
            {
                if (depth_stream->readFrame(depth_frame) != STATUS_OK)
                {
                    printf("Read failed!\n%s\n", OpenNI::getExtendedError());
                    return false;
                }

                depth = cv::Mat(height, width, CV_16UC1, const_cast<void *>(depth_frame->getData()));
                break;
            }
            case 1:
            {
                if (colour_stream->readFrame(colour_frame) != STATUS_OK)
                {
                    printf("Read failed!\n%s\n", OpenNI::getExtendedError());
                    return false;
                }

                colour = cv::Mat(height, width, CV_8UC3, const_cast<void *>(colour_frame->getData()));
                break;
            }
            default:
                printf("Unexpected stream number!\n");
                return false;
            }
        }
    }

    if (!colour_frame || !depth_frame || !colour_frame->isValid() || !depth_frame->isValid())
        return false;

    return true;
}

OpenNICamera::OpenNICamera(int width, int height, int fps)
    : impl(new OpenNICameraImpl(width, height, fps))
{
    impl->start_video_streaming();
}

OpenNICamera::~OpenNICamera()
{
    impl->stop_video_streaming();
}

bool OpenNICamera::read_next_images(cv::Mat &image, cv::Mat &depth)
{
    impl->capture(image, depth);
    return true;
}

Sophus::SE3d OpenNICamera::get_starting_pose() const
{
    return Sophus::SE3d();
}

double OpenNICamera::get_current_timestamp() const
{
    return 0;
}

unsigned int OpenNICamera::get_current_id() const
{
    return 0;
}

std::vector<Sophus::SE3d> OpenNICamera::get_groundtruth() const
{
    return std::vector<Sophus::SE3d>();
}

float OpenNICamera::get_depth_scale() const
{
    return 1.f / 1000.f;
}