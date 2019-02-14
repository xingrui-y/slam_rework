#include "openni_camera.h"
#include <openni2/OpenNI.h>

class OpenNICamera::OpenNICameraImpl
{
  public:
    OpenNICameraImpl(int width, int height, int fps);
    void start_video_streaming();
    void stop_video_streaming();
    bool capture(cv::Mat &colour, cv::Mat &depth);

    openni::Device *device;
    openni::VideoStream *colour_stream;
    openni::VideoStream *depth_stream;
    openni::VideoFrameRef *colour_frame;
    openni::VideoFrameRef *depth_frame;

    int width, height, fps;
};

OpenNICamera::OpenNICameraImpl::OpenNICameraImpl(int width, int height, int fps)
    : width(width), height(height), fps(fps)
{
    if (openni::OpenNI::initialize() != openni::STATUS_OK)
    {
        printf("OpenNI Initialisation Failed with Error Message : %s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }

    device = new openni::Device();
    if (device->open(openni::ANY_DEVICE) != openni::STATUS_OK)
    {
        printf("Couldn't open device\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }

    depth_stream = new openni::VideoStream();
    colour_stream = new openni::VideoStream();
    if (depth_stream->create(*device, openni::SENSOR_DEPTH) != openni::STATUS_OK ||
        colour_stream->create(*device, openni::SENSOR_COLOR) != openni::STATUS_OK)
    {
        printf("Couldn't create streaming service\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }

    openni::VideoMode depth_video_mode = depth_stream->getVideoMode();
    depth_video_mode.setResolution(width, height);
    depth_video_mode.setFps(fps);
    depth_video_mode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

    openni::VideoMode colour_video_mode = colour_stream->getVideoMode();
    colour_video_mode.setResolution(width, height);
    colour_video_mode.setFps(fps);
    colour_video_mode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

    depth_stream->setVideoMode(depth_video_mode);
    colour_stream->setVideoMode(colour_video_mode);

    // Note: Doing image registration earlier than this point seems to fail
    if (device->isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
    {
        if (device->setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR) == openni::STATUS_OK)
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

    if (depth_stream->start() != openni::STATUS_OK)
    {
        printf("Couldn't start depth streaming service\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }

    if (colour_stream->start() != openni::STATUS_OK)
    {
        printf("Couldn't start colour streaming service\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }

    depth_frame = new openni::VideoFrameRef();
    colour_frame = new openni::VideoFrameRef();

    printf("Camera Stream Started!\n");
}

void OpenNICamera::OpenNICameraImpl::stop_video_streaming()
{
    depth_stream->stop();
    colour_stream->stop();

    depth_stream->destroy();
    colour_stream->destroy();

    device->close();

    openni::OpenNI::shutdown();
    printf("Camera Stream Successfully Stopped.\n");
}

bool OpenNICamera::OpenNICameraImpl::capture(cv::Mat &colour, cv::Mat &depth)
{
    openni::VideoStream *streams[] = {depth_stream, colour_stream};
    int streamReady = -1;
    auto state = openni::STATUS_OK;
    while (state == openni::STATUS_OK)
    {
        state = openni::OpenNI::waitForAnyStream(streams, 2, &streamReady, 0);
        if (state == openni::STATUS_OK)
        {
            switch (streamReady)
            {
            case 0:
            {
                if (depth_stream->readFrame(depth_frame) != openni::STATUS_OK)
                {
                    printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
                    return false;
                }

                depth = cv::Mat(height, width, CV_16UC1, const_cast<void *>(depth_frame->getData()));
                break;
            }
            case 1:
            {
                if (colour_stream->readFrame(colour_frame) != openni::STATUS_OK)
                {
                    printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
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
}

OpenNICamera::~OpenNICamera()
{
    stop_video_streaming();
}

bool OpenNICamera::capture(cv::Mat &colour, cv::Mat &depth)
{
    return impl->capture(colour, depth);
}

void OpenNICamera::start_video_streaming()
{
    impl->start_video_streaming();
}

void OpenNICamera::stop_video_streaming()
{
    impl->stop_video_streaming();
}