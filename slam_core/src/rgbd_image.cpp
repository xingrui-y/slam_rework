#include "rgbd_image.h"
#include "stop_watch.h"
#include "image_ops.h"

class RgbdImage::RgbdImageImpl
{
  public:
    RgbdImageImpl();
    RgbdImageImpl(const int &max_level);

    void resize_pyramid(const int &max_level);
    void upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr);

    RgbdFramePtr reference_frame_;
    cv::cuda::GpuMat image_;
    cv::cuda::GpuMat image_float_;
    cv::cuda::GpuMat depth_float_;
    cv::cuda::GpuMat intensity_float_;
    std::vector<cv::cuda::GpuMat> depth_;
    std::vector<cv::cuda::GpuMat> intensity_;
    std::vector<cv::cuda::GpuMat> intensity_dx_;
    std::vector<cv::cuda::GpuMat> intensity_dy_;
    std::vector<cv::cuda::GpuMat> point_cloud_;
    std::vector<cv::cuda::GpuMat> normal_;
};

RgbdImage::RgbdImageImpl::RgbdImageImpl()
{
}

RgbdImage::RgbdImageImpl::RgbdImageImpl(const int &max_level)
{
    resize_pyramid(max_level);
}

void RgbdImage::RgbdImageImpl::resize_pyramid(const int &max_level)
{
    depth_.resize(max_level);
    intensity_.resize(max_level);
    intensity_dx_.resize(max_level);
    intensity_dy_.resize(max_level);
    point_cloud_.resize(max_level);
    normal_.resize(max_level);
}

static inline void imshow(const char *name, const cv::cuda::GpuMat image)
{
    cv::Mat image_cpu;
    image.download(image_cpu);
    cv::imshow(name, image_cpu);
}

void RgbdImage::RgbdImageImpl::upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr)
{
    assert(frame != nullptr);
    assert(intrinsics_pyr != nullptr);

    if (frame == reference_frame_)
        return;

    const int max_level = intrinsics_pyr->get_max_level();

    if (max_level != depth_.size())
        resize_pyramid(max_level);

    cv::Mat image = frame->get_image();
    cv::Mat depth = frame->get_depth();

    image_.upload(image);
    depth_float_.upload(depth);
    image_.convertTo(image_float_, CV_32FC3);
    cv::cuda::cvtColor(image_float_, intensity_float_, cv::COLOR_RGB2GRAY);

    slam::cuda::build_depth_pyramid(depth_float_, depth_, max_level);
    slam::cuda::build_intensity_pyramid(intensity_float_, intensity_, max_level);
    slam::cuda::build_intensity_derivative_pyramid(intensity_, intensity_dx_, intensity_dy_);
    slam::cuda::build_point_cloud_pyramid(depth_, point_cloud_, intrinsics_pyr);
    slam::cuda::build_normal_pyramid(point_cloud_, normal_);

    reference_frame_ = frame;
}

RgbdImage::RgbdImage() : impl(new RgbdImageImpl())
{
}

RgbdImage::RgbdImage(const int &max_level) : impl(new RgbdImageImpl(max_level))
{
}

void RgbdImage::upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr)
{
    impl->upload(frame, intrinsics_pyr);
}

cv::cuda::GpuMat RgbdImage::get_depth(const int &level) const
{
    if (level < impl->depth_.size())
        return impl->depth_[level];
}

cv::cuda::GpuMat RgbdImage::get_image(const int &level) const
{
    if (level == 0)
        return impl->image_;
    else
    {
        return impl->image_;
    }
}

RgbdFramePtr RgbdImage::get_reference_frame() const
{
    return impl->reference_frame_;
}

class RgbdFrame::RgbdFrameImpl
{
  public:
    RgbdFrameImpl(const cv::Mat &image, const cv::Mat &depth_float, ulong id, double time_stamp);

    cv::Mat image_;
    cv::Mat depth_;
    unsigned long id_;
    double time_stamp_;
    Sophus::SE3d pose_;
    IntrinsicMatrixPyramidPtr intrinsics_pyr_;
};

RgbdFrame::RgbdFrameImpl::RgbdFrameImpl(const cv::Mat &image, const cv::Mat &depth_float, ulong id, double time_stamp)
    : image_(image.clone()), depth_(depth_float.clone()), id_(id), time_stamp_(time_stamp)
{
}

RgbdFrame::RgbdFrame(const cv::Mat &image, const cv::Mat &depth_float, ulong id, double time_stamp) : impl(new RgbdFrameImpl(image, depth_float, id, time_stamp))
{
}

cv::Mat RgbdFrame::get_image() const
{
    return impl->image_;
}

cv::Mat RgbdFrame::get_depth() const
{
    return impl->depth_;
}

Sophus::SE3d RgbdFrame::get_pose() const
{
    return impl->pose_;
}

void RgbdFrame::set_pose(const Sophus::SE3d &pose)
{
    impl->pose_ = pose;
}