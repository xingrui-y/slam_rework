#include "rgbd_image.h"

class RgbdImage::RgbdImageImpl
{
  public:
    RgbdImageImpl();

    cv::Mat intensity;
    cv::Mat depth;
    cv::Mat intensity_dx;
    cv::Mat intensity_dy;

    void initialization();
    void compute_intensity_derivatives();
};

RgbdImage::RgbdImageImpl::RgbdImageImpl()
{
}

void RgbdImage::RgbdImageImpl::compute_intensity_derivatives()
{
    for (int x = 0; x < intensity.cols; ++x)
    {
        for (int y = 0; y < intensity.rows; ++y)
        {
            intensity_dx.at<float>(y, x) = (intensity.at<float>(y, std::min(intensity.cols, x + 1)) - intensity.at<float>(y, std::max(0, x - 1))) * 0.5;
            intensity_dy.at<float>(y, x) = (intensity.at<float>(std::min(intensity.rows, y + 1), x) - intensity.at<float>(std::max(0, y - 1), x)) * 0.5;
        }
    }
}

RgbdImage::RgbdImage() : impl(new RgbdImageImpl())
{
}

void RgbdImage::create(const cv::Mat &intensity, const cv::Mat &depth)
{
    impl->intensity = intensity;
    impl->depth = depth;
}

void RgbdImage::compute_intensity_derivatives()
{
    impl->compute_intensity_derivatives();
}