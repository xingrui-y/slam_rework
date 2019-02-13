#include "rgbd_image.h"

RgbdImage::RgbdImage()
{
}

void RgbdImage::create(const cv::Mat &intensity, const cv::Mat &depth)
{
    this->intensity = intensity;
    this->depth = depth;
}

void RgbdImage::compute_intensity_derivatives()
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