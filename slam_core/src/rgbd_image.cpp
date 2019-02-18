#include "rgbd_image.h"

RgbdCamera::RgbdCamera(float fx, float fy, float cx, float cy)
    : fx(fx), fy(fy), cx(cx), cy(cy), inv_fx(1.0f / fx), inv_fy(1.0f / fy)
{
}

cv::Vec3f RgbdCamera::back_project_point(int x, int y, float z) const
{
    return cv::Vec3f((x - cx) * inv_fx * z, (y - cy) * inv_fy * z, z);
}

void RgbdImage::create(const cv::Mat &intensity, const cv::Mat &depth)
{
    this->intensity = intensity;
    this->depth = depth;
}

void RgbdImage::compute_intensity_derivatives()
{
    for (int y = 0; y < intensity.rows; ++y)
    {
        float *row_dx = intensity_dx.ptr<float>(y);
        float *row_dy = intensity_dy.ptr<float>(y);
        const float *row = intensity.ptr<float>(y);
        const float *row_p_1 = intensity.ptr<float>(std::min(intensity.rows, y + 1));
        const float *row_m_1 = intensity.ptr<float>(std::max(0, y - 1));
        for (int x = 0; x < intensity.cols; ++x)
        {
            row_dx[x] = (row[std::min(intensity.cols, x + 1)] - row[std::max(0, x - 1)]) * 0.5;
            row_dy[x] = (row_p_1[x] - row_m_1[x]) * 0.5;
        }
    }
}

void RgbdImage::back_project_points(float fx, float fy, float cx, float cy)
{
    float inv_fx = 1.0f / fx;
    float inv_fy = 1.0f / fy;
    for (int y = 0; y < depth.rows; ++y)
    {
        const float *depth_row = depth.ptr<float>(y);
        cv::Vec3f *point_row = point_cloud.ptr<cv::Vec3f>(y);
        for (int x = 0; x < depth.cols; ++x)
        {
            float z = depth_row[x];
            point_row[x] = camera->back_project_point(x, y, z);
        }
    }
}

RgbdImagePyramid::RgbdImagePyramid(cv::Mat &intensity, cv::Mat &depth, RgbdCameraPyramidPtr cameras)
{

}

RgbdImagePtr RgbdImagePyramid::operator[](int level)
{
    return (level >= 0 && level < levels.size()) ? levels[level] : nullptr;
}