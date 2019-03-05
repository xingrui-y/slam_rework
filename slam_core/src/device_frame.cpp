#include "device_frame.h"

DeviceFrame::DeviceFrame(const RgbdFramePtr data) : owner_(data)
{
    int max_level = data->get_pyramid_level();
    depth_.resize(max_level);
    intensity_.resize(max_level);
    intensity_dx_.resize(max_level);
    intensity_dy_.resize(max_level);
    point_cloud_.resize(max_level);
    normal_.resize(max_level);

    upload(data);
}

void DeviceFrame::upload(const RgbdFramePtr data)
{
    RgbdImagePtr image_pyramid = data->get_image_pyramid();
    int max_level = data->get_pyramid_level();

    for (int level = 0; level < max_level; ++level)
    {
        const cv::Mat &depth_map = image_pyramid->get_depth_map(level);
        const cv::Mat &intensity_map = image_pyramid->get_intensity_map(level);
        const cv::Mat &intensity_dx_map = image_pyramid->get_intensity_dx_map(level);
        const cv::Mat &intensity_dy_map = image_pyramid->get_intensity_dy_map(level);
        const cv::Mat &point_cloud = image_pyramid->get_point_cloud(level);
        const cv::Mat &normal_map = image_pyramid->get_normal_map(level);

        depth_[level].upload(depth_map);
        intensity_[level].upload(intensity_map);
        intensity_dx_[level].upload(intensity_dx_map);
        intensity_dy_[level].upload(intensity_dy_map);
        point_cloud_[level].upload(point_cloud);
        normal_[level].upload(normal_map);
    }
}