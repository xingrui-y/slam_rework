#include "rgbd_image.h"
#include "stop_watch.h"

RgbdImage::RgbdImage(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera)
{
    int max_level = camera.size();
    build_pyramid_gaussian(intensity, intensity_, max_level);
    build_pyramid_subsample(depth, depth_, max_level);
    build_pyramid_derivative(intensity_, intensity_dx_, intensity_dy_);
    compute_point_cloud_pyramid(depth_, point_cloud_, camera);
    compute_surface_normal_pyramid(point_cloud_, normal_);
}

void RgbdImage::build_pyramid_derivative(const ImagePyramid &intensity, ImagePyramid &pyramid_dx, ImagePyramid &pyramid_dy)
{
    int max_level = intensity.size();
    pyramid_dx.resize(max_level);
    pyramid_dy.resize(max_level);
    for (int level = 0; level < max_level; ++level)
    {
        cv::Sobel(intensity[level], pyramid_dx[level], CV_32FC1, 1, 0, CV_SCHARR, 1.0 / 36);
        cv::Sobel(intensity[level], pyramid_dy[level], CV_32FC1, 0, 1, CV_SCHARR, 1.0 / 36);
    }
}

void RgbdImage::build_pyramid_gaussian(const cv::Mat &origin, ImagePyramid &pyramid, int max_level)
{
    cv::buildPyramid(origin, pyramid, max_level - 1);
}

void RgbdImage::build_pyramid_subsample(const cv::Mat &origin, ImagePyramid &pyramid, int max_level)
{
    pyramid.resize(max_level);
    pyramid[0] = origin;
    for (int level = 0; level < max_level - 1; ++level)
    {
        cv::resize(pyramid[level], pyramid[level + 1], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    }
}

void RgbdImage::compute_point_cloud(const cv::Mat &depth, cv::Mat &vmap, const IntrinsicMatrixPtr K)
{
    int cols = depth.cols;
    int rows = depth.rows;
    if (vmap.empty())
        vmap.create(rows, cols, CV_32FC3);
    for (int y = 0; y < rows; ++y)
    {
        const float *depth_row = depth.ptr<float>(y);
        cv::Vec3f *point_row = vmap.ptr<cv::Vec3f>(y);
        for (int x = 0; x < cols; ++x)
        {
            float z = depth_row[x];
            point_row[x] = z * cv::Vec3f((x - K->cx) * K->invfx, (y - K->cy) * K->invfy, 1);
        }
    }
}

void RgbdImage::compute_point_cloud_pyramid(const ImagePyramid &depth, ImagePyramid &vmap, const IntrinsicMatrixPyramid K)
{
    vmap.resize(depth.size());
    for (int level = 0; level < depth.size(); ++level)
    {
        compute_point_cloud(depth[level], vmap[level], K[level]);
    }
}

void compute_surface_normal(const cv::Mat &vmap, cv::Mat &nmap)
{
    int cols = vmap.cols;
    int rows = vmap.rows;

    if (nmap.empty())
        nmap.create(rows, cols, CV_32FC3);

    for (int y = 0; y < rows; ++y)
    {
        const cv::Vec3f *vmap_row = vmap.ptr<cv::Vec3f>(y);
        for (int x = 0; x < cols; ++x)
        {
            int x0 = std::max(x - 1, 0);
            int x1 = std::min(x + 1, cols - 1);
            int y0 = std::max(y - 1, 0);
            int y1 = std::min(y + 1, rows - 1);

            const cv::Vec3f &left = vmap_row[x0];
            const cv::Vec3f &right = vmap_row[x1];
            const cv::Vec3f &up = vmap.ptr<cv::Vec3f>(y0)[x];
            const cv::Vec3f &down = vmap.ptr<cv::Vec3f>(y1)[x];

            nmap.ptr<cv::Vec3f>(y)[x] = cv::normalize((right - left).cross(up - down));
        }
    }
}

void RgbdImage::compute_surface_normal_pyramid(const ImagePyramid &vmap, ImagePyramid &nmap)
{
    nmap.resize(vmap.size());
    for (int level = 0; level < vmap.size(); ++level)
    {
        compute_surface_normal(vmap[level], nmap[level]);
    }
}

RgbdImage::~RgbdImage()
{
    depth_.clear();
    intensity_.clear();
    intensity_dx_.clear();
    intensity_dy_.clear();
}

cv::Mat RgbdImage::get_depth_map(int level) const
{
    return depth_[level];
}

cv::Mat RgbdImage::get_intensity_map(int level) const
{
    return intensity_[level];
}

cv::Mat RgbdImage::get_intensity_dx_map(int level) const
{
    return intensity_dx_[level];
}

cv::Mat RgbdImage::get_intensity_dy_map(int level) const
{
    return intensity_dy_[level];
}

cv::Mat RgbdImage::get_point_cloud(int level) const
{
    return point_cloud_[level];
}

cv::Mat RgbdImage::get_normal_map(int level) const
{
    return normal_[level];
}

// RgbdFrame definition
RgbdFrame::RgbdFrame(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera, unsigned long id, double time_stamp)
    : data_(new RgbdImage(intensity, depth, camera)), pose_(new PoseStruct()), id_(id), time_stamp_(time_stamp),
      intrinsics_(camera), reference_(nullptr)
{
}

unsigned long RgbdFrame::get_id() const
{
    return id_;
}

Sophus::SE3d RgbdFrame::get_pose() const
{
    if (pose_ != nullptr)
        return pose_->world_pose_;
    return Sophus::SE3d();
}

PoseStructPtr RgbdFrame::get_pose_struct() const
{
    return pose_;
}

void RgbdFrame::set_reference(const RgbdFramePtr reference)
{
    pose_->reference_ = reference;
}

void RgbdFrame::set_pose(const Sophus::SE3d &pose)
{
    pose_->world_pose_ = pose;
}

int RgbdFrame::get_pyramid_level() const
{
    return intrinsics_.size();
}

RgbdImagePtr RgbdFrame::get_image_pyramid() const
{
    return data_;
}

IntrinsicMatrixPyramid RgbdFrame::get_intrinsics() const
{
    return intrinsics_;
}