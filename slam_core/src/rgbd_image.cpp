#include "rgbd_image.h"
#include "stop_watch.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/features2d.hpp>

typedef std::vector<cv::Mat> ImagePyramid;

class RgbdImage::RgbdImageImpl
{
  public:
    RgbdImageImpl(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera);
    ~RgbdImageImpl();
    void build_pyramid_derivative(const ImagePyramid &intensity, ImagePyramid &pyramid_dx, ImagePyramid &pyramid_dy);
    void build_pyramid_gaussian(const cv::Mat &origin, ImagePyramid &pyramid, int max_level);
    void build_pyramid_subsample(const cv::Mat &origin, ImagePyramid &pyramid, int max_level);
    void compute_point_cloud_pyramid(const ImagePyramid &depth, ImagePyramid &vmap, const IntrinsicMatrixPyramid K);
    void compute_surface_normal_pyramid(const ImagePyramid &vmap, ImagePyramid &nmap);

    // sse speed up
    void compute_surface_normal_pyramid_sse(const ImagePyramid &vmap, ImagePyramid &nmap);

    ImagePyramid intensity_;
    ImagePyramid depth_;
    ImagePyramid point_cloud_;
    ImagePyramid intensity_dx_;
    ImagePyramid intensity_dy_;
    ImagePyramid normal_;

  private:
    void compute_point_cloud(const cv::Mat &depth, cv::Mat &vmap, const IntrinsicMatrixPtr K);
    void compute_surface_normal(const cv::Mat &vmap, cv::Mat &nmap);
    void compute_surface_normal_sse(const cv::Mat &vmap, cv::Mat &nmap);
    __m128 cross_product_sse(__m128 a, __m128 b);
    __m128 cross_product_normalized_sse(__m128 a, __m128 b);
};

RgbdImage::RgbdImageImpl::RgbdImageImpl(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera)
{
    int max_level = camera.size();
    build_pyramid_gaussian(intensity, intensity_, max_level);
    build_pyramid_subsample(depth, depth_, max_level);
    build_pyramid_derivative(intensity_, intensity_dx_, intensity_dy_);
    compute_point_cloud_pyramid(depth_, point_cloud_, camera);
    compute_surface_normal_pyramid_sse(point_cloud_, normal_);
}

void RgbdImage::RgbdImageImpl::build_pyramid_derivative(const ImagePyramid &intensity, ImagePyramid &pyramid_dx, ImagePyramid &pyramid_dy)
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

void RgbdImage::RgbdImageImpl::build_pyramid_gaussian(const cv::Mat &origin, ImagePyramid &pyramid, int max_level)
{
    cv::buildPyramid(origin, pyramid, max_level - 1);
}

void RgbdImage::RgbdImageImpl::build_pyramid_subsample(const cv::Mat &origin, ImagePyramid &pyramid, int max_level)
{
    pyramid.resize(max_level);
    pyramid[0] = origin;
    for (int level = 0; level < max_level - 1; ++level)
    {
        cv::resize(pyramid[level], pyramid[level + 1], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    }
}

void RgbdImage::RgbdImageImpl::compute_point_cloud(const cv::Mat &depth, cv::Mat &vmap, const IntrinsicMatrixPtr K)
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

void RgbdImage::RgbdImageImpl::compute_point_cloud_pyramid(const ImagePyramid &depth, ImagePyramid &vmap, const IntrinsicMatrixPyramid K)
{
    vmap.resize(depth.size());
    for (int level = 0; level < depth.size(); ++level)
    {
        compute_point_cloud(depth[level], vmap[level], K[level]);
    }
}

__m128 inline RgbdImage::RgbdImageImpl::cross_product_sse(__m128 a, __m128 b)
{
    __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 c = _mm_sub_ps(_mm_mul_ps(a, b_yzx), _mm_mul_ps(a_yzx, b));
    return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
}

__m128 inline RgbdImage::RgbdImageImpl::cross_product_normalized_sse(__m128 a, __m128 b)
{
    __m128 dst = cross_product_sse(a, b);
    return _mm_mul_ps(dst, _mm_rcp_ps(_mm_mul_ps(dst, dst)));
}

void RgbdImage::RgbdImageImpl::compute_surface_normal_sse(const cv::Mat &vmap, cv::Mat &nmap)
{
    int cols = vmap.cols;
    int rows = vmap.rows;

    if (nmap.empty())
        nmap.create(rows, cols, CV_32FC3);

    float *tmp = (float *)aligned_alloc(16, sizeof(float) * 4);
    for (int y = 0; y < rows; ++y)
    {
        cv::Vec3f *row = nmap.ptr<cv::Vec3f>(y);
        for (int x = 0; x < cols; ++x)
        {
            int x0 = std::max(x - 1, 0);
            int x1 = std::min(x + 1, cols - 1);
            int y0 = std::max(y - 1, 0);
            int y1 = std::min(y + 1, rows - 1);

            __m128 left = _mm_loadu_ps(&vmap.ptr<float>(y)[x0 * 3]);
            __m128 right = _mm_loadu_ps(&vmap.ptr<float>(y)[x1 * 3]);
            __m128 up = _mm_loadu_ps(&vmap.ptr<float>(y0)[x * 3]);
            __m128 down = _mm_loadu_ps(&vmap.ptr<float>(y1)[x * 3]);
            __m128 c = cross_product_normalized_sse(_mm_sub_ps(right, left), _mm_sub_ps(up, down));
            _mm_store_ps(tmp, c);
            nmap.ptr<cv::Vec3f>(y)[x] = cv::Vec3f(tmp);
        }
    }
}

void RgbdImage::RgbdImageImpl::compute_surface_normal(const cv::Mat &vmap, cv::Mat &nmap)
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

void RgbdImage::RgbdImageImpl::compute_surface_normal_pyramid(const ImagePyramid &vmap, ImagePyramid &nmap)
{
    nmap.resize(vmap.size());
    for (int level = 0; level < vmap.size(); ++level)
    {
        compute_surface_normal(vmap[level], nmap[level]);
    }
}

void RgbdImage::RgbdImageImpl::compute_surface_normal_pyramid_sse(const ImagePyramid &vmap, ImagePyramid &nmap)
{
    nmap.resize(vmap.size());
    for (int level = 0; level < vmap.size(); ++level)
    {
        compute_surface_normal_sse(vmap[level], nmap[level]);
    }
}

RgbdImage::RgbdImage(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera) : impl(new RgbdImageImpl(intensity, depth, camera))
{
}

RgbdImage::RgbdImageImpl::~RgbdImageImpl()
{
    depth_.clear();
    intensity_.clear();
    intensity_dx_.clear();
    intensity_dy_.clear();
    point_cloud_.clear();
    normal_.clear();
}

cv::Mat RgbdImage::get_depth_map(int level) const
{
    return impl->depth_[level];
}

cv::Mat RgbdImage::get_intensity_map(int level) const
{
    return impl->intensity_[level];
}

cv::Mat RgbdImage::get_intensity_dx_map(int level) const
{
    return impl->intensity_dx_[level];
}

cv::Mat RgbdImage::get_intensity_dy_map(int level) const
{
    return impl->intensity_dy_[level];
}

cv::Mat RgbdImage::get_point_cloud(int level) const
{
    return impl->point_cloud_[level];
}

cv::Mat RgbdImage::get_normal_map(int level) const
{
    return impl->normal_[level];
}

// RgbdFrame definition
class RgbdFrame::RgbdFrameImpl
{
  public:
    RgbdFrameImpl(const cv::Mat &image, const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera, unsigned long id, double time_stamp);

    cv::Mat image;
    double time_stamp_;
    unsigned long id_;
    PoseStructPtr pose_;
    RgbdImagePtr data_;
    RgbdFramePtr reference_;
    IntrinsicMatrixPyramid intrinsics_;
};

RgbdFrame::RgbdFrameImpl::RgbdFrameImpl(const cv::Mat &image, const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera, unsigned long id, double time_stamp)
    : data_(new RgbdImage(intensity, depth, camera)), pose_(new PoseStruct()), id_(id), time_stamp_(time_stamp),
      intrinsics_(camera), reference_(nullptr)
{
    image.copyTo(this->image);
}

RgbdFrame::RgbdFrame(const cv::Mat &image, const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid camera, unsigned long id, double time_stamp)
    : impl(new RgbdFrameImpl(image, intensity, depth, camera, id, time_stamp))
{
}

unsigned long RgbdFrame::get_id() const
{
    return impl->id_;
}

Sophus::SE3d RgbdFrame::get_pose() const
{
    if (impl->pose_ != nullptr)
        return impl->pose_->world_pose_;
    return Sophus::SE3d();
}

PoseStructPtr RgbdFrame::get_pose_struct() const
{
    return impl->pose_;
}

void RgbdFrame::set_reference(const RgbdFramePtr reference)
{
    impl->pose_->reference_ = reference;
}

void RgbdFrame::set_pose(const Sophus::SE3d &pose)
{
    impl->pose_->world_pose_ = pose;
}

int RgbdFrame::get_pyramid_level() const
{
    return impl->intrinsics_.size();
}

RgbdImagePtr RgbdFrame::get_image_pyramid() const
{
    return impl->data_;
}

IntrinsicMatrixPyramid RgbdFrame::get_intrinsics() const
{
    return impl->intrinsics_;
}

cv::Mat RgbdFrame::get_image() const
{
    return impl->image;
}