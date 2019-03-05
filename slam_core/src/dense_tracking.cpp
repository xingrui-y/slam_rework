#include "dense_tracking.h"
#include "se3_reduction.h"
#include "revertable.h"
#include "device_frame.h"
#include "stop_watch.h"

class DenseTracking::DenseTrackingImpl
{
  public:
    DenseTrackingImpl();
    TrackingResult compute_transform(const DeviceFramePtr reference, const DeviceFramePtr current, const TrackingContext &c);
    TrackingResult compute_transform(const RgbdFramePtr reference, const RgbdFramePtr current, const TrackingContext &c);

    Eigen::Matrix<float, 6, 6> JtJ_;
    Eigen::Matrix<float, 6, 1> Jtr_;
    Eigen::Matrix<double, 6, 1> increment_;
    Eigen::Matrix<float, 2, 1> residual_icp_;
    Eigen::Matrix<float, 2, 1> residual_rgb_;

    Revertable<Sophus::SE3d> estimated_update_;
    Revertable<float> residual_sum_;

    cv::cuda::GpuMat sum_se3_;
    cv::cuda::GpuMat out_se3_;

    DeviceFramePtr reference_;
    DeviceFramePtr current_;
};

DenseTracking::DenseTrackingImpl::DenseTrackingImpl()
{
    JtJ_.setZero();
    Jtr_.setZero();
    increment_.setZero();
    residual_icp_.setZero();
    residual_rgb_.setZero();
    residual_sum_ = Revertable<float>(0);
    sum_se3_.create(96, 29, CV_32FC1);
    out_se3_.create(1, 29, CV_32FC1);
}

TrackingResult DenseTracking::DenseTrackingImpl::compute_transform(const DeviceFramePtr reference, const DeviceFramePtr current, const TrackingContext &c)
{
    bool sucess = true;

    if (c.use_initial_guess_)
        estimated_update_ = Revertable<Sophus::SE3d>(c.initial_estimate_);
    else
        estimated_update_ = Revertable<Sophus::SE3d>();

    Eigen::Matrix<float, 6, 6> jtj_icp = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 6> jtj_rgb = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> jtr_icp = Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix<float, 6, 1> jtr_rgb = Eigen::Matrix<float, 6, 1>::Zero();

    for (int level = c.tracking_level_.size() - 1; level >= 0; --level)
    {
        auto curr_vmap = current->point_cloud_[level];
        auto curr_nmap = current->normal_[level];
        auto last_vmap = reference_->point_cloud_[level];
        auto last_nmap = reference_->normal_[level];
        auto curr_intensity = current->intensity_[level];
        auto last_intensity = reference_->intensity_[level];
        auto intensity_dx = reference->intensity_dx_[level];
        auto intensity_dy = reference->intensity_dy_[level];

        IntrinsicMatrixPtr K = c.intrinsics_[level];

        for (int iter = 0; iter < c.tracking_level_[level]; ++iter)
        {
            auto current_estimate = estimated_update_.value();

            icp_reduce(curr_vmap,
                       curr_nmap,
                       last_vmap,
                       last_nmap,
                       sum_se3_,
                       out_se3_,
                       current_estimate,
                       K,
                       jtj_icp.data(),
                       jtr_icp.data(),
                       residual_icp_.data());

            rgb_reduce(curr_intensity,
                       last_intensity,
                       last_vmap,
                       curr_vmap,
                       intensity_dx,
                       intensity_dy,
                       sum_se3_,
                       out_se3_,
                       current_estimate,
                       K,
                       jtj_rgb.data(),
                       jtr_rgb.data(),
                       residual_rgb_.data());

            JtJ_ = jtj_icp + 0.1 * jtj_rgb;
            Jtr_ = jtr_icp + 0.1 * jtr_rgb;

            increment_ = JtJ_.cast<double>().ldlt().solve(Jtr_.cast<double>());
            estimated_update_.update(Sophus::SE3d::exp(increment_) * current_estimate);
        }
    }

    TrackingResult result;
    if (sucess)
    {
        result.sucess = true;
        result.update = estimated_update_.value();
    }
    else
    {
        result.sucess = false;
    }

    return result;
}

TrackingResult DenseTracking::DenseTrackingImpl::compute_transform(const RgbdFramePtr reference, const RgbdFramePtr current, const TrackingContext &c)
{
    if (reference_ == nullptr)
        reference_ = std::make_shared<DeviceFrame>(reference);
    else
        reference_->upload(reference);

    if (current_ == nullptr)
        current_ = std::make_shared<DeviceFrame>(current);
    else
        current_->upload(current);

    return compute_transform(reference_, current_, c);
}

DenseTracking::DenseTracking() : impl(new DenseTrackingImpl())
{
}

TrackingResult DenseTracking::track(const RgbdFramePtr reference, const RgbdFramePtr current, const TrackingContext &c)
{
    return impl->compute_transform(reference, current, c);
}