#include "dense_odometry.h"
#include "se3_reduction.h"
#include "dense_tracking.h"

class DenseOdometry::DenseOdometryImpl
{
  public:
    DenseOdometryImpl();
    void track(const RgbdFramePtr current);
    bool keyframe_required() const;

    std::unique_ptr<DenseTracking> tracker;
    RgbdFramePtr current_keyframe;
    RgbdFramePtr last_frame;
    Sophus::SE3d initial_pose;
    std::vector<Sophus::SE3d> trajectory;
};

DenseOdometry::DenseOdometryImpl::DenseOdometryImpl()
    : tracker(new DenseTracking()), last_frame(nullptr), current_keyframe(nullptr)
{
}

bool DenseOdometry::DenseOdometryImpl::keyframe_required() const
{
    return true;
}

void DenseOdometry::DenseOdometryImpl::track(const RgbdFramePtr current)
{
    if (current_keyframe == nullptr)
    {
        current->set_pose(initial_pose);
        current_keyframe = last_frame = current;
    }
    else
    {
        TrackingContext context;
        context.tracking_level_ = {10, 5, 3, 3, 3};
        context.intrinsics_ = current->get_intrinsics();
        context.use_initial_guess_ = true;
        context.initial_estimate_ = current_keyframe->get_pose().inverse() * last_frame->get_pose();

        TrackingResult result = tracker->track(current_keyframe, current, context);

        if (result.sucess)
        {
            current->set_pose(current_keyframe->get_pose() * result.update);
            trajectory.push_back(current->get_pose());
            last_frame = current;
        }
        else
        {
            return;
        }

        if (keyframe_required())
        {
            current_keyframe = last_frame = current;
        }
    }
}

DenseOdometry::DenseOdometry() : impl(new DenseOdometryImpl())
{
}

void DenseOdometry::track(const cv::Mat &intensity, const cv::Mat &depth, const IntrinsicMatrixPyramid &K, const unsigned long id, const double time_stamp)
{
    impl->track(std::make_shared<RgbdFrame>(intensity, depth, K, id, time_stamp));
}

void DenseOdometry::set_initial_pose(const Sophus::SE3d &pose)
{
    impl->initial_pose = pose;
}

RgbdFramePtr DenseOdometry::get_current_frame() const
{
    return impl->last_frame;
}

RgbdFramePtr DenseOdometry::get_current_keyframe() const
{
    return impl->current_keyframe;
}

std::vector<Sophus::SE3d> DenseOdometry::get_camera_trajectory() const
{
    return impl->trajectory;
}
