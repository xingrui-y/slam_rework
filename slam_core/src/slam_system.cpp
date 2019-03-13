#include "slam_system.h"
#include "dense_odometry.h"
#include "intrinsic_matrix.h"
#include "keyframe_graph.h"
#include "stop_watch.h"

class SlamSystem::SlamSystemImpl
{
public:
  SlamSystemImpl() = default;
  SlamSystemImpl(const SlamSystemImpl &) = delete;
  SlamSystemImpl(const IntrinsicMatrix &K);

  IntrinsicMatrixPyramid intrincis_matrix;
  std::unique_ptr<DenseOdometry> odometry;
  std::unique_ptr<KeyFrameGraph> graph;
};

SlamSystem::SlamSystemImpl::SlamSystemImpl(const IntrinsicMatrix &K)
    : odometry(new DenseOdometry()), graph(new KeyFrameGraph())
{
  intrincis_matrix = K.build_pyramid();
}

SlamSystem::SlamSystem(const IntrinsicMatrix &K) : impl(new SlamSystemImpl(K))
{
}

void SlamSystem::update(const cv::Mat &image, const cv::Mat &intensity, const cv::Mat &depth, const unsigned long id, const double time_stamp)
{
  impl->odometry->track(image, intensity, depth, impl->intrincis_matrix, id, time_stamp);
}

RgbdFramePtr SlamSystem::get_current_frame() const
{
  return impl->odometry->get_current_frame();
}

Sophus::SE3d SlamSystem::get_current_pose() const
{
  RgbdFramePtr current = impl->odometry->get_current_frame();
  if (current != nullptr)
    return current->get_pose();
  return Sophus::SE3d();
}

void SlamSystem::set_initial_pose(const Sophus::SE3d &pose)
{
  impl->odometry->set_initial_pose(pose);
}

std::vector<Sophus::SE3d> SlamSystem::get_camera_trajectory() const
{
  return impl->odometry->get_camera_trajectory();
}