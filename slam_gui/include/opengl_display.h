#ifndef __GL_DISPLAY__
#define __GL_DISPLAY__

#include <memory>
#include <sophus/se3.hpp>

class GlDisplay
{
public:
  GlDisplay();
  GlDisplay(int width, int height);
  ~GlDisplay();

  void set_current_pose(const Sophus::SE3d &pose) const;
  void set_ground_truth_trajectory(const std::vector<Sophus::SE3d> &gt);
  void set_camera_trajectory(const std::vector<Sophus::SE3d> camera);
  void set_keyframe_poses(const std::vector<Sophus::SE3d> keyframes);
  void set_current_key_points(const std::vector<Eigen::Vector3f> keypoints);
  void upload_mesh(const void *vertices, const void *normal, const void *texture, const size_t &size);
  bool should_quit() const;
  void draw_frame() const;

private:
  class GlDisplayImpl;
  std::shared_ptr<GlDisplayImpl> impl;
};

#endif