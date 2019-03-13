#ifndef __DENSE_MAPPING__
#define __DENSE_MAPPING__

#include <memory>
#include "rgbd_image.h"
#include "intrinsic_matrix.h"

class DenseMapping
{
public:
  DenseMapping() = default;
  DenseMapping(const IntrinsicMatrix &base_intrinsic_matrix, const int &update_level);
  DenseMapping(const DenseMapping &) = delete;
  DenseMapping &operator=(const DenseMapping &) = delete;

  bool has_update() const;
  void update() const;
  void update_observation() const;
  bool need_visual_update() const;
  void insert_frame(const RgbdFramePtr frame) const;
  void update_frame_pose(const unsigned long &id, const Sophus::SE3d &update) const;
  void update_frame_pose_batch(const std::vector<unsigned long> &id, const std::vector<Sophus::SE3d> &pose) const;

private:
  class DenseMappingImpl;
  std::shared_ptr<DenseMappingImpl> impl;
};

#endif