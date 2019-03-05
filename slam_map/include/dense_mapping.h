#ifndef __DENSE_MAPPING__
#define __DENSE_MAPPING__

#include <memory>
#include "rgbd_image.h"

class DenseMapping
{
public:
  struct ImageWithPose
  {
    ImageWithPose() = default;
    ImageWithPose(const ImageWithPose &) = delete;
    ImageWithPose(const unsigned long id, const cv::Mat &image, const cv::Mat &depth, const Sophus::SE3d &pose);
    ImageWithPose &operator=(const ImageWithPose &) = delete;

    cv::Mat depth_;
    cv::Mat image_;
    Sophus::SE3d pose_;
    unsigned long id_;
  };

  DenseMapping();
  DenseMapping(const bool &sub_sample, const int &subsample_rate = 1);
  DenseMapping(const DenseMapping &) = delete;
  DenseMapping &operator=(const DenseMapping &) = delete;

  bool has_update() const;
  void update() const;
  void insert_frame(const RgbdFramePtr frame) const;
  void update_frame_pose(const unsigned long &id, const Sophus::SE3d &update) const;
  void update_frame_pose_batch(const std::vector<unsigned long> &id, const std::vector<Sophus::SE3d> &pose) const;

private:
  class DenseMappingImpl;
  std::shared_ptr<DenseMappingImpl> impl;
};

#endif