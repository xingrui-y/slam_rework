#include "dense_mapping.h"
#include "map_struct.h"
#include "device_map_ops.h"

class DenseMapping::DenseMappingImpl
{
public:
  DenseMappingImpl(const IntrinsicMatrixPyramidPtr &intrinsics_pyr);
  ~DenseMappingImpl();
  void update(RgbdImagePtr current_image);
  void raycast(RgbdImagePtr current_image);

  IntrinsicMatrix intrinsic_matrix_;
  std::shared_ptr<MapStruct> map_struct_;
  const int integration_level_ = 0;
  cv::cuda::GpuMat cast_vmap_;
  cv::cuda::GpuMat cast_nmap_;
  cv::cuda::GpuMat zrange_x_;
  cv::cuda::GpuMat zrange_y_;
};

DenseMapping::DenseMappingImpl::DenseMappingImpl(const IntrinsicMatrixPyramidPtr &intrinsics_pyr)
{
  map_struct_ = std::make_shared<MapStruct>(400000, 600000, 200000, 0.005f);
  map_struct_->allocate_device_memory();
  map_struct_->reset_map_struct();

  intrinsic_matrix_ = *(*intrinsics_pyr)[integration_level_];
  cast_vmap_.create(intrinsic_matrix_.height, intrinsic_matrix_.width, CV_32FC4);
  cast_nmap_.create(intrinsic_matrix_.height, intrinsic_matrix_.width, CV_32FC4);
  zrange_x_.create(intrinsic_matrix_.height / 8, intrinsic_matrix_.width / 8, CV_32FC1);
  zrange_y_.create(intrinsic_matrix_.height / 8, intrinsic_matrix_.width / 8, CV_32FC1);
}

DenseMapping::DenseMappingImpl::~DenseMappingImpl()
{
  if (map_struct_)
    map_struct_->release_device_memory();
}

void DenseMapping::DenseMappingImpl::update(RgbdImagePtr current_image)
{
  RgbdFramePtr current_frame = current_image->get_reference_frame();
  if (current_frame == nullptr)
    return;

  cv::cuda::GpuMat depth = current_image->get_depth(integration_level_);
  cv::cuda::GpuMat image = current_image->get_image(integration_level_);
  Sophus::SE3d pose = current_frame->get_pose();
  uint visible_block_count = 0;

  slam::map::update(*map_struct_, depth, image, pose, intrinsic_matrix_, visible_block_count);
}

void DenseMapping::DenseMappingImpl::raycast(RgbdImagePtr current_image)
{
  RgbdFramePtr current_frame = current_image->get_reference_frame();
  if (current_frame == nullptr)
    return;

  Sophus::SE3d pose = current_frame->get_pose();

  slam::map::create_rendering_blocks(*map_struct_, zrange_x_, zrange_y_, pose, intrinsic_matrix_);

  uint rendering_block_count = 0;
  map_struct_->get_rendering_block_count(rendering_block_count);
  if (rendering_block_count != 0)
  {
    slam::map::raycast(*map_struct_, cast_vmap_, cast_nmap_, zrange_x_, zrange_y_, pose, intrinsic_matrix_);
  }
}

DenseMapping::DenseMapping(const IntrinsicMatrixPyramidPtr &intrinsics_pyr) : impl(new DenseMappingImpl(intrinsics_pyr))
{
}

void DenseMapping::integrate_frame(RgbdImagePtr current_image)
{
  impl->update(current_image);
}