#include "dense_mapping.h"
#include "map_struct.h"
#include "device_map_ops.h"
#include "message_logger.h"
#include <mutex>
#include <queue>
#include <vector>

struct ImageWithPose
{
    ImageWithPose() = default;
    ImageWithPose(const ImageWithPose &) = delete;
    ImageWithPose &operator=(const ImageWithPose &) = delete;

    ImageWithPose(const size_t id, const cv::Mat &image, const cv::Mat &depth, const Sophus::SE3d &pose)
    {
        image.copyTo(image_);
        depth.copyTo(depth_);
        pose_ = pose;
        id_ = id;
    }

    cv::Mat depth_;
    cv::Mat image_;
    Sophus::SE3d pose_;
    size_t id_;
};

using ImageWithPosePtr = std::shared_ptr<ImageWithPose>;

class DenseMapping::DenseMappingImpl
{
  public:
    DenseMappingImpl(const IntrinsicMatrix &base_intrinsic_matrix, const int &update_level);
    ~DenseMappingImpl();
    void insert_frame(ImageWithPosePtr image);
    void process_buffer(int nframes = 1);
    void update_observation() const;
    void update_map(const ImageWithPosePtr image);

    int update_level_;
    size_t latest_frame_id_;
    std::shared_ptr<MapStruct> host_map_;
    std::shared_ptr<MapStruct> device_map_;
    std::mutex frame_list_guard_;
    std::queue<ImageWithPosePtr> frame_buffer_;
    std::vector<ImageWithPosePtr> frames_;
    IntrinsicMatrixPyramid intrinsic_matrix;
    uint visible_block_count;
};

DenseMapping::DenseMappingImpl::DenseMappingImpl(const IntrinsicMatrix &base_intrinsic_matrix, const int &update_level)
    : device_map_(nullptr), latest_frame_id_(0), update_level_(update_level)
{
    intrinsic_matrix = base_intrinsic_matrix.build_pyramid();
    device_map_ = std::make_shared<MapStruct>(600000, 800000, 300000, 0.005f);
    device_map_->allocate_device_memory();
    device_map_->reset_map_struct();
    MessageLogger::log("Map created");
}

DenseMapping::DenseMappingImpl::~DenseMappingImpl()
{
    device_map_->release_device_memory();
}

void DenseMapping::DenseMappingImpl::update_observation() const
{
}

void DenseMapping::DenseMappingImpl::insert_frame(ImageWithPosePtr image)
{
    std::lock_guard<std::mutex> lock(frame_list_guard_);
    frame_buffer_.push(image);
}

void DenseMapping::DenseMappingImpl::update_map(const ImageWithPosePtr image)
{
    cv::cuda::GpuMat depth(image->depth_);
    cv::cuda::GpuMat rgb(image->image_);
    slam::map::update(depth, rgb, *device_map_, image->pose_, intrinsic_matrix[0], visible_block_count);
}

void DenseMapping::DenseMappingImpl::process_buffer(int nframes)
{
    for (int i = 0; i < nframes; ++i)
    {
        std::lock_guard<std::mutex> lock(frame_list_guard_);
        ImageWithPosePtr raw_image = frame_buffer_.front();

        update_map(raw_image);

        frames_.push_back(raw_image);
        frame_buffer_.pop();
        latest_frame_id_ = raw_image->id_;
    }
}

bool DenseMapping::has_update() const
{
    return impl->frame_buffer_.size() > 0;
}

void DenseMapping::update() const
{
    impl->process_buffer();
}

void DenseMapping::insert_frame(const RgbdFramePtr frame) const
{
    const RgbdImagePtr image = frame->get_image_pyramid();
    ImageWithPosePtr new_image(new ImageWithPose(frame->get_id(), image->get_intensity_map(0), image->get_depth_map(0), frame->get_pose()));
    impl->insert_frame(new_image);
}

void DenseMapping::update_frame_pose(const size_t &id, const Sophus::SE3d &update) const
{
    for (int i = 0; i < impl->frames_.size(); ++i)
    {
        ImageWithPosePtr ptr = impl->frames_[i];
        if (ptr->id_ == id)
            ptr->pose_ = update;
    }
}

void DenseMapping::update_frame_pose_batch(const std::vector<size_t> &id, const std::vector<Sophus::SE3d> &pose) const
{
    for (int i = 0; i < id.size(); ++i)
    {
        update_frame_pose(id[i], pose[i]);
    }
}

DenseMapping::DenseMapping(const IntrinsicMatrix &base_intrinsic_matrix, const int &update_level) : impl(new DenseMappingImpl(base_intrinsic_matrix, update_level))
{
}

void DenseMapping::update_observation() const
{
}

bool DenseMapping::need_visual_update() const
{
    return true;
}