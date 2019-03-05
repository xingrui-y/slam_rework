#include "dense_mapping.h"
#include "map_struct.h"
#include <mutex>
#include <queue>
#include <vector>

using ImageWithPosePtr = std::shared_ptr<DenseMapping::ImageWithPose>;

DenseMapping::ImageWithPose::ImageWithPose(const unsigned long id, const cv::Mat &image, const cv::Mat &depth, const Sophus::SE3d &pose)
{
    image.copyTo(image_);
    depth.copyTo(depth_);
    pose_ = pose;
    id_ = id;
}

class DenseMapping::DenseMappingImpl
{
  public:
    DenseMappingImpl(const MapState &param);
    void insert_frame(ImageWithPosePtr image);
    void process_buffer(int n_consecutive_frame = 1);

    bool sub_sample_;
    int subsample_rate_;
    std::shared_ptr<MapStruct> host_map_;
    std::shared_ptr<MapStruct> device_map_;
    std::mutex frame_list_guard_;
    std::queue<ImageWithPosePtr> frame_buffer_;
    std::vector<ImageWithPosePtr> frames_;
};

DenseMapping::DenseMappingImpl::DenseMappingImpl(const MapState &state)
    : device_map_(new MapStruct()), host_map_(nullptr)
{
}

void DenseMapping::DenseMappingImpl::insert_frame(ImageWithPosePtr image)
{
    std::unique_lock<std::mutex> lock(frame_list_guard_);
    frame_buffer_.push(image);
}

void DenseMapping::DenseMappingImpl::process_buffer(int n_consecutive_frame)
{
    for (int i = 0; i < n_consecutive_frame; ++i)
    {
        std::unique_lock<std::mutex> lock(frame_list_guard_);
        ImageWithPosePtr raw_image = frame_buffer_.front();
        frames_.push_back(raw_image);
        frame_buffer_.pop();
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

void DenseMapping::update_frame_pose(const unsigned long &id, const Sophus::SE3d &update) const
{
    for (int i = 0; i < impl->frames_.size(); ++i)
    {
        ImageWithPosePtr ptr = impl->frames_[i];
        if (ptr->id_ == id)
            ptr->pose_ = update;
    }
}

void DenseMapping::update_frame_pose_batch(const std::vector<unsigned long> &id, const std::vector<Sophus::SE3d> &pose) const
{
    for (int i = 0; i < id.size(); ++i)
    {
        update_frame_pose(id[i], pose[i]);
    }
}

DenseMapping::DenseMapping()
{
    MapState state;
    impl = std::make_shared<DenseMappingImpl>(state);
}

DenseMapping::DenseMapping(const bool &sub_sample, const int &subsample_rate) : DenseMapping()
{
    impl->sub_sample_ = sub_sample;
    impl->subsample_rate_ = subsample_rate;
}
