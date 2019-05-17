#include "slam_wrapper.h"
#include "stop_watch.h"
#include "slam_system.h"
#include "opengl_display.h"
#include "rgbd_image.h"
#include "data_source.h"
#include "glog/logging.h"
#include "dense_mapping.h"
#include "message_queue.h"
#include <opencv2/opencv.hpp>
#include <thread>

class SlamWrapper::SlamWrapperImpl
{
public:
    SlamWrapperImpl();
    void visualisation_loop();
    void constraint_searching_loop();
    void update_display() const;
    void run_slam_system();

    bool shutdown_;
    bool optimizer_stop_;

    cv::Mat image;
    cv::Mat depth, depth_float;

    DataSource *data_source_;
    Sophus::SE3d initial_pose_;
    IntrinsicMatrix base_intrinsic_matrix_;
    IntrinsicMatrixPyramidPtr intrinsics_pyr_;

    std::shared_ptr<SlamSystem> system_;
    std::shared_ptr<GlDisplay> display_;
    std::shared_ptr<MessageQueue> msg_queue_;

    std::vector<Sophus::SE3d> ground_truth_list_;
    std::vector<double> image_timestamp_list_;
};

SlamWrapper::SlamWrapperImpl::SlamWrapperImpl()
    : shutdown_(false), optimizer_stop_(false),
      msg_queue_(NULL), display_(NULL), system_(NULL)
{
}

void SlamWrapper::SlamWrapperImpl::run_slam_system()
{
    CHECK_NOTNULL(data_source_);
    CHECK_NOTNULL(intrinsics_pyr_.get());

    msg_queue_ = std::make_shared<MessageQueue>();
    system_ = std::make_shared<SlamSystem>(intrinsics_pyr_);

    CHECK_NOTNULL(system_.get());
    CHECK_NOTNULL(msg_queue_.get());

    system_->set_message_queue(msg_queue_);

    system_->set_initial_pose(data_source_->get_initial_pose());

    std::thread t_display(&SlamWrapper::SlamWrapperImpl::visualisation_loop, this);
    std::thread t_optimize(&SlamWrapper::SlamWrapperImpl::constraint_searching_loop, this);

    bool ground_truth_set = false;

    while (data_source_->read_next_images(image, depth) && !shutdown_)
    {
        if (image.empty() || depth.empty())
            continue;

        auto id = data_source_->get_current_id();
        auto time_stamp = data_source_->get_current_timestamp();

        depth.convertTo(depth_float, CV_32FC1, data_source_->get_depth_scale());

        system_->update(image, depth_float, id, time_stamp);

        update_display();

        if (display_ && !ground_truth_set)
        {
            ground_truth_set = true;
            display_->set_ground_truth_trajectory(data_source_->get_groundtruth());
        }
    }

    if (!shutdown_)
    {
        optimizer_stop_ = true;
        t_optimize.join();

        system_->finish_pending_works();
        update_display();
    }

    t_display.join();
}

void SlamWrapper::SlamWrapperImpl::update_display() const
{
    if (display_)
    {
        display_->set_camera_trajectory(system_->get_camera_trajectory());
        display_->set_current_pose(system_->get_current_pose());
        display_->set_keyframe_poses(system_->get_keyframe_poses());
        display_->set_all_key_points(system_->get_all_key_points_with_rgb());
        // display_->set_current_key_points(system_->get_current_key_points());
    }
}

void SlamWrapper::SlamWrapperImpl::visualisation_loop()
{
    display_ = std::make_shared<GlDisplay>();
    display_->set_message_queue(msg_queue_);

    while (!display_->should_quit())
    {
        display_->draw_frame();
    }

    shutdown_ = true;
}

void SlamWrapper::SlamWrapperImpl::constraint_searching_loop()
{
    int counter = 0;

    while (!shutdown_ && !optimizer_stop_)
    {
        if (system_)
        {
            if (system_->search_constraint())
                counter++;

            // if (counter > 2)
            // {
            //     system_->run_bundle_adjustment();
            //     counter = 0;
            // }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

SlamWrapper::SlamWrapper() : impl(new SlamWrapperImpl())
{
}

void SlamWrapper::set_data_source(DataSource *source)
{
    CHECK_NOTNULL(source);
    impl->data_source_ = source;
}

void SlamWrapper::set_configuration(SimpleConfigStruct config)
{
    impl->base_intrinsic_matrix_ = IntrinsicMatrix(config.width, config.height, config.fx, config.fy, config.cx, config.cy);
    impl->intrinsics_pyr_ = std::make_shared<IntrinsicMatrixPyramid>(impl->base_intrinsic_matrix_, config.pyramid_level);
}

void SlamWrapper::run_slam_system()
{
    impl->run_slam_system();
}