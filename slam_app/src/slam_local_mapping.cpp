#include "slam_local_mapping.h"
#include "stop_watch.h"
#include "slam_system.h"
#include "opengl_display.h"
#include "rgbd_image.h"
#include "dense_mapping.h"
#include <opencv2/opencv.hpp>
#include <thread>

class SlamLocalMapping::SlamLocalMappingImpl
{
public:
    SlamLocalMappingImpl(DataSource *source, SimpleConfigStruct config_struct);
    void visualisation_loop();
    void constraint_searching_loop();
    void update_display() const;

    bool shutdown;
    cv::Mat image;
    cv::Mat depth, depth_float;
    cv::cuda::GpuMat cast_vmap_, cast_nmap_;
    IntrinsicMatrixPyramidPtr intrinsics_pyr_;
    std::shared_ptr<SlamSystem> system_;
    std::shared_ptr<GlDisplay> display_;
    std::queue<int> message_queue_;
};

SlamLocalMapping::SlamLocalMappingImpl::SlamLocalMappingImpl(DataSource *source, SimpleConfigStruct config_struct) : shutdown(false)
{
    // IntrinsicMatrix base_intrinsic_matrix = IntrinsicMatrix(640, 480, 520.149963f, 516.175781f, 309.993548f, 227.090932f);
    // IntrinsicMatrix base_intrinsic_matrix = IntrinsicMatrix(640, 480, 525, 525, 320, 240);
    IntrinsicMatrix base_intrinsic_matrix = IntrinsicMatrix(config_struct.width, config_struct.height, config_struct.fx, config_struct.fy, config_struct.cx, config_struct.cy);
    intrinsics_pyr_ = std::make_shared<IntrinsicMatrixPyramid>(base_intrinsic_matrix, config_struct.pyramid_level);

    system_ = std::make_shared<SlamSystem>(intrinsics_pyr_);

    system_->set_initial_pose(source->get_starting_pose());
    std::thread t_display(&SlamLocalMapping::SlamLocalMappingImpl::visualisation_loop, this);
    // std::thread t_optimize(&SlamLocalMapping::SlamLocalMappingImpl::constraint_searching_loop, this);

    bool ground_truth_set = false;

    while (source && source->read_next_images(image, depth) && !shutdown)
    {
        if (image.empty() || depth.empty())
            continue;

        depth.convertTo(depth_float, CV_32FC1, source->get_depth_scale());
        system_->update(image, depth_float, source->get_current_id(), source->get_current_timestamp());

        update_display();

        if (display_ && !ground_truth_set)
        {
            ground_truth_set = false;
            display_->set_ground_truth_trajectory(source->get_groundtruth());
        }
    }

    // system_->finish_pending_works();
    // update_display();

    t_display.join();
    // t_optimize.join();
}

void SlamLocalMapping::SlamLocalMappingImpl::update_display() const
{
    if (display_)
    {
        display_->set_camera_trajectory(system_->get_camera_trajectory());
        display_->set_current_pose(system_->get_current_pose());
        display_->set_keyframe_poses(system_->get_keyframe_poses());
        display_->set_current_key_points(system_->get_current_key_points());
    }
}

void SlamLocalMapping::SlamLocalMappingImpl::visualisation_loop()
{
    display_ = std::make_shared<GlDisplay>();
    while (!display_->should_quit())
    {
        display_->draw_frame();
    }

    shutdown = true;
}

void SlamLocalMapping::SlamLocalMappingImpl::constraint_searching_loop()
{
    int counter = 0;

    while (!shutdown)
    {
        if (system_)
        {
            if (system_->search_constraint())
                counter++;

            if (counter > 10)
            {
                system_->run_bundle_adjustment();
                counter = 0;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

SlamLocalMapping::SlamLocalMapping(DataSource *source, SimpleConfigStruct config_struct) : impl(new SlamLocalMappingImpl(source, config_struct))
{
}
