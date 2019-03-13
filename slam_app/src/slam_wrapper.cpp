#include "slam_wrapper.h"
#include "stop_watch.h"
#include "slam_system.h"
#include "openni_camera.h"
#include "opengl_display.h"
#include "rgbd_image.h"
#include "program_options.h"
#include "dense_mapping.h"
#include <opencv2/opencv.hpp>
#include <exception>
#include <thread>
#include <mutex>

class SlamWrapper::SlamWrapperImpl
{
  public:
    SlamWrapperImpl(DataSource *source);
    ~SlamWrapperImpl();
    void dense_mapping_loop();
    void visualisation_loop();
    void constraint_searching_loop();
    void pose_graph_optimisation_loop();

    bool shutdown = false;
    cv::Mat image;
    cv::Mat image_float;
    cv::Mat intensity, depth;
    cv::Mat depth_float;
    IntrinsicMatrix intrinsic_matrix;
    std::shared_ptr<SlamSystem> system_;
    std::shared_ptr<GlDisplay> display_;
    std::shared_ptr<DenseMapping> mapping_;

    void log_stdout(const std::string msg);
    std::mutex mutex_stdout_;
};

SlamWrapper::SlamWrapperImpl::SlamWrapperImpl(DataSource *source)
{
    intrinsic_matrix = IntrinsicMatrix(640, 480, 517.3f, 516.5, 318.6, 255.3, 5);
    system_ = std::make_shared<SlamSystem>(intrinsic_matrix);
    std::thread t_display(&SlamWrapper::SlamWrapperImpl::visualisation_loop, this);
    std::thread t_mapping(&SlamWrapper::SlamWrapperImpl::dense_mapping_loop, this);
    std::thread t_opt(&SlamWrapper::SlamWrapperImpl::pose_graph_optimisation_loop, this);
    std::thread t_constraint(&SlamWrapper::SlamWrapperImpl::constraint_searching_loop, this);

    if (source->get_groundtruth().size() > 0)
    {
        const Sophus::SE3d initial_pose = source->get_groundtruth()[0];
        system_->set_initial_pose(initial_pose);
    }
    else
    {
        system_->set_initial_pose(Sophus::SE3d());
    }

    while (source->read_next_images(image, depth) && !shutdown)
    {
        image.convertTo(image_float, CV_32FC3);
        cv::cvtColor(image_float, intensity, cv::COLOR_BGR2GRAY);
        depth.convertTo(depth_float, CV_32FC1, source->get_depth_scale());

        system_->update(image, intensity, depth_float, source->get_current_id(), source->get_current_timestamp());

        if (display_)
        {
            display_->set_current_pose(system_->get_current_pose());
            display_->set_ground_truth_trajectory(source->get_groundtruth());
            display_->set_camera_trajectory(system_->get_camera_trajectory());
        }

        if (mapping_)
        {
            mapping_->insert_frame(system_->get_current_frame());
        }
    }

    t_display.join();
    t_mapping.join();
    t_constraint.join();
    t_opt.join();
}

SlamWrapper::SlamWrapperImpl::~SlamWrapperImpl()
{
}

void SlamWrapper::SlamWrapperImpl::log_stdout(const std::string msg)
{
    std::lock_guard<std::mutex> lock(mutex_stdout_);
    std::cout << msg << std::endl;
}

void SlamWrapper::SlamWrapperImpl::dense_mapping_loop()
{
    mapping_ = std::make_shared<DenseMapping>(intrinsic_matrix, 0);
    while (!shutdown)
    {
        if (mapping_->has_update())
            mapping_->update();
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(300));

        if (mapping_->need_visual_update())
        {
            mapping_->update_observation();
            //display_->upload_mesh();
        }
    }
}

void SlamWrapper::SlamWrapperImpl::visualisation_loop()
{
    display_ = std::make_shared<GlDisplay>();
    while (!display_->should_quit())
    {
        display_->draw_frame();
    }

    shutdown = true;
}

void SlamWrapper::SlamWrapperImpl::constraint_searching_loop()
{
    while (!shutdown)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}

void SlamWrapper::SlamWrapperImpl::pose_graph_optimisation_loop()
{
    while (!shutdown)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}

SlamWrapper::SlamWrapper(DataSource *source) : impl(new SlamWrapperImpl(source))
{
}