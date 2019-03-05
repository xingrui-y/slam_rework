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

class SlamWrapper::SlamWrapperImpl
{
  public:
    SlamWrapperImpl(DataSource *source);
    ~SlamWrapperImpl();
    void dense_mapping_loop();
    void visualisation_loop();

    bool shutdown = false;
    cv::Mat image, range;
    cv::Mat intensity, depth;
    std::shared_ptr<SlamSystem> system_;
    std::shared_ptr<GlDisplay> display_;
    std::shared_ptr<DenseMapping> mapping_;
};

SlamWrapper::SlamWrapperImpl::SlamWrapperImpl(DataSource *source)
{
    IntrinsicMatrix K(640, 480, 517.3f, 516.5, 318.6, 255.3, 5);
    system_ = std::make_shared<SlamSystem>(K);
    std::thread t_display(&SlamWrapper::SlamWrapperImpl::visualisation_loop, this);
    std::thread t_mapping(&SlamWrapper::SlamWrapperImpl::dense_mapping_loop, this);

    if (source->get_groundtruth().size() > 0)
    {
        const Sophus::SE3d initial_pose = source->get_groundtruth()[0];
        system_->set_initial_pose(initial_pose);
    }
    else
    {
        system_->set_initial_pose(Sophus::SE3d());
    }

    while (source->read_next_images(image, range) && !shutdown)
    {
        image.convertTo(intensity, CV_32FC1);
        cv::cvtColor(intensity, intensity, cv::COLOR_BGR2GRAY);
        range.convertTo(depth, CV_32FC1, 1 / 5000.f);

        system_->update(intensity, depth, source->get_current_id(), source->get_current_timestamp());

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
}

SlamWrapper::SlamWrapperImpl::~SlamWrapperImpl()
{
}

void SlamWrapper::SlamWrapperImpl::dense_mapping_loop()
{
    std::cout << "Mapping Thread Started!" << std::endl;
    mapping_ = std::make_shared<DenseMapping>();
    while (!shutdown)
    {
        if (mapping_->has_update())
            mapping_->update();
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}

void SlamWrapper::SlamWrapperImpl::visualisation_loop()
{
    std::cout << "GUI Thread Started!" << std::endl;
    display_ = std::make_shared<GlDisplay>();
    while (!display_->should_quit())
    {
        display_->draw_frame();
    }

    shutdown = true;
}

SlamWrapper::SlamWrapper(DataSource *source) : impl(new SlamWrapperImpl(source))
{
}