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
    SlamLocalMappingImpl(DataSource *source);
    void visualisation_loop();

    bool shutdown = false;
    cv::Mat image;
    cv::Mat depth, depth_float;
    IntrinsicMatrixPyramidPtr intrinsics_pyr_;
    std::shared_ptr<SlamSystem> system_;
    std::shared_ptr<GlDisplay> display_;
    std::shared_ptr<DenseMapping> mapping_;
};

SlamLocalMapping::SlamLocalMappingImpl::SlamLocalMappingImpl(DataSource *source)
{
    IntrinsicMatrix base_intrinsic_matrix = IntrinsicMatrix(640, 480, 517.3f, 516.5, 318.6, 255.3);
    intrinsics_pyr_ = std::make_shared<IntrinsicMatrixPyramid>(base_intrinsic_matrix, 5);

    system_ = std::make_shared<SlamSystem>(intrinsics_pyr_);
    mapping_ = std::make_shared<DenseMapping>(intrinsics_pyr_);

    std::thread t_display(&SlamLocalMapping::SlamLocalMappingImpl::visualisation_loop, this);

    while (source && source->read_next_images(image, depth) && !shutdown)
    {
        if (image.empty() || depth.empty())
            continue;

        depth.convertTo(depth_float, CV_32FC1, source->get_depth_scale());
        system_->update(image, depth_float, source->get_current_id(), source->get_current_timestamp());

        RgbdImagePtr current_image = system_->get_current_image();
        mapping_->integrate_frame(current_image);
    }

    t_display.join();
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

SlamLocalMapping::SlamLocalMapping(DataSource *source) : impl(new SlamLocalMappingImpl(source))
{
}
