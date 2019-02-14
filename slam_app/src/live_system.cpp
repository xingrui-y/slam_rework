#include "slam_system.h"
#include "pangolin_gui.h"
#include "openni_camera.h"
#include <thread>
#include <mutex>
#include <sophus/se3.hpp>

bool shutdown = false;
std::vector<Sophus::SE3d> ground_truth_trajectory;

void visualisation_thread(SlamSystem *sys)
{
    PangolinGUI gui(1920, 1080);

    while (!gui.should_quit())
    {
        gui.draw_frame();
    }

    shutdown = true;
}

int main(int argc, char **argv)
{
    SlamSystem sys;
    OpenNICamera camera(640, 480, 30);
    camera.start_video_streaming();

    std::thread vis(&visualisation_thread, &sys);

    cv::Mat image, depth, intensity;
    while( !shutdown)
    if (camera.capture(image, depth) && !shutdown)
    {
        cv::cvtColor(image, intensity, cv::COLOR_RGB2GRAY);

        sys.set_new_images(intensity, depth);
    }

    vis.join();
}