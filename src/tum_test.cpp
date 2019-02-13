#include "slam_system.h"
#include "pangolin_gui.h"
#include "tum_dataset_wrapper.h"
#include <thread>
#include <mutex>
#include <sophus/se3.hpp>

bool shutdown = false;
std::mutex data_access;
std::vector<Sophus::SE3d> ground_truth_trajectory;

void visualisation_thread()
{
    PangolinGUI gui(1920, 1080);

    while (!gui.should_quit())
    {
        std::unique_lock<std::mutex> lock(data_access);

        gui.draw_frame();

        lock.unlock();
    }

    shutdown = true;
}

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        printf("Usage: ./tum_app path-to-dataset\n");
        exit(0);
    }

    SlamSystem sys;
    TUMDatasetWrapper tum(argv[1]);
    tum.load_association_file("association.txt");
    tum.load_ground_truth("groundtruth.txt");

    std::thread vis(&visualisation_thread);

    cv::Mat image, depth, intensity;
    while(tum.read_next_images(image, depth) && !shutdown)
    {
        cv::cvtColor(image, intensity, cv::COLOR_BGR2GRAY);

        std::unique_lock<std::mutex> lock(data_access);
        lock.unlock();
    }

    vis.join();
}