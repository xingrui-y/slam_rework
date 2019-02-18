#include "stop_watch.h"
#include "openni_camera.h"
#include "slam_gui_wrapper.h"
#include "tum_dataset_wrapper.h"
#include "program_options.h"

bool shutdown = false;

int main(int argc, char **argv)
{
    ProgramOptions po;
    if (po.parse(argc, argv))
    {
        SlamSystem sys;
        cv::Mat image, depth;

        if (po.use_dataset)
        {
            TUMDatasetWrapper tum(po.dataset_path);
            if (!tum.load_association_file("association.txt"))
                exit(0);

            tum.load_ground_truth("groundtruth.txt");
            while (tum.read_next_images(image, depth) && !shutdown)
            {
                cv::Mat intensity;
                cv::Mat range;
                cv::cvtColor(image, intensity, cv::COLOR_BGR2GRAY);
                depth.convertTo(range, CV_32FC1, 1 / 5000.f);
                intensity.convertTo(intensity, CV_32FC1);
                sys.set_new_images(intensity, range);
            }
        }
        else
        {
            OpenNICamera camera(640, 480, 30);
            camera.start_video_streaming();

            while (!shutdown)
            {
                if (camera.capture(image, depth))
                {
                    cv::Mat intensity;
                    cv::Mat range;
                    cv::cvtColor(image, intensity, cv::COLOR_RGB2GRAY);
                    depth.convertTo(range, CV_32FC1, 1 / 1000.f);

                    intensity.convertTo(intensity, CV_32FC1);
                    sys.set_new_images(intensity, range);
                }
            }
        }
    }
}