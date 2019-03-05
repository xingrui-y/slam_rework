#include "slam_wrapper.h"
#include "program_options.h"
#include "openni_camera.h"
#include "tum_dataset_wrapper.h"
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    ProgramOptions po;
    if (po.parse(argc, argv))
    {
        if (po.use_dataset)
        {
            TUMDatasetWrapper tum(po.dataset_path);
            if (!tum.load_association_file("association.txt"))
                exit(0);
            tum.load_ground_truth("groundtruth.txt");

            SlamWrapper slam(&tum);
        }
        else
        {
            OpenNICamera camera(640, 480, 30);
            SlamWrapper slam(&camera);
        }
    }
}