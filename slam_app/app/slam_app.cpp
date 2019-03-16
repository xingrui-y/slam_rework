#include <memory>
#include "program_options.h"
#include "openni_camera.h"
#include "slam_local_mapping.h"
#include "tum_dataset_wrapper.h"

int main(int argc, char *argv[])
{
    ProgramOptions po;
    if (!po.parse(argc, argv))
        exit(-1);

    std::shared_ptr<SlamLocalMapping> slam;

    if (po.use_dataset)
    {
        TUMDatasetWrapper tum(po.dataset_path);
        tum.load_association_file("association.txt");
        tum.load_ground_truth("groundtruth.txt");
        slam = std::make_shared<SlamLocalMapping>(&tum);
    }
    else
    {
        OpenNICamera camera(640, 480, 30);
        slam = std::make_shared<SlamLocalMapping>(&camera);
    }
}