#include <memory>
#include "program_options.h"
#include "openni_camera.h"
#include "glog/logging.h"
#include "slam_local_mapping.h"
#include "tum_dataset_wrapper.h"
#include "simple_config_file_loader.h"

int main(int argc, char *argv[])
{
    ProgramOptions po;
    if (!po.parse(argc, argv))
        exit(-1);

    google::InitGoogleLogging(argv[0]);
    std::shared_ptr<SlamLocalMapping> slam;

    SimpleConfigFileLoader loader;
    SimpleConfigStruct config_struct;

    if (po.use_custom_config)
    {
        config_struct = loader.load_config_file(po.config_path);
    }
    else
    {
        config_struct.width = 640;
        config_struct.height = 480;
        config_struct.fx = 528.f;
        config_struct.fy = 528.f;
        config_struct.cx = 320.f;
        config_struct.cy = 240.f;
        config_struct.pyramid_level = 5;
    }

    if (po.use_dataset)
    {
        TUMDatasetWrapper tum(po.dataset_path);
        tum.load_association_file("association.txt");
        tum.load_ground_truth("groundtruth.txt");
        slam = std::make_shared<SlamLocalMapping>(&tum, config_struct);
    }
    else
    {
        OpenNICamera camera(640, 480, 30);
        slam = std::make_shared<SlamLocalMapping>(&camera, config_struct);
    }
}