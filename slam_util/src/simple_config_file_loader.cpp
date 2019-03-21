#include "simple_config_file_loader.h"

SimpleConfigStruct SimpleConfigFileLoader::load_config_file(std::string file_name)
{
    SimpleConfigStruct config;
    file_in_.open(file_name);
    std::string key;
    double val;
    while (file_in_ >> key >> val)
    {
        if (key == "width")
            config.width = val;
        if (key == "height")
            config.height = val;
        if (key == "fx")
            config.fx = val;
        if (key == "fy")
            config.fy = val;
        if (key == "cx")
            config.cx = val;
        if (key == "cy")
            config.cy = val;
        if (key == "pyramid_level")
            config.pyramid_level = val;
    }

    return config;
}
void SimpleConfigFileLoader::write_config_file(std::string file_name, SimpleConfigStruct config)
{
    file_out_.open(file_name);
    file_out_ << "width " << config.width << "\n"
              << "height " << config.height << "\n"
              << "fx " << config.fx << "\n"
              << "fy " << config.fy << "\n"
              << "cx " << config.cx << "\n"
              << "cy " << config.cy << "\n"
              << "pyramid_level " << config.pyramid_level << "\n"
              << "num_hash_entry " << config.num_hash_entry << "\n"
              << "num_voxel_block " << config.num_voxel_block << "\n"
              << "num_bucket " << config.num_bucket << "\n"
              << "zmin_update " << config.zmin_update << "\n"
              << "zmax_update " << config.zmax_update << "\n"
              << "zmin_raycast " << config.zmin_raycast << "\n"
              << "zmax_raycast " << config.zmax_raycast << "\n";
}