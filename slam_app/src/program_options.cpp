#include "program_options.h"

ProgramOptions::ProgramOptions()
    : use_dataset(false), use_custom_config(false)
{
}

bool ProgramOptions::parse(int argc, char **argv)
{
    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()("help,h", "Print help messages")("data,d", po::value<std::string>(&dataset_path), "Using dataset images instead of the live feed")("config,c", po::value<std::string>(&config_path), "Path to the configuration file.");
    po::variables_map vm;

    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return false;
        }

        if (vm.count("data"))
            use_dataset = true;

        if (vm.count("config"))
            use_custom_config = true;

        return true;
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Unknown error!" << std::endl;
        return false;
    }
}