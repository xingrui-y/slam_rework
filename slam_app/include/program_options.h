#ifndef __PROGRAM_OPTIONS__
#define __PROGRAM_OPTIONS__

#include <string>
#include <iostream>
#include <boost/program_options.hpp>

class ProgramOptions
{
  public:
    ProgramOptions();
    bool parse(int argc, char **argv);

    bool use_dataset;
    bool use_custom_config;
    std::string dataset_path;
    std::string config_path;
};

#endif