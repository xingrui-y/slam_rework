#include <cassert>
#include "bundle_adjuster.h"

int main(int argc, char **argv)
{
    std::shared_ptr<BundleAdjuster> bundler(new BundleAdjuster());
    bundler->run_unit_test();
}