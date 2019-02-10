#include <ceres/ceres.h>
#include <glog/logging.h>
#include <sophus/se3.hpp>
#include "io_wrapper.h"
#include "dense_odometry.h"

struct CostFunctor
{
    template <typename T>
    bool operator()(const T *const camera, const T *const reference, const T *const src, T *residual) const
    {
        return true;
    }
};

int main(int argc, char **argv)
{
    TUMDatasetWrapper data("/home/xyang/Downloads/rgbd_dataset_freiburg1_rpy");
    data.load_association_file("association.txt");
    data.load_ground_truth("groundtruth.txt");

    DenseOdometryPtr odom;

    cv::Mat intensity, depth;
    while (data.read_next_images(intensity, depth))
    {
        cv::cvtColor(intensity, intensity, cv::COLOR_BGR2GRAY);
        intensity.convertTo(intensity, CV_32FC1);

        Eigen::Matrix<double, 6, 1> result = Sophus::SE3d().log();
    }
}