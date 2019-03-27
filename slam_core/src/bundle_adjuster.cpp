#include "bundle_adjuster.h"
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <ceres/rotation.h>

namespace Eigen
{
namespace internal
{

// Eigen's ostream operator is not compatible with ceres::Jet types.
// In particular, Eigen assumes that the scalar type (here Jet<T,N>) can be
// casted to an arithmetic type, which is not true for ceres::Jet.
// Unfortunatly, the ceres::Jet class does not define a conversion
// operator (http://en.cppreference.com/w/cpp/language/cast_operator).
//
// This workaround creates a template specilization for Eigen's cast_impl,
// when casting from a ceres::Jet type. It relies on Eigen's internal API and
// might break with future versions of Eigen.
template <class T, int N, typename NewType>
struct cast_impl<ceres::Jet<T, N>, NewType>
{
  EIGEN_DEVICE_FUNC
  static inline NewType run(ceres::Jet<T, N> const &x)
  {
    return static_cast<NewType>(x.a);
  }
};

} // namespace internal
} // namespace Eigen

class LocalParameterizationSE3 : public ceres::LocalParameterization
{
public:
  virtual ~LocalParameterizationSE3() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const *T_raw, double const *delta_raw, double *T_plus_delta_raw) const
  {
    Eigen::Map<Sophus::SE3d const> const T(T_raw);
    Eigen::Map<Sophus::Vector6d const> const delta(delta_raw);
    Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * Sophus::SE3d::exp(delta);
    return true;
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const *T_raw, double *jacobian_raw) const
  {
    Eigen::Map<Sophus::SE3d const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  virtual int GlobalSize() const
  {
    return Sophus::SE3d::num_parameters;
  }

  virtual int LocalSize() const
  {
    return Sophus::SE3d::DoF;
  }
};

struct SophusCostFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SophusCostFunctor(double observed_x, double observed_y, double observed_z) : observed_x_(observed_x), observed_y_(observed_y), observed_z_(observed_z) {}

  template <typename T>
  bool operator()(const T *const K, const T *const Rt_data, const T *const point_data, T *residual) const
  {
    const T &fx = K[0];
    const T &fy = K[1];
    const T &cx = K[2];
    const T &cy = K[3];

    Eigen::Map<Sophus::SE3<T> const> const Rt(Rt_data);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const point(point_data);
    // Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residual_data);
    Eigen::Matrix<T, 3, 1> point_c = Rt * point;

    T project_x = fx * point_c(0) / point_c(2) + cx;
    T project_y = fy * point_c(1) / point_c(2) + cy;

    residual[0] = project_x - observed_x_;
    residual[1] = project_y - observed_y_;
    residual[2] = point_c(2) - observed_z_;

    return true;
  }

  static ceres::CostFunction *create(double observed_x, double observed_y, double observed_z)
  {
    return new ceres::AutoDiffCostFunction<SophusCostFunctor, 3, 4, Sophus::SE3d::num_parameters, 3>(new SophusCostFunctor(observed_x, observed_y, observed_z));
  }

  double observed_x_, observed_y_, observed_z_;
};

class BundleAdjuster::BundleAdjusterImpl
{
public:
  BundleAdjusterImpl();
  void run_bundle_adjustment(const IntrinsicMatrix K);
  void set_up_bundler(std::vector<KeyPointStructPtr> keypoint_structs);

  std::vector<KeyPointStructPtr> keypoint_structs_;
};

BundleAdjuster::BundleAdjusterImpl::BundleAdjusterImpl()
{
}

BundleAdjuster::BundleAdjuster() : impl(new BundleAdjusterImpl())
{
}

void BundleAdjuster::BundleAdjusterImpl::set_up_bundler(std::vector<KeyPointStructPtr> keypoint_structs)
{
  keypoint_structs_ = keypoint_structs;
}

void BundleAdjuster::BundleAdjusterImpl::run_bundle_adjustment(const IntrinsicMatrix K)
{
  if (keypoint_structs_.size() == 1)
    return;

  std::vector<Sophus::SE3d> inv_camera_poses;
  std::vector<size_t> keyframe_ids;
  size_t num_residual_blocks = 0;
  auto *robust_loss = new ceres::CauchyLoss(1.0);
  LocalParameterizationSE3 *se3_parametrization = new LocalParameterizationSE3();

  for (auto key_struct : keypoint_structs_)
  {
    auto keyframe = key_struct->get_reference_frame();
    inv_camera_poses.emplace_back(keyframe->get_pose().inverse());
    keyframe_ids.emplace_back(keyframe->get_id());
  }

  ceres::Problem problem;
  double camera[4] = {K.fx, K.fy, K.cx, K.cy};

  for (int i = 0; i < inv_camera_poses.size(); ++i)
  {
    const size_t keyframe_id = keyframe_ids[i];
    problem.AddParameterBlock(inv_camera_poses[i].data(), Sophus::SE3d::num_parameters, se3_parametrization);

    if (keyframe_id == 0)
    {
      std::cout << "frame_id 0 called " << std::endl;
      problem.SetParameterBlockConstant(inv_camera_poses[i].data());

      problem.AddParameterBlock(camera, 4);
      problem.SetParameterBlockConstant(camera);
    }

    auto keypoints = keypoint_structs_[i]->get_key_points();

    for (int idx = 0; idx < keypoints.size(); ++idx)
    {
      auto &pt = keypoints[idx];
      if (pt.pt3d_ && pt.pt3d_->observations_.size() > 1 && pt.pt3d_->observations_.count(keyframe_id) != 0)
      {

        problem.AddResidualBlock(SophusCostFunctor::create(pt.kp_.pt.x, pt.kp_.pt.y, pt.z_), robust_loss, camera, inv_camera_poses[i].data(), pt.pt3d_->pos_.data());
        // problem.SetParameterBlockConstant(pt.pt3d_->pos_.data());
        num_residual_blocks++;
      }
    }
  }

  if (num_residual_blocks == 0)
    return;

  ceres::Solver::Options options;
  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  // options.update_state_every_iteration = true;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::DENSE_SCHUR;

  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;
  // std::cout << camera[0] << std::endl;

  for (int i = 0; i < inv_camera_poses.size(); ++i)
  {
    auto keyframe = keypoint_structs_[i]->get_reference_frame();
    keyframe->set_pose(inv_camera_poses[i].inverse());
    // std::cout << inv_camera_poses[i].matrix() << std::endl;
  }
}

void BundleAdjuster::set_up_bundler(std::vector<KeyPointStructPtr> keypoint_structs)
{
  impl->set_up_bundler(keypoint_structs);
}

void BundleAdjuster::run_bundle_adjustment(const IntrinsicMatrix K)
{
  impl->run_bundle_adjustment(K);
}

void BundleAdjuster::run_unit_test()
{
  google::InitGoogleLogging("dafs");

  Sophus::SE3d T_raw = Sophus::SE3d(Sophus::SO3d::exp(Sophus::Vector3d(0, 0.05, 0.05)), Sophus::Vector3d(0, 0.05, 0));

  std::cout << "Before :\n " << T_raw.matrix3x4() << std::endl;
  double a[4] = {528, 528, 320, 240};

  std::vector<Eigen::Vector3d> b;
  double x[10] = {23, 24, 55, 112, 66, 35, 64, 13, 90, 455};
  double y[10] = {23, 24, 55, 112, 66, 35, 64, 13, 90, 234};
  double z[10] = {3, 4, 2, 6, 7, 2, 6, 8, 3, 3};

  for (int i = 0; i < 10; ++i)
  {
    Eigen::Vector3d pt(z[i] * (x[i] - a[2]) / a[0],
                       z[i] * (y[i] - a[3]) / a[1],
                       z[i]);
    b.push_back(pt);
  }

  ceres::Problem problem;
  problem.AddParameterBlock(T_raw.data(), Sophus::SE3d::num_parameters, new LocalParameterizationSE3);
  for (int i = 0; i < 10; ++i)
  {
    problem.AddResidualBlock(SophusCostFunctor::create(x[i], y[i], z[i]), NULL, a, T_raw.data(), b[i].data());
    problem.SetParameterBlockConstant(b[i].data());
  }

  ceres::Solver::Options options;
  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.linear_solver_type = ceres::SPARSE_SCHUR;

  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;
  std::cout << "After :\n " << std::endl;
  std::cout << T_raw.matrix3x4() << std::endl;
}